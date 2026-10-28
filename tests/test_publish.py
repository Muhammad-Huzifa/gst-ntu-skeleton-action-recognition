import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from gst.publish import publish_run


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


class PublishRunTests(unittest.TestCase):
    def test_publishes_completed_run_as_release_bundle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "final_weights.pth").write_bytes(b"test weights")
            (run_dir / "final.json").write_text(json.dumps({"args": {"seed": 42}, "final": {"test_top1": 80}}))
            (run_dir / "history.json").write_text(json.dumps([{"epoch": 1}]))
            calls = []

            def fake_urlopen(request, timeout):
                calls.append(request)
                if request.method == "POST" and request.full_url.endswith("/releases"):
                    return FakeResponse({
                        "id": 123,
                        "url": "https://api.github.com/repos/example/project/releases/123",
                        "upload_url": "https://uploads.github.com/repos/example/project/releases/123/assets{?name,label}",
                    })
                if request.method == "POST" and "uploads.github.com" in request.full_url:
                    with zipfile.ZipFile(io.BytesIO(request.data)) as archive:
                        self.assertEqual(
                            set(archive.namelist()),
                            {"final_weights.pth", "final.json", "history.json", "run_metadata.json"},
                        )
                        metadata = json.loads(archive.read("run_metadata.json"))
                        self.assertEqual(metadata["source_commit"], "abc123")
                    return FakeResponse({"name": "run.zip", "size": len(request.data)})
                if request.method == "PATCH":
                    return FakeResponse({"html_url": "https://github.com/example/project/releases/tag/test"})
                self.fail(f"Unexpected request: {request.method} {request.full_url}")

            with patch("gst.publish.urllib.request.urlopen", side_effect=fake_urlopen):
                result = publish_run(run_dir, "example/project", "test-token", "gst", "abc123")

            self.assertTrue(result["tag"].startswith("ntu60-gst-"))
            self.assertEqual(result["html_url"], "https://github.com/example/project/releases/tag/test")
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(call.get_header("Authorization") == "Bearer test-token" for call in calls))
            self.assertNotIn("test-token", repr(result))

    def test_requires_complete_run_before_network_access(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("gst.publish.urllib.request.urlopen") as urlopen:
                with self.assertRaises(FileNotFoundError):
                    publish_run(temp_dir, "example/project", "test-token", "gst", "abc123")
                urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
