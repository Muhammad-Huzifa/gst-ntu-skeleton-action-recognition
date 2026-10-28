import hashlib
import json
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


API_ROOT = "https://api.github.com"


def _request(url, token, method="GET", body=None, content_type="application/vnd.github+json"):
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "gst-ntu-training-publisher",
    }
    data = body
    if isinstance(body, dict):
        data = json.dumps(body).encode("utf-8")
    if data is not None:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_body = response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"GitHub API request failed with HTTP {error.code} ({method} {url}).") from None
    except urllib.error.URLError as error:
        raise RuntimeError(f"GitHub API request failed ({method} {url}): {error.reason}") from None
    if not response_body:
        return {}
    return json.loads(response_body)


def publish_run(run_dir, repo, token, run_name, source_commit):
    if not token:
        raise ValueError("A GitHub token with repository Contents: read and write permission is required.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Repository must be in owner/name format.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError("Run name may contain only letters, numbers, dots, underscores, and hyphens.")

    run_dir = Path(run_dir)
    weights_path = run_dir / "final_weights.pth"
    results_path = run_dir / "final.json"
    history_path = run_dir / "history.json"
    if not weights_path.is_file() or not results_path.is_file():
        raise FileNotFoundError(f"{run_dir} is not a completed run (final_weights.pth and final.json are required).")

    results = json.loads(results_path.read_text(encoding="utf-8"))
    assets = [weights_path, results_path]
    if history_path.is_file():
        assets.append(history_path)

    created_at = datetime.now(timezone.utc)
    tag = f"ntu60-{run_name}-{created_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:6]}"
    manifest = {
        "run_name": run_name,
        "tag": tag,
        "created_at_utc": created_at.isoformat(),
        "source_commit": source_commit,
        "repo": repo,
        "files": {
            path.name: {
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in assets
        },
        "args": results.get("args", {}),
        "final": results.get("final", {}),
    }

    with tempfile.TemporaryDirectory(prefix="gst-release-") as temp_dir:
        archive_path = Path(temp_dir) / f"{tag}.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in assets:
                archive.write(path, arcname=path.name)
            archive.writestr("run_metadata.json", json.dumps(manifest, indent=2))

        release = _request(
            f"{API_ROOT}/repos/{repo}/releases",
            token,
            method="POST",
            body={
                "tag_name": tag,
                "target_commitish": source_commit,
                "name": f"NTU60 {run_name} run {created_at.strftime('%Y-%m-%d %H:%M UTC')}",
                "body": (
                    "Training artifacts for this run. The ZIP contains final inference weights, "
                    "metrics, optional training history, and a metadata manifest. "
                    "The dataset and resume checkpoint are not included."
                ),
                "draft": True,
                "prerelease": False,
            },
        )
        upload_url = release["upload_url"].split("{", 1)[0]
        asset_name = archive_path.name
        upload = _request(
            f"{upload_url}?{urllib.parse.urlencode({'name': asset_name})}",
            token,
            method="POST",
            body=archive_path.read_bytes(),
            content_type="application/zip",
        )
        published = _request(
            release["url"],
            token,
            method="PATCH",
            body={"draft": False},
        )

    return {
        "tag": tag,
        "html_url": published["html_url"],
        "asset_name": upload["name"],
        "asset_size": upload["size"],
    }
