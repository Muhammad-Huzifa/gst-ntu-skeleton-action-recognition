import argparse
import os
import re
from multiprocessing import Pool

import numpy as np

NAME_RE = re.compile(r"S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})")
NUM_JOINT = 25
SPINE_BASE, SPINE_MID, SPINE_SHOULDER, L_SHOULDER, R_SHOULDER = 0, 1, 20, 4, 8


def parse_name(name):
    s, c, p, r, a = (int(g) for g in NAME_RE.search(name).groups())
    return dict(setup=s, camera=c, subject=p, rep=r, action=a)


def read_skeleton(path):
    with open(path) as f:
        lines = f.read().split('\n')
    i = 0
    n_frames = int(lines[i])
    i += 1
    frames = []
    for _ in range(n_frames):
        n_bodies = int(lines[i])
        i += 1
        bodies = {}
        for _ in range(n_bodies):
            bid = lines[i].split()[0]
            i += 1
            n_joints = int(lines[i])
            i += 1
            joints = np.empty((n_joints, 3), np.float32)
            for k in range(n_joints):
                v = lines[i].split()
                i += 1
                joints[k] = (float(v[0]), float(v[1]), float(v[2]))
            bodies[bid] = joints
        frames.append(bodies)
    return frames


def build_tracks(frames, max_body=2):
    T = len(frames)
    seq = np.zeros((max(T, 1), NUM_JOINT, 3, max_body), np.float32)
    mask = np.zeros((max(T, 1), max_body), bool)
    tracks = {}
    for t, bodies in enumerate(frames):
        for bid, j in bodies.items():
            if j.shape[0] == NUM_JOINT:
                tracks.setdefault(bid, {})[t] = j
    if not tracks:
        return seq, mask
    stats = {}
    for bid, d in tracks.items():
        arr = np.stack(list(d.values()))
        spread = float(arr.std(axis=0).mean()) if len(d) > 1 else 0.0
        stats[bid] = (len(d), len(d) * spread)
    longest = max(s[0] for s in stats.values())
    keep = [b for b in tracks if stats[b][0] >= 0.3 * longest]
    keep.sort(key=lambda b: -stats[b][1])
    for m, bid in enumerate(keep[:max_body]):
        for t, j in tracks[bid].items():
            seq[t, :, :, m] = j
            mask[t, m] = True
        present = np.flatnonzero(mask[:, m])
        last = None
        for t in range(present[0], present[-1] + 1):
            if mask[t, m]:
                last = seq[t, :, :, m]
            else:
                seq[t, :, :, m] = last
                mask[t, m] = True
    return seq, mask


def normalize(seq, mask):
    first = np.flatnonzero(mask[:, 0])
    if len(first) == 0:
        return seq
    f0 = seq[first[0], :, :, 0]
    origin = f0[SPINE_MID].reshape(1, 1, 3, 1)
    out = (seq - origin) * mask[:, None, None, :]
    up = f0[SPINE_SHOULDER] - f0[SPINE_BASE]
    norm_up = np.linalg.norm(up)
    if norm_up > 1e-3:
        ey = up / norm_up
        sh = f0[R_SHOULDER] - f0[L_SHOULDER]
        sh = sh - ey * np.dot(sh, ey)
        norm_sh = np.linalg.norm(sh)
        if norm_sh > 1e-3:
            ex = sh / norm_sh
            R = np.stack([ex, ey, np.cross(ex, ey)])
            out = np.einsum('ij,tvjm->tvim', R, out)
    return out.astype(np.float32)


def process_file(args):
    path, target_T = args
    frames = read_skeleton(path)
    seq, mask = build_tracks(frames)
    seq = normalize(seq, mask)
    idx = np.round(np.linspace(0, seq.shape[0] - 1, target_T)).astype(np.int64)
    return seq[idx]


def prepare(input_dir, out_dir, target_T=100, max_action=60, workers=4):
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f[:-9] for f in os.listdir(input_dir) if f.endswith('.skeleton'))
    names = [n for n in names if parse_name(n)['action'] <= max_action]
    info = [parse_name(n) for n in names]
    X = np.lib.format.open_memmap(os.path.join(out_dir, 'X.npy'), mode='w+', dtype=np.float32, shape=(len(names), target_T, NUM_JOINT, 3, 2))
    jobs = [(os.path.join(input_dir, n + '.skeleton'), target_T) for n in names]
    with Pool(workers) as pool:
        for i, x in enumerate(pool.imap(process_file, jobs, chunksize=16)):
            X[i] = x
            if i % 2000 == 0:
                print(f"{i}/{len(names)}", flush=True)
    X.flush()
    np.savez(os.path.join(out_dir, 'meta.npz'), names=np.array(names), y=np.array([d['action'] - 1 for d in info]), subject=np.array([d['subject'] for d in info]), camera=np.array([d['camera'] for d in info]), setup=np.array([d['setup'] for d in info]))
    print(f"done: {len(names)} samples -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--max-action", type=int, default=60)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    prepare(a.input, a.out, a.frames, a.max_action, a.workers)
