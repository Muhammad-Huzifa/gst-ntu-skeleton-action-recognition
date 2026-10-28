import os
import sys

import numpy as np


def write_skeleton(path, T, n_bodies, rng, cls, ghost=False):
    base = rng.normal(0, 0.05, (25, 3)).astype(np.float32)
    base[:, 1] += np.linspace(0, 1.6, 25)
    lines = [str(T)]
    for t in range(T):
        nb = n_bodies + (1 if ghost and t < 3 else 0)
        lines.append(str(nb))
        for b in range(nb):
            lines.append(f"{72057594037927936 + b} 0 1 2 1 2 0 0.0 0.0 2")
            lines.append("25")
            for j in range(25):
                p = base[j] + np.array([0.3 * b, 0, 0])
                p = p + 0.15 * (cls + 1) / 5 * np.sin(2 * np.pi * (cls + 1) * t / T + j) * np.array([1, 0.5, 0.2])
                lines.append(f"{p[0]:.4f} {p[1]:.4f} {p[2] + 2.5:.4f} 0 0 0 0 0 0 0 0 2")
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")


def main(out, n_per_class=24, n_classes=5):
    os.makedirs(out, exist_ok=True)
    rng = np.random.RandomState(0)
    subjects = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12]
    k = 0
    for cls in range(n_classes):
        for i in range(n_per_class):
            subj = subjects[i % len(subjects)]
            name = f"S001C{(i % 3) + 1:03d}P{subj:03d}R{(i // 10) + 1:03d}A{cls + 1:03d}"
            write_skeleton(os.path.join(out, name + ".skeleton"), int(rng.randint(40, 90)), 2 if cls == n_classes - 1 else 1, rng, cls, ghost=(i % 7 == 0))
            k += 1
    write_skeleton(os.path.join(out, "S001C001P001R001A001_missing.skeleton"[:20] + ".skeleton"), 5, 0, rng, 0)
    print("wrote", k + 1)


if __name__ == "__main__":
    main(sys.argv[1])
