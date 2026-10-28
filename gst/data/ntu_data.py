import os

import numpy as np
import torch

CS_TRAIN_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
CV_TRAIN_CAMERAS = [2, 3]


class NTUData:
    def __init__(self, root, protocol='cs', device='cpu', val_frac=0.0, seed=0, max_train=None):
        meta = np.load(os.path.join(root, 'meta.npz'))
        X = np.load(os.path.join(root, 'X.npy'), mmap_mode='r')
        y, subject, camera = meta['y'], meta['subject'], meta['camera']
        if protocol == 'cs':
            is_train = np.isin(subject, CS_TRAIN_SUBJECTS)
        elif protocol == 'cv':
            is_train = np.isin(camera, CV_TRAIN_CAMERAS)
        else:
            raise ValueError(protocol)
        train_idx, test_idx = np.flatnonzero(is_train), np.flatnonzero(~is_train)
        val_idx = np.array([], dtype=np.int64)
        if val_frac > 0:
            rng = np.random.RandomState(seed)
            subs = np.unique(subject[train_idx])
            val_subs = rng.choice(subs, max(1, int(round(len(subs) * val_frac))), replace=False)
            in_val = np.isin(subject[train_idx], val_subs)
            val_idx, train_idx = train_idx[in_val], train_idx[~in_val]
        if max_train:
            train_idx = train_idx[:max_train]
        self.num_classes = int(y.max()) + 1
        self.device = device
        load = lambda idx: (torch.from_numpy(np.ascontiguousarray(X[idx])).to(device), torch.from_numpy(y[idx]).long().to(device))
        self.train = load(train_idx)
        self.val = load(val_idx) if len(val_idx) else None
        self.test = load(test_idx)
        print(f"[NTU-{protocol}] classes={self.num_classes} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
