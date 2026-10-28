# GST: Graph Skeleton Transformer

Graph-aware spatial encoder + part-pooled temporal transformer for skeleton action recognition (NTU RGB+D 60 here; graph size is a parameter, so the 65-joint sign skeleton also works via `gst.graph_utils.slr65_graph`).

## Layout
- `gst/graph_utils.py`: `GraphSpec`, NTU-25 / SLR-65 graphs, shortest-path distances, body parts
- `gst/graph_encoder.py`: spatial encoder (hop-biased attention + per-hop graph-conv branch + joint embedding)
- `gst/temporal_encoder.py`: part attention pooling, relative-position bias, conv-FFN, query pooling
- `gst/graph_model.py`: `GraphActionModel`; `gst/baseline_model.py`: your original AMTF architecture on the same inputs
- `gst/topology_fusion.py`: AMTF multi-topology block (`--mid amtf`; `--mid none` removes it)
- `gst/data/ntu_preprocess.py`: .skeleton -> packed `X.npy` (N,100,25,3,2) + `meta.npz`
- `gst/data/ntu_data.py`, `transforms.py`: GPU-resident data, 3D augmentation, feature building
- `train_ntu.py`: training / resume / final evaluation; `kaggle_ntu.ipynb`: run it on Kaggle

## Differences from the earlier 2D NTU notebook
1. 3D coordinates (x,y,z) instead of x,y only.
2. Two persons (padded + masked mean over persons) instead of one.
3. Centred on the spine and rotated to a body frame (spine / shoulder axes). The old normalisation used joints 3 and 7, which are the head and left hand in 0-indexed NTU, not the shoulders.
4. All data lives on the GPU; augmentation (temporal crop-resize, 3D rotation, scale) and feature computation run on the GPU, no DataLoader.
5. Final numbers are last-epoch weights, never selected on the test set. `--val-frac 0.1` holds out training subjects for tuning.

## Local smoke test (no dataset needed)
```bash
python tests/make_fake_ntu.py /tmp/fake/raw
python -m gst.data.ntu_preprocess --input /tmp/fake/raw --out /tmp/fake/packed --workers 2
python train_ntu.py --data /tmp/fake/packed --out /tmp/fake/out --cpu --hidden 32 --epochs 6 --bs 16 --T 32 --num-virtual 4
python check_graph_model.py
```

## Ablation switches
`--arch baseline|gst`, `--mid amtf|none`, `--share-streams`, `--no-part-pool`, `--pool query|mean`, `--gcn-hops 0` (attention-only encoder).
