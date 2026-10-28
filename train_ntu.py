import argparse
import json
import os
import random
import time

import numpy as np
import torch

from gst.baseline_model import BaselineAMTF
from gst.data.ntu_data import NTUData
from gst.data.transforms import estimate_scales
from gst.engine import evaluate, train_one_epoch
from gst.graph_model import GraphActionModel
from gst.graph_utils import ntu25_graph
from gst.losses import VirtualClassRegularizedLoss, WarmupCosine


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs/ntu")
    ap.add_argument("--protocol", default="cs", choices=["cs", "cv"])
    ap.add_argument("--arch", default="gst", choices=["gst", "baseline"])
    ap.add_argument("--mid", default="amtf", choices=["amtf", "none"])
    ap.add_argument("--share-streams", action="store_true")
    ap.add_argument("--no-part-pool", action="store_true")
    ap.add_argument("--pool", default="query", choices=["query", "mean"])
    ap.add_argument("--gcn-hops", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--sdepth", type=int, default=2)
    ap.add_argument("--tdepth", type=int, default=2)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.03)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--num-virtual", type=int, default=20)
    ap.add_argument("--vs-alpha", type=float, default=0.1)
    ap.add_argument("--rot", type=float, default=15.0)
    ap.add_argument("--scale-min", type=float, default=0.9)
    ap.add_argument("--scale-max", type=float, default=1.1)
    ap.add_argument("--crop-min", type=float, default=0.6)
    ap.add_argument("--tta", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--val-frac", type=float, default=0.0)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.out, exist_ok=True)
    data = NTUData(args.data, args.protocol, device, args.val_frac, args.seed, args.max_train or None)
    args.num_classes = data.num_classes
    graph = ntu25_graph()
    scales = [s.to(device) for s in estimate_scales(data.train[0][:2000].to(device), args.T)]
    kw = dict(graph=graph, num_classes=args.num_classes, num_virtual_classes=args.num_virtual, joint_in_ch=6, bone_in_ch=6, hidden_dim=args.hidden, num_heads=args.heads)
    if args.arch == "gst":
        model = GraphActionModel(spatial_depth=args.sdepth, temporal_depth=args.tdepth, gcn_hops=args.gcn_hops, mid=args.mid, share_streams=args.share_streams, part_pool=not args.no_part_pool, temporal_pool=args.pool, drop=args.drop, **kw)
    else:
        model = BaselineAMTF(transformer_depth=args.sdepth, **kw)
    model.to(device)
    print(f"arch={args.arch} params={sum(p.numel() for p in model.parameters() if p.requires_grad):,} device={device}")
    criterion = VirtualClassRegularizedLoss(args.num_classes, args.num_virtual, args.vs_alpha, args.smoothing)
    decay = [p for n, p in model.named_parameters() if p.ndim > 1 and not n.endswith(('hop_bias', 'edge_bias', 'rel_bias', 'node_embed', 'query'))]
    no_decay = [p for n, p in model.named_parameters() if not (p.ndim > 1 and not n.endswith(('hop_bias', 'edge_bias', 'rel_bias', 'node_embed', 'query')))]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    sched = WarmupCosine(optimizer, args.lr, args.warmup, args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    ckpt_path = os.path.join(args.out, "ckpt_last.pth")
    history, start = [], 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        history, start = ck["history"], ck["epoch"] + 1
        print(f"resumed from epoch {start}")
    eval_set = data.val if data.val is not None else data.test
    eval_name = "val" if data.val is not None else "test(monitor only)"
    for epoch in range(start, args.epochs):
        t0 = time.time()
        lr = sched.set_epoch(epoch)
        loss, acc = train_one_epoch(model, data.train, optimizer, criterion, scaler, scales, args, device)
        row = dict(epoch=epoch + 1, lr=lr, loss=loss, train_acc=acc, time=time.time() - t0)
        msg = f"{epoch + 1:03d}/{args.epochs} {row['time']:6.1f}s lr={lr:.6f} loss={loss:.4f} train_acc={acc:.2f}%"
        if (epoch + 1) % args.eval_every == 0 and epoch + 1 < args.epochs:
            top1, top5 = evaluate(model, eval_set, scales, args, device)
            row.update(eval_top1=top1, eval_top5=top5)
            msg += f" | {eval_name} top1={top1:.2f} top5={top5:.2f}"
        history.append(row)
        print(msg, flush=True)
        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            torch.save(dict(model=model.state_dict(), opt=optimizer.state_dict(), scaler=scaler.state_dict(), history=history, epoch=epoch), ckpt_path)
            with open(os.path.join(args.out, "history.json"), "w") as f:
                json.dump(history, f, indent=1)
    torch.save(model.state_dict(), os.path.join(args.out, "final_weights.pth"))
    final = {}
    if data.val is not None:
        final["val_top1"], final["val_top5"] = evaluate(model, data.val, scales, args, device, tta=args.tta)
    final["test_top1"], final["test_top5"] = evaluate(model, data.test, scales, args, device, tta=args.tta)
    final["test_top1_no_tta"], _ = evaluate(model, data.test, scales, args, device, tta=1)
    final["note"] = "last-epoch weights, no checkpoint selection on test"
    print(json.dumps(final, indent=1))
    with open(os.path.join(args.out, "final.json"), "w") as f:
        json.dump(dict(args=vars(args), final=final), f, indent=1)


if __name__ == "__main__":
    main()
