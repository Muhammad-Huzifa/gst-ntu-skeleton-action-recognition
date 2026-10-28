import numpy as np
import torch
import torch.nn.functional as F

from gst.data.transforms import augment, eval_view, make_features, spatial_jitter


def amp_ctx(device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda')


def train_one_epoch(model, data, optimizer, criterion, scaler, scales, args, device):
    model.train()
    X, y = data
    perm = torch.randperm(X.size(0), device=X.device)
    n_batches = X.size(0) // args.bs
    loss_sum, correct, total = 0.0, 0, 0
    for i in range(n_batches):
        idx = perm[i * args.bs:(i + 1) * args.bs]
        xb, yb = X[idx].to(device, non_blocking=True), y[idx].to(device, non_blocking=True)
        with torch.no_grad():
            xb = augment(xb, args.T, args.rot, (args.scale_min, args.scale_max), args.crop_min)
            joint, bone, valid = make_features(xb, scales)
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx(device):
            logits = model(joint, bone, valid)
            loss = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item()
        correct += (logits[:, :args.num_classes].argmax(1) == yb).sum().item()
        total += yb.numel()
    return loss_sum / n_batches, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, data, scales, args, device, tta=1, batch=256):
    model.eval()
    X, y = data
    probs_all = []
    for i in range(0, X.size(0), batch):
        xb = eval_view(X[i:i + batch].to(device), args.T)
        probs = 0
        for t in range(tta):
            xv = xb if t == 0 else spatial_jitter(xb, 10.0, (0.95, 1.05))
            joint, bone, valid = make_features(xv, scales)
            with amp_ctx(device):
                logits = model(joint, bone, valid)
            probs = probs + F.softmax(logits[:, :args.num_classes].float(), dim=1)
        probs_all.append((probs / tta).cpu())
    probs = torch.cat(probs_all)
    y = y.cpu()
    top1 = 100.0 * (probs.argmax(1) == y).float().mean().item()
    top5 = 100.0 * (probs.topk(5, 1).indices == y[:, None]).any(1).float().mean().item()
    return top1, top5
