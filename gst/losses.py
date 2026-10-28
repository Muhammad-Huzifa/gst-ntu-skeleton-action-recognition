import torch
import torch.nn as nn
import torch.nn.functional as F


class VirtualClassRegularizedLoss(nn.Module):
    def __init__(self, num_real, num_virtual, alpha=0.1, smoothing=0.1):
        super().__init__()
        self.K, self.V, self.alpha, self.smoothing = num_real, num_virtual, alpha, smoothing
    def forward(self, logits, targets):
        logp = F.log_softmax(logits.float(), dim=-1)
        real = logp[:, :self.K]
        true_dist = torch.full_like(real, self.smoothing / (self.K - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        task = -(true_dist * real).sum(-1).mean()
        if self.V == 0 or self.alpha == 0:
            return task
        return (1 - self.alpha) * task + self.alpha * (-logp[:, self.K:].mean())


class WarmupCosine:
    def __init__(self, optimizer, base_lr, warmup, total, min_lr=1e-6):
        self.opt, self.base, self.warmup, self.total, self.min = optimizer, base_lr, warmup, total, min_lr
    def set_epoch(self, epoch):
        if epoch < self.warmup:
            lr = self.base * (epoch + 1) / self.warmup
        else:
            p = min((epoch - self.warmup) / max(1, self.total - self.warmup), 1.0)
            lr = self.min + (self.base - self.min) * (1 + torch.cos(torch.tensor(p * 3.141592653589793)).item()) / 2
        for g in self.opt.param_groups:
            g['lr'] = lr
        return lr
