import math

import torch

from gst.graph_utils import NTU_PAIRS_1IDX

CHILD = [a - 1 for a, b in NTU_PAIRS_1IDX]
PARENT = [b - 1 for a, b in NTU_PAIRS_1IDX]


def resample_time(x, starts, lengths, T_out):
    B, T0 = x.shape[:2]
    u = torch.linspace(0, 1, T_out, device=x.device)
    pos = starts[:, None] + u[None] * (lengths[:, None] - 1)
    lo = pos.floor().long().clamp(0, T0 - 1)
    hi = (lo + 1).clamp(max=T0 - 1)
    w = (pos - lo.float())[:, :, None, None, None]
    b = torch.arange(B, device=x.device)[:, None]
    xl, xh = x[b, lo], x[b, hi]
    return xl + (xh - xl) * w


def random_rotation(B, max_deg, device):
    a = (torch.rand(B, 3, device=device) * 2 - 1) * max_deg * math.pi / 180
    c, s = a.cos(), a.sin()
    one, zero = torch.ones(B, device=device), torch.zeros(B, device=device)
    Rx = torch.stack([one, zero, zero, zero, c[:, 0], -s[:, 0], zero, s[:, 0], c[:, 0]], 1).view(B, 3, 3)
    Ry = torch.stack([c[:, 1], zero, s[:, 1], zero, one, zero, -s[:, 1], zero, c[:, 1]], 1).view(B, 3, 3)
    Rz = torch.stack([c[:, 2], -s[:, 2], zero, s[:, 2], c[:, 2], zero, zero, zero, one], 1).view(B, 3, 3)
    return Rz @ Ry @ Rx


def spatial_jitter(x, rot_deg, scale_range):
    B = x.size(0)
    R = random_rotation(B, rot_deg, x.device)
    s = torch.empty(B, device=x.device).uniform_(*scale_range)
    return torch.einsum('bij,btvjm->btvim', R * s[:, None, None], x)


def augment(x, T_out, rot_deg=15.0, scale_range=(0.9, 1.1), crop_min=0.6):
    B, T0 = x.shape[:2]
    lengths = (torch.empty(B, device=x.device).uniform_(crop_min, 1.0) * T0).clamp(min=2)
    starts = torch.rand(B, device=x.device) * (T0 - lengths)
    x = resample_time(x, starts, lengths, T_out)
    return spatial_jitter(x, rot_deg, scale_range)


def eval_view(x, T_out):
    B, T0 = x.shape[:2]
    return resample_time(x, torch.zeros(B, device=x.device), torch.full((B,), float(T0), device=x.device), T_out)


def make_features(x, scales=None):
    B, T, V, C, M = x.shape
    vel = torch.zeros_like(x)
    vel[:, 1:] = x[:, 1:] - x[:, :-1]
    bone = torch.zeros_like(x)
    bone[:, :, CHILD] = x[:, :, CHILD] - x[:, :, PARENT]
    bvel = torch.zeros_like(x)
    bvel[:, 1:] = bone[:, 1:] - bone[:, :-1]
    joint = torch.cat([x, vel], 3).permute(0, 3, 1, 2, 4).contiguous()
    bone = torch.cat([bone, bvel], 3).permute(0, 3, 1, 2, 4).contiguous()
    if scales is not None:
        joint = joint / scales[0].view(1, -1, 1, 1, 1)
        bone = bone / scales[1].view(1, -1, 1, 1, 1)
    return joint, bone, person_mask(x)


def person_mask(x):
    return x.abs().sum(dim=(1, 2, 3)) > 0


def estimate_scales(x, T_out):
    joint, bone, valid = make_features(eval_view(x, T_out))
    out = []
    for f in (joint, bone):
        f = f.permute(0, 4, 1, 2, 3)[valid]
        out.append(f.std(dim=(0, 2, 3)).clamp(min=1e-4))
    return out
