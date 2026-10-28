import torch
import torch.nn as nn


class PartAttentionPool(nn.Module):
    def __init__(self, dim, parts, num_node):
        super().__init__()
        mask = torch.zeros(len(parts), num_node, dtype=torch.bool)
        for p, idx in enumerate(parts):
            mask[p, idx] = True
        self.register_buffer('mask', mask, persistent=False)
        self.score = nn.Linear(dim, len(parts))
    def forward(self, x):
        s = self.score(x.permute(0, 2, 3, 1)).permute(0, 1, 3, 2)
        w = s.masked_fill(~self.mask, float('-inf')).softmax(dim=-1)
        return torch.einsum('ntpv,nctv->ntpc', w, x)


class RelPosAttention(nn.Module):
    def __init__(self, dim, num_heads, max_rel=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_rel = max_rel
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        slopes = 2.0 ** -torch.arange(1, num_heads + 1).float()
        dist = torch.arange(-max_rel, max_rel + 1).abs().float()
        self.rel_bias = nn.Parameter(-slopes.unsqueeze(1) * dist.unsqueeze(0))
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        t = torch.arange(T, device=x.device)
        idx = (t[None, :] - t[:, None]).clamp(-self.max_rel, self.max_rel) + self.max_rel
        attn = (q @ k.transpose(-2, -1)) * self.scale + self.rel_bias[:, idx].unsqueeze(0)
        return self.proj((attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, T, C))


class ConvFFN(nn.Module):
    def __init__(self, dim, mlp_ratio=2, kernel=5):
        super().__init__()
        h = dim * mlp_ratio
        self.fc1 = nn.Linear(dim, h)
        self.dw = nn.Conv1d(h, h, kernel, padding=kernel // 2, groups=h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h, dim)
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = x + self.act(self.dw(x.transpose(1, 2))).transpose(1, 2)
        return self.fc2(x)


class TemporalBlock(nn.Module):
    def __init__(self, dim, num_heads, max_rel=16, kernel=5, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RelPosAttention(dim, num_heads, max_rel)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = ConvFFN(dim, 2, kernel)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        return x + self.drop(self.ffn(self.norm2(x)))


class GraphTemporalTransformer(nn.Module):
    def __init__(self, graph, in_dim, dim, num_heads=4, depth=2, max_rel=16, kernel=5, part_pool=True, pool='query', drop=0.0):
        super().__init__()
        self.part_pool = PartAttentionPool(in_dim, graph.parts, graph.num_node) if part_pool else None
        tokens = len(graph.parts) if part_pool else 1
        self.proj = nn.Sequential(nn.Linear(tokens * in_dim, dim), nn.LayerNorm(dim))
        self.blocks = nn.ModuleList([TemporalBlock(dim, num_heads, max_rel, kernel, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.pool = pool
        if pool == 'query':
            self.query = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.trunc_normal_(self.query, std=0.02)
            self.pool_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
    def forward(self, x):
        if self.part_pool is not None:
            x = self.part_pool(x).flatten(2)
        else:
            x = x.mean(-1).permute(0, 2, 1)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        if self.pool == 'query':
            return self.pool_attn(self.query.expand(x.size(0), -1, -1), x, x, need_weights=False)[0].squeeze(1)
        return x.mean(1)
