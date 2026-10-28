import torch
import torch.nn as nn

from gst.graph_utils import hop_adjacency


class GraphBiasedAttention(nn.Module):
    def __init__(self, dim, num_heads, spd, num_buckets, attn_drop=0.0):
        super().__init__()
        V = spd.size(0)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.register_buffer('spd', spd.clone(), persistent=False)
        slopes = torch.linspace(0.25, 1.5, num_heads).unsqueeze(1)
        hops = torch.arange(num_buckets).float().unsqueeze(0)
        self.hop_bias = nn.Parameter(-slopes * hops)
        self.edge_bias = nn.Parameter(torch.zeros(num_heads, V, V))
        self.attn_drop = nn.Dropout(attn_drop)
    def forward(self, x):
        B, V, C = x.shape
        q, k, v = self.qkv(x).reshape(B, V, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        bias = self.hop_bias[:, self.spd] + self.edge_bias
        attn = (q @ k.transpose(-2, -1)) * self.scale + bias.unsqueeze(0)
        attn = self.attn_drop(attn.softmax(dim=-1))
        return self.proj((attn @ v).transpose(1, 2).reshape(B, V, C))


class HopGraphConv(nn.Module):
    def __init__(self, dim, hop_adj):
        super().__init__()
        self.register_buffer('hop_adj', hop_adj.clone(), persistent=False)
        self.proj = nn.Linear(hop_adj.size(0) * dim, dim, bias=False)
    def forward(self, x):
        y = torch.einsum('kvw,bwc->bvkc', self.hop_adj, x)
        return self.proj(y.flatten(2))


class GraphEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, spd, num_buckets, hop_adj=None, mlp_ratio=2, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GraphBiasedAttention(dim, num_heads, spd, num_buckets)
        self.gcn = HopGraphConv(dim, hop_adj) if hop_adj is not None else None
        self.gate = nn.Parameter(torch.zeros(dim)) if hop_adj is not None else None
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        h = self.norm1(x)
        y = self.attn(h)
        if self.gcn is not None:
            y = y + torch.sigmoid(self.gate) * self.gcn(h)
        x = x + self.drop(y)
        return x + self.drop(self.mlp(self.norm2(x)))


class GraphSpatialEncoder(nn.Module):
    def __init__(self, graph, dim, num_heads=4, depth=2, gcn_hops=2, drop=0.0):
        super().__init__()
        self.node_embed = nn.Parameter(torch.zeros(1, graph.num_node, dim))
        nn.init.trunc_normal_(self.node_embed, std=0.02)
        hop_adj = hop_adjacency(graph.spd, gcn_hops) if gcn_hops > 0 else None
        self.blocks = nn.ModuleList([GraphEncoderBlock(dim, num_heads, graph.spd, graph.num_buckets, hop_adj, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        N, C, T, V = x.shape
        x = x.permute(0, 2, 3, 1).reshape(N * T, V, C) + self.node_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x.view(N, T, V, C).permute(0, 3, 1, 2).contiguous()
