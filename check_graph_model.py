import torch
from torch.utils.flop_counter import FlopCounterMode

from gst.baseline_attention import PreNormTransformerBlock
from gst.graph_encoder import GraphSpatialEncoder
from gst.graph_model import GraphActionModel
from gst.graph_utils import ntu25_graph, slr65_graph


def params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def gmacs(m, *xs):
    with FlopCounterMode(display=False) as fc:
        m(*xs)
    return fc.get_total_flops() / 2e9


def main():
    torch.manual_seed(0)
    for name, g, T, jc, bc, nc in [("NTU-25", ntu25_graph(), 64, 6, 6, 60), ("SLR-65", slr65_graph(), 100, 6, 4, 100), ("SLR-65 wrist-linked", slr65_graph(link_wrists=True), 100, 6, 4, 100)]:
        V = g.num_node
        unreach = (g.spd == g.max_hop + 2).any(1).float().mean().item()
        far = (g.spd == g.max_hop + 1).float().mean().item()
        print(f"\n== {name}: V={V}, nodes with a disconnected partner: {unreach:.0%}, joint pairs beyond {g.max_hop} hops: {far:.0%}, buckets={g.num_buckets}")
        m = GraphActionModel(g, nc, 0, jc, bc).eval()
        xj, xb = torch.randn(2, jc, T, V), torch.randn(2, bc, T, V)
        y = m(xj, xb)
        print("logits", tuple(y.shape), "params", f"{params(m):,}", "GMACs", f"{gmacs(m, xj, xb):.3f}")
        m.train()
        m(xj, xb).sum().backward()
        dead = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
        print("params with zero grad after one backward:", dead if dead else "none")
    g = ntu25_graph()
    x = torch.randn(2, 64, 8, 25)
    perm = torch.randperm(25)
    enc = GraphSpatialEncoder(g, 64).eval()
    base = PreNormTransformerBlock(64, 4).eval()
    xs = x.permute(0, 2, 3, 1).reshape(16, 25, 64)
    d_base = (base(xs)[:, perm] - base(xs[:, perm])).abs().max().item()
    d_enc = (enc(x)[..., perm] - enc(x[..., perm])).abs().max().item()
    print(f"\njoint-permutation test (0 = blind to joint identity): old spatial block {d_base:.2e} | graph encoder {d_enc:.2e}")
    m5 = GraphActionModel(g, 60, 0, 6, 6).eval()
    print("5D NTU input (N,C,T,V,M=2) ->", tuple(m5(torch.randn(2, 6, 64, 25, 2), torch.randn(2, 6, 64, 25, 2)).shape))
    print("hop biases (head 0 / head 3), block 0:", enc.blocks[0].attn.hop_bias[0].tolist(), enc.blocks[0].attn.hop_bias[3].tolist())


if __name__ == "__main__":
    main()
