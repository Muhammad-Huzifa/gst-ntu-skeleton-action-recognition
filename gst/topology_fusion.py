import torch
import torch.nn as nn

from gst.fusion import MeanTopologyFusion


class DepthwiseSeparableSpatiotemporalConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False), nn.BatchNorm2d(in_ch), nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.conv(x)


class AdaptiveMultiTopologyFusion(nn.Module):
    NUM_TOPOLOGIES = 4

    def __init__(self, in_ch, out_ch, A):
        super().__init__()
        self.num_topologies = self.NUM_TOPOLOGIES
        A_variants = []
        for i in range(self.num_topologies):
            A_init = A.clone()
            if i == 1:
                A_init = A_init + torch.eye(A.size(0)) * 0.5
            elif i == 2:
                A_init = torch.matrix_power(A_init, 2)
                A_init = (A_init > 0).float()
            elif i == 3:
                A_init = A_init * (torch.rand_like(A) * 0.3 + 0.85)
            A_variants.append(A_init)
        self.A = nn.Parameter(torch.stack(A_variants, dim=0), requires_grad=True)
        self.convs = nn.ModuleList([
            DepthwiseSeparableSpatiotemporalConv(in_ch, out_ch)
            for _ in range(self.num_topologies)
        ])
        self.fusion = MeanTopologyFusion()
    def forward(self, x):
        N, C, T, V = x.shape
        topology_outputs = []
        for i in range(self.num_topologies):
            x_graph = torch.einsum('nctv,vw->nctw', x, self.A[i])
            out_i = self.convs[i](x_graph)
            topology_outputs.append(out_i)
        out = self.fusion(topology_outputs)
        return out
