import torch
import torch.nn as nn


class MeanTopologyFusion(nn.Module):
    def forward(self, topology_outputs):
        stacked = torch.stack(topology_outputs, dim=0)
        return stacked.mean(dim=0)
