import torch
import torch.nn as nn

from gst.baseline_attention import PreNormTransformerBlock
from gst.graph_model import run_persons
from gst.topology_fusion import AdaptiveMultiTopologyFusion


class BaselineAMTF(nn.Module):
    def __init__(self, graph, num_classes, num_virtual_classes=0, joint_in_ch=6, bone_in_ch=4, hidden_dim=64, num_heads=4, transformer_depth=2):
        super().__init__()
        h = hidden_dim
        self.joint_embed = nn.Sequential(nn.Conv2d(joint_in_ch, h, 1, bias=False), nn.BatchNorm2d(h), nn.ReLU(inplace=True))
        self.bone_embed = nn.Sequential(nn.Conv2d(bone_in_ch, h, 1, bias=False), nn.BatchNorm2d(h), nn.ReLU(inplace=True))
        self.spatial_blocks = nn.ModuleList([PreNormTransformerBlock(h, num_heads) for _ in range(transformer_depth)])
        self.ctr_block = AdaptiveMultiTopologyFusion(h * 2, h * 3, graph.A)
        self.temporal_blocks = nn.ModuleList([PreNormTransformerBlock(h * 3, num_heads) for _ in range(transformer_depth)])
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(h * 3, num_classes + num_virtual_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def encode(self, x_joint, x_bone):
        N, _, T, V = x_joint.shape
        xj = self.joint_embed(x_joint).permute(0, 2, 3, 1).contiguous().view(N * T, V, -1)
        xb = self.bone_embed(x_bone).permute(0, 2, 3, 1).contiguous().view(N * T, V, -1)
        for block in self.spatial_blocks:
            xj, xb = block(xj), block(xb)
        xj = xj.view(N, T, V, -1).permute(0, 3, 1, 2).contiguous()
        xb = xb.view(N, T, V, -1).permute(0, 3, 1, 2).contiguous()
        x = self.ctr_block(torch.cat([xj, xb], dim=1)).mean(-1).permute(0, 2, 1).contiguous()
        for block in self.temporal_blocks:
            x = block(x)
        return x.mean(1)
    def forward(self, x_joint, x_bone, person_mask=None):
        return self.fc(self.dropout(run_persons(self.encode, x_joint, x_bone, person_mask)))
