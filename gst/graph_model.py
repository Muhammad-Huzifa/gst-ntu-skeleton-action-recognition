import torch
import torch.nn as nn

from gst.topology_fusion import AdaptiveMultiTopologyFusion
from gst.graph_encoder import GraphSpatialEncoder
from gst.temporal_encoder import GraphTemporalTransformer


def run_persons(encode, x_joint, x_bone, person_mask=None):
    if x_joint.dim() == 4:
        return encode(x_joint, x_bone)
    N, C, T, V, M = x_joint.shape
    x_joint = x_joint.permute(0, 4, 1, 2, 3).reshape(N * M, C, T, V)
    x_bone = x_bone.permute(0, 4, 1, 2, 3).reshape(N * M, x_bone.size(1), T, V)
    f = encode(x_joint, x_bone).view(N, M, -1)
    if person_mask is None:
        return f.mean(1)
    w = person_mask.to(f.dtype).unsqueeze(-1)
    return (f * w).sum(1) / w.sum(1).clamp(min=1)


class GraphActionModel(nn.Module):
    def __init__(self, graph, num_classes, num_virtual_classes=0, joint_in_ch=6, bone_in_ch=4, hidden_dim=64, num_heads=4, spatial_depth=2, temporal_depth=2, gcn_hops=2, mid='amtf', share_streams=False, part_pool=True, temporal_pool='query', max_rel=16, drop=0.0):
        super().__init__()
        h = hidden_dim
        self.joint_embed = nn.Sequential(nn.Conv2d(joint_in_ch, h, 1, bias=False), nn.BatchNorm2d(h), nn.ReLU(inplace=True))
        self.bone_embed = nn.Sequential(nn.Conv2d(bone_in_ch, h, 1, bias=False), nn.BatchNorm2d(h), nn.ReLU(inplace=True))
        self.spatial_joint = GraphSpatialEncoder(graph, h, num_heads, spatial_depth, gcn_hops, drop)
        self.spatial_bone = self.spatial_joint if share_streams else GraphSpatialEncoder(graph, h, num_heads, spatial_depth, gcn_hops, drop)
        if mid == 'amtf':
            self.mid = AdaptiveMultiTopologyFusion(h * 2, h * 3, graph.A)
        elif mid == 'none':
            self.mid = nn.Sequential(nn.Conv2d(h * 2, h * 3, 1, bias=False), nn.BatchNorm2d(h * 3), nn.ReLU(inplace=True))
        else:
            raise ValueError(mid)
        self.temporal = GraphTemporalTransformer(graph, h * 3, h * 3, num_heads, temporal_depth, max_rel, part_pool=part_pool, pool=temporal_pool, drop=drop)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(h * 3, num_classes + num_virtual_classes)
        self._init_weights()
    def _init_weights(self):
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
        x_joint = self.spatial_joint(self.joint_embed(x_joint))
        x_bone = self.spatial_bone(self.bone_embed(x_bone))
        return self.temporal(self.mid(torch.cat([x_joint, x_bone], dim=1)))
    def forward(self, x_joint, x_bone, person_mask=None):
        return self.fc(self.dropout(run_persons(self.encode, x_joint, x_bone, person_mask)))
