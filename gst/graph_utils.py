import torch

SLR65_PAIRS = [(1,0), (2,0), (3,1), (4,2), (11,0), (12,0), (13,11), (14,12), (15,13), (16,14), (24,23), (25,24), (26,25), (27,26), (28,23), (29,28), (30,29), (31,30), (32,23), (33,32), (34,33), (35,34), (36,23), (37,36), (38,37), (39,38), (40,23), (41,40), (42,41), (43,42), (45,44), (46,45), (47,46), (48,47), (49,44), (50,49), (51,50), (52,51), (53,44), (54,53), (55,54), (56,55), (57,44), (58,57), (59,58), (60,59), (61,44), (62,61), (63,62), (64,63)]
NTU_PAIRS_1IDX = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
NTU_PARTS = [[0, 1, 2, 3, 20], [4, 5, 6, 7, 21, 22], [8, 9, 10, 11, 23, 24], [12, 13, 14, 15], [16, 17, 18, 19]]
SLR65_PARTS = [list(range(0, 11)), [11, 13, 15, 17, 19, 21], [12, 14, 16, 18, 20, 22], list(range(23, 44)), list(range(44, 65))]


def shortest_path(A, max_hop):
    V = A.size(0)
    unreachable = max_hop + 2
    dist = torch.full((V, V), -1, dtype=torch.long)
    dist.fill_diagonal_(0)
    walk = torch.eye(V)
    for h in range(1, V):
        walk = ((walk @ A) > 0).float()
        new = (walk > 0) & (dist < 0)
        if not new.any():
            break
        dist[new] = h
    return torch.where(dist >= 0, dist.clamp(max=max_hop + 1), torch.full_like(dist, unreachable))


def hop_adjacency(spd, num_hops):
    mats = []
    for k in range(num_hops + 1):
        m = (spd == k).float()
        mats.append(m / m.sum(1, keepdim=True).clamp(min=1))
    return torch.stack(mats, 0)


class GraphSpec:
    def __init__(self, num_node, edges, parts, max_hop=4):
        self.num_node = num_node
        self.parts = parts
        self.max_hop = max_hop
        A = torch.zeros(num_node, num_node)
        for i, j in edges:
            if i != j:
                A[i, j] = 1
                A[j, i] = 1
        self.A_bin = A
        self.A = A + torch.eye(num_node)
        self.spd = shortest_path(A, max_hop)
        self.num_buckets = max_hop + 3
        covered = sorted(j for p in parts for j in p)
        assert covered == list(range(num_node)), "parts must partition all joints exactly once"


def ntu25_graph(max_hop=12):
    edges = [(a - 1, b - 1) for a, b in NTU_PAIRS_1IDX]
    return GraphSpec(25, edges, NTU_PARTS, max_hop)


def slr65_graph(max_hop=4, link_wrists=False):
    edges = list(SLR65_PAIRS)
    if link_wrists:
        edges += [(15, 23), (16, 44)]
    return GraphSpec(65, edges, SLR65_PARTS, max_hop)
