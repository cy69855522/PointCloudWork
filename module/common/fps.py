import torch

from torch_geometric.nn import fps

def farthest_point_sample(pos, num_subset_points, random_start=False):
    # pos[B, N, C]
    B, N, _ = pos.size()
    n = num_subset_points
    sample_ratio = n / N
    device = pos.device

    batch = torch.arange(B, device=device)[:, None].repeat(1, N).view(-1)  # [B*N]
    pos = pos.view(B*N, -1)
    subset_id = fps(pos, batch, sample_ratio, random_start)  # [B*n]
    subset_id = subset_id.view(B, n)
    subset_id -= torch.arange(B, device=device)[:, None] * N
    return subset_id
