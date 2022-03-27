import torch
import torch.nn as nn

from torch_geometric.nn import fps, knn_interpolate

import math

from ..common.knn import knn
from ..common.mlp import MLP
from ..common.norm import Norm

class PointTransformer(nn.Module):
    def __init__(self, channels, norm_type='bn', shared_channels=8):
        super().__init__()
        self.S = shared_channels
        self.mlp1 = MLP([3, 3, channels], 1, norm_type, last_norm=False, last_acti=False)
        self.mlp2 = MLP([1, channels, channels // self.S, channels // self.S],
                        2, norm_type, last_norm=False, last_acti=False, first_linear=False)
        self.fc = nn.Linear(channels, 3 * channels)

    def forward(self, x, p, id_euc):
        # x[B, N, C] p[B, N, C] sid/tid[B*N, K]
        B, N, C = x.size()
        S = self.S
        sid_euc, tid_euc = id_euc
        K = sid_euc.size(1)
        
        p = self.mlp1(p)  # [B, N, C]
        q, k, v = self.fc(x).chunk(3, dim=-1)  # [B, N, C]
        
        p_i = p.view(B*N, -1)[tid_euc]  # [B*N, K, C]
        p_j = p.view(B*N, -1)[sid_euc]  # [B*N, K, C]
        p = (p_i - p_j).view(B, N, K, -1)
        q = q.view(B*N, -1)[tid_euc].view(B, N, K, -1)
        k = k.view(B*N, -1)[sid_euc].view(B, N, K, -1)
        v = v.view(B*N, -1)[sid_euc].view(B, N, K, -1)
        a = self.mlp2(q - k + p).softmax(dim=2)  # [B, N, K, C//S]
        v = (v + p).view(B, N, K, C // S, S)
        x = (a[..., None] * v).sum(dim=2).view(B, N, -1)  # [B, N, C]
        return x

class PointTransformerBlock(nn.Module):
    def __init__(self, in_channels, hid_channels, norm_type='bn', shared_channels=8):
        super().__init__()
        self.mlp1 = MLP([in_channels, hid_channels], 1, norm_type)
        self.mlp2 = MLP([1, hid_channels], 1, norm_type, first_linear=False)
        self.mlp3 = MLP([hid_channels, in_channels], 1, norm_type, last_acti=False)
        self.pt1 = PointTransformer(hid_channels, norm_type, shared_channels)
        self.acti1 = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x, p, id_euc):
        # x[B, N, C] p[B, N, C] sid/tid[B*N, K]
        identity = x
        x = self.mlp1(x)
        x = self.mlp2(self.pt1(x, p, id_euc))
        x = self.acti1(identity + self.mlp3(x))
        return x

class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='bn', ratio=0.25):
        super().__init__()
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], 1, norm_type)
    
    def forward(self, x, p, id_euc):
        # x[B, N, C] p[B, N, C] sid/tid[B*N, K]
        B, N, _ = x.size()
        n = math.ceil(N * self.ratio)
        sid_euc, tid_euc = id_euc
        K = sid_euc.size(1)
        dev = x.device
        
        x = self.mlp(x)
        
        batch = torch.arange(B, device=dev)[:, None].repeat(1, N).view(-1)  # [B*N]
        p = p.view(B*N, -1)
        fid = fps(p, batch, ratio=self.ratio, random_start=self.training)  # [B*n]
        p = p[fid].view(B, n, -1)
        
        sid_euc = sid_euc[fid].view(-1)  # [B*n*K]
        x = x.view(B*N, -1)[sid_euc].view(B, n, K, -1).max(dim=-2)[0]  # [B, n, C]
        
        return x, p

class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='bn', num_neighbors=4):
        super().__init__()
        self.K_inte = num_neighbors
        self.mlp_down = MLP([in_channels, out_channels], 1, norm_type)
        self.mlp_up = MLP([out_channels, out_channels], 1, norm_type)
    
    def forward(self, x_down, x_up, p_down, p_up):
        # x_down[B, n, C] x_up[B, N, C] p_down[B, n, C] p_up[B, N, C]
        B, n, _ = x_down.size()
        N = p_up.size(1)
        K_inte = self.K_inte
        dev = x_down.device
        
        p_down = p_down.view(B*n, -1)
        p_up = p_up.view(B*N, -1)
        x_down = self.mlp_down(x_down)  # [B, n, C]
        x_down = x_down.view(B*n, -1)  # [B*n, C]

        batch_down = torch.arange(B, device=dev)[:, None].repeat(1, n).view(-1)  # [B*n]
        batch_up = torch.arange(B, device=dev)[:, None].repeat(1, N).view(-1)  # [B*N]
        x = knn_interpolate(x_down, p_down, p_up, batch_down, batch_up, k=K_inte)  # [B*N, C]
        x = x.view(B, N, -1)  # [B, N, C]

        x_up = self.mlp_up(x_up)  # [B, N, C]
        x = x + x_up
        
        return x