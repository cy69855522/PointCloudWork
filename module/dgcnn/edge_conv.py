import torch
import torch.nn as nn

from ..common.knn import knn
from ..common.mlp import MLP
from ..common.norm import Norm

class EdgeConv(nn.Module):
    def __init__(self, channels, norm_type='bn'):
        super().__init__()
        channels[0] *= 2
        self.edge_mlp = MLP(channels, 2, norm_type)

    def forward(self, x, id_euc, B):
        #  x[B*N, C] source_id/target_id[B*n, k]
        sid_euc, tid_euc = id_euc
        n = sid_euc.size(0) // B
        k = sid_euc.size(1)

        euc_i, euc_j = x[tid_euc], x[sid_euc]  # [B*n, k, C]
        euc_diff = euc_j - euc_i
        edge = torch.cat([euc_i, euc_diff], dim=1)  # [B*n, k, C]
        edge = edge.view(B, n, k, -1)
        edge = self.edge_mlp(edge)
        x = edge.max(2)[0]  # [B, n, C]
        x = x.view(B*n, -1)
        return x

class EdgeConvBlock(nn.Module):
    def __init__(self, channels, k, d=1, norm_type='bn'):
        super().__init__()
        self.k = k
        self.d = d
        self.conv = EdgeConv(channels, norm_type)
        self.norm = Norm(norm_type, channels[-1])
        self.acti = nn.ReLU(inplace=True)

    def forward(self, x, subset_id=None):
        #  x[B, N, C] subset_id[B*n]
        B, N, _ = x.size()
        n = N if subset_id is None else (subset_id.size(0) // B)
        id_euc = knn(x, self.k, self.d, subset_id) # [B*n, k]

        x = x.view(B*N, -1)
        x = self.conv(x, id_euc, B)  # [B*n, C]
        x = x.view(B, n, -1)
        x = x.transpose(1, 2)  # [B, C, n]
        x = self.acti(self.norm(x))
        x = x.transpose(1, 2).contiguous()  # [B, n, C]
        
        return x