import torch
import torch.nn as nn

from module.common.mlp import MLP
from module.common.norm import Norm
from module.dgcnn.edge_conv import EdgeConvBlock

class Net(nn.Module):
    def __init__(self, out_channels, norm_type='bn', num_neighbors=20, dilation=1):
        super().__init__()
        self.k = num_neighbors
        self.d = dilation

        self.block1 = EdgeConvBlock([3, 64, 128], self.k, self.d, norm_type)
        self.block2 = EdgeConvBlock([128, 128, 128], self.k, self.d, norm_type)
        self.block3 = EdgeConvBlock([128, 128, 128], self.k, self.d, norm_type)
        self.mlp = MLP([128, 256, 1024], 1, norm_type)
        self.fcs = nn.Sequential(
            nn.Linear(1024, 512),
            Norm(norm_type, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            Norm(norm_type, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, out_channels)
        )

    def forward(self, data):
        pos, batch = data.pos, data.batch  # [B*N, C]
        dev = pos.device
        B = batch.max().item() + 1
        N = int(len(pos) / B)
        k = self.k
        
        x = pos.view(B, N, -1)  # [B, N, C]
        x = self.block1(x)  # [B, N, C]
        x = self.block2(x)  # [B, N, C]
        x = self.block3(x)  # [B, N, C]
        x = self.mlp(x)  # [B, N, C]
        x = x.max(dim=1)[0]  # [B, C]
        x = self.fcs(x)  # [B, C]
        
        return x