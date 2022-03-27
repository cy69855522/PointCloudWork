import torch
import torch.nn as nn

from module.common.mlp import MLP
from module.common.norm import Norm
from module.point_transformer.point_transformer import PointTransformerBlock, TransitionDown

from module.common.knn import knn

class Net(nn.Module):
    def __init__(self, out_channels, norm_type='bn', num_neighbors=16, dilation=1):
        super(Net, self).__init__()
        
        self.K = num_neighbors
        self.D = dilation
        
        self.mlp1 = MLP([6, 32, 32], 1, norm_type)
        self.td1 = TransitionDown(32, 64, norm_type, ratio=0.25)
        self.td2 = TransitionDown(64, 128, norm_type, ratio=0.25)
        self.td3 = TransitionDown(128, 256, norm_type, ratio=0.25)
        self.td4 = TransitionDown(256, 512, norm_type, ratio=0.25)
        self.ptbs1 = nn.ModuleList([PointTransformerBlock(32, 32, norm_type) for _ in range(1)])
        self.ptbs2 = nn.ModuleList([PointTransformerBlock(64, 64, norm_type) for _ in range(2)])
        self.ptbs3 = nn.ModuleList([PointTransformerBlock(128, 128, norm_type) for _ in range(3)])
        self.ptbs4 = nn.ModuleList([PointTransformerBlock(256, 256, norm_type) for _ in range(5)])
        self.ptbs5 = nn.ModuleList([PointTransformerBlock(512, 512, norm_type) for _ in range(2)])
        
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            Norm(norm_type, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            Norm(norm_type, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, out_channels)
        )
        
        # init
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, data):
        pos, batch, norm = data.pos, data.batch, data.norm
        dev = pos.device
        B = batch.max().item() + 1
        
        N = int(len(pos) / B)
        K = self.K
        D = self.D
        
        p = pos.view(B, N, 3) # [B, N, C]
        x = torch.cat([pos, norm], dim=-1).view(B, N, 6) # [B, N, C]
        id_euc = knn(p, K, D)
        x = self.mlp1(x) # [B, N, C]
        for ptb in self.ptbs1: x = ptb(x, p, id_euc) # [B, N, C]
        
        x, p = self.td1(x, p, id_euc)
        id_euc = knn(p, K, D)
        for ptb in self.ptbs2: x = ptb(x, p, id_euc)
        
        x, p = self.td2(x, p, id_euc)
        id_euc = knn(p, K, D)
        for ptb in self.ptbs3: x = ptb(x, p, id_euc)
        
        x, p = self.td3(x, p, id_euc)
        id_euc = knn(p, K, D)
        for ptb in self.ptbs4: x = ptb(x, p, id_euc)
        
        x, p = self.td4(x, p, id_euc)
        id_euc = knn(p, min(K, p.size(1)), D)
        for ptb in self.ptbs5: x = ptb(x, p, id_euc)
        
        x = x.mean(dim=1)  # [B, C]
        x = self.fc(x)  # [B, C]
        
        return x