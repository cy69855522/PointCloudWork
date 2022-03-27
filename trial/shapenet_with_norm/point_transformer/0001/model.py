import torch
import torch.nn as nn

from module.common.mlp import MLP
from module.common.norm import Norm
from module.point_transformer.point_transformer import PointTransformerBlock, TransitionDown, TransitionUp

from module.common.knn import knn

class Net(nn.Module):
    def __init__(self, out_channels, norm_type='bn', num_neighbors=16, dilation=1, num_interpolation_neighbors=3):
        super(Net, self).__init__()
        
        self.K = num_neighbors
        self.D = dilation
        self.K_inte = num_interpolation_neighbors
        
        self.mlp1 = MLP([6, 32, 32], 1, norm_type)
        self.mlp2 = MLP([512, 512], 1, norm_type)
        self.mlp3 = MLP([1024, 512], 1, norm_type)
        self.td1 = TransitionDown(32, 64, norm_type, ratio=0.25)
        self.td2 = TransitionDown(64, 128, norm_type, ratio=0.25)
        self.td3 = TransitionDown(128, 256, norm_type, ratio=0.25)
        self.td4 = TransitionDown(256, 512, norm_type, ratio=0.25)
        self.ptbs1 = nn.ModuleList([PointTransformerBlock(32, 32, norm_type) for _ in range(1)])
        self.ptbs2 = nn.ModuleList([PointTransformerBlock(64, 64, norm_type) for _ in range(2)])
        self.ptbs3 = nn.ModuleList([PointTransformerBlock(128, 128, norm_type) for _ in range(3)])
        self.ptbs4 = nn.ModuleList([PointTransformerBlock(256, 256, norm_type) for _ in range(5)])
        self.ptbs5 = nn.ModuleList([PointTransformerBlock(512, 512, norm_type) for _ in range(2)])
        self.ptbs6 = nn.ModuleList([PointTransformerBlock(512, 512, norm_type) for _ in range(2)])
        self.ptbs7 = nn.ModuleList([PointTransformerBlock(256, 256, norm_type) for _ in range(2)])
        self.ptbs8 = nn.ModuleList([PointTransformerBlock(128, 128, norm_type) for _ in range(2)])
        self.ptbs9 = nn.ModuleList([PointTransformerBlock(64, 64, norm_type) for _ in range(2)])
        self.ptbs10 = nn.ModuleList([PointTransformerBlock(32, 32, norm_type) for _ in range(2)])
        self.tu1 = TransitionUp(512, 256, norm_type, self.K_inte)
        self.tu2 = TransitionUp(256, 128, norm_type, self.K_inte)
        self.tu3 = TransitionUp(128, 64, norm_type, self.K_inte)
        self.tu4 = TransitionUp(64, 32, norm_type, self.K_inte)
        
        self.fc = MLP([32, 32, out_channels], 1, norm_type, last_acti=False, last_norm=False)
        
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
        
        p1 = pos.view(B, N, 3) # [B, N, C]
        x1 = torch.cat([pos, norm], dim=-1).view(B, N, 6) # [B, N, C]
        id_euc1 = knn(p1, K, D)
        x1 = self.mlp1(x1) # [B, N, C]
        for ptb in self.ptbs1: x1 = ptb(x1, p1, id_euc1) # [B, N, C]
        
        x2, p2 = self.td1(x1, p1, id_euc1)
        id_euc2 = knn(p2, K, D)
        for ptb in self.ptbs2: x2 = ptb(x2, p2, id_euc2)
        
        x3, p3 = self.td2(x2, p2, id_euc2)
        id_euc3 = knn(p3, K, D)
        for ptb in self.ptbs3: x3 = ptb(x3, p3, id_euc3)
        
        x4, p4 = self.td3(x3, p3, id_euc3)
        id_euc4 = knn(p4, K, D)
        for ptb in self.ptbs4: x4 = ptb(x4, p4, id_euc4)
        
        x5, p5 = self.td4(x4, p4, id_euc4)
        id_euc5 = knn(p5, min(K, p5.size(1)), D)
        for ptb in self.ptbs5: x5 = ptb(x5, p5, id_euc5)
        
        x_mean = x5.mean(dim=1)  # [B, C]
        x_mean = self.mlp2(x_mean)  # [B, C]
        x_mean = x_mean[:, None].expand(B, x5.size(1), x_mean.size(-1))  # [B, N, C]
        x6 = torch.cat([x5, x_mean], dim=-1)  # [B, N, C]
        x6 = self.mlp3(x6)  # [B, N, C]
        for ptb in self.ptbs6: x6 = ptb(x6, p5, id_euc5)

        x7 = self.tu1(x6, x4, p5, p4)
        for ptb in self.ptbs7: x7 = ptb(x7, p4, id_euc4)

        x8 = self.tu2(x7, x3, p4, p3)
        for ptb in self.ptbs8: x8 = ptb(x8, p3, id_euc3)

        x9 = self.tu3(x8, x2, p3, p2)
        for ptb in self.ptbs9: x9 = ptb(x9, p2, id_euc2)

        x10 = self.tu4(x9, x1, p2, p1)
        for ptb in self.ptbs10: x10 = ptb(x10, p1, id_euc1)

        x = self.fc(x10)  # [B, N, C]
        x = x.transpose(1, 2)  # [B, C, N]
        
        return x