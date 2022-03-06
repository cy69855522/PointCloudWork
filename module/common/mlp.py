import torch.nn as nn

from .norm import Norm

class MLP(nn.Module):
    def __init__(self, channels, dimensions=1, norm_type='bn', resi=False, last_acti=True, last_norm=True, **kwargs):
        super().__init__()
        self.linears = nn.ModuleList([eval(f'nn.Conv{dimensions}d')(i, o, 1) for i, o in zip(channels, channels[1:])])
        self.norms = nn.ModuleList([Norm(norm_type, o, dimensions, **kwargs) for o in channels[1:]])
        self.actis = nn.ModuleList([nn.LeakyReLU(0.2, inplace=True) for _ in channels[1:]])
        self.resi = resi
        self.last_acti = last_acti
        self.last_norm = last_norm

    def forward(self, x, feature_last=True): # [B, ..., C]
        twoD = len(x.shape) == 2
        if twoD:
            feature_last = False
            x = x.unsqueeze(-1)
        if feature_last: x = x.transpose(1, -1) # [B, C, ...]
        
        for i, (linear, norm, acti) in enumerate(zip(self.linears, self.norms, self.actis)):
            inp = x if self.resi else None
            x = linear(x)
            if i != len(self.actis)-1 or self.last_norm:
                x = norm(x)
            if i != len(self.actis)-1 or self.last_acti:
                x = acti(x)
            if inp is not None and x.shape == inp.shape:
                x = x + inp
        
        if feature_last:
            x = x.transpose(1, -1) # [B, ..., C]
            x = x.contiguous()
        if twoD: x = x.squeeze(-1)

        return x