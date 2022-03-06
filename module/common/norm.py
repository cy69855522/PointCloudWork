import torch.nn as nn

def Norm(norm_type, num_features, dimension=1, features_per_group=16, momentum=0.1):
    if norm_type == 'bn':
        return eval(f'nn.BatchNorm{dimension}d')(num_features, momentum=momentum)
    elif norm_type == 'gn':
        num_group = num_features // features_per_group
        if num_group * features_per_group != num_features:
            num_group = 1
        return nn.GroupNorm(num_group, num_features)