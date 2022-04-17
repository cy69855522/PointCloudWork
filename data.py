import torch

import math
import numpy as np
import random

class Transform(object):
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class ShufflePoints(Transform):
    def __call__(self, data):
        idx = torch.randperm(data.pos.size(0))
        data['pos'] = data.pos[idx]
        if 'norm' in data:
            data.norm = data.norm[idx]
        if 'seg' in data:
            data.seg = data.seg[idx]
        return data

class UnitSphere(Transform):
    r"""
    Center and normalize point positions to an unit sphere.
    """
    def __call__(self, data):
        data.pos -= data.pos.mean(dim=-2)
        data.pos /= data.pos.norm(dim=-1).max()
        return data

class ChunkPoints(Transform):
    def __init__(self, num_points=1024, random_start=False):
        self.N = num_points
        self.random_start = random_start
        
    def __call__(self, data):
        num_points = data.pos.size(0)
        repeatition = math.ceil(self.N / num_points)
        num_repeated_points = num_points * repeatition
        start = 0
        if self.random_start:
            start = random.randint(0, num_repeated_points - self.N)
        attrs = ['pos']
        if 'norm' in data:
            attrs.append('norm')
        if 'seg' in data:
            attrs.append('seg')
        for attr in attrs:
            if repeatition > 1:
                data[attr] = torch.cat([data[attr]] * repeatition, dim=0)
            data[attr] = data[attr][start : start + self.N]
        return data

class ScalePoints(Transform):
    def __init__(self, scale_range=(2./3, 3./2)):
        self.scale_range = scale_range

    def __call__(self, data):
        dty = data.pos.dtype
        dev = data.pos.device
        data['pos'] *= torch.ones(3, dtype=dty, device=dev).uniform_(*self.scale_range)
        return data

class Jitter(Transform):
    def __init__(self, std=0.01, clip=0.05):
        self.std = std
        self.clip = clip

    def __call__(self, data):
        jittered_data = data.pos.new(data.pos.size(0), 3).normal_(
            mean=0.0, std=self.std
        ).clamp_(-self.clip, self.clip)
        data['pos'] += jittered_data
        return data

class MovePoints(Transform):
    def __init__(self, move_range=(-0.2, 0.2)):
        self.move_range = move_range

    def __call__(self, data):
        dty = data.pos.dtype
        dev = data.pos.device
        data['pos'] += torch.ones(3, dtype=dty, device=dev).uniform_(*self.move_range)
        return data

class FixedPoints(Transform):
    def __init__(self, num, balance=True):
        self.num = num
        self.balance = balance

    def __call__(self, data):
        num_nodes = data.num_nodes

        if self.balance and num_nodes < self.num:
            div = self.num // num_nodes
            mod = self.num % num_nodes
            repetition = [np.random.choice(num_nodes, num_nodes, replace=False) for _ in range(div)]
            overflow = [np.random.choice(num_nodes, mod, replace=False)]
            choice = np.concatenate(repetition + overflow, axis=0)
        else:
            choice = np.random.choice(num_nodes, self.num, replace=True)

        for key, item in data:
            if 'edge' in key:
                continue
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                data[key] = item[choice]

        return data