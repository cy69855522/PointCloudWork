import torch

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
        start = 0
        if self.random_start:
            start = random.randint(0, data.pos.size(0) - self.N)
        data['pos'] = data.pos[start : start + self.N]
        if 'norm' in data:
            data.norm = data.norm[start : start + self.N]
        if 'seg' in data:
            data.seg = data.seg[start : start + self.N]
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
