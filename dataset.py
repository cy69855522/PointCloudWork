import torch

from torch_geometric.data import InMemoryDataset, Data

import glob
import h5py
import os
from itertools import repeat

class ModelNet40PlusID(InMemoryDataset):
    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None, norm=False, radius=False):
        self.norm = norm
        self.radius = radius
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        pass
        
    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])
        
    def process_set(self, dataset):
        data_list = []
        for h5_name in glob.glob(os.path.join(self.raw_dir, 'ply_data_%s*.h5'%dataset)):
            f = h5py.File(h5_name)
            data = torch.from_numpy(f['data'][:].astype('float32'))  # [B, N, C]
            label = torch.from_numpy(f['label'][:].astype('int64'))
            normal = torch.from_numpy(f['normal'][:].astype('float32')) if self.norm else [None] * len(data)
            radius = (data - data.mean(dim=1, keepdim=True)).norm(dim=-1) if self.radius else [None] * len(data)
            data_list.extend([Data(pos=a, y=b, norm=c, radius=d) for a, b, c, d in zip(data, label, normal, radius)])
            f.close()

        return self.collate(data_list)
        
    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))
    
    def get(self, idx, onlyPoint=False):
        data = self.data.__class__()
    
        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]
    
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[self.data.__cat_dim__(key,  item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
    
        data.pos = data.pos.clone()
        if 'norm' in data: data.norm = data.norm.clone()
        data['path_id'] = idx
        
        return data