import torch

from torch_geometric.data import InMemoryDataset, Data

import copy
import glob
import h5py
import json
import numpy as np
import os
from itertools import repeat
from tqdm import tqdm

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
    
    def num_classes(self, task_type):
        if task_type == 'classification':
            return int(self.data.y.max()) + 1

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
            data[key] = item[s].clone()

        data['id'] = idx
        return data

class ShapeNet(InMemoryDataset):
    cat_to_seg = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None, norm=False, radius=False):
        self.norm = norm
        self.radius = radius
        with open(os.path.join(root, 'raw', 'synsetoffset2category.txt'), 'r') as f:
            self.categories = [line.strip().split()[0] for line in f]
            self.categories.sort()
            self.cat_to_y = {cat : y for y, cat in enumerate(self.categories)}

        super(ShapeNet, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [
            'synsetoffset2category.txt', 'train_test_split'
        ]

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']
    
    def num_classes(self, task_type):
        if task_type == 'classification':
            return int(self.data.y.max()) + 1
        elif task_type == 'segmentation':
            return int(self.data.seg.max()) + 1

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset):
        self.cat_to_dir = {}
        with open(os.path.join(self.raw_dir, 'synsetoffset2category.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat_to_dir[ls[0]] = ls[1]
        self.meta = {}
        with open(os.path.join(self.raw_dir, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.raw_dir, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.raw_dir, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for k, v in self.cat_to_dir.items():
            self.meta[k] = []
            dir_point = os.path.join(self.raw_dir, v)
            fns = sorted(os.listdir(dir_point))
            if dataset == 'train':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif dataset == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..'%(dataset))
                exit(-1)
                
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0]) 
                self.meta[k].append(os.path.join(dir_point, token + '.txt'))
        
        data_list = []
        with tqdm([(k, fn) for k in self.cat_to_dir for fn in self.meta[k]],
                  desc=f'Processing the {dataset}ing dataset.') as t:
            for k, fn in t:
                raw_data = torch.from_numpy(np.loadtxt(fn).astype(np.float32))
                data = Data(pos=raw_data[:, 0:3],
                            seg=raw_data[:, -1].long(),
                            y=torch.LongTensor([self.cat_to_y[k]]))
                if self.norm:
                    data['norm'] = raw_data[:, 3:-1]
                if self.radius:
                    data['radius'] = (data.pos - data.pos.mean(dim=0, keepdim=True)).norm(dim=-1)
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)

    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))
    
    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[idx],
                                                       slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s].clone()

        data['id'] = idx
        return data