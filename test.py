# -*- coding: utf8 -*-

import torch
import torch.nn as nn
import torch.optim as optim

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

import argparse
import dill
import glob
import importlib
import os
from tqdm import tqdm

from criterion import Criterion
from data import *
from dataset import *
from utils import parse_config, set_seeds, set_devices, load_pretrained_weight, load_checkpoint

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--trial', default='modelnet40.dgcnn.0001', type=str,
                    help='The trial name with its version, e.g., modelnet40.dgcnn.0001.')
parser.add_argument('--show_configs', default=True, type=bool,
                    help='Whether to print the config at the program beginning.')
parser.add_argument('--show_loss_details', default=False, type=bool,
                    help='Whether to show all loss details.')
parser.add_argument('--separator_bar', default='*' * 100, type=str,
                    help='Separator bar.')
parser.add_argument('--max_float_digits', default=4, type=int,
                    help='Limit the max digits to display.')
parser.add_argument('--pretrained_weight_path', default='', type=str,
                    help='The pre-trained weight path.')
parser.add_argument('--checkpoint_path', default='best', type=str,
                    help='Checkpoint path. "best" means the best checkpoint of saved ones. "last" means the last one.')

def initilize_testing_loader(args):
    pre_transform = T.Compose([*map(eval, args.pre_transforms)])
    transform = T.Compose([*map(eval, args.testing.transforms)])
    dataset = eval(args.data_class)(root=f'data/{args.trial_dataset}',
                                    train=False,
                                    transform=transform,
                                    pre_transform=pre_transform,
                                    **args.data_class_kwargs)
    loader = DataLoader(dataset,
                        batch_size=args.testing.batch_size,
                        shuffle=False,
                        drop_last=False,
                        pin_memory=True,
                        num_workers=args.testing.num_workers)
    return loader

def evaluate(args, loader, model, device, criterion):
    model.eval()
    with torch.no_grad():
        with tqdm(loader, desc=f'Testing') as t:
            for i, data in enumerate(t):
                data = data.to(device)
                prediction = model(data)
                metric_results = criterion(prediction, data, loader.dataset)
                loss = metric_results['loss']
                del metric_results['loss']

                t.set_postfix(loss=f'%.{args.max_float_digits}f' % loss.item(),
                              **metric_results)
            
                if i == len(loader) - 1:
                    general_results = criterion.general_results
                    t.set_postfix(**general_results)

    return general_results

if __name__ == "__main__":
    args = parser.parse_args()
    parse_config(args)
    
    # Dynamically import the model class, Net.
    model_path = os.path.join(args.trial_dir, 'model')
    model_package = importlib.import_module(model_path.replace('/', '.'))

    # Set the environment.
    set_seeds(args.seed)
    device = set_devices(args.cuda_devices, args.separator_bar)

    # Load the dataset.
    loader = initilize_testing_loader(args)

    # Set the model.
    model = model_package.Net(loader.dataset.num_classes(args.task_type),
                              **args.net_arguments).to(device)
    load_pretrained_weight(args.pretrained_weight_path, model, device)
    load_checkpoint(args.trial_dir, args.checkpoint_path, model, device)
    
    criterion = Criterion(args.task_type,
                          args.criterion,
                          args.show_loss_details,
                          args.max_float_digits).to(device)

    # Testing
    evaluate(args, loader, model, device, criterion)
    criterion.reset()
