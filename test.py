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

from data import *
from dataset import *
from module.common.criterion import Criterion
from utils import parse_config, set_seeds, set_devices

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--trial', default='modelnet40_dgcnn_0001', type=str,
                    help='The trial name with its version, e.g., modelnet40_gcnn_0001.')
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
                    help='Checkpoint path. "best" means the best checkpoint of saved ones.')

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
                        shuffle=True,
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
                    global_metric_results = criterion.global_metric_resuls
                    criterion.reset()
                    t.set_postfix(**global_metric_results)

    return global_metric_results

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
    if args.pretrained_weight_path:
        pretrained_weight = torch.load(args.pretrained_weight_path, map_location=device)
        model.load_state_dict(pretrained_weight)
        print(f'Load the pretrained weight: {args.pretrained_weight_path}')
    if args.checkpoint_path == 'best':
        checkpoint_dir = os.path.join(args.trial_dir, 'checkpoints')
        checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, 'epoch_*.weight'))
        get_target_fn = lambda x : x.split('/')[-1].split('.weight')[0].split('_')[-1]
        checkpoint_paths.sort(key=lambda x : float(get_target_fn(x)))
        if checkpoint_paths:
            best_checkpoint_path = checkpoint_paths[-1]
            best_checkpoint = torch.load(best_checkpoint_path,
                                         map_location=device,
                                         pickle_module=dill)
            model.load_state_dict(best_checkpoint['model'].state_dict())
            print(f'Load the best checkpoint: {best_checkpoint_path}')
    elif args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path,
                                map_location=device,
                                pickle_module=dill)
        model.load_state_dict(checkpoint['model'].state_dict())
        print(f'Load the checkpoint: {args.checkpoint_path}')
        
    criterion = Criterion(args.task_type,
                          args.criterion,
                          args.show_loss_details,
                          args.max_float_digits).to(device)

    # Testing
    evaluate(args, loader, model, device, criterion)
