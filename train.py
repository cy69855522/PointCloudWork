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
import random
from collections import deque
from tqdm import tqdm

from criterion import Criterion
from data import *
from dataset import *
from test import initilize_testing_loader, evaluate
from utils import parse_config, set_seeds, set_devices, load_pretrained_weight

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--trial', default='modelnet40.gcnn.0001', type=str,
                    help='The trial name with its version, e.g., modelnet40.gcnn.0001.')
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
parser.add_argument('--breakpoint_continuation', default=True, type=bool,
                    help='Resume the training from the last checkpoint.')

def initilize_training_loader(args):
    pre_transform = T.Compose([*map(eval, args.pre_transforms)])
    transform = T.Compose([*map(eval, args.training.transforms)])
    dataset = eval(args.data_class)(root=f'data/{args.trial_dataset}',
                                    train=True,
                                    transform=transform,
                                    pre_transform=pre_transform,
                                    **args.data_class_kwargs)
    loader = DataLoader(dataset,
                        batch_size=args.training.batch_size,
                        shuffle=True,
                        drop_last=True,
                        pin_memory=True,
                        num_workers=args.training.num_workers)
    return loader

if __name__ == "__main__":
    args = parser.parse_args()
    parse_config(args)
    
    # Dynamically import the model class, Net.
    model_path = os.path.join(args.trial_dir, 'model')
    model_package = importlib.import_module(model_path.replace('/', '.'))

    # Set the environment.
    set_seeds(args.seed + args.breakpoint_continuation * random.randint(1, 9e3))
    device = set_devices(args.cuda_devices, args.separator_bar)

    # Load the dataset.
    training_loader = initilize_training_loader(args)
    testing_loader = initilize_testing_loader(args)

    # Set the model.
    assert len(args.cuda_devices) <= 1
    model = model_package.Net(training_loader.dataset.num_classes(args.task_type),
                              **args.net_arguments).to(device)
    load_pretrained_weight(args.pretrained_weight_path, model, device)
    optimizer = eval(f'optim.{args.training.optimizer}')(params=model.parameters(),
                                                         **args.training.optimizer_kwargs)
    if args.training.scheduler == 'LambdaLR':
        lr_lambda = eval(args.training.scheduler_kwargs['lr_lambda'])
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = eval(f'optim.lr_scheduler.{args.training.scheduler}')(
            optimizer,
            **args.training.scheduler_kwargs)
    criterion = Criterion(args.task_type,
                          args.criterion,
                          args.show_loss_details,
                          args.max_float_digits).to(device)

    # Training
    model.train()
    checkpoint_queue = deque(maxlen=args.num_saved_checkpoints)
    checkpoint_dir = os.path.join(args.trial_dir, 'checkpoints')
    last_checkpoint_path = ''
    training_record_list = []
    testing_record_list = []
    if args.breakpoint_continuation:
        last_checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, 'last_*.weight'))
        assert len(last_checkpoint_paths) <= 1
        if last_checkpoint_paths:
            last_checkpoint_path = last_checkpoint_paths[0]
            last_checkpoint = torch.load(last_checkpoint_path,
                                         map_location=device,
                                         pickle_module=dill)
            model.load_state_dict(last_checkpoint['model'].state_dict())
            optimizer.load_state_dict(last_checkpoint['optimizer'].state_dict())
            scheduler.load_state_dict(last_checkpoint['scheduler'].state_dict())
            training_record_list = last_checkpoint['training_record_list']
            testing_record_list = last_checkpoint['testing_record_list']
            print(f'Resume from checkpoint: {last_checkpoint_path}')
        
        checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, 'epoch_*.weight'))
        get_target_fn = lambda x : x.split('/')[-1].split('.weight')[0].split('_')[-1]
        checkpoint_paths.sort(key=lambda x : float(get_target_fn(x)))
        for checkpoint_path in checkpoint_paths:
            checkpoint_target = get_target_fn(checkpoint_path)
            if len(checkpoint_queue) >= args.num_saved_checkpoints:
                os.remove(checkpoint_queue[0]['path'])
            checkpoint_queue.append({
                'path' : checkpoint_path,
                'target' : checkpoint_target
            })
    
    for epoch in range(scheduler.state_dict()['last_epoch'], args.training.num_epoches):
        best_target = checkpoint_queue[-1]['target'] if checkpoint_queue else 0
        with tqdm(training_loader,
                  desc=f'Epoch [{epoch}/{args.training.num_epoches}] ' +
                       f'Best {args.target} {best_target}') as t:
            for i, data in enumerate(t):
                data = data.to(device)
                prediction = model(data)
                batch_results = criterion(prediction, data, training_loader.dataset)
                loss = batch_results['loss']
                del batch_results['loss']
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                t.set_postfix(lr=scheduler.get_last_lr()[0],
                              loss=f'%.{args.max_float_digits}f' % loss.item(),
                              **batch_results)
            
                if i == len(training_loader) - 1:
                    training_results = criterion.general_results
                    training_record_list.append(training_results)
                    criterion.reset()
                    t.set_postfix(**training_results)

        scheduler.step()
        testing_results = evaluate(args, testing_loader, model, device, criterion)
        testing_record_list.append(testing_results)
        criterion.reset()
        target = testing_results[args.target]
        target_name_without_space = args.target.replace(' ', '-')
        checkpoint_name = \
                f'epoch_{epoch}_{target_name_without_space}_{target}.weight'
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        if last_checkpoint_path:
            os.remove(last_checkpoint_path)
        last_checkpoint_path = os.path.join(checkpoint_dir, 'last_' + checkpoint_name)
        checkpoint = {
            'model' : model,
            'optimizer' : optimizer,
            'scheduler' : scheduler,
            'training_record_list' : training_record_list,
            'testing_record_list' : testing_record_list,
        }
        torch.save(checkpoint, last_checkpoint_path, pickle_module=dill)
        if not checkpoint_queue or \
                float(target) >= float(checkpoint_queue[-1]['target']):
            if len(checkpoint_queue) >= args.num_saved_checkpoints:
                os.remove(checkpoint_queue[0]['path'])
            checkpoint_queue.append({
                'path' : checkpoint_path,
                'target' : target
            })
            torch.save(checkpoint, checkpoint_path, pickle_module=dill)
    
    best_target = checkpoint_queue[-1]['target'] if checkpoint_queue else 0
    print(f'Finish training with the best {args.target} {best_target}')