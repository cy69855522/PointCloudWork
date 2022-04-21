import torch

import dill
import glob
import numpy as np
import os
import platform
import random
import yaml

def parse_config(args):
    trial_dataset, trial_name, trial_version = args.trial.split('.')
    config = {
        'trial_dataset' : trial_dataset,
        'trial_name' : trial_name,
        'trial_version' : trial_version,
        'trial_dir' : f'trial/{trial_dataset}/{trial_name}/{trial_version}/'
    }
    config_path = os.path.join(config['trial_dir'], 'config.yaml')
    with open(config_path, 'r') as f:
        file_content = f.read()
        config = {'trial' : config}
        config |= yaml.load(file_content, yaml.FullLoader)
        if args.show_configs:
            print('\n' + args.separator_bar + '\n')
            print(f'[{trial_dataset} - {trial_name}] <{trial_version}>\n')
            print(file_content)
            print('\n' + args.separator_bar + '\n')
    
    # Update the args in place.
    two_level_keys = {'training', 'testing'}
    for arg_key in config.keys() - two_level_keys:
        for k, v in config[arg_key].items():
            assert not hasattr(args, k), f'Find a duplicate argument, args.{k}.'
            setattr(args, k, v)
    for arg_key in two_level_keys:
        sub_args = args.__class__()
        setattr(args, arg_key, sub_args)
        for k, v in config[arg_key].items():
            assert not hasattr(sub_args, k), f'Find a duplicate argument, args.{arg_key}.{k}.'
            setattr(sub_args, k, v)

def print_dict(d, prefix='', separate_lines=True):
    for k, v in d.items():
        print(f'{prefix}{k}:', end='')
        if isinstance(v, dict):
            print()
            print_dict(v, f'  {prefix}', False)
        else:
            print(f' {v}')
        if separate_lines:
            print()

def set_seeds(seed, strict=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if strict:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def set_devices(cuda_devices, separator_bar):
    if cuda_devices:
        torch.cuda.set_device(cuda_devices[0])
        device = torch.device('cuda')
        print('Running on cuda device:', end=' ')
        for device_id in cuda_devices:
            print(torch.cuda.get_device_name(device_id), end=' ')
        print('\n\n' + separator_bar + '\n')
    else:
        device = torch.device('cpu')
        print(f'Running on cpu device: {platform.processor()}')
        print('\n' + separator_bar + '\n')
    return device

def load_pretrained_weight(pretrained_weight_path, model, device):
    if pretrained_weight_path:
        pretrained_weight = torch.load(pretrained_weight_path, map_location=device)
        model.load_state_dict(pretrained_weight, strict=True)
        print(f'Load the pretrained weight: {pretrained_weight_path}')

def load_checkpoint(trial_dir, checkpoint_path, model, device):
    checkpoint_dir = os.path.join(trial_dir, 'checkpoints')
    if checkpoint_path == 'best':
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
    elif checkpoint_path == 'last':
        checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, 'last_*.weight'))
        assert len(checkpoint_paths) == 1
        checkpoint_path = checkpoint_paths[0]
        checkpoint = torch.load(checkpoint_path,
                                map_location=device,
                                pickle_module=dill)
        model.load_state_dict(checkpoint['model'].state_dict())
        print(f'Load the checkpoint: {checkpoint_path}')
    elif checkpoint_path:
        checkpoint = torch.load(checkpoint_path,
                                map_location=device,
                                pickle_module=dill)
        model.load_state_dict(checkpoint['model'].state_dict())
        print(f'Load the checkpoint: {checkpoint_path}')
