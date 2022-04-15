import torch

import argparse
import dill
import glob
import importlib
import os
import pandas as pd
from tqdm import tqdm

from criterion import Criterion
from data import *
from dataset import *
from test import initilize_testing_loader, evaluate
from utils import parse_config, set_seeds, set_devices, load_pretrained_weight, load_checkpoint

parser = argparse.ArgumentParser(description='Visualizing')
parser.add_argument('--trial', default='shapenet_with_norm.rscnn.0001', type=str,
                    help='The trial name with its version, e.g., modelnet40.dgcnn.0001.')
parser.add_argument('--base_trial', default='', type=str,
                    help='The trial used to be compared.')
parser.add_argument('--show_configs', default=False, type=bool,
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

def run_trial(args, loader, device):
    # Dynamically import the model class, Net.
    model_path = os.path.join(args.trial_dir, 'model')
    model_package = importlib.import_module(model_path.replace('/', '.'))

    # Set the model.
    model = model_package.Net(loader.dataset.num_classes(args.task_type),
                                **args.net_arguments).to(device)
    load_pretrained_weight(args.pretrained_weight_path, model, device)
    load_checkpoint(args.trial_dir, args.checkpoint_path, model, device)

    criterion = Criterion(args.task_type,
                            args.criterion,
                            args.show_loss_details,
                            args.max_float_digits).to(device)

    criterion.reset()
    general_performance = evaluate(args, loader, model, device, criterion)
    detailed_performance = criterion.detailed_results
    return {
        'general' : general_performance,
        'detailed' : detailed_performance
    }

def generate_report(args, loader, trial_result):
    report = {
        'general' : trial_result['general'],
        'detailed' : {}
    }

    if args.task_type == 'segmentation':
        report['general'] = report['general'] | trial_result['detailed']['category_to_miou']
        total_count = sum(trial_result['detailed']['category_to_count'].values())
        for category, category_miou in trial_result['detailed']['category_to_miou'].items():
            category_weight = trial_result['detailed']['category_to_count'][category] / total_count
            category_weight_str = f'%.{new_args.max_float_digits}f' % category_weight
            # Harm to instance miou.
            report['general'][f'{category} harm ({category_weight_str})'] = \
                (1 - category_miou) * category_weight
        for data_id, data_result in trial_result['detailed']['id_to_result'].items():
            data_report = report['detailed'][data_id] = {}
            data_report['miou'] = data_result['miou']
            data_report['accuracy'] = data_result['accuracy']
            for category in loader.dataset.categories:
                data_report[category] = None
            data_report[data_result['category']] = data_result['miou']

    return report

if __name__ == "__main__":
    new_args = parser.parse_args()
    base_args = copy.deepcopy(new_args)
    parse_config(new_args)
    base_args.trial = base_args.base_trial
    if base_args.trial:
        parse_config(base_args)

    # Set the environment.
    pd.set_option('display.precision', new_args.max_float_digits)
    set_seeds(new_args.seed)
    device = set_devices(new_args.cuda_devices, new_args.separator_bar)

    # Load the dataset.
    if base_args.trial:
        assert base_args.task_type == new_args.task_type
        assert base_args.pre_transforms == new_args.pre_transforms
        assert base_args.testing.transforms == new_args.testing.transforms
        for k in base_args.data_class_kwargs.keys() | new_args.data_class_kwargs.keys():
            base_kwarg = base_args.data_class_kwargs.get(k, None)
            new_kwarg = new_args.data_class_kwargs.get(k, None)
            new_args.data_class_kwargs[k] = base_kwarg or new_args
    loader = initilize_testing_loader(new_args)

    # Run trials.
    new_result = run_trial(new_args, loader, device)
    new_report = generate_report(new_args, loader, new_result)
    general_comparison = {'new': new_report['general']}
    detailed_comparison = new_report['detailed']
    
    if base_args.trial:
        base_result = run_trial(base_args, loader, device)
        base_report = generate_report(base_args, loader, base_result)
        general_comparison['base'] = base_report['general']

        # Generate difference report
        general_comparison['diff.'] = {}
        for k, new_v in new_report['general'].items():
            general_comparison['diff.'][k] = float(new_v) - float(base_report['general'][k])
        
        detailed_comparison = {}
        for data_id in new_report['detailed']:
            data_details = detailed_comparison[data_id] = {}
            for k in new_report['detailed'][data_id]:
                new_v = new_report['detailed'][data_id][k]
                base_v = base_report['detailed'][data_id][k]
                data_details['new_' + k] = new_v
                data_details['base_' + k] = base_v
                data_details['diff_' + k] = new_v - base_v if new_v and base_v else None

    # Export the comparison report.
    export_path = f'comparison.xlsx'
    with pd.ExcelWriter(export_path) as excel_writer:
        float_format = f'%.{new_args.max_float_digits}f'
        pd.DataFrame(general_comparison).to_excel(excel_writer,
                                                  'General',
                                                  engine='xlsxwriter',
                                                  float_format=float_format)
        pd.DataFrame(detailed_comparison).T.to_excel(excel_writer,
                                                     'Detailed',
                                                     engine='xlsxwriter',
                                                     float_format=float_format)
        excel_writer.save()
        print(f'Export the comparison report to: {export_path}')
