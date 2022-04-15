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
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from mpl_toolkits.mplot3d import Axes3D
from test import initilize_testing_loader, evaluate
from utils import parse_config, set_seeds, set_devices, load_pretrained_weight, load_checkpoint

parser = argparse.ArgumentParser(description='Visualizing')
parser.add_argument('--trial', default='shapenet_with_norm.rscnn.0001', type=str,
                    help='The trial name with its version, e.g., modelnet40.dgcnn.0001.')
parser.add_argument('--base_trial', default='', type=str,
                    help='The trial used to be compared.')
parser.add_argument('--show_configs', default=False, type=bool,
                    help='Whether to print the config at the program beginning.')
parser.add_argument('--separator_bar', default='*' * 100, type=str,
                    help='Separator bar.')
parser.add_argument('--max_float_digits', default=4, type=int,
                    help='Limit the max digits to display.')
parser.add_argument('--pretrained_weight_path', default='', type=str,
                    help='The pre-trained weight path.')
parser.add_argument('--checkpoint_path', default='best', type=str,
                    help='Checkpoint path. "best" means the best checkpoint of saved ones.')
parser.add_argument('--data_ids', default=[0], nargs='+', type=int,
                    help='Data id list used for visualization, e.g., 10 20.')

def initialize_model(args, device):
    # Dynamically import the model class, Net.
    model_path = os.path.join(args.trial_dir, 'model')
    model_package = importlib.import_module(model_path.replace('/', '.'))

    # Set the model.
    model = model_package.Net(loader.dataset.num_classes(args.task_type),
                                **args.net_arguments).to(device)
    load_pretrained_weight(args.pretrained_weight_path, model, device)
    load_checkpoint(args.trial_dir, args.checkpoint_path, model, device)
    model.eval()
    return model

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

    # Initialize the models.
    new_model = initialize_model(new_args, device)
    if base_args.trial:
        base_model = initialize_model(base_args, device)

    # Plot the pictures.
    dataset = loader.dataset
    # Color Set https://zhuanlan.zhihu.com/p/114420786?ivk_sa=1024320u
    num_colors = 10
    with torch.no_grad():
        for data_id in new_args.data_ids:
            data = dataset[data_id].to(device)
            data.batch = torch.zeros_like(data.pos[:, 0]).long()
            C = dataset.num_classes(new_args.task_type)

            if new_args.task_type == 'segmentation':
                categories = dataset.categories
                seg_classes = dataset.cat_to_seg[categories[data.y.item()]]
                assert len(seg_classes) <= num_colors
                possible_seg_mask = torch.zeros(C)
                possible_seg_mask.scatter_(0, torch.tensor(seg_classes), 1)
                possible_seg_mask = possible_seg_mask[..., None].to(device)  # [C, 1]

                new_prediction = (possible_seg_mask * new_model(data)[0]).max(dim=0)[1]
                if base_args.trial:
                    base_prediction = (possible_seg_mask * base_model(data)[0]).max(dim=0)[1]
                else:
                    base_prediction = new_prediction
                new_true = new_prediction == data.seg
                new_accuracy = new_true.float().mean().item()
                base_true = base_prediction == data.seg
                base_accuracy = base_true.float().mean().item()
                diff_prediction = torch.ones_like(new_prediction) * 7
                diff_prediction[new_true & ~base_true] = 2
                diff_prediction[~new_true & base_true] = 3
                
                new_prediction = new_prediction.cpu()
                base_prediction = base_prediction.cpu()
                diff_prediction = diff_prediction.cpu()
                new_true = (-new_true.int() + 3).cpu()
                base_true = (-base_true.int() + 3).cpu()
                ground_truth = data.seg.cpu()
                pos = data.pos.t().cpu()

                fig = plt.figure(figsize=(6, 6))
                ax = Axes3D(fig, auto_add_to_figure=False)
                fig.add_axes(ax)
                color = plt.cm.tab10(ground_truth % num_colors)
                ax.scatter3D(*pos, c=color, marker='.', s=30)
                plt.title('666')
                ax.set_title('Scatter Plot')
                print(f'Data id: {data_id},',
                      f'New accuracy: {new_accuracy},',
                      f'Base accuracy: {base_accuracy}')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                plt.axis(True)

                rax = plt.axes([0, 0.75, 0.25, 0.25])
                buttons = ('ground_truth',
                           'new_prediction',
                           'new_true',
                           'base_prediction',
                           'base_true',
                           'diff_prediction')
                radio2 = RadioButtons(rax, buttons)
                def colorfunc(label):
                    ax.clear()
                    color = plt.cm.tab10(eval(label) % num_colors)
                    ax.scatter3D(*pos, c=color, marker='.', s=30)
                    plt.draw()
                radio2.on_clicked(colorfunc)

                plt.show()

