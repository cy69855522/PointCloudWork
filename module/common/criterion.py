import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

class Criterion(nn.Module):
    def __init__(self, task_type, metrics, keep_loss_details=False, max_float_digits=4):
        super().__init__()
        assert task_type in ('classification', )
        self.task_type = task_type
        self.metrics = metrics
        self.keep_loss_details = keep_loss_details
        self.max_float_digits = max_float_digits
        for metric in self.metrics:
            loss_class = metric['class']
            setattr(self,
                    loss_class,eval(loss_class)(reduction='none', **metric['kwargs']))
        self.reset()

    def forward(self, prediction, label):
        return_dict = {'loss' : 0}
        for metric in self.metrics:
            loss_class = metric['class']
            weight = metric['weight']
            loss = eval(f'self.{loss_class}')(prediction, label)
            loss_mean = loss.mean()
            return_dict['loss'] = return_dict['loss'] + weight * loss_mean
            if self.keep_loss_details:
                return_dict[loss_class] = loss_mean.item()
            self.global_dict['total_loss'] += weight * \
                    loss.view(loss.size(0), -1).mean(dim=1).sum().item()

        self.global_dict['num_batches'] += 1
        if self.task_type == 'classification':
            num_samples_in_batch = label.size(0)
            self.global_dict['num_samples'] += num_samples_in_batch
            num_correct_in_batch = prediction.max(dim=1)[1].eq(label).sum().item()
            self.global_dict['num_correct'] += num_correct_in_batch
            return_dict['accuracy'] = num_correct_in_batch / num_samples_in_batch
        
        return self.limit_display_digits(return_dict)
    
    def reset(self):
        # The last batch of the dataset may have different batch size.
        self.global_dict = {'total_loss' : 0,
                            'num_samples' : 0,
                            'num_batches' : 0}
        
        if self.task_type == 'classification':
            self.global_dict['num_correct'] = 0

    @property
    def global_metric_resuls(self):
        metric_resuls = {}
        if self.task_type == 'classification':
            metric_resuls['mean loss'] = self.global_dict['total_loss'] / \
                    self.global_dict['num_samples']
            metric_resuls['mean accuracy'] = self.global_dict['num_correct'] / \
                    self.global_dict['num_samples']

        return self.limit_display_digits(metric_resuls)
    
    def limit_display_digits(self, in_dict):
        out_dict = {}
        for k, v in in_dict.items():
            if isinstance(v, float):
                out_dict[k] = f'%.{self.max_float_digits}f' % v
            else:
                out_dict[k] = v
        return out_dict
