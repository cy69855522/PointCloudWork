import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

class Criterion(nn.Module):
    def __init__(self, task_type, metrics, keep_loss_details=False, max_float_digits=4):
        super().__init__()
        assert task_type in ('classification', 'segmentation')
        self.task_type = task_type
        self.metrics = metrics
        self.keep_loss_details = keep_loss_details
        self.max_float_digits = max_float_digits
        for metric in self.metrics:
            loss_class = metric['class']
            metric_kwargs = metric.get('kwargs', {})
            setattr(self,
                    loss_class,eval(loss_class)(reduction='none', **metric_kwargs))
        self.reset()

    def reset(self):
        # The last batch of the dataset may have different batch size.
        self.global_dict = {'total_loss' : 0,
                            'num_samples' : 0,
                            'num_batches' : 0}
        self.id_to_result = {}

        if self.task_type == 'classification':
            self.global_dict['num_correct'] = 0
        if self.task_type == 'segmentation':
            self.global_dict['num_correct'] = 0

    def forward(self, prediction, data, dataset=None):
        label = self.get_label(data)
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
        B = data.batch.max().item() + 1
        self.global_dict['num_samples'] += B
        if self.task_type == 'classification':
            # prediction[B, C]
            num_correct_in_batch = prediction.max(dim=1)[1].eq(label).sum().item()
            self.global_dict['num_correct'] += num_correct_in_batch
            return_dict['accuracy'] = num_correct_in_batch / B
        elif self.task_type == 'segmentation':
            # prediction[B, C, N]
            B, C, N = prediction.size()
            dev = prediction.device

            possible_seg_mask = torch.zeros(B, C)
            for i, y in enumerate(data.y):
                categories = dataset.categories
                seg_classes = dataset.cat_to_seg[categories[y.item()]]
                possible_seg_mask[i].scatter_(0, torch.tensor(seg_classes), 1)
            possible_seg_mask = possible_seg_mask.to(dev)
            masked_prediction = prediction * possible_seg_mask[..., None]
            predicted_label = masked_prediction.max(dim=1)[1]  # [B, N]
            num_correct_in_batch = predicted_label.eq(label).sum().item()
            self.global_dict['num_correct'] += num_correct_in_batch / N
            return_dict['accuracy'] = num_correct_in_batch / B / N

            y_to_ious = {y.item() : [] for y in data.y.unique()}
            # predicted_label_[N] label_[N] y_[1] id_[1]
            for predicted_label_, label_, y_, id_ in zip(predicted_label, label, data.y, data.id):
                category = dataset.categories[y_.item()]
                parts = dataset.cat_to_seg[category]
                part_ious = torch.ones(len(parts), device=dev)
                for i, part in enumerate(parts):
                    union = torch.sum((predicted_label_ == part) | (label_ == part))
                    if union > 0:
                        intersection = torch.sum((predicted_label_ == part) & (label_ == part))
                        part_ious[i] = intersection / union
                miou = part_ious.mean().item()
                y_to_ious[y_.item()].append(miou)
                accuracy = (predicted_label_ == label_).float().mean().item()
                self.id_to_result[id_.item()] = {
                    'miou' : miou,
                    'accuracy' : accuracy,
                    'category' : category,
                }

            instance_miou = sum([*y_to_ious.values()], [])
            instance_miou = sum(instance_miou) / len(instance_miou)
            return_dict['instance miou'] = instance_miou

        return self.limit_display_digits(return_dict)
    
    def get_label(self, data):
        if self.task_type == 'classification':
            return data.y  # [B]
        elif self.task_type == 'segmentation':
            B = data.batch.max().item() + 1
            return data.seg.view(B, -1)  # [B, N]

    @property
    def detailed_results(self):
        return_dict = {'id_to_result' : self.id_to_result}
        if self.task_type == 'classification':
            pass
        elif self.task_type == 'segmentation':
            category_to_miou = {}
            category_to_count = {}
            for data_result in self.id_to_result.values():
                data_miou = data_result['miou']
                data_category = data_result['category']
                if data_category not in category_to_miou:
                    category_to_miou[data_category] = []
                category_to_miou[data_category].append(data_miou)
            for k, v in category_to_miou.items():
                category_to_count[k] = len(v)
                category_to_miou[k] = sum(v) / len(v)
            return_dict['category_to_miou'] = category_to_miou
            return_dict['category_to_count'] = category_to_count
        return return_dict

    @property
    def general_results(self):
        metric_resuls = {}
        metric_resuls['mean loss'] = self.global_dict['total_loss'] / \
                self.global_dict['num_samples']
        if self.task_type == 'classification':
            metric_resuls['mean accuracy'] = self.global_dict['num_correct'] / \
                    self.global_dict['num_samples']
        if self.task_type == 'segmentation':
            metric_resuls['mean accuracy'] = self.global_dict['num_correct'] / \
                    self.global_dict['num_samples']
            
            total_iou = 0
            category_to_miou = {}
            for data_result in self.id_to_result.values():
                data_miou = data_result['miou']
                data_category = data_result['category']
                total_iou += data_miou
                if data_category not in category_to_miou:
                    category_to_miou[data_category] = []
                category_to_miou[data_category].append(data_miou)
            for k, v in category_to_miou.items():
                category_to_miou[k] = sum(v) / len(v)
            class_ious = category_to_miou.values()
            metric_resuls['instance miou'] = total_iou / self.global_dict['num_samples']
            metric_resuls['class miou'] = sum(class_ious) / len(class_ious)

        return self.limit_display_digits(metric_resuls)
    
    def limit_display_digits(self, in_dict):
        out_dict = {}
        for k, v in in_dict.items():
            if isinstance(v, float):
                out_dict[k] = f'%.{self.max_float_digits}f' % v
            else:
                out_dict[k] = v
        return out_dict
