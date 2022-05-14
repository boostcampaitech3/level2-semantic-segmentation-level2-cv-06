import os
import os.path as osp
import math
import yaml
import json
import random

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp

from easydict import EasyDict as edict
from statistics import mean

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import _LRScheduler


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def concat_config(args, config):
    config = edict(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['device'] = device
    
    model = config.model.name
    if config.model.name is None:
        model = config.model.encoder + '_' + config.model.decoder
    loss = config.loss.name
    lr = config.learning_rate
    epoch = config.epoch
    
    name = f"{model}_{loss}_e{epoch}_lr{lr}"
    
    dir_name = osp.join(os.getcwd(), name)
    if osp.exists(dir_name):
        idx = 0
        new_dir_name = osp.join(os.getcwd(), f"{name}_{idx}")
        while osp.exists(new_dir_name):
            idx += 1
            new_dir_name = osp.join(os.getcwd(),f"{name}_{idx}")
        dir_name = new_dir_name
    
    os.makedirs(dir_name)
    
    config['name'] = name
    config['save_dir'] = dir_name
    for key, value in args.__dict__.items():
        config[key] = value
    
    with open(osp.join(dir_name, 'config.json'), 'w') as f:
        json.dump(dict(config), f, indent=4)
    
    return config


def load_config(args):
    with open(args.config_dir, 'r') as f:
        config = yaml.safe_load(f)
    config = concat_config(args, config)
    
    return config

def load_json(args):
    with open(args.config_dir, 'r') as f:
        config = edict(json.loads(f.read()))
    
    for key, value in args.__dict__.items():
        config[key] = value
        
    return config

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def label_accuracy_score(hist):
    """
    Returns accuracy score evaluation result.
      - [acc]: overall accuracy
      - [acc_cls]: mean accuracy
      - [mean_iu]: mean IU
      - [fwavacc]: fwavacc
    """
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide="ignore", invalid="ignore"):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc, iu


def add_hist(hist, label_trues, label_preds, n_class):
    """
    stack hist(confusion matrix)
    """

    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    return hist

def get_metrics(output, mask):
    outputs = torch.tensor(output, dtype=torch.float32)
    with torch.no_grad():
        output_met = torch.argmax(F.softmax(outputs, dim=1), dim=1) - 1
        mask_met = mask - 1
        tp, fp, fn, tn = smp.metrics.get_stats(output_met, mask_met, mode='multiclass', num_classes=10, ignore_index=-1)
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise")
    
    del outputs
    return f1_score, recall, precision


# https://gaussian37.github.io/dl-pytorch-lr_scheduler/
class CustomCosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(
        self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1.0, last_epoch=-1
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CustomCosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [
                (self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [base_lr+(self.eta_max - base_lr) * (1+math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
