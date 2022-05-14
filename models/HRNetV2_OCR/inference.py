import os
import os.path as osp
import yaml
import argparse

from easydict import EasyDict as edict

from dataset import (
    CustomDataLoader,
    CustomTransforms,
    ObejctAugDataLoader
)
from utils import (
    set_seed,
    load_json
)

from model import get_seg_model
from test import test

import torch
import torch.nn as nn
import torch.nn.functional as F

import albumentations as A

CLASSES = {
    0: "Backgroud",
    1: "General trash",
    2: "Paper",
    3: "Paper pack",
    4: "Metal",
    5: "Glass",
    6: "Plastic",
    7: "Styrofoam",
    8: "Plastic bag",
    9: "Battery",
    10: "Clothing",
}

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--config_dir', type=str, default='config_dir')
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--config_model', type=str, default='model_config.yaml')
    parser.add_argument('--checkpoint', type=str, default='model.pth')
    args = parser.parse_args()
    return args


def main(args):
    set_seed(args.seed)
    save_dir = args.config_dir
    args.config_dir = osp.join(save_dir, args.config)
    args = load_json(args)
    model_config = args.config_model
    checkpoint = args.checkpoint
    device = args.device

    with open(osp.join(save_dir, model_config), 'r') as f:
        model_config = yaml.safe_load(f)
    
    args.test.test_dataloader.batch_size = args.batch_size
    transforms = CustomTransforms
    if args.augmix.use_augmix:
        datamodule = ObejctAugDataLoader(cfg=args, transforms=transforms)
    else:
        datamodule = CustomDataLoader(cfg=args, transforms=transforms)
    test_loader = datamodule.test_dataloader()
    
    model_config = edict(model_config)
    model = get_seg_model(model_config)
    checkpoint = osp.join(save_dir, checkpoint)

    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint)

    test(args, model, test_loader, device)


if __name__ == "__main__":
    args = get_parser()
    main(args)