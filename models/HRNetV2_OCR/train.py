import os
import os.path as osp
import yaml
import argparse

from easydict import EasyDict as edict

from dataset import (
    ObejctAugDataLoader,
    CustomDataLoader, 
    CustomTransforms
)
from utils import load_config

from model import get_seg_model
from trainer import train

import torch


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
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--fold", type=bool, default=True)
    parser.add_argument('--config_dir', type=str, default='./config.yaml')
    parser.add_argument('--viz_interval', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=5)
    args = parser.parse_args()
    return args

def main(args):
    args = load_config(args)
    model_config = args.model.config_path
    save_dir = args.save_dir

    args.train.train_dataloader.batch_size = args.batch_size
    args.val.val_dataloader.batch_size = args.batch_size
    args.optimizer.lr = args.learning_rate

    device = args.device
    is_fold = args.fold

    device = torch.device(device)

    transforms = CustomTransforms
    if args.augmix.use_augmix:
        datamodule = ObejctAugDataLoader(cfg=args, transforms=transforms)
    else:
        datamodule = CustomDataLoader(
            cfg=args, transforms=transforms)

    train_loader = datamodule.train_dataloader()
    val_loader = None
    if is_fold:
        val_loader = datamodule.val_dataloader()

    with open(model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    with open(osp.join(save_dir, 'model_config.yaml'), 'w') as f:
        yaml.safe_dump(model_config, f)
        
    model_config = edict(model_config)
    model = get_seg_model(model_config)
    train(args, model, model_config, train_loader, val_loader)
    

if __name__ == "__main__":
    args = get_parser()
    main(args)
