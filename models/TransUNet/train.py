import argparse
import gc


import torch
from utils import load_config

from model import get_seg_model
from dataset import (
    CustomDataLoader, 
    ObejctAugDataLoader,
    CustomTransforms
)
from trainer import train

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
    parser.add_argument("--learning_rate", type=float, default=2e-7)
    parser.add_argument('--n_skip', type=int, default=3)
    parser.add_argument('--vit_patches_size', type=int, default=16)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument('--config_dir', type=str, default='config.yaml')
    parser.add_argument('--viz_interval', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=5)
    args = parser.parse_args()
    return args


def main(args):
    args = load_config(args)
    base_lr = args.learning_rate

    args.train.train_dataloader.batch_size = args.batch_size
    args.val.val_dataloader.batch_size = args.batch_size
    args.optimizer.lr = base_lr

    fold = args.fold
    is_fold = True if fold else False

    gc.collect()
    torch.cuda.empty_cache()

    transforms = CustomTransforms
    if args.augmix.use_augmix:
        datamodule = ObejctAugDataLoader(cfg=args, transforms=transforms)
    else:
        datamodule = CustomDataLoader(cfg=args, transforms=transforms)
    
    train_loader = datamodule.train_dataloader()
    val_loader = None
    if is_fold:
        val_loader = datamodule.val_dataloader()

    model = get_seg_model(args)
    train(args, model, train_loader, val_loader)
    

if __name__ == "__main__":
    args = get_parser()
    main(args)