import argparse
import os
import os.path as osp
import gc

import numpy as np
import wandb

import torch
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm
from utils import (
    DiceLoss,
    set_seed,
    load_config,
    add_hist,
    label_accuracy_score
)

from model import get_seg_model
from optimizer import get_optimizer
from scheduler import get_scheduler
from dataset import (
    CustomDataLoader, 
    ObejctAugDataLoader,
    CustomTransforms
)

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
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-7)
    parser.add_argument('--n_skip', type=int, default=3)
    parser.add_argument('--vit_patches_size', type=int, default=16)
    parser.add_argument("--fold", type=bool, default=True)
    parser.add_argument('--config_dir', type=str, default='config.yaml')
    parser.add_argument('--viz_interval', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=5)
    args = parser.parse_args()
    return args


def main(args):
    set_seed(args.seed)
    args = load_config(args)
    save_dir = args.save_dir
    base_lr = args.learning_rate

    args.train.train_dataloader.batch_size = args.batch_size
    args.val.val_dataloader.batch_size = args.batch_size
    args.optimizer.lr = base_lr

    device = args.device
    is_fold = args.fold
    num_classes = args.num_classes

    gc.collect()
    torch.cuda.empty_cache()

    wandb.init(project="project", entity="entity", name=args.name)
    wandb.run.name = (args.name)
    wandb.config = {
        "learning_rate": base_lr,
        "epochs": args.epoch,
        "batch_size": args.batch_size
    }
    wandb.config.update(args)
    
    device = torch.device(device)

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
    model = model.to(device)
    wandb.watch(model, log='all')

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = get_optimizer(args.optimizer, model)
    scheduler = get_scheduler(args.scheduler, optimizer)

    best_mIoU = 0.0
    best_val_loss = 9999.0
    for epoch in range(1, args.epoch + 1):
        model.train()
        train_loss, train_miou_score, train_accuracy = 0, 0, 0

        hist = np.zeros((num_classes, num_classes))
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch{epoch} : Train")
        for step, data in enumerate(pbar):
            images, masks = data
            images = torch.stack(images).float().to(device)
            masks = torch.stack(masks).long().to(device)
            outputs = model(images)

            loss_ce = ce_loss(outputs, masks)
            loss_dice = dice_loss(outputs, masks, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            masks = masks.detach().cpu().numpy()

            hist = add_hist(hist, masks, outputs, n_class=num_classes)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            train_miou_score += mIoU
            train_accuracy += acc

            pbar.set_postfix(
                Train_Loss=f" {train_loss / (step + 1):.4f}",
                Train_IoU=f" {train_miou_score / (step + 1):.4f}",
                Train_Acc=f" {train_accuracy / (step + 1):.4f}",
            )

            if (step + 1) % args.log_interval == 0:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_loss / (step + 1),
                    'train/miou_score': train_miou_score / (step + 1),
                    'train/pixel_accuracy': train_accuracy / (step + 1),
                    "learning rate": optimizer.param_groups[0]['lr'],
                })

            if (step + 1) % args.viz_interval == 0:
                wandb.log(
                    {
                        "train_image": wandb.Image(
                            images[0, :, :, :],
                            masks={
                                "predictions": {
                                    "mask_data": outputs[0, :, :],
                                    "class_labels": CLASSES,
                                },
                                "ground_truth": {
                                    "mask_data": masks[0, :, :],
                                    "class_labels": CLASSES,
                                },
                            },
                        )
                    }
                )

        if val_loader is None:
            avrg_loss = train_loss / len(train_loader)

            log = {
                'epoch': epoch,
                "train/mIoU": train_miou_score / len(train_loader),
                "train/loss": avrg_loss,
                "train/accuracy": train_accuracy / len(train_loader),
            }
            IoU_by_classes = [
                {classes: round(iou, 4)} for iou, classes in zip(IoU, CLASSES.values())
            ]
            for IoU_by_class in IoU_by_classes:
                for key, value in IoU_by_class.items():
                    log[f"train/{key}_IoU"] = value
                
            wandb.log(log)
            if mIoU > best_mIoU:
                print(f"Best Train mIoU : {round(mIoU,4)}")
                print(f"Save model in {save_dir}")
                torch.save(
                    model.state_dict(),
                f"{save_dir}/{args.name}.pth",
                )
                best_mIoU = mIoU
            if epoch == args.epoch:
                torch.save(
                    model.state_dict(),
                    f"{save_dir}/latest.pth",
                )
            print(
                f"Train #{epoch}  Average Loss: {round(avrg_loss, 4)}, Accuracy : {round(acc, 4)}, mIoU: {round(mIoU, 4)}"
            )
            print(f"IoU by class : {IoU_by_classes}")
            continue

        val_loss, val_miou_score, val_accuracy = 0, 0, 0
        val_pbar = tqdm(val_loader, total=len(val_loader), desc=f"Epoch{epoch} : Val")

        with torch.no_grad():
            model.eval()
            hist = np.zeros((num_classes, num_classes))
            for step, data in enumerate(val_pbar):
                images, masks = data
                images = torch.stack(images).float().to(device)
                masks = torch.stack(masks).long().to(device)

                outputs = model(images)

                loss_ce = ce_loss(outputs, masks)
                loss_dice = dice_loss(outputs, masks, softmax=True)
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                
                val_loss += loss.item()
                
                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()

                masks = masks.detach().cpu().numpy()

                hist = add_hist(hist, masks, outputs, n_class=num_classes)
                acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
                val_miou_score += mIoU
                val_accuracy += acc

                val_pbar.set_postfix(
                    Val_Loss=f" {val_loss / (step + 1):.4f}",
                    Val_IoU=f" {val_miou_score / (step + 1):.4f}",
                    Val_Acc=f" {val_accuracy / (step + 1):.4f}",
                )

                if (step + 1) % args.log_interval == 0:
                    wandb.log({
                    'epoch': epoch,
                    'val/loss': val_loss / (step + 1),
                    'val/miou_score': val_miou_score / (step + 1),
                    'val/pixel_accuracy': val_accuracy / (step + 1),
                })

                if (step + 1) % args.viz_interval == 0:
                    wandb.log(
                        {
                            "valid_image": wandb.Image(
                                images[0, :, :, :],
                                masks={
                                    "predictions": {
                                        "mask_data": outputs[0, :, :],
                                        "class_labels": CLASSES,
                                    },
                                    "ground_truth": {
                                        "mask_data": masks[0, :, :],
                                        "class_labels": CLASSES,
                                    },
                                },
                            )
                        }
                    )

            IoU_by_classes = [
                {classes: round(iou, 4)} for iou, classes in zip(IoU, CLASSES.values())
            ]
            avrg_loss = val_loss / len(val_loader)
            best_val_loss = min(avrg_loss, best_val_loss)

            log = {
                'epoch': epoch,
                "val/mIoU": val_miou_score / len(val_loader),
                "val/loss": avrg_loss,
                "val/accuracy": val_accuracy / len(val_loader),
            }
            
            for IoU_by_class in IoU_by_classes:
                for key, value in IoU_by_class.items():
                    log[f"val/{key}_IoU"] = value
                
            wandb.log(log)
            if mIoU > best_mIoU:
                print(f"Best val mIoU : {round(mIoU,4)}")
                print(f"Save model in {save_dir}")
                torch.save(
                    model.state_dict(),
                f"{save_dir}/{args.name}.pth",
                )
                best_mIoU = mIoU
            if epoch == args.epoch:
                torch.save(
                    model.state_dict(),
                    f"{save_dir}/latest.pth",
                )
            print(
                f"Validation #{epoch}  Average Loss: {round(avrg_loss, 4)}, Accuracy : {round(acc, 4)}, mIoU: {round(mIoU, 4)}"
            )
            print(f"IoU by class : {IoU_by_classes}")

if __name__ == "__main__":
    args = get_parser()
    main(args)