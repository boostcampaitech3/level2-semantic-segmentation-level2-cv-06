import os
import os.path as osp
import gc
import yaml
import wandb

from tqdm import tqdm
from easydict import EasyDict as edict

import numpy as np

import torch
import torch.nn.functional as F

from optimizer import get_optimizer
from scheduler import get_scheduler
from loss import create_criterion
from utils import (
    set_seed,
    add_hist,
    label_accuracy_score
)

best_mIoU = 0.0

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

def train(args, model, model_config, train_loader, val_loader):
    set_seed(args.seed)
    device = args.device
    save_dir = args.save_dir
    num_classes = args.num_classes

    gc.collect()
    torch.cuda.empty_cache()

    wandb.init(project="Ho", entity="omakase", name=args.name)
    wandb.run.name = (args.name)
    wandb.config = {
        "learning_rate": args.learning_rate,
        "epochs": args.epoch,
        "batch_size": args.batch_size
    }
    wandb.config.update(args)

    model = model.to(device)
    wandb.watch(model, log='all')

    criterion = create_criterion(args.loss, model_config)
    optimizer = get_optimizer(args.optimizer, model)
    scheduler = get_scheduler(args.scheduler, optimizer)

    best_mIoU = 0.0
    avrg_loss = 9999.0
    for epoch in range(1, args.epoch+1):
        model.train()
        train_loss, train_miou_score, train_accuracy = 0, 0, 0

        hist = np.zeros((num_classes, num_classes))
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch{epoch} : Train")
        for step, data in enumerate(pbar):
            images, masks = data
            images = torch.stack(images).float().to(device)
            masks = torch.stack(masks).long().to(device)
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss = torch.unsqueeze(loss, 0)
            loss = loss.mean()  

            output = outputs[1]
            ph, pw = output.size(2), output.size(3)
            h, w = masks.size(1), masks.size(2)
            if ph != h or pw != w:
                output = F.interpolate(
                    input=output, size=(h, w), mode="bilinear", align_corners=True
                )

            outputs = output
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
          
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
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

        
        scheduler.step()
        if val_loader is None:
            avrg_loss = train_loss / len(train_loader)
            log = {
                'epoch': epoch,
                "train/mIoU": train_miou_score / len(train_loader),
                "train/loss": train_loss / len(train_loader),
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

        best_mIoU, avrg_loss = validation(args, model, val_loader, criterion, best_mIoU, avrg_loss)

        
def validation(args, model, val_loader, criterion, best_mIoU):
    epoch = args.epoch
    num_classes = args.num_classes
    device = args.device
    save_dir = args.save_dir

    best_val_loss = 9999.0
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

            loss = criterion(outputs, masks)
            loss = torch.unsqueeze(loss, 0)
            loss = loss.mean() 
            val_loss += loss.item()
            
            output = outputs[1]
            ph, pw = output.size(2), output.size(3)
            h, w = masks.size(1), masks.size(2)
            if ph != h or pw != w:
                output = F.interpolate(
                    input=output,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=True,
                )
            outputs = output
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
        val_mIoU = val_miou_score / len(val_loader)

        log = {
            'epoch': epoch,
            "val/mIoU": val_mIoU,
            "val/loss": avrg_loss,
            "val/accuracy": val_accuracy / len(val_loader),
        }
        
        for IoU_by_class in IoU_by_classes:
            for key, value in IoU_by_class.items():
                log[f"val/{key}_IoU"] = value
            
        wandb.log(log)
        if val_mIoU > best_mIoU:
            print(f"Best val mIoU : {round(val_mIoU,4)}")
            print(f"Save model in {save_dir}")
            torch.save(
                model.state_dict(),
            f"{save_dir}/{args.name}.pth",
            )
            best_mIoU = val_mIoU
        if epoch == args.epoch:
            torch.save(
                model.state_dict(),
                f"{save_dir}/latest.pth",
            )
        print(
            f"Validation #{epoch}  Average Loss: {round(avrg_loss, 4)}, Accuracy : {round(acc, 4)}, mIoU: {round(mIoU, 4)}"
        )
        print(f"IoU by class : {IoU_by_classes}")
    return best_mIoU, avrg_loss