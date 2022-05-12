from copyreg import pickle
import os
import os.path as osp
import json
import random
import cv2
import pickle

import numpy as np
import ttach as tta

from easydict import EasyDict as edict

from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader


import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


CLASSES = ['Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

def get_classname(class_id, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == class_id:
            return cats[i]['name']
    return "None"


class CustomDataset(Dataset):    
    def __init__(self, data_dir, annotation, transforms=None, mode="train"):
        super(CustomDataset, self).__init__()
        self.data_dir = data_dir
        self.coco = COCO(osp.join(data_dir, annotation))
        self.transforms = transforms
        self.mode = mode
    
    def __getitem__(self, index:int):
        image_id = self.coco.getImgIds(imgIds=index)
        image_info = self.coco.loadImgs(image_id)[0]
        file_name = image_info['file_name']
        
        image = cv2.imread(osp.join(self.data_dir, file_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.mode in ('train', 'val'):
            ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
            anns = self.coco.loadAnns(ann_ids)

            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)
            
            mask = np.zeros((image_info['height'], image_info['width']))
            anns = sorted(anns, key=lambda idx: idx['area'], reverse=True)
            
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = CLASSES.index(className)
                mask[self.coco.annToMask(anns[i]) == 1] = pixel_value
            mask = mask.astype(np.uint8)
            
            if self.transforms is not None:
                transformed = self.transforms(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
            return image, mask
    
        if self.mode == 'test':
            if self.transforms is not None:
                transformed = self.transforms(image=image)
                image = transformed['image']
            return image, image_info
        
    def __len__(self):
        return len(self.coco.getImgIds())
    
    
def collate_fn(batch):
    return tuple(zip(*batch))


class ObejctAugDataset(Dataset):
    def __init__(self, data_dir, annotation, transforms=None, augmix=None, mode="train"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms
        self.coco = COCO(osp.join(data_dir, annotation))
        self.mode = mode
        with open(osp.join(data_dir, augmix), 'rb') as fr:
            augmix = pickle.load(fr)
        self.augmix = augmix 
        
    def __getitem__(self, index: int):
        image_id = self.coco.getImgIds(imgIds=index)
        image_info = self.coco.loadImgs(image_id)[0]
        file_name = image_info['file_name']
        
        image = cv2.imread(osp.join(self.data_dir, file_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
            anns = self.coco.loadAnns(ann_ids)

            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            mask = np.zeros((image_info["height"], image_info["width"]))
            anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)

            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = CLASSES.index(className)
                mask[self.coco.annToMask(anns[i]) == 1] = pixel_value
            mask = mask.astype(np.uint8)
            
            if self.augmix:
                r = np.random.rand(1) 
                if r <= 0.5:
                    image, mask = self.augmix_search(image, mask)  
                    
            if self.transform is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
            
            return image, mask
        
        if self.mode == 'test':
            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed["image"]
            
            return image, image_info
    
    
    def __len__(self):
        return len(self.coco.getImgIds())

    
    def augmix_search(self, image, mask):
        tfms = A.Compose([
                    A.GridDistortion(p=0.3, distort_limit=[-0.01, 0.01]),
                    A.Rotate(limit=60, p=1.0),
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5)
              ])
        
        num = [3, 4, 5, 6, 9, 10]

        label = random.choice(num)
        idx = np.random.randint(len(self.augmix[label]))
        augmix_img = self.augmix[label][idx]
        augmix_mask = np.zeros((512, 512))

        augmix_mask[augmix_img[:, :, 0] != 0] = label
        transformed = tfms(image=augmix_img, mask=augmix_mask)
        augmix_img = transformed['image']
        augmix_mask = transformed['mask']
        image[augmix_img != 0] = augmix_img[augmix_img != 0]
        mask[augmix_mask != 0] = augmix_mask[augmix_mask != 0]

        return image, mask


class ObejctAugDataLoader:
    def __init__(self, cfg, transforms=None):
        super().__init__()
        cfg = edict(cfg)
        self.data_dir = cfg.data.data_dir
        self.train_data = cfg.train.train_data
        self.val_data = cfg.val.val_data
        self.test_data = cfg.test.test_data
        self.augmix = cfg.augmix.augmix_data
        self.transforms = transforms(cfg)
        self.config = cfg

    def train_dataloader(self):
        train_transforms = self.transforms.train_transforms()
        train_dataset = ObejctAugDataset(
            self.data_dir, self.train_data, transforms=train_transforms, augmix=self.augmix, mode='train')
        return DataLoader(train_dataset, collate_fn=collate_fn, **self.config.train.train_dataloader)

    def val_dataloader(self):
        val_transforms = self.transforms.val_transforms()
        val_dataset = ObejctAugDataset(
            self.data_dir, self.val_data, transforms=val_transforms, augmix=self.augmix, mode="val")
        return DataLoader(val_dataset, collate_fn=collate_fn, **self.config.val.val_dataloader)

    def test_dataloader(self):
        test_transforms = self.transforms.test_transforms()
        test_dataset = ObejctAugDataset(
            self.data_dir, self.test_data, transforms=test_transforms, augmix=self.augmix, mode="test")
        return DataLoader(test_dataset, collate_fn=collate_fn, **self.config.test.test_dataloader)


class CustomDataLoader:
    def __init__(self, cfg, transforms=None):
        super().__init__()
        cfg = edict(cfg)
        self.data_dir = cfg.data.data_dir
        self.train_data = cfg.train.train_data
        self.val_data = cfg.val.val_data
        self.test_data = cfg.test.test_data
        self.transforms = transforms(cfg)
        self.config = cfg

    def train_dataloader(self):
        train_transforms = self.transforms.train_transforms()
        train_dataset = CustomDataset(
            self.data_dir, self.train_data, transforms=train_transforms, mode='train')
        return DataLoader(train_dataset, collate_fn=collate_fn, **self.config.train.train_dataloader)

    def val_dataloader(self):
        val_transforms = self.transforms.val_transforms()
        val_dataset = CustomDataset(
            self.data_dir, self.val_data, transforms=val_transforms, mode="val")
        return DataLoader(val_dataset, collate_fn=collate_fn, **self.config.val.val_dataloader)

    def test_dataloader(self):
        test_transforms = self.transforms.test_transforms()
        test_dataset = CustomDataset(
            self.data_dir, self.test_data, transforms=test_transforms, mode="test")
        return DataLoader(test_dataset, collate_fn=collate_fn, **self.config.test.test_dataloader)
    
class CustomTransforms:
    def __init__(self, cfg, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        self.cfg = cfg
        self.width = 512 if cfg.width == None else cfg.width
        self.height = 512 if cfg.height == None else cfg.height
        self.mean = mean
        self.std = std
        

    def train_transforms(self):
        return  A.Compose([
            A.RandomSizedCrop(
                min_max_height=(256, 512),
                width=self.width, height=self.height,
                p=0.5),
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=0.2, sat_shift_limit= 0.2, val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, p=0.9),
                A.CLAHE(p=0.9),
                ],p=0.9),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf([
                A.Blur(blur_limit=7, p=0.5),
                A.GaussNoise(var_limit=(10, 50), p=0.9),
                ], p=0.9),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(p=1.0),
            ], p=1.0)

    def val_transforms(self):
        return A.Compose([
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(p=1.0),
            ], p=1.0)

    def test_transforms(self):
        return A.Compose([
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(p=1.0),
            ], p=1.0)

    def tta_transforms(self):
        return tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 180]),
            tta.Scale(scales=[0.5, 1, 1.5]),
            tta.Multiply(factors=[0.7, 0.9, 1]),
        ])