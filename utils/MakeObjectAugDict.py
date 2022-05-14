import os
import os.path as osp
import cv2
import json
import argparse
import pickle

import numpy as np
import pandas as pd

import albumentations as A

from pycocotools.coco import COCO

CLASSES = ['Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/opt/ml/input/data')
    parser.add_argument('--data_path', type=str, default='ObjAug.json')
    parser.add_argument('--save_dir', type=str, default='/opt/ml/input/data')
    args = parser.parse_args()
    return args


def get_classname(class_id, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == class_id:
            return cats[i]['name']
    return "None"


def get_transform():
    tfms_to_small = A.Compose([
        A.Resize(256, 256),
        A.PadIfNeeded(512, 512, border_mode=0),
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=0.5),
        A.Resize(512, 512)
    ])
    tfms_to_big = A.Compose([
        # A.CropNonEmptyMaskIfExists(256, 256, ignore_values=[0]),
        A.CropNonEmptyMaskIfExists(384, 384, ignore_values=[0]),
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=1.0),
        A.Resize(512, 512)
    ])
    tfms = A.Compose([
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=0.5),
        A.Resize(512, 512)
    ])
    
    return tfms_to_small, tfms_to_big, tfms

def main(args):
    data_dir = args.data_dir
    data_path = args.data_path
    save_dir = args.save_dir

    classdict = {
        # 1: [], # 1: 'General trash'
        3: [], # 3: 'Paper pack'
        4: [], # 4: 'Metal'
        5: [], # 5: 'Glass'
        6: [], # 6: 'Plastic'
        9: [], # 9: 'Battery'
        10: [], # 10: 'Clothing'
    }

    with open(osp.join(data_dir, data_path), 'r') as f:
        data = json.loads(f.read())
        nums = len(data['images'])

    coco = COCO(osp.join(data_dir, data_path))

    for index in range(nums):
        image_id = coco.getImgIds(imgIds=index)
        image_info = coco.loadImgs(image_id)[0]

        image = cv2.imread(osp.join(data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
        
        ann_ids = coco.getAnnIds(imgIds=image_info['id'])
        anns = coco.loadAnns(ann_ids)

        cat_ids = coco.getCatIds()
        cats = coco.loadCats(cat_ids)

        tfms_to_small, tfms_to_big, tfms = get_transform()

        for ann in anns:
            size = ann['size']
            mask = np.zeros((image_info["height"], image_info["width"]))
            className = get_classname(ann['category_id'], cats)
            pixel_value = CLASSES.index(className)
            mask[coco.annToMask(ann) == 1] = pixel_value
            mask = mask.astype(np.int8)
            mask3d = np.dstack([mask]*3)
            # object 제외한 나머지를 image로 채움
            result = np.where(mask3d, 0, image)
            # object를 image로 채움
            result_mask = cv2.bitwise_and(image, image, mask=mask)

            if size == 'S':
                transformed = tfms_to_big(image=result_mask, mask=result_mask)
            elif size == 'M':
                transformed = tfms(image=result_mask, mask=result_mask)
            else:
                transformed = tfms_to_small(image=result_mask, mask=result_mask)


            mask = transformed['mask']
            tfms_image = transformed['image']

            if np.sum(tfms_image != 0) // 3 > 400:
                classdict[ann['category_id']].append(tfms_image)

   
    with open(osp.join(save_dir, 'classdict.pickle'),'wb') as fw:
        pickle.dump(classdict, fw) 

    print("Done")
    
    
if __name__ == "__main__":
    args = get_parser()
    main(args)
