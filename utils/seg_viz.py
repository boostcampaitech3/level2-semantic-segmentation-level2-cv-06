import os
import json
import argparse
import numpy as np
from collections import Counter
from pycocotools.coco import COCO


BASE_DIR = '/opt/ml/input/data'
CLASSES =  ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

class CUSTOMDATASET:
    def __init__(self, arg):
        self.anno_file = f'{BASE_DIR}/{arg.anno_file}'
        self.coco = COCO(self.anno_file)
        with open(self.anno_file) as f:
            self.train_ann = json.load(f)
    
    def print_cat(self):
        print(self.coco.getCatIds())
    
    def print_detail_cat(self):
        print(*self.coco.loadCats(self.coco.getCatIds()))

    def seg_counter(self):
        
        def id2objnum(ann):
            return len(self.coco.getAnnIds(imgIds=ann['id']))

        obj_num = [id2objnum(ann) for ann in self.train_ann['images']]
        counts = Counter(obj_num)

        for key, val in counts.items():
            print(f'{key} seg per image: {val}')
        
    def load_imgs(self, id):
        self.img = self.coco.loadImgs()[0]
        print(self.img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_file', '-a', type=str, default="train_total.json",
                        help='annotation Data directory')
    args = parser.parse_args()
    coco = CUSTOMDATASET(args)
    # coco.load_imgs(322)
    coco.seg_counter()
    # coco.print_detail_cat()