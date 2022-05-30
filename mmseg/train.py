#!/usr/bin/env python
# coding: utf-8

import os 
from mmcv import Config
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import torch
import gc


# classes = ("Background", "General trash", "Paper", "Paper pack", "Metal", "Glass", 
#            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

cfg = Config.fromfile('/opt/ml/input/code/mmsegmentation/configs/_boostcamp_/_base_/upernet-swin-l-aug.py')



# root='../../dataset/'
root = '/opt/ml/input/data/mmseg'
# cfg.data.train.data_root = root
# cfg.data.val.data_root = root
# cfg.data.test.data_root = '/opt/ml/input/data/test.json'
# cfg.dataset_type = 'CustomDataset'
# train
# dataset config 수정
# cfg.data.train.classes = classes
# cfg.data.train.img_prefix = root
# cfg.data.train.ann_file = root + 'train.json' # train json 정보
cfg.data.train.img_dir = root + '/images/training'
cfg.data.train.ann_dir = root + '/annotations/training'
# cfg.data.train.ann_file = root + 'StratifiedGroupKFold/train_0.json' # train json 정보
# cfg.data.train.pipeline[2]['img_scale'] = (512,512) # Resize
# val
# cfg.data.val.classes = classes
# cfg.data.val.img_prefix = root
# cfg.data.val.ann_file = root + 'StratifiedGroupKFold/val_0.json'
cfg.data.val.img_dir = root + '/images/validation'
cfg.data.val.ann_dir = root + '/annotations/validation'
# cfg.data.val.pipeline[1]['img_scale'] = (512,512)
# test
# cfg.data.test.classes = classes
# cfg.data.test.img_prefix = root
cfg.data.test.ann_dir = '/opt/ml/input/data/test.json' # test json 정보
# cfg.data.test.ann_file = root + 'StratifiedGroupKFold/val_0.json' # test json 정보
# cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize
cfg.data.samples_per_gpu = 4
cfg.seed = 2022
cfg.gpu_ids = [0]
cfg.work_dir = './work_dirs/upernet-swin-l-aug-focal'
cfg.model.decode_head.num_classes = 11
cfg.model.auxiliary_head.num_classes = 11 #segformer에서는 없음
cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
print(f'Config:\n{cfg.pretty_text}')



datasets = [build_dataset(cfg.data.train)]


# 모델 build 및 pretrained network 불러오기
model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')) 
# model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.init_weights()

gc.collect()
torch.cuda.empty_cache()
            
# 모델 학습
train_segmentor(model, datasets[0], cfg, distributed=False, validate=True)

