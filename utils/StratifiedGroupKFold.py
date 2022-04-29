import os
import json 
import argparse
import numpy as np 
import pandas as pd
import random

from tqdm import tqdm
from copy import deepcopy
from sklearn.model_selection import StratifiedGroupKFold 

def update_ids(data):
    new_dict = dict()

    keys = data.keys()
    for key in keys:
        if key in ['images', 'annotations']:
            new_dict[key] = []
        else:
            new_dict[key] = data[key]

    images = deepcopy(data['images'])
    annotations = deepcopy(data['annotations'])

    id2id = dict()

    for img_idx in range(len(images)):
        id2id[images[img_idx]['id']] = images[img_idx]['id']
        images[img_idx]['id'] = img_idx
        new_dict['images'].append(images[img_idx])
    
    for ann_idx in range(len(annotations)):
        annotations[ann_idx]['image_id'] = id2id[annotations[ann_idx]['image_id']]
        new_dict['annotations'].append(annotations[ann_idx])

    return new_dict

def main(args):
    path = args.path
    ann_file = args.ann_file
    data_dir = args.data_dir
    n_split = args.n_split
    seed = args.seed
    
    random.seed(seed)
    
    save_base_name = ann_file.split('.')[0]
    if '_' in ann_file:
        save_base_name = save_base_name.split('_')[-1]
    
    save_path = os.path.join(data_dir, path)
    annotation_path = os.path.join(data_dir, ann_file)

    with open(annotation_path, 'r') as f:
        train_json = json.loads(f.read())
        images = train_json['images']
        annotations = train_json['annotations']
    
    ann_df = pd.DataFrame.from_dict(annotations)

    keys = train_json.keys()

    var = [(ann['image_id'], ann['category_id']) for ann in annotations]
    X = np.ones((len(annotations), 1))
    y = np.array([v[1] for v in var])
    groups = np.array([v[0] for v in var])

    cv = StratifiedGroupKFold(n_splits=n_split, shuffle=True, random_state=seed) 

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with tqdm(total=n_split) as pbar:
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)): 
            train_dict = dict()
            val_dict = dict()

            for key in keys:
                if key in ['images', 'annotations']:
                    train_dict[key] = []
                    val_dict[key] = []
                else:
                    train_dict[key] = train_json[key]
                    val_dict[key] = train_json[key]
            
            train_idx = list(set(groups[train_idx]))
            val_idx = list(set(groups[val_idx]))

            train_idx.sort()
            val_idx.sort()

            train_dict['images'] = np.array(images)[train_idx].tolist()
            val_dict['images'] = np.array(images)[val_idx].tolist()
            
            for annotation in annotations:
                img_id = annotation['image_id']
                if img_id in train_idx:
                    train_dict['annotations'].append(annotation)
                else:
                    val_dict['annotations'].append(annotation)
            
            train_dict = update_ids(train_dict)
            val_dict = update_ids(val_dict)

            with open(os.path.join(save_path, f"train_{save_base_name}_{fold+1}.json"), 'w') as train_file:
                json.dump(train_dict, train_file, indent=4)

            with open(os.path.join(save_path, f"val_{save_base_name}_{fold+1}.json"), 'w') as val_file:
                json.dump(val_dict, val_file, indent=4)

            pbar.update(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', '-d', type=str, default='/opt/ml/input/data',
        help='data directory'
    )
    parser.add_argument(
        '--ann_file', '-a', type=str, default='train_paper.json',
        help='annotation file'
    )
    parser.add_argument(
        '--n_split', '-n', type=int, default=5,
    )
    parser.add_argument(
        '--path', '-p', type=str, default='paper'
    )
    parser.add_argument(
        '--seed', '-s', type=int, default=42
    )
    args = parser.parse_args()
    main(args)
    print("Done")
