import os
import os.path as osp
import json
import argparse
import numpy as np
import pandas as pd

from collections import Counter
from copy import deepcopy

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/opt/ml/input/data')
    parser.add_argument('--data_path', type=str, default='train_paper.json')
    parser.add_argument('--save_dir', type=str, default='/opt/ml/input/data')
    args = parser.parse_args()
    return args

def update_ids(data):
    new_dict = dict()

    for key in ['info', 'licenses', 'images', 'categories', 'annotations']:
        if key in ['images', 'annotations']:
            new_dict[key] = []
        else:
            new_dict[key] = data[key]

    images = deepcopy(data['images'])
    imgs = sorted(images, key=lambda x : x['id'])
    annotations = deepcopy(data['annotations'])
    anns = sorted(annotations, key=lambda x : x['image_id'])
    
    images_ids = np.unique([img['id'] for img in imgs])
    anns = [ann for ann in anns if ann['image_id'] in images_ids]

    id2id = dict()

    for idx, img in enumerate(imgs):
        id2id[img['id']] = idx
        img['id'] = idx
        new_dict['images'].append(img)
    
    for idx, ann in enumerate(anns):
        ann['image_id'] = id2id[ann['image_id']]
        ann['id'] = idx
        new_dict['annotations'].append(ann)

    return new_dict

def check_size(segs):
    cnt = 0
    for seg in segs:
        cnt += len(seg)
    
    if cnt <= 400:
        return 'XS'
    elif 400 < cnt <= 2500:
        return 'S'
    elif 2500 < cnt <= 10000:
        return 'M'
    else:
        return 'L'
    
def use_annotations(size, cat_id):
    if cat_id not in [3, 4, 5, 6, 9, 10]:
        return False

    if size == 'XS':
        return False
    return True

def main(args):
    data_dir = args.data_dir
    data_path = args.data_path
    save_dir = args.save_dir
    
    new_dict = dict()

    with open(osp.join(data_dir, data_path), 'r') as f:
        data = json.loads(f.read())
        images = data['images']
        annotations = data['annotations']

    for key in data.keys():
        if key in ['images', 'annotations']:
            new_dict[key] = []
        else:
            new_dict[key] = data[key]

    one_object_images = Counter([annotation['image_id'] for annotation in annotations])
    one_object_images_ids = [key for key, value in one_object_images.items() if value == 1]
    
    df_anns = pd.DataFrame.from_dict(annotations)
    df_anns = df_anns[df_anns['image_id'].isin(one_object_images_ids)]

    df_imgs = pd.DataFrame.from_dict(images)
    df_imgs = df_imgs[df_imgs['id'].isin(one_object_images_ids)]
    
    df_anns['size'] = df_anns['segmentation'].apply(lambda x : check_size(x))
    df_anns['use_anno'] = df_anns.apply(lambda x : use_annotations(x['size'], x['category_id']), axis=1)

    df = df_anns[df_anns['use_anno'] == True] 
    image_ids = sorted(list(set(df['image_id'])))

    df_imgs = df_imgs[df_imgs['id'].isin(image_ids)]
    df.drop(columns=['use_anno'], axis=1, inplace=True)
    
    for idx, row in df_imgs.iterrows():
        new_dict['images'].append(dict(row))
        
    for idx, row in df.iterrows():
        new_dict['annotations'].append(dict(row))

    new_dict = update_ids(new_dict)

    with open(osp.join(save_dir, 'ObjAug.json'), 'w') as f:
        json.dump(new_dict, f, indent=4)
    
    print("Done")
    
    
if __name__ == "__main__":
    args = get_parser()
    main(args)
