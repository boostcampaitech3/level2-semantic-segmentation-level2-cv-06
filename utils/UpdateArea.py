import os
import os.path as osp
import json 
import argparse

from tqdm import tqdm
from copy import deepcopy


def main(args):
    ann_file = args.ann_file
    data_dir = args.data_dir
    save_path = args.save_path
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    annotation_path = os.path.join(data_dir, ann_file)

    with open(annotation_path, 'r') as f:
        train_json = json.loads(f.read())
        annotations = deepcopy(train_json['annotations'])
    
    new_dict = dict()

    for key in ['info', 'licenses', 'images', 'categories', 'annotations']:
        if key in ['annotations']:
            new_dict[key] = []
        else:
            new_dict[key] = train_json[key]

    anns = sorted(annotations, key=lambda x : x['image_id'])
    
    with tqdm(total=len(annotations)) as pbar:
        for idx, ann in enumerate(anns):
            segs = ann['segmentation']
            area = 0
            for seg in segs:
                area += len(seg)
            anns[idx]['area'] = area

    new_dict['annotations'] = anns

    with open(osp.join(save_path, ann_file), 'w') as f:
        json.dump(new_dict, f, indent=4)


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
        '--save_path', type=str, default='/opt/ml/input/data/paper'
    )
    args = parser.parse_args()
    main(args)
    print("Done")
