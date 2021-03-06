import os
import os.path as osp
import json
import argparse

def main(args):
    train_json = args.train_json
    concat_json = args.concat_json
    save_dir = args.save_dir

    with open(train_json, 'r') as f:
        train_data = json.loads(f.read())
        train_anns = train_data['annotations']
        train_imgs = train_data['images']

    with open(concat_json, 'r') as f:
        concat_data = json.loads(f.read())
        concat_anns = concat_data['annotations']
        concat_imgs = concat_data['images']

    new_json = dict()
    new_json['info'] = train_data['info'].copy()
    new_json['licenses'] = train_data['licenses'].copy()
    new_json['images'] = []
    new_json['categories'] = train_data['categories'].copy()
    new_json['annotations'] = []

    # before_sort_anns = [train_ann['id'] for train_ann in train_anns]
    # print(id(before_sort_anns))

    # train_anns = sorted(train_anns, key=lambda x : x['id'])
    # after_sort_anns = [train_ann['id'] for train_ann in train_anns]
    # print(id(after_sort_anns))

    # if after_sort_anns == before_sort_anns:
    #     print("train_anns are same order")

    # before_sort_imgs = [train_img['id'] for train_img in train_imgs]
    # print(id(before_sort_imgs))

    # concat_imgs = sorted(concat_imgs, key=lambda x : x['id'])
    # after_sort_imgs = [train_img['id'] for train_img in train_imgs]
    # print(id(after_sort_imgs))

    # if after_sort_imgs == before_sort_imgs:
    #     print("train_imgs are same order")

    # before_sort_imgs = [concat_img['id'] for concat_img in concat_imgs]
    # print(id(before_sort_imgs))

    # train_imgs = sorted(train_imgs, key=lambda x : x['id'])
    # after_sort_imgs = [concat_img['id'] for concat_img in concat_imgs]
    # print(id(after_sort_imgs))

    # if after_sort_imgs == before_sort_imgs:
    #     print("concat_imgs are same order")

    # before_sort_anns = [concat_ann['id'] for concat_ann in concat_anns]
    # print(id(before_sort_imgs))
    
    # concat_anns = sorted(concat_anns, key=lambda x : x['id'])
    # after_sort_anns = [concat_ann['id'] for concat_ann in concat_anns]
    # print(id(after_sort_imgs))

    # if after_sort_anns == before_sort_anns:
    #     print("concat_anns are same order")


    for idx, train_img in enumerate(train_imgs):
        file_name = train_img['file_name']
        img_id = train_img['id']
        if file_name.startswith('batch_03'):
            new_json['images'].extend(train_imgs[:idx])

            for train_ann in train_anns:
                image_id = train_ann['image_id']
                ann_id = train_ann['id']
                if image_id == img_id:
                    break
                else:
                    new_json['annotations'].append(train_ann)
            break
    
    # print(img_id, ann_id)

    for concat_img in concat_imgs:
        concat_img['id'] += img_id
        new_json['images'].append(concat_img)

    for concat_ann in concat_anns:
        concat_ann['image_id'] += img_id
        concat_ann['id'] += ann_id
        new_json['annotations'].append(concat_ann)


    with open(osp.join(save_dir, 'train_total.json'), 'w') as f:
        json.dump(new_json, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json', type=str, default="/opt/ml/input/data/train_all.json")
    parser.add_argument('--concat_json', type=str, default="/opt/ml/input/data/data.json")
    parser.add_argument('--save_dir', type=str, default='/opt/ml/input/data/')
    args = parser.parse_args()
    main(args)