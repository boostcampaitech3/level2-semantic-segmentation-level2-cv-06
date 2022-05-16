import os
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from visualization_helper import VzHelper


ssh = dict()
img_idx = 0
img_list = []
categories = [None]
categories_idx = 0
annotation_idx = 0
vz_helper = None


def press(event):
    global ssh, img_idx, img_list, categories, categories_idx, annotation_idx
    print(f"pressed key: {event.key}")
    sys.stdout.flush()  # TODO should search what this is
    if event.key == "x":
        plt.close()
    elif event.key == "right":
        plt.clf()  # TODO should check what's difference between plt.cla() and plt.clf()
        img_idx += 1
        if img_idx == len(img_list):
            img_idx = 0
        target_path = os.path.join(ssh['identity_dir_path'], 'temp.jpg') if args.target_dir_path == "" else os.path.join(args.target_dir_path, 'temp.jpg')
        download_file(ssh, os.path.join(ssh["source_dir_path"], img_list[img_idx]['file_name']), target_path)
        img = mimg.imread(target_path)
        plt.imshow(img)
        categories_idx = 0
        annotation_idx = 0
        anns = vz_helper.get_annotations(img_list[img_idx]["id"])
        vz_helper.showAnns(anns)
        plt.axis("off")
        plt.draw()
        print("all categories")
    elif event.key == "left":
        plt.clf()  # TODO should check what's difference between plt.cla() and plt.clf()
        img_idx -= 1
        if img_idx < 0:
            img_idx = len(img_list) - 1
        target_path = os.path.join(ssh['identity_dir_path'], 'temp.jpg') if args.target_dir_path == "" else os.path.join(args.target_dir_path, 'temp.jpg')
        download_file(ssh, os.path.join(ssh["source_dir_path"], img_list[img_idx]['file_name']), target_path)
        img = mimg.imread(target_path)
        plt.imshow(img)
        categories_idx = 0
        annotation_idx = 0
        anns = vz_helper.get_annotations(img_list[img_idx]["id"])
        vz_helper.showAnns(anns)
        plt.axis("off")
        plt.draw()
        print("all categories")
    elif event.key == "up":
        plt.cla()
        target_path = os.path.join(ssh['identity_dir_path'], 'temp.jpg') if args.target_dir_path == "" else os.path.join(args.target_dir_path, 'temp.jpg')
        img = mimg.imread(target_path)
        plt.imshow(img)
        cat_id, cat_name = categories[categories_idx][0], categories[categories_idx][1]
        anns = vz_helper.get_annotation(img_list[img_idx]["id"], cat_id, annotation_idx)
        annotation_idx += 1
        vz_helper.showAnns(anns)
        plt.axis("off")
        plt.draw()
        print(f"annotation_id: {anns[0]['id']}")
    elif event.key == "down":
        pass        
    elif event.key == "tab":
        plt.cla()
        target_path = os.path.join(ssh['identity_dir_path'], 'temp.jpg') if args.target_dir_path == "" else os.path.join(args.target_dir_path, 'temp.jpg')
        img = mimg.imread(target_path)
        plt.imshow(img)
        plt.axis("off")
        categories_idx = (categories_idx + 1) % len(categories)
        if categories_idx == 0:
            anns = vz_helper.get_annotations(img_list[img_idx]["id"])
        else:
            cat_id, cat_name = categories[categories_idx][0], categories[categories_idx][1]
            anns = vz_helper.get_annotations(img_list[img_idx]["id"], cat_id)
        vz_helper.showAnns(anns)
        plt.draw()
        print(f"file_name :{img_list[img_idx]['file_name']}")
        print(f"category_name: {cat_name}" if categories_idx > 0 else "all categories")
        annotation_idx = 0
    elif event.key.startswith("ctrl"):
        cmd = event.key.split("+")[1]
        if not cmd.isnumeric():
            return
        plt.cla()
        target_path = os.path.join(ssh['identity_dir_path'], 'temp.jpg') if args.target_dir_path == "" else os.path.join(args.target_dir_path, 'temp.jpg')
        img = mimg.imread(target_path)
        plt.imshow(img)
        plt.axis("off")
        categories_idx = int(cmd)
        if categories_idx == 0:
            anns = vz_helper.get_annotations(img_list[img_idx]["id"])
        else:
            cat_id, cat_name = categories[categories_idx][0], categories[categories_idx][1]
            anns = vz_helper.get_annotations(img_list[img_idx]["id"], cat_id)
        vz_helper.showAnns(anns)
        plt.draw()
        print(f"file_name :{img_list[img_idx]['file_name']}")
        print(f"category_name: {cat_name}" if categories_idx > 0 else "all categories")
        


def download_file(ssh, src_path, target_path):
    os.system(f"scp -i {os.path.join(ssh['identity_dir_path'], 'key')} -P {ssh['port_num']} {ssh['user_name']}@{ssh['address']}:{src_path} {target_path}")


def main(args):
    global img_idx, ssh, img_list, categories, categories_idx, vz_helper

    # arguments
    ssh = dict(
        user_name=args.username,
        port_num=args.port,
        address=args.address,
        identity_dir_path=args.identity_dir_path,
        source_dir_path=args.source_dir_path,
        annotations_file_path=args.annotations_file_path
    )
    
    # set pathes for temporaty image and annotation files
    temp_image_path = os.path.join(ssh['identity_dir_path'], 'temp.jpg') if args.target_dir_path == "" else os.path.join(args.target_dir_path, 'temp.jpg')
    temp_json_path = os.path.join(ssh['identity_dir_path'], 'temp.json') if args.target_dir_path == "" else os.path.join(args.target_dir_path, 'temp.json')
    # if temporary json file already exists, skip the download process
    if os.path.isfile(temp_json_path):
        print("temporary json file already exists. skip the download process.")
    else:
        print("temporary json file doesn't exists on your directory. it will be downloaded soon.")
        download_file(ssh, f"{ssh['annotations_file_path']}", temp_json_path)

    # construct json helper
    vz_helper = VzHelper(temp_json_path)
    img_list = vz_helper.get_image_list()
    categories.extend(vz_helper.get_categories())

    # download temporary image file
    download_file(ssh, os.path.join(ssh["source_dir_path"], img_list[img_idx]['file_name']), temp_image_path)

    # set pyplot figure key release event
    fig = plt.figure()
    fig.canvas.mpl_connect("key_press_event", press)

    # add information text
    print(f"file_name :{img_list[img_idx]['file_name']}")
    # info_txt = f"file name: {img_list[img_idx]['file_name']}\ncategories: {categories}"
    # fig.text(.05, .05, info_txt)

    # show image on new window
    img = mimg.imread(temp_image_path)
    plt.imshow(img)
    # draw polygons
    anns = vz_helper.get_annotations(img_list[0]["id"])
    vz_helper.showAnns(anns)

    plt.axis("off")
    plt.show(block=True)

    print("program finished")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="execute sementic segmentation visualization")
    parser.add_argument("--port", "-P", help="port number of ssh server", required=True)
    parser.add_argument("--username", "-U", help="user name of ssh server", default="root")
    parser.add_argument("--address", "-A", help="address of ssh server", required=True)
    parser.add_argument("--identity_dir_path", "-i", help="path of identity file to access ssh host", required=True)
    parser.add_argument("--source_dir_path", "-s", help="path where image data is saved", required=True)
    parser.add_argument("--target_dir_path", "-t", help="path where temporary image will be saved", default="")
    parser.add_argument("--annotations_file_path", "-a", help="path of annotation file", required=True)
    args = parser.parse_args()

    main(args)

# python segmentation_visualization.py -P 2226 -A 49.50.165.163 -i C:/Users/alsrl/.ssh -s /opt/ml/input/data/ -a /opt/ml/input/data/train.json