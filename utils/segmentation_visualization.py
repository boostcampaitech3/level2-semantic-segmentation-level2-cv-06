import os
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from json_helper import JsonHelper


ssh = dict()
img_num = 2


def press(event):
    global img_num
    print(f"pressed key: {event.key}")
    sys.stdout.flush()
    if event.key == 'x':
        plt.close()
    elif event.key == 'n':
        img_num += 1
        target_path = os.path.join(ssh['identity_dir_path'], 'temp.jpg') if args.target_dir_path == "" else os.path.join(args.target_dir_path, 'temp.jpg')
        download_file(ssh, f"/opt/ml/input/data/batch_01_vt/000{img_num}.jpg", target_path)
        img = mimg.imread(target_path)
        plt.imshow(img)
        plt.axis("off")
        plt.draw()
    elif event.key == 'tab': 
        pass


def download_file(ssh, src_path, target_path):
    os.system(f"scp -i {os.path.join(ssh['identity_dir_path'], 'key')} -P {ssh['port_num']} {ssh['user_name']}@{ssh['address']}:{src_path} {target_path}")


def main(args):
    global img_num, ssh

    # arguments
    ssh = dict(
        user_name=args.username,
        port_num=args.port,
        address=args.address,
        identity_dir_path=args.identity_dir_path,
        annotations_file_path=args.annotations_file_path
    )
    
    # download temporaty image and annotation files
    temp_image_path = os.path.join(ssh['identity_dir_path'], 'temp.jpg') if args.target_dir_path == "" else os.path.join(args.target_dir_path, 'temp.jpg')
    download_file(ssh, f"/opt/ml/input/data/batch_01_vt/000{img_num}.jpg", temp_image_path)
    temp_json_path = os.path.join(ssh['identity_dir_path'], 'temp.json') if args.target_dir_path == "" else os.path.join(args.target_dir_path, 'temp.json')
    # if temporary json file already exists, skip download process
    if os.path.isfile(temp_json_path):
        print("temporary json file already exsits. skip the download process.")
    else:
        download_file(ssh, f"{ssh['annotations_file_path']}", temp_json_path)

    # construct json helper
    json_helper = JsonHelper(temp_json_path)
    img_list = json_helper.get_image_list()
    print(f"img_list: {img_list}")

    fig = plt.figure()
    fig.canvas.mpl_connect("key_press_event", press)

    # show image on new window
    img = mimg.imread(temp_image_path)
    plt.imshow(img)
    plt.axis("off")
    plt.show(block=True)


    print("program finished")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="execute sementic segmentation visualization")
    parser.add_argument("--port", "-P", help="port number of ssh server", required=True)
    parser.add_argument("--username", "-U", help="user name of ssh server", default="root")
    parser.add_argument("--address", "-A", help="address of ssh server", required=True)
    parser.add_argument("--identity_dir_path", "-i", help="path of identity file to access ssh host", required=True)
    parser.add_argument("--target_dir_path", "-t", help="path where temporary image will be saved", default="")
    parser.add_argument("--annotations_file_path", "-a", help="path of annotation file", required=True)
    args = parser.parse_args()

    main(args)

# python segmentation_visualization.py -P 2226 -A 49.50.165.163 -i C:/Users/alsrl/.ssh -a /opt/ml/input/data/train.json