import os
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mimg


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
        target_path = os.path.join(ssh['identity_path'], 'temp.jpg') if args.target_path == "" else args.target_path
        download_file(ssh, f"/opt/ml/input/data/batch_01_vt/000{img_num}.jpg", target_path)
        img = mimg.imread(target_path)
        plt.imshow(img)
        plt.axis("off")
        plt.draw()
    elif event.key == 'tab': 
        pass


def download_file(ssh, src_path, target_path):
    os.system(f"scp -i {os.path.join(ssh['identity_path'], 'key')} -P {ssh['port_num']} {ssh['user_name']}@{ssh['address']}:{src_path} {target_path}")


def main(args):
    global img_num, ssh

    # arguments
    ssh = dict(
        user_name=args.username,
        port_num=args.port,
        address=args.address,
        identity_path=args.identity_path
    )
    target_path = os.path.join(ssh['identity_path'], 'temp.jpg') if args.target_path == "" else args.target_path
    

    # download temporaty image file
    download_file(ssh, f"/opt/ml/input/data/batch_01_vt/000{img_num}.jpg", target_path)

    fig = plt.figure()
    fig.canvas.mpl_connect("key_press_event", press)

    # show image on new window
    img = mimg.imread(target_path)
    plt.imshow(img)
    plt.axis("off")
    plt.show(block=True)


    print("program finished")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="execute sementic segmentation visualization")
    parser.add_argument("--port", "-P", help="port number of ssh server", required=True)
    parser.add_argument("--username", "-U", help="user name of ssh server", default="root")
    parser.add_argument("--address", "-A", help="address of ssh server", required=True)
    parser.add_argument("--identity_path", "-I", help="path of identity file to access ssh host", required=True)
    parser.add_argument("--target_path", "-T", help="path where temporary image will be saved", default="")
    args = parser.parse_args()

    main(args)

# python segmentation_visualization.py -P 2226 -A 49.50.165.163 -I C:/Users/alsrl/.ssh