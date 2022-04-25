import fiftyone as fo
import argparse

def main(arg):
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=arg.data_dir,
        labels_path=arg.anno_dir,
        label_types=["detections", "segmentations"],
    )
    session = fo.launch_app(dataset, port=arg.port, address="0.0.0.0")
    session.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', type=str, default="/opt/ml/input/data",
                        help='imageData directory')
    parser.add_argument('--anno_dir', '-a', type=str, default="/opt/ml/input/data/train_all.json",
                        help='annotation Data directory')
    parser.add_argument('--port', '-p', type=int, default=6006,
                        help='Port Number')
    args = parser.parse_args()
    main(args)