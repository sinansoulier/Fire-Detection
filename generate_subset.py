import os
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import yaml

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: parsed arguments
    """
    parser = argparse.ArgumentParser(description='Create a subset of the fire image dataset')
    parser.add_argument("--data_dir", type=str, default="data", required=False)
    parser.add_argument("--output_dir", type=str, default="data_subset", required=False)
    parser.add_argument("--portion", type=float, default=0.1, required=False)
    return parser.parse_args()

def get_dirs(base: str) -> dict:
    """
    Get the list of directories in the base directory.

    Args:
        base (str): path to the base directory

    Returns:
        dict: map of directories in the base directory
    """
    res_dirs = {}
    for d in os.listdir(base):
        for f in os.listdir(os.path.join(base, d)):
            res_dirs[f.split(".")[0]] = os.path.join(base, d, f)
    return res_dirs

def gather_data(data_dir: str) -> tuple:
    """
    Gather the paths to the images and labels in the data directory.

    Args:
        data_dir (str): path to the data directory
    Returns:
        tuple: (images_paths, label_paths)
    """
    image_dir = os.path.join(data_dir, "images")
    label_dir = os.path.join(data_dir, "labels")

    images_map = get_dirs(image_dir)
    labels_map = get_dirs(label_dir)

    list_keys = set(images_map.keys()).intersection(set(labels_map.keys()))

    images_paths = sorted([images_map[k] for k in list_keys])
    label_paths = sorted([labels_map[k] for k in list_keys])

    return images_paths, label_paths

def create_yaml_config(args: argparse.Namespace) -> None:
    """
    Create the yaml configuration file for the subset.

    Args:
        args (argparse.Namespace): parsed arguments
    """
    with open(os.path.join(args.output_dir, "data.yaml"), "w") as f:
        yaml_content = {
            "train": os.path.join(args.output_dir, "train"),
            "val": os.path.join(args.output_dir, "val"),
            "test": os.path.join(args.output_dir, "test"),
            "nc": 1,
            "names": ['smoke']
        }
        yaml.dump(yaml_content, f)


def create_subset(args: argparse.Namespace) -> None:
    """
    Create a subset of the fire image dataset.

    Args:
        args (argparse.Namespace): parsed arguments
    """
    images_paths, label_paths = gather_data(args.data_dir)
    subset_size = int(len(images_paths) * args.portion)
    new_indices = np.random.randint(0, len(images_paths) - 1, subset_size)

    images_paths = np.array(images_paths)[new_indices]
    label_paths = np.array(label_paths)[new_indices]

    # Split the data into train, validation, and test sets
    train_images, test_images, train_labels, test_labels = train_test_split(images_paths, label_paths, test_size=0.20, random_state=42)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.15, random_state=42)

    for i, split in enumerate(["train", "val", "test"]):
        os.makedirs(os.path.join(args.output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, "labels"), exist_ok=True)

        for image, label in zip([train_images, val_images, test_images][i], [train_labels, val_labels, test_labels][i]):
            # Copy the image and label to the new directory
            os.system(f"cp {image} {os.path.join(args.output_dir, split, 'images')}")
            os.system(f"cp {label} {os.path.join(args.output_dir, split, 'labels')}")

    create_yaml_config(args)

if __name__ == "__main__":
    args = parse_args()
    if os.path.exists(args.output_dir):
        os.system(f"rm -rf {args.output_dir}")
    create_subset(args)