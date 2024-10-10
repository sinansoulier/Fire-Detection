import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class FireImageDataset(Dataset):
    """
    Dataset class for the fire image dataset.
    """
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        self.label_files = sorted(os.listdir(label_dir))
        self.image_files = sorted(self.__labeled_images(os.listdir(image_dir)))

    def __len__(self):
        """
        Returns the size of the dataset.

        Returns:
            int: size of the dataset
        """
        return len(self.image_files)

    def __getitem__(self, idx: int):
        """
        Returns the image and label at the given index.

        Args:
            idx (int): index of the image and label to return
        Returns:
            tuple: (image, label)
        """
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        image = Image.open(image_path)
        with open(label_path, 'r') as f:
            str_bboxes = f.readlines()

        bboxes = [self.__parse_bounding_box(str_bbox) for str_bbox in str_bboxes]
        label = torch.tensor(bboxes)

        if self.transform:
            image = self.transform(image)

        return image, label

    def verify(self):
        """
        Verifies that each image has a corresponding label.
        """
        for image_file, label_file in zip(self.image_files, self.label_files):
            assert image_file.split('.')[0] == label_file.split('.')[0]


    def __labeled_images(self, image_files: list[str]) -> list[str]:
        """
        Filter out images that do not have corresponding labels.

        Args:
            image_files (list[str]): list of image file names
        Returns:
            list[str]: list of image file names that have corresponding labels
        """
        return [f for f in image_files if f.split('.')[0] + '.txt' in self.label_files]

    def __parse_bounding_box(self, label: str) -> list[float]:
        """
        Extracts the bounding box from the label.

        Args:
            label (str): label containing bounding box information
        Returns:
            list[float]: list containing bounding box information
        """
        return list(map(float, label.strip().split(' ')))[1:]