import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image

from .genarate_csv import genarate_csv


class NihDataset(Dataset):
    """
    A PyTorch Dataset class for the NIH Chest X-ray dataset.

    Args:
        mode (str): The mode of the dataset, either 'train' or 'test'.
        transform: Transform to be applied on a sample. Defaults to None.
    """

    def __init__(self, mode: str, transform=None):
        self.mode = mode
        self.transform = transform

        if not os.path.exists("datasets/data/nih/train.csv") or not os.path.exists(
            "datasets/data/nih/test.csv"
        ):
            genarate_csv()

        if mode == "train":
            csv_file = "datasets/data/nih/train.csv"
        elif mode == "test":
            csv_file = "datasets/data/nih/test.csv"
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self.df = pd.read_csv(csv_file)
        self.image_dir = "datasets/data/nih/images"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.loc[idx, "Image Index"]
        image_path = f"{self.image_dir}/{image_path}"

        image = decode_image(image_path, mode="RGB")
        label = torch.tensor(
            self.df.iloc[idx, 1:].values.astype(float), dtype=torch.float32
        )

        return image, image_path, label

    def set_transform(self, transform):
        self.transform = transform

    def get_labels(self):
        """
        Returns all the labels in the dataset (if not in test mode).
        """
        return self.df.iloc[:, 1:].values.astype(float)
