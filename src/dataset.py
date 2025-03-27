import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(project_root, "src"))

import torch
from torch.utils.data import Dataset
import numpy as np

from data_utils import load_image, load_mask, normalize_image

class TrainValDataset(Dataset):
    def __init__(self, data_root, sample_indices, augmentations=None):
        """
        data_root: Path to dataset
        sample_indices: which train_X.* files to use
        augmentations: albumentations.Compose or None
        """
        self.image_paths = [
            data_root / "train_images" / f"train_{i}.tif" for i in sample_indices
        ]
        self.mask_paths = [
            data_root / "train_masks" / f"train_{i}.npy" for i in sample_indices
        ]
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_image(self.image_paths[idx])  # shape: (1024, 1024, 12)
        mask = load_mask(self.mask_paths[idx])    # shape: (1024, 1024, 4)

        # albumentations expects dict with keys = ["image", "mask"]
        sample = {"image": image, "mask": mask}
        if self.augmentations is not None:
            sample = self.augmentations(**sample)  # apply aug
        # sample["image"] = (H, W, C), sample["mask"] = (H, W, 4)

        # put channels first
        sample["image"] = sample["image"].transpose(2, 0, 1)
        sample["mask"] = sample["mask"].transpose(2, 0, 1)

        # normalize the image
        sample["image"] = normalize_image(sample["image"])

        return {
            "image": sample["image"],
            "mask": sample["mask"],
            "image_path": str(self.image_paths[idx]),
            "mask_path": str(self.mask_paths[idx]),
        }


class TestDataset(Dataset):
    def __init__(self, data_root):
        """
        data_root: Path to dataset containing evaluation_images dir
        """
        self.image_paths = [
            data_root / "evaluation_images" / f"evaluation_{i}.tif" for i in range(118)
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_image(self.image_paths[idx])
        # shape is (1024, 1024, 12); normalize expects (12, H, W)
        image = image.transpose(2, 0, 1)
        image = normalize_image(image)

        return {
            "image": image,
            "image_path": str(self.image_paths[idx]),
        }
