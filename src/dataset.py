import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(project_root, "src"))

import torch
from torch.utils.data import Dataset
import numpy as np

import oba

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


import json

from pathlib import Path

class OBAValDataset(Dataset):
    def __init__(self, data_root, sample_indices, annotations_path, augmentations=None, use_oba=True, oba_prob=0.5):
        """
        data_root: Path to the dataset.
        sample_indices: Which train_X.* files to use.
        annotations_path: Path to train_annotations.json.
        augmentations: albumentations.Compose or None.
        use_oba: Boolean flag to apply OBA augmentation.
        oba_prob: Probability of applying OBA augmentation.
        """
        self.data_root = data_root
        self.image_paths = [data_root / "train_images" / f"train_{i}.tif" for i in sample_indices]
        self.mask_paths = [data_root / "train_masks" / f"train_{i}.npy" for i in sample_indices]
        self.augmentations = augmentations
        self.use_oba = use_oba
        self.oba_prob = oba_prob

        # Load annotations from the JSON file
        with open(annotations_path, 'r') as f:
            annotations_data = json.load(f)
        self.annotations = annotations_data.get('images', [])

        # Create a mapping from image file name to annotations list
        self.image_to_annotations = {}
        for item in self.annotations:
            self.image_to_annotations[item['file_name']] = item.get('annotations', [])
        
    def annotations_for_image(self, image_path):
        """
        Retrieve annotations for a given image by its file name.
        """
        filename = Path(image_path).name
        return self.image_to_annotations.get(filename, [])
    
    def class_to_channel(self, cls):
        """
        Map the annotation class name to the index of the segmentation mask channel.
        Adjust this mapping as needed for your project.
        """
        mapping = {
            'plantation': 0,
            'grassland_shrubland': 1,
            'mining': 2,
            'logging': 3
        }
        return mapping.get(cls, 0)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load the original image and mask
        image = load_image(self.image_paths[idx])  # shape: (1024, 1024, 12)
        mask = load_mask(self.mask_paths[idx])      # shape: (1024, 1024, 4)
        
        # Optionally perform OBA augmentation with some probability
        if self.use_oba and np.random.rand() < self.oba_prob:
            annotations = self.annotations_for_image(self.image_paths[idx])
            if annotations:
                annotation = np.random.choice(annotations)
                polygon = annotation['segmentation']
                obj_img, obj_mask = oba.extract_object(image, polygon, padding=5)
                if obj_img is not None:
                    # Use current image as background
                    target_img = image.copy()
                    target_mask = mask.copy()
                    class_channel = self.class_to_channel(annotation['class'])
                    image, mask = oba.paste_object(target_img, target_mask, obj_img, obj_mask, class_channel)
        
        # Optionally apply other augmentations
        if self.augmentations is not None:
            sample = self.augmentations(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
        
        # For training, you might need to convert to channels-first:
        image = image.transpose(2, 0, 1)  # (12, H, W)
        mask = mask.transpose(2, 0, 1)    # (4, H, W)
        image = normalize_image(image)
        
        return {"image": image, "mask": mask, "image_path": str(self.image_paths[idx])}