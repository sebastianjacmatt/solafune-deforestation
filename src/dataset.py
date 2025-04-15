import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(project_root, "src"))

import torch
from torch.utils.data import Dataset
import numpy as np

import src.utils.oba as oba

from data_utils import load_image, load_mask, normalize_image

from config import NUM_EVAL_INDICIES

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
            data_root / "evaluation_images" / f"evaluation_{i}.tif" for i in range(NUM_EVAL_INDICIES)
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

# In dataset.py, inside OBAValDataset:

class OBAValDataset(Dataset):
    def __init__(self, data_root, sample_indices, annotations_path, augmentations=None,
                 use_oba=True, oba_prob=0.5, visualize=False, num_oba_objects=1):
        """
        data_root: Path to the dataset.
        sample_indices: Which train_X.* files to use.
        annotations_path: Path to train_annotations.json.
        augmentations: albumentations.Compose or None.
        use_oba: Boolean flag to apply OBA augmentation.
        oba_prob: Probability of applying OBA augmentation.
        visualize: Flag to add additional visualization info.
        num_oba_objects: Number of objects to extract and paste per image.
        """
        self.data_root = data_root
        self.image_paths = [data_root / "train_images" / f"train_{i}.tif" for i in sample_indices]
        self.mask_paths = [data_root / "train_masks" / f"train_{i}.npy" for i in sample_indices]
        self.augmentations = augmentations
        self.use_oba = use_oba
        self.oba_prob = oba_prob
        self.visualize = visualize
        self.num_oba_objects = num_oba_objects
        
        # Load annotations from the JSON file
        with open(annotations_path, 'r') as f:
            annotations_data = json.load(f)
        self.annotations = annotations_data.get('images', [])
        
        # Create a mapping from image file name to annotations list
        self.image_to_annotations = {}
        for item in self.annotations:
            self.image_to_annotations[item['file_name']] = item.get('annotations', [])
            
    def annotations_for_image(self, image_path):
        filename = Path(image_path).name
        return self.image_to_annotations.get(filename, [])
    
    def class_to_channel(self, cls):
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
        image = load_image(self.image_paths[idx])  # Original image (1024, 1024, 12)
        mask = load_mask(self.mask_paths[idx])      # Original mask (1024, 1024, 4)
        
        sample_extra = {}
        
        # Check if we should apply OBA augmentation
        if self.use_oba and np.random.rand() < self.oba_prob:
            annotations = self.annotations_for_image(self.image_paths[idx])
            if annotations:
                # Use a cumulative target image and mask that get updated each iteration.
                cum_image = image.copy()
                cum_mask = mask.copy()
                
                for _ in range(self.num_oba_objects):
                    annotation = np.random.choice(annotations)
                    polygon = annotation['segmentation']
                    obj_img, obj_mask = oba.extract_object(cum_image, polygon, padding=5)
                    if obj_img is not None:
                        class_channel = self.class_to_channel(annotation['class'])
                        # Update the cumulative image and mask with each pasted object.
                        if self.visualize:
                            cum_image, cum_mask, bbox = oba.paste_object(
                                cum_image, cum_mask, obj_img, obj_mask, class_channel, highlight=True
                            )
                            # Store multiple bounding boxes in a list for visualization.
                            if "oba_bbox" not in sample_extra:
                                sample_extra["oba_bbox"] = [bbox]
                            else:
                                sample_extra["oba_bbox"].append(bbox)
                        else:
                            cum_image, cum_mask = oba.paste_object(
                                cum_image, cum_mask, obj_img, obj_mask, class_channel
                            )
                # Replace the original image and mask with the cumulative version.
                image = cum_image
                mask = cum_mask

        # Optionally apply additional augmentations
        if self.augmentations is not None:
            sample_dict = {"image": image, "mask": mask}
            sample_dict = self.augmentations(**sample_dict)
            image, mask = sample_dict["image"], sample_dict["mask"]

        # Convert to channels-first format and normalize before returning
        image = image.transpose(2, 0, 1)  # (12, H, W)
        mask = mask.transpose(2, 0, 1)    # (4, H, W)
        image = normalize_image(image)
        
        sample = {"image": image, "mask": mask, "image_path": str(self.image_paths[idx])}
        sample.update(sample_extra)
        return sample
