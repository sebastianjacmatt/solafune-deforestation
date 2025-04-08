import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(project_root, "src"))

import torch
from torch.utils.data import Dataset
import numpy as np

import oba

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
        image = load_image(self.image_paths[idx])  # (1024, 1024, 12)
        mask = load_mask(self.mask_paths[idx])      # (1024, 1024, 4)
        
        # Initialize a dictionary to hold extra info (like pasted-object bboxes) for visualization.
        sample_extra = {}
        
        # Optionally perform OBA augmentation with some probability
        if self.use_oba and np.random.rand() < self.oba_prob:
            annotations = self.annotations_for_image(self.image_paths[idx])
            if annotations:
                # Paste up to num_oba_objects; note that paste_object updates the image and mask sequentially.
                for _ in range(self.num_oba_objects):
                    # Randomly choose an annotation from available ones.
                    annotation = np.random.choice(annotations)
                    polygon = annotation['segmentation']
                    obj_img, obj_mask = oba.extract_object(image, polygon, padding=5)
                    if obj_img is not None:
                        target_img = image.copy()
                        target_mask = mask.copy()
                        class_channel = self.class_to_channel(annotation['class'])
                        # When visualizing, you might want to get the pasted object's bbox.
                        if self.visualize:
                            image, mask, bbox = oba.paste_object(target_img, target_mask, obj_img, obj_mask,
                                                                 class_channel, highlight=True)
                            # Save the bbox for visualization; if multiple objects are pasted, store them in a list.
                            if "oba_bbox" not in sample_extra:
                                sample_extra["oba_bbox"] = [bbox]
                            else:
                                sample_extra["oba_bbox"].append(bbox)
                        else:
                            image, mask = oba.paste_object(target_img, target_mask, obj_img, obj_mask, class_channel)
                    # End for each object
        # Optionally apply additional augmentations.
        if self.augmentations is not None:
            sample_dict = {"image": image, "mask": mask}
            sample_dict = self.augmentations(**sample_dict)
            image, mask = sample_dict["image"], sample_dict["mask"]

        # Convert to channels-first and normalize.
        image = image.transpose(2, 0, 1)  # (12, H, W)
        mask = mask.transpose(2, 0, 1)    # (4, H, W)
        image = normalize_image(image)
        
        sample = {"image": image, "mask": mask, "image_path": str(self.image_paths[idx])}
        sample.update(sample_extra)
        return sample
