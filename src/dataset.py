import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(os.path.join(project_root, "src"))

src_root = os.path.abspath(os.path.join(project_root, "src/"))
sys.path.append(os.path.join(src_root, "utils"))

utils_root = os.path.abspath(os.path.join(src_root, "utils/"))
sys.path.append(os.path.join(utils_root, "object_based_augmentation"))

import torch
from torch.utils.data import Dataset
import numpy as np

import oba as oba
from object_augmentation import augment_object

from data_utils import load_image, load_mask, normalize_image
from config import NUM_EVAL_INDICIES, CLASS_NAMES

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

import random


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

        # Create a pool of images to use for augmentation
        self.pool = []
        for img_item in self.annotations:
            img_path = data_root/"train_images"/img_item["file_name"]
            for ann in img_item.get("annotations", []):
                self.pool.append((img_path, ann))
        
        # Create a mapping from image file name to annotations list
        self.image_to_annotations = {}
        for item in self.annotations:
            self.image_to_annotations[item['file_name']] = item.get('annotations', [])
            
    def annotations_for_image(self, image_path):
        filename = Path(image_path).name
        return self.image_to_annotations.get(filename, [])
    
    def class_to_channel(self, cls):
        mapping = { name: idx for idx, name in enumerate(CLASS_NAMES) }
        # e.g. { "grassland_shrubland": 0, "logging": 1, "mining": 2, "plantation": 3 }
        return mapping.get(cls, 0)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 1) Load base image & mask
        image = load_image(self.image_paths[idx])  # (1024,1024,12)
        mask  = load_mask(self.mask_paths[idx])    # (1024,1024, 4)

        sample_extra = {}
        cum_image = image.copy()
        cum_mask  = mask.copy()

        # 2) Maybe apply OBA across the pool
        if self.use_oba and np.random.rand() < self.oba_prob:
            MAX_EXTRACT_TRIES = 5

            for _ in range(self.num_oba_objects):
                # Attempt up to MAX_EXTRACT_TRIES to get a valid raw object
                for attempt in range(MAX_EXTRACT_TRIES):
                    # pick a random (source_image, annotation) pair
                    src_path, ann = random.choice(self.pool)
                    # (optionally skip same‐image if you want)
                    if src_path == self.image_paths[idx]:
                        continue

                    # load that source image
                    src_img = load_image(src_path)
                    raw_img, raw_mask = oba.extract_object(
                        src_img,
                        ann["segmentation"],
                        padding=5
                    )
                    if raw_img is not None:
                        break
                else:
                    # no valid crop found for this slot
                    continue

                # 2b) object‐level augs
                obj_img, obj_mask = augment_object(raw_img, raw_mask)

                # 2c) paste into cum_image / cum_mask
                ch = self.class_to_channel(ann["class"])
                if self.visualize:
                    cum_image, cum_mask, bbox = oba.paste_object(
                        cum_image, cum_mask,
                        obj_img, obj_mask,
                        ch, highlight=True
                    )
                    sample_extra.setdefault("oba_bbox", []).append(bbox)
                else:
                    cum_image, cum_mask = oba.paste_object(
                        cum_image, cum_mask,
                        obj_img, obj_mask,
                        ch
                    )

            # commit augmented version
            image, mask = cum_image, cum_mask

        # 3) any further albumentations
        if self.augmentations is not None:
            tmp = self.augmentations(image=image, mask=mask)
            image, mask = tmp["image"], tmp["mask"]

        # 4) format for model: (C, H, W) + normalize
        image = image.transpose(2, 0, 1)
        mask  = mask.transpose(2, 0, 1)
        image = normalize_image(image)

        # 5) build return dict
        sample = {
            "image": image,
            "mask":  mask,
            "image_path": str(self.image_paths[idx])
        }
        sample.update(sample_extra)
        return sample
