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




import random
import tifffile
import rasterio
from rasterio.windows import Window
from data_utils import normalize_image  # uses config.MEAN and config.STD
import json

# Import your original OBA pipeline Generator
from individual_object_functions import Generator

class FullOBADatasetMultiBand(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that applies the full OBA pipeline for instance augmentation,
    background compositing, etc.—adapted for single multi‐band (12-channel) TIFF images,
    with shadow additions disabled.
    """
    def __init__(self, data_root, json_file, sample_indices, IMG_ROW=128, IMG_COL=128, disable_shadows=True):
        """
        data_root: Path to dataset (e.g., DATASET_PATH) as a string.
        json_file: Path to your JSON annotations file (e.g., train_annotations.json).
        sample_indices: List of indices (e.g. [0,1,2,...]) corresponding to training samples.
        IMG_ROW, IMG_COL: Size of the patch to extract.
        disable_shadows: If True, shadow additions are disabled.
        """
        self.data_root = data_root
        self.sample_indices = sample_indices
        self.json_file = json_file
        self.IMG_ROW = IMG_ROW
        self.IMG_COL = IMG_COL
        self.disable_shadows = disable_shadows

        # Load the JSON annotations.
        # Expect keys like "train_0", "train_1", etc.
        with open(self.json_file, "r") as f:
            self.annotations = json.load(f)
        # Only use keys corresponding to your sample indices.
        self.keys = [f"train_{i}" for i in self.sample_indices if f"train_{i}" in self.annotations]

        # Configuration for instance augmentation.
        # (Here, we set instance augmentation to always occur for demonstration.)
        self.instance_augm = True
        self.instance_augm_prob = 1.0

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        annotation = self.annotations[key]
        # Annotation parameters (adjust field names as in your JSON)
        upper_left_x = annotation["upper_left_x"]
        upper_left_y = annotation["upper_left_y"]
        pol_width    = annotation["pol_width"]
        pol_height   = annotation["pol_height"]
        crop_id      = annotation.get("crop_id", 0)
        
        # Construct paths for image and mask.
        image_path = os.path.join(self.data_root, "train_images", f"{key}.tif")
        mask_path  = os.path.join(self.data_root, "train_masks", f"{key}.npy")
        
        # Load full image and mask.
        image_full = tifffile.imread(image_path)  # shape: (1024, 1024, 12)
        mask = np.load(mask_path)                 # shape: (4, 1024, 1024)
        mask = mask.transpose(1, 2, 0)             # shape: (1024,1024,4)
        
        # If instance augmentation is enabled, perform instance-level augmentation.
        if self.instance_augm and random.random() < self.instance_augm_prob:
            # Choose a class to augment. For example, use class 3 ("plantation").
            class_idx = 3
            instance_mask = mask[:, :, class_idx]
            labeled = self.label_instance(instance_mask)
            regions = self.region_props(labeled)
            if regions:
                region = random.choice(regions)
                minr, minc, maxr, maxc = region['bbox']
                obj_img = image_full[minr:maxr, minc:maxc, :]
                obj_mask = mask[minr:maxr, minc:maxc, :]
                
                # Select a random background from another sample.
                bg_idx = random.choice(self.sample_indices)
                bg_key = f"train_{bg_idx}"
                bg_image_path = os.path.join(self.data_root, "train_images", f"{bg_key}.tif")
                bg_full = tifffile.imread(bg_image_path)
                h, w, _ = bg_full.shape
                if self.IMG_ROW < h and self.IMG_COL < w:
                    top = random.randint(0, h - self.IMG_ROW)
                    left = random.randint(0, w - self.IMG_COL)
                    bg_patch = bg_full[top:top+self.IMG_ROW, left:left+self.IMG_COL, :]
                    
                    # Paste the object into the center of bg_patch.
                    ph, pw, _ = obj_img.shape
                    start_h = (self.IMG_ROW - ph) // 2
                    start_w = (self.IMG_COL - pw) // 2
                    obj_bool = (obj_mask[:, :, 0] > 0.5)
                    for c in range(12):
                        bg_patch[start_h:start_h+ph, start_w:start_w+pw, c][obj_bool] = obj_img[:, :, c][obj_bool]
                    image_full = bg_patch
                    # For the mask, create a blank mask and paste the object mask.
                    mask = np.zeros_like(mask)
                    mask[start_h:start_h+ph, start_w:start_w+pw, :] = obj_mask
        
        # Finally, crop a random patch from the (possibly augmented) full image.
        with rasterio.open(image_path) as src:
            size_x = src.width
            size_y = src.height
        rnd_x = random.randint(0, size_x - self.IMG_COL)
        rnd_y = random.randint(0, size_y - self.IMG_ROW)
        # Crop patch from image_full and mask.
        image_patch = image_full[rnd_y:rnd_y+self.IMG_ROW, rnd_x:rnd_x+self.IMG_COL, :]
        mask_patch  = mask[rnd_y:rnd_y+self.IMG_ROW, rnd_x:rnd_x+self.IMG_COL, :]
        
        # Normalize image (expected shape for normalization is (12, H, W)).
        image_patch = image_patch.transpose(2, 0, 1)  # (12, H, W)
        image_patch = normalize_image(image_patch)
        mask_patch = mask_patch.transpose(2, 0, 1)     # (4, H, W)
        
        return {
            "image": image_patch,
            "mask": mask_patch,
            "key": key,
        }
    
    def label_instance(self, instance_mask):
        from skimage.measure import label
        return label(instance_mask > 0.5)
    
    def region_props(self, labeled):
        from skimage.measure import regionprops
        regions = regionprops(labeled)
        # Return list of dictionaries containing at least the bounding box.
        return [{"bbox": region.bbox} for region in regions]
