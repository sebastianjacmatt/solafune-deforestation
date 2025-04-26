import sys
import os
import json
from pathlib import Path
import random
from torch.utils.data import Dataset
import numpy as np

# Append project paths
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(os.path.join(project_root, "src"))

src_root = os.path.abspath(os.path.join(project_root, "src/"))
sys.path.append(os.path.join(src_root, "utils"))

utils_root = os.path.abspath(os.path.join(src_root, "utils/"))
sys.path.append(os.path.join(utils_root, "object_based_augmentation"))

sys.path.append(os.path.join(project_root, "src/utils/object_based_augmentation"))

from data_utils import load_image, load_mask, normalize_image
from config import NUM_EVAL_INDICIES, CLASS_NAMES, MAX_EXTRACT_TRIES
import oba as oba
from object_augmentation import augment_object


class TrainValDataset(Dataset):
    def __init__(self, data_root, sample_indices, augmentations=None):
        """
        data_root: Path to dataset
        sample_indices: Which train_X.* files to use
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

        # Albumentations expects dict with keys = ["image", "mask"]
        sample = {"image": image, "mask": mask}
        if self.augmentations is not None:
            sample = self.augmentations(**sample)  # apply aug
        # sample["image"] = (H, W, C), sample["mask"] = (H, W, 4)

        # Put the channels first
        sample["image"] = sample["image"].transpose(2, 0, 1)
        sample["mask"] = sample["mask"].transpose(2, 0, 1)

        # Normalize the image
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
        # Shape is (1024, 1024, 12); Normalize expects (12, H, W)
        image = image.transpose(2, 0, 1)
        image = normalize_image(image)

        return {
            "image": image,
            "image_path": str(self.image_paths[idx]),
        }


class OBAValDataset(Dataset):
    def __init__(
        self,
        data_root,
        sample_indices,
        annotations_path,
        background_root=None,
        background_prob=0.3,
        extract_from_same_image=False,
        augmentations=None,
        use_oba=True,
        oba_prob=0.5,
        visualize=False,
        num_oba_objects=1
    ):
        """
        Create a dataset that optionally applies Object-Based Augmentation (OBA) 
        by cutting objects from annotated polygons and pasting them either back 
        into the same image or onto other background images.

        Parameters:
            data_root : Path-like
                Root directory of the dataset. Expects subfolders `train_images/` and `train_masks/`.

            sample_indices : Sequence of int
                List of indices indicating which `train_{i}.tif` and `train_{i}.npy` files to load.

            annotations_path : Path-like
                Path to the JSON file containing polygon annotations under `"images"`.

            background_root : Path-like, optional (default=None)
                Directory of additional images to use as background canvases when
                `background_prob` > 0. If None, only original images are used.

            background_prob : float, default=0.3
                Probability of replacing the original background with a random image
                from `background_root` before pasting objects.

            extract_from_same_image : bool, default=False
                If True, objects are always cut from the same image being augmented.
                If False, objects are drawn from a pool of all annotated images.

            augmentations : albumentations.Compose or None, default=None
                Additional per-sample augmentations to apply *after* OBA (e.g., flips, crops).

            use_oba : bool, default=True
                Master switch to enable or disable the OBA cut-and-paste step.

            oba_prob : float, default=0.5
                Probability of performing OBA on any given sample. When disabled, returns
                only the original image and mask.

            visualize : bool, default=False
                If True, records each pasted object's bounding box in `sample["oba_bbox"]`
                for visualization.

            num_oba_objects : int, default=1
                Number of objects to attempt to cut and paste into each sample.

        """
        self.data_root = data_root
        self.image_paths = [data_root / "train_images" / f"train_{i}.tif" for i in sample_indices]
        self.mask_paths = [data_root / "train_masks" / f"train_{i}.npy" for i in sample_indices]
        self.augmentations = augmentations
        self.use_oba = use_oba
        self.oba_prob = oba_prob
        self.visualize = visualize
        self.num_oba_objects = num_oba_objects
        self.extract_from_same_image = extract_from_same_image
        self.background_prob = background_prob

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
        
        # optionally build list of background tif paths
        if background_root:
            bgdir = Path(background_root)
            self.background_paths = [p for p in bgdir.iterdir() if p.suffix.lower() in (".tif", ".png", ".jpg")]
        else:
            self.background_paths = []
        
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
        # Load base image & mask
        image = load_image(self.image_paths[idx])  # (1024,1024,12)
        mask  = load_mask(self.mask_paths[idx])    # (1024,1024, 4)
        sample_extra = {}

        # Decide whether to use a background image instead from separate dataset (for this project de did not)
        if getattr(self, "background_paths", None) and np.random.rand() < self.background_prob:
            # Pick a random background (no pre-existing deforestation mask)
            bg_path = random.choice(self.background_paths)
            cum_image = load_image(bg_path)
            # Start with an empty 4‑channel mask
            H, W, _ = cum_image.shape
            cum_mask = np.zeros((H, W, mask.shape[2]), dtype=mask.dtype)
        else:
            cum_image = image.copy()
            cum_mask  = mask.copy()

        # Maybe apply OBA given defined probability
        if self.use_oba and np.random.rand() < self.oba_prob:
            for _ in range(self.num_oba_objects):
                # Try to get a valid object up to MAX_EXTRACT_TRIES to avoid infinite loop
                for _try in range(MAX_EXTRACT_TRIES):
                    if self.extract_from_same_image:
                        # Only sample from this image's annotations if flag is set to True
                        anns = self.annotations_for_image(self.image_paths[idx])
                        if not anns:
                            break
                        ann    = random.choice(anns)
                        src_img = image
                    else:
                        # Sample from the full pool
                        src_path, ann = random.choice(self.pool)
                        src_img = load_image(src_path)

                    raw_img, raw_mask = oba.extract_object(
                        src_img,
                        ann["segmentation"],
                        padding=5
                    )
                    if raw_img is not None:
                        break
                else:
                    # Failed to get anything this slot
                    continue

                # Object-level augmentation
                obj_img, obj_mask = augment_object(raw_img, raw_mask)

                # Paste into cum_image / cum_mask
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

            # Commit the augmented image + mask
            image, mask = cum_image, cum_mask

        # Global image albumentations
        if self.augmentations is not None:
            tmp = self.augmentations(image=image, mask=mask)
            image, mask = tmp["image"], tmp["mask"]

        # To channels-first & normalize
        image = image.transpose(2, 0, 1)
        mask  = mask.transpose(2, 0, 1)
        image = normalize_image(image)

        # Return
        sample = {
            "image":      image,
            "mask":       mask,
            "image_path": str(self.image_paths[idx])
        }
        sample.update(sample_extra)
        return sample
