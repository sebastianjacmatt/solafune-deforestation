import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
sys.path.append(os.path.join(project_root, "src"))
print(project_root)

import matplotlib.pyplot as plt
from dataset import TrainValDataset, OBAValDataset
from global_paths import DATASET_PATH, TRAIN_ANNOTATIONS_PATH
from train_utils import train_indices


import matplotlib.pyplot as plt
import numpy as np

def to_rgb(img):
    """
    Convert a 12-channel image (C, H, W) to an RGB image using Sentinel-2 natural color.
    For Sentinel-2, natural color is typically defined as:
       Red = channel 3, Green = channel 2, Blue = channel 1.
    The image is assumed to be in channels-first format.
    """
    # Select channels: adjust indices if needed (here indices 3,2,1)
    rgb = img[[3, 2, 1], :, :]
    # Normalize the image to [0, 1] for display
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    # Convert to channels-last format (H, W, C)
    return rgb.transpose(1, 2, 0)


def visualize_sample(dataset, index=0):
    """
    Visualizes an image sample from the dataset.
    Assumes sample["image"] is in channels-first format (C, H, W) and normalized.
    """
    sample = dataset[index]
    # Convert image from (C, H, W) to an RGB composite for display
    rgb_image = to_rgb(sample["image"])
    # Convert mask from (C, H, W) to (H, W, C) for overlay visualization
    mask = sample["mask"].transpose(1, 2, 0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(rgb_image)
    axes[0].set_title("RGB Composite Image")
    axes[0].axis("off")
    
    axes[1].imshow(rgb_image)
    # Overlay mask channels using different colormaps
    colormaps = [plt.cm.Reds, plt.cm.Greens, plt.cm.Blues, plt.cm.Oranges]
    for i in range(mask.shape[-1]):
        axes[1].imshow(mask[:, :, i], cmap=colormaps[i], alpha=0.5)
    axes[1].set_title("Image with Mask Overlay")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()
# Example usage:
# Assume TrainValDataset and OBAValDataset are imported from your dataset.py.
# Also assume that data_root, sample_indices, and annotations_path are defined.


sample_indices = list(range(10))  # example indices
annotations_path = DATASET_PATH / "train_annotations.json"

# Create the original dataset (without OBA augmentation)
original_dataset = TrainValDataset(data_root=DATASET_PATH, sample_indices=sample_indices, augmentations=None)

# Create the OBA dataset (with cut-and-paste augmentation)
oba_dataset = OBAValDataset(data_root=DATASET_PATH, sample_indices=sample_indices,
                            annotations_path=annotations_path, augmentations=None,
                            use_oba=True, oba_prob=1.0)  # use 100% probability for testing

# Visualize a sample from the original dataset
visualize_sample(original_dataset, index=1)

# Visualize a sample from the OBA dataset
visualize_sample(oba_dataset, index=1)
