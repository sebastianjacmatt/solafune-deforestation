import sys
import os
import numpy as np
import tifffile
import torch

from config import MEAN, STD

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(project_root, "src"))

def load_mask(mask_path):
    """
    Loads the mask from .npy and returns a float32 array in [0, 1].
    Shape: (4, 1024, 1024)
    """
    mask = np.load(mask_path)
    assert mask.shape == (4, 1024, 1024), f"Unexpected mask shape: {mask.shape}"
    mask = mask.transpose(1, 2, 0)  # (H, W, 4)
    return (mask.astype(np.float32) / 255.0)  # normalize to [0, 1]


def load_image(image_path):
    """
    Loads the TIF image with shape (1024, 1024, 12).
    Returns a float32 array with no NaNs.
    """
    image = tifffile.imread(image_path)
    assert image.shape == (1024, 1024, 12), f"Unexpected image shape: {image.shape}"
    image = np.nan_to_num(image)
    return image.astype(np.float32)


def normalize_image(image):
    """
    Normalizes an image (C, H, W) or (H, W, C) using precomputed mean and std.
    Ensures the shape is (12, H, W) before applying.
    """
    # If (H,W,C), transpose to (C,H,W)
    if image.shape[0] != 12 and image.shape[-1] == 12:
        image = image.transpose(2, 0, 1)

    mean_arr = np.array(MEAN, dtype=np.float32).reshape(12, 1, 1)
    std_arr = np.array(STD, dtype=np.float32).reshape(12, 1, 1)
    return (image - mean_arr) / std_arr

def pad_image(image, channels , padding="zeroing"):
    """
    Pads an image except spesified channels

    Args:
        image
        channels: list of channels to be padded
        padding: "zeroing" or "repitition"
    Returns:
        padded image
    """
    assert padding in ["zeroing", "repitition"], f"Invalid padding method: {padding}"

def domain_image_split(image, mask , channels: list[list[int]]) -> list[dict[str, torch.Tensor]]:
    """
    Splits an image into spesified channel images where channels outside domains are padded
    Args:
        image: input image
        mask: mask of the image
        channels: list of channels to be split
    Returns:
        domain_images: dictionary of 12-channel images of different domains 
        example:
        ```python
        domain_samples = [
            {"image": image1, "mask": mask1},
            {"image": image2, "mask": mask2},
            ...,
        ]
        ```

    """
