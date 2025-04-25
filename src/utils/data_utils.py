import sys
import os
from typing import Dict, List
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

# Methods related to Interpolation Robustness (IR)
def pad_image(
    image: np.ndarray,
    keep_channels: List[int],
    padding: str = "zeroing",
) -> np.ndarray:
    """
    Overwrite every spectral band *except* those in `keep_channels`.

    Parameters
    ----------
    image : np.ndarray
        A 12-band image, shape (12, H, W) **or** (H, W, 12).
    keep_channels : list[int]
        Channel indices (0-based) to preserve.
    padding : {"zeroing", "repetition"}, default "zeroing"
        * "zeroing"    – set discarded bands to 0  
        * "repetition" – copy the first kept band into every discarded band

    Returns
    -------
    np.ndarray  (12, H, W)  same dtype as input.
    """
    assert padding in {"zeroing", "repetition"}, f"Invalid padding method: {padding}"

    # ensure (C, H, W)
    if image.shape[0] != 12 and image.shape[-1] == 12:
        image = image.transpose(2, 0, 1)

    out = image.copy()
    discard = [i for i in range(12) if i not in keep_channels]

    if padding == "zeroing":
        out[discard] = 0.0
    else:  # "repetition"
        ref = out[keep_channels[0]]
        for ch in discard:
            out[ch] = ref

    return out


def domain_image_split(
    image: np.ndarray,
    mask: np.ndarray,
    channel_groups: List[List[int]],
    padding: str = "zeroing",
) -> List[Dict[str, torch.Tensor]]:
    """
    Produce one 12-band sample per spectral domain.

    Each sub-list in `channel_groups` defines a *domain*; its channels keep
    real data, every other band is padded via `padding`.

    Returns
    -------
    list of dict
        [
          {"image": torch.FloatTensor(12, H, W),
           "mask" : torch.FloatTensor(4 , H, W)},
          ...
        ]
    """
    # standardise layout once
    if image.shape[0] != 12 and image.shape[-1] == 12:
        image = image.transpose(2, 0, 1)
    if mask.shape[0] != 4 and mask.shape[-1] == 4:
        mask = mask.transpose(2, 0, 1)

    samples = []
    for grp in channel_groups:
        padded = pad_image(image, grp, padding=padding)
        samples.append({
            "image": torch.from_numpy(padded.copy()),
            "mask" : torch.from_numpy(mask.copy()),
        })
    return samples


