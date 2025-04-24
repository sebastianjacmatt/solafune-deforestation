import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Append project paths
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(os.path.join(project_root, "src"))
sys.path.append(os.path.join(project_root, "src/utils"))

from dataset import TrainValDataset, OBAValDataset
from global_paths import DATASET_PATH, TRAIN_ANNOTATIONS_PATH, SEPARATE_BACKGROUND_IMAGES

def to_rgb(img, ignore_bboxes=None):
    """
    Convert a 12-channel image (C, H, W) to an RGB image using Sentinel-2 natural color,
    stretching only on the background (excluding any pasted-object bboxes).
    """
    # Select bands RGB
    rgb = img[[3, 2, 1], :, :].astype(np.float32)   # (3, H, W)

    H, W = rgb.shape[1], rgb.shape[2]
    bg_mask = np.ones((H, W), dtype=bool)

    # Mark pasted-object regions as False
    if ignore_bboxes:
        for bbox in ignore_bboxes:
            if not bbox or len(bbox) != 4:
                continue
            top, left, height, width = bbox
            bg_mask[top:top+height, left:left+width] = False

    # Compute min/max only over background pixels
    vals = rgb[:, bg_mask]  # shape (3, num_bg)
    vmin, vmax = vals.min(), vals.max()

    # Stretch & clip
    rgb = (rgb - vmin) / (vmax - vmin + 1e-6)
    rgb = np.clip(rgb, 0.0, 1.0)

    return rgb.transpose(1, 2, 0)  # → (H, W, 3)


def visualize_both_samples(original_dataset, oba_dataset, index=0):
    """
    Show a 2x2 plot comparing an original sample and its OBA-augmented version.
    Top row: original RGB and mask overlay. Bottom row: OBA RGB and mask+bbox overlay.
    """
    sample_orig = original_dataset[index]
    sample_oba  = oba_dataset[index]

    # extract any OBA bboxes (list or None)
    bboxes = sample_oba.get("oba_bbox", None)

    # Make RGB panels, ignoring new objects for the OBA image
    rgb_orig = to_rgb(sample_orig["image"], ignore_bboxes=None)
    rgb_oba  = to_rgb(sample_oba["image"],  ignore_bboxes=bboxes)

    # prepare masks for overlay
    mask_orig = sample_orig["mask"].transpose(1, 2, 0)
    mask_oba  = sample_oba["mask"].transpose(1, 2, 0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Top‐left: original RGB
    axes[0, 0].imshow(rgb_orig)
    axes[0, 0].set_title("Original RGB Composite")
    axes[0, 0].axis("off")

    # Top‐right: original w/ mask overlay
    axes[0, 1].imshow(rgb_orig)
    colormaps = [plt.cm.Reds, plt.cm.Greens, plt.cm.Blues, plt.cm.Oranges]
    for i in range(mask_orig.shape[-1]):
        axes[0, 1].imshow(mask_orig[:, :, i], cmap=colormaps[i], alpha=0.5)
    axes[0, 1].set_title("Original with Mask Overlay")
    axes[0, 1].axis("off")

    # Bottom‐left: OBA RGB composite
    axes[1, 0].imshow(rgb_oba)
    axes[1, 0].set_title("OBA RGB Composite")
    axes[1, 0].axis("off")

    # Bottom‐right: OBA + masks + bboxes
    axes[1, 1].imshow(rgb_oba)
    for i in range(mask_oba.shape[-1]):
        axes[1, 1].imshow(mask_oba[:, :, i], cmap=colormaps[i], alpha=0.5)

    # Draw the bboxes
    if bboxes:
        for bbox in bboxes:
            if not bbox or len(bbox) != 4:
                continue
            top, left, height, width = bbox
            rect = plt.Rectangle((left, top), width, height,
                                 edgecolor='magenta', facecolor='none', linewidth=2)
            axes[1, 1].add_patch(rect)

    axes[1, 1].set_title("OBA with Mask and Highlight")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_external_background_pair(
    data_root: Path,
    annotations_path: Path,
    background_root: Path,
    sample_indices: list,
    index: int = 0,
    num_oba_objects: int = 5
):
    """
    Left panel: ONLY the external background (no OBA).
    Right panel: The same background + Pasted objects.
    """
    # External background only
    original_bg = OBAValDataset(
        data_root=data_root,
        sample_indices=sample_indices,
        annotations_path=annotations_path,
        augmentations=None,
        use_oba=True,
        oba_prob=1.0,
        visualize=True,
        num_oba_objects=0,  # No objects
        background_root=str(background_root),
        background_prob=1.0,
        extract_from_same_image=True
    )

    # External background + Pasted objects
    oba_bg = OBAValDataset(
        data_root=data_root,
        sample_indices=sample_indices,
        annotations_path=annotations_path,
        augmentations=None,
        use_oba=True,
        oba_prob=1.0,
        visualize=True,
        num_oba_objects=num_oba_objects,
        background_root=str(background_root),
        background_prob=1.0,
        extract_from_same_image=True
    )

    # Visualize both samples
    visualize_both_samples(original_bg, oba_bg, index=index)


if __name__ == "__main__":
    sample_indices = list(range(10))

    original_dataset = TrainValDataset(
        data_root=DATASET_PATH,
        sample_indices=sample_indices,
        augmentations=None
    )
    oba_dataset = OBAValDataset(
        data_root=DATASET_PATH,
        sample_indices=sample_indices,
        annotations_path=TRAIN_ANNOTATIONS_PATH,
        augmentations=None,
        use_oba=True,
        oba_prob=1.0,
        visualize=True,
        num_oba_objects=5,
        extract_from_same_image=True
    )

    # Visualize how the OBA pipeline works
    visualize_both_samples(original_dataset, oba_dataset, index=1)
    
    # Visualize a sample from the background_images folder, uncomment the line below to try out
    ### visualize_external_background_pair(data_root=DATASET_PATH, annotations_path=TRAIN_ANNOTATIONS_PATH, background_root=SEPARATE_BACKGROUND_IMAGES, sample_indices=sample_indices, index=1, num_oba_objects=5)

