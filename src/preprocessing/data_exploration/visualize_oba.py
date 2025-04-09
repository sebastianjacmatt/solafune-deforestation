import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
sys.path.append(os.path.join(project_root, "src"))
print(project_root)


import matplotlib.pyplot as plt

def to_rgb(img):
    """
    Convert a 12-channel image (C, H, W) to an RGB image using Sentinel-2 natural color.
    For Sentinel-2, natural color is typically defined as:
       Red = channel 3, Green = channel 2, Blue = channel 1.
    The image is assumed to be in channels-first format.
    """
    rgb = img[[3, 2, 1], :, :]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    return rgb.transpose(1, 2, 0)

def visualize_both_samples(original_dataset, oba_dataset, index=0):
    sample_orig = original_dataset[index]
    sample_oba = oba_dataset[index]
    
    rgb_orig = to_rgb(sample_orig["image"])
    rgb_oba = to_rgb(sample_oba["image"])
    
    mask_orig = sample_orig["mask"].transpose(1, 2, 0)
    mask_oba = sample_oba["mask"].transpose(1, 2, 0)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Top left: Original RGB composite
    axes[0, 0].imshow(rgb_orig)
    axes[0, 0].set_title("Original RGB Composite")
    axes[0, 0].axis("off")
    
    # Top right: Original image with mask overlay
    axes[0, 1].imshow(rgb_orig)
    colormaps = [plt.cm.Reds, plt.cm.Greens, plt.cm.Blues, plt.cm.Oranges]
    for i in range(mask_orig.shape[-1]):
        axes[0, 1].imshow(mask_orig[:, :, i], cmap=colormaps[i], alpha=0.5)
    axes[0, 1].set_title("Original with Mask Overlay")
    axes[0, 1].axis("off")
    
    # Bottom left: OBA RGB composite
    axes[1, 0].imshow(rgb_oba)
    axes[1, 0].set_title("OBA RGB Composite")
    axes[1, 0].axis("off")
    
    # Bottom right: OBA image with mask overlay and highlighted pasted objects
    axes[1, 1].imshow(rgb_oba)
    for i in range(mask_oba.shape[-1]):
        axes[1, 1].imshow(mask_oba[:, :, i], cmap=colormaps[i], alpha=0.5)
    
    # Draw bounding boxes for the pasted objects if available.
    bboxes = sample_oba.get("oba_bbox", None)
    if bboxes is not None:
        # If it's a list of bounding boxes, iterate over each one.
        if isinstance(bboxes, list):
            for bbox in bboxes:
                if bbox is not None:  # sometimes you might get None if pasting was skipped
                    top, left, h_obj, w_obj = bbox
                    rect = plt.Rectangle((left, top), w_obj, h_obj, edgecolor='magenta', facecolor='none', linewidth=2)
                    axes[1, 1].add_patch(rect)
        else:
            # If it's a single bounding box tuple.
            top, left, h_obj, w_obj = bboxes
            rect = plt.Rectangle((left, top), w_obj, h_obj, edgecolor='magenta', facecolor='none', linewidth=2)
            axes[1, 1].add_patch(rect)
    
    axes[1, 1].set_title("OBA with Mask and Highlight")
    axes[1, 1].axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from dataset import TrainValDataset, OBAValDataset
    from global_paths import DATASET_PATH

    sample_indices = list(range(10))  # Example indices
    annotations_path = DATASET_PATH / "train_annotations.json"
    
    # Create the original dataset (without OBA augmentation)
    original_dataset = TrainValDataset(data_root=DATASET_PATH, sample_indices=sample_indices, augmentations=None)
    
    # Create the OBA dataset (with cut-and-paste augmentation and visualization enabled)
    oba_dataset = OBAValDataset(
        data_root=DATASET_PATH,
        sample_indices=sample_indices,
        annotations_path=annotations_path,
        augmentations=None,
        use_oba=True,
        oba_prob=1.0,
        visualize=True,  # Enable visualization mode
        num_oba_objects=5,
    )
    
    # Visualize both samples (the same index from original and OBA datasets)
    visualize_both_samples(original_dataset, oba_dataset, index=1)
