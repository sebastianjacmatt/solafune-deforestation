import sys
import os
import tifffile
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.patches as mpatches

# Append project paths
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(os.path.join(project_root, "src"))
sys.path.append(os.path.join(project_root, "src/utils"))

from config import CLASS_NAMES, MIN_AREA

# Define nice colors for each class
CLASS_COLORS = {
    "grassland_shrubland": "red",
    "logging": "blue",
    "mining": "green",
    "plantation": "yellow",
}

def load_image(image_path: str) -> np.ndarray:
    img = tifffile.imread(image_path)
    assert img.ndim == 3 and img.shape[2] == 12
    return np.nan_to_num(img).astype(np.float32)

def load_mask(mask_path: str) -> np.ndarray:
    m = np.load(mask_path)
    assert m.shape[0] == 4
    m = m.transpose(1, 2, 0)
    return m  # no /255

def plot_rgb_with_mask_per_class(image_path, mask_path, class_names, threshold=0.5, show=True, save_path=None):
    image = load_image(image_path)
    mask = load_mask(mask_path)

    rgb = np.stack([
        image[..., 3],
        image[..., 2],
        image[..., 1],
    ], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))  # 2x2 grid
    axes = axes.flatten()

    for idx, (i, class_name) in enumerate(enumerate(class_names)):
        ax = axes[idx]

        mask_i = (mask[..., i] > threshold).astype(np.float32)
        if mask_i.sum() < MIN_AREA:
            ax.set_title(f"{class_name} (Skipped, Area < {MIN_AREA})")
            ax.imshow(rgb)
            ax.axis("off")
            continue

        color = CLASS_COLORS.get(class_name, "red")
        rgba = plt.matplotlib.colors.to_rgba(color)
        colored_mask = np.zeros((*mask_i.shape, 4))  # RGBA
        colored_mask[..., :3] = rgba[:3]
        colored_mask[..., 3] = mask_i * 0.4

        ax.imshow(rgb)
        ax.imshow(colored_mask)
        ax.set_title(f"{class_name}", fontsize=14)
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()

    return fig

# Example usage
if __name__ == "__main__":
    SAMPLE_IMAGE = os.path.join(project_root, "data/train_images", "train_0.tif")
    SAMPLE_MASK  = os.path.join(project_root, "outputs/predictions/val_preds", "train_0.npy")
    plot_rgb_with_mask_per_class(
        image_path=SAMPLE_IMAGE,
        mask_path=SAMPLE_MASK,
        class_names=CLASS_NAMES,
        threshold=0.9,
        show=True,
        save_path=None
    )
