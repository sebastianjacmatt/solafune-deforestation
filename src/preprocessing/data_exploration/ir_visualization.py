import sys
import os
import tifffile
import matplotlib.pyplot as plt
import numpy as np

# Append project paths
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(os.path.join(project_root, "src"))
sys.path.append(os.path.join(project_root, "src/utils"))

from data_utils import domain_image_split

def load_image(image_path: str) -> np.ndarray:
    """Load a (H, W, 12) float32 TIF."""
    img = tifffile.imread(image_path)
    assert img.ndim == 3 and img.shape[2] == 12
    return np.nan_to_num(img).astype(np.float32)

def load_mask(mask_path: str) -> np.ndarray:
    """Load a (4, H, W) mask npy → (H, W, 4) [0–1] float32."""
    m = np.load(mask_path)
    assert m.shape[0] == 4
    m = m.transpose(1, 2, 0)
    return (m.astype(np.float32) / 255.0)

if __name__ == "__main__":
    SAMPLE_IMAGE = os.path.join(project_root, "data/train_images", "train_0.tif")
    SAMPLE_MASK  = os.path.join(project_root, "data/train_masks", "train_0.npy")
    
    # Load data
    img  = load_image(SAMPLE_IMAGE)
    msk  = load_mask(SAMPLE_MASK)
    
    # Spectral domains
    channel_groups = [
        [1,2,3],
        [4,5,6],
    ]
    
    samples = domain_image_split(img, msk, channel_groups, padding="zeroing")
    
    orig = img.transpose(2, 0, 1)      
    dom1 = samples[0]["image"].numpy() 
    dom2 = samples[1]["image"].numpy() 
    
    # Plot original + both domains: 3 rows × 12 columns
    n_bands = orig.shape[0]
    fig, axes = plt.subplots(3, n_bands, figsize=(n_bands * 1.5, 6))
    
    for b in range(n_bands):
        # Row 0: original band
        axes[0, b].imshow(orig[b], cmap="gray")
        axes[0, b].axis("off")
        if b == 0:
            axes[0, b].set_title("Original")
        
        # Row 1: domain 1
        axes[1, b].imshow(dom1[b], cmap="gray")
        axes[1, b].axis("off")
        if b == 0:
            axes[1, b].set_title("Domain 1")
        
        # Row 2: domain 2
        axes[2, b].imshow(dom2[b], cmap="gray")
        axes[2, b].axis("off")
        if b == 0:
            axes[2, b].set_title("Domain 2")
    
    plt.tight_layout()
    plt.show()
