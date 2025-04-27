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

def plot_domain_split(orig, dom1, dom2, channel_groups, band_names=None, save_path=None, show=True):
    """
    Plot and optionally save the original and domain-split images.

    Parameters
    ----------
    orig : np.ndarray
        Original image (C, H, W) format.
    dom1 : np.ndarray
        Domain 1 image (C, H, W).
    dom2 : np.ndarray
        Domain 2 image (C, H, W).
    channel_groups : list of list of int
        List of channel indices for each domain.
    band_names : list of str
        Names of the spectral bands.
    save_path : str or None
        If set, saves the figure to this path.
    show : bool
        If True, displays the plot (use False when just saving or returning).

    Returns
    -------
    matplotlib.figure.Figure
        The Matplotlib figure object (can be saved or modified later).
    """

    if band_names is None:
        band_names = [
            'B1(Aerosols)', 'B2(Blue)', 'B3(Green)', 'B4(Red)',
            'B5(Red Edge 1)', 'B6(Red Edge 2)', 'B7(Red Edge 3)',
            'B8(NIR)', 'B8A(Red Edge 4)', 'B9(Water vapor)', 'B11(SWIR 1)', 'B12(SWIR 2)'
        ]

    n_bands = orig.shape[0]
    fig, axes = plt.subplots(3, n_bands, figsize=(n_bands * 1.5, 6))

    # Create domain-specific labels
    dom1_labels = []
    dom2_labels = []

    for ch in range(n_bands):
        if ch in channel_groups[0]:
            dom1_labels.append(f"{band_names[ch]}")
        else:
            repeated_from = band_names[channel_groups[0][ch % len(channel_groups[0])]]
            dom1_labels.append(f"{repeated_from}")

        if ch in channel_groups[1]:
            dom2_labels.append(f"{band_names[ch]}")
        else:
            repeated_from = band_names[channel_groups[1][ch % len(channel_groups[1])]]
            dom2_labels.append(f"{repeated_from}")

    for b in range(n_bands):
        # Row 0: original band
        axes[0, b].imshow(orig[b], cmap="gray")
        axes[0, b].axis("off")
        axes[0, b].set_title(band_names[b], fontsize=8)

        # Row 1: domain 1
        axes[1, b].imshow(dom1[b], cmap="gray")
        axes[1, b].axis("off")
        axes[1, b].set_title(dom1_labels[b], fontsize=7)

        # Row 2: domain 2
        axes[2, b].imshow(dom2[b], cmap="gray")
        axes[2, b].axis("off")
        axes[2, b].set_title(dom2_labels[b], fontsize=7)

    # Add row labels
    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Domain 1", fontsize=10)
    axes[2, 0].set_ylabel("Domain 2", fontsize=10)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()

    return fig

if __name__ == "__main__":
    SAMPLE_IMAGE = os.path.join(project_root, "data/train_images", "train_0.tif")
    SAMPLE_MASK  = os.path.join(project_root, "data/train_masks", "train_0.npy")
    
    # Load data
    img = load_image(SAMPLE_IMAGE)
    msk = load_mask(SAMPLE_MASK)
    
    # Spectral domains
    channel_groups = [
        [1, 2, 3],   # Domain 1
        [4, 5, 6],   # Domain 2
    ]
    
    samples = domain_image_split(img, msk, channel_groups, padding="repetition")
    
    orig = img.transpose(2, 0, 1)
    dom1 = samples[0]["image"].numpy()
    dom2 = samples[1]["image"].numpy()
    
    band_names = [
        'B1(Aerosols)', 'B2(Blue)', 'B3(Green)', 'B4(Red)',
        'B5(Red Edge 1)', 'B6(Red Edge 2)', 'B7(Red Edge 3)',
        'B8(NIR)', 'B8A(Red Edge 4)', 'B9(Water vapor)', 'B11(SWIR 1)', 'B12(SWIR 2)'
    ]
    
    # Plot and show immediately
    plot_domain_split(
        orig=orig,
        dom1=dom1,
        dom2=dom2,
        channel_groups=channel_groups,
        band_names=band_names,
        save_path=None,  # Don't save when running directly
        show=True
    )
