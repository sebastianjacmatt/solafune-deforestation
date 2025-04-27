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
def plot_domain_interp(dom1, dom2, z_interp, channel_groups, band_names=None, save_path=None, show=True):
    """
    Plot and optionally save the domain images along with the interpolated image.

    Parameters
    ----------
    dom1 : np.ndarray
        Domain 1 image (C, H, W).
    dom2 : np.ndarray
        Domain 2 image (C, H, W).
    z_interp : np.ndarray
        Interpolated image (C, H, W).
    channel_groups : list of list of int
        List of channel indices for each domain.
    band_names : list of str
        Names of the spectral bands.
    save_path : str or None
        If set, saves the figure to this path.
    show : bool
        If True, displays the plot; otherwise closes it.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if band_names is None:
        band_names = [
            'B1(Aerosols)', 'B2(Blue)', 'B3(Green)', 'B4(Red)',
            'B5(Red Edge 1)', 'B6(Red Edge 2)', 'B7(Red Edge 3)',
            'B8(NIR)', 'B8A(Red Edge 4)', 'B9(Water vapor)', 'B11(SWIR 1)', 'B12(SWIR 2)'
        ]

    # We'll plot the three sets side by side with true-color composites
    # Choose bands for RGB: [3,2,1]
    def to_rgb(img):
        rgb = np.stack([img[3], img[2], img[1]], axis=-1)
        return (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

    rgb1 = to_rgb(dom1)
    rgb2 = to_rgb(dom2)
    rgbz = to_rgb(z_interp)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb1)
    axes[0].set_title("Domain 1")
    axes[0].axis("off")

    axes[1].imshow(rgb2)
    axes[1].set_title("Domain 2")
    axes[1].axis("off")

    axes[2].imshow(rgbz)
    axes[2].set_title("Interpolated")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()
    return fig
    
