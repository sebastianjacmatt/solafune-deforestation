import tifffile
import numpy as np
from PIL import Image

def preview_tif_band_image(filepath: str, band: list[int] = [1, 2, 3], pillow_mode: str = "RGB"):
    """
    Preview any .tif image with a specified band (default is 1,2,3), and pillow mode.

    Parameters:
    filepath (str): Path to the .tif file.
    band (list[int], optional): List of bands to display. Defaults to [1, 2, 3].
    pillow_mode (str, optional): Mode for PIL.Image. Defaults to "RGB".
    """

    # Read the TIFF image into a NumPy array
    tif_data = tifffile.imread(filepath)  # Shape: (height, width, num_bands)

    # Convert band indices from 1-based to 0-based
    band = [b - 1 for b in band]

    # Select the desired bands
    selected_data = tif_data[:, :, band].astype(np.float32)  # Convert to float32 for scaling

    # Normalize each band independently to [0, 255]
    for i in range(selected_data.shape[2]):
        min_val = np.percentile(selected_data[:, :, i], 2)  # Avoid outliers (2nd percentile)
        max_val = np.percentile(selected_data[:, :, i], 98) # Avoid outliers (98th percentile)
        selected_data[:, :, i] = np.clip((selected_data[:, :, i] - min_val) / (max_val - min_val) * 255, 0, 255)

    # Convert to uint8
    selected_data = selected_data.astype(np.uint8)

    # Create a Pillow image
    img = Image.fromarray(selected_data, pillow_mode)
    img.show()

# Example usage: Show True Color (RGB) Image using Sentinel-2 Bands 4, 3, 2
preview_tif_band_image("../../data/train_images/train_20.tif", [4, 3, 2])