import tifffile
import numpy as np
from PIL import Image

def preview_tif_band_image(filepath: str, band: list[int] = [1, 2, 3], pillow_mode:str = "RGB") :
    """
    Preview any .tif image with a specified band (default is 1,2,3), and pillow mode.

    Parameters:
    filepath (str): Path to the .tif file.
    band (list[int], optional): List of bands to display. Defaults to [1, 2, 3].

    See Also:
    https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED#bands
    for band specifications.
    """
    
    # Read the TIFF image into a NumPy array
    tif_data = tifffile.imread(filepath)
    #print(tif_data.shape)  # e.g. (height, width, 12)

    # selecting wanted bands of image
    selected_data = tif_data[:,:, band] # slicing into specified band
    

    selected_data = selected_data.astype("uint8") # really bad conversion here
    
    # Create a Pillow image in 'RGB' mode
    img = Image.fromarray(selected_data,pillow_mode) # RGB (3x8-bit pixels, true color)
    img.show()

preview_tif_band_image("data/train_images/train_20.tif",[3,2,1])
