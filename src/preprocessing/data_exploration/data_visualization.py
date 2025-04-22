import rasterio
import matplotlib.pyplot as plt
import json
import geopandas as gpd
import numpy as np

import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(os.path.join(project_root, "src"))



from paths import DATASET_PATH, TRAIN_IMAGES_PATH, EVAL_IMAGES_PATH, TRAIN_ANNOTATIONS_PATH

from utilities import convert_to_geojson


# Load annotations
with open(TRAIN_ANNOTATIONS_PATH, 'r') as file:
    train_annotations = json.load(file)

# Sample TIF path
SAMPLE_TIF_PATH = f'{TRAIN_IMAGES_PATH}/train_0.tif'

# Open TIF file
src = rasterio.open(SAMPLE_TIF_PATH)

# Checking metadata
print(src.meta.keys())

# Checking band count
band_count = src.count
print(f"#Bands: {band_count}")

# Display band information
for i in range(band_count):
    print(f"Band {i+1}: {src.descriptions[i] if src.descriptions[i] else 'No description'}")

# Create subplots to show all bands in one figure
fig, axes = plt.subplots(1, band_count, figsize=(15, 5))
fig.suptitle('All Bands Visualization', fontsize=16)

for i in range(band_count):
    band_data = src.read(i + 1)
    axes[i].imshow(band_data)  # Use cmap='gray' for grayscale bands
    axes[i].set_title(f'Band {i+1}')
    axes[i].axis('off')  # Hide axis for better visualization

plt.tight_layout()
plt.show()


# Sample annotation data
annotation_data = train_annotations['images'][0]['annotations']
print([d['class'] for d in annotation_data])

# Convert to GeoJSON
geojson_data = convert_to_geojson(annotation_data)

# Print the GeoJSON data
print(json.dumps(geojson_data))

gdf = gpd.GeoDataFrame.from_features(geojson_data)
 
gdf.head(10)

fig, (ax1) = plt.subplots(1, 1, figsize=(12,6))

# Plot raster
ax1.imshow(src.read(8))
ax1.set_title('plantation area')

# Plot field segmentation
gdf.boundary.plot(ax=ax1, color='red')


plt.show()