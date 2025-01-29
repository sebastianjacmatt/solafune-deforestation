import rasterio
import matplotlib.pyplot as plt
import json
import geopandas as gpd
import numpy as np

# Paths
DATASET_PATH = '../../data'
TRAIN_IMAGES_PATH = f'{DATASET_PATH}/train_images'
EVAL_IMAGES_PATH = f'{DATASET_PATH}/evaluation_images'
TRAIN_ANNOTATIONS_PATH = f'{DATASET_PATH}/train_annotations.json'

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


def convert_to_geojson(data):
  """
  Converts a list of dictionaries in the specified format to GeoJSON

  Args:
      data: A list of dictionaries containing 'class' and 'segmentation' keys

  Returns:
      A GeoJSON feature collection
  """
  features = []
  for item in data:
    polygon = []
    for i in range(0, len(item['segmentation']), 2):
      polygon.append([item['segmentation'][i], item['segmentation'][i+1]])
    features.append({
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [polygon]
      },
      "properties": {"class": item['class']}
    })
  return { "type": "FeatureCollection", "features": features}

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