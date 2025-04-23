import sys
import os
import json
import geopandas as gpd

from global_paths import TRAIN_ANNOTATIONS_PATH

# Append project paths
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(os.path.join(project_root, "src"))
sys.path.append(os.path.join(project_root, "src/utils"))

with open(TRAIN_ANNOTATIONS_PATH) as f:
    data = json.load(f)

json_data = data["images"][1]["annotations"]

# Taken from https://solafune.com/competitions/68ad4759-4686-4bb3-94b8-7063f755b43d?menu=discussion&tab=&page=3&topicId=d689d4a8-a939-4f0e-87bb-273707e8263f
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
