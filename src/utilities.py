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