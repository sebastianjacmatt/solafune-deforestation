import os
import sys
import json
import matplotlib.pyplot as plt
from collections import Counter

from global_paths import TRAIN_ANNOTATIONS_PATH

# Append project paths
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(os.path.join(project_root, "src"))
sys.path.append(os.path.join(project_root, "src/utils"))

# Load all annotations
with open(TRAIN_ANNOTATIONS_PATH, 'r') as file:
    data = json.load(file)

# Collect all classes
all_classes = []

for image in data['images']:
    for annotation in image['annotations']:
        all_classes.append(annotation['class'])

# Count occurrences of each class
class_counts = Counter(all_classes)

# Print the counts (optional)
for cls, count in class_counts.items():
    print(f"Class '{cls}': {count} instances")

# Plot histogram
plt.figure(figsize=(12, 6))
plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
plt.xlabel('Class')
plt.ylabel('Number of Instances')
plt.title('Class Distribution in Training Set')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
