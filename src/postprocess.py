import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(project_root, "src"))

import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import shape
from skimage import measure
from rasterio import features

from config import CLASS_NAMES, SCORE_THRESH, MIN_AREA

def detect_polygons(pred_dir, score_thresh=SCORE_THRESH, min_area=MIN_AREA):
    pred_dir = Path(pred_dir)
    pred_paths = sorted(pred_dir.glob("*.npy"))

    polygons_all_imgs = {}
    for pred_path in tqdm(pred_paths, desc="Detect Polygons"):
        mask = np.load(pred_path)  # shape: (4, 1024, 1024)
        mask = mask > score_thresh

        polygons_for_img = {}
        for i, class_name in enumerate(CLASS_NAMES):
            mask_class = mask[i]
            # Remove small areas
            if mask_class.sum() < min_area:
                mask_class = np.zeros_like(mask_class)

            label = measure.label(mask_class, connectivity=2, background=0).astype(np.uint8)
            polygons_list = []
            for p, value in features.shapes(label, label=label):
                # shapely geometry
                poly = shape(p).buffer(0.5).simplify(tolerance=0.5)
                polygons_list.append(poly)
            polygons_for_img[class_name] = polygons_list

        # record polygons
        polygons_all_imgs[pred_path.name.replace(".npy", ".tif")] = polygons_for_img

    return polygons_all_imgs

def generate_submission(test_pred_polygons, save_path):
    images = []
    for file_name in sorted(test_pred_polygons.keys()):
        annotations = []
        for class_name in CLASS_NAMES:
            for poly in test_pred_polygons[file_name][class_name]:
                seg = []
                for xy in poly.exterior.coords:
                    seg.extend(xy)
                annotations.append({"class": class_name, "segmentation": seg})
        images.append({"file_name": file_name, "annotations": annotations})

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"images": images}, f, indent=4)
