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

import concurrent.futures

from config import CLASS_NAMES, SCORE_THRESH, MIN_AREA, NUM_EVAL_INDICIES


def detect_polygons(pred_dir, score_thresh, min_area):
    pred_dir = Path(pred_dir)
    pred_paths = list(pred_dir.glob("*.npy"))
    pred_paths = sorted(pred_paths)

    polygons_all_imgs = {}
    for pred_path in tqdm(pred_paths, desc="Detect Polygons"):
        polygons_all_classes = {}

        mask = np.load(pred_path)  # (4, 1024, 1024)
        mask = mask > score_thresh  # binarize
        for i, class_name in enumerate(CLASS_NAMES):
            mask_for_a_class = mask[i]
            if mask_for_a_class.sum() < min_area:
                mask_for_a_class = np.zeros_like(mask_for_a_class)  # set all to zero if the predicted area is less than `min_area`

            # extract polygons from the binarized mask
            label = measure.label(mask_for_a_class, connectivity=2, background=0).astype(np.uint8)
            polygons = []
            for p, value in features.shapes(label, label):
                # p = shape(p).buffer(0.5)
                p = p.simplify(tolerance=0.5)
                polygons.append(p)
            polygons_all_classes[class_name] = polygons
        polygons_all_imgs[pred_path.name.replace(".npy", ".tif")] = polygons_all_classes

    return polygons_all_imgs

def generate_submission(test_pred_polygons, save_path):
    images = []
    for img_id in range(NUM_EVAL_INDICIES):  # evaluation_0.tif to evaluation_117.tif
        annotations = []
        for class_name in CLASS_NAMES:
            for poly in test_pred_polygons[f"evaluation_{img_id}.tif"][class_name]:
                seg: list[float] = []  # [x0, y0, x1, y1, ..., xN, yN]
                for xy in poly.exterior.coords:
                    seg.extend(xy)

                annotations.append({"class": class_name, "segmentation": seg})

        images.append({"file_name": f"evaluation_{img_id}.tif", "annotations": annotations})

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"images": images}, f, indent=4)