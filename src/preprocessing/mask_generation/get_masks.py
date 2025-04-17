
import json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
sys.path.append(os.path.join(project_root, "src"))

src_root = os.path.abspath(os.path.join(project_root, "src/"))
sys.path.append(os.path.join(src_root, "utils"))

from global_paths import DATASET_PATH


train_file_names = [f"train_{i}.tif" for i in range(176)]  # train_0.tif ~ train_175.tif
class_names = ["grassland_shrubland", "logging", "mining", "plantation"]

# Ensure DATASET_PATH is a Path object before using "/"
DATASET_PATH = Path(DATASET_PATH)  # Convert string to Path

mask_save_dir = DATASET_PATH / "train_masks"
mask_save_dir.mkdir(parents=True, exist_ok=True)

def get_and_save_masks():
    with open(DATASET_PATH / "train_annotations.json", "r") as f:
        raw_annotations = json.load(f)

    annotations: dict[str, dict[str, list[list[float]]]] = {}  # file_name -> class_name -> polygons
    for fn in tqdm(train_file_names):
        ann: dict[str, list[list[float]]] = {}  # class_name -> polygons
        for class_name in class_names:
            ann[class_name] = []

        for tmp_img in raw_annotations["images"]:
            if tmp_img["file_name"] == fn:
                for tmp_ann in tmp_img["annotations"]:
                    ann[tmp_ann["class"]].append(tmp_ann["segmentation"])

        annotations[fn] = ann

    for fn in tqdm(train_file_names):
        mask = np.zeros((4, 1024, 1024), dtype=np.uint8)
        anns = annotations[fn]
        for class_idx, class_name in enumerate(class_names):
            polygons = anns[class_name]
            cv2.fillPoly(mask[class_idx], [np.array(poly).astype(np.int32).reshape(-1, 2) for poly in polygons], 255)

        np.save(mask_save_dir / fn.replace(".tif", ".npy"), mask)