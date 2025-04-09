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

from config import CLASS_NAMES, SCORE_THRESH, MIN_AREA, NUM_EVAL_INDICIES

class PostProcess:
    def __init__(self, pred_dir, score_thresh, min_area, save_path):
        self.pred_dir = pred_dir
        self.score_thresh = score_thresh
        self.min_area = min_area
        self.save_path = save_path

    def generate_submission(self, save_path=None):
        if save_path is None:
            save_path = self.save_path
        pred_paths = sorted(self.pred_dir.glob("*.npy"))

        with open(save_path, "w", encoding="utf-8") as f:
            f.write('{"images": [\n')

            for i, image_entry in enumerate(self.stream_image_entries(pred_paths)):
                json.dump(image_entry, f, indent=4)
                if i < len(pred_paths) - 1:
                    f.write(",\n")
                else:
                    f.write("\n")

            f.write("]}\n")  # Close the JSON object

    def stream_image_entries(self, pred_paths):
        for pred in tqdm(pred_paths, desc="Detect Polygons", total=len(pred_paths)):
            mask = np.load(pred, mmap_mode='r') # only read, saves memory
            image_segments = self.generate_segment_polygons(mask)
            annotations = self.build_annotations(image_segments)
            pred_name = pred.name.replace(".npy", ".tif")
            yield {
                "file_name": pred_name,
                "annotations": annotations
            }

    def build_annotations(self, image_segments):
        annotations = []
        for class_name in CLASS_NAMES:
            for poly in image_segments.get(class_name, []):
                seg = [int(round(coord)) for xy in poly.exterior.coords for coord in xy] # list comprehension for more efficient rounding
                annotations.append({
                    "class": class_name,
                    "segmentation": seg
                })
        return annotations


    def generate_segment_polygons(self,mask)-> dict:
        """
        Generates the polygons for for different classes from the mask.
        Args:
            mask (numpy.ndarray): The predicted mask for a specific class.
        Returns:
            list: A list of polygons for the detected objects in the mask.
        """
        polygons_all_classes = {} # polygon for all classes within an image
        for i, class_name in enumerate(CLASS_NAMES):
            mask_for_a_class = mask[i]
            if mask_for_a_class.astype(np.int64).sum() < self.min_area:
                mask_for_a_class = np.zeros_like(mask_for_a_class)  # set all to zero if the predicted area is less than `min_area`
            # extract polygons from the binarized mask
            label = measure.label(mask_for_a_class, connectivity=2, background=0).astype(np.uint8)
            polygons = []
            for p, value in features.shapes(label, label):
                p = shape(p)
                if not p.is_valid:
                    continue
                #p = p.simplify(tolerance=0.5)
                polygons.append(p)
            polygons_all_classes[class_name] = polygons
        return polygons_all_classes

