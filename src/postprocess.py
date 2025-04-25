import sys
import os
import numpy as np
import json
import os
from tqdm import tqdm
from shapely.geometry import shape
from skimage import measure
from rasterio import features
import subprocess

# Append project paths
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(project_root, "src"))

src_root = os.path.abspath(os.path.join(project_root, "src/"))
sys.path.append(os.path.join(src_root, "utils"))

from config import CLASS_NAMES

class PostProcess:
    """
    Handles post-processing of prediction masks to generate polygon annotations for submission.
    """
    def __init__(self, pred_dir, test_pred_dir, score_thresh, min_area, save_path):
        self.pred_dir = pred_dir
        self.test_pred_dir = test_pred_dir
        self.score_thresh = score_thresh
        self.min_area = min_area
        self.save_path = save_path

    def generate_submission(self, save_path=None):
        """
        Generates a JSON submission file with polygon annotations.
        """
        if save_path is None:
            save_path = self.save_path

        with open(save_path, "w", encoding="utf-8") as f:
            f.write('{"images": [\n')

            for idx, image_entry in enumerate(self.stream_image_entries()):
                json.dump(image_entry, f, indent=4)
                if idx < 117:
                    f.write(",\n")
                else:
                    f.write("\n")

            f.write("]}\n")  # Close the JSON object

        self.format_json_python_tool(save_path, save_path)

    def format_json_python_tool(self, input_path, output_path):
        """
        Formats the JSON file using the python json.tool module.
        """
        tmp_path = f"{output_path}.tmp"
        result = subprocess.run(["python", "-m", "json.tool", input_path, tmp_path])

        if result.returncode == 0:
            os.replace(tmp_path, output_path)
        else:
            print("Formatting failed. Original file kept.")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


    def stream_image_entries(self):
        """
        Streams JSON entries for all 118 evaluation files.
        """
        for idx in tqdm(range(118), desc="Detect Polygons"):
            pred_file = self.test_pred_dir / f"evaluation_{idx}.npy"

            if pred_file.exists():
                mask = np.load(pred_file, mmap_mode='r')
                image_segments = self.generate_segment_polygons(mask)
                annotations = self.build_annotations(image_segments)
            else:
                annotations = []

            yield {
                "file_name": f"evaluation_{idx}.tif",
                "annotations": annotations
            }


    def build_annotations(self, image_segments):
        """
        Converts polygons into formatted annotation dictionaries.
        Args:
            image_segments (dict): A dictionary where keys are class names and values are lists
        Returns:
            list: A list of annotation dictionaries, each containing the class name and 
                  a flattened list of polygon coordinates.
        """
        annotations = []
        for class_name in CLASS_NAMES:
            for poly in image_segments.get(class_name, []):
                seg = [int(round(coord)) for xy in poly.exterior.coords for coord in xy]
                annotations.append({
                    "class": class_name,
                    "segmentation": seg
                })
        return annotations


    def generate_segment_polygons(self, mask)-> dict:
        """
        Generates the polygons for for different classes from the mask.
        Args:
            mask (numpy.ndarray): The predicted mask for a specific class.
        Returns:
            list: A list of polygons for the detected objects in the mask.
        """
        polygons_all_classes = {}
        for i, class_name in enumerate(CLASS_NAMES):
            mask_for_a_class = mask[i]
            
            # Threshold the prediction into binary
            mask_for_a_class = (mask_for_a_class > self.score_thresh).astype(np.uint8)

            # Area filter AFTER threshold
            if mask_for_a_class.sum() < self.min_area:
                mask_for_a_class = np.zeros_like(mask_for_a_class)

            # Extract polygons
            label = measure.label(mask_for_a_class, connectivity=2, background=0).astype(np.uint8)
            polygons = []
            for p, value in features.shapes(label, label):
                p = shape(p).simplify(tolerance=0.5, preserve_topology=True)
                if not p.is_valid or p.area < 3 or len(p.exterior.coords) < 4:
                    continue
                polygons.append(p)

            polygons_all_classes[class_name] = polygons
        return polygons_all_classes
