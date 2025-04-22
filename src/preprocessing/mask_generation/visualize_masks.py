import cv2
import numpy as np
import tifffile
from tqdm import tqdm

import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
sys.path.append(os.path.join(project_root, "src"))

src_root = os.path.abspath(os.path.join(project_root, "src/"))
sys.path.append(os.path.join(src_root, "utils"))

from global_paths import DATASET_PATH, VISUALIZATIONS_PATH
from get_masks import train_file_names, class_names, mask_save_dir


# Visualize masks and save as a png file along with the RGB image
vis_save_dir = VISUALIZATIONS_PATH / "vis_train"
vis_save_dir.mkdir(parents=True, exist_ok=True)


def visualize_masks():
    for fn in tqdm(train_file_names):
        mask = np.load(mask_save_dir / fn.replace(".tif", ".npy"))  # (4, 1024, 1024)
        vis_masks = [np.zeros((1024, 1024, 3), dtype=np.uint8) for _ in range(4)]  # 4: (glassland_shrubland, logging, mining, plantation)
        for class_idx, class_name in enumerate(class_names):
            vis_masks[class_idx][mask[class_idx] > 0] = np.array([255, 0, 0])  # blue
            # put class_name as text on the mask
            cv2.putText(vis_masks[class_idx], class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        vis_image = tifffile.imread(DATASET_PATH / "train_images" / fn)
        vis_image = vis_image[:, :, [1, 2, 3]]  # extract BGR channels (B2, B3, and B4 band of Sentinel-2)
        vis_image = np.nan_to_num(vis_image, nan=0)
        vis_image = (vis_image / 8).clip(0, 255).astype(np.uint8)

        partition = np.ones((1024, 5, 3), dtype=np.uint8) * 255  # white partition
        vis = np.concatenate([vis_image, partition, vis_masks[0], partition, vis_masks[1], partition, vis_masks[2], partition, vis_masks[3]], axis=1)
        cv2.imwrite(str(vis_save_dir / fn.replace(".tif", ".png")), vis)