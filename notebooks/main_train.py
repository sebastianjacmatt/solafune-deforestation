import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(project_root, "src"))

import torch
torch.set_float32_matmul_precision("high")
from train_utils import train_model
from global_paths import TRAIN_OUTPUT_DIR, VAL_PRED_DIR, TEST_PRED_DIR, SUBMISSION_SAVE_PATH, DATASET_PATH
from model import Model
from config import SCORE_THRESH, MIN_AREA
from dataset import TestDataset
from torch.utils.data import DataLoader
from inference_utils import run_inference
from postprocess import detect_polygons, generate_submission

def main():
    # 1) Train
    model, train_loader, val_loader, train_indices, val_indices = train_model()


    # 2) Inference on val set
    run_inference(model, val_loader, VAL_PRED_DIR)

    # 3) Inference on test set
    test_dataset = TestDataset(DATASET_PATH)
    test_loader = DataLoader(test_dataset, batch_size=4, num_workers=8, shuffle=False)
    run_inference(model, test_loader, TEST_PRED_DIR)

    # 4) Postprocess (detect polygons, generate JSON)
    test_pred_polygons = detect_polygons(
        pred_dir=TEST_PRED_DIR,
        score_thresh=SCORE_THRESH,
        min_area=MIN_AREA
    )
    generate_submission(test_pred_polygons, SUBMISSION_SAVE_PATH)

if __name__ == "__main__":
    main()