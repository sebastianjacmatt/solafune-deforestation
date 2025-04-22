import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), ""))
sys.path.append(os.path.join(project_root, "src"))

src_root = os.path.abspath(os.path.join(os.getcwd(), "src/"))
sys.path.append(os.path.join(src_root, "utils"))

import torch
torch.set_float32_matmul_precision("high")

from train_utils import train_model
from global_paths import VAL_PRED_DIR, TEST_PRED_DIR, SUBMISSION_SAVE_PATH, DATASET_PATH
from config import SCORE_THRESH, MIN_AREA, NUM_WORKERS_TEST, BATCH_SIZE_TEST
from dataset import TestDataset
from torch.utils.data import DataLoader
from inference_utils import run_inference
from postprocess import PostProcess

def main():
    # 1) Train
    model, train_loader, val_loader = train_model(use_oba=False)


    # 2) Inference on val set
    run_inference(model, val_loader, VAL_PRED_DIR)

    # 3) Inference on test set
    test_dataset = TestDataset(DATASET_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, num_workers=NUM_WORKERS_TEST, shuffle=False)
    run_inference(model, test_loader, TEST_PRED_DIR)
    
    # 4) Postprocess (detect polygons, generate JSON)
    post = PostProcess(
        pred_dir=VAL_PRED_DIR,
        score_thresh=SCORE_THRESH,
        min_area=MIN_AREA,
        save_path=SUBMISSION_SAVE_PATH,
    )

    post.generate_submission()

if __name__ == "__main__":
    main()