import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(project_root, "src"))

import os
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

def run_inference(model, loader, pred_output_dir):
    pred_output_dir = Path(pred_output_dir)
    pred_output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            imgs = batch["image"].to(device)  # move input to GPU/CPU
            logits = model(imgs)
            probs = logits.sigmoid()

            for i in range(imgs.size(0)):
                file_name = os.path.basename(batch["image_path"][i])
                prob_mask = probs[i].cpu().numpy()  # back to numpy
                np.save(
                    pred_output_dir / file_name.replace(".tif", ".npy"),
                    prob_mask.astype(np.float16)
                )

