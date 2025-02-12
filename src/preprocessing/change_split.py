import os
import shutil
import random
import time
from pathlib import Path
import sys

# Get the absolute path of the project's root directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__) + "/../..")
sys.path.insert(0, BASE_DIR)  # Add project root to Python path

from src.paths import TRAIN_IMAGES_PATH, EVAL_IMAGES_PATH

# Define split percentage
TRAIN_SPLIT = 0.92  # 80% train, 20% validation

def get_sorted_files(directory, prefix):
    """Return a sorted list of file names matching the prefix (e.g., train_ or evaluation_)."""
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".tif")]
    return sorted(files, key=lambda x: int(x.split("_")[1].split(".")[0]))  # Sort numerically

def unlock_file(file_path):
    """Try to unlock a file if it's being used by another process."""
    retries = 5  # Number of retry attempts
    for _ in range(retries):
        try:
            with open(file_path, "rb") as f:
                pass  # Try to open the file in read mode
            return True  # If successful, file is unlocked
        except Exception as e:
            print(f"⚠️ WARNING: {file_path} is locked. Retrying in 2 seconds...")
            time.sleep(2)  # Wait and retry
    return False  # File is still locked after retries

def move_and_copy():
    """Copy files instead of moving to prevent data loss, with file unlocking."""
    os.makedirs(TRAIN_IMAGES_PATH, exist_ok=True)
    os.makedirs(EVAL_IMAGES_PATH, exist_ok=True)

    all_eval_files = get_sorted_files(EVAL_IMAGES_PATH, "evaluation_")
    all_train_files = get_sorted_files(TRAIN_IMAGES_PATH, "train_")

    all_files = all_eval_files + all_train_files
    random.shuffle(all_files)

    new_train_size = int(len(all_files) * TRAIN_SPLIT)
    new_eval_size = len(all_files) - new_train_size

    print(f"\n🔄 Total images before copy: {len(all_files)}")
    print(f"➡️ Assigning {new_train_size} to training and {new_eval_size} to evaluation.")

    missing_files = []

    print("\n📌 Copying Training Images:")
    for i, file in enumerate(all_files[:new_train_size]):
        old_path = (EVAL_IMAGES_PATH if "evaluation_" in file else TRAIN_IMAGES_PATH) / file
        new_path = TRAIN_IMAGES_PATH / f"train_{i}.tif"
        
        if not old_path.exists():
            print(f"⚠️ ERROR: File not found {old_path}")
            missing_files.append(file)
            continue

        if not unlock_file(old_path):
            print(f"❌ ERROR: {old_path} is still locked! Skipping.")
            missing_files.append(file)
            continue

        try:
            shutil.copy2(old_path, new_path)
        except Exception as e:
            print(f"⚠️ ERROR copying {file}: {e}")
            missing_files.append(file)

    print("\n📌 Copying Evaluation Images:")
    for i, file in enumerate(all_files[new_train_size:]):
        old_path = (EVAL_IMAGES_PATH if "evaluation_" in file else TRAIN_IMAGES_PATH) / file
        new_path = EVAL_IMAGES_PATH / f"evaluation_{i}.tif"
        
        if not old_path.exists():
            print(f"⚠️ ERROR: File not found {old_path}")
            missing_files.append(file)
            continue

        if not unlock_file(old_path):
            print(f"❌ ERROR: {old_path} is still locked! Skipping.")
            missing_files.append(file)
            continue

        try:
            shutil.copy2(old_path, new_path)
        except Exception as e:
            print(f"⚠️ ERROR copying {file}: {e}")
            missing_files.append(file)

    train_files_after = get_sorted_files(TRAIN_IMAGES_PATH, "train_")
    eval_files_after = get_sorted_files(EVAL_IMAGES_PATH, "evaluation_")

    print(f"\n✅ Final count - Training images: {len(train_files_after)} (expected: {new_train_size})")
    print(f"✅ Final count - Evaluation images: {len(eval_files_after)} (expected: {new_eval_size})")

    if missing_files:
        print(f"\n⚠️ WARNING: {len(missing_files)} files were not copied successfully!")
        for f in missing_files:
            print(f"  - {f}")

# Run the function (with file unlocking)
move_and_copy()
