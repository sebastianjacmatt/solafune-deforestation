import os

# Get the absolute path of the project's root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Paths to the dataset
DATASET_PATH = os.path.join(BASE_DIR, "data")
TRAIN_IMAGES_PATH = os.path.join(DATASET_PATH, "train_images")
EVAL_IMAGES_PATH = os.path.join(DATASET_PATH, "evaluation_images")
TRAIN_ANNOTATIONS_PATH = os.path.join(DATASET_PATH, "train_annotations.json")

# Outputs and models
OUTPUTS_PATH = os.path.join(BASE_DIR, "outputs")
MODELS_PATH = os.path.join(BASE_DIR, "models")

# Ensure directories exist
os.makedirs(OUTPUTS_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)