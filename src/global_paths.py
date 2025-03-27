from pathlib import Path

# Get the base directory of the project (one level above `src`)
BASE_DIR = Path(__file__).resolve().parent.parent

# Define dataset paths
DATASET_PATH = BASE_DIR / "data"
TRAIN_IMAGES_PATH = DATASET_PATH / "train_images"
EVAL_IMAGES_PATH = DATASET_PATH / "evaluation_images"
TRAIN_ANNOTATIONS_PATH = DATASET_PATH / "train_annotations.json"
SAMPLE_ANSWER_PATH = DATASET_PATH / "sample_answer.json"

# Define other paths
TRAIN_OUTPUT_DIR = DATASET_PATH / "training_result"
VAL_PRED_DIR = DATASET_PATH / "val_preds"
TEST_PRED_DIR = DATASET_PATH / "test_preds"
MODELS_PATH = BASE_DIR / "src" / "models"
OUTPUTS_PATH = BASE_DIR / "src" / "outputs"
PREDICTIONS_PATH = BASE_DIR / "src" / "outputs" / "predictions"
VISUALIZATIONS_PATH = BASE_DIR / "src" / "outputs" / "visualizations"
SUBMISSIONS_PATH = BASE_DIR / "src" / "outputs" / "submissions"
SUBMISSION_SAVE_PATH = SUBMISSIONS_PATH / "submission.json"

# Ensure directories exist where needed
OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)