from pathlib import Path

# Get the base directory of the project (one level above `src`)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Define dataset paths
DATASET_PATH = BASE_DIR / "data"
TRAIN_IMAGES_PATH = DATASET_PATH / "train_images"
EVAL_IMAGES_PATH = DATASET_PATH / "evaluation_images"
TRAIN_ANNOTATIONS_PATH = DATASET_PATH / "train_annotations.json"


# Define other paths

# Ensure directories exist where needed
SRC_PATH = BASE_DIR / "src"
OUTPUTS_PATH = SRC_PATH / "outputs"
MODELS_PATH = SRC_PATH / "models"

OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = OUTPUTS_PATH / "predictions"

TRAIN_OUTPUT_DIR = PREDICTIONS_PATH / "training_result"
VAL_PRED_DIR = PREDICTIONS_PATH / "val_preds"
TEST_PRED_DIR = PREDICTIONS_PATH / "test_preds"

SUBMISSIONS_PATH = OUTPUTS_PATH / "submissions"
SUBMISSION_SAVE_PATH = SUBMISSIONS_PATH / "submission.json"
SAMPLE_ANSWER_PATH = SUBMISSIONS_PATH / "sample_answer.json"

VISUALIZATIONS_PATH = OUTPUTS_PATH / "visualizations"