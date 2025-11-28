# src/config.py
from pathlib import Path
import mlflow
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_DIR = DATA_DIR / "train"

TEST_CSV_NAME = "test.csv"
TEST_ID_COL = "id"
TEST_PATH_COL = "path"
TEST_IMAGES_ROOT = DATA_DIR

INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 100  

N_SPLITS = 4
EPOCHS_OPTUNA = 10
BEST_MODEL_NAME = "convnext_base"

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "Butterfly_Classification"
try:
    if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)
except Exception:
    pass 

print(f"MLflow setup complete. Tracking to: {MLFLOW_TRACKING_URI}")