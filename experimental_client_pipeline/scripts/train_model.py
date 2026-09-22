import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.training import train_model

train_model.SPLITS_DIR = EXPERIMENT_ROOT / "data" / "windowed_features" / "splits"
train_model.MODELS_DIR = EXPERIMENT_ROOT / "models"

if __name__ == "__main__":
    train_model.main()
