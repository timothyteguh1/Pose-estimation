import sys
from pathlib import Path

import torch  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.features import build_dataset

build_dataset.EXTRACTED_DIR = EXPERIMENT_ROOT / "data" / "labeled"
build_dataset.WINDOWED_DIR = EXPERIMENT_ROOT / "data" / "windowed_features"
build_dataset.SPLITS_DIR = build_dataset.WINDOWED_DIR / "splits"
build_dataset.MODELS_DIR = EXPERIMENT_ROOT / "models"

if __name__ == "__main__":
    build_dataset.main()
