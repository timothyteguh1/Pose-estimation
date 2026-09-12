"""Tahap 3: training & testing Random Forest (bab 6.1.3 & 8.3.2 proposal).

Model = Random Forest saja -- proposal eksplisit cuma sebut 1 model (beda
dari Ko et al. yang bandingkan LR/Ridge/GB). 1 exercise = 1 model terpisah
(squat_rf.pkl, dst).

Alur:
    1. Load data/windowed_features/splits/{exercise}_train.csv + _test.csv
       (sudah di-window + dinormalisasi oleh build_dataset.py)
    2. Kolom fitur diambil dinamis dari header CSV
    3. RandomForestClassifier(class_weight='balanced') -- kelas kita timpang
       (concentric jauh lebih banyak dari eccentric)
    4. Evaluasi ke test set: accuracy, precision, recall, F1 (weighted,
       indikator utama sesuai bab 8.4), classification report, confusion matrix
    5. Simpan model final sebagai 1 file ({exercise}_rf.pkl): scaler (hasil
       build_dataset.py) + classifier digabung jadi 1 sklearn Pipeline, gaya
       Ko et al -- angka evaluasi tidak berubah (pipeline final setara,
       bukan re-fit).

Usage:
    python src/training/train_model.py squat
    python src/training/train_model.py squat --n-estimators 200
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.build_dataset import META_COLUMNS
from src.io_utils import atomic_write_csv

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "windowed_features" / "splits"
MODELS_DIR = REPO_ROOT / "models"


def load_split(exercise, split_name):
    path = SPLITS_DIR / f"{exercise}_{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} tidak ada -- jalankan dulu: python src/features/build_dataset.py {exercise}"
        )
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c not in META_COLUMNS]
    X = df[feat_cols]
    y = df["class"]
    return X, y, feat_cols


def plot_confusion_matrix(cm, labels, out_path, title):
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 2, 1.1 * len(labels) + 2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual (ground truth)")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exercise", choices=["squat", "benchpress", "deadlift"])
    parser.add_argument("--n-estimators", type=int, default=100,
                         help="jumlah pohon RF (default scikit-learn = 100)")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    print(f"=== Training Random Forest: {args.exercise} ===")
    X_train, y_train, feat_cols = load_split(args.exercise, "train")
    X_test, y_test, feat_cols_test = load_split(args.exercise, "test")

    if feat_cols != feat_cols_test:
        print("[error] kolom fitur train dan test BEDA -- data tidak konsisten, cek build_dataset.py.")
        return

    print(f"train: {len(X_train)} sampel, {len(feat_cols)} fitur, "
          f"{y_train.nunique()} kelas: {sorted(y_train.unique())}")
    print(f"test : {len(X_test)} sampel")

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight="balanced",
        random_state=args.random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n=== Evaluasi ({args.exercise}) ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (weighted)")
    print(f"Recall   : {recall:.4f} (weighted)")
    print(f"F1-score : {f1:.4f} (weighted) <- indikator utama (bab 8.4 proposal)")

    labels = sorted(y_test.unique())
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    print("\nClassification report per kelas:")
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Gabungkan scaler (build_dataset.py) + classifier baru jadi 1 Pipeline,
    # gaya Ko et al -- scaler-nya sama persis (tidak di-fit ulang), pipeline
    # final ini setara, bukan model baru.
    scaler_path = MODELS_DIR / f"{args.exercise}_scaler.pkl"
    if not scaler_path.exists():
        print(f"[error] {scaler_path} tidak ada -- jalankan dulu build_dataset.py {args.exercise}.")
        return
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    pipeline = Pipeline([("scaler", scaler), ("rf", model)])

    model_path = MODELS_DIR / f"{args.exercise}_rf.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n[ok] model (scaler+RF digabung 1 pipeline) disimpan: {model_path}")

    scaler_path.unlink()  # sudah nempel di dalam pipeline, file terpisah jadi redundan
    print(f"[ok] {scaler_path.name} dihapus (sudah nempel di dalam {model_path.name})")

    eval_path = MODELS_DIR / f"{args.exercise}_evaluation.json"
    eval_result = {
        "exercise": args.exercise,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feat_cols),
        "classes": labels,
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "hyperparameters": {"n_estimators": args.n_estimators, "class_weight": "balanced",
                             "random_state": args.random_state},
    }
    with open(eval_path, "w") as f:
        json.dump(eval_result, f, indent=2)
    print(f"[ok] hasil evaluasi disimpan: {eval_path}")

    cm_path = MODELS_DIR / f"{args.exercise}_confusion_matrix.png"
    plot_confusion_matrix(cm, labels, cm_path, f"Confusion Matrix -- {args.exercise} (RF)")
    print(f"[ok] confusion matrix disimpan: {cm_path}")


if __name__ == "__main__":
    main()
