"""Melatih ulang model v2 (231 fitur: 165 sudut + 66 koordinat relatif) dari data latih utama.

Sama dengan model utama (split 70:30 per run, flip augmentation, Random Forest 100 pohon, scaler yang sama)
kecuali 132 kolom coord_mean_* tidak dipakai. Hasil ditulis ke folder keluaran (bawaan: models_baru/ di sini)
dan tidak menimpa models/.
Pemakaian: python latih_v2.py [folder_keluaran]
"""
import torch  # noqa: F401  (Windows: harus diimpor sebelum pandas)
import copy
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.features.build_dataset import META_COLUMNS  # noqa: E402

UTAMA = REPO / "experimental_client_pipeline"
SPLIT = UTAMA / "data" / "windowed_features" / "splits"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "models_baru"
OUT.mkdir(parents=True, exist_ok=True)

for ex in ("squat", "benchpress", "deadlift"):
    train = pd.read_csv(SPLIT / f"{ex}_train.csv")
    test = pd.read_csv(SPLIT / f"{ex}_test.csv")
    kolom = [c for c in train.columns if c not in META_COLUMNS and not c.startswith("coord_mean_")]
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=4, max_features="sqrt", class_weight="balanced",
                                random_state=42, n_jobs=-1).fit(train[kolom], train["class"])
    rf.n_jobs = None  # n_jobs=-1 membuat prediksi satu pesan 2x lebih lambat, server tertinggal dari video
    utama = pickle.load(open(UTAMA / "models" / f"{ex}_rf.pkl", "rb"))
    pipe = Pipeline([("scaler", copy.deepcopy(utama.named_steps["scaler"])), ("rf", rf)])
    config = json.load(open(UTAMA / "models" / f"{ex}_feature_config.json"))
    config["feat_cols"] = kolom
    pickle.dump(pipe, open(OUT / f"{ex}_rf.pkl", "wb"))
    json.dump(config, open(OUT / f"{ex}_feature_config.json", "w"))
    f1 = f1_score(test["class"], rf.predict(test[kolom]), average="weighted")
    print(f"{ex}: {len(kolom)} fitur, F1 berbobot 70:30 {f1:.4f}")
print("model ditulis ke", OUT)
