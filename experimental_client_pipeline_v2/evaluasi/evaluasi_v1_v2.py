"""Evaluasi model utama (v1, 363 fitur) dan v2 (231 fitur): split 70:30 dan LOPO (3 seed).

Keluaran:
- ../models/<latihan>_evaluation.json dan <latihan>_confusion_matrix.png untuk v2, format sama dengan model utama.
- <latihan>_v1_vs_v2.json di folder ini: 70:30 dan LOPO untuk v1 dan v2 per kelas, plus daftar perubahan per kelas
  (urut dari yang paling memburuk).
"""
import torch  # noqa: F401  (Windows: harus diimpor sebelum pandas)
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from src.features.augmentation import flip_horizontal  # noqa: E402
from src.features.build_dataset import META_COLUMNS  # noqa: E402
from src.training.train_model import plot_confusion_matrix  # noqa: E402

UTAMA = REPO / "experimental_client_pipeline"
MODELS_V2 = HERE.parent / "models"
SPLIT = UTAMA / "data" / "windowed_features" / "splits"
EXERCISES = ("squat", "benchpress", "deadlift")
SEEDS = (42, 7, 2026)
VERSI = ("v1", "v2")


def latih(x, y, seed):
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=4, max_features="sqrt", class_weight="balanced",
                                  random_state=seed, n_jobs=-1).fit(x, y)


def f1w(y, pred):
    return f1_score(y, pred, average="weighted", zero_division=0)


def laporan(y, pred, labels):
    cm = confusion_matrix(y, pred, labels=labels)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision_weighted": precision_score(y, pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y, pred, average="weighted", zero_division=0),
        "f1_weighted": f1w(y, pred),
        "classification_report": classification_report(y, pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": cm.tolist(),
    }


def perubahan(rep1, rep2, labels):
    baris = []
    for k in labels:
        a, b = rep1[k], rep2[k]
        baris.append({"kelas": k, "support": int(a["support"]),
                      "precision_v1": round(a["precision"], 4), "precision_v2": round(b["precision"], 4),
                      "recall_v1": round(a["recall"], 4), "recall_v2": round(b["recall"], 4), "selisih_recall": round(b["recall"] - a["recall"], 4),
                      "f1_v1": round(a["f1-score"], 4), "f1_v2": round(b["f1-score"], 4), "selisih_f1": round(b["f1-score"] - a["f1-score"], 4)})
    baris.sort(key=lambda r: r["selisih_recall"])
    d = [r["selisih_recall"] for r in baris]
    ringkasan = {"kelas_recall_turun": sum(x < -0.005 for x in d), "kelas_recall_naik": sum(x > 0.005 for x in d),
                 "jumlah_turun": round(sum(x for x in d if x < 0), 3), "jumlah_naik": round(sum(x for x in d if x > 0), 3)}
    return baris, ringkasan


for ex in EXERCISES:
    train = pd.read_csv(SPLIT / f"{ex}_train.csv")
    test = pd.read_csv(SPLIT / f"{ex}_test.csv")
    semua = [c for c in train.columns if c not in META_COLUMNS]
    labels = sorted(test["class"].unique())
    kolom = {"v1": semua, "v2": [c for c in semua if not c.startswith("coord_mean_")]}

    # 70:30 dengan berkas model yang dipakai aplikasi
    hasil = {}
    for versi, folder in (("v1", UTAMA / "models"), ("v2", MODELS_V2)):
        rf = pickle.load(open(folder / f"{ex}_rf.pkl", "rb")).named_steps["rf"]
        assert list(rf.feature_names_in_) == kolom[versi], f"{ex} {versi}: kolom model berbeda dengan data"
        hasil[versi] = laporan(test["class"], rf.predict(test[kolom[versi]]), labels)
    v1_asli = json.load(open(UTAMA / "models" / f"{ex}_evaluation.json"))
    assert abs(hasil["v1"]["f1_weighted"] - v1_asli["f1_weighted"]) < 1e-9, f"{ex}: F1 v1 tidak sama dengan evaluation.json utama"

    evaluasi_v2 = {"exercise": ex, "n_train": len(train), "n_test": len(test), "n_features": len(kolom["v2"]), "classes": labels, **hasil["v2"],
                   "hyperparameters": v1_asli["hyperparameters"]}
    json.dump(evaluasi_v2, open(MODELS_V2 / f"{ex}_evaluation.json", "w"), indent=2)
    plot_confusion_matrix(np.array(hasil["v2"]["confusion_matrix"]), labels, MODELS_V2 / f"{ex}_confusion_matrix.png", f"Confusion Matrix -- {ex} (RF v2, 231 fitur)")
    baris_7030, ring_7030 = perubahan(hasil["v1"]["classification_report"], hasil["v2"]["classification_report"], labels)

    # LOPO: satu partisipan ditahan sebagai orang baru
    pool = pd.concat([train[~train["window_id"].str.endswith("_flip")], test], ignore_index=True)
    f1_lopo = {v: {} for v in VERSI}
    y_all, pred_all = [], {v: [] for v in VERSI}
    for pid in sorted(pool["participant_id"].unique()):
        dilatih = pool[pool["participant_id"] != pid]
        dilatih = pd.concat([dilatih, flip_horizontal(dilatih)], ignore_index=True)
        diuji = pool[pool["participant_id"] == pid]
        nama = f"P{str(pid).lstrip('pP')}"
        y_all += list(diuji["class"])
        for versi in VERSI:
            f1_lopo[versi][nama] = {}
            for seed in SEEDS:
                pred = latih(dilatih[kolom[versi]], dilatih["class"], seed).predict(diuji[kolom[versi]])
                f1_lopo[versi][nama][seed] = f1w(diuji["class"], pred)
                if seed == SEEDS[0]:
                    pred_all[versi] += list(pred)
    lopo = {}
    for versi in VERSI:
        per_seed = [float(np.mean([f1_lopo[versi][p][s] for p in f1_lopo[versi]])) for s in SEEDS]
        lopo[versi] = {"f1_weighted_rata_rata": round(float(np.mean(per_seed)), 4), "sd_antar_seed": round(float(np.std(per_seed)), 4),
                       "f1_weighted_per_partisipan": {p: round(float(np.mean(list(v.values()))), 4) for p, v in f1_lopo[versi].items()},
                       "catatan": "classification_report dan confusion_matrix = gabungan semua partisipan yang ditahan, seed 42",
                       **laporan(y_all, pred_all[versi], labels)}
    baris_lopo, ring_lopo = perubahan(lopo["v1"]["classification_report"], lopo["v2"]["classification_report"], labels)

    json.dump({"exercise": ex, "classes": labels,
               "split_70_30": {"v1": hasil["v1"], "v2": hasil["v2"], "perubahan_per_kelas": baris_7030, "ringkasan": ring_7030},
               "lopo": {**lopo, "perubahan_per_kelas": baris_lopo, "ringkasan": ring_lopo}},
              open(HERE / f"{ex}_v1_vs_v2.json", "w"), indent=2)
    print(f"{ex}: 70:30 F1 {hasil['v1']['f1_weighted']:.4f} -> {hasil['v2']['f1_weighted']:.4f} | LOPO {lopo['v1']['f1_weighted_rata_rata']:.4f} -> {lopo['v2']['f1_weighted_rata_rata']:.4f}", flush=True)
print("selesai")
