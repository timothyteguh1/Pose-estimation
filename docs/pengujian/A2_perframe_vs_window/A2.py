"""A2: sliding window (15 frame) vs per-frame pada model final, split 70:30 per run (seed 42).

Window = model produksi (angka A1). Per-frame = pipeline build_dataset yang sama dengan panjang window 1 frame,
RF sama, dinilai pada semua frame uji. Keluaran satu file, A2_tabel.csv: tabel resmi A2 (8 kolom) ditambah
Perubahan_prediksi_persen, yaitu persen pasangan titik waktu berurutan (video dan label asli sama) yang
prediksinya berganti; makin kecil makin stabil.

Hanya membaca model dan data proyek; dataset per-frame dibangun sementara di folder temp.
Jalankan dari akar proyek:  venv\\Scripts\\python docs\\pengujian\\A2_perframe_vs_window\\A2.py
"""
import contextlib
import io
import json
import pickle
import sys
import tempfile
from pathlib import Path

import torch  # noqa: F401  (di Windows torch harus diimpor sebelum pandas)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import src.features.window_scaler  # noqa: F401  (agar pickle model bisa dibuka)
from src.features import build_dataset as bd
from src.features.build_dataset import META_COLUMNS

EXP = REPO / "experimental_client_pipeline"
SPLIT_WINDOW = EXP / "data" / "windowed_features" / "splits"
KERJA = Path(tempfile.gettempdir()) / "A2_perframe_kerja"
LATIHAN = {"squat": "Squat", "benchpress": "Bench press", "deadlift": "Deadlift"}
KOLOM = ["Latihan", "Pendekatan", "n_test", "Accuracy", "Precision_w", "Recall_w", "F1_weighted", "F1_macro",
         "Perubahan_prediksi_persen"]


def bangun_perframe(kode):
    # window_sec 0.0333 membuat panjang window 1 frame; alokasi run train/test identik dengan produksi
    train, test = KERJA / "splits" / f"{kode}_train.csv", KERJA / "splits" / f"{kode}_test.csv"
    if train.exists() and test.exists():
        return
    bd.EXTRACTED_DIR = EXP / "data" / "labeled"
    bd.WINDOWED_DIR = KERJA
    bd.SPLITS_DIR = KERJA / "splits"
    bd.MODELS_DIR = KERJA / "models"
    sys.argv = ["build_dataset.py", kode, "--window-sec", "0.0333", "--coord-relative", "--flip-augment"]
    with contextlib.redirect_stdout(io.StringIO()):
        bd.main()
    if not (train.exists() and test.exists()):
        raise RuntimeError(f"dataset per-frame {kode} gagal dibangun")


def metrik(y, p):
    pw, rw, fw, _ = precision_recall_fscore_support(y, p, average="weighted", zero_division=0)
    return {"n_test": len(y), "Accuracy": accuracy_score(y, p), "Precision_w": pw, "Recall_w": rw,
            "F1_weighted": fw, "F1_macro": f1_score(y, p, average="macro", zero_division=0)}


def pasangan_berubah(df, pred):
    # dua titik waktu berurutan (frame akhir selisih 1) di video dan label asli yang sama
    d = pd.DataFrame({"v": df["source_video"].to_numpy(), "t": df["end_frame_idx"].to_numpy(),
                      "y": df["class"].to_numpy(), "p": pred}).sort_values(["v", "t"])
    v, t, y, p = d["v"].to_numpy(), d["t"].to_numpy(), d["y"].to_numpy(), d["p"].to_numpy()
    sama = (v[1:] == v[:-1]) & (y[1:] == y[:-1]) & (np.diff(t) == 1)
    return int(sama.sum()), int((sama & (p[1:] != p[:-1])).sum())


def kunci(df):
    return (df["source_video"].astype(str) + "|" + df["end_frame_idx"].astype(int).astype(str)).to_numpy()


def main():
    baris = []
    total ={p: {"y": [], "pred": [], "pasang": 0, "ubah": 0} for p in ("Window", "Per-frame")}
    for kode, nama in LATIHAN.items():
        bangun_perframe(kode)

        cfg = json.load(open(EXP / "models" / f"{kode}_feature_config.json"))
        rf_window = pickle.load(open(EXP / "models" / f"{kode}_rf.pkl", "rb")).named_steps["rf"]
        te_w = pd.read_csv(SPLIT_WINDOW / f"{kode}_test.csv")
        p_w = rf_window.predict(te_w[cfg["feat_cols"]])

        tr_f = pd.read_csv(KERJA / "splits" / f"{kode}_train.csv")
        te_f = pd.read_csv(KERJA / "splits" / f"{kode}_test.csv")
        fitur = [c for c in tr_f.columns if c not in META_COLUMNS]
        rf_frame = RandomForestClassifier(n_estimators=100, min_samples_leaf=4, class_weight="balanced",
                                          random_state=42, n_jobs=-1).fit(tr_f[fitur], tr_f["class"])
        p_f = rf_frame.predict(te_f[fitur])

        # window dan per-frame harus berasal dari run uji yang sama
        if np.isin(kunci(te_w), kunci(te_f)).sum() != len(te_w):
            raise RuntimeError(f"{kode}: frame akhir window uji tidak semuanya ada di frame uji per-frame")

        for pendekatan, df, pred in (("Window", te_w, p_w), ("Per-frame", te_f, p_f)):
            y = df["class"].to_numpy()
            pasang, ubah = pasangan_berubah(df, pred)
            baris.append({"Latihan": nama, "Pendekatan": pendekatan, **metrik(y, pred),
                          "Perubahan_prediksi_persen": 100 * ubah / pasang})
            t = total[pendekatan]
            t["y"].append(y); t["pred"].append(pred); t["pasang"] += pasang; t["ubah"] += ubah
            print(f"[{nama}] {pendekatan}: prediksi berganti pada {ubah} dari {pasang} pasangan", flush=True)

    for pendekatan, t in total.items():
        baris.append({"Latihan": "Gabungan 3 latihan", "Pendekatan": pendekatan,
                      **metrik(np.concatenate(t["y"]), np.concatenate(t["pred"])),
                      "Perubahan_prediksi_persen": 100 * t["ubah"] / t["pasang"]})

    # gabungan = semua prediksi uji digabung (bukan rata-rata tiga angka)
    tabel = pd.DataFrame(baris)[KOLOM]
    tabel["n_test"] = tabel["n_test"].astype(int)
    tabel.to_csv(HERE / "A2_tabel.csv", index=False, sep=";", decimal=",", float_format="%.4f")

    tampil = tabel.copy()
    for k in tabel.select_dtypes("float").columns:
        tampil[k] = tabel[k].map(lambda x: f"{x:.4f}".replace(".", ","))
    print("\n" + tampil.to_string(index=False))
    print(f"\nTersimpan: {HERE / 'A2_tabel.csv'}")


if __name__ == "__main__":
    main()
