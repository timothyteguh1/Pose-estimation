# %% [markdown]
# # EDA -- Exploratory Data Analysis
#
# **PENTING (baca sebelum pakai di laporan):** notebook ini dibangun pertama
# kali pakai data PILOT (baru squat, 1 partisipan p1, 9 video). Sebelum
# dipakai jadi bagian FINAL skripsi, notebook ini WAJIB dijalankan ULANG
# setelah semua data (3 partisipan x 3 gerakan: squat, benchpress, deadlift)
# lengkap -- kode di bawah generik (baca semua video lewat glob, tidak
# hardcode nama file), jadi cukup jalan ulang semua cell, tidak perlu edit.
#
# Read-only -- notebook ini TIDAK mengubah file apa pun di data/ atau models/,
# cuma baca & tampilkan ringkasan/plot. Aman dijalankan kapan saja.
#
# Isi:
# 1. Ringkasan video & partisipan yang tersedia
# 2. Distribusi kelas (posture_class x phase)
# 3. Panjang run/repetisi per kelas (relevan utk pilih window_sec)
# 4. Cek visibility/occlusion per video
# 5. Visualisasi primary angle (sanity-check fase konsentrik/eksentrik)
# 6. Ringkasan dataset hasil windowing (kalau sudah pernah build_dataset.py)

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from src.features.build_dataset import find_labeled_videos, load_video_runs
from src.features.joint_angles import PRIMARY_ANGLE_BASE_BY_EXERCISE, primary_angle_for_frame

EXERCISES = ["squat", "benchpress", "deadlift"]
VISIBILITY_THRESHOLD = 0.6

pd.set_option("display.width", 120)

# %% [markdown]
# ## 1. Ringkasan video & partisipan yang tersedia

# %%
overview_rows = []
labeled_dfs = {}  # exercise -> list of (path, df)

for exercise in EXERCISES:
    ready, skipped = find_labeled_videos(exercise)
    dfs = []
    for p in ready:
        df = pd.read_csv(p)
        dfs.append((p, df))
    labeled_dfs[exercise] = dfs
    for p, df in dfs:
        fps = float(df["fps"].iloc[0]) if "fps" in df.columns else np.nan
        overview_rows.append({
            "exercise": exercise,
            "video": p.stem,
            "posture_class": df["posture_class"].iloc[0],
            "participant_id": df["participant_id"].iloc[0],
            "camera_angle": df["camera_angle"].iloc[0],
            "n_frames": len(df),
            "fps": fps,
            "duration_sec": len(df) / fps if fps else np.nan,
        })
    print(f"[{exercise}] {len(ready)} video siap (100% manual), {len(skipped)} dilewati")
    for p, reason in skipped:
        print(f"    [skip] {p.name}: {reason}")

overview_df = pd.DataFrame(overview_rows)
overview_df

# %%
if not overview_df.empty:
    print("Partisipan per exercise:")
    print(overview_df.groupby("exercise")["participant_id"].unique())
    print("\nTotal durasi video (menit) per exercise:")
    print(overview_df.groupby("exercise")["duration_sec"].sum() / 60)
    print("\nDistribusi camera_angle:")
    print(overview_df.groupby(["exercise", "camera_angle"]).size())

# %% [markdown]
# ## 2. Distribusi kelas (posture_class x phase)
#
# Cek keseimbangan kelas sesuai bab 8.2 proposal ("distribusi data pun
# diusahakan seimbang antara postur benar dan salah").

# %%
for exercise, dfs in labeled_dfs.items():
    if not dfs:
        continue
    all_frames = pd.concat([df for _, df in dfs], ignore_index=True)
    print(f"\n=== {exercise}: distribusi frame per kelas ===")
    print(all_frames["class"].value_counts())

# %% [markdown]
# ## 3. Panjang run/repetisi per kelas
#
# Run = segmen frame berurutan 1 fase (dipakai jadi 1 window candidate).
# PENTING buat pilih `window_sec`: kalau window_sec bikin swl (window
# length) lebih besar dari run TERPENDEK, run itu 0 window (hilang total).

# %%
run_rows = []
for exercise, dfs in labeled_dfs.items():
    for p, df in dfs:
        _, _, runs = load_video_runs(p, VISIBILITY_THRESHOLD)
        for run in runs:
            cls = df["class"].iloc[run[0]]
            run_rows.append({
                "exercise": exercise, "video": p.stem, "class": cls,
                "len_frames": run[1] - run[0],
            })
run_df = pd.DataFrame(run_rows)

if not run_df.empty:
    summary = run_df.groupby("class")["len_frames"].agg(["count", "min", "median", "max"])
    print(summary)
    print("\nCatatan: swl=15 (window_sec=0.5 @ ~30fps) butuh run>=15 frame; "
          "swl=30 (window_sec=1.0) butuh run>=30 frame -- cek kolom 'min' di atas "
          "buat lihat kelas mana yang berisiko kehilangan run kalau window_sec dinaikkan.")

# %%
if not run_df.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    run_df.boxplot(column="len_frames", by="class", ax=ax, rot=45)
    ax.set_ylabel("panjang run (frame)")
    ax.set_title("Distribusi panjang run per kelas")
    plt.suptitle("")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 4. Cek visibility / occlusion per video
#
# Berapa persen frame per video yang lolos ambang visibility >= 0.6
# (bab 8.3.1). Video dengan persentase rendah butuh perhatian khusus
# (mis. video squat_min.mp4 dari Ko et al. yang dicek manual: 0% lolos
# karena wrist tidak terdeteksi -- lihat diskusi proyek).

# %%
from src.features.sliding_window import compute_frame_features

vis_rows = []
for exercise, dfs in labeled_dfs.items():
    for p, df in dfs:
        feat = compute_frame_features(df, VISIBILITY_THRESHOLD)
        pct = feat["usable"].mean() * 100
        vis_rows.append({"exercise": exercise, "video": p.stem, "pct_usable": pct})
vis_df = pd.DataFrame(vis_rows)
if not vis_df.empty:
    print(vis_df.sort_values("pct_usable"))

# %% [markdown]
# ## 5. Visualisasi primary angle (sanity-check fase)
#
# Plot sudut utama (knee utk squat, elbow utk benchpress, hip utk deadlift)
# sepanjang waktu, buat 1 video contoh per exercise -- pola naik-turun harus
# terlihat jelas seirama dengan repetisi.

# %%
for exercise, dfs in labeled_dfs.items():
    if not dfs:
        continue
    p, df = dfs[0]  # video pertama sbg contoh
    primary = df.apply(lambda row: primary_angle_for_frame(row, exercise), axis=1)
    fps = float(df["fps"].iloc[0]) if "fps" in df.columns else 30.0
    t = np.arange(len(df)) / fps

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, primary)
    ax.set_title(f"{exercise}: {PRIMARY_ANGLE_BASE_BY_EXERCISE[exercise]} -- contoh {p.stem}")
    ax.set_xlabel("detik")
    ax.set_ylabel("sudut (derajat)")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 6. Ringkasan dataset hasil windowing (kalau sudah pernah build_dataset.py)

# %%
SPLITS_DIR = REPO_ROOT / "data" / "windowed_features" / "splits"
for exercise in EXERCISES:
    train_path = SPLITS_DIR / f"{exercise}_train.csv"
    test_path = SPLITS_DIR / f"{exercise}_test.csv"
    if not (train_path.exists() and test_path.exists()):
        print(f"[{exercise}] belum ada train/test split -- jalankan build_dataset.py dulu")
        continue
    train_df = pd.read_csv(train_path, usecols=["class"])
    test_df = pd.read_csv(test_path, usecols=["class"])
    total = len(train_df) + len(test_df)
    print(f"\n=== {exercise}: {total} window (train={len(train_df)}, test={len(test_df)}, "
          f"rasio test={len(test_df)/total*100:.1f}%) ===")
    dist = pd.DataFrame({"train": train_df["class"].value_counts(),
                          "test": test_df["class"].value_counts()}).fillna(0).astype(int)
    print(dist)
