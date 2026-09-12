"""Sliding window (Eq. 2 & 3, proposal bab 6.2.2) -- dibangun dari nol, tidak
ada di kode Ko et al.

Eq. 2: swl = fps x t              (panjang window, dalam frame)
Eq. 3: swm = Frames - swl + 1     (jumlah window dari 1 video, stride=1)

1 baris CSV extracted_landmarks = 1 frame. 1 baris output di sini = 1 window
(beberapa frame digabung jadi 1 sampel temporal).

Aturan window: tidak boleh melompati frame 'excluded', tidak boleh melompati
frame yang benar-benar tanpa data (semua 11 angle NaN -- filter visibility
sendiri per-titik di joint_angles.py, jadi occlusion 1 titik saja tidak
memutus run), dan tidak boleh mencampur 2 fase berbeda dalam 1 window.
Ketiga aturan ini memecah 1 video jadi beberapa "run" valid, window
dibangun stride=1 di dalam tiap run.

Fitur per window (bab 6.2.1: joint angle = fitur utama, coordinate =
pendukung):
- Fitur utama: 11 joint angle di-flatten per frame dalam window (bukan
  dirata-ratakan), supaya pola temporal gerakan tetap terekam.
- Fitur pendukung: koordinat (x,y,z,v) semua 33 landmark MediaPipe,
  dirata-ratakan 1 angka per window (coord_mean_*, 132 kolom tetap). Sempat
  dicoba di-flatten juga (spt angle) -- feature_importances_ RF menunjukkan
  itu bikin koordinat mendominasi keputusan model, bertentangan dgn aturan
  "angle = fitur utama" -- dikembalikan ke mean. Perbandingan F1 kedua
  mode didokumentasikan di evaluasi model (bab IV).
- use_z (default False): angle dihitung dari titik 2D saja, persis Ko et
  al -- use_z=True cuma opsi eksperimen pembanding.

Split train/test (build_dataset.py): run dialokasikan utuh ke train ATAU
test (bukan acak per baris window, bukan dipotong) -- karena window tidak
pernah dibangun melintasi batas run, alokasi run utuh otomatis aman dari
leakage. (split_runs_by_frame_fraction() masih ada sbg utility, tapi tidak
dipakai build_dataset.py lagi.)
"""
import numpy as np
import pandas as pd

from src.features.joint_angles import (
    ANGLE_COLUMNS,
    POSE_LANDMARK_NAMES,
    VISIBILITY_THRESHOLD,
    compute_frame_angles,
)


def compute_frame_features(df, visibility_threshold=VISIBILITY_THRESHOLD, use_z=False):
    """Hitung 11 angle + flag usable per frame utk seluruh df 1 video.

    Returns DataFrame baru (index sama dgn df) berisi kolom ANGLE_COLUMNS +
    'usable' (bool) + koordinat (x,y,z,v) SEMUA 33 landmark per-frame
    (dipakai nanti buat di-flatten per-window jadi fitur pendukung).

    `use_z`: default False (2D, sama Ko et al) -- lihat landmark_point() di
    joint_angles.py. Cuma memengaruhi ANGLE (ANGLE_COLUMNS), TIDAK memengaruhi
    kolom koordinat mentah (x,y,z,v tetap disalin apa adanya, sudah 3D dari
    sananya)."""
    angle_rows = []
    usable_flags = []
    for _, row in df.iterrows():
        angles, usable = compute_frame_angles(row, visibility_threshold, use_z)
        angle_rows.append(angles)
        usable_flags.append(usable)

    angles_df = pd.DataFrame(angle_rows, index=df.index)[ANGLE_COLUMNS]
    angles_df["usable"] = usable_flags
    for name in POSE_LANDMARK_NAMES:
        for axis in ("x", "y", "z", "v"):
            angles_df[f"{name}_{axis}"] = df[f"{name}_{axis}"].to_numpy()
    return angles_df


def find_valid_runs(df, feat_df):
    """Pecah video jadi run frame berurutan yang valid buat windowing.

    Valid = phase bukan 'excluded' DAN usable (visibility cukup) DAN class
    sama dgn frame sebelumnya (run putus tiap kali class berganti, mis. pas
    transisi concentric->eccentric).

    Returns list of (start_idx, end_idx_exclusive) posisi INTEGER (bukan
    frame_idx asli) relatif ke df yang sudah di-reset_index.
    """
    ok = (df["phase"] != "excluded").to_numpy() & feat_df["usable"].to_numpy()
    classes = df["class"].to_numpy()

    runs = []
    run_start = None
    for i in range(len(df)):
        if ok[i] and (run_start is None or classes[i] == classes[run_start]):
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start > 0:
                runs.append((run_start, i))
            run_start = i if ok[i] else None
    if run_start is not None:
        runs.append((run_start, len(df)))
    return runs


def split_runs_by_frame_fraction(runs, front_fraction):
    """Potong daftar run KRONOLOGIS jadi 2 kelompok: `front_fraction` dari
    total frame (dihitung urut waktu dari run pertama) masuk kelompok
    "depan", sisanya kelompok "belakang". Kalau titik potong jatuh di TENGAH
    1 run, run itu dipecah jadi 2 run baru (bukan window yang dipecah --
    window baru dibangun setelah ini, jadi tidak ada window yang nyebrang
    titik potong).

    Dipakai buat split train/test PER VIDEO tanpa data leakage antar window
    yang overlap (lihat docstring modul).

    Returns (front_runs, back_runs).
    """
    total = sum(e - s for s, e in runs)
    target = round(total * front_fraction)
    front, back = [], []
    accumulated = 0
    for s, e in runs:
        length = e - s
        if accumulated >= target:
            back.append((s, e))
            continue
        remaining_needed = target - accumulated
        if length <= remaining_needed:
            front.append((s, e))
            accumulated += length
        else:
            cut = s + remaining_needed
            if remaining_needed > 0:
                front.append((s, cut))
            if cut < e:
                back.append((cut, e))
            accumulated = target
    return front, back


def build_windows_from_runs(df, feat_df, runs, window_sec, stride=1, split_label=None, coord_mode="mean"):
    """Bangun window (Eq.2 & 3) dari daftar run yang sudah ditentukan
    (biasanya hasil find_valid_runs() atau split_runs_by_frame_fraction()).

    split_label: opsional, ditulis ke kolom 'split' tiap window
    (traceability di file per-video, lihat build_dataset.py).

    coord_mode: "mean" (default, produksi) atau "flatten" (eksperimen
    pembanding, lihat build_dataset.py --coord-mode dan docstring modul).

    Returns DataFrame (1 baris = 1 window) atau None kalau tidak ada window
    yang bisa dibangun (semua run < swl).
    """
    fps = float(df["fps"].iloc[0]) if "fps" in df.columns else 30.0
    swl = max(1, round(fps * window_sec))

    meta_cols = ["exercise", "posture_class", "participant_id", "camera_angle", "take", "source_video"]
    windows = []
    window_idx = 0
    n_dropped_nan = 0
    for run_start, run_end in runs:
        run_len = run_end - run_start
        if run_len < swl:
            continue
        swm = run_len - swl + 1  # Eq.3, dalam lingkup 1 run
        for w_start in range(run_start, run_start + swm, stride):
            w_end = w_start + swl  # exclusive
            window_feat = feat_df.iloc[w_start:w_end]

            # fitur UTAMA: 11 angle di-flatten per frame dlm window (urutan temporal terjaga)
            angle_block = {}
            for f in range(swl):
                for col in ANGLE_COLUMNS:
                    angle_block[f"{col}_f{f}"] = window_feat[col].iloc[f]

            # fitur PENDUKUNG: koordinat (x,y,z,v) semua 33 landmark -- flatten
            # per frame (default/produksi) ATAU mean per window (eksperimen).
            coord_block = {}
            for name in POSE_LANDMARK_NAMES:
                for axis in ("x", "y", "z", "v"):
                    col = f"{name}_{axis}"
                    if coord_mode == "mean":
                        coord_block[f"coord_mean_{col}"] = window_feat[col].mean()
                    else:
                        for f in range(swl):
                            coord_block[f"coord_{col}_f{f}"] = window_feat[col].iloc[f]

            # Occlusion 1 titik sesaat tidak memutus run (usable longgar),
            # tapi window yg kebetulan masih mengandung NaN dibuang di sini
            # -- RF tidak boleh terima NaN, discard bukan interpolasi.
            if any(v != v for v in angle_block.values()) or any(v != v for v in coord_block.values()):
                n_dropped_nan += 1
                continue

            row = {c: df[c].iloc[w_start] for c in meta_cols}
            row["class"] = df["class"].iloc[w_start]
            row["phase"] = df["phase"].iloc[w_start]
            suffix = f"_{split_label}" if split_label else ""
            row["window_id"] = f"{df['source_video'].iloc[w_start]}{suffix}_w{window_idx}"
            row["split"] = split_label if split_label else ""
            row["start_frame_idx"] = int(df["frame_idx"].iloc[w_start])
            row["end_frame_idx"] = int(df["frame_idx"].iloc[w_end - 1])
            row["start_timestamp_sec"] = float(df["timestamp_sec"].iloc[w_start])
            row["window_len_frames"] = swl
            row["fps"] = fps
            row.update(angle_block)
            row.update(coord_block)

            windows.append(row)
            window_idx += 1

    if n_dropped_nan:
        print(f"    [info] {n_dropped_nan} window dibuang (masih mengandung NaN -- "
              f"occlusion 1/lebih titik jatuh persis di window itu)")

    if not windows:
        return None
    return pd.DataFrame(windows)


def find_usable_runs(feat_df):
    """Run kontinu murni dari flag 'usable' (visibility cukup) -- TANPA
    filter phase/class, karena video BARU (belum dilabeli) tidak punya kolom
    itu sama sekali. Dipakai src/training/predict_video.py buat live
    inference ke video yang belum pernah lewat label_phase.py."""
    ok = feat_df["usable"].to_numpy()
    runs = []
    run_start = None
    for i in range(len(feat_df)):
        if ok[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                runs.append((run_start, i))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(feat_df)))
    return runs


def build_inference_windows(df, feat_df, runs, window_sec, stride=1, coord_mode="mean"):
    """Bangun window (fitur SAJA, tanpa class/phase/label apa pun) dari
    video yang BELUM dilabeli -- dipakai src/training/predict_video.py buat
    live inference nonstop ke video baru manapun (lihat find_usable_runs).

    coord_mode: HARUS SAMA dgn yg dipakai model yg akan dipakai predict
    (lihat build_windows_from_runs) -- default "mean" (produksi).

    Returns DataFrame (1 baris = 1 window: start/end frame + fitur) atau
    None kalau tidak ada window yang bisa dibangun.
    """
    fps = float(df["fps"].iloc[0]) if "fps" in df.columns else 30.0
    swl = max(1, round(fps * window_sec))

    windows = []
    for run_start, run_end in runs:
        run_len = run_end - run_start
        if run_len < swl:
            continue
        swm = run_len - swl + 1
        for w_start in range(run_start, run_start + swm, stride):
            window_feat = feat_df.iloc[w_start:w_start + swl]

            angle_block = {}
            for f in range(swl):
                for col in ANGLE_COLUMNS:
                    angle_block[f"{col}_f{f}"] = window_feat[col].iloc[f]

            coord_block = {}
            for name in POSE_LANDMARK_NAMES:
                for axis in ("x", "y", "z", "v"):
                    col = f"{name}_{axis}"
                    if coord_mode == "mean":
                        coord_block[f"coord_mean_{col}"] = window_feat[col].mean()
                    else:
                        for f in range(swl):
                            coord_block[f"coord_{col}_f{f}"] = window_feat[col].iloc[f]

            # NaN (angle ATAU coordinate) -- occlusion 1 titik jatuh di window ini
            if any(v != v for v in angle_block.values()) or any(v != v for v in coord_block.values()):
                continue

            row = {
                "start_frame_idx": int(df["frame_idx"].iloc[w_start]),
                "end_frame_idx": int(df["frame_idx"].iloc[w_start + swl - 1]),
                "window_len_frames": swl,
                "fps": fps,
            }
            row.update(angle_block)
            row.update(coord_block)
            windows.append(row)

    if not windows:
        return None
    return pd.DataFrame(windows)


def build_windows_for_video(df, window_sec, stride=1, visibility_threshold=VISIBILITY_THRESHOLD):
    """Bangun SEMUA window dari 1 video (tanpa split train/test) -- dipakai
    kalau memang tidak butuh pemisahan train/test (mis. eksplorasi/QA).
    Untuk pipeline training pakai compute_frame_features + find_valid_runs +
    split_runs_by_frame_fraction + build_windows_from_runs (lihat build_dataset.py).
    """
    df = df.reset_index(drop=True)
    feat_df = compute_frame_features(df, visibility_threshold)
    runs = find_valid_runs(df, feat_df)
    return build_windows_from_runs(df, feat_df, runs, window_sec, stride)
