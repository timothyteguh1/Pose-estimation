"""Tahap 2: preprocessing penuh -- landmark berlabel -> dataset siap training.

Urutan (bab 8.3.1 proposal, per-exercise -- 3 model terpisah ala Ko et al.):
    1. Hitung 11 joint angle per frame (Eq.1)                    [joint_angles.py]
    2. Filter visibility >= 0.6                                  [sliding_window.py]
    3. GLOBAL lintas semua video 1 exercise: run (1 segmen 1 fase) dikumpulkan
       per class, dialokasikan UTUH ke train ATAU test pakai "largest deficit
       first" supaya presisi ke 70:30 dan tiap kelas terwakili di kedua sisi.
       Window tidak pernah nyebrang batas run, jadi alokasi run utuh otomatis
       aman dari leakage. group_fragmented_runs() menggabung run kelas-sama
       yang cuma terpecah gap pendek (occlusion sesaat) supaya pecahan dari
       1 repetisi fisik yang sama selalu dialokasikan ke sisi yang sama.
    3.5. Normalisasi Min-Max (Eq.5) angle saja (11 kolom) di level frame, fit
       hanya dari frame train, transform ke semua frame -- sebelum windowing
       (bab 8.3.1). Coordinate sengaja tidak dinormalisasi (lihat
       window_scaler.py).                                       [window_scaler.py]
    4. Sliding window (Eq.2 & 3), dibangun terpisah utk train & test dari
       angle yang sudah ternormalisasi + coordinate mentah.     [sliding_window.py]
    5. Gabung semua video 1 exercise jadi 1 dataset train + 1 dataset test.

Video yang belum 100% dilabel manual (masih ada sisa 'auto') di-skip dengan
warning.

Output (per exercise, bab 8.2 proposal):
    data/windowed_features/{exercise}/{video_stem}_windowed.csv   <- per video, belum dinormalisasi
    data/windowed_features/splits/{exercise}_train.csv            <- ~70%, sudah dinormalisasi
    data/windowed_features/splits/{exercise}_test.csv             <- ~30%, sudah dinormalisasi

Usage:
    python src/features/build_dataset.py squat
    python src/features/build_dataset.py squat --window-sec 0.5 --test-size 0.3
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.joint_angles import ANGLE_COLUMNS, POSE_LANDMARK_NAMES
from src.features.sliding_window import (
    build_windows_from_runs,
    compute_frame_features,
    find_valid_runs,
)
from src.features.window_scaler import WindowFeatureScaler
from src.io_utils import atomic_write_csv

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACTED_DIR = REPO_ROOT / "data" / "extracted_landmarks"
WINDOWED_DIR = REPO_ROOT / "data" / "windowed_features"
SPLITS_DIR = WINDOWED_DIR / "splits"
MODELS_DIR = REPO_ROOT / "models"

META_COLUMNS = [
    "window_id", "exercise", "posture_class", "participant_id", "camera_angle",
    "take", "source_video", "class", "phase", "split",
    "start_frame_idx", "end_frame_idx", "start_timestamp_sec", "window_len_frames", "fps",
]


def find_labeled_videos(exercise):
    """Cari semua CSV di extracted_landmarks/{exercise} yang sudah 100%
    manual. rglob (rekursif) -- CSV boleh nested per-partisipan/per-kelas."""
    exercise_dir = EXTRACTED_DIR / exercise
    if not exercise_dir.exists():
        return [], []
    ready, skipped = [], []
    for p in sorted(exercise_dir.rglob("*.csv")):
        df = pd.read_csv(p, usecols=lambda c: c in ("label_source",))
        if "label_source" not in df.columns:
            skipped.append((p, "belum dilabel sama sekali"))
            continue
        n_auto = (df["label_source"] == "auto").sum()
        if n_auto > 0:
            skipped.append((p, f"masih ada {n_auto} frame 'auto' (belum 100% manual)"))
            continue
        ready.append(p)
    return ready, skipped


def load_video_runs(csv_path, visibility_threshold, use_z=False):
    """Baca 1 video, hitung fitur per-frame, cari run valid (Eq. filter bab 8.3.1).

    use_z: default False (2D, sama Ko et al) -- True HANYA utk eksperimen
    pembanding --use-z (lihat argparse), TIDAK memengaruhi kolom koordinat
    mentah (selalu 3D dr sananya), cuma memengaruhi ANGLE."""
    df = pd.read_csv(csv_path).reset_index(drop=True)
    feat_df = compute_frame_features(df, visibility_threshold, use_z)
    runs = find_valid_runs(df, feat_df)
    return df, feat_df, runs


def allocate_runs_balanced(items, test_fraction, rng):
    """Alokasikan run (UTUH, tidak dipotong) ke 'train'/'test' presisi ke
    target rasio, pakai heuristik "largest deficit first": run terbesar
    dialokasikan duluan, tiap kali ke sisi yang saat ini paling jauh dari
    targetnya. Jauh lebih presisi drpd greedy sekuensial biasa, terutama
    kalau jumlah run per kelompok sedikit.

    items: list of (key, length) -- key bebas (dipakai caller buat lacak
    balik run mana masuk sisi mana).

    Returns dict {key: 'train'/'test'}.
    """
    total = sum(length for _, length in items)
    if total == 0:
        return {}
    target_test = total * test_fraction
    target_train = total - target_test

    order = list(items)
    rng.shuffle(order)  # acak dulu buat tie-break yang tidak selalu sama
    order.sort(key=lambda kv: kv[1], reverse=True)

    assignment = {}
    test_sum = 0.0
    train_sum = 0.0
    for key, length in order:
        test_deficit = target_test - test_sum
        train_deficit = target_train - train_sum
        if test_deficit >= train_deficit:
            assignment[key] = "test"
            test_sum += length
        else:
            assignment[key] = "train"
            train_sum += length

    # kalau kebetulan 1 sisi kosong padahal run-nya >=2, paksa tukar 1 run
    # terkecil dari sisi mayoritas -- supaya kelas ini tetap terwakili di
    # kedua sisi (lebih penting daripada presisi rasio utk grup sekecil ini).
    sides = set(assignment.values())
    if len(items) >= 2 and len(sides) == 1:
        smallest_key = min(order, key=lambda kv: kv[1])[0]
        only_side = sides.pop()
        assignment[smallest_key] = "test" if only_side == "train" else "train"

    return assignment


def group_fragmented_runs(df, runs, max_gap_frames):
    """Gabungkan run berurutan kelas sama yang dipisah gap pendek
    (<=max_gap_frames) jadi 1 grup -- dianggap 1 repetisi fisik yang sama
    yang kebetulan terpecah gara-gara occlusion sesaat, bukan 2 repetisi
    independen. Tanpa ini, allocate_runs_balanced() bisa taruh 2 pecahan
    dari 1 rep yang sama ke train DAN test sekaligus.

    Returns list of list-of-run."""
    if not runs:
        return []
    groups = [[runs[0]]]
    for run in runs[1:]:
        prev_run = groups[-1][-1]
        gap = run[0] - prev_run[1]
        same_class = df["class"].iloc[run[0]] == df["class"].iloc[prev_run[0]]
        if same_class and 0 <= gap <= max_gap_frames:
            groups[-1].append(run)
        else:
            groups.append([run])
    return groups


def feature_columns(window_len_frames, coord_mode="mean"):
    """Urutan kolom fitur HARUS sama persis dgn yg dibangun
    build_windows_from_runs()/_predict_from_buffer() (live_pipeline.py).
    coord_mode="mean" (default/produksi) atau "flatten" (eksperimen
    pembanding --coord-mode, lihat sliding_window.py)."""
    cols = []
    for f in range(window_len_frames):
        for col in ANGLE_COLUMNS:
            cols.append(f"{col}_f{f}")
    for name in POSE_LANDMARK_NAMES:
        for axis in ("x", "y", "z", "v"):
            col = f"{name}_{axis}"
            if coord_mode == "mean":
                cols.append(f"coord_mean_{col}")
            else:
                for f in range(window_len_frames):
                    cols.append(f"coord_{col}_f{f}")
    return cols


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exercise", choices=["squat", "benchpress", "deadlift"])
    parser.add_argument("--window-sec", type=float, default=1.0,
                         help="t di Eq.2 (durasi window, detik).")
    parser.add_argument("--stride", type=int, default=1,
                         help="pergeseran antar window (frame), Eq.3 default 1.")
    parser.add_argument("--visibility-threshold", type=float, default=0.6)
    parser.add_argument("--max-fragment-gap", type=int, default=5,
                         help="run kelas sama berurutan dgn gap <= sekian frame dianggap "
                              "1 repetisi terpecah, digabung jadi 1 unit alokasi train/test")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--use-z", action="store_true",
                         help="eksperimen pembanding saja (default 2D, sama Ko et al) -- "
                              "pakai (x,y,z) utk hitung 11 angle. Output ke file '_3d'.")
    parser.add_argument("--coord-mode", choices=["flatten", "mean"], default="mean",
                         help="'mean' = produksi (default), koordinat dirata-rata 1 angka/window. "
                              "'flatten' = eksperimen pembanding saja, output ke file '_flatten'.")
    args = parser.parse_args()
    tag = args.exercise
    if args.use_z:
        tag += "_3d"
    if args.coord_mode == "flatten":
        tag += "_flatten"

    ready, skipped = find_labeled_videos(args.exercise)
    print(f"=== {args.exercise}: {len(ready)} video siap, {len(skipped)} video dilewati ===")
    for p, reason in skipped:
        print(f"[skip] {p.name}: {reason}")
    if not ready:
        print("Tidak ada video yang siap diproses.")
        return

    out_dir = WINDOWED_DIR / args.exercise
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.random_state)

    # --- 1. Baca semua video, kumpulkan run per class LINTAS SEMUA VIDEO ---
    videos = []  # list of dict: path, df, feat_df, runs
    pool_by_class = {}  # class -> list of ((video_idx, group_of_runs), length)
    for vi, p in enumerate(ready):
        df, feat_df, runs = load_video_runs(p, args.visibility_threshold, args.use_z)
        videos.append({"path": p, "df": df, "feat_df": feat_df})
        groups = group_fragmented_runs(df, runs, args.max_fragment_gap)
        for group in groups:
            cls = df["class"].iloc[group[0][0]]
            length = sum(r[1] - r[0] for r in group)
            pool_by_class.setdefault(cls, []).append(((vi, tuple(group)), length))
        n_merged = len(runs) - len(groups)
        merged_note = f", {n_merged} run digabung (gap pendek) jadi {len(groups)} unit alokasi" if n_merged else ""
        print(f"[baca] {p.name}: {len(runs)} run/repetisi ditemukan{merged_note}")

    # --- 2. Alokasikan run per class, GLOBAL lintas video ---
    per_video_train_runs = {vi: [] for vi in range(len(videos))}
    per_video_test_runs = {vi: [] for vi in range(len(videos))}
    print(f"\n=== Alokasi run per kelas (global, target train:test = "
          f"{(1 - args.test_size) * 100:.0f}:{args.test_size * 100:.0f}) ===")
    for cls, items in sorted(pool_by_class.items()):
        assignment = allocate_runs_balanced(items, args.test_size, rng)
        n_train = sum(1 for v in assignment.values() if v == "train")
        n_test = sum(1 for v in assignment.values() if v == "test")
        frames_train = sum(length for (key, length) in items if assignment.get(key) == "train")
        frames_test = sum(length for (key, length) in items if assignment.get(key) == "test")
        total_frames = frames_train + frames_test
        pct = frames_test / total_frames * 100 if total_frames else 0
        print(f"  {cls}: {n_train} run train + {n_test} run test "
              f"({frames_train} + {frames_test} frame, test={pct:.1f}%)")
        for (vi, group), length in items:
            target = per_video_train_runs[vi] if assignment[(vi, group)] == "train" else per_video_test_runs[vi]
            target.extend(group)  # semua run dalam 1 grup ikut ke sisi yang sama

    # Normalisasi angle (11 kolom, Eq.5) di level frame, sebelum windowing
    # (bab 8.3.1) -- fit hanya dari frame train, transform semua frame
    # (train+test), in-place, sebelum build_windows_from_runs() flatten jadi
    # kolom _f0.._fN. Coordinate sengaja tidak dinormalisasi (window_scaler.py).
    scaler = WindowFeatureScaler()
    train_frame_chunks = []
    for vi, v in enumerate(videos):
        for run_start, run_end in per_video_train_runs[vi]:
            train_frame_chunks.append(v["feat_df"].iloc[run_start:run_end][ANGLE_COLUMNS])
    train_frame_angles = (pd.concat(train_frame_chunks, ignore_index=True)
                           if train_frame_chunks else pd.DataFrame(columns=ANGLE_COLUMNS))
    scaler.fit_frame(train_frame_angles)
    for v in videos:
        v["feat_df"] = scaler.transform_frame(v["feat_df"])
    print(f"\n[ok] {len(ANGLE_COLUMNS)} kolom angle dinormalisasi di level frame "
          f"(dari {len(train_frame_angles)} frame train), SEBELUM windowing -- sesuai urutan bab 8.3.1 "
          f"(coordinate TIDAK dinormalisasi, sesuai Ko et al. & bab 6.2.4)")

    # --- 3. Bangun window per video, dari run yg sudah dialokasikan ---
    train_parts, test_parts = [], []
    for vi, v in enumerate(videos):
        train_w = build_windows_from_runs(
            v["df"], v["feat_df"], per_video_train_runs[vi], args.window_sec, args.stride,
            split_label="train", coord_mode=args.coord_mode)
        test_w = build_windows_from_runs(
            v["df"], v["feat_df"], per_video_test_runs[vi], args.window_sec, args.stride,
            split_label="test", coord_mode=args.coord_mode)
        n_train = len(train_w) if train_w is not None else 0
        n_test = len(test_w) if test_w is not None else 0
        print(f"[ok] {v['path'].name}: {n_train} window train + {n_test} window test")

        parts = [w for w in (train_w, test_w) if w is not None]
        if not parts:
            print(f"[warn] {v['path'].name}: 0 window dihasilkan sama sekali (durasi run < window_sec)")
            continue
        combined_video = pd.concat(parts, ignore_index=True)
        out_path = out_dir / f"{v['path'].stem}_windowed.csv"
        atomic_write_csv(combined_video, out_path)
        if train_w is not None:
            train_parts.append(train_w)
        if test_w is not None:
            test_parts.append(test_w)

    if not train_parts and not test_parts:
        print("Tidak ada window yang dihasilkan sama sekali.")
        return

    # Guard: swl dihitung per video dari fps video itu sendiri -- kalau ada
    # video fps beda, swl-nya beda, struktur kolom _f0.._fN antar video jadi
    # tidak match (pd.concat diam-diam isi NaN, bukan error). Wajib dicek eksplisit.
    all_lens = {int(w["window_len_frames"].iloc[0]) for w in (train_parts + test_parts)}
    if len(all_lens) > 1:
        print(f"\n[error] video-video ini punya window_len_frames (swl) BEDA: {sorted(all_lens)} "
              f"-- kemungkinan fps video tidak seragam. Tidak bisa digabung jadi 1 dataset "
              f"(kolom fitur bakal beda struktur / NaN diam-diam). Samakan dulu fps video "
              f"(re-export/convert) atau proses terpisah, baru jalankan lagi.")
        return

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()
    total = len(train_df) + len(test_df)

    print(f"\n=== Total {total} window dari {len(ready)} video "
          f"(train={len(train_df)}, test={len(test_df)}) ===")
    if total:
        print(f"Proporsi aktual: train={len(train_df)/total*100:.1f}%, test={len(test_df)/total*100:.1f}%")

    if train_df.empty or test_df.empty:
        print("[error] train atau test kosong -- tidak bisa lanjut normalisasi/simpan.")
        return

    window_len = int(train_df["window_len_frames"].iloc[0])
    feat_cols = feature_columns(window_len, coord_mode=args.coord_mode)

    print(f"\nDistribusi kelas train:\n{train_df['class'].value_counts().to_string()}")
    print(f"\nDistribusi kelas test:\n{test_df['class'].value_counts().to_string()}")

    missing = set(train_df["class"].unique()) - set(test_df["class"].unique())
    if missing:
        print(f"\n[error] kelas berikut 0 sampel di test: {missing} -- cek datanya!")
    else:
        print("\n[ok] semua kelas train ada juga di test.")

    # Fitur sudah dinormalisasi di level frame (langkah 2.5) -- configure_columns()
    # di sini bukan buat normalisasi train/test, tapi supaya scaler yang
    # disimpan tahu struktur kolom window, dibutuhkan predict_video.py.
    train_df = train_df.copy()
    test_df = test_df.copy()
    scaler.configure_columns(feat_cols)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_path = SPLITS_DIR / f"{tag}_train.csv"
    test_path = SPLITS_DIR / f"{tag}_test.csv"
    atomic_write_csv(train_df[META_COLUMNS + feat_cols], train_path)
    atomic_write_csv(test_df[META_COLUMNS + feat_cols], test_path)

    print(f"\n[ok] disimpan: {train_path}")
    print(f"[ok] disimpan: {test_path}")

    # Simpan scaler (Eq.5) + config kolom fitur -- dibutuhkan predict_video.py
    # utk normalisasi video baru dgn skala persis sama dgn training.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = MODELS_DIR / f"{tag}_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    # Rasio lebar:tinggi video raw training, disimpan per-exercise (dipakai
    # live_pipeline.py utk match_frame_aspect di expand_bbox()) -- exercise
    # beda bisa direkam dgn device/orientasi beda.
    import cv2
    from src.extraction.label_phase import find_source_video
    from src.io_utils import open_video_capture
    raw_video_path = find_source_video(args.exercise, f"{ready[0].stem}.mp4")
    training_aspect_ratio = None
    if raw_video_path is not None:
        cap = open_video_capture(str(raw_video_path))  # penting: baca dimensi SETELAH rotasi diterapkan
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if vw > 0 and vh > 0:
            training_aspect_ratio = vw / vh
    if training_aspect_ratio is None:
        print(f"[warn] tidak bisa baca dimensi video raw ({raw_video_path}) -- "
              f"training_video_aspect_ratio TIDAK disimpan, live_pipeline akan pakai fallback")

    config_path = MODELS_DIR / f"{tag}_feature_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "exercise": args.exercise,
            "use_z": args.use_z,
            "window_sec": args.window_sec,
            "stride": args.stride,
            "visibility_threshold": args.visibility_threshold,
            "training_video_aspect_ratio": training_aspect_ratio,
            "feat_cols": feat_cols,
        }, f, indent=2)
    print(f"[ok] disimpan: {scaler_path}")
    print(f"[ok] disimpan: {config_path}")


if __name__ == "__main__":
    main()
