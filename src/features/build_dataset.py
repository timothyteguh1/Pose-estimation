"""Tahap 2: preprocessing penuh -- landmark berlabel -> dataset siap training.

Urutan (bab 8.3.1 proposal, per-exercise karena kita pakai 3 model terpisah
ala Ko et al., bukan 1 model gabungan -- lihat diskusi di project_context.md):
    1. Hitung 11 joint angle per frame (Eq.1)                    [src/features/joint_angles.py]
    2. Filter visibility >= 0.6                                  [src/features/sliding_window.py]
    3. GLOBAL lintas semua video 1 exercise: run (1 segmen 1 fase = 1
       repetisi) dikumpulkan per class (concentric/eccentric x postur),
       lalu dialokasikan UTUH ke train ATAU test pakai algoritma "largest
       deficit first" (largest-processing-time-first bin balancing) supaya
       presisi ke 70:30 DAN tiap kelas tetap terwakili di kedua sisi.
       Window TIDAK PERNAH nyebrang batas run (lihat build_windows_from_runs),
       jadi alokasi run utuh otomatis aman dari leakage, TANPA perlu
       memotong run sama sekali. Riwayat kenapa sampai ke sini (4 percobaan):
         a) split acak per baris window -> BOCOR (window bertetangga overlap
            ~93%, bisa kepisah train/test)
         b) split per video utuh -> terlalu kasar (cuma 3 video/postur,
            jauh dari 70:30, 1 video test tidak representatif)
         c) potong kronologis (1x global per video ATAU per-run) -> segmen
            eccentric pendek, begitu dipotong jadi 2 bagian, KEDUA bagian
            sering < swl -> kelas itu 0 sampel total di 1/2 sisi
         d) alokasi run utuh TAPI per-video sendiri2 -> tiap video cuma
            9-18 run per kelas (kasar), overshoot jauh dari 70:30 (pernah
            dapat 52:48)
       (d) diperbaiki jadi GLOBAL (gabung run dari semua video per kelas
       dulu, baru dialokasikan) -- jauh lebih presisi karena jumlah run per
       kelas jadi puluhan-ratusan, bukan cuma belasan.
       e) [refinement lanjutan, bukan iterasi baru] ditemukan: kadang 1
       repetisi FISIK yang sama terpecah jadi 2 run gara-gara visibility
       jatuh sesaat di tengah gerakan (occlusion singkat, mis. tangan
       lewat posisi ke-blok) -- filter visibility Eq. ini SENGAJA "discard,
       bukan interpolasi", sama persis Ko et al. (dikonfirmasi baca ulang
       paper mereka: "data with visibility values vi<=0.6 were discarded"),
       jadi run tetap benar terpecah, TIDAK diperbaiki nilainya. Yang
       diperbaiki cuma ALOKASI-nya: tanpa penanganan, 2 pecahan rep yang
       sama bisa kejatah sisi BEDA (1 train, 1 test) -- bukan window-level
       leakage (frame tidak overlap), tapi kebocoran lebih halus (model
       dites pakai potongan repetisi fisik yang sama dgn yang dipakai
       training). group_fragmented_runs() gabung run kelas-sama-gap-pendek
       jadi 1 unit alokasi SEBELUM masuk allocate_runs_balanced(), supaya
       pecahan begini selalu dialokasikan ke sisi yang SAMA.
    3.5. Normalisasi Min-Max (Eq.5) ANGLE SAJA (11 kolom) di level FRAME, fit
       HANYA dari frame yang masuk alokasi train (langkah 3), transform ke
       SEMUA frame -- SEBELUM windowing, literal sesuai urutan bab 8.3.1
       ("normalisasi... berikutnya disusun ke dalam bentuk segmen... sliding
       window"). Coordinate SENGAJA TIDAK dinormalisasi -- bab 6.2.4 &
       Ko et al. (paper, "DATA NORMALIZATION") eksplisit cuma bicara joint
       angle; MediaPipe coordinate dianggap sudah cukup konsisten skalanya
       ([-1,1] native MediaPipe). Riwayat: sempat coba normalisasi
       angle+coordinate (143 kolom) supaya "konsisten", dikoreksi balik ke
       angle-saja setelah dicek ulang 3 sumber sepakat cuma angle.
       [src/features/window_scaler.py]
    4. Sliding window (Eq.2 & 3) -- dibangun terpisah utk sisi train & test
       dari tiap video, dari angle yang SUDAH ternormalisasi (flatten) +
       coordinate mentah (rata-rata, tanpa normalisasi apa pun)
                                                                    [src/features/sliding_window.py]
    5. Gabung semua video 1 exercise jadi 1 dataset train + 1 dataset test,
       simpan langsung (mencegah data leakage; ini juga beda dari kode Ko
       yang fit scaler ke seluruh data sebelum split, DAN beda dari kode Ko
       yang normalisasi di tahap training bukan preprocessing)

Video yang belum 100% dilabel manual (masih ada sisa 'auto') di-SKIP dengan
warning -- data belum direview manusia tidak boleh ikut jadi training/testing.

Output (folder dinamai per exercise spy gampang dilacak, bab 8.2 proposal):
    data/windowed_features/{exercise}/{video_stem}_windowed.csv   <- per video (train+test
                                                                       digabung, ada kolom 'split'
                                                                       buat traceability), belum dinormalisasi
    data/windowed_features/splits/{exercise}_train.csv            <- ~70%, SUDAH dinormalisasi
    data/windowed_features/splits/{exercise}_test.csv             <- ~30%, SUDAH dinormalisasi

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
    """Cari semua CSV di extracted_landmarks/{exercise} yang SUDAH 100% manual.
    rglob (REKURSIF, bukan glob) -- CSV boleh ditaruh nested per-partisipan/
    per-kelas (mis. extracted_landmarks/deadlift/p2/armsspread/...csv) buat
    kerapian, tetap ketemu. Ini MURNI cara mencari file di disk, TIDAK
    mengubah data/fitur/logika apa pun -- rglob adalah superset dari glob
    (semua yg dulu ketemu via glob tetap ketemu, plus yg di subfolder)."""
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
    """Gabungkan run BERURUTAN kelas SAMA yang dipisah gap PENDEK (<=max_gap_frames)
    jadi 1 grup -- dianggap 1 repetisi fisik yang sama yang kebetulan terpecah
    gara-gara visibility jatuh sesaat di tengah gerakan (mis. tangan sempat lewat
    posisi ke-occlude), BUKAN 2 repetisi independen.

    Tanpa ini, allocate_runs_balanced() bisa taruh 2 pecahan dari 1 rep yang sama
    ke sisi train DAN test sekaligus -- bukan window-level leakage (frame-nya tetap
    tidak overlap), tapi bentuk kebocoran lebih halus: model dites pakai potongan
    dari repetisi fisik yang SAMA dgn yang dipakai training, bukan repetisi yang
    benar-benar independen. Grouping ini TIDAK mengubah 1 pun nilai fitur -- cuma
    memastikan pecahan begini selalu dialokasikan ke sisi yang SAMA (train bareng,
    atau test bareng).

    Returns list of list-of-run (tiap grup = list berisi 1 atau lebih run tuple).
    """
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
                         help="t di Eq.2 (durasi window, detik). Default 1.0s -- BELUM final, "
                              "proposal minta dicoba beberapa nilai lewat eksperimen.")
    parser.add_argument("--stride", type=int, default=1,
                         help="pergeseran antar window (frame). Default 1 sesuai Eq.3 apa adanya.")
    parser.add_argument("--visibility-threshold", type=float, default=0.6)
    parser.add_argument("--max-fragment-gap", type=int, default=5,
                         help="run kelas sama, berurutan, dipisah gap <= sekian frame "
                              "dianggap 1 repetisi yang terpecah (visibility jatuh sesaat) "
                              "-- digabung jadi 1 unit alokasi train/test, tidak pernah kepisah")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--use-z", action="store_true",
                         help="EKSPERIMEN PEMBANDING SAJA (default OFF/2D, sama Ko et al -- "
                              "dicek langsung ke seluruh kode mereka, z tidak pernah dipakai "
                              "di rumus sudut manapun). --use-z pakai (x,y,z) utk hitung 11 "
                              "angle -- Eq.1 tetap sama persis, cuma dimensi vektornya beda, "
                              "BUKAN rumus baru. Output ditulis ke file BERTANDA '_3d' (tidak "
                              "menimpa hasil 2D produksi) supaya bisa dibandingkan.")
    parser.add_argument("--coord-mode", choices=["flatten", "mean"], default="mean",
                         help="'mean' = PRODUKSI (default) -- koordinat dirata-rata 1 angka/window "
                              "(132 kolom, 33 landmark x/y/z/v UTUH, tidak ada yg dibuang -- "
                              "match persis 'joint coordinate' Ko et al., 33 landmark). 'flatten' "
                              "= EKSPERIMEN PEMBANDING SAJA (sempat jadi produksi, DIKEMBALIKAN ke "
                              "'mean' setelah ditemukan flatten bikin RF didominasi koordinat mentah "
                              "(93.9% squat, 95.6% deadlift dari feature_importances_), bertentangan "
                              "dgn aturan keras #2 'joint angle fitur utama, koordinat pendukung' -- "
                              "lihat diskusi proyek). Output 'flatten' ditulis ke file BERTANDA "
                              "'_flatten' (tidak menimpa produksi 'mean').")
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

    # --- 2.5 Normalisasi ANGLE (11 kolom, Eq.5) di level FRAME, SEBELUM
    # windowing (bab 8.3.1: "normalisasi... berikutnya disusun ke dalam
    # bentuk segmen... sliding window"). Coordinate SENGAJA TIDAK
    # dinormalisasi -- bab 6.2.4 & Ko et al. (paper, "DATA NORMALIZATION")
    # eksplisit cuma bicara joint angle; MediaPipe coordinate dianggap sudah
    # cukup konsisten skalanya ([-1,1] native). Lihat window_scaler.py utk
    # kutipan lengkap & alasan. Fit HANYA dari frame yang masuk alokasi TRAIN,
    # transform SEMUA frame (train+test) di semua video, in-place, SEBELUM
    # build_windows_from_runs() flatten nilainya jadi kolom _f0.._fN.
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

    # GUARD: swl (window_len_frames) dihitung per video dari fps video itu
    # sendiri (round(fps * window_sec)). Kalau ada video dengan fps beda
    # (mis. direkam HP lain di 25fps/60fps), swl-nya beda -> nama kolom
    # angle (_f0.._fN) beda struktur antar video -> pd.concat diam-diam
    # ngisi NaN di kolom yang tidak match, BUKAN error. Wajib dicek eksplisit
    # di sini spy ketahuan jelas, bukan nyusup jadi data rusak.
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

    # SEMUA fitur (angle+coordinate) SUDAH dinormalisasi di langkah 2.5, di
    # level FRAME, SEBELUM window ini dibangun -- tidak ada normalisasi
    # susulan di sini lagi (window cuma menyusun ulang nilai yg sudah [0,1]).
    # configure_columns() TETAP dipanggil -- bukan buat normalisasi train/test
    # (sudah beres), tapi supaya scaler yang DISIMPAN nanti tahu struktur
    # kolom window, dibutuhkan predict_video.py buat normalisasi video BARU
    # (window MENTAH total) lewat transform() nanti.
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

    # Simpan scaler (Eq.5, sudah di-fit di atas) + config kolom fitur --
    # dibutuhkan nanti kalau mau prediksi ke video BARU di luar dataset ini
    # (lihat diskusi src/app/): angka mentah video baru harus dikonversi
    # pakai skala persis yang sama dgn training, scaler ini yang nyimpen itu.
    # TIDAK mengubah cara training/metodologi kita -- cuma nyimpen artefak
    # yang sudah ada di proses ini, sebelumnya kelupaan disimpan.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = MODELS_DIR / f"{tag}_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    # Rasio lebar:tinggi video RAW training (bukan CSV landmark) -- dibaca dari
    # video pertama yg dipakai (asumsi 1 exercise = 1 setup kamera/device yg
    # konsisten, sesuai protokol rekaman kita). Disimpan PER-EXERCISE di sini
    # (bukan angka tetap global di kode) -- dipakai live_pipeline.py utk
    # match_frame_aspect di expand_bbox() (lihat yolo_detector.py), supaya
    # kalau squat/deadlift ternyata direkam dgn device/orientasi beda dari
    # benchpress, angka target rasio-nya ikut benar per-exercise, bukan
    # ketinggalan pakai angka benchpress.
    import cv2
    from src.extraction.label_phase import find_source_video
    from src.io_utils import open_video_capture
    # find_source_video() (BUKAN path flat manual) -- cari rekursif, video
    # boleh nested per-partisipan/per-kelas (lihat diskusi proyek soal
    # rapikan folder data), sama seperti dipakai label_phase.py/preview_*.py.
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
