"""Live-preview prediksi model RF di atas video asli (read-only, baca saja).

Ambil 1 video dari TEST SET (`data/windowed_features/splits/{exercise}_test.csv`),
buka video mentahnya, mainkan, overlay skeleton + PREDIKSI MODEL vs GROUND
TRUTH (label asli) per frame -- pakai fitur yang SAMA PERSIS dgn yang dipakai
pas evaluasi (`train_model.py`), jadi ini reproduksi visual dari angka
evaluasi, bukan hitungan baru.

PENTING: video ini "bolong-bolong" -- cuma bagian yang windownya kebetulan
dialokasikan ke TEST yang ada overlay prediksi. Bagian yang windownya masuk
TRAIN (mayoritas frame di video yang sama) ditandai "bukan bagian test",
bukan tidak ada datanya.

Mode `--test-only`: skip semua bagian train (abu-abu) sama sekali -- video
cuma diputar per SEGMEN test (kelompok frame test yang berurutan). Sebelum
tiap segmen mulai, muncul kartu jeda (nama video + prediksi vs ground truth)
supaya jelas ini video/segmen mana dan tebakan modelnya apa, lanjut dgn SPACE.

Usage:
    python src/training/preview_prediction.py squat                     # video test pertama yg ketemu
    python src/training/preview_prediction.py squat --video squat_backbend_p1_right45_take01
    python src/training/preview_prediction.py squat --list              # cuma list video test yg tersedia
    python src/training/preview_prediction.py squat --test-only         # skip bagian train, per-segmen + kartu jeda
    python src/training/preview_prediction.py squat --all --test-only   # SEMUA video test, berurutan
    python src/training/preview_prediction.py squat --all --test-only --posture backbend  # cuma posture ini (semua sudut kamera)

Kontrol saat live view:
    space        play / pause  (mode --test-only: di kartu jeda, space = lanjut ke segmen)
    n            loncat ke video berikutnya (mode --all)
    , / .        mundur / maju 1 frame (saat pause)
    [ / ]        pelan / cepat-kan playback
    q / ESC      keluar (stop total, tidak lanjut ke video berikutnya)
"""
import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.extraction.label_phase import find_source_video
from src.features.build_dataset import META_COLUMNS
from src.features.skeleton_draw import draw_skeleton
from src.io_utils import open_video_capture

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "windowed_features" / "splits"
EXTRACTED_DIR = REPO_ROOT / "data" / "extracted_landmarks"
MODELS_DIR = REPO_ROOT / "models"

CORRECT_COLOR = (60, 200, 60)   # hijau: prediksi = ground truth
WRONG_COLOR = (40, 40, 230)     # merah: prediksi != ground truth
NOT_TESTED_COLOR = (140, 140, 140)  # abu: frame ini bukan bagian window test


def load_model(exercise):
    """Load {exercise}_rf.pkl. File ini sekarang Pipeline(scaler, rf) --
    lihat train_model.py. Tool ini kerja dari squat_test.csv yang SUDAH
    ternormalisasi (build_dataset.py), jadi kita ambil cuma step 'rf'-nya
    (classifier mentah), BUKAN pipeline.predict() penuh -- kalau dipakai
    penuh, data yang sudah ternormalisasi bakal di-scale 2x (salah).
    Untuk prediksi ke video BARU (belum ternormalisasi) pakai pipeline penuh
    -- lihat predict_video.py."""
    path = MODELS_DIR / f"{exercise}_rf.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} tidak ada -- jalankan dulu: python src/training/train_model.py {exercise}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if hasattr(obj, "named_steps") and "rf" in obj.named_steps:
        return obj.named_steps["rf"]
    return obj


def build_frame_predictions(test_df, model, feat_cols, video_stem):
    """Untuk 1 video: dict {frame_idx: (predicted_class, actual_class)} dari
    semua window test milik video itu (window bertetangga overlap -- kalau 1
    frame ke-cover >1 window, yang terakhir diproses yang menang, cukup buat
    tujuan visual)."""
    rows = test_df[test_df["source_video"] == f"{video_stem}.mp4"]
    if rows.empty:
        return {}
    y_pred = model.predict(rows[feat_cols])
    frame_map = {}
    for (_, row), pred in zip(rows.iterrows(), y_pred):
        for f in range(int(row["start_frame_idx"]), int(row["end_frame_idx"]) + 1):
            frame_map[f] = (pred, row["class"])
    return frame_map


def build_test_segments(frame_map):
    """Kelompokkan frame_idx di frame_map (test) jadi segmen frame yang
    BERURUTAN (contiguous) -- dipakai mode --test-only supaya bisa loncat
    per segmen, skip total bagian train di antaranya.

    Returns list of (start_frame, end_frame_inclusive), terurut kronologis.
    Prediksi/aktual buat kartu jeda diambil dari frame pertama tiap segmen
    (dalam 1 segmen prediksi bisa saja beda antar-frame kalau ada window
    overlap dgn hasil beda -- overlay per-frame saat play tetap akurat,
    kartu jeda cuma ringkasan pembuka)."""
    idxs = sorted(frame_map.keys())
    if not idxs:
        return []
    segments = []
    seg_start = idxs[0]
    prev = idxs[0]
    for i in idxs[1:]:
        if i == prev + 1:
            prev = i
            continue
        segments.append((seg_start, prev))
        seg_start = i
        prev = i
    segments.append((seg_start, prev))
    return segments


def show_segment_card(win_name, start, end, predicted, actual, seg_num, total_segments,
                       video_stem, video_pos=None):
    """Kartu jeda sebelum 1 segmen test mulai diputar -- nama video, nomor
    segmen, prediksi vs ground truth. Tunggu SPACE buat lanjut, q/ESC keluar.
    video_pos: opsional (n, total) posisi video ke berapa (mode --all).
    Returns True kalau lanjut, False kalau user keluar."""
    is_correct = predicted == actual
    color = CORRECT_COLOR if is_correct else WRONG_COLOR
    mark = "BENAR" if is_correct else "SALAH"
    video_line = f"Video: {video_stem}"
    if video_pos:
        video_line += f"   [video {video_pos[0]}/{video_pos[1]}]"
    lines = [
        (video_line, (255, 255, 255)),
        (f"Segmen test {seg_num}/{total_segments}  (frame {start}-{end})", (255, 255, 255)),
        ("", (0, 0, 0)),
        (f"Prediksi model : {predicted}", color),
        (f"Ground truth   : {actual}", color),
        (f"[{mark}]", color),
        ("", (0, 0, 0)),
        ("SPACE = mulai   n = video berikutnya   q = keluar", (200, 200, 200)),
    ]
    card = np.zeros((360, 760, 3), dtype=np.uint8)
    for i, (line, col) in enumerate(lines):
        if line:
            cv2.putText(card, line, (20, 40 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, col, 2, cv2.LINE_AA)
    cv2.imshow(win_name, card)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord(" "):
            return "play"
        if key == ord("n"):
            return "next_video"
        if key in (ord("q"), 27):
            return "quit"


def play_segment(cap, win_name, landmarks_df, frame_map, start, end, fps, speed):
    """Putar 1 segmen test (frame start..end inklusif) dgn kontrol
    play/pause/step/speed yang sama seperti mode full-video.

    Returns (status, speed) -- status: "done" (segmen selesai, lanjut segmen
    berikutnya), "next_video" (user tekan n), "quit" (user tekan q/ESC)."""
    n_frames = len(landmarks_df)
    idx = start
    cap_pos = -1
    playing = True

    while True:
        if idx != cap_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            cap_pos = idx
        ok, frame = cap.read()
        if not ok:
            return "done", speed
        cap_pos = idx + 1

        info = frame_map.get(idx)
        if info is None:
            color = NOT_TESTED_COLOR
            status_line = "(frame di luar window test -- seharusnya tidak terjadi di mode --test-only)"
        else:
            predicted, actual = info
            is_correct = predicted == actual
            color = CORRECT_COLOR if is_correct else WRONG_COLOR
            mark = "BENAR" if is_correct else "SALAH"
            status_line = f"prediksi={predicted}  aktual={actual}  [{mark}]"

        draw_skeleton(frame, landmarks_df.iloc[idx], color=color)
        lines = [
            f"frame {idx}/{n_frames - 1}  (segmen {start}-{end})   {'PLAY' if playing else 'PAUSE'} {speed:.2f}x",
            status_line,
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 32 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2, cv2.LINE_AA)

        cv2.imshow(win_name, frame)
        wait_ms = max(1, int(1000.0 / (fps * speed))) if playing else 0
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == 255:
            if playing:
                if idx >= end:
                    return "done", speed
                idx += 1
            continue

        if key in (ord("q"), 27):
            return "quit", speed
        elif key == ord("n"):
            return "next_video", speed
        elif key == ord(" "):
            playing = not playing
        elif key == ord(","):
            playing = False
            idx = max(start, idx - 1)
        elif key == ord("."):
            playing = False
            idx = min(end, idx + 1)
        elif key == ord("["):
            speed = max(0.1, round(speed - 0.25, 2))
        elif key == ord("]"):
            speed = min(4.0, round(speed + 0.25, 2))


def run_test_only_view(video_path, landmarks_df, frame_map, exercise, video_stem, speed=1.0,
                        video_pos=None):
    """Mode --test-only: skip semua bagian train (abu-abu), putar cuma
    segmen-segmen test berurutan, tiap segmen didahului kartu jeda (nama
    video + prediksi vs ground truth), lanjut dgn SPACE.

    Returns True kalau selesai normal (lanjut ke video berikutnya di mode
    --all), False kalau user tekan q/ESC (stop total)."""
    segments = build_test_segments(frame_map)
    if not segments:
        print("[error] tidak ada window test untuk video ini.")
        return True

    fps = float(landmarks_df["fps"].iloc[0]) if "fps" in landmarks_df.columns else 30.0
    cap = open_video_capture(str(video_path))
    if not cap.isOpened():
        print(f"[warn] gagal membuka {video_path}")
        return True

    win_name = f"prediksi model ({exercise}, test-only)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    print(f"[info] {len(segments)} segmen test ditemukan untuk {video_stem}")

    keep_going = True
    for seg_num, (start, end) in enumerate(segments, start=1):
        predicted, actual = frame_map[start]
        action = show_segment_card(win_name, start, end, predicted, actual, seg_num,
                                    len(segments), video_stem, video_pos)
        if action == "quit":
            keep_going = False
            break
        if action == "next_video":
            break
        status, speed = play_segment(cap, win_name, landmarks_df, frame_map, start, end, fps, speed)
        if status == "quit":
            keep_going = False
            break
        if status == "next_video":
            break

    cap.release()
    cv2.destroyWindow(win_name)
    return keep_going


def run_live_view(video_path, landmarks_df, frame_map, exercise, speed=1.0):
    """Mode default: putar video UTUH, bagian train ditandai abu-abu.

    Returns True kalau selesai normal (lanjut ke video berikutnya di mode
    --all), False kalau user tekan q/ESC (stop total)."""
    cap = open_video_capture(str(video_path))
    if not cap.isOpened():
        print(f"[warn] gagal membuka {video_path}")
        return True

    n_frames = len(landmarks_df)
    fps = float(landmarks_df["fps"].iloc[0]) if "fps" in landmarks_df.columns else 30.0
    idx = 0
    playing = True
    cap_pos = 0
    win_name = f"prediksi model ({exercise})"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    keep_going = True

    n_tested_frames = sum(1 for f in range(n_frames) if f in frame_map)
    print(f"[info] {n_tested_frames}/{n_frames} frame ({n_tested_frames/n_frames*100:.1f}%) "
          f"masuk window test (sisanya train, abu-abu)")

    while True:
        if idx != cap_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            cap_pos = idx
        ok, frame = cap.read()
        if not ok:
            playing = False
            idx = max(0, min(idx, n_frames - 1))
            cap_pos = -1
            continue
        cap_pos = idx + 1

        info = frame_map.get(idx)
        if info is None:
            color = NOT_TESTED_COLOR
            status_line = "BUKAN bagian test (window video ini masuk train)"
        else:
            predicted, actual = info
            is_correct = predicted == actual
            color = CORRECT_COLOR if is_correct else WRONG_COLOR
            mark = "BENAR" if is_correct else "SALAH"
            status_line = f"prediksi={predicted}  aktual={actual}  [{mark}]"

        draw_skeleton(frame, landmarks_df.iloc[idx], color=color)

        lines = [
            f"frame {idx}/{n_frames - 1}   {'PLAY' if playing else 'PAUSE'} {speed:.2f}x",
            status_line,
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 32 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2, cv2.LINE_AA)

        cv2.imshow(win_name, frame)
        wait_ms = max(1, int(1000.0 / (fps * speed))) if playing else 0
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == 255:
            if playing:
                if idx >= n_frames - 1:
                    playing = False
                else:
                    idx += 1
            continue

        if key in (ord("q"), 27):
            keep_going = False
            break
        elif key == ord("n"):  # lanjut ke video berikutnya (mode --all)
            break
        elif key == ord(" "):
            playing = not playing
        elif key == ord(","):
            playing = False
            idx = max(0, idx - 1)
        elif key == ord("."):
            playing = False
            idx = min(n_frames - 1, idx + 1)
        elif key == ord("["):
            speed = max(0.1, round(speed - 0.25, 2))
        elif key == ord("]"):
            speed = min(4.0, round(speed + 0.25, 2))

    cap.release()
    cv2.destroyWindow(win_name)
    return keep_going


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exercise", choices=["squat", "benchpress", "deadlift"])
    parser.add_argument("--video", default=None,
                         help="nama video (tanpa .mp4), default: video test pertama yg ketemu")
    parser.add_argument("--list", action="store_true", help="cuma tampilkan daftar video test yg tersedia")
    parser.add_argument("--test-only", action="store_true",
                         help="skip bagian train (abu-abu) total -- putar cuma segmen test, "
                              "tiap segmen didahului kartu jeda (nama video + prediksi vs ground truth)")
    parser.add_argument("--all", action="store_true",
                         help="putar SEMUA video yang punya window test di test csv, berurutan "
                              "(tanpa flag ini cuma 1 video). Tekan n buat loncat ke video berikutnya.")
    parser.add_argument("--posture", default=None,
                         help="cuma video dgn posture_class ini (mis. 'backbend', 'correct', "
                              "'kneeinward') -- kombinasikan dgn --all buat muter semua sudut "
                              "kamera 1 posture_class saja tanpa ganti argumen --video 1-1")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    test_path = SPLITS_DIR / f"{args.exercise}_test.csv"
    if not test_path.exists():
        print(f"[error] {test_path} tidak ada -- jalankan dulu build_dataset.py & train_model.py.")
        return
    test_df = pd.read_csv(test_path)
    feat_cols = [c for c in test_df.columns if c not in META_COLUMNS]

    available = sorted(v.replace(".mp4", "") for v in test_df["source_video"].unique())

    if args.posture:
        by_posture = test_df[test_df["posture_class"] == args.posture]
        matched = sorted(v.replace(".mp4", "") for v in by_posture["source_video"].unique())
        if not matched:
            all_postures = sorted(test_df["posture_class"].unique())
            print(f"[error] posture_class '{args.posture}' tidak ada window test-nya. "
                  f"Pilihan: {all_postures}")
            return
        available = matched

    if args.list or not available:
        print(f"Video yang punya window test ({args.exercise}):")
        for v in available:
            n = (test_df["source_video"] == f"{v}.mp4").sum()
            print(f"  {v}  ({n} window test)")
        if not args.list:
            print("\n[error] tidak ada video test.")
        return

    if args.all:
        stems = available
        if args.video:
            print("[warn] --all dipakai, argumen --video diabaikan.")
    else:
        stems = [args.video or available[0]]
        if stems[0] not in available:
            print(f"[error] '{stems[0]}' tidak punya window test. Pilihan: {available}")
            return

    model = load_model(args.exercise)
    posture_note = f", posture_class='{args.posture}'" if args.posture else ""
    print(f"[info] preview prediksi {args.exercise}: {len(stems)} video "
          f"({'semua video test' if args.all else 'satu video'}{posture_note})")

    for n, video_stem in enumerate(stems, start=1):
        frame_map = build_frame_predictions(test_df, model, feat_cols, video_stem)

        landmarks_path = EXTRACTED_DIR / args.exercise / f"{video_stem}.csv"
        if not landmarks_path.exists():
            print(f"[warn] {landmarks_path} tidak ada -- video ini dilewati.")
            continue
        landmarks_df = pd.read_csv(landmarks_path)

        video_path = find_source_video(args.exercise, f"{video_stem}.mp4")
        if video_path is None:
            print(f"[warn] video mentah {video_stem}.mp4 tidak ditemukan -- dilewati.")
            continue

        print(f"[info] ({n}/{len(stems)}) {video_stem}")
        if args.test_only:
            keep_going = run_test_only_view(video_path, landmarks_df, frame_map, args.exercise,
                                             video_stem, args.speed, video_pos=(n, len(stems)))
        else:
            keep_going = run_live_view(video_path, landmarks_df, frame_map, args.exercise, args.speed)

        if not keep_going:
            print("[info] dihentikan user (q).")
            break


if __name__ == "__main__":
    main()
