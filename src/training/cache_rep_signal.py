"""Tahap 1/2 validasi rep-counting: cache sinyal MENTAH (primary angle + 11
joint angle per frame) dari jalur PRODUKSI ASLI (YOLO+crop+padding+MediaPipe,
PERSIS LivePosturePipeline) -- supaya tahap 2 (tune_rep_counting.py) bisa
sweep parameter (prominence/distance/countdown_sec) berkali-kali TANPA perlu
re-run YOLO/MediaPipe tiap kali (parameter itu TIDAK memengaruhi sinyal
mentahnya sama sekali, cuma memengaruhi cara MENGHITUNG dari sinyal yang
sama).

countdown_sec dipaksa 0 saat caching (BUKAN mengubah countdown produksi --
cuma supaya SEMUA frame ikut tercatat, jadi macam-macam nilai countdown_sec
bisa disimulasikan ulang nanti di tune_rep_counting.py tanpa run ulang
YOLO/MediaPipe).

Usage:
    python src/training/cache_rep_signal.py squat data/raw_videos/squat/p2
    python src/training/cache_rep_signal.py deadlift data/raw_videos/deadlift/p1/correct/deadlift_correct_p1_front_take01.mp4

Output: 1 CSV per video di data/rep_signal_cache/{exercise}/{video_stem}.csv
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.app.live_pipeline import LivePosturePipeline
from src.features.joint_angles import compute_frame_angles
from src.io_utils import open_video_capture

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_VIDEOS_DIR = REPO_ROOT / "data" / "raw_videos"
CACHE_DIR = REPO_ROOT / "data" / "rep_signal_cache"


def find_videos(inputs):
    videos = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            videos.extend(sorted(p.rglob("*.mp4")))
        elif p.is_file():
            videos.append(p)
        else:
            print(f"[skip] path tidak ditemukan: {p}")
    return videos


def gt_csv_for(video_path, exercise):
    """CSV extracted_landmarks yang cocok (buat catat n_head_excluded_frames
    referensi -- BUKAN dipakai hitung sinyal, cuma metadata tambahan)."""
    parts = list(video_path.parts)
    try:
        parts[parts.index("raw_videos")] = "extracted_landmarks"
    except ValueError:
        return None
    return Path(*parts).with_suffix(".csv")


def cache_one_video(pipeline, video_path, exercise, out_path, force=False):
    if out_path.exists() and not force:
        print(f"[skip] {video_path.name} sudah ada cache ({out_path.name})")
        return

    n_head_excluded = 0
    gt_csv = gt_csv_for(video_path, exercise)
    if gt_csv is not None and gt_csv.exists():
        phase = pd.read_csv(gt_csv, usecols=["phase"])["phase"].tolist()
        for p in phase:
            if p == "excluded":
                n_head_excluded += 1
            else:
                break

    cap = open_video_capture(str(video_path))
    if not cap.isOpened():
        print(f"[error] gagal membuka video: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pipeline._session_start_ts = None
    pipeline._ensure_buffer(fps)
    pipeline._buffer.clear()

    rows = []
    idx = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        result = pipeline.process_frame(frame, t, fps=fps)
        angles, _ = compute_frame_angles(result["row"], pipeline.visibility_threshold)
        row = {"frame_idx": idx, "t": t}
        row.update(angles)
        rows.append(row)
        idx += 1
    cap.release()

    df = pd.DataFrame(rows)
    df["fps"] = fps
    df["n_head_excluded_frames_label"] = n_head_excluded
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[ok] {video_path.name}: {idx} frame, {time.time()-t0:.1f}s -> {out_path.relative_to(REPO_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exercise", choices=["squat", "benchpress", "deadlift"])
    parser.add_argument("inputs", nargs="+", help="video file(s) atau folder berisi video")
    parser.add_argument("--force", action="store_true", help="cache ulang walau sudah ada")
    args = parser.parse_args()

    videos = find_videos(args.inputs)
    if not videos:
        print("Tidak ada video ditemukan.")
        return

    print(f"[info] load model+YOLO {args.exercise} ...")
    pipeline = LivePosturePipeline(args.exercise, countdown_sec=0.0)

    for video_path in videos:
        out_path = CACHE_DIR / args.exercise / (video_path.stem + ".csv")
        cache_one_video(pipeline, video_path, args.exercise, out_path, force=args.force)

    print(f"\nSelesai: {len(videos)} video diproses.")


if __name__ == "__main__":
    main()
