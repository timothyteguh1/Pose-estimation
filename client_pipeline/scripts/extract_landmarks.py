import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.extraction.filename_parser import FilenameParseError, parse_video_filename
from src.features.joint_angles import POSE_LANDMARK_NAMES
from src.detection.yolo_detector import PersonDetector, expand_bbox, TRAINING_VIDEO_ASPECT_RATIO
from src.io_utils import atomic_write_csv, open_video_capture

RAW_VIDEOS_DIR = REPO_ROOT / "data" / "raw_videos"
DEFAULT_OUT_DIR = EXPERIMENT_ROOT / "data" / "extracted_landmarks"

LANDMARK_COLUMNS = [
    f"{name}_{axis}" for name in POSE_LANDMARK_NAMES for axis in ("x", "y", "z", "v")
]


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


def extract_one_video(video_path, min_detection_confidence, min_tracking_confidence,
                       yolo_confidence=0.7, crop_padding=0.75):
    cap = open_video_capture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"gagal membuka video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows = []

    detector = PersonDetector(confidence_threshold=yolo_confidence)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(EXPERIMENT_ROOT / "models" / "pose_landmarker_full.task")),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        bbox = detector.detect(frame)
        pose_input = frame
        if bbox:
            h, w = frame.shape[:2]
            crop_box = expand_bbox(bbox, crop_padding, w, h, target_aspect_ratio=TRAINING_VIDEO_ASPECT_RATIO)
            pose_input = frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]].copy()

        rgb = cv2.cvtColor(pose_input, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(frame_idx * 1000 / fps)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        row = {
            "frame_idx": frame_idx,
            "timestamp_sec": frame_idx / fps,
        }
        if bbox and result.pose_landmarks:
            lm = result.pose_landmarks[0]
            for name, p in zip(POSE_LANDMARK_NAMES, lm):
                row[f"{name}_x"] = p.x
                row[f"{name}_y"] = p.y
                row[f"{name}_z"] = p.z
                row[f"{name}_v"] = p.visibility
        else:
            for col in LANDMARK_COLUMNS:
                row[col] = float("nan")

        rows.append(row)
        frame_idx += 1

    cap.release()
    landmarker.close()
    return pd.DataFrame(rows), fps


def _relative_output_subpath(video_path, exercise):
    exercise_root = RAW_VIDEOS_DIR / exercise
    try:
        rel = video_path.resolve().relative_to(exercise_root.resolve())
    except ValueError:
        rel = Path(video_path.name)
    return rel.with_suffix(".csv")


def process_video(video_path, out_dir, min_detection_confidence, min_tracking_confidence, force=False,
                   yolo_confidence=0.7, crop_padding=0.75):
    try:
        meta = parse_video_filename(video_path)
    except FilenameParseError as e:
        print(f"[skip] {e}")
        return None

    out_path = Path(out_dir) / meta["exercise"] / _relative_output_subpath(video_path, meta["exercise"])
    if out_path.exists() and not force:
        print(f"[skip] {out_path.name} sudah ada, pakai --force utk ekstrak ulang")
        return None

    print(f"[extract] {video_path.name} (exercise={meta['exercise']}, "
          f"posture_class={meta['posture_class']}, camera_angle={meta['camera_angle']}, "
          f"take={meta['take']})")

    df, fps = extract_one_video(video_path, min_detection_confidence, min_tracking_confidence,
                                 yolo_confidence=yolo_confidence, crop_padding=crop_padding)
    for key, value in meta.items():
        df[key] = value
    df["fps"] = fps

    ordered_cols = (
        ["exercise", "posture_class", "participant_id", "camera_angle", "take",
         "source_video", "fps", "frame_idx", "timestamp_sec"]
        + LANDMARK_COLUMNS
    )
    df = df[ordered_cols]
    atomic_write_csv(df, out_path)

    detected = df[LANDMARK_COLUMNS[0]].notna().sum()
    print(f"  -> {out_path} ({len(df)} frame, {detected} frame terdeteksi pose)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--yolo-confidence", type=float, default=0.7)
    parser.add_argument("--crop-padding", type=float, default=0.75)
    args = parser.parse_args()

    videos = find_videos(args.inputs)
    if not videos:
        print("Tidak ada video ditemukan.")
        return

    results = []
    for video_path in videos:
        out_path = process_video(
            video_path, args.out_dir,
            args.min_detection_confidence, args.min_tracking_confidence,
            force=args.force, yolo_confidence=args.yolo_confidence, crop_padding=args.crop_padding,
        )
        if out_path is not None:
            results.append(out_path)

    print(f"\nSelesai: {len(results)}/{len(videos)} video berhasil diekstrak.")


if __name__ == "__main__":
    main()
