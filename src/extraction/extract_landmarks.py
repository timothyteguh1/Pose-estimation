"""Tahap 1a: ekstraksi MediaPipe Pose landmarks dari raw video -> CSV per frame.

Usage:
    python src/extraction/extract_landmarks.py data/raw_videos/squat
    python src/extraction/extract_landmarks.py data/raw_videos/squat/squat_correct_p1_left45_take01.mp4
    python src/extraction/extract_landmarks.py data/raw_videos --out-dir data/extracted_landmarks

Output: one CSV per video at data/extracted_landmarks/{exercise}/{video_stem}.csv
-- KALAU video mentahnya ditaruh nested per-partisipan/per-kelas (mis.
data/raw_videos/deadlift/p2/armsspread/xxx.mp4, lihat diskusi proyek soal
rapikan folder data), CSV hasilnya ikut nested SAMA PERSIS strukturnya
(data/extracted_landmarks/deadlift/p2/armsspread/xxx.csv) -- lihat
_relative_output_subpath(). Video flat tetap hasilkan CSV flat (kompatibel
dgn data lama). Metadata columns (exercise, posture_class, participant_id,
camera_angle, take, source_video, frame_idx, timestamp_sec) + 132 raw landmark
columns (one x/y/z/v per MediaPipe landmark). No 'class'/'phase' column yet --
that's added by label_phase.py.
"""
import argparse
import sys
from pathlib import Path

# PENTING (murni soal urutan import di Windows, TIDAK ada hubungannya dgn
# logika/metodologi): torch (dipakai YOLOv11 via ultralytics, lewat
# PersonDetector kalau --use-yolo-crop dipakai) HARUS di-import SEBELUM
# pandas -- kalau kebalik, keduanya rebutan runtime DLL yg sama (c10.dll)
# dan proses bisa CRASH (WinError 1114) -- ditemukan+dikonfirmasi langsung
# lewat tes terisolasi (lihat diskusi proyek). Baris ini SENGAJA di atas
# `import pandas`, walau modul ini sendiri baru pakai torch belakangan
# (import lokal di dalam PersonDetector, opsional).
import torch  # noqa: F401

import cv2
import mediapipe as mp
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.extraction.filename_parser import FilenameParseError, parse_video_filename
from src.features.joint_angles import POSE_LANDMARK_NAMES
from src.io_utils import atomic_write_csv, open_video_capture

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "extracted_landmarks"
RAW_VIDEOS_DIR = REPO_ROOT / "data" / "raw_videos"

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
                        use_yolo_crop=False, yolo_confidence=0.7, crop_padding=0.75):
    """Run MediaPipe Pose over every frame of one video. Returns a DataFrame.

    use_yolo_crop (BARU -- lihat diskusi proyek soal mismatch train/live):
    - False (default lama): MediaPipe langsung ke frame UTUH, TANPA YOLO --
      ini perilaku ASLI extract_landmarks.py, dipertahankan sbg opsi
      perbandingan, TAPI SUDAH BUKAN produksi (lihat build_dataset.py).
    - True (PRODUKSI BARU): YOLO deteksi orang dulu -> crop (expand_bbox,
      padding+aspect-ratio target SAMA PERSIS dgn live_pipeline.py) -> baru
      MediaPipe jalan di HASIL CROP -- supaya proses training & live 100%
      identik. DITEMUKAN LANGSUNG (bukan dugaan): tanpa ini, neck_angle bisa
      geser sampai 55 derajat dibanding cara training lama, di frame yg
      FISIKNYA PERSIS SAMA -- murni akibat crop, bukan gerakan beda. MediaPipe
      TETAP dipanggil TIAP frame (bbox ada atau tidak -- JANGAN di-skip, lihat
      alasan tracker internal MediaPipe di live_pipeline.py); hasilnya
      dipaksa NaN kalau YOLO tidak nemu orang di frame itu -- PERSIS gerbang
      yg sudah diperbaiki di live_pipeline.py (konsistensi total, termasuk
      utk kasus tepi/edge case)."""
    mp_pose = mp.solutions.pose
    cap = open_video_capture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"gagal membuka video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows = []

    detector = None
    expand_bbox_fn = None
    training_aspect_ratio = None
    if use_yolo_crop:
        from src.detection.yolo_detector import PersonDetector, expand_bbox, TRAINING_VIDEO_ASPECT_RATIO
        detector = PersonDetector(confidence_threshold=yolo_confidence)
        expand_bbox_fn = expand_bbox
        training_aspect_ratio = TRAINING_VIDEO_ASPECT_RATIO

    with mp_pose.Pose(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            bbox = None
            pose_input = frame
            if use_yolo_crop:
                bbox = detector.detect(frame)
                if bbox:
                    h, w = frame.shape[:2]
                    crop_box = expand_bbox_fn(bbox, crop_padding, w, h,
                                                target_aspect_ratio=training_aspect_ratio)
                    pose_input = frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]].copy()

            image = cv2.cvtColor(pose_input, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)  # SELALU dipanggil (bbox ada ATAU tidak) --
            # PERSIS live_pipeline.py, JANGAN di-skip (tracker internal MediaPipe
            # butuh dipanggil terus-menerus supaya tidak "reset").

            row = {
                "frame_idx": frame_idx,
                "timestamp_sec": frame_idx / fps,
            }
            # use_yolo_crop=True DAN YOLO tidak nemu orang -> paksa NaN (PERSIS
            # gerbang yg diperbaiki di live_pipeline.py). use_yolo_crop=False
            # (mode lama) -> perilaku asli, cuma cek pose_landmarks spt biasa.
            if (not use_yolo_crop or bbox) and results.pose_landmarks is not None:
                for name, lm in zip(POSE_LANDMARK_NAMES, results.pose_landmarks.landmark):
                    row[f"{name}_x"] = lm.x
                    row[f"{name}_y"] = lm.y
                    row[f"{name}_z"] = lm.z
                    row[f"{name}_v"] = lm.visibility
            else:
                for col in LANDMARK_COLUMNS:
                    row[col] = float("nan")

            rows.append(row)
            frame_idx += 1

    cap.release()
    return pd.DataFrame(rows), fps


def _relative_output_subpath(video_path, exercise):
    """Subpath CSV output MENIRU lokasi video mentahnya relatif ke
    raw_videos/{exercise}/ -- video flat (langsung di raw_videos/{exercise}/)
    tetap flat, video yg di-nested per-partisipan/per-kelas (mis.
    raw_videos/deadlift/p2/armsspread/xxx.mp4) hasil CSV-nya ikut nested sama
    (extracted_landmarks/deadlift/p2/armsspread/xxx.csv) -- supaya struktur
    folder CSV konsisten dgn struktur folder video (lihat diskusi proyek soal
    rapikan data per-partisipan/per-kelas). MURNI lokasi file, tidak mengubah
    data/fitur apa pun. Kalau video_path ternyata di luar raw_videos/{exercise}/
    (mis. dipanggil dgn path custom), fallback ke nama file saja (flat) --
    tidak error."""
    exercise_root = RAW_VIDEOS_DIR / exercise
    try:
        rel = video_path.resolve().relative_to(exercise_root.resolve())
    except ValueError:
        rel = Path(video_path.name)
    return rel.with_suffix(".csv")


def process_video(video_path, out_dir, min_detection_confidence, min_tracking_confidence, force=False,
                   use_yolo_crop=False, yolo_confidence=0.7, crop_padding=0.75):
    try:
        meta = parse_video_filename(video_path)
    except FilenameParseError as e:
        print(f"[skip] {e}")
        return None

    out_path = Path(out_dir) / meta["exercise"] / _relative_output_subpath(video_path, meta["exercise"])
    if out_path.exists() and not force:
        existing = pd.read_csv(out_path, nrows=1)
        if "phase" in existing.columns:
            print(f"[skip] {out_path.name} sudah pernah dilabeli (ada kolom 'phase') -- "
                  f"tidak ditimpa. Pakai --force kalau memang mau ekstrak ulang (label lama akan HILANG). "
                  f"Utk video yg SUDAH dilabeli dan mau migrasi ke use_yolo_crop=True TANPA kehilangan "
                  f"label, pakai scripts/migrate_yolo_crop.py (menggabung ulang, bukan menimpa polos).")
            return None

    print(f"[extract] {video_path.name} (exercise={meta['exercise']}, "
          f"posture_class={meta['posture_class']}, camera_angle={meta['camera_angle']}, "
          f"take={meta['take']})")

    df, fps = extract_one_video(video_path, min_detection_confidence, min_tracking_confidence,
                                  use_yolo_crop=use_yolo_crop, yolo_confidence=yolo_confidence,
                                  crop_padding=crop_padding)
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
    parser.add_argument("inputs", nargs="+", help="video file(s) atau folder berisi video")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--force", action="store_true",
                         help="ekstrak ulang walau CSV tujuan sudah pernah dilabeli (label lama akan HILANG)")
    parser.add_argument("--use-yolo-crop", action="store_true",
                         help="PRODUKSI BARU (lihat diskusi proyek) -- YOLO deteksi+crop dulu sebelum "
                              "MediaPipe, SAMA PERSIS proses live_pipeline.py. Default OFF (perilaku lama, "
                              "frame utuh) HANYA dipertahankan utk perbandingan, BUKAN produksi lagi. Utk "
                              "video yg SUDAH dilabeli, JANGAN pakai flag ini langsung (label lama hilang) "
                              "-- pakai scripts/migrate_yolo_crop.py.")
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
            force=args.force, use_yolo_crop=args.use_yolo_crop,
            yolo_confidence=args.yolo_confidence, crop_padding=args.crop_padding,
        )
        if out_path is not None:
            results.append(out_path)

    print(f"\nSelesai: {len(results)}/{len(videos)} video berhasil diekstrak.")


if __name__ == "__main__":
    main()
