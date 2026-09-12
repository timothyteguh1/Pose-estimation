"""Live-inference cek cepat: apakah model ({exercise}_rf.pkl) bisa jalan ke
video apapun (belum pernah lewat pipeline label kita)?

Lingkup minimal -- bukan aplikasi src/app/ final. Cuma nunjukkin skeleton +
prediksi kelas per window, terus-menerus sepanjang video. Tidak ada
repetition counting/feedback/ringkasan sesi (itu di app.py).

Alur: MediaPipe per frame -> hitung fitur (sama strukturnya dgn training) ->
window kontinu nonstop (beda dari preview_prediction.py yg cuma test set) ->
pipeline.predict() (scaler+RF 1 pkl) -> overlay. Tidak ada ground truth,
jadi warnanya netral.

Usage:
    python src/training/predict_video.py squat "C:\\path\\video.mp4"

Kontrol saat live view:
    space        play / pause
    , / .        mundur / maju 1 frame (saat pause)
    [ / ]        pelan / cepat-kan playback
    q / ESC      keluar
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.extraction.extract_landmarks import extract_one_video
from src.features.sliding_window import (
    build_inference_windows,
    compute_frame_features,
    find_usable_runs,
)
from src.io_utils import open_video_capture
from src.features.skeleton_draw import draw_skeleton

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"

PREDICTED_COLOR = (230, 160, 40)   # netral -- tidak ada ground truth, jadi bukan hijau/merah
NO_WINDOW_COLOR = (140, 140, 140)  # abu -- belum ada window (awal/akhir video / visibility rendah)


def load_artifacts(exercise):
    """{exercise}_rf.pkl = Pipeline(scaler, rf) -- pakai pipeline penuh
    (pipeline.predict() otomatis scaling dulu), beda dari
    preview_prediction.py yg kerja dari data yang sudah ternormalisasi."""
    model_path = MODELS_DIR / f"{exercise}_rf.pkl"
    config_path = MODELS_DIR / f"{exercise}_feature_config.json"
    for p in (model_path, config_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} tidak ada -- jalankan dulu:\n"
                f"  python src/features/build_dataset.py {exercise}\n"
                f"  python src/training/train_model.py {exercise}"
            )
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    with open(config_path) as f:
        config = json.load(f)
    return pipeline, config


def run_live_view(video_path, landmarks_df, frame_map, exercise, fps, speed=1.0):
    cap = open_video_capture(str(video_path))
    if not cap.isOpened():
        print(f"[warn] gagal membuka {video_path}")
        return

    n_frames = len(landmarks_df)
    idx = 0
    playing = True
    cap_pos = 0
    win_name = f"prediksi live ({exercise}): {video_path.name}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

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

        pred = frame_map.get(idx)
        if pred is None:
            color = NO_WINDOW_COLOR
            status = "belum ada window (awal/akhir video atau visibility rendah)"
        else:
            color = PREDICTED_COLOR
            status = f"prediksi model: {pred}"

        draw_skeleton(frame, landmarks_df.iloc[idx], color=color)
        lines = [
            f"frame {idx}/{n_frames - 1}   {'PLAY' if playing else 'PAUSE'} {speed:.2f}x",
            status,
            "(cek model saja -- belum ada rep counting/feedback/ringkasan)",
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exercise", choices=["squat", "benchpress", "deadlift"])
    parser.add_argument("video_path", help="path ke video APAPUN (tidak perlu ikut konvensi nama)")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"[error] video tidak ditemukan: {video_path}")
        return

    pipeline, config = load_artifacts(args.exercise)
    feat_cols = config["feat_cols"]
    window_sec = config["window_sec"]
    stride = config["stride"]
    visibility_threshold = config["visibility_threshold"]

    print(f"[info] ekstraksi MediaPipe dari {video_path.name} ... (bisa agak lama)")
    df, fps = extract_one_video(video_path, args.min_detection_confidence, args.min_tracking_confidence)
    df["fps"] = fps
    print(f"[info] {len(df)} frame @ {fps:.2f} fps")

    feat_df = compute_frame_features(df, visibility_threshold)
    runs = find_usable_runs(feat_df)
    windows = build_inference_windows(df, feat_df, runs, window_sec, stride)
    if windows is None:
        print("[error] tidak ada window yang bisa dibangun (video terlalu pendek / visibility rendah terus).")
        return
    print(f"[info] {len(windows)} window dibangun (window_sec={window_sec}, sama seperti pas training)")

    missing = [c for c in feat_cols if c not in windows.columns]
    if missing:
        print(f"[error] kolom fitur tidak lengkap ({len(missing)} kolom hilang) "
              f"-- config beda versi dgn model, jalankan ulang build_dataset.py & train_model.py.")
        return

    y_pred = pipeline.predict(windows[feat_cols])

    frame_map = {}
    for (_, row), pred in zip(windows.iterrows(), y_pred):
        for f in range(int(row["start_frame_idx"]), int(row["end_frame_idx"]) + 1):
            frame_map[f] = pred

    n_covered = len(frame_map)
    print(f"[info] {n_covered}/{len(df)} frame ({n_covered/len(df)*100:.1f}%) ke-cover window prediksi")

    run_live_view(video_path, df, frame_map, args.exercise, fps, args.speed)


if __name__ == "__main__":
    main()
