"""Preview cepat YOLOv11 (read-only) -- putar video, tampilkan bounding box
deteksi orang + hasil crop (area di luar kotak dibuang), biar bisa
dicek visual apakah deteksinya akurat sebelum dirangkai ke pipeline live.

Usage:
    python src/detection/preview_detection.py data/raw_videos/benchpress/benchpress_correct_p1_front_take01.mp4
    python src/detection/preview_detection.py <video> --confidence 0.5

Kontrol:
    space        play / pause
    , / .        mundur / maju 1 frame (saat pause)
    m            toggle tampilan: bounding box biasa <-> hasil crop
    q / ESC      keluar
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.detection.yolo_detector import PersonDetector, crop_bbox
from src.io_utils import open_video_capture

BOX_COLOR = (60, 200, 60)      # kotak YOLO asli (ketat)
PAD_COLOR = (40, 180, 255)     # kotak setelah padding (oranye) -- area yg BENERAN dikasih ke MediaPipe


def padded_bbox(bbox, padding_ratio, w, h):
    """Hitung kotak SETELAH padding (buat digambar), logika sama persis
    dengan crop_bbox() di yolo_detector.py -- biar visualnya konsisten."""
    x1, y1, x2, y2 = bbox
    if padding_ratio > 0:
        pad_x = int((x2 - x1) * padding_ratio)
        pad_y = int((y2 - y1) * padding_ratio)
        x1, y1, x2, y2 = x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video_path")
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--padding", type=float, default=0.0,
                         help="padding_ratio yg dipakai live_pipeline.py -- digambar sbg kotak "
                              "oranye TAMBAHAN di luar kotak YOLO hijau, biar keliatan seberapa "
                              "besar area yg BENERAN diproses MediaPipe (mode crop 'm' jg ikut pakai ini).")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"[error] video tidak ditemukan: {video_path}")
        return

    detector = PersonDetector(confidence_threshold=args.confidence)
    cap = open_video_capture(str(video_path))
    if not cap.isOpened():
        print(f"[error] gagal membuka {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0
    playing = True
    cap_pos = 0
    show_masked = False
    speed = args.speed
    win_name = f"preview YOLOv11: {video_path.name}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    n_detected = 0
    n_checked = 0

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

        bbox = detector.detect(frame)
        n_checked += 1
        if bbox:
            n_detected += 1

        display = frame.copy()
        h, w = frame.shape[:2]
        if bbox:
            if show_masked:
                display = crop_bbox(display, bbox, padding_ratio=args.padding)  # ukuran beda dari frame asli
            else:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), BOX_COLOR, 3)
                if args.padding > 0:
                    px1, py1, px2, py2 = padded_bbox(bbox, args.padding, w, h)
                    cv2.rectangle(display, (px1, py1), (px2, py2), PAD_COLOR, 2)
        status = f"person {'terdeteksi' if bbox else 'TIDAK terdeteksi'}"

        lines = [
            f"frame {idx}/{n_frames - 1}   {'PLAY' if playing else 'PAUSE'} {speed:.2f}x   "
            f"mode={'crop' if show_masked else 'bbox'} (m utk ganti)   padding={args.padding:.2f}",
            status,
            f"terdeteksi {n_detected}/{n_checked} frame ({n_detected/n_checked*100:.1f}%) sejauh ini",
        ]
        for i, line in enumerate(lines):
            cv2.putText(display, line, (10, 32 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, BOX_COLOR, 2, cv2.LINE_AA)

        cv2.imshow(win_name, display)
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
        elif key == ord("m"):
            show_masked = not show_masked

    cap.release()
    cv2.destroyWindow(win_name)
    print(f"[info] ringkasan: {n_detected}/{n_checked} frame ({n_detected/n_checked*100:.1f}%) berhasil deteksi orang")


if __name__ == "__main__":
    main()
