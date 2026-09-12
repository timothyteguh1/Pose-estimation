"""Preview visual pipeline live (bab 8.3.3) -- YOLO+crop(+padding) -> MediaPipe
-> window -> prediksi -> rep counting, SEMUA jalan bareng, ditampilkan live.

Tampilan: frame UTUH ukuran video asli (TIDAK berubah-ubah ukuran), dengan
kotak YOLO (hijau), kotak setelah padding (oranye, = area yg diproses
MediaPipe), skeleton, dan teks prediksi/rep count digambar di atasnya --
gaya sama seperti demo Ko et al.

Bisa pakai video file (buat tes tanpa kamera) ATAU webcam asli.

Usage:
    python src/app/preview_live_pipeline.py benchpress --video data/raw_videos/benchpress/benchpress_correct_p1_front_take01.mp4
    python src/app/preview_live_pipeline.py benchpress --camera 0

Kontrol:
    space   pause/resume (mode video) -- webcam selalu live, tidak bisa pause
    q/ESC   keluar
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.app.live_pipeline import LivePosturePipeline
from src.features.skeleton_draw import draw_skeleton
from src.io_utils import open_video_capture

PRED_COLOR = (230, 160, 40)
NO_PRED_COLOR = (140, 140, 140)
BOX_COLOR = (60, 200, 60)      # kotak YOLO asli (ketat)
PAD_COLOR = (40, 180, 255)     # kotak setelah padding -- area yg BENERAN diproses MediaPipe


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exercise", choices=["squat", "benchpress", "deadlift"])
    parser.add_argument("--video", default=None, help="path video (kalau tidak pakai webcam)")
    parser.add_argument("--camera", type=int, default=None, help="index webcam (mis. 0)")
    parser.add_argument("--yolo-confidence", type=float, default=0.7)
    parser.add_argument("--padding", type=float, default=None,
                         help="override crop_padding LivePosturePipeline (default: nilai bawaan kelasnya)")
    args = parser.parse_args()

    if not args.video and args.camera is None:
        print("[error] kasih salah satu: --video <path> atau --camera <index>")
        return

    print(f"[info] load model+YOLO utk {args.exercise} ... (bisa agak lama pas pertama kali)")
    pipeline_kwargs = {"yolo_confidence": args.yolo_confidence}
    if args.padding is not None:
        pipeline_kwargs["crop_padding"] = args.padding
    pipeline = LivePosturePipeline(args.exercise, **pipeline_kwargs)
    print(f"[info] crop_padding aktif: {pipeline.crop_padding}")

    source = args.video if args.video else args.camera
    cap = open_video_capture(source)
    if not cap.isOpened():
        print(f"[error] gagal membuka sumber: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    is_live_camera = args.video is None
    win_name = f"live pipeline ({args.exercise}): {'webcam' if is_live_camera else Path(args.video).name}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    playing = True
    idx = 0
    start_time = time.time()
    last_pred = None

    while True:
        if not playing and not is_live_camera:
            key = cv2.waitKey(50) & 0xFF
            if key == ord(" "):
                playing = True
            elif key in (ord("q"), 27):
                break
            continue

        ok, frame = cap.read()
        if not ok:
            if is_live_camera:
                print("[warn] gagal baca frame kamera, coba lagi...")
                continue
            print("[info] video selesai.")
            break

        t = idx / fps if not is_live_camera else time.time() - start_time
        result = pipeline.process_frame(frame, t, fps=fps)

        if result["countdown_remaining"] is not None:
            # Masa persiapan -- tampilkan hitung mundur besar, TIDAK gambar
            # skeleton/kotak apa pun (memang belum diproses sama sekali).
            secs_left = int(result["countdown_remaining"]) + 1
            text = f"BERSIAP... {secs_left}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)
            h, w = frame.shape[:2]
            cv2.putText(frame, text, ((w - tw) // 2, (h + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 200, 255), 4, cv2.LINE_AA)
            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            idx += 1
            continue

        if result["predicted_class"] is not None:
            last_pred = result["predicted_class"]

        # Skeleton digambar di display_frame (crop, bukan frame utuh) --
        # koordinat landmark MediaPipe relatif ke gambar yg diproses.
        display = result["display_frame"]
        color = PRED_COLOR if result["bbox"] else NO_PRED_COLOR
        n_points = draw_skeleton(display, result["row"], color=color)

        # Tempel balik ke frame utuh di posisi crop_box asalnya -- window
        # yg ditampilkan tetap berukuran tetap (ukuran video asli).
        canvas = frame.copy()
        if result["crop_box"] is not None:
            x1, y1, x2, y2 = result["crop_box"]
            canvas[y1:y2, x1:x2] = display
            if result["bbox"] is not None:
                bx1, by1, bx2, by2 = result["bbox"]
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), BOX_COLOR, 2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), PAD_COLOR, 2)

        angle_txt = f"{result['primary_angle']:.1f}deg" if result["primary_angle"] is not None else "N/A"
        lines = [
            f"frame {idx}   t={t:.2f}s   {'PLAY' if playing else 'PAUSE'}   landmark badan={n_points}/17",
            f"prediksi (window terbaru): {result['predicted_class'] or '(belum ada window penuh)'}",
            f"prediksi terakhir yg valid: {last_pred or '-'}",
            f"sudut utama: {angle_txt}   jumlah repetisi: {result['rep_count']}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (10, 32 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, color, 2, cv2.LINE_AA)

        cv2.imshow(win_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" ") and not is_live_camera:
            playing = False

        idx += 1

    cap.release()
    cv2.destroyWindow(win_name)
    print(f"[selesai] total frame diproses: {idx}, jumlah repetisi terhitung: {pipeline.rep_counter.rep_count}")


if __name__ == "__main__":
    main()
