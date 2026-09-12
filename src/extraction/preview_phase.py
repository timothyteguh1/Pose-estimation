"""Preview cepat hasil ekstraksi + deteksi fase, TANPA kontrol override (baca
saja -- beda dari label_phase.py yang bisa ubah label). Untuk sanity-check:
apakah landmark CSV nempel benar ke badan di video asli, dan apakah pola
naik-turun fase auto-detect masuk akal.

Default: PNG plot + LIVE VIEW (window video langsung mainkan skeleton+fase di
atas video mentah asli, tanpa nunggu render ke file dulu). Skeleton digambar
ULANG DARI KOORDINAT CSV (tidak menjalankan ulang MediaPipe -- cepat).

KALAU build_dataset.py SUDAH pernah dijalankan utk video ini, preview ini
OTOMATIS deteksi file `{video}_windowed.csv`-nya (data/windowed_features/)
dan tambahkan 1 lapis info lagi: frame mana yang akhirnya MASUK window
(dipakai training/testing) vs yang TERBUANG walau bukan excluded manual --
yaitu korban filter visibility (Eq. bab 8.3.1, frame usable<0.6) atau run
yang kependekan buat window_sec yang dipakai. Ini jawaban visual buat
pertanyaan "yang kepotong karena visibility itu kelihatannya kayak apa?".
Kalau file windowed belum ada, preview jalan seperti biasa (cuma phase),
tidak ada perubahan perilaku.

Usage:
    python src/extraction/preview_phase.py data/extracted_landmarks/squat/squat_correct_p1_left45_take01.csv
    python src/extraction/preview_phase.py <csv> --no-live          # cuma plot PNG, skip window (lebih cepat)
    python src/extraction/preview_phase.py <csv> --save-video       # juga export ke file mp4
    python src/extraction/preview_phase.py <csv> --no-window-check  # matikan overlay hasil preprocessing
    python src/extraction/preview_phase.py <csv> --only-in-window   # cuma putar bagian yg lolos preprocessing

Kontrol saat live view:
    space        play / pause
    , / .        mundur / maju 1 frame (saat pause)
    [ / ]        pelan / cepat-kan playback
    q / ESC      keluar
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.extraction.label_phase import compute_auto_phase, estimate_rep_count, find_source_video
from src.features.joint_angles import PRIMARY_ANGLE_BASE_BY_EXERCISE
from src.features.skeleton_draw import draw_skeleton
from src.io_utils import open_video_capture

PHASE_COLORS_BGR = {"concentric": (60, 180, 60), "eccentric": (40, 40, 220), "excluded": (140, 140, 140)}
PHASE_COLORS_PLT = {"concentric": "#4caf50", "eccentric": "#e53935", "excluded": "#9e9e9e"}

REPO_ROOT = Path(__file__).resolve().parents[2]
WINDOWED_DIR = REPO_ROOT / "data" / "windowed_features"
# frame yang BUKAN excluded manual tapi tetap tidak pernah masuk window
# manapun -- korban filter visibility (bab 8.3.1) atau run kependekan buat
# window_sec yang dipakai. Ini beda dari abu-abu (excluded) -- oranye
# menandakan "harusnya bisa jadi data, tapi kebuang oleh kualitas deteksi".
DROPPED_COLOR_BGR = (0, 140, 255)


def load_window_coverage(exercise, video_stem, n_frames):
    """Cari {video}_windowed.csv hasil build_dataset.py, kalau ada bangun array
    status per-frame_idx: 'in_window' (masuk window manapun, train/test) atau
    None (tidak pernah masuk window). Returns None kalau file belum ada sama
    sekali (build_dataset.py belum pernah dijalankan utk video ini)."""
    path = WINDOWED_DIR / exercise / f"{video_stem}_windowed.csv"
    if not path.exists():
        return None
    wdf = pd.read_csv(path, usecols=["start_frame_idx", "end_frame_idx"])
    covered = np.zeros(n_frames, dtype=bool)
    for start, end in zip(wdf["start_frame_idx"], wdf["end_frame_idx"]):
        covered[start:end + 1] = True
    return covered


def make_plot(df, exercise, primary, display_phase, smoothed, fps, out_path,
              phase_auto=None, is_saved=False):
    """Plot latar = display_phase (hasil TERSIMPAN kalau ada, else draft auto).
    Kalau CSV sudah pernah dilabeli, titik hitam menandai frame yang draft
    auto SAAT INI beda dari yang tersimpan (mis. karena kamu koreksi manual,
    atau algoritma auto-nya sudah di-tuning ulang sejak terakhir disimpan)."""
    frame_idx = df["frame_idx"].to_numpy()
    fig, ax = plt.subplots(figsize=(14, 4))

    # latar belakang per-segmen fase yang DITAMPILKAN (tersimpan, kalau ada)
    seg_start = 0
    for i in range(1, len(display_phase) + 1):
        if i == len(display_phase) or display_phase[i] != display_phase[seg_start]:
            ax.axvspan(frame_idx[seg_start], frame_idx[i - 1],
                       color=PHASE_COLORS_PLT[display_phase[seg_start]], alpha=0.15, lw=0)
            seg_start = i

    ax.plot(frame_idx, primary, color="#999999", lw=0.8, alpha=0.6, label="primary_angle (mentah)")
    ax.plot(frame_idx, smoothed, color="#1565c0", lw=1.5, label="primary_angle (smoothed)")

    if is_saved and phase_auto is not None:
        mismatch_mask = display_phase != phase_auto
        if mismatch_mask.any():
            ax.scatter(frame_idx[mismatch_mask], smoothed.to_numpy()[mismatch_mask],
                       color="black", s=10, zorder=5,
                       label=f"beda dgn draft auto saat ini ({mismatch_mask.sum()} frame)")

    ax.set_xlabel("frame_idx")
    ax.set_ylabel(f"{PRIMARY_ANGLE_BASE_BY_EXERCISE[exercise]} (derajat)")
    src_desc = "hasil TERSIMPAN (termasuk koreksi manual)" if is_saved else "draft auto-detect"
    ax.set_title(f"{out_path.stem} -- hijau=concentric, merah=eccentric, abu=excluded ({src_desc})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_annotated_video(video_path, df, display_phase, label_source, primary, out_path):
    cap = open_video_capture(str(video_path))
    if not cap.isOpened():
        print(f"[warn] gagal membuka {video_path}, lewati render video.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for i in range(len(df)):
        ok, frame = cap.read()
        if not ok:
            break
        row = df.iloc[i]
        phase = display_phase[i]
        color = PHASE_COLORS_BGR.get(phase, (200, 200, 200))

        draw_skeleton(frame, row, color=color)

        angle_txt = f"{primary.iloc[i]:.1f} deg" if pd.notna(primary.iloc[i]) else "N/A"
        cv2.putText(frame, f"frame {i}  phase={phase} [{label_source[i]}]  angle={angle_txt}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        writer.write(frame)

    cap.release()
    writer.release()
    return out_path


def _snap_to_covered(idx, direction, n_frames, window_coverage):
    """Geser idx searah `direction` (+1/-1) sampai ketemu frame yang masuk
    window (covered) -- dipakai mode --only-in-window buat skip TOTAL frame
    yang kebuang preprocessing, bukan cuma ditandai warna."""
    if window_coverage is None:
        return idx
    while 0 <= idx < n_frames and not window_coverage[idx]:
        idx += direction
    return max(0, min(idx, n_frames - 1))


def run_live_view(video_path, df, display_phase, label_source, primary, meta, fps, speed=1.0,
                   window_coverage=None, only_in_window=False):
    cap = open_video_capture(str(video_path))
    if not cap.isOpened():
        print(f"[warn] gagal membuka {video_path}, lewati live view.")
        return

    n_frames = len(df)
    idx = 0
    if only_in_window and window_coverage is not None:
        idx = _snap_to_covered(idx, +1, n_frames, window_coverage)
    playing = True
    cap_pos = 0
    win_name = f"preview (read-only): {meta['source_video']}"
    if only_in_window:
        win_name += "  [ONLY-IN-WINDOW -- skip frame yang kebuang preprocessing]"
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

        phase = display_phase[idx]
        dropped = (window_coverage is not None and phase != "excluded"
                   and not window_coverage[idx])
        color = DROPPED_COLOR_BGR if dropped else PHASE_COLORS_BGR.get(phase, (200, 200, 200))
        draw_skeleton(frame, df.iloc[idx], color=color)

        angle = primary.iloc[idx]
        angle_txt = f"{angle:.1f} deg" if pd.notna(angle) else "N/A"
        window_status = ""
        if window_coverage is not None:
            if dropped:
                window_status = "  [DIBUANG dari window -- visibility rendah/run kependekan]"
            elif phase != "excluded":
                window_status = "  [masuk window]"
        lines = [
            f"frame {idx}/{n_frames - 1}   t={idx / fps:.2f}s   {'PLAY' if playing else 'PAUSE'} {speed:.2f}x",
            f"phase={phase} [{label_source[idx]}]   angle={angle_txt}{window_status}",
            f"{meta['exercise']}_{meta['posture_class']}   (read-only -- pakai label_phase.py utk edit)",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 32 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, color, 2, cv2.LINE_AA)

        cv2.imshow(win_name, frame)
        wait_ms = max(1, int(1000.0 / (fps * speed))) if playing else 0
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == 255:
            if playing:
                if idx >= n_frames - 1:
                    playing = False
                else:
                    idx += 1
                    if only_in_window:
                        idx = _snap_to_covered(idx, +1, n_frames, window_coverage)
            continue

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            playing = not playing
        elif key == ord(","):
            playing = False
            idx = max(0, idx - 1)
            if only_in_window:
                idx = _snap_to_covered(idx, -1, n_frames, window_coverage)
        elif key == ord("."):
            playing = False
            idx = min(n_frames - 1, idx + 1)
            if only_in_window:
                idx = _snap_to_covered(idx, +1, n_frames, window_coverage)
        elif key == ord("["):
            speed = max(0.1, round(speed - 0.25, 2))
        elif key == ord("]"):
            speed = min(4.0, round(speed + 0.25, 2))

    cap.release()
    cv2.destroyWindow(win_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path")
    parser.add_argument("--no-live", action="store_true",
                         help="skip live view window (default: live view SELALU dibuka)")
    parser.add_argument("--save-video", action="store_true",
                         help="juga export ke file mp4 (default: tidak, live view sudah cukup utk cek cepat)")
    parser.add_argument("--speed", type=float, default=1.0, help="kecepatan live view awal")
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--no-window-check", action="store_true",
                         help="matikan overlay hasil preprocessing (frame yang kebuang "
                              "dari window walau bukan excluded manual)")
    parser.add_argument("--only-in-window", action="store_true",
                         help="SKIP TOTAL frame yang tidak masuk window manapun (excluded "
                              "manual maupun kebuang visibility/run-kependekan) -- video cuma "
                              "muter bagian yang benar-benar dipakai training/testing. "
                              "Butuh build_dataset.py sudah pernah dijalankan utk video ini.")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    df = pd.read_csv(csv_path)
    if df.empty:
        print("[error] CSV kosong.")
        return

    exercise = df["exercise"].iloc[0]
    if exercise not in PRIMARY_ANGLE_BASE_BY_EXERCISE:
        print(f"[error] exercise '{exercise}' tidak dikenal.")
        return

    fps = df["fps"].iloc[0] if "fps" in df.columns else 30.0
    primary, phase_auto, smoothed = compute_auto_phase(df, exercise, args.smooth_window, fps)
    rep_estimate = estimate_rep_count(smoothed, fps)
    phase_reviewed = df["phase"] if "phase" in df.columns else None

    print(f"[info] exercise={exercise} frames={len(df)} "
          f"primary_angle_base={PRIMARY_ANGLE_BASE_BY_EXERCISE[exercise]}")
    print(f"[info] estimasi jumlah repetisi (draft auto, QA saja): {rep_estimate}")

    if phase_reviewed is not None:
        # CSV sudah pernah dilabeli (label_phase.py) -- tampilkan hasil TERSIMPAN
        # itu (termasuk override manual), BUKAN hitung ulang auto dari nol.
        n_diff = (phase_reviewed.to_numpy() != phase_auto).sum()
        print(f"[info] CSV sudah punya kolom 'phase' tersimpan -- dipakai utk tampilan "
              f"(beda dgn draft auto saat ini: {n_diff}/{len(df)} frame)")
        display_phase = phase_reviewed.to_numpy()
        if "label_source" in df.columns:
            label_source = df["label_source"].to_numpy()
        else:
            label_source = np.where(phase_reviewed.to_numpy() == phase_auto, "auto", "manual")
    else:
        print("[info] CSV belum pernah dilabeli -- tampilan pakai draft auto-detect")
        display_phase = phase_auto
        label_source = np.array(["auto"] * len(df))

    out_png = csv_path.with_name(csv_path.stem + "_phase_preview.png")
    make_plot(df, exercise, primary, display_phase, smoothed, fps, out_png,
              phase_auto=phase_auto, is_saved=(phase_reviewed is not None))
    print(f"[ok] plot disimpan: {out_png}")

    meta = {
        "exercise": exercise,
        "posture_class": df["posture_class"].iloc[0],
        "source_video": df["source_video"].iloc[0],
    }

    if args.only_in_window and args.no_window_check:
        print("[error] --only-in-window butuh info window (jangan pakai bareng --no-window-check).")
        return

    window_coverage = None
    if not args.no_window_check or args.only_in_window:
        video_stem = Path(meta["source_video"]).stem
        window_coverage = load_window_coverage(exercise, video_stem, len(df))
        if window_coverage is None:
            msg = (f"[error] --only-in-window butuh {video_stem}_windowed.csv, tapi belum ada "
                   f"-- jalankan build_dataset.py dulu." if args.only_in_window else
                   f"[info] belum ada data hasil preprocessing utk video ini "
                   f"({video_stem}_windowed.csv tidak ditemukan) -- jalankan "
                   f"build_dataset.py dulu kalau mau lihat overlay 'dibuang dari window'.")
            print(msg)
            if args.only_in_window:
                return
        else:
            not_excluded = display_phase != "excluded"
            n_dropped = int((not_excluded & ~window_coverage).sum())
            n_in_window = int(window_coverage.sum())
            print(f"[info] hasil preprocessing: {n_in_window}/{len(df)} frame masuk window "
                  f"({n_in_window/len(df)*100:.1f}%), {n_dropped} frame DIBUANG walau bukan "
                  f"excluded manual (visibility rendah/run kependekan, warna oranye di live view)")

    video_path = None
    if not args.no_live or args.save_video:
        video_path = find_source_video(exercise, meta["source_video"])
        if video_path is None:
            print(f"[warn] video sumber tidak ditemukan untuk {meta['source_video']} "
                  f"di data/raw_videos/{exercise}/, lewati live view/render.")

    if video_path is not None and not args.no_live:
        print(f"[info] live view (read-only, 'q' utk keluar): {video_path}")
        run_live_view(video_path, df, display_phase, label_source, primary, meta, fps, args.speed,
                      window_coverage=window_coverage, only_in_window=args.only_in_window)

    if video_path is not None and args.save_video:
        out_mp4 = csv_path.with_name(csv_path.stem + "_preview.mp4")
        print(f"[info] render skeleton dari CSV di atas video asli: {video_path}")
        result = render_annotated_video(video_path, df, display_phase, label_source, primary, out_mp4)
        if result:
            print(f"[ok] video preview (skeleton di atas video asli) disimpan: {result}")


if __name__ == "__main__":
    main()
