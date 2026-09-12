"""Tahap 1b: label fase konsentrik/eksentrik per frame.

Auto-detect jalan dulu sebagai draft (deteksi titik balik pada sudut sendi
utama per exercise: squat -> knee_angle, benchpress -> elbow_angle,
deadlift -> hip_angle, via scipy.find_peaks dgn prominence+distance minimum,
per segmen antar titik balik -- bukan diff per-frame, supaya tidak
bolak-balik karena noise kecil). Lalu kamu review/koreksi manual lewat
playback + keypress (ala Ko et al.) -- override manual selalu menang atas
hasil auto.

Resume: CSV yang sudah pernah dilabeli, frame manual-nya otomatis dimuat
balik jadi override awal -- bebas quit & lanjut sesi lain waktu.

Quit ('q'/ESC): frame yang belum kamu putuskan manual (u/d/e) otomatis
ditandai EXCLUDED, supaya cuma frame yang benar-benar dilabeli manual yang
jadi kelas nyata -- video langsung lolos syarat "0 sisa auto", dan bisa
di-resume kapan saja utk melabel ulang bagian itu.

Auto-trim: penekanan SPACE PERTAMA di tiap sesi menandai semua frame
sebelumnya sebagai EXCLUDED (anggap masa persiapan) -- beda dari Ko et al.
yang cuma menyimpan frame saat tombol ditekan (kita simpan semua frame
kontinu, krn sliding window butuh itu). Matikan dgn --no-auto-trim-start.

`posture_class` sudah pasti dari nama file video (1 video = 1 postur), tool
ini cuma menentukan fase, kelas final 18-way = f"{exercise}_{posture_class}_{phase}".

Usage:
    python src/extraction/label_phase.py data/extracted_landmarks/squat/squat_correct_p1_left45_take01.csv
    python src/extraction/label_phase.py <csv> --start-sec 12 --end-sec 42
    python src/extraction/label_phase.py <csv> --auto-only   # skip playback, simpan draft auto saja

Kontrol saat playback:
    space        play / pause (PERTAMA KALI = tandai mulai gerakan, buang bagian sebelumnya)
    u            tandai dari frame ini = UP / konsentrik   (berlaku terus)
    d            tandai dari frame ini = DOWN / eksentrik  (berlaku terus)
    e            tandai dari frame ini = EXCLUDED (dibuang, tidak jadi data)
    r            kembali ikut hasil auto-detect
    , / .        mundur / maju 1 frame (saat pause)
    [ / ]        pelan / cepat-kan playback
    s            simpan (tanpa keluar)
    q / ESC      simpan & keluar (frame yang belum diputuskan manual -> otomatis EXCLUDED)
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.joint_angles import (
    PRIMARY_ANGLE_BASE_BY_EXERCISE,
    VISIBILITY_THRESHOLD,
    primary_angle_for_frame,
)
from src.features.skeleton_draw import draw_skeleton
from src.io_utils import atomic_write_csv, open_video_capture

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_VIDEOS_DIR = REPO_ROOT / "data" / "raw_videos"

CONCENTRIC = "concentric"
ECCENTRIC = "eccentric"
EXCLUDED = "excluded"

# Tuned against squat_correct_p1_*_take01: prominence/distance rendah (mis. 10deg)
# menangkap wobble kecil (warm-up, noise) sebagai "rep palsu"; nilai ini menyaring
# itu sambil tetap menangkap repetisi asli (ayunan sudut ~80-130 derajat).
DEFAULT_MIN_PROMINENCE_DEG = 40.0
DEFAULT_MIN_DISTANCE_SEC = 0.8

PHASE_COLORS_BGR = {
    CONCENTRIC: (60, 200, 60),
    ECCENTRIC: (40, 40, 230),
    EXCLUDED: (140, 140, 140),
}


def find_turning_points(smoothed, fps, min_prominence_deg=DEFAULT_MIN_PROMINENCE_DEG,
                         min_distance_sec=DEFAULT_MIN_DISTANCE_SEC):
    """Titik balik (peak=puncak, valley=lembah) pada smoothed angle.

    prominence+distance minimum membuat titik balik ini tahan terhadap noise
    kecil di sekitar titik balik asli (beda dengan diff per-frame yang bisa
    kebalik-balik hanya karena fluktuasi kecil mendekati turunan nol).
    """
    values = smoothed.to_numpy()
    distance = max(1, int(min_distance_sec * fps))
    peak_idx, _ = find_peaks(values, prominence=min_prominence_deg, distance=distance)
    valley_idx, _ = find_peaks(-values, prominence=min_prominence_deg, distance=distance)
    points = sorted(
        [(int(i), "peak") for i in peak_idx] + [(int(i), "valley") for i in valley_idx]
    )
    return points


def compute_auto_phase(df, exercise, smooth_window=5, fps=30.0,
                        min_prominence_deg=DEFAULT_MIN_PROMINENCE_DEG,
                        min_distance_sec=DEFAULT_MIN_DISTANCE_SEC):
    """Auto-detect phase per SEGMEN antar titik balik (bukan diff per-frame).

    Naik menuju/menjauhi lembah = konsentrik, turun menuju/menjauhi puncak =
    eksentrik. Karena batas segmen adalah titik balik yang sudah disaring
    prominence+distance, fase tidak lagi bolak-balik akibat noise kecil persis
    di sekitar titik balik (deepest point squat, top position, dst).

    Returns (primary_angle series, phase_auto array, smoothed angle series).
    """
    primary = df.apply(lambda row: primary_angle_for_frame(row, exercise), axis=1)
    smoothed = primary.rolling(smooth_window, center=True, min_periods=1).mean()
    smoothed = smoothed.interpolate(limit_direction="both")

    n = len(smoothed)
    turning_points = find_turning_points(smoothed, fps, min_prominence_deg, min_distance_sec)
    phase = np.empty(n, dtype=object)

    if not turning_points:
        # tidak ada titik balik yang cukup jelas (mis. video terlalu pendek/flat):
        # fallback ke arah tren keseluruhan.
        phase[:] = CONCENTRIC if smoothed.iloc[-1] >= smoothed.iloc[0] else ECCENTRIC
        return primary, phase, smoothed

    # segmen sebelum titik balik pertama: arahnya "menuju" titik itu
    first_idx, first_kind = turning_points[0]
    phase[: first_idx + 1] = CONCENTRIC if first_kind == "peak" else ECCENTRIC

    # segmen antar titik balik: arahnya "menjauhi" titik balik di awal segmen
    for (idx_a, kind_a), (idx_b, _kind_b) in zip(turning_points, turning_points[1:]):
        phase[idx_a: idx_b + 1] = CONCENTRIC if kind_a == "valley" else ECCENTRIC

    # segmen setelah titik balik terakhir: arahnya "menjauhi" titik itu
    last_idx, last_kind = turning_points[-1]
    phase[last_idx:] = CONCENTRIC if last_kind == "valley" else ECCENTRIC

    return primary, phase, smoothed


def estimate_rep_count(smoothed, fps, min_prominence_deg=DEFAULT_MIN_PROMINENCE_DEG,
                        min_distance_sec=DEFAULT_MIN_DISTANCE_SEC):
    """Estimasi jumlah repetisi (QA saja) dari jumlah puncak -- titik balik yang
    sama dipakai compute_auto_phase(), supaya konsisten dgn segmentasi fase."""
    turning_points = find_turning_points(smoothed, fps, min_prominence_deg, min_distance_sec)
    return sum(1 for _, kind in turning_points if kind == "peak")


def find_source_video(exercise, source_video):
    """Cari video mentah -- coba lokasi flat dulu, baru fallback ke
    pencarian rekursif by nama file (video boleh nested per-partisipan/
    per-kelas, mis. raw_videos/deadlift/p2/armsspread/...)."""
    candidate = RAW_VIDEOS_DIR / exercise / source_video
    if candidate.exists():
        return candidate
    matches = list((RAW_VIDEOS_DIR / exercise).rglob(source_video))
    return matches[0] if matches else None


def apply_time_trim(override, fps, n_frames, start_sec, end_sec):
    """Tandai frame sebelum start_sec / sesudah end_sec sebagai EXCLUDED."""
    n_start = n_end = 0
    if start_sec is not None:
        cutoff = min(n_frames, int(round(start_sec * fps)))
        override[:cutoff] = EXCLUDED
        n_start = cutoff
    if end_sec is not None:
        cutoff = max(0, int(round(end_sec * fps)))
        if cutoff < n_frames:
            override[cutoff:] = EXCLUDED
            n_end = n_frames - cutoff
    if n_start or n_end:
        print(f"[trim] {n_start} frame di awal + {n_end} frame di akhir ditandai EXCLUDED "
              f"(start_sec={start_sec}, end_sec={end_sec})")
    return override


def _phase_summary(phase, override):
    counts = pd.Series(phase).value_counts()
    n_manual = sum(1 for o in override if o is not None)
    parts = [f"{k}={counts.get(k, 0)}" for k in (CONCENTRIC, ECCENTRIC, EXCLUDED)]
    return f"{' '.join(parts)} | manual/override={n_manual}/{len(phase)}"


def run_interactive_review(video_path, df, phase_auto, primary, meta, fps, override, speed=1.0,
                            auto_trim_start=True):
    cap = open_video_capture(str(video_path))
    if not cap.isOpened():
        print(f"[warn] gagal membuka {video_path}, lewati review interaktif.")
        return phase_auto, override

    n_frames = len(df)
    active_override = None
    idx = 0
    playing = False
    cap_pos = 0
    # True kalau session ini sudah pernah ditekan space (play) sekali -- dipakai
    # buat auto-trim start SEKALI SAJA di penekanan space pertama, bukan tiap toggle.
    session_started = False

    def current_phase(i):
        return override[i] if override[i] is not None else phase_auto[i]

    def fmt(i):
        angle = primary.iloc[i]
        angle_txt = f"{angle:6.1f}deg" if pd.notna(angle) else "   N/A  "
        src = "MANUAL" if override[i] is not None else "auto  "
        return (f"frame {i:5d}/{n_frames - 1}  t={i / fps:6.2f}s  "
                f"angle={angle_txt}  phase={current_phase(i):<10} [{src}]")

    print("\n=== REVIEW INTERAKTIF ===")
    print("space=play/pause  u=UP/konsentrik  d=DOWN/eksentrik  e=EXCLUDE  r=ikut auto")
    print(",=mundur  .=maju  [ ]=kecepatan  s=simpan  q=simpan&keluar")
    if auto_trim_start:
        print("CATATAN: penekanan SPACE PERTAMA KALI akan menandai semua frame SEBELUM titik itu\n"
              "sebagai EXCLUDED (anggap masa persiapan) -- geser dulu ke titik gerakan mulai (','/'.')\n"
              "sebelum menekan space kalau video ini ada bagian ancang-ancang di awal. Ini TIDAK\n"
              "berlaku ke frame yang sudah dilabel manual dari sesi sebelumnya (aman, resume tetap jalan).\n"
              "Quit ('q') akan menandai EXCLUDED semua frame yang SAMPAI SAAT ITU belum kamu putuskan\n"
              "manual (u/d/e) -- jadi cuma yang benar-benar kamu labeli yang jadi kelas nyata. Bisa\n"
              "direview ulang & ditimpa kapan saja lewat resume.")
    print(f"[PAUSE] {fmt(idx)}")

    win_name = f"label_phase: {meta['source_video']}"
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

        if active_override is not None:
            override[idx] = active_override

        label = current_phase(idx)
        color = PHASE_COLORS_BGR.get(label, (255, 255, 255))
        angle = primary.iloc[idx]
        angle_txt = f"{angle:.1f} deg" if pd.notna(angle) else "N/A"
        src = "MANUAL" if override[idx] is not None else "auto"

        n_points = draw_skeleton(frame, df.iloc[idx], color=color)

        lines = [
            f"frame {idx}/{n_frames - 1}   t={idx / fps:.2f}s   {'PLAY' if playing else 'PAUSE'} {speed:.2f}x",
            f"phase={label} [{src}]   angle={angle_txt}   landmark badan={n_points}/17",
            f"{meta['exercise']}_{meta['posture_class']}   u=up d=down e=excl r=auto q=save&quit",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 32 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, color, 2, cv2.LINE_AA)

        cv2.imshow(win_name, frame)

        if playing:
            print(f"\r[PLAY ] {fmt(idx)}   ", end="", flush=True)
            wait_ms = max(1, int(1000.0 / (fps * speed)))
        else:
            wait_ms = 0
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == 255:  # tidak ada tombol ditekan (timeout saat playing)
            if playing:
                if idx >= n_frames - 1:
                    playing = False
                    print(f"\n[END  ] {fmt(idx)}")
                else:
                    idx += 1
            continue

        if key in (ord("q"), 27):
            n_defaulted = 0
            for i in range(n_frames):
                if override[i] is None:
                    override[i] = EXCLUDED
                    n_defaulted += 1
            if n_defaulted:
                print(f"\n[AUTO-EXCLUDE] {n_defaulted}/{n_frames} frame belum sempat kamu putuskan "
                      f"manual (u/d/e) -- otomatis ditandai EXCLUDED. Video ini langsung lolos syarat "
                      f"0-sisa-'auto' (build_dataset.py akan pakai video ini). Kalau nanti mau melabel "
                      f"ulang bagian ini, resume & override manual seperti biasa.")
            print(f"\n[QUIT ] {fmt(idx)}")
            break
        elif key == ord(" "):
            playing = not playing
            if playing and not session_started:
                session_started = True
                if auto_trim_start and idx > 0:
                    n_trimmed = 0
                    for i in range(idx):
                        if override[i] is None:
                            override[i] = EXCLUDED
                            n_trimmed += 1
                    if n_trimmed:
                        print(f"\n[AUTO-TRIM] frame 0-{idx - 1} ({n_trimmed} frame) ditandai EXCLUDED "
                              f"(sebelum titik play pertama)")
            print(f"\n[{'PLAY ' if playing else 'PAUSE'}] {fmt(idx)}")
        elif key == ord("u"):
            active_override = CONCENTRIC
            override[idx] = CONCENTRIC
            print(f"\n[OVERRIDE] -> UP/konsentrik mulai {fmt(idx)}")
        elif key == ord("d"):
            active_override = ECCENTRIC
            override[idx] = ECCENTRIC
            print(f"\n[OVERRIDE] -> DOWN/eksentrik mulai {fmt(idx)}")
        elif key == ord("e"):
            active_override = EXCLUDED
            override[idx] = EXCLUDED
            print(f"\n[OVERRIDE] -> EXCLUDED mulai {fmt(idx)}")
        elif key == ord("r"):
            active_override = None
            override[idx] = None
            print(f"\n[AUTO ] kembali ikut auto-detect mulai {fmt(idx)}")
        elif key == ord(","):
            playing = False
            idx = max(0, idx - 1)
            print(f"\n[STEP<] {fmt(idx)}")
        elif key == ord("."):
            playing = False
            idx = min(n_frames - 1, idx + 1)
            print(f"\n[STEP>] {fmt(idx)}")
        elif key == ord("["):
            speed = max(0.1, round(speed - 0.25, 2))
            print(f"\n[SPEED] {speed:.2f}x")
        elif key == ord("]"):
            speed = min(4.0, round(speed + 0.25, 2))
            print(f"\n[SPEED] {speed:.2f}x")
        elif key == ord("s"):
            merged = _merge(phase_auto, override)
            _save_labeled_csv(df, merged, override, meta)
            print(f"[SAVED] {_phase_summary(merged, override)}")

    cap.release()
    cv2.destroyWindow(win_name)
    return phase_auto, override


def _merge(phase_auto, override):
    return np.array([o if o is not None else a for a, o in zip(phase_auto, override)], dtype=object)


def _save_labeled_csv(df, phase, override, meta):
    out = df.copy()
    out["phase"] = phase
    out["label_source"] = ["manual" if o is not None else "auto" for o in override]
    base = out["exercise"] + "_" + out["posture_class"] + "_" + out["phase"]
    out["class"] = np.where(phase == EXCLUDED, EXCLUDED, base)
    atomic_write_csv(out, meta["csv_path"])
    print(f"  -> disimpan: {meta['csv_path']}")


def add_common_args(parser):
    """Opsi bersama label_phase.py & label_batch.py (tanpa positional csv_path/inputs)."""
    parser.add_argument("--auto-only", action="store_true",
                         help="skip playback interaktif, simpan draft auto saja")
    parser.add_argument("--start-sec", type=float, default=None,
                         help="detik mulai gerakan; frame sebelum ini ditandai EXCLUDED (bagian ancang-ancang)")
    parser.add_argument("--end-sec", type=float, default=None,
                         help="detik selesai gerakan; frame sesudah ini ditandai EXCLUDED")
    parser.add_argument("--speed", type=float, default=1.0,
                         help="kecepatan playback awal (bisa diubah live pakai [ dan ])")
    parser.add_argument("--no-auto-trim-start", action="store_true",
                         help="matikan auto-EXCLUDE frame sebelum penekanan space pertama "
                              "(default: aktif -- space pertama = tanda mulai gerakan asli)")
    parser.add_argument("--smooth-window", type=int, default=5,
                         help="jumlah frame untuk rolling-mean smoothing sudut utama")
    parser.add_argument("--min-peak-prominence", type=float, default=DEFAULT_MIN_PROMINENCE_DEG,
                         help="derajat minimum prominence titik balik (dipakai utk segmentasi fase & estimasi rep)")
    parser.add_argument("--min-peak-distance-sec", type=float, default=DEFAULT_MIN_DISTANCE_SEC,
                         help="jarak minimum antar titik balik (detik), cegah 2 titik balik terlalu rapat")
    return parser


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="CSV hasil extract_landmarks.py")
    add_common_args(parser)
    return parser


def label_one_csv(csv_path, args):
    """Jalankan alur review 1 CSV (dipakai oleh main() dan label_batch.py).

    Returns True kalau berhasil diproses, False kalau di-skip/error.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if df.empty:
        print("[error] CSV kosong.")
        return False

    exercise = df["exercise"].iloc[0]
    if exercise not in PRIMARY_ANGLE_BASE_BY_EXERCISE:
        print(f"[error] exercise '{exercise}' tidak dikenal.")
        return False

    meta = {
        "exercise": exercise,
        "posture_class": df["posture_class"].iloc[0],
        "source_video": df["source_video"].iloc[0],
        "csv_path": csv_path,
    }
    fps = float(df["fps"].iloc[0]) if "fps" in df.columns else 30.0

    primary, phase_auto, smoothed = compute_auto_phase(
        df, exercise, args.smooth_window, fps,
        min_prominence_deg=args.min_peak_prominence,
        min_distance_sec=args.min_peak_distance_sec,
    )
    rep_estimate = estimate_rep_count(
        smoothed, fps,
        min_prominence_deg=args.min_peak_prominence,
        min_distance_sec=args.min_peak_distance_sec,
    )
    n_missing = primary.isna().sum()

    print(f"[file] {csv_path.name}  ({len(df)} frame @ {fps:.2f} fps = {len(df) / fps:.1f} detik)")
    print(f"[auto] exercise={exercise}  posture_class={meta['posture_class']}  "
          f"primary_angle={PRIMARY_ANGLE_BASE_BY_EXERCISE[exercise]}  "
          f"visibility_threshold={VISIBILITY_THRESHOLD}")
    print(f"[auto] estimasi repetisi (draft, QA saja): {rep_estimate}")
    if n_missing:
        print(f"[warn] {n_missing}/{len(df)} frame primary_angle NaN (occluded/tidak terdeteksi)")

    override = np.array([None] * len(df), dtype=object)

    # Resume: muat balik frame yang sudah ditandai manual dari sesi
    # sebelumnya sebagai override awal.
    if "phase" in df.columns and "label_source" in df.columns:
        prev_manual_mask = (df["label_source"] == "manual").to_numpy()
        n_resumed = int(prev_manual_mask.sum())
        if n_resumed:
            prev_phase = df["phase"].to_numpy()
            for i in np.flatnonzero(prev_manual_mask):
                override[i] = prev_phase[i]
            print(f"[resume] {n_resumed}/{len(df)} frame manual dari sesi sebelumnya dimuat balik "
                  f"sebagai override awal (tidak akan hilang kalau kamu tidak sentuh ulang).")

    override = apply_time_trim(override, fps, len(df), args.start_sec, args.end_sec)

    if not args.auto_only:
        video_path = find_source_video(exercise, meta["source_video"])
        if video_path is None:
            print(f"[warn] video sumber tidak ditemukan di data/raw_videos/{exercise}/"
                  f"{meta['source_video']} -- lewati review interaktif, simpan draft auto.")
        else:
            phase_auto, override = run_interactive_review(
                video_path, df, phase_auto, primary, meta, fps, override, args.speed,
                auto_trim_start=not args.no_auto_trim_start,
            )

    final_phase = _merge(phase_auto, override)
    _save_labeled_csv(df, final_phase, override, meta)
    print(f"[hasil] {_phase_summary(final_phase, override)}")
    return True


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    label_one_csv(args.csv_path, args)


if __name__ == "__main__":
    main()
