"""Sweep parameter rep-counting (prominence/distance/countdown_sec) dari
sinyal yang sudah di-cache (cache_rep_signal.py) -- tidak re-run
YOLO/MediaPipe, cuma cara menghitung repetisi dari sinyal yang sama.

Video dibagi tuning (cari parameter) vs held-out (dikunci) -- supaya MAE
yang dilaporkan tidak "menghafal" video tuning (lihat REP_COUNTING_PARAMS
di rep_counter.py utk histori). Ground truth = jumlah transisi
concentric->eccentric dari label manual (1 rep = 1 puncak terkonfirmasi,
sama seperti OnlineRepCounter).

2 metrik MAE (bab 8.4 proposal): standar (1/n)*sum|pred-GT|, dan ala Hsu et
al. "MAE of the count" (1/n)*sum(|pred-GT|/GT), pembanding saja.

countdown_sec ikut jadi parameter yg di-tuning (bukan cuma
prominence/distance) -- countdown fixed tidak selalu cocok dgn durasi
ancang-ancang asli tiap video.

Usage:
    python src/training/cache_rep_signal.py squat data/raw_videos/squat/p2
    python src/training/tune_rep_counting.py squat \\
        --held-out squat_correct_p2_front_take01 squat_kneeinward_p2_left45_take01 squat_backbend_p2_right45_take01
"""
import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.app.rep_counter import OnlineRepCounter, REP_COUNTING_PARAMS
from src.features.joint_angles import ANGLE_COLUMNS, PRIMARY_ANGLE_BASE_BY_EXERCISE

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "rep_signal_cache"
EXTRACTED_DIR = REPO_ROOT / "data" / "extracted_landmarks"


def gt_rep_count(video_stem, exercise):
    """GT = jumlah transisi concentric->eccentric di label manual (kolom
    'phase'), 'excluded' cuma di awal/akhir video (dibuang dari perhitungan,
    bukan diinterpolasi) -- konsisten definisi "1 rep = 1 puncak"."""
    matches = list((EXTRACTED_DIR / exercise).rglob(f"{video_stem}.csv"))
    if not matches:
        return None
    phase = pd.read_csv(matches[0], usecols=["phase"])["phase"].tolist()
    trimmed = [p for p in phase if p != "excluded"]
    n_peaks = 0
    for i in range(1, len(trimmed)):
        if trimmed[i - 1] == "concentric" and trimmed[i] == "eccentric":
            n_peaks += 1
    return n_peaks


def load_signal(video_stem, exercise):
    path = CACHE_DIR / exercise / f"{video_stem}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def simulate_rep_count(signal_df, exercise, prominence_deg, distance_sec, countdown_sec,
                        min_pattern_similarity=0.70):
    """Replay OnlineRepCounter dari sinyal ter-cache -- PERSIS logika
    LivePosturePipeline.process_frame() (countdown mengecualikan window
    buffer & rep_counter, bukan proses YOLO/MediaPipe -- di sini prosesnya
    sudah selesai di tahap cache, jadi cukup skip update() saja)."""
    base = PRIMARY_ANGLE_BASE_BY_EXERCISE[exercise]
    counter = OnlineRepCounter(exercise, min_prominence_deg=prominence_deg,
                                min_distance_sec=distance_sec,
                                min_pattern_similarity=min_pattern_similarity)
    session_start = None
    for _, row in signal_df.iterrows():
        t = row["t"]
        if session_start is None:
            session_start = t
        if (t - session_start) < countdown_sec:
            continue
        left = row.get(f"left_{base}")
        right = row.get(f"right_{base}")
        vals = [v for v in (left, right) if pd.notna(v)]
        primary = float(np.mean(vals)) if vals else float("nan")
        primary = None if primary != primary else primary
        feature_vector = {c: row[c] for c in ANGLE_COLUMNS} if all(c in row for c in ANGLE_COLUMNS) else None
        counter.update(primary, t, feature_vector=feature_vector)
    return counter.rep_count


def evaluate_config(video_stems, exercise, prominence_deg, distance_sec, countdown_sec):
    errors, errors_norm = [], []
    per_video = []
    for stem in video_stems:
        signal_df = load_signal(stem, exercise)
        gt = gt_rep_count(stem, exercise)
        if signal_df is None or gt is None or gt == 0:
            continue
        pred = simulate_rep_count(signal_df, exercise, prominence_deg, distance_sec, countdown_sec)
        err = abs(pred - gt)
        errors.append(err)
        errors_norm.append(err / gt)
        per_video.append((stem, gt, pred, err))
    if not errors:
        return None
    return {
        "mae": float(np.mean(errors)),
        "mae_hsu": float(np.mean(errors_norm)),
        "per_video": per_video,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exercise", choices=["squat", "benchpress", "deadlift"])
    parser.add_argument("--held-out", nargs="+", required=True,
                         help="nama video (tanpa .mp4/.csv) yang DIKUNCI sbg held-out -- "
                              "sisanya (yang ada cache-nya) otomatis jadi tuning set")
    parser.add_argument("--prominence-grid", type=str, default=None,
                         help="comma-separated derajat, default: sekitar nilai produksi saat ini")
    parser.add_argument("--distance-grid", type=str, default=None,
                         help="comma-separated detik, default: sekitar nilai produksi saat ini")
    parser.add_argument("--countdown-grid", type=str, default="0,1,2,3,4,5",
                         help="comma-separated detik (default 0-5s, 5s = perilaku produksi saat ini)")
    args = parser.parse_args()

    cache_dir = CACHE_DIR / args.exercise
    if not cache_dir.exists():
        print(f"[error] {cache_dir} tidak ada -- jalankan dulu cache_rep_signal.py {args.exercise} ...")
        return
    all_stems = sorted(p.stem for p in cache_dir.glob("*.csv"))
    held_out = list(args.held_out)
    missing = [s for s in held_out if s not in all_stems]
    if missing:
        print(f"[error] held-out berikut belum ada cache-nya: {missing}")
        return
    tuning = [s for s in all_stems if s not in held_out]
    if not tuning:
        print("[error] tuning set kosong -- cache lebih banyak video dulu.")
        return

    print(f"=== {args.exercise} ===")
    print(f"Tuning set ({len(tuning)}): {tuning}")
    print(f"Held-out set ({len(held_out)}): {held_out}")

    current = REP_COUNTING_PARAMS.get(args.exercise, {"min_prominence_deg": 40.0, "min_distance_sec": 0.8})
    prom_default = current["min_prominence_deg"]
    dist_default = current["min_distance_sec"]
    prom_grid = ([float(x) for x in args.prominence_grid.split(",")] if args.prominence_grid
                 else sorted({prom_default - 10, prom_default - 5, prom_default, prom_default + 5, prom_default + 10}))
    dist_grid = ([float(x) for x in args.distance_grid.split(",")] if args.distance_grid
                 else sorted({max(0.2, dist_default - 0.4), dist_default, dist_default + 0.4}))
    countdown_grid = [float(x) for x in args.countdown_grid.split(",")]

    print(f"\n[info] parameter produksi SAAT INI: prominence={prom_default}deg, distance={dist_default}s, countdown=5.0s (default LivePosturePipeline)")
    baseline = evaluate_config(tuning, args.exercise, prom_default, dist_default, 5.0)
    if baseline:
        print(f"  -> MAE tuning (param produksi, countdown=5.0s) = {baseline['mae']:.4f}  "
              f"(MAE of the count ala Hsu et al.={baseline['mae_hsu']*100:.2f}%)")

    print(f"\n[info] sweep {len(prom_grid)} prominence x {len(dist_grid)} distance x {len(countdown_grid)} countdown = "
          f"{len(prom_grid)*len(dist_grid)*len(countdown_grid)} kombinasi, di TUNING SET saja ...")
    results = []
    for prom, dist, cd in product(prom_grid, dist_grid, countdown_grid):
        r = evaluate_config(tuning, args.exercise, prom, dist, cd)
        if r:
            results.append((prom, dist, cd, r["mae"], r["mae_hsu"]))
    results.sort(key=lambda x: x[3])

    print("\nTop 5 kombinasi (urut MAE tuning terkecil):")
    print(f"{'prominence':<12}{'distance':<10}{'countdown':<11}{'MAE':<10}{'MAE of count (Hsu)':<20}")
    for prom, dist, cd, mae, mae_hsu in results[:5]:
        print(f"{prom:<12}{dist:<10}{cd:<11}{mae:<10.4f}{mae_hsu*100:<19.2f}%")

    best_prom, best_dist, best_cd, best_mae, best_mae_hsu = results[0]
    print(f"\n[terpilih] prominence={best_prom}deg, distance={best_dist}s, countdown={best_cd}s "
          f"(MAE tuning={best_mae:.4f})")

    held_result = evaluate_config(held_out, args.exercise, best_prom, best_dist, best_cd)
    print(f"\n=== Validasi HELD-OUT (video ini TIDAK ikut proses pencarian di atas) ===")
    if held_result:
        print(f"MAE held-out = {held_result['mae']:.4f}  "
              f"(MAE of the count ala Hsu et al.={held_result['mae_hsu']*100:.2f}%)")
        print(f"{'video':<45}{'GT':<6}{'pred':<6}{'|err|':<6}")
        for stem, gt, pred, err in held_result["per_video"]:
            print(f"{stem:<45}{gt:<6}{pred:<6}{err:<6}")
    else:
        print("[warn] tidak ada hasil held-out (cek nama video/cache).")

    print(f"\n[info] SIMPAN INI KE LAPORAN: tuning MAE={best_mae:.4f} "
          f"(MAE of the count ala Hsu et al.={best_mae_hsu*100:.2f}%), "
          + (f"held-out MAE={held_result['mae']:.4f} "
             f"(MAE of the count ala Hsu et al.={held_result['mae_hsu']*100:.2f}%) "
             f"dgn prominence={best_prom}deg, distance={best_dist}s, countdown={best_cd}s"
             if held_result else "held-out: tidak ada hasil"))


if __name__ == "__main__":
    main()
