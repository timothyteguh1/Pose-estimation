import argparse
import sys
from pathlib import Path

import torch  # noqa: F401

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.extraction.extract_landmarks import LANDMARK_COLUMNS, _relative_output_subpath
from src.extraction.filename_parser import FilenameParseError, parse_video_filename
from src.extraction.label_phase import compute_auto_phase
from src.io_utils import atomic_write_csv

OLD_DIR = REPO_ROOT / "data" / "extracted_landmarks"
NEW_DIR = EXPERIMENT_ROOT / "data" / "extracted_landmarks"
OUT_DIR = EXPERIMENT_ROOT / "data" / "labeled"


def check_mismatches(old_df, new_df, exercise, fps):
    _, auto_phase_new, _ = compute_auto_phase(new_df, exercise, fps=fps)

    old_phase = old_df["phase"].to_numpy()
    labeled_mask = old_phase != "excluded"

    agree = (auto_phase_new[labeled_mask] == old_phase[labeled_mask])
    n_labeled = int(labeled_mask.sum())
    n_agree = int(agree.sum())
    n_disagree = n_labeled - n_agree

    disagree_idx = np.where(labeled_mask)[0][~agree]
    mismatches = [(int(i), None, f"auto={auto_phase_new[i]} label_lama={old_phase[i]}") for i in disagree_idx]
    return mismatches, n_labeled, n_agree, n_disagree


def migrate_one(video_path, dry_run=False):
    try:
        meta = parse_video_filename(video_path)
    except FilenameParseError as e:
        print(f"[skip] {e}")
        return None

    rel = _relative_output_subpath(video_path, meta["exercise"])
    old_path = OLD_DIR / meta["exercise"] / rel
    new_path = NEW_DIR / meta["exercise"] / rel

    if not old_path.exists():
        print(f"[skip] {video_path.name}: label lama tidak ada ({old_path})")
        return None
    if not new_path.exists():
        print(f"[skip] {video_path.name}: hasil ekstraksi Tasks belum ada ({new_path}) -- "
              f"jalankan extract_landmarks.py (folder experimental_client_pipeline/scripts/) dulu")
        return None

    old_df = pd.read_csv(old_path)
    new_df = pd.read_csv(new_path)
    if "phase" not in old_df.columns or "class" not in old_df.columns:
        print(f"[skip] {video_path.name}: {old_path.name} belum dilabeli")
        return None
    if len(old_df) != len(new_df) or not (old_df["frame_idx"].to_numpy() == new_df["frame_idx"].to_numpy()).all():
        print(f"[SKIP -- TIDAK AMAN] {video_path.name}: jumlah/urutan frame beda")
        return None

    merged = old_df.drop(columns=LANDMARK_COLUMNS).merge(
        new_df[["frame_idx"] + LANDMARK_COLUMNS], on="frame_idx", how="left", validate="one_to_one",
    )
    old_meta_cols = [c for c in old_df.columns if c not in LANDMARK_COLUMNS]
    merged = merged[old_meta_cols + LANDMARK_COLUMNS]

    fps = float(old_df["fps"].iloc[0]) if "fps" in old_df.columns else 30.0
    mismatches, n_labeled, n_agree, n_disagree = check_mismatches(old_df, merged, meta["exercise"], fps)

    out_path = OUT_DIR / meta["exercise"] / rel
    if not dry_run:
        atomic_write_csv(merged, out_path)

    pct = 100.0 * n_agree / n_labeled if n_labeled else 0.0
    print(f"[{video_path.name}] cocok {n_agree}/{n_labeled} frame ({pct:.1f}%), {n_disagree} beda")
    if n_disagree:
        idx_preview = [m[0] for m in mismatches[:10]]
        print(f"    contoh frame beda: {idx_preview}{' ...' if n_disagree > 10 else ''}")

    return {"video": video_path.name, "out_path": str(out_path),
            "n_labeled": n_labeled, "n_agree": n_agree, "n_disagree": n_disagree}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from src.extraction.extract_landmarks import find_videos
    videos = find_videos(args.inputs)
    if not videos:
        print("Tidak ada video ditemukan.")
        return

    results = []
    for v in videos:
        r = migrate_one(v, dry_run=args.dry_run)
        if r is not None:
            results.append(r)

    total_labeled = sum(r["n_labeled"] for r in results)
    total_agree = sum(r["n_agree"] for r in results)
    pct = 100.0 * total_agree / total_labeled if total_labeled else 0.0
    print(f"\n=== Selesai: {len(results)} video diproses, "
          f"kecocokan fase keseluruhan {total_agree}/{total_labeled} ({pct:.1f}%) ===")


if __name__ == "__main__":
    main()
