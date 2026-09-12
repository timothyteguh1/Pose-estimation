"""Migrasi extracted_landmarks/*.csv YANG SUDAH DILABELI ke use_yolo_crop=True
(YOLO deteksi+crop sebelum MediaPipe, PERSIS proses live_pipeline.py -- lihat
diskusi proyek soal mismatch train/live, neck_angle geser sampai 55 derajat
tanpa ini) -- TANPA kehilangan label 'class'/'phase' yang sudah dibuat manual.

Cara kerja (aman, bukan menimpa polos):
1. Video mentah di-ekstrak ULANG dari nol pakai use_yolo_crop=True -> landmark
   BARU (koordinat mentah beda krn crop, frame_idx/timestamp_sec SAMA).
2. CSV LAMA (sudah ada kolom class/phase) dibaca sbg sumber label.
3. Cek frame_idx & jumlah baris SAMA PERSIS antara lama vs baru -- kalau
   TIDAK sama, video itu DILEWATI (bukan dipaksa, supaya tidak salah pasang
   label ke frame yg salah).
4. Kolom landmark (x/y/z/v) diganti ke versi BARU (hasil crop), kolom
   class/phase/metadata TETAP dari yang LAMA -- digabung per frame_idx.
5. Ditulis balik ke file yg SAMA (atomic write, ada backup .bak sebelum
   ditimpa) -- kamu TIDAK PERLU labeling ulang dari nol.

Usage:
    python scripts/migrate_yolo_crop.py data/raw_videos/squat/p1/correct/squat_correct_p1_left45_take01.mp4
    python scripts/migrate_yolo_crop.py data/raw_videos/squat   # semua video di folder itu (rekursif)
    python scripts/migrate_yolo_crop.py data/raw_videos/squat --dry-run   # cek dulu tanpa nulis apa pun
"""
import argparse
import shutil
import sys
from pathlib import Path

# PENTING (murni soal urutan import di Windows, TIDAK ada hubungannya dgn
# logika/metodologi): torch (dipakai YOLOv11 via ultralytics) HARUS di-import
# SEBELUM pandas -- kalau kebalik, keduanya rebutan runtime DLL yg sama
# (c10.dll) dan proses CRASH (WinError 1114). Dikonfirmasi langsung lewat
# tes terisolasi. Baris ini SENGAJA di atas `import pandas`.
import torch  # noqa: F401

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.extraction.extract_landmarks import (
    LANDMARK_COLUMNS, _relative_output_subpath, extract_one_video, find_videos,
)
from src.extraction.filename_parser import FilenameParseError, parse_video_filename
from src.io_utils import atomic_write_csv

EXTRACTED_DIR = REPO_ROOT / "data" / "extracted_landmarks"


def migrate_one(video_path, dry_run=False):
    try:
        meta = parse_video_filename(video_path)
    except FilenameParseError as e:
        print(f"[skip] {e}")
        return False

    old_path = EXTRACTED_DIR / meta["exercise"] / _relative_output_subpath(video_path, meta["exercise"])
    if not old_path.exists():
        print(f"[skip] {video_path.name}: belum pernah diekstrak sama sekali ({old_path} tidak ada)")
        return False
    old_df = pd.read_csv(old_path)
    if "phase" not in old_df.columns or "class" not in old_df.columns:
        print(f"[skip] {video_path.name}: {old_path.name} belum dilabeli (tidak ada kolom class/phase) -- "
              f"pakai extract_landmarks.py --use-yolo-crop langsung, bukan skrip ini.")
        return False

    print(f"[migrate] {video_path.name} -- ekstrak ulang dgn YOLO crop...")
    new_df, fps = extract_one_video(
        video_path, min_detection_confidence=0.5, min_tracking_confidence=0.5,
        use_yolo_crop=True, yolo_confidence=0.7, crop_padding=0.75,
    )

    if len(new_df) != len(old_df):
        print(f"[SKIP -- TIDAK AMAN] {video_path.name}: jumlah frame beda "
              f"(lama={len(old_df)}, baru={len(new_df)}) -- kemungkinan video sumbernya "
              f"sudah berubah sejak label lama dibuat. TIDAK di-migrate, label lama dibiarkan apa adanya.")
        return False
    if not (new_df["frame_idx"].to_numpy() == old_df["frame_idx"].to_numpy()).all():
        print(f"[SKIP -- TIDAK AMAN] {video_path.name}: urutan frame_idx tidak cocok persis -- dilewati.")
        return False

    # Gabung: landmark BARU (hasil crop) + metadata/class/phase dari yg LAMA,
    # dicocokkan per frame_idx (bukan asumsi urutan baris, walau seharusnya sama).
    merged = old_df.drop(columns=LANDMARK_COLUMNS).merge(
        new_df[["frame_idx"] + LANDMARK_COLUMNS], on="frame_idx", how="left", validate="one_to_one",
    )
    # urutan kolom: metadata dulu (persis punya lama), landmark BARU, class/phase di akhir (ikut posisi lama)
    old_meta_cols = [c for c in old_df.columns if c not in LANDMARK_COLUMNS]
    merged = merged[old_meta_cols + LANDMARK_COLUMNS] if all(c in merged.columns for c in old_meta_cols) \
        else merged

    n_old_valid = old_df[LANDMARK_COLUMNS[0]].notna().sum()
    n_new_valid = merged[LANDMARK_COLUMNS[0]].notna().sum()
    print(f"  frame terdeteksi pose -- lama (tanpa crop): {n_old_valid}/{len(old_df)}, "
          f"baru (dgn crop YOLO): {n_new_valid}/{len(merged)}")

    if dry_run:
        print(f"  [dry-run] TIDAK ditulis. ({old_path})")
        return True

    backup_path = old_path.with_suffix(".csv.bak")
    if not backup_path.exists():
        shutil.copy2(old_path, backup_path)
        print(f"  [ok] backup disimpan: {backup_path.name}")
    atomic_write_csv(merged, old_path)
    print(f"  [ok] ditulis ulang: {old_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="video mentah (file/folder) yg mau di-migrate labelnya")
    parser.add_argument("--dry-run", action="store_true", help="cek kecocokan frame tanpa menulis apa pun")
    args = parser.parse_args()

    videos = find_videos(args.inputs)
    if not videos:
        print("Tidak ada video ditemukan.")
        return

    ok, skipped = 0, 0
    for v in videos:
        if migrate_one(v, dry_run=args.dry_run):
            ok += 1
        else:
            skipped += 1
    print(f"\n=== Selesai: {ok} berhasil, {skipped} dilewati (dari {len(videos)} video) ===")


if __name__ == "__main__":
    main()
