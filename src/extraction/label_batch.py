import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.extraction.label_phase import add_common_args, label_one_csv


def find_csvs(inputs):
    paths = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.csv")))
        elif p.is_file():
            paths.append(p)
        else:
            print(f"[skip] path tidak ditemukan: {p}")
    return paths


def is_fully_labeled(csv_path):
    df = pd.read_csv(csv_path, usecols=lambda c: c in ("label_source",))
    if "label_source" not in df.columns:
        return False
    return (df["label_source"] == "manual").all()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="CSV file(s) atau folder berisi CSV")
    parser.add_argument("--skip-fully-labeled", action="store_true",
                         help="lewati CSV yang seluruh frame-nya sudah 'manual' (tidak ada sisa auto)")
    add_common_args(parser)
    args = parser.parse_args()

    csvs = find_csvs(args.inputs)
    if not csvs:
        print("Tidak ada CSV ditemukan.")
        return

    if args.skip_fully_labeled:
        before = len(csvs)
        csvs = [p for p in csvs if not is_fully_labeled(p)]
        skipped = before - len(csvs)
        if skipped:
            print(f"[info] {skipped} CSV sudah 100% manual, dilewati.")

    if not csvs:
        print("Semua CSV sudah 100% dilabel manual. Tidak ada yang perlu direview.")
        return

    print(f"=== BATCH LABELING: {len(csvs)} video ===")
    for i, p in enumerate(csvs, 1):
        print(f"\n{'=' * 60}\n[{i}/{len(csvs)}] {p.name}\n{'=' * 60}")
        label_one_csv(p, args)

    print(f"\n=== SELESAI: {len(csvs)} video sudah diproses/direview ===")


if __name__ == "__main__":
    main()
