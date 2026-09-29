"""Horizontal flipping augmentation (Zhu & Zhu, 2021, Traitement du Signal
38(2):529-538, Persamaan 5), diterapkan ke window fitur sebelum training.
Hanya dipakai di data TRAINING (build_dataset.py --flip-augment) -- tidak
pernah dipanggil saat live/inference.
"""
import pandas as pd

from src.features.joint_angles import (
    LEFT_RIGHT_ANGLE_PAIRS,
    LEFT_RIGHT_LANDMARK_PAIRS,
    POSE_LANDMARK_NAMES,
    compute_relative_coordinates,
)


def flip_horizontal(df):
    """1 baris window -> 1 baris window hasil flip horizontal. Label kelas
    tidak berubah (postur tidak bergantung sisi kiri/kanan)."""
    out = df.copy()

    for left, right in LEFT_RIGHT_ANGLE_PAIRS:
        n_frames = sum(1 for c in df.columns if c.startswith(left + "_f"))
        for f in range(n_frames):
            a, b = f"{left}_f{f}", f"{right}_f{f}"
            out[a], out[b] = df[b], df[a]

    for left, right in LEFT_RIGHT_LANDMARK_PAIRS:
        for axis in ("x", "y", "z", "v"):
            a, b = f"coord_mean_{left}_{axis}", f"coord_mean_{right}_{axis}"
            if a in df.columns:
                out[a], out[b] = df[b], df[a]

    for name in POSE_LANDMARK_NAMES:
        col = f"coord_mean_{name}_x"
        if col in df.columns:
            out[col] = 1.0 - out[col]

    if f"coord_rel_{POSE_LANDMARK_NAMES[0]}_x" in df.columns:
        out.update(pd.DataFrame(compute_relative_coordinates(out), index=out.index))

    return out
