"""Gambar skeleton (titik+garis) dari 1 baris landmark CSV ke atas frame video.

Dipakai bareng oleh src/extraction/label_phase.py (review interaktif) dan
src/extraction/preview_phase.py (render video QA), supaya konsisten.
"""
import cv2
import mediapipe as mp
import pandas as pd

from src.features.joint_angles import POSE_LANDMARK_NAMES, VISIBILITY_THRESHOLD

NAME_TO_INDEX = {name: i for i, name in enumerate(POSE_LANDMARK_NAMES)}
FULL_MP_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS  # 33 titik lengkap (termasuk mata/mulut/jari)

# Skeleton "badan saja" (gaya Ko et al.) -- cuma titik yang dipakai 11 joint
# angle + garis torso, tanpa detail wajah/jari. Murni tampilan, tidak
# mengubah fitur training (tetap dari JOINT_ANGLE_TRIPLES di joint_angles.py).
_BODY_LINKS = [
    ("nose", "left_shoulder"), ("nose", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("left_heel", "left_foot_index"), ("left_ankle", "left_foot_index"),
    ("right_ankle", "right_heel"), ("right_heel", "right_foot_index"), ("right_ankle", "right_foot_index"),
]
BODY_CONNECTIONS = frozenset(
    (NAME_TO_INDEX[a], NAME_TO_INDEX[b]) for a, b in _BODY_LINKS
)

# Sama persis threshold data (0.6, bab 8.3.1) -- disamakan supaya titik yang
# digambar jujur mencerminkan titik yang beneran lolos jadi fitur (dulu 0.3,
# lebih longgar, bikin titik yg sebenarnya dibuang dari data tetap kelihatan
# tergambar normal).
DEFAULT_MIN_DRAW_VISIBILITY = VISIBILITY_THRESHOLD


def landmark_pixel_points(row, w, h, min_visibility=DEFAULT_MIN_DRAW_VISIBILITY):
    """dict {landmark_index: (px, py)} untuk titik yang cukup visible."""
    points = {}
    for name in POSE_LANDMARK_NAMES:
        x = pd.to_numeric(row.get(f"{name}_x"), errors="coerce")
        y = pd.to_numeric(row.get(f"{name}_y"), errors="coerce")
        v = pd.to_numeric(row.get(f"{name}_v"), errors="coerce")
        if pd.notna(x) and pd.notna(y) and pd.notna(v) and v >= min_visibility:
            points[NAME_TO_INDEX[name]] = (int(x * w), int(y * h))
    return points


def draw_skeleton(frame, row, color=(60, 200, 60), min_visibility=DEFAULT_MIN_DRAW_VISIBILITY,
                   connections=BODY_CONNECTIONS):
    """Gambar skeleton dari 1 baris landmark CSV langsung ke `frame` (in-place).

    Default `connections=BODY_CONNECTIONS` -- badan saja, gaya Ko et al.
    Pakai `connections=FULL_MP_CONNECTIONS` utk tampilan 33 titik lengkap.

    Returns jumlah titik badan yang berhasil digambar (0 = orang tidak
    terdeteksi/occluded total di frame ini)."""
    h, w = frame.shape[:2]
    all_points = landmark_pixel_points(row, w, h, min_visibility)
    body_indices = {i for pair in connections for i in pair}
    points = {i: pt for i, pt in all_points.items() if i in body_indices}

    for a_idx, b_idx in connections:
        if a_idx in points and b_idx in points:
            cv2.line(frame, points[a_idx], points[b_idx], color, 2)
    for pt in points.values():
        cv2.circle(frame, pt, 3, color, -1)

    return len(points)
