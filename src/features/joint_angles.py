"""11 joint angles (Eq. 1, proposal bab 6.2.1) from MediaPipe Pose landmarks.

Angle at vertex B formed by points A-B-C, computed via the arccos dot-product
form specified in the proposal (Eq. 1), not the arctan2 form used in Ko et
al.'s code (mathematically equivalent, both give the unsigned angle in
[0, 180] degrees).
"""
import numpy as np

# MediaPipe Pose landmark names in official landmark-index order (0-32).
POSE_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

VISIBILITY_THRESHOLD = 0.6

# Three-point (A, B, C) definitions for each joint angle. Vertex = B.
# Matches the 11 angles Ko et al. (2024) extract, referenced by the proposal
# (bab 6.2.1): Neck, L/R Shoulder, L/R Elbow, L/R Hip, L/R Knee, L/R Ankle.
JOINT_ANGLE_TRIPLES = {
    "left_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder_angle": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder_angle": ("right_elbow", "right_shoulder", "right_hip"),
    "left_hip_angle": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip_angle": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
    "left_ankle_angle": ("left_knee", "left_ankle", "left_heel"),
    "right_ankle_angle": ("right_knee", "right_ankle", "right_heel"),
}
# neck_angle is the average of two triples (left and right), handled separately.
NECK_ANGLE_TRIPLES = (
    ("left_shoulder", "nose", "left_hip"),
    ("right_shoulder", "nose", "right_hip"),
)

ANGLE_COLUMNS = ["neck_angle"] + list(JOINT_ANGLE_TRIPLES.keys())

# Primary joint angle used for automatic concentric/eccentric phase detection
# (user-specified mapping): squat -> knee, bench press -> elbow, deadlift -> hip.
PRIMARY_ANGLE_BASE_BY_EXERCISE = {
    "squat": "knee_angle",
    "benchpress": "elbow_angle",
    "deadlift": "hip_angle",
}


def calculate_angle(a, b, c):
    """Eq. 1: angle at vertex b (degrees), via arccos of the BA.BC dot product."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return np.nan
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def landmark_xy(row, name):
    return (row[f"{name}_x"], row[f"{name}_y"])


def landmark_xyz(row, name):
    return (row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_z"])


def landmark_point(row, name, use_z=False):
    """2D (x,y) secara default -- PERSIS kode asli Ko et al. (`calculateAngle`
    di Streamlit.py/Afterprocessing.ipynb cuma pakai index [0],[1], BUKTI
    dicek langsung: `right_knee = [landmarks[...].x, landmarks[...].y]`, z
    TIDAK PERNAH dipakai di rumus sudut manapun di seluruh repo mereka,
    walau z TETAP disimpan di data mentah -- pola yang SAMA kita ikuti:
    132 kolom koordinat (termasuk z) tetap ada sbg fitur pendukung, angle
    tetap 2D). `use_z=True` HANYA untuk eksperimen pembanding (lihat
    build_dataset.py --use-z) -- proposal Eq.1 sendiri tidak menspesifikasi
    dimensi ("tiga titik persendian" generik), jadi 3D bukan pelanggaran
    metodologi, TAPI juga BUKAN default -- default TETAP 2D (konsisten Ko
    et al & histori sistem ini)."""
    return landmark_xyz(row, name) if use_z else landmark_xy(row, name)


def landmark_visibility(row, name):
    return row[f"{name}_v"]


def compute_joint_angles(row, visibility_threshold=VISIBILITY_THRESHOLD, use_z=False):
    """Compute all 11 joint angles (degrees) from one landmark row (dict-like).

    Filter visibility PER-TITIK (bab 6.2.1 proposal: "hanya titik persendian
    dengan nilai visibility di atas ambang tertentu... yang digunakan" --
    "titik persendian" jamak/individual, BUKAN per-frame all-or-nothing).
    Ko et al. (paper, Eq.2) juga notasinya per-poin: {pi | vi>=0.6}.

    Tiap angle di-cek SENDIRI-SENDIRI dari 3 titik triple-nya sendiri -- kalau
    ada 1 SAJA dari 3 titik itu di bawah ambang, angle itu jadi NaN, TAPI
    angle LAIN yang titiknya lengkap tetap dihitung normal. Penting utk
    kamera 1 sisi (45 derajat) yang wajar nutup 1 sisi tubuh (mis. tangan
    belakang badan di bench press) -- occlusion 1 titik dulu TIDAK BOLEH
    membuang seluruh 11 angle di frame itu (lihat diskusi proyek)."""
    angles = {}
    for angle_name, (a_name, b_name, c_name) in JOINT_ANGLE_TRIPLES.items():
        triple_ok = all(landmark_visibility(row, n) >= visibility_threshold
                         for n in (a_name, b_name, c_name))
        angles[angle_name] = (
            calculate_angle(landmark_point(row, a_name, use_z), landmark_point(row, b_name, use_z),
                             landmark_point(row, c_name, use_z))
            if triple_ok else np.nan
        )

    neck_values = []
    for a_name, b_name, c_name in NECK_ANGLE_TRIPLES:
        triple_ok = all(landmark_visibility(row, n) >= visibility_threshold
                         for n in (a_name, b_name, c_name))
        neck_values.append(
            calculate_angle(landmark_point(row, a_name, use_z), landmark_point(row, b_name, use_z),
                             landmark_point(row, c_name, use_z))
            if triple_ok else np.nan
        )
    angles["neck_angle"] = float(np.nanmean(neck_values)) if not all(np.isnan(v) for v in neck_values) else np.nan
    return angles


def compute_frame_angles(row, visibility_threshold=VISIBILITY_THRESHOLD, use_z=False):
    """Hitung 11 joint angle (per-titik, lihat compute_joint_angles) + tandai
    apakah frame ini "usable" secara LONGGAR -- artinya orangnya masih
    terdeteksi & SETIDAKNYA 1 dari 11 angle bisa dihitung (bukan lagi
    "SEMUA 15 titik harus visible").

    Returns (angles: dict[str, float] -- BISA berisi NaN per-angle, punya_data: bool).
    "punya_data=False" cuma kalau BENAR-BENAR semua angle NaN (orang
    hilang total dari frame). Window yang MASIH mengandung NaN di salah
    satu angle-nya dibuang belakangan di level WINDOW (build_windows_from_runs
    di sliding_window.py) -- bukan di level frame -- supaya run tetap utuh
    (tidak terpecah gara-gara occlusion sesaat 1 titik), tapi tidak ada NaN
    yang lolos sampai ke classifier (RF tidak terima NaN)."""
    angles = compute_joint_angles(row, visibility_threshold, use_z)
    has_data = not all(np.isnan(v) for v in angles.values())
    return angles, bool(has_data)


def primary_angle_for_frame(row, exercise, visibility_threshold=VISIBILITY_THRESHOLD):
    """Pick the primary angle (per PRIMARY_ANGLE_BASE_BY_EXERCISE) for one frame.

    Averages left/right when both sides meet the visibility threshold, else
    falls back to whichever side is visible, else NaN (occluded frame).
    """
    base = PRIMARY_ANGLE_BASE_BY_EXERCISE[exercise]
    triple_l = JOINT_ANGLE_TRIPLES[f"left_{base}"]
    triple_r = JOINT_ANGLE_TRIPLES[f"right_{base}"]

    vis_l = min(landmark_visibility(row, n) for n in triple_l)
    vis_r = min(landmark_visibility(row, n) for n in triple_r)

    angle_l = calculate_angle(*[landmark_xy(row, n) for n in triple_l])
    angle_r = calculate_angle(*[landmark_xy(row, n) for n in triple_r])

    ok_l = vis_l >= visibility_threshold
    ok_r = vis_r >= visibility_threshold

    if ok_l and ok_r:
        return (angle_l + angle_r) / 2.0
    if ok_l:
        return angle_l
    if ok_r:
        return angle_r
    return np.nan
