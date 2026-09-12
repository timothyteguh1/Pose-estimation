"""Scaler khusus fitur ANGLE (165 kolom: 11 angle x swl posisi frame, hasil
flatten). Kolom coordinate (132, rata-rata per window) SENGAJA TIDAK
disentuh sama sekali -- lihat penjelasan di bawah kenapa.

KENAPA CUMA ANGLE (bukan angle+coordinate)
-------------------------------------------------------------
Sempat coba normalisasi keduanya (angle+coordinate, 143 kolom) supaya
"konsisten", tapi setelah dicek ulang ke 3 sumber -- SEMUANYA cuma bicara
angle, TIDAK ADA yang bilang coordinate perlu dinormalisasi:
  - Bab 6.2.4 proposal, eksplisit: "Data JOINT ANGLE yang dihasilkan dari
    proses ekstraksi memiliki rentang nilai yang berbeda-beda, sehingga
    perlu adanya normalisasi data" -- cuma sebut joint angle.
  - Bab 8.3.1, dibaca DALAM KONTEKS (bukan berdiri sendiri): "...diolah
    menjadi fitur berupa JOINT ANGLE... Setelah FITUR sudah diperoleh...
    normalisasi data... untuk menyamakan rentang nilai FITUR." Kata "fitur"
    di sini merujuk balik ke "fitur berupa joint angle" yg baru disebut --
    bab 8.3.1 TIDAK PERNAH menyebut "coordinate" sbg entitas terpisah dalam
    narasinya sendiri (itu murni konsep bab 6.2.1 yang kita gabung sendiri).
  - Ko et al. (paper, bagian "DATA NORMALIZATION"), eksplisit: "the Pose
    Estimation module of MediaPipe provides all joint position vectors as
    normalized values in the range of [-1, 1]... this study used Min-Max
    scaling to normalize the extracted ANGLE data" -- coordinate dianggap
    SUDAH cukup konsisten skalanya (native MediaPipe [-1,1]), yang perlu
    diseragamkan cuma angle (rentangnya beda-beda tiap sendi, 0-180 derajat).

Jadi coordinate TETAP jadi fitur pendukung (rata-rata mentah per window,
33 landmark x x/y/z/v = 132 kolom -- "selengkap kolom mentah Ko et al.",
lihat sliding_window.py), TAPI TIDAK di-Min-Max -- persis Ko et al.

URUTAN (literal, bab 8.3.1: normalisasi ANGLE dulu, baru windowing)
-------------------------------------------------------------
build_dataset.py: fit_frame()+transform_frame() dipanggil di level FRAME
(11 kolom ANGLE_COLUMNS SAJA), fit HANYA dari frame yang masuk alokasi
TRAIN, diterapkan ke SEMUA frame (train+test) -- SEBELUM
build_windows_from_runs() flatten nilainya jadi kolom _f0.._fN. Kolom
coordinate tetap dihitung rata-rata mentah per window seperti biasa, tanpa
langkah normalisasi apa pun.

Untuk INFERENSI ke video BARU (predict_video.py): transform() cuma
menyentuh kolom angle_fN (broadcast per base-angle), kolom coord_mean_*
dibiarkan apa adanya.

CATATAN: karena classifier akhir adalah Random Forest yang invariant
terhadap transformasi monoton per kolom, keputusan normalisasi/tidak pada
coordinate TIDAK mengubah hasil evaluasi (accuracy/F1) -- perbaikan ini
murni supaya proses match literal ke 3 sumber di atas.
"""
import re

import numpy as np
import pandas as pd

from src.features.joint_angles import ANGLE_COLUMNS

_FRAME_COL_RE = re.compile(r"^(.+)_f(\d+)$")


class WindowFeatureScaler:
    def __init__(self):
        self.angle_min_ = {}  # base-angle (mis. "left_knee_angle") -> min
        self.angle_max_ = {}
        self.angle_cols_by_base_ = {}  # base-angle -> [window col _f0.._fN, ...] (butuh broadcast)

    # ------------------------------------------------------------------
    # Tahap SEBELUM windowing: fit & transform ANGLE di level FRAME.
    # ------------------------------------------------------------------
    def fit_frame(self, frame_df, columns=ANGLE_COLUMNS):
        """frame_df: DataFrame level FRAME (1 baris = 1 frame), kolom =
        ANGLE_COLUMNS. HANYA dari frame yang masuk alokasi TRAIN (cegah
        leakage)."""
        for col in columns:
            if col not in frame_df.columns:
                continue
            values = frame_df[col].to_numpy(dtype=float)
            self.angle_min_[col] = float(np.nanmin(values))
            self.angle_max_[col] = float(np.nanmax(values))
        return self

    def transform_frame(self, frame_df, columns=ANGLE_COLUMNS):
        """Terapkan angle_min_/angle_max_ ke DataFrame level FRAME manapun
        (train MAUPUN test -- scaler-nya sama, cuma di-fit dari train).
        Dipanggil SEBELUM windowing. Kolom di luar ANGLE_COLUMNS (termasuk
        coordinate) dibiarkan tidak berubah."""
        frame_df = frame_df.copy()
        for col in columns:
            if col not in self.angle_min_ or col not in frame_df.columns:
                continue
            lo, hi = self.angle_min_[col], self.angle_max_[col]
            span = hi - lo
            frame_df[col] = 0.0 if span == 0 else (frame_df[col] - lo) / span
        return frame_df

    # ------------------------------------------------------------------
    # Kenali struktur kolom WINDOW (dibutuhkan transform() utk inferensi).
    # ------------------------------------------------------------------
    def configure_columns(self, window_columns):
        angle_cols_by_base = {}
        for c in window_columns:
            m = _FRAME_COL_RE.match(c)
            if m and m.group(1) in ANGLE_COLUMNS:
                angle_cols_by_base.setdefault(m.group(1), []).append(c)
        for cols in angle_cols_by_base.values():
            cols.sort(key=lambda c: int(c.rsplit("_f", 1)[1]))
        self.angle_cols_by_base_ = angle_cols_by_base
        return self

    # ------------------------------------------------------------------
    # Inferensi ke video BARU (predict_video.py): window MENTAH TOTAL
    # (angle belum ternormalisasi) -> 1 panggilan. coord_mean_* dibiarkan.
    # ------------------------------------------------------------------
    def transform(self, X):
        was_df = isinstance(X, pd.DataFrame)
        X = pd.DataFrame(X).copy()

        for base, cols in self.angle_cols_by_base_.items():
            if base not in self.angle_min_:
                continue
            lo, hi = self.angle_min_[base], self.angle_max_[base]
            span = hi - lo
            X[cols] = 0.0 if span == 0 else (X[cols] - lo) / span

        return X if was_df else X.to_numpy()
