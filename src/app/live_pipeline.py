"""Pipeline live utuh (bab 8.3.3 proposal): kamera -> YOLOv11+mask -> MediaPipe
-> joint angle -> sliding window -> normalisasi+prediksi -> repetition
counting -> hasil per frame.

Alur PERSIS proposal (bab 8.3.3, kata per kata):
    "Proses pertama dimulai dengan deteksi manusia menggunakan YOLO...
    Kemudian, dilakukan ekstraksi pose dengan menggunakan MediaPipe...
    Data yang diperoleh, nantinya diolah menjadi fitur... Data fitur
    tersebut kemudian dapat digunakan dalam proses klasifikasi postur...
    sistem juga mampu melakukan perhitungan repetisi berdasarkan deteksi
    siklus gerakan... dari perubahan joint angle secara temporal."

Reuse fungsi yang SUDAH ada (bukan tulis ulang) supaya konsisten 100% dengan
tahap preprocessing/training:
    - compute_frame_angles()      [src/features/joint_angles.py] -- per-titik
    - primary_angle_for_frame()   [src/features/joint_angles.py] -- utk rep counting
    - draw_skeleton()             [src/features/skeleton_draw.py]
    - PersonDetector/expand_bbox() [src/detection/yolo_detector.py]
    - OnlineRepCounter            [src/app/rep_counter.py]
    - Pipeline (scaler+rf) hasil train_model.py -- SATU pkl, sama seperti
      dipakai predict_video.py, TIDAK fit ulang apa pun.

Window (sliding, Eq.2/3) dibangun dari buffer ROLLING -- deque maxlen=swl,
begeser 1 frame tiap ada frame baru masuk (persis stride=1 training).
"""
import json
import pickle
from collections import deque
from pathlib import Path

# torch harus di-import sebelum pandas di Windows -- kalau kebalik, rebutan
# DLL c10.dll dan proses crash (WinError 1114).
import torch  # noqa: F401

import cv2
import mediapipe as mp
import pandas as pd

from src.detection.yolo_detector import PersonDetector, expand_bbox, TRAINING_VIDEO_ASPECT_RATIO
from src.features.joint_angles import (
    ANGLE_COLUMNS,
    POSE_LANDMARK_NAMES,
    PRIMARY_ANGLE_BASE_BY_EXERCISE,
    compute_frame_angles,
    primary_angle_for_frame,
)
from src.app.rep_counter import OnlineRepCounter

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"


class LivePosturePipeline:
    def __init__(self, exercise, yolo_confidence=0.7,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 countdown_sec=5.0, crop_padding=0.75, class_confidence_threshold=0.0,
                 yolo_redetect_every=1):
        """crop_padding: perluasan bbox YOLO sebelum di-crop utk MediaPipe
        (beda dari crop ketat Ko et al.) -- 0.75 dipilih dari sweep empiris,
        cukup mengurangi window hilang akibat occlusion tanpa memperbesar
        risiko crop kena orang lain di background.

        countdown_sec: detik pertama sesi = masa persiapan, tidak diproses
        jadi window/rep counting -- padanan live dari auto-trim-start di
        label_phase.py."""
        self.exercise = exercise
        self.countdown_sec = countdown_sec
        self.crop_padding = crop_padding
        # Ambang keyakinan KLASIFIKASI POSTUR oleh RF (beda dari yolo_confidence
        # yg keyakinan DETEKSI ORANG). Default 0.0 = mati -- efeknya beda jauh
        # per exercise di data kita, jadi tidak dipasang angka baku. Public
        # attribute supaya app.py bisa ubah tanpa reconstruct pipeline (mahal).
        self.class_confidence_threshold = class_confidence_threshold
        self._session_start_ts = None

        model_path = MODELS_DIR / f"{exercise}_rf.pkl"
        config_path = MODELS_DIR / f"{exercise}_feature_config.json"
        for p in (model_path, config_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} tidak ada -- jalankan dulu build_dataset.py & train_model.py {exercise}")
        with open(model_path, "rb") as f:
            self.pipeline = pickle.load(f)  # Pipeline(scaler, rf) -- SAMA persis dgn predict_video.py
        with open(config_path) as f:
            self.config = json.load(f)
        self.feat_cols = self.config["feat_cols"]
        self.window_sec = self.config["window_sec"]
        self.visibility_threshold = self.config["visibility_threshold"]
        # False (2D, sama Ko et al) kalau config lama belum punya field ini.
        self.use_z = self.config.get("use_z", False)
        # Fallback ke TRAINING_VIDEO_ASPECT_RATIO kalau config lama belum
        # simpan rasio video training exercise ini sendiri.
        self.training_video_aspect_ratio = (
            self.config.get("training_video_aspect_ratio") or TRAINING_VIDEO_ASPECT_RATIO)

        self.detector = PersonDetector(confidence_threshold=yolo_confidence)
        self._mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        # Counter resmi -- persis proposal bab 6.2.3 (prominence+distance) +
        # Eq.4 cosine similarity, param dipilih empiris (lihat rep_counter.py).
        self.rep_counter = OnlineRepCounter(exercise, min_pattern_similarity=0.70)

        self._buffer = None  # deque, dibuat begitu fps diketahui (frame pertama)
        self._swl = None
        self._fps = None
        self._last_confidence = None  # keyakinan RF window terakhir, diekspos via process_frame

        # Default 1 = deteksi tiap frame (perilaku lama). > 1: bbox dipakai
        # ulang sampai N frame, tapi kalau deteksi terakhir gagal, coba lagi
        # tiap frame (pulih cepat begitu orang balik ke frame).
        self.yolo_redetect_every = max(1, int(yolo_redetect_every))
        self._bbox_cache = None
        self._frames_since_detect = 0

    def _ensure_buffer(self, fps):
        if self._buffer is not None:
            return
        self._fps = fps
        self._swl = max(1, round(fps * self.window_sec))  # Eq.2
        self._buffer = deque(maxlen=self._swl)

    def process_frame(self, frame_bgr, timestamp_sec, fps=30.0):
        """1 frame kamera -> hasil (dict). Panggil ini tiap ada frame baru.

        Returns dict:
            row                 : landmark row (dict) -- buat draw_skeleton()
            bbox                : hasil YOLO (atau None)
            primary_angle       : sudut utama frame ini (atau None kalau NaN)
            predicted_class     : hasil klasifikasi window TERBARU (atau None
                                   kalau buffer belum penuh / window ini NaN)
            rep_count           : jumlah repetisi terkonfirmasi sejauh ini
            countdown_remaining : detik tersisa masa persiapan (None kalau
                                   sudah lewat masa itu -- pemrosesan normal)
        """
        if self._session_start_ts is None:
            self._session_start_ts = timestamp_sec
        elapsed = timestamp_sec - self._session_start_ts
        in_countdown = elapsed < self.countdown_sec
        # Masa persiapan tetap diproses via YOLO+MediaPipe (tracker tidak
        # idle) -- cuma hasilnya (window buffer + rep_counter) yg tidak
        # disentuh selama countdown, biar tracking tidak mulai dari nol
        # begitu countdown lewat (dulu idle total -> NaN menjalar berdetik-detik).

        self._ensure_buffer(fps)

        # Bbox cache dipakai ulang hanya kalau deteksi terakhir berhasil --
        # lihat yolo_redetect_every di __init__.
        if self._bbox_cache is not None and self._frames_since_detect < self.yolo_redetect_every:
            bbox = self._bbox_cache
            self._frames_since_detect += 1
        else:
            bbox = self.detector.detect(frame_bgr)
            self._bbox_cache = bbox
            self._frames_since_detect = 0
        crop_box = None
        if bbox:
            h, w = frame_bgr.shape[:2]
            crop_box = expand_bbox(bbox, self.crop_padding, w, h,
                                    target_aspect_ratio=self.training_video_aspect_ratio)  # (x1,y1,x2,y2) SETELAH padding+clamp
            pose_input = frame_bgr[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]].copy()
        else:
            pose_input = frame_bgr

        image = cv2.cvtColor(pose_input, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # Selalu dipanggil (bbox ada atau tidak) -- MediaPipe idle bikin
        # tracker reset, NaN menjalar berdetik-detik setelahnya.
        results = self._mp_pose.process(image)

        # Hasil MediaPipe dipaksa NaN kalau bbox kosong (YOLO tidak temukan
        # orang) -- tanpa gerbang ini, MediaPipe bisa "mengarang" pose dari
        # tekstur non-manusia. Gerbang di titik PERCAYA, bukan titik PANGGIL.
        row = {}
        if bbox and results.pose_landmarks is not None:
            for name, lm in zip(POSE_LANDMARK_NAMES, results.pose_landmarks.landmark):
                row[f"{name}_x"] = lm.x
                row[f"{name}_y"] = lm.y
                row[f"{name}_z"] = lm.z
                row[f"{name}_v"] = lm.visibility
        else:
            for name in POSE_LANDMARK_NAMES:
                for axis in ("x", "y", "z", "v"):
                    row[f"{name}_{axis}"] = float("nan")

        angles, _has_data = compute_frame_angles(row, self.visibility_threshold, self.use_z)

        primary = primary_angle_for_frame(row, self.exercise, self.visibility_threshold)
        primary = None if primary != primary else primary  # NaN -> None (v!=v <=> NaN)

        # Rep counting persis bab 6.2.3 + Eq.4 -- masa persiapan tetap
        # dikecualikan dari rep_counter/buffer window.
        predicted_class = None
        confidence = None
        if not in_countdown:
            self.rep_counter.update(primary, timestamp_sec, feature_vector=angles)

            frame_features = dict(angles)
            for name in POSE_LANDMARK_NAMES:
                for axis in ("x", "y", "z", "v"):
                    frame_features[f"{name}_{axis}"] = row[f"{name}_{axis}"]
            self._buffer.append(frame_features)

            if len(self._buffer) == self._swl:
                predicted_class = self._predict_from_buffer()
                confidence = self._last_confidence

        return {
            "row": row,
            "bbox": bbox,
            "crop_box": crop_box,  # posisi display_frame di dalam frame_bgr asli, setelah padding+clamp
            # display_frame: gambar yg koordinat landmark-nya cocok (relatif
            # ke crop, bukan frame_bgr asli) -- skeleton harus digambar di sini.
            "display_frame": pose_input,
            "primary_angle": primary,
            "predicted_class": predicted_class,
            "confidence": confidence,  # keyakinan RF window ini (None kalau buffer belum penuh / window NaN)
            "rep_count": self.rep_counter.rep_count,
            "countdown_remaining": (self.countdown_sec - elapsed) if in_countdown else None,
        }

    def _predict_from_buffer(self):
        """Bangun 1 baris window fitur dari buffer -- harus identik struktur
        kolomnya dgn build_windows_from_runs() (sliding_window.py). Coordinate
        dirata-ratakan per window (coord_mean_*)."""
        window_row = {}
        for f, frame_feat in enumerate(self._buffer):
            for col in ANGLE_COLUMNS:
                window_row[f"{col}_f{f}"] = frame_feat[col]
        for name in POSE_LANDMARK_NAMES:
            for axis in ("x", "y", "z", "v"):
                col = f"{name}_{axis}"
                # Rata-rata skip NaN (persis pandas .mean()) -- NaN cuma
                # kalau semua frame di window ini NaN (occlusion penuh).
                values = [frame_feat[col] for frame_feat in self._buffer]
                valid = [v for v in values if v == v]
                window_row[f"coord_mean_{col}"] = (sum(valid) / len(valid)) if valid else float("nan")
        # NaN di window (angle ATAU coordinate) -- window ini TIDAK reliable,
        # sama aturan build_windows_from_runs() (discard, bukan interpolasi).
        if any(v != v for v in window_row.values()):
            self._last_confidence = None
            return None

        X = pd.DataFrame([{c: window_row[c] for c in self.feat_cols}], columns=self.feat_cols)
        # predict_proba() (bukan predict()) supaya bisa cek keyakinan RF
        # sebelum dipakai. classes_ dari step 'rf' -- Pipeline gabungan
        # scaler+rf tidak punya .classes_ sendiri.
        proba = self.pipeline.predict_proba(X)[0]
        rf_step = self.pipeline.named_steps["rf"]
        best_idx = proba.argmax()
        confidence = float(proba[best_idx])
        self._last_confidence = confidence  # diekspos via process_frame (lihat __init__)
        if confidence < self.class_confidence_threshold:
            return None  # RF "tidak cukup yakin" -- diperlakukan sama spt window blm penuh
        return rf_step.classes_[best_idx]
