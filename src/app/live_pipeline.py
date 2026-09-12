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

# PENTING (murni soal urutan import di Windows, TIDAK ada hubungannya dgn
# logika/metodologi): torch (dipakai YOLOv11 via ultralytics, lewat
# PersonDetector di bawah) HARUS di-import SEBELUM pandas -- kalau kebalik,
# keduanya rebutan runtime DLL yg sama (c10.dll) dan proses bisa CRASH
# (WinError 1114) -- ditemukan+dikonfirmasi langsung lewat tes terisolasi
# (lihat diskusi proyek). Baris ini SENGAJA di atas `import pandas`.
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
        (NOVELTY kita, beda dari crop ketat Ko et al. -- lihat docstring
        crop_bbox() di yolo_detector.py). Default 0.75 dipilih empiris: sweep
        di video paling occluded (benchpress_armsspread_left45), diukur di
        THRESHOLD VISIBILITY ASLI 0.6 (bukan diturunkan) --
            padding=0.0 (persis Ko et al.) -> window_survive 19.1%
            padding=0.5                     -> window_survive 19.1% (BELUM cukup)
            padding=0.75                    -> window_survive 34.7%
            padding=1.0                     -> window_survive 34.8%
            padding=FULL (frame utuh, tanpa YOLO sama sekali)
                                             -> window_survive 35.9% (batas atas teoretis,
                                                dikonfirmasi PERSIS sama dgn hasil offline:
                                                418/1165 window, sanity-check tervalidasi)
        0.75 dan 1.0 hasilnya nyaris sama (34.7% vs 34.8%, beda 0.1pp) --
        dipilih 0.75 (bukan 1.0) krn area crop-nya lebih kecil (risiko
        "kena" orang lain di background lebih rendah, lihat diskusi ruang
        lingkup #17) tanpa kehilangan manfaatnya sama sekali. Threshold
        visibility TETAP 0.6, PERSIS proposal/Ko et al., nol deviasi angka.
        (padding lebih besar dari 1.0 dicoba juga, tidak konsisten lebih
        baik -- 1.5 malah turun ke 27.8%, kemungkinan noise dari ROI
        detector internal MediaPipe sendiri.)

        countdown_sec: beberapa detik pertama sesi dianggap "persiapan"
        (masa siap-siap, belum gerakan asli) -- TIDAK diproses sama sekali
        (YOLO/MediaPipe/window/rep counting semua di-skip, hemat komputasi
        sekalian). Ini padanan LIVE dari auto-trim-start di label_phase.py
        (frame sebelum penekanan space pertama = excluded, bukan gerakan
        asli) -- supaya training dan implementasi konsisten memperlakukan
        masa persiapan yang sama."""
        self.exercise = exercise
        self.countdown_sec = countdown_sec
        self.crop_padding = crop_padding
        # Ambang keyakinan klasifikasi (BARU, permintaan user, gaya slider
        # "confidence threshold" Ko et al -- lihat Streamlit.py mereka).
        # BEDA dari yolo_confidence (itu keyakinan DETEKSI ORANG oleh YOLO) --
        # ini keyakinan KLASIFIKASI POSTUR oleh RF (predict_proba, lihat
        # _predict_from_buffer). Default 0.0 = MATI (perilaku lama: RF selalu
        # pakai argmax apapun keyakinannya) -- SENGAJA tidak dipasang angka
        # "bagus" sepihak, karena diukur langsung: keyakinan RF kita median
        # cuma ~0.45 dan variasinya BEDA jauh per exercise (squat & deadlift
        # threshold 0.5 justru menaikkan akurasi window yg lolos ke 0.72-0.99,
        # tapi benchpress di threshold sama malah TURUN ke 0.48) -- jadi user
        # yg coba-coba sendiri lewat slider, bukan angka baku dari kami.
        # Public attribute (bukan lewat method) SENGAJA -- supaya app.py bisa
        # ubah live tiap rerun tanpa reconstruct pipeline (reload YOLO/
        # MediaPipe/RF ulang itu mahal, ganti 1 angka ambang tidak perlu itu).
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
        # Rasio video training EXERCISE INI SENDIRI (disimpan build_dataset.py) --
        # fallback ke TRAINING_VIDEO_ASPECT_RATIO (angka benchpress) cuma kalau
        # config lama belum punya field ini (mis. dibuat sebelum fix ini ada).
        self.training_video_aspect_ratio = (
            self.config.get("training_video_aspect_ratio") or TRAINING_VIDEO_ASPECT_RATIO)

        self.detector = PersonDetector(confidence_threshold=yolo_confidence)
        self._mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        # Counter RESMI -- PERSIS sesuai proposal bab 6.2.3 (prominence+distance)
        # + Eq.4 cosine similarity (min_prominence_deg/min_distance_sec/
        # min_pattern_similarity dipilih empiris dari data kita sendiri, lihat
        # diskusi proyek & histori tuning di REP_COUNTING_PARAMS,
        # rep_counter.py). Kelas OnlineRepCounter ini murni -- TIDAK ada
        # mekanisme tambahan di luar proposal (mis. "motion onset detection"
        # yg pernah dicoba sempat ditambahkan di sini lalu DIHAPUS TOTAL --
        # bukan bagian dari bab 6.2.3, dan project decision-nya: hanya
        # countdown_sec di atas yg dipakai utk exclude masa persiapan, bukan
        # mekanisme tambahan yg lebih kompleks).
        self.rep_counter = OnlineRepCounter(exercise, min_pattern_similarity=0.70)

        self._buffer = None  # deque, dibuat begitu fps diketahui (frame pertama)
        self._swl = None
        self._fps = None
        self._last_confidence = None  # keyakinan RF window terakhir (diekspos di
        # process_frame -- buat logging/analisis & lapisan "warning timing".
        # BUKAN dipakai gating verdict: conf-gate terbukti merugikan video yg
        # landmark-nya banyak NaN, lihat diskusi proyek).

        # yolo_redetect_every (OPT-IN, default 1 = perilaku LAMA, deteksi tiap
        # frame, TIDAK berubah kalau tidak diset eksplisit): YOLO itu komponen
        # terberat (~40-50ms/frame, diukur langsung) -- kalau > 1, bbox dipakai
        # ULANG sampai N frame sebelum YOLO dipanggil lagi, TAPI kalau deteksi
        # TERAKHIR gagal (bbox None -- tidak ada orang/gagal deteksi), coba
        # lagi TIAP frame (bukan nunggu N frame) supaya cepat pulih begitu
        # orang balik ke frame -- bukan malah lebih lambat sadar orangnya
        # sudah hilang/sudah balik. Divalidasi (diskusi proyek 2026-09-11):
        # deteksi 1x di awal video TERBUKTI GAGAL di data kita sendiri
        # (benchpress_p1, deadlift_p1 -- bbox frame pertama cuma nangkap
        # sepotong badan krn orang baru masuk frame, 97-99% frame lain jadi
        # salah crop kalau bbox itu dipakai terus) -- makanya redeteksi
        # PERIODIK, BUKAN cuma sekali, adalah jalan tengah yg aman.
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
        # PENTING (perbaikan arsitektur, lihat diskusi proyek): masa persiapan
        # ini TETAP diproses via YOLO+MediaPipe (baris di bawah TIDAK di-skip
        # lagi) -- HANYA hasilnya (window buffer + rep_counter) yang tidak
        # disentuh selama countdown, bukan prosesnya. Dulu MediaPipe dibiarkan
        # idle TOTAL selama countdown_sec detik (0 frame diproses sama
        # sekali) -- terbukti lewat eksperimen ini bikin tracking internal
        # MediaPipe mulai dari NOL begitu countdown lewat (persis kondisi
        # "fresh pose, tanpa histori" yang terbukti gagal deteksi di frame
        # yang justru berhasil kalau diproses BERURUTAN dari awal, sama
        # seperti extract_landmarks.py). Dampaknya: NaN berkepanjangan
        # (bukan cuma di masa countdown, tapi MENJALAR ke beberapa detik
        # SETELAHNYA juga) -- dibuktikan turun dari rep_count=2 (GT=5, 125
        # frame NaN) jadi rep_count=5 PERSIS (0 frame NaN) di video uji,
        # cuma dengan menghilangkan jeda idle ini (BUKAN soal smoothing bbox
        # -- itu sudah dicoba terpisah & terbukti TIDAK berpengaruh sama
        # sekali, ditinggalkan). Window buffer & rep_counter TETAP tidak
        # disentuh selama countdown (exclude behavior TIDAK berubah).

        self._ensure_buffer(fps)

        # YOLO PERIODIK (lihat __init__/yolo_redetect_every) -- default 1 =
        # deteksi tiap frame, PERSIS perilaku lama. Bbox cache dipakai ulang
        # HANYA kalau deteksi terakhir berhasil (bbox_cache bukan None);
        # kalau gagal, coba lagi tiap frame (pulih cepat begitu orang kembali
        # terdeteksi) -- lihat komentar lengkap di __init__.
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
        results = self._mp_pose.process(image)  # SELALU dipanggil tiap frame (bbox
        # ada ATAU tidak) -- JANGAN di-skip, lihat komentar countdown di atas:
        # sempat dicoba MediaPipe idle saat tidak perlu, terbukti bikin tracker
        # internalnya reset lalu NaN menjalar berdetik-detik setelahnya.

        # BUG NYATA yg ditemukan dari laporan user (kamera diarahkan ke
        # langit-langit gym TANPA orang, tetap keluar prediksi+repetisi):
        # tanpa gerbang `bbox` di sini, hasil MediaPipe TETAP DIPERCAYA walau
        # YOLO sendiri TIDAK menemukan orang sama sekali -- MediaPipe kadang
        # "mengarang" pose dari tekstur non-manusia (rak besi, langit-langit)
        # yg angkanya kelihatan valid tapi FIKTIF, lolos ke rep counter &
        # classifier. Proposal bab 8.3.3 sendiri urutannya "YOLO deteksi
        # manusia DULU, BARU MediaPipe" -- jadi hasil MediaPipe SEHARUSNYA
        # cuma dipakai kalau YOLO sendiri yakin ada orang. Fix: MediaPipe
        # tetap DIPANGGIL (baris di atas, demi tracker tetap hangat), tapi
        # hasilnya dipaksa NaN di sini kalau `bbox` kosong -- gerbang di titik
        # PERCAYA, bukan titik PANGGIL.
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

        # Rep counting -- PERSIS bab 6.2.3 (prominence+distance) + Eq.4
        # (cosine similarity), tanpa mekanisme tambahan apapun. Masa
        # persiapan (in_countdown) TETAP dikecualikan dari rep_counter/buffer
        # window (perilaku exclude TIDAK berubah) -- yang berubah cuma YOLO+
        # MediaPipe di atas TETAP jalan selama countdown (lihat komentar di
        # atas), bukan bagian ini.
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
            # "crop_box": (x1,y1,x2,y2) SETELAH padding+clamp -- posisi persis
            # `display_frame` itu di dalam `frame_bgr` asli. Dipakai caller yg
            # mau nempel `display_frame` (yg sudah digambar skeleton) balik ke
            # frame utuh, jadi tampilan stabil ukuran video asli (bukan crop
            # yg ukurannya loncat-loncat tiap frame) -- lihat preview_live_pipeline.py.
            "crop_box": crop_box,
            # "display_frame": gambar yang landmark row-nya COCOK -- MediaPipe
            # kasih koordinat relatif ke gambar yg DIPROSES (crop, kalau ada
            # bbox), BUKAN relatif ke frame_bgr asli. Skeleton HARUS digambar
            # di gambar ini (persis pola Ko et al.: draw_landmarks ke
            # object_frame, lalu tampilkan object_frame-nya, BUKAN overlay ke
            # frame utuh -- beda ukuran/rasio, salah gambar kalau dipaksa).
            "display_frame": pose_input,
            "primary_angle": primary,
            "predicted_class": predicted_class,
            "confidence": confidence,  # keyakinan RF window ini (None kalau buffer belum penuh / window NaN)
            "rep_count": self.rep_counter.rep_count,
            "countdown_remaining": (self.countdown_sec - elapsed) if in_countdown else None,
        }

    def _predict_from_buffer(self):
        """Bangun 1 baris window fitur dari buffer -- HARUS identik strukturnya
        (nama+urutan kolom) dgn build_windows_from_runs() (sliding_window.py),
        supaya self.feat_cols (dari feature_config.json hasil build_dataset.py)
        selalu ketemu kolomnya. Coordinate DIRATA-RATAKAN per window
        (coord_mean_*, PRODUKSI -- lihat riwayat coord_mode di
        sliding_window.py: sempat di-flatten per frame, dikembalikan ke mean
        setelah flatten terbukti bikin RF didominasi koordinat mentah)."""
        window_row = {}
        for f, frame_feat in enumerate(self._buffer):
            for col in ANGLE_COLUMNS:
                window_row[f"{col}_f{f}"] = frame_feat[col]
        for name in POSE_LANDMARK_NAMES:
            for axis in ("x", "y", "z", "v"):
                col = f"{name}_{axis}"
                # Rata-rata SKIP NaN (persis pandas .mean() dipakai
                # build_windows_from_runs()) -- NaN cuma kalau SEMUA frame di
                # window ini kebetulan NaN utk titik itu (occlusion penuh).
                values = [frame_feat[col] for frame_feat in self._buffer]
                valid = [v for v in values if v == v]
                window_row[f"coord_mean_{col}"] = (sum(valid) / len(valid)) if valid else float("nan")
        # NaN di window (angle ATAU coordinate) -- window ini TIDAK reliable,
        # sama aturan build_windows_from_runs() (discard, bukan interpolasi).
        if any(v != v for v in window_row.values()):
            self._last_confidence = None
            return None

        X = pd.DataFrame([{c: window_row[c] for c in self.feat_cols}], columns=self.feat_cols)
        # predict_proba() (BUKAN predict() langsung) supaya bisa cek keyakinan
        # RF (fraksi pohon yg setuju ke kelas pemenang) sebelum dipakai --
        # predict() sendirian TIDAK PERNAH bilang "tidak yakin", selalu pilih
        # argmax walau menangnya cuma tipis (lihat class_confidence_threshold
        # di __init__). classes_ diambil dari step 'rf' pipeline (bukan dari
        # self.pipeline langsung -- itu Pipeline scaler+rf gabungan, tidak
        # punya .classes_ sendiri di semua versi sklearn)."""
        proba = self.pipeline.predict_proba(X)[0]
        rf_step = self.pipeline.named_steps["rf"]
        best_idx = proba.argmax()
        confidence = float(proba[best_idx])
        self._last_confidence = confidence  # diekspos via process_frame (lihat __init__)
        if confidence < self.class_confidence_threshold:
            return None  # RF "tidak cukup yakin" -- diperlakukan sama spt window blm penuh
        return rf_step.classes_[best_idx]
