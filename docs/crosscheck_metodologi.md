# Cross-Check Metodologi: Proposal vs Ko et al. vs Referensi Lain vs Implementasi Sistem

> Dokumen ini disusun untuk verifikasi independen (mis. lewat NotebookLM, dibandingkan
> langsung ke file PDF proposal & referensi asli). Setiap baris menyebut lokasi kode
> persis (`file:baris`) supaya bisa dicek langsung ke source code sebenarnya.
>
> Status per item: ✅ Sesuai proposal | 🔧 Tambahan implementasi kita (disclosed) | ⏳ Placeholder/belum lengkap

## 0. Referensi yang dipakai proposal (bab 7 tinjauan pustaka)

| Referensi | Kontribusi ke proyek ini |
|---|---|
| **Ko et al. (2024)** — *Real-Time AI Posture Correction for Powerlifting Exercises Using YOLOv5 and MediaPipe* | Fondasi teknis utama: YOLOv5+MediaPipe+RF, 11 joint angle, MPED dataset (18 kelas: 6 benchpress + 8 squat + 8 deadlift), Streamlit app. Proyek ini SENGAJA beda di: YOLOv11 (bukan v5), sliding window (bukan per-frame), fitur utama angle (bukan koordinat), rep counting dari joint angle (bukan state-machine substring kelas) |
| **Raza et al. (2025)** | Basis format sliding window `(1, swl, n)` -- window dibawa utuh, bukan diringkas |
| **Hsu et al. (2024)** — *Viewpoint-invariant exercise repetition counting* | Basis konsep "pola gerakan berulang dari data skeleton" utk repetition counting; definisi metrik **MAE of the count** (dipakai sbg pembanding, lihat bagian 9) |
| **Kongchansawang et al. (2025)** | Justifikasi MediaPipe (bukan YOLOv11) utk ekstraksi landmark tubuh -- YOLOv11 disebut "lacked several keypoints" |
| **Farhani et al. (2022)** | Justifikasi pemilihan Random Forest (akurasi >90%, robust thd noise) |
| **Qi et al. (2025)** — *A Single-Camera System* | Justifikasi arsitektur 1 kamera |

---

## 1. Deteksi Objek Manusia (YOLO)

**Proposal (ruang lingkup #3a, #9):** "Proses deteksi objek manusia menggunakan YOLOv11" + "Implementasi sistem tidak membangun model pose estimation dari awal, tetapi menggunakan model **pre-trained** YOLOv11... "

**Ko et al.:** YOLOv5, di-**fine-tune ulang** pakai >4800 gambar tambahan (2 varian: margin minim vs margin cukup -- Table 15, mAP@0.5-0.95 0.843 vs 0.705, TAPI margin cukup malah lebih baik utk akurasi landmark).

**Implementasi kita:**
| Item | Status | Lokasi |
|---|---|---|
| YOLOv11 (bukan v5) | ✅ | `src/detection/yolo_detector.py:28` (`yolo11n.pt`) |
| Pre-trained SAJA, tidak di-training ulang | ✅ | `yolo_detector.py:43-46` (load bobot resmi Ultralytics langsung) |
| Confidence threshold 0.7 | 🔧 pola Ko et al (`Streamlit.py: if conf>=0.7`) | `yolo_detector.py:43` |
| Cuma kelas 'person', ambil confidence tertinggi (1 orang) | ✅ ruang lingkup #17 | `yolo_detector.py:49-67` |
| `padding_ratio=0.75` (crop margin) | 🔧 kita sendiri, PRINSIP divalidasi Table 15 Ko et al (mekanisme beda: algoritmik vs retrain, krn proposal larang retrain) | `yolo_detector.py:70-121` |
| `match_frame_aspect` (koreksi rasio crop) | 🔧 NOVELTY kita, tidak ada di Ko et al | `yolo_detector.py:128-181` |

---

## 2. Ekstraksi Pose (MediaPipe)

**Proposal (ruang lingkup #3b, #9):** "Proses ekstraksi koordinat titik persendian menggunakan MediaPipe" + pre-trained saja.

**Kongchansawang et al.:** Alasan MediaPipe dipilih drpd YOLOv11 utk landmark ("YOLOv11 model lacked several... keypoints").

**Implementasi kita:**
| Item | Status | Lokasi |
|---|---|---|
| 33 landmark MediaPipe Pose, pre-trained | ✅ | `src/extraction/extract_landmarks.py` |
| TRAINING: MediaPipe di FRAME UTUH (tanpa YOLO) | ✅ desain sendiri (training tidak butuh crop) | `extract_landmarks.py` |
| LIVE: MediaPipe di frame HASIL CROP YOLO | ✅ bab 8.3.3 (urutan YOLO dulu baru MediaPipe) | `src/app/live_pipeline.py:171-183` |
| Fix rotasi video (portrait/landscape metadata) | 🔧 murni implementasi, ditemukan lewat pengujian data 2 HP berbeda | `src/io_utils.py` (`open_video_capture`, `CAP_PROP_ORIENTATION_AUTO`) |

---

## 3. Fitur Joint Angle

**Proposal (bab 6.2.1, Eq.1):** "Pengolahan data pose menjadi fitur sudut antar sendi" -- formula arccos dot-product, filter visibility (bab 8.3.1: "misalnya di bawah 0.6" diabaikan).

**Ko et al.:** 11 joint angle yang sama (Neck, L/R Shoulder/Elbow/Hip/Knee/Ankle), formula arctan2 (proposal pilih arccos, matematis setara).

**Implementasi kita:**
| Item | Status | Lokasi |
|---|---|---|
| Formula Eq.1 (arccos) | ✅ | `src/features/joint_angles.py:57-68` |
| 11 sudut (sama Ko et al) | ✅ | `joint_angles.py:28-44` |
| Visibility threshold 0.6 | ✅ proposal sendiri + Ko et al Eq.2 | `joint_angles.py:23` |
| Filter PER-TITIK (bukan per-frame all-or-nothing) | ✅ kutipan literal proposal bab 6.2.1 | `joint_angles.py:79-100` |
| **Angle = fitur UTAMA, koordinat = fitur PENDUKUNG** | ✅ **PERBEDAAN UTAMA dari Ko et al** (mereka tempel angle scaled ke koordinat mentah) | `src/features/sliding_window.py:31-45` |

---

## 4. Labeling & Fase Konsentrik/Eksentrik

**Proposal (ruang lingkup #4d, bab 8.2):** 18 kelas (3 exercise x 3 postur x 2 fase) ditentukan SEJAK AWAL. "Concentric (gerakan naik)" / "Eccentric (gerakan turun)".

**Ko et al.:** Definisi sama -- "eccentric contraction... muscle lengthens... concentric... muscle shortens", "eccentric (down) and concentric (up)" (dicek cross-reference LANGSUNG ke teks paper, dikonfirmasi konsisten dgn kode kita).

**Implementasi kita:**
| Item | Status | Lokasi |
|---|---|---|
| 18 kelas granular sejak awal labeling (bukan digabung belakangan) | ✅ | `src/extraction/filename_parser.py` (posture_class), `label_phase.py` (phase) |
| Auto-detect titik balik (draft) + review manual | ✅ bab 8.3.1 (manual review) | `src/extraction/label_phase.py` (`find_turning_points`, `run_interactive_review`) |
| `camera_angle` metadata wajib per sample | ✅ ruang lingkup #4 (kebutuhan split Model A/B) | `filename_parser.py`, kolom `camera_angle` di semua CSV |
| Kelas benchpress: "flat back" (bukan "back arch") | 🔧 revisi PT-consultation, DIDOKUMENTASIKAN bab 8.2 (excessive arch jarang ditemui di lapangan) | `docs/project_context.md:89-96` |

---

## 5. Sliding Window

**Proposal (bab 6.2.2, Eq.2 & 3):** `swl = fps × t`, `swm = frame - swl + 1`. **Bagian ini TIDAK ADA di kode Ko et al** (dibangun dari nol).

**Raza et al. (2025) bagian 3.2:** Format `(1, swl, n)` -- window dibawa utuh (tidak diringkas).

**Implementasi kita:**
| Item | Status | Lokasi |
|---|---|---|
| Formula Eq.2/3 persis | ✅ | `src/features/sliding_window.py:154-222` (offline), `live_pipeline.py:131` (live, IDENTIK) |
| Format window (1,swl,n), flatten bukan rata-rata | ✅ sesuai Raza et al 3.2 | `sliding_window.py:180-184` |
| `window_sec` (nilai "t") | 🔧 **0.5 detik, disamakan ke-3 exercise** -- kita sendiri, proposal TIDAK tetapkan angka "t" | `models/*_feature_config.json`. **BARU dari 1-2 partisipan, WAJIB divalidasi ulang** |
| `stride=1` | 🔧 kita sendiri, tidak eksplisit dari referensi manapun | `sliding_window.py` |
| Window TIDAK boleh lompat 'excluded'/beda fase | ✅ desain sendiri, konsisten Ko et al (discard bukan interpolasi) | `sliding_window.py:90-115` |

---

## 6. Normalisasi Data

**Proposal (bab 6.2.4, Eq.5):** Min-Max scaling, `X_norm = (X-min)/(max-min)`, method dari Ko et al.

**Ko et al.:** Min-Max juga, TAPI ditempel PARSIAL ke koordinat mentah (angle scaled + coordinate mentah digabung tanpa aturan jelas).

**Implementasi kita:**
| Item | Status | Lokasi |
|---|---|---|
| Formula Eq.5 | ✅ | `src/features/window_scaler.py:79-91` |
| **HANYA kolom ANGLE yang dinormalisasi** (coordinate TIDAK) | ✅ **PERBEDAAN UTAMA dari Ko et al** -- konsisten diterapkan (bab 6.2.4: "joint angle... dinormalisasi") | `window_scaler.py` (cek `columns=ANGLE_COLUMNS`) |
| Fit HANYA dari data TRAIN (cegah leakage) | ✅ | `src/features/build_dataset.py` (`scaler.fit_frame(train_frame_angles)`) |
| Normalisasi di level FRAME, SEBELUM windowing | ✅ urutan sesuai bab 8.3.1 | `build_dataset.py` |

---

## 7. Split Data Train/Test

**Proposal (bab 8.2):** 70:30, stratified per kelas, dataset bersih target 15.000-20.000 sampel, ≥3 partisipan.

**Implementasi kita:**
| Item | Status | Lokasi |
|---|---|---|
| Rasio 70:30 | ✅ | `build_dataset.py` (target `test_size=0.3`) |
| Split PER-RUN (1 repetisi utuh ke 1 sisi, tidak dipotong) | 🔧 cara TEKNIS kita capai rasio, proposal tidak atur ini secara spesifik. **Terbukti TIDAK ADA leakage** (window tidak pernah lintas batas run) | `build_dataset.py` (`allocate_runs_balanced`, `group_fragmented_runs`) |
| Total sampel saat ini | ⏳ **13.380 / target 15.000-20.000** (89% dari batas bawah) | `data/windowed_features/splits/` |
| ≥3 partisipan | ⏳ **BARU 1-2 partisipan per exercise** (squat: 1, benchpress: 2, deadlift: 2) | - |

---

## 8. Training Model (Random Forest)

**Proposal (bab 8.3.2):** "Model yang digunakan... adalah Random Forest" -- TIDAK ada perbandingan model lain (beda dari Ko et al yang bandingkan RF/LR/Ridge/GB).

**Ko et al.:** `n_estimators=100`, `max_depth=None`.

**Farhani et al. (2022):** Justifikasi RF (akurasi >90%, robust thd noise).

**Implementasi kita:**
| Item | Status | Lokasi |
|---|---|---|
| RF saja, tanpa model pembanding | ✅ | `src/training/train_model.py` |
| `n_estimators=100`, `max_depth=None` | ✅ sama Ko et al | `train_model.py:127-131` |
| `class_weight="balanced"` | 🔧 **tambahan kita**, bukan dari proposal/Ko et al -- jaring pengaman sementara (solusi idealnya bab 8.2: "usahakan seimbang" via data collection). **Diuji empiris: efeknya konsisten membantu tapi kecil di data saat ini** | `train_model.py:129` |
| 1 model per exercise (3 model terpisah) | 🔧 keputusan arsitektur kita | `models/{exercise}_rf.pkl` |

---

## 9. Evaluasi

**Proposal (bab 8.4):** Accuracy/Precision/Recall/F1 (F1 = indikator utama), confusion matrix, perbandingan per-frame vs sliding window, **MAE** untuk repetition counting, kuesioner Likert ≥15 responden.

**Implementasi kita:**
| Item | Status | Lokasi |
|---|---|---|
| F1-score sbg indikator utama | ✅ | `train_model.py:145` |
| Confusion matrix | ✅ | `train_model.py` (`plot_confusion_matrix`) |
| **MAE standar** (proposal, `(1/n)Σ\|pred-GT\|`) | ✅ **rumus YANG DIPAKAI RESMI** | Dihitung manual per exercise, hasil: squat=1.00, benchpress=1.00-1.50, deadlift=0.50-1.25 (tuning/held-out) |
| MAE ala Hsu et al. ("MAE of the count", `(1/n)Σ\|pred-GT\|/GT`) | 🔧 **TAMBAHAN pembanding saja** (Hsu et al. sendiri menyebutnya begini di papernya, dikutip di proposal) -- BUKAN pengganti MAE resmi | Hasil: squat=0.167, benchpress=0.139-0.194, deadlift=0.083-0.213 (banding Hsu et al: 0.06-0.07) |
| Perbandingan YOLOv5 vs YOLOv11 | ⏳ **BELUM DIKERJAKAN** | - |
| Perbandingan per-frame vs sliding window | ⏳ **BELUM DIKERJAKAN** | - |
| Perbandingan 2-sudut vs +sudut depan (occlusion, Model A/B) | ⏳ **BELUM DIKERJAKAN** (metadata `camera_angle` sudah siap) | - |
| Kuesioner Likert ≥15 responden | ⏳ **BELUM DIMULAI** -- perlu app sudah online | - |

---

## 10. Repetition Counting

**Proposal (bab 6.2.3):** "Satu repetisi = satu siklus lengkap dari fase minimum ke maksimum dan kembali ke minimum" -- deteksi dari perubahan joint angle, DIBANTU cosine similarity (Eq.4) utk validasi kemiripan pola SEBELUM hitung repetisi.

**Ko et al.:** State machine dari SUBSTRING NAMA KELAS ("down"/"up") -- **BEDA TOTAL** dari pendekatan proposal (joint angle temporal).

**Implementasi kita:**
| Item | Status | Lokasi |
|---|---|---|
| Deteksi dari pola naik-turun joint angle (bukan substring kelas) | ✅ **PERBEDAAN UTAMA dari Ko et al** | `src/app/rep_counter.py` (`OnlineRepCounter`) |
| Peak+valley detection (prominence+distance) | ✅ konsep umum, angka dari data kita sendiri | `rep_counter.py:189-210` |
| Cosine similarity Eq.4 (gerbang validasi pola) | ✅ formula, 🔧 objek A/B (puncak vs template pertama sesi itu) adalah INTERPRETASI kita dari kalimat proposal yg generik ("dua frame berbeda") | `rep_counter.py:59-72, 220-235` |
| Parameter prominence/distance per exercise | 🔧 kita sendiri, ditemukan dari analisis data kita. Squat: 20°/0.6s (1 partisipan). Benchpress: 28°/0.6s (2 partisipan). Deadlift: 25°/1.0s (2 partisipan) | `rep_counter.py` (`REP_COUNTING_PARAMS`) |
| Ambang cosine 0.70 | 🔧 kita sendiri. **DIUJI dgn data asli (simulasi & video nyata): TIDAK efektif menolak gerakan asimetris (1 tangan/kaki naik) krn dilusi vektor 11-dimensi** -- didokumentasikan sbg keterbatasan | `rep_counter.py:134` |
| Motion onset detection | ❌ **DIHAPUS TOTAL** (pernah dicoba, terbukti bukan bagian proposal, tidak ada jejak kode) | - |

---

## 11. Aplikasi Web (Implementasi)

**Proposal (bab 6.4, 8.3.3, ruang lingkup #15-16):** Client-server, server proses semua (YOLO+MediaPipe+RF+rep counting), client cuma render. Output: bbox+skeleton, klasifikasi, teks+suara peringatan, jumlah repetisi, ringkasan sesi.

**Ko et al.:** Streamlit + streamlit-webrtc, `st.error()` + `pygame.mixer` audio pre-recorded (BUKAN TTS live).

**Implementasi kita:**
| Item | Status | Lokasi |
|---|---|---|
| Client-server (server proses semua) | ✅ | `app.py` + `src/app/live_pipeline.py` |
| Streamlit + streamlit-webrtc | ✅ pola Ko et al (framework tidak diatur proposal) | `app.py` |
| Bbox+skeleton, klasifikasi, repetisi, ringkasan | ✅ | `app.py` |
| Teks+audio peringatan (mp3 pre-recorded, gTTS sekali generate) | ✅ pola Ko et al | `src/app/warnings_content.py`, `scripts/generate_warning_audio.py` |
| Deploy ke cloud (Streamlit Community Cloud) | ⏳ **BELUM DIMULAI** -- git repo juga belum di-setup | - |
| Uji performa di hosting nyata | ⏳ **BELUM DILAKUKAN** | - |

---

## Ringkasan Status Keseluruhan

- ✅ **Metodologi INTI (formula, alur, fitur utama) 100% sesuai proposal**, semua perbedaan dari Ko et al. adalah perbedaan yang MEMANG diminta proposal (YOLOv11, sliding window, angle sbg fitur utama, rep counting dari joint angle).
- 🔧 Semua tambahan implementasi KITA SENDIRI (di luar angka eksak yg proposal tentukan) **sudah didisclose** dengan alasan di komentar kode masing-masing, TIDAK ada yang disembunyikan.
- ⏳ Yang **masih pending** (bukan salah, tapi belum lengkap): partisipan (baru 1-2 dari target ≥3), 2 perbandingan wajib (YOLOv5 vs v11, per-frame vs window), kuesioner Likert, deployment cloud, validasi ulang parameter rep-counting begitu data lengkap.
