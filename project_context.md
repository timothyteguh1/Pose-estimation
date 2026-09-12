# Konteks Proyek: Aplikasi Web Pemantau Postur Beban Bebas (YOLOv11 + MediaPipe)

## Identitas
- Mahasiswa: Timothy Jonathan David Teguh (C14230133), Informatika, Universitas Kristen Petra
- Judul: Aplikasi Web Pemantau Postur Beban Bebas Dengan YOLOv11 dan MediaPipe di Derich Fitness Gym
- Mitra: Derich Fitness Gym

## Referensi utama (paper acuan)
Ko, Y-M., Nasridinov, A., Park, S-H. (2024). "Real-Time AI Posture Correction for
Powerlifting Exercises Using YOLOv5 and MediaPipe." IEEE Access.
https://ieeexplore.ieee.org/abstract/document/10798440
Repo asli penulis (dipelajari untuk fondasi teknis, BUKAN untuk dikopi mentah):
https://github.com/PSLeon24/AI_Exercise_Pose_Feedback

## Apa yang sudah dipelajari dari repo Ko et al.
- `labeling/*.ipynb`: labeling manual per-frame lewat keystroke (u=up, d=down) saat
  playback video, ekspor ke CSV 132 kolom (33 landmark MediaPipe x [x,y,z,visibility]).
- `Afterprocessing.ipynb`: menghitung 11 joint angle pakai `calculateAngle(a,b,c)`
  berbasis `arctan2` (bukan `arccos` dot-product seperti Eq.1 di proposal kita, tapi
  hasilnya ekuivalen secara matematis), lalu Min-Max scaling HANYA pada kolom angle,
  dan hasilnya DITEMPEL (concat) ke CSV koordinat mentah -> `*_with_scaled_angles.csv`.
- `models/{exercise}/*_classification[Angle].ipynb`: Random Forest (dan LR/Ridge/GB)
  dilatih dari gabungan koordinat mentah + angle scaled, split 70:30, evaluasi
  accuracy/precision/recall/F1 (weighted).
- `DeepLearning V1-V7*.ipynb`: FCNN, MLP-Mixer, Transformer (PyTorch) - dilatih
  LANGSUNG dari koordinat mentah (132 kolom), TIDAK pakai angle sebagai fitur.
- `Streamlit.py`: real-time app. YOLOv5 crop bbox person terbesar -> MediaPipe Pose
  pada crop -> hitung 11 sudut untuk ditampilkan -> classifier (`model_e.predict`)
  memprediksi kelas dari 132 kolom koordinat mentah PER FRAME TUNGGAL (bukan window)
  -> state machine sederhana: kelas mengandung "down" -> stage=down; lalu kelas
  mengandung "up" saat stage=down -> stage=up, counter+=1 -> cek most_frequent(class)
  utk feedback suara/teks. Bug kecil: `random` dipakai tapi tidak di-import di
  `Streamlit.py` (ada di `Streamlit_NoneYolo.py`).
- Kode ini SECARA EKSPLISIT membaca garis antar dua landmark sebagai VEKTOR GARIS
  LURUS, bukan kurva otot -- ini relevan untuk menjawab poin revisi dosen penguji
  soal "output skeleton, garis lurus atau kurva antara 2 titik".

## Perbedaan WAJIB antara proposal kita vs kode Ko et al. (jangan sampai ketuker)
| Aspek | Ko et al. (kode) | Proposal kita (yang harus dibangun) |
|---|---|---|
| Deteksi objek | YOLOv5 | YOLOv11 |
| Unit input classifier | per-frame tunggal | SLIDING WINDOW (beberapa frame -> 1 sampel temporal), rujuk Raza et al. 2025 |
| Fitur utama | koordinat mentah + angle ditempel | JOINT ANGLE sebagai fitur utama, joint coordinate pendukung |
| Repetition counting | state machine dari label kelas up/down | dari POLA PERUBAHAN JOINT ANGLE (naik/turun), bisa dibantu cosine similarity ala Hsu et al. 2025 |
| Kelas | 6 (bench)/8 (squat)/8 (deadlift), fase up/down terpisah per exercise | 18 kelas eksplisit: 3 exercise x 3 postur x 2 fase (konsentrik/eksentrik) sejak awal |
| Normalisasi | Min-Max hanya pada kolom angle, ditempel ke data mentah | Min-Max diterapkan konsisten ke fitur final (angle) sebelum training |
| Evaluasi tambahan | tidak ada | wajib: F1-score (utama), confusion matrix, MAE utk repetition counting, kuesioner Likert utk UI, perbandingan YOLOv5 vs YOLOv11, perbandingan 2-sudut vs +sudut depan (occlusion) |

## Ruang lingkup teknis (dari proposal, rangkuman)
- 3 exercise: Squat, Bench Press, Deadlift
- Input: video 1 kamera, sudut diagonal kiri/kanan 45 derajat (+ sisi depan utk
  skenario tambahan occlusion)

### PENTING - Skema pengujian 2 sudut vs 3 sudut (occlusion test)
- Setiap sesi rekam WAJIB menghasilkan 3 video: diagonal kiri 45, diagonal kanan 45,
  dan depan (frontal) -- meskipun model utama nanti hanya dilatih dari 2 sudut.
- Data depan tetap harus dikumpulkan dari awal supaya tersedia untuk skenario
  pengujian tambahan di bawah.
- Saat proses ekstraksi landmark & windowing, WAJIB ada kolom metadata
  `camera_angle` (nilai: left45 / right45 / front) yang menempel di setiap
  sample/row, supaya nanti gampang displit jadi dua skenario dataset:
  - Model A (2 sudut): hanya sample dgn camera_angle in [left45, right45]
  - Model B (3 sudut): semua sample [left45, right45, front]
- Kedua model dievaluasi dgn metrik yang sama (accuracy, precision, recall, F1,
  confusion matrix) utk melihat apakah menambah sudut depan signifikan mengurangi
  dampak occlusion atau tidak.
- Implikasi struktur folder data: nama file/folder video dan CSV hasil ekstraksi
  sebaiknya menyertakan tag sudut kamera sejak awal (misal:
  squat_p1_left45_rep03.mp4, squat_p1_front_rep03.mp4) supaya tidak perlu
  relabeling manual belakangan.
- Pipeline: YOLOv11 (deteksi manusia) -> MediaPipe (33 landmark) -> hitung 11 joint
  angle (neck, L/R shoulder, L/R elbow, L/R hip, L/R knee, L/R ankle) -> filter
  visibility >= 0.6 -> sliding window (Eq. 2: swl = fps x t; Eq. 3: swm = frames - swl + 1)
  -> Min-Max normalization -> Random Forest classification -> repetition counting dari
  siklus naik-turun joint angle
- Dataset target: 15.000-20.000 sampel temporal, split 70:30, dari >=3 partisipan
- Output: bounding box + skeleton, klasifikasi postur, teks+suara peringatan, hitung
  repetisi, ringkasan hasil setelah sesi
- Arsitektur: aplikasi web client-server

## Urutan kerja yang disarankan (mengikuti bagian 8.3 proposal)
1. Ekstraksi pose MediaPipe dari video -> simpan landmark mentah per frame,
   label granular ke 18 kelas SEJAK AWAL (jangan digabung nanti seperti Ko et al.)
2. Hitung 11 joint angle per frame (boleh reuse pola calculateAngle, sudah tervalidasi)
3. Bangun SLIDING WINDOW: kelompokkan N frame berurutan jadi 1 sampel (agregasi/
   flatten angle dari beberapa frame) -- INI BAGIAN YANG TIDAK ADA di kode Ko,
   harus dibangun dari nol sesuai Eq. 2 & 3 proposal
4. Normalisasi Min-Max pada fitur angle hasil window
5. Split 70:30, training Random Forest (bisa bandingkan dengan LR/Ridge/GB seperti
   Ko et al. utk pembanding, tapi laporkan F1-score sebagai indikator utama)
6. Repetition counting dari pola naik-turun joint angle temporal (bukan sekadar
   label kelas dari classifier), evaluasi pakai MAE vs ground truth manual
7. Integrasi ke aplikasi web real-time (boleh pakai kerangka Streamlit Ko sbg
   referensi struktur kode, tapi ganti YOLOv5->YOLOv11, ganti logic prediksi
   dari single-frame jadi window-based)
8. Pengujian: F1/confusion matrix (per-frame vs sliding window), MAE repetition
   counting, kuesioner Likert (>=15 responden, valid jika rata-rata >=4.0 atau
   >=80% positif), YOLOv5 vs YOLOv11 (waktu inferensi + akurasi deteksi),
   2 sudut diagonal vs +sudut depan (dampak occlusion)

## Status saat ini
- Sudah ada beberapa video latihan yang dikumpulkan.
- Belum ada kode implementasi sendiri -- baru tahap mempelajari kode Ko et al.
  sebagai referensi teknis dan menyusun rencana kerja.
- Revisi dosen penguji yang perlu dijawab di proposal/skripsi:
  1. Persamaan matematika diberi nomor (sudah diperbaiki di draf terbaru)
  2. Perhatikan output skeleton -- tegaskan bahwa garis antar titik dibaca sebagai
     vektor garis lurus (bukan kurva otot), ini keterbatasan yang disadari & konsisten
     dengan pendekatan MediaPipe+joint-angle di literatur (Ko et al. 2024, Qi et al. 2025)
  3. Perbaiki rumus normalisasi (pastikan Eq. 5 min-max scaling di proposal benar
     dan konsisten notasinya)
  4. Perjelas: apakah squat/bench/deadlift dilakukan bergantian atau tidak, dan
     kalau data dicampur, jelaskan alasannya (proposal saat ini: data dipisah
     per jenis gerakan spesifik, bukan campuran bergantian)
  5. Perjelas metodologi split dataset 70:30 (jumlah data, variasi background,
     variasi partisipan)
  6. Masukkan detail pengujian ke ruang lingkup sesuai Table 4 paper acuan
     (klasifikasi postur x fase kontraksi otot)
  7. Tonjolkan research gap "sliding window" dan jadikan acuan pengujian dibanding
     pendekatan per-frame
