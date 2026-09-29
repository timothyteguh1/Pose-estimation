# Skripsi: Aplikasi Web Pemantau Postur Beban Bebas (YOLOv11 + MediaPipe)

Timothy Jonathan David Teguh (C14230133) — Informatika, Universitas Kristen Petra
Mitra: Derich Fitness Gym

Aplikasi web yang menilai postur squat, bench press, dan deadlift dari video atau
kamera, menghitung repetisi, lalu memberi peringatan teks + suara kalau postur salah.

Sistem yang dipakai sekarang ada di folder `experimental_client_pipeline/` (arsitektur baru).
Repo referensi Ko et al. (2024) hanya dipelajari, tidak dikopi.
Ringkasan lengkap ada di `docs/project_context.md`.

## Peta folder

- `data/raw_videos/` — video mentah (`{squat,benchpress,deadlift}/pN/{postur}/`) dan `testing/` untuk video uji
- `experimental_client_pipeline/` — **sistem yang aktif**
  - `scripts/` — `extract_landmarks.py`, `migrate_labels.py`, `build_dataset.py`, `train_model.py`
  - `data/extracted_landmarks/` — hasil ekstraksi pose (CSV)
  - `data/labeled/` — CSV yang sudah dilabel fase (dipakai untuk bikin dataset)
  - `data/windowed_features/` — dataset hasil sliding window (train dan test)
  - `models/` — model hasil training (`*_rf.pkl`), `*_feature_config.json`, `*_evaluation.json`
  - `webapp/app.py` — server (halaman web + WebSocket, satu port)
  - `webapp/frontend/` — `index.html`, `yolo.js`, `crop.js`, model YOLO dan MediaPipe untuk browser
  - `webapp/backend/landmark_pipeline.py` — otak di server: sudut, window, klasifikasi, hitung repetisi
- `src/` — kode yang dipakai bareng: `extraction/`, `features/`, `training/`, `detection/`, `app/` (hanya `rep_counter.py` dan `warnings_content.py` yang dipakai)
- `models/yolo11n.pt` — model YOLO untuk ekstraksi (dipakai script ekstraksi, jangan dihapus)
- `docs/` — proposal, referensi, catatan

## Cara kerja singkat

**Offline (training):**
video mentah -> ekstrak pose (YOLO crop + MediaPipe Tasks, 33 titik) -> label fase naik/turun ->
hitung 11 sudut sendi -> bagi train/test (target 70:30) -> normalisasi Min-Max pada sudut
(dihitung dari data train) -> sliding window 0,5 detik (15 frame) -> koordinat relatif ->
flip horizontal (khusus data train) -> latih Random Forest (1 model per latihan).

**Live (web):**
- Browser: YOLO11n cari orang -> crop -> MediaPipe ambil 33 titik -> titik dikirim ke server lewat WebSocket.
- Server: hitung sudut -> kumpulkan 15 frame -> koordinat relatif -> Random Forest klasifikasi ->
  hitung repetisi -> nilai tiap siklus -> kirim hasil balik ke browser.

## Setelan produksi

Nilai default di script **tidak sama** dengan nilai produksi. Tanpa flag di bawah, model yang
keluar bukan model yang dipakai aplikasi.

| Setelan | Nilai | Ditetapkan di mana |
|---|---|---|
| Panjang window | 0,5 detik (15 frame) | `build_dataset.py --window-sec 0.5` (default script 1.0) |
| Koordinat relatif | aktif | `build_dataset.py --coord-relative` |
| Flip horizontal | aktif (train saja) | `build_dataset.py --flip-augment` |
| `min_samples_leaf` Random Forest | 4 | `train_model.py --min-samples-leaf 4` (default script 1) |
| `n_estimators` | 100 | default script |
| `class_weight`, `random_state` | balanced, 42 | tertulis di kode |
| Batas visibility titik | 0,6 | default di `joint_angles.py` |

File `*_feature_config.json` dan `*_evaluation.json` di folder `models/` hanyalah catatan
nilai yang dipakai saat training. Nilai aslinya ditentukan lewat command di bawah.

## Command dari awal sampai akhir

Semua command dijalankan dari root folder project (`skripsi-postur-gym`), pakai PowerShell.

### 0. Persiapan (sekali saja)

```powershell
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m pip install websockets
```

`websockets` dipakai server web arsitektur baru, belum tercatat di `requirements.txt`.

### 1. Ekstrak pose dari video

```powershell
venv\Scripts\python.exe experimental_client_pipeline\scripts\extract_landmarks.py data\raw_videos\squat
venv\Scripts\python.exe experimental_client_pipeline\scripts\extract_landmarks.py data\raw_videos\benchpress
venv\Scripts\python.exe experimental_client_pipeline\scripts\extract_landmarks.py data\raw_videos\deadlift
```

Hasil masuk ke `experimental_client_pipeline\data\extracted_landmarks\`. CSV yang sudah ada
dilewati. Tambah `--force` kalau mau ekstrak ulang.
Nama file video harus mengikuti pola `{latihan}_{postur}_p{N}_{sudut}_take01.mp4`.

### 2. Label fase naik/turun (manual)

Untuk video P1-P3, label lama sudah digabung ke data baru. Tidak perlu diulang, kecuali habis ekstrak ulang:

```powershell
venv\Scripts\python.exe experimental_client_pipeline\scripts\migrate_labels.py data\raw_videos\squat data\raw_videos\benchpress data\raw_videos\deadlift
```

Untuk video baru, label manual pakai satu CSV atau satu folder:

```powershell
venv\Scripts\python.exe src\extraction\label_phase.py experimental_client_pipeline\data\extracted_landmarks\squat\p4\correct\squat_correct_p4_left45_take01.csv
venv\Scripts\python.exe src\extraction\label_batch.py experimental_client_pipeline\data\extracted_landmarks\squat\p4\correct
```

Kontrol saat playback: `space` play/pause (pertama kali = tandai mulai gerakan), `u`/`d` tandai
naik/turun, `e` buang, `r` ikut auto, `,` `.` mundur/maju 1 frame, `s` simpan, `q` simpan lalu keluar.

Setelah semua CSV di folder itu selesai dilabel, salin ke folder `labeled` (struktur folder sama):

```powershell
Copy-Item experimental_client_pipeline\data\extracted_landmarks\squat\p4 -Destination experimental_client_pipeline\data\labeled\squat\p4 -Recurse
```

### 3. Bikin dataset

```powershell
venv\Scripts\python.exe experimental_client_pipeline\scripts\build_dataset.py squat --window-sec 0.5 --coord-relative --flip-augment
venv\Scripts\python.exe experimental_client_pipeline\scripts\build_dataset.py benchpress --window-sec 0.5 --coord-relative --flip-augment
venv\Scripts\python.exe experimental_client_pipeline\scripts\build_dataset.py deadlift --window-sec 0.5 --coord-relative --flip-augment
```

### 4. Latih model

```powershell
venv\Scripts\python.exe experimental_client_pipeline\scripts\train_model.py squat --min-samples-leaf 4
venv\Scripts\python.exe experimental_client_pipeline\scripts\train_model.py benchpress --min-samples-leaf 4
venv\Scripts\python.exe experimental_client_pipeline\scripts\train_model.py deadlift --min-samples-leaf 4
```

Hasil di `experimental_client_pipeline\models\`: `*_rf.pkl`, `*_evaluation.json`, `*_confusion_matrix.png`.
Tiap kali training, jalankan `build_dataset.py` dulu. Training menghapus file scaler sementara,
jadi kalau dijalankan dua kali berturut-turut tanpa build ulang akan error.

### 5. Jalankan aplikasi web

```powershell
venv\Scripts\python.exe experimental_client_pipeline\webapp\app.py
```

Lalu buka browser ke http://localhost:8600

Model dibaca ulang setiap klik "Mulai Analisis", jadi setelah training ulang server tidak perlu
dimatikan. Stop server dengan `Ctrl+C`.

### 6. Akses dari luar jaringan (misalnya HP)

Buka terminal KEDUA, biarkan terminal langkah 5 tetap jalan:

```powershell
.\cloudflared.exe tunnel --url http://localhost:8600
```

Akan muncul link `https://xxxx.trycloudflare.com`. Link berubah tiap kali dijalankan ulang, dan mati
kalau salah satu terminal ditutup.

### 7. Tes pakai video

Di halaman web pilih mode "Upload Video", pilih latihan, pilih file (misalnya dari
`data\raw_videos\testing\`), lalu klik "Mulai Analisis". Hasil (jumlah repetisi dan kelas tiap siklus)
muncul di halaman setelah selesai.

## Kalau ada data baru (misalnya P4)

Urutannya: langkah 1 -> langkah 2 (label + salin ke `labeled`) -> langkah 3 -> langkah 4 -> langkah 5.
Angka jumlah sampel di laporan berubah kalau data baru ikut dilatih.

## Catatan

- Arsitektur lama (Streamlit: `app.py` di root dan `src/app/live_pipeline.py`) masih ada sebagai riwayat
  percobaan dan sumber beberapa fungsi. Tidak dipakai lagi.
- Kalau `import` error soal DLL `onnxruntime` di Windows saat menjalankan script tambahan, tambahkan
  `import torch` di baris paling atas script itu.
