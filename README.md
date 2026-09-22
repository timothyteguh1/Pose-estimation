# Skripsi: Aplikasi Web Pemantau Postur Beban Bebas (YOLOv11 + MediaPipe)

Timothy Jonathan David Teguh (C14230133) — Informatika, Universitas Kristen Petra
Mitra: Derich Fitness Gym

Implementasi ini terpisah total dari repo referensi `AI_Exercise_Pose_Feedback-main`
(Ko et al., 2024). Repo referensi dipelajari sebagai fondasi teknis, bukan dikopi
mentah. Lihat `docs/project_context.md` untuk ringkasan lengkap perbedaan wajib
antara pendekatan referensi vs implementasi skripsi ini (sliding window, joint angle
sebagai fitur utama, 18 kelas granular, YOLOv11, dsb).

## Struktur folder

- `data/raw_videos/{squat,benchpress,deadlift}/` — video mentah per exercise, per
  partisipan, per sudut kamera (left45/right45/front)
- `data/extracted_landmarks/` — hasil ekstraksi MediaPipe per frame (CSV)
- `data/windowed_features/` — hasil sliding window + joint angle + normalisasi Min-Max
- `src/extraction/` — script ekstraksi pose MediaPipe dari video
- `src/features/` — perhitungan joint angle, sliding window, normalisasi
- `src/training/` — training Random Forest (+ pembanding LR/Ridge/GB) dan evaluasi
- `src/detection/` — wrapper YOLOv11 untuk deteksi person
- `src/app/` — aplikasi web real-time (client-server)
- `models/` — model hasil training (.pkl, .pt)
- `notebooks/` — eksplorasi/eksperimen (opsional)
- `docs/` — proposal, catatan revisi dosen, project_context.md

## Status

Aplikasi web sudah jalan (`app.py`, mode Kamera Live & Upload Video). Lihat bagian
"Menjalankan Aplikasi" di bawah.

## Menjalankan Aplikasi

Semua command di bawah dijalankan dari root folder project (`skripsi-postur-gym`),
pakai PowerShell.

### 1. Jalankan server lokal (WAJIB, langkah pertama semua mode)

```powershell
venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

Local URL:   http://localhost:8501
Network URL: http://192.168.x.x:8501     lagi konek, tidak perlu diset manual


**Kalau muncul error "Port 8501 is not available"** — berarti ada proses lain
(termasuk punya Claude kalau sedang bantu debug bareng) yang masih pakai port itu.
Ganti angka port di command (mis. `--server.port 8502`) atau matikan dulu proses lamanya.

Untuk STOP server: tekan `Ctrl+C` di terminal itu.

### 2. Buka akses dari internet (bukan cuma 1 jaringan lokal) — Cloudflare Tunnel

Perlu file `cloudflared.exe` (sudah ada di root project). Jalankan di terminal
**KEDUA** (biarkan terminal langkah 1 tetap jalan):

```powershell
.\cloudflared.exe tunnel --url http://localhost:8501
```

Tunggu beberapa detik, akan muncul 1 URL `https://xxxxx-xxxx-xxxx-xxxx.trycloudflare.com`
— itu yang dibagikan ke penguji di luar kota/pulau, bisa diakses dari jaringan
manapun (bukan cuma 1 wifi yang sama).

**Penting**:
- Setiap kali `cloudflared` dijalankan ulang, URL yang keluar **BEDA/acak** (versi
  gratis tanpa akun) — jangan simpan 1 link dan pakai selamanya, selalu ambil link
  TERBARU tiap sesi baru.
- Kalau salah satu dari 2 terminal (langkah 1 atau 2) ditutup/`Ctrl+C`, link mati.
  Keduanya harus tetap terbuka selama sesi pengujian berlangsung.
- Mode Upload Video jalan normal lewat link ini. Mode Kamera Live ikut jalan
  untuk akses HALAMAN webnya, tapi video call WebRTC-nya sendiri baru andal
  lintas jaringan kalau TURN server sudah dipasang (lihat diskusi proyek —
  belum dikerjakan, perlu daftar akun Metered.ca dulu).

### Ringkasan skenario

| Skenario | Command yang perlu jalan | Link yang dipakai |
|---|---|---|
| Tes sendiri di laptop | Langkah 1 saja | `Local URL` |
| Tes dari HP sendiri, 1 wifi/hotspot yang sama | Langkah 1 saja | `Network URL` |
| Penguji di luar kota/pulau, jaringan beda | Langkah 1 + Langkah 2 | URL `trycloudflare.com` |

## Labeling Video Baru

Video mentah baru harus diekstrak lalu dilabel manual (fase konsentrik/eksentrik)
sebelum bisa dipakai `build_dataset.py`. Labeling WAJIB manual (nonton video +
tekan tombol) -- tidak bisa diotomasi.

### 1. Ekstraksi (skip kalau CSV-nya sudah ada)

```powershell
venv\Scripts\python.exe src\extraction\extract_landmarks.py data\raw_videos\squat\p3 --use-yolo-crop
venv\Scripts\python.exe src\extraction\extract_landmarks.py data\raw_videos\benchpress\p3\flatback --use-yolo-crop
```

*(Status saat ini: squat P3 sudah diekstrak. Benchpress P3 flatback juga sudah.)*

### 2. Labeling manual

`label_batch.py` menjalankan `label_phase.py` untuk banyak CSV berurutan, tapi
**tidak baca subfolder rekursif** -- kalau videonya nested per posture_class
(seperti P3), tiap subfolder harus disebut terpisah:

```powershell
venv\Scripts\python.exe src\extraction\label_batch.py data\extracted_landmarks\squat\p3\backbend data\extracted_landmarks\squat\p3\correct data\extracted_landmarks\squat\p3\kneeinward
```

Sisa benchpress P3 flatback yang belum dilabel (cuma left45 -- front & right45 sudah):

```powershell
venv\Scripts\python.exe src\extraction\label_phase.py data\extracted_landmarks\benchpress\p3\flatback\benchpress_flatback_p3_left45_take01.csv
```

Kontrol saat playback: `space`=play/pause (pertama kali = tandai mulai gerakan),
`u`/`d`=tandai konsentrik/eksentrik, `e`=exclude, `r`=ikut auto-detect,
`,`/`.`=mundur/maju 1 frame, `s`=simpan, `q`=simpan & keluar.

### 3. Rebuild dataset + retrain (setelah SEMUA video di atas selesai dilabel)

```powershell
venv\Scripts\python.exe src\features\build_dataset.py squat
venv\Scripts\python.exe src\features\build_dataset.py benchpress
venv\Scripts\python.exe src\training\train_model.py squat
venv\Scripts\python.exe src\training\train_model.py benchpress
```
### jalankan cmd utk arsitektur baru
venv\Scripts\python.exe experimental_client_pipeline\webapp\app.py
Setelah itu buka browser ke:
http://localhost:8600

### untuk di akses di luar network
cd C:\Users\totit\Downloads\skripsi-postur-gym
.\cloudflared.exe tunnel --url http://localhost:8600
