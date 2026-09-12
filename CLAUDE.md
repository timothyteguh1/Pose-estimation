# CLAUDE.md — Konteks Proyek Skripsi

Baca `docs/project_context.md` di awal sesi ini SEBELUM menjawab atau menulis kode
apa pun — file itu berisi ringkasan proposal lengkap dan tabel perbedaan wajib
antara pendekatan referensi (Ko et al., 2024) vs implementasi skripsi ini.

PDF proposal skripsi asli (`docs/C14230133_Timothy Jonathan David Teguh - Copy.pdf`)
adalah rujukan yang MENGIKAT — kalau project_context.md dan PDF berbeda, PDF yang
menang. project_context.md sudah diselaraskan dengan PDF, tapi kalau ragu/PDF
direvisi lagi, baca ulang PDF-nya.

## Ringkasan sangat singkat
Skripsi: "Aplikasi Web Pemantau Postur Beban Bebas Dengan YOLOv11 dan MediaPipe di
Derich Fitness Gym" — Timothy Jonathan David Teguh (C14230133), UK Petra.

Referensi teknis: repo Ko et al. (`AI_Exercise_Pose_Feedback-main`, folder terpisah
di luar proyek ini, JANGAN diedit/dipakai langsung) — dipelajari untuk fondasi
teknis saja.

## Aturan keras (jangan sampai ketuker dengan pola referensi)
1. Unit input classifier = SLIDING WINDOW (beberapa frame -> 1 sampel), bukan
   per-frame tunggal seperti Ko et al.
2. Fitur utama = JOINT ANGLE (11 sudut), koordinat mentah hanya pendukung —
   kebalikan dari Ko et al. yang menempel angle scaled ke koordinat mentah.
3. Label 18 kelas granular (3 exercise x 3 postur x 2 fase konsentrik/eksentrik)
   ditentukan SEJAK AWAL saat labeling, jangan digabung belakangan.
4. Setiap sample/row wajib punya kolom metadata `camera_angle`
   (left45 / right45 / front) untuk keperluan split Model A (2 sudut) vs
   Model B (3 sudut, occlusion test).
5. Repetition counting dari pola naik-turun joint angle temporal, bukan dari
   substring label kelas (state machine) seperti Ko et al.
6. Deteksi objek pakai YOLOv11 (bukan YOLOv5).
7. Normalisasi Min-Max diterapkan konsisten ke fitur final (angle) sebelum
   training — bukan hanya ditempel parsial seperti kode referensi.

## Struktur folder
Lihat `README.md` di root untuk peta folder (`data/`, `src/`, `models/`,
`notebooks/`, `docs/`). Tiap subfolder punya `README.md` singkat sendiri.

## Status & urutan kerja
Ikuti urutan 8 tahap di `docs/project_context.md` bagian "Urutan kerja yang
disarankan". Saat ini masih tahap struktur folder awal — belum ada logic
implementasi. Jangan lompat ke tahap lanjut (training, app) sebelum tahap
sebelumnya (ekstraksi pose, joint angle, sliding window) selesai dan
dikonfirmasi.
