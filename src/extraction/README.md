# src/extraction/

Tahap 1 dari urutan kerja (bagian 8.3 proposal). Implementasi:

- `filename_parser.py` — parse nama file raw video
  (`{exercise}_{posture_class}_p{id}_{camera_angle}_take{NN}.mp4`) jadi
  metadata dict. `exercise`, `posture_class`, `participant_id`,
  `camera_angle`, `take` semuanya di-parse otomatis dari nama file (posture
  class SUDAH pasti dari nama file karena 1 video = 1 postur konsisten).
- `extract_landmarks.py` — jalankan MediaPipe Pose per frame dari video di
  `data/raw_videos/` -> CSV mentah per frame di `data/extracted_landmarks/`
  (132 kolom landmark + metadata). Belum ada kolom `phase`/`class`.
  Jalankan: `python src/extraction/extract_landmarks.py data/raw_videos/squat`
- `label_phase.py` — tahap 1b: tentukan `phase` per frame. Auto-detect jalan
  dulu sebagai DRAFT (segmentasi antar titik balik pada sudut sendi utama:
  squat=knee_angle, benchpress=elbow_angle, deadlift=hip_angle — lihat
  `src/features/joint_angles.py`), lalu direview/dikoreksi manual lewat
  playback + keypress. Override manual SELALU menang atas auto.
  Jalankan: `python src/extraction/label_phase.py <csv> [--start-sec N] [--end-sec N]`

  Kontrol: `space`=play/pause, `u`=UP/konsentrik, `d`=DOWN/eksentrik,
  `e`=EXCLUDED, `r`=ikut auto lagi, `,`/`.`=mundur/maju 1 frame,
  `[`/`]`=kecepatan, `s`=simpan, `q`=simpan&keluar. Setiap aksi dicetak ke
  terminal (frame, timestamp, sudut, fase, sumber label) untuk verifikasi.

  **Auto-trim start**: penekanan `space` PERTAMA KALI di tiap sesi otomatis
  menandai semua frame sebelum titik itu sebagai `excluded` (anggap masa
  persiapan/ancang-ancang) — sistem tidak perlu menebak fase sebelum kamu
  benar-benar mulai gerakan. Geser dulu ke frame awal gerakan (`,`/`.`)
  sebelum menekan space kalau perlu. Matikan dengan `--no-auto-trim-start`.
  **`q` (quit) TIDAK menandai apa pun setelahnya** — cuma berhenti sesi
  review, sisa frame yang belum direview tetap disimpan apa adanya (draft
  auto) dan bisa dilanjut kapan saja lewat fitur **resume** (frame yang
  sudah `manual` dari sesi sebelumnya otomatis dimuat balik saat CSV dibuka
  lagi, tidak akan hilang).

  `--start-sec` / `--end-sec` (alternatif manual, kalau sudah tahu detiknya)
  menandai bagian ancang-ancang di awal dan bagian mati di akhir sebagai
  `excluded` tanpa perlu buka playback sama sekali.

  Output kolom baru di CSV yang sama: `phase`
  (`concentric`/`eccentric`/`excluded`), `label_source` (`auto`/`manual`,
  untuk transparansi proses pelabelan di laporan), dan `class`
  (`{exercise}_{posture_class}_{phase}`, atau `excluded`).

- `preview_phase.py` — QA visual READ-ONLY (tidak bisa ubah label, beda dari
  label_phase.py): PNG plot sudut+fase, dan LIVE VIEW (window video langsung
  mainkan skeleton dari CSV di atas video mentah asli, tanpa nunggu render ke
  file). Kontrol: `space`=play/pause, `,`/`.`=step, `[`/`]`=kecepatan,
  `q`=keluar. Jalankan:
  `python src/extraction/preview_phase.py <csv> [--no-live] [--save-video]`

CATATAN penting vs Ko et al.: notebook labeling mereka hanya MENYIMPAN frame
saat tombol ditekan (sampling jarang, ~230 baris/video, CSV 133 kolom tanpa
metadata apa pun). Kita menyimpan SEMUA frame dengan label kontinu + metadata
lengkap, karena sliding window (Eq. 2 & 3 proposal) butuh frame berurutan dan
skema Model A vs Model B butuh kolom `camera_angle`.

Video `front` yang belum tersedia untuk suatu take otomatis tidak diproses
(bukan error) — `extract_landmarks.py` hanya memproses video yang ada di
folder input.

**Proteksi anti-timpa label**: `extract_landmarks.py` otomatis SKIP video yang
CSV tujuannya sudah punya kolom `phase` (sudah pernah dilabeli via
`label_phase.py`) — supaya aman dijalankan ulang kapan pun video baru masuk
tanpa risiko menghapus kerja labeling manual yang sudah ada. Pakai `--force`
kalau memang sengaja mau ekstrak ulang (label lama akan hilang, harus
dilabeli ulang dari nol).
