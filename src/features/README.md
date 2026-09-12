# src/features/

Perhitungan 11 joint angle per frame, sliding window (Eq. 2 & 3), dan
normalisasi Min-Max pada fitur angle hasil window.

`joint_angles.py` — sudah diimplementasikan (dipakai oleh
`src/extraction/label_phase.py` untuk deteksi fase otomatis):
- `calculate_angle(a, b, c)` — Eq.1 proposal (arccos dot-product BA.BC),
  bukan arctan2 seperti kode Ko et al., tapi hasilnya ekuivalen secara
  matematis (unsigned angle 0-180 derajat)
- `compute_joint_angles(row)` — 11 sudut (neck, L/R shoulder/elbow/hip/knee/
  ankle) dari satu baris landmark, definisi triple titik sama persis dengan
  Ko et al. `Afterprocessing.ipynb` (`calculate_joint_angles`)
- `primary_angle_for_frame(row, exercise)` — sudut utama per exercise untuk
  deteksi fase (squat=knee, benchpress=elbow, deadlift=hip), pakai rata-rata
  L/R kalau dua-duanya visible (>=0.6), fallback ke sisi yang visible

Sliding window (Eq. 2 & 3) dan normalisasi Min-Max (Eq. 5) BELUM
diimplementasikan — bagian baru yang tidak ada di kode referensi, jadwal
setelah tahap labeling fase selesai untuk video yang tersedia.
