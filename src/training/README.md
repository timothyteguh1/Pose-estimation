# src/training/

`train_model.py` — training & evaluasi Random Forest. **Cuma RF** (bab 6.1.3
& 8.3.2 proposal eksplisit cuma sebut 1 model, TIDAK ada pembanding
LR/Ridge/GB seperti Ko et al — itu pola Ko yang sengaja tidak diikuti di
sini). 1 exercise = 1 model terpisah (`models/{exercise}_rf.pkl`).

Jalankan: `python src/training/train_model.py squat`
(reusable ke `benchpress`/`deadlift` begitu datanya siap, tidak ada yang
hardcode nama exercise di kode).

Input: `data/windowed_features/splits/{exercise}_train.csv` + `_test.csv`
(hasil `src/features/build_dataset.py`). Kolom fitur diambil dinamis dari
header CSV, bukan hardcode jumlah.

Evaluasi wajib (bab 8.4): accuracy, precision, recall, F1-score (weighted,
**F1 = indikator utama**), classification report per kelas, confusion
matrix — semua disimpan ke `models/` (lihat `models/README.md`).

Perbandingan **per-frame vs sliding window**, dan **Model A (2 sudut) vs
Model B (3 sudut, occlusion test)** itu pengujian TERPISAH (bab 8.4 poin 11
& 14) — bukan bagian dari `train_model.py` ini, belum dibangun.
