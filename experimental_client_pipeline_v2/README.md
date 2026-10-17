# experimental_client_pipeline_v2

Model v2 untuk arsitektur baru (`experimental_client_pipeline/`), supaya v1 dan v2 bisa dijalankan berdampingan dan dibandingkan.

Satu-satunya beda dari model utama: 132 fitur koordinat rata-rata landmark (absolut) tidak masuk ke Random Forest. Fitur klasifikasi menjadi 165 sudut + 66 koordinat relatif = 231 fitur.

## Status

Dibekukan 12 Oktober 2026 sebagai solusi tanpa data partisipan tambahan. Berkas di `models/` tidak diubah selama E2 v2 berlangsung (hash di bawah). Kalau data partisipan ditambah, itu pengujian baru dengan folder baru; v2 tetap menjadi pembanding.

## Menjalankan

Dari root proyek:

    venv\Scripts\python.exe experimental_client_pipeline_v2\run_v2.py

- v2 di http://localhost:8601. Aplikasi utama (model v1) di http://localhost:8600. Port lain: `run_v2.py 8602`.
- Terminal harus mencetak `Model v2 aktif (231 fitur ...)`.
- Video diputar real-time, jadi jumlah frame yang sempat diproses bergantung pada beban laptop. Dua halaman yang memproses video bersamaan bisa menggeser hasil bukan karena model. Untuk perbandingan yang adil, analisis satu video pada satu waktu.

## Isi run_v2.py

Kode aplikasi tidak disalin dan tidak diubah: `app.py` dan `landmark_pipeline.py` dipakai langsung dari `experimental_client_pipeline/`. `run_v2.py` hanya (1) mengarahkan folder model ke `models/` di sini, (2) memeriksa tiap model benar-benar 231 fitur, (3) mengganti port. Folder utama dan aplikasi lama (`app.py` di root) tidak disentuh.

## Sama dan beda dengan model utama

Sama: kode aplikasi dan frontend, 11 sudut x 15 frame dengan scaler Min-Max yang sama, syarat visibility >= 0,6 untuk sudut dan aturan window tidak valid, window 0,5 detik, data latih dan split 70:30 per run, flip augmentation, Random Forest (100 pohon, min_samples_leaf 4, max_features sqrt, class_weight balanced, random_state 42, n_jobs None), enam kelas per latihan.

Beda: `feat_cols` di `*_feature_config.json` (132 kolom `coord_mean_*` dibuang). Akibatnya `sqrt` memilih 15 fitur per cabang (model utama 19). Nilai visibility dan z tidak lagi menjadi fitur klasifikasi. Hitungan repetisi dan siklus tidak berubah karena tidak memakai model.

`n_jobs=-1` (sisa pelatihan) sempat terpasang di berkas v2 dan membuat prediksi per pesan sekitar 2x lebih lambat (49 ms vs 24 ms), sehingga server tertinggal dari video. Sudah diganti None; prediksi tidak berubah.

## Berkas model (sha256, 16 karakter awal)

    squat_rf.pkl                    927baf3a640fa9f4
    benchpress_rf.pkl               b5226fbd664c7726
    deadlift_rf.pkl                 fe22f16c05dcc31c
    squat_feature_config.json       d0ac95110426b942
    benchpress_feature_config.json  79bdd1ceb9996247
    deadlift_feature_config.json    9330670ead6ba82c

## Hasil offline (dasar pengujian E2 v2)

Skrip dan hasil (JSON) ada di `evaluasi/`. v1 -> v2:

F1 berbobot, split 70:30 (7.432 window uji):

| Latihan | v1 | v2 |
|---|---|---|
| squat | 0,8818 | 0,8612 |
| benchpress | 0,9148 | 0,9152 |
| deadlift | 0,9545 | 0,9195 |
| gabungan | 0,9203 | 0,8990 |

LOPO (satu partisipan ditahan sebagai orang baru), F1 berbobot rata-rata 3 seed:

| Latihan | P1 | P2 | P3 | rata-rata |
|---|---|---|---|---|
| squat | 0,617 -> 0,646 | 0,465 -> 0,458 | 0,522 -> 0,495 | 0,5345 -> 0,5329 |
| benchpress | 0,330 -> 0,397 | 0,488 -> 0,549 | 0,411 -> 0,580 | 0,4096 -> 0,5087 |
| deadlift | 0,664 -> 0,624 | 0,596 -> 0,660 | 0,603 -> 0,644 | 0,6210 -> 0,6428 |
| rata-rata 3 latihan | | | | 0,5217 -> 0,5615 |

Dari 9 sel (latihan x partisipan), 6 naik dan 3 turun (squat P2 dan P3, deadlift P1). Syarat awal "naik di minimal 7 dari 9 sel" tidak terpenuhi persis. Di split 70:30 video yang sama ada di data latih dan uji, sehingga koordinat absolut yang mengenali rekaman ikut membantu dan membuang koordinat itu menurunkan F1; LOPO lebih mewakili orang baru.

Replay landmark browser asli E2 dengan masukan identik (bukan hasil E2 resmi), akurasi window dari yang berkelas: squat 62,3% -> 65,2%; benchpress 61,9% -> 72,8% (3 dari 6 video terpetakan); deadlift 59,8% -> 70,7%.

Fitur dipilih dengan LOPO pada data latih. Hasil E2 tidak dipakai untuk memilih, tetapi arah kandidat (membuang koordinat absolut) berasal dari diagnosis kesalahan E2 v1.

## Isi evaluasi/

- `evaluasi_v1_v2.py`: F1 70:30 dan LOPO (3 seed) untuk v1 dan v2. Menulis `../models/<latihan>_evaluation.json` dan `<latihan>_confusion_matrix.png` untuk v2 (format sama dengan model utama), dan `<latihan>_v1_vs_v2.json` di folder ini (hasil per kelas v1 dan v2 serta daftar perubahan per kelas).
- `latih_v2.py`: melatih ulang v2 dari data utama. Prediksinya identik dengan berkas di `models/` (selisih probabilitas 0).
- `pencarian/`: skrip dan hasil pencarian fitur (11 kumpulan fitur, LOPO seed 42) dan parameter Random Forest yang mendasari pemilihan v2.
