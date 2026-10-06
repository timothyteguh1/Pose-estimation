# A2 — Sliding window vs per-frame

Jalankan dari akar proyek: `venv\Scripts\python docs\pengujian\A2_perframe_vs_window\A2.py`
Keluaran: `A2_tabel.csv` (satu-satunya file hasil, pemisah `;`, desimal koma): 8 kolom tabel resmi ditambah
`Perubahan_prediksi_persen`. Pertama kali ±5 menit karena data per-frame dibangun di folder temp; berkas proyek tidak
diubah. Layar juga mencetak jumlah pasangan change rate. Dihitung 5 Okt 2026.

## Yang dibandingkan

Model final (Method E, dibekukan 23 Sep 2026), 81 video, pembagian 70:30 per run (seed 42). Satu-satunya yang
dibedakan adalah panjang window: 15 frame (0,5 detik) vs 1 frame.

- **Window**: model produksi `experimental_client_pipeline/models/*_rf.pkl` (yang dimuat app), dinilai pada semua window
  uji. Angkanya sama dengan A1.
- **Per-frame**: `build_dataset` yang sama dengan `--window-sec 0.0333` dari data berlabel yang sama, RF sama (100 pohon,
  `min_samples_leaf` 4, balanced, seed 42), dinilai pada **semua frame uji**.
- Konfigurasi per-frame dan produksi sama (relatif, flip hanya train, tanpa z, visibility 0,6, stride 1) kecuali
  `window_sec` dan jumlah fitur (209 vs 363). Skrip menghentikan diri bila ada frame akhir window uji yang tidak ada di
  frame uji per-frame.

## Tabel resmi (8 kolom pertama `A2_tabel.csv`; per-frame dinilai pada SEMUA frame uji)

| Latihan | Pendekatan | n test | Accuracy | Precision (w) | Recall (w) | F1 weighted | F1 macro |
|---|---|---|---|---|---|---|---|
| Squat | Window | 2.477 | 0,8845 | 0,8953 | 0,8845 | 0,8818 | 0,8805 |
| Squat | Per-frame | 3.979 | 0,8193 | 0,8233 | 0,8193 | 0,8153 | 0,8071 |
| Bench press | Window | 1.868 | 0,9160 | 0,9221 | 0,9160 | 0,9148 | 0,9113 |
| Bench press | Per-frame | 3.458 | 0,7817 | 0,7833 | 0,7817 | 0,7782 | 0,7765 |
| Deadlift | Window | 3.087 | 0,9550 | 0,9564 | 0,9550 | 0,9545 | 0,9504 |
| Deadlift | Per-frame | 4.523 | 0,9043 | 0,9062 | 0,9043 | 0,9043 | 0,9017 |
| Gabungan 3 latihan | Window | 7.432 | 0,9217 | 0,9274 | 0,9217 | 0,9203 | 0,9141 |
| Gabungan 3 latihan | Per-frame | 11.960 | 0,8406 | 0,8431 | 0,8406 | 0,8382 | 0,8284 |

Selisih F1 weighted (window − per-frame): squat +0,0665, bench press +0,1366, deadlift +0,0502, gabungan +0,0821.
Gabungan = semua prediksi uji digabung (bukan rata-rata tiga angka).

## Uji stabilitas (kolom `Perubahan_prediksi_persen`)

Persentase pasangan titik waktu berurutan (video sama, label asli sama, frame akhir selisih 1) yang prediksinya
berganti. Makin kecil makin stabil. Jumlah pasangan dicetak di layar saat skrip dijalankan (tidak masuk CSV).

| Latihan | Window | Per-frame |
|---|---|---|
| Squat | 1,76% (dari 2.383 pasangan) | 5,01% (dari 3.832) |
| Bench press | 1,31% (dari 1.762) | 5,37% (dari 3.335) |
| Deadlift | 0,64% (dari 2.989) | 2,93% (dari 4.410) |
| Gabungan | 1,18% (dari 7.134) | 4,32% (dari 11.577) |

Window berurutan berbagi 14 dari 15 frame, jadi halus karena cara kerjanya. Itu yang dialami pengguna saat live,
tetapi pasangan titik waktunya tidak independen.

## Catatan penjelasan (untuk Bab 4)

**Kenapa F1 window lebih tinggi.** Window baru ada setelah 15 frame terkumpul, jadi 14 frame pertama tiap run (dan run yang
lebih pendek dari 15 frame) tidak punya window. Per-frame tetap menilai semua frame itu, makanya n test-nya lebih banyak
(squat: 3.979 vs 2.477). Jadi keduanya tidak dinilai pada frame yang persis sama, dan itu perlu diingat saat membaca selisih
F1. Selisih ini tidak boleh dikaitkan dengan informasi urutan 15 frame, karena A2 tidak menguji hal itu.

**Kenapa change rate window lebih kecil.** Dua window berurutan (stride 1) berbagi 14 dari 15 frame, jadi inputnya hampir
sama dan prediksinya jarang berganti. Per-frame memakai 1 frame sebagai input, sehingga fluktuasi kecil antar frame
langsung memengaruhi prediksinya. Ini efek cara kerja window, bukan bukti window lebih akurat; akurasi dilihat dari F1.

## Kalimat Bab IV

> Pada pembagian data yang sama, pendekatan sliding window menghasilkan weighted F1-score yang lebih tinggi daripada
> pendekatan per-frame pada ketiga latihan, yaitu 0,882 dibanding 0,815 pada squat, 0,915 dibanding 0,778 pada bench
> press, dan 0,955 dibanding 0,904 pada deadlift; secara gabungan 0,920 dibanding 0,838. Selisih ini perlu dibaca
> bersama perbedaan himpunan yang dinilai: window hanya dibentuk dari 15 frame yang seluruhnya berada dalam satu fase,
> sehingga pada 14 frame pertama tiap run (dan pada run yang lebih pendek dari 15 frame) tidak ada prediksi window yang
> dinilai, sedangkan per-frame menilai seluruh frame (n uji 3.979, 3.458, dan 4.523 dibanding 2.477, 1.868, dan 3.087).
> Prediksi window juga lebih stabil: prediksi berubah pada 0,6–1,8% pasangan titik waktu berurutan, dibandingkan
> 2,9–5,4% pada per-frame. Imbalnya, window memerlukan 0,5 detik sebelum prediksi pertama.

Jangan menulis bahwa "informasi dari rangkaian 15 frame" yang menyebabkan selisih F1, atau bahwa change rate yang lebih
kecil berarti window lebih akurat.

## Batasan

- Satu seed, satu split, tanpa uji signifikansi formal; sampel uji berurutan tidak independen.
- Window yang melintasi titik balik (kelas berganti) tidak dibentuk: run dipotong tiap kelas berganti, jadi window campur
  tidak ada di data latih maupun uji (A1 dan A2). Saat live sistem tetap memprediksi di sana; dampaknya pada hasil per
  siklus diukur lewat E2, bukan di A2.
- A2 menilai klasifikasi offline pada data berlabel (seperti A1). Rantai live (YOLO + MediaPipe di browser,
  WebSocket), penghitung repetisi, dan evaluasi siklus diuji di B1 dan E2.
- Per-frame di sini memakai resep fitur yang sama dengan window (209 fitur), bukan replikasi persis Ko et al.
