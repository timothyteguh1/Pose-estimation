# A3 — Model 2 sudut diagonal vs 3 sudut

## Cara menjalankan

Dari akar proyek:

```
venv\Scripts\python docs\pengujian\A3_diagonal_vs_depan\A3.py
```

Beberapa menit. Tabel tercetak di layar, hasil lengkap ke `hasil\{latihan}_evaluation_2sudut.json` (3 file). Skrip hanya
membaca model dan data proyek; model proyek tidak berubah.

## Notes

- **Model 2 sudut**: RF yang sama dengan produksi (100 pohon, `min_samples_leaf` 4, balanced, seed 42), dilatih hanya dari
  window left45 dan right45 (window latih asli squat 4.422, bench press 3.921, deadlift 5.290).
- **Model 3 sudut** = model final (A1), tidak diuji ulang. Pembandingnya `experimental_client_pipeline/models/*_evaluation.json`.
- Model 2 sudut diuji dua cara:
  - `uji_diagonal`: hanya left45 + right45. Jumlah per sudut ada di `camera_angle_uji` (front = 0).
  - `uji_data_lengkap`: seluruh test split 3 sudut (2.477 / 1.868 / 3.087), sama dengan test A1, jadi bisa dibandingkan
    baris demi baris dengan tabel A1.
- Isi JSON: bentuk seperti `*_evaluation.json` (n_train, classes, hyperparameters) plus dua blok uji berisi n_test,
  accuracy, precision/recall/f1 weighted, `classification_report` per kelas, `confusion_matrix`. F1 macro =
  `classification_report["macro avg"]["f1-score"]`.
- **A3 hanya pembanding.** Model final tetap 3 sudut. Hasilnya menunjukkan sudut kamera memengaruhi kegunaan model, bukan
  alasan memakai model 2 sudut.

## Hasil ringkas (F1 weighted)

| | Squat | Bench press | Deadlift |
|---|---|---|---|
| 2 sudut, uji diagonal saja | 0,9134 | 0,9504 | 0,9540 |
| 2 sudut, uji data lengkap | 0,7447 | 0,7896 | 0,7500 |
| 3 sudut (A1), data lengkap | 0,8818 | 0,9148 | 0,9545 |

n test: uji diagonal saja 1.459 / 1.167 / 2.106; data lengkap 2.477 / 1.868 / 3.087. Tanpa baris gabungan: tiap latihan
adalah model sendiri, dan arah hasilnya sama di ketiganya. Tabel lengkap (accuracy, precision, recall, F1 macro) ada di
layar saat skrip dijalankan dan di JSON.

## Batasan

Satu seed, satu split, tiga partisipan (ketiganya ada di data latih dan uji), tanpa uji signifikansi formal. Penyebab
turunnya model 2 sudut di sudut depan (occlusion atau hal lain) tidak diuji.
