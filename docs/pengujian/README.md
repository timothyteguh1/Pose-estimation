# Folder pengujian (Bab IV)

Tiap uji punya folder sendiri berisi skrip, hasil, dan README singkat (perintah dan notes).

## Cara menjalankan (dari akar proyek)

```
venv\Scripts\python docs\pengujian\A2_perframe_vs_window\A2.py     # hasil: A2_tabel.csv
venv\Scripts\python docs\pengujian\A3_diagonal_vs_depan\A3.py      # hasil: hasil\*_evaluation_2sudut.json
```

Skrip hanya membaca model dan data proyek; model dan data tidak berubah.

## Daftar uji

| ID | Uji | Status | Folder |
|---|---|---|---|
| A1 | Metrik model | angka dari `experimental_client_pipeline/models/*_evaluation.json` | - |
| A2 | Sliding window vs per-frame (+ stabilitas) | selesai | `A2_perframe_vs_window/` |
| A3 | Model 2 sudut diagonal vs 3 sudut | selesai | `A3_diagonal_vs_depan/` |
| A4 | Konfigurasi final (Method E) | belum | - |
| B1 | MAE repetisi | menunggu video E2 | - |
| B2 | Repetisi tidak memengaruhi klasifikasi | belum | - |
| C1 | YOLOv5 vs YOLOv11 | dites 26 Sep, belum dirapikan | - |
| D1 | Antarmuka per modul | belum | - |
| E1 | Kuesioner Likert | menunggu responden | - |
| E2 | Pengguna baru | berjalan | - |

## Aturan

- Sistem dibekukan: kode dan model tidak diubah.
- Rumus sama dengan buku: F1 Pers. 11 (weighted dan macro), MAE Pers. 12.
- Satu angka resmi per hasil; angka pembanding diberi label jelas.
- Catat sumber tiap angka (file dan tanggal) di tab Log pada Sheets.
