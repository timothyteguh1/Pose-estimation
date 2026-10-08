# C1 — YOLOv5 vs YOLOv11 (deteksi orang, di browser)

Janji proposal (ruang lingkup poin 13): membandingkan YOLOv5 dan YOLOv11 berdasarkan waktu inferensi dan ketepatan deteksi objek manusia. Tidak ada batas lulus; hasilnya perbandingan. Ini pengujian tambahan.

## Perintah (dari akar proyek)

```
venv\Scripts\python docs\pengujian\C1_yolov5_vs_yolov11\C1_yolo.py
```

Pilihan: `--browser chrome` (atau `edge`) untuk memilih browser, `--detik 2` (satu frame tiap 2 detik), `--no-open` (buka alamat yang dicetak sendiri), `--backend wasm` (paksa CPU, opsional untuk run kedua; beri `--out` folder lain supaya hasil pertama tidak tertimpa).

Alur: skrip mengekspor YOLOv5nu ke ONNX (sekali saja, hasilnya `yolov5nu_640.onnx`), mengambil frame dari 81 video dataset, menyalakan server lokal, lalu membuka halaman uji di browser. **Biarkan tab browser terlihat sampai selesai** (sekitar 5-10 menit). Hasil dicetak di terminal dan disimpan otomatis:

- `C1_tabel.csv`: tabel utama (sama persis dengan yang tercetak di terminal).
- `C1_detail.json`: rincian (median, p95, confidence, per frame, lingkungan uji, angka resmi).
- `C1_terminal.log`: rekaman layar.

## Yang disamakan

Hanya model yang berbeda. Sama untuk keduanya: frame yang persis sama (dipakai bergantian), browser dan runtime yang sama (ONNX Runtime Web 1.30.0, webgpu atau wasm untuk keduanya), kode deteksi aplikasi (`yolo.js`, tidak diubah), input 640, confidence 0,7, NMS 0,45, frame diperkecil ke sisi panjang 720 seperti di aplikasi, pemanasan sebelum mengukur, urutan model digilir tiap frame. YOLOv5nu diekspor dengan pengaturan yang sama dengan model aplikasi (input 640, simplify, tidak dinamis, tanpa NMS bawaan).

## Yang diukur

- Waktu inferensi: milidetik satu panggilan `detect()` di browser (praproses + inferensi + pascaproses), seperti "YOLO nyata" di aplikasi.
- Ketepatan deteksi: laju deteksi (persen frame yang orangnya terdeteksi pada confidence 0,7). Kotak sebenarnya tidak ada, jadi ini bukan precision atau recall.
- `mAP50_95_resmi`: angka dokumentasi Ultralytics (COCO, 80 kelas, bukan khusus orang), bukan hasil run ini. Sumber: docs.ultralytics.com/models/yolo11 dan /models/yolov5, diakses 5 Okt 2026.

## Data

81 video dataset (27 per latihan), satu frame tiap 2 detik, sekitar 970 frame. Frame diambil dengan OpenCV (46 dari 81 video berformat HEVC yang tidak selalu bisa diputar browser) lalu dimuat browser sebagai gambar. Tidak memakai video tester.

## Batas klaim

- Kedua model dipakai apa adanya (pre-trained), tidak dilatih ulang.
- YOLOv5nu adalah varian Ultralytics dari YOLOv5 (tanpa anchor), bukan repositori YOLOv5 asli yang dipakai Ko et al.
- Waktu adalah waktu browser di PC ini (lihat baris "Lingkungan uji" di terminal), bukan di HP, dan bergantung pada PC serta beban saat run.
- Ini perbandingan dua detektor. Bukan uji "dengan YOLO atau tanpa YOLO" dan bukan uji akurasi klasifikasi.
