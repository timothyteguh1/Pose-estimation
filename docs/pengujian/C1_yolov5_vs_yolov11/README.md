# C1 — YOLOv5 vs YOLOv11 (deteksi orang, di browser)

Janji proposal (ruang lingkup poin 13): membandingkan YOLOv5 dan YOLOv11 berdasarkan waktu inferensi dan ketepatan deteksi objek manusia. Tidak ada batas lulus; hasilnya perbandingan. Ini pengujian tambahan.

## Satu perintah (dari akar proyek)

```
venv\Scripts\python docs\pengujian\C1_yolov5_vs_yolov11\C1_yolo.py --browser chrome
```

Perintah ini menjalankan semuanya berurutan. Yang diminta dari peneliti hanya satu: menggambar kotak orang pada 90 frame saat jendela pelabelan muncul (sekitar 15-25 menit, hanya pada run pertama).

1. YOLOv5nu diekspor ke ONNX dengan pengaturan yang sama dengan model aplikasi (sekali saja; hasilnya `yolov5nu_640.onnx`).
2. Frame diambil dari 81 video dataset, satu frame tiap 2 detik (sekitar 970 frame).
3. Jendela pelabelan terbuka untuk 90 frame: 10 frame untuk tiap kombinasi latihan x sudut kamera (9 kombinasi), dipilih dengan seed 42 supaya pilihannya sama bila diulang. Kotak tersimpan otomatis di `C1_kotak_sebenarnya.json` setelah tiap frame. Tekan Q untuk berhenti, lalu jalankan perintah yang sama untuk melanjutkan. Bila 90 frame sudah berlabel, tahap ini dilewati.
4. Server lokal menyala dan Chrome membuka halaman uji. **Biarkan tab terlihat sampai selesai** (sekitar 5-10 menit).
5. Tabel dicetak di terminal dan disimpan.

Hasil (menimpa hasil lama; hasil lama masih ada di riwayat git):

- `C1_tabel.csv`: tabel utama, sama dengan yang tercetak di terminal (Model, waktu rata-rata dalam ms, kotak tepat dalam persen).
- `C1_detail.json`: rincian (median, p95, kotak per frame, lingkungan uji).
- `C1_terminal.log`: rekaman layar.
- `C1_kotak_sebenarnya.json`: kotak sebenarnya yang digambar peneliti. Ini bukti pengujian; jangan dihapus, karena menghapusnya berarti mengulang pelabelan dari awal.

Pilihan lain: `--browser edge`, `--detik 2`, `--per-sel 10`, `--no-open` (buka alamat yang dicetak sendiri), `--backend wasm` (paksa CPU, opsional untuk run kedua; beri `--out` folder lain supaya hasil pertama tidak tertimpa), `--tanpa-ketepatan` (hanya waktu, tanpa pelabelan), `--gt` (berkas kotak sebenarnya lain), `--simpan-frame`.

## Cara menggambar kotak

Kotak sebenarnya = kotak BADAN orang yang sedang berlatih: dari kepala sampai kaki dan sampai ujung lengan. Barbel, plate, rak, dan bangku TIDAK ikut, walau menonjol jauh keluar dari badan (kelas orang pada model deteksi hanya mencakup badan, jadi kotak yang ikut barbel tidak sebanding). Bagian tubuh yang keluar frame: kotak sampai tepi frame. Orang di latar belakang diabaikan. Bila orang yang berlatih tidak terlihat sama sekali, tekan N; frame itu tidak masuk persentase.

Aturan ini juga tertulis di judul jendela dan di pojok kiri atas gambar. Jendela hanya menampilkan frame polos, jadi kotak digambar sebelum hasil model diketahui.

Tombol: ENTER atau A = mulai gambar (seret mouse, lalu ENTER atau SPASI; C = batal) | N = tidak ada orang | S = lewati (muncul lagi di akhir) | Q = simpan dan keluar. Jendela bisa muncul di belakang jendela lain; cek taskbar.

## Yang disamakan

Hanya model yang berbeda. Sama untuk keduanya: frame yang persis sama (dipakai bergantian), browser dan runtime yang sama (ONNX Runtime Web 1.30.0, webgpu atau wasm untuk keduanya), kode deteksi aplikasi (`yolo.js`, tidak diubah), input 640, confidence 0,7, NMS 0,45, frame diperkecil ke sisi panjang 720 seperti di aplikasi, pemanasan sebelum mengukur, urutan model digilir tiap frame. YOLOv5nu diekspor dengan pengaturan yang sama dengan model aplikasi (input 640, simplify, tidak dinamis, tanpa NMS bawaan).

## Yang diukur

- Waktu inferensi: milidetik satu panggilan `detect()` di browser (praproses + inferensi + pascaproses), seperti "YOLO nyata" di aplikasi.
- Ketepatan deteksi: persen frame berlabel yang kotak orangnya tepat. Kotak yang dinilai adalah kotak dengan confidence tertinggi (seperti di aplikasi); tepat bila IoU >= 0,5 terhadap kotak sebenarnya. Frame tanpa deteksi dihitung gagal. Frame berlabel "tidak ada orang" tidak masuk persentase; kotak yang muncul di frame itu (misalnya pada orang di latar belakang) dicatat sebagai kotak pada orang lain. Terminal juga mencetak selang kepercayaan 95% (Wilson), rata-rata IoU, hasil per latihan, dan hitungan berpasangan (frame yang sama).
- Laju deteksi (persen frame dengan minimal satu orang terdeteksi) dan confidence rata-rata tetap dicatat di `C1_detail.json` sebagai cadangan; tidak masuk tabel.
- mAP COCO dari dokumentasi Ultralytics tidak dipakai: 80 kelas, bukan khusus orang, dan bukan hasil sendiri.

## Data

81 video dataset (27 per latihan; 9 per kombinasi latihan x sudut kamera), satu frame tiap 2 detik, sekitar 970 frame; 90 di antaranya diberi kotak sebenarnya. Frame diambil dengan OpenCV (46 dari 81 video berformat HEVC yang tidak selalu bisa diputar browser) lalu dimuat browser sebagai gambar. Tidak memakai video tester.

## Batas klaim

- Kedua model dipakai apa adanya (pre-trained), tidak dilatih ulang.
- YOLOv5nu adalah varian Ultralytics dari YOLOv5 (tanpa anchor), bukan repositori YOLOv5 asli yang dipakai Ko et al.
- Kotak sebenarnya digambar satu orang (peneliti) pada 90 frame; persentase dari 90 frame punya selang kepercayaan yang cukup lebar (dicetak di terminal).
- Frame berasal dari 3 partisipan di satu gym; keragaman orang dan tempat terbatas.
- Waktu adalah waktu browser di PC ini (lihat baris "Lingkungan uji" di terminal), bukan di HP, dan bergantung pada PC serta beban saat run.
- Ini perbandingan dua detektor. Bukan uji "dengan YOLO atau tanpa YOLO" dan bukan uji akurasi klasifikasi.

## Catatan

- Tabel lama (run 7 Okt) memakai laju deteksi dan mAP resmi Ultralytics sebagai ukuran ketepatan; keduanya diganti kotak tepat terhadap kotak sebenarnya.
- Pelabelan pertama (22 Okt 2026) memasukkan barbel dan plate ke dalam kotak pada banyak frame, menyimpang dari aturan di atas, sehingga persen kotak tepat terlalu rendah. Penyebabnya terlihat pada frame yang gagal untuk kedua model: kotak kedua model hampir sama dan menempel di badan, sedangkan kotak peneliti jauh lebih lebar. Seluruh 84 frame yang berkotak digambar ulang dengan aturan badan saja (bukan hanya yang gagal) tanpa melihat hasil model; 6 frame "tidak ada orang" dipertahankan. Hasil pelabelan pertama tidak dipakai.
- Nama video dataset memuat nama kelas. Kelas bench flat back dinamai ulang `headbacklift` pada 17 Okt 2026, jadi nama video di keluaran memakai nama baru.
