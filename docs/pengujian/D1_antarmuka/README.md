# D1 — Pengujian antarmuka per modul

Tujuan: memeriksa apakah tampilan dan fitur aplikasi bekerja sesuai rancangan Subbab 3.3.3.5 (Gambar 3.6 sampai 3.8). Ini bukan uji akurasi. Akurasi model ada di A1-A4, B1, C1, dan E2.

Tabel skenario: `D1_skenario.csv` (pemisah `;`, buka lewat Import di Sheets). Kolom `Hasil_pengujian`, `Status`, dan `Bukti` diisi saat tes.

## Mode yang diuji

Hanya Upload Video, karena videonya bisa dipilih (benar, salah, tiga latihan) dan hasilnya bisa diulang. Upload Video adalah salah satu sumber video yang dirancang di 3.3.3.5. Panel hasil, hitungan repetisi, peringatan teks dan suara, serta ringkasan memakai kode tampilan yang sama dengan Kamera Live. Yang khusus Kamera Live (izin kamera, tombol Stop) tidak diuji di D1. Jika dosen memintanya, tambahkan satu skenario Kamera Live.

7 skenario ini menutup semua keluaran sistem pada ruang lingkup poin 15: kotak dan skeleton (2, 3), klasifikasi postur (3, 6), teks dan suara peringatan (7), jumlah repetisi (4, 5), dan ringkasan dengan gambar (2).

## Persiapan

- Satu HP untuk semua skenario. Catat tipe HP, browser dan versinya, jaringan, alamat aplikasi, dan tanggal uji di bawah.
- Lima video di galeri HP:
  - video A: squat yang memuat kesalahan postur dan memicu peringatan,
  - video B: postur benar yang hasilnya "correct",
  - video C: orang berdiri diam sekitar 15 detik (rekam sendiri),
  - satu video Bench Press dan satu video Deadlift (bebas).
- Video A dan B dipilih agar tampilannya bisa diuji. Jika video tidak memicu hasil yang dimaksud, ganti videonya, bukan kriterianya.
- Simpan screenshot di folder `bukti/` dengan nama sesuai kolom `Bukti` (contoh `D1-02.png`).

## Cara mengisi

- `Hasil_pengujian`: tulis yang terlihat apa adanya.
- `Status`: Berhasil / Sebagian / Tidak berhasil. Jika bukan Berhasil, tulis penyebab singkat di `Hasil_pengujian`.
- Satu sesi dipakai untuk beberapa skenario: sesi video A untuk skenario 2, 3, 4, 7. Jadi jalannya sekitar 6 kali.
- Aplikasi dibekukan: hasil yang tidak sesuai dicatat, tidak diperbaiki.
- Kondisi di luar rancangan 3.3.3.5 sengaja tidak dimasukkan (izin kamera ditolak, server terputus, kamera tanpa orang). Bisa ditambah bila dosen meminta.

## Teks peringatan yang benar (`src/app/warnings_content.py`)

| Kelas | Teks |
|---|---|
| squat_kneeinward | Jaga lutut tetap sejajar ujung kaki, jangan sampai masuk ke dalam |
| squat_backbend | Jaga punggung tetap tegak, jangan membungkuk ke depan |
| deadlift_backround | Pertahankan punggung lurus/netral, jangan membulat |
| deadlift_armsspread | Posisikan tangan selebar bahu, jangan terlalu lebar |
| benchpress_flatback | Pertahankan sedikit lengkungan alami di punggung bawah, jangan menempel rata ke bench |
| benchpress_armsspread | Jaga siku tidak terlalu melebar ke samping saat menurunkan beban |

## Catatan perangkat (isi saat tes)

- HP:
- Browser dan versi:
- Jaringan:
- Alamat aplikasi:
- Tanggal uji:
