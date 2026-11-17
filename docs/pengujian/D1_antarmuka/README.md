# D1 — Pengujian antarmuka per modul

Tujuan: memeriksa apakah tampilan dan fitur aplikasi bekerja sesuai rancangan Subbab 3.3.3.5 (Gambar 3.6 sampai 3.8). Ini bukan uji akurasi. Akurasi model ada di A1-A4, B1, C1, dan E2.

Tabel skenario: `D1_skenario.csv` (pemisah `;`, buka lewat Import di Google Sheets). Berkas ini hanya memuat skenario. Hasil, status, dan bukti diisi di dokumen terpisah (Sheets hasil impor CSV ini), bukan di berkas ini.

## Mode yang diuji

Kamera langsung adalah mode utama (skenario 1 sampai 6) karena itu cara aplikasi dipakai sebenarnya. Unggah Video diuji pada satu skenario (7), karena 3.3.3.5 juga merancangnya sebagai sumber video. Kelas yang muncul bergantung pada keluaran sistem, jadi kriteria menilai tampilannya terhadap keluaran itu, bukan ketepatan klasifikasinya.

7 skenario ini menutup semua keluaran sistem pada ruang lingkup poin 15: kotak dan skeleton (2), klasifikasi postur (3), teks dan suara peringatan (6), jumlah repetisi (4), dan ringkasan dengan gambar (2, 6). Kolom Modul memakai nama modul di Subbab 3.3.3.

## Persiapan

- Satu HP untuk semua skenario. Pasang di penyangga atau tripod dengan kamera depan. Seluruh badan terlihat dari kepala sampai kaki, sudut diagonal sekitar 45° atau dari depan, hanya satu orang di bingkai. Volume HP menyala.
- Satu orang peraga (peneliti atau pelatih).
- Dua video di galeri HP untuk skenario 7: satu video bench press dan satu video deadlift (bebas).
- Hitung mundur ada di bagian “Pengaturan” (nilai awal 5 detik). Skenario 2 dan 6 memakai 3 detik.
- Catat tipe HP, browser dan versinya, jaringan, alamat aplikasi, dan tanggal uji di bagian paling bawah.

## Sesi

Satu sesi dipakai untuk beberapa skenario supaya cepat. Skenario 1 tidak butuh sesi. Jalannya sekitar 4 kali.

| Sesi | Isi | Skenario |
|---|---|---|
| K1 | Kamera, Squat, hitung mundur 3 detik. Setelah hitung mundur berdiri diam sekitar 10 detik, lalu 5 repetisi dengan postur benar. Stop, lihat ringkasan, muat ulang halaman. | 2, 3, 4, 5 |
| K2 | Kamera, Squat, hitung mundur 3 detik. 5 repetisi dengan lutut sengaja masuk ke dalam. Stop. | 6 |
| U1 | Unggah Video: satu video bench press, lalu satu video deadlift. | 7 |

## Cara mengisi (di dokumen hasil)

- `Hasil_pengujian`: tulis yang terlihat apa adanya.
- `Status`: Berhasil / Sebagian / Tidak berhasil. Jika bukan Berhasil, tulis penyebab singkat di `Hasil_pengujian`.
- `Bukti`: screenshot dengan nama sesuai kolom `Bukti` (contoh `D1-02.png`). Simpan di luar repositori karena memuat wajah, dan jangan di-commit.
- Aplikasi dibekukan: hasil yang tidak sesuai dicatat, tidak diperbaiki.
- Jika gerakan tidak memicu hasil yang dimaksud (skenario 5 dan 6), ulangi gerakan. Kriterianya tidak diubah.
- Kondisi di luar rancangan 3.3.3.5 sengaja tidak dimasukkan (izin kamera ditolak, server terputus, kamera tanpa orang). Bisa ditambah bila dosen meminta.

## Teks peringatan yang benar (`src/app/warnings_content.py`) dan nama di layar

| Kelas | Nama di layar | Teks peringatan |
|---|---|---|
| squat_kneeinward | Lutut masuk ke dalam | Jaga lutut tetap sejajar ujung kaki, jangan sampai masuk ke dalam |
| squat_backbend | Punggung membungkuk | Jaga punggung tetap tegak, jangan membungkuk ke depan |
| deadlift_backround | Punggung membulat | Pertahankan punggung lurus/netral, jangan membulat |
| deadlift_armsspread | Tangan terlalu lebar | Posisikan tangan selebar bahu, jangan terlalu lebar |
| benchpress_headbacklift | Kepala dan punggung terangkat | Jaga kepala dan punggung tetap menempel di bench, jangan terangkat |
| benchpress_armsspread | Tangan terlalu lebar | Jaga pegangan tangan tidak terlalu lebar |

## Catatan perangkat (isi saat tes)

- HP:
- Browser dan versi:
- Jaringan:
- Alamat aplikasi:
- Tanggal uji:
