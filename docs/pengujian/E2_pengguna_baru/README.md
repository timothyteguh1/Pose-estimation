# E2 — Pengujian pengguna baru

Tujuan: memeriksa apakah repetisi yang dihitung sistem benar, dan apakah kelas postur yang ditampilkan sistem benar, pada orang yang tidak ada di data latih. Hitungan dan penilaian postur dilakukan sendiri dari video (ground truth, GT).

E2 adalah bukti pendukung pada orang baru. Tabel MAE utama di buku adalah B1.

Tabel: `E2_tabel.csv` (pemisah `;`, buka lewat Import di Sheets). Satu baris = satu video. Semua video di folder tester sudah terdaftar. Video yang dipakai dipilih setelah semua diproses, lalu dibahas bersama.

## Kolom

| Kolom | Isi |
|---|---|
| Rep_sistem | angka Repetisi akhir yang ditampilkan sistem |
| Rep_GT | jumlah repetisi menurut hitungan sendiri dari video |
| Postur_GT | urutan postur tiap repetisi menurut penilaian sendiri |
| Sistem_bilang | urutan kelas yang ditampilkan sistem untuk tiap repetisi |
| Keterangan | catatan bebas (run ke berapa, FPS, status backend, hal yang perlu dijelaskan) |

## Cara menulis urutan

- `B` berarti benar. Kesalahan ditulis dengan nama kelas tanpa awalan latihan: `kneeinward`, `backbend`, `backround`, `armsspread`, `flatback`.
- Contoh 5 repetisi: `B B B kneeinward backbend`.
- Satu repetisi satu kata. Jika dua siklusnya berbeda, tulis `a/b`.
- Jika jumlah repetisi sistem dan GT berbeda, tandai agar urutannya tetap sejajar: `–` untuk repetisi yang terlewat sistem, `+` di depan kata untuk repetisi yang berlebih (contoh `B B B +B flatback B`).

## Cara menjalankan

1. Jalankan aplikasi, pilih Unggah Video, pilih latihan yang sesuai, pilih video.
2. Buka "Pengaturan" dan isi hitung mundur 0 (nilai awal 5 detik membuang 5 detik pertama video).
3. Laptop dicolok, aplikasi berat lain ditutup. Setelah selesai, buka "Info teknis" dan catat baris status backend dan FPS rata-rata di `Keterangan`.
4. Setelah analisis selesai, putar ulang videonya: angka repetisi dan banner mengikuti posisi video, jadi bisa dijeda per repetisi untuk dibaca.
5. Angka diisi dari run pertama. Aturan ini ditetapkan sebelum melihat hasil. Run lain, bila ada, ditulis di `Keterangan`, bukan menggantikan angka.
