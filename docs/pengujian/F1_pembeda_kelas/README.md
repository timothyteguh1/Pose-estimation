# F1 — Pembeda kelas dibanding variasi antar orang

Menjawab catatan dosen 16 Okt (butir 5): kelas yang tertukar itu karena pembeda kelasnya kecil, karena variasi orang, atau karena salah pengujian. Ini analisis data yang sudah ada: tanpa pelatihan ulang dan tanpa pengujian baru.

## Satu perintah (dari akar proyek)

```
venv\Scripts\python docs\pengujian\F1_pembeda_kelas\F1_pembeda_kelas.py
```

Sekitar 1 menit. Keluaran: `F1_tabel.csv` (ringkasan per pasangan kelas), `F1_detail.csv` (semua fitur), `F1_terminal.log` (rekaman layar).

## Cara kerja

Data: semua jendela asli (bukan cermin) dari split train dan test tiap latihan (`client_pipeline/data/windowed_features/splits`). Sudut diubah kembali ke derajat dengan penskala model aplikasi.

Untuk tiap pasangan (postur benar vs satu kesalahan) dan tiap fitur kunci (11 sudut rata-rata jendela, tinggi hidung, tinggi bahu, jarak tangan per jarak bahu, jarak lutut per jarak pergelangan kaki):

- **Selisih kelas**: rata-rata kelas salah dikurangi kelas benar, dihitung per partisipan di dalam sel partisipan x kamera x fase yang sama (supaya posisi kamera dan fase tidak ikut), lalu dirata-rata antar partisipan. Sel dipakai bila kedua kelas punya minimal 20 jendela.
- **Searah**: berapa partisipan yang selisihnya bertanda sama.
- **Sebaran orang**: simpangan baku antar partisipan pada kelas benar. **R** = |selisih kelas| / sebaran orang.
- **Sebaran dalam**: simpangan baku antar jendela di dalam satu kelas. **d** = |selisih kelas| / sebaran dalam.
- **Pembeda kuat**: searah pada semua partisipan, R >= 1, dan d >= 1. Batas 1 dipilih sederhana dan ditetapkan sebelum melihat hasil.

## Cara membaca

R >= 1: pembeda kelas lebih besar daripada perbedaan antar orang. d >= 1: pembeda kelas lebih besar daripada variasi di dalam kelas. Kelas dengan banyak fitur pembeda kuat mudah dibedakan dari kelas benar; kelas tanpa fitur pembeda yang searah dan cukup besar rawan tertukar.

## Batas klaim

- Hanya 3 partisipan; sebaran antar orang dari 3 nilai rata-rata hanya gambaran kasar.
- Tiap kelas direkam di video terpisah, jadi selisih kelas bisa memuat perbedaan insidental (misalnya lebar pegangan) selain gerakan yang dimaksud.
- Hanya fitur yang dipakai model (sudut sendi dan posisi titik MediaPipe pada sudut kamera diagonal kiri, diagonal kanan, depan). Pembeda yang tidak terukur oleh fitur ini (misalnya lengkung punggung) tidak terlihat di sini.
- Ini analisis deskriptif, bukan uji sebab-akibat.
