# src/app/

Aplikasi web real-time (arsitektur client-server): YOLOv11 -> MediaPipe ->
sliding window joint angle -> Random Forest -> repetition counting dari pola
naik-turun joint angle -> feedback teks+suara -> ringkasan sesi.

Boleh memakai kerangka `Streamlit.py` milik Ko et al. sebagai referensi
struktur kode, tapi logic prediksi harus window-based (bukan per-frame
tunggal) dan deteksi objek harus YOLOv11 (bukan YOLOv5).
