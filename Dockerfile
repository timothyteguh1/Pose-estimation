# Dockerfile aplikasi web (bab 6.4 proposal, client-server) -- MEMBUNGKUS
# app.py + semua dependency-nya jadi 1 image yg bisa dijalankan di komputer
# manapun yg ada Docker, TANPA install Python/library manual satu-satu.
#
# TIDAK mengubah kode/model/akurasi APAPUN -- ini murni cara PAKET/JALANKAN
# aplikasi yang SAMA PERSIS dgn yg sudah divalidasi di venv (lihat diskusi
# proyek 2026-09-11: Docker = alat packaging/reproducibility, BUKAN alat
# percepat compute -- performa CPU-bound (YOLO+MediaPipe) di dalam container
# diperkirakan MIRIP dgn jalan langsung di venv, bukan lebih cepat/lambat
# signifikan).
#
# CATATAN percobaan build (2026-09-11):
# 1. `--index-url` (BUKAN --extra-index-url) MENGGANTIKAN total sumber paket
#    (PyPI) jadi cuma server PyTorch -- pip butuh paket pendukung yg cuma ada
#    di PyPI biasa (flit_core) gagal ketemu. Diperbaiki -> --extra-index-url
#    (MENAMBAH sumber, bukan mengganti).
# 2. mediapipe (butuh protobuf<5) vs streamlit (butuh protobuf>=5.26.1) --
#    rentang TIDAK berpotongan. protobuf==4.25.9 (versi TERBUKTI jalan di
#    venv) dipaksa di requirements.txt + --use-deprecated=legacy-resolver di
#    bawah, spy pip tidak menolak kombinasi yg sebenarnya jalan ini.

FROM python:3.10-slim

WORKDIR /app

# Library sistem yg dibutuhkan opencv-python (libGL, dll -- OpenCV wheel Linux
# butuh ini utk operasi gambar/video walau headless) & mediapipe/av (codec
# video dasar). --no-install-recommends + hapus cache apt supaya image tidak
# bengkak.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install dependency Python DULU (sebelum copy source code) -- supaya kalau
# cuma source code yg berubah (bukan requirements.txt), Docker bisa pakai
# CACHE layer install ini (build ulang jauh lebih cepat), tidak install ulang
# semua library tiap kali.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.0 torchvision==0.29.0 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt

# Copy HANYA yg dibutuhkan app jalan di RUNTIME (lihat .dockerignore utk yg
# DIKECUALIKAN -- data/raw_videos, venv, notebooks, dst TIDAK perlu masuk
# image, bikin bengkak tanpa guna: model hasil training sudah ada di models/,
# tidak perlu video mentah/notebook eksperimen ikut ter-bundle).
COPY app.py .
COPY src/ src/
COPY models/ models/
COPY assets/ assets/

EXPOSE 8501

# --server.address=0.0.0.0 WAJIB (bukan default localhost-only) -- supaya
# bisa diakses dari LUAR container (host laptop, HP di jaringan yg sama, atau
# lewat Cloudflare Tunnel yg jalan di HOST, bukan di dalam container ini).
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
