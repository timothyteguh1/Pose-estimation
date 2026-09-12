# Dockerfile aplikasi web (bab 6.4 proposal, client-server) -- membungkus
# app.py + dependency-nya jadi 1 image, tanpa install Python/library manual.
# Murni packaging/reproducibility (bukan alat percepat compute).

FROM python:3.10-slim

WORKDIR /app

# Library sistem yg dibutuhkan opencv-python & mediapipe/av (codec video).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install dependency dulu (sebelum copy source code) -- biar Docker bisa
# pakai cache layer ini kalau cuma source code yg berubah.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.0 torchvision==0.29.0 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt

# Copy hanya yg dibutuhkan runtime (lihat .dockerignore utk yg dikecualikan).
COPY app.py .
COPY src/ src/
COPY models/ models/
COPY assets/ assets/

EXPOSE 8501

# --server.address=0.0.0.0 wajib supaya bisa diakses dari luar container.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
