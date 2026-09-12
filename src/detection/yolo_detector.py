"""Wrapper YOLOv11 -- deteksi manusia (bab 6.1.1 & 8.3.3 proposal, tahap
IMPLEMENTASI, bukan preprocessing/training -- lihat diskusi proyek).

Proposal eksplisit (ruang lingkup #9): "Implementasi sistem tidak membangun
model pose estimation dari awal, tetapi menggunakan model **pre-trained**
YOLOv11 dan MediaPipe yang sudah tersedia." -- jadi pakai bobot YOLOv11
resmi (dilatih di dataset COCO oleh Ultralytics), TIDAK training/fine-tune
model sendiri. Ini beda dari Ko et al. yang justru fine-tune YOLOv5 mereka
sendiri (paper mereka sebut kumpulkan >4800 gambar tambahan khusus buat
melatih ulang deteksinya) -- proposal kita SENGAJA lebih sederhana di sini,
sesuai kalimat ruang lingkup #9 di atas.

Cuma cari kelas 'person' (kelas ke-0 di COCO) -- proposal: "mengidentifikasi
keberadaan manusia", bukan objek lain. Kalau ada >1 orang terdeteksi, ambil
yang confidence-nya PALING TINGGI saja -- sesuai batasan sistem (ruang
lingkup #17: "Sistem tidak mendukung multi-user tracking dalam satu frame").

Ambang confidence default 0.7, mengikuti pola Ko et al. (`Streamlit.py`:
`if conf >= 0.7`) -- angka itu sendiri bukan hasil training kita, cuma
konvensi ambang kepercayaan deteksi yang wajar dipakai ulang (beda dari
angka rep-counting yang memang harus dari data kita sendiri, karena itu
langsung soal karakteristik gerakan kita).
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
DEFAULT_WEIGHTS = MODELS_DIR / "yolo11n.pt"  # varian "nano" -- paling cepat, cocok real-time
COCO_PERSON_CLASS = 0

# Rasio lebar:tinggi SEMUA video training kita (dicek langsung: 1080x1920,
# portrait, di keempat video benchpress). Dipakai sbg target TETAP di
# expand_bbox() (match_frame_aspect) -- SENGAJA bukan diambil dari frame yg
# lagi diproses saat itu (frame_w/frame_h runtime), karena kalau live nanti
# pakai kamera lain (webcam laptop landscape, dll, rasio beda dari HP yg
# dipakai rekam training), "menyamakan ke rasio kamera baru itu sendiri"
# TIDAK ada gunanya -- yang perlu disamakan itu ke rasio yg DIPELAJARI MODEL
# (video training), bukan ke rasio device apapun yg kebetulan dipakai live.
TRAINING_VIDEO_ASPECT_RATIO = 1080 / 1920


class PersonDetector:
    def __init__(self, weights_path=DEFAULT_WEIGHTS, confidence_threshold=0.7):
        from ultralytics import YOLO  # import lokal -- biar modul lain yg tidak butuh YOLO tetap ringan
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(weights_path))  # auto-download bobot pretrained kalau belum ada
        self.confidence_threshold = confidence_threshold

    def detect(self, frame_bgr):
        """Deteksi 1 orang dengan confidence tertinggi di 1 frame (BGR, format OpenCV).

        Returns (x1, y1, x2, y2) piksel (int) kalau ketemu orang dgn confidence
        >= threshold, else None (tidak ada orang terdeteksi -- caller boleh
        skip frame ini, lihat live_pipeline.py).
        """
        results = self.model.predict(frame_bgr, verbose=False)[0]
        best_box, best_conf = None, -1.0
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls != COCO_PERSON_CLASS or conf < self.confidence_threshold:
                continue
            if conf > best_conf:
                best_conf = conf
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                best_box = (int(x1), int(y1), int(x2), int(y2))
        return best_box


def crop_bbox(frame, bbox, padding_ratio=0.0):
    """Crop frame ke area bounding box -- dasarnya PERSIS pola Ko et al.
    (`Streamlit.py`: `object_frame = frame[c1[1]:c2[1], c1[0]:c2[0]]`, yaitu
    `padding_ratio=0.0`).

    RIWAYAT #1 -- MASK vs CROP: sempat coba pendekatan MASK (hitamkan di luar
    bbox, TANPA ubah ukuran frame) supaya rasio/skala koordinat MediaPipe
    konsisten dgn training (frame utuh) -- alasannya masuk akal secara
    matematis (angle dihitung dari rasio vektor, rentan distorsi kalau rasio
    lebar:tinggi berubah). TAPI dites langsung ke data kita: masking bikin
    visibility MediaPipe ANJLOK drastis (contoh nyata: left_wrist_v 0.93 di
    frame utuh -> 0.21 di frame masked, di frame yg SAMA) -- MediaPipe
    ternyata butuh KONTEKS VISUAL sekitar, bukan cuma area orangnya, buat
    estimasi confidence yang akurat. Crop (background dibuang total, bukan
    dihitamkan) terbukti JAUH lebih baik. Jadi dipilih CROP, ikut Ko et al.

    RIWAYAT #2 -- padding_ratio: crop ketat Ko et al. (padding_ratio=0.0)
    masih lebih rendah visibility-nya drpd frame utuh, walau jauh lbh baik
    drpd mask. Alasannya: MediaPipe Pose itu 2 tahap internal -- (1) detector
    ringan cari ROI orang lalu SENGAJA menambah margin/padding di sekitarnya,
    (2) model landmark baru jalan di ROI yg sudah dikasih margin itu, dan
    modelnya dikalibrasi dgn asumsi ada margin tsb. Kotak YOLO itu ketat
    (buat deteksi objek, bukan dirancang utk pose estimation) -- TANPA
    margin.

    PENTING -- ini TERNYATA BUKAN penyimpangan dari Ko et al., malah
    divalidasi LANGSUNG oleh eksperimen mereka sendiri (paper mereka, bagian
    "Exercise Object Detection Model using YOLOv5"): mereka melatih 2 model
    YOLOv5 dgn data training yg dilabeli beda -- model 1 pakai margin bbox
    PALING MINIM ("lowest possible margin from head to toe"), model 2 pakai
    "sufficient margin on all sides". Hasilnya: model 2 justru mAP@0.5-0.95
    LEBIH RENDAH (0.705 vs 0.843 model 1) TAPI "higher accuracy was achieved
    for joint landmark estimation" -- PERSIS pola yg kita temukan sendiri
    (window_survive naik drastis 19.1% -> 34.7% pakai padding, walau bukan
    metrik yg sama). Ko et al mengatasi ini dgn RETRAIN ulang YOLOv5 mereka
    pakai data berlabel margin -- opsi itu TERTUTUP buat kita krn proposal
    eksplisit: "tidak membangun model pose estimation dari awal, tetapi
    menggunakan model pre-trained YOLOv11" (TIDAK fine-tune/retrain). Jadi
    `padding_ratio` di sini adalah cara ALGORITMIK kita mencapai EFEK YANG
    SAMA (margin di sekitar deteksi) TANPA retrain -- tujuannya divalidasi
    Ko et al sendiri, mekanismenya beda krn keterbatasan scope proposal kita
    (pre-trained only). TIDAK mengubah training (training tidak pakai YOLO
    sama sekali -- MediaPipe jalan di frame utuh, lihat build_dataset.py) --
    murni penyesuaian tahap IMPLEMENTASI, mengurangi gap train-vs-live (crop
    makin dekat ke "frame utuh" yg dilihat saat training).

    padding_ratio: perluasan bbox di tiap sisi, proporsional ke lebar/tinggi
    box itu sendiri (0.2 = perbesar 20% dari lebar & tinggi box, masing2 sisi).

    Returns crop (ukuran BEDA dari frame asli -- sebesar bounding box yg
    sudah di-padding).
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = expand_bbox(bbox, padding_ratio, w, h)
    return frame[y1:y2, x1:x2].copy()  # .copy() -- slice numpy itu VIEW, jangan sampai
    # gambar skeleton di atasnya ikut mengubah `frame` asli secara diam-diam


def expand_bbox(bbox, padding_ratio, frame_w, frame_h, match_frame_aspect=True,
                 target_aspect_ratio=TRAINING_VIDEO_ASPECT_RATIO):
    """Hitung koordinat (x1,y1,x2,y2) SETELAH padding + clamp ke tepi frame --
    logika inti yg dipakai crop_bbox(), dipisah biar caller (live_pipeline.py,
    preview tools) bisa tau persis area mana yg BENERAN diproses MediaPipe,
    mis. buat gambar kotaknya atau nempel crop balik ke frame asli.

    RIWAYAT #3 -- match_frame_aspect (NOVELTY kita, penyempurnaan #2 di atas):
    padding_ratio SENDIRI ternyata belum cukup -- dibuktikan lewat perbandingan
    langsung angle offline vs live (frame yg SAMA): drift 0.2-12.8 derajat
    tetap ada meski padding sudah 0.75. Akar masalahnya BUKAN soal jumlah
    konteks visual, tapi soal RASIO lebar:tinggi crop yg BEDA dari rasio
    frame video asli. `calculate_angle()` (joint_angles.py) pakai koordinat
    MediaPipe MENTAH (x dinormalisasi ke LEBAR gambar, y ke TINGGI gambar,
    TERPISAH, tanpa koreksi rasio) -- rumus cosine-nya scale-invariant HANYA
    kalau x & y discale dgn faktor SAMA (zoom seragam). Crop ke kotak orang
    (apalagi dipadding) mengubah rasio lebar:tinggi dari rasio frame asli
    (mis. 16:9) ke rasio badan orang (biasa lebih tinggi drpd lebar) --
    zoom-nya jadi TIDAK SERAGAM antara x & y, mendistorsi sudut walau posisi
    fisik orangnya di dunia nyata sama sekali tidak berubah.

    Fix: SETELAH padding, perluas sisi yg lebih pendek (lebar ATAU tinggi)
    supaya rasio akhir crop PERSIS sama dgn rasio VIDEO TRAINING (bukan
    rasio frame_w/frame_h yg lagi diproses saat itu!) -- jadi normalisasi
    x:y MediaPipe kembali konsisten dgn kondisi training. SENGAJA pakai
    `target_aspect_ratio` tetap (default TRAINING_VIDEO_ASPECT_RATIO,
    1080/1920), BUKAN dihitung dari frame_w/frame_h runtime -- kalau live
    nanti pakai kamera lain yg rasionya beda dari HP training (mis. webcam
    laptop landscape), menyamakan ke rasio KAMERA ITU SENDIRI tidak ada
    gunanya; yg perlu disamakan itu ke rasio yg DIPELAJARI MODEL (video
    training), berapa pun rasio kamera live-nya. TIDAK mengubah
    training/preprocessing sama sekali -- murni penyempurnaan cara crop
    di tahap implementasi."""
    x1, y1, x2, y2 = bbox
    if padding_ratio > 0:
        pad_x = int((x2 - x1) * padding_ratio)
        pad_y = int((y2 - y1) * padding_ratio)
        x1, y1, x2, y2 = x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y

    if match_frame_aspect:
        box_w, box_h = x2 - x1, y2 - y1
        if box_w > 0 and box_h > 0:
            current_ratio = box_w / box_h
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if current_ratio < target_aspect_ratio:
                new_w = box_h * target_aspect_ratio  # box terlalu "kurus" -- lebarkan
                x1, x2 = cx - new_w / 2, cx + new_w / 2
            elif current_ratio > target_aspect_ratio:
                new_h = box_w / target_aspect_ratio  # box terlalu "gemuk" -- tinggikan
                y1, y2 = cy - new_h / 2, cy + new_h / 2

    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(frame_w, int(round(x2))), min(frame_h, int(round(y2)))
    return x1, y1, x2, y2
