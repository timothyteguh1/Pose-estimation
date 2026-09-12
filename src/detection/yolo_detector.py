"""Wrapper YOLOv11 -- deteksi manusia (bab 6.1.1 & 8.3.3 proposal).

Proposal (ruang lingkup #9): pakai model pre-trained YOLOv11 (COCO), tidak
training/fine-tune sendiri -- beda dari Ko et al. yang fine-tune YOLOv5
mereka sendiri.

Cuma cari kelas 'person' (COCO kelas 0), ambil confidence tertinggi kalau
ada >1 orang (sistem tidak mendukung multi-user tracking). Ambang confidence
default 0.7, mengikuti konvensi Ko et al."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
DEFAULT_WEIGHTS = MODELS_DIR / "yolo11n.pt"  # varian "nano" -- paling cepat, cocok real-time
COCO_PERSON_CLASS = 0

# Rasio lebar:tinggi video training kita (1080x1920, portrait) -- target
# tetap di expand_bbox(), bukan dihitung dari kamera live (bisa beda rasio
# dari HP training).
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
    """Crop frame ke area bounding box -- pola Ko et al (crop, bukan mask:
    masking bikin visibility MediaPipe anjlok drastis, MediaPipe butuh
    konteks visual sekitar, bukan cuma area orangnya).

    padding_ratio: perluasan bbox di tiap sisi, proporsional ke lebar/tinggi
    box (0.2 = perbesar 20%). Crop ketat (Ko et al, padding=0) masih lebih
    rendah visibility-nya drpd frame utuh, krn MediaPipe Pose internal
    mengasumsikan ada margin di sekitar ROI orang -- padding adalah cara kita
    dapat efek margin itu tanpa retrain YOLO (proposal: pre-trained only).
    Divalidasi juga oleh eksperimen Ko et al sendiri (margin bbox lebih besar
    = mAP deteksi turun tapi akurasi landmark naik, pola yg sama kita temukan).

    Returns crop (ukuran beda dari frame asli -- sebesar bbox yg di-padding).
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = expand_bbox(bbox, padding_ratio, w, h)
    return frame[y1:y2, x1:x2].copy()  # .copy() -- slice numpy itu VIEW, jangan sampai
    # gambar skeleton di atasnya ikut mengubah `frame` asli secara diam-diam


def expand_bbox(bbox, padding_ratio, frame_w, frame_h, match_frame_aspect=True,
                 target_aspect_ratio=TRAINING_VIDEO_ASPECT_RATIO):
    """Hitung (x1,y1,x2,y2) setelah padding + clamp ke tepi frame -- dipisah
    dari crop_bbox() biar caller (live_pipeline.py) tau persis area yg
    diproses MediaPipe.

    match_frame_aspect: crop ke kotak orang mengubah rasio lebar:tinggi dari
    rasio frame asli ke rasio badan orang -- karena calculate_angle()
    (joint_angles.py) pakai koordinat MediaPipe mentah (x/y dinormalisasi
    terpisah ke lebar/tinggi gambar), zoom yg tidak seragam ini mendistorsi
    sudut walau posisi fisik orangnya tidak berubah. Fix: perluas sisi yg
    lebih pendek supaya rasio akhir crop sama dgn target_aspect_ratio (rasio
    video training tetap, bukan rasio kamera live yg mungkin beda)."""
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
