"""Shared I/O helpers."""
import os
import tempfile
from pathlib import Path

import cv2


def open_video_capture(source):
    """cv2.VideoCapture(source) + CAP_PROP_ORIENTATION_AUTO=1.

    Sebagian video HP nyimpen frame MENTAH dalam orientasi sensor asli
    (landscape) + tag metadata rotasi (mis. 90 derajat) yg baru diterapkan
    pas DITAMPILKAN di pemutar video normal -- cv2.VideoCapture.read() BAWAAN
    TIDAK menerapkan rotasi itu (cuma baca tag-nya doang), jadi frame yg
    dikembalikan tetap MENYAMPING (landscape) walau videonya keliatan portrait
    kalau diputar normal. Ditemukan langsung dari 2 video baru (benchpress,
    direkam 5 Sept) yg punya CAP_PROP_ORIENTATION_META=90, beda
    dari video lama yg videonya sudah portrait dari sensornya (metadata=0,
    tidak perlu apa-apa). CAP_PROP_ORIENTATION_AUTO=1 bikin OpenCV menerapkan
    rotasi itu sendiri sebelum frame dikembalikan -- diverifikasi langsung:
    tanpa ini frame.shape=(1080,1920,3) [landscape, SALAH utk video yg
    metadatanya bilang portrait], dengan ini frame.shape=(1920,1080,3)
    [portrait, BENAR]. Pakai fungsi ini di SEMUA tempat yg buka file video
    (bukan live webcam -- kamera live umumnya tidak punya metadata rotasi
    begini, tapi properti ini aman di-set juga, no-op kalau tidak relevan).
    """
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return cap


def atomic_write_csv(df, path):
    """Write a DataFrame to CSV atomically (temp file + os.replace).

    Prevents corruption if the target file is open/locked by another process
    (e.g. an editor/IDE previewing it) at the exact moment of writing --
    os.replace() is a single filesystem operation, unlike writing directly
    into the target file which can interleave with a concurrent reader/writer.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.stem}.", suffix=".tmp.csv"
    )
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
