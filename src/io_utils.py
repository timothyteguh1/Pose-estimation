"""Shared I/O helpers."""
import os
import tempfile
from pathlib import Path

import cv2


def open_video_capture(source):
    """cv2.VideoCapture(source) + CAP_PROP_ORIENTATION_AUTO=1.

    Sebagian video HP nyimpen frame mentah dalam orientasi sensor asli
    (landscape) + tag metadata rotasi -- cv2.VideoCapture.read() bawaan
    tidak menerapkan rotasi itu (cuma baca tag-nya), jadi frame yg
    dikembalikan tetap landscape walau videonya portrait kalau diputar
    normal. CAP_PROP_ORIENTATION_AUTO=1 bikin OpenCV menerapkan rotasi
    sendiri sebelum frame dikembalikan. Pakai fungsi ini di semua tempat
    yg buka file video (aman jadi no-op utk live webcam tanpa metadata
    rotasi)."""
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
