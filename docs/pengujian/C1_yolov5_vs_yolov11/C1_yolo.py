"""C1: perbandingan YOLO11n (dipakai aplikasi) dan YOLOv5nu (pembanding) untuk deteksi orang, DI BROWSER.

Sesuai janji proposal (ruang lingkup poin 13): waktu inferensi dan ketepatan deteksi manusia.
Yang berbeda hanya model. Semua lain sama: frame yang persis sama, browser yang sama, runtime yang sama (ONNX Runtime Web
1.30.0, webgpu atau wasm untuk keduanya), kode deteksi aplikasi yang sama (yolo.js), input 640, confidence 0,7, NMS 0,45,
frame diperkecil ke sisi panjang 720 seperti di aplikasi, pemanasan dulu, urutan model digilir tiap frame.

Alur: skrip ini (1) mengekspor YOLOv5nu ke ONNX dengan pengaturan yang sama dengan model aplikasi bila belum ada,
(2) mengambil 1 frame tiap beberapa detik dari 81 video dataset (video HEVC tidak selalu bisa diputar browser, jadi frame
diambil dengan OpenCV), (3) menyalakan server lokal dan membuka halaman uji c1_browser.html, (4) menunggu hasil dari
browser, lalu mencetak tabel di terminal dan menyimpan CSV, JSON, dan log.

Waktu inferensi = waktu satu panggilan detect() di browser (praproses + inferensi + pascaproses), seperti "YOLO nyata" di
aplikasi. Ketepatan deteksi = laju deteksi (persen frame yang orangnya terdeteksi pada confidence 0,7), bukan precision
atau recall, karena kotak sebenarnya tidak ada. mAP50-95 resmi (COCO, 80 kelas) dari dokumentasi Ultralytics dicantumkan
sebagai pembanding; angka itu bukan hasil run ini.
Hanya membaca video dan model; keluaran: C1_tabel.csv, C1_detail.json, C1_terminal.log.
Jalankan dari akar proyek:  venv\\Scripts\\python docs\\pengujian\\C1_yolov5_vs_yolov11\\C1_yolo.py
"""
import argparse
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
FRONTEND = REPO / "experimental_client_pipeline" / "webapp" / "frontend"

MODEL_A, MODEL_B = "YOLO11n (dipakai)", "YOLOv5nu (pembanding)"
ONNX_A = FRONTEND / "models" / "yolo11n_640.onnx"
PT_B = HERE / "yolov5nu.pt"
ONNX_B = HERE / "yolov5nu_640.onnx"
HALAMAN = HERE / "c1_browser.html"
YOLO_JS = FRONTEND / "yolo.js"
CONF, IOU, IMGSZ = 0.7, 0.45, 640
BROWSER_EXE = {"chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
               "edge": r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"}

# Angka resmi (bukan hasil run ini): dokumentasi Ultralytics, COCO (80 kelas), input 640.
REFERENSI = {MODEL_A: {"mAP50_95": 39.5, "CPU_ONNX_ms": 56.1, "GFLOPs": 6.5, "parameter_juta": 2.6},
             MODEL_B: {"mAP50_95": 34.3, "CPU_ONNX_ms": 73.6, "GFLOPs": 7.7, "parameter_juta": 2.6}}
SUMBER_REFERENSI = "docs.ultralytics.com/models/yolo11 dan docs.ultralytics.com/models/yolov5, diakses 5 Okt 2026"

HASIL = {"data": None, "terhubung": False}
SELESAI = threading.Event()
FRAMES_DIR = None


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, s):
        for f in self.files:
            f.write(s)

    def flush(self):
        for f in self.files:
            f.flush()


def kom(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def nama_cpu():
    try:
        import winreg
        kunci = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(kunci, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor() or "tidak diketahui"


def daftar_video(maks):
    vids = []
    for latihan in ("squat", "benchpress", "deadlift"):
        vids += sorted(glob.glob(str(REPO / "data" / "raw_videos" / latihan / "p*" / "*" / "*.mp4")))
    return vids[:maks] if maks else vids


def ambil_frame(tugas):
    """Satu proses: ambil frame tiap `detik` detik dari satu video, simpan sebagai JPEG ukuran asli."""
    nomor, path, detik, folder = tugas
    from src.io_utils import open_video_capture  # orientasi video HP diterapkan seperti di pipeline
    cap = open_video_capture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    durasi = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    sasaran, t = [], detik / 2.0
    while t < durasi - 0.05:
        sasaran.append(int(round(t * fps)))
        t += detik
    hasil, i, k = [], 0, 0
    while k < len(sasaran):
        if not cap.grab():
            break
        if i == sasaran[k]:
            ok, frame = cap.retrieve()
            if ok:
                nama = f"v{nomor:03d}_{k:03d}.jpg"
                cv2.imwrite(str(Path(folder) / nama), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                hasil.append({"video": Path(path).name, "t": round(i / fps, 2), "file": nama,
                              "w": int(frame.shape[1]), "h": int(frame.shape[0])})
            k += 1
        i += 1
    cap.release()
    return hasil


def ekstrak(videos, detik, folder):
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    tugas = [(i, p, detik, str(folder)) for i, p in enumerate(videos)]
    n_proses = max(2, min(6, (os.cpu_count() or 4) // 2))
    frames = []
    with Pool(n_proses) as pool:
        for i, hasil in enumerate(pool.imap(ambil_frame, tugas), 1):
            frames.extend(hasil)
            if i % 10 == 0 or i == len(videos):
                print(f"  frame diambil dari {i}/{len(videos)} video ({len(frames)} frame)", flush=True)
    for k, f in enumerate(frames):
        f["id"] = k
    (folder / "manifest.json").write_text(json.dumps({"detik": detik, "frames": frames}), encoding="utf-8")
    return frames


def siapkan_onnx_b():
    if ONNX_B.exists():
        return
    print("Mengekspor YOLOv5nu ke ONNX dengan pengaturan yang sama dengan model aplikasi (sekali saja) ...", flush=True)
    from ultralytics import YOLO
    keluaran = YOLO(str(PT_B)).export(format="onnx", imgsz=IMGSZ, simplify=True, dynamic=False, nms=False, verbose=False)
    shutil.move(str(keluaran), str(ONNX_B))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _kirim(self, kode, tipe, isi):
        self.send_response(kode)
        self.send_header("Content-Type", tipe)
        self.send_header("Content-Length", str(len(isi)))
        # sama dengan server aplikasi (app.py)
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(isi)

    def do_GET(self):
        p = self.path.split("?")[0]
        peta = {"/": (HALAMAN, "text/html; charset=utf-8"), "/c1_browser.html": (HALAMAN, "text/html; charset=utf-8"),
                "/yolo.js": (YOLO_JS, "text/javascript; charset=utf-8"),
                "/models/yolo11n_640.onnx": (ONNX_A, "application/octet-stream"),
                "/models/yolov5nu_640.onnx": (ONNX_B, "application/octet-stream")}
        if p in peta:
            file, tipe = peta[p]
            if p in ("/", "/c1_browser.html"):
                HASIL["terhubung"] = True
        elif p.startswith("/frames/"):
            nama = p[len("/frames/"):]
            if not nama or "/" in nama or "\\" in nama or ".." in nama:
                return self._kirim(403, "text/plain", b"forbidden")
            file = FRAMES_DIR / nama
            tipe = "image/jpeg" if nama.endswith(".jpg") else "application/json"
        else:
            return self._kirim(404, "text/plain", b"not found")
        if not file.is_file():
            return self._kirim(404, "text/plain", b"not found")
        self._kirim(200, tipe, file.read_bytes())

    def do_POST(self):
        panjang = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(panjang) or b"{}")
        except Exception:
            data = {}
        if self.path == "/progress":
            print(f"  browser: {data.get('selesai')}/{data.get('total')} frame | rata-rata sementara "
                  f"YOLO11n {kom(data.get('rataA', 0))} ms, YOLOv5nu {kom(data.get('rataB', 0))} ms", flush=True)
        elif self.path == "/hasil":
            HASIL["data"] = data
            SELESAI.set()
        self._kirim(200, "application/json", b"{}")


def ringkas(catat, kunci):
    ms = np.array([r[kunci]["ms"] for r in catat])
    det = [r[kunci] for r in catat if r[kunci]["det"]]
    return {"n_frame": len(catat), "n_terdeteksi": len(det),
            "laju_deteksi_persen": 100.0 * len(det) / len(catat),
            "conf_rata2": float(np.mean([d["conf"] for d in det])) if det else float("nan"),
            "waktu_rata2_ms": float(ms.mean()), "waktu_median_ms": float(np.median(ms)),
            "waktu_p95_ms": float(np.percentile(ms, 95)), "waktu_std_ms": float(ms.std())}


def versi_browser(ua):
    m = re.search(r"(Edg|Chrome|Firefox)/([\d.]+)", ua or "")
    if not m:
        return ua or "tidak diketahui"
    return f"{'Edge' if m.group(1) == 'Edg' else m.group(1)} {m.group(2).split('.')[0]}"


def buka_browser(url, pilihan):
    if pilihan in BROWSER_EXE and Path(BROWSER_EXE[pilihan]).exists():
        subprocess.Popen([BROWSER_EXE[pilihan], url])
    else:
        webbrowser.open(url)


def main():
    ap = argparse.ArgumentParser(description="C1: YOLO11n vs YOLOv5nu di browser")
    ap.add_argument("--detik", type=float, default=2.0, help="ambil 1 frame tiap N detik (default 2)")
    ap.add_argument("--maks-video", type=int, default=0, help="batasi jumlah video (0 = semua 81)")
    ap.add_argument("--out", default=str(HERE), help="folder keluaran (default folder ini)")
    ap.add_argument("--port", type=int, default=8701)
    ap.add_argument("--browser", choices=["default", "chrome", "edge"], default="default")
    ap.add_argument("--backend", choices=["auto", "wasm"], default="auto",
                    help="auto = webgpu bila ada (seperti aplikasi); wasm = paksa CPU (opsional, run kedua)")
    ap.add_argument("--no-open", action="store_true", help="jangan buka browser otomatis (buka URL sendiri)")
    ap.add_argument("--simpan-frame", action="store_true", help="jangan hapus folder frame sementara")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "C1_terminal.log", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    try:
        sys.__stdout__.reconfigure(errors="replace")
    except Exception:
        pass
    try:
        selesai_main(args, out, time.time())
    finally:
        sys.stdout = sys.__stdout__
        log.close()


def selesai_main(args, out, mulai):
    global FRAMES_DIR
    import torch  # noqa: F401  (di Windows torch harus diimpor sebelum pandas)
    import pandas as pd

    cpu = nama_cpu()
    print(f"PC: {cpu} | {os.cpu_count()} thread | Python {platform.python_version()}", flush=True)
    for f in (ONNX_A, YOLO_JS, HALAMAN):
        if not f.exists():
            print(f"Berkas tidak ditemukan: {f}")
            return
    if not ONNX_B.exists() and not PT_B.exists():
        print("Bobot YOLOv5nu (yolov5nu.pt) tidak ada di folder ini.")
        return
    siapkan_onnx_b()

    videos = daftar_video(args.maks_video)
    print(f"{len(videos)} video dataset; 1 frame tiap {kom(args.detik)} detik; confidence {kom(CONF)}, "
          f"IoU NMS {kom(IOU, 2)}, input {IMGSZ}", flush=True)
    FRAMES_DIR = Path(tempfile.gettempdir()) / "C1_frames"
    frames = ekstrak(videos, args.detik, FRAMES_DIR)
    print(f"{len(frames)} frame siap ({(time.time() - mulai) / 60:.1f} menit sejak mulai)", flush=True)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{args.port}/c1_browser.html" + ("?backend=wasm" if args.backend == "wasm" else "")
    print(f"\nServer lokal siap: {url}", flush=True)
    if args.no_open:
        print("Buka alamat itu di browser (tab harus tetap terlihat sampai selesai).", flush=True)
    else:
        print("Membuka browser. Biarkan tab terlihat sampai selesai (jangan diminimalkan).", flush=True)
        buka_browser(url, args.browser)

    t_tunggu = time.time()
    peringatan = False
    try:
        while not SELESAI.wait(1.0):
            if not HASIL["terhubung"] and not peringatan and time.time() - t_tunggu > 60:
                print(f"Browser belum terhubung. Buka manual: {url}", flush=True)
                peringatan = True
            if time.time() - t_tunggu > 3 * 3600:
                print("Menunggu terlalu lama, dihentikan.")
                break
    except KeyboardInterrupt:
        print("\nDihentikan (Ctrl+C).")
    finally:
        server.shutdown()
        if not args.simpan_frame:
            shutil.rmtree(FRAMES_DIR, ignore_errors=True)

    data = HASIL["data"]
    if not data:
        print("Tidak ada hasil dari browser.")
        return
    if "error" in data:
        print(f"Browser melaporkan error: {data['error']}")
        return

    catat = data["catat"]
    meta = {f["id"]: f for f in frames}
    a, b = ringkas(catat, "A"), ringkas(catat, "B")
    semua = {MODEL_A: a, MODEL_B: b}
    selisih = np.array([r["A"]["ms"] - r["B"]["ms"] for r in catat])
    a_lebih_cepat = float(np.mean(selisih < 0)) * 100.0
    gpu = data.get("gpu") or {}
    gpu_teks = " ".join(str(gpu.get(k)) for k in ("vendor", "architecture", "description") if gpu.get(k)) or "tidak tersedia"
    browser = versi_browser(data.get("userAgent"))

    tabel = pd.DataFrame([{
        "Model": nama,
        "Waktu_rata2_ms": kom(semua[nama]["waktu_rata2_ms"]),
        "Laju_deteksi_persen": kom(semua[nama]["laju_deteksi_persen"]),
        "mAP50_95_resmi": kom(REFERENSI[nama]["mAP50_95"]),
    } for nama in (MODEL_A, MODEL_B)])
    tabel.to_csv(out / "C1_tabel.csv", index=False, sep=";", encoding="utf-8-sig")

    print(f"\nLingkungan uji: {browser} | backend {data.get('backend')} | GPU: {gpu_teks} | ONNX Runtime Web {data.get('ortWeb')} "
          f"| cross-origin isolated: {data.get('crossOriginIsolated')}")
    if data.get("catatan"):
        print(f"Catatan browser: {data['catatan']}")
    print(f"\nDeteksi orang di browser, {len(catat)} frame dari {len(videos)} video dataset "
          f"(input {IMGSZ}, confidence {kom(CONF)}, frame sama untuk kedua model)")
    with pd.option_context("display.width", 200):
        print(tabel.to_string(index=False))
    print("mAP50_95_resmi = angka dokumentasi Ultralytics (COCO, 80 kelas), bukan hasil run ini.")

    print(f"\nSelisih (YOLO11n dikurangi YOLOv5nu): waktu rata-rata {kom(a['waktu_rata2_ms'] - b['waktu_rata2_ms'])} ms "
          f"(berpasangan per frame {kom(float(selisih.mean()))} ms) | YOLO11n lebih cepat pada {kom(a_lebih_cepat)}% frame | "
          f"laju deteksi {kom(a['laju_deteksi_persen'] - b['laju_deteksi_persen'])} poin")
    print("\nRincian (cadangan, tidak masuk tabel)")
    for nama in (MODEL_A, MODEL_B):
        s = semua[nama]
        print(f"  {nama:23s} waktu rata-rata {kom(s['waktu_rata2_ms'])} ms (median {kom(s['waktu_median_ms'])}, "
              f"p95 {kom(s['waktu_p95_ms'])}, simpangan baku {kom(s['waktu_std_ms'])}) | laju deteksi {kom(s['laju_deteksi_persen'])}% "
              f"({s['n_terdeteksi']}/{s['n_frame']}) | confidence rata-rata {kom(s['conf_rata2'], 3)}")
    if data.get("gagal_muat_frame"):
        print(f"  Frame gagal dimuat browser: {data['gagal_muat_frame']}")

    with open(out / "C1_detail.json", "w", encoding="utf-8") as f:
        json.dump({
            "keterangan": "Uji di browser: frame sama untuk kedua model, backend sama, kode deteksi aplikasi (yolo.js). Waktu = satu panggilan "
                          "detect() (praproses + inferensi + pascaproses). Laju deteksi = persen frame dengan minimal satu orang terdeteksi pada "
                          "confidence 0,7 (bukan precision atau recall). mAP resmi dari dokumentasi Ultralytics (80 kelas COCO).",
            "pengaturan": {"video": len(videos), "detik_antar_frame": args.detik, "frame": len(catat), "confidence": CONF,
                           "iou_nms": IOU, "input": IMGSZ, "sisi_panjang_maks_px": 720,
                           "model": {MODEL_A: {"berkas": str(ONNX_A.relative_to(REPO)), "ukuran_byte": ONNX_A.stat().st_size},
                                     MODEL_B: {"berkas": str(ONNX_B.relative_to(REPO)), "ukuran_byte": ONNX_B.stat().st_size}}},
            "lingkungan": {"pc_cpu": cpu, "thread": os.cpu_count(), "browser": browser, "user_agent": data.get("userAgent"),
                           "backend": data.get("backend"), "catatan_backend": data.get("catatan"), "gpu": gpu,
                           "onnxruntime_web": data.get("ortWeb"), "cross_origin_isolated": data.get("crossOriginIsolated"),
                           "hardwareConcurrency_browser": data.get("hardwareConcurrency")},
            "referensi_resmi": {**REFERENSI, "sumber": SUMBER_REFERENSI},
            "hasil": semua,
            "berpasangan": {"selisih_rata2_ms_A_kurang_B": float(selisih.mean()), "YOLO11n_lebih_cepat_persen_frame": a_lebih_cepat},
            "gagal_muat_frame": data.get("gagal_muat_frame", 0),
            "lama_di_browser_detik": data.get("lama_detik"),
            "per_frame": [{"id": r["id"], "video": meta[r["id"]]["video"], "t": meta[r["id"]]["t"],
                           "A_ms": r["A"]["ms"], "B_ms": r["B"]["ms"], "A_terdeteksi": r["A"]["det"], "B_terdeteksi": r["B"]["det"],
                           "A_conf": r["A"]["conf"], "B_conf": r["B"]["conf"]} for r in catat],
        }, f, ensure_ascii=False, indent=2)
    print(f"\nTersimpan: {out / 'C1_tabel.csv'}, C1_detail.json, C1_terminal.log  (run ini {(time.time() - mulai) / 60:.1f} menit)")


if __name__ == "__main__":
    main()
