"""A4b: panjang window pada konfigurasi final (jawaban untuk "kenapa 0,5 detik").

Window 0,4 / 0,5 / 0,6 / 0,8 / 1,0 detik (12 / 15 / 18 / 24 / 30 frame pada 30 fps) dibangun dengan build_dataset produksi:
fitur final (koordinat relatif, flip hanya data latih) dan alokasi run train/test yang sama untuk semua panjang window
(seed 42). Model: RF final (100 pohon, min_samples_leaf 4, balanced, seed 42). Dinilai pada test split 70:30 dengan
F1 weighted (6 kelas per latihan). Tanpa LOPO.
Jumlah window turun saat window memanjang karena run yang lebih pendek dari window tidak menghasilkan window, dan
test setnya ikut berubah. Jeda sampai prediksi pertama sama dengan panjang window.

Dataset dibangun di folder temp (berkas proyek tidak diubah) dan disimpan di sana untuk run berikutnya; hapus folder
A4b_kerja di temp untuk mengulang dari nol. Window 0,5 detik ikut dibangun dan dicocokkan dengan split dan A1 produksi.
Jalankan dari akar proyek:  venv\\Scripts\\python docs\\pengujian\\A4_konfigurasi_final\\A4b_panjang_window.py
Sebagian saja (hasil digabung dari berkas sementara):  ... A4b_panjang_window.py --latihan squat --window 0.4 1.0
"""
import argparse
import contextlib
import io
import json
import sys
import tempfile
import time
import warnings
from pathlib import Path

import torch  # noqa: F401  (di Windows torch harus diimpor sebelum pandas)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from src.features import build_dataset as bd
from src.features.build_dataset import META_COLUMNS

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

EXP = REPO / "experimental_client_pipeline"
SPLIT_PROD = EXP / "data" / "windowed_features" / "splits"
KERJA = Path(tempfile.gettempdir()) / "A4b_kerja"
LATIHAN = {"squat": "Squat", "benchpress": "Bench press", "deadlift": "Deadlift"}
WINDOW = [0.4, 0.5, 0.6, 0.8, 1.0]
PRODUKSI = 0.5
VERSI = 2  # hasil sementara versi lain (misalnya yang masih memakai LOPO) dihitung ulang


def bangun(kode, w):
    akar = KERJA / f"w{w}"
    train, test = akar / "splits" / f"{kode}_train.csv", akar / "splits" / f"{kode}_test.csv"
    if train.exists() and test.exists():
        return train, test
    bd.EXTRACTED_DIR = EXP / "data" / "labeled"
    bd.WINDOWED_DIR = akar
    bd.SPLITS_DIR = akar / "splits"
    bd.MODELS_DIR = akar / "models"
    sys.argv = ["build_dataset.py", kode, "--window-sec", str(w), "--coord-relative", "--flip-augment"]
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        bd.main()
    if not (train.exists() and test.exists()):
        raise RuntimeError(f"dataset {kode} window {w} detik gagal dibangun. Akhir log:\n{log.getvalue()[-1500:]}")
    return train, test


def kom(x, d=4):
    return f"{x:.{d}f}".replace(".", ",")


def ringkas(h):
    return (f"[{LATIHAN[h['latihan']]}] window {kom(h['window_detik'], 1)} detik ({h['window_frame']} frame): "
            f"{h['jumlah_window']} window | 70:30 {kom(h['f1_weighted_70_30'])}")


def baca(kode, w):
    p = KERJA / f"hasil_{kode}_w{w}.json"
    if p.exists():
        h = json.load(open(p))
        if h.get("versi") == VERSI:
            return h
    return None


def nilai(kode, w):
    h = baca(kode, w)
    if h is not None:
        return h
    mulai = time.time()
    train, test = bangun(kode, w)
    tr, te = pd.read_csv(train), pd.read_csv(test)
    fitur = [c for c in tr.columns if c not in META_COLUMNS]
    asli = tr[~tr["window_id"].str.endswith("_flip")]

    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=4, class_weight="balanced",
                                random_state=42, n_jobs=-1).fit(tr[fitur], tr["class"])
    pred = rf.predict(te[fitur])
    f_split = f1_score(te["class"], pred, average="weighted", zero_division=0)

    h = {"versi": VERSI, "latihan": kode, "window_detik": w, "window_frame": int(tr["window_len_frames"].iloc[0]),
         "jumlah_window": len(asli) + len(te), "n_latih_asli": len(asli), "n_latih_dengan_flip": len(tr),
         "n_uji": len(te), "jumlah_fitur": len(fitur),
         "accuracy_70_30": float(accuracy_score(te["class"], pred)), "f1_weighted_70_30": float(f_split),
         "f1_macro_70_30": float(f1_score(te["class"], pred, average="macro", zero_division=0))}

    if abs(w - PRODUKSI) < 1e-9:  # bangunan ulang 0,5 detik harus sama dengan produksi
        a1 = json.load(open(EXP / "models" / f"{kode}_evaluation.json"))
        n_tr = len(pd.read_csv(SPLIT_PROD / f"{kode}_train.csv", usecols=["window_id"]))
        n_te = len(pd.read_csv(SPLIT_PROD / f"{kode}_test.csv", usecols=["window_id"]))
        h["cek_produksi"] = {"n_latih_sama": n_tr == len(tr), "n_uji_sama": n_te == len(te),
                             "f1_weighted_A1_json": a1["f1_weighted"], "selisih_f1_weighted": f_split - a1["f1_weighted"]}
    KERJA.mkdir(parents=True, exist_ok=True)
    tmp = KERJA / f"hasil_{kode}_w{w}.tmp"
    tmp.write_text(json.dumps(h, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(KERJA / f"hasil_{kode}_w{w}.json")  # ditulis utuh, tidak setengah jadi
    print(f"{ringkas(h)}  [{(time.time() - mulai) / 60:.1f} menit]", flush=True)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latihan", nargs="+", choices=list(LATIHAN), default=list(LATIHAN))
    ap.add_argument("--window", nargs="+", type=float, default=WINDOW)
    args = ap.parse_args()
    KERJA.mkdir(parents=True, exist_ok=True)

    mulai = time.time()
    for kode in args.latihan:
        for w in args.window:
            nilai(kode, w)

    semua_window = sorted(set(WINDOW) | set(args.window))
    hasil, kosong = [], []
    for kode, nama in LATIHAN.items():
        for w in semua_window:
            h = baca(kode, w)
            if h is not None:
                hasil.append(h)
            else:
                kosong.append(f"{nama} {w}")
    if kosong:
        print(f"\n[peringatan] belum ada hasil untuk: {', '.join(kosong)}")

    print("\nHasil per kombinasi")
    for h in hasil:
        print(ringkas(h))

    tabel = pd.DataFrame([{"Latihan": LATIHAN[h["latihan"]], "Window_detik": str(h["window_detik"]).replace(".", ","),
                           "Window_frame": h["window_frame"], "Jumlah_window": h["jumlah_window"],
                           "F1w_70_30": h["f1_weighted_70_30"]} for h in hasil])
    tabel.to_csv(HERE / "A4b_tabel.csv", index=False, sep=";", decimal=",", float_format="%.4f")
    tampil = tabel.copy()
    tampil["F1w_70_30"] = tabel["F1w_70_30"].map(kom)
    print("\n" + tampil.to_string(index=False))

    print("\nWindow dengan F1 weighted 70:30 tertinggi per latihan")
    for kode, nama in LATIHAN.items():
        hk = [h for h in hasil if h["latihan"] == kode]
        if hk:
            b = max(hk, key=lambda h: h["f1_weighted_70_30"])
            print(f"  {nama:12s} {kom(b['window_detik'], 1)} detik ({kom(b['f1_weighted_70_30'])})")

    print("\nCek window 0,5 detik terhadap produksi (split dan A1)")
    for h in hasil:
        c = h.get("cek_produksi")
        if c:
            print(f"  {LATIHAN[h['latihan']]:12s} latih sama: {c['n_latih_sama']} | uji sama: {c['n_uji_sama']} | "
                  f"F1 A4b {kom(h['f1_weighted_70_30'])} lawan A1 {kom(c['f1_weighted_A1_json'])} "
                  f"(selisih {c['selisih_f1_weighted']:+.6f})".replace(".", ","))

    with open(HERE / "A4b_detail.json", "w", encoding="utf-8") as f:
        json.dump({"keterangan": "F1 weighted pada test split 70:30, 6 kelas per latihan; flip hanya pada data latih; "
                                 "alokasi run train/test sama untuk semua panjang window; tanpa LOPO",
                   "hasil": hasil}, f, ensure_ascii=False, indent=2)
    print(f"\nTersimpan: {HERE / 'A4b_tabel.csv'} dan A4b_detail.json  (run ini {(time.time() - mulai) / 60:.1f} menit)")


if __name__ == "__main__":
    main()
