"""B1: akurasi hitungan repetisi (MAE Pers. 12) untuk tiga set parameter, pada data tuning.

Data: 81 video P1-P3 (27 per latihan) dari CSV berlabel di experimental_client_pipeline/data/labeled. CSV itu sudah
berisi landmark tiap frame dan label fase, jadi tidak ada ekstraksi ulang. Hitungan repetisi memakai OnlineRepCounter
produksi (smoothing 5, cosine 0,70). Jalur live di browser (YOLO ONNX) diuji pada E2, bukan di sini.

Hitungan sebenarnya C = jumlah segmen konsentrik pada kolom phase (label fase yang dikoreksi manual).
Seluruh video dihitung apa adanya, dari detik 0 sampai akhir: tanpa countdown dan tanpa pemotongan.
MAE (Pers. 12) = rata-rata |C_sistem - C| / C per video, per latihan; tidak ada rata-rata antar latihan.
Semua video ini juga dipakai saat tuning parameter, jadi hasilnya "pada data tuning" (optimistik), bukan uji independen.

Tabel utama: tiga set parameter x tiga latihan. Di JSON juga ada rincian per video, MAE P1 lawan P2+P3, dan tiga protokol
pembanding (lewati 5 detik, countdown 5 detik hanya bila persiapan lebih dari 5 detik, margin 3 detik).
Hanya membaca data; keluaran: B1_tabel.csv, B1_detail.json, B1_terminal.log (rekaman layar).
Jalankan dari akar proyek:  venv\\Scripts\\python docs\\pengujian\\B1_repetisi\\B1_repetisi.py
"""
import glob
import json
import re
import sys
import time
import warnings
from pathlib import Path

import torch  # noqa: F401  (di Windows torch harus diimpor sebelum pandas)
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from src.app.rep_counter import OnlineRepCounter, REP_COUNTING_PARAMS
from src.features.joint_angles import POSE_LANDMARK_NAMES, compute_frame_angles, primary_angle_for_frame

LABELED = REPO / "experimental_client_pipeline" / "data" / "labeled"
LATIHAN = {"squat": "Squat", "benchpress": "Bench_press", "deadlift": "Deadlift"}
VISIBILITY, COSINE = 0.6, 0.70
KOLOM_LANDMARK = [f"{n}_{a}" for n in POSE_LANDMARK_NAMES for a in "xyzv"]

# set parameter: (prominence derajat, jarak minimum detik) per latihan
SET = [
    ("Set 1 (lama)", {"squat": (20, 0.6), "benchpress": (28, 0.6), "deadlift": (25, 1.0)}),
    ("Set 2 (tengah)", {"squat": (16, 1.0), "benchpress": (16, 1.0), "deadlift": (16, 1.0)}),
    ("Set 3 (sekarang)", {"squat": (12, 1.6), "benchpress": (12, 1.6), "deadlift": (14, 1.6)}),
]
PROTOKOL = ["utama", "lewati_5s", "countdown_selektif", "margin_3s"]


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, s):
        for f in self.files:
            f.write(s)

    def flush(self):
        for f in self.files:
            f.flush()


def kom(x, d=4):
    return f"{x:.{d}f}".replace(".", ",")


def fitur_video(kode, df):
    keluar = []
    for rec in df[["timestamp_sec"] + KOLOM_LANDMARK].to_dict("records"):
        baris = {c: rec[c] for c in KOLOM_LANDMARK}
        sudut, _ = compute_frame_angles(baris, VISIBILITY, False)
        utama = primary_angle_for_frame(baris, kode, VISIBILITY)
        keluar.append((float(rec["timestamp_sec"]), None if utama != utama else float(utama), sudut))
    return keluar


def hitung(kode, frame, prominence, jarak):
    c = OnlineRepCounter(kode, min_prominence_deg=float(prominence), min_distance_sec=float(jarak),
                         min_pattern_similarity=COSINE)
    for t, utama, sudut in frame:
        c.update(utama, t, feature_vector=sudut)
    return c.rep_count


def gt_konsentrik(t, fase, min_t):
    jumlah, sebelumnya = 0, False
    for ti, f in zip(t, fase):
        sekarang = f == "concentric"
        jumlah += int(sekarang and not sebelumnya and ti >= min_t)
        sebelumnya = sekarang
    return jumlah


def potong(frame, t_awal, t_akhir, protokol):
    """(frame yang dihitung, batas waktu minimal untuk hitungan sebenarnya)"""
    if protokol == "utama":
        mulai, selesai, min_t = 0.0, None, 0.0
    elif protokol == "lewati_5s":
        mulai, selesai, min_t = 5.0, None, 5.0
    elif protokol == "countdown_selektif":
        mulai = 5.0 if t_awal > 5.0 else 0.0
        selesai, min_t = None, mulai
    else:  # margin_3s
        mulai, selesai, min_t = t_awal - 3.0, t_akhir + 3.0, 0.0
    sub = [x for x in frame if x[0] >= mulai and (selesai is None or x[0] <= selesai)]
    return sub, min_t


def baca_semua():
    data = {}
    for kode in LATIHAN:
        data[kode] = []
        for path in sorted(glob.glob(str(LABELED / kode / "**" / "*.csv"), recursive=True)):
            m = re.search(r"_p(\d)_", Path(path).name)
            if not m:
                continue
            df = pd.read_csv(path, usecols=["timestamp_sec", "phase"] + KOLOM_LANDMARK)
            berlabel = df[df["phase"] != "excluded"]
            if berlabel.empty:
                continue
            data[kode].append({"video": Path(path).stem, "p": int(m.group(1)),
                               "t": df["timestamp_sec"].to_numpy(), "fase": df["phase"].to_numpy(),
                               "t_awal": float(berlabel["timestamp_sec"].iloc[0]),
                               "t_akhir": float(berlabel["timestamp_sec"].iloc[-1]),
                               "frame": fitur_video(kode, df)})
    return data


def main():
    log = open(HERE / "B1_terminal.log", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    try:
        sys.__stdout__.reconfigure(errors="replace")
    except Exception:
        pass
    mulai = time.time()
    try:
        selesai_main(mulai)
    finally:
        sys.stdout = sys.__stdout__
        log.close()


def selesai_main(mulai):
    print("Membaca landmark dan label dari CSV berlabel ...", flush=True)
    data = baca_semua()
    print("Jumlah video: " + ", ".join(f"{LATIHAN[k]} {len(v)}" for k, v in data.items()) +
          f" (total {sum(len(v) for v in data.values())})", flush=True)

    sekarang = {k: (v["min_prominence_deg"], v["min_distance_sec"]) for k, v in REP_COUNTING_PARAMS.items()}
    set3_sama = all(tuple(map(float, SET[2][1][k])) == tuple(map(float, sekarang[k])) for k in LATIHAN)

    hasil, kepekaan = {}, {p: {} for p in PROTOKOL}
    for kode, videos in data.items():
        hasil[kode] = {}
        for nama, param in SET:
            prom, jarak = param[kode]
            for protokol in PROTOKOL:
                baris = []
                for v in videos:
                    sub, min_t = potong(v["frame"], v["t_awal"], v["t_akhir"], protokol)
                    gt = gt_konsentrik(v["t"], v["fase"], min_t)
                    if gt == 0:
                        continue
                    baris.append({"video": v["video"], "p": v["p"], "sebenarnya": gt,
                                  "sistem": hitung(kode, sub, prom, jarak)})
                err = [abs(b["sistem"] - b["sebenarnya"]) / b["sebenarnya"] for b in baris]
                mae = float(np.mean(err))
                if protokol == "utama":
                    p1 = [e for e, b in zip(err, baris) if b["p"] == 1]
                    p23 = [e for e, b in zip(err, baris) if b["p"] != 1]
                    hasil[kode][nama] = {
                        "prominence": prom, "jarak": jarak, "mae": mae, "n": len(baris),
                        "mae_P1": float(np.mean(p1)), "n_P1": len(p1),
                        "mae_P2_P3": float(np.mean(p23)), "n_P2_P3": len(p23),
                        "bias_rata2": float(np.mean([b["sistem"] - b["sebenarnya"] for b in baris])),
                        "per_video": baris}
                else:
                    kepekaan[protokol].setdefault(kode, {})[nama] = mae

    def param_teks(nama):
        par = dict(SET)[nama]
        return " / ".join(f"{par[k][0]}°; {kom(float(par[k][1]), 1)} s" for k in LATIHAN)

    baris_tabel = []
    for nama, _ in SET:
        r = {"Set_parameter": f"{nama}: {param_teks(nama)}"}
        for kode, kolom in LATIHAN.items():
            r[kolom] = hasil[kode][nama]["mae"]
        baris_tabel.append(r)
    tabel = pd.DataFrame(baris_tabel)
    tabel.to_csv(HERE / "B1_tabel.csv", index=False, sep=";", decimal=",", float_format="%.4f", encoding="utf-8-sig")

    print("\nMAE Pers. 12 per latihan (pada data tuning; video penuh, tanpa countdown dan tanpa pemotongan)")
    tampil = tabel.copy()
    for kolom in LATIHAN.values():
        tampil[kolom] = tabel[kolom].map(kom)
    with pd.option_context("display.width", 220, "display.max_colwidth", 90):
        print(tampil.to_string(index=False))

    print("\nPer kelompok partisipan (P1 | P2 dan P3) dan bias rata-rata (sistem - sebenarnya)")
    for kode, nama_lat in LATIHAN.items():
        for nama, _ in SET:
            s = hasil[kode][nama]
            print(f"  {nama_lat:12s} {nama:17s} n {s['n']} | MAE P1 {kom(s['mae_P1'])} (n {s['n_P1']}) | "
                  f"P2+P3 {kom(s['mae_P2_P3'])} (n {s['n_P2_P3']}) | bias {str(round(s['bias_rata2'], 2)).replace('.', ',')}")

    print("\nSet dengan MAE terkecil per latihan (di antara tiga set yang dibandingkan)")
    for kode, nama_lat in LATIHAN.items():
        terbaik = min(SET, key=lambda s: hasil[kode][s[0]]["mae"])[0]
        print(f"  {nama_lat:12s} {terbaik} (MAE {kom(hasil[kode][terbaik]['mae'])})")
    print(f"\nSet 3 sama dengan REP_COUNTING_PARAMS di kode: {'ya' if set3_sama else 'TIDAK'}")

    with open(HERE / "B1_detail.json", "w", encoding="utf-8") as f:
        json.dump({"keterangan": "MAE Pers. 12 = rata-rata |C_sistem - C| / C per video; per latihan; pada data tuning; "
                                 "protokol utama = video penuh dari detik 0 sampai akhir, tanpa countdown dan tanpa pemotongan; "
                                 "landmark dari CSV berlabel",
                   "set_parameter": {n: {k: {"prominence": p[k][0], "jarak": p[k][1]} for k in LATIHAN} for n, p in SET},
                   "set3_sama_dengan_kode": set3_sama,
                   "hasil": hasil,
                   "kepekaan_protokol_lain_MAE": kepekaan}, f, ensure_ascii=False, indent=2)
    print(f"\nTersimpan: {HERE / 'B1_tabel.csv'}, B1_detail.json, B1_terminal.log  (run ini {(time.time() - mulai) / 60:.1f} menit)")


if __name__ == "__main__":
    main()
