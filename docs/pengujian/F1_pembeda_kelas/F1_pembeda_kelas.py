"""F1: pembeda kelas dibanding variasi antar orang (analisis data yang sudah ada, tanpa pelatihan ulang).

Menjawab pertanyaan dosen: kelas yang tertukar itu karena pembeda kelasnya kecil, karena variasi orang, atau karena salah pengujian.
Untuk tiap pasangan (postur benar vs satu kesalahan) dan tiap fitur kunci (11 sudut dalam derajat dan 4 ukuran posisi tubuh):
  selisih kelas  = rata-rata fitur kelas salah dikurangi kelas benar, dihitung per partisipan di dalam sel partisipan x kamera x fase
                   yang sama (supaya posisi kamera dan fase tidak ikut), lalu dirata-rata antar partisipan;
  konsistensi    = berapa partisipan yang selisihnya searah;
  sebaran orang  = simpangan baku antar partisipan pada kelas benar;
  rasio R        = |selisih kelas| / sebaran orang. R >= 1 berarti pembeda kelas lebih besar daripada perbedaan antar orang;
  sebaran dalam  = simpangan baku antar jendela di dalam satu kelas (getaran gerakan dan pose), rata-rata antar sel;
  rasio d        = |selisih kelas| / sebaran dalam. d >= 1 berarti selisih kelas lebih besar daripada variasi di dalam kelas;
  skor           = min(R, d). Fitur disebut pembeda kuat bila searah pada semua partisipan, R >= 1, dan d >= 1 (batas 1 dipilih
                   sederhana dan ditetapkan sebelum melihat hasil).
Data: semua jendela asli (bukan cermin) dari split train dan test. Sudut diubah kembali ke derajat dengan penskala model aplikasi.
Keluaran: F1_tabel.csv (ringkasan per pasangan), F1_detail.csv (semua fitur), F1_terminal.log.
Jalankan dari akar proyek:  venv\\Scripts\\python docs\\pengujian\\F1_pembeda_kelas\\F1_pembeda_kelas.py
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
MODELS = REPO / "client_pipeline" / "models"
SPLITS = REPO / "client_pipeline" / "data" / "windowed_features" / "splits"

KESALAHAN = {"squat": ("kneeinward", "backbend"), "benchpress": ("armsspread", "headbacklift"), "deadlift": ("armsspread", "backround")}
SUDUT = ["neck_angle", "left_shoulder_angle", "right_shoulder_angle", "left_elbow_angle", "right_elbow_angle", "left_hip_angle",
         "right_hip_angle", "left_knee_angle", "right_knee_angle", "left_ankle_angle", "right_ankle_angle"]
POSISI = {"tinggi_hidung": "satuan panjang torso", "tinggi_bahu": "satuan panjang torso", "jarak_tangan_per_bahu": "rasio",
          "jarak_lutut_per_kaki": "rasio"}
MIN_WINDOW = 20


def kom(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def rasio(a, b):
    return a / np.maximum(b, 0.05)


def muat(latihan):
    df = pd.concat([pd.read_csv(SPLITS / f"{latihan}_{s}.csv") for s in ("train", "test")], ignore_index=True)
    df = df[~df["window_id"].str.contains("flip", case=False)].reset_index(drop=True)
    with open(MODELS / f"{latihan}_rf.pkl", "rb") as f:
        skala = pickle.load(f).steps[0][1]
    fit = pd.DataFrame({"peserta": df["participant_id"], "kamera": df["camera_angle"], "fase": df["phase"], "kelas": df["posture_class"]})
    for dasar in SUDUT:
        norm = df[skala.angle_cols_by_base_[dasar]].mean(axis=1)
        fit[dasar] = norm * (skala.angle_max_[dasar] - skala.angle_min_[dasar]) + skala.angle_min_[dasar]
    c = lambda nama: df[f"coord_rel_{nama}"]
    fit["tinggi_hidung"] = -c("nose_y")
    fit["tinggi_bahu"] = -(c("left_shoulder_y") + c("right_shoulder_y")) / 2
    fit["jarak_tangan_per_bahu"] = rasio((c("right_wrist_x") - c("left_wrist_x")).abs(), (c("right_shoulder_x") - c("left_shoulder_x")).abs())
    fit["jarak_lutut_per_kaki"] = rasio((c("right_knee_x") - c("left_knee_x")).abs(), (c("right_ankle_x") - c("left_ankle_x")).abs())
    return fit


def bandingkan(fit, benar, salah, fitur):
    sub = fit[fit["kelas"].isin([benar, salah])]
    grup = sub.groupby(["peserta", "kamera", "fase", "kelas"])[fitur]
    rata, sd, n = grup.mean(), grup.std(), grup.size()
    baris = []
    for f in fitur:
        selisih, rata_benar, sd_dalam = {}, {}, []
        for (p, kam, fase) in {k[:3] for k in rata.index}:
            kb, ks = (p, kam, fase, benar), (p, kam, fase, salah)
            if kb in rata.index and ks in rata.index and n[kb] >= MIN_WINDOW and n[ks] >= MIN_WINDOW:
                selisih.setdefault(p, []).append(rata.loc[ks, f] - rata.loc[kb, f])
                rata_benar.setdefault(p, []).append(rata.loc[kb, f])
                sd_dalam.append(np.sqrt((sd.loc[kb, f] ** 2 + sd.loc[ks, f] ** 2) / 2))
        orang = sorted(selisih)
        d = np.array([np.mean(selisih[p]) for p in orang])
        m = np.array([np.mean(rata_benar[p]) for p in orang])
        sebaran = float(np.std(m, ddof=1)) if len(m) > 1 else float("nan")
        dalam = float(np.mean(sd_dalam))
        selisih_rata = float(d.mean())
        r = abs(selisih_rata) / sebaran if sebaran > 0 else float("nan")
        rd = abs(selisih_rata) / dalam if dalam > 0 else float("nan")
        searah = int(max((d > 0).sum(), (d < 0).sum()))
        baris.append({"fitur": f, "satuan": "derajat" if f in SUDUT else POSISI[f], "selisih": selisih_rata,
                      "per_peserta": {p: float(x) for p, x in zip(orang, d)}, "searah": searah, "n_peserta": len(orang),
                      "sebaran_orang": sebaran, "R": r, "sebaran_dalam": dalam, "d": rd, "skor": min(r, rd),
                      "kuat": bool(searah == len(orang) and r >= 1 and rd >= 1)})
    return baris


def main():
    ap = argparse.ArgumentParser(description="F1: pembeda kelas vs variasi antar orang")
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fitur = SUDUT + list(POSISI)
    ringkas, detail, log = [], [], []

    def cetak(s=""):
        print(s, flush=True)
        log.append(s)

    cetak("Pembeda kelas dibanding variasi antar orang (jendela asli train+test; selisih = kelas salah dikurangi kelas benar)")
    for latihan, salah_list in KESALAHAN.items():
        fit = muat(latihan)
        cetak(f"\n== {latihan}: {len(fit)} jendela, peserta {sorted(fit['peserta'].unique())}")
        for salah in salah_list:
            hasil = bandingkan(fit, "correct", salah, fitur)
            kuat = [h for h in hasil if h["kuat"]]
            urut = sorted(hasil, key=lambda h: -(h["skor"] if h["skor"] == h["skor"] else 0))
            cetak(f"\n  correct vs {salah}: pembeda kuat (searah semua peserta, R >= 1, d >= 1): {len(kuat)} dari {len(hasil)} fitur")
            cetak(f"  {'fitur (4 skor tertinggi)':24s} {'satuan':20s} {'selisih':>8s} {'per peserta':>26s} {'searah':>6s} "
                  f"{'seb.orang':>9s} {'R':>5s} {'seb.dlm':>8s} {'d':>5s}")
            for h in urut[:4]:
                pp = " ".join(f"{p}:{kom(x, 2):>6s}" for p, x in h["per_peserta"].items())
                cetak(f"  {h['fitur']:24s} {h['satuan']:20s} {kom(h['selisih']):>8s} {pp:>26s} {h['searah']}/{h['n_peserta']:<4d} "
                      f"{kom(h['sebaran_orang']):>9s} {kom(h['R'], 1):>5s} {kom(h['sebaran_dalam']):>8s} {kom(h['d'], 1):>5s}")
            for h in hasil:
                detail.append({"latihan": latihan, "pasangan": f"correct vs {salah}", **{k: v for k, v in h.items() if k != "per_peserta"},
                               **{f"selisih_p{p}": x for p, x in h["per_peserta"].items()}})
            terbaik = urut[0]
            ringkas.append({"latihan": latihan, "pasangan": f"correct vs {salah}", "fitur_kuat": f"{len(kuat)} dari {len(hasil)}",
                            "skor_tertinggi": kom(terbaik["skor"], 1), "fitur_skor_tertinggi": terbaik["fitur"],
                            "searah": f"{terbaik['searah']}/{terbaik['n_peserta']}", "R": kom(terbaik["R"], 1), "d": kom(terbaik["d"], 1)})
    cetak("\nRingkasan")
    tabel = pd.DataFrame(ringkas)
    cetak(tabel.to_string(index=False))
    tabel.to_csv(out / "F1_tabel.csv", index=False, sep=";", encoding="utf-8-sig")
    pd.DataFrame(detail).to_csv(out / "F1_detail.csv", index=False, sep=";", encoding="utf-8-sig", decimal=",")
    (out / "F1_terminal.log").write_text("\n".join(log), encoding="utf-8")
    cetak(f"\nTersimpan: {out / 'F1_tabel.csv'}, F1_detail.csv, F1_terminal.log")


if __name__ == "__main__":
    main()
