"""A4a: pengaruh tiga tambahan Method E pada model final, dinilai dengan dua cara.

Empat varian bertingkat (RF 100 pohon, class_weight balanced, seed 42, window 15 frame):
  A baseline   297 fitur (joint angle + rata-rata koordinat), tanpa flip, min_samples_leaf 1
  B +relatif   A + 66 koordinat relatif (363 fitur)
  C +flip      B dengan data latih di-flip horizontal
  D final      C dengan min_samples_leaf 4 (= model produksi)
Dua kolom hasil, F1 weighted pada 6 kelas per latihan:
  70:30  test split 70:30 per run (test yang sama dengan A1)
  LOPO   rata-rata tiga lipatan: satu partisipan ditahan sebagai uji, dua lainnya untuk melatih

Hanya membaca data dan model di experimental_client_pipeline; tidak ada berkas proyek yang diubah.
Normalisasi Min-Max sudut berasal dari split 70:30 dan tidak dihitung ulang per lipatan LOPO (berlaku sama untuk keempat varian).
Jalankan dari akar proyek:  venv\\Scripts\\python docs\\pengujian\\A4_konfigurasi_final\\A4a_komponen_model.py
"""
import json
import sys
import time
from pathlib import Path

import torch  # noqa: F401  (di Windows torch harus diimpor sebelum pandas)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from src.features.augmentation import flip_horizontal
from src.features.build_dataset import META_COLUMNS

EXP = REPO / "experimental_client_pipeline"
SPLIT = EXP / "data" / "windowed_features" / "splits"
LATIHAN = {"squat": "Squat", "benchpress": "Bench press", "deadlift": "Deadlift"}
# nama varian, koordinat relatif, flip data latih, min_samples_leaf
VARIAN = [("A baseline", False, False, 1), ("B +relatif", True, False, 1),
          ("C +flip", True, True, 1), ("D final (+leaf 4)", True, True, 4)]
SAMA_JIKA_SELISIH_KURANG = 0.005


def latih(x, y, leaf):
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=leaf, class_weight="balanced",
                                  random_state=42, n_jobs=-1).fit(x, y)


def f1w(y, p):
    return f1_score(y, p, average="weighted", zero_division=0)


def kom(x, d=4):
    return f"{x:.{d}f}".replace(".", ",")


def main():
    mulai = time.time()
    hasil, cek = [], []
    for kode, nama in LATIHAN.items():
        prod = json.load(open(EXP / "models" / f"{kode}_evaluation.json"))
        fitur_prod = json.load(open(EXP / "models" / f"{kode}_feature_config.json"))["feat_cols"]
        tr = pd.read_csv(SPLIT / f"{kode}_train.csv")
        te = pd.read_csv(SPLIT / f"{kode}_test.csv")
        semua = [c for c in tr.columns if c not in META_COLUMNS]
        if set(semua) != set(fitur_prod) or len(te) != prod["n_test"]:
            raise RuntimeError(f"{kode}: kolom fitur atau jumlah test tidak sama dengan model produksi")
        asli = tr[~tr["window_id"].str.endswith("_flip")].reset_index(drop=True)
        dasar = [c for c in semua if not c.startswith("coord_rel_")]
        pool = pd.concat([asli, te], ignore_index=True)  # semua window asli, dipakai untuk LOPO
        partisipan = sorted(pool["participant_id"].unique())
        if len(partisipan) != 3:
            raise RuntimeError(f"{kode}: jumlah partisipan {len(partisipan)}, seharusnya 3")

        for varian, relatif, flip, leaf in VARIAN:
            kolom = semua if relatif else dasar
            data_latih = tr if flip else asli
            pred = latih(data_latih[kolom], data_latih["class"], leaf).predict(te[kolom])
            f_split = f1w(te["class"], pred)

            lopo = {}
            for pid in partisipan:
                a, b = pool[pool["participant_id"] != pid], pool[pool["participant_id"] == pid]
                if flip:
                    a = pd.concat([a, flip_horizontal(a)], ignore_index=True)
                lopo[f"P{str(pid).lstrip('pP')}"] = f1w(b["class"], latih(a[kolom], a["class"], leaf).predict(b[kolom]))

            hasil.append({"Latihan": nama, "Varian": varian, "Jumlah_fitur": len(kolom),
                          "F1w_70_30": f_split, "F1w_LOPO": float(np.mean(list(lopo.values()))),
                          "_kode": kode, "_lopo": lopo,
                          "_detail": {"n_latih": len(data_latih), "n_uji": len(te),
                                      "accuracy_70_30": accuracy_score(te["class"], pred),
                                      "f1_macro_70_30": f1_score(te["class"], pred, average="macro", zero_division=0)}})
            lipatan = "; ".join(f"{p} {kom(v, 3)}" for p, v in lopo.items())
            print(f"[{nama}] {varian:18s} {len(kolom)} fitur | 70:30 {kom(f_split)} | "
                  f"LOPO {kom(hasil[-1]['F1w_LOPO'], 3)} ({lipatan})  [{(time.time() - mulai) / 60:.1f} menit]",
                  flush=True)
            if varian.startswith("D"):
                cek.append({"latihan": nama, "f1_weighted_A4a_varian_D": f_split,
                            "f1_weighted_A1_json": prod["f1_weighted"], "selisih": f_split - prod["f1_weighted"]})

    tabel = pd.DataFrame(hasil)[["Latihan", "Varian", "Jumlah_fitur", "F1w_70_30", "F1w_LOPO"]]
    tabel.to_csv(HERE / "A4a_tabel.csv", index=False, sep=";", decimal=",", float_format="%.4f")
    tampil = tabel.copy()
    for k in ("F1w_70_30", "F1w_LOPO"):
        tampil[k] = tabel[k].map(kom)
    print("\n" + tampil.to_string(index=False))

    # final (D) dibanding baseline (A): selisih rata-rata dan per lipatan
    selisih, naik, sama, turun = {}, 0, 0, 0
    print("\nFinal dibanding baseline")
    for kode, nama in LATIHAN.items():
        a = next(h for h in hasil if h["_kode"] == kode and h["Varian"].startswith("A"))
        d = next(h for h in hasil if h["_kode"] == kode and h["Varian"].startswith("D"))
        s = {"selisih_70_30": d["F1w_70_30"] - a["F1w_70_30"], "selisih_LOPO": d["F1w_LOPO"] - a["F1w_LOPO"]}
        selisih[nama] = s
        for pid in a["_lopo"]:
            delta = d["_lopo"][pid] - a["_lopo"][pid]
            naik += delta >= SAMA_JIKA_SELISIH_KURANG
            turun += delta <= -SAMA_JIKA_SELISIH_KURANG
            sama += abs(delta) < SAMA_JIKA_SELISIH_KURANG
        print(f"  {nama:12s} 70:30 {s['selisih_70_30']:+.4f} | LOPO {s['selisih_LOPO']:+.4f}".replace(".", ","))
    print(f"  Lipatan LOPO (final lawan baseline, batas sama = {SAMA_JIKA_SELISIH_KURANG}): "
          f"naik {naik}, sama {sama}, turun {turun} dari {naik + sama + turun}".replace(".", ","))

    # efek tiap tambahan: selisih dari varian sebelumnya (A ke B, B ke C, C ke D)
    efek = {}
    print("\nEfek tiap tambahan (selisih dari varian sebelumnya)")
    for kode, nama in LATIHAN.items():
        hk = [h for h in hasil if h["_kode"] == kode]
        efek[nama] = {}
        for sebelum, sesudah in zip(hk, hk[1:]):
            e = {"selisih_70_30": sesudah["F1w_70_30"] - sebelum["F1w_70_30"],
                 "selisih_LOPO": sesudah["F1w_LOPO"] - sebelum["F1w_LOPO"]}
            efek[nama][sesudah["Varian"]] = e
            print(f"  {nama:12s} {sesudah['Varian']:18s} 70:30 {e['selisih_70_30']:+.4f} | "
                  f"LOPO {e['selisih_LOPO']:+.4f}".replace(".", ","))

    print("\nCek varian D terhadap *_evaluation.json produksi (A1)")
    for c in cek:
        print(f"  {c['latihan']:12s} A4a {kom(c['f1_weighted_A4a_varian_D'])} | A1 {kom(c['f1_weighted_A1_json'])} | "
              f"selisih {c['selisih']:+.6f}".replace(".", ","))

    detail = {"keterangan": "F1 weighted, 6 kelas per latihan; LOPO = tahan satu partisipan; flip hanya pada data latih",
              "varian": [{"latihan": h["Latihan"], "varian": h["Varian"], "jumlah_fitur": h["Jumlah_fitur"],
                          "f1_weighted_70_30": h["F1w_70_30"], **h["_detail"],
                          "f1_weighted_LOPO_per_lipatan": h["_lopo"], "f1_weighted_LOPO_rata2": h["F1w_LOPO"]}
                         for h in hasil],
              "selisih_final_vs_baseline": selisih,
              "efek_tiap_tambahan": efek,
              "lipatan_LOPO_final_vs_baseline": {"naik": int(naik), "sama": int(sama), "turun": int(turun),
                                                 "batas_sama": SAMA_JIKA_SELISIH_KURANG},
              "cek_varian_D_vs_A1": cek}
    with open(HERE / "A4a_detail.json", "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)
    print(f"\nTersimpan: {HERE / 'A4a_tabel.csv'} dan A4a_detail.json  (total {(time.time() - mulai) / 60:.1f} menit)")


if __name__ == "__main__":
    main()
