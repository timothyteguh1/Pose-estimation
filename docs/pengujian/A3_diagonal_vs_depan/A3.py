"""A3: model dua sudut diagonal vs model tiga sudut (A1), split 70:30 per run (seed 42).

Model 2 sudut = RF yang sama dengan produksi (100 pohon, min_samples_leaf 4, balanced, seed 42), dilatih dari data latih
produksi tanpa window sudut depan (camera_angle front; baris flip ikut tersaring). Dua pengujian untuk model 2 sudut:
  uji_diagonal      hanya window left45 dan right45 dari test split
  uji_data_lengkap  seluruh test split (tiga sudut), sama dengan test A1
Model 3 sudut tidak diuji ulang: pembandingnya adalah *_evaluation.json produksi (A1).

Keluaran: hasil/{latihan}_evaluation_2sudut.json (satu per latihan, bentuk seperti *_evaluation.json) dan tabel di layar.
Hanya membaca model dan data proyek; tidak ada berkas model yang ditulis ulang.
Jalankan dari akar proyek:  venv\\Scripts\\python docs\\pengujian\\A3_diagonal_vs_depan\\A3.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[2] / "experimental_client_pipeline"
SPLIT = EXP / "data" / "windowed_features" / "splits"
HASIL = HERE / "hasil"
LATIHAN = {"squat": "Squat", "benchpress": "Bench press", "deadlift": "Deadlift"}
SUDUT = ("left45", "right45", "front")
HYPER = {"n_estimators": 100, "min_samples_leaf": 4, "class_weight": "balanced", "random_state": 42}
KOLOM = ["Latihan", "n_test", "Accuracy", "Precision_w", "Recall_w", "F1_weighted", "F1_macro"]


def blok(y, p, sudut, kelas):
    pw, rw, fw, _ = precision_recall_fscore_support(y, p, average="weighted", zero_division=0)
    return {"camera_angle_uji": {a: int((sudut == a).sum()) for a in SUDUT}, "n_test": len(y),
            "accuracy": accuracy_score(y, p), "precision_weighted": pw, "recall_weighted": rw, "f1_weighted": fw,
            "classification_report": classification_report(y, p, labels=kelas, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, p, labels=kelas).tolist()}


def ringkas(nama, b):
    return {"Latihan": nama, "n_test": b["n_test"], "Accuracy": b["accuracy"], "Precision_w": b["precision_weighted"],
            "Recall_w": b["recall_weighted"], "F1_weighted": b["f1_weighted"],
            "F1_macro": b["classification_report"]["macro avg"]["f1-score"]}


def gabungan(baris):
    # kelas tiap latihan terpisah (6 + 6 + 6): metrik weighted = rata-rata terbobot n, macro = rata-rata tiga macro
    n = sum(r["n_test"] for r in baris)
    g = {"Latihan": "Gabungan 3 latihan", "n_test": n}
    for k in ("Accuracy", "Precision_w", "Recall_w", "F1_weighted"):
        g[k] = sum(r[k] * r["n_test"] for r in baris) / n
    g["F1_macro"] = float(np.mean([r["F1_macro"] for r in baris]))
    return baris + [g]


def cetak(judul, baris):
    t = pd.DataFrame(baris)[KOLOM]
    for k in KOLOM[2:]:
        t[k] = t[k].map(lambda x: f"{x:.4f}".replace(".", ","))
    print(f"\n{judul}\n{t.to_string(index=False)}")


def main():
    HASIL.mkdir(exist_ok=True)
    tabel = {"diagonal": [], "lengkap": [], "a1": []}
    per_kelas = []
    for kode, nama in LATIHAN.items():
        fitur = json.load(open(EXP / "models" / f"{kode}_feature_config.json"))["feat_cols"]
        a1 = json.load(open(EXP / "models" / f"{kode}_evaluation.json"))
        kelas = a1["classes"]
        tr = pd.read_csv(SPLIT / f"{kode}_train.csv")
        te = pd.read_csv(SPLIT / f"{kode}_test.csv")
        if len(te) != a1["n_test"]:
            raise RuntimeError(f"{kode}: test split ({len(te)}) tidak sama dengan test A1 ({a1['n_test']})")

        tr2 = tr[tr["camera_angle"] != "front"]
        rf = RandomForestClassifier(**HYPER, n_jobs=-1).fit(tr2[fitur], tr2["class"])
        if sorted(rf.classes_) != sorted(kelas):
            raise RuntimeError(f"{kode}: kelas model 2 sudut tidak sama dengan kelas A1")

        te_d = te[te["camera_angle"] != "front"]
        if not set(te_d["camera_angle"]) <= {"left45", "right45"}:
            raise RuntimeError(f"{kode}: uji diagonal masih memuat sudut selain left45 dan right45")
        diag = blok(te_d["class"].to_numpy(), rf.predict(te_d[fitur]), te_d["camera_angle"].to_numpy(), kelas)
        lengkap = blok(te["class"].to_numpy(), rf.predict(te[fitur]), te["camera_angle"].to_numpy(), kelas)

        hasil = {"exercise": kode, "camera_angles_latih": ["left45", "right45"], "n_train": len(tr2),
                 "n_features": len(fitur), "classes": kelas, "hyperparameters": HYPER,
                 "uji_diagonal": diag, "uji_data_lengkap": lengkap}
        with open(HASIL / f"{kode}_evaluation_2sudut.json", "w", encoding="utf-8") as f:
            json.dump(hasil, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

        tabel["diagonal"].append(ringkas(nama, diag))
        tabel["lengkap"].append(ringkas(nama, lengkap))
        tabel["a1"].append(ringkas(nama, {**a1, "camera_angle_uji": {}}))
        for k in kelas:
            r = lengkap["classification_report"][k]
            per_kelas.append({"Latihan": nama, "Kelas": k.split("_", 1)[1], "Precision": r["precision"],
                              "Recall": r["recall"], "F1": r["f1-score"], "Support": int(r["support"])})
        print(f"[{nama}] window latih: {n_asli(tr2)} asli (model 3 sudut: {n_asli(tr)}) | uji diagonal "
              f"{diag['camera_angle_uji']} | uji data lengkap {lengkap['camera_angle_uji']} (= test A1, n {a1['n_test']})",
              flush=True)

    cetak("Model 2 sudut, diuji pada diagonal saja", gabungan(tabel["diagonal"]))
    cetak("Model 2 sudut, diuji pada data lengkap (3 sudut, sama dengan test A1)", gabungan(tabel["lengkap"]))
    cetak("Model 3 sudut (A1, dari *_evaluation.json)", gabungan(tabel["a1"]))
    pk = pd.DataFrame(per_kelas)
    for k in ("Precision", "Recall", "F1"):
        pk[k] = pk[k].map(lambda x: f"{x:.3f}".replace(".", ","))
    print("\nModel 2 sudut pada data lengkap, per kelas\n" + pk.to_string(index=False))
    print(f"\nTersimpan di {HASIL}: {', '.join(f'{k}_evaluation_2sudut.json' for k in LATIHAN)}")


def n_asli(df):
    return int((~df["window_id"].str.endswith("_flip")).sum())


if __name__ == "__main__":
    main()
