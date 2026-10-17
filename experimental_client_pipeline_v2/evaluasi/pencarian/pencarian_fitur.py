# V2 tahap 1: seleksi FITUR (tanpa metode baru). Pemilihan hanya dgn LOPO data latih; harness = A4a varian D (produksi).
# Hanya membaca data/model; keluaran ke scratchpad.
import json, re, sys, time
from pathlib import Path
import torch  # noqa
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

REPO = Path(r"C:\Users\totit\Downloads\skripsi-postur-gym"); sys.path.insert(0, str(REPO))
from src.features.augmentation import flip_horizontal
from src.features.build_dataset import META_COLUMNS
from src.features.joint_angles import JOINT_ANGLE_TRIPLES, NECK_ANGLE_TRIPLES

SCR = Path(__file__).resolve().parent
EXP = REPO / "experimental_client_pipeline"; SPLIT = EXP / "data" / "windowed_features" / "splits"
LATIHAN = ["squat", "benchpress", "deadlift"]
ANGLM = set()
for a, b, c in list(JOINT_ANGLE_TRIPLES.values()) + list(NECK_ANGLE_TRIPLES):
    ANGLM.update((a, b, c))


def f1w(y, p):
    return f1_score(y, p, average="weighted", zero_division=0)


def fit(x, y, n=100, leaf=4, mf="sqrt", seed=42):
    return RandomForestClassifier(n_estimators=n, min_samples_leaf=leaf, max_features=mf, class_weight="balanced",
                                  random_state=seed, n_jobs=-1).fit(x, y)


def lmname(c):
    return c.replace("coord_mean_", "").replace("coord_rel_", "")[:-2]


def make_sets(cols):
    ANG = [c for c in cols if re.match(r".+_angle_f\d+$", c)]
    ABS = [c for c in cols if c.startswith("coord_mean_")]
    REL = [c for c in cols if c.startswith("coord_rel_")]
    ab = lambda ax: [c for c in ABS if c.endswith("_" + ax)]
    in15 = lambda L: [c for c in L if lmname(c) in ANGLM]
    return {
        "S0 semua 363 (v1 produksi)": ANG + ABS + REL,
        "S1 tanpa visibility": ANG + ab("x") + ab("y") + ab("z") + REL,
        "S2 tanpa z": ANG + ab("x") + ab("y") + ab("v") + REL,
        "S3 tanpa visibility & z": ANG + ab("x") + ab("y") + REL,
        "S4 sudut + relatif": ANG + REL,
        "S5 sudut saja": ANG,
        "S6 sudut + abs x,y": ANG + ab("x") + ab("y"),
        "S7 sudut + abs semua (tanpa relatif)": ANG + ABS,
        "S8 sudut + abs&rel 15 landmark sudut": ANG + in15(ABS) + in15(REL),
        "S9 sudut + abs x,y & rel 15 landmark sudut": ANG + in15(ab("x") + ab("y")) + in15(REL),
        "S10 sudut + rel 15 landmark sudut": ANG + in15(REL),
    }


def load(ex):
    tr = pd.read_csv(SPLIT / f"{ex}_train.csv"); te = pd.read_csv(SPLIT / f"{ex}_test.csv")
    cols = [c for c in tr.columns if c not in META_COLUMNS]
    asli = tr[~tr["window_id"].str.endswith("_flip")].reset_index(drop=True)
    pool = pd.concat([asli, te], ignore_index=True)
    folds = {}
    for pid in sorted(pool["participant_id"].unique()):
        a, b = pool[pool["participant_id"] != pid], pool[pool["participant_id"] == pid]
        folds[f"P{str(pid).lstrip('pP')}"] = (pd.concat([a, flip_horizontal(a)], ignore_index=True), b)
    return tr, te, cols, folds


def main():
    t0 = time.time()
    out = {}
    data = {ex: load(ex) for ex in LATIHAN}
    print(f"data siap [{(time.time() - t0) / 60:.1f} mnt]", flush=True)
    sets = make_sets(data["squat"][2])
    for name in sets:
        out[name] = {"n_fitur": len(sets[name])}
    for ex in LATIHAN:
        tr, te, cols, folds = data[ex]
        sets = make_sets(cols)
        for name, kol in sets.items():
            m = fit(tr[kol], tr["class"]); f70 = f1w(te["class"], m.predict(te[kol]))
            lopo = {}
            for k, (a, b) in folds.items():
                lopo[k] = f1w(b["class"], fit(a[kol], a["class"]).predict(b[kol]))
            out[name][ex] = {"f1_7030": f70, "lopo": lopo, "lopo_rata2": float(np.mean(list(lopo.values())))}
            print(f"[{ex:10s}] {name:44s} {len(kol):3d} fitur | 70:30 {f70:.4f} | LOPO {np.mean(list(lopo.values())):.4f} ({', '.join(f'{k} {v:.3f}' for k, v in lopo.items())}) [{(time.time() - t0) / 60:.1f} mnt]", flush=True)
    json.dump(out, open(SCR / "v2_tahap1.json", "w"), indent=1)
    print("\nselesai", f"{(time.time() - t0) / 60:.1f} mnt")


if __name__ == "__main__":
    main()
