# V2 tahap 1b (masih hanya data latih/LOPO): derau acak (3 seed), recall per postur, dan fitur yg dipakai model.
import torch  # noqa  (di Windows harus sebelum pandas)
import json, re, sys, time, collections
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pencarian_fitur import *  # noqa  (fit, f1w, load, make_sets, LATIHAN)
from sklearn.metrics import recall_score, precision_score

SEEDS = [42, 7, 2026]
PICK = ["S0 semua 363 (v1 produksi)", "S4 sudut + relatif", "S10 sudut + rel 15 landmark sudut"]
t0 = time.time()
data = {ex: load(ex) for ex in LATIHAN}
res = {}
print("== A. derau acak: LOPO rata-rata (± sd antar seed) per latihan ==")
for ex in LATIHAN:
    tr, te, cols, folds = data[ex]
    sets = make_sets(cols)
    for name in PICK:
        kol = sets[name]
        per_seed = []
        for sd in SEEDS:
            f = {k: f1w(b["class"], fit(a[kol], a["class"], seed=sd).predict(b[kol])) for k, (a, b) in folds.items()}
            per_seed.append(f)
        res[(ex, name)] = per_seed
        m = [np.mean(list(f.values())) for f in per_seed]
        print(f"[{ex:10s}] {name:36s} LOPO {np.mean(m):.4f} ± {np.std(m):.4f}   per lipatan (rata-rata seed): " + ", ".join(f"{k} {np.mean([f[k] for f in per_seed]):.3f}" for k in folds), flush=True)
print(f"[{(time.time() - t0) / 60:.1f} mnt]")

print("\n== B. selisih per sel (rata-rata 3 seed) dibanding S0 ==")
for name in PICK[1:]:
    ups = downs = same = 0
    exm = {}
    for ex in LATIHAN:
        base = {k: np.mean([f[k] for f in res[(ex, PICK[0])]]) for k in ("P1", "P2", "P3")}
        cur = {k: np.mean([f[k] for f in res[(ex, name)]]) for k in ("P1", "P2", "P3")}
        d = {k: cur[k] - base[k] for k in base}
        for v in d.values():
            ups += v >= 0.005; downs += v <= -0.005; same += abs(v) < 0.005
        exm[ex] = np.mean(list(d.values()))
    print(f"{name:36s} sel naik {ups}, sama {same}, turun {downs} dari 9 | selisih rata-rata per latihan: " + ", ".join(f"{k} {v:+.4f}" for k, v in exm.items()) + f" | rata2 3 latihan {np.mean(list(exm.values())):+.4f}")

print("\n== C. DARI DALAM: recall per POSTUR pada lipatan LOPO (seed 42, gabungan 3 lipatan) ==")
POST = lambda c: c.split("_")[1]
for ex in LATIHAN:
    tr, te, cols, folds = data[ex]; sets = make_sets(cols)
    out = {}
    for name in PICK[:2]:
        kol = sets[name]; yt, yp = [], []
        for k, (a, b) in folds.items():
            yt += [POST(c) for c in b["class"]]; yp += [POST(c) for c in fit(a[kol], a["class"]).predict(b[kol])]
        yt, yp = np.array(yt), np.array(yp)
        out[name] = {p: (np.mean(yp[yt == p] == p), np.mean(yt[yp == p] == p) if (yp == p).any() else float("nan")) for p in sorted(set(yt))}
    print(f"{ex}:")
    for p in sorted(out[PICK[0]]):
        a0, a4 = out[PICK[0]][p], out[PICK[1]][p]
        print(f"   {p:11s} recall S0 {a0[0]:.2f} -> S4 {a4[0]:.2f} | presisi S0 {a0[1]:.2f} -> S4 {a4[1]:.2f}")

print("\n== D. DARI DALAM: fitur yang dipakai model (kepentingan, dilatih pada split 70:30 seperti produksi) ==")
for ex in LATIHAN:
    tr, te, cols, folds = data[ex]; sets = make_sets(cols)
    for name in PICK[:2]:
        kol = sets[name]; m = fit(tr[kol], tr["class"]); imp = dict(zip(kol, m.feature_importances_))
        g = {"sudut": sum(v for c, v in imp.items() if re.match(r".+_angle_f\d+$", c)),
             "koordinat absolut": sum(v for c, v in imp.items() if c.startswith("coord_mean_")),
             "koordinat relatif": sum(v for c, v in imp.items() if c.startswith("coord_rel_"))}
        top = sorted(imp.items(), key=lambda t: -t[1])[:5]
        print(f"[{ex:10s}] {name:36s} " + " | ".join(f"{k} {v:.0%}" for k, v in g.items()) + "  || 5 teratas: " + ", ".join(f"{c.replace('coord_mean_', 'abs_').replace('coord_rel_', 'rel_')} {v:.1%}" for c, v in top))
print(f"\nselesai [{(time.time() - t0) / 60:.1f} mnt]")
