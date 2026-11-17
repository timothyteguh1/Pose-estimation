# V2 tahap 2: parameter RF pada SATU set fitur terpilih (S4 sudut+relatif). Pemilihan hanya LOPO (2 seed). E2 tidak dipakai.
import torch  # noqa
import json, sys, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pencarian_fitur import fit, f1w, load, make_sets, LATIHAN, SCR

SEEDS = [42, 7]
GRID = [(leaf, mf) for leaf in (4, 8, 16) for mf in ("sqrt", 0.3)]   # leaf 1 dan mf 0,6 dibuang karena terlalu lambat (waktu komputasi), bukan karena hasil
t0 = time.time()
data = {ex: load(ex) for ex in LATIHAN}
out = {}
for ex in LATIHAN:
    tr, te, cols, folds = data[ex]
    kol = make_sets(cols)["S4 sudut + relatif"]
    for leaf, mf in GRID:
        key = f"leaf{leaf}_mf{mf}"
        f70 = f1w(te["class"], fit(tr[kol], tr["class"], leaf=leaf, mf=mf).predict(te[kol]))
        lopo = {}
        for k, (a, b) in folds.items():
            lopo[k] = float(np.mean([f1w(b["class"], fit(a[kol], a["class"], leaf=leaf, mf=mf, seed=sd).predict(b[kol])) for sd in SEEDS]))
        out.setdefault(key, {})[ex] = {"f1_7030": f70, "lopo": lopo, "lopo_rata2": float(np.mean(list(lopo.values())))}
        print(f"[{ex:10s}] daun {leaf:2d} fitur/cabang {str(mf):5s} | 70:30 {f70:.4f} | LOPO {np.mean(list(lopo.values())):.4f} ({', '.join(f'{k} {v:.3f}' for k, v in lopo.items())}) [{(time.time() - t0) / 60:.1f} mnt]", flush=True)
json.dump(out, open(SCR / "v2_tahap2.json", "w"), indent=1)
print("selesai", f"{(time.time() - t0) / 60:.1f} mnt")
