# V2 final: latih persis seperti produksi (split 70:30 per run, flip data latih, RF 100 pohon, class_weight balanced, seed 42)
# pada fitur terpilih; simpan di scratchpad/v2_models (TIDAK menyentuh models/ proyek). Pemakaian: python v2_latih_final.py <leaf> <max_features>
import torch  # noqa
import json, sys, copy, pickle
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pencarian_fitur import fit, f1w, load, make_sets, LATIHAN, SCR, EXP

leaf = int(sys.argv[1]); mf = sys.argv[2]; mf = mf if mf == "sqrt" else float(mf)
OUT = SCR / "v2_models"; OUT.mkdir(exist_ok=True)
for ex in LATIHAN:
    tr, te, cols, folds = load(ex)
    kol = make_sets(cols)["S4 sudut + relatif"]
    m = fit(tr[kol], tr["class"], leaf=leaf, mf=mf)
    pred = m.predict(te[kol])
    prod = pickle.load(open(EXP / "models" / f"{ex}_rf.pkl", "rb"))
    pipe = Pipeline([("scaler", copy.deepcopy(prod.named_steps["scaler"])), ("rf", m)])
    cfg = json.load(open(EXP / "models" / f"{ex}_feature_config.json")); cfg["feat_cols"] = kol
    pickle.dump(pipe, open(OUT / f"{ex}_rf.pkl", "wb")); json.dump(cfg, open(OUT / f"{ex}_feature_config.json", "w"))
    # kesamaan dengan produksi: model v1 harus menghasilkan F1 produksi
    ev = json.load(open(EXP / "models" / f"{ex}_evaluation.json"))
    print(f"{ex:10s} v2: {len(kol)} fitur | 70:30 F1w {f1w(te['class'], pred):.4f} akurasi {accuracy_score(te['class'], pred):.4f} | v1 produksi F1w {ev['f1_weighted']:.4f}", flush=True)
print("model v2 tersimpan di", OUT)
