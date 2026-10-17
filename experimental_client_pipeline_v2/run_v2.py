"""Menjalankan aplikasi arsitektur baru (experimental_client_pipeline) dengan model v2.

Kode aplikasi tidak disalin dan tidak diubah: app.py dan landmark_pipeline.py dipakai
dari folder utama, hanya folder model dan port yang diganti.
Pemakaian: python run_v2.py [port]   (bawaan 8601, aplikasi utama memakai 8600)
"""
import asyncio
import importlib.util
import json
import pickle
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent / "experimental_client_pipeline"
MODELS_V2 = HERE / "models"
N_FITUR_V2 = 231  # 165 sudut + 66 koordinat relatif
PORT_V2 = int(sys.argv[1]) if len(sys.argv) > 1 else 8601

sys.path.insert(0, str(MAIN / "webapp" / "backend"))
import landmark_pipeline  # noqa: E402

landmark_pipeline.MODELS_DIR = MODELS_V2

for exercise in ("squat", "benchpress", "deadlift"):
    cfg = json.loads((MODELS_V2 / f"{exercise}_feature_config.json").read_text())
    with open(MODELS_V2 / f"{exercise}_rf.pkl", "rb") as f:
        n_model = pickle.load(f).named_steps["rf"].n_features_in_
    if not (n_model == len(cfg["feat_cols"]) == N_FITUR_V2):
        sys.exit(f"{exercise}: model {n_model} fitur, config {len(cfg['feat_cols'])}, seharusnya {N_FITUR_V2}")
print(f"Model v2 aktif ({N_FITUR_V2} fitur per latihan, tanpa koordinat absolut): {MODELS_V2}")

# dimuat lewat path karena root proyek juga punya app.py (aplikasi lama)
spec = importlib.util.spec_from_file_location("webapp_app", MAIN / "webapp" / "app.py")
webapp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(webapp)
webapp.PORT = PORT_V2

if __name__ == "__main__":
    asyncio.run(webapp.main())
