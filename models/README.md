# models/

Output `src/training/train_model.py`, per exercise (1 model = 1 exercise,
cuma Random Forest, tidak ada model lain -- lihat `src/training/README.md`):

- `{exercise}_rf.pkl` — model terlatih (pickle)
- `{exercise}_evaluation.json` — accuracy/precision/recall/F1 (weighted),
  classification report per kelas, confusion matrix (angka), hyperparameter
- `{exercise}_confusion_matrix.png` — visualisasi confusion matrix

`.pt` untuk weight YOLOv11 (dan YOLOv5 pembanding, bab 8.4 poin 13) belum
ada, menunggu tahap `src/detection/`.
