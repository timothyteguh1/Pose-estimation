"""C1: ketepatan kotak deteksi orang terhadap kotak sebenarnya (dipakai C1_yolo.py).

Frame dipilih acak berstrata (latihan x sudut kamera). Kotak BADAN orang yang sedang berlatih (kepala sampai kaki dan ujung
lengan; barbel, plate, rak, dan bangku tidak ikut) digambar peneliti SEBELUM hasil model dilihat dan disimpan di
C1_kotak_sebenarnya.json. Kotak tepat = kotak dengan confidence tertinggi (seperti di aplikasi) yang IoU-nya >= 0,5 terhadap
kotak sebenarnya; frame tanpa deteksi dihitung gagal.

Opsi ganti=True (C1_yolo.py --ganti-tanpa-orang): frame berlabel "tidak ada orang" (N) diganti frame acak lain dari sel yang sama
dengan rng terpisah (SEED_PENGGANTI), sampai tiap sel punya per_sel frame berorang. Pilihan lama dan kotaknya tidak berubah;
yang dilabeli hanya frame baru.
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

AMBANG_IOU = 0.5
SEED_PENGGANTI = 43
JENDELA = "Kotak = BADAN orang saja (barbel dan plate TIDAK ikut)"


def kom(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def kunci(video, t):
    return f"{video}|{round(float(t), 2)}"


def sel_frame(video):
    latihan = video.split("_", 1)[0]
    sudut = re.search(r"_(left45|right45|front)_", video)
    return latihan, (sudut.group(1) if sudut else "lain")


def pilih_frame(frames, per_sel, seed=42):
    """per_sel frame acak untuk tiap sel (latihan x sudut kamera); urutan tampil diacak."""
    sel = defaultdict(list)
    for f in frames:
        sel[sel_frame(f["video"])].append(f)
    rng = random.Random(seed)
    dipilih = []
    for sel_kunci in sorted(sel):
        daftar = sorted(sel[sel_kunci], key=lambda f: (f["video"], f["t"]))
        dipilih += rng.sample(daftar, min(per_sel, len(daftar)))
    rng.shuffle(dipilih)
    return dipilih


def tarik_pengganti(frames, gt, per_sel, rng):
    """Tambahkan ke gt["dipilih"] frame acak yang belum pernah dipilih pada sel yang punya kurang dari per_sel frame berorang."""
    terpilih = {kunci(d["video"], d["t"]) for d in gt["dipilih"]}
    kandidat = defaultdict(list)
    for f in frames:
        if kunci(f["video"], f["t"]) not in terpilih:
            kandidat[sel_frame(f["video"])].append(f)
    berorang = defaultdict(int)
    for d in gt["dipilih"]:
        lab = gt["kotak"].get(kunci(d["video"], d["t"]))
        if lab is None or lab["ada_orang"]:   # yang belum berlabel dihitung calon berorang
            berorang[sel_frame(d["video"])] += 1
    tambah = []
    for sel in sorted({sel_frame(d["video"]) for d in gt["dipilih"]}):
        kurang = per_sel - berorang[sel]
        if kurang > 0:
            daftar = sorted(kandidat[sel], key=lambda f: (f["video"], f["t"]))
            tambah += [{"video": f["video"], "t": f["t"]} for f in rng.sample(daftar, min(kurang, len(daftar)))]
    gt["dipilih"].extend(tambah)
    return tambah


def _simpan(path, gt):
    Path(path).write_text(json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")


PETUNJUK = (
    "\nGambar kotak BADAN orang yang berlatih saja: kepala sampai kaki dan sampai ujung lengan. Barbel, plate, rak, dan bangku "
    "TIDAK ikut, walau menonjol jauh keluar dari badan. Bila ada bagian tubuh keluar frame, kotak sampai tepi frame. "
    "Orang di latar belakang diabaikan; bila orang yang berlatih tidak terlihat sama sekali, tekan N.\n"
    "Tombol: ENTER atau A = mulai gambar (seret mouse, lalu ENTER atau SPASI; C = batal) | N = tidak ada orang | "
    "S = lewati | Q = simpan dan keluar.\n"
    "Jendela bisa muncul di belakang jendela lain; cek taskbar. Hasil tersimpan tiap satu frame, jadi boleh berhenti kapan saja."
)


def _teks(img, teks, org, warna):
    import cv2
    font, skala = cv2.FONT_HERSHEY_SIMPLEX, 0.6
    while skala > 0.35 and cv2.getTextSize(teks, font, skala, 2)[0][0] > img.shape[1] - org[0] - 6:
        skala -= 0.05   # frame potret sempit: kecilkan huruf supaya tidak terpotong
    cv2.putText(img, teks, org, font, skala, (0, 0, 0), 5)   # garis tepi hitam supaya terbaca di latar apa pun
    cv2.putText(img, teks, org, font, skala, warna, 2)


def _label(sisa, file_dari, folder, gt, path):
    import cv2
    print(PETUNJUK, flush=True)
    cv2.namedWindow(JENDELA, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(JENDELA, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass
    total = len(gt["dipilih"])
    antrian = list(sisa)
    while antrian:
        item = antrian.pop(0)
        k = kunci(item["video"], item["t"])
        img = cv2.imread(str(Path(folder) / file_dari[k]))
        if img is None:
            print(f"Gagal membuka frame {file_dari[k]}, dilewati.")
            continue
        h, w = img.shape[:2]
        skala = min(1.0, 950.0 / max(h, w))
        tampil = cv2.resize(img, (int(w * skala), int(h * skala))) if skala < 1.0 else img.copy()
        th, tw = tampil.shape[:2]
        hud = tampil.copy()
        _teks(hud, f"{len(gt['kotak']) + 1}/{total}  ENTER=gambar kotak  N=tidak ada orang  S=lewati  Q=keluar", (10, 28), (0, 255, 255))
        _teks(hud, "KOTAK = BADAN SAJA. Barbel dan plate TIDAK ikut.", (10, 58), (80, 80, 255))
        cv2.imshow(JENDELA, hud)
        kode = cv2.waitKey(0) & 0xFF
        try:
            ditutup = cv2.getWindowProperty(JENDELA, cv2.WND_PROP_VISIBLE) < 1
        except Exception:
            ditutup = True
        if ditutup:
            break
        tombol = chr(kode).lower()
        if kode == 27 or tombol == "q":
            break
        if tombol == "s":
            antrian.append(item)
            continue
        if tombol == "n":
            gt["kotak"][k] = {"ada_orang": False}
            _simpan(path, gt)
            continue
        if kode == 13 or tombol == "a":
            x, y, bw, bh = cv2.selectROI(JENDELA, tampil, showCrosshair=False, fromCenter=False)
            if bw > 5 and bh > 5:
                gt["kotak"][k] = {"ada_orang": True, "box": [round(x / tw, 5), round(y / th, 5),
                                                             round((x + bw) / tw, 5), round((y + bh) / th, 5)]}
                _simpan(path, gt)
                continue
        antrian.insert(0, item)
    cv2.destroyAllWindows()


def siapkan_gt(frames, folder, path, per_sel=10, seed=42, ganti=False):
    """Muat atau buat daftar frame berlabel, lalu buka jendela pelabelan bila belum lengkap. None bila belum selesai.

    ganti=True: frame berlabel "tidak ada orang" diganti frame acak lain dari sel yang sama sampai tiap sel punya per_sel frame berorang."""
    path = Path(path)
    if path.exists():
        gt = json.loads(path.read_text(encoding="utf-8"))
    else:
        dipilih = pilih_frame(frames, per_sel, seed)
        gt = {"seed": seed, "per_sel": per_sel, "dipilih": [{"video": f["video"], "t": f["t"]} for f in dipilih], "kotak": {}}
        _simpan(path, gt)
    file_dari = {kunci(f["video"], f["t"]): f["file"] for f in frames}
    ada = {kunci(d["video"], d["t"]) for d in gt["dipilih"]}
    if not ada <= set(file_dari):
        raise RuntimeError(f"Ada frame di {path.name} yang tidak ada pada daftar frame sekarang; hapus berkas itu untuk mengulang.")
    rng = random.Random(SEED_PENGGANTI)
    while True:
        if ganti:
            tambah = tarik_pengganti(frames, gt, per_sel, rng)
            if tambah:
                gt.setdefault("pengganti", {"seed": SEED_PENGGANTI, "frame": []})["frame"] += tambah
                _simpan(path, gt)
                print(f"{len(tambah)} frame pengganti ditarik dari sel yang sama dengan frame tanpa orang.", flush=True)
        sisa = [d for d in gt["dipilih"] if kunci(d["video"], d["t"]) not in gt["kotak"]]
        if not sisa:
            return gt
        print(f"Pelabelan kotak: {len(gt['dipilih']) - len(sisa)} dari {len(gt['dipilih'])} frame sudah berlabel.", flush=True)
        _label(sisa, file_dari, folder, gt, path)
        sisa = [d for d in gt["dipilih"] if kunci(d["video"], d["t"]) not in gt["kotak"]]
        if sisa:
            print(f"Pelabelan belum selesai ({len(sisa)} frame tersisa). Jalankan perintah yang sama untuk melanjutkan.")
            return None


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    luas = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / luas if luas > 0 else 0.0


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def hitung(catat, meta, gt, nama):
    """catat: hasil browser (A dan B per frame); meta: id -> frame; nama: {"A": model A, "B": model B}."""
    per = {kunci(meta[r["id"]]["video"], meta[r["id"]]["t"]): r for r in catat}
    hasil, tepat_per_model = {}, {}
    for m, model in nama.items():
        nilai_iou, tepat, latihan = [], [], []
        tanpa, salah, kosong, orang_lain = 0, 0, 0, 0
        for k, lab in gt["kotak"].items():
            rec = per.get(k)
            if rec is None:
                continue
            kb = rec[m].get("box")
            if not lab["ada_orang"]:
                kosong += 1
                orang_lain += int(kb is not None)
                continue
            n = iou(kb, lab["box"]) if kb else 0.0
            nilai_iou.append(n)
            tepat.append(n >= AMBANG_IOU)
            latihan.append(k.split("_", 1)[0])
            tanpa += int(kb is None)
            salah += int(kb is not None and n < AMBANG_IOU)
        n_uji = len(tepat)
        lo, hi = wilson(sum(tepat), n_uji)
        per_latihan = {}
        for lat in sorted(set(latihan)):
            idx = [i for i, l in enumerate(latihan) if l == lat]
            per_latihan[lat] = {"n": len(idx), "tepat": sum(tepat[i] for i in idx)}
        hasil[model] = {"n": n_uji, "tepat": sum(tepat), "persen": 100.0 * sum(tepat) / n_uji if n_uji else float("nan"),
                        "ci95_persen": [100 * lo, 100 * hi], "iou_rata2": sum(nilai_iou) / n_uji if n_uji else float("nan"),
                        "tanpa_deteksi": tanpa, "salah_kotak": salah, "frame_tanpa_orang": kosong,
                        "kotak_pada_orang_lain_di_frame_tanpa_orang": orang_lain, "per_latihan": per_latihan}
        tepat_per_model[m] = tepat
    a, b = tepat_per_model["A"], tepat_per_model["B"]
    hasil["berpasangan"] = {"keduanya_tepat": sum(x and y for x, y in zip(a, b)),
                            "hanya_A": sum(x and not y for x, y in zip(a, b)),
                            "hanya_B": sum(y and not x for x, y in zip(a, b)),
                            "keduanya_gagal": sum(not x and not y for x, y in zip(a, b))}
    return hasil


def cetak(ket):
    print(f"\nKetepatan kotak orang (IoU >= {kom(AMBANG_IOU, 2)}, kotak confidence tertinggi, kotak sebenarnya digambar peneliti)")
    for model, s in ket.items():
        if model == "berpasangan":
            continue
        print(f"  {model:23s} kotak tepat {s['tepat']}/{s['n']} = {kom(s['persen'])}% (95% CI {kom(s['ci95_persen'][0])}-"
              f"{kom(s['ci95_persen'][1])}) | rata-rata IoU {kom(s['iou_rata2'], 3)} | tanpa deteksi {s['tanpa_deteksi']} | "
              f"kotak salah {s['salah_kotak']} | kotak pada orang lain di frame tanpa orang yang berlatih "
              f"{s['kotak_pada_orang_lain_di_frame_tanpa_orang']}/{s['frame_tanpa_orang']}")
        print("    per latihan: " + ", ".join(f"{lat} {v['tepat']}/{v['n']}" for lat, v in s["per_latihan"].items()))
    p = ket["berpasangan"]
    print(f"  Berpasangan (frame yang sama): keduanya tepat {p['keduanya_tepat']} | hanya YOLO11n {p['hanya_A']} | "
          f"hanya YOLOv5nu {p['hanya_B']} | keduanya gagal {p['keduanya_gagal']}")
