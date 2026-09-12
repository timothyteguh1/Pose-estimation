"""Aplikasi web (bab 8.3.3 proposal) -- client-server: YOLO+MediaPipe+RF+rep
counting jalan di server, browser/HP cuma kirim video kamera & terima hasil.
Lihat src/app/live_pipeline.py utk pipeline inti.

video_frame_callback() jalan di thread terpisah dari script utama Streamlit,
jadi objek yang dipakai bareng (pipeline, hasil terakhir, log sesi) disimpan
sbg variabel Python biasa via closure, bukan akses st.session_state langsung
dari dalam callback.

Jalankan lokal:
    streamlit run app.py
"""
import base64
import io
import sys
import threading
import time
import wave
from collections import Counter, deque
from pathlib import Path

import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoHTMLAttributes

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.app.live_pipeline import LivePosturePipeline
from src.app.warnings_content import get_warning, get_warning_audio_path
from src.features.skeleton_draw import BODY_CONNECTIONS, landmark_pixel_points
from src.io_utils import open_video_capture

st.set_page_config(page_title="Pemantau Postur Gym", layout="wide", page_icon="🏋️")
st.markdown(
    """<style>
    /* Responsif hp/tablet: 1 kolom, video/gambar max-width 100%. */
    .block-container {padding-top: 1.5rem; padding-left: 1rem; padding-right: 1rem;}
    div[data-testid="stMetricValue"] {font-size: 2.2rem;}
    /* Batasi tinggi video/gambar ke viewport (video portrait bisa lebih
    tinggi dari layar) -- object-fit:contain jaga rasio asli. */
    img, video {
        max-width: 100% !important; max-height: 65vh !important;
        width: auto !important; height: auto !important; object-fit: contain;
        display: block; margin-left: auto; margin-right: auto;
    }
    div[data-testid="stFileUploaderDropzone"] {flex-wrap: wrap;}
    @media (max-width: 640px) {
        .block-container {padding-left: 0.6rem; padding-right: 0.6rem; padding-top: 1rem;}
        h1 {font-size: 1.35rem !important;}
        div[data-testid="stMetricValue"] {font-size: 1.6rem;}
        div[data-testid="stMetricLabel"] {font-size: 0.85rem;}
        .stButton button {width: 100%;}
    }
    </style>""",
    unsafe_allow_html=True,
)

st.title("🏋️ Aplikasi Web Pemantau Postur Beban Bebas")
st.caption("Derich Fitness Gym -- YOLOv11 + MediaPipe + Random Forest")

EXERCISES = {"Bench Press": "benchpress", "Squat": "squat", "Deadlift": "deadlift"}

# Redeteksi YOLO tiap 5 frame (bukan tiap frame, lihat yolo_redetect_every di
# live_pipeline.py) -- lebih cepat, akurasi verdict tidak turun di pengujian kita.
YOLO_REDETECT_EVERY = 5

# Kecepatan poll panel kanan (st.fragment run_every) -- murni refresh UI,
# tidak terkait kapan/bagaimana verdict postur dihitung (lihat blok SEG_*).
COMMIT_INTERVAL_SEC = 1.0

# Deteksi siklus gerakan (utk klasifikasi postur) -- terpisah dari
# OnlineRepCounter (utk hitung "Repetisi: N", tidak disentuh di sini). Lacak
# arah gerak sudut utama + nilai ekstrem; siklus selesai begitu sudut
# membalik >= SEG_HYSTERESIS_DEG dari ekstrem (dgn debounce SEG_MIN_SEC
# supaya noise kecil tidak kehitung sbg siklus).
SEG_HYSTERESIS_DEG = 8.0
SEG_MIN_SEC = 0.5

# Verdict 1 siklus = majority vote prediksi window mentah di siklus itu,
# kecuali window di PHASE_GATE_TOP_FRAC teratas rentang sudut (fase lockout,
# kesalahan bentuk tidak teramati di situ). None kalau window fase-kerja yg
# tersisa < MIN_SEG_WINDOWS (data kurang, tidak menebak).
PHASE_GATE_TOP_FRAC = 0.30
MIN_SEG_WINDOWS = 3


def _silent_wav_b64():
    """WAV hening super pendek (0.05 detik, ~450 byte) -- BUKAN audio
    peringatan sungguhan, cuma dipakai tombol "Aktifkan Suara" di bawah utk
    'membuka kunci' izin autoplay audio HP (lihat _UNLOCK_AUDIO_BTN_HTML).
    Dibuat langsung via modul `wave` bawaan Python (bukan baca file) supaya
    kecil & tidak nambah aset."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)  # 8-bit unsigned PCM, 128 = titik nol/hening
        w.setframerate(8000)
        w.writeframes(bytes([128]) * 400)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# Tombol HTML MENTAH + onclick JS murni client-side (BUKAN st.button()) --
# alasannya: st.button() memicu round-trip ke server Streamlit (rerun) dulu
# sebelum HTML/JS baru sampai ke browser; pada saat itu gesture tap
# pengguna sudah "basi" (browser tidak lagi anggap panggilan Audio().play()
# berikutnya sbg respons LANGSUNG dari tap), jadi kebijakan autoplay HP
# (terutama Chrome/Safari Android & iOS -- diduga kuat penyebab suara
# peringatan tidak bunyi sama sekali saat ditest di HP, lihat diskusi
# proyek) tetap memblokirnya. Di sini Audio().play() dipanggil LANGSUNG di
# dalam handler onclick (tanpa nunggu Streamlit sama sekali) supaya
# gesture-nya masih "segar" -- kebanyakan browser lalu mengizinkan audio
# BERIKUTNYA di tab/origin yg sama (termasuk <audio autoplay> yg disisipkan
# render_audio() belakangan) diputar otomatis tanpa gesture baru lagi.
# Placeholder __B64__ diganti via str.replace() (BUKAN .format()/f-string)
# supaya kurung kurawal JS di dalamnya tidak perlu di-escape.
_UNLOCK_AUDIO_BTN_HTML = """<button onclick="
        var a = new Audio('data:audio/wav;base64,__B64__');
        a.play().then(function(){ this.innerText = '🔊 Suara sudah aktif'; this.disabled = true; })
                .catch(function(){ this.innerText = '⚠️ Gagal, coba tap lagi'; });
    " style="width:100%;padding:0.5rem;margin-bottom:0.5rem;border-radius:0.5rem;
             border:1px solid #4CAF50;background:#e8f5e9;color:#1b5e20;
             font-size:0.9rem;cursor:pointer;">
        🔊 Aktifkan Suara Peringatan (tap sekali di awal sesi -- khusus HP)
    </button>"""

# Kontrol utama SENGAJA ditumpuk vertikal (bukan st.columns berdampingan) --
# st.columns bisa jadi sempit banget kalau viewport-nya hp/tablet (portrait),
# jadi 1 kolom penuh lebih aman & konsisten di semua ukuran layar.
with st.container(border=True):
    exercise_label = st.selectbox("🏋️ Pilih Latihan", list(EXERCISES.keys()))
    exercise = EXERCISES[exercise_label]
    source_mode = st.radio("📷 Sumber Video", ["Kamera Live", "Upload Video"], horizontal=True)
    with st.expander("⚙️ Pengaturan tambahan"):
        # Proposal tidak menetapkan durasi persiapan tertentu, jadi ini aman
        # dibuat adjustable oleh pengguna.
        countdown_sec = st.slider(
            "Durasi persiapan (detik)", min_value=0, max_value=30, value=5, step=1,
            help="Waktu 'BERSIAP...' sebelum repetisi mulai dihitung -- atur sesuai kebutuhan.",
        )
        # Default 0.0 (mati) -- efeknya beda jauh per exercise di data kita,
        # jadi tidak dipasang angka baku, user coba sendiri lewat slider.
        confidence_threshold = st.slider(
            "Ambang keyakinan klasifikasi RF (confidence threshold)",
            min_value=0.0, max_value=1.0, value=0.0, step=0.05,
            help="Kelas cuma dipakai kalau RF cukup yakin (fraksi pohon yg setuju >= "
                 "angka ini). 0.0 = mati (perilaku lama, RF selalu pilih suara terbanyak "
                 "walau tipis). Makin tinggi = makin sering '(belum ada window penuh)' "
                 "tapi yg lolos lebih bisa dipercaya -- coba-coba sendiri, efeknya beda "
                 "tiap exercise (sudah diukur, lihat diskusi proyek).",
        )
        # Koreksi manual kalau kamera/video hasil rekaman selfie ke-mirror.
        flip_camera = st.checkbox(
            "🔄 Flip kamera horizontal",
            value=False,
            help="Nyalakan kalau video (live ATAU upload) terlihat terbalik kiri-kanan "
                 "(tangan kanan tampil sbg tangan kiri, dst) -- terutama utk video hasil "
                 "rekaman selfie dari aplikasi kamera bawaan HP.",
        )
        # Koreksi manual kalau frame dari kamera HP tertentu datang landscape
        # walau HP dipegang tegak -- server tidak bisa deteksi otomatis.
        rotate_camera = st.selectbox(
            "🔃 Putar orientasi kamera",
            ["Tidak diputar", "90° searah jarum jam", "90° berlawanan jarum jam", "180°"],
            index=0,
            help="Nyalakan kalau video (live ATAU upload) kepotong/kelihatan landscape "
                 "padahal harusnya tegak (portrait) -- coba tiap pilihan sampai orientasinya "
                 "benar. Beda HP bisa butuh arah putaran berbeda.",
        )

if (
    "exercise" not in st.session_state
    or st.session_state["exercise"] != exercise
    or st.session_state.get("countdown_sec") != countdown_sec
):
    old_worker = st.session_state.get("live_worker")
    if old_worker is not None:
        old_worker.stop()  # hentikan thread lama SEBELUM ganti pipeline/exercise (lihat LiveWorker)
    try:
        st.session_state["pipeline"] = LivePosturePipeline(exercise, countdown_sec=float(countdown_sec),
                                                       yolo_redetect_every=YOLO_REDETECT_EVERY)
        st.session_state["exercise"] = exercise
        st.session_state["countdown_sec"] = countdown_sec
        st.session_state["session_start_box"] = {"t0": None}
        st.session_state["result_box"] = {
            "rep_count": 0, "primary_angle": None, "countdown_remaining": None,
            # DEBUG: ukuran frame mentah dari kamera, sebelum diproses.
            "debug_frame_shape": None,
        }
        st.session_state["session_log"] = []  # [{seg_no, class, warning, image, t}] -- 1 entri per siklus gerakan
        st.session_state["played_log_len"] = 0  # audio: jumlah entri session_log yg sudah diputar
        # seg_state: state mesin hysteresis deteksi siklus. seg_windows: buffer
        # prediksi window mentah sejak batas siklus terakhir. seg_evidence:
        # frame fase terdalam siklus, dipakai jadi snapshot ringkasan.
        st.session_state["seg_state"] = {"direction": None, "extreme_val": None, "last_confirm_t": None}
        st.session_state["seg_windows"] = []
        st.session_state["seg_evidence"] = {
            "angle": None, "img": None, "skeleton_points_px": None,
            "n_points": 0, "crop_box": None, "bbox": None,
        }
        st.session_state["load_error"] = None
        st.session_state["live_worker"] = None  # dibuat di bawah, setelah pipeline/result_box dll siap
    except FileNotFoundError as e:
        st.session_state["pipeline"] = None
        st.session_state["load_error"] = str(e)

if st.session_state.get("load_error"):
    st.error(f"⚠️ Model untuk **{exercise_label}** belum tersedia:\n\n{st.session_state['load_error']}")
    st.stop()

pipeline = st.session_state["pipeline"]
# Update langsung via attribute -- tidak perlu reconstruct pipeline
# (reload YOLO/MediaPipe/RF) cuma buat ganti 1 angka.
pipeline.class_confidence_threshold = confidence_threshold
session_start_box = st.session_state["session_start_box"]
result_box = st.session_state["result_box"]
session_log = st.session_state["session_log"]
seg_state = st.session_state["seg_state"]
seg_windows = st.session_state["seg_windows"]
seg_evidence = st.session_state["seg_evidence"]


def _display_name(class_name):
    """Nama kelas TANPA suffix _concentric/_eccentric, khusus tampilan ke user
    (mis. "squat_kneeinward_concentric" -> "squat_kneeinward") -- fase tetap
    dipakai apa adanya di internal (get_warning, audio, seg_windows)."""
    if class_name is None:
        return None
    return class_name.rsplit("_", 1)[0]


def _majority_class(preds):
    """Kelas paling sering muncul dari daftar prediksi (None diabaikan) --
    dipakai _phase_gated_seg_class() utk verdict 1 siklus gerakan."""
    non_none = [p for p in preds if p is not None]
    if not non_none:
        return None
    return Counter(non_none).most_common(1)[0][0]


def _phase_gated_seg_class(seg_windows):
    """Verdict 1 siklus gerakan dari seg_windows (list (raw_class,
    primary_angle) per window sejak siklus terakhir): buang window fase
    lockout (PHASE_GATE_TOP_FRAC teratas rentang sudut, lockout = sudut
    utama maks), lalu majority vote sisanya. None kalau window fase-kerja
    < MIN_SEG_WINDOWS."""
    pairs = [(c, a) for (c, a) in seg_windows if c is not None and a is not None]
    if len(pairs) < MIN_SEG_WINDOWS:
        return None
    angles = [a for (_, a) in pairs]
    lo, hi = min(angles), max(angles)
    cut = lo + (hi - lo) * (1.0 - PHASE_GATE_TOP_FRAC)
    kept = [c for (c, a) in pairs if a <= cut]
    if len(kept) < MIN_SEG_WINDOWS:
        return None
    return _majority_class(kept)


def _update_seg_state(seg_state, a, t):
    """Hysteresis 1 langkah: lacak arah gerak sudut utama + nilai ekstrem.
    Return True sekali saat 1 puncak (lockout) terkonfirmasi -- 1 siklus
    turun-naik selesai. Pembalikan arah lebih cepat dari SEG_MIN_SEC
    dianggap noise, diabaikan sbg batas siklus."""
    if a is None:
        return False
    if seg_state["extreme_val"] is None:
        seg_state["extreme_val"], seg_state["direction"] = a, "up"
        return False
    if seg_state["direction"] == "up":
        if a >= seg_state["extreme_val"]:
            seg_state["extreme_val"] = a
            return False
        if seg_state["extreme_val"] - a < SEG_HYSTERESIS_DEG:
            return False
        too_soon = (seg_state["last_confirm_t"] is not None
                    and (t - seg_state["last_confirm_t"]) < SEG_MIN_SEC)
        seg_state["extreme_val"] = a
        if too_soon:
            return False
        seg_state["direction"] = "down"
        seg_state["last_confirm_t"] = t
        return True  # PUNCAK terkonfirmasi -> 1 siklus selesai
    else:  # direction == "down", menuju lembah -- tidak memicu batas siklus
        if a <= seg_state["extreme_val"]:
            seg_state["extreme_val"] = a
            return False
        if a - seg_state["extreme_val"] < SEG_HYSTERESIS_DEG:
            return False
        too_soon = (seg_state["last_confirm_t"] is not None
                    and (t - seg_state["last_confirm_t"]) < SEG_MIN_SEC)
        seg_state["direction"] = "up"
        seg_state["extreme_val"] = a
        if not too_soon:
            seg_state["last_confirm_t"] = t
        return False


def compute_result(img, t):
    """Bagian BERAT (YOLO+MediaPipe+RF+rep counting) -- panggil pipeline inti
    & simpan efek samping (result_box/session_log). TIDAK menggambar apa pun
    (digambar terpisah oleh draw_overlay(), lihat di bawah) -- pemisahan ini
    MURNI detail implementasi server (arsitektur client-server bab 6.4
    proposal TIDAK berubah: server tetap satu-satunya yg proses YOLO+
    MediaPipe+RF+rep counting, client cuma render; sini cuma mengatur GILIRAN
    proses di sisi server, bukan pindahkan proses ke client).

    Dipakai 2 cara:
    - Mode Kamera Live: dipanggil dari THREAD TERPISAH (LiveWorker di bawah)
      supaya video yg ditampilkan (di video_frame_callback) tetap mulus
      mengikuti frame rate kamera, TIDAK ikut lambat walau pipeline penuh ini
      cuma sanggup ~7-8x/detik di CPU biasa (diukur langsung, lihat diskusi
      proyek) -- overlay (skeleton/teks) yg "nunggu" hasil terbaru, BUKAN
      videonya. Sebelum perubahan ini, video_frame_callback memanggil
      pipeline penuh LANGSUNG di thread callback webrtc -- video jadi ikut
      selambat proses (~130ms/frame, terlihat kayak slow-motion).
    - Mode Upload Video: tetap dipanggil SINKRON (langsung disusul
      draw_overlay() pada frame yg SAMA persis) -- perilakunya TIDAK berubah
      sama sekali dari sebelumnya, cuma dipecah jadi 2 pemanggilan fungsi."""
    result = pipeline.process_frame(img, t, fps=30.0)
    in_countdown = result["countdown_remaining"] is not None

    # Disalin ke result_box TANPA SYARAT (beda dari rep_count/primary_angle
    # di bawah yg cuma diisi kalau BUKAN countdown) -- supaya panel sidebar
    # (render_result(), widget Streamlit BIASA yg update tiap poll 0.5 detik,
    # TIDAK terikat rendering video WebRTC) py jalur TAMPILAN CADANGAN utk
    # hitung mundur, independen dari teks "BERSIAP..." yg dibakar ke frame
    # video (draw_overlay()). Alasan: dicurigai di HP, video WebRTC kadang
    # butuh sesaat utk mulai render (negosiasi kamera/kodek) SEBELUM frame
    # pertama kelihatan -- kalau itu terjadi, teks di DALAM video ikut
    # "hilang"/terlewat, padahal sidebar (jalur terpisah) seharusnya tetap
    # update normal.
    result_box["countdown_remaining"] = result["countdown_remaining"]

    # Konversi ke koordinat piksel frame utuh sekali di sini, supaya
    # draw_overlay() tinggal gambar tanpa perlu tau crop_box lagi.
    skeleton_points_px = {}
    n_points = 0
    if result["crop_box"] is not None:
        x1, y1, x2, y2 = result["crop_box"]
        crop_w, crop_h = x2 - x1, y2 - y1
        raw_pts = landmark_pixel_points(result["row"], crop_w, crop_h)
        body_indices = {i for pair in BODY_CONNECTIONS for i in pair}
        skeleton_points_px = {i: (px + x1, py + y1) for i, (px, py) in raw_pts.items() if i in body_indices}
        n_points = len(skeleton_points_px)

    warning = None
    if not in_countdown:
        raw_class = result["predicted_class"]
        a = result["primary_angle"]

        result_box["rep_count"] = result["rep_count"]  # angka RESMI, dari OnlineRepCounter, TIDAK disentuh
        result_box["primary_angle"] = a

        # Buffer prediksi window mentah sejak batas siklus terakhir -- tidak
        # dikosongkan saat bbox kosong (window itu otomatis tersaring di
        # _phase_gated_seg_class, drpd verdict jadi None terus).
        seg_windows.append((raw_class, a))
        # Frame bukti snapshot = fase paling dalam siklus (sudut minimum),
        # bukan frame batas-siklus (orang berdiri tegak = selalu "benar").
        if a is not None and (seg_evidence["angle"] is None or a < seg_evidence["angle"]):
            seg_evidence.update(angle=a, img=img.copy(),
                                skeleton_points_px=skeleton_points_px, n_points=n_points,
                                crop_box=result["crop_box"], bbox=result["bbox"])

        # Deteksi batas siklus (hysteresis+debounce, TERPISAH dari
        # OnlineRepCounter -- lihat _update_seg_state & blok komentar SEG_*).
        seg_done = _update_seg_state(seg_state, a, t)

        if seg_done:
            # None kalau window fase-kerja kurang dari MIN_SEG_WINDOWS
            # (data kurang, tidak menebak).
            seg_class = _phase_gated_seg_class(seg_windows)
            seg_windows.clear()
            seg_warning = get_warning(seg_class)
            # Snapshot dari frame bukti (fase terdalam) kalau ada, fallback
            # ke frame batas-siklus ini kalau tidak.
            ev = seg_evidence
            if ev["img"] is not None:
                snap_img = ev["img"]
                snap_result = dict(result, predicted_class=seg_class,
                                   crop_box=ev["crop_box"], bbox=ev["bbox"])
                snap_skel, snap_npts = ev["skeleton_points_px"], ev["n_points"]
            else:
                snap_img = img
                snap_result = dict(result, predicted_class=seg_class)
                snap_skel, snap_npts = skeleton_points_px, n_points
            seg_evidence.update(angle=None, img=None, skeleton_points_px=None,
                                n_points=0, crop_box=None, bbox=None)
            computed_now = {
                "result": snap_result, "in_countdown": False, "skeleton_points_px": snap_skel,
                "n_points": snap_npts, "warning": seg_warning,
            }
            # Ringkasan sesi -- 1 entri per siklus selesai (beda dari
            # "Repetisi: N" resmi), termasuk siklus yg benar, bukan cuma salah.
            warning = seg_warning
            session_log.append({
                "rep_no": len(session_log) + 1,  # nomor SIKLUS, bukan repetisi resmi
                "class": seg_class,
                "warning": seg_warning,
                "image": draw_overlay(snap_img, computed_now),  # frame fase kerja, bukan frame berdiri
                "t": t,
            })

    return {
        "result": result, "in_countdown": in_countdown, "skeleton_points_px": skeleton_points_px,
        "n_points": n_points, "warning": warning,
    }


def draw_overlay(raw_frame, computed):
    """Gambar bbox/skeleton (murni cv2, tidak panggil pipeline) -- aman
    dipanggil tiap frame walau `computed` sedikit lebih lama (mode Live
    async), video tetap frame terbaru, cuma overlay yg sedikit telat."""
    canvas = raw_frame.copy()
    result = computed["result"]
    color = (230, 160, 40) if result["bbox"] else (140, 140, 140)

    if result["crop_box"] is not None:
        x1, y1, x2, y2 = result["crop_box"]
        pts = computed["skeleton_points_px"]
        for a_idx, b_idx in BODY_CONNECTIONS:
            if a_idx in pts and b_idx in pts:
                cv2.line(canvas, pts[a_idx], pts[b_idx], color, 2)
        for pt in pts.values():
            cv2.circle(canvas, pt, 3, color, -1)
        if result["bbox"] is not None:
            bx1, by1, bx2, by2 = result["bbox"]
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (60, 200, 60), 2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 180, 255), 2)

    if computed["in_countdown"]:
        secs_left = int(result["countdown_remaining"]) + 1
        n_points = computed["n_points"]
        status = "badan terdeteksi lengkap" if n_points == 17 else f"baru {n_points}/17 titik badan terlihat"
        for i, line in enumerate([f"BERSIAP... {secs_left}", f"({status}, atur posisi kamera)"]):
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            h, w = canvas.shape[:2]
            cv2.putText(canvas, line, ((w - tw) // 2, h // 2 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2, cv2.LINE_AA)
        return canvas

    # TEKS PREDIKSI/SUDUT/REPETISI/PERINGATAN SENGAJA TIDAK LAGI DIGAMBAR DI
    # ATAS VIDEO (dihapus atas permintaan user, lihat diskusi proyek).
    # Teks prediksi/repetisi sengaja tidak digambar di video (bisa beda
    # waktu dgn panel kanan krn worker async) -- satu-satunya sumber
    # tampilan kelas = render_result(), video cuma bbox+skeleton.
    return canvas


class LiveWorker:
    """Thread async khusus Kamera Live -- jalankan compute_result() di
    latar belakang, selalu proses frame TERBARU (frame lama yg belum
    sempat diproses dibuang, tidak diantre), supaya video tetap mulus
    walau proses beratnya lambat."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pending = None  # (img, t) frame TERBARU yg belum diproses
        self._latest_computed = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, img, t):
        with self._lock:
            self._pending = (img, t)  # SELALU timpa -- cuma proses yg TERBARU

    def latest(self):
        with self._lock:
            return self._latest_computed

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            with self._lock:
                job, self._pending = self._pending, None
            if job is None:
                time.sleep(0.005)
                continue
            img, t = job
            computed = compute_result(img, t)
            with self._lock:
                self._latest_computed = computed


# Dibuat sekali per pipeline/exercise -- LiveWorker butuh compute_result()
# sudah terdefinisi. Upload Video tidak memakai ini (tetap sinkron).
if st.session_state.get("live_worker") is None:
    st.session_state["live_worker"] = LiveWorker()
live_worker = st.session_state["live_worker"]


_ROTATE_CODES = {
    "90° searah jarum jam": cv2.ROTATE_90_CLOCKWISE,
    "90° berlawanan jarum jam": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180°": cv2.ROTATE_180,
}


def _apply_rotation(img, choice):
    """Putar frame mentah sebelum diproses -- manual (bukan auto-detect)
    krn server cuma terima array piksel, tidak tau orientasi fisik device.
    Dipakai di Live & Upload spy konsisten."""
    code = _ROTATE_CODES.get(choice)
    return cv2.rotate(img, code) if code is not None else img


def _waiting_overlay(img):
    """Placeholder SINGKAT (cuma dipakai < 1 detik di awal sesi, sebelum
    worker sempat menghasilkan compute_result() pertama kali)."""
    canvas = img.copy()
    text = "Memulai..."
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    h, w = canvas.shape[:2]
    cv2.putText(canvas, text, ((w - tw) // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 200, 255), 2, cv2.LINE_AA)
    return canvas


LIVE_RETURN_MAX_W = 640  # lihat komentar video_frame_callback


def video_frame_callback(frame):
    """Thread terpisah dari LiveWorker (jalur callback streamlit-webrtc) --
    sengaja ringan: submit() frame ke worker (non-blocking), tidak panggil
    compute_result()/pipeline di sini, supaya video tetap mulus walau
    proses beratnya lambat. Skeleton/bbox tetap digambar via draw_overlay()
    (dikecilkan ke LIVE_RETURN_MAX_W sebelum encode WebRTC)."""
    img = frame.to_ndarray(format="bgr24")
    img = _apply_rotation(img, rotate_camera)
    result_box["debug_frame_shape"] = img.shape  # DEBUG: ukuran setelah rotasi manual (kalau dipilih)
    # getUserMedia tidak membalik data mentah kamera (efek mirror yg biasa
    # terlihat itu CSS tampilan doang) -- sudah diverifikasi tidak ke-mirror
    # di data yg diterima server.
    if flip_camera:
        img = cv2.flip(img, 1)
    if session_start_box["t0"] is None:
        session_start_box["t0"] = time.time()
    t = time.time() - session_start_box["t0"]

    live_worker.submit(img, t)
    computed = live_worker.latest()
    canvas = draw_overlay(img, computed) if computed is not None else _waiting_overlay(img)

    h, w = canvas.shape[:2]
    if w > LIVE_RETURN_MAX_W:
        out = cv2.resize(canvas, (LIVE_RETURN_MAX_W, int(h * (LIVE_RETURN_MAX_W / w))),
                          interpolation=cv2.INTER_AREA)
    else:
        out = canvas
    return av.VideoFrame.from_ndarray(out, format="bgr24")


st.divider()
col_video, col_info = st.columns([2, 1], gap="large")

with col_info:
    st.subheader("📊 Hasil Real-time")
    st.markdown(_UNLOCK_AUDIO_BTN_HTML.replace("__B64__", _silent_wav_b64()), unsafe_allow_html=True)
    countdown_slot = st.empty()
    status_slot = st.empty()  # 1 kartu gabungan (repetisi+sudut+prediksi+peringatan)
    audio_slot = st.empty()


def render_result():
    # Hitung mundur "BERSIAP..." -- JALUR TAMPILAN KEDUA (widget Streamlit
    # biasa, lewat result_box["countdown_remaining"]), TERPISAH dari teks
    # yg dibakar ke frame video di draw_overlay(). Diduga di HP, video WebRTC
    # butuh sesaat sebelum benar2 tampil (negosiasi kamera/kodek) -- kalau
    # itu terjadi, teks DI DALAM video bisa terlewat, tapi panel sidebar ini
    # (poll independen, bukan bagian video) tetap seharusnya update normal.
    countdown = result_box.get("countdown_remaining")
    if countdown is not None:
        secs_left = int(countdown) + 1
        countdown_slot.info(f"⏳ **BERSIAP... {secs_left} detik** -- atur posisi kamera, pastikan seluruh badan terlihat")
    else:
        countdown_slot.empty()

    # Kelas cuma tampil setelah 1 siklus selesai dikonfirmasi, sumbernya
    # sama dgn render_summary() (session_log) -- panel & Ringkasan Sesi tidak
    # mungkin saling bertentangan. "Repetisi" (OnlineRepCounter) bisa beda
    # angka dari "Siklus" (deteksi sendiri) -- disengaja, 2 sinyal berbeda.
    angle_txt = f"{result_box['primary_angle']:.1f}°" if result_box["primary_angle"] is not None else "-"
    with status_slot.container(border=True):
        c1, c2 = st.columns(2)
        c1.metric("Repetisi", result_box["rep_count"])
        c2.metric("Sudut Utama", angle_txt)
        if not session_log:
            st.caption("Menganalisis siklus gerakan pertama...")
        else:
            last = session_log[-1]
            label = _display_name(last["class"]) or "tidak cukup yakin"
            if last["warning"]:
                st.error(f"⚠️ Siklus {last['rep_no']}: {label} -- {last['warning']}")
            elif last["class"]:
                st.success(f"✅ Siklus {last['rep_no']}: {label}")
            else:
                st.caption(f"Siklus {last['rep_no']}: model kurang yakin (data kurang)")
            st.caption("Riwayat lengkap semua siklus ada di 'Ringkasan Sesi' di bawah.")
        # DEBUG: ukuran frame mentah dari kamera.
        shp = result_box.get("debug_frame_shape")
        if shp is not None:
            st.caption(f"🔧 DEBUG: ukuran frame mentah dari kamera = {shp[1]}x{shp[0]} (lebar x tinggi)")


def render_audio():
    """Putar audio peringatan (file .mp3 pre-recorded, lihat
    warnings_content.py) -- cuma utk entry session_log yg ada warning-nya,
    tetap ditandai "sudah dilihat" biar tidak diputar ulang."""
    played = st.session_state.get("played_log_len", 0)
    if len(session_log) > played:
        entry = session_log[-1]
        audio_path = get_warning_audio_path(entry["class"])
        # Jeda minimal 3 detik antar audio -- kalau kurang, elemen <audio>
        # lama ke-timpa yg baru sebelum selesai diputar (kedengaran kepotong).
        now = time.time()
        last_played_at = st.session_state.get("last_audio_at", 0.0)
        if audio_path is not None and (now - last_played_at) < 3.0:
            audio_path = None  # lewati pemutaran, tapi entry TETAP ditandai sudah dilihat
        if audio_path is not None:
            st.session_state["last_audio_at"] = now
            # st.audio() tidak punya parameter `key` di versi streamlit ini,
            # jadi audio_path yg identik berulang bikin StreamlitDuplicateElementId.
            # Solusi: tag <audio> HTML manual dgn komentar nomor urut entry
            # biar stringnya selalu unik (komentar HTML tidak pengaruhi audio).
            b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
            audio_slot.markdown(
                f"<!-- warn_audio_{len(session_log)} -->"
                f'<audio autoplay="true" style="display:none">'
                f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
                unsafe_allow_html=True,
            )
        st.session_state["played_log_len"] = len(session_log)


def render_summary():
    # DIPERLUAS atas permintaan -- ringkasan sekarang menampilkan SEMUA
    # siklus gerakan (benar maupun salah), TAPI DI-GROUP PER KELAS supaya
    # tidak numpuk foto identik: kalau kesalahan yg SAMA muncul lagi di siklus
    # lain (mis. siklus 1 & 3 sama2 "armsspread"), digabung jadi 1 kartu
    # ("Siklus 1, 3 -- armsspread", 1 foto wakil), BUKAN 2 kartu terpisah.
    # Begitu kelasnya BEDA (mis. siklus 5 ternyata "backround", bukan
    # "armsspread" lagi), itu otomatis jadi kartu baru krn key group-nya beda.
    # Grouping ini MURNI presentasi (session_log mentah tetap granular 1
    # entri/siklus, tidak diubah) -- proposal (ruang lingkup #15e) cuma
    # mewajibkan gambar utk KESALAHAN; menampilkan siklus BENAR & meng-group
    # itu tambahan presentasi semata, bukan perubahan metodologi klasifikasi.
    #
    # "Total Repetisi" (OnlineRepCounter, resmi) & "Siklus Dianalisis"
    # (deteksi sendiri, lihat blok SEG_* di compute_result) SENGAJA
    # ditampilkan TERPISAH -- keduanya sinyal berbeda (lihat diskusi proyek
    # 2026-09-11: rep counting != klasifikasi postur, tidak wajib sama angka).
    st.divider()
    st.subheader("📋 Ringkasan Sesi")
    c1, c2 = st.columns(2)
    c1.metric("Total Repetisi", result_box["rep_count"])
    c2.metric("Siklus Dianalisis", len(session_log))
    if not session_log:
        st.info("Belum ada siklus gerakan yang tercatat pada sesi ini.")
        return

    # Pisah dulu confirmed (class bukan None) dari unconfirmed -- siklus
    # None (data kurang) tidak ditampilkan sbg kartu, cuma disebut jumlahnya.
    confirmed = [e for e in session_log if e["class"] is not None]
    n_unconfirmed = len(session_log) - len(confirmed)
    if not confirmed:
        st.info("Belum ada siklus dengan data cukup untuk diklasifikasi.")
        if n_unconfirmed:
            st.caption(f"({n_unconfirmed} siklus terdeteksi tapi datanya kurang -- "
                       f"mis. sebagian badan tidak terlihat kamera.)")
        return
    n_wrong = sum(1 for e in confirmed if e["warning"])
    if n_wrong == 0:
        st.success(f"Semua {len(confirmed)} siklus gerakan terpantau postur BENAR. 👍")
    else:
        st.write(f"**{n_wrong} dari {len(confirmed)} siklus gerakan terdeteksi postur salah:**")
    if n_unconfirmed:
        st.caption(f"({n_unconfirmed} siklus lain datanya kurang, tidak cukup utk diklasifikasi -- "
                   f"tidak ditampilkan di bawah)")

    groups = {}
    order = []
    for entry in confirmed:
        key = _display_name(entry["class"])
        if key not in groups:
            groups[key] = {"segs": [], "warning": entry["warning"], "image": entry["image"]}
            order.append(key)
        groups[key]["segs"].append(entry["rep_no"])

    # 3 kolom galeri OK di desktop, tapi di hp (sempit) st.columns otomatis
    # menumpuk 1 kolom -- tetap rapi tanpa kode tambahan.
    cols = st.columns(3)
    for i, key in enumerate(order):
        g = groups[key]
        segs_txt = ", ".join(str(r) for r in g["segs"])
        with cols[i % 3]:
            st.image(cv2.cvtColor(g["image"], cv2.COLOR_BGR2RGB),
                      caption=f"Siklus {segs_txt} -- {key}",
                      width="stretch")
            if g["warning"]:
                st.caption(f"⚠️ {g['warning']}")
            else:
                st.caption("✅ Postur benar")


@st.fragment(run_every=COMMIT_INTERVAL_SEC)
def _live_panel():
    """Panel kanan (repetisi/prediksi/peringatan/audio/ringkasan) sbg fragmen
    independen (st.fragment) -- bukan st.rerun() penuh, supaya webrtc_streamer()
    (dipanggil di luar fragmen ini) tidak ikut di-render ulang tiap refresh."""
    render_result()
    render_audio()
    # Ringkasan ditampilkan terus-menerus selama sesi (bukan cuma setelah
    # tombol STOP) -- riwayat yg terus tumbuh terlihat selama latihan.
    render_summary()


if source_mode == "Kamera Live":
    with col_video:
        st.subheader("📹 Kamera")
        webrtc_ctx = webrtc_streamer(
            key=f"posture-{exercise}",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            # facingMode="user" = kamera depan/selfie. Tanpa aspectRatio/
            # width/height eksplisit -- browser HP tertentu tidak menghormati
            # constraint itu (selalu balik landscape meski HP dipegang
            # tegak), constraint minimal justru kasih hasil portrait yg benar.
            media_stream_constraints={
                "video": {"facingMode": {"ideal": "user"}},
                "audio": False,
            },
            # Default streamlit-webrtc cuma style width:100% (tanpa height/
            # objectFit) -- video bisa kepotong di tampilan normal (beda dari
            # fullscreen native yg render ulang sesuai rasio asli).
            video_html_attrs=VideoHTMLAttributes(
                autoPlay=True, controls=True,
                style={"width": "100%", "height": "auto", "objectFit": "contain"},
            ),
        )

    # Di luar fragmen sengaja -- webrtc_streamer() cuma dipanggil sekali per
    # full script run, tidak ikut ter-panggil ulang tiap _live_panel() refresh.
    _live_panel()

else:  # Upload Video
    # Loop di bawah (YOLO+MediaPipe+RF per frame) jalan sinkron di thread
    # utama -- kunci uploader+tombol selama "analyzing" aktif, supaya tidak
    # ada upload baru nyelonong di tengah proses lama.
    is_analyzing = st.session_state.get("analyzing", False)
    with col_video:
        st.subheader("📹 Video")
        uploaded = st.file_uploader("Upload video latihan (mp4/mov)", type=["mp4", "mov", "avi"],
                                     disabled=is_analyzing)
        video_slot = st.empty()
        progress_slot = st.empty()
        start_btn = st.button("▶️ Mulai Analisis", disabled=uploaded is None or is_analyzing)
        if is_analyzing:
            st.caption("⏳ Video sebelumnya masih diproses -- tunggu sampai selesai sebelum upload lagi.")

    if uploaded is not None and start_btn:
        st.session_state["analyzing"] = True
        try:
            tmp_path = Path("data") / "_upload_tmp.mp4"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_bytes(uploaded.read())

            # Reset pipeline/log tiap mulai analisis video baru
            st.session_state["pipeline"] = LivePosturePipeline(exercise, countdown_sec=float(countdown_sec),
                                                       yolo_redetect_every=YOLO_REDETECT_EVERY)
            pipeline = st.session_state["pipeline"]
            pipeline.class_confidence_threshold = confidence_threshold  # pipeline BARU, set lagi
            session_log.clear()
            result_box.update({"rep_count": 0, "primary_angle": None, "countdown_remaining": None})
            st.session_state["played_log_len"] = 0
            # Reset juga state deteksi siklus, supaya buffer video sebelumnya
            # tidak bocor ke video baru ini.
            seg_state.update({"direction": None, "extreme_val": None, "last_confirm_t": None})
            seg_windows.clear()
            seg_evidence.update(angle=None, img=None, skeleton_points_px=None,
                                n_points=0, crop_box=None, bbox=None)

            cap = open_video_capture(str(tmp_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = _apply_rotation(frame, rotate_camera)
                if flip_camera:
                    frame = cv2.flip(frame, 1)
                t = idx / fps
                computed = compute_result(frame, t)  # frame resolusi asli -- akurasi tidak tersentuh
                canvas = draw_overlay(frame, computed)
                # Preview di-update tiap frame (bukan skip-frame) & dikecilkan
                # ke 480px -- biar tidak patah-patah/freeze lewat tunnel,
                # sementara compute_result() tetap proses frame resolusi asli.
                preview_w = 480
                ph, pw = canvas.shape[:2]
                preview_h = int(ph * (preview_w / pw))
                preview = cv2.resize(canvas, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
                video_slot.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), channels="RGB",
                                  width="stretch")
                progress_slot.progress(min(1.0, idx / max(1, n_frames)))
                render_result()  # panel kanan ikut update sinkron dgn video (bug sebelumnya: cuma di-render 1x di akhir)
                render_audio()
                idx += 1
            cap.release()
            tmp_path.unlink(missing_ok=True)
            progress_slot.empty()
            st.success("Analisis video selesai.")
        finally:
            # Wajib finally -- kalau video rusak/error di tengah loop,
            # uploader tidak boleh terkunci selamanya.
            st.session_state["analyzing"] = False

    render_result()
    render_audio()
    if session_log:
        render_summary()

st.caption("Kelas postur dipajang setelah 1 siklus gerakan (turun-naik) selesai "
            "dikonfirmasi -- bukan tiap detik -- supaya tidak menuduh salah sebelum "
            "siklusnya benar-benar tuntas.")
