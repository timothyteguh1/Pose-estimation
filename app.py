"""Aplikasi web (bab 8.3.3 proposal) -- Streamlit, gaya sama seperti Ko et al.
(Streamlit.py mereka: st.selectbox exercise, st.sidebar utk info, st.error utk
peringatan). Client-server: SEMUA proses berat (YOLO, MediaPipe, RF, rep
counting) jalan di SERVER (kode Python ini) -- browser/HP cuma jadi CLIENT
(kirim video kamera, terima hasil). Lihat src/app/live_pipeline.py utk
pipeline inti -- TIDAK diubah/di-reimplementasi di sini, cuma dipanggil.

CATATAN THREADING (sumber bug "skeleton tidak muncul" versi sebelumnya):
streamlit-webrtc menjalankan video_frame_callback() di THREAD TERPISAH dari
script utama Streamlit. Akses `st.session_state[...]` LANGSUNG dari dalam
callback itu TIDAK RELIABLE. Solusi: objek yang perlu dibagi (pipeline, hasil
terakhir, log sesi) disimpan sbg variabel Python BIASA (dict/list) yang
di-capture lewat closure SEKALI di luar callback, lalu di dalam callback cuma
MUTATE isinya (bukan lookup ulang `st.session_state[key]`).

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
    /* RESPONSIF hp/tablet -- kontrol utama SELALU 1 kolom (bukan st.columns
    berdampingan) supaya tidak sempit di layar sempit; video & gambar dipaksa
    max-width 100% biar tidak overflow/scroll horizontal di hp. */
    .block-container {padding-top: 1.5rem; padding-left: 1rem; padding-right: 1rem;}
    div[data-testid="stMetricValue"] {font-size: 2.2rem;}
    /* BATASI TINGGI video/gambar ke viewport (bukan cuma lebar kolom) --
    laporan langsung user: "kamera sudah nampilkan seluruh badan, tapi UI
    videonya kegedean utk 1 layar" (karena training/live kita portrait 9:16,
    height:auto ikut lebar kolom bikin tinggi videonya ~1.8x lebar -- gampang
    melebihi tinggi layar, jadi harus scroll & badan bagian bawah kepotong
    dari PANDANGAN, bukan dari data/proses). max-height membatasi tinggi
    tampilan ke viewport, width:auto+object-fit:contain menjaga rasio aslinya
    (tidak gepeng), video/gambar jadi utuh terlihat tanpa scroll. */
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

# YOLO PERIODIK (bukan tiap frame) -- lihat yolo_redetect_every di
# live_pipeline.py. Divalidasi (diskusi proyek 2026-09-11): fps naik ~32%
# (14.19 -> 18.69 di sampel 200 frame), akurasi verdict TIDAK turun di semua
# video ground-truth -- malah 2 video (WA0103, video perempuan) sedikit
# membaik (redeteksi periodik mengurangi jitter bbox). N=10 kelihatan lebih
# cepat lagi (20.36 fps) tapi baru diuji sampel kecil, belum divalidasi
# selengkap N=5 -- makanya dipasang N=5, bukan asal ambil yg tercepat.
YOLO_REDETECT_EVERY = 5

# Kecepatan poll panel kanan (st.fragment run_every, lihat _live_panel) --
# MURNI seberapa sering UI di-refresh, TIDAK ADA LAGI kaitan dengan
# kapan/bagaimana verdict postur dihitung (lihat blok SEG_* di bawah,
# 2026-09-11 -- riwayat sebelumnya sempat mengunci verdict "per detik" lewat
# variabel ini, sudah dibongkar total karena kompresi 1-detik-1-vote terbukti
# merusak verdict -- lihat riwayat/diskusi proyek kalau perlu ditelusuri).
COMMIT_INTERVAL_SEC = 1.0

# === DETEKSI SIKLUS GERAKAN, TERPISAH DARI OnlineRepCounter (2026-09-11) ===
# Permintaan user (diteruskan dari masukan dosen pembimbing): rep counting
# (bab 6.2.3 proposal -- prominence+distance+cosine similarity Eq.4) itu utk
# MENGHITUNG REPETISI, tujuannya beda dari KLASIFIKASI POSTUR -- keduanya
# TIDAK PERLU saling terikat secara implementasi. rep_count (OnlineRepCounter)
# TETAP dipakai APA ADANYA, HANYA utk angka "Repetisi: N" yang dipajang --
# tidak diubah, tidak dihapus.
#
# Klasifikasi sekarang pakai deteksi batas siklusnya SENDIRI (lebih sederhana
# & lebih cepat dari OnlineRepCounter, karena TIDAK butuh presisi anti-salah-
# hitung setinggi itu -- taruhannya beda: 1 siklus meleset dikit dampaknya
# kecil ke penilaian postur, beda dgn salah hitung TOTAL repetisi yg fatal):
#   - lacak ARAH gerak sudut utama (naik/turun) + nilai EKSTREM (puncak/lembah)
#     sejauh arah itu berjalan.
#   - begitu sudut membalik >= SEG_HYSTERESIS_DEG dari ekstrem -> 1 ekstrem
#     "terlewati" (zigzag/swing detector, bukan prominence+cosine-similarity).
#   - SEG_MIN_SEC = debounce waktu -- tolak pembalikan yg KECEPETAN (getaran
#     kecil/noise deteksi), supaya tidak kepecah jadi puluhan siklus palsu.
# 1 SIKLUS = dari 1 puncak (lockout/berdiri) ke puncak berikutnya (1 putaran
# turun-naik penuh). Verdict siklus = _phase_gated_seg_class() di bawah.
#
# DIVALIDASI (diskusi proyek 2026-09-11): dibandingkan OnlineRepCounter (lag
# konfirmasi struktural ~1.1-1.5 detik, diukur offline tanpa lag hardware),
# deteksi ini ~2x lebih cepat (~0.25-0.65 detik) di video deteksi bagus (WA0102,
# WA0101, semua benchpress P3, deadlift armsspread/backround P3) -- verdict
# tetap akurat. Di video deteksi BURUK (occlusion berat, mis. deadlift_correct
# P3 yg bbox cuma kedeteksi 49%) segmennya kepecah lebih banyak dari repetisi
# asli -- keterbatasan yang diterima (user: kondisi pengujian akan dikontrol,
# masalah lingkungan di luar scope), BUKAN dianggap gagal karena tidak pernah
# menuduh SALAH secara serampangan (mayoritas jadi None, bukan tebakan liar).
SEG_HYSTERESIS_DEG = 8.0
SEG_MIN_SEC = 0.5

# --- Phase-gate verdict siklus (LAPISAN PELAPORAN, BUKAN model) ---
# Verdict 1 siklus = majority vote prediksi window MENTAH di siklus itu, TAPI
# hanya window pada FASE KERJA. Window di PHASE_GATE_TOP_FRAC teratas rentang
# sudut utama siklus (fase lockout/berdiri, sendi lurus penuh) dibuang --
# kesalahan bentuk tidak teramati di situ (valgus lutut mustahil dinilai dari
# kaki lurus; punggung deadlift natural netral saat berdiri; siku benchpress
# tidak melebar saat lengan terkunci ke atas). Berlaku SAMA utk squat/
# benchpress/deadlift: di ketiganya lockout = sudut utama MAKSIMUM (divalidasi
# leave-one-participant-out -- akurasi verdict naik di 3 exercise; kontrol
# "buang fase kerja" malah turun -> arah gate benar). Ini TIDAK menyentuh
# model/training/prediksi per-window -- F1 per-window resmi (test split
# 70:30) TETAP sama persis. MIN_SEG_WINDOWS: kalau window fase-kerja yg
# tersisa < ini, verdict = None ("kurang data", jujur, drpd menebak -- user:
# "kalau none yaa gkpp"). Dipasang KECIL (3): cukup menolak siklus yg hampir
# tanpa data, tapi tidak membuang verdict video yg deteksi-nya kurang stabil
# (WA0103/occlusion) -- floor 8 sempat dicoba, bikin 5/6 repetisi backbend
# WA0103 & 3/4 video perempuan jadi None (itu rugi, bukan jujur).
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
        # Durasi "BERSIAP..." bebas diatur pengguna (ada yg butuh waktu lama
        # utk pasang posisi/beban, ada yg cepat) -- proposal TIDAK menetapkan
        # durasi persiapan tertentu (fitur countdown_sec ini murni tambahan
        # implementasi kita, lihat docstring LivePosturePipeline di
        # live_pipeline.py: padanan live dari auto-trim-start label_phase.py),
        # jadi menjadikannya adjustable TIDAK menyimpang dari proposal.
        countdown_sec = st.slider(
            "Durasi persiapan (detik)", min_value=0, max_value=30, value=5, step=1,
            help="Waktu 'BERSIAP...' sebelum repetisi mulai dihitung -- atur sesuai kebutuhan.",
        )
        # Ambang keyakinan klasifikasi (BARU, permintaan user -- gaya slider
        # "confidence threshold" Ko et al di app mereka). Default 0.0 (mati,
        # perilaku lama) -- SENGAJA tidak dipasang angka "terbaik" sepihak,
        # karena diukur langsung ke data kita: keyakinan RF median cuma ~0.45
        # dan efeknya BEDA jauh per exercise (squat/deadlift threshold 0.5
        # menaikkan akurasi window yg lolos ke 0.72-0.99, tapi benchpress di
        # angka sama malah TURUN ke 0.48) -- jadi biar user coba-coba sendiri.
        # Attribute pipeline diupdate LANGSUNG (baris di bawah dekat load
        # pipeline) -- BUKAN via reconstruct LivePosturePipeline, supaya
        # geser slider ini TIDAK memicu reload YOLO/MediaPipe/RF yg mahal.
        confidence_threshold = st.slider(
            "Ambang keyakinan klasifikasi RF (confidence threshold)",
            min_value=0.0, max_value=1.0, value=0.0, step=0.05,
            help="Kelas cuma dipakai kalau RF cukup yakin (fraksi pohon yg setuju >= "
                 "angka ini). 0.0 = mati (perilaku lama, RF selalu pilih suara terbanyak "
                 "walau tipis). Makin tinggi = makin sering '(belum ada window penuh)' "
                 "tapi yg lolos lebih bisa dipercaya -- coba-coba sendiri, efeknya beda "
                 "tiap exercise (sudah diukur, lihat diskusi proyek).",
        )
        # Jalan keluar manual utk kasus mirroring kamera selfie (lihat diskusi
        # proyek). Berlaku ke KEDUA mode:
        # - Kamera Live (WebRTC/getUserMedia): SEHARUSNYA tidak membalik data
        #   mentah, sudah diverifikasi tidak ke-mirror di 1 device (tes angkat
        #   tangan kanan) -- tapi toggle ini jadi koreksi cepat kalau ketemu
        #   device yg ternyata beda.
        # - Upload Video: BEDA KASUS -- kalau video direkam pakai APLIKASI
        #   KAMERA BAWAAN HP dalam mode selfie/depan (bukan live WebRTC),
        #   banyak aplikasi kamera bawaan MEMANG membalik file yg disimpan.
        #   Ini belum pernah diverifikasi (beda jalur dari WebRTC) -- cek
        #   manual per video kalau upload hasil rekaman selfie.
        # Default OFF (tidak flip) krn itu perilaku normal utk sebagian besar kasus.
        flip_camera = st.checkbox(
            "🔄 Flip kamera horizontal",
            value=False,
            help="Nyalakan kalau video (live ATAU upload) terlihat terbalik kiri-kanan "
                 "(tangan kanan tampil sbg tangan kiri, dst) -- terutama utk video hasil "
                 "rekaman selfie dari aplikasi kamera bawaan HP.",
        )
        # BARU 2026-09-11 (bukti nyata, lihat diskusi proyek): frame mentah
        # dari kamera HP tertentu (dicek DEBUG ukuran frame) sampai ke server
        # sudah LANDSCAPE (mis. 480x270) walau HP dipegang TEGAK -- browser/
        # sensor device itu tidak menerapkan auto-rotate ke data yg dikirim
        # (bug WebRTC yg dikenal luas, terutama Android Chrome). TIDAK BISA
        # dideteksi otomatis dari server (cuma terima array piksel, tidak tau
        # device-nya HP/laptop/orientasi fisiknya) -- makanya manual, PERSIS
        # pola flip_camera di atas. Arah yg benar (searah/berlawanan jarum
        # jam) beda2 tergantung device, HARUS dicoba langsung lihat hasilnya.
        # Diterapkan ke KEDUA mode (Live & Upload) spy konsisten.
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
            # DEBUG SEMENTARA (2026-09-11): ukuran frame MENTAH dari kamera HP
            # SEBELUM diproses apa pun -- buat cari tau PASTI apakah data yg
            # sampai ke server sudah kebalik (lebar>tinggi padahal HP dipegang
            # tegak) atau tidak. Dihapus lagi setelah akar masalah ketemu.
            "debug_frame_shape": None,
        }
        st.session_state["session_log"] = []  # [{seg_no, class, warning, image, t}] -- 1 entri per SIKLUS gerakan selesai
        st.session_state["played_log_len"] = 0  # audio: jumlah entri session_log yg sudah diputar
        # Deteksi SIKLUS gerakan, TERPISAH dari OnlineRepCounter -- lihat blok
        # SEG_HYSTERESIS_DEG/SEG_MIN_SEC di atas:
        # - seg_state: state mesin hysteresis (arah gerak sudut + nilai
        #   ekstrem sejauh ini + waktu konfirmasi terakhir, buat debounce).
        # - seg_windows: prediksi window MENTAH + sudut utamanya, sejak batas
        #   siklus terakhir. VERDICT siklus dihitung dari sini (phase-gated)
        #   -- lihat _phase_gated_seg_class().
        # - seg_evidence: frame fase TERDALAM siklus (sudut utama minimum) ->
        #   dipakai jadi snapshot ringkasan, supaya foto benar2 menampilkan
        #   posisi termuat, bukan frame batas-siklus (orang sudah berdiri
        #   tegak = selalu tampak "benar", bikin foto tiap kelas identik).
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
# Update ambang keyakinan LANGSUNG tiap rerun (attribute biasa, BUKAN lewat
# reconstruct pipeline) -- geser slider-nya jadi instan, tidak perlu reload
# YOLO/MediaPipe/RF yg mahal (~<100ms tapi tetap tidak perlu utk 1 angka saja).
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
    """Kelas paling sering muncul dari daftar prediksi mentah (voting) --
    None kalau kosong/semua None. Murni statistik sederhana (Counter bawaan
    Python), BUKAN model/rumus baru -- post-processing tampilan saja. Ini inti
    dari _phase_gated_seg_class() untuk VERDICT 1 SIKLUS gerakan, dijalankan
    pada window fase-kerja saja (phase-gate), majority vote POLOS, gaya Ko et
    al (most_frequent() di Streamlit.py mereka), atas KEPUTUSAN SADAR user
    (bukan default kami).

    RIWAYAT (supaya tidak dikira lupa/oversight): sempat ada kebijakan lebih
    ketat KHUSUS utk ringkasan repetisi (_rep_summary_class, sudah dihapus) --
    "1 window salah di antara N window benar = seluruh repetisi ditandai
    salah", dipicu kejadian nyata (window tengah sempat squat_backbend, tapi
    krn mayoritas correct, ringkasan lama nulis 'correct', kesalahan tidak
    pernah dilaporkan). User tahu risiko itu bisa terulang, dan TETAP memilih
    majority polos -- alasan: exclude-low-confidence (class_confidence_
    threshold di live_pipeline.py) sudah jadi lapisan pertahanan sendiri
    duluan (window yg RF-nya ragu dibuang TOTAL dari voting, tidak ikut
    dihitung sama sekali) -- baru SETELAH itu majority polos dijalankan di
    sisa window yg RF-nya yakin. Kalau RF yakin DAN mayoritas window yakin itu
    salah, itu sah dilaporkan salah apa adanya."""
    non_none = [p for p in preds if p is not None]
    if not non_none:
        return None
    return Counter(non_none).most_common(1)[0][0]


def _phase_gated_seg_class(seg_windows):
    """VERDICT 1 SIKLUS gerakan (bukan 1 "repetisi" dari OnlineRepCounter --
    lihat blok SEG_HYSTERESIS_DEG/SEG_MIN_SEC di atas). seg_windows = daftar
    (raw_class, primary_angle) dari SETIAP window sepanjang siklus (prediksi
    MENTAH per-window, BUKAN commit per-detik).

    Langkah:
    1. Buang window yg raw_class/primary_angle None.
    2. Kalau sisa window < MIN_SEG_WINDOWS -> return None ("kurang data").
    3. cut = min_sudut + (max_sudut - min_sudut) * (1 - PHASE_GATE_TOP_FRAC).
       Buang window dgn primary_angle > cut (fase lockout/berdiri -- sendi
       lurus penuh, tidak ada kesalahan bentuk yg teramati). Lockout = sudut
       utama MAKS di squat (lutut), benchpress (siku), deadlift (pinggul).
    4. Kalau window fase-kerja tersisa < MIN_SEG_WINDOWS -> None.
    5. return _majority_class(window fase-kerja).

    LAPISAN PELAPORAN -- tidak menyentuh model/training/prediksi per-window.
    F1 per-window resmi (test split) TIDAK berubah."""
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
    """Mesin hysteresis 1 langkah -- lacak arah gerak sudut utama (naik/turun)
    + nilai ekstrem sejauh arah itu berjalan. Return True TEPAT SEKALI saat 1
    PUNCAK (fase lockout/berdiri) terlewati & terkonfirmasi (bukan noise) --
    itu artinya 1 SIKLUS (turun-naik) baru saja selesai. Lembah (titik
    terdalam) juga dilacak (perlu tau kapan balik arah), tapi TIDAK memicu
    True -- 1 siklus = puncak ke puncak, bukan puncak ke lembah.

    seg_state (dict, di-mutate in-place): direction ("up"/"down"/None),
    extreme_val (nilai sudut ekstrem terkini), last_confirm_t (waktu
    konfirmasi ekstrem terakhir -- buat debounce SEG_MIN_SEC).

    Debounce: pembalikan arah yg lebih cepat dari SEG_MIN_SEC sejak konfirmasi
    terakhir dianggap NOISE/getaran deteksi kecil, diabaikan (arah tetap
    dibalik biar tracking lanjut, tapi TIDAK dihitung sbg batas siklus).

    LAPISAN PELAPORAN -- terpisah total dari OnlineRepCounter (lihat blok
    komentar SEG_* di atas), tidak menyentuh model/prediksi per-window."""
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

    # Titik skeleton dikonversi ke KOORDINAT PIKSEL FRAME UTUH (offset dari
    # crop_box) SEKALI di sini -- supaya draw_overlay() tinggal gambar
    # garis/titik ke frame APAPUN (termasuk frame kamera yg lebih baru dari
    # yg dipakai proses berat ini, mode async) tanpa perlu tau crop_box lagi.
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

        # --- Buffer VERDICT SIKLUS (phase-gated, LAPISAN PELAPORAN) ---
        # Kumpulkan prediksi window MENTAH + sudut utamanya sejak batas siklus
        # terakhir (BUKAN sejak rep_count terakhir -- lihat blok SEG_* di atas).
        # CATATAN: TIDAK dikosongkan saat bbox kosong (YOLO miss 1-2 frame itu
        # sering/deteksi kedip) -- window bbox-kosong toh otomatis (None, None)
        # -> disaring _phase_gated_seg_class. Kalau tiap miss menghapus buffer,
        # video yg deteksinya kurang stabil (mis. WA0103) verdict-nya jadi None
        # terus -- lebih rugi drpd cuma dibiarkan disaring wajar.
        seg_windows.append((raw_class, a))
        # Frame BUKTI utk snapshot ringkasan = frame fase PALING DALAM (sudut
        # utama minimum) sejauh siklus ini -- dijamin fase kerja & memang
        # menampilkan posisi termuat, BUKAN frame batas-siklus (orang sudah
        # berdiri tegak = selalu tampak "benar", dulu bikin foto tiap kelas
        # identik). 1 salinan frame, dibersihkan tiap batas siklus.
        if a is not None and (seg_evidence["angle"] is None or a < seg_evidence["angle"]):
            seg_evidence.update(angle=a, img=img.copy(),
                                skeleton_points_px=skeleton_points_px, n_points=n_points,
                                crop_box=result["crop_box"], bbox=result["bbox"])

        # Deteksi batas siklus (hysteresis+debounce, TERPISAH dari
        # OnlineRepCounter -- lihat _update_seg_state & blok komentar SEG_*).
        seg_done = _update_seg_state(seg_state, a, t)

        if seg_done:
            # VERDICT 1 SIKLUS = majority vote prediksi window MENTAH di
            # siklus td, HANYA window fase kerja (phase-gate). Divalidasi 4
            # video ground-truth + leave-one-participant-out 3 exercise. None
            # kalau window fase-kerja < MIN_SEG_WINDOWS ("kurang data", jujur
            # -- user: "kalau none yaa gkpp").
            seg_class = _phase_gated_seg_class(seg_windows)
            seg_windows.clear()
            seg_warning = get_warning(seg_class)
            # Snapshot dari FRAME BUKTI (fase terdalam siklus) kalau ada --
            # kalau tidak (sudut utama NaN sepanjang siklus), fallback ke
            # frame batas-siklus ini.
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
            # Ringkasan sesi (bab ruang lingkup #15e proposal: "gambar saat
            # kesalahan terdeteksi", DIPERLUAS atas permintaan -- juga rekam
            # siklus yg BENAR, bukan cuma yg salah). 1 entri PER SIKLUS
            # gerakan selesai (angka INI beda dari "Repetisi: N" resmi, bisa
            # lebih banyak/sedikit -- lihat blok komentar SEG_* di atas).
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
    """Bagian RINGAN (gambar bbox/skeleton/teks) -- TIDAK memanggil pipeline
    sama sekali (murni cv2 drawing calls), aman dipanggil SESERING MUNGKIN
    (tiap frame kamera, di thread callback webrtc) walau `computed` berasal
    dari frame yg sedikit lebih lama (mode Kamera Live async) -- skeleton/teks
    bisa tertinggal sepersekian detik di belakang gerakan asli, TAPI video
    latar belakangnya sendiri tetap frame TERBARU/mulus (beda dari sebelumnya
    yg videonya ikut lambat)."""
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
    # ALASANNYA nyata, bukan sekadar selera: teks di video itu di-render dari
    # frame yg BERBEDA WAKTU dgn panel kanan (video = frame terbaru hasil
    # worker async; panel = state saat rerun Streamlit terakhir), jadi
    # keduanya BISA MENAMPILKAN KELAS BERBEDA di saat yg sama -- persis yg
    # dikeluhkan ("yang di sini kanan itu beda-beda, jadi tidak konsisten").
    # Sekarang SATU-SATUNYA sumber tampilan angka/kelas = panel kanan
    # (render_result()), jadi tidak mungkin bertentangan dgn dirinya sendiri.
    # Video tinggal menampilkan bbox + skeleton saja -- gaya Ko et al, yg
    # videonya juga bersih tanpa teks (semua info di panel samping).
    return canvas


class LiveWorker:
    """Thread ASYNC khusus mode Kamera Live -- jalankan compute_result()
    (proses berat) di LATAR BELAKANG, terus-menerus mengolah frame TERBARU
    yg di-submit() (frame lama yg belum sempat diproses langsung DIBUANG,
    TIDAK diantre -- supaya tidak ada penumpukan/lag yg makin membesar).
    video_frame_callback() (thread webrtc, TERPISAH lagi dari thread ini)
    tinggal submit() frame masuk & baca latest() kapan saja tanpa nunggu --
    video yg ditampilkan selalu frame kamera TERBARU, overlay-nya yg
    menyesuaikan kecepatan proses ASLI. Murni pengaturan thread di sisi
    SERVER, tidak mengubah apa pun yg diproses (masih YOLO+MediaPipe+RF+rep
    counting yg SAMA, lewat compute_result() yg SAMA)."""

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


# Dibuat SEKALI per pipeline/exercise (guard sama seperti blok pipeline di
# atas) -- diletakkan di sini (bukan di blok itu langsung) krn LiveWorker
# butuh compute_result() sudah terdefinisi. Mode Upload Video TIDAK memakai
# ini sama sekali (tetap sinkron, lihat loop di bawah).
if st.session_state.get("live_worker") is None:
    st.session_state["live_worker"] = LiveWorker()
live_worker = st.session_state["live_worker"]


_ROTATE_CODES = {
    "90° searah jarum jam": cv2.ROTATE_90_CLOCKWISE,
    "90° berlawanan jarum jam": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180°": cv2.ROTATE_180,
}


def _apply_rotation(img, choice):
    """Putar frame MENTAH sebelum diproses apa pun -- lihat komentar toggle
    `rotate_camera` di atas (bukti nyata 2026-09-11: kamera HP tertentu kirim
    data landscape walau dipegang tegak, browser tidak auto-rotate). Manual
    (bukan auto-detect) krn server cuma terima array piksel, tidak tau device/
    orientasi fisiknya. Dipakai di KEDUA mode (Live & Upload) spy konsisten."""
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
    """THREAD TERPISAH lagi dari LiveWorker (streamlit-webrtc jalankan ini di
    thread callback-nya sendiri) -- FUNGSI INI SENGAJA RINGAN: submit() frame
    ke worker (non-blocking, langsung lanjut), TIDAK memanggil
    compute_result()/pipeline sama sekali di sini -- itu sebabnya video tetap
    mulus walau proses beratnya lambat (lihat LiveWorker/compute_result).

    PERCOBAAN 2026-09-11 (permintaan user): sempat dicoba TANPA gambar
    skeleton/kotak di frame yg dikirim balik (biar lebih ringan lewat
    jaringan, lihat riwayat diskusi proyek) -- efeknya kecil (bukan biaya
    utama, YOLO+MediaPipe tetap dominan) DAN user kehilangan cara visual buat
    cek badan/posisi terdeteksi dgn benar -- DIKEMBALIKAN atas permintaan
    user ("mending pake skeleton") -- skeleton+kotak digambar lagi via
    draw_overlay(), TAPI hasil gambarnya TETAP dikecilkan ke LIVE_RETURN_MAX_W
    sebelum di-encode WebRTC (bagian ini TETAP dipertahankan, tidak ada
    kerugian menyimpannya). compute_result() (YOLO+MediaPipe+RF, biaya UTAMA)
    TETAP jalan penuh di LiveWorker, TIDAK disentuh sama sekali."""
    img = frame.to_ndarray(format="bgr24")
    img = _apply_rotation(img, rotate_camera)
    result_box["debug_frame_shape"] = img.shape  # DEBUG SEMENTARA -- ukuran SETELAH rotasi manual (kalau dipilih)
    # CATATAN mirroring kamera selfie (bab 6.4 proposal, kamera depan dipakai
    # supaya user bisa lihat layarnya sendiri): standar web (getUserMedia)
    # TIDAK membalik data mentah kamera (efek mirror yg biasa terlihat itu cuma
    # CSS tampilan, bukan data asli -- app ini tidak menerapkan CSS itu), dan
    # SUDAH DIVERIFIKASI user via tes angkat tangan kanan di 1 device (frame
    # yg diterima server konsisten, TIDAK ke-mirror).
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

    # Digabung jadi 1 kartu (dulu 4 kotak terpisah bertumpuk -- terkesan
    # "teknis"/berantakan) -- repetisi & sudut berdampingan (ringkas), lalu
    # status jadi 1 alur baca yg jelas.
    #
    # PANEL INI TIDAK LAGI menebak kelas per-detik (versi lama gampang flicker
    # & bisa menuduh SALAH sebelum siklus gerakan itu benar2 selesai). Kelas
    # cuma tampil SETELAH 1 siklus (turun-naik) selesai dikonfirmasi (lihat
    # _update_seg_state/_phase_gated_seg_class di compute_result) -- dan
    # sumbernya SATU-SATUNYA session_log, PERSIS yg dipakai render_summary()
    # di bawah. Jadi panel & Ringkasan Sesi TIDAK MUNGKIN saling bertentangan
    # (dulu keluhan user: "yang di sini kanan itu beda-beda, tidak konsisten").
    # "Repetisi" (metrik resmi, OnlineRepCounter) bisa BEDA ANGKA dari nomor
    # "Siklus" di bawah (deteksi sendiri, lebih cepat -- lihat blok komentar
    # SEG_* di atas compute_result) -- itu memang disengaja, 2 sinyal berbeda.
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
        # DEBUG SEMENTARA (2026-09-11, lihat komentar init result_box) --
        # HAPUS setelah akar masalah orientasi kamera HP ketemu pasti.
        shp = result_box.get("debug_frame_shape")
        if shp is not None:
            st.caption(f"🔧 DEBUG: ukuran frame mentah dari kamera = {shp[1]}x{shp[0]} (lebar x tinggi)")


def render_audio():
    """Putar audio peringatan (bab 8.3.3 proposal, gaya Ko et al: file .mp3
    pre-recorded, lihat warnings_content.py + scripts/generate_warning_audio.py).
    session_log sekarang berisi SEMUA repetisi (benar maupun salah, 1 entri per
    repetisi selesai) -- audio cuma benar2 dimainkan kalau entrinya punya
    warning (get_warning_audio_path return None utk repetisi benar, jadi tidak
    ada suara utk itu, cuma tetap dicatat "sudah dilihat" biar tidak diputar
    ulang belakangan). Dipanggil dari THREAD UTAMA saja (sama spt render_result),
    session_log-nya sendiri boleh diisi dari thread callback webrtc, tapi
    tracking "sudah diputar sampai mana" (played_log_len) pakai st.session_state
    yg cuma disentuh di sini (thread utama) -- aman."""
    played = st.session_state.get("played_log_len", 0)
    if len(session_log) > played:
        entry = session_log[-1]
        audio_path = get_warning_audio_path(entry["class"])
        # JEDA MINIMAL antar audio (dilaporkan user: "audio pun putus-putus").
        # Sebabnya: kalau repetisi selesai beruntun cepat, elemen <audio> lama
        # KE-TIMPA audio baru sebelum sempat selesai diputar -> kedengaran
        # kepotong. Ko et al. punya masalah & solusi yg sama persis di
        # Streamlit.py mereka: `if current_time - previous_alert_time >= 3`
        # (jeda minimal 3 detik antar peringatan) -- pola itu yg diikuti di
        # sini, jadi bukan mekanisme karangan sendiri.
        now = time.time()
        last_played_at = st.session_state.get("last_audio_at", 0.0)
        if audio_path is not None and (now - last_played_at) < 3.0:
            audio_path = None  # lewati pemutaran, tapi entry TETAP ditandai sudah dilihat
        if audio_path is not None:
            st.session_state["last_audio_at"] = now
            # PENTING soal duplicate-element error: st.audio() di versi
            # streamlit yg dipasang (1.63.0) TIDAK punya parameter `key`
            # (dicek langsung via inspect.signature) -- jadi kalau kesalahan
            # yg sama sempat hilang lalu muncul lagi (mis. correct -> flatback
            # -> correct -> flatback), audio_path & argumennya identik dan
            # Streamlit menolaknya (StreamlitDuplicateElementId). Solusinya:
            # tempel langsung tag HTML5 <audio autoplay> (data URI base64) via
            # st.markdown, dengan komentar HTML tersembunyi berisi nomor urut
            # entry -- SUPAYA ISI STRING-nya selalu unik tiap kali kesalahan
            # baru tercatat, tanpa bergantung pada parameter `key` yg tidak
            # tersedia. Komentar ini tidak mempengaruhi audio-nya sama sekali.
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

    # BUG NYATA yg ditemukan (laporan user, lihat diskusi proyek 2026-09-11):
    # siklus dgn class=None ("kelas belum terkonfirmasi" -- window fase kerja
    # < MIN_SEG_WINDOWS, BUKAN kesalahan) ikut masuk grup galeri, dan karena
    # entry["warning"] juga None utk kelas None, kartunya salah kena cabang
    # "else" di bawah -> tampil "✅ Postur benar" -- PADAHAL belum pernah
    # dinilai sama sekali. Fix: pisah dulu confirmed (class bukan None) dari
    # unconfirmed SEBELUM dihitung/di-gallery-kan -- None TIDAK ditampilkan
    # sbg kartu apa pun (user: "kalau mmg None yaa di ringkasan gk perlu
    # muncul"), cukup disebut jumlahnya secara terpisah (transparan, bukan
    # disembunyikan diam-diam).
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
    """Panel kanan (repetisi/prediksi/peringatan/audio/ringkasan) sebagai
    FRAGMEN independen (st.fragment, fitur Streamlit >=1.33 -- dicek tersedia
    di versi yg dipasang, 1.63.0) -- BUKAN st.rerun() penuh + time.sleep()
    lagi (versi lama).

    KENAPA diganti (laporan langsung user: video masih "patah-patah"/tidak
    semulus kamera aslinya): st.rerun() itu me-render ULANG SELURUH HALAMAN,
    termasuk elemen webrtc_streamer() -- walau video-nya sendiri jalan lewat
    koneksi WebRTC browser (harusnya tidak ikut lambat), proses reconcile
    ulang SELURUH pohon elemen Streamlit tiap request tetap bisa mengganggu
    tampilan di sekitarnya. Dokumentasi st.fragment eksplisit: "the rest of
    the app is persisted during a fragment rerun" -- jadi kalau
    webrtc_streamer() dipanggil DI LUAR fragmen ini (lihat di bawah), dia
    TIDAK PERNAH ikut di-render ulang oleh refresh panel ini sama sekali."""
    render_result()
    render_audio()
    # Ringkasan/riwayat DITAMPILKAN TERUS-MENERUS selama sesi berjalan (bukan
    # cuma setelah tombol STOP) -- gaya Ko et al (Streamlit.py mereka: tiap
    # kesalahan baru langsung nempel ke halaman via st.error() berturut-turut,
    # jadi riwayat yg terus tumbuh terlihat SELAMA latihan).
    render_summary()


if source_mode == "Kamera Live":
    with col_video:
        st.subheader("📹 Kamera")
        webrtc_ctx = webrtc_streamer(
            key=f"posture-{exercise}",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            # GANTI 2026-09-11 (laporan user: video Live di HP kepaksa
            # LANDSCAPE walau dipegang tegak): dugaan PALING didukung fakta
            # WebRTC -- constraint width/height ANGKA EKSAK (versi lama:
            # 1080x1920) bisa bikin sebagian browser HP (terutama Android
            # Chrome) pilih MODE SENSOR MENTAH yg paling cocok ke angka itu
            # dan MELEWATI auto-rotate berbasis orientasi device -- video
            # mentah (landscape, sensor fisik HP memang landscape) lolos apa
            # adanya. Fix: pakai `aspectRatio` (rasio, BUKAN angka piksel
            # mutlak) -- tidak memaksa mode sensor tertentu, jadi tidak
            # mengganggu jalur auto-rotate normal browser. 9/16 = rasio
            # portrait video training kita (data/raw_videos/* rata2
            # 1080x1920). BELUM 100% dipastikan ini akar masalah PASTINYA --
            # tidak ada akses kamera HP fisik utk tes langsung -- WAJIB
            # dikonfirmasi user di HP asli sebelum dianggap solved.
            # facingMode="user" -- kamera DEPAN/selfie (skenario deploy asli,
            # bab 6.4 proposal); sudah diverifikasi user (tes angkat tangan
            # kanan) TIDAK mirrored, zoom lensa depan/belakang TERBUKTI SAMA.
            media_stream_constraints={
                # width/height "ideal" (BUKAN exact/min/max) -- PERCOBAAN
                # SEMENTARA 2026-09-12, BELUM final, buat diagnosis bug
                # "zoom" HP: debug_frame_shape kepergok cuma 320x180 (kecil,
                # 16:9 -- KEBALIKAN dari aspectRatio 9/16 yg diminta),
                # padahal aspectRatio-nya sendiri sudah ideal 9/16 -- dugaan:
                # tanpa petunjuk width/height, browser bebas pilih resolusi
                # default sendiri (kecil), aspectRatio doang tidak cukup
                # kuat. "ideal" dipilih (bukan exact spt versi lama 1080x1920
                # yg TERBUKTI bikin bug lain -- forced landscape sensor mode)
                # supaya cuma jadi SARAN, tidak maksa mode sensor tertentu.
                # HARUS dicek langsung: apakah debug_frame_shape berubah
                # jadi lebih besar/portrait di HP asli -- kalau tidak,
                # hipotesis ini SALAH, buang lagi baris ini.
                "video": {"aspectRatio": {"ideal": 9 / 16},
                          "facingMode": {"ideal": "user"},
                          "width": {"ideal": 720},
                          "height": {"ideal": 1280}},
                "audio": False,
            },
            # GANTI 2026-09-12 (laporan user: video Live di LAPTOP kelihatan
            # "zoom"/terpotong di tampilan normal, tapi balik NORMAL begitu
            # ditekan tombol fullscreen bawaan browser). Root cause: BEDA
            # LAYER dari bug orientasi HP di atas -- itu soal ISI/rotasi
            # piksel, ini soal CSS TAMPILAN elemen <video>. Default
            # streamlit-webrtc (VideoHTMLAttributes bawaan library, lihat
            # DEFAULT_VIDEO_HTML_ATTRS di config.py) cuma set
            # style={"width": "100%"} -- TANPA height/objectFit eksplisit.
            # CSS global kita (blok <style> di atas, target `img, video`)
            # TIDAK TEMBUS ke sini krn streamlit-webrtc dirender di iframe
            # custom component terpisah, bukan DOM halaman utama. Akibatnya
            # ukuran video ditentukan style bawaan/layout komponennya sendiri
            # (bisa memotong rasio 9:16 kita ke kotak yg tidak sesuai) --
            # fullscreen native browser melewati kotak itu & render ulang
            # sesuai rasio asli video, makanya kelihatan "normal" lagi (bukti
            # konten video-nya sendiri SUDAH benar, murni soal tampilan).
            # Fix: paksa objectFit "contain" (jaga rasio asli, tidak
            # memotong) + height "auto" (ikut lebar, bukan kotak tetap).
            video_html_attrs=VideoHTMLAttributes(
                autoPlay=True, controls=True,
                style={"width": "100%", "height": "auto", "objectFit": "contain"},
            ),
        )

    # DI LUAR fragmen di atas SENGAJA -- ini kuncinya (lihat docstring
    # _live_panel). webrtc_streamer() cuma dipanggil SEKALI per full script
    # run (start/stop tombol, ganti exercise, dst), TIDAK ikut ter-panggil
    # ulang tiap _live_panel() refresh sendiri tiap COMMIT_INTERVAL_SEC.
    _live_panel()

else:  # Upload Video
    # PENTING (soal "upload lagi error/lambat"): pemrosesan 1 video (loop
    # while di bawah, YOLO+MediaPipe+RF PER FRAME) jalan SINKRON di script
    # thread utama Streamlit -- selama loop itu jalan, script TIDAK bisa
    # merespons interaksi lain (termasuk upload file BARU) sampai loop
    # selesai. Kalau pengguna coba upload/klik lagi SELAGI video sebelumnya
    # masih diproses, request upload browser bisa timeout/dibatalkan
    # ("CanceledError: canceled"). Dicek langsung (bukan tebakan): membangun
    # ulang LivePosturePipeline() itu SENDIRI TERNYATA CEPAT (<100ms) selama
    # model sudah pernah dimuat sekali di proses server ini -- jadi bukan itu
    # penyebabnya. Fix: kunci uploader+tombol selama status "analyzing" aktif,
    # supaya tidak ada upload baru yg nyelonong di tengah proses lama.
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
            # Reset juga state deteksi siklus -- kalau tidak, sisa buffer dari
            # video SEBELUMNYA ikut "bocor" ke detik-detik pertama video BARU ini.
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
                computed = compute_result(frame, t)  # proses pakai frame RESOLUSI ASLI -- akurasi tidak tersentuh
                canvas = draw_overlay(frame, computed)
                # Preview di-update TIAP frame yg diproses (bukan tiap 3 frame
                # spt versi lama) -- laporan user 2026-09-11: preview terasa
                # patah-patah pas dites LOKAL (localhost, bandwidth bukan
                # masalah di sini). Skip 1-dari-3-frame versi lama itu justru
                # BIKIN preview cuma jalan di ~1/3 kecepatan proses asli
                # (~18-21fps jadi terasa ~6-7fps) -- lebih patah drpd perlu.
                # compute_result() TETAP jalan sinkron di frame ASLI di atas
                # (akurasi tidak tersentuh) -- ini cuma soal SESERING APA
                # hasilnya ditampilkan.
                #
                # PREVIEW SAJA dikecilkan ke 480px (murni tampilan) -- laporan
                # user sebelumnya: preview "freeze" saat diakses lewat
                # Cloudflare Tunnel (bukan localhost), krn gambar resolusi
                # penuh (mis. 1080x1920) dikirim ulang lewat websocket
                # Streamlit -- upload bandwidth laptop ke tunnel biasanya jauh
                # lebih kecil dari download (umum di internet rumahan).
                # Dikecilkan SEKALIGUS diupdate tiap frame -- dua perbaikan
                # ini saling melengkapi (ringan DAN sering), bukan konflik.
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
            # WAJIB finally -- kalau video-nya rusak/error di tengah loop,
            # status "analyzing" tetap harus lepas, supaya uploader tidak
            # terkunci selamanya gara-gara 1 file bermasalah.
            st.session_state["analyzing"] = False

    render_result()
    render_audio()
    if session_log:
        render_summary()

st.caption("Kelas postur dipajang setelah 1 siklus gerakan (turun-naik) selesai "
            "dikonfirmasi -- bukan tiap detik -- supaya tidak menuduh salah sebelum "
            "siklusnya benar-benar tuntas.")
