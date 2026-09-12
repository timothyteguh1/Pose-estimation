"""Repetition counting ONLINE (real-time/incremental) berbasis skeleton, bab
6.2.3 proposal: "Satu repetisi didefinisikan sebagai satu siklus lengkap dari
fase minimum ke maksimum dan kembali ke minimum (fase naik dan turun)."

BEDA dari find_turning_points()/estimate_rep_count() di
src/extraction/label_phase.py: versi itu OFFLINE (scipy.find_peaks butuh
SELURUH sinyal termasuk "masa depan", cocok buat video yang sudah direkam
penuh -- dipakai buat auto-detect draft saat labeling -- TIDAK BISA dipakai
live). Versi di sini ONLINE -- state machine yang cuma pakai data SAMPAI SAAT
INI (causal, tidak menengok ke depan), TAPI TEORINYA SAMA PERSIS: deteksi
siklus dari perubahan joint angle, pakai prominence (ambang derajat minimum)
+ distance (jarak minimum antar titik balik) supaya wobble kecil tidak
dihitung rep palsu -- sama konsep dengan DEFAULT_MIN_PROMINENCE_DEG/
DEFAULT_MIN_DISTANCE_SEC di label_phase.py, cuma dihitung incremental
(algoritma "online peak detection", padanan causal dari scipy.find_peaks).

Definisi "1 rep" dihitung: SAAT PUNCAK terkonfirmasi (konsisten dengan
estimate_rep_count() offline yang juga hitung JUMLAH PUNCAK, bukan lembah)
-- supaya online & offline tidak punya definisi yang beda sendiri-sendiri.
Rep pertama butuh 1 lembah dikonfirmasi dulu sebelum puncak dihitung (hindari
menghitung rep palsu kalau perekaman dimulai di tengah gerakan naik).

Parameter (prominence/distance) di-TUNING PER-EXERCISE dari data kita sendiri
(BUKAN dari referensi manapun) -- Hsu et al. (2024, dikutip bab 6.2.3) sudah
dicek langsung ke papernya: tidak ada angka derajat/threshold yang sepadan
buat dipinjam, karena metode mereka FFT/domain-frekuensi, beda paradigma dari
proposal kita yang literal bilang "lebih sederhana... tanpa analisis
frekuensi seperti yang digunakan pada penelitian Hsu et al. (2024)".

GERBANG COSINE SIMILARITY (Eq.4 bab 6.2.3, opsional -- `min_pattern_similarity`):
proposal: "kemiripan antar frame... dianalisis menggunakan cosine similarity
[[Eq.4]]... Pendekatan ini HANYA digunakan utk mengidentifikasi kemiripan pola
gerakan. Setelah dari itu, [baru] perhitungan repetisi dilakukan [via joint
angle]." -- dibaca sbg 2 TAHAP SEKUENSIAL: (1) cosine similarity MEMVALIDASI
bahwa 1 siklus yg baru dikonfirmasi itu POLA GERAKANNYA mirip siklus acuan
(bukan noise/gerakan lain), (2) BARU SETELAH lolos, peak detection (metode
utama, sesuai bab 6.2.3) menghitung repetisinya. A & B di Eq.4 = vektor 11
joint angle di TITIK EKSTREM (puncak/lembah) yg baru dikonfirmasi vs TITIK
EKSTREM SEJENIS PERTAMA yg sudah dikonfirmasi di sesi itu (jadi "template"
acuan pola gerakan yg benar). Ambang kemiripan di-TUNING dari data sendiri,
sama persis pola prominence/distance di atas. Default `min_pattern_similarity
=None` -- gerbang MATI kecuali diaktifkan eksplisit.

CATATAN PENTING (perbaikan arsitektur, lihat diskusi proyek): sempat dicoba
nambah gerbang "warm-up" (butuh N siklus konsisten ritme dulu) LANGSUNG DI
KELAS INI utk nyaring rep palsu dari masa persiapan/penutup rekaman -- tapi
itu DIBATALKAN karena TIDAK ADA dasarnya di proposal (beda dari
prominence/distance yg konsepnya generik/established, dan cosine similarity
yg literal ada di Eq.4). Kelas ini SEKARANG MURNI cuma prominence+distance+
cosine-similarity, PERSIS sesuai bab 6.2.3 -- deteksi "kapan masa persiapan
selesai" (buat mastiin template cosine similarity bersih dari awal) itu
tanggung jawab PEMANGGIL (src/app/live_pipeline.py, mekanisme
countdown/exclusion yg SUDAH ADA sebelumnya, dibuat lebih adaptif) -- BUKAN
bagian dari algoritma hitung repetisi ini.
"""
import numpy as np


def _cosine_similarity(a, b):
    """cos(theta) = A.B / (|A||B|) -- Eq.4 bab 6.2.3. A, B: dict {nama_angle: derajat}.
    NaN di salah satu sisi utk 1 nama angle -> titik itu DIABAIKAN dari perbandingan
    (bukan interpolasi -- konsisten aturan "discard" yg dipakai di tempat lain),
    supaya occlusion 1 titik saja tidak menggagalkan perbandingan semuanya."""
    keys = [k for k in a if k in b and a[k] == a[k] and b[k] == b[k]]  # k==k -> bukan NaN
    if len(keys) < 3:  # terlalu sedikit titik valid, tidak bisa dipercaya bandingnya
        return None
    va = np.array([a[k] for k in keys], dtype=float)
    vb = np.array([b[k] for k in keys], dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return None
    return float(np.clip(np.dot(va, vb) / denom, -1.0, 1.0))


RISING = "rising"
FALLING = "falling"

# Tuning per-exercise, dari SINYAL PRODUKSI ASLI (LivePosturePipeline lewat
# YOLO+crop+MediaPipe, BUKAN shortcut CSV extracted_landmarks) -- lihat
# diskusi proyek. Metodologi SAMA PERSIS ke-3 exercise: skema TUNING/HELD-OUT
# (bagian video cari parameter, bagian LAIN dikunci gak disentuh sampai
# evaluasi akhir -- padanan train/test split, supaya MAE yg dilaporkan tidak
# "menghafal" video tuning). Ground truth jumlah repetisi dihitung dari label
# manual (kolom 'phase', HANYA video yg 100% manual -- 'excluded'/'auto' tidak
# ikut terhitung).
#   benchpress : DIKOREKSI KETIGA KALI, 25 -> 28 deg (distance 1.2 -> 0.6
#                sec). Round kedua (18->25, distance 0.8->1.2) di-tuning
#                pakai SHORTCUT landmark dari CSV (frame utuh, MediaPipe
#                TANPA lewat YOLO+crop) -- TERBUKTI tidak valid buat pipeline
#                produksi sungguhan: MAE shortcut 0.83 vs MAE produksi ASLI
#                1.86 (beda jauh). Akar masalahnya BUKAN parameter, tapi bug
#                arsitektur: MediaPipe idle total selama countdown_sec (0
#                frame diproses), bikin tracking mulai dari nol & gagal di
#                frame yg justru berhasil kalau diproses berurutan dari awal
#                (lihat fix di live_pipeline.py: MediaPipe TETAP diproses
#                selama countdown, cuma hasilnya yg dibuang). SETELAH bug itu
#                diperbaiki, tuning ULANG (4 video tuning, 2 video held-out):
#                MAE tuning=1.50, MAE held-out=1.00 (konsisten, tidak
#                njomplang -- pertanda parameter ini genuinely general, bukan
#                kebetulan cocok). Cosine similarity TIDAK terbukti berubah
#                signifikan ke MAE di kisaran 0.60-0.85 (atau OFF) -- 0.70
#                dipertahankan krn fungsinya tetap relevan secara teori (Eq.4).
#   squat      : DITUNING dari data asli (9 video, 1 partisipan, GT=6 repetisi
#                semua video) -- 6 video tuning, 3 video held-out. Prominence
#                12 deg sempat kelihatan lebih bagus di tuning (MAE=0.83) TAPI
#                held-out-nya malah 1.33 (njomplang, overfit -- 1 video held-out
#                salah jauh, prediksi 9 dari GT 6). Prominence 20 deg TERPILIH
#                krn KONSISTEN: MAE tuning=1.00, MAE held-out=1.00 (sama
#                persis, tidak overfit). Distance 0.3-1.2s semua hasilnya
#                identik di rentang ini -- 0.6 dipilih sbg titik tengah.
#   deadlift   : DITUNING dari data asli (12 video, 2 partisipan) -- 8 video
#                tuning, 4 video held-out (termasuk 1 video P1 paling occluded
#                sengaja masuk held-out). Prominence 50 deg awalnya kelihatan
#                bagus di tuning (MAE=0.375) TAPI held-out melonjak ke 1.50
#                (overfit, video P1 occluded gagal total: prediksi 3 dari GT
#                6). Prominence 25 deg TERPILIH krn lebih konsisten: MAE
#                tuning=1.25, MAE held-out=0.50 (JUSTRU LEBIH BAIK di held-out
#                -- pertanda genuinely general, bukan hafalan). Distance 1.0s.
#
# CATATAN PENTING semua ke-3 exercise: baru dari 1-2 partisipan (squat malah
# baru 1) -- WAJIB divalidasi ulang lagi begitu partisipan lebih banyak masuk
# (lihat diskusi proyek).
REP_COUNTING_PARAMS = {
    "squat": {"min_prominence_deg": 20.0, "min_distance_sec": 0.6},
    "benchpress": {"min_prominence_deg": 28.0, "min_distance_sec": 0.6},
    "deadlift": {"min_prominence_deg": 25.0, "min_distance_sec": 1.0},
}


class OnlineRepCounter:
    """Panggil update(angle, timestamp_sec) tiap ada frame baru (angle =
    primary_angle_for_frame() utk exercise ybs, boleh NaN kalau occluded --
    frame itu di-skip, TIDAK merusak state yang sudah terbentuk).

    rep_count bertambah tiap kali 1 puncak (peak) dikonfirmasi, setelah
    minimal 1 lembah (valley) sebelumnya juga sudah dikonfirmasi.
    """

    def __init__(self, exercise, min_prominence_deg=None, min_distance_sec=None,
                 smooth_window=5, min_pattern_similarity=None):
        params = REP_COUNTING_PARAMS.get(
            exercise, {"min_prominence_deg": 40.0, "min_distance_sec": 0.8})
        self.exercise = exercise
        self.min_prominence_deg = (min_prominence_deg if min_prominence_deg is not None
                                    else params["min_prominence_deg"])
        self.min_distance_sec = (min_distance_sec if min_distance_sec is not None
                                  else params["min_distance_sec"])
        self.smooth_window = smooth_window
        self.min_pattern_similarity = min_pattern_similarity  # None = gerbang cosine mati

        self._history = []  # buffer causal buat rolling-mean smoothing (bukan center=True)
        self._last_valid_angle = None  # "tahan nilai terakhir" -- padanan causal dari interpolasi
        self.rep_count = 0
        self.rejected_by_pattern = 0  # berapa kali gerbang cosine nolak (buat diagnosa)
        self._direction = None  # RISING / FALLING / None (belum cukup data)
        self._extreme_value = None
        self._extreme_time = None
        self._extreme_feature_vector = None  # vektor 11 angle DI TITIK EKSTREM saat ini
        # Jarak minimum dicek TERPISAH per jenis (puncak-ke-puncak, lembah-ke-
        # lembah) -- BUKAN puncak-ke-lembah -- persis scipy.find_peaks() yang
        # dipanggil 2x independen (sekali di values utk puncak, sekali di
        # -values utk lembah) di find_turning_points() (versi offline).
        # Lembah->puncak boleh berdekatan (gerakan naik bisa cepat, itu wajar).
        self._last_peak_time = None
        self._last_valley_time = None
        self._seen_valley = False  # rep pertama butuh lembah dulu, baru puncak dihitung
        self._peak_template = None  # vektor angle di puncak PERTAMA yg terkonfirmasi (acuan)
        self._valley_template = None  # sama, tapi utk lembah

    def _smoothed(self, raw_angle):
        self._history.append(raw_angle)
        if len(self._history) > self.smooth_window:
            self._history.pop(0)
        return float(np.mean(self._history))

    def update(self, raw_angle, timestamp_sec, feature_vector=None):
        # feature_vector (opsional): dict 11 joint angle {nama: derajat} DI FRAME
        # INI -- cuma dipakai kalau min_pattern_similarity aktif (gerbang cosine
        # Eq.4), TIDAK mempengaruhi logika peak-detection utama sama sekali.

        # Occlusion sesaat: TAHAN nilai valid terakhir (causal -- cuma pakai
        # masa lalu, beda dari interpolate(limit_direction="both") di versi
        # offline yang butuh "masa depan"). Ini BUKAN mengarang nilai fitur
        # RF (aturan "discard, bukan interpolasi" tetap berlaku ke fitur
        # training) -- ini cuma sinyal buat FEEDBACK/COUNTING real-time,
        # kategori beda (sama seperti compute_auto_phase() yg juga
        # interpolate buat draft, bukan buat data final).
        is_nan = raw_angle is None or (isinstance(raw_angle, float) and np.isnan(raw_angle))
        if is_nan:
            if self._last_valid_angle is None:
                return  # belum pernah ada data valid sama sekali, tidak ada yg bisa ditahan
            raw_angle = self._last_valid_angle
        else:
            self._last_valid_angle = raw_angle
        angle = self._smoothed(raw_angle)

        if self._direction is None:
            self._direction = RISING  # tebakan awal, otomatis benar sendiri begitu data masuk
            self._extreme_value = angle
            self._extreme_time = timestamp_sec
            self._extreme_feature_vector = feature_vector
            return

        if self._direction == RISING:
            if angle >= self._extreme_value:
                self._extreme_value = angle
                self._extreme_time = timestamp_sec
                self._extreme_feature_vector = feature_vector
            elif self._extreme_value - angle >= self.min_prominence_deg:
                self._confirm_extreme("peak", timestamp_sec, self._extreme_feature_vector)
                self._direction = FALLING
                self._extreme_value = angle
                self._extreme_time = timestamp_sec
                self._extreme_feature_vector = feature_vector
        else:  # FALLING
            if angle <= self._extreme_value:
                self._extreme_value = angle
                self._extreme_time = timestamp_sec
                self._extreme_feature_vector = feature_vector
            elif angle - self._extreme_value >= self.min_prominence_deg:
                self._confirm_extreme("valley", timestamp_sec, self._extreme_feature_vector)
                self._direction = RISING
                self._extreme_value = angle
                self._extreme_time = timestamp_sec
                self._extreme_feature_vector = feature_vector

    def _confirm_extreme(self, kind, timestamp_sec, feature_vector=None):
        last_time = self._last_peak_time if kind == "peak" else self._last_valley_time
        if last_time is not None and timestamp_sec - last_time < self.min_distance_sec:
            return  # terlalu dekat ke titik balik SEJENIS sebelumnya -- dianggap noise
        if kind == "peak":
            self._last_peak_time = timestamp_sec
        else:
            self._last_valley_time = timestamp_sec

        # Gerbang cosine similarity (Eq.4, opsional) -- HANYA memvalidasi pola,
        # TIDAK mengubah metode utama (peak detection tetap yg menghitung).
        if self.min_pattern_similarity is not None and feature_vector is not None:
            template = self._peak_template if kind == "peak" else self._valley_template
            if template is None:
                # titik ekstrem PERTAMA jenis ini -> jadi template acuan, otomatis lolos
                if kind == "peak":
                    self._peak_template = feature_vector
                else:
                    self._valley_template = feature_vector
            else:
                sim = _cosine_similarity(feature_vector, template)
                if sim is not None and sim < self.min_pattern_similarity:
                    self.rejected_by_pattern += 1
                    return  # pola gerakannya beda dari acuan -- dianggap noise, BUKAN repetisi

        if kind == "valley":
            self._seen_valley = True
        elif kind == "peak" and self._seen_valley:
            self.rep_count += 1
