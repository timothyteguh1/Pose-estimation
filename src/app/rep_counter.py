"""Repetition counting ONLINE (real-time/incremental) berbasis skeleton, bab
6.2.3 proposal: "Satu repetisi didefinisikan sebagai satu siklus lengkap dari
fase minimum ke maksimum dan kembali ke minimum (fase naik dan turun)."

Beda dari find_turning_points()/estimate_rep_count() (label_phase.py, OFFLINE
via scipy.find_peaks, butuh seluruh sinyal): versi ini ONLINE, state machine
causal yg cuma pakai data sampai saat ini, dgn prominence+distance sbg
gerbang anti-noise -- konsep sama, dihitung incremental.

1 rep dihitung saat PUNCAK terkonfirmasi (konsisten dgn versi offline yg
hitung jumlah puncak). Rep pertama butuh 1 lembah dulu sebelum puncak
dihitung.

Gerbang cosine similarity (Eq.4, opsional via min_pattern_similarity):
memvalidasi bahwa 1 siklus yg baru dikonfirmasi pola gerakannya mirip
siklus acuan (template = titik ekstrem sejenis pertama yg terkonfirmasi di
sesi itu) -- setelah lolos, baru peak detection (metode utama) menghitung
repetisinya. Default None (mati kecuali diaktifkan eksplisit)."""
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

# Tuning per-exercise dari sinyal produksi asli (LivePosturePipeline, bukan
# shortcut CSV) -- skema tuning/held-out per exercise (bagian video cari
# parameter, bagian lain dikunci sampai evaluasi akhir, padanan train/test
# split), ground truth dari label manual. Params ini baru divalidasi dari
# 1-2 partisipan -- wajib divalidasi ulang begitu partisipan lebih banyak.
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
        # Jarak minimum dicek terpisah per jenis (puncak-ke-puncak, lembah-ke-
        # lembah), bukan puncak-ke-lembah -- persis scipy.find_peaks() 2x
        # independen di find_turning_points() (versi offline).
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

        # Occlusion sesaat: tahan nilai valid terakhir (causal, beda dari
        # interpolate(limit_direction="both") di versi offline). Ini cuma
        # sinyal counting real-time, tidak mempengaruhi fitur training RF.
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
