import json
import pickle
import sys
from collections import Counter, deque
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.app.rep_counter import OnlineRepCounter
from src.app.warnings_content import get_warning
from src.features.joint_angles import (
    ANGLE_COLUMNS,
    POSE_LANDMARK_NAMES,
    compute_frame_angles,
    primary_angle_for_frame,
)

MODELS_DIR = EXPERIMENT_ROOT / "models"

SEG_HYSTERESIS_DEG = 8.0
SEG_MIN_SEC = 0.5
PHASE_GATE_TOP_FRAC = 0.30
MIN_SEG_WINDOWS = 3


def _majority_class(preds):
    non_none = [p for p in preds if p is not None]
    if not non_none:
        return None
    return Counter(non_none).most_common(1)[0][0]


def _phase_gated_seg_class(seg_windows, min_amplitude_deg):
    # min_amplitude_deg: reuse OnlineRepCounter.min_prominence_deg (BUKAN angka
    # baru) -- siklus dgn rentang sudut (hi-lo) di bawah ini dianggap goyangan
    # kecil (mis. berdiri diam), BUKAN repetisi sungguhan. Sama persis dgn
    # app.py produksi.
    pairs = [(c, a) for (c, a) in seg_windows if c is not None and a is not None]
    if len(pairs) < MIN_SEG_WINDOWS:
        return None
    angles = [a for (_, a) in pairs]
    lo, hi = min(angles), max(angles)
    if hi - lo < min_amplitude_deg:
        return None
    cut = lo + (hi - lo) * (1.0 - PHASE_GATE_TOP_FRAC)
    kept = [c for (c, a) in pairs if a <= cut]
    if len(kept) < MIN_SEG_WINDOWS:
        return None
    return _majority_class(kept)


def _update_seg_state(seg_state, a, t):
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
        return True
    else:
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


class LandmarkPosturePipeline:
    def __init__(self, exercise, countdown_sec=5.0, class_confidence_threshold=0.0):
        self.exercise = exercise
        self.countdown_sec = countdown_sec
        self.class_confidence_threshold = class_confidence_threshold
        self._session_start_ts = None

        model_path = MODELS_DIR / f"{exercise}_rf.pkl"
        config_path = MODELS_DIR / f"{exercise}_feature_config.json"
        with open(model_path, "rb") as f:
            self.pipeline = pickle.load(f)
        with open(config_path) as f:
            self.config = json.load(f)
        self.feat_cols = self.config["feat_cols"]
        self.window_sec = self.config["window_sec"]
        self.visibility_threshold = self.config["visibility_threshold"]
        self.use_z = self.config.get("use_z", False)

        self.rep_counter = OnlineRepCounter(exercise, min_pattern_similarity=0.70)

        self._buffer = None
        self._swl = None
        self._last_confidence = None

        self.seg_state = {"direction": None, "extreme_val": None, "last_confirm_t": None}
        self.seg_windows = []
        self.session_log = []

    def _ensure_buffer(self, fps):
        if self._buffer is not None:
            return
        if not (fps == fps) or fps <= 0 or fps > 240:
            fps = 30.0
        self._swl = max(1, round(fps * self.window_sec))
        self._buffer = deque(maxlen=self._swl)

    def process_landmarks(self, landmarks, timestamp_sec, fps):
        if self._session_start_ts is None:
            self._session_start_ts = timestamp_sec
        elapsed = timestamp_sec - self._session_start_ts
        in_countdown = elapsed < self.countdown_sec

        self._ensure_buffer(fps)

        row = {}
        if landmarks is not None:
            for name, (x, y, z, v) in zip(POSE_LANDMARK_NAMES, landmarks):
                row[f"{name}_x"] = x
                row[f"{name}_y"] = y
                row[f"{name}_z"] = z
                row[f"{name}_v"] = v
        else:
            for name in POSE_LANDMARK_NAMES:
                for axis in ("x", "y", "z", "v"):
                    row[f"{name}_{axis}"] = float("nan")

        angles, _has_data = compute_frame_angles(row, self.visibility_threshold, self.use_z)
        primary = primary_angle_for_frame(row, self.exercise, self.visibility_threshold)
        primary = None if primary != primary else primary

        predicted_class = None
        confidence = None
        cycle_result = None
        if not in_countdown:
            self.rep_counter.update(primary, timestamp_sec, feature_vector=angles)

            frame_features = dict(angles)
            for name in POSE_LANDMARK_NAMES:
                for axis in ("x", "y", "z", "v"):
                    frame_features[f"{name}_{axis}"] = row[f"{name}_{axis}"]
            self._buffer.append(frame_features)

            if len(self._buffer) == self._swl:
                predicted_class = self._predict_from_buffer()
                confidence = self._last_confidence

            self.seg_windows.append((predicted_class, primary))
            seg_done = _update_seg_state(self.seg_state, primary, timestamp_sec)
            if seg_done:
                seg_class = _phase_gated_seg_class(self.seg_windows, self.rep_counter.min_prominence_deg)
                self.seg_windows = []
                warning = get_warning(seg_class)
                cycle_result = {
                    "rep_no": len(self.session_log) + 1,
                    "class": seg_class,
                    "warning": warning,
                }
                self.session_log.append(cycle_result)

        return {
            "primary_angle": primary,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "rep_count": self.rep_counter.rep_count,
            "countdown_remaining": (self.countdown_sec - elapsed) if in_countdown else None,
            "cycle_result": cycle_result,
        }

    def _predict_from_buffer(self):
        window_row = {}
        for f, frame_feat in enumerate(self._buffer):
            for col in ANGLE_COLUMNS:
                window_row[f"{col}_f{f}"] = frame_feat[col]
        for name in POSE_LANDMARK_NAMES:
            for axis in ("x", "y", "z", "v"):
                col = f"{name}_{axis}"
                values = [frame_feat[col] for frame_feat in self._buffer]
                valid = [v for v in values if v == v]
                window_row[f"coord_mean_{col}"] = (sum(valid) / len(valid)) if valid else float("nan")

        if any(v != v for v in window_row.values()):
            self._last_confidence = None
            return None

        X = pd.DataFrame([{c: window_row[c] for c in self.feat_cols}], columns=self.feat_cols)
        proba = self.pipeline.predict_proba(X)[0]
        rf_step = self.pipeline.named_steps["rf"]
        best_idx = proba.argmax()
        confidence = float(proba[best_idx])
        self._last_confidence = confidence
        if confidence < self.class_confidence_threshold:
            return None
        return rf_step.classes_[best_idx]
