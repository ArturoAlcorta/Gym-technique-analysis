"""
Conteo de repeticiones por ejercicio desde keypoints normalizados.

Para añadir un nuevo ejercicio:
  1. Subclasifica ExerciseRepCounter e implementa count_reps_from_frames y rep_counting_config.
  2. Añade una entrada en REP_COUNTERS con el exercise_id correspondiente.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

SMOOTHING_WINDOW = 5
MIN_FRAMES_BETWEEN_REPS = 10


# ── Shared geometry utilities ─────────────────────────────────────────────────

def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    ba, bc = a - b, c - b
    na, nb = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-6 or nb < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (na * nb), -1.0, 1.0))))


def _best_available(keypoints: dict, candidates: list[str]) -> np.ndarray | None:
    best, best_score = None, -1.0
    for name in candidates:
        kp = keypoints.get(name)
        if kp is not None and kp["score"] > best_score:
            best, best_score = np.array([kp["x"], kp["y"]]), kp["score"]
    return best


def _torso_inclination(hip: np.ndarray, shoulder: np.ndarray) -> float | None:
    vec = shoulder - hip
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(vec, np.array([0.0, -1.0])) / norm, -1.0, 1.0))))


def _smooth(signal: np.ndarray, window: int) -> np.ndarray:
    smoothed = np.full_like(signal, np.nan)
    half = window // 2
    for i in range(len(signal)):
        s, e = max(0, i - half), min(len(signal), i + half + 1)
        chunk = signal[s:e]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) > 0:
            smoothed[i] = np.mean(valid)
    return smoothed


# ── Base class ────────────────────────────────────────────────────────────────

class ExerciseRepCounter(ABC):

    @property
    @abstractmethod
    def exercise_name(self) -> str: ...

    @abstractmethod
    def count_reps_from_frames(self, data: dict) -> tuple[list[dict], dict]: ...

    @abstractmethod
    def rep_counting_config(self) -> dict: ...


# ── Squat ─────────────────────────────────────────────────────────────────────

class SquatRepCounter(ExerciseRepCounter):
    exercise_name = "squat"

    _KNEE_TOP    = 160
    _KNEE_BOTTOM = 90
    _HIP_DOWN    = 120
    _HIP_UP      = 160

    _ANKLE    = ["l_ankle", "r_ankle", "L_Ankle", "R_Ankle"]
    _KNEE     = ["l_knee", "r_knee", "l_mknee", "r_mknee", "L_Knee", "R_Knee"]
    _HIP      = ["l_ASIS", "r_ASIS", "L_Hip", "R_Hip"]
    _SHOULDER = ["L_Shoulder", "R_Shoulder", "lshoulder", "rshoulder"]

    def count_reps_from_frames(self, data: dict) -> tuple[list[dict], dict]:
        frames = data["frames"]
        knee_raw, hip_raw, ts = self._extract(frames)
        return self._count(frames, _smooth(knee_raw, SMOOTHING_WINDOW), _smooth(hip_raw, SMOOTHING_WINDOW), ts)

    def _extract(self, frames):
        knee_a, hip_a, ts = [], [], []
        for f in frames:
            kps = f["keypoints"]
            ts.append(f["timestamp_s"])
            a = _best_available(kps, self._ANKLE)
            k = _best_available(kps, self._KNEE)
            h = _best_available(kps, self._HIP)
            s = _best_available(kps, self._SHOULDER)
            knee_a.append(_angle_deg(a, k, h) if all(x is not None for x in [a, k, h]) else np.nan)
            hip_a.append(_angle_deg(s, h, k) if all(x is not None for x in [s, h, k]) else np.nan)
        return np.array(knee_a), np.array(hip_a), ts

    def _count(self, frames, knee_s, hip_s, ts):
        knee_st, hip_st = "TOP", "UP"
        reps = []
        last_end = -MIN_FRAMES_BETWEEN_REPS
        sf = bf = None
        mn_k = mn_h = np.inf
        hip_down = reached = False

        for i, (k, h) in enumerate(zip(knee_s, hip_s)):
            fid = frames[i]["frame_idx"]

            if not np.isnan(k):
                if knee_st == "TOP" and k < self._KNEE_TOP:
                    knee_st = "DESCENDING"
                    sf = i; bf = None; mn_k = mn_h = np.inf; reached = False
                    hip_down = hip_st == "DOWN"

                elif knee_st == "DESCENDING":
                    if k < self._KNEE_BOTTOM:
                        knee_st = "BOTTOM"; reached = True
                    elif k > self._KNEE_TOP:
                        knee_st = "TOP"
                        sf = bf = None; reached = hip_down = False; mn_k = mn_h = np.inf

                elif knee_st == "BOTTOM" and k > self._KNEE_BOTTOM:
                    knee_st = "ASCENDING"

                elif knee_st == "ASCENDING":
                    if k > self._KNEE_TOP:
                        knee_st = "TOP"
                        if sf is not None and (i - last_end) >= MIN_FRAMES_BETWEEN_REPS:
                            t0 = ts[sf]
                            tb = ts[bf] if bf is not None else ts[i]
                            te = ts[i]
                            reps.append({
                                "rep_number":            len(reps) + 1,
                                "frame_start":           frames[sf]["frame_idx"],
                                "frame_bottom":          frames[bf]["frame_idx"] if bf is not None else fid,
                                "frame_end":             fid,
                                "timestamp_start":       t0,
                                "timestamp_bottom":      tb,
                                "timestamp_end":         te,
                                "duration_total_s":      round(te - t0, 3),
                                "duration_eccentric_s":  round(tb - t0, 3),
                                "duration_concentric_s": round(te - tb, 3),
                                "min_knee_angle":        round(mn_k, 2) if mn_k != np.inf else None,
                                "min_hip_angle":         round(mn_h, 2) if mn_h != np.inf else None,
                                "confirmed_by":          "both" if hip_down else "knee_only",
                                "reached_bottom":        reached,
                            })
                            last_end = i
                        sf = bf = None; reached = hip_down = False; mn_k = mn_h = np.inf

                    elif k < self._KNEE_BOTTOM:
                        knee_st = "BOTTOM"

            if not np.isnan(h):
                if hip_st == "UP" and h < self._HIP_DOWN:
                    hip_st = "DOWN"; hip_down = True
                elif hip_st == "DOWN" and h > self._HIP_UP:
                    hip_st = "UP"

            if knee_st in ("DESCENDING", "BOTTOM", "ASCENDING"):
                if not np.isnan(k) and k < mn_k:
                    mn_k = k; bf = i
                if not np.isnan(h):
                    mn_h = min(mn_h, h)

        stats = {
            "total_reps":                len(reps),
            "confirmed_by_both":         sum(1 for r in reps if r["confirmed_by"] == "both"),
            "confirmed_by_knee_only":    sum(1 for r in reps if r["confirmed_by"] == "knee_only"),
            "reps_reached_bottom":       sum(1 for r in reps if r["reached_bottom"]),
            "avg_knee_angle_bottom":     round(float(np.nanmean([r["min_knee_angle"] for r in reps if r["min_knee_angle"]])), 2) if reps else None,
            "avg_hip_angle_bottom":      round(float(np.nanmean([r["min_hip_angle"]  for r in reps if r["min_hip_angle"]])),  2) if reps else None,
            "avg_rep_duration_s":        round(float(np.mean([r["duration_total_s"]      for r in reps])), 3) if reps else None,
            "avg_eccentric_duration_s":  round(float(np.mean([r["duration_eccentric_s"]  for r in reps])), 3) if reps else None,
            "avg_concentric_duration_s": round(float(np.mean([r["duration_concentric_s"] for r in reps])), 3) if reps else None,
        }
        return reps, stats

    def rep_counting_config(self) -> dict:
        return {
            "exercise":                self.exercise_name,
            "state_machine":           "4-state",
            "smoothing_window":        SMOOTHING_WINDOW,
            "knee_threshold_top":      self._KNEE_TOP,
            "knee_threshold_bottom":   self._KNEE_BOTTOM,
            "hip_threshold_down":      self._HIP_DOWN,
            "hip_threshold_up":        self._HIP_UP,
            "min_frames_between_reps": MIN_FRAMES_BETWEEN_REPS,
        }


# ── Bench press ───────────────────────────────────────────────────────────────

class BenchRepCounter(ExerciseRepCounter):
    exercise_name = "bench_press"

    _ELBOW_TOP    = 145
    _ELBOW_BOTTOM = 100
    _ASYM_THRESH  = 15.0

    _L_SHOULDER = ["L_Shoulder", "lshoulder"]
    _R_SHOULDER = ["R_Shoulder", "rshoulder"]
    _L_ELBOW    = ["l_lelbow", "l_melbow", "L_Elbow"]
    _R_ELBOW    = ["r_lelbow", "r_melbow", "R_Elbow"]
    _L_WRIST    = ["l_lwrist", "l_mwrist", "L_Wrist"]
    _R_WRIST    = ["r_lwrist", "r_mwrist", "R_Wrist"]

    def count_reps_from_frames(self, data: dict) -> tuple[list[dict], dict]:
        frames = data["frames"]
        el_raw, er_raw, avg_raw, ts = self._extract(frames)
        el_s   = _smooth(el_raw,  SMOOTHING_WINDOW)
        er_s   = _smooth(er_raw,  SMOOTHING_WINDOW)
        avg_s  = _smooth(avg_raw, SMOOTHING_WINDOW)
        return self._count(frames, el_s, er_s, avg_s, ts)

    def _extract(self, frames):
        el_a, er_a, avg_a, ts = [], [], [], []
        for f in frames:
            kps = f["keypoints"]
            ts.append(f["timestamp_s"])
            ls = _best_available(kps, self._L_SHOULDER)
            rs = _best_available(kps, self._R_SHOULDER)
            le = _best_available(kps, self._L_ELBOW)
            re = _best_available(kps, self._R_ELBOW)
            lw = _best_available(kps, self._L_WRIST)
            rw = _best_available(kps, self._R_WRIST)
            al = _angle_deg(ls, le, lw) if all(x is not None for x in [ls, le, lw]) else np.nan
            ar = _angle_deg(rs, re, rw) if all(x is not None for x in [rs, re, rw]) else np.nan
            el_a.append(al); er_a.append(ar)
            vals = [v for v in [al, ar] if not np.isnan(v)]
            avg_a.append(np.mean(vals) if vals else np.nan)
        return np.array(el_a), np.array(er_a), np.array(avg_a), ts

    def _count(self, frames, el_s, er_s, avg_s, ts):
        state = "TOP"
        reps = []
        last_end = -MIN_FRAMES_BETWEEN_REPS
        sf = bf = None
        mn_avg = mn_l = mn_r = np.inf
        asymmetries: list[float] = []
        reached = False

        for i, (al, ar, aa) in enumerate(zip(el_s, er_s, avg_s)):
            fid = frames[i]["frame_idx"]

            if not np.isnan(aa):
                if state == "TOP" and aa < self._ELBOW_TOP:
                    state = "DESCENDING"
                    sf = i; bf = None; mn_avg = mn_l = mn_r = np.inf; asymmetries = []; reached = False

                elif state == "DESCENDING":
                    if aa < self._ELBOW_BOTTOM:
                        state = "BOTTOM"; reached = True
                    elif aa > self._ELBOW_TOP:
                        state = "TOP"
                        sf = bf = None; reached = False; mn_avg = mn_l = mn_r = np.inf; asymmetries = []

                elif state == "BOTTOM" and aa > self._ELBOW_BOTTOM:
                    state = "ASCENDING"

                elif state == "ASCENDING":
                    if aa > self._ELBOW_TOP:
                        state = "TOP"
                        if sf is not None and (i - last_end) >= MIN_FRAMES_BETWEEN_REPS:
                            avg_asym = float(np.mean(asymmetries)) if asymmetries else 0.0
                            max_asym = float(np.max(asymmetries))  if asymmetries else 0.0
                            t0 = ts[sf]
                            tb = ts[bf] if bf is not None else ts[i]
                            te = ts[i]
                            reps.append({
                                "rep_number":            len(reps) + 1,
                                "frame_start":           frames[sf]["frame_idx"],
                                "frame_bottom":          frames[bf]["frame_idx"] if bf is not None else fid,
                                "frame_end":             fid,
                                "timestamp_start":       t0,
                                "timestamp_bottom":      tb,
                                "timestamp_end":         te,
                                "duration_total_s":      round(te - t0, 3),
                                "duration_eccentric_s":  round(tb - t0, 3),
                                "duration_concentric_s": round(te - tb, 3),
                                "min_elbow_avg":         round(mn_avg, 2) if mn_avg != np.inf else None,
                                "min_elbow_l":           round(mn_l,   2) if mn_l   != np.inf else None,
                                "min_elbow_r":           round(mn_r,   2) if mn_r   != np.inf else None,
                                "max_asymmetry":         round(max_asym, 2),
                                "avg_asymmetry":         round(avg_asym, 2),
                                "asymmetry_flag":        avg_asym > self._ASYM_THRESH,
                                "reached_bottom":        reached,
                            })
                            last_end = i
                        sf = bf = None; reached = False; mn_avg = mn_l = mn_r = np.inf; asymmetries = []

                    elif aa < self._ELBOW_BOTTOM:
                        state = "BOTTOM"

            if state in ("DESCENDING", "BOTTOM", "ASCENDING"):
                if not np.isnan(aa) and aa < mn_avg:
                    mn_avg = aa; bf = i
                if not np.isnan(al): mn_l = min(mn_l, al)
                if not np.isnan(ar): mn_r = min(mn_r, ar)
                if not np.isnan(al) and not np.isnan(ar):
                    asymmetries.append(abs(al - ar))

        stats = {
            "total_reps":                len(reps),
            "reps_with_asymmetry_flag":  sum(1 for r in reps if r["asymmetry_flag"]),
            "reps_reached_bottom":       sum(1 for r in reps if r["reached_bottom"]),
            "avg_elbow_angle_bottom":    round(float(np.nanmean([r["min_elbow_avg"] for r in reps if r["min_elbow_avg"]])), 2) if reps else None,
            "avg_asymmetry_deg":         round(float(np.nanmean([r["avg_asymmetry"] for r in reps])), 2) if reps else None,
            "avg_rep_duration_s":        round(float(np.mean([r["duration_total_s"]      for r in reps])), 3) if reps else None,
            "avg_eccentric_duration_s":  round(float(np.mean([r["duration_eccentric_s"]  for r in reps])), 3) if reps else None,
            "avg_concentric_duration_s": round(float(np.mean([r["duration_concentric_s"] for r in reps])), 3) if reps else None,
        }
        return reps, stats

    def rep_counting_config(self) -> dict:
        return {
            "exercise":                self.exercise_name,
            "state_machine":           "4-state",
            "smoothing_window":        SMOOTHING_WINDOW,
            "elbow_threshold_top":     self._ELBOW_TOP,
            "elbow_threshold_bottom":  self._ELBOW_BOTTOM,
            "asymmetry_threshold_deg": self._ASYM_THRESH,
            "min_frames_between_reps": MIN_FRAMES_BETWEEN_REPS,
        }


# ── RDL ───────────────────────────────────────────────────────────────────────

class RDLRepCounter(ExerciseRepCounter):
    exercise_name = "rdl"

    _HIP_TOP    = 165
    # Un RDL real hace bisagra de cadera hasta ~75-100°. Con BOTTOM=150 (banda de
    # solo 15°) cualquier balanceo leve entre reps cruzaba el umbral y contaba una
    # rep fantasma (14 detectadas vs 5 reales en vídeo de referencia). 125° separa
    # limpiamente las reps reales (mín. ~99°) del ruido superficial (mín. ~133°).
    _HIP_BOTTOM = 125
    _TORSO_DOWN = 30
    _TORSO_UP   = 15

    _HIP      = ["l_ASIS", "r_ASIS", "L_Hip", "R_Hip"]
    _KNEE     = ["l_knee", "r_knee", "l_mknee", "r_mknee", "L_Knee", "R_Knee"]
    _SHOULDER = ["L_Shoulder", "R_Shoulder", "lshoulder", "rshoulder"]

    def count_reps_from_frames(self, data: dict) -> tuple[list[dict], dict]:
        frames = data["frames"]
        hip_raw, torso_raw, ts = self._extract(frames)
        return self._count(
            frames,
            _smooth(hip_raw,   SMOOTHING_WINDOW),
            _smooth(torso_raw, SMOOTHING_WINDOW),
            ts,
        )

    def _extract(self, frames):
        hip_a, torso_a, ts = [], [], []
        for f in frames:
            kps = f["keypoints"]
            ts.append(f["timestamp_s"])
            hip  = _best_available(kps, self._HIP)
            knee = _best_available(kps, self._KNEE)
            sho  = _best_available(kps, self._SHOULDER)
            hip_a.append(_angle_deg(sho, hip, knee) if all(x is not None for x in [sho, hip, knee]) else np.nan)
            torso_a.append(_torso_inclination(hip, sho) if all(x is not None for x in [hip, sho]) else np.nan)
        return np.array(hip_a), np.array(torso_a), ts

    def _count(self, frames, hip_s, torso_s, ts):
        hip_st   = "TOP"
        torso_st = "UP"
        reps = []
        last_end = -MIN_FRAMES_BETWEEN_REPS
        sf = bf = None
        mn_hip = np.inf
        mx_torso = -np.inf
        torso_down = reached = False

        for i, (h, t) in enumerate(zip(hip_s, torso_s)):
            fid = frames[i]["frame_idx"]

            if not np.isnan(h):
                if hip_st == "TOP" and h < self._HIP_TOP:
                    hip_st = "DESCENDING"
                    sf = i; bf = None; mn_hip = np.inf; mx_torso = -np.inf; reached = False
                    torso_down = torso_st == "DOWN"

                elif hip_st == "DESCENDING":
                    if h < self._HIP_BOTTOM:
                        hip_st = "BOTTOM"; reached = True
                    elif h > self._HIP_TOP:
                        hip_st = "TOP"
                        sf = bf = None; reached = torso_down = False; mn_hip = np.inf; mx_torso = -np.inf

                elif hip_st == "BOTTOM" and h > self._HIP_BOTTOM:
                    hip_st = "ASCENDING"

                elif hip_st == "ASCENDING":
                    if h > self._HIP_TOP:
                        hip_st = "TOP"
                        if sf is not None and (i - last_end) >= MIN_FRAMES_BETWEEN_REPS:
                            t0 = ts[sf]
                            tb = ts[bf] if bf is not None else ts[i]
                            te = ts[i]
                            reps.append({
                                "rep_number":            len(reps) + 1,
                                "frame_start":           frames[sf]["frame_idx"],
                                "frame_bottom":          frames[bf]["frame_idx"] if bf is not None else fid,
                                "frame_end":             fid,
                                "timestamp_start":       t0,
                                "timestamp_bottom":      tb,
                                "timestamp_end":         te,
                                "duration_total_s":      round(te - t0, 3),
                                "duration_eccentric_s":  round(tb - t0, 3),
                                "duration_concentric_s": round(te - tb, 3),
                                "min_hip_angle":         round(mn_hip,   2) if mn_hip   != np.inf  else None,
                                "max_torso_angle":       round(mx_torso, 2) if mx_torso != -np.inf else None,
                                "confirmed_by":          "both" if torso_down else "hip_only",
                                "reached_bottom":        reached,
                            })
                            last_end = i
                        sf = bf = None; reached = torso_down = False; mn_hip = np.inf; mx_torso = -np.inf

                    elif h < self._HIP_BOTTOM:
                        hip_st = "BOTTOM"

            if not np.isnan(t):
                if torso_st == "UP" and t > self._TORSO_DOWN:
                    torso_st = "DOWN"; torso_down = True
                elif torso_st == "DOWN" and t < self._TORSO_UP:
                    torso_st = "UP"

            if hip_st in ("DESCENDING", "BOTTOM", "ASCENDING"):
                if not np.isnan(h) and h < mn_hip:
                    mn_hip = h; bf = i
                if not np.isnan(t):
                    mx_torso = max(mx_torso, t)

        stats = {
            "total_reps":                len(reps),
            "confirmed_by_both":         sum(1 for r in reps if r["confirmed_by"] == "both"),
            "confirmed_by_hip_only":     sum(1 for r in reps if r["confirmed_by"] == "hip_only"),
            "reps_reached_bottom":       sum(1 for r in reps if r["reached_bottom"]),
            "avg_hip_angle_bottom":      round(float(np.nanmean([r["min_hip_angle"]   for r in reps if r["min_hip_angle"]])),   2) if reps else None,
            "avg_torso_angle_bottom":    round(float(np.nanmean([r["max_torso_angle"] for r in reps if r["max_torso_angle"]])), 2) if reps else None,
            "avg_rep_duration_s":        round(float(np.mean([r["duration_total_s"]      for r in reps])), 3) if reps else None,
            "avg_eccentric_duration_s":  round(float(np.mean([r["duration_eccentric_s"]  for r in reps])), 3) if reps else None,
            "avg_concentric_duration_s": round(float(np.mean([r["duration_concentric_s"] for r in reps])), 3) if reps else None,
        }
        return reps, stats

    def rep_counting_config(self) -> dict:
        return {
            "exercise":                self.exercise_name,
            "state_machine":           "4-state",
            "smoothing_window":        SMOOTHING_WINDOW,
            "hip_threshold_top":       self._HIP_TOP,
            "hip_threshold_bottom":    self._HIP_BOTTOM,
            "torso_threshold_down":    self._TORSO_DOWN,
            "torso_threshold_up":      self._TORSO_UP,
            "min_frames_between_reps": MIN_FRAMES_BETWEEN_REPS,
        }


# ── Registry ──────────────────────────────────────────────────────────────────

_squat = SquatRepCounter()
_bench = BenchRepCounter()
_rdl   = RDLRepCounter()

REP_COUNTERS: dict[int, ExerciseRepCounter] = {
    6:  _squat,   # sentadilla
    7:  _rdl,     # RDL
    1:  _bench,   # press banca
    2:  _bench,   # press banca (alias)
    48: _rdl,     # RDL (alias)
}


# ── Public function ───────────────────────────────────────────────────────────

def count_repetitions(
    normalized_json_path: Path,
    exercise_id: int,
) -> tuple[list[dict], dict, Path]:
    """
    Cuenta repeticiones a partir del JSON de keypoints normalizados.
    Devuelve (reps, stats, output_path).
    """
    counter = REP_COUNTERS.get(exercise_id)
    if counter is None:
        raise ValueError(
            f"exercise_id={exercise_id} no tiene contador registrado. "
            f"IDs disponibles: {list(REP_COUNTERS)}"
        )

    with open(normalized_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    reps, stats = counter.count_reps_from_frames(data)

    output_path = normalized_json_path.parent / "rep_count.json"
    result = {"reps": reps, "stats": stats}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Conteo de repeticiones → {output_path}  (total: {stats['total_reps']})")
    return reps, stats, output_path