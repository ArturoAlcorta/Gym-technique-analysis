"""
Within-rep joint coordination for squat / RDL (hip ↔ knee).

Coordination is measured here (NOT from the DTW cost — DTW warps the time axis and
absorbs exactly the lead/lag we want to read, spec Principle 4 / anti-pattern). Two
complementary measures, both on the hip and knee angle series of a single rep:

  hip_knee_lag_pct  — resample the rep to a fixed phase axis, take each joint's angular
                      velocity, cross-correlate → signed lag at peak, in % of rep.
                      sign > 0 ⇒ hip leads knee; < 0 ⇒ knee leads hip. Magnitude scores,
                      sign drives the coaching cue.
  marp_deg          — Continuous Relative Phase: per-joint phase angle from a normalized
                      phase portrait (angle vs angular velocity), CRP(t)=φ_hip−φ_knee;
                      MARP = mean |CRP| over the rep (overall hip/knee coupling).

Resampling to a phase axis is appropriate *here* (cross-correlation needs equal-length
signals and lag must be expressed in % of rep) — unlike the DTW pattern score, where
DTW already handles tempo so no resampling is used.
"""

import numpy as np
from scipy.signal import correlate, correlation_lags

COORD_METRICS = ["hip_knee_lag_pct", "marp_deg"]
_PHASE_POINTS = 101


def _joint_series(angles_rep: list[dict], left: str, right: str) -> np.ndarray | None:
    """Mean of the two sides per frame (e.g. knee_L/knee_R), forward-filled."""
    vals, prev = [], None
    for f in angles_rep:
        a, b = f["angles"].get(left), f["angles"].get(right)
        pair = [v for v in (a, b) if v is not None]
        v = float(np.mean(pair)) if pair else prev
        vals.append(v)
        prev = v
    arr = np.array([v for v in vals if v is not None], dtype=np.float64)
    return arr if len(arr) >= 4 else None


def _resample(x: np.ndarray, n: int = _PHASE_POINTS) -> np.ndarray:
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x)


def _norm(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-range normalization to [-1, 1]-ish for phase portraits / xcorr."""
    x = x - x.mean()
    m = np.max(np.abs(x))
    return x / m if m > 1e-9 else x


def hip_knee_lag_pct(hip: np.ndarray, knee: np.ndarray) -> float:
    """Signed lag (% of rep) of the knee vs hip angular-velocity signals."""
    hv = _norm(np.gradient(_resample(hip)))
    kv = _norm(np.gradient(_resample(knee)))
    c = correlate(hv, kv, mode="full")
    lags = correlation_lags(len(hv), len(kv), mode="full")
    lag = int(lags[int(np.argmax(c))])        # >0 ⇒ knee lags hip ⇒ hip leads
    return round(100.0 * lag / _PHASE_POINTS, 3)


def marp_deg(hip: np.ndarray, knee: np.ndarray) -> float:
    """Mean Absolute Relative Phase (degrees) between hip and knee."""
    def phi(x):
        xn = _norm(_resample(x))
        vn = _norm(np.gradient(_resample(x)))
        return np.degrees(np.arctan2(vn, xn))
    crp = phi(hip) - phi(knee)
    crp = (crp + 180.0) % 360.0 - 180.0       # wrap to [-180, 180]
    return round(float(np.mean(np.abs(crp))), 3)


def compute_coordination(angles_rep: list[dict]) -> dict[str, float | None]:
    hip = _joint_series(angles_rep, "hip_L", "hip_R")
    knee = _joint_series(angles_rep, "knee_L", "knee_R")
    if hip is None or knee is None:
        return {"hip_knee_lag_pct": None, "marp_deg": None}
    return {"hip_knee_lag_pct": hip_knee_lag_pct(hip, knee), "marp_deg": marp_deg(hip, knee)}
