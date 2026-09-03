"""
Per-rep relational fault metrics (squat / RDL), pure numpy — 2D.

All are *relations between* joints — robust to ROM and body-proportion differences,
never absolute key-event thresholds. Shared by the reference-band builder and the scorer
so trainers and users are measured identically. Computed on 2D normalized keypoints
(filming ~30° from front/back, both sides visible).

Inputs are per-rep slices:
  angles_rep : list of {"angles": {channel: deg}} frames  (from angles2d_service)
  pos_rep    : list of {"keypoints": {name: {x,y,score}}} normalized-2D frames

Metrics:
  valgus_index         peak ankle_width / knee_width (>1 = caving)    (ratio)
                       None unless the camera actually sees the frontal plane
  spine_flex_delta     max thoraco-lumbar flexion below standing      (deg)
  hip_knee_rom_ratio   hip ROM / knee ROM (RDL: high = hip-dominant)  (ratio)
"""

import numpy as np

from .coordination_service import COORD_METRICS, compute_coordination

_SPINE_TOP = ["C7", "T6"]        # C7 preferred, T6 fallback if occluded
_SPINE_MID = ["L2", "T11"]       # L2 preferred, T11 fallback
_KNEE_ROM_FLOOR = 3.0            # deg — avoid div-by-tiny in the hip/knee ROM ratio
_HIP_KNEE_RATIO_CAP = 20.0


def _pos(frame: dict, name: str) -> np.ndarray | None:
    k = frame["keypoints"].get(name)
    if k is None:
        return None
    return np.array([k["x"], k["y"]], dtype=np.float64)


def _first_pos(frame: dict, names: list[str]) -> np.ndarray | None:
    for n in names:
        p = _pos(frame, name=n)
        if p is not None:
            return p
    return None


def _mid_hip(frame: dict) -> np.ndarray | None:
    l, r = _pos(frame, "L_Hip"), _pos(frame, "R_Hip")
    pair = [p for p in (l, r) if p is not None]
    return np.mean(pair, axis=0) if pair else None


def _series(angles_rep: list[dict], channel: str) -> np.ndarray:
    vals = [f["angles"].get(channel) for f in angles_rep]
    return np.array([v for v in vals if v is not None], dtype=np.float64)


def _rom(angles_rep: list[dict], channel: str) -> float | None:
    s = _series(angles_rep, channel)
    return float(s.max() - s.min()) if len(s) else None


def _joint_angle(a, b, c):
    if a is None or b is None or c is None:
        return None
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


_VALGUS_INDEX_CAP = 3.0

# Valgus needs the frontal plane: if the camera is side-on, both ankles (and both knees)
# project onto nearly the same x, and the ankle/knee width ratio becomes pure keypoint
# jitter — a squat filmed sagittally measured a "3.0" (capped, i.e. severe cave) purely
# from ~1px of noise. Gate on how far apart the ankles actually project, relative to the
# shank length in the same frame (dimensionless, so it survives any normalization scale):
# side-on footage sits at 0.005–0.07, whereas a hip-width stance seen from ~30° off the
# sagittal plane projects to roughly 0.4. Below the cutoff we report no valgus at all
# rather than a number that is not measurable from that angle.
VALGUS_MIN_VIEW_RATIO = 0.15

# Valgus is judged against an ABSOLUTE cutoff, not a reference band: a single camera at one
# angle can't support a per-side or reference-relative valgus measure, so a fixed ankle/knee
# width ratio is the honest yardstick. The cue is raised on index > the flag.
#
# The cutoff is per exercise, because the neutral ratio is. In a squat you actively drive the
# knees out, so knees wider than the feet is the expectation and anything approaching parity
# is already a fault. An RDL is a narrow stance with the knees soft and tracking over the feet,
# so a ratio around 1 is plain geometry, not knee cave — every reference RDL rep in this repo
# measures 1.01–1.14, and against the squat cutoff all five would be told their knees were
# caving in. Both thresholds keep the same shape: flag just clear of the reference spread,
# sub-score reaching 0 a further 0.35 out. The RDL numbers rest on five reps from one athlete,
# so treat them as provisional until that reference set grows.
_VALGUS_CUTOFFS: dict[int, tuple[float, float]] = {
    6: (0.90, 1.25),    # squat      — references measure 0.63–0.82
    7: (1.20, 1.55),    # RDL        — references measure 1.01–1.14
    48: (1.20, 1.55),   # RDL (alias)
}
_DEFAULT_CUTOFF = (0.90, 1.25)

# Kept for callers that only need the squat figure (and as the documented default).
VALGUS_FLAG = _DEFAULT_CUTOFF[0]


def valgus_cutoffs(exercise_id: int | None = None) -> tuple[float, float]:
    """(flag, zero) for an exercise: the ratio above which knee cave is called, and
    the ratio at which its sub-score bottoms out."""
    return _VALGUS_CUTOFFS.get(exercise_id, _DEFAULT_CUTOFF)


def valgus_score(index: float, exercise_id: int | None = None) -> float:
    """0-100 sub-score for the knee-cave index: a flat 100 at or below the flag
    (no valgus, so no penalty), decaying linearly to 0 at the zero point. Decoupled
    from the fault, which is raised on index > flag, so a rep just inside the
    cutoff keeps full marks and no warning."""
    flag, zero = valgus_cutoffs(exercise_id)
    if index <= flag:
        return 100.0
    return float(max(0.0, 100.0 * (zero - index) / (zero - flag)))


def _valgus_index(pos_rep: list[dict]) -> float | None:
    """
    Knee-cave index at the bottom of the rep = median over the deepest quartile of frames of
    ( ankle_width / knee_width ), where each width is the mediolateral (horizontal) separation
    between the two ankles / two knees.

    Standing / good form: knees track over or wider than the ankles → ratio ≤ ~1.
    Valgus: the knees pull in narrower than the ankles → ratio > 1 (higher = worse).

    Two design choices, both validated on real footage:
      • differential (both knees vs both ankles) cancels the common-mode perspective offset
        of an oblique (~30°) camera — a per-side knee-to-line deviation could not separate a
        caving rep because the offset swamped it;
      • evaluated at the *bottom* (deepest hip position), because valgus appears under load at
        depth — a max/mean over the whole rep is dominated by unrelated top-of-rep frames.
    """
    ratios, depths, view = [], [], []
    for f in pos_rep:
        lk, rk = _pos(f, "L_Knee"), _pos(f, "R_Knee")
        la, ra = _pos(f, "L_Ankle"), _pos(f, "R_Ankle")
        lh, rh = _pos(f, "L_Hip"), _pos(f, "R_Hip")
        if any(p is None for p in (lk, rk, la, ra, lh, rh)):
            continue
        knee_w = abs(rk[0] - lk[0])
        ankle_w = abs(ra[0] - la[0])
        shank = np.mean([np.linalg.norm(lk - la), np.linalg.norm(rk - ra)])
        if knee_w < 1e-6 or ankle_w < 1e-6 or shank < 1e-6:
            continue
        view.append(ankle_w / float(shank))
        ratios.append(min(ankle_w / knee_w, _VALGUS_INDEX_CAP))
        depths.append((lh[1] + rh[1]) / 2.0)     # hip height; larger y = deeper (image y grows down)
    if not ratios or float(np.median(view)) < VALGUS_MIN_VIEW_RATIO:
        return None
    ratios, depths = np.array(ratios), np.array(depths)
    deep = depths >= np.percentile(depths, 75)    # deepest quartile of the rep
    sel = ratios[deep] if deep.any() else ratios
    return float(np.median(sel))


def _spine_flex_delta(pos_rep: list[dict]) -> float | None:
    """
    Thoraco-lumbar flexion range within the rep: (mid-hip → L2 → C7) angle is ~180° with
    a neutral spine and drops as the back rounds. Returns how far below the rep's most
    upright (standing) value the spine flexes — proportion/posture-invariant (measures the
    *change*, not each athlete's absolute neutral). Higher = more rounding.
    """
    angles = []
    for f in pos_rep:
        mid = _mid_hip(f)
        l2 = _first_pos(f, _SPINE_MID)
        top = _first_pos(f, _SPINE_TOP)
        a = _joint_angle(mid, l2, top)
        if a is not None:
            angles.append(a)
    if len(angles) < 2:
        return None
    arr = np.array(angles, dtype=np.float64)
    return float(arr.max() - arr.min())


def _hip_knee_rom_ratio(angles_rep: list[dict]) -> float | None:
    """hip ROM / knee ROM. High = hip-dominant hinge (good RDL); low = knees moving (squatting an RDL)."""
    hip = [r for r in (_rom(angles_rep, "hip_L"), _rom(angles_rep, "hip_R")) if r is not None]
    knee = [r for r in (_rom(angles_rep, "knee_L"), _rom(angles_rep, "knee_R")) if r is not None]
    if not hip or not knee:
        return None
    hip_rom = float(np.mean(hip))
    knee_rom = max(float(np.mean(knee)), _KNEE_ROM_FLOOR)
    return float(min(hip_rom / knee_rom, _HIP_KNEE_RATIO_CAP))


def compute_rep_metrics(angles_rep: list[dict], pos_rep: list[dict]) -> dict[str, float | None]:
    return {
        "valgus_index": _valgus_index(pos_rep),
        "spine_flex_delta": _spine_flex_delta(pos_rep),
        "hip_knee_rom_ratio": _hip_knee_rom_ratio(angles_rep),
        **compute_coordination(angles_rep),   # hip_knee_lag_pct, marp_deg
    }
