"""
Extracción de ángulos articulares desde keypoints normalizados (plano sagital).

Cada ángulo es el ángulo articular real (3 puntos), en grados.
Son invariantes a escala, translación y espejo — no requieren direction_sign.

Ángulos para sentadilla:
  knee_angle   : tobillo → rodilla → cadera    (ángulo en la rodilla)
  hip_angle    : rodilla → cadera  → L2        (ángulo en la cadera)
  tibia_angle  : rodilla → tobillo → dedo pie  (ángulo en el tobillo)
  lumbar_angle : cadera  → L2     → T6/T11     (ángulo en L2; T11 como fallback si T6 ocluido)

Ángulos para RDL:
  knee_angle, hip_angle, lumbar_angle

Ángulos para banca:
  elbow_angle   : hombro → codo   → muñeca  (ángulo en el codo)
  shoulder_angle : cadera → hombro → codo   (ángulo en el hombro)
"""

import math
import statistics

SCORE_THRESHOLD = 0.5

_JOINTS = {
    "left": {
        "hip":     ["L_Hip", "l_ASIS"],
        "knee":    ["L_Knee", "l_knee"],
        "ankle":   ["L_Ankle", "l_ankle"],
        "big_toe": ["l_big_toe"],
    },
    "right": {
        "hip":     ["R_Hip", "r_ASIS"],
        "knee":    ["R_Knee", "r_knee"],
        "ankle":   ["R_Ankle", "r_ankle"],
        "big_toe": ["r_big_toe"],
    },
}

_L2  = "L2"
_T6  = "T6"
_T11 = "T11"  # fallback when T6 is occluded (e.g. by a barbell disc)


def _best(kps: dict, names: list[str]) -> dict | None:
    """First listed candidate that clears the confidence threshold.

    A fallback chain, deliberately *not* a confidence contest. The candidates are
    different anatomical landmarks, not competing estimates of one: SynthPose's
    first 17 keypoints are COCO (`L_Hip` is the hip joint centre, id 11) and the
    remaining 35 are anatomical markers (`l_ASIS`, id 29, is the iliac spine,
    several centimetres forward and above it). Taking whichever scored higher on
    a given frame swapped the measurement mid-rep — on real footage the left
    "hip" alternated between the two across a single squat and put 47° steps into
    the hip-angle series. It went unnoticed while clips were filmed side-on,
    where the two landmarks project almost on top of each other; at the 30°
    filming angle they separate laterally and the swap becomes large.

    So: COCO first (same source as the shoulder, knee and ankle points), marker
    as the fallback for when the COCO one is occluded."""
    for name in names:
        kp = kps.get(name)
        if kp is not None and kp["score"] >= SCORE_THRESHOLD:
            return kp
    return None


def _kp(kps: dict, name: str) -> dict | None:
    p = kps.get(name)
    return p if p is not None and p["score"] >= SCORE_THRESHOLD else None


def _joint_angle(p_prox: dict | None, p_joint: dict | None, p_dist: dict | None) -> float | None:
    """
    Joint angle at p_joint (degrees) — angle between vectors p_joint→p_prox and p_joint→p_dist.
    180° = fully extended, 90° = right angle. Invariant to mirror and camera tilt.
    """
    if p_prox is None or p_joint is None or p_dist is None:
        return None
    v1 = (p_prox["x"] - p_joint["x"], p_prox["y"] - p_joint["y"])
    v2 = (p_dist["x"] - p_joint["x"], p_dist["y"] - p_joint["y"])
    mag = math.sqrt(v1[0] ** 2 + v1[1] ** 2) * math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if mag < 1e-9:
        return None
    cos_a = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / mag))
    return math.degrees(math.acos(cos_a))


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _shank_lengths(frames: list[dict], side: str) -> list[float]:
    """Projected knee→ankle distance per frame, for frames where both are confident."""
    out = []
    for f in frames:
        kps = f.get("keypoints", {})
        knee, ankle = kps.get(f"{side}_Knee"), kps.get(f"{side}_Ankle")
        if knee is None or ankle is None:
            continue
        if knee["score"] < SCORE_THRESHOLD or ankle["score"] < SCORE_THRESHOLD:
            continue
        out.append(math.hypot(knee["x"] - ankle["x"], knee["y"] - ankle["y"]))
    return out


def get_dominant_side(frames: list[dict]) -> str:
    """
    Which leg the camera actually sees, from the projected length of the shank.

    The near leg is closer to the lens, so perspective renders it larger; the far
    leg is both smaller and partly occluded by it. Measured over the clip, the
    median knee→ankle distance separates the two by 12–20% on real footage, and
    the shank is the segment to use for it: it holds its orientation through the
    whole movement in all three lifts, so its projected length reflects distance
    to the camera and little else. The femur does not — in an RDL the thighs stay
    near vertical and the hips travel back, and the two femurs came out 0.6%
    apart, which is a coin flip. The trunk is worse still (0.6–6.4%), being close
    to the midline where perspective barely separates the sides.

    This replaced a comparison of mean hip-keypoint *confidence*, which is not a
    proxy for visibility: on one squat it scored the occluded leg 10.6% higher
    than the visible one, and because confidence is averaged over whatever frames
    it is handed, the same clip answered "left" over the whole video and "right"
    over five of its six individual reps — so a user's rep and a reference rep cut
    from the same footage were compared leg-against-leg. Matching sides is what
    matters here, more than which side wins: with both on the same leg a rep
    scores 100 against itself, and cross-matched it scores 68.

    Ties and missing data fall back to the left, arbitrarily but deterministically.
    """
    left = _median(_shank_lengths(frames, "L"))
    right = _median(_shank_lengths(frames, "R"))
    return "left" if left >= right else "right"


def angles_for_frame(kps: dict, side: str) -> dict[str, float | None]:
    joints  = _JOINTS[side]
    hip     = _best(kps, joints["hip"])
    knee    = _best(kps, joints["knee"])
    ankle   = _best(kps, joints["ankle"])
    big_toe = _best(kps, joints["big_toe"])

    L2        = _kp(kps, _L2)
    T6        = _kp(kps, _T6)
    T11       = _kp(kps, _T11)
    spine_top = T6 if T6 is not None else T11

    return {
        "knee_angle":   _joint_angle(ankle, knee,  hip),
        "hip_angle":    _joint_angle(knee,  hip,   L2),
        "tibia_angle":  _joint_angle(knee,  ankle, big_toe),
        "lumbar_angle": _joint_angle(hip,   L2,    spine_top),
    }


ANGLE_NAMES     = ["knee_angle", "hip_angle", "tibia_angle", "lumbar_angle"]
ANGLE_NAMES_RDL = ["knee_angle", "hip_angle", "lumbar_angle"]


def extract_angle_sequence(
    frames: list[dict],
    frame_start: int,
    frame_end: int,
    dominant_side: str | None = None,
    angle_names: list[str] | None = None,
) -> tuple[list[dict[str, float]], int]:
    """
    Extract per-frame angle dicts for a rep window.

    Returns (angle_frames, n_frames) where angle_frames is a list of dicts
    {angle_name: degrees}, one per frame. Missing values are forward-filled;
    if still None after fill, defaults to 0.0.

    angle_names: subset of ANGLE_NAMES to compute. Defaults to all (ANGLE_NAMES).
    """
    if dominant_side is None:
        dominant_side = get_dominant_side(frames)
    if angle_names is None:
        angle_names = ANGLE_NAMES

    rep_frames = sorted(
        [f for f in frames if frame_start <= f["frame_idx"] <= frame_end],
        key=lambda f: f["frame_idx"],
    )

    result: list[dict[str, float]] = []
    prev: dict[str, float | None] = {a: None for a in angle_names}

    for f in rep_frames:
        raw = angles_for_frame(f["keypoints"], dominant_side)
        frame_angles: dict[str, float] = {}
        for name in angle_names:
            val = raw[name]
            if val is None:
                val = prev[name]
            frame_angles[name] = float(val) if val is not None else 0.0
            prev[name] = frame_angles[name]
        result.append(frame_angles)

    return result, len(result)


# ── Bench Press ───────────────────────────────────────────────────────────────

BENCH_JOINTS = {
    "left": {
        "hip":      ["L_Hip", "l_ASIS"],
        "shoulder": ["L_Shoulder", "lshoulder"],
        "elbow":    ["L_Elbow", "l_lelbow", "l_melbow"],
        "wrist":    ["L_Wrist", "l_lwrist", "l_mwrist"],
    },
    "right": {
        "hip":      ["R_Hip", "r_ASIS"],
        "shoulder": ["R_Shoulder", "rshoulder"],
        "elbow":    ["R_Elbow", "r_lelbow", "r_melbow"],
        "wrist":    ["R_Wrist", "r_lwrist", "r_mwrist"],
    },
}

ANGLE_NAMES_BENCH = ["elbow_angle", "shoulder_angle"]


def _bench_angles_one_side(kps: dict, side: str) -> dict[str, float | None]:
    joints   = BENCH_JOINTS[side]
    hip      = _best(kps, joints["hip"])
    shoulder = _best(kps, joints["shoulder"])
    elbow    = _best(kps, joints["elbow"])
    wrist    = _best(kps, joints["wrist"])
    return {
        "elbow_angle":    _joint_angle(shoulder, elbow, wrist),
        "shoulder_angle": _joint_angle(hip, shoulder, elbow),
    }


def angles_for_bench_frame(kps: dict) -> dict[str, float | None]:
    """Compute bench press angles by averaging both sides — invariant to filming diagonal."""
    left  = _bench_angles_one_side(kps, "left")
    right = _bench_angles_one_side(kps, "right")
    result: dict[str, float | None] = {}
    for name in ANGLE_NAMES_BENCH:
        lv, rv = left[name], right[name]
        if lv is not None and rv is not None:
            result[name] = (lv + rv) / 2.0
        else:
            result[name] = lv if lv is not None else rv
    return result


BENCH_SIDE_ANGLES = ["elbow_L", "elbow_R", "shoulder_L", "shoulder_R"]


def extract_bench_side_sequences(
    frames: list[dict],
    frame_start: int,
    frame_end: int,
) -> dict[str, list[float]]:
    """
    Per-side bench angle series for a rep window (left/right kept separate, NOT averaged)
    — the input to the L/R symmetry check. Missing values forward-filled; 0.0 if never seen.
    Returns {elbow_L, elbow_R, shoulder_L, shoulder_R} → list[float].
    """
    rep_frames = sorted(
        [f for f in frames if frame_start <= f["frame_idx"] <= frame_end],
        key=lambda f: f["frame_idx"],
    )
    out: dict[str, list[float]] = {a: [] for a in BENCH_SIDE_ANGLES}
    prev: dict[str, float | None] = {a: None for a in BENCH_SIDE_ANGLES}
    for f in rep_frames:
        left = _bench_angles_one_side(f["keypoints"], "left")
        right = _bench_angles_one_side(f["keypoints"], "right")
        raw = {
            "elbow_L": left["elbow_angle"], "elbow_R": right["elbow_angle"],
            "shoulder_L": left["shoulder_angle"], "shoulder_R": right["shoulder_angle"],
        }
        for a in BENCH_SIDE_ANGLES:
            v = raw[a] if raw[a] is not None else prev[a]
            out[a].append(float(v) if v is not None else 0.0)
            prev[a] = out[a][-1]
    return out


def extract_bench_angle_sequence(
    frames: list[dict],
    frame_start: int,
    frame_end: int,
) -> tuple[list[dict[str, float]], int]:
    """
    Extract bench press angle sequence (both arms averaged, no dominant_side needed).
    Missing values are forward-filled; defaults to 0.0 if never seen.
    """
    rep_frames = sorted(
        [f for f in frames if frame_start <= f["frame_idx"] <= frame_end],
        key=lambda f: f["frame_idx"],
    )
    result: list[dict[str, float]] = []
    prev: dict[str, float | None] = {a: None for a in ANGLE_NAMES_BENCH}

    for f in rep_frames:
        raw = angles_for_bench_frame(f["keypoints"])
        frame_angles: dict[str, float] = {}
        for name in ANGLE_NAMES_BENCH:
            val = raw[name]
            if val is None:
                val = prev[name]
            frame_angles[name] = float(val) if val is not None else 0.0
            prev[name] = frame_angles[name]
        result.append(frame_angles)

    return result, len(result)