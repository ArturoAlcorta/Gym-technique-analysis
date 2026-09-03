"""
2D joint-angle features from normalized SynthPose keypoints (squat / RDL).

Replaces the 3D lift for squat/RDL scoring: the 2D→3D lift shortened/lengthened limbs
frame-to-frame (model artifact, not fixable), so we score directly on the 2D keypoints.
The filming angle moved from pure-sagittal to ~30° from the front/back, so **both sides
are visible** — we can compute per-side channels (knee_L/R, hip_L/R), unlike the old
single-dominant-side 2D path.

Output schema matches what the downstream scorer, segmentation, coordination and reference
builder expect, so those stages are reused unchanged:
    {metadata, frames: [{frame_idx, timestamp_s, angles: {channel: deg}}]}

Channels (same names/order as the 3D service so DTW_CHANNELS / segmentation match):
  knee_L/R : Hip–Knee–Ankle
  hip_L/R  : C7–Hip–Knee            (trunk-to-thigh angle)
  trunk    : (C7 − mid-hip)  vs up
Plus knee_mean / hip_mean composite signals used by the rep segmentation.

Bench press is not a sagittal lift and takes the separate branch at the bottom of
this module: its channels are elbow_mean / shoulder_mean, from the bilateral
extractor in `angle_extraction_service`.

"vs up": the up axis is estimated from the athlete's most-extended (standing) frames
(median mid-hip→C7 direction) rather than assuming image-vertical, which cancels most of
the camera-tilt bias as long as user and reference are filmed similarly (same rationale as
the 3D gravity_up estimate).
"""

import json
import math
from pathlib import Path

import numpy as np

from .angle_extraction_service import angles_for_bench_frame

ANGLE_NAMES = ["knee_L", "knee_R", "hip_L", "hip_R", "trunk"]
ANGLE_NAMES_BENCH = ["elbow_mean", "shoulder_mean"]

# Exercise names, as the normalizer writes them into the keypoints metadata.
_SAGITTAL = ("sagittal", "squat", "rdl")

# Central upper-spine point for trunk / hip angles (C7 preferred, T6 fallback if occluded).
_SPINE_TOP = ["C7", "T6"]


def _pt(kps: dict, name: str) -> np.ndarray | None:
    """2D point for a normalized keypoint, or None if missing/low-confidence."""
    k = kps.get(name)
    if k is None:
        return None
    return np.array([k["x"], k["y"]], dtype=np.float64)


def _first(kps: dict, names: list[str]) -> np.ndarray | None:
    for n in names:
        p = _pt(kps, n)
        if p is not None:
            return p
    return None


def _mid_hip(kps: dict) -> np.ndarray | None:
    l, r = _pt(kps, "L_Hip"), _pt(kps, "R_Hip")
    pair = [p for p in (l, r) if p is not None]
    return np.mean(pair, axis=0) if pair else None


def _joint_angle(a, b, c):
    """Angle at b formed by a-b-c, degrees. 180° = straight."""
    if a is None or b is None or c is None:
        return None
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    cos = np.clip(float(np.dot(v1, v2)) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos))


def _angle_to_axis(vec, axis):
    """Angle between `vec` and `axis` (both 2-vectors), degrees."""
    if vec is None:
        return None
    n = np.linalg.norm(vec)
    if n < 1e-9:
        return None
    cos = np.clip(float(np.dot(vec, axis)) / n, -1.0, 1.0)
    return math.degrees(math.acos(cos))


def estimate_up(frames: list[dict]) -> np.ndarray:
    """
    Up axis from the most-extended (standing) frames: median, over the top-quartile of
    knee extension, of the normalized mid-hip→C7 vector. Falls back to image-up (0,-1).
    """
    knee, trunk_vecs = [], []
    for f in frames:
        kps = f["keypoints"]
        kl = _joint_angle(_pt(kps, "L_Hip"), _pt(kps, "L_Knee"), _pt(kps, "L_Ankle"))
        kr = _joint_angle(_pt(kps, "R_Hip"), _pt(kps, "R_Knee"), _pt(kps, "R_Ankle"))
        vals = [v for v in (kl, kr) if v is not None]
        knee.append(np.mean(vals) if vals else np.nan)
        top, mid = _first(kps, _SPINE_TOP), _mid_hip(kps)
        trunk_vecs.append((top - mid) if (top is not None and mid is not None) else np.array([np.nan, np.nan]))

    knee = np.array(knee)
    trunk_vecs = np.array(trunk_vecs, dtype=np.float64)
    valid = ~np.isnan(knee) & ~np.isnan(trunk_vecs[:, 0])
    if valid.sum() == 0:
        return np.array([0.0, -1.0])
    thr = np.nanpercentile(knee[valid], 75)
    sel = valid & (knee >= thr)
    if sel.sum() == 0:
        sel = valid
    v = np.median(trunk_vecs[sel], axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([0.0, -1.0])


def angles_for_frame(kps: dict, up: np.ndarray) -> dict[str, float | None]:
    L_Hip, L_Knee, L_Ankle = _pt(kps, "L_Hip"), _pt(kps, "L_Knee"), _pt(kps, "L_Ankle")
    R_Hip, R_Knee, R_Ankle = _pt(kps, "R_Hip"), _pt(kps, "R_Knee"), _pt(kps, "R_Ankle")
    top, mid = _first(kps, _SPINE_TOP), _mid_hip(kps)
    return {
        "knee_L": _joint_angle(L_Hip, L_Knee, L_Ankle),
        "knee_R": _joint_angle(R_Hip, R_Knee, R_Ankle),
        "hip_L":  _joint_angle(top, L_Hip, L_Knee),
        "hip_R":  _joint_angle(top, R_Hip, R_Knee),
        "trunk":  _angle_to_axis((top - mid) if (top is not None and mid is not None) else None, up),
    }


def _bench_series(frames: list[dict]) -> list[dict]:
    """Bench press channels: both arms averaged, forward-filled like the sagittal path."""
    series, prev = [], {a: None for a in ANGLE_NAMES_BENCH}
    for f in frames:
        raw = angles_for_bench_frame(f["keypoints"])
        row = {}
        for name, source in (("elbow_mean", "elbow_angle"), ("shoulder_mean", "shoulder_angle")):
            v = raw.get(source)
            v = v if v is not None else prev[name]
            row[name] = float(v) if v is not None else None
            prev[name] = row[name]
        series.append(row)
    return series


def extract_angles(normalized: dict) -> dict:
    """
    Per-frame 2D angle series + primary signals from a normalized 2D keypoints dict.
    One output row per input frame (index-aligned with the keypoints, which double as the
    position source for valgus/spine metrics). Missing angles are forward-filled.
    """
    frames = normalized["frames"]
    meta = normalized.get("metadata", {})
    if meta.get("exercise", "sagittal") not in _SAGITTAL:
        series = _bench_series(frames)
        return {
            "metadata": {"source_video": meta.get("source_video"), "fps": meta.get("fps", 15.0),
                         "total_frames": len(series), "angle_names": ANGLE_NAMES_BENCH},
            "frames": [{"frame_idx": f["frame_idx"], "timestamp_s": f.get("timestamp_s"), "angles": row}
                       for f, row in zip(frames, series)],
        }

    up = estimate_up(frames)

    series: list[dict] = []
    prev: dict[str, float | None] = {a: None for a in ANGLE_NAMES}
    for f in frames:
        raw = angles_for_frame(f["keypoints"], up)
        row = {}
        for a in ANGLE_NAMES:
            v = raw[a] if raw[a] is not None else prev[a]
            row[a] = float(v) if v is not None else None
            prev[a] = row[a]
        kn = [row["knee_L"], row["knee_R"]]
        hp = [row["hip_L"], row["hip_R"]]
        row["knee_mean"] = float(np.nanmean([v for v in kn if v is not None])) if any(v is not None for v in kn) else None
        row["hip_mean"]  = float(np.nanmean([v for v in hp if v is not None])) if any(v is not None for v in hp) else None
        series.append(row)

    meta = normalized.get("metadata", {})
    return {
        "metadata": {
            "source_video": meta.get("source_video"),
            "fps": meta.get("fps", 15.0),
            "total_frames": len(series),
            "up_axis": [round(float(c), 6) for c in up],
            "angle_names": ANGLE_NAMES,
        },
        "frames": [
            {"frame_idx": f["frame_idx"], "timestamp_s": f.get("timestamp_s"), "angles": row}
            for f, row in zip(frames, series)
        ],
    }


def extract_angles_file(normalized_json_path: Path, output_path: Path | None = None) -> tuple[dict, Path]:
    normalized_json_path = Path(normalized_json_path)
    with open(normalized_json_path, "r", encoding="utf-8") as f:
        normalized = json.load(f)
    result = extract_angles(normalized)
    output_path = output_path or normalized_json_path.parent / "angles2d.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"✅ 2D angles → {output_path}")
    return result, output_path
