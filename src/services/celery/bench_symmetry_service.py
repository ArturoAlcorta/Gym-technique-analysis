"""
Bench press L/R symmetry analysis — reference-free personal review.

Computes per-rep symmetry metrics from per-side angle sequences and emits faults
when asymmetry exceeds thresholds. Intended as a personal check (not vs references),
surfaced in the UI as per-rep warnings.
"""

import numpy as np

from .angle_extraction_service import BENCH_SIDE_ANGLES, extract_bench_side_sequences


ASYM_THRESH_DEG = 15.0  # degrees — absolute L/R difference above which asymmetry is flagged
ASYM_ZERO_DEG = 30.0    # degrees — at/above this the symmetry sub-score is 0


def symmetry_score(avg_asymmetry_deg: float) -> float:
    """0-100 sub-score for a mean L/R angle difference: a flat 100 up to the flag
    threshold, decaying linearly to 0 at twice it. Bench has no reference band,
    so this absolute cutoff is what its metric score is built from."""
    if avg_asymmetry_deg <= ASYM_THRESH_DEG:
        return 100.0
    return float(max(0.0, 100.0 * (ASYM_ZERO_DEG - avg_asymmetry_deg) / (ASYM_ZERO_DEG - ASYM_THRESH_DEG)))


def _asymmetry_at_bottom(angle_seq: np.ndarray) -> float | None:
    """Abs difference at the bottom frame (min elbow angle)."""
    if len(angle_seq) < 3:
        return None
    bottom_idx = int(np.argmin(angle_seq))
    return None  # We need left/right per-frame, not averaged


def _max_asymmetry(l_seq: np.ndarray, r_seq: np.ndarray) -> float:
    """Max absolute difference between L and R across the rep."""
    if len(l_seq) != len(r_seq):
        return 0.0
    return float(np.max(np.abs(l_seq - r_seq)))


def _avg_asymmetry(l_seq: np.ndarray, r_seq: np.ndarray) -> float:
    """Mean absolute difference between L and R across the rep."""
    if len(l_seq) != len(r_seq):
        return 0.0
    return float(np.mean(np.abs(l_seq - r_seq)))


def compute_bench_symmetry(
    frames: list[dict],
    frame_start: int,
    frame_end: int,
) -> dict:
    """
    Compute L/R symmetry metrics for a bench press rep.
    Returns dict with metrics and faults for the UI.
    """
    side_angles = extract_bench_side_sequences(frames, frame_start, frame_end)

    elbow_L = np.array(side_angles["elbow_L"])
    elbow_R = np.array(side_angles["elbow_R"])
    shoulder_L = np.array(side_angles["shoulder_L"])
    shoulder_R = np.array(side_angles["shoulder_R"])

    # Max and average asymmetry
    elbow_max_asym = _max_asymmetry(elbow_L, elbow_R)
    elbow_avg_asym = _avg_asymmetry(elbow_L, elbow_R)
    shoulder_max_asym = _max_asymmetry(shoulder_L, shoulder_R)
    shoulder_avg_asym = _avg_asymmetry(shoulder_L, shoulder_R)

    # Overall flags
    elbow_flag = elbow_avg_asym > ASYM_THRESH_DEG
    shoulder_flag = shoulder_avg_asym > ASYM_THRESH_DEG

    faults = []
    if elbow_flag:
        faults.append({
            "code": "elbow_asymmetry",
            "severity": "moderate" if elbow_avg_asym < 25 else "major",
            "cue_en": f"Elbow asymmetry detected ({elbow_avg_asym:.1f}° avg difference). Keep elbows moving symmetrically.",
            "cue_es": f"Asimetría de codos detectada ({elbow_avg_asym:.1f}° de diferencia media). Mantén los codos moviéndose simétricamente.",
        })
    if shoulder_flag:
        faults.append({
            "code": "shoulder_asymmetry",
            "severity": "moderate" if shoulder_avg_asym < 25 else "major",
            "cue_en": f"Shoulder asymmetry detected ({shoulder_avg_asym:.1f}° avg difference). Check for uneven bar path.",
            "cue_es": f"Asimetría de hombros detectada ({shoulder_avg_asym:.1f}° de diferencia media). Revisa la trayectoria desigual de la barra.",
        })

    return {
        "metrics": {
            "elbow_max_asymmetry_deg": round(elbow_max_asym, 2),
            "elbow_avg_asymmetry_deg": round(elbow_avg_asym, 2),
            "shoulder_max_asymmetry_deg": round(shoulder_max_asym, 2),
            "shoulder_avg_asymmetry_deg": round(shoulder_avg_asym, 2),
            "elbow_asymmetry_flag": elbow_flag,
            "shoulder_asymmetry_flag": shoulder_flag,
        },
        "faults": faults,
    }