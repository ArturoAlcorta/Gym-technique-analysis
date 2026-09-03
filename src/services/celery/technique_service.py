"""
Per-rep technique analysis from the lifter's own video — reference-free.

Everything computed here comes from a single submission: joint ROM, depth at the
bottom of the rep, tempo, and the three checks that have a defensible *absolute*
cutoff (knee valgus, bench L/R symmetry, and the rep counter's own bottom
confirmation). Nothing is judged against another lifter's execution; that is
exactly what the optional DTW comparison (`dtw_service`) adds on top.

Metrics without a defensible absolute cutoff (spine-flexion range, hip/knee ROM
ratio, hip-knee coordination) are still *reported* — they are informative on
their own and comparable across your own sessions — but they are not scored or
flagged here, because the only meaningful yardstick for them is a reference
population, which lives on the DTW side.

Output (one entry per rep from the 2D counter, so rep numbers and timings match
what the counter reported):

    {"metadata": {...},
     "reps": [{"rep_number": 1, "timing": {...}, "metrics": {...}, "faults": [...]}]}
"""

from pathlib import Path
import json

import numpy as np

from .angles2d_service import extract_angles
from .bench_symmetry_service import ASYM_THRESH_DEG, compute_bench_symmetry
from .feedback_service import valgus_fault
from .relational_metrics import compute_rep_metrics, valgus_cutoffs, valgus_score

# Angle channels reported (ROM + value at the bottom) per exercise. Same composite
# channels the rest of the 2D pipeline uses (angles2d_service).
ANALYSIS_CHANNELS: dict[int, list[str]] = {
    1:  ["elbow_mean", "shoulder_mean"],       # bench press
    2:  ["elbow_mean", "shoulder_mean"],       # bench press (alias)
    6:  ["knee_mean", "hip_mean", "trunk"],    # squat
    7:  ["hip_mean", "knee_mean", "trunk"],    # RDL
    48: ["hip_mean", "knee_mean", "trunk"],    # RDL (alias)
}

EXERCISE_NAMES: dict[int, str] = {
    1: "bench_press", 2: "bench_press", 6: "squat", 7: "rdl", 48: "rdl",
}

_BENCH_IDS = {1, 2}


def _rep_window(angle_frames: list[dict], frame_start: int, frame_end: int) -> tuple[int, int] | None:
    """Map a [frame_start, frame_end] rep window (rep-counter frame_idx values) to
    start/end indices into the angle frames. None if no frame falls inside."""
    idxs = [i for i, f in enumerate(angle_frames) if frame_start <= f["frame_idx"] <= frame_end]
    return (idxs[0], idxs[-1]) if idxs else None


def _series(angle_frames: list[dict], channel: str) -> np.ndarray:
    vals = [f["angles"].get(channel) for f in angle_frames]
    return np.array([v for v in vals if v is not None], dtype=np.float64)


def _rom(angle_frames: list[dict], channel: str) -> float | None:
    s = _series(angle_frames, channel)
    return round(float(s.max() - s.min()), 2) if len(s) else None


def _at_bottom(angle_frames: list[dict], channel: str, bottom_idx: int | None) -> float | None:
    """Channel value at the bottom frame; falls back to the rep's minimum."""
    if bottom_idx is not None and 0 <= bottom_idx < len(angle_frames):
        v = angle_frames[bottom_idx]["angles"].get(channel)
        if v is not None:
            return round(float(v), 2)
    s = _series(angle_frames, channel)
    return round(float(s.min()), 2) if len(s) else None


def _timing(rep: dict) -> dict:
    return {
        "start_s": rep.get("timestamp_start"),
        "end_s": rep.get("timestamp_end"),
        "total_s": rep.get("duration_total_s"),
        "eccentric_s": rep.get("duration_eccentric_s"),
        "concentric_s": rep.get("duration_concentric_s"),
    }


def _bench_metrics(frames: list[dict], rep: dict, angle_frames: list[dict],
                   bottom_idx: int | None, channels: list[str]) -> tuple[dict, list[dict]]:
    sym = compute_bench_symmetry(frames, rep["frame_start"], rep["frame_end"])
    metrics = {
        "depth_elbow_deg": rep.get("min_elbow_avg"),
        "reached_bottom": rep.get("reached_bottom"),
        **sym["metrics"],
    }
    for ch in channels:
        metrics[f"rom_{ch}_deg"] = _rom(angle_frames, ch)
        metrics[f"bottom_{ch}_deg"] = _at_bottom(angle_frames, ch, bottom_idx)
    return metrics, list(sym["faults"])


def _sagittal_metrics(rep: dict, angle_frames: list[dict], pos_frames: list[dict],
                      bottom_idx: int | None, channels: list[str],
                      exercise_id: int) -> tuple[dict, list[dict]]:
    rel = compute_rep_metrics(angle_frames, pos_frames)
    metrics = {
        "reached_bottom": rep.get("reached_bottom"),
        # Squat: knee/hip at the bottom. RDL: hip angle at the bottom + peak torso lean.
        "depth_knee_deg": rep.get("min_knee_angle"),
        "depth_hip_deg": rep.get("min_hip_angle"),
        "torso_lean_deg": rep.get("max_torso_angle"),
        "valgus_index": _round(rel.get("valgus_index"), 3),
        "spine_flex_delta_deg": _round(rel.get("spine_flex_delta"), 2),
        "hip_knee_rom_ratio": _round(rel.get("hip_knee_rom_ratio"), 2),
        "hip_knee_lag_pct": _round(rel.get("hip_knee_lag_pct"), 1),
        "marp_deg": _round(rel.get("marp_deg"), 2),
    }
    for ch in channels:
        metrics[f"rom_{ch}_deg"] = _rom(angle_frames, ch)
        metrics[f"bottom_{ch}_deg"] = _at_bottom(angle_frames, ch, bottom_idx)

    faults: list[dict] = []
    vi = rel.get("valgus_index")
    if vi is not None and float(vi) > valgus_cutoffs(exercise_id)[0]:
        faults.append(valgus_fault(float(vi), valgus_score(float(vi), exercise_id)))
    return metrics, faults


def _round(value, digits: int):
    return round(float(value), digits) if value is not None else None


def analyze_technique(normalized: dict, reps: list[dict], exercise_id: int) -> dict:
    """
    Reference-free per-rep analysis.

    normalized : normalized keypoints dict (normalization_service output).
    reps       : rep windows from the 2D counter (rep_counting_service), so rep
                 numbers/timings line up with everything else the app shows.
    """
    channels = ANALYSIS_CHANNELS.get(exercise_id)
    if channels is None:
        raise ValueError(
            f"exercise_id={exercise_id} has no technique-analysis config. "
            f"Available IDs: {sorted(ANALYSIS_CHANNELS)}"
        )

    angles = extract_angles(normalized)
    frames = normalized["frames"]
    is_bench = exercise_id in _BENCH_IDS

    rep_results: list[dict] = []
    for rep in reps:
        win = _rep_window(angles["frames"], rep["frame_start"], rep["frame_end"])
        if win is None:
            continue
        idx_start, idx_end = win
        angle_frames = angles["frames"][idx_start:idx_end + 1]
        pos_frames = frames[idx_start:idx_end + 1]

        bottom_idx = None
        if rep.get("frame_bottom") is not None:
            for i, f in enumerate(angle_frames):
                if f["frame_idx"] == rep["frame_bottom"]:
                    bottom_idx = i
                    break

        if is_bench:
            metrics, faults = _bench_metrics(frames, rep, angle_frames, bottom_idx, channels)
        else:
            metrics, faults = _sagittal_metrics(rep, angle_frames, pos_frames, bottom_idx, channels, exercise_id)

        rep_results.append({
            "rep_number": rep["rep_number"],
            "frame_start": rep["frame_start"],
            "frame_bottom": rep.get("frame_bottom"),
            "frame_end": rep["frame_end"],
            "n_frames": len(angle_frames),
            "timing": _timing(rep),
            "metrics": metrics,
            "faults": faults,
        })

    return {
        "metadata": {
            "exercise_id": exercise_id,
            "exercise_name": EXERCISE_NAMES.get(exercise_id, "unknown"),
            "channels": channels,
            "total_reps": len(rep_results),
            "fps": normalized.get("metadata", {}).get("fps"),
            "thresholds": {
                "valgus_flag": valgus_cutoffs(exercise_id)[0],
                "asymmetry_deg": ASYM_THRESH_DEG,
            },
        },
        "reps": rep_results,
    }


def analyze_technique_files(normalized_json_path: Path, rep_count_json_path: Path,
                            exercise_id: int, output_path: Path | None = None) -> tuple[dict, Path]:
    """File-in/file-out wrapper, mirroring the other services in this package."""
    normalized_json_path = Path(normalized_json_path)
    with open(normalized_json_path, "r", encoding="utf-8") as f:
        normalized = json.load(f)
    with open(rep_count_json_path, "r", encoding="utf-8") as f:
        reps = json.load(f).get("reps", [])

    result = analyze_technique(normalized, reps, exercise_id)
    output_path = Path(output_path) if output_path else normalized_json_path.parent / "technique.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Technique analysis → {output_path}  ({result['metadata']['total_reps']} reps)")
    return result, output_path
