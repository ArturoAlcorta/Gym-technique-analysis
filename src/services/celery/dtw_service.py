"""
Comparación DTW entre las repeticiones del usuario y repeticiones de referencia.

Para añadir un nuevo ejercicio:
  1. Añade una entrada en _DTW_CONFIGS con el exercise_id y los campos requeridos.
  2. Coloca los JSON de repeticiones de referencia en
     {KEYPOINTS_BASE}/{exercise_id}/*.json

Modos de features:
  "coordinates" — DTW sobre vectores 2D (x,y) normalizados por joint.
                  Requiere: relevant_joints, centering_joints, ref_distance_per_joint.
  "angles"      — DTW sobre ángulos articulares (grados). Invariante a espejo y
                  proporciones corporales. Requiere: ref_angle_per_joint.
                  Los ángulos (knee, hip, tibia) se calculan automáticamente.
"""

import json
from pathlib import Path

import numpy as np
from dtw import dtw as dtw_lib

from .angle_extraction_service import (
    ANGLE_NAMES_RDL, ANGLE_NAMES_BENCH,
    extract_angle_sequence, extract_bench_angle_sequence, get_dominant_side,
)
from .bench_symmetry_service import compute_bench_symmetry

ANGLE_NAMES_SQUAT = ["knee_angle", "hip_angle", "tibia_angle"]

SCORE_THRESHOLD = 0.5


# ── Per-exercise DTW configuration ────────────────────────────────────────────

_DTW_CONFIGS: dict[int, dict] = {
    1: {  # press banca
        "exercise_name":       "bench_press",
        "features":            "angles",
        "angle_type":          "bench_press",   # extractor bilateral (ambos brazos promediados)
        "relevant_angles":     ANGLE_NAMES_BENCH,
        "ref_angle_per_joint": 15.0,
    },
    6: {  # sentadilla
        "exercise_name":       "squat",
        "features":            "angles",
        "relevant_angles":     ANGLE_NAMES_SQUAT,   # knee, hip, tibia (lumbar excluido — T6 demasiado ruidoso)
        "ref_angle_per_joint": 15.0,
    },
    7: {  # RDL
        "exercise_name":       "rdl",
        "features":            "angles",
        "relevant_angles":     ANGLE_NAMES_RDL,   # knee, hip, lumbar (sin tibia — rodillas casi fijas)
        "ref_angle_per_joint": 15.0,
    },
}
_DTW_CONFIGS[2]  = _DTW_CONFIGS[1]   # press banca (alias)
_DTW_CONFIGS[48] = _DTW_CONFIGS[7]   # RDL (alias)


# ── Coordinate-based sequence extraction ─────────────────────────────────────

def _extract_coord_sequence(
    frames: list[dict],
    frame_start: int,
    frame_end: int,
    relevant_joints: list[str],
    centering_joints: list[str],
) -> tuple[np.ndarray, int]:
    """
    Extrae los keypoints de un rango de frames como matriz (T, J, 2)
    centrada respecto al midpoint medio de centering_joints en esa ventana.
    """
    rep_frames = sorted(
        [f for f in frames if frame_start <= f["frame_idx"] <= frame_end],
        key=lambda f: f["frame_idx"],
    )

    cx, cy = [], []
    for f in rep_frames:
        kps = f["keypoints"]
        coords = [
            [kps[j]["x"], kps[j]["y"]]
            for j in centering_joints
            if kps.get(j) and kps[j]["score"] >= SCORE_THRESHOLD
        ]
        if coords:
            cx.append(np.mean([c[0] for c in coords]))
            cy.append(np.mean([c[1] for c in coords]))
    offset = np.array([np.mean(cx) if cx else 0.0, np.mean(cy) if cy else 0.0])

    rows = []
    prev_row = None
    for f in rep_frames:
        kps = f["keypoints"]
        row = []
        for j_idx, joint in enumerate(relevant_joints):
            kp = kps.get(joint)
            if kp and kp["score"] >= SCORE_THRESHOLD:
                row.append([kp["x"] - offset[0], kp["y"] - offset[1]])
            else:
                row.append(prev_row[j_idx] if prev_row is not None else [0.0, 0.0])
        rows.append(row)
        prev_row = row

    if not rows:
        return np.zeros((1, len(relevant_joints), 2), dtype=np.float64), 0

    return np.array(rows, dtype=np.float64), len(rows)


# ── Angle-based sequence extraction ──────────────────────────────────────────

def _extract_bench_angle_sequence(
    frames: list[dict],
    frame_start: int,
    frame_end: int,
    angle_names: list[str],
) -> tuple[np.ndarray, int]:
    """Bench press: ambos brazos promediados, sin dominant_side."""
    angle_frames, n = extract_bench_angle_sequence(frames, frame_start, frame_end)
    if not angle_frames:
        return np.zeros((1, len(angle_names)), dtype=np.float64), 0
    rows = [[f[a] for a in angle_names] for f in angle_frames]
    return np.array(rows, dtype=np.float64), n


def _extract_angle_sequence(
    frames: list[dict],
    frame_start: int,
    frame_end: int,
    dominant_side: str | None,
    angle_names: list[str],
) -> tuple[np.ndarray, int]:
    """
    Extrae ángulos articulares de un rango de frames como matriz (T, A)
    donde A = len(angle_names). Invariante a espejo y proporciones corporales.
    """
    angle_frames, n = extract_angle_sequence(frames, frame_start, frame_end, dominant_side, angle_names)
    if not angle_frames:
        return np.zeros((1, len(angle_names)), dtype=np.float64), 0
    rows = [[f[a] for a in angle_names] for f in angle_frames]
    return np.array(rows, dtype=np.float64), n


# ── DTW comparison ────────────────────────────────────────────────────────────

def _compare_dtw_coords(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    relevant_joints: list[str],
    ref_distance: float,
) -> dict:
    """DTW independiente por joint sobre coordenadas (T, J, 2)."""
    joint_results: dict[str, dict] = {}
    for k, joint in enumerate(relevant_joints):
        sa = seq_a[:, k, :]
        sb = seq_b[:, k, :]
        alignment = dtw_lib(sa, sb, distance_only=False)
        dist  = float(alignment.normalizedDistance)
        score = max(0.0, round(100.0 * (1.0 - dist / ref_distance), 1))
        joint_results[joint] = {"distance": round(dist, 6), "score": score}

    distances = [v["distance"] for v in joint_results.values()]
    scores    = [v["score"]    for v in joint_results.values()]

    top_divs = sorted(
        [{"joint": j, "distance": v["distance"], "score": v["score"]} for j, v in joint_results.items()],
        key=lambda x: x["distance"],
        reverse=True,
    )[:5]

    return {
        "global_score":         round(float(np.mean(scores)), 1),
        "mean_joint_distance":  round(float(np.mean(distances)), 6),
        "max_joint_distance":   round(float(np.max(distances)), 6),
        "top_divergent_joints": top_divs,
        "score_by_joint":       {j: v["score"] for j, v in joint_results.items()},
    }


def _compare_dtw_angles(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    ref_angle: float,
    angle_names: list[str],
) -> dict:
    """DTW independiente por ángulo sobre matrices (T, A) en grados."""
    angle_results: dict[str, dict] = {}
    for k, angle_name in enumerate(angle_names):
        sa = seq_a[:, k].reshape(-1, 1)
        sb = seq_b[:, k].reshape(-1, 1)
        alignment = dtw_lib(sa, sb, distance_only=False)
        dist  = float(alignment.normalizedDistance)
        score = max(0.0, round(100.0 * (1.0 - dist / ref_angle), 1))
        angle_results[angle_name] = {"distance": round(dist, 6), "score": score}

    distances = [v["distance"] for v in angle_results.values()]
    scores    = [v["score"]    for v in angle_results.values()]

    top_divs = sorted(
        [{"joint": a, "distance": v["distance"], "score": v["score"]} for a, v in angle_results.items()],
        key=lambda x: x["distance"],
        reverse=True,
    )[:5]

    return {
        "global_score":         round(float(np.mean(scores)), 1),
        "mean_joint_distance":  round(float(np.mean(distances)), 6),
        "max_joint_distance":   round(float(np.max(distances)), 6),
        "top_divergent_joints": top_divs,
        "score_by_joint":       {a: v["score"] for a, v in angle_results.items()},
    }


# ── Public function ───────────────────────────────────────────────────────────

def analyze_dtw(
    rep_count_json_path: Path,
    normalized_json_path: Path,
    exercise_id: int,
    keypoints_base: Path,
) -> tuple[dict, Path]:
    """
    Compara cada repetición contra todas las repeticiones de referencia disponibles.
    Devuelve (result_dict, output_path).
    """
    cfg = _DTW_CONFIGS.get(exercise_id)
    if cfg is None:
        raise ValueError(
            f"exercise_id={exercise_id} no tiene config DTW. "
            f"IDs disponibles: {list(_DTW_CONFIGS)}"
        )

    ref_dir   = keypoints_base / str(exercise_id)
    ref_files = sorted(ref_dir.glob("*.json")) if ref_dir.exists() else []
    if not ref_files:
        raise FileNotFoundError(
            f"No se encontraron referencias en {ref_dir}. "
            "Añade archivos JSON de repeticiones de referencia para este ejercicio."
        )

    use_angles   = cfg.get("features", "coordinates") == "angles"
    is_bench     = cfg.get("angle_type") == "bench_press"
    angle_names  = cfg.get("relevant_angles", ANGLE_NAMES_SQUAT) if use_angles else []

    with open(rep_count_json_path, "r", encoding="utf-8") as f:
        rep_data = json.load(f)
    with open(normalized_json_path, "r", encoding="utf-8") as f:
        norm_data = json.load(f)

    user_frames = norm_data["frames"]
    reps        = rep_data["reps"]

    # Dominant side for the user video (stored in metadata or detected from frames)
    # Not needed for bench press (bilateral extraction), but computed for sagittal exercises.
    user_side = (
        norm_data.get("metadata", {}).get("normalization", {}).get("dominant_side")
        or get_dominant_side(user_frames)
    ) if not is_bench else None

    # Load all reference sequences (each file = one rep)
    ref_sequences: list[dict] = []
    for rf in ref_files:
        with open(rf, "r", encoding="utf-8") as f:
            ref_json = json.load(f)
        ref_frames = ref_json["frames"]
        fidxs = [fr["frame_idx"] for fr in ref_frames]
        if is_bench:
            seq, n_frames = _extract_bench_angle_sequence(
                ref_frames, min(fidxs), max(fidxs), angle_names
            )
            ref_side = None
        elif use_angles:
            ref_side = (
                ref_json.get("metadata", {}).get("normalization", {}).get("dominant_side")
                or get_dominant_side(ref_frames)
            )
            seq, n_frames = _extract_angle_sequence(
                ref_frames, min(fidxs), max(fidxs), ref_side, angle_names
            )
        else:
            ref_side = None
            seq, n_frames = _extract_coord_sequence(
                ref_frames, min(fidxs), max(fidxs),
                cfg["relevant_joints"], cfg["centering_joints"],
            )
        ref_sequences.append({"file": rf.name, "seq": seq, "n_frames": n_frames, "side": ref_side})

    # Compare each user rep against all references
    rep_results: list[dict] = []
    for rep in reps:
        if is_bench:
            user_seq, n_frames = _extract_bench_angle_sequence(
                user_frames, rep["frame_start"], rep["frame_end"], angle_names,
            )
        elif use_angles:
            user_seq, n_frames = _extract_angle_sequence(
                user_frames, rep["frame_start"], rep["frame_end"], user_side, angle_names,
            )
        else:
            user_seq, n_frames = _extract_coord_sequence(
                user_frames, rep["frame_start"], rep["frame_end"],
                cfg["relevant_joints"], cfg["centering_joints"],
            )

        per_ref: list[dict] = []
        for ref in ref_sequences:
            if use_angles:
                comparison = _compare_dtw_angles(user_seq, ref["seq"], cfg["ref_angle_per_joint"], angle_names)
            else:
                comparison = _compare_dtw_coords(user_seq, ref["seq"], cfg["relevant_joints"], cfg["ref_distance_per_joint"])
            per_ref.append({
                "reference_file": ref["file"],
                "n_ref_frames":   ref["n_frames"],
                **comparison,
            })

        global_scores = [r["global_score"] for r in per_ref]
        rep_entry = {
            "rep_number": rep["rep_number"],
            "n_frames":   n_frames,
            "best_score": round(max(global_scores),            1) if global_scores else None,
            "mean_score": round(float(np.mean(global_scores)), 1) if global_scores else None,
            "references": per_ref,
        }
        if is_bench:
            # Personal (reference-free) L/R symmetry review, surfaced as per-rep faults.
            sym = compute_bench_symmetry(user_frames, rep["frame_start"], rep["frame_end"])
            rep_entry["symmetry"] = sym["metrics"]
            rep_entry["faults"] = sym["faults"]
        rep_results.append(rep_entry)

    metadata: dict = {
        "exercise_id":      exercise_id,
        "exercise_name":    cfg["exercise_name"],
        "features":         cfg.get("features", "coordinates"),
        "total_reps":       len(reps),
        "total_references": len(ref_sequences),
        "reference_files":  [r["file"] for r in ref_sequences],
    }
    if use_angles:
        metadata["relevant_angles"]     = angle_names
        metadata["ref_angle_per_joint"] = cfg["ref_angle_per_joint"]
    else:
        metadata["relevant_joints"]        = cfg["relevant_joints"]
        metadata["centering_joints"]       = cfg["centering_joints"]
        metadata["ref_distance_per_joint"] = cfg["ref_distance_per_joint"]

    result = {"metadata": metadata, "reps": rep_results}

    output_path = normalized_json_path.parent / "dtw_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    best_scores = [r["best_score"] for r in rep_results if r["best_score"] is not None]
    overall = round(float(np.mean(best_scores)), 1) if best_scores else None
    print(f"DTW completado: {len(reps)} reps × {len(ref_sequences)} refs | score medio (best): {overall} → {output_path}")
    return result, output_path