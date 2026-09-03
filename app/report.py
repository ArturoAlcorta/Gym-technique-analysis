"""Turns the engine's JSON artifacts into the shape the frontend renders.

Three inputs, two of them optional:
  technique.json   — always produced: per-rep, reference-free measurements.
  dtw_result.json  — only when the analysis was run with the comparison enabled;
                     supplies the movement-pattern half of each rep's score.
  band             — the reference range for this exercise's relational metrics
                     (`reference_band_service`); absent for bench press, which
                     scores its relational half against an absolute cutoff.

Scoring itself lives in the engine (`scoring_service`); what happens here is the
mapping from metric keys to labels, units and badges. Keeping that split means
the engine never learns about the UI, and the UI never re-implements a threshold.
"""

from typing import Any

from app.exercises import JOINT_LABELS, Exercise, status_for
from services.celery.scoring_service import score_rep


def _display(value: Any, unit: str, digits: int) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{value:.{digits}f}{unit}"


def _metric_rows(metrics: dict, sub_scores: dict, exercise: Exercise) -> list[dict]:
    rows = []
    for spec in exercise.metrics:
        if spec.key not in metrics:
            continue
        value = metrics[spec.key]
        score = sub_scores.get(spec.key)
        rows.append({
            "key": spec.key,
            "label": spec.label,
            "value": value,
            "display": _display(value, spec.unit, spec.digits),
            "score": score,
            "status": status_for(value, score),
            "hint": spec.hint,
        })
    return rows


def _comparison(dtw_rep: dict | None) -> dict | None:
    if dtw_rep is None:
        return None
    refs = dtw_rep.get("references", [])
    best = max(refs, key=lambda r: r["global_score"]) if refs else None
    by_joint = []
    if best:
        for joint, score in best.get("score_by_joint", {}).items():
            by_joint.append({"joint": joint, "label": JOINT_LABELS.get(joint, joint), "score": score})
        by_joint.sort(key=lambda j: j["score"])
    return {
        "best_score": dtw_rep.get("best_score"),
        "mean_score": dtw_rep.get("mean_score"),
        "best_reference": best["reference_file"] if best else None,
        "n_references": len(refs),
        "by_joint": by_joint,
    }


def _faults(rep: dict, dtw_rep: dict | None) -> list[dict]:
    """Reference-free faults, plus the ones the bench DTW path attaches (symmetry)."""
    seen, out = set(), []
    for f in list(rep.get("faults", [])) + list((dtw_rep or {}).get("faults", [])):
        if f["code"] in seen:
            continue
        seen.add(f["code"])
        out.append({"code": f["code"], "severity": f.get("severity", "moderate"),
                    "cue": f.get("cue_en") or f.get("cue_es", "")})
    return out


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def build_report(technique: dict, dtw: dict | None, exercise: Exercise, band: dict | None = None) -> dict:
    dtw_by_rep = {r["rep_number"]: r for r in (dtw or {}).get("reps", [])}

    reps = []
    for rep in technique["reps"]:
        dtw_rep = dtw_by_rep.get(rep["rep_number"])
        comparison = _comparison(dtw_rep)
        scored = score_rep(rep["metrics"], band, comparison["best_score"] if comparison else None,
                           exercise.id)
        reps.append({
            "rep_number": rep["rep_number"],
            "timing": rep["timing"],
            "score": scored["combined_score"],
            "metric_score": scored["metric_score"],
            "pattern_score": scored["pattern_score"],
            "metrics": _metric_rows(rep["metrics"], scored["sub_scores"], exercise),
            "comparison": comparison,
            "faults": _faults(rep, dtw_rep),
        })

    reference = None
    if dtw:
        reference = {
            "files": dtw["metadata"].get("reference_files", []),
            "n": dtw["metadata"].get("total_references"),
            "angles": dtw["metadata"].get("relevant_angles", []),
        }

    return {
        "exercise": {"slug": exercise.slug, "name": exercise.name, "id": exercise.id},
        "compare": dtw is not None,
        "total_reps": len(reps),
        "summary": {
            "score": _mean([r["score"] for r in reps]),
            "metric_score": _mean([r["metric_score"] for r in reps]),
            "pattern_score": _mean([r["pattern_score"] for r in reps]),
            "avg_total_s": _mean([r["timing"]["total_s"] for r in reps]),
            "avg_eccentric_s": _mean([r["timing"]["eccentric_s"] for r in reps]),
            "avg_concentric_s": _mean([r["timing"]["concentric_s"] for r in reps]),
            "n_faults": sum(len(r["faults"]) for r in reps),
        },
        "reference": reference,
        "reps": reps,
    }
