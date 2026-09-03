"""
Human-readable fault cues (English + Spanish) for the technique feedback.

Cue text lives here rather than next to the metric that raises it, so the
wording of what a lifter is told stays in one place. The thresholds that decide
*whether* a cue is raised stay with the metric (see `relational_metrics` and
`bench_symmetry_service`).
"""

from typing import Any


def valgus_fault(vi: float, score: float) -> dict[str, Any]:
    """Knee-cave cue. Severity follows the sub-score rather than the raw index, so
    it tracks whichever per-exercise cutoff the index was judged against."""
    severity = "moderate" if score >= 50.0 else "major"
    return {
        "code": "knee_valgus",
        "severity": severity,
        "cue_en": f"Knees caving inward (valgus index: {vi:.2f}). Push knees out over toes.",
        "cue_es": f"Rodillas colapsando hacia adentro (índice valgus: {vi:.2f}). Empuja las rodillas hacia afuera sobre los dedos.",
    }
