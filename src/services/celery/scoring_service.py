"""
Per-rep technique score: half movement pattern, half relational metrics.

    combined = 0.5 · pattern + 0.5 · metrics

**Pattern** is the DTW comparison against the reference reps (`dtw_service`) —
how closely the shape of the movement matches someone doing it well. It only
exists when the analysis was run with the comparison enabled.

**Metrics** is the mean of the per-metric sub-scores, and needs no DTW at all,
which is the point of the split: an analysis run without the comparison still
gets a score, just built from the relational half alone. Two kinds of sub-score
feed it:

  • *banded* — spine flexion, hip/knee ROM ratio and the two coordination
    measures are scored against the pooled acceptable range of the reference
    reps (`reference_band_service`). Deviation is one-sided for metrics where
    only one direction is a fault (spine flexion), two-sided otherwise.
  • *absolute* — knee valgus and bench L/R symmetry have a defensible fixed
    cutoff and are scored against it directly, no band involved.

With only one half available the score is that half, rather than a combined
number that silently means something different from rep to rep.
"""

import numpy as np

from .bench_symmetry_service import symmetry_score
from .relational_metrics import valgus_score

PATTERN_WEIGHT = 0.5

# Metric key as `technique_service` reports it → key as the band stores it.
_BANDED = {
    "spine_flex_delta_deg": "spine_flex_delta",
    "hip_knee_rom_ratio": "hip_knee_rom_ratio",
    "hip_knee_lag_pct": "hip_knee_lag_pct",
    "marp_deg": "marp_deg",
}
# Metrics where only the high side is a fault; the rest are penalised either way.
_HIGHER_IS_WORSE = {"spine_flex_delta"}

# Scored against a fixed cutoff rather than a band. Knee valgus takes the exercise,
# because the neutral ankle/knee ratio differs between a squat and a hinge.
_ABSOLUTE = {
    "valgus_index": lambda v, exercise_id: valgus_score(v, exercise_id),
    "elbow_avg_asymmetry_deg": lambda v, exercise_id: symmetry_score(v),
    "shoulder_avg_asymmetry_deg": lambda v, exercise_id: symmetry_score(v),
}


def _band_penalty(value: float, rng: dict, higher_is_worse: bool) -> float:
    """0 inside the acceptable range, rising to 1 a half-width outside it."""
    low, high, mean = rng["low"], rng["high"], rng["mean"]
    if higher_is_worse:
        deviation = max(0.0, value - high)
        halfwidth = max(high - mean, 1e-6)
    else:
        deviation = min(0.0, value - low) if value < low else max(0.0, value - high)
        halfwidth = max(high - mean, mean - low, 1e-6)
    return min(abs(deviation) / halfwidth, 1.0)


def sub_scores(metrics: dict, band: dict | None, exercise_id: int | None = None) -> dict[str, float]:
    """0-100 per metric, keyed exactly as `technique_service` reports the metric."""
    ranges = (band or {}).get("metric_ranges", {})
    out: dict[str, float] = {}

    for key, band_key in _BANDED.items():
        value, rng = metrics.get(key), ranges.get(band_key)
        if value is None or rng is None:
            continue
        penalty = _band_penalty(float(value), rng, band_key in _HIGHER_IS_WORSE)
        out[key] = round(100.0 * (1.0 - penalty), 1)

    for key, scorer in _ABSOLUTE.items():
        value = metrics.get(key)
        if value is not None:
            out[key] = round(scorer(float(value), exercise_id), 1)

    return out


def score_rep(metrics: dict, band: dict | None, pattern_score: float | None,
              exercise_id: int | None = None) -> dict:
    subs = sub_scores(metrics, band, exercise_id)
    metric_score = round(float(np.mean(list(subs.values()))), 1) if subs else None

    if metric_score is not None and pattern_score is not None:
        combined = PATTERN_WEIGHT * pattern_score + (1.0 - PATTERN_WEIGHT) * metric_score
    elif metric_score is not None:
        combined = metric_score
    else:
        combined = pattern_score

    return {
        "pattern_score": round(pattern_score, 1) if pattern_score is not None else None,
        "metric_score": metric_score,
        "combined_score": round(combined, 1) if combined is not None else None,
        "sub_scores": subs,
    }
