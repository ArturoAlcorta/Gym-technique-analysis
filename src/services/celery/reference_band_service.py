"""
Build the acceptable range of each relational metric from the reference reps.

The reference reps are run through the *same* code path a user submission takes
(`extract_angles` → `compute_rep_metrics`), so trainers and users are measured
identically; a band imported from elsewhere is worthless, because these metrics
are defined by their implementation as much as by their name. Pooling a single
range across every reference rep is only defensible because the metrics are
proportion-invariant by design (ratios, ranges, phase) — that is the whole
reason for choosing them.

Metrics judged against an absolute cutoff instead (knee valgus, bench L/R
symmetry) are deliberately absent: they need no band, and the reference footage
is filmed side-on, so it cannot measure valgus at all.

    python -m services.celery.reference_band_service <exercise_id> [references_dir]
"""

import json
from pathlib import Path

import numpy as np

from .angles2d_service import extract_angles
from .relational_metrics import compute_rep_metrics

# Metrics that get a pooled range. Keyed as `compute_rep_metrics` names them.
BAND_METRICS = ["spine_flex_delta", "hip_knee_rom_ratio", "hip_knee_lag_pct", "marp_deg"]

# Added beyond the observed span so a tight pool still tolerates normal variation.
_MARGIN = {
    "spine_flex_delta": 4.0,
    "hip_knee_rom_ratio": 1.0,
    "hip_knee_lag_pct": 3.0,
    "marp_deg": 10.0,
}


def reference_reps_dir(references_dir: Path, exercise_id: int) -> Path:
    return Path(references_dir) / str(exercise_id)


def band_path(references_dir: Path, exercise_id: int) -> Path:
    return Path(references_dir) / "bands" / f"{exercise_id}.json"


def load_band(references_dir: Path, exercise_id: int) -> dict | None:
    """The band for this exercise, or None if it has none (e.g. bench press)."""
    path = band_path(references_dir, exercise_id)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build_band(references_dir: Path, exercise_id: int) -> dict:
    files = sorted(reference_reps_dir(references_dir, exercise_id).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no reference reps in {reference_reps_dir(references_dir, exercise_id)}")

    per_metric: dict[str, list[float]] = {m: [] for m in BAND_METRICS}
    for path in files:
        with open(path, encoding="utf-8") as fh:
            rep = json.load(fh)
        # Each reference file is exactly one rep, so the whole file is the window.
        metrics = compute_rep_metrics(extract_angles(rep)["frames"], rep["frames"])
        for m in BAND_METRICS:
            if metrics.get(m) is not None:
                per_metric[m].append(float(metrics[m]))

    ranges = {}
    for m, values in per_metric.items():
        if not values:
            continue
        arr = np.array(values, dtype=np.float64)
        margin = _MARGIN.get(m, 0.0)
        ranges[m] = {
            "low": round(float(arr.min()) - margin, 5),
            "high": round(float(arr.max()) + margin, 5),
            "mean": round(float(arr.mean()), 5),
            "n": int(len(values)),
        }

    return {
        "metadata": {
            "exercise_id": exercise_id,
            "n_reps": len(files),
            "source_reps": [p.name for p in files],
            "margins": _MARGIN,
        },
        "metric_ranges": ranges,
    }


def build_and_cache(references_dir: Path, exercise_id: int) -> Path:
    band = build_band(references_dir, exercise_id)
    out = band_path(references_dir, exercise_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(band, fh, ensure_ascii=False, indent=2)
    print(f"Band for exercise {exercise_id} ({band['metadata']['n_reps']} reference reps) → {out}")
    return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(__doc__.strip().splitlines()[-1])
        raise SystemExit(1)
    build_and_cache(Path(sys.argv[2] if len(sys.argv) > 2 else "references"), int(sys.argv[1]))
