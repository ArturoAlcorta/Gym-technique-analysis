"""Tests for the engine-JSON → frontend-report mapping."""

from app.exercises import BENCH, RDL, SQUAT
from app.report import build_report

# The pooled reference range the sagittal metrics are scored against.
BAND = {"metric_ranges": {
    "spine_flex_delta": {"low": 5.9, "high": 27.1, "mean": 17.1, "n": 10},
    "hip_knee_rom_ratio": {"low": -0.4, "high": 2.0, "mean": 0.8, "n": 10},
    "hip_knee_lag_pct": {"low": 6.1, "high": 37.8, "mean": 23.4, "n": 10},
    "marp_deg": {"low": 42.3, "high": 85.6, "mean": 63.9, "n": 10},
}}


def _technique(metrics: dict, faults: list | None = None) -> dict:
    return {
        "metadata": {"exercise_id": 6, "total_reps": 1},
        "reps": [{
            "rep_number": 1,
            "timing": {"start_s": 0.0, "end_s": 2.0, "total_s": 2.0,
                       "eccentric_s": 1.2, "concentric_s": 0.8},
            "metrics": metrics,
            "faults": faults or [],
        }],
    }


def _dtw(reps: list[dict]) -> dict:
    return {
        "metadata": {"exercise_id": 6, "total_references": 2,
                     "reference_files": ["a_rep_001.json", "b_rep_001.json"],
                     "relevant_angles": ["knee_angle", "hip_angle"]},
        "reps": reps,
    }


def test_only_the_scored_metrics_are_shown_in_spec_order():
    """The engine measures more than the grid shows — descriptive numbers like ROM
    and depth stay in technique.json; only what fed the score is rendered."""
    report = build_report(
        _technique({"spine_flex_delta_deg": 31.2, "valgus_index": 0.8,
                    "rom_knee_mean_deg": 99.9, "depth_knee_deg": 78.4, "reached_bottom": True}),
        None, SQUAT,
    )

    keys = [m["key"] for m in report["reps"][0]["metrics"]]
    assert keys == ["valgus_index", "spine_flex_delta_deg"]
    assert report["reps"][0]["metrics"][1]["display"] == "31.2°"


def test_only_metrics_with_an_absolute_cutoff_get_a_verdict():
    report = build_report(
        _technique({"spine_flex_delta_deg": 31.2, "valgus_index": 1.4,
                    "hip_knee_lag_pct": 28.6}),
        None, SQUAT,
    )
    status = {m["key"]: m["status"] for m in report["reps"][0]["metrics"]}

    assert status["spine_flex_delta_deg"] == "info"   # no band passed in -> no verdict
    assert status["hip_knee_lag_pct"] == "info"
    assert status["valgus_index"] == "bad"            # absolute cutoff, needs no band


def test_unmeasurable_metrics_are_marked_not_available():
    report = build_report(_technique({"valgus_index": None}), None, SQUAT)
    metric = report["reps"][0]["metrics"][0]

    assert metric["status"] == "na"
    assert metric["display"] == "n/a"


def test_without_dtw_there_is_no_comparison_block():
    report = build_report(_technique({"spine_flex_delta_deg": 31.2}), None, SQUAT)

    assert report["compare"] is False
    assert report["reference"] is None
    assert report["reps"][0]["comparison"] is None
    assert report["reps"][0]["pattern_score"] is None


def test_comparison_reports_the_closest_reference_and_its_weakest_joint():
    dtw = _dtw([{
        "rep_number": 1, "best_score": 84.9, "mean_score": 71.0,
        "references": [
            {"reference_file": "a_rep_001.json", "global_score": 57.1,
             "score_by_joint": {"knee_angle": 60.0, "hip_angle": 54.2}},
            {"reference_file": "b_rep_001.json", "global_score": 84.9,
             "score_by_joint": {"knee_angle": 91.0, "hip_angle": 78.8}},
        ],
    }])
    report = build_report(_technique({"spine_flex_delta_deg": 31.2}), dtw, SQUAT)
    cmp = report["reps"][0]["comparison"]

    assert report["compare"] is True
    assert cmp["best_reference"] == "b_rep_001.json"
    assert cmp["best_score"] == 84.9
    assert cmp["n_references"] == 2
    # weakest joint first, with a human label
    assert [j["label"] for j in cmp["by_joint"]] == ["Hip", "Knee"]
    assert report["reps"][0]["pattern_score"] == 84.9
    assert report["reference"]["n"] == 2


def test_faults_from_both_paths_are_merged_without_duplicates():
    technique = _technique(
        {"elbow_avg_asymmetry_deg": 21.0},
        faults=[{"code": "elbow_asymmetry", "severity": "moderate",
                 "cue_en": "Elbow asymmetry detected.", "cue_es": "Asimetría de codos."}],
    )
    dtw = _dtw([{
        "rep_number": 1, "best_score": 70.0, "mean_score": 65.0, "references": [],
        # the bench DTW path attaches its own copy of the symmetry cues
        "faults": [{"code": "elbow_asymmetry", "severity": "moderate",
                    "cue_en": "Elbow asymmetry detected.", "cue_es": "Asimetría de codos."},
                   {"code": "shoulder_asymmetry", "severity": "major",
                    "cue_en": "Shoulder asymmetry detected.", "cue_es": "Asimetría de hombros."}],
    }])
    report = build_report(technique, dtw, BENCH)
    faults = report["reps"][0]["faults"]

    assert [f["code"] for f in faults] == ["elbow_asymmetry", "shoulder_asymmetry"]
    assert faults[0]["cue"] == "Elbow asymmetry detected."


def test_summary_averages_the_per_rep_timings():
    technique = _technique({"spine_flex_delta_deg": 31.2})
    technique["reps"].append({
        "rep_number": 2,
        "timing": {"start_s": 2.0, "end_s": 5.0, "total_s": 3.0,
                   "eccentric_s": 1.8, "concentric_s": 1.2},
        "metrics": {"spine_flex_delta_deg": 28.0},
        "faults": [],
    })
    report = build_report(technique, None, SQUAT)

    assert report["total_reps"] == 2
    assert report["summary"]["avg_total_s"] == 2.5
    assert report["summary"]["avg_concentric_s"] == 1.0



# ── Scoring ───────────────────────────────────────────────────────────────────

def test_a_rep_is_scored_from_the_relational_half_alone_without_dtw():
    """The 50/50 split exists so an analysis run without the comparison still
    gets a score — the relational half needs no DTW."""
    report = build_report(
        _technique({"spine_flex_delta_deg": 17.1, "hip_knee_lag_pct": 23.4,
                    "marp_deg": 63.9, "valgus_index": 0.5}),
        None, SQUAT, BAND,
    )
    rep = report["reps"][0]

    # every value sits inside the band, so every sub-score is full marks
    assert rep["metric_score"] == 100.0
    assert rep["pattern_score"] is None
    assert rep["score"] == 100.0
    assert report["summary"]["score"] == 100.0


def test_dtw_and_metrics_are_weighted_half_and_half():
    dtw = _dtw([{"rep_number": 1, "best_score": 60.0, "mean_score": 50.0,
                 "references": [{"reference_file": "a.json", "global_score": 60.0,
                                 "score_by_joint": {"knee_angle": 60.0}}]}])
    report = build_report(
        _technique({"spine_flex_delta_deg": 17.1, "hip_knee_lag_pct": 23.4,
                    "marp_deg": 63.9, "valgus_index": 0.5}),
        dtw, SQUAT, BAND,
    )
    rep = report["reps"][0]

    assert (rep["metric_score"], rep["pattern_score"]) == (100.0, 60.0)
    assert rep["score"] == 80.0   # 0.5 * 60 + 0.5 * 100


def test_a_metric_outside_the_band_drags_its_own_badge_and_the_score_down():
    report = build_report(
        _technique({"spine_flex_delta_deg": 17.1, "hip_knee_lag_pct": 23.4,
                    "marp_deg": 200.0, "valgus_index": 0.5}),
        None, SQUAT, BAND,
    )
    rows = {m["key"]: m for m in report["reps"][0]["metrics"]}

    assert rows["marp_deg"]["score"] == 0.0        # far above the band's high bound
    assert rows["marp_deg"]["status"] == "bad"
    assert rows["spine_flex_delta_deg"]["status"] == "ok"
    assert report["reps"][0]["score"] == 75.0      # three metrics at 100, one at 0


def test_bench_scores_its_relational_half_from_symmetry_without_a_band():
    report = build_report(
        _technique({"elbow_avg_asymmetry_deg": 6.5, "shoulder_avg_asymmetry_deg": 22.5}),
        None, BENCH, None,
    )
    rows = {m["key"]: m for m in report["reps"][0]["metrics"]}

    assert rows["elbow_avg_asymmetry_deg"]["score"] == 100.0   # under the 15° flag
    assert rows["shoulder_avg_asymmetry_deg"]["score"] == 50.0  # halfway to the 30° zero
    assert report["reps"][0]["metric_score"] == 75.0


def test_the_knee_cave_cutoff_is_per_exercise():
    """The same ankle/knee ratio means different things in a squat and a hinge: an
    RDL is a narrow stance with the knees over the feet, so ~1.1 is geometry, not
    a fault. Every reference RDL rep sits there."""
    metrics = {"valgus_index": 1.10}

    squat = build_report(_technique(metrics), None, SQUAT, BAND)["reps"][0]
    rdl = build_report(_technique(metrics), None, RDL, BAND)["reps"][0]

    # (the cue itself is raised in the engine, not here — see the technique tests)
    assert squat["metrics"][0]["score"] == 42.9
    assert squat["metrics"][0]["status"] == "bad"

    assert rdl["metrics"][0]["score"] == 100.0
    assert rdl["metrics"][0]["status"] == "ok"
