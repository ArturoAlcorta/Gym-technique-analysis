"""Tests for the reference-free per-rep technique analysis."""

import math

import pytest

from services.celery.technique_service import analyze_technique

SHANK = 1.0  # knee→ankle distance in the synthetic skeleton, so ankle separations
             # below can be read directly as a fraction of shank length.


def _kp(x: float, y: float, score: float = 0.9) -> dict:
    return {"x": x, "y": y, "score": score}


def _leg(offset_ankle: float, offset_knee: float, knee_deg: float, prefix: str) -> dict:
    """One leg with a prescribed knee angle: knee at the origin, ankle straight
    'below' it, hip rotated away by `knee_deg`. y grows downward, as in image
    coordinates, so a small knee angle puts the hip lower = deeper."""
    theta = math.radians(knee_deg)
    return {
        f"{prefix}_Ankle": _kp(offset_ankle, SHANK),
        f"{prefix}_Knee": _kp(offset_knee, 0.0),
        f"{prefix}_Hip": _kp(offset_knee + math.sin(theta), math.cos(theta)),
        f"{prefix.lower()[0]}_big_toe": _kp(offset_ankle + 0.2, SHANK + 0.1),
    }


def _squat_frame(idx: int, knee_deg: float, ankle_sep: float, knee_sep: float) -> dict:
    kps = {}
    kps.update(_leg(-ankle_sep / 2, -knee_sep / 2, knee_deg, "L"))
    kps.update(_leg(ankle_sep / 2, knee_sep / 2, knee_deg, "R"))
    theta = math.radians(knee_deg)
    hip_y = math.cos(theta)
    kps["L2"] = _kp(math.sin(theta), hip_y - 0.5)
    kps["T6"] = _kp(math.sin(theta), hip_y - 1.0)
    kps["L_Shoulder"] = _kp(math.sin(theta) - 0.2, hip_y - 1.1)
    kps["R_Shoulder"] = _kp(math.sin(theta) + 0.2, hip_y - 1.1)
    return {"frame_idx": idx, "timestamp_s": round(idx / 15.0, 4), "keypoints": kps}


def _squat_video(ankle_sep: float, knee_sep: float, n: int = 21) -> dict:
    """One squat rep: knee angle 170° → 80° → 170°."""
    frames = []
    for i in range(n):
        phase = 1 - abs(2 * i / (n - 1) - 1)          # 0 → 1 → 0
        frames.append(_squat_frame(i, 170 - 90 * phase, ankle_sep, knee_sep))
    return {
        "metadata": {"exercise": "sagittal", "fps": 15.0,
                     "normalization": {"dominant_side": "left"}},
        "frames": frames,
    }


def _squat_rep(n: int = 21) -> dict:
    return {
        "rep_number": 1, "frame_start": 0, "frame_bottom": n // 2, "frame_end": n - 1,
        "timestamp_start": 0.0, "timestamp_bottom": round((n // 2) / 15, 4),
        "timestamp_end": round((n - 1) / 15, 4),
        "duration_total_s": 1.333, "duration_eccentric_s": 0.667, "duration_concentric_s": 0.666,
        "min_knee_angle": 80.0, "min_hip_angle": 95.0, "reached_bottom": True,
    }


def test_reports_one_entry_per_rep_with_timing_passed_through():
    result = analyze_technique(_squat_video(0.6, 0.8), [_squat_rep()], 6)

    assert result["metadata"]["total_reps"] == 1
    rep = result["reps"][0]
    assert rep["rep_number"] == 1
    assert rep["timing"]["total_s"] == 1.333
    assert rep["timing"]["eccentric_s"] == 0.667
    assert rep["metrics"]["depth_knee_deg"] == 80.0
    assert rep["metrics"]["reached_bottom"] is True


def test_rom_matches_the_swing_of_the_synthetic_rep():
    # Equal ankle/knee separation keeps the shank vertical, so the measured knee
    # angle is exactly the one the frames were built with (a splayed stance tilts
    # the shank and shifts every angle by a few degrees).
    rep = analyze_technique(_squat_video(0.6, 0.6), [_squat_rep()], 6)["reps"][0]
    # knee angle swings 170° → 80° → 170°
    assert rep["metrics"]["rom_knee_mean_deg"] == pytest.approx(90.0, abs=1.0)
    assert rep["metrics"]["bottom_knee_mean_deg"] == pytest.approx(80.0, abs=1.0)


def test_valgus_is_flagged_when_the_knees_track_inside_the_feet():
    # Frontal-ish view (ankles 0.6 of a shank apart) with the knees pulled in.
    rep = analyze_technique(_squat_video(0.6, 0.3), [_squat_rep()], 6)["reps"][0]

    assert rep["metrics"]["valgus_index"] == pytest.approx(2.0, abs=0.01)
    assert [f["code"] for f in rep["faults"]] == ["knee_valgus"]


def test_valgus_is_not_flagged_when_the_knees_track_over_the_feet():
    rep = analyze_technique(_squat_video(0.6, 0.8), [_squat_rep()], 6)["reps"][0]

    assert rep["metrics"]["valgus_index"] == pytest.approx(0.75, abs=0.01)
    assert rep["faults"] == []


def test_valgus_is_unavailable_on_a_side_on_camera():
    """A sagittal view collapses both ankles onto the same x: the ankle/knee width
    ratio there is keypoint jitter, not knee cave, so it must not be reported."""
    rep = analyze_technique(_squat_video(0.02, 0.01), [_squat_rep()], 6)["reps"][0]

    assert rep["metrics"]["valgus_index"] is None
    assert rep["faults"] == []


def test_reps_outside_the_captured_frames_are_skipped():
    rep = dict(_squat_rep(), frame_start=500, frame_bottom=510, frame_end=520)
    result = analyze_technique(_squat_video(0.6, 0.8), [rep], 6)

    assert result["reps"] == []
    assert result["metadata"]["total_reps"] == 0


def test_unknown_exercise_is_rejected():
    with pytest.raises(ValueError, match="no technique-analysis config"):
        analyze_technique(_squat_video(0.6, 0.8), [_squat_rep()], 99)


# ── Bench press ───────────────────────────────────────────────────────────────

def _arm(elbow_deg: float, prefix: str) -> dict:
    """One arm with a prescribed elbow angle: elbow at the origin, shoulder straight
    'above' it, wrist rotated away by `elbow_deg`."""
    theta = math.radians(elbow_deg)
    return {
        f"{prefix}_Hip": _kp(0.0, 1.5),
        f"{prefix}_Shoulder": _kp(0.0, 1.0),
        f"{prefix}_Elbow": _kp(0.0, 0.0),
        f"{prefix}_Wrist": _kp(math.sin(theta), -math.cos(theta)),
    }


def _bench_video(left_offset_deg: float, n: int = 21) -> dict:
    """One bench rep: elbow angle 170° → 70° → 170°, with the left elbow optionally
    held a constant number of degrees more bent than the right. The offset is
    subtracted so the left angle stays under 180°, where the joint-angle formula
    would fold it back down and hide the asymmetry."""
    frames = []
    for i in range(n):
        phase = 1 - abs(2 * i / (n - 1) - 1)
        elbow = 170 - 100 * phase
        kps = {**_arm(elbow - left_offset_deg, "L"), **_arm(elbow, "R")}
        frames.append({"frame_idx": i, "timestamp_s": round(i / 15.0, 4), "keypoints": kps})
    return {"metadata": {"exercise": "bench_press", "fps": 15.0, "normalization": {}},
            "frames": frames}


def _bench_rep(n: int = 21) -> dict:
    return {
        "rep_number": 1, "frame_start": 0, "frame_bottom": n // 2, "frame_end": n - 1,
        "timestamp_start": 0.0, "timestamp_bottom": round((n // 2) / 15, 4),
        "timestamp_end": round((n - 1) / 15, 4),
        "duration_total_s": 1.333, "duration_eccentric_s": 0.667, "duration_concentric_s": 0.666,
        "min_elbow_avg": 70.0, "reached_bottom": True,
    }


def test_bench_symmetry_is_clean_when_both_arms_track_together():
    rep = analyze_technique(_bench_video(0.0), [_bench_rep()], 1)["reps"][0]

    assert rep["metrics"]["elbow_avg_asymmetry_deg"] == pytest.approx(0.0, abs=0.01)
    assert rep["metrics"]["elbow_asymmetry_flag"] is False
    assert rep["faults"] == []


def test_bench_asymmetry_above_the_cutoff_raises_a_cue():
    rep = analyze_technique(_bench_video(20.0), [_bench_rep()], 1)["reps"][0]

    assert rep["metrics"]["elbow_avg_asymmetry_deg"] == pytest.approx(20.0, abs=0.5)
    assert "elbow_asymmetry" in [f["code"] for f in rep["faults"]]


def test_the_knee_cave_cue_uses_a_per_exercise_cutoff():
    """An ankle/knee ratio of ~1.1 is a fault in a squat, where the knees are driven
    out past the feet, and normal geometry in an RDL's narrow stance — every
    reference RDL rep measures 1.01–1.14."""
    video = _squat_video(0.66, 0.60)          # ankles 10% wider than the knees
    rep = _squat_rep()

    squat = analyze_technique(video, [rep], 6)["reps"][0]
    rdl = analyze_technique(video, [rep], 7)["reps"][0]

    assert squat["metrics"]["valgus_index"] == pytest.approx(1.1, abs=0.01)
    assert [f["code"] for f in squat["faults"]] == ["knee_valgus"]

    assert rdl["metrics"]["valgus_index"] == pytest.approx(1.1, abs=0.01)
    assert rdl["faults"] == []
