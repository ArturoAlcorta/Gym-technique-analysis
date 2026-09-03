"""Tests for the keypoint-selection rules the angle extraction depends on."""

from services.celery.angle_extraction_service import _JOINTS, _best, get_dominant_side


def _kp(x: float, y: float, score: float = 0.9) -> dict:
    return {"x": x, "y": y, "score": score}


def _frames(left_shank: float, right_shank: float, left_score: float = 0.9,
            right_score: float = 0.9, n: int = 20) -> list[dict]:
    """Frames whose only meaningful difference is how long each shank projects."""
    return [{
        "frame_idx": i,
        "keypoints": {
            "L_Knee": _kp(0.0, 0.0, left_score), "L_Ankle": _kp(0.0, left_shank, left_score),
            "R_Knee": _kp(1.0, 0.0, right_score), "R_Ankle": _kp(1.0, right_shank, right_score),
        },
    } for i in range(n)]


def test_the_visible_side_is_the_one_that_projects_longer():
    """Perspective renders the leg nearer the lens larger, so the longer shank is
    the leg the camera actually sees."""
    assert get_dominant_side(_frames(left_shank=1.2, right_shank=1.0)) == "left"
    assert get_dominant_side(_frames(left_shank=1.0, right_shank=1.2)) == "right"


def test_confidence_does_not_decide_the_side():
    """The occluded leg can score higher than the visible one — on real footage it
    did, by 10.6% — so keypoint confidence must not be what picks the side."""
    frames = _frames(left_shank=1.2, right_shank=1.0, left_score=0.6, right_score=0.95)

    assert get_dominant_side(frames) == "left"


def test_the_side_does_not_change_with_the_window_it_is_measured_over():
    """A user's rep and a reference rep cut from the same footage must land on the
    same leg, or the comparison is left-against-right."""
    frames = _frames(left_shank=1.2, right_shank=1.0, n=60)
    whole = get_dominant_side(frames)

    assert all(get_dominant_side(frames[i:i + 10]) == whole for i in range(0, 60, 10))


def test_landmark_choice_is_a_fallback_chain_not_a_confidence_contest():
    """`L_Hip` (COCO, the joint centre) and `l_ASIS` (an anatomical marker several
    centimetres away) are different points, so the more confident one must not win
    — that swapped the measurement mid-rep and put 47° steps in the angle series."""
    kps = {"L_Hip": _kp(0.0, 0.0, 0.6), "l_ASIS": _kp(0.3, -0.2, 0.99)}

    assert _best(kps, _JOINTS["left"]["hip"]) is kps["L_Hip"]


def test_the_fallback_is_used_when_the_preferred_landmark_is_occluded():
    kps = {"L_Hip": _kp(0.0, 0.0, 0.2), "l_ASIS": _kp(0.3, -0.2, 0.8)}

    assert _best(kps, _JOINTS["left"]["hip"]) is kps["l_ASIS"]
