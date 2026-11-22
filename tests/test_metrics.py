"""Tests of the accuracy measures, on boxes and frame states made by hand."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pitchvlm.evalx.metrics import (
    DetectionScores,
    detection_scores,
    foot_errors_m,
    identity_scores,
    match_boxes,
    number_exact_match,
    position_errors_m,
    summarize,
    t_interval,
)
from pitchvlm.schemas import Box, FramePeople, FrameState, PersonDet, PlayerState


def box(x: float, y: float, w: float = 10.0, h: float = 20.0) -> Box:
    """Build a box from its top-left corner and its size."""
    return Box(x1=x, y1=y, x2=x + w, y2=y + h)


def people(frame: int, *boxes: Box) -> FramePeople:
    """Build the people of one frame from boxes."""
    return FramePeople(frame=frame, people=tuple(PersonDet(box=one, role="player") for one in boxes))


def state(frame: int, *players: tuple[int, float, float]) -> FrameState:
    """Build a frame state from ``(track id, pitch x, pitch y)`` triples."""
    return FrameState(
        frame=frame,
        players=tuple(
            PlayerState(track_id=track_id, pitch_x=x, pitch_y=y, role="player") for track_id, x, y in players
        ),
    )


def identity(frame: int, xy: np.ndarray) -> np.ndarray:
    """A pitch mapping that returns the image points as they are."""
    return xy


# --- match_boxes -------------------------------------------------------------------


def test_match_boxes_gives_pred_index_then_truth_index() -> None:
    pred = [box(0, 0), box(100, 100)]
    truth = [box(100, 100), box(1, 0)]

    # Pred 0 overlaps truth 1 (IoU 180 / 220), pred 1 is truth 0.
    assert match_boxes(pred, truth) == [(0, 1), (1, 0)]


def test_match_boxes_keeps_pairs_at_or_above_the_threshold() -> None:
    pred = [box(0, 0)]
    third = [box(5, 0)]  # IoU = 100 / 300
    half = [box(0, 0, 10.0, 10.0)]  # IoU = 100 / 200, exactly the default threshold

    assert match_boxes(pred, third) == []
    assert match_boxes(pred, third, iou_threshold=0.3) == [(0, 0)]
    assert match_boxes(pred, half) == [(0, 0)]
    assert match_boxes(pred, half, iou_threshold=0.51) == []


def test_match_boxes_is_one_to_one() -> None:
    pred = [box(0, 1), box(0, 0)]  # both overlap the one true box; pred 1 is the same box
    truth = [box(0, 0)]

    assert match_boxes(pred, truth) == [(1, 0)]


def test_match_boxes_finds_the_best_pairing_of_all_boxes() -> None:
    # Width 10, same height. Offsets: a-t1 2 (IoU 8/12), a-t2 3 (7/13), b-t1 3 (7/13), b-t2 8 (2/18).
    a, b = box(2, 0), box(-3, 0)
    t1, t2 = box(0, 0), box(5, 0)

    # Taking the best single pair first (a with t1) would leave b without a match.
    assert match_boxes([a, b], [t1, t2]) == [(0, 1), (1, 0)]


def test_match_boxes_does_not_give_up_a_pair_that_counts_for_pairs_that_do_not_count() -> None:
    # Width 10, same height. Offsets: a-t1 2 (IoU 8/12), a-t2 4 (6/14), b-t1 4 (6/14), b-t2 10 (no overlap).
    a, b = box(2, 0), box(-4, 0)
    t1, t2 = box(0, 0), box(6, 0)

    # The pairing of a with t2 and of b with t1 has the larger total IoU (12/14 against 8/12), but
    # both of its pairs are below the threshold of 0.5. It must not win over the pair of a with t1.
    assert match_boxes([a, b], [t1, t2]) == [(0, 0)]
    # With a threshold of 0.4 the two weaker pairs count, and together they are the best pairing.
    assert match_boxes([a, b], [t1, t2], iou_threshold=0.4) == [(0, 1), (1, 0)]


def test_match_boxes_without_boxes() -> None:
    assert match_boxes([], [box(0, 0)]) == []
    assert match_boxes([box(0, 0)], []) == []
    assert match_boxes([], []) == []


# --- detection_scores --------------------------------------------------------------

SMALL = box(0, 0, 10.0, 20.0)  # 20 pixels high
TALL = box(100, 0, 10.0, 200.0)  # 200 pixels high
MISSED = box(300, 0, 10.0, 30.0)  # 30 pixels high, never predicted
EXTRA = box(500, 500, 10.0, 20.0)  # predicted, but not a true box


def detection_case() -> tuple[list[FramePeople], list[FramePeople]]:
    """Return ``(pred, truth)``: one frame with hits, a miss and a wrong box; one true frame without predictions."""
    truth = [people(0, SMALL, TALL, MISSED), people(1, box(0, 0, 10.0, 20.0))]
    pred = [people(0, SMALL, TALL, EXTRA), people(7, box(0, 0, 10.0, 20.0))]
    return pred, truth


def test_recall_and_precision_come_from_the_counts() -> None:
    scores = DetectionScores(tp=6, fp=2, fn=4)

    assert scores.recall == pytest.approx(0.6)
    assert scores.precision == pytest.approx(0.75)


def test_recall_and_precision_without_boxes_are_nan() -> None:
    scores = DetectionScores(tp=0, fp=0, fn=0)

    assert math.isnan(scores.recall)
    assert math.isnan(scores.precision)


def test_detection_scores_count_hits_wrong_boxes_and_misses() -> None:
    pred, truth = detection_case()

    scores = detection_scores(pred, truth)

    # Frame 0: two hits, one wrong box, one miss. Frame 1 has no predicted frame: one miss.
    # The predicted frame 7 has no true frame and is ignored.
    assert (scores.tp, scores.fp, scores.fn) == (2, 1, 2)
    assert scores.recall == pytest.approx(0.5)
    assert scores.precision == pytest.approx(2 / 3)


def test_small_box_restriction_ignores_a_prediction_on_a_tall_true_box() -> None:
    pred, truth = detection_case()

    scores = detection_scores(pred, truth, max_height_px=50.0)

    # The tall true box does not count, and its prediction is neither a hit nor a false positive.
    assert (scores.tp, scores.fp, scores.fn) == (1, 1, 2)
    assert scores.recall == pytest.approx(1 / 3)


def test_small_box_restriction_counts_a_box_of_exactly_the_limit() -> None:
    pred, truth = detection_case()

    scores = detection_scores(pred, truth, max_height_px=20.0)

    # Only the two true boxes of 20 pixels count: one hit in frame 0, one miss in frame 1.
    assert (scores.tp, scores.fp, scores.fn) == (1, 1, 1)


def test_detection_scores_use_the_iou_threshold() -> None:
    pred = [people(0, box(5, 0))]
    truth = [people(0, box(0, 0))]  # IoU = 1 / 3

    strict = detection_scores(pred, truth)
    loose = detection_scores(pred, truth, iou_threshold=0.3)

    assert (strict.tp, strict.fp, strict.fn) == (0, 1, 1)
    assert (loose.tp, loose.fp, loose.fn) == (1, 0, 0)


def test_detection_scores_keep_a_hit_next_to_two_weak_overlaps() -> None:
    # The boxes of ``test_match_boxes_does_not_give_up_a_pair_that_counts_for_pairs_that_do_not_count``.
    pred = [people(0, box(2, 0), box(-4, 0))]
    truth = [people(0, box(0, 0), box(6, 0))]

    scores = detection_scores(pred, truth, iou_threshold=0.5)

    # The box at x = 2 finds the true box at x = 0. The other predicted box and the other true box have no pair.
    assert (scores.tp, scores.fp, scores.fn) == (1, 1, 1)


def test_detection_scores_of_the_truth_itself_are_perfect() -> None:
    _, truth = detection_case()

    scores = detection_scores(truth, truth)

    assert (scores.tp, scores.fp, scores.fn) == (4, 0, 0)
    assert (scores.recall, scores.precision) == (1.0, 1.0)


def test_a_frame_that_is_present_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="pred holds frame 0 more than once"):
        detection_scores([people(0), people(0)], [people(0)])
    with pytest.raises(ValueError, match="truth holds frame 3 more than once"):
        position_errors_m([state(3)], [state(3), state(3)])


# --- foot_errors_m -----------------------------------------------------------------

BASE = Box(x1=100.0, y1=100.0, x2=120.0, y2=160.0)  # foot point (110, 160)
SHIFTED = Box(x1=103.0, y1=96.0, x2=123.0, y2=156.0)  # foot point (113, 156): 3 right and 4 up, IoU 952 / 1448
OTHER = Box(x1=300.0, y1=100.0, x2=320.0, y2=160.0)


def test_foot_errors_with_an_identity_mapping_are_distances_of_the_foot_points() -> None:
    truth = [people(0, BASE, OTHER)]
    pred = [people(0, OTHER, SHIFTED)]

    errors = foot_errors_m(pred, truth, identity)

    # Pred 0 is the same box as truth 1. Pred 1 is truth 0 moved by (3, -4): 5 pixels.
    assert errors.shape == (2,)
    assert errors.tolist() == pytest.approx([0.0, 5.0])


def test_foot_errors_use_the_bottom_centre_of_the_box() -> None:
    truth = [people(0, BASE)]
    taller = [people(0, Box(x1=100.0, y1=90.0, x2=120.0, y2=160.0))]  # the top moves, the foot point stays
    deeper = [people(0, Box(x1=100.0, y1=100.0, x2=120.0, y2=166.0))]  # the bottom moves down by 6

    assert foot_errors_m(taller, truth, identity).tolist() == pytest.approx([0.0])
    assert foot_errors_m(deeper, truth, identity).tolist() == pytest.approx([6.0])


def test_foot_errors_map_both_points_with_the_same_mapping() -> None:
    def ten_pixels_per_metre(frame: int, xy: np.ndarray) -> np.ndarray:
        return xy * 0.1

    errors = foot_errors_m([people(0, SHIFTED)], [people(0, BASE)], ten_pixels_per_metre)

    assert errors.tolist() == pytest.approx([0.5])


def test_foot_errors_pass_the_frame_index_and_skip_frames_without_a_mapping() -> None:
    calls: list[tuple[int, tuple[int, ...]]] = []

    def to_pitch(frame: int, xy: np.ndarray) -> np.ndarray | None:
        calls.append((frame, xy.shape))
        return None if frame == 1 else xy * (frame + 1)

    truth = [people(0, BASE), people(1, BASE), people(2, BASE)]
    pred = [people(0, SHIFTED), people(1, SHIFTED), people(2, SHIFTED)]

    errors = foot_errors_m(pred, truth, to_pitch)

    # One call per frame, with the pred point and the true point: shape (2, 2).
    assert calls == [(0, (2, 2)), (1, (2, 2)), (2, (2, 2))]
    # Frame 0 has the scale 1 and frame 2 the scale 3. Frame 1 has no mapping.
    assert errors.tolist() == pytest.approx([5.0, 15.0])


def test_foot_errors_without_a_matched_box_are_empty() -> None:
    errors = foot_errors_m([people(0, box(0, 0))], [people(0, box(500, 500)), people(1, BASE)], identity)

    assert errors.shape == (0,)
    assert errors.dtype == np.float64


def test_foot_errors_use_an_iou_threshold_of_one_half_by_default() -> None:
    truth = [people(0, box(0, 0))]
    pred = [people(0, box(5, 0))]  # IoU = 1 / 3; the foot points are 5 pixels apart

    # The pair does not count at the default threshold of 0.5. It counts at a threshold of 0.3.
    assert foot_errors_m(pred, truth, identity).shape == (0,)
    assert foot_errors_m(pred, truth, identity, iou_threshold=0.3).tolist() == pytest.approx([5.0])


# --- position_errors_m -------------------------------------------------------------


def test_position_errors_are_the_distances_of_matched_people() -> None:
    truth = [state(0, (1, 0.0, 0.0), (2, 10.0, -5.0))]
    pred = [state(0, (7, 10.0, -4.0), (8, 3.0, 4.0), (9, 40.0, 30.0))]

    # Pred 7 is 1 m from true 2, pred 8 is 5 m from true 1, pred 9 has no partner.
    assert position_errors_m(pred, truth).tolist() == pytest.approx([1.0, 5.0])
    assert position_errors_m(pred, truth, max_match_m=3.0).tolist() == pytest.approx([1.0])


def test_position_errors_do_not_break_up_a_close_pair_for_a_pair_that_does_not_count() -> None:
    truth = [state(0, (1, 0.0, 0.0), (2, 8.0, 0.0))]
    pred = [state(0, (7, 0.5, 0.0), (8, -8.0, 0.0))]

    # Pred 7 is 0.5 m from true 1. Pred 8 is 16 m from true 2, so that pair does not count.
    # The pairing of 7 with 2 and of 8 with 1 has the smaller total distance (7.5 + 8.0 against
    # 0.5 + 16.0), but it breaks up the close pair and gives the errors 7.5 and 8.0.
    assert position_errors_m(pred, truth).tolist() == pytest.approx([0.5])


def test_position_errors_keep_a_pair_next_to_two_pairs_that_do_not_count() -> None:
    truth = [state(0, (1, 0.0, 0.0), (2, 6.0, 0.0))]
    pred = [state(0, (7, 2.0, 0.0), (8, -4.0, 0.0))]

    # Pred 7 is 2 m from true 1. The pairing of 7 with 2 and of 8 with 1 has the smaller total distance
    # (4 + 4 against 2 + 10), but with a limit of 3 m none of its pairs counts.
    assert position_errors_m(pred, truth, max_match_m=3.0).tolist() == pytest.approx([2.0])


@pytest.mark.parametrize("far_one_first", [True, False])
def test_position_errors_keep_a_pair_at_exactly_the_limit(far_one_first: bool) -> None:
    near, far = (1, 10.0, 0.0), (2, 20.0, 0.0)
    truth = [state(0, far, near) if far_one_first else state(0, near, far)]
    pred = [state(0, (7, 0.0, 0.0))]

    # The pair of 10 m counts at a limit of 10 m, whatever the order of the true people is. The pair of
    # 20 m does not count, so it must not cost the same as the pair at the limit.
    assert position_errors_m(pred, truth, max_match_m=10.0).tolist() == pytest.approx([10.0])


def test_position_errors_keep_the_sign_of_y() -> None:
    truth = [state(0, (1, 0.0, -4.0))]  # far side: negative y

    assert position_errors_m([state(0, (7, 0.0, -4.0))], truth).tolist() == pytest.approx([0.0])
    # The mirrored place on the near side is 8 m away.
    assert position_errors_m([state(0, (7, 0.0, 4.0))], truth).tolist() == pytest.approx([8.0])


def test_position_errors_keep_x_and_y_apart() -> None:
    truth = [state(0, (1, 3.0, 0.0))]

    assert position_errors_m([state(0, (7, 0.0, 3.0))], truth).tolist() == pytest.approx([math.sqrt(18.0)])


def test_position_errors_pair_frames_by_frame_index() -> None:
    truth = [state(5, (1, 0.0, 0.0)), state(6, (1, 1.0, 0.0))]
    pred = [state(9, (7, 0.0, 0.0)), state(6, (7, 1.0, 2.0))]  # other order, no frame 5, one more frame

    assert position_errors_m(pred, truth).tolist() == pytest.approx([2.0])


def test_position_errors_without_a_pair_are_empty() -> None:
    errors = position_errors_m([state(0)], [state(0, (1, 0.0, 0.0))])

    assert errors.shape == (0,)
    assert errors.dtype == np.float64


def test_position_errors_count_a_pair_at_one_place_with_a_limit_of_zero() -> None:
    truth = [state(0, (1, 0.0, 0.0), (2, 4.0, 0.0))]
    pred = [state(0, (7, 0.0, 0.0), (8, 4.0, 0.5))]

    assert position_errors_m(pred, truth, max_match_m=0.0).tolist() == [0.0]


@pytest.mark.parametrize("limit", [-1.0, -1e-9, math.nan, math.inf])
def test_matchers_by_position_refuse_a_limit_that_is_not_a_finite_number_from_zero(limit: float) -> None:
    truth = [state(0, (1, 0.0, 0.0))]
    pred = [state(0, (7, 1.0, 0.0))]

    with pytest.raises(ValueError, match="max_match_m"):
        position_errors_m(pred, truth, max_match_m=limit)
    with pytest.raises(ValueError, match="max_match_m"):
        identity_scores(pred, truth, max_match_m=limit)
    # The limit is checked also when there is nothing to match.
    with pytest.raises(ValueError, match="max_match_m"):
        position_errors_m([], [], max_match_m=limit)
    with pytest.raises(ValueError, match="max_match_m"):
        identity_scores([], [], max_match_m=limit)


# --- identity_scores ---------------------------------------------------------------


def identity_case() -> tuple[list[FrameState], list[FrameState]]:
    """Return ``(pred, truth)``: two true tracks over four frames, one id switch and one frame without a match."""
    truth = [state(frame, (1, 0.0, 0.0), (2, 20.0, 10.0)) for frame in range(4)]
    pred = [
        state(0, (5, 0.5, 0.0), (6, 20.0, 10.5)),
        state(1, (5, 0.5, 0.0), (6, 20.0, 10.5)),
        state(2, (7, 0.5, 0.0), (6, 20.0, 10.5)),  # true track 1 changes from id 5 to id 7
        state(3, (7, 0.5, 0.0), (8, 50.0, 30.0)),  # true track 2 has no prediction within 5 m
    ]
    return pred, truth


def test_identity_scores_give_coverage_switches_and_track_ratio() -> None:
    pred, truth = identity_case()

    scores = identity_scores(pred, truth)

    # True track 1 has the ids 5, 5, 7, 7: best share 2 / 4, one switch.
    # True track 2 has the ids 6, 6, 6 and one frame without an id: best share 3 / 4.
    assert scores.coverage == pytest.approx((2 / 4 + 3 / 4) / 2)
    assert scores.switches == 1
    # Four predicted ids (5, 6, 7, 8) for two true tracks.
    assert scores.track_ratio == pytest.approx(2.0)


def test_identity_scores_of_the_truth_itself_are_perfect() -> None:
    _, truth = identity_case()

    scores = identity_scores(truth, truth)

    assert (scores.coverage, scores.switches, scores.track_ratio, scores.purity) == (1.0, 0, 1.0, 1.0)


def test_identity_purity_is_the_mean_share_of_a_predicted_id_that_one_true_track_covers() -> None:
    pred, truth = identity_case()

    scores = identity_scores(pred, truth)

    # Ids 5, 6 and 7 follow one true track each; id 8 is matched in no frame and does not count.
    assert scores.purity == 1.0


def test_identity_purity_sees_a_predicted_id_that_follows_two_people_one_after_the_other() -> None:
    # True person 1 is seen in frames 0 to 2, true person 2 in frames 3 and 4. One predicted id
    # follows both: a wrong join. Coverage, switches and track ratio do not see it.
    truth = [state(frame, (1, 0.0, 0.0)) for frame in range(3)] + [state(frame, (2, 30.0, 0.0)) for frame in (3, 4)]
    pred = [state(frame, (5, 0.0, 0.0)) for frame in range(3)] + [state(frame, (5, 30.0, 0.0)) for frame in (3, 4)]

    scores = identity_scores(pred, truth)

    assert (scores.coverage, scores.switches, scores.track_ratio) == (1.0, 0, 0.5)
    assert scores.purity == pytest.approx(3 / 5)


def test_identity_purity_without_a_matched_prediction_is_nan() -> None:
    truth = [state(0, (1, 0.0, 0.0))]
    pred = [state(0, (5, 40.0, 30.0))]

    assert math.isnan(identity_scores(pred, truth).purity)


def test_identity_scores_do_not_depend_on_the_order_of_the_predicted_people() -> None:
    truth = [state(frame, (1, 0.0, 0.0), (2, 20.0, 10.0)) for frame in range(2)]
    pred = [
        state(0, (5, 0.5, 0.0), (6, 20.0, 10.5)),
        state(1, (6, 20.0, 10.5), (5, 0.5, 0.0)),  # the same people, listed in the other order
    ]

    scores = identity_scores(pred, truth)

    # A predicted id belongs to the true person that it is paired with, not to its place in the list.
    assert (scores.coverage, scores.switches, scores.track_ratio) == (1.0, 0, 1.0)


def test_identity_scores_find_the_pair_of_a_prediction_that_is_not_listed_first() -> None:
    truth = [state(0, (1, 0.0, 0.0))]
    pred = [state(0, (9, 40.0, 30.0), (5, 0.5, 0.0))]  # the first prediction is far from every true person

    scores = identity_scores(pred, truth)

    # True person 0 is paired with prediction 1, so the true track is followed by the id 5.
    assert scores.coverage == 1.0
    assert scores.switches == 0
    assert scores.track_ratio == 2.0


def test_identity_scores_follow_a_true_track_in_frame_order() -> None:
    by_frame = {frame: state(frame, (1, 0.0, 0.0)) for frame in range(3)}
    truth = [by_frame[2], by_frame[0], by_frame[1]]  # the true frames are not listed in frame order
    pred = [state(0, (5, 0.0, 0.0)), state(1, (7, 0.0, 0.0)), state(2, (5, 0.0, 0.0))]

    scores = identity_scores(pred, truth)

    # In frame order the predicted ids are 5, 7, 5: two switches. In the order of the list they
    # would be 5, 5, 7: one switch.
    assert scores.switches == 2
    assert scores.coverage == pytest.approx(2 / 3)


def test_a_gap_is_not_a_switch_but_a_new_id_after_a_gap_is_one() -> None:
    truth = [state(frame, (1, 0.0, 0.0)) for frame in range(3)]
    same_id = [state(0, (5, 0.0, 1.0)), state(2, (5, 0.0, 1.0))]  # no predicted frame 1
    new_id = [state(0, (5, 0.0, 1.0)), state(2, (9, 0.0, 1.0))]

    kept = identity_scores(same_id, truth)
    changed = identity_scores(new_id, truth)

    assert kept.coverage == pytest.approx(2 / 3)
    assert (kept.switches, kept.track_ratio) == (0, 1.0)
    assert changed.coverage == pytest.approx(1 / 3)
    assert (changed.switches, changed.track_ratio) == (1, 2.0)


def test_identity_scores_use_the_match_radius() -> None:
    truth = [state(0, (1, 0.0, 0.0))]
    pred = [state(0, (5, 0.0, 6.0))]

    # 6 m is outside the default radius of 5 m.
    assert identity_scores(pred, truth).coverage == 0.0
    assert identity_scores(pred, truth, max_match_m=7.0).coverage == 1.0


def test_identity_scores_do_not_switch_ids_for_a_pair_that_does_not_count() -> None:
    truth = [state(frame, (1, 0.0, 0.0), (2, 4.0, 0.0)) for frame in range(2)]
    pred = [
        state(0, (5, 0.5, 0.0), (6, 4.0, 0.5)),
        # True track 2 has no prediction near it. The wrong prediction 9 is 8 m from it: outside the radius of 5 m.
        state(1, (5, 0.5, 0.0), (9, -4.0, 0.0)),
    ]

    scores = identity_scores(pred, truth)

    # True track 1 keeps the id 5. True track 2 has the id 6 and then no id. In frame 1 the pairing of 5
    # with track 2 and of 9 with track 1 has the smaller total distance (3.5 + 4.0 against 0.5 + 8.0), and
    # both of its pairs are inside the radius: it would give two switches and a coverage of 0.5.
    assert scores.switches == 0
    assert scores.coverage == pytest.approx((2 / 2 + 1 / 2) / 2)
    assert scores.track_ratio == pytest.approx(1.5)


def test_identity_scores_keep_a_pair_next_to_two_pairs_that_do_not_count() -> None:
    truth = [state(0, (1, 0.0, 0.0), (2, 6.0, 0.0))]
    pred = [state(0, (7, 2.0, 0.0), (8, -4.0, 0.0))]

    scores = identity_scores(pred, truth, max_match_m=3.0)

    # True track 1 is followed by the id 7. True track 2 has no prediction within 3 m.
    assert scores.coverage == pytest.approx((1.0 + 0.0) / 2)
    assert scores.switches == 0


@pytest.mark.parametrize("far_one_first", [True, False])
def test_identity_scores_keep_a_pair_at_exactly_the_radius(far_one_first: bool) -> None:
    near, far = (1, 5.0, 0.0), (2, 9.0, 0.0)
    truth = [state(0, far, near) if far_one_first else state(0, near, far)]
    pred = [state(0, (7, 0.0, 0.0))]

    # True track 1 is exactly 5 m from the prediction: inside the default radius, in every order.
    assert identity_scores(pred, truth).coverage == pytest.approx((1.0 + 0.0) / 2)


def test_track_ratio_counts_predicted_ids_in_true_frames_only() -> None:
    truth = [state(0, (1, 0.0, 0.0))]
    pred = [state(0, (5, 0.0, 0.0)), state(1, (6, 0.0, 0.0))]

    assert identity_scores(pred, truth).track_ratio == 1.0


def test_identity_scores_without_true_tracks_are_nan() -> None:
    scores = identity_scores([state(0, (5, 0.0, 0.0))], [state(0)])

    assert math.isnan(scores.coverage)
    assert math.isnan(scores.track_ratio)
    assert scores.switches == 0


# --- number_exact_match, summarize, t_interval ---------------------------------------


def test_number_exact_match_counts_no_number_as_a_value() -> None:
    truth = {1: 10, 2: None, 3: 9, 4: 4}
    pred = {1: 10, 2: None, 3: 7, 4: None}

    # Track 1 and track 2 (None against None) are right; a wrong number and a missing number are wrong.
    assert number_exact_match(pred, truth) == pytest.approx(0.5)


def test_number_exact_match_needs_the_same_track_ids() -> None:
    with pytest.raises(ValueError, match=r"\[2, 3\]"):
        number_exact_match({1: 10, 2: 7}, {1: 10, 3: 7})


def test_number_exact_match_without_tracks_is_nan() -> None:
    assert math.isnan(number_exact_match({}, {}))


def test_summarize_gives_count_median_p90_and_mean() -> None:
    summary = summarize(np.array([4.0, 1.0, 10.0, 2.0, 3.0]))

    assert set(summary) == {"n", "median", "p90", "mean"}
    assert summary["n"] == 5.0
    assert summary["median"] == pytest.approx(3.0)
    # Sorted values 1, 2, 3, 4, 10: the 90th percentile is 4 + 0.6 * (10 - 4).
    assert summary["p90"] == pytest.approx(7.6)
    assert summary["mean"] == pytest.approx(4.0)


def test_summarize_takes_every_value_of_an_array() -> None:
    assert summarize(np.array([[1.0, 2.0], [3.0, 4.0]]))["n"] == 4.0


def test_summarize_without_values() -> None:
    summary = summarize(np.zeros(0))

    assert summary["n"] == 0.0
    assert all(math.isnan(summary[key]) for key in ("median", "p90", "mean"))


def test_t_interval_of_five_values() -> None:
    mean, low, high = t_interval([1.0, 2.0, 3.0, 4.0, 5.0])

    # Student t, 4 degrees of freedom, 97.5 %: 2.7764. Standard error: sqrt(2.5 / 5).
    assert mean == pytest.approx(3.0)
    assert low == pytest.approx(3.0 - 1.9632, abs=1e-3)
    assert high == pytest.approx(3.0 + 1.9632, abs=1e-3)


def test_t_interval_of_two_values() -> None:
    # Student t, 1 degree of freedom, 97.5 %: 12.7062. Standard error: 1.
    assert t_interval([0.0, 2.0]) == pytest.approx((1.0, 1.0 - 12.7062, 1.0 + 12.7062), abs=1e-3)


def test_t_interval_with_one_value_has_no_bounds() -> None:
    mean, low, high = t_interval([2.5])

    assert mean == 2.5
    assert math.isnan(low)
    assert math.isnan(high)


def test_t_interval_of_equal_values_has_no_width() -> None:
    assert t_interval([4.0, 4.0, 4.0]) == pytest.approx((4.0, 4.0, 4.0))


def test_t_interval_gets_wider_with_more_confidence() -> None:
    _, low_95, high_95 = t_interval([1.0, 2.0, 4.0])
    _, low_99, high_99 = t_interval([1.0, 2.0, 4.0], confidence=0.99)

    assert low_99 < low_95 < high_95 < high_99


@pytest.mark.parametrize("confidence", [0.0, 1.0, 95.0, -0.5])
def test_t_interval_refuses_a_confidence_outside_0_to_1(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        t_interval([1.0, 2.0], confidence=confidence)


def test_t_interval_refuses_an_empty_sequence() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        t_interval([])


@pytest.mark.parametrize("values", [[1.0, math.nan, 2.0], [1.0, math.inf], [-math.inf], [math.nan]])
def test_t_interval_refuses_a_value_that_is_not_finite(values: list[float]) -> None:
    with pytest.raises(ValueError, match="finite values"):
        t_interval(values)
