"""Tests of the fallback adapters: detector boxes to people, and a calibration matrix to a pitch mapping.

The camera of the calibration tests is written by hand. It stands on the near side, 70 m from
the centre mark and 15 m above the grass, and it looks at the centre mark.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

from pitchvlm.fallbacks import payload_field, payload_number
from pitchvlm.fallbacks.pnlcalib import PITCH_ORIGIN, homography_from_payload
from pitchvlm.fallbacks.rfdetr import PERSON_LABEL, people_from_detections
from pitchvlm.geometry import apply_homography, matrix_from_record
from pitchvlm.schemas import Box, FramePeople, HomographyRecord, PersonDet

# --- detector payloads ---------------------------------------------------------------

#: Score limit of the tests that do not test the limit. The default score of ``detection`` is above it.
MIN_SCORE = 0.3
#: Three valid boxes ``[x1, y1, x2, y2]``. Each one is higher than wide, so a swap of x and y shows.
BOX_A = [100.0, 50.0, 140.0, 150.0]
BOX_B = [300.0, 200.0, 330.0, 290.0]
BOX_C = [800.0, 400.0, 860.0, 560.0]


def detection(box: Any, score: Any = 0.9, label: Any = "person") -> dict[str, Any]:
    """Return one detection in the payload format."""
    return {"box_xyxy": box, "score": score, "label": label}


def detections_payload(*detections: Any) -> dict[str, Any]:
    """Return the detector payload of one frame."""
    return {"detections": list(detections)}


def boxes(result: FramePeople) -> list[list[float]]:
    """Return the boxes of the people as ``[x1, y1, x2, y2]`` lists, in their order."""
    return [[person.box.x1, person.box.y1, person.box.x2, person.box.y2] for person in result.people]


#: Payloads that do not follow the format. Every one must raise ``ValueError``.
BAD_DETECTION_PAYLOADS: dict[str, Any] = {
    "payload is a list": [detection(BOX_A)],
    "detections key is missing": {"boxes": [detection(BOX_A)]},
    "detections are a mapping": {"detections": {"0": detection(BOX_A)}},
    "detections are null": {"detections": None},
    "a detection is a list": {"detections": [[100.0, 50.0, 140.0, 150.0, 0.9, "person"]]},
    "box key is missing": {"detections": [{"score": 0.9, "label": "person"}]},
    "score key is missing": {"detections": [{"box_xyxy": BOX_A, "label": "person"}]},
    "label key is missing": {"detections": [{"box_xyxy": BOX_A, "score": 0.9}]},
    "box has 3 values": detections_payload(detection([100.0, 50.0, 140.0])),
    "box has 5 values": detections_payload(detection([100.0, 50.0, 140.0, 150.0, 0.9])),
    "box is a mapping": detections_payload(detection({"x1": 100.0, "y1": 50.0, "x2": 140.0, "y2": 150.0})),
    "box is null": detections_payload(detection(None)),
    "a box value is text": detections_payload(detection(["100", 50.0, 140.0, 150.0])),
    "a box value is a truth value": detections_payload(detection([True, 50.0, 140.0, 150.0])),
    "a box value is null": detections_payload(detection([None, 50.0, 140.0, 150.0])),
    "score is text": detections_payload(detection(BOX_A, score="0.9")),
    "score is a truth value": detections_payload(detection(BOX_A, score=True)),
    "score is null": detections_payload(detection(BOX_A, score=None)),
    "score is in percent": detections_payload(detection(BOX_A, score=90)),
    "score is negative": detections_payload(detection(BOX_A, score=-0.1)),
    "score is NaN": detections_payload(detection(BOX_A, score=float("nan"))),
    "label is a class number": detections_payload(detection(BOX_A, label=1)),
    "label is null": detections_payload(detection(BOX_A, label=None)),
    # The format is checked for every detection, also for one that the label or the score would drop.
    "a wrong box with another label": detections_payload(detection([100.0, 50.0, 140.0], label="sports ball")),
    "a wrong box below the score limit": detections_payload(detection([100.0, 50.0, 140.0], score=0.1)),
}

#: Boxes that are not valid. A person detection with such a box is dropped.
INVALID_BOXES = {
    "no width": [100.0, 50.0, 100.0, 150.0],
    "no height": [100.0, 50.0, 140.0, 50.0],
    "left and right are swapped": [140.0, 50.0, 100.0, 150.0],
    "top and bottom are swapped": [100.0, 150.0, 140.0, 50.0],
    "a value is NaN": [100.0, float("nan"), 140.0, 150.0],
    "a value is infinite": [100.0, 50.0, float("inf"), 150.0],
}


def test_people_come_from_the_person_detections() -> None:
    result = people_from_detections(
        detections_payload(
            detection([100.0, 50.0, 140.0, 150.0], score=0.92),
            detection([300, 200, 330, 290], score=0.55),  # whole numbers are numbers too
        ),
        frame=24,
        min_score=MIN_SCORE,
    )

    assert result == FramePeople(
        frame=24,
        people=(
            PersonDet(box=Box(x1=100.0, y1=50.0, x2=140.0, y2=150.0), role="player", team_key="none", number=None),
            PersonDet(box=Box(x1=300.0, y1=200.0, x2=330.0, y2=290.0), role="player", team_key="none", number=None),
        ),
    )


def test_box_values_are_x1_y1_x2_y2() -> None:
    payload = detections_payload(detection([100.0, 50.0, 140.0, 150.0]))
    person = people_from_detections(payload, frame=0, min_score=MIN_SCORE).people[0]

    assert (person.box.x1, person.box.y1, person.box.x2, person.box.y2) == (100.0, 50.0, 140.0, 150.0)
    # The box is 40 px wide and 100 px high. The foot point is the middle of its bottom edge.
    assert (person.box.width, person.box.height) == (40.0, 100.0)
    assert person.box.foot == (120.0, 150.0)


def test_every_person_is_a_player_without_team_and_number() -> None:
    payload = detections_payload(detection(BOX_A), detection(BOX_B))
    result = people_from_detections(payload, frame=3, min_score=MIN_SCORE)

    assert [(person.role, person.team_key, person.number) for person in result.people] == [("player", "none", None)] * 2


def test_only_the_label_person_is_kept() -> None:
    assert PERSON_LABEL == "person"
    result = people_from_detections(
        detections_payload(
            detection(BOX_A, label="sports ball"),
            detection(BOX_B, label="person"),
            detection(BOX_C, label="Person"),  # the label is compared as it is written
            detection(BOX_C, label="referee"),
        ),
        frame=0,
        min_score=MIN_SCORE,
    )

    assert boxes(result) == [BOX_B]


def test_score_equal_to_the_limit_is_kept() -> None:
    result = people_from_detections(
        detections_payload(detection(BOX_A, score=0.29), detection(BOX_B, score=0.3), detection(BOX_C, score=0.31)),
        frame=0,
        min_score=0.3,
    )

    # A score equal to the limit is kept.
    assert boxes(result) == [BOX_B, BOX_C]


def test_min_score_sets_the_score_limit() -> None:
    payload = detections_payload(detection(BOX_A, score=0.5), detection(BOX_B, score=0.8), detection(BOX_C, score=0.0))

    assert boxes(people_from_detections(payload, frame=0, min_score=0.8)) == [BOX_B]
    assert boxes(people_from_detections(payload, frame=0, min_score=0.0)) == [BOX_A, BOX_B, BOX_C]
    assert boxes(people_from_detections(payload, frame=0, min_score=1.0)) == []


@pytest.mark.parametrize("box", list(INVALID_BOXES.values()), ids=list(INVALID_BOXES))
def test_an_invalid_box_is_dropped(box: list[float]) -> None:
    payload = detections_payload(detection(BOX_A), detection(box), detection(BOX_B))
    result = people_from_detections(payload, frame=0, min_score=MIN_SCORE)

    assert boxes(result) == [BOX_A, BOX_B]


def test_people_keep_the_order_of_the_payload() -> None:
    # Not sorted by score and not sorted by position.
    payload = detections_payload(detection(BOX_C, score=0.4), detection(BOX_A, score=0.9), detection(BOX_B, score=0.6))

    assert boxes(people_from_detections(payload, frame=0, min_score=MIN_SCORE)) == [BOX_C, BOX_A, BOX_B]


def test_other_keys_of_the_payload_are_ignored() -> None:
    text = (
        '{"frame": 5, "model": "detector", '
        '"detections": [{"box_xyxy": [10, 20, 30, 80], "score": 0.9, "label": "person", "class_id": 1}]}'
    )

    result = people_from_detections(json.loads(text), frame=5, min_score=MIN_SCORE)

    assert result.frame == 5
    assert boxes(result) == [[10.0, 20.0, 30.0, 80.0]]


def test_a_frame_without_detections_has_no_people() -> None:
    assert people_from_detections(detections_payload(), frame=9, min_score=MIN_SCORE) == FramePeople(frame=9, people=())


@pytest.mark.parametrize("bad_payload", list(BAD_DETECTION_PAYLOADS.values()), ids=list(BAD_DETECTION_PAYLOADS))
def test_a_detector_payload_that_breaks_the_format_is_refused(bad_payload: Any) -> None:
    with pytest.raises(ValueError):
        people_from_detections(bad_payload, frame=0, min_score=MIN_SCORE)


def test_detector_error_names_the_place_in_the_payload() -> None:
    payload = detections_payload(detection(BOX_A), detection(BOX_B), detection([100.0, 50.0, 140.0]))

    with pytest.raises(ValueError, match=r"detections\[2\].*box_xyxy"):
        people_from_detections(payload, frame=0, min_score=MIN_SCORE)


@pytest.mark.parametrize("min_score", [-0.1, 1.5, float("nan")])
def test_min_score_must_be_from_0_to_1(min_score: float) -> None:
    with pytest.raises(ValueError, match="min_score"):
        people_from_detections(detections_payload(detection(BOX_A)), frame=0, min_score=min_score)


def test_detector_adapter_refuses_a_negative_frame() -> None:
    with pytest.raises(ValueError):
        people_from_detections(detections_payload(detection(BOX_A)), frame=-1, min_score=MIN_SCORE)


# --- calibration payloads ------------------------------------------------------------

WIDTH, HEIGHT = 1280, 720
FOCAL_PX = 1500.0
#: Place of the test camera: the pitch y below it and its height above the grass, metres. Its pitch x is 0.
CAMERA_Y, CAMERA_HEIGHT = 70.0, 15.0
#: Distance from the test camera to the centre mark, metres.
CAMERA_DISTANCE = math.hypot(CAMERA_Y, CAMERA_HEIGHT)

#: Pitch points for the mapping tests: the centre mark, both touchlines, two corners and two other points.
PITCH_POINTS = [(0.0, 0.0), (0.0, -34.0), (0.0, 34.0), (52.5, -34.0), (-52.5, 34.0), (-20.0, -10.0), (30.0, 12.5)]

#: A pitch-to-image matrix of a view from straight above: 10 pixels per metre, centre mark at pixel (640, 360).
TOP_DOWN = [[10.0, 0.0, 640.0], [0.0, 10.0, 360.0], [0.0, 0.0, 1.0]]

#: Matrices without an inverse. The fourth one is singular only up to rounding errors.
SINGULAR_MATRICES = {
    "all values are 0": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    "two rows are equal": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.0, 0.0, 1.0]],
    "a row is twice another row": [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]],
    "the rows have equal steps": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    "a column is 0": [[1.0, 0.0, 3.0], [4.0, 0.0, 6.0], [7.0, 0.0, 10.0]],
}

IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

#: Payloads that do not follow the format. Every one must raise ``ValueError``.
BAD_CALIBRATION_PAYLOADS: dict[str, Any] = {
    "payload is a list": IDENTITY,
    "matrix key is missing": {"pitch_origin": "centre"},
    "pitch origin is missing": {"homography_pitch_to_image": IDENTITY},
    "pitch origin is null": {"homography_pitch_to_image": IDENTITY, "pitch_origin": None},
    "matrix has 2 rows": {"homography_pitch_to_image": IDENTITY[:2], "pitch_origin": "centre"},
    "matrix has 4 rows": {"homography_pitch_to_image": [*IDENTITY, [0.0, 0.0, 1.0]], "pitch_origin": "centre"},
    "a row has 2 values": {
        "homography_pitch_to_image": [[1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "pitch_origin": "centre",
    },
    "a row has 4 values": {
        "homography_pitch_to_image": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "pitch_origin": "centre",
    },
    "matrix is a flat list": {
        "homography_pitch_to_image": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "pitch_origin": "centre",
    },
    "matrix is text": {"homography_pitch_to_image": "abc", "pitch_origin": "centre"},
    "a value is text": {
        "homography_pitch_to_image": [["1.0", 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "pitch_origin": "centre",
    },
    "a value is a truth value": {
        "homography_pitch_to_image": [[True, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "pitch_origin": "centre",
    },
    "a value is null": {
        "homography_pitch_to_image": [[None, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "pitch_origin": "centre",
    },
    "a value is NaN": {
        "homography_pitch_to_image": [[float("nan"), 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "pitch_origin": "centre",
    },
    "a value is infinite": {
        "homography_pitch_to_image": [[1.0, 0.0, float("inf")], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "pitch_origin": "centre",
    },
}


def project(x: float, y: float) -> tuple[float, float]:
    """Project a pitch point into the image of the test camera, with plain geometry and no matrix.

    The camera looks at the centre mark. ``right`` and ``down`` are the offsets of the point
    from the viewing line of the camera, and ``depth`` is its distance along that line, all in
    metres.
    """
    right = x
    down = CAMERA_HEIGHT * y / CAMERA_DISTANCE
    depth = (CAMERA_Y * (CAMERA_Y - y) + CAMERA_HEIGHT**2) / CAMERA_DISTANCE
    return (WIDTH / 2.0 + FOCAL_PX * right / depth, HEIGHT / 2.0 + FOCAL_PX * down / depth)


def camera_matrix() -> np.ndarray:
    """Return the pitch-to-image matrix of the test camera: the projection of ``project`` as a homography."""
    reach = CAMERA_DISTANCE**2
    centre_x, centre_y = WIDTH / 2.0, HEIGHT / 2.0
    return np.array(
        [
            [FOCAL_PX * CAMERA_DISTANCE, -centre_x * CAMERA_Y, centre_x * reach],
            [0.0, FOCAL_PX * CAMERA_HEIGHT - centre_y * CAMERA_Y, centre_y * reach],
            [0.0, -CAMERA_Y, reach],
        ]
    )


def calibration_payload(matrix: np.ndarray) -> dict[str, Any]:
    """Return the calibration payload of one frame, as the JSON reader gives it: nested lists."""
    return {"homography_pitch_to_image": matrix.tolist(), "pitch_origin": "centre"}


def image_points() -> np.ndarray:
    """Return the image points of ``PITCH_POINTS`` in the test camera, shape (n, 2)."""
    return np.array([project(x, y) for x, y in PITCH_POINTS])


def test_the_test_camera_follows_the_conventions() -> None:
    matrix = camera_matrix()
    for x, y in PITCH_POINTS:
        u_w, v_w, w = matrix @ np.array([x, y, 1.0])
        assert (u_w / w, v_w / w) == pytest.approx(project(x, y))  # the matrix agrees with the plain geometry
    assert project(0.0, 0.0) == pytest.approx((640.0, 360.0))  # the camera looks at the centre mark
    assert project(0.0, -34.0)[1] < 360.0 < project(0.0, 34.0)[1]  # the far side is higher in the picture
    assert project(-20.0, 0.0)[0] < 640.0 < project(20.0, 0.0)[0]  # the right goal is to the right


def test_top_down_matrix_is_inverted() -> None:
    record = homography_from_payload({"homography_pitch_to_image": TOP_DOWN, "pitch_origin": "centre"}, frame=4)

    # Pixels of the centre mark, the far-left corner, the near-right corner and the far end of the halfway line.
    image_xy = np.array([[640.0, 360.0], [115.0, 20.0], [1165.0, 700.0], [640.0, 20.0]])
    pitch_xy = [[0.0, 0.0], [-52.5, -34.0], [52.5, 34.0], [0.0, -34.0]]
    np.testing.assert_allclose(apply_homography(matrix_from_record(record), image_xy), pitch_xy, atol=1e-9)


def test_record_maps_image_points_back_to_the_pitch() -> None:
    record = homography_from_payload(calibration_payload(camera_matrix()), frame=12)

    mapped = apply_homography(matrix_from_record(record), image_points())

    np.testing.assert_allclose(mapped, PITCH_POINTS, atol=1e-6)


def test_inverted_mapping_keeps_the_pitch_signs() -> None:
    image_to_pitch = matrix_from_record(homography_from_payload(calibration_payload(camera_matrix()), frame=0))

    higher, lower = apply_homography(image_to_pitch, np.array([[640.0, 300.0], [640.0, 600.0]]))
    assert higher[1] < 0.0 < lower[1]  # higher in the picture is the far side: negative pitch y
    left, right = apply_homography(image_to_pitch, np.array([[340.0, 360.0], [940.0, 360.0]]))
    assert left[0] < 0.0 < right[0]  # to the right in the picture is towards the right goal: positive pitch x


def test_record_of_a_matrix_is_a_landmark_mapping() -> None:
    record = homography_from_payload(calibration_payload(camera_matrix()), frame=12)

    assert record.frame == 12
    assert record.source == "landmarks"
    assert record.valid
    # The payload has no inlier count and no reprojection error.
    assert record.n_inliers == 0
    assert record.reproj_px is None
    # Same scale rule as the landmark fits of geometry.py.
    assert np.linalg.norm(matrix_from_record(record)) == pytest.approx(1.0)


@pytest.mark.parametrize("factor", [1e-6, -3.0, 1e6])
def test_scale_and_sign_of_the_payload_matrix_do_not_matter(factor: float) -> None:
    record = homography_from_payload(calibration_payload(camera_matrix() * factor), frame=0)

    np.testing.assert_allclose(apply_homography(matrix_from_record(record), image_points()), PITCH_POINTS, atol=1e-6)
    assert np.linalg.norm(matrix_from_record(record)) == pytest.approx(1.0)


def test_calibration_payload_read_from_json_text() -> None:
    text = (
        '{"homography_pitch_to_image": [[10, 0, 640], [0, 10, 360], [0, 0, 1]], '
        '"pitch_origin": "centre", "tool": "calibration"}'
    )

    record = homography_from_payload(json.loads(text), frame=2)

    centre = apply_homography(matrix_from_record(record), np.array([[640.0, 360.0]]))
    np.testing.assert_allclose(centre, [[0.0, 0.0]], atol=1e-9)


def test_null_matrix_gives_a_record_without_a_mapping() -> None:
    record = homography_from_payload({"homography_pitch_to_image": None, "pitch_origin": "centre"}, frame=7)

    assert record == HomographyRecord(frame=7)
    assert record.source == "none"
    assert record.matrix is None
    assert not record.valid


def test_null_matrix_needs_no_pitch_origin() -> None:
    assert homography_from_payload({"homography_pitch_to_image": None}, frame=7) == HomographyRecord(frame=7)
    with_null_origin = {"homography_pitch_to_image": None, "pitch_origin": None}
    assert homography_from_payload(with_null_origin, frame=7) == HomographyRecord(frame=7)


@pytest.mark.parametrize("origin", ["corner", "center", "Centre", "top-left", pytest.param("", id="empty text"), 0])
@pytest.mark.parametrize("matrix", [TOP_DOWN, None], ids=["with a matrix", "without a matrix"])
def test_another_pitch_origin_is_refused(matrix: Any, origin: Any) -> None:
    assert PITCH_ORIGIN == "centre"

    with pytest.raises(ValueError, match="pitch_origin"):
        homography_from_payload({"homography_pitch_to_image": matrix, "pitch_origin": origin}, frame=0)


@pytest.mark.parametrize("matrix", list(SINGULAR_MATRICES.values()), ids=list(SINGULAR_MATRICES))
def test_a_singular_matrix_is_refused(matrix: list[list[float]]) -> None:
    with pytest.raises(ValueError, match="singular"):
        homography_from_payload({"homography_pitch_to_image": matrix, "pitch_origin": "centre"}, frame=0)


@pytest.mark.parametrize("bad_payload", list(BAD_CALIBRATION_PAYLOADS.values()), ids=list(BAD_CALIBRATION_PAYLOADS))
def test_a_calibration_payload_that_breaks_the_format_is_refused(bad_payload: Any) -> None:
    with pytest.raises(ValueError):
        homography_from_payload(bad_payload, frame=0)


@pytest.mark.parametrize("matrix", [TOP_DOWN, None], ids=["with a matrix", "without a matrix"])
def test_calibration_adapter_refuses_a_negative_frame(matrix: Any) -> None:
    with pytest.raises(ValueError):
        homography_from_payload({"homography_pitch_to_image": matrix, "pitch_origin": "centre"}, frame=-1)


# --- payload checks ------------------------------------------------------------------


def test_payload_field_returns_the_value_of_a_key() -> None:
    assert payload_field({"score": 0.5}, "score", "detections[0]") == 0.5
    assert payload_field({"matrix": None}, "matrix", "payload") is None  # null is a value, not a missing key


def test_payload_field_names_the_place_of_a_problem() -> None:
    with pytest.raises(ValueError, match=r"detections\[2\].*'score' is missing"):
        payload_field({"label": "person"}, "score", "detections[2]")
    with pytest.raises(ValueError, match=r"detections\[2\].*expected a mapping"):
        payload_field(["person"], "score", "detections[2]")


def test_payload_number_gives_a_float() -> None:
    whole = payload_number(3, "payload")
    assert whole == 3.0
    assert type(whole) is float
    assert payload_number(np.float32(0.5), "payload") == 0.5
    assert math.isnan(payload_number(float("nan"), "payload"))  # the caller decides about NaN


@pytest.mark.parametrize("value", [True, False, "3", None, [3.0]])
def test_payload_number_refuses_what_is_not_a_number(value: Any) -> None:
    with pytest.raises(ValueError, match="expected a number"):
        payload_number(value, "payload")
