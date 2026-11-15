"""Tests for the pitch model and the drawing helpers of ``pitchvlm.pitch``.

Conventions under test: metres, origin at the centre mark, x grows towards the
right goal, y grows towards the main camera. So the far touchline has negative
y and is at the top of a drawing. Images are BGR.
"""

from __future__ import annotations

import dataclasses
import math

import cv2
import numpy as np
import pytest

from pitchvlm.pitch import (
    ARC_HALF_CHORD,
    CIRCLE_RADIUS,
    HALF_LENGTH,
    HALF_WIDTH,
    LANDMARK_BY_ID,
    LANDMARKS,
    PITCH_LENGTH,
    PITCH_WIDTH,
    Landmark,
    PitchCanvas,
    Polyline,
    draw_landmark_diagram,
    inside_pitch,
    landmark_legend_text,
    landmark_xy,
    pitch_polylines,
    pitch_spots,
)

ALL_IDS = tuple(range(1, 36))
#: Half the chord where a penalty arc (radius 9.15 m around the penalty spot) crosses the penalty-area
#: line, which is 5.5 m in front of the spot.
ARC = math.sqrt(9.15**2 - 5.5**2)
#: The 35 landmarks in id order, written by hand from the Laws of the Game and not from the constants
#: of ``pitch.py``: pitch 105 x 68 m, penalty area 16.5 x 40.32 m, goal area 5.5 x 18.32 m, penalty
#: spot at 11 m, circle radius 9.15 m, goal 7.32 m wide. Each entry is (name, x, y) in metres.
LANDMARKS_BY_HAND: tuple[tuple[str, float, float], ...] = (
    ("far-left corner", -52.5, -34.0),
    ("left penalty area, far corner on the goal line", -52.5, -20.16),
    ("left goal area, far corner on the goal line", -52.5, -9.16),
    ("left goal, far post base", -52.5, -3.66),
    ("left goal, near post base", -52.5, 3.66),
    ("left goal area, near corner on the goal line", -52.5, 9.16),
    ("left penalty area, near corner on the goal line", -52.5, 20.16),
    ("near-left corner", -52.5, 34.0),
    ("left goal area, far front corner", -47.0, -9.16),
    ("left goal area, near front corner", -47.0, 9.16),
    ("left penalty spot", -41.5, 0.0),
    ("left penalty area, far front corner", -36.0, -20.16),
    ("left penalty arc, far crossing with the penalty-area line", -36.0, -ARC),
    ("left penalty arc, near crossing with the penalty-area line", -36.0, ARC),
    ("left penalty area, near front corner", -36.0, 20.16),
    ("halfway line, far end", 0.0, -34.0),
    ("centre circle, far crossing with the halfway line", 0.0, -9.15),
    ("centre spot", 0.0, 0.0),
    ("centre circle, near crossing with the halfway line", 0.0, 9.15),
    ("halfway line, near end", 0.0, 34.0),
    ("right penalty area, far front corner", 36.0, -20.16),
    ("right penalty arc, far crossing with the penalty-area line", 36.0, -ARC),
    ("right penalty arc, near crossing with the penalty-area line", 36.0, ARC),
    ("right penalty area, near front corner", 36.0, 20.16),
    ("right penalty spot", 41.5, 0.0),
    ("right goal area, far front corner", 47.0, -9.16),
    ("right goal area, near front corner", 47.0, 9.16),
    ("far-right corner", 52.5, -34.0),
    ("right penalty area, far corner on the goal line", 52.5, -20.16),
    ("right goal area, far corner on the goal line", 52.5, -9.16),
    ("right goal, far post base", 52.5, -3.66),
    ("right goal, near post base", 52.5, 3.66),
    ("right goal area, near corner on the goal line", 52.5, 9.16),
    ("right penalty area, near corner on the goal line", 52.5, 20.16),
    ("near-right corner", 52.5, 34.0),
)
#: Crossings of a penalty arc with a penalty-area line: left far, left near, right far, right near.
ARC_CROSSING_IDS = (13, 14, 22, 23)
#: Canvas of the diagram tests. At this scale no number label covers the dot of another landmark.
DIAGRAM_CANVAS = PitchCanvas(scale=14.0, margin_m=6.0)


def all_xy() -> np.ndarray:
    """Return the pitch coordinates of all landmarks in id order, shape (35, 2)."""
    return landmark_xy(ALL_IDS)


def same_point_set(first: np.ndarray, second: np.ndarray, tolerance: float = 1e-9) -> bool:
    """Return True when both (n, 2) arrays hold the same points, in any order."""
    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    return bool((distances.min(axis=1) < tolerance).all() and (distances.min(axis=0) < tolerance).all())


def distance_to_polyline(point: np.ndarray, line: Polyline) -> float:
    """Return the distance in metres from a point to the nearest segment of a polyline."""
    points = np.vstack([line.points, line.points[:1]]) if line.closed else line.points
    starts, ends = points[:-1], points[1:]
    segments = ends - starts
    along = np.sum((point - starts) * segments, axis=1) / np.sum(segments * segments, axis=1)
    nearest = starts + np.clip(along, 0.0, 1.0)[:, None] * segments
    return float(np.linalg.norm(nearest - point, axis=1).min())


def penalty_arcs(arc_samples: int = 64) -> list[Polyline]:
    """Return the open polylines that start at a penalty arc crossing: the penalty arcs."""
    crossings = landmark_xy(ARC_CROSSING_IDS)
    arcs: list[Polyline] = []
    for line in pitch_polylines(arc_samples):
        gap = float(np.linalg.norm(crossings - line.points[0], axis=1).min())
        if not line.closed and gap < 1e-6:
            arcs.append(line)
    return arcs


def name_words(name: str) -> set[str]:
    """Split a landmark name into its words."""
    return set(name.replace("-", " ").replace(",", " ").split())


# --- Landmark table --------------------------------------------------------------------


def test_pitch_size() -> None:
    """The pitch is 105 x 68 m."""
    assert (PITCH_LENGTH, PITCH_WIDTH) == (105.0, 68.0)
    assert (HALF_LENGTH, HALF_WIDTH) == (52.5, 34.0)


def test_landmark_table_has_35_ids_in_order() -> None:
    """The table has the ids 1 to 35, in this order."""
    assert len(LANDMARKS) == 35
    assert [lm.id for lm in LANDMARKS] == list(range(1, 36))
    assert sorted(LANDMARK_BY_ID) == list(range(1, 36))
    assert all(LANDMARK_BY_ID[lm.id] is lm for lm in LANDMARKS)


def test_landmark_names_and_positions_are_unique() -> None:
    """No two landmarks share a name or a position."""
    assert len({lm.name for lm in LANDMARKS}) == 35
    assert len({(lm.x, lm.y) for lm in LANDMARKS}) == 35


def test_landmark_table_is_read_only() -> None:
    """The table is a tuple and a landmark cannot be changed."""
    assert isinstance(LANDMARKS, tuple)
    landmark = Landmark(id=99, name="test mark", x=1.0, y=2.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        landmark.x = 0.0


@pytest.mark.parametrize(
    ("landmark_id", "x", "y"),
    [
        pytest.param(1, -52.5, -34.0, id="far-left corner"),
        pytest.param(8, -52.5, 34.0, id="near-left corner"),
        pytest.param(11, -41.5, 0.0, id="left penalty spot"),
        pytest.param(16, 0.0, -34.0, id="halfway line, far end"),
        pytest.param(18, 0.0, 0.0, id="centre spot"),
        pytest.param(20, 0.0, 34.0, id="halfway line, near end"),
        pytest.param(25, 41.5, 0.0, id="right penalty spot"),
        pytest.param(28, 52.5, -34.0, id="far-right corner"),
        pytest.param(35, 52.5, 34.0, id="near-right corner"),
    ],
)
def test_known_landmark_positions(landmark_id: int, x: float, y: float) -> None:
    """Well-known marks have their ids and their positions in metres."""
    landmark = LANDMARK_BY_ID[landmark_id]
    assert (landmark.x, landmark.y) == pytest.approx((x, y))


def test_every_landmark_has_the_name_and_the_position_of_the_hand_written_table() -> None:
    """The whole table is pinned: 35 names and 35 positions from the Laws of the Game.

    The synthetic scene and the prompt legend read the same table, so no other test can see a
    wrong goal width or two swapped names: both sides of every comparison would move together.
    """
    assert [landmark.name for landmark in LANDMARKS] == [name for name, _, _ in LANDMARKS_BY_HAND]
    by_hand = np.array([[x, y] for _, x, y in LANDMARKS_BY_HAND])
    np.testing.assert_allclose(all_xy(), by_hand, rtol=0.0, atol=1e-9)
    for landmark, (name, x, y) in zip(LANDMARKS, LANDMARKS_BY_HAND, strict=True):
        assert (landmark.x, landmark.y) == pytest.approx((x, y), abs=1e-9), name


def test_landmark_names_say_which_marking_the_mark_belongs_to() -> None:
    """The words of a name fix the distances of the mark: the kind of area, the line and the post."""
    for lm in LANDMARKS:
        name = lm.name
        if "penalty area" in name:
            assert abs(lm.y) == pytest.approx(20.16), name
        if "goal area" in name:
            assert abs(lm.y) == pytest.approx(9.16), name
        if "post base" in name:
            assert (abs(lm.x), abs(lm.y)) == pytest.approx((52.5, 3.66)), name
        if "on the goal line" in name:
            assert abs(lm.x) == pytest.approx(52.5), name
        if "front corner" in name:
            assert abs(lm.x) == pytest.approx(36.0 if "penalty area" in name else 47.0), name
        if "penalty spot" in name:
            assert (abs(lm.x), lm.y) == pytest.approx((41.5, 0.0)), name
        if "penalty arc" in name:
            assert (abs(lm.x), abs(lm.y)) == pytest.approx((36.0, ARC)), name
        if "halfway line" in name or "centre" in name:
            assert lm.x == 0.0, name
        if "centre circle" in name:
            assert abs(lm.y) == pytest.approx(9.15), name
        if name.endswith("corner") and "front" not in name:
            assert (abs(lm.x), abs(lm.y)) == pytest.approx((52.5, 34.0)), name
        if name.endswith("end"):
            assert abs(lm.y) == pytest.approx(34.0), name
    # The four kinds of marks on a goal line, from the corner inwards: penalty area, goal area, post.
    assert [LANDMARK_BY_ID[i].name.split(",")[0] for i in (2, 3, 4)] == [
        "left penalty area",
        "left goal area",
        "left goal",
    ]


def test_landmarks_are_symmetric_about_both_axes() -> None:
    """The mirror image of the table, left to right or far to near, is the same set of points."""
    xy = all_xy()
    assert same_point_set(xy * np.array([-1.0, 1.0]), xy)
    assert same_point_set(xy * np.array([1.0, -1.0]), xy)


def test_landmarks_are_ordered_left_to_right_then_far_to_near() -> None:
    """Ids grow with x, and with y for marks on the same x. Far (negative y) comes first."""
    positions = [(lm.x, lm.y) for lm in LANDMARKS]
    assert positions == sorted(positions)


def test_landmark_names_follow_the_sign_conventions() -> None:
    """A "far" mark has negative y, "near" positive y, "left" negative x and "right" positive x."""
    for lm in LANDMARKS:
        words = name_words(lm.name)
        assert ("far" in words) == (lm.y < 0), lm.name
        assert ("near" in words) == (lm.y > 0), lm.name
        assert ("left" in words) == (lm.x < 0), lm.name
        assert ("right" in words) == (lm.x > 0), lm.name


def test_penalty_arc_crossings() -> None:
    """The four arc crossings are at |x| = 36.0 and |y| close to 7.31, far ones with negative y."""
    crossings = [lm for lm in LANDMARKS if "penalty arc" in lm.name]
    assert sorted(lm.id for lm in crossings) == list(ARC_CROSSING_IDS)
    for lm in crossings:
        assert abs(lm.x) == pytest.approx(36.0)
        assert abs(lm.y) == pytest.approx(7.31, abs=0.01)
        assert abs(lm.y) == pytest.approx(ARC_HALF_CHORD)
    left_far, left_near, right_far, right_near = (LANDMARK_BY_ID[i] for i in ARC_CROSSING_IDS)
    assert left_far.x < 0 and left_far.y < 0
    assert left_near.x < 0 and left_near.y > 0
    assert right_far.x > 0 and right_far.y < 0
    assert right_near.x > 0 and right_near.y > 0


def test_arc_crossings_are_one_radius_away_from_the_penalty_spot() -> None:
    """Each arc crossing is 9.15 m away from the penalty spot of its side."""
    assert ARC_HALF_CHORD == pytest.approx(math.sqrt(9.15**2 - 5.5**2))
    left_spot, right_spot = landmark_xy([11, 25])
    for landmark_id, spot in ((13, left_spot), (14, left_spot), (22, right_spot), (23, right_spot)):
        crossing = landmark_xy([landmark_id])[0]
        assert float(np.linalg.norm(crossing - spot)) == pytest.approx(CIRCLE_RADIUS)


# --- landmark_xy and the legend ------------------------------------------------------------


def test_landmark_xy_shape_type_and_order() -> None:
    """The result has shape (n, 2), type float64, and follows the order of the given ids."""
    xy = landmark_xy([35, 18, 1])
    assert xy.shape == (3, 2)
    assert xy.dtype == np.float64
    np.testing.assert_allclose(xy, [[52.5, 34.0], [0.0, 0.0], [-52.5, -34.0]])


def test_landmark_xy_of_all_ids_matches_the_table() -> None:
    """Column 0 is x and column 1 is y."""
    xy = all_xy()
    assert xy.shape == (35, 2)
    np.testing.assert_array_equal(xy, [[lm.x, lm.y] for lm in LANDMARKS])


def test_landmark_xy_keeps_repeated_ids() -> None:
    """An id that is given twice gives two rows."""
    np.testing.assert_array_equal(landmark_xy([18, 18]), np.zeros((2, 2)))


def test_landmark_xy_of_no_ids_is_empty() -> None:
    """No ids give an empty array that still has two columns."""
    xy = landmark_xy([])
    assert xy.shape == (0, 2)
    assert xy.dtype == np.float64


@pytest.mark.parametrize("unknown_id", [0, 36, -1])
def test_landmark_xy_rejects_unknown_ids(unknown_id: int) -> None:
    """An id outside 1 to 35 raises KeyError."""
    with pytest.raises(KeyError):
        landmark_xy([1, unknown_id])


def test_landmark_legend_text_has_one_line_per_landmark() -> None:
    """The legend lists every landmark as "<id>: <name>", in id order."""
    lines = landmark_legend_text().splitlines()
    assert lines == [f"{lm.id}: {lm.name}" for lm in LANDMARKS]
    assert lines[0] == "1: far-left corner"
    assert lines[-1] == "35: near-right corner"


# --- inside_pitch -----------------------------------------------------------------------------


def test_all_landmarks_are_inside_the_pitch() -> None:
    """Every landmark is on the pitch, lines included."""
    assert inside_pitch(all_xy()).all()


def test_inside_pitch_without_margin() -> None:
    """Points on the lines are inside. Points beyond a line are outside."""
    points = np.array(
        [
            [0.0, 0.0],  # centre spot
            [52.5, 34.0],  # near-right corner
            [-52.5, -34.0],  # far-left corner
            [53.0, 0.0],  # behind the right goal line
            [0.0, -34.5],  # beyond the far touchline
            [-60.0, 40.0],  # outside on both axes
        ]
    )
    mask = inside_pitch(points)
    assert mask.shape == (6,)
    assert mask.dtype == np.bool_
    assert mask.tolist() == [True, True, True, False, False, False]


def test_inside_pitch_with_a_margin() -> None:
    """The margin enlarges the pitch on every side."""
    points = np.array(
        [
            [54.5, 0.0],  # 2 m behind the right goal line
            [0.0, -36.0],  # 2 m beyond the far touchline
            [-54.5, 36.0],  # 2 m outside on both axes
            [56.0, 0.0],  # 3.5 m behind the right goal line
            [0.0, 38.0],  # 4 m beyond the near touchline
        ]
    )
    assert inside_pitch(points, margin_m=1.0).tolist() == [False, False, False, False, False]
    assert inside_pitch(points, margin_m=3.0).tolist() == [True, True, True, False, False]
    assert inside_pitch(points, margin_m=5.0).tolist() == [True, True, True, True, True]


def test_inside_pitch_margin_edge_is_inside() -> None:
    """A point exactly on the enlarged edge counts as inside."""
    assert inside_pitch(np.array([[55.5, 37.0], [-55.5, -37.0]]), margin_m=3.0).tolist() == [True, True]


def test_inside_pitch_is_false_for_a_missing_position() -> None:
    """A NaN position is not inside the pitch."""
    assert inside_pitch(np.array([[np.nan, 0.0], [0.0, np.nan]]), margin_m=3.0).tolist() == [False, False]


def test_inside_pitch_takes_one_point_and_no_points() -> None:
    """One point of shape (2,) gives a mask of one value. An input without points gives an empty mask."""
    assert inside_pitch(np.array([52.5, 34.0])).tolist() == [True]
    assert inside_pitch(np.array([53.0, 0.0])).tolist() == [False]
    assert inside_pitch([(0.0, 0.0), (60.0, 0.0)]).tolist() == [True, False]
    for empty in (np.zeros((0, 2)), np.array([]), []):
        mask = inside_pitch(empty)
        assert mask.shape == (0,)
        assert mask.dtype == np.bool_


#: Arrays that are not a list of (x, y) points. A plain reshape would cut the values of some of them into pairs.
NOT_POINTS = [
    pytest.param(np.zeros((4, 3)), id="rows of three values"),
    pytest.param(np.zeros((2, 6)), id="x row and y row"),
    pytest.param(np.zeros((3, 2, 2)), id="three dimensions"),
    pytest.param(np.zeros(4), id="flat list of four values"),
    pytest.param(np.zeros(3), id="flat list of three values"),
    pytest.param(np.float64(1.0), id="one number"),
]


@pytest.mark.parametrize("not_points", NOT_POINTS)
def test_inside_pitch_rejects_arrays_that_are_not_points(not_points: np.ndarray) -> None:
    """Only shape (n, 2), or (2,) for one point, is a list of points."""
    with pytest.raises(ValueError, match="shape"):
        inside_pitch(not_points)


# --- Markings ---------------------------------------------------------------------------------


@pytest.mark.parametrize("arc_samples", [8, 64, 200])
def test_pitch_polylines_stay_inside_the_pitch(arc_samples: int) -> None:
    """Every marking is an (n, 2) float64 array of points on the pitch."""
    lines = pitch_polylines(arc_samples)
    assert lines
    for line in lines:
        assert line.points.ndim == 2
        assert line.points.shape[0] >= 2
        assert line.points.shape[1] == 2
        assert line.points.dtype == np.float64
        assert inside_pitch(line.points, margin_m=1e-9).all()


@pytest.mark.parametrize("arc_samples", [7, 3, 2, 0, -8])
def test_pitch_polylines_need_enough_arc_samples(arc_samples: int) -> None:
    """Fewer than 8 samples do not give a circle and two arcs, so they are an error."""
    with pytest.raises(ValueError, match="arc_samples"):
        pitch_polylines(arc_samples)


def test_pitch_polylines_arc_sample_counts() -> None:
    """The centre circle has ``arc_samples`` points and each penalty arc has half as many."""
    circles = [line for line in pitch_polylines(8) if line.closed and line.points.shape == (8, 2)]
    assert len(circles) == 1
    np.testing.assert_allclose(np.hypot(circles[0].points[:, 0], circles[0].points[:, 1]), CIRCLE_RADIUS)
    assert [arc.points.shape for arc in penalty_arcs(8)] == [(4, 2), (4, 2)]
    assert [arc.points.shape for arc in penalty_arcs(64)] == [(32, 2), (32, 2)]


def test_polylines_compare_by_identity() -> None:
    """Two polylines are equal only when they are the same object, and a polyline can be a set member.

    The points are an array, so a comparison by value would raise an error.
    """
    first, second = pitch_polylines()[0], pitch_polylines()[0]
    np.testing.assert_array_equal(first.points, second.points)
    assert first == first
    assert first != second
    assert len({first, second, first}) == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.closed = False


def test_pitch_polylines_reach_all_four_sides() -> None:
    """The markings span the full length and the full width of the pitch."""
    points = np.vstack([line.points for line in pitch_polylines()])
    np.testing.assert_allclose(points.min(axis=0), [-52.5, -34.0])
    np.testing.assert_allclose(points.max(axis=0), [52.5, 34.0])


def test_touchlines_and_goal_lines_are_one_closed_polyline() -> None:
    """Exactly one closed polyline joins the four corners of the pitch."""
    corners = landmark_xy([1, 8, 28, 35])
    outlines = [
        line
        for line in pitch_polylines()
        if line.closed and line.points.shape == (4, 2) and same_point_set(line.points, corners)
    ]
    assert len(outlines) == 1


def test_every_landmark_lies_on_a_marking() -> None:
    """Each landmark is on a drawn line or on a spot, so a model can point at it."""
    lines = pitch_polylines()
    spots = pitch_spots()
    for lm in LANDMARKS:
        point = np.array([lm.x, lm.y])
        to_line = min(distance_to_polyline(point, line) for line in lines)
        to_spot = float(np.linalg.norm(spots - point, axis=1).min())
        assert min(to_line, to_spot) < 1e-6, lm.name


def test_penalty_arcs_end_at_the_arc_crossings() -> None:
    """There is one arc per side, and its two ends are the arc crossing landmarks."""
    arcs = penalty_arcs()
    assert len(arcs) == 2
    assert sorted(float(np.sign(arc.points[0, 0])) for arc in arcs) == [-1.0, 1.0]
    ends = np.vstack([arc.points[[0, -1]] for arc in arcs])
    assert same_point_set(ends, landmark_xy(ARC_CROSSING_IDS), tolerance=1e-6)


def test_penalty_arcs_are_outside_the_penalty_areas() -> None:
    """An arc is on the midfield side of the penalty-area line, 9.15 m around the penalty spot."""
    arcs = penalty_arcs()
    assert len(arcs) == 2
    for arc in arcs:
        from_centre_line = np.abs(arc.points[:, 0])
        assert (from_centre_line <= 36.0 + 1e-9).all()
        assert float(from_centre_line.min()) == pytest.approx(41.5 - CIRCLE_RADIUS, abs=0.05)
        to_penalty_spot = np.hypot(from_centre_line - 41.5, arc.points[:, 1])
        np.testing.assert_allclose(to_penalty_spot, CIRCLE_RADIUS)


def test_pitch_spots_are_the_centre_spot_and_the_penalty_spots() -> None:
    """The three spots are the landmarks 11, 18 and 25."""
    spots = pitch_spots()
    assert spots.shape == (3, 2)
    assert same_point_set(spots, landmark_xy([11, 18, 25]))


# --- Canvas -----------------------------------------------------------------------------------


def test_canvas_size() -> None:
    """The canvas is the pitch plus the margin on every side, times the scale."""
    canvas = PitchCanvas(scale=10.0, margin_m=4.0)
    assert (canvas.width, canvas.height) == (1130, 760)
    assert (DIAGRAM_CANVAS.width, DIAGRAM_CANVAS.height) == (1638, 1120)


def test_canvas_puts_far_left_at_the_top_left_and_near_right_at_the_bottom_right() -> None:
    """The far-left corner maps near pixel (0, 0), the near-right corner near (width, height)."""
    canvas = PitchCanvas(scale=10.0, margin_m=4.0)
    pixels = canvas.to_px(landmark_xy([1, 35, 18]))
    assert pixels.shape == (3, 2)
    assert pixels.dtype == np.float64
    far_left, near_right, centre = pixels
    np.testing.assert_allclose(far_left, [40.0, 40.0])
    np.testing.assert_allclose(near_right, [1090.0, 720.0])
    np.testing.assert_allclose(centre, [canvas.width / 2.0, canvas.height / 2.0])
    assert far_left[0] < canvas.width / 4 and far_left[1] < canvas.height / 4
    assert near_right[0] > 3 * canvas.width / 4 and near_right[1] > 3 * canvas.height / 4


def test_canvas_axes_follow_the_tv_picture() -> None:
    """Pitch +x goes right in the picture, and pitch +y (towards the camera) goes down."""
    canvas = PitchCanvas(scale=8.0, margin_m=5.0)
    origin, one_metre_right, one_metre_near = canvas.to_px(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    np.testing.assert_allclose(one_metre_right - origin, [8.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(one_metre_near - origin, [0.0, 8.0], atol=1e-9)


def test_canvas_takes_one_point_and_no_points() -> None:
    """One point of shape (2,) gives one pixel row. An input without points gives an array of shape (0, 2)."""
    canvas = PitchCanvas(scale=10.0, margin_m=4.0)
    one = canvas.to_px(np.array([-52.5, -34.0]))
    assert one.shape == (1, 2)
    np.testing.assert_allclose(one[0], [40.0, 40.0])
    np.testing.assert_allclose(canvas.to_px([(52.5, 34.0)]), [[1090.0, 720.0]])
    for empty in (np.zeros((0, 2)), np.array([]), []):
        pixels = canvas.to_px(empty)
        assert pixels.shape == (0, 2)
        assert pixels.dtype == np.float64


@pytest.mark.parametrize("not_points", NOT_POINTS)
def test_canvas_rejects_arrays_that_are_not_points(not_points: np.ndarray) -> None:
    """Only shape (n, 2), or (2,) for one point, is a list of points."""
    with pytest.raises(ValueError, match="shape"):
        PitchCanvas().to_px(not_points)


def test_canvas_draw_returns_a_bgr_image_with_the_markings() -> None:
    """The drawing has the canvas size, the background colour outside and line colour on a line."""
    canvas = PitchCanvas(scale=10.0, margin_m=4.0)
    background, line = (10, 20, 30), (200, 150, 100)
    image = canvas.draw(background_bgr=background, line_bgr=line)
    assert image.shape == (canvas.height, canvas.width, 3)
    assert image.dtype == np.uint8
    assert image[0, 0].tolist() == list(background)
    # A point on the far touchline. It is in the top quarter of the picture.
    u, v = np.round(canvas.to_px(np.array([[-26.5, -34.0]]))[0]).astype(int)
    assert v < canvas.height / 4
    assert int(np.abs(image[v, u].astype(int) - np.array(line)).max()) <= 5


# --- Landmark diagram ---------------------------------------------------------------------------


def test_landmark_diagram_default_is_a_bgr_image() -> None:
    """Without a canvas the diagram is a landscape uint8 image with three channels."""
    image = draw_landmark_diagram()
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.shape[1] > image.shape[0]
    assert image.dtype == np.uint8


@pytest.mark.parametrize("canvas", [DIAGRAM_CANVAS, PitchCanvas(scale=8.0, margin_m=5.0)])
def test_landmark_diagram_has_the_canvas_size(canvas: PitchCanvas) -> None:
    """The diagram has the height and the width of the given canvas."""
    image = draw_landmark_diagram(canvas)
    assert image.shape == (canvas.height, canvas.width, 3)
    assert image.dtype == np.uint8


def test_landmark_diagram_marks_every_landmark_with_a_red_dot() -> None:
    """The pixel at each landmark is red. Images are BGR, so red is the last channel."""
    image = draw_landmark_diagram(DIAGRAM_CANVAS)
    pixels = np.round(DIAGRAM_CANVAS.to_px(all_xy())).astype(int)
    for lm, (u, v) in zip(LANDMARKS, pixels, strict=True):
        blue, green, red = (int(channel) for channel in image[v, u])
        assert red > 200 and blue < 60 and green < 60, lm.name


def test_landmark_diagram_has_one_red_dot_at_the_place_of_every_landmark() -> None:
    """The red dots are found in the picture and compared with the table, not the other way round.

    The expected pixel of a landmark is written here from the convention: the left goal is at the
    left, the far touchline is at the top, 14 pixels are one metre and the margin is 6 m. A dot
    that is drawn at the mirrored place is a dot at a place where the table has no landmark with
    that number, and the model would then answer wrong ids for every frame.
    """
    image = draw_landmark_diagram(DIAGRAM_CANVAS)
    red = (image[:, :, 2] > 200) & (image[:, :, 1] < 60) & (image[:, :, 0] < 60)
    count, _, stats, centres = cv2.connectedComponentsWithStats(red.astype(np.uint8), connectivity=8)
    dots = np.array([centres[index] for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= 30])
    assert len(dots) == 35

    expected = np.array([[(x + 52.5 + 6.0) * 14.0, (y + 34.0 + 6.0) * 14.0] for _, x, y in LANDMARKS_BY_HAND])
    distances = np.linalg.norm(expected[:, None, :] - dots[None, :, :], axis=2)
    assert distances.min(axis=1).max() < 1.5  # every landmark has its dot
    assert distances.min(axis=0).max() < 1.5  # every dot belongs to a landmark

    height, width = image.shape[:2]
    far_left_corner, near_right_corner = expected[0], expected[34]
    assert far_left_corner[0] < width / 2 and far_left_corner[1] < height / 2  # landmark 1 is top left
    assert near_right_corner[0] > width / 2 and near_right_corner[1] > height / 2  # landmark 35 is bottom right


@pytest.mark.parametrize("landmark_id", [1, 2, 9, 13, 16, 24, 31, 35])
def test_landmark_diagram_writes_the_number_next_to_its_dot(landmark_id: int) -> None:
    """The label box of a landmark holds the digits of its own id.

    The digits are compared with digits drawn here with the same font. A label on the wrong side
    of the dot, or with the number of another landmark, differs in many pixels.
    """
    image = draw_landmark_diagram(DIAGRAM_CANVAS)
    landmark = LANDMARK_BY_ID[landmark_id]
    u, v = (int(round(value)) for value in DIAGRAM_CANVAS.to_px(np.array([[landmark.x, landmark.y]]))[0])
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = str(landmark_id)
    (text_w, text_h), _ = cv2.getTextSize(label, font, 0.6, 2)
    # Left of the centre the number stands right of the dot. Far marks have it below, near marks above.
    text_u = u + 9 if landmark.x <= 0 else u - 9 - text_w
    text_v = v - 8 if landmark.y > 0 else v + 8 + text_h
    box = image[text_v - text_h - 2 : text_v + 4, text_u - 2 : text_u + text_w + 3]

    drawn = np.full(box.shape, 255, dtype=np.uint8)
    cv2.putText(drawn, label, (2, text_h + 2), font, 0.6, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    other = np.full(box.shape, 255, dtype=np.uint8)
    cv2.putText(other, str(landmark_id % 35 + 1), (2, text_h + 2), font, 0.6, (0, 0, 0), 2, lineType=cv2.LINE_AA)

    dark = box.max(axis=2) < 96
    assert dark.sum() >= 10
    same = int((dark == (drawn.max(axis=2) < 96)).sum())
    assert same > 0.97 * dark.size
    assert same > int((dark == (other.max(axis=2) < 96)).sum())


def test_landmark_diagram_writes_the_numbers() -> None:
    """The diagram adds white label boxes with dark digits to the plain pitch drawing.

    A dark pixel has all three channels below 96. The count does not ask for pixels that are
    exactly black, because some OpenCV versions draw the digits with smooth grey edges.
    """
    plain = DIAGRAM_CANVAS.draw()
    image = draw_landmark_diagram(DIAGRAM_CANVAS)
    white_before = int((plain == 255).all(axis=2).sum())
    white_after = int((image == 255).all(axis=2).sum())
    dark_before = int((plain.max(axis=2) < 96).sum())
    dark_after = int((image.max(axis=2) < 96).sum())
    assert dark_before == 0
    assert dark_after >= 35 * 10
    assert white_after >= white_before + 35 * 50


def test_landmark_diagram_keeps_dots_and_numbers_inside_the_picture() -> None:
    """Nothing is drawn in the outer 50 pixels: dots and numbers are not cut by the border."""
    plain = DIAGRAM_CANVAS.draw()
    image = draw_landmark_diagram(DIAGRAM_CANVAS)
    background = plain[0, 0]
    border = 50
    for band in (image[:border], image[-border:], image[:, :border], image[:, -border:]):
        assert (band == background).all()
