"""Tests for the shared records of ``pitchvlm.schemas``.

Conventions under test: image y grows downwards, so the foot point of a box is
on its bottom edge. Records are frozen, and a changed copy is validated like a
new record. Sequence fields are tuples. A shirt number is 0 to 99 in every
record. A homography matrix is stored row by row and maps image pixels to
pitch metres.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import pytest
from pydantic import ValidationError

from pitchvlm.schemas import (
    ROLES,
    TEAM_ROLES,
    Box,
    ClipInfo,
    ClipResult,
    FrameLandmarks,
    FramePeople,
    FrameState,
    GoalkeeperKit,
    HomographyRecord,
    KitDescription,
    KitLegend,
    LandmarkObs,
    NumberReading,
    PersonDet,
    PlayerState,
    Record,
    Track,
    TrackObs,
)

#: The identity mapping, as the nested tuples that ``HomographyRecord.matrix`` stores.
IDENTITY_ROWS = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def make_box(x1: float = 100.0, y1: float = 50.0, x2: float = 140.0, y2: float = 150.0) -> Box:
    """Return a valid box. The default box is 40 px wide and 100 px high."""
    return Box(x1=x1, y1=y1, x2=x2, y2=y2)


def make_obs(frame: int) -> TrackObs:
    """Return one observation of a team A player at the given frame."""
    return TrackObs(frame=frame, box=make_box(), role="player", team_key="A")


def make_track(frames: Sequence[int] = (0, 12, 24)) -> Track:
    """Return a track with one observation per given frame."""
    return Track(track_id=3, observations=tuple(make_obs(frame) for frame in frames))


def make_legend() -> KitLegend:
    """Return a kit legend with three goalkeeper kits and a referee kit."""
    return KitLegend(
        team_a=KitDescription(shirt="red", shorts="white", socks="red"),
        team_b=KitDescription(shirt="sky blue", shorts="navy", socks="navy"),
        goalkeepers=(
            GoalkeeperKit(kit="all yellow", team_key="A"),
            GoalkeeperKit(kit="all green", team_key="B"),
            GoalkeeperKit(kit="orange shirt, black shorts"),
        ),
        referee_kit="black shirt, black shorts",
    )


def player_state(**fields: object) -> PlayerState:
    """Return the state of a player at the centre spot. Keyword arguments set further fields."""
    return PlayerState(track_id=0, pitch_x=0.0, pitch_y=0.0, role="player", **fields)


#: Every record field that holds a shirt number. Each entry builds a record with the given number
#: in that field and returns the number that the record stores.
SHIRT_NUMBER_FIELDS: dict[str, Callable[[int], int | None]] = {
    "PersonDet.number": lambda number: PersonDet(box=make_box(), role="player", number=number).number,
    "NumberReading.number": lambda number: NumberReading(number=number).number,
    "NumberReading.per_crop": lambda number: NumberReading(per_crop=(None, number)).per_crop[1],
    "TrackObs.number": lambda number: TrackObs(frame=0, box=make_box(), role="player", number=number).number,
    "Track.number_export": lambda number: Track(
        track_id=0, observations=(make_obs(0),), number_export=number
    ).number_export,
    "Track.number_display": lambda number: Track(
        track_id=0, observations=(make_obs(0),), number_display=number
    ).number_display,
    "PlayerState.number_export": lambda number: player_state(number_export=number).number_export,
    "PlayerState.number_display": lambda number: player_state(number_display=number).number_display,
}


def sample_records() -> dict[str, Record]:
    """Return one valid instance of every record type, keyed by the class name."""
    box = make_box()
    info = ClipInfo(clip_id="clip-001", n_frames=25, fps=25.0, width=1920, height=1080)
    records: list[Record] = [
        box,
        PersonDet(box=box, role="player", team_key="A", number=7),
        FramePeople(frame=0),
        LandmarkObs(landmark_id=18, x=960.0, y=540.0),
        FrameLandmarks(frame=0),
        KitDescription(shirt="red", shorts="white", socks="red"),
        GoalkeeperKit(kit="all yellow", team_key="A"),
        make_legend(),
        NumberReading(),
        HomographyRecord(frame=0),
        make_obs(0),
        make_track(),
        PlayerState(track_id=3, pitch_x=-10.0, pitch_y=-20.0, role="player"),
        FrameState(frame=0),
        info,
        ClipResult(info=info, frames=(), tracks=(), homographies=()),
    ]
    return {type(record).__name__: record for record in records}


# --- Box -----------------------------------------------------------------------


def test_box_size_and_centre() -> None:
    """A box reports its width, its height and its centre in pixels."""
    box = make_box()
    assert box.width == 40.0
    assert box.height == 100.0
    assert box.center == (120.0, 100.0)


def test_box_foot_is_the_bottom_centre() -> None:
    """The foot point is the middle of the bottom edge, which is the larger y value."""
    box = make_box()
    foot_x, foot_y = box.foot
    assert (foot_x, foot_y) == (120.0, 150.0)
    assert foot_x == box.center[0]
    assert foot_y == box.y2
    assert foot_y > box.center[1]


@pytest.mark.parametrize(
    ("x1", "y1", "x2", "y2"),
    [
        pytest.param(10.0, 10.0, 10.0, 20.0, id="no width"),
        pytest.param(10.0, 10.0, 20.0, 10.0, id="no height"),
        pytest.param(20.0, 10.0, 10.0, 20.0, id="x order swapped"),
        pytest.param(10.0, 20.0, 20.0, 10.0, id="y order swapped"),
    ],
)
def test_box_without_area_is_rejected(x1: float, y1: float, x2: float, y2: float) -> None:
    """A box needs x1 < x2 and y1 < y2."""
    with pytest.raises(ValidationError, match="no area"):
        Box(x1=x1, y1=y1, x2=x2, y2=y2)


def test_box_rejects_missing_and_unknown_fields() -> None:
    """A box has exactly the four edge fields."""
    with pytest.raises(ValidationError):
        Box(x1=0.0, y1=0.0, x2=10.0)
    with pytest.raises(ValidationError):
        Box(x1=0.0, y1=0.0, x2=10.0, y2=10.0, score=0.9)


@pytest.mark.parametrize("field", ["x1", "y1", "x2", "y2"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "minus inf"])
def test_box_rejects_values_that_are_not_finite(field: str, value: float) -> None:
    """A box edge must be a real number. NaN would pass the area check, because every comparison with it is false."""
    edges = {"x1": 100.0, "y1": 50.0, "x2": 140.0, "y2": 150.0}
    edges[field] = value
    with pytest.raises(ValidationError):
        Box(**edges)


def test_iou_of_identical_boxes_is_one() -> None:
    """A box overlaps fully with an equal box."""
    assert make_box().iou(make_box()) == pytest.approx(1.0)


def test_iou_of_separate_or_touching_boxes_is_zero() -> None:
    """Boxes that are apart, or that share only an edge, have no overlap."""
    box = make_box(0.0, 0.0, 10.0, 10.0)
    far_away = make_box(50.0, 50.0, 60.0, 60.0)
    touching = make_box(10.0, 0.0, 20.0, 10.0)
    assert box.iou(far_away) == 0.0
    assert box.iou(touching) == 0.0


@pytest.mark.parametrize(
    ("other", "expected"),
    [
        pytest.param((5.0, 5.0, 15.0, 15.0), 25.0 / 175.0, id="corner overlap"),
        pytest.param((5.0, 0.0, 15.0, 10.0), 50.0 / 150.0, id="half overlap along x"),
        pytest.param((0.0, 5.0, 10.0, 15.0), 50.0 / 150.0, id="half overlap along y"),
        pytest.param((2.0, 2.0, 6.0, 6.0), 16.0 / 100.0, id="other box inside"),
    ],
)
def test_iou_known_values(other: tuple[float, float, float, float], expected: float) -> None:
    """The IoU of a 10 x 10 box with a second box matches the value computed by hand."""
    box = make_box(0.0, 0.0, 10.0, 10.0)
    other_box = make_box(*other)
    assert box.iou(other_box) == pytest.approx(expected)
    assert other_box.iou(box) == pytest.approx(expected)


def test_iou_is_symmetric_and_between_zero_and_one() -> None:
    """Random boxes give an IoU in [0, 1] that is the same in both directions."""
    rng = np.random.default_rng(20261113)
    for _ in range(200):
        ax, ay, bx, by = rng.uniform(0.0, 100.0, size=4).tolist()
        aw, ah, bw, bh = rng.uniform(1.0, 50.0, size=4).tolist()
        first = make_box(ax, ay, ax + aw, ay + ah)
        second = make_box(bx, by, bx + bw, by + bh)
        iou = first.iou(second)
        assert 0.0 <= iou <= 1.0
        assert iou == pytest.approx(second.iou(first))
        assert first.iou(first) == pytest.approx(1.0)


# --- Record behaviour ------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "field", "value"),
    [
        ("Box", "x1", 5.0),
        ("PersonDet", "role", "referee"),
        ("FramePeople", "frame", 3),
        ("LandmarkObs", "x", 1.0),
        ("FrameLandmarks", "view", "left"),
        ("KitDescription", "shirt", "blue"),
        ("GoalkeeperKit", "kit", "all green"),
        ("KitLegend", "referee_kit", "pink"),
        ("NumberReading", "number", 9),
        ("HomographyRecord", "frame", 4),
        ("TrackObs", "frame", 99),
        ("Track", "role", "player"),
        ("PlayerState", "pitch_y", 0.0),
        ("FrameState", "frame", 3),
        ("ClipInfo", "fps", 30.0),
        ("ClipResult", "engines", {"people": "code"}),
    ],
)
def test_frozen_records_reject_assignment(name: str, field: str, value: object) -> None:
    """A field of a record cannot be assigned, and the record keeps its old value."""
    record = sample_records()[name]
    before = getattr(record, field)
    with pytest.raises(ValidationError, match="frozen"):
        setattr(record, field, value)
    assert getattr(record, field) == before


def test_model_copy_makes_a_new_record() -> None:
    """An update goes through ``model_copy`` and leaves the first record as it was."""
    track = make_track()
    voted = track.model_copy(update={"role": "player", "team_key": "A"})
    assert voted is not track
    assert (voted.role, voted.team_key) == ("player", "A")
    assert (track.role, track.team_key) == (None, None)
    assert voted.observations == track.observations


def test_model_copy_without_an_update_gives_an_equal_record() -> None:
    """A plain copy and a deep copy are new records with the same values."""
    track = make_track()
    for copied in (track.model_copy(), track.model_copy(update={}), track.model_copy(deep=True)):
        assert copied is not track
        assert copied == track


def test_model_copy_rejects_an_unknown_field_name() -> None:
    """A misspelled field name in an update is an error. Plain pydantic would store it and say nothing."""
    track = make_track()
    with pytest.raises(ValidationError):
        track.model_copy(update={"team_sid": "left"})
    with pytest.raises(ValidationError):
        make_box().model_copy(update={"x3": 5.0})


def test_model_copy_checks_the_new_values() -> None:
    """An update is validated like a new record: labels, limits and the rules of the record."""
    track = make_track()
    with pytest.raises(ValidationError):
        track.model_copy(update={"team_side": "middle"})
    with pytest.raises(ValidationError):
        track.model_copy(update={"number_export": 100})
    with pytest.raises(ValidationError, match="increasing"):
        track.model_copy(update={"observations": (make_obs(12), make_obs(0))})
    with pytest.raises(ValidationError, match="at least one observation"):
        track.model_copy(update={"observations": ()})
    with pytest.raises(ValidationError, match="needs a 3x3 matrix"):
        HomographyRecord(frame=0).model_copy(update={"source": "landmarks"})
    with pytest.raises(ValidationError, match="no area"):
        make_box().model_copy(update={"x2": 100.0})
    # A refused update leaves the first record as it was.
    assert track == make_track()


def test_model_copy_stores_a_list_as_a_tuple() -> None:
    """A list in an update becomes a tuple, as in a new record."""
    det = PersonDet(box=make_box(), role="player")
    people = FramePeople(frame=0).model_copy(update={"people": [det]})
    assert isinstance(people.people, tuple)
    assert people == FramePeople(frame=0, people=(det,))


def test_model_copy_keeps_the_fields_that_the_update_does_not_name() -> None:
    """A changed copy of a record with nested records keeps the other fields, also in a deep copy."""
    info = ClipInfo(clip_id="clip-001", n_frames=2, fps=25.0, width=1920, height=1080)
    result = ClipResult(
        info=info,
        frames=(FrameState(frame=0), FrameState(frame=1)),
        tracks=(make_track((0, 1)),),
        homographies=(HomographyRecord(frame=0), HomographyRecord(frame=1)),
        engines={"people": "code"},
    )
    for deep in (False, True):
        shorter = result.model_copy(update={"frames": result.frames[:1]}, deep=deep)
        assert shorter.frames == (FrameState(frame=0),)
        assert (shorter.info, shorter.tracks, shorter.homographies) == (info, result.tracks, result.homographies)
        assert shorter.engines == {"people": "code"}
    assert len(result.frames) == 2


def test_equal_records_compare_and_hash_equal() -> None:
    """Two records with the same values are equal and count once in a set."""
    assert make_box() == make_box()
    assert make_box() != make_box(x2=141.0)
    assert len({make_box(), make_box(), make_box(x2=141.0)}) == 2


def test_sequence_fields_are_stored_as_tuples() -> None:
    """A list given for a sequence field is stored as a tuple."""
    det = PersonDet(box=make_box(), role="player")
    people = FramePeople(frame=0, people=[det])
    landmarks = FrameLandmarks(frame=0, landmarks=[LandmarkObs(landmark_id=1, x=10.0, y=20.0)])
    reading = NumberReading(per_crop=[7, None, 7], number=7, confidence="high")
    track = Track(track_id=0, observations=[make_obs(0), make_obs(12)])
    state = FrameState(frame=0, players=[PlayerState(track_id=0, pitch_x=0.0, pitch_y=0.0, role="referee")])
    for value in (people.people, landmarks.landmarks, reading.per_crop, track.observations, state.players):
        assert isinstance(value, tuple)
    assert people.people == (det,)
    assert reading.per_crop == (7, None, 7)


def test_unknown_fields_are_rejected() -> None:
    """A record does not accept a field that it does not define."""
    with pytest.raises(ValidationError):
        PersonDet(box=make_box(), role="player", confidence=0.9)
    with pytest.raises(ValidationError):
        GoalkeeperKit(kit="all yellow", team="A")


# --- Perception answers ------------------------------------------------------------


def test_person_defaults() -> None:
    """A person has no team key and no shirt number until they are given."""
    det = PersonDet(box=make_box(), role="referee")
    assert det.team_key == "none"
    assert det.number is None


@pytest.mark.parametrize("field", sorted(SHIRT_NUMBER_FIELDS))
@pytest.mark.parametrize("number", [0, 7, 99])
def test_shirt_number_in_range_is_accepted(field: str, number: int) -> None:
    """Shirt numbers 0 to 99 are valid in every record field that holds a shirt number."""
    assert SHIRT_NUMBER_FIELDS[field](number) == number


@pytest.mark.parametrize("field", sorted(SHIRT_NUMBER_FIELDS))
@pytest.mark.parametrize("number", [-1, 100])
def test_shirt_number_out_of_range_is_rejected(field: str, number: int) -> None:
    """Shirt numbers below 0 or above 99 are an error in every record field that holds a shirt number."""
    with pytest.raises(ValidationError):
        SHIRT_NUMBER_FIELDS[field](number)


def test_shirt_number_fields_can_be_empty() -> None:
    """Every shirt-number field is None until a number is known, and a crop without a number stays None."""
    assert NumberReading(per_crop=(None, 7, None)).per_crop == (None, 7, None)
    assert make_obs(0).number is None
    assert NumberReading().number is None


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "minus inf"])
def test_landmark_position_must_be_finite(value: float) -> None:
    """A landmark point is a real pixel position."""
    with pytest.raises(ValidationError):
        LandmarkObs(landmark_id=18, x=value, y=540.0)
    with pytest.raises(ValidationError):
        LandmarkObs(landmark_id=18, x=960.0, y=value)


def test_unknown_role_team_side_or_view_is_rejected() -> None:
    """Only the listed labels are valid. The view label uses the spelling "centre"."""
    with pytest.raises(ValidationError):
        PersonDet(box=make_box(), role="striker")
    with pytest.raises(ValidationError):
        PersonDet(box=make_box(), role="player", team_key="C")
    with pytest.raises(ValidationError):
        Track(track_id=0, observations=(make_obs(0),), team_side="middle")
    with pytest.raises(ValidationError):
        FrameLandmarks(frame=0, view="center")
    assert FrameLandmarks(frame=0, view="centre").view == "centre"


def test_team_roles_are_players_and_goalkeepers() -> None:
    """Referees and other people have no team."""
    assert ROLES == ("player", "goalkeeper", "referee", "other")
    assert TEAM_ROLES == ("player", "goalkeeper")


def test_negative_frame_index_is_rejected() -> None:
    """Frame indices start at 0."""
    with pytest.raises(ValidationError):
        FramePeople(frame=-1)
    with pytest.raises(ValidationError):
        FrameLandmarks(frame=-1)
    with pytest.raises(ValidationError):
        HomographyRecord(frame=-1)
    with pytest.raises(ValidationError):
        make_obs(-1)
    with pytest.raises(ValidationError):
        FrameState(frame=-1)


# --- Kit legend ----------------------------------------------------------------------


def test_kit_legend_prompt_text() -> None:
    """The prompt text has one line per team, per goalkeeper kit and for the referees."""
    assert make_legend().prompt_text() == (
        "Team A: red shirt, white shorts, red socks.\n"
        "Team B: sky blue shirt, navy shorts, navy socks.\n"
        "Goalkeeper (team A): all yellow.\n"
        "Goalkeeper (team B): all green.\n"
        "Goalkeeper (unknown team): orange shirt, black shorts.\n"
        "Referees: black shirt, black shorts."
    )


def test_kit_legend_prompt_text_leaves_out_the_uncertain_notes() -> None:
    """The notes of the model are not pasted into prompts, so they do not change the prompt hash of later requests."""
    legend = make_legend()
    note = "the goalkeeper of team B may wear grey"
    with_note = legend.model_copy(update={"uncertain": (note,)})
    assert with_note.uncertain == (note,)
    assert with_note != legend
    assert with_note.prompt_text() == legend.prompt_text()
    assert note not in with_note.prompt_text()


def test_kit_legend_prompt_text_without_goalkeepers_and_referees() -> None:
    """A legend with the two teams only gives two lines."""
    legend = KitLegend(
        team_a=KitDescription(shirt="white", shorts="white", socks="white"),
        team_b=KitDescription(shirt="black", shorts="black", socks="black"),
    )
    assert (legend.goalkeepers, legend.referee_kit, legend.uncertain) == ((), "", ())
    text = legend.prompt_text()
    assert text.splitlines() == [
        "Team A: white shirt, white shorts, white socks.",
        "Team B: black shirt, black shorts, black socks.",
    ]
    assert "Goalkeeper" not in text
    assert "Referees" not in text


# --- Homography record -----------------------------------------------------------------


def test_homography_record_without_mapping() -> None:
    """The default record says that the frame has no mapping."""
    record = HomographyRecord(frame=3)
    assert record.source == "none"
    assert record.matrix is None
    assert record.valid is False
    assert record.n_inliers == 0
    assert record.reproj_px is None


@pytest.mark.parametrize("source", ["landmarks", "propagated"])
def test_homography_record_with_mapping(source: str) -> None:
    """A record with a 3x3 matrix and a source is valid."""
    record = HomographyRecord(frame=3, matrix=IDENTITY_ROWS, source=source, n_inliers=6, reproj_px=1.5)
    assert record.valid is True
    assert record.matrix == IDENTITY_ROWS
    assert record.source == source


def test_homography_matrix_is_stored_as_nested_tuples_of_floats() -> None:
    """Nested lists of whole numbers are stored as nested tuples of floats."""
    record = HomographyRecord(frame=0, matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], source="landmarks")
    assert isinstance(record.matrix, tuple)
    assert all(isinstance(row, tuple) and len(row) == 3 for row in record.matrix)
    assert all(isinstance(value, float) for row in record.matrix for value in row)
    assert record.matrix == IDENTITY_ROWS


def test_homography_source_none_must_not_carry_a_matrix() -> None:
    """A matrix without a source is an error."""
    with pytest.raises(ValidationError, match="must not carry a matrix"):
        HomographyRecord(frame=0, matrix=IDENTITY_ROWS)
    with pytest.raises(ValidationError, match="must not carry a matrix"):
        HomographyRecord(frame=0, matrix=IDENTITY_ROWS, source="none")


@pytest.mark.parametrize("source", ["landmarks", "propagated"])
def test_homography_source_needs_a_matrix(source: str) -> None:
    """A source without a matrix is an error."""
    with pytest.raises(ValidationError, match="needs a 3x3 matrix"):
        HomographyRecord(frame=0, source=source)


@pytest.mark.parametrize(
    "matrix",
    [
        pytest.param(IDENTITY_ROWS[:2], id="two rows"),
        pytest.param(IDENTITY_ROWS + ((0.0, 0.0, 1.0),), id="four rows"),
        pytest.param(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)), id="rows of two values"),
        pytest.param(
            ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)), id="rows of four values"
        ),
    ],
)
def test_homography_matrix_must_be_3x3(matrix: tuple[tuple[float, ...], ...]) -> None:
    """A matrix with another shape than 3x3 is an error."""
    with pytest.raises(ValidationError):
        HomographyRecord(frame=0, matrix=matrix, source="landmarks")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "minus inf"])
def test_homography_values_must_be_finite(value: float) -> None:
    """A matrix value and the reprojection error are real numbers."""
    rows = ((1.0, 0.0, 0.0), (0.0, value, 0.0), (0.0, 0.0, 1.0))
    with pytest.raises(ValidationError):
        HomographyRecord(frame=0, matrix=rows, source="landmarks")
    with pytest.raises(ValidationError):
        HomographyRecord(frame=0, matrix=IDENTITY_ROWS, source="landmarks", reproj_px=value)


def test_homography_rejects_negative_inlier_count() -> None:
    """The number of inliers cannot be negative."""
    with pytest.raises(ValidationError):
        HomographyRecord(frame=0, matrix=IDENTITY_ROWS, source="landmarks", n_inliers=-1)


def test_homography_matrix_is_row_major_and_maps_image_to_pitch() -> None:
    """The stored rows are the matrix rows, and ``H @ (x_px, y_px, 1)`` gives pitch metres."""
    # 10 pixels per metre. The image origin (top-left) is the far-left corner of the pitch.
    matrix = np.array([[0.1, 0.0, -52.5], [0.0, 0.1, -34.0], [0.0, 0.0, 1.0]])
    record = HomographyRecord(frame=0, matrix=matrix.tolist(), source="landmarks")
    stored = np.array(record.matrix, dtype=np.float64)
    assert stored.shape == (3, 3)
    np.testing.assert_array_equal(stored, matrix)
    top_left = stored @ np.array([0.0, 0.0, 1.0])
    bottom_right = stored @ np.array([1050.0, 680.0, 1.0])
    # The top of the image is the far side: negative pitch y.
    np.testing.assert_allclose(top_left[:2] / top_left[2], [-52.5, -34.0])
    np.testing.assert_allclose(bottom_right[:2] / bottom_right[2], [52.5, 34.0])


# --- Tracks ----------------------------------------------------------------------------


def test_track_first_and_last_frame() -> None:
    """A track reports the frames of its first and last observation."""
    track = make_track((5, 17, 29))
    assert track.track_id == 3
    assert track.first_frame == 5
    assert track.last_frame == 29


def test_track_with_one_observation_is_valid() -> None:
    """One observation is enough for a track."""
    track = make_track((7,))
    assert track.first_frame == 7
    assert track.last_frame == 7


def test_track_attributes_are_empty_before_the_votes() -> None:
    """Role, team, numbers and name are None on a new track."""
    track = make_track()
    assert (track.role, track.team_key, track.team_side) == (None, None, None)
    assert (track.number_export, track.number_display, track.name) == (None, None, None)


def test_track_without_observations_is_rejected() -> None:
    """A track needs at least one observation."""
    with pytest.raises(ValidationError, match="at least one observation"):
        Track(track_id=0, observations=())


@pytest.mark.parametrize("frames", [(12, 0), (0, 24, 12), (24, 12, 0)])
def test_track_rejects_unordered_frames(frames: tuple[int, ...]) -> None:
    """Observations must be in increasing frame order."""
    with pytest.raises(ValidationError, match="increasing"):
        make_track(frames)


@pytest.mark.parametrize("frames", [(0, 0), (0, 12, 12), (12, 12, 24)])
def test_track_rejects_duplicate_frames(frames: tuple[int, ...]) -> None:
    """A track has at most one observation per frame."""
    with pytest.raises(ValidationError, match="unique"):
        make_track(frames)


def test_track_rejects_negative_id() -> None:
    """Track ids start at 0."""
    with pytest.raises(ValidationError):
        Track(track_id=-1, observations=(make_obs(0),))


# --- Dense output ------------------------------------------------------------------------


def test_player_state_defaults() -> None:
    """A player state needs a track id, a pitch position and a role only."""
    state = PlayerState(track_id=1, pitch_x=-30.0, pitch_y=-12.5, role="goalkeeper")
    assert state.box is None
    assert state.team_key == "none"
    assert state.team_side is None
    assert (state.number_export, state.number_display, state.name) == (None, None, None)
    assert state.observed is False


@pytest.mark.parametrize("field", ["pitch_x", "pitch_y"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "minus inf"])
def test_player_state_position_must_be_finite(field: str, value: float) -> None:
    """A pitch position is two real numbers. A person without a position has no state."""
    position = {"pitch_x": -30.0, "pitch_y": -12.5}
    position[field] = value
    with pytest.raises(ValidationError):
        PlayerState(track_id=1, role="player", **position)
    with pytest.raises(ValidationError):
        player_state().model_copy(update={field: value})


@pytest.mark.parametrize("field", ["n_frames", "fps", "width", "height"])
@pytest.mark.parametrize("value", [0, -1])
def test_clip_info_rejects_non_positive_values(field: str, value: int) -> None:
    """Frame count, frame rate, width and height must be above zero."""
    values: dict[str, object] = {"n_frames": 750, "fps": 25.0, "width": 1920, "height": 1080}
    values[field] = value
    with pytest.raises(ValidationError):
        ClipInfo(clip_id="clip-001", **values)


def test_clip_result_holds_tuples_and_an_engine_map() -> None:
    """A clip result stores its frames, tracks and mappings as tuples."""
    info = ClipInfo(clip_id="clip-001", n_frames=2, fps=25.0, width=1920, height=1080)
    result = ClipResult(
        info=info,
        frames=[FrameState(frame=0), FrameState(frame=1)],
        tracks=[make_track((0, 1))],
        homographies=[HomographyRecord(frame=0), HomographyRecord(frame=1)],
    )
    assert isinstance(result.frames, tuple)
    assert isinstance(result.tracks, tuple)
    assert isinstance(result.homographies, tuple)
    assert [state.frame for state in result.frames] == [0, 1]
    assert result.engines == {}
    with_engines = ClipResult(info=info, frames=(), tracks=(), homographies=(), engines={"people": "code"})
    assert with_engines.engines == {"people": "code"}
