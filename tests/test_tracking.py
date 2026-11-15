"""Tests of tracking: matching between keyframes, fragment joining, votes, team side and numbers.

The scenes are hand-made, but for two tests at the end that follow the people
of the synthetic scene. The image is 1280 x 720. Foot points are given in
pixels; the test pitch maps are simple scalings, so the distances in metres
are easy to check by hand.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import yaml

from pitchvlm.config import TrackingConfig
from pitchvlm.evalx.metrics import identity_scores
from pitchvlm.fill import dense_frame_states
from pitchvlm.frames import keyframe_indices
from pitchvlm.geometry import apply_homography, matrix_to_rows
from pitchvlm.motion import MotionField, bridged_warp
from pitchvlm.roster import Roster
from pitchvlm.schemas import (
    TEAM_ROLES,
    Box,
    FramePeople,
    HomographyRecord,
    NumberReading,
    PersonDet,
    Role,
    TeamKey,
    Track,
    TrackObs,
)
from pitchvlm.synth import SyntheticScene
from pitchvlm.tracking import (
    GOAL_ZONE_MIN_X_M,
    TWO_PEOPLE_MARGIN_M,
    PitchMap,
    assign_team_sides,
    associate,
    decide_number,
    finalize_tracks,
    goal_side,
    stitch_fragments,
    vote_role_team,
)

FPS = 25.0
#: Frames between two keyframes: 0.48 s at 25 fps.
STEP = 12
BOX_HEIGHT = 40.0
IMAGE_CENTRE = np.array([640.0, 360.0])
#: The picture of the panning test camera moves 25 px to the left per frame.
PAN_PX_PER_FRAME = np.array([-25.0, 0.0])


# --- Helpers -------------------------------------------------------------------


def person(
    x: float,
    y: float,
    *,
    height: float = BOX_HEIGHT,
    role: Role = "player",
    team_key: TeamKey = "A",
    number: int | None = None,
) -> PersonDet:
    """Make a detection whose foot point is the image point (x, y)."""
    half_width = height / 4.0
    box = Box(x1=x - half_width, y1=y - height, x2=x + half_width, y2=y)
    return PersonDet(box=box, role=role, team_key=team_key, number=number)


def keyframe(frame: int, *people: PersonDet) -> FramePeople:
    """Make the model answer of one keyframe."""
    return FramePeople(frame=frame, people=people)


def still_track(
    track_id: int,
    frames: Sequence[int],
    x: float,
    *,
    height: float = BOX_HEIGHT,
    role: Role = "player",
    team_key: TeamKey = "A",
    number: int | None = None,
) -> Track:
    """Make a track of a person who stands at the image point (x, 400) in the given frames."""
    box = person(x, 400.0, height=height).box
    observations = tuple(
        TrackObs(frame=frame, box=box, role=role, team_key=team_key, number=number) for frame in frames
    )
    return Track(track_id=track_id, observations=observations)


def frames_of(track: Track) -> list[int]:
    """Return the frames of the observations of a track."""
    return [observation.frame for observation in track.observations]


def feet_x_of(track: Track) -> list[float]:
    """Return the image x of the foot point of every observation of a track."""
    return [observation.box.foot[0] for observation in track.observations]


def high_reading(number: int, crops: int = 3) -> NumberReading:
    """Make a confident shirt-number reading in which every crop shows the number."""
    return NumberReading(per_crop=(number,) * crops, number=number, confidence="high")


def no_motion(src: int, dst: int, xy: np.ndarray) -> np.ndarray:
    """Warp of a camera that does not move."""
    return xy


def no_pitch(frame: int, xy: np.ndarray) -> None:
    """Pitch map of a clip without any mapping."""
    return None


#: Arguments of ``associate`` for a camera that does not move and a clip without a pitch mapping.
STILL_CAMERA = {"fps": 25.0, "warp": no_motion, "to_pitch": no_pitch}


def broken_motion(src: int, dst: int, xy: np.ndarray) -> None:
    """Warp of a clip whose motion chain is broken."""
    return None


def pan_motion(src: int, dst: int, xy: np.ndarray) -> np.ndarray:
    """Warp of the panning camera."""
    return xy + PAN_PX_PER_FRAME * (dst - src)


def flat_pitch(frame: int, xy: np.ndarray) -> np.ndarray:
    """Pitch map of a still camera: 20 px are 1 m, the image centre is the centre mark."""
    return (xy - IMAGE_CENTRE) * 0.05


#: The flat pitch map as a matrix: ``flat_pitch`` for homography records.
FLAT_PITCH_MATRIX = np.array(
    [[0.05, 0.0, -0.05 * IMAGE_CENTRE[0]], [0.0, 0.05, -0.05 * IMAGE_CENTRE[1]], [0.0, 0.0, 1.0]]
)


def coarse_pitch(frame: int, xy: np.ndarray) -> np.ndarray:
    """Pitch map of a still camera that is far away: 5 px are 1 m."""
    return (xy - IMAGE_CENTRE) * 0.2


#: Arguments of ``finalize_tracks`` for a still camera with the flat pitch map: 20 px are 1 m.
ON_THE_FLAT_PITCH = {"fps": FPS, "to_pitch": flat_pitch}


def pan_pitch(frame: int, xy: np.ndarray) -> np.ndarray:
    """Pitch map of the panning camera: undo the pan, then use the flat map."""
    return flat_pitch(0, xy - PAN_PX_PER_FRAME * frame)


def pan_record(frame: int) -> HomographyRecord:
    """Return the pitch mapping of one frame of the panning camera, as a record (the matrix of ``pan_pitch``)."""
    offset = IMAGE_CENTRE + PAN_PX_PER_FRAME * frame
    matrix = np.array([[0.05, 0.0, -0.05 * offset[0]], [0.0, 0.05, -0.05 * offset[1]], [0.0, 0.0, 1.0]])
    return HomographyRecord(frame=frame, matrix=matrix_to_rows(matrix), source="landmarks")


def pan_field(n_frames: int, missing_steps: Sequence[int] = ()) -> MotionField:
    """Return the motion field of the panning camera. The steps in ``missing_steps`` are not known."""
    step = np.array([[1.0, 0.0, PAN_PX_PER_FRAME[0]], [0.0, 1.0, PAN_PX_PER_FRAME[1]], [0.0, 0.0, 1.0]])
    return MotionField([None if index in missing_steps else step for index in range(n_frames - 1)])


# --- associate -------------------------------------------------------------------


def test_two_players_who_cross_paths_keep_their_ids() -> None:
    # Player one runs to the right on the row y = 400. Player two runs to the left
    # on the row y = 440. They pass each other, so their left-right order flips.
    keyframes = [
        keyframe(k * STEP, person(300.0 + 50.0 * k, 400.0), person(500.0 - 50.0 * k, 440.0)) for k in range(5)
    ]

    tracks = associate(keyframes, fps=FPS, warp=no_motion, to_pitch=flat_pitch)

    assert [track.track_id for track in tracks] == [0, 1]
    assert [frames_of(track) for track in tracks] == [[0, 12, 24, 36, 48], [0, 12, 24, 36, 48]]
    assert feet_x_of(tracks[0]) == [300.0, 350.0, 400.0, 450.0, 500.0]
    assert feet_x_of(tracks[1]) == [500.0, 450.0, 400.0, 350.0, 300.0]


@pytest.mark.parametrize(("first_team", "second_team"), [("A", "A"), ("A", "B")])
def test_two_people_who_run_through_each_other_keep_their_ids(first_team: TeamKey, second_team: TeamKey) -> None:
    # Two people run through each other on one row, 60 px per keyframe each. Between the third and
    # the fourth keyframe they pass, so the nearest detection of each track is then the other person.
    # A track is expected where its person runs to, so the ids stay, also for two teammates.
    keyframes = [
        keyframe(
            k * STEP,
            person(300.0 + 60.0 * k, 400.0, team_key=first_team),
            person(550.0 - 60.0 * k, 400.0, team_key=second_team),
        )
        for k in range(5)
    ]

    tracks = associate(keyframes, fps=FPS, warp=no_motion, to_pitch=flat_pitch)

    assert [feet_x_of(track) for track in tracks] == [
        [300.0, 360.0, 420.0, 480.0, 540.0],
        [550.0, 490.0, 430.0, 370.0, 310.0],
    ]
    assert [vote_role_team(track) for track in tracks] == [("player", first_team), ("player", second_team)]


def test_a_track_is_expected_at_a_share_of_its_last_step() -> None:
    # The person moved 100 px between the first two keyframes. The track is expected 85 px further
    # on at the third keyframe, at x = 285. Two detections lie at nearly the same distance from that
    # point, one on each side. The nearer one is taken.
    start = [keyframe(0, person(100.0, 400.0)), keyframe(12, person(200.0, 400.0))]

    behind = associate([*start, keyframe(24, person(225.0, 400.0), person(346.0, 400.0))], **STILL_CAMERA)
    assert feet_x_of(behind[0]) == [100.0, 200.0, 225.0]  # 60 px behind the expected point, against 61 px ahead

    ahead = associate([*start, keyframe(24, person(224.0, 400.0), person(345.0, 400.0))], **STILL_CAMERA)
    assert feet_x_of(ahead[0]) == [100.0, 200.0, 345.0]  # 61 px behind, against 60 px ahead


def test_a_track_is_moved_on_for_two_gaps_at_most() -> None:
    # The person moves 100 px per keyframe: 2.5 box heights, the largest jump that is allowed.
    start = [keyframe(0, person(100.0, 400.0)), keyframe(12, person(200.0, 400.0))]

    # Seen again two keyframes later, 200 px further on. The track is expected at 200 + 0.85 * 200 = 370.
    two_gaps = associate([*start, keyframe(24), keyframe(36, person(400.0, 400.0))], **STILL_CAMERA)
    assert [(frames_of(track), feet_x_of(track)) for track in two_gaps] == [([0, 12, 36], [100.0, 200.0, 400.0])]

    # Three keyframes later the velocity is too old: the track is expected at its last place, x = 200.
    far = associate([*start, keyframe(24), keyframe(36), keyframe(48, person(500.0, 400.0))], **STILL_CAMERA)
    assert [frames_of(track) for track in far] == [[0, 12], [48]]
    near = associate([*start, keyframe(24), keyframe(36), keyframe(48, person(210.0, 400.0))], **STILL_CAMERA)
    assert [(frames_of(track), feet_x_of(track)) for track in near] == [([0, 12, 48], [100.0, 200.0, 210.0])]


def test_velocity_is_the_motion_of_the_person_without_the_camera_motion() -> None:
    # The camera pans, so the picture moves 300 px to the left per keyframe. A player walks 30 px
    # per keyframe to the right on the pitch. Its image position moves 270 px to the left.
    keyframes = [keyframe(k * STEP, person(900.0 - 270.0 * k, 400.0)) for k in range(4)]
    seen: list[np.ndarray] = []

    def recording_pan(src: int, dst: int, xy: np.ndarray) -> np.ndarray:
        seen.append(np.array(xy))
        return pan_motion(src, dst, xy)

    tracks = associate(keyframes, fps=FPS, warp=recording_pan, to_pitch=pan_pitch)

    assert [frames_of(track) for track in tracks] == [[0, 12, 24, 36]]
    # Each call gets the foot point as it is and the foot point moved on with the velocity.
    # The velocity is 30 px per keyframe, not -270: the expected point is 0.85 * 30 px further right.
    np.testing.assert_allclose(seen[1], [[630.0, 400.0], [655.5, 400.0]])
    np.testing.assert_allclose(seen[2], [[360.0, 400.0], [385.5, 400.0]])


def test_mismatch_penalty_decides_between_a_near_detection_and_one_with_the_right_labels() -> None:
    # Two opponents stand 40 px apart. Both tracks have one observation, so they have no velocity.
    # At the next keyframe the detection of team B is 12 px from the track of team A, and the
    # detection of team A is 28 px from it: the nearest pairs cost 0.3 + 0.3 box heights plus two
    # penalties, the pairs with the same labels cost 0.7 + 0.7.
    first = keyframe(0, person(300.0, 400.0, team_key="A"), person(340.0, 400.0, team_key="B"))
    second = keyframe(12, person(312.0, 400.0, team_key="B"), person(328.0, 400.0, team_key="A"))

    tracks = associate([first, second], fps=FPS, warp=no_motion, to_pitch=flat_pitch)
    assert [feet_x_of(track) for track in tracks] == [[300.0, 328.0], [340.0, 312.0]]

    # Without the penalty the nearest detection is taken.
    no_penalty = TrackingConfig(mismatch_penalty=0.0)
    nearest = associate([first, second], fps=FPS, warp=no_motion, to_pitch=flat_pitch, config=no_penalty)
    assert [feet_x_of(track) for track in nearest] == [[300.0, 312.0], [340.0, 328.0]]


def test_labels_of_a_detection_are_compared_with_the_vote_of_the_track() -> None:
    # The track says team A three times and then, in its newest observation, team B.
    # Its vote is still team A. The detection of team A is 20 px away (0.5 box heights), the
    # detection of team B is 8 px away (0.2 box heights plus the penalty).
    keyframes = [keyframe(k * STEP, person(600.0, 400.0, team_key="A")) for k in range(3)]
    keyframes.append(keyframe(36, person(600.0, 400.0, team_key="B")))
    keyframes.append(keyframe(48, person(608.0, 400.0, team_key="B"), person(580.0, 400.0, team_key="A")))

    tracks = associate(keyframes, fps=FPS, warp=no_motion, to_pitch=flat_pitch)

    assert [(frames_of(track), feet_x_of(track)) for track in tracks] == [
        ([0, 12, 24, 36, 48], [600.0, 600.0, 600.0, 600.0, 580.0]),
        ([48], [608.0]),
    ]


@pytest.mark.parametrize(("empty_keyframes", "followed"), [(1, True), (2, False)])
def test_label_penalty_counts_once_more_for_every_missed_keyframe(empty_keyframes: int, followed: bool) -> None:
    # A player of team A stands still and is then not seen for one or two keyframes. After that, a
    # person stands 44 px (1.1 box heights) beside the place of the player.
    start = [keyframe(0, person(600.0, 400.0, team_key="A")), keyframe(12, person(600.0, 400.0, team_key="A"))]
    empty = [keyframe(24 + 12 * index) for index in range(empty_keyframes)]
    last_frame = 24 + 12 * empty_keyframes

    # A person of team A is taken as the player, however long the wait was.
    same_team = [*start, *empty, keyframe(last_frame, person(644.0, 400.0, team_key="A"))]
    tracks = associate(same_team, fps=FPS, warp=no_motion, to_pitch=flat_pitch)
    assert [frames_of(track) for track in tracks] == [[0, 12, last_frame]]

    # A person of team B costs 1.1 + 2 * 0.5 = 2.1 after one missed keyframe, and 1.1 + 3 * 0.5 = 2.6
    # after two. The limit is 2.5.
    other_team = [*start, *empty, keyframe(last_frame, person(644.0, 400.0, team_key="B"))]
    tracks = associate(other_team, fps=FPS, warp=no_motion, to_pitch=flat_pitch)
    expected = [[0, 12, last_frame]] if followed else [[0, 12], [last_frame]]
    assert [frames_of(track) for track in tracks] == expected


def test_team_key_none_is_no_mismatch_but_another_role_is() -> None:
    track_start = keyframe(0, person(600.0, 400.0, team_key="A"))

    # The model could not tell the team of the near detection. That says nothing against the track.
    unknown_team = keyframe(12, person(608.0, 400.0, team_key="none"), person(580.0, 400.0, team_key="A"))
    tracks = associate([track_start, unknown_team], fps=FPS, warp=no_motion, to_pitch=flat_pitch)
    assert feet_x_of(tracks[0]) == [600.0, 608.0]

    # A track without any team vote takes a detection with a team at no extra cost.
    no_vote = keyframe(0, person(600.0, 400.0, team_key="none"))
    with_team = keyframe(12, person(608.0, 400.0, team_key="B"), person(580.0, 400.0, team_key="none"))
    tracks = associate([no_vote, with_team], fps=FPS, warp=no_motion, to_pitch=flat_pitch)
    assert feet_x_of(tracks[0]) == [600.0, 608.0]

    # The near detection is a referee: another role costs the penalty, so the player is taken.
    referee_near = keyframe(
        12, person(608.0, 400.0, role="referee", team_key="none"), person(580.0, 400.0, team_key="A")
    )
    tracks = associate([track_start, referee_near], fps=FPS, warp=no_motion, to_pitch=flat_pitch)
    assert feet_x_of(tracks[0]) == [600.0, 580.0]


def test_a_lost_track_and_a_new_person_do_not_shift_the_matches_between_them() -> None:
    # Three people stand in a row, 80 px apart (2 box heights, 4 m). At the next
    # keyframe the left person is gone and a new person stands at the right end of
    # the row. A shift of every track by one place would pair all three tracks, at
    # a cost of 2.0 each. The right answer pairs the two people who did not move at
    # no cost: the left track gets no detection and the new person starts a track.
    first = keyframe(0, person(100.0, 400.0), person(180.0, 400.0), person(260.0, 400.0))
    second = keyframe(12, person(180.0, 400.0), person(260.0, 400.0), person(340.0, 400.0))

    tracks = associate([first, second], fps=FPS, warp=no_motion, to_pitch=flat_pitch)

    assert [(track.track_id, frames_of(track), feet_x_of(track)) for track in tracks] == [
        (0, [0], [100.0]),
        (1, [0, 12], [180.0, 180.0]),
        (2, [0, 12], [260.0, 260.0]),
        (3, [12], [340.0]),
    ]


def test_a_pair_inside_the_limit_is_better_than_no_match() -> None:
    # 96 px are 2.4 box heights: a costly pair, but inside the limit of 2.5.
    # The other detection is far away from the track.
    first = keyframe(0, person(600.0, 400.0))
    second = keyframe(12, person(1000.0, 400.0), person(696.0, 400.0))

    tracks = associate([first, second], fps=FPS, warp=no_motion, to_pitch=flat_pitch)

    assert [(track.track_id, frames_of(track), feet_x_of(track)) for track in tracks] == [
        (0, [0, 12], [600.0, 696.0]),
        (1, [12], [1000.0]),
    ]


def test_a_label_flip_does_not_break_a_track() -> None:
    keyframes = [
        keyframe(0, person(600.0, 400.0, team_key="A")),
        keyframe(12, person(605.0, 400.0, team_key="B")),
        keyframe(24, person(610.0, 400.0, team_key="A")),
    ]

    tracks = associate(keyframes, fps=FPS, warp=no_motion, to_pitch=flat_pitch)

    assert [frames_of(track) for track in tracks] == [[0, 12, 24]]
    assert vote_role_team(tracks[0]) == ("player", "A")


def test_camera_pan_does_not_break_ids() -> None:
    # Two players stand still on the pitch, 300 px apart. The camera pans, so the
    # picture moves 300 px to the left between keyframes: player one lands exactly
    # where player two was.
    keyframes = [
        keyframe(k * STEP, person(700.0 - 300.0 * k, 400.0), person(1000.0 - 300.0 * k, 400.0)) for k in range(3)
    ]

    tracks = associate(keyframes, fps=FPS, warp=pan_motion, to_pitch=pan_pitch)

    assert [track.track_id for track in tracks] == [0, 1]
    assert feet_x_of(tracks[0]) == [700.0, 400.0, 100.0]
    assert feet_x_of(tracks[1]) == [1000.0, 700.0, 400.0]


def test_no_track_is_continued_across_a_break_of_the_motion_chain() -> None:
    # The same pan as above, but the camera motion is not known, and there is no pitch map that
    # could refuse a pair. A track that stayed at its image position would take the wrong player:
    # player one lands where player two was. So no track is continued, and every keyframe starts
    # new tracks.
    keyframes = [
        keyframe(k * STEP, person(700.0 - 300.0 * k, 400.0), person(1000.0 - 300.0 * k, 400.0)) for k in range(3)
    ]

    tracks = associate(keyframes, fps=FPS, warp=broken_motion, to_pitch=no_pitch)

    assert [(frames_of(track), feet_x_of(track)) for track in tracks] == [
        ([0], [700.0]),
        ([0], [1000.0]),
        ([12], [400.0]),
        ([12], [700.0]),
        ([24], [100.0]),
        ([24], [400.0]),
    ]


def test_people_are_followed_across_a_missing_motion_step_through_the_pitch_mappings() -> None:
    # The camera pans 25 px per frame, and the motion step from frame 17 to frame 18 is missing.
    # Two players stand still on the pitch, 300 px apart, so each keyframe shows player one where
    # the keyframe before showed player two.
    keyframes = [
        keyframe(k * STEP, person(700.0 - 300.0 * k, 400.0), person(1000.0 - 300.0 * k, 400.0)) for k in range(3)
    ]
    field = pan_field(25, missing_steps=[17])
    assert field.warp(12, 24, np.array([[400.0, 400.0]])) is None

    # The chain alone: the tracks end at the break, and nobody is mixed up. No pitch map is given
    # to the matching here, so only the warp can keep the two players apart.
    broken = associate(keyframes, fps=FPS, warp=field.warp, to_pitch=no_pitch)
    assert [(frames_of(track), feet_x_of(track)) for track in broken] == [
        ([0, 12], [700.0, 400.0]),
        ([0, 12], [1000.0, 700.0]),
        ([24], [100.0]),
        ([24], [400.0]),
    ]

    # With the pitch mappings of the frames, the break is crossed and the ids stay.
    warp = bridged_warp(field, [pan_record(frame) for frame in range(25)])
    tracks = associate(keyframes, fps=FPS, warp=warp, to_pitch=no_pitch)
    assert [(track.track_id, feet_x_of(track)) for track in tracks] == [
        (0, [700.0, 400.0, 100.0]),
        (1, [1000.0, 700.0, 400.0]),
    ]


def test_a_track_waits_for_a_frame_that_the_warp_reaches() -> None:
    def no_way_into_frame_12(src: int, dst: int, xy: np.ndarray) -> np.ndarray | None:
        """Still camera whose frame 12 cannot be compared with any other frame."""
        return None if 12 in (src, dst) else xy

    player = person(600.0, 400.0)
    keyframes = [keyframe(0, player), keyframe(12, player), keyframe(24, player), keyframe(36, player)]

    tracks = associate(keyframes, fps=FPS, warp=no_way_into_frame_12, to_pitch=flat_pitch)

    # The detection of frame 12 starts a track that no later frame can reach. The first track
    # misses one keyframe and goes on at frame 24.
    assert [(track.track_id, frames_of(track)) for track in tracks] == [(0, [0, 24, 36]), (1, [12])]


def test_points_that_the_warp_sends_to_infinity_give_no_match() -> None:
    def to_infinity(src: int, dst: int, xy: np.ndarray) -> np.ndarray:
        return np.full(xy.shape, np.inf)

    keyframes = [keyframe(0, person(600.0, 400.0)), keyframe(12, person(600.0, 400.0))]

    tracks = associate(keyframes, fps=FPS, warp=to_infinity, to_pitch=flat_pitch)

    assert [frames_of(track) for track in tracks] == [[0], [12]]

    def first_point_to_infinity(src: int, dst: int, xy: np.ndarray) -> np.ndarray:
        """Wrong warp: only the first point, the foot point as it is, goes to infinity."""
        moved = np.array(xy, dtype=np.float64)
        moved[0] = np.inf
        return moved

    # The expected point is finite and at the detection, but the velocity of the track could not
    # be measured from a foot point at infinity. So this is no match either.
    tracks = associate(keyframes, fps=FPS, warp=first_point_to_infinity, to_pitch=flat_pitch)

    assert [frames_of(track) for track in tracks] == [[0], [12]]


def test_associate_refuses_a_warp_that_loses_points() -> None:
    def one_point(src: int, dst: int, xy: np.ndarray) -> np.ndarray:
        """Wrong warp: it returns one point, whatever it gets."""
        return np.zeros((1, 2))

    keyframes = [keyframe(0, person(600.0, 400.0)), keyframe(12, person(600.0, 400.0))]

    with pytest.raises(ValueError, match="warp"):
        associate(keyframes, fps=FPS, warp=one_point, to_pitch=flat_pitch)


def test_warp_moves_points_from_the_last_observation_into_the_new_frame() -> None:
    calls: list[tuple[int, int]] = []

    def recording_warp(src: int, dst: int, xy: np.ndarray) -> np.ndarray:
        """Still camera that notes which frames it was asked for."""
        assert xy.ndim == 2 and xy.shape[1] == 2
        calls.append((src, dst))
        return xy

    player = person(600.0, 400.0)
    # The player is not seen at frame 12, so the next warp starts at frame 0 again.
    keyframes = [keyframe(0, player), keyframe(12), keyframe(24, player), keyframe(36, player)]

    tracks = associate(keyframes, fps=FPS, warp=recording_warp, to_pitch=flat_pitch)

    assert [frames_of(track) for track in tracks] == [[0, 24, 36]]
    assert set(calls) == {(0, 24), (24, 36)}


def test_pitch_map_gets_image_points_as_x_then_y() -> None:
    def x_only_pitch(frame: int, xy: np.ndarray) -> np.ndarray:
        """Pitch map that ignores the image y: 5 px in x are 1 m."""
        return xy * np.array([0.2, 0.0])

    # A step of 60 px down the image does not move the person on this pitch map.
    # A step of 60 px to the right is 12 m, which is out of reach.
    down = [keyframe(0, person(600.0, 400.0)), keyframe(12, person(600.0, 460.0))]
    right = [keyframe(0, person(600.0, 400.0)), keyframe(12, person(660.0, 400.0))]

    assert len(associate(down, fps=FPS, warp=no_motion, to_pitch=x_only_pitch)) == 1
    assert len(associate(right, fps=FPS, warp=no_motion, to_pitch=x_only_pitch)) == 2


def test_reach_limit_blocks_a_jump_that_is_too_far_on_the_pitch() -> None:
    # 60 px are 1.5 box heights, so the image limit (2.5) allows the pair.
    # On the coarse pitch map 60 px are 12 m. The reach in 0.48 s is 9 * 0.48 + 2 = 6.32 m.
    keyframes = [keyframe(0, person(600.0, 400.0)), keyframe(12, person(660.0, 400.0))]

    blocked = associate(keyframes, fps=FPS, warp=no_motion, to_pitch=coarse_pitch)
    assert [(track.track_id, frames_of(track)) for track in blocked] == [(0, [0]), (1, [12])]

    # Without pitch positions only the image limit counts.
    free = associate(keyframes, fps=FPS, warp=no_motion, to_pitch=no_pitch)
    assert [(track.track_id, frames_of(track)) for track in free] == [(0, [0, 12])]

    # 30 px are 6 m: this is inside the reach.
    near = [keyframe(0, person(600.0, 400.0)), keyframe(12, person(630.0, 400.0))]
    reachable = associate(near, fps=FPS, warp=no_motion, to_pitch=coarse_pitch)
    assert [(track.track_id, frames_of(track)) for track in reachable] == [(0, [0, 12])]


def test_a_pair_at_exactly_the_reach_is_allowed() -> None:
    # On this pitch map 2 px are 1 m, and one second passes between the two keyframes. With a
    # speed of 8 m/s and the margin of 2 m the reach is 10 m, which is 20 px. All these numbers
    # are exact in binary, so the first pair lies at exactly the reach.
    def half_scale(frame: int, xy: np.ndarray) -> np.ndarray:
        return xy * 0.5

    def tracks_of(second_x: float) -> list[list[int]]:
        keyframes = [keyframe(0, person(600.0, 400.0)), keyframe(25, person(second_x, 400.0))]
        config = TrackingConfig(max_speed_mps=8.0)
        tracks = associate(keyframes, fps=FPS, warp=no_motion, to_pitch=half_scale, config=config)
        return [frames_of(track) for track in tracks]

    assert tracks_of(620.0) == [[0, 25]]
    assert tracks_of(620.001) == [[0], [25]]


def test_reach_grows_with_the_time_since_the_last_observation() -> None:
    # 50 px are 10 m on the coarse pitch map. The reach is 6.32 m after one
    # keyframe step and 9 * 0.96 + 2 = 10.64 m after two steps.
    one_step = [keyframe(0, person(600.0, 400.0)), keyframe(12, person(650.0, 400.0))]
    two_steps = [keyframe(0, person(600.0, 400.0)), keyframe(12), keyframe(24, person(650.0, 400.0))]

    assert len(associate(one_step, fps=FPS, warp=no_motion, to_pitch=coarse_pitch)) == 2
    tracks = associate(two_steps, fps=FPS, warp=no_motion, to_pitch=coarse_pitch)
    assert [frames_of(track) for track in tracks] == [[0, 24]]


def test_a_pitch_position_that_is_not_finite_counts_as_missing() -> None:
    def broken_pitch(frame: int, xy: np.ndarray) -> np.ndarray:
        """Pitch map that cannot place any point."""
        return np.full(xy.shape, np.nan)

    # The same jump is blocked on the coarse pitch map (see the reach test above).
    keyframes = [keyframe(0, person(600.0, 400.0)), keyframe(12, person(660.0, 400.0))]

    tracks = associate(keyframes, fps=FPS, warp=no_motion, to_pitch=broken_pitch)

    assert [frames_of(track) for track in tracks] == [[0, 12]]


def test_a_pitch_position_at_infinity_counts_as_missing() -> None:
    def horizon_in_frame_12(frame: int, xy: np.ndarray) -> np.ndarray:
        """Coarse pitch map whose frame 12 sends every point to infinity."""
        return np.full(xy.shape, np.inf) if frame == 12 else coarse_pitch(frame, xy)

    # The track has a pitch position, the detection has none. A distance to infinity would block the pair.
    keyframes = [keyframe(0, person(600.0, 400.0)), keyframe(12, person(610.0, 400.0))]

    tracks = associate(keyframes, fps=FPS, warp=no_motion, to_pitch=horizon_in_frame_12)

    assert [frames_of(track) for track in tracks] == [[0, 12]]


def test_associate_refuses_a_pitch_map_that_loses_points() -> None:
    def short_pitch(frame: int, xy: np.ndarray) -> np.ndarray:
        """Wrong pitch map: it returns one point, whatever it gets."""
        return np.zeros((1, 2))

    keyframes = [keyframe(0, person(200.0, 400.0), person(600.0, 400.0))]

    with pytest.raises(ValueError, match="pitch map"):
        associate(keyframes, fps=FPS, warp=no_motion, to_pitch=short_pitch)


def test_image_jump_is_measured_in_box_heights() -> None:
    # A jump of 90 px is 3 box heights for a 30 px box and 1.5 box heights for a 60 px box.
    small = [keyframe(0, person(600.0, 400.0, height=30.0)), keyframe(12, person(690.0, 400.0, height=30.0))]
    large = [keyframe(0, person(600.0, 400.0, height=60.0)), keyframe(12, person(690.0, 400.0, height=60.0))]

    assert len(associate(small, fps=FPS, warp=no_motion, to_pitch=no_pitch)) == 2
    assert len(associate(large, fps=FPS, warp=no_motion, to_pitch=no_pitch)) == 1


def test_image_jump_is_measured_in_the_mean_height_of_both_boxes() -> None:
    # The box of the track is 30 px high and the new box is 60 px high: the mean is 45 px.
    # A jump of 100 px is 2.2 mean heights, and 3.3 heights of the track box alone.
    keyframes = [keyframe(0, person(600.0, 400.0, height=30.0)), keyframe(12, person(700.0, 400.0, height=60.0))]
    assert [frames_of(track) for track in associate(keyframes, **STILL_CAMERA)] == [[0, 12]]

    # The other way round: 100 px are 1.7 heights of the 60 px box alone, and the same 2.2 mean heights.
    # 120 px are 2.7 mean heights, which is too far, and 2.0 heights of the track box alone.
    keyframes = [keyframe(0, person(600.0, 400.0, height=60.0)), keyframe(12, person(720.0, 400.0, height=30.0))]
    assert [frames_of(track) for track in associate(keyframes, **STILL_CAMERA)] == [[0], [12]]


def test_a_pair_at_exactly_the_image_jump_limit_is_allowed() -> None:
    # 100 px are exactly 2.5 heights of a 40 px box.
    at_the_limit = [keyframe(0, person(600.0, 400.0)), keyframe(12, person(700.0, 400.0))]
    assert [frames_of(track) for track in associate(at_the_limit, **STILL_CAMERA)] == [[0, 12]]

    beyond = [keyframe(0, person(600.0, 400.0)), keyframe(12, person(700.5, 400.0))]
    assert [frames_of(track) for track in associate(beyond, **STILL_CAMERA)] == [[0], [12]]


def test_missed_keyframes_are_counted_since_the_last_match_only() -> None:
    config = TrackingConfig(max_missed_keyframes=2)
    seen = person(600.0, 400.0)
    # Two misses, a match, two misses, a match: four misses in all, but never more than two in a row.
    keyframes = [
        keyframe(0, seen),
        keyframe(12),
        keyframe(24),
        keyframe(36, seen),
        keyframe(48),
        keyframe(60),
        keyframe(72, seen),
    ]

    tracks = associate(keyframes, fps=FPS, warp=no_motion, to_pitch=flat_pitch, config=config)

    assert [(track.track_id, frames_of(track)) for track in tracks] == [(0, [0, 36, 72])]


def test_track_survives_missed_keyframes_up_to_the_limit() -> None:
    config = TrackingConfig(max_missed_keyframes=2)
    player = person(600.0, 400.0)
    keyframes = [keyframe(0, player), keyframe(12), keyframe(24), keyframe(36, player)]

    tracks = associate(keyframes, fps=FPS, warp=no_motion, to_pitch=flat_pitch, config=config)

    assert [(track.track_id, frames_of(track)) for track in tracks] == [(0, [0, 36])]


def test_track_ends_after_the_missed_keyframe_limit() -> None:
    config = TrackingConfig(max_missed_keyframes=2)
    player = person(600.0, 400.0)
    keyframes = [keyframe(0, player), keyframe(12), keyframe(24), keyframe(36), keyframe(48, player)]

    tracks = associate(keyframes, fps=FPS, warp=no_motion, to_pitch=flat_pitch, config=config)

    assert [(track.track_id, frames_of(track)) for track in tracks] == [(0, [0]), (1, [48])]


def test_no_missed_keyframe_is_allowed_when_the_limit_is_zero() -> None:
    config = TrackingConfig(max_missed_keyframes=0)
    player = person(600.0, 400.0)
    keyframes = [keyframe(0, player), keyframe(12), keyframe(24, player)]

    tracks = associate(keyframes, fps=FPS, warp=no_motion, to_pitch=flat_pitch, config=config)

    assert [(track.track_id, frames_of(track)) for track in tracks] == [(0, [0]), (1, [24])]


def test_track_ids_count_up_in_the_order_in_which_tracks_start() -> None:
    first = keyframe(0, person(200.0, 400.0), person(600.0, 400.0), person(1000.0, 400.0))
    # The model lists the same people in another order, and one new person.
    second = keyframe(12, person(1000.0, 400.0), person(200.0, 400.0), person(600.0, 400.0), person(100.0, 600.0))

    tracks = associate([first, second], fps=FPS, warp=no_motion, to_pitch=flat_pitch)

    assert [track.track_id for track in tracks] == [0, 1, 2, 3]
    assert [feet_x_of(track) for track in tracks] == [[200.0, 200.0], [600.0, 600.0], [1000.0, 1000.0], [100.0]]
    assert frames_of(tracks[3]) == [12]


def test_shuffled_detection_order_keeps_ids() -> None:
    # Ten people stand on a grid, 150 px apart, and move a few pixels. The shirt
    # number is used here only as the true identity of a detection.
    rng = np.random.default_rng(0)
    spots = [(200.0 + 150.0 * column, 300.0 + 200.0 * row) for row in range(2) for column in range(5)]
    keyframes: list[FramePeople] = []
    for k in range(6):
        people: list[PersonDet] = []
        for identity in rng.permutation(len(spots)):
            x, y = spots[identity]
            dx, dy = rng.uniform(-5.0, 5.0, size=2)
            people.append(person(x + float(dx), y + float(dy), number=int(identity)))
        keyframes.append(keyframe(k * STEP, *people))

    tracks = associate(keyframes, fps=FPS, warp=no_motion, to_pitch=flat_pitch)

    assert len(tracks) == len(spots)
    for track in tracks:
        assert frames_of(track) == [k * STEP for k in range(6)]
        assert len({observation.number for observation in track.observations}) == 1
    assert {track.observations[0].number for track in tracks} == set(range(len(spots)))


def test_observations_keep_the_fields_of_the_detection() -> None:
    keeper = person(600.0, 400.0, role="goalkeeper", team_key="B", number=1)

    tracks = associate([keyframe(0, keeper)], fps=FPS, warp=no_motion, to_pitch=flat_pitch)

    expected = TrackObs(frame=0, box=keeper.box, role="goalkeeper", team_key="B", number=1)
    assert tracks == [Track(track_id=0, observations=(expected,))]
    # The votes are not done yet.
    assert tracks[0].role is None and tracks[0].team_key is None and tracks[0].team_side is None


def test_keyframes_are_processed_in_frame_order() -> None:
    later = keyframe(12, person(610.0, 400.0))
    earlier = keyframe(0, person(600.0, 400.0))

    tracks = associate([later, earlier], fps=FPS, warp=no_motion, to_pitch=flat_pitch)

    assert [(frames_of(track), feet_x_of(track)) for track in tracks] == [([0, 12], [600.0, 610.0])]


def test_associate_without_keyframes_gives_no_tracks() -> None:
    assert associate([], fps=FPS, warp=no_motion, to_pitch=flat_pitch) == []


def test_associate_refuses_wrong_input() -> None:
    twice = [keyframe(0, person(600.0, 400.0)), keyframe(0, person(610.0, 400.0))]
    with pytest.raises(ValueError, match="same frame"):
        associate(twice, fps=FPS, warp=no_motion, to_pitch=flat_pitch)

    for fps in (0.0, -25.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="fps"):
            associate([keyframe(0, person(600.0, 400.0))], fps=fps, warp=no_motion, to_pitch=flat_pitch)


# --- stitch_fragments -----------------------------------------------------------


def test_stitch_joins_two_fragments() -> None:
    # The second fragment starts 36 frames (1.44 s) later and 20 px (1 m) away.
    first = still_track(0, [0, 12, 24], 600.0)
    second = still_track(1, [60, 72], 620.0)

    joined = stitch_fragments([first, second], fps=FPS, to_pitch=flat_pitch)

    assert [(track.track_id, frames_of(track)) for track in joined] == [(0, [0, 12, 24, 60, 72])]
    assert joined[0].observations == first.observations + second.observations


@pytest.mark.parametrize("second_frames", [[24, 36], [18, 30], [0, 12]])
def test_stitch_refuses_fragments_that_overlap_in_time(second_frames: list[int]) -> None:
    first = still_track(0, [0, 12, 24], 600.0)
    second = still_track(1, second_frames, 620.0)

    result = stitch_fragments([first, second], fps=FPS, to_pitch=flat_pitch)

    assert [(track.track_id, frames_of(track)) for track in result] == [(0, [0, 12, 24]), (1, second_frames)]


def test_stitch_refuses_a_gap_that_is_out_of_reach() -> None:
    # 200 px are 10 m on the flat pitch map.
    first = still_track(0, [0, 12, 24], 600.0)

    # 12 frames later the reach is 4 * 0.48 + 2 = 3.92 m.
    soon = still_track(1, [36, 48], 800.0)
    assert len(stitch_fragments([first, soon], fps=FPS, to_pitch=flat_pitch)) == 2

    # 60 frames later the reach is 4 * 2.4 + 2 = 11.6 m.
    late = still_track(1, [84, 96], 800.0)
    assert len(stitch_fragments([first, late], fps=FPS, to_pitch=flat_pitch)) == 1


def test_stitch_reach_uses_the_stitch_speed_and_the_reach_margin() -> None:
    # 400 px are 20 m, and 60 frames (2.4 s) lie between the fragments.
    first = still_track(0, [0, 12, 24], 600.0)
    second = still_track(1, [84, 96], 1000.0)

    # The reach of the stitch speed is 4 * 2.4 + 2 = 11.6 m. The speed of a sprinter would give 23.6 m.
    assert TrackingConfig().max_speed_mps * 2.4 + 2.0 > 20.0
    assert len(stitch_fragments([first, second], fps=FPS, to_pitch=flat_pitch)) == 2

    # 7.5 m/s give a reach of exactly 20 m.
    fast = TrackingConfig(stitch_speed_mps=7.5)
    assert len(stitch_fragments([first, second], fps=FPS, to_pitch=flat_pitch, config=fast)) == 1
    slower = TrackingConfig(stitch_speed_mps=7.4)
    assert len(stitch_fragments([first, second], fps=FPS, to_pitch=flat_pitch, config=slower)) == 2
    # A margin of 10.4 m gives the same reach with the default speed.
    wide = TrackingConfig(reach_margin_m=10.4)
    assert len(stitch_fragments([first, second], fps=FPS, to_pitch=flat_pitch, config=wide)) == 1


def test_stitch_refuses_a_time_gap_above_three_seconds() -> None:
    # Both fragments are at the same place, so only the time between them counts.
    first = still_track(0, [0, 12], 600.0)

    # 75 frames are 3.0 s: this is the longest gap that is joined.
    at_the_limit = still_track(1, [87, 99], 600.0)
    joined = stitch_fragments([first, at_the_limit], fps=FPS, to_pitch=flat_pitch)
    assert [frames_of(track) for track in joined] == [[0, 12, 87, 99]]

    # 76 frames are 3.04 s.
    too_late = still_track(1, [88, 100], 600.0)
    apart = stitch_fragments([first, too_late], fps=FPS, to_pitch=flat_pitch)
    assert [frames_of(track) for track in apart] == [[0, 12], [88, 100]]


def test_stitch_does_not_join_distant_fragments_after_a_long_gap() -> None:
    # The fragments are 1100 px (55 m) apart, and 250 frames (10 s) lie between them.
    # The reach after 10 s is 9 * 10 + 2 = 92 m, so the reach alone would allow the join.
    first = still_track(0, [0, 12], 90.0)
    second = still_track(1, [262, 274], 1190.0)

    result = stitch_fragments([first, second], fps=FPS, to_pitch=flat_pitch)

    assert [(track.track_id, frames_of(track)) for track in result] == [(0, [0, 12]), (1, [262, 274])]


def test_stitch_time_limit_is_in_seconds_not_in_frames() -> None:
    # 150 frames are 6 s at 25 frames per second and 3 s at 50 frames per second.
    first = still_track(0, [0, 12], 600.0)
    second = still_track(1, [162, 174], 600.0)

    assert len(stitch_fragments([first, second], fps=25.0, to_pitch=flat_pitch)) == 2
    assert len(stitch_fragments([first, second], fps=50.0, to_pitch=flat_pitch)) == 1


def test_stitch_time_limit_comes_from_the_settings() -> None:
    # Both fragments are at the same place. 150 frames (6 s) lie between them.
    first = still_track(0, [0, 12], 600.0)
    second = still_track(1, [162, 174], 600.0)

    default = stitch_fragments([first, second], fps=FPS, to_pitch=flat_pitch)
    assert [frames_of(track) for track in default] == [[0, 12], [162, 174]]

    six_seconds = TrackingConfig(max_stitch_gap_s=6.0)
    joined = stitch_fragments([first, second], fps=FPS, to_pitch=flat_pitch, config=six_seconds)
    assert [frames_of(track) for track in joined] == [[0, 12, 162, 174]]

    # A shorter limit refuses a gap that the default joins: 36 frames are 1.44 s.
    soon = still_track(1, [48, 60], 600.0)
    assert len(stitch_fragments([first, soon], fps=FPS, to_pitch=flat_pitch)) == 1
    one_second = TrackingConfig(max_stitch_gap_s=1.0)
    assert len(stitch_fragments([first, soon], fps=FPS, to_pitch=flat_pitch, config=one_second)) == 2


def test_stitch_time_limit_counts_for_each_join_not_for_the_whole_track() -> None:
    # Each gap is 60 frames (2.4 s). The joined track is 156 frames (6.24 s) long.
    parts = [still_track(0, [0, 12], 600.0), still_track(1, [72, 84], 605.0), still_track(2, [144, 156], 610.0)]

    result = stitch_fragments(parts, fps=FPS, to_pitch=flat_pitch)

    assert [(track.track_id, frames_of(track)) for track in result] == [(0, [0, 12, 72, 84, 144, 156])]


@pytest.mark.parametrize(("role", "team_key"), [("player", "B"), ("goalkeeper", "A"), ("referee", "none")])
def test_stitch_refuses_fragments_with_different_votes(role: Role, team_key: TeamKey) -> None:
    first = still_track(0, [0, 12, 24], 600.0, role="player", team_key="A")
    second = still_track(1, [60, 72], 620.0, role=role, team_key=team_key)

    assert len(stitch_fragments([first, second], fps=FPS, to_pitch=flat_pitch)) == 2


@pytest.mark.parametrize("frame_without_mapping", [24, 60])
def test_stitch_refuses_when_a_pitch_position_is_missing(frame_without_mapping: int) -> None:
    first = still_track(0, [0, 12, 24], 600.0)
    second = still_track(1, [60, 72], 620.0)

    def partial_pitch(frame: int, xy: np.ndarray) -> np.ndarray | None:
        """Flat pitch map with one frame that has no mapping."""
        return None if frame == frame_without_mapping else flat_pitch(frame, xy)

    assert len(stitch_fragments([first, second], fps=FPS, to_pitch=partial_pitch)) == 2
    assert len(stitch_fragments([first, second], fps=FPS, to_pitch=no_pitch)) == 2


def test_stitch_makes_no_join_when_two_later_fragments_are_seen_at_the_same_time() -> None:
    # Two later fragments are in reach of the early one, and both are in the image at frame 48.
    # They are two people, and nothing tells which of them the early fragment is.
    early = still_track(4, [0, 12, 24], 600.0)
    close = still_track(9, [36, 48], 610.0)
    later = still_track(7, [48, 60], 590.0)

    result = stitch_fragments([later, early, close], fps=FPS, to_pitch=flat_pitch)

    # The new ids follow the first frame.
    assert [(track.track_id, frames_of(track)) for track in result] == [(0, [0, 12, 24]), (1, [36, 48]), (2, [48, 60])]
    assert [feet_x_of(track)[0] for track in result] == [600.0, 610.0, 590.0]


def test_stitch_makes_no_join_when_two_fragments_start_together() -> None:
    # 80 px are 4 m, and the reach after 24 frames is 4 * 0.96 + 2 = 5.84 m: both are in reach.
    early = still_track(0, [0, 12], 600.0)
    far = still_track(1, [36, 48], 680.0)
    near = still_track(2, [36, 48], 620.0)

    result = stitch_fragments([early, far, near], fps=FPS, to_pitch=flat_pitch)

    assert [(track.track_id, frames_of(track)) for track in result] == [(0, [0, 12]), (1, [36, 48]), (2, [36, 48])]


def test_stitch_joins_the_only_fragment_in_reach() -> None:
    # The far fragment is 200 px (10 m) away: out of reach, so it is no rival.
    early = still_track(0, [0, 12], 600.0)
    far = still_track(1, [36, 48], 800.0)
    near = still_track(2, [36, 48], 620.0)

    result = stitch_fragments([early, far, near], fps=FPS, to_pitch=flat_pitch)

    assert [(track.track_id, frames_of(track)) for track in result] == [(0, [0, 12, 36, 48]), (1, [36, 48])]
    assert feet_x_of(result[0]) == [600.0, 600.0, 620.0, 620.0]
    assert feet_x_of(result[1]) == [800.0, 800.0]


def test_stitch_makes_no_join_when_two_earlier_fragments_are_seen_at_the_same_time() -> None:
    # The same rule backwards: two fragments end near the start of a later one, and both are in
    # the image at frame 12.
    first = still_track(0, [0, 12], 600.0)
    second = still_track(1, [12, 24], 620.0)
    later = still_track(2, [48, 60], 610.0)

    result = stitch_fragments([first, second, later], fps=FPS, to_pitch=flat_pitch)

    assert [(track.track_id, frames_of(track)) for track in result] == [(0, [0, 12]), (1, [12, 24]), (2, [48, 60])]


def test_stitch_joins_a_fragment_whose_other_candidate_can_follow_the_first_one() -> None:
    # The early fragment can be followed by the middle one and by the late one. The late one starts
    # after the middle one ended, so it is no rival: it follows the middle one.
    early = still_track(0, [0, 12], 600.0)
    middle = still_track(1, [36, 48], 605.0)
    late = still_track(2, [60, 72], 610.0)

    result = stitch_fragments([early, middle, late], fps=FPS, to_pitch=flat_pitch)

    assert [(track.track_id, frames_of(track)) for track in result] == [(0, [0, 12, 36, 48, 60, 72])]


def test_stitch_does_not_join_fragments_that_show_different_shirt_numbers() -> None:
    first = still_track(0, [0, 12, 24], 600.0, number=7)
    other_number = still_track(1, [60, 72], 620.0, number=9)
    assert len(stitch_fragments([first, other_number], fps=FPS, to_pitch=flat_pitch)) == 2

    same_number = still_track(1, [60, 72], 620.0, number=7)
    assert len(stitch_fragments([first, same_number], fps=FPS, to_pitch=flat_pitch)) == 1

    # A fragment without any per-frame number says nothing against a join.
    no_number = still_track(1, [60, 72], 620.0)
    assert len(stitch_fragments([first, no_number], fps=FPS, to_pitch=flat_pitch)) == 1
    assert len(stitch_fragments([still_track(0, [0, 12, 24], 600.0), other_number], fps=FPS, to_pitch=flat_pitch)) == 1


def test_stitch_compares_the_most_frequent_shirt_number_of_each_fragment() -> None:
    box = person(600.0, 400.0).box
    # The first fragment shows 9 one time and then 7 two times: its number is 7.
    numbers = {0: 9, 12: 7, 24: 7}
    first = Track(
        track_id=0,
        observations=tuple(
            TrackObs(frame=frame, box=box, role="player", team_key="A", number=number)
            for frame, number in numbers.items()
        ),
    )

    nine = still_track(1, [60, 72], 620.0, number=9)
    assert len(stitch_fragments([first, nine], fps=FPS, to_pitch=flat_pitch)) == 2
    seven = still_track(1, [60, 72], 620.0, number=7)
    assert len(stitch_fragments([first, seven], fps=FPS, to_pitch=flat_pitch)) == 1


def test_stitch_joins_a_chain_of_fragments() -> None:
    parts = [still_track(0, [0, 12], 600.0), still_track(1, [36, 48], 605.0), still_track(2, [72, 84], 610.0)]

    result = stitch_fragments(parts, fps=FPS, to_pitch=flat_pitch)

    assert [(track.track_id, frames_of(track)) for track in result] == [(0, [0, 12, 36, 48, 72, 84])]


def test_stitch_drops_short_tracks_before_the_joins() -> None:
    # Two lone detections of one team, 12 frames and 10 px apart: two false detections must not
    # become a track, so they are dropped before the joins.
    lone = still_track(0, [0], 100.0)
    first_half = still_track(1, [0], 600.0)
    second_half = still_track(2, [12], 610.0)
    fragments = [lone, first_half, second_half]

    assert stitch_fragments(fragments, fps=FPS, to_pitch=flat_pitch) == []

    keep_all = TrackingConfig(min_observations=1)
    result = stitch_fragments(fragments, fps=FPS, to_pitch=flat_pitch, config=keep_all)
    assert [(track.track_id, frames_of(track)) for track in result] == [(0, [0]), (1, [0, 12])]
    assert feet_x_of(result[1]) == [600.0, 610.0]


def test_stitch_ignores_a_lone_detection_between_two_fragments() -> None:
    # The lone detection lies between the two fragments in time. It is dropped, so it neither
    # joins them nor stands between them as a rival.
    first = still_track(0, [0, 12, 24], 600.0)
    lone = still_track(1, [36], 605.0)
    second = still_track(2, [36, 48, 60], 610.0)

    result = stitch_fragments([first, lone, second], fps=FPS, to_pitch=flat_pitch)

    assert [(track.track_id, frames_of(track)) for track in result] == [(0, [0, 12, 24, 36, 48, 60])]
    assert feet_x_of(result[0]) == [600.0] * 3 + [610.0] * 3

    # With three observations as the minimum, a fragment of two observations is dropped too.
    three = TrackingConfig(min_observations=3)
    short = still_track(1, [36, 48], 605.0)
    result = stitch_fragments([first, short], fps=FPS, to_pitch=flat_pitch, config=three)
    assert [(track.track_id, frames_of(track)) for track in result] == [(0, [0, 12, 24])]


def test_stitch_without_tracks_gives_no_tracks() -> None:
    assert stitch_fragments([], fps=FPS, to_pitch=flat_pitch) == []


@pytest.mark.parametrize("fps", [0.0, -25.0, float("inf"), float("nan")])
def test_stitch_refuses_a_wrong_frame_rate(fps: float) -> None:
    with pytest.raises(ValueError, match="fps"):
        stitch_fragments([still_track(0, [0, 12], 600.0)], fps=fps, to_pitch=flat_pitch)


# --- vote_role_team --------------------------------------------------------------


def test_vote_is_weighted_by_box_height() -> None:
    # Two small boxes say (player, B), one large box says (goalkeeper, A): 60 against 100.
    track = Track(
        track_id=0,
        observations=(
            TrackObs(frame=0, box=person(600.0, 400.0, height=30.0).box, role="player", team_key="B"),
            TrackObs(frame=12, box=person(600.0, 400.0, height=30.0).box, role="player", team_key="B"),
            TrackObs(frame=24, box=person(600.0, 400.0, height=100.0).box, role="goalkeeper", team_key="A"),
        ),
    )

    assert vote_role_team(track) == ("goalkeeper", "A")


def test_vote_is_a_plain_majority_for_boxes_of_one_size() -> None:
    track = Track(
        track_id=0,
        observations=(
            TrackObs(frame=0, box=person(600.0, 400.0).box, role="player", team_key="B"),
            TrackObs(frame=12, box=person(600.0, 400.0).box, role="player", team_key="B"),
            TrackObs(frame=24, box=person(600.0, 400.0).box, role="goalkeeper", team_key="A"),
        ),
    )

    assert vote_role_team(track) == ("player", "B")


def test_vote_gives_a_referee_no_team() -> None:
    track = still_track(0, [0, 12], 600.0, role="referee", team_key="A")

    assert vote_role_team(track) == ("referee", "none")


def test_tied_vote_goes_to_the_first_label() -> None:
    track = Track(
        track_id=0,
        observations=(
            TrackObs(frame=0, box=person(600.0, 400.0).box, role="goalkeeper", team_key="B"),
            TrackObs(frame=12, box=person(600.0, 400.0).box, role="player", team_key="A"),
        ),
    )

    assert vote_role_team(track) == ("player", "A")


def labelled_track(track_id: int, labels: Sequence[tuple[Role, TeamKey, float]]) -> Track:
    """Make a track with one observation per entry of ``labels``: (role, team key, box height)."""
    observations = tuple(
        TrackObs(frame=index * STEP, box=person(600.0, 400.0, height=height).box, role=role, team_key=team_key)
        for index, (role, team_key, height) in enumerate(labels)
    )
    return Track(track_id=track_id, observations=observations)


def test_team_key_none_does_not_outvote_a_team() -> None:
    # Three large boxes without a team and one small box of team B: 300 against 30.
    # "none" on a player says that the model could not tell the team there, so it is no vote.
    track = labelled_track(0, [("player", "none", 100.0)] * 3 + [("player", "B", 30.0)])
    assert vote_role_team(track) == ("player", "B")

    goalkeeper = labelled_track(0, [("goalkeeper", "none", 100.0)] * 3 + [("goalkeeper", "A", 30.0)])
    assert vote_role_team(goalkeeper) == ("goalkeeper", "A")

    # The teams are voted among themselves, by box height: 60 for team A against 50 for team B.
    mixed = labelled_track(0, [("player", "none", 100.0), ("player", "B", 50.0), ("player", "A", 60.0)])
    assert vote_role_team(mixed) == ("player", "A")


def test_vote_without_any_team_gives_the_key_none() -> None:
    assert vote_role_team(labelled_track(0, [("player", "none", 40.0)] * 2)) == ("player", "none")
    assert vote_role_team(labelled_track(0, [("goalkeeper", "none", 40.0)])) == ("goalkeeper", "none")


# --- assign_team_sides -----------------------------------------------------------


def two_team_tracks() -> list[Track]:
    """Make two player tracks of team A (ids 0, 1) and two of team B (ids 2, 3)."""
    return [
        still_track(0, [0, 12], 300.0, team_key="A"),
        still_track(1, [0, 12], 350.0, team_key="A"),
        still_track(2, [0, 12], 900.0, team_key="B"),
        still_track(3, [0, 12], 950.0, team_key="B"),
    ]


def test_team_with_the_lower_mean_pitch_x_is_left() -> None:
    tracks = two_team_tracks()

    assert assign_team_sides(tracks, {0: -20.0, 1: -5.0, 2: 10.0, 3: 30.0}) == {"A": "left", "B": "right"}
    assert assign_team_sides(tracks, {0: 20.0, 1: 5.0, 2: -10.0, 3: -30.0}) == {"A": "right", "B": "left"}
    # Both teams are in the right half: the lower mean x is still the left team.
    assert assign_team_sides(tracks, {0: 2.0, 1: 8.0, 2: 20.0, 3: 30.0}) == {"A": "left", "B": "right"}


def test_team_sides_use_players_with_a_team_and_a_position_only() -> None:
    tracks = [
        still_track(0, [0, 12], 300.0, team_key="A"),
        still_track(1, [0, 12], 900.0, team_key="B"),
        still_track(2, [0, 12, 24, 36], 100.0, role="goalkeeper", team_key="A"),
        still_track(3, [0, 12, 24, 36], 500.0, role="referee", team_key="none"),
        still_track(4, [0, 12, 24, 36], 550.0, role="player", team_key="none"),
        still_track(5, [0, 12, 24, 36], 950.0, role="player", team_key="B"),
    ]
    # The goalkeeper (track 2) would flip the answer if it counted as a player. Tracks 3
    # and 4 have no team. Track 5 has no position.
    mean_pitch_x = {0: -10.0, 1: 10.0, 2: 50.0, 3: 50.0, 4: 50.0}

    assert assign_team_sides(tracks, mean_pitch_x) == {"A": "left", "B": "right"}


def test_team_sides_weight_a_track_by_its_observations() -> None:
    tracks = [
        still_track(0, range(0, 120, 12), 300.0, team_key="A"),
        still_track(1, [0], 350.0, team_key="A"),
        still_track(2, [0, 12, 24, 36, 48], 900.0, team_key="B"),
    ]
    # Team A: (10 * -10 + 1 * 30) / 11 = -6.4, which is left of team B at 0.
    # The plain mean of the two tracks of team A is +10.
    assert assign_team_sides(tracks, {0: -10.0, 1: 30.0, 2: 0.0}) == {"A": "left", "B": "right"}


def test_one_team_gets_its_side_from_the_sign_of_its_mean_x() -> None:
    team_a = [still_track(0, [0, 12], 300.0, team_key="A"), still_track(1, [0, 12], 350.0, team_key="A")]
    team_b = [still_track(0, [0, 12], 300.0, team_key="B")]

    assert assign_team_sides(team_a, {0: -20.0, 1: -4.0}) == {"A": "left", "B": "right"}
    assert assign_team_sides(team_a, {0: 20.0, 1: 4.0}) == {"A": "right", "B": "left"}
    assert assign_team_sides(team_b, {0: -3.0}) == {"B": "left", "A": "right"}
    # A mean x of exactly 0 is not negative: the team is not the left one.
    assert assign_team_sides(team_b, {0: 0.0}) == {"B": "right", "A": "left"}


def corner_kick_tracks() -> list[Track]:
    """Make the tracks of a corner kick at the left goal: two attackers, two defenders and the goalkeeper.

    Team B attacks and stands deeper than the defenders of team A: use the positions
    ``CORNER_KICK_X``. The goalkeeper of team A is track 4.
    """
    return [
        still_track(0, [0, 12], 300.0, team_key="A"),
        still_track(1, [0, 12], 350.0, team_key="A"),
        still_track(2, [0, 12], 400.0, team_key="B"),
        still_track(3, [0, 12], 450.0, team_key="B"),
        still_track(4, [0, 12], 100.0, role="goalkeeper", team_key="A"),
    ]


#: Mean pitch x of the corner-kick tracks: team A at -45.0 m, team B at -46.5 m, the goalkeeper on the goal line.
CORNER_KICK_X = {0: -51.0, 1: -39.0, 2: -47.0, 3: -46.0, 4: -52.0}


def test_goalkeeper_decides_the_sides_when_the_player_means_are_close() -> None:
    tracks = corner_kick_tracks()

    # The attackers stand 1.5 m deeper than the defenders. The mean-x rule alone names team B "left".
    players_only = {track_id: x for track_id, x in CORNER_KICK_X.items() if track_id != 4}
    assert assign_team_sides(tracks, players_only) == {"B": "left", "A": "right"}

    # The goalkeeper of team A stands in the left goal, so team A defends it.
    assert assign_team_sides(tracks, CORNER_KICK_X) == {"A": "left", "B": "right"}


def test_goalkeeper_decides_the_sides_when_one_team_only_has_players_in_view() -> None:
    # Three attackers of team B in the left half and the goalkeeper of team A. No defender is seen.
    tracks = [
        still_track(0, [0, 12], 300.0, team_key="B"),
        still_track(1, [0, 12], 350.0, team_key="B"),
        still_track(2, [0, 12], 100.0, role="goalkeeper", team_key="A"),
    ]

    # The sign of the mean x of team B alone says "left".
    assert assign_team_sides(tracks, {0: -38.0, 1: -34.0}) == {"B": "left", "A": "right"}
    assert assign_team_sides(tracks, {0: -38.0, 1: -34.0, 2: -51.5}) == {"A": "left", "B": "right"}
    # A goalkeeper in the right goal says the same as the players here.
    assert assign_team_sides(tracks, {0: -38.0, 1: -34.0, 2: 51.5}) == {"B": "left", "A": "right"}


def test_players_decide_the_sides_alone_when_their_means_are_three_metres_apart() -> None:
    tracks = corner_kick_tracks()
    # The goalkeeper of team A stands in the left goal and says "team A is left".
    close = {0: -40.0, 1: -40.0, 2: -42.9, 3: -42.9, 4: -52.0}  # 2.9 m apart: the goalkeeper decides
    assert assign_team_sides(tracks, close) == {"A": "left", "B": "right"}

    clear = {0: -40.0, 1: -40.0, 2: -43.0, 3: -43.0, 4: -52.0}  # 3.0 m apart: the players decide
    assert assign_team_sides(tracks, clear) == {"B": "left", "A": "right"}


def test_goalkeeper_counts_only_with_a_team_key_and_a_place_in_a_goal_zone() -> None:
    tracks = corner_kick_tracks()
    players_say = {"B": "left", "A": "right"}

    # 35.9 m from the halfway line is in front of the penalty area: this track is not in a goal.
    assert assign_team_sides(tracks, {**CORNER_KICK_X, 4: -35.9}) == players_say
    assert assign_team_sides(tracks, {**CORNER_KICK_X, 4: -36.1}) == {"A": "left", "B": "right"}
    # A goalkeeper without a position.
    assert assign_team_sides(tracks, {track_id: x for track_id, x in CORNER_KICK_X.items() if track_id != 4}) == (
        players_say
    )
    # A goalkeeper without a team key.
    unknown_keeper = [*corner_kick_tracks()[:4], still_track(4, [0, 12], 100.0, role="goalkeeper", team_key="none")]
    assert assign_team_sides(unknown_keeper, CORNER_KICK_X) == players_say
    # A goalkeeper of team B in the right goal says that team A is left.
    right_keeper = [*corner_kick_tracks()[:4], still_track(4, [0, 12], 100.0, role="goalkeeper", team_key="B")]
    assert assign_team_sides(right_keeper, {**CORNER_KICK_X, 4: 52.0}) == {"A": "left", "B": "right"}


def test_goalkeepers_count_by_observations_and_by_depth_in_the_goal_zone() -> None:
    def with_second_keeper(frames: Sequence[int]) -> list[Track]:
        """Add a goalkeeper of team B, the attacking team, who came up for the corner kick."""
        return [*corner_kick_tracks(), still_track(5, frames, 150.0, role="goalkeeper", team_key="B")]

    # Both goalkeepers have two observations. The one of team A is 16 m deep in the zone (weight 32),
    # the one of team B is 12 m deep (weight 24): team A is in goal.
    x = {**CORNER_KICK_X, 5: -48.0}
    assert assign_team_sides(with_second_keeper([0, 12]), x) == {"A": "left", "B": "right"}
    # With three observations the second goalkeeper weighs 36 and wins.
    assert assign_team_sides(with_second_keeper([0, 12, 24]), x) == {"B": "left", "A": "right"}
    # The same depth and the same number of observations: a tie, so the players decide.
    assert assign_team_sides(with_second_keeper([0, 12]), {**CORNER_KICK_X, 5: -52.0}) == {"B": "left", "A": "right"}


def test_goalkeepers_alone_give_the_sides() -> None:
    keeper = [still_track(0, [0, 12], 100.0, role="goalkeeper", team_key="B")]

    assert assign_team_sides(keeper, {0: -50.0}) == {"B": "left", "A": "right"}
    assert assign_team_sides(keeper, {0: 50.0}) == {"A": "left", "B": "right"}


def test_no_side_without_a_player_position() -> None:
    assert assign_team_sides(two_team_tracks(), {}) == {}
    assert assign_team_sides([], {0: 5.0}) == {}


def test_team_sides_refuse_a_position_that_is_not_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        assign_team_sides(two_team_tracks(), {0: float("nan"), 2: 10.0})


# --- decide_number ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("reading", "frame_numbers", "expected"),
    [
        # A confident reading with two agreeing crops: exported and shown.
        (NumberReading(per_crop=(10, 10, None), number=10, confidence="high"), [], (10, 10)),
        # Only one crop agrees: exported, not shown.
        (NumberReading(per_crop=(10, None, None), number=10, confidence="high"), [], (10, None)),
        (NumberReading(per_crop=(7, 7, 10), number=10, confidence="high"), [], (10, None)),
        # Not confident: exported, not shown.
        (NumberReading(per_crop=(10, 10, 10), number=10, confidence="medium"), [], (10, None)),
        (NumberReading(per_crop=(10, 10, 10), number=10, confidence="low"), [], (10, None)),
        # A sure reading wins over the per-frame numbers: high confidence, or two crops that agree.
        (NumberReading(per_crop=(10, 10), number=10, confidence="high"), [7, 7, 7], (10, 10)),
        (NumberReading(per_crop=(10, None), number=10, confidence="high"), [7, 7, 7], (10, None)),
        (NumberReading(per_crop=(10, 10, None), number=10, confidence="low"), [7, 7, 7], (10, None)),
        # One crop that the model was not sure about loses against the frames that agree.
        (NumberReading(per_crop=(10, None), number=10, confidence="low"), [7, 7, 7], (7, None)),
        (NumberReading(per_crop=(10, None), number=10, confidence="medium"), [7, None, 7], (7, None)),
        # The crops and the frames vote together: 10 from one crop and one frame, 7 from one frame.
        (NumberReading(per_crop=(10, None), number=10, confidence="low"), [7, 10], (10, None)),
        # A tie between a crop and a frame goes to the crop, which is listed first.
        (NumberReading(per_crop=(10,), number=10, confidence="low"), [7], (10, None)),
        (NumberReading(per_crop=(None, 4), number=None, confidence="low"), [7], (4, None)),
        # A reading that is not sure and stands alone is still the most likely number.
        (NumberReading(per_crop=(10, None), number=10, confidence="low"), [], (10, None)),
        (NumberReading(per_crop=(None, None), number=10, confidence="low"), [None, None], (10, None)),
        # A reading without a number: the most frequent number of the crops and the frames is exported.
        (NumberReading(per_crop=(None, None), number=None, confidence="high"), [7, None, 7, 9], (7, None)),
        (NumberReading(per_crop=(4, 4), number=None, confidence="high"), [], (4, None)),
        (NumberReading(per_crop=(4, 4), number=None, confidence="high"), [7, 7, 7], (7, None)),
        (NumberReading(per_crop=(None,), number=None, confidence="high"), [], (None, None)),
        # A digit of the two-digit number of the reading counts for that number: the crops saw the
        # number 10 in parts, and the model put the parts together.
        (NumberReading(per_crop=(1, 0, None), number=10, confidence="medium"), [None, None], (10, None)),
        (NumberReading(per_crop=(10, 1, 1), number=10, confidence="medium"), [None], (10, None)),
        (NumberReading(per_crop=(17, None), number=17, confidence="low"), [7, 7, 7], (17, None)),
        (NumberReading(per_crop=(None, None), number=23, confidence="medium"), [3, 9, 9], (9, None)),
        (NumberReading(per_crop=(None, None), number=23, confidence="medium"), [3, 2, 9, 9], (23, None)),
        # A number that is no digit of the reading is not a part of it, and a one-digit reading has no parts.
        (NumberReading(per_crop=(10, None), number=10, confidence="medium"), [7, 7, 1], (10, None)),
        (NumberReading(per_crop=(10, None), number=10, confidence="medium"), [7, 7, 7, 1], (7, None)),
        (NumberReading(per_crop=(7, None), number=7, confidence="medium"), [1, 1], (1, None)),
        (NumberReading(per_crop=(7, None), number=7, confidence="medium"), [0, 0], (0, None)),
        (NumberReading(per_crop=(7, None), number=7, confidence="medium"), [77, 77], (77, None)),
        # No reading at all.
        (None, [None, 9, 7, 7, 9, 7], (7, None)),
        (None, [None, None], (None, None)),
        (None, [], (None, None)),
        # A tie goes to the number that was seen first.
        (None, [9, 7, 7, 9], (9, None)),
    ],
)
def test_decide_number_export_and_display(
    reading: NumberReading | None, frame_numbers: list[int | None], expected: tuple[int | None, int | None]
) -> None:
    assert decide_number(reading, frame_numbers) == expected


# --- finalize_tracks -------------------------------------------------------------


def summary_of(track: Track) -> tuple:
    """Return the decided fields of a track."""
    return (track.role, track.team_key, track.team_side, track.number_export, track.number_display, track.name)


def test_finalize_fills_votes_sides_numbers_and_names() -> None:
    tracks = [
        still_track(0, [0, 12, 24], 300.0, team_key="A"),
        still_track(1, [12, 24, 36, 48, 60], 320.0, team_key="A"),
        still_track(2, [0, 12, 24, 36], 900.0, team_key="B"),
        still_track(3, [0, 12, 24, 36], 600.0, role="referee", team_key="none", number=5),
        still_track(4, [0, 12, 24, 36], 1200.0, role="goalkeeper", team_key="B", number=1),
        still_track(5, [0, 12], 50.0, role="other", team_key="none"),
    ]
    readings = {
        0: high_reading(10),
        1: high_reading(10, crops=2),
        2: NumberReading(per_crop=(10, None, 10), number=10, confidence="high"),
        3: high_reading(5),
    }
    mean_pitch_x = {0: -20.0, 1: -10.0, 2: 15.0, 3: 0.0, 4: 48.0}
    roster = Roster(left={10: "Left Ten"}, right={10: "Right Ten", 1: "Right Keeper"})

    result = finalize_tracks(tracks, readings=readings, mean_pitch_x=mean_pitch_x, roster=roster, **ON_THE_FLAT_PITCH)

    assert [summary_of(track) for track in result] == [
        # Tracks 0 and 1 are on the left side, both read 10 and both are in the image at the frames
        # 12 and 24: two people, and one reading is wrong. The longer track 1 keeps the display.
        ("player", "A", "left", 10, None, None),
        ("player", "A", "left", 10, 10, "Left Ten"),
        # The same number on the other side is a different player.
        ("player", "B", "right", 10, 10, "Right Ten"),
        # A referee has no team side and no number, also when a number was read.
        ("referee", "none", None, None, None, None),
        # The goalkeeper has per-frame numbers only: exported, not shown, so no name.
        ("goalkeeper", "B", "right", 1, None, None),
        ("other", "none", None, None, None, None),
    ]
    assert [track.track_id for track in result] == [0, 1, 2, 3, 4, 5]
    assert [track.observations for track in result] == [track.observations for track in tracks]
    # A copy with updates is not validated, so check that every result is a valid record.
    assert [Track.model_validate(track.model_dump()) for track in result] == result
    # The input records are frozen and stay as they were.
    assert all(track.role is None and track.number_export is None for track in tracks)


def test_finalize_gives_a_repeated_display_to_the_lower_id_when_lengths_are_equal() -> None:
    tracks = [
        still_track(0, [0, 12, 24], 300.0, team_key="A"),
        still_track(1, [24, 36, 48], 320.0, team_key="A"),
    ]
    readings = {0: high_reading(8), 1: high_reading(8)}

    result = finalize_tracks(tracks, readings=readings, mean_pitch_x={0: -20.0, 1: -10.0}, **ON_THE_FLAT_PITCH)

    # Both tracks are in the image at frame 24.
    assert [(track.team_side, track.number_export, track.number_display) for track in result] == [
        ("left", 8, 8),
        ("left", 8, None),
    ]


def test_tracks_overlap_in_time_also_without_a_common_frame() -> None:
    # The two tracks have their observations at different keyframes, but both run through the same time.
    tracks = [
        still_track(0, [0, 24, 48, 72], 300.0, team_key="A"),
        still_track(1, [36, 60], 320.0, team_key="A"),
    ]
    readings = {0: high_reading(8), 1: high_reading(8)}

    result = finalize_tracks(tracks, readings=readings, mean_pitch_x={0: -20.0, 1: -10.0}, **ON_THE_FLAT_PITCH)

    assert [(track.track_id, frames_of(track), track.number_display) for track in result] == [
        (0, [0, 24, 48, 72], 8),
        (1, [36, 60], None),
    ]


def test_fragments_of_one_person_with_the_same_number_become_one_track() -> None:
    # The player left the image after frame 24 and came back at frame 300, 11 seconds later.
    # Both fragments read number 10 with confidence, on the same team side, at different times.
    tracks = [
        still_track(0, [0, 12, 24], 300.0, team_key="A"),
        still_track(1, [0, 12, 24, 36], 900.0, team_key="B"),
        still_track(2, [300, 312, 324, 336, 348], 320.0, team_key="A"),
    ]
    readings = {0: high_reading(10), 1: high_reading(10), 2: high_reading(10, crops=2)}
    roster = Roster(left={10: "Left Ten"}, right={10: "Right Ten"})

    result = finalize_tracks(
        tracks, readings=readings, mean_pitch_x={0: -20.0, 1: 15.0, 2: -10.0}, roster=roster, **ON_THE_FLAT_PITCH
    )

    # The later fragment is part of track 0 now. Number and name are shown in both parts.
    assert [(track.track_id, frames_of(track)) for track in result] == [
        (0, [0, 12, 24, 300, 312, 324, 336, 348]),
        (1, [0, 12, 24, 36]),
    ]
    assert [summary_of(track) for track in result] == [
        ("player", "A", "left", 10, 10, "Left Ten"),
        ("player", "B", "right", 10, 10, "Right Ten"),
    ]
    assert result[0].observations == tracks[0].observations + tracks[2].observations
    assert [Track.model_validate(track.model_dump()) for track in result] == result


def test_linked_track_has_the_id_and_the_place_of_its_earliest_fragment() -> None:
    # Three fragments of the left number 7, given out of time order, and another track between them.
    tracks = [
        still_track(5, [600, 612], 300.0, team_key="A"),
        still_track(2, [0, 12], 900.0, team_key="B"),
        still_track(9, [0, 12, 24], 310.0, team_key="A"),
        still_track(4, [200, 212], 320.0, team_key="A"),
    ]
    readings = {5: high_reading(7), 9: high_reading(7), 4: high_reading(7)}

    result = finalize_tracks(
        tracks, readings=readings, mean_pitch_x={5: -20.0, 2: 15.0, 9: -22.0, 4: -18.0}, **ON_THE_FLAT_PITCH
    )

    assert [(track.track_id, frames_of(track)) for track in result] == [
        (2, [0, 12]),
        (9, [0, 12, 24, 200, 212, 600, 612]),
    ]
    assert feet_x_of(result[1]) == [310.0] * 3 + [320.0] * 2 + [300.0] * 2


def test_a_second_track_of_one_person_loses_its_number_and_the_other_tracks_are_linked() -> None:
    # Track 1 is in the image together with the longer track 0 (frame 36), one metre beside it:
    # a second box of the same person, or a piece of a broken track. It loses the displayed
    # number. Track 2 comes later and is linked to track 0.
    tracks = [
        still_track(0, [0, 12, 24, 36], 300.0, team_key="A"),
        still_track(1, [36, 48, 60], 320.0, team_key="A"),
        still_track(2, [300, 312], 310.0, team_key="A"),
    ]
    readings = {0: high_reading(9), 1: high_reading(9), 2: high_reading(9)}

    result = finalize_tracks(
        tracks, readings=readings, mean_pitch_x={0: -20.0, 1: -10.0, 2: -15.0}, **ON_THE_FLAT_PITCH
    )

    assert [(track.track_id, frames_of(track), track.number_export, track.number_display) for track in result] == [
        (0, [0, 12, 24, 36, 300, 312], 9, 9),
        (1, [36, 48, 60], 9, None),
    ]


def test_a_number_that_two_people_show_at_the_same_time_links_no_track() -> None:
    # One reading is wrong: the players P (at x = 100 px) and Q (at x = 1100 px, 50 m away) of the
    # left side both show the number 3. Both are seen at the start of the clip, and both are seen
    # again later. In the first part P has the longer track, in the second part Q has it.
    tracks = [
        still_track(0, [0, 12, 24, 36, 48], 100.0, team_key="A"),
        still_track(1, [0, 12, 24, 36], 1100.0, team_key="A"),
        still_track(2, [300, 312, 324], 120.0, team_key="A"),
        still_track(3, [300, 312, 324, 336], 1080.0, team_key="A"),
        still_track(4, [0, 12], 900.0, team_key="B"),
    ]
    readings = {track_id: high_reading(3) for track_id in range(4)}
    mean_pitch_x = {0: -27.0, 1: -26.0, 2: -26.0, 3: -25.0, 4: 13.0}

    result = finalize_tracks(tracks, readings=readings, mean_pitch_x=mean_pitch_x, **ON_THE_FLAT_PITCH)

    # Nothing says which later track belongs to which person. So no track is linked: the track
    # of P is not joined with the later track of Q. Only the longest track shows the number.
    summary = [(track.track_id, len(track.observations), track.number_export, track.number_display) for track in result]
    assert summary == [(0, 5, 3, 3), (1, 4, 3, None), (2, 3, 3, None), (3, 4, 3, None), (4, 2, None, None)]


@pytest.mark.parametrize(("distance_px", "same_person"), [(120.0, True), (121.0, False)])
def test_tracks_at_the_same_time_are_two_people_when_they_are_more_than_six_metres_apart(
    distance_px: float, same_person: bool
) -> None:
    # 20 px are 1 m. Both tracks are seen at frame 24, and a third track comes later.
    tracks = [
        still_track(0, [0, 12, 24, 36], 300.0, team_key="A"),
        still_track(1, [24, 36, 48], 300.0 + distance_px, team_key="A"),
        still_track(2, [300, 312], 310.0, team_key="A"),
    ]
    readings = {0: high_reading(9), 1: high_reading(9), 2: high_reading(9)}

    result = finalize_tracks(
        tracks, readings=readings, mean_pitch_x={0: -17.0, 1: -12.0, 2: -16.0}, **ON_THE_FLAT_PITCH
    )

    assert TWO_PEOPLE_MARGIN_M == 6.0
    if same_person:
        # 6 m apart: the second track can be the same person with a mapping error. The later track is linked.
        assert [(track.track_id, len(track.observations), track.number_display) for track in result] == [
            (0, 6, 9),
            (1, 3, None),
        ]
    else:
        assert [(track.track_id, len(track.observations), track.number_display) for track in result] == [
            (0, 4, 9),
            (1, 3, None),
            (2, 2, None),
        ]


def test_a_track_between_two_observations_of_a_longer_track_is_compared_at_the_nearest_time() -> None:
    # The long track has a gap from frame 24 to frame 96. The short track is seen in that gap,
    # from frame 60 on. Its first observation is nearest to frame 24 and to frame 96 alike (36
    # frames, 1.44 s): the earlier one is taken. A person runs 9 * 1.44 + 6 = 18.96 m in that time.
    def result_for(distance_px: float) -> list[tuple[int, int | None]]:
        tracks = [
            still_track(0, [0, 12, 24, 96, 108], 300.0, team_key="A"),
            still_track(1, [60, 72], 300.0 + distance_px, team_key="A"),
            still_track(2, [400, 412], 300.0, team_key="A"),
        ]
        readings = {0: high_reading(9), 1: high_reading(9), 2: high_reading(9)}
        result = finalize_tracks(
            tracks, readings=readings, mean_pitch_x={0: -17.0, 1: -10.0, 2: -17.0}, **ON_THE_FLAT_PITCH
        )
        return [(track.track_id, track.number_display) for track in result]

    assert result_for(378.0) == [(0, 9), (1, None)]  # 18.9 m: one person, and track 2 is linked
    assert result_for(380.0) == [(0, 9), (1, None), (2, None)]  # 19.0 m: two people


@pytest.mark.parametrize(("second_start", "linked"), [(25, True), (24, False)])
def test_tracks_that_share_one_frame_are_not_linked(second_start: int, linked: bool) -> None:
    # The second track starts one frame after the last frame of the first one, or in that frame.
    tracks = [
        still_track(0, [0, 12, 24], 300.0, team_key="A"),
        still_track(1, [second_start, 36, 48, 60], 320.0, team_key="A"),
    ]
    readings = {0: high_reading(8), 1: high_reading(8)}

    result = finalize_tracks(tracks, readings=readings, mean_pitch_x={0: -20.0, 1: -10.0}, **ON_THE_FLAT_PITCH)

    if linked:
        assert [(track.track_id, len(track.observations), track.number_display) for track in result] == [(0, 7, 8)]
    else:
        # The longer track keeps the number, also when it is the later one.
        assert [(track.track_id, len(track.observations), track.number_display) for track in result] == [
            (0, 3, None),
            (1, 4, 8),
        ]


def test_two_people_with_the_same_number_far_apart_are_not_linked() -> None:
    # One reading is wrong: two players of the left side show the number 7. The first one is
    # seen until frame 48 at x = 100 px. The second one is first seen one second later, 1000 px
    # further right: 50 m on the flat pitch map. Nobody runs 50 m in one second.
    first = still_track(0, [0, 12, 24, 36, 48], 100.0, team_key="A")
    second = still_track(1, [73, 85, 97], 1100.0, team_key="A")
    others = [
        still_track(2, [0, 12], 300.0, team_key="A"),
        still_track(3, [0, 12], 900.0, team_key="B"),
    ]
    tracks = [first, second, *others]
    readings = {0: high_reading(7), 1: high_reading(7)}
    mean_pitch_x = {0: -27.0, 1: 23.0, 2: -17.0, 3: 13.0}

    result = finalize_tracks(tracks, readings=readings, mean_pitch_x=mean_pitch_x, **ON_THE_FLAT_PITCH)

    # Both stay their own track. The shorter one loses the displayed number and keeps the export number.
    assert [(track.track_id, frames_of(track), track.number_export, track.number_display) for track in result] == [
        (0, [0, 12, 24, 36, 48], 7, 7),
        (1, [73, 85, 97], 7, None),
        (2, [0, 12], None, None),
        (3, [0, 12], None, None),
    ]
    # So the dense output draws nobody on the way between the two people.
    records = [
        HomographyRecord(frame=frame, matrix=matrix_to_rows(FLAT_PITCH_MATRIX), source="propagated")
        for frame in range(98)
    ]
    states = dense_frame_states(
        result, records, 98, warp=no_motion, image_size=(1280, 720), max_fill_gap_frames=75
    )
    assert all(player.track_id not in (0, 1) for frame in range(49, 73) for player in states[frame].players)


def test_link_reach_is_the_fastest_run_plus_the_margin_of_six_metres() -> None:
    # 20 px are 1 m. The gap between the two fragments is 25 frames, one second: a person
    # reaches 9 m plus the margin of 6 m, so 15 m (300 px), and no more.
    def linked(distance_px: float, config: TrackingConfig = TrackingConfig(), fps: float = FPS) -> bool:
        tracks = [
            still_track(0, [0, 12], 300.0, team_key="A"),
            still_track(1, [37, 49], 300.0 + distance_px, team_key="A"),
        ]
        readings = {0: high_reading(7), 1: high_reading(7)}
        result = finalize_tracks(
            tracks, readings=readings, mean_pitch_x={0: -17.0, 1: -10.0}, config=config, fps=fps, to_pitch=flat_pitch
        )
        return len(result) == 1

    assert linked(300.0)
    assert not linked(301.0)
    # The speed is the fastest run of the settings. The margin is not the reach margin of the settings.
    assert linked(480.0, TrackingConfig(max_speed_mps=18.0))
    assert not linked(301.0, TrackingConfig(reach_margin_m=20.0))
    # The time is measured in seconds: at 50 frames per second the same gap is half a second (4.5 m + 6 m).
    assert linked(210.0, fps=50.0)
    assert not linked(211.0, fps=50.0)


@pytest.mark.parametrize("frame_without_mapping", [24, 300])
def test_tracks_are_not_linked_when_a_pitch_position_is_missing(frame_without_mapping: int) -> None:
    # The end of the first track, or the start of the second one, has no pitch position, so the
    # way between them cannot be tested. They stay two tracks, and both show their number.
    def to_pitch(frame: int, xy: np.ndarray) -> np.ndarray | None:
        return None if frame == frame_without_mapping else flat_pitch(frame, xy)

    tracks = [
        still_track(0, [0, 12, 24], 300.0, team_key="A"),
        still_track(1, [300, 312], 320.0, team_key="A"),
    ]
    readings = {0: high_reading(7), 1: high_reading(7)}

    result = finalize_tracks(tracks, readings=readings, mean_pitch_x={0: -17.0, 1: -16.0}, fps=FPS, to_pitch=to_pitch)

    assert [(track.track_id, frames_of(track), track.number_display) for track in result] == [
        (0, [0, 12, 24], 7),
        (1, [300, 312], 7),
    ]


def test_a_group_with_two_people_is_not_linked_at_all() -> None:
    # Three fragments show the left number 5. The second one starts 50 m from the end of the first
    # one, one second later: another person. The third one could follow both.
    tracks = [
        still_track(0, [0, 12, 24, 36], 100.0, team_key="A"),
        still_track(1, [61, 73], 1100.0, team_key="A"),
        still_track(2, [400, 412, 424], 600.0, team_key="A"),
    ]
    readings = {0: high_reading(5), 1: high_reading(5), 2: high_reading(5)}

    result = finalize_tracks(tracks, readings=readings, mean_pitch_x={0: -27.0, 1: 23.0, 2: -2.0}, **ON_THE_FLAT_PITCH)

    assert [(track.track_id, len(track.observations), track.number_display) for track in result] == [
        (0, 4, 5),
        (1, 2, None),
        (2, 3, None),
    ]


@pytest.mark.parametrize("fps", [0.0, -25.0, float("inf"), float("nan")])
def test_finalize_refuses_a_wrong_frame_rate(fps: float) -> None:
    with pytest.raises(ValueError, match="fps"):
        finalize_tracks([], readings={}, mean_pitch_x={}, fps=fps, to_pitch=flat_pitch)


@pytest.mark.parametrize(
    ("second_role", "second_key", "second_reading", "second_x"),
    [
        pytest.param("goalkeeper", "A", high_reading(1), -20.0, id="another role"),
        pytest.param("player", "B", high_reading(1), 20.0, id="another team side"),
        pytest.param("player", "A", high_reading(2), -20.0, id="another number"),
        pytest.param(
            "player", "A", NumberReading(per_crop=(1, None), number=1, confidence="high"), -20.0, id="shown by one crop"
        ),
        pytest.param(
            "player", "A", NumberReading(per_crop=(1, 1), number=1, confidence="medium"), -20.0, id="not confident"
        ),
    ],
)
def test_tracks_are_linked_only_with_the_same_role_side_and_displayed_number(
    second_role: Role, second_key: TeamKey, second_reading: NumberReading, second_x: float
) -> None:
    tracks = [
        still_track(0, [0, 12, 24], 300.0, team_key="A"),
        still_track(1, [300, 312], 320.0, role=second_role, team_key=second_key),
        still_track(2, [0, 12, 24], 900.0, team_key="B"),
    ]
    readings = {0: high_reading(1), 1: second_reading}

    result = finalize_tracks(
        tracks, readings=readings, mean_pitch_x={0: -20.0, 1: second_x, 2: 20.0}, **ON_THE_FLAT_PITCH
    )

    assert [(track.track_id, frames_of(track)) for track in result] == [
        (0, [0, 12, 24]),
        (1, [300, 312]),
        (2, [0, 12, 24]),
    ]
    assert result[0].number_display == 1


def test_goalkeeper_in_a_goal_has_the_side_of_that_goal_whatever_its_team_key_is() -> None:
    tracks = [
        still_track(0, [0, 12], 300.0, team_key="A"),
        still_track(1, [0, 12], 900.0, team_key="B"),
        # The kit of a goalkeeper matches no team, so the model often gives no team key.
        still_track(2, [0, 12], 100.0, role="goalkeeper", team_key="none"),
        still_track(3, [0, 12], 1200.0, role="goalkeeper", team_key="none"),
        # The model gave this goalkeeper the key of the team that defends the other goal.
        still_track(4, [24, 36], 110.0, role="goalkeeper", team_key="B"),
        # A track that is voted goalkeeper at the halfway line stands in no goal: its key decides.
        still_track(5, [0, 12], 640.0, role="goalkeeper", team_key="B"),
        still_track(6, [0, 12], 660.0, role="goalkeeper", team_key="none"),
        # A goalkeeper without any pitch position.
        still_track(7, [0, 12], 120.0, role="goalkeeper", team_key="none"),
    ]
    mean_pitch_x = {0: -10.0, 1: 10.0, 2: -50.0, 3: 51.0, 4: -49.0, 5: 2.0, 6: -3.0}

    result = finalize_tracks(tracks, readings={}, mean_pitch_x=mean_pitch_x, **ON_THE_FLAT_PITCH)

    assert [(track.role, track.team_key, track.team_side) for track in result] == [
        ("player", "A", "left"),
        ("player", "B", "right"),
        ("goalkeeper", "none", "left"),
        ("goalkeeper", "none", "right"),
        ("goalkeeper", "B", "left"),
        ("goalkeeper", "B", "right"),
        ("goalkeeper", "none", None),
        ("goalkeeper", "none", None),
    ]


def test_goal_zone_starts_at_the_penalty_area_line() -> None:
    def keeper_side(mean_x: float) -> str | None:
        tracks = [still_track(0, [0, 12], 100.0, role="goalkeeper", team_key="none")]
        return finalize_tracks(tracks, readings={}, mean_pitch_x={0: mean_x}, **ON_THE_FLAT_PITCH)[0].team_side

    assert [keeper_side(x) for x in (-36.0, -35.9, 35.9, 36.0, -55.0, 55.0)] == [
        "left",
        None,
        None,
        "right",
        "left",
        "right",
    ]


def test_goal_side_names_the_goal_zone_of_a_mean_pitch_x() -> None:
    """The label stage uses this function too, so the goal zone of the labels is the goal zone of the tracks."""
    assert GOAL_ZONE_MIN_X_M == 36.0
    assert [goal_side(x) for x in (-36.0, 36.0, -52.5, 52.5)] == ["left", "right", "left", "right"]
    assert [goal_side(x) for x in (-35.99, 35.99, 0.0, None)] == [None, None, None, None]


def test_a_player_never_gets_the_side_of_the_goal_zone_it_stands_in() -> None:
    # A defender of the right team who stands deep in the left half is still on the right team.
    tracks = [
        still_track(0, [0, 12], 300.0, team_key="A"),
        still_track(1, [0, 12], 900.0, team_key="B"),
        still_track(2, [0, 12], 100.0, team_key="B"),
    ]

    result = finalize_tracks(tracks, readings={}, mean_pitch_x={0: -10.0, 1: 30.0, 2: -50.0}, **ON_THE_FLAT_PITCH)

    assert [track.team_side for track in result] == ["left", "right", "right"]


def test_linked_goalkeeper_fragments_vote_one_team_key() -> None:
    # The goalkeeper of the left goal: the model gave no team in the first fragment and team A in the second.
    tracks = [
        still_track(0, [0, 12, 24], 100.0, role="goalkeeper", team_key="none"),
        still_track(1, [300, 312], 110.0, role="goalkeeper", team_key="A"),
        still_track(2, [0, 12], 300.0, team_key="A"),
        still_track(3, [0, 12], 900.0, team_key="B"),
    ]
    readings = {0: high_reading(1), 1: high_reading(1)}

    result = finalize_tracks(
        tracks, readings=readings, mean_pitch_x={0: -50.0, 1: -50.5, 2: -10.0, 3: 10.0}, **ON_THE_FLAT_PITCH
    )

    assert [(track.track_id, frames_of(track)) for track in result] == [
        (0, [0, 12, 24, 300, 312]),
        (2, [0, 12]),
        (3, [0, 12]),
    ]
    assert summary_of(result[0]) == ("goalkeeper", "A", "left", 1, 1, None)


def test_finalize_without_pitch_positions_gives_no_side_and_no_name() -> None:
    tracks = [
        still_track(0, [0, 12, 24], 300.0, team_key="A"),
        still_track(1, [60, 72], 320.0, team_key="A"),
    ]
    readings = {0: high_reading(10), 1: high_reading(10)}
    roster = Roster(left={10: "Left Ten"}, right={10: "Right Ten"})

    result = finalize_tracks(tracks, readings=readings, mean_pitch_x={}, roster=roster, **ON_THE_FLAT_PITCH)

    # Without a side there is no "unique per side" rule and no roster lookup.
    assert [(track.team_side, track.number_display, track.name) for track in result] == [
        (None, 10, None),
        (None, 10, None),
    ]


def test_finalize_gives_no_name_without_a_roster_entry() -> None:
    tracks = [still_track(0, [0, 12, 24], 300.0, team_key="A")]
    readings = {0: high_reading(10)}
    mean_pitch_x = {0: -20.0}

    # The left team of this roster has no number 10.
    other_number = Roster(left={9: "Left Nine"}, right={10: "Right Ten"})

    for roster in (None, other_number):
        result = finalize_tracks(
            tracks, readings=readings, mean_pitch_x=mean_pitch_x, roster=roster, **ON_THE_FLAT_PITCH
        )
        assert [(track.team_side, track.number_display, track.name) for track in result] == [("left", 10, None)]


def test_finalize_refuses_repeated_track_ids() -> None:
    tracks = [still_track(3, [0, 12], 300.0), still_track(3, [24, 36], 320.0)]

    with pytest.raises(ValueError, match="unique"):
        finalize_tracks(tracks, readings={}, mean_pitch_x={}, **ON_THE_FLAT_PITCH)


# --- the people of the synthetic scene -----------------------------------------------------------

GATES_FILE = Path(__file__).resolve().parents[1] / "configs" / "gates.yaml"
#: Time between two keyframes of the scene tests, seconds: the default of the pipeline.
KEYFRAME_INTERVAL_S = 0.48


def scene_keyframes(scene: SyntheticScene) -> list[FramePeople]:
    """Return the true people of every keyframe of a scene, as an exact model would answer."""
    frames = keyframe_indices(scene.info.n_frames, scene.info.fps, KEYFRAME_INTERVAL_S)
    return [scene.people(frame) for frame in frames]


def scene_records(scene: SyntheticScene) -> list[HomographyRecord]:
    """Return the true pitch mapping of every frame of a scene."""
    return [
        HomographyRecord(frame=frame, matrix=matrix_to_rows(scene.homography(frame)), source="landmarks")
        for frame in range(scene.info.n_frames)
    ]


def scene_motion(scene: SyntheticScene, missing_steps: Sequence[int] = ()) -> MotionField:
    """Return the true camera motion of a scene. The steps in ``missing_steps`` are not known."""
    to_pitch = [scene.homography(frame) for frame in range(scene.info.n_frames)]
    steps = [
        None if frame in missing_steps else np.linalg.inv(to_pitch[frame + 1]) @ to_pitch[frame]
        for frame in range(scene.info.n_frames - 1)
    ]
    return MotionField(steps)


def scene_pitch_map(scene: SyntheticScene) -> PitchMap:
    """Return the true pitch map of a scene."""

    def to_pitch(frame: int, xy: np.ndarray) -> np.ndarray:
        return apply_homography(scene.homography(frame), xy)

    return to_pitch


def mean_x_of(track: Track, to_pitch: PitchMap) -> float:
    """Return the mean pitch x of the foot points of a track."""
    xs = [to_pitch(observation.frame, np.array([observation.box.foot]))[0, 0] for observation in track.observations]
    return float(np.mean(xs))


def people_of(track: Track) -> set[tuple[Role, TeamKey, int | None]]:
    """Return the people that a track of a scene shows. Role, team and shirt number name a person of the scene."""
    return {(observation.role, observation.team_key, observation.number) for observation in track.observations}


def test_people_of_a_panning_clip_are_not_mixed_up_at_missing_motion_steps() -> None:
    # The camera of the scene pans by about 100 px per keyframe interval, more than the height of a
    # box. Five motion steps are missing, each between two keyframes.
    scene = SyntheticScene(n_frames=250, seed=0)
    keyframes = scene_keyframes(scene)
    to_pitch = scene_pitch_map(scene)
    whole = associate(keyframes, fps=scene.info.fps, warp=scene_motion(scene).warp, to_pitch=to_pitch)
    assert all(len(people_of(track)) == 1 for track in whole)
    field = scene_motion(scene, missing_steps=[30, 78, 126, 174, 222])

    # The chain alone: every track ends at a break, and no track shows two people.
    broken = associate(keyframes, fps=scene.info.fps, warp=field.warp, to_pitch=to_pitch)
    assert all(len(people_of(track)) == 1 for track in broken)
    assert len(broken) > 3 * len(whole)

    # With the pitch mappings of the frames the tracks are those of the whole chain.
    warp = bridged_warp(field, scene_records(scene))
    bridged = associate(keyframes, fps=scene.info.fps, warp=warp, to_pitch=to_pitch)
    assert [frames_of(track) for track in bridged] == [frames_of(track) for track in whole]
    assert [people_of(track) for track in bridged] == [people_of(track) for track in whole]


def test_long_clip_gives_one_id_per_person_within_the_identity_gate() -> None:
    # 30 seconds: the camera pans across the pitch, and 22 of the 23 people leave the image for
    # 2 to 10 seconds at least once. The answers are exact, so every loss comes from the tracking code.
    scene = SyntheticScene(n_frames=750, seed=0)
    info = scene.info
    to_pitch = scene_pitch_map(scene)
    field = scene_motion(scene)
    records = scene_records(scene)

    fragments = associate(scene_keyframes(scene), fps=info.fps, warp=field.warp, to_pitch=to_pitch)
    tracks = stitch_fragments(fragments, fps=info.fps, to_pitch=to_pitch)
    readings = {}
    for track in tracks:
        number = Counter(observation.number for observation in track.observations).most_common(1)[0][0]
        if number is not None:
            readings[track.track_id] = high_reading(number)
    mean_pitch_x = {track.track_id: mean_x_of(track, to_pitch) for track in tracks}
    decided = finalize_tracks(tracks, readings=readings, mean_pitch_x=mean_pitch_x, fps=info.fps, to_pitch=to_pitch)
    states = dense_frame_states(
        decided,
        records,
        info.n_frames,
        warp=field.warp,
        image_size=(info.width, info.height),
        max_fill_gap_frames=round(TrackingConfig().max_stitch_gap_s * info.fps),
    )

    # The fragments before the links: more than two per person.
    assert len(tracks) > 40
    assert len(decided) == 23
    truth = [scene.frame_state(frame) for frame in range(info.n_frames)]
    scores = identity_scores(states, truth)
    gate = yaml.safe_load(GATES_FILE.read_text(encoding="utf-8"))["g4_identity"]
    assert scores.coverage >= gate["coverage_min"]
    assert scores.switches <= gate["switches_per_clip_max"]
    assert scores.track_ratio <= gate["track_ratio_max"]
    # Every state of a player or a goalkeeper shows its shirt number, also after the person came back.
    team_states = [player for state in states for player in state.players if player.role in TEAM_ROLES]
    assert all(player.number_display is not None for player in team_states)
    # No state for a person whose box is outside the image: at most a handful more states than the truth has.
    assert sum(len(state.players) for state in states) <= sum(len(state.players) for state in truth)
