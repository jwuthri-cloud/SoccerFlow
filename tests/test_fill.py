"""Tests of fill.py: dense frame states from keyframe tracks.

The pitch mapping of the tests is a simple top-down picture: 10 px per metre,
with the centre mark at pixel (500, 250). So pitch x = x_px / 10 - 50 and
pitch y = y_px / 10 - 25. A point high in the picture (small y_px) is on the
far side and has a negative pitch y.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any

import numpy as np
import pytest

from pitchvlm.fill import dense_frame_states
from pitchvlm.geometry import matrix_to_rows
from pitchvlm.motion import MotionField
from pitchvlm.schemas import Box, FrameState, HomographyRecord, PlayerState, Track, TrackObs

PITCH_H = np.array([[0.1, 0.0, -50.0], [0.0, 0.1, -25.0], [0.0, 0.0, 1.0]])
#: Gap limit of the tests that are not about the limit, frames: 3 seconds at 25 frames per second.
MAX_GAP = 75
#: Image size of the tests that are not about the image border, (width, height) in pixels.
IMAGE_SIZE = (1200, 600)


def box_with_foot(x: float, y: float) -> Box:
    """Return a box of 20 x 60 px whose foot point (bottom centre) is (x, y)."""
    return Box(x1=x - 10.0, y1=y - 60.0, x2=x + 10.0, y2=y)


def box_values(box: Box | None) -> tuple[float, float, float, float]:
    """Return a box as (x1, y1, x2, y2)."""
    assert box is not None
    return (box.x1, box.y1, box.x2, box.y2)


def mapping_records(
    n_frames: int, missing: Collection[int] = (), matrix: np.ndarray = PITCH_H
) -> list[HomographyRecord]:
    """Return one homography record per frame. The frames in ``missing`` have no mapping."""
    rows = matrix_to_rows(matrix)
    records = []
    for frame in range(n_frames):
        if frame in missing:
            records.append(HomographyRecord(frame=frame))
        else:
            records.append(HomographyRecord(frame=frame, matrix=rows, source="landmarks"))
    return records


def make_track(track_id: int, feet: dict[int, tuple[float, float]], **attributes: Any) -> Track:
    """Return a track with one observation per entry of ``feet`` (frame -> foot point in pixels).

    The observations say "player" of team A. The voted fields of the track are
    "player" and team A too, unless ``attributes`` says otherwise.
    """
    observations = tuple(
        TrackObs(frame=frame, box=box_with_foot(*foot), role="player", team_key="A")
        for frame, foot in sorted(feet.items())
    )
    fields: dict[str, Any] = {"role": "player", "team_key": "A", **attributes}
    return Track(track_id=track_id, observations=observations, **fields)


def static_camera(src: int, dst: int, xy: np.ndarray) -> np.ndarray:
    """Motion of a camera that does not move."""
    return xy


def broken_chain(src: int, dst: int, xy: np.ndarray) -> None:
    """Motion that is never known."""
    return None


def fill(
    tracks: Sequence[Track], homographies: Sequence[HomographyRecord], n_frames: int, **options: Any
) -> list[FrameState]:
    """Call ``dense_frame_states`` with the gap limit ``MAX_GAP`` and the image size ``IMAGE_SIZE``.

    No two observations of the tests that use this helper are that far apart, so
    the limit changes nothing there. The tests of the limit give their own limit.
    The image holds every box of these tests, unless a test gives its own size.
    """
    settings = {"max_fill_gap_frames": MAX_GAP, "image_size": IMAGE_SIZE, **options}
    return dense_frame_states(tracks, homographies, n_frames, **settings)


def players_per_frame(states: list[FrameState]) -> list[int]:
    """Return the number of players in every frame."""
    return [len(state.players) for state in states]


def pitch_position(player: PlayerState) -> tuple[float, float]:
    """Return the pitch position of a player state as (x, y) in metres."""
    return (player.pitch_x, player.pitch_y)


# --- positions ---------------------------------------------------------------


def test_positions_are_interpolated_between_keyframes() -> None:
    track = make_track(4, {0: (400.0, 100.0), 10: (600.0, 300.0)})

    states = fill([track], mapping_records(11), 11, warp=static_camera)

    assert [state.frame for state in states] == list(range(11))
    assert players_per_frame(states) == [1] * 11
    first, quarter, middle, last = (states[frame].players[0] for frame in (0, 2, 5, 10))
    assert pitch_position(first) == pytest.approx((-10.0, -15.0))
    assert pitch_position(last) == pytest.approx((10.0, 5.0))
    assert pitch_position(middle) == pytest.approx((0.0, -5.0))  # the midpoint in pitch space
    assert pitch_position(quarter) == pytest.approx((-6.0, -11.0))
    assert all(state.players[0].track_id == 4 for state in states)


def test_only_keyframe_observations_are_marked_observed() -> None:
    track = make_track(0, {0: (400.0, 100.0), 10: (600.0, 300.0)})

    states = fill([track], mapping_records(11), 11, warp=static_camera)

    assert [state.players[0].observed for state in states] == [True] + [False] * 9 + [True]
    assert states[0].players[0].box == track.observations[0].box
    assert states[10].players[0].box == track.observations[1].box


def test_no_state_before_the_first_or_after_the_last_observation() -> None:
    track = make_track(0, {5: (400.0, 100.0), 15: (600.0, 300.0)})

    states = fill([track], mapping_records(20), 20, warp=static_camera)

    assert len(states) == 20
    assert [state.frame for state in states] == list(range(20))
    assert players_per_frame(states) == [0] * 5 + [1] * 11 + [0] * 4


def test_far_side_person_has_negative_pitch_y() -> None:
    far_side = make_track(0, {0: (500.0, 60.0)})  # high in the picture
    near_side = make_track(1, {0: (500.0, 440.0)})  # low in the picture

    players = fill([far_side, near_side], mapping_records(1), 1, warp=static_camera)[0].players

    assert [player.track_id for player in players] == [0, 1]
    assert players[0].pitch_y == pytest.approx(-19.0)
    assert players[0].pitch_y < 0.0
    assert players[1].pitch_y == pytest.approx(19.0)
    assert players[1].pitch_y > 0.0
    assert players[0].pitch_x == pytest.approx(0.0)


def test_position_is_the_foot_point_not_the_box_centre() -> None:
    track = make_track(0, {0: (400.0, 100.0)})
    assert track.observations[0].box.center == pytest.approx((400.0, 70.0))

    player = fill([track], mapping_records(1), 1, warp=static_camera)[0].players[0]

    assert pitch_position(player) == pytest.approx((-10.0, -15.0))  # from the foot (400, 100)


def test_observation_without_a_mapping_gets_an_interpolated_position() -> None:
    track = make_track(0, {0: (400.0, 100.0), 10: (700.0, 100.0), 20: (600.0, 300.0)})

    states = fill([track], mapping_records(21, missing={10}), 21, warp=static_camera)

    assert players_per_frame(states) == [1] * 21
    middle = states[10].players[0]
    # Halfway between the positions of frames 0 and 20. The foot point of frame 10 itself would give (20, -15).
    assert pitch_position(middle) == pytest.approx((0.0, -5.0))
    assert middle.observed
    assert middle.box == track.observations[1].box


def test_states_end_at_the_last_observation_that_has_a_mapping() -> None:
    track = make_track(0, {0: (400.0, 100.0), 10: (500.0, 100.0), 20: (600.0, 100.0)})

    states = fill([track], mapping_records(21, missing={0, 20}), 21, warp=static_camera)
    assert players_per_frame(states) == [0] * 10 + [1] + [0] * 10

    states = fill([track], mapping_records(21, missing={20}), 21, warp=static_camera)
    assert players_per_frame(states) == [1] * 11 + [0] * 10

    states = fill([track], mapping_records(21, missing={0, 10, 20}), 21, warp=static_camera)
    assert players_per_frame(states) == [0] * 21


def test_single_observation_gives_a_single_state() -> None:
    track = make_track(0, {3: (400.0, 100.0)})

    states = fill([track], mapping_records(6), 6, warp=static_camera)

    assert players_per_frame(states) == [0, 0, 0, 1, 0, 0]
    assert states[3].players[0].observed


def test_mappings_are_needed_at_the_observation_frames_only() -> None:
    track = make_track(0, {0: (400.0, 100.0), 10: (600.0, 300.0)})
    keyframe_records = [record for record in mapping_records(11) if record.frame in (0, 10)]

    states = fill([track], keyframe_records, 11, warp=static_camera)

    assert players_per_frame(states) == [1] * 11
    assert pitch_position(states[5].players[0]) == pytest.approx((0.0, -5.0))


def test_positions_far_outside_the_pitch_are_dropped() -> None:
    # Pitch x runs from 50 m to 60 m. The goal line is at 52.5 m, so the margin of 3 m ends at 55.5 m.
    track = make_track(0, {0: (1000.0, 250.0), 10: (1100.0, 250.0)})

    states = fill([track], mapping_records(11), 11, warp=static_camera)
    assert players_per_frame(states) == [1] * 6 + [0] * 5
    assert states[5].players[0].pitch_x == pytest.approx(55.0)

    wide_margin = fill([track], mapping_records(11), 11, warp=static_camera, pitch_margin_m=10.0)
    assert players_per_frame(wide_margin) == [1] * 11

    no_margin = fill([track], mapping_records(11), 11, warp=static_camera, pitch_margin_m=0.0)
    assert players_per_frame(no_margin) == [1] * 3 + [0] * 8


def test_foot_point_on_the_horizon_is_not_used() -> None:
    # The image row y = 10 maps to infinity: the third coordinate is y - 10. Rows below it map onto the pitch.
    with_horizon = np.array([[0.1, 0.0, -50.0], [0.0, 0.1, -25.0], [0.0, 1.0, -10.0]])
    track = make_track(0, {0: (400.0, 10.0), 5: (400.0, 210.0), 10: (400.0, 410.0)})

    states = fill([track], mapping_records(11, matrix=with_horizon), 11, warp=static_camera)

    assert players_per_frame(states) == [0] * 5 + [1] * 6
    assert pitch_position(states[5].players[0]) == pytest.approx((-10.0 / 200.0, -4.0 / 200.0))
    assert pitch_position(states[10].players[0]) == pytest.approx((-10.0 / 400.0, 16.0 / 400.0))


# --- gap limit ---------------------------------------------------------------


def test_gap_at_the_limit_is_filled_and_a_longer_gap_is_not() -> None:
    track = make_track(0, {0: (400.0, 100.0), 10: (600.0, 300.0)})

    at_the_limit = fill([track], mapping_records(11), 11, warp=static_camera, max_fill_gap_frames=10)
    assert players_per_frame(at_the_limit) == [1] * 11

    too_long = fill([track], mapping_records(11), 11, warp=static_camera, max_fill_gap_frames=9)
    # The two observations keep their states. The frames between them have none.
    assert players_per_frame(too_long) == [1] + [0] * 9 + [1]
    assert pitch_position(too_long[0].players[0]) == pytest.approx((-10.0, -15.0))
    assert pitch_position(too_long[10].players[0]) == pytest.approx((10.0, 5.0))
    assert [too_long[frame].players[0].observed for frame in (0, 10)] == [True, True]


def test_gap_limit_counts_for_each_gap_of_a_track() -> None:
    # Observations in the frames 0, 4, 20 and 24: two gaps of 4 frames and one gap of 16 frames.
    track = make_track(0, {0: (400.0, 100.0), 4: (420.0, 100.0), 20: (500.0, 100.0), 24: (520.0, 100.0)})

    states = fill([track], mapping_records(25), 25, warp=static_camera, max_fill_gap_frames=8)

    assert players_per_frame(states) == [1] * 5 + [0] * 15 + [1] * 5
    # The short gaps are filled: frame 2 is halfway between the frames 0 and 4, frame 22 between 20 and 24.
    assert pitch_position(states[2].players[0]) == pytest.approx((-9.0, -15.0))
    assert pitch_position(states[22].players[0]) == pytest.approx((1.0, -15.0))
    assert box_values(states[22].players[0].box) == pytest.approx((500.0, 40.0, 520.0, 100.0))


def test_gap_is_measured_between_observations_that_have_a_position() -> None:
    # The observation of frame 10 has no pitch mapping, so the gap runs from frame 0 to frame 20.
    track = make_track(0, {0: (400.0, 100.0), 10: (500.0, 100.0), 20: (600.0, 100.0)})
    records = mapping_records(21, missing={10})

    filled = fill([track], records, 21, warp=static_camera, max_fill_gap_frames=20)
    assert players_per_frame(filled) == [1] * 21

    # 12 frames cover the time between two observations (10 frames), but not the time between two positions.
    open_gap = fill([track], records, 21, warp=static_camera, max_fill_gap_frames=12)
    assert players_per_frame(open_gap) == [1] + [0] * 19 + [1]
    # The observation without a position lies inside the gap, so it has no state either.
    assert open_gap[10].players == ()


def test_limit_of_zero_keeps_the_observations_only() -> None:
    track = make_track(0, {0: (400.0, 100.0), 1: (410.0, 100.0), 3: (430.0, 100.0)})

    states = fill([track], mapping_records(4), 4, warp=static_camera, max_fill_gap_frames=0)

    assert players_per_frame(states) == [1, 1, 0, 1]
    assert all(state.players[0].observed for state in states if state.players)


def test_gap_limit_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match="max_fill_gap_frames"):
        fill([], [], 3, warp=static_camera, max_fill_gap_frames=-1)


def test_gap_limit_and_image_size_have_no_default() -> None:
    track = make_track(0, {0: (400.0, 100.0), 10: (600.0, 300.0)})
    with pytest.raises(TypeError, match="max_fill_gap_frames"):
        dense_frame_states([track], mapping_records(11), 11, warp=static_camera, image_size=IMAGE_SIZE)
    with pytest.raises(TypeError, match="image_size"):
        dense_frame_states([track], mapping_records(11), 11, warp=static_camera, max_fill_gap_frames=MAX_GAP)


# --- people outside the image ------------------------------------------------


def pan_away_and_back(src: int, dst: int, xy: np.ndarray) -> np.ndarray:
    """Motion of a camera that pans away for 10 frames and comes back: the picture moves 50 px left per frame."""
    shift = [-50.0 * min(frame, 20 - frame) for frame in (src, dst)]
    return xy + np.array([shift[1] - shift[0], 0.0])


def test_frames_in_which_the_person_is_outside_the_image_get_no_state() -> None:
    # The person stands still on the pitch. The box spans x from 90 to 110 px in the frames 0 and 20.
    # In between it moves 50 px to the left per frame and then back: it is at x = 40 to 60 in frame 1,
    # at -10 to 10 in frame 2, and outside the image from frame 3 to frame 17.
    track = make_track(0, {0: (100.0, 100.0), 20: (100.0, 100.0)})

    states = fill([track], mapping_records(21), 21, warp=pan_away_and_back)

    assert players_per_frame(states) == [1, 1, 1] + [0] * 15 + [1, 1, 1]
    assert box_values(states[2].players[0].box) == pytest.approx((-10.0, 40.0, 10.0, 100.0))  # a part is inside
    assert not states[2].players[0].observed
    # The same clip with a camera that does not move: the person is in the image all the time.
    assert players_per_frame(fill([track], mapping_records(21), 21, warp=static_camera)) == [1] * 21


def test_a_box_that_only_touches_the_image_border_from_outside_gets_no_state() -> None:
    # The box of the frames 0 and 2 spans x from 90 to 110 and y from 40 to 100, in an image of 300 x 200 pixels.
    track = make_track(0, {0: (100.0, 100.0), 2: (100.0, 100.0)})

    def middle_frame(dx: float, dy: float) -> tuple[PlayerState, ...]:
        """Return the players of frame 1 for a camera that shows the picture moved by (dx, dy) in that frame only."""

        def warp(src: int, dst: int, xy: np.ndarray) -> np.ndarray:
            return xy + ((dst == 1) - (src == 1)) * np.array([dx, dy])

        return fill([track], mapping_records(3), 3, warp=warp, image_size=(300, 200))[1].players

    assert len(middle_frame(0.0, 0.0)) == 1
    # Left border: the right edge of the box is at x = 0.5, then exactly at x = 0.
    assert len(middle_frame(-109.5, 0.0)) == 1
    assert middle_frame(-110.0, 0.0) == ()
    # Right border: the left edge is at x = 299.5, then exactly at the image width.
    assert len(middle_frame(209.5, 0.0)) == 1
    assert middle_frame(210.0, 0.0) == ()
    # Top border: the bottom edge is at y = 0.5, then exactly at y = 0.
    assert len(middle_frame(0.0, -99.5)) == 1
    assert middle_frame(0.0, -100.0) == ()
    # Bottom border: the top edge is at y = 199.5, then exactly at the image height.
    assert len(middle_frame(0.0, 159.5)) == 1
    assert middle_frame(0.0, 160.0) == ()


def test_an_observation_keeps_its_state_whatever_the_image_size_is() -> None:
    # Both observed boxes lie outside an image of 300 x 200 pixels, and so do the boxes between them.
    track = make_track(0, {0: (400.0, 100.0), 4: (440.0, 100.0)})

    states = fill([track], mapping_records(5), 5, warp=static_camera, image_size=(300, 200))

    assert players_per_frame(states) == [1, 0, 0, 0, 1]
    assert [states[frame].players[0].observed for frame in (0, 4)] == [True, True]


def test_boxes_that_cannot_be_moved_are_tested_against_the_image_as_they_are_mixed() -> None:
    # The motion chain is broken, so the two observed boxes are mixed directly. Both are inside the image.
    track = make_track(0, {0: (100.0, 100.0), 20: (100.0, 100.0)})

    states = fill([track], mapping_records(21), 21, warp=broken_chain)

    assert players_per_frame(states) == [1] * 21


# --- boxes -------------------------------------------------------------------


def test_boxes_are_mixed_by_time_when_the_camera_stands_still() -> None:
    track = make_track(0, {0: (400.0, 100.0), 10: (600.0, 300.0)})

    states = fill([track], mapping_records(11), 11, warp=static_camera)

    assert box_values(states[5].players[0].box) == pytest.approx((490.0, 140.0, 510.0, 200.0))
    assert box_values(states[2].players[0].box) == pytest.approx((430.0, 80.0, 450.0, 140.0))


def test_boxes_between_keyframes_follow_the_camera_motion() -> None:
    # The camera pans fast at first and then more slowly. The picture content moves by this many pixels in x.
    shift = np.array([0.0, 8.0, 15.0, 21.0, 26.0, 30.0, 33.0, 36.0, 38.0, 39.0, 40.0])

    def pan(src: int, dst: int, xy: np.ndarray) -> np.ndarray:
        return xy + np.array([shift[dst] - shift[src], 0.0])

    # A person who stands still on the pitch moves with the picture content: 40 px in 10 frames.
    track = make_track(0, {0: (400.0, 100.0), 10: (440.0, 100.0)})

    states = fill([track], mapping_records(11), 11, warp=pan)

    for frame in range(11):
        expected = (390.0 + shift[frame], 40.0, 410.0 + shift[frame], 100.0)
        assert box_values(states[frame].players[0].box) == pytest.approx(expected)


def test_boxes_are_mixed_directly_when_the_motion_chain_is_broken() -> None:
    track = make_track(0, {0: (400.0, 100.0), 10: (440.0, 100.0)})

    states = fill([track], mapping_records(11), 11, warp=broken_chain)

    assert box_values(states[5].players[0].box) == pytest.approx((410.0, 40.0, 430.0, 100.0))
    assert not states[5].players[0].observed

    def broken_from_the_later_keyframe(src: int, dst: int, xy: np.ndarray) -> np.ndarray | None:
        return None if src == 10 else xy + np.array([30.0, 0.0])

    states = fill([track], mapping_records(11), 11, warp=broken_from_the_later_keyframe)
    assert box_values(states[5].players[0].box) == pytest.approx((410.0, 40.0, 430.0, 100.0))


def test_box_that_cannot_be_moved_is_mixed_directly() -> None:
    def to_infinity(src: int, dst: int, xy: np.ndarray) -> np.ndarray:
        return np.full_like(xy, np.inf)

    track = make_track(0, {0: (400.0, 100.0), 10: (440.0, 100.0)})

    states = fill([track], mapping_records(11), 11, warp=to_infinity)

    assert box_values(states[5].players[0].box) == pytest.approx((410.0, 40.0, 430.0, 100.0))


def test_box_is_the_frame_around_all_four_moved_corners() -> None:
    angle = np.deg2rad(30.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    def rolled(src: int, dst: int, xy: np.ndarray) -> np.ndarray:
        """Motion of a camera that rolls by 30 degrees between a keyframe and any other frame."""
        return xy if src == dst else xy @ rotation.T

    track = make_track(0, {0: (400.0, 100.0), 10: (400.0, 100.0)})

    states = fill([track], mapping_records(11), 11, warp=rolled)

    corners = np.array([[390.0, 40.0], [410.0, 40.0], [410.0, 100.0], [390.0, 100.0]]) @ rotation.T
    # Two opposite corners alone would give a box that is too narrow.
    expected = (*corners.min(axis=0), *corners.max(axis=0))
    assert box_values(states[5].players[0].box) == pytest.approx(expected)
    assert expected[2] - expected[0] > 1.5 * abs(corners[2, 0] - corners[0, 0])


def test_box_with_one_corner_at_infinity_is_mixed_directly() -> None:
    def one_corner_to_infinity(src: int, dst: int, xy: np.ndarray) -> np.ndarray:
        moved = xy.copy()
        moved[np.argmax(moved[:, 0] + moved[:, 1]), 0] = np.inf  # the bottom-right corner
        return moved

    track = make_track(0, {0: (400.0, 100.0), 10: (440.0, 100.0)})

    states = fill([track], mapping_records(11), 11, warp=one_corner_to_infinity)

    assert box_values(states[5].players[0].box) == pytest.approx((410.0, 40.0, 430.0, 100.0))


def test_box_that_is_moved_onto_one_point_is_mixed_directly() -> None:
    def onto_one_point(src: int, dst: int, xy: np.ndarray) -> np.ndarray:
        return np.full_like(xy, 7.0)

    track = make_track(0, {0: (400.0, 100.0), 10: (440.0, 100.0)})

    states = fill([track], mapping_records(11), 11, warp=onto_one_point)

    assert box_values(states[5].players[0].box) == pytest.approx((410.0, 40.0, 430.0, 100.0))


def test_boxes_follow_a_motion_field() -> None:
    motion = MotionField([np.array([[1.0, 0.0, 4.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])] * 10)  # 4 px right per frame
    track = make_track(0, {0: (400.0, 100.0), 10: (440.0, 100.0)})  # a person who stands still on the pitch

    states = fill([track], mapping_records(11), 11, warp=motion.warp)

    assert box_values(states[3].players[0].box) == pytest.approx((402.0, 40.0, 422.0, 100.0))
    assert box_values(states[7].players[0].box) == pytest.approx((418.0, 40.0, 438.0, 100.0))


def test_boxes_use_the_two_observations_around_the_frame() -> None:
    track = make_track(0, {0: (400.0, 100.0), 4: (440.0, 100.0), 8: (440.0, 180.0)})

    states = fill([track], mapping_records(9), 9, warp=static_camera)

    assert box_values(states[2].players[0].box) == pytest.approx((410.0, 40.0, 430.0, 100.0))
    assert box_values(states[4].players[0].box) == pytest.approx((430.0, 40.0, 450.0, 100.0))
    assert box_values(states[6].players[0].box) == pytest.approx((430.0, 80.0, 450.0, 140.0))
    assert [frame for frame, state in enumerate(states) if state.players[0].observed] == [0, 4, 8]


# --- track attributes --------------------------------------------------------


def test_attributes_come_from_the_track() -> None:
    track = make_track(
        7,
        {0: (400.0, 100.0), 4: (420.0, 100.0)},
        role="goalkeeper",
        team_key="B",
        team_side="right",
        number_export=1,
        number_display=13,
        name="Keeper Example",
    )

    states = fill([track], mapping_records(5), 5, warp=static_camera)

    for state in states:  # the observations say "player" of team A, the voted track fields win
        player = state.players[0]
        assert player.track_id == 7
        assert player.role == "goalkeeper"
        assert player.team_key == "B"
        assert player.team_side == "right"
        assert player.number_export == 1
        assert player.number_display == 13
        assert player.name == "Keeper Example"


def test_track_without_a_team_key_gets_the_label_none() -> None:
    referee = make_track(2, {0: (400.0, 100.0)}, role="referee", team_key=None)

    player = fill([referee], mapping_records(1), 1, warp=static_camera)[0].players[0]

    assert player.role == "referee"
    assert player.team_key == "none"
    assert player.team_side is None
    assert player.number_export is None
    assert player.number_display is None
    assert player.name is None


def test_tracks_without_a_role_are_skipped() -> None:
    undecided = make_track(0, {0: (400.0, 100.0), 2: (410.0, 100.0)}, role=None)
    decided = make_track(1, {0: (300.0, 200.0), 2: (310.0, 200.0)})

    states = fill([undecided, decided], mapping_records(3), 3, warp=static_camera)

    assert [[player.track_id for player in state.players] for state in states] == [[1], [1], [1]]


def test_players_keep_the_order_of_the_tracks() -> None:
    tracks = [
        make_track(5, {0: (300.0, 200.0), 4: (310.0, 200.0)}),
        make_track(2, {2: (500.0, 300.0), 4: (510.0, 300.0)}),
        make_track(9, {0: (700.0, 100.0), 2: (710.0, 100.0)}),
    ]

    states = fill(tracks, mapping_records(5), 5, warp=static_camera)

    ids_per_frame = [[player.track_id for player in state.players] for state in states]
    assert ids_per_frame == [[5, 9], [5, 9], [5, 2, 9], [5, 2], [5, 2]]


# --- frames and wrong input --------------------------------------------------


def test_one_frame_state_per_frame_even_without_tracks() -> None:
    states = fill([], [], 3, warp=static_camera)
    assert states == [FrameState(frame=0), FrameState(frame=1), FrameState(frame=2)]


def test_wrong_input_raises() -> None:
    track = make_track(0, {0: (400.0, 100.0), 10: (440.0, 100.0)})
    with pytest.raises(ValueError):
        fill([track], mapping_records(5), 5, warp=static_camera)  # an observation after the clip end
    with pytest.raises(ValueError):
        fill([track, track], mapping_records(11), 11, warp=static_camera)  # the same track id twice
    with pytest.raises(ValueError):
        fill([track], mapping_records(11) + mapping_records(1), 11, warp=static_camera)  # frame 0 twice
    with pytest.raises(ValueError):
        fill([track], mapping_records(12), 11, warp=static_camera)  # a record after the clip end
    with pytest.raises(ValueError):
        fill([], [], 0, warp=static_camera)
    last_frame_is_the_frame_count = make_track(0, {0: (400.0, 100.0), 5: (440.0, 100.0)})
    with pytest.raises(ValueError, match="frame 5"):
        fill([last_frame_is_the_frame_count], mapping_records(5), 5, warp=static_camera)


@pytest.mark.parametrize("margin", [-1.0, -100.0, float("nan")])
def test_margin_must_not_be_negative(margin: float) -> None:
    track = make_track(0, {0: (400.0, 100.0), 10: (440.0, 100.0)})
    with pytest.raises(ValueError, match="pitch_margin_m"):
        fill([track], mapping_records(11), 11, warp=static_camera, pitch_margin_m=margin)


@pytest.mark.parametrize("image_size", [(0, 600), (1200, 0), (-1200, 600)])
def test_image_size_must_be_positive(image_size: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="image_size"):
        fill([], [], 3, warp=static_camera, image_size=image_size)


def test_no_frame_is_filled_across_a_cut() -> None:
    track = make_track(0, {0: (500.0, 250.0), 10: (600.0, 250.0), 20: (700.0, 250.0)})
    states = fill([track], mapping_records(21), 21, warp=static_camera, cuts=(15,))
    filled = [state.frame for state in states if state.players]
    # Frames 0 to 10 are filled inside the first shot; frame 20 is the only state of the second.
    assert filled == [*range(0, 11), 20]
    assert [state.players[0].observed for state in states if state.frame in (10, 20)] == [True, True]
    without_cut = fill([track], mapping_records(21), 21, warp=static_camera)
    assert [state.frame for state in without_cut if state.players] == list(range(21))


def test_a_cut_on_an_observation_frame_ends_the_fill_before_it() -> None:
    track = make_track(0, {0: (500.0, 250.0), 10: (600.0, 250.0), 20: (700.0, 250.0)})
    states = fill([track], mapping_records(21), 21, warp=static_camera, cuts=(10,))
    assert [state.frame for state in states if state.players] == [0, *range(10, 21)]
