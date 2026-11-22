"""Tests of the tracking values that need no labels, on clip results made by hand.

The image is 1280 x 720 at 25 frames per second. Pitch positions are given
directly in the frame states, so speeds are easy to check by hand.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

import pytest

from pitchvlm.evalx.diagnostics import BORDER_BOX_HEIGHTS, track_diagnostics
from pitchvlm.schemas import Box, ClipInfo, ClipResult, FrameState, PlayerState, Role, TeamKey, Track, TrackObs

INFO = ClipInfo(clip_id="SNGS-999", n_frames=100, fps=25.0, width=1280, height=720)
BOX_HEIGHT = 40.0


def box_at(x: float, y: float = 400.0) -> Box:
    """Return a box of a person whose foot point is the image point (x, y)."""
    return Box(x1=x - BOX_HEIGHT / 4.0, y1=y - BOX_HEIGHT, x2=x + BOX_HEIGHT / 4.0, y2=y)


def track(
    track_id: int,
    frames: Sequence[int],
    *,
    x: float = 640.0,
    role: Role = "player",
    team_key: TeamKey = "A",
    keys: str | None = None,
) -> Track:
    """Make a decided track of a person at the image point (x, 400). ``keys`` gives a team key per frame: a, b or ."""
    names: dict[str, TeamKey] = {"a": "A", "b": "B", ".": "none"}
    per_frame = [team_key] * len(frames) if keys is None else [names[letter] for letter in keys]
    observations = tuple(
        TrackObs(frame=frame, box=box_at(x), role=role, team_key=key) for frame, key in zip(frames, per_frame)
    )
    return Track(track_id=track_id, observations=observations, role=role, team_key=team_key)


def result_of(
    tracks: Sequence[Track],
    positions: dict[int, dict[int, tuple[float, float]]] | None = None,
    cuts: tuple[int, ...] = (),
) -> ClipResult:
    """Make a clip result. ``positions`` maps a track id to its observed pitch position by frame."""
    positions = positions or {}
    by_frame: dict[int, list[PlayerState]] = {}
    for one in tracks:
        for frame, (x, y) in positions.get(one.track_id, {}).items():
            by_frame.setdefault(frame, []).append(
                PlayerState(track_id=one.track_id, pitch_x=x, pitch_y=y, role=one.role or "player", observed=True)
            )
    frames = tuple(FrameState(frame=frame, players=tuple(by_frame.get(frame, ()))) for frame in range(INFO.n_frames))
    return ClipResult(info=INFO, frames=frames, tracks=tuple(tracks), homographies=(), cuts=cuts)


def test_a_track_that_starts_after_the_clip_start_away_from_the_border_is_a_birth_inside() -> None:
    tracks = [
        track(0, [0, 12, 24]),  # starts with the clip: no birth
        track(1, [12, 24], x=640.0),  # starts later in the middle of the image: a birth inside
        track(2, [12, 24], x=15.0),  # starts later at the left border: a person who entered
        track(3, [0, 12], x=640.0),  # ends before the clip end in the middle: an end inside
    ]

    values = track_diagnostics(result_of(tracks), max_speed_mps=9.0)

    assert (values.tracks, values.births_inside, values.ends_inside) == (4, 1, 1)


def test_the_first_and_last_observations_of_each_shot_bound_births_and_ends() -> None:
    # A cut at frame 50 starts a new shot. The tracks that start with shot 1 (frame 60) and the
    # tracks that end with shot 0 (frame 36) are no births and no ends inside the image.
    tracks = [
        track(0, [0, 12, 24, 36]),  # ends with shot 0
        track(1, [60, 72, 84]),  # starts with shot 1
        track(2, [72, 84]),  # starts later in shot 1, in the middle: a birth inside
    ]

    values = track_diagnostics(result_of(tracks, cuts=(50,)), max_speed_mps=9.0)

    assert (values.births_inside, values.ends_inside) == (1, 0)
    without_cut = track_diagnostics(result_of(tracks), max_speed_mps=9.0)
    assert (without_cut.births_inside, without_cut.ends_inside) == (2, 1)


def test_no_step_crosses_a_cut() -> None:
    # A track that a shown shirt number joined across the cut at frame 50 is 40 m away after it.
    tracks = [track(0, [36, 60, 72])]
    positions = {0: {36: (0.0, 0.0), 60: (40.0, 0.0), 72: (41.0, 0.0)}}

    values = track_diagnostics(result_of(tracks, positions, cuts=(50,)), max_speed_mps=9.0)

    assert (values.steps, values.impossible_steps) == (1, 0)
    assert values.top_speed_mps["median"] == pytest.approx(1.0 / 0.48)
    without_cut = track_diagnostics(result_of(tracks, positions), max_speed_mps=9.0)
    assert (without_cut.steps, without_cut.impossible_steps) == (2, 1)


def test_the_border_is_one_box_height_wide() -> None:
    assert BORDER_BOX_HEIGHTS == 1.0
    # The box is 20 px wide; its left edge is at x - 10. One box height is 40 px.
    at_border = track(1, [12, 24], x=10.0 + 39.0)
    inside = track(2, [12, 24], x=10.0 + 40.0)

    values = track_diagnostics(result_of([track(0, [0, 24]), at_border, inside]), max_speed_mps=9.0)

    assert values.births_inside == 1


def test_steps_faster_than_the_limit_are_counted_between_observed_positions() -> None:
    # Track 0 moves 2 m per keyframe (0.48 s): 4.2 m/s. Track 1 jumps 6 m in one keyframe: 12.5 m/s.
    tracks = [track(0, [0, 12, 24]), track(1, [0, 12, 24])]
    positions = {
        0: {0: (0.0, 0.0), 12: (2.0, 0.0), 24: (4.0, 0.0)},
        1: {0: (10.0, 0.0), 12: (16.0, 0.0), 24: (16.0, 0.0)},
    }

    values = track_diagnostics(result_of(tracks, positions), max_speed_mps=9.0)

    assert (values.steps, values.impossible_steps) == (4, 1)
    assert values.top_speed_mps["n"] == 2
    assert values.top_speed_mps["median"] == pytest.approx((2.0 / 0.48 + 6.0 / 0.48) / 2)


def test_filled_positions_are_not_steps() -> None:
    one = track(0, [0, 24])
    state = [
        FrameState(
            frame=frame,
            players=(
                PlayerState(track_id=0, pitch_x=float(frame), pitch_y=0.0, role="player", observed=frame in (0, 24)),
            ),
        )
        for frame in range(25)
    ]
    frames = tuple(state) + tuple(FrameState(frame=frame) for frame in range(25, INFO.n_frames))
    result = ClipResult(info=INFO, frames=frames, tracks=(one,), homographies=())

    values = track_diagnostics(result, max_speed_mps=9.0)

    assert values.steps == 1
    assert values.top_speed_mps["median"] == pytest.approx(24.0 / (24 / 25.0))


def test_team_key_changes_count_the_runs_inside_tracks_of_players_and_goalkeepers() -> None:
    frames = [12 * index for index in range(8)]
    tracks = [
        track(0, frames, keys="aaaabbbb"),  # one change
        track(1, frames, keys="aaabaaaa"),  # one wrong answer: no change
        track(2, frames, role="referee", team_key="none", keys="aaaabbbb"),  # a referee has no team
    ]

    values = track_diagnostics(result_of(tracks), max_speed_mps=9.0)

    assert (values.team_changes, values.tracks_with_team_change) == (1, 1)


def test_tracks_per_team_over_time() -> None:
    tracks = [
        track(0, [0, 12, 24], team_key="A"),
        track(1, [0, 24], team_key="A"),  # alive at frame 12, but not seen there
        track(2, [12], team_key="B"),
        track(3, [0, 12], role="referee", team_key="none"),
    ]

    values = track_diagnostics(result_of(tracks), max_speed_mps=9.0)

    assert values.alive_per_keyframe == ((0, 2, 0), (12, 2, 1), (24, 2, 0))
    assert values.teams["A"].ids == 2
    assert values.teams["A"].max_alive == 2
    assert values.teams["A"].max_seen == 2
    assert values.teams["B"].ids == 1
    assert values.teams["A"].keyframes_over_slots == 0


def test_keyframes_with_more_tracks_alive_than_a_team_has_people_are_counted() -> None:
    tracks = [track(index, [0, 12], x=50.0 + 100.0 * index) for index in range(12)]

    values = track_diagnostics(result_of(tracks), max_speed_mps=9.0)

    assert values.teams["A"].max_alive == 12
    assert values.teams["A"].keyframes_over_slots == 2


def test_diagnostics_are_ready_for_json_and_for_the_terminal() -> None:
    tracks = [track(0, [0, 12, 24]), track(1, [12, 24])]
    positions = {0: {0: (0.0, 0.0), 12: (1.0, 0.0), 24: (2.0, 0.0)}}

    values = track_diagnostics(result_of(tracks, positions), max_speed_mps=9.0)

    document = json.loads(json.dumps(values.as_dict()))
    assert document["births_inside"] == 1
    assert document["teams"]["A"]["ids"] == 2
    assert document["alive_per_keyframe"] == [[0, 1, 0], [12, 2, 0], [24, 2, 0]]
    lines = values.lines()
    assert lines[0] == "tracks: 2; started inside the image: 1, ended inside the image: 0"
    assert lines[1].startswith("steps faster than 9 m/s: 0 of 2; top speed per track: median 2.1")
    assert lines[2] == "team key changes inside a track: 0 in 0 tracks"
    assert lines[3].startswith("team A: 2 ids, at most 2 alive and 2 seen at one keyframe")


def test_a_clip_without_tracks_has_empty_values() -> None:
    values = track_diagnostics(result_of([]), max_speed_mps=9.0)

    assert (values.tracks, values.steps, values.team_changes) == (0, 0, 0)
    assert math.isnan(values.top_speed_mps["median"])
    assert values.alive_per_keyframe == ()
    # JSON has no number for nan: the mapping holds None there.
    document = json.loads(json.dumps(values.as_dict(), allow_nan=False))
    assert document["top_speed_mps"] == {"n": 0.0, "median": None, "p90": None, "mean": None}


@pytest.mark.parametrize("speed", [0.0, -1.0, math.inf, math.nan])
def test_diagnostics_refuse_a_wrong_speed_limit(speed: float) -> None:
    with pytest.raises(ValueError, match="max_speed_mps"):
        track_diagnostics(result_of([]), max_speed_mps=speed)
