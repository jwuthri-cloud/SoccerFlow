"""Tests of the ball tracker (``ball``)."""

from __future__ import annotations

import numpy as np
import pytest

from pitchvlm.ball import BALL_DIAMETER_M, MAX_BALL_SPEED_MPS, MIN_RUN_SIGHTINGS, BallCandidate, gate_px, track_ball
from pitchvlm.schemas import Box

FPS = 25.0
SIZE = 10.0


def still(src: int, dst: int, xy: np.ndarray) -> np.ndarray:
    """Warp of a camera that does not move."""
    return xy


def candidate(frame: int, x: float, y: float = 300.0, score: float = 0.8) -> BallCandidate:
    return BallCandidate(
        frame=frame, box=Box(x1=x - SIZE / 2, y1=y - SIZE / 2, x2=x + SIZE / 2, y2=y + SIZE / 2), score=score
    )


def rolling(frames: range, speed_px: float = 8.0) -> dict[int, list[BallCandidate]]:
    """A ball that rolls to the right by ``speed_px`` pixels per frame."""
    return {frame: [candidate(frame, 100.0 + speed_px * frame)] for frame in frames}


def centre_x(box: Box) -> float:
    return box.center[0]


def test_ball_seen_in_every_frame_is_taken_in_every_frame() -> None:
    track = track_ball(rolling(range(20)), 20, fps=FPS, warp=still, max_gap_frames=12)
    assert sorted(track.boxes) == list(range(20))
    assert track.observed == frozenset(range(20))
    assert centre_x(track.boxes[7]) == pytest.approx(156.0)


def test_short_gap_is_filled_on_the_line_between_the_sightings() -> None:
    candidates = rolling([*range(0, 5), *range(10, 15)])
    track = track_ball(candidates, 15, fps=FPS, warp=still, max_gap_frames=12)
    assert sorted(track.boxes) == list(range(15))
    assert track.observed == frozenset([*range(0, 5), *range(10, 15)])
    for frame in range(5, 10):
        assert centre_x(track.boxes[frame]) == pytest.approx(100.0 + 8.0 * frame)


def test_long_gap_ends_the_run_and_its_frames_have_no_ball() -> None:
    candidates = rolling([*range(0, 5), *range(20, 25)])
    track = track_ball(candidates, 25, fps=FPS, warp=still, max_gap_frames=5)
    assert sorted(track.boxes) == [*range(0, 5), *range(20, 25)]


def test_nearest_candidate_to_the_expected_point_is_taken() -> None:
    candidates = rolling(range(10))
    # A false candidate with a higher score far from the ball in frame 6.
    candidates[6].insert(0, candidate(6, 700.0, score=0.99))
    track = track_ball(candidates, 10, fps=FPS, warp=still, max_gap_frames=5)
    assert centre_x(track.boxes[6]) == pytest.approx(148.0)


def test_single_false_detection_does_not_become_a_ball() -> None:
    candidates = {4: [candidate(4, 500.0)]}
    assert track_ball(candidates, 10, fps=FPS, warp=still, max_gap_frames=5).boxes == {}
    # A run needs MIN_RUN_SIGHTINGS sightings.
    assert len(track_ball(rolling(range(MIN_RUN_SIGHTINGS)), 10, fps=FPS, warp=still, max_gap_frames=5).boxes) == 3
    assert track_ball(rolling(range(MIN_RUN_SIGHTINGS - 1)), 10, fps=FPS, warp=still, max_gap_frames=5).boxes == {}


def test_jump_beyond_the_gate_is_not_linked() -> None:
    candidates = rolling(range(5))
    for frame in range(5, 10):
        candidates[frame] = [candidate(frame, 1100.0 + 8.0 * frame)]
    track = track_ball(candidates, 10, fps=FPS, warp=still, max_gap_frames=2)
    # The ball of frames 0 to 4 waits for 2 frames, then the far candidates start a new run.
    assert sorted(track.boxes) == [0, 1, 2, 3, 4, 7, 8, 9]
    assert centre_x(track.boxes[8]) == pytest.approx(1164.0)


def test_gate_follows_the_speed_limit_and_the_size_of_the_ball() -> None:
    box = Box(x1=0, y1=0, x2=SIZE, y2=SIZE)
    per_frame = MAX_BALL_SPEED_MPS / FPS * SIZE / BALL_DIAMETER_M
    assert gate_px(box, 1, FPS) == pytest.approx(per_frame + SIZE)
    assert gate_px(box, 3, FPS) == pytest.approx(3 * per_frame + SIZE)


def test_camera_motion_is_taken_out() -> None:
    """The camera pans: a ball that lies still on the grass moves 30 pixels per frame in the image."""

    def pan(src: int, dst: int, xy: np.ndarray) -> np.ndarray:
        return xy + np.array([30.0 * (dst - src), 0.0])

    candidates = {frame: [candidate(frame, 100.0 + 30.0 * frame)] for frame in (*range(0, 4), *range(8, 12))}
    # A false candidate where the ball would be without the camera motion.
    candidates[8].append(candidate(8, 100.0))
    track = track_ball(candidates, 12, fps=FPS, warp=pan, max_gap_frames=5)
    assert centre_x(track.boxes[8]) == pytest.approx(340.0)
    assert centre_x(track.boxes[5]) == pytest.approx(250.0)


def test_warp_without_an_answer_ends_the_run_and_fills_nothing() -> None:
    def broken_after_4(src: int, dst: int, xy: np.ndarray) -> np.ndarray | None:
        return None if max(src, dst) > 4 and min(src, dst) <= 4 else xy

    candidates = rolling([*range(0, 4), *range(6, 10)])
    track = track_ball(candidates, 10, fps=FPS, warp=broken_after_4, max_gap_frames=5)
    assert 4 not in track.boxes and 5 not in track.boxes
    assert sorted(track.boxes) == [0, 1, 2, 3, 6, 7, 8, 9]


@pytest.mark.parametrize(
    ("candidates", "fps", "gap"),
    [
        ({}, 0.0, 5),
        ({}, float("inf"), 5),
        ({}, FPS, -1),
        ({3: [candidate(4, 100.0)]}, FPS, 5),
        ({12: [candidate(12, 100.0)]}, FPS, 5),
    ],
)
def test_wrong_input_is_refused(candidates: dict[int, list[BallCandidate]], fps: float, gap: int) -> None:
    with pytest.raises(ValueError):
        track_ball(candidates, 10, fps=fps, warp=still, max_gap_frames=gap)
