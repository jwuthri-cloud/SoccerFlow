"""A small tracker for the ball: one ball candidate per frame, chosen by motion.

The ball is small and fast. Linking it by box overlap fails: in one frame at 25 frames per
second a shot moves the ball by more than its own size. So the ball gets its own tracker,
apart from the people (the idea of a separate ball tracker is common in detector stacks).

The detector engine gives the candidates: the detections with the ball class of every
frame (``fallbacks.rfdetr.ball_candidates``). The tracker works in image pixels and takes
the camera motion out with the warp of the pipeline (``motion.bridged_warp``):

1. A ball that is followed is moved into the next frame with the warp and with its last
   velocity. The candidate nearest to that point is taken when it lies within the gate.
2. The gate is the distance that a ball at ``MAX_BALL_SPEED_MPS`` covers in the frames since
   the last sighting, in pixels at the scale of the last box (``BALL_DIAMETER_M`` gives the
   pixels per metre), plus one box width. No value is fitted to data.
3. A ball that was not seen for more than ``max_gap_frames`` frames is lost. Then the
   candidate with the highest score starts a new run.
4. A run with fewer than ``MIN_RUN_SIGHTINGS`` sightings is dropped: a single false detection
   does not become a ball.
5. Inside a run, a frame between two sightings gets a box: both boxes are moved into the
   frame with the warp and blended linearly. Where the warp has no answer, the frame has no
   ball.

Limits, not measured: a false detection that stays where it is (a penalty spot, a head) can
hold a run while the real ball is not detected; a ball in the air gets a wrong pitch
position (``schemas.BallState``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import Field

from pitchvlm.motion import MotionWarp
from pitchvlm.schemas import Box, FiniteFloat, Record

#: Diameter of a match ball (size 5: 68 to 70 cm around), metres.
BALL_DIAMETER_M = 0.22
#: Fastest ball that the gate allows, metres per second (144 km/h). An estimate above the
#: speed of hard shots; it is a limit of the gate and not a measured value.
MAX_BALL_SPEED_MPS = 40.0
#: A run of the ball needs at least this many sightings.
MIN_RUN_SIGHTINGS = 3


class BallCandidate(Record):
    """One detection of the ball class in one frame."""

    frame: int = Field(ge=0)
    box: Box
    score: FiniteFloat = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class BallTrack:
    """The ball as the tracker found it: the box of each frame that has one.

    ``observed`` holds the frames where a candidate was taken; the other frames of ``boxes``
    were filled in between two sightings of one run.
    """

    boxes: Mapping[int, Box]
    observed: frozenset[int]


def track_ball(
    candidates: Mapping[int, Sequence[BallCandidate]],
    n_frames: int,
    *,
    fps: float,
    warp: MotionWarp,
    max_gap_frames: int,
) -> BallTrack:
    """Choose at most one ball candidate per frame, and fill the frames between the sightings of a run.

    ``candidates`` maps a frame to its candidates; a frame that is not in it has none. A
    candidate of a frame outside the clip, a candidate stored under another frame than its
    own, ``fps`` that is not a finite number above 0 and a negative ``max_gap_frames`` raise
    ``ValueError``.
    """
    if not 0.0 < fps < float("inf"):
        raise ValueError(f"fps must be a finite number above 0, got {fps}")
    if max_gap_frames < 0:
        raise ValueError(f"max_gap_frames must not be negative, got {max_gap_frames}")
    for frame, found in candidates.items():
        if not 0 <= frame < n_frames or any(candidate.frame != frame for candidate in found):
            raise ValueError(f"the candidates of frame {frame} do not belong to frame {frame} of a clip of {n_frames}")
    runs = _runs(candidates, n_frames, fps, warp, max_gap_frames)
    kept = [run for run in runs if len(run) >= MIN_RUN_SIGHTINGS]
    boxes: dict[int, Box] = {}
    observed: set[int] = set()
    for run in kept:
        for (frame, box), (next_frame, next_box) in zip(run, run[1:], strict=False):
            boxes[frame] = box
            observed.add(frame)
            for between in range(frame + 1, next_frame):
                filled = _blend(frame, box, next_frame, next_box, between, warp)
                if filled is not None:
                    boxes[between] = filled
        last_frame, last_box = run[-1]
        boxes[last_frame] = last_box
        observed.add(last_frame)
    return BallTrack(boxes=dict(sorted(boxes.items())), observed=frozenset(observed))


def gate_px(box: Box, frames: int, fps: float) -> float:
    """Return how far from the expected point a candidate may lie, pixels, after ``frames`` frames.

    The ball can cover ``MAX_BALL_SPEED_MPS`` metres per second. The width of the last box is
    the ball diameter, which gives the pixels per metre. One box width is added for the noise
    of the boxes.
    """
    pixels_per_metre = box.width / BALL_DIAMETER_M
    return MAX_BALL_SPEED_MPS / fps * frames * pixels_per_metre + box.width


def _centre(box: Box) -> np.ndarray:
    return np.array([box.center], dtype=np.float64)


def _runs(
    candidates: Mapping[int, Sequence[BallCandidate]],
    n_frames: int,
    fps: float,
    warp: MotionWarp,
    max_gap_frames: int,
) -> list[list[tuple[int, Box]]]:
    """Follow the ball through the clip and return its runs: lists of (frame, box) in frame order."""
    runs: list[list[tuple[int, Box]]] = []
    run: list[tuple[int, Box]] = []
    velocity = np.zeros(2)  # pixels per frame, in the image of the last sighting
    for frame in range(n_frames):
        found = candidates.get(frame, ())
        if run:
            last_frame, last_box = run[-1]
            gap = frame - last_frame
            moved = warp(last_frame, frame, _centre(last_box))
            known = moved is not None and bool(np.isfinite(moved).all())
            if known and found:
                expected = moved[0] + velocity * gap
                limit = gate_px(last_box, gap, fps)
                distances = [float(np.linalg.norm(_centre(item.box)[0] - expected)) for item in found]
                best = min(range(len(found)), key=lambda index: (distances[index], -found[index].score))
                if distances[best] <= limit:
                    velocity = (_centre(found[best].box)[0] - moved[0]) / gap
                    run.append((frame, found[best].box))
                    continue
            # This frame has no sighting of the run: it is the gap-th frame without one.
            if not known or gap > max_gap_frames:
                runs.append(run)
                run = []
        if not run and found:
            best_candidate = max(found, key=lambda item: item.score)
            run = [(frame, best_candidate.box)]
            velocity = np.zeros(2)
    if run:
        runs.append(run)
    return runs


def _blend(frame: int, box: Box, next_frame: int, next_box: Box, between: int, warp: MotionWarp) -> Box | None:
    """Return the box of the ball in a frame between two sightings, or None where the warp has no answer.

    Both centres are moved into the frame and blended linearly in time; the size is blended too.
    """
    start = warp(frame, between, _centre(box))
    end = warp(next_frame, between, _centre(next_box))
    if start is None or end is None or not (np.isfinite(start).all() and np.isfinite(end).all()):
        return None
    share = (between - frame) / (next_frame - frame)
    centre_x, centre_y = (1.0 - share) * start[0] + share * end[0]
    width = (1.0 - share) * box.width + share * next_box.width
    height = (1.0 - share) * box.height + share * next_box.height
    return Box(x1=centre_x - width / 2, y1=centre_y - height / 2, x2=centre_x + width / 2, y2=centre_y + height / 2)
