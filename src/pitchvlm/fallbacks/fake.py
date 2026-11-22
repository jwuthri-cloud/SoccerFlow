"""Stand-ins of the two external tools that answer from the ground truth of a synthetic scene.

They need no tool, no weights and no network. The demo and the tests run the engines
``detector`` and ``calibration`` of the pipeline with them, without a clip run file:

- ``fake_detector_answers``: the true people of every frame as a detector gives them (role
  ``player``, team key ``none``, no number), with optional misses and box noise, and a ball
  that rolls along a straight line on the grass. The synthetic picture does not show that
  ball; it exists to drive the ball tracker. The ball is not given in a few short runs of
  frames, so the tracker must fill them.
- ``fake_calibration_answers``: the true pitch mapping of every frame.
"""

from __future__ import annotations

import numpy as np

from pitchvlm.ball import BALL_DIAMETER_M, BallCandidate
from pitchvlm.geometry import apply_homography, matrix_to_rows
from pitchvlm.schemas import Box, FramePeople, HomographyRecord, PersonDet
from pitchvlm.synth import SyntheticScene

#: The fake ball starts at this pitch position, metres, and rolls with this velocity, metres per second.
BALL_START_M = (-10.0, 5.0)
BALL_VELOCITY_MPS = (6.0, -2.0)
#: Score of a fake ball candidate.
BALL_SCORE = 0.8
#: The fake ball is not given in frames whose index modulo ``BALL_GAP_PERIOD`` is below ``BALL_GAP_FRAMES``.
BALL_GAP_PERIOD = 40
BALL_GAP_FRAMES = 6


def fake_detector_answers(
    scene: SyntheticScene, *, miss_rate: float = 0.0, box_noise_px: float = 0.0, seed: int = 0
) -> tuple[list[FramePeople], dict[int, tuple[BallCandidate, ...]]]:
    """Return the people of every frame and the ball candidates, as the detector engine reads them from a clip run.

    ``miss_rate`` is the chance that a person is left out of a frame; ``box_noise_px`` is the
    standard deviation of the noise of each box edge, pixels. A box that the noise turns
    inside out is left out. The same seed gives the same answers.
    """
    if not 0.0 <= miss_rate <= 1.0:
        raise ValueError("miss_rate must be between 0 and 1")
    if box_noise_px < 0.0:
        raise ValueError("box_noise_px must not be negative")
    rng = np.random.default_rng([seed, 7])
    people: list[FramePeople] = []
    balls: dict[int, tuple[BallCandidate, ...]] = {}
    for frame in range(scene.info.n_frames):
        found: list[PersonDet] = []
        for person in scene.people(frame).people:
            noise = rng.normal(0.0, box_noise_px, size=4) if box_noise_px > 0.0 else np.zeros(4)
            if rng.random() < miss_rate:
                continue
            x1, y1, x2, y2 = np.array([person.box.x1, person.box.y1, person.box.x2, person.box.y2]) + noise
            if x2 > x1 and y2 > y1:
                found.append(PersonDet(box=Box(x1=x1, y1=y1, x2=x2, y2=y2), role="player", team_key="none"))
        people.append(FramePeople(frame=frame, people=tuple(found)))
        ball = _ball_box(scene, frame)
        if ball is not None and frame % BALL_GAP_PERIOD >= BALL_GAP_FRAMES:
            balls[frame] = (BallCandidate(frame=frame, box=ball, score=BALL_SCORE),)
    return people, balls


def fake_calibration_answers(scene: SyntheticScene) -> list[HomographyRecord]:
    """Return the true pitch mapping of every frame, as the calibration engine reads it from a clip run."""
    records = []
    for frame in range(scene.info.n_frames):
        matrix = scene.homography(frame)
        records.append(
            HomographyRecord(frame=frame, matrix=matrix_to_rows(matrix / np.linalg.norm(matrix)), source="landmarks")
        )
    return records


def fake_ball_position(scene: SyntheticScene, frame: int) -> tuple[float, float]:
    """Return the true pitch position of the fake ball in a frame, metres."""
    seconds = frame / scene.info.fps
    return (BALL_START_M[0] + BALL_VELOCITY_MPS[0] * seconds, BALL_START_M[1] + BALL_VELOCITY_MPS[1] * seconds)


def _ball_box(scene: SyntheticScene, frame: int) -> Box | None:
    """Return the image box of the fake ball in a frame, or None when it lies outside the image."""
    to_image = np.linalg.inv(scene.homography(frame))
    x, y = fake_ball_position(scene, frame)
    (foot_x, foot_y), (side_x, _) = apply_homography(to_image, np.array([[x, y], [x + BALL_DIAMETER_M, y]]))
    width = max(abs(side_x - foot_x), 2.0)
    box = Box(x1=foot_x - width / 2.0, y1=foot_y - width, x2=foot_x + width / 2.0, y2=foot_y)
    inside = box.x1 >= 0.0 and box.y1 >= 0.0 and box.x2 <= scene.info.width and box.y2 <= scene.info.height
    return box if inside else None
