"""Tests of the hard-cut test of motion.py and of the cuts of a motion field.

The synthetic clips come from ``synth.RetimedScene``: a jump in its scene frames is a hard cut,
a step of 10 scene frames per clip frame is a fast pan (83 pixels per frame at 1280 pixels).
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from pitchvlm.geometry import matrix_to_rows
from pitchvlm.motion import (
    MIN_SHOT_CORRELATION,
    MotionField,
    bridged_warp,
    compute_motion_field,
    is_cut,
    propagate_homographies,
    shot_correlation,
)
from pitchvlm.schemas import HomographyRecord
from pitchvlm.synth import RetimedScene, ScoreGraphicSource, SyntheticSource

#: Image-to-pitch mapping of a simple top-down picture: 10 px per metre, centre mark at pixel (500, 250).
PITCH_H = np.array([[0.1, 0.0, -50.0], [0.0, 0.1, -25.0], [0.0, 0.0, 1.0]])


def translation(dx: float, dy: float) -> np.ndarray:
    """Return the matrix that moves points by (dx, dy)."""
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def landmark_record(frame: int, matrix: np.ndarray) -> HomographyRecord:
    """Return the record of a keyframe with a landmark fit."""
    return HomographyRecord(frame=frame, matrix=matrix_to_rows(matrix), source="landmarks", n_inliers=8, reproj_px=1.5)


def small_clip(scene_frames: list[int]) -> RetimedScene:
    """Return a small synthetic clip (640 x 360) that shows the given scene frames."""
    return RetimedScene(scene_frames, width=640, height=360, seed=0)


# --- shot_correlation and is_cut ---------------------------------------------


def test_a_frame_correlates_fully_with_itself() -> None:
    image = small_clip([0]).image(0)
    assert shot_correlation(image, image, []) == pytest.approx(1.0)
    assert not is_cut(image, image, [])


def test_the_measured_motion_explains_a_pan() -> None:
    scene = small_clip([0, 10])
    first, second = scene.image(0), scene.image(1)
    # The camera motion between the two frames, from the true pitch mappings.
    motion = np.linalg.inv(scene.homography(1)) @ scene.homography(0)
    assert shot_correlation(first, second, [motion]) > 0.85
    assert shot_correlation(first, second, [motion]) > shot_correlation(first, second, [])
    assert not is_cut(first, second, [motion])


def test_frames_of_two_shots_are_a_cut_with_any_motion() -> None:
    scene = small_clip([0, 154])
    first, second = scene.image(0), scene.image(1)
    assert not is_cut(first, first, [])
    assert shot_correlation(first, second, [translation(3.0, 0.0)]) < MIN_SHOT_CORRELATION
    assert is_cut(first, second, [translation(3.0, 0.0), np.eye(3)])


def test_frames_without_contrast_are_a_cut() -> None:
    flat = np.full((360, 640, 3), 90, dtype=np.uint8)
    picture = small_clip([0]).image(0)
    assert shot_correlation(flat, picture, []) is None
    assert is_cut(flat, picture, [])
    assert is_cut(flat, flat, [])


@pytest.mark.parametrize("shape", [(2, 640), (10, 1920)])
def test_an_image_too_flat_for_the_test_grid_gives_no_correlation(shape: tuple[int, int]) -> None:
    # Shrunk to 160 px wide, the image has fewer rows than the grid: some blocks are empty.
    image = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert shot_correlation(image, image, []) is None


def test_a_motion_that_moves_the_frame_out_of_the_picture_does_not_count() -> None:
    scene = small_clip([0, 1])
    first, second = scene.image(0), scene.image(1)
    away = translation(5000.0, 0.0)
    assert shot_correlation(first, second, [away]) == pytest.approx(shot_correlation(first, second, []))


def test_the_cut_test_takes_grey_and_colour_images_and_refuses_two_sizes() -> None:
    image = small_clip([0]).image(0)
    grey = image.mean(axis=2).astype(np.uint8)
    assert shot_correlation(grey, grey, []) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="differ in size"):
        shot_correlation(image, image[:100], [])


# --- compute_motion_field ----------------------------------------------------


def test_a_clip_without_a_cut_has_no_cut() -> None:
    field = compute_motion_field(SyntheticSource(small_clip(list(range(0, 12)))))
    assert field.cuts == ()
    assert all(field.between(frame, frame + 1) is not None for frame in range(11))


@pytest.mark.parametrize("graphic", [False, True])
def test_a_fast_pan_is_no_cut(graphic: bool) -> None:
    scene = RetimedScene(list(range(0, 160, 10)), seed=0)  # 1280 x 720, up to 83 pixels per frame
    source = ScoreGraphicSource(scene) if graphic else SyntheticSource(scene)
    field = compute_motion_field(source)
    assert field.cuts == ()


@pytest.mark.parametrize("graphic", [False, True])
def test_a_hard_cut_is_found_and_breaks_the_motion_chain(graphic: bool) -> None:
    scene = RetimedScene([*range(0, 6), *range(154, 160)], seed=0)
    source = ScoreGraphicSource(scene) if graphic else SyntheticSource(scene)
    field = compute_motion_field(source)
    assert field.cuts == (6,)
    assert field.between(5, 6) is None
    assert field.between(0, 5) is not None
    assert field.between(6, 11) is not None


def test_two_cuts_are_both_found() -> None:
    scene = RetimedScene([*range(0, 4), *range(100, 104), *range(175, 179)], seed=0)
    assert compute_motion_field(SyntheticSource(scene)).cuts == (4, 8)


def test_a_jump_in_time_that_the_camera_motion_explains_is_not_found() -> None:
    # A known limit. The synthetic camera only turns and zooms, so any two of its frames are one
    # homography apart. When the optical flow can follow the jump, the test sees a camera move,
    # and only the people jump. Scene frames 103 and 250 show almost the same pan and zoom (the
    # pan repeats every 16 s). In the smaller image the jump from 103 to 175 is within reach too.
    assert compute_motion_field(SyntheticSource(small_clip([*range(100, 104), *range(250, 254)]))).cuts == ()
    assert compute_motion_field(SyntheticSource(small_clip([*range(100, 104), *range(175, 179)]))).cuts == ()


# --- MotionField with cuts ---------------------------------------------------


def test_a_motion_field_names_the_shot_of_every_frame() -> None:
    steps: list[np.ndarray | None] = [np.eye(3)] * 9
    steps[3] = None
    steps[6] = None
    field = MotionField(steps, cuts=(4, 7))
    assert [field.shot_of(frame) for frame in range(10)] == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert field.same_shot(4, 6)
    assert not field.same_shot(3, 4)
    assert not field.same_shot(9, 0)
    with pytest.raises(IndexError):
        field.shot_of(10)
    assert MotionField.identity(5).cuts == ()


def test_a_cut_must_break_the_chain_and_lie_inside_the_clip() -> None:
    steps: list[np.ndarray | None] = [np.eye(3), None, np.eye(3)]
    assert MotionField(steps, cuts=[2]).cuts == (2,)
    with pytest.raises(ValueError, match="must be missing"):
        MotionField(steps, cuts=[1])
    with pytest.raises(ValueError, match="not a frame after the first"):
        MotionField(steps, cuts=[0])
    with pytest.raises(ValueError, match="not a frame after the first"):
        MotionField(steps, cuts=[4])
    with pytest.raises(ValueError, match="strictly increasing"):
        MotionField([None, None, np.eye(3)], cuts=[2, 1])
    with pytest.raises(ValueError, match="strictly increasing"):
        MotionField([None, None, np.eye(3)], cuts=[1, 1])
    with pytest.raises(TypeError):
        MotionField(steps, cuts=[True])


def test_the_cuts_survive_a_json_round_trip() -> None:
    field = MotionField([np.eye(3), None, translation(2.0, 0.0)], cuts=(2,))
    again = MotionField.from_json(json.loads(json.dumps(field.to_json())))
    assert again.cuts == (2,)
    with pytest.raises(KeyError):
        MotionField.from_json({"n_frames": 4, "steps": [None, None, None]})


def test_bridged_warp_does_not_cross_a_cut() -> None:
    steps: list[np.ndarray | None] = [translation(5.0, 0.0)] * 9
    steps[2] = None  # a break inside the first shot
    steps[5] = None  # the cut into frame 6
    field = MotionField(steps, cuts=(6,))
    records = [landmark_record(frame, PITCH_H @ translation(-5.0 * frame, 0.0)) for frame in range(10)]
    warp = bridged_warp(field, records)
    points = np.array([[100.0, 80.0]])
    np.testing.assert_allclose(warp(1, 4, points), [[115.0, 80.0]], atol=1e-9)  # bridged inside the shot
    assert warp(4, 7, points) is None
    assert warp(7, 4, points) is None
    assert warp(5, 6, points) is None
    np.testing.assert_allclose(warp(6, 9, points), [[115.0, 80.0]], atol=1e-9)


def test_no_pitch_mapping_is_propagated_across_a_cut() -> None:
    steps: list[np.ndarray | None] = [np.eye(3)] * 9
    steps[4] = None
    field = MotionField(steps, cuts=(5,))
    records = propagate_homographies({3: landmark_record(3, PITCH_H)}, field, 10, max_gap_frames=75)
    assert [record.valid for record in records] == [True] * 5 + [False] * 5
