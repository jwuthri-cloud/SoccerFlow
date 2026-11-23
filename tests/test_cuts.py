"""Tests of the hard-cut test and the dissolve test of motion.py, and of the cuts of a motion field.

The synthetic clips come from ``synth.RetimedScene``: a jump in its scene frames is a hard cut,
a step of 10 scene frames per clip frame is a fast pan (83 pixels per frame at 1280 pixels).
A dissolve blends two such clips (``BlendSource``). The window test also runs on plain test
images (``CUT_TEST_WIDTH`` wide) with known motion.
"""

from __future__ import annotations

import json
import warnings

import cv2
import numpy as np
import pytest

from pitchvlm import motion
from pitchvlm.geometry import matrix_to_rows
from pitchvlm.motion import (
    DISSOLVE_WINDOW_FRAMES,
    MAX_TRANSITION_CORRELATION,
    MIN_SHOT_CORRELATION,
    MotionField,
    ShotTests,
    bridged_warp,
    compute_motion_field,
    find_transitions,
    is_cut,
    measure_shots,
    propagate_homographies,
    shot_correlation,
    window_correlations,
)
from pitchvlm.schemas import ClipInfo, HomographyRecord
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


# --- The dissolve test on test images with known motion ----------------------------------------

#: Size (width, height) of the images of the cut tests.
TEST_SIZE = (160, 90)


def texture(seed: int) -> np.ndarray:
    """Return a ``float32`` test image with smooth random structure, grey levels around 128."""
    rng = np.random.default_rng(seed)
    total = np.zeros((TEST_SIZE[1], TEST_SIZE[0]), dtype=np.float32)
    for blur_px in (1.5, 4.0):
        layer = cv2.GaussianBlur(rng.random(total.shape, dtype=np.float32), (0, 0), blur_px)
        total += (layer - layer.mean()) / layer.std()
    return 128.0 + 30.0 * total


def two_shots(length: int, *, before: int = 40, after: int = 40) -> list[np.ndarray]:
    """Return a still shot, ``length`` frames that blend it into a second still shot, and the second shot.

    Frame ``before + i`` of the blend shows the second shot with the weight ``(i + 1) / (length + 1)``.
    """
    old, new = texture(1), texture(2)
    blends = [old + (i + 1) / (length + 1) * (new - old) for i in range(length)]
    return [old] * before + blends + [new] * after


def still(n_frames: int) -> list[np.ndarray | None]:
    """Return the steps of a camera that does not move."""
    return [np.eye(3)] * (n_frames - 1)


def test_a_still_shot_explains_every_window() -> None:
    images = [texture(1)] * 50
    values = window_correlations(images, still(50))
    assert values[:DISSOLVE_WINDOW_FRAMES] == [None] * DISSOLVE_WINDOW_FRAMES
    assert all(value is not None and value > 0.99 for value in values[DISSOLVE_WINDOW_FRAMES:])
    assert find_transitions(images, still(50), values) == []


def test_a_dissolve_is_found_with_its_blend_frames() -> None:
    images = two_shots(5)
    values = window_correlations(images, still(len(images)))
    assert min(value for value in values if value is not None) < MAX_TRANSITION_CORRELATION
    # The blend weights are 1/6 to 5/6: every blend frame shows less than MIN_SHOT_SHARE of either shot.
    assert find_transitions(images, still(len(images)), values) == [(40, 45)]


def test_frames_that_show_a_shot_almost_alone_stay_in_that_shot() -> None:
    # Weights 1/26 to 25/26: the first and the last frame of the blend show 96% of one shot,
    # more than MIN_SHOT_SHARE, so they stay in that shot; the next frames show 92%.
    images = two_shots(25)
    values = window_correlations(images, still(len(images)))
    assert find_transitions(images, still(len(images)), values) == [(41, 64)]


def test_a_change_without_a_blend_frame_is_a_cut() -> None:
    images = two_shots(0)
    values = window_correlations(images, still(len(images)))
    assert find_transitions(images, still(len(images)), values) == [(40, 40)]


def test_a_window_with_a_missing_step_is_not_judged() -> None:
    images = two_shots(5)
    steps = still(len(images))
    steps[42] = None
    values = window_correlations(images, steps)
    assert values[43 : 43 + DISSOLVE_WINDOW_FRAMES] == [None] * DISSOLVE_WINDOW_FRAMES
    assert find_transitions(images, steps, values) == []


def test_the_window_test_checks_its_input() -> None:
    images = [texture(1)] * 5
    with pytest.raises(ValueError, match="need 4 steps"):
        window_correlations(images, still(4))
    with pytest.raises(ValueError, match="need 5 window values"):
        find_transitions(images, still(5), [None] * 4)


def test_a_chain_that_follows_a_pan_explains_its_window() -> None:
    image = texture(3)
    images = [np.roll(image, frame, axis=1) for frame in range(DISSOLVE_WINDOW_FRAMES + 1)]
    steps: list[np.ndarray | None] = [translation(1.0, 0.0)] * DISSOLVE_WINDOW_FRAMES
    value = window_correlations(images, steps)[DISSOLVE_WINDOW_FRAMES]
    assert value is not None and value > 0.99
    # The identity does not explain the window: only the chained motion is tried.
    assert window_correlations(images, still(len(images)))[DISSOLVE_WINDOW_FRAMES] < MAX_TRANSITION_CORRELATION


# --- The dissolve test on synthetic clips ------------------------------------------------------


class BlendSource:
    """Two synthetic clips that are blended over ``length`` frames from frame ``start`` on.

    Frame ``start + i`` shows the second clip with the weight ``(i + 1) / (length + 1)``.
    """

    def __init__(self, old: RetimedScene, new: RetimedScene, start: int, length: int) -> None:
        self._old, self._new = SyntheticSource(old), SyntheticSource(new)
        self._start, self._length = start, length
        width, height = old.info.width, old.info.height
        self.info = ClipInfo(clip_id="blend", n_frames=old.info.n_frames, fps=25.0, width=width, height=height)

    def read(self, frame: int) -> np.ndarray:
        if frame < self._start:
            return self._old.read(frame)
        if frame >= self._start + self._length:
            return self._new.read(frame)
        weight = (frame - self._start + 1) / (self._length + 1)
        blend = (1.0 - weight) * self._old.read(frame).astype(np.float32) + weight * self._new.read(frame)
        return np.clip(blend, 0, 255).astype(np.uint8)


def dissolve_clip(length: int) -> BlendSource:
    """Return a small clip whose frames 40 to ``40 + length - 1`` blend two shots 100 scene frames apart."""
    n_frames = 80 + length
    return BlendSource(small_clip(list(range(n_frames))), small_clip(list(range(100, 100 + n_frames))), 40, length)


@pytest.mark.parametrize("length", [5, 25])
def test_a_dissolve_in_a_clip_is_found_and_every_frame_of_it_starts_a_shot(length: int) -> None:
    tests = measure_shots(dissolve_clip(length))
    assert tests.hard_cuts == ()  # the pair test alone does not see the dissolve
    [(start, stop)] = tests.transitions
    assert 40 <= start <= 40 + length // 4  # synthetic seed 0: at most 4 frames late at 25 frames
    assert 40 + length - length // 4 <= stop <= 40 + length
    field = compute_motion_field(dissolve_clip(length))
    assert field.cuts == tuple(range(start, stop + 1))
    assert field.transition_frames() == frozenset(range(start, stop))
    assert field.between(start - 1, start) is None
    assert field.between(stop - 1, stop) is None
    assert field.between(0, start - 1) is not None
    assert field.between(stop, field.n_frames - 1) is not None


def test_shot_tests_name_the_hard_cuts_and_every_frame_of_a_transition() -> None:
    tests = ShotTests(
        steps=(None,) * 9,
        test_images=(),
        test_steps=(),
        pair_correlations=(None,) * 9,
        hard_cuts=(2, 8),
        window_correlations=(None,) * 10,
        transitions=((4, 6), (7, 7)),
    )
    assert tests.cuts() == (2, 4, 5, 6, 7, 8)


def test_a_shot_of_one_frame_next_to_a_cut_is_a_transition_frame() -> None:
    def field(n_frames: int, cuts: tuple[int, ...]) -> MotionField:
        return MotionField([None if frame + 1 in cuts else np.eye(3) for frame in range(n_frames - 1)], cuts)

    assert field(10, ()).transition_frames() == frozenset()
    assert field(1, ()).transition_frames() == frozenset()
    assert field(10, (4,)).transition_frames() == frozenset()
    assert field(10, (3, 4, 5, 6)).transition_frames() == frozenset({3, 4, 5})
    assert field(10, (1,)).transition_frames() == frozenset({0})
    assert field(10, (9,)).transition_frames() == frozenset({9})


# --- Second read of 2026-11-22 ---------------------------------------------------------


def test_a_pan_whose_flow_fails_for_some_pairs_has_no_transition(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pair without a measured motion is a missing step of the window chain, not the identity.

    A pan of about 25 pixels per frame without a cut (synthetic seed 21, a new seed); the flow
    gives no motion for the pairs into frames 1 to 10, as on a frame with little texture. The
    identity was taken for those pairs, and the chain made up a dissolve at frames 2 to 9.
    """
    seed = 21
    start = int(np.random.default_rng([seed, 5]).integers(0, 400))
    source = SyntheticSource(RetimedScene([start + 3 * k for k in range(60)], seed=seed))
    real = motion.estimate_motion
    pairs = {"done": 0}

    def flow(prev: np.ndarray, nxt: np.ndarray, **kwargs: object) -> np.ndarray | None:
        pairs["done"] += 1
        return None if pairs["done"] <= 10 else real(prev, nxt, **kwargs)

    monkeypatch.setattr(motion, "estimate_motion", flow)
    tests = motion.measure_shots(source)
    assert tests.hard_cuts == ()
    assert tests.test_steps[:10] == (None,) * 10
    assert tests.transitions == ()


def test_a_clip_too_short_for_a_window_says_that_it_has_no_dissolve_test(caplog: pytest.LogCaptureFixture) -> None:
    source = SyntheticSource(RetimedScene(list(range(DISSOLVE_WINDOW_FRAMES)), seed=20))
    with caplog.at_level("INFO", logger=motion.logger.name):
        tests = motion.measure_shots(source)
    assert tests.window_correlations == (None,) * DISSOLVE_WINDOW_FRAMES
    assert f"not more than the window of {DISSOLVE_WINDOW_FRAMES} frames: no dissolve test" in caplog.text
