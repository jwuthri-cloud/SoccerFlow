"""Tests of motion.py: camera motion between images, motion chains and carried pitch mappings.

All images are built in the tests from seeded random numbers.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from pitchvlm import motion as motion_module
from pitchvlm.geometry import apply_homography, matrix_from_record, matrix_to_rows
from pitchvlm.motion import (
    MotionField,
    bridged_warp,
    compute_motion_field,
    estimate_motion,
    propagate_homographies,
)
from pitchvlm.schemas import Box, ClipInfo, HomographyRecord

#: Image-to-pitch mapping of a simple top-down picture: 10 px per metre, centre mark at pixel (500, 250).
PITCH_H = np.array([[0.1, 0.0, -50.0], [0.0, 0.1, -25.0], [0.0, 0.0, 1.0]])


def translation(dx: float, dy: float) -> np.ndarray:
    """Return the matrix that moves points by (dx, dy)."""
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def scaling(factor: float) -> np.ndarray:
    """Return the matrix that scales points about the origin."""
    return np.diag([factor, factor, 1.0])


def random_texture(height: int, width: int, seed: int) -> np.ndarray:
    """Return a grey ``uint8`` image with random structure of several sizes.

    The fine structure gives many corners. The coarse structure lets the
    optical flow lock on at every pyramid level, as in a real picture.
    """
    rng = np.random.default_rng(seed)
    total = np.zeros((height, width), dtype=np.float32)
    for blur_px in (1.5, 4.0, 12.0):
        layer = cv2.GaussianBlur(rng.random((height, width), dtype=np.float32), (0, 0), blur_px)
        total += (layer - layer.mean()) / layer.std()
    scaled = (total - total.min()) / (total.max() - total.min())
    return (255.0 * scaled).astype(np.uint8)


def image_grid(width: int, height: int, margin: int) -> np.ndarray:
    """Return a grid of image points that keeps ``margin`` pixels away from the border, shape (n, 2)."""
    xs = np.arange(margin, width - margin + 1, 40.0)
    ys = np.arange(margin, height - margin + 1, 40.0)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.column_stack([grid_x.ravel(), grid_y.ravel()])


def landmark_record(frame: int, matrix: np.ndarray) -> HomographyRecord:
    """Return the record of a keyframe with a landmark fit."""
    return HomographyRecord(frame=frame, matrix=matrix_to_rows(matrix), source="landmarks", n_inliers=8, reproj_px=1.5)


class ArraySource:
    """Frame source over images that are held in memory."""

    def __init__(self, images: list[np.ndarray]) -> None:
        self._images = images
        height, width = images[0].shape[:2]
        self.info = ClipInfo(clip_id="test", n_frames=len(images), fps=25.0, width=width, height=height)

    def read(self, frame: int) -> np.ndarray:
        return self._images[frame]


# --- estimate_motion ---------------------------------------------------------


def test_estimate_motion_recovers_a_small_homography() -> None:
    prev = random_texture(480, 640, seed=1)
    true_motion = np.array([[1.006, 0.003, 2.0], [-0.002, 1.004, -1.5], [6e-6, -3e-6, 1.0]])
    nxt = cv2.warpPerspective(prev, true_motion, (640, 480))  # a point p of prev is seen at true_motion @ p
    grid = image_grid(640, 480, margin=40)
    expected = apply_homography(true_motion, grid)

    estimate = estimate_motion(prev, nxt)

    assert estimate is not None
    assert estimate.shape == (3, 3)
    assert estimate.dtype == np.float64
    assert np.linalg.norm(apply_homography(estimate, grid) - expected, axis=1).max() < 0.5

    in_colour = estimate_motion(cv2.cvtColor(prev, cv2.COLOR_GRAY2BGR), cv2.cvtColor(nxt, cv2.COLOR_GRAY2BGR))
    assert in_colour is not None
    assert np.linalg.norm(apply_homography(in_colour, grid) - expected, axis=1).max() < 0.5


def test_estimate_motion_maps_the_first_image_to_the_second() -> None:
    prev = random_texture(480, 640, seed=2)
    nxt = np.roll(np.roll(prev, 5, axis=1), -3, axis=0)  # the content moves 5 px to the right and 3 px up
    grid = image_grid(640, 480, margin=40)

    estimate = estimate_motion(prev, nxt)

    assert estimate is not None
    np.testing.assert_allclose(apply_homography(estimate, grid), grid + np.array([5.0, -3.0]), atol=0.3)
    back = estimate_motion(nxt, prev)
    assert back is not None
    np.testing.assert_allclose(apply_homography(back, grid), grid + np.array([-5.0, 3.0]), atol=0.3)


def test_estimate_motion_takes_no_corners_where_the_mask_is_zero() -> None:
    prev = random_texture(480, 640, seed=3)
    split = 200
    nxt = np.empty_like(prev)
    nxt[:, :split] = np.roll(prev, 3, axis=1)[:, :split]  # the left part moves 3 px to the right
    nxt[:, split:] = np.roll(prev, -12, axis=1)[:, split:]  # the larger right part moves 12 px to the left
    mask = np.zeros(prev.shape, dtype=np.uint8)
    mask[:, : split - 30] = 255  # corners may come from the left part only

    estimate = estimate_motion(prev, nxt, mask=mask)

    # An estimate that used the whole image would follow the larger right part.
    assert estimate is not None
    left_points = np.array([[x, y] for x in (30.0, 80.0, 130.0) for y in (40.0, 240.0, 440.0)])
    np.testing.assert_allclose(apply_homography(estimate, left_points), left_points + np.array([3.0, 0.0]), atol=0.5)


def test_estimate_motion_without_enough_matches_gives_none() -> None:
    texture = random_texture(240, 320, seed=4)
    flat = np.full((240, 320), 128, dtype=np.uint8)
    assert estimate_motion(flat, flat) is None  # no corners at all
    assert estimate_motion(texture, texture, mask=np.zeros(texture.shape, dtype=np.uint8)) is None  # all masked
    assert estimate_motion(texture, texture, min_matches=100_000) is None  # more matches than corners
    assert estimate_motion(texture, texture) is not None


def test_estimate_motion_needs_enough_matches_that_agree_with_one_homography() -> None:
    prev = random_texture(480, 640, seed=3)
    nxt = np.empty_like(prev)
    # The four quarters of the picture move 6 px in four directions, so no homography fits most corners.
    for (row, column), (dx, dy) in {(0, 0): (6, 0), (0, 1): (-6, 0), (1, 0): (0, 6), (1, 1): (0, -6)}.items():
        rolled = np.roll(np.roll(prev, dx, axis=1), dy, axis=0)
        rows = slice(0, 240) if row == 0 else slice(240, 480)
        columns = slice(0, 320) if column == 0 else slice(320, 640)
        nxt[rows, columns] = rolled[rows, columns]

    # About 950 corners are followed, and about 450 of them agree with the best homography within 3 px.
    assert estimate_motion(prev, nxt, min_matches=300) is not None
    assert estimate_motion(prev, nxt, min_matches=600) is None
    # With a limit of 30 px every corner agrees with a camera that stands still.
    assert estimate_motion(prev, nxt, min_matches=600, ransac_px=30.0) is not None


def test_estimate_motion_accepts_exactly_the_smallest_number_of_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    # 40 corners that move 5 px to the right and 3 px down, and 6 corners that jump 200 px and more.
    grid_x, grid_y = np.meshgrid(np.arange(50.0, 450.0, 50.0), np.arange(50.0, 300.0, 50.0))
    start = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
    assert len(start) == 40
    wrong_start = np.array([[75.0 + 60.0 * index, 125.0] for index in range(6)], dtype=np.float32)
    jumps = np.array([[200.0, 90.0], [-210.0, 150.0], [230.0, -120.0], [260.0, 170.0], [-240.0, -100.0], [300.0, 40.0]])
    image = np.zeros((360, 480), dtype=np.uint8)

    def corners(all_start: np.ndarray, all_end: np.ndarray) -> None:
        monkeypatch.setattr(motion_module, "_track_corners", lambda prev, nxt, mask: (all_start, all_end))

    # 40 corners are followed and all agree with the motion: 40 matches are enough, 41 are not.
    corners(start, start + np.array([5.0, 3.0], dtype=np.float32))
    exact = estimate_motion(image, image, min_matches=40)
    assert exact is not None
    np.testing.assert_allclose(exact / exact[2, 2], translation(5.0, 3.0), atol=1e-3)
    assert estimate_motion(image, image, min_matches=41) is None

    # 46 corners are followed, but only 40 agree with the motion.
    all_start = np.vstack([start, wrong_start])
    all_end = np.vstack([start + np.array([5.0, 3.0]), wrong_start + jumps]).astype(np.float32)
    corners(all_start, all_end)
    assert estimate_motion(image, image, min_matches=40) is not None
    assert estimate_motion(image, image, min_matches=41) is None


def test_estimate_motion_gives_none_for_two_unrelated_pictures() -> None:
    # A cut between two cameras: nothing in the second picture is a moved part of the first one.
    first = random_texture(480, 640, seed=21)
    second = random_texture(480, 640, seed=22)

    assert estimate_motion(first, second) is None
    # About 700 corners are followed to some place, but only about 25 of them come back to their
    # start when they are followed backward. Without that check, 14 of the 700 chance matches
    # agree with one homography. With it, 7 do.
    assert estimate_motion(first, second, min_matches=11) is None


def test_estimate_motion_needs_30_matches_by_default() -> None:
    image = np.zeros((240, 320), dtype=np.uint8)
    for index in range(5):
        left = 30 + 55 * index
        image[100:130, left : left + 30] = 255  # five squares: 20 corners
    assert estimate_motion(image, image, min_matches=4) is not None
    assert estimate_motion(image, image) is None


def test_estimate_motion_rejects_wrong_input() -> None:
    texture = random_texture(240, 320, seed=5)
    with pytest.raises(ValueError):
        estimate_motion(texture.astype(np.float32), texture)
    with pytest.raises(ValueError):
        estimate_motion(texture, texture[:200])
    with pytest.raises(ValueError):
        estimate_motion(texture, texture, mask=np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        estimate_motion(texture, texture, mask=np.ones(texture.shape, dtype=bool))
    with pytest.raises(ValueError):
        estimate_motion(texture, texture, min_matches=3)
    for ransac_px in (0.0, -3.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="ransac_px"):
            estimate_motion(texture, texture, ransac_px=ransac_px)


# --- MotionField -------------------------------------------------------------

STEPS = [translation(10.0, 0.0), scaling(2.0), translation(0.0, -5.0)]


def test_motion_field_composes_the_steps_in_frame_order() -> None:
    field = MotionField(STEPS)
    assert field.n_frames == 4
    np.testing.assert_allclose(field.between(0, 1), STEPS[0])
    np.testing.assert_allclose(field.between(1, 2), STEPS[1])
    np.testing.assert_allclose(field.between(0, 2), STEPS[1] @ STEPS[0])
    np.testing.assert_allclose(field.between(0, 3), STEPS[2] @ STEPS[1] @ STEPS[0])
    # A point of frame 0 moves 10 px in x, is then doubled, and then moves -5 px in y.
    np.testing.assert_allclose(field.warp(0, 3, np.array([[1.0, 1.0]])), [[22.0, -3.0]])


def test_motion_field_same_frame_is_the_identity() -> None:
    field = MotionField(STEPS)
    for frame in range(4):
        np.testing.assert_array_equal(field.between(frame, frame), np.eye(3))
    points = np.array([[3.0, 4.0], [5.0, 6.0]])
    np.testing.assert_allclose(field.warp(2, 2, points), points)


def test_motion_field_backward_is_the_inverse_of_forward() -> None:
    field = MotionField(STEPS)
    for src, dst in [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)]:
        np.testing.assert_allclose(field.between(dst, src) @ field.between(src, dst), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(field.warp(3, 0, np.array([[22.0, -3.0]])), [[1.0, 1.0]])
    points = np.array([[100.0, 50.0], [640.0, 360.0], [0.0, 0.0]])
    there = field.warp(0, 3, points)
    assert there is not None
    np.testing.assert_allclose(field.warp(3, 0, there), points, atol=1e-9)


def test_motion_field_chains_agree_with_each_other() -> None:
    field = MotionField(STEPS)
    np.testing.assert_allclose(field.between(1, 3) @ field.between(0, 1), field.between(0, 3))
    np.testing.assert_allclose(field.between(3, 1) @ field.between(0, 3), field.between(0, 1), atol=1e-12)
    np.testing.assert_allclose(field.between(2, 0) @ field.between(3, 2), field.between(3, 0), atol=1e-12)


def test_motion_field_warp_keeps_the_point_shape() -> None:
    field = MotionField(STEPS)
    moved = field.warp(0, 2, np.zeros((7, 2)))
    assert moved is not None
    assert moved.shape == (7, 2)
    assert moved.dtype == np.float64


def test_motion_field_broken_chain_gives_none() -> None:
    field = MotionField([translation(1.0, 0.0), None, translation(2.0, 0.0)])
    assert field.n_frames == 4
    np.testing.assert_allclose(field.between(0, 1), translation(1.0, 0.0))
    np.testing.assert_allclose(field.between(2, 3), translation(2.0, 0.0))
    np.testing.assert_allclose(field.between(3, 2), translation(-2.0, 0.0))
    np.testing.assert_array_equal(field.between(1, 1), np.eye(3))
    for src, dst in [(1, 2), (2, 1), (0, 2), (2, 0), (0, 3), (3, 0), (1, 3)]:
        assert field.between(src, dst) is None
        assert field.warp(src, dst, np.array([[0.0, 0.0]])) is None


def test_motion_field_identity() -> None:
    field = MotionField.identity(5)
    assert field.n_frames == 5
    np.testing.assert_array_equal(field.between(0, 4), np.eye(3))
    np.testing.assert_array_equal(field.between(4, 1), np.eye(3))
    assert MotionField.identity(1).n_frames == 1
    with pytest.raises(ValueError):
        MotionField.identity(0)


def test_motion_field_json_round_trip() -> None:
    field = MotionField([STEPS[0], None, STEPS[2]])
    data = json.loads(json.dumps(field.to_json()))

    copy = MotionField.from_json(data)

    assert copy.n_frames == 4
    np.testing.assert_array_equal(copy.between(0, 1), STEPS[0])
    np.testing.assert_array_equal(copy.between(2, 3), STEPS[2])
    assert copy.between(1, 2) is None
    for wrong_count in (9, 2):  # more frames than the steps give, and fewer
        data["n_frames"] = wrong_count
        with pytest.raises(ValueError):
            MotionField.from_json(data)


def test_motion_field_keeps_its_own_copy_of_the_steps() -> None:
    step = translation(1.0, 0.0)
    field = MotionField([step])
    step[0, 2] = 99.0
    result = field.between(0, 1)
    assert result is not None
    result[0, 2] = 77.0
    np.testing.assert_array_equal(field.between(0, 1), translation(1.0, 0.0))


def test_motion_field_rejects_wrong_input() -> None:
    field = MotionField(STEPS)
    for src, dst in [(-1, 0), (0, 4), (4, 0), (0, -1)]:
        with pytest.raises(IndexError):
            field.between(src, dst)
    with pytest.raises(IndexError):
        field.warp(0, 4, np.zeros((1, 2)))
    with pytest.raises(ValueError):
        MotionField([np.eye(2)])
    with pytest.raises(ValueError):
        MotionField([np.zeros((3, 3))])  # this matrix has no inverse
    with pytest.raises(ValueError):
        MotionField([np.full((3, 3), np.nan)])
    with pytest.raises(ValueError, match="motion step 1"):
        # The second row of this step is twice its first row, but for a rounding error.
        MotionField([np.eye(3), np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.000000000000001], [0.0, 1.0, 1.0]])])


def test_motion_field_takes_whole_numbers_as_frames_only() -> None:
    field = MotionField(STEPS)
    np.testing.assert_allclose(field.between(np.int64(0), np.int64(1)), STEPS[0])
    # Python counts True as the number 1, so a truth value would be taken for a frame.
    for src, dst in [(True, 1), (0, True), (False, 1), (0.0, 1), (0, 1.0), ("0", 1), (None, 1)]:
        with pytest.raises(TypeError, match="frame index"):
            field.between(src, dst)
    with pytest.raises(TypeError, match="frame index"):
        field.warp(True, 1, np.zeros((1, 2)))


# --- bridged_warp ------------------------------------------------------------


def pan_records(n_frames: int, missing: tuple[int, ...] = ()) -> list[HomographyRecord]:
    """Return the pitch mappings of a camera whose picture content moves 5 px to the right per frame.

    A pitch point that is seen at pixel p in frame 0 is seen at p + (5 * frame, 0) later. The
    frames in ``missing`` have no mapping.
    """
    return [
        HomographyRecord(frame=frame)
        if frame in missing
        else landmark_record(frame, PITCH_H @ translation(-5.0 * frame, 0.0))
        for frame in range(n_frames)
    ]


def test_bridged_warp_crosses_a_missing_step_through_the_pitch_mappings() -> None:
    steps: list[np.ndarray | None] = [translation(5.0, 0.0)] * 9
    steps[4] = None  # the motion between the frames 4 and 5 is not known
    field = MotionField(steps)
    points = np.array([[100.0, 80.0], [640.0, 360.0]])
    assert field.warp(2, 7, points) is None

    warp = bridged_warp(field, pan_records(10))

    np.testing.assert_allclose(warp(2, 7, points), points + np.array([25.0, 0.0]), atol=1e-9)
    np.testing.assert_allclose(warp(7, 2, points), points - np.array([25.0, 0.0]), atol=1e-9)
    np.testing.assert_allclose(warp(4, 5, points), points + np.array([5.0, 0.0]), atol=1e-9)
    moved = warp(0, 9, np.zeros((7, 2)))
    assert moved is not None
    assert moved.shape == (7, 2)


def test_bridged_warp_uses_the_chain_wherever_it_exists() -> None:
    steps: list[np.ndarray | None] = [translation(5.0, 0.0)] * 9
    steps[4] = None
    field = MotionField(steps)
    # The pitch mappings of this camera say that the picture does not move at all. The chain knows better.
    still = [landmark_record(frame, PITCH_H) for frame in range(10)]
    points = np.array([[100.0, 80.0]])

    warp = bridged_warp(field, still)

    np.testing.assert_allclose(warp(0, 4, points), [[120.0, 80.0]])
    np.testing.assert_allclose(warp(9, 5, points), [[80.0, 80.0]])
    np.testing.assert_allclose(warp(3, 3, points), points)
    # Only across the missing step the mappings are used.
    np.testing.assert_allclose(warp(0, 9, points), points)


def test_bridged_warp_gives_none_when_a_frame_of_a_broken_chain_has_no_mapping() -> None:
    steps: list[np.ndarray | None] = [translation(5.0, 0.0)] * 9
    steps[4] = None
    field = MotionField(steps)
    points = np.array([[100.0, 80.0]])

    without_source = bridged_warp(field, pan_records(10, missing=(2,)))
    assert without_source(2, 7, points) is None
    assert without_source(7, 2, points) is None
    np.testing.assert_allclose(without_source(3, 7, points), [[120.0, 80.0]], atol=1e-9)
    np.testing.assert_allclose(without_source(2, 4, points), [[110.0, 80.0]])  # the chain needs no mapping

    # Records are needed for the frames of a bridge only: here the frames 0 to 4 have none at all.
    late_only = bridged_warp(field, pan_records(10)[5:])
    assert late_only(2, 7, points) is None
    np.testing.assert_allclose(late_only(5, 9, points), [[120.0, 80.0]])
    assert bridged_warp(field, [])(2, 7, points) is None


def test_bridged_warp_moves_a_point_above_the_grass_like_the_camera_does() -> None:
    # A camera that turns: every pixel moves with one homography, also the head of a person. The
    # two pitch mappings give that homography, although they are fitted to the grass only.
    turn = np.array([[0.98, -0.05, 30.0], [0.04, 1.01, -12.0], [2e-5, 1e-5, 1.0]])  # frame 0 pixels -> frame 1 pixels
    with_horizon = np.array([[0.1, 0.02, -50.0], [0.0, 0.25, -30.0], [0.0, 0.004, -0.2]])  # frame 0 pixels -> pitch
    records = [landmark_record(0, with_horizon), landmark_record(1, with_horizon @ np.linalg.inv(turn))]
    head_and_foot = np.array([[400.0, 20.0], [400.0, 300.0]])  # the row y = 50 is the horizon of frame 0

    warp = bridged_warp(MotionField([None]), records)

    np.testing.assert_allclose(warp(0, 1, head_and_foot), apply_homography(turn, head_and_foot), atol=1e-6)
    np.testing.assert_allclose(warp(1, 0, apply_homography(turn, head_and_foot)), head_and_foot, atol=1e-6)


def test_bridged_warp_rejects_wrong_input() -> None:
    field = MotionField.identity(4)
    with pytest.raises(ValueError, match="frame 2"):
        bridged_warp(field, [landmark_record(2, PITCH_H), landmark_record(2, PITCH_H)])
    flat = np.array([[0.1, 0.0, -50.0], [0.2, 0.0, -100.0], [0.0, 0.0, 1.0]])  # every pixel lands on one pitch line
    with pytest.raises(ValueError, match="frame 1"):
        bridged_warp(field, [landmark_record(1, flat)])
    warp = bridged_warp(field, [landmark_record(0, PITCH_H)])
    with pytest.raises(IndexError):
        warp(0, 4, np.zeros((1, 2)))


# --- compute_motion_field ----------------------------------------------------


def test_compute_motion_field_works_in_full_size_pixels() -> None:
    texture = random_texture(700, 900, seed=6)
    step = np.array([[1.004, 0.0, 6.0], [0.0, 1.004, -4.0], [0.0, 0.0, 1.0]])  # frame i pixels -> frame i + 1 pixels
    placement = translation(-120.0, -100.0)  # texture pixels -> frame 0 pixels
    frames = []
    for _ in range(4):
        frames.append(cv2.cvtColor(cv2.warpPerspective(texture, placement, (640, 480)), cv2.COLOR_GRAY2BGR))
        placement = step @ placement

    field = compute_motion_field(ArraySource(frames))

    assert field.n_frames == 4
    grid = image_grid(640, 480, margin=60)
    chains = [(0, 1, step), (0, 3, step @ step @ step), (3, 1, np.linalg.inv(step @ step))]
    for src, dst, expected in chains:
        moved = field.warp(src, dst, grid)
        assert moved is not None
        assert np.linalg.norm(moved - apply_homography(expected, grid), axis=1).max() < 1.0


def test_small_image_matrix_follows_the_pixel_centres() -> None:
    # 640 x 480 pixels become 320 x 120: the factor is 1/2 in x and 1/4 in y. The first small pixel
    # covers the full-size pixels 0 and 1 in x (centre 0.5) and the pixels 0 to 3 in y (centre 1.5).
    to_small = motion_module._to_small_matrix((640, 480), (320, 120))

    corners = np.array([[0.5, 1.5], [638.5, 477.5]])
    np.testing.assert_allclose(apply_homography(to_small, corners), [[0.0, 0.0], [319.0, 119.0]], atol=1e-9)


def test_small_image_matrix_agrees_with_the_resize_of_an_image() -> None:
    # A bright blob on a dark picture. Its centre of mass is measured before and after the resize.
    def centre_of_mass(image: np.ndarray) -> np.ndarray:
        ys, xs = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
        weights = image.astype(np.float64)
        return np.array([(xs * weights).sum(), (ys * weights).sum()]) / weights.sum()

    full = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(full, (201, 150), 37, 255, thickness=-1)
    for small_size in ((320, 240), (160, 240), (213, 97)):
        small = cv2.resize(full, small_size, interpolation=cv2.INTER_AREA)
        to_small = motion_module._to_small_matrix((640, 480), small_size)
        expected = apply_homography(to_small, centre_of_mass(full))[0]
        # Without the half-pixel term of the matrix the two centres differ by up to 0.4 px.
        np.testing.assert_allclose(centre_of_mass(small), expected, atol=0.02)


def test_people_mask_covers_the_box_grown_by_ten_percent() -> None:
    to_small = motion_module._to_small_matrix((640, 480), (320, 240))
    # The box is 100 x 80 pixels around (150, 140). Grown by 10% it spans x 95 to 205 and y 96 to 184,
    # which is x 47.25 to 102.25 and y 47.75 to 91.75 in the small image.
    mask = motion_module._people_mask((320, 240), [Box(x1=100.0, y1=100.0, x2=200.0, y2=180.0)], to_small)

    assert mask is not None
    assert mask.dtype == np.uint8
    rows, columns = np.nonzero(mask == 0)
    assert (rows.min(), rows.max(), columns.min(), columns.max()) == (47, 92, 47, 103)
    assert (mask[47:93, 47:104] == 0).all()
    assert (mask[mask != 0] == 255).all()
    assert motion_module._people_mask((320, 240), [], to_small) is None


def test_compute_motion_field_measures_on_images_of_the_given_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    shapes: list[tuple[int, ...]] = []

    def standing_still(prev: np.ndarray, nxt: np.ndarray, **kwargs: object) -> np.ndarray:
        shapes.append(prev.shape)
        return np.eye(3)

    monkeypatch.setattr(motion_module, "estimate_motion", standing_still)
    frame = cv2.cvtColor(random_texture(120, 160, seed=9), cv2.COLOR_GRAY2BGR)

    compute_motion_field(ArraySource([frame, frame]), downscale=0.25)
    compute_motion_field(ArraySource([frame, frame]))
    compute_motion_field(ArraySource([frame, frame]), downscale=1.0)

    assert shapes == [(30, 40), (60, 80), (120, 160)]


def test_compute_motion_field_can_measure_at_full_size() -> None:
    texture = random_texture(300, 400, seed=11)
    frames = [cv2.cvtColor(np.roll(texture, 3 * index, axis=1), cv2.COLOR_GRAY2BGR) for index in range(2)]

    field = compute_motion_field(ArraySource(frames), downscale=1.0)

    grid = image_grid(400, 300, margin=40)
    moved = field.warp(0, 1, grid)
    assert moved is not None
    np.testing.assert_allclose(moved, grid + np.array([3.0, 0.0]), atol=0.3)


def test_compute_motion_field_takes_no_corners_on_people() -> None:
    height, width = 480, 640
    weak = random_texture(height, width, seed=7)
    background = (98.0 + 60.0 * weak / 255.0).astype(np.uint8)  # weak texture that stands still
    background[40:440, 180:620] = 128  # a flat area around the moving block
    block = random_texture(300, 300, seed=8)  # strong texture: it stands for the people
    frames = []
    boxes = {}
    for index in range(4):
        left = 220 + 2 * index  # the block moves 2 px to the right per frame
        image = background.copy()
        image[90:390, left : left + 300] = block
        frames.append(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        boxes[index] = [Box(x1=left, y1=90, x2=left + 300, y2=390)]
    grid = image_grid(width, height, margin=40)

    # Boxes in every frame, and boxes in frames 1 and 3 only: a pair of frames uses the boxes of both frames.
    for people in (boxes, {1: boxes[1], 3: boxes[3]}):
        field = compute_motion_field(ArraySource(frames), people=people)
        moved = field.warp(0, 3, grid)
        assert moved is not None
        assert np.linalg.norm(moved - grid, axis=1).max() < 0.5  # the background does not move


def test_compute_motion_field_ignores_boxes_outside_the_image() -> None:
    frame = cv2.cvtColor(random_texture(240, 320, seed=10), cv2.COLOR_GRAY2BGR)
    outside = [
        Box(x1=-400.0, y1=-1000.0, x2=-40.0, y2=1000.0),  # left of the image, also after it is grown
        Box(x1=400.0, y1=300.0, x2=500.0, y2=400.0),  # right of and below the image
    ]

    field = compute_motion_field(ArraySource([frame, frame]), people={0: outside})

    step = field.between(0, 1)
    assert step is not None  # the boxes mask nothing, so there are enough corners
    grid = image_grid(320, 240, margin=40)
    np.testing.assert_allclose(apply_homography(step, grid), grid, atol=0.2)


def test_compute_motion_field_single_frame_and_bad_input() -> None:
    frame = cv2.cvtColor(random_texture(120, 160, seed=9), cv2.COLOR_GRAY2BGR)
    assert compute_motion_field(ArraySource([frame])).n_frames == 1
    with pytest.raises(ValueError):
        compute_motion_field(ArraySource([frame, frame]), downscale=0.0)
    with pytest.raises(ValueError):
        compute_motion_field(ArraySource([frame, frame]), downscale=1.5)
    with pytest.raises(ValueError):
        compute_motion_field(ArraySource([frame, frame[:100]]))  # the second frame has another size


def test_compute_motion_field_marks_a_pair_without_texture_as_missing() -> None:
    flat = np.full((120, 160, 3), 90, dtype=np.uint8)
    field = compute_motion_field(ArraySource([flat, flat, flat]))
    assert field.n_frames == 3
    assert field.between(0, 1) is None
    assert field.between(2, 0) is None


# --- propagate_homographies --------------------------------------------------


def test_propagate_follows_the_camera_motion() -> None:
    n_frames = 20
    motion = MotionField([translation(5.0, 0.0)] * (n_frames - 1))  # the picture content moves 5 px right per frame
    # A pitch point seen at pixel p in frame 0 is seen at p + (5 * frame, 0) in a later frame.
    true_h = [PITCH_H @ translation(-5.0 * frame, 0.0) for frame in range(n_frames)]
    key_fits = {0: landmark_record(0, true_h[0]), 12: landmark_record(12, true_h[12])}

    records = propagate_homographies(key_fits, motion, n_frames)

    assert [record.frame for record in records] == list(range(n_frames))
    assert [record.source for record in records] == (
        ["landmarks"] + ["propagated"] * 11 + ["landmarks"] + ["propagated"] * 7
    )
    assert records[0] == key_fits[0]
    assert records[12] == key_fits[12]
    assert records[5].n_inliers == 0
    assert records[5].reproj_px is None
    pixels = np.array([[100.0, 80.0], [640.0, 360.0], [900.0, 500.0]])
    for frame, record in enumerate(records):
        np.testing.assert_allclose(
            apply_homography(matrix_from_record(record), pixels), apply_homography(true_h[frame], pixels), atol=1e-9
        )


def test_propagate_takes_the_nearest_keyframe_and_prefers_the_earlier_one() -> None:
    n_frames = 20
    first = PITCH_H
    second = PITCH_H @ translation(-40.0, 0.0)  # another mapping, so that the source keyframe can be told
    key_fits = {0: landmark_record(0, first), 12: landmark_record(12, second)}

    records = propagate_homographies(key_fits, MotionField.identity(n_frames), n_frames)

    for frame in range(1, 7):  # frame 6 is as far from keyframe 0 as from keyframe 12: the earlier one wins
        np.testing.assert_allclose(matrix_from_record(records[frame]), first)
    for frame in [7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19]:
        np.testing.assert_allclose(matrix_from_record(records[frame]), second)


def test_propagate_respects_the_gap_limit() -> None:
    n_frames = 30
    key_fits = {10: landmark_record(10, PITCH_H)}

    records = propagate_homographies(key_fits, MotionField.identity(n_frames), n_frames, max_gap_frames=4)

    assert [record.source for record in records] == (
        ["none"] * 6 + ["propagated"] * 4 + ["landmarks"] + ["propagated"] * 4 + ["none"] * 15
    )
    assert records[0].matrix is None
    assert not records[0].valid
    assert records[14].valid

    only_keyframe = propagate_homographies(key_fits, MotionField.identity(n_frames), n_frames, max_gap_frames=0)
    assert [record.source for record in only_keyframe] == ["none"] * 10 + ["landmarks"] + ["none"] * 19

    default_gap = propagate_homographies({0: landmark_record(0, PITCH_H)}, MotionField.identity(100), 100)
    assert [record.source for record in default_gap] == ["landmarks"] + ["propagated"] * 75 + ["none"] * 24


def test_propagate_skips_a_keyframe_behind_a_broken_chain() -> None:
    n_frames = 16
    steps: list[np.ndarray | None] = [np.eye(3)] * (n_frames - 1)
    steps[3] = None  # the motion between frames 3 and 4 is not known
    motion = MotionField(steps)
    first = PITCH_H
    second = PITCH_H @ translation(-40.0, 0.0)

    records = propagate_homographies({2: landmark_record(2, first), 12: landmark_record(12, second)}, motion, n_frames)

    for frame in (0, 1, 3):
        np.testing.assert_allclose(matrix_from_record(records[frame]), first)
    for frame in (4, 5, 6, 7, 8, 11, 13, 15):  # keyframe 2 is nearer to frames 4 to 6, but the chain to it is broken
        np.testing.assert_allclose(matrix_from_record(records[frame]), second)

    alone = propagate_homographies({2: landmark_record(2, first)}, motion, n_frames)
    assert [record.source for record in alone] == ["propagated"] * 2 + ["landmarks"] + ["propagated"] + ["none"] * 12


def test_propagate_ignores_a_keyframe_without_a_fit() -> None:
    key_fits = {0: landmark_record(0, PITCH_H), 6: HomographyRecord(frame=6)}

    records = propagate_homographies(key_fits, MotionField.identity(8), 8)

    assert records[6].source == "propagated"
    np.testing.assert_allclose(matrix_from_record(records[6]), PITCH_H)
    assert [record.source for record in propagate_homographies({}, MotionField.identity(3), 3)] == ["none"] * 3


def test_propagate_rejects_wrong_input() -> None:
    motion = MotionField.identity(5)
    with pytest.raises(ValueError):
        propagate_homographies({}, motion, 6)  # the frame counts differ
    with pytest.raises(ValueError):
        propagate_homographies({1: landmark_record(2, PITCH_H)}, motion, 5)  # the record belongs to another frame
    with pytest.raises(ValueError):
        propagate_homographies({7: landmark_record(7, PITCH_H)}, motion, 5)  # the keyframe is outside the clip
    with pytest.raises(ValueError):
        propagate_homographies({5: landmark_record(5, PITCH_H)}, motion, 5)  # the first frame after the clip
    with pytest.raises(ValueError):
        propagate_homographies({}, motion, 5, max_gap_frames=-1)
