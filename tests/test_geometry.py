"""Tests of geometry.py: point mapping, collinearity, homography fits and plausibility checks.

The test camera is a pinhole camera built by hand. It stands on the near side,
30 m above the grass and 36 m behind the near touchline, and looks at the
right half of the pitch.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pitchvlm.geometry import (
    MIN_NEAR_SHARE,
    NEAR_LIMIT_FACTOR,
    HomographyFit,
    apply_homography,
    are_collinear,
    check_homography,
    count_near,
    fit_from_landmarks,
    fit_homography,
    invert_homography,
    landmark_pairs,
    matrix_from_record,
    matrix_to_rows,
)
from pitchvlm.pitch import LANDMARKS, landmark_xy
from pitchvlm.schemas import FrameLandmarks, HomographyRecord, LandmarkObs

WIDTH, HEIGHT = 1920, 1080
SIZE = (WIDTH, HEIGHT)


def camera_matrix(position: tuple[float, float, float], target: tuple[float, float], focal_px: float) -> np.ndarray:
    """Return the pitch-to-image matrix of a pinhole camera without roll.

    ``position`` is (pitch x, pitch y, height above the grass) in metres. The
    camera looks at the pitch point ``target``, which lands on the image centre.
    The camera axes are written out by hand, so no cross product can hide a
    mirror error.
    """
    eye = np.array(position, dtype=np.float64)
    forward = np.array([target[0], target[1], 0.0]) - eye
    forward /= np.linalg.norm(forward)
    right = np.array([-forward[1], forward[0], 0.0])  # level, and to the right of the viewing direction
    right /= np.linalg.norm(right)
    up = np.array([0.0, 0.0, 1.0]) - forward[2] * forward  # the part of "up" that is square to the view
    down = -up / np.linalg.norm(up)
    rotation = np.stack([right, down, forward])  # rows: image x, image y and depth in world coordinates
    intrinsics = np.array([[focal_px, 0.0, WIDTH / 2.0], [0.0, focal_px, HEIGHT / 2.0], [0.0, 0.0, 1.0]])
    ground = np.column_stack([rotation[:, 0], rotation[:, 1], -rotation @ eye])  # the pitch has height 0
    return intrinsics @ ground


def project(pitch_to_image: np.ndarray, pitch_xy: np.ndarray) -> np.ndarray:
    """Project pitch points into the image by hand, shape (n, 2)."""
    homogeneous = np.column_stack([pitch_xy, np.ones(len(pitch_xy))]) @ pitch_to_image.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def pitch_grid(step_x: float, step_y: float, offset_x: float = 0.0, offset_y: float = 0.0) -> np.ndarray:
    """Return a regular grid of pitch points, shape (n, 2)."""
    xs = np.arange(-52.5 + offset_x, 52.5 + 1e-9, step_x)
    ys = np.arange(-34.0 + offset_y, 34.0 + 1e-9, step_y)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.column_stack([grid_x.ravel(), grid_y.ravel()])


def visible_pairs(
    pitch_to_image: np.ndarray, pitch_xy: np.ndarray, margin_px: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return the image points and the pitch points of the pitch points that the camera sees.

    ``margin_px`` keeps a border of that width free at every image edge.
    """
    image_xy = project(pitch_to_image, pitch_xy)
    inside = (
        (image_xy[:, 0] >= margin_px)
        & (image_xy[:, 0] <= WIDTH - margin_px)
        & (image_xy[:, 1] >= margin_px)
        & (image_xy[:, 1] <= HEIGHT - margin_px)
    )
    return image_xy[inside], pitch_xy[inside]


def landmark_observations(pitch_to_image: np.ndarray) -> list[LandmarkObs]:
    """Return the true observations of the landmarks that the camera sees."""
    ids = [landmark.id for landmark in LANDMARKS]
    image_xy = project(pitch_to_image, landmark_xy(ids))
    return [
        LandmarkObs(landmark_id=landmark_id, x=float(x), y=float(y))
        for landmark_id, (x, y) in zip(ids, image_xy)
        if 0.0 <= x <= WIDTH and 0.0 <= y <= HEIGHT
    ]


def shift_pitch(dx: float, dy: float) -> np.ndarray:
    """Return the matrix that moves pitch coordinates by (dx, dy) metres."""
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


#: Pitch to image.
CAMERA = camera_matrix((8.0, 70.0, 30.0), (15.0, -2.0), 2100.0)
#: Image to pitch: the mapping that a fit must recover.
TRUE_H = np.linalg.inv(CAMERA)
#: Six pitch points that the camera sees. No three of them are on one line.
SPREAD_PITCH = np.array([[5.0, -20.0], [10.0, 5.0], [30.0, 5.0], [45.0, -25.0], [30.0, -30.0], [25.0, -15.0]])
#: The same camera place with a wider view: it sees 19 landmarks.
WIDE_CAMERA = camera_matrix((8.0, 70.0, 30.0), (15.0, -2.0), 1500.0)
#: A mapping of a wide image of 2000 x 200 pixels: 20 px are 1 m along the pitch, 10 px are 3 m across it.
#: The image centre (1000, 100) is the centre mark.
WIDE_IMAGE_H = np.array([[0.05, 0.0, -50.0], [0.0, 0.3, -30.0], [0.0, 0.0, 1.0]])


# --- the hand-made camera ----------------------------------------------------


def test_camera_follows_the_conventions() -> None:
    centre = project(CAMERA, np.array([[15.0, -2.0]]))[0]
    np.testing.assert_allclose(centre, [WIDTH / 2.0, HEIGHT / 2.0], atol=1e-6)
    far, near = project(CAMERA, np.array([[15.0, -30.0], [15.0, 20.0]]))
    assert far[1] < HEIGHT / 2.0 < near[1]  # the far side is higher in the picture
    left, right = project(CAMERA, np.array([[0.0, -2.0], [30.0, -2.0]]))
    assert left[0] < WIDTH / 2.0 < right[0]  # the right goal is to the right


# --- apply_homography --------------------------------------------------------


def test_apply_homography_maps_points() -> None:
    matrix = np.array([[2.0, 0.0, 1.0], [0.0, 3.0, -1.0], [0.0, 0.0, 1.0]])
    mapped = apply_homography(matrix, np.array([[1.0, 1.0], [0.0, 2.0]]))
    assert mapped.shape == (2, 2)
    assert mapped.dtype == np.float64
    np.testing.assert_allclose(mapped, [[3.0, 2.0], [1.0, 5.0]])


def test_apply_homography_divides_by_the_third_coordinate() -> None:
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.001, 0.0, 1.0]])
    points = np.array([[100.0, 50.0]])
    mapped = apply_homography(matrix, points)
    np.testing.assert_allclose(mapped, [[100.0 / 1.1, 50.0 / 1.1]])
    np.testing.assert_allclose(apply_homography(-5.0 * matrix, points), mapped)  # the scale of the matrix is free


def test_apply_homography_matches_the_camera() -> None:
    pitch_xy = np.array([[0.0, 0.0], [36.0, -20.16], [52.5, -34.0], [20.0, 25.0]])
    image_xy = project(CAMERA, pitch_xy)
    np.testing.assert_allclose(apply_homography(CAMERA, pitch_xy), image_xy, atol=1e-9)
    np.testing.assert_allclose(apply_homography(TRUE_H, image_xy), pitch_xy, atol=1e-8)


def test_apply_homography_shapes() -> None:
    assert apply_homography(np.eye(3), np.array([4.0, 5.0])).shape == (1, 2)
    assert apply_homography(np.eye(3), np.empty((0, 2))).shape == (0, 2)
    assert apply_homography(np.eye(3), [[1, 2], [3, 4]]).dtype == np.float64
    with pytest.raises(ValueError):
        apply_homography(np.eye(2), np.zeros((1, 2)))
    # A stack of point lists, shape (3, 2, 2), is not a list of points: it is not cut into six points.
    for not_points in (np.zeros((4, 3)), np.zeros((3, 2, 2)), np.zeros((2, 4)), np.float64(5.0)):
        with pytest.raises(ValueError, match="points must have shape"):
            apply_homography(np.eye(3), not_points)


def test_apply_homography_point_at_infinity_is_not_finite() -> None:
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, -10.0]])  # the image row y = 10 goes to infinity
    mapped = apply_homography(matrix, np.array([[5.0, 10.0], [5.0, 20.0]]))
    assert not np.isfinite(mapped[0]).any()
    np.testing.assert_allclose(mapped[1], [0.5, 2.0])


# --- small matrix helpers ----------------------------------------------------


def test_invert_homography() -> None:
    inverse = invert_homography(CAMERA)
    assert inverse is not None
    np.testing.assert_allclose(inverse @ CAMERA, np.eye(3), atol=1e-8)
    assert invert_homography(np.zeros((3, 3))) is None
    assert invert_homography(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]])) is None
    assert invert_homography(np.full((3, 3), np.nan)) is None


def test_invert_homography_refuses_a_matrix_that_is_singular_up_to_rounding() -> None:
    # The second row is twice the first row, but for a rounding error in one value. A plain
    # inversion accepts this matrix and returns values near 1e15.
    nearly_singular = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.000000000000001], [0.0, 1.0, 1.0]])
    assert np.isfinite(np.linalg.inv(nearly_singular)).all()
    assert invert_homography(nearly_singular) is None
    # The test does not depend on the scale of the matrix.
    assert invert_homography(1e-9 * nearly_singular) is None
    scaled = invert_homography(1e-9 * CAMERA)
    assert scaled is not None
    np.testing.assert_allclose(scaled @ (1e-9 * CAMERA), np.eye(3), atol=1e-8)


def test_matrix_helpers_round_trip_through_a_record() -> None:
    rows = matrix_to_rows(TRUE_H)
    assert len(rows) == 3
    assert all(len(row) == 3 and all(type(value) is float for value in row) for row in rows)
    record = HomographyRecord(frame=0, matrix=rows, source="landmarks")
    np.testing.assert_array_equal(matrix_from_record(record), TRUE_H)
    with pytest.raises(ValueError):
        matrix_from_record(HomographyRecord(frame=0))
    with pytest.raises(ValueError):
        matrix_to_rows(np.eye(4))


# --- are_collinear -----------------------------------------------------------


def test_are_collinear_on_a_line() -> None:
    line = np.array([[0.0, 0.0], [10.0, 5.0], [20.0, 10.0], [-4.0, -2.0]])
    assert are_collinear(line, 1e-9)
    assert are_collinear(np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.1]]), 0.5)  # 10 cm off the line


def test_are_collinear_off_a_line() -> None:
    assert not are_collinear(np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]), 0.5)
    assert not are_collinear(np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 10.0]]), 0.5)  # 10 m off the line


def test_are_collinear_measures_the_distance_to_the_best_line() -> None:
    # Four points 1 m beside the line x = 0, placed so that this line is their best line.
    # Their root mean square distance to it is 1 m.
    points = np.array([[1.0, -30.0], [-1.0, -10.0], [-1.0, 10.0], [1.0, 30.0]])
    assert are_collinear(points, 1.01)
    assert not are_collinear(points, 0.99)


def test_are_collinear_with_tolerance_zero_needs_an_exact_line() -> None:
    # The points lie exactly on a sloped line. The rounding error of the measure is about 1e-15.
    line = np.array([[0.0, 0.0], [10.0, 5.0], [20.0, 10.0], [-4.0, -2.0]])
    assert are_collinear(line, 0.0)
    assert are_collinear(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), 0.0)
    assert are_collinear(np.array([[7.0, 7.0], [7.0, 7.0], [7.0, 7.0]]), 0.0)  # three points at one place
    one_micrometre_off = line + np.array([[0.0, 0.0], [0.0, 1e-6], [0.0, 0.0], [0.0, 0.0]])
    assert not are_collinear(one_micrometre_off, 0.0)


@pytest.mark.parametrize("tolerance", [-0.1, float("nan"), float("inf")])
def test_are_collinear_refuses_a_wrong_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        are_collinear(np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]), tolerance)


def test_are_collinear_with_few_points() -> None:
    assert are_collinear(np.empty((0, 2)), 0.5)
    assert are_collinear(np.array([[3.0, 4.0]]), 0.5)
    assert are_collinear(np.array([[3.0, 4.0], [50.0, -20.0]]), 0.0)


def test_are_collinear_on_pitch_landmarks() -> None:
    halfway_line = landmark_xy([16, 17, 18, 19, 20])
    assert are_collinear(halfway_line, 0.5)
    assert not are_collinear(np.vstack([halfway_line, landmark_xy([25])]), 0.5)  # plus the right penalty spot


# --- fit_homography ----------------------------------------------------------


def test_fit_homography_recovers_pitch_positions_without_noise() -> None:
    image_xy, pitch_xy = visible_pairs(CAMERA, pitch_grid(2.5, 2.0))
    assert len(image_xy) > 500

    fit = fit_homography(image_xy, pitch_xy)

    assert isinstance(fit, HomographyFit)
    assert fit.H.shape == (3, 3)
    assert fit.H.dtype == np.float64
    assert fit.inliers.shape == (len(image_xy),)
    assert fit.inliers.dtype == np.bool_
    assert fit.inliers.all()
    assert fit.reproj_px < 0.05
    test_image, test_pitch = visible_pairs(CAMERA, pitch_grid(5.0, 4.0, 2.5, 2.0), margin_px=40.0)
    assert len(test_pitch) > 100
    errors_m = np.linalg.norm(apply_homography(fit.H, test_image) - test_pitch, axis=1)
    assert errors_m.max() < 0.05


def test_fit_homography_with_pixel_noise_and_outliers() -> None:
    rng = np.random.default_rng(7)
    image_xy, pitch_xy = visible_pairs(CAMERA, pitch_grid(2.5, 2.0))
    count = len(image_xy)
    noisy = image_xy + rng.normal(0.0, 2.0, size=image_xy.shape)  # 2 px noise on every point
    outliers = rng.choice(count, size=count // 5, replace=False)  # 20% of the pairs are wrong
    angle = rng.uniform(0.0, 2.0 * np.pi, size=len(outliers))
    radius = rng.uniform(100.0, 400.0, size=len(outliers))
    noisy[outliers] += np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    clean = np.ones(count, dtype=bool)
    clean[outliers] = False

    fit = fit_homography(noisy, pitch_xy)

    assert fit is not None
    assert not fit.inliers[outliers].any()
    assert fit.inliers[clean].mean() > 0.5
    assert 1.0 < fit.reproj_px < 4.0
    test_image, test_pitch = visible_pairs(CAMERA, pitch_grid(5.0, 4.0, 2.5, 2.0), margin_px=40.0)
    errors_m = np.linalg.norm(apply_homography(fit.H, test_image) - test_pitch, axis=1)
    # RANSAC keeps the pairs that agree with one random sample, so the fit changes a little with the
    # sample. The far touchline is the weakest place: one pixel is about 0.2 m of depth there.
    assert np.percentile(errors_m, 95) < 0.5
    assert errors_m.max() < 1.0


def test_fit_homography_follows_the_sign_conventions() -> None:
    image_xy, pitch_xy = visible_pairs(CAMERA, pitch_grid(5.0, 4.0))
    fit = fit_homography(image_xy, pitch_xy)
    assert fit is not None
    centre = np.array([WIDTH / 2.0, HEIGHT / 2.0])
    steps = np.array([[0.0, 0.0], [0.0, -300.0], [0.0, 300.0], [300.0, 0.0]])
    middle, higher, lower, to_the_right = apply_homography(fit.H, centre + steps)
    np.testing.assert_allclose(middle, [15.0, -2.0], atol=0.05)
    assert higher[1] < middle[1] < lower[1]  # up in the picture is the far side, so pitch y falls
    assert higher[1] < 0.0  # a far-side point has a negative y
    assert lower[1] > 0.0  # a near-side point has a positive y
    assert to_the_right[0] > middle[0]  # right in the picture is towards the right goal


def test_fit_homography_has_unit_scale() -> None:
    image_xy, pitch_xy = visible_pairs(CAMERA, pitch_grid(5.0, 4.0))
    fit = fit_homography(image_xy, pitch_xy)
    assert fit is not None
    assert np.linalg.norm(fit.H) == pytest.approx(1.0)


def test_fit_homography_collinear_points_give_none() -> None:
    on_the_halfway_line = np.array([[0.0, y] for y in (-34.0, -20.0, -9.15, 0.0, 9.15, 20.0)])
    assert fit_homography(project(CAMERA, on_the_halfway_line), on_the_halfway_line) is None
    four_points = on_the_halfway_line[:4]
    assert fit_homography(project(CAMERA, four_points), four_points) is None


def test_fit_homography_nearly_collinear_points_give_none() -> None:
    pitch_xy = np.array([[0.2, -34.0], [-0.2, -20.0], [0.1, -9.15], [-0.1, 0.0], [0.2, 9.15], [-0.2, 20.0]])
    assert are_collinear(pitch_xy, 0.5)
    assert fit_homography(project(CAMERA, pitch_xy), pitch_xy) is None


def test_fit_homography_all_points_but_one_on_a_line_give_none() -> None:
    # Marks on the halfway line plus one other mark: many mappings fit them perfectly, so there is no fit.
    pitch_xy = np.array([[0.0, -34.0], [0.0, -9.15], [0.0, 0.0], [0.0, 9.15], [0.0, 20.0], [36.0, 7.3]])
    assert not are_collinear(pitch_xy, 0.5)
    assert fit_homography(project(CAMERA, pitch_xy), pitch_xy) is None
    three_on_a_line = pitch_xy[[0, 2, 4, 5]]
    assert fit_homography(project(CAMERA, three_on_a_line), three_on_a_line) is None


def test_fit_homography_marks_on_the_goal_line_and_the_goal_area_line_give_a_fit() -> None:
    # Four marks on the right goal line and the two front corners of the goal area, 5.5 m in front
    # of it. Their root mean square distance to one line is 2.6 m: close to a line for a pitch,
    # but far from a line for a homography.
    ids = [29, 30, 33, 34, 26, 27]
    pitch_xy = landmark_xy(ids)
    assert not are_collinear(pitch_xy, 2.0)
    assert are_collinear(pitch_xy, 3.0)

    fit = fit_homography(project(CAMERA, pitch_xy), pitch_xy)

    assert fit is not None
    assert fit.inliers.all()
    test_pitch = np.array([[40.0, 0.0], [45.0, -15.0]])
    np.testing.assert_allclose(apply_homography(fit.H, project(CAMERA, test_pitch)), test_pitch, atol=0.05)


def test_fit_homography_two_pairs_at_one_image_point_give_none() -> None:
    # The model gave one image point for two marks. Four such pairs have an exact solution, a
    # matrix that sends the whole pitch onto one image line, and the solver marks all four pairs.
    image_xy = np.array([[100.0, 100.0], [100.0, 100.0], [900.0, 500.0], [300.0, 600.0]])
    pitch_xy = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    assert fit_homography(image_xy, pitch_xy) is None

    # The same with image points that a camera can give for the other three marks.
    seen = project(CAMERA, SPREAD_PITCH[:4])
    seen[1] = seen[0]
    assert fit_homography(seen, SPREAD_PITCH[:4]) is None


def test_fit_homography_three_image_points_on_a_line_give_none() -> None:
    # No three of the pitch points are on one line, so no camera sees three of them on one line.
    image_xy = np.array([[100.0, 100.0], [200.0, 200.0], [300.0, 300.0], [300.0, 600.0]])
    pitch_xy = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    assert fit_homography(image_xy, pitch_xy) is None

    # Half a pixel beside the line is still on the line for a fit.
    nearly = image_xy + np.array([[0.0, 0.0], [0.5, -0.5], [0.0, 0.0], [0.0, 0.0]])
    assert fit_homography(nearly, pitch_xy) is None

    # Six pairs: five image points on one line and one point beside it.
    six_pitch = SPREAD_PITCH
    six_image = np.array(
        [[100.0, 100.0], [200.0, 200.0], [300.0, 300.0], [400.0, 400.0], [500.0, 500.0], [300.0, 600.0]]
    )
    assert fit_homography(six_image, six_pitch, ransac_px=1000.0) is None


def test_fit_homography_inliers_are_the_pairs_within_the_limit() -> None:
    image_xy, pitch_xy = visible_pairs(CAMERA, pitch_grid(5.0, 4.0))
    assert len(image_xy) > 100
    image_xy = image_xy.copy()
    image_xy[10] += [3.0, 4.0]  # 5 px beside its true place
    image_xy[20] += [12.0, 16.0]  # 20 px beside its true place

    for ransac_px, expected in ((8.0, (True, False)), (3.0, (False, False)), (30.0, (True, True))):
        fit = fit_homography(image_xy, pitch_xy, ransac_px=ransac_px)
        assert fit is not None
        assert (bool(fit.inliers[10]), bool(fit.inliers[20])) == expected
        errors_px = np.linalg.norm(project(np.linalg.inv(fit.H), pitch_xy) - image_xy, axis=1)
        assert fit.inliers.tolist() == (errors_px <= ransac_px).tolist()
        assert int(fit.inliers.sum()) == len(image_xy) - expected.count(False)


def test_fit_homography_reports_the_median_error_of_the_inliers() -> None:
    image_xy, pitch_xy = visible_pairs(CAMERA, pitch_grid(10.0, 8.0))
    assert len(image_xy) == 58
    image_xy = image_xy.copy()
    image_xy[::8] += [0.0, 7.0]  # 8 of the 58 image points are 7 px too low

    fit = fit_homography(image_xy, pitch_xy)

    assert fit is not None
    assert fit.inliers.all()
    errors_px = np.linalg.norm(project(np.linalg.inv(fit.H), pitch_xy) - image_xy, axis=1)
    assert fit.reproj_px == pytest.approx(float(np.median(errors_px)))
    # The eight points pull the mean up to about 1.6 px. The median stays at about 1.0 px.
    assert float(np.mean(errors_px)) > 1.4 * fit.reproj_px


def test_fit_homography_needs_a_share_of_many_pairs() -> None:
    right_image = project(CAMERA, SPREAD_PITCH)
    # Nine wrong pairs: pitch points that the camera sees, with image points that are 300 px and more away.
    wrong_pitch = np.array([[x, y] for x in (0.0, 15.0, 40.0) for y in (-28.0, -12.0, 2.0)])
    angles = 2.4 * np.arange(9)
    shifts = (300.0 + 40.0 * np.arange(9))[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])
    wrong_image = project(CAMERA, wrong_pitch) + shifts

    # 5 right pairs of 14: the fifth pair confirms the sample of the other four, but 5 is less than 40% of 14.
    image_xy = np.vstack([right_image[:5], wrong_image])
    pitch_xy = np.vstack([SPREAD_PITCH[:5], wrong_pitch])
    assert fit_homography(image_xy, pitch_xy) is None
    with_low_share = fit_homography(image_xy, pitch_xy, min_near_share=0.3)
    assert with_low_share is not None
    assert with_low_share.inliers.tolist() == [True] * 5 + [False] * 9
    # With the share 0 the rule is off.
    assert fit_homography(image_xy, pitch_xy, min_near_share=0.0) is not None

    # 6 right pairs of 15 are exactly 40%.
    image_xy = np.vstack([right_image, wrong_image])
    pitch_xy = np.vstack([SPREAD_PITCH, wrong_pitch])
    fit = fit_homography(image_xy, pitch_xy)
    assert fit is not None
    assert fit.inliers.tolist() == [True] * 6 + [False] * 9
    assert fit_homography(image_xy, pitch_xy, min_near_share=0.45) is None


def test_fit_homography_counts_the_share_with_the_wider_limit() -> None:
    assert (MIN_NEAR_SHARE, NEAR_LIMIT_FACTOR) == (0.4, 3.0)
    right_image = project(CAMERA, SPREAD_PITCH)
    # Four more right pairs whose image points are 20 px beside their place: more than the RANSAC
    # threshold of 8 px, less than the near limit of 24 px. Each one is moved in another direction.
    noisy_pitch = np.array([[15.0, -10.0], [35.0, -18.0], [20.0, -25.0], [40.0, -5.0]])
    directions = np.array([[1.0, 0.0], [-0.6, 0.8], [0.0, -1.0], [-0.8, -0.6]])
    # Ten wrong pairs: their image points are 300 px and more away.
    wrong_pitch = np.array([[x, y] for x in (0.0, 12.0, 28.0, 44.0) for y in (-28.0, -12.0, 2.0)])[:10]
    angles = 2.4 * np.arange(10)
    shifts = (300.0 + 40.0 * np.arange(10))[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])
    wrong_image = project(CAMERA, wrong_pitch) + shifts
    pitch_xy = np.vstack([SPREAD_PITCH, noisy_pitch, wrong_pitch])

    # 6 inliers are less than 40% of the 20 pairs. With the 4 noisy pairs, 10 pairs are near the fit.
    image_xy = np.vstack([right_image, project(CAMERA, noisy_pitch) + 20.0 * directions, wrong_image])
    fit = fit_homography(image_xy, pitch_xy)
    assert fit is not None
    assert fit.inliers.tolist() == [True] * 6 + [False] * 14
    np.testing.assert_allclose(apply_homography(fit.H, right_image), SPREAD_PITCH, atol=0.05)
    assert fit_homography(image_xy, pitch_xy, min_near_share=0.55) is None

    # The same with the four noisy points 40 px beside their place: beyond the near limit.
    image_xy = np.vstack([right_image, project(CAMERA, noisy_pitch) + 40.0 * directions, wrong_image])
    assert fit_homography(image_xy, pitch_xy) is None
    assert fit_homography(image_xy, pitch_xy, min_near_share=0.3) is not None


@pytest.mark.parametrize(("noise_px", "min_valid"), [(10.0, 38), (15.0, 30), (20.0, 18)])
def test_fit_from_landmarks_keeps_true_ids_with_noise_above_the_ransac_threshold(
    noise_px: float, min_valid: int
) -> None:
    # 19 landmarks with their true ids. Every coordinate gets Gaussian noise, 40 times with
    # another seed. With 15 px of noise, fewer than 40% of the points lie within the RANSAC
    # threshold of 8 px of a fit, but most lie within the near limit of 24 px.
    observations = landmark_observations(WIDE_CAMERA)
    assert len(observations) == 19
    n_valid = 0
    for seed in range(40):
        rng = np.random.default_rng([seed, int(noise_px)])
        noisy = tuple(
            LandmarkObs(
                landmark_id=item.landmark_id, x=item.x + rng.normal(0.0, noise_px), y=item.y + rng.normal(0.0, noise_px)
            )
            for item in observations
        )
        n_valid += fit_from_landmarks(FrameLandmarks(frame=0, landmarks=noisy)).valid

    assert n_valid >= min_valid


def test_fit_homography_counts_a_pair_at_exactly_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    # A solver that returns an exact matrix: 2 px per metre. So the errors below are exact numbers.
    monkeypatch.setattr(cv2, "findHomography", lambda *args: (np.diag([2.0, 2.0, 1.0]), None))
    pitch_xy = np.array(
        [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0], [60.0, 10.0], [50.0, 20.0], [30.0, 40.0]]
    )
    image_xy = 2.0 * pitch_xy
    image_xy[5] += [8.0, 0.0]  # at exactly the RANSAC threshold
    image_xy[6] += [0.0, 24.0]  # at exactly the near limit

    fit = fit_homography(image_xy, pitch_xy, ransac_px=8.0, min_near_share=1.0)

    assert fit is not None
    assert fit.inliers.tolist() == [True] * 6 + [False]
    # A little further, and the pair is no inlier any more.
    image_xy[5] += [1e-9, 0.0]
    further = fit_homography(image_xy, pitch_xy, ransac_px=8.0, min_near_share=1.0)
    assert further is not None
    assert further.inliers.tolist() == [True] * 5 + [False] * 2
    # A little further, and the last pair is not near the fit any more: 6 of 7 pairs are.
    image_xy[6] += [0.0, 1e-9]
    assert fit_homography(image_xy, pitch_xy, ransac_px=8.0, min_near_share=1.0) is None
    assert fit_homography(image_xy, pitch_xy, ransac_px=8.0, min_near_share=6 / 7) is not None


def test_count_near_counts_the_pairs_within_three_times_the_threshold() -> None:
    to_pitch = np.diag([0.5, 0.5, 1.0])  # 2 px per metre
    pitch_xy = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]])
    image_xy = 2.0 * pitch_xy + np.array([[0.0, 0.0], [24.0, 0.0], [0.0, 24.5], [-30.0, 0.0]])

    assert count_near(to_pitch, image_xy, pitch_xy, ransac_px=8.0) == 2
    assert count_near(to_pitch, image_xy, pitch_xy, ransac_px=10.0) == 4
    assert count_near(to_pitch, np.zeros((0, 2)), np.zeros((0, 2)), ransac_px=8.0) == 0
    # A mapping without an inverse places no point in the image.
    assert count_near(np.zeros((3, 3)), image_xy, pitch_xy, ransac_px=8.0) == 0
    with pytest.raises(ValueError, match="pair"):
        count_near(to_pitch, image_xy, pitch_xy[:3], ransac_px=8.0)
    for ransac_px in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="ransac_px"):
            count_near(to_pitch, image_xy, pitch_xy, ransac_px=ransac_px)


def test_fit_homography_points_on_two_lines_give_a_fit() -> None:
    pitch_xy = np.array([[0.0, -34.0], [0.0, -9.15], [0.0, 9.15], [0.0, 20.0], [36.0, -20.16], [36.0, 7.3]])
    fit = fit_homography(project(CAMERA, pitch_xy), pitch_xy)
    assert fit is not None
    assert fit.inliers.all()
    test_pitch = np.array([[20.0, 0.0], [10.0, -15.0]])
    np.testing.assert_allclose(apply_homography(fit.H, project(CAMERA, test_pitch)), test_pitch, atol=0.05)


def test_fit_homography_needs_four_pairs() -> None:
    pitch_xy = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    assert fit_homography(project(CAMERA, pitch_xy), pitch_xy) is None
    assert fit_homography(np.empty((0, 2)), np.empty((0, 2))) is None


def test_fit_homography_accepts_exactly_four_pairs() -> None:
    pitch_xy = SPREAD_PITCH[:4]
    image_xy = project(CAMERA, pitch_xy)

    fit = fit_homography(image_xy, pitch_xy)

    # Four pairs fit one homography exactly. There is no other pair that can speak against the fit.
    assert fit is not None
    assert fit.inliers.tolist() == [True, True, True, True]
    assert fit.reproj_px < 0.05
    np.testing.assert_allclose(apply_homography(fit.H, project(CAMERA, SPREAD_PITCH[4:])), SPREAD_PITCH[4:], atol=0.05)
    assert fit_homography(image_xy, pitch_xy, min_inliers=5) is None


def test_fit_homography_respects_min_inliers() -> None:
    pitch_xy = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 15.0], [0.0, 15.0], [10.0, -10.0], [30.0, 5.0]])
    image_xy = project(CAMERA, pitch_xy)
    fit = fit_homography(image_xy, pitch_xy)
    assert fit is not None
    assert int(fit.inliers.sum()) == 6
    assert fit_homography(image_xy, pitch_xy, min_inliers=7) is None


def test_fit_homography_needs_a_pair_that_confirms_the_sample() -> None:
    image_xy = project(CAMERA, SPREAD_PITCH)
    image_xy[5, 1] -= 300.0  # the last pair is wrong: its image point is 300 pixels too high
    four_right_one_wrong = [0, 1, 2, 3, 5]

    # Each four of these five pairs fit one homography exactly, and the fifth pair is hundreds of
    # pixels away from it. So every fit has four inliers, its own sample, and no pair confirms it.
    assert fit_homography(image_xy[four_right_one_wrong], SPREAD_PITCH[four_right_one_wrong]) is None

    # The five right pairs alone: each pair confirms the fit of the other four.
    all_right = fit_homography(image_xy[:5], SPREAD_PITCH[:5])
    assert all_right is not None
    assert all_right.inliers.tolist() == [True, True, True, True, True]

    # The five right pairs and the wrong pair: the fifth right pair confirms the fit, and the wrong
    # pair is left out.
    fit = fit_homography(image_xy, SPREAD_PITCH)
    assert fit is not None
    assert fit.inliers.tolist() == [True, True, True, True, True, False]
    assert fit.reproj_px < 0.05
    test_pitch = np.array([[20.0, 0.0], [15.0, -10.0]])
    np.testing.assert_allclose(apply_homography(fit.H, project(CAMERA, test_pitch)), test_pitch, atol=0.05)


def test_fit_homography_rejects_wrong_input() -> None:
    with pytest.raises(ValueError):
        fit_homography(np.zeros((5, 2)), np.zeros((4, 2)))
    with pytest.raises(ValueError):
        fit_homography(np.zeros((5, 2)), np.zeros((5, 2)), min_inliers=3)
    with pytest.raises(ValueError):
        fit_homography(np.zeros((5, 2)), np.zeros((5, 2)), ransac_px=0.0)
    with pytest.raises(ValueError):
        fit_homography(np.full((5, 2), np.nan), np.zeros((5, 2)))
    for ransac_px in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="ransac_px"):
            fit_homography(np.zeros((5, 2)), np.zeros((5, 2)), ransac_px=ransac_px)
    for share in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="min_near_share"):
            fit_homography(np.zeros((5, 2)), np.zeros((5, 2)), min_near_share=share)


# --- fit_from_landmarks ------------------------------------------------------


def test_fit_from_landmarks_gives_a_landmark_record() -> None:
    observations = landmark_observations(CAMERA)
    assert len(observations) == 14  # the camera sees the halfway line and most of the right half

    record = fit_from_landmarks(FrameLandmarks(frame=12, view="right", landmarks=tuple(observations)))

    assert record.frame == 12
    assert record.source == "landmarks"
    assert record.valid
    assert record.n_inliers == 14
    assert record.reproj_px is not None
    assert record.reproj_px < 0.05
    matrix = matrix_from_record(record)
    image_xy = np.array([[item.x, item.y] for item in observations])
    pitch_xy = landmark_xy([item.landmark_id for item in observations])
    np.testing.assert_allclose(apply_homography(matrix, image_xy), pitch_xy, atol=0.05)
    np.testing.assert_allclose(apply_homography(matrix, image_xy), apply_homography(TRUE_H, image_xy), atol=0.05)


def test_fit_from_landmarks_far_side_has_negative_y() -> None:
    observations = landmark_observations(CAMERA)
    record = fit_from_landmarks(FrameLandmarks(frame=0, landmarks=tuple(observations)))
    by_id = {item.landmark_id: item for item in observations}
    far_end, centre_spot = by_id[16], by_id[18]  # halfway line: far end and centre spot
    assert far_end.y < centre_spot.y  # the far end is higher in the picture
    far_xy, centre_xy = apply_homography(
        matrix_from_record(record), np.array([[far_end.x, far_end.y], [centre_spot.x, centre_spot.y]])
    )
    np.testing.assert_allclose(far_xy, [0.0, -34.0], atol=0.05)
    np.testing.assert_allclose(centre_xy, [0.0, 0.0], atol=0.05)


def test_fit_from_landmarks_drops_unknown_and_repeated_ids() -> None:
    observations = landmark_observations(CAMERA)
    repeated_id = observations[0].landmark_id
    extra = (
        LandmarkObs(landmark_id=99, x=10.0, y=10.0),  # no such landmark
        LandmarkObs(landmark_id=0, x=20.0, y=900.0),  # ids start at 1
        LandmarkObs(landmark_id=repeated_id, x=1500.0, y=900.0),  # a second answer for the same landmark
    )

    record = fit_from_landmarks(FrameLandmarks(frame=0, landmarks=tuple(observations) + extra))

    assert record.source == "landmarks"
    assert record.n_inliers == len(observations) - 1  # both answers for the repeated id are dropped
    assert record.reproj_px is not None
    assert record.reproj_px < 0.05
    test_image = np.array([[400.0, 400.0], [960.0, 540.0], [1700.0, 700.0]])
    np.testing.assert_allclose(
        apply_homography(matrix_from_record(record), test_image), apply_homography(TRUE_H, test_image), atol=0.05
    )


def test_landmark_pairs_holds_the_landmarks_that_a_fit_can_use() -> None:
    landmarks = (
        LandmarkObs(landmark_id=16, x=432.0, y=372.5),
        LandmarkObs(landmark_id=99, x=10.0, y=10.0),  # no such landmark
        LandmarkObs(landmark_id=24, x=1918.0, y=913.5),
        LandmarkObs(landmark_id=19, x=500.0, y=600.0),
        LandmarkObs(landmark_id=19, x=520.0, y=610.0),  # a second answer for landmark 19: both are dropped
    )

    image_xy, pitch_xy = landmark_pairs(FrameLandmarks(frame=4, landmarks=landmarks))

    assert image_xy.tolist() == [[432.0, 372.5], [1918.0, 913.5]]
    assert pitch_xy.tolist() == landmark_xy([16, 24]).tolist()
    empty_image, empty_pitch = landmark_pairs(FrameLandmarks(frame=4))
    assert (empty_image.shape, empty_pitch.shape) == ((0, 2), (0, 2))


def test_fit_from_landmarks_needs_a_landmark_that_confirms_the_fit() -> None:
    by_id = {item.landmark_id: item for item in landmark_observations(CAMERA)}
    right = tuple(by_id[landmark_id] for landmark_id in (16, 19, 24, 25))
    # A mix-up of left and right: the point of landmark 21 (right penalty area, far front corner)
    # with the id of landmark 12 (left penalty area, far front corner).
    wrong = LandmarkObs(landmark_id=12, x=by_id[21].x, y=by_id[21].y)

    # Four right landmarks and the wrong one: each fit through four of them puts the fifth one
    # hundreds of pixels away. So no landmark confirms a fit, and the frame gets no mapping.
    unconfirmed = fit_from_landmarks(FrameLandmarks(frame=7, landmarks=right + (wrong,)))
    assert unconfirmed.frame == 7
    assert unconfirmed.source == "none"
    assert unconfirmed.matrix is None
    assert unconfirmed.n_inliers == 0

    # One more right landmark confirms the fit of the right ones. The wrong landmark is left out.
    confirmed = fit_from_landmarks(FrameLandmarks(frame=7, landmarks=right + (by_id[29], wrong)))
    assert confirmed.source == "landmarks"
    assert confirmed.n_inliers == 5
    assert confirmed.reproj_px is not None
    assert confirmed.reproj_px < 0.05
    test_image = np.array([[400.0, 400.0], [960.0, 540.0], [1700.0, 700.0]])
    np.testing.assert_allclose(
        apply_homography(matrix_from_record(confirmed), test_image), apply_homography(TRUE_H, test_image), atol=0.05
    )


def test_fit_from_landmarks_with_wrong_ids_gives_source_none() -> None:
    observations = landmark_observations(WIDE_CAMERA)
    assert len(observations) == 19
    assert fit_from_landmarks(FrameLandmarks(frame=5, landmarks=tuple(observations))).n_inliers == 19

    # Every point is right, but every id is the id of the next landmark that the camera sees.
    # Among 19 wrong pairs, RANSAC finds a sample that one or two more pairs agree with by chance.
    ids = [item.landmark_id for item in observations]
    shifted = tuple(
        LandmarkObs(landmark_id=ids[(index + 1) % len(ids)], x=item.x, y=item.y)
        for index, item in enumerate(observations)
    )

    record = fit_from_landmarks(FrameLandmarks(frame=5, landmarks=shifted))

    assert record.source == "none"
    assert record.matrix is None


def test_fit_from_landmarks_with_two_landmarks_at_one_image_point_gives_source_none() -> None:
    # Four landmarks, and the model gave the same image point for two of them.
    landmarks = (
        LandmarkObs(landmark_id=16, x=432.0, y=372.6),
        LandmarkObs(landmark_id=19, x=432.0, y=372.6),
        LandmarkObs(landmark_id=24, x=1918.08, y=913.68),
        LandmarkObs(landmark_id=21, x=1392.0, y=428.76),
    )

    record = fit_from_landmarks(FrameLandmarks(frame=3, landmarks=landmarks))

    assert record == HomographyRecord(frame=3)


def test_fit_from_landmarks_without_a_fit_gives_source_none() -> None:
    observations = landmark_observations(CAMERA)

    too_few = fit_from_landmarks(FrameLandmarks(frame=3, landmarks=tuple(observations[:3])))
    assert too_few.frame == 3
    assert too_few.source == "none"
    assert too_few.matrix is None
    assert not too_few.valid
    assert too_few.n_inliers == 0
    assert too_few.reproj_px is None

    halfway_line = tuple(item for item in observations if item.landmark_id in (16, 17, 18, 19))
    assert len(halfway_line) == 4
    assert fit_from_landmarks(FrameLandmarks(frame=3, landmarks=halfway_line)).source == "none"
    penalty_spot = tuple(item for item in observations if item.landmark_id == 25)
    assert len(penalty_spot) == 1
    assert fit_from_landmarks(FrameLandmarks(frame=3, landmarks=halfway_line + penalty_spot)).source == "none"
    assert fit_from_landmarks(FrameLandmarks(frame=3)).source == "none"


# --- check_homography --------------------------------------------------------


def test_check_homography_accepts_the_true_mapping() -> None:
    feet, _ = visible_pairs(CAMERA, pitch_grid(10.0, 8.0))
    assert len(feet) > 20
    assert check_homography(TRUE_H, SIZE) == []
    assert check_homography(TRUE_H, SIZE, feet) == []
    assert check_homography(-3.0 * TRUE_H, SIZE, feet) == []  # the scale and the sign of the matrix are free


def test_check_homography_flags_a_mirrored_mapping() -> None:
    wrong_pitch_x = np.diag([-1.0, 1.0, 1.0]) @ TRUE_H  # the left and the right goal are swapped
    problems = check_homography(wrong_pitch_x, SIZE)
    assert len(problems) == 1
    assert "mirrored" in problems[0]

    flip_image = np.array([[-1.0, 0.0, float(WIDTH)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # x_px -> WIDTH - x_px
    problems = check_homography(TRUE_H @ flip_image, SIZE)
    assert len(problems) == 1
    assert "mirrored" in problems[0]


def test_check_homography_flags_a_flipped_mapping() -> None:
    wrong_pitch_y = np.diag([1.0, -1.0, 1.0]) @ TRUE_H  # the near and the far side are swapped
    problems = check_homography(wrong_pitch_y, SIZE)
    assert len(problems) == 1
    assert "flipped" in problems[0]

    turned_around = np.diag([-1.0, -1.0, 1.0]) @ TRUE_H  # both axes have the wrong sign
    assert len(check_homography(turned_around, SIZE)) == 2


def test_check_homography_flags_a_centre_far_from_the_pitch() -> None:
    assert check_homography(shift_pitch(65.0, 0.0) @ TRUE_H, SIZE) == []  # centre at x = 80 m: inside the 30 m margin
    problems = check_homography(shift_pitch(200.0, 0.0) @ TRUE_H, SIZE)  # centre at x = 215 m
    assert len(problems) == 1
    assert "centre" in problems[0]


def test_check_homography_counts_the_foot_points_inside_the_pitch() -> None:
    on_the_pitch = project(CAMERA, np.array([[float(x), -10.0] for x in range(0, 38, 2)]))  # 19 points
    behind_the_line = project(CAMERA, np.array([[20.0, -36.0]]))  # 2 m behind the far touchline
    far_away = project(CAMERA, np.array([[20.0, -50.0], [30.0, -50.0], [40.0, -50.0]]))  # 16 m behind it

    assert check_homography(TRUE_H, SIZE, np.vstack([on_the_pitch, behind_the_line])) == []  # inside the 3 m margin
    assert check_homography(TRUE_H, SIZE, np.vstack([on_the_pitch, far_away[:1]])) == []  # 19 of 20 are inside
    problems = check_homography(TRUE_H, SIZE, np.vstack([on_the_pitch, far_away]))  # 19 of 22 are inside
    assert len(problems) == 1
    assert "foot" in problems[0]

    eight_and_one = np.vstack([on_the_pitch[:8], behind_the_line])
    assert check_homography(TRUE_H, SIZE, eight_and_one) == []
    assert len(check_homography(TRUE_H, SIZE, eight_and_one, pitch_margin_m=1.0)) == 1  # 8 of 9 are inside
    assert check_homography(TRUE_H, SIZE, eight_and_one, pitch_margin_m=1.0, min_inside_fraction=0.8) == []
    assert check_homography(TRUE_H, SIZE, np.empty((0, 2))) == []
    # A margin of 0 is allowed: the foot points must then lie on the pitch itself.
    assert check_homography(TRUE_H, SIZE, on_the_pitch, pitch_margin_m=0.0) == []
    assert len(check_homography(TRUE_H, SIZE, eight_and_one, pitch_margin_m=0.0)) == 1


def test_check_homography_takes_the_image_size_as_width_then_height() -> None:
    # The centre of the wide image maps to the centre mark. The point (100, 1000), the centre of
    # an image of 200 x 2000 pixels, maps to y = 270 m, far outside the pitch.
    np.testing.assert_allclose(apply_homography(WIDE_IMAGE_H, np.array([[1000.0, 100.0]])), [[0.0, 0.0]], atol=1e-9)
    assert check_homography(WIDE_IMAGE_H, (2000, 200)) == []
    problems = check_homography(WIDE_IMAGE_H, (200, 2000))
    assert len(problems) == 1
    assert "centre" in problems[0]


def test_check_homography_accepts_exactly_the_minimum_share_of_foot_points() -> None:
    on_the_pitch = project(CAMERA, np.array([[float(x), -10.0] for x in range(0, 36, 4)]))  # 9 points
    far_away = project(CAMERA, np.array([[20.0, -50.0], [30.0, -50.0]]))  # 16 m behind the far touchline

    assert check_homography(TRUE_H, SIZE, np.vstack([on_the_pitch, far_away[:1]])) == []  # 9 of 10 are inside
    assert len(check_homography(TRUE_H, SIZE, np.vstack([on_the_pitch, far_away]))) == 1  # 9 of 11 are inside


def test_check_homography_rejects_bad_matrices() -> None:
    with pytest.raises(ValueError):
        check_homography(np.eye(2), SIZE)
    for size in ((0, HEIGHT), (WIDTH, 0), (-WIDTH, HEIGHT), (WIDTH, -HEIGHT)):
        with pytest.raises(ValueError, match="image_size"):
            check_homography(TRUE_H, size)
    for margin in (-1.0, float("nan")):
        with pytest.raises(ValueError, match="pitch_margin_m"):
            check_homography(TRUE_H, SIZE, pitch_margin_m=margin)
    for fraction in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="min_inside_fraction"):
            check_homography(TRUE_H, SIZE, min_inside_fraction=fraction)
    assert check_homography(np.full((3, 3), np.nan), SIZE) != []
    horizon_through_centre = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, -HEIGHT / 2.0]])
    assert check_homography(horizon_through_centre, SIZE) != []
