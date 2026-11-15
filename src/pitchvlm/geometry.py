"""Homography between the image and the pitch, fitted from landmark points.

A homography is a 3x3 ``float64`` matrix. In this module it maps image pixels
to pitch metres: ``pitch ~ H @ (x_px, y_px, 1)``. Image x grows to the right
and image y grows down. Pitch x grows towards the right goal and pitch y grows
towards the main camera, so a point on the far side has a negative y.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

from pitchvlm.pitch import LANDMARKS, inside_pitch, landmark_xy
from pitchvlm.schemas import FrameLandmarks, HomographyRecord

#: A homography has 8 free values, so a fit needs at least 4 point pairs.
MIN_POINT_PAIRS = 4
#: Pitch points this close to one line (root mean square distance, metres) cannot fix a homography.
COLLINEAR_TOLERANCE_M = 0.5
#: Image points this close to one line (root mean square distance, pixels) cannot fix a homography.
COLLINEAR_TOLERANCE_PX = 1.0
#: A point pair is near a fit when its error is at most this many times the RANSAC threshold. A
#: point with a wrong landmark id lies tens to hundreds of pixels beside its place. A point that
#: the model placed with some noise does not.
NEAR_LIMIT_FACTOR = 3.0
#: With more than 4 point pairs, at least this share of the pairs must be near a fit.
MIN_NEAR_SHARE = 0.4
#: A matrix has no usable inverse when its smallest singular value is at most this share of its largest one.
SINGULAR_RATIO = 1e-12
#: The image centre must map into the pitch enlarged by this margin, metres.
CENTRE_MARGIN_M = 30.0

_KNOWN_LANDMARK_IDS = frozenset(landmark.id for landmark in LANDMARKS)


@dataclass(frozen=True)
class HomographyFit:
    """Result of a homography fit from point pairs."""

    #: 3x3 matrix that maps image pixels to pitch metres.
    H: np.ndarray
    #: Boolean array with one entry per input pair. True means that the pair agrees with the fit.
    inliers: np.ndarray
    #: Median reprojection error of the inliers, in image pixels.
    reproj_px: float


def _as_points(xy: np.ndarray) -> np.ndarray:
    """Return points as a ``float64`` array of shape (n, 2).

    A single point of shape (2,) becomes shape (1, 2). An input without values
    becomes shape (0, 2). Any other shape raises ``ValueError``, so that an
    array of shape (n, 3) or (k, n, 2) is never cut into wrong points.
    """
    points = np.asarray(xy, dtype=np.float64)
    if points.size == 0:
        return points.reshape(0, 2)
    if points.shape == (2,):
        return points.reshape(1, 2)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (n, 2), got {points.shape}")
    return points


def _as_matrix(H: np.ndarray) -> np.ndarray:
    """Return a homography as a ``float64`` array. A shape other than (3, 3) raises ``ValueError``."""
    matrix = np.asarray(H, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"a homography must have shape (3, 3), got {matrix.shape}")
    return matrix


def apply_homography(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Map points through a homography.

    ``xy`` has shape (n, 2) and the result has shape (n, 2). The scale of ``H``
    does not change the result. A point that ``H`` sends to infinity (a point
    on the horizon line) gets coordinates that are not finite.
    """
    matrix = _as_matrix(H)
    points = _as_points(xy)
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return homogeneous[:, :2] / homogeneous[:, 2:3]


def invert_homography(H: np.ndarray) -> np.ndarray | None:
    """Return the inverse of a homography, or None when it has no usable inverse.

    A matrix has no usable inverse when one of its values is not finite, or
    when it is singular. The singular test compares the smallest singular value
    with the largest one (``SINGULAR_RATIO``). So it does not depend on the
    scale of the matrix, and it also finds a matrix that is singular only up to
    rounding errors. A plain inversion accepts such a matrix and returns huge
    values that mean nothing.
    """
    matrix = _as_matrix(H)
    if not np.isfinite(matrix).all():
        return None
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values[-1] <= SINGULAR_RATIO * singular_values[0]:
        return None
    inverse = np.linalg.inv(matrix)
    if not np.isfinite(inverse).all():
        return None
    return inverse


def matrix_to_rows(H: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    """Turn a 3x3 matrix into the row tuples that a ``HomographyRecord`` stores."""
    matrix = _as_matrix(H)
    return tuple((float(row[0]), float(row[1]), float(row[2])) for row in matrix)


def matrix_from_record(record: HomographyRecord) -> np.ndarray:
    """Return the matrix of a record as a 3x3 ``float64`` array.

    A record without a matrix raises ``ValueError``. Test ``record.valid`` first.
    """
    if record.matrix is None:
        raise ValueError(f"the homography record of frame {record.frame} has no matrix")
    return _as_matrix(np.array(record.matrix, dtype=np.float64))


def are_collinear(xy: np.ndarray, tolerance: float) -> bool:
    """Tell whether all points lie close to one straight line.

    The measure is the root mean square distance of the points to their best
    line. This is the second singular value of the centred points divided by
    ``sqrt(n)``. The points are collinear when the measure is at most
    ``tolerance``, which has the unit of the points. Fewer than three points
    are always collinear.

    ``tolerance`` is a finite number that is not negative. With 0 the points
    must lie exactly on one line. The rounding error of the measure is then
    allowed for, so points on an exact line that is not level or upright still
    count.
    """
    if not 0.0 <= tolerance < math.inf:
        raise ValueError("tolerance must be a finite number that is not negative")
    points = _as_points(xy)
    count = len(points)
    if count < 3:
        return True
    singular_values = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    rounding = count * np.finfo(np.float64).eps * singular_values[0]
    return bool(singular_values[1] <= tolerance * math.sqrt(count) + rounding)


def _leave_homography_open(xy: np.ndarray, tolerance: float) -> bool:
    """Tell whether point pairs at these points cannot fix a homography.

    A homography is fixed by four points of which no three lie on one line.
    Such four points do not exist when all points lie on one line, or all
    points but one. A fit through such points looks perfect but is wrong away
    from the line. The rule holds on both sides of the point pairs, the pitch
    and the image. The line test is ``are_collinear`` with ``tolerance``, so
    two points at one place count as well.
    """
    others = np.ones(len(xy), dtype=bool)
    for index in range(len(xy)):
        others[index] = False
        if are_collinear(xy[others], tolerance):
            return True
        others[index] = True
    return False


def _inliers_needed(pair_count: int, min_inliers: int) -> int:
    """Return the smallest number of inliers that a fit from ``pair_count`` point pairs must have.

    Four pairs always fit one homography exactly. So the four pairs of a
    RANSAC sample always agree with their own model, and a fit with four
    inliers is supported by its own sample only. When more than four pairs are
    given, at least one more pair must agree with the fit. Without such a
    pair, nothing confirms the fit and all other pairs speak against it.
    """
    if pair_count > MIN_POINT_PAIRS:
        return max(min_inliers, MIN_POINT_PAIRS + 1)
    return min_inliers


def _near_pairs_needed(pair_count: int, min_near_share: float) -> int:
    """Return the smallest number of pairs that must be near a fit from ``pair_count`` point pairs.

    One pair that confirms a sample is not enough when many pairs are given.
    RANSAC tries thousands of samples, so with 10 to 18 wrong pairs one of them
    often agrees with some sample by chance. So with more than four pairs, at
    least ``min_near_share`` of them must be near the fit.

    The share is not counted with the RANSAC threshold. With noise of the size
    of that threshold, fewer than 40% of the right points lie inside it, and a
    fit that is as exact as the others would be refused. A wrong pair is far
    outside the wider limit, a noisy one is inside it.
    """
    if pair_count > MIN_POINT_PAIRS:
        return math.ceil(min_near_share * pair_count)
    return 0


def fit_homography(
    image_xy: np.ndarray,
    pitch_xy: np.ndarray,
    *,
    ransac_px: float = 8.0,
    min_inliers: int = 4,
    min_near_share: float = MIN_NEAR_SHARE,
) -> HomographyFit | None:
    """Fit the image-to-pitch homography from point pairs with RANSAC.

    ``image_xy`` holds pixels and ``pitch_xy`` holds metres. Both have shape
    (n, 2), and row ``i`` of one belongs to row ``i`` of the other. The fit runs
    from pitch to image, so that ``ransac_px`` is a distance in pixels. The
    result is then inverted.

    An inlier is a pair whose pitch point lands at most ``ransac_px`` from its
    image point under the returned matrix. The solver's own inlier marks are
    not used: with exactly 4 pairs it marks every pair, whatever its error is.

    The answer is None when no trustworthy fit exists: fewer than 4 pairs, a
    failed fit, a matrix without a usable inverse, too few inliers, or inliers
    that cannot fix a homography. That is the case when all inlier points lie
    on one line (for example marks on the halfway line only), and also when
    all but one of them do. The rule holds for the pitch points
    (``COLLINEAR_TOLERANCE_M``) and for the image points
    (``COLLINEAR_TOLERANCE_PX``): two landmarks at one image point, or three
    image points on one line, give no fit from four pairs.

    A fit needs at least ``min_inliers`` inliers. When more than 4 pairs are
    given, it also needs at least 5 inliers. The 4 pairs of a RANSAC sample
    always agree with their own model. So a fit with 4 inliers has no pair that
    confirms it, and every other pair speaks against it. With more than 4 pairs,
    at least ``min_near_share`` of all pairs must also be near the fit, because
    among many wrong pairs a fifth one agrees with some sample by chance. A
    pair is near the fit when its error is at most ``NEAR_LIMIT_FACTOR`` times
    ``ransac_px``: a point with a wrong landmark id is not, a point with some
    noise is. With ``min_near_share`` 0 this rule is off. Exactly 4 pairs give
    an exact fit that no other pair can confirm or contradict. That fit is
    returned when ``min_inliers`` is 4.

    The scale of a homography is free. ``H`` is returned with a Frobenius norm
    of 1. It is not divided by ``H[2, 2]``, because that entry is zero when the
    horizon passes through the image origin.
    """
    image_points = _as_points(image_xy)
    pitch_points = _as_points(pitch_xy)
    if len(image_points) != len(pitch_points):
        raise ValueError(f"{len(image_points)} image points do not pair with {len(pitch_points)} pitch points")
    if not (np.isfinite(image_points).all() and np.isfinite(pitch_points).all()):
        raise ValueError("point coordinates must be finite")
    if not 0.0 < ransac_px < math.inf:
        raise ValueError("ransac_px must be a finite number above 0")
    if min_inliers < MIN_POINT_PAIRS:
        raise ValueError(f"min_inliers must be at least {MIN_POINT_PAIRS}")
    if not 0.0 <= min_near_share <= 1.0:
        raise ValueError("min_near_share must be between 0 and 1")
    if len(image_points) < MIN_POINT_PAIRS:
        return None

    pitch_to_image, _ = cv2.findHomography(pitch_points, image_points, cv2.RANSAC, float(ransac_px))
    if pitch_to_image is None:
        return None
    image_to_pitch = invert_homography(pitch_to_image)
    if image_to_pitch is None:
        return None
    errors_px = np.linalg.norm(apply_homography(pitch_to_image, pitch_points) - image_points, axis=1)
    # An error that is not finite (a pitch point that lands at infinity) is not below the limit.
    inliers = errors_px <= ransac_px
    if int(inliers.sum()) < _inliers_needed(len(image_points), min_inliers):
        return None
    near = errors_px <= NEAR_LIMIT_FACTOR * ransac_px
    if int(near.sum()) < _near_pairs_needed(len(image_points), min_near_share):
        return None
    if _leave_homography_open(pitch_points[inliers], COLLINEAR_TOLERANCE_M):
        return None
    if _leave_homography_open(image_points[inliers], COLLINEAR_TOLERANCE_PX):
        return None

    reproj_px = float(np.median(errors_px[inliers]))
    return HomographyFit(H=image_to_pitch / np.linalg.norm(image_to_pitch), inliers=inliers, reproj_px=reproj_px)


def landmark_pairs(obs: FrameLandmarks) -> tuple[np.ndarray, np.ndarray]:
    """Return the point pairs of the landmark observations of a frame that a fit can use.

    The result is the image points in pixels and the pitch points in metres,
    both of shape (n, 2); row ``i`` of one belongs to row ``i`` of the other.
    An observation with an unknown landmark id is dropped. When a landmark id
    appears more than once, every observation of that id is dropped, because it
    is not known which one is right.
    """
    counts = Counter(item.landmark_id for item in obs.landmarks)
    kept = [item for item in obs.landmarks if item.landmark_id in _KNOWN_LANDMARK_IDS and counts[item.landmark_id] == 1]
    image_xy = np.array([[item.x, item.y] for item in kept], dtype=np.float64).reshape(-1, 2)
    return image_xy, landmark_xy([item.landmark_id for item in kept])


def count_near(H: np.ndarray, image_xy: np.ndarray, pitch_xy: np.ndarray, *, ransac_px: float) -> int:
    """Count the point pairs that are near an image-to-pitch mapping.

    A pair is near the mapping when its pitch point lands at most
    ``NEAR_LIMIT_FACTOR`` times ``ransac_px`` from its image point, in image
    pixels: this is the limit of the share rule of ``fit_homography``. A
    mapping without a usable inverse places no pitch point in the image, so no
    pair is near it.
    """
    if not 0.0 < ransac_px < math.inf:
        raise ValueError("ransac_px must be a finite number above 0")
    image_points = _as_points(image_xy)
    pitch_points = _as_points(pitch_xy)
    if len(image_points) != len(pitch_points):
        raise ValueError(f"{len(image_points)} image points do not pair with {len(pitch_points)} pitch points")
    pitch_to_image = invert_homography(H)
    if pitch_to_image is None:
        return 0
    errors_px = np.linalg.norm(apply_homography(pitch_to_image, pitch_points) - image_points, axis=1)
    # An error that is not finite (a pitch point that lands at infinity) is not below the limit.
    return int((errors_px <= NEAR_LIMIT_FACTOR * ransac_px).sum())


def fit_from_landmarks(
    obs: FrameLandmarks, *, ransac_px: float = 8.0, min_near_share: float = MIN_NEAR_SHARE
) -> HomographyRecord:
    """Fit the pitch mapping of one frame from its landmark observations.

    The fit uses the point pairs of ``landmark_pairs``: an observation with an
    unknown landmark id, or with an id that appears more than once, is
    dropped. The record has ``source="none"`` and no matrix when
    ``fit_homography`` finds no trustworthy fit for the landmarks that are
    left. With more than 4 such landmarks, at least 5 of them must agree with
    the fit, and at least ``min_near_share`` of them must be near it. An
    answer never raises here: landmarks at one image point, or on one image
    line, give a record without a matrix.
    """
    image_xy, pitch_xy = landmark_pairs(obs)
    fit = fit_homography(image_xy, pitch_xy, ransac_px=ransac_px, min_near_share=min_near_share)
    if fit is None:
        return HomographyRecord(frame=obs.frame)
    return HomographyRecord(
        frame=obs.frame,
        matrix=matrix_to_rows(fit.H),
        source="landmarks",
        n_inliers=int(fit.inliers.sum()),
        reproj_px=fit.reproj_px,
    )


def _centre_problems(matrix: np.ndarray, width: float, height: float) -> list[str]:
    """Check where the image centre lands on the pitch and how the pitch axes run in the image."""
    centre = np.array([width / 2.0, height / 2.0])
    steps = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    at_centre, at_right, at_below = apply_homography(matrix, centre + steps)
    if not np.isfinite([at_centre, at_right, at_below]).all():
        return ["the image centre has no finite position on the pitch"]
    problems: list[str] = []
    if not inside_pitch(at_centre, CENTRE_MARGIN_M)[0]:
        problems.append(
            f"the image centre maps to x={at_centre[0]:.1f} m, y={at_centre[1]:.1f} m, "
            f"more than {CENTRE_MARGIN_M:.0f} m outside the pitch"
        )
    if not at_right[0] > at_centre[0]:
        problems.append("pitch x does not grow to the right of the image: the mapping is mirrored")
    if not at_below[1] > at_centre[1]:
        problems.append("pitch y does not grow towards the bottom of the image: the mapping is flipped")
    return problems


def _foot_problems(
    matrix: np.ndarray, feet_xy: np.ndarray, pitch_margin_m: float, min_inside_fraction: float
) -> list[str]:
    """Check that enough foot points land inside the pitch enlarged by the margin."""
    feet = _as_points(feet_xy)
    if len(feet) == 0:
        return []
    with np.errstate(invalid="ignore"):  # A foot point at infinity is outside the pitch.
        inside_fraction = float(inside_pitch(apply_homography(matrix, feet), pitch_margin_m).mean())
    if inside_fraction >= min_inside_fraction:
        return []
    return [
        f"only {inside_fraction:.0%} of the foot points map into the pitch plus {pitch_margin_m:g} m, "
        f"the minimum is {min_inside_fraction:.0%}"
    ]


def check_homography(
    H: np.ndarray,
    image_size: tuple[int, int],
    feet_xy: np.ndarray | None = None,
    *,
    pitch_margin_m: float = 3.0,
    min_inside_fraction: float = 0.9,
) -> list[str]:
    """List the reasons why an image-to-pitch homography is not plausible.

    An empty list means that the mapping is plausible for a main camera on the
    near side. ``image_size`` is ``(width, height)`` in pixels. ``feet_xy``
    holds the foot points of the people in the frame, in image pixels, shape
    (n, 2).

    The checks are:

    - The image centre maps into the pitch enlarged by 30 m.
    - One step to the right in the image increases pitch x. If not, the
      mapping is mirrored.
    - One step down in the image increases pitch y. If not, the mapping is
      flipped (near side and far side are swapped).
    - At least ``min_inside_fraction`` of the foot points map into the pitch
      enlarged by ``pitch_margin_m``.
    """
    matrix = _as_matrix(H)
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must be (width, height) with positive values, got {image_size}")
    if not pitch_margin_m >= 0.0:
        raise ValueError("pitch_margin_m must not be negative")
    if not 0.0 <= min_inside_fraction <= 1.0:
        raise ValueError("min_inside_fraction must be between 0 and 1")
    if not np.isfinite(matrix).all():
        return ["the matrix has values that are not finite"]
    problems = _centre_problems(matrix, width, height)
    if feet_xy is not None:
        problems.extend(_foot_problems(matrix, feet_xy, pitch_margin_m, min_inside_fraction))
    return problems
