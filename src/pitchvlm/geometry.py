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

from pitchvlm.pitch import LANDMARKS, inside_pitch, landmark_xy, pitch_polylines
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

# Paint alignment (``paint_alignment``). Sizes in pixels are given as a share of the image
# width or height, so that they hold for every image size. The values were chosen on the
# synthetic clip (``docs/DECISIONS.md``, 2026-11-22); real broadcast frames may need others.

#: Side of the square of the white top-hat that finds the paint, as a share of the image
#: width (17 pixels at 1280). A line that is thinner than the square is kept; a wide bright
#: area (a white advertising board, a sunny stripe of grass) is not.
PAINT_TOPHAT_PER_WIDTH = 0.0125
#: A paint pixel is at least this much brighter than its surroundings, in grey levels of the
#: darkest colour channel. White paint is bright in all three channels, grass is not. The
#: dark marks of the synthetic grass make ridges of 55 levels between them; the synthetic
#: lines stand 185 levels above the grass.
PAINT_MIN_CONTRAST = 80
#: A projected line pixel lies on the paint when the nearest paint pixel is at most this far,
#: as a share of the image width (7.7 pixels at 1280).
PAINT_TOLERANCE_PER_WIDTH = 0.006
#: A frame shows too little paint to judge a mapping below this share of paint pixels.
PAINT_MIN_SHARE = 0.002
#: A frame shows too much white to judge a mapping when this share of the image or more lies
#: within the tolerance of some paint: a line placed anywhere would then hit paint.
PAINT_MAX_CHANCE = 0.5
#: The hits of a mapping are spread well when their root mean square distance to their best
#: straight line is at least this share of the image height (58 pixels at 720).
PAINT_SPREAD_PER_HEIGHT = 0.08

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


# --- Paint alignment -------------------------------------------------------------------


@dataclass(frozen=True)
class PaintAlignment:
    """How well the pitch lines that a mapping projects into a frame lie on the white paint of the frame.

    ``score`` is in 0 to 1, or None when the frame cannot judge the mapping;
    ``reason`` then says why. The other values are the parts of the score.
    """

    #: ``precision_above_chance * spread``, or None when the frame cannot judge a mapping.
    score: float | None
    #: Why the frame cannot judge a mapping, or None when it can.
    reason: str | None
    #: Share of the image pixels that are paint.
    paint_share: float
    #: Share of the image pixels that lie within the tolerance of some paint: the precision of
    #: lines placed by chance.
    chance: float
    #: Number of image pixels that the projected pitch lines cover.
    n_model_px: int
    #: Share of these pixels that lie on paint, or None when the lines miss the image.
    precision: float | None
    #: Spread of the pixels that lie on paint, 0 to 1 (``PAINT_SPREAD_PER_HEIGHT``).
    spread: float


def _pitch_segments() -> np.ndarray:
    """Return every straight piece of the pitch markings as rows ``[x0, y0, x1, y1]`` in metres."""
    pieces: list[np.ndarray] = []
    for line in pitch_polylines():
        ends = np.roll(line.points, -1, axis=0) if line.closed else line.points[1:]
        pieces.append(np.hstack([line.points[: len(ends)], ends]))
    return np.vstack(pieces)


_PITCH_SEGMENTS = _pitch_segments()


def _odd_size(pixels: float) -> int:
    """Return the odd whole number of at least 3 that is nearest to ``pixels``."""
    size = max(3, round(pixels))
    return size if size % 2 == 1 else size + 1


def paint_mask(image: np.ndarray) -> np.ndarray:
    """Return the white paint of a frame as a ``uint8`` mask of shape (height, width): 1 for paint.

    ``image`` is a BGR ``uint8`` frame. Paint is a thin bright ridge on
    grass: the white top-hat of the darkest colour channel, with a square of
    ``PAINT_TOPHAT_PER_WIDTH`` of the width, is at least
    ``PAINT_MIN_CONTRAST``, and the pixel lies on grass or within that square
    of grass. A grass pixel is greener than it is red and blue. So white kits
    on the grass count as paint, but the crowd, the sky and the boards do not.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"the frame must be a BGR uint8 image of shape (height, width, 3), got {image.shape}")
    size = _odd_size(PAINT_TOPHAT_PER_WIDTH * image.shape[1])
    square = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    ridges = cv2.morphologyEx(image.min(axis=2), cv2.MORPH_TOPHAT, square)
    blue, green, red = (image[:, :, channel].astype(np.int16) for channel in range(3))
    near_grass = cv2.dilate(((green > red) & (green > blue)).astype(np.uint8), square)
    return ((ridges >= PAINT_MIN_CONTRAST) & (near_grass > 0)).astype(np.uint8)


def _clip_to_front(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Cut a segment of homogeneous image points to the part with a positive third coordinate.

    The third coordinate changes linearly along the segment, so this part is
    one piece. The answer is None when no part of the segment is in front.
    """
    w_start, w_end = start[2], end[2]
    if w_start <= 0.0 and w_end <= 0.0:
        return None
    floor = 1e-9 * max(abs(w_start), abs(w_end))
    if w_start < floor:
        start = start + (floor - w_start) / (w_end - w_start) * (end - start)
    elif w_end < floor:
        end = end + (floor - w_end) / (w_start - w_end) * (start - end)
    return start, end


def _clip_to_image(
    start: np.ndarray, end: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Cut an image segment to the pixel centres of the image (Liang-Barsky). None means outside."""
    step = end - start
    low, high = 0.0, 1.0
    for direction, room in (
        (-step[0], start[0]),
        (step[0], width - 1 - start[0]),
        (-step[1], start[1]),
        (step[1], height - 1 - start[1]),
    ):
        if direction == 0.0:
            if room < 0.0:
                return None
            continue
        limit = room / direction
        if direction < 0.0:
            low = max(low, limit)
        else:
            high = min(high, limit)
    if low > high:
        return None
    return start + low * step, start + high * step


def projected_line_pixels(H: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Return the image pixels that the pitch lines cover under an image-to-pitch mapping.

    ``image_size`` is ``(width, height)``. The answer holds whole pixel
    coordinates ``(x, y)``, shape (n, 2), each pixel once. Every straight
    piece of ``pitch.pitch_polylines`` (the circle and the arcs as short
    chords) is projected into the image, cut to the part in front of the
    camera and to the image, and sampled once per pixel of its length. The
    spots are not used.

    In front of the camera is the side of the horizon on which the pitch
    point lies that the mapping gives for the bottom centre of the image: a
    main camera sees the grass at the bottom of its picture. A mapping without
    a usable inverse raises ``ValueError``.
    """
    matrix = _as_matrix(H)
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must be (width, height) with positive values, got {image_size}")
    to_image = invert_homography(matrix)
    if to_image is None:
        raise ValueError("the mapping has no usable inverse")
    bottom = matrix @ np.array([width / 2.0, height - 1.0, 1.0])
    if bottom[2] < 0.0:
        to_image = -to_image
    ones = np.ones((len(_PITCH_SEGMENTS), 1))
    starts = np.hstack([_PITCH_SEGMENTS[:, :2], ones]) @ to_image.T
    ends = np.hstack([_PITCH_SEGMENTS[:, 2:], ones]) @ to_image.T
    samples: list[np.ndarray] = []
    for start, end in zip(starts, ends, strict=True):
        in_front = _clip_to_front(start, end)
        if in_front is None:
            continue
        front_start, front_end = in_front
        inside = _clip_to_image(front_start[:2] / front_start[2], front_end[:2] / front_end[2], width, height)
        if inside is None:
            continue
        first, last = inside
        count = math.ceil(float(np.linalg.norm(last - first))) + 1
        samples.append(first + np.linspace(0.0, 1.0, count)[:, None] * (last - first))
    if not samples:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.round(np.vstack(samples)).astype(np.int64), axis=0)


def _line_spread(points: np.ndarray) -> float:
    """Return the root mean square distance of points to their best straight line. Fewer than 3 points give 0."""
    if len(points) < 3:
        return 0.0
    singular_values = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    return float(singular_values[1] / math.sqrt(len(points)))


def paint_alignment(image: np.ndarray, H: np.ndarray) -> PaintAlignment:
    """Measure how well the pitch lines of an image-to-pitch mapping lie on the white paint of a frame.

    ``image`` is the BGR ``uint8`` frame and ``H`` maps its pixels to pitch
    metres. The paint is ``paint_mask``; the projected lines are
    ``projected_line_pixels``. A projected pixel hits the paint when a paint
    pixel lies within ``PAINT_TOLERANCE_PER_WIDTH`` of the width.

    - ``precision`` is the share of the projected pixels that hit. ``chance``
      is the share of all image pixels that would hit; the precision is
      counted above it: ``(precision - chance) / (1 - chance)``, at least 0.
    - ``spread`` is the root mean square distance of the hits to their best
      straight line, divided by ``PAINT_SPREAD_PER_HEIGHT`` of the height, at
      most 1. Hits on one line, or in one corner, have a small spread: one
      line does not fix a mapping.
    - ``score`` is the product of the two, from 0 to 1. Lines that miss the
      image score 0 in a frame that shows paint.

    The frame cannot judge a mapping, and ``score`` is None, when less than
    ``PAINT_MIN_SHARE`` of it is paint (a close-up, a view of the crowd, a
    frame without lines), and when ``PAINT_MAX_CHANCE`` or more of it lies
    within the tolerance of paint. Such a frame says nothing about the
    mapping, so it gets no low score.

    The pitch markings are symmetric: a mapping that swaps the two halves or
    the two sides of the pitch projects the same lines and gets the same
    score. ``check_homography`` finds such a mapping. The value is a
    diagnostic, not a rule of the pipeline (``docs/DECISIONS.md``).
    """
    height, width = image.shape[:2]
    mask = paint_mask(image)
    distance = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    near_paint = distance <= PAINT_TOLERANCE_PER_WIDTH * width
    paint_share = float(mask.mean())
    chance = float(near_paint.mean())
    pixels = projected_line_pixels(H, (width, height))
    hits = near_paint[pixels[:, 1], pixels[:, 0]]
    precision = float(hits.mean()) if len(pixels) else None
    spread = min(1.0, _line_spread(pixels[hits].astype(np.float64)) / (PAINT_SPREAD_PER_HEIGHT * height))
    reason: str | None = None
    if paint_share < PAINT_MIN_SHARE:
        reason = f"too little paint: {paint_share:.2%} of the frame, the minimum is {PAINT_MIN_SHARE:.2%}"
    elif chance >= PAINT_MAX_CHANCE:
        reason = f"too much white: {chance:.0%} of the frame is near paint, the limit is {PAINT_MAX_CHANCE:.0%}"
    score: float | None = None
    if reason is None:
        above_chance = 0.0 if precision is None else max(0.0, (precision - chance) / (1.0 - chance))
        score = above_chance * spread
    return PaintAlignment(
        score=score,
        reason=reason,
        paint_share=paint_share,
        chance=chance,
        n_model_px=len(pixels),
        precision=precision,
        spread=spread,
    )
