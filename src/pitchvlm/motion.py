"""Camera motion between the frames of a clip.

The main camera stays in one place. It only turns and zooms. So the pixels of
two frames are linked by a homography, for the pitch and for everything else
that does not move. This module measures that homography for neighbouring
frames, chains it over longer ranges, and uses it to carry the pitch mapping
of a keyframe to the frames around it. Where the chain is broken, the pitch
mappings of two frames give the motion between them (``bridged_warp``).

A hard cut starts a new shot: another camera, a replay or a close-up. No camera
motion explains the change between the two frames of a cut, so the motion chain
is broken there, and nothing is carried across it (``MotionField.cuts``).

All matrices are 3x3 ``float64`` arrays between the pixel grids of two frames.
"""

from __future__ import annotations

import bisect
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
from typing import Any, Self

import cv2
import numpy as np

from pitchvlm.frames import FrameSource
from pitchvlm.geometry import (
    MIN_POINT_PAIRS,
    apply_homography,
    invert_homography,
    matrix_from_record,
    matrix_to_rows,
)
from pitchvlm.schemas import Box, HomographyRecord

logger = logging.getLogger(__name__)

#: Moves image points from a source frame into a destination frame: ``(src, dst, xy) -> xy``.
#: The answer is None when the motion chain between the two frames is broken.
MotionWarp = Callable[[int, int, np.ndarray], np.ndarray | None]

#: Largest number of corners that are followed between two images.
MAX_CORNERS = 1000
#: A corner must reach this share of the quality of the best corner.
CORNER_QUALITY = 0.01
#: Smallest distance between two corners, pixels.
CORNER_MIN_DISTANCE_PX = 8.0
#: Side of the square in which the corner quality is measured, pixels.
CORNER_BLOCK_PX = 7
#: Search window of the optical flow at every pyramid level, pixels.
FLOW_WINDOW_PX = (21, 21)
#: Highest pyramid level of the optical flow. Level 3 is the image at 1/8 size.
FLOW_MAX_LEVEL = 3
#: The optical flow of a corner stops after 30 rounds or when a round moves it less than 0.01 pixels.
FLOW_STOP = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
#: A corner is kept when following it forward and then backward ends this close to its start, pixels.
ROUND_TRIP_MAX_PX = 1.0
#: People boxes are made this much wider and taller before they are masked.
BOX_GROWTH = 1.1
#: Width of the grey images on which a cut is tested, pixels. Small images ignore noise and
#: the small errors of a measured motion.
CUT_TEST_WIDTH = 160
#: The cut test splits the small images into this many columns and rows of blocks and takes the
#: median of the block correlations. So a static graphic (a score box) that covers one or two
#: blocks cannot make two shots look alike.
CUT_TEST_GRID = 4
#: Two neighbouring frames belong to one shot when a camera motion moves the first one onto
#: the second with a median block correlation of at least this. Measured on the synthetic clip:
#: at least 0.88 inside a shot, also for a pan of 83 pixels per frame at 1280 pixels and with a
#: static score graphic; at most 0.06 across a cut (``docs/DECISIONS.md``, entry "Paint
#: alignment of pitch mappings, and hard cuts" of 2026-11-22, item 6).
MIN_SHOT_CORRELATION = 0.5
#: A block counts when the moved image covers at least this share of it.
MIN_CUT_TEST_BLOCK_COVER = 0.5
#: A block counts when both images have at least this standard deviation in it, in grey levels.
MIN_CUT_TEST_STD = 1.0
#: A motion counts when at least this many blocks count.
MIN_CUT_TEST_BLOCKS = 4


def _to_grey(image: np.ndarray) -> np.ndarray:
    """Return a BGR or grey ``uint8`` image as a grey image of shape (height, width)."""
    if image.dtype != np.uint8:
        raise ValueError(f"the image must be uint8, got {image.dtype}")
    if image.ndim == 2:
        return np.ascontiguousarray(image)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"the image must have shape (height, width) or (height, width, 3), got {image.shape}")


def _track_corners(
    prev_grey: np.ndarray, next_grey: np.ndarray, mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Find corners in the first image and follow them into the second image.

    The answer holds the corner positions in the first and in the second image,
    each of shape (k, 2). A corner is kept only when it can be followed forward
    and backward, and the backward track ends close to where the corner started.
    """
    corners = cv2.goodFeaturesToTrack(
        prev_grey,
        maxCorners=MAX_CORNERS,
        qualityLevel=CORNER_QUALITY,
        minDistance=CORNER_MIN_DISTANCE_PX,
        mask=mask,
        blockSize=CORNER_BLOCK_PX,
    )
    if corners is None or len(corners) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    moved, found, _ = cv2.calcOpticalFlowPyrLK(
        prev_grey, next_grey, corners, None, winSize=FLOW_WINDOW_PX, maxLevel=FLOW_MAX_LEVEL, criteria=FLOW_STOP
    )
    back, found_back, _ = cv2.calcOpticalFlowPyrLK(
        next_grey, prev_grey, moved, None, winSize=FLOW_WINDOW_PX, maxLevel=FLOW_MAX_LEVEL, criteria=FLOW_STOP
    )
    start = corners.reshape(-1, 2)
    end = moved.reshape(-1, 2)
    round_trip_px = np.linalg.norm(start - back.reshape(-1, 2), axis=1)
    keep = (found.ravel() == 1) & (found_back.ravel() == 1) & (round_trip_px <= ROUND_TRIP_MAX_PX)
    return start[keep], end[keep]


def estimate_motion(
    prev: np.ndarray,
    nxt: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    ransac_px: float = 3.0,
    min_matches: int = 30,
) -> np.ndarray | None:
    """Estimate the camera motion between two images.

    The answer is the 3x3 matrix that maps pixels of ``prev`` to pixels of
    ``nxt``. Corners of ``prev`` are followed into ``nxt`` with sparse optical
    flow, and RANSAC fits one homography to them. The answer is None when the
    motion cannot be measured: fewer than ``min_matches`` corners agree with
    the homography, or the homography has no inverse.

    Both images are BGR or grey ``uint8`` arrays of the same size. ``mask`` is
    a ``uint8`` image of that size. No corner is taken where it is 0 (people,
    graphics). ``ransac_px`` is the RANSAC threshold in pixels of these images.
    """
    if not 0.0 < ransac_px < math.inf:
        raise ValueError("ransac_px must be a finite number above 0")
    if min_matches < MIN_POINT_PAIRS:
        raise ValueError(f"min_matches must be at least {MIN_POINT_PAIRS}")
    prev_grey = _to_grey(prev)
    next_grey = _to_grey(nxt)
    if prev_grey.shape != next_grey.shape:
        raise ValueError(f"the two images differ in size: {prev_grey.shape} and {next_grey.shape}")
    if mask is not None and (mask.dtype != np.uint8 or mask.shape != prev_grey.shape):
        raise ValueError("the mask must be a uint8 image with the height and width of the images")

    start, end = _track_corners(prev_grey, next_grey, mask)
    if len(start) < min_matches:
        return None
    matrix, inlier_flags = cv2.findHomography(start, end, cv2.RANSAC, float(ransac_px))
    if matrix is None or np.count_nonzero(inlier_flags) < min_matches:
        return None
    if invert_homography(matrix) is None:
        return None
    return np.asarray(matrix, dtype=np.float64)


def _block_correlation(moved: np.ndarray, target: np.ndarray) -> float | None:
    """Return the median Pearson correlation of two images over the blocks of ``CUT_TEST_GRID``.

    ``moved`` is NaN where it covers nothing. A block counts when ``moved``
    covers ``MIN_CUT_TEST_BLOCK_COVER`` of it and both images have contrast
    there (``MIN_CUT_TEST_STD``). The answer is None when fewer than
    ``MIN_CUT_TEST_BLOCKS`` blocks count.
    """
    height, width = target.shape
    values: list[float] = []
    for row in range(CUT_TEST_GRID):
        rows = slice(row * height // CUT_TEST_GRID, (row + 1) * height // CUT_TEST_GRID)
        for column in range(CUT_TEST_GRID):
            columns = slice(column * width // CUT_TEST_GRID, (column + 1) * width // CUT_TEST_GRID)
            block = moved[rows, columns]
            covered = np.isfinite(block)
            if covered.mean() < MIN_CUT_TEST_BLOCK_COVER:
                continue
            first = block[covered]
            second = target[rows, columns][covered]
            if first.std() < MIN_CUT_TEST_STD or second.std() < MIN_CUT_TEST_STD:
                continue
            first = first - first.mean()
            second = second - second.mean()
            scale = math.sqrt(float((first * first).sum()) * float((second * second).sum()))
            values.append(float((first * second).sum()) / scale)
    if len(values) < MIN_CUT_TEST_BLOCKS:
        return None
    return float(np.median(values))


def shot_correlation(prev: np.ndarray, nxt: np.ndarray, motions: Sequence[np.ndarray]) -> float | None:
    """Return how well a camera motion explains the change from one image to the next.

    ``prev`` and ``nxt`` are BGR or grey ``uint8`` images of the same size.
    ``motions`` are 3x3 matrices that map pixels of ``prev`` to pixels of
    ``nxt``, for example the measured motion and the motion of the step
    before; the identity is always tried as well. Both images are shrunk to
    ``CUT_TEST_WIDTH`` pixels wide. ``prev`` is moved with each motion, and the
    median of the block correlations with ``nxt`` is taken
    (``_block_correlation``). The answer is the highest median, or None when
    no motion can be counted: the moved image covers too little, or the
    images have too little contrast.
    """
    prev_grey = _to_grey(prev)
    next_grey = _to_grey(nxt)
    if prev_grey.shape != next_grey.shape:
        raise ValueError(f"the two images differ in size: {prev_grey.shape} and {next_grey.shape}")
    full_size = (prev_grey.shape[1], prev_grey.shape[0])
    test_size = (CUT_TEST_WIDTH, max(1, round(CUT_TEST_WIDTH * full_size[1] / full_size[0])))
    to_test = _to_small_matrix(full_size, test_size)
    from_test = np.linalg.inv(to_test)
    first = cv2.resize(prev_grey, test_size, interpolation=cv2.INTER_AREA).astype(np.float32)
    second = cv2.resize(next_grey, test_size, interpolation=cv2.INTER_AREA).astype(np.float32)
    best: float | None = None
    for motion in [np.eye(3), *motions]:
        matrix = to_test @ np.asarray(motion, dtype=np.float64) @ from_test
        moved = cv2.warpPerspective(first, matrix, test_size, flags=cv2.INTER_LINEAR, borderValue=np.nan)
        value = _block_correlation(moved, second)
        if value is not None and (best is None or value > best):
            best = value
    return best


def is_cut(prev: np.ndarray, nxt: np.ndarray, motions: Sequence[np.ndarray]) -> bool:
    """Tell whether a hard cut lies between two neighbouring frames.

    It does when no camera motion moves ``prev`` onto ``nxt`` with a
    correlation of at least ``MIN_SHOT_CORRELATION`` (``shot_correlation``),
    and also when the correlation is not defined: nothing then links the two
    frames. A pan that is too fast to follow therefore counts as a cut.
    """
    correlation = shot_correlation(prev, nxt, motions)
    return correlation is None or correlation < MIN_SHOT_CORRELATION


def _checked_step(index: int, step: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a copy of a motion step and its inverse. A step that is not a usable matrix raises ``ValueError``."""
    matrix = np.array(step, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"motion step {index} must have shape (3, 3), got {matrix.shape}")
    inverse = invert_homography(matrix)
    if inverse is None:
        raise ValueError(f"motion step {index} has no usable inverse")
    return matrix, inverse


def _checked_cuts(cuts: Sequence[int], steps: Sequence[np.ndarray | None]) -> tuple[int, ...]:
    """Return the cut frames as a tuple. Cuts that are not strictly increasing frames with a missing step raise."""
    checked: list[int] = []
    for cut in cuts:
        if isinstance(cut, bool) or not isinstance(cut, Integral):
            raise TypeError(f"a cut must be a frame index, got {cut!r}")
        if not 1 <= cut <= len(steps):
            raise ValueError(f"cut {cut} is not a frame after the first one (frames 1 to {len(steps)})")
        if checked and cut <= checked[-1]:
            raise ValueError("the cuts must be strictly increasing")
        if steps[cut - 1] is not None:
            raise ValueError(f"the motion step into the cut at frame {cut} must be missing")
        checked.append(int(cut))
    return tuple(checked)


class MotionField:
    """Camera motion of a whole clip.

    ``steps[i]`` is the 3x3 matrix that maps pixels of frame ``i`` to pixels of
    frame ``i + 1``. A step is None when the motion between these two frames is
    not known. Such a step breaks every chain that crosses it.

    ``cuts`` holds the frames that start a new shot, in increasing order. The
    step into a cut is always missing: a cut breaks every chain. A missing
    step without a cut is a break inside one shot (the camera was too fast,
    or the frame had too few corners); the camera still stands at the same
    place there.
    """

    def __init__(self, steps: Sequence[np.ndarray | None], cuts: Sequence[int] = ()) -> None:
        forward: list[np.ndarray | None] = []
        backward: list[np.ndarray | None] = []
        for index, step in enumerate(steps):
            if step is None:
                forward.append(None)
                backward.append(None)
                continue
            matrix, inverse = _checked_step(index, step)
            forward.append(matrix)
            backward.append(inverse)
        self._forward = tuple(forward)
        self._backward = tuple(backward)
        #: Number of frames of the clip: one more than the number of steps.
        self.n_frames: int = len(forward) + 1
        #: The frames that start a new shot, in increasing order.
        self.cuts: tuple[int, ...] = _checked_cuts(cuts, steps)

    def _check_frame(self, frame: int) -> None:
        """Raise when ``frame`` is not a frame of the clip.

        A frame index is a whole number. ``True`` is refused although Python
        counts it as the number 1. An index outside the clip raises
        ``IndexError``.
        """
        if isinstance(frame, bool) or not isinstance(frame, Integral):
            raise TypeError(f"a frame index must be a whole number, got {frame!r}")
        if not 0 <= frame < self.n_frames:
            raise IndexError(f"frame {frame} is outside the clip (frames 0 to {self.n_frames - 1})")

    def between(self, src: int, dst: int) -> np.ndarray | None:
        """Return the 3x3 matrix that maps pixels of frame ``src`` to pixels of frame ``dst``.

        The steps between the two frames are multiplied in frame order. When
        ``src`` is the later frame, the inverse steps are used. For
        ``src == dst`` the answer is the identity. The answer is None when a
        step between the two frames is missing.
        """
        self._check_frame(src)
        self._check_frame(dst)
        chain = self._forward[src:dst] if src <= dst else self._backward[dst:src][::-1]
        matrix = np.eye(3)
        for step in chain:
            if step is None:
                return None
            matrix = step @ matrix
        return matrix

    def warp(self, src: int, dst: int, xy: np.ndarray) -> np.ndarray | None:
        """Move image points of frame ``src`` into frame ``dst``.

        ``xy`` has shape (n, 2) and so has the answer. The answer is None when
        the motion chain between the two frames is broken.
        """
        matrix = self.between(src, dst)
        if matrix is None:
            return None
        return apply_homography(matrix, xy)

    def shot_of(self, frame: int) -> int:
        """Return the number of the shot that holds a frame: 0 before the first cut, 1 after it, and so on."""
        self._check_frame(frame)
        return bisect.bisect_right(self.cuts, frame)

    def same_shot(self, first: int, second: int) -> bool:
        """Tell whether no cut lies between two frames."""
        return self.shot_of(first) == self.shot_of(second)

    @classmethod
    def identity(cls, n_frames: int) -> Self:
        """Return the motion field of a camera that does not move."""
        if n_frames < 1:
            raise ValueError("n_frames must be at least 1")
        return cls([np.eye(3)] * (n_frames - 1))

    def to_json(self) -> dict[str, Any]:
        """Return the field as plain data that the ``json`` module can write."""
        return {
            "n_frames": self.n_frames,
            "steps": [None if step is None else step.tolist() for step in self._forward],
            "cuts": list(self.cuts),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Self:
        """Build a field from the data that ``to_json`` returned."""
        steps = [None if rows is None else np.array(rows, dtype=np.float64) for rows in data["steps"]]
        field = cls(steps, data["cuts"])
        if data["n_frames"] != field.n_frames:
            raise ValueError(f"the data names {data['n_frames']} frames but holds steps for {field.n_frames} frames")
        return field


def bridged_warp(motion: MotionField, homographies: Sequence[HomographyRecord]) -> MotionWarp:
    """Return a warp that crosses a break of the motion chain through the pitch mappings.

    The result moves image points like ``motion.warp`` wherever the motion
    chain between the two frames exists. Across a missing step it goes through
    the pitch: the pitch mapping of the source frame takes the points to the
    pitch, and the inverse mapping of the destination frame takes them back
    into the image. For a camera that only turns and zooms, this is the image
    motion of every point, also of a point above the grass. It is only as
    exact as the two pitch mappings, so the chain is used whenever it exists.
    The answer is None when the chain is broken and one of the two frames has
    no pitch mapping, and when a cut lies between the two frames: another
    shot may show another camera or another moment, so its people are not
    the people at the same pitch place before the cut.

    ``homographies`` holds at most one record per frame, for example the
    result of ``propagate_homographies``. A record whose matrix has no usable
    inverse raises ``ValueError``.
    """
    to_pitch: dict[int, np.ndarray] = {}
    to_image: dict[int, np.ndarray] = {}
    seen: set[int] = set()
    for record in homographies:
        if record.frame in seen:
            raise ValueError(f"frame {record.frame} has more than one homography record")
        seen.add(record.frame)
        if not record.valid:
            continue
        matrix = matrix_from_record(record)
        inverse = invert_homography(matrix)
        if inverse is None:
            raise ValueError(f"the pitch mapping of frame {record.frame} has no usable inverse")
        to_pitch[record.frame] = matrix
        to_image[record.frame] = inverse

    def warp(src: int, dst: int, xy: np.ndarray) -> np.ndarray | None:
        moved = motion.warp(src, dst, xy)
        if moved is not None:
            return moved
        if not motion.same_shot(src, dst) or src not in to_pitch or dst not in to_image:
            return None
        return apply_homography(to_image[dst] @ to_pitch[src], xy)

    return warp


def _small_grey(image: np.ndarray, full_size: tuple[int, int], small_size: tuple[int, int]) -> np.ndarray:
    """Return a frame as a grey image of ``small_size``. Sizes are (width, height)."""
    grey = _to_grey(image)
    if (grey.shape[1], grey.shape[0]) != full_size:
        raise ValueError(f"a frame has size {(grey.shape[1], grey.shape[0])}, the clip says {full_size}")
    return cv2.resize(grey, small_size, interpolation=cv2.INTER_AREA)


def _to_small_matrix(full_size: tuple[int, int], small_size: tuple[int, int]) -> np.ndarray:
    """Return the matrix that maps full-size pixels to pixels of the small image. Sizes are (width, height).

    Pixel centres sit at whole numbers. So a resize by the factor ``s`` moves
    ``x`` to ``s * x + (s - 1) / 2``.
    """
    scale_x = small_size[0] / full_size[0]
    scale_y = small_size[1] / full_size[1]
    return np.array(
        [
            [scale_x, 0.0, (scale_x - 1.0) / 2.0],
            [0.0, scale_y, (scale_y - 1.0) / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _clamp(value: int, upper: int) -> int:
    """Limit a pixel index to the range 0 to ``upper``."""
    return min(max(value, 0), upper)


def _people_mask(small_size: tuple[int, int], boxes: Sequence[Box], to_small: np.ndarray) -> np.ndarray | None:
    """Build the corner mask of a small image: 0 inside the grown people boxes, 255 elsewhere.

    ``small_size`` is (width, height). The boxes are in full-size pixels. Each
    box is made 10% wider and 10% taller around its centre. The answer is None
    when there is no box.
    """
    if not boxes:
        return None
    width, height = small_size
    mask = np.full((height, width), 255, dtype=np.uint8)
    for box in boxes:
        centre_x, centre_y = box.center
        half_width = box.width * BOX_GROWTH / 2.0
        half_height = box.height * BOX_GROWTH / 2.0
        corners = np.array(
            [[centre_x - half_width, centre_y - half_height], [centre_x + half_width, centre_y + half_height]]
        )
        (left, top), (right, bottom) = apply_homography(to_small, corners).tolist()
        columns = slice(_clamp(math.floor(left), width), _clamp(math.ceil(right) + 1, width))
        rows = slice(_clamp(math.floor(top), height), _clamp(math.ceil(bottom) + 1, height))
        mask[rows, columns] = 0
    return mask


def compute_motion_field(
    source: FrameSource,
    *,
    people: Mapping[int, Sequence[Box]] | None = None,
    downscale: float = 0.5,
) -> MotionField:
    """Measure the camera motion between all neighbouring frames of a clip.

    The frames are read in order. The work is done on grey images scaled by
    ``downscale`` (for speed), and every matrix is then converted back to
    full-size pixels. ``people`` maps a frame index to the boxes of the people
    in that frame, in full-size pixels. For a pair of frames, the boxes of both
    frames are grown by 10% and no corner is taken inside them. A pair whose
    motion cannot be measured gives a missing step.

    Each pair is also tested for a hard cut (``is_cut``) with the measured
    motion and the last measured motion of the same shot. A cut gives a
    missing step, also when a motion was measured: static broadcast graphics
    can give a motion across a cut. The cuts are in ``MotionField.cuts``.
    """
    if not 0.0 < downscale <= 1.0:
        raise ValueError("downscale must be above 0 and at most 1")
    info = source.info
    full_size = (info.width, info.height)
    small_size = (max(1, round(info.width * downscale)), max(1, round(info.height * downscale)))
    to_small = _to_small_matrix(full_size, small_size)
    to_full = np.linalg.inv(to_small)
    boxes: Mapping[int, Sequence[Box]] = {} if people is None else people

    steps: list[np.ndarray | None] = []
    cuts: list[int] = []
    last_step: np.ndarray | None = None
    previous = _small_grey(source.read(0), full_size, small_size)
    for frame in range(1, info.n_frames):
        current = _small_grey(source.read(frame), full_size, small_size)
        mask = _people_mask(small_size, [*boxes.get(frame - 1, ()), *boxes.get(frame, ())], to_small)
        step = estimate_motion(previous, current, mask=mask)
        if is_cut(previous, current, [motion for motion in (step, last_step) if motion is not None]):
            logger.info("clip %s: frame %d starts a new shot", info.clip_id, frame)
            cuts.append(frame)
            steps.append(None)
            last_step = None
        else:
            steps.append(None if step is None else to_full @ step @ to_small)
            last_step = step if step is not None else last_step
        previous = current
    return MotionField(steps, cuts)


def _propagated_record(
    frame: int,
    key_matrices: Mapping[int, np.ndarray],
    motion: MotionField,
    max_gap_frames: int,
) -> HomographyRecord:
    """Return the pitch mapping of a frame that has no landmark fit of its own.

    The keyframes are tried from the nearest to the farthest. At equal distance
    the earlier keyframe comes first. The first keyframe that the motion chain
    reaches gives the mapping.
    """
    in_reach = [key for key in key_matrices if abs(key - frame) <= max_gap_frames]
    for key in sorted(in_reach, key=lambda candidate: (abs(candidate - frame), candidate)):
        to_key = motion.between(frame, key)
        if to_key is not None:
            matrix = key_matrices[key] @ to_key
            return HomographyRecord(frame=frame, matrix=matrix_to_rows(matrix), source="propagated")
    return HomographyRecord(frame=frame)


def propagate_homographies(
    key_fits: Mapping[int, HomographyRecord],
    motion: MotionField,
    n_frames: int,
    *,
    max_gap_frames: int = 75,
) -> list[HomographyRecord]:
    """Give every frame of a clip a pitch mapping, as far as one can be found.

    ``key_fits`` maps a keyframe index to its landmark fit. The answer has one
    record per frame, in frame order:

    - A frame with a valid fit of its own keeps that record.
    - Any other frame takes the fit of the nearest keyframe that is at most
      ``max_gap_frames`` away and that the motion chain reaches. The camera
      motion from the frame to the keyframe is put in front of the keyframe
      mapping: ``H_frame = H_key @ M(frame -> key)``. The record has
      ``source="propagated"``, no inlier count and no reprojection error.
    - A frame without such a keyframe gets ``source="none"``.
    """
    if n_frames != motion.n_frames:
        raise ValueError(f"the clip has {n_frames} frames but the motion field has {motion.n_frames}")
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must not be negative")
    key_matrices: dict[int, np.ndarray] = {}
    for frame, record in key_fits.items():
        if record.frame != frame:
            raise ValueError(f"the record stored for frame {frame} belongs to frame {record.frame}")
        if not 0 <= frame < n_frames:
            raise ValueError(f"keyframe {frame} is outside the clip (frames 0 to {n_frames - 1})")
        if record.valid:
            key_matrices[frame] = matrix_from_record(record)

    records: list[HomographyRecord] = []
    for frame in range(n_frames):
        if frame in key_matrices:
            records.append(key_fits[frame])
        else:
            records.append(_propagated_record(frame, key_matrices, motion, max_gap_frames))
    return records
