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

A dissolve blends two shots over some frames. Each frame differs from the next
by a small step, so the test of neighbouring frames does not see it. A window
test does: no camera motion moves the first frame of a window onto its last
frame (``window_correlations``). Every frame of a dissolve then starts a shot of
its own, and the frame after it starts the next shot (``find_transitions``). A
shot of one frame carries nothing (``MotionField.transition_frames``).

All matrices are 3x3 ``float64`` arrays between the pixel grids of two frames.
"""

from __future__ import annotations

import bisect
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
#: Length of the window of the dissolve test, in frames: the test compares frame ``t - 30`` with
#: frame ``t``. A dissolve of up to 29 frames lies between the two ends of some window. Chosen for
#: the dissolves of 5 to 25 frames of the measurement, at 25 frames per second (``docs/DECISIONS.md``,
#: entry "A dissolve is a transition, not a shot" of 2026-11-22).
DISSOLVE_WINDOW_FRAMES = 30
#: A window holds a transition when the chained camera motion moves its first frame onto its last
#: with a median block correlation below this. Fixed on the synthetic seeds 0 to 9, halfway between
#: the highest lowest window value of a found dissolve (0.087) and the lowest one of a control
#: without a cut (0.268, a pan of 99 px per frame) (``docs/DECISIONS.md``, entry "A dissolve is a
#: transition, not a shot"). It is lower than ``MIN_SHOT_CORRELATION``: small errors of 30 steps
#: add up in a fast pan.
MAX_TRANSITION_CORRELATION = 0.18
#: A frame of a transition still shows the shot of an anchor frame when the anchor frame, moved
#: with the camera motion, explains it with a regression slope of at least this. A frame that
#: blends in another shot with the weight ``w`` has a slope near ``1 - w``. Fixed on the synthetic
#: seeds 0 to 9, halfway between the highest slope of a frame with a blend weight above 0.2 (0.898)
#: and the lowest slope of a frame of the anchor's own shot (1.009) (``docs/DECISIONS.md``, entry
#: "A dissolve is a transition, not a shot").
MIN_SHOT_SHARE = 0.95


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


def _block_correlation(
    moved: np.ndarray, target: np.ndarray, whole_blocks: dict[tuple[int, int], Any] | None = None
) -> float | None:
    """Return the median Pearson correlation of two images over the blocks of ``CUT_TEST_GRID``.

    ``moved`` is NaN where it covers nothing. A block counts when ``moved``
    covers ``MIN_CUT_TEST_BLOCK_COVER`` of it and both images have contrast
    there (``MIN_CUT_TEST_STD``). The answer is None when fewer than
    ``MIN_CUT_TEST_BLOCKS`` blocks count.

    ``whole_blocks`` keeps, between calls with the same ``target``, what a block of ``target``
    gives when ``moved`` covers all of it (its contrast test, its values less their mean, and
    their sum of squares): the same numbers, computed once (``_correlations``).
    """
    height, width = target.shape
    values: list[float] = []
    for row in range(CUT_TEST_GRID):
        rows = slice(row * height // CUT_TEST_GRID, (row + 1) * height // CUT_TEST_GRID)
        for column in range(CUT_TEST_GRID):
            columns = slice(column * width // CUT_TEST_GRID, (column + 1) * width // CUT_TEST_GRID)
            block = moved[rows, columns]
            covered = np.isfinite(block)
            if block.size == 0 or covered.mean() < MIN_CUT_TEST_BLOCK_COVER:  # A very flat image has empty blocks.
                continue
            first = block[covered]
            if first.std() < MIN_CUT_TEST_STD:
                continue
            whole = whole_blocks is not None and first.size == block.size
            known = whole_blocks.get((row, column)) if whole and whole_blocks is not None else None
            if known is None:
                second = target[rows, columns][covered]
                flat = second.std() < MIN_CUT_TEST_STD
                if not flat:
                    second = second - second.mean()
                known = (flat, second, None if flat else float((second * second).sum()))
                if whole and whole_blocks is not None:
                    whole_blocks[(row, column)] = known
            flat, second, second_squares = known
            if flat:
                continue
            first = first - first.mean()
            scale = math.sqrt(float((first * first).sum()) * second_squares)
            values.append(float((first * second).sum()) / scale)
    if len(values) < MIN_CUT_TEST_BLOCKS:
        return None
    return float(np.median(values))


def _test_size(size: tuple[int, int]) -> tuple[int, int]:
    """Return the size (width, height) of the image of the cut tests for frames of ``size``."""
    return CUT_TEST_WIDTH, max(1, round(CUT_TEST_WIDTH * size[1] / size[0]))


def _test_image(grey: np.ndarray, test_size: tuple[int, int]) -> np.ndarray:
    """Return a grey ``uint8`` image shrunk to ``test_size`` (width, height), as ``float32``."""
    return cv2.resize(grey, test_size, interpolation=cv2.INTER_AREA).astype(np.float32)


def _moved(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Move a test image with a 3x3 matrix in test pixels. The answer is NaN where it covers nothing."""
    size = (image.shape[1], image.shape[0])
    return cv2.warpPerspective(image, matrix, size, flags=cv2.INTER_LINEAR, borderValue=np.nan)


def _correlations(first: np.ndarray, second: np.ndarray, matrices: Sequence[np.ndarray]) -> list[float | None]:
    """Return the block correlation of ``first`` moved with each matrix onto ``second`` (None where none counts)."""
    whole_blocks: dict[tuple[int, int], Any] = {}
    return [_block_correlation(_moved(first, matrix), second, whole_blocks) for matrix in matrices]


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
    test_size = _test_size(full_size)
    to_test = _to_small_matrix(full_size, test_size)
    from_test = np.linalg.inv(to_test)
    matrices = [to_test @ np.asarray(motion, dtype=np.float64) @ from_test for motion in [np.eye(3), *motions]]
    values = _correlations(_test_image(prev_grey, test_size), _test_image(next_grey, test_size), matrices)
    return max((value for value in values if value is not None), default=None)


def is_cut(prev: np.ndarray, nxt: np.ndarray, motions: Sequence[np.ndarray]) -> bool:
    """Tell whether a hard cut lies between two neighbouring frames.

    It does when no camera motion moves ``prev`` onto ``nxt`` with a
    correlation of at least ``MIN_SHOT_CORRELATION`` (``shot_correlation``),
    and also when the correlation is not defined: nothing then links the two
    frames. A pan that is too fast to follow therefore counts as a cut.
    """
    correlation = shot_correlation(prev, nxt, motions)
    return correlation is None or correlation < MIN_SHOT_CORRELATION


def _chain(steps: Sequence[np.ndarray | None], src: int, dst: int) -> np.ndarray | None:
    """Return the product of the steps from frame ``src`` to frame ``dst``, inverted when ``dst`` is earlier.

    ``steps[i]`` maps frame ``i`` to frame ``i + 1``. The answer is None when a step between the two frames is missing.
    """
    first, last = min(src, dst), max(src, dst)
    matrix = np.eye(3)
    for step in steps[first:last]:
        if step is None:
            return None
        matrix = step @ matrix
    return matrix if src <= dst else np.linalg.inv(matrix)


def window_correlations(images: Sequence[np.ndarray], steps: Sequence[np.ndarray | None]) -> list[float | None]:
    """Return the dissolve-test value of every frame: how well the camera motion explains a whole window.

    ``images`` are the test images of the clip (``CUT_TEST_WIDTH`` pixels wide, ``float32``).
    ``steps[i]`` maps test pixels of frame ``i`` to frame ``i + 1``; it is None where no
    motion was measured for the pair, and into a hard cut. The value of frame ``t`` is the median block
    correlation (``_block_correlation``) of frame ``t - DISSOLVE_WINDOW_FRAMES``, moved with the
    chained steps of the window, with frame ``t``. Only the chained motion is tried, not the
    identity: a pan that the chain follows explains its window, a dissolve does not, whatever
    the motion. The value is None, so the window is not judged, for the first
    ``DISSOLVE_WINDOW_FRAMES`` frames, when a step of the window is missing, and when the moved
    frame covers too little of the last one (a fast pan leaves the picture within the window).
    """
    if len(steps) != len(images) - 1:
        raise ValueError(f"{len(images)} images need {len(images) - 1} steps, got {len(steps)}")
    values: list[float | None] = [None] * len(images)
    for frame in range(DISSOLVE_WINDOW_FRAMES, len(images)):
        matrix = _chain(steps, frame - DISSOLVE_WINDOW_FRAMES, frame)
        if matrix is not None:
            values[frame] = _block_correlation(_moved(images[frame - DISSOLVE_WINDOW_FRAMES], matrix), images[frame])
    return values


def _shot_share(anchor: np.ndarray, image: np.ndarray, matrix: np.ndarray | None) -> float | None:
    """Return how much of the shot of an anchor image an image shows.

    The answer is the regression slope of the image on the moved anchor.

    ``matrix`` maps test pixels of the anchor to test pixels of the image. The slope is taken
    over the pixels that the moved anchor covers. The answer is None when the chain is broken,
    when the moved anchor covers less than ``MIN_CUT_TEST_BLOCK_COVER`` of the image, and when
    the moved anchor has no contrast there.
    """
    if matrix is None:
        return None
    moved = _moved(anchor, matrix)
    covered = np.isfinite(moved)
    if covered.mean() < MIN_CUT_TEST_BLOCK_COVER:
        return None
    first = moved[covered] - moved[covered].mean()
    if first.std() < MIN_CUT_TEST_STD:
        return None
    second = image[covered] - image[covered].mean()
    return float((first * second).sum() / (first * first).sum())


def _shows_shot(anchor: np.ndarray, image: np.ndarray, matrix: np.ndarray | None) -> bool:
    """Tell whether an image still shows the shot of an anchor image: ``_shot_share`` of ``MIN_SHOT_SHARE`` or more."""
    share = _shot_share(anchor, image, matrix)
    return share is not None and share >= MIN_SHOT_SHARE


def find_transitions(
    images: Sequence[np.ndarray], steps: Sequence[np.ndarray | None], values: Sequence[float | None]
) -> list[tuple[int, int]]:
    """Find the transitions between two shots that the test of neighbouring frames does not see: dissolves.

    ``images`` and ``steps`` are as for ``window_correlations``, and ``values`` is its answer.
    A transition lies in every run of frames whose window value is below
    ``MAX_TRANSITION_CORRELATION``: no camera motion moves the first frame of such a window onto
    its last. For a run from frame ``t1`` to frame ``t2``, frame ``a = t1 - DISSOLVE_WINDOW_FRAMES``
    shows the old shot alone and frame ``z = t2`` the new shot alone, when the dissolve is
    shorter than the window. The transition starts at the first frame after ``a`` that ``a``,
    moved with the camera motion, no longer explains (``_shows_shot``), and ends after the last
    frame before ``z`` that ``z`` does not explain. The motion inside a dissolve follows the
    shot that dominates the blend: the old shot in the first half, which ``a`` tests, the new
    shot in the second half, which ``z`` tests.

    Each answer is ``(start, stop)``: frames ``start`` to ``stop - 1`` are blends of the two
    shots, and frame ``stop`` starts the new shot. ``start == stop`` is a change without a
    blend frame: a hard cut at ``start``. When the two tests cross (a frame that both anchors
    explain), the frames between them are the transition; when an anchor explains every frame
    up to the other anchor, all frames between the anchors are. Transitions that touch are merged.
    """
    if len(values) != len(images):
        raise ValueError(f"{len(images)} images need {len(images)} window values, got {len(values)}")
    runs: list[tuple[int, int]] = []
    for frame, value in enumerate(values):
        if value is None or value >= MAX_TRANSITION_CORRELATION:
            continue
        if runs and runs[-1][1] == frame - 1:
            runs[-1] = (runs[-1][0], frame)
        else:
            runs.append((frame, frame))
    transitions: list[tuple[int, int]] = []
    for first, last in runs:
        _add_transition(transitions, transition_span(images, steps, first, last))
    return transitions


def transition_span(
    images: Sequence[np.ndarray | None], steps: Sequence[np.ndarray | None], first: int, last: int
) -> tuple[int, int]:
    """Return the transition ``(start, stop)`` of one run of low window values, frames ``first`` to ``last``.

    See ``find_transitions``, which calls this for each run. The images of the frames from
    ``first - DISSOLVE_WINDOW_FRAMES`` to ``last`` are used, no other image: the streaming path
    keeps only these (``ShotMeter``).
    """
    old, new = first - DISSOLVE_WINDOW_FRAMES, last
    start = next(
        (
            frame
            for frame in range(old + 1, new + 1)
            if not _shows_shot(_image(images, old), _image(images, frame), _chain(steps, old, frame))
        ),
        new,
    )
    end = next(
        (
            frame
            for frame in range(new - 1, old - 1, -1)
            if not _shows_shot(_image(images, new), _image(images, frame), _chain(steps, new, frame))
        ),
        old,
    )
    return min(start, end + 1), max(start, end + 1)


def _image(images: Sequence[np.ndarray | None], frame: int) -> np.ndarray:
    """Return a test image that must still be kept. A forgotten image raises ``ValueError``."""
    image = images[frame]
    if image is None:
        raise ValueError(f"the test image of frame {frame} was forgotten, but a transition test needs it")
    return image


def _add_transition(transitions: list[tuple[int, int]], span: tuple[int, int]) -> None:
    """Append a transition, or merge it into the last one when the two touch."""
    if transitions and span[0] <= transitions[-1][1]:
        transitions[-1] = (transitions[-1][0], max(transitions[-1][1], span[1]))
    else:
        transitions.append(span)


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

    Every frame of a dissolve starts a shot of its own (``find_transitions``). So a
    shot of one frame next to a cut is a transition frame (``transition_frames``).
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

    def transition_frames(self) -> frozenset[int]:
        """Return the frames that form a shot of one frame: a cut starts them, and a cut or the end of the clip follows.

        These are the frames of a dissolve, each a blend of two shots, and a lone frame
        between two hard cuts. They carry nothing: no people, no ball and no pitch
        mapping (``pipeline.run_clip``). A clip without a cut has none, also a clip of one frame.
        """
        if not self.cuts:
            return frozenset()
        starts = (0, *self.cuts)
        ends = (*self.cuts, self.n_frames)
        return frozenset(start for start, end in zip(starts, ends, strict=True) if end - start == 1)

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


@dataclass(frozen=True)
class ShotTests:
    """What the cut tests measured on a clip (``measure_shots``).

    ``steps[i]`` is the measured camera motion from frame ``i`` to frame ``i + 1`` in
    full-size pixels; it is None where it could not be measured and into a hard cut.
    ``test_images`` are the frames in the pixels of the cut tests (``CUT_TEST_WIDTH`` pixels
    wide), and ``test_steps[i]`` is the motion from frame ``i`` to frame ``i + 1`` in these
    pixels: the measured motion when it passes the pair test alone, else the motion of the
    pair test that explains the pair best (the identity or the last measured motion); None
    into a hard cut and where no motion was measured, except for a lone such pair between two
    pairs with a step, which takes the last measured motion when that explains it alone. They are the input of ``window_correlations`` and
    ``find_transitions``. ``pair_correlations[i]`` is the value of the test of frames ``i``
    and ``i + 1`` (``shot_correlation``), and ``hard_cuts`` holds the frames where it is below
    ``MIN_SHOT_CORRELATION`` or None. ``window_correlations[t]`` is the value of the window
    that ends at frame ``t`` (``window_correlations``), and ``transitions`` are the
    transitions it finds (``find_transitions``).
    """

    steps: tuple[np.ndarray | None, ...]
    test_images: tuple[np.ndarray, ...]
    test_steps: tuple[np.ndarray | None, ...]
    pair_correlations: tuple[float | None, ...]
    hard_cuts: tuple[int, ...]
    window_correlations: tuple[float | None, ...]
    transitions: tuple[tuple[int, int], ...]

    def cuts(self) -> tuple[int, ...]:
        """Return the frames that start a new shot: the hard cuts, every frame of a transition, the frame after it."""
        frames = set(self.hard_cuts)
        for start, stop in self.transitions:
            frames.update(range(start, stop + 1))
        return tuple(sorted(frames))


def measure_shots(
    source: FrameSource,
    *,
    people: Mapping[int, Sequence[Box]] | None = None,
    downscale: float = 0.5,
) -> ShotTests:
    """Measure the camera motion between all neighbouring frames of a clip, and test for cuts and dissolves.

    The frames are read in order. The work is done on grey images scaled by
    ``downscale`` (for speed), and every matrix is then converted back to
    full-size pixels. ``people`` maps a frame index to the boxes of the people
    in that frame, in full-size pixels. For a pair of frames, the boxes of both
    frames are grown by 10% and no corner is taken inside them. A pair whose
    motion cannot be measured gives a missing step.

    Each pair is tested for a hard cut with the identity, the measured motion and the last
    measured motion of the same shot, as ``is_cut`` does (the test is repeated here, so that
    the test images and the values are kept). A hard cut gives a missing step, also when a
    motion was measured: static broadcast graphics can give a motion across a cut. Then the
    windows of the clip are tested for a dissolve (``window_correlations``,
    ``find_transitions``). Their motion chain takes the measured motion where it explains its
    pair alone; in a fast pan the optical flow can be wrong by tens of pixels while the last
    motion explains the pair, and that motion is taken. A pair without a measured motion has
    no step in the chain, so the windows over it are not judged, unless it is a lone such pair
    between two pairs with a step and the last measured motion explains it alone. A clip of
    ``DISSOLVE_WINDOW_FRAMES`` frames or fewer has no window, and an info line says so.
    """
    info = source.info
    meter = ShotMeter((info.width, info.height), downscale=downscale)
    boxes: Mapping[int, Sequence[Box]] = {} if people is None else people
    for frame in range(info.n_frames):
        meter.add(source.read(frame), boxes.get(frame, ()))
    if info.n_frames <= DISSOLVE_WINDOW_FRAMES:
        logger.info(
            "clip %s: %d frames, not more than the window of %d frames: no dissolve test",
            info.clip_id,
            info.n_frames,
            DISSOLVE_WINDOW_FRAMES,
        )
    tests = meter.finish()
    for start, stop in tests.transitions:
        if start == stop:
            logger.info("clip %s: a window test finds a cut without blend frames at frame %d", info.clip_id, start)
        else:
            logger.info(
                "clip %s: frames %d to %d blend two shots, frame %d starts the next",
                info.clip_id,
                start,
                stop - 1,
                stop,
            )
    return tests


@dataclass(frozen=True)
class PairMeasure:
    """What the pair meter measured for one frame (``PairMeter``), before ``ShotMeter.take`` tests and books it.

    ``kind`` is ``first`` (frame 0: no pair), ``pair`` (the frame and the frame before) or
    ``repeat`` (a frame without a picture, ``PairMeter.repeat``). For a pair, ``step`` is the
    measured motion in the pixels of the small images (None where it was not measured), and
    ``previous_test`` the test image of the frame before.
    """

    frame: int
    kind: str
    test_image: np.ndarray
    step: np.ndarray | None = None
    previous_test: np.ndarray | None = None


class PairMeter:
    """The part of ``ShotMeter.add`` that needs the pictures: the small images and the camera motion of each pair.

    ``measure`` takes the picture and the people boxes of the next frame: it makes the small
    grey image and the test image, and measures the camera motion from the frame before
    (``estimate_motion``, most of the time of a frame, in OpenCV). It holds only what the next
    pair needs (the small picture and the boxes of the frame before), so it can run on another
    thread than the ``ShotMeter`` that tests and books its answers (``ShotMeter.take``), as long
    as one thread calls it, in frame order. ``repeat`` stands for a frame without a picture (a
    frame that the streaming path dropped): it shows the frame before, so the step into it is
    the identity and the next pair is measured against the frame before it.
    """

    def __init__(self, size: tuple[int, int], *, downscale: float = 0.5) -> None:
        if not 0.0 < downscale <= 1.0:
            raise ValueError("downscale must be above 0 and at most 1")
        width, height = size
        self.full_size = (width, height)
        self.small_size = (max(1, round(width * downscale)), max(1, round(height * downscale)))
        self.to_small = _to_small_matrix(self.full_size, self.small_size)
        self.test_size = _test_size(self.small_size)
        self._previous: np.ndarray | None = None
        self._previous_test: np.ndarray | None = None
        self._previous_boxes: Sequence[Box] = ()
        self.n_frames = 0

    def measure(self, image: np.ndarray, boxes: Sequence[Box] = ()) -> PairMeasure:
        """Take the picture and the people boxes (full-size pixels) of the next frame, and measure the pair it ends."""
        current = _small_grey(image, self.full_size, self.small_size)
        test_image = _test_image(current, self.test_size)
        frame = self.n_frames
        self.n_frames += 1
        previous, previous_test = self._previous, self._previous_test
        self._previous, self._previous_test = current, test_image
        if previous is None or previous_test is None:
            self._previous_boxes = boxes
            return PairMeasure(frame=frame, kind="first", test_image=test_image)
        mask = _people_mask(self.small_size, [*self._previous_boxes, *boxes], self.to_small)
        self._previous_boxes = boxes
        step = estimate_motion(previous, current, mask=mask)
        return PairMeasure(frame=frame, kind="pair", test_image=test_image, step=step, previous_test=previous_test)

    def repeat(self) -> PairMeasure:
        """Stand for a frame without a picture: it shows the frame before (class docstring)."""
        if self._previous_test is None:
            raise ValueError("the first frame needs a picture: no frame before it can stand for it")
        frame = self.n_frames
        self.n_frames += 1
        return PairMeasure(frame=frame, kind="repeat", test_image=self._previous_test)


class ShotMeter:
    """The tests of ``measure_shots``, one frame at a time: ``add`` each frame in order, then ``finish``.

    ``finish`` returns the ``ShotTests`` that ``measure_shots`` gives for the same frames and
    boxes. The streaming path reads what is known before the end:

    - after ``add`` of frame ``t``: ``steps[t - 1]`` (None into a hard cut) and ``hard_cuts``;
    - ``update_transitions`` judges every window whose value can no longer change and returns
      the transitions that are known now. A window that ends at frame ``t`` is judged once
      frame ``t + 1`` is in (a lone pair without a measured motion takes the last motion only
      when the next pair has a step), and a transition once its run of low windows has ended.
      So a dissolve that starts at frame ``s`` is known at about frame ``s + DISSOLVE_WINDOW_FRAMES``.

    ``forget_images_before`` drops the test images that no later test can need, so a long
    stream keeps a bounded number of images.
    """

    def __init__(self, size: tuple[int, int], *, downscale: float = 0.5) -> None:
        self._pairs: PairMeter | None = PairMeter(size, downscale=downscale)
        self._to_full = np.linalg.inv(self._pairs.to_small)
        self._to_small = self._pairs.to_small
        self._test_size = self._pairs.test_size
        self._to_test = _to_small_matrix(self._pairs.small_size, self._test_size)
        self._from_test = np.linalg.inv(self._to_test)
        self._last_step: np.ndarray | None = None
        #: Measured motion of each pair in full-size pixels, None where missing and into a hard cut.
        self.steps: list[np.ndarray | None] = []
        #: The steps of the window test (see ``ShotTests``); a bridge is put in by ``_bridge``.
        self.test_steps: list[np.ndarray | None] = []
        self.pair_values: list[float | None] = []
        self.hard_cuts: list[int] = []
        #: Test image of every frame; a forgotten image is None.
        self.images: list[np.ndarray | None] = []
        #: The last motion of each pair without a measured motion that the last motion explains alone.
        self._bridges: dict[int, np.ndarray] = {}
        self._bridged: set[int] = set()
        #: Window value of every judged frame (``window_correlations``), in frame order.
        self.window_values: list[float | None] = []
        #: The open run of low window values, (first, last), or None.
        self._run: tuple[int, int] | None = None
        self.transitions: list[tuple[int, int]] = []
        #: The open run as last tested by ``open_transition``: (first, last, span, times in a row), and whether it is taken.
        self._open: tuple[int, int, tuple[int, int], int] | None = None
        self._open_taken = False
        self._finished = False

    @property
    def n_frames(self) -> int:
        """Number of frames added so far."""
        return len(self.images)

    def add(self, image: np.ndarray, boxes: Sequence[Box] = ()) -> None:
        """Add the next frame and the people boxes of that frame (full-size pixels), and test the pair it ends."""
        if self._finished:
            raise ValueError("the meter is finished: no frame can be added")
        if self._pairs is None:
            raise ValueError("the pair tests of this meter were handed out: give their answers to take")
        self.take(self._pairs.measure(image, boxes))

    def hand_out_pairs(self) -> PairMeter:
        """Hand the pair tests to the caller (``PairMeter``), to run on another thread; ``add`` refuses from then on.

        Only a meter without a frame hands them out, and only once.
        """
        if self._pairs is None or self.images:
            raise ValueError("only a meter without a frame hands out its pair tests, and only once")
        pairs, self._pairs = self._pairs, None
        return pairs

    def take(self, measure: PairMeasure) -> None:
        """Test and book the pair of the next frame (``PairMeter.measure`` or ``repeat``): what ``add`` does next."""
        if self._finished:
            raise ValueError("the meter is finished: no frame can be added")
        frame = len(self.images)
        if measure.frame != frame:
            raise ValueError(f"the pair test of frame {measure.frame} came, frame {frame} was expected")
        if (measure.kind == "first") != (frame == 0):
            raise ValueError(f"frame {frame} got a pair test of the kind {measure.kind}")
        self.images.append(measure.test_image)
        if measure.kind == "first":
            return
        if measure.kind == "repeat":
            # A frame without a picture shows the frame before: the identity, and no test of the pair.
            self.pair_values.append(None)
            self.steps.append(np.eye(3))
            self.test_steps.append(np.eye(3))
        else:
            self._take_pair(frame, measure)
        # The pair before this one is final now: its bridge needs to know whether this pair has a step.
        self._bridge(len(self.test_steps) - 2)

    def _take_pair(self, frame: int, measure: PairMeasure) -> None:
        """Test the pair (the identity, the measured and the last motion) and book it: the test of ``add``."""
        step, to_test, from_test = measure.step, self._to_test, self._from_test
        assert measure.previous_test is not None
        motions = [motion for motion in (step, self._last_step) if motion is not None]
        matrices = [to_test @ motion @ from_test for motion in [np.eye(3), *motions]]
        values = _correlations(measure.previous_test, measure.test_image, matrices)
        known = [(value, matrix) for value, matrix in zip(values, matrices, strict=True) if value is not None]
        best = max(known, key=lambda pair: pair[0], default=None)  # the first of equal values
        self.pair_values.append(None if best is None else best[0])
        if best is None or best[0] < MIN_SHOT_CORRELATION:
            self.hard_cuts.append(frame)
            self.steps.append(None)
            self.test_steps.append(None)
            self._last_step = None
        elif step is None:
            # No motion was measured: the windows over this pair are not judged. The identity or the
            # last motion explains the pair well enough for the pair test, but chained over a window
            # in a pan it would make up a dissolve. A lone such pair may be bridged (``_bridge``).
            self.steps.append(None)
            self.test_steps.append(None)
            if self._last_step is not None and values[1] is not None and values[1] >= MIN_SHOT_CORRELATION:
                self._bridges[len(self.test_steps) - 1] = matrices[1]
        else:
            self.steps.append(self._to_full @ step @ self._to_small)
            # The measured motion, when it explains the pair alone, else the motion that explains it best.
            measured_explains = values[1] is not None and values[1] >= MIN_SHOT_CORRELATION
            self.test_steps.append(matrices[1] if measured_explains else best[1])
            self._last_step = step

    def _bridge(self, index: int) -> None:
        """Give a lone pair without a measured motion between two pairs with a step the last motion.

        One step that is a little wrong does not make up a dissolve, and one failed pair inside a
        dissolve does not hide it (second read of 2026-11-22). Two or more in a row stay missing.
        The answer of the test of a pair is taken before any bridge, as in ``measure_shots``.
        """
        matrix = self._bridges.pop(index, None)
        if matrix is None or not 0 < index < len(self.test_steps) - 1:
            return
        if self._missing(index - 1) or self._missing(index + 1):
            return
        self.test_steps[index] = matrix
        self._bridged.add(index)

    def _missing(self, index: int) -> bool:
        """Tell whether a pair had no test step before any bridge: a bridged pair counts as missing."""
        return self.test_steps[index] is None or index in self._bridged

    def update_transitions(self, *, final: bool = False) -> list[tuple[int, int]]:
        """Judge the windows whose value is final now, and return every transition known so far.

        With ``final`` (the end of the clip) the last window and the open run are judged too.
        """
        n_frames = len(self.images)
        last_final = n_frames - 1 if final else n_frames - 2
        for frame in range(len(self.window_values), last_final + 1):
            value: float | None = None
            if frame >= DISSOLVE_WINDOW_FRAMES:
                matrix = _chain(self.test_steps, frame - DISSOLVE_WINDOW_FRAMES, frame)
                if matrix is not None:
                    moved = _moved(_image(self.images, frame - DISSOLVE_WINDOW_FRAMES), matrix)
                    value = _block_correlation(moved, _image(self.images, frame))
            self.window_values.append(value)
            low = value is not None and value < MAX_TRANSITION_CORRELATION
            if low:
                self._run = (frame, frame) if self._run is None else (self._run[0], frame)
            elif self._run is not None:
                self._close_run()
        if final and self._run is not None:
            self._close_run()
        return list(self.transitions)

    @property
    def open_run(self) -> tuple[int, int] | None:
        """The open run of low window values, (first, last) judged frame, or None."""
        return self._run

    def open_transition(self, stable_windows: int) -> tuple[int, int] | None:
        """Return the transition of the open run of low windows when it is known before the run ends, else None.

        For the streaming path only: ``measure_shots`` and ``finish`` never call it, so the clip
        path is unchanged. A transition is final when its run of low windows has ended
        (``update_transitions``), about ``DISSOLVE_WINDOW_FRAMES`` frames after the dissolve
        starts. While the run is open, ``transition_span`` is taken with the newest judged frame
        of the run as the anchor of the new shot. While that anchor is still a blend, the end of
        the span moves with it; once the anchor shows the new shot alone, the span stays the same.
        The span is taken when it came out the same for ``stable_windows`` judged windows in a
        row and its new shot starts at or before the anchor; from then on it is returned until
        the run ends, and then the final span of ``update_transitions`` replaces it. Call it
        after ``update_transitions``, once per added frame.
        """
        if stable_windows < 1:
            raise ValueError(f"stable_windows must be at least 1, got {stable_windows}")
        if self._run is None:
            self._open, self._open_taken = None, False
            return None
        first, last = self._run
        known = self._open
        if known is None or known[0] != first:
            known, self._open_taken = None, False
        if known is None or known[1] != last:
            if self._open_taken and known is not None:
                self._open = (first, last, known[2], known[3])
            else:
                span = transition_span(self.images, self.test_steps, first, last)
                count = known[3] + 1 if known is not None and known[2] == span else 1
                self._open = (first, last, span, count)
        _, last, span, count = self._open
        if not self._open_taken and count >= stable_windows and span[1] <= last:
            self._open_taken = True
        return span if self._open_taken else None

    def _close_run(self) -> None:
        assert self._run is not None
        first, last = self._run
        self._run = None
        _add_transition(self.transitions, transition_span(self.images, self.test_steps, first, last))

    def keep_images_from(self) -> int:
        """Return the first frame whose test image a later test can still need."""
        newest = len(self.images) - 1
        first = min(newest - 1, len(self.window_values)) - DISSOLVE_WINDOW_FRAMES
        if self._run is not None:
            first = min(first, self._run[0] - DISSOLVE_WINDOW_FRAMES)
        return max(0, first)

    def forget_images_before(self, frame: int) -> None:
        """Drop the test images of the frames before ``frame`` (at most up to ``keep_images_from``)."""
        for index in range(min(frame, self.keep_images_from())):
            self.images[index] = None

    def finish(self) -> ShotTests:
        """Judge the last windows and return the tests of the whole clip. No frame can be added afterwards."""
        if not self.images:
            raise ValueError("the meter has no frame")
        self._bridge(len(self.test_steps) - 1)
        transitions = self.update_transitions(final=True)
        self._finished = True
        return ShotTests(
            steps=tuple(self.steps),
            test_images=tuple(self.images),
            test_steps=tuple(self.test_steps),
            pair_correlations=tuple(self.pair_values),
            hard_cuts=tuple(self.hard_cuts),
            window_correlations=tuple(self.window_values),
            transitions=tuple(transitions),
        )


def compute_motion_field(
    source: FrameSource,
    *,
    people: Mapping[int, Sequence[Box]] | None = None,
    downscale: float = 0.5,
) -> MotionField:
    """Measure the camera motion of a clip (``measure_shots``) and return it as a motion field.

    Every frame that starts a new shot is a cut of the field (``ShotTests.cuts``): a hard
    cut, every frame of a dissolve and the frame after it. The step into a cut is missing.
    """
    tests = measure_shots(source, people=people, downscale=downscale)
    cuts = tests.cuts()
    for frame in cuts:
        logger.info("clip %s: frame %d starts a new shot", source.info.clip_id, frame)
    starts = set(cuts)
    return MotionField([None if index + 1 in starts else step for index, step in enumerate(tests.steps)], cuts)


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
