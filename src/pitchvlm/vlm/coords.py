"""Conversion between model coordinates and image pixels.

Model coordinates are integers from 0 to 1000, normalised to the image: 0 is
the top or the left edge and 1000 is the bottom or the right edge. A box is
``[ymin, xmin, ymax, xmax]`` and a point is ``[y, x]``, so y always comes
first. Image pixels have their origin at the top-left corner, x to the right
and y down. No other module converts between the two.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

from pitchvlm.schemas import Box

#: Model coordinate of the right edge and of the bottom edge of the image.
MODEL_MAX = 1000


def _check_size(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(f"the image size must be positive, got {width} x {height}")


def _clamp(value: float) -> float:
    """Return a model coordinate as a float, moved into the range 0-1000."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"a model coordinate must be a number, got {value!r}")
    if value != value:
        raise ValueError("a model coordinate must not be NaN")
    return float(min(max(value, 0), MODEL_MAX))


def _to_pixels(value: float, size: int) -> float:
    """Convert one model coordinate to pixels along an image side of ``size`` pixels."""
    return _clamp(value) / MODEL_MAX * size


def _to_model(pixels: float, size: int) -> int:
    """Convert one pixel coordinate to the nearest model coordinate in the range 0-1000."""
    if not math.isfinite(pixels):
        raise ValueError(f"a pixel coordinate must be finite, got {pixels!r}")
    return round(_clamp(pixels / size * MODEL_MAX))


def box_from_model(box_2d: Sequence[float], width: int, height: int) -> Box | None:
    """Convert a model box ``[ymin, xmin, ymax, xmax]`` to a box in image pixels.

    Values outside 0-1000 are moved to the nearest limit first. The result is
    None when the box is degenerate: it has no width or no height after that.
    A wrong number of values or a value that is not a number raises ValueError.
    """
    _check_size(width, height)
    if len(box_2d) != 4:
        raise ValueError(f"a model box needs 4 values [ymin, xmin, ymax, xmax], got {len(box_2d)}")
    ymin, xmin, ymax, xmax = box_2d
    x1, x2 = _to_pixels(xmin, width), _to_pixels(xmax, width)
    y1, y2 = _to_pixels(ymin, height), _to_pixels(ymax, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return Box(x1=x1, y1=y1, x2=x2, y2=y2)


def point_from_model(point: Sequence[float], width: int, height: int) -> tuple[float, float]:
    """Convert a model point ``[y, x]`` to image pixels ``(x_px, y_px)``.

    Values outside 0-1000 are moved to the nearest limit first. A wrong number
    of values or a value that is not a number raises ValueError.
    """
    _check_size(width, height)
    if len(point) != 2:
        raise ValueError(f"a model point needs 2 values [y, x], got {len(point)}")
    y, x = point
    return (_to_pixels(x, width), _to_pixels(y, height))


def box_to_model(box: Box, width: int, height: int) -> list[int]:
    """Convert a box in image pixels to a model box ``[ymin, xmin, ymax, xmax]``.

    Each value is rounded to the nearest integer and limited to 0-1000.
    """
    _check_size(width, height)
    return [
        _to_model(box.y1, height),
        _to_model(box.x1, width),
        _to_model(box.y2, height),
        _to_model(box.x2, width),
    ]


def point_to_model(x: float, y: float, width: int, height: int) -> list[int]:
    """Convert an image point ``(x_px, y_px)`` to a model point ``[y, x]``.

    Each value is rounded to the nearest integer and limited to 0-1000.
    """
    _check_size(width, height)
    return [_to_model(y, height), _to_model(x, width)]
