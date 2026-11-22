"""Conversion between model coordinates and image pixels.

Model coordinates are integers from 0 to 1000, normalised to the image: 0 is
the top or the left edge and 1000 is the bottom or the right edge. A box is
``[ymin, xmin, ymax, xmax]`` and a point is ``[y, x]``, so y always comes
first. Image pixels have their origin at the top-left corner, x to the right
and y down. No other module converts between the two.

The functions at the end convert the coordinates of an answer in another convention
(``CoordConvention``): other models answer x first, in pixels or in shares of the
image side.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import Literal, get_args

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


# --- Coordinate conventions of the answers -------------------------------------------
#
# A model answers coordinates in the convention that its prompt asks for. Gemini is trained on
# ``yx_1000``, the convention of the functions above. Other models answer x first, in pixels of
# the image or in shares of the image side. The step settings name the convention of their model
# (``config.StepSettings.coords``), the prompt states it (``vlm.steps``), and the contract check
# of ``doctor`` (gate G0) proves it for each model: a wrong convention puts no answered box on a
# true box.

#: ``yx_1000``: ``[ymin, xmin, ymax, xmax]`` and ``[y, x]``, integers from 0 to 1000 (Gemini).
#: ``xy_1000``: ``[xmin, ymin, xmax, ymax]`` and ``[x, y]``, integers from 0 to 1000.
#: ``xy_unit``: ``[xmin, ymin, xmax, ymax]`` and ``[x, y]``, numbers from 0 to 1.
#: ``xy_pixel``: ``[xmin, ymin, xmax, ymax]`` and ``[x, y]``, integer pixels of the image that was sent.
CoordConvention = Literal["yx_1000", "xy_1000", "xy_unit", "xy_pixel"]
COORD_CONVENTIONS: tuple[CoordConvention, ...] = get_args(CoordConvention)
#: The convention of the functions ``box_from_model`` to ``point_to_model``, and of every Gemini request.
GEMINI_CONVENTION: CoordConvention = "yx_1000"
#: Decimals of a coordinate of the ``xy_unit`` convention that the fake backend writes.
UNIT_DECIMALS = 4


def _check_convention(convention: str) -> None:
    if convention not in COORD_CONVENTIONS:
        raise ValueError(f"unknown coordinate convention {convention!r}: use one of {', '.join(COORD_CONVENTIONS)}")


def _number(value: object) -> float:
    """Return a coordinate of an answer as a float. A value that is not a finite number raises ValueError."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"a coordinate must be a finite number, got {value!r}")
    return float(value)


def _model_scale(convention: CoordConvention, size: int) -> float:
    """Return the factor that turns one coordinate of the convention into a model coordinate (0-1000)."""
    if convention == "xy_unit":
        return float(MODEL_MAX)
    if convention == "xy_pixel":
        return MODEL_MAX / size
    return 1.0


def _xy_to_model(x: object, y: object, width: int, height: int, convention: CoordConvention) -> tuple[float, float]:
    """Turn one answered x and y of the convention into model coordinates ``(y, x)`` of 0-1000, not yet clamped."""
    return (_number(y) * _model_scale(convention, height), _number(x) * _model_scale(convention, width))


def box_from_answer(values: Sequence[float], width: int, height: int, convention: CoordConvention) -> Box | None:
    """Convert an answered box of a convention to a box in image pixels.

    ``width`` and ``height`` are the size of the image that was sent. Values outside the
    image are moved to the nearest edge first. The result is None when the box is degenerate,
    as with ``box_from_model``. A wrong number of values, a value that is not a finite number
    and an unknown convention raise ValueError.
    """
    _check_convention(convention)
    _check_size(width, height)
    if len(values) != 4:
        raise ValueError(f"an answered box needs 4 values, got {len(values)}")
    if convention == GEMINI_CONVENTION:
        return box_from_model(values, width, height)
    xmin, ymin, xmax, ymax = values
    top, left = _xy_to_model(xmin, ymin, width, height, convention)
    bottom, right = _xy_to_model(xmax, ymax, width, height, convention)
    return box_from_model([top, left, bottom, right], width, height)


def point_from_answer(
    values: Sequence[float], width: int, height: int, convention: CoordConvention
) -> tuple[float, float]:
    """Convert an answered point of a convention to image pixels ``(x_px, y_px)``.

    ``width`` and ``height`` are the size of the image that was sent. Values outside the
    image are moved to the nearest edge first. A wrong number of values, a value that is not a
    finite number and an unknown convention raise ValueError.
    """
    _check_convention(convention)
    _check_size(width, height)
    if len(values) != 2:
        raise ValueError(f"an answered point needs 2 values, got {len(values)}")
    if convention == GEMINI_CONVENTION:
        return point_from_model(values, width, height)
    x, y = values
    return point_from_model(list(_xy_to_model(x, y, width, height, convention)), width, height)


def _from_model(value: int, size: int, convention: CoordConvention) -> float | int:
    """Write one model coordinate (0-1000) in a convention: an integer, or a share of the side with a few decimals."""
    if convention == "xy_unit":
        return round(value / MODEL_MAX, UNIT_DECIMALS)
    if convention == "xy_pixel":
        return round(value / MODEL_MAX * size)
    return value


def box_to_answer(box: Box, width: int, height: int, convention: CoordConvention) -> list[float | int]:
    """Write a box in image pixels as a model of the convention answers it. The fake backend uses this."""
    _check_convention(convention)
    ymin, xmin, ymax, xmax = box_to_model(box, width, height)
    if convention == GEMINI_CONVENTION:
        return [ymin, xmin, ymax, xmax]
    if convention == "xy_pixel":
        return [round(box.x1), round(box.y1), round(box.x2), round(box.y2)]
    return [_from_model(value, size, convention) for value, size in ((xmin, width), (ymin, height), (xmax, width), (ymax, height))]


def point_to_answer(x: float, y: float, width: int, height: int, convention: CoordConvention) -> list[float | int]:
    """Write an image point as a model of the convention answers it. The fake backend uses this."""
    _check_convention(convention)
    model_y, model_x = point_to_model(x, y, width, height)
    if convention == GEMINI_CONVENTION:
        return [model_y, model_x]
    if convention == "xy_pixel":
        return [round(min(max(x, 0.0), width)), round(min(max(y, 0.0), height))]
    return [_from_model(model_x, width, convention), _from_model(model_y, height, convention)]
