"""Adapter for the pitch mappings of the PnLCalib calibration tool.

PnLCalib estimates the camera of a broadcast frame from the pitch markings. It is the fallback
of the pitch mapping. The tool is not imported here. It runs as its own process on the run
machine, and the script ``tools/run_tool.py`` writes its result as one JSON payload per frame
(``fallbacks.runs`` describes the file that holds them). This module turns such a payload into
a ``HomographyRecord``.

Payload format (one frame)::

    {
      "homography_pitch_to_image": [[h11, h12, h13], [h21, h22, h23], [h31, h32, h33]],
      "pitch_origin": "centre"
    }

- ``homography_pitch_to_image``: the 3x3 matrix that maps a point on the grass to the image,
  ``(u * w, v * w, w) = H @ (x, y, 1)``. ``(x, y)`` is in pitch metres. ``(u, v)`` is in pixels
  of the full frame: the origin is the top-left corner of the image, u grows to the right and v
  grows down. The scale of the matrix is free. The value is ``null`` when the tool found no
  mapping for the frame.
- ``pitch_origin``: must be ``"centre"``. It states the pitch convention of this package: the
  origin is the centre mark, x grows towards the right goal and y grows towards the main camera.
  So the far touchline is y = -34 and the near touchline is y = +34, on a pitch of 105 x 68 m.
  Only a payload without a matrix may leave this key out.
- Other keys are ignored. The script also writes keys that this adapter does not read:
  ``rep_err`` (the reprojection error of the keypoints of the tool, pixels), ``mode``,
  ``use_ransac`` and ``calib_plane`` (the setting and the plane of the camera that the tool
  took), ``camera_position_m`` (the place of the camera, pitch metres) and ``fault`` (why a
  frame has no matrix).

The script converts the result of the tool into this convention. For a 3x4 projection matrix
``P`` whose world frame has these x and y axes and the grass at height 0, the homography is
made of the columns 1, 2 and 4 of ``P``. The world frame of PnLCalib has these axes, so the
script changes no axis. A homography has no lens distortion, and this tool estimates none. The
matrix is in pixels of the full frame, also when the tool worked on a smaller image.

The record holds the inverse matrix, image to pitch, like every homography of this package.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from pitchvlm.fallbacks import payload_field, payload_number
from pitchvlm.geometry import invert_homography, matrix_to_rows
from pitchvlm.schemas import HomographyRecord

#: The only pitch convention that a payload may state: the origin is the centre mark.
PITCH_ORIGIN = "centre"
_MATRIX_KEY = "homography_pitch_to_image"
_ORIGIN_KEY = "pitch_origin"
_MATRIX_SIDE = 3


def homography_from_payload(payload: dict[str, Any], *, frame: int) -> HomographyRecord:
    """Turn the calibration payload of one frame into the pitch mapping of that frame.

    The matrix of the payload maps pitch metres to image pixels. The record holds its inverse,
    which maps image pixels to pitch metres, with ``source="landmarks"``. The matrix of the
    record is scaled to a Frobenius norm of 1, like the landmark fits of ``geometry``. The scale
    of the payload matrix is free, also a very small or a very large one. The record has no
    inlier count and no reprojection error: the adapter does not read ``rep_err``, which is the
    error of the keypoints of the tool and not of a landmark fit.

    A payload whose matrix is null gives a record with ``source="none"``.

    ``ValueError`` is raised for a singular matrix, for a ``pitch_origin`` other than
    ``"centre"``, and for a payload that does not follow the format of the module docstring.
    """
    rows = payload_field(payload, _MATRIX_KEY, "payload")
    _check_origin(payload, has_matrix=rows is not None)
    if rows is None:
        return HomographyRecord(frame=frame)
    pitch_to_image = _matrix(rows)
    largest = float(np.abs(pitch_to_image).max())
    if largest > 0.0:
        pitch_to_image = pitch_to_image / largest  # The scale is free: the inverse then has no huge values.
    image_to_pitch = invert_homography(pitch_to_image)  # None for a matrix that is singular up to rounding.
    if image_to_pitch is None:
        raise ValueError(f"payload: '{_MATRIX_KEY}' is singular, so it has no inverse")
    scaled = image_to_pitch / np.linalg.norm(image_to_pitch)
    return HomographyRecord(frame=frame, matrix=matrix_to_rows(scaled), source="landmarks")


def _check_origin(payload: Mapping[str, Any], *, has_matrix: bool) -> None:
    """Raise ``ValueError`` unless the payload states the pitch convention of this package.

    A payload without a matrix may leave the key out, because it has nothing to convert.
    """
    origin = payload.get(_ORIGIN_KEY)
    if origin is None and not has_matrix:
        return
    if origin != PITCH_ORIGIN:
        raise ValueError(f"payload: '{_ORIGIN_KEY}' must be {PITCH_ORIGIN!r}, got {origin!r}")


def _matrix(rows: Any) -> np.ndarray:
    """Return the matrix of the payload as a 3x3 ``float64`` array.

    ``ValueError`` is raised for anything but 3 rows of 3 finite numbers.
    """
    if not _is_triple(rows) or not all(_is_triple(row) for row in rows):
        raise ValueError(f"payload: '{_MATRIX_KEY}' must be 3 rows of 3 numbers or null, got {rows!r}")
    where = f"payload: '{_MATRIX_KEY}'"
    matrix = np.array([[payload_number(value, where) for value in row] for row in rows], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{where} has values that are not finite")
    return matrix


def _is_triple(value: Any) -> bool:
    """Tell whether a payload value is a list of 3 items."""
    return isinstance(value, (list, tuple)) and len(value) == _MATRIX_SIDE
