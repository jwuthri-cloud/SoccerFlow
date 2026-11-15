"""Adapters that turn the JSON output of external tools into the records of this package.

The external tools (the RF-DETR detector, the PnLCalib pitch calibration) are never imported
here. Each tool runs as its own process on the run machine. The script ``tools/run_tool.py``
converts the tool's own output into a payload, and the adapter module of the tool documents the
format of that payload. A payload is plain JSON data: a mapping that holds lists, numbers and
text. The module ``runs`` describes the job file that the script reads and the run file that it
writes.

A fallback replaces a model step only after that step failed its gate and the fallback passed
the same gate on the same frames. Both tools are measured on the frames of the probes for this
reason (the commands ``tool-job`` and ``tool-score``).

This file also holds the two checks that both adapters use to read a payload.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any


def payload_field(mapping: Any, key: str, where: str) -> Any:
    """Return ``mapping[key]`` of a payload.

    ``where`` names the place in the payload for the error text. ``ValueError`` is raised when
    ``mapping`` is not a mapping and when it has no such key.
    """
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {type(mapping).__name__}")
    if key not in mapping:
        raise ValueError(f"{where}: the key '{key}' is missing")
    return mapping[key]


def payload_number(value: Any, where: str) -> float:
    """Return a payload value as a float.

    ``where`` names the place in the payload for the error text. ``ValueError`` is raised for a
    value that is not a number. A truth value is not a number here, although Python counts it as
    an integer. NaN and infinity pass this check: the caller decides what they mean.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{where}: expected a number, got {value!r}")
    return float(value)
