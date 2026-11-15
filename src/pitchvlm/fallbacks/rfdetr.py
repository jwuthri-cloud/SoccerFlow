"""Adapter for the people boxes of the RF-DETR detector.

RF-DETR is an object detector. It is the fallback of the people step. The detector is not
imported here. It runs as its own process on the run machine, and the script
``tools/run_tool.py`` writes its result as one JSON payload per frame (``fallbacks.runs``
describes the file that holds them). This module turns such a payload into the ``FramePeople``
record of the people step.

Payload format (one frame)::

    {
      "detections": [
        {"box_xyxy": [x1, y1, x2, y2], "score": 0.9, "label": "person"}
      ]
    }

- ``box_xyxy``: the box in pixels of the full frame. The origin is the top-left corner of the
  image, x grows to the right and y grows down. ``(x1, y1)`` is the top-left corner of the box
  and ``(x2, y2)`` is its bottom-right corner.
- ``score``: the confidence of the detection, a number from 0 to 1.
- ``label``: the class name of the detection as text. Only ``person`` (lower case) is used.
- Other keys are ignored.

The payload holds class names, not class numbers, and boxes in pixels of the full frame, also
when the detector worked on a smaller image. The detector gives both in this form.

A detector does not know roles, teams or shirt numbers. So every person gets the role
``player``, the team key ``none`` and no number. No step of the package assigns these
attributes to detector boxes yet, and the pipeline does not read this adapter: it is used to
measure the detector (command ``tool-score``).
"""

from __future__ import annotations

import math
from typing import Any

from pitchvlm.fallbacks import payload_field, payload_number
from pitchvlm.schemas import Box, FramePeople, PersonDet

#: Label of a person in the payload. A detection with another label is not a person.
PERSON_LABEL = "person"
#: Number of values of a box: x1, y1, x2, y2.
_BOX_VALUES = 4


def people_from_detections(payload: dict[str, Any], *, frame: int, min_score: float) -> FramePeople:
    """Turn the detector payload of one frame into the people of that frame.

    A detection is kept when its label is ``person``, its score is at least ``min_score`` and
    its box is valid. A box is not valid when one of its values is not finite, or when it has no
    width or no height. Such a detection is dropped. The people keep the order of the payload.
    ``min_score`` has no default: the experiment file of a measurement names it.

    ``ValueError`` is raised for a payload that does not follow the format of the module
    docstring (a missing key, a wrong type, a box without 4 numbers, a score outside 0-1) and
    for a ``min_score`` outside 0-1.
    """
    if not 0.0 <= min_score <= 1.0:
        raise ValueError(f"min_score must be from 0 to 1, got {min_score}")
    detections = payload_field(payload, "detections", "payload")
    if not isinstance(detections, (list, tuple)):
        raise ValueError(f"payload: 'detections' must be a list, got {type(detections).__name__}")
    boxes = [
        _person_box(detection, f"detections[{position}]", min_score) for position, detection in enumerate(detections)
    ]
    people = tuple(PersonDet(box=box, role="player", team_key="none") for box in boxes if box is not None)
    return FramePeople(frame=frame, people=people)


def _person_box(detection: Any, where: str, min_score: float) -> Box | None:
    """Return the box of one detection, or None when the detection is not kept.

    The detection is read in full first. So a detection that does not follow the payload format
    raises ``ValueError`` even when its label or its score would drop it.
    """
    label = payload_field(detection, "label", where)
    if not isinstance(label, str):
        raise ValueError(f"{where}: 'label' must be text, got {label!r}")
    score = payload_number(payload_field(detection, "score", where), f"{where}: 'score'")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{where}: 'score' must be from 0 to 1, got {score}")
    values = payload_field(detection, "box_xyxy", where)
    if not isinstance(values, (list, tuple)) or len(values) != _BOX_VALUES:
        raise ValueError(f"{where}: 'box_xyxy' must be a list of 4 numbers [x1, y1, x2, y2], got {values!r}")
    x1, y1, x2, y2 = (payload_number(value, f"{where}: 'box_xyxy'") for value in values)
    if label != PERSON_LABEL or score < min_score:
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    return Box(x1=x1, y1=y1, x2=x2, y2=y2)
