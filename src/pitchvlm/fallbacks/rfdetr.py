"""Adapter for the people boxes of a detector: RF-DETR, D-FINE or RT-DETRv2.

A detector is the fallback of the people step, and the people engine ``detector`` of the
pipeline. Three families have a place in the settings (``fallbacks.runs.DETECTOR_FAMILIES``):
RF-DETR through its own package, D-FINE and RT-DETRv2 through the object-detection interface of
Hugging Face transformers. No detector is imported here. Each one runs as its own process on the
run machine, and the script ``tools/run_tool.py`` writes its result as one JSON payload per
frame (``fallbacks.runs`` describes the file that holds them). Every family gives the payload
below, so this module and the pipeline do not care which family made a run: the settings of
the run name it, and the engine table names it as ``detector:<family>:<size>``. This module
turns such a payload into the ``FramePeople`` record of the people step.

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
when the detector worked on a smaller image. RF-DETR gives both in this form. D-FINE and
RT-DETRv2 give a class score per query and a box as its centre, width and height in parts of
the model input; the script turns them into this form (``detections_of_queries`` of the script)
and keeps only the classes ``person`` and ``sports ball``.

A detector with weights for the everyday classes does not know roles, teams or shirt
numbers. So every person gets the role ``player``, the team key ``none`` and no number. The
detector engine of the pipeline gives them a team key and a role from the kit colours
(``kitcolour``), and the command ``tool-score`` measures the boxes alone.

The class ``sports ball`` gives the ball candidates of a frame (``ball_candidates``).

The run of a tool on every frame of a clip stores a compact payload (``fallbacks.runs``)::

    {"labels": ["person", "sports ball"], "rows": [[x1, y1, x2, y2, score, label_index]]}

``expand_rows`` turns it into the payload above.
"""

from __future__ import annotations

import math
from typing import Any

from pitchvlm.ball import BallCandidate
from pitchvlm.fallbacks import payload_field, payload_number
from pitchvlm.schemas import Box, FramePeople, PersonDet

#: Label of a person in the payload. A detection with another label is not a person.
PERSON_LABEL = "person"
#: Label of the ball in the payload: the class name of the everyday classes (COCO).
BALL_LABEL = "sports ball"
#: Number of values of a box: x1, y1, x2, y2.
_BOX_VALUES = 4
#: Number of values of a compact row: the box, the score and the place of the label.
_ROW_VALUES = 6


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
    kept = _kept(payload, PERSON_LABEL, min_score)
    people = tuple(PersonDet(box=box, role="player", team_key="none") for box, _ in kept)
    return FramePeople(frame=frame, people=people)


def ball_candidates(payload: dict[str, Any], *, frame: int, min_score: float) -> tuple[BallCandidate, ...]:
    """Return the ball candidates of one frame: the detections with the label ``sports ball``.

    A candidate is kept under the same rules as a person (``people_from_detections``): its
    score is at least ``min_score`` and its box is valid. The candidates keep the order of the
    payload. ``ValueError`` is raised for the same faults as there.
    """
    kept = _kept(payload, BALL_LABEL, min_score)
    return tuple(BallCandidate(frame=frame, box=box, score=score) for box, score in kept)


def expand_rows(payload: dict[str, Any]) -> dict[str, Any]:
    """Turn the compact payload of a clip run into the payload of this module.

    ``ValueError`` is raised when ``labels`` is not a list of text, when ``rows`` is not a
    list, and for a row that is not 5 numbers and a whole number that names a place in
    ``labels``. The values themselves are checked by the functions that read the result.
    """
    labels = payload_field(payload, "labels", "payload")
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ValueError(f"payload: 'labels' must be a list of class names, got {labels!r}")
    rows = payload_field(payload, "rows", "payload")
    if not isinstance(rows, list):
        raise ValueError(f"payload: 'rows' must be a list, got {type(rows).__name__}")
    detections = []
    for position, row in enumerate(rows):
        where = f"rows[{position}]"
        if not isinstance(row, list) or len(row) != _ROW_VALUES:
            raise ValueError(f"{where}: a row must be [x1, y1, x2, y2, score, label index], got {row!r}")
        index = row[-1]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(labels):
            raise ValueError(
                f"{where}: the label index must be a place in 'labels' (0 to {len(labels) - 1}), got {index!r}"
            )
        detections.append({"box_xyxy": row[:_BOX_VALUES], "score": row[_BOX_VALUES], "label": labels[index]})
    return {"detections": detections}


def _kept(payload: dict[str, Any], label: str, min_score: float) -> list[tuple[Box, float]]:
    """Return the box and the score of every detection with ``label`` that is kept, in payload order.

    Every detection of the payload is read, also those with another label.
    """
    if not 0.0 <= min_score <= 1.0:
        raise ValueError(f"min_score must be from 0 to 1, got {min_score}")
    detections = payload_field(payload, "detections", "payload")
    if not isinstance(detections, (list, tuple)):
        raise ValueError(f"payload: 'detections' must be a list, got {type(detections).__name__}")
    found = [_detection(detection, f"detections[{position}]") for position, detection in enumerate(detections)]
    return [
        (box, score)
        for found_label, score, box in found
        if found_label == label and score >= min_score and box is not None
    ]


def _detection(detection: Any, where: str) -> tuple[str, float, Box | None]:
    """Read one detection: its label, its score and its box, or None for a box that is not valid.

    The detection is read in full. So a detection that does not follow the payload format
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
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return label, score, None
    return label, score, Box(x1=x1, y1=y1, x2=x2, y2=y2)
