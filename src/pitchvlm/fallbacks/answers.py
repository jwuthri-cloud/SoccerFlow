"""The answers of a clip run as the records of the pipeline engines.

A clip run (``fallbacks.runs.ClipRun``) holds the payload of a tool for every frame of one
clip. The people engine ``detector`` reads the people and the ball candidates of every frame
from the run of the detector, and the pitch engine ``calibration`` reads the camera of every
frame from the run of the calibration. These functions convert the payloads with the adapter
of each tool; ``pipeline.ToolAnswers`` holds the result.
"""

from __future__ import annotations

from pitchvlm.ball import BallCandidate
from pitchvlm.fallbacks.pnlcalib import homography_from_payload
from pitchvlm.fallbacks.rfdetr import ball_candidates, expand_rows, people_from_detections
from pitchvlm.fallbacks.runs import ClipRun, PnlcalibSettings, is_detector
from pitchvlm.schemas import FramePeople, HomographyRecord


class AnswerError(ValueError):
    """The payload of one frame of a clip run cannot be used. The message names the frame."""


def detector_answers(run: ClipRun) -> tuple[list[FramePeople], dict[int, tuple[BallCandidate, ...]]]:
    """Return the people of every frame and the ball candidates of the frames that have any.

    The detector can be of any family (``fallbacks.runs.DETECTOR_FAMILIES``): all of them give
    the payload of ``fallbacks.rfdetr``. The score limit is the one that the detector ran with
    (``min_score`` of its settings). ``ValueError`` is raised for a run of another tool, and
    ``AnswerError`` for a payload that does not follow the compact format of a clip run or the
    format of the adapter.
    """
    if not is_detector(run.tool):
        raise ValueError(f"the clip run of clip {run.clip} is a run of {run.tool.name}, not of a detector")
    people: list[FramePeople] = []
    balls: dict[int, tuple[BallCandidate, ...]] = {}
    for answer in run.frames:
        try:
            payload = expand_rows(answer.payload)
            people.append(people_from_detections(payload, frame=answer.frame, min_score=run.tool.min_score))
            found = ball_candidates(payload, frame=answer.frame, min_score=run.tool.min_score)
        except ValueError as error:
            raise AnswerError(f"frame {answer.frame}: {error}") from error
        if found:
            balls[answer.frame] = found
    return people, balls


def calibration_answers(run: ClipRun) -> list[HomographyRecord]:
    """Return the pitch mapping of every frame (``source="none"`` where the tool found no camera).

    ``ValueError`` is raised for a run of another tool, and ``AnswerError`` for a payload
    that does not follow the format of ``fallbacks.pnlcalib``.
    """
    if not isinstance(run.tool, PnlcalibSettings):
        raise ValueError(f"the clip run of clip {run.clip} is a run of {run.tool.name}, not of a calibration")
    cameras: list[HomographyRecord] = []
    for answer in run.frames:
        try:
            cameras.append(homography_from_payload(answer.payload, frame=answer.frame))
        except ValueError as error:
            raise AnswerError(f"frame {answer.frame}: {error}") from error
    return cameras
