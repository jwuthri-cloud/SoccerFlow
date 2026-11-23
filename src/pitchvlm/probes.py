"""Probes: measure one model step on labelled frames.

A probe sends the requests of one model step (people, pitch landmarks or shirt
numbers) for labelled frames of SoccerNet-GSR clips. It compares the answers
with the labels and returns the measured values as a dict of plain JSON types.
The command line writes this dict to a file with ``write_metrics``. A probe
never tunes anything: it changes no setting and it chooses nothing from the
results.

Rules that all probes share:

- Only scored frames are used. The people probe and the pitch probe take
  their frames from ``sample_frames``: evenly spread, fixed by the seed and
  the clip id.
- The label work (frame choice, true boxes, crop choice) is done for all clips
  before the first model request. So a problem in the labels stops a run
  before it costs money.
- The requests are those of the pipeline (``vlm.steps``), with the clip id of
  the frame source. They are sent one after the other.
- A value that cannot be computed (for example a recall without any true box)
  is ``nan`` in the result, as in ``evalx.metrics``. ``write_metrics`` writes
  it as ``null``.
- An answer of the people, landmarks or number step that cannot be parsed
  counts as an empty answer: no people, no landmarks, no number. The result
  counts these answers in ``n_answers_not_parsed``. A kit legend answer that
  cannot be parsed raises ``ParseError``. Errors of the backend are not caught.

The people probe can send video in place of images (with a ``VideoProbe``): the same frames, the same labels and the same
values, so an experiment of each kind can be compared on one gate (G2). Each
request then sends a short video of the target frames and of ``context_frames``
frames before and after each one (``vlm.video``), and the answer lists the
people of each target frame by its time in the video. The kit legend is asked
from a video of the same three frames as with images.

Every result holds ``probe``, ``n_clips``, ``n_frames`` or ``n_tracks``,
``settings`` (all values that define the measurement), ``prompts`` (the prompt
names of the steps), ``main_metric`` (the mean of the per-clip values with its
95% interval) and ``per_clip`` (clip id -> values of that clip).

An external tool (a detector, a calibration tool) is measured on the same
frames with the same values. ``probe_frames`` names the frames of the people
probe and the pitch probe, so that the tool can answer for them first. Then
``score_people_answers`` and ``score_pitch_mappings`` take what the tool
answered for each frame. They send no request, and their result has no
``prompts``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import operator
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

import cv2
import numpy as np
from pydantic import Field

from pitchvlm import __version__
from pitchvlm.config import PipelineConfig, StepSettings
from pitchvlm.evalx.metrics import (
    DetectionScores,
    detection_scores,
    foot_errors_m,
    match_boxes,
    number_exact_match,
    summarize,
    t_interval,
)
from pitchvlm.evalx.oracle import LABEL_RANSAC_PX, MIN_ATHLETES_FOR_FIT
from pitchvlm.frames import FrameSource
from pitchvlm.geometry import (
    PaintAlignment,
    apply_homography,
    fit_from_landmarks,
    fit_homography,
    matrix_from_record,
    paint_alignment,
)
from pitchvlm.labels import GsrAthlete, GsrLabels, gsr_frame_people
from pitchvlm.pipeline import MIN_CROP_GAP_S, crop_choice, fit_problems
from pitchvlm.pitch import inside_pitch
from pitchvlm.schemas import (
    Box,
    FrameLandmarks,
    FramePeople,
    HomographyRecord,
    KitDescription,
    KitLegend,
    Record,
    TeamKey,
)
from pitchvlm.vlm import steps
from pitchvlm.vlm.prompts import PROMPT_NAMES
from pitchvlm.vlm.types import MAX_VIDEO_FPS, VIDEO_RESOLUTIONS, Backend, VideoPart, VLMRequest
from pitchvlm.vlm.video import DEFAULT_WRITE_FPS, context_frames, frames_video_part

logger = logging.getLogger(__name__)

#: A predicted box counts as found when its overlap (IoU) with a labelled box is at least this.
IOU_THRESHOLD = 0.5
#: A frame is a goal view when the mean ``|x|`` of its labelled people is above this, metres.
GOAL_VIEW_MIN_MEAN_ABS_X_M = 20.0
#: The two views of the pitch probe.
VIEWS = ("goal", "midfield")
#: Team key of every number request of the number probe. The labels hold the side of a team
#: (left or right), not its key in the kit legend (A or B), so the team is not known.
NUMBER_TEAM_KEY: TeamKey = "none"
#: Role of the labelled people whose shirt number is probed.
PLAYER_ROLE = "player"
#: Labelled people with this role stand next to the pitch. The plausibility check does not use them.
ROLE_NEXT_TO_PITCH = "other"
#: Margin of the plausibility check around the pitch, metres: the default of the pipeline.
PITCH_MARGIN_M = PipelineConfig().pitch_margin_m
#: Confidence of the interval over the per-clip values of the main metric.
INTERVAL_CONFIDENCE = 0.95
#: Sign changes of the pitch axes that the axis check of a tool mapping tries: name -> (sign of x, sign of y).
AXIS_SIGNS = {
    "as_given": (1.0, 1.0),
    "x_flipped": (-1.0, 1.0),
    "y_flipped": (1.0, -1.0),
    "both_flipped": (-1.0, -1.0),
}
#: The paint-alignment separation compares the mappings whose median position error is at most
#: this, metres ...
PAINT_GOOD_MAX_ERROR_M = 2.0
#: ... with the mappings whose median position error is above this, metres.
PAINT_BAD_MIN_ERROR_M = 3.0
#: Keys that ``write_metrics`` adds to the payload.
PROVENANCE_KEY = "provenance"
WRITTEN_AT_KEY = "written_at"
PACKAGE_VERSION_KEY = "package_version"

#: Gives the people that an external tool found in one frame: (clip id, frame) -> people.
PeopleOfFrame = Callable[[str, int], FramePeople]
#: Gives the pitch mapping of an external tool for one frame: (clip id, frame) -> record.
MappingOfFrame = Callable[[str, int], HomographyRecord]

_Answer = TypeVar("_Answer")
_FrameAnswer = TypeVar("_FrameAnswer", FramePeople, HomographyRecord)


@dataclass(frozen=True)
class LabelledClip:
    """One clip with its labels: the frames come from ``source``, the truth comes from ``labels``.

    ``ValueError`` is raised when the two do not belong together: another clip
    id, another number of frames or another image size.
    """

    source: FrameSource
    labels: GsrLabels

    def __post_init__(self) -> None:
        info, labels = self.source.info, self.labels
        differences: list[str] = []
        if info.clip_id != labels.clip_id:
            differences.append(f"clip id {info.clip_id!r} against {labels.clip_id!r}")
        if info.n_frames != len(labels.frames):
            differences.append(f"{info.n_frames} frames against {len(labels.frames)} labelled images")
        if (info.width, info.height) != (labels.width, labels.height):
            differences.append(
                f"{info.width} x {info.height} pixels against {labels.width} x {labels.height} pixels"
            )
        if differences:
            raise ValueError("the frame source and the labels do not belong together: " + "; ".join(differences))


class VideoProbe(Record):
    """How a people probe sends video in place of images: the section ``video`` of its experiment file.

    A people probe with this section sends video; one without it sends the images of
    the pipeline. The pipeline itself sends images only: video input is measured by
    the probe first (docs/DECISIONS.md, entry of 2026-11-23).

    ``fps`` is the rate at which the frames of each video are written, frames per
    second, so the model sees every written frame. A Gemini step sends it as the
    ``fps`` of the video, and needs it. An OpenRouter step cannot send a rate (the
    video page says that only Gemini reads it), so it has none: its frames are
    written at ``vlm.video.DEFAULT_WRITE_FPS`` and the provider samples at its own
    rate. ``context_frames`` frames before and after each target frame are written
    into its video, ``round(clip fps / fps)`` frames of the clip apart
    (``vlm.video.context_frames``). ``targets_per_request`` target frames of a clip
    share one request, in frame order: their windows go into one video. The value 1
    gives one request per target frame. The media resolution of the video is the
    ``resolution`` of the step; it must be one of ``VIDEO_RESOLUTIONS``.
    """

    fps: float | None = Field(default=None, gt=0, le=MAX_VIDEO_FPS, allow_inf_nan=False)
    context_frames: int = Field(default=2, ge=0)
    targets_per_request: int = Field(default=1, ge=1)

    @property
    def write_fps(self) -> float:
        """Return the rate at which the frames are written: ``fps``, or ``DEFAULT_WRITE_FPS`` without one."""
        return self.fps if self.fps is not None else DEFAULT_WRITE_FPS


def check_video_settings(settings: StepSettings, video: VideoProbe) -> None:
    """Raise ``ValueError`` when the video settings do not fit the step settings or their provider.

    A Gemini step needs a rate (``VideoProbe.fps``); an OpenRouter step must not have
    one. The resolution of the step must be a video resolution (``VIDEO_RESOLUTIONS``).
    """
    if settings.resolution not in VIDEO_RESOLUTIONS:
        raise ValueError(
            f"a video step has no resolution {settings.resolution}: use {', '.join(VIDEO_RESOLUTIONS)}"
        )
    if settings.provider == "gemini" and video.fps is None:
        raise ValueError("a Gemini video step needs the rate fps, at which its frames are written and sampled")
    if settings.provider != "gemini" and video.fps is not None:
        raise ValueError("an OpenRouter video step cannot send a sampling rate (only Gemini reads it): remove fps")


@dataclass(frozen=True)
class _VideoRequestPlan:
    """The frames of one video request of the people probe: its target frames and all frames of its video."""

    targets: tuple[int, ...]
    video_frames: tuple[int, ...]
    video: VideoPart


@dataclass(frozen=True)
class _VideoClipPlan:
    """The videos of one clip of a video people probe: the kit legend video and the people requests."""

    legend_frames: tuple[int, ...]
    legend_video: VideoPart
    requests: tuple[_VideoRequestPlan, ...]


#: The kit legend of the requests that ``plan_people_video`` builds for the cost estimate. The real
#: legend comes from the model; the estimate counts a fixed number of text tokens for the prompt.
_ESTIMATE_LEGEND = KitLegend(
    team_a=KitDescription(shirt="unknown", shorts="unknown", socks="unknown"),
    team_b=KitDescription(shirt="unknown", shorts="unknown", socks="unknown"),
    referee_kit="unknown",
)


def _video_clip_plan(
    clip: LabelledClip, targets: Sequence[int], settings: StepSettings, video: VideoProbe, folder: Path
) -> _VideoClipPlan:
    """Write (or reuse) the videos of one clip and return their plan. No model is called here."""
    info = clip.source.info
    fps = video.write_fps

    def part(step: str, frames: Sequence[int]) -> VideoPart:
        return frames_video_part(
            clip.source,
            frames,
            folder=folder,
            step=step,
            fps=fps,
            resolution=settings.resolution,
            send_fps=video.fps is not None,
        )

    legend_frames = tuple(_legend_frames(clip.labels))
    requests: list[_VideoRequestPlan] = []
    for start in range(0, len(targets), video.targets_per_request):
        chunk = tuple(targets[start : start + video.targets_per_request])
        around = (
            context_frames(
                frame, n_frames=info.n_frames, clip_fps=info.fps, video_fps=fps, context=video.context_frames
            )
            for frame in chunk
        )
        video_frames = tuple(sorted({frame for window in around for frame in window}))
        requests.append(_VideoRequestPlan(chunk, video_frames, part("people_video", video_frames)))
    return _VideoClipPlan(legend_frames, part("kit_legend_video", legend_frames), tuple(requests))


def _check_video_probe(settings: StepSettings, video: VideoProbe | None, folder: Path | None) -> None:
    """Refuse a video probe without the folder of its videos, and video settings that do not fit the step."""
    if video is None:
        return
    if folder is None:
        raise ValueError("a people probe with video input needs the folder of its videos")
    check_video_settings(settings, video)


def plan_people_video(
    clips: Sequence[LabelledClip],
    settings: StepSettings,
    *,
    frames_per_clip: int,
    seed: int,
    video: VideoProbe,
    folder: Path,
    repeat: int = 0,
) -> list[VLMRequest]:
    """Write the videos of a video people probe and return its requests, for the cost estimate before the run.

    The requests are those of ``probe_people`` with a placeholder kit legend in the
    people prompts: the same steps, video parts and target frames, and the same kit
    legend requests, but other prompt texts, so the people requests have other cache
    keys than the sent ones. The estimate counts the video parts and a fixed number
    of text tokens, not the text, so it is the same. They are never sent. The videos are the ones that the probe
    sends: a file that is already there is used as it is.
    """
    _check_clips(clips)
    _check_video_probe(settings, video, folder)
    requests: list[VLMRequest] = []
    for clip in clips:
        info = clip.source.info
        plan = _video_clip_plan(clip, sample_frames(clip.labels, frames_per_clip, seed), settings, video, folder)
        requests.append(_legend_video_request(clip, plan, settings, video, repeat))
        requests.extend(
            _people_video_request(clip, entry, _ESTIMATE_LEGEND, settings, video, repeat) for entry in plan.requests
        )
        logger.info("video people probe: clip %s, %d people requests", info.clip_id, len(plan.requests))
    return requests


def _legend_video_request(
    clip: LabelledClip, plan: _VideoClipPlan, settings: StepSettings, video: VideoProbe, repeat: int
) -> VLMRequest:
    """Build the kit legend request of a clip from its legend video."""
    return steps.build_kit_legend_video_request(
        plan.legend_video,
        clip_id=clip.source.info.clip_id,
        video_frames=plan.legend_frames,
        fps=video.write_fps,
        settings=settings,
        repeat=repeat,
    )


def _people_video_request(
    clip: LabelledClip,
    entry: _VideoRequestPlan,
    legend: KitLegend,
    settings: StepSettings,
    video: VideoProbe,
    repeat: int,
) -> VLMRequest:
    """Build one people request of the video probe."""
    info = clip.source.info
    return steps.build_people_video_request(
        entry.video,
        legend,
        clip_id=info.clip_id,
        video_frames=entry.video_frames,
        targets=entry.targets,
        fps=video.write_fps,
        width=info.width,
        height=info.height,
        settings=settings,
        repeat=repeat,
    )


# --- Frame choice --------------------------------------------------------------------


def sample_frames(labels: GsrLabels, per_clip: int, seed: int) -> list[int]:
    """Choose ``per_clip`` scored frames of a clip, evenly spread over the clip. The result is in frame order.

    The scored frames are cut into ``per_clip`` parts of equal length, and one
    frame is taken at the same place in every part. The place comes from a
    hash of the seed and the clip id. So the choice is the same on every
    machine and with every library version, and a cached answer is found again.
    Another seed or another clip gives another place.

    A clip with at most ``per_clip`` scored frames gives all its scored frames.
    ``ValueError`` is raised for ``per_clip`` below 1 and for a clip without a
    scored frame.
    """
    _check_at_least("per_clip", per_clip, 1)
    scored = _scored_frames(labels)
    if per_clip >= len(scored):
        return scored
    shift = _seeded_hash(seed, labels.clip_id) % len(scored)
    return [scored[position] for position in _spread(len(scored), per_clip, shift)]


def probe_frames(clips: Sequence[LabelledClip], *, frames_per_clip: int, seed: int) -> dict[str, list[int]]:
    """Return the frames that the people probe and the pitch probe measure, by clip id.

    The clips keep their order, and the frames of a clip are in frame order
    (``sample_frames``). An external tool must answer for these frames before
    ``score_people_answers`` or ``score_pitch_mappings`` can measure it.
    """
    _check_clips(clips)
    _check_at_least("frames_per_clip", frames_per_clip, 1)
    return {clip.source.info.clip_id: sample_frames(clip.labels, frames_per_clip, seed) for clip in clips}


def _scored_frames(labels: GsrLabels) -> list[int]:
    """Return the indices of the scored frames, in frame order. A clip without a scored frame raises ``ValueError``."""
    scored = [frame.frame for frame in labels.frames if frame.scored]
    if not scored:
        raise ValueError(f"clip {labels.clip_id}: the labels have no scored frame")
    return scored


def _spread(n_items: int, count: int, shift: int) -> list[int]:
    """Return ``count`` positions in ``range(n_items)``: one in each of ``count`` equal parts.

    ``shift`` is from 0 to ``n_items - 1``. It moves all positions together
    inside their parts. ``count`` must not be above ``n_items``. The positions
    are then all different, and they increase.
    """
    return [(index * n_items + shift) // count for index in range(count)]


def _seeded_hash(seed: int, *names: object) -> int:
    """Return a 64-bit number that depends only on the seed and the names.

    The number comes from SHA-256 and not from a random generator, so it does
    not change with the machine or with a library version.
    """
    text = ":".join(str(part) for part in (operator.index(seed), *names))
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _legend_frames(labels: GsrLabels) -> list[int]:
    """Return the frames for the kit legend: the first, the middle and the last scored frame.

    They do not depend on the seed, so a clip always gets the same kit legend
    request. A clip with fewer than three scored frames gives each frame one
    time only.
    """
    scored = _scored_frames(labels)
    return list(dict.fromkeys((scored[0], scored[len(scored) // 2], scored[-1])))


# --- Requests and answers ------------------------------------------------------------


def _read_frame(clip: LabelledClip, frame: int) -> np.ndarray:
    """Read one frame and check that it is a BGR ``uint8`` image with the size of the clip."""
    info = clip.source.info
    image = clip.source.read(frame)
    expected = (info.height, info.width, 3)
    if image.dtype != np.uint8 or image.shape != expected:
        raise ValueError(
            f"clip {info.clip_id}, frame {frame}: got a {image.dtype} image of shape {image.shape}, "
            f"the clip needs a uint8 image of shape {expected}"
        )
    return image


def _describe(request: VLMRequest) -> str:
    """Name the answer of a request for a log line or an error. The cache key names the stored answer."""
    meta = request.meta
    return (
        f"the {meta.step} answer of clip {meta.clip_id} (frame {meta.frame}, item {meta.item}, "
        f"repeat {meta.repeat}, key {request.cache_key()})"
    )


def _parsed(request: VLMRequest, backend: Backend, parse: Callable[[str], _Answer]) -> _Answer | None:
    """Send a request and parse its answer. The result is None when the answer cannot be parsed.

    Such an answer is a measured failure of the step, so the probe goes on. A
    warning names the answer.
    """
    text = backend.generate(request).text
    try:
        return parse(text)
    except steps.ParseError as error:
        logger.warning("%s cannot be parsed and counts as an empty answer: %s", _describe(request), error)
        return None


def _ask_kit_legend(clip: LabelledClip, backend: Backend, settings: StepSettings, repeat: int) -> KitLegend:
    """Ask the model for the kit colours of a clip. An answer that cannot be parsed raises ``ParseError``."""
    images = [_read_frame(clip, frame) for frame in _legend_frames(clip.labels)]
    request = steps.build_kit_legend_request(
        images, clip_id=clip.source.info.clip_id, settings=settings, repeat=repeat
    )
    text = backend.generate(request).text
    try:
        return steps.parse_kit_legend(text)
    except steps.ParseError as error:
        raise steps.ParseError(f"{_describe(request)} cannot be parsed: {error}") from error


# --- Labels as arrays ----------------------------------------------------------------


def _label_points(athletes: Sequence[GsrAthlete]) -> tuple[np.ndarray, np.ndarray]:
    """Return the foot points (pixels) and the pitch positions (metres) of the athletes that have both.

    Both arrays have the shape (n, 2), and row i of both belongs to the same
    athlete. The foot point is the bottom centre of the labelled box.
    """
    feet: list[tuple[float, float]] = []
    positions: list[tuple[float, float]] = []
    for athlete in athletes:
        if athlete.box is not None and athlete.pitch_x is not None and athlete.pitch_y is not None:
            feet.append(athlete.box.foot)
            positions.append((athlete.pitch_x, athlete.pitch_y))
    return (
        np.array(feet, dtype=np.float64).reshape(-1, 2),
        np.array(positions, dtype=np.float64).reshape(-1, 2),
    )


def _label_mapping(athletes: Sequence[GsrAthlete]) -> np.ndarray | None:
    """Fit the image-to-pitch matrix of one frame from its labels. None means that the labels give no fit.

    The fit is the one of the label check ``evalx.oracle.label_homography_residuals_m``:
    at least 6 athletes with a box and a pitch position, RANSAC with 5 pixels.
    """
    feet, positions = _label_points(athletes)
    if len(feet) < MIN_ATHLETES_FOR_FIT:
        return None
    fit = fit_homography(feet, positions, ransac_px=LABEL_RANSAC_PX)
    return None if fit is None else fit.H


def _feet_on_pitch(athletes: Sequence[GsrAthlete]) -> np.ndarray:
    """Return the foot points of the labelled people who stand on the pitch, in pixels, shape (n, 2)."""
    feet = [
        athlete.box.foot for athlete in athletes if athlete.box is not None and athlete.role != ROLE_NEXT_TO_PITCH
    ]
    return np.array(feet, dtype=np.float64).reshape(-1, 2)


# --- People probe --------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _PeopleTask:
    """The label side of the people probe for one clip."""

    clip: LabelledClip
    frames: tuple[int, ...]
    #: The labelled people of each frame, in the order of ``frames``.
    truth: tuple[FramePeople, ...]
    #: Frame -> image-to-pitch matrix fitted from the labels, or None when the labels give no fit.
    mappings: dict[int, np.ndarray | None]


@dataclass(frozen=True, eq=False)
class _PeopleResult:
    """What the people probe measured on one clip."""

    clip_id: str
    n_frames: int
    scores: DetectionScores
    small_scores: DetectionScores
    #: Foot-point errors of the matched boxes in metres, shape (n,).
    foot_errors_m: np.ndarray
    n_without_label_mapping: int
    #: Answered boxes without a match whose foot point lies off the pitch (``_wrong_boxes_off_pitch``).
    n_wrong_off_pitch: int


def probe_people(
    clips: Sequence[LabelledClip],
    backend: Backend,
    settings: StepSettings,
    *,
    frames_per_clip: int,
    seed: int,
    small_box_max_height_px: float,
    repeat: int = 0,
    video: VideoProbe | None = None,
    video_folder: Path | None = None,
) -> dict[str, Any]:
    """Measure the people step on sampled frames of labelled clips.

    For each clip the kit legend is asked one time (from the first, the middle
    and the last scored frame), then the people step is asked for each sampled
    frame: one kit legend request per clip and one people request per frame.
    ``settings`` is used for both steps. The answers are compared with the
    labelled people of the frame (``labels.gsr_frame_people``, all roles).

    The result holds:

    - ``recall`` and ``precision`` at IoU 0.5 over all frames, and ``counts``.
      ``counts.fp_off_pitch`` is the part of the wrong boxes (``fp``) whose
      foot point lies more than 3 m outside the pitch, by the mapping that is
      fitted from the labels of the frame. The pipeline drops such people,
      and the people prompt tells the model to leave the spectators out. The
      precision counts them as wrong boxes. A frame whose labels give no
      mapping adds nothing to this count.
    - ``small_boxes``: the same for the labelled boxes up to
      ``small_box_max_height_px`` high. A prediction without a match counts as
      wrong here too (see ``evalx.metrics.detection_scores``).
    - ``foot_error_m``: the pitch distance between the foot points of matched
      boxes. Both points go through a mapping fitted from the labels of the
      frame, so the value measures the box placement only. Frames whose labels
      give no mapping are left out and counted in
      ``n_frames_without_label_mapping``.
    - ``main_metric`` and ``per_clip``: the recall per clip.

    With ``video`` (``VideoProbe``) the probe sends video, and ``video_folder`` is
    needed: the videos are written to ``video_folder`` for all clips before the
    first request (``plan_people_video``), the requests are the video requests of
    the same frames, and the result also holds ``video`` (its settings and the
    number of requests) and ``n_entries_wrong_time``: the answer entries whose time
    did not name their target frame, which count as no people
    (``steps.parse_people_video``).
    """
    _check_clips(clips)
    _check_at_least("frames_per_clip", frames_per_clip, 1)
    _check_at_least("repeat", repeat, 0)
    _check_above_zero("small_box_max_height_px", small_box_max_height_px)
    _check_video_probe(settings, video, video_folder)
    measurement = _people_measurement(frames_per_clip, small_box_max_height_px)
    if video is not None:
        measurement["video"] = {**video.model_dump(), "write_fps": video.write_fps}
    used_settings = _settings(settings, seed, repeat, **measurement)
    tasks = [_people_task(clip, frames_per_clip, seed) for clip in clips]
    if video is None or video_folder is None:
        results: list[_PeopleResult] = []
        n_not_parsed = 0
        for task in tasks:
            answers, not_parsed = _ask_people(task, backend, settings, repeat)
            results.append(_score_people(task, answers, small_box_max_height_px))
            n_not_parsed += not_parsed
        step_facts = {"prompts": _prompt_names("kit_legend", "people"), "n_answers_not_parsed": n_not_parsed}
        return _people_report(results, used_settings, step_facts)
    plans = [_video_clip_plan(task.clip, task.frames, settings, video, video_folder) for task in tasks]
    results = []
    n_not_parsed = n_wrong_time = n_requests = 0
    for task, plan in zip(tasks, plans, strict=True):
        answers, not_parsed, wrong_time = _ask_people_video(task, plan, backend, settings, video, repeat)
        results.append(_score_people(task, answers, small_box_max_height_px))
        n_not_parsed += not_parsed
        n_wrong_time += wrong_time
        n_requests += len(plan.requests)
    step_facts = {
        "prompts": _prompt_names("kit_legend_video", "people_video"),
        "n_answers_not_parsed": n_not_parsed,
        "n_entries_wrong_time": n_wrong_time,
        "n_people_requests": n_requests,
    }
    return _people_report(results, used_settings, step_facts)


def score_people_answers(
    clips: Sequence[LabelledClip],
    people_of: PeopleOfFrame,
    tool: Mapping[str, Any],
    *,
    frames_per_clip: int,
    seed: int,
    small_box_max_height_px: float,
) -> dict[str, Any]:
    """Measure the people that an external tool found, on the frames of the people probe.

    No model is called. ``people_of(clip_id, frame)`` gives the people that
    the tool found in one frame. It is called one time for each frame of
    ``probe_frames``, and its answer must name that frame. ``tool`` holds the
    settings of the tool as plain JSON types; it goes into the settings of
    the result.

    The frames, the truth and the measured values are those of
    ``probe_people`` with the same clips, ``frames_per_clip`` and ``seed``.
    The result has the keys of ``probe_people`` without ``prompts`` and
    ``n_answers_not_parsed``. A tool gives no role, so every person counts as
    in ``probe_people``: by the box alone.
    """
    _check_clips(clips)
    _check_at_least("frames_per_clip", frames_per_clip, 1)
    _check_above_zero("small_box_max_height_px", small_box_max_height_px)
    used_settings = _tool_settings(tool, seed, **_people_measurement(frames_per_clip, small_box_max_height_px))
    tasks = [_people_task(clip, frames_per_clip, seed) for clip in clips]
    results: list[_PeopleResult] = []
    for task in tasks:
        clip_id = task.clip.source.info.clip_id
        answers = [_answer_of_frame(people_of(clip_id, frame), clip_id, frame) for frame in task.frames]
        results.append(_score_people(task, answers, small_box_max_height_px))
    return _people_report(results, used_settings, step_facts={})


def _people_measurement(frames_per_clip: int, small_box_max_height_px: float) -> dict[str, Any]:
    """Return the settings that define how the people of a frame are measured against the labels."""
    return {
        "frames_per_clip": operator.index(frames_per_clip),
        "iou_threshold": IOU_THRESHOLD,
        "small_box_max_height_px": float(small_box_max_height_px),
        "label_ransac_px": LABEL_RANSAC_PX,
        "label_min_athletes": MIN_ATHLETES_FOR_FIT,
    }


def _people_report(
    results: Sequence[_PeopleResult], used_settings: dict[str, Any], step_facts: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the result of a people measurement from what was measured on each clip.

    ``step_facts`` holds the values that only a model step has: its prompt
    names and the number of its answers that could not be parsed.
    """
    scores = _sum_scores(result.scores for result in results)
    small_scores = _sum_scores(result.small_scores for result in results)
    return {
        "probe": "people",
        "n_clips": len(results),
        "n_frames": sum(result.n_frames for result in results),
        "settings": used_settings,
        **step_facts,
        "recall": scores.recall,
        "precision": scores.precision,
        "counts": {
            "predicted_boxes": scores.tp + scores.fp,
            **_counts(scores),
            "fp_off_pitch": sum(result.n_wrong_off_pitch for result in results),
        },
        "small_boxes": {"recall": small_scores.recall, "precision": small_scores.precision, **_counts(small_scores)},
        "foot_error_m": _error_summary(_concat([result.foot_errors_m for result in results])),
        "n_frames_without_label_mapping": sum(result.n_without_label_mapping for result in results),
        "main_metric": _main_metric("recall", [result.scores.recall for result in results]),
        "per_clip": {
            result.clip_id: {
                "n_frames": result.n_frames,
                "recall": result.scores.recall,
                "precision": result.scores.precision,
                "small_box_recall": result.small_scores.recall,
                "foot_error_median_m": _error_summary(result.foot_errors_m)["median"],
            }
            for result in results
        },
    }


def _people_task(clip: LabelledClip, frames_per_clip: int, seed: int) -> _PeopleTask:
    """Choose the frames of a clip and read their truth from the labels. No model is called here."""
    labels = clip.labels
    frames = sample_frames(labels, frames_per_clip, seed)
    return _PeopleTask(
        clip=clip,
        frames=tuple(frames),
        truth=tuple(gsr_frame_people(labels, frame) for frame in frames),
        mappings={frame: _label_mapping(labels.frames[frame].athletes) for frame in frames},
    )


def _ask_people(
    task: _PeopleTask, backend: Backend, settings: StepSettings, repeat: int
) -> tuple[list[FramePeople], int]:
    """Ask the kit legend and the people of every frame of one clip.

    The result is the answer of each frame, in the order of the frames, and
    the number of answers that could not be parsed. Such an answer counts as
    no people.
    """
    clip = task.clip
    info = clip.source.info
    legend = _ask_kit_legend(clip, backend, settings, repeat)
    answers: list[FramePeople] = []
    n_not_parsed = 0
    for frame in task.frames:
        request = steps.build_people_request(
            _read_frame(clip, frame), legend, clip_id=info.clip_id, frame=frame, settings=settings, repeat=repeat
        )
        parse = partial(
            steps.parse_people, frame=frame, width=info.width, height=info.height, coords=settings.coords
        )
        people = _parsed(request, backend, parse)
        if people is None:
            n_not_parsed += 1
            people = FramePeople(frame=frame)
        answers.append(people)
    return answers, n_not_parsed


def _ask_people_video(
    task: _PeopleTask,
    plan: _VideoClipPlan,
    backend: Backend,
    settings: StepSettings,
    video: VideoProbe,
    repeat: int,
) -> tuple[list[FramePeople], int, int]:
    """Ask the kit legend and the people of every frame of one clip with video requests.

    The result is the answer of each frame, in the order of the frames, the number of
    answers that could not be parsed (all their target frames count as no people) and
    the number of answer entries whose time did not name their target frame.
    """
    clip = task.clip
    info = clip.source.info
    legend_request = _legend_video_request(clip, plan, settings, video, repeat)
    text = backend.generate(legend_request).text
    try:
        legend = steps.parse_kit_legend(text)
    except steps.ParseError as error:
        raise steps.ParseError(f"{_describe(legend_request)} cannot be parsed: {error}") from error
    fps = video.write_fps
    by_frame: dict[int, FramePeople] = {}
    n_not_parsed = n_wrong_time = 0
    for entry in plan.requests:
        request = _people_video_request(clip, entry, legend, settings, video, repeat)
        parse = partial(
            steps.parse_people_video,
            targets=entry.targets,
            times_s=request.context["times_s"],
            fps=fps,
            width=info.width,
            height=info.height,
            coords=settings.coords,
        )
        parsed = _parsed(request, backend, parse)
        if parsed is None:
            n_not_parsed += 1
            parsed = ([FramePeople(frame=frame) for frame in entry.targets], 0)
        people, wrong_time = parsed
        n_wrong_time += wrong_time
        by_frame.update((answer.frame, answer) for answer in people)
    return [by_frame[frame] for frame in task.frames], n_not_parsed, n_wrong_time


def _score_people(task: _PeopleTask, answers: Sequence[FramePeople], small_box_max_height_px: float) -> _PeopleResult:
    """Score the people that were answered for the frames of one clip against the labels."""

    def to_pitch(frame: int, xy: np.ndarray) -> np.ndarray | None:
        mapping = task.mappings[frame]
        return None if mapping is None else apply_homography(mapping, xy)

    clip_id = task.clip.source.info.clip_id
    logger.info("people probe: clip %s, %d frames", clip_id, len(task.frames))
    return _PeopleResult(
        clip_id=clip_id,
        n_frames=len(task.frames),
        scores=detection_scores(answers, task.truth, iou_threshold=IOU_THRESHOLD),
        small_scores=detection_scores(
            answers, task.truth, iou_threshold=IOU_THRESHOLD, max_height_px=small_box_max_height_px
        ),
        foot_errors_m=foot_errors_m(answers, task.truth, to_pitch, iou_threshold=IOU_THRESHOLD),
        n_without_label_mapping=sum(1 for mapping in task.mappings.values() if mapping is None),
        n_wrong_off_pitch=_wrong_boxes_off_pitch(task, answers),
    )


def _wrong_boxes_off_pitch(task: _PeopleTask, answers: Sequence[FramePeople]) -> int:
    """Count the answered boxes without a match whose foot point lies off the pitch.

    ``answers`` holds the answer of each frame in the order of ``task.frames``.
    A box is off the pitch when the label mapping of its frame puts its foot
    point more than ``PITCH_MARGIN_M`` outside the pitch. That is the rule
    with which the pipeline drops a person (``pipeline._people_on_pitch``): a
    foot point without a finite pitch position counts as outside. A frame
    whose labels give no mapping adds nothing, because its boxes cannot be
    judged.
    """
    count = 0
    for people, truth in zip(answers, task.truth, strict=True):
        mapping = task.mappings[truth.frame]
        if mapping is None:
            continue
        boxes = [person.box for person in people.people]
        matched = {index for index, _ in match_boxes(boxes, [person.box for person in truth.people], IOU_THRESHOLD)}
        feet = [box.foot for index, box in enumerate(boxes) if index not in matched]
        if not feet:
            continue
        with np.errstate(invalid="ignore"):  # A foot point at infinity is outside the pitch.
            on_pitch = inside_pitch(apply_homography(mapping, np.array(feet, dtype=np.float64)), PITCH_MARGIN_M)
        count += int((~on_pitch).sum())
    return count


def _answer_of_frame(answer: _FrameAnswer, clip_id: str, frame: int) -> _FrameAnswer:
    """Return the answer of an external tool for one frame. ``ValueError`` is raised when it names another frame."""
    if answer.frame != frame:
        raise ValueError(f"clip {clip_id}: the answer for frame {frame} names frame {answer.frame}")
    return answer


def _sum_scores(parts: Iterable[DetectionScores]) -> DetectionScores:
    """Add the counts of several detection scores."""
    tp = fp = fn = 0
    for part in parts:
        tp += part.tp
        fp += part.fp
        fn += part.fn
    return DetectionScores(tp=tp, fp=fp, fn=fn)


def _counts(scores: DetectionScores) -> dict[str, int]:
    """Return the counts of detection scores: the true boxes that count, and the found, wrong and missed boxes."""
    return {"true_boxes": scores.tp + scores.fn, "tp": scores.tp, "fp": scores.fp, "fn": scores.fn}


# --- Pitch probe ---------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _MappedFrame:
    """The pitch mapping of one frame, measured at the labelled foot points of that frame."""

    frame: int
    #: ``goal`` or ``midfield``, or None for a frame without labelled people.
    view: str | None
    #: True when the frame has a mapping.
    valid: bool
    #: True when the mapping also passes the plausibility check that the pipeline uses.
    plausible: bool
    #: Where the mapping puts the labelled foot points, pitch metres, shape (n, 2). Empty without a mapping.
    mapped_xy: np.ndarray
    #: The labelled pitch positions of the same people, shape (n, 2). Empty without a mapping.
    label_xy: np.ndarray
    #: The paint alignment of the mapping in its frame (``geometry.paint_alignment``). None without a mapping.
    paint: PaintAlignment | None

    @property
    def errors_m(self) -> np.ndarray:
        """Position errors at the labelled foot points in metres, shape (n,). Empty without a mapping."""
        return np.linalg.norm(self.mapped_xy - self.label_xy, axis=1)


@dataclass(frozen=True, eq=False)
class _PitchFrame:
    """What the pitch probe measured on one frame: the mapping of the landmark fit, and facts of the answer."""

    mapped: _MappedFrame
    #: True when the landmark fit gives a mapping without its share rule (``min_near_share`` 0).
    valid_without_share_rule: bool
    n_landmarks: int
    n_inliers: int
    #: False when the answer could not be parsed.
    parsed: bool


def probe_pitch(
    clips: Sequence[LabelledClip],
    backend: Backend,
    settings: StepSettings,
    *,
    frames_per_clip: int,
    seed: int,
    ransac_px: float,
    repeat: int = 0,
) -> dict[str, Any]:
    """Measure the pitch mapping from the landmarks step on sampled frames of labelled clips.

    For each sampled frame the landmarks step is asked (one request per
    frame), and ``geometry.fit_from_landmarks`` fits the image-to-pitch
    mapping with the RANSAC threshold ``ransac_px``. The position error is the
    pitch distance between the mapped foot point of a labelled box and the
    labelled pitch position of that person.

    The result holds, for all frames and under ``by_view`` for the two views:

    - ``n_frames``, ``n_frames_valid`` and ``valid_share``: a frame is valid
      when the fit gave a mapping. A frame without a mapping is not valid.
    - ``n_frames_valid_without_share_rule`` and
      ``valid_share_without_share_rule``: the same for a fit without the
      share rule of ``geometry.fit_homography`` (``min_near_share`` 0). The
      difference to ``valid_share`` is the share of frames that this rule
      refuses, so it shows what the rule costs on real answers.
    - ``error_m``: the position errors of all labelled people in the valid frames.
    - ``n_frames_plausible``, ``plausible_share`` and ``plausible_error_m``:
      the same for the valid frames whose mapping also passes
      ``pipeline.fit_problems`` with the labelled foot points. This is the
      check that the pipeline uses before it keeps a landmark fit: the pitch
      must be the right way round, and the fit must put the people on the
      pitch (9 of 10 of them, or 2 of 3 for a fit with 7 or more agreeing
      landmarks; one person outside never decides).

    The view of a frame is ``goal`` when the mean ``|x|`` of its labelled
    pitch positions is above 20 m, else ``midfield``. A frame without labelled
    people has no view and is in the values of all frames only.

    ``landmarks_per_frame`` and ``inliers_per_valid_frame`` describe the
    answers. ``main_metric`` and ``per_clip`` use the median position error
    per clip. ``paint_alignment`` scores every valid mapping (``paint_report``).
    """
    _check_clips(clips)
    _check_at_least("frames_per_clip", frames_per_clip, 1)
    _check_at_least("repeat", repeat, 0)
    _check_above_zero("ransac_px", ransac_px)
    used_settings = _settings(
        settings,
        seed,
        repeat,
        frames_per_clip=operator.index(frames_per_clip),
        ransac_px=float(ransac_px),
        goal_view_min_mean_abs_x_m=GOAL_VIEW_MIN_MEAN_ABS_X_M,
    )
    sampled = [(clip, sample_frames(clip.labels, frames_per_clip, seed)) for clip in clips]
    frames_of_clip: dict[str, list[_PitchFrame]] = {}
    for clip, frames in sampled:
        clip_id = clip.source.info.clip_id
        frames_of_clip[clip_id] = [
            _probe_pitch_frame(clip, frame, backend, settings, ransac_px, repeat) for frame in frames
        ]
        logger.info("pitch probe: clip %s, %d frames", clip_id, len(frames))
    return _pitch_report(frames_of_clip, used_settings)


def score_pitch_mappings(
    clips: Sequence[LabelledClip],
    mapping_of: MappingOfFrame,
    tool: Mapping[str, Any],
    *,
    frames_per_clip: int,
    seed: int,
) -> dict[str, Any]:
    """Measure the pitch mappings of an external tool, on the frames of the pitch probe.

    No model is called. ``mapping_of(clip_id, frame)`` gives the image-to-pitch
    mapping of the tool for one frame, or a record without a matrix when the
    tool found none. It is called one time for each frame of ``probe_frames``,
    and its answer must name that frame. ``tool`` holds the settings of the
    tool as plain JSON types; it goes into the settings of the result.

    The frames and the measured values are those of ``probe_pitch`` with the
    same clips, ``frames_per_clip`` and ``seed``. The result holds, for all
    frames and under ``by_view``: ``n_frames``, ``n_frames_valid``,
    ``valid_share``, ``error_m``, ``n_frames_plausible``, ``plausible_share``
    and ``plausible_error_m``. A frame is valid when the tool gave a mapping.
    The plausibility check is the one of the pipeline. A tool mapping has no
    count of agreeing landmarks, so the check wants 9 of 10 people on the
    pitch. The values that belong to a landmark fit are not in the result.
    ``main_metric`` and ``per_clip`` use the median position error per clip.

    ``axis_check`` tests the pitch convention of the tool output. It is not a
    result of the tool. It holds the median position error of all valid
    frames for the mapped points as they are (``as_given``), and with the
    sign of x, of y or of both changed. The convention is right when
    ``as_given`` is the smallest of the four and small (some metres at most).
    When ``as_given`` is large and another value is small, either the
    matrices are in another pitch convention (the script that ran the tool
    has a fault, and no frame is plausible), or the tool mirrors the pitch on
    more than half of the frames (some frames are plausible). The other
    values then say nothing about the tool. When all four values are large,
    the check decides nothing: the output of the tool is wrong for another
    reason.

    The error values are counted on the valid frames. So the results of two
    engines that refuse a different number of frames are not counted on the
    same frames; a weak mapping of an engine that never refuses shows in the
    90th percentile and in the mean of ``error_m``, not in its median.

    ``paint_alignment`` scores every valid mapping (``paint_report``).
    """
    _check_clips(clips)
    _check_at_least("frames_per_clip", frames_per_clip, 1)
    used_settings = _tool_settings(
        tool,
        seed,
        frames_per_clip=operator.index(frames_per_clip),
        goal_view_min_mean_abs_x_m=GOAL_VIEW_MIN_MEAN_ABS_X_M,
    )
    sampled = [(clip, sample_frames(clip.labels, frames_per_clip, seed)) for clip in clips]
    frames_of_clip: dict[str, list[_MappedFrame]] = {}
    for clip, frames in sampled:
        clip_id = clip.source.info.clip_id
        records = [_answer_of_frame(mapping_of(clip_id, frame), clip_id, frame) for frame in frames]
        frames_of_clip[clip_id] = [
            _measure_mapping(clip, frame, record) for frame, record in zip(frames, records, strict=True)
        ]
        logger.info("pitch probe: clip %s, %d frames", clip_id, len(frames))
    all_frames = [frame for frames in frames_of_clip.values() for frame in frames]
    return {
        "probe": "pitch",
        "n_clips": len(frames_of_clip),
        "settings": used_settings,
        **_mapping_summary(all_frames),
        "by_view": {view: _mapping_summary([frame for frame in all_frames if frame.view == view]) for view in VIEWS},
        "axis_check": _axis_check(all_frames),
        "paint_alignment": paint_report(frames_of_clip),
        **_error_by_clip({clip_id: _mapping_summary(frames) for clip_id, frames in frames_of_clip.items()}),
    }


def _pitch_report(frames_of_clip: dict[str, list[_PitchFrame]], used_settings: dict[str, Any]) -> dict[str, Any]:
    """Build the result of the pitch probe from what was measured on the frames of each clip."""
    all_frames = [frame for frames in frames_of_clip.values() for frame in frames]
    return {
        "probe": "pitch",
        "n_clips": len(frames_of_clip),
        "settings": used_settings,
        "prompts": _prompt_names("landmarks"),
        **_pitch_summary(all_frames),
        "by_view": {
            view: _pitch_summary([frame for frame in all_frames if frame.mapped.view == view]) for view in VIEWS
        },
        "landmarks_per_frame": summarize(np.array([frame.n_landmarks for frame in all_frames], dtype=np.float64)),
        "inliers_per_valid_frame": summarize(
            np.array([frame.n_inliers for frame in all_frames if frame.mapped.valid], dtype=np.float64)
        ),
        "n_answers_not_parsed": sum(1 for frame in all_frames if not frame.parsed),
        "paint_alignment": paint_report(
            {clip_id: [frame.mapped for frame in frames] for clip_id, frames in frames_of_clip.items()}
        ),
        **_error_by_clip({clip_id: _pitch_summary(frames) for clip_id, frames in frames_of_clip.items()}),
    }


def _error_by_clip(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Return ``main_metric`` and ``per_clip`` of a pitch measurement from the summary of each clip."""
    return {
        "main_metric": _main_metric(
            "error_median_m", [summary["error_m"]["median"] for summary in summaries.values()]
        ),
        "per_clip": {
            clip_id: {
                "n_frames": summary["n_frames"],
                "valid_share": summary["valid_share"],
                "error_median_m": summary["error_m"]["median"],
            }
            for clip_id, summary in summaries.items()
        },
    }


def pitch_probe_fits(
    clips: Sequence[LabelledClip],
    backend: Backend,
    settings: StepSettings,
    *,
    frames_per_clip: int,
    seed: int,
    ransac_px: float,
    repeat: int = 0,
) -> dict[str, list[tuple[HomographyRecord, bool]]]:
    """Return the landmark fit of every frame of the pitch probe, by clip id, in frame order.

    The frames, the requests and the fit are those of ``probe_pitch`` with the same
    arguments, so a backend that replays the stored answers of a pitch probe gives the
    mappings that the probe measured. Each fit comes with True when the answer could be
    parsed. An answer that cannot be parsed gives a record without a matrix, as in the probe.
    """
    _check_clips(clips)
    _check_at_least("frames_per_clip", frames_per_clip, 1)
    _check_at_least("repeat", repeat, 0)
    _check_above_zero("ransac_px", ransac_px)
    return {
        clip.source.info.clip_id: [
            (fit_from_landmarks(landmarks, ransac_px=ransac_px), parsed)
            for landmarks, parsed in (
                _landmarks(clip, frame, backend, settings, repeat)
                for frame in sample_frames(clip.labels, frames_per_clip, seed)
            )
        ]
        for clip in clips
    }


def _landmarks(
    clip: LabelledClip, frame: int, backend: Backend, settings: StepSettings, repeat: int
) -> tuple[FrameLandmarks, bool]:
    """Ask the landmarks of one frame. Return them, and False when the answer could not be parsed (no landmark then)."""
    info = clip.source.info
    request = steps.build_landmarks_request(
        _read_frame(clip, frame), clip_id=info.clip_id, frame=frame, settings=settings, repeat=repeat
    )
    parse = partial(
        steps.parse_landmarks, frame=frame, width=info.width, height=info.height, coords=settings.coords
    )
    answer = _parsed(request, backend, parse)
    return (FrameLandmarks(frame=frame), False) if answer is None else (answer, True)


def _probe_pitch_frame(
    clip: LabelledClip, frame: int, backend: Backend, settings: StepSettings, ransac_px: float, repeat: int
) -> _PitchFrame:
    """Ask the landmarks of one frame, fit the pitch mapping and measure it at the labelled foot points."""
    landmarks, parsed = _landmarks(clip, frame, backend, settings, repeat)
    record = fit_from_landmarks(landmarks, ransac_px=ransac_px)
    without_share_rule = fit_from_landmarks(landmarks, ransac_px=ransac_px, min_near_share=0.0)
    return _PitchFrame(
        mapped=_measure_mapping(clip, frame, record),
        valid_without_share_rule=without_share_rule.valid,
        n_landmarks=len(landmarks.landmarks),
        n_inliers=record.n_inliers,
        parsed=parsed,
    )


def _measure_mapping(clip: LabelledClip, frame: int, record: HomographyRecord) -> _MappedFrame:
    """Measure the pitch mapping of one frame at the labelled foot points of that frame.

    A mapping is plausible when it passes ``pipeline.fit_problems`` with the
    foot points of the labelled people who stand on the pitch. A mapping also
    gets its paint alignment in the frame (``geometry.paint_alignment``).
    """
    info = clip.source.info
    athletes = clip.labels.frames[frame].athletes
    feet, positions = _label_points(athletes)
    view = _view(positions)
    if not record.valid:
        nowhere = np.zeros((0, 2), dtype=np.float64)
        return _MappedFrame(
            frame=frame, view=view, valid=False, plausible=False, mapped_xy=nowhere, label_xy=nowhere, paint=None
        )
    problems = fit_problems(
        record, _feet_on_pitch(athletes), (info.width, info.height), pitch_margin_m=PITCH_MARGIN_M
    )
    matrix = matrix_from_record(record)
    return _MappedFrame(
        frame=frame,
        view=view,
        valid=True,
        plausible=not problems,
        mapped_xy=apply_homography(matrix, feet),
        label_xy=positions,
        paint=paint_alignment(_read_frame(clip, frame), matrix),
    )


def _view(positions: np.ndarray) -> str | None:
    """Name the view of a frame from its labelled pitch positions, shape (n, 2).

    The view is ``goal`` when the mean ``|x|`` is above 20 m, else
    ``midfield``. A frame without labelled people has no view.
    """
    if len(positions) == 0:
        return None
    return "goal" if float(np.abs(positions[:, 0]).mean()) > GOAL_VIEW_MIN_MEAN_ABS_X_M else "midfield"


def _mapping_summary(frames: Sequence[_MappedFrame]) -> dict[str, Any]:
    """Summarise measured mappings: the shares of valid and plausible frames and their position errors."""
    valid = [frame for frame in frames if frame.valid]
    plausible = [frame for frame in frames if frame.plausible]
    return {
        "n_frames": len(frames),
        "n_frames_valid": len(valid),
        "valid_share": _share(len(valid), len(frames)),
        "error_m": _error_summary(_concat([frame.errors_m for frame in valid])),
        "n_frames_plausible": len(plausible),
        "plausible_share": _share(len(plausible), len(frames)),
        "plausible_error_m": _error_summary(_concat([frame.errors_m for frame in plausible])),
    }


def _pitch_summary(frames: Sequence[_PitchFrame]) -> dict[str, Any]:
    """Summarise frames of the pitch probe: the summary of their mappings, and what the share rule costs.

    The valid share is given with and without the share rule of the landmark fit.
    """
    n_without_share_rule = sum(1 for frame in frames if frame.valid_without_share_rule)
    return {
        **_mapping_summary([frame.mapped for frame in frames]),
        "n_frames_valid_without_share_rule": n_without_share_rule,
        "valid_share_without_share_rule": _share(n_without_share_rule, len(frames)),
    }


def auroc(higher: Sequence[float], lower: Sequence[float]) -> float:
    """Return the share of pairs in which a value of ``higher`` is above a value of ``lower``.

    A tie counts as half a pair. This is the area under the ROC curve of a
    score that should be higher for the first group (Mann-Whitney U divided
    by the number of pairs): 1 separates the groups, 0.5 is chance. The value
    is nan when a group is empty.
    """
    first = np.asarray(higher, dtype=np.float64).ravel()
    second = np.asarray(lower, dtype=np.float64).ravel()
    if first.size == 0 or second.size == 0:
        return math.nan
    above = (first[:, None] > second[None, :]).sum()
    ties = (first[:, None] == second[None, :]).sum()
    return float((above + 0.5 * ties) / (first.size * second.size))


def auroc_standard_error(area: float, n_higher: int, n_lower: int) -> float:
    """Return the standard error of an AUROC by Hanley and McNeil (1982). nan when a group is empty."""
    if n_higher == 0 or n_lower == 0 or not math.isfinite(area):
        return math.nan
    q1 = area / (2.0 - area)
    q2 = 2.0 * area * area / (1.0 + area)
    variance = (
        area * (1.0 - area) + (n_higher - 1) * (q1 - area * area) + (n_lower - 1) * (q2 - area * area)
    ) / (n_higher * n_lower)
    return math.sqrt(max(variance, 0.0))


def _median_error_m(frame: _MappedFrame) -> float:
    """Return the median finite position error of a mapped frame, or nan when it has none."""
    errors = frame.errors_m
    finite = errors[np.isfinite(errors)]
    return float(np.median(finite)) if finite.size else math.nan


def paint_report(frames_of_clip: Mapping[str, Sequence[_MappedFrame]]) -> dict[str, Any]:
    """Report the paint alignment of the valid mappings of a pitch measurement, and how it separates good from bad.

    The result holds:

    - ``n_mappings``, ``n_scored`` and ``n_cannot_judge``: the valid
      mappings, those with a score, and those whose frame cannot judge a
      mapping; ``cannot_judge_reasons`` counts the reasons by their first words.
    - ``score``: the distribution of the scores (``n``, ``median``, ``p90``, ``mean``).
    - ``separation``: the scored mappings whose median position error is at
      most ``PAINT_GOOD_MAX_ERROR_M`` (good) against those above
      ``PAINT_BAD_MIN_ERROR_M`` (bad): ``n_good``, ``n_bad``, the score
      distribution of each group, ``auroc`` (``auroc``: the chance that a good
      mapping scores above a bad one) and its standard error. A mapping
      between the two limits, or without a finite error, is in neither group.
    - ``frames``: one entry per valid mapping: clip id, frame, score (None
      when the frame cannot judge) and the median position error. Text only.

    It is a diagnostic: no rule of the pipeline uses the score (``docs/DECISIONS.md``).
    """
    entries: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    good: list[float] = []
    bad: list[float] = []
    for clip_id, frames in frames_of_clip.items():
        for frame in frames:
            if frame.paint is None:
                continue
            error_m = _median_error_m(frame)
            score = frame.paint.score
            entries.append({"clip_id": clip_id, "frame": frame.frame, "score": score, "error_median_m": error_m})
            if score is None:
                kind = (frame.paint.reason or "").split(":")[0]
                reasons[kind] = reasons.get(kind, 0) + 1
            elif error_m <= PAINT_GOOD_MAX_ERROR_M:
                good.append(score)
            elif error_m > PAINT_BAD_MIN_ERROR_M:
                bad.append(score)
    scores = [entry["score"] for entry in entries if entry["score"] is not None]
    area = auroc(good, bad)
    return {
        "n_mappings": len(entries),
        "n_scored": len(scores),
        "n_cannot_judge": len(entries) - len(scores),
        "cannot_judge_reasons": reasons,
        "score": summarize(np.array(scores, dtype=np.float64)),
        "separation": {
            "good_max_error_m": PAINT_GOOD_MAX_ERROR_M,
            "bad_min_error_m": PAINT_BAD_MIN_ERROR_M,
            "n_good": len(good),
            "n_bad": len(bad),
            "score_good": summarize(np.array(good, dtype=np.float64)),
            "score_bad": summarize(np.array(bad, dtype=np.float64)),
            "auroc": area,
            "auroc_standard_error": auroc_standard_error(area, len(good), len(bad)),
        },
        "frames": entries,
    }


def _axis_check(frames: Sequence[_MappedFrame]) -> dict[str, float]:
    """Return the median position error of the valid frames for each sign change of ``AXIS_SIGNS``.

    The value is nan when no frame is valid.
    """
    valid = [frame for frame in frames if frame.valid]
    nowhere = np.zeros((0, 2), dtype=np.float64)
    mapped_xy = np.concatenate([frame.mapped_xy for frame in valid]) if valid else nowhere
    label_xy = np.concatenate([frame.label_xy for frame in valid]) if valid else nowhere
    return {
        name: float(_error_summary(np.linalg.norm(mapped_xy * np.array(signs) - label_xy, axis=1))["median"])
        for name, signs in AXIS_SIGNS.items()
    }


# --- Number probe --------------------------------------------------------------------


@dataclass(frozen=True)
class _TrackPlan:
    """The crops of one labelled player track, and the shirt number that the labels give."""

    track_id: int
    label_number: int | None
    #: Frame and labelled box of each crop, in frame order.
    crops: tuple[tuple[int, Box], ...]


@dataclass(frozen=True)
class _NumberResult:
    """What the number probe measured on one clip."""

    clip_id: str
    #: (labelled number, answered number) of each track.
    outcomes: tuple[tuple[int | None, int | None], ...]
    n_crops: int
    n_not_parsed: int


def probe_number(
    clips: Sequence[LabelledClip],
    backend: Backend,
    settings: StepSettings,
    *,
    tracks_per_clip: int,
    crops_per_track: int,
    crop_upscale: float,
    seed: int,
    repeat: int = 0,
) -> dict[str, Any]:
    """Measure the number step on crops that are cut from the labelled boxes of player tracks.

    A track is the labelled person with one track id in the scored frames
    where the role is ``player``. Its labelled number is the most frequent
    shirt number of these frames, or None when no frame has one. At most
    ``tracks_per_clip`` tracks per clip are probed; the seed and the clip id
    fix which ones.

    The crops of a track are chosen and cut as the pipeline does it
    (``pipeline.crop_choice``): the tallest boxes first, each at least one
    second away from the others, at most ``crops_per_track``, and two crops
    also for a track that is too short for that; each crop is 15% larger than
    the box on every side and is enlarged by ``crop_upscale``. The kit legend
    is asked one time per clip with ``settings``, and the number request names
    the team as not known. So the probe sends one kit legend request per clip
    that has a player track, and one number request per track.

    Each request gets the item ``track-<id>-<hash>`` with the track id of the
    label file, which is a lasting name of the track. The hash is made from
    the frames, the pixel areas and the enlargement of the crops, so that
    another crop choice never gets the stored answer of an earlier one.

    The result holds ``exact_match`` (the share of tracks whose answered
    number equals the labelled number; "no number" is a value like any
    other), ``null_answer_share``, ``n_tracks``, ``n_crops`` and ``by_label``
    (the same values for the tracks with and without a labelled number).
    ``main_metric`` and ``per_clip`` use the exact-match rate per clip.
    """
    _check_clips(clips)
    _check_at_least("tracks_per_clip", tracks_per_clip, 1)
    _check_at_least("crops_per_track", crops_per_track, 1)
    _check_at_least("crop_upscale", crop_upscale, 1.0)
    _check_at_least("repeat", repeat, 0)
    used_settings = _settings(
        settings,
        seed,
        repeat,
        tracks_per_clip=operator.index(tracks_per_clip),
        crops_per_track=operator.index(crops_per_track),
        crop_upscale=float(crop_upscale),
        crop_padding=steps.CROP_PADDING,
        min_crop_gap_s=MIN_CROP_GAP_S,
        team_key=NUMBER_TEAM_KEY,
    )
    plans = [_track_plans(clip, tracks_per_clip, crops_per_track, seed) for clip in clips]
    results = [
        _run_track_plans(clip, clip_plans, backend, settings, crop_upscale, repeat)
        for clip, clip_plans in zip(clips, plans, strict=True)
    ]
    return _number_report(results, used_settings)


def _number_report(results: Sequence[_NumberResult], used_settings: dict[str, Any]) -> dict[str, Any]:
    """Build the result of the number probe from what was measured on each clip."""
    outcomes = [outcome for result in results for outcome in result.outcomes]
    scores = _number_scores(outcomes)
    return {
        "probe": "number",
        "n_clips": len(results),
        "n_tracks": scores["n_tracks"],
        "settings": used_settings,
        "prompts": _prompt_names("kit_legend", "number"),
        "exact_match": scores["exact_match"],
        "null_answer_share": scores["null_answer_share"],
        "n_crops": sum(result.n_crops for result in results),
        "by_label": {
            "number": _number_scores([outcome for outcome in outcomes if outcome[0] is not None]),
            "null": _number_scores([outcome for outcome in outcomes if outcome[0] is None]),
        },
        "n_answers_not_parsed": sum(result.n_not_parsed for result in results),
        "main_metric": _main_metric(
            "exact_match", [_number_scores(result.outcomes)["exact_match"] for result in results]
        ),
        "per_clip": {result.clip_id: _number_scores(result.outcomes) for result in results},
    }


def _track_plans(clip: LabelledClip, tracks_per_clip: int, crops_per_track: int, seed: int) -> list[_TrackPlan]:
    """Choose the player tracks of a clip and their crops from the labels. No model is called here.

    The result is ordered by track id. A track needs at least one labelled
    box that lies, with its padding, at least partly inside the image. The
    frame rate is the one of the label file.
    """
    labels = clip.labels
    athletes_of_track: dict[int, list[GsrAthlete]] = {}
    boxes_of_track: dict[int, list[tuple[int, Box]]] = {}
    for frame in labels.frames:
        if not frame.scored:
            continue
        for athlete in frame.athletes:
            if athlete.role != PLAYER_ROLE:
                continue
            athletes_of_track.setdefault(athlete.track_id, []).append(athlete)
            if athlete.box is not None and steps.crop_rect(athlete.box, labels.width, labels.height) is not None:
                boxes_of_track.setdefault(athlete.track_id, []).append((frame.frame, athlete.box))
    ranked = sorted(boxes_of_track, key=lambda track_id: (_seeded_hash(seed, labels.clip_id, track_id), track_id))
    return [
        _TrackPlan(
            track_id=track_id,
            label_number=_label_number(athletes_of_track[track_id]),
            crops=crop_choice(boxes_of_track[track_id], labels.fps, crops_per_track),
        )
        for track_id in sorted(ranked[:tracks_per_clip])
    ]


def _label_number(athletes: Sequence[GsrAthlete]) -> int | None:
    """Return the shirt number of a labelled track: its most frequent number, or None when no label has one."""
    numbers = Counter(athlete.jersey for athlete in athletes if athlete.jersey is not None)
    return numbers.most_common(1)[0][0] if numbers else None


def _cut_crop(image: np.ndarray, box: Box, upscale: float) -> np.ndarray:
    """Cut the box of a person, with its padding, out of an image and enlarge it with cubic interpolation.

    The pixel area of the crop is the one of ``steps.crop_rect``.
    """
    height, width = image.shape[:2]
    rect = steps.crop_rect(box, width, height)
    if rect is None:
        raise ValueError(f"the box ({box}) lies outside the image of {width} x {height} pixels")
    left, top, right, bottom = rect
    crop = np.ascontiguousarray(image[top:bottom, left:right])  # rows are y, columns are x
    size = (round((right - left) * upscale), round((bottom - top) * upscale))  # OpenCV wants (width, height).
    return cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC)


def _number_request(
    clip: LabelledClip, plan: _TrackPlan, legend: KitLegend, settings: StepSettings, crop_upscale: float, repeat: int
) -> VLMRequest:
    """Cut the crops of one track and build its number request.

    ``steps.build_number_request`` makes the item of the request from the
    frames, the pixel areas and the enlargement of the crops.
    """
    info = clip.source.info
    crops = [_cut_crop(_read_frame(clip, frame), box, crop_upscale) for frame, box in plan.crops]
    return steps.build_number_request(
        crops,
        legend,
        NUMBER_TEAM_KEY,
        clip_id=info.clip_id,
        track_id=plan.track_id,
        crop_frames=[frame for frame, _ in plan.crops],
        crop_boxes=[box for _, box in plan.crops],
        image_size=(info.width, info.height),
        crop_upscale=crop_upscale,
        settings=settings,
        repeat=repeat,
    )


def _run_track_plans(
    clip: LabelledClip,
    plans: Sequence[_TrackPlan],
    backend: Backend,
    settings: StepSettings,
    crop_upscale: float,
    repeat: int,
) -> _NumberResult:
    """Ask the kit legend of one clip and the shirt number of each planned track.

    A clip without a planned track gets no request at all.
    """
    clip_id = clip.source.info.clip_id
    outcomes: list[tuple[int | None, int | None]] = []
    n_not_parsed = 0
    if plans:
        legend = _ask_kit_legend(clip, backend, settings, repeat)
        for plan in plans:
            request = _number_request(clip, plan, legend, settings, crop_upscale, repeat)
            reading = _parsed(request, backend, steps.parse_number)
            if reading is None:
                n_not_parsed += 1
            outcomes.append((plan.label_number, None if reading is None else reading.number))
    logger.info("number probe: clip %s, %d tracks", clip_id, len(plans))
    return _NumberResult(
        clip_id=clip_id,
        outcomes=tuple(outcomes),
        n_crops=sum(len(plan.crops) for plan in plans),
        n_not_parsed=n_not_parsed,
    )


def _number_scores(outcomes: Sequence[tuple[int | None, int | None]]) -> dict[str, Any]:
    """Score (labelled number, answered number) pairs: the number of tracks and the two shares."""
    truth = {index: label for index, (label, _) in enumerate(outcomes)}
    answers = {index: answer for index, (_, answer) in enumerate(outcomes)}
    return {
        "n_tracks": len(outcomes),
        "exact_match": number_exact_match(answers, truth),
        "null_answer_share": _share(sum(1 for answer in answers.values() if answer is None), len(outcomes)),
    }


# --- Result values -------------------------------------------------------------------


def _check_clips(clips: Sequence[LabelledClip]) -> None:
    """Raise ``ValueError`` for an empty clip list and for a clip id that is used more than once."""
    if len(clips) == 0:
        raise ValueError("a probe needs at least one clip")
    counts = Counter(clip.source.info.clip_id for clip in clips)
    twice = sorted(clip_id for clip_id, count in counts.items() if count > 1)
    if twice:
        raise ValueError(f"every clip needs its own clip id, but these are used more than once: {twice}")


def _check_at_least(name: str, value: float, minimum: float) -> None:
    """Raise ``ValueError`` when a setting is below its smallest value or is not a finite number."""
    if not minimum <= value < math.inf:
        raise ValueError(f"{name} must be a finite number of at least {minimum}, got {value}")


def _check_above_zero(name: str, value: float) -> None:
    """Raise ``ValueError`` when a setting is not above zero or is not a number."""
    if not value > 0:
        raise ValueError(f"{name} must be above 0, got {value}")


def _settings(settings: StepSettings, seed: int, repeat: int, **more: Any) -> dict[str, Any]:
    """Return the settings block of a result: the step settings, the seed, the repeat index and the other values."""
    return {
        "step_settings": settings.model_dump(),
        "seed": operator.index(seed),
        "repeat": operator.index(repeat),
        **more,
    }


def _tool_settings(tool: Mapping[str, Any], seed: int, **more: Any) -> dict[str, Any]:
    """Return the settings block of a tool result: the settings of the tool, the seed and the other values."""
    return {"tool": dict(tool), "seed": operator.index(seed), **more}


def _prompt_names(*step_names: str) -> dict[str, str]:
    """Return the prompt names of the steps that a probe sends, by step."""
    return {step: PROMPT_NAMES[step] for step in step_names}


def _share(part: int, whole: int) -> float:
    """Return ``part / whole``, or nan when ``whole`` is 0."""
    return part / whole if whole else math.nan


def _concat(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Join arrays of shape (n,) into one. No array gives an empty array."""
    return np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.float64)


def _error_summary(errors_m: np.ndarray) -> dict[str, float | int]:
    """Summarise errors in metres: ``n``, ``median``, ``p90`` and ``mean`` of the finite values.

    A point that a mapping sends to infinity has no finite error. Such values
    are left out of the statistics and counted in ``n_not_finite``.
    """
    values = np.asarray(errors_m, dtype=np.float64).ravel()
    finite = values[np.isfinite(values)]
    return {**summarize(finite), "n_not_finite": int(values.size - finite.size)}


def _main_metric(name: str, per_clip: Sequence[float]) -> dict[str, Any]:
    """Describe the main metric of a probe over the clips.

    ``mean`` is the mean of the per-clip values, and ``low`` and ``high`` are
    the ends of its two-sided Student t interval. Clips without a value (nan)
    are left out. One clip gives no interval and no clip gives no mean: these
    values are then nan.
    """
    values = [value for value in per_clip if math.isfinite(value)]
    mean, low, high = t_interval(values, INTERVAL_CONFIDENCE) if values else (math.nan, math.nan, math.nan)
    return {
        "name": name,
        "n_clips": len(values),
        "mean": mean,
        "low": low,
        "high": high,
        "confidence": INTERVAL_CONFIDENCE,
    }


# --- Metrics file --------------------------------------------------------------------


def write_metrics(path: Path, payload: dict[str, Any], *, provenance: dict[str, Any]) -> None:
    """Write a result dict to a JSON file, together with where the result comes from.

    The file holds the keys of ``payload`` and two more keys:

    - ``provenance``: the given dict (model ids, prompt names, seed, repeat
      and other facts of the run). The version of this package is added under
      ``package_version`` when the caller did not give one.
    - ``written_at``: the UTC time of the call, ISO 8601.

    JSON has no number for nan or infinity, so such values are written as
    ``null``. The folder of the file is made when it is missing, and a file
    that is already there is replaced. ``ValueError`` is raised when
    ``payload`` already has one of the two keys.
    """
    taken = sorted({PROVENANCE_KEY, WRITTEN_AT_KEY} & payload.keys())
    if taken:
        raise ValueError(f"the payload must not have the keys {taken}: write_metrics sets them")
    document = {
        **payload,
        PROVENANCE_KEY: {PACKAGE_VERSION_KEY: __version__, **provenance},
        WRITTEN_AT_KEY: datetime.now(UTC).isoformat(timespec="seconds"),
    }
    text = json.dumps(_json_ready(document), indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _json_ready(value: Any) -> Any:
    """Return a copy of a JSON-like value in which every float that is not finite is None."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
