"""Pipeline settings. Defaults are the first-result design; a YAML file can override them."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml
from pydantic import Field, model_validator

from pitchvlm.schemas import Record
from pitchvlm.vlm.types import Resolution, ServiceTier, ThinkingLevel

DEFAULT_MODEL = "gemini-4-argon"
#: Largest value of ``PipelineConfig.max_parallel``. A budget stop comes after the answer that
#: crosses a limit. The requests in flight at that time are paid too, so a run can pass its limit,
#: and the cost ledger the hard stop, by the cost of ``max_parallel - 1`` answers. Without an
#: upper limit a settings file could put a whole step in flight, and no limit could stop that step.
MAX_PARALLEL_LIMIT = 8

#: The thinking levels that the provider documents for each priced model (state of 2026-11-13).
#: The level ``low`` is in every list. The documentation does not say what the service does with
#: a level that a model does not have, so such a pair is refused before any request is built.
THINKING_LEVELS: Mapping[str, tuple[ThinkingLevel, ...]] = MappingProxyType(
    {
        "gemini-4-argon": ("low", "medium", "high"),
        "gemini-4-carbon": ("low", "medium", "high"),
        "gemini-4-hydrogen": ("low", "medium", "high", "xhigh"),
        "gemini-3.8-flash": ("low", "medium", "high"),
        "gemini-robotics-er-2-preview": ("minimal", "low", "medium", "high"),
    }
)


class StepSettings(Record):
    """How one model step is called.

    The thinking level must be one that the model has (``THINKING_LEVELS``). A
    model that is not in that table is not checked.
    """

    model: str = DEFAULT_MODEL
    thinking_level: ThinkingLevel = "low"
    resolution: Resolution = "high"
    service_tier: ServiceTier = "standard"

    @model_validator(mode="after")
    def _check_thinking_level(self) -> StepSettings:
        levels = THINKING_LEVELS.get(self.model)
        if levels is not None and self.thinking_level not in levels:
            raise ValueError(
                f"the model {self.model} has no thinking level '{self.thinking_level}': "
                f"its levels are {', '.join(levels)}"
            )
        return self


class TrackingConfig(Record):
    """Limits used when people are matched between keyframes."""

    #: Fastest plausible speed on the pitch, metres per second.
    max_speed_mps: float = Field(default=9.0, gt=0)
    #: Slack added to the reachable distance, metres (covers mapping error).
    reach_margin_m: float = Field(default=2.0, ge=0)
    #: Largest allowed jump between the expected foot point of a track and a detection, in box heights.
    max_image_jump: float = Field(default=2.5, gt=0)
    #: Cost added when role or team key differ from the vote of the track, in box heights. It is
    #: added once more for every keyframe that the track missed.
    mismatch_penalty: float = Field(default=0.5, ge=0)
    #: A track ends after this many keyframes without a match.
    max_missed_keyframes: int = Field(default=4, ge=0)
    #: Longest time between two fragments that are joined to one track, seconds. The
    #: dense output fills no longer time between two positions of a track either.
    max_stitch_gap_s: float = Field(default=3.0, gt=0, allow_inf_nan=False)
    #: Speed that limits the distance between two fragments that are joined, metres per second.
    #: It is lower than ``max_speed_mps``: nobody sprints on a straight line for seconds.
    stitch_speed_mps: float = Field(default=4.0, gt=0, allow_inf_nan=False)
    #: Tracks with fewer observations are dropped before fragments are joined.
    min_observations: int = Field(default=2, ge=1)


class PipelineConfig(Record):
    #: Time between keyframes in seconds (0.48 s is 12 frames at 25 fps).
    keyframe_interval_s: float = Field(default=0.48, gt=0)
    kit_legend: StepSettings = StepSettings()
    people: StepSettings = StepSettings()
    landmarks: StepSettings = StepSettings()
    number: StepSettings = StepSettings(resolution="low")
    #: Crops per track sent to the shirt-number step.
    number_crops: int = Field(default=8, ge=1)
    #: Enlargement of each crop before it is sent.
    crop_upscale: float = Field(default=4.0, ge=1)
    #: RANSAC threshold of the landmark fit, image pixels.
    ransac_px: float = Field(default=8.0, gt=0)
    #: A frame may borrow the pitch mapping of a keyframe at most this many frames away.
    max_gap_frames: int = Field(default=75, ge=0)
    #: People further outside the pitch than this are dropped, metres.
    pitch_margin_m: float = Field(default=3.0, ge=0)
    tracking: TrackingConfig = TrackingConfig()
    #: Model requests in flight at the same time, at most ``MAX_PARALLEL_LIMIT``.
    max_parallel: int = Field(default=4, ge=1, le=MAX_PARALLEL_LIMIT)
    #: Repeat index of the run. Repeat runs use 1, 2, ... to bypass the cache.
    repeat: int = Field(default=0, ge=0)


def _laid_over(defaults: dict[str, Any], changes: dict[Any, Any]) -> dict[Any, Any]:
    """Return ``defaults`` with the values of ``changes`` laid over them.

    A section (a mapping on both sides) is merged key by key, so a section
    that names one key keeps the defaults of its other keys. Every other value
    of ``changes`` replaces the default. Keys that ``defaults`` does not have
    are kept, so that the validation can report them.
    """
    merged: dict[Any, Any] = dict(defaults)
    for key, value in changes.items():
        default = defaults.get(key)
        if isinstance(default, dict) and isinstance(value, dict):
            merged[key] = _laid_over(default, value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path | None = None) -> PipelineConfig:
    """Read a YAML settings file. ``None`` gives the defaults. Unknown keys are an error.

    The file is laid over the defaults. A setting that the file does not name
    keeps its default, also inside a section: ``number: {model: ...}`` keeps
    the low resolution of the number step. An empty file gives the defaults.
    Any other content that is not a mapping raises ``ValueError``.
    """
    if path is None:
        return PipelineConfig()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the top level must be a mapping")
    return PipelineConfig.model_validate(_laid_over(PipelineConfig().model_dump(), data))
