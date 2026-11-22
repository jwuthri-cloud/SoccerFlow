"""Pipeline settings. Defaults are the first-result design; a YAML file can override them."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from pitchvlm.fallbacks.runs import PnlcalibSettings, RfdetrSettings
from pitchvlm.schemas import Record
from pitchvlm.vlm.coords import GEMINI_CONVENTION, CoordConvention
from pitchvlm.vlm.cost import FREE_MODEL_SUFFIX
from pitchvlm.vlm.types import (
    DEFAULT_PROVIDER,
    OPENROUTER_RESOLUTIONS,
    Provider,
    Resolution,
    ServiceTier,
    ThinkingLevel,
)

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

    ``provider`` names the model service and ``model`` the model there: a Gemini
    model name, or an OpenRouter model slug (``author/name``). ``coords`` is the
    coordinate convention that the prompts ask the model for (``vlm.coords``).

    A Gemini step must use the Gemini convention ``yx_1000``, and its thinking
    level must be one that the model has (``THINKING_LEVELS``; a model that is not
    in that table is not checked). An OpenRouter step must name a paid model (no
    slug with the free suffix), the standard service tier and an image resolution
    of ``OPENROUTER_RESOLUTIONS``. Its thinking level goes to the model as the
    reasoning effort; ``model_default`` sends none, for a model without a thinking
    setting.
    """

    provider: Provider = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    thinking_level: ThinkingLevel = "low"
    resolution: Resolution = "high"
    service_tier: ServiceTier = "standard"
    coords: CoordConvention = GEMINI_CONVENTION

    @model_validator(mode="after")
    def _check_provider_settings(self) -> StepSettings:
        if self.provider == DEFAULT_PROVIDER:
            self._check_gemini()
        else:
            self._check_openrouter()
        return self

    def _check_gemini(self) -> None:
        """Refuse settings that a Gemini model cannot take."""
        if self.coords != GEMINI_CONVENTION:
            raise ValueError(f"a Gemini step uses the coordinate convention {GEMINI_CONVENTION}, not {self.coords}")
        if self.thinking_level == "model_default":
            raise ValueError("a Gemini step needs a thinking level: model_default is for OpenRouter steps only")
        levels = THINKING_LEVELS.get(self.model)
        if levels is not None and self.thinking_level not in levels:
            raise ValueError(
                f"the model {self.model} has no thinking level '{self.thinking_level}': "
                f"its levels are {', '.join(levels)}"
            )

    def _check_openrouter(self) -> None:
        """Refuse settings that an OpenRouter request cannot take, and a free model (rule 5)."""
        if "/" not in self.model:
            raise ValueError(f"an OpenRouter model is named by its slug author/name, got {self.model!r}")
        if self.model.endswith(FREE_MODEL_SUFFIX):
            raise ValueError(f"{self.model} is a free model: only paid models are allowed (rule 5)")
        if self.service_tier != "standard":
            raise ValueError(f"an OpenRouter step has no service tier {self.service_tier}: use standard")
        if self.resolution not in OPENROUTER_RESOLUTIONS:
            raise ValueError(
                f"an OpenRouter step has no image resolution {self.resolution}: "
                f"use {' or '.join(OPENROUTER_RESOLUTIONS)}"
            )


def model_ref(text: str) -> tuple[Provider, str]:
    """Read a model of the command line: ``<model>`` for Gemini, or ``openrouter:<slug>``.

    The part before the first colon names the provider only when it is a provider
    name, so a Gemini model name stays valid as it is. Raises ValueError for an empty
    model name.
    """
    prefix, colon, rest = text.partition(":")
    provider, model = ("openrouter", rest) if colon and prefix == "openrouter" else (DEFAULT_PROVIDER, text)
    if not model.strip():
        raise ValueError(f"no model name in {text!r}")
    return provider, model


def model_label(provider: Provider, model: str) -> str:
    """Write a model as the command line reads it (``model_ref``): the bare name for Gemini."""
    return model if provider == DEFAULT_PROVIDER else f"{provider}:{model}"


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
    #: Arm of gate G4: split a track where its team key flips and stays flipped next to a person of
    #: the other team, before fragments are joined (``tracking.split_team_swaps``). Off until G4 has measured it.
    split_team_swaps: bool = False
    #: Arm of gate G4: link the tracks of a team side when the clip has more of them than a team has
    #: people (``tracking.TEAM_SLOTS``), in ``tracking.finalize_tracks``. Off until G4 has measured it.
    team_slots: bool = False


#: Engine of the people step: the people requests of the model at the keyframes, or a detector on every frame.
PeopleEngine = Literal["model", "detector"]
#: Engine of the pitch mapping: the landmark requests of the model at the keyframes, or a calibration on every frame.
PitchEngine = Literal["model", "calibration"]
#: Engine of the shirt numbers: the number requests of the model on track crops, or none.
NumberEngine = Literal["model", "none"]


class EngineSettings(Record):
    """Which engine does each step that has more than one.

    ``detector`` and ``calibration`` read the clip run file of an external tool (``run
    --detections``, ``run --calibration``): the package never imports a tool. With the
    people engine ``detector`` the team key and the role come from the kit colours
    (``kitcolour``), and the shirt numbers only from the number engine ``model``.
    """

    people: PeopleEngine = "model"
    pitch_mapping: PitchEngine = "model"
    number: NumberEngine = "model"


#: The detector of the people engine ``detector`` by default: RF-DETR Medium with its own threshold.
DEFAULT_DETECTOR = RfdetrSettings(name="rfdetr", model="RFDETRMedium", min_score=0.5, antialias=False)
#: The calibration of the pitch engine ``calibration`` by default: the settings of the pitch probe of PnLCalib.
DEFAULT_CALIBRATION = PnlcalibSettings(name="pnlcalib", kp_threshold=0.3434, line_threshold=0.7867, pnl_refine=True)


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
    #: Engine of the people step, of the pitch mapping and of the shirt numbers.
    engines: EngineSettings = EngineSettings()
    #: Settings of the detector; the clip run file of the people engine ``detector`` must have them.
    detector: RfdetrSettings = DEFAULT_DETECTOR
    #: Settings of the calibration; the clip run file of the pitch engine ``calibration`` must have them.
    calibration: PnlcalibSettings = DEFAULT_CALIBRATION


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
