"""Cost of model calls: dated prices, estimates, the cost ledger and the budget guard.

Prices are in USD per million tokens. Thinking tokens are billed at the output
price. The ``flex`` service tier costs ``FLEX_FACTOR`` times the standard price
and the ``priority`` tier ``PRIORITY_FACTOR`` times. Only the models of
``FLEX_TIER_MODELS`` have a flex tier: a flex call of another model has no
price. A price is valid from its ``valid_from`` day (UTC) until the next entry
of the same model starts.

Two providers share one ledger and one budget guard. A Gemini call is priced from
the table ``PRICES`` of this module. An OpenRouter call is priced at the cost that
its answer reports (``VLMResponse.reported_usd``); the price table of the
OpenRouter models, ``configs/openrouter_prices.yaml`` (``load_price_table``),
gives its estimate before a run. A model without a price in its table has no
estimate, so its run does not start.

Video input has a table of its own: the video rates (``VideoRate``). A rate
says how many tokens a sampled video frame and a second of sound cost, at which
rate the provider samples a video that names none, and at which price these
tokens are billed. The Gemini rates are ``GEMINI_VIDEO_RATES``; the OpenRouter
rates are the ``video`` entries of the price table file
(``load_video_rates``). A model without a video rate takes no video: a request
with a video part for it has no estimate and is not sent.
"""

from __future__ import annotations

import errno
import json
import logging
import math
import os
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType

import yaml
from pydantic import Field

from pitchvlm.schemas import Record
from pitchvlm.vlm.types import (
    DEFAULT_PROVIDER,
    VIDEO_RESOLUTIONS,
    BilledTier,
    Provider,
    Resolution,
    Usage,
    VideoPart,
    VLMRequest,
    VLMResponse,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Price:
    """Price of one model from one day on, in USD per million tokens."""

    model: str
    valid_from: date
    input_usd_per_million: float
    output_usd_per_million: float


#: First day of the price table. An entry with no start day of its own begins here.
_TABLE_START = date(2026, 10, 13)

PRICES: tuple[Price, ...] = (
    Price("gemini-4-argon", _TABLE_START, 2.0, 10.0),
    Price("gemini-4-argon", date(2026, 12, 1), 4.0, 20.0),
    Price("gemini-4-carbon", _TABLE_START, 2.0, 10.0),
    Price("gemini-4-carbon", date(2026, 12, 1), 4.0, 20.0),
    Price("gemini-4-hydrogen", _TABLE_START, 1.5, 7.5),
    Price("gemini-3.8-flash", _TABLE_START, 0.75, 3.75),
    Price("gemini-3.8-flash", date(2027, 1, 1), 1.5, 7.5),
    Price("gemini-robotics-er-2-preview", _TABLE_START, 1.0, 5.0),
    Price("gemini-robotics-er-2-preview", date(2027, 1, 1), 2.0, 10.0),
)

#: Models that have a flex tier. The provider documents none for the robotics model. A flex
#: request for a model without that tier has no price: no estimate counts it, and none is sent.
FLEX_TIER_MODELS: frozenset[str] = frozenset(
    {"gemini-4-argon", "gemini-4-carbon", "gemini-4-hydrogen", "gemini-3.8-flash"}
)
#: The flex tier costs this share of the standard price.
FLEX_FACTOR = 0.5
#: The priority tier costs this many times the standard price. No request asks for this tier, but
#: an answer can report it. The factor is the one of the price list of gemini-4-argon (3.6 / 18
#: against 2 / 10 USD per million tokens), and it is used for every model.
PRIORITY_FACTOR = 1.8

_TIER_FACTORS: Mapping[BilledTier, float] = MappingProxyType(
    {"standard": 1.0, "flex": FLEX_FACTOR, "priority": PRIORITY_FACTOR}
)
#: The tier with the highest price. An answer that reports a tier without a price is billed like
#: this tier, so that the ledger and the budget guard never count too little.
HIGHEST_PRICE_TIER: BilledTier = max(_TIER_FACTORS, key=lambda tier: _TIER_FACTORS[tier])
#: Largest size of one file of the cost ledger, bytes. It is half of the size limit of the
#: repository guard (2,000,000 bytes), so that every ledger file can go to the results branch.
MAX_LEDGER_FILE_BYTES = 1_000_000


#: The model slug of OpenRouter ends with this for the free variant of a model. Rule 5 of
#: ``CLAUDE.md`` allows paid models only, so such a slug is refused everywhere.
FREE_MODEL_SUFFIX = ":free"


def price_for(model: str, on: date, table: Sequence[Price] = PRICES) -> Price:
    """Return the price of a model on a day: the entry with the latest start that is not after the day.

    ``table`` is the price table: the Gemini prices of this module by default, or the
    table of the OpenRouter models (``load_price_table``). Raises KeyError for a model
    that is not in the table, and for a day before the first entry of the model.
    """
    entries = [price for price in table if price.model == model]
    if not entries:
        raise KeyError(f"no price for the model {model!r}")
    started = [price for price in entries if price.valid_from <= on]
    if not started:
        first_day = min(price.valid_from for price in entries)
        raise KeyError(f"no price for the model {model!r} on {on.isoformat()}: its prices start on {first_day}")
    return max(started, key=lambda price: price.valid_from)


def has_flex_tier(model: str) -> bool:
    """Tell whether the provider has a flex tier for a model (``FLEX_TIER_MODELS``)."""
    return model in FLEX_TIER_MODELS


def cost_usd(model: str, usage: Usage, tier: BilledTier, on: date) -> float:
    """Return the cost of one call in USD, at the price of the day ``on``.

    Thinking tokens count as output tokens. Cached tokens get no discount.
    A tier without a price factor raises ValueError. So does the flex tier of
    a model that has none (``FLEX_TIER_MODELS``): half the standard price
    would be a price that the provider does not have.
    """
    if tier not in _TIER_FACTORS:
        raise ValueError(f"unknown service tier: {tier!r}")
    price = price_for(model, on)
    if tier == "flex" and not has_flex_tier(model):
        raise ValueError(f"the model {model} has no flex tier, so a flex call of it has no price")
    standard_usd = (
        usage.input_tokens * price.input_usd_per_million
        + (usage.output_tokens + usage.thought_tokens) * price.output_usd_per_million
    ) / 1e6
    return standard_usd * _TIER_FACTORS[tier]


class TokenEstimate(Record):
    """Expected tokens of one request of a step. The output count includes the thinking tokens."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


#: Token estimates per step (``meta.step``), used until real token counts are measured.
#:
#: The estimate of a video step counts its text only: the tokens of its video parts are added
#: from the video rate of the model (``video_input_tokens``). The output of ``people_video`` is
#: the estimate of one frame and is counted once for each target frame of the request
#: (``PER_FRAME_OUTPUT_STEPS``). The text counts of the two video steps are estimated, not
#: measured: about one token per four characters of the prompt, with room for a longer kit legend
#: (people_video: 1,843 characters with a kit legend of 136 characters and 5 target times, about
#: 460 tokens; kit_legend_video: 1,156 characters, about 290 tokens). The output of a video
#: frame is taken to be the output of an image frame; the probe measures it.
DEFAULT_TOKEN_ESTIMATES: Mapping[str, TokenEstimate] = MappingProxyType(
    {
        "people": TokenEstimate(input_tokens=1500, output_tokens=1400),
        "landmarks": TokenEstimate(input_tokens=2200, output_tokens=800),
        "number": TokenEstimate(input_tokens=2500, output_tokens=400),
        "kit_legend": TokenEstimate(input_tokens=3600, output_tokens=600),
        "people_video": TokenEstimate(input_tokens=600, output_tokens=1400),
        "kit_legend_video": TokenEstimate(input_tokens=500, output_tokens=600),
    }
)
#: Steps whose output estimate is the output of one target frame: it is counted once for each
#: entry of the context list ``frames`` of the request.
PER_FRAME_OUTPUT_STEPS = frozenset({"people_video"})


@dataclass(frozen=True)
class VideoRate:
    """What video input costs for one model from one day on.

    A video part of a request costs ``sampled frames x tokens_per_frame[resolution]``
    video tokens and ``seconds of its range x audio_tokens_per_second`` sound tokens.
    The sound is counted for every part, because the estimate does not know whether a
    file has a sound track. ``default_fps`` is the sampling rate of the provider for a
    part that names none. The video tokens cost ``video_usd_per_million`` and the
    sound tokens ``audio_usd_per_million``; None stands for the input price of the
    model on the same day. ``takes_clips`` says whether the provider takes a range
    and a sampling rate of a part; a provider without it gets the whole file.
    """

    model: str
    valid_from: date
    tokens_per_frame: Mapping[Resolution, int]
    audio_tokens_per_second: float
    default_fps: float
    takes_clips: bool
    video_usd_per_million: float | None = None
    audio_usd_per_million: float | None = None


#: Media resolution -> tokens per video frame of the Gemini 4 and Gemini 3 models, from the
#: media-resolution page (read on 2026-11-23, published, not measured): 70 at low, medium and
#: unspecified, 280 at high. The video page gives 66 and 258; the larger value of each pair is
#: taken. The tokens page gives "263 tokens per second" for video in general: that value is not
#: used, and the first doctor check of video input measures the count (docs/DECISIONS.md,
#: entry of 2026-11-23).
_GEMINI_FRAME_TOKENS: Mapping[Resolution, int] = MappingProxyType({"low": 70, "medium": 70, "high": 280})
#: Sound tokens per second of Gemini: the media-resolution page gives 25, the tokens page 32;
#: the larger value is taken (published, not measured).
_GEMINI_AUDIO_TOKENS_PER_SECOND = 32.0
#: Default sampling rate of Gemini: "extracting frames at 1 FPS" (video page, published).
_GEMINI_DEFAULT_FPS = 1.0

#: The Gemini models that take video, with their video rates. The pricing page (read on
#: 2026-11-23) gives one input price for every input of these models, with no separate price for
#: video or sound, so video and sound tokens cost the input price. The robotics model has no
#: rate: its page names no media resolution table for video, so it takes no video here.
GEMINI_VIDEO_RATES: tuple[VideoRate, ...] = tuple(
    VideoRate(
        model,
        _TABLE_START,
        _GEMINI_FRAME_TOKENS,
        _GEMINI_AUDIO_TOKENS_PER_SECOND,
        _GEMINI_DEFAULT_FPS,
        takes_clips=True,
    )
    for model in ("gemini-4-argon", "gemini-4-carbon", "gemini-4-hydrogen", "gemini-3.8-flash")
)


def video_rate_for(model: str, on: date, table: Sequence[VideoRate]) -> VideoRate:
    """Return the video rate of a model on a day: the entry with the latest start that is not after the day.

    Raises KeyError for a model without a video rate (it takes no video) and for a day
    before its first rate.
    """
    entries = [rate for rate in table if rate.model == model]
    if not entries:
        raise KeyError(f"the model {model!r} has no video rate, so it takes no video input here")
    started = [rate for rate in entries if rate.valid_from <= on]
    if not started:
        first_day = min(rate.valid_from for rate in entries)
        raise KeyError(f"no video rate for the model {model!r} on {on.isoformat()}: its rates start on {first_day}")
    return max(started, key=lambda rate: rate.valid_from)


def video_rates_of(provider: Provider, openrouter_video_rates: Sequence[VideoRate]) -> Sequence[VideoRate]:
    """Return the video rate table of a provider: ``GEMINI_VIDEO_RATES``, or the OpenRouter rates of the price file."""
    return GEMINI_VIDEO_RATES if provider == DEFAULT_PROVIDER else openrouter_video_rates


def video_input_tokens(part: VideoPart, rate: VideoRate) -> tuple[int, int]:
    """Return the expected video tokens and sound tokens of one video part.

    The video tokens are the sampled frames of the range (``VideoPart.sampled_frames``
    at the rate of the part, or at ``rate.default_fps``) times the tokens per frame of
    its resolution. The sound tokens are the seconds of the range times
    ``rate.audio_tokens_per_second``, rounded up. A resolution without a token count
    raises KeyError.
    """
    if part.resolution not in rate.tokens_per_frame:
        raise KeyError(f"the video rate of {rate.model} has no token count for the resolution {part.resolution!r}")
    start, end = part.range_s
    frames = part.sampled_frames(rate.default_fps)
    return frames * rate.tokens_per_frame[part.resolution], math.ceil((end - start) * rate.audio_tokens_per_second)


def _video_usd(request: VLMRequest, input_price: float, on: date, video_rates: Sequence[VideoRate]) -> float:
    """Return the expected cost of the video parts of a request, at standard prices. KeyError: no video rate."""
    total = 0.0
    for part in request.videos:
        rate = video_rate_for(request.model, on, video_rates)
        video_tokens, audio_tokens = video_input_tokens(part, rate)
        video_price = input_price if rate.video_usd_per_million is None else rate.video_usd_per_million
        audio_price = input_price if rate.audio_usd_per_million is None else rate.audio_usd_per_million
        total += (video_tokens * video_price + audio_tokens * audio_price) / 1e6
    return total


def _output_count(request: VLMRequest) -> int:
    """Return how many times the output estimate of the step of a request counts: the target frames, or 1."""
    if request.meta.step not in PER_FRAME_OUTPUT_STEPS:
        return 1
    frames = request.context.get("frames")
    if not isinstance(frames, list) or not frames:
        raise KeyError(f"a {request.meta.step} request needs the context list 'frames' for its estimate")
    return len(frames)


def _estimated_usd(
    request: VLMRequest,
    estimates: Mapping[str, TokenEstimate],
    on: date,
    openrouter_prices: Sequence[Price],
    openrouter_video_rates: Sequence[VideoRate],
) -> float:
    """Return the expected cost of one request from the token estimate of its step and its video parts."""
    step = request.meta.step
    if step not in estimates:
        raise KeyError(f"no token estimate for the step {step!r}")
    estimate = estimates[step]
    usage = Usage(input_tokens=estimate.input_tokens, output_tokens=estimate.output_tokens * _output_count(request))
    if request.provider == DEFAULT_PROVIDER:
        text_usd = cost_usd(request.model, usage, request.service_tier, on)
        if not request.videos:
            return text_usd
        price = price_for(request.model, on)
        video_usd = _video_usd(request, price.input_usd_per_million, on, GEMINI_VIDEO_RATES)
        return text_usd + video_usd * _TIER_FACTORS[request.service_tier]
    if request.service_tier != "standard":
        raise ValueError(f"an OpenRouter request has no service tier {request.service_tier!r}: use standard")
    price = price_for(request.model, on, openrouter_prices)
    text_usd = (
        usage.input_tokens * price.input_usd_per_million + usage.output_tokens * price.output_usd_per_million
    ) / 1e6
    return text_usd + _video_usd(request, price.input_usd_per_million, on, openrouter_video_rates)


def estimate_usd(
    requests: Iterable[VLMRequest],
    estimates: Mapping[str, TokenEstimate],
    on: date,
    *,
    openrouter_prices: Sequence[Price] = (),
    openrouter_video_rates: Sequence[VideoRate] = (),
) -> float:
    """Return the expected cost in USD of the requests, before they are sent.

    ``estimates`` maps a step (``meta.step``) to its token estimate. Each request
    is priced with its own model and service tier, at the price of the day ``on``.
    A Gemini request takes the table ``PRICES``, an OpenRouter request the table
    ``openrouter_prices`` (``load_price_table``). A step without an estimate and a
    model without a price raise KeyError. A request that asks for the flex tier of
    a model without one raises ValueError (``cost_usd``), and so does an OpenRouter
    request for any tier but the standard one.

    The video parts of a request add their tokens at the video rate of the model
    (``video_input_tokens``): ``GEMINI_VIDEO_RATES`` for Gemini, and
    ``openrouter_video_rates`` (``load_video_rates``) for OpenRouter. A request with
    a video part for a model without a video rate raises KeyError: the model takes no
    video, so the request has no estimate.
    """
    return math.fsum(
        _estimated_usd(request, estimates, on, openrouter_prices, openrouter_video_rates) for request in requests
    )


class _PriceEntry(Record):
    """One dated price of one model in the price table file."""

    valid_from: date
    input_usd_per_million: float = Field(gt=0, allow_inf_nan=False)
    output_usd_per_million: float = Field(gt=0, allow_inf_nan=False)


class _VideoRateEntry(Record):
    """One dated video rate of one model in the price table file (``VideoRate``).

    An OpenRouter model has one number of tokens per frame for every resolution: the
    chat completions API sends no resolution with a video.
    """

    valid_from: date
    tokens_per_frame: int = Field(gt=0)
    audio_tokens_per_second: float = Field(ge=0, allow_inf_nan=False)
    frames_per_second: float = Field(gt=0, allow_inf_nan=False)
    video_usd_per_million: float = Field(gt=0, allow_inf_nan=False)
    audio_usd_per_million: float = Field(gt=0, allow_inf_nan=False)


class _PricedModel(Record):
    """One model of the price table file with its dated prices, and its video rates when it takes video."""

    model: str = Field(min_length=1)
    prices: tuple[_PriceEntry, ...] = Field(min_length=1)
    video: tuple[_VideoRateEntry, ...] = ()


class _PriceTableFile(Record):
    """The content of the price table file of the OpenRouter models."""

    models: tuple[_PricedModel, ...] = ()


def load_price_table(path: Path) -> tuple[Price, ...]:
    """Read the price table of the OpenRouter models (YAML) and return its prices.

    The file holds ``models``, a list of models. Each one has its ``model`` slug and
    ``prices``, a list of entries with ``valid_from`` (a day), ``input_usd_per_million``
    and ``output_usd_per_million``. Every price must be above 0: a model without a
    price for its calls is a free model, which rule 5 does not allow. So a slug with
    the free suffix (``FREE_MODEL_SUFFIX``) is refused too, and so is a model or a start
    day that the file names two times. Wrong content raises ValueError with the path.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    try:
        table = _PriceTableFile.model_validate({} if data is None else data)
    except ValueError as error:
        raise ValueError(f"{path}: the price table is not valid: {error}") from error
    prices: list[Price] = []
    seen: set[str] = set()
    for entry in table.models:
        if entry.model.endswith(FREE_MODEL_SUFFIX):
            raise ValueError(f"{path}: {entry.model} is a free model, and only paid models are allowed")
        if entry.model in seen:
            raise ValueError(f"{path}: the model {entry.model} is listed two times")
        seen.add(entry.model)
        days = [price.valid_from for price in entry.prices]
        if len(set(days)) != len(days):
            raise ValueError(f"{path}: the model {entry.model} has two prices with the same start day")
        prices.extend(
            Price(entry.model, price.valid_from, price.input_usd_per_million, price.output_usd_per_million)
            for price in entry.prices
        )
    return tuple(prices)


def load_video_rates(path: Path) -> tuple[VideoRate, ...]:
    """Read the video rates of the OpenRouter models from the price table file (YAML).

    A model takes video when its entry has a ``video`` list. Each entry of the list has
    ``valid_from``, ``tokens_per_frame``, ``audio_tokens_per_second``,
    ``frames_per_second`` (the sampling of the provider: an OpenRouter request sends no
    rate), ``video_usd_per_million`` and ``audio_usd_per_million``. An OpenRouter rate
    never takes a range or a sampling rate (``VideoRate.takes_clips``): the video page
    says that only Gemini reads them. Wrong content raises ValueError with the path; a
    file that ``load_price_table`` refuses is refused here too.
    """
    load_price_table(path)
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    table = _PriceTableFile.model_validate({} if data is None else data)
    rates: list[VideoRate] = []
    for entry in table.models:
        days = [rate.valid_from for rate in entry.video]
        if len(set(days)) != len(days):
            raise ValueError(f"{path}: the model {entry.model} has two video rates with the same start day")
        rates.extend(
            VideoRate(
                entry.model,
                rate.valid_from,
                MappingProxyType(dict.fromkeys(VIDEO_RESOLUTIONS, rate.tokens_per_frame)),
                rate.audio_tokens_per_second,
                rate.frames_per_second,
                takes_clips=False,
                video_usd_per_million=rate.video_usd_per_million,
                audio_usd_per_million=rate.audio_usd_per_million,
            )
            for rate in entry.video
        )
    return tuple(rates)


def _utc_time(text: str) -> datetime:
    """Read an ISO 8601 time and return it in UTC. A time without an offset is already UTC."""
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"not an ISO 8601 time: {text!r}") from error
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def billed_usd(response: VLMResponse) -> float:
    """Return the cost of an answer in USD.

    An answer that carries the cost that the API reported (``reported_usd``, an
    OpenRouter answer) costs that amount. Any other answer costs its tokens at the
    price of its model and of its billed tier, on the UTC day of the answer
    (``response.created_at``). An answer whose model has no price on that day
    raises KeyError.
    """
    if response.reported_usd is not None:
        return response.reported_usd
    return cost_usd(response.model, response.usage, response.service_tier, _utc_time(response.created_at).date())


def _spend_of_line(line: str, number: int, path: Path) -> tuple[str, float]:
    """Return the run id and the USD of one ledger line.

    A damaged line raises ValueError. It is never skipped, because the totals
    guard real money. A line with a negative cost is damaged too: no answer
    has one, and such a line would lower the total that the hard stop reads.
    """
    try:
        # Whole numbers are read as floats, so that a cost written as 2 is a number like 2.0.
        entry = json.loads(line, parse_int=float)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: line {number} is not valid JSON") from error
    if not isinstance(entry, dict):
        raise ValueError(f"{path}: line {number} is not a ledger entry")
    run_id, usd = entry.get("run_id"), entry.get("usd")
    if not isinstance(run_id, str) or not isinstance(usd, float) or not math.isfinite(usd):
        raise ValueError(f"{path}: line {number} is not a ledger entry")
    if usd < 0:
        raise ValueError(f"{path}: line {number} has a negative cost, which no answer has")
    return run_id, usd


def ledger_file_number(first_name: str, name: str) -> int | None:
    """Return the number of a file of a cost ledger, or None when the name is no file of that ledger.

    ``first_name`` is the name of the first file of the ledger, for example
    ``ledger.jsonl``. That file has the number 1. A later file has its number
    between the stem and the ending of that name: ``ledger-0002.jsonl`` has
    the number 2. This is the one place that knows the names of the ledger
    files: ``Ledger`` and the listing of the ``sync`` command both ask it.
    """
    if name == first_name:
        return 1
    first = Path(first_name)
    later_name = re.fullmatch(re.escape(first.stem) + r"-(\d+)" + re.escape(first.suffix), name)
    return int(later_name.group(1)) if later_name else None


class Ledger:
    """The cost record of all paid answers: JSON lines, one line per answer.

    ``path`` is the first file of the ledger. A file takes lines until it holds
    ``MAX_LEDGER_FILE_BYTES``. The next line then starts a new file next to it:
    ``ledger.jsonl``, ``ledger-0002.jsonl``, ``ledger-0003.jsonl`` and so on.
    So no file grows above the size limit of the repository guard, and new
    answers change the last file only. Processes that write at the same moment
    can pass the size by a few lines.

    Lines are only added, never changed. Totals are read from all files each
    time, so they include the lines of earlier runs and of other processes.
    The files must be a complete series, with the numbers 1 to n: a total
    raises ``ValueError`` when a file of the series is missing, because its
    lines would be missing in the total that the budget guard reads. One
    instance can be used from several threads.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def files(self) -> list[Path]:
        """Return the files of the ledger that exist, in the order in which they were filled."""
        return [path for _, path in self._numbered_files()]

    def record(self, run_id: str, request: VLMRequest, response: VLMResponse) -> float:
        """Add one paid answer to the ledger and return its cost in USD.

        The request says where the answer belongs: clip, step, frame, item and
        repeat index. ``request`` in the line is the cache key of the request:
        two lines with the same value are two payments for the same question,
        for example after a rejected answer or a removed cassette. A repeat
        run has another repeat index and other values.

        The response says what was billed: its model, service tier and tokens,
        at the price of the UTC day of the answer (``response.created_at``),
        and how many attempts the call needed. An OpenRouter answer costs what
        its API reported (``billed_usd``). The line names the provider of the
        request and, for OpenRouter, the provider that served the call. An
        answer whose model has no price on that day raises KeyError, and an
        OpenRouter answer without a reported cost raises ValueError: no line is
        written for either.

        A file whose last line has no line end gets one before the new line,
        so that two answers never share a line. A warning is logged then.
        """
        if request.provider != DEFAULT_PROVIDER and response.reported_usd is None:
            raise ValueError(
                f"the {request.provider} answer for {request.model} has no reported cost, so it cannot be priced"
            )
        usd = billed_usd(response)
        entry = {
            "time": _utc_time(response.created_at).isoformat(),
            "run_id": run_id,
            "clip_id": request.meta.clip_id,
            "step": request.meta.step,
            "frame": request.meta.frame,
            "item": request.meta.item,
            "repeat": request.meta.repeat,
            "request": request.cache_key(),
            "provider": request.provider,
            "model": response.model,
            "served_by": response.served_by,
            "service_tier": response.service_tier,
            "attempts": response.attempts,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "thought_tokens": response.usage.thought_tokens,
            "usd": usd,
        }
        line = (json.dumps(entry) + "\n").encode("utf-8")
        with self._lock:
            target = self._file_for_line(len(line))
            # The file is opened for reading and appending: every write goes to its end.
            with target.open("a+b") as stream:
                if stream.seek(0, os.SEEK_END) > 0:
                    stream.seek(-1, os.SEEK_END)
                    if stream.read(1) != b"\n":
                        logger.warning(
                            "The ledger file %s did not end with a line end. One is added before the new line.",
                            target,
                        )
                        line = b"\n" + line
                stream.write(line)
        return usd

    def check_writable(self) -> None:
        """Raise ``OSError`` when the next line could not be added to the ledger. Nothing is created or changed.

        The folder of the ledger must exist or be one that can be made, and it
        must take a new file. The last file of the ledger must be a file that
        can be written. ``BudgetGuard.start_run`` calls this before the first
        paid call of a run: an answer whose cost cannot be recorded must not
        be paid for.
        """
        folder = next(parent for parent in self.path.parents if parent.exists())
        if not (folder.is_dir() and os.access(folder, os.W_OK | os.X_OK)):
            raise PermissionError(errno.EACCES, "the folder of the cost ledger cannot be written", str(folder))
        files = self.files()
        if files and not (files[-1].is_file() and os.access(files[-1], os.W_OK)):
            raise PermissionError(errno.EACCES, "the cost ledger cannot be written", str(files[-1]))

    def total_usd(self) -> float:
        """Return the cost of all answers in the ledger, in USD.

        A damaged line and an incomplete series of files raise ValueError.
        """
        return math.fsum(usd for _, usd in self._spend())

    def run_usd(self, run_id: str) -> float:
        """Return the cost of the answers of one run, in USD. An unknown run costs 0.

        A damaged line and an incomplete series of files raise ValueError.
        """
        return math.fsum(usd for run, usd in self._spend() if run == run_id)

    def _numbered_files(self) -> list[tuple[int, Path]]:
        """Return the existing files of the ledger with their numbers, in increasing order. The first file is 1."""
        try:
            names = os.listdir(self.path.parent)
        except FileNotFoundError:
            return []
        numbered = [
            (number, name) for name in names if (number := ledger_file_number(self.path.name, name)) is not None
        ]
        return [(number, self.path.with_name(name)) for number, name in sorted(numbered)]

    def _complete_series(self) -> list[Path]:
        """Return the files of the ledger in the order of their numbers, or raise ValueError when one is missing.

        The numbers of the files must be 1 to n, each one time. A ledger whose
        first file is gone, or with a gap in its numbers, has lost the lines of
        a whole file: about 1,000,000 bytes of paid answers that the hard stop
        would not count. The error names what was found and the repair.
        """
        numbered = self._numbered_files()
        if [number for number, _ in numbered] != list(range(1, len(numbered) + 1)):
            found = ", ".join(path.name for _, path in numbered)
            raise ValueError(
                f"{self.path.parent}: the files of the cost ledger are not a complete series. Found: {found}. "
                f"Expected: {self.path.name}, then {self.path.stem}-0002{self.path.suffix} and so on, with no "
                "number missing and none two times. The costs of a missing file would be missing in the total "
                "that the budget guard reads. Bring the missing file back, for example from the results branch, "
                "before the next paid command."
            )
        return [path for _, path in numbered]

    def _file_for_line(self, n_bytes: int) -> Path:
        """Return the file that takes the next line of ``n_bytes`` bytes. The caller holds the lock.

        It is the last file of the ledger, or a new file with the next number
        when the last file is full. A file is full when the line would make it
        larger than ``MAX_LEDGER_FILE_BYTES``. A ledger without a file starts
        with ``path``, and its folder is made here.
        """
        files = self._numbered_files()
        if not files:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            return self.path
        number, last = files[-1]
        if last.stat().st_size + n_bytes <= MAX_LEDGER_FILE_BYTES:
            return last
        return self.path.with_name(f"{self.path.stem}-{number + 1:04d}{self.path.suffix}")

    def _spend(self) -> list[tuple[str, float]]:
        """Return the run id and the USD of every line of every file. A ledger without a file has no lines.

        The files must be a complete series (``_complete_series``).
        """
        with self._lock:
            texts = [(path, path.read_text(encoding="utf-8")) for path in self._complete_series()]
        spend: list[tuple[str, float]] = []
        for path, text in texts:
            lines = enumerate(text.splitlines(), start=1)
            spend.extend(_spend_of_line(line, number, path) for number, line in lines if line.strip())
        return spend


class BudgetExceeded(RuntimeError):
    """A run, or all runs together, spent more than allowed."""


@dataclass(frozen=True)
class _RunLimit:
    """The spend limit of one started run."""

    usd: float
    #: The words that name the limit in an error.
    named: str
    #: Decimals of the spent amount in that error. A run stops after the answer that passes its
    #: limit, so the spent amount and a spend limit can be the same at two decimals.
    decimals: int


class BudgetGuard:
    """Stops paid work when it costs too much.

    A run may spend at most ``run_factor`` times its estimate, or at most its
    own spend limit when it was started with one. The limit of a run counts
    the spend of that run only. All runs in the ledger together must stay
    under ``hard_stop_usd``.
    """

    def __init__(self, ledger: Ledger, *, hard_stop_usd: float = 135.0, run_factor: float = 1.3) -> None:
        if not hard_stop_usd > 0:
            raise ValueError("hard_stop_usd must be positive")
        if not run_factor > 0:
            raise ValueError("run_factor must be positive")
        self._ledger = ledger
        self._hard_stop_usd = hard_stop_usd
        self._run_factor = run_factor
        self._limits: dict[str, _RunLimit] = {}

    def start_run(self, run_id: str, estimate_usd: float, *, limit_usd: float | None = None) -> None:
        """Register a run with its estimated cost in USD, before its first paid call.

        The run may spend ``run_factor`` times its estimate. ``limit_usd``
        replaces that limit for a run that has its own spend limit, for
        example the contract checks of gate G0. It must be a number above 0.

        The ledger must be able to take the answers of the run: ``OSError`` is
        raised when it cannot be written (``Ledger.check_writable``). The guard
        sees only what the ledger holds, so a run whose answers cannot be
        recorded must not start.
        """
        if not math.isfinite(estimate_usd) or estimate_usd < 0:
            raise ValueError(f"the estimate of a run must be a number that is not negative, got {estimate_usd!r}")
        if limit_usd is None:
            named = f"{self._run_factor} times its estimate of {estimate_usd:.2f} USD"
            limit = _RunLimit(self._run_factor * estimate_usd, named, decimals=2)
        elif math.isfinite(limit_usd) and limit_usd > 0:
            limit = _RunLimit(limit_usd, f"its spend limit of {limit_usd:.3f} USD", decimals=3)
        else:
            raise ValueError(f"the spend limit of a run must be a number above 0, got {limit_usd!r}")
        self._ledger.check_writable()
        self._limits[run_id] = limit

    def check(self, run_id: str) -> None:
        """Raise ``BudgetExceeded`` when the run or the whole ledger is over its limit.

        The ledger is over its limit when its total reaches ``hard_stop_usd``.
        The run is over its limit when it spent more than the limit that
        ``start_run`` gave it: ``run_factor`` times its estimate, or its own
        spend limit. A run that was not started raises ValueError.
        """
        if run_id not in self._limits:
            raise ValueError(f"the run {run_id!r} has no estimate: call start_run first")
        total = self._ledger.total_usd()
        if total >= self._hard_stop_usd:
            raise BudgetExceeded(
                f"hard stop: the ledger total is {total:.2f} USD, the limit is {self._hard_stop_usd:.2f} USD"
            )
        spent = self._ledger.run_usd(run_id)
        limit = self._limits[run_id]
        if spent > limit.usd:
            raise BudgetExceeded(f"the run {run_id!r} spent {spent:.{limit.decimals}f} USD, more than {limit.named}")
