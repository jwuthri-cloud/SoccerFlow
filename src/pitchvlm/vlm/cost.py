"""Cost of model calls: dated prices, estimates, the cost ledger and the budget guard.

Prices are in USD per million tokens. Thinking tokens are billed at the output
price. The ``flex`` service tier costs ``FLEX_FACTOR`` times the standard price
and the ``priority`` tier ``PRIORITY_FACTOR`` times. Only the models of
``FLEX_TIER_MODELS`` have a flex tier: a flex call of another model has no
price. A price is valid from its ``valid_from`` day (UTC) until the next entry
of the same model starts.
"""

from __future__ import annotations

import errno
import json
import logging
import math
import os
import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType

from pydantic import Field

from pitchvlm.schemas import Record
from pitchvlm.vlm.types import BilledTier, Usage, VLMRequest, VLMResponse

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


def price_for(model: str, on: date) -> Price:
    """Return the price of a model on a day: the entry with the latest start that is not after the day.

    Raises KeyError for a model that is not in the table, and for a day before
    the first entry of the model.
    """
    entries = [price for price in PRICES if price.model == model]
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
DEFAULT_TOKEN_ESTIMATES: Mapping[str, TokenEstimate] = MappingProxyType(
    {
        "people": TokenEstimate(input_tokens=1500, output_tokens=1400),
        "landmarks": TokenEstimate(input_tokens=2200, output_tokens=800),
        "number": TokenEstimate(input_tokens=2500, output_tokens=400),
        "kit_legend": TokenEstimate(input_tokens=3600, output_tokens=600),
    }
)


def _estimated_usd(request: VLMRequest, estimates: Mapping[str, TokenEstimate], on: date) -> float:
    """Return the expected cost of one request from the token estimate of its step."""
    step = request.meta.step
    if step not in estimates:
        raise KeyError(f"no token estimate for the step {step!r}")
    estimate = estimates[step]
    usage = Usage(input_tokens=estimate.input_tokens, output_tokens=estimate.output_tokens)
    return cost_usd(request.model, usage, request.service_tier, on)


def estimate_usd(requests: Iterable[VLMRequest], estimates: Mapping[str, TokenEstimate], on: date) -> float:
    """Return the expected cost in USD of the requests, before they are sent.

    ``estimates`` maps a step (``meta.step``) to its token estimate. Each request
    is priced with its own model and service tier, at the price of the day ``on``.
    A step without an estimate raises KeyError. A request that asks for the flex
    tier of a model without one raises ValueError (``cost_usd``).
    """
    return math.fsum(_estimated_usd(request, estimates, on) for request in requests)


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
    """Return the cost of an answer in USD: its tokens at the price of its model and of its billed tier.

    The price is the one of the UTC day of the answer (``response.created_at``).
    An answer whose model has no price on that day raises KeyError.
    """
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
        and how many attempts the call needed. An answer whose model has no
        price on that day raises KeyError, and no line is written.

        A file whose last line has no line end gets one before the new line,
        so that two answers never share a line. A warning is logged then.
        """
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
            "model": response.model,
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
