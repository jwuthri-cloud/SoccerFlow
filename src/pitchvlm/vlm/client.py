"""The real model backend: Gemini through the Interactions API.

This is the only module that may import the model SDK (``google-genai``). The
import is lazy: it runs when the first request needs a client. So this module
imports, and its pure helpers work, on a machine that has no SDK.

How a request is sent (API pages and SDK source read again on 2026-11-14, SDK 3.3.0):

- One call of ``client.interactions.create(...)`` per attempt, and one HTTP
  request per call. The call passes ``automatic_continuation=False`` for this.
  By default the SDK continues an answer that stops at the token cap of one
  request (status ``requires_continuation``, 32,768 tokens of thinking and
  output on ``gemini-4-argon`` and ``gemini-4-carbon``) with further billed
  requests. When one of them fails, the SDK raises
  ``ContinuationInterruptedError``, which has no status code: the backend
  would take it for a call that was not billed, and the requests that were
  billed before it would get no ledger line. Without the continuation such
  an answer comes back with its own usage, and it is a billed answer that
  cannot be used (``BilledAnswerError``).
- ``store=False`` always. The provider keeps an interaction for 55 days otherwise.
- No sampling parameter. Gemini 4 models ignore temperature, top_p and top_k.
- The answer is JSON text that follows the request's schema.
- Images go inline as base64 text, each with its own resolution.
- The SDK makes one attempt per call. This module does the retries, so that it
  can count the attempts. This needs SDK 3.0 or later: an older SDK reads the
  setting as one repeat and sends a failed call a second time by itself.
- The client timeout is in milliseconds. The timeout of one call is in seconds.
  It depends on the service tier: a flex request can wait in a queue of the
  provider, so it gets ``FLEX_TIMEOUT_S`` and not ``STANDARD_TIMEOUT_S``.

Which failed call is tried again:

- A status code 408, 409, 429 or 5xx: yes, at most ``max_attempts`` attempts in
  total. The provider does not bill a call that it refused with an error status.
- A status code 503 or 429 of a request for the flex tier: yes, for as long as
  ``flex_busy_wait_s`` seconds after the first attempt. These are the normal
  answers of the flex tier while it has no capacity. The provider never moves
  such a request to the standard tier, and the client must try again with
  backoff. So these attempts do not use up the ``max_attempts`` of the request.
  When the waiting time is used up, the request ends with ``FlexBusyError``.
- A lost connection (the SDK error ``APIConnectionError``, no status code): yes,
  within ``max_attempts``. A run of about 180 calls must not stop for one dropped
  connection. The lost attempt can still be billed when the request reached the
  provider; its log line says so.
- A client timeout (``APITimeoutError``): no. The provider had the request for
  the whole timeout, so it can still finish it and bill it, and a new attempt
  could pay a second time.
- An SDK error with a success status (2xx): no. The provider answered, but the
  SDK could not read the body of the answer (``APIResponseValidationError``).
- Any other error: no.

A call that ends by a timeout, by a lost connection or by an answer that cannot
be read raises ``UnconfirmedCallError`` (``vlm.types``): nobody knows whether it
is billed, and the cost ledger has no line for it.

A stop request ends the waits of a request. The backend takes an event
(``stop``). While it is set, no attempt is sent: a request that waits for a new
attempt or for room in the spend window ends with a ``BackendError`` that says
so. A call that is in flight is not cut off: its answer comes back as usual.

The token usage is normalised: ``Usage.output_tokens`` never contains thinking
tokens. The cost of every finished call goes into a sliding ten-minute window,
and a new call waits while that window is full. The cost follows the service
tier that the answer reports. A reported tier without a price counts as the
most expensive tier. A request for the flex tier of a model that has none
(``cost.FLEX_TIER_MODELS``) is not sent, because its cost could not be priced.

An answer that was billed but cannot be used raises ``BilledAnswerError``
(``vlm.types``). This error carries the cost record of the call. The wrapper
that reports new answers to the cost ledger reports this record too, and it is
the only place that does (see ``vlm.cache.CachedBackend``).
"""

from __future__ import annotations

import base64
import copy
import importlib.metadata
import logging
import math
import os
import random
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any, Literal, Protocol, get_args

from pitchvlm.vlm.cost import HIGHEST_PRICE_TIER, cost_usd, has_flex_tier, price_for
from pitchvlm.vlm.types import (
    BackendError,
    BilledAnswerError,
    BilledTier,
    FlexBusyError,
    ImagePart,
    ServiceTier,
    TextPart,
    UnconfirmedCallError,
    Usage,
    VLMRequest,
    VLMResponse,
)

logger = logging.getLogger(__name__)

#: Name of the SDK distribution on the package index.
SDK_DISTRIBUTION = "google-genai"
#: The only interaction status that carries a full answer.
COMPLETED_STATUS = "completed"
#: Status codes that are worth a new attempt, next to every 5xx code.
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})
#: Status codes with which the flex tier answers a request while it has no capacity (the flex page
#: of the provider, read on 2026-11-14). The flex tier is best effort: the provider never moves such
#: a request to the standard tier, and it asks the client to try again with a growing wait.
FLEX_BUSY_STATUS_CODES = frozenset({429, 503})
#: How long a request for the flex tier waits for capacity, seconds, counted from its first attempt.
#: It is the upper end of the latency target of the flex tier (1 to 15 minutes).
FLEX_BUSY_WAIT_S = 900.0
#: Class name of the SDK error for a call that got no answer from the server (a refused or lost connection).
SDK_CONNECTION_ERROR = "APIConnectionError"
#: Class name of the SDK error for a call whose client timeout ran out. It is a subclass of the connection error.
SDK_TIMEOUT_ERROR = "APITimeoutError"
#: Client timeout of one call with the standard service tier, seconds.
STANDARD_TIMEOUT_S = 300.0
#: Client timeout of one call with the flex service tier, seconds. A flex request can wait in a
#: queue of the provider: the latency target of the flex tier is 1 to 15 minutes, and the provider
#: recommends a client timeout of 10 minutes or more.
FLEX_TIMEOUT_S = 900.0
#: What the error and the log line of a call without an answer that can be read say about its cost.
UNCONFIRMED_NOTE = (
    "this call may still be billed: the provider can finish a request that it got, "
    "and the cost ledger has no line for it"
)
#: Nominal wait before the first retry, seconds. It doubles with every retry.
BACKOFF_BASE_S = 10.0
#: Longest wait between two attempts, seconds.
BACKOFF_CAP_S = 60.0
#: Length of the provider's spend window, seconds.
SPEND_WINDOW_S = 600.0
#: Allowed gap between the reported token total and the sum of its parts, as a share of the total.
USAGE_TOLERANCE = 0.01
#: Longest SDK error text that is copied into a log line or an error message, characters.
MAX_ERROR_CHARS = 500
#: The service tiers that have a price. An answer can report one of them as the tier that served the call.
BILLED_TIERS: tuple[BilledTier, ...] = get_args(BilledTier)

#: How the reported output count relates to the thinking tokens.
UsageAccounting = Literal["separate", "included", "unknown"]
#: How a call can end without any answer of the server.
TransportFailure = Literal["timeout", "connection"]


class InteractionsApi(Protocol):
    """The part of the SDK client that sends one request."""

    def create(self, **kwargs: Any) -> Any: ...


class InteractionsClient(Protocol):
    """Anything with ``.interactions.create``: the SDK client, or a fake object in tests."""

    @property
    def interactions(self) -> InteractionsApi: ...


# --- Request ---------------------------------------------------------------------


def _input_item(part: TextPart | ImagePart) -> dict[str, str]:
    """Turn one request part into one input item of the Interactions API."""
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    return {
        "type": "image",
        "data": base64.b64encode(part.data).decode("ascii"),
        "mime_type": part.mime_type,
        "resolution": part.resolution,
    }


def build_interaction_kwargs(request: VLMRequest) -> dict[str, Any]:
    """Turn a request into the keyword arguments of ``client.interactions.create``.

    Text and image parts keep their order. Image bytes are sent inline as base64
    text, each image with its own resolution. The answer must be JSON that follows
    the request's schema. The call is never stored on the provider's side, and it
    carries no sampling parameter. ``request.context`` is never sent.
    """
    if not request.parts:
        raise ValueError("a request needs at least one text or image part")
    return {
        "model": request.model,
        "input": [_input_item(part) for part in request.parts],
        "generation_config": {"thinking_level": request.thinking_level},
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": copy.deepcopy(request.response_schema),
        },
        "store": False,
        "service_tier": request.service_tier,
    }


def _request_label(request: VLMRequest) -> str:
    """Name a request for log lines and error messages."""
    meta = request.meta
    return (
        f"{meta.step} request (clip {meta.clip_id}, frame {meta.frame}, item {meta.item}, "
        f"repeat {meta.repeat}, model {request.model})"
    )


# --- Token usage -----------------------------------------------------------------


def usage_accounting(
    input_tokens: int, output_tokens: int, thought_tokens: int, total_tokens: int | None
) -> UsageAccounting:
    """Say how the reported output count relates to the thinking tokens.

    ``separate``: the total is input + output + thought, so the output count has
    no thinking tokens. ``included``: the total is input + output, so the output
    count contains the thinking tokens. ``unknown``: the total fits neither sum,
    or the API gave no total.

    An exact match wins. After that, a total within ``USAGE_TOLERANCE`` of the
    first sum still counts as ``separate``, because the total can hold a few
    internal tokens.
    """
    if total_tokens is None:
        return "unknown"
    separate_sum = input_tokens + output_tokens + thought_tokens
    included_sum = input_tokens + output_tokens
    if total_tokens == separate_sum:
        return "separate"
    if total_tokens == included_sum and output_tokens >= thought_tokens:
        return "included"
    if abs(total_tokens - separate_sum) <= USAGE_TOLERANCE * total_tokens:
        return "separate"
    return "unknown"


def normalize_usage(
    *, input_tokens: int, output_tokens: int, thought_tokens: int, cached_tokens: int, total_tokens: int | None
) -> Usage:
    """Build a ``Usage`` whose output count has no thinking tokens.

    The API pages do not say whether the reported output count contains the
    thinking tokens, so the reported total decides (see ``usage_accounting``).
    When the total fits neither reading, the counts are kept as reported and a
    warning is logged. No error is raised in that case.
    """
    accounting = usage_accounting(input_tokens, output_tokens, thought_tokens, total_tokens)
    if accounting == "included":
        output_tokens -= thought_tokens
    elif accounting == "unknown":
        logger.warning(
            "Token accounting is unclear: total=%s, input=%d, output=%d, thought=%d. "
            "The counts are kept as reported, so the cost may count thinking tokens twice.",
            total_tokens,
            input_tokens,
            output_tokens,
            thought_tokens,
        )
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thought_tokens=thought_tokens,
        cached_tokens=cached_tokens,
    )


def _token_count(raw_usage: Any, name: str, label: str) -> int | None:
    """Read one token counter of the API's usage object. A counter that the API left out is None."""
    value = getattr(raw_usage, name, None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BackendError(f"{label}: the usage field {name} is not a token count: {value!r}")
    return value


def read_usage(interaction: Any, label: str) -> Usage:
    """Read the token usage of an interaction and normalise it.

    The API can leave out a counter that is zero, so a missing output, thought
    or cached counter counts as zero. The input counter is required, because
    every request has input tokens. Raises ``BackendError`` when the interaction
    has no usage at all: then its cost is unknown.
    """
    raw_usage = getattr(interaction, "usage", None)
    if raw_usage is None:
        status = getattr(interaction, "status", None)
        raise BackendError(f"{label}: the answer (status {status!r}) has no token usage, so its cost is unknown")
    input_tokens = _token_count(raw_usage, "total_input_tokens", label)
    if input_tokens is None:
        raise BackendError(f"{label}: the token usage has no total_input_tokens, so the cost is unknown")
    return normalize_usage(
        input_tokens=input_tokens,
        output_tokens=_token_count(raw_usage, "total_output_tokens", label) or 0,
        thought_tokens=_token_count(raw_usage, "total_thought_tokens", label) or 0,
        cached_tokens=_token_count(raw_usage, "total_cached_tokens", label) or 0,
        total_tokens=_token_count(raw_usage, "total_tokens", label),
    )


# --- Billed answers --------------------------------------------------------------


def billed_tier(interaction: Any, requested: ServiceTier, label: str, *, model: str) -> BilledTier:
    """Return the service tier whose price the call is billed at.

    An answer can report the tier that served the call, and the price follows
    that tier. So a reported tier that has a price (``standard``, ``flex`` or
    ``priority``) wins over the requested tier. The requested tier is used
    when the answer reports no tier. A reported tier without a price is billed
    like the most expensive tier (``cost.HIGHEST_PRICE_TIER``), so that the
    cost ledger and the budget guard never count too little. The flex tier
    has no price for a model that has no flex tier (``cost.FLEX_TIER_MODELS``),
    so the cost of a paid call can always be computed. A warning is logged
    whenever the reported tier is not the requested one.
    """
    raw = getattr(interaction, "service_tier", None)
    if raw is None:
        return requested
    # SDK 3.3.0 gives the tier as plain text. An enumeration member is read too, because the tier decides the price.
    reported = str(getattr(raw, "value", raw))
    served = next((tier for tier in BILLED_TIERS if tier == reported.lower()), None)
    if served == "flex" and not has_flex_tier(model):
        served = None
    if served == requested:
        return requested
    if served is None:
        logger.warning(
            "%s: asked for service tier %r, but the answer reports %r, which has no price for this model. "
            "The cost and the response use the most expensive tier, %r.",
            label,
            requested,
            reported,
            HIGHEST_PRICE_TIER,
        )
        return HIGHEST_PRICE_TIER
    logger.warning(
        "%s: asked for service tier %r, but the answer reports %r. "
        "The cost and the response use the reported tier.",
        label,
        requested,
        served,
    )
    return served


def read_answer_text(interaction: Any, billed: VLMResponse, label: str) -> str:
    """Return the answer text of an interaction whose call was billed.

    ``billed`` is the cost record of the call. Raises ``BilledAnswerError``
    with that record when the status is not ``completed`` (an incomplete answer
    is an error), when the text is missing or empty, or when the usage reports
    no output tokens for the text.
    """
    status = getattr(interaction, "status", None)
    if status != COMPLETED_STATUS:
        raise BilledAnswerError(f"{label}: the answer has status {status!r}, expected {COMPLETED_STATUS!r}", billed)
    text = getattr(interaction, "output_text", None)
    if not isinstance(text, str) or not text.strip():
        raise BilledAnswerError(f"{label}: the answer has no text", billed)
    if billed.usage.output_tokens + billed.usage.thought_tokens == 0:
        raise BilledAnswerError(f"{label}: the answer has text, but its usage reports no output tokens", billed)
    return text


# --- Retry -----------------------------------------------------------------------


def status_code_of(error: BaseException) -> int | None:
    """Read the HTTP status code of an SDK error, or None when the error has none.

    Errors of the Interactions API keep the code in ``status_code``. An error
    for a call that got no answer from the server (a lost connection or a
    client timeout) has the value None there.
    """
    value = getattr(error, "status_code", None)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def is_retryable_status(status: int | None) -> bool:
    """Say whether a call that failed with this status code may be tried again."""
    if status is None:
        return False
    return status in RETRYABLE_STATUS_CODES or 500 <= status <= 599


def transport_failure_of(error: BaseException) -> TransportFailure | None:
    """Say whether an SDK error stands for a call that got no answer from the server, and why.

    ``timeout``: the client timeout ran out (``APITimeoutError``). ``connection``:
    the connection was refused or lost (``APIConnectionError``). Any other
    error gives None, so that a wrong argument or a failed validation is never
    taken for a network fault.

    The two classes live in a private module of the SDK, and that module has
    moved between releases. So an error is recognised by the names of its
    classes, and no SDK module is imported for it. The timeout error is a
    subclass of the connection error, so it is tested first.
    """
    names = {cls.__name__ for cls in type(error).__mro__}
    if SDK_TIMEOUT_ERROR in names:
        return "timeout"
    if SDK_CONNECTION_ERROR in names:
        return "connection"
    return None


def is_unread_answer(status: int | None) -> bool:
    """Say whether an SDK error with this status code stands for an answer that could not be read.

    The SDK raises an error with a success status (2xx) when the provider
    answered and the body of the answer is not what the SDK can read
    (``APIResponseValidationError``). The request was done, so it can be
    billed, and its token usage is lost with the body.
    """
    return status is not None and 200 <= status <= 299


def _failure_reason(status: int | None, transport: TransportFailure | None, timeout_s: float) -> str:
    """Say in a few words why an attempt failed: how the call ended without an answer, or its status code."""
    if transport == "timeout":
        return f"no answer within the client timeout of {timeout_s:g} s"
    if transport == "connection":
        return "no answer from the server, the connection failed"
    if is_unread_answer(status):
        return f"status {status}, but the answer could not be read"
    return f"status {status}"


def backoff_delay_s(retry_index: int, rng: random.Random) -> float:
    """Return the wait before a retry in seconds. ``retry_index`` is 0 for the first retry.

    The nominal wait starts at ``BACKOFF_BASE_S``, doubles with every retry and
    stops at ``BACKOFF_CAP_S``. The result is a random value between half the
    nominal wait and the nominal wait, so that parallel callers do not retry at
    the same moment.
    """
    if retry_index < 0:
        raise ValueError("retry_index must not be negative")
    # Every exponent above 16 gives the cap. The limit only prevents a float overflow.
    nominal_s = min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2.0 ** min(retry_index, 16))
    return nominal_s * (0.5 + 0.5 * rng.random())


def _error_text(error: BaseException, secret: str) -> str:
    """Describe an error in one short text that never contains the key value."""
    text = f"{type(error).__name__}: {error}"
    if secret:
        text = text.replace(secret, "[key removed]")
    if len(text) > MAX_ERROR_CHARS:
        text = text[:MAX_ERROR_CHARS] + " ..."
    return text


# --- Spend rate ------------------------------------------------------------------


class SpendRateLimiter:
    """Keeps the cost of finished calls under a limit per sliding time window.

    The provider blocks an account that spends more than a fixed amount in any
    ten minutes. ``record`` stores the cost of each finished call.
    ``wait_for_room`` blocks while the cost inside the window is at or above
    the limit. Calls that are still in flight are not counted, so the limit
    must stay below the provider's limit. Safe to use from several threads.
    """

    def __init__(
        self,
        max_usd: float,
        *,
        window_s: float = SPEND_WINDOW_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], object] = time.sleep,
    ) -> None:
        # "not x > 0" is also true for NaN, which would switch the limit off.
        if not max_usd > 0:
            raise ValueError("max_usd must be positive")
        if not window_s > 0:
            raise ValueError("window_s must be positive")
        self._max_usd = max_usd
        self._window_s = window_s
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        #: (finish time, cost in USD) of each call inside the window, oldest first.
        self._entries: deque[tuple[float, float]] = deque()

    def record(self, usd: float) -> None:
        """Add the cost of a call that has just finished."""
        if not usd >= 0:
            raise ValueError("a cost must be a number that is not negative")
        with self._lock:
            self._entries.append((self._clock(), usd))

    def spent_usd(self) -> float:
        """Return the cost of the calls that finished inside the window."""
        with self._lock:
            self._drop_expired(self._clock())
            return self._window_total()

    def wait_for_room(self, stop: threading.Event | None = None) -> None:
        """Block until the cost inside the window is under the limit.

        With ``stop``, the wait also ends as soon as that event is set. The
        window can then still be full: the caller must not send its call.
        """
        while True:
            if stop is not None and stop.is_set():
                return
            with self._lock:
                delay_s = self._seconds_until_room(self._clock())
                spent = self._window_total()
            if delay_s <= 0.0:
                return
            logger.warning(
                "Spend limit: %.2f USD in the last %.0f s (limit %.2f USD). Waiting %.1f s before the next call.",
                spent,
                self._window_s,
                self._max_usd,
                delay_s,
            )
            self._sleep(delay_s)

    def _window_total(self) -> float:
        """Sum the stored costs. The caller holds the lock."""
        return math.fsum(usd for _, usd in self._entries)

    def _drop_expired(self, now: float) -> None:
        """Forget the calls that have left the window. The caller holds the lock."""
        while self._entries and self._entries[0][0] + self._window_s <= now:
            self._entries.popleft()

    def _seconds_until_room(self, now: float) -> float:
        """Return how long to wait until the window is under the limit. The caller holds the lock.

        The costs are added from the newest call to the oldest. The call at which
        the sum reaches the limit, and every older call, must leave the window.
        """
        self._drop_expired(now)
        newest_first_usd = 0.0
        for finished_at, usd in reversed(self._entries):
            newest_first_usd += usd
            if newest_first_usd >= self._max_usd:
                return finished_at + self._window_s - now
        return 0.0


# --- SDK client ------------------------------------------------------------------


def installed_sdk_version() -> str | None:
    """Return the installed version of the model SDK, or None when the SDK is not installed."""
    try:
        return importlib.metadata.version(SDK_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return None


def _require_price(model: str, tier: ServiceTier, on: date) -> None:
    """Raise ``BackendError`` when a call of the model at the service tier has no price, before any money is spent."""
    try:
        price_for(model, on)
    except KeyError as exc:
        raise BackendError(
            f"The model {model!r} has no price on {on.isoformat()}, so its spend cannot be limited. "
            "Add the price to the table in pitchvlm.vlm.cost first."
        ) from exc
    if tier == "flex" and not has_flex_tier(model):
        raise BackendError(
            f"The model {model!r} has no flex tier, so a flex request for it has no price and is not sent. "
            "Use the standard tier for this model."
        )


def build_sdk_client(api_key_env: str, timeout_s: float) -> InteractionsClient:
    """Build the SDK client with the key from the environment.

    The SDK is imported here, so only a real call needs it. The SDK is set to
    one attempt per call, because the backend does the retries. ``timeout_s``
    is the client timeout in seconds; the SDK gets it in milliseconds. It must
    be the longest timeout of any call: ``GeminiBackend`` gives every call its
    own timeout, and a flex call needs ``FLEX_TIMEOUT_S``. The key value goes
    to the SDK and nowhere else: an error of the SDK is raised again without
    it, in the message and in the traceback.

    ``GeminiBackend`` calls this on its first request. Other code can call it
    to wrap the real client, for example to record raw answers, and then pass
    the wrapper to ``GeminiBackend(client=...)``.
    """
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise BackendError(
            f"The environment variable {api_key_env} is not set or is empty. "
            "Set it to the API key on the machine that calls the model."
        )
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise BackendError(
            f"The model SDK ({SDK_DISTRIBUTION}) is not installed. "
            "Install the 'gemini' extra of this package on the machine that calls the model."
        ) from exc
    try:
        options = types.HttpOptions(
            timeout=round(timeout_s * 1000),
            retry_options=types.HttpRetryOptions(attempts=1),
        )
        return genai.Client(api_key=api_key, http_options=options)
    except Exception as exc:
        # The cause is dropped on purpose: only the cleaned text of the SDK error is kept.
        raise BackendError(f"The SDK client could not be created: {_error_text(exc, api_key)}") from None


# --- Backend ---------------------------------------------------------------------


class GeminiBackend:
    """Backend that answers requests with Gemini through the Interactions API.

    The key is read from the environment variable ``api_key_env`` when the first
    request is sent. It goes to the SDK client and to nothing else: this class
    does not keep it, does not log it and does not put it into an error message.

    A failed call is tried again when its status code is 408, 409, 429 or 5xx,
    and when its connection was lost, at most ``max_attempts`` times in total,
    with a growing random wait in between. Any other error ends the request at
    once, also a client timeout. Before every attempt the spend window is
    checked: when the calls that finished in the last ten minutes cost
    ``max_usd_per_10_min`` or more, the call waits.

    A request for the flex tier has more patience with the two status codes of
    a busy flex tier, 503 and 429 (``FLEX_BUSY_STATUS_CODES``). Such an attempt
    is not billed and is not one of the ``max_attempts``: the request is tried
    again, with the same growing wait, until ``flex_busy_wait_s`` seconds have
    passed since its first attempt. Then it ends with ``FlexBusyError``.

    ``timeout_s`` is the client timeout of one call with the standard service
    tier, and ``flex_timeout_s`` of one call with the flex tier, in seconds.
    The SDK client is built with the longer one.

    ``stop`` is an event that the caller sets to stop the work, for example
    after Ctrl-C. While it is set, the backend sends no attempt: a request
    that waits for a new attempt or for room in the spend window ends its wait
    at once and raises ``BackendError``. A call that is in flight goes on, and
    its answer is returned.

    ``client`` replaces the SDK client: a fake object in tests, or a wrapper
    around ``build_sdk_client(...)``. It can be anything with
    ``.interactions.create``. ``clock`` (monotonic seconds) and ``sleep`` exist
    for tests; without ``sleep`` a wait ends when the stop event is set. One
    instance can be used from several threads.
    """

    def __init__(
        self,
        *,
        api_key_env: str = "GEMINI_API_KEY",
        timeout_s: float = STANDARD_TIMEOUT_S,
        flex_timeout_s: float = FLEX_TIMEOUT_S,
        max_attempts: int = 4,
        flex_busy_wait_s: float = FLEX_BUSY_WAIT_S,
        max_usd_per_10_min: float = 8.0,
        stop: threading.Event | None = None,
        client: InteractionsClient | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], object] | None = None,
    ) -> None:
        if not api_key_env:
            raise ValueError("api_key_env must name an environment variable")
        if not timeout_s > 0:
            raise ValueError("timeout_s must be positive")
        if not flex_timeout_s > 0:
            raise ValueError("flex_timeout_s must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if not flex_busy_wait_s > 0:
            raise ValueError("flex_busy_wait_s must be positive")
        if not max_usd_per_10_min > 0:
            raise ValueError("max_usd_per_10_min must be positive")
        self._api_key_env = api_key_env
        self._timeouts_s: dict[ServiceTier, float] = {"standard": float(timeout_s), "flex": float(flex_timeout_s)}
        self._max_attempts = max_attempts
        self._flex_busy_wait_s = float(flex_busy_wait_s)
        self._client = client
        self._client_lock = threading.Lock()
        self._clock = clock
        self._stop = stop if stop is not None else threading.Event()
        # The wait on the stop event is a sleep that ends at once when the event is set.
        self._sleep: Callable[[float], object] = sleep if sleep is not None else self._stop.wait
        self._rng = random.Random()
        self._limiter = SpendRateLimiter(max_usd_per_10_min, clock=clock, sleep=self._sleep)
        self._sdk_version = installed_sdk_version()

    def generate(self, request: VLMRequest) -> VLMResponse:
        """Send one request and return the JSON answer with its normalised token usage.

        ``latency_s`` is the time of the successful attempt only. ``attempts``
        counts every attempt. ``model`` repeats the request. ``service_tier`` is
        the tier whose price is billed: the tier that the answer reports, else
        the tier of the request (``billed_tier``). The cost for the spend
        window uses that tier and the price of the UTC day of the answer, the
        same day that the cost ledger uses.

        Raises ``BackendError`` when the key or the SDK is missing, the model has
        no price, the request asks for the flex tier of a model that has none,
        the call fails, the stop event is set before an attempt is sent, or the
        token usage of the answer cannot be read. The cost of such an answer is
        unknown, so it cannot be recorded.

        Raises ``UnconfirmedCallError`` when the call ended by a client timeout,
        by a lost connection or by an answer that the SDK could not read. This
        error is a ``BackendError`` whose message says that the call may still
        be billed.

        Raises ``FlexBusyError`` when the flex tier refused the request with
        its busy answers for ``flex_busy_wait_s`` seconds. This error is a
        ``BackendError``. None of these attempts is billed.

        Raises ``BilledAnswerError`` when the token usage is known but the
        answer cannot be used: it is not completed (also an answer that stopped
        at the token cap of one request, status ``requires_continuation``), it
        has no text, or its usage reports no output tokens. This error is a
        ``BackendError`` that carries the cost record of the call. The cost is
        already in the spend window. This backend writes no cost ledger: the
        wrapper that keeps the ledger records the cost one time.

        A request without parts raises ``ValueError``.
        """
        kwargs = build_interaction_kwargs(request)
        label = _request_label(request)
        _require_price(request.model, request.service_tier, datetime.now(UTC).date())
        client = self._client_for_call()
        interaction, attempts, latency_s = self._create_with_retries(client, kwargs, label, request.service_tier)
        answered_at = datetime.now(UTC)
        usage = read_usage(interaction, label)
        tier = billed_tier(interaction, request.service_tier, label, model=request.model)
        # The cost counts also when the answer is rejected below: the call was billed.
        self._limiter.record(cost_usd(request.model, usage, tier, answered_at.date()))
        # The cost record of the call: the answer without its text.
        billed = VLMResponse(
            text="",
            usage=usage,
            model=request.model,
            service_tier=tier,
            latency_s=latency_s,
            attempts=attempts,
            sdk_version=self._sdk_version,
            created_at=answered_at.isoformat(timespec="seconds"),
        )
        text = read_answer_text(interaction, billed, label)
        return billed.model_copy(update={"text": text})

    def _client_for_call(self) -> InteractionsClient:
        """Return the client. The SDK client is built on the first call, with the longest call timeout, and kept."""
        with self._client_lock:
            if self._client is None:
                self._client = build_sdk_client(self._api_key_env, max(self._timeouts_s.values()))
            return self._client

    def _create_with_retries(
        self, client: InteractionsClient, kwargs: dict[str, Any], label: str, tier: ServiceTier
    ) -> tuple[Any, int, float]:
        """Call the API until one attempt succeeds.

        Returns the interaction, the number of attempts and the time of the
        successful attempt in seconds. Every attempt is one HTTP request, with
        the client timeout of the service tier ``tier``: the SDK does not
        continue an answer with further requests.

        An attempt is tried again after a retryable status code and after a
        lost connection, as long as the request has attempts left. It has
        ``max_attempts`` attempts. An attempt that a busy flex tier refused
        (``FLEX_BUSY_STATUS_CODES``) gives its attempt back: the request is
        tried again until ``flex_busy_wait_s`` seconds have passed since its
        first attempt, and then it raises ``FlexBusyError``.

        Any other failure that ends the request raises ``BackendError``. When
        that failure is a client timeout, a lost connection or an answer that
        could not be read, the error is an ``UnconfirmedCallError``, and a
        warning is logged, so that every such call is in the log.

        No attempt is sent while the stop event is set (``_refuse_when_stopped``).
        """
        timeout_s = self._timeouts_s[tier]
        attempt = 0
        busy_attempts = 0
        first_started: float | None = None
        while True:
            self._limiter.wait_for_room(self._stop)
            self._refuse_when_stopped(label, attempt)
            attempt += 1
            started = self._clock()
            if first_started is None:
                first_started = started
            try:
                interaction = client.interactions.create(**kwargs, timeout=timeout_s, automatic_continuation=False)
            except Exception as exc:
                status = status_code_of(exc)
                transport = None if status is not None else transport_failure_of(exc)
                unconfirmed = transport is not None or is_unread_answer(status)
                detail = _error_text(exc, os.environ.get(self._api_key_env, "").strip())
                failure = f"({_failure_reason(status, transport, timeout_s)}): {detail}"
                if unconfirmed:
                    failure += f" [{UNCONFIRMED_NOTE}]"
                if tier == "flex" and status in FLEX_BUSY_STATUS_CODES:
                    busy_attempts += 1
                    waited_s = self._clock() - first_started
                    if waited_s >= self._flex_busy_wait_s:
                        raise FlexBusyError(
                            f"{label}: the flex tier had no capacity for {waited_s:.0f} s after the first attempt "
                            f"(the request waits {self._flex_busy_wait_s:g} s for capacity). The provider refused "
                            f"all {busy_attempts} attempts that it answered in this way, so they are not billed. "
                            f"The last one was attempt {attempt} {failure}"
                        ) from exc
                    delay_s = backoff_delay_s(attempt - 1, self._rng)
                    logger.warning(
                        "%s: attempt %d failed %s. The flex tier has no capacity: the provider refused this "
                        "attempt, so it is not billed and does not use up one of the %d attempts of the request. "
                        "Next attempt in %.1f s. The request waits for capacity until %.0f s after its first "
                        "attempt; %.0f s have passed.",
                        label,
                        attempt,
                        failure,
                        self._max_attempts,
                        delay_s,
                        self._flex_busy_wait_s,
                        waited_s,
                    )
                else:
                    allowed = self._max_attempts + busy_attempts
                    retryable = is_retryable_status(status) or transport == "connection"
                    if attempt >= allowed or not retryable:
                        message = f"{label}: the call failed on attempt {attempt} of {allowed} {failure}"
                        if not unconfirmed:
                            raise BackendError(message) from exc
                        logger.warning("%s", message)
                        raise UnconfirmedCallError(message) from exc
                    delay_s = backoff_delay_s(attempt - 1, self._rng)
                    logger.warning(
                        "%s: attempt %d of %d failed %s. Next attempt in %.1f s.",
                        label,
                        attempt,
                        allowed,
                        failure,
                        delay_s,
                    )
                self._sleep(delay_s)
            else:
                return interaction, attempt, self._clock() - started

    def _refuse_when_stopped(self, label: str, attempts_made: int) -> None:
        """Raise ``BackendError`` when the stop event is set, so that no attempt is sent after a stop request.

        ``attempts_made`` is the number of attempts that the request has made so
        far. A warning is logged, because nobody may read the error: after an
        interruption the thread pool drops the errors of its requests.
        """
        if not self._stop.is_set():
            return
        if attempts_made == 0:
            message = f"{label}: a stop was requested before this request was sent, so it was not sent"
        else:
            message = (
                f"{label}: a stop was requested after attempt {attempts_made} failed, "
                "so no further attempt was sent"
            )
        logger.warning("%s", message)
        raise BackendError(message)
