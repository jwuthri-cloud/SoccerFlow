"""Transport types between pipeline steps and a model backend.

A step builds a :class:`VLMRequest` (text, images, a JSON schema). A backend
returns a :class:`VLMResponse` (JSON text plus token usage). Requests are
stateless: nothing from an earlier answer is implied, so every answer can be
cached and replayed. A backend that cannot answer raises ``BackendError``.
Its subclasses name the cases that need their own handling:

- ``BilledAnswerError``: the call was billed although its answer cannot be used.
- ``UnconfirmedCallError``: the call ended without an answer that can be read,
  so nobody knows whether it is billed.
- ``FlexBusyError``: the flex tier had no capacity for the request for the
  whole time that the backend waits for it.
- ``CacheMiss``: replay mode found no stored answer.
- ``OtherImagesError``: a stored answer was recorded with other image bytes.

All types are records (see ``schemas.Record``): frozen, unknown fields are an
error, and ``model_copy(update=...)`` validates the copy.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Protocol

from pydantic import Field, field_validator

from pitchvlm.schemas import Record

Resolution = Literal["low", "medium", "high", "ultra_high"]
#: The image resolutions that an OpenRouter request can ask for: the ``detail`` of an image part of
#: the chat completions API, which has no level between them.
OPENROUTER_RESOLUTIONS: tuple[Resolution, ...] = ("low", "high")
#: How much a model thinks before it answers. ``model_default`` sends no thinking setting, so the
#: model uses its own default. Only an OpenRouter step may use it: a model that has no thinking
#: setting needs it (``config.StepSettings``).
ThinkingLevel = Literal["minimal", "low", "medium", "high", "xhigh", "model_default"]
#: The model services that a request can go to. ``gemini`` is the Interactions API of Google;
#: ``openrouter`` is the chat completions API of OpenRouter, which reaches the models of many
#: providers. Only ``vlm/client.py`` talks to them.
Provider = Literal["gemini", "openrouter"]
#: The provider of a request that names none. A Gemini request has the cache key that it had
#: before providers were added (``VLMRequest.cache_key``).
DEFAULT_PROVIDER: Provider = "gemini"
#: The service tiers that a request can ask for.
ServiceTier = Literal["standard", "flex"]
#: The service tiers that an answer can be billed at. No request asks for ``priority``, but an
#: answer can report it, and it costs more than the standard tier.
BilledTier = Literal["standard", "flex", "priority"]
#: The entries of a request context that a stored answer keeps: the frame numbers of the images.
STORED_CONTEXT_KEYS = frozenset({"frames"})


class TextPart(Record):
    kind: Literal["text"] = "text"
    text: str


class ImagePart(Record):
    kind: Literal["image"] = "image"
    #: Encoded image bytes (JPEG by default). Never written to the cache.
    data: bytes = Field(repr=False)
    mime_type: str = "image/jpeg"
    resolution: Resolution = "high"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


class RequestMeta(Record):
    """Where a request belongs. Part of the cache key."""

    clip_id: str
    #: Pipeline step, e.g. "kit_legend", "people", "landmarks", "number".
    step: str
    frame: int | None = None
    #: Sub-item inside the step, e.g. "crops-0a1b2c3d4e5f6a7b" (the crops of a track) or "tile-2of6".
    item: str | None = None
    #: Repeat index. Repeat runs use 1, 2, ... so that they are not served from the cache.
    repeat: int = Field(default=0, ge=0)


class VLMRequest(Record):
    #: The model service that answers the request.
    provider: Provider = DEFAULT_PROVIDER
    #: The name of the model at that service: a Gemini model name, or an OpenRouter model slug.
    model: str
    thinking_level: ThinkingLevel
    parts: tuple[TextPart | ImagePart, ...]
    #: JSON schema that the answer must follow.
    response_schema: dict[str, Any]
    #: Human-readable prompt name, e.g. "people/v1".
    prompt_name: str
    meta: RequestMeta
    service_tier: ServiceTier = "standard"
    #: Facts about what the images show that ``meta`` does not hold, e.g. the
    #: frames and boxes of the crops in a shirt-number request. Plain JSON
    #: data. It is never sent to the model. It is part of the cache key: the
    #: key does not see the pixels, so the context is what tells two sets of
    #: images apart. Fake backends read it too. A stored answer keeps only a
    #: part of it (``stored_context``).
    context: dict[str, Any] = Field(default_factory=dict)

    @field_validator("context")
    @classmethod
    def _check_context(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Refuse a context that cannot be written as JSON, because the cache key is made from its JSON text."""
        try:
            json.dumps(value, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f"the context must be plain JSON data: {error}") from error
        return value

    def stored_context(self) -> dict[str, Any]:
        """Return the part of the context that a stored answer keeps: the entries of ``STORED_CONTEXT_KEYS``.

        Every other entry stays on the machine that made the request. The
        boxes of a probe request are boxes of the label files, and label data
        must not be part of a result file. An entry that is new is not stored
        until its name is added to ``STORED_CONTEXT_KEYS``. The cache key
        still stands for the whole context.
        """
        return {name: value for name, value in self.context.items() if name in STORED_CONTEXT_KEYS}

    @property
    def prompt_hash(self) -> str:
        """Hash of all text parts. A changed prompt gives a new cache key."""
        digest = hashlib.sha256()
        for part in self.parts:
            if isinstance(part, TextPart):
                digest.update(part.text.encode("utf-8"))
                digest.update(b"\x00")
        return digest.hexdigest()[:16]

    @property
    def image_hashes(self) -> tuple[str, ...]:
        return tuple(part.sha256 for part in self.parts if isinstance(part, ImagePart))

    @property
    def resolutions(self) -> tuple[Resolution, ...]:
        return tuple(part.resolution for part in self.parts if isinstance(part, ImagePart))

    def cache_key(self) -> str:
        """Key of the stored answer.

        The key does not depend on pixel bytes. The place of the request
        (clip, step, frame, item) and its context stand for the pixels. So a
        request that shows other images must have another place or another
        context: the shirt-number request of a track gets another key when
        the track is made of other crops. The key does not depend on the
        service tier either: the tier changes the price, not the question.

        Why no pixels: the same picture can get other bytes from another
        version of the image library. With pixels in the key, such a request
        would miss its stored answer without a word and would be paid again.
        The stored answer keeps the image hashes instead, and a hit with other
        image bytes is an error that costs nothing (``vlm.cache``). A replay
        still needs the footage: the requests are built from the frames.

        The key holds the provider and the model. The provider enters the key
        only when it is not Gemini, so a Gemini request has the same key as
        before providers were added, and its stored answers stay valid. The
        coordinate convention of a step is not in the key: it changes the
        prompt text and the answer schema, and both are in the key.
        """
        identity = {
            "clip_id": self.meta.clip_id,
            "step": self.meta.step,
            "frame": self.meta.frame,
            "item": self.meta.item,
            "repeat": self.meta.repeat,
            "model": self.model,
            "thinking_level": self.thinking_level,
            "resolutions": list(self.resolutions),
            "prompt_hash": self.prompt_hash,
            "schema": hashlib.sha256(
                json.dumps(self.response_schema, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16],
            "context": self.context,
        }
        if self.provider != DEFAULT_PROVIDER:
            identity["provider"] = self.provider
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:32]


class Usage(Record):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    #: Thinking tokens. They are billed at the output price.
    thought_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)


class VLMResponse(Record):
    #: The answer: JSON text that follows the request's schema.
    text: str
    usage: Usage
    model: str
    #: The billed tier: the tier whose price the answer costs. The real backend
    #: takes the tier that the answer reports, so this can differ from the tier
    #: of the request. A reported tier without a price counts as the most
    #: expensive tier. The cost ledger prices this field.
    service_tier: BilledTier = "standard"
    latency_s: float = Field(ge=0)
    attempts: int = Field(default=1, ge=1)
    sdk_version: str | None = None
    #: UTC time of the answer, ISO 8601.
    created_at: str
    #: The cost of the call in USD as the API reported it, or None when the API reports none.
    #: OpenRouter reports the cost of every call, and the cost ledger takes it as it is
    #: (``cost.billed_usd``). Gemini reports tokens only: its cost comes from the price table.
    reported_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    #: The provider behind OpenRouter that served the call, as the answer names it. None for Gemini.
    served_by: str | None = None


class Backend(Protocol):
    """Anything that can answer a request: the real API, a cache, or a fake."""

    def generate(self, request: VLMRequest) -> VLMResponse: ...


class BackendError(RuntimeError):
    """A backend could not produce an answer."""


class BilledAnswerError(BackendError):
    """The model answered and billed the call, but the answer cannot be used.

    ``billed`` is the cost record of the call in the shape of a response, so
    that it fits the cost ledger and an ``on_new_answer`` callback. Its
    ``usage``, ``model``, ``service_tier`` and ``created_at`` say which tokens
    were billed, for which model, at the price of which tier, and when. Its
    ``text`` is empty, because there is no answer to use.

    The cost of such a call is reported in one place only: the wrapper that
    reports the new answers of the real backend to the cost ledger
    (``vlm.cache.CachedBackend``, or the metered backend of the command line
    for calls without a cache). That wrapper reports ``billed`` like a new
    answer and raises this error again. Code above the wrapper must not record
    ``billed`` a second time. ``billed`` must never be stored as an answer.
    """

    def __init__(self, message: str, billed: VLMResponse) -> None:
        super().__init__(message)
        self.billed = billed


class UnconfirmedCallError(BackendError):
    """A call ended without an answer that can be read.

    There are two cases. The client timeout ran out or the connection was
    lost: no status code and no token usage came back, so nobody knows whether
    the provider got the request. When it did, it can still finish the request
    and bill it. Or the provider answered with a success status, but the body
    of the answer could not be read: the request was done, and its token usage
    is lost. In both cases the cost of the call is unknown, so the cost ledger
    has no line for it: compare the ledger with the bill of the provider after
    such an error. The message of the error says this.
    """


class FlexBusyError(BackendError):
    """The flex tier had no capacity for a request for the whole time that the backend waits for it.

    The provider answers a flex request with the status 503 or 429 while the
    flex tier is busy. It never moves such a request to the standard tier. The
    backend tries the request again until its waiting time for capacity is
    used up, and then raises this error. The provider refused every one of
    these attempts, so none of them is billed. The error says nothing about
    the client: the same request can pass at another time.
    """


class CacheMiss(BackendError):
    """Replay mode found no stored answer for a request."""


class OtherImagesError(BackendError):
    """A stored answer was recorded with other image bytes than its request shows now.

    The cache key has no pixels, so the stored answer can describe other
    pixels: for example the clip was replaced under the same name. The answer
    is not used, and the model is not asked. The message names the two ways
    out.
    """
