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
- ``VideoInputError``: a video part cannot be sent: its file changed, it is too
  large to go inline, or the model or its provider does not take video.

All types are records (see ``schemas.Record``): frozen, unknown fields are an
error, and ``model_copy(update=...)`` validates the copy.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field, field_validator, model_validator

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


#: The resolutions of a video part. The media-resolution page of Gemini lists tokens per video
#: frame for these levels only (``vlm.cost.GEMINI_VIDEO_RATES``); it lists none for ``ultra_high``.
VIDEO_RESOLUTIONS: tuple[Resolution, ...] = ("low", "medium", "high")
#: Largest sampling rate of a video part, frames per second: the range ``(0, 24]`` of the video
#: page of Gemini (published, not measured).
MAX_VIDEO_FPS = 24.0
#: The video file types that a video part can name: the types that both video pages list.
VIDEO_MIME_TYPES = frozenset({"video/mp4", "video/mpeg", "video/mov", "video/webm"})


class VideoFrames(Record):
    """What a video file that the package wrote shows: frames of a clip, written at a rate and a size.

    ``frames`` are the frames of the clip in the order of the file, ``write_fps`` the rate
    at which they were written and ``width`` x ``height`` their size. ``pixels_sha256`` is a
    hash of the decoded pixels of these frames of the clip (``vlm.video.frames_digest``),
    before the video codec compressed them. The clip is ``RequestMeta.clip_id``.
    """

    frames: tuple[int, ...] = Field(min_length=1)
    write_fps: float = Field(gt=0, allow_inf_nan=False)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    pixels_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class VideoPart(Record):
    """A reference to a video file, or to a time range of it, that a request sends to a model.

    The part holds no bytes: the backend reads the file when it sends the
    request, and it refuses the call when the bytes do not have the hash
    ``sha256`` (``vlm.client``). ``path`` is the place of the file on the run
    machine, outside the repository. It is never written to a cassette, a
    ledger line or a result file: ``stored`` is what a stored answer keeps (its
    identity and two hashes, no path and no bytes).

    ``source`` says what a file that the package wrote shows (``VideoFrames``);
    it is None for a file that came from elsewhere. A file written again from
    the same frames can get other bytes (another codec or another OpenCV build),
    so the identity of a written file is its frames and settings, not its bytes
    (``identity``).

    ``duration_s`` is the length of the whole file in seconds. ``start_s`` and
    ``end_s`` cut a range of it, ``fps`` is the sampling rate that the model is
    asked for (None: the default of the provider), and ``resolution`` is the
    media resolution of each sampled frame. The time 0 is the first frame of the
    file. The cost estimate counts the sampled frames of the range (``vlm.cost``).
    """

    kind: Literal["video"] = "video"
    path: Path = Field(repr=False)
    #: SHA-256 of the bytes of the file, as 64 small hexadecimal digits.
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mime_type: str = "video/mp4"
    duration_s: float = Field(gt=0, allow_inf_nan=False)
    start_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    end_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    fps: float | None = Field(default=None, gt=0, le=MAX_VIDEO_FPS, allow_inf_nan=False)
    resolution: Resolution = "high"
    source: VideoFrames | None = None

    @model_validator(mode="after")
    def _check_part(self) -> VideoPart:
        """Refuse an unknown file type, a resolution without a video rate, and a range that is empty or too long."""
        if self.mime_type not in VIDEO_MIME_TYPES:
            raise ValueError(f"a video part has no file type {self.mime_type!r}: use one of {sorted(VIDEO_MIME_TYPES)}")
        if self.resolution not in VIDEO_RESOLUTIONS:
            raise ValueError(f"a video part has no resolution {self.resolution!r}: use {', '.join(VIDEO_RESOLUTIONS)}")
        start, end = self.range_s
        if end > self.duration_s:
            raise ValueError(f"the range of a video part ends at {end:g} s, after the end of its file at {self.duration_s:g} s")
        if not start < end:
            raise ValueError(f"the range of a video part is empty: {start:g} s to {end:g} s")
        if self.source is not None and not math.isclose(
            self.duration_s, len(self.source.frames) / self.source.write_fps, rel_tol=1e-9
        ):
            raise ValueError(
                f"a video of {len(self.source.frames)} frames written at {self.source.write_fps:g} frames per second "
                f"lasts {len(self.source.frames) / self.source.write_fps:g} s, not {self.duration_s:g} s"
            )
        return self

    @property
    def range_s(self) -> tuple[float, float]:
        """Return the start and the end of the range that the model sees, in seconds of the file."""
        start = 0.0 if self.start_s is None else self.start_s
        end = self.duration_s if self.end_s is None else self.end_s
        return start, end

    @property
    def clipped(self) -> bool:
        """Say whether the part asks the provider for a range or a sampling rate of its own."""
        return self.start_s is not None or self.end_s is not None or self.fps is not None

    def sampled_frames(self, default_fps: float) -> int:
        """Return how many frames the model samples from the range: one at its start, then one every ``1 / fps`` s.

        ``default_fps`` is the rate of the provider, used when the part names
        none. The count is rounded up, so the estimate never counts too few.
        """
        rate = self.fps if self.fps is not None else default_fps
        if not 0 < rate < math.inf:
            raise ValueError(f"a sampling rate must be a number above 0, got {rate!r}")
        start, end = self.range_s
        return max(1, math.ceil((end - start) * rate - 1e-9))

    def identity(self) -> dict[str, Any]:
        """Return what tells this video part apart, as plain JSON data: no path and no bytes. It is part of the cache key.

        For a file that the package wrote (``source``): the frames of the clip, the rate and
        the size at which they were written, the range, the sampling rate and the resolution.
        Neither the hash of the bytes nor the hash of the pixels is in it, so the same frames
        written again with other bytes find their stored answer, and another decoder of the
        source images does not change the key either; a stored answer keeps both hashes
        (``stored``), and a hit compares them (``vlm.cache``). For any other file: the hash of
        its bytes with the range, the rate and the resolution.
        """
        identity: dict[str, Any] = {
            "mime_type": self.mime_type,
            "duration_s": self.duration_s,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "fps": self.fps,
            "resolution": self.resolution,
        }
        if self.source is None:
            identity["sha256"] = self.sha256
        else:
            identity["frames"] = list(self.source.frames)
            identity["write_fps"] = self.source.write_fps
            identity["size"] = [self.source.width, self.source.height]
        return identity

    def stored(self) -> dict[str, Any]:
        """Return what a stored answer keeps of this video part: ``identity``, the hash of the bytes that were sent, and
        the hash of the pixels of its frames (a written file). No path and no bytes."""
        stored = {**self.identity(), "sha256": self.sha256}
        if self.source is not None:
            stored["pixels_sha256"] = self.source.pixels_sha256
        return stored


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
    parts: tuple[TextPart | ImagePart | VideoPart, ...]
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

    @property
    def videos(self) -> tuple[VideoPart, ...]:
        """Return the video parts of the request, in their order."""
        return tuple(part for part in self.parts if isinstance(part, VideoPart))

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

        A request with video parts has their identities in the key
        (``VideoPart.identity``). For a file that the package wrote, that is the
        frames of the clip, the rate and the size at which they were written, the
        range, the sampling rate and the resolution, and not the bytes: a file that
        is written again from the same frames can get other bytes (another OpenCV
        build or codec: measured with the codecs ``avc1`` and ``mp4v`` on the build
        machine), and with the bytes in the key its request would miss its stored
        answer without an error and be paid again. The stored answer keeps the
        hash of the bytes that were sent; a hit on other bytes is counted and said
        (``vlm.cache``), and a hit on other pixels of the frames is refused as for
        images (``OtherImagesError``). The decision: docs/DECISIONS.md, entry
        "Second read of the live feed, and a cache key of video requests that
        survives a new write" (2026-11-23). A request without video parts has no
        such entry, so its key is the one of before video parts were added.
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
        if self.videos:
            identity["videos"] = [video.identity() for video in self.videos]
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


class VideoInputError(BackendError):
    """A video part cannot be sent, so no call was made and nothing is billed.

    The cases: the model or its provider does not take video (no video rate in
    its price table), the file is missing or has other bytes than its hash, the
    request is too large to go inline, or the provider cannot take the range or
    the sampling rate of the part.
    """
