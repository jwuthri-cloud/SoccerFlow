"""Tests for the transport types of ``pitchvlm.vlm.types``.

The cache key decides which stored answer a request gets. It must change with
everything that changes the question (text, model, frame, item, repeat index)
and with the context, which says what the images show. It must not change
with the pixel bytes or the service tier.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from pitchvlm.vlm.types import (
    BackendError,
    BilledAnswerError,
    BilledTier,
    CacheMiss,
    FlexBusyError,
    ImagePart,
    OtherImagesError,
    RequestMeta,
    Resolution,
    ServiceTier,
    TextPart,
    ThinkingLevel,
    UnconfirmedCallError,
    Usage,
    VLMRequest,
    VLMResponse,
)

PROMPT_TEXT = "Find every person in the image."
PEOPLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"people": {"type": "array"}},
    "required": ["people"],
}


def make_request(
    *,
    text: str = PROMPT_TEXT,
    image: bytes = b"jpeg-bytes-one",
    resolution: Resolution = "high",
    parts: tuple[TextPart | ImagePart, ...] | None = None,
    clip_id: str = "clip-001",
    step: str = "people",
    frame: int | None = 12,
    item: str | None = None,
    repeat: int = 0,
    model: str = "gemini-4-argon",
    thinking_level: ThinkingLevel = "low",
    service_tier: ServiceTier = "standard",
    schema: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> VLMRequest:
    """Build a people request with one text part and one image part.

    Each test changes one argument. ``parts`` replaces the text and the image part when it is given.
    """
    if parts is None:
        parts = (TextPart(text=text), ImagePart(data=image, resolution=resolution))
    return VLMRequest(
        model=model,
        thinking_level=thinking_level,
        parts=parts,
        response_schema=PEOPLE_SCHEMA if schema is None else schema,
        prompt_name="people/v1",
        meta=RequestMeta(clip_id=clip_id, step=step, frame=frame, item=item, repeat=repeat),
        service_tier=service_tier,
        context={} if context is None else context,
    )


def make_response(**overrides: Any) -> VLMResponse:
    """Build a valid response. Keyword arguments replace single fields."""
    fields: dict[str, Any] = {
        "text": '{"people": []}',
        "usage": Usage(input_tokens=1500, output_tokens=400, thought_tokens=1000),
        "model": "gemini-4-argon",
        "latency_s": 6.5,
        "created_at": "2026-11-13T10:00:00+00:00",
    }
    fields.update(overrides)
    return VLMResponse(**fields)


# --- Parts -------------------------------------------------------------------------------


def test_image_part_sha256_is_the_hash_of_the_bytes() -> None:
    """The hash is the SHA-256 hex digest of the image bytes."""
    part = ImagePart(data=b"abc")
    assert part.sha256 == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert part.sha256 == hashlib.sha256(b"abc").hexdigest()


def test_image_part_sha256_follows_the_bytes_only() -> None:
    """Other bytes give another hash. Resolution and MIME type do not change it."""
    first = ImagePart(data=b"jpeg-bytes-one")
    same_bytes = ImagePart(data=b"jpeg-bytes-one", resolution="low", mime_type="image/png")
    other_bytes = ImagePart(data=b"jpeg-bytes-two")
    assert same_bytes.sha256 == first.sha256
    assert other_bytes.sha256 != first.sha256


def test_part_defaults() -> None:
    """An image part is a JPEG at high resolution unless told otherwise."""
    part = ImagePart(data=b"x")
    assert (part.kind, part.mime_type, part.resolution) == ("image", "image/jpeg", "high")
    assert TextPart(text="hello").kind == "text"


def test_image_bytes_are_not_printed() -> None:
    """The text form of a part or a request does not show the image bytes."""
    part = ImagePart(data=b"PIXELDATA")
    request = make_request(image=b"PIXELDATA")
    assert "PIXELDATA" not in repr(part)
    assert "PIXELDATA" not in str(part)
    assert "PIXELDATA" not in repr(request)


def test_image_hashes_and_resolutions_follow_the_part_order() -> None:
    """Hashes and resolutions are listed for the image parts only, in request order."""
    diagram = ImagePart(data=b"diagram-bytes", resolution="medium")
    frame = ImagePart(data=b"frame-bytes", resolution="high")
    request = make_request(parts=(diagram, TextPart(text="Find the landmarks."), frame))
    assert request.image_hashes == (diagram.sha256, frame.sha256)
    assert request.resolutions == ("medium", "high")


# --- Prompt hash -------------------------------------------------------------------------


def test_prompt_hash_is_a_stable_short_hex_string() -> None:
    """The prompt hash has 16 hex characters and does not change between runs."""
    prompt_hash = make_request().prompt_hash
    assert len(prompt_hash) == 16
    assert set(prompt_hash) <= set("0123456789abcdef")
    assert prompt_hash == "4de16e0f6d1f6566"
    assert make_request().prompt_hash == prompt_hash


def test_prompt_hash_changes_with_the_text() -> None:
    """Another prompt text gives another prompt hash."""
    assert make_request(text="Find every referee in the image.").prompt_hash != make_request().prompt_hash


def test_prompt_hash_does_not_depend_on_images_or_meta() -> None:
    """Image bytes, image resolution, frame, repeat index and service tier leave the hash as it is."""
    base = make_request().prompt_hash
    assert make_request(image=b"jpeg-bytes-two").prompt_hash == base
    assert make_request(resolution="low").prompt_hash == base
    assert make_request(frame=24, repeat=1, service_tier="flex").prompt_hash == base


def test_prompt_hash_covers_every_text_part() -> None:
    """All text parts go into the hash, each one closed by a zero byte."""
    one = make_request(parts=(TextPart(text="first"),))
    two = make_request(parts=(TextPart(text="first"), TextPart(text="second")))
    assert one.prompt_hash != two.prompt_hash
    assert two.prompt_hash == hashlib.sha256(b"first\x00second\x00").hexdigest()[:16]


def test_prompt_hash_keeps_the_borders_between_text_parts() -> None:
    """The same characters split at another place give another hash."""
    split_late = make_request(parts=(TextPart(text="ab"), TextPart(text="c")))
    split_early = make_request(parts=(TextPart(text="a"), TextPart(text="bc")))
    assert split_late.prompt_hash != split_early.prompt_hash


# --- Cache key ---------------------------------------------------------------------------


def test_cache_key_is_stable() -> None:
    """The key of a fixed request has 32 hex characters and a fixed value.

    Stored answers are found by this key on both machines. Another value here means that every
    stored answer is lost, so the key recipe must change on purpose only.
    """
    key = make_request().cache_key()
    assert len(key) == 32
    assert set(key) <= set("0123456789abcdef")
    assert key == "7392b9c8b061888c4efceffcf8b15b4c"
    assert make_request().cache_key() == key


def test_cache_key_recipe() -> None:
    """The key is the SHA-256 of the identity fields as JSON with sorted keys, cut to 32 hex characters."""
    context = {"frames": [0, 12], "boxes": [[10.0, 20.0, 30.0, 80.0], [15.0, 20.0, 35.0, 80.0]]}
    request = make_request(frame=None, item="track-12", context=context)
    identity = {
        "clip_id": "clip-001",
        "step": "people",
        "frame": None,
        "item": "track-12",
        "repeat": 0,
        "model": "gemini-4-argon",
        "thinking_level": "low",
        "resolutions": ["high"],
        "prompt_hash": hashlib.sha256(PROMPT_TEXT.encode("utf-8") + b"\x00").hexdigest()[:16],
        "schema": hashlib.sha256(json.dumps(PEOPLE_SCHEMA, sort_keys=True).encode("utf-8")).hexdigest()[:16],
        "context": context,
    }
    expected = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:32]
    assert request.cache_key() == expected


def test_cache_key_changes_when_the_text_changes() -> None:
    """A changed prompt text gives a new key."""
    assert make_request(text="Find every referee in the image.").cache_key() != make_request().cache_key()


def test_cache_key_does_not_depend_on_the_image_bytes() -> None:
    """Other pixel bytes for the same clip, step and frame give the same key."""
    first = make_request(image=b"jpeg-bytes-one")
    second = make_request(image=b"jpeg-bytes-two")
    assert first.image_hashes != second.image_hashes
    assert first.cache_key() == second.cache_key()


def test_cache_key_changes_with_the_repeat_index() -> None:
    """Each repeat index has its own key, so a repeat run is not served from the cache."""
    keys = {make_request(repeat=repeat).cache_key() for repeat in (0, 1, 2)}
    assert len(keys) == 3


def test_cache_key_does_not_depend_on_the_service_tier() -> None:
    """The tier changes the price, not the question."""
    assert make_request(service_tier="flex").cache_key() == make_request(service_tier="standard").cache_key()


def test_cache_key_changes_with_the_context() -> None:
    """The context stands for the pixels: other crop frames or other crop boxes are another question.

    A track id is not a lasting name. After a change in the tracking code, "track-12" can be another
    player. The frames and boxes of the crops then differ, so the stored answer is not used again.
    """
    boxes = [[10.0, 20.0, 30.0, 80.0], [15.0, 20.0, 35.0, 80.0]]
    moved_boxes = [[10.0, 20.0, 30.0, 80.0], [15.0, 20.0, 36.0, 80.0]]
    requests = [
        make_request(frame=None, item="track-12"),
        make_request(frame=None, item="track-12", context={"frames": [0, 12], "boxes": boxes}),
        make_request(frame=None, item="track-12", context={"frames": [0, 24], "boxes": boxes}),
        make_request(frame=None, item="track-12", context={"frames": [0, 12], "boxes": moved_boxes}),
    ]
    assert len({request.cache_key() for request in requests}) == 4


def test_stored_context_keeps_the_frames_and_no_other_entry() -> None:
    """A stored answer keeps the frame numbers of the context. Boxes, and every entry that is new, stay out."""
    boxes = [[10.0, 20.0, 30.0, 80.0], [15.0, 20.0, 35.0, 80.0]]
    request = make_request(context={"frames": [0, 12], "boxes": boxes, "note": "a new entry"})

    assert request.stored_context() == {"frames": [0, 12]}
    assert make_request(context={"boxes": boxes}).stored_context() == {}
    assert make_request().stored_context() == {}
    # The request itself keeps the whole context: the cache key and the fake backends read it.
    assert request.context == {"frames": [0, 12], "boxes": boxes, "note": "a new entry"}


def test_cache_key_is_the_same_for_an_equal_context() -> None:
    """Equal contexts give the same key. The order of the keys inside the context does not matter."""
    first = make_request(context={"frames": [0, 12], "boxes": [[10.0, 20.0, 30.0, 80.0], [15.0, 20.0, 35.0, 80.0]]})
    second = make_request(context={"boxes": [[10.0, 20.0, 30.0, 80.0], [15.0, 20.0, 35.0, 80.0]], "frames": [0, 12]})
    assert list(first.context) != list(second.context)
    assert first.cache_key() == second.cache_key()
    assert first.cache_key() != make_request().cache_key()


@pytest.mark.parametrize(
    "context",
    [
        pytest.param({"frames": {0, 12}}, id="a set"),
        pytest.param({"crop": b"jpeg-bytes"}, id="bytes"),
        pytest.param({"box": [10.0, float("nan"), 30.0, 80.0]}, id="nan"),
        pytest.param({"source": object()}, id="an object"),
    ],
)
def test_context_must_be_plain_json_data(context: dict[str, Any]) -> None:
    """The key is made from the JSON text of the context, so a context that JSON cannot hold is an error."""
    with pytest.raises(ValidationError, match="plain JSON"):
        make_request(context=context)


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"clip_id": "clip-002"}, id="clip"),
        pytest.param({"step": "landmarks"}, id="step"),
        pytest.param({"frame": 24}, id="frame"),
        pytest.param({"frame": None}, id="no frame"),
        pytest.param({"item": "tile-2of6"}, id="item"),
        pytest.param({"model": "gemini-3.8-flash"}, id="model"),
        pytest.param({"thinking_level": "high"}, id="thinking level"),
        pytest.param({"resolution": "low"}, id="image resolution"),
        pytest.param({"schema": {"type": "object"}}, id="answer schema"),
    ],
)
def test_cache_key_changes_with_every_identity_field(change: dict[str, Any]) -> None:
    """Clip, step, frame, item, model, thinking level, resolution and schema are part of the key."""
    assert make_request(**change).cache_key() != make_request().cache_key()


def test_cache_key_changes_with_the_number_of_images() -> None:
    """A request with one more image is another question."""
    one_image = make_request()
    two_images = make_request(parts=one_image.parts + (ImagePart(data=b"jpeg-bytes-two"),))
    assert two_images.cache_key() != one_image.cache_key()


def test_cache_key_ignores_the_key_order_of_the_schema() -> None:
    """The same schema written in another key order gives the same key."""
    reordered = {"required": ["people"], "properties": {"people": {"type": "array"}}, "type": "object"}
    assert list(reordered) != list(PEOPLE_SCHEMA)
    assert make_request(schema=reordered).cache_key() == make_request().cache_key()


# --- Request, response and errors ----------------------------------------------------------


def test_request_defaults() -> None:
    """A request has the standard tier, no context and a tuple of parts by default."""
    request = VLMRequest(
        model="gemini-4-argon",
        thinking_level="low",
        parts=[TextPart(text="Describe the kits.")],
        response_schema={"type": "object"},
        prompt_name="kit_legend/v1",
        meta=RequestMeta(clip_id="clip-001", step="kit_legend"),
    )
    assert request.service_tier == "standard"
    assert request.context == {}
    assert isinstance(request.parts, tuple)
    assert (request.meta.frame, request.meta.item, request.meta.repeat) == (None, None, 0)


def test_request_is_frozen() -> None:
    """A request and its meta block cannot be assigned."""
    request = make_request()
    with pytest.raises(ValidationError, match="frozen"):
        request.model = "other-model"
    with pytest.raises(ValidationError, match="frozen"):
        request.meta.repeat = 1
    assert request.cache_key() == make_request().cache_key()


def test_changed_copy_of_a_request_is_validated() -> None:
    """A changed copy goes through the checks of a new request, so a misspelled field name is an error."""
    request = make_request()
    item = "track-12-0a1b2c3d"
    renamed = request.model_copy(update={"meta": request.meta.model_copy(update={"item": item})})
    assert renamed.meta.item == item
    assert request.meta.item is None
    assert renamed.parts == request.parts
    assert renamed.cache_key() != request.cache_key()
    with pytest.raises(ValidationError):
        request.meta.model_copy(update={"itme": item})
    with pytest.raises(ValidationError):
        request.model_copy(update={"thinking_level": "extreme"})
    with pytest.raises(ValidationError, match="plain JSON"):
        request.model_copy(update={"context": {"frames": {0, 12}}})
    with pytest.raises(ValidationError):
        make_response().model_copy(update={"attempts": 0})


def test_request_rejects_unknown_fields_and_values() -> None:
    """Unknown fields and unknown labels are an error, so a request cannot carry a sampling parameter."""
    meta = RequestMeta(clip_id="clip-001", step="people")
    with pytest.raises(ValidationError):
        VLMRequest(
            model="gemini-4-argon",
            thinking_level="low",
            parts=(TextPart(text=PROMPT_TEXT),),
            response_schema={"type": "object"},
            prompt_name="people/v1",
            meta=meta,
            temperature=0.2,
        )
    with pytest.raises(ValidationError):
        RequestMeta(clip_id="clip-001", step="people", camera="main")
    with pytest.raises(ValidationError):
        RequestMeta(clip_id="clip-001", step="people", repeat=-1)
    with pytest.raises(ValidationError):
        make_request(thinking_level="extreme")
    with pytest.raises(ValidationError):
        make_request(resolution="huge")
    with pytest.raises(ValidationError):
        make_request(service_tier="batch")


def test_usage_defaults_and_limits() -> None:
    """Thinking and cached tokens default to 0, and no token count can be negative."""
    usage = Usage(input_tokens=10, output_tokens=5)
    assert (usage.thought_tokens, usage.cached_tokens) == (0, 0)
    for field in ("input_tokens", "output_tokens", "thought_tokens", "cached_tokens"):
        values = {"input_tokens": 10, "output_tokens": 5}
        values[field] = -1
        with pytest.raises(ValidationError):
            Usage(**values)


def test_response_defaults() -> None:
    """A response is on the standard tier after one attempt unless told otherwise."""
    response = make_response()
    assert response.service_tier == "standard"
    assert response.attempts == 1
    assert response.sdk_version is None
    assert response.usage.cached_tokens == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("latency_s", -0.1, id="negative latency"),
        pytest.param("attempts", 0, id="no attempt"),
        pytest.param("service_tier", "batch", id="unknown tier"),
        pytest.param("usage", None, id="no usage"),
    ],
)
def test_response_rejects_bad_values(field: str, value: Any) -> None:
    """A response with a value outside its range is an error."""
    with pytest.raises(ValidationError):
        make_response(**{field: value})


def test_response_survives_a_json_round_trip() -> None:
    """A response written as JSON text and read back is equal to the first one."""
    response = make_response(service_tier="flex", attempts=2, sdk_version="2.34.0")
    assert VLMResponse.model_validate_json(response.model_dump_json()) == response


def test_priority_is_a_billed_tier_that_no_request_can_ask_for() -> None:
    """An answer can report the priority tier, and its price is higher. A request asks for standard or flex only."""
    assert get_args(ServiceTier) == ("standard", "flex")
    assert get_args(BilledTier) == ("standard", "flex", "priority")
    response = make_response(service_tier="priority")
    assert response.service_tier == "priority"
    assert VLMResponse.model_validate_json(response.model_dump_json()) == response
    with pytest.raises(ValidationError):
        make_request(service_tier="priority")  # type: ignore[arg-type]


def test_cache_miss_is_a_backend_error() -> None:
    """A cache miss can be caught as a backend error, and both are runtime errors."""
    assert issubclass(CacheMiss, BackendError)
    assert issubclass(BackendError, RuntimeError)
    with pytest.raises(BackendError):
        raise CacheMiss("no stored answer")


def test_errors_with_their_own_handling_are_backend_errors_of_their_own_kind() -> None:
    """A command prints a backend error as one line. No kind is taken for another kind."""
    kinds = (CacheMiss, OtherImagesError, UnconfirmedCallError, BilledAnswerError, FlexBusyError)
    for kind in kinds:
        assert issubclass(kind, BackendError)
        assert not any(issubclass(kind, other) for other in kinds if other is not kind)
    # A stored answer with other image bytes is not a missing answer: a replay must not treat it as one.
    with pytest.raises(BackendError, match="other image bytes"):
        raise OtherImagesError("the stored answer was recorded with other image bytes")
    with pytest.raises(BackendError, match="may still be billed"):
        raise UnconfirmedCallError("no answer within the client timeout; this call may still be billed")
    # A busy flex tier refused the request, so the call is not billed: it is no unconfirmed call.
    with pytest.raises(BackendError, match="no capacity"):
        raise FlexBusyError("the flex tier had no capacity for 900 s")


def test_billed_answer_error_is_a_backend_error_with_the_cost_record() -> None:
    """The error of a billed answer that cannot be used carries what was billed, and it is a backend error."""
    billed = make_response(text="", service_tier="flex")
    error = BilledAnswerError("the answer has no text", billed)
    assert isinstance(error, BackendError)
    assert not issubclass(BilledAnswerError, CacheMiss)
    assert str(error) == "the answer has no text"
    assert error.billed is billed
    with pytest.raises(BackendError, match="no text"):
        raise error
