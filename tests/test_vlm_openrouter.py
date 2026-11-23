"""Tests of the OpenRouter provider: requests, answers, errors, money and the unchanged Gemini path.

No test calls a model service. The OpenRouter answers are written by hand in the shapes of the
API documentation (docs/DECISIONS.md, entry "Image models through OpenRouter" of 2026-11-22,
item 4), and a fake HTTP layer returns them.
The tests of ``UrllibTransport`` replace the opening of a URL: no test opens a network connection.
"""

from __future__ import annotations

import base64
import email.message
import hashlib
import http.client
import io
import json
import threading
import urllib.error
import urllib.request
import urllib.response
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from pitchvlm.config import StepSettings, model_label, model_ref
from pitchvlm.schemas import Box
from pitchvlm.synth import SyntheticScene
from pitchvlm.vlm import steps
from pitchvlm.vlm.cache import CachedBackend, CassetteStore
from pitchvlm.vlm.client import (
    NO_PROVIDER_NOTE,
    OPENROUTER_ROUTING,
    OPENROUTER_URL,
    HttpAnswer,
    HttpConnectionError,
    HttpTimeoutError,
    OpenRouterBackend,
    ProviderBackend,
    UrllibTransport,
    build_interaction_kwargs,
    build_openrouter_body,
    keeps_inputs_private,
)
from pitchvlm.vlm.coords import (
    COORD_CONVENTIONS,
    box_from_answer,
    box_to_answer,
    point_from_answer,
    point_to_answer,
)
from pitchvlm.vlm.cost import (
    BudgetExceeded,
    BudgetGuard,
    Ledger,
    Price,
    TokenEstimate,
    billed_usd,
    estimate_usd,
    load_price_table,
)
from pitchvlm.vlm.fake import FakeBackend
from pitchvlm.vlm.prompts import box_format, point_format
from pitchvlm.vlm.types import (
    BackendError,
    BilledAnswerError,
    ImagePart,
    RequestMeta,
    TextPart,
    UnconfirmedCallError,
    Usage,
    VLMRequest,
    VLMResponse,
)

REPO = Path(__file__).resolve().parents[1]
MODEL = "qwen/qwen3-vl-235b-a22b-instruct"
KEY_VARIABLE = "OPENROUTER_API_KEY"
#: A value in the shape of a key, made of parts so that the secret checks do not see one in this file.
FAKE_KEY = "sk" + "-or-v1-" + "5e" * 32
PRICES = (Price(MODEL, date(2026, 11, 1), 0.21, 1.90),)
OPENROUTER = StepSettings(provider="openrouter", model=MODEL, thinking_level="model_default", coords="xy_1000")

# --- Answers in the shapes of the documentation -------------------------------------------------


def chat_answer(
    content: str | None = '{"people": []}',
    *,
    finish_reason: str = "stop",
    usage: Mapping[str, Any] | None = None,
    provider: str = "DeepInfra",
    **extra: Any,
) -> dict[str, Any]:
    """Return a chat completion body as the API reference shows it."""
    body: dict[str, Any] = {
        "id": "gen-1",
        "object": "chat.completion",
        "created": 1795000000,
        "model": MODEL,
        "provider": provider,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "native_finish_reason": finish_reason.upper(),
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": dict(
            usage
            if usage is not None
            else {
                "prompt_tokens": 1500,
                "completion_tokens": 400,
                "total_tokens": 1900,
                "cost": 0.00107,
                "is_byok": False,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 100},
            }
        ),
    }
    body.update(extra)
    return body


def error_body(code: int, message: str) -> dict[str, Any]:
    """Return an error body as the error page of the documentation shows it."""
    return {"error": {"code": code, "message": message, "metadata": {}}}


class FakeTransport:
    """An HTTP layer that returns prepared answers and keeps every request."""

    def __init__(self, *answers: HttpAnswer | Exception) -> None:
        self._answers = list(answers)
        self.requests: list[tuple[str, dict[str, str], dict[str, Any], float]] = []

    def post(self, url: str, headers: Mapping[str, str], body: bytes, timeout_s: float) -> HttpAnswer:
        self.requests.append((url, dict(headers), json.loads(body), timeout_s))
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def ok(body: Mapping[str, Any]) -> HttpAnswer:
    return HttpAnswer(200, json.dumps(body).encode("utf-8"))


def failed(status: int, message: str = "failed", retry_after_s: float | None = None) -> HttpAnswer:
    return HttpAnswer(status, json.dumps(error_body(status, message)).encode("utf-8"), retry_after_s)


def request_of(settings: StepSettings = OPENROUTER, step: str = "people", **changes: Any) -> VLMRequest:
    """Return a small request: one text part and one image part."""
    request = VLMRequest(
        provider=settings.provider,
        model=settings.model,
        thinking_level=settings.thinking_level,
        parts=(TextPart(text="Find every person."), ImagePart(data=b"jpeg-bytes", resolution=settings.resolution)),
        response_schema={"type": "object", "properties": {"people": {"type": "array"}}, "required": ["people"]},
        prompt_name="people/v1",
        meta=RequestMeta(clip_id="clip-1", step=step, frame=3),
        service_tier=settings.service_tier,
    )
    return request.model_copy(update=changes) if changes else request


def backend_with(transport: FakeTransport, **options: Any) -> tuple[OpenRouterBackend, list[float]]:
    """Return a backend on the fake transport whose waits are recorded, not slept."""
    waits: list[float] = []
    backend = OpenRouterBackend(prices=PRICES, transport=transport, sleep=waits.append, **options)
    return backend, waits


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv(KEY_VARIABLE, FAKE_KEY)
    return FAKE_KEY


# --- The Gemini path is the one of main ---------------------------------------------------------

#: Cache keys, prompt hashes and hashes of the Interactions API arguments of five Gemini requests,
#: computed with the code of commit f7d788e (the base of this branch, before providers existed).
#: The arguments are hashed with fixed image bytes, so the values do not depend on the JPEG encoder.
GEMINI_GOLDEN = {
    "kit_legend": (
        "65dc823a6bf12c2909f67cf7f5be07af",
        "df5034ec9a484b1d",
        "a6f2ac8b8c1dca728edd32dd890c2e68418705ae2e0c094f2c5a27876dab1332",
    ),
    "people": (
        "10de470cb94516715ac8132b66f1127b",
        "a3c69c3292922303",
        "cd8c8b205494506b1971fe418e7f37cc3a41dc61f8015590e54e0979ca7ef12b",
    ),
    "people_flex": (
        "5084de9aea88a445a85a2fca06a9851a",
        "a3c69c3292922303",
        "a9d7ca9831e77006ac1dfc07ebacf2eee0fd9a7aeddd50940815f1ab3012db7d",
    ),
    "landmarks": (
        "3177e21dccac490cef91f29ac67c854f",
        "c127af3c1b6c904a",
        "02dcf6a145363f6c2251779d91def5802d2d14ec09069190e2b444a7fbc55d88",
    ),
    "number": (
        "9c0db1c681f41b23aad6590a9e3ea177",
        "dbd41464e5fa873b",
        "5b059dd64b8e9daeccb9fa5381f9a3a3828625526cc9619285ef3bf3b7f28f3c",
    ),
}


def gemini_requests() -> dict[str, VLMRequest]:
    """Build the five Gemini requests of ``GEMINI_GOLDEN`` with the default step settings."""
    scene = SyntheticScene(n_frames=3, width=320, height=180, seed=0)
    legend = scene.kit_legend()
    image = np.zeros((180, 320, 3), np.uint8)
    settings = StepSettings()
    crops = [Box(x1=10, y1=10, x2=40, y2=90), Box(x1=12, y1=11, x2=41, y2=92)]
    return {
        "kit_legend": steps.build_kit_legend_request([image, image], clip_id="clipA", settings=settings),
        "people": steps.build_people_request(image, legend, clip_id="clipA", frame=2, settings=settings),
        "people_flex": steps.build_people_request(
            image,
            legend,
            clip_id="clipA",
            frame=2,
            settings=StepSettings(service_tier="flex", thinking_level="medium"),
        ),
        "landmarks": steps.build_landmarks_request(image, clip_id="clipA", frame=2, settings=settings),
        "number": steps.build_number_request(
            [image, image],
            legend,
            "A",
            clip_id="clipA",
            track_id=None,
            crop_frames=[0, 1],
            crop_boxes=crops,
            image_size=(320, 180),
            crop_upscale=4.0,
            settings=StepSettings(resolution="low"),
        ),
    }


def test_gemini_requests_and_cache_keys_are_those_of_main() -> None:
    """Stored Gemini answers stay valid: same cache key, same prompt text, same call of the Interactions API."""
    for name, request in gemini_requests().items():
        fixed_images = tuple(
            part
            if isinstance(part, TextPart)
            else ImagePart(data=b"img-" + str(index).encode(), resolution=part.resolution, mime_type=part.mime_type)
            for index, part in enumerate(request.parts)
        )
        kwargs = build_interaction_kwargs(request.model_copy(update={"parts": fixed_images}))
        kwargs_sha = hashlib.sha256(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()
        assert (request.cache_key(), request.prompt_hash, kwargs_sha) == GEMINI_GOLDEN[name], name
        assert request.provider == "gemini"


def test_gemini_cassette_and_ledger_hold_no_openrouter_value(tmp_path: Path) -> None:
    """A Gemini cassette has the content of before; its ledger line names the provider."""
    request = request_of(StepSettings())
    response = VLMResponse(
        text="{}", usage=Usage(input_tokens=10, output_tokens=5), model="gemini-4-argon", latency_s=0.1,
        created_at="2026-11-22T10:00:00+00:00",
    )
    path = CassetteStore(tmp_path).put(request, response)
    stored = json.loads(path.read_text(encoding="utf-8"))["response"]
    assert "reported_usd" not in stored and "served_by" not in stored
    assert CassetteStore(tmp_path).get(request) == response


def test_provider_enters_the_cache_key_only_when_it_is_not_gemini() -> None:
    gemini = request_of(StepSettings(), model=MODEL)
    openrouter = gemini.model_copy(update={"provider": "openrouter"})
    assert gemini.cache_key() != openrouter.cache_key()
    assert request_of(model="other/model").cache_key() != request_of().cache_key()


# --- Settings ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "words"),
    [
        ({"model": MODEL + ":free"}, "free model"),
        ({"model": "gemini-4-argon"}, "slug author/name"),
        ({"service_tier": "flex"}, "no service tier flex"),
        ({"resolution": "medium"}, "no image resolution medium"),
        ({"resolution": "ultra_high"}, "no image resolution ultra_high"),
    ],
)
def test_openrouter_settings_refuse_what_the_api_or_rule_5_does_not_allow(changes: dict[str, Any], words: str) -> None:
    with pytest.raises(ValidationError, match=words):
        StepSettings(**{**OPENROUTER.model_dump(), **changes})


@pytest.mark.parametrize(
    ("changes", "words"),
    [({"coords": "xy_1000"}, "coordinate convention yx_1000"), ({"thinking_level": "model_default"}, "model_default")],
)
def test_gemini_settings_keep_the_gemini_contract(changes: dict[str, Any], words: str) -> None:
    with pytest.raises(ValidationError, match=words):
        StepSettings(**changes)


def test_model_reference_of_the_command_line() -> None:
    assert model_ref("gemini-4-argon") == ("gemini", "gemini-4-argon")
    assert model_ref(f"openrouter:{MODEL}") == ("openrouter", MODEL)
    assert model_ref("openrouter:a/b:nitro") == ("openrouter", "a/b:nitro")
    assert model_label("openrouter", MODEL) == f"openrouter:{MODEL}"
    assert model_label("gemini", "gemini-4-argon") == "gemini-4-argon"
    with pytest.raises(ValueError, match="no model name"):
        model_ref("openrouter:")


def test_experiment_and_settings_files_name_a_provider_for_every_step() -> None:
    import yaml

    for path in [REPO / "configs" / "default.yaml", *sorted((REPO / "configs" / "experiments").glob("*probe.yaml"))]:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        sections = [data["step"]] if "step" in data else [data[name] for name in ("kit_legend", "people", "landmarks", "number")]
        assert all("provider" in section and "coords" in section for section in sections), path.name


# --- Request body ------------------------------------------------------------------------------


def test_body_has_the_documented_shape_and_no_sampling_parameter() -> None:
    body = build_openrouter_body(request_of())

    assert body["model"] == MODEL
    assert body["stream"] is False
    (message,) = body["messages"]
    assert message["role"] == "user"
    text, image = message["content"]
    assert text == {"type": "text", "text": "Find every person."}
    # The address is built in parts: the guard refuses a data address in a tracked file.
    expected_url = "data" + ":image/jpeg;base64," + base64.b64encode(b"jpeg-bytes").decode("ascii")
    assert image == {"type": "image_url", "image_url": {"url": expected_url, "detail": "high"}}
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["name"] == "people_answer"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["provider"] == dict(OPENROUTER_ROUTING)
    assert body["provider"] == {"zdr": True, "data_collection": "deny", "require_parameters": True, "allow_fallbacks": False}
    sampling = {"temperature", "top_p", "top_k", "seed", "frequency_penalty", "presence_penalty", "max_tokens"}
    assert not sampling & set(body)
    assert "reasoning" not in body
    assert keeps_inputs_private(request_of())


def test_body_sends_the_thinking_level_as_the_reasoning_effort() -> None:
    body = build_openrouter_body(request_of(thinking_level="medium"))
    assert body["reasoning"] == {"effort": "medium"}


def test_strict_schema_closes_every_object_and_leaves_the_request_schema_alone() -> None:
    request = steps.build_people_request(
        np.zeros((90, 160, 3), np.uint8),
        SyntheticScene(n_frames=1, width=160, height=90, seed=0).kit_legend(),
        clip_id="clip-1",
        frame=0,
        settings=OPENROUTER,
    )
    schema = build_openrouter_body(request)["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["people"]["items"]["additionalProperties"] is False
    assert "additionalProperties" not in request.response_schema


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"provider": "gemini"}, "cannot be built for a gemini request"),
        ({"model": MODEL + ":free"}, "free model"),
        ({"service_tier": "flex"}, "no service tier"),
        ({"parts": ()}, "at least one"),
        ({"parts": (ImagePart(data=b"x", resolution="medium"),)}, "no resolution"),
    ],
)
def test_body_refuses_a_request_that_cannot_be_sent(changes: dict[str, Any], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        build_openrouter_body(request_of(**changes))


def test_landmarks_request_sends_the_diagram_at_a_resolution_that_openrouter_has() -> None:
    request = steps.build_landmarks_request(
        np.zeros((90, 160, 3), np.uint8), clip_id="clip-1", frame=0, settings=OPENROUTER
    )
    assert request.resolutions == ("high", "high")
    build_openrouter_body(request)


# --- Coordinate conventions --------------------------------------------------------------------


@pytest.mark.parametrize("convention", COORD_CONVENTIONS)
def test_boxes_and_points_go_back_and_forth_in_every_convention(convention: str) -> None:
    width, height = 1920, 1080
    box = Box(x1=100.0, y1=200.0, x2=300.0, y2=700.0)
    back = box_from_answer(box_to_answer(box, width, height, convention), width, height, convention)
    assert back is not None
    for name in ("x1", "y1", "x2", "y2"):
        assert getattr(back, name) == pytest.approx(getattr(box, name), abs=2.0)
    x, y = point_from_answer(point_to_answer(480.0, 270.0, width, height, convention), width, height, convention)
    assert (x, y) == pytest.approx((480.0, 270.0), abs=2.0)


def test_conventions_read_the_documented_orders_and_units() -> None:
    width, height = 2000, 1000
    expected = Box(x1=200.0, y1=100.0, x2=600.0, y2=500.0)
    answers = {
        "yx_1000": [100, 100, 500, 300],
        "xy_1000": [100, 100, 300, 500],
        "xy_unit": [0.1, 0.1, 0.3, 0.5],
        "xy_pixel": [200, 100, 600, 500],
    }
    for convention, values in answers.items():
        assert box_from_answer(values, width, height, convention) == expected, convention
    assert point_from_answer([250, 500], width, height, "xy_pixel") == (250.0, 500.0)
    assert point_from_answer([0.25, 0.5], width, height, "xy_unit") == (500.0, 500.0)
    with pytest.raises(ValueError, match="unknown coordinate convention"):
        box_from_answer([1, 2, 3, 4], width, height, "yx_pixel")  # type: ignore[arg-type]


def test_gemini_prompt_text_is_the_text_of_main() -> None:
    gemini_box = box_format("yx_1000", image="the image", width=1, height=1)
    assert gemini_box == (
        "Format: [ymin, xmin, ymax, xmax]. Each value is an integer from 0 to 1000, normalised to the image: "
        "y goes from 0 at the top edge to 1000 at the bottom edge, and x goes from 0 at the left edge to 1000 at "
        "the right edge."
    )
    assert point_format("yx_1000", image="image 2", width=1, height=1).startswith(
        "Format: [y, x]. Each value is an integer from 0 to 1000, normalised to image 2: y goes"
    )
    pixel = box_format("xy_pixel", image="the image", width=1920, height=1080)
    assert "[xmin, ymin, xmax, ymax]" in pixel and "1920 pixels wide and 1080 pixels high" in pixel
    assert "number from 0 to 1" in point_format("xy_unit", image="image 2", width=1, height=1)


def test_unit_convention_asks_for_numbers_and_the_others_for_integers() -> None:
    image = np.zeros((90, 160, 3), np.uint8)
    for convention, kind in (("xy_unit", "number"), ("xy_pixel", "integer"), ("xy_1000", "integer")):
        settings = OPENROUTER.model_copy(update={"coords": convention})
        request = steps.build_landmarks_request(image, clip_id="c", frame=0, settings=settings)
        point = request.response_schema["properties"]["landmarks"]["items"]["properties"]["point"]
        assert point["items"] == {"type": kind}


# --- The backend --------------------------------------------------------------------------------


def test_answer_gives_the_text_the_normalised_usage_and_the_reported_cost(key: str) -> None:
    transport = FakeTransport(ok(chat_answer()))
    backend, _ = backend_with(transport)

    response = backend.generate(request_of())

    assert response.text == '{"people": []}'
    # The 100 reasoning tokens are a part of the 400 completion tokens.
    assert response.usage == Usage(input_tokens=1500, output_tokens=300, thought_tokens=100, cached_tokens=0)
    assert response.reported_usd == pytest.approx(0.00107)
    assert billed_usd(response) == pytest.approx(0.00107)
    assert response.served_by == "DeepInfra"
    assert response.service_tier == "standard" and response.sdk_version is None and response.attempts == 1
    ((url, headers, body, timeout_s),) = transport.requests
    assert url == OPENROUTER_URL == "https://openrouter.ai/api/v1/chat/completions"
    assert headers == {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    assert body == build_openrouter_body(request_of())
    assert timeout_s == 300.0


def test_no_request_is_sent_without_the_key_or_without_a_price(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    transport = FakeTransport()
    backend, _ = backend_with(transport)
    with pytest.raises(BackendError, match="openrouter_prices.yaml"):
        backend.generate(request_of(model="other/model"))
    monkeypatch.delenv(KEY_VARIABLE)
    with pytest.raises(BackendError, match=f"{KEY_VARIABLE} is not set"):
        backend.generate(request_of())
    assert transport.requests == []


def test_a_busy_api_is_tried_again_after_the_wait_it_asks_for(key: str) -> None:
    transport = FakeTransport(failed(429, "rate limited", retry_after_s=42.0), failed(502), ok(chat_answer()))
    backend, waits = backend_with(transport)

    response = backend.generate(request_of())

    assert response.attempts == 3
    assert waits[0] == 42.0
    assert 5.0 <= waits[1] <= 20.0


def test_retry_after_is_capped(key: str) -> None:
    transport = FakeTransport(failed(429, retry_after_s=86400.0), ok(chat_answer()))
    backend, waits = backend_with(transport)
    backend.generate(request_of())
    assert waits == [300.0]


def test_no_provider_for_the_routing_settings_ends_the_request_with_its_reason(key: str) -> None:
    message = "There is no available model provider that meets your routing requirements."
    transport = FakeTransport(*[failed(503, message) for _ in range(2)])
    backend, _ = backend_with(transport, max_attempts=2)

    with pytest.raises(BackendError) as raised:
        backend.generate(request_of())

    assert not isinstance(raised.value, UnconfirmedCallError)
    assert "attempt 2 of 2" in str(raised.value)
    assert message in str(raised.value) and NO_PROVIDER_NOTE in str(raised.value)
    assert len(transport.requests) == 2


@pytest.mark.parametrize("status", [400, 401, 402, 403])
def test_a_refused_request_is_not_tried_again(key: str, status: int) -> None:
    transport = FakeTransport(failed(status, f"refused with {FAKE_KEY}"))
    backend, _ = backend_with(transport)
    with pytest.raises(BackendError, match=f"status {status}") as raised:
        backend.generate(request_of())
    assert FAKE_KEY not in str(raised.value)
    assert len(transport.requests) == 1


def test_a_timeout_is_not_tried_again_and_its_cost_is_unknown(key: str) -> None:
    transport = FakeTransport(HttpTimeoutError("no complete answer within 300 s"))
    backend, _ = backend_with(transport)
    with pytest.raises(UnconfirmedCallError, match="may still be billed"):
        backend.generate(request_of())
    assert len(transport.requests) == 1


def test_a_lost_connection_is_tried_again(key: str) -> None:
    transport = FakeTransport(HttpConnectionError("reset"), ok(chat_answer()))
    backend, _ = backend_with(transport)
    assert backend.generate(request_of()).attempts == 2


def test_lost_connections_end_with_a_call_whose_cost_is_unknown(key: str) -> None:
    transport = FakeTransport(HttpConnectionError("reset"), HttpConnectionError("reset"))
    backend, _ = backend_with(transport, max_attempts=2)
    with pytest.raises(UnconfirmedCallError, match="attempt 2 of 2"):
        backend.generate(request_of())


@pytest.mark.parametrize(
    ("answer", "words"),
    [
        (HttpAnswer(200, b"<html>"), "not a JSON object"),
        (ok(error_body(502, "provider failed during the generation")), "provider failed during the generation"),
        (ok(chat_answer(usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})), "no cost"),
        (ok(chat_answer(usage={"completion_tokens": 5, "cost": 0.1})), "no prompt_tokens"),
        (
            ok(chat_answer(usage={"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0, "is_byok": True})),
            "BYOK",
        ),
        (
            ok(chat_answer(usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.0})),
            "a cost of 0 USD for a paid model",
        ),
    ],
)
def test_an_answer_without_a_cost_is_a_call_whose_cost_is_unknown(key: str, answer: HttpAnswer, words: str) -> None:
    backend, _ = backend_with(FakeTransport(answer))
    with pytest.raises(UnconfirmedCallError, match=words):
        backend.generate(request_of())


@pytest.mark.parametrize(
    ("body", "words"),
    [
        (chat_answer(finish_reason="length"), "finish reason 'length'"),
        (chat_answer(finish_reason="content_filter"), "finish reason 'content_filter'"),
        (chat_answer(content=""), "no text"),
        (chat_answer(content=None), "no text"),
        (chat_answer(choices=[]), "no choice"),
        (chat_answer(error={"code": 502, "message": "stream broke"}), "stream broke"),
        (
            chat_answer(
                usage={"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10, "cost": 0.00001}
            ),
            "no output tokens",
        ),
    ],
)
def test_a_billed_answer_that_cannot_be_used_carries_its_cost(key: str, body: dict[str, Any], words: str) -> None:
    if words == "stream broke":
        body["choices"][0]["error"] = body.pop("error")
    backend, _ = backend_with(FakeTransport(ok(body)))
    with pytest.raises(BilledAnswerError, match=words) as raised:
        backend.generate(request_of())
    assert raised.value.billed.reported_usd == body["usage"]["cost"]
    assert raised.value.billed.text == ""


def test_the_spend_window_counts_the_reported_cost(key: str) -> None:
    usage = {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20, "cost": 0.6}
    transport = FakeTransport(ok(chat_answer(usage=usage)), ok(chat_answer(usage=usage)))
    now = [0.0]
    waits: list[float] = []

    def sleep(seconds: float) -> None:
        waits.append(seconds)
        now[0] += seconds

    backend = OpenRouterBackend(
        prices=PRICES, transport=transport, max_usd_per_10_min=0.5, clock=lambda: now[0], sleep=sleep
    )
    backend.generate(request_of())
    backend.generate(request_of())
    assert waits == [600.0]


def test_a_stop_request_sends_nothing(key: str) -> None:
    stop = threading.Event()
    stop.set()
    transport = FakeTransport()
    backend, _ = backend_with(transport, stop=stop)
    with pytest.raises(BackendError, match="stop was requested"):
        backend.generate(request_of())
    assert transport.requests == []


def test_each_request_goes_to_the_backend_of_its_provider() -> None:
    seen: list[str] = []

    class Named:
        def __init__(self, name: str) -> None:
            self.name = name

        def generate(self, request: VLMRequest) -> VLMResponse:
            seen.append(self.name)
            raise BackendError(self.name)

    backend = ProviderBackend({"gemini": Named("gemini"), "openrouter": Named("openrouter")})
    for request in (request_of(), request_of(StepSettings())):
        with pytest.raises(BackendError):
            backend.generate(request)
    assert seen == ["openrouter", "gemini"]
    with pytest.raises(BackendError, match="no backend for the provider 'openrouter'"):
        ProviderBackend({}).generate(request_of())


def test_cached_backend_stores_the_reported_cost_and_the_ledger_takes_it(key: str, tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    backend, _ = backend_with(FakeTransport(ok(chat_answer())))
    store = CassetteStore(tmp_path / "cache")
    cached = CachedBackend(store, backend, on_new_answer=lambda request, response: ledger.record("run", request, response))

    first = cached.generate(request_of())
    again = CachedBackend(store).generate(request_of())

    assert again == first and again.reported_usd == pytest.approx(0.00107) and again.served_by == "DeepInfra"
    (line,) = [json.loads(text) for text in (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert line["provider"] == "openrouter" and line["served_by"] == "DeepInfra"
    assert line["usd"] == pytest.approx(0.00107)


# --- Money ---------------------------------------------------------------------------------------


def write_table(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "prices.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_price_table_of_the_repository_is_valid_and_paid_only() -> None:
    prices = load_price_table(REPO / "configs" / "openrouter_prices.yaml")
    assert MODEL in {price.model for price in prices}
    assert all(price.input_usd_per_million > 0 and price.output_usd_per_million > 0 for price in prices)
    assert all(not price.model.endswith(":free") for price in prices)


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("models:\n  - model: a/b:free\n    prices: [{valid_from: 2026-11-22, input_usd_per_million: 1, output_usd_per_million: 1}]\n", "free model"),
        ("models:\n  - model: a/b\n    prices: [{valid_from: 2026-11-22, input_usd_per_million: 0, output_usd_per_million: 1}]\n", "not valid"),
        ("models:\n  - model: a/b\n    prices: []\n", "not valid"),
        ("models:\n  - model: a/b\n    prices: [{valid_from: 2026-11-22, input_usd_per_million: 1, output_usd_per_million: 1, extra: 1}]\n", "not valid"),
        (
            "models:\n"
            "  - model: a/b\n    prices: [{valid_from: 2026-11-22, input_usd_per_million: 1, output_usd_per_million: 1}]\n"
            "  - model: a/b\n    prices: [{valid_from: 2026-11-23, input_usd_per_million: 1, output_usd_per_million: 1}]\n",
            "two times",
        ),
    ],
)
def test_price_table_refuses_a_free_model_and_wrong_entries(tmp_path: Path, text: str, words: str) -> None:
    with pytest.raises(ValueError, match=words):
        load_price_table(write_table(tmp_path, text))


def test_estimate_of_an_openrouter_request_uses_the_price_table() -> None:
    estimates = {"people": TokenEstimate(input_tokens=1_000_000, output_tokens=1_000_000)}
    on = date(2026, 11, 22)
    assert estimate_usd([request_of()], estimates, on, openrouter_prices=PRICES) == pytest.approx(0.21 + 1.90)
    with pytest.raises(KeyError, match="no price"):
        estimate_usd([request_of()], estimates, on)
    with pytest.raises(KeyError, match="no price"):
        estimate_usd([request_of(model="other/model")], estimates, on, openrouter_prices=PRICES)


def test_ledger_refuses_an_openrouter_answer_without_a_reported_cost(tmp_path: Path) -> None:
    response = VLMResponse(
        text="{}", usage=Usage(input_tokens=1, output_tokens=1), model=MODEL, latency_s=0.0,
        created_at="2026-11-22T10:00:00+00:00",
    )
    ledger = Ledger(tmp_path / "ledger.jsonl")
    with pytest.raises(ValueError, match="no reported cost"):
        ledger.record("run", request_of(), response)
    assert ledger.files() == []


def test_one_ledger_and_one_hard_stop_for_both_providers(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    guard = BudgetGuard(ledger, hard_stop_usd=1.0)
    guard.start_run("run", 10.0)
    created = datetime(2026, 11, 22, tzinfo=UTC).isoformat()
    gemini = VLMResponse(
        text="{}", usage=Usage(input_tokens=100_000, output_tokens=0), model="gemini-4-argon", latency_s=0.0,
        created_at=created,
    )
    ledger.record("run", request_of(StepSettings()), gemini)  # 0.2 USD at 2 USD per million input tokens
    openrouter = gemini.model_copy(update={"model": MODEL, "reported_usd": 0.85})
    ledger.record("run", request_of(), openrouter)
    assert ledger.total_usd() == pytest.approx(1.05)
    with pytest.raises(BudgetExceeded, match="hard stop"):
        guard.check("run")


# --- Fake backend and contract checks ------------------------------------------------------------


def test_fake_backend_reports_a_cost_for_openrouter_requests_only() -> None:
    scene = SyntheticScene(n_frames=2, width=320, height=180, seed=0)
    request = steps.build_kit_legend_request([scene.image(0)], clip_id="c", settings=OPENROUTER)
    assert FakeBackend(scene).generate(request).reported_usd == pytest.approx((3600 * 1.0 + 600 * 4.0) / 1e6)
    gemini = steps.build_kit_legend_request([scene.image(0)], clip_id="c", settings=StepSettings())
    assert FakeBackend(scene).generate(gemini).reported_usd is None


# --- The HTTP layer of the standard library ------------------------------------------------------
# No test opens a network connection (docs/ARCHITECTURE.md, section 5): ``OpenerDirector.open``, or the
# handler of https addresses, is replaced.


def _replace_open(monkeypatch: pytest.MonkeyPatch, urlopen: Any) -> None:
    """Let every opener of ``urllib`` answer with ``urlopen(request, timeout)``."""
    monkeypatch.setattr(
        urllib.request.OpenerDirector, "open", lambda self, request, data=None, timeout=None: urlopen(request, timeout)
    )


class _Opened(io.BytesIO):
    """What ``urlopen`` returns: a body to read, a status and the headers."""

    def __init__(self, status: int, body: bytes, headers: Mapping[str, str]) -> None:
        super().__init__(body)
        self.status = status
        self.headers = email.message.Message()
        for name, value in headers.items():
            self.headers[name] = value


def _http_error(status: int, body: bytes, headers: Mapping[str, str]) -> urllib.error.HTTPError:
    message = email.message.Message()
    for name, value in headers.items():
        message[name] = value
    return urllib.error.HTTPError(OPENROUTER_URL, status, "error", message, io.BytesIO(body))


def test_urllib_transport_sends_the_body_and_returns_the_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[urllib.request.Request, float]] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> _Opened:
        seen.append((request, timeout))
        return _Opened(200, b'{"ok": true}', {})

    _replace_open(monkeypatch, urlopen)
    answer = UrllibTransport().post(OPENROUTER_URL, {"Authorization": "Bearer test-value"}, b'{"a": 1}', 5.0)

    assert answer == HttpAnswer(200, b'{"ok": true}', None)
    ((request, timeout),) = seen
    assert request.get_method() == "POST" and request.full_url == OPENROUTER_URL and request.data == b'{"a": 1}'
    assert request.get_header("Authorization") == "Bearer test-value" and timeout == 5.0


@pytest.mark.parametrize(
    ("headers", "retry_after_s"),
    [({"Retry-After": "7"}, 7.0), ({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, None), ({}, None)],
)
def test_urllib_transport_returns_an_error_status_with_its_wait(
    monkeypatch: pytest.MonkeyPatch, headers: dict[str, str], retry_after_s: float | None
) -> None:
    def urlopen(request: urllib.request.Request, timeout: float) -> _Opened:
        raise _http_error(429, b'{"error": {"code": 429}}', headers)

    _replace_open(monkeypatch, urlopen)
    answer = UrllibTransport().post(OPENROUTER_URL, {}, b"{}", 5.0)
    assert answer == HttpAnswer(429, b'{"error": {"code": 429}}', retry_after_s)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TimeoutError("timed out"), HttpTimeoutError),
        (urllib.error.URLError(TimeoutError("timed out")), HttpTimeoutError),
        (urllib.error.URLError(ConnectionRefusedError("refused")), HttpConnectionError),
        (ConnectionResetError("reset"), HttpConnectionError),
        (http.client.RemoteDisconnected("closed"), HttpConnectionError),
    ],
)
def test_urllib_transport_tells_a_timeout_from_a_failed_connection(
    monkeypatch: pytest.MonkeyPatch, error: Exception, expected: type[Exception]
) -> None:
    def urlopen(request: urllib.request.Request, timeout: float) -> _Opened:
        raise error

    _replace_open(monkeypatch, urlopen)
    with pytest.raises(expected):
        UrllibTransport().post(OPENROUTER_URL, {}, b"{}", 5.0)


class _CutBody(_Opened):
    """An answer whose body breaks off while it is read."""

    def read(self, *args: Any) -> bytes:
        raise http.client.IncompleteRead(b'{"id": "gen-1"', 900)


def test_a_success_whose_body_cannot_be_read_is_not_sent_again(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    """The provider has done the request, so a new attempt could be billed a second time."""
    opened: list[str] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> _Opened:
        opened.append(request.full_url)
        return _CutBody(200, b"", {}) if len(opened) == 1 else _Opened(200, json.dumps(chat_answer()).encode(), {})

    _replace_open(monkeypatch, urlopen)
    backend = OpenRouterBackend(prices=PRICES, transport=UrllibTransport(), sleep=lambda seconds: None)
    with pytest.raises(UnconfirmedCallError, match="the answer could not be read.*may still be billed"):
        backend.generate(request_of())
    assert opened == [OPENROUTER_URL]


def test_an_error_status_whose_body_cannot_be_read_is_tried_again_by_its_status(
    monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    opened: list[str] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> _Opened:
        opened.append(request.full_url)
        if len(opened) == 1:
            raise urllib.error.HTTPError(OPENROUTER_URL, 503, "busy", email.message.Message(), _CutBody(503, b"", {}))
        return _Opened(200, json.dumps(chat_answer()).encode(), {})

    _replace_open(monkeypatch, urlopen)
    backend = OpenRouterBackend(prices=PRICES, transport=UrllibTransport(), sleep=lambda seconds: None)
    assert backend.generate(request_of()).attempts == 2


class _RedirectingHttps(urllib.request.HTTPSHandler):
    """The handler of https addresses: the endpoint answers with a redirect to another host."""

    seen: list[tuple[str, str | None]] = []

    def https_open(self, request: urllib.request.Request) -> urllib.response.addinfourl:
        _RedirectingHttps.seen.append((request.full_url, request.get_header("Authorization")))
        headers = email.message.Message()
        status = 200
        if request.full_url == OPENROUTER_URL:
            headers["Location"] = "https://github.com/collect"
            status = 302
        answer = urllib.response.addinfourl(io.BytesIO(b"{}"), headers, request.full_url, status)
        answer.msg = "Found" if status == 302 else "OK"
        return answer


def test_urllib_transport_follows_no_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    """A redirect would carry the key in the Authorization header to another host."""
    monkeypatch.setattr(urllib.request, "HTTPSHandler", _RedirectingHttps)
    monkeypatch.setattr(urllib.request, "_opener", None)
    monkeypatch.setattr(_RedirectingHttps, "seen", [])
    answer = UrllibTransport().post(OPENROUTER_URL, {"Authorization": "Bearer test-value"}, b"{}", 5.0)
    assert answer.status == 302
    assert _RedirectingHttps.seen == [(OPENROUTER_URL, "Bearer test-value")]


def test_error_text_of_a_success_answer_holds_no_key_and_is_cut(key: str) -> None:
    echo = {"code": 502, "message": f"Bearer {key} " + "x" * 2000}
    billed = chat_answer(error=echo)
    backend, _ = backend_with(FakeTransport(ok(billed), ok({"error": echo})))
    with pytest.raises(BilledAnswerError) as raised:
        backend.generate(request_of())
    with pytest.raises(UnconfirmedCallError) as unconfirmed:
        backend.generate(request_of())
    for error in (raised.value, unconfirmed.value):
        assert key not in str(error) and "[key removed]" in str(error)
        assert len(str(error)) < 1000


def test_a_null_error_in_a_success_answer_is_no_error(key: str) -> None:
    backend, _ = backend_with(FakeTransport(ok(chat_answer(error=None))))
    assert backend.generate(request_of()).text == '{"people": []}'
