"""Tests of video input: the video part, its key and cassette, both providers, money, the steps and the probe.

No test calls a model service. The backends only read the bytes of a video file, so most tests
give them a stand-in file of a few bytes that is no real video; the request shapes come from the
documentation pages of docs/DECISIONS.md (entry of 2026-11-23). The tests of ``vlm.video`` and of
the probe write real MP4 files with OpenCV into the temporary folder of the test.
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import pytest
from pydantic import ValidationError

from pitchvlm import doctor
from pitchvlm.config import StepSettings
from pitchvlm.frames import VideoFileSource
from pitchvlm.labels import load_gsr_labels
from pitchvlm.probes import (
    LabelledClip,
    VideoProbe,
    check_video_settings,
    plan_people_video,
    probe_people,
    sample_frames,
)
from pitchvlm.schemas import KitDescription, KitLegend
from pitchvlm.synth import SyntheticScene, SyntheticSource
from pitchvlm.vlm import client, steps
from pitchvlm.vlm.cache import CassetteStore
from pitchvlm.vlm.client import (
    GeminiBackend,
    OpenRouterBackend,
    build_interaction_kwargs,
    build_openrouter_body,
    keeps_inputs_private,
)
from pitchvlm.vlm.cost import (
    DEFAULT_TOKEN_ESTIMATES,
    GEMINI_VIDEO_RATES,
    Price,
    VideoRate,
    estimate_usd,
    load_price_table,
    load_video_rates,
    video_input_tokens,
    video_rate_for,
)
from pitchvlm.vlm.fake import FakeBackend
from pitchvlm.vlm.types import (
    ImagePart,
    RequestMeta,
    TextPart,
    Usage,
    VideoInputError,
    VideoPart,
    VLMRequest,
    VLMResponse,
)
from pitchvlm.vlm.video import (
    context_frames,
    file_sha256,
    frames_video_part,
    video_file_name,
    write_frames_video,
)

REPO = Path(__file__).resolve().parents[1]
NOVEMBER = date(2026, 11, 23)
OPENROUTER_MODEL = "qwen/qwen3-vl-235b-a22b-instruct"
OPENROUTER_PRICES = (Price(OPENROUTER_MODEL, date(2026, 11, 1), 0.21, 1.90),)
OPENROUTER_RATES = (
    VideoRate(
        OPENROUTER_MODEL,
        date(2026, 11, 1),
        {"low": 100, "medium": 100, "high": 100},
        10.0,
        2.0,
        takes_clips=False,
        video_usd_per_million=0.5,
        audio_usd_per_million=3.0,
    ),
)
LEGEND = KitLegend(
    team_a=KitDescription(shirt="white", shorts="white", socks="white"),
    team_b=KitDescription(shirt="red", shorts="black", socks="red"),
    referee_kit="yellow",
)


def stand_in(folder: Path, content: bytes = b"stand-in for a video file") -> tuple[Path, str]:
    """Write a stand-in video file: a few bytes that no decoder reads. The backends only read its bytes."""
    path = folder / "clip.mp4"
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def video_part(folder: Path, **changes: Any) -> VideoPart:
    path, sha = stand_in(folder)
    fields: dict[str, Any] = {"path": path, "sha256": sha, "duration_s": 2.5, "fps": 2.0, "resolution": "high"}
    fields.update(changes)
    return VideoPart(**fields)


def video_request(part: VideoPart, **changes: Any) -> VLMRequest:
    fields: dict[str, Any] = {
        "model": "gemini-4-argon",
        "thinking_level": "low",
        "parts": (TextPart(text="Find every person."), part),
        "response_schema": {"type": "object", "properties": {}, "required": []},
        "prompt_name": "people_video/v1",
        "meta": RequestMeta(clip_id="clip-a", step="people_video", frame=12, item="video-1"),
        "context": {"frames": [12], "video_frames": [0, 12, 24], "times_s": [0.5]},
    }
    fields.update(changes)
    return VLMRequest(**fields)


# --- VideoPart ------------------------------------------------------------------------


def test_video_part_counts_the_sampled_frames_of_its_range(tmp_path: Path) -> None:
    whole = video_part(tmp_path)
    assert whole.range_s == (0.0, 2.5)
    assert whole.sampled_frames(1.0) == 5  # its own rate of 2 frames per second wins over the default
    clipped = video_part(tmp_path, duration_s=10.0, start_s=2.0, end_s=4.4, fps=None)
    assert clipped.range_s == (2.0, 4.4)
    assert clipped.sampled_frames(1.0) == 3  # 2.4 s at 1 frame per second, rounded up
    assert clipped.clipped and not video_part(tmp_path, fps=None).clipped


@pytest.mark.parametrize(
    "changes",
    [
        {"mime_type": "image/jpeg"},
        {"resolution": "ultra_high"},
        {"end_s": 3.0},  # after the end of the file
        {"start_s": 2.0, "end_s": 2.0},  # empty
        {"start_s": 2.5},  # empty: starts at the end
        {"fps": 25.0},  # above the published range (0, 24]
        {"fps": 0.0},
        {"sha256": "ab" * 31},
        {"duration_s": float("inf")},
    ],
)
def test_video_part_refuses_wrong_values(tmp_path: Path, changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        video_part(tmp_path, **changes)


def test_identity_of_a_video_part_holds_no_path_and_no_bytes(tmp_path: Path) -> None:
    part = video_part(tmp_path)
    identity = json.dumps(part.identity())

    assert str(tmp_path) not in identity and "clip.mp4" not in identity
    assert part.sha256 in identity
    assert "path" not in repr(part)


# --- Cache key and cassette ------------------------------------------------------------


def test_cache_key_of_a_video_request_follows_hash_range_rate_and_resolution_but_not_the_path(tmp_path: Path) -> None:
    part = video_part(tmp_path)
    key = video_request(part).cache_key()
    moved = tmp_path / "elsewhere.mp4"
    moved.write_bytes(part.path.read_bytes())

    assert video_request(part.model_copy(update={"path": moved})).cache_key() == key
    for changes in (
        {"sha256": "cd" * 32},
        {"duration_s": 5.0, "start_s": 1.0},
        {"duration_s": 5.0, "end_s": 2.0},
        {"fps": 1.0},
        {"resolution": "low"},
    ):
        assert video_request(part.model_copy(update=changes)).cache_key() != key, changes


def test_cassette_of_a_video_request_names_the_video_without_its_path_or_bytes(tmp_path: Path) -> None:
    content = b"stand-in bytes that must never reach a cassette"
    path, sha = stand_in(tmp_path, content)
    part = VideoPart(path=path, sha256=sha, duration_s=2.5, fps=2.0)
    request = video_request(part)
    response = VLMResponse(
        text='{"frames": []}', usage=Usage(input_tokens=1, output_tokens=1), model="gemini-4-argon", latency_s=0.0,
        created_at="2026-11-23T10:00:00+00:00",
    )
    store = CassetteStore(tmp_path / "cache")

    written = store.put(request, response).read_text(encoding="utf-8")

    cassette = json.loads(written)
    assert cassette["videos"] == [part.identity()]
    assert cassette["context"] == {"frames": [12]}  # the stored context keeps the target frames only
    assert str(tmp_path) not in written and "clip.mp4" not in written
    assert base64.b64encode(content).decode() not in written and "stand-in bytes" not in written
    assert store.get(request) == response


def test_cassette_of_an_image_request_has_no_video_entry(tmp_path: Path) -> None:
    request = video_request(None, parts=(TextPart(text="x"),))  # type: ignore[arg-type]
    response = VLMResponse(
        text="{}", usage=Usage(input_tokens=1, output_tokens=1), model="gemini-4-argon", latency_s=0.0,
        created_at="2026-11-23T10:00:00+00:00",
    )
    assert "videos" not in json.loads(CassetteStore(tmp_path).put(request, response).read_text(encoding="utf-8"))


# --- Gemini ------------------------------------------------------------------------------


def test_gemini_video_goes_inline_with_its_resolution_range_and_rate(tmp_path: Path) -> None:
    part = video_part(tmp_path, duration_s=10.0, start_s=1.5, end_s=4.0)
    kwargs = build_interaction_kwargs(video_request(part))

    item = kwargs["input"][1]
    assert item == {
        "type": "video",
        "data": base64.b64encode(part.path.read_bytes()).decode("ascii"),
        "mime_type": "video/mp4",
        "resolution": "high",
        "processing": {"type": "static", "start_offset": "1.5s", "end_offset": "4s", "fps": 2.0},
    }
    assert kwargs["store"] is False
    assert "temperature" not in json.dumps(kwargs) and "top_p" not in json.dumps(kwargs)
    assert "context" not in kwargs and "times_s" not in json.dumps(kwargs)
    assert keeps_inputs_private(video_request(part))


def test_privacy_check_of_a_video_request_reads_no_video_file(tmp_path: Path) -> None:
    """The doctor asks this of every request, also of one whose video failed before its call."""
    part = video_part(tmp_path, fps=None)
    part.path.unlink()

    assert keeps_inputs_private(video_request(part))
    assert keeps_inputs_private(openrouter_video_request(part))


def test_gemini_video_without_range_and_rate_sends_no_processing(tmp_path: Path) -> None:
    item = build_interaction_kwargs(video_request(video_part(tmp_path, fps=None)))["input"][1]
    assert "processing" not in item


def test_a_changed_or_missing_video_file_is_refused_before_any_call(tmp_path: Path) -> None:
    part = video_part(tmp_path)
    part.path.write_bytes(b"other bytes")
    with pytest.raises(VideoInputError, match="other bytes"):
        build_interaction_kwargs(video_request(part))
    part.path.unlink()
    with pytest.raises(VideoInputError, match="cannot be read"):
        build_interaction_kwargs(video_request(part))
    with pytest.raises(VideoInputError, match="cannot be read"):
        build_openrouter_body(video_request(part.model_copy(update={"fps": None}), provider="openrouter", model=OPENROUTER_MODEL))


def test_inline_data_above_the_limit_is_refused_for_both_providers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    part = video_part(tmp_path, fps=None)
    monkeypatch.setattr(client, "INLINE_REQUEST_MAX_BYTES", 10)
    with pytest.raises(VideoInputError, match="never uploads"):
        build_interaction_kwargs(video_request(part))
    with pytest.raises(VideoInputError, match="never uploads"):
        build_openrouter_body(video_request(part, provider="openrouter", model=OPENROUTER_MODEL))
    # A request of images only has no such limit, as before video parts were added.
    build_interaction_kwargs(video_request(part, parts=(TextPart(text="x"), ImagePart(data=b"j" * 100))))


class RecordingInteractions:
    """Stands for ``client.interactions``: it keeps every call and answers with a fixed interaction."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        usage = SimpleNamespace(
            total_input_tokens=2000, total_output_tokens=300, total_thought_tokens=100, total_cached_tokens=0,
            total_tokens=2400,
        )
        return SimpleNamespace(output_text='{"frames": []}', status="completed", usage=usage, service_tier=None)


def test_gemini_backend_sends_a_video_of_a_model_with_a_video_rate(tmp_path: Path) -> None:
    interactions = RecordingInteractions()
    backend = GeminiBackend(client=SimpleNamespace(interactions=interactions))

    response = backend.generate(video_request(video_part(tmp_path)))

    assert response.usage.input_tokens == 2000
    assert [call["input"][1]["type"] for call in interactions.calls] == ["video"]
    assert interactions.calls[0]["store"] is False


def test_gemini_model_without_a_video_rate_is_refused_before_any_call(tmp_path: Path) -> None:
    """The robotics model has a price but no video rate: it takes no video here."""
    interactions = RecordingInteractions()
    backend = GeminiBackend(client=SimpleNamespace(interactions=interactions))

    with pytest.raises(VideoInputError, match="no video rate"):
        backend.generate(video_request(video_part(tmp_path), model="gemini-robotics-er-2-preview"))
    assert interactions.calls == []


# --- OpenRouter --------------------------------------------------------------------------


class RecordingTransport:
    """An HTTP layer that keeps every request and answers with one chat completion."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    def post(self, url: str, headers: Any, body: bytes, timeout_s: float) -> client.HttpAnswer:
        self.bodies.append(json.loads(body))
        answer = {
            "provider": "DeepInfra",
            "choices": [{"finish_reason": "stop", "message": {"content": '{"frames": []}'}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 200, "total_tokens": 1100, "cost": 0.0006},
        }
        return client.HttpAnswer(200, json.dumps(answer).encode("utf-8"))


def openrouter_video_request(part: VideoPart) -> VLMRequest:
    return video_request(part, provider="openrouter", model=OPENROUTER_MODEL, thinking_level="model_default")


def test_openrouter_video_is_a_video_url_part_with_a_data_url(tmp_path: Path) -> None:
    part = video_part(tmp_path, fps=None)
    body = build_openrouter_body(openrouter_video_request(part))

    content = body["messages"][0]["content"]
    data = base64.b64encode(part.path.read_bytes()).decode("ascii")
    # The address is checked in two pieces: the guard refuses a data address with its encoding in a tracked file.
    assert content[1]["type"] == "video_url" and set(content[1]["video_url"]) == {"url"}
    prefix, payload = content[1]["video_url"]["url"].split(",", 1)
    assert (prefix, payload) == ("data:video/mp4" + ";base64", data)
    assert body["provider"] == dict(client.OPENROUTER_ROUTING)
    assert keeps_inputs_private(openrouter_video_request(part))


def test_openrouter_video_with_a_range_or_a_rate_is_refused(tmp_path: Path) -> None:
    with pytest.raises(VideoInputError, match="whole file"):
        build_openrouter_body(openrouter_video_request(video_part(tmp_path)))


def test_openrouter_backend_refuses_a_model_without_a_video_rate_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk" + "-or-v1-" + "5e" * 32)
    transport = RecordingTransport()
    backend = OpenRouterBackend(prices=OPENROUTER_PRICES, transport=transport)

    with pytest.raises(VideoInputError, match="no video rate"):
        backend.generate(openrouter_video_request(video_part(tmp_path, fps=None)))
    assert transport.bodies == []

    with_rate = OpenRouterBackend(prices=OPENROUTER_PRICES, video_rates=OPENROUTER_RATES, transport=transport)
    with pytest.raises(VideoInputError, match="no range and no sampling rate"):
        with_rate.generate(openrouter_video_request(video_part(tmp_path)))
    assert transport.bodies == []
    response = with_rate.generate(openrouter_video_request(video_part(tmp_path, fps=None)))
    assert response.reported_usd == 0.0006
    assert transport.bodies[0]["messages"][0]["content"][1]["type"] == "video_url"


# --- Money -------------------------------------------------------------------------------


def test_a_new_video_rate_holds_from_its_first_day() -> None:
    """As a price, a video rate holds from its own start day on: an estimate on that day must not use the old rate."""
    new_day = date(2026, 12, 1)
    newer = VideoRate(
        OPENROUTER_MODEL,
        new_day,
        {"low": 200, "medium": 200, "high": 200},
        10.0,
        2.0,
        takes_clips=False,
        video_usd_per_million=1.0,
        audio_usd_per_million=3.0,
    )
    table = (*OPENROUTER_RATES, newer)

    assert video_rate_for(OPENROUTER_MODEL, date(2026, 11, 30), table) is OPENROUTER_RATES[0]
    assert video_rate_for(OPENROUTER_MODEL, new_day, table) is newer
    with pytest.raises(KeyError, match="its rates start on 2026-11-01"):
        video_rate_for(OPENROUTER_MODEL, date(2026, 10, 31), table)


def test_video_tokens_follow_the_published_gemini_rate(tmp_path: Path) -> None:
    rate = video_rate_for("gemini-4-argon", NOVEMBER, GEMINI_VIDEO_RATES)
    # 5 frames at 2 frames per second: 5 x 280 video tokens at high, and 2.5 s x 32 sound tokens.
    assert video_input_tokens(video_part(tmp_path), rate) == (1400, 80)
    assert video_input_tokens(video_part(tmp_path, resolution="low"), rate) == (350, 80)
    # Without a rate of its own a part is sampled at the default rate of 1 frame per second.
    assert video_input_tokens(video_part(tmp_path, fps=None), rate) == (3 * 280, 80)


def test_estimate_of_a_gemini_video_request_adds_the_video_tokens_at_the_input_price(tmp_path: Path) -> None:
    request = video_request(video_part(tmp_path), context={"frames": [12, 24]})
    text = DEFAULT_TOKEN_ESTIMATES["people_video"]
    # gemini-4-argon on 2026-11-23: 2 USD in, 10 USD out per million; two target frames double the output.
    expected = (text.input_tokens * 2.0 + 2 * text.output_tokens * 10.0 + (1400 + 80) * 2.0) / 1e6

    assert estimate_usd([request], DEFAULT_TOKEN_ESTIMATES, NOVEMBER) == pytest.approx(expected)
    flex = request.model_copy(update={"service_tier": "flex"})
    assert estimate_usd([flex], DEFAULT_TOKEN_ESTIMATES, NOVEMBER) == pytest.approx(expected / 2)
    # The prices of gemini-4-argon double on 2026-12-01, and so does the estimate.
    assert estimate_usd([request], DEFAULT_TOKEN_ESTIMATES, date(2026, 12, 1)) == pytest.approx(2 * expected)


def test_estimate_of_an_openrouter_video_request_uses_the_video_and_sound_prices(tmp_path: Path) -> None:
    request = openrouter_video_request(video_part(tmp_path, fps=None))
    text = DEFAULT_TOKEN_ESTIMATES["people_video"]
    # 2.5 s at the provider rate of 2 frames per second: 5 x 100 video tokens and 25 sound tokens.
    expected = (text.input_tokens * 0.21 + text.output_tokens * 1.90 + 500 * 0.5 + 25 * 3.0) / 1e6

    estimate = estimate_usd(
        [request], DEFAULT_TOKEN_ESTIMATES, NOVEMBER, openrouter_prices=OPENROUTER_PRICES,
        openrouter_video_rates=OPENROUTER_RATES,
    )
    assert estimate == pytest.approx(expected)


def test_a_model_without_a_video_rate_has_no_estimate(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="no video rate"):
        estimate_usd(
            [video_request(video_part(tmp_path), model="gemini-robotics-er-2-preview")],
            DEFAULT_TOKEN_ESTIMATES,
            NOVEMBER,
        )
    with pytest.raises(KeyError, match="no video rate"):
        estimate_usd(
            [openrouter_video_request(video_part(tmp_path, fps=None))],
            DEFAULT_TOKEN_ESTIMATES,
            NOVEMBER,
            openrouter_prices=OPENROUTER_PRICES,
        )


def test_video_rates_are_read_from_the_price_table_file(tmp_path: Path) -> None:
    path = tmp_path / "prices.yaml"
    path.write_text(
        "models:\n"
        f"  - model: {OPENROUTER_MODEL}\n"
        "    prices:\n      - {valid_from: 2026-11-01, input_usd_per_million: 0.21, output_usd_per_million: 1.9}\n"
        "    video:\n"
        "      - {valid_from: 2026-11-01, tokens_per_frame: 100, audio_tokens_per_second: 10, frames_per_second: 2,\n"
        "         video_usd_per_million: 0.5, audio_usd_per_million: 3.0}\n",
        encoding="utf-8",
    )

    (rate,) = load_video_rates(path)
    assert (rate.model, rate.default_fps, rate.takes_clips) == (OPENROUTER_MODEL, 2.0, False)
    assert dict(rate.tokens_per_frame) == {"low": 100, "medium": 100, "high": 100}
    assert len(load_price_table(path)) == 1
    path.write_text(path.read_text(encoding="utf-8").replace("tokens_per_frame: 100", "tokens_per_frame: 0"), "utf-8")
    with pytest.raises(ValueError, match="prices.yaml"):
        load_video_rates(path)


def test_the_price_table_of_the_repository_names_no_openrouter_video_rate_yet() -> None:
    """No page publishes a video token rate of these models: the owner adds one (docs/DECISIONS.md, 2026-11-23)."""
    assert load_video_rates(REPO / "configs" / "openrouter_prices.yaml") == ()


# --- Steps -------------------------------------------------------------------------------


def test_people_video_request_names_its_targets_by_their_time(tmp_path: Path) -> None:
    part = video_part(tmp_path, duration_s=2.5, fps=2.0)
    request = steps.build_people_video_request(
        part, LEGEND, clip_id="clip-a", video_frames=[0, 12, 24, 36, 48], targets=[24, 48], fps=2.0,
        width=1920, height=1080, settings=StepSettings(resolution="high"),
    )

    text = request.parts[0].text  # type: ignore[union-attr]
    assert "5 frames, at 2 frames per second" in text and "1.00 s, 2.00 s" in text
    assert "Do not use faces and do not give names" in text and "shirt number" in text
    assert request.parts[1] == part
    assert request.context == {"frames": [24, 48], "video_frames": [0, 12, 24, 36, 48], "times_s": [1.0, 2.0]}
    assert request.meta.step == "people_video" and request.meta.item == f"video-{part.sha256[:16]}"
    schema = request.response_schema["properties"]["frames"]
    assert (schema["minItems"], schema["maxItems"]) == (2, 2)
    assert request.stored_context() == {"frames": [24, 48]}


@pytest.mark.parametrize(
    "changes",
    [
        {"targets": [7]},  # not a frame of the video
        {"targets": [12, 12]},
        {"targets": []},
        {"fps": 1.0},  # 5 frames at 1 frame per second are 5 s, the part is 2.5 s
    ],
)
def test_people_video_request_refuses_targets_and_rates_that_do_not_fit_the_video(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    arguments: dict[str, Any] = {
        "clip_id": "c", "video_frames": [0, 12, 24, 36, 48], "targets": [24], "fps": 2.0, "width": 64, "height": 48,
        "settings": StepSettings(),
    }
    arguments.update(changes)
    with pytest.raises(ValueError):
        steps.build_people_video_request(video_part(tmp_path), LEGEND, **arguments)


def test_parse_people_video_maps_entries_to_targets_by_their_time() -> None:
    person = {"box_2d": [100, 100, 500, 200], "role": "player", "team": "A", "number": 7}
    answer = {
        "frames": [
            {"time_s": 1.0, "people": [person]},
            {"time_s": 3.0, "people": [person]},  # another frame: more than half an interval (0.25 s) off
        ]
    }
    people, wrong = steps.parse_people_video(
        json.dumps(answer), targets=[24, 48, 60], times_s=[1.0, 2.0, 2.5], fps=2.0, width=1000, height=1000,
        coords="yx_1000",
    )

    assert [answer.frame for answer in people] == [24, 48, 60]
    assert [len(answer.people) for answer in people] == [1, 0, 0]
    assert wrong == 2  # one entry with the wrong time, one missing entry
    with pytest.raises(steps.ParseError):
        steps.parse_people_video("{}", targets=[1], times_s=[0.0], fps=1.0, width=10, height=10, coords="yx_1000")


# --- The video files ---------------------------------------------------------------------


def test_written_video_has_the_frames_and_keeps_its_bytes(tmp_path: Path) -> None:
    scene = SyntheticScene(n_frames=30, width=320, height=180, seed=1)
    source = SyntheticSource(scene)
    path = tmp_path / video_file_name(scene.info.clip_id, "people_video", [0, 12, 24], 2.0, "pixels")

    write_frames_video(source, [0, 12, 24], path, 2.0)
    first = file_sha256(path)
    read = VideoFileSource(path)
    assert (read.info.n_frames, read.info.width, read.info.height) == (3, 320, 180)
    assert read.info.fps == pytest.approx(2.0, rel=1e-2)

    # A second write of the same frames to another file gives the same bytes (build machine, OpenCV 5.0) ...
    again = tmp_path / "again.mp4"
    write_frames_video(source, [0, 12, 24], again, 2.0)
    assert file_sha256(again) == first
    # ... and a file that is there already is kept as it is.
    path.write_bytes(b"kept")
    write_frames_video(source, [0, 12, 24], path, 2.0)
    assert path.read_bytes() == b"kept"
    assert not list(tmp_path.glob(".*partial*"))


def test_frames_video_part_names_the_file_by_its_clip_frames_and_rate(tmp_path: Path) -> None:
    source = SyntheticSource(SyntheticScene(n_frames=30, width=320, height=180, seed=1))
    part = frames_video_part(
        source, [0, 12, 24], folder=tmp_path, step="people_video", fps=2.0, resolution="low", send_fps=True
    )
    assert part.path.parent == tmp_path / source.info.clip_id
    assert (part.duration_s, part.fps, part.resolution) == (1.5, 2.0, "low")
    assert part.sha256 == file_sha256(part.path)
    other = frames_video_part(
        source, [0, 12, 24], folder=tmp_path, step="people_video", fps=1.0, resolution="low", send_fps=False
    )
    assert other.path != part.path and other.fps is None and other.duration_s == 3.0


def test_new_pixels_under_an_old_clip_name_get_a_new_video_file(tmp_path: Path) -> None:
    """A clip replaced under the same name must not reuse the old file: its hash would hit the old answers."""
    old = SyntheticSource(SyntheticScene(n_frames=30, width=320, height=180, seed=1))
    new = SyntheticSource(SyntheticScene(n_frames=30, width=320, height=180, seed=2))
    assert old.info.clip_id == new.info.clip_id
    arguments: dict[str, Any] = {"folder": tmp_path, "step": "people_video", "fps": 2.0, "resolution": "low",
                                 "send_fps": True}

    first = frames_video_part(old, [0, 12, 24], **arguments)
    second = frames_video_part(new, [0, 12, 24], **arguments)

    assert second.path != first.path and second.sha256 != first.sha256
    assert frames_video_part(old, [0, 12, 24], **arguments) == first


def test_context_frames_are_cut_at_the_edges_of_the_clip() -> None:
    assert context_frames(24, n_frames=100, clip_fps=25.0, video_fps=2.0, context=2) == [0, 12, 24, 36, 48]
    assert context_frames(5, n_frames=100, clip_fps=25.0, video_fps=2.0, context=2) == [5, 17, 29]
    assert context_frames(99, n_frames=100, clip_fps=25.0, video_fps=2.0, context=0) == [99]
    with pytest.raises(ValueError):
        context_frames(5, n_frames=100, clip_fps=25.0, video_fps=2.0, context=-1)


# --- Settings ----------------------------------------------------------------------------


def test_video_settings_must_fit_the_step_and_its_provider() -> None:
    check_video_settings(StepSettings(resolution="low"), VideoProbe(fps=2.0))
    openrouter = StepSettings(provider="openrouter", model=OPENROUTER_MODEL, coords="xy_1000")
    check_video_settings(openrouter, VideoProbe())
    assert VideoProbe().write_fps == 1.0 and VideoProbe(fps=2.0).write_fps == 2.0
    for settings, video in (
        (StepSettings(), VideoProbe()),  # a Gemini video step needs its rate
        (StepSettings(resolution="ultra_high"), VideoProbe(fps=2.0)),
        (openrouter, VideoProbe(fps=2.0)),  # OpenRouter cannot send a rate
    ):
        with pytest.raises(ValueError):
            check_video_settings(settings, video)
    with pytest.raises(ValidationError):
        VideoProbe(fps=30.0)


def test_step_settings_of_the_pipeline_have_no_video_input() -> None:
    """The pipeline sends images: its settings file and its defaults did not change."""
    assert set(StepSettings.model_fields) == {
        "provider", "model", "thinking_level", "resolution", "service_tier", "coords"
    }


# --- Fake backend, probe and doctor --------------------------------------------------------


@pytest.fixture(scope="module")
def scene() -> SyntheticScene:
    """A synthetic clip of 100 frames: 4 seconds at 25 frames per second, 1280 x 720 pixels."""
    return SyntheticScene(seed=3)


@pytest.fixture(scope="module")
def labelled(scene: SyntheticScene, tmp_path_factory: pytest.TempPathFactory) -> LabelledClip:
    path = tmp_path_factory.mktemp("labels") / scene.info.clip_id / "Labels-GameState.json"
    path.parent.mkdir()
    path.write_text(json.dumps(scene.gsr_labels()), encoding="utf-8")
    return LabelledClip(SyntheticSource(scene), load_gsr_labels(path))


class CountingBackend:
    """A fake backend that keeps every request."""

    def __init__(self, inner: FakeBackend) -> None:
        self.inner = inner
        self.requests: list[VLMRequest] = []

    def generate(self, request: VLMRequest) -> VLMResponse:
        self.requests.append(request)
        return self.inner.generate(request)


VIDEO_SETTINGS = StepSettings(resolution="low")


@pytest.mark.parametrize(("per_request", "n_requests"), [(1, 5), (5, 1)])
def test_video_people_probe_measures_the_frames_of_the_image_probe(
    labelled: LabelledClip, scene: SyntheticScene, tmp_path: Path, per_request: int, n_requests: int
) -> None:
    """End to end with the fake model: the same frames, the same truth and the same values as with images."""
    images = probe_people([labelled], FakeBackend(scene), StepSettings(), frames_per_clip=5, seed=1,
                          small_box_max_height_px=40.0)
    backend = CountingBackend(FakeBackend(scene))
    video = VideoProbe(fps=2.0, context_frames=2, targets_per_request=per_request)

    result = probe_people(
        [labelled], backend, VIDEO_SETTINGS, frames_per_clip=5, seed=1, small_box_max_height_px=40.0,
        video=video, video_folder=tmp_path,
    )

    steps_sent = [request.meta.step for request in backend.requests]
    assert steps_sent == ["kit_legend_video"] + ["people_video"] * n_requests
    assert all(len(request.videos) == 1 for request in backend.requests)
    assert result["n_people_requests"] == n_requests and result["n_entries_wrong_time"] == 0
    assert result["prompts"] == {"kit_legend_video": "kit_legend_video/v1", "people_video": "people_video/v1"}
    assert result["settings"]["video"] == {
        "fps": 2.0, "context_frames": 2, "targets_per_request": per_request, "write_fps": 2.0
    }
    targets = [frame for request in backend.requests[1:] for frame in request.context["frames"]]
    assert targets == sample_frames(labelled.labels, 5, 1)
    for key in ("recall", "precision", "counts", "n_frames", "foot_error_m"):
        assert result[key] == images[key], key
    # Every video is a file of the video folder, outside the repository.
    assert all(request.videos[0].path.is_relative_to(tmp_path) for request in backend.requests)


def test_video_people_probe_refuses_wrong_video_settings_before_any_request(
    labelled: LabelledClip, scene: SyntheticScene
) -> None:
    backend = CountingBackend(FakeBackend(scene))
    with pytest.raises(ValueError, match="folder of its videos"):
        probe_people([labelled], backend, VIDEO_SETTINGS, frames_per_clip=5, seed=1, small_box_max_height_px=40.0,
                     video=VideoProbe(fps=2.0))
    with pytest.raises(ValueError, match="needs the rate"):
        probe_people([labelled], backend, VIDEO_SETTINGS, frames_per_clip=5, seed=1, small_box_max_height_px=40.0,
                     video=VideoProbe(), video_folder=Path("unused"))
    assert backend.requests == []


def test_plan_of_a_video_probe_gives_the_requests_of_the_probe_for_the_estimate(
    labelled: LabelledClip, scene: SyntheticScene, tmp_path: Path
) -> None:
    video = VideoProbe(fps=2.0, context_frames=2, targets_per_request=1)
    planned = plan_people_video(
        [labelled], VIDEO_SETTINGS, frames_per_clip=5, seed=1, video=video, folder=tmp_path
    )
    backend = CountingBackend(FakeBackend(scene))
    probe_people([labelled], backend, VIDEO_SETTINGS, frames_per_clip=5, seed=1, small_box_max_height_px=40.0,
                 video=video, video_folder=tmp_path)

    assert [request.meta.step for request in planned] == [request.meta.step for request in backend.requests]
    assert [request.videos for request in planned] == [request.videos for request in backend.requests]
    assert [request.context["frames"] for request in planned] == [request.context["frames"] for request in backend.requests]
    assert estimate_usd(planned, DEFAULT_TOKEN_ESTIMATES, NOVEMBER) > 0


class WrongTimes(FakeBackend):
    """A fake model that answers every video entry with a time one second late."""

    def generate(self, request: VLMRequest) -> VLMResponse:
        response = super().generate(request)
        if request.meta.step != "people_video":
            return response
        answer = json.loads(response.text)
        for entry in answer["frames"]:
            entry["time_s"] += 1.0
        return response.model_copy(update={"text": json.dumps(answer)})


def contract_limits() -> doctor.ContractLimits:
    return doctor.ContractLimits(
        box_mean_iou_min=0.6, point_error_max_fraction=0.01, usage_sum_tolerance=0.01, flex_valid_answer_rate_min=0.95
    )


def checks_by_name(result: doctor.ContractResult) -> dict[str, doctor.CheckResult]:
    return {check.name: check for check in result.checks}


@pytest.fixture(scope="module")
def contract_scene() -> SyntheticScene:
    return doctor.contract_scene()


def test_doctor_video_check_passes_with_the_fake_model(contract_scene: SyntheticScene, tmp_path: Path) -> None:
    settings = StepSettings()
    request = doctor.video_contract_request(settings, contract_scene, tmp_path)
    assert request is not None and request.context["frames"] == list(doctor.CONTRACT_FRAMES)

    result = doctor.contract_checks(
        FakeBackend(contract_scene), contract_scene, settings, contract_limits(), video_request=request,
        check_video_input=True,
    )

    checks = checks_by_name(result)
    assert checks[doctor.VIDEO_BOXES].status == "ok"
    assert checks[doctor.VIDEO_TIMES].status == "ok"
    assert checks[doctor.VIDEO_TOKENS].status == "info"
    # 600 text tokens, 7 frames x 280 tokens at high, 7 s x 32 sound tokens.
    assert checks[doctor.VIDEO_TOKENS].limit == 600 + 7 * 280 + 7 * 32
    assert [answer["step"] for answer in result.answers][-1] == "people_video"


def test_doctor_plans_a_video_request_for_an_openrouter_model_from_the_rates_of_the_price_file(
    contract_scene: SyntheticScene, tmp_path: Path
) -> None:
    """An OpenRouter model takes its video rate from the price file, not from the Gemini table: with a rate it
    gets the video check (a whole file, no sampling rate), and its estimate counts that request."""
    settings = StepSettings(provider="openrouter", model=OPENROUTER_MODEL, coords="xy_1000")

    request = doctor.video_contract_request(settings, contract_scene, tmp_path, OPENROUTER_RATES)

    assert request is not None and request.provider == "openrouter"
    assert [part.fps for part in request.videos] == [None]
    assert doctor.video_contract_request(settings, contract_scene, tmp_path / "none", ()) is None


def test_doctor_video_check_fails_when_the_model_names_other_times(
    contract_scene: SyntheticScene, tmp_path: Path
) -> None:
    request = doctor.video_contract_request(StepSettings(), contract_scene, tmp_path)
    result = doctor.contract_checks(
        WrongTimes(contract_scene), contract_scene, StepSettings(), contract_limits(), video_request=request,
        check_video_input=True,
    )
    checks = checks_by_name(result)
    assert checks[doctor.VIDEO_BOXES].status == "fail"  # every target counts as no people
    assert checks[doctor.VIDEO_TIMES].status == "fail"


def test_doctor_video_check_of_a_model_without_video_fails_without_a_call(
    contract_scene: SyntheticScene, tmp_path: Path
) -> None:
    settings = StepSettings(model="gemini-robotics-er-2-preview", thinking_level="low")
    assert doctor.video_contract_request(settings, contract_scene, tmp_path) is None
    assert not list(tmp_path.iterdir())  # no video was written for it
    backend = CountingBackend(FakeBackend(contract_scene))

    result = doctor.contract_checks(
        backend, contract_scene, settings, contract_limits(), video_request=None, check_video_input=True
    )

    assert checks_by_name(result)[doctor.VIDEO_BOXES].status == "fail"
    assert "people_video" not in [request.meta.step for request in backend.requests]


def test_doctor_reports_a_video_that_fails_before_its_call_and_finishes(
    contract_scene: SyntheticScene, tmp_path: Path
) -> None:
    request = doctor.video_contract_request(StepSettings(), contract_scene, tmp_path)
    assert request is not None
    request.videos[0].path.write_bytes(b"changed after the request was built")
    interactions = RecordingInteractions()
    backend = GeminiBackend(client=SimpleNamespace(interactions=interactions))

    checks = doctor.check_video(backend, contract_scene, StepSettings(), request, min_mean_iou=0.6)
    assert [(check.name, check.status) for check in checks] == [(doctor.VIDEO_BOXES, "fail")]
    assert "other bytes" in checks[0].detail and interactions.calls == []
    # The checks of the raw calls read every request again: a failed video request does not stop them.
    assert doctor.check_store_flag([request], []).name == doctor.STORE


def test_doctor_without_the_video_option_sends_no_video(contract_scene: SyntheticScene) -> None:
    backend = CountingBackend(FakeBackend(contract_scene))
    result = doctor.contract_checks(backend, contract_scene, StepSettings(), contract_limits())
    assert doctor.VIDEO_BOXES not in checks_by_name(result)
    assert all(not request.videos for request in backend.requests)
