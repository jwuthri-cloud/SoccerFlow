"""Tests of the cassette store and of the caching backend."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import random
import re
import shutil
import string
import subprocess
import tomllib
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from pitchvlm.vlm import cache as cache_module
from pitchvlm.vlm.cache import CachedBackend, CassetteStore
from pitchvlm.vlm.types import (
    BackendError,
    BilledAnswerError,
    CacheMiss,
    ImagePart,
    OtherImagesError,
    RequestMeta,
    ServiceTier,
    TextPart,
    Usage,
    VLMRequest,
    VLMResponse,
)

LOGGER_NAME = "pitchvlm.vlm.cache"
#: The rules of the secret scan of the git hooks. Cassettes go to the results branch through these hooks.
SECRET_SCAN_FILE = Path(__file__).resolve().parents[1] / ".gitleaks.toml"
#: Stand-in for the encoded bytes of an image. Random bytes cannot be in a text file by accident.
IMAGE_BYTES = random.Random(0).randbytes(4096)
#: Other pixels for the same question. The cache key does not see them.
OTHER_IMAGE_BYTES = random.Random(1).randbytes(4096)
#: Stand-in for the pitch diagram, the first image of a landmarks request.
DIAGRAM_BYTES = random.Random(2).randbytes(512)
#: The images of the tests by name.
IMAGES = {"frame": IMAGE_BYTES, "other": OTHER_IMAGE_BYTES, "diagram": DIAGRAM_BYTES}


def make_request(
    *,
    text: str = "Find every person.",
    model: str = "gemini-4-argon",
    repeat: int = 0,
    service_tier: ServiceTier = "standard",
    frame: int = 12,
    clip_id: str = "clip-1",
    image: bytes = IMAGE_BYTES,
) -> VLMRequest:
    """Build a small people request: one text part and one image part."""
    return VLMRequest(
        model=model,
        thinking_level="low",
        parts=(TextPart(text=text), ImagePart(data=image, resolution="high")),
        response_schema={"type": "object", "required": ["people"]},
        prompt_name="people/v1",
        meta=RequestMeta(clip_id=clip_id, step="people", frame=frame, repeat=repeat),
        service_tier=service_tier,
    )


def make_landmarks_request(diagram: bytes = DIAGRAM_BYTES, frame_image: bytes = IMAGE_BYTES) -> VLMRequest:
    """Build a small landmarks request: one text part, then the pitch diagram and the frame."""
    return VLMRequest(
        model="gemini-4-argon",
        thinking_level="low",
        parts=(
            TextPart(text="Find the pitch marks."),
            ImagePart(data=diagram, resolution="medium"),
            ImagePart(data=frame_image, resolution="high"),
        ),
        response_schema={"type": "object", "required": ["landmarks"]},
        prompt_name="landmarks/v1",
        meta=RequestMeta(clip_id="clip-1", step="landmarks", frame=12),
    )


def make_response(text: str = '{"people": []}') -> VLMResponse:
    """Build an answer with the given JSON text."""
    return VLMResponse(
        text=text,
        usage=Usage(input_tokens=1500, output_tokens=300, thought_tokens=100),
        model="gemini-4-argon",
        latency_s=1.5,
        attempts=2,
        sdk_version="1.2.3",
        created_at="2026-11-20T10:00:00+00:00",
    )


class CountingBackend:
    """A backend that answers with the frame number of the request and keeps a list of its calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, request: VLMRequest) -> VLMResponse:
        self.calls.append(request.cache_key())
        return make_response(json.dumps({"frame": request.meta.frame}))


class RejectingBackend:
    """A backend whose answers are billed and then rejected: every call raises ``BilledAnswerError``."""

    def __init__(self) -> None:
        #: The cost record of every call, in the order of the calls: an answer without text.
        self.billed: list[VLMResponse] = []

    def generate(self, request: VLMRequest) -> VLMResponse:
        billed = make_response("")
        self.billed.append(billed)
        raise BilledAnswerError(f"call {len(self.billed)}: the answer has no text", billed)


class FailingBackend:
    """A backend whose calls fail before the model answers: no answer and no cost."""

    def generate(self, request: VLMRequest) -> VLMResponse:
        raise BackendError("the call failed")


def answered_frame(response: VLMResponse) -> int:
    """Return the frame number inside an answer of ``CountingBackend``."""
    return json.loads(response.text)["frame"]


def cache_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Return the log records of the cache module."""
    return [record for record in caplog.records if record.name == LOGGER_NAME]


def files_below(folder: Path) -> list[str]:
    """Return every file below a folder as a path relative to it, in sorted order."""
    return sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file())


@pytest.fixture
def read_only_folder(tmp_path: Path) -> Iterator[Path]:
    """A folder that cannot be written. The test is skipped where the user may write everywhere."""
    folder = tmp_path / "read-only"
    folder.mkdir()
    folder.chmod(0o555)
    if os.access(folder, os.W_OK):
        folder.chmod(0o755)
        pytest.skip("this user can write into a folder without the write permission")
    yield folder
    folder.chmod(0o755)


# --- Cache key ------------------------------------------------------------------


def test_cache_key_is_stable() -> None:
    assert make_request().cache_key() == make_request().cache_key()


def test_cache_key_changes_with_prompt_text_model_and_repeat() -> None:
    base = make_request().cache_key()

    assert make_request(text="Find every player.").cache_key() != base
    assert make_request(model="gemini-4-carbon").cache_key() != base
    assert make_request(repeat=1).cache_key() != base
    assert make_request(frame=13).cache_key() != base


def test_cache_key_ignores_service_tier_and_pixels() -> None:
    base = make_request().cache_key()

    assert make_request(service_tier="flex").cache_key() == base
    assert make_request(image=OTHER_IMAGE_BYTES).cache_key() == base


# --- CassetteStore --------------------------------------------------------------


def test_path_is_root_clip_step_key(tmp_path: Path) -> None:
    request = make_request()
    store = CassetteStore(tmp_path)

    assert store.path_for(request) == tmp_path / "clip-1" / "people" / f"{request.cache_key()}.json"
    # Asking for a path creates nothing.
    assert list(tmp_path.iterdir()) == []


def test_path_does_not_depend_on_the_service_tier(tmp_path: Path) -> None:
    store = CassetteStore(tmp_path)

    assert store.path_for(make_request(service_tier="flex")) == store.path_for(make_request())
    assert store.path_for(make_request(repeat=1)) != store.path_for(make_request())


@pytest.mark.parametrize("clip_id", ["", ".", "..", "a/b", "a\\b"])
def test_clip_id_must_be_one_folder_name(tmp_path: Path, clip_id: str) -> None:
    with pytest.raises(ValueError):
        CassetteStore(tmp_path).path_for(make_request(clip_id=clip_id))


def test_put_then_get_gives_the_stored_answer(tmp_path: Path) -> None:
    store = CassetteStore(tmp_path)
    request, response = make_request(), make_response()

    assert store.get(request) is None
    path = store.put(request, response)

    assert path == store.path_for(request)
    assert path.is_file()
    assert store.get(request) == response
    # Another store object on the same folder finds it too, also for the other service tier.
    assert CassetteStore(tmp_path).get(make_request(service_tier="flex")) == response
    assert store.get(make_request(repeat=1)) is None


def test_cassette_content(tmp_path: Path) -> None:
    request, response = make_request(), make_response()
    path = CassetteStore(tmp_path).put(request, response)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "key": request.cache_key(),
        "meta": {"clip_id": "clip-1", "step": "people", "frame": 12, "item": None, "repeat": 0},
        "context": {},
        "model": "gemini-4-argon",
        "thinking_level": "low",
        "prompt_name": "people/v1",
        "prompt_hash": request.prompt_hash,
        "resolutions": ["high"],
        "image_hashes": [hashlib.sha256(IMAGE_BYTES).hexdigest()],
        "response": {
            "text": '{"people": []}',
            "usage": {"input_tokens": 1500, "output_tokens": 300, "thought_tokens": 100, "cached_tokens": 0},
            "model": "gemini-4-argon",
            "service_tier": "standard",
            "latency_s": 1.5,
            "attempts": 2,
            "sdk_version": "1.2.3",
            "created_at": "2026-11-20T10:00:00+00:00",
        },
    }


def test_cassette_holds_the_frames_of_the_context_and_no_box(tmp_path: Path) -> None:
    """The cache key of a shirt-number request stands for crops. The cassette names the frames of the crops.

    The boxes of the crops stay out of the cassette. A probe cuts its crops at the boxes of the label
    files, and label data must not be part of a result file.
    """
    boxes = [[101.625, 213.375, 157.875, 409.125], [101.625, 277.625, 157.875, 409.125]]
    context = {"frames": [10, 35], "boxes": boxes}
    request = VLMRequest(
        model="gemini-4-argon",
        thinking_level="low",
        parts=(
            TextPart(text="Read the shirt number."),
            ImagePart(data=IMAGE_BYTES, resolution="low"),
            ImagePart(data=OTHER_IMAGE_BYTES, resolution="low"),
        ),
        response_schema={"type": "object", "required": ["number"]},
        prompt_name="number/v1",
        meta=RequestMeta(clip_id="clip-1", step="number", item="track-7-0a1b2c3d"),
        context=context,
    )
    store = CassetteStore(tmp_path)
    response = make_response('{"per_crop": [10, 10], "number": 10, "confidence": "high"}')

    path = store.put(request, response)

    text = path.read_text(encoding="utf-8")
    cassette = json.loads(text)
    assert cassette["context"] == {"frames": [10, 35]}
    assert cassette["meta"]["item"] == "track-7-0a1b2c3d"
    # No coordinate of a box is in the file, in any place.
    assert "boxes" not in text
    for coordinate in {coordinate for box in boxes for coordinate in box}:
        assert str(coordinate) not in text
    # The file holds no image bytes in any form.
    for image in (IMAGE_BYTES, OTHER_IMAGE_BYTES):
        assert base64.b64encode(image)[:40].decode("ascii") not in text
        assert image.hex()[:40] not in text
    # The key still stands for the whole context: the stored answer is found again by a request with
    # the same frames and boxes, and by no request with other frames or other boxes.
    assert store.get(request) == response
    other_frames = request.model_copy(update={"context": {**context, "frames": [10, 36]}})
    assert store.get(other_frames) is None
    moved = [[box[0] + 1.0, *box[1:]] for box in boxes]
    other_boxes = request.model_copy(update={"context": {**context, "boxes": moved}})
    assert store.get(other_boxes) is None


def test_cassette_has_no_image_bytes(tmp_path: Path) -> None:
    path = CassetteStore(tmp_path).put(make_request(), make_response())
    text = path.read_text(encoding="utf-8")

    for encoded in (base64.b64encode(IMAGE_BYTES), base64.urlsafe_b64encode(IMAGE_BYTES)):
        assert encoded.decode("ascii") not in text
        # Not even the start of the image is in the file.
        assert encoded[:40].decode("ascii") not in text
    assert IMAGE_BYTES.hex() not in text
    assert IMAGE_BYTES.hex()[:40] not in text
    # The whole cassette is smaller than the image.
    assert path.stat().st_size < len(IMAGE_BYTES)


def test_cache_key_line_and_place_of_a_cassette_fit_the_allowed_line_of_the_secret_scan(tmp_path: Path) -> None:
    """The secret scan of the git hooks takes the line of the cache key for a secret. One rule allows that line.

    The rule names the text of the line and the place of the file. So a change of the field name, of
    the layout of the JSON text or of the folders needs a change of the rule in ``.gitleaks.toml``.
    Without it, no cassette can be committed to the results branch.
    """
    (allowed,) = tomllib.loads(SECRET_SCAN_FILE.read_text(encoding="utf-8"))["allowlists"]
    (line_pattern,), (path_pattern,) = allowed["regexes"], allowed["paths"]
    request = make_request()
    root = tmp_path / "results" / "cache"
    path = CassetteStore(root).put(request, make_response())
    lines = path.read_text(encoding="utf-8").splitlines()

    key_lines = [line for line in lines if request.cache_key() in line]
    assert key_lines == [f'  "key": "{request.cache_key()}",']
    assert re.search(line_pattern, key_lines[0])
    # The rule allows this one line of the cassette and no other line.
    assert [line for line in lines if re.search(line_pattern, line)] == key_lines
    # The scan sees the path of the file inside the work tree of the results branch.
    assert re.search(path_pattern, path.relative_to(tmp_path).as_posix())
    assert re.search(path_pattern, path.relative_to(root).as_posix())


def test_cassettes_pass_the_secret_scan_and_a_planted_key_does_not(tmp_path: Path) -> None:
    """The scan runs with the rules of the repository on a folder of cassettes, as a paid run writes them."""
    if shutil.which("gitleaks") is None:
        pytest.skip("gitleaks is not installed")
    store = CassetteStore(tmp_path / "results" / "cache")
    for frame in range(12):
        store.put(make_request(frame=frame), make_response())
    store.put(make_landmarks_request(), make_response('{"landmarks": []}'))

    def scan() -> int:
        """Scan the result folder. The exit code is 0 without a finding, else 1."""
        options = ["--config", str(SECRET_SCAN_FILE), "--redact", "--no-banner", "--no-color"]
        command = ["gitleaks", "dir", *options, str(tmp_path / "results")]
        return subprocess.run(command, capture_output=True, check=False).returncode

    assert scan() == 0
    # A made-up value in the shape of a Google API key as the answer text of one more cassette is still found.
    # It is built here, so this file does not hold it.
    made_up = "AI" + "za" + "".join(random.Random(7).choices(string.ascii_letters + string.digits, k=35))
    store.put(make_request(frame=99), make_response(made_up))
    assert scan() == 1


def test_put_replaces_an_older_answer_and_leaves_no_other_file(tmp_path: Path) -> None:
    store = CassetteStore(tmp_path)
    request = make_request()
    path = store.put(request, make_response('{"people": []}'))
    newer = make_response('{"people": [1]}')

    assert store.put(request, newer) == path
    assert store.get(request) == newer
    assert [entry.name for entry in path.parent.iterdir()] == [path.name]


def test_write_that_fails_leaves_no_temporary_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = CassetteStore(tmp_path)
    request = make_request()
    older = make_response('{"people": []}')
    path = store.put(request, older)

    def cannot_replace(source: Any, target: Any) -> None:
        raise OSError("the disk is full")

    monkeypatch.setattr(cache_module.os, "replace", cannot_replace)
    with pytest.raises(OSError, match="the disk is full"):
        store.put(request, make_response('{"people": [1]}'))

    # The older answer is whole, and the temporary file of the failed write is gone.
    assert files_below(tmp_path) == [path.relative_to(tmp_path).as_posix()]
    assert store.get(request) == older


def test_damaged_cassette_is_an_error_and_not_a_miss(tmp_path: Path) -> None:
    store = CassetteStore(tmp_path)
    path = store.put(make_request(), make_response())
    key = make_request().cache_key()
    hashes = list(make_request().image_hashes)
    response = make_response().model_dump(mode="json")
    # A read needs the key, the image hashes and the answer. With these three parts there is no error.
    path.write_text(json.dumps({"key": key, "image_hashes": hashes, "response": response}), encoding="utf-8")
    assert store.get(make_request()) == make_response()
    damaged_texts = [
        "{not json",
        "[]",
        '{"key": "x"}',
        json.dumps({"key": key, "image_hashes": hashes, "response": {}}),
        # The image hashes are missing, or they are not a list of texts.
        json.dumps({"key": key, "response": response}),
        json.dumps({"key": key, "image_hashes": None, "response": response}),
        json.dumps({"key": key, "image_hashes": hashes[0], "response": response}),
        json.dumps({"key": key, "image_hashes": [1], "response": response}),
    ]

    for damaged in damaged_texts:
        path.write_text(damaged, encoding="utf-8")
        with pytest.raises(ValueError, match="cannot be read"):
            store.get(make_request())


def test_cassette_of_another_request_is_an_error(tmp_path: Path) -> None:
    store = CassetteStore(tmp_path)
    source = store.put(make_request(frame=1), make_response())
    target = store.path_for(make_request(frame=2))
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ValueError, match="another request"):
        store.get(make_request(frame=2))


# --- Image hashes of a hit ------------------------------------------------------


def test_hit_with_the_recorded_image_bytes_logs_nothing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    store = CassetteStore(tmp_path)
    response = make_response()
    store.put(make_request(), response)

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        assert store.get(make_request()) == response
        assert store.get(make_request(service_tier="flex")) == response
        # A request without a cassette is a miss. It is not compared with any recorded image.
        assert store.get(make_request(repeat=1)) is None

    assert cache_records(caplog) == []


def test_hit_with_other_image_bytes_is_an_error_that_names_the_request_and_the_two_ways_out(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A clip that was replaced under the same name must not get the answers of the old pixels."""
    store = CassetteStore(tmp_path)
    path = store.put(make_request(), make_response())
    recorded_text = path.read_text(encoding="utf-8")
    other_pixels = make_request(image=OTHER_IMAGE_BYTES)
    # The key has no pixels, so the request with other pixels finds the same cassette.
    assert store.path_for(other_pixels) == path

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME), pytest.raises(OtherImagesError) as excinfo:
        store.get(other_pixels)

    message = str(excinfo.value)
    assert "people request of clip clip-1 (frame 12, item None, repeat 0)" in message
    assert "other image bytes (changed images, counted from 1: 1)" in message
    # The two ways out: another clip name for new footage, or the option of the store for the same footage.
    # The option is named as the store has it and as the command line has it.
    assert "give the clip another name" in message
    assert "allow_other_images" in message
    assert "--allow-other-images" in message
    assert str(path) in message
    # The error is a backend error, so a command ends with one error line. Nothing is logged next to it.
    assert isinstance(excinfo.value, BackendError)
    assert not isinstance(excinfo.value, CacheMiss)
    assert cache_records(caplog) == []
    # A read never changes the cassette: it still holds the hash of the recorded image.
    assert path.read_text(encoding="utf-8") == recorded_text


def test_store_that_allows_other_image_bytes_gives_the_stored_answer_and_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    response = make_response()
    path = CassetteStore(tmp_path).put(make_request(), response)
    recorded_text = path.read_text(encoding="utf-8")
    store = CassetteStore(tmp_path, allow_other_images=True)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert store.get(make_request(image=OTHER_IMAGE_BYTES)) == response

    (record,) = cache_records(caplog)
    message = record.getMessage()
    assert record.levelno == logging.WARNING
    assert "people request of clip clip-1 (frame 12, item None, repeat 0)" in message
    assert "other image bytes (changed images, counted from 1: 1)" in message
    assert "It can describe other pixels" in message
    assert str(path) in message
    assert path.read_text(encoding="utf-8") == recorded_text


def test_store_that_allows_other_image_bytes_logs_nothing_for_the_recorded_images(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = CassetteStore(tmp_path, allow_other_images=True)
    response = make_response()
    store.put(make_request(), response)

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        assert store.get(make_request()) == response

    assert cache_records(caplog) == []


@pytest.mark.parametrize(
    ("first", "second", "changed"),
    [
        ("other", "frame", "1"),
        ("diagram", "other", "2"),
        ("other", "other", "1, 2"),
        # The recorded images in the other order are other images too.
        ("frame", "diagram", "1, 2"),
    ],
)
def test_error_and_warning_name_the_changed_images_in_request_order(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, first: str, second: str, changed: str
) -> None:
    response = make_response('{"landmarks": []}')
    # Recorded: the diagram is image 1 and the frame is image 2.
    CassetteStore(tmp_path).put(make_landmarks_request(IMAGES["diagram"], IMAGES["frame"]), response)
    request = make_landmarks_request(IMAGES[first], IMAGES[second])
    place = "landmarks request of clip clip-1 (frame 12, item None, repeat 0)"

    with pytest.raises(OtherImagesError) as excinfo:
        CassetteStore(tmp_path).get(request)
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert CassetteStore(tmp_path, allow_other_images=True).get(request) == response

    (record,) = cache_records(caplog)
    for message in (str(excinfo.value), record.getMessage()):
        assert place in message
        assert f"(changed images, counted from 1: {changed})" in message


def test_image_without_a_recorded_hash_counts_as_changed(tmp_path: Path) -> None:
    store = CassetteStore(tmp_path)
    path = store.put(make_landmarks_request(), make_response('{"landmarks": []}'))
    # A cassette that was edited by hand: the hash of the second image is gone.
    cassette = json.loads(path.read_text(encoding="utf-8"))
    cassette["image_hashes"] = cassette["image_hashes"][:1]
    path.write_text(json.dumps(cassette), encoding="utf-8")

    with pytest.raises(OtherImagesError, match=r"\(changed images, counted from 1: 2\)"):
        store.get(make_landmarks_request())


def test_recorded_hash_without_an_image_counts_as_changed(tmp_path: Path) -> None:
    store = CassetteStore(tmp_path)
    path = store.put(make_request(), make_response())
    # A cassette that was edited by hand: it names one image more than the request shows.
    cassette = json.loads(path.read_text(encoding="utf-8"))
    cassette["image_hashes"].append(hashlib.sha256(OTHER_IMAGE_BYTES).hexdigest())
    path.write_text(json.dumps(cassette), encoding="utf-8")

    with pytest.raises(OtherImagesError, match=r"\(changed images, counted from 1: 2\)"):
        store.get(make_request())


# --- CachedBackend --------------------------------------------------------------


def test_replay_miss_raises_cache_miss(tmp_path: Path) -> None:
    backend = CachedBackend(CassetteStore(tmp_path))

    with pytest.raises(CacheMiss):
        backend.generate(make_request())
    assert issubclass(CacheMiss, BackendError)
    # A miss in replay mode writes nothing.
    assert list(tmp_path.iterdir()) == []


def test_miss_calls_inner_stores_the_answer_and_reports_it(tmp_path: Path) -> None:
    store, inner = CassetteStore(tmp_path), CountingBackend()
    new_answers: list[tuple[VLMRequest, VLMResponse]] = []
    backend = CachedBackend(
        store, inner, on_new_answer=lambda request, response: new_answers.append((request, response))
    )
    request = make_request()

    first = backend.generate(request)

    assert answered_frame(first) == 12
    assert inner.calls == [request.cache_key()]
    assert new_answers == [(request, first)]
    assert store.get(request) == first


def test_hit_returns_the_stored_answer_without_a_new_call(tmp_path: Path) -> None:
    store, inner = CassetteStore(tmp_path), CountingBackend()
    new_answers: list[VLMResponse] = []
    backend = CachedBackend(store, inner, on_new_answer=lambda request, response: new_answers.append(response))

    first = backend.generate(make_request())
    second = backend.generate(make_request())
    other_tier = backend.generate(make_request(service_tier="flex"))

    assert second == first
    assert other_tier == first
    assert len(inner.calls) == 1
    assert len(new_answers) == 1


def test_other_pixels_stop_a_paid_run_and_a_replay_without_a_model_call(tmp_path: Path) -> None:
    """A clip that was replaced under the same name: no stored answer is used, and no new answer is paid for."""
    store, inner = CassetteStore(tmp_path), CountingBackend()
    new_answers: list[VLMResponse] = []
    backend = CachedBackend(store, inner, on_new_answer=lambda request, response: new_answers.append(response))
    recorded = backend.generate(make_request())
    cassettes = files_below(tmp_path)
    other_pixels = make_request(image=OTHER_IMAGE_BYTES)

    with pytest.raises(OtherImagesError, match="clip clip-1"):
        backend.generate(other_pixels)
    with pytest.raises(OtherImagesError, match="clip clip-1"):
        CachedBackend(store).generate(other_pixels)

    # The model was asked one time, for the recorded pixels. The stored answer is as it was.
    assert len(inner.calls) == 1
    assert len(new_answers) == 1
    assert files_below(tmp_path) == cassettes
    assert backend.generate(make_request()) == recorded


def test_other_pixels_are_a_hit_with_a_warning_when_the_store_allows_them(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    inner = CountingBackend()
    new_answers: list[VLMResponse] = []
    recorded = CachedBackend(CassetteStore(tmp_path), inner).generate(make_request())
    store = CassetteStore(tmp_path, allow_other_images=True)
    backend = CachedBackend(store, inner, on_new_answer=lambda request, response: new_answers.append(response))
    other_pixels = make_request(image=OTHER_IMAGE_BYTES)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        again = backend.generate(other_pixels)
        replayed = CachedBackend(store).generate(other_pixels)

    # The stored answer is used: the model is not asked, and not paid, a second time.
    assert again == recorded
    assert replayed == recorded
    assert len(inner.calls) == 1
    assert new_answers == []
    # Each of the two hits logged one warning.
    assert [record.levelno for record in cache_records(caplog)] == [logging.WARNING, logging.WARNING]
    assert all("other image bytes" in record.getMessage() for record in cache_records(caplog))


def test_a_changed_question_is_a_miss(tmp_path: Path) -> None:
    inner = CountingBackend()
    backend = CachedBackend(CassetteStore(tmp_path), inner)

    backend.generate(make_request())
    backend.generate(make_request(repeat=1))
    backend.generate(make_request(text="Find every player."))
    backend.generate(make_request(model="gemini-4-carbon"))

    assert len(inner.calls) == 4
    assert len(set(inner.calls)) == 4


def test_replay_gives_the_recorded_answers(tmp_path: Path) -> None:
    recorded = CachedBackend(CassetteStore(tmp_path), CountingBackend()).generate(make_request())
    replay = CachedBackend(CassetteStore(tmp_path))

    assert replay.generate(make_request()) == recorded
    assert replay.generate(make_request(service_tier="flex")) == recorded
    with pytest.raises(CacheMiss):
        replay.generate(make_request(repeat=1))


def test_works_without_a_callback(tmp_path: Path) -> None:
    backend = CachedBackend(CassetteStore(tmp_path), CountingBackend())

    assert answered_frame(backend.generate(make_request(frame=3))) == 3


def test_answer_is_stored_before_the_callback_runs(tmp_path: Path) -> None:
    store = CassetteStore(tmp_path)

    def stop(request: VLMRequest, response: VLMResponse) -> None:
        raise RuntimeError("budget")

    backend = CachedBackend(store, CountingBackend(), on_new_answer=stop)

    with pytest.raises(RuntimeError, match="budget"):
        backend.generate(make_request())
    # The paid answer is kept, so the next call is a hit and the callback does not run again.
    assert answered_frame(backend.generate(make_request())) == 12
    # The lock is free again after the error: another new answer gets through to the callback.
    with pytest.raises(RuntimeError, match="budget"):
        backend.generate(make_request(frame=13))


# --- CachedBackend: a store that cannot be written ---------------------------------


def test_check_writable_passes_for_a_folder_that_exists_or_can_be_made(tmp_path: Path) -> None:
    store = CassetteStore(tmp_path / "results" / "cache")
    request = make_request()

    store.check_writable(request)
    # The check creates nothing: not the root folder, and no folder of the clip or of the step.
    assert list(tmp_path.iterdir()) == []

    store.put(request, make_response())
    store.check_writable(request)
    store.check_writable(make_request(frame=13))


def test_check_writable_refuses_a_folder_without_the_write_permission(read_only_folder: Path) -> None:
    request = make_request()

    # The root folder is missing below a folder that cannot be written, so it cannot be made.
    with pytest.raises(PermissionError, match="the folder of the stored answers cannot be written") as excinfo:
        CassetteStore(read_only_folder / "cache").check_writable(request)
    assert excinfo.value.filename == str(read_only_folder)
    # The root folder itself cannot be written.
    with pytest.raises(PermissionError):
        CassetteStore(read_only_folder).check_writable(request)
    assert list(read_only_folder.iterdir()) == []


def test_check_writable_refuses_a_file_in_the_place_of_a_folder(tmp_path: Path) -> None:
    (tmp_path / "clip-1").write_text("not a folder", encoding="utf-8")

    with pytest.raises(OSError, match="the folder of the stored answers cannot be written"):
        CassetteStore(tmp_path).check_writable(make_request())


def test_model_is_not_asked_when_the_answer_could_not_be_stored(read_only_folder: Path) -> None:
    """An answer that cannot be stored would be paid for again by every new start."""
    inner = CountingBackend()
    reported: list[VLMResponse] = []
    backend = CachedBackend(
        CassetteStore(read_only_folder), inner, on_new_answer=lambda request, response: reported.append(response)
    )

    with pytest.raises(PermissionError, match="the folder of the stored answers cannot be written"):
        backend.generate(make_request())

    assert inner.calls == []
    assert reported == []


def test_paid_answer_is_reported_also_when_its_cassette_cannot_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full disk: the check before the call passes, and the write after it fails. The cost must reach the ledger."""
    store, inner = CassetteStore(tmp_path), CountingBackend()
    reported: list[tuple[VLMRequest, VLMResponse]] = []
    backend = CachedBackend(
        store, inner, on_new_answer=lambda request, response: reported.append((request, response))
    )
    request = make_request()

    def cannot_replace(source: Any, target: Any) -> None:
        raise OSError("the disk is full")

    monkeypatch.setattr(cache_module.os, "replace", cannot_replace)
    with pytest.raises(OSError, match="the disk is full"):
        backend.generate(request)

    # The model was asked and paid one time, and the answer went to the callback one time.
    assert inner.calls == [request.cache_key()]
    assert [(asked, answered_frame(response)) for asked, response in reported] == [(request, 12)]
    assert files_below(tmp_path) == []
    # The lock is free again: with a disk that works, the next request is stored and reported.
    monkeypatch.undo()
    assert answered_frame(backend.generate(make_request(frame=13))) == 13
    assert len(reported) == 2
    assert len(files_below(tmp_path)) == 1


def test_error_of_the_callback_replaces_the_error_of_a_failed_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stop(request: VLMRequest, response: VLMResponse) -> None:
        raise RuntimeError("budget")

    def cannot_replace(source: Any, target: Any) -> None:
        raise OSError("the disk is full")

    monkeypatch.setattr(cache_module.os, "replace", cannot_replace)
    backend = CachedBackend(CassetteStore(tmp_path), CountingBackend(), on_new_answer=stop)

    # A budget stop must reach the caller. The error of the store is kept as its context.
    with pytest.raises(RuntimeError, match="budget") as excinfo:
        backend.generate(make_request())
    assert isinstance(excinfo.value.__context__, OSError)


# --- CachedBackend: answers that are billed and then rejected ----------------------


def test_rejected_answer_reports_its_cost_one_time_and_stores_nothing(tmp_path: Path) -> None:
    store, inner = CassetteStore(tmp_path), RejectingBackend()
    reported: list[tuple[VLMRequest, VLMResponse]] = []
    backend = CachedBackend(
        store, inner, on_new_answer=lambda request, response: reported.append((request, response))
    )
    request = make_request()

    with pytest.raises(BilledAnswerError, match="call 1") as excinfo:
        backend.generate(request)

    # The cost record of the error goes to the callback one time, like a new answer.
    assert reported == [(request, inner.billed[0])]
    assert excinfo.value.billed is inner.billed[0]
    # There is no answer to store: no cassette is written, and no folder for one.
    assert store.get(request) is None
    assert list(tmp_path.iterdir()) == []
    # So the next call is a miss again. The model is asked again, and that call is billed and reported too.
    with pytest.raises(BilledAnswerError, match="call 2"):
        backend.generate(request)
    assert [response for _, response in reported] == inner.billed
    assert len(inner.billed) == 2


def test_rejected_answer_is_raised_again_without_a_callback(tmp_path: Path) -> None:
    backend = CachedBackend(CassetteStore(tmp_path), RejectingBackend())

    with pytest.raises(BilledAnswerError, match="call 1"):
        backend.generate(make_request())
    assert list(tmp_path.iterdir()) == []


def test_failed_call_without_a_cost_record_is_not_reported(tmp_path: Path) -> None:
    reported: list[VLMResponse] = []
    backend = CachedBackend(
        CassetteStore(tmp_path), FailingBackend(), on_new_answer=lambda request, response: reported.append(response)
    )

    with pytest.raises(BackendError, match="the call failed"):
        backend.generate(make_request())
    # A call that the model did not answer costs nothing, so there is nothing to report or to store.
    assert reported == []
    assert list(tmp_path.iterdir()) == []


def test_error_of_the_callback_replaces_the_error_of_a_rejected_answer(tmp_path: Path) -> None:
    def stop(request: VLMRequest, response: VLMResponse) -> None:
        raise RuntimeError("budget")

    backend = CachedBackend(CassetteStore(tmp_path), RejectingBackend(), on_new_answer=stop)

    # A budget stop must reach the caller, also when the answer that reached the limit was rejected.
    with pytest.raises(RuntimeError, match="budget") as excinfo:
        backend.generate(make_request())
    assert isinstance(excinfo.value.__context__, BilledAnswerError)
    # The lock is free again after the error: the next rejected answer gets through to the callback.
    with pytest.raises(RuntimeError, match="budget"):
        backend.generate(make_request(frame=13))


def test_request_that_another_command_paid_for_at_the_same_moment_logs_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Two commands on one clip have no lock between them. The second answer says that the request was paid twice."""
    store = CassetteStore(tmp_path)
    other_command = CachedBackend(CassetteStore(tmp_path), CountingBackend())
    request = make_request()

    class SlowBackend(CountingBackend):
        """A backend during whose call for frame 12 another command asks for the same request and stores its answer."""

        def generate(self, request: VLMRequest) -> VLMResponse:
            if request.meta.frame == 12:
                other_command.generate(request)
            return super().generate(request)

    inner = SlowBackend()
    reported: list[VLMRequest] = []
    backend = CachedBackend(store, inner, on_new_answer=lambda request, response: reported.append(request))

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        backend.generate(request)
        # A request that only this command pays for gives no warning.
        backend.generate(make_request(frame=24))

    assert len(inner.calls) == 2
    assert [record.levelno for record in cache_records(caplog)] == [logging.WARNING]
    message = cache_records(caplog)[0].getMessage()
    assert "people request of clip clip-1 (frame 12, item None, repeat 0" in message
    assert request.cache_key() in message
    assert "paid for the same request" in message
    # Both answers were paid, so the answer of this command is reported like every new answer.
    assert reported == [request, make_request(frame=24)]


def test_cached_backend_is_safe_from_several_threads(tmp_path: Path) -> None:
    store, inner = CassetteStore(tmp_path), CountingBackend()
    new_answers: list[int] = []
    backend = CachedBackend(
        store, inner, on_new_answer=lambda request, response: new_answers.append(answered_frame(response))
    )
    # Eight workers ask for the same 20 frames, so many calls miss on the same key at the same time.
    frames = [frame for _ in range(8) for frame in range(20)]

    def ask(frame: int) -> int:
        return answered_frame(backend.generate(make_request(frame=frame)))

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(ask, frames))

    assert answers == frames
    # Two threads may both call the inner backend for one key, but never more calls than requests.
    assert 20 <= len(inner.calls) <= len(frames)
    assert len(new_answers) == len(inner.calls)
    assert set(new_answers) == set(range(20))
    # One whole cassette per frame, and no temporary file is left.
    files = sorted((tmp_path / "clip-1" / "people").iterdir())
    assert len(files) == 20
    assert all(json.loads(path.read_text(encoding="utf-8"))["key"] == path.stem for path in files)
    replay = CachedBackend(store)
    assert [answered_frame(replay.generate(make_request(frame=frame))) for frame in range(20)] == list(range(20))
