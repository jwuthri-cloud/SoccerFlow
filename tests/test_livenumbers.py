"""Tests of the live number step of the streaming path (``livenumbers``, ``stream`` with the number engine model).

The model is a fake: the fake backend of the synthetic scene, or a stub whose answers the test
sets. No test depends on the speed of the machine: a request that must stay open waits on an
event that the test sets, the order of the worker is the order of its queue, and the queue of
images is large enough that no image is dropped (``QUEUE``; the drop itself has its own test).
Nothing here says how a real model reads a real shirt.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pitchvlm import cli, livenumbers, radar, stopsignals
from pitchvlm.config import EngineSettings, PipelineConfig
from pitchvlm.fallbacks.fake import fake_calibration_answers, fake_detector_answers
from pitchvlm.livenumbers import LiveNumbers, NumberLine, NumberSettings, apply_numbers, wire
from pitchvlm.liveout import LIVE_FORMAT, LIVE_FORMAT_NUMBERS, LiveOut, open_live_out
from pitchvlm.schemas import Box, ClipInfo, FrameState, PersonDet, PlayerState
from pitchvlm.stream import DetectorStream, StreamSettings, default_history_frames, stand_in_inputs
from pitchvlm.synth import SyntheticScene
from pitchvlm.vlm.cost import BudgetExceeded
from pitchvlm.vlm.fake import FakeBackend
from pitchvlm.vlm.types import Usage, VLMRequest, VLMResponse

REPO = Path(__file__).resolve().parents[1]
NUMBERS = PipelineConfig(engines=EngineSettings(people="detector", pitch_mapping="calibration", number="model"))
NONE = PipelineConfig(engines=EngineSettings(people="detector", pitch_mapping="calibration", number="none"))
NUMBERS_FILE = REPO / "configs" / "engines" / "detector_calibration_numbers.yaml"
INFO = ClipInfo(clip_id="c", n_frames=400, fps=25.0, width=640, height=360)
IMAGE = np.zeros((360, 640, 3), dtype=np.uint8)
#: A queue that never fills in the tests that hand frames faster than the worker takes them (a real frame path is slower).
QUEUE = 10_000
LEGEND_ANSWER = {
    "team_a": {"shirt": "red", "shorts": "white", "socks": "red"},
    "team_b": {"shirt": "blue", "shorts": "blue", "socks": "white"},
    "goalkeepers": [],
    "referee_kit": "black",
    "uncertain": [],
}


def response(request: VLMRequest, answer: dict[str, Any]) -> VLMResponse:
    return VLMResponse(
        text=json.dumps(answer),
        usage=Usage(input_tokens=10, output_tokens=1),
        model=request.model,
        service_tier=request.service_tier,
        latency_s=0.0,
        created_at="2026-11-23T00:00:00+00:00",
    )


class StubBackend:
    """Answer the kit legend, and each number request with the number of the box: ``x1 // 10 + 1``.

    ``gate`` (when given) holds every number request until it is set. ``fail_on`` raises the
    error at that number request (1 for the first). It records every request.
    """

    def __init__(self, gate: threading.Event | None = None, fail_on: tuple[int, Exception] | None = None) -> None:
        self.gate = gate
        self.fail_on = fail_on
        self.requests: list[VLMRequest] = []
        self.numbers_started = threading.Semaphore(0)
        self._lock = threading.Lock()

    def generate(self, request: VLMRequest) -> VLMResponse:
        with self._lock:
            self.requests.append(request)
            count = sum(1 for item in self.requests if item.meta.step == "number")
        if request.meta.step == "kit_legend":
            return response(request, LEGEND_ANSWER)
        self.numbers_started.release()
        if self.fail_on is not None and count == self.fail_on[0]:
            raise self.fail_on[1]
        if self.gate is not None and not self.gate.wait(60):
            raise TimeoutError("the test did not open the gate")
        boxes = request.context["boxes"]
        per_crop = [int(box[0]) // 10 + 1 for box in boxes]
        return response(request, {"per_crop": per_crop, "number": per_crop[0], "confidence": "high"})

    def number_requests(self) -> list[VLMRequest]:
        with self._lock:
            return [item for item in self.requests if item.meta.step == "number"]


def box_of(track_id: int) -> Box:
    """The box of a stub id: its x1 names its number (``StubBackend``)."""
    return Box(x1=10.0 * track_id, y1=100.0, x2=10.0 * track_id + 8.0, y2=180.0)


def player(track_id: int, role: str = "player", observed: bool = True) -> PlayerState:
    return PlayerState(track_id=track_id, box=box_of(track_id), pitch_x=0.0, pitch_y=0.0, role=role, observed=observed)


def drive(numbers: LiveNumbers, seen: dict[int, range], roles: dict[int, str] | None = None, n_frames: int = 0) -> None:
    """Hand frames 0 to n_frames - 1 to the worker: each id of ``seen`` is in the frames of its range."""
    roles = roles or {}
    last = max([n_frames, *(span.stop for span in seen.values())])
    for frame in range(last):
        here = [track_id for track_id, span in seen.items() if frame in span]
        people = [PersonDet(box=box_of(track_id), role="player") for track_id in here]
        numbers.arrived(frame, IMAGE, people)
        numbers.given(FrameState(frame=frame, players=tuple(player(track_id, roles.get(track_id, "player")) for track_id in here)))
    assert numbers.images_dropped == 0


def by_id(lines: list[NumberLine]) -> dict[int, NumberLine]:
    return {line.track_id: line for line in lines}


# --- The rules of the worker -----------------------------------------------------------------------


def test_an_id_is_asked_once_when_it_is_ready_closed_or_at_the_end_and_a_referee_never() -> None:
    backend = StubBackend()
    numbers = LiveNumbers(NUMBERS, backend, NumberSettings(close_frames=50, max_pending_images=QUEUE), info=INFO)
    # id 0: frames 0-119 (ready at frame 108: crops 0, 36, 72, 108 are 1 s apart); id 1: frames 0-29, closed
    # 50 frames later; id 2: a referee; id 3: frames 100-119, open at the end.
    drive(numbers, {0: range(0, 120), 1: range(0, 30), 2: range(0, 120), 3: range(100, 120)}, roles={2: "referee"})
    lines = by_id(numbers.finish())
    assert sorted(lines) == [0, 1, 3]
    assert (lines[0].reason, lines[0].crop_frames, lines[0].export, lines[0].display) == ("ready", (0, 36, 72, 108), 1, 1)
    assert (lines[1].reason, lines[1].crop_frames, lines[1].export) == ("closed", (0, 24), 2)
    assert (lines[3].reason, lines[3].crop_frames, lines[3].export) == ("end", (108,), 4)
    assert (lines[0].first_frame, lines[0].last_frame, lines[0].first_seen_frame) == (0, 108, 0)
    assert len(backend.number_requests()) == 3
    assert [request.meta.step for request in backend.requests].count("kit_legend") == 1
    report = numbers.report()
    assert (report["requests_planned"], report["answers"], report["refused_by_cap"], report["failure"]) == (3, 3, 0, None)


def test_the_request_is_the_number_request_of_the_clip_path_with_the_kit_legend() -> None:
    backend = StubBackend()
    numbers = LiveNumbers(NUMBERS, backend, NumberSettings(max_pending_images=QUEUE), info=INFO)
    drive(numbers, {5: range(0, 120)})
    numbers.finish()
    (request,) = backend.number_requests()
    assert request.context["frames"] == [0, 36, 72, 108]
    assert request.context["boxes"] == [[50.0, 100.0, 58.0, 180.0]] * 4
    assert request.meta.clip_id == "c" and request.meta.item.startswith("crops-")
    assert request.model == NUMBERS.number.model and request.service_tier == NUMBERS.number.service_tier
    assert "red" in request.parts[0].text  # the kit legend of the answer is in the prompt
    assert len(request.parts) == 1 + 4  # the text and four crops


def test_the_cap_refuses_the_ids_after_it_and_says_so_once(caplog: pytest.LogCaptureFixture) -> None:
    backend = StubBackend()
    numbers = LiveNumbers(NUMBERS, backend, NumberSettings(max_requests=2, max_pending_images=QUEUE), info=INFO)
    with caplog.at_level(logging.WARNING, logger="pitchvlm.livenumbers"):
        drive(numbers, {0: range(0, 120), 1: range(0, 120), 2: range(0, 120), 3: range(0, 120)})
        lines = numbers.finish()
    assert len(lines) == 2 and len(backend.number_requests()) == 2
    assert numbers.refused_by_cap == 2 and numbers.cap_reached
    assert sum("cap of 2 number requests" in record.getMessage() for record in caplog.records) == 1


def test_a_cap_of_zero_sends_nothing_not_even_the_kit_legend() -> None:
    backend = StubBackend()
    numbers = LiveNumbers(NUMBERS, backend, NumberSettings(max_requests=0, max_pending_images=QUEUE), info=INFO)
    drive(numbers, {0: range(0, 120)})
    assert numbers.finish() == [] and backend.requests == [] and numbers.refused_by_cap == 1


def test_after_the_first_failure_no_request_is_sent_and_the_failure_is_reported() -> None:
    backend = StubBackend(fail_on=(1, BudgetExceeded("the run spent more than 1.3 times its estimate")))
    numbers = LiveNumbers(NUMBERS.model_copy(update={"max_parallel": 1}), backend, NumberSettings(max_pending_images=QUEUE), info=INFO)
    drive(numbers, {0: range(0, 120), 1: range(0, 120), 2: range(0, 120)})
    lines = numbers.finish()
    assert lines == [] and len(backend.number_requests()) == 1
    assert isinstance(numbers.failure, BudgetExceeded)
    assert numbers.report()["failure"].startswith("BudgetExceeded")


def test_an_answer_that_cannot_be_parsed_stops_the_asking() -> None:
    class Broken(StubBackend):
        def generate(self, request: VLMRequest) -> VLMResponse:
            if request.meta.step == "number":
                with self._lock:
                    self.requests.append(request)
                return response(request, {"per_crop": [1]})  # no confidence
            return super().generate(request)

    backend = Broken()
    numbers = LiveNumbers(NUMBERS.model_copy(update={"max_parallel": 1}), backend, NumberSettings(max_pending_images=QUEUE), info=INFO)
    drive(numbers, {0: range(0, 120)})
    assert numbers.finish() == []
    assert numbers.failure is not None and "confidence" in str(numbers.failure)


def test_a_full_queue_drops_new_images_and_counts_them(monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()
    cutting = threading.Event()
    real_cut = livenumbers._cut_crop

    def held_cut(image: np.ndarray, box: Box) -> np.ndarray:
        cutting.set()
        assert release.wait(60)
        return real_cut(image, box)

    monkeypatch.setattr(livenumbers, "_cut_crop", held_cut)
    numbers = LiveNumbers(NUMBERS, StubBackend(), NumberSettings(stride_frames=1, max_pending_images=2), info=INFO)
    people = [PersonDet(box=box_of(0), role="player")]
    numbers.arrived(0, IMAGE, people)
    assert cutting.wait(60)  # the worker holds frame 0; two more images may wait
    for frame in range(1, 6):
        numbers.arrived(frame, IMAGE, people)
    assert numbers.images_dropped == 3
    release.set()
    numbers.finish()
    assert numbers.report()["images_dropped"] == 3


def test_a_frame_without_image_and_a_decision_without_crop_need_no_rule() -> None:
    backend = StubBackend()
    numbers = LiveNumbers(NUMBERS, backend, NumberSettings(max_pending_images=QUEUE), info=INFO)
    # Frames 12 and 24 never arrive (a frame path that dropped them); frame 36 is decided without its image.
    for frame in range(0, 150):
        if frame not in (12, 24, 36):
            numbers.arrived(frame, IMAGE, [PersonDet(box=box_of(0), role="player")])
        if frame not in (12, 24):
            numbers.given(FrameState(frame=frame, players=(player(0),)))
    (line,) = numbers.finish()
    assert 12 not in line.crop_frames and 36 not in line.crop_frames and line.reason == "ready"


def test_the_frame_path_never_waits_for_an_answer_and_finish_waits_for_every_request() -> None:
    scene = SyntheticScene(n_frames=150, seed=2291)
    gate = threading.Event()
    backend = StubBackend(gate=gate)
    stream, numbers, live = numbers_stream(scene, backend, None)
    done = threading.Event()
    pushed: list[int] = []

    def push_all() -> None:
        for item in inputs_of(scene):
            stream.push(item, scene.image(item.frame))
            pushed.append(item.frame)
        stream.finish()
        done.set()

    runner = threading.Thread(target=push_all, daemon=True)
    runner.start()
    assert done.wait(120), "the frame path waited for a model answer"
    assert pushed == list(range(150))
    assert backend.numbers_started.acquire(timeout=60)  # a number request is in flight, held by the gate
    assert numbers.lines == []
    finishing = threading.Thread(target=lambda: finished.extend(numbers.finish()), daemon=True)
    finished: list[NumberLine] = []
    finishing.start()
    finishing.join(0.3)
    assert finishing.is_alive(), "finish returned while a paid request was in flight"
    gate.set()
    finishing.join(120)
    assert not finishing.is_alive()
    assert len(finished) == len(backend.number_requests()) > 0


def test_abort_sends_nothing_more_and_waits_for_the_requests_in_flight() -> None:
    gate = threading.Event()
    backend = StubBackend(gate=gate)
    numbers = LiveNumbers(NUMBERS.model_copy(update={"max_parallel": 1}), backend, NumberSettings(max_pending_images=QUEUE), info=INFO)
    drive(numbers, {0: range(0, 120), 1: range(0, 120), 2: range(0, 120)})
    assert backend.numbers_started.acquire(timeout=60)
    aborting = threading.Thread(target=numbers.abort, daemon=True)
    aborting.start()
    deadline = time.monotonic() + 60
    while not numbers._in_flight._closed and time.monotonic() < deadline:  # the abort refuses new requests now
        time.sleep(0.01)
    assert numbers._in_flight._closed and aborting.is_alive(), "abort returned while a paid request was in flight"
    gate.set()
    aborting.join(60)
    assert not aborting.is_alive()
    assert len(backend.number_requests()) == 1  # the waiting requests were not sent
    numbers.abort()  # a second stop does nothing


def test_the_settings_refuse_values_without_meaning() -> None:
    for wrong in ({"max_requests": -1}, {"ready_crops": 0}, {"max_pending_images": 0}, {"stride_frames": 0}):
        with pytest.raises(ValueError):
            NumberSettings(**wrong)


# --- The stream, the live output and the result ----------------------------------------------------


def inputs_of(scene: SyntheticScene) -> Any:
    people, balls = fake_detector_answers(scene)
    return stand_in_inputs(people, balls, fake_calibration_answers(scene))


def numbers_stream(
    scene: SyntheticScene, backend: Any, live: LiveOut | None, delay: int = 25
) -> tuple[DetectorStream, LiveNumbers, LiveOut | None]:
    info = scene.info
    history = default_history_frames(NUMBERS, info.fps)
    settings = StreamSettings(delay_frames=delay, history_frames=history)
    stream = DetectorStream(NUMBERS, settings, clip_id=info.clip_id, fps=info.fps, size=(info.width, info.height))
    numbers = LiveNumbers(NUMBERS, backend, NumberSettings(close_frames=history, max_pending_images=QUEUE), info=info)
    if live is not None:
        live.start(delay_frames=delay, history_frames=history, numbers=True)
    writer = wire(stream, numbers, live)
    stream._test_writer = writer  # type: ignore[attr-defined]
    return stream, numbers, live


def run_numbers(scene: SyntheticScene, path: Path) -> tuple[ClipResultLike, list[NumberLine], list[dict[str, Any]]]:
    live = open_live_out(str(path), scene.info)
    stream, numbers, _ = numbers_stream(scene, FakeBackend(scene), live)
    for item in inputs_of(scene):
        stream.push(item, scene.image(item.frame))
    stream.finish()
    stream._test_writer.write(numbers.finish(), stream.n_frames - 1)  # type: ignore[attr-defined]
    live.finish()
    live.close()
    result = apply_numbers(stream.result(), numbers.lines)
    return result, numbers.lines, [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


ClipResultLike = Any


def test_the_numbers_of_the_result_equal_the_number_lines_and_the_frame_lines_stay_as_given(tmp_path: Path) -> None:
    scene = SyntheticScene(n_frames=200, seed=2292)
    result, lines, feed = run_numbers(scene, tmp_path / "live.jsonl")
    header, body, end = feed[0], feed[1:-1], feed[-1]
    assert header["format"] == LIVE_FORMAT_NUMBERS and end["end"] is True
    number_lines = [line for line in body if line.get("kind") == "number"]
    frame_lines = [line for line in body if "kind" not in line]
    assert len(number_lines) == len(lines) > 5
    told = {line["id"]: (line["export"], line["display"]) for line in number_lines}
    assert len(told) == len(number_lines)  # one line per id
    for state in result.frames:
        for item in state.players:
            assert (item.number_export, item.number_display) == told.get(item.track_id, (None, None))
    for track in result.tracks:
        assert (track.number_export, track.number_display) == told.get(track.track_id, (None, None))
    # A number line comes after the frame line of the last frame that it was read from, and changes no frame line.
    position = {line["frame"]: index for index, line in enumerate(body) if "kind" not in line}
    for index, line in enumerate(body):
        if line.get("kind") == "number":
            assert index > position[line["frames"][1]] and line["decided_at"] >= line["frames"][1]
    assert all(player["number"] is None for line in frame_lines for player in line["players"])
    # The numbers are the true numbers of the scene (the fake model reads the truth).
    truth = {}
    for frame in range(scene.info.n_frames):
        for person in scene.frame_state(frame).players:
            truth.setdefault(person.track_id, person.number_export)
    assert {line["export"] for line in number_lines} <= {number for number in truth.values() if number is not None}


def test_the_frame_lines_with_numbers_equal_those_without(tmp_path: Path) -> None:
    scene = SyntheticScene(n_frames=120, seed=2293)
    _, _, with_numbers = run_numbers(scene, tmp_path / "numbers.jsonl")
    path = tmp_path / "none.jsonl"
    live = open_live_out(str(path), scene.info)
    info = scene.info
    history = default_history_frames(NONE, info.fps)
    stream = DetectorStream(
        NONE, StreamSettings(delay_frames=25, history_frames=history), clip_id=info.clip_id, fps=info.fps,
        size=(info.width, info.height),
    )
    live.start(delay_frames=25, history_frames=history)
    assert wire(stream, None, live) is None and stream.on_arrived is None
    for item in inputs_of(scene):
        stream.push(item, scene.image(item.frame))
    stream.finish()
    live.finish()
    live.close()
    without = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert without[0]["format"] == LIVE_FORMAT and all("kind" not in line for line in without)

    def frames(feed: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in line.items() if k != "decided_unix"} for line in feed[1:-1] if "kind" not in line]

    assert frames(with_numbers) == frames(without)
    assert {k: v for k, v in with_numbers[0].items() if k != "format"} == {
        k: v for k, v in without[0].items() if k != "format"
    }


def test_apply_numbers_refuses_two_lines_of_one_id() -> None:
    line = NumberLine(1, 7, 7, 0, 0, (0,), (7,), "high", "end", 0)
    with pytest.raises(ValueError, match="two number lines"):
        apply_numbers(None, [line, line])  # type: ignore[arg-type]


def test_a_number_line_is_never_left_out_by_the_pipe_policy() -> None:
    read_fd, write_fd = os.pipe()
    live = LiveOut(write_fd, INFO, target="pipe", pipe=True, close_fd=True)
    live.start(delay_frames=0, history_frames=0, numbers=True)
    scene = SyntheticScene(n_frames=2, seed=2294)
    state = scene.frame_state(0)
    from pitchvlm.schemas import HomographyRecord

    for frame in range(400):  # the reader reads nothing yet: the pipe fills, frame lines are left out
        live.given(state.model_copy(update={"frame": frame}), camera=HomographyRecord(frame=frame), last=frame,
                   transition=False, cut=False)
        if frame % 80 == 0:
            live.number(track_id=frame, export=7, display=None, first_frame=0, last_frame=frame, crop_frames=(0,),
                        reason="ready", last=frame)
    live.finish()
    chunks: list[bytes] = []
    reader = threading.Thread(target=lambda: chunks.extend(iter(lambda: os.read(read_fd, 65536), b"")), daemon=True)
    reader.start()
    report = live.close(wait_s=60)
    reader.join(60)
    os.close(read_fd)
    lines = [json.loads(line) for line in b"".join(chunks).decode().splitlines()]
    assert [line["id"] for line in lines if line.get("kind") == "number"] == [0, 80, 160, 240, 320]
    assert lines[-1]["end"] is True and report.undelivered == 0
    assert (live.numbers_given, live.numbers_written, live.numbers_undelivered) == (5, 5, 0)
    assert report.written + report.dropped == report.given  # every frame line went out or was left out


def test_number_lines_have_their_own_counts_when_the_reader_goes_away() -> None:
    """The counts of the report stay those of frame lines and the end line (``written + dropped + undelivered ==
    given + 1``), and the number lines have their own (``numbers_written + numbers_undelivered == numbers_given``),
    whether the writer thread ran before the reader went away or not."""
    from pitchvlm.schemas import HomographyRecord

    read_fd, write_fd = os.pipe()
    live = LiveOut(write_fd, INFO, target="pipe", pipe=True, close_fd=True)
    live.start(delay_frames=0, history_frames=0, numbers=True)
    os.close(read_fd)  # the reader goes away at once
    state = SyntheticScene(n_frames=2, seed=2294).frame_state(0)
    for frame in range(30):
        live.given(state.model_copy(update={"frame": frame}), camera=HomographyRecord(frame=frame), last=frame,
                   transition=False, cut=False)
        live.number(track_id=frame, export=7, display=None, first_frame=0, last_frame=frame, crop_frames=(0,),
                    reason="ready", last=frame)
    live.finish()
    report = live.close(wait_s=60)
    assert report.reader_gone is not None
    assert report.written + report.dropped + report.undelivered == report.given + 1 == 31
    assert live.numbers_written + live.numbers_undelivered == live.numbers_given == 30


def test_the_live_output_takes_number_lines_only_when_started_with_them(tmp_path: Path) -> None:
    live = open_live_out(str(tmp_path / "a.jsonl"), INFO)
    live.start(delay_frames=0, history_frames=0)
    with pytest.raises(ValueError, match="without number lines"):
        live.number(track_id=1, export=1, display=1, first_frame=0, last_frame=0, crop_frames=(0,), reason="end", last=0)
    live.finish()
    live.close()
    assert json.loads((tmp_path / "a.jsonl").read_text().splitlines()[0])["format"] == LIVE_FORMAT


# --- The radar ---------------------------------------------------------------------------------------


def live_header(fmt: str) -> dict[str, Any]:
    return {"format": fmt, "clip": "c", "n_frames": 10, "fps": 25.0, "width": 640, "height": 360,
            "delay_frames": 0, "history_frames": 0, "pitch_m": [105.0, 68.0], "policy": "file"}


def number_line(track_id: int, number: int | None) -> dict[str, Any]:
    return {"kind": "number", "id": track_id, "number": number, "export": number, "display": number,
            "frames": [0, 5], "crops": [0, 5], "reason": "ready", "decided_at": 6, "decided_unix": 1.0}


def frame_line(frame: int) -> dict[str, Any]:
    return {"frame": frame, "t_s": frame / 25.0, "players": [], "ball": None, "dropped": 0}


def test_the_radar_reads_number_lines_of_the_second_format_only() -> None:
    lines = [json.dumps(live_header(LIVE_FORMAT_NUMBERS)), json.dumps(frame_line(0)), json.dumps(number_line(3, 10))]
    documents = list(radar.feed_documents(lines))
    assert documents[-1]["kind"] == "number"
    with pytest.raises(radar.RadarError, match="not a number line"):
        list(radar.feed_documents([json.dumps(live_header(LIVE_FORMAT)), json.dumps(number_line(3, 10))]))
    for wrong in (number_line(-1, 10), number_line(3, 100), {**number_line(3, 1), "frames": [5, 0]}):
        with pytest.raises(radar.RadarError, match="not a number line"):
            list(radar.feed_documents([json.dumps(live_header(LIVE_FORMAT_NUMBERS)), json.dumps(wrong)]))


def test_the_radar_gives_every_known_number_to_a_new_page_and_each_new_one_to_every_page() -> None:
    board = radar.Board()
    board.header(live_header(LIVE_FORMAT_NUMBERS))
    for frame in range(40):
        board.frame(frame_line(frame))
    board.number(number_line(3, 10))
    board.number(number_line(4, 7))
    board.number(number_line(3, 11))  # a newer line of id 3 replaces the older one
    events = board.wait(0, 0.1)
    assert [event.kind for event in events] == ["header", "frame", "number", "number"]  # in the order of the board
    assert sorted(json.loads(event.data)["number"] for event in events if event.kind == "number") == [7, 11]
    for frame in range(40, 40 + radar.CLIENT_MAX_BEHIND + 5):
        board.frame(frame_line(frame))
    board.number(number_line(5, 9))
    lagged = board.wait(events[-1].seq, 0.1)
    assert [event.kind for event in lagged] == ["frame", "number"]  # the jump keeps the number
    assert "numbers.get(p.id)" in radar.PAGE_HTML and 'addEventListener("number"' in radar.PAGE_HTML


def test_pump_puts_number_lines_on_the_board() -> None:
    board = radar.Board()
    lines = [json.dumps(live_header(LIVE_FORMAT_NUMBERS)), json.dumps(frame_line(0)), json.dumps(number_line(2, 4)),
             json.dumps({"end": True, "frames": 1, "written": 1, "dropped": 0})]
    assert radar.pump(lines, board) == 1
    assert [event.kind for event in board.wait(0, 0.1)] == ["header", "frame", "number", "end"]


# --- The command ---------------------------------------------------------------------------------------


def test_stream_with_the_number_engine_model_writes_numbers_that_equal_its_number_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    feed = tmp_path / "live.jsonl"
    code = cli.main([
        "stream", "--synthetic", "--frames", "150", "--seed", "2295", "--config", str(NUMBERS_FILE),
        "--out", str(tmp_path / "out"), "--live-out", str(feed), "--allow-dirty",
    ])
    printed = capsys.readouterr().out
    assert code == cli.EXIT_OK
    assert "fake model of the synthetic scene" in printed and "number lines" in printed
    assert "no step of the settings asks a model" not in printed
    lines = [json.loads(line) for line in feed.read_text(encoding="utf-8").splitlines()]
    told = {line["id"]: line["export"] for line in lines if line.get("kind") == "number"}
    assert told and lines[0]["format"] == LIVE_FORMAT_NUMBERS
    document = json.loads((tmp_path / "out" / "SNGS-999" / "result.json").read_text())
    assert {line["track_id"]: line["export"] for line in document["numbers"]["lines"]} == told
    assert document["numbers"]["live_out"] == {"given": len(told), "written": len(told), "undelivered": 0}
    assert document["result"]["engines"]["number"].startswith("model:")
    assert "numbers" in document["frame_timing"]
    for state in document["result"]["frames"]:
        for item in state["players"]:
            assert item["number_export"] == told.get(item["track_id"])
    predictions = json.loads(next((tmp_path / "out" / "predictions").rglob("*.json")).read_text())
    for item in predictions["predictions"]:
        jersey = item["attributes"]["jersey"]
        expected = told.get(item["track_id"]) if item["attributes"]["role"] == "player" else None
        assert jersey == (None if expected is None else type(jersey)(expected))


def test_stream_with_the_number_engine_none_has_no_number_step(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    feed = tmp_path / "live.jsonl"
    code = cli.main(["stream", "--synthetic", "--frames", "40", "--out", str(tmp_path / "out"), "--live-out", str(feed),
                     "--allow-dirty"])
    assert code == cli.EXIT_OK
    assert "no step of the settings asks a model" in capsys.readouterr().out
    document = json.loads((tmp_path / "out" / "SNGS-999" / "result.json").read_text())
    assert "numbers" not in document and "numbers" not in document["frame_timing"]
    assert "cost" not in document["provenance"]
    lines = feed.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["format"] == LIVE_FORMAT and not any('"kind"' in line for line in lines)


def test_the_options_of_the_number_step_are_refused_where_they_do_not_apply(tmp_path: Path) -> None:
    base = ["stream", "--synthetic", "--frames", "10", "--out", str(tmp_path / "out"), "--allow-dirty"]
    assert cli.main([*base, "--numbers-max", "3"]) == cli.EXIT_ERROR
    assert cli.main([*base, "--config", str(NUMBERS_FILE), "--cache", str(tmp_path / "cache")]) == cli.EXIT_ERROR
    assert cli.main([*base, "--config", str(NUMBERS_FILE), "--replay"]) == cli.EXIT_ERROR


def test_the_stream_wiring_without_numbers_is_the_hook_of_the_live_output(tmp_path: Path) -> None:
    stream = DetectorStream(NONE, StreamSettings(), clip_id="c", fps=25.0, size=(640, 360))
    live = open_live_out(str(tmp_path / "x.jsonl"), INFO)
    assert wire(stream, None, live) is None
    assert stream.on_given == live.given and stream.on_arrived is None
    live.close()


def test_a_fault_of_the_reading_of_an_answer_is_a_failure_that_stops_the_asking(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(reading: Any, frame_numbers: Any) -> Any:
        raise RuntimeError("a fault of the code that reads the answer")

    monkeypatch.setattr(livenumbers, "decide_number", broken)
    backend = StubBackend()
    numbers = LiveNumbers(
        NUMBERS.model_copy(update={"max_parallel": 1}), backend, NumberSettings(max_pending_images=QUEUE), info=INFO
    )
    drive(numbers, {0: range(0, 120), 1: range(0, 120), 2: range(0, 120)})
    assert numbers.finish() == []
    assert isinstance(numbers.failure, RuntimeError) and len(backend.number_requests()) == 1


def test_after_a_fault_of_the_worker_the_frame_path_keeps_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    numbers = LiveNumbers(NUMBERS, StubBackend(), NumberSettings(max_pending_images=QUEUE), info=INFO)

    def broken(state: FrameState) -> None:
        raise RuntimeError("a fault of the worker")

    monkeypatch.setattr(numbers, "_decided", broken)
    numbers.given(FrameState(frame=0, players=(player(0),)))
    deadline = time.monotonic() + 60
    while numbers.failure is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert isinstance(numbers.failure, RuntimeError)
    for frame in range(1, 50):
        numbers.arrived(frame, IMAGE, [PersonDet(box=box_of(0), role="player")])
        numbers.given(FrameState(frame=frame, players=(player(0),)))
    assert not numbers._events and not numbers._arrival
    assert numbers.finish() == []


def test_stream_with_numbers_and_a_history_of_zero_frames_runs(tmp_path: Path) -> None:
    code = cli.main([
        "stream", "--synthetic", "--frames", "60", "--seed", "2296", "--config", str(NUMBERS_FILE),
        "--history", "0", "--out", str(tmp_path / "out"), "--allow-dirty",
    ])
    assert code == cli.EXIT_OK
    assert json.loads((tmp_path / "out" / "SNGS-999" / "result.json").read_text())["numbers"]["close_frames"] == 1


def test_the_model_session_is_closed_when_the_number_step_cannot_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[Any] = []
    monkeypatch.setattr(cli, "open_session", lambda *args, **kwargs: type("S", (), {"backend": None})())
    monkeypatch.setattr(cli, "close_session", lambda opened: closed.append(opened) or {})

    def no_start(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the number step cannot start")

    monkeypatch.setattr(cli, "LiveNumbers", no_start)
    args = cli.build_parser().parse_args(
        ["stream", "--out", str(tmp_path), "--clip", str(tmp_path), "--numbers-max", "3", "--cache", str(tmp_path / "c")]
    )
    config = PipelineConfig.model_validate(NUMBERS.model_dump())
    with pytest.raises(RuntimeError, match="cannot start"), contextlib.ExitStack() as stack:
        cli.open_live_numbers(args, config, INFO, None, StreamSettings(), stack)
    assert len(closed) == 1


@pytest.fixture
def stop_handler() -> Any:
    """The stop handler of the command line on SIGINT, with fresh holds; the old handler afterwards."""
    stopsignals.reset()
    cli.STOP_REQUESTED.clear()
    old = signal.signal(signal.SIGINT, cli._interrupt)
    yield
    signal.signal(signal.SIGINT, old)
    stopsignals.reset()
    cli.STOP_REQUESTED.clear()


def _until(condition: Any, timeout_s: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def test_stop_signals_at_the_end_of_the_stream_wait_for_the_number_requests_in_flight(
    stop_handler: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """50 stop signals while one paid number request is at the model and three more ids wait.

    The stop is held back (not raised) while the request is in flight, no other request is sent,
    a warning names the request in flight, and the stop is raised when its answer came.
    """
    gate = threading.Event()
    backend = StubBackend(gate=gate)
    numbers = LiveNumbers(
        NUMBERS.model_copy(update={"max_parallel": 1}), backend, NumberSettings(max_pending_images=QUEUE), info=INFO
    )
    drive(numbers, {0: range(0, 120), 1: range(0, 120), 2: range(0, 120), 3: range(100, 120)})
    assert backend.numbers_started.acquire(timeout=60)
    seen: dict[str, Any] = {}

    def signals_then_release() -> None:
        _until(lambda: stopsignals._HOLD.depth > 0)  # finish holds the stops now
        for _ in range(50):
            os.kill(os.getpid(), signal.SIGINT)
        _until(lambda: stopsignals.held_cause() is not None)
        seen["held"], seen["raised"] = stopsignals.held_cause(), stopsignals._HOLD.raised
        _until(lambda: numbers._stopping)  # the wait saw the stop (and warned) before the answer comes
        gate.set()

    helper = threading.Thread(target=signals_then_release, daemon=True)
    helper.start()
    with caplog.at_level(logging.WARNING, logger="pitchvlm"), pytest.raises(KeyboardInterrupt, match="Ctrl-C"):
        numbers.finish()
    helper.join(60)
    assert seen == {"held": "Ctrl-C", "raised": False}
    assert len(backend.number_requests()) == 1 and numbers.answers == 1
    assert "a stop signal came while 1 model requests were in flight" in caplog.text


# --- With the frame feeder and the overload policy of the stream ------------------------------------


class _StillClock:
    """A clock that the test sets: the overload policy judges each line by it, not by the machine."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def fed_numbers(
    scene: SyntheticScene,
    images: list[Any],
    waits: list[float] | None,
    backend: Any,
    live: LiveOut | None = None,
) -> tuple[DetectorStream, LiveNumbers, Any]:
    """A stream fed by a frame feeder, with the overload policy (1 s) and the waits of its lines set by the test
    (frame ``i`` waited ``waits[i]`` s; above 0.5 s it is dropped), and the live number step. ``finish`` of the
    number step is left to the caller."""
    from pitchvlm.stream import FrameFeeder, OverloadPolicy

    info = scene.info
    history = default_history_frames(NUMBERS, info.fps)
    stream = DetectorStream(
        NUMBERS, StreamSettings(delay_frames=25, history_frames=history), clip_id=info.clip_id, fps=info.fps,
        size=(info.width, info.height),
    )
    numbers = LiveNumbers(NUMBERS, backend, NumberSettings(close_frames=history, max_pending_images=QUEUE), info=info)
    clock = _StillClock()
    taken = {"frame": -1}

    def line_clock() -> tuple[float | None, int]:
        return clock.now - (waits[taken["frame"]] if waits is not None else 0.0), 0

    def items() -> Any:
        for item in inputs_of(scene):
            taken["frame"] = item.frame
            yield item

    if live is not None:
        live.overload(max_behind_s=1.0)
        stream.on_dropped = live.skipped
        live.start(delay_frames=25, history_frames=history, numbers=True)
    writer = wire(stream, numbers, live)
    feeder = FrameFeeder(
        stream, items(), lambda frame: images[frame], policy=OverloadPolicy(1.0, clock=clock), line_clock=line_clock
    )
    try:
        for fed in feeder:
            stream.push_fed(fed)
    finally:
        feeder.close()
    stream.finish()
    return stream, numbers, writer


def test_frames_dropped_in_the_middle_of_tracks_give_the_same_numbers() -> None:
    scene = SyntheticScene(n_frames=200, seed=2290)
    images = [scene.image(frame) for frame in range(200)]
    dropped = {40, 41, 42, 43, 60, 61, 90}  # each waited 0.7 s: above half the setting, so not used
    waits = [0.7 if frame in dropped else 0.01 for frame in range(200)]
    stream, numbers, _ = fed_numbers(scene, images, waits, FakeBackend(scene))
    numbers.finish()
    with_drops = numbers.lines  # every line: those taken during the stream and those of the end
    assert set(stream.dropped_frames) == dropped
    _, control, _ = fed_numbers(scene, images, None, FakeBackend(scene))
    control.finish()
    without = control.lines
    assert {(line.track_id, line.export) for line in with_drops} == {(line.track_id, line.export) for line in without}
    assert len(with_drops) > 5 and numbers.report()["images_dropped"] == 0
    assert not any(frame in dropped for line in with_drops for frame in line.crop_frames)


def test_the_number_step_and_the_overload_policy_in_one_stream_and_its_end_holds_stop_signals(
    tmp_path: Path, stop_handler: Any, caplog: pytest.LogCaptureFixture
) -> None:
    scene = SyntheticScene(n_frames=200, seed=2291)
    images = [scene.image(frame) for frame in range(200)]
    waits = [0.7 if frame in (50, 51) else 0.01 for frame in range(200)]
    gate = threading.Event()
    backend = StubBackend(gate=gate)  # every number request waits for the gate
    path = tmp_path / "live.jsonl"
    live = open_live_out(str(path), scene.info)
    stream, numbers, writer = fed_numbers(scene, images, waits, backend, live)
    assert backend.numbers_started.acquire(timeout=60)  # a paid request is in flight at the end of the stream

    def signals_then_release() -> None:
        _until(lambda: stopsignals._HOLD.depth > 0)
        for _ in range(20):
            os.kill(os.getpid(), signal.SIGINT)
        _until(lambda: stopsignals.held_cause() is not None)
        _until(lambda: numbers._stopping)  # the wait saw the stop (and warned) before the answer comes
        gate.set()

    helper = threading.Thread(target=signals_then_release, daemon=True)
    helper.start()
    with caplog.at_level(logging.WARNING, logger="pitchvlm"), pytest.raises(KeyboardInterrupt, match="Ctrl-C"):
        numbers.finish()
    helper.join(60)
    sent = len(backend.number_requests())
    assert numbers.answers == sent >= 1  # every request that reached the model was answered, and no other sent
    writer.write(numbers.take(), stream.n_frames - 1)
    live.finish()
    live.close()
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["format"] == LIVE_FORMAT_NUMBERS and lines[0]["overload_max_behind_s"] == 1.0
    frames = [line for line in lines[1:-1] if "kind" not in line]
    assert {line["frame"] for line in frames}.isdisjoint({50, 51}) and len(frames) == 198
    assert all("overload_dropped" in line for line in frames) and lines[-1]["overload_dropped"] == 2
    assert all("overload_dropped" not in line for line in lines if line.get("kind") == "number")
    assert "a stop signal came while" in caplog.text


def test_a_dropped_decision_writes_the_numbers_that_came(tmp_path: Path) -> None:
    """A run of dropped decisions (the overload policy) does not hold the number lines back."""
    stream = DetectorStream(NUMBERS, StreamSettings(), clip_id="c", fps=25.0, size=(640, 360))
    numbers = LiveNumbers(NUMBERS, StubBackend(), NumberSettings(), info=INFO)
    path = tmp_path / "live.jsonl"
    live = open_live_out(str(path), INFO)
    live.overload(max_behind_s=1.0)
    stream.on_dropped = live.skipped
    live.start(delay_frames=25, history_frames=123, numbers=True)
    wire(stream, numbers, live)
    numbers._answers.append((NumberLine(4, 9, 9, 0, 36, (0, 36), (9, 9), "high", "ready", 0), None))
    stream.on_dropped(40)
    live.finish()
    live.close()
    numbers.abort()
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [line["id"] for line in lines if line.get("kind") == "number"] == [4]
    assert lines[-1]["overload_dropped"] == 1 and live.number_counts == {"given": 1, "written": 1, "undelivered": 0}
