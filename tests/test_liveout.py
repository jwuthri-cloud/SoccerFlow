"""Tests of the live output of the streaming path (``liveout``, ``stream --live-out``).

The stream runs on the synthetic scene with the stand-ins of the tools (``fallbacks.fake``).
Nothing here says how a real detector, a real calibration or a real reader behave.
"""

from __future__ import annotations

import functools
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from pitchvlm import cli, liveout
from pitchvlm.config import EngineSettings, PipelineConfig
from pitchvlm.evalx.streambench import ArraySource, clip_of
from pitchvlm.fallbacks.fake import fake_calibration_answers, fake_detector_answers
from pitchvlm.liveout import LIVE_FORMAT, LiveOut, frame_document, open_live_out
from pitchvlm.schemas import ClipResult, FrameState
from pitchvlm.stream import DetectorStream, StreamSettings, default_history_frames, stand_in_inputs
from pitchvlm.synth import SyntheticScene

REPO = Path(__file__).resolve().parents[1]
CONFIG = PipelineConfig(engines=EngineSettings(people="detector", pitch_mapping="calibration", number="none"))
CLIP_ID = "SNGS-999"


def read_lines(path: Path) -> list[dict[str, Any]]:
    """Read the lines of a live output file."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run_with_live_out(
    scene: Any, images: list[Any], delay: int | None, live: LiveOut, after_push: Any = None
) -> tuple[DetectorStream, ClipResult]:
    """Run a stream on a synthetic clip with a live output; return the stream and its result."""
    people, balls = fake_detector_answers(scene)
    cameras = fake_calibration_answers(scene)
    info = scene.info
    history = None if delay is None else default_history_frames(CONFIG, info.fps)
    settings = StreamSettings(delay_frames=delay, history_frames=history)
    stream = DetectorStream(CONFIG, settings, clip_id=info.clip_id, fps=info.fps, size=(info.width, info.height))
    live.start(delay_frames=delay, history_frames=history)
    stream.on_given = live.given
    for item in stand_in_inputs(people, balls, cameras):
        given = stream.push(item, images[item.frame])
        if after_push is not None:
            after_push(item.frame, given)
    stream.finish()
    live.finish()
    return stream, stream.result()


def without_clock(document: dict[str, Any]) -> dict[str, Any]:
    """Return a frame line without its wall clock (the only value that two runs do not share)."""
    return {key: value for key, value in document.items() if key != "decided_unix"}


def transition_frames(spans: Any) -> set[int]:
    """Return the frames inside the given transition spans."""
    return {frame for start, stop in spans for frame in range(start, stop + 1)}


@pytest.mark.parametrize("delay", [0, 5, 25, None])
def test_each_frame_line_holds_the_state_of_its_frame_in_the_final_result(tmp_path: Path, delay: int | None) -> None:
    scene = SyntheticScene(n_frames=50, seed=3)
    images = [scene.image(frame) for frame in range(50)]
    path = tmp_path / "live.jsonl"
    live = open_live_out(str(path), scene.info)
    stream, result = run_with_live_out(scene, images, delay, live)
    report = live.close()
    lines = read_lines(path)
    header, frames, end = lines[0], lines[1:-1], lines[-1]
    assert header["format"] == LIVE_FORMAT and header["policy"] == "file"
    assert (header["clip"], header["n_frames"], header["fps"], header["delay_frames"]) == (CLIP_ID, 50, 25.0, delay)
    assert header["pitch_m"] == [105.0, 68.0]
    # Once each, in frame order, and value for value the state and the camera of the result file.
    assert [line["frame"] for line in frames] == list(range(50))
    for line, state, camera in zip(frames, result.frames, result.homographies, strict=True):
        expected = frame_document(
            state,
            fps=25.0,
            camera=camera,
            last=line["decided_at"],
            transition=False,  # a plain clip: no dissolve and no cut
            cut=False,
            decided_unix=0.0,
            dropped=0,
        )
        assert without_clock(line) == without_clock(expected)
        assert line["camera"] == camera.source
        assert line["delay_frames"] == line["decided_at"] - line["frame"]
        if delay is None:
            assert line["decided_at"] == 49
        else:
            assert line["decided_at"] == min(49, line["frame"] + delay)
    assert end == {"end": True, "frames": 50, "written": 50, "dropped": 0}
    assert (report.given, report.written, report.dropped, report.undelivered) == (50, 50, 0, 0)


def test_the_file_policy_writes_each_line_before_the_next_frame_comes(tmp_path: Path) -> None:
    scene = SyntheticScene(n_frames=40, seed=3)
    images = [scene.image(frame) for frame in range(40)]
    path = tmp_path / "live.jsonl"
    live = open_live_out(str(path), scene.info)
    seen: list[tuple[int, int, int]] = []

    def check(frame: int, given: list[FrameState]) -> None:
        # Read from the disk by another handle: the lines of every frame given so far are there.
        lines = read_lines(path)
        seen.append((frame, len(lines) - 1, len(given)))

    run_with_live_out(scene, images, 10, live, after_push=check)
    live.close()
    assert [frames for frame, frames, _ in seen] == [max(0, frame - 10 + 1) for frame, _, _ in seen]
    assert sum(given for _, _, given in seen) == 30


@pytest.mark.slow
def test_the_transition_flag_is_the_one_known_at_the_decision_and_the_result_cuts_may_hold_more(
    tmp_path: Path,
) -> None:
    # A dissolve of 10 frames from frame 120: with a delay of 5 its frames are decided before the
    # dissolve is known, so their flag stays false; the cuts of the result hold the dissolve.
    scene, images = clip_of("dissolve", 0)
    path = tmp_path / "live.jsonl"
    live = open_live_out(str(path), scene.info)
    stream, result = run_with_live_out(scene, images, 5, live)
    live.close()
    frames = read_lines(path)[1:-1]
    flagged = {line["frame"] for line in frames if line["transition"]}
    final = transition_frames(span for kind, span, _ in stream.transition_log if kind == "final")
    assert final and final <= set(result.cuts)
    assert flagged <= final | transition_frames(span for kind, span, _ in stream.transition_log if kind == "early")
    assert final - flagged, "with a delay of 5 some frames of the dissolve were decided before it was known"
    # The states of those frames are still the states of the result.
    for line, state in zip(frames, result.frames, strict=True):
        assert [player["id"] for player in line["players"]] == [player.track_id for player in state.players]


def slow_line_reader(fd: int, lines: list[bytes], pause_s: float, fast: threading.Event) -> None:
    """Read a pipe one line at a time, one byte per read, with ``pause_s`` after each line until ``fast`` is set."""
    line = b""
    while True:
        byte = os.read(fd, 1)
        if not byte:
            if line:
                lines.append(line)
            return
        line += byte
        if byte == b"\n":
            lines.append(line)
            line = b""
            if not fast.is_set():
                fast.wait(pause_s)


@functools.cache
def states_of(n_frames: int) -> tuple[SyntheticScene, ClipResult]:
    """Return a scene and the result of a stream on it with a delay of 5 (the states that a live output gets)."""
    scene = SyntheticScene(n_frames=n_frames, seed=3)
    people, balls = fake_detector_answers(scene)
    stream = DetectorStream(
        CONFIG,
        StreamSettings(delay_frames=5, history_frames=default_history_frames(CONFIG, 25.0)),
        clip_id=CLIP_ID,
        fps=25.0,
        size=(scene.info.width, scene.info.height),
    )
    for item in stand_in_inputs(people, balls, fake_calibration_answers(scene)):
        stream.push(item, scene.image(item.frame))
    stream.finish()
    return scene, stream.result()


def test_a_reader_of_one_line_per_second_does_not_stall_the_stream_and_gets_the_newest_lines() -> None:
    scene, result = states_of(60)
    read_fd, write_fd = os.pipe()
    live = LiveOut(write_fd, scene.info, target="a pipe", pipe=True, close_fd=True)
    lines: list[bytes] = []
    fast = threading.Event()
    reader = threading.Thread(target=slow_line_reader, args=(read_fd, lines, 1.0, fast), daemon=True)
    reader.start()
    live.start(delay_frames=5, history_frames=123)
    hand_over: list[float] = []
    started = time.monotonic()
    for state, camera in zip(result.frames, result.homographies, strict=True):
        before = time.perf_counter()
        live.given(state, camera=camera, last=state.frame, transition=False, cut=False)
        hand_over.append(time.perf_counter() - before)
        time.sleep(max(0.0, started + (state.frame + 1) / 25.0 - time.monotonic()))
    live.finish()
    slow_lines = len(lines)
    fast.set()
    report = live.close(wait_s=10.0)
    reader.join(10.0)
    os.close(read_fd)
    documents = [json.loads(line) for line in lines]
    frames = [document for document in documents[1:] if "frame" in document]
    # The stream never waited for the reader: 60 frames at 25 fps against a reader of 1 line per second.
    # A hand-over takes the lock and appends; 0.2 s is far above it and far below a reader of 1 line per second.
    assert max(hand_over) < 0.2, f"a hand-over took {max(hand_over) * 1000:.1f} ms"
    assert slow_lines <= 8
    assert documents[0]["format"] == LIVE_FORMAT and documents[0]["policy"] == "pipe"
    assert documents[-1]["end"] is True
    numbers = [line["frame"] for line in frames]
    assert numbers == sorted(set(numbers)) and numbers[-1] == 59, "frame order, each once, the newest at the end"
    assert len(numbers) + report.dropped == 60 and report.dropped > 0
    assert documents[-1] == {"end": True, "frames": 60, "written": len(numbers), "dropped": report.dropped}
    # The count of lines left out before each line grows with the gaps.
    for previous, line in zip(frames, frames[1:]):
        assert line["dropped"] - previous["dropped"] >= line["frame"] - previous["frame"] - 1 >= 0
    assert (report.given, report.written, report.undelivered, report.reader_gone) == (60, len(numbers), 0, None)


def test_a_reader_that_never_reads_does_not_hold_the_end_of_the_stream() -> None:
    scene, result = states_of(30)
    read_fd, write_fd = os.pipe()
    live = LiveOut(write_fd, scene.info, target="a pipe", pipe=True, close_fd=True)
    live.start(delay_frames=5, history_frames=123)
    for state, camera in zip(result.frames, result.homographies, strict=True):
        live.given(state, camera=camera, last=state.frame, transition=False, cut=False)
        time.sleep(0.005)  # the writer takes each line, until the pipe is full
    live.finish()
    before = time.monotonic()
    report = live.close(wait_s=0.3)
    assert time.monotonic() - before < 2.0
    # The pipe is full (30 lines of about 5 kB); the writer is blocked, and the end line waits.
    assert report.written < 30 and report.undelivered >= 2 and report.given == 30
    assert report.written + report.dropped + report.undelivered == 31  # 30 frames and the end line
    os.close(read_fd)


@pytest.mark.parametrize("header_out", [True, False], ids=["header-written-before-the-close", "header-not-written"])
def test_a_reader_that_goes_away_ends_the_output_and_not_the_stream(
    monkeypatch: pytest.MonkeyPatch, header_out: bool
) -> None:
    """The reader goes away right after the start. Whether the writer had written the header by then was left to
    timing, and the counts differed by one: a header that was not written counted as not delivered, one that was
    written counted nowhere (CI on Linux, 1 of 3 jobs). The header is in no count now; both orders are forced."""
    scene, result = states_of(20)
    read_fd, write_fd = os.pipe()
    closed = threading.Event()
    write_all = liveout._write_all

    def header_after_the_close(fd: int, data: bytes) -> None:
        if data.startswith(b'{"format"'):
            assert closed.wait(timeout=30.0)
        write_all(fd, data)

    if not header_out:
        monkeypatch.setattr(liveout, "_write_all", header_after_the_close)
    live = LiveOut(write_fd, scene.info, target="a pipe", pipe=True, close_fd=True)
    live.start(delay_frames=5, history_frames=123)
    if header_out:
        received = b""
        while not received.endswith(b"\n"):
            received += os.read(read_fd, 65536)
        assert json.loads(received)["format"] == LIVE_FORMAT
    os.close(read_fd)
    closed.set()
    for state, camera in zip(result.frames, result.homographies, strict=True):
        live.given(state, camera=camera, last=state.frame, transition=False, cut=False)
        time.sleep(0.002)
    live.finish()
    report = live.close(wait_s=10.0)
    assert report.reader_gone is not None and report.given == 20 and report.written == 0
    assert report.written + report.dropped + report.undelivered == 21  # 20 frames and the end line


def test_a_header_whose_write_is_still_held_at_the_close_is_in_no_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reader holds the write of the header until after the close (a terminal that is stopped): the frames wait
    or are left out, the end line waits, and the header that never went out is not a line not delivered."""
    scene, result = states_of(20)
    release = threading.Event()
    written: list[bytes] = []

    def held_header(fd: int, data: bytes) -> None:
        if data.startswith(b'{"format"'):
            assert release.wait(timeout=30.0)
        written.append(data)

    monkeypatch.setattr(liveout, "_write_all", held_header)
    live = LiveOut(-1, scene.info, target="a pipe", pipe=True, close_fd=False)
    live.start(delay_frames=5, history_frames=123)
    time.sleep(liveout.PIPE_BLOCKED_S * 4)
    for state, camera in zip(result.frames, result.homographies, strict=True):
        live.given(state, camera=camera, last=state.frame, transition=False, cut=False)
    live.finish()
    try:
        report = live.close(wait_s=0.2)
    finally:
        release.set()
    assert (report.given, report.written, report.dropped, report.undelivered) == (20, 0, 19, 2)


def test_a_named_pipe_takes_the_pipe_policy(tmp_path: Path) -> None:
    scene, result = states_of(10)
    fifo = tmp_path / "live.pipe"
    os.mkfifo(fifo)
    lines: list[bytes] = []
    fast = threading.Event()
    fast.set()

    def read() -> None:
        fd = os.open(fifo, os.O_RDONLY)
        slow_line_reader(fd, lines, 0.0, fast)
        os.close(fd)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    live = open_live_out(str(fifo), scene.info)
    assert live.policy == "pipe"
    live.start(delay_frames=5, history_frames=123)
    for state, camera in zip(result.frames, result.homographies, strict=True):
        live.given(state, camera=camera, last=state.frame, transition=False, cut=False)
        time.sleep(0.005)
    live.finish()
    report = live.close(wait_s=5.0)
    reader.join(5.0)
    documents = [json.loads(line) for line in lines]
    assert documents[0]["policy"] == "pipe" and documents[-1]["end"] is True
    assert report.written + report.dropped == 10 and report.undelivered == 0


def test_a_live_output_takes_frames_only_between_start_and_finish(tmp_path: Path) -> None:
    scene, result = states_of(8)
    live = open_live_out(str(tmp_path / "live.jsonl"), scene.info)
    state, camera = result.frames[0], result.homographies[0]
    with pytest.raises(ValueError, match="between start and finish"):
        live.given(state, camera=camera, last=0, transition=False, cut=False)
    live.start(delay_frames=0, history_frames=0)
    with pytest.raises(ValueError, match="names frame"):
        live.given(state, camera=result.homographies[1], last=0, transition=False, cut=False)
    live.finish()
    with pytest.raises(ValueError, match="between start and finish"):
        live.given(state, camera=camera, last=0, transition=False, cut=False)
    live.close()
    with pytest.raises(ValueError, match="already closed"):
        live.close()


# --- The command line ----------------------------------------------------------------------


def test_stream_with_live_out_gives_the_result_of_stream_without_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "render_clip", lambda source, result, out_path: out_path)
    common = ["stream", "--synthetic", "--frames", "40", "--delay", "10", "--seed", "3", "--allow-dirty"]
    live = tmp_path / "feed" / "live.jsonl"
    live.parent.mkdir()
    assert cli.main([*common, "--out", str(tmp_path / "plain")]) == 0
    assert "live output" not in capsys.readouterr().out
    assert cli.main([*common, "--out", str(tmp_path / "live"), "--live-out", str(live)]) == 0
    printed = capsys.readouterr().out
    assert f"live output {live} (policy file): 40 frame lines written of 40 decided, 0 left out" in printed
    plain = json.loads((tmp_path / "plain" / CLIP_ID / "result.json").read_text(encoding="utf-8"))
    with_live = json.loads((tmp_path / "live" / CLIP_ID / "result.json").read_text(encoding="utf-8"))
    assert "live_out" not in plain
    assert with_live["live_out"]["written"] == 40 and with_live["live_out"]["policy"] == "file"
    assert plain["result"] == with_live["result"]
    lines = read_lines(live)
    frames = with_live["result"]["frames"]
    assert len(lines) == 42 and lines[-1]["end"] is True
    for line, frame in zip(lines[1:-1], frames, strict=True):
        assert line["frame"] == frame["frame"]
        assert [(p["id"], p["x"], p["y"], p["team"], p["role"]) for p in line["players"]] == [
            (p["track_id"], p["pitch_x"], p["pitch_y"], p["team_key"], p["role"]) for p in frame["players"]
        ]


def test_live_out_refuses_the_repository_a_folder_and_an_input_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    common = ["stream", "--synthetic", "--frames", "10", "--out", str(tmp_path / "out"), "--allow-dirty"]
    assert cli.main([*common, "--live-out", str(REPO / "outputs" / "live.jsonl")]) == 2
    assert "lies inside the work tree" in capsys.readouterr().err
    assert not (REPO / "outputs" / "live.jsonl").exists()
    assert cli.main([*common, "--live-out", str(tmp_path)]) == 2
    assert "is a folder" in capsys.readouterr().err
    settings = tmp_path / "settings.yaml"
    settings.write_text("engines:\n  people: detector\n  pitch_mapping: calibration\n  number: none\n", encoding="utf-8")
    assert cli.main([*common, "--config", str(settings), "--live-out", str(settings)]) == 2
    assert "names the same file as --config" in capsys.readouterr().err
    assert settings.read_text(encoding="utf-8").startswith("engines:")


def test_live_out_dash_sends_the_lines_to_the_standard_output_and_the_report_to_the_error_stream(
    tmp_path: Path,
) -> None:
    command = [
        sys.executable,
        "-m",
        "pitchvlm.cli",
        "stream",
        "--synthetic",
        "--frames",
        "20",
        "--delay",
        "5",
        "--out",
        str(tmp_path / "out"),
        "--allow-dirty",
        "--live-out",
        "-",
    ]
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    done = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    lines = [json.loads(line) for line in done.stdout.splitlines()]
    assert lines[0]["format"] == LIVE_FORMAT and lines[0]["policy"] == "pipe"
    assert lines[-1]["end"] is True and lines[-1]["frames"] == 20
    assert [line["frame"] for line in lines[1:-1]] == sorted({line["frame"] for line in lines[1:-1]})
    assert "live output standard output (policy pipe)" in done.stderr
    assert "result:" in done.stderr


def test_the_lines_of_a_dissolve_clip_hold_every_frame_once(tmp_path: Path) -> None:
    # The source of the bench: ArraySource reads the same images as the stream does.
    scene, images = clip_of("cut", 1, 60)
    source = ArraySource(scene.info, images)
    path = tmp_path / "live.jsonl"
    live = open_live_out(str(path), scene.info)
    stream, result = run_with_live_out(scene, [source.read(frame) for frame in range(60)], 25, live)
    live.close()
    frames = read_lines(path)[1:-1]
    assert [line["frame"] for line in frames] == list(range(60))
    cuts = [line["frame"] for line in frames if line["cut"]]
    assert cuts == [30] and 30 in result.cuts


# --- Second read ---------------------------------------------------------------------------


def test_an_error_of_the_pipe_writer_is_raised_by_close(monkeypatch: pytest.MonkeyPatch) -> None:
    scene, result = states_of(10)
    read_fd, write_fd = os.pipe()
    real = liveout.dumps

    def broken(document: dict[str, Any]) -> bytes:
        if document.get("frame") == 3:
            raise RuntimeError("planted")
        return real(document)

    monkeypatch.setattr(liveout, "dumps", broken)
    live = LiveOut(write_fd, scene.info, target="a pipe", pipe=True, close_fd=True)
    live.start(delay_frames=5, history_frames=123)
    for state, camera in zip(result.frames, result.homographies, strict=True):
        live.given(state, camera=camera, last=state.frame, transition=False, cut=False)
        time.sleep(0.01)
    live.finish()
    with pytest.raises(liveout.LiveOutError, match="planted"):
        live.close(wait_s=5.0)
    os.close(read_fd)


def test_a_frame_line_is_checked_in_the_thread_of_the_stream() -> None:
    scene, result = states_of(10)
    read_fd, write_fd = os.pipe()
    live = LiveOut(write_fd, scene.info, target="a pipe", pipe=True, close_fd=True)
    live.start(delay_frames=5, history_frames=123)
    with pytest.raises(ValueError, match="names frame"):
        live.given(result.frames[3], camera=result.homographies[4], last=3, transition=False, cut=False)
    live.finish()
    live.close(wait_s=5.0)
    os.close(read_fd)


def test_a_write_error_of_a_file_stops_the_stream_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "live.jsonl"
    path.write_text("", encoding="utf-8")

    def read_only(target: str, info: Any) -> LiveOut:
        return LiveOut(os.open(path, os.O_RDONLY), info, target=target, pipe=False, close_fd=True)

    monkeypatch.setattr(cli, "open_live_out", read_only)
    command = ["stream", "--synthetic", "--frames", "10", "--out", str(tmp_path / "out"), "--allow-dirty"]
    assert cli.main([*command, "--live-out", str(path)]) == 2
    assert "cannot be written" in capsys.readouterr().err


def test_a_close_without_finish_stops_the_writer_and_ends_the_pipe() -> None:
    scene, result = states_of(10)
    read_fd, write_fd = os.pipe()
    live = LiveOut(write_fd, scene.info, target="a pipe", pipe=True, close_fd=True)
    live.start(delay_frames=5, history_frames=123)
    live.given(result.frames[0], camera=result.homographies[0], last=5, transition=False, cut=False)
    received = b""
    while received.count(b"\n") < 2:  # the writer is idle once both lines are out (a sleep left it to timing)
        received += os.read(read_fd, 65536)
    before = time.monotonic()
    report = live.close(wait_s=5.0)  # the stream stopped: the idle writer stops at once, nothing waits
    assert time.monotonic() - before < 1.0
    lines: list[bytes] = []
    fast = threading.Event()
    fast.set()
    slow_line_reader(read_fd, lines, 0.0, fast)  # returns at the end of the pipe: the writer closed it
    os.close(read_fd)
    assert [json.loads(line).get("frame") for line in received.splitlines()] == [None, 0] and lines == []
    assert report.undelivered == 0 and report.written == 1


def test_live_out_refuses_a_link_into_the_repository_the_standard_output_and_the_out_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    common = ["stream", "--synthetic", "--frames", "10", "--out", str(out), "--allow-dirty"]
    link = tmp_path / "live.jsonl"
    inside = REPO / "outputs" / "linked-live.jsonl"
    link.symlink_to(inside)
    assert cli.main([*common, "--live-out", str(link)]) == 2
    assert "lies inside the work tree" in capsys.readouterr().err
    assert not inside.exists()
    for descriptor in ("/dev/fd/1", "/dev/stdout"):
        assert cli.main([*common, "--live-out", descriptor]) == 2
        assert "is the standard output or another open file of the command: give - instead" in capsys.readouterr().err
    assert cli.main([*common, "--live-out", str(out / "SNGS-999" / "result.json")]) == 2
    assert "lies in --out" in capsys.readouterr().err


def test_a_fast_reader_gets_every_line_of_a_burst_of_decisions() -> None:
    # The last frames of a clip are decided together at its end: the writer is not held by its
    # reader, it only waits for its turn, so no line is left out.
    scene, result = states_of(60)
    read_fd, write_fd = os.pipe()
    live = LiveOut(write_fd, scene.info, target="a pipe", pipe=True, close_fd=True)
    lines: list[bytes] = []
    fast = threading.Event()
    fast.set()
    reader = threading.Thread(target=slow_line_reader, args=(read_fd, lines, 0.0, fast), daemon=True)
    reader.start()
    live.start(delay_frames=5, history_frames=123)
    for state, camera in zip(result.frames, result.homographies, strict=True):
        live.given(state, camera=camera, last=state.frame, transition=False, cut=False)
    live.finish()
    report = live.close(wait_s=10.0)
    reader.join(10.0)
    os.close(read_fd)
    frames = [json.loads(line)["frame"] for line in lines[1:-1]]
    assert frames == list(range(60)) and (report.written, report.dropped) == (60, 0)


# --- Planted bugs that no test caught (second read of round 8, docs/DECISIONS.md) ----------------------------


def test_a_frame_line_gives_the_time_of_its_frame_and_refuses_a_camera_of_another_frame() -> None:
    scene, result = states_of(12)
    for frame in (0, 5, 11):
        document = frame_document(
            result.frames[frame],
            fps=12.5,
            camera=result.homographies[frame],
            last=frame + 1,
            transition=False,
            cut=False,
            decided_unix=0.0,
            dropped=0,
        )
        assert document["t_s"] == frame / 12.5
        assert document["delay_frames"] == 1
    for other in (4, 6):  # an earlier and a later camera
        with pytest.raises(ValueError, match="names frame"):
            frame_document(
                result.frames[5],
                fps=25.0,
                camera=result.homographies[other],
                last=5,
                transition=False,
                cut=False,
                decided_unix=0.0,
                dropped=0,
            )


def test_the_shirt_number_of_a_line_is_the_displayed_one_and_else_the_exported_one() -> None:
    scene, result = states_of(8)
    state = result.frames[4]
    player = state.players[0]
    cases = [((7, 9), 7), ((None, 9), 9), ((7, None), 7), ((None, None), None)]
    for (display, export), expected in cases:
        changed = player.model_copy(update={"number_display": display, "number_export": export})
        document = frame_document(
            state.model_copy(update={"players": [changed]}),
            fps=25.0,
            camera=result.homographies[4],
            last=4,
            transition=False,
            cut=False,
            decided_unix=0.0,
            dropped=0,
        )
        assert document["players"][0]["number"] == expected


def test_a_held_writer_keeps_exactly_the_pending_lines_and_leaves_out_the_older_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The writer of frame 0 is held by its reader (a write that lasts longer than ``PIPE_BLOCKED_S``). Frames 1 to 5
    come meanwhile: one line waits (``PIPE_PENDING_LINES``), so frames 1 to 4 are left out and frame 5 goes next."""
    scene, result = states_of(8)
    written: list[bytes] = []
    in_write, release = threading.Event(), threading.Event()

    def held_write(fd: int, data: bytes) -> None:
        if b'"frame":0,' in data:
            in_write.set()
            assert release.wait(timeout=30.0)
        written.append(data)

    monkeypatch.setattr(liveout, "_write_all", held_write)
    live = LiveOut(-1, scene.info, target="a pipe", pipe=True, close_fd=False)
    live.start(delay_frames=0, history_frames=None)
    live.given(result.frames[0], camera=result.homographies[0], last=0, transition=False, cut=False)
    assert in_write.wait(timeout=30.0)
    time.sleep(liveout.PIPE_BLOCKED_S * 4)
    for frame in range(1, 6):
        live.given(result.frames[frame], camera=result.homographies[frame], last=frame, transition=False, cut=False)
    release.set()
    live.finish()
    report = live.close(wait_s=30.0)
    lines = [json.loads(data) for data in written]
    assert [line.get("frame") for line in lines[1:-1]] == [0, 5]
    assert lines[2]["dropped"] == 4 and lines[-1] == {"end": True, "frames": 6, "written": 2, "dropped": 4}
    assert (report.given, report.written, report.dropped, report.undelivered) == (6, 2, 4, 0)


def test_anything_but_a_regular_file_takes_the_pipe_policy(tmp_path: Path) -> None:
    """A terminal, a socket or a device would block the stream with the file policy: only a regular file takes it."""
    path = tmp_path / "live.jsonl"
    path.write_text("", encoding="utf-8")
    with path.open("a") as handle:
        assert not liveout.is_pipe_like(handle.fileno())
    first, second = socket.socketpair()
    try:
        assert liveout.is_pipe_like(first.fileno())
    finally:
        first.close()
        second.close()
    device = os.open(os.devnull, os.O_WRONLY)
    try:
        assert liveout.is_pipe_like(device)
    finally:
        os.close(device)
    read_fd, write_fd = os.pipe()
    try:
        assert liveout.is_pipe_like(write_fd)
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_a_live_output_file_that_exists_is_emptied_first(tmp_path: Path) -> None:
    """A second stream to the same file starts a new feed: lines added after the old ones would give two headers."""
    scene, result = states_of(4)
    path = tmp_path / "live.jsonl"
    path.write_text('{"format": "an old feed"}\n' * 3, encoding="utf-8")
    live = open_live_out(str(path), scene.info)
    live.start(delay_frames=0, history_frames=None)
    live.given(result.frames[0], camera=result.homographies[0], last=0, transition=False, cut=False)
    live.finish()
    live.close()
    lines = read_lines(path)
    assert [lines[0]["format"], lines[1]["frame"], lines[2]["end"]] == [LIVE_FORMAT, 0, True] and len(lines) == 3
