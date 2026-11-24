"""Tests of the live feed written for the planted bugs that the other tests did not catch (a mutation check).

Each test names the planted change that it catches. They call the functions directly, with
times given by the test, and start no process.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from pitchvlm import cli
from pitchvlm.feed import TOOL_FEED_FORMAT, FeedMerger
from pitchvlm.stream import STREAM_FORMAT, DetectorStream, QueuedLines, read_stream_lines

REPO = Path(__file__).resolve().parents[1]
DETECTOR = {"name": "rfdetr", "model": "RFDETRMedium", "min_score": 0.5, "antialias": False}
CALIBRATION = {"name": "pnlcalib", "kp_threshold": 0.3434, "line_threshold": 0.7867, "pnl_refine": True}
CAMERA = {"homography_pitch_to_image": [[10.0, 0.0, 640.0], [0.0, 10.0, 360.0], [0.0, 0.0, 1.0]], "pitch_origin": "centre"}
PEOPLE = {"labels": ["person"], "rows": [[100.0, 200.0, 120.0, 260.0, 0.9, 0]]}


def header(tool: str, n_frames: int) -> str:
    settings = DETECTOR if tool == "detector" else CALIBRATION
    return json.dumps(
        {
            "format": TOOL_FEED_FORMAT,
            "clip": "CLIP-A",
            "n_frames": n_frames,
            "video_sha256": None,
            "job_sha256": "0" * 64,
            "tool": settings,
        }
    )


def line(tool: str, frame: int, arrived_at: float = 1000.0) -> str:
    payload = PEOPLE if tool == "detector" else CAMERA
    return json.dumps({"frame": frame, "seconds": 0.02, "arrived_at": arrived_at, "payload": payload})


def skipped(frame: int) -> str:
    return json.dumps({"frame": frame, "arrived_at": 1000.0, "skipped": True})


def started(n_frames: int, wait_s: float = 1.0) -> FeedMerger:
    merger = FeedMerger(wait_s)
    merger.header("detector", header("detector", n_frames), 0.0)
    merger.header("calibration", header("calibration", n_frames), 0.0)
    return merger


def frames_of(out: list[str]) -> list[dict[str, Any]]:
    return [json.loads(text) for text in out if '"frame"' in text]


# --- feed.py ------------------------------------------------------------------------------------------


def test_a_line_of_the_frame_just_past_the_clip_is_broken() -> None:
    """F3: the range check of a frame moved by one (``frame <= n_frames``) took frame n_frames as a part."""
    merger = started(3)
    merger.line("detector", line("detector", 3), 0.0)
    assert merger.counts.broken_lines["detector"] == 1
    assert merger.deadline() is None  # no clock started: frame 0 still waits for a line


def test_a_part_after_a_skip_line_of_the_same_frame_is_a_repeated_line() -> None:
    """F13: the repeated-line check ignored the skipped frames, so a later part of a skipped frame was kept."""
    merger = started(2)
    merger.take(0.0)
    merger.line("calibration", skipped(0), 0.0)
    merger.line("calibration", line("calibration", 0), 0.0)
    merger.line("detector", line("detector", 0), 0.0)
    out = frames_of(merger.take(0.0))
    assert out[0]["calibration"] == {"missing": "skipped"}
    assert merger.counts.repeated_lines == {"detector": 0, "calibration": 1}
    assert merger.counts.skipped == {"detector": 0, "calibration": 1}


# --- tools/run_tool.py --------------------------------------------------------------------------------


def load_run_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_tool_for_planted", REPO / "tools" / "run_tool.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_tool_for_planted"] = module
    spec.loader.exec_module(module)
    return module


def test_a_reader_that_comes_back_after_some_lines_gets_the_whole_feed_from_its_header(tmp_path: Path) -> None:
    """R2: the reopen sent only the last line, so a reader that came back after the header had gone out got no header.
    (The process test breaks the pipe on the header itself, where the last line is the whole feed.)"""
    run_tool = load_run_tool()
    pipe = tmp_path / "feed.fifo"
    os.mkfifo(pipe)
    first: dict[str, Any] = {}
    opener = threading.Thread(target=lambda: first.setdefault("handle", open(pipe, encoding="utf-8")), daemon=True)
    opener.start()
    got: list[str] = []
    second = threading.Thread(target=lambda: got.append(Path(pipe).read_text(encoding="utf-8")), daemon=True)

    def sleep(seconds: float) -> None:
        if second.ident is None:  # the old end is closed now: the new reader cannot pair with it
            second.start()
        time.sleep(seconds)

    feed = run_tool.NamedPipeFeed(str(pipe), wait_s=10.0, poll_s=0.01, sleep=sleep)
    opener.join(10.0)
    try:
        feed.write("H\n")
        feed.write("A\n")
        assert [first["handle"].readline() for _ in range(2)] == ["H\n", "A\n"]
        first["handle"].close()  # the first reader goes away
        feed.write("B\n")
    finally:
        feed.close()
    second.join(10.0)
    assert got == ["H\nA\nB\n"] and feed.reopened == 1


# --- stream.py ----------------------------------------------------------------------------------------


def stream_lines(detector: dict[str, Any], calibration: dict[str, Any]) -> list[str]:
    head = {
        "format": STREAM_FORMAT,
        "clip": "CLIP-A",
        "n_frames": 1,
        "video_sha256": None,
        "detector": DETECTOR,
        "calibration": CALIBRATION,
    }
    return [json.dumps(head), json.dumps({"frame": 0, "detector": detector, "calibration": calibration})]


def test_the_arrival_of_a_frame_is_the_earlier_arrival_of_its_two_parts() -> None:
    """S1: ``arrived_at`` took the later of the two parts (``max``) instead of the earlier one."""
    lines = stream_lines(
        {"seconds": 0.02, "payload": PEOPLE, "arrived_at": 12.0},
        {"seconds": 0.03, "payload": CAMERA, "arrived_at": 10.0},
    )
    _, inputs = read_stream_lines(lines)
    assert next(inputs).arrived_at == 10.0


def test_a_missing_part_with_an_unknown_reason_is_refused() -> None:
    """S3: the reason of a missing part was not checked against ``MISSING_REASONS``."""
    lines = stream_lines({"seconds": 0.02, "payload": PEOPLE}, {"missing": "lost"})
    _, inputs = read_stream_lines(lines)
    with pytest.raises(ValueError, match="line 2: a missing part of the calibration"):
        next(inputs)


def test_the_queue_gives_the_time_at_which_a_line_was_read_not_the_time_at_which_it_was_taken() -> None:
    """S9: ``last_read_at`` took the time at which the reader took the line, so the wait in the queue was lost."""
    now = [5.0]
    pulled: list[str] = []

    def lines() -> Any:
        for text in ("a\n", "b\n"):
            pulled.append(text)
            yield text

    queued = QueuedLines(lines(), max_lines=1, clock=lambda: now[0])
    deadline = time.monotonic() + 10.0
    while len(pulled) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.1)  # the thread reads the clock for "b" right after it pulled it, then waits for room
    now[0] = 9.0  # the reader takes the lines 4 s later
    seen = []
    for text in queued:
        seen.append((text, queued.last_read_at))
    assert seen == [("a\n", 5.0), ("b\n", 5.0)]


def test_a_span_known_early_is_logged_again_when_it_is_final() -> None:
    """S12: the log of the transitions kept one entry per span, so the final entry of a span known early was lost."""
    holder = SimpleNamespace(transition_log=[])
    DetectorStream._log_transitions(holder, "early", [(40, 60)], 45)  # type: ignore[arg-type]
    DetectorStream._log_transitions(holder, "early", [(40, 60)], 46)  # type: ignore[arg-type]
    DetectorStream._log_transitions(holder, "final", [(40, 60)], 70)  # type: ignore[arg-type]
    assert holder.transition_log == [("early", (40, 60), 45), ("final", (40, 60), 70)]


# --- cli.py -------------------------------------------------------------------------------------------


def test_stream_lines_take_the_queue_size_of_the_command_line(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """C3: ``stream --lines FILE --queue N`` built its queue with the default size, not N."""
    sizes: list[int] = []
    real = cli.QueuedLines

    def recording(lines: Any, max_lines: int, *rest: Any) -> QueuedLines:
        sizes.append(max_lines)
        return real(lines, max_lines, *rest)

    monkeypatch.setattr(cli, "QueuedLines", recording)
    path = tmp_path / "lines.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    args = argparse.Namespace(lines=path, queue=7, detections=None, calibration=None)
    with pytest.raises(cli.CliError, match="the stream cannot be read"):
        cli.stream_inputs(args, None, None, None)  # type: ignore[arg-type]
    assert sizes == [7]
