"""Tests of the merger of the two tool feeds (``feed.py``): its policy event by event, and its threads on real pipes.

The policy cases of the decision entry: one tool slower than the other, a tool that dies,
lines out of order, a cut-off line, a frame that one tool never sends, a skipped frame, and the
back-pressure of a slow reader. The time is given by the tests (``FeedMerger``) or is that of
the machine (``merge_feeds``).
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

import pytest

from pitchvlm.feed import TOOL_FEED_FORMAT, FeedError, FeedMerger, fd_lines, merge_feeds
from pitchvlm.stream import STREAM_FORMAT, read_stream_lines

DETECTOR = {"name": "rfdetr", "model": "RFDETRMedium", "min_score": 0.5, "antialias": False}
CALIBRATION = {"name": "pnlcalib", "kp_threshold": 0.3434, "line_threshold": 0.7867, "pnl_refine": True}
CAMERA = {"homography_pitch_to_image": [[10.0, 0.0, 640.0], [0.0, 10.0, 360.0], [0.0, 0.0, 1.0]], "pitch_origin": "centre"}
PEOPLE = {"labels": ["person"], "rows": [[100.0, 200.0, 120.0, 260.0, 0.9, 0]]}


def header(tool: str, n_frames: int = 5, clip: str = "CLIP-A", video: str | None = None) -> str:
    settings = DETECTOR if tool == "detector" else CALIBRATION
    return json.dumps(
        {
            "format": TOOL_FEED_FORMAT,
            "clip": clip,
            "n_frames": n_frames,
            "video_sha256": video,
            "job_sha256": "0" * 64,
            "tool": settings,
        }
    )


def line(tool: str, frame: int, seconds: float = 0.02, arrived_at: float = 1000.0) -> str:
    payload = PEOPLE if tool == "detector" else CAMERA
    return json.dumps({"frame": frame, "seconds": seconds, "arrived_at": arrived_at, "payload": payload})


def skipped(frame: int) -> str:
    return json.dumps({"frame": frame, "arrived_at": 1000.0, "skipped": True})


def started(n_frames: int = 5, wait_s: float = 1.0) -> FeedMerger:
    merger = FeedMerger(wait_s)
    merger.header("detector", header("detector", n_frames), 0.0)
    merger.header("calibration", header("calibration", n_frames), 0.0)
    return merger


def frames_of(out: list[str]) -> list[dict[str, Any]]:
    return [json.loads(text) for text in out if '"frame"' in text]


def test_both_parts_in_order_give_one_line_per_frame_that_the_stream_reads() -> None:
    merger = started(3)
    out = merger.take(0.0)
    for frame in range(3):
        merger.line("detector", line("detector", frame), 0.1)
        merger.line("calibration", line("calibration", frame), 0.1)
        out += merger.take(0.1)
    assert merger.finished and merger.counts.problems() == 0
    assert json.loads(out[0])["format"] == STREAM_FORMAT
    stream_header, inputs = read_stream_lines(out)
    items = list(inputs)
    assert (stream_header.clip, stream_header.n_frames, stream_header.video_sha256) == ("CLIP-A", 3, None)
    assert [item.frame for item in items] == [0, 1, 2]
    assert all(len(item.people) == 1 and item.camera.valid and item.missing == () for item in items)
    assert items[0].arrived_at == 1000.0 and items[0].detector_seconds == 0.02


def test_a_slower_tool_holds_a_frame_for_at_most_the_wait_from_the_first_part() -> None:
    merger = started(3, wait_s=1.0)
    merger.take(0.0)
    merger.line("detector", line("detector", 0), 10.0)
    assert frames_of(merger.take(10.5)) == []
    assert merger.deadline() == 11.0
    merger.line("calibration", line("calibration", 0), 10.8)  # in time: the frame goes out whole
    assert [frame["calibration"].get("missing") for frame in frames_of(merger.take(10.8))] == [None]
    merger.line("detector", line("detector", 1), 12.0)
    merger.line("detector", line("detector", 2), 12.1)
    assert frames_of(merger.take(12.9)) == []
    # Frame 1 waited from 12.0, frame 2 from 12.1: each clock runs on its own, so a stall costs one wait.
    late = frames_of(merger.take(13.05))
    assert [(frame["frame"], frame["calibration"]) for frame in late] == [(1, {"missing": "late"})]
    late = frames_of(merger.take(13.1))
    assert [(frame["frame"], frame["calibration"]) for frame in late] == [(2, {"missing": "late"})]
    assert merger.counts.late == {"detector": 0, "calibration": 2}
    # The part of frame 1 that comes after its frame went out is dropped and counted.
    merger.line("calibration", line("calibration", 1), 13.2)
    assert merger.counts.late_lines["calibration"] == 1 and merger.finished


def test_a_tool_that_dies_stops_the_wait_and_its_parts_are_marked_ended() -> None:
    merger = started(4)
    merger.take(0.0)
    merger.line("detector", line("detector", 0), 0.1)
    merger.line("calibration", line("calibration", 0), 0.1)
    merger.line("calibration", line("calibration", 1), 0.2)
    merger.end("detector")  # its pipe closed after frame 0
    merger.line("calibration", line("calibration", 2), 0.3)
    out = frames_of(merger.take(0.3))
    assert [frame["frame"] for frame in out] == [0, 1, 2]
    assert out[1]["detector"] == {"missing": "ended"} and "payload" in out[1]["calibration"]
    merger.end("calibration")  # both ended: frame 3 was sent by neither, so the stream ends early
    assert merger.take(0.4) == [] and merger.finished and merger.next_frame == 3
    assert merger.counts.ended == {"detector": 2, "calibration": 0}


def test_lines_out_of_order_wait_for_their_frame_and_a_repeated_line_is_dropped() -> None:
    merger = started(3)
    merger.take(0.0)
    for frame in (2, 0, 1):
        merger.line("detector", line("detector", frame), 0.1)
    for frame in (1, 2, 0):
        merger.line("calibration", line("calibration", frame), 0.1)
    merger.line("calibration", line("calibration", 2), 0.1)  # a second line of frame 2
    assert [frame["frame"] for frame in frames_of(merger.take(0.1))] == [0, 1, 2]
    assert merger.counts.repeated_lines == {"detector": 0, "calibration": 1}
    assert merger.counts.late == {"detector": 0, "calibration": 0}


@pytest.mark.parametrize(
    "text",
    [
        '{"frame": 0, "seconds": 0.02, "arrived_at": 1000.0, "payl',  # cut in the middle
        "not json",
        json.dumps({"frame": 0, "seconds": -1.0, "arrived_at": 1000.0, "payload": CAMERA}),
        json.dumps({"frame": 9, "seconds": 0.02, "arrived_at": 1000.0, "payload": CAMERA}),  # outside the clip
        json.dumps({"frame": 0, "seconds": 0.02, "payload": CAMERA}),  # no arrival time
        '{"frame": 0, "seconds": NaN, "arrived_at": 1000.0, "payload": {}}',
    ],
)
def test_a_broken_line_is_dropped_and_its_frame_waits_as_if_it_had_not_come(text: str) -> None:
    merger = started(3)
    merger.take(0.0)
    merger.line("calibration", text, 0.0)
    merger.line("detector", line("detector", 0), 0.0)
    assert frames_of(merger.take(0.5)) == []
    assert [frame["calibration"] for frame in frames_of(merger.take(1.0))] == [{"missing": "late"}]
    assert merger.counts.broken_lines["calibration"] == 1


def test_a_frame_that_one_tool_never_sends_goes_out_after_the_wait_and_the_later_frames_follow() -> None:
    merger = started(3)
    merger.take(0.0)
    for frame in range(3):
        merger.line("detector", line("detector", frame), 0.0)
        if frame != 1:
            merger.line("calibration", line("calibration", frame), 0.0)
    out = frames_of(merger.take(0.0))
    assert [frame["frame"] for frame in out] == [0]
    out = frames_of(merger.take(1.0))
    assert [(frame["frame"], frame["calibration"].get("missing")) for frame in out] == [(1, "late"), (2, None)]


def test_a_frame_that_both_tools_skip_waits_from_the_first_line_of_a_later_frame() -> None:
    merger = started(3)
    merger.take(0.0)
    merger.line("detector", line("detector", 0), 0.0)
    merger.line("calibration", line("calibration", 0), 0.0)
    merger.take(0.0)
    merger.line("detector", line("detector", 2), 5.0)  # nothing of frame 1 from either tool
    assert merger.deadline() == 6.0
    out = frames_of(merger.take(6.0))
    # Frame 2 waited from the same line: its calibration part is late too.
    assert [frame["frame"] for frame in out] == [1, 2]
    assert out[0]["detector"] == out[0]["calibration"] == out[1]["calibration"] == {"missing": "late"}


def test_a_skipped_frame_does_not_wait_and_reads_as_a_frame_without_a_camera() -> None:
    merger = started(2)
    out = merger.take(0.0)
    merger.line("detector", line("detector", 0), 0.0)
    merger.line("calibration", skipped(0), 0.0)
    out += merger.take(0.0)
    assert frames_of(out)[0]["calibration"] == {"missing": "skipped"}
    _, inputs = read_stream_lines(out)
    item = next(inputs)
    assert not item.camera.valid and item.missing == (("calibration", "skipped"),) and len(item.people) == 1


def test_the_wait_starts_only_when_both_headers_are_in() -> None:
    merger = FeedMerger(1.0)
    merger.header("detector", header("detector", 2), 0.0)
    merger.line("detector", line("detector", 0), 0.0)
    assert merger.take(30.0) == [] and merger.deadline() is None  # the calibration still loads its model
    merger.header("calibration", header("calibration", 2), 40.0)
    assert frames_of(merger.take(40.5)) == []
    assert [frame["calibration"] for frame in frames_of(merger.take(41.0))] == [{"missing": "late"}]


@pytest.mark.parametrize(
    ("first", "second", "words"),
    [
        (header("detector"), header("calibration", clip="CLIP-B"), "different values of 'clip'"),
        (header("detector"), header("calibration", n_frames=6), "different values of 'n_frames'"),
        (header("detector", video="a" * 64), header("calibration"), "different values of 'video_sha256'"),
        (header("calibration"), header("calibration"), "is not a feed of the detector"),
        (json.dumps({"format": "other"}), header("calibration"), "does not start with a header"),
    ],
)
def test_headers_that_do_not_fit_together_stop_the_merger_before_any_line(first: str, second: str, words: str) -> None:
    merger = FeedMerger(1.0)
    with pytest.raises(FeedError, match=words):
        merger.header("detector", first, 0.0)
        merger.header("calibration", second, 0.0)
        merger.take(0.0)


def test_a_feed_that_ends_before_its_header_stops_the_merger() -> None:
    merger = FeedMerger(1.0)
    merger.header("detector", header("detector"), 0.0)
    with pytest.raises(FeedError, match="ended before its header"):
        merger.end("calibration")


# --- Threads and pipes ----------------------------------------------------------------------------


def pipe_writer(lines: list[str], delays: list[float] | None = None, *, cut: str | None = None) -> int:
    """Return the read end of a pipe that a thread fills with the lines (each after its delay), then closes."""
    read_end, write_end = os.pipe()

    def write() -> None:
        with os.fdopen(write_end, "w", encoding="utf-8") as handle:
            for position, text in enumerate(lines):
                if delays:
                    time.sleep(delays[position])
                handle.write(text + "\n")
                handle.flush()
            if cut is not None:
                handle.write(cut)

    threading.Thread(target=write, daemon=True).start()
    return read_end


def test_merge_feeds_on_two_pipes_with_a_cut_off_last_line_and_a_slow_tool() -> None:
    n_frames = 4
    detector = [header("detector", n_frames), *(line("detector", frame) for frame in range(n_frames))]
    calibration = [header("calibration", n_frames), *(line("calibration", frame) for frame in range(n_frames - 1))]
    slow = [0.0, *([0.05] * (n_frames - 1))]
    out: list[str] = []
    merger = merge_feeds(
        {
            "detector": fd_lines(pipe_writer(detector)),
            "calibration": fd_lines(pipe_writer(calibration, slow, cut='{"frame": 3, "secon')),
        },
        out.append,
        wait_s=0.5,
    )
    frames = frames_of(out)
    assert [frame["frame"] for frame in frames] == [0, 1, 2, 3]
    assert all("payload" in frame["calibration"] for frame in frames[:3])
    assert frames[3]["calibration"] == {"missing": "ended"}
    assert merger.counts.cut_off_lines["calibration"] == 1 and merger.counts.late["calibration"] == 0


def test_a_slow_reader_holds_the_feeds_back_and_nothing_is_dropped() -> None:
    """With a queue of 2 lines per feed and a writer of the merged lines that takes 20 ms per line, every
    frame goes out whole: the reading of a full feed waits, and no part is declared late for it."""
    n_frames = 30
    feeds = {
        tool: fd_lines(pipe_writer([header(tool, n_frames), *(line(tool, frame) for frame in range(n_frames))]))
        for tool in ("detector", "calibration")
    }
    out: list[str] = []

    def slow_write(text: str) -> None:
        time.sleep(0.02)
        out.append(text)

    merger = merge_feeds(feeds, slow_write, wait_s=0.05, queue_lines=2)
    assert len(frames_of(out)) == n_frames and merger.counts.problems() == 0


def test_fd_lines_keeps_the_line_ends_and_gives_a_last_line_without_one() -> None:
    read_end = pipe_writer(["a", "b"], cut="c")
    assert list(fd_lines(read_end, chunk=1)) == ["a\n", "b\n", "c"]


# --- Second read of this work (2026-11-23) ------------------------------------------------------------


def test_a_slow_reader_does_not_make_a_part_late_that_came_while_a_line_was_written() -> None:
    """The detector sends a line every 30 ms, the reader of the merged lines takes 200 ms per line: every
    detector line comes long before its frame is written. Before the fix 6 to 10 of 12 parts went out late."""
    n_frames = 12
    feeds = {
        "detector": fd_lines(
            pipe_writer([header("detector", n_frames), *(line("detector", f) for f in range(n_frames))], [0.03] * 13)
        ),
        "calibration": fd_lines(
            pipe_writer([header("calibration", n_frames), *(line("calibration", f) for f in range(n_frames))])
        ),
    }

    def slow(text: str) -> None:
        time.sleep(0.2)

    merger = merge_feeds(feeds, slow, wait_s=0.1)
    assert merger.next_frame == n_frames and merger.counts.problems() == 0


def test_the_merger_reads_a_slow_tool_to_its_end_so_that_its_pipe_does_not_break() -> None:
    """Every frame went out without the slow calibration; the merger still reads its late lines to the end.
    Before the fix it returned, and the next write of the tool broke its pipe."""
    n_frames = 5
    broke: list[BaseException] = []
    read_end, write_end = os.pipe()

    def slow_tool() -> None:
        try:
            with os.fdopen(write_end, "w", encoding="utf-8") as handle:
                for text in [header("calibration", n_frames), *(line("calibration", f) for f in range(n_frames))]:
                    handle.write(text + "\n")
                    handle.flush()
                    time.sleep(0.1)
        except BrokenPipeError as error:
            broke.append(error)

    writer = threading.Thread(target=slow_tool)
    writer.start()
    detector = [header("detector", n_frames), *(line("detector", f) for f in range(n_frames))]
    merger = merge_feeds(
        {"detector": fd_lines(pipe_writer(detector)), "calibration": fd_lines(read_end)}, lambda text: None, wait_s=0.01
    )
    writer.join(10.0)
    assert broke == [] and merger.next_frame == n_frames
    assert merger.counts.late["calibration"] >= 1 and merger.counts.late_lines["calibration"] >= 1


def test_a_tool_a_little_slower_than_the_pace_loses_every_part_once_its_lag_passes_the_wait() -> None:
    """Second read of round 7: the wait does not bound the time of a tool per frame but its lag. A calibration of 60 ms
    per frame at 25 frames per second falls 20 ms further behind with each frame, so from about frame 1.0 / 0.020 = 50
    on its parts come after the wait of 1 s, and they never catch up (the advice is --skip-behind, RUNBOOK section 14)."""
    n_frames = 200
    merger = started(n_frames, wait_s=1.0)
    events = sorted(
        [(frame / 25 + 0.02, "detector", frame) for frame in range(n_frames)]
        + [((frame + 1) * 0.060, "calibration", frame) for frame in range(n_frames)]
    )
    late_from = None
    for at, tool, frame in events:
        for text in merger.take(at):
            part = json.loads(text).get("calibration")
            if part == {"missing": "late"} and late_from is None:
                late_from = json.loads(text)["frame"]
        merger.line(tool, line(tool, frame), at)
    merger.end("detector")
    merger.end("calibration")
    merger.take(1e9)
    assert late_from is not None and 45 <= late_from <= 55
    assert merger.counts.late["calibration"] == n_frames - late_from
    assert merger.counts.late_lines["calibration"] == n_frames - late_from


def test_a_tool_that_hangs_after_the_last_frame_does_not_hold_the_merger() -> None:
    """Planted-bug check of round 7: a tool that hung (its pipe open, nothing more written) after every frame had gone
    out kept merge_feeds reading it for ever, and the merged lines never ended. Now the lines end with the last frame
    (``end``), and a feed silent for ``drain_s`` is left and counted."""
    n_frames = 4
    hang = threading.Event()

    def calibration() -> Any:
        yield header("calibration", n_frames) + "\n"
        for frame in range(n_frames):
            yield line("calibration", frame) + "\n"
        hang.wait()  # alive, nothing more written

    detector = [header("detector", n_frames) + "\n", *(line("detector", frame) + "\n" for frame in range(n_frames))]
    out: list[str] = []
    ended: list[int] = []
    done: dict[str, Any] = {}

    def run() -> None:
        done["merger"] = merge_feeds(
            {"detector": iter(detector), "calibration": calibration()},
            out.append,
            wait_s=0.2,
            end=lambda: ended.append(len(out)),
            drain_s=0.3,
        )

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(10.0)
    hang.set()
    assert not worker.is_alive()
    assert ended == [1 + n_frames]  # the end came right after the header and the last frame
    assert done["merger"].counts.left_open == {"detector": 0, "calibration": 1}


def test_the_stream_reads_no_line_after_the_last_frame() -> None:
    """Planted-bug check of round 7: the stream read on after the last frame, so a pipe that stayed open held it."""
    head = json.dumps({"format": STREAM_FORMAT, "clip": "CLIP-A", "n_frames": 2, "video_sha256": None,
                       "detector": DETECTOR, "calibration": CALIBRATION})

    def lines() -> Any:
        yield head
        for frame in range(2):
            yield json.dumps({"frame": frame, "detector": {"seconds": 0.02, "payload": PEOPLE},
                              "calibration": {"seconds": 0.03, "payload": CAMERA}})
        raise AssertionError("a line after the last frame was asked for")

    _, inputs = read_stream_lines(lines())
    assert [item.frame for item in inputs] == [0, 1]
