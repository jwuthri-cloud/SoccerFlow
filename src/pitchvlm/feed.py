"""The live feeds of the two tools, merged by frame into the lines of a stream.

The detector and the calibration run in two environments, so in two processes
(``tools/run_tool.py``, rule 4: the package imports no tool). In a live run each one writes a
tool feed (``TOOL_FEED_FORMAT``), one line per frame as soon as the frame is answered, and the
merger here zips the two feeds by frame into the lines of ``stream.STREAM_FORMAT`` that
``pitchvlm stream --lines -`` reads.

A tool feed: a header line ``{"format": TOOL_FEED_FORMAT, "clip", "n_frames", "video_sha256",
"job_sha256", "tool": <the settings of the job>}``, then one line per frame,
``{"frame": i, "seconds": s, "arrived_at": t, "payload": <the payload of a clip run frame>}``,
or ``{"frame": i, "arrived_at": t, "skipped": true}`` for a frame that the tool left out because
it was behind (``run_tool.py --skip-behind``). ``arrived_at`` is the time, in seconds since the
epoch, at which the frame came to the tool.

The policy of the merger (``FeedMerger``), written down in ``docs/DECISIONS.md`` (entry "The
live feed of the detector engine", 2026-11-23):

1. The header of the stream is written once both headers are in. They must name the same clip,
   the same number of frames and the same video, and a detector and the calibration; otherwise
   the merger stops (``FeedError``) before it writes anything.
2. Frames go out in frame order, each once. Frame ``n`` goes out as soon as both parts are in.
3. The wait is bounded. The clock of frame ``n`` starts when a line of frame ``n`` or of a later
   frame first comes in from either tool (and not before both headers are in). When ``wait_s``
   has passed on that clock, frame ``n`` goes out with each part that is not in marked
   ``{"missing": "late"}``. The clock of each frame starts on its own, so a tool that stalls
   delays the stream by at most ``wait_s``, not by ``wait_s`` per frame.
4. A part that the tool skipped is ``{"missing": "skipped"}``, and the frame does not wait for
   it. After the feed of a tool has ended (its pipe closed: it finished or it died), its parts
   are ``{"missing": "ended"}`` and no frame waits for it. When both feeds have ended, the frames
   after the last one that either tool sent are not written: the stream then ends early, and
   ``stream`` says so.
5. A line can come out of frame order: it waits in the merger until its frame goes out. A line
   of a frame that has gone out already is dropped and counted (``late lines``); so is a second
   line of a frame (``repeated lines``). A line that is not a JSON object of a frame line, or
   that names a frame outside the clip, is dropped and counted (``broken lines``): its frame
   waits as if the line had not come. A last line without its line end (a tool that died while
   it wrote) is dropped and counted (``cut-off lines``).
6. The merger holds the parts of the frames that have not gone out. Its reading is bounded:
   each feed has a queue of ``queue_lines`` lines, and a full queue stops the reading of that
   pipe, so a slow reader of the merged lines slows the tools down (back-pressure) and nothing
   is dropped for it. Frames are decided one at a time, and the lines that came while a line was
   written count before the next frame is decided: a part is late only when its tool had not
   written it by the time its frame goes out.
7. After the last frame went out, the merged lines end (``end``), so their reader finishes at
   once. A feed that is still open is then read to its end (its lines are late and counted), so
   that the pipe of a slow tool does not break under it and its copy at ``--out`` is whole; but
   a feed that sends nothing for ``drain_s`` seconds is left (``left_open``): a tool that hangs
   does not hold the merger for ever.

No model is called and nothing is paid here.
"""

from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import TypeAdapter, ValidationError

from pitchvlm.fallbacks.runs import DetectorSettings, PnlcalibSettings
from pitchvlm.stream import STREAM_TOOLS, stream_frame_line, stream_header_line

#: Format name of the feed of one tool in a live run (``tools/run_tool.py --feed``).
TOOL_FEED_FORMAT = "pitchvlm-tool-feed-1"
#: Default bound of the wait for the part of the other tool, seconds.
DEFAULT_WAIT_S = 1.0
#: Default number of lines that the merger reads ahead from each feed.
DEFAULT_QUEUE_LINES = 250
#: After the last frame, a feed that sends nothing for this many seconds is no longer read (policy 7).
DEFAULT_DRAIN_S = 10.0

_HEADER_KEYS = frozenset({"format", "clip", "n_frames", "video_sha256", "job_sha256", "tool"})
_DETECTOR = TypeAdapter(DetectorSettings)


class FeedError(ValueError):
    """The feeds cannot be merged: a header is wrong, or the two headers do not fit together."""


@dataclass(frozen=True)
class _Answer:
    """The part of one tool for one frame, as it goes into the merged line."""

    part: dict[str, Any]


@dataclass
class MergeCounts:
    """What the merger dropped or marked, by tool."""

    late: dict[str, int] = field(default_factory=lambda: dict.fromkeys(STREAM_TOOLS, 0))
    skipped: dict[str, int] = field(default_factory=lambda: dict.fromkeys(STREAM_TOOLS, 0))
    ended: dict[str, int] = field(default_factory=lambda: dict.fromkeys(STREAM_TOOLS, 0))
    late_lines: dict[str, int] = field(default_factory=lambda: dict.fromkeys(STREAM_TOOLS, 0))
    repeated_lines: dict[str, int] = field(default_factory=lambda: dict.fromkeys(STREAM_TOOLS, 0))
    broken_lines: dict[str, int] = field(default_factory=lambda: dict.fromkeys(STREAM_TOOLS, 0))
    cut_off_lines: dict[str, int] = field(default_factory=lambda: dict.fromkeys(STREAM_TOOLS, 0))
    #: Feeds still open, and silent for the drain time, when the merger stopped reading after the last frame.
    left_open: dict[str, int] = field(default_factory=lambda: dict.fromkeys(STREAM_TOOLS, 0))

    def as_dict(self) -> dict[str, dict[str, int]]:
        return {name: dict(values) for name, values in vars(self).items()}

    def problems(self) -> int:
        """Return the number of parts that went out without an answer, and of lines that were dropped."""
        return sum(sum(values.values()) for values in vars(self).values())


class FeedMerger:
    """The policy of the merger (module docstring), one event at a time; the caller gives the time of each event.

    ``header``, ``line``, ``cut_off`` and ``end`` take what came in from the feed of a tool, at the
    time ``now`` (seconds, any clock that does not go back). ``take(now)`` returns the lines
    that can go out now, in order; ``deadline()`` says when ``take`` must be called again if
    nothing else comes in. ``finished`` is true when every line that will ever go out has gone.
    """

    def __init__(self, wait_s: float = DEFAULT_WAIT_S) -> None:
        if not 0.0 <= wait_s < math.inf:
            raise ValueError(f"the wait must be a number of 0 or more seconds, got {wait_s}")
        self.wait_s = wait_s
        self._headers: dict[str, dict[str, Any]] = {}
        self._headers_at: float | None = None
        self._header_written = False
        self._parts: dict[str, dict[int, _Answer]] = {tool: {} for tool in STREAM_TOOLS}
        self._skipped: dict[str, set[int]] = {tool: set() for tool in STREAM_TOOLS}
        self._ended: set[str] = set()
        #: The newest frame that either tool sent a line of, and the time when each frame's clock started.
        self._newest = -1
        self._clock: dict[int, float] = {}
        self.n_frames: int | None = None
        self.next_frame = 0
        self.counts = MergeCounts()

    # --- What comes in --------------------------------------------------------------------------

    def header(self, tool: str, text: str, now: float) -> None:
        """Take the first line of the feed of ``tool``. A wrong header raises ``FeedError``."""
        _check_tool(tool)
        if tool in self._headers:
            raise FeedError(f"the feed of the {tool} has a second header")
        data = _json_object(text)
        if data is None or data.get("format") != TOOL_FEED_FORMAT:
            raise FeedError(f"the feed of the {tool} does not start with a header of the format {TOOL_FEED_FORMAT}")
        if set(data) != _HEADER_KEYS:
            raise FeedError(f"the header of the {tool} must have the keys {sorted(_HEADER_KEYS)}, it has {sorted(data)}")
        try:
            if tool == "detector":
                _DETECTOR.validate_python(data["tool"])
            else:
                PnlcalibSettings.model_validate(data["tool"])
        except ValidationError as error:
            raise FeedError(f"the feed of the {tool} is not a feed of the {tool}: {error}") from error
        n_frames = data["n_frames"]
        if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames < 1:
            raise FeedError(f"the header of the {tool}: 'n_frames' must be a whole number from 1, got {n_frames!r}")
        self._headers[tool] = data
        if len(self._headers) == len(STREAM_TOOLS):
            first, second = (self._headers[name] for name in STREAM_TOOLS)
            for key in ("clip", "n_frames", "video_sha256"):
                if first[key] != second[key]:
                    raise FeedError(f"the two feeds name different values of {key!r}: {first[key]!r} and {second[key]!r}")
            self.n_frames = n_frames
            self._headers_at = now

    def line(self, tool: str, text: str, now: float) -> None:
        """Take a frame line of the feed of ``tool`` (after its header)."""
        _check_tool(tool)
        if tool not in self._headers:
            raise FeedError(f"the feed of the {tool} has a frame line before its header")
        n_frames = int(self._headers[tool]["n_frames"])
        data = _json_object(text)
        frame = None if data is None else data.get("frame")
        if (
            data is None
            or isinstance(frame, bool)
            or not isinstance(frame, int)
            or not 0 <= frame < n_frames
            or not _is_frame_line(data)
        ):
            self.counts.broken_lines[tool] += 1
            return
        if frame < self.next_frame:
            self.counts.late_lines[tool] += 1
            return
        if frame in self._parts[tool] or frame in self._skipped[tool]:
            self.counts.repeated_lines[tool] += 1
            return
        if data.get("skipped") is True:
            self._skipped[tool].add(frame)
        else:
            part = {"seconds": data["seconds"], "payload": data["payload"], "arrived_at": data["arrived_at"]}
            self._parts[tool][frame] = _Answer(part)
        for later in range(self._newest + 1, frame + 1):
            self._clock[later] = now
        self._newest = max(self._newest, frame)

    def cut_off(self, tool: str) -> None:
        """Count a last line of a feed that has no line end: it is dropped."""
        _check_tool(tool)
        self.counts.cut_off_lines[tool] += 1

    def end(self, tool: str) -> None:
        """Take the end of the feed of ``tool``: its pipe closed. A feed that ends before its header raises ``FeedError``."""
        _check_tool(tool)
        if tool not in self._headers:
            raise FeedError(f"the feed of the {tool} ended before its header")
        self._ended.add(tool)

    @property
    def ended(self) -> frozenset[str]:
        """The tools whose feed has ended."""
        return frozenset(self._ended)

    # --- What goes out --------------------------------------------------------------------------

    @property
    def finished(self) -> bool:
        """Whether nothing more will go out: every frame went out, or both feeds ended and the rest has no line."""
        if self.n_frames is None:
            return False
        if self.next_frame >= self.n_frames:
            return True
        return len(self._ended) == len(STREAM_TOOLS) and self.next_frame > self._newest

    def deadline(self) -> float | None:
        """Return the time at which the next frame goes out without a missing part, or None (it waits for a line)."""
        if self.n_frames is None or self.finished or self.next_frame not in self._clock or self._headers_at is None:
            return None
        return max(self._clock[self.next_frame], self._headers_at) + self.wait_s

    def take(self, now: float, limit: int | None = None) -> list[str]:
        """Return the lines that go out now, in order: the header of the stream first, then at most ``limit`` frames.

        A part is marked late only when its frame goes out: ``merge_feeds`` takes one frame at a
        time and applies the lines that came meanwhile before the next one.
        """
        out: list[str] = []
        if self.n_frames is None:
            return out
        if not self._header_written:
            detector, calibration = (self._headers[tool] for tool in STREAM_TOOLS)
            out.append(
                stream_header_line(
                    detector["clip"], self.n_frames, detector["video_sha256"], detector["tool"], calibration["tool"]
                )
            )
            self._header_written = True
        taken = 0
        while not self.finished and (limit is None or taken < limit):
            frame = self.next_frame
            waiting = [tool for tool in STREAM_TOOLS if self._waits_for(tool, frame)]
            if waiting:
                deadline = self.deadline()
                if deadline is None or now < deadline:
                    break
            out.append(stream_frame_line(frame, *(self._part_of(tool, frame) for tool in STREAM_TOOLS)))
            self._clock.pop(frame, None)
            self.next_frame += 1
            taken += 1
        return out

    def _waits_for(self, tool: str, frame: int) -> bool:
        return frame not in self._parts[tool] and frame not in self._skipped[tool] and tool not in self._ended

    def _part_of(self, tool: str, frame: int) -> dict[str, Any]:
        answer = self._parts[tool].pop(frame, None)
        if answer is not None:
            return answer.part
        if frame in self._skipped[tool]:
            self._skipped[tool].discard(frame)
            self.counts.skipped[tool] += 1
            return {"missing": "skipped"}
        if tool in self._ended:
            self.counts.ended[tool] += 1
            return {"missing": "ended"}
        self.counts.late[tool] += 1
        return {"missing": "late"}


def _check_tool(tool: str) -> None:
    if tool not in STREAM_TOOLS:
        raise ValueError(f"the tool must be one of {', '.join(STREAM_TOOLS)}, got {tool!r}")


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text, parse_constant=_refuse)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _refuse(token: str) -> Any:
    raise ValueError(token)


def _is_time(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and 0.0 <= value < math.inf


def _is_frame_line(data: Mapping[str, Any]) -> bool:
    """Tell whether a JSON object is a frame line of a tool feed (module docstring)."""
    if data.get("skipped") is True:
        return set(data) == {"frame", "arrived_at", "skipped"} and _is_time(data["arrived_at"])
    return (
        set(data) == {"frame", "seconds", "arrived_at", "payload"}
        and _is_time(data["seconds"])
        and _is_time(data["arrived_at"])
        and isinstance(data["payload"], dict)
    )


# --- Reading two pipes ------------------------------------------------------------------------------


_END = object()


def _read_feed(
    tool: str,
    handle: Iterable[str],
    events: queue.Queue[tuple[str, Any, float]],
    room: threading.Semaphore,
    clock: Callable[[], float],
) -> None:
    """Put each line of a feed on the queue with the time it was read; a held slot of ``room`` per line."""
    try:
        for text in handle:
            room.acquire()
            events.put((tool, text, clock()))
    except Exception as error:  # an error of reading goes to the merger, which raises it
        events.put((tool, error, clock()))
        return
    events.put((tool, _END, clock()))


def merge_feeds(
    feeds: Mapping[str, Iterable[str]],
    write: Callable[[str], None],
    *,
    wait_s: float = DEFAULT_WAIT_S,
    queue_lines: int = DEFAULT_QUEUE_LINES,
    clock: Callable[[], float] = time.monotonic,
    end: Callable[[], None] | None = None,
    drain_s: float = DEFAULT_DRAIN_S,
) -> FeedMerger:
    """Read the feed of each tool on its own thread, merge them (``FeedMerger``) and ``write`` each merged line.

    ``feeds`` maps ``"detector"`` and ``"calibration"`` to their lines (an open pipe or file).
    ``write`` gets each line without its line end, as soon as it can go out; ``end`` is called
    once the last line went out, before the feeds that are still open are read to their end
    (policy 7). Returns the merger (its ``counts``, ``next_frame`` and ``n_frames``). A wrong
    header raises ``FeedError``; an error of reading a feed is raised as it is.
    """
    if set(feeds) != set(STREAM_TOOLS):
        raise ValueError(f"the feeds must be {', '.join(STREAM_TOOLS)}")
    if queue_lines < 1:
        raise ValueError(f"the queue must hold at least 1 line, got {queue_lines}")
    merger = FeedMerger(wait_s)
    events: queue.Queue[tuple[str, Any, float]] = queue.Queue()
    rooms = {tool: threading.Semaphore(queue_lines) for tool in STREAM_TOOLS}
    for tool in STREAM_TOOLS:
        thread = threading.Thread(
            target=_read_feed,
            args=(tool, feeds[tool], events, rooms[tool], clock),
            name=f"pitchvlm-feed-{tool}",
            daemon=True,
        )
        thread.start()
    started: set[str] = set()

    def apply_waiting(event: tuple[str, Any, float] | None) -> None:
        """Apply ``event`` (when given) and every event that waits in the queue now."""
        while True:
            if event is None:
                try:
                    event = events.get_nowait()
                except queue.Empty:
                    return
            _apply(merger, event, started, rooms)
            event = None

    while not merger.finished:
        deadline = merger.deadline()
        timeout = None if deadline is None else max(0.0, deadline - clock())
        try:
            event: tuple[str, Any, float] | None = events.get(timeout=timeout)
        except queue.Empty:
            event = None
        apply_waiting(event)
        # One frame at a time: a write can wait for a slow reader, and the lines that come meanwhile
        # count before the next frame is decided (policy 6: nothing is late because of the reader).
        while True:
            lines = merger.take(clock(), limit=1)
            if not lines:
                break
            for text in lines:
                write(text)
            apply_waiting(None)
    if end is not None:
        end()
    # Every frame went out, or both feeds ended. A tool that is still writing is read to its end, so
    # that its pipe does not break under it: its lines are late and counted (policy 5). A tool that
    # sends nothing for ``drain_s`` is left (policy 7).
    while len(merger.ended) < len(STREAM_TOOLS):
        try:
            event = events.get(timeout=drain_s)
        except queue.Empty:
            for tool in STREAM_TOOLS:
                if tool not in merger.ended:
                    merger.counts.left_open[tool] += 1
            break
        apply_waiting(event)
    return merger


def _apply(
    merger: FeedMerger, event: tuple[str, Any, float], started: set[str], rooms: Mapping[str, threading.Semaphore]
) -> None:
    tool, item, at = event
    if item is _END:
        merger.end(tool)
        return
    if isinstance(item, BaseException):
        raise item
    rooms[tool].release()
    if not item.endswith("\n"):
        merger.cut_off(tool)
        return
    if not item.strip():
        return
    if tool not in started:
        started.add(tool)
        merger.header(tool, item, at)
    else:
        merger.line(tool, item, at)


def fd_lines(fd: int, chunk: int = 65536) -> Iterator[str]:
    """Yield the lines of an open file descriptor (a pipe or a file) one by one, with their line ends, as UTF-8 text.

    A last line without a line end is yielded as it is (``merge_feeds`` counts it as cut off).
    The reading uses ``os.read`` and no buffered file object: a thread that waits in it holds
    no lock of Python, so a process whose reader stopped early can still end while the thread
    waits for a writer.
    """
    rest = b""
    while True:
        data = os.read(fd, chunk)
        if not data:
            break
        rest += data
        *lines, rest = rest.split(b"\n")
        for line in lines:
            yield line.decode("utf-8", errors="replace") + "\n"
    if rest:
        yield rest.decode("utf-8", errors="replace")
