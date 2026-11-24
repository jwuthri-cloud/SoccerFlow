"""The live output of the streaming path: one JSON line per decided frame, written as it is decided.

``stream --live-out FILE|-`` gives each state of ``stream.DetectorStream`` to ``LiveOut.given``
at the moment it is decided (the hook ``DetectorStream.on_given``). The lines are text: no
pixel, no image, no label value (rule 2 of ``CLAUDE.md``). The format (``LIVE_FORMAT``):

- a header line ``{"format", "clip", "n_frames", "fps", "width", "height", "delay_frames",
  "history_frames", "pitch_m", "policy"}`` (``header_document``);
- one line per decided frame, in frame order (``frame_document``): ``"frame"``, its time
  ``"t_s"``, ``"decided_at"`` (the newest frame when it was decided), ``"delay_frames"``,
  ``"decided_unix"`` (the wall clock when it was decided, seconds since the epoch), the flags
  ``"camera"`` (``landmarks``/``propagated``/``none``: the source of the camera of the frame),
  ``"transition"`` (the frame lies in a dissolve known when it was decided) and ``"cut"`` (a
  hard cut starts a new shot at the frame), ``"dropped"`` (lines left out before this one), the
  ``"players"`` and the ``"ball"``;
- an end line ``{"end": true, "frames", "written", "dropped"}`` when the stream ends.

With the live number step (``livenumbers``) the header names the format ``LIVE_FORMAT_NUMBERS``
(``pitchvlm-live-2``): the same lines, plus a number line ``{"kind": "number", "id", "number",
"export", "display", "frames": [first, last], "crops", "reason", "decided_at", "decided_unix"}``
for each number that the model read (``number_document``), after the frame lines of the frames
that it was read from. A number line changes no frame line; the pipe policy never leaves one out.
Without the number step the output is ``pitchvlm-live-1``, as before.

A frame line is final: the stream gives each state once and never changes it, so the line of
frame ``f`` holds the state of frame ``f`` of the result file, value for value
(``tests/test_liveout.py``). Only two things of the result file can differ from the lines, and
neither is a state: the list of tracks (``ClipResult.tracks``) takes the role and team of each id
from its last state, and the cuts of the result (``ClipResult.cuts``) hold a dissolve that was
known only after some of its frames were decided (their flag ``transition`` stays false).

Policy (``docs/DECISIONS.md``, entry "A live output and a radar viewer" of 2026-11-23):

- a regular file (``policy: file``): every line is written and flushed when its frame is
  decided, in the thread of the stream. The file is always complete; an error of the disk
  stops the stream (``LiveOutError``).
- a pipe, a named pipe or a terminal (``policy: pipe``): the stream only hands the state to a
  writer thread and never waits for the reader. While the reader holds the writer (one write
  has taken longer than ``PIPE_BLOCKED_S``: the pipe is full), at most ``PIPE_PENDING_LINES``
  lines wait; when a newer frame comes, the oldest waiting line is left out and counted
  (``dropped``). So a slow reader gets the newest state, with gaps and their count, and the time
  per frame of the stream does not change. A writer that its reader does not hold takes every
  line, also from a burst of decisions (the last frames, decided together at the end of a
  clip), up to ``PIPE_MAX_LINES`` waiting lines. The header and the end line are never left out. A
  reader that goes away ends the output; the stream goes on and writes its result files.
  The pipe of the system is a second queue that no count sees: it holds 16 to 64 kB, a few
  lines of about 5 kB, so a reader that falls behind reads lines that are up to that many
  frames old. ``written`` counts the lines that went into the pipe, not those that were read;
  ``decided_unix`` lets a reader measure how old a line is.
"""

from __future__ import annotations

import json
import os
import stat
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pitchvlm.pitch import PITCH_LENGTH, PITCH_WIDTH
from pitchvlm.schemas import BallState, Box, ClipInfo, FrameState, HomographyRecord, PlayerState

#: Format name of the live output: a header line, one line per decided frame, an end line.
LIVE_FORMAT = "pitchvlm-live-1"
#: Format name of the live output with number lines (module docstring): ``LIVE_FORMAT`` plus number lines.
LIVE_FORMAT_NUMBERS = "pitchvlm-live-2"
#: The value of ``"kind"`` of a number line.
NUMBER_KIND = "number"
#: Lines that wait for the writer of a pipe that its reader holds, before the oldest one is left out: the newest state.
PIPE_PENDING_LINES = 1
#: A write of a pipe that takes longer than this, seconds, means that the reader holds the writer (the pipe is full).
#: Ten switch intervals of Python (5 ms): a writer that only waits for its turn is not held.
PIPE_BLOCKED_S = 0.050
#: Lines that wait for a writer that its reader does not hold (it is not scheduled yet), before the oldest one goes.
PIPE_MAX_LINES = 250
#: Seconds that the end of a stream waits for a slow pipe reader to take the last lines.
CLOSE_WAIT_S = 5.0


class LiveOutError(RuntimeError):
    """The live output cannot be written (a file that cannot be opened or written)."""


@dataclass(frozen=True)
class LiveReport:
    """What the live output did: frames given by the stream, lines written, frames left out, lines not delivered."""

    target: str
    policy: str
    given: int
    written: int
    dropped: int
    undelivered: int
    reader_gone: str | None

    def text(self) -> str:
        """Return the report as one line for the output of ``stream``."""
        parts = [f"{self.written} frame lines written of {self.given} decided", f"{self.dropped} left out"]
        if self.undelivered:
            parts.append(f"{self.undelivered} lines not delivered when the stream ended (frame lines and the end line)")
        if self.reader_gone is not None:
            parts.append(f"the reader went away ({self.reader_gone})")
        return f"live output {self.target} (policy {self.policy}): " + ", ".join(parts)


def header_document(
    info: ClipInfo, *, delay_frames: int | None, history_frames: int | None, policy: str, numbers: bool = False
) -> dict:
    """Return the header line of a live output (``LIVE_FORMAT_NUMBERS`` when number lines can come)."""
    return {
        "format": LIVE_FORMAT_NUMBERS if numbers else LIVE_FORMAT,
        "clip": info.clip_id,
        "n_frames": info.n_frames,
        "fps": info.fps,
        "width": info.width,
        "height": info.height,
        "delay_frames": delay_frames,
        "history_frames": history_frames,
        "pitch_m": [PITCH_LENGTH, PITCH_WIDTH],
        "policy": policy,
    }


def _box(box: Box | None) -> list[float] | None:
    return None if box is None else [box.x1, box.y1, box.x2, box.y2]


def _player(player: PlayerState) -> dict[str, Any]:
    number = player.number_display if player.number_display is not None else player.number_export
    return {
        "id": player.track_id,
        "role": player.role,
        "team": player.team_key,
        "side": player.team_side,
        "number": number,
        "x": player.pitch_x,
        "y": player.pitch_y,
        "box": _box(player.box),
        "observed": player.observed,
    }


def _ball(ball: BallState | None) -> dict[str, Any] | None:
    if ball is None:
        return None
    return {"x": ball.pitch_x, "y": ball.pitch_y, "box": _box(ball.box), "observed": ball.observed}


def frame_document(
    state: FrameState,
    *,
    fps: float,
    camera: HomographyRecord,
    last: int,
    transition: bool,
    cut: bool,
    decided_unix: float,
    dropped: int,
) -> dict[str, Any]:
    """Return the line of one decided frame (module docstring)."""
    if camera.frame != state.frame:
        raise ValueError(f"the camera of frame {state.frame} names frame {camera.frame}")
    if last < state.frame:
        raise ValueError(f"frame {state.frame} cannot be decided at the earlier frame {last}")
    return {
        "frame": state.frame,
        "t_s": state.frame / fps,
        "decided_at": last,
        "delay_frames": last - state.frame,
        "decided_unix": decided_unix,
        "camera": camera.source,
        "transition": transition,
        "cut": cut,
        "dropped": dropped,
        "players": [_player(player) for player in state.players],
        "ball": _ball(state.ball),
    }


def number_document(
    *,
    track_id: int,
    export: int | None,
    display: int | None,
    first_frame: int,
    last_frame: int,
    crop_frames: tuple[int, ...],
    reason: str,
    decided_at: int,
    decided_unix: float,
) -> dict[str, Any]:
    """Return a number line (module docstring): ``number`` is the shown number, the displayed one else the export one."""
    if not 0 <= first_frame <= last_frame:
        raise ValueError(f"the frames of a number line must be in order from 0, got {first_frame} and {last_frame}")
    return {
        "kind": NUMBER_KIND,
        "id": track_id,
        "number": display if display is not None else export,
        "export": export,
        "display": display,
        "frames": [first_frame, last_frame],
        "crops": list(crop_frames),
        "reason": reason,
        "decided_at": decided_at,
        "decided_unix": decided_unix,
    }


def dumps(document: dict[str, Any]) -> bytes:
    """Return one line of the live output: compact JSON without NaN, and its line end."""
    return (json.dumps(document, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def _write_all(fd: int, data: bytes) -> None:
    """Write all bytes to a file descriptor (``os.write`` can write less than it is given to a pipe)."""
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


#: The kinds of the items that wait for the writer of a pipe.
_HEADER, _FRAME, _END, _NUMBER = "header", "frame", "end", "number"


class LiveOut:
    """Write the live output to a file descriptor, with the policy of a file or of a pipe (module docstring).

    ``start`` writes the header; ``given`` is the hook ``DetectorStream.on_given``; ``finish``
    writes the end line; ``close`` waits at most ``wait_s`` for a slow pipe reader and returns the
    report. The file descriptor is closed by ``close`` when ``close_fd`` is true.

    Each frame line is made (and checked) in the thread of the stream; the writer of a pipe
    only adds the count ``dropped`` and writes it. An error of the writer that is not one of
    its reader is raised by ``close`` (``LiveOutError``), so it does not pass without a word.
    """

    def __init__(
        self,
        fd: int,
        info: ClipInfo,
        *,
        target: str,
        pipe: bool,
        close_fd: bool,
        pending_lines: int = PIPE_PENDING_LINES,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if pending_lines < 1:
            raise ValueError("at least one line must be able to wait for the writer")
        self._fd = fd
        self._info = info
        self._target = target
        self._pipe = pipe
        self._close_fd = close_fd
        self._pending_lines = pending_lines
        self._clock = clock
        self._lock = threading.Condition()
        #: Items that wait for the writer of a pipe: (kind, line or frame document), in order.
        self._queue: deque[tuple[str, Any]] = deque()
        self._waiting_frames = 0
        #: The writer of a pipe is in a write now (outside the lock), since this time of ``time.monotonic``.
        self._writing = False
        self._write_started = 0.0
        #: ``close`` gave the writer up: it stops after the write it is in.
        self._abandoned = False
        self._given = 0
        self._written = 0
        self._dropped = 0
        #: Lines of a pipe that never went out because the reader went away.
        self._lost = 0
        self._reader_gone: str | None = None
        self._failure: BaseException | None = None
        self._started = False
        self._finished = False
        self._closed = False
        self._thread: threading.Thread | None = None
        self._numbers = False
        #: Number lines written (a file) or handed to the writer (a pipe).
        self.numbers_given = 0

    @property
    def closed(self) -> bool:
        """Whether ``close`` was called."""
        return self._closed

    @property
    def policy(self) -> str:
        """``pipe`` or ``file`` (module docstring)."""
        return "pipe" if self._pipe else "file"

    def _write(self, data: bytes) -> None:
        """Write one line in the thread of the stream; an error of a file stops the stream."""
        try:
            _write_all(self._fd, data)
        except OSError as error:
            raise LiveOutError(f"the live output {self._target} cannot be written: {error}") from error

    def start(self, *, delay_frames: int | None, history_frames: int | None, numbers: bool = False) -> None:
        """Write the header line (a file), or hand it to the writer thread that starts here (a pipe).

        ``numbers`` names the format with number lines (``LIVE_FORMAT_NUMBERS``) and allows ``number``.
        """
        if self._started:
            raise ValueError("the live output is already started")
        self._started = True
        self._numbers = numbers
        header = dumps(
            header_document(
                self._info, delay_frames=delay_frames, history_frames=history_frames, policy=self.policy, numbers=numbers
            )
        )
        if not self._pipe:
            self._write(header)
            return
        self._queue.append((_HEADER, header))
        self._thread = threading.Thread(target=self._run, name="live-out", daemon=True)
        self._thread.start()

    def given(
        self, state: FrameState, *, camera: HomographyRecord, last: int, transition: bool, cut: bool
    ) -> None:
        """Take one decided state (the hook ``DetectorStream.on_given``): write it, or hand it to the writer."""
        if not self._started or self._finished:
            raise ValueError("the live output takes frames only between start and finish")
        document = frame_document(
            state,
            fps=self._info.fps,
            camera=camera,
            last=last,
            transition=transition,
            cut=cut,
            decided_unix=self._clock(),
            dropped=0,
        )
        self._given += 1
        if not self._pipe:
            self._write(dumps(document))
            self._written += 1
            return
        with self._lock:
            if self._reader_gone is not None or self._failure is not None:
                self._lost += 1
                return
            held = self._writing and time.monotonic() - self._write_started > PIPE_BLOCKED_S
            if self._waiting_frames >= (self._pending_lines if held else PIPE_MAX_LINES):
                oldest = next(index for index, (kind, _) in enumerate(self._queue) if kind == _FRAME)
                del self._queue[oldest]
                self._waiting_frames -= 1
                self._dropped += 1
            self._queue.append((_FRAME, document))
            self._waiting_frames += 1
            self._lock.notify()

    def number(
        self,
        *,
        track_id: int,
        export: int | None,
        display: int | None,
        first_frame: int,
        last_frame: int,
        crop_frames: tuple[int, ...],
        reason: str,
        last: int,
    ) -> None:
        """Take one number line, decided at the frame ``last``: write it, or hand it to the writer (never left out)."""
        if not self._numbers:
            raise ValueError("the live output was started without number lines")
        if not self._started or self._finished:
            raise ValueError("the live output takes number lines only between start and finish")
        document = number_document(
            track_id=track_id,
            export=export,
            display=display,
            first_frame=first_frame,
            last_frame=last_frame,
            crop_frames=crop_frames,
            reason=reason,
            decided_at=last,
            decided_unix=self._clock(),
        )
        self.numbers_given += 1
        if not self._pipe:
            self._write(dumps(document))
            return
        with self._lock:
            if self._reader_gone is not None or self._failure is not None:
                self._lost += 1
                return
            self._queue.append((_NUMBER, document))
            self._lock.notify()

    def _end_document(self) -> dict[str, Any]:
        return {"end": True, "frames": self._given, "written": self._written, "dropped": self._dropped}

    def _run(self) -> None:
        """Write the waiting lines of a pipe in order, until the end line is out, the reader went away or ``close``."""
        try:
            self._write_queue()
        except BaseException as error:  # noqa: BLE001 (raised again by close)
            with self._lock:
                self._failure = error
                self._writing = False
                self._lost += self._waiting_frames + sum(1 for kind, _ in self._queue if kind in (_END, _NUMBER))
                self._queue.clear()
                self._waiting_frames = 0

    def _write_queue(self) -> None:
        while True:
            with self._lock:
                while not self._queue and not self._abandoned:
                    self._lock.wait()
                if self._abandoned:
                    return
                kind, value = self._queue.popleft()
                if kind == _FRAME:
                    self._waiting_frames -= 1
                    value = {**value, "dropped": self._dropped}
                elif kind == _END:
                    value = self._end_document()
                self._writing = True
                self._write_started = time.monotonic()
            # The line is made outside the lock, so that the stream never waits for it.
            data = value if kind == _HEADER else dumps(value)
            try:
                _write_all(self._fd, data)
            except OSError as error:
                with self._lock:
                    self._writing = False
                    self._reader_gone = str(error) or type(error).__name__
                    self._lost += 1 + self._waiting_frames + sum(1 for queued, _ in self._queue if queued in (_END, _NUMBER))
                    self._queue.clear()
                    self._waiting_frames = 0
                return
            with self._lock:
                self._writing = False
                if kind == _FRAME:
                    self._written += 1
                if kind == _END:
                    return

    def finish(self) -> None:
        """Write the end line (a file), or hand it to the writer (a pipe)."""
        if not self._started:
            raise ValueError("the live output is not started")
        if self._finished:
            raise ValueError("the live output is already finished")
        self._finished = True
        if not self._pipe:
            self._write(dumps(self._end_document()))
            return
        with self._lock:
            if self._reader_gone is None and self._failure is None:
                self._queue.append((_END, None))
                self._lock.notify()
            else:
                self._lost += 1

    def close(self, wait_s: float = CLOSE_WAIT_S) -> LiveReport:
        """Wait at most ``wait_s`` for the writer of a pipe, close the file descriptor when it is ours, and report.

        A stream that stopped with an error closes without ``finish``: the output has no end
        line, and the writer of a pipe stops at once. A writer that is still in a write blocked
        by its reader after ``wait_s`` is given up: it stops after that write, which may go out
        or not (it counts as not delivered); the lines that still wait count as not delivered
        too, and the file descriptor stays open until the process ends, because closing it under
        a blocked write is not safe. ``LiveOutError`` is raised when the writer failed with an
        error that is not one of its reader.
        """
        if self._closed:
            raise ValueError("the live output is already closed")
        self._closed = True
        thread = self._thread
        if thread is not None:
            if not self._finished:
                with self._lock:
                    self._abandoned = True
                    self._lock.notify_all()
            thread.join(wait_s)
        with self._lock:
            self._abandoned = True
            self._lock.notify_all()
            stuck = thread is not None and thread.is_alive() and self._writing
            waiting = self._waiting_frames + sum(1 for kind, _ in self._queue if kind in (_END, _NUMBER))
            report = LiveReport(
                target=self._target,
                policy=self.policy,
                given=self._given,
                written=self._written,
                dropped=self._dropped,
                undelivered=self._lost + waiting + int(stuck),
                reader_gone=self._reader_gone,
            )
            failure = self._failure
        if thread is not None and not stuck:
            thread.join(1.0)
        if self._close_fd and not stuck:
            os.close(self._fd)
        if failure is not None:
            raise LiveOutError(f"the writer of the live output {self._target} failed: {failure!r}") from failure
        return report


def is_pipe_like(fd: int) -> bool:
    """Say whether a file descriptor takes the policy of a pipe: anything but a regular file."""
    return not stat.S_ISREG(os.fstat(fd).st_mode)


def open_live_out(target: str, info: ClipInfo) -> LiveOut:
    """Open the live output ``target``: ``-`` for the standard output, else a path (a file or a named pipe).

    A path that is a named pipe waits here until its reader opens it. A regular file is made
    or emptied. The standard output is duplicated, so the output of ``stream`` itself must go
    elsewhere (``cli`` sends it to the error stream).
    """
    if target == "-":
        fd = os.dup(1)
        return LiveOut(fd, info, target="standard output", pipe=is_pipe_like(fd), close_fd=True)
    try:
        if os.path.exists(target) and stat.S_ISFIFO(os.stat(target).st_mode):
            fd = os.open(target, os.O_WRONLY)
        else:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    except OSError as error:
        raise LiveOutError(f"the live output {target} cannot be opened: {error}") from error
    return LiveOut(fd, info, target=target, pipe=is_pipe_like(fd), close_fd=True)
