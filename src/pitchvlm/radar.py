"""A live 2D radar of a live output (``liveout``), served to a browser on this machine.

``radar FILE|-`` reads the lines of ``stream --live-out`` (a file that grows, a named pipe, or
the standard input) and serves one page and one event stream with the standard library
(``http.server``, server-sent events). The server:

- binds to a loopback address only (``check_loopback``); any other address is refused before
  a socket is made. The default port 0 takes a free port, which is printed;
- answers ``GET /`` (the page) and ``GET /events`` (the event stream) and nothing else: no file
  is served, no path or query of a request is used, every other path is 404 and every other
  method 405. A ``Host`` header that is not the address of the server is refused (403), so a
  page of another site cannot reach the server through a name that points to the loopback
  address (DNS rebinding);
- sends ``Cache-Control: no-store`` and a content security policy that allows only the inline
  script and style of the page (by their SHA-256) and connections to the server itself: no
  external load of any kind;
- re-writes every line as compact JSON before it goes out, so a line of the feed cannot add an
  event or a field of the event stream;
- takes at most ``MAX_CLIENTS`` event streams at a time.

The page (``PAGE_HTML``) is one self-contained document: the pitch is drawn on a canvas by
code, the players by team colour with the shirt number when it is known, the ball, the time
of the frame, and how far the page is behind the decision (wall clock of this machine).

Events: ``header`` (the header line, with ``"replay"``), then ``frame`` (one frame line, its
frame index as the event id), then ``end`` (the end line, or ``{"end": true, "complete": false}``
when the feed closed without one). A client gets the header and then every frame from the
newest one on; a client that falls more than ``CLIENT_MAX_BEHIND`` frames behind jumps to the
newest frame (a viewer needs the newest state, the gaps show in the frame index). The server
ends when the feed ended and every client got the end event (or after ``END_WAIT_S``).
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import logging
import math
import os
import re
import socket
import socketserver
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from pitchvlm.liveout import LIVE_FORMAT

logger = logging.getLogger(__name__)

#: Event streams that the server serves at one time.
MAX_CLIENTS = 8
#: Frames that a client may lag behind the newest frame before it jumps to the newest one: 1 s at 25 fps.
CLIENT_MAX_BEHIND = 25
#: Events that the server keeps for the clients.
RING_EVENTS = 100
#: Seconds between two keep-alive comments of an idle event stream (a client that went away is found then).
KEEP_ALIVE_S = 10.0
#: Seconds that the server waits, after the end of the feed, for the clients to get the end event.
END_WAIT_S = 2.0
#: Seconds between two looks at a file that has not grown.
FOLLOW_POLL_S = 0.05
#: Seconds that one read or write of a request may wait (a client that stops is dropped).
REQUEST_TIMEOUT_S = 10.0
#: Connections that the server holds at one time; a connection above it is closed at once.
MAX_CONNECTIONS = 32
#: The longest line of a feed, bytes: a frame line of 23 people is about 5 kB.
MAX_LINE_BYTES = 1 << 20


class RadarError(ValueError):
    """Wrong input of the viewer: an address that is not loopback, or a feed that is not a live output."""


def check_loopback(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Return the address when it is a loopback address written as numbers, and raise ``RadarError`` otherwise.

    A host name is refused too (``localhost`` included): a name can be resolved to another address.
    An IPv6 address with a zone or one that maps an IPv4 address is refused: a browser cannot
    name it in a ``Host`` header that the server takes.
    """
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        raise RadarError(f"--host {host!r} is not an address: give a loopback address such as 127.0.0.1") from None
    if isinstance(address, ipaddress.IPv6Address) and (address.scope_id or address.ipv4_mapped):
        raise RadarError(f"--host {host} is not a plain loopback address: give 127.0.0.1 or ::1")
    if not address.is_loopback:
        raise RadarError(f"--host {host} is not a loopback address: the radar serves this machine only")
    return address


# --- The feed ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """One event of the event stream: its number, its kind, its frame (frames only) and its data (compact JSON)."""

    seq: int
    kind: str
    frame: int | None
    data: str


class Board:
    """The events of the feed for the clients: the header, the newest frames, the end (module docstring)."""

    def __init__(self, *, replay: bool = False) -> None:
        self._lock = threading.Condition()
        self._replay = replay
        self._header: Event | None = None
        self._frames: deque[Event] = deque(maxlen=RING_EVENTS)
        self._end: Event | None = None
        self._seq = 0
        self._closed = False
        #: Event streams open now, and served in all.
        self.clients = 0
        self.served = 0
        self.first_client = threading.Event()

    def close(self) -> None:
        """Wake every client that waits: the server stops."""
        with self._lock:
            self._closed = True
            self._lock.notify_all()

    @property
    def closed(self) -> bool:
        """Whether the server stops."""
        with self._lock:
            return self._closed

    def sleep(self, seconds: float) -> None:
        """Wait ``seconds``, or less when the board is closed (the server stops)."""
        deadline = time.monotonic() + seconds
        with self._lock:
            while not self._closed:
                left = deadline - time.monotonic()
                if left <= 0:
                    return
                self._lock.wait(left)

    def open_client(self) -> bool:
        """Count a new event stream, or return False when ``MAX_CLIENTS`` are open."""
        with self._lock:
            if self.clients >= MAX_CLIENTS:
                return False
            self.clients += 1
            self.served += 1
        self.first_client.set()
        return True

    def close_client(self) -> None:
        """Count an event stream that ended."""
        with self._lock:
            self.clients -= 1
            self._lock.notify_all()

    def wait_for_clients(self, timeout: float) -> None:
        """Wait at most ``timeout`` seconds until no event stream is open."""
        deadline = time.monotonic() + timeout
        with self._lock:
            while self.clients and time.monotonic() < deadline:
                self._lock.wait(max(0.0, deadline - time.monotonic()))

    def _next(self, kind: str, frame: int | None, document: dict[str, Any]) -> Event:
        self._seq += 1
        return Event(self._seq, kind, frame, json.dumps(document, allow_nan=False, separators=(",", ":")))

    def header(self, document: dict[str, Any]) -> None:
        """Take the header line."""
        with self._lock:
            self._header = self._next("header", None, {**document, "replay": self._replay})
            self._lock.notify_all()

    def frame(self, document: dict[str, Any]) -> None:
        """Take one frame line."""
        with self._lock:
            self._frames.append(self._next("frame", int(document["frame"]), document))
            self._lock.notify_all()

    def end(self, document: dict[str, Any]) -> None:
        """Take the end line (or the end of a feed without one)."""
        with self._lock:
            if self._end is None:
                self._end = self._next("end", None, document)
            self._lock.notify_all()

    @property
    def ended(self) -> bool:
        """Whether the end event is there."""
        with self._lock:
            return self._end is not None

    def wait(self, after: int, timeout: float) -> list[Event]:
        """Return the events for a client whose last event has the number ``after`` (0: none yet).

        A new client gets the header and the newest frame; a client that lags more than
        ``CLIENT_MAX_BEHIND`` frames gets the newest frame; after the frames, the end. The
        list is empty when nothing came within ``timeout`` seconds, or when the board is closed.
        """
        deadline = time.monotonic() + timeout
        with self._lock:
            while not self._closed:
                events = self._after(after)
                if events:
                    return events
                left = deadline - time.monotonic()
                if left <= 0:
                    return []
                self._lock.wait(left)
            return []

    def _after(self, after: int) -> list[Event]:
        if self._header is None:
            return []
        events: list[Event] = []
        if after < self._header.seq:
            events.append(self._header)
            newer = list(self._frames)[-1:]
        else:
            newer = [event for event in self._frames if event.seq > after]
            if len(newer) > CLIENT_MAX_BEHIND:
                newer = newer[-1:]
        events.extend(newer)
        if self._end is not None and after < self._end.seq:
            events.append(self._end)
        return events


def feed_documents(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Yield the header, the frame lines and the end line of a live output, checked; stop after the end line.

    ``RadarError`` is raised for a first line that is not a header of ``LIVE_FORMAT`` and
    for a line that is not a JSON object of the format (a frame line needs a whole frame
    number of 0 or more and a time ``t_s`` that is a finite number of 0 or more).
    """
    seen_header = False
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            document = json.loads(line, parse_constant=_no_constant)
        except (ValueError, RecursionError) as error:
            raise RadarError(f"line {number} of the feed is not JSON: {error}") from None
        if not isinstance(document, dict):
            raise RadarError(f"line {number} of the feed is not a JSON object")
        if not seen_header:
            if document.get("format") != LIVE_FORMAT:
                raise RadarError(f"the feed is not a live output of the format {LIVE_FORMAT} (line {number})")
            seen_header = True
            yield document
            continue
        if document.get("end") is True:
            yield document
            return
        frame = document.get("frame")
        if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
            raise RadarError(f"line {number} of the feed is neither a frame line nor the end line")
        t_s = document.get("t_s")
        if not isinstance(t_s, (int, float)) or isinstance(t_s, bool) or not math.isfinite(t_s) or t_s < 0:
            raise RadarError(f"line {number} of the feed has no time t_s of 0 or more")
        yield document


def _no_constant(token: str) -> Any:
    raise ValueError(f"{token} is not a number of the feed")


def follow_file(path: str, stop: threading.Event, poll_s: float = FOLLOW_POLL_S) -> Iterator[str]:
    """Yield the whole lines of a file that grows, as they are written, until ``stop`` is set.

    A last line without its line end waits until its end is written. A named pipe or the
    standard input ends at its end of file instead (``read_lines``).
    """
    splitter = _Lines()
    with open(path, "rb") as handle:
        while not stop.is_set():
            chunk = handle.read(65536)
            if not chunk:
                stop.wait(poll_s)
                continue
            yield from splitter.add(chunk)


def read_lines(fd: int) -> Iterator[str]:
    """Yield the lines of a pipe until its end of file (a last line without its end as it is)."""
    return read_lines_of(iter(lambda: os.read(fd, 65536), b""))


def read_lines_of(chunks: Iterable[bytes]) -> Iterator[str]:
    """Yield the lines of a sequence of chunks of bytes (a last line without its end as it is)."""
    splitter = _Lines()
    for chunk in chunks:
        yield from splitter.add(chunk)
    yield from splitter.rest()


class _Lines:
    """Cut chunks of bytes into lines of UTF-8 text, each at most ``MAX_LINE_BYTES`` long (``RadarError``)."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def add(self, chunk: bytes) -> Iterator[str]:
        """Yield the whole lines that the chunk ends; keep the part after the last line end."""
        start = len(self._buffer)
        self._buffer += chunk
        lines: list[str] = []
        begin = 0
        end = self._buffer.find(b"\n", start)
        while end != -1:
            lines.append(self._text(self._buffer[begin:end]))
            begin = end + 1
            end = self._buffer.find(b"\n", begin)
        del self._buffer[:begin]
        if len(self._buffer) > MAX_LINE_BYTES:
            raise RadarError(f"a line of the feed is longer than {MAX_LINE_BYTES} bytes")
        return iter(lines)

    def rest(self) -> Iterator[str]:
        """Yield the last line when it has no line end."""
        if self._buffer:
            yield self._text(self._buffer)

    @staticmethod
    def _text(line: bytes | bytearray) -> str:
        if len(line) > MAX_LINE_BYTES:
            raise RadarError(f"a line of the feed is longer than {MAX_LINE_BYTES} bytes")
        try:
            return bytes(line).decode("utf-8")
        except UnicodeDecodeError as error:
            raise RadarError(f"a line of the feed is not UTF-8 text: {error}") from None


def pump(
    lines: Iterable[str],
    board: Board,
    *,
    replay: bool = False,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Put the documents of a feed on the board; with ``replay``, a frame not before the first page plus its time.

    Return the number of frame lines. A feed that closes without its end line ends the
    board with ``{"end": true, "complete": false}``.
    """
    frames = 0
    start: float | None = None
    try:
        for index, document in enumerate(feed_documents(lines)):
            if index == 0:
                board.header(document)
                continue
            if document.get("end") is True:
                board.end({**document, "complete": True})
                return frames
            if replay:
                if start is None:
                    while not board.first_client.wait(0.1):
                        if board.closed:
                            return frames
                    start = clock() - float(document["t_s"])
                wait = start + float(document["t_s"]) - clock()
                if wait > 0:
                    board.sleep(wait)
                if board.closed:
                    return frames
            board.frame(document)
            frames += 1
    finally:
        board.end({"end": True, "complete": False})
    return frames


# --- The server ----------------------------------------------------------------------------


PAGE_STYLE = """
:root { color-scheme: dark; --bg: #10151c; --fg: #e8edf2; --muted: #93a1b0; --grass: #1f5c33; --line: #e9f1e9; }
* { box-sizing: border-box; }
body { margin: 0; padding: 16px; background: var(--bg); color: var(--fg);
       font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { display: flex; flex-wrap: wrap; gap: 8px 24px; align-items: baseline; margin-bottom: 12px; }
h1 { font-size: 16px; margin: 0; font-weight: 600; }
.stat { color: var(--muted); }
.stat b { color: var(--fg); font-variant-numeric: tabular-nums; font-weight: 600; }
canvas { display: block; width: 100%; max-width: 1100px; height: auto; border-radius: 6px; background: var(--grass); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin-top: 10px; color: var(--muted); }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }
.a { background: #3d7eff; } .b { background: #ff4d5e; } .n { background: #b9c2cc; }
.r { background: #f5d33a; } .ball { background: #ffffff; border: 1px solid #000; }
""".strip()

PAGE_SCRIPT = r"""
"use strict";
const L = 105, W = 68, M = 4;
const canvas = document.getElementById("pitch");
const ctx = canvas.getContext("2d");
const el = (id) => document.getElementById(id);
const COLOURS = { A: "#3d7eff", B: "#ff4d5e", none: "#b9c2cc" };
const TEXT_ON = { A: true, B: true };
let header = null, latest = null, lastFrame = null, gaps = 0, lagMs = null, state = "waiting for the feed";

function size() {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 1000;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(width * (W + 2 * M) / (L + 2 * M) * ratio);
  draw();
}
function sx(x) { return (x + L / 2 + M) * canvas.width / (L + 2 * M); }
function sy(y) { return (y + W / 2 + M) * canvas.height / (W + 2 * M); }
function sm(m) { return m * canvas.width / (L + 2 * M); }

function pitch() {
  ctx.fillStyle = "#1f5c33";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#e9f1e9";
  ctx.lineWidth = Math.max(1, sm(0.25));
  const rect = (x1, y1, x2, y2) => ctx.strokeRect(sx(x1), sy(y1), sx(x2) - sx(x1), sy(y2) - sy(y1));
  rect(-L / 2, -W / 2, L / 2, W / 2);
  ctx.beginPath(); ctx.moveTo(sx(0), sy(-W / 2)); ctx.lineTo(sx(0), sy(W / 2)); ctx.stroke();
  ctx.beginPath(); ctx.arc(sx(0), sy(0), sm(9.15), 0, 2 * Math.PI); ctx.stroke();
  for (const side of [-1, 1]) {
    const goal = side * L / 2;
    rect(Math.min(goal, goal - side * 16.5), -20.16, Math.max(goal, goal - side * 16.5), 20.16);
    rect(Math.min(goal, goal - side * 5.5), -9.16, Math.max(goal, goal - side * 5.5), 9.16);
    rect(Math.min(goal, goal + side * 1.5), -3.66, Math.max(goal, goal + side * 1.5), 3.66);
    const spot = goal - side * 11;
    ctx.beginPath(); ctx.arc(sx(spot), sy(0), sm(0.3), 0, 2 * Math.PI); ctx.fillStyle = "#e9f1e9"; ctx.fill();
    const a = Math.acos(5.5 / 9.15);
    ctx.beginPath();
    if (side > 0) ctx.arc(sx(spot), sy(0), sm(9.15), Math.PI - a, Math.PI + a);
    else ctx.arc(sx(spot), sy(0), sm(9.15), -a, a);
    ctx.stroke();
  }
  ctx.beginPath(); ctx.arc(sx(0), sy(0), sm(0.3), 0, 2 * Math.PI); ctx.fill();
}

function draw() {
  pitch();
  if (!latest) return;
  const r = Math.max(5, sm(1.1));
  for (const p of latest.players) {
    const colour = p.role === "referee" ? "#f5d33a" : (COLOURS[p.team] || COLOURS.none);
    ctx.globalAlpha = p.observed ? 1 : 0.55;
    ctx.beginPath(); ctx.arc(sx(p.x), sy(p.y), r, 0, 2 * Math.PI);
    ctx.fillStyle = colour; ctx.fill();
    ctx.lineWidth = p.role === "goalkeeper" ? Math.max(2, r * 0.35) : 1;
    ctx.strokeStyle = p.role === "goalkeeper" ? "#ffffff" : "#0b0f14"; ctx.stroke();
    if (Number.isInteger(p.number)) {
      ctx.fillStyle = p.role === "referee" || !(p.team in TEXT_ON) ? "#0b0f14" : "#ffffff";
      ctx.font = "600 " + Math.round(r * 1.1) + "px system-ui, sans-serif";
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(String(p.number), sx(p.x), sy(p.y) + 0.5);
    }
  }
  ctx.globalAlpha = 1;
  const ball = latest.ball;
  if (ball && ball.x !== null && ball.y !== null) {
    ctx.beginPath(); ctx.arc(sx(ball.x), sy(ball.y), Math.max(3, r * 0.55), 0, 2 * Math.PI);
    ctx.fillStyle = "#ffffff"; ctx.fill(); ctx.lineWidth = 1.5; ctx.strokeStyle = "#000000"; ctx.stroke();
  }
}

function clock(t) {
  const minutes = Math.floor(t / 60), seconds = t - 60 * minutes;
  return String(minutes).padStart(2, "0") + ":" + seconds.toFixed(2).padStart(5, "0");
}
function text() {
  el("status").textContent = state;
  if (!header) return;
  el("clip").textContent = header.clip;
  if (!latest) return;
  el("time").textContent = clock(latest.t_s);
  el("frame").textContent = latest.frame + " / " + header.n_frames;
  el("delay").textContent = latest.delay_frames + " frames";
  el("behind").textContent = header.replay ? "replay" : (lagMs === null ? "-" : Math.round(lagMs) + " ms");
  el("gaps").textContent = String(gaps + latest.dropped);
  el("count").textContent = String(latest.players.length);
  const flags = [];
  if (latest.camera === "none") flags.push("no camera");
  if (latest.transition) flags.push("transition");
  if (latest.cut) flags.push("cut");
  el("flags").textContent = flags.length ? flags.join(", ") : "-";
}

const events = new EventSource("/events");
events.addEventListener("header", (e) => { header = JSON.parse(e.data); state = "live"; text(); });
events.addEventListener("frame", (e) => {
  const line = JSON.parse(e.data);
  if (lastFrame !== null && line.frame > lastFrame + 1) gaps += line.frame - lastFrame - 1;
  lastFrame = line.frame;
  lagMs = Date.now() - line.decided_unix * 1000;
  latest = line;
  window.requestAnimationFrame(() => { draw(); text(); });
});
events.addEventListener("end", (e) => {
  const end = JSON.parse(e.data);
  state = end.complete ? "the feed ended" : "the feed closed without its end line";
  events.close(); text();
});
events.onerror = () => { if (events.readyState === EventSource.CLOSED) { state = "disconnected"; text(); } };
window.addEventListener("resize", size);
size(); text();
""".strip()

PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pitchvlm radar</title>
<style>{style}</style>
</head>
<body>
<header>
<h1>pitchvlm radar <span class="stat" id="clip"></span></h1>
<span class="stat">time <b id="time">-</b></span>
<span class="stat">frame <b id="frame">-</b></span>
<span class="stat">delay <b id="delay">-</b></span>
<span class="stat">page behind the decision <b id="behind">-</b></span>
<span class="stat">frames not shown <b id="gaps">0</b></span>
<span class="stat">people <b id="count">-</b></span>
<span class="stat">flags <b id="flags">-</b></span>
<span class="stat"><b id="status"></b></span>
</header>
<canvas id="pitch" width="1000" height="660"></canvas>
<div class="legend">
<span><span class="dot a"></span>team A</span><span><span class="dot b"></span>team B</span>
<span><span class="dot n"></span>no team</span><span><span class="dot r"></span>referee</span>
<span><span class="dot ball"></span>ball</span><span>white ring: goalkeeper; faded: filled in, not seen</span>
</div>
<script>{script}</script>
</body>
</html>
"""


def _sha256(text: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode("ascii") + "'"


PAGE_HTML = PAGE_TEMPLATE.format(style=PAGE_STYLE, script=PAGE_SCRIPT)
#: The content security policy: the inline script and style of the page, and connections to the server only.
CONTENT_SECURITY_POLICY = (
    f"default-src 'none'; script-src {_sha256(PAGE_SCRIPT)}; style-src {_sha256(PAGE_STYLE)}; "
    "connect-src 'self'; img-src 'none'; font-src 'none'; media-src 'none'; object-src 'none'; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)
SECURITY_HEADERS: tuple[tuple[str, str], ...] = (
    ("Cache-Control", "no-store"),
    ("Content-Security-Policy", CONTENT_SECURITY_POLICY),
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "no-referrer"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
    ("Cross-Origin-Opener-Policy", "same-origin"),
)
_PAGE_BYTES = PAGE_HTML.encode("utf-8")
#: The scheme of the address of the server (the server speaks plain HTTP on the loopback address only).
_SCHEME = "http" + "://"
_HOST_RE = re.compile(r"^[A-Za-z0-9.:\[\]-]+$")


class RadarServer(ThreadingHTTPServer):
    """The HTTP server of the radar: the board, and the ``Host`` values that name it."""

    daemon_threads = True
    allow_reuse_address = False
    request_queue_size = 16

    def __init__(self, address: ipaddress.IPv4Address | ipaddress.IPv6Address, port: int, board: Board) -> None:
        self.address_family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
        super().__init__((str(address), port), RadarHandler)
        self.board = board
        self.stopping = threading.Event()
        written = f"[{address}]" if address.version == 6 else str(address)
        #: The ``Host`` values of a request to this server: the address, and ``localhost`` for 127.0.0.1 and ::1.
        names = {written}
        if address in (ipaddress.ip_address("127.0.0.1"), ipaddress.ip_address("::1")):
            names.add("localhost")
        self.hosts = frozenset(f"{name}:{self.port}" for name in names)
        self.origins = frozenset(_SCHEME + host for host in self.hosts)
        self.url = f"{_SCHEME}{written}:{self.port}/"
        self._connections = threading.BoundedSemaphore(MAX_CONNECTIONS)

    def process_request(self, request: Any, client_address: Any) -> None:
        """Serve a connection on its own thread, or close it at once when ``MAX_CONNECTIONS`` are open."""
        if not self._connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._connections.release()
            raise

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Pass over a client that went away; log any other error of a request, without its traceback on the screen."""
        error = sys.exc_info()[1]
        if isinstance(error, (ConnectionError, TimeoutError)):
            return
        logger.warning("a request failed: %r", error)

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        """Serve one connection and give its place back."""
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connections.release()

    def server_bind(self) -> None:
        """Bind the socket without a look-up of the name of the address (``HTTPServer.server_bind`` asks DNS)."""
        socketserver.TCPServer.server_bind(self)
        self.server_name = str(self.server_address[0])
        self.server_port = int(self.server_address[1])

    @property
    def port(self) -> int:
        """The port that the server listens on."""
        return int(self.server_address[1])


class RadarHandler(BaseHTTPRequestHandler):
    """Answer ``GET /`` and ``GET /events`` (module docstring); refuse every other request."""

    server: RadarServer
    server_version = "pitchvlm-radar"
    sys_version = ""
    protocol_version = "HTTP/1.1"
    timeout = REQUEST_TIMEOUT_S

    def version_string(self) -> str:
        """Name the server without the version of Python."""
        return self.server_version

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (the name of the base class)
        """Write no line per request."""

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        """Answer an error of the request line or the headers (sent before ``do_*``) as the other answers: plain text."""
        self._plain(code, "bad request" if code < 500 else "not supported")

    def _plain(self, code: int, text: str, extra: tuple[tuple[str, str], ...] = ()) -> None:
        body = (text + "\n").encode("utf-8")
        self.send_response(code)
        for name, value in (*SECURITY_HEADERS, *extra):
            self.send_header(name, value)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        self.close_connection = True

    def _refusal(self) -> str | None:
        """Return why the request is refused (403), or None.

        The ``Host`` header must name this server (no other name that points to the loopback
        address: DNS rebinding). A request of a page of another site is refused: by its
        ``Origin`` header, or by ``Sec-Fetch-Site`` of a browser (only ``same-origin``, and
        ``none`` for an address typed by the person, pass). So no other site can hold the
        event streams, nor start a replay.
        """
        hosts = self.headers.get_all("Host") or []
        if not (len(hosts) == 1 and bool(_HOST_RE.match(hosts[0])) and hosts[0].lower() in self.server.hosts):
            return "the Host header does not name this server"
        origins = self.headers.get_all("Origin") or []
        if origins and (len(origins) != 1 or origins[0].lower() not in self.server.origins):
            return "the request comes from a page of another site"
        sites = self.headers.get_all("Sec-Fetch-Site") or []
        if sites and (len(sites) != 1 or sites[0].lower() not in ("same-origin", "none")):
            return "the request comes from a page of another site"
        return None

    def do_GET(self) -> None:  # noqa: N802 (the name of the base class)
        """Serve the page or the event stream."""
        refusal = self._refusal()
        if refusal is not None:
            self._plain(403, f"forbidden: {refusal}")
            return
        if self.path == "/":
            self.send_response(200)
            for name, value in SECURITY_HEADERS:
                self.send_header(name, value)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(_PAGE_BYTES)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            if self.command != "HEAD":
                self.wfile.write(_PAGE_BYTES)
            return
        if self.path == "/events" and self.command == "GET":
            self._events()
            return
        self._plain(404, "not found")

    def do_HEAD(self) -> None:  # noqa: N802 (the name of the base class)
        """Answer the headers of the page."""
        if self._refusal() is None and self.path == "/events":
            self._plain(405, "method not allowed", (("Allow", "GET"),))
            return
        self.do_GET()

    def _refuse_method(self) -> None:
        self._plain(405, "method not allowed", (("Allow", "GET, HEAD"),))

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_TRACE = do_CONNECT = _refuse_method

    def _events(self) -> None:
        board = self.server.board
        if not board.open_client():
            self._plain(503, "too many event streams", (("Retry-After", "5"),))
            return
        try:
            self.send_response(200)
            for name, value in SECURITY_HEADERS:
                self.send_header(name, value)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.wfile.write(b"retry: 2000\n\n")
            self.wfile.flush()
            after = 0
            while not self.server.stopping.is_set():
                events = board.wait(after, KEEP_ALIVE_S)
                if not events:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue
                for event in events:
                    head = f"id: {event.frame}\n" if event.frame is not None else ""
                    self.wfile.write(f"{head}event: {event.kind}\ndata: {event.data}\n\n".encode("utf-8"))
                    after = event.seq
                self.wfile.flush()
                if events[-1].kind == "end":
                    return
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError, OSError):
            return
        finally:
            board.close_client()


def serve(
    lines: Iterable[str],
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    replay: bool = False,
    ready: Callable[[RadarServer], None] | None = None,
    stop: threading.Event | None = None,
) -> tuple[int, Board]:
    """Serve the radar of a feed until the feed ended and its clients got the end, or ``stop`` is set.

    ``ready`` is called with the server once it listens (the command prints the address).
    Return the number of frame lines and the board. ``RadarError`` is raised for an address
    that is not loopback (before any socket is made) and for a feed that is not a live output.
    """
    address = check_loopback(host)
    if not 0 <= port <= 65535:
        raise RadarError(f"--port {port} is not a port")
    board = Board(replay=replay)
    server = RadarServer(address, port, board)
    errors: list[BaseException] = []
    counted: list[int] = []

    def read() -> None:
        try:
            counted.append(pump(lines, board, replay=replay))
        except BaseException as error:  # noqa: BLE001 (handed to the caller)
            errors.append(error)

    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, name="radar-http", daemon=True)
    reader = threading.Thread(target=read, name="radar-feed", daemon=True)
    try:
        worker.start()
        if ready is not None:
            ready(server)
        reader.start()
        while reader.is_alive():
            reader.join(0.1)
            if stop is not None and stop.is_set():
                break
        board.wait_for_clients(END_WAIT_S)
    finally:
        server.stopping.set()
        board.close()
        server.shutdown()
        server.server_close()
    if errors:
        raise errors[0]
    return (counted[0] if counted else 0), board
