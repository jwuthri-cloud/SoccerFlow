"""Tests of the radar viewer (``radar``): the server, its event stream, its page and its refusals.

Each test starts the server on a free loopback port inside the test, feeds it lines and reads
it with ``urllib`` or a raw socket. No browser runs here: the picture of the page is not tested.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from pitchvlm import cli, radar
from pitchvlm.liveout import LIVE_FORMAT

REPO = Path(__file__).resolve().parents[1]


def header(n_frames: int = 5) -> dict[str, Any]:
    """Return a header line of a live output."""
    return {
        "format": LIVE_FORMAT,
        "clip": "SNGS-999",
        "n_frames": n_frames,
        "fps": 25.0,
        "width": 1280,
        "height": 720,
        "delay_frames": 25,
        "history_frames": 123,
        "pitch_m": [105.0, 68.0],
        "policy": "pipe",
    }


def frame(index: int) -> dict[str, Any]:
    """Return a frame line of a live output with one player and the ball."""
    return {
        "frame": index,
        "t_s": index / 25.0,
        "decided_at": index + 25,
        "delay_frames": 25,
        "decided_unix": time.time(),
        "camera": "landmarks",
        "transition": False,
        "cut": False,
        "dropped": 0,
        "players": [
            {
                "id": 3,
                "role": "player",
                "team": "A",
                "side": "left",
                "number": 10,
                "x": -10.5 + index,
                "y": 4.25,
                "box": [100.0, 200.0, 120.0, 260.0],
                "observed": True,
            }
        ],
        "ball": {"x": 1.0, "y": 2.0, "box": [10.0, 10.0, 14.0, 14.0], "observed": False},
    }


class Feed:
    """Lines that a test hands to the server one by one (an iterator that waits)."""

    def __init__(self) -> None:
        self._lines: list[str | None] = []
        self._cond = threading.Condition()

    def send(self, document: dict[str, Any] | str) -> None:
        with self._cond:
            self._lines.append(document if isinstance(document, str) else json.dumps(document))
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self._lines.append(None)
            self._cond.notify()

    def __iter__(self) -> Iterator[str]:
        index = 0
        while True:
            with self._cond:
                while index >= len(self._lines):
                    self._cond.wait()
                line = self._lines[index]
            index += 1
            if line is None:
                return
            yield line


class Running:
    """A radar server on a thread: its address, its result or its error."""

    def __init__(self, lines: Any, **options: Any) -> None:
        self.ready = threading.Event()
        self.server: radar.RadarServer | None = None
        self.result: tuple[int, radar.Board] | None = None
        self.error: BaseException | None = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(lines, options), daemon=True)
        self.thread.start()
        assert self.ready.wait(5), "the server did not start"

    def _run(self, lines: Any, options: dict[str, Any]) -> None:
        def ready(server: radar.RadarServer) -> None:
            self.server = server
            self.ready.set()

        try:
            self.result = radar.serve(lines, ready=ready, stop=self.stop, **options)
        except BaseException as error:  # noqa: BLE001 (the test reads it)
            self.error = error
            self.ready.set()

    @property
    def url(self) -> str:
        assert self.server is not None
        return self.server.url

    def join(self, timeout: float = 10.0) -> None:
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "the server did not end"


def read_events(response: Any) -> Iterator[tuple[str, str | None, Any]]:
    """Yield the events (kind, id, data) of an event stream read with urllib."""
    kind, event_id, data = "message", None, []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\n")
        if line == "":
            if data:
                yield kind, event_id, json.loads("\n".join(data))
            kind, event_id, data = "message", None, []
        elif line.startswith(":") or line.startswith("retry:"):
            continue
        elif line.startswith("event: "):
            kind = line.removeprefix("event: ")
        elif line.startswith("id: "):
            event_id = line.removeprefix("id: ")
        elif line.startswith("data: "):
            data.append(line.removeprefix("data: "))


def test_the_events_are_the_lines_of_the_feed_and_the_server_ends_with_the_feed() -> None:
    feed = Feed()
    running = Running(feed)
    response = urllib.request.urlopen(running.url + "events", timeout=10)
    assert response.headers["Content-Type"] == "text/event-stream; charset=utf-8"
    assert response.headers["Cache-Control"] == "no-store"
    lines = [header(8), *(frame(index) for index in range(8)), {"end": True, "frames": 8, "written": 8, "dropped": 0}]
    for line in lines:
        feed.send(line)
        time.sleep(0.02)
    events = list(read_events(response))
    assert [kind for kind, _, _ in events] == ["header", *["frame"] * 8, "end"]
    assert events[0][2] == {**lines[0], "replay": False}
    for (kind, event_id, data), line in zip(events[1:-1], lines[1:-1], strict=True):
        assert data == line and event_id == str(line["frame"])
    assert events[-1][2] == {**lines[-1], "complete": True}
    running.join()
    assert running.error is None and running.result is not None
    frames, board = running.result
    assert frames == 8 and board.served == 1


def test_a_feed_without_its_end_line_ends_the_clients_with_an_incomplete_end() -> None:
    feed = Feed()
    running = Running(feed)
    response = urllib.request.urlopen(running.url + "events", timeout=10)
    feed.send(header(3))
    time.sleep(0.05)
    feed.send(frame(0))
    time.sleep(0.05)
    feed.close()
    events = list(read_events(response))
    assert [kind for kind, _, _ in events] == ["header", "frame", "end"]
    assert events[-1][2] == {"end": True, "complete": False}
    running.join()


def test_a_client_that_goes_away_does_not_stop_the_server() -> None:
    feed = Feed()
    running = Running(feed)
    feed.send(header(4))
    host, port = running.url.removeprefix("http://").rstrip("/").split(":")
    with socket.create_connection((host, int(port)), timeout=5) as sock:
        sock.sendall(f"GET /events HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode())
        assert b"200 OK" in sock.recv(4096)
    # The client is gone; the server goes on, serves another client and ends with the feed.
    feed.send(frame(0))
    time.sleep(0.1)
    response = urllib.request.urlopen(running.url + "events", timeout=10)
    feed.send(frame(1))
    feed.send({"end": True, "frames": 2, "written": 2, "dropped": 0})
    kinds = [kind for kind, _, _ in read_events(response)]
    assert kinds[0] == "header" and kinds[-1] == "end"
    running.join()
    assert running.error is None and running.result is not None
    assert running.result[1].served == 2


def test_a_client_that_lags_jumps_to_the_newest_frame() -> None:
    board = radar.Board()
    board.header(header(100))
    for index in range(60):
        board.frame(frame(index))
    events = board.wait(0, 0.1)
    assert [event.kind for event in events] == ["header", "frame"] and events[1].frame == 59
    board.frame(frame(60))
    board.frame(frame(61))
    assert [event.frame for event in board.wait(events[-1].seq, 0.1)] == [60, 61]
    for index in range(62, 62 + radar.CLIENT_MAX_BEHIND + 1):
        board.frame(frame(index))
    lagged = board.wait(events[-1].seq + 2, 0.1)
    assert [event.frame for event in lagged] == [62 + radar.CLIENT_MAX_BEHIND]


def test_the_page_names_no_external_host_and_loads_nothing() -> None:
    page = radar.PAGE_HTML
    assert not re.search(r"(?i)\b[a-z][a-z0-9+.-]*://|//[a-z0-9-]+\.[a-z]", page), "no address of any host"
    assert not re.search(r"(?i)\b(src|href|action|srcset|poster|data)\s*=", page)
    assert not re.search(r"(?i)@import|url\(|<link|<img|<iframe|<object|<embed|import\(|fetch\(|XMLHttpRequest|WebSocket", page)
    assert "new EventSource(\"/events\")" in page
    assert "innerHTML" not in page and "style=" not in page
    policy = radar.CONTENT_SECURITY_POLICY
    assert policy.startswith("default-src 'none';")
    assert "connect-src 'self'" in policy and "unsafe-inline" not in policy and "unsafe-eval" not in policy
    assert "frame-ancestors 'none'" in policy and "base-uri 'none'" in policy


def test_the_server_serves_only_the_page_and_the_events_to_its_own_host_name() -> None:
    feed = Feed()
    running = Running(feed)
    host = running.url.removeprefix("http://").rstrip("/")
    page = urllib.request.urlopen(running.url, timeout=5)
    assert page.status == 200 and page.read().decode("utf-8") == radar.PAGE_HTML
    assert page.headers["Content-Security-Policy"] == radar.CONTENT_SECURITY_POLICY
    assert page.headers["Cache-Control"] == "no-store" and page.headers["X-Content-Type-Options"] == "nosniff"
    assert page.headers["Server"] == "pitchvlm-radar"

    def status(path: str, *, method: str = "GET", host_header: str | None = host) -> int:
        connection = http.client.HTTPConnection(*host.split(":"), timeout=5)
        connection.putrequest(method, path, skip_host=True)
        if host_header is not None:
            connection.putheader("Host", host_header)
        connection.endheaders()
        response = connection.getresponse()
        body = response.read()
        assert response.getheader("Cache-Control") == "no-store"
        assert path.encode() not in body or path == "/"
        connection.close()
        return response.status

    for path in ("/index.html", "/events/", "/?x=1", "/events?x=1", "/../etc/passwd", "/%2e%2e/", "///x", "/favicon.ico", "/%65vents"):
        assert status(path) == 404, path
    for method in ("POST", "PUT", "DELETE", "OPTIONS", "PATCH", "TRACE"):
        assert status("/", method=method) == 405, method
    assert status("/", method="HEAD") == 200
    port = host.split(":")[1]
    assert status("/", host_header=f"localhost:{port}") == 200
    for bad in ("attacker.example:" + port, f"127.0.0.1:{int(port) + 1}", "127.0.0.1", "", "localhost"):
        assert status("/", host_header=bad) == 403, bad
        assert status("/events", host_header=bad) == 403, bad
    assert status("/", host_header=None) == 403
    feed.send(header(1))
    feed.send({"end": True, "frames": 0, "written": 0, "dropped": 0})
    running.join()


@pytest.mark.parametrize(
    "address", ["0.0.0.0", "::", "203.0.113.7", "198.51.100.1", "localhost", "example.com", "", "::ffff:127.0.0.1", "::1%lo0"]
)
def test_an_address_that_is_not_loopback_is_refused_before_a_socket_is_made(
    address: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_socket(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a socket was made")

    monkeypatch.setattr(radar, "RadarServer", no_socket)
    with pytest.raises(radar.RadarError):
        radar.serve(iter([]), host=address)


def test_the_command_refuses_an_address_that_is_not_loopback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    feed = tmp_path / "live.jsonl"
    feed.write_text(json.dumps(header(0)) + "\n", encoding="utf-8")
    assert cli.main(["radar", str(feed), "--host", "0.0.0.0"]) == 2
    assert "is not a loopback address" in capsys.readouterr().err
    assert cli.main(["radar", str(tmp_path / "missing.jsonl")]) == 2
    assert "is not there" in capsys.readouterr().err


def test_a_feed_that_is_not_a_live_output_is_refused() -> None:
    with pytest.raises(radar.RadarError, match="not a live output"):
        list(radar.feed_documents([json.dumps({"format": "pitchvlm-stream-2"})]))
    with pytest.raises(radar.RadarError, match="not JSON"):
        list(radar.feed_documents([json.dumps(header()), "{broken"]))
    with pytest.raises(radar.RadarError, match="neither a frame line"):
        list(radar.feed_documents([json.dumps(header()), json.dumps({"frame": True})]))
    with pytest.raises(radar.RadarError, match="not JSON"):
        list(radar.feed_documents([json.dumps(header()), '{"frame": 1, "x": NaN}']))


def test_a_line_cannot_add_an_event_to_the_stream() -> None:
    board = radar.Board()
    board.header(header())
    evil = frame(0)
    evil["clip"] = "x\n\nevent: end\ndata: {}\n\n"
    board.frame(evil)
    data = board.wait(0, 0.1)[1].data
    assert "\n" not in data and "\r" not in data and json.loads(data)["clip"] == evil["clip"]


def test_the_viewer_follows_a_file_that_grows(tmp_path: Path) -> None:
    path = tmp_path / "live.jsonl"
    path.write_text(json.dumps(header(3)) + "\n", encoding="utf-8")
    stop = threading.Event()
    running = Running(radar.follow_file(str(path), stop))
    response = urllib.request.urlopen(running.url + "events", timeout=10)
    with path.open("a", encoding="utf-8") as handle:
        for index in range(3):
            line = json.dumps(frame(index)) + "\n"
            handle.write(line[:20])
            handle.flush()
            time.sleep(0.1)  # half a line on the disk: the viewer waits for its end
            handle.write(line[20:])
            handle.flush()
            time.sleep(0.05)
        handle.write(json.dumps({"end": True, "frames": 3, "written": 3, "dropped": 0}) + "\n")
    events = list(read_events(response))
    assert [kind for kind, _, _ in events] == ["header", "frame", "frame", "frame", "end"]
    running.join()
    stop.set()


def test_the_radar_command_serves_a_stream_from_its_standard_input(tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    viewer = subprocess.Popen(
        [sys.executable, "-m", "pitchvlm.cli", "radar", "-"],
        cwd=REPO,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert viewer.stdout is not None and viewer.stdin is not None
        first = viewer.stdout.readline()
        match = re.match(r"radar: (\S+) ", first)
        assert match and match.group(1).startswith("http://127.0.0.1:"), first
        response = urllib.request.urlopen(match.group(1) + "events", timeout=10)
        for line in (header(2), frame(0), frame(1), {"end": True, "frames": 2, "written": 2, "dropped": 0}):
            viewer.stdin.write(json.dumps(line) + "\n")
            viewer.stdin.flush()
            time.sleep(0.05)
        kinds = [kind for kind, _, _ in read_events(response)]
        assert kinds == ["header", "frame", "frame", "end"]
        out, err = viewer.communicate(timeout=20)
    finally:
        if viewer.poll() is None:
            viewer.kill()
    assert viewer.returncode == 0, err
    assert "the feed ended: 2 frame lines, 1 event streams served" in out
    assert "Traceback" not in err


def test_the_number_of_event_streams_is_bounded() -> None:
    board = radar.Board()
    assert all(board.open_client() for _ in range(radar.MAX_CLIENTS))
    assert not board.open_client()
    board.close_client()
    assert board.open_client()


def test_a_page_of_another_site_is_refused_and_the_own_page_is_served() -> None:
    feed = Feed()
    running = Running(feed)
    host = running.url.removeprefix("http://").rstrip("/")

    def status(path: str, headers: dict[str, str]) -> int:
        connection = http.client.HTTPConnection(*host.split(":"), timeout=5)
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        response.read()
        connection.close()
        return response.status

    for path in ("/", "/events"):
        assert status(path, {"Origin": "https://github.com"}) == 403
        assert status(path, {"Sec-Fetch-Site": "cross-site"}) == 403
        assert status(path, {"Sec-Fetch-Site": "same-site"}) == 403
    assert status("/", {"Sec-Fetch-Site": "none"}) == 200
    feed.send(header(1))
    with urllib.request.urlopen(
        urllib.request.Request(running.url + "events", headers={"Origin": running.url.rstrip("/"), "Sec-Fetch-Site": "same-origin"}),
        timeout=10,
    ) as response:
        feed.send({"end": True, "frames": 0, "written": 0, "dropped": 0})
        assert [kind for kind, _, _ in read_events(response)] == ["header", "end"]
    running.join()
    assert running.result is not None and running.result[1].served == 1


def test_a_hostile_feed_gives_a_clean_refusal() -> None:
    taken: list[int] = []

    def endless() -> Iterator[bytes]:
        for index in range(1000):  # 64 MB without a line end
            taken.append(index)
            yield b"x" * 65536

    with pytest.raises(radar.RadarError, match="longer than"):
        list(radar.read_lines_of(endless()))
    assert len(taken) <= radar.MAX_LINE_BYTES // 65536 + 1, "the reader stops at the limit, it does not read on"
    with pytest.raises(radar.RadarError, match="not UTF-8"):
        list(radar.read_lines_of([b"\xff\xfe\n"]))
    assert list(radar.read_lines_of([b"ab", b"c\nde", b"f\n", b"g"])) == ["abc", "def", "g"]
    with pytest.raises(radar.RadarError, match="not JSON"):
        list(radar.feed_documents([json.dumps(header()), "[" * 100000 + "]" * 100000]))
    for t_s in (None, "1", -1.0, True):
        line = {**frame(0), "t_s": t_s}
        with pytest.raises(radar.RadarError, match="t_s"):
            list(radar.feed_documents([json.dumps(header()), json.dumps(line)]))


def test_a_number_too_large_for_a_float_is_refused_as_wrong_input_not_as_a_fault() -> None:
    """``1e400`` reads as an infinite float. In a time it was refused; anywhere else it reached the events, whose
    JSON has no infinity, and the command stopped with a ValueError and a traceback (a fault of the code)."""
    for field, value in (("t_s", "1e400"), ("delay_frames", "1e400"), ("x", "-1e999")):
        line = json.dumps({**frame(0), field: 0.123456}).replace("0.123456", value)
        with pytest.raises(radar.RadarError, match="too large"):
            list(radar.feed_documents([json.dumps(header()), line]))
    with pytest.raises(radar.RadarError, match="too large"):
        radar.serve([json.dumps(header()), json.dumps({**frame(0), "x": 0.123456}).replace("0.123456", "1e400")])
    # A whole number of 401 digits: math.isfinite raised OverflowError on it (second reader of round 8).
    for field in ("t_s", "frame", "x"):
        line = json.dumps({**frame(0), field: 0.123456}).replace("0.123456", "1" + "0" * 400)
        with pytest.raises(radar.RadarError, match="too large"):
            list(radar.feed_documents([json.dumps(header()), line]))
    assert next(iter(radar.feed_documents([json.dumps({**header(), "n_frames": 2**53})])))["n_frames"] == 2**53
    for value in (2**53 + 1, -(2**53) - 1):
        with pytest.raises(radar.RadarError, match="too large"):
            list(radar.feed_documents([json.dumps({**header(), "n_frames": value})]))


def test_the_viewer_stops_when_its_file_is_emptied_or_replaced(tmp_path: Path) -> None:
    """A second ``stream --live-out`` to the same path empties the file (or a new file takes its name). The viewer
    read on from its old place and waited for ever; now it stops with a refusal."""
    lines = json.dumps(header(3)) + "\n" + json.dumps(frame(0)) + "\n"
    for change in ("emptied", "replaced"):
        path = tmp_path / f"{change}.jsonl"
        path.write_text(lines, encoding="utf-8")
        stop = threading.Event()
        followed = radar.follow_file(str(path), stop, poll_s=0.01)
        assert [json.loads(next(followed))["format"], json.loads(next(followed))["frame"]] == [LIVE_FORMAT, 0]
        if change == "emptied":
            path.write_text(json.dumps(header(3)) + "\n", encoding="utf-8")  # shorter than what was read
        else:
            (tmp_path / "new.jsonl").write_text(lines + json.dumps(frame(1)) + "\n", encoding="utf-8")
            os.replace(tmp_path / "new.jsonl", path)
        stopper = threading.Timer(10.0, stop.set)  # the old code waited here until this stop
        stopper.start()
        try:
            with pytest.raises(radar.RadarError, match=change):
                next(followed)
        finally:
            stopper.cancel()
            stop.set()


def test_connections_up_to_the_limit_that_come_while_the_server_is_late_wait_for_their_accept() -> None:
    """The loop of the server accepts nothing here (a busy machine). Connections up to ``MAX_CONNECTIONS`` wait in the
    queue of the system; with a queue of 16 the system reset the others (macOS) or let them hang (Linux), and the test
    of the limit failed at a load of 58."""
    server = radar.RadarServer(radar.check_loopback("127.0.0.1"), 0, radar.Board())
    held: list[socket.socket] = []
    try:
        for _ in range(radar.MAX_CONNECTIONS):
            held.append(socket.create_connection(("127.0.0.1", server.port), timeout=2))
    finally:
        for sock in held:
            sock.close()
        server.server_close()
    assert len(held) == radar.MAX_CONNECTIONS


def test_connections_above_the_limit_are_closed_at_once() -> None:
    feed = Feed()
    running = Running(feed)
    host, port = running.url.removeprefix("http://").rstrip("/").split(":")
    held = [socket.create_connection((host, int(port)), timeout=5) for _ in range(radar.MAX_CONNECTIONS)]
    try:
        for sock in held:
            sock.sendall(b"GET / HTTP/1.1\r\n")  # a request that never ends holds its connection
        time.sleep(0.3)
        with socket.create_connection((host, int(port)), timeout=5) as extra:
            extra.sendall(f"GET / HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode())
            try:
                answer = extra.recv(100)
            except ConnectionResetError:
                answer = b""
            assert answer == b"", "a connection above the limit is closed without an answer"
    finally:
        for sock in held:
            sock.close()
    time.sleep(0.3)
    page = urllib.request.urlopen(running.url, timeout=5)
    assert page.status == 200
    feed.send(header(1))
    feed.send({"end": True, "frames": 0, "written": 0, "dropped": 0})
    running.join()


def test_a_replay_gives_each_frame_at_its_time_from_the_first_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--replay`` waits for the first page, then gives frame ``f`` at ``t_s`` of ``f`` after the first frame
    (a planted bug that gave every frame at once was caught by no test). The clock of the test moves only when the
    board sleeps."""
    now = [100.0]
    slept: list[float] = []
    board = radar.Board(replay=True)

    def sleep(seconds: float) -> None:
        slept.append(round(seconds, 9))
        now[0] += seconds

    monkeypatch.setattr(board, "sleep", sleep)
    board.first_client.set()
    lines = [json.dumps(header(4))] + [json.dumps({**frame(index), "t_s": index * 0.5}) for index in range(4)]
    assert radar.pump([*lines, json.dumps({"end": True})], board, replay=True, clock=lambda: now[0]) == 4
    assert slept == [0.5, 0.5, 0.5]
    assert [event.frame for event in board.wait(1, 0.0) if event.kind == "frame"] == [0, 1, 2, 3]
