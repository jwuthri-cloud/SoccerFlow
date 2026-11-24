"""Stop signals that wait until the paid requests in flight have ended (``stopsignals``, ``cli._interrupt``).

No model is called: the requests go to a fake model.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from pitchvlm import cli, pipeline, stopsignals

SRC = Path(__file__).resolve().parents[1] / "src"

#: A child process: the wait of the thread pool of the pipeline under the stop handler of the command line.
#: Four requests reach the fake model; the child prints "ready", the test sends its signals, then a line on
#: the standard input lets the model answer 200 ms later. The child prints how many answers came after the
#: wait returned ("late").
BURST_CHILD = r"""
import faulthandler, json, signal, sys, threading, time
from pitchvlm import cli, pipeline

faulthandler.dump_traceback_later(30, exit=True)
lock = threading.Lock()
reached, answered, late = [], [], []
returned, release = threading.Event(), threading.Event()


class Model:
    def generate(self, request):
        with lock:
            reached.append(request)
            if len(reached) == 4:
                print("ready", flush=True)
        release.wait(30.0)
        time.sleep(0.2)
        with lock:
            answered.append(request)
            if returned.is_set():
                late.append(request)
        return request


def go():
    sys.stdin.readline()
    release.set()


for number in cli.STOP_SIGNALS:
    signal.signal(number, cli._interrupt)
threading.Thread(target=go, daemon=True).start()
outcome = "returned"
try:
    pipeline._generate_all(Model(), list(range(12)), 4)
except KeyboardInterrupt as stop:
    outcome = "interrupted"
while True:
    try:
        returned.set()
        for number in cli.STOP_SIGNALS:
            signal.signal(number, signal.SIG_IGN)
        break
    except KeyboardInterrupt:
        continue
time.sleep(0.6)
with lock:
    print(json.dumps({"reached": len(reached), "answered": len(answered), "late": len(late), "outcome": outcome}))
"""


@pytest.fixture(autouse=True)
def fresh_holds() -> None:
    stopsignals.reset()
    cli.STOP_REQUESTED.clear()
    yield
    stopsignals.reset()
    cli.STOP_REQUESTED.clear()


def test_a_burst_of_stop_signals_from_another_process_does_not_end_the_wait_for_the_requests_in_flight() -> None:
    """2,000 stop signals (SIGINT and SIGTERM in turn) from this process, while four paid requests are at the model.

    The wait returns only after the four answers, by the interruption of the first signal, and no other request
    is sent. On the code of round 7 every signal raised in the main thread, also inside the handler of the one
    before, so the wait ended early or the handler blocked for ever in the lock of its stop event (4 of 4 runs;
    docs/DECISIONS.md, entry "Second read of the live output, and stop signals that cannot end the wait for
    paid requests"). Three bursts; each child ends within its 30 s or is stopped by faulthandler.
    """
    kinds = (signal.SIGINT, signal.SIGTERM)
    for _ in range(3):
        child = subprocess.Popen(
            [sys.executable, "-c", BURST_CHILD],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONPATH": str(SRC)},
        )
        try:
            assert child.stdout is not None and child.stdin is not None
            assert child.stdout.readline().strip() == "ready"
            for index in range(2000):
                os.kill(child.pid, kinds[index % 2])
            child.stdin.write("go\n")
            child.stdin.flush()
            out, err = child.communicate(timeout=60)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
        lines = out.strip().splitlines()
        assert lines and lines[-1].startswith("{"), f"the child gave no record (it hung): {err[-3000:]}"
        record = json.loads(lines[-1])
        assert record == {"reached": 4, "answered": 4, "late": 0, "outcome": "interrupted"}


def test_a_stop_signal_inside_the_handler_is_counted_and_raises_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A signal that comes while the handler sets the stop event: the inner handler returns, the outer one finishes
    the set and raises once. Raised inside the set, the interruption would leave the lock of the event taken."""
    calls: list[str] = []

    class EventWithASignalInside:
        def set(self) -> None:
            calls.append("set begins")
            if len(calls) == 1:
                cli._interrupt(signal.SIGTERM, None)  # the next signal of a burst
            calls.append("set ends")

        def is_set(self) -> bool:
            return True

    monkeypatch.setattr(cli, "STOP_REQUESTED", EventWithASignalInside())
    with pytest.raises(KeyboardInterrupt, match="Ctrl-C"):
        cli._interrupt(signal.SIGINT, None)
    assert calls == ["set begins", "set ends"]
    assert stopsignals.held_signals() == 1


def test_a_hold_records_any_number_of_signals_and_raises_the_first_when_it_ends() -> None:
    with pytest.raises(KeyboardInterrupt, match="SIGTERM"):
        with stopsignals.held_stops():
            for index in range(2000):
                cli._interrupt(signal.SIGTERM if index % 2 == 0 else signal.SIGINT, None)
            assert stopsignals.held_cause() == "SIGTERM"
            assert cli.STOP_REQUESTED.is_set()
    assert stopsignals.held_signals() == 2000
    assert stopsignals.held_cause() is None
    # Outside a hold a signal interrupts at once.
    with pytest.raises(KeyboardInterrupt, match="Ctrl-C"):
        cli._interrupt(signal.SIGINT, None)


def test_holds_nest_and_hold_nothing_outside_the_main_thread() -> None:
    with pytest.raises(KeyboardInterrupt, match="SIGHUP"):
        with stopsignals.held_stops():
            with stopsignals.held_stops():
                assert stopsignals.hold_back("SIGHUP")
            # The inner hold ended; the outer one still holds the stop.
            assert stopsignals.held_cause() == "SIGHUP"
    seen: list[bool] = []

    def other_thread() -> None:
        with stopsignals.held_stops():
            seen.append(stopsignals.hold_back("SIGTERM"))

    thread = threading.Thread(target=other_thread)
    thread.start()
    thread.join()
    assert seen == [False]


def test_the_stop_event_can_be_set_by_the_thread_that_holds_its_lock() -> None:
    """The handler runs in the main thread, also while that thread is inside ``wait`` of the event (a client that
    waits before a new attempt). The lock of ``threading.Event`` would block the handler there for ever."""
    event = stopsignals.StopEvent()
    assert isinstance(event, threading.Event)
    finished = threading.Event()

    def set_inside_the_lock() -> None:
        with event._stop_cond:
            event.set()
        finished.set()

    thread = threading.Thread(target=set_inside_the_lock, daemon=True)
    thread.start()
    assert finished.wait(timeout=10.0)
    assert event.is_set() and event.wait(0.0)
    event.clear()
    assert not event.is_set() and not event.wait(0.01)
    waiter = threading.Thread(target=lambda: event.wait(10.0))
    waiter.start()
    event.set()
    waiter.join(timeout=10.0)
    assert not waiter.is_alive()


def test_no_request_starts_after_a_stop_that_a_hold_recorded() -> None:
    """The thread pool sends no request once a stop is recorded, so the wait ends with the requests in flight."""
    sent: list[int] = []
    gate = threading.Event()

    class Model:
        def generate(self, request: int) -> int:
            sent.append(request)
            gate.wait(10.0)
            return request

    def stop_then_release() -> None:
        while len(sent) < 2:
            threading.Event().wait(0.01)
        stopsignals._HOLD.cause = "Ctrl-C"  # what the handler records in the main thread
        stopsignals._HOLD.signals += 1
        gate.set()

    helper = threading.Thread(target=stop_then_release)
    helper.start()
    with pytest.raises(KeyboardInterrupt, match="Ctrl-C"):
        pipeline._generate_all(Model(), list(range(10)), 2)  # type: ignore[arg-type]
    helper.join()
    assert sorted(sent) == [0, 1]
