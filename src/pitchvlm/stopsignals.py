"""Stop signals that wait until the paid model requests in flight have ended.

A stop signal (Ctrl-C, a plain kill, a closed terminal) raises ``KeyboardInterrupt`` in the
main thread through the handler of the command line (``cli._interrupt``). Raised while the
answers of paid requests are still to come, it would throw them away: the provider bills them,
but they are neither stored nor recorded. So the code that waits for such requests holds the
signals back (``held_stops``): while it runs in the main thread, the handler only records the
stop (``hold_back``) and returns, however many signals come. The code reads the record
(``held_signals``) and sends no new request; when the hold ends, the first recorded stop is
raised as ``KeyboardInterrupt`` with its cause.

Raised at any point, a ``KeyboardInterrupt`` could also leave a lock taken in the main thread,
and the next stop signal then blocked its own handler for ever (measured: a burst of 2,000
signals from another process blocked the handler inside ``threading.Event.set`` in 3 of 4 runs
of the code of round 7; ``docs/DECISIONS.md``, entry "Second read of the live output, and stop
signals that cannot end the wait for paid requests", 2026-11-23). Hence two more rules:

- the event that tells the model clients to send nothing more (``StopEvent``) takes a lock
  that the same thread can take again, so its handler can set it while the main thread is
  inside one of its own methods;
- a signal that comes while the handler runs only records its cause (``enter_handler``).

The way to end a wait that must not wait any longer (a request that never returns) is a
signal that Python does not handle: Ctrl-\\ (SIGQUIT) or ``kill -KILL``. The process ends at
once; the answers in flight are lost, but the provider can still bill them, and the cost
ledger has no line for them (``docs/RUNBOOK.md``, section 2).

Only the main thread holds signals back: a handler of Python runs there and nowhere else.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class _Hold:
    """The state of the holds; only the main thread reads and writes it (the handler runs there too)."""

    #: Holds that are open now (they nest).
    depth = 0
    #: The cause of the first stop signal that a hold recorded, or None.
    cause: str | None = None
    #: Stop signals recorded since the last ``reset`` (also those that came while the handler ran).
    signals = 0
    #: True while the handler runs (``enter_handler``).
    in_handler = False


_HOLD = _Hold()


def _in_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


def reset() -> None:
    """Forget every hold and every recorded stop: a command starts (``cli.main``)."""
    _HOLD.depth, _HOLD.cause, _HOLD.signals, _HOLD.in_handler = 0, None, 0, False


@contextmanager
def held_stops() -> Iterator[None]:
    """Hold stop signals back while the block runs in the main thread; raise the first one when it ends.

    In any other thread the block runs as it is: no signal is raised there. Holds nest; the
    recorded stop is raised when the outermost one ends, also when the block raised an error
    (the stop replaces it).
    """
    if not _in_main_thread():
        yield
        return
    _HOLD.depth += 1
    try:
        yield
    finally:
        _HOLD.depth -= 1
        if _HOLD.depth == 0 and _HOLD.cause is not None:
            cause, _HOLD.cause = _HOLD.cause, None
            raise KeyboardInterrupt(cause)


def hold_back(cause: str) -> bool:
    """Record a stop signal when a hold is open, and return whether it was held back (the handler then returns)."""
    if _HOLD.depth == 0:
        return False
    _HOLD.signals += 1
    if _HOLD.cause is None:
        _HOLD.cause = cause
    return True


def held_cause() -> str | None:
    """Return the cause of the stop that the open hold recorded, or None (no stop yet)."""
    return _HOLD.cause


def held_signals() -> int:
    """Return the number of stop signals recorded since the command started."""
    return _HOLD.signals


def enter_handler() -> bool:
    """Mark the start of the stop handler, or return False when it already runs (a signal inside the handler)."""
    if _HOLD.in_handler:
        _HOLD.signals += 1
        return False
    _HOLD.in_handler = True
    return True


def leave_handler() -> None:
    """Mark the end of the stop handler."""
    _HOLD.in_handler = False


class StopEvent(threading.Event):
    """A ``threading.Event`` whose lock the same thread can take again.

    The stop handler sets it in the main thread, at any point of the code of that thread, also
    while the main thread is inside ``wait`` of this event (a model client that waits before a
    new attempt). The lock of ``threading.Event`` cannot be taken twice by one thread: the
    handler would block there for ever.
    """

    def __init__(self) -> None:
        super().__init__()
        self._stop_cond = threading.Condition(threading.RLock())
        self._stop_flag = False

    def is_set(self) -> bool:
        return self._stop_flag

    def set(self) -> None:
        with self._stop_cond:
            self._stop_flag = True
            self._stop_cond.notify_all()

    def clear(self) -> None:
        with self._stop_cond:
            self._stop_flag = False

    def wait(self, timeout: float | None = None) -> bool:
        with self._stop_cond:
            if not self._stop_flag:
                self._stop_cond.wait(timeout)
            return self._stop_flag
