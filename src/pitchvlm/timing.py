"""Measured time per stage of a run: seconds, milliseconds per frame and frames per second.

The run report holds these values so that a claim about speed rests on a measurement. A
stage of the package is timed where it runs (``StageClock.stage``). The time of an external
tool comes from its clip run file, which the script measured per frame (``StageClock.add``).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager


class StageClock:
    """Adds up the time of named stages. A stage that runs twice adds both times."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._seconds: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time the code of a ``with`` block as the stage ``name``. The time counts also when the block raises."""
        start = self._clock()
        try:
            yield
        finally:
            self.add(name, self._clock() - start)

    def add(self, name: str, seconds: float) -> None:
        """Add a time that was measured elsewhere to the stage ``name``. A negative time raises ``ValueError``."""
        if not seconds >= 0.0:
            raise ValueError(f"the time of the stage {name!r} must be a number of 0 or more, got {seconds}")
        self._seconds[name] = self._seconds.get(name, 0.0) + seconds

    def report(self, n_frames: int) -> dict[str, dict[str, float | None]]:
        """Return the time of each stage, in the order of their first use, and of all stages together.

        Each entry holds ``seconds``, ``ms_per_frame`` and ``frames_per_second`` over the
        ``n_frames`` frames of the clip. ``frames_per_second`` is None for a stage that took no
        measurable time. ``ValueError`` is raised for ``n_frames`` below 1.
        """
        if n_frames < 1:
            raise ValueError(f"a clip has at least one frame, got {n_frames}")
        stages = {**self._seconds, "total": sum(self._seconds.values())}
        return {
            name: {
                "seconds": seconds,
                "ms_per_frame": 1000.0 * seconds / n_frames,
                "frames_per_second": n_frames / seconds if seconds > 0.0 else None,
            }
            for name, seconds in stages.items()
        }
