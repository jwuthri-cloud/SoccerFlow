"""Measured time per stage of a run: seconds, milliseconds per frame and frames per second.

The run report holds these values so that a claim about speed rests on a measurement. A
stage of the package is timed where it runs (``StageClock.stage``). The time of an external
tool comes from its clip run file, which the script measured per frame (``StageClock.add``).

A mean hides stalls. The streaming path (``stream``) handles one frame at a time inside
``StageClock.frame``; each such block is one sample, and ``frame_report`` gives the median,
the 95th percentile and the largest time of each stage and of the whole block over the samples.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager

import numpy as np

#: Name of the time of a whole frame block in ``frame_report``.
FRAME_TOTAL = "total"


class StageClock:
    """Adds up the time of named stages. A stage that runs twice adds both times."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._seconds: dict[str, float] = {}
        #: Stage times of the frame block that is open, or None outside a block.
        self._frame: dict[str, float] | None = None
        #: One entry per closed frame block: the stage times of the block and its whole time (``FRAME_TOTAL``).
        self._samples: list[dict[str, float]] = []

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
        if self._frame is not None:
            self._frame[name] = self._frame.get(name, 0.0) + seconds

    @contextmanager
    def frame(self) -> Iterator[None]:
        """Time the code of a ``with`` block as one frame sample (``frame_report``). Blocks do not nest.

        The stages timed inside the block count for the sample, and the whole time of the block
        is the sample's ``FRAME_TOTAL``. The sample counts also when the block raises.
        """
        if self._frame is not None:
            raise ValueError("a frame block is already open")
        self._frame = {}
        start = self._clock()
        try:
            yield
        finally:
            sample, self._frame = self._frame, None
            sample[FRAME_TOTAL] = self._clock() - start
            self._samples.append(sample)

    def frame_report(self) -> dict[str, dict[str, float]]:
        """Return the per-frame times of each stage and of the whole block, over all frame samples.

        Each entry holds ``n`` (the number of samples), and ``p50_ms``, ``p95_ms`` and ``max_ms``,
        milliseconds per frame (``numpy.percentile``, linear). A stage that did not run in a
        sample counts 0 there, so the median of a stage that runs in one frame of many is 0.
        The stages come in the order of their first use, and ``FRAME_TOTAL`` last. The answer is
        empty when no frame block ran.
        """
        names = [name for sample in self._samples for name in sample if name != FRAME_TOTAL]
        ordered = [*dict.fromkeys(names), FRAME_TOTAL] if self._samples else []
        return {name: percentiles_ms([sample.get(name, 0.0) for sample in self._samples]) for name in ordered}

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


def percentiles_ms(seconds: Sequence[float]) -> dict[str, float]:
    """Return ``n``, ``p50_ms``, ``p95_ms`` and ``max_ms`` of some times in seconds (``numpy.percentile``, linear).

    ``ValueError`` is raised for no time.
    """
    if not seconds:
        raise ValueError("percentiles need at least one time")
    values = 1000.0 * np.array(seconds, dtype=np.float64)
    return {
        "n": float(len(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(values.max()),
    }
