"""Tests of the measured time per stage (``timing``)."""

from __future__ import annotations

import pytest

from pitchvlm.timing import StageClock


class FakeClock:
    """A clock that moves on by the given steps, one step per reading."""

    def __init__(self, *steps: float) -> None:
        self.now = 0.0
        self._steps = list(steps)

    def __call__(self) -> float:
        value = self.now
        if self._steps:
            self.now += self._steps.pop(0)
        return value


def test_stages_add_up_and_report_per_frame() -> None:
    clock = StageClock(FakeClock(2.0, 0.0, 1.0, 0.0, 0.5))
    with clock.stage("motion"):
        pass
    with clock.stage("tracks"):
        pass
    with clock.stage("motion"):
        pass
    clock.add("detector", 1.5)
    report = clock.report(100)
    assert list(report) == ["motion", "tracks", "detector", "total"]
    assert report["motion"] == {"seconds": 2.5, "ms_per_frame": 25.0, "frames_per_second": 40.0}
    assert report["total"]["seconds"] == pytest.approx(5.0)
    assert report["total"]["frames_per_second"] == pytest.approx(20.0)


def test_stage_that_raises_still_counts() -> None:
    clock = StageClock(FakeClock(3.0))
    with pytest.raises(RuntimeError), clock.stage("number"):
        raise RuntimeError("model failed")
    assert clock.report(1)["number"]["seconds"] == 3.0


def test_stage_without_time_has_no_frame_rate() -> None:
    clock = StageClock(FakeClock())
    clock.add("ball", 0.0)
    assert clock.report(10)["ball"] == {"seconds": 0.0, "ms_per_frame": 0.0, "frames_per_second": None}


@pytest.mark.parametrize("seconds", [-0.1, float("nan")])
def test_negative_or_missing_time_is_refused(seconds: float) -> None:
    with pytest.raises(ValueError):
        StageClock().add("detector", seconds)


def test_report_needs_a_frame() -> None:
    with pytest.raises(ValueError):
        StageClock().report(0)
