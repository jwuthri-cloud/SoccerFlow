"""Tests of the frame feeder and of the overload policy of the streaming path (``stream.FrameFeeder``,
``stream.OverloadPolicy``) and of the pair meter they use (``motion.PairMeter``).

Every clock here is injected: no test depends on the speed of the machine. The tools are the
stand-ins of ``fallbacks.fake`` on the synthetic scene. Nothing here says how a real detector,
a real calibration or a live pipe behave.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pitchvlm import cli
from pitchvlm.config import EngineSettings, PipelineConfig
from pitchvlm.evalx.streambench import ArraySource, clip_of
from pitchvlm.fallbacks.fake import fake_calibration_answers, fake_detector_answers
from pitchvlm.liveout import LiveOut
from pitchvlm.motion import PairMeter, ShotMeter, _block_correlation, _correlations, _moved, measure_shots
from pitchvlm.schemas import ClipResult, FrameState, HomographyRecord
from pitchvlm.stream import (
    DetectorStream,
    FedFrame,
    FrameFeeder,
    OverloadPolicy,
    StreamInput,
    StreamSettings,
    _StreamMotion,
    default_history_frames,
    stand_in_inputs,
)
from pitchvlm.synth import SyntheticScene

REPO = Path(__file__).resolve().parents[1]
CONFIG = PipelineConfig(engines=EngineSettings(people="detector", pitch_mapping="calibration", number="none"))
N_FRAMES = 60
SETTINGS = StreamSettings(delay_frames=5, history_frames=default_history_frames(CONFIG, 25.0))


def inputs_of(scene: SyntheticScene) -> list[StreamInput]:
    people, balls = fake_detector_answers(scene)
    return list(stand_in_inputs(people, balls, fake_calibration_answers(scene)))


def new_stream(scene: SyntheticScene, settings: StreamSettings = SETTINGS) -> DetectorStream:
    info = scene.info
    return DetectorStream(CONFIG, settings, clip_id=info.clip_id, fps=info.fps, size=(info.width, info.height))


def pushed(scene: SyntheticScene, images: Sequence[np.ndarray], settings: StreamSettings = SETTINGS) -> ClipResult:
    """The stream as before the feeder: push every frame with its picture, then finish."""
    stream = new_stream(scene, settings)
    for item in inputs_of(scene):
        stream.push(item, images[item.frame])
    stream.finish()
    return stream.result()


def fed(
    scene: SyntheticScene,
    images: Sequence[np.ndarray],
    settings: StreamSettings = SETTINGS,
    *,
    policy: OverloadPolicy | None = None,
    line_clock: Callable[[], tuple[float | None, int]] | None = None,
    inputs: Iterator[StreamInput] | None = None,
    on_dropped: Callable[[int], None] | None = None,
    on_given: Callable[..., None] | None = None,
) -> tuple[DetectorStream, ClipResult]:
    """The stream through a frame feeder."""
    stream = new_stream(scene, settings)
    stream.on_dropped = on_dropped
    stream.on_given = on_given
    feeder = FrameFeeder(
        stream,
        inputs if inputs is not None else iter(inputs_of(scene)),
        lambda frame: images[frame],
        policy=policy,
        line_clock=line_clock,
    )
    try:
        for frame in feeder:
            stream.push_fed(frame)
    finally:
        feeder.close()
    stream.finish()
    return stream, stream.result()


def as_json(result: ClipResult) -> str:
    return json.dumps(result.model_dump(mode="json"), sort_keys=True)


class Clock:
    """A clock that the test sets."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


# --- The pair meter: the same tests on another thread ---------------------------------------------


@pytest.mark.parametrize("kind", ["plain", "cut", "dissolve"])
def test_the_pair_meter_and_take_give_the_tests_of_add(kind: str) -> None:
    scene, images = clip_of(kind, 5, n_frames=N_FRAMES)
    people, _ = fake_detector_answers(scene)
    size = (scene.info.width, scene.info.height)
    added = ShotMeter(size)
    taken = ShotMeter(size)
    pairs = taken.hand_out_pairs()
    for item in people:
        boxes = [person.box for person in item.people]
        added.add(images[item.frame], boxes)
        taken.take(pairs.measure(images[item.frame], boxes))
    first, second = added.finish(), taken.finish()
    assert first.hard_cuts == second.hard_cuts
    assert first.transitions == second.transitions
    assert first.pair_correlations == second.pair_correlations
    assert first.window_correlations == second.window_correlations
    for one, other in zip(first.steps, second.steps, strict=True):
        assert (one is None and other is None) or np.array_equal(one, other)
    expected = measure_shots(ArraySource(scene.info, images), people={item.frame: [p.box for p in item.people] for item in people})
    assert expected.hard_cuts == first.hard_cuts and expected.transitions == first.transitions


def test_a_meter_that_handed_out_its_pair_tests_refuses_add_and_a_pair_test_out_of_order() -> None:
    scene = SyntheticScene(n_frames=3, seed=4)
    meter = ShotMeter((scene.info.width, scene.info.height))
    pairs = meter.hand_out_pairs()
    with pytest.raises(ValueError, match="handed out"):
        meter.add(scene.image(0))
    with pytest.raises(ValueError, match="only a meter without a frame"):
        meter.hand_out_pairs()
    first = pairs.measure(scene.image(0))
    second = pairs.measure(scene.image(1))
    with pytest.raises(ValueError, match="frame 1 came, frame 0 was expected"):
        meter.take(second)
    meter.take(first)
    meter.take(second)
    with pytest.raises(ValueError, match="the first frame needs a picture"):
        PairMeter((scene.info.width, scene.info.height)).repeat()


def test_a_repeated_frame_is_the_identity_and_no_cut_and_the_next_pair_is_measured_from_the_frame_before() -> None:
    scene = SyntheticScene(n_frames=4, seed=4)
    size = (scene.info.width, scene.info.height)
    meter = ShotMeter(size)
    pairs = meter.hand_out_pairs()
    meter.take(pairs.measure(scene.image(0)))
    meter.take(pairs.repeat())
    meter.take(pairs.measure(scene.image(2)))
    assert np.array_equal(meter.steps[0], np.eye(3)) and meter.hard_cuts == []
    assert meter.images[1] is meter.images[0]
    # The step into frame 2 is the motion from the picture of frame 0, as two frames of the clip.
    reference = ShotMeter(size)
    reference.add(scene.image(0))
    reference.add(scene.image(2))
    assert np.array_equal(meter.steps[1], reference.steps[0])


def test_the_kept_blocks_of_the_target_give_the_same_correlations() -> None:
    scene = SyntheticScene(n_frames=2, seed=6)
    meter = ShotMeter((scene.info.width, scene.info.height))
    meter.add(scene.image(0))
    meter.add(scene.image(1))
    first, second = meter.images
    shift = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]])
    matrices = [np.eye(3), shift, np.linalg.inv(shift)]
    assert _correlations(first, second, matrices) == [_block_correlation(_moved(first, m), second) for m in matrices]


def test_a_kept_motion_chain_is_dropped_with_new_cuts_and_forgotten_steps() -> None:
    motion = _StreamMotion()
    step = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    for index in range(4):
        motion.add_frame(None if index == 0 else step)
    chained = motion.between(0, 3)
    assert chained is not None and np.allclose(chained[0, 2], 6.0)
    assert motion.between(0, 3) is chained
    motion.cuts = (2,)
    assert motion.between(0, 3) is None
    motion.cuts = ()
    assert np.array_equal(motion.between(0, 3), chained)
    motion.forget_before(1)
    assert motion.between(0, 3) is None
    assert motion.between(1, 3) is not None


# --- The feeder: the result of push ------------------------------------------------------------


@pytest.mark.parametrize("kind", ["plain", "cut", "dissolve"])
def test_the_feeder_gives_the_result_of_push(kind: str) -> None:
    scene, images = clip_of(kind, 7, n_frames=N_FRAMES)
    stream, result = fed(scene, images)
    assert as_json(result) == as_json(pushed(scene, images))
    assert stream.dropped_frames == ()
    assert "overload" not in result.engines


def test_a_stream_takes_frames_from_its_feeder_or_from_push_never_both() -> None:
    scene = SyntheticScene(n_frames=3, seed=8)
    items = inputs_of(scene)
    stream = new_stream(scene)
    with pytest.raises(ValueError, match="push_fed takes the frames of a frame feeder"):
        stream.push_fed(FedFrame(item=items[0], image=scene.image(0), measure=None))
    feeder = FrameFeeder(stream, iter(items), scene.image)
    try:
        with pytest.raises(ValueError, match="give its frames to push_fed"):
            stream.push(items[0], scene.image(0))
        frames = list(feeder)
    finally:
        feeder.close()
    assert [frame.item.frame for frame in frames] == [0, 1, 2]
    stream.push_fed(frames[0])
    with pytest.raises(ValueError, match="frame 1 was expected, got frame 2"):
        stream.push_fed(frames[2])


def test_an_error_of_a_picture_is_raised_at_its_frame_after_the_frames_before() -> None:
    scene = SyntheticScene(n_frames=6, seed=9)
    stream = new_stream(scene)

    def read(frame: int) -> np.ndarray:
        if frame == 3:
            raise OSError("cannot read image 4")
        return scene.image(frame)

    feeder = FrameFeeder(stream, iter(inputs_of(scene)), read)
    seen: list[int] = []
    try:
        with pytest.raises(OSError, match="cannot read image 4"):
            for frame in feeder:
                stream.push_fed(frame)
                seen.append(frame.item.frame)
    finally:
        feeder.close()
    assert seen == [0, 1, 2]


def test_an_input_out_of_order_is_refused_by_the_stream_with_the_message_of_push() -> None:
    scene = SyntheticScene(n_frames=4, seed=10)
    items = inputs_of(scene)
    stream = new_stream(scene)
    feeder = FrameFeeder(stream, iter([items[0], items[2]]), scene.image)
    try:
        frames = iter(feeder)
        stream.push_fed(next(frames))
        with pytest.raises(ValueError, match="frame 1 was expected, got frame 2"):
            stream.push_fed(next(frames))
    finally:
        feeder.close()


def test_close_stops_a_feeder_whose_frames_are_not_taken() -> None:
    scene = SyntheticScene(n_frames=30, seed=11)
    stream = new_stream(scene)
    feeder = FrameFeeder(stream, iter(inputs_of(scene)), scene.image, ahead=1)
    next(iter(feeder))
    feeder.close(wait_s=5.0)
    assert not any(thread.is_alive() for thread in feeder._threads)


# --- The overload policy, with the clock of the test --------------------------------------------


def test_the_policy_drops_a_frame_above_half_its_setting_and_the_decision_due_above_the_setting() -> None:
    clock = Clock()
    policy = OverloadPolicy(1.0, clock=clock)
    assert policy.unused_after_s == 0.5
    clock.now = 10.0
    assert policy.drop(0, read_at=1.0) is False  # frame 0 starts the pair tests
    assert policy.drop(1, read_at=9.5) is False  # waited exactly 0.5 s
    assert policy.drop(2, read_at=9.4) is True
    assert policy.waits_s == [9.0, 0.5, pytest.approx(0.6)]
    # The second look, on the frame path, has its own order.
    clock.now = 10.3
    assert policy.late(0, read_at=1.0) == (False, False)
    assert policy.late(1, read_at=9.5) == (True, False)  # 0.8 s by now: not used, its decision due is kept
    assert policy.late(2, read_at=9.2) == (True, True)  # 1.1 s: the decision due with it is dropped too
    assert policy.late_waits_s == [pytest.approx(9.3), pytest.approx(0.8), pytest.approx(1.1)]


def test_the_policy_fails_closed_without_a_clock_or_a_setting_and_takes_frames_in_order() -> None:
    policy = OverloadPolicy(0.5, clock=Clock())
    with pytest.raises(ValueError, match="no time at which it was read"):
        policy.drop(0, read_at=None)
    with pytest.raises(ValueError, match="no time at which it was read"):
        policy.late(0, read_at=None)
    with pytest.raises(ValueError, match="expected frame 0, got frame 1"):
        policy.drop(1, read_at=0.0)
    with pytest.raises(ValueError, match="expected frame 0 on the frame path, got frame 1"):
        policy.late(1, read_at=0.0)
    for wrong in (0.0, -1.0, float("inf"), float("nan"), True):
        with pytest.raises(ValueError, match="more than 0 seconds"):
            OverloadPolicy(wrong)  # type: ignore[arg-type]
    scene = SyntheticScene(n_frames=2, seed=12)
    with pytest.raises(ValueError, match="needs the time at which each line was read"):
        FrameFeeder(new_stream(scene), iter(inputs_of(scene)), scene.image, policy=OverloadPolicy(0.5))


def scripted_waits(waits: Sequence[float]) -> tuple[Clock, Callable[[], tuple[float | None, int]], Callable[..., Iterator[StreamInput]]]:
    """A clock that stands still and a queue of lines whose line of frame ``i`` waited ``waits[i]`` seconds.

    Both looks of the policy see the same wait, whatever the threads do: the test does not
    depend on the speed of the machine.
    """
    clock = Clock()
    clock.now = 1000.0
    taken = {"frame": -1}

    def line_clock() -> tuple[float | None, int]:
        return clock.now - waits[taken["frame"]], 0

    def inputs(items: Sequence[StreamInput]) -> Iterator[StreamInput]:
        for item in items:
            taken["frame"] = item.frame
            yield item

    return clock, line_clock, inputs


def test_a_dropped_frame_has_no_measurement_and_a_very_late_line_drops_the_decision_due_with_it() -> None:
    scene, images = clip_of("plain", 13, n_frames=N_FRAMES)
    late = {20, 21, 22, 40}
    unused = {30}  # waited more than half the setting: not used, but the decision of frame 25 is kept
    delay = SETTINGS.delay_frames
    assert delay == 5
    decided_late = {frame - delay for frame in late} - late
    waits = [2.0 if frame in late else 0.7 if frame in unused else 0.01 for frame in range(N_FRAMES)]
    late = late | unused
    clock, line_clock, inputs = scripted_waits(waits)
    told: list[int] = []
    given: list[int] = []
    stream, result = fed(
        scene,
        images,
        policy=OverloadPolicy(1.0, clock=clock),
        line_clock=line_clock,
        inputs=inputs(inputs_of(scene)),
        on_dropped=told.append,
        on_given=lambda state, **_: given.append(state.frame),
    )
    expected = sorted(late | decided_late)
    assert stream.dropped_frames == tuple(expected) and told == expected
    report = stream.overload_report()
    assert report["not_used"] == sorted(late) and report["decided_late"] == sorted(decided_late)
    assert report["dropped"] == len(expected)
    assert sorted(set(given) | set(told)) == list(range(N_FRAMES)) and not set(given) & set(told)
    for frame in expected:
        assert result.frames[frame] == FrameState(frame=frame, players=(), ball=None)
        assert result.homographies[frame] == HomographyRecord(frame=frame)
    assert result.engines["overload"] == (
        "drop a frame whose line waited more than 0.5 s, and a decision due with a line that waited more than 1 s"
    )
    # Every other frame is measured and decided: it has observed players.
    assert all(
        any(player.observed for player in result.frames[frame].players)
        for frame in range(N_FRAMES)
        if frame not in expected
    )


def test_with_the_policy_on_and_no_line_late_the_result_is_that_of_the_stream_without_it() -> None:
    scene, images = clip_of("cut", 14, n_frames=N_FRAMES)
    clock, line_clock, inputs = scripted_waits([0.2] * N_FRAMES)
    stream, result = fed(
        scene, images, policy=OverloadPolicy(0.5, clock=clock), line_clock=line_clock, inputs=inputs(inputs_of(scene))
    )
    assert stream.dropped_frames == ()
    expected = pushed(scene, images)
    assert as_json(result.model_copy(update={"engines": expected.engines})) == as_json(expected)
    assert result.engines["overload"].startswith("drop a frame whose line waited more than 0.25 s")


def test_the_last_frame_dropped_gets_no_paint_score() -> None:
    scene, images = clip_of("plain", 15, n_frames=N_FRAMES - 1)
    last = N_FRAMES - 2
    assert last % 12 != 0
    runs = {}
    for name, last_wait in (("kept", 0.0), ("dropped", 5.0)):
        clock, line_clock, inputs = scripted_waits([0.0] * last + [last_wait])
        _, runs[name] = fed(
            scene, images, policy=OverloadPolicy(1.0, clock=clock), line_clock=line_clock, inputs=inputs(inputs_of(scene))
        )
    assert any(check.frame == last for check in runs["kept"].paint)
    assert all(check.frame != last for check in runs["dropped"].paint)
    assert runs["dropped"].frames[last].players == () and runs["kept"].frames[last].players != ()


# --- The live output of a stream with the policy ------------------------------------------------


def test_the_live_output_names_the_policy_counts_the_dropped_frames_and_writes_no_line_for_them(tmp_path: Path) -> None:
    scene, images = clip_of("plain", 16, n_frames=40)
    waits = [3.0 if frame in (10, 11) else 0.0 for frame in range(40)]
    clock, line_clock, inputs = scripted_waits(waits)
    path = tmp_path / "live.jsonl"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
    live = LiveOut(fd, scene.info, target=str(path), pipe=False, close_fd=True)
    live.overload(max_behind_s=1.0)
    live.start(delay_frames=SETTINGS.delay_frames, history_frames=SETTINGS.history_frames)
    fed(
        scene,
        images,
        policy=OverloadPolicy(1.0, clock=clock),
        line_clock=line_clock,
        inputs=inputs(inputs_of(scene)),
        on_dropped=live.skipped,
        on_given=live.given,
    )
    live.finish()
    live.close()
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    header, frames, end = lines[0], lines[1:-1], lines[-1]
    dropped = {5, 6, 10, 11}  # 10 and 11 came late, and the decisions of 5 and 6 were due with them
    assert header["format"] == "pitchvlm-live-1" and header["overload_max_behind_s"] == 1.0
    assert [line["frame"] for line in frames] == [frame for frame in range(40) if frame not in dropped]
    # The count on a line: the frames before it that the stream dropped.
    assert [line["overload_dropped"] for line in frames if line["frame"] in (4, 7, 12)] == [0, 2, 4]
    assert end == {"end": True, "frames": 36, "written": 36, "dropped": 0, "overload_dropped": 4}


def test_a_live_output_without_the_policy_keeps_its_lines_and_refuses_a_dropped_frame(tmp_path: Path) -> None:
    scene = SyntheticScene(n_frames=2, seed=17)
    path = tmp_path / "live.jsonl"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
    live = LiveOut(fd, scene.info, target=str(path), pipe=False, close_fd=True)
    live.start(delay_frames=0, history_frames=0)
    with pytest.raises(ValueError, match="not told of an overload policy"):
        live.skipped(1)
    with pytest.raises(ValueError, match="before the header"):
        live.overload(max_behind_s=1.0)
    live.finish()
    live.close()
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert "overload_max_behind_s" not in lines[0] and "overload_dropped" not in lines[-1]


# --- The command line ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--synthetic"], "stream --synthetic takes no --max-behind"),
        (["--clip", "{clip}", "--detections", "{file}", "--calibration", "{file}"], "applies only to a live stream"),
        (["--clip", "{clip}", "--lines", "{file}"], "applies only to a live stream"),
        (["--clip", "{clip}", "--lines", "-", "--delay", "clip"], "--max-behind needs a delay in frames"),
    ],
)
def test_stream_refuses_the_policy_without_a_live_clock(
    arguments: list[str], message: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clip = tmp_path / "clip"
    clip.mkdir()
    lines = tmp_path / "lines.jsonl"
    lines.write_text("", encoding="utf-8")
    given = [value.format(clip=clip, file=lines) for value in arguments]
    code = cli.main(["stream", *given, "--max-behind", "0.5", "--out", str(tmp_path / "out"), "--allow-dirty"])
    assert code == 2
    assert message in capsys.readouterr().err


def test_run_and_demo_take_no_overload_policy(tmp_path: Path) -> None:
    for command in (["demo"], ["run", "--clip", str(tmp_path)]):
        with pytest.raises(SystemExit) as stopped:
            cli.main([*command, "--out", str(tmp_path / "out"), "--max-behind", "0.5"])
        assert stopped.value.code == 2


def test_without_delay_a_late_frame_is_not_used_and_counted_once() -> None:
    scene, images = clip_of("plain", 19, n_frames=30)
    clock, line_clock, inputs = scripted_waits([2.0 if frame == 10 else 0.0 for frame in range(30)])
    settings = StreamSettings(delay_frames=0, history_frames=default_history_frames(CONFIG, 25.0))
    stream, result = fed(
        scene, images, settings, policy=OverloadPolicy(1.0, clock=clock), line_clock=line_clock,
        inputs=inputs(inputs_of(scene)),
    )
    assert stream.overload_report() == {"dropped": 1, "dropped_frames": [10], "not_used": [10], "decided_late": []}
    assert result.frames[10].players == () and result.frames[9].players != ()


def test_a_frame_kept_at_the_feeder_and_late_on_the_frame_path_is_read_and_measured_but_not_used() -> None:
    """The second look alone drops frames 12 and 13: the clock of the frame thread runs ahead of the feeder's."""
    scene, images = clip_of("plain", 20, n_frames=40)
    ahead = {"s": 0.0}

    def clock() -> float:
        return 1000.0 + (ahead["s"] if threading.current_thread() is threading.main_thread() else 0.0)

    taken = {"frame": -1}

    def inputs() -> Iterator[StreamInput]:
        for item in inputs_of(scene):
            taken["frame"] = item.frame
            yield item

    stream = new_stream(scene)
    feeder = FrameFeeder(
        stream, inputs(), lambda frame: images[frame], policy=OverloadPolicy(1.0, clock=clock),
        line_clock=lambda: (1000.0, 0),
    )
    seen: list[FedFrame] = []
    try:
        for frame in feeder:
            seen.append(frame)
            stream.push_fed(frame)
            ahead["s"] = 0.7 if frame.item.frame in (11, 12) else 0.0  # the wait that the next frame shows here
    finally:
        feeder.close()
    stream.finish()
    result = stream.result()
    assert stream.overload_report() == {"dropped": 2, "dropped_frames": [12, 13], "not_used": [12, 13], "decided_late": []}
    late = [frame for frame in seen if frame.item.frame in (12, 13)]
    assert all(frame.dropped and not frame.late and frame.image is not None for frame in late)
    assert all(frame.measure is not None and frame.measure.kind == "pair" for frame in late)
    assert result.frames[12].players == () and result.frames[14].players != ()


def test_stream_refuses_the_policy_on_a_file_given_as_the_standard_input_and_with_a_queue(tmp_path: Path) -> None:
    clip = tmp_path / "clip"
    clip.mkdir()
    lines = tmp_path / "lines.jsonl"
    lines.write_text("", encoding="utf-8")
    base = [sys.executable, "-m", "pitchvlm.cli", "stream", "--clip", str(clip), "--lines", "-", "--max-behind", "0.5",
            "--out", str(tmp_path / "out"), "--allow-dirty"]
    environment = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    with lines.open("rb") as stdin:
        done = subprocess.run(base, stdin=stdin, capture_output=True, text=True, env=environment, timeout=120, check=False)
    assert done.returncode == 2 and "applies only to a live stream" in done.stderr
    done = subprocess.run([*base, "--queue", "5"], input="", capture_output=True, text=True, env=environment, timeout=120,
                          check=False)
    assert done.returncode == 2 and "--max-behind takes no --queue" in done.stderr
