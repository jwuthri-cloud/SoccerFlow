"""Tests of the live feed: the live mode of the tool script, the queue of the stream, the paint thread, the command
line, and the whole chain as real processes with real pipes.

The tools are stand-ins (``tools/stand_in_tool.py`` for the processes, a module of the tests in
the process): the script, its per-frame loop and its feed are those of the run machine. Nothing
here says how a real tool, real footage or the run machine behave.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
import pytest

from pitchvlm import cli, feed, stream
from pitchvlm.evalx.livebench import CLIP_ID, LiveClip, prepare_clip, run_live
from pitchvlm.fallbacks import runs
from pitchvlm.stream import (
    PAINT_MAX_PENDING,
    DetectorStream,
    QueuedLines,
    StreamSettings,
    clip_run_lines,
    stand_in_inputs,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools" / "run_tool.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_tool_for_live", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_tool_for_live"] = module
    spec.loader.exec_module(module)
    return module


run_tool = load_script()


@pytest.fixture(scope="module")
def live_clip(tmp_path_factory: pytest.TempPathFactory) -> LiveClip:
    return prepare_clip(tmp_path_factory.mktemp("live"), 40, 1)


# --- The live mode of the tool script ---------------------------------------------------------------


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert image is not None
    return np.ascontiguousarray(image[:, :, ::-1])


@dataclasses.dataclass
class Calls:
    """The feed file that the stand-in detector looks at, and the feed lines written before each of its calls."""

    feed: Path | None = None
    lines_before: list[int] = dataclasses.field(default_factory=list)


@pytest.fixture
def stand_in_detector(monkeypatch: pytest.MonkeyPatch, live_clip: LiveClip) -> Calls:
    """Replace the detector of the script by the answers of the clip, and note what the feed holds at each call."""
    by_image = json.loads(live_clip.answers["rfdetr"].read_text(encoding="utf-8"))["by_image"]
    calls = Calls()

    def answer(image: np.ndarray) -> dict[str, Any]:
        written = calls.feed is not None and calls.feed.exists()
        calls.lines_before.append(len(calls.feed.read_text(encoding="utf-8").splitlines()) if written else 0)
        return by_image[hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()]

    monkeypatch.setattr(run_tool, "read_rgb", read_rgb)
    monkeypatch.setattr(run_tool, "installed_version", lambda package: "stand-in")
    monkeypatch.setattr(run_tool, "rfdetr_engine", lambda settings, device: ({"tool_version": "stand-in"}, answer))
    return calls


def test_a_live_run_writes_each_line_before_it_answers_the_next_frame(
    tmp_path: Path, live_clip: LiveClip, stand_in_detector: Calls
) -> None:
    feed_path, copy = tmp_path / "feed.jsonl", tmp_path / "copy.jsonl"
    stand_in_detector.feed = feed_path
    arguments = ["rfdetr", "--job", str(live_clip.jobs["rfdetr"]), "--root", str(live_clip.root)]
    assert run_tool.main([*arguments, "--out", str(copy), "--device", "cpu", "--feed", str(feed_path)]) == 0
    lines = feed_path.read_text(encoding="utf-8").splitlines()
    assert copy.read_text(encoding="utf-8").splitlines() == lines
    head = json.loads(lines[0])
    assert head["format"] == feed.TOOL_FEED_FORMAT == run_tool.TOOL_FEED_FORMAT
    assert (head["clip"], head["n_frames"], head["video_sha256"]) == (CLIP_ID, 40, None)
    assert head["job_sha256"] == hashlib.sha256(live_clip.jobs["rfdetr"].read_bytes()).hexdigest()
    frames = [json.loads(text) for text in lines[1:]]
    assert [frame["frame"] for frame in frames] == list(range(40))
    assert set(frames[0]) == {"frame", "seconds", "arrived_at", "payload"} and "rows" in frames[0]["payload"]
    # The warm-up call comes before the header; then the call of frame i sees the header and the i lines before it.
    assert stand_in_detector.lines_before == [0, *(1 + frame for frame in range(40))]
    # A live run writes no clip run.
    assert sorted(path.name for path in tmp_path.iterdir()) == ["copy.jsonl", "feed.jsonl"]


class Clock:
    """A clock that moves only when the code sleeps or a test moves it."""

    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class Sink:
    def __init__(self) -> None:
        self.lines: list[dict[str, Any]] = []
        self.flushed = 0

    def write(self, text: str) -> None:
        self.lines.append(json.loads(text))

    def flush(self) -> None:
        self.flushed += 1


HEAD = {"clip": "C", "n_frames": 6, "video_sha256": None, "job_sha256": "0" * 64, "tool": {"name": "pnlcalib"}}
FRAMES = [{"clip": "C", "frame": index, "image": f"{index}.png"} for index in range(6)]


def test_a_pace_holds_each_frame_until_it_comes() -> None:
    clock, sink = Clock(), Sink()
    live = run_tool.LiveFeed([sink], HEAD, pace=10.0, start_at=105.0, clock=clock, sleep=clock.sleep)
    live.start()
    seen = []
    for group in live.groups(FRAMES, 2):
        seen.append(([entry["frame"] for entry in group], round(clock.now, 6)))
    # A batch of two frames waits for its second frame: frame 1 comes at 105.1, frame 3 at 105.3.
    assert seen == [([0, 1], 105.1), ([2, 3], 105.3), ([4, 5], 105.5)]
    assert live.arrived_at(4) == pytest.approx(105.4) and sink.flushed == 1


def test_skip_behind_takes_the_newest_frame_and_writes_a_line_for_each_skipped_one() -> None:
    clock, sink = Clock(), Sink()
    live = run_tool.LiveFeed([sink], HEAD, pace=10.0, start_at=100.0, skip_behind=True, clock=clock, sleep=clock.sleep)
    live.start()
    answered = []
    for group in live.groups(FRAMES, 1):
        answered.append(group[0]["frame"])
        clock.now += 0.25  # the tool takes 250 ms per frame, the frames come every 100 ms
    assert answered == [0, 2, 5]
    skipped = [line["frame"] for line in sink.lines if line.get("skipped")]
    assert skipped == [1, 3, 4] and live.skipped == 3
    assert all(set(line) == {"frame", "arrived_at", "skipped"} for line in sink.lines if line.get("skipped"))


def test_live_options_are_refused_where_they_do_not_apply(tmp_path: Path, live_clip: LiveClip) -> None:
    with pytest.raises(run_tool.RunnerError, match="need --pace"):
        run_tool.LiveFeed([Sink()], HEAD, skip_behind=True)
    with pytest.raises(run_tool.RunnerError, match="batches of one frame"):
        live = run_tool.LiveFeed([Sink()], HEAD, pace=10.0, skip_behind=True)
        live.start()
        next(live.groups(FRAMES, 2))
    job = tmp_path / "job.json"
    job.write_text(
        json.dumps({"format": runs.JOB_FORMAT, "name": "a", "tool": {"name": "rfdetr", "model": "RFDETRSmall",
                    "min_score": 0.5, "antialias": False}, "frames": [{"clip": "C", "frame": 0, "image": "a.png"}]}),
        encoding="utf-8",
    )
    common = ["--root", str(tmp_path), "--out", str(tmp_path / "out.json"), "--device", "cpu"]
    assert run_tool.main(["rfdetr", "--job", str(job), *common, "--feed", "-"]) == 2
    assert run_tool.main(["rfdetr", "--job", str(live_clip.jobs["rfdetr"]), *common, "--pace", "25"]) == 2


def test_the_copy_gets_each_line_before_the_feed() -> None:
    """Second read of this work: a feed that breaks leaves the copy with every line that the feed got."""

    class Broken(Sink):
        def write(self, text: str) -> None:
            if len(self.lines) == 1:
                raise BrokenPipeError(32, "Broken pipe")
            super().write(text)

    copy, broken = Sink(), Broken()
    live = run_tool.LiveFeed([copy, broken], HEAD)
    live.start()
    with pytest.raises(BrokenPipeError):
        live.write({"frame": 0, "seconds": 0.01, "arrived_at": 1.0, "payload": {}})
    assert [line.get("frame") for line in copy.lines] == [None, 0] and len(broken.lines) == 1


def test_skip_behind_with_batches_is_refused_before_the_tool_is_built_and_the_feed_opened(tmp_path: Path) -> None:
    """Second read of this work: before, the refusal came after the header had gone into the feed."""
    job = tmp_path / "job.json"
    settings = {"name": "dfine", "model": "small", "min_score": 0.5, "input_size": 640, "batch": 2, "half": False}
    job.write_text(json.dumps({"format": runs.CLIP_JOB_FORMAT, "clip": "C", "tool": settings, "images": ["a.png"]}),
                   encoding="utf-8")
    feed_path = tmp_path / "feed.jsonl"
    arguments = ["dfine", "--job", str(job), "--root", str(tmp_path), "--out", str(tmp_path / "out.jsonl")]
    code = run_tool.main([*arguments, "--device", "cpu", "--feed", str(feed_path), "--pace", "25", "--skip-behind"])
    assert code == 2 and not feed_path.exists()


# --- The queue of the stream: back-pressure ---------------------------------------------------------


def test_the_queue_of_the_stream_holds_back_the_reading_and_drops_nothing() -> None:
    pulled: list[int] = []

    def lines() -> Any:
        for index in range(20):
            pulled.append(index)
            yield f"{index}\n"

    queued = QueuedLines(lines(), max_lines=3)
    time.sleep(0.2)
    # 3 lines wait in the queue and the thread holds the 4th: the reading stops there.
    assert len(pulled) == 4
    taken = []
    for text in queued:
        taken.append(int(text))
        time.sleep(0.005)
    assert taken == list(range(20)) and 2 <= queued.largest_backlog <= 3


def test_an_error_of_the_reading_is_raised_in_the_reader() -> None:
    def lines() -> Any:
        yield "0\n"
        raise OSError("the pipe broke")

    queued = QueuedLines(lines(), max_lines=2)
    with pytest.raises(OSError, match="the pipe broke"):
        list(queued)


# --- The paint scores beside the frame path ------------------------------------------------------------


def test_the_paint_scores_run_on_their_own_thread_and_a_frame_waits_only_when_too_many_are_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pitchvlm.config import EngineSettings, PipelineConfig
    from pitchvlm.fallbacks.fake import fake_calibration_answers, fake_detector_answers
    from pitchvlm.synth import SyntheticScene

    release = threading.Event()
    threads: set[str] = set()
    real = stream.paint_alignment

    def held_paint(image: np.ndarray, matrix: np.ndarray) -> Any:
        threads.add(threading.current_thread().name)
        release.wait(10.0)
        return real(image, matrix)

    monkeypatch.setattr(stream, "paint_alignment", held_paint)
    scene = SyntheticScene(n_frames=12 * (PAINT_MAX_PENDING + 2), seed=4)
    config = PipelineConfig(engines=EngineSettings(people="detector", pitch_mapping="calibration", number="none"))
    info = scene.info
    detector = DetectorStream(config, StreamSettings(delay_frames=5, history_frames=20), clip_id=info.clip_id,
                              fps=info.fps, size=(info.width, info.height))
    people, balls = fake_detector_answers(scene)
    items = list(stand_in_inputs(people, balls, fake_calibration_answers(scene)))
    pushed = threading.Thread(target=lambda: [detector.push(item, scene.image(item.frame)) for item in items])
    pushed.start()
    deadline = time.monotonic() + 60.0
    while detector.n_frames < 12 * PAINT_MAX_PENDING and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.5)
    # Keyframes 0, 12, 24, 36 are open (one of them on the thread); the push of keyframe 48 waits.
    assert detector.n_frames == 12 * PAINT_MAX_PENDING
    release.set()
    pushed.join(30.0)
    detector.finish()
    assert threads == {"pitchvlm-paint_0"}
    expected = [real(scene.image(f), np.array(c.matrix)) for f, c in enumerate(fake_calibration_answers(scene)) if f % 12 == 0]
    result = detector.result()
    assert [check.score for check in result.paint if check.frame % 12 == 0] == [found.score for found in expected]
    assert len(detector.paint_seconds) == len(result.paint)


# --- The command line ----------------------------------------------------------------------------------


@pytest.fixture
def no_video(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "render_clip", lambda source, result, out_path: out_path)


def clip_runs_of(live_clip: LiveClip, work: Path) -> tuple[Path, Path]:
    """Write the clip runs of the two stand-ins from the payloads of the clip, as the tool script writes them."""
    paths = []
    for tool in ("rfdetr", "pnlcalib"):
        job = run_tool.read_job(live_clip.jobs[tool], tool)
        frames = [
            {"clip": CLIP_ID, "frame": frame, "seconds": 0.0, "payload": payload}
            for frame, payload in enumerate(live_clip.payloads[tool])
        ]
        out = work / f"{tool}-run.json"
        run_tool.write_clip_run(out, job, run_tool.file_sha256(live_clip.jobs[tool]), {"device": "stand-in"}, frames)
        paths.append(out)
    return paths[0], paths[1]


def lines_with(live_clip: LiveClip, work: Path, **header: Any) -> Path:
    detections, calibration = (runs.load_clip_run(path) for path in clip_runs_of(live_clip, work))
    lines = list(clip_run_lines(detections, calibration))
    lines[0] = json.dumps({**json.loads(lines[0]), **header})
    path = work / "lines.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("header", "words"),
    [
        ({"video_sha256": "a" * 64}, "made from other pixels than --clip"),
        ({"n_frames": 41}, "the stream names 41 frames, the clip has 40 frames"),
    ],
)
def test_the_lines_must_name_the_video_and_the_frames_of_the_clip(
    tmp_path: Path, live_clip: LiveClip, no_video: None, capsys: pytest.CaptureFixture[str], header: Any, words: str
) -> None:
    path = lines_with(live_clip, tmp_path, **header)
    base = ["stream", "--clip", str(live_clip.clip), "--fps", "25", "--allow-dirty", "--out", str(tmp_path / "o")]
    assert cli.main([*base, "--lines", str(path)]) == 2
    assert words in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra", "words"),
    [
        (["--synthetic", "--fps", "25"], "stream --synthetic takes no --fps"),
        (["--synthetic", "--queue", "5"], "stream --synthetic takes no --queue"),
        (["--synthetic", "--split", "valid"], "stream --synthetic takes no --split"),
        (["--clip", "C", "--frames", "30"], "--frames apply only to stream --synthetic"),
        (["--clip", "C", "--seed", "3"], "--seed apply only to stream --synthetic"),
        (["--clip", "C", "--lines", "x", "--detections", "d"], "stream --lines takes no --detections"),
        (["--clip", "C", "--detections", "d", "--calibration", "c", "--queue", "5"], "--queue applies only to"),
        (["--clip", "C", "--lines", "x", "--split", "valid"], "--split and --tracker name the prediction file"),
    ],
)
def test_options_that_do_not_apply_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str], words: str
) -> None:
    assert cli.main(["stream", "--out", str(tmp_path), "--allow-dirty", *extra]) == 2
    assert words in capsys.readouterr().err


def test_merge_feeds_on_two_files_gives_the_lines_of_the_clip_runs(
    tmp_path: Path, live_clip: LiveClip, capsys: pytest.CaptureFixture[str]
) -> None:
    feeds = {}
    for tool, job in live_clip.jobs.items():
        detector = tool == "rfdetr"
        head = {"format": feed.TOOL_FEED_FORMAT, "clip": CLIP_ID, "n_frames": live_clip.n_frames, "video_sha256": None,
                "job_sha256": run_tool.file_sha256(job), "tool": run_tool.read_job(job, tool)["tool"]}
        rows = [json.dumps(head)]
        for frame, payload in enumerate(live_clip.payloads[tool]):
            body = run_tool.compact_detections(payload) if detector else payload
            rows.append(json.dumps({"frame": frame, "seconds": 0.0, "arrived_at": 0.0, "payload": body}))
        feeds[tool] = tmp_path / f"{tool}.jsonl"
        feeds[tool].write_text("\n".join(rows[:-1] if not detector else rows) + "\n", encoding="utf-8")
    code = cli.main(["merge-feeds", "--detector", str(feeds["rfdetr"]), "--calibration", str(feeds["pnlcalib"])])
    captured = capsys.readouterr()
    assert code == 1  # the last line of the calibration is missing
    assert "40 of 40 frames went out" in captured.err and "ended: detector 0, calibration 1" in captured.err
    merged = [json.loads(text) for text in captured.out.splitlines()]
    expected = [json.loads(text) for text in clip_run_lines(*(runs.load_clip_run(p) for p in clip_runs_of(live_clip, tmp_path)))]
    assert merged[0] == expected[0]
    for got, want in zip(merged[1:-1], expected[1:-1], strict=True):
        assert got["detector"]["payload"] == want["detector"]["payload"]
        assert got["calibration"]["payload"] == want["calibration"]["payload"]
    assert merged[-1]["calibration"] == {"missing": "ended"}


# --- The whole chain: four processes, two named pipes and a pipe ----------------------------------------


def test_the_live_chain_gives_the_result_of_the_clip_runs(tmp_path: Path, live_clip: LiveClip, no_video: None) -> None:
    outcome = run_live(live_clip, tmp_path / "live", delay=5, pace=None, simulated_ms={"detector": 0, "calibration": 0},
                       timeout_s=120)
    assert outcome.codes == {"stream": 0, "merge-feeds": 0, "rfdetr": 0, "pnlcalib": 0}, outcome.merge_report
    detections, calibration = clip_runs_of(live_clip, tmp_path)
    base = ["stream", "--clip", str(live_clip.clip), "--fps", "25", "--delay", "5", "--allow-dirty"]
    assert cli.main([*base, "--detections", str(detections), "--calibration", str(calibration), "--out", str(tmp_path / "f")]) == 0
    offline = json.loads((tmp_path / "f" / CLIP_ID / "result.json").read_text(encoding="utf-8"))
    assert outcome.document is not None
    assert outcome.document["result"] == offline["result"]
    latency = outcome.document["latency"]
    assert latency["decided_live"] == 35 and latency["source_to_decision"]["n"] == 35
    assert outcome.document["missing_parts"] == {}


def test_the_chain_ends_when_the_stream_stops_before_it_reads(tmp_path: Path, live_clip: LiveClip) -> None:
    start = time.monotonic()
    outcome = run_live(live_clip, tmp_path / "live", delay=5, pace=None, simulated_ms={"detector": 0, "calibration": 0},
                       timeout_s=60, stream_arguments=["--history", "abc"])
    assert time.monotonic() - start < 60
    assert outcome.codes == {"stream": 2, "merge-feeds": 2, "rfdetr": 2, "pnlcalib": 2}


def test_a_tool_that_dies_leaves_its_part_missing_and_the_stream_goes_on(
    tmp_path: Path, live_clip: LiveClip
) -> None:
    by_image = json.loads(live_clip.answers["pnlcalib"].read_text(encoding="utf-8"))["by_image"]
    short = tmp_path / "pnlcalib-answers.json"
    short.write_text(json.dumps({"by_image": dict(list(by_image.items())[:25])}), encoding="utf-8")
    dying = dataclasses.replace(live_clip, answers={**live_clip.answers, "pnlcalib": short})
    outcome = run_live(dying, tmp_path / "live", delay=5, pace=None, simulated_ms={"detector": 0, "calibration": 0},
                       timeout_s=120)
    assert outcome.codes == {"stream": 0, "merge-feeds": 1, "rfdetr": 0, "pnlcalib": 2}
    assert outcome.document is not None and outcome.document["missing_parts"] == {"calibration": {"ended": 15}}
    assert "frames with a missing part of a tool: calibration: 15 ended" in outcome.stream_output
    # The cameras of frames 25 to 39 come from frame 24, moved with the camera motion.
    sources = [record["source"] for record in outcome.document["result"]["homographies"]]
    assert sources[:25] == ["landmarks"] * 25 and set(sources[25:]) == {"propagated"}


def test_the_synthetic_clip_still_takes_a_tracker_name(tmp_path: Path, no_video: None) -> None:
    """Second read of this work: --tracker names the prediction path of the synthetic clip too."""
    out = tmp_path / "out"
    assert cli.main(["stream", "--synthetic", "--frames", "30", "--delay", "5", "--tracker", "mine", "--out", str(out),
                     "--allow-dirty"]) == 0
    assert (out / "predictions" / "SoccerNetGS-demo" / "mine" / "SNGS-999.json").exists()
