"""Tests of the streaming path of the detector engine (``stream``) and of the parts it shares with the clip path.

The tools are the stand-ins of ``fallbacks.fake`` on the synthetic scene; the clip runs of the
command-line tests are written with the part writer of ``tools/run_tool.py``. Nothing here says
how a real detector, a real calibration or a live pipe behave.
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
import pytest

from pitchvlm import cli
from pitchvlm.ball import BallCandidate, track_ball
from pitchvlm.config import EngineSettings, PipelineConfig
from pitchvlm.evalx.metrics import identity_scores
from pitchvlm.evalx.streambench import ArraySource, clip_of, run_case
from pitchvlm.fallbacks.fake import fake_calibration_answers, fake_detector_answers
from pitchvlm.fallbacks.runs import load_clip_run
from pitchvlm.fill import dense_frame_states, players_in_frame
from pitchvlm.motion import DISSOLVE_WINDOW_FRAMES, MotionField, ShotMeter, measure_shots
from pitchvlm.pipeline import ToolAnswers, _people_on_pitch, _pitch_map, clip_kit_clusters, run_clip
from pitchvlm.schemas import Box, FramePeople, HomographyRecord, PersonDet, PlayerState, Track, TrackObs
from pitchvlm.stream import (
    EARLY_DISSOLVE_WINDOWS,
    STREAM_FORMAT,
    DetectorStream,
    StreamInput,
    StreamSettings,
    _StreamIds,
    check_stream_engines,
    clip_run_lines,
    default_history_frames,
    read_stream_lines,
    run_stream,
    stand_in_inputs,
)
from pitchvlm.synth import RetimedScene, SyntheticScene, SyntheticSource
from pitchvlm.timing import FRAME_TOTAL, StageClock
from pitchvlm.tracking import Associator, associate

REPO = Path(__file__).resolve().parents[1]
DETECTOR_ONLY = EngineSettings(people="detector", pitch_mapping="calibration", number="none")
CONFIG = PipelineConfig(engines=DETECTOR_ONLY)
CLIP_ID = "SNGS-999"
LABEL_NAME = "Labels-GameState.json"
N_FRAMES = 60


def load_script() -> ModuleType:
    """Load ``tools/run_tool.py`` as a module (it is not part of a package)."""
    spec = importlib.util.spec_from_file_location("run_tool_for_stream", REPO / "tools" / "run_tool.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_tool_for_stream"] = module
    spec.loader.exec_module(module)
    return module


run_tool = load_script()


def stream_of(
    scene: SyntheticScene,
    settings: StreamSettings,
    *,
    images: Sequence[np.ndarray] | None = None,
    people: Sequence[FramePeople] | None = None,
    kits: Any = None,
) -> tuple[DetectorStream, list[list[Any]]]:
    """Push every frame of a scene into a stream; return the stream (not finished) and what each push gave."""
    found, balls = fake_detector_answers(scene)
    people = found if people is None else people
    cameras = fake_calibration_answers(scene)
    info = scene.info
    stream = DetectorStream(
        CONFIG, settings, clip_id=info.clip_id, fps=info.fps, size=(info.width, info.height), kit_clusters=kits
    )
    pictures = images if images is not None else [scene.image(frame) for frame in range(info.n_frames)]
    given = [stream.push(item, pictures[item.frame]) for item in stand_in_inputs(people, balls, cameras)]
    return stream, given


def offline_and_kits(scene: SyntheticScene, source: Any) -> tuple[Any, Any]:
    """Run the clip path and return its result and the kit clusters it fitted (``streambench._clip_kits``)."""
    people, balls = fake_detector_answers(scene)
    cameras = fake_calibration_answers(scene)
    offline = run_clip(source, None, CONFIG, tools=ToolAnswers(people=people, balls=balls, cameras=cameras))
    to_pitch = _pitch_map(offline.homographies)
    on_pitch = [_people_on_pitch(item, to_pitch, CONFIG.pitch_margin_m) for item in people]
    clusters, _ = clip_kit_clusters(source, on_pitch)
    return offline, clusters


# --- The whole clip: the control equals the clip path -----------------------------------------


@pytest.mark.parametrize("kind", ["plain", "cut"])
def test_a_delay_and_a_window_of_the_whole_clip_give_the_result_of_the_clip_path(kind: str) -> None:
    """With the kit clusters of the clip path, the stream decides every frame as ``run_clip`` does."""
    scene, images = clip_of(kind, 3, n_frames=N_FRAMES)
    source = ArraySource(scene.info, images)
    offline, kits = offline_and_kits(scene, source)
    assert kits is not None
    people, balls = fake_detector_answers(scene)
    result = run_stream(
        stand_in_inputs(people, balls, fake_calibration_answers(scene)),
        source.read,
        CONFIG,
        StreamSettings(delay_frames=None, history_frames=None),
        clip_id=scene.info.clip_id,
        fps=scene.info.fps,
        size=(scene.info.width, scene.info.height),
        kit_clusters=kits,
    )
    assert result.frames == offline.frames
    assert result.homographies == offline.homographies
    assert result.cuts == offline.cuts
    assert result.paint == offline.paint
    assert (kind == "cut") == bool(result.cuts)
    assert result.engines["delay"] == result.engines["history"] == "whole clip"
    assert {key: value for key, value in result.engines.items() if key not in ("delay", "history")} == {
        **offline.engines,
        "tracks": "code:stream",
        "fill": "code:stream",
    }


def test_the_bench_finds_the_control_equal_to_the_clip_path_on_a_dissolve_too() -> None:
    """On a dissolve the control still equals the clip path here (with exact boxes); the bench reports it per case."""
    results = {result.arm: result for result in run_case("dissolve", "exact", 3, n_frames=200, arms=("offline", "clip-kits"))}
    assert results["offline"].equal_to_offline
    assert results["clip-kits"].switches == results["offline"].switches
    assert results["clip-kits"].coverage == pytest.approx(results["offline"].coverage, abs=0.02)


# --- Causality ----------------------------------------------------------------------------------


@pytest.mark.parametrize("delay", [0, 5])
def test_a_given_state_depends_on_no_frame_after_its_delay(delay: int) -> None:
    """Two streams that agree up to frame t give the same states up to frame t - delay, however they go on."""
    scene = SyntheticScene(n_frames=N_FRAMES, seed=3)
    images = [scene.image(frame) for frame in range(N_FRAMES)]
    people, _ = fake_detector_answers(scene)
    split = 30
    # After frame 30 the second stream sees nobody and a black picture.
    changed_people = [item if item.frame <= split else FramePeople(frame=item.frame, people=()) for item in people]
    changed_images = [image if frame <= split else np.zeros_like(image) for frame, image in enumerate(images)]
    settings = StreamSettings(delay_frames=delay, history_frames=40)
    _, first = stream_of(scene, settings, images=images)
    _, second = stream_of(scene, settings, images=changed_images, people=changed_people)
    given_first = [state for states in first[: split + 1] for state in states]
    given_second = [state for states in second[: split + 1] for state in states]
    assert [state.frame for state in given_first] == list(range(split + 1 - delay))
    assert given_first == given_second
    assert any(state.players for state in given_first)
    # The control: after frame 30 the two streams do differ.
    assert [state for states in first[split + 1 :] for state in states] != [
        state for states in second[split + 1 :] for state in states
    ]


def test_a_delay_of_zero_gives_each_frame_when_it_comes_in_and_finish_gives_the_rest() -> None:
    scene = SyntheticScene(n_frames=20, seed=1)
    stream, given = stream_of(scene, StreamSettings(delay_frames=0, history_frames=10))
    assert [[state.frame for state in states] for states in given] == [[frame] for frame in range(20)]
    assert stream.finish() == []
    late, given = stream_of(scene, StreamSettings(delay_frames=7, history_frames=10))
    assert [state.frame for states in given for state in states] == list(range(13))
    assert [state.frame for state in late.finish()] == list(range(13, 20))
    assert [state.frame for state in late.result().frames] == list(range(20))
    with pytest.raises(ValueError, match="finished"):
        late.finish()


def test_the_stream_refuses_frames_out_of_order_and_engines_that_do_not_stream() -> None:
    scene = SyntheticScene(n_frames=3, seed=1)
    stream = DetectorStream(CONFIG, StreamSettings(), clip_id="c", fps=25.0, size=(1280, 720))
    item = StreamInput(frame=1, people=(), balls=(), camera=HomographyRecord(frame=1))
    with pytest.raises(ValueError, match="frame 0 was expected"):
        stream.push(item, scene.image(0))
    wrong_camera = StreamInput(frame=0, people=(), balls=(), camera=HomographyRecord(frame=2))
    with pytest.raises(ValueError, match="names frame 2"):
        stream.push(wrong_camera, scene.image(0))
    with pytest.raises(ValueError, match="number engine none"):
        check_stream_engines(PipelineConfig(engines=EngineSettings(people="detector", pitch_mapping="calibration")))
    with pytest.raises(ValueError, match="whole clip"):
        StreamSettings(delay_frames=None, history_frames=10)
    with pytest.raises(ValueError, match="0 frames or more"):
        StreamSettings(delay_frames=-1)


def test_the_stream_follows_every_person_of_the_synthetic_clip_with_a_short_delay() -> None:
    scene = SyntheticScene(n_frames=N_FRAMES, seed=3)
    stream, _ = stream_of(scene, StreamSettings(delay_frames=12, history_frames=default_history_frames(CONFIG, 25.0)))
    stream.finish()
    result = stream.result()
    truth = [scene.frame_state(frame) for frame in range(N_FRAMES)]
    scores = identity_scores(result.frames, truth)
    assert scores.switches == 0
    assert scores.coverage > 0.95
    assert scores.track_ratio <= 1.05
    assert result.engines["delay"] == "12 frames (0.48 s)"
    assert {track.track_id for track in result.tracks} == {
        player.track_id for state in result.frames for player in state.players
    }
    assert sum(state.ball is not None for state in result.frames) > N_FRAMES // 2


def test_the_kit_clusters_are_fixed_after_the_warm_up() -> None:
    scene = SyntheticScene(n_frames=12, seed=3)
    stream = DetectorStream(
        CONFIG, StreamSettings(delay_frames=0, history_frames=5, kit_warmup_frames=4), clip_id="c", fps=25.0,
        size=(1280, 720),
    )
    people, balls = fake_detector_answers(scene)
    seen = []
    for item in stand_in_inputs(people, balls, fake_calibration_answers(scene)):
        stream.push(item, scene.image(item.frame))
        seen.append(stream._clusters)
    # Seed 3: the 13 torso features of one frame do not pass the two-kit check, so the fit is tried on every
    # frame (on the features of the last 4 frames after the warm-up) until one passes; then it stays.
    fixed = next(index for index, clusters in enumerate(seen) if clusters is not None and index >= 3)
    assert seen[0] is None
    assert all(clusters is seen[fixed] for clusters in seen[fixed:])
    assert stream._features == []
    assert len(people[0].people) == 13


# --- Identities ---------------------------------------------------------------------------------


def obs(frame: int, x: float) -> TrackObs:
    return TrackObs(frame=frame, box=Box(x1=x, y1=100.0, x2=x + 20.0, y2=160.0), role="player", team_key="A")


def state(track_id: int, observation: TrackObs) -> PlayerState:
    return PlayerState(track_id=track_id, box=observation.box, pitch_x=0.0, pitch_y=0.0, role="player", observed=True)


def test_a_track_keeps_the_id_of_its_newest_given_observation_and_the_newest_claim_wins() -> None:
    ids = _StreamIds()
    a0, a1, b0, b1 = obs(0, 10.0), obs(1, 10.0), obs(0, 50.0), obs(1, 50.0)
    for given_id, observation in ((7, a0), (7, a1), (9, b0), (9, b1)):
        ids.note(observation.frame, state(given_id, observation))
    # A window that joins the two pieces wrongly: track 0 holds a0 (id 7) then b1 (id 9); track 1 holds b0 (id 9).
    tracks = [
        Track(track_id=0, observations=(a0, b1, obs(2, 50.0))),
        Track(track_id=1, observations=(b0, obs(2, 90.0))),
    ]
    # Track 0 had id 9 at frame 1, track 1 had id 9 at frame 0: track 0 wins id 9. Track 1 has no other given
    # id, so it takes a new one: its window id 1, which no state carried yet.
    assert ids.ids(tracks, 2) == {0: 9, 1: 1}
    fresh = ids.ids([Track(track_id=5, observations=(obs(2, 200.0),))], 2)
    assert fresh == {5: 5}
    assert ids.ids([Track(track_id=5, observations=(obs(3, 210.0),))], 3) != {5: 5}


# --- Shared code: the incremental parts give the clip results -----------------------------------


def test_the_associator_gives_the_tracks_of_associate_also_when_it_forgets_old_observations() -> None:
    scene = SyntheticScene(n_frames=80, seed=2)
    people, _ = fake_detector_answers(scene, miss_rate=0.2, box_noise_px=2.0, seed=2)
    matrices = [scene.homography(frame) for frame in range(80)]
    steps = [np.linalg.inv(matrices[frame + 1]) @ matrices[frame] for frame in range(79)]
    field = MotionField(steps)

    def to_pitch(frame: int, xy: np.ndarray) -> np.ndarray:
        from pitchvlm.geometry import apply_homography

        return apply_homography(matrices[frame], xy)

    whole = associate(people, fps=25.0, warp=field.warp, to_pitch=to_pitch, keyframe_frames=12)
    forgetting = Associator(fps=25.0, warp=field.warp, to_pitch=to_pitch, keyframe_frames=12)
    for item in people:
        forgetting.add(item)
        forgetting.forget_before(item.frame - 20)
    window = forgetting.tracks(since=60)
    expected = [
        Track(track_id=track.track_id, observations=tuple(o for o in track.observations if o.frame >= 60))
        for track in whole
        if track.last_frame >= 60
    ]
    assert window == expected
    with pytest.raises(ValueError, match="does not come after"):
        forgetting.add(people[3])


def test_the_shot_meter_gives_the_tests_of_measure_shots_and_knows_a_dissolve_a_window_after_it() -> None:
    scene, images = clip_of("dissolve", 0, n_frames=200)
    source = ArraySource(scene.info, images)
    whole = measure_shots(source)
    meter = ShotMeter((scene.info.width, scene.info.height))
    known_at = None
    for frame, image in enumerate(images):
        meter.add(image)
        if meter.update_transitions() and known_at is None:
            known_at = frame
        meter.forget_images_before(frame - 1)
    tests = meter.finish()
    assert whole.transitions and tests.transitions == whole.transitions
    assert tests.hard_cuts == whole.hard_cuts and tests.cuts() == whole.cuts()
    assert tests.window_correlations == whole.window_correlations
    assert all(np.array_equal(a, b) for a, b in zip(tests.steps, whole.steps, strict=True) if a is not None)
    start = whole.transitions[0][0]
    assert known_at is not None and start < known_at <= start + DISSOLVE_WINDOW_FRAMES + 12
    assert sum(image is not None for image in tests.test_images) <= 2 * DISSOLVE_WINDOW_FRAMES + 3


def test_players_in_frame_gives_the_players_of_dense_frame_states() -> None:
    scene = SyntheticScene(n_frames=40, seed=5)
    rng = np.random.default_rng(0)
    records = fake_calibration_answers(scene)
    tracks = []
    for track_id, person in enumerate(scene.people(0).people[:8]):
        frames = sorted(rng.choice(40, size=12, replace=False).tolist())
        observations = tuple(
            TrackObs(frame=frame, box=person.box, role="player", team_key="A") for frame in frames
        )
        tracks.append(Track(track_id=track_id, observations=observations, role="player", team_key="A"))
    records = [record if frame % 7 else HomographyRecord(frame=frame) for frame, record in enumerate(records)]
    options = dict(warp=lambda src, dst, xy: xy, image_size=(1280, 720), max_fill_gap_frames=6, cuts=(25,))
    dense = dense_frame_states(tracks, records, 40, **options)
    for frame in range(40):
        assert players_in_frame(tracks, records, frame, n_frames=40, **options) == dense[frame].players


def test_the_ball_tracker_from_a_first_frame_ignores_nothing_after_it() -> None:
    box = Box(x1=100.0, y1=100.0, x2=104.0, y2=104.0)
    candidates = {
        frame: (BallCandidate(frame=frame, box=box.model_copy(update={"x1": 100.0 + frame, "x2": 104.0 + frame}), score=0.8),)
        for frame in range(30)
        if frame % 5
    }

    def warp(src: int, dst: int, xy: np.ndarray) -> np.ndarray:
        return xy

    whole = track_ball(candidates, 30, fps=25.0, warp=warp, max_gap_frames=3)
    later = {frame: found for frame, found in candidates.items() if frame >= 11}
    part = track_ball(later, 30, fps=25.0, warp=warp, max_gap_frames=3, first_frame=11)
    assert {frame: b for frame, b in whole.boxes.items() if frame >= 11} == part.boxes
    with pytest.raises(ValueError, match="do not belong"):
        track_ball(candidates, 30, fps=25.0, warp=warp, max_gap_frames=3, first_frame=11)


def test_the_frame_report_gives_the_median_the_95th_percentile_and_the_largest_time() -> None:
    ticks = iter(np.arange(0.0, 1000.0, 0.001))
    clock = StageClock(lambda: float(next(ticks)))
    for _ in range(20):
        with clock.frame(), clock.stage("work"):
            pass
    report = clock.frame_report()
    assert list(report) == ["work", FRAME_TOTAL]
    assert report["work"]["n"] == 20
    assert report["work"]["p50_ms"] == pytest.approx(1.0)
    assert report[FRAME_TOTAL]["max_ms"] == pytest.approx(3.0)
    with pytest.raises(ValueError, match="already open"), clock.frame(), clock.frame():
        pass
    assert StageClock().frame_report() == {}


# --- Lines of a stream, and the command -------------------------------------------------------


@pytest.fixture(scope="module")
def clip_on_disk(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, SyntheticScene]:
    scene = SyntheticScene(n_frames=N_FRAMES, seed=3)
    root = tmp_path_factory.mktemp("data")
    clip_dir = root / "SoccerNetGS-valid" / CLIP_ID
    (clip_dir / "img1").mkdir(parents=True)
    for frame in range(N_FRAMES):
        assert cv2.imwrite(str(clip_dir / "img1" / f"{frame + 1:06d}.png"), scene.image(frame))
    (clip_dir / LABEL_NAME).write_text(json.dumps(scene.gsr_labels()), encoding="utf-8")
    return root, scene


@pytest.fixture(scope="module")
def clip_runs(tmp_path_factory: pytest.TempPathFactory, clip_on_disk: tuple[Path, SyntheticScene]) -> tuple[Path, Path]:
    root, scene = clip_on_disk
    work = tmp_path_factory.mktemp("tools")
    config = str(REPO / "configs" / "engines" / "detector_calibration.yaml")
    people, balls = fake_detector_answers(scene)
    detector = []
    for item in people:
        rows = [{"box_xyxy": [p.box.x1, p.box.y1, p.box.x2, p.box.y2], "score": 0.9, "label": "person"} for p in item.people]
        rows += [
            {"box_xyxy": [b.box.x1, b.box.y1, b.box.x2, b.box.y2], "score": b.score, "label": "sports ball"}
            for b in balls.get(item.frame, ())
        ]
        detector.append({"detections": rows})
    calibration = [
        {"homography_pitch_to_image": np.linalg.inv(scene.homography(frame)).tolist(), "pitch_origin": "centre"}
        for frame in range(N_FRAMES)
    ]
    paths = []
    for tool, payloads in (("rfdetr", detector), ("pnlcalib", calibration)):
        job = work / tool / "job.json"
        clip = root / "SoccerNetGS-valid" / CLIP_ID
        arguments = ["clip-job", "--tool", tool, "--clip", str(clip), "--root", str(root), "--out", str(job)]
        assert cli.main([*arguments, "--config", config]) == 0
        read = run_tool.read_job(job, tool)
        frames = [{"clip": CLIP_ID, "frame": frame, "seconds": 0.02, "payload": payload} for frame, payload in enumerate(payloads)]
        out = work / tool / "run.json"
        run_tool.write_clip_run(out, read, run_tool.file_sha256(job), {"device": "stand-in"}, frames)
        paths.append(out)
    return paths[0], paths[1]


def test_the_lines_of_two_clip_runs_read_back_as_the_inputs_of_the_stand_ins(
    clip_on_disk: tuple[Path, SyntheticScene], clip_runs: tuple[Path, Path]
) -> None:
    _, scene = clip_on_disk
    lines = list(clip_run_lines(load_clip_run(clip_runs[0]), load_clip_run(clip_runs[1])))
    assert len(lines) == N_FRAMES + 1
    assert json.loads(lines[0])["format"] == STREAM_FORMAT
    header, inputs = read_stream_lines(iter(lines))
    assert (header.clip, header.detector, header.calibration) == (CLIP_ID, CONFIG.detector, CONFIG.calibration)
    items = list(inputs)
    people, balls = fake_detector_answers(scene)
    assert [len(item.people) for item in items] == [len(p.people) for p in people]
    assert [len(item.balls) for item in items] == [len(balls.get(frame, ())) for frame in range(N_FRAMES)]
    assert all(item.camera.valid and item.detector_seconds == 0.02 for item in items)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda lines: lines[:1] + lines[2:], "frame 0 was expected"),
        (lambda lines: [lines[0], "{not json", *lines[1:]], "line 2: no JSON"),
        (lambda lines: [json.dumps({"format": "other"}), *lines[1:]], "does not have the format"),
        (lambda lines: [lines[0], json.dumps({"frame": 0})], "line 2"),
        (lambda lines: [], "no header line"),
    ],
)
def test_a_wrong_line_of_a_stream_is_refused_with_its_line_number(
    clip_runs: tuple[Path, Path], change: Any, message: str
) -> None:
    lines = list(clip_run_lines(load_clip_run(clip_runs[0]), load_clip_run(clip_runs[1])))
    with pytest.raises(ValueError, match=message):
        _, inputs = read_stream_lines(iter(change(lines)))
        list(inputs)


def test_stream_command_gives_the_same_result_from_clip_runs_and_from_lines(
    tmp_path: Path, clip_on_disk: tuple[Path, SyntheticScene], clip_runs: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = clip_on_disk
    monkeypatch.setattr(cli, "render_clip", lambda source, result, out_path: out_path)
    for variable in cli.KEY_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    clip = root / "SoccerNetGS-valid" / CLIP_ID
    base = ["stream", "--clip", str(clip), "--labels", str(clip / LABEL_NAME), "--allow-dirty", "--delay", "5"]
    runs = ["--detections", str(clip_runs[0]), "--calibration", str(clip_runs[1])]
    assert cli.main([*base, *runs, "--out", str(tmp_path / "a")]) == 0
    printed = capsys.readouterr().out
    assert "delay 5 frames (0.20 s)" in printed
    assert "measured time per frame (median / 95th percentile / largest, milliseconds):" in printed
    assert "detector (as the tool measured it, not in the total): 20.00 / 20.00 / 20.00" in printed
    assert "hold no detector and no calibration" in printed
    lines_file = tmp_path / "stream.jsonl"
    lines_file.write_text(
        "\n".join(clip_run_lines(load_clip_run(clip_runs[0]), load_clip_run(clip_runs[1]))) + "\n", encoding="utf-8"
    )
    assert cli.main([*base, "--lines", str(lines_file), "--out", str(tmp_path / "b")]) == 0
    first = json.loads((tmp_path / "a" / CLIP_ID / "result.json").read_text(encoding="utf-8"))
    second = json.loads((tmp_path / "b" / CLIP_ID / "result.json").read_text(encoding="utf-8"))
    assert first["result"] == second["result"]
    assert first["provenance"]["stream"] == {
        "delay_frames": 5,
        "history_frames": 123,
        "kit_warmup_frames": 25,
        "early_dissolve_windows": EARLY_DISSOLVE_WINDOWS,
    }
    # The pictures and the pair tests of the motion run on the frame feeder: their times are apart.
    assert set(first["frame_timing"]) >= {"motion", "camera", "kit_colour", "tracks", "decide", "total"}
    assert set(first["feeder_timing"]) == {"read", "motion"}
    assert first["prediction_file"]["problems"] == []
    assert first["result"]["engines"]["tracks"] == "code:stream"


def test_stream_with_the_overload_policy_on_a_pipe_and_no_line_late_gives_the_result_without_it(
    clip_on_disk: tuple[Path, SyntheticScene], clip_runs: tuple[Path, Path], tmp_path: Path
) -> None:
    """The policy is on (a wait of 600 s, far above any wait of a line here) and drops nothing."""
    root, _ = clip_on_disk
    clip = root / "SoccerNetGS-valid" / CLIP_ID
    lines_file = tmp_path / "stream.jsonl"
    lines_file.write_text(
        "\n".join(clip_run_lines(load_clip_run(clip_runs[0]), load_clip_run(clip_runs[1]))) + "\n", encoding="utf-8"
    )
    base = ["stream", "--clip", str(clip), "--labels", str(clip / LABEL_NAME), "--allow-dirty", "--delay", "5"]
    assert cli.main([*base, "--lines", str(lines_file), "--out", str(tmp_path / "off")]) == 0
    environment = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    done = subprocess.run(
        [sys.executable, "-m", "pitchvlm.cli", *base, "--lines", "-", "--max-behind", "600", "--out", str(tmp_path / "on")],
        input=lines_file.read_text(encoding="utf-8"), capture_output=True, text=True, env=environment, timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "a line that waited more than 600 s): 0 of 60 frames dropped (0 not used, 0 measured but decided late)" in done.stdout
    off = json.loads((tmp_path / "off" / CLIP_ID / "result.json").read_text(encoding="utf-8"))
    on = json.loads((tmp_path / "on" / CLIP_ID / "result.json").read_text(encoding="utf-8"))
    assert on["overload"]["dropped"] == 0 and on["overload"]["dropped_frames"] == [] and "overload" not in off
    assert on["result"]["engines"].pop("overload") == (
        "drop a frame whose line waited more than 300 s, and a decision due with a line that waited more than 600 s"
    )
    assert on["result"] == off["result"]
    for name in ("identity", "diagnostics", "transitions_known_at", "missing_parts", "prediction_file"):
        assert on[name] == off[name]


def test_stream_with_a_wait_no_line_can_keep_drops_every_frame_and_says_so_in_every_output(
    clip_on_disk: tuple[Path, SyntheticScene], clip_runs: tuple[Path, Path], tmp_path: Path
) -> None:
    """A wait of 1 ns: every line waits longer, whatever the speed of the machine, so every frame but 0 is not used,
    and frame 0, whose decision is due with a late line, gets no state either."""
    root, _ = clip_on_disk
    clip = root / "SoccerNetGS-valid" / CLIP_ID
    text = "\n".join(clip_run_lines(load_clip_run(clip_runs[0]), load_clip_run(clip_runs[1]))) + "\n"
    live = tmp_path / "live.jsonl"
    command = [sys.executable, "-m", "pitchvlm.cli", "stream", "--clip", str(clip), "--labels", str(clip / LABEL_NAME),
               "--allow-dirty", "--delay", "5", "--lines", "-", "--max-behind", "0.000000001", "--live-out", str(live),
               "--out", str(tmp_path / "out")]
    done = subprocess.run(command, input=text, capture_output=True, text=True, timeout=300, check=False,
                          env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert done.returncode == 1, done.stderr  # no frame has a state: the result is empty
    assert "60 of 60 frames dropped (59 not used, 1 measured but decided late)" in done.stdout
    document = json.loads((tmp_path / "out" / CLIP_ID / "result.json").read_text(encoding="utf-8"))
    assert document["overload"]["dropped_frames"] == list(range(60)) and document["overload"]["decided_late"] == [0]
    assert all(frame["players"] == [] and frame["ball"] is None for frame in document["result"]["frames"])
    assert all(record["source"] == "none" for record in document["result"]["homographies"])
    assert document["latency"]["decided_live"] == 0
    lines = [json.loads(line) for line in live.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2 and lines[0]["overload_max_behind_s"] == 1e-9
    assert lines[1] == {"end": True, "frames": 0, "written": 0, "dropped": 0, "overload_dropped": 60}


@pytest.mark.slow
@pytest.mark.parametrize("policy", [[], ["--max-behind", "600"]])
def test_one_stop_signal_ends_a_live_stream_with_its_frame_feeder_cleanly(
    policy: list[str], clip_on_disk: tuple[Path, SyntheticScene], clip_runs: tuple[Path, Path], tmp_path: Path
) -> None:
    """Ctrl-C while the stream waits for its next line (the pipe stays open): exit 2, the line of a stop, no traceback,
    no result file, and a live output without its end line, as before the frame feeder."""
    root, _ = clip_on_disk
    clip = root / "SoccerNetGS-valid" / CLIP_ID
    lines = list(clip_run_lines(load_clip_run(clip_runs[0]), load_clip_run(clip_runs[1])))
    live = tmp_path / "live.jsonl"
    command = [sys.executable, "-m", "pitchvlm.cli", "stream", "--clip", str(clip), "--labels", str(clip / LABEL_NAME),
               "--allow-dirty", "--delay", "5", "--lines", "-", "--live-out", str(live), "--out", str(tmp_path / "out"),
               *policy]
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
    )
    try:
        assert process.stdin is not None
        process.stdin.write("\n".join(lines[:21]) + "\n")  # the header and frames 0 to 19: frames 0 to 14 are decided
        process.stdin.flush()
        deadline = time.monotonic() + 120.0
        while not (live.exists() and live.read_text(encoding="utf-8").count("\n") >= 16):
            if process.poll() is not None:
                pytest.fail(f"the stream ended before the signal: {process.communicate()[1]}")
            assert time.monotonic() < deadline, "the stream did not decide its frames"
            time.sleep(0.05)
        process.send_signal(signal.SIGINT)
        _, errors = process.communicate(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    assert process.returncode == 2
    assert "error: the command was interrupted (Ctrl-C)" in errors
    assert "Traceback" not in errors
    assert not (tmp_path / "out" / CLIP_ID / "result.json").exists()
    written = [json.loads(line) for line in live.read_text(encoding="utf-8").splitlines()]
    assert written[0]["format"] == "pitchvlm-live-1" and all("end" not in line for line in written)
    assert [line["frame"] for line in written[1:]] == list(range(len(written) - 1))


def test_stream_command_runs_the_synthetic_clip_and_refuses_a_model_engine(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "render_clip", lambda source, result, out_path: out_path)
    out = tmp_path / "out"
    assert cli.main(["stream", "--synthetic", "--frames", "30", "--delay", "0", "--out", str(out), "--allow-dirty"]) == 0
    printed = capsys.readouterr().out
    # With no delay a person has no state in the first frame of a track: a track needs 2 observations.
    assert "identity against the synthetic truth: 96% of a true track under one id, 0 id switches" in printed
    document = json.loads((out / CLIP_ID / "result.json").read_text(encoding="utf-8"))
    assert document["result"]["engines"]["delay"] == "0 frames (0.00 s)"
    assert document["identity"]["switches"] == 0
    settings = tmp_path / "model.yaml"
    settings.write_text("engines:\n  people: model\n", encoding="utf-8")
    assert cli.main(["stream", "--synthetic", "--config", str(settings), "--out", str(out), "--allow-dirty"]) == 2
    assert "needs the people engine detector" in capsys.readouterr().err


# --- Second read of 2026-11-23 -------------------------------------------------------------------


def test_the_control_drops_the_people_off_the_pitch_of_frames_that_had_no_camera_at_arrival() -> None:
    """Frames 0 to 4 have no camera, and one more person stands off the pitch there.

    At arrival the stream has no camera to test them; the clip path takes the camera of frame 5
    and drops the person. The window of the decision drops them too, so the control still equals
    the clip path (before the fix: 48 of 60 frames differed).
    """
    scene, images = clip_of("plain", 3, n_frames=N_FRAMES)
    source = ArraySource(scene.info, images)
    people, balls = fake_detector_answers(scene)
    extra = PersonDet(box=Box(x1=5, y1=650, x2=35, y2=715), role="player", team_key="none")
    people = [FramePeople(frame=p.frame, people=(*p.people, extra)) if p.frame < 5 else p for p in people]
    cameras = [HomographyRecord(frame=c.frame) if c.frame < 5 else c for c in fake_calibration_answers(scene)]
    offline = run_clip(source, None, CONFIG, tools=ToolAnswers(people=people, balls=balls, cameras=cameras))
    to_pitch = _pitch_map(offline.homographies)
    kits, _ = clip_kit_clusters(source, [_people_on_pitch(item, to_pitch, CONFIG.pitch_margin_m) for item in people])
    result = run_stream(
        stand_in_inputs(people, balls, cameras),
        source.read,
        CONFIG,
        StreamSettings(delay_frames=None, history_frames=None),
        clip_id=scene.info.clip_id,
        fps=scene.info.fps,
        size=(scene.info.width, scene.info.height),
        kit_clusters=kits,
    )
    assert result.homographies == offline.homographies
    assert result.frames == offline.frames


def test_forgetting_keeps_what_the_matching_needs_with_a_short_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """A history of 2 frames and no camera reach: the stream that forgets equals the one that keeps everything."""
    scene = SyntheticScene(n_frames=80, seed=3)
    people, balls = fake_detector_answers(scene, miss_rate=0.1, box_noise_px=2.0, seed=3)
    cameras = fake_calibration_answers(scene)
    config = CONFIG.model_copy(update={"max_gap_frames": 0})
    images = [scene.image(frame) for frame in range(80)]

    def frames() -> tuple[Any, ...]:
        return run_stream(
            stand_in_inputs(people, balls, cameras),
            lambda frame: images[frame],
            config,
            StreamSettings(delay_frames=12, history_frames=2),
            clip_id="c",
            fps=25.0,
            size=(1280, 720),
        ).frames

    forgetting = frames()
    monkeypatch.setattr(DetectorStream, "_forget", lambda self, next_frame: None)
    assert forgetting == frames()


def test_a_line_number_counts_the_blank_lines_and_a_frame_must_be_a_whole_number(clip_runs: tuple[Path, Path]) -> None:
    lines = list(clip_run_lines(load_clip_run(clip_runs[0]), load_clip_run(clip_runs[1])))
    with pytest.raises(ValueError, match="line 4: no JSON"):
        _, inputs = read_stream_lines(iter([lines[0], "", "  ", "{not json"]))
        list(inputs)
    first = json.loads(lines[1])
    first["frame"] = 0.0
    with pytest.raises(ValueError, match="frame 0 was expected, the line holds frame 0.0"):
        list(read_stream_lines(iter([lines[0], json.dumps(first)]))[1])


def test_a_wrong_history_is_a_usage_error_and_the_ball_tracker_takes_an_empty_clip() -> None:
    with pytest.raises(SystemExit) as stop:
        cli.build_parser().parse_args(["stream", "--synthetic", "--out", "o", "--history", "abc"])
    assert stop.value.code == 2
    assert cli.build_parser().parse_args(["stream", "--synthetic", "--out", "o", "--history", "clip"]).history is None
    empty = track_ball({}, 0, fps=25.0, warp=lambda src, dst, xy: xy, max_gap_frames=3)
    assert dict(empty.boxes) == {} and empty.observed == frozenset()


# --- Round 6: the early dissolve test --------------------------------------------------------------------


def test_the_shot_meter_takes_the_final_span_of_a_dissolve_before_its_window_test_ends() -> None:
    """The span of the open run is taken once it came out the same for 3 judged windows; it is the final span,
    and it is known within 25 frames of the start of the dissolve (the final one about a window later)."""
    scene, images = clip_of("dissolve", 0, n_frames=200)
    meter = ShotMeter((scene.info.width, scene.info.height))
    early_at, early, final_at = None, None, None
    for frame, image in enumerate(images):
        meter.add(image)
        if frame == 0:
            continue
        if meter.update_transitions() and final_at is None:
            final_at = frame
        found = meter.open_transition(3)
        if found is not None and early is None:
            early_at, early = frame, found
    tests = meter.finish()
    start = tests.transitions[0][0]
    assert early == tests.transitions[0]
    assert early_at is not None and final_at is not None and early_at <= start + 25 < final_at
    with pytest.raises(ValueError, match="at least 1"):
        meter.open_transition(0)


def test_the_shot_meter_takes_nothing_early_on_a_clip_without_a_cut() -> None:
    scene, images = clip_of("plain", 0, n_frames=120)
    meter = ShotMeter((scene.info.width, scene.info.height))
    for frame, image in enumerate(images):
        meter.add(image)
        if frame:
            meter.update_transitions()
            assert meter.open_transition(1) is None


@pytest.mark.slow
def test_with_the_early_test_a_delay_of_25_gives_no_state_on_the_blend_frames_of_a_dissolve() -> None:
    """Without it the states of the blend frames were given before the dissolve was known (round 5)."""
    scene, images = clip_of("dissolve", 0, n_frames=180)
    source = ArraySource(scene.info, images)
    people, balls = fake_detector_answers(scene)
    cameras = fake_calibration_answers(scene)
    offline = run_clip(source, None, CONFIG, tools=ToolAnswers(people=people, balls=balls, cameras=cameras))
    blends = sorted(MotionField([None] * 179, offline.cuts).transition_frames())
    assert blends
    given = {}
    for windows in (None, 3):
        settings = StreamSettings(delay_frames=25, history_frames=default_history_frames(CONFIG, 25.0),
                                  early_dissolve_windows=windows)
        result = run_stream(stand_in_inputs(people, balls, cameras), source.read, CONFIG, settings,
                            clip_id="c", fps=25.0, size=(scene.info.width, scene.info.height))
        given[windows] = sum(len(result.frames[frame].players) for frame in blends)
        assert result.cuts == offline.cuts
    assert given[None] > 0 and given[3] == 0


# --- Round 6: second read of stream.py -------------------------------------------------------------------


def test_the_control_matches_the_first_frame_of_a_shot_without_a_camera_with_a_later_camera() -> None:
    """Second read of 2026-11-23, finding 1: frame 30 starts a shot and has no camera. At arrival the stream has no
    camera for it; the clip path takes the camera of frame 31. Before the fix 29 or 30 of 60 frames differed."""
    scene, images = clip_of("cut", 1, n_frames=60)
    source = ArraySource(scene.info, images)
    people, balls = fake_detector_answers(scene, miss_rate=0.3, box_noise_px=4.0, seed=1)
    cameras = [HomographyRecord(frame=camera.frame) if camera.frame == 30 else camera for camera in fake_calibration_answers(scene)]
    offline = run_clip(source, None, CONFIG, tools=ToolAnswers(people=people, balls=balls, cameras=cameras))
    assert offline.cuts == (30,)
    _, kits = offline_and_kits_of(source, people, balls, cameras)
    result = run_stream(stand_in_inputs(people, balls, cameras), source.read, CONFIG,
                        StreamSettings(delay_frames=None, history_frames=None), clip_id=scene.info.clip_id,
                        fps=scene.info.fps, size=(scene.info.width, scene.info.height), kit_clusters=kits)
    assert result.frames == offline.frames


def offline_and_kits_of(source: Any, people: Any, balls: Any, cameras: Any) -> tuple[Any, Any]:
    offline = run_clip(source, None, CONFIG, tools=ToolAnswers(people=people, balls=balls, cameras=cameras))
    to_pitch = _pitch_map(offline.homographies)
    on_pitch = [_people_on_pitch(item, to_pitch, CONFIG.pitch_margin_m) for item in people]
    clusters, _ = clip_kit_clusters(source, on_pitch)
    return offline, clusters


def test_a_ball_candidate_of_another_frame_is_refused_at_its_push() -> None:
    """Second read, finding 4: before, the error came 5 frames later, named frame 0 and left the stream broken."""
    scene = SyntheticScene(n_frames=8, seed=2)
    people, balls = fake_detector_answers(scene)
    item = next(stand_in_inputs(people, balls, fake_calibration_answers(scene)))
    wrong = BallCandidate(frame=7, box=Box(x1=1.0, y1=1.0, x2=5.0, y2=5.0), score=0.9)
    detector = DetectorStream(CONFIG, StreamSettings(delay_frames=5, history_frames=5), clip_id="c", fps=25.0,
                              size=(scene.info.width, scene.info.height))
    with pytest.raises(ValueError, match="a ball candidate of frame 0 names another frame"):
        detector.push(StreamInput(frame=0, people=item.people, balls=(wrong,), camera=item.camera), scene.image(0))
    assert detector.n_frames == 0
    assert detector.push(item, scene.image(0)) == []


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("[" * 100_000 + "]" * 100_000, "line 2: no JSON"),  # finding 3: RecursionError escaped
        (None, "line 2: Infinity is not a number of JSON"),  # finding 2: Infinity passed
    ],
)
def test_a_line_that_breaks_the_reader_is_refused_with_its_line_number(
    clip_runs: tuple[Path, Path], text: str | None, words: str
) -> None:
    lines = list(clip_run_lines(load_clip_run(clip_runs[0]), load_clip_run(clip_runs[1])))
    if text is None:
        text = lines[1].replace('"seconds": 0.02', '"seconds": Infinity', 1)
        assert "Infinity" in text
    with pytest.raises(ValueError, match=words):
        list(read_stream_lines(iter([lines[0], text]))[1])
