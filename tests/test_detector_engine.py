"""Tests of the engines ``detector`` and ``calibration`` of the pipeline and of the command line.

The tools are stand-ins (``fallbacks.fake``): the true people of a synthetic scene as a detector
gives them (no role, no team, no number), a stand-in ball, and the true camera. The clip run files
of the command-line tests are written with the part writer of ``tools/run_tool.py``, so the
package reads what the script writes. Nothing here says how a real detector behaves.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
import pytest

from pitchvlm import cli, kitcolour, pipeline
from pitchvlm.config import EngineSettings, PipelineConfig, TrackingConfig, load_config
from pitchvlm.evalx.metrics import identity_scores, position_errors_m, summarize
from pitchvlm.export_gsr import check_predictions, prediction_items
from pitchvlm.fallbacks.fake import fake_ball_position, fake_calibration_answers, fake_detector_answers
from pitchvlm.frames import keyframe_indices
from pitchvlm.geometry import apply_homography, matrix_to_rows
from pitchvlm.pipeline import ToolAnswers, every_frame_tracking, run_clip
from pitchvlm.render import RadarRenderer, draw_ball
from pitchvlm.schemas import BallState, Box, ClipResult, FrameState, HomographyRecord, Track
from pitchvlm.synth import RetimedScene, SyntheticScene, SyntheticSource
from pitchvlm.timing import StageClock
from pitchvlm.tracking import finalize_tracks, split_team_swaps
from pitchvlm.vlm.fake import FakeBackend
from pitchvlm.vlm.types import TextPart, VLMRequest, VLMResponse

REPO = Path(__file__).resolve().parents[1]
ENGINE_FILES = REPO / "configs" / "engines"
CLIP_ID = "SNGS-999"
LABEL_NAME = "Labels-GameState.json"
N_FRAMES = 75
DETECTOR_ONLY = EngineSettings(people="detector", pitch_mapping="calibration", number="none")


def load_script() -> ModuleType:
    """Load ``tools/run_tool.py`` as a module (it is not part of a package)."""
    spec = importlib.util.spec_from_file_location("run_tool_for_engines", REPO / "tools" / "run_tool.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_tool_for_engines"] = module
    spec.loader.exec_module(module)
    return module


run_tool = load_script()


class Recorder:
    """A backend that counts the requests of each step and passes them to the fake model."""

    def __init__(self, inner: FakeBackend) -> None:
        self.inner = inner
        self.requests: list[VLMRequest] = []

    def generate(self, request: VLMRequest) -> VLMResponse:
        self.requests.append(request)
        return self.inner.generate(request)

    def steps(self) -> Counter[str]:
        return Counter(request.meta.step for request in self.requests)


@pytest.fixture(scope="module")
def scene() -> SyntheticScene:
    return SyntheticScene(n_frames=N_FRAMES, seed=3)


def tools_of(scene: SyntheticScene, engines: EngineSettings) -> ToolAnswers:
    people, balls = fake_detector_answers(scene) if engines.people == "detector" else (None, None)
    cameras = fake_calibration_answers(scene) if engines.pitch_mapping == "calibration" else None
    return ToolAnswers(people=people, balls=balls, cameras=cameras)


@pytest.fixture(scope="module")
def detector_result(scene: SyntheticScene) -> tuple[ClipResult, StageClock]:
    clock = StageClock()
    config = PipelineConfig(engines=DETECTOR_ONLY)
    return run_clip(SyntheticSource(scene), None, config, tools=tools_of(scene, DETECTOR_ONLY), clock=clock), clock


def truth(scene: SyntheticScene) -> list[Any]:
    return [scene.frame_state(frame) for frame in range(scene.info.n_frames)]


# --- Pipeline: detector and calibration, no model -------------------------------------


def test_detector_engine_places_and_follows_every_person(
    scene: SyntheticScene, detector_result: tuple[ClipResult, StageClock]
) -> None:
    result, _ = detector_result
    errors = summarize(position_errors_m(result.frames, truth(scene)))
    identity = identity_scores(result.frames, truth(scene))
    assert errors["median"] < 0.05
    assert identity.coverage > 0.95
    assert identity.switches == 0
    assert identity.track_ratio <= 1.05


def test_kit_colours_give_the_team_side_and_the_special_roles(
    scene: SyntheticScene, detector_result: tuple[ClipResult, StageClock]
) -> None:
    """Every predicted person is matched to the nearest true person of the frame: role and side must agree."""
    result, _ = detector_result
    checked = Counter()
    for state, true_state in zip(result.frames, truth(scene), strict=True):
        for player in state.players:
            nearest = min(
                true_state.players,
                key=lambda true: (true.pitch_x - player.pitch_x) ** 2 + (true.pitch_y - player.pitch_y) ** 2,
            )
            assert (player.role, player.team_side) == (nearest.role, nearest.team_side)
            checked[player.role] += 1
    assert checked["player"] > 500
    assert checked["referee"] > 0
    assert checked["goalkeeper"] > 0


def test_detector_engine_has_no_number_and_names_its_engines(detector_result: tuple[ClipResult, StageClock]) -> None:
    result, _ = detector_result
    assert all(track.number_export is None and track.number_display is None for track in result.tracks)
    assert result.engines == {
        "kit_legend": "none",
        "people": "detector:rfdetr:RFDETRMedium",
        "team_role": "code:kit-colour",
        "landmarks": "none",
        "number": "none",
        "ball": "code:ball-tracker",
        "motion": "code",
        "pitch_mapping": "tool:pnlcalib",
        "tracks": "code",
        "fill": "code",
    }


def test_ball_follows_the_stand_in_ball(scene: SyntheticScene, detector_result: tuple[ClipResult, StageClock]) -> None:
    result, _ = detector_result
    with_ball = [state for state in result.frames if state.ball is not None]
    assert len(with_ball) > N_FRAMES // 2
    filled = [state for state in with_ball if not state.ball.observed]
    assert filled, "the gaps of the stand-in ball are filled by the tracker"
    for state in with_ball:
        assert state.ball.pitch_x is not None and state.ball.pitch_y is not None
        true_x, true_y = fake_ball_position(scene, state.frame)
        assert abs(state.ball.pitch_x - true_x) < 0.5 and abs(state.ball.pitch_y - true_y) < 0.5


def test_stand_in_ball_rolls_from_its_start_at_its_speed(scene: SyntheticScene) -> None:
    """The stand-in ball starts at (-10, 5) m and rolls at (6, -2) m/s: the numbers of its design.

    The expectation does not come from ``fake_ball_position``: the check reads each ball box
    that the detector answers give, maps the bottom centre of the box (the point under the
    ball) to the pitch with the true mapping of the scene, and compares that point with the
    numbers written here. The ball is left out in frames 0 to 5 and 40 to 45 of every 40.
    """
    _, balls = fake_detector_answers(scene)
    assert sorted(balls) == [frame for frame in range(N_FRAMES) if frame % 40 >= 6]
    for frame, (candidate,) in balls.items():
        seconds = frame / 25.0
        box = candidate.box
        under = np.array([[(box.x1 + box.x2) / 2.0, box.y2]])
        (pitch_x, pitch_y), = apply_homography(scene.homography(frame), under)
        assert (pitch_x, pitch_y) == pytest.approx((-10.0 + 6.0 * seconds, 5.0 - 2.0 * seconds), abs=1e-6), frame


def test_prediction_file_of_the_detector_engine_is_valid_and_has_no_ball(
    scene: SyntheticScene, detector_result: tuple[ClipResult, StageClock]
) -> None:
    result, _ = detector_result
    image_ids = {frame: f"{frame + 1:06d}" for frame in range(N_FRAMES)}
    items = prediction_items(result.frames, image_ids, jersey_as_str=True)
    assert check_predictions({"predictions": items}, expected_image_ids=image_ids.values(), jersey_as_str=True) == []
    assert {item["attributes"]["role"] for item in items} <= {"player", "goalkeeper", "referee"}
    assert all(item["attributes"]["jersey"] is None for item in items)


def test_every_stage_of_the_detector_engine_is_timed(detector_result: tuple[ClipResult, StageClock]) -> None:
    _, clock = detector_result
    report = clock.report(N_FRAMES)
    assert set(report) == {"motion", "pitch_mapping", "paint", "kit_colour", "tracks", "votes_fill", "ball", "total"}
    assert report["total"]["seconds"] == pytest.approx(
        sum(report[stage]["seconds"] for stage in report if stage != "total")
    )


# --- Hard cuts -----------------------------------------------------------------------

#: The clip frame that starts the second shot of the clip with a cut: scene frames 0 to 53, then 154 to 207.
CUT = 54


@pytest.fixture(scope="module")
def cut_scene() -> RetimedScene:
    return RetimedScene([*range(0, CUT), *range(CUT + 100, 2 * CUT + 100)], seed=3)


@pytest.fixture(scope="module")
def cut_result(cut_scene: RetimedScene) -> ClipResult:
    """The detector engine on the clip with a cut. The ball candidates of frames 52 to 56 are taken out."""
    tools = tools_of(cut_scene, DETECTOR_ONLY)
    assert tools.balls is not None
    balls = {frame: found for frame, found in tools.balls.items() if not CUT - 2 <= frame <= CUT + 2}
    tools = ToolAnswers(people=tools.people, balls=balls, cameras=tools.cameras)
    return run_clip(SyntheticSource(cut_scene), None, PipelineConfig(engines=DETECTOR_ONLY), tools=tools)


def test_no_track_of_the_detector_engine_crosses_a_cut(cut_result: ClipResult) -> None:
    """Without the tracking per shot, 3 tracks of this clip cross the cut (measured)."""
    assert cut_result.cuts == (CUT,)
    assert [track.track_id for track in cut_result.tracks if track.first_frame < CUT <= track.last_frame] == []
    assert [track.track_id for track in cut_result.tracks] == list(range(len(cut_result.tracks)))


def test_ball_is_not_filled_across_a_cut(cut_result: ClipResult) -> None:
    """The gap of 5 frames around the cut is shorter than the fill limit, but it holds the cut."""
    with_ball = {state.frame for state in cut_result.frames if state.ball is not None}
    assert with_ball.isdisjoint(range(CUT - 2, CUT + 3))
    # The ball is followed again in both shots.
    assert {CUT - 3, CUT + 5} <= with_ball


def test_cameras_of_the_calibration_get_the_paint_alignment_of_the_keyframes(
    cut_scene: RetimedScene, cut_result: ClipResult
) -> None:
    keyframes = keyframe_indices(cut_scene.info.n_frames, cut_scene.info.fps, PipelineConfig().keyframe_interval_s)
    assert [check.frame for check in cut_result.paint] == keyframes
    # The cameras of the stand-in are exact, so their lines lie on the painted lines.
    assert all(check.score is not None and check.score > 0.9 for check in cut_result.paint)


# --- Track repairs (arms of gate G4) ----------------------------------------------------

#: The detector and the calibration with both track repairs on.
DETECTOR_REPAIRS = PipelineConfig(
    engines=DETECTOR_ONLY, tracking=TrackingConfig(split_team_swaps=True, team_slots=True)
)


def test_split_of_team_swaps_runs_per_shot_after_the_kit_roles(
    cut_scene: RetimedScene, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The split gets the fragments of one shot, whose goalkeepers and referees have no team key."""
    calls: list[tuple[set[bool], set[tuple[str, str]]]] = []
    min_votes: list[int] = []

    def watched(tracks: Sequence[Track], **options: int) -> list[Track]:
        shots = {obs.frame >= CUT for track in tracks for obs in track.observations}
        roles = {(obs.role, obs.team_key) for track in tracks for obs in track.observations}
        calls.append((shots, roles))
        min_votes.append(options["min_votes"])
        return split_team_swaps(tracks, **options)

    monkeypatch.setattr(pipeline, "split_team_swaps", watched)
    tools = tools_of(cut_scene, DETECTOR_ONLY)
    run_clip(SyntheticSource(cut_scene), None, PipelineConfig(engines=DETECTOR_ONLY), tools=tools)
    assert calls == []
    repaired = run_clip(SyntheticSource(cut_scene), None, DETECTOR_REPAIRS, tools=tools)
    assert [shots for shots, _ in calls] == [{False}, {True}]
    # A team-key change must hold for 4 keyframe intervals of 0.48 s, 12 frames each at 25 fps.
    assert min_votes == [48, 48]
    # kit_roles ran first: a goalkeeper carries the team key none, so the split never cuts it.
    roles = set().union(*(roles for _, roles in calls))
    assert ("goalkeeper", "none") in roles
    assert ("goalkeeper", "A") not in roles and ("goalkeeper", "B") not in roles
    assert [track.track_id for track in repaired.tracks if track.first_frame < CUT <= track.last_frame] == []


@pytest.mark.slow
def test_team_slots_link_the_fragments_of_the_detector_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """On 250 frames the plain detector engine breaks 4 people into two tracks; the slots link them right."""
    long_scene = SyntheticScene(n_frames=250, seed=3)
    tools = tools_of(long_scene, DETECTOR_ONLY)
    seen_cuts: list[tuple[int, ...]] = []

    def watched(*args: Any, **kwargs: Any) -> list[Track]:
        seen_cuts.append(tuple(kwargs["cuts"]))
        return finalize_tracks(*args, **kwargs)

    monkeypatch.setattr(pipeline, "finalize_tracks", watched)
    plain = run_clip(SyntheticSource(long_scene), None, PipelineConfig(engines=DETECTOR_ONLY), tools=tools)
    slots = PipelineConfig(engines=DETECTOR_ONLY, tracking=TrackingConfig(team_slots=True))
    linked = run_clip(SyntheticSource(long_scene), None, slots, tools=tools)
    assert seen_cuts == [(), ()]
    players = Counter(track.team_side for track in linked.tracks if track.role == "player")
    assert set(players.values()) == {10}
    before = identity_scores(plain.frames, truth(long_scene))
    after = identity_scores(linked.frames, truth(long_scene))
    assert (before.switches, after.switches) == (4, 0)
    assert after.track_ratio == pytest.approx(1.0)
    assert before.track_ratio > 1.1


def test_team_slots_link_no_track_of_the_detector_engine_across_a_cut(cut_scene: RetimedScene) -> None:
    repaired = run_clip(SyntheticSource(cut_scene), None, DETECTOR_REPAIRS, tools=tools_of(cut_scene, DETECTOR_ONLY))
    assert repaired.cuts == (CUT,)
    assert [track.track_id for track in repaired.tracks if track.first_frame < CUT <= track.last_frame] == []


def test_detector_run_writes_and_prints_the_tracking_values_without_labels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["demo", "--engine", "detector", "--frames", "50", "--out", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "tracking values without labels (no gate uses them):" in printed
    summary = json.loads((tmp_path / cli.SUMMARY_NAME).read_text(encoding="utf-8"))
    assert summary["diagnostics"]["tracks"] == summary["n_tracks"]


def test_model_engine_has_no_ball(scene: SyntheticScene) -> None:
    result = run_clip(SyntheticSource(scene), FakeBackend(scene))
    assert all(state.ball is None for state in result.frames)


# --- Pipeline: mixed engines ------------------------------------------------------------


def test_detector_with_model_landmarks_asks_only_the_landmarks(scene: SyntheticScene) -> None:
    engines = EngineSettings(people="detector", pitch_mapping="model", number="none")
    backend = Recorder(FakeBackend(scene))
    result = run_clip(SyntheticSource(scene), backend, PipelineConfig(engines=engines), tools=tools_of(scene, engines))
    assert set(backend.steps()) == {"landmarks"}
    assert result.engines["landmarks"] == "model:gemini-4-argon"
    assert result.engines["pitch_mapping"] == "code"
    assert summarize(position_errors_m(result.frames, truth(scene)))["median"] < 0.2


def test_detector_with_model_numbers_names_no_team_in_the_number_request(scene: SyntheticScene) -> None:
    engines = EngineSettings(people="detector", pitch_mapping="calibration", number="model")
    backend = Recorder(FakeBackend(scene))
    result = run_clip(SyntheticSource(scene), backend, PipelineConfig(engines=engines), tools=tools_of(scene, engines))
    steps = backend.steps()
    assert set(steps) == {"kit_legend", "number"}
    assert steps["kit_legend"] == 1
    number_requests = [request for request in backend.requests if request.meta.step == "number"]
    texts = [" ".join(part.text for part in request.parts if isinstance(part, TextPart)) for request in number_requests]
    assert len(texts) == steps["number"] > 0
    assert all("The person belongs to an unknown team." in text for text in texts)
    numbered = [track for track in result.tracks if track.role == "player" and track.number_export is not None]
    assert len(numbered) >= 10


def test_model_people_with_the_calibration_asks_no_landmarks(scene: SyntheticScene) -> None:
    engines = EngineSettings(people="model", pitch_mapping="calibration", number="model")
    backend = Recorder(FakeBackend(scene))
    result = run_clip(SyntheticSource(scene), backend, PipelineConfig(engines=engines), tools=tools_of(scene, engines))
    assert "landmarks" not in backend.steps()
    assert result.engines["pitch_mapping"] == "tool:pnlcalib"
    assert all(state.ball is None for state in result.frames)
    assert summarize(position_errors_m(result.frames, truth(scene)))["median"] < 0.2


def test_camera_that_is_not_plausible_is_refused_and_its_frame_borrows_a_neighbour(scene: SyntheticScene) -> None:
    tools = tools_of(scene, DETECTOR_ONLY)
    cameras = list(tools.cameras or ())
    mirrored = np.diag([-1.0, 1.0, 1.0]) @ scene.homography(10)  # pitch x changes its sign
    cameras[10] = HomographyRecord(
        frame=10, matrix=matrix_to_rows(mirrored / np.linalg.norm(mirrored)), source="landmarks"
    )
    tools = ToolAnswers(people=tools.people, balls=tools.balls, cameras=cameras)
    result = run_clip(SyntheticSource(scene), None, PipelineConfig(engines=DETECTOR_ONLY), tools=tools)
    assert result.homographies[10].source == "propagated"
    assert result.homographies[9].source == "landmarks"


# --- Pipeline: wrong inputs -------------------------------------------------------------


def test_engine_without_its_tool_answers_is_refused(scene: SyntheticScene) -> None:
    source = SyntheticSource(scene)
    full = tools_of(scene, DETECTOR_ONLY)
    config = PipelineConfig(engines=DETECTOR_ONLY)
    with pytest.raises(ValueError, match="have no people"):
        run_clip(source, None, config, tools=ToolAnswers(balls=full.balls, cameras=full.cameras))
    with pytest.raises(ValueError, match="have no cameras"):
        run_clip(source, None, config, tools=ToolAnswers(people=full.people, balls=full.balls))
    with pytest.raises(ValueError, match="no engine of the settings uses them"):
        run_clip(source, FakeBackend(scene), PipelineConfig(), tools=full)
    with pytest.raises(ValueError, match="every frame"):
        run_clip(
            source,
            None,
            config,
            tools=ToolAnswers(people=(full.people or [])[1:], balls=full.balls, cameras=full.cameras),
        )
    with pytest.raises(ValueError, match="outside the clip"):
        run_clip(
            source, None, config, tools=ToolAnswers(people=full.people, balls={N_FRAMES: ()}, cameras=full.cameras)
        )
    with pytest.raises(ValueError, match="needs a model backend"):
        run_clip(
            source,
            None,
            PipelineConfig(engines=EngineSettings(people="detector", pitch_mapping="calibration")),
            tools=full,
        )


def test_every_frame_tracking_keeps_the_time_of_the_keyframe_limits() -> None:
    config = PipelineConfig()  # 4 missed keyframes of 0.48 s: 1.92 s
    assert every_frame_tracking(config, 25.0).max_missed_keyframes == 48
    assert every_frame_tracking(config, 50.0).max_missed_keyframes == 96
    assert every_frame_tracking(config, 25.0).max_stitch_gap_s == config.tracking.max_stitch_gap_s


def test_the_kit_clusters_are_the_two_teams_and_the_referee_is_an_outlier() -> None:
    # On this clip one k-means start ends in the referee against both teams: the referee got the
    # key A and the players of one team the key none. The fit with the smallest cost does not.
    scene = SyntheticScene(n_frames=50, seed=0)
    keyed = pipeline._kit_team_keys(SyntheticSource(scene), fake_detector_answers(scene)[0])
    keys: dict[str, Counter[str]] = {}
    for frame_people in keyed:
        truth = {(person.box.x1, person.box.y1): person for person in scene.frame_state(frame_people.frame).players}
        for person in frame_people.people:
            true_person = truth[(person.box.x1, person.box.y1)]
            keys.setdefault(true_person.role + true_person.team_key, Counter())[person.team_key] += 1

    assert set(keys["referee" + "none"]) == {"none"}
    team_keys = {keys["playerA"].most_common(1)[0][0], keys["playerB"].most_common(1)[0][0]}
    assert team_keys == {"A", "B"}


def test_without_kit_clusters_no_track_becomes_a_referee_or_a_goalkeeper(
    scene: SyntheticScene, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A clip without two kit colours: every person keeps the team key none. That is no outlier
    # kit, so the kit roles must not make every track a referee or a goalkeeper.
    monkeypatch.setattr(
        pipeline,
        "_kit_team_keys",
        lambda source, people: [
            frame_people.model_copy(
                update={
                    "people": tuple(person.model_copy(update={"team_key": "none"}) for person in frame_people.people)
                }
            )
            for frame_people in people
        ],
    )
    tools = tools_of(scene, DETECTOR_ONLY)
    result = run_clip(SyntheticSource(scene), None, PipelineConfig(engines=DETECTOR_ONLY), tools=tools)

    assert result.tracks
    assert {track.role for track in result.tracks} == {"player"}


def test_clip_of_one_kit_colour_gives_no_team_key_and_warns(
    scene: SyntheticScene, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Every torso shows the same blue kit with a little noise, as crops of one kit do.
    rng = np.random.default_rng(0)
    blue = np.zeros(kitcolour.FEATURE_SIZE)
    blue[7] = 1.0

    def one_kit(image: np.ndarray, box: Box) -> np.ndarray:
        row = np.abs(blue + rng.normal(0.0, 0.03, size=blue.shape))
        return row / np.linalg.norm(row)

    monkeypatch.setattr(pipeline, "torso_feature", one_kit)
    people = fake_detector_answers(scene)[0]

    keyed = pipeline._kit_team_keys(SyntheticSource(scene), people)

    assert {person.team_key for frame_people in keyed for person in frame_people.people} == {"none"}
    assert "one kit colour, not two" in caplog.text


def test_clip_without_two_kit_colours_gives_no_team_key_and_warns(
    scene: SyntheticScene, caplog: pytest.LogCaptureFixture
) -> None:
    people = [frame_people.model_copy(update={"people": ()}) for frame_people in fake_detector_answers(scene)[0]]
    assert pipeline._kit_team_keys(SyntheticSource(scene), people) == people
    assert "not two different kit colours" in caplog.text


# --- Command line -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def clip_on_disk(tmp_path_factory: pytest.TempPathFactory, scene: SyntheticScene) -> Path:
    """The synthetic clip in the layout of the dataset. The result is the data root."""
    root = tmp_path_factory.mktemp("data")
    clip_dir = root / "SoccerNetGS-valid" / CLIP_ID
    (clip_dir / "img1").mkdir(parents=True)
    for frame in range(N_FRAMES):
        assert cv2.imwrite(str(clip_dir / "img1" / f"{frame + 1:06d}.png"), scene.image(frame))
    (clip_dir / LABEL_NAME).write_text(json.dumps(scene.gsr_labels()), encoding="utf-8")
    return root


def clip_run(path: Path, job: dict[str, Any], payloads: list[dict[str, Any]]) -> Path:
    """Write the clip run of a job with the part writer of the script; one payload per frame.

    The job is the file ``job.json`` next to ``path``, as ``clip_job`` writes it.
    """
    frames = [
        {"clip": job["clip"], "frame": frame, "seconds": 0.02, "payload": payload}
        for frame, payload in enumerate(payloads)
    ]
    read = run_tool.read_job(path.with_name("job.json"), job["tool"]["name"])
    run_tool.write_clip_run(path, read, "c" * 64, {"device": "cpu"}, frames)
    return path


def detector_payloads(scene: SyntheticScene) -> list[dict[str, Any]]:
    people, balls = fake_detector_answers(scene)
    payloads = []
    for frame_people in people:
        detections = [
            {"box_xyxy": [p.box.x1, p.box.y1, p.box.x2, p.box.y2], "score": 0.9, "label": "person"}
            for p in frame_people.people
        ]
        detections += [
            {"box_xyxy": [b.box.x1, b.box.y1, b.box.x2, b.box.y2], "score": b.score, "label": "sports ball"}
            for b in balls.get(frame_people.frame, ())
        ]
        payloads.append({"detections": detections})
    return payloads


def calibration_payloads(scene: SyntheticScene) -> list[dict[str, Any]]:
    return [
        {"homography_pitch_to_image": np.linalg.inv(scene.homography(frame)).tolist(), "pitch_origin": "centre"}
        for frame in range(scene.info.n_frames)
    ]


def clip_job(tmp_path: Path, root: Path, tool: str, *more: str) -> dict[str, Any]:
    """Run ``clip-job`` into ``tmp_path/<tool>/job.json`` and return the job."""
    out = tmp_path / tool / "job.json"
    clip = root / "SoccerNetGS-valid" / CLIP_ID
    assert (
        cli.main(["clip-job", "--tool", tool, "--clip", str(clip), "--root", str(root), "--out", str(out), *more]) == 0
    )
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.fixture
def tool_runs(
    tmp_path: Path, clip_on_disk: Path, scene: SyntheticScene, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Clip runs of both tools for the synthetic clip, the detector in several parts of at most 20,000 bytes."""
    config = str(ENGINE_FILES / "detector_calibration.yaml")
    detector_job = clip_job(tmp_path, clip_on_disk, "rfdetr", "--config", config)
    calibration_job = clip_job(tmp_path, clip_on_disk, "pnlcalib", "--config", config)
    with monkeypatch.context() as patch:
        patch.setattr(run_tool, "CLIP_RUN_PART_MAX_BYTES", 20_000)
        detections = clip_run(tmp_path / "rfdetr" / "detections.json", detector_job, detector_payloads(scene))
    calibration = clip_run(tmp_path / "pnlcalib" / "calibration.json", calibration_job, calibration_payloads(scene))
    return detections, calibration


def test_clip_job_names_every_frame_below_the_data_folder(tmp_path: Path, clip_on_disk: Path) -> None:
    job = clip_job(tmp_path, clip_on_disk, "rfdetr")
    assert (job["format"], job["clip"], len(job["images"])) == ("pitchvlm-clip-job-1", CLIP_ID, N_FRAMES)
    assert job["images"][0] == f"SoccerNetGS-valid/{CLIP_ID}/img1/000001.png"
    assert job["tool"] == PipelineConfig().detector.model_dump(mode="json")
    assert clip_job(tmp_path, clip_on_disk, "pnlcalib")["tool"] == PipelineConfig().calibration.model_dump(mode="json")


def test_clip_job_refuses_a_video_and_a_clip_outside_the_data_folder(
    tmp_path: Path, clip_on_disk: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    arguments = ["clip-job", "--tool", "rfdetr", "--root", str(clip_on_disk), "--out", str(tmp_path / "job.json")]
    assert cli.main([*arguments, "--clip", str(video)]) == 2
    assert "write the frames of a video as images" in capsys.readouterr().err
    other_root = tmp_path / "elsewhere"
    other_root.mkdir()
    clip = clip_on_disk / "SoccerNetGS-valid" / CLIP_ID
    assert (
        cli.main(
            [
                "clip-job",
                "--tool",
                "rfdetr",
                "--root",
                str(other_root),
                "--clip",
                str(clip),
                "--out",
                str(tmp_path / "j.json"),
            ]
        )
        == 2
    )
    assert "does not lie below the data folder" in capsys.readouterr().err


def run_arguments(root: Path, out: Path, *more: str) -> list[str]:
    clip = root / "SoccerNetGS-valid" / CLIP_ID
    return ["run", "--clip", str(clip), "--labels", str(clip / LABEL_NAME), "--out", str(out), "--allow-dirty", *more]


def test_run_reads_the_clip_runs_of_both_tools_and_makes_no_model_call(
    tmp_path: Path,
    clip_on_disk: Path,
    tool_runs: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detections, calibration = tool_runs
    # No key of any provider is needed, and no model backend is built.
    for variable in cli.KEY_VARIABLES:
        monkeypatch.delenv(variable, raising=False)

    def no_backend(max_usd_per_10_min: float) -> None:
        raise AssertionError("a run without a model step must not build a model backend")

    monkeypatch.setattr(cli, "model_backend", no_backend)
    assert len(list(detections.parent.glob("detections*.json"))) > 1, "the detector run must have several parts"
    out = tmp_path / "out"
    settings = str(ENGINE_FILES / "detector_calibration.yaml")
    code = cli.main(
        run_arguments(
            clip_on_disk, out, "--config", settings, "--detections", str(detections), "--calibration", str(calibration)
        )
    )
    printed = capsys.readouterr().out
    assert code == 0
    assert "this run makes no model call and costs nothing" in printed
    assert "no player of this clip has a number" in printed
    document = json.loads((out / CLIP_ID / "result.json").read_text(encoding="utf-8"))
    n_ball = sum(frame["ball"] is not None for frame in document["result"]["frames"])
    assert f"ball: {n_ball} of {N_FRAMES} frames" in printed
    assert document["provenance"]["cost"] is None
    assert document["prediction_file"]["problems"] == []
    assert document["prediction_file"]["scored_frames_without_prediction"] == 0
    runs = document["provenance"]["tool_runs"]
    assert runs["detections"]["parts"] > 1 and runs["calibration"]["parts"] == 1
    assert runs["detections"]["runner"] == {"device": "cpu"}
    assert runs["detections"]["seconds_per_frame"]["n"] == N_FRAMES
    timing = document["timing"]
    assert {"detector", "calibration", "motion", "kit_colour", "tracks", "ball", "total"} <= set(timing)
    # The tool time is what the run files report: 0.02 s per frame.
    assert timing["detector"]["ms_per_frame"] == pytest.approx(20.0)
    assert document["result"]["engines"]["people"] == "detector:rfdetr:RFDETRMedium"
    assert any(frame["ball"] is not None for frame in document["result"]["frames"])
    assert (out / CLIP_ID / f"{CLIP_ID}.mp4").is_file()


def test_run_needs_the_clip_run_of_each_tool_engine_and_no_other(
    tmp_path: Path, clip_on_disk: Path, tool_runs: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    detections, calibration = tool_runs
    settings = ["--config", str(ENGINE_FILES / "detector_calibration.yaml")]
    out = tmp_path / "out"
    assert cli.main(run_arguments(clip_on_disk, out, *settings, "--calibration", str(calibration))) == 2
    assert "give the clip run file of its tool with --detections" in capsys.readouterr().err
    assert cli.main(run_arguments(clip_on_disk, out, "--detections", str(detections))) == 2
    assert "--detections is given, but the settings do not name the people engine detector" in capsys.readouterr().err
    # The run of one tool given for the other.
    assert (
        cli.main(
            run_arguments(
                clip_on_disk, out, *settings, "--detections", str(calibration), "--calibration", str(calibration)
            )
        )
        == 2
    )
    assert "other tool settings than the settings file" in capsys.readouterr().err


def test_run_refuses_a_clip_run_with_other_settings_or_of_another_clip(
    tmp_path: Path, clip_on_disk: Path, tool_runs: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    detections, calibration = tool_runs
    other = tmp_path / "other.yaml"
    other.write_text(
        (ENGINE_FILES / "detector_calibration.yaml").read_text(encoding="utf-8") + "detector:\n  model: RFDETRLarge\n",
        encoding="utf-8",
    )
    arguments = ["--detections", str(detections), "--calibration", str(calibration)]
    assert cli.main(run_arguments(clip_on_disk, tmp_path / "out", "--config", str(other), *arguments)) == 2
    assert "Write the clip job again and run the tool again" in capsys.readouterr().err
    first = json.loads(calibration.read_text(encoding="utf-8"))
    calibration.write_text(json.dumps({**first, "clip": "SNGS-001"}), encoding="utf-8")
    settings = ["--config", str(ENGINE_FILES / "detector_calibration.yaml")]
    assert cli.main(run_arguments(clip_on_disk, tmp_path / "out", *settings, *arguments)) == 2
    assert f"is a run of the clip SNGS-001 with {N_FRAMES} frames, not of the clip {CLIP_ID}" in capsys.readouterr().err


def test_run_names_the_frame_of_a_payload_that_cannot_be_used(
    tmp_path: Path, clip_on_disk: Path, tool_runs: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    detections, calibration = tool_runs
    first = json.loads(detections.read_text(encoding="utf-8"))
    first["frames"][1][2] = {"labels": ["person"], "rows": [[1.0, 2.0, 3.0, 4.0, 0.9, 5]]}
    detections.write_text(json.dumps(first), encoding="utf-8")
    settings = ["--config", str(ENGINE_FILES / "detector_calibration.yaml")]
    code = cli.main(
        run_arguments(
            clip_on_disk,
            tmp_path / "out",
            *settings,
            "--detections",
            str(detections),
            "--calibration",
            str(calibration),
        )
    )
    assert code == 2
    assert "frame 1: rows[0]: the label index must be a place in 'labels'" in capsys.readouterr().err


def test_model_plan_asks_only_the_model_steps_of_the_engines() -> None:
    def steps(engines: EngineSettings) -> list[str]:
        return [step for step, _, _ in cli.model_plan(PipelineConfig(engines=engines), 10)]

    assert steps(EngineSettings()) == ["kit_legend", "people", "landmarks", "number"]
    assert steps(DETECTOR_ONLY) == []
    assert steps(EngineSettings(people="detector", pitch_mapping="calibration", number="model")) == [
        "kit_legend",
        "number",
    ]
    assert steps(EngineSettings(people="detector", pitch_mapping="model", number="none")) == ["landmarks"]
    assert steps(EngineSettings(people="model", pitch_mapping="calibration", number="none")) == ["kit_legend", "people"]


def test_engine_files_load_as_settings() -> None:
    files = sorted(ENGINE_FILES.glob("*.yaml"))
    assert [path.name for path in files] == ["detector_calibration.yaml", "detector_calibration_numbers.yaml"]
    assert cli.read_settings(files[0]).engines == DETECTOR_ONLY
    assert cli.read_settings(files[1]).engines == EngineSettings(
        people="detector", pitch_mapping="calibration", number="model"
    )


def test_demo_runs_through_the_detector_engine(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["demo", "--engine", "detector", "--frames", "50", "--out", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["engine"] == "detector"
    assert summary["checker"]["valid"]
    assert summary["frames_with_ball"] > 0
    assert summary["identity"]["coverage"] > 0.95
    assert "total" in summary["timing"]
    printed = capsys.readouterr().out
    assert "measured time per stage" in printed
    assert (tmp_path / "demo.mp4").is_file()


# --- Records, settings and drawing of the engines ---------------------------------------


def test_ball_state_has_both_pitch_values_or_neither() -> None:
    box = Box(x1=10.0, y1=10.0, x2=16.0, y2=16.0)
    assert BallState(box=box).pitch_x is None
    assert BallState(box=box, pitch_x=1.0, pitch_y=-2.0, observed=True).pitch_y == -2.0
    with pytest.raises(ValueError):
        BallState(box=box, pitch_x=1.0)
    with pytest.raises(ValueError):
        BallState(box=box, pitch_x=float("nan"), pitch_y=0.0)


def test_settings_file_names_a_part_of_the_detector_and_keeps_the_rest(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text("engines: {people: detector}\ndetector: {model: RFDETRLarge}\n", encoding="utf-8")
    config = load_config(path)
    assert config.engines == EngineSettings(people="detector")
    assert config.detector == PipelineConfig().detector.model_copy(update={"model": "RFDETRLarge"})
    for wrong in ("engines: {people: yolo}\n", "detector: {model: RFDETRXLarge}\n", "calibration: {pnl_refine: 1}\n"):
        path.write_text(wrong, encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(path)


def test_ball_is_drawn_on_the_frame_and_on_the_radar() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    ball = BallState(box=Box(x1=96.0, y1=46.0, x2=104.0, y2=54.0), pitch_x=0.0, pitch_y=0.0, observed=True)
    drawn = draw_ball(image, ball)
    assert image.max() == 0, "the input image is not changed"
    assert drawn[50, 92].tolist() == [255, 255, 255]  # on the ring, 8 pixels left of the centre
    radar = RadarRenderer(25.0)
    plain = radar.draw(FrameState(frame=0))
    with_ball = RadarRenderer(25.0).draw(FrameState(frame=0, ball=ball))
    assert np.count_nonzero(np.any(plain != with_ball, axis=2)) > 10
    # A ball without a pitch position is not on the radar.
    assert np.array_equal(RadarRenderer(25.0).draw(FrameState(frame=0, ball=BallState(box=ball.box))), plain)
