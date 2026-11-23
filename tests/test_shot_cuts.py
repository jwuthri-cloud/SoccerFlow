"""Tests of the pipeline on a synthetic clip with a hard cut, and on one with a dissolve.

The clip shows the synthetic match from scene frame 0 to 53, then jumps 100 frames (4 seconds)
ahead: clip frame 54 shows scene frame 154. So the camera and the people are somewhere else after
the cut. The fake model answers with the exact truth. A second version of the clip carries a
static score graphic in its top left corner, as a broadcast does. The corners of that graphic do
not move across the cut, so they can give a camera motion across it.

The dissolve clip blends the same two moments over 10 frames instead (``BlendSource``). The
answers of the model and of the stand-in tools are those of the hard cut in the middle of the
dissolve; the pipeline must drop them on the frames of the dissolve.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from pitchvlm import cli, motion, pipeline
from pitchvlm.config import EngineSettings, PipelineConfig, TrackingConfig
from pitchvlm.fallbacks.fake import fake_calibration_answers, fake_detector_answers
from pitchvlm.frames import keyframe_indices
from pitchvlm.geometry import apply_homography, matrix_from_record
from pitchvlm.pipeline import ToolAnswers, run_clip
from pitchvlm.schemas import Box, ClipInfo, ClipResult, Track
from pitchvlm.synth import RetimedScene, ScoreGraphicSource, SyntheticSource
from pitchvlm.tracking import split_team_swaps
from pitchvlm.vlm.fake import FakeBackend

#: The clip frame that starts the second shot.
CUT = 54
#: Scene frames of the clip: two shots of 54 frames each.
SCENE_FRAMES = [*range(0, CUT), *range(CUT + 100, 2 * CUT + 100)]
#: A box of the answer shows a true person when the IoU of the two boxes is at least this.
SAME_BOX_MIN_IOU = 0.9


def _run(scene: RetimedScene, *, graphic: bool) -> ClipResult:
    source = ScoreGraphicSource(scene) if graphic else SyntheticSource(scene)
    return run_clip(source, FakeBackend(scene), PipelineConfig())


@pytest.fixture(scope="module")
def scene() -> RetimedScene:
    return RetimedScene(SCENE_FRAMES, seed=0)


@pytest.fixture(scope="module")
def plain(scene: RetimedScene) -> ClipResult:
    return _run(scene, graphic=False)


@pytest.fixture(scope="module")
def with_graphic(scene: RetimedScene) -> ClipResult:
    return _run(scene, graphic=True)


def _true_person(scene: RetimedScene, frame: int, box: Box) -> int:
    """Return the true track id of the person that a box of the answer shows."""
    players = scene.frame_state(frame).players
    best = max(players, key=lambda player: box.iou(player.box))
    assert box.iou(best.box) >= SAME_BOX_MIN_IOU
    return best.track_id


def _tracks_across_the_cut(result: ClipResult) -> list[Track]:
    """Return the tracks that have observations before and after the cut."""
    return [track for track in result.tracks if track.first_frame < CUT <= track.last_frame]


def _people_across_the_cut(scene: RetimedScene, result: ClipResult) -> list[tuple[int, int]]:
    """Return (true person before the cut, true person after it) for every track that crosses the cut."""
    pairs: list[tuple[int, int]] = []
    for track in _tracks_across_the_cut(result):
        before = [obs for obs in track.observations if obs.frame < CUT][-1]
        after = [obs for obs in track.observations if obs.frame >= CUT][0]
        pairs.append((_true_person(scene, before.frame, before.box), _true_person(scene, after.frame, after.box)))
    return pairs


def _mapping_errors_m(scene: RetimedScene, result: ClipResult, frames: Sequence[int]) -> list[float]:
    """Return the largest pitch error of the true foot points under the mapping of each frame that has one."""
    errors: list[float] = []
    for frame in frames:
        record = result.homographies[frame]
        if not record.valid:
            continue
        # A box cut at the bottom border has no true foot point in the picture.
        truth = [player for player in scene.frame_state(frame).players if player.box.y2 < scene.info.height - 1]
        feet = np.array([player.box.foot for player in truth])
        positions = np.array([[player.pitch_x, player.pitch_y] for player in truth])
        mapped = apply_homography(matrix_from_record(record), feet)
        errors.append(float(np.linalg.norm(mapped - positions, axis=1).max()))
    return errors


@pytest.mark.parametrize("graphic", [False, True])
def test_only_a_shown_shirt_number_joins_tracks_across_a_cut(
    plain: ClipResult, with_graphic: ClipResult, graphic: bool
) -> None:
    # Positions and image motion do not link people across a cut. One shown shirt number of one
    # team names one person in every shot, so ``finalize_tracks`` may join such tracks.
    result = with_graphic if graphic else plain
    crossing = _tracks_across_the_cut(result)
    assert [track.track_id for track in crossing if track.number_display is None] == []


@pytest.mark.parametrize("graphic", [False, True])
def test_no_track_joins_two_people_across_a_cut(
    scene: RetimedScene, plain: ClipResult, with_graphic: ClipResult, graphic: bool
) -> None:
    result = with_graphic if graphic else plain
    pairs = _people_across_the_cut(scene, result)
    assert [pair for pair in pairs if pair[0] != pair[1]] == []


@pytest.mark.parametrize("graphic", [False, True])
def test_no_pitch_mapping_is_carried_across_a_cut(
    scene: RetimedScene, plain: ClipResult, with_graphic: ClipResult, graphic: bool
) -> None:
    result = with_graphic if graphic else plain
    errors = _mapping_errors_m(scene, result, range(CUT, CUT + 12))
    assert errors and max(errors) < 1.0


@pytest.mark.parametrize("graphic", [False, True])
def test_no_state_is_filled_across_a_cut(
    scene: RetimedScene, plain: ClipResult, with_graphic: ClipResult, graphic: bool
) -> None:
    result = with_graphic if graphic else plain
    last_keyframe_before = max(frame for frame in range(0, CUT, 12))
    filled = [
        (state.frame, player.track_id)
        for state in result.frames[last_keyframe_before + 1 : CUT]
        for player in state.players
        if not player.observed
    ]
    assert filled == []


@pytest.mark.parametrize("graphic", [False, True])
def test_the_result_names_the_cut_and_scores_the_landmark_fits(
    plain: ClipResult, with_graphic: ClipResult, graphic: bool
) -> None:
    result = with_graphic if graphic else plain
    assert result.cuts == (CUT,)
    keyframes = [frame for frame in range(0, result.info.n_frames, 12)] + [result.info.n_frames - 1]
    assert [check.frame for check in result.paint] == keyframes
    assert all(check.score is not None and check.score > 0.9 for check in result.paint)


# --- The track repairs (arms of gate G4) keep the rule of the cuts ---------------------------------

#: Both track repairs on: the split of team swaps in each shot and the team slots over the clip.
BOTH_REPAIRS = PipelineConfig(tracking=TrackingConfig(split_team_swaps=True, team_slots=True))


@pytest.fixture(scope="module")
def repaired(scene: RetimedScene) -> ClipResult:
    return run_clip(SyntheticSource(scene), FakeBackend(scene), BOTH_REPAIRS)


def test_the_split_of_team_swaps_runs_in_each_shot_on_its_own(
    scene: RetimedScene, monkeypatch: pytest.MonkeyPatch
) -> None:
    shots_of_calls: list[set[bool]] = []

    def watched(tracks: Sequence[Track]) -> list[Track]:
        shots_of_calls.append({obs.frame >= CUT for track in tracks for obs in track.observations})
        return split_team_swaps(tracks)

    monkeypatch.setattr(pipeline, "split_team_swaps", watched)
    run_clip(SyntheticSource(scene), FakeBackend(scene), BOTH_REPAIRS)

    # One call per shot, and the fragments of a call lie in one shot.
    assert shots_of_calls == [{False}, {True}]


def test_with_both_repairs_only_a_shown_shirt_number_joins_tracks_across_a_cut(
    scene: RetimedScene, plain: ClipResult, repaired: ClipResult
) -> None:
    assert repaired.cuts == (CUT,)
    crossing = _tracks_across_the_cut(repaired)
    assert [track.track_id for track in crossing if track.number_display is None] == []
    assert [pair for pair in _people_across_the_cut(scene, repaired) if pair[0] != pair[1]] == []
    # The team slots link no track across the cut: the tracks across it are those of the
    # shirt numbers, as without the repairs.
    assert len(crossing) == len(_tracks_across_the_cut(plain))



# --- A dissolve is a transition: its frames carry nothing --------------------------------------

#: The first frame of the dissolve, and its length in frames.
DISSOLVE_START = 40
DISSOLVE_FRAMES = 10
#: Frames of the dissolve clip: 40 frames of each shot around the dissolve.
DISSOLVE_CLIP_FRAMES = 2 * DISSOLVE_START + DISSOLVE_FRAMES
#: The truth of the dissolve clip switches shots in the middle of the dissolve.
DISSOLVE_MIDDLE = DISSOLVE_START + DISSOLVE_FRAMES // 2


class BlendSource:
    """The first shot (scene frames 0 on) blended into the second (scene frames 100 on) over ``DISSOLVE_FRAMES``."""

    def __init__(self) -> None:
        frames = range(DISSOLVE_CLIP_FRAMES)
        self._old = SyntheticSource(RetimedScene(list(frames), seed=0))
        self._new = SyntheticSource(RetimedScene([frame + 100 for frame in frames], seed=0))
        self.info = ClipInfo(clip_id="dissolve", n_frames=DISSOLVE_CLIP_FRAMES, fps=25.0, width=1280, height=720)

    def read(self, frame: int) -> np.ndarray:
        if frame < DISSOLVE_START:
            return self._old.read(frame)
        if frame >= DISSOLVE_START + DISSOLVE_FRAMES:
            return self._new.read(frame)
        weight = (frame - DISSOLVE_START + 1) / (DISSOLVE_FRAMES + 1)
        blend = (1.0 - weight) * self._old.read(frame).astype(np.float32) + weight * self._new.read(frame)
        return np.clip(blend, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def dissolve_truth() -> RetimedScene:
    frames = range(DISSOLVE_CLIP_FRAMES)
    return RetimedScene([frame if frame < DISSOLVE_MIDDLE else frame + 100 for frame in frames], seed=0)


@pytest.fixture(scope="module")
def dissolve_results(dissolve_truth: RetimedScene) -> dict[str, ClipResult]:
    people, balls = fake_detector_answers(dissolve_truth)
    detector = PipelineConfig(engines=EngineSettings(people="detector", pitch_mapping="calibration", number="none"))
    tools = ToolAnswers(people=people, balls=balls, cameras=fake_calibration_answers(dissolve_truth))
    return {
        "model": run_clip(BlendSource(), FakeBackend(dissolve_truth), PipelineConfig()),
        "detector": run_clip(BlendSource(), None, detector, tools=tools),
    }


def _blend_frames(result: ClipResult) -> list[int]:
    """Return the frames of the result that form a shot of one frame."""
    starts = (0, *result.cuts)
    ends = (*result.cuts, result.info.n_frames)
    return [start for start, end in zip(starts, ends, strict=True) if end - start == 1]


@pytest.mark.parametrize("engine", ["model", "detector"])
def test_every_frame_of_a_dissolve_starts_a_shot_of_its_own(
    dissolve_results: dict[str, ClipResult], engine: str
) -> None:
    result = dissolve_results[engine]
    blends = _blend_frames(result)
    assert blends == list(range(blends[0], blends[-1] + 1))
    assert result.cuts == (*blends, blends[-1] + 1)
    # The window test leaves at most the faint ends of the dissolve in the shots (synthetic seed 0).
    assert DISSOLVE_START <= blends[0] <= DISSOLVE_START + 2
    assert DISSOLVE_START + DISSOLVE_FRAMES - 3 <= blends[-1] <= DISSOLVE_START + DISSOLVE_FRAMES - 1


@pytest.mark.parametrize("engine", ["model", "detector"])
def test_the_frames_of_a_dissolve_get_no_state_no_ball_and_no_mapping(
    dissolve_results: dict[str, ClipResult], engine: str
) -> None:
    result = dissolve_results[engine]
    for frame in _blend_frames(result):
        assert result.frames[frame].players == ()
        assert result.frames[frame].ball is None
        assert not result.homographies[frame].valid
        assert all(obs.frame != frame for track in result.tracks for obs in track.observations)


@pytest.mark.parametrize("engine", ["model", "detector"])
def test_no_track_crosses_a_dissolve_without_a_shown_number(
    dissolve_results: dict[str, ClipResult], engine: str
) -> None:
    result = dissolve_results[engine]
    blends = _blend_frames(result)
    crossing = [track for track in result.tracks if track.first_frame < blends[0] and track.last_frame > blends[-1]]
    assert [track.track_id for track in crossing if track.number_display is None] == []


@pytest.mark.parametrize("engine", ["model", "detector"])
def test_the_shots_around_a_dissolve_keep_their_own_mappings(
    dissolve_truth: RetimedScene, dissolve_results: dict[str, ClipResult], engine: str
) -> None:
    result = dissolve_results[engine]
    blends = _blend_frames(result)
    before = range(blends[0] - 12, DISSOLVE_START)  # frames of the first shot alone
    after = range(DISSOLVE_START + DISSOLVE_FRAMES, blends[-1] + 13)  # frames of the second shot alone
    errors = _mapping_errors_m(dissolve_truth, result, [*before, *after])
    assert len(errors) == len(before) + len(after)
    assert max(errors) < 1.0


def test_the_cli_names_the_transition_frames_and_the_keyframes_whose_answers_are_dropped(
    dissolve_results: dict[str, ClipResult], capsys: pytest.CaptureFixture[str]
) -> None:
    """A paid model answer of a keyframe on a blend frame is dropped: the run says which (second read of 2026-11-22)."""
    result = dissolve_results["model"]
    config = PipelineConfig()
    blends = _blend_frames(result)
    keyframes = keyframe_indices(result.info.n_frames, result.info.fps, config.keyframe_interval_s)
    dropped = [frame for frame in keyframes if frame in blends]
    assert dropped, "a keyframe lies on a blend frame"
    cli.print_shots_and_paint(result, config)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"cuts (frames that start a new shot): {blends[-1] + 1}"
    named = "transition frames (a dissolve, or a lone frame between two cuts), which carry nothing"
    assert lines[1].startswith(f"{named}: {', '.join(map(str, blends))};")
    assert lines[1].endswith(
        f"{len(dropped)} of {len(keyframes)} keyframes among them, their answers dropped: {', '.join(map(str, dropped))}"
    )


@pytest.mark.parametrize("failed", [[DISSOLVE_START + 4], [DISSOLVE_START + 2, DISSOLVE_START + 6]])
def test_one_pair_of_a_dissolve_without_a_measured_motion_does_not_hide_it(
    failed: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lone pair without a measured motion takes the last motion, so the windows over it are judged.

    Without it every window that holds the dissolve held a missing step, and the dissolve was not
    found (the reader of the fixes of the second read of 2026-11-22). The flow fails for the
    pairs into the frames ``failed`` only.
    """
    real = motion.estimate_motion
    pairs = {"done": 0}

    def flow(prev: np.ndarray, nxt: np.ndarray, **kwargs: object) -> np.ndarray | None:
        pairs["done"] += 1
        return None if pairs["done"] in failed else real(prev, nxt, **kwargs)

    monkeypatch.setattr(motion, "estimate_motion", flow)
    tests = motion.measure_shots(BlendSource())
    assert tests.hard_cuts == ()
    assert all(tests.test_steps[frame - 1] is not None for frame in failed)
    assert len(tests.transitions) == 1
    start, stop = tests.transitions[0]
    assert DISSOLVE_START <= start <= DISSOLVE_START + 2
    assert DISSOLVE_START + DISSOLVE_FRAMES - 2 <= stop <= DISSOLVE_START + DISSOLVE_FRAMES
