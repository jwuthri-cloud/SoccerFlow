"""Tests of the pipeline on a synthetic clip with a hard cut.

The clip shows the synthetic match from scene frame 0 to 53, then jumps 100 frames (4 seconds)
ahead: clip frame 54 shows scene frame 154. So the camera and the people are somewhere else after
the cut. The fake model answers with the exact truth. A second version of the clip carries a
static score graphic in its top left corner, as a broadcast does. The corners of that graphic do
not move across the cut, so they can give a camera motion across it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from pitchvlm import pipeline
from pitchvlm.config import PipelineConfig, TrackingConfig
from pitchvlm.geometry import apply_homography, matrix_from_record
from pitchvlm.pipeline import run_clip
from pitchvlm.schemas import Box, ClipResult, Track
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

