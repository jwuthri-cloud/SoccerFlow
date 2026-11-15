"""Tests of the pipeline: the synthetic clip goes from its frames to the dense result with the fake model.

The clip is ``SyntheticScene(n_frames=75, seed=1)``: 3 seconds at 25 frames per second, 1280 x 720.
The camera starts on the left side and pans to the halfway line, so it never shows the right goal
area. Team A defends the left goal. The fake model answers with the exact truth, rounded to model
coordinates.

One run measures the camera motion from the frames, as a real run does. The other runs get the exact
camera motion of the scene: they are faster and they give the same result every time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import signal
import threading
import time
from collections.abc import Callable, Collection, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import yaml

from pitchvlm import pipeline
from pitchvlm.config import DEFAULT_MODEL, PipelineConfig, StepSettings, TrackingConfig, load_config
from pitchvlm.evalx.metrics import identity_scores, position_errors_m
from pitchvlm.frames import keyframe_indices
from pitchvlm.geometry import apply_homography, matrix_from_record
from pitchvlm.motion import MotionField
from pitchvlm.pipeline import run_clip
from pitchvlm.pitch import inside_pitch
from pitchvlm.roster import Roster
from pitchvlm.schemas import (
    TEAM_ROLES,
    Box,
    ClipInfo,
    ClipResult,
    FrameState,
    HomographyRecord,
    PersonDet,
    Track,
    TrackObs,
)
from pitchvlm.synth import SyntheticScene, SyntheticSource
from pitchvlm.vlm.cache import CachedBackend, CassetteStore
from pitchvlm.vlm.coords import box_to_model
from pitchvlm.vlm.fake import FakeBackend
from pitchvlm.vlm.steps import ParseError, encode_jpeg
from pitchvlm.vlm.types import Backend, BackendError, CacheMiss, ImagePart, VLMRequest, VLMResponse

N_FRAMES = 75
SEED = 1
#: Keyframes of the clip: one frame every 0.48 s (12 frames), plus the last frame.
KEYFRAMES = [0, 12, 24, 36, 48, 60, 72, 74]
#: The scene has 23 people.
N_PEOPLE = 23
MODEL_STEPS = ("kit_legend", "people", "landmarks", "number")
CODE_STEPS = ("motion", "pitch_mapping", "tracks", "fill")
#: Team A of the scene defends the left goal. The referee has no side.
SIDE_OF_TEAM = {"A": "left", "B": "right", "none": None}
#: A roster with a name for every shirt number of the scene.
ROSTER = Roster(
    left={number: f"Left {number}" for number in range(1, 12)},
    right={number: f"Right {number}" for number in range(1, 12)},
)
#: Settings that differ from the defaults, to see that the requests follow the settings.
OTHER_SETTINGS = PipelineConfig(repeat=1, max_parallel=2, people=StepSettings(model="gemini-3.8-flash"))
#: A pitch point 11 m behind the far touchline. The camera sees it in every keyframe.
BEHIND_FAR_TOUCHLINE_XY = (-20.0, -45.0)
#: Pitch points 6 m and 5 m behind the far touchline, outside the pitch margin of 3 m. The camera sees them too.
WARM_UP_SPOTS = ((-20.0, -40.0), (-17.0, -40.0))
FIVE_METRES_OUT_SPOTS = ((-20.0, -39.0), (-17.0, -39.0))
#: Twelve pitch points near the left penalty area. The camera sees them in every keyframe.
ON_PITCH_SPOTS = tuple((-30.0 + 2.0 * index, -10.0 + 1.5 * index) for index in range(12))
DEFAULT_SETTINGS_FILE = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
#: A clip of 1280 x 720 pixels, for the test of the item of a number request.
ITEM_CLIP = ClipInfo(clip_id="clip-1", n_frames=100, fps=25.0, width=1280, height=720)

AnswerRewrite = Callable[[VLMRequest, str], str]


# --- Backends of the tests -----------------------------------------------------------------


class RecordingBackend:
    """A backend that keeps every request it gets and passes it on to another backend."""

    def __init__(self, inner: Backend) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self.requests: list[VLMRequest] = []

    def generate(self, request: VLMRequest) -> VLMResponse:
        with self._lock:
            self.requests.append(request)
        return self._inner.generate(request)


class FailingBackend(RecordingBackend):
    """A recording backend that raises ``BackendError`` for request number ``fail_at`` (the first request is 1)."""

    def __init__(self, inner: Backend, fail_at: int) -> None:
        super().__init__(inner)
        self._fail_at = fail_at

    def generate(self, request: VLMRequest) -> VLMResponse:
        with self._lock:
            self.requests.append(request)
            number = len(self.requests)
        if number == self._fail_at:
            raise BackendError(f"planned failure of request {number}")
        return self._inner.generate(request)


class RewritingBackend:
    """A backend that rewrites the answer text of another backend for the requests of one step."""

    def __init__(self, inner: Backend, step: str, rewrite: AnswerRewrite) -> None:
        self._inner = inner
        self._step = step
        self._rewrite = rewrite

    def generate(self, request: VLMRequest) -> VLMResponse:
        response = self._inner.generate(request)
        if request.meta.step != self._step:
            return response
        return response.model_copy(update={"text": self._rewrite(request, response.text)})


def broken_answer_of_frame(frame: int) -> AnswerRewrite:
    """Return a rewrite that replaces the answer of one frame by text that is not JSON."""

    def rewrite(request: VLMRequest, text: str) -> str:
        return "this is not JSON" if request.meta.frame == frame else text

    return rewrite


def mirrored_landmarks_of_frame(frame: int) -> AnswerRewrite:
    """Return a rewrite that swaps left and right in the landmark points of one frame."""

    def rewrite(request: VLMRequest, text: str) -> str:
        if request.meta.frame != frame:
            return text
        answer = json.loads(text)
        for landmark in answer["landmarks"]:
            y, x = landmark["point"]  # A model point is [y, x].
            landmark["point"] = [y, 1000 - x]
        return json.dumps(answer)

    return rewrite


def person_behind_the_far_touchline(scene: SyntheticScene) -> AnswerRewrite:
    """Return a rewrite that adds one person next to the pitch to every people answer.

    The person has the role "other" and stands still at ``BEHIND_FAR_TOUCHLINE_XY``.
    """
    width, height = scene.info.width, scene.info.height

    def rewrite(request: VLMRequest, text: str) -> str:
        assert request.meta.frame is not None
        pitch_to_image = np.linalg.inv(scene.homography(request.meta.frame))
        foot_x, foot_y = apply_homography(pitch_to_image, np.array([BEHIND_FAR_TOUCHLINE_XY]))[0].tolist()
        box = Box(x1=foot_x - 5.0, y1=foot_y - 25.0, x2=foot_x + 5.0, y2=foot_y)
        answer = json.loads(text)
        answer["people"].append(
            {"box_2d": box_to_model(box, width, height), "role": "other", "team": "none", "number": None}
        )
        return json.dumps(answer)

    return rewrite


def people_added(
    scene: SyntheticScene, spots: Sequence[tuple[float, float]], *, role: str = "player", keep: int | None = None
) -> AnswerRewrite:
    """Return a rewrite that adds one standing person per pitch position to every people answer.

    The people have the given role and the team key A. With ``keep``, only the first ``keep``
    people of the answer of the model stay in front of them.
    """
    width, height = scene.info.width, scene.info.height

    def rewrite(request: VLMRequest, text: str) -> str:
        assert request.meta.frame is not None
        pitch_to_image = np.linalg.inv(scene.homography(request.meta.frame))
        answer = json.loads(text)
        if keep is not None:
            answer["people"] = answer["people"][:keep]
        for spot in spots:
            foot_x, foot_y = apply_homography(pitch_to_image, np.array([spot]))[0].tolist()
            box = Box(x1=foot_x - 5.0, y1=foot_y - 25.0, x2=foot_x + 5.0, y2=foot_y)
            answer["people"].append(
                {"box_2d": box_to_model(box, width, height), "role": role, "team": "A", "number": None}
            )
        return json.dumps(answer)

    return rewrite


def everybody_twice(request: VLMRequest, text: str) -> str:
    """A rewrite of a people answer that lists every person two times, one after the other."""
    answer = json.loads(text)
    answer["people"] = [person for person in answer["people"] for _ in range(2)]
    return json.dumps(answer)


def few_landmarks(
    count: int, frames: Collection[int] | None = None, moves: Sequence[tuple[int, int]] = ()
) -> AnswerRewrite:
    """Return a rewrite that keeps ``count`` landmarks of a landmarks answer, spread over its list.

    ``frames`` names the keyframes whose answers are changed; None means all keyframes. ``moves``
    shifts the kept landmarks, in their order, by (y, x) model units. The landmarks of an answer
    are in the order of their ids, so landmarks that are spread over the list do not lie on one line.
    """

    def rewrite(request: VLMRequest, text: str) -> str:
        if frames is not None and request.meta.frame not in frames:
            return text
        answer = json.loads(text)
        marks = answer["landmarks"]
        kept = [marks[round(index * (len(marks) - 1) / (count - 1))] for index in range(count)]
        for mark, (move_y, move_x) in zip(kept, moves):
            y, x = mark["point"]  # A model point is [y, x].
            mark["point"] = [y + move_y, x + move_x]
        answer["landmarks"] = kept
        return json.dumps(answer)

    return rewrite


#: Shifts of four landmarks, in model units, that put a pitch mapping from these four landmarks some metres off.
WRONG_FIT_MOVES = ((0, 0), (0, 14), (-14, 0), (0, 0))


def landmarks_shifted(frames: Collection[int], move_x: int) -> AnswerRewrite:
    """Return a rewrite that moves every landmark of some keyframes by ``move_x`` model units to the right.

    All landmarks of such an answer still agree with one mapping, but that mapping is wrong.
    """

    def rewrite(request: VLMRequest, text: str) -> str:
        if request.meta.frame not in frames:
            return text
        answer = json.loads(text)
        for landmark in answer["landmarks"]:
            landmark["point"][1] += move_x  # A model point is [y, x].
        return json.dumps(answer)

    return rewrite


def person_missed(team: str, number: int, frames: Collection[int]) -> AnswerRewrite:
    """Return a rewrite that leaves one player out of the people answers of some keyframes."""

    def rewrite(request: VLMRequest, text: str) -> str:
        if request.meta.frame not in frames:
            return text
        answer = json.loads(text)
        answer["people"] = [
            person for person in answer["people"] if (person["team"], person["number"]) != (team, number)
        ]
        return json.dumps(answer)

    return rewrite


class Spy:
    """Keeps the arguments of every call of a function of the pipeline module, and passes the call on."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self._real = getattr(pipeline, name)
        monkeypatch.setattr(pipeline, name, self)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return self._real(*args, **kwargs)


class WrongSizeSource:
    """A frame source whose frames do not have the size that its clip information names."""

    def __init__(self, info: ClipInfo) -> None:
        self.info = info

    def read(self, frame: int) -> np.ndarray:
        return np.zeros((10, 10, 3), dtype=np.uint8)


# --- Helpers -------------------------------------------------------------------------------


def exact_motion(scene: SyntheticScene) -> MotionField:
    """Return the true camera motion: pixels of a frame go to the pitch and then to pixels of the next frame."""
    to_pitch = [scene.homography(frame) for frame in range(scene.info.n_frames)]
    steps = [np.linalg.inv(to_pitch[frame + 1]) @ to_pitch[frame] for frame in range(scene.info.n_frames - 1)]
    return MotionField(steps)


def near_half_grid() -> np.ndarray:
    """Return image points in the lower part of a 1280 x 720 picture, which shows the near half of the pitch."""
    grid_x, grid_y = np.meshgrid(np.arange(100.0, 1200.0, 100.0), np.arange(300.0, 700.0, 50.0))
    return np.column_stack([grid_x.ravel(), grid_y.ravel()])


def mapping_errors_m(record: HomographyRecord, scene: SyntheticScene) -> np.ndarray:
    """Return the distance in metres between the pitch mapping of a record and the true camera, per grid point."""
    grid = near_half_grid()
    mapped = apply_homography(matrix_from_record(record), grid)
    return np.linalg.norm(mapped - apply_homography(scene.homography(record.frame), grid), axis=1)


def true_person(scene: SyntheticScene, frame: int, box: Box) -> PersonDet:
    """Return the true person of a frame whose box overlaps ``box`` best."""
    return max(scene.people(frame).people, key=lambda person: box.iou(person.box))


def parse_exact_people(scene: SyntheticScene, frame: int) -> tuple[PersonDet, ...]:
    """Return the people of a frame as the pipeline reads them from the answer of the exact fake model.

    The answer holds the true boxes in model coordinates, so the boxes are rounded to model units.
    """
    request = pipeline.build_people_request(
        scene.image(frame), scene.kit_legend(), clip_id=scene.info.clip_id, frame=frame, settings=StepSettings()
    )
    answer = FakeBackend(scene).generate(request).text
    return pipeline.parse_people(answer, frame=frame, width=scene.info.width, height=scene.info.height).people


def images_of(request: VLMRequest) -> list[ImagePart]:
    """Return the image parts of a request, in their order."""
    return [part for part in request.parts if isinstance(part, ImagePart)]


def requests_of(requests: list[VLMRequest], step: str) -> list[VLMRequest]:
    """Return the requests of one step."""
    return [request for request in requests if request.meta.step == step]


def corners(box: Box) -> list[float]:
    """Return a box as the context of a number request holds it: [x1, y1, x2, y2]."""
    return [box.x1, box.y1, box.x2, box.y2]


def number_request_of(requests: list[VLMRequest], track: Track) -> VLMRequest:
    """Return the one number request that shows crops of a track.

    The context of a number request names the frame and the box of every crop. The item of the
    request is not used here: a track id is not a lasting name, so no test reads it from the item.
    """
    boxes = {observation.frame: corners(observation.box) for observation in track.observations}
    shown = [
        request
        for request in requests
        if all(
            boxes.get(frame) == box
            for frame, box in zip(request.context["frames"], request.context["boxes"], strict=True)
        )
    ]
    assert len(shown) == 1, f"{len(shown)} number requests show crops of track {track.track_id}"
    return shown[0]


def number_request_of_track_7(
    scene: SyntheticScene, crops: list[tuple[int, Box]], crop_upscale: float = 4.0, track_id: int = 7
) -> VLMRequest:
    """Build the number request of the pipeline for track 7 of team A from its (frame, box) crops.

    Every crop is the same black picture, whatever its frame and its box are. So only the
    item of the request can tell the cache which crops the request shows.
    """
    observations = tuple(TrackObs(frame=frame, box=box, role="player", team_key="A") for frame, box in crops)
    plan = pipeline._CropPlan(track_id=track_id, team_key="A", observations=observations)
    pictures = {(track_id, frame): np.zeros((24, 12, 3), dtype=np.uint8) for frame, _ in crops}
    config = PipelineConfig(crop_upscale=crop_upscale)
    return pipeline._number_request(plan, pictures, scene.kit_legend(), ITEM_CLIP, config)


def true_record(scene: SyntheticScene, frame: int) -> HomographyRecord:
    """Return the true pitch mapping of a frame of the scene as a well-supported landmark fit."""
    matrix = tuple(tuple(row) for row in scene.homography(frame).tolist())
    return HomographyRecord(frame=frame, matrix=matrix, source="landmarks", n_inliers=12, reproj_px=0.0)


def key_sources(result: ClipResult) -> list[str]:
    """Return the source of the pitch mapping of every keyframe of the clip."""
    return [result.homographies[frame].source for frame in KEYFRAMES]


def track_with_boxes(frames_and_heights: Sequence[tuple[int, float]]) -> Track:
    """Make a track of a player with one observation per (frame, box height). Every box is 20 pixels wide."""
    observations = tuple(
        TrackObs(frame=frame, box=Box(x1=100.0, y1=300.0 - height, x2=120.0, y2=300.0), role="player", team_key="A")
        for frame, height in frames_and_heights
    )
    return Track(track_id=0, observations=observations)


def decoded(image: ImagePart) -> np.ndarray:
    """Decode the JPEG bytes of an image part to a BGR image of shape (height, width, 3)."""
    return cv2.imdecode(np.frombuffer(image.data, dtype=np.uint8), cv2.IMREAD_COLOR)


def count_keys(settings: dict[str, Any]) -> int:
    """Count the keys of a nested mapping, at every level."""
    return sum(1 + (count_keys(value) if isinstance(value, dict) else 0) for value in settings.values())


# --- Fixtures ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scene() -> SyntheticScene:
    return SyntheticScene(n_frames=N_FRAMES, seed=SEED)


@pytest.fixture(scope="module")
def source(scene: SyntheticScene) -> SyntheticSource:
    return SyntheticSource(scene)


@pytest.fixture(scope="module")
def truth(scene: SyntheticScene) -> list[FrameState]:
    """The true state of every frame."""
    return [scene.frame_state(frame) for frame in range(scene.info.n_frames)]


@pytest.fixture(scope="module")
def true_motion(scene: SyntheticScene) -> MotionField:
    return exact_motion(scene)


@pytest.fixture(scope="module")
def result(scene: SyntheticScene, source: SyntheticSource) -> ClipResult:
    """The pipeline result with the default settings. The camera motion is measured from the frames."""
    return run_clip(source, FakeBackend(scene))


@pytest.fixture(scope="module")
def recorded(
    scene: SyntheticScene, source: SyntheticSource, true_motion: MotionField
) -> tuple[ClipResult, list[VLMRequest]]:
    """A run with other settings, a roster and the exact camera motion: the result and all model requests."""
    backend = RecordingBackend(FakeBackend(scene))
    clip_result = run_clip(source, backend, OTHER_SETTINGS, roster=ROSTER, motion=true_motion)
    return clip_result, backend.requests


# --- The synthetic clip from end to end ------------------------------------------------------


def test_keyframes_of_the_clip() -> None:
    """The clip has the keyframes that the other tests name."""
    assert keyframe_indices(N_FRAMES, 25.0, PipelineConfig().keyframe_interval_s) == KEYFRAMES


def test_every_frame_has_a_state(result: ClipResult, scene: SyntheticScene) -> None:
    assert result.info == scene.info
    assert [state.frame for state in result.frames] == list(range(N_FRAMES))
    for state in result.frames:
        assert len(state.players) >= 8
        track_ids = [player.track_id for player in state.players]
        assert len(track_ids) == len(set(track_ids))
        for player in state.players:
            assert player.box is not None
            # The model sees people at keyframes only. The other frames are filled in.
            assert not player.observed or state.frame in KEYFRAMES
    assert all(player.observed for player in result.frames[KEYFRAMES[0]].players)


def test_positions_are_close_to_the_truth(result: ClipResult, truth: list[FrameState]) -> None:
    """The median distance between a predicted and a true pitch position is below 0.5 m."""
    errors = position_errors_m(result.frames, truth)
    n_true_positions = sum(len(state.players) for state in truth)
    # The median must stand on most of the true positions, not on a few lucky ones.
    assert errors.size > 0.8 * n_true_positions
    assert float(np.median(errors)) < 0.5


def test_number_of_tracks_is_close_to_the_number_of_people(result: ClipResult) -> None:
    """The camera does not show every person, and a person can be split: 23 plus or minus 4 tracks."""
    assert abs(len(result.tracks) - N_PEOPLE) <= 4
    track_ids = [track.track_id for track in result.tracks]
    assert len(track_ids) == len(set(track_ids))
    used_ids = {player.track_id for state in result.frames for player in state.players}
    assert used_ids <= set(track_ids)


def test_team_a_defends_the_left_goal(result: ClipResult) -> None:
    team_a = [track for track in result.tracks if track.role == "player" and track.team_key == "A"]
    team_b = [track for track in result.tracks if track.role == "player" and track.team_key == "B"]
    assert len(team_a) >= 5
    assert len(team_b) >= 5
    assert all(track.team_side == "left" for track in team_a)
    assert all(track.team_side == "right" for track in team_b)
    for state in result.frames:
        for player in state.players:
            assert player.team_side == SIDE_OF_TEAM[player.team_key]


def test_engines_name_every_step(result: ClipResult) -> None:
    assert set(result.engines) == {*MODEL_STEPS, *CODE_STEPS}
    for step in MODEL_STEPS:
        assert result.engines[step] == f"model:{DEFAULT_MODEL}"
    for step in CODE_STEPS:
        assert result.engines[step] == "code"


def test_far_side_is_negative_y_and_left_goal_is_negative_x(result: ClipResult) -> None:
    """Sign conventions: a person high in the picture is on the far side, a person on its left is near the left goal."""
    players = result.frames[36].players
    highest = min(players, key=lambda player: player.box.y2)
    lowest = max(players, key=lambda player: player.box.y2)
    assert highest.pitch_y < 0.0 < lowest.pitch_y
    leftmost = min(players, key=lambda player: player.box.foot[0])
    rightmost = max(players, key=lambda player: player.box.foot[0])
    assert leftmost.pitch_x < 0.0 < rightmost.pitch_x


def test_tracks_have_the_true_role_team_and_number(result: ClipResult, scene: SyntheticScene) -> None:
    for track in result.tracks:
        persons = [true_person(scene, observation.frame, observation.box) for observation in track.observations]
        assert {person.role for person in persons} == {track.role}
        assert {person.team_key for person in persons} == {track.team_key}
        assert track.team_side == SIDE_OF_TEAM[track.team_key]
        if track.role in TEAM_ROLES:
            assert track.number_export in {person.number for person in persons}
            assert track.number_display in (None, track.number_export)
        else:
            assert track.number_export is None
            assert track.number_display is None
    # A number is shown when two crops of a track, one second apart, agree on it.
    assert sum(track.number_display is not None for track in result.tracks) >= 10


def test_every_frame_has_a_pitch_mapping_that_follows_the_camera(result: ClipResult, scene: SyntheticScene) -> None:
    """Keyframes keep their landmark fit. The other frames borrow a fit through the measured camera motion."""
    assert [record.frame for record in result.homographies] == list(range(N_FRAMES))
    errors = []
    for record in result.homographies:
        assert record.source == ("landmarks" if record.frame in KEYFRAMES else "propagated")
        frame_errors = mapping_errors_m(record, scene)
        # A mapping in the wrong direction, or a mirrored one, is tens of metres away.
        assert float(np.median(frame_errors)) < 1.0
        errors.append(frame_errors)
    assert float(np.median(np.concatenate(errors))) < 0.5


def test_clip_of_thirty_seconds_keeps_one_track_per_person() -> None:
    """A clip of 750 frames: the camera pans over the whole pitch, so people leave the picture and come back.

    The median position error does not see a person who comes back with another track id. The
    identity values, the shown numbers and the boxes outside the picture do.
    """
    long_scene = SyntheticScene(n_frames=750, seed=0)
    clip_result = run_clip(SyntheticSource(long_scene), FakeBackend(long_scene), motion=exact_motion(long_scene))
    long_truth = [long_scene.frame_state(frame) for frame in range(750)]
    identity = identity_scores(clip_result.frames, long_truth)
    # Measured with the exact fake model: 23 tracks for 23 people, a coverage of 0.96 and no switch.
    assert identity.coverage > 0.85
    assert identity.switches <= 5
    assert identity.track_ratio < 1.2
    states = [player for state in clip_result.frames for player in state.players]
    assert len(states) > 0.9 * sum(len(state.players) for state in long_truth)
    # A track that is cut into pieces has too few crops for a confident number.
    team_states = [player for player in states if player.role in TEAM_ROLES]
    assert sum(player.number_display is not None for player in team_states) > 0.8 * len(team_states)
    # No state stands for a person whose box lies outside the picture.
    width, height = long_scene.info.width, long_scene.info.height
    for player in states:
        assert player.box.x2 > 0 and player.box.y2 > 0 and player.box.x1 < width and player.box.y1 < height
    assert float(np.median(position_errors_m(clip_result.frames, long_truth))) < 0.5


# --- Model requests ----------------------------------------------------------------------------


def test_requests_follow_the_settings(recorded: tuple[ClipResult, list[VLMRequest]], scene: SyntheticScene) -> None:
    clip_result, requests = recorded
    assert {request.meta.step for request in requests} == set(MODEL_STEPS)
    for request in requests:
        assert request.meta.clip_id == scene.info.clip_id
        assert request.meta.repeat == OTHER_SETTINGS.repeat
        assert request.thinking_level == "low"
        assert request.service_tier == "standard"
        assert request.model == ("gemini-3.8-flash" if request.meta.step == "people" else DEFAULT_MODEL)
    assert clip_result.engines["people"] == "model:gemini-3.8-flash"
    assert clip_result.engines["landmarks"] == f"model:{DEFAULT_MODEL}"


def test_kit_legend_request_shows_the_first_middle_and_last_keyframe(
    recorded: tuple[ClipResult, list[VLMRequest]], scene: SyntheticScene
) -> None:
    _, requests = recorded
    (request,) = requests_of(requests, "kit_legend")
    assert request.meta.frame is None
    assert request.resolutions == ("high", "high", "high")
    shown_frames = (KEYFRAMES[0], KEYFRAMES[len(KEYFRAMES) // 2], KEYFRAMES[-1])
    assert shown_frames == (0, 48, 74)
    expected = tuple(hashlib.sha256(encode_jpeg(scene.image(frame))).hexdigest() for frame in shown_frames)
    assert request.image_hashes == expected


def test_each_keyframe_has_one_people_and_one_landmarks_request(
    recorded: tuple[ClipResult, list[VLMRequest]], scene: SyntheticScene
) -> None:
    _, requests = recorded
    people_requests = requests_of(requests, "people")
    landmarks_requests = requests_of(requests, "landmarks")
    assert sorted(request.meta.frame for request in people_requests) == KEYFRAMES
    assert sorted(request.meta.frame for request in landmarks_requests) == KEYFRAMES
    for request in people_requests:
        assert request.meta.item is None
        assert request.resolutions == ("high",)
        assert request.image_hashes == (hashlib.sha256(encode_jpeg(scene.image(request.meta.frame))).hexdigest(),)
    for request in landmarks_requests:
        # The numbered pitch diagram comes first, then the frame.
        assert request.resolutions == ("medium", "high")
        assert request.image_hashes[1] == hashlib.sha256(encode_jpeg(scene.image(request.meta.frame))).hexdigest()


def test_number_requests_show_tall_crops_one_second_apart(
    recorded: tuple[ClipResult, list[VLMRequest]], scene: SyntheticScene
) -> None:
    clip_result, requests = recorded
    fps = scene.info.fps
    number_requests = requests_of(requests, "number")
    numbered_tracks = [track for track in clip_result.tracks if track.role in TEAM_ROLES]
    # One request per player or goalkeeper track, and none for the referee.
    by_track = {track.track_id: number_request_of(number_requests, track) for track in numbered_tracks}
    assert len(number_requests) == len(numbered_tracks)
    assert len({id(request) for request in by_track.values()}) == len(numbered_tracks)
    assert len(numbered_tracks) < len(clip_result.tracks)

    crop_counts = []
    for track in numbered_tracks:
        request = by_track[track.track_id]
        frames = request.context["frames"]
        boxes = {observation.frame: observation.box for observation in track.observations}
        crop_counts.append(len(frames))
        # Every track has two observations or more, so every request shows two crops or more.
        assert 2 <= len(frames) <= OTHER_SETTINGS.number_crops
        tallest = max(track.observations, key=lambda observation: observation.box.height).frame
        assert tallest in frames
        if all(later - earlier >= fps for earlier, later in zip(frames, frames[1:])):
            # The crops are in frame order and at least one second apart, and no other observation could be added.
            for observation in track.observations:
                assert observation.frame in frames or any(abs(observation.frame - frame) < fps for frame in frames)
        else:
            # No observation is one second away from the tallest box: the second crop is the
            # observation that is furthest in time from it.
            (other,) = [frame for frame in frames if frame != tallest]
            assert all(abs(observation.frame - tallest) < fps for observation in track.observations)
            assert abs(other - tallest) == max(abs(observation.frame - tallest) for observation in track.observations)
        # The context names the box of the person in the frame, as [x1, y1, x2, y2].
        assert request.context["boxes"] == [corners(boxes[frame]) for frame in frames]

        images = images_of(request)
        assert len(images) == len(frames)
        for image, frame in zip(images, frames, strict=True):
            assert image.resolution == "low"
            crop = decoded(image)
            # The crop is the box plus 15% on every side, cut at the image border, and 4 times enlarged.
            # Its height is along axis 0 and its width along axis 1.
            for size_px, box_side in ((crop.shape[0], boxes[frame].height), (crop.shape[1], boxes[frame].width)):
                assert size_px % 4 == 0
                assert 4.0 * box_side <= size_px <= 4.0 * (1.3 * box_side + 2.0)
    # A person who is in the picture for the whole clip gives three crops in three seconds.
    assert max(crop_counts) == 3


def test_number_request_item_follows_what_the_crops_show(scene: SyntheticScene) -> None:
    """The cache key of a request sees its item and not its pixels. So the item must change with the crops."""
    box = Box(x1=100.0, y1=200.0, x2=150.0, y2=290.0)
    box_to_the_right = Box(x1=300.0, y1=200.0, x2=350.0, y2=290.0)
    request = number_request_of_track_7(scene, [(10, box), (40, box)])
    # The item names the crops and not the track: a track id of the pipeline is a place in a list.
    assert re.fullmatch(r"crops-[0-9a-f]{16}", request.meta.item or ""), request.meta.item
    # The same crops give the same item and the same cache key, so that their stored answer is found again,
    # also when their track has another id.
    same_crops = number_request_of_track_7(scene, [(10, box), (40, box)])
    assert (same_crops.meta.item, same_crops.cache_key()) == (request.meta.item, request.cache_key())
    other_id = number_request_of_track_7(scene, [(10, box), (40, box)], track_id=8)
    assert (other_id.meta.item, other_id.cache_key()) == (request.meta.item, request.cache_key())
    # The same track id, team and number of crops, but another frame, another place in the frame or
    # another enlargement: the crops show other pixels, so the stored answer must not be found.
    other_crops = [
        number_request_of_track_7(scene, [(10, box), (41, box)]),
        number_request_of_track_7(scene, [(10, box), (40, box_to_the_right)]),
        number_request_of_track_7(scene, [(10, box), (40, box)], crop_upscale=2.0),
    ]
    for other in other_crops:
        assert other.meta.item != request.meta.item
        assert other.cache_key() != request.cache_key()


def test_roster_gives_names_to_displayed_numbers(recorded: tuple[ClipResult, list[VLMRequest]]) -> None:
    clip_result, _ = recorded
    names = {}
    for track in clip_result.tracks:
        if track.number_display is None or track.team_side is None:
            assert track.name is None
        else:
            assert track.name == f"{track.team_side.capitalize()} {track.number_display}"
        names[track.track_id] = track.name
    assert sum(name is not None for name in names.values()) >= 10
    assert {name for name in names.values() if name is not None} >= {"Left 7", "Right 10"}
    for state in clip_result.frames:
        for player in state.players:
            assert player.name == names[player.track_id]


# --- Cache -------------------------------------------------------------------------------------


def test_second_run_makes_no_new_model_call(
    scene: SyntheticScene, source: SyntheticSource, true_motion: MotionField, tmp_path: Path
) -> None:
    """With a cassette store, a second run of the same clip is answered from the stored answers."""
    inner = RecordingBackend(FakeBackend(scene))
    cassette_folder = tmp_path / "cassettes"
    backend = CachedBackend(CassetteStore(cassette_folder), inner)

    first = run_clip(source, backend, motion=true_motion)
    first_calls = len(inner.requests)
    numbered_tracks = sum(track.role in TEAM_ROLES for track in first.tracks)
    assert first_calls == 1 + 2 * len(KEYFRAMES) + numbered_tracks
    assert len(list(cassette_folder.rglob("*.json"))) == first_calls

    second = run_clip(source, backend, motion=true_motion)
    assert len(inner.requests) == first_calls
    assert second == first

    # Replay mode: a backend without an inner backend needs no model at all.
    replayed = run_clip(source, CachedBackend(CassetteStore(cassette_folder)), motion=true_motion)
    assert replayed == first


def test_stored_number_answer_is_not_used_for_other_crops(
    scene: SyntheticScene, source: SyntheticSource, true_motion: MotionField, tmp_path: Path
) -> None:
    """A run that cuts other crops asks for the shirt numbers again. It does not take the stored answers.

    Another enlargement keeps the tracks, their ids, the crop frames and the boxes. Only the pixels of
    the crops change, and the cache key does not see pixels.
    """
    inner = RecordingBackend(FakeBackend(scene))
    cassette_folder = tmp_path / "cassettes"
    backend = CachedBackend(CassetteStore(cassette_folder), inner)
    first = run_clip(source, backend, motion=true_motion)
    first_calls = len(inner.requests)
    numbered_tracks = [track for track in first.tracks if track.role in TEAM_ROLES]
    assert numbered_tracks

    run_clip(source, backend, PipelineConfig(crop_upscale=2.0), motion=true_motion)
    # The kit legend, the people and the landmarks come from the store. Every number request is new.
    new_requests = inner.requests[first_calls:]
    assert [request.meta.step for request in new_requests] == ["number"] * len(numbered_tracks)
    for track in numbered_tracks:
        number_request_of(new_requests, track)

    # Replay mode has no answer for crops that were never sent. The run stops, and no other answer is taken.
    replay = RecordingBackend(CachedBackend(CassetteStore(cassette_folder)))
    with pytest.raises(CacheMiss):
        run_clip(source, replay, PipelineConfig(crop_upscale=3.0), motion=true_motion)
    # The number step is the last model step. The run came that far, so the missing answer is one of that step.
    assert requests_of(replay.requests, "number")


def test_stored_number_answers_are_found_again_when_the_tracks_get_other_ids(
    scene: SyntheticScene,
    source: SyntheticSource,
    true_motion: MotionField,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A track id is a place in the track list. A change of the tracking code or settings gives other ids.

    The second run has the same tracks with other ids. Its number requests show the same crops with the
    same prompt, so their answers are in the store and no request is paid again.
    """
    inner = RecordingBackend(FakeBackend(scene))
    backend = CachedBackend(CassetteStore(tmp_path / "cassettes"), inner)
    first = run_clip(source, backend, motion=true_motion)
    first_calls = len(inner.requests)
    assert requests_of(inner.requests, "number")
    real_tracks = pipeline._tracks

    def tracks_with_other_ids(*args: Any, **kwargs: Any) -> list[Track]:
        return [track.model_copy(update={"track_id": track.track_id + 100}) for track in real_tracks(*args, **kwargs)]

    monkeypatch.setattr(pipeline, "_tracks", tracks_with_other_ids)

    second = run_clip(source, backend, motion=true_motion)

    assert [track.track_id for track in second.tracks] == [track.track_id + 100 for track in first.tracks]
    assert inner.requests[first_calls:] == []
    assert [track.number_export for track in second.tracks] == [track.number_export for track in first.tracks]


# --- Checks of the code steps ----------------------------------------------------------------


def test_person_far_outside_the_pitch_is_dropped(
    scene: SyntheticScene,
    source: SyntheticSource,
    true_motion: MotionField,
    recorded: tuple[ClipResult, list[VLMRequest]],
) -> None:
    """A person who stands 11 m behind the far touchline in every keyframe gets no track."""
    clean_result, _ = recorded
    backend = RewritingBackend(FakeBackend(scene), "people", person_behind_the_far_touchline(scene))
    clip_result = run_clip(source, backend, motion=true_motion)
    assert all(track.role != "other" for track in clip_result.tracks)
    assert len(clip_result.tracks) == len(clean_result.tracks)
    positions = np.array([[player.pitch_x, player.pitch_y] for state in clip_result.frames for player in state.players])
    assert inside_pitch(positions, PipelineConfig().pitch_margin_m).all()


def test_mirrored_landmark_fit_is_rejected(
    scene: SyntheticScene,
    source: SyntheticSource,
    true_motion: MotionField,
    truth: list[FrameState],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A keyframe whose landmarks give a mirrored mapping borrows the mapping of a keyframe near it."""
    backend = RewritingBackend(FakeBackend(scene), "landmarks", mirrored_landmarks_of_frame(24))
    with caplog.at_level(logging.INFO, logger="pitchvlm.pipeline"):
        clip_result = run_clip(source, backend, motion=true_motion)
    assert "keyframe 24" in caplog.text
    assert "mirrored" in caplog.text
    for frame in KEYFRAMES:
        assert clip_result.homographies[frame].source == ("propagated" if frame == 24 else "landmarks")
    assert float(np.median(mapping_errors_m(clip_result.homographies[24], scene))) < 0.5
    assert float(np.median(position_errors_m(clip_result.frames, truth))) < 0.5


@pytest.mark.parametrize(("max_stitch_gap_s", "filled"), [(0.48, True), (0.44, False)])
def test_dense_output_fills_no_gap_above_the_stitch_gap_setting(
    scene: SyntheticScene,
    source: SyntheticSource,
    true_motion: MotionField,
    max_stitch_gap_s: float,
    filled: bool,
) -> None:
    """Two keyframes are 12 frames apart: 0.48 s. A limit of 0.44 s is 11 frames at 25 frames per second."""
    config = PipelineConfig(tracking=TrackingConfig(max_stitch_gap_s=max_stitch_gap_s))
    clip_result = run_clip(source, FakeBackend(scene), config, motion=true_motion)
    frames_with_people = [state.frame for state in clip_result.frames if state.players]
    if filled:
        assert frames_with_people == list(range(N_FRAMES))
    else:
        # Only the keyframes have people. The last two keyframes are 2 frames apart, so frame 73 is filled.
        assert frames_with_people == [*KEYFRAMES[:-1], 73, KEYFRAMES[-1]]
        assert all(player.observed for frame in KEYFRAMES for player in clip_result.frames[frame].players)


def test_clip_of_one_frame_runs() -> None:
    """One frame gives one keyframe: three requests, and no track, because a track needs two observations."""
    one_frame_scene = SyntheticScene(n_frames=1, seed=SEED)
    backend = RecordingBackend(FakeBackend(one_frame_scene))
    clip_result = run_clip(SyntheticSource(one_frame_scene), backend)
    assert sorted(request.meta.step for request in backend.requests) == ["kit_legend", "landmarks", "people"]
    (kit_legend_request,) = requests_of(backend.requests, "kit_legend")
    # The first, the middle and the last keyframe are the same frame. It is sent one time.
    assert len(images_of(kit_legend_request)) == 1
    assert clip_result.tracks == ()
    assert clip_result.frames == (FrameState(frame=0),)
    assert [record.source for record in clip_result.homographies] == ["landmarks"]


def test_clip_without_any_pitch_mapping_gets_no_shirt_number_request(
    scene: SyntheticScene, source: SyntheticSource, true_motion: MotionField, caplog: pytest.LogCaptureFixture
) -> None:
    """No landmarks answer names a landmark. No person gets a state then, so no shirt number is asked for."""

    def without_landmarks(request: VLMRequest, text: str) -> str:
        answer = json.loads(text)
        answer["landmarks"] = []
        return json.dumps(answer)

    backend = RecordingBackend(RewritingBackend(FakeBackend(scene), "landmarks", without_landmarks))
    with caplog.at_level(logging.WARNING, logger="pitchvlm.pipeline"):
        clip_result = run_clip(source, backend, motion=true_motion)

    # One kit legend request, and one people and one landmarks request for each keyframe.
    assert len(backend.requests) == 1 + 2 * len(KEYFRAMES)
    assert requests_of(backend.requests, "number") == []
    assert not any(record.valid for record in clip_result.homographies)
    assert all(not state.players for state in clip_result.frames)
    # The people are still tracked. No track has a number for the video, because no crop was read.
    assert len(clip_result.tracks) > 0
    assert all(track.number_display is None for track in clip_result.tracks)
    assert "no keyframe of clip SNGS-999 has a pitch mapping" in caplog.text
    assert "no shirt-number request is sent" in caplog.text


# --- Pitch mapping: how well the landmarks support a fit -------------------------------------


def test_weak_landmark_fit_takes_the_mapping_of_a_well_supported_keyframe(
    scene: SyntheticScene,
    source: SyntheticSource,
    true_motion: MotionField,
    truth: list[FrameState],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keyframe 24 names 4 landmarks, and two of them are 18 pixels off: a fit from them alone is 3.5 m off.

    The keyframes next to it name 18 and 15 landmarks. The camera motion carries their mapping over.
    """
    backend = RewritingBackend(FakeBackend(scene), "landmarks", few_landmarks(4, frames={24}, moves=WRONG_FIT_MOVES))
    with caplog.at_level(logging.INFO, logger="pitchvlm.pipeline"):
        clip_result = run_clip(source, backend, motion=true_motion)
    assert key_sources(clip_result) == ["landmarks", "landmarks", "propagated", *["landmarks"] * 5]
    assert "keyframe 24: only 4 landmarks agree with the landmark fit" in caplog.text
    assert float(np.median(mapping_errors_m(clip_result.homographies[24], scene))) < 0.1
    # No person is some metres off around keyframe 24.
    assert float(np.max(position_errors_m(clip_result.frames, truth))) < 1.0


def test_weak_landmark_fit_is_kept_when_its_landmarks_do_not_agree_with_the_carried_mapping(
    scene: SyntheticScene,
    source: SyntheticSource,
    true_motion: MotionField,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The keyframes 12 and 36 name all their landmarks 51 pixels (40 model units) to the right of their place.

    All landmarks of such a keyframe agree with one mapping, so its fit is well supported, but it is
    about 2 m off. Keyframe 24 between them names 5 landmarks at their right places. They lie 51
    pixels beside the carried mapping, more than the near limit of 24 pixels, so keyframe 24 keeps
    its own fit.
    """
    wrong_neighbours = RewritingBackend(FakeBackend(scene), "landmarks", landmarks_shifted({12, 36}, 40))
    backend = RewritingBackend(wrong_neighbours, "landmarks", few_landmarks(5, frames={24}))
    with caplog.at_level(logging.INFO, logger="pitchvlm.pipeline"):
        clip_result = run_clip(source, backend, motion=true_motion)
    assert float(np.median(mapping_errors_m(clip_result.homographies[12], scene))) > 1.0
    record = clip_result.homographies[24]
    assert (record.source, record.n_inliers) == ("landmarks", 5)
    assert float(np.median(mapping_errors_m(record, scene))) < 0.1
    assert "keyframe 24: its landmarks do not agree with the mapping of a well-supported keyframe" in caplog.text


@pytest.mark.parametrize(("n_near", "agree"), [(3, True), (2, False)])
def test_carried_mapping_needs_more_than_half_of_the_landmarks_near_it(
    scene: SyntheticScene, n_near: int, agree: bool
) -> None:
    """Four landmarks of keyframe 24: some at their place, the others 64 pixels (50 model units) beside it."""
    exact = true_record(scene, 24)
    marks = scene.landmarks(24).landmarks[:4]
    moved = tuple(
        mark if index < n_near else mark.model_copy(update={"x": mark.x + 64.0}) for index, mark in enumerate(marks)
    )
    landmarks = pipeline.FrameLandmarks(frame=24, landmarks=moved)
    assert pipeline._landmarks_agree(landmarks, exact, 8.0) is agree
    # The limit is three times the RANSAC threshold: with 22 px, 64 px are near the mapping.
    assert pipeline._landmarks_agree(landmarks, exact, 22.0)


@pytest.mark.parametrize(("count", "own_fit"), [(6, False), (7, True)])
def test_landmark_fit_is_well_supported_from_seven_agreeing_landmarks(
    scene: SyntheticScene, source: SyntheticSource, true_motion: MotionField, count: int, own_fit: bool
) -> None:
    backend = RewritingBackend(FakeBackend(scene), "landmarks", few_landmarks(count, frames={24}))
    record = run_clip(source, backend, motion=true_motion).homographies[24]
    assert pipeline.STRONG_FIT_MIN_INLIERS == 7
    assert (record.source, record.n_inliers) == (("landmarks", count) if own_fit else ("propagated", 0))


def test_weak_landmark_fits_are_kept_when_no_keyframe_is_well_supported(
    scene: SyntheticScene, source: SyntheticSource, true_motion: MotionField, truth: list[FrameState]
) -> None:
    """Every keyframe names 5 landmarks. A mapping from few landmarks is better than no mapping."""
    backend = RewritingBackend(FakeBackend(scene), "landmarks", few_landmarks(5))
    clip_result = run_clip(source, backend, motion=true_motion)
    assert key_sources(clip_result) == ["landmarks"] * len(KEYFRAMES)
    assert [clip_result.homographies[frame].n_inliers for frame in KEYFRAMES] == [5] * len(KEYFRAMES)
    assert float(np.median(position_errors_m(clip_result.frames, truth))) < 0.5


def test_weak_landmark_fit_is_kept_when_no_well_supported_keyframe_is_in_reach(
    scene: SyntheticScene, source: SyntheticSource, true_motion: MotionField
) -> None:
    """With a reach of 6 frames, the keyframes 12 frames away cannot give their mapping to keyframe 24."""
    backend = RewritingBackend(FakeBackend(scene), "landmarks", few_landmarks(4, frames={24}, moves=WRONG_FIT_MOVES))
    clip_result = run_clip(source, backend, PipelineConfig(max_gap_frames=6), motion=true_motion)
    assert (clip_result.homographies[24].source, clip_result.homographies[24].n_inliers) == ("landmarks", 4)


def test_keyframe_without_a_fit_takes_a_well_supported_mapping_and_not_a_nearer_weak_one(
    scene: SyntheticScene, source: SyntheticSource, true_motion: MotionField
) -> None:
    """Keyframe 72 names 3 landmarks: no fit. Keyframe 74, 2 frames away, has a weak fit that is some metres off.

    Keyframe 60 has 8 landmarks. With a reach of 13 frames it reaches keyframe 72 and not keyframe 74.
    """
    no_fit_at_72 = RewritingBackend(FakeBackend(scene), "landmarks", few_landmarks(3, frames={72}))
    backend = RewritingBackend(no_fit_at_72, "landmarks", few_landmarks(4, frames={74}, moves=WRONG_FIT_MOVES))
    clip_result = run_clip(source, backend, PipelineConfig(max_gap_frames=13), motion=true_motion)
    sources = [clip_result.homographies[frame].source for frame in (60, 72, 74)]
    assert sources == ["landmarks", "propagated", "landmarks"]
    assert clip_result.homographies[74].n_inliers == 4
    assert float(np.median(mapping_errors_m(clip_result.homographies[74], scene))) > 1.0
    assert float(np.median(mapping_errors_m(clip_result.homographies[72], scene))) < 0.1


# --- Pitch mapping: people outside the pitch -----------------------------------------------


def test_people_outside_the_pitch_do_not_reject_a_well_supported_fit(
    scene: SyntheticScene,
    source: SyntheticSource,
    true_motion: MotionField,
    recorded: tuple[ClipResult, list[VLMRequest]],
) -> None:
    """Two people whom the model names players warm up 6 m behind the far touchline in every keyframe.

    A keyframe shows 12 to 19 people, so with these two fewer than 90% of the foot points are inside
    the pitch plus 3 m. The landmark fits have 7 to 18 agreeing landmarks and are right.
    """
    clean_result, _ = recorded
    backend = RewritingBackend(FakeBackend(scene), "people", people_added(scene, WARM_UP_SPOTS))
    clip_result = run_clip(source, backend, motion=true_motion)
    assert key_sources(clip_result) == ["landmarks"] * len(KEYFRAMES)
    # The two people are dropped when the mapping is known: they get no track and no state.
    assert len(clip_result.tracks) == len(clean_result.tracks)
    positions = np.array([[player.pitch_x, player.pitch_y] for state in clip_result.frames for player in state.players])
    assert inside_pitch(positions, PipelineConfig().pitch_margin_m).all()


@pytest.mark.parametrize(
    ("n_inliers", "n_on_pitch", "n_outside", "plausible"),
    [
        # A well-supported fit: one person outside never decides, and up to a third of the people may be outside.
        (7, 0, 1, True),
        (7, 6, 3, True),
        (7, 5, 3, False),
        (7, 3, 6, False),
        (7, 0, 9, False),
        (12, 0, 2, False),
        # A weaker fit must put 90 percent of the people on the pitch.
        (6, 8, 1, True),
        (6, 18, 2, True),
        (6, 17, 2, False),
        (6, 6, 3, False),
    ],
)
def test_fit_is_not_plausible_when_too_many_people_map_outside_the_pitch(
    scene: SyntheticScene, n_inliers: int, n_on_pitch: int, n_outside: int, plausible: bool
) -> None:
    """The mapping is the true mapping of frame 0. The people outside stand 6 m behind the far touchline."""
    assert pipeline.STRONG_FIT_MIN_SHARE_ON_PITCH == 2.0 / 3.0
    assert pipeline.MAX_PEOPLE_OUTSIDE == 1
    record = true_record(scene, 0).model_copy(update={"n_inliers": n_inliers})
    on_pitch = [(-40.0 + 2.0 * index, -20.0 + 2.0 * index) for index in range(n_on_pitch)]
    outside = [(-40.0 + 3.0 * index, -40.0) for index in range(n_outside)]
    to_image = np.linalg.inv(scene.homography(0))
    feet = apply_homography(to_image, np.array([*on_pitch, *outside], dtype=np.float64).reshape(-1, 2))

    problems = pipeline.fit_problems(record, feet, (1280, 720), pitch_margin_m=3.0)

    assert (problems == []) is plausible
    if not plausible:
        assert "of the foot points map into the pitch plus 3 m" in problems[0]


@pytest.mark.parametrize(("n_outside", "accepted"), [(1, True), (2, False)])
def test_one_person_outside_the_pitch_does_not_reject_a_weak_fit(
    scene: SyntheticScene,
    source: SyntheticSource,
    true_motion: MotionField,
    caplog: pytest.LogCaptureFixture,
    n_outside: int,
    accepted: bool,
) -> None:
    """Every keyframe names 5 landmarks and shows 8 people on the pitch, plus 1 or 2 people outside it.

    With 1 person outside, 8 of 9 foot points (89%) are inside: below the limit of 90%, but one
    person must not decide. With 2 people outside every fit is rejected. The clip then has no
    mapping and no state, and a warning says so.
    """
    weak_fits = RewritingBackend(FakeBackend(scene), "landmarks", few_landmarks(5))
    backend = RewritingBackend(weak_fits, "people", people_added(scene, WARM_UP_SPOTS[:n_outside], keep=8))
    with caplog.at_level(logging.WARNING, logger="pitchvlm.pipeline"):
        clip_result = run_clip(source, backend, motion=true_motion)
    if accepted:
        assert key_sources(clip_result) == ["landmarks"] * len(KEYFRAMES)
        assert "has a pitch mapping" not in caplog.text
        assert all(state.players for state in clip_result.frames)
    else:
        assert key_sources(clip_result) == ["none"] * len(KEYFRAMES)
        assert all(not state.players for state in clip_result.frames)
        assert "no keyframe of clip SNGS-999 has a pitch mapping" in caplog.text
        assert "8 of 8 keyframes gave a landmark fit that was rejected" in caplog.text


def test_people_with_the_role_other_do_not_count_against_a_weak_fit(
    scene: SyntheticScene, source: SyntheticSource, true_motion: MotionField
) -> None:
    """Six people of a bench sit 11 m behind the far touchline. They do not stand on the pitch."""
    bench = [(BEHIND_FAR_TOUCHLINE_XY[0] + 1.5 * index, BEHIND_FAR_TOUCHLINE_XY[1]) for index in range(6)]
    weak_fits = RewritingBackend(FakeBackend(scene), "landmarks", few_landmarks(5))
    backend = RewritingBackend(weak_fits, "people", people_added(scene, bench, role="other", keep=8))
    clip_result = run_clip(source, backend, motion=true_motion)
    assert key_sources(clip_result) == ["landmarks"] * len(KEYFRAMES)


@pytest.mark.parametrize(
    ("keep", "added"),
    [
        # 7 people on the pitch, one 11 m outside and one 5 m outside. With the margin of 7 m one of 9 is outside.
        (7, (BEHIND_FAR_TOUCHLINE_XY, FIVE_METRES_OUT_SPOTS[0])),
        # 20 people on the pitch, two 11 m outside and two 5 m outside. With the margin of 7 m, 22 of 24 foot
        # points (92%) are inside. With the default margin of 3 m it would be 20 of 24 (83%).
        (8, (*ON_PITCH_SPOTS, BEHIND_FAR_TOUCHLINE_XY, (-17.0, -45.0), *FIVE_METRES_OUT_SPOTS)),
    ],
)
def test_pitch_margin_of_the_settings_reaches_the_plausibility_check_of_a_weak_fit(
    scene: SyntheticScene,
    source: SyntheticSource,
    true_motion: MotionField,
    keep: int,
    added: tuple[tuple[float, float], ...],
) -> None:
    weak_fits = RewritingBackend(FakeBackend(scene), "landmarks", few_landmarks(5))
    backend = RewritingBackend(weak_fits, "people", people_added(scene, added, keep=keep))
    clip_result = run_clip(source, backend, PipelineConfig(pitch_margin_m=7.0), motion=true_motion)
    assert key_sources(clip_result) == ["landmarks"] * len(KEYFRAMES)


def test_plausibility_check_gets_the_width_and_then_the_height_of_the_image(
    scene: SyntheticScene, source: SyntheticSource, true_motion: MotionField, monkeypatch: pytest.MonkeyPatch
) -> None:
    check = Spy(monkeypatch, "check_homography")
    run_clip(source, FakeBackend(scene), motion=true_motion)
    assert len(check.calls) == len(KEYFRAMES)
    assert all(args[1] == (1280, 720) for args, _ in check.calls)


def test_pitch_margin_of_the_settings_keeps_a_person_next_to_the_pitch(
    scene: SyntheticScene, source: SyntheticSource, true_motion: MotionField
) -> None:
    """A person 11 m behind the far touchline is dropped with the default margin of 3 m, and kept with 15 m."""
    backend = RewritingBackend(FakeBackend(scene), "people", person_behind_the_far_touchline(scene))
    clip_result = run_clip(source, backend, PipelineConfig(pitch_margin_m=15.0), motion=true_motion)
    (other,) = [track for track in clip_result.tracks if track.role == "other"]
    assert len(other.observations) == len(KEYFRAMES)
    states = [player for state in clip_result.frames for player in state.players if player.track_id == other.track_id]
    assert len(states) == N_FRAMES
    for player in states:
        assert (player.pitch_x, player.pitch_y) == pytest.approx(BEHIND_FAR_TOUCHLINE_XY, abs=1.0)


def test_keyframe_without_a_pitch_mapping_keeps_its_people(scene: SyntheticScene) -> None:
    people = scene.people(0)
    assert len(people.people) >= 8
    assert pipeline._people_on_pitch(people, lambda frame, xy: None, 3.0) == people


# --- People answers --------------------------------------------------------------------------


def test_box_that_a_people_answer_repeats_gives_no_second_track(
    scene: SyntheticScene,
    source: SyntheticSource,
    true_motion: MotionField,
    recorded: tuple[ClipResult, list[VLMRequest]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every people answer lists every person two times. The tracks and the states are those of clean answers."""
    clean_result, _ = recorded
    n_listed = sum(len(scene.people(frame).people) for frame in KEYFRAMES)
    backend = RecordingBackend(RewritingBackend(FakeBackend(scene), "people", everybody_twice))
    with caplog.at_level(logging.WARNING, logger="pitchvlm.pipeline"):
        clip_result = run_clip(source, backend, motion=true_motion)
    assert len(clip_result.tracks) == len(clean_result.tracks)
    assert [len(state.players) for state in clip_result.frames] == [len(state.players) for state in clean_result.frames]
    n_numbered_tracks = sum(track.role in TEAM_ROLES for track in clean_result.tracks)
    assert len(requests_of(backend.requests, "number")) == n_numbered_tracks
    assert f"the people answers of clip SNGS-999 repeat {n_listed} boxes" in caplog.text


def test_repeated_box_is_a_box_with_an_overlap_above_ninety_percent() -> None:
    def person(x1: float, team_key: str) -> PersonDet:
        return PersonDet(box=Box(x1=x1, y1=200.0, x2=x1 + 100.0, y2=400.0), role="player", team_key=team_key)

    first = person(100.0, "A")
    # 4 pixels to the right: the overlap is 96 / 104 = 0.92 of the union. 6 pixels: 94 / 106 = 0.89.
    answered = pipeline.FramePeople(frame=0, people=(first, person(104.0, "B"), person(106.0, "B")))
    kept = pipeline._without_repeated_boxes(answered)
    assert pipeline.REPEATED_BOX_MIN_IOU == 0.9
    assert [(kept_person.box.x1, kept_person.team_key) for kept_person in kept.people] == [(100.0, "A"), (106.0, "B")]
    assert kept.frame == 0


def test_shifted_box_with_the_same_labels_is_a_repeated_box_above_sixty_percent_overlap() -> None:
    def person(x1: float, team_key: str = "A", role: str = "player") -> PersonDet:
        return PersonDet(box=Box(x1=x1, y1=200.0, x2=x1 + 100.0, y2=400.0), role=role, team_key=team_key)

    assert pipeline.REPEATED_BOX_SAME_LABELS_MIN_IOU == 0.6
    first = person(100.0)
    # 24 pixels to the right: the overlap is 76 / 124 = 0.61 of the union. 25 pixels: 75 / 125 = 0.60.
    same_labels = pipeline.FramePeople(frame=3, people=(first, person(124.0), person(125.0)))
    kept = pipeline._without_repeated_boxes(same_labels)
    assert [kept_person.box.x1 for kept_person in kept.people] == [100.0, 125.0]
    # With another team key or another role the two boxes can be two people in a duel: both stay.
    for other in (person(124.0, team_key="B"), person(124.0, role="goalkeeper"), person(124.0, team_key="none")):
        other_labels = pipeline.FramePeople(frame=3, people=(first, other))
        assert pipeline._without_repeated_boxes(other_labels) == other_labels
    # The team key "none" is a label like the others: two boxes with it repeat each other.
    unknown_team = pipeline.FramePeople(frame=3, people=(person(100.0, "none"), person(124.0, "none")))
    assert len(pipeline._without_repeated_boxes(unknown_team).people) == 1


def test_person_whom_the_model_boxes_two_times_with_a_small_shift_gets_one_track(
    scene: SyntheticScene,
    source: SyntheticSource,
    true_motion: MotionField,
    recorded: tuple[ClipResult, list[VLMRequest]],
) -> None:
    """Every people answer lists every person a second time, with a box that is an eighth of its width further right."""
    clean_result, _ = recorded

    def everybody_twice_with_a_shift(request: VLMRequest, text: str) -> str:
        answer = json.loads(text)
        doubles = []
        for person in answer["people"]:
            y1, x1, y2, x2 = person["box_2d"]  # A model box is [ymin, xmin, ymax, xmax].
            shift = max(1, round((x2 - x1) / 8))
            doubles.append({**person, "box_2d": [y1, x1 + shift, y2, x2 + shift]})
        answer["people"] = [*answer["people"], *doubles]
        return json.dumps(answer)

    backend = RewritingBackend(FakeBackend(scene), "people", everybody_twice_with_a_shift)
    clip_result = run_clip(source, backend, motion=true_motion)
    assert len(clip_result.tracks) == len(clean_result.tracks)
    assert [len(state.players) for state in clip_result.frames] == [len(state.players) for state in clean_result.frames]


# --- Settings and inputs that must reach the code steps ------------------------------------------


def test_tracking_settings_and_the_frame_rate_reach_both_tracking_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clip at 12.5 frames per second, with tracking settings that are not the defaults."""
    slow_scene = SyntheticScene(n_frames=38, fps=12.5, seed=SEED)
    tracking = TrackingConfig(max_missed_keyframes=1, min_observations=3, max_stitch_gap_s=2.0)
    associate, stitch = Spy(monkeypatch, "associate"), Spy(monkeypatch, "stitch_fragments")
    crops, fill = Spy(monkeypatch, "_crop_plans"), Spy(monkeypatch, "dense_frame_states")

    config, motion = PipelineConfig(tracking=tracking), exact_motion(slow_scene)
    clip_result = run_clip(SyntheticSource(slow_scene), FakeBackend(slow_scene), config, motion=motion)

    assert [(kwargs["config"], kwargs["fps"]) for _, kwargs in associate.calls] == [(tracking, 12.5)]
    assert [(kwargs["config"], kwargs["fps"]) for _, kwargs in stitch.calls] == [(tracking, 12.5)]
    # The crops of a track are one second apart, so the crop choice needs the frame rate too.
    assert [args[1] for args, _ in crops.calls] == [12.5]
    # The dense output fills gaps up to 2 seconds: 25 frames at 12.5 frames per second.
    assert [kwargs["max_fill_gap_frames"] for _, kwargs in fill.calls] == [25]
    assert clip_result.tracks
    assert all(len(track.observations) >= 3 for track in clip_result.tracks)


def test_crops_of_a_clip_at_another_frame_rate_are_one_second_apart() -> None:
    """At 12.5 frames per second, one second is 12.5 frames: frame 13 is far enough from frame 0, frame 12 is not."""
    track = track_with_boxes([(0, 60.0), (12, 55.0), (13, 50.0)])
    assert [observation.frame for observation in pipeline._crop_plans([track], 12.5, 8)[0].observations] == [0, 13]


def test_gap_limit_of_the_settings_reaches_the_pitch_mapping(
    scene: SyntheticScene, source: SyntheticSource, true_motion: MotionField
) -> None:
    clip_result = run_clip(source, FakeBackend(scene), PipelineConfig(max_gap_frames=2), motion=true_motion)
    sources = [record.source for record in clip_result.homographies]
    # Keyframes 0 and 12: the frames 3 to 9 are more than 2 frames away from both.
    borrowed = ["propagated", "propagated"]
    assert sources[:13] == ["landmarks", *borrowed, *["none"] * 7, *borrowed, "landmarks"]


@pytest.mark.parametrize(("ransac_px", "agrees"), [(8.0, True), (2.0, False)])
def test_ransac_threshold_of_the_settings_reaches_the_landmark_fit(
    scene: SyntheticScene, source: SyntheticSource, true_motion: MotionField, ransac_px: float, agrees: bool
) -> None:
    """The first landmark of every answer is 5 model units (6.4 pixels) to the right of its place."""

    def move_first(request: VLMRequest, text: str) -> str:
        answer = json.loads(text)
        answer["landmarks"][0]["point"][1] += 5  # A model point is [y, x].
        return json.dumps(answer)

    backend = RewritingBackend(FakeBackend(scene), "landmarks", move_first)
    clip_result = run_clip(source, backend, PipelineConfig(ransac_px=ransac_px), motion=true_motion)
    n_landmarks = len(scene.landmarks(0).landmarks)
    assert clip_result.homographies[0].n_inliers == (n_landmarks if agrees else n_landmarks - 1)


def test_keyframe_interval_of_the_settings_is_used(
    scene: SyntheticScene, source: SyntheticSource, true_motion: MotionField
) -> None:
    backend = RecordingBackend(FakeBackend(scene))
    run_clip(source, backend, PipelineConfig(keyframe_interval_s=1.0), motion=true_motion)
    assert sorted(request.meta.frame for request in requests_of(backend.requests, "people")) == [0, 25, 50, 74]


def test_number_of_crops_of_the_settings_is_used(
    scene: SyntheticScene, source: SyntheticSource, true_motion: MotionField
) -> None:
    backend = RecordingBackend(FakeBackend(scene))
    run_clip(source, backend, PipelineConfig(number_crops=1), motion=true_motion)
    number_requests = requests_of(backend.requests, "number")
    assert len(number_requests) >= 10
    assert {len(images_of(request)) for request in number_requests} == {1}


def test_given_motion_field_is_used_and_no_motion_is_measured(
    scene: SyntheticScene, source: SyntheticSource, true_motion: MotionField, monkeypatch: pytest.MonkeyPatch
) -> None:
    def must_not_run(*args: Any, **kwargs: Any) -> MotionField:
        raise AssertionError("the motion was measured although a motion field was given")

    monkeypatch.setattr(pipeline, "compute_motion_field", must_not_run)
    assert len(run_clip(source, FakeBackend(scene), motion=true_motion).frames) == N_FRAMES


def test_people_of_the_keyframes_are_masked_when_the_motion_is_measured(
    scene: SyntheticScene, source: SyntheticSource, true_motion: MotionField, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []

    def exact(clip: object, **kwargs: Any) -> MotionField:
        seen.append(kwargs)
        return true_motion

    monkeypatch.setattr(pipeline, "compute_motion_field", exact)
    run_clip(source, FakeBackend(scene))
    (kwargs,) = seen
    # The scale of the motion images is the default of the motion module.
    assert set(kwargs) == {"people"}
    assert sorted(kwargs["people"]) == KEYFRAMES
    for frame, boxes in kwargs["people"].items():
        assert boxes == [person.box for person in parse_exact_people(scene, frame)]


def test_broken_motion_chain_between_two_keyframes_keeps_the_mappings_and_the_tracks(
    scene: SyntheticScene,
    source: SyntheticSource,
    truth: list[FrameState],
    recorded: tuple[ClipResult, list[VLMRequest]],
) -> None:
    """The camera motion between the frames 30 and 31 is not known, for example after a blurred frame."""
    clean_result, _ = recorded
    to_pitch = [scene.homography(frame) for frame in range(N_FRAMES)]
    steps: list[np.ndarray | None] = [
        np.linalg.inv(to_pitch[frame + 1]) @ to_pitch[frame] for frame in range(N_FRAMES - 1)
    ]
    steps[30] = None
    clip_result = run_clip(source, FakeBackend(scene), motion=MotionField(steps))
    # Every frame reaches a keyframe on its own side of the break.
    assert all(record.valid for record in clip_result.homographies)
    assert float(np.max([np.median(mapping_errors_m(record, scene)) for record in clip_result.homographies])) < 0.2
    # Across the missing step the people of the keyframes 24 and 36 are matched through the pitch mappings
    # of the two keyframes.
    assert len(clip_result.tracks) == len(clean_result.tracks)
    assert float(np.median(position_errors_m(clip_result.frames, truth))) < 0.5


def test_tracks_cross_missing_motion_steps_through_the_pitch_mappings(
    scene: SyntheticScene,
    source: SyntheticSource,
    truth: list[FrameState],
    recorded: tuple[ClipResult, list[VLMRequest]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One motion step is missing between every two keyframes, so the motion chain alone links no two keyframes.

    The tracker never matches by raw image positions. Without the way through the pitch mappings of
    the two keyframes every track would end after one observation, and the clip would have no track.
    """
    clean_result, _ = recorded
    to_pitch = [scene.homography(frame) for frame in range(N_FRAMES)]
    steps: list[np.ndarray | None] = [
        np.linalg.inv(to_pitch[frame + 1]) @ to_pitch[frame] for frame in range(N_FRAMES - 1)
    ]
    for missing in (5, 17, 29, 41, 53, 65, 73):
        steps[missing] = None
    field = MotionField(steps)
    assert all(field.between(earlier, later) is None for earlier, later in zip(KEYFRAMES, KEYFRAMES[1:]))
    fill = Spy(monkeypatch, "dense_frame_states")

    clip_result = run_clip(source, FakeBackend(scene), motion=field)

    assert len(clip_result.tracks) == len(clean_result.tracks)
    assert identity_scores(clip_result.frames, truth).switches == 0
    assert float(np.median(position_errors_m(clip_result.frames, truth))) < 0.5
    # The dense output gets the same warp, so it moves a box across a missing step like the camera does.
    ((_, options),) = fill.calls
    points = near_half_grid()
    moved = options["warp"](0, 12, points)
    assert moved is not None
    camera_motion = apply_homography(np.linalg.inv(to_pitch[12]) @ to_pitch[0], points)
    assert float(np.max(np.linalg.norm(moved - camera_motion, axis=1))) < 5.0


def test_people_keep_their_tracks_across_missing_motion_steps_during_a_pan() -> None:
    """Five motion steps are missing while the camera pans by about 100 pixels per keyframe interval.

    A matcher that falls back to raw image positions at a missing step gives a person the track
    of a neighbour there.
    """
    long_scene = SyntheticScene(n_frames=250, seed=0)
    long_source = SyntheticSource(long_scene)
    to_pitch = [long_scene.homography(frame) for frame in range(250)]
    steps: list[np.ndarray | None] = [np.linalg.inv(to_pitch[frame + 1]) @ to_pitch[frame] for frame in range(249)]
    for missing in (30, 78, 126, 174, 222):
        steps[missing] = None

    whole = run_clip(long_source, FakeBackend(long_scene), motion=exact_motion(long_scene))
    broken = run_clip(long_source, FakeBackend(long_scene), motion=MotionField(steps))

    # No track shows two people: role, team and shirt number name one person of the scene.
    for track in broken.tracks:
        assert len({(obs.role, obs.team_key, obs.number) for obs in track.observations}) == 1
    assert len(broken.tracks) == len(whole.tracks)
    long_truth = [long_scene.frame_state(frame) for frame in range(250)]
    assert identity_scores(broken.frames, long_truth).switches == identity_scores(whole.frames, long_truth).switches


# --- Crops of the shirt-number step ------------------------------------------------------------


def test_crops_exactly_one_second_apart_are_both_taken() -> None:
    track = track_with_boxes([(0, 60.0), (25, 50.0), (49, 55.0)])
    # Frame 49 has the taller box, but it is less than one second after frame 25 and after frame 0 it is taken first.
    assert [observation.frame for observation in pipeline._crop_observations(track, 25.0, 8)] == [0, 49]
    two = track_with_boxes([(0, 60.0), (25, 50.0)])
    assert [observation.frame for observation in pipeline._crop_observations(two, 25.0, 8)] == [0, 25]


def test_at_most_the_wanted_number_of_crops_is_taken() -> None:
    track = track_with_boxes([(0, 60.0), (30, 50.0), (60, 40.0), (90, 30.0)])
    assert [observation.frame for observation in pipeline._crop_observations(track, 25.0, 2)] == [0, 30]
    assert [observation.frame for observation in pipeline._crop_observations(track, 25.0, 1)] == [0]
    assert [observation.frame for observation in pipeline._crop_observations(track, 25.0, 8)] == [0, 30, 60, 90]


def test_track_that_is_too_short_for_two_crops_one_second_apart_still_gets_two_crops() -> None:
    def crop_frames(frames_and_heights: Sequence[tuple[int, float]], max_crops: int = 8) -> list[int]:
        track = track_with_boxes(frames_and_heights)
        return [observation.frame for observation in pipeline._crop_observations(track, 25.0, max_crops)]

    # Two keyframes, 0.48 s apart: one crop can never show a number in the video, so both are taken.
    assert crop_frames([(0, 60.0), (12, 50.0)]) == [0, 12]
    # Three keyframes with the tallest box at the end: the crop that is furthest in time is added.
    assert crop_frames([(0, 40.0), (12, 50.0), (24, 60.0)]) == [0, 24]
    # The tallest box in the middle of five keyframes: no observation is one second (25 frames)
    # away from it. Frames 0 and 48 are equally far, and frame 48 has the taller box.
    assert crop_frames([(0, 40.0), (12, 45.0), (24, 60.0), (36, 45.0), (48, 50.0)]) == [24, 48]
    # With boxes of one height the earlier frame is taken.
    assert crop_frames([(0, 50.0), (12, 50.0), (24, 60.0), (36, 50.0), (48, 50.0)]) == [0, 24]
    # One crop stays one crop when the settings ask for one, and for a track with one observation.
    assert crop_frames([(0, 60.0), (12, 50.0)], max_crops=1) == [0]
    assert crop_frames([(7, 60.0)]) == [7]


def test_crop_choice_takes_frames_and_boxes_and_keeps_them_together() -> None:
    tall = Box(x1=100.0, y1=200.0, x2=140.0, y2=300.0)
    small = Box(x1=300.0, y1=250.0, x2=320.0, y2=300.0)
    middle = Box(x1=500.0, y1=220.0, x2=530.0, y2=300.0)

    chosen = pipeline.crop_choice([(60, small), (0, middle), (30, tall), (90, small)], 25.0, 3)

    # The tallest boxes first: frame 30, then frame 0, then the earlier one of the two small boxes.
    assert chosen == ((0, middle), (30, tall), (60, small))
    assert pipeline.crop_choice([], 25.0, 3) == ()


def test_person_who_is_seen_in_two_keyframes_and_comes_back_later_keeps_one_track() -> None:
    """Player 9 of team B is seen in the keyframes 12 and 24, missed for 4.3 seconds, and seen again from frame 132.

    The gap is longer than the joins of fragments allow (3 seconds). So only the shown shirt number
    gives the two fragments one id, and the first fragment needs two crops that agree for that.
    """
    long_scene = SyntheticScene(n_frames=175, seed=1)
    missed = [36, 48, 60, 72, 84, 96, 108, 120]
    backend = RecordingBackend(RewritingBackend(FakeBackend(long_scene), "people", person_missed("B", 9, missed)))

    clip_result = run_clip(SyntheticSource(long_scene), backend, motion=exact_motion(long_scene))

    nines = [track for track in clip_result.tracks if (track.team_key, track.number_export) == ("B", 9)]
    assert len(nines) == 1
    assert [observation.frame for observation in nines[0].observations][:3] == [12, 24, 132]
    assert nines[0].number_display == 9
    # The request of the first fragment showed the crops of both keyframes.
    assert [12, 24] in [request.context["frames"] for request in requests_of(backend.requests, "number")]


def test_crop_is_the_box_with_its_padding_and_rows_are_y() -> None:
    box = Box(x1=100.0, y1=200.0, x2=140.0, y2=280.0)  # 40 x 80 pixels: the padding is 6 and 12 pixels.
    rows, columns = np.meshgrid(np.arange(720), np.arange(1280), indexing="ij")
    image = np.stack([rows % 256, columns % 256, (rows + columns) % 256], axis=2).astype(np.uint8)
    crop = pipeline._cut_crop(image, box)
    np.testing.assert_array_equal(crop, image[188:292, 94:146])
    # The enlarged crop keeps the sides apart: 52 pixels wide and 104 pixels high, 3 times enlarged.
    assert pipeline._enlarge(crop, 3.0).shape == (312, 156, 3)
    with pytest.raises(ValueError, match="outside the image"):
        pipeline._cut_crop(image, Box(x1=1400.0, y1=200.0, x2=1440.0, y2=280.0))


# --- Votes and the engine table ----------------------------------------------------------------


def test_mean_pitch_x_uses_the_observations_with_a_finite_position_only() -> None:
    """The image x of every box is 110. The pitch x comes from the mapping, where a frame has one."""
    track = track_with_boxes([(0, 60.0), (12, 60.0), (24, 60.0), (36, 60.0)])

    def to_pitch(frame: int, xy: np.ndarray) -> np.ndarray | None:
        if frame == 12:
            return None  # This frame has no pitch mapping.
        if frame == 24:
            return np.full(xy.shape, np.nan)  # The foot point lies on the horizon.
        return np.full(xy.shape, -8.0 if frame == 0 else -4.0)

    assert pipeline._mean_pitch_x([track], to_pitch) == {0: -6.0}
    assert pipeline._mean_pitch_x([track], lambda frame, xy: None) == {}


def test_engine_table_names_the_model_of_each_step() -> None:
    config = PipelineConfig(
        kit_legend=StepSettings(model="gemini-4-carbon"),
        people=StepSettings(model="gemini-3.8-flash"),
        landmarks=StepSettings(model="gemini-4-hydrogen"),
        number=StepSettings(model="gemini-robotics-er-2-preview", resolution="low"),
    )
    assert pipeline._engines(config) == {
        "kit_legend": "model:gemini-4-carbon",
        "people": "model:gemini-3.8-flash",
        "landmarks": "model:gemini-4-hydrogen",
        "number": "model:gemini-robotics-er-2-preview",
        "motion": "code",
        "pitch_mapping": "code",
        "tracks": "code",
        "fill": "code",
    }


# --- Errors ------------------------------------------------------------------------------------


def test_motion_field_of_another_length_is_refused_before_any_request(
    scene: SyntheticScene, source: SyntheticSource
) -> None:
    backend = RecordingBackend(FakeBackend(scene))
    with pytest.raises(ValueError, match="motion field"):
        run_clip(source, backend, motion=MotionField.identity(N_FRAMES + 1))
    assert backend.requests == []


def test_frame_of_the_wrong_size_is_refused_before_any_request(scene: SyntheticScene) -> None:
    backend = RecordingBackend(FakeBackend(scene))
    with pytest.raises(ValueError, match="frame 0"):
        run_clip(WrongSizeSource(scene.info), backend)
    assert backend.requests == []


def test_no_request_is_sent_after_a_failed_request(
    scene: SyntheticScene, source: SyntheticSource, true_motion: MotionField
) -> None:
    """A failed request ends the run, and the requests that wait in the thread pool are not sent."""
    backend = FailingBackend(FakeBackend(scene), fail_at=4)
    with pytest.raises(BackendError, match="planned failure of request 4"):
        run_clip(source, backend, PipelineConfig(max_parallel=1), motion=true_motion)
    # With one worker the order is fixed: the kit legend, then people and landmarks keyframe by keyframe.
    sent = [(request.meta.step, request.meta.frame) for request in backend.requests]
    assert sent == [("kit_legend", None), ("people", 0), ("landmarks", 0), ("people", 12)]


def test_at_most_max_parallel_requests_are_in_flight(
    scene: SyntheticScene, source: SyntheticSource, true_motion: MotionField
) -> None:
    """Every request takes 20 milliseconds, so the pool is full most of the time."""
    lock = threading.Lock()
    in_flight: list[int] = []
    fake = FakeBackend(scene)

    class SlowBackend:
        """The fake model with a short wait. It notes the number of requests in flight at each start."""

        def __init__(self) -> None:
            self.running = 0

        def generate(self, request: VLMRequest) -> VLMResponse:
            with lock:
                self.running += 1
                in_flight.append(self.running)
            time.sleep(0.02)
            try:
                return fake.generate(request)
            finally:
                with lock:
                    self.running -= 1

    run_clip(source, SlowBackend(), PipelineConfig(max_parallel=2), motion=true_motion)
    assert max(in_flight) == 2


def test_interrupted_thread_pool_waits_for_every_request_in_flight(
    scene: SyntheticScene, recorded: tuple[ClipResult, list[VLMRequest]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interruption returns only after the requests in flight are answered, so that their answers are kept.

    The hard case is an interruption that comes while the pool starts a thread: that thread works, but
    the pool does not have it in its list and does not wait for it. The pool of this test has forgotten
    its threads when it shuts down, and the wait for the answers is interrupted after 50 milliseconds.
    """
    _, requests = recorded
    fake = FakeBackend(scene)
    answered: list[str] = []

    class SlowBackend:
        """The fake model. Every request takes 200 milliseconds."""

        def generate(self, request: VLMRequest) -> VLMResponse:
            time.sleep(0.2)
            response = fake.generate(request)
            answered.append(request.cache_key())
            return response

    class ForgetfulPool(pipeline.ThreadPoolExecutor):
        """A thread pool that does not know its threads any more when it shuts down."""

        def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
            self._threads.clear()
            super().shutdown(wait, cancel_futures=cancel_futures)

    def interrupted_wait(futures: object) -> None:
        time.sleep(0.05)
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "ThreadPoolExecutor", ForgetfulPool)
    monkeypatch.setattr(pipeline, "wait", interrupted_wait)
    two_requests = requests_of(requests, "people")[:2]

    with pytest.raises(KeyboardInterrupt):
        pipeline._generate_all(SlowBackend(), [*two_requests, *requests_of(requests, "landmarks")], 2)

    # The two requests that were in flight are answered. The other requests were never sent.
    assert sorted(answered) == sorted(request.cache_key() for request in two_requests)


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="this system cannot send a signal to one thread")
def test_second_stop_signal_does_not_end_the_wait_for_the_requests_in_flight(
    scene: SyntheticScene,
    recorded: tuple[ClipResult, list[VLMRequest]],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An operator presses Ctrl-C a second time while the command waits for the answers of the requests in flight.

    The second signal is a real one, sent to the thread of the test. It is sent after the pool began
    to wait for its threads, and the two requests in flight take 150 milliseconds more from then on.
    """
    _, requests = recorded
    fake = FakeBackend(scene)
    answered: list[str] = []
    pool_waits = threading.Event()
    second_signal_sent = threading.Event()
    waiting_thread = threading.get_ident()

    class SlowBackend:
        """The fake model. A request is answered 150 milliseconds after the second signal."""

        def generate(self, request: VLMRequest) -> VLMResponse:
            assert second_signal_sent.wait(timeout=10.0)
            time.sleep(0.15)
            response = fake.generate(request)
            answered.append(request.cache_key())
            return response

    class WatchedPool(pipeline.ThreadPoolExecutor):
        """A thread pool that says when it begins to wait for its threads."""

        def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
            pool_waits.set()
            super().shutdown(wait, cancel_futures=cancel_futures)

    def interrupted_wait(futures: object) -> None:
        raise KeyboardInterrupt

    def send_second_signal() -> None:
        assert pool_waits.wait(timeout=10.0)
        time.sleep(0.05)
        signal.pthread_kill(waiting_thread, signal.SIGINT)
        second_signal_sent.set()

    monkeypatch.setattr(pipeline, "ThreadPoolExecutor", WatchedPool)
    monkeypatch.setattr(pipeline, "wait", interrupted_wait)
    two_requests = requests_of(requests, "people")[:2]
    # A process that a shell started in the background ignores SIGINT. The test needs the handler of Python.
    earlier_handler = signal.signal(signal.SIGINT, signal.default_int_handler)
    sender = threading.Thread(target=send_second_signal)
    sender.start()

    try:
        with caplog.at_level(logging.WARNING, logger="pitchvlm.pipeline"), pytest.raises(KeyboardInterrupt):
            pipeline._generate_all(SlowBackend(), [*two_requests, *requests_of(requests, "landmarks")], 2)
        in_flight_answered = sorted(answered)
    finally:
        sender.join()
        signal.signal(signal.SIGINT, earlier_handler)

    # The function returned only after both requests in flight were answered.
    assert in_flight_answered == sorted(request.cache_key() for request in two_requests)
    assert "a stop signal came while 2 model requests were in flight" in caplog.text


def test_wait_for_the_requests_in_flight_starts_again_after_every_stop_signal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Worker:
        """A stand-in for a thread: the first two waits for it are interrupted, the third one ends it."""

        def __init__(self) -> None:
            self.joins = 0

        def is_alive(self) -> bool:
            return self.joins < 3

        def join(self) -> None:
            self.joins += 1
            if self.joins < 3:
                raise KeyboardInterrupt(f"signal {self.joins}")

    class Pool:
        """A stand-in for the thread pool that counts its shutdown calls."""

        def __init__(self) -> None:
            self.shutdowns: list[bool] = []

        def shutdown(self, *, cancel_futures: bool) -> None:
            self.shutdowns.append(cancel_futures)

    worker, pool = Worker(), Pool()
    with caplog.at_level(logging.WARNING, logger="pitchvlm.pipeline"):
        with pytest.raises(KeyboardInterrupt, match="signal 2"):
            pipeline._finish_requests_in_flight(pool, {worker})  # type: ignore[arg-type]
    assert worker.joins == 3
    assert pool.shutdowns == [True, True, True]
    assert caplog.text.count("a stop signal came while 1 model requests were in flight") == 2

    # Without a signal the wait ends without an error and without a warning.
    caplog.clear()
    done = Worker()
    done.joins = 2
    pipeline._finish_requests_in_flight(Pool(), {done})  # type: ignore[arg-type]
    assert caplog.text == ""


def test_answer_that_cannot_be_parsed_names_its_request(scene: SyntheticScene, source: SyntheticSource) -> None:
    backend = RewritingBackend(FakeBackend(scene), "landmarks", broken_answer_of_frame(12))
    with pytest.raises(ParseError, match=r"landmarks answer of clip SNGS-999 \(frame 12, .*not valid JSON"):
        run_clip(source, backend)


# --- The settings file with the defaults -------------------------------------------------------


def test_default_settings_file_gives_the_default_settings() -> None:
    assert load_config(DEFAULT_SETTINGS_FILE) == PipelineConfig()


def test_default_settings_file_writes_out_every_setting() -> None:
    written = yaml.safe_load(DEFAULT_SETTINGS_FILE.read_text(encoding="utf-8"))
    assert written == PipelineConfig().model_dump(mode="json")


def test_default_settings_file_has_one_comment_line_per_key() -> None:
    lines = DEFAULT_SETTINGS_FILE.read_text(encoding="utf-8").splitlines()
    key_lines = [index for index, line in enumerate(lines) if re.match(r" *[a-z_]+:", line)]
    assert len(key_lines) == count_keys(PipelineConfig().model_dump(mode="json"))
    for index in key_lines:
        indent = re.match(r" *", lines[index]).group()
        assert lines[index - 1].startswith(f"{indent}# "), lines[index]
        assert not lines[index - 2].lstrip().startswith("#"), lines[index]
