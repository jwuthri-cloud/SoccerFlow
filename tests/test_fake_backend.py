"""Tests of the fake model backend: answer formats, coordinate order, misses, noise and repeatability."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from typing import Any

import pytest

from pitchvlm.pitch import LANDMARKS
from pitchvlm.schemas import PlayerState
from pitchvlm.synth import SyntheticScene
from pitchvlm.vlm.fake import STEP_USAGE, FakeBackend
from pitchvlm.vlm.types import BackendError, RequestMeta, ServiceTier, TextPart, VLMRequest

#: A corner of the image that shows the ground far behind the pitch: no person is there.
EMPTY_BOX = [0.0, 0.0, 5.0, 5.0]


@pytest.fixture(scope="module")
def scene() -> SyntheticScene:
    return SyntheticScene(seed=11)


def make_request(
    scene: SyntheticScene,
    step: str,
    *,
    frame: int | None = None,
    item: str | None = None,
    repeat: int = 0,
    context: dict[str, Any] | None = None,
    service_tier: ServiceTier = "standard",
) -> VLMRequest:
    """Build a small request for one step. The fake backend reads only ``meta`` and ``context``."""
    return VLMRequest(
        model="gemini-4-argon",
        thinking_level="low",
        parts=(TextPart(text=f"test prompt of the step {step}"),),
        response_schema={"type": "object"},
        prompt_name=f"{step}/test",
        meta=RequestMeta(clip_id=scene.info.clip_id, step=step, frame=frame, item=item, repeat=repeat),
        service_tier=service_tier,
        context={} if context is None else context,
    )


def answer(backend: FakeBackend, request: VLMRequest) -> dict[str, Any]:
    """Return the answer of the backend as parsed JSON."""
    return json.loads(backend.generate(request).text)


def corners(player: PlayerState) -> list[float]:
    """Return the box of a player as ``[x1, y1, x2, y2]``, the form of the number request context."""
    assert player.box is not None
    return [player.box.x1, player.box.y1, player.box.x2, player.box.y2]


# --- Answer formats ------------------------------------------------------------------


def test_kit_legend_answer_has_the_contract_keys(scene: SyntheticScene) -> None:
    reply = answer(FakeBackend(scene), make_request(scene, "kit_legend"))
    assert set(reply) == {"team_a", "team_b", "goalkeepers", "referee_kit", "uncertain"}
    assert reply["team_a"] == {"shirt": "red", "shorts": "red", "socks": "red"}
    assert reply["team_b"] == {"shirt": "blue", "shorts": "blue", "socks": "blue"}
    assert reply["goalkeepers"] == [{"kit": "yellow", "team": "A"}, {"kit": "green", "team": "B"}]
    assert reply["referee_kit"] == "black"
    assert reply["uncertain"] == []


@pytest.mark.parametrize("frame", [0, 48, 99])
def test_people_answer_is_the_truth_in_model_coordinates(scene: SyntheticScene, frame: int) -> None:
    width, height = scene.info.width, scene.info.height
    reply = answer(FakeBackend(scene), make_request(scene, "people", frame=frame))
    assert set(reply) == {"people"}
    truth = scene.people(frame).people
    assert len(reply["people"]) == len(truth)
    for item, person in zip(reply["people"], truth, strict=True):
        assert set(item) == {"box_2d", "role", "team", "number"}
        assert (item["role"], item["team"], item["number"]) == (person.role, person.team_key, person.number)
        assert all(isinstance(value, int) and 0 <= value <= 1000 for value in item["box_2d"])
        # A model box is [ymin, xmin, ymax, xmax], normalised to 0-1000: y comes first.
        ymin, xmin, ymax, xmax = item["box_2d"]
        assert ymin == pytest.approx(1000.0 * person.box.y1 / height, abs=1.0)
        assert xmin == pytest.approx(1000.0 * person.box.x1 / width, abs=1.0)
        assert ymax == pytest.approx(1000.0 * person.box.y2 / height, abs=1.0)
        assert xmax == pytest.approx(1000.0 * person.box.x2 / width, abs=1.0)
        assert ymin < ymax and xmin < xmax


@pytest.mark.parametrize("frame", [0, 75, 99])
def test_landmarks_answer_is_the_truth_in_model_coordinates(scene: SyntheticScene, frame: int) -> None:
    width, height = scene.info.width, scene.info.height
    reply = answer(FakeBackend(scene), make_request(scene, "landmarks", frame=frame))
    assert set(reply) == {"view", "landmarks"}
    truth = scene.landmarks(frame)
    assert reply["view"] == truth.view
    assert [item["id"] for item in reply["landmarks"]] == [seen.landmark_id for seen in truth.landmarks]
    for item, seen in zip(reply["landmarks"], truth.landmarks, strict=True):
        assert set(item) == {"id", "point"}
        assert all(isinstance(value, int) and 0 <= value <= 1000 for value in item["point"])
        # A model point is [y, x], normalised to 0-1000: y comes first.
        y, x = item["point"]
        assert y == pytest.approx(1000.0 * seen.y / height, abs=1.0)
        assert x == pytest.approx(1000.0 * seen.x / width, abs=1.0)


def test_landmark_points_have_y_first_and_the_far_side_at_the_top(scene: SyntheticScene) -> None:
    """In frame 75 the camera looks straight at the halfway line."""
    reply = answer(FakeBackend(scene), make_request(scene, "landmarks", frame=75))
    assert reply["view"] == "centre"
    point_of = {item["id"]: item["point"] for item in reply["landmarks"]}
    id_of = {landmark.name: landmark.id for landmark in LANDMARKS}
    far_end = point_of[id_of["halfway line, far end"]]
    centre_spot = point_of[id_of["centre spot"]]
    near_end = point_of[id_of["halfway line, near end"]]
    # The halfway line is the middle column of the picture: x is 500 for all three points.
    assert [far_end[1], centre_spot[1], near_end[1]] == [500, 500, 500]
    # The far side is at the top of the picture: y grows from the far end to the near end.
    assert far_end[0] < centre_spot[0] < near_end[0]
    # The left penalty area is on the left of the picture, the right one on the right.
    left_corner = point_of[id_of["left penalty area, far front corner"]]
    right_corner = point_of[id_of["right penalty area, far front corner"]]
    assert left_corner[1] < 500 < right_corner[1]
    assert left_corner[0] == right_corner[0]


def test_number_answer_votes_over_the_crops(scene: SyntheticScene) -> None:
    backend = FakeBackend(scene)
    frames = [0, 1, 2]
    states = [scene.frame_state(frame) for frame in frames]
    in_all_frames = set.intersection(
        *({player.track_id for player in state.players if player.role == "player"} for state in states)
    )
    first_id, second_id = sorted(in_all_frames)[:2]
    crops = [next(player for player in state.players if player.track_id == first_id) for state in states]
    other = next(player for player in states[1].players if player.track_id == second_id)
    number, other_number = crops[0].number_export, other.number_export
    assert number is not None and other_number is not None and number != other_number
    boxes = [corners(player) for player in crops]

    def reply(crop_frames: list[int], crop_boxes: list[list[float]]) -> dict[str, Any]:
        context = {"frames": crop_frames, "boxes": crop_boxes}
        return answer(backend, make_request(scene, "number", item=f"track-{first_id}", context=context))

    three_crops = reply(frames, boxes)
    assert set(three_crops) == {"per_crop", "number", "confidence"}
    assert three_crops == {"per_crop": [number, number, number], "number": number, "confidence": "high"}
    # One crop alone is not enough for a high confidence.
    assert reply(frames[:1], boxes[:1]) == {"per_crop": [number], "number": number, "confidence": "low"}
    # A crop without a person gives null for that crop only.
    assert reply(frames, [boxes[0], EMPTY_BOX, boxes[2]]) == {
        "per_crop": [number, None, number],
        "number": number,
        "confidence": "high",
    }
    assert reply([0], [EMPTY_BOX]) == {"per_crop": [None], "number": None, "confidence": "low"}
    # Two crops that do not agree: the number seen first wins, with low confidence.
    assert reply([0, 1], [boxes[0], corners(other)]) == {
        "per_crop": [number, other_number],
        "number": number,
        "confidence": "low",
    }
    # Two against one: the most frequent number wins.
    assert reply([1, 0, 2], [corners(other), boxes[0], boxes[2]]) == {
        "per_crop": [other_number, number, number],
        "number": number,
        "confidence": "high",
    }


def test_number_answer_is_null_for_the_referee(scene: SyntheticScene) -> None:
    referee = next(player for player in scene.frame_state(60).players if player.role == "referee")
    context = {"frames": [60, 60], "boxes": [corners(referee), corners(referee)]}
    reply = answer(FakeBackend(scene), make_request(scene, "number", item="track-22", context=context))
    assert reply == {"per_crop": [None, None], "number": None, "confidence": "low"}


def test_response_has_usage_and_names_the_model_of_the_request(scene: SyntheticScene) -> None:
    backend = FakeBackend(scene)
    for step, frame in (("kit_legend", None), ("people", 0), ("landmarks", 0)):
        request = make_request(scene, step, frame=frame, service_tier="flex")
        response = backend.generate(request)
        assert response.usage == STEP_USAGE[step]
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0
        assert response.model == request.model
        assert response.service_tier == "flex"
        assert response.attempts == 1
        assert response.latency_s == 0.0
        assert datetime.fromisoformat(response.created_at).utcoffset() == timedelta(0)
    number_request = make_request(scene, "number", item="track-0", context={"frames": [0], "boxes": [EMPTY_BOX]})
    assert backend.generate(number_request).usage == STEP_USAGE["number"]
    # The token counts of a step do not depend on the frame.
    assert backend.generate(make_request(scene, "people", frame=50)).usage == STEP_USAGE["people"]


# --- Misses, noise and repeatability -------------------------------------------------


def test_miss_rate_drops_people(scene: SyntheticScene) -> None:
    exact = FakeBackend(scene)
    half = FakeBackend(scene, miss_rate=0.5, seed=3)
    blind = FakeBackend(scene, miss_rate=1.0)
    kept = total = 0
    for frame in (0, 12, 24, 36, 48):
        request = make_request(scene, "people", frame=frame)
        all_people = answer(exact, request)["people"]
        some_people = answer(half, request)["people"]
        assert all(person in all_people for person in some_people)
        kept += len(some_people)
        total += len(all_people)
        assert answer(blind, request)["people"] == []
    assert 0 < kept < total


def test_box_noise_moves_boxes_but_keeps_them_near_the_truth(scene: SyntheticScene) -> None:
    frame, noise_px = 36, 2.0
    width, height = scene.info.width, scene.info.height
    request = make_request(scene, "people", frame=frame)
    exact = answer(FakeBackend(scene), request)["people"]
    noisy = answer(FakeBackend(scene, box_noise_px=noise_px, seed=1), request)["people"]
    assert 0 < len(noisy) <= len(exact)
    assert noisy != exact
    true_centres = [person.box.center for person in scene.people(frame).people]
    for item in noisy:
        ymin, xmin, ymax, xmax = item["box_2d"]
        assert 0 <= ymin <= ymax <= 1000
        assert 0 <= xmin <= xmax <= 1000
        centre_x = (xmin + xmax) / 2000.0 * width
        centre_y = (ymin + ymax) / 2000.0 * height
        nearest = min(math.hypot(centre_x - true_x, centre_y - true_y) for true_x, true_y in true_centres)
        assert nearest < 10.0 * noise_px


def test_point_noise_moves_landmarks_but_keeps_them_near_the_truth(scene: SyntheticScene) -> None:
    frame, noise_px = 12, 3.0
    width, height = scene.info.width, scene.info.height
    request = make_request(scene, "landmarks", frame=frame)
    exact = answer(FakeBackend(scene), request)
    noisy = answer(FakeBackend(scene, point_noise_px=noise_px, seed=1), request)
    assert noisy["view"] == exact["view"]
    assert [item["id"] for item in noisy["landmarks"]] == [item["id"] for item in exact["landmarks"]]
    assert noisy["landmarks"] != exact["landmarks"]
    for item, seen in zip(noisy["landmarks"], scene.landmarks(frame).landmarks, strict=True):
        y, x = item["point"]
        assert 0 <= y <= 1000 and 0 <= x <= 1000
        assert abs(x / 1000.0 * width - seen.x) < 10.0 * noise_px
        assert abs(y / 1000.0 * height - seen.y) < 10.0 * noise_px


@pytest.mark.parametrize("step", ["people", "landmarks"])
def test_answers_are_repeatable(scene: SyntheticScene, step: str) -> None:
    def noisy_backend(seed: int) -> FakeBackend:
        return FakeBackend(scene, miss_rate=0.2, box_noise_px=3.0, point_noise_px=3.0, seed=seed)

    request = make_request(scene, step, frame=24)
    backend = noisy_backend(5)
    first = backend.generate(request).text
    # The same request gets the same answer, from the same backend and from a new one.
    assert backend.generate(request).text == first
    assert noisy_backend(5).generate(request).text == first
    # Another seed and another repeat index (another cache key) give another answer.
    assert noisy_backend(6).generate(request).text != first
    assert backend.generate(make_request(scene, step, frame=24, repeat=1)).text != first


def test_exact_answers_do_not_depend_on_the_seed(scene: SyntheticScene) -> None:
    for step in ("people", "landmarks"):
        request = make_request(scene, step, frame=24)
        assert FakeBackend(scene, seed=1).generate(request).text == FakeBackend(scene, seed=2).generate(request).text


# --- Errors ------------------------------------------------------------------------------


def test_unknown_step_raises(scene: SyntheticScene) -> None:
    with pytest.raises(BackendError):
        FakeBackend(scene).generate(make_request(scene, "ball", frame=0))


def test_requests_without_the_needed_facts_raise(scene: SyntheticScene) -> None:
    backend = FakeBackend(scene)
    with pytest.raises(BackendError):
        backend.generate(make_request(scene, "people"))
    with pytest.raises(BackendError):
        backend.generate(make_request(scene, "landmarks", frame=scene.info.n_frames))
    with pytest.raises(BackendError):
        backend.generate(make_request(scene, "number", item="track-1"))
    with pytest.raises(BackendError):
        backend.generate(make_request(scene, "number", context={"frames": [0, 1], "boxes": [EMPTY_BOX]}))
    with pytest.raises(BackendError):
        backend.generate(make_request(scene, "number", context={"frames": [1000], "boxes": [EMPTY_BOX]}))


@pytest.mark.parametrize(
    "settings",
    [{"miss_rate": 1.5}, {"miss_rate": -0.1}, {"box_noise_px": -1.0}, {"point_noise_px": -1.0}, {"seed": -1}],
)
def test_wrong_settings_raise(scene: SyntheticScene, settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        FakeBackend(scene, **settings)


# --- Together with the step parsers ------------------------------------------------------


def test_answers_go_through_the_step_parsers(scene: SyntheticScene) -> None:
    """The parsers of the real model answers read the fake answers and give the truth back."""
    from pitchvlm.vlm import steps

    backend = FakeBackend(scene)
    frame = 30
    width, height = scene.info.width, scene.info.height

    legend = steps.parse_kit_legend(backend.generate(make_request(scene, "kit_legend")).text)
    assert legend == scene.kit_legend()

    people_text = backend.generate(make_request(scene, "people", frame=frame)).text
    people = steps.parse_people(people_text, frame=frame, width=width, height=height)
    true_people = scene.people(frame)
    assert people.frame == frame
    assert len(people.people) == len(true_people.people)
    for person, true_person in zip(people.people, true_people.people, strict=True):
        assert (person.role, person.team_key) == (true_person.role, true_person.team_key)
        assert person.number == true_person.number
        # One model unit is 1.28 pixels wide and 0.72 pixels high in a 1280 x 720 image.
        assert person.box.x1 == pytest.approx(true_person.box.x1, abs=1.0)
        assert person.box.y1 == pytest.approx(true_person.box.y1, abs=1.0)
        assert person.box.x2 == pytest.approx(true_person.box.x2, abs=1.0)
        assert person.box.y2 == pytest.approx(true_person.box.y2, abs=1.0)

    landmarks_text = backend.generate(make_request(scene, "landmarks", frame=frame)).text
    landmarks = steps.parse_landmarks(landmarks_text, frame=frame, width=width, height=height)
    true_landmarks = scene.landmarks(frame)
    assert landmarks.view == true_landmarks.view
    assert [seen.landmark_id for seen in landmarks.landmarks] == [seen.landmark_id for seen in true_landmarks.landmarks]
    for seen, true_seen in zip(landmarks.landmarks, true_landmarks.landmarks, strict=True):
        assert seen.x == pytest.approx(true_seen.x, abs=1.0)
        assert seen.y == pytest.approx(true_seen.y, abs=1.0)

    keeper = next(player for player in scene.frame_state(0).players if player.track_id == 0)
    context = {"frames": [0, 0], "boxes": [corners(keeper), corners(keeper)]}
    number_text = backend.generate(make_request(scene, "number", item="track-0", context=context)).text
    reading = steps.parse_number(number_text)
    assert (reading.per_crop, reading.number, reading.confidence) == ((1, 1), 1, "high")
