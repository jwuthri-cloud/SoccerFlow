"""Tests of the probes: frame choice, the three probes and the metrics file.

Two kinds of clips are used. The synthetic clip (``SyntheticScene``) goes through a label file and
the fake model. The hand-made clips have labels that are built in the test and a stub model with
fixed answers, so that every expected number can be checked by hand.

The camera of the hand-made clips is simple: pixel x = 640 + 10 * pitch x and
pixel y = 360 + 8 * pitch y, in an image of 1280 x 720 pixels. So the right goal is on the right
of the picture, and the far side (negative pitch y) is at the top. The boxes of the people tests
have corners at whole model units (1.28 pixels in x, 0.72 pixels in y), so a stub answer in model
coordinates gives exactly the labelled box.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Collection, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

import pitchvlm
from pitchvlm import probes
from pitchvlm.config import StepSettings
from pitchvlm.evalx.oracle import LABEL_RANSAC_PX, MIN_ATHLETES_FOR_FIT
from pitchvlm.geometry import PaintAlignment, fit_from_landmarks, matrix_to_rows
from pitchvlm.labels import GsrAthlete, GsrFrame, GsrLabels, gsr_frame_people, load_gsr_labels
from pitchvlm.pitch import LANDMARK_BY_ID
from pitchvlm.probes import (
    LabelledClip,
    probe_frames,
    probe_number,
    probe_people,
    probe_pitch,
    sample_frames,
    score_people_answers,
    score_pitch_mappings,
    write_metrics,
)
from pitchvlm.schemas import Box, ClipInfo, FramePeople, HomographyRecord, PersonDet
from pitchvlm.synth import SyntheticScene, SyntheticSource
from pitchvlm.vlm import steps
from pitchvlm.vlm.cache import CachedBackend, CassetteStore
from pitchvlm.vlm.fake import FakeBackend
from pitchvlm.vlm.steps import ParseError
from pitchvlm.vlm.types import ImagePart, Usage, VLMRequest, VLMResponse

WIDTH, HEIGHT, FPS = 1280, 720, 25.0
SETTINGS = StepSettings()
#: Width of every box of the hand-made people and pitch tests, and the two box heights, pixels.
BOX_WIDTH = 64.0
TALL, SHORT = 144.0, 72.0
#: Boxes up to this height count as small in the hand-made people tests: the short boxes only.
SMALL_BOX_MAX_HEIGHT_PX = 100.0
#: Boxes up to this height count as small in the synthetic clip: about two thirds of its boxes.
SYNTHETIC_SMALL_BOX_PX = 40.0
LEGEND_ANSWER: dict[str, Any] = {
    "team_a": {"shirt": "white", "shorts": "white", "socks": "white"},
    "team_b": {"shirt": "red", "shorts": "black", "socks": "red"},
    "goalkeepers": [{"kit": "green", "team": "A"}],
    "referee_kit": "yellow",
    "uncertain": [],
}
#: The four corners of the pitch and the four front corners of the penalty areas. No three of
#: these marks lie on one line, so every four of them fix the pitch mapping.
LANDMARK_IDS = (1, 8, 28, 35, 12, 15, 21, 24)
#: Pitch positions of six people. Mean |x| is 34.7 m: a goal view.
GOAL_SPOTS = ((-48.0, -18.0), (-32.0, 9.0), (-48.0, 18.0), (-32.0, -27.0), (-16.0, 0.0), (-32.0, 27.0))
#: Pitch positions of six people. Every |x| is 16 m: a midfield view.
MIDFIELD_SPOTS = ((-16.0, -18.0), (16.0, 9.0), (-16.0, 18.0), (16.0, -27.0), (-16.0, 0.0), (16.0, 27.0))
#: Pitch positions of two people. Mean |x| is exactly 20 m, which is not above 20 m: a midfield view.
BORDER_SPOTS = ((16.0, 0.0), (24.0, 9.0))

Answer = Callable[[VLMRequest], Any]


# --- Hand-made clips -------------------------------------------------------------------


def to_pixels(pitch_xy: tuple[float, float]) -> tuple[float, float]:
    """The camera of the hand-made clips: pitch metres to image pixels."""
    x, y = pitch_xy
    return 640.0 + 10.0 * x, 360.0 + 8.0 * y


def model_point(x_px: float, y_px: float) -> list[int]:
    """Return an image point as a model point: [y, x], each from 0 to 1000."""
    return [round(1000.0 * y_px / HEIGHT), round(1000.0 * x_px / WIDTH)]


def model_box(box: Box) -> list[int]:
    """Return an image box as a model box: [ymin, xmin, ymax, xmax], each from 0 to 1000."""
    return [
        round(1000.0 * box.y1 / HEIGHT),
        round(1000.0 * box.x1 / WIDTH),
        round(1000.0 * box.y2 / HEIGHT),
        round(1000.0 * box.x2 / WIDTH),
    ]


def standing(
    track_id: int,
    pitch_xy: tuple[float, float],
    *,
    role: str = "player",
    team: str | None = "left",
    jersey: int | None = None,
    height_px: float = SHORT,
    pitch_known: bool = True,
) -> GsrAthlete:
    """Return a labelled person who stands at a pitch position. The foot point is the bottom centre of the box."""
    foot_x, foot_y = to_pixels(pitch_xy)
    box = Box(x1=foot_x - BOX_WIDTH / 2.0, y1=foot_y - height_px, x2=foot_x + BOX_WIDTH / 2.0, y2=foot_y)
    return GsrAthlete(
        track_id=track_id,
        box=box,
        pitch_x=pitch_xy[0] if pitch_known else None,
        pitch_y=pitch_xy[1] if pitch_known else None,
        role=role,
        team=team,
        jersey=jersey,
    )


def people_at(spots: Sequence[tuple[float, float]]) -> list[GsrAthlete]:
    """Return one labelled player at each pitch position. The track ids are 1, 2, 3, ..."""
    return [standing(index + 1, spot) for index, spot in enumerate(spots)]


def box_of(athlete: GsrAthlete) -> Box:
    """Return the box of a labelled person who has one."""
    assert athlete.box is not None
    return athlete.box


def make_labels(
    clip_id: str, athletes_by_frame: Sequence[Sequence[GsrAthlete]], *, unscored: Collection[int] = ()
) -> GsrLabels:
    """Build the labels of a clip: one frame per entry. The frames in ``unscored`` are not scored."""
    frames = tuple(
        GsrFrame(
            frame=index,
            image_id=f"9{index + 1:06d}",
            file_name=f"{index + 1:06d}.jpg",
            scored=index not in unscored,
            athletes=tuple(athletes),
            lines={},
        )
        for index, athletes in enumerate(athletes_by_frame)
    )
    return GsrLabels(
        clip_id=clip_id,
        fps=FPS,
        n_frames=len(frames),
        width=WIDTH,
        height=HEIGHT,
        frames=frames,
        jersey_is_str=True,
        version=None,
    )


class PaintedSource:
    """The frames of a hand-made clip. A frame is black, and every labelled box of the frame is
    filled with the grey level ``100 + frame``.
    """

    def __init__(self, labels: GsrLabels) -> None:
        self._labels = labels
        self.info = ClipInfo(
            clip_id=labels.clip_id,
            n_frames=len(labels.frames),
            fps=labels.fps,
            width=labels.width,
            height=labels.height,
        )

    def read(self, frame: int) -> np.ndarray:
        """Return one frame as a BGR uint8 image of shape (height, width, 3)."""
        image = np.zeros((self.info.height, self.info.width, 3), dtype=np.uint8)
        for athlete in self._labels.frames[frame].athletes:
            if athlete.box is not None:
                box = athlete.box
                # Rows are y and columns are x.
                image[round(box.y1) : round(box.y2), round(box.x1) : round(box.x2)] = 100 + frame
        return image


def hand_clip(
    clip_id: str, athletes_by_frame: Sequence[Sequence[GsrAthlete]], *, unscored: Collection[int] = ()
) -> LabelledClip:
    """Build a hand-made clip: its labels and its painted frames."""
    labels = make_labels(clip_id, athletes_by_frame, unscored=unscored)
    return LabelledClip(PaintedSource(labels), labels)


class StubBackend:
    """A model backend with fixed answers. It keeps every request that it gets.

    ``answers`` maps a step to a function of the request. The function returns the answer as JSON
    data, or as text when the answer must not be valid JSON. A step without a function raises
    ``KeyError``, so a test sees a request that it does not expect.
    """

    def __init__(self, answers: dict[str, Answer]) -> None:
        self._answers = answers
        self.requests: list[VLMRequest] = []

    def generate(self, request: VLMRequest) -> VLMResponse:
        """Keep the request and return the fixed answer of its step."""
        self.requests.append(request)
        answer = self._answers[request.meta.step](request)
        return VLMResponse(
            text=answer if isinstance(answer, str) else json.dumps(answer),
            usage=Usage(input_tokens=10, output_tokens=10),
            model=request.model,
            latency_s=0.0,
            created_at="2026-11-13T12:00:00+00:00",
        )

    def sent(self, step: str) -> list[VLMRequest]:
        """Return the requests of one step, in the order in which they came."""
        return [request for request in self.requests if request.meta.step == step]


def legend_answer(request: VLMRequest) -> dict[str, Any]:
    """Answer a kit legend request."""
    return LEGEND_ANSWER


def assert_plain(value: Any) -> None:
    """Fail when a value is not made of plain JSON types only: no numpy numbers, no tuples, no records."""
    if isinstance(value, dict):
        for key, item in value.items():
            assert type(key) is str, key
            assert_plain(item)
    elif isinstance(value, list):
        for item in value:
            assert_plain(item)
    else:
        assert value is None or type(value) in (str, bool, int, float), repr(value)


def load_strict(path: Path) -> Any:
    """Read a JSON file. A value that JSON does not have (NaN, Infinity) fails the test."""

    def refuse(token: str) -> None:
        raise AssertionError(f"{token} is not valid JSON")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=refuse)


# --- Hand-made clip of the people tests ------------------------------------------------


def people_layout(*, pitch_known_for_all: bool = True) -> list[GsrAthlete]:
    """Seven labelled people: tracks 1 to 4 have tall boxes (144 px), tracks 5 to 7 have short boxes (72 px)."""
    return [
        standing(1, (-32.0, -18.0), jersey=4, height_px=TALL),
        standing(2, (-16.0, 9.0), team="right", jersey=9, height_px=TALL),
        standing(3, (0.0, -9.0), height_px=TALL),
        standing(4, (16.0, 18.0), team="right", height_px=TALL),
        standing(5, (32.0, -27.0), jersey=7),
        standing(6, (48.0, 0.0), role="goalkeeper", team="right", jersey=1, pitch_known=pitch_known_for_all),
        standing(7, (0.0, 27.0), role="referee", team=None, pitch_known=pitch_known_for_all),
    ]


def people_clip(clip_id: str = "HAND-1") -> LabelledClip:
    """A clip of four frames with the same seven people.

    In frame 2 only five people have a pitch position, so the labels give no pitch mapping there.
    Frame 3 is not scored.
    """
    frames = [people_layout(), people_layout(), people_layout(pitch_known_for_all=False), people_layout()]
    return hand_clip(clip_id, frames, unscored=[3])


def exact_boxes() -> list[list[int]]:
    """Return the seven labelled boxes of the people layout as model boxes."""
    return [model_box(box_of(athlete)) for athlete in people_layout()]


def shifted(box_2d: list[int], x_units: int) -> list[int]:
    """Move a model box [ymin, xmin, ymax, xmax] to the right by whole model units."""
    ymin, xmin, ymax, xmax = box_2d
    return [ymin, xmin + x_units, ymax, xmax + x_units]


def people_answer(boxes: Sequence[list[int]]) -> dict[str, Any]:
    """Return a people answer with one player per model box."""
    return {"people": [{"box_2d": box, "role": "player", "team": "A", "number": None} for box in boxes]}


def people_answer_of_frame(frame: int) -> dict[str, Any]:
    """Return the people answer of one frame of the people clip. Frames 0 and 2 get all seven boxes.
    Frame 1 gets a faulty answer:

    - tracks 1 to 4: the exact box;
    - track 5: no box;
    - track 6: the box 10 model units (12.8 pixels) to the right, IoU 51.2 / 76.8 = 0.67;
    - track 7: the box 50 model units (64 pixels, its own width) to the right, IoU 0;
    - two more boxes where nobody is.

    So the answer has 3 wrong boxes and 2 missed people: the recall and the precision differ.
    """
    exact = exact_boxes()
    if frame != 1:
        return people_answer(exact)
    nobody = [[800, 75, 1000, 125], [800, 200, 1000, 250]]
    return people_answer([*exact[:4], shifted(exact[5], 10), shifted(exact[6], 50), *nobody])


def people_answers(request: VLMRequest) -> dict[str, Any]:
    """Answer a people request of the people clip: see ``people_answer_of_frame``."""
    return people_answer_of_frame(request.meta.frame)


def people_backend() -> StubBackend:
    """Return a stub model for the people clip."""
    return StubBackend({"kit_legend": legend_answer, "people": people_answers})


def run_people_probe(clips: Sequence[LabelledClip], backend: StubBackend, **changes: Any) -> dict[str, Any]:
    """Run the people probe on hand-made clips with all frames (10 is more than a clip has)."""
    arguments: dict[str, Any] = {
        "frames_per_clip": 10,
        "seed": 0,
        "small_box_max_height_px": SMALL_BOX_MAX_HEIGHT_PX,
        **changes,
    }
    return probe_people(clips, backend, SETTINGS, **arguments)


# --- LabelledClip and sample_frames ----------------------------------------------------


def test_labelled_clip_needs_frames_and_labels_of_the_same_clip() -> None:
    labels = make_labels("HAND-1", [[], []])
    source = PaintedSource(labels)
    clip = LabelledClip(source, labels)
    assert clip.source is source
    assert clip.labels is labels
    with pytest.raises(AttributeError):
        clip.labels = labels  # type: ignore[misc]
    with pytest.raises(ValueError, match="clip id 'HAND-1' against 'HAND-2'"):
        LabelledClip(source, labels.model_copy(update={"clip_id": "HAND-2"}))
    with pytest.raises(ValueError, match="2 frames against 3 labelled images"):
        LabelledClip(source, make_labels("HAND-1", [[], [], []]))
    with pytest.raises(ValueError, match="1280 x 720 pixels against 1920 x 1080 pixels"):
        LabelledClip(source, labels.model_copy(update={"width": 1920, "height": 1080}))


def test_sample_frames_are_evenly_spread() -> None:
    labels = make_labels("HAND-1", [[] for _ in range(100)])
    frames = sample_frames(labels, 5, seed=3)
    assert len(frames) == 5
    assert all(0 <= frame < 100 for frame in frames)
    # 100 frames in 5 parts of 20 frames: one frame at the same place in every part.
    assert frames[0] < 20
    assert [later - earlier for earlier, later in zip(frames, frames[1:])] == [20, 20, 20, 20]
    # 7 does not divide 100: the frames are 14 or 15 frames apart (100 / 7 = 14.3).
    seven = sample_frames(labels, 7, seed=3)
    assert len(seven) == 7
    assert seven[0] < 15
    assert {later - earlier for earlier, later in zip(seven, seven[1:])} <= {14, 15}


def test_sample_frames_take_scored_frames_only() -> None:
    labels = make_labels("HAND-1", [[] for _ in range(60)], unscored=range(1, 60, 2))
    frames = sample_frames(labels, 6, seed=0)
    assert len(frames) == 6
    assert all(frame % 2 == 0 for frame in frames)
    # The 30 scored frames are the even frames. 6 parts of 5 scored frames are 10 frames long.
    assert [later - earlier for earlier, later in zip(frames, frames[1:])] == [10, 10, 10, 10, 10]
    # A clip with at most as many scored frames as asked gives all of them, in frame order.
    assert sample_frames(labels, 30, seed=0) == list(range(0, 60, 2))
    assert sample_frames(labels, 31, seed=0) == list(range(0, 60, 2))


def test_sample_frames_follow_the_seed_and_the_clip_id() -> None:
    labels = make_labels("HAND-1", [[] for _ in range(100)])
    assert sample_frames(labels, 5, seed=7) == sample_frames(labels, 5, seed=7)
    by_seed = {tuple(sample_frames(labels, 5, seed=seed)) for seed in range(20)}
    assert len(by_seed) > 1
    by_clip = {
        tuple(sample_frames(make_labels(f"HAND-{index}", [[] for _ in range(100)]), 5, seed=7)) for index in range(20)
    }
    assert len(by_clip) > 1


def test_sample_frames_refuse_wrong_input() -> None:
    labels = make_labels("HAND-1", [[], [], []])
    with pytest.raises(ValueError, match="per_clip"):
        sample_frames(labels, 0, seed=0)
    with pytest.raises(ValueError, match="no scored frame"):
        sample_frames(make_labels("HAND-1", [[], []], unscored=[0, 1]), 1, seed=0)


# --- People probe on hand-made clips ---------------------------------------------------


def test_probe_people_counts_found_wrong_and_missed_boxes() -> None:
    backend = people_backend()
    result = run_people_probe([people_clip()], backend)

    assert (result["probe"], result["n_clips"], result["n_frames"]) == ("people", 1, 3)
    # Frames 0 and 2: 7 of 7 found. Frame 1: tracks 1, 2, 3, 4 and 6 found, tracks 5 and 7 missed,
    # and three boxes that match nobody.
    counts = result["counts"]
    assert counts == {"predicted_boxes": 22, "true_boxes": 21, "tp": 19, "fp": 3, "fn": 2, "fp_off_pitch": 2}
    assert result["recall"] == pytest.approx(19 / 21)
    assert result["precision"] == pytest.approx(19 / 22)
    # Small boxes are the three short boxes of each frame. In frame 1 only track 6 is found.
    small = result["small_boxes"]
    assert (small["true_boxes"], small["tp"], small["fp"], small["fn"]) == (9, 7, 3, 2)
    assert small["recall"] == pytest.approx(7 / 9)
    assert small["precision"] == pytest.approx(7 / 10)
    assert result["n_answers_not_parsed"] == 0
    assert_plain(result)


def test_wrong_boxes_off_the_pitch_are_counted_next_to_the_wrong_boxes() -> None:
    """Every frame gets the seven right boxes and wrong boxes of 64 x 72 pixels.

    - frame 0: one box with its foot at the pitch position (16, -18), where nobody is, and one
      box with its foot at (0, 45), 11 m behind the near touchline;
    - frame 1: one box with its foot at (-60, 0), 7.5 m behind the left goal line, and one box
      with its foot at (0, 36.9), 2.9 m behind the near touchline: inside the margin of 3 m;
    - frame 2: one box with its foot at (0, 45). The labels of this frame give no mapping.
    """

    def wrong_box(pitch_xy: tuple[float, float]) -> list[int]:
        foot_x, foot_y = to_pixels(pitch_xy)
        return model_box(Box(x1=foot_x - 32.0, y1=foot_y - 72.0, x2=foot_x + 32.0, y2=foot_y))

    wrong = {0: [(16.0, -18.0), (0.0, 45.0)], 1: [(-60.0, 0.0), (0.0, 36.9)], 2: [(0.0, 45.0)]}

    def answers(request: VLMRequest) -> dict[str, Any]:
        return people_answer([*exact_boxes(), *(wrong_box(spot) for spot in wrong[request.meta.frame])])

    def people_of(clip_id: str, frame: int) -> FramePeople:
        boxes = [*exact_boxes(), *(wrong_box(spot) for spot in wrong[frame])]
        return steps.parse_people(json.dumps(people_answer(boxes)), frame=frame, width=WIDTH, height=HEIGHT, coords="yx_1000")

    model = run_people_probe([people_clip()], StubBackend({"kit_legend": legend_answer, "people": answers}))
    tool = score_people([people_clip()], people_of=people_of)

    for result in (model, tool):
        counts = result["counts"]
        # Five wrong boxes. Two of them are off the pitch in a frame whose labels give a mapping.
        assert (counts["tp"], counts["fp"], counts["fp_off_pitch"]) == (21, 5, 2)
        assert result["n_frames_without_label_mapping"] == 1
        # The precision counts every wrong box.
        assert result["precision"] == pytest.approx(21 / 26)


def test_right_box_off_the_pitch_is_not_counted_as_a_wrong_box_off_the_pitch() -> None:
    """A labelled person stands 11 m behind the near touchline, and the tool finds that person.

    The tool also gives one box where nobody is, 7.5 m behind the right goal line. Only that
    box is a wrong box off the pitch.
    """
    layout = [*people_layout(), standing(8, (0.0, 45.0), role="other", team=None)]
    clip = hand_clip("HAND-8", [layout, layout])
    foot_x, foot_y = to_pixels((60.0, 0.0))
    nobody = model_box(Box(x1=foot_x - 32.0, y1=foot_y - 72.0, x2=foot_x + 32.0, y2=foot_y))
    boxes = [*(model_box(box_of(athlete)) for athlete in layout), nobody]

    def people_of(clip_id: str, frame: int) -> FramePeople:
        return steps.parse_people(json.dumps(people_answer(boxes)), frame=frame, width=WIDTH, height=HEIGHT, coords="yx_1000")

    result = score_people([clip], people_of=people_of)

    assert result["n_frames_without_label_mapping"] == 0
    counts = result["counts"]
    assert (counts["true_boxes"], counts["tp"], counts["fp"], counts["fp_off_pitch"]) == (16, 16, 2, 2)


def test_probe_people_measures_foot_errors_through_the_label_mapping() -> None:
    result = run_people_probe([people_clip()], people_backend())
    foot = result["foot_error_m"]
    # Frames 0 and 1 have a label mapping: 7 + 5 matched boxes. The labels of frame 2 give no
    # mapping, so its 7 matched boxes are left out.
    assert foot["n"] == 12
    assert foot["n_not_finite"] == 0
    assert result["n_frames_without_label_mapping"] == 1
    # Eleven boxes are exact. The box of track 6 in frame 1 is 12.8 pixels to the right, and
    # 10 pixels are 1 m of pitch x: an error of 1.28 m.
    assert foot["median"] == pytest.approx(0.0, abs=1e-4)
    assert foot["mean"] == pytest.approx(1.28 / 12, abs=1e-4)
    assert result["per_clip"]["HAND-1"]["foot_error_median_m"] == pytest.approx(0.0, abs=1e-4)


def test_probe_people_reports_settings_prompts_and_the_main_metric() -> None:
    backend = people_backend()
    result = run_people_probe([people_clip()], backend)
    assert result["settings"] == {
        "step_settings": SETTINGS.model_dump(),
        "seed": 0,
        "repeat": 0,
        "frames_per_clip": 10,
        "iou_threshold": 0.5,
        "small_box_max_height_px": SMALL_BOX_MAX_HEIGHT_PX,
        "label_ransac_px": LABEL_RANSAC_PX,
        "label_min_athletes": MIN_ATHLETES_FOR_FIT,
    }
    assert set(result["prompts"]) == {"kit_legend", "people"}
    assert result["prompts"] == {request.meta.step: request.prompt_name for request in backend.requests}
    # The main metric of the people probe is the recall (19 of 21), not the precision (19 of 22).
    main = result["main_metric"]
    assert (main["name"], main["n_clips"], main["confidence"]) == ("recall", 1, 0.95)
    assert main["mean"] == pytest.approx(19 / 21)
    # One clip gives no interval.
    assert math.isnan(main["low"])
    assert math.isnan(main["high"])
    per_clip = result["per_clip"]
    assert set(per_clip) == {"HAND-1"}
    assert per_clip["HAND-1"]["n_frames"] == 3
    assert per_clip["HAND-1"]["recall"] == pytest.approx(19 / 21)
    assert per_clip["HAND-1"]["precision"] == pytest.approx(19 / 22)
    assert per_clip["HAND-1"]["small_box_recall"] == pytest.approx(7 / 9)


def test_probe_people_asks_the_legend_once_and_the_people_of_scored_frames() -> None:
    backend = people_backend()
    run_people_probe([people_clip()], backend)
    assert [request.meta.step for request in backend.requests] == ["kit_legend", "people", "people", "people"]
    legend_request = backend.requests[0]
    # The first, the middle and the last scored frame go into the kit legend request.
    assert len(legend_request.image_hashes) == 3
    assert legend_request.meta.frame is None
    people_requests = backend.sent("people")
    # Frame 3 is not scored, so it is never sent.
    assert [request.meta.frame for request in people_requests] == [0, 1, 2]
    assert all(len(request.image_hashes) == 1 for request in people_requests)
    assert all(request.meta.clip_id == "HAND-1" and request.meta.repeat == 0 for request in backend.requests)


def test_probe_people_sends_the_given_settings_and_repeat() -> None:
    other = StepSettings(model="gemini-3.8-flash", thinking_level="medium", resolution="medium", service_tier="flex")
    backend = people_backend()
    result = probe_people(
        [people_clip()], backend, other, frames_per_clip=2, seed=5, small_box_max_height_px=90.0, repeat=2
    )
    assert len(backend.sent("people")) == 2
    for request in backend.requests:
        assert (request.model, request.thinking_level, request.service_tier) == ("gemini-3.8-flash", "medium", "flex")
        assert set(request.resolutions) == {"medium"}
        assert request.meta.repeat == 2
    assert result["n_frames"] == 2
    assert result["settings"]["step_settings"] == other.model_dump()
    assert (result["settings"]["seed"], result["settings"]["repeat"]) == (5, 2)
    assert (result["settings"]["frames_per_clip"], result["settings"]["small_box_max_height_px"]) == (2, 90.0)


def test_probe_people_reports_every_clip() -> None:
    backend = people_backend()
    result = run_people_probe([people_clip("HAND-1"), people_clip("HAND-2")], backend)
    assert (result["n_clips"], result["n_frames"]) == (2, 6)
    counts = result["counts"]
    assert counts == {"predicted_boxes": 44, "true_boxes": 42, "tp": 38, "fp": 6, "fn": 4, "fp_off_pitch": 4}
    assert result["foot_error_m"]["n"] == 24
    assert result["n_frames_without_label_mapping"] == 2
    assert set(result["per_clip"]) == {"HAND-1", "HAND-2"}
    assert all(values["recall"] == pytest.approx(19 / 21) for values in result["per_clip"].values())
    # Both clips have the same recall, so the interval of the mean has no width.
    main = result["main_metric"]
    assert main["n_clips"] == 2
    assert main["mean"] == pytest.approx(19 / 21)
    assert main["low"] == pytest.approx(19 / 21)
    assert main["high"] == pytest.approx(19 / 21)
    # Each clip gets its own kit legend, asked before the people of that clip.
    assert [(request.meta.clip_id, request.meta.step) for request in backend.requests] == [
        ("HAND-1", "kit_legend"),
        ("HAND-1", "people"),
        ("HAND-1", "people"),
        ("HAND-1", "people"),
        ("HAND-2", "kit_legend"),
        ("HAND-2", "people"),
        ("HAND-2", "people"),
        ("HAND-2", "people"),
    ]
    assert_plain(result)


def test_main_metric_of_the_people_probe_is_the_recall_also_when_every_person_is_found() -> None:
    """Every labelled box is answered, and each answer has one more box where nobody is."""

    def answers(request: VLMRequest) -> dict[str, Any]:
        return people_answer([*exact_boxes(), [800, 75, 1000, 125]])

    result = run_people_probe([people_clip()], StubBackend({"kit_legend": legend_answer, "people": answers}))
    assert result["recall"] == 1.0
    assert result["precision"] == pytest.approx(21 / 24)
    assert result["main_metric"]["mean"] == 1.0
    assert result["per_clip"]["HAND-1"]["recall"] == 1.0
    assert result["per_clip"]["HAND-1"]["precision"] == pytest.approx(21 / 24)


def test_people_answer_that_cannot_be_parsed_counts_as_no_people() -> None:
    def answers(request: VLMRequest) -> Any:
        return "this is not JSON" if request.meta.frame == 1 else people_answer(exact_boxes())

    backend = StubBackend({"kit_legend": legend_answer, "people": answers})
    result = run_people_probe([people_clip()], backend)
    assert result["n_answers_not_parsed"] == 1
    # The seven people of frame 1 are missed. Nothing is found wrongly.
    counts = result["counts"]
    assert counts == {"predicted_boxes": 14, "true_boxes": 21, "tp": 14, "fp": 0, "fn": 7, "fp_off_pitch": 0}
    assert result["recall"] == pytest.approx(14 / 21)
    assert result["precision"] == 1.0
    # Only frame 0 has matched boxes and a label mapping.
    assert result["foot_error_m"]["n"] == 7
    assert len(backend.sent("people")) == 3


def test_unparsed_people_answers_of_every_clip_are_counted() -> None:
    """Frame 1 of the first clip cannot be parsed. The second clip has no such answer."""

    def answers(request: VLMRequest) -> Any:
        if (request.meta.clip_id, request.meta.frame) == ("HAND-1", 1):
            return "this is not JSON"
        return people_answer(exact_boxes())

    backend = StubBackend({"kit_legend": legend_answer, "people": answers})
    result = run_people_probe([people_clip("HAND-1"), people_clip("HAND-2")], backend)

    assert result["n_answers_not_parsed"] == 1


def test_kit_legend_that_cannot_be_parsed_stops_the_probe() -> None:
    backend = StubBackend({"kit_legend": lambda request: "this is not JSON", "people": people_answers})
    with pytest.raises(ParseError, match="kit_legend answer of clip HAND-1"):
        run_people_probe([people_clip()], backend)
    assert [request.meta.step for request in backend.requests] == ["kit_legend"]


@pytest.mark.parametrize(
    "changes",
    [
        {"frames_per_clip": 0},
        {"small_box_max_height_px": 0.0},
        {"small_box_max_height_px": math.nan},
        {"repeat": -1},
    ],
)
def test_probe_people_refuses_wrong_settings_before_any_request(changes: dict[str, Any]) -> None:
    backend = people_backend()
    with pytest.raises(ValueError):
        run_people_probe([people_clip()], backend, **changes)
    assert backend.requests == []


def test_probes_refuse_no_clip_and_two_clips_with_one_id() -> None:
    backend = StubBackend({})
    clip = people_clip()
    with pytest.raises(ValueError, match="at least one clip"):
        run_people_probe([], backend)
    with pytest.raises(ValueError, match="HAND-1"):
        run_people_probe([clip, people_clip()], backend)
    with pytest.raises(ValueError, match="HAND-1"):
        probe_pitch([clip, people_clip()], backend, SETTINGS, frames_per_clip=1, seed=0, ransac_px=8.0)
    with pytest.raises(ValueError, match="HAND-1"):
        probe_number(
            [clip, people_clip()], backend, SETTINGS, tracks_per_clip=1, crops_per_track=1, crop_upscale=2.0, seed=0
        )
    assert backend.requests == []


def test_labels_of_all_clips_are_read_before_the_first_request() -> None:
    """A clip without a scored frame is the last clip. The probe stops before it asks anything."""
    no_scored_frame = hand_clip("HAND-2", [[], []], unscored=[0, 1])
    backend = StubBackend({})
    with pytest.raises(ValueError, match="no scored frame"):
        run_people_probe([people_clip(), no_scored_frame], backend)
    with pytest.raises(ValueError, match="no scored frame"):
        probe_pitch([people_clip(), no_scored_frame], backend, SETTINGS, frames_per_clip=2, seed=0, ransac_px=8.0)
    assert backend.requests == []


# --- Pitch probe on hand-made clips ----------------------------------------------------


def landmark_answer(
    ids: Sequence[int], *, mirrored: bool = False, moved: dict[int, float] | None = None
) -> dict[str, Any]:
    """Point at landmarks as the camera of the hand-made clips sees them.

    ``mirrored`` swaps left and right in the picture. ``moved`` maps a landmark id to a shift of
    its point in pixels along x.
    """
    landmarks = []
    for landmark_id in ids:
        landmark = LANDMARK_BY_ID[landmark_id]
        x_px, y_px = to_pixels((-landmark.x if mirrored else landmark.x, landmark.y))
        x_px += (moved or {}).get(landmark_id, 0.0)
        landmarks.append({"id": landmark_id, "point": model_point(x_px, y_px)})
    return {"view": "centre", "landmarks": landmarks}


def pitch_answers(request: VLMRequest) -> Any:
    """Answer a landmarks request of the pitch clip: see ``pitch_clip``."""
    return pitch_answer_of_frame(request.meta.frame)


def pitch_answer_of_frame(frame: int) -> Any:
    """Return the landmarks answer of one frame of the pitch clip: see ``pitch_clip``."""
    if frame == 2:
        return landmark_answer(LANDMARK_IDS, mirrored=True)
    if frame == 3:
        return landmark_answer(LANDMARK_IDS[:3])
    if frame == 4:
        return "this is not JSON"
    return landmark_answer(LANDMARK_IDS)


def pitch_clip() -> LabelledClip:
    """A clip of five frames for the pitch probe.

    - frame 0: goal view, right landmarks;
    - frame 1: midfield view, right landmarks;
    - frame 2: midfield view, landmarks mirrored (left and right swapped);
    - frame 3: goal view, three landmarks only: no mapping;
    - frame 4: midfield view (mean |x| exactly 20 m), an answer that cannot be parsed.
    """
    frames = [
        people_at(GOAL_SPOTS),
        people_at(MIDFIELD_SPOTS),
        people_at(MIDFIELD_SPOTS),
        people_at(GOAL_SPOTS),
        people_at(BORDER_SPOTS),
    ]
    return hand_clip("HAND-1", frames)


def run_pitch_probe(clip: LabelledClip, backend: StubBackend, ransac_px: float = 8.0) -> dict[str, Any]:
    """Run the pitch probe on a hand-made clip with all its frames (10 is more than it has)."""
    return probe_pitch([clip], backend, SETTINGS, frames_per_clip=10, seed=0, ransac_px=ransac_px)


def test_probe_pitch_counts_valid_and_plausible_mappings() -> None:
    backend = StubBackend({"landmarks": pitch_answers})
    result = run_pitch_probe(pitch_clip(), backend)

    assert (result["probe"], result["n_clips"], result["n_frames"]) == ("pitch", 1, 5)
    # Frames 0, 1 and 2 give a mapping. A frame without a mapping is not valid.
    assert result["n_frames_valid"] == 3
    assert result["valid_share"] == pytest.approx(3 / 5)
    # The share rule of the landmark fit refuses none of these answers.
    assert result["n_frames_valid_without_share_rule"] == 3
    assert result["valid_share_without_share_rule"] == pytest.approx(3 / 5)
    # The mapping of frame 2 is mirrored: it is valid, but the pipeline would reject it.
    assert result["n_frames_plausible"] == 2
    assert result["plausible_share"] == pytest.approx(2 / 5)
    assert result["n_answers_not_parsed"] == 1
    # Landmarks per frame: 8, 8, 8, 3 and 0. All 8 landmarks agree with each of the three fits.
    assert (result["landmarks_per_frame"]["n"], result["landmarks_per_frame"]["median"]) == (5, 8)
    assert (result["inliers_per_valid_frame"]["n"], result["inliers_per_valid_frame"]["median"]) == (3, 8)
    # One request per frame, with two images: the pitch diagram and the frame. No kit legend is asked.
    assert [(request.meta.step, request.meta.frame) for request in backend.requests] == [
        ("landmarks", frame) for frame in range(5)
    ]
    assert all(len(request.image_hashes) == 2 for request in backend.requests)
    assert result["prompts"] == {"landmarks": backend.requests[0].prompt_name}
    assert_plain(result)


def test_probe_pitch_counts_the_frames_that_only_the_share_rule_refuses() -> None:
    """Frame 1 names 14 landmarks: 5 at their place, and 9 that are 150 to 470 pixels beside it.

    The 5 right ones give a fit, but they are less than 40% of the landmarks, so the share rule of
    the landmark fit refuses it. The probe counts such a frame, so that the cost of the rule shows.
    """
    wrong = {15: 150.0, 21: -190.0, 24: -230.0, 16: 270.0, 17: -310.0, 18: 350.0, 19: -390.0, 20: 430.0, 11: 470.0}

    def answers(request: VLMRequest) -> Any:
        if request.meta.frame == 1:
            return landmark_answer((1, 8, 28, 35, 12, *wrong), moved=wrong)
        return landmark_answer(LANDMARK_IDS)

    clip = hand_clip("HAND-1", [people_at(GOAL_SPOTS), people_at(MIDFIELD_SPOTS)])
    result = run_pitch_probe(clip, StubBackend({"landmarks": answers}))

    assert (result["n_frames"], result["n_frames_valid"], result["n_frames_valid_without_share_rule"]) == (2, 1, 2)
    assert (result["valid_share"], result["valid_share_without_share_rule"]) == (0.5, 1.0)
    midfield = result["by_view"]["midfield"]
    assert (midfield["n_frames_valid"], midfield["n_frames_valid_without_share_rule"]) == (0, 1)
    assert (midfield["valid_share"], midfield["valid_share_without_share_rule"]) == (0.0, 1.0)


def test_probe_pitch_measures_position_errors_with_the_right_signs() -> None:
    result = run_pitch_probe(pitch_clip(), StubBackend({"landmarks": pitch_answers}))
    errors = result["error_m"]
    # 6 people in each of the three valid frames. The landmark points are rounded to model units,
    # so the right mappings are exact to a few centimetres.
    assert errors["n"] == 18
    assert errors["n_not_finite"] == 0
    assert errors["median"] < 0.1
    # The mirrored mapping of frame 2 puts every person at -x. The people stand at x = -16 m or
    # x = +16 m, so each of them is 32 m away from the label.
    assert errors["p90"] == pytest.approx(32.0, abs=0.2)
    # The plausible frames are frames 0 and 1.
    assert result["plausible_error_m"]["n"] == 12
    assert result["plausible_error_m"]["p90"] < 0.1


def test_probe_pitch_splits_the_frames_by_view() -> None:
    result = run_pitch_probe(pitch_clip(), StubBackend({"landmarks": pitch_answers}))
    goal, midfield = result["by_view"]["goal"], result["by_view"]["midfield"]
    # Goal view: frames 0 and 3. Only frame 0 has a mapping.
    assert (goal["n_frames"], goal["n_frames_valid"], goal["n_frames_plausible"]) == (2, 1, 1)
    assert goal["valid_share"] == pytest.approx(1 / 2)
    assert goal["error_m"]["n"] == 6
    assert goal["error_m"]["p90"] < 0.1
    # Midfield view: frames 1, 2 and 4. The mean |x| of frame 4 is exactly 20 m, which is not above 20 m.
    assert (midfield["n_frames"], midfield["n_frames_valid"], midfield["n_frames_plausible"]) == (3, 2, 1)
    assert midfield["valid_share"] == pytest.approx(2 / 3)
    assert midfield["plausible_share"] == pytest.approx(1 / 3)
    assert midfield["error_m"]["n"] == 12
    assert midfield["error_m"]["p90"] == pytest.approx(32.0, abs=0.2)
    assert midfield["plausible_error_m"]["n"] == 6


def test_probe_pitch_reports_settings_and_the_clip() -> None:
    result = run_pitch_probe(pitch_clip(), StubBackend({"landmarks": pitch_answers}))
    assert result["settings"] == {
        "step_settings": SETTINGS.model_dump(),
        "seed": 0,
        "repeat": 0,
        "frames_per_clip": 10,
        "ransac_px": 8.0,
        "goal_view_min_mean_abs_x_m": 20.0,
    }
    clip_values = result["per_clip"]["HAND-1"]
    assert (clip_values["n_frames"], clip_values["valid_share"]) == (5, pytest.approx(3 / 5))
    assert clip_values["error_median_m"] == result["error_m"]["median"]
    main = result["main_metric"]
    assert (main["name"], main["n_clips"]) == ("error_median_m", 1)
    assert main["mean"] == pytest.approx(result["error_m"]["median"])


def test_probe_pitch_uses_the_ransac_threshold() -> None:
    """One landmark is 200 pixels (20 m) to the left of its place."""

    def answers(request: VLMRequest) -> dict[str, Any]:
        return landmark_answer(LANDMARK_IDS, moved={24: -200.0})

    clip = hand_clip("HAND-1", [people_at(MIDFIELD_SPOTS)])
    strict = run_pitch_probe(clip, StubBackend({"landmarks": answers}), ransac_px=8.0)
    loose = run_pitch_probe(clip, StubBackend({"landmarks": answers}), ransac_px=300.0)
    # With 8 pixels the wrong landmark does not agree with the others. The fit uses the 7 right ones.
    assert strict["inliers_per_valid_frame"]["median"] == 7
    assert strict["error_m"]["median"] < 0.1
    # With 300 pixels the wrong landmark is used for the fit, and the fit is some metres off.
    assert loose["error_m"]["median"] > 10.0 * strict["error_m"]["median"]
    assert (strict["settings"]["ransac_px"], loose["settings"]["ransac_px"]) == (8.0, 300.0)


def test_position_error_of_the_pitch_probe_is_a_distance_also_across_the_pitch() -> None:
    """Every landmark is 22 model units (15.84 pixels) too low in the picture: 1.98 m of pitch y, and 0 m of pitch x."""

    def answers(request: VLMRequest) -> dict[str, Any]:
        answer = landmark_answer(LANDMARK_IDS)
        for landmark in answer["landmarks"]:
            landmark["point"][0] += 22  # A model point is [y, x].
        return answer

    clip = hand_clip("HAND-1", [people_at(MIDFIELD_SPOTS)])
    result = run_pitch_probe(clip, StubBackend({"landmarks": answers}))
    assert result["n_frames_valid"] == 1
    assert result["error_m"]["median"] == pytest.approx(1.98, abs=0.1)
    assert result["error_m"]["p90"] == pytest.approx(1.98, abs=0.1)


def bench_clip(role: str) -> LabelledClip:
    """One frame: six players on the pitch and two people 10 m behind the far touchline, without a pitch position."""
    team = None if role == "other" else "left"
    outside = [
        standing(8, (0.0, -44.0), role=role, team=team, pitch_known=False),
        standing(9, (10.0, -44.0), role=role, team=team, pitch_known=False),
    ]
    return hand_clip("HAND-1", [[*people_at(MIDFIELD_SPOTS), *outside]])


def test_people_next_to_the_pitch_do_not_count_in_the_plausibility_of_a_mapping() -> None:
    """The answer names 6 landmarks, so the mapping is checked with the foot points of the people on the pitch.

    Two people of a bench have the role "other": they do not count. The same two people as players
    do: 6 of 8 foot points (75%) are inside the pitch plus its margin, and the check wants 90%.
    """
    six_landmarks = {"landmarks": lambda request: landmark_answer(LANDMARK_IDS[:6])}
    bench = run_pitch_probe(bench_clip("other"), StubBackend(six_landmarks))
    assert (bench["n_frames_valid"], bench["n_frames_plausible"]) == (1, 1)
    players = run_pitch_probe(bench_clip("player"), StubBackend(six_landmarks))
    assert (players["n_frames_valid"], players["n_frames_plausible"]) == (1, 0)


def test_plausibility_of_the_pitch_probe_is_the_check_of_the_pipeline() -> None:
    """The pipeline keeps a fit from 7 or more agreeing landmarks whatever the foot points say. So does the probe."""
    eight_landmarks = {"landmarks": lambda request: landmark_answer(LANDMARK_IDS)}
    result = run_pitch_probe(bench_clip("player"), StubBackend(eight_landmarks))
    assert (result["n_frames_valid"], result["n_frames_plausible"]) == (1, 1)
    assert result["inliers_per_valid_frame"]["median"] == 8


class WideSource:
    """Black frames of a clip that is 2000 pixels wide and 200 pixels high."""

    def __init__(self, labels: GsrLabels) -> None:
        self.info = ClipInfo(clip_id=labels.clip_id, n_frames=len(labels.frames), fps=FPS, width=2000, height=200)

    def read(self, frame: int) -> np.ndarray:
        """Return one frame as a BGR uint8 image of shape (height, width, 3)."""
        return np.zeros((200, 2000, 3), dtype=np.uint8)


def test_plausibility_check_of_the_pitch_probe_gets_the_width_and_then_the_height() -> None:
    """A wide clip: pixel x = 1000 + 10 * pitch x and pixel y = 100 + 2 * pitch y, in 2000 x 200 pixels.

    The image centre (1000, 100) shows the centre mark. With the two sides swapped, the centre
    would be the point (100, 1000), which this camera puts 90 m to the left of the centre mark.
    """

    def wide_pixels(x: float, y: float) -> tuple[float, float]:
        return 1000.0 + 10.0 * x, 100.0 + 2.0 * y

    athletes = []
    for index, (x, y) in enumerate(MIDFIELD_SPOTS):
        foot_x, foot_y = wide_pixels(x, y)
        box = Box(x1=foot_x - 4.0, y1=foot_y - 20.0, x2=foot_x + 4.0, y2=foot_y)
        athletes.append(
            GsrAthlete(track_id=index + 1, box=box, pitch_x=x, pitch_y=y, role="player", team="left", jersey=None)
        )
    frame = GsrFrame(
        frame=0, image_id="9000001", file_name="000001.jpg", scored=True, athletes=tuple(athletes), lines={}
    )
    labels = GsrLabels(
        clip_id="WIDE-1", fps=FPS, n_frames=1, width=2000, height=200, frames=(frame,), jersey_is_str=True, version=None
    )

    def answers(request: VLMRequest) -> dict[str, Any]:
        landmarks = []
        for landmark_id in LANDMARK_IDS:
            x_px, y_px = wide_pixels(LANDMARK_BY_ID[landmark_id].x, LANDMARK_BY_ID[landmark_id].y)
            point = [round(1000.0 * y_px / 200.0), round(1000.0 * x_px / 2000.0)]  # A model point is [y, x].
            landmarks.append({"id": landmark_id, "point": point})
        return {"view": "centre", "landmarks": landmarks}

    result = probe_pitch(
        [LabelledClip(WideSource(labels), labels)],
        StubBackend({"landmarks": answers}),
        SETTINGS,
        frames_per_clip=1,
        seed=0,
        ransac_px=8.0,
    )
    assert (result["n_frames_valid"], result["n_frames_plausible"]) == (1, 1)
    assert result["error_m"]["median"] < 0.2


def test_error_summary_leaves_out_the_values_that_are_not_finite() -> None:
    summary = probes._error_summary(np.array([1.0, np.inf, 3.0, np.nan]))
    assert summary == {"n": 2.0, "median": 2.0, "p90": pytest.approx(2.8), "mean": 2.0, "n_not_finite": 2}


def test_frame_without_labelled_people_has_no_view() -> None:
    clip = hand_clip("HAND-1", [[], people_at(MIDFIELD_SPOTS)])
    result = run_pitch_probe(clip, StubBackend({"landmarks": lambda request: landmark_answer(LANDMARK_IDS)}))
    assert (result["n_frames"], result["n_frames_valid"], result["n_frames_plausible"]) == (2, 2, 2)
    assert result["error_m"]["n"] == 6
    assert result["by_view"]["goal"]["n_frames"] == 0
    assert math.isnan(result["by_view"]["goal"]["valid_share"])
    assert result["by_view"]["midfield"]["n_frames"] == 1


@pytest.mark.parametrize("changes", [{"frames_per_clip": 0}, {"ransac_px": 0.0}, {"ransac_px": -1.0}, {"repeat": -1}])
def test_probe_pitch_refuses_wrong_settings_before_any_request(changes: dict[str, Any]) -> None:
    backend = StubBackend({"landmarks": pitch_answers})
    arguments: dict[str, Any] = {"frames_per_clip": 2, "seed": 0, "ransac_px": 8.0, **changes}
    with pytest.raises(ValueError):
        probe_pitch([pitch_clip()], backend, SETTINGS, **arguments)
    assert backend.requests == []


# --- External tools on the frames of a probe -------------------------------------------

#: Settings of a tool, as plain JSON types. The score functions copy them into the result.
DETECTOR = {"name": "rfdetr", "model": "RFDETRSmall", "min_score": 0.5}
CALIBRATION = {"name": "pnlcalib", "kp_threshold": 0.3, "line_threshold": 0.7, "pnl_refine": True}
#: Keys of a pitch result that belong to the landmark fit of the model step.
LANDMARK_KEYS = {
    "prompts",
    "landmarks_per_frame",
    "inliers_per_valid_frame",
    "n_answers_not_parsed",
    "n_frames_valid_without_share_rule",
    "valid_share_without_share_rule",
}
#: The camera of the hand-made clips as a matrix from image pixels to pitch metres, and the
#: same camera with the pitch mirrored in x and in y.
CAMERA = ((0.1, 0.0, -64.0), (0.0, 0.125, -45.0), (0.0, 0.0, 1.0))
CAMERA_X_MIRRORED = ((-0.1, 0.0, 64.0), (0.0, 0.125, -45.0), (0.0, 0.0, 1.0))
#: The camera of the hand-made clips with every point 2 m too far right on the pitch.
CAMERA_2M_RIGHT = ((0.1, 0.0, -62.0), (0.0, 0.125, -45.0), (0.0, 0.0, 1.0))
CAMERA_Y_MIRRORED = ((0.1, 0.0, -64.0), (0.0, -0.125, 45.0), (0.0, 0.0, 1.0))


def tool_people(clip_id: str, frame: int) -> FramePeople:
    """The people that a tool found in one frame of the people clip: the boxes of the stub model."""
    answer = people_answer_of_frame(frame)
    return steps.parse_people(json.dumps(answer), frame=frame, width=WIDTH, height=HEIGHT, coords="yx_1000")


def tool_mapping(matrix: tuple[tuple[float, float, float], ...] | None) -> Callable[[str, int], HomographyRecord]:
    """Return a tool that gives the same mapping for every frame. None means that it finds no mapping."""

    def mapping_of(clip_id: str, frame: int) -> HomographyRecord:
        if matrix is None:
            return HomographyRecord(frame=frame)
        return HomographyRecord(frame=frame, matrix=matrix, source="landmarks")

    return mapping_of


def without_keys(result: dict[str, Any], keys: Collection[str]) -> dict[str, Any]:
    """Return a result without some keys, at the top level and in the summaries of ``by_view``."""
    cut = {key: value for key, value in result.items() if key not in keys}
    if "by_view" in cut:
        cut["by_view"] = {
            view: {key: value for key, value in summary.items() if key not in keys}
            for view, summary in cut["by_view"].items()
        }
    return cut


def score_people(
    clips: Sequence[LabelledClip], people_of: Callable[[str, int], FramePeople] | None = None, **changes: Any
) -> dict[str, Any]:
    """Score the people of a tool on hand-made clips with all frames (10 is more than a clip has).

    Without ``people_of`` the tool gives the boxes of the stub model (``tool_people``).
    """
    arguments: dict[str, Any] = {
        "frames_per_clip": 10,
        "seed": 0,
        "small_box_max_height_px": SMALL_BOX_MAX_HEIGHT_PX,
        **changes,
    }
    return score_people_answers(clips, tool_people if people_of is None else people_of, DETECTOR, **arguments)


def test_probe_frames_are_the_frames_that_the_probes_sample() -> None:
    first = hand_clip("HAND-1", [people_layout()] * 20, unscored=[0, 7])
    second = hand_clip("HAND-2", [people_layout()] * 12)

    frames = probe_frames([first, second], frames_per_clip=4, seed=11)

    assert list(frames) == ["HAND-1", "HAND-2"]
    assert frames == {"HAND-1": sample_frames(first.labels, 4, 11), "HAND-2": sample_frames(second.labels, 4, 11)}
    assert all(len(clip_frames) == 4 for clip_frames in frames.values())


def test_probe_frames_refuse_wrong_input() -> None:
    clip = people_clip()
    with pytest.raises(ValueError, match="frames_per_clip"):
        probe_frames([clip], frames_per_clip=0, seed=0)
    with pytest.raises(ValueError, match="at least one clip"):
        probe_frames([], frames_per_clip=2, seed=0)
    with pytest.raises(ValueError, match="more than once"):
        probe_frames([clip, people_clip()], frames_per_clip=2, seed=0)


def test_tool_people_are_measured_like_the_answers_of_the_model() -> None:
    """The tool gives the boxes of the stub model, so every measured value must be the same."""
    model = run_people_probe([people_clip()], people_backend())

    tool = score_people([people_clip()])

    counts = tool["counts"]
    assert counts == {"predicted_boxes": 22, "true_boxes": 21, "tp": 19, "fp": 3, "fn": 2, "fp_off_pitch": 2}
    assert tool["foot_error_m"]["n"] == 12
    step_only = {"settings", "prompts", "n_answers_not_parsed"}
    assert without_keys(tool, step_only) == without_keys(model, step_only)
    assert set(model) - set(tool) == {"prompts", "n_answers_not_parsed"}
    assert_plain(tool)


def test_tool_people_report_the_settings_of_the_tool_and_of_the_measurement() -> None:
    result = score_people([people_clip()], seed=5)

    assert result["settings"] == {
        "tool": DETECTOR,
        "seed": 5,
        "frames_per_clip": 10,
        "iou_threshold": 0.5,
        "small_box_max_height_px": SMALL_BOX_MAX_HEIGHT_PX,
        "label_ransac_px": LABEL_RANSAC_PX,
        "label_min_athletes": MIN_ATHLETES_FOR_FIT,
    }
    # The result holds a copy: a later change of the settings of the caller does not reach it.
    assert result["settings"]["tool"] is not DETECTOR


def test_tool_people_are_asked_for_the_frames_of_the_probe_only() -> None:
    asked: list[tuple[str, int]] = []

    def people_of(clip_id: str, frame: int) -> FramePeople:
        asked.append((clip_id, frame))
        return tool_people(clip_id, frame)

    clips = [people_clip("HAND-1"), people_clip("HAND-2")]
    result = score_people_answers(
        clips, people_of, DETECTOR, frames_per_clip=2, seed=3, small_box_max_height_px=SMALL_BOX_MAX_HEIGHT_PX
    )

    frames = probe_frames(clips, frames_per_clip=2, seed=3)
    assert asked == [(clip_id, frame) for clip_id, clip_frames in frames.items() for frame in clip_frames]
    assert (result["n_clips"], result["n_frames"]) == (2, 4)
    assert set(result["per_clip"]) == {"HAND-1", "HAND-2"}


def test_tool_answer_must_name_its_frame() -> None:
    def people_of(clip_id: str, frame: int) -> FramePeople:
        return FramePeople(frame=frame + 1)

    with pytest.raises(ValueError, match="clip HAND-1: the answer for frame 0 names frame 1"):
        score_people_answers(
            [people_clip()], people_of, DETECTOR, frames_per_clip=10, seed=0, small_box_max_height_px=100.0
        )
    with pytest.raises(ValueError, match="the answer for frame 0 names frame 3"):
        score_pitch_mappings(
            [pitch_clip()], lambda clip_id, frame: HomographyRecord(frame=3), CALIBRATION, frames_per_clip=10, seed=0
        )


def test_tool_answer_that_names_an_earlier_frame_is_refused() -> None:
    def people_of(clip_id: str, frame: int) -> FramePeople:
        return FramePeople(frame=max(frame - 1, 0))

    with pytest.raises(ValueError, match="the answer for frame 1 names frame 0"):
        score_people_answers(
            [people_clip()], people_of, DETECTOR, frames_per_clip=10, seed=0, small_box_max_height_px=100.0
        )
    with pytest.raises(ValueError, match="the answer for frame 1 names frame 0"):
        score_pitch_mappings(
            [pitch_clip()], lambda clip_id, frame: HomographyRecord(frame=0), CALIBRATION, frames_per_clip=10, seed=0
        )


def test_tool_people_refuse_no_clip_and_two_clips_with_one_id() -> None:
    def people_of(clip_id: str, frame: int) -> FramePeople:
        raise AssertionError("the tool must not be asked")

    arguments: dict[str, Any] = {"frames_per_clip": 2, "seed": 0, "small_box_max_height_px": 100.0}
    with pytest.raises(ValueError, match="at least one clip"):
        score_people_answers([], people_of, DETECTOR, **arguments)
    with pytest.raises(ValueError, match="more than once"):
        score_people_answers([people_clip(), people_clip()], people_of, DETECTOR, **arguments)


@pytest.mark.parametrize(
    "changes", [{"frames_per_clip": 0}, {"small_box_max_height_px": 0.0}, {"small_box_max_height_px": math.nan}]
)
def test_tool_people_refuse_wrong_settings_before_the_tool_is_asked(changes: dict[str, Any]) -> None:
    def people_of(clip_id: str, frame: int) -> FramePeople:
        raise AssertionError("the tool must not be asked")

    arguments: dict[str, Any] = {"frames_per_clip": 2, "seed": 0, "small_box_max_height_px": 100.0, **changes}
    with pytest.raises(ValueError):
        score_people_answers([people_clip()], people_of, DETECTOR, **arguments)


def test_tool_mappings_are_measured_like_the_fits_of_the_model() -> None:
    """The tool gives the landmark fit of each stub answer, so every shared value must be the same."""
    model = run_pitch_probe(pitch_clip(), StubBackend({"landmarks": pitch_answers}))

    def mapping_of(clip_id: str, frame: int) -> HomographyRecord:
        answer = pitch_answer_of_frame(frame)
        if isinstance(answer, str):
            return HomographyRecord(frame=frame)
        landmarks = steps.parse_landmarks(json.dumps(answer), frame=frame, width=WIDTH, height=HEIGHT, coords="yx_1000")
        return fit_from_landmarks(landmarks, ransac_px=8.0)

    tool = score_pitch_mappings([pitch_clip()], mapping_of, CALIBRATION, frames_per_clip=10, seed=0)

    assert (tool["probe"], tool["n_clips"], tool["n_frames"]) == ("pitch", 1, 5)
    assert (tool["n_frames_valid"], tool["n_frames_plausible"]) == (3, 2)
    not_shared = {"settings", "axis_check", *LANDMARK_KEYS}
    assert without_keys(tool, not_shared) == without_keys(model, not_shared)
    # The tool result has no value of a landmark fit, and one value more: the axis check.
    assert set(model) - set(tool) == LANDMARK_KEYS
    assert set(tool) - set(model) == {"axis_check"}
    assert not LANDMARK_KEYS & set(tool["by_view"]["goal"])
    assert_plain(tool)


def test_tool_mappings_of_two_clips_keep_the_values_of_each_clip() -> None:
    """HAND-1 (one midfield frame) has the exact camera. HAND-2 (two goal frames) has a camera that is 2 m off."""
    first = hand_clip("HAND-1", [people_at(MIDFIELD_SPOTS)])
    second = hand_clip("HAND-2", [people_at(GOAL_SPOTS), people_at(GOAL_SPOTS)])

    def mapping_of(clip_id: str, frame: int) -> HomographyRecord:
        return tool_mapping(CAMERA if clip_id == "HAND-1" else CAMERA_2M_RIGHT)(clip_id, frame)

    result = score_pitch_mappings([first, second], mapping_of, CALIBRATION, frames_per_clip=10, seed=0)

    assert (result["n_clips"], result["n_frames"], result["n_frames_valid"]) == (2, 3, 3)
    assert result["per_clip"] == {
        "HAND-1": {"n_frames": 1, "valid_share": 1.0, "error_median_m": pytest.approx(0.0, abs=1e-9)},
        "HAND-2": {"n_frames": 2, "valid_share": 1.0, "error_median_m": pytest.approx(2.0)},
    }
    main = result["main_metric"]
    assert (main["n_clips"], main["mean"]) == (2, pytest.approx(1.0))
    # 6 people without an error and 12 people with 2 m: the median of all 18 is 2 m.
    assert (result["error_m"]["n"], result["error_m"]["median"]) == (18, pytest.approx(2.0))
    assert result["axis_check"]["as_given"] == pytest.approx(2.0)
    # The views take the frames of both clips.
    goal, midfield = result["by_view"]["goal"], result["by_view"]["midfield"]
    assert (goal["n_frames"], goal["error_m"]["n"], goal["error_m"]["median"]) == (2, 12, pytest.approx(2.0))
    assert (midfield["n_frames"], midfield["error_m"]["n"]) == (1, 6)
    assert midfield["error_m"]["median"] == pytest.approx(0.0, abs=1e-9)


def test_probe_pitch_of_two_clips_keeps_the_values_of_each_clip() -> None:
    """HAND-1 (one midfield frame) gets the right landmarks. HAND-2 (two goal frames): the landmarks of
    frame 0 are 22 model units too low in the picture (1.98 m of pitch y), and the answer of frame 1
    cannot be parsed.
    """

    def answers(request: VLMRequest) -> Any:
        answer = landmark_answer(LANDMARK_IDS)
        if request.meta.clip_id == "HAND-2":
            if request.meta.frame == 1:
                return "this is not JSON"
            for landmark in answer["landmarks"]:
                landmark["point"][0] += 22  # A model point is [y, x].
        return answer

    clips = [hand_clip("HAND-1", [people_at(MIDFIELD_SPOTS)]), hand_clip("HAND-2", [people_at(GOAL_SPOTS)] * 2)]
    backend = StubBackend({"landmarks": answers})
    result = probe_pitch(clips, backend, SETTINGS, frames_per_clip=10, seed=0, ransac_px=8.0)

    assert (result["n_clips"], result["n_frames"], result["n_frames_valid"]) == (2, 3, 2)
    assert set(result["per_clip"]) == {"HAND-1", "HAND-2"}
    first, second = result["per_clip"]["HAND-1"], result["per_clip"]["HAND-2"]
    assert (first["n_frames"], first["valid_share"]) == (1, 1.0)
    assert first["error_median_m"] < 0.1
    assert (second["n_frames"], second["valid_share"]) == (2, 0.5)
    assert second["error_median_m"] == pytest.approx(1.98, abs=0.1)
    main = result["main_metric"]
    assert (main["n_clips"], main["mean"]) == (2, pytest.approx(0.99, abs=0.1))
    # The values of all frames, the views and the facts of the answers take the frames of both clips.
    assert result["error_m"]["n"] == 12
    goal, midfield = result["by_view"]["goal"], result["by_view"]["midfield"]
    assert (goal["n_frames"], goal["n_frames_valid"], goal["error_m"]["n"]) == (2, 1, 6)
    assert (midfield["n_frames"], midfield["n_frames_valid"], midfield["error_m"]["n"]) == (1, 1, 6)
    assert result["n_answers_not_parsed"] == 1
    # Landmarks per frame: 8, 8 and 0.
    assert (result["landmarks_per_frame"]["n"], result["landmarks_per_frame"]["mean"]) == (3, pytest.approx(16 / 3))
    assert (result["inliers_per_valid_frame"]["n"], result["inliers_per_valid_frame"]["median"]) == (2, 8)


def test_tool_mappings_count_valid_and_plausible_frames() -> None:
    """Frames 0, 1 and 4 get the camera, frame 2 gets the camera mirrored in x, and frame 3 gets no mapping."""

    def mapping_of(clip_id: str, frame: int) -> HomographyRecord:
        matrix = {2: CAMERA_X_MIRRORED, 3: None}.get(frame, CAMERA)
        return tool_mapping(matrix)(clip_id, frame)

    result = score_pitch_mappings([pitch_clip()], mapping_of, CALIBRATION, frames_per_clip=10, seed=0)

    assert (result["n_frames"], result["n_frames_valid"], result["n_frames_plausible"]) == (5, 4, 3)
    assert (result["valid_share"], result["plausible_share"]) == (pytest.approx(4 / 5), pytest.approx(3 / 5))
    # 6 + 6 + 2 people in the frames with the camera, and 6 people in the mirrored frame: each of
    # them stands at x = -16 m or x = +16 m, so the mirrored mapping puts each one 32 m away.
    assert result["error_m"]["n"] == 20
    assert result["error_m"]["median"] == pytest.approx(0.0, abs=1e-9)
    assert result["error_m"]["p90"] == pytest.approx(32.0)
    assert result["plausible_error_m"]["n"] == 14
    assert result["plausible_error_m"]["p90"] == pytest.approx(0.0, abs=1e-9)
    goal, midfield = result["by_view"]["goal"], result["by_view"]["midfield"]
    assert (goal["n_frames"], goal["n_frames_valid"], goal["n_frames_plausible"]) == (2, 1, 1)
    assert (midfield["n_frames"], midfield["n_frames_valid"], midfield["n_frames_plausible"]) == (3, 3, 2)
    assert result["settings"] == {
        "tool": CALIBRATION,
        "seed": 0,
        "frames_per_clip": 10,
        "goal_view_min_mean_abs_x_m": 20.0,
    }
    clip_values = result["per_clip"]["HAND-1"]
    assert (clip_values["n_frames"], clip_values["valid_share"]) == (5, pytest.approx(4 / 5))
    main = result["main_metric"]
    assert (main["name"], main["n_clips"], main["mean"]) == ("error_median_m", 1, pytest.approx(0.0, abs=1e-9))


@pytest.mark.parametrize(
    ("matrix", "smallest"),
    [(CAMERA, "as_given"), (CAMERA_X_MIRRORED, "x_flipped"), (CAMERA_Y_MIRRORED, "y_flipped")],
)
def test_axis_check_names_the_pitch_convention_of_the_tool_output(
    matrix: tuple[tuple[float, float, float], ...], smallest: str
) -> None:
    """A tool output with the sign of one pitch axis changed is exact again when that sign is changed back."""
    clip = hand_clip("HAND-1", [people_at(GOAL_SPOTS), people_at(MIDFIELD_SPOTS)])

    check = score_pitch_mappings([clip], tool_mapping(matrix), CALIBRATION, frames_per_clip=10, seed=0)["axis_check"]

    assert set(check) == {"as_given", "x_flipped", "y_flipped", "both_flipped"}
    assert min(check, key=check.__getitem__) == smallest
    assert check[smallest] == pytest.approx(0.0, abs=1e-9)
    assert all(value > 10.0 for name, value in check.items() if name != smallest)


def test_axis_check_of_the_right_convention_gives_the_distances_by_hand() -> None:
    """Two people at (16, 9) and (-32, 18): a changed sign of x moves them by 32 m and 64 m, of y by 18 m and 36 m."""
    clip = hand_clip("HAND-1", [[standing(1, (16.0, 9.0)), standing(2, (-32.0, 18.0))]])

    check = score_pitch_mappings([clip], tool_mapping(CAMERA), CALIBRATION, frames_per_clip=10, seed=0)["axis_check"]

    assert check["as_given"] == pytest.approx(0.0, abs=1e-9)
    assert check["x_flipped"] == pytest.approx((32.0 + 64.0) / 2)
    assert check["y_flipped"] == pytest.approx((18.0 + 36.0) / 2)
    assert check["both_flipped"] == pytest.approx((math.hypot(32.0, 18.0) + math.hypot(64.0, 36.0)) / 2)


def test_tool_without_any_mapping_gives_no_error_and_no_axis_check() -> None:
    result = score_pitch_mappings([pitch_clip()], tool_mapping(None), CALIBRATION, frames_per_clip=10, seed=0)

    assert (result["n_frames"], result["n_frames_valid"], result["valid_share"]) == (5, 0, 0.0)
    assert result["error_m"]["n"] == 0
    assert all(math.isnan(value) for value in result["axis_check"].values())
    assert math.isnan(result["main_metric"]["mean"])
    assert_plain(result)


def test_tool_mappings_refuse_wrong_settings_before_the_tool_is_asked() -> None:
    def mapping_of(clip_id: str, frame: int) -> HomographyRecord:
        raise AssertionError("the tool must not be asked")

    with pytest.raises(ValueError, match="frames_per_clip"):
        score_pitch_mappings([pitch_clip()], mapping_of, CALIBRATION, frames_per_clip=0, seed=0)
    with pytest.raises(ValueError, match="at least one clip"):
        score_pitch_mappings([], mapping_of, CALIBRATION, frames_per_clip=2, seed=0)


# --- Number probe on hand-made clips ---------------------------------------------------

#: Answers of the stub to the number requests, by track id.
NUMBER_ANSWERS: dict[int, Any] = {
    7: {"per_crop": [10, 10, 10], "number": 10, "confidence": "high"},
    8: {"per_crop": [None, None], "number": None, "confidence": "low"},
    9: {"per_crop": [7], "number": 7, "confidence": "medium"},
    11: {"per_crop": [None], "number": None, "confidence": "low"},
    13: "this is not JSON",
}


def number_clip() -> LabelledClip:
    """A clip of 100 frames (4 seconds) for the number probe. Every box is 50 pixels wide and ends at y = 400.

    - track 7, player, number 10: boxes in frames 0, 10, 30, 35, 60 and 90, with different heights;
    - track 8, player, no number: frames 5 and 50;
    - tracks 9, 11 and 13, players, numbers 4, 9 and 5: one frame each;
    - track 3, goalkeeper, and track 12, referee: not players.
    """
    frames: list[list[GsrAthlete]] = [[] for _ in range(100)]

    def add(
        frame: int,
        track_id: int,
        left: float,
        height_px: float,
        *,
        role: str = "player",
        team: str | None = "left",
        jersey: int | None = None,
    ) -> None:
        box = Box(x1=left, y1=400.0 - height_px, x2=left + 50.0, y2=400.0)
        frames[frame].append(
            GsrAthlete(track_id=track_id, box=box, pitch_x=0.0, pitch_y=0.0, role=role, team=team, jersey=jersey)
        )

    for frame, height_px in ((0, 110.0), (10, 190.0), (30, 170.0), (35, 130.0), (60, 90.0), (90, 150.0)):
        add(frame, 7, 100.0, height_px, jersey=10)
    add(5, 8, 300.0, 90.0, team="right")
    add(50, 8, 300.0, 110.0, team="right")
    add(20, 9, 500.0, 130.0, jersey=4)
    add(20, 11, 700.0, 90.0, team="right", jersey=9)
    add(40, 13, 900.0, 110.0, jersey=5)
    add(10, 3, 1100.0, 150.0, role="goalkeeper", jersey=1)
    add(60, 3, 1100.0, 150.0, role="goalkeeper", jersey=1)
    add(10, 12, 1200.0, 130.0, role="referee", team=None)
    return hand_clip("HAND-1", frames)


def track_of(request: VLMRequest) -> int:
    """Read the track id out of the item of a number request: ``track-<id>-<8 hex digits>``."""
    match = re.fullmatch(r"track-(\d+)-[0-9a-f]{8}", request.meta.item or "")
    assert match is not None, request.meta.item
    return int(match.group(1))


def number_backend() -> StubBackend:
    """Return a stub model for the number clip. It answers by the track id of the request."""
    return StubBackend({"kit_legend": legend_answer, "number": lambda request: NUMBER_ANSWERS[track_of(request)]})


def run_number_probe(clip: LabelledClip, backend: StubBackend, **changes: Any) -> dict[str, Any]:
    """Run the number probe on a hand-made clip with all its player tracks (10 is more than it has)."""
    arguments: dict[str, Any] = {
        "tracks_per_clip": 10,
        "crops_per_track": 3,
        "crop_upscale": 2.0,
        "seed": 0,
        **changes,
    }
    return probe_number([clip], backend, SETTINGS, **arguments)


def test_probe_number_scores_the_answers_against_the_labels() -> None:
    backend = number_backend()
    result = run_number_probe(number_clip(), backend)

    assert (result["probe"], result["n_clips"], result["n_tracks"]) == ("number", 1, 5)
    # Track 7: 10 against 10. Track 8: no number against no number, which is a match too.
    # Track 9: 7 against 4. Track 11: no number against 9. Track 13: no usable answer against 5.
    assert result["exact_match"] == pytest.approx(2 / 5)
    assert result["null_answer_share"] == pytest.approx(3 / 5)
    assert result["by_label"]["number"] == {
        "n_tracks": 4,
        "exact_match": pytest.approx(1 / 4),
        "null_answer_share": pytest.approx(2 / 4),
    }
    assert result["by_label"]["null"] == {"n_tracks": 1, "exact_match": 1.0, "null_answer_share": 1.0}
    assert result["n_answers_not_parsed"] == 1
    assert result["n_crops"] == 3 + 2 + 1 + 1 + 1
    assert result["per_clip"] == {
        "HAND-1": {"n_tracks": 5, "exact_match": pytest.approx(2 / 5), "null_answer_share": pytest.approx(3 / 5)}
    }
    main = result["main_metric"]
    assert (main["name"], main["n_clips"]) == ("exact_match", 1)
    assert main["mean"] == pytest.approx(2 / 5)
    # One kit legend request, then one number request per player track, in the order of the track ids.
    assert [request.meta.step for request in backend.requests] == ["kit_legend", *["number"] * 5]
    assert [track_of(request) for request in backend.sent("number")] == [7, 8, 9, 11, 13]
    assert result["prompts"] == {request.meta.step: request.prompt_name for request in backend.requests}
    assert_plain(result)


def lists_of_four_numbers(value: Any) -> list[tuple[float, ...]]:
    """Return every list of four numbers inside JSON data, at any depth."""
    if isinstance(value, dict):
        return [found for item in value.values() for found in lists_of_four_numbers(item)]
    if not isinstance(value, list):
        return []
    if len(value) == 4 and all(type(item) in (int, float) for item in value):
        return [tuple(value)]
    return [found for item in value for found in lists_of_four_numbers(item)]


def test_stored_answers_of_the_number_probe_hold_no_box_of_the_labels(tmp_path: Path) -> None:
    """The number probe cuts its crops at boxes of the label file. No such box reaches a stored answer."""
    clip = number_clip()
    stub = number_backend()
    backend = CachedBackend(CassetteStore(tmp_path), stub)

    probe_number([clip], backend, SETTINGS, tracks_per_clip=10, crops_per_track=3, crop_upscale=2.0, seed=0)

    label_boxes = {
        (athlete.box.x1, athlete.box.y1, athlete.box.x2, athlete.box.y2)
        for frame in clip.labels.frames
        for athlete in frame.athletes
    }
    # Control: the requests carry label boxes. The cache key and the fake models read them.
    sent_boxes = {tuple(box) for request in stub.sent("number") for box in request.context["boxes"]}
    assert len(sent_boxes) == 8
    assert sent_boxes <= label_boxes
    cassettes = sorted(tmp_path.glob("HAND-1/number/*.json"))
    assert len(cassettes) == 5
    for path in cassettes:
        text = path.read_text(encoding="utf-8")
        stored = json.loads(text)
        assert set(stored["context"]) == {"frames"}
        assert "boxes" not in text
        assert not label_boxes.intersection(lists_of_four_numbers(stored))


def test_probe_number_reports_its_settings() -> None:
    result = run_number_probe(number_clip(), number_backend(), seed=4, repeat=1)
    assert result["settings"] == {
        "step_settings": SETTINGS.model_dump(),
        "seed": 4,
        "repeat": 1,
        "tracks_per_clip": 10,
        "crops_per_track": 3,
        "crop_upscale": 2.0,
        "crop_padding": 0.15,
        "min_crop_gap_s": 1.0,
        "team_key": "none",
    }


def test_probe_number_cuts_the_tallest_boxes_one_second_apart() -> None:
    backend = number_backend()
    run_number_probe(number_clip(), backend)
    request = backend.sent("number")[0]
    assert track_of(request) == 7
    assert (request.meta.frame, request.meta.clip_id, request.meta.repeat) == (None, "HAND-1", 0)
    # The tallest box of track 7 is in frame 10 (190 px). The next one is in frame 30 (170 px), but
    # that is less than one second (25 frames) after frame 10. Then come frame 90 (150 px) and
    # frame 35 (130 px), which is exactly one second after frame 10. The crops are in frame order.
    assert request.context == {
        "frames": [10, 35, 90],
        "boxes": [[100.0, 210.0, 150.0, 400.0], [100.0, 270.0, 150.0, 400.0], [100.0, 250.0, 150.0, 400.0]],
    }
    images = [part for part in request.parts if isinstance(part, ImagePart)]
    assert len(images) == 3
    # A crop is the box plus 15% of its size on every side, in whole pixels, enlarged 2 times.
    # The box of frame 10 is 50 x 190 pixels: 66 x 248 with the padding, 132 x 496 enlarged.
    shapes = [(496, 132, 3), (340, 132, 3), (392, 132, 3)]
    for part, frame, shape in zip(images, (10, 35, 90), shapes, strict=True):
        crop = cv2.imdecode(np.frombuffer(part.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        # An image has the shape (height, width, 3).
        assert crop.shape == shape
        # The middle of the crop shows the painted box of that frame. The corner shows the black padding.
        middle = crop[shape[0] // 2, shape[1] // 2]
        assert all(abs(int(value) - (100 + frame)) <= 6 for value in middle)
        assert all(int(value) < 40 for value in crop[0, 0])
    # Track 8 has two boxes, 45 frames apart: both are cut.
    assert backend.sent("number")[1].context["frames"] == [5, 50]


def test_number_requests_get_an_item_that_follows_the_crops() -> None:
    clip = number_clip()

    def first_request(**changes: Any) -> VLMRequest:
        backend = number_backend()
        run_number_probe(clip, backend, **changes)
        return backend.sent("number")[0]

    request = first_request()
    assert track_of(request) == 7
    # The same crops give the same item and the same cache key.
    assert first_request().meta.item == request.meta.item
    assert first_request().cache_key() == request.cache_key()
    # Another enlargement shows other pixels, so the stored answer must not be used again.
    enlarged = first_request(crop_upscale=3.0)
    assert track_of(enlarged) == 7
    assert enlarged.context == request.context
    assert enlarged.meta.item != request.meta.item
    assert enlarged.cache_key() != request.cache_key()
    # Other crops give another item too.
    two_crops = first_request(crops_per_track=2)
    assert two_crops.context["frames"] == [10, 90]
    assert two_crops.meta.item != request.meta.item


def test_probe_number_takes_a_fixed_choice_of_tracks() -> None:
    clip = number_clip()

    def probed_tracks(seed: int) -> tuple[int, ...]:
        backend = number_backend()
        result = run_number_probe(clip, backend, tracks_per_clip=2, crop_upscale=1.0, seed=seed)
        assert result["n_tracks"] == 2
        return tuple(track_of(request) for request in backend.sent("number"))

    assert probed_tracks(4) == probed_tracks(4)
    choices = {probed_tracks(seed) for seed in range(20)}
    assert len(choices) > 1
    for choice in choices:
        assert set(choice) <= {7, 8, 9, 11, 13}
        assert list(choice) == sorted(set(choice))


def test_labelled_number_and_scored_frames_of_a_track() -> None:
    """Track 5 has a number in one of its four labelled frames. Track 6 is in a frame that is not scored."""
    frames: list[list[GsrAthlete]] = [[] for _ in range(100)]
    for frame, jersey in ((0, None), (30, None), (60, 8), (90, None)):
        frames[frame].append(standing(5, (0.0, 0.0), jersey=jersey))
    frames[10].append(standing(6, (16.0, 9.0), jersey=3))
    clip = hand_clip("HAND-1", frames, unscored=[10])
    backend = StubBackend(
        {"kit_legend": legend_answer, "number": lambda request: {"per_crop": [8], "number": 8, "confidence": "low"}}
    )
    result = run_number_probe(clip, backend)
    assert [track_of(request) for request in backend.sent("number")] == [5]
    assert result["n_tracks"] == 1
    # The labelled number of track 5 is 8, not "no number".
    assert result["by_label"]["number"]["n_tracks"] == 1
    assert result["exact_match"] == 1.0
    assert result["null_answer_share"] == 0.0


def test_labelled_number_of_a_track_is_its_most_frequent_number() -> None:
    """Track 5 has the number 8 in three labelled frames and the number 3 in one frame."""
    frames: list[list[GsrAthlete]] = [[] for _ in range(100)]
    for frame, jersey in ((0, 8), (30, 8), (60, 3), (90, 8)):
        frames[frame].append(standing(5, (0.0, 0.0), jersey=jersey))
    answer = {"per_crop": [8], "number": 8, "confidence": "low"}
    backend = StubBackend({"kit_legend": legend_answer, "number": lambda request: answer})
    assert run_number_probe(hand_clip("HAND-1", frames), backend)["exact_match"] == 1.0


def test_number_probe_sends_the_repeat_index_with_every_request() -> None:
    backend = number_backend()
    run_number_probe(number_clip(), backend, repeat=2)
    assert len(backend.sent("number")) == 5
    assert {request.meta.repeat for request in backend.requests} == {2}


def test_every_clip_of_the_number_probe_reports_its_own_scores() -> None:
    frames: list[list[GsrAthlete]] = [[] for _ in range(100)]
    frames[0].append(standing(7, (0.0, 0.0), jersey=10))  # The stub answers 10 for track 7.
    clips = [number_clip(), hand_clip("HAND-2", frames)]
    result = probe_number(
        clips, number_backend(), SETTINGS, tracks_per_clip=10, crops_per_track=3, crop_upscale=2.0, seed=0
    )
    assert result["per_clip"] == {
        "HAND-1": {"n_tracks": 5, "exact_match": pytest.approx(2 / 5), "null_answer_share": pytest.approx(3 / 5)},
        "HAND-2": {"n_tracks": 1, "exact_match": 1.0, "null_answer_share": 0.0},
    }
    assert result["main_metric"]["mean"] == pytest.approx((2 / 5 + 1.0) / 2)
    assert (result["n_tracks"], result["exact_match"]) == (6, pytest.approx(3 / 6))


def test_probe_number_asks_nothing_for_a_clip_without_player_tracks() -> None:
    people = [standing(1, (0.0, 0.0), role="goalkeeper", jersey=1), standing(2, (16.0, 9.0), role="referee", team=None)]
    backend = number_backend()
    result = run_number_probe(hand_clip("HAND-1", [people, people, people]), backend)
    assert (result["n_clips"], result["n_tracks"], result["n_crops"]) == (1, 0, 0)
    assert math.isnan(result["exact_match"])
    assert math.isnan(result["null_answer_share"])
    assert result["main_metric"]["n_clips"] == 0
    assert math.isnan(result["main_metric"]["mean"])
    assert backend.requests == []


@pytest.mark.parametrize(
    "changes", [{"tracks_per_clip": 0}, {"crops_per_track": 0}, {"crop_upscale": 0.5}, {"repeat": -1}]
)
def test_probe_number_refuses_wrong_settings_before_any_request(changes: dict[str, Any]) -> None:
    backend = number_backend()
    with pytest.raises(ValueError):
        run_number_probe(number_clip(), backend, **changes)
    assert backend.requests == []


def test_crops_are_chosen_like_the_crops_of_the_pipeline() -> None:
    """The number probe must show the model the crops that the pipeline would send for the same track."""
    from pitchvlm import pipeline
    from pitchvlm.schemas import Track, TrackObs

    clip = number_clip()
    backend = number_backend()
    run_number_probe(clip, backend)
    for request in backend.sent("number"):
        track_id = track_of(request)
        boxes = [
            (frame.frame, box_of(athlete))
            for frame in clip.labels.frames
            for athlete in frame.athletes
            if athlete.track_id == track_id
        ]
        track = Track(
            track_id=track_id,
            observations=tuple(TrackObs(frame=frame, box=box, role="player", team_key="A") for frame, box in boxes),
        )
        from_pipeline = pipeline._crop_observations(track, clip.labels.fps, 3)
        assert request.context["frames"] == [observation.frame for observation in from_pipeline]
    # One rule and one constant for both: the probe has no copy of them.
    assert probes.crop_choice is pipeline.crop_choice
    assert probes.MIN_CROP_GAP_S is pipeline.MIN_CROP_GAP_S


def test_probe_number_gives_two_crops_to_a_track_that_is_seen_for_less_than_one_second() -> None:
    """A player with boxes in the frames 3, 13 and 23: no two of them are one second (25 frames) apart."""
    frames: list[list[GsrAthlete]] = [[] for _ in range(30)]
    for frame, height_px in ((3, 100.0), (13, 140.0), (23, 120.0)):
        frames[frame].append(standing(7, (0.0, 0.0), jersey=10, height_px=height_px))
    backend = number_backend()
    run_number_probe(hand_clip("HAND-1", frames), backend)
    # The tallest box is in frame 13. Frames 3 and 23 are equally far from it, and frame 23 has the taller box.
    assert backend.sent("number")[0].context["frames"] == [13, 23]


def test_crops_are_cut_like_the_crops_of_the_pipeline() -> None:
    """The number probe must show the model the same kind of crop as the pipeline does."""
    from pitchvlm import pipeline

    image = np.random.default_rng(0).integers(0, 256, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    # One box inside the image, and one that the right and the bottom border cut.
    for box in (Box(x1=100.0, y1=200.0, x2=150.0, y2=290.0), Box(x1=1250.0, y1=650.0, x2=1300.0, y2=740.0)):
        from_pipeline = pipeline._enlarge(pipeline._cut_crop(image, box), 2.0)
        from_probe = probes._cut_crop(image, box, 2.0)
        assert np.array_equal(from_probe, from_pipeline)
        # Both cut the pixel area of the number request (``steps.crop_rect``), 2 times enlarged.
        left, top, right, bottom = steps.crop_rect(box, WIDTH, HEIGHT)
        assert from_probe.shape == (2 * (bottom - top), 2 * (right - left), 3)


# --- The probes on the synthetic clip --------------------------------------------------


@pytest.fixture(scope="module")
def scene() -> SyntheticScene:
    """A synthetic clip of 100 frames: 4 seconds at 25 frames per second, 1280 x 720 pixels."""
    return SyntheticScene(seed=3)


@pytest.fixture(scope="module")
def synthetic_clip(scene: SyntheticScene, tmp_path_factory: pytest.TempPathFactory) -> LabelledClip:
    """The synthetic clip with labels that went through a label file."""
    path = tmp_path_factory.mktemp("labels") / scene.info.clip_id / "Labels-GameState.json"
    path.parent.mkdir()
    path.write_text(json.dumps(scene.gsr_labels()), encoding="utf-8")
    return LabelledClip(SyntheticSource(scene), load_gsr_labels(path))


def synthetic_people_probe(clip: LabelledClip, backend: FakeBackend) -> dict[str, Any]:
    """Run the people probe on 20 frames of the synthetic clip."""
    return probe_people(
        [clip], backend, SETTINGS, frames_per_clip=20, seed=1, small_box_max_height_px=SYNTHETIC_SMALL_BOX_PX
    )


def synthetic_pitch_probe(clip: LabelledClip, backend: FakeBackend) -> dict[str, Any]:
    """Run the pitch probe on 12 frames of the synthetic clip."""
    return probe_pitch([clip], backend, SETTINGS, frames_per_clip=12, seed=1, ransac_px=8.0)


@pytest.fixture(scope="module")
def exact_people(synthetic_clip: LabelledClip, scene: SyntheticScene) -> dict[str, Any]:
    """The people probe with a model that answers the truth, rounded to model coordinates."""
    return synthetic_people_probe(synthetic_clip, FakeBackend(scene))


@pytest.fixture(scope="module")
def exact_pitch(synthetic_clip: LabelledClip, scene: SyntheticScene) -> dict[str, Any]:
    """The pitch probe with a model that answers the truth, rounded to model coordinates."""
    return synthetic_pitch_probe(synthetic_clip, FakeBackend(scene))


def test_people_probe_finds_every_person_of_the_synthetic_clip(
    exact_people: dict[str, Any], synthetic_clip: LabelledClip
) -> None:
    labels = synthetic_clip.labels
    frames = sample_frames(labels, 20, seed=1)
    n_people = sum(len(gsr_frame_people(labels, frame).people) for frame in frames)
    assert n_people >= 20 * 8

    assert (exact_people["n_clips"], exact_people["n_frames"]) == (1, 20)
    assert exact_people["recall"] == 1.0
    assert exact_people["precision"] == 1.0
    assert exact_people["counts"] == {
        "predicted_boxes": n_people,
        "true_boxes": n_people,
        "tp": n_people,
        "fp": 0,
        "fn": 0,
        "fp_off_pitch": 0,
    }
    # Far people have small boxes and near people have tall boxes: both kinds are in the clip.
    small = exact_people["small_boxes"]
    assert 0 < small["true_boxes"] < n_people
    assert (small["recall"], small["precision"]) == (1.0, 1.0)
    assert exact_people["n_frames_without_label_mapping"] == 0
    assert exact_people["n_answers_not_parsed"] == 0
    assert exact_people["per_clip"][labels.clip_id]["recall"] == 1.0
    assert exact_people["main_metric"]["mean"] == 1.0
    assert_plain(exact_people)


def test_people_probe_has_small_foot_errors_for_exact_boxes(exact_people: dict[str, Any]) -> None:
    foot = exact_people["foot_error_m"]
    # Every box is matched and every frame has a label mapping.
    assert foot["n"] == exact_people["counts"]["tp"]
    assert foot["n_not_finite"] == 0
    # The rounding to model coordinates moves a foot point by less than one pixel.
    assert foot["median"] < 0.05


def test_people_probe_measures_missed_people(synthetic_clip: LabelledClip, scene: SyntheticScene) -> None:
    result = synthetic_people_probe(synthetic_clip, FakeBackend(scene, miss_rate=0.3, seed=5))
    assert 0.55 < result["recall"] < 0.85
    # A missed person is not a wrong box.
    assert result["precision"] == 1.0
    assert result["counts"]["fp"] == 0
    assert result["counts"]["fn"] > 0
    assert result["counts"]["tp"] + result["counts"]["fn"] == result["counts"]["true_boxes"]


def test_people_probe_measures_box_noise(
    exact_people: dict[str, Any], synthetic_clip: LabelledClip, scene: SyntheticScene
) -> None:
    noisy = synthetic_people_probe(synthetic_clip, FakeBackend(scene, box_noise_px=2.0, seed=5))
    assert noisy["foot_error_m"]["median"] > 2.0 * exact_people["foot_error_m"]["median"]
    assert noisy["recall"] <= 1.0


def test_pitch_probe_maps_the_synthetic_clip_from_exact_landmarks(
    exact_pitch: dict[str, Any], synthetic_clip: LabelledClip
) -> None:
    labels = synthetic_clip.labels
    frames = sample_frames(labels, 12, seed=1)
    n_people = sum(len(gsr_frame_people(labels, frame).people) for frame in frames)

    assert (exact_pitch["n_clips"], exact_pitch["n_frames"]) == (1, 12)
    assert exact_pitch["valid_share"] == 1.0
    assert exact_pitch["plausible_share"] == 1.0
    assert exact_pitch["n_answers_not_parsed"] == 0
    # Every labelled person of the clip has a box and a pitch position.
    assert exact_pitch["error_m"]["n"] == n_people
    assert exact_pitch["error_m"]["median"] < 0.1
    assert exact_pitch["landmarks_per_frame"]["median"] >= 7
    views = exact_pitch["by_view"]
    assert views["goal"]["n_frames"] + views["midfield"]["n_frames"] == 12
    assert exact_pitch["per_clip"][labels.clip_id]["valid_share"] == 1.0
    assert_plain(exact_pitch)


def test_pitch_probe_measures_landmark_noise(
    exact_pitch: dict[str, Any], synthetic_clip: LabelledClip, scene: SyntheticScene
) -> None:
    noisy = synthetic_pitch_probe(synthetic_clip, FakeBackend(scene, point_noise_px=30.0, seed=5))
    assert noisy["n_frames_valid"] >= 1
    assert noisy["error_m"]["median"] > 10.0 * exact_pitch["error_m"]["median"]


def test_number_probe_reads_the_numbers_of_the_synthetic_clip(
    synthetic_clip: LabelledClip, scene: SyntheticScene
) -> None:
    result = probe_number(
        [synthetic_clip],
        FakeBackend(scene),
        StepSettings(resolution="low"),
        tracks_per_clip=6,
        crops_per_track=3,
        crop_upscale=2.0,
        seed=1,
    )
    assert (result["n_clips"], result["n_tracks"]) == (1, 6)
    assert result["exact_match"] == 1.0
    assert result["null_answer_share"] == 0.0
    assert 6 <= result["n_crops"] <= 6 * 3
    # Every player of the scene has a shirt number.
    assert result["by_label"]["number"] == {"n_tracks": 6, "exact_match": 1.0, "null_answer_share": 0.0}
    assert result["by_label"]["null"]["n_tracks"] == 0
    assert result["n_answers_not_parsed"] == 0
    assert result["per_clip"] == {
        synthetic_clip.labels.clip_id: {"n_tracks": 6, "exact_match": 1.0, "null_answer_share": 0.0}
    }
    assert result["settings"]["step_settings"]["resolution"] == "low"
    assert_plain(result)


# --- Metrics file ----------------------------------------------------------------------


def test_write_metrics_writes_the_payload_with_provenance_and_time(tmp_path: Path) -> None:
    path = tmp_path / "metrics" / "people" / "result.json"
    payload = {"probe": "people", "n_clips": 2, "recall": 0.9, "per_clip": {"clip-a": {"recall": 1.0}}}
    provenance = {
        "models": ["gemini-4-argon"],
        "prompts": {"people": "people/v1"},
        "seed": 3,
        "repeat": 0,
    }
    before = datetime.now(UTC)
    write_metrics(path, payload, provenance=provenance)
    after = datetime.now(UTC)

    data = load_strict(path)
    assert set(data) == {*payload, "provenance", "written_at"}
    assert {key: data[key] for key in payload} == payload
    # The provenance keys of the caller, and the version of the package.
    assert data["provenance"] == {**provenance, "package_version": pitchvlm.__version__}
    written_at = datetime.fromisoformat(data["written_at"])
    assert written_at.utcoffset() == timedelta(0)
    assert before - timedelta(seconds=1) <= written_at <= after
    assert path.read_text(encoding="utf-8").endswith("}\n")
    # The dicts of the caller are not changed.
    assert "package_version" not in provenance
    assert set(payload) == {"probe", "n_clips", "recall", "per_clip"}


def test_write_metrics_writes_values_that_are_not_finite_as_null(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    payload = {"recall": math.nan, "errors": {"p90": math.inf, "values": [1.5, -math.inf, 2]}, "ok": True}
    write_metrics(path, payload, provenance={})
    data = load_strict(path)
    assert data["recall"] is None
    assert data["errors"] == {"p90": None, "values": [1.5, None, 2]}
    assert data["ok"] is True
    assert data["provenance"] == {"package_version": pitchvlm.__version__}


def test_write_metrics_keeps_a_package_version_of_the_caller_and_replaces_an_old_file(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    write_metrics(path, {"first": 1}, provenance={"package_version": "0.0.1+local"})
    assert load_strict(path)["provenance"] == {"package_version": "0.0.1+local"}
    write_metrics(path, {"second": 2}, provenance={})
    data = load_strict(path)
    assert "first" not in data
    assert data["second"] == 2


@pytest.mark.parametrize("key", ["provenance", "written_at"])
def test_write_metrics_refuses_a_payload_with_one_of_its_keys(tmp_path: Path, key: str) -> None:
    path = tmp_path / "metrics.json"
    with pytest.raises(ValueError, match=key):
        write_metrics(path, {"recall": 1.0, key: "taken"}, provenance={})
    assert not path.exists()


def test_a_probe_result_goes_into_a_valid_json_file(tmp_path: Path) -> None:
    result = run_people_probe([people_clip()], people_backend())
    # One clip has no interval: the ends are nan in the result and null in the file.
    assert math.isnan(result["main_metric"]["low"])
    path = tmp_path / "people.json"
    write_metrics(path, result, provenance={"seed": 0, "repeat": 0})
    data = load_strict(path)
    assert data["main_metric"]["low"] is None
    assert data["main_metric"]["high"] is None
    assert data["recall"] == pytest.approx(19 / 21)
    assert data["counts"] == result["counts"]
    assert data["per_clip"]["HAND-1"]["n_frames"] == 3
    assert data["settings"] == result["settings"]
    assert data["provenance"]["seed"] == 0


def test_wrong_box_with_its_foot_on_the_horizon_is_off_the_pitch() -> None:
    """The label mapping has its horizon at the image row 100: a foot point there has no finite pitch position."""
    mapping = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.01, -1.0]])
    labelled = PersonDet(box=Box(x1=500.0, y1=500.0, x2=540.0, y2=600.0), role="player", team_key="none")
    on_horizon = PersonDet(box=Box(x1=40.0, y1=20.0, x2=60.0, y2=100.0), role="player", team_key="none")
    truth = FramePeople(frame=0, people=(labelled,))
    task = probes._PeopleTask(clip=people_clip(), frames=(0,), truth=(truth,), mappings={0: mapping})

    assert probes._wrong_boxes_off_pitch(task, [FramePeople(frame=0, people=(labelled, on_horizon))]) == 1


# --- Paint alignment of the pitch mappings ---------------------------------------------


def test_auroc_counts_the_pairs_in_order_and_ties_as_half() -> None:
    assert probes.auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert probes.auroc([0.1], [0.9]) == 0.0
    assert probes.auroc([0.5, 0.9], [0.5, 0.1]) == pytest.approx(0.875)
    assert math.isnan(probes.auroc([], [0.1]))
    assert probes.auroc_standard_error(0.875, 2, 2) > 0.0
    assert math.isnan(probes.auroc_standard_error(0.5, 0, 3))


def paint_frame(frame: int, error_m: float, score: float | None) -> probes._MappedFrame:
    """Return a mapped frame whose people are all ``error_m`` off, with a paint score (None: cannot judge)."""
    label_xy = np.array([[0.0, 0.0], [10.0, 5.0], [-20.0, -10.0]])
    paint = PaintAlignment(
        score=score,
        reason=None if score is not None else "too little paint: 0.01% of the frame, the minimum is 0.20%",
        paint_share=0.01,
        chance=0.1,
        n_model_px=500,
        precision=0.5,
        spread=1.0,
    )
    return probes._MappedFrame(
        frame=frame,
        view="midfield",
        valid=True,
        plausible=True,
        mapped_xy=label_xy + np.array([error_m, 0.0]),
        label_xy=label_xy,
        paint=paint,
    )


def test_the_paint_report_separates_good_from_bad_mappings() -> None:
    frames = {
        "A": [paint_frame(0, 0.5, 0.9), paint_frame(1, 2.5, 0.6), paint_frame(2, 6.0, 0.2)],
        "B": [paint_frame(0, 1.0, 0.7), paint_frame(1, 4.0, 0.75), paint_frame(2, 9.0, None)],
    }
    report = probes.paint_report(frames)
    assert (report["n_mappings"], report["n_scored"], report["n_cannot_judge"]) == (6, 5, 1)
    assert report["cannot_judge_reasons"] == {"too little paint": 1}
    separation = report["separation"]
    # Good: 0.9 and 0.7 (errors 0.5 and 1 m). Bad: 0.2 and 0.75 (6 and 4 m). 2.5 m is in neither group.
    assert (separation["n_good"], separation["n_bad"]) == (2, 2)
    assert separation["auroc"] == pytest.approx(0.75)
    assert separation["score_good"]["median"] == pytest.approx(0.8)
    assert report["frames"][5] == {"clip_id": "B", "frame": 2, "score": None, "error_median_m": pytest.approx(9.0)}
    assert_plain(report)


def test_the_pitch_probe_scores_every_valid_mapping_with_the_paint(exact_pitch: dict[str, Any]) -> None:
    paint = exact_pitch["paint_alignment"]
    assert paint["n_mappings"] == exact_pitch["n_frames_valid"] == 12
    assert paint["n_cannot_judge"] == 0
    assert paint["score"]["median"] > 0.9
    # The exact answers give no mapping above 3 m, so the separation has no bad group.
    assert paint["separation"]["n_bad"] == 0
    assert math.isnan(paint["separation"]["auroc"])


def test_a_tool_mapping_that_is_10_m_off_scores_low_with_the_paint(
    synthetic_clip: LabelledClip, scene: SyntheticScene
) -> None:
    def mapping_of(clip_id: str, frame: int) -> HomographyRecord:
        moved = np.array([[1.0, 0.0, 7.07], [0.0, 1.0, 7.07], [0.0, 0.0, 1.0]]) @ scene.homography(frame)
        return HomographyRecord(frame=frame, matrix=matrix_to_rows(moved), source="landmarks")

    result = score_pitch_mappings([synthetic_clip], mapping_of, CALIBRATION, frames_per_clip=12, seed=1)
    paint = result["paint_alignment"]
    assert paint["n_scored"] == 12
    assert paint["score"]["median"] < 0.3
    assert paint["separation"]["n_bad"] == 12
