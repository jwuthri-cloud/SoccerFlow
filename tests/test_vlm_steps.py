"""Tests of the four model steps: request builders, prompt texts and answer parsers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence

import cv2
import numpy as np
import pytest

from pitchvlm import pitch
from pitchvlm.config import StepSettings
from pitchvlm.schemas import (
    Box,
    FrameLandmarks,
    FramePeople,
    GoalkeeperKit,
    KitDescription,
    KitLegend,
    LandmarkObs,
    NumberReading,
)
from pitchvlm.vlm.coords import box_to_model, point_to_model
from pitchvlm.vlm.prompts import PROMPT_NAMES, render_prompt
from pitchvlm.vlm.steps import (
    CROP_PADDING,
    ParseError,
    build_kit_legend_request,
    build_landmarks_request,
    build_number_request,
    build_people_request,
    crop_rect,
    encode_jpeg,
    parse_kit_legend,
    parse_landmarks,
    parse_number,
    parse_people,
)
from pitchvlm.vlm.types import ImagePart, RequestMeta, TextPart, VLMRequest

#: Size of the frame that the parser tests assume. It is wide, so a swapped x and y shows.
WIDTH = 1920
HEIGHT = 1080

LEGEND = KitLegend(
    team_a=KitDescription(shirt="white", shorts="white", socks="white"),
    team_b=KitDescription(shirt="dark red", shorts="black", socks="dark red"),
    goalkeepers=(GoalkeeperKit(kit="neon green", team_key="A"),),
    referee_kit="yellow shirt, black shorts",
)
#: Settings that differ from every default, so that a builder cannot pass by accident.
SETTINGS = StepSettings(
    model="gemini-4-carbon", thinking_level="medium", resolution="ultra_high", service_tier="flex"
)

LEGEND_ANSWER = {
    "team_a": {"shirt": "white", "shorts": "white", "socks": "white"},
    "team_b": {"shirt": "dark red", "shorts": "black", "socks": "dark red"},
    "goalkeepers": [{"kit": "neon green", "team": "A"}, {"kit": "orange", "team": "none"}],
    "referee_kit": "yellow shirt, black shorts",
    "uncertain": ["socks of team B partly hidden"],
}
#: The boxes of one person in two frames of 1920 x 1080 pixels. With 15% of the box size on every
#: side, in whole pixels, their crops are the pixel areas (92, 186, 158, 304) and (294, 188, 346, 292).
NUMBER_BOXES = (Box(x1=100.0, y1=200.0, x2=150.0, y2=290.0), Box(x1=300.0, y1=200.0, x2=340.0, y2=280.0))
NUMBER_RECTS = [[92, 186, 158, 304], [294, 188, 346, 292]]
GOOD_PERSON = {"box_2d": [100, 200, 300, 400], "role": "player", "team": "A", "number": 10}
GOOD_LANDMARK = {"id": 1, "point": [250, 750]}
NUMBER_ANSWER = {"per_crop": [10, None, 10], "number": 10, "confidence": "high"}


def make_image(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Return a BGR test image of random pixels."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def decode(data: bytes) -> np.ndarray:
    """Decode image bytes back into a BGR image."""
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    return image


def text_of(request: VLMRequest) -> str:
    """Return the prompt text of a request. The text must be the first part."""
    first = request.parts[0]
    assert isinstance(first, TextPart)
    return first.text


def images_of(request: VLMRequest) -> list[ImagePart]:
    """Return the image parts of a request. Every part after the text must be an image."""
    images = [part for part in request.parts[1:] if isinstance(part, ImagePart)]
    assert len(images) == len(request.parts) - 1
    return images


def people_request(seed: int = 0, repeat: int = 0, legend: KitLegend = LEGEND) -> VLMRequest:
    """Build a people request for frame 24 of a small test image."""
    return build_people_request(
        make_image(72, 128, seed), legend, clip_id="clip-1", frame=24, settings=SETTINGS, repeat=repeat
    )


def number_request(
    *,
    track_id: int | None = 7,
    crop_frames: Sequence[int] = (10, 22),
    crop_boxes: Sequence[Box] = NUMBER_BOXES,
    image_size: tuple[int, int] = (WIDTH, HEIGHT),
    crop_upscale: float = 2.0,
) -> VLMRequest:
    """Build a number request with two crops of track 7.

    The two pictures have the size of the crops of ``NUMBER_BOXES`` after an enlargement by 2:
    66 x 118 and 52 x 104 pixels give 132 x 236 and 104 x 208 pixels.
    """
    return build_number_request(
        [make_image(236, 132, seed=1), make_image(208, 104, seed=2)],
        LEGEND,
        "B",
        clip_id="clip-1",
        track_id=track_id,
        crop_frames=crop_frames,
        crop_boxes=crop_boxes,
        image_size=image_size,
        crop_upscale=crop_upscale,
        settings=StepSettings(resolution="low"),
        repeat=3,
    )


def all_requests() -> list[VLMRequest]:
    """Build one request of each step."""
    image = make_image(72, 128)
    return [
        build_kit_legend_request([image, image], clip_id="clip-1", settings=SETTINGS),
        people_request(),
        build_landmarks_request(image, clip_id="clip-1", frame=36, settings=SETTINGS),
        number_request(),
    ]


# --- encode_jpeg ----------------------------------------------------------------


def test_encode_jpeg_gives_a_jpeg_of_the_same_size() -> None:
    image = make_image(48, 80)
    data = encode_jpeg(image)

    assert data[:2] == b"\xff\xd8"
    assert data[-2:] == b"\xff\xd9"
    assert decode(data).shape == (48, 80, 3)


def test_encode_jpeg_quality_changes_the_size() -> None:
    image = make_image(48, 80)

    assert len(encode_jpeg(image, quality=30)) < len(encode_jpeg(image, quality=95))


def test_encode_jpeg_has_the_quality_90_by_default() -> None:
    """Every image for the model is encoded with this default, and no fake model looks at the pixels."""
    image = make_image(96, 128)

    assert encode_jpeg(image) == encode_jpeg(image, quality=90)
    assert encode_jpeg(image) != encode_jpeg(image, quality=89)
    # The images of a request are encoded with the default too.
    request = build_people_request(image, LEGEND, clip_id="clip-1", frame=0, settings=SETTINGS)
    assert [part.data for part in images_of(request)] == [encode_jpeg(image, quality=90)]


def test_encode_jpeg_takes_a_slice_of_an_image() -> None:
    crop = make_image(48, 80)[8:40, 16:48]

    assert decode(encode_jpeg(crop)).shape == (32, 32, 3)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((48, 80), dtype=np.uint8),
        np.zeros((48, 80, 4), dtype=np.uint8),
        np.zeros((48, 80, 3), dtype=np.float32),
        np.zeros((0, 80, 3), dtype=np.uint8),
    ],
    ids=["grey", "four channels", "float", "empty"],
)
def test_encode_jpeg_refuses_a_wrong_image(image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        encode_jpeg(image)


@pytest.mark.parametrize("quality", [0, 101])
def test_encode_jpeg_refuses_a_wrong_quality(quality: int) -> None:
    with pytest.raises(ValueError):
        encode_jpeg(make_image(48, 80), quality=quality)


# --- Prompts --------------------------------------------------------------------


def test_prompt_names() -> None:
    assert dict(PROMPT_NAMES) == {
        "kit_legend": "kit_legend/v1",
        "people": "people/v1",
        "landmarks": "landmarks/v1",
        "number": "number/v1",
    }


def test_render_prompt_refuses_an_unknown_step() -> None:
    with pytest.raises(KeyError):
        render_prompt("offside")


def test_render_prompt_needs_every_placeholder() -> None:
    with pytest.raises(KeyError):
        render_prompt("people")


def test_no_placeholder_is_left_in_a_prompt() -> None:
    for request in all_requests():
        text = text_of(request)
        assert text == text.strip()
        assert "{" not in text
        assert "}" not in text


def test_prompts_with_people_forbid_faces_and_names() -> None:
    legend_text, people_text, _, number_text = (text_of(request) for request in all_requests())

    assert "Do not describe faces" in legend_text
    assert "Do not give names" in legend_text
    assert "Do not use faces and do not give names." in people_text
    assert "Do not use faces and do not give names." in number_text


# --- Builders -------------------------------------------------------------------


def test_every_request_has_the_text_first_and_then_the_images() -> None:
    for request, n_images in zip(all_requests(), [2, 1, 2, 2], strict=True):
        assert isinstance(request.parts[0], TextPart)
        assert len(images_of(request)) == n_images
        assert all(part.mime_type == "image/jpeg" for part in images_of(request))


def test_kit_legend_request() -> None:
    images = [make_image(72, 128, seed) for seed in range(3)]
    request = build_kit_legend_request(images, clip_id="clip-1", settings=SETTINGS, repeat=2)

    assert (request.model, request.thinking_level, request.service_tier) == ("gemini-4-carbon", "medium", "flex")
    assert request.prompt_name == "kit_legend/v1"
    assert request.meta == RequestMeta(clip_id="clip-1", step="kit_legend", frame=None, item=None, repeat=2)
    assert request.context == {}
    assert request.resolutions == ("ultra_high", "ultra_high", "ultra_high")
    assert [decode(part.data).shape for part in images_of(request)] == [image.shape for image in images]

    text = text_of(request)
    assert "Number of images: 3." in text
    assert "colour" in text

    schema = request.response_schema
    assert schema["required"] == ["team_a", "team_b", "goalkeepers", "referee_kit", "uncertain"]
    for team in ("team_a", "team_b"):
        assert schema["properties"][team]["required"] == ["shirt", "shorts", "socks"]
    goalkeeper = schema["properties"]["goalkeepers"]["items"]
    assert goalkeeper["required"] == ["kit", "team"]
    assert goalkeeper["properties"]["team"] == {"type": "string", "enum": ["A", "B", "none"]}
    assert schema["properties"]["uncertain"] == {"type": "array", "items": {"type": "string"}}


def test_kit_legend_request_needs_an_image() -> None:
    with pytest.raises(ValueError):
        build_kit_legend_request([], clip_id="clip-1", settings=SETTINGS)


def test_people_request() -> None:
    image = make_image(72, 128)
    request = build_people_request(
        image, LEGEND, clip_id="clip-1", frame=24, settings=SETTINGS, repeat=1, item="tile-2of6"
    )

    assert (request.model, request.thinking_level, request.service_tier) == ("gemini-4-carbon", "medium", "flex")
    assert request.prompt_name == "people/v1"
    assert request.meta == RequestMeta(clip_id="clip-1", step="people", frame=24, item="tile-2of6", repeat=1)
    assert request.context == {}
    assert request.resolutions == ("ultra_high",)
    (part,) = images_of(request)
    assert decode(part.data).shape == image.shape

    text = text_of(request)
    assert LEGEND.prompt_text() in text
    assert "[ymin, xmin, ymax, xmax]" in text
    assert "integer from 0 to 1000" in text
    assert "feet" in text
    assert "null" in text

    schema = request.response_schema
    assert schema["required"] == ["people"]
    person = schema["properties"]["people"]["items"]
    assert person["required"] == ["box_2d", "role", "team", "number"]
    assert person["properties"]["box_2d"] == {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 4,
        "maxItems": 4,
    }
    assert person["properties"]["role"] == {"type": "string", "enum": ["player", "goalkeeper", "referee", "other"]}
    assert person["properties"]["team"] == {"type": "string", "enum": ["A", "B", "none"]}
    assert person["properties"]["number"] == {"type": ["integer", "null"]}


def test_people_request_defaults() -> None:
    request = build_people_request(make_image(72, 128), LEGEND, clip_id="clip-1", frame=0, settings=StepSettings())

    assert request.meta == RequestMeta(clip_id="clip-1", step="people", frame=0, item=None, repeat=0)
    assert (request.model, request.thinking_level, request.service_tier) == ("gemini-4-argon", "low", "standard")
    assert request.resolutions == ("high",)


def test_landmarks_request_sends_the_diagram_and_then_the_frame() -> None:
    image = make_image(72, 128)
    request = build_landmarks_request(image, clip_id="clip-1", frame=36, settings=SETTINGS, repeat=1)

    assert (request.model, request.thinking_level, request.service_tier) == ("gemini-4-carbon", "medium", "flex")
    assert request.prompt_name == "landmarks/v1"
    assert request.meta == RequestMeta(clip_id="clip-1", step="landmarks", frame=36, item=None, repeat=1)
    assert request.context == {}

    # The diagram keeps its own resolution. The frame takes the resolution of the step.
    assert request.resolutions == ("medium", "ultra_high")
    diagram, frame = images_of(request)
    assert diagram.data == encode_jpeg(pitch.draw_landmark_diagram())
    assert decode(diagram.data).shape == pitch.draw_landmark_diagram().shape
    assert decode(frame.data).shape == image.shape

    text = text_of(request)
    assert pitch.landmark_legend_text() in text
    assert "Image 1 is a diagram" in text
    assert "Image 2 is one frame" in text
    assert "[y, x]" in text
    assert "integer from 0 to 1000" in text
    assert "Never extrapolate" in text

    schema = request.response_schema
    assert schema["required"] == ["view", "landmarks"]
    assert schema["properties"]["view"] == {"type": "string", "enum": ["left", "centre", "right"]}
    landmark = schema["properties"]["landmarks"]["items"]
    assert landmark["required"] == ["id", "point"]
    assert landmark["properties"]["id"] == {"type": "integer"}
    assert landmark["properties"]["point"] == {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 2,
        "maxItems": 2,
    }


def test_number_request() -> None:
    request = number_request()

    assert (request.model, request.thinking_level, request.service_tier) == ("gemini-4-argon", "low", "standard")
    assert request.prompt_name == "number/v1"
    # The item names the track and holds 8 hex digits that stand for the crops.
    assert re.fullmatch(r"track-7-[0-9a-f]{8}", request.meta.item or ""), request.meta.item
    assert request.meta == RequestMeta(
        clip_id="clip-1", step="number", frame=None, item=request.meta.item, repeat=3
    )
    assert request.context == {
        "frames": [10, 22],
        "boxes": [[100.0, 200.0, 150.0, 290.0], [300.0, 200.0, 340.0, 280.0]],
    }
    assert json.loads(json.dumps(request.context)) == request.context
    assert request.resolutions == ("low", "low")
    # The crops are sent as they are given: the builder does not cut or enlarge them.
    assert [decode(part.data).shape for part in images_of(request)] == [(236, 132, 3), (208, 104, 3)]

    text = text_of(request)
    assert "Number of images: 2." in text
    assert "exactly 2 entries" in text
    assert LEGEND.prompt_text() in text
    assert "The person belongs to team B." in text
    assert "null" in text

    schema = request.response_schema
    assert schema["required"] == ["per_crop", "number", "confidence"]
    assert schema["properties"]["per_crop"] == {
        "type": "array",
        "items": {"type": ["integer", "null"]},
        "minItems": 2,
        "maxItems": 2,
    }
    assert schema["properties"]["number"] == {"type": ["integer", "null"]}
    assert schema["properties"]["confidence"] == {"type": "string", "enum": ["high", "medium", "low"]}


def test_number_request_item_is_a_hash_of_the_crop_frames_areas_and_enlargement() -> None:
    """The cache key of a request sees its item and not its pixels. So the item must change with the crops."""
    request = number_request()
    # The recipe: the SHA-256 of these facts as JSON with sorted keys, cut to 8 hex digits.
    facts = {"frames": [10, 22], "rects": NUMBER_RECTS, "upscale": 2.0}
    digest = hashlib.sha256(json.dumps(facts, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    assert request.meta.item == f"track-7-{digest}"
    # The same crops give the same item and the same cache key, so that their stored answer is found again.
    assert (number_request().meta.item, number_request().cache_key()) == (request.meta.item, request.cache_key())

    moved_box = Box(x1=500.0, y1=200.0, x2=540.0, y2=280.0)
    other_crops = {
        "another frame": number_request(crop_frames=(10, 23)),
        "another place in the frame": number_request(crop_boxes=(NUMBER_BOXES[0], moved_box)),
        "another enlargement": number_request(crop_upscale=3.0),
        # In a frame of 320 pixels width the border cuts the second crop at x = 320.
        "another image border": number_request(image_size=(320, HEIGHT)),
        "another track": number_request(track_id=8),
    }
    items = {name: other.meta.item for name, other in other_crops.items()}
    assert len({request.meta.item, *items.values()}) == 6, items
    for other in other_crops.values():
        assert other.cache_key() != request.cache_key()
    # The enlargement and the image border are not in the context. Only the item tells these crops apart.
    assert other_crops["another enlargement"].context == request.context
    assert other_crops["another image border"].context == request.context
    # Only the track id differs between these two items: the digits stand for the crops alone.
    assert items["another track"] == f"track-8-{digest}"


def test_number_request_without_a_track_id_names_its_crops_only() -> None:
    """A track id of the pipeline is a place in a list. Without it the same crops always get the same cache key."""
    request = number_request(track_id=None)
    # The recipe: the same SHA-256 as with a track id, cut to 16 hex digits.
    facts = {"frames": [10, 22], "rects": NUMBER_RECTS, "upscale": 2.0}
    digest = hashlib.sha256(json.dumps(facts, sort_keys=True).encode("utf-8")).hexdigest()
    assert request.meta.item == f"crops-{digest[:16]}"
    assert number_request(track_id=7).meta.item == f"track-7-{digest[:8]}"
    assert number_request(track_id=None).cache_key() == request.cache_key()
    assert number_request(track_id=0).cache_key() != request.cache_key()
    # Other crops get another item and another cache key.
    other_crops = [
        number_request(track_id=None, crop_frames=(10, 23)),
        number_request(track_id=None, crop_upscale=3.0),
        number_request(track_id=None, image_size=(320, HEIGHT)),
    ]
    assert len({request.meta.item, *(other.meta.item for other in other_crops)}) == 4
    assert len({request.cache_key(), *(other.cache_key() for other in other_crops)}) == 4


def test_number_request_context_takes_numpy_frame_numbers() -> None:
    def build(frames: Sequence[int]) -> VLMRequest:
        return build_number_request(
            [make_image(236, 132)],
            LEGEND,
            "none",
            clip_id="clip-1",
            track_id=0,
            crop_frames=frames,
            crop_boxes=NUMBER_BOXES[:1],
            image_size=(WIDTH, HEIGHT),
            crop_upscale=2.0,
            settings=StepSettings(),
        )

    request = build(np.array([12]))

    assert request.context["frames"] == [12]
    assert type(request.context["frames"][0]) is int
    assert request.meta.item == build([12]).meta.item
    assert "The person belongs to an unknown team." in text_of(request)


def test_number_request_refuses_wrong_input() -> None:
    crop = make_image(236, 132)
    box = NUMBER_BOXES[0]
    outside = Box(x1=2000.0, y1=200.0, x2=2050.0, y2=290.0)

    def build(
        crops: list[np.ndarray], team_key: str, frames: list[int], boxes: list[Box], crop_upscale: float = 2.0
    ) -> VLMRequest:
        return build_number_request(
            crops,
            LEGEND,
            team_key,  # type: ignore[arg-type]
            clip_id="clip-1",
            track_id=1,
            crop_frames=frames,
            crop_boxes=boxes,
            image_size=(WIDTH, HEIGHT),
            crop_upscale=crop_upscale,
            settings=SETTINGS,
        )

    # The same call with right input works, so each error below comes from the wrong input.
    assert (build([crop], "A", [1], [box]).meta.item or "").startswith("track-1-")
    with pytest.raises(ValueError):
        build([], "A", [], [])  # no crop
    with pytest.raises(ValueError):
        build([crop, crop], "A", [1], [box, box])  # one frame for two crops
    with pytest.raises(ValueError):
        build([crop], "A", [1], [])  # no box for the crop
    with pytest.raises(ValueError):
        build([crop], "C", [1], [box])  # unknown team key
    with pytest.raises(ValueError, match="outside the image of 1920 x 1080 pixels"):
        build([crop], "A", [1], [outside])  # the box has no crop
    for wrong_upscale in (0.0, -2.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="enlargement"):
            build([crop], "A", [1], [box], crop_upscale=wrong_upscale)


def test_crop_rect_is_left_top_right_bottom_inside_the_image() -> None:
    assert CROP_PADDING == 0.15
    # 15% of 50 is 7.5 and 15% of 90 is 13.5: the crop grows to the next whole pixel on every side.
    assert crop_rect(Box(x1=100.0, y1=200.0, x2=150.0, y2=290.0), 1280, 720) == (92, 186, 158, 304)
    # A box at the left and the top border of the image: the crop is cut at 0.
    assert crop_rect(Box(x1=-10.0, y1=3.0, x2=40.0, y2=93.0), 1280, 720) == (0, 0, 48, 107)
    # A box at the right and the bottom border: the crop is cut at the width and at the height.
    assert crop_rect(Box(x1=1250.0, y1=650.0, x2=1300.0, y2=740.0), 1280, 720) == (1242, 636, 1280, 720)
    # A box outside the image has no crop.
    assert crop_rect(Box(x1=1400.0, y1=100.0, x2=1450.0, y2=190.0), 1280, 720) is None
    # The crops of the number request of these tests.
    assert [list(crop_rect(box, WIDTH, HEIGHT) or ()) for box in NUMBER_BOXES] == NUMBER_RECTS


@pytest.mark.parametrize(
    ("box", "step_into_the_image", "one_pixel_crop"),
    [
        pytest.param(Box(x1=-58.0, y1=200.0, x2=-8.0, y2=290.0), (1.0, 0.0), (0, 186, 1, 304), id="left"),
        pytest.param(Box(x1=1288.0, y1=200.0, x2=1338.0, y2=290.0), (-1.0, 0.0), (1279, 186, 1280, 304), id="right"),
        pytest.param(Box(x1=100.0, y1=-104.0, x2=150.0, y2=-14.0), (0.0, 1.0), (92, 0, 158, 1), id="above"),
        pytest.param(Box(x1=100.0, y1=734.0, x2=150.0, y2=824.0), (0.0, -1.0), (92, 719, 158, 720), id="below"),
    ],
)
def test_crop_rect_is_none_for_a_crop_without_a_column_or_without_a_row(
    box: Box, step_into_the_image: tuple[float, float], one_pixel_crop: tuple[int, int, int, int]
) -> None:
    """The padded box ends half a pixel outside an image of 1280 x 720 pixels, so its crop has a width or a height of 0.

    The same box one pixel further inside gives a crop of one column or of one row.
    """
    assert crop_rect(box, 1280, 720) is None
    step_x, step_y = step_into_the_image
    moved = Box(x1=box.x1 + step_x, y1=box.y1 + step_y, x2=box.x2 + step_x, y2=box.y2 + step_y)
    assert crop_rect(moved, 1280, 720) == one_pixel_crop


def test_each_request_gets_its_own_schema() -> None:
    first = people_request()
    first.response_schema["properties"]["people"]["items"]["required"].append("extra")

    second = people_request()
    assert "extra" not in second.response_schema["properties"]["people"]["items"]["required"]


def test_cache_key_follows_the_question_and_not_the_pixels() -> None:
    first = people_request(seed=1)
    same = people_request(seed=1)
    other_pixels = people_request(seed=2)

    assert same.cache_key() == first.cache_key()
    assert same.image_hashes == first.image_hashes
    # Other pixels give another image hash, but the same key.
    assert other_pixels.image_hashes != first.image_hashes
    assert other_pixels.cache_key() == first.cache_key()
    # A repeat run and a changed prompt text ask a new question.
    assert people_request(seed=1, repeat=1).cache_key() != first.cache_key()
    other_legend = LEGEND.model_copy(update={"referee_kit": "pink"})
    assert people_request(seed=1, legend=other_legend).cache_key() != first.cache_key()


# --- Parsers: invalid JSON ------------------------------------------------------

PARSERS: dict[str, Callable[[str], object]] = {
    "kit_legend": parse_kit_legend,
    "people": lambda text: parse_people(text, frame=0, width=WIDTH, height=HEIGHT, coords="yx_1000"),
    "landmarks": lambda text: parse_landmarks(text, frame=0, width=WIDTH, height=HEIGHT, coords="yx_1000"),
    "number": parse_number,
}


def test_parse_error_is_a_value_error() -> None:
    assert issubclass(ParseError, ValueError)


@pytest.mark.parametrize("parse", list(PARSERS.values()), ids=list(PARSERS))
@pytest.mark.parametrize(
    "text",
    ["", "no json here", '{"people": [', "```json\n{}\n```", "[1, 2, 3]", "null", '"text"', "12"],
    ids=["empty", "words", "cut off", "code fence", "list", "null", "string", "number"],
)
def test_answer_that_is_not_a_json_object_raises_parse_error(parse: Callable[[str], object], text: str) -> None:
    with pytest.raises(ParseError):
        parse(text)


@pytest.mark.parametrize("parse", list(PARSERS.values()), ids=list(PARSERS))
def test_empty_object_raises_parse_error(parse: Callable[[str], object]) -> None:
    with pytest.raises(ParseError):
        parse("{}")


# --- parse_kit_legend -----------------------------------------------------------


def test_parse_kit_legend_maps_team_to_team_key() -> None:
    legend = parse_kit_legend(json.dumps(LEGEND_ANSWER))

    assert legend == KitLegend(
        team_a=KitDescription(shirt="white", shorts="white", socks="white"),
        team_b=KitDescription(shirt="dark red", shorts="black", socks="dark red"),
        goalkeepers=(GoalkeeperKit(kit="neon green", team_key="A"), GoalkeeperKit(kit="orange", team_key="none")),
        referee_kit="yellow shirt, black shorts",
        uncertain=("socks of team B partly hidden",),
    )


def test_parse_kit_legend_drops_bad_list_entries() -> None:
    answer = {
        **LEGEND_ANSWER,
        "goalkeepers": [
            {"kit": "neon green", "team": "A"},
            {"kit": "pink", "team": "C"},
            {"kit": "", "team": "B"},
            {"kit": 5, "team": "B"},
            {"team": "B"},
            "yellow",
            None,
            {"kit": "orange", "team": "B"},
        ],
        "uncertain": ["shorts of team A", 3, None, "", ["nested"]],
    }
    legend = parse_kit_legend(json.dumps(answer))

    assert legend.goalkeepers == (
        GoalkeeperKit(kit="neon green", team_key="A"),
        GoalkeeperKit(kit="orange", team_key="B"),
    )
    assert legend.uncertain == ("shorts of team A",)


def test_parse_kit_legend_takes_empty_lists_and_referee_kit() -> None:
    answer = {**LEGEND_ANSWER, "goalkeepers": [], "referee_kit": "", "uncertain": []}
    legend = parse_kit_legend(json.dumps(answer))

    assert legend.goalkeepers == ()
    assert legend.referee_kit == ""
    assert legend.uncertain == ()


@pytest.mark.parametrize("key", list(LEGEND_ANSWER))
def test_parse_kit_legend_needs_every_part(key: str) -> None:
    answer = {name: value for name, value in LEGEND_ANSWER.items() if name != key}

    with pytest.raises(ParseError):
        parse_kit_legend(json.dumps(answer))


@pytest.mark.parametrize(
    "change",
    [
        {"team_a": {"shirt": "white", "shorts": "white"}},
        {"team_b": {"shirt": "white", "shorts": "white", "socks": 3}},
        {"team_a": "white"},
        {"goalkeepers": {"kit": "green", "team": "A"}},
        {"referee_kit": None},
        {"uncertain": "nothing"},
    ],
    ids=["no socks", "socks not text", "kit not an object", "goalkeepers not a list", "no referee text", "notes"],
)
def test_parse_kit_legend_refuses_a_part_of_the_wrong_type(change: dict[str, object]) -> None:
    with pytest.raises(ParseError):
        parse_kit_legend(json.dumps({**LEGEND_ANSWER, **change}))


# --- parse_people ---------------------------------------------------------------


def test_parse_people_reads_boxes_as_ymin_xmin_ymax_xmax() -> None:
    answer = {
        "people": [
            GOOD_PERSON,
            {"box_2d": [500, 600, 700, 650], "role": "referee", "team": "none", "number": None},
        ]
    }
    result = parse_people(json.dumps(answer), frame=7, width=WIDTH, height=HEIGHT, coords="yx_1000")

    assert isinstance(result, FramePeople)
    assert result.frame == 7
    assert isinstance(result.people, tuple)
    player, referee = result.people
    # [ymin 100, xmin 200, ymax 300, xmax 400] of a 1920 x 1080 image.
    assert (player.box.x1, player.box.y1, player.box.x2, player.box.y2) == pytest.approx((384.0, 108.0, 768.0, 324.0))
    assert (player.role, player.team_key, player.number) == ("player", "A", 10)
    assert (referee.box.x1, referee.box.y1, referee.box.x2, referee.box.y2) == pytest.approx(
        (1152.0, 540.0, 1248.0, 756.0)
    )
    assert (referee.role, referee.team_key, referee.number) == ("referee", "none", None)


def test_parse_people_drops_bad_people_and_keeps_the_rest() -> None:
    last = {"box_2d": [500, 600, 700, 650], "role": "goalkeeper", "team": "B", "number": 1}
    people = [
        GOOD_PERSON,
        {**GOOD_PERSON, "box_2d": [300, 200, 100, 400]},  # ymax above ymin
        {**GOOD_PERSON, "box_2d": [100, 200, 100, 400]},  # no height
        {**GOOD_PERSON, "box_2d": [100, 200, 300]},  # three values
        {**GOOD_PERSON, "box_2d": [100, "200", 300, 400]},  # a text in the box
        {**GOOD_PERSON, "box_2d": [100, True, 300, 400]},  # a truth value in the box
        {**GOOD_PERSON, "box_2d": [float("nan"), 200, 300, 400]},
        {**GOOD_PERSON, "box_2d": None},
        {**GOOD_PERSON, "role": "coach"},
        {**GOOD_PERSON, "role": None},
        {**GOOD_PERSON, "team": "C"},
        {**GOOD_PERSON, "team": None},
        {"role": "player", "team": "A", "number": 10},  # no box
        "player",
        None,
        [100, 200, 300, 400],
        last,
    ]
    result = parse_people(json.dumps({"people": people}), frame=7, width=WIDTH, height=HEIGHT, coords="yx_1000")

    assert [(person.role, person.team_key, person.number) for person in result.people] == [
        ("player", "A", 10),
        ("goalkeeper", "B", 1),
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(0, 0), (10, 10), (99, 99), (100, None), (-1, None), (None, None), ("10", None), (True, None), (7.5, None)],
)
def test_parse_people_keeps_a_number_only_from_0_to_99(raw: object, expected: int | None) -> None:
    answer = {"people": [{**GOOD_PERSON, "number": raw}]}
    (person,) = parse_people(json.dumps(answer), frame=0, width=WIDTH, height=HEIGHT, coords="yx_1000").people

    assert person.number == expected
    assert person.number is None or type(person.number) is int


def test_parse_people_keeps_a_person_without_a_number_key() -> None:
    answer = {"people": [{"box_2d": [100, 200, 300, 400], "role": "other", "team": "none"}]}
    (person,) = parse_people(json.dumps(answer), frame=0, width=WIDTH, height=HEIGHT, coords="yx_1000").people

    assert (person.role, person.team_key, person.number) == ("other", "none", None)


def test_parse_people_moves_coordinates_into_the_image() -> None:
    answer = {"people": [{**GOOD_PERSON, "box_2d": [-10, -10, 1100, 500]}]}
    (person,) = parse_people(json.dumps(answer), frame=0, width=WIDTH, height=HEIGHT, coords="yx_1000").people

    assert person.box == Box(x1=0.0, y1=0.0, x2=960.0, y2=1080.0)


def test_parse_people_takes_an_empty_list() -> None:
    assert parse_people('{"people": []}', frame=3, width=WIDTH, height=HEIGHT, coords="yx_1000") == FramePeople(frame=3)


@pytest.mark.parametrize("answer", ['{"people": null}', '{"people": {"box_2d": [0, 0, 10, 10]}}', '{"persons": []}'])
def test_parse_people_needs_the_people_list(answer: str) -> None:
    with pytest.raises(ParseError):
        parse_people(answer, frame=0, width=WIDTH, height=HEIGHT, coords="yx_1000")


def test_parse_people_reads_back_boxes_of_box_to_model() -> None:
    boxes = [Box(x1=100.0, y1=400.0, x2=140.0, y2=500.0), Box(x1=1500.0, y1=200.0, x2=1530.0, y2=270.0)]
    people = [
        {"box_2d": box_to_model(box, WIDTH, HEIGHT), "role": "player", "team": "B", "number": None} for box in boxes
    ]
    result = parse_people(json.dumps({"people": people}), frame=5, width=WIDTH, height=HEIGHT, coords="yx_1000")

    assert len(result.people) == 2
    for person, box in zip(result.people, boxes, strict=True):
        assert person.box.iou(box) > 0.9
        # The foot point is the bottom centre: it moves by less than one pixel.
        assert person.box.foot == pytest.approx(box.foot, abs=1.0)


# --- parse_landmarks ------------------------------------------------------------


def test_parse_landmarks_reads_points_as_y_then_x() -> None:
    answer = {"view": "left", "landmarks": [GOOD_LANDMARK, {"id": 18, "point": [500, 500]}]}
    result = parse_landmarks(json.dumps(answer), frame=3, width=WIDTH, height=HEIGHT, coords="yx_1000")

    assert isinstance(result, FrameLandmarks)
    assert result.frame == 3
    assert result.view == "left"
    assert isinstance(result.landmarks, tuple)
    corner, centre = result.landmarks
    # [y 250, x 750] of a 1920 x 1080 image.
    assert (corner.landmark_id, corner.x, corner.y) == (1, pytest.approx(1440.0), pytest.approx(270.0))
    assert (centre.landmark_id, centre.x, centre.y) == (18, pytest.approx(960.0), pytest.approx(540.0))


def test_parse_landmarks_drops_bad_landmarks_and_keeps_the_rest() -> None:
    landmarks = [
        GOOD_LANDMARK,
        {"id": 0, "point": [100, 100]},  # ids start at 1
        {"id": 36, "point": [100, 100]},  # the table ends at 35
        {"id": "3", "point": [100, 100]},
        {"id": True, "point": [100, 100]},
        {"id": 2.0, "point": [100, 100]},
        {"id": None, "point": [100, 100]},
        {"id": [4], "point": [100, 100]},
        {"point": [100, 100]},
        {"id": 5, "point": [100, 100, 100]},
        {"id": 6, "point": [100]},
        {"id": 7, "point": ["100", 100]},
        {"id": 8, "point": [float("nan"), 100]},
        {"id": 9, "point": None},
        {"id": 10},
        "corner",
        None,
        {"id": 35, "point": [900, 100]},
    ]
    answer = {"view": "right", "landmarks": landmarks}
    result = parse_landmarks(json.dumps(answer), frame=0, width=WIDTH, height=HEIGHT, coords="yx_1000")

    assert [landmark.landmark_id for landmark in result.landmarks] == [1, 35]
    assert all(landmark.landmark_id in pitch.LANDMARK_BY_ID for landmark in result.landmarks)


def test_parse_landmarks_keeps_every_id_of_the_pitch_table() -> None:
    answer = {"view": "centre", "landmarks": [{"id": landmark.id, "point": [500, 500]} for landmark in pitch.LANDMARKS]}
    result = parse_landmarks(json.dumps(answer), frame=0, width=WIDTH, height=HEIGHT, coords="yx_1000")

    assert [landmark.landmark_id for landmark in result.landmarks] == list(range(1, 36))


def test_parse_landmarks_moves_coordinates_into_the_image() -> None:
    answer = {"view": "centre", "landmarks": [{"id": 18, "point": [1200, -5]}]}
    (landmark,) = parse_landmarks(json.dumps(answer), frame=0, width=WIDTH, height=HEIGHT, coords="yx_1000").landmarks

    assert landmark == LandmarkObs(landmark_id=18, x=0.0, y=float(HEIGHT))


@pytest.mark.parametrize(
    ("view", "expected"),
    [("left", "left"), ("centre", "centre"), ("right", "right"), ("middle", None), (None, None), (3, None), ([], None)],
)
def test_parse_landmarks_keeps_only_a_known_view(view: object, expected: str | None) -> None:
    answer = {"view": view, "landmarks": [GOOD_LANDMARK]}
    result = parse_landmarks(json.dumps(answer), frame=0, width=WIDTH, height=HEIGHT, coords="yx_1000")

    assert result.view == expected
    assert len(result.landmarks) == 1


def test_parse_landmarks_takes_an_answer_without_a_view() -> None:
    result = parse_landmarks('{"landmarks": []}', frame=4, width=WIDTH, height=HEIGHT, coords="yx_1000")

    assert result == FrameLandmarks(frame=4)


@pytest.mark.parametrize("answer", ['{"view": "left"}', '{"view": "left", "landmarks": null}'])
def test_parse_landmarks_needs_the_landmarks_list(answer: str) -> None:
    with pytest.raises(ParseError):
        parse_landmarks(answer, frame=0, width=WIDTH, height=HEIGHT, coords="yx_1000")


def test_parse_landmarks_reads_back_points_of_point_to_model() -> None:
    points = {1: (123.0, 45.0), 18: (960.0, 540.0), 35: (1900.0, 1070.0)}
    landmarks = [{"id": key, "point": point_to_model(x, y, WIDTH, HEIGHT)} for key, (x, y) in points.items()]
    answer = {"view": "centre", "landmarks": landmarks}
    result = parse_landmarks(json.dumps(answer), frame=0, width=WIDTH, height=HEIGHT, coords="yx_1000")

    assert len(result.landmarks) == 3
    for landmark in result.landmarks:
        x, y = points[landmark.landmark_id]
        assert landmark.x == pytest.approx(x, abs=1.0)
        assert landmark.y == pytest.approx(y, abs=1.0)


# --- parse_number ---------------------------------------------------------------


def test_parse_number_keeps_the_order_of_the_crops() -> None:
    reading = parse_number(json.dumps(NUMBER_ANSWER))

    assert reading == NumberReading(per_crop=(10, None, 10), number=10, confidence="high")


def test_parse_number_takes_an_answer_with_no_number() -> None:
    reading = parse_number('{"per_crop": [null, null], "number": null, "confidence": "low"}')

    assert reading == NumberReading(per_crop=(None, None), number=None, confidence="low")


def test_parse_number_turns_bad_values_into_none() -> None:
    answer = {"per_crop": [10, 123, "7", None, True, -4, 9.5, 99, 0], "number": 150, "confidence": "medium"}
    reading = parse_number(json.dumps(answer))

    # A bad entry becomes None. It is not removed, so entry i still belongs to crop i.
    assert reading.per_crop == (10, None, None, None, None, None, None, 99, 0)
    assert reading.number is None
    assert reading.confidence == "medium"


@pytest.mark.parametrize(
    "change",
    [{"confidence": "sure"}, {"confidence": None}, {"per_crop": None}, {"per_crop": 10}],
    ids=["unknown confidence", "no confidence", "no list", "list is a number"],
)
def test_parse_number_refuses_a_part_of_the_wrong_type(change: dict[str, object]) -> None:
    with pytest.raises(ParseError):
        parse_number(json.dumps({**NUMBER_ANSWER, **change}))


@pytest.mark.parametrize("key", ["per_crop", "confidence"])
def test_parse_number_needs_the_list_and_the_confidence(key: str) -> None:
    answer = {name: value for name, value in NUMBER_ANSWER.items() if name != key}

    with pytest.raises(ParseError):
        parse_number(json.dumps(answer))
