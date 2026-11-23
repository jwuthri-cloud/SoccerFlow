"""The four model steps: build each request and parse each answer.

A builder returns a stateless request: the prompt text first, then the images.
Two steps also have a video form, for the probe of video input: ``kit_legend_video``
and ``people_video``. Their request sends one video part (``vlm.video``) in
place of the images, and the people answer lists the people of several frames
of the video, each named by its time.
A parser turns the JSON answer of the model into a record of ``schemas.py``.

What a parser does with a bad answer:

- It raises ``ParseError`` when the answer is not a JSON object, or when a
  required part is missing or has the wrong type. The required parts are the
  lists (``people``, ``landmarks``, ``goalkeepers``, ``uncertain``,
  ``per_crop``), the two team kits, ``referee_kit`` and ``confidence``.
- It drops a bad entry of a list and keeps the other entries: a person with an
  unusable box, role or team, a landmark with an unknown id or an unusable
  point, a goalkeeper kit with no colour or an unknown team.
- It turns a shirt number that is not an integer from 0 to 99 into None, also
  inside ``per_crop``, so that the entries stay in the order of the crops. It
  turns a view that is not a known view into None.
- Coordinates outside the image are moved to the nearest edge by ``coords.py``.

Boxes and points follow the coordinate convention of the step settings
(``StepSettings.coords``): the prompt states it, the answer schema asks for
integers or, in the ``xy_unit`` convention, for numbers, and the parser reads
it. A parser must be given the convention of the request whose answer it reads.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import operator
from collections.abc import Sequence
from typing import Any, get_args

import cv2
import numpy as np

from pitchvlm import pitch
from pitchvlm.config import StepSettings
from pitchvlm.schemas import (
    ROLES,
    Box,
    Confidence,
    FrameLandmarks,
    FramePeople,
    GoalkeeperKit,
    KitDescription,
    KitLegend,
    LandmarkObs,
    NumberReading,
    PersonDet,
    PitchView,
    TeamKey,
)
from pitchvlm.vlm.coords import CoordConvention, box_from_answer, point_from_answer
from pitchvlm.vlm.prompts import PROMPT_NAMES, box_format, point_format, render_prompt
from pitchvlm.vlm.types import ImagePart, RequestMeta, Resolution, TextPart, VideoPart, VLMRequest

_TEAM_KEYS: tuple[str, ...] = get_args(TeamKey)
_VIEWS: tuple[str, ...] = get_args(PitchView)
_CONFIDENCES: tuple[str, ...] = get_args(Confidence)
_KIT_PARTS: tuple[str, ...] = ("shirt", "shorts", "socks")
#: How the number prompt names the team of the person in the crops.
_TEAM_PHRASES: dict[str, str] = {"A": "team A", "B": "team B", "none": "an unknown team"}
#: The pitch diagram has large dots and numbers, so a medium resolution is enough.
_DIAGRAM_RESOLUTION: Resolution = "medium"
#: A shirt-number crop is larger than the box of the person by this share of the box size, on every side.
CROP_PADDING = 0.15
#: An entry of a video answer belongs to its target frame when its time is at most this share of
#: the frame interval of the video away from the time of that frame.
VIDEO_TIME_TOLERANCE = 0.5


class ParseError(ValueError):
    """A model answer cannot be turned into a record."""


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    """Encode a BGR ``uint8`` image of shape (height, width, 3) as JPEG bytes."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError(f"expected a uint8 image of shape (height, width, 3), got {image.dtype} {image.shape}")
    if not 1 <= quality <= 100:
        raise ValueError(f"the JPEG quality must be from 1 to 100, got {quality}")
    ok, encoded = cv2.imencode(".jpg", np.ascontiguousarray(image), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("OpenCV could not encode the image as JPEG")
    return encoded.tobytes()


# --- Answer schemas ----------------------------------------------------------
# Each function returns a new dict, so that a request never shares its schema.


def _object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    """Return the schema of an object in which every listed key is required."""
    return {"type": "object", "properties": properties, "required": list(properties)}


def _list_schema(items: dict[str, Any], length: int | None = None) -> dict[str, Any]:
    """Return the schema of a list. ``length`` fixes the number of entries."""
    schema: dict[str, Any] = {"type": "array", "items": items}
    if length is not None:
        schema["minItems"] = length
        schema["maxItems"] = length
    return schema


def _text_schema() -> dict[str, Any]:
    return {"type": "string"}


def _choice_schema(values: Sequence[str]) -> dict[str, Any]:
    return {"type": "string", "enum": list(values)}


def _integer_schema() -> dict[str, Any]:
    return {"type": "integer"}


def _integer_or_null_schema() -> dict[str, Any]:
    return {"type": ["integer", "null"]}


def _kit_schema() -> dict[str, Any]:
    return _object_schema({part: _text_schema() for part in _KIT_PARTS})


def _kit_legend_schema() -> dict[str, Any]:
    goalkeeper = _object_schema({"kit": _text_schema(), "team": _choice_schema(_TEAM_KEYS)})
    return _object_schema(
        {
            "team_a": _kit_schema(),
            "team_b": _kit_schema(),
            "goalkeepers": _list_schema(goalkeeper),
            "referee_kit": _text_schema(),
            "uncertain": _list_schema(_text_schema()),
        }
    )


def _coordinate_schema(convention: CoordConvention) -> dict[str, Any]:
    """Return the schema of one coordinate: a number in the ``xy_unit`` convention, else an integer."""
    return {"type": "number"} if convention == "xy_unit" else _integer_schema()


def _people_schema(convention: CoordConvention) -> dict[str, Any]:
    person = _object_schema(
        {
            "box_2d": _list_schema(_coordinate_schema(convention), length=4),
            "role": _choice_schema(ROLES),
            "team": _choice_schema(_TEAM_KEYS),
            "number": _integer_or_null_schema(),
        }
    )
    return _object_schema({"people": _list_schema(person)})


def _landmarks_schema(convention: CoordConvention) -> dict[str, Any]:
    landmark = _object_schema(
        {"id": _integer_schema(), "point": _list_schema(_coordinate_schema(convention), length=2)}
    )
    return _object_schema({"view": _choice_schema(_VIEWS), "landmarks": _list_schema(landmark)})


def _people_video_schema(convention: CoordConvention, n_targets: int) -> dict[str, Any]:
    """Return the schema of a video people answer: one entry per target frame, with its time and its people."""
    frame = _object_schema(
        {"time_s": {"type": "number"}, "people": _people_schema(convention)["properties"]["people"]}
    )
    return _object_schema({"frames": _list_schema(frame, length=n_targets)})


def _number_schema(n_crops: int) -> dict[str, Any]:
    return _object_schema(
        {
            "per_crop": _list_schema(_integer_or_null_schema(), length=n_crops),
            "number": _integer_or_null_schema(),
            "confidence": _choice_schema(_CONFIDENCES),
        }
    )


# --- Request builders --------------------------------------------------------


def _image_part(image: np.ndarray, resolution: Resolution) -> ImagePart:
    return ImagePart(data=encode_jpeg(image), resolution=resolution)


@functools.cache
def _diagram_jpeg() -> bytes:
    """Return the numbered pitch diagram as JPEG bytes. It never changes, so it is drawn once."""
    return encode_jpeg(pitch.draw_landmark_diagram())


def _request(
    *,
    text: str,
    images: Sequence[ImagePart | VideoPart],
    schema: dict[str, Any],
    meta: RequestMeta,
    settings: StepSettings,
    context: dict[str, Any],
) -> VLMRequest:
    """Assemble a request: the text part first, then the images in their order."""
    return VLMRequest(
        provider=settings.provider,
        model=settings.model,
        thinking_level=settings.thinking_level,
        parts=(TextPart(text=text), *images),
        response_schema=schema,
        prompt_name=PROMPT_NAMES[meta.step],
        meta=meta,
        service_tier=settings.service_tier,
        context=context,
    )


def build_kit_legend_request(
    images: Sequence[np.ndarray], *, clip_id: str, settings: StepSettings, repeat: int = 0
) -> VLMRequest:
    """Build the request for the kit colours of a clip. ``images`` are a few frames of the clip."""
    if len(images) == 0:
        raise ValueError("the kit legend request needs at least one image")
    return _request(
        text=render_prompt("kit_legend", n_images=len(images)),
        images=[_image_part(image, settings.resolution) for image in images],
        schema=_kit_legend_schema(),
        meta=RequestMeta(clip_id=clip_id, step="kit_legend", repeat=repeat),
        settings=settings,
        context={},
    )


def build_people_request(
    image: np.ndarray,
    legend: KitLegend,
    *,
    clip_id: str,
    frame: int,
    settings: StepSettings,
    repeat: int = 0,
    item: str | None = None,
) -> VLMRequest:
    """Build the request for the people in one frame.

    ``item`` names a part of the frame, for example one tile, when a frame is
    sent in several requests. The prompt asks for the boxes in the coordinate
    convention of ``settings``.
    """
    height, width = image.shape[:2]
    box_text = box_format(settings.coords, image="the image", width=width, height=height)
    return _request(
        text=render_prompt("people", legend=legend.prompt_text(), box_format=box_text),
        images=[_image_part(image, settings.resolution)],
        schema=_people_schema(settings.coords),
        meta=RequestMeta(clip_id=clip_id, step="people", frame=frame, item=item, repeat=repeat),
        settings=settings,
        context={},
    )


def build_landmarks_request(
    image: np.ndarray, *, clip_id: str, frame: int, settings: StepSettings, repeat: int = 0
) -> VLMRequest:
    """Build the request for the pitch landmarks in one frame.

    The request has two images: first the numbered pitch diagram, then the frame.
    The prompt asks for the points in the coordinate convention of ``settings``.
    An OpenRouter request sends the diagram at the resolution of the step: the
    chat completions API has no medium resolution.
    """
    height, width = image.shape[:2]
    point_text = point_format(settings.coords, image="image 2", width=width, height=height)
    diagram_resolution = _DIAGRAM_RESOLUTION if settings.provider == "gemini" else settings.resolution
    diagram = ImagePart(data=_diagram_jpeg(), resolution=diagram_resolution)
    return _request(
        text=render_prompt("landmarks", landmark_list=pitch.landmark_legend_text(), point_format=point_text),
        images=[diagram, _image_part(image, settings.resolution)],
        schema=_landmarks_schema(settings.coords),
        meta=RequestMeta(clip_id=clip_id, step="landmarks", frame=frame, repeat=repeat),
        settings=settings,
        context={},
    )


def _video_times(n_video_frames: int, fps: float) -> list[float]:
    """Return the time of each frame of a written video in seconds: ``k / fps`` for frame ``k``."""
    return [index / fps for index in range(n_video_frames)]


def _time_text(seconds: float) -> str:
    """Write the time of a frame for a prompt, e.g. ``1.44 s``."""
    return f"{seconds:.2f} s"


def _check_video(video: VideoPart, n_video_frames: int, fps: float) -> None:
    """Refuse a video part whose file does not hold ``n_video_frames`` frames at ``fps``, or that names a range."""
    if n_video_frames < 1:
        raise ValueError("a video request needs at least one video frame")
    if not 0 < fps < math.inf:
        raise ValueError(f"the rate of a video must be a number above 0, got {fps!r}")
    if video.start_s is not None or video.end_s is not None:
        raise ValueError("the video part of a video step is a whole written file, without start and end")
    if not math.isclose(video.duration_s, n_video_frames / fps, rel_tol=1e-9):
        raise ValueError(
            f"the video part is {video.duration_s:g} s long, but {n_video_frames} frames at {fps:g} frames per "
            f"second are {n_video_frames / fps:g} s"
        )
    if video.fps is not None and not math.isclose(video.fps, fps, rel_tol=1e-9):
        raise ValueError(f"the video part asks for {video.fps:g} frames per second, but its frames are at {fps:g}")


def build_kit_legend_video_request(
    video: VideoPart,
    *,
    clip_id: str,
    video_frames: Sequence[int],
    fps: float,
    settings: StepSettings,
    repeat: int = 0,
) -> VLMRequest:
    """Build the kit legend request of a clip from a video of a few of its frames.

    ``video`` is the file of the frames ``video_frames`` of the clip, written at
    ``fps`` frames per second (``vlm.video.frames_video_part``). The frames go into
    the context, so the cache key and a stored answer name them. The answer is
    read by ``parse_kit_legend``.
    """
    _check_video(video, len(video_frames), fps)
    frames = [operator.index(frame) for frame in video_frames]
    return _request(
        text=render_prompt("kit_legend_video", n_video_frames=len(frames)),
        images=[video],
        schema=_kit_legend_schema(),
        meta=RequestMeta(clip_id=clip_id, step="kit_legend_video", repeat=repeat),
        settings=settings,
        context={"frames": frames},
    )


def build_people_video_request(
    video: VideoPart,
    legend: KitLegend,
    *,
    clip_id: str,
    video_frames: Sequence[int],
    targets: Sequence[int],
    fps: float,
    width: int,
    height: int,
    settings: StepSettings,
    repeat: int = 0,
) -> VLMRequest:
    """Build the request for the people of some frames of a clip, from a video of these frames and their neighbours.

    ``video`` is the file of the frames ``video_frames`` of the clip, in this
    order, written at ``fps`` frames per second: frame ``k`` of the list is at
    ``k / fps`` seconds of the file. ``targets`` are the frames whose people the
    answer must list; each one must be in ``video_frames``. The prompt names each
    target by its time in the file, and the answer gives one entry per target, in
    the order of ``targets``, with that time (``parse_people_video``). ``width``
    and ``height`` are the size of the frames.

    The context holds the targets (``frames``, which a stored answer keeps), all
    written frames (``video_frames``) and the times of the targets (``times_s``).
    The item names the hash of the file, so another file gives another key.
    """
    _check_video(video, len(video_frames), fps)
    written = [operator.index(frame) for frame in video_frames]
    wanted = [operator.index(frame) for frame in targets]
    if not wanted:
        raise ValueError("a video people request needs at least one target frame")
    if len(set(wanted)) != len(wanted):
        raise ValueError("the target frames of a video people request must be different")
    missing = [frame for frame in wanted if frame not in written]
    if missing:
        raise ValueError(f"the target frames {missing} are not frames of the video")
    times = _video_times(len(written), fps)
    target_times = [times[written.index(frame)] for frame in wanted]
    box_text = box_format(settings.coords, image="each frame", width=width, height=height)
    text = render_prompt(
        "people_video",
        n_video_frames=len(written),
        fps_text=f"{fps:g}",
        legend=legend.prompt_text(),
        target_times=", ".join(_time_text(seconds) for seconds in target_times),
        box_format=box_text,
    )
    return _request(
        text=text,
        images=[video],
        schema=_people_video_schema(settings.coords, len(wanted)),
        meta=RequestMeta(
            clip_id=clip_id,
            step="people_video",
            frame=wanted[0],
            item=f"video-{video.sha256[:16]}",
            repeat=repeat,
        ),
        settings=settings,
        context={"frames": wanted, "video_frames": written, "times_s": target_times},
    )


def crop_rect(box: Box, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Return the pixel area of the shirt-number crop of a box: ``(left, top, right, bottom)``.

    ``width`` and ``height`` are the size of the frame. The box gets the
    padding ``CROP_PADDING`` on every side, is widened to whole pixels and is
    cut at the image border. Rows ``top`` to ``bottom - 1`` and columns
    ``left`` to ``right - 1`` of the frame are the crop. The result is None
    when no pixel is left, because the box lies outside the image.

    Every crop of a number request must be cut with this function: the item
    of the request is made from the same pixel areas.
    """
    pad_x, pad_y = CROP_PADDING * box.width, CROP_PADDING * box.height
    left, right = max(math.floor(box.x1 - pad_x), 0), min(math.ceil(box.x2 + pad_x), width)
    top, bottom = max(math.floor(box.y1 - pad_y), 0), min(math.ceil(box.y2 + pad_y), height)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _number_item(
    track_id: int | None, frames: Sequence[int], rects: Sequence[tuple[int, int, int, int]], crop_upscale: float
) -> str:
    """Return the item of a number request: a hash of what its crops show, after the track id when there is one.

    The cache key of a request sees the item and not the pixels. The hash is
    made from the frames, the pixel areas and the enlargement of the crops.
    So a stored answer is used again only for the same crops.

    Without a track id the item is ``crops-<16 hex digits>``: the same crops
    find their stored answer whatever id their track has. With a track id the
    item is ``track-<id>-<8 hex digits>``: the id is part of the cache key,
    so it must be a lasting name of the track.
    """
    facts = {"frames": list(frames), "rects": [list(rect) for rect in rects], "upscale": float(crop_upscale)}
    digest = hashlib.sha256(json.dumps(facts, sort_keys=True).encode("utf-8")).hexdigest()
    if track_id is None:
        return f"crops-{digest[:16]}"
    return f"track-{track_id}-{digest[:8]}"


def build_number_request(
    crops: Sequence[np.ndarray],
    legend: KitLegend,
    team_key: TeamKey,
    *,
    clip_id: str,
    track_id: int | None,
    crop_frames: Sequence[int],
    crop_boxes: Sequence[Box],
    image_size: tuple[int, int],
    crop_upscale: float,
    settings: StepSettings,
    repeat: int = 0,
) -> VLMRequest:
    """Build the request for the shirt number of one track from several crops of it.

    ``crops`` are the pictures that are sent: each one is cut out of its frame
    at ``crop_rect`` of its box and then enlarged by ``crop_upscale``.
    ``crop_frames`` and ``crop_boxes`` say where each crop comes from. They go
    into the request context, not to the model. The context is part of the
    cache key, and fake backends read it. A stored answer keeps the frames
    and not the boxes (``VLMRequest.stored_context``).

    ``track_id`` must be a lasting name of the track, for example the track
    id of a label file, or None. A track id of the pipeline is no lasting
    name: it is a place in the track list, and it changes with the tracking
    code and its settings. So the pipeline gives None.

    The item of the request is ``crops-<16 hex digits>`` without a track id
    and ``track-<id>-<8 hex digits>`` with one. The digits are a hash of the
    crop frames, of the pixel areas of the crops and of the enlargement.
    ``image_size`` is the ``(width, height)`` of the frames; it gives the
    pixel areas (``crop_rect``). The context alone does not tell two
    enlargements apart, and the cache key sees no pixels, so the item does.
    A box that lies outside the image raises ``ValueError``.
    """
    if len(crops) == 0:
        raise ValueError("the number request needs at least one crop")
    if not len(crops) == len(crop_frames) == len(crop_boxes):
        raise ValueError(
            f"got {len(crops)} crops, {len(crop_frames)} crop frames and {len(crop_boxes)} crop boxes: "
            "the three counts must be equal"
        )
    if team_key not in _TEAM_PHRASES:
        raise ValueError(f"unknown team key: {team_key!r}")
    if not 0.0 < crop_upscale < math.inf:
        raise ValueError(f"the enlargement of the crops must be a finite number above 0, got {crop_upscale}")
    width, height = image_size
    frames = [operator.index(frame) for frame in crop_frames]
    rects: list[tuple[int, int, int, int]] = []
    for frame, box in zip(frames, crop_boxes, strict=True):
        rect = crop_rect(box, width, height)
        if rect is None:
            raise ValueError(
                f"the box of the crop of frame {frame} ({box}) lies outside the image of {width} x {height} pixels"
            )
        rects.append(rect)
    context = {"frames": frames, "boxes": [[box.x1, box.y1, box.x2, box.y2] for box in crop_boxes]}
    return _request(
        text=render_prompt(
            "number", n_crops=len(crops), legend=legend.prompt_text(), team=_TEAM_PHRASES[team_key]
        ),
        images=[_image_part(crop, settings.resolution) for crop in crops],
        schema=_number_schema(len(crops)),
        meta=RequestMeta(
            clip_id=clip_id, step="number", item=_number_item(track_id, frames, rects, crop_upscale), repeat=repeat
        ),
        settings=settings,
        context=context,
    )


# --- Answer parsers ----------------------------------------------------------


def _load_object(text: str) -> dict[str, Any]:
    """Read the answer text as a JSON object."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ParseError(f"the answer is not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise ParseError(f"the answer must be a JSON object, got {type(data).__name__}")
    return data


def _list_field(data: dict[str, Any], key: str) -> list[Any]:
    """Return the list under ``key``. The list is required."""
    value = data.get(key)
    if not isinstance(value, list):
        raise ParseError(f'the answer needs a list under "{key}"')
    return value


def _is_finite_number(value: object) -> bool:
    """Say whether a JSON value is a number that can be a coordinate. A truth value is not."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _coordinates(value: object, count: int) -> list[float] | None:
    """Return the value when it is a list of ``count`` numbers, else None."""
    if not isinstance(value, list) or len(value) != count:
        return None
    if not all(_is_finite_number(entry) for entry in value):
        return None
    return value


def _shirt_number(value: object) -> int | None:
    """Return the value when it is an integer from 0 to 99, else None."""
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 99:
        return value
    return None


def _is_landmark_id(value: object) -> bool:
    """Say whether a JSON value is the id of a landmark of the pitch table."""
    return isinstance(value, int) and not isinstance(value, bool) and value in pitch.LANDMARK_BY_ID


def _kit_description(data: dict[str, Any], key: str) -> KitDescription:
    """Read the kit of one team. The kit and its three colour texts are required."""
    kit = data.get(key)
    if not isinstance(kit, dict) or not all(isinstance(kit.get(part), str) for part in _KIT_PARTS):
        raise ParseError(f'the answer needs "{key}" with the texts shirt, shorts and socks')
    return KitDescription(shirt=kit["shirt"], shorts=kit["shorts"], socks=kit["socks"])


def _goalkeeper_kit(item: object) -> GoalkeeperKit | None:
    """Build one goalkeeper kit, or return None when the colour or the team is not usable."""
    if not isinstance(item, dict):
        return None
    kit, team = item.get("kit"), item.get("team")
    if not isinstance(kit, str) or not kit.strip() or team not in _TEAM_KEYS:
        return None
    return GoalkeeperKit(kit=kit, team_key=team)


def _person(item: object, width: int, height: int, coords: CoordConvention) -> PersonDet | None:
    """Build one person, or return None when the box, the role or the team is not usable."""
    if not isinstance(item, dict):
        return None
    coordinates = _coordinates(item.get("box_2d"), 4)
    role, team = item.get("role"), item.get("team")
    if coordinates is None or role not in ROLES or team not in _TEAM_KEYS:
        return None
    box = box_from_answer(coordinates, width, height, coords)
    if box is None:
        return None
    return PersonDet(box=box, role=role, team_key=team, number=_shirt_number(item.get("number")))


def _landmark(item: object, width: int, height: int, coords: CoordConvention) -> LandmarkObs | None:
    """Build one landmark observation, or return None when the id or the point is not usable."""
    if not isinstance(item, dict):
        return None
    landmark_id = item.get("id")
    coordinates = _coordinates(item.get("point"), 2)
    if coordinates is None or not _is_landmark_id(landmark_id):
        return None
    x, y = point_from_answer(coordinates, width, height, coords)
    return LandmarkObs(landmark_id=landmark_id, x=x, y=y)


def parse_kit_legend(text: str) -> KitLegend:
    """Turn the answer of the kit legend request into a ``KitLegend``.

    The answer key ``team`` of a goalkeeper becomes ``team_key``. Goalkeeper
    kits and notes that are not usable are dropped.
    """
    data = _load_object(text)
    referee_kit = data.get("referee_kit")
    if not isinstance(referee_kit, str):
        raise ParseError('the answer needs a text under "referee_kit"')
    goalkeepers = (_goalkeeper_kit(item) for item in _list_field(data, "goalkeepers"))
    notes = (note for note in _list_field(data, "uncertain") if isinstance(note, str) and note.strip())
    return KitLegend(
        team_a=_kit_description(data, "team_a"),
        team_b=_kit_description(data, "team_b"),
        goalkeepers=tuple(keeper for keeper in goalkeepers if keeper is not None),
        referee_kit=referee_kit,
        uncertain=tuple(notes),
    )


def parse_people(text: str, *, frame: int, width: int, height: int, coords: CoordConvention) -> FramePeople:
    """Turn the answer of the people request into a ``FramePeople`` with boxes in pixels.

    ``width`` and ``height`` are the size of the image that was sent, and
    ``coords`` is the coordinate convention of its request. The answer
    key ``team`` becomes ``team_key``. A person with a bad box, role or team is
    dropped. A number that is not an integer from 0 to 99 becomes None.
    """
    data = _load_object(text)
    people = (_person(item, width, height, coords) for item in _list_field(data, "people"))
    return FramePeople(frame=frame, people=tuple(person for person in people if person is not None))


def parse_people_video(
    text: str,
    *,
    targets: Sequence[int],
    times_s: Sequence[float],
    fps: float,
    width: int,
    height: int,
    coords: CoordConvention,
) -> tuple[list[FramePeople], int]:
    """Turn the answer of a video people request into one ``FramePeople`` per target frame.

    ``targets`` and ``times_s`` are the target frames and their times in the video
    (the context of the request), and ``fps`` is the rate of the video. Entry ``k``
    of the answer belongs to target ``k``. Its ``time_s`` must be within
    ``VIDEO_TIME_TOLERANCE`` frame intervals of the time of that target: the model
    then describes the right frame. An entry with another time, a missing time or
    a missing entry gives no people for its target, and it is counted. The result
    is the people of each target and that count. An answer without a ``frames``
    list raises ``ParseError``. The people of an entry are read as in
    ``parse_people``.
    """
    if len(targets) != len(times_s):
        raise ValueError("every target frame needs its time")
    data = _load_object(text)
    entries = _list_field(data, "frames")
    tolerance_s = VIDEO_TIME_TOLERANCE / fps
    answers: list[FramePeople] = []
    n_wrong_time = 0
    for index, (frame, time_s) in enumerate(zip(targets, times_s, strict=True)):
        entry = entries[index] if index < len(entries) else None
        answered_time = entry.get("time_s") if isinstance(entry, dict) else None
        people = entry.get("people") if isinstance(entry, dict) else None
        if not _is_finite_number(answered_time) or abs(answered_time - time_s) > tolerance_s or not isinstance(
            people, list
        ):
            n_wrong_time += 1
            answers.append(FramePeople(frame=frame))
            continue
        found = (_person(item, width, height, coords) for item in people)
        answers.append(FramePeople(frame=frame, people=tuple(person for person in found if person is not None)))
    return answers, n_wrong_time


def parse_landmarks(text: str, *, frame: int, width: int, height: int, coords: CoordConvention) -> FrameLandmarks:
    """Turn the answer of the landmarks request into a ``FrameLandmarks`` with points in pixels.

    ``width`` and ``height`` are the size of the frame that was sent, not of the
    diagram, and ``coords`` is the coordinate convention of its request. A landmark with an unknown id or a bad point is dropped. A view that
    is not a known view becomes None.
    """
    data = _load_object(text)
    view = data.get("view")
    landmarks = (_landmark(item, width, height, coords) for item in _list_field(data, "landmarks"))
    return FrameLandmarks(
        frame=frame,
        view=view if view in _VIEWS else None,
        landmarks=tuple(landmark for landmark in landmarks if landmark is not None),
    )


def parse_number(text: str) -> NumberReading:
    """Turn the answer of the number request into a ``NumberReading``.

    ``per_crop`` keeps one entry for each crop, in order. An entry or a final
    number that is not an integer from 0 to 99 becomes None.
    """
    data = _load_object(text)
    confidence = data.get("confidence")
    if confidence not in _CONFIDENCES:
        raise ParseError(f'the answer needs "confidence" as one of {", ".join(_CONFIDENCES)}')
    return NumberReading(
        per_crop=tuple(_shirt_number(value) for value in _list_field(data, "per_crop")),
        number=_shirt_number(data.get("number")),
        confidence=confidence,
    )
