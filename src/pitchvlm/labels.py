"""Reader for SoccerNet-GSR label files (``Labels-GameState.json``).

One file holds the labels of one clip. ``load_gsr_labels`` turns it into
frozen records. ``gsr_frame_states`` and ``gsr_frame_people`` give the same
labels in the internal schema, so that ground truth and pipeline output go
through the same export and metric code.

Frame index: the images are put in the order of the numbers in their file
names, and ``000001.jpg`` is frame 0. This is the order in which
``frames.ImageFolderSource`` reads the frame files. Image boxes are in pixels.
Pitch positions are in metres, with a negative y on the far side of the pitch.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import model_validator

from pitchvlm.schemas import (
    ROLES,
    TEAM_ROLES,
    Box,
    FramePeople,
    FrameState,
    PersonDet,
    PlayerState,
    Record,
    Role,
    TeamKey,
    TeamSide,
)

#: Role of the ball in the label files. The ball is not an athlete, so it is skipped.
BALL_ROLE = "ball"
#: Image flags that must all be true for a frame to be scored.
SCORED_FLAGS = ("has_labeled_person", "has_labeled_pitch", "has_labeled_camera")
#: Team key that stands for each labelled team side in ``gsr_frame_people``.
TEAM_KEY_OF_SIDE: dict[TeamSide, TeamKey] = {"left": "A", "right": "B"}

#: Pitch lines of one frame: line class -> points ``(x, y)``, normalised to 0-1 of the image size.
Lines = dict[str, tuple[tuple[float, float], ...]]


class GsrAthlete(Record):
    """One labelled person in one frame."""

    track_id: int
    #: Image box, or None when the label has no ``bbox_image``.
    box: Box | None
    #: Pitch position in metres, or None when the label has no ``bbox_pitch``.
    pitch_x: float | None
    pitch_y: float | None
    #: Role text of the label: ``player``, ``goalkeeper``, ``referee`` or ``other``.
    role: str
    #: ``left``, ``right`` or None.
    team: str | None
    #: Shirt number as an integer, or None.
    jersey: int | None


class GsrFrame(Record):
    """The labels of one frame."""

    frame: int
    image_id: str
    file_name: str
    #: True when the scorer counts this frame: it has person, pitch and camera labels.
    scored: bool
    athletes: tuple[GsrAthlete, ...]
    lines: Lines


class GsrLabels(Record):
    """The labels of one clip."""

    clip_id: str
    fps: float
    n_frames: int
    width: int
    height: int
    frames: tuple[GsrFrame, ...]
    #: True when the file stores shirt numbers as text. Predictions must use the same type.
    jersey_is_str: bool
    version: str | None

    @model_validator(mode="after")
    def _check_frame_order(self) -> GsrLabels:
        if [frame.frame for frame in self.frames] != list(range(len(self.frames))):
            raise ValueError("frames must be in frame order: 0, 1, 2, ...")
        return self


def load_gsr_labels(path: Path) -> GsrLabels:
    """Read the label file of one clip.

    - The frame index is the position of the image in the image list, put in
      the order of the numbers in the file names. So ``000001.jpg`` is frame 0,
      and ``2.jpg`` comes before ``10.jpg``. The name of an image file must be
      a whole number, and no two images may have the same number: this is the
      rule of ``frames.ImageFolderSource`` for the frame files, so a label and
      its picture get the same frame index.
    - ``clip_id`` is ``info.name``, or the name of the folder of the file when
      the name is missing. ``n_frames`` is the number of images; when
      ``info.seq_length`` is there, it must be the same number. The frame rate
      must be above zero. Width and height come from the first image.
    - A frame is scored only when its three ``has_labeled_*`` flags are true.
    - Annotations with the role ``ball`` are skipped. Annotations that are
      neither an object nor the pitch (the camera) are not read.
    - A track id must be a whole number, and an image must not hold two
      athletes with the same track id.
    - For ``jersey``, ``team``, ``bbox_image`` and ``bbox_pitch`` a missing key
      means the same as null: no number, no team, no box, no pitch position.
      A pitch position that is there must be finite.
    - A shirt number may be a string of digits or a whole number. It is stored
      as an integer. ``jersey_is_str`` is True when at least one shirt number
      in the file is a string.

    Any other content raises ``ValueError`` with the path and the place of the
    problem.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        return _parse_labels(json.loads(text), folder_name=path.resolve().parent.name)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def gsr_frame_states(labels: GsrLabels) -> list[FrameState]:
    """Return the ground truth of every frame in the internal schema, one state per frame.

    Athletes without a pitch position are skipped. The team side comes from the
    label team for players and goalkeepers. The export number comes from the
    shirt number for players only. The team key is ``none`` because labels
    have no kit legend. Every state is marked as observed.

    ``ValueError`` is raised for a role outside the four roles of the schema,
    for a player or goalkeeper whose team is neither ``left`` nor ``right``
    nor null, for a player whose shirt number is above 99 (the schema holds 0
    to 99), and for a pitch position that is not a finite number. The shirt
    number of another role is not used, so it raises no error here.
    """
    return [
        FrameState(
            frame=frame.frame,
            players=tuple(
                _player_state(athlete, athlete.pitch_x, athlete.pitch_y, frame.frame)
                for athlete in frame.athletes
                if athlete.pitch_x is not None and athlete.pitch_y is not None
            ),
        )
        for frame in labels.frames
    ]


def gsr_frame_people(labels: GsrLabels, frame: int) -> FramePeople:
    """Return the labelled people of one frame in the schema of the people step.

    Athletes without an image box are skipped. ``FramePeople`` has no field
    for the team side, so the label team is carried in the team key: ``left``
    is ``A``, ``right`` is ``B``, and people without a team get ``none``. The
    number is the shirt number of the label, for every role that has one.

    ``ValueError`` is raised for a frame outside the clip, for a role outside
    the four roles of the schema, and for a shirt number above 99 (the people
    schema holds 0 to 99).
    """
    if not 0 <= frame < len(labels.frames):
        raise ValueError(f"frame {frame} is not in the labels: they hold frames 0 to {len(labels.frames) - 1}")
    people = [
        _person(athlete, athlete.box, frame) for athlete in labels.frames[frame].athletes if athlete.box is not None
    ]
    return FramePeople(frame=frame, people=tuple(people))


# --- Internal schema -----------------------------------------------------------


def _player_state(athlete: GsrAthlete, pitch_x: float, pitch_y: float, frame: int) -> PlayerState:
    """Turn a labelled athlete with a pitch position into a player state."""
    role = _role(athlete, frame)
    return PlayerState(
        track_id=athlete.track_id,
        box=athlete.box,
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        role=role,
        team_key="none",
        team_side=_team_side(athlete, role, frame),
        number_export=athlete.jersey if role == "player" else None,
        observed=True,
    )


def _person(athlete: GsrAthlete, box: Box, frame: int) -> PersonDet:
    """Turn a labelled athlete with an image box into a detection of the people step."""
    role = _role(athlete, frame)
    side = _team_side(athlete, role, frame)
    team_key: TeamKey = "none" if side is None else TEAM_KEY_OF_SIDE[side]
    return PersonDet(box=box, role=role, team_key=team_key, number=athlete.jersey)


def _role(athlete: GsrAthlete, frame: int) -> Role:
    """Return the role of an athlete as one of the four roles of the internal schema."""
    for role in ROLES:
        if athlete.role == role:
            return role
    raise ValueError(f"frame {frame}, track {athlete.track_id}: unknown role {athlete.role!r}")


def _team_side(athlete: GsrAthlete, role: Role, frame: int) -> TeamSide | None:
    """Return the labelled team side of a player or goalkeeper. Other roles have no side."""
    if role not in TEAM_ROLES or athlete.team is None:
        return None
    if athlete.team == "left":
        return "left"
    if athlete.team == "right":
        return "right"
    raise ValueError(f"frame {frame}, track {athlete.track_id}: unknown team {athlete.team!r}")


# --- File content ----------------------------------------------------------------


def _parse_labels(data: Any, folder_name: str) -> GsrLabels:
    """Turn the content of a label file into records."""
    info = _required(data, "info", "label file")
    fps = _frame_rate(info)
    images = _images_in_frame_order(_required(data, "images", "label file"))
    if not images:
        raise ValueError("the label file lists no images")
    seq_length = info.get("seq_length")
    if seq_length is not None and _whole_number(seq_length, "info", "seq_length") != len(images):
        raise ValueError(f"info: seq_length is {seq_length!r}, but the label file lists {len(images)} images")
    image_ids = [_image_id(image) for image in images]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("two images have the same image_id")
    annotations = _required(data, "annotations", "label file")
    athletes, lines = _annotations_by_image(annotations, frozenset(image_ids))
    frames = tuple(
        GsrFrame(
            frame=index,
            image_id=image_id,
            file_name=str(image["file_name"]),
            scored=_is_scored(image),
            athletes=tuple(athletes.get(image_id, ())),
            lines=lines.get(image_id, {}),
        )
        for index, (image, image_id) in enumerate(zip(images, image_ids))
    )
    name = info.get("name")
    version = info.get("version")
    return GsrLabels(
        clip_id=folder_name if name is None else str(name),
        fps=fps,
        n_frames=len(images),
        width=int(_required(images[0], "width", "image")),
        height=int(_required(images[0], "height", "image")),
        frames=frames,
        jersey_is_str=_jersey_is_str(annotations),
        version=None if version is None else str(version),
    )


def _required(mapping: Any, key: str, where: str) -> Any:
    """Return ``mapping[key]``. A missing key or a null value raises ``ValueError``."""
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{where}: expected a mapping, found {type(mapping).__name__}")
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"{where}: the key '{key}' is missing or null")
    return value


def _whole_number(value: Any, where: str, name: str) -> int:
    """Return a whole number of the file as an integer.

    The file may hold an integer, a float without a fraction or a string of digits. Any other value
    raises ``ValueError``: a number with a fraction is not cut down to a whole number.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.isascii() and value.isdigit():
        return int(value)
    raise ValueError(f"{where}: {name} {value!r} is not a whole number")


def _frame_rate(info: Any) -> float:
    """Return the frame rate of the clip in frames per second. It must be a finite number above zero."""
    fps = float(_required(info, "frame_rate", "info"))
    if not 0.0 < fps < math.inf:
        raise ValueError(f"info: frame_rate must be a number above 0, found {fps}")
    return fps


def _images_in_frame_order(images: Sequence[Any]) -> list[Mapping[str, Any]]:
    """Return the image rows in the order of the numbers in their file names.

    The name of an image file, without a folder and without its ending, must be a whole number, like
    ``000001.jpg``. A name that is not a number, and two names with the same number, raise
    ``ValueError``: the frame order would not be sure. ``frames.ImageFolderSource`` has the same rule
    for the frame files.
    """
    by_number: dict[int, Mapping[str, Any]] = {}
    for image in images:
        file_name = str(_required(image, "file_name", "image"))
        stem = PurePosixPath(file_name).stem
        if not (stem.isascii() and stem.isdigit()):
            raise ValueError(f"image {file_name!r}: the name of an image file must be a number, like 000001.jpg")
        number = int(stem)
        if number in by_number:
            other = by_number[number]["file_name"]
            raise ValueError(f"the images {str(other)!r} and {file_name!r} have the same number")
        by_number[number] = image
    return [by_number[number] for number in sorted(by_number)]


def _image_id(image: Mapping[str, Any]) -> str:
    """Return the image id of an image row. It must be a string: predictions repeat it as it is."""
    image_id = _required(image, "image_id", "image")
    if not isinstance(image_id, str):
        raise ValueError(f"image {image.get('file_name')!r}: image_id must be a string, found {image_id!r}")
    return image_id


def _is_scored(image: Mapping[str, Any]) -> bool:
    """Return True when the image has person, pitch and camera labels."""
    flags = [_required(image, key, "image") for key in SCORED_FLAGS]
    if not all(isinstance(flag, bool) for flag in flags):
        raise ValueError(f"image {image.get('file_name')!r}: the has_labeled flags must be true or false")
    return all(flags)


def _annotations_by_image(
    annotations: Sequence[Any], image_ids: frozenset[str]
) -> tuple[dict[str, list[GsrAthlete]], dict[str, Lines]]:
    """Group the athletes and the pitch lines by image id."""
    athletes: dict[str, list[GsrAthlete]] = {}
    lines: dict[str, Lines] = {}
    #: (image id, track id) of every athlete that was read.
    seen: set[tuple[str, int]] = set()
    for position, annotation in enumerate(annotations):
        where = f"annotations[{position}]"
        supercategory = _required(annotation, "supercategory", where)
        if supercategory not in ("object", "pitch"):
            continue
        image_id = _required(annotation, "image_id", where)
        if not isinstance(image_id, str) or image_id not in image_ids:
            raise ValueError(f"{where}: image_id {image_id!r} is not in the image list")
        if supercategory == "pitch":
            if image_id in lines:
                raise ValueError(f"{where}: image {image_id!r} has more than one pitch annotation")
            lines[image_id] = _parse_lines(annotation, where)
            continue
        attributes = _required(annotation, "attributes", where)
        role = str(_required(attributes, "role", where))
        if role == BALL_ROLE:
            continue
        athlete = _parse_athlete(annotation, attributes, role, where)
        if (image_id, athlete.track_id) in seen:
            raise ValueError(f"{where}: track_id {athlete.track_id} is used twice in image {image_id!r}")
        seen.add((image_id, athlete.track_id))
        athletes.setdefault(image_id, []).append(athlete)
    return athletes, lines


def _parse_athlete(annotation: Mapping[str, Any], attributes: Mapping[str, Any], role: str, where: str) -> GsrAthlete:
    """Turn one object annotation into an athlete record."""
    team = attributes.get("team")
    pitch_x, pitch_y = _parse_pitch(annotation.get("bbox_pitch"), where)
    return GsrAthlete(
        track_id=_whole_number(_required(annotation, "track_id", where), where, "track_id"),
        box=_parse_box(annotation.get("bbox_image"), where),
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        role=role,
        team=None if team is None else str(team),
        jersey=_parse_jersey(attributes.get("jersey"), where),
    )


def _parse_box(bbox: Any, where: str) -> Box | None:
    """Turn ``bbox_image`` into a box. ``x`` and ``y`` are the top-left corner, ``w`` and ``h`` the size."""
    if bbox is None:
        return None
    x, y, w, h = (float(_required(bbox, key, f"{where} bbox_image")) for key in ("x", "y", "w", "h"))
    if w <= 0.0 or h <= 0.0:
        raise ValueError(f"{where}: bbox_image has no area (w={w}, h={h})")
    return Box(x1=x, y1=y, x2=x + w, y2=y + h)


def _parse_pitch(bbox: Any, where: str) -> tuple[float | None, float | None]:
    """Return the pitch position ``(x, y)`` of ``bbox_pitch`` in metres: its bottom-middle point.

    A position that is not finite raises ``ValueError``: it is not the same as a missing position.
    """
    if bbox is None:
        return None, None
    x, y = (float(_required(bbox, key, f"{where} bbox_pitch")) for key in ("x_bottom_middle", "y_bottom_middle"))
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"{where}: the pitch position of bbox_pitch must be finite, found x={x} and y={y}")
    return x, y


def _parse_jersey(value: Any, where: str) -> int | None:
    """Return a shirt number as an integer. The file may hold a string of digits or a whole number."""
    if value is None:
        return None
    if isinstance(value, str) and value.isascii() and value.isdigit():
        return int(value)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    if isinstance(value, float) and value.is_integer() and value >= 0.0:
        return int(value)
    raise ValueError(f"{where}: jersey {value!r} is neither a string of digits nor a whole number")


def _jersey_is_str(annotations: Sequence[Mapping[str, Any]]) -> bool:
    """Return True when at least one shirt number in the file is stored as a string."""
    for annotation in annotations:
        attributes = annotation.get("attributes")
        if isinstance(attributes, Mapping) and isinstance(attributes.get("jersey"), str):
            return True
    return False


def _parse_lines(annotation: Mapping[str, Any], where: str) -> Lines:
    """Read the lines of one pitch annotation: line class -> points, normalised to 0-1."""
    lines = _required(annotation, "lines", where)
    if not isinstance(lines, Mapping):
        raise ValueError(f"{where}: lines must be a mapping, found {type(lines).__name__}")
    return {
        str(name): tuple(
            (
                float(_required(point, "x", f"{where} line {name!r}")),
                float(_required(point, "y", f"{where} line {name!r}")),
            )
            for point in points
        )
        for name, points in lines.items()
    }
