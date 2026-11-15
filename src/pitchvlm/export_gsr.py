"""Writer and checker for SoccerNet-GSR prediction files.

The scorer reads one JSON file per clip: ``{"predictions": [item, ...]}``. A
file with a wrong format gives a score of 0 without any error message. So
``check_predictions`` tests every rule of the format before a file is scored.

A scored frame without any item is not a format problem. The scorer reads
such a file and counts every true person of that frame as missed. The
pipeline leaves a frame without items when no track has a pitch position
there. ``image_ids_without_items`` lists these frames, so that a caller can
report their number.
"""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from pitchvlm.schemas import ROLES, TEAM_ROLES, Box, FrameState, PlayerState

#: Keys of ``bbox_pitch``, metres. The left and the right point repeat the middle point.
PITCH_KEYS = (
    "x_bottom_left",
    "y_bottom_left",
    "x_bottom_middle",
    "y_bottom_middle",
    "x_bottom_right",
    "y_bottom_right",
)
#: Keys of ``bbox_image``, pixels: top-left corner, centre and size.
IMAGE_KEYS = ("x", "y", "x_center", "y_center", "w", "h")
ATTRIBUTE_KEYS = ("role", "team", "jersey")
TEAM_SIDES = ("left", "right")
#: What is wrong with file content that holds no list of items.
NO_ITEM_LIST = 'the content must be a mapping with a list under the key "predictions"'


def prediction_items(
    frames: Sequence[FrameState], image_ids: Mapping[int, str], *, jersey_as_str: bool
) -> list[dict[str, Any]]:
    """Turn frame states into the prediction items of the scorer.

    ``image_ids`` maps a frame index to the image id of the label file. Frames
    that are not in ``image_ids`` are skipped. The item ids run from "1".
    ``jersey_as_str`` must be the type of the shirt numbers in the label file
    (``GsrLabels.jersey_is_str``): True writes ``"10"``, False writes ``10``.
    """
    items: list[dict[str, Any]] = []
    for state in frames:
        if state.frame not in image_ids:
            continue
        for player in state.players:
            items.append(_item(len(items) + 1, image_ids[state.frame], player, jersey_as_str))
    return items


def predictions_path(root: Path, split: str, tracker: str, clip_id: str) -> Path:
    """Return the place of a prediction file: ``root/SoccerNetGS-<split>/<tracker>/<clip_id>.json``."""
    return Path(root) / f"SoccerNetGS-{split}" / tracker / f"{clip_id}.json"


def write_predictions(
    frames: Sequence[FrameState],
    image_ids: Mapping[int, str],
    root: Path,
    *,
    split: str,
    tracker: str,
    clip_id: str,
    jersey_as_str: bool,
) -> Path:
    """Write the prediction file of one clip and return its path.

    A position or a box value that is not a finite number raises ``ValueError``;
    then no file is written.
    """
    payload = {"predictions": prediction_items(frames, image_ids, jersey_as_str=jersey_as_str)}
    text = json.dumps(payload, allow_nan=False)
    path = predictions_path(root, split, tracker, clip_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def check_predictions(
    payload: dict[str, Any], *, expected_image_ids: Collection[str], jersey_as_str: bool
) -> list[str]:
    """Test the content of a prediction file against every rule of the format.

    The result is a list of problems; an empty list means that the content is
    valid. There is one message per broken rule. It starts with the name of
    the rule and names the first offending item by its position in the list
    (counted from 0) and its id. When more items break the same rule, the
    message ends with their number.

    Rules: ``id`` is a string of digits and grows along the list; ``image_id``
    is a string and one of ``expected_image_ids``; ``track_id`` is an integer
    and is used once per image; ``supercategory`` is "object";
    ``category_id`` is 1; ``attributes`` holds ``role``, ``team`` and
    ``jersey``; the role is one of the four roles; the team is ``left``,
    ``right`` or null for players and goalkeepers and null for the other
    roles; the jersey is a string of digits (``jersey_as_str``) or an integer,
    or null, for players and null for the other roles; ``bbox_pitch`` holds
    six finite numbers and its left and right points repeat the middle point;
    ``bbox_image``, when present, holds six finite numbers and has a positive
    size.

    An expected image id without any item breaks no rule (see the module
    text): ``image_ids_without_items`` returns these image ids.
    """
    items = _item_list(payload)
    if items is None:
        return [f"predictions: {NO_ITEM_LIST}"]
    checker = _Checker(expected_image_ids, jersey_as_str)
    for position, item in enumerate(items):
        checker.check_item(position, item)
    return checker.finish()


def image_ids_without_items(payload: dict[str, Any], *, expected_image_ids: Collection[str]) -> list[str]:
    """Return the expected image ids for which the content of a prediction file holds no item, sorted.

    These are the scored frames without a prediction. An item stands for the
    image that its ``image_id`` names; an item whose ``image_id`` is not a
    string stands for no image. ``ValueError`` is raised when the content
    holds no list of items.
    """
    items = _item_list(payload)
    if items is None:
        raise ValueError(NO_ITEM_LIST)
    named = {item["image_id"] for item in items if isinstance(item, dict) and isinstance(item.get("image_id"), str)}
    return sorted(frozenset(expected_image_ids) - named)


# --- Writer ----------------------------------------------------------------------


def _item(number: int, image_id: str, player: PlayerState, jersey_as_str: bool) -> dict[str, Any]:
    """Build the prediction item of one person in one frame."""
    item: dict[str, Any] = {
        "id": str(number),
        "image_id": image_id,
        "track_id": player.track_id,
        "supercategory": "object",
        "category_id": 1,
        "attributes": _attributes(player, jersey_as_str),
        "bbox_pitch": _bbox_pitch(player.pitch_x, player.pitch_y),
    }
    if player.box is not None:
        item["bbox_image"] = _bbox_image(player.box)
    return item


def _attributes(player: PlayerState, jersey_as_str: bool) -> dict[str, str | int | None]:
    """Return role, team and jersey. Only players and goalkeepers have a team; only players have a jersey."""
    team = player.team_side if player.role in TEAM_ROLES else None
    number = player.number_export if player.role == "player" else None
    jersey: str | int | None = str(number) if number is not None and jersey_as_str else number
    return {"role": player.role, "team": team, "jersey": jersey}


def _bbox_pitch(x: float, y: float) -> dict[str, float]:
    """Return the pitch position with all six keys. The left and the right point repeat the middle point."""
    return {
        "x_bottom_left": x,
        "y_bottom_left": y,
        "x_bottom_middle": x,
        "y_bottom_middle": y,
        "x_bottom_right": x,
        "y_bottom_right": y,
    }


def _bbox_image(box: Box) -> dict[str, float]:
    """Return an image box in the label format: ``x`` and ``y`` are the top-left corner."""
    center_x, center_y = box.center
    return {"x": box.x1, "y": box.y1, "x_center": center_x, "y_center": center_y, "w": box.width, "h": box.height}


# --- Checker ---------------------------------------------------------------------


def _item_list(payload: object) -> list[Any] | None:
    """Return the list under the key "predictions", or None when the content holds no such list."""
    items = payload.get("predictions") if isinstance(payload, dict) else None
    return items if isinstance(items, list) else None


def _is_integer(value: object) -> bool:
    """Return True for an integer. Truth values do not count as integers."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    """Return True for an integer or a finite float. Truth values do not count as numbers."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _is_digit_string(value: object) -> bool:
    """Return True for a string that holds only the digits 0 to 9."""
    return isinstance(value, str) and value.isascii() and value.isdigit()


class _Checker:
    """Tests prediction items one by one. It keeps the first offending item of every broken rule."""

    def __init__(self, expected_image_ids: Collection[str], jersey_as_str: bool) -> None:
        self._expected = frozenset(expected_image_ids)
        self._jersey_as_str = jersey_as_str
        #: Rule name -> message about the first item that breaks the rule.
        self._first: dict[str, str] = {}
        #: Rule name -> number of items that break the rule.
        self._cases: dict[str, int] = {}
        self._last_id: int | None = None
        self._tracks: set[tuple[str, int]] = set()

    def check_item(self, position: int, item: object) -> None:
        """Test one item against every rule."""
        if not isinstance(item, dict):
            self._report("item", f"item {position} is not a mapping")
            return
        where = f"item {position} (id {item.get('id')!r})"
        self._check_id(item, where)
        self._check_image_and_track(item, where)
        self._check_constants(item, where)
        self._check_attributes(item, where)
        self._check_bbox_pitch(item, where)
        self._check_bbox_image(item, where)

    def finish(self) -> list[str]:
        """Return one message per broken rule."""
        return [
            f"{rule}: {first}" if self._cases[rule] == 1 else f"{rule}: {first} ({self._cases[rule]} cases in total)"
            for rule, first in self._first.items()
        ]

    def _report(self, rule: str, message: str) -> None:
        """Count a broken rule and keep the message of its first case."""
        self._first.setdefault(rule, message)
        self._cases[rule] = self._cases.get(rule, 0) + 1

    def _check_id(self, item: dict[str, Any], where: str) -> None:
        """Rule: ``id`` is a running integer, written as a string."""
        value = item.get("id")
        if not _is_digit_string(value):
            self._report("id", f"{where}: id must be a string of digits")
            return
        number = int(value)
        if self._last_id is not None and number <= self._last_id:
            self._report("id order", f"{where}: id {number} does not come after id {self._last_id}")
        self._last_id = number

    def _check_image_and_track(self, item: dict[str, Any], where: str) -> None:
        """Rules: ``image_id`` is an expected string; ``track_id`` is an integer, used once per image."""
        image_id = item.get("image_id")
        track_id = item.get("track_id")
        if not isinstance(image_id, str):
            self._report("image_id", f"{where}: image_id must be a string, found {image_id!r}")
        elif image_id not in self._expected:
            self._report("unknown image_id", f"{where}: image_id {image_id!r} is not an expected image id")
        if not _is_integer(track_id):
            self._report("track_id", f"{where}: track_id must be an integer, found {track_id!r}")
        elif isinstance(image_id, str):
            if (image_id, track_id) in self._tracks:
                self._report("duplicate track_id", f"{where}: track_id {track_id} is used twice in image {image_id!r}")
            self._tracks.add((image_id, track_id))

    def _check_constants(self, item: dict[str, Any], where: str) -> None:
        """Rules: ``supercategory`` is "object" and ``category_id`` is 1."""
        supercategory = item.get("supercategory")
        category_id = item.get("category_id")
        if supercategory != "object":
            self._report("supercategory", f'{where}: supercategory must be "object", found {supercategory!r}')
        if not _is_integer(category_id) or category_id != 1:
            self._report("category_id", f"{where}: category_id must be the integer 1, found {category_id!r}")

    def _check_attributes(self, item: dict[str, Any], where: str) -> None:
        """Rules: ``attributes`` holds role, team and jersey, and their values fit the role."""
        attributes = item.get("attributes")
        if not isinstance(attributes, dict) or any(key not in attributes for key in ATTRIBUTE_KEYS):
            self._report("attributes", f"{where}: attributes must be a mapping with the keys role, team and jersey")
            return
        role = attributes["role"]
        if role not in ROLES:
            self._report("role", f"{where}: role must be one of {', '.join(ROLES)}, found {role!r}")
            return
        self._check_team(attributes["team"], role, where)
        self._check_jersey(attributes["jersey"], role, where)

    def _check_team(self, team: Any, role: str, where: str) -> None:
        """Rule: ``team`` is a team side or null for players and goalkeepers, and null for the other roles."""
        if team is None:
            return
        if role not in TEAM_ROLES:
            self._report("team", f"{where}: team of a {role} must be null, found {team!r}")
        elif team not in TEAM_SIDES:
            self._report("team", f"{where}: team must be left, right or null, found {team!r}")

    def _check_jersey(self, jersey: Any, role: str, where: str) -> None:
        """Rule: ``jersey`` is a number of the agreed type or null for players, and null for the other roles."""
        if jersey is None:
            return
        if role != "player":
            self._report("jersey", f"{where}: jersey of a {role} must be null, found {jersey!r}")
        elif self._jersey_as_str and not _is_digit_string(jersey):
            self._report("jersey", f"{where}: jersey must be a string of digits or null, found {jersey!r}")
        elif not self._jersey_as_str and not (_is_integer(jersey) and jersey >= 0):
            self._report("jersey", f"{where}: jersey must be an integer or null, found {jersey!r}")

    def _check_bbox_pitch(self, item: dict[str, Any], where: str) -> None:
        """Rules: ``bbox_pitch`` holds six finite numbers; its left and right points repeat the middle point."""
        bbox = item.get("bbox_pitch")
        if not isinstance(bbox, dict) or not all(_is_number(bbox.get(key)) for key in PITCH_KEYS):
            keys = ", ".join(PITCH_KEYS)
            self._report("bbox_pitch", f"{where}: bbox_pitch must hold a finite number for each of {keys}")
            return
        same_x = bbox["x_bottom_left"] == bbox["x_bottom_middle"] == bbox["x_bottom_right"]
        same_y = bbox["y_bottom_left"] == bbox["y_bottom_middle"] == bbox["y_bottom_right"]
        if not (same_x and same_y):
            self._report(
                "bbox_pitch corners", f"{where}: the left and right points of bbox_pitch must repeat the middle point"
            )

    def _check_bbox_image(self, item: dict[str, Any], where: str) -> None:
        """Rules: ``bbox_image``, when present, holds six finite numbers and has a positive size."""
        if "bbox_image" not in item:
            return
        bbox = item["bbox_image"]
        if not isinstance(bbox, dict) or not all(_is_number(bbox.get(key)) for key in IMAGE_KEYS):
            keys = ", ".join(IMAGE_KEYS)
            self._report("bbox_image", f"{where}: bbox_image must hold a finite number for each of {keys}")
        elif bbox["w"] <= 0 or bbox["h"] <= 0:
            self._report("bbox_image size", f"{where}: bbox_image must have a positive width and height")
