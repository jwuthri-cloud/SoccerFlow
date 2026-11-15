"""Team rosters: the player name for a shirt number, per team side.

Players are identified by shirt number plus roster, never by face.
"""

from __future__ import annotations

import json
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import yaml

from pitchvlm.schemas import Record, TeamSide

_YAML_SUFFIXES = (".yaml", ".yml")
_JSON_SUFFIX = ".json"
_SIDES: tuple[TeamSide, ...] = ("left", "right")
_TEAM_FIELDS = ("name", "players")
#: Range of a shirt number. It is the range of ``PersonDet.number``.
_MIN_NUMBER = 0
_MAX_NUMBER = 99


class Roster(Record):
    """Player names of the two teams, by shirt number.

    ``left`` is the team that defends the left goal as seen by the main camera,
    ``right`` is the other team.
    """

    left: dict[int, str]
    right: dict[int, str]
    left_name: str = ""
    right_name: str = ""

    def name_for(self, side: TeamSide, number: int) -> str | None:
        """Return the name of the player with this shirt number, or None when the roster has no such player."""
        if side == "left":
            return self.left.get(number)
        if side == "right":
            return self.right.get(number)
        raise ValueError(f"unknown team side: {side!r}")


def load_roster(path: Path) -> Roster:
    """Read a roster file.

    The file is YAML (``.yaml``, ``.yml``) or JSON (``.json``) with this layout::

        left:
          name: Home FC
          players:
            1: First Keeper
            10: First Playmaker
        right:
          name: Away United
          players:
            9: Second Striker

    Both sides are required. ``name`` is optional. A shirt number may be written
    as text (a JSON key is always text); it becomes an integer. Wrong content
    raises ``ValueError`` with the name of the file. This holds also for a key
    that is written twice in one mapping (the readers of YAML and JSON would
    keep the last value without a word), for a file that is not UTF-8 text and
    for a file that is not valid YAML or JSON.
    """
    path = Path(path)
    data = _mapping(_parse(path), f"{path}: the top level")
    _check_keys(data, _SIDES, str(path))
    left_name, left = _team(data, "left", path)
    right_name, right = _team(data, "right", path)
    return Roster(left=left, right=right, left_name=left_name, right_name=right_name)


class _RepeatedKey(ValueError):
    """A mapping of the file has the same key twice."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """A safe YAML loader that refuses a mapping with the same key written twice."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Hashable] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            if not isinstance(key, Hashable):
                continue  # The parent class reports a key that cannot be a key.
            if key in seen:
                raise _RepeatedKey(f"the key {key!r} is written twice (line {key_node.start_mark.line + 1})")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Turn the key-value pairs of a JSON object into a mapping. The same key twice raises."""
    mapping: dict[str, Any] = {}
    for key, value in pairs:
        if key in mapping:
            raise _RepeatedKey(f"the key {key!r} is written twice")
        mapping[key] = value
    return mapping


def _parse(path: Path) -> Any:
    """Parse the file as YAML or JSON. The file suffix decides. A file that cannot be parsed raises ``ValueError``."""
    suffix = path.suffix.lower()
    if suffix != _JSON_SUFFIX and suffix not in _YAML_SUFFIXES:
        raise ValueError(f"{path}: a roster file must end with .yaml, .yml or .json")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: the file is not UTF-8 text ({error})") from error
    kind = "JSON" if suffix == _JSON_SUFFIX else "YAML"
    try:
        if suffix == _JSON_SUFFIX:
            return json.loads(text, object_pairs_hook=_unique_pairs)
        return yaml.load(text, Loader=_UniqueKeyLoader)
    except _RepeatedKey as error:
        raise ValueError(f"{path}: {error}") from error
    except (yaml.YAMLError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: the file is not valid {kind}: {error}") from error


def _mapping(value: Any, where: str) -> dict[Any, Any]:
    """Return ``value`` when it is a mapping, else raise."""
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    return value


def _check_keys(entry: dict[Any, Any], allowed: tuple[str, ...], where: str) -> None:
    """Raise when the mapping has a key that is not allowed."""
    unknown = [key for key in entry if key not in allowed]
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown}; allowed keys are {list(allowed)}")


def _team(data: dict[Any, Any], side: TeamSide, path: Path) -> tuple[str, dict[int, str]]:
    """Return the team name and the players of one side."""
    if side not in data:
        raise ValueError(f"{path}: the key '{side}' is missing")
    where = f"{path}: '{side}'"
    entry = _mapping(data[side], where)
    _check_keys(entry, _TEAM_FIELDS, where)
    if "players" not in entry:
        raise ValueError(f"{where}: the key 'players' is missing")
    name = entry.get("name", "")
    if not isinstance(name, str):
        raise ValueError(f"{where}: the team name must be text (put it in quotes)")
    return name.strip(), _players(entry["players"], where)


def _players(raw: Any, where: str) -> dict[int, str]:
    """Turn the ``players`` mapping of the file into ``{shirt number: name}``."""
    players: dict[int, str] = {}
    for key, name in _mapping(raw, f"{where}: 'players'").items():
        number = _shirt_number(key, where)
        if number in players:
            raise ValueError(f"{where}: shirt number {number} is listed twice")
        if not isinstance(name, str):
            raise ValueError(f"{where}: the name of shirt number {number} must be text (put it in quotes)")
        if not name.strip():
            raise ValueError(f"{where}: the name of shirt number {number} is empty")
        players[number] = name.strip()
    return players


def _shirt_number(key: Any, where: str) -> int:
    """Turn a key of the ``players`` mapping into a shirt number.

    The key is an integer, or text made of digits only. ``bool`` is refused
    although Python counts it as an integer.
    """
    if isinstance(key, int) and not isinstance(key, bool):
        number = key
    elif isinstance(key, str) and key.strip().isascii() and key.strip().isdigit():
        number = int(key.strip())
    else:
        raise ValueError(f"{where}: shirt number {key!r} is not a whole number")
    if not _MIN_NUMBER <= number <= _MAX_NUMBER:
        raise ValueError(f"{where}: shirt number {number} is outside {_MIN_NUMBER}-{_MAX_NUMBER}")
    return number
