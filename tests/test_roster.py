"""Tests of the roster record and of the roster file reader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pitchvlm.roster import Roster, load_roster

ROSTER_YAML = """\
left:
  name: Home FC
  players:
    1: First Keeper
    "10": First Playmaker
right:
  name: Away United
  players:
    9: Second Striker
    "07": "  Second Winger  "
"""

#: Roster files with wrong content. Every one must raise ``ValueError``.
BAD_YAML = {
    "number is a word": "left: {players: {ten: A}}\nright: {players: {}}\n",
    "number is text with a sign": 'left: {players: {"-3": A}}\nright: {players: {}}\n',
    "number is negative": "left: {players: {-3: A}}\nright: {players: {}}\n",
    "number is too large": "left: {players: {100: A}}\nright: {players: {}}\n",
    "number is a truth value": "left: {players: {true: A}}\nright: {players: {}}\n",
    "number is listed twice": 'left: {players: {10: A, "10": B}}\nright: {players: {}}\n',
    "name is not text": "left: {players: {10: 123}}\nright: {players: {}}\n",
    "name is empty": 'left: {players: {10: ""}}\nright: {players: {}}\n',
    "side is missing": "left: {players: {}}\n",
    "players are missing": "left: {name: Home FC}\nright: {players: {}}\n",
    "players are a list": "left: {players: [A, B]}\nright: {players: {}}\n",
    "third side": "left: {players: {}}\nright: {players: {}}\ncentre: {players: {}}\n",
    "unknown team field": "left: {players: {}, coach: X}\nright: {players: {}}\n",
    "team name is not text": "left: {name: 12, players: {}}\nright: {players: {}}\n",
    "top level is a list": "- left\n- right\n",
}


def write(tmp_path: Path, file_name: str, text: str) -> Path:
    """Write a text file into the test folder and return its path."""
    path = tmp_path / file_name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("file_name", ["roster.yaml", "roster.yml", "ROSTER.YAML"])
def test_load_yaml_turns_text_numbers_into_integers(tmp_path: Path, file_name: str) -> None:
    roster = load_roster(write(tmp_path, file_name, ROSTER_YAML))

    assert roster == Roster(
        left={1: "First Keeper", 10: "First Playmaker"},
        right={9: "Second Striker", 7: "Second Winger"},
        left_name="Home FC",
        right_name="Away United",
    )
    assert all(type(number) is int for number in [*roster.left, *roster.right])


def test_load_json_turns_text_numbers_into_integers(tmp_path: Path) -> None:
    data = {
        "left": {"name": "Home FC", "players": {"1": "First Keeper", "10": "José Example"}},
        "right": {"players": {"9": "Second Striker"}},
    }
    roster = load_roster(write(tmp_path, "roster.json", json.dumps(data, ensure_ascii=False)))

    assert roster.left == {1: "First Keeper", 10: "José Example"}
    assert roster.right == {9: "Second Striker"}
    assert roster.left_name == "Home FC"
    # The team name is optional.
    assert roster.right_name == ""


def test_loaded_roster_gives_names_by_side_and_number(tmp_path: Path) -> None:
    roster = load_roster(write(tmp_path, "roster.yaml", ROSTER_YAML))

    assert roster.name_for("left", 10) == "First Playmaker"
    assert roster.name_for("right", 7) == "Second Winger"
    # Number 9 plays for the right team only.
    assert roster.name_for("right", 9) == "Second Striker"
    assert roster.name_for("left", 9) is None


def test_name_for_keeps_the_two_sides_apart() -> None:
    roster = Roster(left={10: "Left Ten"}, right={10: "Right Ten", 4: "Right Four"})

    assert roster.name_for("left", 10) == "Left Ten"
    assert roster.name_for("right", 10) == "Right Ten"
    assert roster.name_for("right", 4) == "Right Four"
    assert roster.name_for("left", 4) is None


def test_name_for_refuses_an_unknown_side() -> None:
    roster = Roster(left={10: "Left Ten"}, right={10: "Right Ten"})

    with pytest.raises(ValueError, match="unknown team side"):
        roster.name_for("centre", 10)  # type: ignore[arg-type]


def test_roster_is_frozen() -> None:
    roster = Roster(left={}, right={})

    with pytest.raises(ValidationError):
        roster.left_name = "Home FC"  # type: ignore[misc]


@pytest.mark.parametrize("text", list(BAD_YAML.values()), ids=list(BAD_YAML))
def test_load_refuses_wrong_yaml_content(tmp_path: Path, text: str) -> None:
    with pytest.raises(ValueError):
        load_roster(write(tmp_path, "roster.yaml", text))


def test_load_refuses_wrong_json_content(tmp_path: Path) -> None:
    data = {"left": {"players": {"x1": "A"}}, "right": {"players": {}}}

    with pytest.raises(ValueError, match="not a whole number"):
        load_roster(write(tmp_path, "roster.json", json.dumps(data)))


#: Roster files in which a key is written twice. The readers of YAML and JSON keep the last value of such a key.
REPEATED_KEYS = {
    "yaml number twice": (
        "roster.yaml",
        "left:\n  players:\n    10: First Playmaker\n    10: Second Playmaker\nright:\n  players: {}\n",
        "10",
    ),
    "yaml number twice in braces": (
        "roster.yaml",
        "left: {players: {10: First Playmaker, 10: Second Playmaker}}\nright: {players: {}}\n",
        "10",
    ),
    "yaml side twice": (
        "roster.yml",
        "left:\n  players: {10: Playmaker}\nleft:\n  players: {7: Winger}\nright:\n  players: {}\n",
        "'left'",
    ),
    "yaml team name twice": (
        "roster.yaml",
        "left: {name: Home FC, name: Home United, players: {}}\nright: {players: {}}\n",
        "'name'",
    ),
    "json number twice": (
        "roster.json",
        '{"left": {"players": {"10": "First Playmaker", "10": "Second Playmaker"}}, "right": {"players": {}}}',
        "'10'",
    ),
    "json side twice": (
        "roster.json",
        '{"left": {"players": {"10": "Playmaker"}}, "left": {"players": {"7": "Winger"}}, "right": {"players": {}}}',
        "'left'",
    ),
}


@pytest.mark.parametrize(("file_name", "text", "key"), list(REPEATED_KEYS.values()), ids=list(REPEATED_KEYS))
def test_load_refuses_a_key_that_is_written_twice(tmp_path: Path, file_name: str, text: str, key: str) -> None:
    path = write(tmp_path, file_name, text)

    with pytest.raises(ValueError, match=f"the key {key} is written twice") as caught:
        load_roster(path)

    assert str(path) in str(caught.value)


def test_load_names_the_line_of_a_repeated_yaml_key(tmp_path: Path) -> None:
    players = "    10: First Playmaker\n    7: Winger\n    10: Second Playmaker\n"
    text = f"left:\n  players:\n{players}right:\n  players: {{}}\n"

    with pytest.raises(ValueError, match=r"line 5"):
        load_roster(write(tmp_path, "roster.yaml", text))


def test_same_number_on_both_sides_is_no_repeated_key(tmp_path: Path) -> None:
    text = "left: {players: {10: Left Ten}}\nright: {players: {10: Right Ten}}\n"

    roster = load_roster(write(tmp_path, "roster.yaml", text))

    assert (roster.name_for("left", 10), roster.name_for("right", 10)) == ("Left Ten", "Right Ten")


@pytest.mark.parametrize("file_name", ["roster.yaml", "roster.json"])
def test_load_names_a_file_that_is_not_utf8(tmp_path: Path, file_name: str) -> None:
    path = tmp_path / file_name
    path.write_bytes('{"left": {"players": {"10": "José"}}, "right": {"players": {}}}'.encode("latin-1"))

    with pytest.raises(ValueError, match="not UTF-8") as caught:
        load_roster(path)

    assert str(path) in str(caught.value)


@pytest.mark.parametrize(
    ("file_name", "text", "kind"),
    [("roster.yaml", "left: {players: {10: A}\n", "YAML"), ("roster.json", '{"left": {"players": ', "JSON")],
)
def test_load_names_a_file_that_cannot_be_parsed(tmp_path: Path, file_name: str, text: str, kind: str) -> None:
    path = write(tmp_path, file_name, text)

    with pytest.raises(ValueError, match=f"not valid {kind}") as caught:
        load_roster(path)

    assert str(path) in str(caught.value)


def test_load_refuses_an_unknown_file_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must end with"):
        load_roster(write(tmp_path, "roster.txt", ROSTER_YAML))


def test_load_of_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_roster(tmp_path / "missing.yaml")
