"""Tests of the SoccerNet-GSR prediction writer and of the format checker."""

from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pytest

from pitchvlm.export_gsr import (
    PITCH_KEYS,
    check_predictions,
    image_ids_without_items,
    prediction_items,
    predictions_path,
    write_predictions,
)
from pitchvlm.labels import gsr_frame_states, load_gsr_labels
from pitchvlm.schemas import Box, FrameState, PlayerState

#: Frame index -> image id of the label file. Frame 2 has no image id, so it is not exported.
IMAGE_IDS = {0: "9001", 1: "9002"}
EXPECTED_IMAGE_IDS = ("9001", "9002")


def frame_states() -> list[FrameState]:
    """Three frames made by hand: every role, with and without a box, with and without a number.

    Exported with ``IMAGE_IDS`` they give seven items. Their positions in the
    item list are used by the tests: 0 player (box, number 10), 1 player (no
    box, no number), 2 goalkeeper, 3 referee, 4 other, 5 and 6 players of the
    second frame.
    """
    box = Box(x1=100.0, y1=200.0, x2=140.0, y2=280.0)
    return [
        FrameState(
            frame=0,
            players=(
                PlayerState(
                    track_id=1, box=box, pitch_x=-10.0, pitch_y=-20.5, role="player", team_side="left", number_export=10
                ),
                PlayerState(track_id=2, pitch_x=12.5, pitch_y=5.0, role="player", team_side="right"),
                PlayerState(
                    track_id=3, pitch_x=-50.0, pitch_y=0.5, role="goalkeeper", team_side="left", number_export=1
                ),
                PlayerState(track_id=4, pitch_x=3.0, pitch_y=10.0, role="referee", team_side="right", number_export=5),
                PlayerState(track_id=5, pitch_x=0.0, pitch_y=33.0, role="other", team_side="left", number_export=8),
            ),
        ),
        FrameState(
            frame=1,
            players=(
                PlayerState(track_id=1, pitch_x=-9.0, pitch_y=-20.0, role="player", team_side="left", number_export=10),
                PlayerState(track_id=2, pitch_x=13.0, pitch_y=5.5, role="player", team_side="right", number_export=7),
            ),
        ),
        FrameState(frame=2, players=(PlayerState(track_id=1, pitch_x=-8.0, pitch_y=-19.5, role="player"),)),
    ]


def valid_payload(*, jersey_as_str: bool = True) -> dict[str, Any]:
    """Return file content that passes the checker. The tests break one rule of it at a time."""
    return {"predictions": prediction_items(frame_states(), IMAGE_IDS, jersey_as_str=jersey_as_str)}


def only_problem(
    payload: Any, *, expected: Collection[str] = EXPECTED_IMAGE_IDS, jersey_as_str: bool = True
) -> str:
    """Run the checker and return its one message. More or fewer messages fail the test."""
    problems = check_predictions(payload, expected_image_ids=expected, jersey_as_str=jersey_as_str)
    assert len(problems) == 1, problems
    return problems[0]


# --- Writer ----------------------------------------------------------------------


def test_items_have_running_ids_and_the_image_ids_of_the_labels() -> None:
    items = prediction_items(frame_states(), IMAGE_IDS, jersey_as_str=True)

    assert [item["id"] for item in items] == ["1", "2", "3", "4", "5", "6", "7"]
    assert [item["image_id"] for item in items] == ["9001"] * 5 + ["9002"] * 2
    assert [item["track_id"] for item in items] == [1, 2, 3, 4, 5, 1, 2]
    assert all(type(item["track_id"]) is int for item in items)
    assert all(item["supercategory"] == "object" for item in items)
    assert all(item["category_id"] == 1 and type(item["category_id"]) is int for item in items)


def test_frames_without_an_image_id_are_skipped() -> None:
    items = prediction_items(frame_states(), {1: "9002"}, jersey_as_str=True)

    assert [item["image_id"] for item in items] == ["9002", "9002"]
    assert [item["id"] for item in items] == ["1", "2"]
    assert prediction_items(frame_states(), {}, jersey_as_str=True) == []


def test_bbox_pitch_has_all_six_keys_and_repeats_the_middle_point() -> None:
    item = prediction_items(frame_states(), IMAGE_IDS, jersey_as_str=True)[0]

    assert item["bbox_pitch"] == {
        "x_bottom_left": -10.0,
        "y_bottom_left": -20.5,
        "x_bottom_middle": -10.0,
        "y_bottom_middle": -20.5,
        "x_bottom_right": -10.0,
        "y_bottom_right": -20.5,
    }
    assert set(item["bbox_pitch"]) == set(PITCH_KEYS)
    # pitch_x goes into the x keys and pitch_y into the y keys: a far-side player keeps the negative y.
    assert item["bbox_pitch"]["y_bottom_middle"] < 0.0


def test_bbox_image_is_written_only_when_a_box_exists() -> None:
    items = prediction_items(frame_states(), IMAGE_IDS, jersey_as_str=True)

    # Box corners (100, 200) and (140, 280): x and y are the top-left corner, then the centre and the size.
    assert items[0]["bbox_image"] == {
        "x": 100.0,
        "y": 200.0,
        "x_center": 120.0,
        "y_center": 240.0,
        "w": 40.0,
        "h": 80.0,
    }
    assert all("bbox_image" not in item for item in items[1:])


def test_jersey_is_a_string_or_an_integer_as_asked() -> None:
    as_str = prediction_items(frame_states(), IMAGE_IDS, jersey_as_str=True)
    as_int = prediction_items(frame_states(), IMAGE_IDS, jersey_as_str=False)

    assert as_str[0]["attributes"]["jersey"] == "10"
    assert as_int[0]["attributes"]["jersey"] == 10
    assert type(as_int[0]["attributes"]["jersey"]) is int
    # A player without a number has null in both forms.
    assert as_str[1]["attributes"]["jersey"] is None
    assert as_int[1]["attributes"]["jersey"] is None


def test_team_and_jersey_are_null_where_the_role_has_none() -> None:
    attributes = [item["attributes"] for item in prediction_items(frame_states(), IMAGE_IDS, jersey_as_str=True)]

    assert attributes[0] == {"role": "player", "team": "left", "jersey": "10"}
    assert attributes[1] == {"role": "player", "team": "right", "jersey": None}
    # A goalkeeper has a team but no jersey. A referee and other people have neither,
    # also when the frame state holds a team side and a number for them.
    assert attributes[2] == {"role": "goalkeeper", "team": "left", "jersey": None}
    assert attributes[3] == {"role": "referee", "team": None, "jersey": None}
    assert attributes[4] == {"role": "other", "team": None, "jersey": None}


def test_player_without_a_team_side_has_a_null_team() -> None:
    states = [FrameState(frame=0, players=(PlayerState(track_id=1, pitch_x=0.0, pitch_y=0.0, role="player"),))]

    items = prediction_items(states, IMAGE_IDS, jersey_as_str=True)

    assert items[0]["attributes"] == {"role": "player", "team": None, "jersey": None}


def test_predictions_path_follows_the_layout_of_the_scorer() -> None:
    path = predictions_path(Path("out"), "valid", "pitchvlm", "SNGS-021")

    assert path == Path("out") / "SoccerNetGS-valid" / "pitchvlm" / "SNGS-021.json"


def test_write_predictions_round_trip(tmp_path: Path) -> None:
    path = write_predictions(
        frame_states(), IMAGE_IDS, tmp_path, split="valid", tracker="pitchvlm", clip_id="SNGS-021", jersey_as_str=True
    )

    assert path == tmp_path / "SoccerNetGS-valid" / "pitchvlm" / "SNGS-021.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert list(payload) == ["predictions"]
    assert payload["predictions"] == prediction_items(frame_states(), IMAGE_IDS, jersey_as_str=True)
    assert check_predictions(payload, expected_image_ids=EXPECTED_IMAGE_IDS, jersey_as_str=True) == []


def test_write_predictions_with_integer_jerseys_round_trip(tmp_path: Path) -> None:
    path = write_predictions(
        frame_states(), IMAGE_IDS, tmp_path, split="test", tracker="other", clip_id="SNGS-150", jersey_as_str=False
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["predictions"][0]["attributes"]["jersey"] == 10
    assert check_predictions(payload, expected_image_ids=EXPECTED_IMAGE_IDS, jersey_as_str=False) == []


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_write_predictions_refuses_a_position_that_is_not_a_number(tmp_path: Path, bad_value: float) -> None:
    # A player state refuses such a position. ``model_construct`` builds the state without that check,
    # so that the refusal of the writer itself is tested.
    broken = PlayerState.model_construct(track_id=1, pitch_x=bad_value, pitch_y=0.0, role="player")
    states = [FrameState(frame=0, players=(broken,))]

    with pytest.raises(ValueError):
        write_predictions(
            states, IMAGE_IDS, tmp_path, split="valid", tracker="pitchvlm", clip_id="SNGS-021", jersey_as_str=True
        )
    assert not predictions_path(tmp_path, "valid", "pitchvlm", "SNGS-021").exists()


def test_labels_exported_as_predictions_pass_the_checker(tmp_path: Path) -> None:
    def athlete(track_id: int, role: str, team: str | None, jersey: str | None, x: float, y: float) -> dict[str, Any]:
        return {
            "id": f"9001-{track_id}",
            "image_id": "9001",
            "track_id": track_id,
            "supercategory": "object",
            "category_id": 1,
            "attributes": {"role": role, "jersey": jersey, "team": team},
            "bbox_image": {"x": 100, "y": 200, "x_center": 120.0, "y_center": 240.0, "w": 40, "h": 80},
            "bbox_pitch": {
                "x_bottom_left": x - 0.3,
                "y_bottom_left": y,
                "x_bottom_middle": x,
                "y_bottom_middle": y,
                "x_bottom_right": x + 0.3,
                "y_bottom_right": y,
            },
        }

    def image(image_id: str, file_name: str, labelled: bool) -> dict[str, Any]:
        return {
            "image_id": image_id,
            "file_name": file_name,
            "width": 1920,
            "height": 1080,
            "is_labeled": labelled,
            "has_labeled_person": labelled,
            "has_labeled_pitch": labelled,
            "has_labeled_camera": labelled,
        }

    label_file = tmp_path / "SNGS-900" / "Labels-GameState.json"
    label_file.parent.mkdir()
    label_file.write_text(
        json.dumps(
            {
                "info": {"name": "SNGS-900", "frame_rate": 25, "seq_length": 2},
                "images": [image("9001", "000001.jpg", True), image("9002", "000002.jpg", False)],
                "annotations": [
                    athlete(1, "player", "left", "10", -10.0, -20.5),
                    athlete(2, "goalkeeper", "right", "1", 50.0, 0.5),
                    athlete(3, "referee", None, None, 3.0, 10.0),
                    athlete(4, "ball", None, None, 1.0, 1.0),
                ],
                "categories": [],
            }
        ),
        encoding="utf-8",
    )
    labels = load_gsr_labels(label_file)
    image_ids = {frame.frame: frame.image_id for frame in labels.frames if frame.scored}

    items = prediction_items(gsr_frame_states(labels), image_ids, jersey_as_str=labels.jersey_is_str)

    assert image_ids == {0: "9001"}
    assert check_predictions({"predictions": items}, expected_image_ids=["9001"], jersey_as_str=True) == []
    assert [item["track_id"] for item in items] == [1, 2, 3]
    # The item repeats the image id, the position and the jersey type of the label file.
    assert items[0]["image_id"] == "9001"
    assert items[0]["attributes"] == {"role": "player", "team": "left", "jersey": "10"}
    assert (items[0]["bbox_pitch"]["x_bottom_middle"], items[0]["bbox_pitch"]["y_bottom_middle"]) == (-10.0, -20.5)
    assert items[1]["attributes"] == {"role": "goalkeeper", "team": "right", "jersey": None}


# --- Checker: valid content ------------------------------------------------------


def test_valid_content_has_no_problem() -> None:
    assert check_predictions(valid_payload(), expected_image_ids=EXPECTED_IMAGE_IDS, jersey_as_str=True) == []


def test_valid_content_with_integer_jerseys_has_no_problem() -> None:
    payload = valid_payload(jersey_as_str=False)

    assert check_predictions(payload, expected_image_ids=EXPECTED_IMAGE_IDS, jersey_as_str=False) == []


def test_null_team_and_null_jersey_of_a_player_are_valid() -> None:
    payload = valid_payload()
    payload["predictions"][0]["attributes"]["team"] = None
    payload["predictions"][0]["attributes"]["jersey"] = None

    assert check_predictions(payload, expected_image_ids=EXPECTED_IMAGE_IDS, jersey_as_str=True) == []


# --- Checker: one test per rule --------------------------------------------------


@pytest.mark.parametrize("payload", [{}, {"predictions": None}, {"predictions": {"0": {}}}, []])
def test_content_must_hold_a_list_of_predictions(payload: Any) -> None:
    assert only_problem(payload).startswith("predictions:")


def test_item_must_be_a_mapping() -> None:
    payload = valid_payload()
    payload["predictions"][3] = "referee"

    problem = only_problem(payload)

    assert problem.startswith("item:")
    assert "item 3" in problem


#: The Arabic-Indic digits one and three. Python reads them as numbers, the scorer does not.
OTHER_SCRIPT_ONE = "\u0661"
OTHER_SCRIPT_THREE = "\u0663"


@pytest.mark.parametrize("bad_id", [2, "two", "", None, "2.0", "-2", OTHER_SCRIPT_ONE])
def test_id_must_be_a_string_of_digits(bad_id: Any) -> None:
    payload = valid_payload()
    payload["predictions"][1]["id"] = bad_id

    problem = only_problem(payload)

    assert problem.startswith("id:")
    assert "item 1" in problem


def test_id_must_be_there() -> None:
    payload = valid_payload()
    del payload["predictions"][1]["id"]

    assert only_problem(payload).startswith("id:")


def test_ids_must_grow_along_the_list() -> None:
    payload = valid_payload()
    payload["predictions"][2]["id"] = "2"  # the same id as the item before

    problem = only_problem(payload)

    assert problem.startswith("id order:")
    assert "item 2" in problem


def test_image_id_must_be_a_string() -> None:
    payload = valid_payload()
    payload["predictions"][0]["image_id"] = 9001

    problem = only_problem(payload)

    assert problem.startswith("image_id:")
    assert "item 0" in problem


def test_image_id_must_be_an_expected_one() -> None:
    payload = valid_payload()
    payload["predictions"][0]["image_id"] = "404"

    problem = only_problem(payload)

    assert problem.startswith("unknown image_id:")
    assert "item 0" in problem
    assert "'404'" in problem


def test_an_expected_image_id_without_an_item_is_no_problem() -> None:
    # The scorer reads a file with an empty frame: it counts the true people of that frame as missed.
    expected = ("9001", "9002", "9004", "9003")

    assert check_predictions(valid_payload(), expected_image_ids=expected, jersey_as_str=True) == []
    # The image ids without an item are listed apart, in sorted order.
    assert image_ids_without_items(valid_payload(), expected_image_ids=expected) == ["9003", "9004"]


def test_a_frame_state_without_people_gives_a_valid_file_with_one_image_without_items() -> None:
    # The pipeline gives such a state when no track has a pitch position in a scored frame.
    states = [frame_states()[0], FrameState(frame=1)]
    payload = {"predictions": prediction_items(states, IMAGE_IDS, jersey_as_str=True)}

    assert check_predictions(payload, expected_image_ids=EXPECTED_IMAGE_IDS, jersey_as_str=True) == []
    assert image_ids_without_items(payload, expected_image_ids=EXPECTED_IMAGE_IDS) == ["9002"]


def test_a_file_without_any_item_is_valid_and_leaves_every_image_without_items() -> None:
    payload: dict[str, Any] = {"predictions": []}

    assert check_predictions(payload, expected_image_ids=EXPECTED_IMAGE_IDS, jersey_as_str=True) == []
    assert image_ids_without_items(payload, expected_image_ids=EXPECTED_IMAGE_IDS) == ["9001", "9002"]


def test_a_file_with_an_item_for_every_image_leaves_no_image_without_items() -> None:
    assert image_ids_without_items(valid_payload(), expected_image_ids=EXPECTED_IMAGE_IDS) == []


def test_an_item_without_a_string_image_id_stands_for_no_image() -> None:
    payload = valid_payload()
    for item in payload["predictions"][5:]:  # the two items of the image 9002
        item["image_id"] = 9002
    payload["predictions"].append("not an item")

    assert image_ids_without_items(payload, expected_image_ids=EXPECTED_IMAGE_IDS) == ["9002"]


@pytest.mark.parametrize("payload", [{}, {"predictions": None}, {"predictions": {"0": {}}}, []])
def test_image_ids_without_items_need_a_list_of_predictions(payload: Any) -> None:
    with pytest.raises(ValueError, match='a list under the key "predictions"'):
        image_ids_without_items(payload, expected_image_ids=EXPECTED_IMAGE_IDS)


@pytest.mark.parametrize("bad_track_id", ["1", 1.0, True, None])
def test_track_id_must_be_an_integer(bad_track_id: Any) -> None:
    payload = valid_payload()
    payload["predictions"][0]["track_id"] = bad_track_id

    problem = only_problem(payload)

    assert problem.startswith("track_id:")
    assert "item 0" in problem


def test_track_id_must_be_used_once_per_image() -> None:
    payload = valid_payload()
    payload["predictions"][1]["track_id"] = 1  # the track id of item 0, in the same image

    problem = only_problem(payload)

    assert problem.startswith("duplicate track_id:")
    assert "item 1" in problem


def test_supercategory_must_be_object() -> None:
    payload = valid_payload()
    payload["predictions"][4]["supercategory"] = "person"

    problem = only_problem(payload)

    assert problem.startswith("supercategory:")
    assert "item 4" in problem


@pytest.mark.parametrize("bad_category", [2, "1", 1.0, True, None])
def test_category_id_must_be_the_integer_one(bad_category: Any) -> None:
    payload = valid_payload()
    payload["predictions"][0]["category_id"] = bad_category

    problem = only_problem(payload)

    assert problem.startswith("category_id:")
    assert "item 0" in problem


@pytest.mark.parametrize("key", ["role", "team", "jersey"])
def test_attributes_must_hold_role_team_and_jersey(key: str) -> None:
    payload = valid_payload()
    del payload["predictions"][0]["attributes"][key]

    problem = only_problem(payload)

    assert problem.startswith("attributes:")
    assert "item 0" in problem


def test_attributes_must_be_a_mapping() -> None:
    payload = valid_payload()
    payload["predictions"][0]["attributes"] = None

    assert only_problem(payload).startswith("attributes:")


@pytest.mark.parametrize("bad_role", ["ball", "Player", None, 1])
def test_role_must_be_one_of_the_four_roles(bad_role: Any) -> None:
    payload = valid_payload()
    payload["predictions"][0]["attributes"]["role"] = bad_role

    problem = only_problem(payload)

    assert problem.startswith("role:")
    assert "item 0" in problem


@pytest.mark.parametrize("position", [3, 4])
def test_team_of_a_referee_and_of_other_people_must_be_null(position: int) -> None:
    payload = valid_payload()
    payload["predictions"][position]["attributes"]["team"] = "left"

    problem = only_problem(payload)

    assert problem.startswith("team:")
    assert f"item {position}" in problem


@pytest.mark.parametrize("bad_team", ["home", "Left", "A", 0])
def test_team_of_a_player_must_be_a_side_or_null(bad_team: Any) -> None:
    payload = valid_payload()
    payload["predictions"][0]["attributes"]["team"] = bad_team

    problem = only_problem(payload)

    assert problem.startswith("team:")
    assert "item 0" in problem


def test_jersey_type_must_be_the_one_of_the_labels() -> None:
    # String jerseys where the labels hold integers: the three players with a number are wrong.
    problem = only_problem(valid_payload(jersey_as_str=True), jersey_as_str=False)

    assert problem.startswith("jersey:")
    assert "item 0" in problem
    assert "3 cases" in problem

    # And the other way round.
    problem = only_problem(valid_payload(jersey_as_str=False), jersey_as_str=True)

    assert problem.startswith("jersey:")
    assert "item 0" in problem
    assert "3 cases" in problem


@pytest.mark.parametrize("jersey", ["0", "7", "07", "99"])
def test_string_jersey_of_digits_is_valid(jersey: str) -> None:
    payload = valid_payload(jersey_as_str=True)
    payload["predictions"][0]["attributes"]["jersey"] = jersey

    assert check_predictions(payload, expected_image_ids=EXPECTED_IMAGE_IDS, jersey_as_str=True) == []


@pytest.mark.parametrize("jersey", [0, 7, 99])
def test_integer_jersey_from_zero_is_valid(jersey: int) -> None:
    payload = valid_payload(jersey_as_str=False)
    payload["predictions"][0]["attributes"]["jersey"] = jersey

    assert check_predictions(payload, expected_image_ids=EXPECTED_IMAGE_IDS, jersey_as_str=False) == []


@pytest.mark.parametrize("bad_jersey", ["ten", "", "1.5", "-1", 10.0, True, OTHER_SCRIPT_THREE])
def test_string_jersey_must_hold_only_digits(bad_jersey: Any) -> None:
    payload = valid_payload(jersey_as_str=True)
    payload["predictions"][0]["attributes"]["jersey"] = bad_jersey

    problem = only_problem(payload, jersey_as_str=True)

    assert problem.startswith("jersey:")
    assert "item 0" in problem


@pytest.mark.parametrize("bad_jersey", [10.0, True, -1, "ten"])
def test_integer_jersey_must_be_a_whole_number(bad_jersey: Any) -> None:
    payload = valid_payload(jersey_as_str=False)
    payload["predictions"][0]["attributes"]["jersey"] = bad_jersey

    problem = only_problem(payload, jersey_as_str=False)

    assert problem.startswith("jersey:")
    assert "item 0" in problem


@pytest.mark.parametrize("position", [2, 3, 4])
def test_jersey_of_other_roles_than_player_must_be_null(position: int) -> None:
    payload = valid_payload()
    payload["predictions"][position]["attributes"]["jersey"] = "1"

    problem = only_problem(payload)

    assert problem.startswith("jersey:")
    assert f"item {position}" in problem


@pytest.mark.parametrize("key", PITCH_KEYS)
def test_bbox_pitch_needs_all_six_keys(key: str) -> None:
    payload = valid_payload()
    del payload["predictions"][0]["bbox_pitch"][key]

    problem = only_problem(payload)

    assert problem.startswith("bbox_pitch:")
    assert "item 0" in problem


@pytest.mark.parametrize("bad_value", [None, "1.0", float("nan"), float("inf"), True])
def test_bbox_pitch_values_must_be_finite_numbers(bad_value: Any) -> None:
    payload = valid_payload()
    payload["predictions"][0]["bbox_pitch"]["y_bottom_middle"] = bad_value

    assert only_problem(payload).startswith("bbox_pitch:")


def test_bbox_pitch_must_be_there() -> None:
    payload = valid_payload()
    del payload["predictions"][6]["bbox_pitch"]

    problem = only_problem(payload)

    assert problem.startswith("bbox_pitch:")
    assert "item 6" in problem


@pytest.mark.parametrize("key", ["x_bottom_left", "y_bottom_left", "x_bottom_right", "y_bottom_right"])
def test_bbox_pitch_left_and_right_must_repeat_the_middle(key: str) -> None:
    payload = valid_payload()
    payload["predictions"][0]["bbox_pitch"][key] += 0.3

    problem = only_problem(payload)

    assert problem.startswith("bbox_pitch corners:")
    assert "item 0" in problem


@pytest.mark.parametrize("bad_box", [None, [100, 200, 40, 80], {"x": 100.0, "y": 200.0, "w": 40.0, "h": 80.0}])
def test_bbox_image_must_hold_six_numbers_when_it_is_there(bad_box: Any) -> None:
    payload = valid_payload()
    payload["predictions"][0]["bbox_image"] = bad_box

    problem = only_problem(payload)

    assert problem.startswith("bbox_image:")
    assert "item 0" in problem


@pytest.mark.parametrize("key", ["w", "h"])
def test_bbox_image_must_have_a_positive_size(key: str) -> None:
    payload = valid_payload()
    payload["predictions"][0]["bbox_image"][key] = 0.0

    problem = only_problem(payload)

    assert problem.startswith("bbox_image size:")
    assert "item 0" in problem


# --- Checker: messages -------------------------------------------------------------


def test_message_names_the_first_offending_item_and_counts_the_cases() -> None:
    payload = valid_payload()
    for position in (4, 2, 6):
        payload["predictions"][position]["supercategory"] = "person"

    problem = only_problem(payload)

    # Item 2 is the first of the three wrong items in the list.
    expected = "supercategory: item 2 (id '3'): supercategory must be \"object\", found 'person' (3 cases in total)"
    assert problem == expected


def test_every_broken_rule_gets_its_own_message() -> None:
    payload = valid_payload()
    payload["predictions"][0]["category_id"] = 2
    payload["predictions"][5]["track_id"] = "1"
    payload["predictions"][6]["bbox_pitch"]["x_bottom_right"] = 99.0

    problems = check_predictions(payload, expected_image_ids=EXPECTED_IMAGE_IDS, jersey_as_str=True)

    assert [problem.split(":")[0] for problem in problems] == ["category_id", "track_id", "bbox_pitch corners"]
    assert ["item 0" in problems[0], "item 5" in problems[1], "item 6" in problems[2]] == [True, True, True]
