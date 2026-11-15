"""Tests of the SoccerNet-GSR label reader. The label file is made by hand in the test folder."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from pitchvlm.labels import GsrLabels, gsr_frame_people, gsr_frame_states, load_gsr_labels
from pitchvlm.schemas import Box, FramePeople, PlayerState


def image_row(image_id: str, file_name: str, *, labelled: bool = True) -> dict[str, Any]:
    """Build one row of the ``images`` list."""
    return {
        "is_labeled": labelled,
        "image_id": image_id,
        "file_name": file_name,
        "height": 1080,
        "width": 1920,
        "has_labeled_person": labelled,
        "has_labeled_pitch": labelled,
        "has_labeled_camera": labelled,
    }


def athlete_row(
    image_id: str,
    track_id: int,
    role: str,
    *,
    team: str | None = None,
    jersey: str | int | None = None,
    box: tuple[float, float, float, float] | None = (100, 200, 40, 80),
    pitch: tuple[float, float] | None = (0.0, 0.0),
) -> dict[str, Any]:
    """Build one object annotation.

    ``box`` is (x, y, w, h) with the top-left corner. ``pitch`` is (x, y) in
    metres. With None the key is left out of the annotation.
    """
    row: dict[str, Any] = {
        "id": f"{image_id}-{track_id}",
        "image_id": image_id,
        "track_id": track_id,
        "supercategory": "object",
        "category_id": 1,
        "attributes": {"role": role, "jersey": jersey, "team": team},
    }
    if box is not None:
        x, y, w, h = box
        row["bbox_image"] = {"x": x, "y": y, "x_center": x + w / 2, "y_center": y + h / 2, "w": w, "h": h}
    if pitch is not None:
        pitch_x, pitch_y = pitch
        row["bbox_pitch"] = {
            "x_bottom_left": pitch_x - 0.3,
            "y_bottom_left": pitch_y,
            "x_bottom_middle": pitch_x,
            "y_bottom_middle": pitch_y,
            "x_bottom_right": pitch_x + 0.3,
            "y_bottom_right": pitch_y,
        }
    return row


def label_dict() -> dict[str, Any]:
    """Labels of a clip with three frames: two labelled frames and one frame without labels.

    The images and the annotations are not in frame order, to test the sorting.
    The positions in the annotation list are used by the tests:
    0-3 belong to frame 1 (file 000002.jpg), 4-10 to frame 0 (file 000001.jpg).
    """
    return {
        "info": {"version": "1.3", "name": "SNGS-900", "frame_rate": 25, "seq_length": 3, "im_ext": ".jpg"},
        "images": [
            image_row("9002", "000002.jpg"),
            image_row("9003", "000003.jpg", labelled=False),
            image_row("9001", "000001.jpg"),
        ],
        "annotations": [
            athlete_row("9002", 1, "player", team="left", jersey="10", box=(110, 205, 40, 80), pitch=(-9.0, -20.0)),
            athlete_row("9002", 2, "player", team="right", jersey="7", box=(510, 300, 50, 100), pitch=None),
            athlete_row("9002", 6, "other", box=None, pitch=(0.0, 33.0)),
            athlete_row("9002", 7, "player", team="right", box=(700, 400, 30, 90), pitch=(20.0, 3.0)),
            athlete_row("9001", 1, "player", team="left", jersey="10", box=(100, 200, 40, 80), pitch=(-10.0, -20.5)),
            athlete_row("9001", 2, "player", team="right", jersey="7", box=(500, 300, 50, 100), pitch=(12.5, 5.0)),
            athlete_row("9001", 3, "goalkeeper", team="left", jersey="1", box=(50, 400, 30, 60), pitch=(-50.0, 0.5)),
            athlete_row("9001", 4, "referee", box=(800, 350, 30, 70), pitch=(3.0, 10.0)),
            athlete_row("9001", 5, "ball", box=(600, 500, 10, 10), pitch=(1.0, 1.0)),
            {
                "id": "9001-pitch",
                "image_id": "9001",
                "supercategory": "pitch",
                "category_id": 5,
                "lines": {"Middle line": [{"x": 0.5, "y": 0.1}, {"x": 0.52, "y": 0.9}], "Circle central": []},
            },
            {
                "id": "9001-camera",
                "image_id": "9001",
                "supercategory": "camera",
                "category_id": 6,
                "parameters": {"pan_degrees": 1.5},
            },
        ],
        "categories": [{"supercategory": "object", "id": 1, "name": "player"}],
    }


def write_labels(tmp_path: Path, data: Any, folder: str = "clip") -> Path:
    """Write a label file into ``tmp_path/<folder>`` and return its path."""
    path = tmp_path / folder / "Labels-GameState.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def labels(tmp_path: Path) -> GsrLabels:
    """The hand-made label file, written to the test folder and read back."""
    return load_gsr_labels(write_labels(tmp_path, label_dict()))


# --- Reading ---------------------------------------------------------------------


def test_clip_facts_come_from_info_and_the_first_image(labels: GsrLabels) -> None:
    assert labels.clip_id == "SNGS-900"
    assert labels.fps == 25.0
    assert labels.n_frames == 3
    assert (labels.width, labels.height) == (1920, 1080)
    assert labels.version == "1.3"
    assert labels.jersey_is_str is True


def test_frame_index_is_the_position_in_file_name_order(labels: GsrLabels) -> None:
    # File 000001.jpg is frame 0, although it is the last row of the image list.
    assert [frame.frame for frame in labels.frames] == [0, 1, 2]
    assert [frame.file_name for frame in labels.frames] == ["000001.jpg", "000002.jpg", "000003.jpg"]
    assert [frame.image_id for frame in labels.frames] == ["9001", "9002", "9003"]
    assert [frame.scored for frame in labels.frames] == [True, True, False]


def test_frame_index_follows_the_numbers_in_the_file_names(tmp_path: Path) -> None:
    data = label_dict()
    data["info"]["seq_length"] = 12
    # Names without leading zeros, listed from the last to the first. As text, "10.jpg" comes before "2.jpg".
    data["images"] = [image_row(str(9000 + number), f"{number}.jpg") for number in range(12, 0, -1)]

    loaded = load_gsr_labels(write_labels(tmp_path, data))

    # The order is the one in which ImageFolderSource reads the frame files of the clip.
    assert [frame.file_name for frame in loaded.frames] == [f"{number}.jpg" for number in range(1, 13)]
    assert [frame.image_id for frame in loaded.frames] == [str(9000 + number) for number in range(1, 13)]
    assert [frame.frame for frame in loaded.frames] == list(range(12))
    # The athletes of the image 9002 (file 2.jpg) are in frame 1, not in frame 4.
    assert [athlete.track_id for athlete in loaded.frames[1].athletes] == [1, 2, 6, 7]
    assert loaded.frames[4].athletes == ()


@pytest.mark.parametrize("flag", ["has_labeled_person", "has_labeled_pitch", "has_labeled_camera"])
def test_a_frame_is_scored_only_with_person_pitch_and_camera_labels(tmp_path: Path, flag: str) -> None:
    data = label_dict()
    data["images"][0][flag] = False  # file 000002.jpg, frame 1

    loaded = load_gsr_labels(write_labels(tmp_path, data))

    assert [frame.scored for frame in loaded.frames] == [True, False, False]


def test_athletes_of_a_frame_without_the_ball(labels: GsrLabels) -> None:
    athletes = labels.frames[0].athletes

    assert [athlete.track_id for athlete in athletes] == [1, 2, 3, 4]
    assert [athlete.role for athlete in athletes] == ["player", "player", "goalkeeper", "referee"]
    assert [athlete.team for athlete in athletes] == ["left", "right", "left", None]


def test_string_jerseys_are_stored_as_integers(labels: GsrLabels) -> None:
    athletes = labels.frames[0].athletes

    assert [athlete.jersey for athlete in athletes] == [10, 7, 1, None]
    assert all(type(athlete.jersey) is int for athlete in athletes[:3])


def test_image_box_uses_the_top_left_corner_and_the_size(labels: GsrLabels) -> None:
    box = labels.frames[0].athletes[0].box

    # bbox_image: x = 100, y = 200, w = 40, h = 80.
    assert box == Box(x1=100.0, y1=200.0, x2=140.0, y2=280.0)
    assert box is not None and box.foot == (120.0, 280.0)


def test_pitch_position_is_the_bottom_middle_point(labels: GsrLabels) -> None:
    first = labels.frames[0].athletes[0]

    # Not x_bottom_left (-10.3). The player stands on the far side, so y is negative.
    assert (first.pitch_x, first.pitch_y) == (-10.0, -20.5)


def test_missing_box_and_missing_pitch_position_are_none(labels: GsrLabels) -> None:
    by_track = {athlete.track_id: athlete for athlete in labels.frames[1].athletes}

    assert sorted(by_track) == [1, 2, 6, 7]
    assert by_track[2].box == Box(x1=510.0, y1=300.0, x2=560.0, y2=400.0)
    assert (by_track[2].pitch_x, by_track[2].pitch_y) == (None, None)
    assert by_track[6].box is None
    assert (by_track[6].pitch_x, by_track[6].pitch_y) == (0.0, 33.0)
    assert by_track[7].jersey is None


def test_null_box_and_null_pitch_position_are_none(tmp_path: Path) -> None:
    data = label_dict()
    data["annotations"][4]["bbox_image"] = None  # track 1 in frame 0
    data["annotations"][4]["bbox_pitch"] = None

    first = load_gsr_labels(write_labels(tmp_path, data)).frames[0].athletes[0]

    assert first.track_id == 1
    assert (first.box, first.pitch_x, first.pitch_y) == (None, None, None)


def test_zero_is_a_value_and_not_a_missing_key(tmp_path: Path) -> None:
    data = label_dict()
    row = data["annotations"][4]  # track 1 in frame 0
    row["track_id"] = 0
    row["bbox_image"].update({"x": 0, "y": 0})
    row["bbox_pitch"].update({"x_bottom_middle": 0, "y_bottom_middle": 0})

    first = load_gsr_labels(write_labels(tmp_path, data)).frames[0].athletes[0]

    # Track id 0, a box in the top-left image corner and the centre spot are all valid.
    assert first.track_id == 0
    assert first.box == Box(x1=0.0, y1=0.0, x2=40.0, y2=80.0)
    assert (first.pitch_x, first.pitch_y) == (0.0, 0.0)


def test_pitch_lines_are_kept_per_frame(labels: GsrLabels) -> None:
    assert labels.frames[0].lines == {"Middle line": ((0.5, 0.1), (0.52, 0.9)), "Circle central": ()}
    assert labels.frames[1].lines == {}


def test_frame_without_labels_has_no_athletes(labels: GsrLabels) -> None:
    last = labels.frames[2]

    assert last.scored is False
    assert last.athletes == ()
    assert last.lines == {}


def test_clip_id_and_length_come_from_the_folder_and_the_images_without_info(tmp_path: Path) -> None:
    data = label_dict()
    data["info"] = {"frame_rate": "25"}

    loaded = load_gsr_labels(write_labels(tmp_path, data, folder="SNGS-123"))

    assert loaded.clip_id == "SNGS-123"
    assert loaded.n_frames == 3
    assert loaded.fps == 25.0
    assert loaded.version is None


@pytest.mark.parametrize("seq_length", [3, "3", 3.0])
def test_length_of_info_may_be_a_number_or_a_string_of_digits(tmp_path: Path, seq_length: Any) -> None:
    data = label_dict()
    data["info"]["seq_length"] = seq_length

    loaded = load_gsr_labels(write_labels(tmp_path, data))

    assert loaded.n_frames == 3
    assert type(loaded.n_frames) is int


@pytest.mark.parametrize("seq_length", [750, 2])
def test_length_of_info_must_be_the_number_of_images(tmp_path: Path, seq_length: int) -> None:
    data = label_dict()
    data["info"]["seq_length"] = seq_length  # the file lists 3 images

    with pytest.raises(ValueError, match=f"info: seq_length is {seq_length}, but the label file lists 3 images"):
        load_gsr_labels(write_labels(tmp_path, data))


@pytest.mark.parametrize("track_id", [12, 12.0, "12"])
def test_track_id_may_be_a_number_without_a_fraction_or_a_string_of_digits(tmp_path: Path, track_id: Any) -> None:
    data = label_dict()
    data["annotations"][4]["track_id"] = track_id  # track 1 in frame 0

    first = load_gsr_labels(write_labels(tmp_path, data)).frames[0].athletes[0]

    assert first.track_id == 12
    assert type(first.track_id) is int


def test_the_ball_may_have_the_track_id_of_an_athlete(tmp_path: Path) -> None:
    data = label_dict()
    data["annotations"][8]["track_id"] = 1  # the ball of frame 0 gets the track id of the first player

    athletes = load_gsr_labels(write_labels(tmp_path, data)).frames[0].athletes

    # The ball is not an athlete, so the track id 1 is still used once.
    assert [athlete.track_id for athlete in athletes] == [1, 2, 3, 4]


def test_number_jerseys_are_read_and_are_not_marked_as_strings(tmp_path: Path) -> None:
    data = label_dict()
    for row in data["annotations"]:
        if row.get("attributes", {}).get("jersey") is not None:
            row["attributes"]["jersey"] = int(row["attributes"]["jersey"])

    loaded = load_gsr_labels(write_labels(tmp_path, data))

    assert loaded.jersey_is_str is False
    assert [athlete.jersey for athlete in loaded.frames[0].athletes] == [10, 7, 1, None]


def test_one_string_jersey_marks_the_file_as_string(tmp_path: Path) -> None:
    data = label_dict()
    for row in data["annotations"]:
        if row.get("attributes", {}).get("jersey") is not None:
            row["attributes"]["jersey"] = int(row["attributes"]["jersey"])
    data["annotations"][5]["attributes"]["jersey"] = "7"  # track 2 in frame 0

    loaded = load_gsr_labels(write_labels(tmp_path, data))

    assert loaded.jersey_is_str is True
    assert [athlete.jersey for athlete in loaded.frames[0].athletes] == [10, 7, 1, None]


def test_file_without_any_jersey_is_not_marked_as_string(tmp_path: Path) -> None:
    data = label_dict()
    for row in data["annotations"]:
        if "attributes" in row:
            row["attributes"]["jersey"] = None

    assert load_gsr_labels(write_labels(tmp_path, data)).jersey_is_str is False


def _jersey_with_a_letter(data: dict[str, Any]) -> None:
    data["annotations"][0]["attributes"]["jersey"] = "1a"


def _jersey_is_a_truth_value(data: dict[str, Any]) -> None:
    data["annotations"][0]["attributes"]["jersey"] = True


def _jersey_is_negative(data: dict[str, Any]) -> None:
    data["annotations"][0]["attributes"]["jersey"] = -4


def _jersey_with_a_fraction(data: dict[str, Any]) -> None:
    data["annotations"][0]["attributes"]["jersey"] = 7.5


def _jersey_is_a_digit_of_another_script(data: dict[str, Any]) -> None:
    data["annotations"][0]["attributes"]["jersey"] = "\u0663"  # the Arabic-Indic digit three


def _annotation_of_an_unknown_image(data: dict[str, Any]) -> None:
    data["annotations"][0]["image_id"] = "404"


def _no_images(data: dict[str, Any]) -> None:
    data["images"] = []


def _box_without_area(data: dict[str, Any]) -> None:
    data["annotations"][0]["bbox_image"]["w"] = 0


def _image_id_is_a_number(data: dict[str, Any]) -> None:
    data["images"][0]["image_id"] = 9002


def _two_images_with_one_id(data: dict[str, Any]) -> None:
    # The image 9003 has no annotation, so every annotation still belongs to an image of the list.
    data["images"][1]["image_id"] = "9001"


def _scored_flag_is_missing(data: dict[str, Any]) -> None:
    del data["images"][0]["has_labeled_camera"]


def _scored_flag_is_text(data: dict[str, Any]) -> None:
    data["images"][0]["has_labeled_camera"] = "false"


def _no_frame_rate(data: dict[str, Any]) -> None:
    del data["info"]["frame_rate"]


def _no_track_id(data: dict[str, Any]) -> None:
    del data["annotations"][0]["track_id"]


def _no_role(data: dict[str, Any]) -> None:
    del data["annotations"][0]["attributes"]["role"]


def _pitch_position_without_the_middle_point(data: dict[str, Any]) -> None:
    del data["annotations"][0]["bbox_pitch"]["x_bottom_middle"]


def _two_pitch_annotations_for_one_image(data: dict[str, Any]) -> None:
    data["annotations"].append(copy.deepcopy(data["annotations"][9]))


def _no_annotations(data: dict[str, Any]) -> None:
    del data["annotations"]


def _image_name_is_not_a_number(data: dict[str, Any]) -> None:
    data["images"][0]["file_name"] = "frame_b.jpg"


def _two_images_with_one_number(data: dict[str, Any]) -> None:
    data["images"][0]["file_name"] = "1.jpg"  # the same frame as 000001.jpg


def _length_differs_from_the_images(data: dict[str, Any]) -> None:
    data["info"]["seq_length"] = 750


def _length_with_a_fraction(data: dict[str, Any]) -> None:
    data["info"]["seq_length"] = 3.5


def _frame_rate_of_zero(data: dict[str, Any]) -> None:
    data["info"]["frame_rate"] = 0


def _negative_frame_rate(data: dict[str, Any]) -> None:
    data["info"]["frame_rate"] = -25


def _frame_rate_is_not_finite(data: dict[str, Any]) -> None:
    data["info"]["frame_rate"] = float("inf")


def _track_id_with_a_fraction(data: dict[str, Any]) -> None:
    data["annotations"][0]["track_id"] = 2.7


def _track_id_is_a_truth_value(data: dict[str, Any]) -> None:
    data["annotations"][0]["track_id"] = True


def _track_id_is_a_word(data: dict[str, Any]) -> None:
    data["annotations"][0]["track_id"] = "one"


def _track_id_twice_in_one_image(data: dict[str, Any]) -> None:
    data["annotations"][1]["track_id"] = 1  # the track id of annotation 0, in the same image


def _pitch_position_is_not_a_number(data: dict[str, Any]) -> None:
    data["annotations"][0]["bbox_pitch"]["x_bottom_middle"] = float("nan")


def _pitch_position_is_infinite(data: dict[str, Any]) -> None:
    data["annotations"][0]["bbox_pitch"]["y_bottom_middle"] = float("-inf")


#: Wrong content of a label file: a change of the hand-made labels, and the message of the reader for it.
BAD_CONTENT: dict[str, tuple[Callable[[dict[str, Any]], None], str]] = {
    "jersey with a letter": (
        _jersey_with_a_letter,
        "annotations[0]: jersey '1a' is neither a string of digits nor a whole number",
    ),
    "jersey is a truth value": (
        _jersey_is_a_truth_value,
        "annotations[0]: jersey True is neither a string of digits nor a whole number",
    ),
    "jersey is negative": (
        _jersey_is_negative,
        "annotations[0]: jersey -4 is neither a string of digits nor a whole number",
    ),
    "jersey with a fraction": (
        _jersey_with_a_fraction,
        "annotations[0]: jersey 7.5 is neither a string of digits nor a whole number",
    ),
    "jersey is a digit of another script": (
        _jersey_is_a_digit_of_another_script,
        "annotations[0]: jersey '\u0663' is neither a string of digits nor a whole number",
    ),
    "annotation of an unknown image": (
        _annotation_of_an_unknown_image,
        "annotations[0]: image_id '404' is not in the image list",
    ),
    "no images": (_no_images, "the label file lists no images"),
    "box without area": (_box_without_area, "annotations[0]: bbox_image has no area (w=0.0, h=80.0)"),
    "image id is a number": (_image_id_is_a_number, "image '000002.jpg': image_id must be a string, found 9002"),
    "two images with one id": (_two_images_with_one_id, "two images have the same image_id"),
    "scored flag is missing": (_scored_flag_is_missing, "image: the key 'has_labeled_camera' is missing or null"),
    "scored flag is text": (
        _scored_flag_is_text,
        "image '000002.jpg': the has_labeled flags must be true or false",
    ),
    "no frame rate": (_no_frame_rate, "info: the key 'frame_rate' is missing or null"),
    "no track id": (_no_track_id, "annotations[0]: the key 'track_id' is missing or null"),
    "no role": (_no_role, "annotations[0]: the key 'role' is missing or null"),
    "pitch position without the middle point": (
        _pitch_position_without_the_middle_point,
        "annotations[0] bbox_pitch: the key 'x_bottom_middle' is missing or null",
    ),
    "two pitch annotations for one image": (
        _two_pitch_annotations_for_one_image,
        "annotations[11]: image '9001' has more than one pitch annotation",
    ),
    "no annotations": (_no_annotations, "label file: the key 'annotations' is missing or null"),
    "image name is not a number": (
        _image_name_is_not_a_number,
        "image 'frame_b.jpg': the name of an image file must be a number, like 000001.jpg",
    ),
    "two images with one number": (
        _two_images_with_one_number,
        "the images '1.jpg' and '000001.jpg' have the same number",
    ),
    "length differs from the images": (
        _length_differs_from_the_images,
        "info: seq_length is 750, but the label file lists 3 images",
    ),
    "length with a fraction": (_length_with_a_fraction, "info: seq_length 3.5 is not a whole number"),
    "frame rate of zero": (_frame_rate_of_zero, "info: frame_rate must be a number above 0, found 0.0"),
    "negative frame rate": (_negative_frame_rate, "info: frame_rate must be a number above 0, found -25.0"),
    "frame rate is not finite": (_frame_rate_is_not_finite, "info: frame_rate must be a number above 0, found inf"),
    "track id with a fraction": (_track_id_with_a_fraction, "annotations[0]: track_id 2.7 is not a whole number"),
    "track id is a truth value": (
        _track_id_is_a_truth_value,
        "annotations[0]: track_id True is not a whole number",
    ),
    "track id is a word": (_track_id_is_a_word, "annotations[0]: track_id 'one' is not a whole number"),
    "track id twice in one image": (
        _track_id_twice_in_one_image,
        "annotations[1]: track_id 1 is used twice in image '9002'",
    ),
    "pitch position is not a number": (
        _pitch_position_is_not_a_number,
        "annotations[0]: the pitch position of bbox_pitch must be finite, found x=nan and y=-20.0",
    ),
    "pitch position is infinite": (
        _pitch_position_is_infinite,
        "annotations[0]: the pitch position of bbox_pitch must be finite, found x=-9.0 and y=-inf",
    ),
}


@pytest.mark.parametrize(("change", "message"), list(BAD_CONTENT.values()), ids=list(BAD_CONTENT))
def test_load_refuses_wrong_content_and_names_the_file_and_the_place(
    tmp_path: Path, change: Callable[[dict[str, Any]], None], message: str
) -> None:
    """Every kind of wrong content has its own message, so a rule that is switched off cannot hide behind another."""
    data = label_dict()
    change(data)

    with pytest.raises(ValueError, match=re.escape(f"Labels-GameState.json: {message}")):
        load_gsr_labels(write_labels(tmp_path, data))


def test_load_names_the_annotation_with_a_wrong_jersey(tmp_path: Path) -> None:
    data = label_dict()
    data["annotations"][3]["attributes"]["jersey"] = "1a"

    with pytest.raises(ValueError, match=r"annotations\[3\]: jersey '1a'"):
        load_gsr_labels(write_labels(tmp_path, data))


@pytest.mark.parametrize("content", [[], "labels", 12])
def test_load_refuses_a_file_that_is_not_a_mapping(tmp_path: Path, content: Any) -> None:
    with pytest.raises(ValueError, match="expected a mapping"):
        load_gsr_labels(write_labels(tmp_path, content))


def test_load_refuses_text_that_is_not_json(tmp_path: Path) -> None:
    path = tmp_path / "Labels-GameState.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="Labels-GameState.json"):
        load_gsr_labels(path)


def test_load_of_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_gsr_labels(tmp_path / "missing" / "Labels-GameState.json")


def test_labels_must_hold_the_frames_in_frame_order(labels: GsrLabels) -> None:
    with pytest.raises(ValidationError, match="frame order"):
        GsrLabels(
            clip_id="clip",
            fps=25.0,
            n_frames=2,
            width=1920,
            height=1080,
            frames=(labels.frames[1], labels.frames[0]),
            jersey_is_str=True,
            version=None,
        )


def test_label_records_are_frozen(labels: GsrLabels) -> None:
    with pytest.raises(ValidationError):
        labels.frames[0].athletes[0].jersey = 99  # type: ignore[misc]


# --- Ground truth in the internal schema -----------------------------------------


def test_frame_states_have_one_state_per_frame(labels: GsrLabels) -> None:
    states = gsr_frame_states(labels)

    assert [state.frame for state in states] == [0, 1, 2]
    assert [player.track_id for player in states[0].players] == [1, 2, 3, 4]
    # Track 2 has no pitch position in frame 1, so it is skipped there.
    assert [player.track_id for player in states[1].players] == [1, 6, 7]
    assert states[2].players == ()


def test_frame_state_of_a_player(labels: GsrLabels) -> None:
    player = gsr_frame_states(labels)[0].players[0]

    assert player == PlayerState(
        track_id=1,
        box=Box(x1=100.0, y1=200.0, x2=140.0, y2=280.0),
        pitch_x=-10.0,
        pitch_y=-20.5,
        role="player",
        team_key="none",
        team_side="left",
        number_export=10,
        number_display=None,
        name=None,
        observed=True,
    )
    # x is first and y is second; a far-side player has a negative y.
    assert player.pitch_x == -10.0
    assert player.pitch_y < 0.0


def test_team_side_and_number_follow_the_role(labels: GsrLabels) -> None:
    players = {player.track_id: player for player in gsr_frame_states(labels)[0].players}

    assert (players[2].role, players[2].team_side, players[2].number_export) == ("player", "right", 7)
    # A goalkeeper has a team side, but the export number is for players only.
    assert (players[3].role, players[3].team_side, players[3].number_export) == ("goalkeeper", "left", None)
    assert (players[4].role, players[4].team_side, players[4].number_export) == ("referee", None, None)
    assert all(player.team_key == "none" for player in players.values())
    assert all(player.observed for player in players.values())
    assert all(player.number_display is None and player.name is None for player in players.values())


def test_frame_state_without_a_box_and_without_a_number(labels: GsrLabels) -> None:
    players = {player.track_id: player for player in gsr_frame_states(labels)[1].players}

    assert (players[6].role, players[6].box, players[6].team_side) == ("other", None, None)
    # Near the near touchline: y is positive.
    assert (players[6].pitch_x, players[6].pitch_y) == (0.0, 33.0)
    assert (players[7].team_side, players[7].number_export) == ("right", None)


def test_frame_states_and_frame_people_refuse_an_unknown_role(tmp_path: Path) -> None:
    data = label_dict()
    data["annotations"][4]["attributes"]["role"] = "coach"  # track 1 in frame 0
    loaded = load_gsr_labels(write_labels(tmp_path, data))

    # The reader keeps the role text of the file.
    assert loaded.frames[0].athletes[0].role == "coach"
    with pytest.raises(ValueError, match="frame 0, track 1: unknown role 'coach'"):
        gsr_frame_states(loaded)
    with pytest.raises(ValueError, match="frame 0, track 1: unknown role 'coach'"):
        gsr_frame_people(loaded, 0)


def test_frame_states_refuse_an_unknown_team(tmp_path: Path) -> None:
    data = label_dict()
    data["annotations"][4]["attributes"]["team"] = "home"  # track 1 in frame 0
    loaded = load_gsr_labels(write_labels(tmp_path, data))

    with pytest.raises(ValueError, match="frame 0, track 1: unknown team 'home'"):
        gsr_frame_states(loaded)


def test_frame_states_refuse_a_player_number_above_99(tmp_path: Path) -> None:
    data = label_dict()
    data["annotations"][4]["attributes"]["jersey"] = "100"  # track 1 in frame 0, a player
    loaded = load_gsr_labels(write_labels(tmp_path, data))

    # The reader keeps the number of the file. The schema of the frame states holds 0 to 99.
    assert loaded.frames[0].athletes[0].jersey == 100
    with pytest.raises(ValueError, match="number_export"):
        gsr_frame_states(loaded)
    with pytest.raises(ValueError, match="number"):
        gsr_frame_people(loaded, 0)


def test_frame_states_do_not_use_the_number_of_a_goalkeeper(tmp_path: Path) -> None:
    data = label_dict()
    data["annotations"][6]["attributes"]["jersey"] = "100"  # track 3 in frame 0, a goalkeeper
    loaded = load_gsr_labels(write_labels(tmp_path, data))

    assert (loaded.frames[0].athletes[2].role, loaded.frames[0].athletes[2].jersey) == ("goalkeeper", 100)
    # The export number is for players only, so this number is not read and raises no error.
    assert gsr_frame_states(loaded)[0].players[2].number_export is None
    # The people schema holds the number of every role, so there it is an error.
    with pytest.raises(ValueError, match="number"):
        gsr_frame_people(loaded, 0)


def test_frame_states_refuse_a_pitch_position_that_is_not_finite(labels: GsrLabels) -> None:
    athlete = labels.frames[0].athletes[0].model_copy(update={"pitch_x": float("inf")})
    frame = labels.frames[0].model_copy(update={"athletes": (athlete,)})
    broken = labels.model_copy(update={"frames": (frame, *labels.frames[1:])})

    with pytest.raises(ValueError, match="pitch_x"):
        gsr_frame_states(broken)


def test_frame_people_of_a_labelled_frame(labels: GsrLabels) -> None:
    people = gsr_frame_people(labels, 0)

    assert people.frame == 0
    assert [person.role for person in people.people] == ["player", "player", "goalkeeper", "referee"]
    # The label team is carried in the team key: left is A, right is B.
    assert [person.team_key for person in people.people] == ["A", "B", "A", "none"]
    assert [person.number for person in people.people] == [10, 7, 1, None]
    assert people.people[0].box == Box(x1=100.0, y1=200.0, x2=140.0, y2=280.0)


def test_frame_people_skip_athletes_without_a_box(labels: GsrLabels) -> None:
    people = gsr_frame_people(labels, 1)

    # Track 6 has no image box. Track 2 has a box but no pitch position, so it stays.
    assert [person.box.x1 for person in people.people] == [110.0, 510.0, 700.0]
    assert [person.role for person in people.people] == ["player", "player", "player"]


def test_frame_people_of_a_frame_without_labels_are_empty(labels: GsrLabels) -> None:
    assert gsr_frame_people(labels, 2) == FramePeople(frame=2)


@pytest.mark.parametrize("frame", [-1, 3])
def test_frame_people_refuse_a_frame_outside_the_clip(labels: GsrLabels, frame: int) -> None:
    with pytest.raises(ValueError, match="not in the labels"):
        gsr_frame_people(labels, frame)
