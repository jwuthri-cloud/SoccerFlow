"""Tests of the synthetic scene: camera geometry, sign conventions and ground-truth records.

The camera path of a scene depends on time only. In the default clip (25 frames per second)
frame 0 looks at the left side, frame 75 has pan angle zero and frame 99 looks a little to
the right of the centre.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pitchvlm import synth
from pitchvlm.pitch import HALF_LENGTH, HALF_WIDTH, LANDMARKS, inside_pitch, landmark_xy
from pitchvlm.schemas import Box, ClipInfo
from pitchvlm.synth import SyntheticScene, SyntheticSource

#: Frames of the default clip that the geometry tests look at.
FRAMES = (0, 30, 75, 99)
#: The frame of the default clip in which the pan angle is zero.
CENTRE_FRAME = 75


def apply(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to points of shape (n, 2) and return shape (n, 2)."""
    points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    mapped = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    return mapped[:, :2] / mapped[:, 2:3]


def is_whole(box: Box, info: ClipInfo) -> bool:
    """Return True when the box does not touch the image border, so it was not cut."""
    return box.x1 > 0.0 and box.y1 > 0.0 and box.x2 < info.width and box.y2 < info.height


def count_colour(image: np.ndarray, bgr: tuple[int, int, int]) -> int:
    """Count the pixels of an image that have exactly one colour."""
    return int(np.all(image == np.array(bgr, dtype=np.uint8), axis=2).sum())


@pytest.fixture(scope="module")
def scene() -> SyntheticScene:
    return SyntheticScene(seed=7)


# --- Clip and frame source -----------------------------------------------------------


def test_info_describes_the_clip(scene: SyntheticScene) -> None:
    assert scene.info == ClipInfo(clip_id=synth.CLIP_ID, n_frames=100, fps=25.0, width=1280, height=720)
    small = SyntheticScene(n_frames=10, fps=30.0, width=640, height=360, seed=1)
    assert (small.info.n_frames, small.info.fps, small.info.width, small.info.height) == (10, 30.0, 640, 360)


def test_source_reads_the_images_of_the_scene(scene: SyntheticScene) -> None:
    source = SyntheticSource(scene)
    assert source.info == scene.info
    assert np.array_equal(source.read(4), scene.image(4))


def test_wrong_input_raises(scene: SyntheticScene) -> None:
    with pytest.raises(IndexError):
        scene.image(scene.info.n_frames)
    with pytest.raises(IndexError):
        scene.people(-1)
    with pytest.raises(IndexError):
        scene.homography(1000)
    with pytest.raises(IndexError):
        scene.true_number(-1, Box(x1=0.0, y1=0.0, x2=5.0, y2=5.0))
    with pytest.raises(ValueError):
        SyntheticScene(n_frames=0)
    with pytest.raises(ValueError):
        SyntheticScene(seed=-1)


# --- Camera and homography -----------------------------------------------------------


@pytest.mark.parametrize("frame", FRAMES)
def test_homography_round_trip(scene: SyntheticScene, frame: int) -> None:
    image_to_pitch = scene.homography(frame)
    assert image_to_pitch.shape == (3, 3)
    assert image_to_pitch.dtype == np.float64
    grid_x, grid_y = np.meshgrid(np.linspace(-HALF_LENGTH, HALF_LENGTH, 8), np.linspace(-HALF_WIDTH, HALF_WIDTH, 6))
    pitch_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    pixels = apply(np.linalg.inv(image_to_pitch), pitch_points)
    assert np.abs(apply(image_to_pitch, pixels) - pitch_points).max() < 1e-6


def test_left_goal_is_left_and_far_side_is_up_in_the_centre_view(scene: SyntheticScene) -> None:
    """In frame 75 the camera looks straight at the halfway line."""
    width, height = scene.info.width, scene.info.height
    image_to_pitch = scene.homography(CENTRE_FRAME)
    centre = apply(image_to_pitch, np.array([[width / 2.0, height / 2.0]]))[0]
    assert centre.tolist() == pytest.approx([0.0, synth.AIM_Y_M], abs=1e-6)

    pitch_to_image = np.linalg.inv(image_to_pitch)
    left_goal, right_goal = apply(pitch_to_image, np.array([[-HALF_LENGTH, 0.0], [HALF_LENGTH, 0.0]]))
    assert left_goal[0] < width / 2.0 < right_goal[0]
    # With pan angle zero the picture is symmetric about its middle column.
    assert left_goal[0] + right_goal[0] == pytest.approx(width, abs=1e-6)

    far, middle, near = apply(pitch_to_image, np.array([[0.0, -HALF_WIDTH], [0.0, 0.0], [0.0, HALF_WIDTH]]))
    assert far[1] < middle[1] < near[1]
    assert far[0] == pytest.approx(width / 2.0, abs=1e-6)
    assert near[0] == pytest.approx(width / 2.0, abs=1e-6)


@pytest.mark.parametrize("frame", FRAMES)
def test_image_axes_follow_the_pitch_axes(scene: SyntheticScene, frame: int) -> None:
    """Right in the image is towards the right goal. Down in the image is towards the camera."""
    width, height = scene.info.width, scene.info.height
    image_to_pitch = scene.homography(frame)
    steps = np.array([[0.0, 0.0], [50.0, 0.0], [0.0, 50.0]])
    at_centre, at_right, at_below = apply(image_to_pitch, np.array([width / 2.0, height / 2.0]) + steps)
    assert at_right[0] > at_centre[0]
    assert at_below[1] > at_centre[1]

    pitch_to_image = np.linalg.inv(image_to_pitch)
    left_goal, right_goal = apply(pitch_to_image, np.array([[-HALF_LENGTH, 0.0], [HALF_LENGTH, 0.0]]))
    assert left_goal[0] < right_goal[0]
    for x in (-HALF_LENGTH, 0.0, HALF_LENGTH):
        far, near = apply(pitch_to_image, np.array([[x, -HALF_WIDTH], [x, HALF_WIDTH]]))
        assert far[1] < near[1]


def test_camera_moves_and_does_not_depend_on_the_seed(scene: SyntheticScene) -> None:
    other = SyntheticScene(seed=8)
    assert np.array_equal(other.homography(40), scene.homography(40))
    assert other.frame_state(40) != scene.frame_state(40)
    centre = np.array([[scene.info.width / 2.0, scene.info.height / 2.0]])
    start_x = apply(scene.homography(0), centre)[0, 0]
    end_x = apply(scene.homography(99), centre)[0, 0]
    # The camera pans from the left side to the right of the halfway line.
    assert start_x < -15.0
    assert end_x > 5.0


# --- People ------------------------------------------------------------------------------


def test_people_boxes_stay_inside_the_image(scene: SyntheticScene) -> None:
    width, height = scene.info.width, scene.info.height
    cut_boxes = 0
    for frame in range(scene.info.n_frames):
        people = scene.people(frame)
        assert people.frame == frame
        assert 8 <= len(people.people) <= 23
        for person in people.people:
            box = person.box
            assert 0.0 <= box.x1 < box.x2 <= width
            assert 0.0 <= box.y1 < box.y2 <= height
            assert box.width >= synth.MIN_BOX_PX
            assert box.height >= synth.MIN_BOX_PX
            if is_whole(box, scene.info):
                assert box.width == pytest.approx(synth.BOX_ASPECT * box.height, abs=1e-9)
            else:
                cut_boxes += 1
    # The camera pans, so some people stand on the image border in some frames.
    assert cut_boxes > 0


@pytest.mark.parametrize("frame", FRAMES)
def test_people_have_the_right_roles_teams_and_numbers(scene: SyntheticScene, frame: int) -> None:
    people = scene.people(frame).people
    for team_key in ("A", "B"):
        keepers = [person for person in people if person.team_key == team_key and person.role == "goalkeeper"]
        players = [person for person in people if person.team_key == team_key and person.role == "player"]
        assert len(keepers) <= 1
        assert all(keeper.number == 1 for keeper in keepers)
        numbers = [player.number for player in players]
        assert len(numbers) == len(set(numbers))
        assert all(number is not None and 2 <= number <= 11 for number in numbers)
    without_team = [person for person in people if person.team_key == "none"]
    assert len(without_team) <= 1
    assert all(person.role == "referee" and person.number is None for person in without_team)
    assert all(person.team_key in ("A", "B") for person in people if person.role != "referee")


@pytest.mark.parametrize("frame", FRAMES)
def test_far_boxes_are_smaller_than_near_boxes(scene: SyntheticScene, frame: int) -> None:
    whole = [player for player in scene.frame_state(frame).players if is_whole(player.box, scene.info)]
    far = min(whole, key=lambda player: player.pitch_y)
    near = max(whole, key=lambda player: player.pitch_y)
    assert far.pitch_y < -10.0
    assert near.pitch_y > 10.0
    assert far.box.height < 0.8 * near.box.height
    assert far.box.width < 0.8 * near.box.width


@pytest.mark.parametrize("frame", FRAMES)
def test_feet_of_whole_boxes_map_to_the_true_pitch_position(scene: SyntheticScene, frame: int) -> None:
    whole = [player for player in scene.frame_state(frame).players if is_whole(player.box, scene.info)]
    assert len(whole) >= 2
    feet = np.array([player.box.foot for player in whole])
    truth = np.array([[player.pitch_x, player.pitch_y] for player in whole])
    assert np.abs(apply(scene.homography(frame), feet) - truth).max() < 1e-6


@pytest.mark.parametrize("frame", FRAMES)
def test_frame_state_matches_people_and_has_the_right_signs(scene: SyntheticScene, frame: int) -> None:
    state = scene.frame_state(frame)
    people = scene.people(frame).people
    assert state.frame == frame
    assert len(state.players) == len(people)
    track_ids = [player.track_id for player in state.players]
    assert track_ids == sorted(set(track_ids))
    side_of_team = {"A": "left", "B": "right", "none": None}
    for person, player in zip(people, state.players, strict=True):
        assert player.box == person.box
        assert (player.role, player.team_key) == (person.role, person.team_key)
        assert player.number_export == person.number
        assert player.number_display == person.number
        assert player.team_side == side_of_team[player.team_key]
        assert player.observed is True
        assert player.name is None
    # The person highest in the image is on the far side, so the pitch y is negative.
    # The person lowest in the image is on the near side, so the pitch y is positive.
    highest = min(state.players, key=lambda player: player.box.y2)
    lowest = max(state.players, key=lambda player: player.box.y2)
    assert highest.pitch_y < 0.0 < lowest.pitch_y


def test_track_ids_are_the_fixed_indices_of_the_people(scene: SyntheticScene) -> None:
    seen: dict[int, tuple[str, str, int | None]] = {}
    for frame in range(scene.info.n_frames):
        for player in scene.frame_state(frame).players:
            identity = (player.role, player.team_key, player.number_export)
            assert seen.setdefault(player.track_id, identity) == identity
    assert seen[0] == ("goalkeeper", "A", 1)
    for track_id, (role, team_key, number) in seen.items():
        if track_id == 22:
            assert (role, team_key, number) == ("referee", "none", None)
        else:
            assert 0 <= track_id < 22
            assert team_key == ("A" if track_id < 11 else "B")
            assert number == track_id % 11 + 1
            assert role == ("goalkeeper" if number == 1 else "player")


def test_people_move_smoothly_and_stay_on_the_pitch(scene: SyntheticScene) -> None:
    previous: dict[int, tuple[float, float]] = {}
    for frame in range(scene.info.n_frames):
        players = scene.frame_state(frame).players
        assert inside_pitch(np.array([[player.pitch_x, player.pitch_y] for player in players])).all()
        current: dict[int, tuple[float, float]] = {}
        for player in players:
            if player.role == "goalkeeper":
                # Team A defends the left goal and team B the right goal.
                own_goal_x = -HALF_LENGTH if player.team_key == "A" else HALF_LENGTH
                assert abs(player.pitch_x - own_goal_x) < 6.0
                assert abs(player.pitch_y) < 5.0
            current[player.track_id] = (player.pitch_x, player.pitch_y)
            if player.track_id in previous:
                # 0.25 m in one frame is 6.25 m/s at 25 frames per second.
                assert math.dist(previous[player.track_id], current[player.track_id]) < 0.25
        previous = current


def test_team_a_plays_on_the_left(scene: SyntheticScene) -> None:
    for frame in range(0, scene.info.n_frames, 5):
        players = [player for player in scene.frame_state(frame).players if player.role == "player"]
        team_a_x = [player.pitch_x for player in players if player.team_key == "A"]
        team_b_x = [player.pitch_x for player in players if player.team_key == "B"]
        assert team_a_x and team_b_x
        assert sum(team_a_x) / len(team_a_x) < sum(team_b_x) / len(team_b_x)


def test_true_number_finds_the_person_by_overlap(scene: SyntheticScene) -> None:
    frame = 0
    for person in scene.people(frame).people:
        assert scene.true_number(frame, person.box) == person.number

    # Goalkeeper A stands alone near the left goal in frame 0.
    box = next(player.box for player in scene.frame_state(frame).players if player.track_id == 0)

    def moved(fraction: float) -> Box:
        shift = fraction * box.width
        return Box(x1=box.x1 + shift, y1=box.y1, x2=box.x2 + shift, y2=box.y2)

    assert scene.true_number(frame, moved(0.1)) == 1  # overlap 0.9 / 1.1 = 0.82
    assert scene.true_number(frame, moved(0.6)) is None  # overlap 0.4 / 1.6 = 0.25, below 0.3
    assert scene.true_number(frame, moved(2.0)) is None  # no overlap
    assert scene.true_number(frame, Box(x1=0.0, y1=0.0, x2=5.0, y2=5.0)) is None

    referee = next(player for player in scene.frame_state(60).players if player.role == "referee")
    assert scene.true_number(60, referee.box) is None


def test_kit_legend_names_the_colours_of_the_scene(scene: SyntheticScene) -> None:
    legend = scene.kit_legend()
    assert (legend.team_a.shirt, legend.team_a.shorts, legend.team_a.socks) == ("red", "red", "red")
    assert (legend.team_b.shirt, legend.team_b.shorts, legend.team_b.socks) == ("blue", "blue", "blue")
    assert [(keeper.kit, keeper.team_key) for keeper in legend.goalkeepers] == [("yellow", "A"), ("green", "B")]
    assert legend.referee_kit == "black"
    assert legend.uncertain == ()
    assert "Team A: red shirt" in legend.prompt_text()


def test_kit_colours_are_in_bgr_order() -> None:
    blue, green, red = synth.TEAM_A_KIT.fill_bgr
    assert red > 200 and blue < 100 and green < 100
    blue, green, red = synth.TEAM_B_KIT.fill_bgr
    assert blue > 200 and red < 100
    blue, green, red = synth.KEEPER_A_KIT.fill_bgr  # yellow is red plus green
    assert red > 200 and green > 200 and blue < 100
    blue, green, red = synth.KEEPER_B_KIT.fill_bgr
    assert green > 200 and green > red and green > blue
    assert max(synth.REFEREE_KIT.fill_bgr) < 50


# --- Landmarks -------------------------------------------------------------------------


@pytest.mark.parametrize("frame", [*FRAMES, 60])
def test_landmarks_agree_with_the_homography(scene: SyntheticScene, frame: int) -> None:
    width, height = scene.info.width, scene.info.height
    seen = scene.landmarks(frame)
    assert seen.frame == frame
    ids = [landmark.landmark_id for landmark in seen.landmarks]
    assert len(ids) >= 4
    pixels = np.array([[landmark.x, landmark.y] for landmark in seen.landmarks])
    assert np.abs(apply(scene.homography(frame), pixels) - landmark_xy(ids)).max() < 1e-6

    # Exactly the landmarks whose image point is inside the picture are listed, in table order.
    all_ids = [landmark.id for landmark in LANDMARKS]
    projected = apply(np.linalg.inv(scene.homography(frame)), landmark_xy(all_ids))
    inside = [
        landmark_id
        for landmark_id, (x, y) in zip(all_ids, projected.tolist(), strict=True)
        if 0.0 <= x <= width and 0.0 <= y <= height
    ]
    assert ids == inside


def test_landmarks_show_the_side_that_the_camera_looks_at(scene: SyntheticScene) -> None:
    id_of = {landmark.name: landmark.id for landmark in LANDMARKS}
    at_start = {landmark.landmark_id for landmark in scene.landmarks(0).landmarks}
    at_end = {landmark.landmark_id for landmark in scene.landmarks(99).landmarks}
    assert id_of["left penalty spot"] in at_start
    assert id_of["right penalty spot"] not in at_start
    assert id_of["right penalty spot"] in at_end
    assert id_of["left penalty spot"] not in at_end
    assert id_of["centre spot"] in at_start
    assert id_of["centre spot"] in at_end


def test_view_follows_the_pitch_x_of_the_image_centre(scene: SyntheticScene) -> None:
    centre = np.array([[scene.info.width / 2.0, scene.info.height / 2.0]])
    for frame in range(0, scene.info.n_frames, 3):
        centre_x = apply(scene.homography(frame), centre)[0, 0]
        expected = "left" if centre_x < -17.5 else "right" if centre_x > 17.5 else "centre"
        assert scene.landmarks(frame).view == expected
    assert scene.landmarks(0).view == "left"
    assert scene.landmarks(CENTRE_FRAME).view == "centre"
    long_scene = SyntheticScene(n_frames=176, seed=7)
    assert long_scene.landmarks(175).view == "right"


# --- Label file ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def short_scene() -> SyntheticScene:
    return SyntheticScene(n_frames=12, seed=5)


def test_gsr_labels_have_the_structure_of_a_label_file(short_scene: SyntheticScene) -> None:
    labels = short_scene.gsr_labels()
    assert set(labels) == {"info", "images", "annotations", "categories"}
    assert labels["info"] == {
        "name": synth.CLIP_ID,
        "frame_rate": 25.0,
        "seq_length": 12,
        "im_dir": "img1",
        "im_ext": ".jpg",
    }
    assert len(labels["images"]) == 12
    for frame, image in enumerate(labels["images"]):
        assert image["image_id"] == f"2999{frame + 1:06d}"
        assert len(image["image_id"]) == 10 and image["image_id"].isdigit()
        assert image["file_name"] == f"{frame + 1:06d}.jpg"
        assert (image["width"], image["height"]) == (1280, 720)
        for flag in ("is_labeled", "has_labeled_person", "has_labeled_pitch", "has_labeled_camera"):
            assert image[flag] is True
    category_ids = [category["id"] for category in labels["categories"]]
    assert len(category_ids) == len(set(category_ids))
    category_names = {category["name"] for category in labels["categories"]}
    assert {"player", "goalkeeper", "referee", "ball", "other"} <= category_names
    # The content is plain JSON data.
    assert json.loads(json.dumps(labels)) == labels


def test_gsr_annotations_match_the_true_state(short_scene: SyntheticScene) -> None:
    labels = short_scene.gsr_labels()
    annotation_ids = [annotation["id"] for annotation in labels["annotations"]]
    assert all(isinstance(annotation_id, str) for annotation_id in annotation_ids)
    assert len(annotation_ids) == len(set(annotation_ids))
    category_name = {category["id"]: category["name"] for category in labels["categories"]}
    by_image: dict[str, list[dict]] = {}
    for annotation in labels["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)
    assert set(by_image) == {image["image_id"] for image in labels["images"]}

    for frame, image in enumerate(labels["images"]):
        players = short_scene.frame_state(frame).players
        annotations = by_image[image["image_id"]]
        assert len(annotations) == len(players)
        for annotation, player in zip(annotations, players, strict=True):
            assert annotation["supercategory"] == "object"
            assert annotation["track_id"] == player.track_id
            assert category_name[annotation["category_id"]] == player.role
            attributes = annotation["attributes"]
            assert set(attributes) == {"role", "jersey", "team"}
            assert attributes["role"] == player.role
            # The team of a label is the side of the goal that the team defends.
            assert attributes["team"] == player.team_side
            if player.number_export is None:
                assert attributes["jersey"] is None
            else:
                assert attributes["jersey"] == str(player.number_export)
            box = annotation["bbox_image"]
            assert set(box) == {"x", "y", "x_center", "y_center", "w", "h"}
            assert (box["x"], box["y"], box["w"], box["h"]) == (
                player.box.x1,
                player.box.y1,
                player.box.width,
                player.box.height,
            )
            assert (box["x_center"], box["y_center"]) == player.box.center
            pitch = annotation["bbox_pitch"]
            assert set(pitch) == {
                "x_bottom_left",
                "y_bottom_left",
                "x_bottom_middle",
                "y_bottom_middle",
                "x_bottom_right",
                "y_bottom_right",
            }
            assert (pitch["x_bottom_middle"], pitch["y_bottom_middle"]) == (player.pitch_x, player.pitch_y)
            assert pitch["x_bottom_left"] < pitch["x_bottom_middle"] < pitch["x_bottom_right"]


def test_gsr_labels_load_with_the_label_reader(short_scene: SyntheticScene, tmp_path: Path) -> None:
    """The label reader of the package reads the synthetic labels and gives the truth back."""
    from pitchvlm.labels import gsr_frame_people, gsr_frame_states, load_gsr_labels

    path = tmp_path / synth.CLIP_ID / "Labels-GameState.json"
    path.parent.mkdir()
    path.write_text(json.dumps(short_scene.gsr_labels()), encoding="utf-8")
    labels = load_gsr_labels(path)

    info = short_scene.info
    assert (labels.clip_id, labels.fps, labels.n_frames) == (info.clip_id, info.fps, info.n_frames)
    assert (labels.width, labels.height) == (info.width, info.height)
    assert labels.jersey_is_str is True
    assert all(frame.scored for frame in labels.frames)
    assert [frame.image_id for frame in labels.frames] == [f"2999{frame + 1:06d}" for frame in range(info.n_frames)]

    for frame, state in enumerate(gsr_frame_states(labels)):
        truth = short_scene.frame_state(frame)
        assert [player.track_id for player in state.players] == [player.track_id for player in truth.players]
        for player, true_player in zip(state.players, truth.players, strict=True):
            assert (player.pitch_x, player.pitch_y) == (true_player.pitch_x, true_player.pitch_y)
            assert (player.role, player.team_side) == (true_player.role, true_player.team_side)
    people = gsr_frame_people(labels, 3).people
    true_people = short_scene.people(3).people
    assert [(person.role, person.team_key, person.number) for person in people] == [
        (person.role, person.team_key, person.number) for person in true_people
    ]


# --- Image -----------------------------------------------------------------------------


def test_image_shows_grass_marks_lines_and_kits(scene: SyntheticScene) -> None:
    image = scene.image(0)
    assert image.shape == (720, 1280, 3)
    assert image.dtype == np.uint8
    assert len(np.unique(image.reshape(-1, 3), axis=0)) > 10
    assert count_colour(image, synth.GRASS_BGR) > image.shape[0] * image.shape[1] // 2
    assert count_colour(image, synth.MARK_BGR) > 100
    # Only the white pitch lines and white shirt numbers are bright in all three channels.
    assert int((image.min(axis=2) > 200).sum()) > 300
    assert count_colour(image, synth.TEAM_A_KIT.fill_bgr) > 100

    # Goalkeeper A stands alone in frame 0. The legs part of the box has the plain kit colour.
    box = next(player.box for player in scene.frame_state(0).players if player.track_id == 0)
    row = round(box.y2 - 0.15 * box.height)
    column = round((box.x1 + box.x2) / 2.0)
    assert tuple(int(value) for value in image[row, column]) == synth.KEEPER_A_KIT.fill_bgr


def test_image_follows_the_frame_the_seed_and_the_size(scene: SyntheticScene) -> None:
    assert np.array_equal(scene.image(3), SyntheticScene(seed=7).image(3))
    assert not np.array_equal(scene.image(3), scene.image(4))
    assert not np.array_equal(scene.image(3), SyntheticScene(seed=8).image(3))
    small = SyntheticScene(n_frames=5, width=640, height=360, seed=2).image(4)
    assert small.shape == (360, 640, 3)
    assert small.dtype == np.uint8


# --- Together with the motion module -------------------------------------------------


def pitch_part_grid() -> np.ndarray:
    """Return image points on the part of a 1280 x 720 picture that shows the pitch, shape (n, 2)."""
    grid_x, grid_y = np.meshgrid(np.arange(100.0, 1200.0, 100.0), np.arange(250.0, 700.0, 50.0))
    return np.column_stack([grid_x.ravel(), grid_y.ravel()])


def test_camera_motion_can_be_measured_between_two_images(scene: SyntheticScene) -> None:
    """The grass marks and the lines let the motion module follow the camera."""
    from pitchvlm.motion import estimate_motion

    first, second = 74, 75  # the pan is fastest here: about 9 pixels per frame
    estimate = estimate_motion(scene.image(first), scene.image(second))
    assert estimate is not None
    # A point of the grass goes from pixels of the first frame to the pitch and to pixels of the second frame.
    true_motion = np.linalg.inv(scene.homography(second)) @ scene.homography(first)
    grid = pitch_part_grid()
    true_points = apply(true_motion, grid)
    assert np.linalg.norm(true_points - grid, axis=1).min() > 3.0
    assert np.median(np.linalg.norm(apply(estimate, grid) - true_points, axis=1)) < 1.0


def test_motion_field_of_a_scene_follows_the_true_camera() -> None:
    """The motion field of the pipeline (half-size images, people masked) agrees with the truth."""
    from pitchvlm.motion import compute_motion_field

    scene = SyntheticScene(n_frames=6, seed=3)
    people = {frame: [person.box for person in scene.people(frame).people] for frame in range(scene.info.n_frames)}
    field = compute_motion_field(SyntheticSource(scene), people=people)
    assert field.n_frames == 6
    measured = field.between(0, 5)
    assert measured is not None
    true_motion = np.linalg.inv(scene.homography(5)) @ scene.homography(0)
    grid = pitch_part_grid()
    true_points = apply(true_motion, grid)
    assert np.linalg.norm(true_points - grid, axis=1).min() > 3.0
    assert np.median(np.linalg.norm(apply(measured, grid) - true_points, axis=1)) < 1.5
