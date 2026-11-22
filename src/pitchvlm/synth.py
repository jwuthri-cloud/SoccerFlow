"""A small synthetic match with exact ground truth, for tests and the offline demo.

Two teams of eleven and one referee move on a 105 x 68 m pitch. One camera stands at a
fixed place on the near side. It pans and zooms slowly. Everything that the pipeline must
find is known exactly: the boxes, the pitch landmarks, the image-to-pitch mapping, the
pitch positions and the shirt numbers.

Conventions are those of ``schemas.py``. Image: pixels, origin at the top left, x to the
right, y down. Pitch: metres, x towards the right goal, y towards the camera, so a
far-side person has a negative y. Team A defends the left goal.

The camera path depends on time only. The seed changes the paths of the people and the
grass texture, not the camera. A clip starts with a view of the left side. The pan angle
is zero after 3 seconds (frame 75 at 25 frames per second), and the camera looks at the
right side after about 5.2 seconds.

The scene is made for 16:9 images that are at least 640 pixels wide. In such an image
every frame shows at least 7 pitch landmarks, enough to fix the pitch mapping. In a
smaller image the far players are below the smallest box size.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from pitchvlm.pitch import HALF_LENGTH, HALF_WIDTH, LANDMARKS, landmark_xy, pitch_polylines, pitch_spots
from pitchvlm.schemas import (
    Box,
    ClipInfo,
    FrameLandmarks,
    FramePeople,
    FrameState,
    GoalkeeperKit,
    KitDescription,
    KitLegend,
    LandmarkObs,
    PersonDet,
    PitchView,
    PlayerState,
    Role,
    TeamKey,
    TeamSide,
)

# --- Clip and label file -------------------------------------------------------

#: Clip id of every synthetic scene: the SoccerNet naming with sequence number 999.
CLIP_ID = "SNGS-999"
#: Start of every image id: split digit 2, then sequence number 999. Six frame digits follow.
IMAGE_ID_PREFIX = "2999"
#: Category table of a SoccerNet-GSR label file: (id, supercategory, name).
GSR_CATEGORIES: tuple[tuple[int, str, str], ...] = (
    (1, "object", "player"),
    (2, "object", "goalkeeper"),
    (3, "object", "referee"),
    (4, "object", "ball"),
    (5, "pitch", "pitch"),
    (6, "camera", "camera"),
    (7, "object", "other"),
)
_GSR_CATEGORY_ID = {name: category_id for category_id, _, name in GSR_CATEGORIES}

# --- People ----------------------------------------------------------------------

PERSON_HEIGHT_M = 1.8
#: Width of a person's box as a fraction of its height.
BOX_ASPECT = 0.4
#: A box that is smaller than this in either direction, after clipping, is not visible.
MIN_BOX_PX = 4.0
#: ``true_number`` needs at least this overlap (IoU) between the given box and a true box.
MIN_NUMBER_IOU = 0.3
#: Start positions of team A, which defends the left goal: the goalkeeper, then ten players.
#: Team B starts at the mirrored positions (-x, -y).
FORMATION_A: tuple[tuple[float, float], ...] = (
    (-48.5, 0.0),
    (-26.0, -23.0),
    (-29.0, -8.0),
    (-29.0, 8.0),
    (-26.0, 23.0),
    (-9.0, -27.0),
    (-12.0, -10.0),
    (-12.0, 10.0),
    (-9.0, 27.0),
    (9.0, -14.0),
    (7.0, 13.0),
)
REFEREE_START: tuple[float, float] = (2.0, -3.0)
#: Largest random shift of a start position, metres on each axis. Goalkeepers do not shift.
START_JITTER_M = 1.0
#: Smallest and largest drift around the start position, metres, as (x, y).
PLAYER_DRIFT_MIN_M = (1.0, 1.0)
PLAYER_DRIFT_MAX_M = (3.0, 3.0)
KEEPER_DRIFT_MIN_M = (0.3, 1.0)
KEEPER_DRIFT_MAX_M = (1.0, 2.5)
#: Drift frequency range in hertz: one swing takes about 6 to 14 seconds.
DRIFT_MIN_HZ = 0.07
DRIFT_MAX_HZ = 0.17

# --- Camera ----------------------------------------------------------------------

#: Camera position: level with the halfway line, 30 m behind the near touchline, 18 m high.
CAMERA_X_M = 0.0
CAMERA_Y_M = HALF_WIDTH + 30.0
CAMERA_HEIGHT_M = 18.0
#: With pan 0 the camera looks at the pitch point (0, AIM_Y_M). This fixes the tilt angle.
AIM_Y_M = 8.0
TILT_RAD = math.atan2(CAMERA_HEIGHT_M, CAMERA_Y_M - AIM_Y_M)
#: The pan angle is a slow sine wave in time. A negative angle looks towards the left goal.
#: With these values every point of the pitch stays in front of the camera.
PAN_AMPLITUDE_RAD = math.radians(24.0)
PAN_PERIOD_S = 16.0
#: Time shift of the pan wave: a clip starts on the left side with the camera already moving.
PAN_LEAD_S = 1.0
#: Focal length in image widths, and the size and period of its slow change (the zoom).
FOCAL_PER_WIDTH = 1.0
ZOOM_SWING = 0.04
ZOOM_PERIOD_S = 10.0
#: The view is "left" or "right" when the image centre shows a pitch x beyond this, metres.
SIDE_VIEW_X_M = 17.5

# --- Drawing ---------------------------------------------------------------------

GRASS_BGR = (70, 160, 80)
#: The marks are much darker than the grass, so that a corner detector finds them.
MARK_BGR = (15, 70, 20)
LINE_BGR = (255, 255, 255)
LINE_THICKNESS_PX = 2
SPOT_RADIUS_PX = 3
#: Number of dark marks on the grass, and how far they go beyond the pitch lines.
TEXTURE_MARKS = 2000
TEXTURE_MARGIN_M = 3.0
#: A mark is a triangle. Its corners are this far from its centre, metres.
MARK_MIN_RADIUS_M = 0.2
MARK_MAX_RADIUS_M = 0.45
#: Marks are longer along y, because the camera sees the grass at a flat angle.
MARK_DEPTH_STRETCH = 2.0
#: Marks are drawn when they are inside the image plus this fraction of its width.
MARK_PAD_PER_WIDTH = 0.05
#: Fractional bits of the pixel coordinates given to OpenCV (sub-pixel drawing).
DRAW_SHIFT = 4
NUMBER_FONT = cv2.FONT_HERSHEY_SIMPLEX
#: Largest size of the shirt number: a fraction of the box width and of the box height.
NUMBER_MAX_WIDTH = 0.8
NUMBER_MAX_HEIGHT = 0.35
#: Height of the middle of the shirt number above the feet, as a fraction of the box height.
NUMBER_CENTRE_HEIGHT = 0.62


@dataclass(frozen=True)
class Kit:
    """One kit: the colour word for the legend and the BGR colours for drawing."""

    colour: str
    fill_bgr: tuple[int, int, int]
    number_bgr: tuple[int, int, int]


TEAM_A_KIT = Kit("red", (40, 40, 220), (255, 255, 255))
TEAM_B_KIT = Kit("blue", (220, 90, 40), (255, 255, 255))
KEEPER_A_KIT = Kit("yellow", (0, 230, 255), (0, 0, 0))
KEEPER_B_KIT = Kit("green", (80, 255, 120), (0, 0, 0))
REFEREE_KIT = Kit("black", (20, 20, 20), (255, 255, 255))


@dataclass(frozen=True)
class _Person:
    """One person of the scene. ``index`` never changes and is the true track id."""

    index: int
    role: Role
    team_key: TeamKey
    team_side: TeamSide | None
    number: int | None
    kit: Kit


@dataclass(frozen=True)
class _Sighting:
    """One person as the camera sees them in one frame."""

    person: _Person
    pitch_x: float
    pitch_y: float
    #: Image point of the feet and box height in pixels, before the box is cut at the image border.
    foot_x: float
    foot_y: float
    height_px: float
    #: The part of the box that is inside the image.
    box: Box


@dataclass(frozen=True, eq=False)
class _Paths:
    """Paths of all people: a start position plus a slow sine drift on each axis.

    Every array has shape (n, 2): one row per person, columns x and y.
    """

    start: np.ndarray
    amplitude: np.ndarray
    frequency: np.ndarray
    phase: np.ndarray

    def at(self, seconds: float) -> np.ndarray:
        """Return the pitch positions of all people at one time, shape (n, 2), metres."""
        return self.start + self.amplitude * np.sin(2.0 * math.pi * self.frequency * seconds + self.phase)


def _build_people() -> tuple[_Person, ...]:
    """Return the 23 people in their fixed order: team A, team B, then the referee.

    Inside a team the goalkeeper comes first with shirt number 1. The players have 2 to 11.
    """
    teams: tuple[tuple[TeamKey, TeamSide, Kit, Kit], ...] = (
        ("A", "left", KEEPER_A_KIT, TEAM_A_KIT),
        ("B", "right", KEEPER_B_KIT, TEAM_B_KIT),
    )
    people: list[_Person] = []
    for team_key, team_side, keeper_kit, player_kit in teams:
        for number in range(1, len(FORMATION_A) + 1):
            is_keeper = number == 1
            people.append(
                _Person(
                    index=len(people),
                    role="goalkeeper" if is_keeper else "player",
                    team_key=team_key,
                    team_side=team_side,
                    number=number,
                    kit=keeper_kit if is_keeper else player_kit,
                )
            )
    people.append(
        _Person(index=len(people), role="referee", team_key="none", team_side=None, number=None, kit=REFEREE_KIT)
    )
    return tuple(people)


def _random_paths(people: Sequence[_Person], rng: np.random.Generator) -> _Paths:
    """Draw one smooth path per person, in the order of ``_build_people``.

    The largest start shift plus the largest drift keeps every person inside the pitch
    and keeps each goalkeeper within a few metres of their goal.
    """
    team_a = np.array(FORMATION_A, dtype=np.float64)
    spots = np.vstack([team_a, -team_a, np.array([REFEREE_START], dtype=np.float64)])
    is_keeper = np.array([person.role == "goalkeeper" for person in people])[:, None]
    jitter = rng.uniform(-START_JITTER_M, START_JITTER_M, size=spots.shape)
    return _Paths(
        start=spots + np.where(is_keeper, 0.0, jitter),
        amplitude=rng.uniform(
            np.where(is_keeper, KEEPER_DRIFT_MIN_M, PLAYER_DRIFT_MIN_M),
            np.where(is_keeper, KEEPER_DRIFT_MAX_M, PLAYER_DRIFT_MAX_M),
        ),
        frequency=rng.uniform(DRIFT_MIN_HZ, DRIFT_MAX_HZ, size=spots.shape),
        phase=rng.uniform(0.0, 2.0 * math.pi, size=spots.shape),
    )


def _texture_marks(rng: np.random.Generator) -> np.ndarray:
    """Return the grass marks: random small triangles in pitch metres, shape (TEXTURE_MARKS, 3, 2)."""
    half_x = HALF_LENGTH + TEXTURE_MARGIN_M
    half_y = HALF_WIDTH + TEXTURE_MARGIN_M
    centres = rng.uniform([-half_x, -half_y], [half_x, half_y], size=(TEXTURE_MARKS, 2))
    radii = rng.uniform(MARK_MIN_RADIUS_M, MARK_MAX_RADIUS_M, size=(TEXTURE_MARKS, 3))
    third_turn = 2.0 * math.pi / 3.0
    angles = rng.uniform(0.0, 2.0 * math.pi, size=(TEXTURE_MARKS, 1)) + third_turn * np.arange(3)
    corners = np.stack([radii * np.cos(angles), MARK_DEPTH_STRETCH * radii * np.sin(angles)], axis=-1)
    return centres[:, None, :] + corners


def _camera_pose(seconds: float, width: int) -> tuple[float, float]:
    """Return the pan angle in radians and the focal length in pixels at one time."""
    pan = -PAN_AMPLITUDE_RAD * math.cos(2.0 * math.pi * (seconds + PAN_LEAD_S) / PAN_PERIOD_S)
    zoom = 1.0 + ZOOM_SWING * math.sin(2.0 * math.pi * seconds / ZOOM_PERIOD_S)
    return pan, FOCAL_PER_WIDTH * width * zoom


def _projection_matrix(pan: float, focal_px: float, width: int, height: int) -> np.ndarray:
    """Return the 3x4 matrix that projects (pitch x, pitch y, height above the grass) to pixels.

    The camera is a pinhole with no roll and with the principal point at the image centre.
    The rows of ``rotation`` are the camera axes in pitch coordinates: image right, image
    down and the viewing direction. With pan 0 the camera looks along -y, so the right goal
    (+x) is on the right of the picture and the far side (-y) is at the top.
    """
    sin_pan, cos_pan = math.sin(pan), math.cos(pan)
    sin_tilt, cos_tilt = math.sin(TILT_RAD), math.cos(TILT_RAD)
    rotation = np.array(
        [
            [cos_pan, sin_pan, 0.0],
            [-sin_pan * sin_tilt, cos_pan * sin_tilt, -cos_tilt],
            [sin_pan * cos_tilt, -cos_pan * cos_tilt, -sin_tilt],
        ]
    )
    intrinsics = np.array([[focal_px, 0.0, width / 2.0], [0.0, focal_px, height / 2.0], [0.0, 0.0, 1.0]])
    position = np.array([CAMERA_X_M, CAMERA_Y_M, CAMERA_HEIGHT_M])
    return intrinsics @ np.column_stack([rotation, -rotation @ position])


def _project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a projective map to points and return the result, shape (n, 2).

    ``matrix`` is 3x3 for plane points of shape (n, 2), or 3x4 for space points of shape
    (n, 3). The third homogeneous coordinate of every result must be positive. It is not
    for a point behind the camera, or for a pixel above the horizon when the map goes from
    the image to the pitch: such a point raises ``ValueError``.
    """
    homogeneous = np.column_stack([points, np.ones(len(points))])
    mapped = homogeneous @ matrix.T
    scale = mapped[:, 2]
    if np.any(scale <= 0.0):
        raise ValueError("a point is behind the camera or above the horizon and cannot be projected")
    return mapped[:, :2] / scale[:, None]


def _fixed_point(pixels: np.ndarray) -> np.ndarray:
    """Convert pixel coordinates to the integers that OpenCV draws with sub-pixel accuracy."""
    return np.round(pixels * (1 << DRAW_SHIFT)).astype(np.int32)


def _draw_person(image: np.ndarray, sighting: _Sighting) -> None:
    """Draw one person: a filled box in the kit colour, with the shirt number on the chest."""
    box, kit, number = sighting.box, sighting.person.kit, sighting.person.number
    cv2.rectangle(image, (round(box.x1), round(box.y1)), (round(box.x2), round(box.y2)), kit.fill_bgr, thickness=-1)
    if number is None:
        return
    label = str(number)
    (label_width, label_height), _ = cv2.getTextSize(label, NUMBER_FONT, 1.0, 1)
    scale = min(
        NUMBER_MAX_WIDTH * BOX_ASPECT * sighting.height_px / label_width,
        NUMBER_MAX_HEIGHT * sighting.height_px / label_height,
    )
    left = sighting.foot_x - scale * label_width / 2.0
    baseline = sighting.foot_y - NUMBER_CENTRE_HEIGHT * sighting.height_px + scale * label_height / 2.0
    cv2.putText(image, label, (round(left), round(baseline)), NUMBER_FONT, scale, kit.number_bgr, 1, cv2.LINE_AA)


def _athlete_annotation(
    annotation_id: int, image_id: str, sighting: _Sighting, image_to_pitch: np.ndarray
) -> dict[str, Any]:
    """Return one athlete annotation of a SoccerNet-GSR label file.

    ``bbox_image`` is the visible box. In ``bbox_pitch`` the middle point is the true pitch
    position. The left and right points are the bottom corners of the uncut box, mapped to
    the pitch.
    """
    person, box = sighting.person, sighting.box
    half_width = BOX_ASPECT * sighting.height_px / 2.0
    bottom_corners = np.array(
        [[sighting.foot_x - half_width, sighting.foot_y], [sighting.foot_x + half_width, sighting.foot_y]]
    )
    (left_x, left_y), (right_x, right_y) = _project(image_to_pitch, bottom_corners).tolist()
    centre_x, centre_y = box.center
    return {
        "id": str(annotation_id),
        "image_id": image_id,
        "track_id": person.index,
        "supercategory": "object",
        "category_id": _GSR_CATEGORY_ID[person.role],
        "attributes": {
            "role": person.role,
            "jersey": None if person.number is None else str(person.number),
            "team": person.team_side,
        },
        "bbox_image": {
            "x": box.x1,
            "y": box.y1,
            "x_center": centre_x,
            "y_center": centre_y,
            "w": box.width,
            "h": box.height,
        },
        "bbox_pitch": {
            "x_bottom_left": left_x,
            "y_bottom_left": left_y,
            "x_bottom_right": right_x,
            "y_bottom_right": right_y,
            "x_bottom_middle": sighting.pitch_x,
            "y_bottom_middle": sighting.pitch_y,
        },
    }


class SyntheticScene:
    """A synthetic match clip with exact ground truth for every frame.

    The people are 23: team A (indices 0 to 10), team B (11 to 21) and the referee (22).
    Inside a team the goalkeeper comes first. The index of a person is their true track id.
    Every method is a pure function of the frame index, so one scene can be read from
    several threads.
    """

    info: ClipInfo

    def __init__(
        self, *, n_frames: int = 100, fps: float = 25.0, width: int = 1280, height: int = 720, seed: int = 0
    ) -> None:
        if seed < 0:
            raise ValueError("seed must not be negative")
        self.info = ClipInfo(clip_id=CLIP_ID, n_frames=n_frames, fps=fps, width=width, height=height)
        self._people = _build_people()
        self._paths = _random_paths(self._people, np.random.default_rng([seed, 0]))
        self._marks = _texture_marks(np.random.default_rng([seed, 1]))
        self._markings = pitch_polylines()
        self._spots = pitch_spots()

    def image(self, frame: int) -> np.ndarray:
        """Return one frame as a BGR uint8 image of shape (height, width, 3).

        The picture shows green grass with a fixed texture of darker marks, the white pitch
        markings, and one filled box per visible person in the kit colour with the shirt
        number on it. People nearer to the camera are drawn over people further away.
        """
        ground_to_image = self._ground_to_image(frame)
        image = np.full((self.info.height, self.info.width, 3), GRASS_BGR, dtype=np.uint8)
        self._draw_texture(image, ground_to_image)
        self._draw_markings(image, ground_to_image)
        for sighting in sorted(self._sightings(frame), key=lambda seen: seen.foot_y):
            _draw_person(image, sighting)
        return image

    def homography(self, frame: int) -> np.ndarray:
        """Return the true 3x3 mapping from image pixels to pitch metres.

        ``pitch ~ H @ (x_px, y_px, 1)``. The matrix is the exact inverse of the projection of
        the pitch plane. It is not scaled to ``H[2, 2] = 1``.
        """
        return np.linalg.inv(self._ground_to_image(frame))

    def people(self, frame: int) -> FramePeople:
        """Return the true people whose box is at least partly inside the image.

        A person is a vertical segment of 1.8 m at their pitch position. The box has the
        feet at its bottom centre, the height of the projected segment and 0.4 times that
        height as width. Boxes are cut at the image border. A box that is smaller than 4
        pixels in either direction after the cut is left out.
        """
        detections = tuple(
            PersonDet(box=seen.box, role=seen.person.role, team_key=seen.person.team_key, number=seen.person.number)
            for seen in self._sightings(frame)
        )
        return FramePeople(frame=frame, people=detections)

    def landmarks(self, frame: int) -> FrameLandmarks:
        """Return the true pitch landmarks whose image point is inside the image.

        ``view`` names the part of the pitch that the image centre shows: "left" when its
        pitch x is below -17.5 m, "right" when it is above 17.5 m, else "centre".
        """
        ground_to_image = self._ground_to_image(frame)
        ids = [landmark.id for landmark in LANDMARKS]
        pixels = _project(ground_to_image, landmark_xy(ids)).tolist()
        seen = tuple(
            LandmarkObs(landmark_id=landmark_id, x=x, y=y)
            for landmark_id, (x, y) in zip(ids, pixels, strict=True)
            if 0.0 <= x <= self.info.width and 0.0 <= y <= self.info.height
        )
        return FrameLandmarks(frame=frame, view=self._view(ground_to_image), landmarks=seen)

    def frame_state(self, frame: int) -> FrameState:
        """Return the true dense state: one entry per person with a visible box.

        The pitch position is the true one, also when the box is cut at the image border.
        Team A has side "left", team B has side "right" and the referee has no side. The
        track id is the fixed index of the person.
        """
        players = tuple(
            PlayerState(
                track_id=seen.person.index,
                box=seen.box,
                pitch_x=seen.pitch_x,
                pitch_y=seen.pitch_y,
                role=seen.person.role,
                team_key=seen.person.team_key,
                team_side=seen.person.team_side,
                number_export=seen.person.number,
                number_display=seen.person.number,
                observed=True,
            )
            for seen in self._sightings(frame)
        )
        return FrameState(frame=frame, players=players)

    def kit_legend(self) -> KitLegend:
        """Return the kit colours of the scene, as the kit legend step must report them."""
        return KitLegend(
            team_a=KitDescription(shirt=TEAM_A_KIT.colour, shorts=TEAM_A_KIT.colour, socks=TEAM_A_KIT.colour),
            team_b=KitDescription(shirt=TEAM_B_KIT.colour, shorts=TEAM_B_KIT.colour, socks=TEAM_B_KIT.colour),
            goalkeepers=(
                GoalkeeperKit(kit=KEEPER_A_KIT.colour, team_key="A"),
                GoalkeeperKit(kit=KEEPER_B_KIT.colour, team_key="B"),
            ),
            referee_kit=REFEREE_KIT.colour,
        )

    def true_number(self, frame: int, box: Box) -> int | None:
        """Return the shirt number of the true person that ``box`` shows in ``frame``.

        The person is the one whose visible box has the best overlap (IoU) with ``box``.
        The result is None when that overlap is below 0.3, and when the person is the
        referee, who has no number.
        """
        sightings = self._sightings(frame)
        if not sightings:
            return None
        best = max(sightings, key=lambda seen: box.iou(seen.box))
        if box.iou(best.box) < MIN_NUMBER_IOU:
            return None
        return best.person.number

    def gsr_labels(self) -> dict[str, Any]:
        """Return the content of a ``Labels-GameState.json`` file for this clip.

        The file has one image entry per frame, with all "labelled" flags true, and one
        athlete annotation per visible person. It has no pitch-line and no camera
        annotations. Image ids are ten digits: "2999" and the 1-based frame number. Track
        ids are the fixed indices of the people. Shirt numbers are strings of digits.
        """
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        for frame in range(self.info.n_frames):
            image_id = f"{IMAGE_ID_PREFIX}{frame + 1:06d}"
            images.append(
                {
                    "is_labeled": True,
                    "image_id": image_id,
                    "file_name": f"{frame + 1:06d}.jpg",
                    "height": self.info.height,
                    "width": self.info.width,
                    "has_labeled_person": True,
                    "has_labeled_pitch": True,
                    "has_labeled_camera": True,
                }
            )
            image_to_pitch = self.homography(frame)
            for sighting in self._sightings(frame):
                annotations.append(_athlete_annotation(len(annotations) + 1, image_id, sighting, image_to_pitch))
        return {
            "info": {
                "name": self.info.clip_id,
                "frame_rate": self.info.fps,
                "seq_length": self.info.n_frames,
                "im_dir": "img1",
                "im_ext": ".jpg",
            },
            "images": images,
            "annotations": annotations,
            "categories": [
                {"supercategory": supercategory, "id": category_id, "name": name}
                for category_id, supercategory, name in GSR_CATEGORIES
            ],
        }

    def _seconds(self, frame: int) -> float:
        """Return the scene time that one frame shows. A frame outside the clip raises ``IndexError``."""
        if not 0 <= frame < self.info.n_frames:
            raise IndexError(f"frame {frame} is outside the clip (0 to {self.info.n_frames - 1})")
        return frame / self.info.fps

    def _projection(self, frame: int) -> np.ndarray:
        """Return the 3x4 camera matrix of one frame. A frame outside the clip raises ``IndexError``."""
        pan, focal_px = _camera_pose(self._seconds(frame), self.info.width)
        return _projection_matrix(pan, focal_px, self.info.width, self.info.height)

    def _ground_to_image(self, frame: int) -> np.ndarray:
        """Return the 3x3 map from pitch metres to pixels: the camera matrix for height 0."""
        return self._projection(frame)[:, [0, 1, 3]]

    def _view(self, ground_to_image: np.ndarray) -> PitchView:
        """Name the part of the pitch that the image centre shows."""
        centre_px = np.array([[self.info.width / 2.0, self.info.height / 2.0]])
        centre_x = float(_project(np.linalg.inv(ground_to_image), centre_px)[0, 0])
        if centre_x < -SIDE_VIEW_X_M:
            return "left"
        if centre_x > SIDE_VIEW_X_M:
            return "right"
        return "centre"

    def _sightings(self, frame: int) -> list[_Sighting]:
        """Return every person whose box is visible in one frame, in the fixed order of the people."""
        projection = self._projection(frame)
        positions = self._paths.at(self._seconds(frame))
        on_grass = np.zeros(len(positions))
        feet = _project(projection, np.column_stack([positions, on_grass]))
        heads = _project(projection, np.column_stack([positions, on_grass + PERSON_HEIGHT_M]))
        width, height = float(self.info.width), float(self.info.height)
        sightings: list[_Sighting] = []
        for person, (pitch_x, pitch_y), (foot_x, foot_y), (_, head_y) in zip(
            self._people, positions.tolist(), feet.tolist(), heads.tolist(), strict=True
        ):
            height_px = foot_y - head_y
            half_width = BOX_ASPECT * height_px / 2.0
            x1, y1 = max(foot_x - half_width, 0.0), max(foot_y - height_px, 0.0)
            x2, y2 = min(foot_x + half_width, width), min(foot_y, height)
            if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
                continue
            sightings.append(
                _Sighting(
                    person=person,
                    pitch_x=pitch_x,
                    pitch_y=pitch_y,
                    foot_x=foot_x,
                    foot_y=foot_y,
                    height_px=height_px,
                    box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
        return sightings

    def _draw_texture(self, image: np.ndarray, ground_to_image: np.ndarray) -> None:
        """Draw the grass marks. They are fixed on the pitch, so they move with the camera."""
        pixels = _project(ground_to_image, self._marks.reshape(-1, 2)).reshape(-1, 3, 2)
        pad = MARK_PAD_PER_WIDTH * self.info.width
        upper = np.array([self.info.width, self.info.height]) + pad
        near_image = ((pixels > -pad) & (pixels < upper)).all(axis=(1, 2))
        for triangle in _fixed_point(pixels[near_image]):
            cv2.fillConvexPoly(image, triangle, MARK_BGR, lineType=cv2.LINE_AA, shift=DRAW_SHIFT)

    def _draw_markings(self, image: np.ndarray, ground_to_image: np.ndarray) -> None:
        """Draw the white pitch lines and spots through the camera."""
        for line in self._markings:
            corners = _fixed_point(_project(ground_to_image, line.points))
            cv2.polylines(
                image,
                [corners],
                isClosed=line.closed,
                color=LINE_BGR,
                thickness=LINE_THICKNESS_PX,
                lineType=cv2.LINE_AA,
                shift=DRAW_SHIFT,
            )
        for centre_x, centre_y in _fixed_point(_project(ground_to_image, self._spots)).tolist():
            cv2.circle(
                image,
                (centre_x, centre_y),
                SPOT_RADIUS_PX << DRAW_SHIFT,
                LINE_BGR,
                thickness=-1,
                lineType=cv2.LINE_AA,
                shift=DRAW_SHIFT,
            )


class RetimedScene(SyntheticScene):
    """A synthetic clip whose frames show chosen moments of the synthetic match.

    Frame ``k`` of the clip shows the match at frame ``scene_frames[k]`` of a
    ``SyntheticScene`` with the same frame rate, size and seed. So a jump in
    ``scene_frames`` is a hard cut: the camera and the people are somewhere
    else in the next frame. A step larger than 1 makes a faster camera: with
    a step of 10 the camera pans and zooms 10 times as fast, without a cut.
    The ground truth of every frame is that of the moment it shows. The
    people keep their track ids across a cut.
    """

    def __init__(
        self,
        scene_frames: Sequence[int],
        *,
        fps: float = 25.0,
        width: int = 1280,
        height: int = 720,
        seed: int = 0,
    ) -> None:
        if not scene_frames:
            raise ValueError("scene_frames must name at least one frame")
        if min(scene_frames) < 0:
            raise ValueError("scene_frames must not be negative")
        super().__init__(n_frames=len(scene_frames), fps=fps, width=width, height=height, seed=seed)
        self._scene_frames = tuple(int(frame) for frame in scene_frames)

    def _seconds(self, frame: int) -> float:
        """Return the scene time that one frame of the clip shows."""
        if not 0 <= frame < self.info.n_frames:
            raise IndexError(f"frame {frame} is outside the clip (0 to {self.info.n_frames - 1})")
        return self._scene_frames[frame] / self.info.fps


class SyntheticSource:
    """A frame source that draws the frames of a synthetic scene on request."""

    info: ClipInfo

    def __init__(self, scene: SyntheticScene) -> None:
        self._scene = scene
        self.info = scene.info

    def read(self, frame: int) -> np.ndarray:
        """Return one frame as a BGR uint8 image of shape (height, width, 3)."""
        return self._scene.image(frame)


class ScoreGraphicSource:
    """The frames of a synthetic scene with a static score graphic in the top left corner, as a broadcast shows.

    The graphic does not move with the camera, and it stays across a cut. Its
    corners can give a camera motion across a cut, so it tests that the cut
    test does not trust such a motion.
    """

    info: ClipInfo

    def __init__(self, scene: SyntheticScene) -> None:
        self._scene = scene
        self.info = scene.info

    def read(self, frame: int) -> np.ndarray:
        """Return one frame with the graphic, as a BGR uint8 image of shape (height, width, 3)."""
        image = self._scene.image(frame)
        cv2.rectangle(image, (20, 20), (520, 120), (30, 30, 30), thickness=-1)
        cv2.putText(image, "HOME 2 - 1 AWAY  67:12", (30, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(image, "LIVE  1ST HALF", (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
        return image
