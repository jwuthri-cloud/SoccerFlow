"""Pitch model in SoccerNet-GSR coordinates, and drawing helpers.

Metres. Origin at the centre mark. x grows towards the right goal, y grows
towards the main camera. So the far touchline (top of the TV picture) is
y = -34 and the near touchline is y = +34. The pitch is 105 x 68 m.

Every landmark is a real mark on the grass: a line crossing, a line end, a
spot or a goal-post base. A model can point at these. Tangent points and
other invisible positions are not in the table.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
HALF_LENGTH = PITCH_LENGTH / 2.0
HALF_WIDTH = PITCH_WIDTH / 2.0

PENALTY_AREA_DEPTH = 16.5
PENALTY_AREA_HALF_WIDTH = 20.16
GOAL_AREA_DEPTH = 5.5
GOAL_AREA_HALF_WIDTH = 9.16
PENALTY_SPOT_DISTANCE = 11.0
CIRCLE_RADIUS = 9.15
GOAL_HALF_WIDTH = 3.66

_PENALTY_LINE_X = HALF_LENGTH - PENALTY_AREA_DEPTH
_GOAL_AREA_LINE_X = HALF_LENGTH - GOAL_AREA_DEPTH
_PENALTY_SPOT_X = HALF_LENGTH - PENALTY_SPOT_DISTANCE
#: Half the chord where the penalty arc crosses the penalty-area line (about 7.31 m).
ARC_HALF_CHORD = math.sqrt(CIRCLE_RADIUS**2 - (PENALTY_AREA_DEPTH - PENALTY_SPOT_DISTANCE) ** 2)


@dataclass(frozen=True)
class Landmark:
    id: int
    name: str
    x: float
    y: float


def _build_landmarks() -> tuple[Landmark, ...]:
    far, near = -HALF_WIDTH, HALF_WIDTH
    pa, ga, post, arc = PENALTY_AREA_HALF_WIDTH, GOAL_AREA_HALF_WIDTH, GOAL_HALF_WIDTH, ARC_HALF_CHORD
    left, right = -HALF_LENGTH, HALF_LENGTH
    rows: list[tuple[str, float, float]] = [
        # Left goal line, far to near.
        ("far-left corner", left, far),
        ("left penalty area, far corner on the goal line", left, -pa),
        ("left goal area, far corner on the goal line", left, -ga),
        ("left goal, far post base", left, -post),
        ("left goal, near post base", left, post),
        ("left goal area, near corner on the goal line", left, ga),
        ("left penalty area, near corner on the goal line", left, pa),
        ("near-left corner", left, near),
        # Left goal area and penalty spot.
        ("left goal area, far front corner", -_GOAL_AREA_LINE_X, -ga),
        ("left goal area, near front corner", -_GOAL_AREA_LINE_X, ga),
        ("left penalty spot", -_PENALTY_SPOT_X, 0.0),
        # Left penalty-area line, far to near.
        ("left penalty area, far front corner", -_PENALTY_LINE_X, -pa),
        ("left penalty arc, far crossing with the penalty-area line", -_PENALTY_LINE_X, -arc),
        ("left penalty arc, near crossing with the penalty-area line", -_PENALTY_LINE_X, arc),
        ("left penalty area, near front corner", -_PENALTY_LINE_X, pa),
        # Halfway line, far to near.
        ("halfway line, far end", 0.0, far),
        ("centre circle, far crossing with the halfway line", 0.0, -CIRCLE_RADIUS),
        ("centre spot", 0.0, 0.0),
        ("centre circle, near crossing with the halfway line", 0.0, CIRCLE_RADIUS),
        ("halfway line, near end", 0.0, near),
        # Right penalty-area line, far to near.
        ("right penalty area, far front corner", _PENALTY_LINE_X, -pa),
        ("right penalty arc, far crossing with the penalty-area line", _PENALTY_LINE_X, -arc),
        ("right penalty arc, near crossing with the penalty-area line", _PENALTY_LINE_X, arc),
        ("right penalty area, near front corner", _PENALTY_LINE_X, pa),
        # Right penalty spot and goal area.
        ("right penalty spot", _PENALTY_SPOT_X, 0.0),
        ("right goal area, far front corner", _GOAL_AREA_LINE_X, -ga),
        ("right goal area, near front corner", _GOAL_AREA_LINE_X, ga),
        # Right goal line, far to near.
        ("far-right corner", right, far),
        ("right penalty area, far corner on the goal line", right, -pa),
        ("right goal area, far corner on the goal line", right, -ga),
        ("right goal, far post base", right, -post),
        ("right goal, near post base", right, post),
        ("right goal area, near corner on the goal line", right, ga),
        ("right penalty area, near corner on the goal line", right, pa),
        ("near-right corner", right, near),
    ]
    return tuple(Landmark(i, name, x, y) for i, (name, x, y) in enumerate(rows, start=1))


#: All landmarks, ids 1..35, ordered left to right and far to near.
LANDMARKS: tuple[Landmark, ...] = _build_landmarks()
LANDMARK_BY_ID: dict[int, Landmark] = {lm.id: lm for lm in LANDMARKS}


def landmark_xy(ids: Sequence[int]) -> np.ndarray:
    """Pitch coordinates of the given landmark ids, shape (n, 2). Unknown ids raise KeyError."""
    return np.array([[LANDMARK_BY_ID[i].x, LANDMARK_BY_ID[i].y] for i in ids], dtype=np.float64).reshape(-1, 2)


def landmark_legend_text() -> str:
    """One line per landmark, for prompts: ``<id>: <name>``."""
    return "\n".join(f"{lm.id}: {lm.name}" for lm in LANDMARKS)


def _as_points(xy: np.ndarray) -> np.ndarray:
    """Return points as a ``float64`` array of shape (n, 2).

    One point of shape (2,) becomes shape (1, 2). An input without values
    becomes shape (0, 2). Any other shape raises ``ValueError``, so that an
    array of shape (n, 3) or (2, n) is never cut into wrong points.
    """
    points = np.asarray(xy, dtype=np.float64)
    if points.size == 0:
        return points.reshape(0, 2)
    if points.shape == (2,):
        return points.reshape(1, 2)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have the shape (n, 2), got {points.shape}")
    return points


def inside_pitch(xy: np.ndarray, margin_m: float = 0.0) -> np.ndarray:
    """Boolean mask of points inside the pitch, enlarged by ``margin_m`` on every side.

    ``xy`` has shape (n, 2), or shape (2,) for one point. The mask has shape (n,).
    """
    pts = _as_points(xy)
    return (np.abs(pts[:, 0]) <= HALF_LENGTH + margin_m) & (np.abs(pts[:, 1]) <= HALF_WIDTH + margin_m)


@dataclass(frozen=True, eq=False)
class Polyline:
    """A line of the pitch markings in metres.

    Two polylines are equal only when they are the same object: the points are
    an array, and arrays have no single truth value for a comparison.
    """

    points: np.ndarray
    closed: bool


def _arc(cx: float, direction: float, samples: int) -> np.ndarray:
    """Part of a penalty arc outside the penalty area. ``direction`` is +1 (left arc) or -1."""
    half_angle = math.acos((PENALTY_AREA_DEPTH - PENALTY_SPOT_DISTANCE) / CIRCLE_RADIUS)
    angles = np.linspace(-half_angle, half_angle, samples)
    return np.stack([cx + direction * CIRCLE_RADIUS * np.cos(angles), CIRCLE_RADIUS * np.sin(angles)], axis=1)


def pitch_polylines(arc_samples: int = 64) -> list[Polyline]:
    """All pitch markings as polylines in metres.

    The centre circle has ``arc_samples`` points and each penalty arc has half
    as many. Fewer than 8 samples do not give a curve, so they raise ``ValueError``.
    """
    if arc_samples < 8:
        raise ValueError(f"arc_samples must be at least 8, got {arc_samples}")
    hl, hw = HALF_LENGTH, HALF_WIDTH
    pa_x, ga_x = _PENALTY_LINE_X, _GOAL_AREA_LINE_X
    pa, ga = PENALTY_AREA_HALF_WIDTH, GOAL_AREA_HALF_WIDTH
    circle_angles = np.linspace(0.0, 2.0 * math.pi, arc_samples, endpoint=False)
    circle = np.stack([CIRCLE_RADIUS * np.cos(circle_angles), CIRCLE_RADIUS * np.sin(circle_angles)], axis=1)

    def open_box(goal_x: float, front_x: float, half: float) -> np.ndarray:
        return np.array([[goal_x, -half], [front_x, -half], [front_x, half], [goal_x, half]], dtype=np.float64)

    return [
        Polyline(np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]], dtype=np.float64), closed=True),
        Polyline(np.array([[0.0, -hw], [0.0, hw]], dtype=np.float64), closed=False),
        Polyline(circle, closed=True),
        Polyline(open_box(-hl, -pa_x, pa), closed=False),
        Polyline(open_box(hl, pa_x, pa), closed=False),
        Polyline(open_box(-hl, -ga_x, ga), closed=False),
        Polyline(open_box(hl, ga_x, ga), closed=False),
        Polyline(_arc(-_PENALTY_SPOT_X, 1.0, arc_samples // 2), closed=False),
        Polyline(_arc(_PENALTY_SPOT_X, -1.0, arc_samples // 2), closed=False),
    ]


def pitch_spots() -> np.ndarray:
    """Centre spot and the two penalty spots, shape (3, 2)."""
    return np.array([[0.0, 0.0], [-_PENALTY_SPOT_X, 0.0], [_PENALTY_SPOT_X, 0.0]], dtype=np.float64)


@dataclass(frozen=True)
class PitchCanvas:
    """Top-down drawing surface. The far touchline is at the top, the left goal at the left."""

    scale: float = 10.0  # pixels per metre
    margin_m: float = 4.0

    @property
    def width(self) -> int:
        return int(round((PITCH_LENGTH + 2.0 * self.margin_m) * self.scale))

    @property
    def height(self) -> int:
        return int(round((PITCH_WIDTH + 2.0 * self.margin_m) * self.scale))

    def to_px(self, xy: np.ndarray) -> np.ndarray:
        """Pitch metres to canvas pixels, shape (n, 2), float.

        ``xy`` has shape (n, 2), or shape (2,) for one point.
        """
        pts = _as_points(xy)
        u = (pts[:, 0] + HALF_LENGTH + self.margin_m) * self.scale
        v = (pts[:, 1] + HALF_WIDTH + self.margin_m) * self.scale
        return np.stack([u, v], axis=1)

    def draw(
        self,
        background_bgr: tuple[int, int, int] = (40, 110, 50),
        line_bgr: tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """Empty pitch with its markings, as a BGR image."""
        image = np.full((self.height, self.width, 3), background_bgr, dtype=np.uint8)
        for line in pitch_polylines():
            pixels = np.round(self.to_px(line.points)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(image, [pixels], isClosed=line.closed, color=line_bgr, thickness=thickness, lineType=cv2.LINE_AA)
        for u, v in np.round(self.to_px(pitch_spots())).astype(int):
            cv2.circle(image, (int(u), int(v)), max(2, thickness + 1), line_bgr, thickness=-1, lineType=cv2.LINE_AA)
        return image


def draw_landmark_diagram(canvas: PitchCanvas | None = None) -> np.ndarray:
    """Numbered pitch diagram for the landmark prompt, as seen from the main camera side.

    Each landmark is a red dot with its id next to it. Ids left of the centre
    are written to the right of the dot and the other way round, so that
    numbers stay inside the picture.
    """
    canvas = canvas or PitchCanvas(scale=14.0, margin_m=6.0)
    image = canvas.draw(thickness=2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for lm in LANDMARKS:
        u, v = (int(round(c)) for c in canvas.to_px(np.array([[lm.x, lm.y]]))[0])
        cv2.circle(image, (u, v), 6, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
        label = str(lm.id)
        (text_w, text_h), _ = cv2.getTextSize(label, font, 0.6, 2)
        text_u = u + 9 if lm.x <= 0 else u - 9 - text_w
        text_v = v - 8 if lm.y > 0 else v + 8 + text_h
        cv2.rectangle(image, (text_u - 2, text_v - text_h - 2), (text_u + text_w + 2, text_v + 3), (255, 255, 255), thickness=-1)
        cv2.putText(image, label, (text_u, text_v), font, 0.6, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    return image
