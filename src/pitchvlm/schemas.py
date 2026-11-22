"""Shared records. Modules exchange data through these types only.

Coordinate conventions
----------------------
Image:  pixels, origin top-left, x to the right, y down.
Pitch:  metres, origin at the centre mark, x towards the right goal, y towards
        the main camera (far touchline y = -34, near touchline y = +34).
        The pitch is 105 x 68 m. This is the SoccerNet-GSR convention.
Frames: 0-based index into the clip. ``t = frame / fps`` seconds.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

Role = Literal["player", "goalkeeper", "referee", "other"]
TeamKey = Literal["A", "B", "none"]
TeamSide = Literal["left", "right"]
PitchView = Literal["left", "centre", "right"]
HomographySource = Literal["landmarks", "propagated", "none"]
Confidence = Literal["high", "medium", "low"]
#: A shirt number. Every record that holds one accepts the whole numbers 0 to 99 only.
ShirtNumber = Annotated[int, Field(ge=0, le=99)]

ROLES: tuple[Role, ...] = ("player", "goalkeeper", "referee", "other")
#: Roles that belong to a team. Referees and "other" people have no team.
TEAM_ROLES: tuple[Role, ...] = ("player", "goalkeeper")


class Record(BaseModel):
    """Base class: immutable, and unknown fields are an error."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Return a copy of the record. ``update`` gives new values for some fields.

        Plain pydantic does not check the values of ``update``. This method
        does: the copy is made from the old values and the new values, and it
        is validated like a new record. So a misspelled field name, a value of
        the wrong type and a broken record rule raise ``ValidationError``.
        """
        copied = super().model_copy(deep=deep)
        if not update:
            return copied
        values = {name: getattr(copied, name) for name in type(self).model_fields}
        return type(self).model_validate({**values, **update})


class Box(Record):
    """Axis-aligned box in image pixels. The four values are finite numbers."""

    x1: FiniteFloat
    y1: FiniteFloat
    x2: FiniteFloat
    y2: FiniteFloat

    @model_validator(mode="after")
    def _check_order(self) -> Box:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"box has no area: {self.x1}, {self.y1}, {self.x2}, {self.y2}")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def foot(self) -> tuple[float, float]:
        """Bottom-centre of the box: the point where the person touches the pitch."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def iou(self, other: Box) -> float:
        """Intersection over union with another box, in [0, 1]."""
        iw = min(self.x2, other.x2) - max(self.x1, other.x1)
        ih = min(self.y2, other.y2) - max(self.y1, other.y1)
        if iw <= 0.0 or ih <= 0.0:
            return 0.0
        inter = iw * ih
        union = self.width * self.height + other.width * other.height - inter
        return inter / union


# --- Perception answers (one keyframe) --------------------------------------


class PersonDet(Record):
    """One person found in one frame."""

    box: Box
    role: Role
    team_key: TeamKey = "none"
    #: Shirt number read in this frame, or None when it is not readable.
    number: ShirtNumber | None = None


class FramePeople(Record):
    frame: int = Field(ge=0)
    people: tuple[PersonDet, ...] = ()


class LandmarkObs(Record):
    """One pitch landmark seen in the image. ``landmark_id`` refers to ``pitch.LANDMARKS``."""

    landmark_id: int
    x: FiniteFloat
    y: FiniteFloat


class FrameLandmarks(Record):
    frame: int = Field(ge=0)
    view: PitchView | None = None
    landmarks: tuple[LandmarkObs, ...] = ()


class KitDescription(Record):
    shirt: str
    shorts: str
    socks: str


class GoalkeeperKit(Record):
    kit: str
    team_key: TeamKey = "none"


class KitLegend(Record):
    """Kit colours of one clip. Pasted into later prompts so that requests share team labels."""

    team_a: KitDescription
    team_b: KitDescription
    goalkeepers: tuple[GoalkeeperKit, ...] = ()
    referee_kit: str = ""
    #: Notes of the model about what it could not tell. They do not go into prompts (see ``prompt_text``).
    uncertain: tuple[str, ...] = ()

    def prompt_text(self) -> str:
        """Short text block for prompts.

        The ``uncertain`` notes are left out on purpose. They are free text of
        the model, and this text is part of the prompt hash of every later
        request: with the notes in it, two legends with the same kits would
        give different cache keys.
        """
        lines = [
            f"Team A: {self.team_a.shirt} shirt, {self.team_a.shorts} shorts, {self.team_a.socks} socks.",
            f"Team B: {self.team_b.shirt} shirt, {self.team_b.shorts} shorts, {self.team_b.socks} socks.",
        ]
        for keeper in self.goalkeepers:
            owner = "unknown team" if keeper.team_key == "none" else f"team {keeper.team_key}"
            lines.append(f"Goalkeeper ({owner}): {keeper.kit}.")
        if self.referee_kit:
            lines.append(f"Referees: {self.referee_kit}.")
        return "\n".join(lines)


class NumberReading(Record):
    """Shirt number read from several crops of one track."""

    per_crop: tuple[ShirtNumber | None, ...] = ()
    number: ShirtNumber | None = None
    confidence: Confidence = "low"


# --- Geometry ----------------------------------------------------------------


class HomographyRecord(Record):
    """Image-to-pitch mapping of one frame: ``pitch = H @ (x_px, y_px, 1)``."""

    frame: int = Field(ge=0)
    #: 3x3 row-major matrix of finite numbers, or None when the frame has no mapping.
    matrix: tuple[tuple[FiniteFloat, FiniteFloat, FiniteFloat], ...] | None = None
    source: HomographySource = "none"
    n_inliers: int = Field(default=0, ge=0)
    #: Median reprojection error of the inliers in image pixels (landmark fits only).
    reproj_px: FiniteFloat | None = None

    @model_validator(mode="after")
    def _check_matrix(self) -> HomographyRecord:
        if self.source == "none" and self.matrix is not None:
            raise ValueError("source 'none' must not carry a matrix")
        if self.source != "none" and (self.matrix is None or len(self.matrix) != 3):
            raise ValueError(f"source '{self.source}' needs a 3x3 matrix")
        return self

    @property
    def valid(self) -> bool:
        return self.matrix is not None


# --- Tracks --------------------------------------------------------------------


class TrackObs(Record):
    """One keyframe observation of a track."""

    frame: int = Field(ge=0)
    box: Box
    role: Role
    team_key: TeamKey = "none"
    number: ShirtNumber | None = None


class Track(Record):
    """One person over time. Attribute fields are None until the votes are done."""

    track_id: int = Field(ge=0)
    observations: tuple[TrackObs, ...]
    role: Role | None = None
    team_key: TeamKey | None = None
    team_side: TeamSide | None = None
    #: Most likely shirt number (scoring). None means "no number".
    number_export: ShirtNumber | None = None
    #: Shirt number shown in the video. Set only when the reading is confident.
    number_display: ShirtNumber | None = None
    name: str | None = None

    @model_validator(mode="after")
    def _check_observations(self) -> Track:
        frames = [obs.frame for obs in self.observations]
        if not frames:
            raise ValueError("a track needs at least one observation")
        if frames != sorted(set(frames)):
            raise ValueError("track observations must have increasing, unique frames")
        return self

    @property
    def first_frame(self) -> int:
        return self.observations[0].frame

    @property
    def last_frame(self) -> int:
        return self.observations[-1].frame


# --- Dense output (every frame) ---------------------------------------------


class PlayerState(Record):
    """One person in one frame of the final output. The pitch position is two finite numbers."""

    track_id: int = Field(ge=0)
    #: Image box, or None when no box is known for this frame.
    box: Box | None = None
    #: Pitch position in metres. A state without a position does not exist: leave the person out.
    pitch_x: FiniteFloat
    pitch_y: FiniteFloat
    role: Role
    team_key: TeamKey = "none"
    team_side: TeamSide | None = None
    number_export: ShirtNumber | None = None
    number_display: ShirtNumber | None = None
    name: str | None = None
    #: True at keyframes where the model saw the person; False for filled-in frames.
    observed: bool = False


class BallState(Record):
    """The ball in one frame of the final output (detector engine only).

    The pitch position is the foot point of the box on the grass. A ball in the air is not on
    the grass, so its pitch position is wrong by the height of the ball: it is drawn on the
    radar and never exported. Both pitch values are given, or neither.
    """

    box: Box
    pitch_x: FiniteFloat | None = None
    pitch_y: FiniteFloat | None = None
    #: True in frames where the detector saw the ball; False where the ball tracker filled it in.
    observed: bool = False

    @model_validator(mode="after")
    def _check_pitch(self) -> BallState:
        if (self.pitch_x is None) != (self.pitch_y is None):
            raise ValueError("a ball state needs both pitch values or neither")
        return self


class FrameState(Record):
    frame: int = Field(ge=0)
    players: tuple[PlayerState, ...] = ()
    #: The ball, or None when the frame has no ball (always None for the model engine).
    ball: BallState | None = None


class ClipInfo(Record):
    clip_id: str
    n_frames: int = Field(gt=0)
    fps: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class ClipResult(Record):
    """Everything the pipeline produced for one clip."""

    info: ClipInfo
    frames: tuple[FrameState, ...]
    tracks: tuple[Track, ...]
    homographies: tuple[HomographyRecord, ...]
    #: Which engine did each step, e.g. {"people": "model:gemini-4-argon", "tracks": "code"}.
    #: A dict can be changed in place although the record is frozen: read it, do not edit it.
    engines: dict[str, str] = Field(default_factory=dict)
