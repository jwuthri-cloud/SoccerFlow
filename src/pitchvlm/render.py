"""Drawing: boxes and labels on the frame, the pitch radar with trails, the output video, and the
check image of a pitch mapping (the pitch lines that it projects into its frame).

Colours are BGR tuples. Image points are (x, y) pixels with the origin at the
top-left corner. The radar is a top-down view of the pitch: the far touchline
is at the top and the left goal is at the left, as in ``pitch.PitchCanvas``.
"""

from __future__ import annotations

import os
import unicodedata
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import cv2
import numpy as np

from pitchvlm.frames import FrameSource
from pitchvlm.geometry import projected_line_pixels
from pitchvlm.pitch import PitchCanvas
from pitchvlm.schemas import BallState, Box, ClipResult, FrameState, PlayerState, TeamKey

Colour = tuple[int, int, int]

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TEXT_THICKNESS = 1
_BOX_THICKNESS = 2
#: Space between the label text and the edge of its tab, pixels.
_LABEL_PAD = 3
#: Fixed colour of a team key. It is used while the team side is not known.
_TEAM_KEY_BGR: dict[TeamKey, Colour] = {"A": (0, 190, 255), "B": (200, 70, 190)}
_GRASS_BGR: Colour = (40, 110, 50)
_NUMBER_BGR: Colour = (255, 255, 255)
#: Radar sizes in metres, so that they follow the radar scale.
_DOT_RADIUS_M = 0.8
_TRAIL_WIDTH_M = 0.3
#: Radius of the ball on the radar, metres, and its smallest circle on the frame, pixels.
_BALL_RADIUS_M = 0.6
_BALL_MIN_RADIUS_PX = 4
#: The pitch lines of a mapping check image: magenta, a colour that neither grass nor paint has,
#: one pixel wide per this many pixels of the frame width (at least one pixel), and this opaque,
#: so that the white paint stays visible beside and below a line.
_MAPPING_LINE_BGR: Colour = (255, 0, 255)
_MAPPING_LINE_WIDTH_OF = 1280
_MAPPING_LINE_OPACITY = 0.7
#: The caption band of a mapping check image: white text on black, font scale per pixel of the frame width.
_CAPTION_FONT_PER_WIDTH = 1 / 1600
_CAPTION_MIN_FONT = 0.5


@dataclass(frozen=True)
class RenderStyle:
    """Look of the overlay and of the radar. Every colour is a BGR tuple.

    A team key without a team side has a fixed colour that is not part of the style.
    """

    #: Team that defends the left goal.
    left_bgr: Colour = (60, 60, 220)
    #: Team that defends the right goal.
    right_bgr: Colour = (220, 130, 30)
    referee_bgr: Colour = (200, 200, 200)
    #: People with the role "other", and team members with no side and no team key.
    other_bgr: Colour = (130, 130, 130)
    #: Length of a radar trail, seconds.
    trail_seconds: float = 4.0
    #: Size of the radar, pixels per metre.
    radar_scale: float = 10.0
    #: Font scale of the labels and of the radar numbers.
    font_scale: float = 0.6
    #: The ball (detector engine): a ring on the frame and a dot on the radar.
    ball_bgr: Colour = (255, 255, 255)

    def __post_init__(self) -> None:
        if self.trail_seconds < 0:
            raise ValueError(f"trail_seconds must not be negative, got {self.trail_seconds}")
        if self.radar_scale <= 0:
            raise ValueError(f"radar_scale must be positive, got {self.radar_scale}")
        if self.font_scale <= 0:
            raise ValueError(f"font_scale must be positive, got {self.font_scale}")


def player_colour(player: PlayerState, style: RenderStyle) -> Colour:
    """Return the drawing colour of one person.

    Referees and other people are grey. A team member takes the colour of the
    team side. When the side is not known, the team key gives a fixed colour.
    A team member with no side and no team key is grey.
    """
    if player.role == "referee":
        return style.referee_bgr
    if player.role == "other":
        return style.other_bgr
    if player.team_side == "left":
        return style.left_bgr
    if player.team_side == "right":
        return style.right_bgr
    return _TEAM_KEY_BGR.get(player.team_key, style.other_bgr)


def label_text(player: PlayerState) -> str:
    """Return the label of one person: ``#<number_display> <name>``.

    A part that is missing is left out. The text is empty when the person has
    no displayed number and no name. Accents are removed from the name,
    because the OpenCV fonts draw ASCII characters only.
    """
    parts: list[str] = []
    if player.number_display is not None:
        parts.append(f"#{player.number_display}")
    name = _to_ascii(player.name or "").strip()
    if name:
        parts.append(name)
    return " ".join(parts)


def _to_ascii(text: str) -> str:
    """Remove the accents and replace every other non-ASCII character by a question mark."""
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return without_accents.encode("ascii", "replace").decode("ascii")


def _check_image(image: np.ndarray) -> None:
    """Raise ValueError when the array is not a BGR uint8 image of shape (height, width, 3)."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError(
            f"expected a BGR uint8 image of shape (height, width, 3), got {image.dtype} with shape {image.shape}"
        )


def _touches_image(box: Box, width: int, height: int) -> bool:
    """Tell whether a part of the box is inside an image of the given size."""
    return box.x2 > 0 and box.y2 > 0 and box.x1 < width and box.y1 < height


def _text_colour(background: Colour) -> Colour:
    """Return black for a light background and white for a dark background."""
    blue, green, red = background
    luminance = 0.114 * blue + 0.587 * green + 0.299 * red
    return (0, 0, 0) if luminance > 140.0 else (255, 255, 255)


def _draw_box(canvas: np.ndarray, box: Box, colour: Colour) -> None:
    """Draw the outline of a box on the canvas."""
    top_left = (round(box.x1), round(box.y1))
    bottom_right = (round(box.x2), round(box.y2))
    cv2.rectangle(canvas, top_left, bottom_right, colour, thickness=_BOX_THICKNESS)


def _draw_label(canvas: np.ndarray, text: str, box: Box, colour: Colour, font_scale: float) -> None:
    """Draw the label on a filled tab above the box.

    The tab is ``tab_w`` x ``tab_h`` pixels and ends one row above the top
    edge of the box. It moves sideways or down when it would leave the image,
    so that the text of a person at the edge of the picture stays readable.
    """
    (text_w, text_h), baseline = cv2.getTextSize(text, _FONT, font_scale, _TEXT_THICKNESS)
    tab_w = text_w + 2 * _LABEL_PAD
    tab_h = text_h + baseline + 2 * _LABEL_PAD
    left = max(0, min(round(box.x1), canvas.shape[1] - tab_w))
    top = max(0, round(box.y1) - tab_h)
    cv2.rectangle(canvas, (left, top), (left + tab_w - 1, top + tab_h - 1), colour, thickness=cv2.FILLED)
    # The text origin of OpenCV is the left end of the base line of the text.
    text_origin = (left + _LABEL_PAD, top + _LABEL_PAD + text_h)
    cv2.putText(canvas, text, text_origin, _FONT, font_scale, _text_colour(colour), _TEXT_THICKNESS, cv2.LINE_AA)


def draw_overlay(image: np.ndarray, players: Sequence[PlayerState], style: RenderStyle = RenderStyle()) -> np.ndarray:
    """Return a copy of the frame with a box and a label for every person.

    The image is BGR uint8 of shape (height, width, 3) and is not changed. A
    person without a box, or with a box that is fully outside the image, is
    skipped. The label is left out when the person has no displayed number
    and no name.
    """
    _check_image(image)
    canvas = image.copy()
    height, width = canvas.shape[:2]
    for player in players:
        box = player.box
        if box is None or not _touches_image(box, width, height):
            continue
        colour = player_colour(player, style)
        _draw_box(canvas, box, colour)
        text = label_text(player)
        if text:
            _draw_label(canvas, text, box, colour, style.font_scale)
    return canvas


def draw_ball(image: np.ndarray, ball: BallState, style: RenderStyle = RenderStyle()) -> np.ndarray:
    """Return a copy of the frame with a ring around the ball.

    A ball that the tracker filled in (``observed`` False) gets a thinner ring. The image
    is not changed.
    """
    _check_image(image)
    canvas = image.copy()
    centre_x, centre_y = ball.box.center
    radius = max(_BALL_MIN_RADIUS_PX, round(max(ball.box.width, ball.box.height)))
    thickness = _BOX_THICKNESS if ball.observed else 1
    cv2.circle(canvas, (round(centre_x), round(centre_y)), radius, style.ball_bgr, thickness, cv2.LINE_AA)
    return canvas


def _mix(colour: Colour, background: Colour, weight: float) -> Colour:
    """Blend a colour with a background. Weight 1 gives the colour and weight 0 gives the background."""
    channels = zip(colour, background, strict=True)
    blue, green, red = (round(back + (front - back) * weight) for front, back in channels)
    return (blue, green, red)


class RadarRenderer:
    """Top-down pitch view with one dot per person and a fading trail per track id.

    Call ``draw`` for the frames of a clip in their order. The renderer
    remembers the positions of every track from the frames that it drew, for
    ``trail_seconds`` seconds of clip time. A track that is not seen any more
    loses its trail with time.

    A trail joins two positions of a track only when they come from two frames
    that were drawn one after the other. Where a track had no state in a frame
    that was drawn, its trail has a gap: nobody measured the way of the person
    there, so no line is drawn across it.

    ``cuts`` are the frames that start a new shot (``ClipResult.cuts``). A
    position of one shot says nothing about the pitch of the next, so every
    trail ends at a cut: the first frame of a shot shows no trail from before.
    """

    def __init__(self, fps: float, style: RenderStyle = RenderStyle(), *, cuts: Sequence[int] = ()) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self._style = style
        self._canvas = PitchCanvas(scale=style.radar_scale)
        self._pitch = self._canvas.draw(background_bgr=_GRASS_BGR)
        #: True where a pitch marking is. The markings are put back on top of the trails.
        self._markings = np.any(self._pitch != np.array(_GRASS_BGR, dtype=np.uint8), axis=2)
        #: Largest age of a remembered position, in frames.
        self._trail_frames = round(style.trail_seconds * fps)
        #: Track id -> remembered (frame, pitch x, pitch y), oldest first, frames strictly increasing.
        self._trails: dict[int, deque[tuple[int, float, float]]] = {}
        #: The frames that were drawn and are not older than the trail time, in increasing order.
        self._drawn_frames: deque[int] = deque()
        #: Track id -> colour of the track when it was last seen.
        self._colours: dict[int, Colour] = {}
        self._cuts = tuple(cuts)

    def draw(self, state: FrameState) -> np.ndarray:
        """Return the radar of one frame as a new BGR image.

        The positions of the frame are added to the trails first. A trail
        never holds a position from a later frame, so a frame that is drawn
        again gives the same image. A track id that appears twice in the
        frame raises ValueError.
        """
        self._remember(state)
        image = self._pitch.copy()
        self._draw_trails(image, state.frame)
        image[self._markings] = self._pitch[self._markings]
        self._draw_players(image, state.players)
        if state.ball is not None and state.ball.pitch_x is not None and state.ball.pitch_y is not None:
            radius = max(2, round(_BALL_RADIUS_M * self._style.radar_scale))
            (centre,) = self._to_pixels([(state.ball.pitch_x, state.ball.pitch_y)])
            cv2.circle(image, centre, radius, self._style.ball_bgr, thickness=cv2.FILLED, lineType=cv2.LINE_AA)
            cv2.circle(image, centre, radius, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        return image

    def trail(self, track_id: int) -> np.ndarray:
        """Return the remembered pitch positions of one track in metres, oldest first, shape (n, 2).

        A track with no remembered position gives an array of shape (0, 2).
        """
        points = self._trails.get(track_id, ())
        return np.array([(x, y) for _, x, y in points], dtype=np.float64).reshape(-1, 2)

    def _remember(self, state: FrameState) -> None:
        """Bring the trails to the time of the frame, and then add the positions of the frame.

        Positions that are older than the trail time are dropped. Positions of
        this frame or of a later frame are dropped too: they exist only when a
        frame is drawn again or when the caller goes back in the clip.
        """
        track_ids = [player.track_id for player in state.players]
        if len(set(track_ids)) != len(track_ids):
            raise ValueError(f"a track id appears more than once in frame {state.frame}")
        # A trail keeps no position from another shot: the oldest frame is the start of the shot at most.
        shot_start = max((cut for cut in self._cuts if cut <= state.frame), default=0)
        oldest_frame = max(state.frame - self._trail_frames, shot_start)
        while self._drawn_frames and self._drawn_frames[-1] >= state.frame:
            self._drawn_frames.pop()
        while self._drawn_frames and self._drawn_frames[0] < oldest_frame:
            self._drawn_frames.popleft()
        self._drawn_frames.append(state.frame)
        for track_id in list(self._trails):
            trail = self._trails[track_id]
            while trail and trail[-1][0] >= state.frame:
                trail.pop()
            while trail and trail[0][0] < oldest_frame:
                trail.popleft()
            if not trail:
                del self._trails[track_id]
                del self._colours[track_id]
        for player in state.players:
            trail = self._trails.setdefault(player.track_id, deque())
            trail.append((state.frame, player.pitch_x, player.pitch_y))
            self._colours[player.track_id] = player_colour(player, self._style)

    def _to_pixels(self, positions: Sequence[tuple[float, float]]) -> list[tuple[int, int]]:
        """Convert pitch positions in metres to whole canvas pixels (x, y)."""
        pixels = np.rint(self._canvas.to_px(np.array(positions, dtype=np.float64)))
        return [(int(u), int(v)) for u, v in pixels]

    def _draw_trails(self, image: np.ndarray, frame: int) -> None:
        """Draw every trail. A segment fades into the grass colour as it gets older.

        A segment joins two remembered positions of a track. It is drawn only
        when the two frames were drawn one after the other: a track without a
        state in a drawn frame between them has a gap in its trail there.
        """
        thickness = max(1, round(_TRAIL_WIDTH_M * self._style.radar_scale))
        drawn_before = dict(zip(list(self._drawn_frames)[1:], self._drawn_frames))
        for track_id, trail in self._trails.items():
            colour = self._colours[track_id]
            point_frames = [point_frame for point_frame, _, _ in trail]
            pixels = self._to_pixels([(x, y) for _, x, y in trail])
            for index in range(1, len(pixels)):
                if drawn_before.get(point_frames[index]) != point_frames[index - 1]:
                    continue
                age = (frame - point_frames[index]) / self._trail_frames
                segment_colour = _mix(colour, _GRASS_BGR, 1.0 - age)
                cv2.line(image, pixels[index - 1], pixels[index], segment_colour, thickness, cv2.LINE_AA)

    def _draw_players(self, image: np.ndarray, players: Sequence[PlayerState]) -> None:
        """Draw one dot per person, and then the displayed numbers to the right of the dots."""
        radius = max(2, round(_DOT_RADIUS_M * self._style.radar_scale))
        centres = self._to_pixels([(player.pitch_x, player.pitch_y) for player in players])
        for player, centre in zip(players, centres, strict=True):
            colour = player_colour(player, self._style)
            cv2.circle(image, centre, radius, colour, thickness=cv2.FILLED, lineType=cv2.LINE_AA)
        font_scale = self._style.font_scale
        (_, text_h), _ = cv2.getTextSize("0", _FONT, font_scale, _TEXT_THICKNESS)
        for player, (u, v) in zip(players, centres, strict=True):
            if player.number_display is None:
                continue
            text_origin = (u + radius + 2, v + text_h // 2)
            number = str(player.number_display)
            cv2.putText(image, number, text_origin, _FONT, font_scale, _NUMBER_BGR, _TEXT_THICKNESS, cv2.LINE_AA)


def compose(overlay: np.ndarray, radar: np.ndarray) -> np.ndarray:
    """Return one image with the overlay on top and the radar below.

    The radar is resized to the width of the overlay and keeps its aspect
    ratio. Both inputs stay unchanged.
    """
    _check_image(overlay)
    _check_image(radar)
    width = overlay.shape[1]
    radar_h, radar_w = radar.shape[:2]
    height = max(1, round(radar_h * width / radar_w))
    interpolation = cv2.INTER_AREA if width < radar_w else cv2.INTER_LINEAR
    resized = cv2.resize(radar, (width, height), interpolation=interpolation)
    return np.vstack([overlay, resized])


class VideoSink:
    """Video file writer on top of ``cv2.VideoWriter`` with the codec mp4v.

    ``size`` is (width, height) in pixels. Use the sink as a context manager,
    so that the file is closed and complete at the end.
    """

    def __init__(self, path: Path, fps: float, size: tuple[int, int]) -> None:
        width, height = int(size[0]), int(size[1])
        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"fps and both sides must be positive, got fps {fps} and size {width}x{height}")
        self._size = (width, height)
        self._writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), self._size)
        if not self._writer.isOpened():
            self._writer.release()
            raise OSError(f"cannot open a video writer for {path} (codec mp4v, {width}x{height} pixels)")

    def write(self, image: np.ndarray) -> None:
        """Add one frame. The image must be BGR uint8 and must have the size of the sink."""
        _check_image(image)
        height, width = image.shape[:2]
        if (width, height) != self._size:
            raise ValueError(
                f"the image is {width}x{height} pixels, but the sink needs {self._size[0]}x{self._size[1]}"
            )
        if not self._writer.isOpened():
            raise OSError("the video sink is closed")
        self._writer.write(image)

    def close(self) -> None:
        """Finish the file. No frame can be written after this call."""
        self._writer.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()


def _states_by_frame(result: ClipResult, n_frames: int) -> list[FrameState]:
    """Return the state of every frame of the clip, in frame order.

    The result must hold exactly one state for each frame 0 to n_frames - 1.
    Any other content raises ValueError.
    """
    by_frame = {state.frame: state for state in result.frames}
    if len(result.frames) != n_frames or sorted(by_frame) != list(range(n_frames)):
        raise ValueError(
            f"the result must hold exactly one state for each frame 0 to {n_frames - 1}, "
            f"but it holds {len(result.frames)} states for {len(by_frame)} different frames"
        )
    return [by_frame[frame] for frame in range(n_frames)]


def _crop_to_even(image: np.ndarray) -> np.ndarray:
    """Cut the last row or column when that side is odd, because video codecs need even sides."""
    height, width = image.shape[:2]
    return np.ascontiguousarray(image[: height - height % 2, : width - width % 2])


def _video_frames(
    source: FrameSource, states: Sequence[FrameState], style: RenderStyle, cuts: Sequence[int]
) -> Iterator[np.ndarray]:
    """Yield the video image of every state: the overlay on top and the radar below."""
    radar = RadarRenderer(source.info.fps, style, cuts=cuts)
    for state in states:
        overlay = draw_overlay(source.read(state.frame), state.players, style)
        if state.ball is not None:
            overlay = draw_ball(overlay, state.ball, style)
        yield _crop_to_even(compose(overlay, radar.draw(state)))


def render_clip(source: FrameSource, result: ClipResult, out_path: Path, style: RenderStyle = RenderStyle()) -> Path:
    """Write the annotated video of a clip and return its path.

    Every frame of the source is drawn with its overlay on top and the radar
    below. The result must hold one state for each frame of the source. The
    folder of ``out_path`` is made when it is missing. A video side must be
    even, so one pixel row or column is cut from an odd side.

    The video is written under another name in the same folder
    (``<name>.partial<ending>``) and gets its own name when it is complete. So
    a file with the name ``out_path`` is always a complete video: a run that
    is stopped while it writes leaves an older video as it was. A write that
    fails removes the unfinished file.
    """
    out_path = Path(out_path)
    images = _video_frames(source, _states_by_frame(result, source.info.n_frames), style, result.cuts)
    first = next(images)
    height, width = first.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_name(f"{out_path.stem}.partial{out_path.suffix}")
    try:
        with VideoSink(partial, source.info.fps, (width, height)) as sink:
            sink.write(first)
            for image in images:
                sink.write(image)
        os.replace(partial, out_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return out_path


def draw_mapping_check(image: np.ndarray, H: np.ndarray, caption: Sequence[str]) -> np.ndarray:
    """Return the check image of a pitch mapping: the frame with the pitch lines that the mapping projects into it.

    ``H`` maps the pixels of ``image`` to pitch metres. The lines are the pixels of
    ``geometry.projected_line_pixels``, the same pixels that the paint alignment score
    measures, 1 pixel wide in a frame of 1280 pixels and 2 in a frame of 1920 pixels, in
    magenta that is 70 % opaque. A painted line of a broadcast frame is wider, so a good
    mapping draws a thin magenta line inside the white paint, and a person sees by eye where
    it does not.
    The caption goes on a black band above the frame, one line of ASCII text per entry, so
    that no text hides a pixel of the frame. ``ValueError`` is raised for an image that is
    not a BGR ``uint8`` image and for a mapping without a usable inverse.
    """
    _check_image(image)
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    pixels = projected_line_pixels(H, (width, height))
    mask[pixels[:, 1], pixels[:, 0]] = 1
    line_width = max(1, round(width / _MAPPING_LINE_WIDTH_OF))
    if line_width > 1:
        mask = cv2.dilate(mask, np.ones((line_width, line_width), dtype=np.uint8))
    framed = image.copy()
    on_line = mask > 0
    blended = (1.0 - _MAPPING_LINE_OPACITY) * framed[on_line] + _MAPPING_LINE_OPACITY * np.array(_MAPPING_LINE_BGR)
    framed[on_line] = np.round(blended).astype(np.uint8)
    font_scale = max(_CAPTION_MIN_FONT, width * _CAPTION_FONT_PER_WIDTH)
    texts = [_to_ascii(line) for line in caption]
    (_, text_h), baseline = cv2.getTextSize("Ag", _FONT, font_scale, _TEXT_THICKNESS)
    line_h = text_h + baseline + 2 * _LABEL_PAD
    band = np.zeros((line_h * len(texts) + 2 * _LABEL_PAD, width, 3), dtype=np.uint8)
    for index, text in enumerate(texts):
        origin = (2 * _LABEL_PAD, _LABEL_PAD + index * line_h + _LABEL_PAD + text_h)
        cv2.putText(band, text, origin, _FONT, font_scale, (255, 255, 255), _TEXT_THICKNESS, cv2.LINE_AA)
    return np.vstack([band, framed])


def write_png(path: Path, image: np.ndarray) -> None:
    """Write an image as a PNG file. ``OSError`` is raised when it cannot be written."""
    if not cv2.imwrite(str(path), image):
        raise OSError(f"cannot write the image {path}")
