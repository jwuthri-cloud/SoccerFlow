"""Tests of render.py: overlay, radar with trails, composition, video sink and clip rendering.

The tests build every image in memory. Image arrays are indexed [row, column],
so a pixel (x, y) is read as image[y, x].
"""

from __future__ import annotations

import dataclasses
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from pitchvlm.frames import FrameSource
from pitchvlm.pitch import PitchCanvas
from pitchvlm.render import (
    RadarRenderer,
    RenderStyle,
    VideoSink,
    compose,
    draw_overlay,
    label_text,
    player_colour,
    render_clip,
)
from pitchvlm.schemas import Box, ClipInfo, ClipResult, FrameState, PlayerState

#: A box that is higher than wide, so that a swap of x and y shows up.
BOX = Box(x1=40.0, y1=50.0, x2=80.0, y2=100.0)


# --- Helpers -------------------------------------------------------------------


def _player(**fields: Any) -> PlayerState:
    """Build a player at the centre mark. The given fields replace the defaults."""
    values: dict[str, Any] = {"track_id": 1, "pitch_x": 0.0, "pitch_y": 0.0, "role": "player"}
    values.update(fields)
    return PlayerState(**values)


def _noise_image(height: int, width: int, seed: int) -> np.ndarray:
    """Build a random BGR image from a seed."""
    return np.random.default_rng(seed).integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _changed(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return a (height, width) mask of the pixels that differ between two images."""
    return np.any(first != second, axis=2)


def _bgr(image: np.ndarray, x: int, y: int) -> tuple[int, ...]:
    """Return the colour of the pixel (x, y) as plain integers."""
    return tuple(int(value) for value in image[y, x])


def _difference(image: np.ndarray, reference: np.ndarray, x: int, y: int) -> int:
    """Return how far the pixel (x, y) of an image is from the same pixel of a reference image."""
    return int(np.abs(image[y, x].astype(int) - reference[y, x].astype(int)).sum())


def _radar_pixel(style: RenderStyle, pitch_x: float, pitch_y: float) -> tuple[int, int]:
    """Return the radar pixel (x, y) of a pitch position in metres."""
    u, v = PitchCanvas(scale=style.radar_scale).to_px(np.array([[pitch_x, pitch_y]]))[0]
    return round(float(u)), round(float(v))


def _empty_radar(fps: float, style: RenderStyle) -> np.ndarray:
    """Return the radar image of a frame without people."""
    return RadarRenderer(fps, style).draw(FrameState(frame=0))


class _ListSource:
    """Frame source over a list of images that are held in memory."""

    def __init__(self, frames: list[np.ndarray], fps: float = 25.0) -> None:
        height, width = frames[0].shape[:2]
        self.info = ClipInfo(clip_id="test-clip", n_frames=len(frames), fps=fps, width=width, height=height)
        self._frames = frames

    def read(self, frame: int) -> np.ndarray:
        return self._frames[frame]


def _clip(n_frames: int, height: int, width: int, seed: int) -> tuple[_ListSource, ClipResult]:
    """Build a small clip of noise frames with one player that moves to the right."""
    rng = np.random.default_rng(seed)
    frames = [rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8) for _ in range(n_frames)]
    source = _ListSource(frames)
    states = tuple(
        FrameState(
            frame=frame,
            players=(
                _player(
                    track_id=1,
                    box=Box(x1=10.0 + frame, y1=10.0, x2=30.0 + frame, y2=50.0),
                    pitch_x=-20.0 + 4.0 * frame,
                    pitch_y=-10.0,
                    team_side="left",
                    number_display=8,
                ),
            ),
        )
        for frame in range(n_frames)
    )
    return source, ClipResult(info=source.info, frames=states, tracks=(), homographies=())


class _FakeWriter:
    """Stand-in for ``cv2.VideoWriter`` that keeps the frames in memory.

    Like the real writer, it leaves a file under its name: an empty one, when it is released.
    """

    def __init__(self, filename: str, fourcc: int, fps: float, frame_size: tuple[int, int]) -> None:
        self.filename = filename
        self.fourcc = fourcc
        self.fps = fps
        self.frame_size = frame_size
        self.frames: list[np.ndarray] = []
        self._open = True

    def isOpened(self) -> bool:  # noqa: N802 (the OpenCV name)
        return self._open

    def write(self, image: np.ndarray) -> None:
        self.frames.append(image.copy())

    def release(self) -> None:
        self._open = False
        Path(self.filename).touch()


@pytest.fixture
def fake_writers(monkeypatch: pytest.MonkeyPatch) -> list[_FakeWriter]:
    """Replace ``cv2.VideoWriter`` by the fake. The list holds every writer that the test opens."""
    opened: list[_FakeWriter] = []

    def make_writer(filename: str, fourcc: int, fps: float, frame_size: tuple[int, int]) -> _FakeWriter:
        writer = _FakeWriter(filename, fourcc, fps, frame_size)
        opened.append(writer)
        return writer

    monkeypatch.setattr(cv2, "VideoWriter", make_writer)
    return opened


def _mp4v_available(folder: Path) -> bool:
    """Tell whether the installed OpenCV build can write a video with the codec mp4v."""
    probe = cv2.VideoWriter(str(folder / "probe.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
    opened = probe.isOpened()
    probe.release()
    return opened


def _read_video(path: Path) -> list[np.ndarray]:
    """Read every frame of a video file."""
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    frames: list[np.ndarray] = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frames.append(image)
    capture.release()
    return frames


# --- Style, colours and labels -------------------------------------------------


def test_default_style_follows_the_contract() -> None:
    style = RenderStyle()
    assert style.trail_seconds == 4.0
    assert style.left_bgr != style.right_bgr
    for grey in (style.referee_bgr, style.other_bgr):
        assert grey[0] == grey[1] == grey[2]
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(style, "radar_scale", 5.0)


@pytest.mark.parametrize(
    "fields",
    [{"trail_seconds": -1.0}, {"radar_scale": 0.0}, {"font_scale": 0.0}],
)
def test_style_rejects_wrong_values(fields: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        RenderStyle(**fields)


def test_colour_comes_from_the_team_side() -> None:
    style = RenderStyle()
    assert player_colour(_player(team_side="left"), style) == style.left_bgr
    assert player_colour(_player(team_side="right"), style) == style.right_bgr
    assert player_colour(_player(role="goalkeeper", team_side="right"), style) == style.right_bgr
    # The side wins over the team key.
    assert player_colour(_player(team_key="B", team_side="left"), style) == style.left_bgr


def test_referees_and_others_are_grey() -> None:
    style = RenderStyle()
    assert player_colour(_player(role="referee"), style) == style.referee_bgr
    assert player_colour(_player(role="other"), style) == style.other_bgr
    # The role wins over a team side that should not be there.
    assert player_colour(_player(role="referee", team_side="left"), style) == style.referee_bgr
    # A team member with no side and no team key is grey too.
    assert player_colour(_player(), style) == style.other_bgr


def test_team_key_gives_a_fixed_colour_when_the_side_is_unknown() -> None:
    style = RenderStyle()
    other_style = RenderStyle(left_bgr=(1, 2, 3), right_bgr=(4, 5, 6), referee_bgr=(7, 7, 7), other_bgr=(9, 9, 9))
    colour_a = player_colour(_player(team_key="A"), style)
    colour_b = player_colour(_player(team_key="B"), style)
    assert colour_a != colour_b
    assert {colour_a, colour_b}.isdisjoint({style.left_bgr, style.right_bgr, style.referee_bgr, style.other_bgr})
    assert player_colour(_player(team_key="A"), other_style) == colour_a
    assert player_colour(_player(team_key="B"), other_style) == colour_b


def test_label_text_shows_the_displayed_number_and_the_name() -> None:
    assert label_text(_player(number_display=10, name="Ada Lovelace")) == "#10 Ada Lovelace"
    assert label_text(_player(number_display=7)) == "#7"
    assert label_text(_player(number_display=0)) == "#0"
    assert label_text(_player(name="Ada Lovelace")) == "Ada Lovelace"


def test_label_text_is_empty_without_displayed_number_and_name() -> None:
    assert label_text(_player()) == ""
    assert label_text(_player(name="  ")) == ""
    # The export number is for the scorer only. It is never shown.
    assert label_text(_player(number_export=9)) == ""


def test_label_text_removes_accents() -> None:
    # The OpenCV fonts draw ASCII only: "u" with a diaeresis becomes "u", and "O" with a stroke becomes "?".
    assert label_text(_player(number_display=13, name="M\N{LATIN SMALL LETTER U WITH DIAERESIS}ller")) == "#13 Muller"
    assert label_text(_player(name="\N{LATIN CAPITAL LETTER O WITH STROKE}degaard")) == "?degaard"


# --- Overlay -------------------------------------------------------------------


def test_draw_overlay_keeps_its_input_and_draws_the_box() -> None:
    image = _noise_image(120, 160, seed=1)
    before = image.copy()
    style = RenderStyle()

    out = draw_overlay(image, [_player(box=BOX, team_side="left")], style)

    assert np.array_equal(image, before)
    assert not np.shares_memory(out, image)
    assert out.shape == image.shape
    assert out.dtype == np.uint8
    # Pixels change inside the box: rows 50..100 (y), columns 40..80 (x).
    assert _changed(out, image)[50:101, 40:81].any()
    # The left edge is at x = 40 and the bottom edge is at y = 100, in the colour of the left team.
    assert _bgr(out, x=40, y=75) == style.left_bgr
    assert _bgr(out, x=60, y=100) == style.left_bgr


def test_draw_overlay_uses_the_default_style() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    out = draw_overlay(image, [_player(box=BOX, team_side="right")])
    assert _bgr(out, x=80, y=75) == RenderStyle().right_bgr


def test_draw_overlay_writes_a_label_only_with_a_number_or_a_name() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    outside_box = np.ones(image.shape[:2], dtype=bool)
    outside_box[45:106, 35:86] = False  # the box, its outline and a small margin

    plain = _changed(draw_overlay(image, [_player(box=BOX, team_side="right")]), image)
    numbered = _changed(draw_overlay(image, [_player(box=BOX, team_side="right", number_display=7)]), image)
    named = _changed(draw_overlay(image, [_player(box=BOX, team_side="right", name="Ada")]), image)

    assert not (plain & outside_box).any()
    for labelled in (numbered, named):
        rows, _ = np.nonzero(labelled & outside_box)
        assert rows.size > 0
        # The label sits above the box, whose top edge is at y = 50.
        assert rows.max() < 50


def test_draw_overlay_keeps_the_label_inside_the_image() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)

    def label_area(box: Box) -> int:
        player = _player(box=box, team_side="left", number_display=7)
        changed = _changed(draw_overlay(image, [player]), image)
        return int(changed[: round(box.y1) - 3].sum())  # the rows above the outline of the box

    in_the_middle = label_area(Box(x1=60.0, y1=40.0, x2=69.0, y2=90.0))
    at_the_right_edge = label_area(Box(x1=150.0, y1=40.0, x2=159.0, y2=90.0))
    assert in_the_middle > 0
    assert at_the_right_edge == in_the_middle


def test_draw_overlay_skips_people_without_a_visible_box() -> None:
    image = _noise_image(120, 160, seed=2)
    no_box = _player(track_id=1, number_display=9, name="Ada")
    outside = _player(track_id=2, box=Box(x1=200.0, y1=10.0, x2=240.0, y2=60.0), number_display=4)

    out = draw_overlay(image, [no_box, outside])

    assert np.array_equal(out, image)
    assert not np.shares_memory(out, image)


def test_draw_overlay_draws_a_box_that_is_partly_outside() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    style = RenderStyle()
    out = draw_overlay(image, [_player(box=Box(x1=-20.0, y1=30.0, x2=20.0, y2=90.0), team_side="right")], style)
    assert _bgr(out, x=20, y=60) == style.right_bgr


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((20, 30), dtype=np.uint8),
        np.zeros((20, 30, 4), dtype=np.uint8),
        np.zeros((20, 30, 3), dtype=np.float32),
    ],
)
def test_draw_overlay_rejects_a_wrong_image(image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        draw_overlay(image, [])


# --- Radar ---------------------------------------------------------------------


def test_radar_has_the_size_of_the_pitch_canvas() -> None:
    style = RenderStyle(radar_scale=4.0)
    canvas = PitchCanvas(scale=4.0)
    image = _empty_radar(25.0, style)
    assert image.shape == (canvas.height, canvas.width, 3)
    assert image.dtype == np.uint8


def test_radar_puts_far_left_at_top_left_and_near_right_at_bottom_right() -> None:
    style = RenderStyle()
    empty = _empty_radar(25.0, style)
    height, width = empty.shape[:2]

    # Far side is negative y, left goal is negative x.
    far_left = _player(track_id=1, pitch_x=-40.0, pitch_y=-25.0, team_side="left")
    image = RadarRenderer(25.0, style).draw(FrameState(frame=0, players=(far_left,)))
    rows, columns = np.nonzero(_changed(image, empty))
    assert rows.size > 0
    assert rows.max() < height / 2
    assert columns.max() < width / 2
    x, y = _radar_pixel(style, -40.0, -25.0)
    assert _bgr(image, x, y) == style.left_bgr

    near_right = _player(track_id=2, pitch_x=40.0, pitch_y=25.0, team_side="right")
    image = RadarRenderer(25.0, style).draw(FrameState(frame=0, players=(near_right,)))
    rows, columns = np.nonzero(_changed(image, empty))
    assert rows.size > 0
    assert rows.min() > height / 2
    assert columns.min() > width / 2
    x, y = _radar_pixel(style, 40.0, 25.0)
    assert _bgr(image, x, y) == style.right_bgr


def test_radar_writes_the_displayed_number_right_of_the_dot() -> None:
    style = RenderStyle()
    plain = RadarRenderer(25.0, style).draw(FrameState(frame=0, players=(_player(team_side="left"),)))
    exported = RadarRenderer(25.0, style).draw(
        FrameState(frame=0, players=(_player(team_side="left", number_export=9),))
    )
    numbered = RadarRenderer(25.0, style).draw(
        FrameState(frame=0, players=(_player(team_side="left", number_display=9),))
    )

    assert np.array_equal(exported, plain)
    _, columns = np.nonzero(_changed(numbered, plain))
    dot_x, _ = _radar_pixel(style, 0.0, 0.0)
    assert columns.size > 0
    assert columns.min() > dot_x


def test_radar_returns_a_new_image_every_time() -> None:
    style = RenderStyle(radar_scale=2.0)
    renderer = RadarRenderer(25.0, style)
    first = renderer.draw(FrameState(frame=0))
    expected = first.copy()
    first[:] = 0
    second = renderer.draw(FrameState(frame=1))
    assert np.array_equal(second, expected)


def test_trails_grow_and_then_stay_bounded() -> None:
    # 0.5 s at 10 frames per second is 5 frame steps: the current position and 5 older ones.
    renderer = RadarRenderer(10.0, RenderStyle(trail_seconds=0.5, radar_scale=4.0))
    lengths = []
    for frame in range(20):
        player = _player(track_id=3, pitch_x=-20.0 + frame, pitch_y=5.0)
        renderer.draw(FrameState(frame=frame, players=(player,)))
        lengths.append(len(renderer.trail(3)))

    assert lengths[:6] == [1, 2, 3, 4, 5, 6]
    assert lengths[6:] == [6] * 14

    trail = renderer.trail(3)
    assert trail.shape == (6, 2)
    assert trail.dtype == np.float64
    # Oldest first: frame 14 at x = -6, then up to frame 19 at x = -1.
    assert np.allclose(trail[:, 0], [-6.0, -5.0, -4.0, -3.0, -2.0, -1.0])
    assert np.allclose(trail[:, 1], 5.0)
    assert renderer.trail(99).shape == (0, 2)


def test_trail_of_a_missing_track_ages_out() -> None:
    style = RenderStyle(trail_seconds=0.5, radar_scale=4.0)
    renderer = RadarRenderer(10.0, style)
    for frame in range(3):
        player = _player(track_id=1, pitch_x=float(frame), pitch_y=0.0)
        renderer.draw(FrameState(frame=frame, players=(player,)))
    assert len(renderer.trail(1)) == 3

    renderer.draw(FrameState(frame=5))  # frames 0, 1 and 2 are at most 5 frames old
    assert len(renderer.trail(1)) == 3
    renderer.draw(FrameState(frame=7))  # only frame 2 is at most 5 frames old
    assert len(renderer.trail(1)) == 1
    image = renderer.draw(FrameState(frame=8))
    assert renderer.trail(1).shape == (0, 2)
    assert np.array_equal(image, _empty_radar(10.0, style))


def test_trail_is_drawn_and_fades_with_age() -> None:
    fps = 10.0
    style = RenderStyle(trail_seconds=1.0)
    empty = _empty_radar(fps, style)
    renderer = RadarRenderer(fps, style)
    image = empty
    # The player runs to the right at 2 m per frame along y = 15 m, where no pitch marking is.
    for frame in range(11):
        player = _player(track_id=5, pitch_x=-30.0 + 2.0 * frame, pitch_y=15.0, team_side="left")
        image = renderer.draw(FrameState(frame=frame, players=(player,)))

    # Middle of the nine oldest segments, then a point of the newest segment that the dot does not cover.
    sample_xs = [-29.0 + 2.0 * index for index in range(9)] + [-11.5]
    differences = [_difference(image, empty, *_radar_pixel(style, x, 15.0)) for x in sample_xs]
    assert differences[0] > 0
    assert all(older < newer for older, newer in pairwise(differences))


def test_trail_has_a_gap_where_the_track_has_no_state() -> None:
    fps = 10.0
    style = RenderStyle(trail_seconds=4.0)
    empty = _empty_radar(fps, style)
    renderer = RadarRenderer(fps, style)
    image = empty
    # The player runs to the right at 1 m per frame along y = 15 m. The frames 6 to 19 have no
    # state of the track: a gap that the dense output does not fill.
    for frame in range(26):
        players = ()
        if frame <= 5 or frame >= 20:
            players = (_player(track_id=5, pitch_x=-30.0 + frame, pitch_y=15.0, team_side="left"),)
        image = renderer.draw(FrameState(frame=frame, players=players))

    # The trail is drawn before the gap (x = -30 to -25 m) and after it (x = -10 to -5 m).
    assert _difference(image, empty, *_radar_pixel(style, -27.5, 15.0)) > 0
    assert _difference(image, empty, *_radar_pixel(style, -7.5, 15.0)) > 0
    # No line joins the position of frame 5 with the position of frame 20.
    for x in (-23.0, -20.0, -17.5, -15.0, -12.0):
        assert _difference(image, empty, *_radar_pixel(style, x, 15.0)) == 0
    # The positions on both sides of the gap are still remembered.
    assert len(renderer.trail(5)) == 12


def test_trail_joins_the_frames_that_were_drawn_one_after_the_other() -> None:
    fps = 10.0
    style = RenderStyle(trail_seconds=4.0)
    empty = _empty_radar(fps, style)
    renderer = RadarRenderer(fps, style)
    image = empty
    # Only every fifth frame of the clip is drawn. The track has a state in each of them.
    for frame in range(0, 26, 5):
        player = _player(track_id=5, pitch_x=-30.0 + frame, pitch_y=15.0, team_side="left")
        image = renderer.draw(FrameState(frame=frame, players=(player,)))

    for x in (-27.5, -22.5, -17.5, -12.5):
        assert _difference(image, empty, *_radar_pixel(style, x, 15.0)) > 0


def test_trail_disappears_after_the_trail_time() -> None:
    fps = 10.0
    style = RenderStyle(trail_seconds=1.0)
    empty = _empty_radar(fps, style)
    renderer = RadarRenderer(fps, style)
    image = empty
    for frame in range(26):
        player = _player(track_id=5, pitch_x=-30.0 + 2.0 * frame, pitch_y=15.0, team_side="left")
        image = renderer.draw(FrameState(frame=frame, players=(player,)))

    # At frame 25 the trail holds the frames 15..25, so it starts at x = 0.
    _, columns = np.nonzero(_changed(image, empty))
    start_x, _ = _radar_pixel(style, 0.0, 15.0)
    assert columns.min() >= start_x - 5
    old_x, old_y = _radar_pixel(style, -29.0, 15.0)
    assert _difference(image, empty, old_x, old_y) == 0


def test_pitch_markings_stay_on_top_of_a_trail() -> None:
    fps = 10.0
    style = RenderStyle(trail_seconds=1.0)
    empty = _empty_radar(fps, style)
    renderer = RadarRenderer(fps, style)
    image = empty
    # At frame 17 the player is at x = 4 m and the trail crosses the halfway line at x = 0.
    for frame in range(18):
        player = _player(track_id=5, pitch_x=-30.0 + 2.0 * frame, pitch_y=15.0, team_side="left")
        image = renderer.draw(FrameState(frame=frame, players=(player,)))

    line_x, line_y = _radar_pixel(style, 0.0, 15.0)
    grass = _bgr(empty, 0, 0)
    assert _bgr(empty, line_x, line_y) != grass  # the halfway line is here
    assert _difference(image, empty, line_x, line_y) == 0
    trail_x, trail_y = _radar_pixel(style, -1.0, 15.0)
    assert _bgr(empty, trail_x, trail_y) == grass
    assert _difference(image, empty, trail_x, trail_y) > 0


def test_zero_trail_time_draws_dots_only() -> None:
    style = RenderStyle(trail_seconds=0.0, radar_scale=4.0)
    renderer = RadarRenderer(25.0, style)
    image = _empty_radar(25.0, style)
    for frame in range(3):
        player = _player(track_id=1, pitch_x=10.0 * frame, pitch_y=0.0)
        image = renderer.draw(FrameState(frame=frame, players=(player,)))

    dot_only = RadarRenderer(25.0, style).draw(
        FrameState(frame=2, players=(_player(track_id=1, pitch_x=20.0, pitch_y=0.0),))
    )
    assert len(renderer.trail(1)) == 1
    assert np.array_equal(image, dot_only)


def test_radar_gives_the_same_image_when_a_frame_is_drawn_again() -> None:
    renderer = RadarRenderer(10.0, RenderStyle(trail_seconds=1.0, radar_scale=4.0))
    states = [
        FrameState(frame=frame, players=(_player(track_id=1, pitch_x=4.0 * frame, pitch_y=0.0),))
        for frame in range(4)
    ]
    images = [renderer.draw(state) for state in states]
    assert len(renderer.trail(1)) == 4

    again = renderer.draw(states[3])
    assert np.array_equal(again, images[3])
    assert len(renderer.trail(1)) == 4

    # A step back to frame 1 forgets the positions of the frames 1, 2 and 3.
    back = renderer.draw(states[1])
    assert np.array_equal(back, images[1])
    assert np.allclose(renderer.trail(1), [[0.0, 0.0], [4.0, 0.0]])


def test_radar_rejects_a_track_id_that_appears_twice() -> None:
    twins = (_player(track_id=4), _player(track_id=4, pitch_x=5.0))
    with pytest.raises(ValueError):
        RadarRenderer(25.0, RenderStyle(radar_scale=2.0)).draw(FrameState(frame=0, players=twins))


def test_radar_rejects_a_wrong_frame_rate() -> None:
    with pytest.raises(ValueError):
        RadarRenderer(0.0)


# --- Composition ---------------------------------------------------------------


def test_compose_stacks_the_radar_below_the_overlay_at_the_same_width() -> None:
    overlay = _noise_image(90, 160, seed=3)
    radar = np.full((76, 113, 3), 200, dtype=np.uint8)

    out = compose(overlay, radar)

    # The radar keeps its aspect ratio: 76 * 160 / 113 = 107.6 rows.
    assert out.shape == (90 + 108, 160, 3)
    assert out.dtype == np.uint8
    assert np.array_equal(out[:90], overlay)
    assert np.abs(out[90:].astype(int) - 200).max() <= 1


def test_compose_shrinks_a_wide_radar() -> None:
    overlay = np.zeros((90, 160, 3), dtype=np.uint8)
    radar = _empty_radar(25.0, RenderStyle())  # 1130 x 760 pixels at 10 pixels per metre
    out = compose(overlay, radar)
    assert out.shape == (90 + round(760 * 160 / 1130), 160, 3)
    assert out[90:].any()


def test_compose_keeps_a_radar_of_the_same_width() -> None:
    overlay = _noise_image(40, 64, seed=4)
    radar = _noise_image(30, 64, seed=5)
    out = compose(overlay, radar)
    assert out.shape == (70, 64, 3)
    assert np.array_equal(out[:40], overlay)
    assert np.array_equal(out[40:], radar)


def test_compose_rejects_a_wrong_image() -> None:
    overlay = np.zeros((40, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        compose(overlay, np.zeros((30, 64), dtype=np.uint8))
    with pytest.raises(ValueError):
        compose(np.zeros((40, 64), dtype=np.uint8), overlay)


# --- Video sink ----------------------------------------------------------------


def test_video_sink_opens_an_mp4v_writer_with_width_and_height(
    fake_writers: list[_FakeWriter], tmp_path: Path
) -> None:
    path = tmp_path / "clip.mp4"
    with VideoSink(path, 25.0, (64, 48)) as sink:
        sink.write(np.zeros((48, 64, 3), dtype=np.uint8))  # 48 rows and 64 columns
        sink.write(np.zeros((48, 64, 3), dtype=np.uint8))
        assert fake_writers[0].isOpened()

    writer = fake_writers[0]
    assert len(fake_writers) == 1
    assert writer.filename == str(path)
    assert writer.fourcc == cv2.VideoWriter_fourcc(*"mp4v")
    assert writer.fps == 25.0
    assert writer.frame_size == (64, 48)
    assert len(writer.frames) == 2
    assert not writer.isOpened()  # the sink closes the writer at the end of the block


def test_video_sink_rejects_a_wrong_image_size(fake_writers: list[_FakeWriter], tmp_path: Path) -> None:
    with VideoSink(tmp_path / "clip.mp4", 25.0, (64, 48)) as sink:
        with pytest.raises(ValueError):
            sink.write(np.zeros((64, 48, 3), dtype=np.uint8))  # width and height are swapped
        with pytest.raises(ValueError):
            sink.write(np.zeros((48, 60, 3), dtype=np.uint8))
        with pytest.raises(ValueError):
            sink.write(np.zeros((48, 64), dtype=np.uint8))
    assert fake_writers[0].frames == []


def test_video_sink_rejects_a_frame_after_close(fake_writers: list[_FakeWriter], tmp_path: Path) -> None:
    sink = VideoSink(tmp_path / "clip.mp4", 25.0, (64, 48))
    sink.close()
    with pytest.raises(OSError):
        sink.write(np.zeros((48, 64, 3), dtype=np.uint8))


def test_video_sink_closes_the_writer_after_an_error(fake_writers: list[_FakeWriter], tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        with VideoSink(tmp_path / "clip.mp4", 25.0, (64, 48)):
            raise RuntimeError("stop")
    assert not fake_writers[0].isOpened()


@pytest.mark.parametrize(("fps", "size"), [(0.0, (64, 48)), (25.0, (0, 48)), (25.0, (64, -2))])
def test_video_sink_rejects_wrong_settings(
    fake_writers: list[_FakeWriter], tmp_path: Path, fps: float, size: tuple[int, int]
) -> None:
    with pytest.raises(ValueError):
        VideoSink(tmp_path / "clip.mp4", fps, size)
    assert fake_writers == []


def test_video_sink_that_cannot_open_raises_os_error(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        VideoSink(tmp_path / "no_such_folder" / "clip.mp4", 25.0, (64, 48))


def test_video_sink_writes_a_file_that_opens_again(tmp_path: Path) -> None:
    if not _mp4v_available(tmp_path):
        pytest.skip("the installed OpenCV build cannot write the codec mp4v")
    path = tmp_path / "clip.mp4"

    with VideoSink(path, 25.0, (64, 48)) as sink:
        for index in range(5):
            sink.write(np.full((48, 64, 3), 40 * index, dtype=np.uint8))

    frames = _read_video(path)
    assert len(frames) == 5
    assert all(frame.shape == (48, 64, 3) for frame in frames)
    # The frames come back in the order of writing: each one is brighter than the one before.
    brightness = [float(frame.mean()) for frame in frames]
    assert all(darker < brighter for darker, brighter in pairwise(brightness))


# --- Whole clip ----------------------------------------------------------------


def test_render_clip_writes_one_composed_image_per_frame(fake_writers: list[_FakeWriter], tmp_path: Path) -> None:
    source, result = _clip(n_frames=4, height=60, width=96, seed=7)
    originals = [source.read(frame).copy() for frame in range(4)]
    style = RenderStyle(radar_scale=2.0)
    out_path = tmp_path / "videos" / "clip.mp4"

    returned = render_clip(source, result, out_path, style)

    assert returned == out_path
    assert out_path.parent.is_dir()
    writer = fake_writers[0]
    # The writer gets another name in the same folder. The finished file then gets the wanted name.
    assert writer.filename == str(out_path.with_name("clip.partial.mp4"))
    assert [entry.name for entry in out_path.parent.iterdir()] == ["clip.mp4"]
    assert writer.fps == source.info.fps
    assert len(writer.frames) == 4

    # The first image is the overlay on top of the radar, cut to even sides.
    first_state = result.frames[0]
    overlay = draw_overlay(source.read(0), first_state.players, style)
    stacked = compose(overlay, RadarRenderer(source.info.fps, style).draw(first_state))
    height, width = stacked.shape[:2]
    even_height, even_width = height - height % 2, width - width % 2
    assert width == 96
    assert height > 60
    assert writer.frame_size == (even_width, even_height)
    assert all(image.shape == (even_height, even_width, 3) for image in writer.frames)
    assert np.array_equal(writer.frames[0], stacked[:even_height, :even_width])
    # The video part is on top: the box of the player is drawn there.
    assert _bgr(writer.frames[0], x=10, y=30) == style.left_bgr

    for frame, original in enumerate(originals):
        assert np.array_equal(source.read(frame), original)


class _BrokenSource:
    """A frame source that fails when the frame ``fails_at`` is read."""

    def __init__(self, source: FrameSource, fails_at: int) -> None:
        self.info = source.info
        self._source = source
        self._fails_at = fails_at

    def read(self, frame: int) -> np.ndarray:
        if frame == self._fails_at:
            raise OSError(f"frame {frame} cannot be read")
        return self._source.read(frame)


def test_render_clip_that_fails_leaves_an_older_video_as_it_was(
    fake_writers: list[_FakeWriter], tmp_path: Path
) -> None:
    """A file with the name of the video is always a complete video: an unfinished one never gets that name."""
    source, result = _clip(n_frames=4, height=60, width=96, seed=7)
    out_path = tmp_path / "clip.mp4"
    out_path.write_bytes(b"an older, complete video")

    with pytest.raises(OSError, match="frame 2 cannot be read"):
        render_clip(_BrokenSource(source, fails_at=2), result, out_path, RenderStyle(radar_scale=2.0))

    assert len(fake_writers[0].frames) == 2
    assert out_path.read_bytes() == b"an older, complete video"
    assert [entry.name for entry in tmp_path.iterdir()] == ["clip.mp4"]

    # A complete write replaces the older video.
    render_clip(source, result, out_path, RenderStyle(radar_scale=2.0))
    assert out_path.read_bytes() == b""
    assert [entry.name for entry in tmp_path.iterdir()] == ["clip.mp4"]


def test_render_clip_rejects_a_result_that_does_not_cover_the_clip(
    fake_writers: list[_FakeWriter], tmp_path: Path
) -> None:
    source, result = _clip(n_frames=4, height=60, width=96, seed=8)
    too_short = result.model_copy(update={"frames": result.frames[:3]})
    repeated = result.model_copy(update={"frames": result.frames[:3] + result.frames[:1]})

    for wrong in (too_short, repeated):
        with pytest.raises(ValueError):
            render_clip(source, wrong, tmp_path / "clip.mp4", RenderStyle(radar_scale=2.0))
    assert fake_writers == []


def test_render_clip_writes_a_video_file_that_opens_again(tmp_path: Path) -> None:
    if not _mp4v_available(tmp_path):
        pytest.skip("the installed OpenCV build cannot write the codec mp4v")
    source, result = _clip(n_frames=5, height=64, width=96, seed=9)
    out_path = tmp_path / "clip.mp4"

    returned = render_clip(source, result, out_path, RenderStyle(radar_scale=2.0))

    assert returned == out_path
    frames = _read_video(out_path)
    assert len(frames) == source.info.n_frames
    height, width = frames[0].shape[:2]
    assert width == 96
    assert height > 64
    assert height % 2 == 0
