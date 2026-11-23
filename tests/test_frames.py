"""Tests for the keyframe rule and the frame sources of ``pitchvlm.frames``.

Conventions under test: frame indices start at 0, so the SoccerNet file
``000001`` is frame 0. Frame files are ordered by their numbers. A frame index
outside the clip is an error, also a negative one. An image has shape
(height, width, 3), BGR, uint8, and every read gives a new array.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest

from pitchvlm.frames import (
    ImageFolderSource,
    VideoFileSource,
    check_frame_times,
    keyframe_indices,
    read_video_record,
    write_video_frames,
)
from pitchvlm.schemas import ClipInfo

#: Size of the test images. Height and width differ so that a swap is found.
HEIGHT, WIDTH = 48, 64
#: Size of the test video frames, (height, width).
VIDEO_SIZE = (96, 128)
#: Brightness of the frames of the long test video. The video has more frames than a video source
#: keeps in memory, and the levels are far enough apart to tell the frames apart after a lossy codec.
VIDEO_LEVELS = tuple(range(20, 237, 24))


def random_image(seed: int) -> np.ndarray:
    """Return a seeded random BGR image of the test size."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an image file with OpenCV and stop the test when the write fails."""
    assert cv2.imwrite(str(path), image), f"cannot write {path.name}"


def make_folder(tmp_path: Path, count: int = 3) -> tuple[Path, list[np.ndarray]]:
    """Write ``count`` numbered PNG frames and return the folder and the images in frame order."""
    folder = tmp_path / "img1"
    folder.mkdir()
    images = [random_image(seed) for seed in range(count)]
    # The files are written last frame first: the source must sort by number, not by age.
    for index in reversed(range(count)):
        write_image(folder / f"{index + 1:06d}.png", images[index])
    return folder, images


def write_video(path: Path, frames: Sequence[np.ndarray], fps: float) -> None:
    """Write an mp4v video. The test is skipped when this OpenCV build cannot write one."""
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write an mp4v video")
    for frame in frames:
        writer.write(frame)
    writer.release()


def write_level_video(folder: Path, levels: Sequence[int]) -> Path:
    """Write a video whose frames are flat grey images with the given brightness levels, in this order."""
    height, width = VIDEO_SIZE
    path = folder / "demo_clip.mp4"
    write_video(path, [np.full((height, width, 3), level, dtype=np.uint8) for level in levels], fps=25.0)
    return path


def level_of(image: np.ndarray, levels: Sequence[int]) -> int:
    """Return the level that is nearest to the mean brightness of a decoded frame. The codec is lossy."""
    mean = float(image.mean())
    return min(levels, key=lambda level: abs(level - mean))


# --- Keyframes ---------------------------------------------------------------------------


def test_keyframes_of_a_30_second_clip_at_25_fps() -> None:
    """750 frames at 25 fps give 64 keyframes: every 12th frame from 0, plus the last frame."""
    indices = keyframe_indices(750, 25.0)
    assert len(indices) == 64
    assert indices[0] == 0
    assert indices[-1] == 749
    assert indices[1] - indices[0] == 12
    assert indices[:-1] == list(range(0, 750, 12))
    assert indices == keyframe_indices(750, 25.0, interval_s=0.48)


@pytest.mark.parametrize("n_frames", [1, 2, 11, 12, 13, 14, 100, 750, 751])
def test_keyframes_are_increasing_and_inside_the_clip(n_frames: int) -> None:
    """Keyframes are unique, sorted, start at 0, end at the last frame and are at most 12 apart."""
    indices = keyframe_indices(n_frames, 25.0)
    assert all(isinstance(index, int) for index in indices)
    assert indices == sorted(set(indices))
    assert indices[0] == 0
    assert indices[-1] == n_frames - 1
    assert all(later - earlier <= 12 for earlier, later in zip(indices, indices[1:]))


@pytest.mark.parametrize(
    ("n_frames", "expected"),
    [
        pytest.param(1, [0], id="one frame"),
        pytest.param(2, [0, 1], id="two frames"),
        pytest.param(5, [0, 4], id="shorter than one step"),
        pytest.param(12, [0, 11], id="one frame short of a step"),
        pytest.param(13, [0, 12], id="last frame on the grid"),
        pytest.param(14, [0, 12, 13], id="last frame after the grid"),
        pytest.param(25, [0, 12, 24], id="one second"),
    ],
)
def test_keyframes_of_short_clips(n_frames: int, expected: list[int]) -> None:
    """A short clip gets frame 0, the grid frames and the last frame, with no frame twice."""
    assert keyframe_indices(n_frames, 25.0) == expected


def test_keyframe_step_follows_the_frame_rate_and_the_interval() -> None:
    """The step is the interval in frames: 24 at 50 fps, and 25 for 1 s at 25 fps."""
    assert keyframe_indices(100, 50.0) == [0, 24, 48, 72, 96, 99]
    assert keyframe_indices(100, 25.0, interval_s=1.0) == [0, 25, 50, 75, 99]


def test_keyframe_step_is_at_least_one_frame() -> None:
    """An interval shorter than one frame gives every frame once."""
    assert keyframe_indices(10, 25.0, interval_s=0.01) == list(range(10))


@pytest.mark.parametrize("n_frames", [0, -5])
def test_keyframes_need_a_positive_frame_count(n_frames: int) -> None:
    """A clip without frames is an error."""
    with pytest.raises(ValueError, match="n_frames"):
        keyframe_indices(n_frames, 25.0)


@pytest.mark.parametrize("fps", [0.0, -25.0, math.nan, math.inf], ids=["zero", "negative", "nan", "inf"])
def test_keyframes_need_a_positive_frame_rate(fps: float) -> None:
    """A frame rate that is not above zero is an error. It must not make every frame a keyframe."""
    with pytest.raises(ValueError, match="fps"):
        keyframe_indices(750, fps)


@pytest.mark.parametrize("interval_s", [0.0, -0.48, math.nan, math.inf], ids=["zero", "negative", "nan", "inf"])
def test_keyframes_need_a_positive_interval(interval_s: float) -> None:
    """An interval that is not above zero is an error. It must not make every frame a keyframe."""
    with pytest.raises(ValueError, match="interval_s"):
        keyframe_indices(750, 25.0, interval_s)


# --- Image folder ----------------------------------------------------------------------------


def test_image_folder_source_reports_the_clip(tmp_path: Path) -> None:
    """The source reports the clip id, the frame count, the frame rate and the image size."""
    folder, _ = make_folder(tmp_path)
    source = ImageFolderSource(folder, clip_id="SNGS-021")
    assert source.info == ClipInfo(clip_id="SNGS-021", n_frames=3, fps=25.0, width=WIDTH, height=HEIGHT)
    assert (source.info.width, source.info.height) == (64, 48)


def test_image_folder_source_uses_the_given_frame_rate(tmp_path: Path) -> None:
    """The frame rate comes from the caller, because image files do not hold one."""
    folder, _ = make_folder(tmp_path)
    assert ImageFolderSource(folder, clip_id="clip", fps=50.0).info.fps == 50.0


def test_frame_zero_is_the_first_numbered_file(tmp_path: Path) -> None:
    """Frame 0 is file 000001, and every frame comes back as the BGR image that was written."""
    folder, images = make_folder(tmp_path)
    source = ImageFolderSource(folder, clip_id="clip")
    for frame, expected in enumerate(images):
        image = source.read(frame)
        assert image.shape == (HEIGHT, WIDTH, 3)
        assert image.dtype == np.uint8
        np.testing.assert_array_equal(image, expected)


def test_files_that_are_not_images_are_ignored(tmp_path: Path) -> None:
    """Only image files count as frames."""
    folder, _ = make_folder(tmp_path, count=2)
    (folder / "notes.txt").write_text("not a frame", encoding="utf-8")
    (folder / "crops").mkdir()
    assert ImageFolderSource(folder, clip_id="clip").info.n_frames == 2


def test_jpeg_and_upper_case_suffixes_count_as_frames(tmp_path: Path) -> None:
    """The suffixes .jpg, .jpeg and .png are frames, in lower or upper case."""
    folder = tmp_path / "img1"
    folder.mkdir()
    image = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)
    for name in ("000001.jpg", "000002.jpeg", "000003.PNG"):
        write_image(folder / name, image)
    source = ImageFolderSource(folder, clip_id="clip")
    assert source.info.n_frames == 3
    for frame in range(3):
        assert source.read(frame).shape == (HEIGHT, WIDTH, 3)


def test_grey_image_is_read_with_three_channels(tmp_path: Path) -> None:
    """A one-channel file still gives an image of shape (height, width, 3)."""
    folder = tmp_path / "img1"
    folder.mkdir()
    grey = np.random.default_rng(7).integers(0, 256, size=(HEIGHT, WIDTH), dtype=np.uint8)
    write_image(folder / "000001.png", grey)
    image = ImageFolderSource(folder, clip_id="clip").read(0)
    assert image.shape == (HEIGHT, WIDTH, 3)
    assert image.dtype == np.uint8
    for channel in range(3):
        np.testing.assert_array_equal(image[:, :, channel], grey)


def test_empty_folder_is_an_error(tmp_path: Path) -> None:
    """A folder without files is an error."""
    folder = tmp_path / "empty"
    folder.mkdir()
    with pytest.raises(FileNotFoundError, match="no frame images"):
        ImageFolderSource(folder, clip_id="clip")


def test_folder_without_images_is_an_error(tmp_path: Path) -> None:
    """A folder that holds only other files is an error."""
    folder = tmp_path / "text-only"
    folder.mkdir()
    (folder / "notes.txt").write_text("not a frame", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no frame images"):
        ImageFolderSource(folder, clip_id="clip")


def test_missing_folder_is_an_error(tmp_path: Path) -> None:
    """A folder that does not exist is an error."""
    with pytest.raises(FileNotFoundError):
        ImageFolderSource(tmp_path / "missing", clip_id="clip")


def test_unreadable_image_is_an_error(tmp_path: Path) -> None:
    """A frame file that is not a picture is an error."""
    folder = tmp_path / "img1"
    folder.mkdir()
    (folder / "000001.png").write_bytes(b"this is not a picture")
    with pytest.raises(OSError, match="cannot read image"):
        ImageFolderSource(folder, clip_id="clip")


def test_reading_after_the_last_frame_is_an_error(tmp_path: Path) -> None:
    """A frame index equal to the frame count is out of range."""
    folder, _ = make_folder(tmp_path)
    source = ImageFolderSource(folder, clip_id="clip")
    with pytest.raises(IndexError):
        source.read(source.info.n_frames)


def test_reading_before_the_first_frame_is_an_error(tmp_path: Path) -> None:
    """A negative frame index does not count from the end of the clip."""
    folder, _ = make_folder(tmp_path)
    source = ImageFolderSource(folder, clip_id="clip")
    for frame in (-1, -source.info.n_frames):
        with pytest.raises(IndexError):
            source.read(frame)


def test_image_folder_read_gives_a_new_image_on_every_call(tmp_path: Path) -> None:
    """A caller may draw on a frame: the next read of that frame still gives the picture of the file."""
    folder, images = make_folder(tmp_path)
    source = ImageFolderSource(folder, clip_id="clip")
    first = source.read(1)
    first[:] = 0
    np.testing.assert_array_equal(source.read(1), images[1])


def test_frame_files_are_ordered_by_number_and_not_by_text(tmp_path: Path) -> None:
    """Names without leading zeros are read in the order of their numbers: 2 comes before 10."""
    folder = tmp_path / "frames"
    folder.mkdir()
    numbers = (1, 2, 9, 10, 11, 100)
    images = {number: random_image(number) for number in numbers}
    for number in reversed(numbers):
        write_image(folder / f"{number}.png", images[number])
    source = ImageFolderSource(folder, clip_id="clip")
    assert source.info.n_frames == len(numbers)
    for frame, number in enumerate(numbers):
        np.testing.assert_array_equal(source.read(frame), images[number])


def test_hidden_files_are_not_frames(tmp_path: Path) -> None:
    """A hidden file is skipped, for example the ``._000001.png`` that macOS writes next to a file on some disks."""
    folder, images = make_folder(tmp_path)
    (folder / "._000001.png").write_bytes(b"this is not a picture")
    (folder / ".000000.png").write_bytes(b"this is not a picture")
    source = ImageFolderSource(folder, clip_id="clip")
    assert source.info.n_frames == len(images)
    for frame, expected in enumerate(images):
        np.testing.assert_array_equal(source.read(frame), expected)


@pytest.mark.parametrize("name", ["frame_000004.png", "000004a.png", "4.5.png", "four.png"])
def test_frame_file_name_that_is_not_a_number_is_an_error(tmp_path: Path, name: str) -> None:
    """The frame order comes from the numbers, so an image file whose name is not a number is an error."""
    folder, _ = make_folder(tmp_path)
    write_image(folder / name, random_image(99))
    with pytest.raises(ValueError, match="must be a number"):
        ImageFolderSource(folder, clip_id="clip")


@pytest.mark.parametrize("name", ["1.png", "000002.jpg"])
def test_two_frame_files_with_the_same_number_are_an_error(tmp_path: Path, name: str) -> None:
    """Two files with the same number, like ``1.png`` and ``000001.png``, leave the frame order open."""
    folder, _ = make_folder(tmp_path)
    write_image(folder / name, random_image(99))
    with pytest.raises(ValueError, match="same number"):
        ImageFolderSource(folder, clip_id="clip")


# --- Video file ------------------------------------------------------------------------------


def test_missing_video_is_an_error(tmp_path: Path) -> None:
    """A video file that does not exist cannot be opened."""
    with pytest.raises(OSError, match="cannot open video"):
        VideoFileSource(tmp_path / "missing.mp4")


def test_video_file_source_decodes_every_frame_in_order(tmp_path: Path) -> None:
    """A written video comes back with its frame count, frame rate, size and frame order."""
    levels = (30, 70, 110, 150, 190, 230)
    height, width = 96, 128
    frames = [np.full((height, width, 3), level, dtype=np.uint8) for level in levels]
    path = tmp_path / "demo_clip.mp4"
    write_video(path, frames, fps=25.0)
    source = VideoFileSource(path)
    assert source.info.clip_id == "demo_clip"
    assert source.info.n_frames == len(levels)
    assert source.info.fps == pytest.approx(25.0, abs=0.01)
    assert (source.info.width, source.info.height) == (width, height)
    for frame, level in enumerate(levels):
        image = source.read(frame)
        assert image.shape == (height, width, 3)
        assert image.dtype == np.uint8
        # The codec is lossy, so only the mean brightness of the frame is compared.
        assert abs(float(image.mean()) - level) < 12.0
    assert VideoFileSource(path, clip_id="custom-id").info.clip_id == "custom-id"


def test_video_frames_can_be_read_in_any_order(tmp_path: Path) -> None:
    """Every frame index gives its own frame, whatever was read before.

    The source decodes forward and keeps only a few frames. A step back to a frame that it no
    longer keeps starts the decoder again, and that must give the same frame as the first read.
    """
    source = VideoFileSource(write_level_video(tmp_path, VIDEO_LEVELS))
    n_frames = len(VIDEO_LEVELS)
    assert source.info.n_frames == n_frames
    last = n_frames - 1
    order = [last, 0, 4, 4, 3, last, 1, *range(n_frames), 0, last, 2]
    for frame in order:
        image = source.read(frame)
        assert image.shape == (*VIDEO_SIZE, 3)
        assert level_of(image, VIDEO_LEVELS) == VIDEO_LEVELS[frame], f"frame {frame}"


def test_video_read_gives_a_new_image_on_every_call(tmp_path: Path) -> None:
    """A caller may draw on a frame: the next read of that frame still gives the picture of the file."""
    source = VideoFileSource(write_level_video(tmp_path, VIDEO_LEVELS))
    first = source.read(3)
    first[:] = 0
    again = source.read(3)
    assert not np.shares_memory(again, first)
    assert level_of(again, VIDEO_LEVELS) == VIDEO_LEVELS[3]


def test_video_read_outside_the_clip_is_an_error(tmp_path: Path) -> None:
    """A negative frame index does not count from the end, and an index after the last frame is an error."""
    levels = VIDEO_LEVELS[:3]
    source = VideoFileSource(write_level_video(tmp_path, levels))
    assert source.info.n_frames == 3
    for frame in (-1, -3, 3, 100):
        with pytest.raises(IndexError):
            source.read(frame)
    assert level_of(source.read(2), levels) == levels[2]


# --- Checks of a video, and its frames written as images ---------------------------------


def write_mjpeg_avi(path: Path, frames: Sequence[np.ndarray], fps: float) -> None:
    """Write a Motion JPEG AVI video. The test is skipped when this OpenCV build cannot write one."""
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write a Motion JPEG video")
    for frame in frames:
        writer.write(frame)
    writer.release()


def _mp4_box_path(data: bytes, start: int, end: int, wanted: bytes, path: list[int]) -> list[int] | None:
    """Return the offsets of the boxes from the top down to the first box of type ``wanted``, or None."""
    containers = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}
    offset = start
    while offset < end:
        size, kind = struct.unpack(">I4s", data[offset : offset + 8])
        if kind == wanted:
            return [*path, offset]
        if kind in containers:
            found = _mp4_box_path(data, offset + 8, offset + size, wanted, [*path, offset])
            if found is not None:
                return found
        offset += size
    return None


def _rewrite_frame_times(video: Path, out: Path, runs: Callable[[int, int], list[tuple[int, int]]]) -> None:
    """Copy an mp4v video of OpenCV with other frame lengths: ``runs(n_samples, delta)`` gives the new entries.

    The time-to-sample box ``stts`` of the copy has two entries in the place of one, each a
    count of frames and their length. The boxes that hold it grow by 8 bytes. OpenCV writes
    the box ``moov`` after the media data, so no offset of a sample moves.
    """
    data = bytearray(video.read_bytes())
    path = _mp4_box_path(bytes(data), 0, len(data), b"stts", [])
    assert path is not None, "the video has no time-to-sample box"
    stts = path[-1]
    size = struct.unpack(">I", data[stts : stts + 4])[0]
    entries, n_samples, delta = struct.unpack(">III", data[stts + 12 : stts + 24])
    assert (size, entries) == (24, 1), "the test expects one entry, as OpenCV writes it"
    (count_1, delta_1), (count_2, delta_2) = runs(n_samples, delta)
    data[stts : stts + size] = struct.pack(">I4sIIIIII", 32, b"stts", 0, 2, count_1, delta_1, count_2, delta_2)
    for parent in path[:-1]:
        grown = struct.unpack(">I", data[parent : parent + 4])[0] + 32 - size
        data[parent : parent + 4] = struct.pack(">I", grown)
    out.write_bytes(bytes(data))


def make_variable_rate(video: Path, out: Path) -> None:
    """Copy an mp4v video of OpenCV and give its second half frames that last twice as long."""
    _rewrite_frame_times(video, out, lambda n, delta: [(n // 2, delta), (n - n // 2, 2 * delta)])


def make_short_last_frame(video: Path, out: Path) -> None:
    """Copy an mp4v video of OpenCV whose last frame lasts 1/40 of a frame. No frame time moves."""
    _rewrite_frame_times(video, out, lambda n, delta: [(n - 1, delta), (1, delta // 40)])


def textured_frames(count: int) -> list[np.ndarray]:
    """Return frames with random pixels, each one different: a lossy codec keeps them apart."""
    rng = np.random.default_rng(5)
    return [rng.integers(0, 256, size=(*VIDEO_SIZE, 3), dtype=np.uint8) for _ in range(count)]


def test_frame_times_of_a_constant_rate_pass_also_with_a_late_first_frame() -> None:
    check_frame_times(Path("v.mp4"), [0.0, 40.0, 80.0, 120.0], fps=25.0, header_frames=4.0)
    check_frame_times(Path("v.mp4"), [500.0, 540.0, 580.4, 619.7], fps=25.0, header_frames=4.0)


@pytest.mark.parametrize(
    ("times_ms", "header_frames", "words"),
    [
        ([0.0, 40.0, 80.0], 0.0, "names no frame count"),
        ([0.0, 40.0, 80.0], -1.0, "names no frame count"),
        ([0.0, 40.0, 80.0], 4.0, "its header names 4 frames, but 3 frames can be decoded"),
        ([0.0, 40.0, 80.0, 120.0], 3.0, "its header names 3 frames, but 4 frames can be decoded"),
        ([0.0, 40.0, 120.0, 160.0], 4.0, "frame 2 is shown at 120.0 ms"),
        ([0.0, 40.0, 40.0, 80.0], 4.0, "frame 2 is shown at 40.0 ms"),
        ([0.0, 40.0, 80.0, 160.0, 240.0], 5.0, "variable frame rate"),
    ],
)
def test_frame_times_that_are_not_sure_are_refused(
    times_ms: list[float], header_frames: float, words: str
) -> None:
    with pytest.raises(OSError, match=words):
        check_frame_times(Path("v.mp4"), times_ms, fps=25.0, header_frames=header_frames)


def test_video_with_a_variable_frame_rate_is_refused(tmp_path: Path) -> None:
    """OpenCV decodes such a video with frames left out: the source refuses it with the line about the rate."""
    constant = tmp_path / "constant.mp4"
    write_video(constant, textured_frames(10), fps=25.0)
    assert VideoFileSource(constant).info.n_frames == 10
    variable = tmp_path / "variable.mp4"
    make_variable_rate(constant, variable)
    with pytest.raises(OSError, match="variable frame rate"):
        VideoFileSource(variable)


def test_video_with_a_short_last_frame_keeps_its_rate(tmp_path: Path) -> None:
    """A good video: every frame 40 ms after the one before, only the last frame lasts 1 ms.

    OpenCV gives the rate of the header as the frame count over the length of the stream, here
    25.03 frames per second, which put frame 384 more than half a frame away from its time
    (second read of 2026-11-22). The rate of the frames comes from their times.
    """
    whole = tmp_path / "whole.mp4"
    write_video(whole, textured_frames(750), fps=25.0)
    cut = tmp_path / "cut.mp4"
    make_short_last_frame(whole, cut)
    capture = cv2.VideoCapture(str(cut))
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(25.03, abs=0.01)
    capture.release()
    info = VideoFileSource(cut).info
    assert (info.n_frames, info.fps) == (750, 25.0)


@pytest.mark.parametrize("rate", [Fraction(25), Fraction(30000, 1001), Fraction(60000, 1001), Fraction(24000, 1001)])
def test_frame_times_give_the_rate_of_the_frames_whatever_the_header_rate(rate: Fraction) -> None:
    """Times rounded to the millisecond, as a container with a millisecond time base holds them."""
    times = [float(round(index * 1000 / rate)) for index in range(750)]
    for header_fps in (float(rate), float(rate) * 1.0013, round(float(rate), 2)):
        measured = check_frame_times(Path("v.mkv"), times, fps=header_fps, header_frames=750.0)
        assert measured == pytest.approx(float(rate), rel=1e-4)


def test_a_frame_missing_in_the_middle_of_a_long_clip_is_refused() -> None:
    """The mean step then grows by one frame over the clip only, but the step into the gap is two frames."""
    times = [40.0 * index for index in range(751) if index != 375]
    with pytest.raises(OSError, match="frame 375 is shown at 15040.0 ms, 80.0 ms after the frame before it"):
        check_frame_times(Path("v.mp4"), times, fps=25.0, header_frames=750.0)


def test_a_step_half_the_usual_step_away_is_refused() -> None:
    """60 % of the steps 80 ms, 40 % 40 ms: a variable rate whose short steps are exactly half a step away."""
    steps = [80.0, 80.0, 40.0, 80.0, 40.0] * 20
    times = [sum(steps[:index]) for index in range(len(steps) + 1)]
    with pytest.raises(OSError, match="frame 3 is shown at 200.0 ms, 40.0 ms after the frame before it"):
        check_frame_times(Path("v.mp4"), times, fps=15.625, header_frames=float(len(times)))


def test_video_with_a_frame_that_cannot_be_read_is_refused(tmp_path: Path) -> None:
    """The end of the file is cut off: the header names more frames than can be decoded.

    The cut takes about one and a half frames. A cut of a few bytes only is not found: the
    decoder of OpenCV gives the damaged last frame without an error.
    """
    whole = tmp_path / "whole.avi"
    write_mjpeg_avi(whole, textured_frames(10), fps=25.0)
    assert VideoFileSource(whole).info.n_frames == 10
    data = whole.read_bytes()
    cut = tmp_path / "cut.avi"
    cut.write_bytes(data[: len(data) - 15 * len(data) // 100])
    with pytest.raises(OSError, match="its header names 10 frames, but [89] frames can be decoded"):
        VideoFileSource(cut)


def test_written_frames_are_the_frames_of_the_video_source(tmp_path: Path) -> None:
    """The images that a tool reads are the frames that ``run`` reads: same count, same order, same pixels.

    The video has more frames than the source keeps in memory, so the images come from the
    decoder that runs forward, and the second read of the video starts it again.
    """
    video = tmp_path / "match_07.mp4"
    write_video(video, textured_frames(20), fps=25.0)
    source = VideoFileSource(video)
    record = write_video_frames(source, "a" * 64, tmp_path / "data" / "match_07")
    clip_folder = tmp_path / "data" / "match_07"
    assert sorted(path.name for path in clip_folder.iterdir()) == ["img1", "video.json"]
    assert [path.name for path in sorted((clip_folder / "img1").iterdir())] == [
        f"{frame + 1:06d}.png" for frame in range(20)
    ]
    assert read_video_record(clip_folder) == record
    assert (record.video, record.video_sha256, record.n_frames) == ("match_07.mp4", "a" * 64, 20)
    assert (record.width, record.height, record.fps) == (VIDEO_SIZE[1], VIDEO_SIZE[0], source.info.fps)
    images = ImageFolderSource(clip_folder / "img1", "match_07", fps=source.info.fps)
    again = VideoFileSource(video)
    assert images.info == again.info
    for frame in range(20):
        assert np.array_equal(images.read(frame), again.read(frame)), f"frame {frame}"
    # The frames differ from each other, so a shift of the order would fail above.
    assert not np.array_equal(again.read(0), again.read(1))


def test_frames_are_not_written_over_a_folder_and_a_failure_leaves_no_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    write_video(video, textured_frames(4), fps=25.0)
    source = VideoFileSource(video)
    taken = tmp_path / "data" / "clip"
    taken.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="is already there"):
        write_video_frames(source, "b" * 64, taken)
    written: list[str] = []

    def fail_third(path: str, image: np.ndarray, params: Sequence[int]) -> bool:
        written.append(path)
        return len(written) < 3

    monkeypatch.setattr(cv2, "imwrite", fail_third)
    with pytest.raises(OSError, match="cannot write the image"):
        write_video_frames(source, "b" * 64, tmp_path / "data" / "other")
    assert sorted(path.name for path in (tmp_path / "data").iterdir()) == ["clip"]


def test_video_record_is_none_without_a_file_and_refused_with_wrong_content(tmp_path: Path) -> None:
    assert read_video_record(tmp_path) is None
    (tmp_path / "video.json").write_text(json.dumps({"format": "pitchvlm-video-frames-1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="video.json"):
        read_video_record(tmp_path)
