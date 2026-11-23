"""Frame sources: where the pixels of a clip come from."""

from __future__ import annotations

import json
import math
import operator
import os
import secrets
import shutil
import threading
from pathlib import Path
from typing import Literal, Protocol

import cv2
import numpy as np
from pydantic import Field, FiniteFloat

from pitchvlm.schemas import ClipInfo, Record

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
#: Largest distance of the time of a video frame from its place at a constant frame rate, in frames.
FRAME_TIME_TOLERANCE = 0.5
#: Folder of the frame images inside a clip folder, as in the dataset.
FRAME_FOLDER = "img1"
#: The record of a video next to the folder of its frame images (``write_video_frames``), and its format.
VIDEO_RECORD_NAME = "video.json"
VIDEO_RECORD_FORMAT = "pitchvlm-video-frames-1"
#: PNG compression of the written frames: 1 is fast; every level keeps every pixel.
PNG_COMPRESSION = 1
#: Decoded frames that a video source keeps for a repeated read. One 1080p frame needs about 6 MB.
_RECENT_FRAMES = 8


class FrameSource(Protocol):
    """A clip that can return any frame as a BGR uint8 image of shape (height, width, 3).

    ``read`` takes a frame index from 0 to ``info.n_frames - 1``. The two
    sources of this module raise ``IndexError`` for any other index (a negative
    index does not count from the end), and they return a new array on every
    call, so the caller may draw on it.
    """

    info: ClipInfo

    def read(self, frame: int) -> np.ndarray: ...


def keyframe_indices(n_frames: int, fps: float, interval_s: float = 0.48) -> list[int]:
    """Frames sent to the model: one every ``interval_s`` seconds from frame 0, plus the last frame.

    At 25 fps and 0.48 s this is every 12th frame: 64 keyframes for a 750-frame clip.
    The step is at least one frame, so an interval shorter than one frame gives every frame.
    ``n_frames``, ``fps`` and ``interval_s`` must be above zero, and the last two finite.
    """
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if not 0 < fps < math.inf:
        raise ValueError(f"fps must be a positive number, got {fps}")
    if not 0 < interval_s < math.inf:
        raise ValueError(f"interval_s must be a positive number, got {interval_s}")
    step = max(1, round(interval_s * fps))
    indices = list(range(0, n_frames, step))
    if indices[-1] != n_frames - 1:
        indices.append(n_frames - 1)
    return indices


def _frame_index(frame: int, n_frames: int) -> int:
    """Return a frame index as a plain int. An index outside 0 to ``n_frames - 1`` raises ``IndexError``."""
    index = operator.index(frame)
    if not 0 <= index < n_frames:
        raise IndexError(f"frame {index} is outside the clip: it has the frames 0 to {n_frames - 1}")
    return index


def _frame_files(folder: Path) -> list[Path]:
    """Return the frame files of a folder in the order of their numbers.

    A frame file has an image suffix and a name that is a whole number, like
    ``000001.jpg``. Hidden files (names that start with a dot) are skipped.
    A name that is not a number, and two files with the same number, raise
    ``ValueError``: the frame order would not be sure.
    """
    by_number: dict[int, Path] = {}
    for path in folder.iterdir():
        if path.suffix.lower() not in IMAGE_SUFFIXES or path.name.startswith("."):
            continue
        if not (path.stem.isascii() and path.stem.isdigit()):
            raise ValueError(f"the name of a frame file must be a number, like 000001.jpg: {path}")
        number = int(path.stem)
        if number in by_number:
            raise ValueError(f"{by_number[number].name} and {path.name} in {folder} have the same number")
        by_number[number] = path
    if not by_number:
        raise FileNotFoundError(f"no frame images in {folder}")
    return [by_number[number] for number in sorted(by_number)]


class ImageFolderSource:
    """A folder of numbered frame images (the SoccerNet layout: ``img1/000001.jpg`` ...).

    The files are read in the order of their numbers, so ``2.jpg`` comes before
    ``10.jpg``. The file with the lowest number is frame 0.
    """

    def __init__(self, folder: Path, clip_id: str, fps: float = 25.0) -> None:
        self._paths = _frame_files(Path(folder))
        first = self._load(self._paths[0])
        height, width = first.shape[:2]
        self.info = ClipInfo(clip_id=clip_id, n_frames=len(self._paths), fps=fps, width=width, height=height)

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"cannot read image {path}")
        return image

    def path(self, frame: int) -> Path:
        """Return the image file of a frame."""
        return self._paths[_frame_index(frame, self.info.n_frames)]

    def read(self, frame: int) -> np.ndarray:
        return self._load(self.path(frame))


def check_frame_times(path: Path, times_ms: list[float], *, fps: float, header_frames: float) -> float:
    """Return the frame rate of the decoded frames of a video, and raise ``OSError`` unless they are all its frames.

    ``times_ms`` holds the time of every decoded frame in milliseconds, in the order that the
    decoder gives them (the order in which they are shown); ``fps`` and ``header_frames`` are
    the frame rate and the frame count of the header of the video. The rate of the frames is
    measured from their times: the time from the first to the last frame over the frame count
    less one. The rate of the header is not used for it, because OpenCV gives the frame count
    over the duration of the stream, and the duration holds the length of the last frame, which
    a cut file can change: a good video of 750 frames 40 ms apart has a header rate of 25.03
    when its last frame lasts 1 ms. The rate of the header is the answer only for a video of one
    frame. Three faults raise, each with one line that says what to do:

    - The header names no frame count (0 or less): nothing shows a frame that could not be decoded.
    - A frame is ``FRAME_TIME_TOLERANCE`` steps or more away from the frame before it, at the
      usual length of a frame (the median of the steps), or more than that from its place counted
      from the first frame, at the measured rate. The first test finds a frame that the decoder dropped or
      repeated anywhere in the clip, the second a frame rate that changes slowly.
    - The header names another number of frames than were decoded: a frame cannot be read, or
      the header is wrong.

    The times are checked before the count, so a video with a variable frame rate gets the
    line about the rate also when the decoder dropped frames of it.
    """
    if header_frames <= 0:
        raise OSError(
            f"video {path} names no frame count in its header, so a frame that cannot be decoded would not be seen. "
            "Convert it to a video file with a frame count (for example MP4) first"
        )
    if len(times_ms) > 1:
        usual_ms = float(np.median(np.diff(times_ms)))
        step_ms = (times_ms[-1] - times_ms[0]) / (len(times_ms) - 1)
        for index in range(1, len(times_ms)):
            shown_ms = times_ms[index] - times_ms[0]
            after_ms = times_ms[index] - times_ms[index - 1]
            if not (
                usual_ms > 0
                and abs(after_ms - usual_ms) < FRAME_TIME_TOLERANCE * usual_ms
                and abs(shown_ms - index * step_ms) <= FRAME_TIME_TOLERANCE * step_ms
            ):
                raise OSError(
                    f"video {path}: frame {index} is shown at {shown_ms:.1f} ms, {after_ms:.1f} ms after the frame "
                    f"before it, but the frames are {usual_ms:.1f} ms apart as a rule and {step_ms:.1f} ms apart on "
                    "average. The video has a variable frame rate, or a frame is missing or doubled: convert it to a "
                    "constant frame rate first"
                )
        # Every step is more than half the usual step, so the steps and their mean are above 0.
        rate = 1000.0 / step_ms
    else:
        rate = fps
    if round(header_frames) != len(times_ms):
        raise OSError(
            f"video {path}: its header names {round(header_frames)} frames, but {len(times_ms)} frames can be decoded. "
            "A frame cannot be read, or the header is wrong: convert the video again from its source"
        )
    return rate


class VideoFileSource:
    """A video file. Frames are decoded when they are read, so the memory use does not grow with the clip.

    The file is decoded one time when the source is made, to count its frames and to
    check them (``check_frame_times``): a frame that cannot be decoded, a frame count of
    the header that differs from the decoded frames, and a variable frame rate raise
    ``OSError``. The decoder of OpenCV drops or repeats frames of such a video without
    an error, so the frames would not be sure.
    After that the decoder runs forward only. Reading frames in increasing
    order is fast. A read of an earlier frame starts the decoder again at the
    first frame, unless the frame is one of the last frames that were read:
    the source keeps a few of them. Frames can still be read in any order, and
    from several threads.
    """

    def __init__(self, path: Path, clip_id: str | None = None) -> None:
        self._path = Path(path)
        capture = self._open()
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        header_frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        first: np.ndarray | None = None
        times_ms: list[float] = []
        while capture.grab():
            if first is None:
                ok, first = capture.retrieve()
                if not ok:
                    capture.release()
                    raise OSError(f"cannot decode frame 0 of video {self._path}")
            times_ms.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)))
        capture.release()
        if first is None or not 0 < fps < math.inf:
            raise OSError(f"video {self._path} has no readable frames or no frame rate")
        rate = check_frame_times(self._path, times_ms, fps=fps, header_frames=header_frames)
        height, width = first.shape[:2]
        self.info = ClipInfo(
            clip_id=clip_id or self._path.stem, n_frames=len(times_ms), fps=rate, width=width, height=height
        )
        self._lock = threading.Lock()
        #: The last frames that were read, by frame index, oldest first.
        self._recent: dict[int, np.ndarray] = {}
        self._capture = self._open()
        #: Index of the frame that the decoder gives next.
        self._position = 0

    @property
    def path(self) -> Path:
        """Return the video file."""
        return self._path

    def _open(self) -> cv2.VideoCapture:
        """Open the video file. The decoder then stands at the first frame."""
        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            raise OSError(f"cannot open video {self._path}")
        return capture

    def read(self, frame: int) -> np.ndarray:
        index = _frame_index(frame, self.info.n_frames)
        with self._lock:
            image = self._recent.get(index)
            if image is None:
                image = self._decode(index)
            return image.copy()

    def _decode(self, index: int) -> np.ndarray:
        """Run the decoder forward to a frame, keep the frame for a repeated read and return it."""
        if index < self._position:
            self._capture.release()
            self._capture = self._open()
            self._position = 0
        while self._position <= index:
            if not self._capture.grab():
                raise OSError(f"cannot decode frame {self._position} of video {self._path}")
            self._position += 1
        ok, image = self._capture.retrieve()
        if not ok:
            raise OSError(f"cannot decode frame {index} of video {self._path}")
        self._recent[index] = image
        if len(self._recent) > _RECENT_FRAMES:
            del self._recent[next(iter(self._recent))]
        return image


class VideoRecord(Record):
    """What the clip folder of the frame images of a video says about the video (``write_video_frames``).

    The record holds no path: ``video`` is the name of the video file without its folder.
    """

    format: Literal["pitchvlm-video-frames-1"]
    video: str = Field(min_length=1)
    #: SHA-256 of the bytes of the video file, as 64 small hexadecimal digits.
    video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    n_frames: int = Field(gt=0)
    fps: FiniteFloat = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


def frame_image_name(frame: int) -> str:
    """Return the name of the image file of a frame: frame 0 is ``000001.png``, as in the dataset."""
    return f"{operator.index(frame) + 1:06d}.png"


def write_video_frames(source: VideoFileSource, video_sha256: str, clip_folder: Path) -> VideoRecord:
    """Write every frame of a video as a PNG image, and the record of the video, into a new clip folder.

    The images go to ``clip_folder/img1`` (``frame_image_name``) and the record to
    ``clip_folder/video.json``. PNG keeps every pixel, so ``ImageFolderSource`` reads the
    frames of ``source``: the same count, the same order and the same pixels. The folder
    is first written under a hidden name next to ``clip_folder`` and gets its name only
    when every file is written, so a stopped command leaves no clip folder with a part of
    the frames. ``FileExistsError`` is raised when ``clip_folder`` is already there:
    nothing is replaced. An image that cannot be written raises ``OSError``, and the
    hidden folder is removed.
    """
    clip_folder = Path(clip_folder)
    if clip_folder.exists():
        raise FileExistsError(f"{clip_folder} is already there: remove it, or write the frames to another place")
    info = source.info
    record = VideoRecord(
        format=VIDEO_RECORD_FORMAT,
        video=source.path.name,
        video_sha256=video_sha256,
        n_frames=info.n_frames,
        fps=info.fps,
        width=info.width,
        height=info.height,
    )
    clip_folder.parent.mkdir(parents=True, exist_ok=True)
    partial = clip_folder.with_name(f".{clip_folder.name}.partial-{secrets.token_hex(4)}")
    images = partial / FRAME_FOLDER
    images.mkdir(parents=True)
    try:
        for frame in range(info.n_frames):
            path = images / frame_image_name(frame)
            if not cv2.imwrite(str(path), source.read(frame), [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]):
                raise OSError(f"cannot write the image {path}")
        (partial / VIDEO_RECORD_NAME).write_text(record.model_dump_json() + "\n", encoding="utf-8")
        os.rename(partial, clip_folder)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return record


def read_video_record(clip_folder: Path) -> VideoRecord | None:
    """Return the record of the video whose frames a clip folder holds, or None when it holds no record.

    ``ValueError`` is raised for a record that does not follow the format of ``VideoRecord``.
    """
    path = Path(clip_folder) / VIDEO_RECORD_NAME
    if not path.is_file():
        return None
    try:
        return VideoRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
