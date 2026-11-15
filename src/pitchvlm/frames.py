"""Frame sources: where the pixels of a clip come from."""

from __future__ import annotations

import math
import operator
import threading
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from pitchvlm.schemas import ClipInfo

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
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


class VideoFileSource:
    """A video file. Frames are decoded when they are read, so the memory use does not grow with the clip.

    The file is decoded one time when the source is made, to count its frames.
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
        ok, first = capture.read()
        n_frames = 0
        while ok:
            n_frames += 1
            ok, _ = capture.read()
        capture.release()
        if n_frames == 0 or not 0 < fps < math.inf:
            raise OSError(f"video {self._path} has no readable frames or no frame rate")
        height, width = first.shape[:2]
        self.info = ClipInfo(
            clip_id=clip_id or self._path.stem, n_frames=n_frames, fps=fps, width=width, height=height
        )
        self._lock = threading.Lock()
        #: The last frames that were read, by frame index, oldest first.
        self._recent: dict[int, np.ndarray] = {}
        self._capture = self._open()
        #: Index of the frame that the decoder gives next.
        self._position = 0

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
