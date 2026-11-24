"""Short video files of chosen frames of a clip, for the video input of a model step.

A video step (``StepSettings.input == "video"``) sends a video file instead of a
list of images. This module writes such a file: the chosen frames of a frame
source, in their order, at the sampling rate of the step. So the model samples
every written frame, and frame ``k`` of the file is at ``k / fps`` seconds. The
time of each frame in the file is known exactly, and the answer of the model is
mapped back to the frames of the clip by that time (``vlm.steps``).

Rules of the files:

- They go to a folder outside the repository on the run machine (the data folder),
  never to the cache folder: the cache folder goes to the results branch, which is
  text only (rule 2).
- A file that is already there is used as it is and is not written again. The
  cache key of its request holds the frames of the clip and the settings of the
  file, not its bytes (``VideoPart.identity``): a file written again from the same
  frames can get other bytes (another codec or OpenCV build), and it still finds the
  stored answer, which keeps the hash of the bytes that were sent and counts a hit
  on other bytes (``vlm.cache``). The name of a file holds a hash of the decoded
  pixels of its frames, so new footage under an old clip name gets new files and
  never the old ones; a stored answer of the old footage is refused as for images.
  Two writes of the same frames with one codec gave the same bytes on the build
  machine (H.264 and MPEG-4 Part 2, OpenCV 5.0); the two codecs gave other bytes.
- A file is written under a hidden name and gets its name when it is complete.

The written frames are compressed by the video codec, so the model sees other
pixels than the JPEG images of an image step. The probe of video input compares
the two on the same frames (``probes.probe_people``).
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import os
import secrets
from collections.abc import Sequence
from pathlib import Path

import cv2

from pitchvlm.frames import FrameSource
from pitchvlm.vlm.types import Resolution, VideoFrames, VideoPart

#: The video codecs that are tried, in order, as four-character codes of OpenCV: H.264, then
#: MPEG-4 Part 2. Both pages name the container type video/mp4; neither names a codec.
VIDEO_CODECS: tuple[str, ...] = ("avc1", "mp4v")
#: File type of the written files.
VIDEO_MIME_TYPE = "video/mp4"
#: Rate at which the frames of a video step are written when the step sends no rate (an OpenRouter
#: step): the default sampling rate of Gemini, 1 frame per second (video page, published). The
#: sampling rate of the other providers is not published, so a provider that samples less often
#: can miss a target frame; the probe counts such answers (``n_entries_wrong_time``).
DEFAULT_WRITE_FPS = 1.0


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of the bytes of a file, as 64 small hexadecimal digits."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frames_digest(source: FrameSource, frames: Sequence[int]) -> str:
    """Return a SHA-256 of the decoded pixels of some frames of a source, in their order, with their sizes."""
    digest = hashlib.sha256()
    for frame in frames:
        image = source.read(frame)
        digest.update(json.dumps([operator.index(frame), list(image.shape), str(image.dtype)]).encode("utf-8"))
        digest.update(image.tobytes())
    return digest.hexdigest()


def video_file_name(clip_id: str, step: str, frames: Sequence[int], fps: float, pixels: str) -> str:
    """Return the file name of the video of some frames: the step, and a hash of the clip, frames, rate and pixels.

    ``pixels`` is ``frames_digest`` of the frames. So new footage under the name of an
    old clip gets new files, and the old file, whose hash is in the key of an old
    answer, is never used for it.
    """
    facts = {
        "clip_id": clip_id,
        "frames": [operator.index(frame) for frame in frames],
        "fps": float(fps),
        "pixels": pixels,
    }
    digest = hashlib.sha256(json.dumps(facts, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"{step}-{digest}.mp4"


def write_frames_video(source: FrameSource, frames: Sequence[int], path: Path, fps: float) -> None:
    """Write the frames of a source, in the given order, as an MP4 file at ``fps`` frames per second.

    A file that is already at ``path`` is kept as it is. The first codec of
    ``VIDEO_CODECS`` that OpenCV can open is used. ``ValueError`` is raised for no
    frames and a rate that is not above 0; ``OSError`` when no codec can be opened
    or a frame has another size than the first one.
    """
    path = Path(path)
    if path.exists():
        return
    if len(frames) == 0:
        raise ValueError("a video needs at least one frame")
    if not 0 < fps < math.inf:
        raise ValueError(f"the rate of a video must be a number above 0, got {fps!r}")
    info = source.info
    size = (info.width, info.height)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{secrets.token_hex(4)}.mp4")
    writer = None
    try:
        for codec in VIDEO_CODECS:
            writer = cv2.VideoWriter(str(partial), cv2.VideoWriter_fourcc(*codec), float(fps), size)
            if writer.isOpened():
                break
            writer.release()
            writer = None
        if writer is None:
            raise OSError(f"OpenCV can open none of the video codecs {', '.join(VIDEO_CODECS)}")
        for frame in frames:
            image = source.read(frame)
            if (image.shape[1], image.shape[0]) != size:
                raise OSError(f"frame {frame} has the size {image.shape[1]} x {image.shape[0]}, the clip {size}")
            writer.write(image)
        writer.release()
        writer = None
        os.replace(partial, path)
    finally:
        if writer is not None:
            writer.release()
        partial.unlink(missing_ok=True)


def frames_video_part(
    source: FrameSource,
    frames: Sequence[int],
    *,
    folder: Path,
    step: str,
    fps: float,
    resolution: Resolution,
    send_fps: bool,
) -> VideoPart:
    """Write (or reuse) the video of some frames of a clip and return its video part.

    The file is ``folder / <clip id> / video_file_name(...)``: its name holds a hash
    of the decoded pixels of the frames (``frames_digest``), so a file is used again
    only for the same pixels. Frame ``k`` of
    ``frames`` is at ``k / fps`` seconds of the file, and the file is
    ``len(frames) / fps`` seconds long. ``send_fps`` says whether the part asks
    the provider for the rate ``fps`` (Gemini); without it the provider samples
    at its own rate (OpenRouter).
    """
    info = source.info
    pixels = frames_digest(source, frames)
    path = Path(folder) / info.clip_id / video_file_name(info.clip_id, step, frames, fps, pixels)
    write_frames_video(source, frames, path, fps)
    return VideoPart(
        path=path,
        sha256=file_sha256(path),
        mime_type=VIDEO_MIME_TYPE,
        duration_s=len(frames) / fps,
        fps=fps if send_fps else None,
        resolution=resolution,
        source=VideoFrames(
            frames=tuple(operator.index(frame) for frame in frames),
            write_fps=float(fps),
            width=info.width,
            height=info.height,
            pixels_sha256=pixels,
        ),
    )


def context_frames(frame: int, *, n_frames: int, clip_fps: float, video_fps: float, context: int) -> list[int]:
    """Return the frames of a video around one frame: ``context`` frames before it and after it, at the video rate.

    The frames are ``round(clip_fps / video_fps)`` frames of the clip apart, so the
    video shows time at about its real speed. Frames outside the clip are left out,
    so a frame near an edge of the clip has less context on that side.
    """
    if context < 0:
        raise ValueError(f"the context must not be negative, got {context}")
    if not (0 < clip_fps < math.inf and 0 < video_fps < math.inf):
        raise ValueError("the rates of the clip and of the video must be numbers above 0")
    step = max(1, round(clip_fps / video_fps))
    around = (frame + offset * step for offset in range(-context, context + 1))
    return [index for index in around if 0 <= index < n_frames]
