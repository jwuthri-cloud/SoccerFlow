"""Run an external tool on the frames of a job file and write the run file.

The package writes the job file (``python -m pitchvlm.cli tool-job``) and measures the run file
(``python -m pitchvlm.cli tool-score``). For the engines of the pipeline it writes a clip job
of every frame of one clip (``python -m pitchvlm.cli clip-job``), and ``run`` reads the clip
run (``--detections``, ``--calibration``). The four formats are described in
``src/pitchvlm/fallbacks/runs.py``. The script tells a clip job by its format name; it writes
the clip run in parts of at most ``CLIP_RUN_PART_MAX_BYTES`` bytes, with the compact rows of
the detector. The answer of a tool for one frame is a payload in the
format of its adapter module: ``src/pitchvlm/fallbacks/rfdetr.py`` for the detector and
``src/pitchvlm/fallbacks/pnlcalib.py`` for the calibration.

This script runs in the environment of the tool, not in the environment of the package. It
imports the tool and what the tool needs (PyTorch, NumPy, Pillow and, for PnLCalib, PyYAML),
and it does not import pitchvlm. It needs Python 3.10 or newer. The detector families D-FINE
(``dfine``) and RT-DETRv2 (``rtdetrv2``) share one tool, Hugging Face transformers, and one
environment.

    python tools/run_tool.py rfdetr --job JOB --root DATA --out RUN --device mps
    python tools/run_tool.py dfine --job JOB --root DATA --out RUN --device cuda:0
    python tools/run_tool.py pnlcalib --job JOB --root DATA --out RUN --device cpu \\
        --repo PNLCALIB --weights-kp SV_kp --weights-lines SV_lines

``DATA`` is the data folder: the job names each image by its path below it. A clip job of a
video names the PNG images that ``python -m pitchvlm.cli video-frames`` wrote: PNG keeps every
pixel, so the tool reads the frames that ``run`` decodes from the video, and the clip run names
the video by its SHA-256. ``--device`` is
the device name of PyTorch, for example ``mps`` (Apple Silicon), ``cpu`` or ``cuda:0``.

Exit code 0 means that the run file was written. Exit code 2 means that the script could not
run; its last line on the standard error stream then starts with ``error:``. A wrong command
line also gives the exit code 2, and its last line is the ``error:`` line of the argument parser.

The script takes the variables ``GEMINI_API_KEY`` and ``OPENROUTER_API_KEY`` out of its
environment before it loads a tool (``KEY_VARIABLES``): the tools are code of other
projects, and they must not see a key of a model API. Code of the tool environment runs
before that line (the start-up code of a package, and NumPy), so start the script without
the variables: ``env -u GEMINI_API_KEY -u OPENROUTER_API_KEY python ...``.

What each tool gets, and what the script takes from it:

- RF-DETR (package ``rfdetr``, versions ``RFDETR_VERSIONS``): the frame as an RGB ``uint8``
  array of shape (height, width, 3) goes to ``predict`` of the model class that the job names,
  with the score limit of the job as threshold and with the resize of the job (with or
  without antialiasing). The tool scales its boxes to the pixels of the frame and gives the
  class name of each detection. Every detection goes into the payload: the adapter of the
  package keeps the people.
- D-FINE and RT-DETRv2 (package ``transformers``, versions ``TRANSFORMERS_VERSIONS``): the
  checkpoint of the size that the job names, at its commit (``TRANSFORMERS_CHECKPOINTS``),
  through ``AutoImageProcessor`` and ``AutoModelForObjectDetection``. The image processor
  resizes the frames of a batch to the square input size of the job; the model gives a class
  score per query and class and a box per query (centre, width, height in parts of the
  input); ``detections_of_queries`` turns them into boxes in pixels of the frame, like the
  ``post_process_object_detection`` of the image processor, and keeps the classes ``person``
  and ``sports ball`` (``KEPT_LABELS``). On the first call the script checks its boxes against
  that function of the tool and stops when they differ.
- PnLCalib (a clone of its repository at the commit ``PNLCALIB_COMMIT``): the script follows
  the steps of ``inference.py`` of the clone. It takes the projection matrix of the camera and
  writes the mapping of the grass (pitch metres to pixels of the frame). The world frame of
  the tool is the pitch convention of the package: the origin at the centre mark, x towards
  the right goal, y towards the main camera.

The time of a frame is the time of the tool for that frame, without the reading of the image
file. A detector of transformers takes the frames in batches of ``batch`` frames; the time of a
frame is then the time of its batch divided by the frames of the batch, and the run facts add
the median time per frame of the three parts of a call (``stage_ms_per_frame``: the image
processor, the model, the conversion of its output). The first frame (or the first batch) is
given to the tool one more time before the measured calls, because the first call of a model
also builds what it needs.

This script was written from the source of the tools. It was never run with them: the
tests replace each tool by a stand-in with the same interface.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import ModuleType, SimpleNamespace
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:
    print(
        "error: NumPy is not installed here. Run this script with the Python of the environment of the tool",
        file=sys.stderr,
    )
    raise SystemExit(2) from None

#: Format names of the job file and the run file: those of ``pitchvlm.fallbacks.runs``.
JOB_FORMAT = "pitchvlm-tool-job-1"
RUN_FORMAT = "pitchvlm-tool-run-1"
#: Format names of the clip job and the clip run: those of ``pitchvlm.fallbacks.runs``.
CLIP_JOB_FORMAT = "pitchvlm-clip-job-1"
CLIP_RUN_FORMAT = "pitchvlm-clip-run-1"
#: Format name of the feed of a live run (``--feed``): that of ``pitchvlm.feed``.
TOOL_FEED_FORMAT = "pitchvlm-tool-feed-1"
#: Largest size of one part of a clip run, bytes: that of ``pitchvlm.fallbacks.runs``.
CLIP_RUN_PART_MAX_BYTES = 1_000_000
#: Digits of the SHA-256 of a video in a clip job.
HEX_DIGITS = "0123456789abcdef"
#: Rounding of a compact detection row: the box to 0.1 pixel, the score to 0.0001.
BOX_DECIMALS = 1
SCORE_DECIMALS = 4
#: Keys of the tool section of a job of a detector of transformers: those of every family of it.
TRANSFORMERS_KEYS = frozenset({"name", "model", "min_score", "input_size", "batch", "half"})
#: Checkpoints of each detector family of transformers, by size: the repository on the Hugging Face hub
#: and the commit whose model card was read (COCO-only weights, Apache-2.0; docs/DECISIONS.md, 2026-11-23).
TRANSFORMERS_CHECKPOINTS: dict[str, dict[str, tuple[str, str]]] = {
    "dfine": {
        "nano": ("ustc-community/dfine-nano-coco", "066438d3d8f0da137a37b38fdf3368fd4afceced"),
        "small": ("ustc-community/dfine-small-coco", "f79e65b5fbb33ceb9d3ebba042955d7410c608f8"),
        "medium": ("ustc-community/dfine-medium-coco", "4ab9f5e466432f4fcf9a9a023da6aa5ecc9c9829"),
        "large": ("ustc-community/dfine-large-coco", "aa8f8211545e4cff5adf52ea512f9ac74a9824ec"),
        "xlarge": ("ustc-community/dfine-xlarge-coco", "ea4f6be7350bbe3c199ec6febc74168346cb5a68"),
    },
    "rtdetrv2": {
        "r18vd": ("PekingU/rtdetr_v2_r18vd", "5650961749fa93567c0d46fc7f43ea4f9e914107"),
        "r34vd": ("PekingU/rtdetr_v2_r34vd", "5d60d8fb5eb2a3ee5575bf200ecbce4e2cfa420c"),
        "r50vd": ("PekingU/rtdetr_v2_r50vd", "282494075698cab9faa1096ae26856890030c817"),
        "r101vd": ("PekingU/rtdetr_v2_r101vd", "2c5dbbd2d4d8c8814827a3b42737ba1afce3cf2a"),
    },
}
#: The model type that the configuration of a checkpoint of each family names.
TRANSFORMERS_MODEL_TYPES = {"dfine": "d_fine", "rtdetrv2": "rt_detr_v2"}
#: Versions of the package ``transformers`` whose source was read for this script.
TRANSFORMERS_VERSIONS = ("5.23.0",)
#: Class names of the COCO weights that a detector payload of transformers keeps: those that the package reads.
KEPT_LABELS = ("person", "sports ball")
#: Largest difference between the boxes of the script and those of the tool on the first call (pixels), and
#: between their scores.
SELF_CHECK_BOX_PX = 0.01
SELF_CHECK_SCORE = 1e-5
#: Keys of the tool section of a job, by tool.
TOOL_KEYS = {
    "rfdetr": frozenset({"name", "model", "min_score", "antialias"}),
    **dict.fromkeys(TRANSFORMERS_CHECKPOINTS, TRANSFORMERS_KEYS),
    "pnlcalib": frozenset({"name", "kp_threshold", "line_threshold", "pnl_refine"}),
}
#: The tools whose payload is a list of boxes: the detector families.
DETECTOR_TOOLS = ("rfdetr", *TRANSFORMERS_CHECKPOINTS)
#: Versions of the package ``rfdetr`` whose source was read for this script.
RFDETR_VERSIONS = ("1.12.0", "1.12.1")
#: Model classes of ``rfdetr`` that come with pretrained weights for the 80 everyday classes.
RFDETR_MODELS = ("RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge")
#: Commit of the PnLCalib repository whose source was read for this script.
PNLCALIB_COMMIT = "42088864170f75ea02e76d20e9a9a1db59edb3e3"
#: Packages of the PnLCalib clone: folders of its top level whose files the script imports.
PNLCALIB_PACKAGES = ("model", "utils")
#: Input size of the two PnLCalib models, pixels: (width, height).
PNLCALIB_INPUT = (960, 540)
#: Pitch convention that a calibration payload states.
PITCH_ORIGIN = "centre"
#: Variables that hold a key of a model API on the run machine: the Gemini key of ``pitchvlm.cli``
#: and the key of the OpenRouter provider.
KEY_VARIABLES = ("GEMINI_API_KEY", "OPENROUTER_API_KEY")
EXIT_OK = 0
EXIT_ERROR = 2
#: What to do when a package or a module of a tool is missing.
WRONG_ENVIRONMENT = "Run this script with the Python of the environment of the tool"

#: Turns one frame, an RGB ``uint8`` array of shape (height, width, 3), into the payload of the frame.
Answer = Callable[[np.ndarray], dict[str, Any]]
#: Turns a batch of frames into the payload of each frame, in the order of the frames.
BatchAnswer = Callable[[Sequence[np.ndarray]], list[dict[str, Any]]]


class RunnerError(Exception):
    """The script cannot run. The message says why and what to do."""


# --- Job file and run file -----------------------------------------------------------


def read_job(path: Path, tool: str) -> dict[str, Any]:
    """Read a job file or a clip job and check that it is a job for ``tool``. Wrong content raises ``RunnerError``.

    The job must have one of the two job format names of this script, the tool section of
    ``tool`` with exactly its keys, and at least one frame. The image of a frame must be a
    path below the data folder: no absolute path and no step to a parent folder.

    A clip job names its clip and the image of every frame (``images``). The job that is
    returned then also holds ``frames`` in the form of a job file, frame ``i`` with the
    image ``images[i]``, and the name of the clip as its ``name``.
    """
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RunnerError(f"the job file {path} cannot be read: {error}") from error
    except ValueError as error:
        raise RunnerError(f"the job file {path} holds no JSON: {error}") from error
    if not isinstance(job, dict) or job.get("format") not in (JOB_FORMAT, CLIP_JOB_FORMAT):
        raise RunnerError(
            f"the job file {path} is not in the format {JOB_FORMAT} or {CLIP_JOB_FORMAT}: "
            "write it with the tool-job or the clip-job command"
        )
    if job["format"] == CLIP_JOB_FORMAT:
        job = _clip_job_frames(job, path)
    if not isinstance(job.get("name"), str) or not job["name"]:
        raise RunnerError(f"the job file {path} has no name")
    settings = job.get("tool")
    if not isinstance(settings, dict) or settings.get("name") != tool:
        named = settings.get("name") if isinstance(settings, dict) else None
        raise RunnerError(f"the job file {path} is a job for the tool {named!r}, not for {tool!r}")
    if set(settings) != TOOL_KEYS[tool]:
        raise RunnerError(
            f"the job file {path} must have the tool settings {sorted(TOOL_KEYS[tool])}, it has {sorted(settings)}"
        )
    frames = job.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RunnerError(f"the job file {path} names no frame")
    for position, entry in enumerate(frames):
        _check_job_frame(entry, f"the job file {path}, frames[{position}]")
    return job


def _clip_job_frames(job: dict[str, Any], path: Path) -> dict[str, Any]:
    """Return a clip job with the frames of a job file: frame ``i`` reads ``images[i]``.

    ``RunnerError`` is raised for a clip job without the name of its clip, for one whose
    ``images`` is not a list of at least one path, and for a ``video_sha256`` that is neither
    null nor 64 small hexadecimal digits. A job without ``video_sha256`` reads as null.
    """
    clip, images = job.get("clip"), job.get("images")
    if not isinstance(clip, str) or not clip:
        raise RunnerError(f"the clip job {path} has no clip")
    if not isinstance(images, list) or not images:
        raise RunnerError(f"the clip job {path} names no frame")
    video = job.get("video_sha256")
    if video is not None and not (isinstance(video, str) and len(video) == 64 and set(video) <= set(HEX_DIGITS)):
        raise RunnerError(f"the clip job {path}: 'video_sha256' must be null or 64 small hexadecimal digits")
    frames = [{"clip": clip, "frame": index, "image": image} for index, image in enumerate(images)]
    return {**job, "name": clip, "frames": frames, "video_sha256": video}


def _check_job_frame(entry: Any, where: str) -> None:
    """Raise ``RunnerError`` unless a frame entry of a job names a clip, a frame index and an image.

    The image must be a path below the data folder.
    """
    if not isinstance(entry, dict):
        raise RunnerError(f"{where}: expected a mapping")
    clip, frame, image = entry.get("clip"), entry.get("frame"), entry.get("image")
    if not isinstance(clip, str) or not clip:
        raise RunnerError(f"{where}: 'clip' must be a name")
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise RunnerError(f"{where}: 'frame' must be a whole number from 0")
    if not isinstance(image, str) or not image:
        raise RunnerError(f"{where}: 'image' must be a path")
    image_path = PurePosixPath(image)
    if image_path.is_absolute() or ".." in image_path.parts:
        raise RunnerError(f"{where}: 'image' must be a path below the data folder, got {image!r}")


def read_rgb(path: Path) -> np.ndarray:
    """Read an image file as an RGB ``uint8`` array of shape (height, width, 3)."""
    from PIL import Image  # Pillow is in the environment of both tools.

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def answer_frames(
    job: Mapping[str, Any], root: Path, answer: BatchAnswer, batch: int = 1, live: LiveFeed | None = None
) -> list[dict[str, Any]]:
    """Give every frame of a job to the tool and return the frame entries of the run file.

    The frames go to ``answer`` in batches of ``batch`` frames, in the order of the job; the
    last batch can be smaller. A tool that takes one frame at a time is given with
    ``one_at_a_time``. The first batch is given to the tool one more time before the measured
    calls, and that answer is thrown away. ``seconds`` of an entry is the time of ``answer``
    for its batch divided by the frames of the batch. An answer with another number of payloads
    than frames raises ``RunnerError``.

    An error of the tool on a batch is not caught: a line on the error stream names the frame
    (or the first and the last frame of the batch), and the error goes on. So a frame is never
    counted as a frame without an answer because of an error. An error of the operating system
    inside the tool also shows its traceback here, because ``main`` prints only one line for
    such an error. An image file that cannot be read stops the run with a line that names the
    frame.

    With ``live`` (a live run, ``--feed``) the frames come in the groups of ``LiveFeed.groups``,
    which may wait for a frame to come (``--pace``) or skip frames (``--skip-behind``), and each
    entry is written to the feed as soon as its batch is answered. An entry then also holds
    ``arrived_at``, the time at which its frame came to the tool.
    """

    def where(entries: Sequence[Mapping[str, Any]]) -> str:
        first, last = entries[0], entries[-1]
        if len(entries) == 1:
            return f"clip {first['clip']}, frame {first['frame']} ({first['image']})"
        return f"the batch of clip {first['clip']}, frame {first['frame']} ({first['image']}) to frame {last['frame']}"

    def ask(entries: Sequence[Mapping[str, Any]], images: list[np.ndarray]) -> list[dict[str, Any]]:
        try:
            payloads = answer(images)
        except Exception as error:
            print(f"the tool stopped on {where(entries)}", file=sys.stderr)
            if isinstance(error, OSError):
                traceback.print_exc()  # main prints one line for an OSError: the traceback says where in the tool.
            raise
        if len(payloads) != len(images):
            raise RunnerError(f"the tool gave {len(payloads)} answers for the {len(images)} frames of {where(entries)}")
        return payloads

    def read(entry: Mapping[str, Any]) -> np.ndarray:
        try:
            return read_rgb(root / entry["image"])
        except OSError:
            print(f"the image cannot be read: {where([entry])}", file=sys.stderr)
            raise

    frames = job["frames"]
    batches = [frames[start : start + batch] for start in range(0, len(frames), batch)]
    ask(batches[0], [read(entry) for entry in batches[0]])
    if live is not None:
        live.start()
    entries = []
    for group in batches if live is None else live.groups(frames, batch):
        arrived = [live.arrived_at(entry["frame"]) for entry in group] if live is not None else []
        images = [read(entry) for entry in group]
        start = time.perf_counter()
        payloads = ask(group, images)
        seconds = (time.perf_counter() - start) / len(group)
        for position, (entry, payload) in enumerate(zip(group, payloads, strict=True)):
            done = {"clip": entry["clip"], "frame": entry["frame"], "seconds": round(seconds, 6), "payload": payload}
            if live is not None:
                done["arrived_at"] = arrived[position]
                live.write(done)
            entries.append(done)
    return entries


def one_at_a_time(answer: Answer) -> BatchAnswer:
    """Return the batch answer of a tool that takes one frame at a time (batches of one frame)."""

    def batch_answer(images: Sequence[np.ndarray]) -> list[dict[str, Any]]:
        return [answer(image) for image in images]

    return batch_answer


def write_run(
    path: Path, job: Mapping[str, Any], runner: Mapping[str, Any], frames: Sequence[Mapping[str, Any]]
) -> None:
    """Write the run file of a job: its name and tool settings, the facts of the run and the answers.

    A value that JSON does not have (NaN, infinity) raises ``ValueError``.
    """
    document = {
        "format": RUN_FORMAT,
        "name": job["name"],
        "tool": job["tool"],
        "runner": dict(runner),
        "frames": list(frames),
    }
    path.write_text(json.dumps(document, allow_nan=False) + "\n", encoding="utf-8")


def compact_detections(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Turn the payload of the detector into the compact payload of a clip run.

    ``labels`` lists the class names of the frame in the order in which they first come, and
    each detection becomes one row: the box rounded to ``BOX_DECIMALS`` decimals, the score
    rounded to ``SCORE_DECIMALS`` decimals and the place of its class name in ``labels``. A
    score at or above a threshold with at most 4 decimals stays at or above it.
    """
    labels: list[str] = []
    rows: list[list[float | int]] = []
    for detection in payload["detections"]:
        label = detection["label"]
        if label not in labels:
            labels.append(label)
        box = [round(value, BOX_DECIMALS) for value in detection["box_xyxy"]]
        rows.append([*box, round(detection["score"], SCORE_DECIMALS), labels.index(label)])
    return {"labels": labels, "rows": rows}


def clip_run_part_path(first: Path, part: int) -> Path:
    """Return the file of one part of a clip run: ``first`` for part 1, ``<stem>-<part:04d><suffix>`` after it."""
    return first if part == 1 else first.with_name(f"{first.stem}-{part:04d}{first.suffix}")


def later_parts(first: Path) -> list[Path]:
    """Return the files next to ``first`` that have the name of a later part of a clip run, sorted."""
    pattern = f"{glob.escape(first.stem)}-[0-9][0-9][0-9][0-9]{glob.escape(first.suffix)}"
    return sorted(path for path in first.parent.glob(pattern) if int(path.stem[-4:]) >= 2)


def clip_run_texts(head: Mapping[str, Any], entries: Sequence[Sequence[Any]]) -> list[str]:
    """Return the text of each part of a clip run, each one at most ``CLIP_RUN_PART_MAX_BYTES`` bytes.

    ``head`` holds the keys that every part repeats; ``entries`` holds ``[frame, seconds,
    payload]`` per frame, in frame order. A part takes frames until the next one would pass
    the limit. One frame that does not fit into a part alone raises ``RunnerError``, and a
    value that JSON does not have (NaN, infinity) raises ``ValueError``.
    """
    compact = {"separators": (",", ":"), "allow_nan": False}
    texts = [json.dumps(list(entry), **compact) for entry in entries]
    widest_head = json.dumps({**head, "part": 9999, "parts": 9999}, **compact)
    room = CLIP_RUN_PART_MAX_BYTES - len(widest_head.encode()) - len(',"frames":[]}\n')
    groups: list[list[str]] = [[]]
    used = 0
    for text in texts:
        size = len(text.encode())
        if size > room:
            raise RunnerError(f"the answer of one frame has {size} bytes: it does not fit into a part of the clip run")
        if groups[-1] and used + 1 + size > room:
            groups.append([])
            used = 0
        used += size + (1 if groups[-1] else 0)
        groups[-1].append(text)
    return [
        json.dumps({**head, "part": number, "parts": len(groups)}, **compact)[:-1]
        + ',"frames":['
        + ",".join(group)
        + "]}\n"
        for number, group in enumerate(groups, start=1)
    ]


def write_clip_run(
    path: Path, job: Mapping[str, Any], job_sha256: str, runner: Mapping[str, Any], frames: Sequence[Mapping[str, Any]]
) -> list[Path]:
    """Write the clip run of a clip job in parts, and return the part files in order.

    The head of each part names the video of the job (``video_sha256``), so that the package
    reads the run only with the pixels that the tool read.

    The answers of the detector are stored as compact rows (``compact_detections``), those of
    the calibration as they are.
    """
    head = {
        "format": CLIP_RUN_FORMAT,
        "clip": job["clip"],
        "tool": job["tool"],
        "n_frames": len(job["frames"]),
        "job_sha256": job_sha256,
        "video_sha256": job["video_sha256"],
        "runner": dict(runner),
    }
    detector = job["tool"]["name"] in DETECTOR_TOOLS
    entries = [
        [entry["frame"], entry["seconds"], compact_detections(entry["payload"]) if detector else entry["payload"]]
        for entry in frames
    ]
    paths = []
    for number, text in enumerate(clip_run_texts(head, entries), start=1):
        part = clip_run_part_path(path, number)
        part.write_text(text, encoding="utf-8")
        paths.append(part)
    return paths


class LiveFeed:
    """The feed of a live run (``--feed``): a header line, then one line per frame as soon as it is answered.

    The lines go to every stream of ``outputs`` in their order (the copy at ``--out`` first, so
    that it holds every line that the feed got, then the feed), each one flushed at once. The header is that of ``pitchvlm.feed``: ``{"format": TOOL_FEED_FORMAT,
    "clip", "n_frames", "video_sha256", "job_sha256", "tool"}``; a frame line is ``{"frame",
    "seconds", "arrived_at", "payload"}`` (the compact payload for a detector, as in a clip run),
    or ``{"frame", "arrived_at", "skipped": true}``.

    ``pace`` (frames per second) plays the clip as a camera would send it: frame ``i`` comes at
    ``start + i / pace`` and is not read before. ``start`` is ``start_at`` (seconds since the
    epoch, so that two tools share one clock) or the time of ``start``. ``skip_behind`` (with a
    pace and batches of one frame) takes the newest frame that has come and skips the frames
    before it that the tool did not reach. Without a pace, a frame comes when the tool reads it.
    """

    def __init__(
        self,
        outputs: Sequence[Any],
        head: Mapping[str, Any],
        *,
        pace: float | None = None,
        start_at: float | None = None,
        skip_behind: bool = False,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if pace is not None and not 0.0 < pace < float("inf"):
            raise RunnerError(f"--pace must be a number of frames per second above 0, got {pace}")
        if (start_at is not None or skip_behind) and pace is None:
            raise RunnerError("--start-at and --skip-behind need --pace")
        self._outputs = list(outputs)
        self._head = dict(head)
        self._detector = head["tool"]["name"] in DETECTOR_TOOLS
        self._pace = pace
        self._start: float | None = start_at
        self._skip_behind = skip_behind
        self._clock = clock
        self._sleep = sleep
        self.skipped = 0

    def start(self) -> None:
        """Fix the start of the pace (when ``start_at`` was not given) and write the header line."""
        if self._start is None:
            self._start = self._clock()
        self._line({"format": TOOL_FEED_FORMAT, **self._head})

    def due(self, frame: int) -> float | None:
        """Return the time at which a frame comes, with a pace; else None."""
        if self._pace is None:
            return None
        assert self._start is not None
        return self._start + frame / self._pace

    def arrived_at(self, frame: int) -> float:
        """Return the time at which a frame came: its due time with a pace, else now (it comes when it is read)."""
        due = self.due(frame)
        return self._clock() if due is None else due

    def groups(self, frames: Sequence[Mapping[str, Any]], batch: int) -> Iterator[list[Mapping[str, Any]]]:
        """Yield the groups of frames to answer, in order, waiting for the last frame of each group to come."""
        if self._skip_behind and batch != 1:
            raise RunnerError(f"--skip-behind needs batches of one frame, and the job has batches of {batch}")
        index = 0
        while index < len(frames):
            if self._skip_behind:
                newest = self._newest_come(len(frames))
                for skipped in range(index, max(index, newest)):
                    self._line({"frame": frames[skipped]["frame"], "arrived_at": self.arrived_at(skipped), "skipped": True})
                    self.skipped += 1
                index = max(index, newest)
            group = list(frames[index : index + batch])
            due = self.due(index + len(group) - 1)
            if due is not None:
                wait = due - self._clock()
                if wait > 0:
                    self._sleep(wait)
            yield group
            index += len(group)

    def _newest_come(self, n_frames: int) -> int:
        """Return the newest frame that has come by now (with a pace), at most the last frame."""
        assert self._start is not None and self._pace is not None
        return min(n_frames - 1, int(math.floor((self._clock() - self._start) * self._pace)))

    def write(self, entry: Mapping[str, Any]) -> None:
        """Write the line of an answered frame."""
        payload = compact_detections(entry["payload"]) if self._detector else entry["payload"]
        self._line(
            {"frame": entry["frame"], "seconds": entry["seconds"], "arrived_at": entry["arrived_at"], "payload": payload}
        )

    def _line(self, data: Mapping[str, Any]) -> None:
        text = json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n"
        for output in self._outputs:
            output.write(text)
            output.flush()


def check_out_place(out: Path, job: Path) -> None:
    """Make the folder of the run file and raise ``RunnerError`` when the run file cannot be written there.

    The check runs before the tool is built, so that the answers of a long run are not lost
    at its end. It refuses a path that is a folder, a place without write access, and the
    job file, which the script would replace.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_dir() or not os.access(out if out.exists() else out.parent, os.W_OK):
        raise RunnerError(f"the run file {out} cannot be written: it is a folder, or its place has no write access")
    if out.exists() and out.samefile(job):
        raise RunnerError(f"--out names the job file ({out}): the script would replace its input")


def check_no_later_parts(out: Path) -> None:
    """Raise ``RunnerError`` when files with the names of later parts of a clip run lie next to ``out``.

    A clip run of fewer parts would leave them there, and the package would refuse the
    run as a mix of two runs. They are not removed here: the script deletes nothing.
    """
    found = later_parts(out)
    if found:
        raise RunnerError(
            f"{len(found)} files with the names of later parts of a clip run lie next to {out} (for example "
            f"{found[0].name}): remove them, or write the run to another place"
        )


def machine_facts(device: str) -> dict[str, Any]:
    """Return the facts of the run that do not depend on the tool: the device, the machine and the versions.

    ``script_sha256`` is the SHA-256 of this script: the command ``tool-score`` compares it
    with the script of its work tree.
    """
    return {
        "device": device,
        "machine": platform.machine(),
        "system": platform.system(),
        "python": platform.python_version(),
        "torch": installed_version("torch"),
        "script_sha256": file_sha256(Path(__file__).resolve()),
    }


def installed_version(package: str) -> str:
    """Return the version of an installed package. A package that is not installed raises ``RunnerError``."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError as error:
        raise RunnerError(f"the package {package} is not installed here. {WRONG_ENVIRONMENT}") from error


def tool_module(name: str) -> ModuleType:
    """Import a module of the tool or of its environment.

    A module that is not installed (the module or one of its parent packages) raises
    ``RunnerError``. Any other import error goes on as it is, with its traceback.
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name is None or not f"{name}.".startswith(f"{error.name}."):
            raise
        raise RunnerError(f"the module {name} cannot be imported: {error}. {WRONG_ENVIRONMENT}") from error


def truth_value(settings: Mapping[str, Any], key: str) -> bool:
    """Return a tool setting that must be true or false. Another value raises ``RunnerError``."""
    value = settings[key]
    if not isinstance(value, bool):
        raise RunnerError(f"the tool setting {key!r} must be true or false, got {value!r}")
    return value


# --- RF-DETR -------------------------------------------------------------------------


def rfdetr_engine(settings: Mapping[str, Any], device: str) -> tuple[dict[str, Any], Answer]:
    """Build the detector of a job. Return the facts of the tool and the function that answers one frame.

    ``RunnerError`` is raised for a version of ``rfdetr`` whose source was not read for this
    script, and for a model class that is not in ``RFDETR_MODELS``. The answer function raises
    ``RunnerError`` for a box or a score that is not a finite number.
    """
    version = installed_version("rfdetr")
    if version not in RFDETR_VERSIONS:
        raise RunnerError(
            f"this script was written for rfdetr {' and '.join(RFDETR_VERSIONS)}, and rfdetr {version} is installed. "
            "Install one of these versions. Or read what changed in predict() of the installed version, "
            "change the script where needed and add the version to RFDETR_VERSIONS"
        )
    model_name = settings["model"]
    if model_name not in RFDETR_MODELS:
        raise RunnerError(f"the model {model_name!r} is not one of {', '.join(RFDETR_MODELS)}")
    antialias = truth_value(settings, "antialias")
    model = getattr(tool_module("rfdetr"), model_name)(device=device)
    min_score = float(settings["min_score"])

    def answer(image: np.ndarray) -> dict[str, Any]:
        detections = model.predict(image, threshold=min_score, include_source_image=False, antialias=antialias)
        if not (np.isfinite(detections.xyxy).all() and np.isfinite(detections.confidence).all()):
            raise RunnerError("the detector gave a box or a score that is not a finite number")
        names = detections.data["class_name"]
        return {
            "detections": [
                {"box_xyxy": [float(value) for value in box], "score": float(score), "label": str(name)}
                for box, score, name in zip(detections.xyxy, detections.confidence, names, strict=True)
            ]
        }

    return {"tool_version": version, "predictor": "predict() of the model, not compiled"}, answer


# --- D-FINE and RT-DETRv2 through transformers -----------------------------------------


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Return the logistic function of each value, without an overflow for large negative values."""
    return np.exp(-np.logaddexp(0.0, -values))


def detections_of_queries(
    logits: np.ndarray,
    boxes: np.ndarray,
    *,
    width: int,
    height: int,
    labels: Mapping[int, str],
    min_score: float,
) -> dict[str, Any]:
    """Turn the output of a DETR detector for one frame into the payload of the frame.

    ``logits`` has the shape (queries, classes): the class scores before the logistic function
    (the weights of both families are trained with the focal loss, so each class has its own
    score). ``boxes`` has the shape (queries, 4): the centre x, the centre y, the width and the
    height of each box in parts of the model input, from 0 to 1. The model input is the whole
    frame resized to a square without padding, so a part of the input is the same part of the
    frame of ``width`` x ``height`` pixels.

    The steps are those of ``post_process_object_detection`` of the image processor of the tool
    (transformers 5.23.0, ``RTDetrImageProcessor``): the logistic function of every class
    score, the best ``queries`` pairs of query and class over all pairs, each with the box of
    its query as corners in pixels of the frame. A pair is kept when its score is at least
    ``min_score`` and its class name (``labels``) is one of ``KEPT_LABELS``; the tool keeps a
    score above its threshold, so the two differ only for a score equal to it. The detections
    are in the order of their scores, best first.

    ``RunnerError`` is raised for an output with a value that is not a finite number and for
    shapes that do not fit together or do not fit ``labels``.
    """
    if logits.ndim != 2 or boxes.shape != (logits.shape[0], 4):
        raise RunnerError(
            f"the detector gave scores of the shape {logits.shape} and boxes of the shape {boxes.shape}: "
            "expected (queries, classes) and (queries, 4)"
        )
    queries, classes = logits.shape
    if classes != len(labels):
        raise RunnerError(
            f"the detector gave {classes} class scores per query, and its weights name {len(labels)} classes"
        )
    if not (np.isfinite(logits).all() and np.isfinite(boxes).all()):
        raise RunnerError("the detector gave a box or a score that is not a finite number")
    scores = sigmoid(logits.astype(np.float64)).ravel()
    best = np.argsort(-scores, kind="stable")[:queries]
    scale = np.array([width, height, width, height], dtype=np.float64)
    detections = []
    for pair in best:
        score = float(scores[pair])
        if score < min_score:
            break
        query, label = divmod(int(pair), classes)
        name = labels[label]
        if name not in KEPT_LABELS:
            continue
        centre_x, centre_y, box_width, box_height = boxes[query].astype(np.float64)
        corners = np.array(
            [centre_x - box_width / 2, centre_y - box_height / 2, centre_x + box_width / 2, centre_y + box_height / 2]
        )
        detections.append({"box_xyxy": [float(value) for value in corners * scale], "score": score, "label": name})
    return {"detections": detections}


def tool_detections(
    result: Mapping[str, Any], labels: Mapping[int, str], min_score: float
) -> list[tuple[str, float, list[float]]]:
    """Return the detections of one frame as ``post_process_object_detection`` of the tool gave them.

    Only the classes of ``KEPT_LABELS`` are kept, and only scores above ``min_score`` (the tool
    keeps those), as (class name, score, box).
    """
    return [
        (labels[int(label)], float(score), [float(value) for value in box])
        for score, label, box in zip(result["scores"].tolist(), result["labels"].tolist(), result["boxes"].tolist())
        if labels[int(label)] in KEPT_LABELS and float(score) > min_score
    ]


def check_against_tool(
    payload: Mapping[str, Any], tool: list[tuple[str, float, list[float]]], min_score: float
) -> None:
    """Raise ``RunnerError`` unless the payload of the script holds the detections of the tool for one frame.

    The detections of the payload with a score equal to ``min_score`` are left out: the tool
    keeps only scores above its threshold. Each detection of the tool must match its own
    detection of the payload: the same class, boxes at most ``SELF_CHECK_BOX_PX`` pixels apart
    and scores at most ``SELF_CHECK_SCORE`` apart. The detections are matched by these
    tolerances, not by their order: two scores that are equal in float32 (the tool) can differ
    in float64 (the script), so the two orders can differ.
    """
    own = [
        (detection["label"], detection["score"], detection["box_xyxy"])
        for detection in payload["detections"]
        if detection["score"] > min_score
    ]

    def same(mine: tuple[str, float, list[float]], theirs: tuple[str, float, list[float]]) -> bool:
        return (
            mine[0] == theirs[0]
            and abs(mine[1] - theirs[1]) <= SELF_CHECK_SCORE
            and max(abs(a - b) for a, b in zip(mine[2], theirs[2], strict=True)) <= SELF_CHECK_BOX_PX
        )

    unmatched = list(own)
    for theirs in tool:
        match = next((position for position, mine in enumerate(unmatched) if same(mine, theirs)), None)
        if match is None:
            break
        unmatched.pop(match)
    else:
        if not unmatched:
            return
    raise RunnerError(
        f"the boxes of the script differ from those of post_process_object_detection of the tool on the first "
        f"frame ({len(own)} against {len(tool)} detections of {', '.join(KEPT_LABELS)}): the output of the "
        "installed version is not the one that the script was written for. Read what changed and change "
        "detections_of_queries"
    )


def synchronize(torch: ModuleType, device: str) -> None:
    """Wait until the device has done the work that was given to it, so that a clock reads its time."""
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elif device.startswith("mps"):
        torch.mps.synchronize()


def transformers_engine(
    family: str, settings: Mapping[str, Any], device: str
) -> tuple[dict[str, Any], BatchAnswer, Callable[[], dict[str, Any]]]:
    """Build a detector of transformers for a job. Return its facts, the batch answer and the facts of the run.

    The facts of the run are known only after the run: the median time per frame of the image
    processor, of the model and of the conversion of its output (``stage_ms_per_frame``). The
    time of the model is read after the device has done its work (``synchronize``).

    ``RunnerError`` is raised for a version of ``transformers`` whose source was not read for
    this script, for a size that has no checkpoint in ``TRANSFORMERS_CHECKPOINTS``, for half
    precision on the device ``cpu``, and for a checkpoint whose configuration is not one of the
    family (model type, focal loss, the classes of ``KEPT_LABELS``). The weights are loaded
    only from a safetensors file: a downloaded file must not run code.
    """
    version = installed_version("transformers")
    if version not in TRANSFORMERS_VERSIONS:
        raise RunnerError(
            f"this script was written for transformers {' and '.join(TRANSFORMERS_VERSIONS)}, and transformers "
            f"{version} is installed. Install one of these versions. Or read what changed in the image processor "
            "and the output of the detector, change the script where needed and add the version to "
            "TRANSFORMERS_VERSIONS"
        )
    checkpoints = TRANSFORMERS_CHECKPOINTS[family]
    size = settings["model"]
    if size not in checkpoints:
        raise RunnerError(f"the size {size!r} of {family} is not one of {', '.join(checkpoints)}")
    half = truth_value(settings, "half")
    if half and device == "cpu":
        raise RunnerError("half precision needs a GPU: set half to false for the device cpu")
    input_size, batch, min_score = int(settings["input_size"]), int(settings["batch"]), float(settings["min_score"])
    repository, revision = checkpoints[size]
    torch = tool_module("torch")
    transformers = tool_module("transformers")
    dtype = torch.float16 if half else torch.float32
    processor = transformers.AutoImageProcessor.from_pretrained(repository, revision=revision)
    model = transformers.AutoModelForObjectDetection.from_pretrained(
        repository, revision=revision, dtype=dtype, use_safetensors=True
    )
    model.to(device)
    model.eval()
    config = model.config
    if config.model_type != TRANSFORMERS_MODEL_TYPES[family]:
        raise RunnerError(f"the checkpoint {repository} is of the model type {config.model_type!r}, not of {family}")
    if getattr(config, "use_focal_loss", None) is not True:
        raise RunnerError(
            f"the checkpoint {repository} is not trained with the focal loss: the script reads only such scores"
        )
    labels = {int(index): str(name) for index, name in config.id2label.items()}
    missing = [name for name in KEPT_LABELS if name not in labels.values()]
    if missing:
        raise RunnerError(f"the weights of {repository} have no class {', '.join(missing)}")
    times: list[tuple[float, float, float, int]] = []
    checked = False

    def answer(images: Sequence[np.ndarray]) -> list[dict[str, Any]]:
        nonlocal checked
        start = time.perf_counter()
        inputs = processor(images=list(images), size={"height": input_size, "width": input_size}, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
        synchronize(torch, device)
        prepared = time.perf_counter()
        with torch.inference_mode():
            outputs = model(pixel_values=pixel_values)
        synchronize(torch, device)
        modelled = time.perf_counter()
        logits = outputs.logits.float().cpu().numpy()
        boxes = outputs.pred_boxes.float().cpu().numpy()
        payloads = [
            detections_of_queries(
                logits[index],
                boxes[index],
                width=image.shape[1],
                height=image.shape[0],
                labels=labels,
                min_score=min_score,
            )
            for index, image in enumerate(images)
        ]
        done = time.perf_counter()
        if not checked:
            # The first call is the warm-up call, which is not measured. Check the boxes of the script
            # against those of the tool on it.
            # The function of the tool reads only these two fields, and it gets them as float32 like the script.
            output = SimpleNamespace(logits=outputs.logits.float(), pred_boxes=outputs.pred_boxes.float())
            sizes = [(image.shape[0], image.shape[1]) for image in images]
            results = processor.post_process_object_detection(output, threshold=min_score, target_sizes=sizes)
            for payload, result in zip(payloads, results, strict=True):
                check_against_tool(payload, tool_detections(result, labels, min_score), min_score)
            checked = True
        else:
            times.append((prepared - start, modelled - prepared, done - modelled, len(images)))
        return payloads

    def run_facts() -> dict[str, Any]:
        per_frame = {
            name: [part[position] / part[3] for part in times for _ in range(part[3])]
            for position, name in enumerate(("preprocess", "model", "postprocess"))
        }
        return {
            "stage_ms_per_frame": {
                name: round(1000 * statistics.median(values), 4) for name, values in per_frame.items()
            }
        }

    facts = {
        "tool_version": version,
        "checkpoint": repository,
        "checkpoint_revision": revision,
        "model_type": config.model_type,
        "dtype": "float16" if half else "float32",
        "predictor": "the model of AutoModelForObjectDetection in eager mode, not compiled",
    }
    return facts, answer, run_facts


# --- PnLCalib ------------------------------------------------------------------------


def camera_homography(cam_params: Mapping[str, Any]) -> np.ndarray:
    """Return the mapping of the grass from pitch metres to image pixels for a camera of PnLCalib.

    ``cam_params`` holds the focal lengths and the principal point in pixels of the frame,
    the place of the camera in the world frame in metres, and the rotation from the world
    frame to the camera. The projection of a world point is ``K @ R @ (X - C)``. A point on
    the grass has the height 0, so its mapping is the projection matrix without the column
    of the height. The result has the shape (3, 3) and a free scale.
    """
    intrinsics = np.array(
        [
            [cam_params["x_focal_length"], 0.0, cam_params["principal_point"][0]],
            [0.0, cam_params["y_focal_length"], cam_params["principal_point"][1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rotation = np.array(cam_params["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    position = np.array(cam_params["position_meters"], dtype=np.float64).reshape(3)
    shift = np.hstack([np.eye(3), -position.reshape(3, 1)])
    projection = intrinsics @ rotation @ shift
    return projection[:, [0, 1, 3]]


def calibration_payload(result: Mapping[str, Any] | None) -> dict[str, Any]:
    """Turn the result of the calibration of one frame into the payload of the frame.

    ``result`` is what ``heuristic_voting`` of the tool returns: None when it found no camera,
    else a mapping with the camera (``cam_params``) and the reprojection error of its
    keypoints in pixels (``rep_err``). A frame without a camera, and a camera whose mapping
    has a value that is not finite, give a payload without a matrix; the key ``fault`` then
    says why. The key ``rep_err`` is None when the tool gave no finite error.

    The payload of a camera also holds facts that the adapter of the package does not read:
    the setting that won the vote of the tool (``mode``, ``use_ransac``), the plane that the
    camera was calibrated on (``calib_plane``: 0 is the grass) and the place of the camera in
    pitch metres (``camera_position_m``). A main camera stands on the near side above the
    grass: y above 34 and z below 0.
    """
    if result is None:
        return {"homography_pitch_to_image": None, "fault": "the tool found no camera"}
    error = float(result["rep_err"])
    rep_err = error if np.isfinite(error) else None
    homography = camera_homography(result["cam_params"])
    if not np.isfinite(homography).all():
        return {
            "homography_pitch_to_image": None,
            "fault": "the camera of the tool has a value that is not finite",
            "rep_err": rep_err,
        }
    return {
        "homography_pitch_to_image": homography.tolist(),
        "pitch_origin": PITCH_ORIGIN,
        "rep_err": rep_err,
        "mode": str(result["mode"]),
        "use_ransac": float(result["use_ransac"]),
        "calib_plane": int(result["calib_plane"]),
        "camera_position_m": [float(value) for value in np.ravel(result["cam_params"]["position_meters"])],
    }


def clone_commit(repo: Path) -> str:
    """Return the commit that a clone of a git repository has checked out. No clone raises ``RunnerError``."""
    done = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        raise RunnerError(f"{repo} is not a git clone with a commit: {' '.join(done.stderr.split())}")
    return done.stdout.strip()


def clone_changes(repo: Path) -> list[str]:
    """Return the tracked files of a clone that differ from its commit, as the lines of ``git status``.

    Files that git does not track are left out: the clone has no ignore file, so the bytecode
    folders that an import makes are such files.
    """
    done = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise RunnerError(f"the state of the clone {repo} cannot be read: {' '.join(done.stderr.split())}")
    return done.stdout.splitlines()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of a file as hexadecimal text."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_clone_packages(repo: Path) -> None:
    """Make ``model`` and ``utils`` the packages of the two folders with these names in the clone.

    The clone has no install: its files import each other as ``model.<file>`` and
    ``utils.<file>``. Python would take an installed package with one of these two names
    before the folder of the clone, also when the clone comes first on the module path. So
    each name is bound to its folder of the clone here, before anything is imported from it.
    """
    for name in PNLCALIB_PACKAGES:
        folder = repo / name
        if not folder.is_dir():
            raise RunnerError(f"{repo} has no folder {name}: it is not a clone of the PnLCalib repository")
        package = ModuleType(name)
        package.__path__ = [str(folder)]
        sys.modules[name] = package


def pnlcalib_engine(
    settings: Mapping[str, Any], device: str, repo: Path, weights_kp: Path, weights_lines: Path
) -> tuple[dict[str, Any], Answer]:
    """Build the calibration of a job. Return the facts of the tool and the function that answers one frame.

    ``repo`` is a clone of the PnLCalib repository. It must have the commit
    ``PNLCALIB_COMMIT`` checked out, and no tracked file may differ from that commit, else
    ``RunnerError`` is raised. The two weight files are
    loaded as plain weights (``weights_only=True``): a downloaded file must not run code.

    The function follows the steps of ``inference.py`` of the clone for one frame: the frame
    as RGB values from 0 to 1, resized to the input size of the models unless it has that
    width; the keypoint model and the line model; the points of the heatmaps above the two
    thresholds of the job; the keypoints completed from the lines and scaled to the size of
    the frame; the calibration of the frame, with the line refinement when the job asks
    for it. Each frame gets its own calibration object, so no state goes from one frame to
    the next.
    """
    repo = repo.resolve()
    commit = clone_commit(repo)
    if commit != PNLCALIB_COMMIT:
        raise RunnerError(
            f"this script was written for the commit {PNLCALIB_COMMIT} of PnLCalib, and {repo} has {commit}. "
            "Check out that commit. Or read what changed in inference.py of the clone, change the script "
            "where needed and set PNLCALIB_COMMIT"
        )
    changes = clone_changes(repo)
    if changes:
        raise RunnerError(
            f"{len(changes)} tracked files of {repo} differ from the commit {commit} "
            f"(for example: {changes[0].strip()}). The run file names the commit, so the files must be "
            "those of the commit: undo the changes in the clone"
        )
    bind_clone_packages(repo)
    # A file of the clone imports the plot library. Without a display backend it opens no window.
    os.environ.setdefault("MPLBACKEND", "Agg")
    torch = tool_module("torch")
    yaml = tool_module("yaml")
    transforms = tool_module("torchvision.transforms")
    to_tensor = tool_module("torchvision.transforms.functional").to_tensor
    get_cls_net = tool_module("model.cls_hrnet").get_cls_net
    get_cls_net_lines = tool_module("model.cls_hrnet_l").get_cls_net
    calibration_class = tool_module("utils.utils_calib").FramebyFrameCalib
    heatmap = tool_module("utils.utils_heatmap")

    def load_model(build: Callable[[Any], Any], config_name: str, weights: Path) -> Any:
        config = yaml.safe_load((repo / "config" / config_name).read_text(encoding="utf-8"))
        model = build(config)
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        model.to(device)
        model.eval()
        return model

    keypoint_model = load_model(get_cls_net, "hrnetv2_w48.yaml", weights_kp)
    line_model = load_model(get_cls_net_lines, "hrnetv2_w48_l.yaml", weights_lines)
    input_width, input_height = PNLCALIB_INPUT
    resize = transforms.Resize((input_height, input_width))
    kp_threshold, line_threshold = float(settings["kp_threshold"]), float(settings["line_threshold"])
    pnl_refine = truth_value(settings, "pnl_refine")

    def answer(image: np.ndarray) -> dict[str, Any]:
        frame_height, frame_width = image.shape[:2]
        frame = to_tensor(image).float().unsqueeze(0)
        if frame.size()[-1] != input_width:
            frame = resize(frame)
        frame = frame.to(device)
        _, _, height, width = frame.size()
        with torch.no_grad():
            heatmaps = keypoint_model(frame)
            line_heatmaps = line_model(frame)
        keypoints = heatmap.get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
        lines = heatmap.get_keypoints_from_heatmap_batch_maxpool_l(line_heatmaps[:, :-1, :, :])
        keypoints_of_frame = heatmap.coords_to_dict(keypoints, threshold=kp_threshold)[0]
        lines_of_frame = heatmap.coords_to_dict(lines, threshold=line_threshold)[0]
        keypoints_of_frame, lines_of_frame = heatmap.complete_keypoints(
            keypoints_of_frame, lines_of_frame, w=width, h=height, normalize=True
        )
        calibration = calibration_class(iwidth=frame_width, iheight=frame_height, denormalize=True)
        calibration.update(keypoints_of_frame, lines_of_frame)
        return calibration_payload(calibration.heuristic_voting(refine_lines=pnl_refine))

    facts = {
        "tool_commit": commit,
        "weights_kp": weights_kp.name,
        "weights_kp_sha256": file_sha256(weights_kp),
        "weights_lines": weights_lines.name,
        "weights_lines_sha256": file_sha256(weights_lines),
        "calibration_size": "the size of the frame, as in inference.py",
    }
    return facts, answer


# --- Command line --------------------------------------------------------------------


def _path(text: str) -> Path:
    """Turn a command-line value into a path. A leading ``~`` becomes the home folder."""
    return Path(text).expanduser()


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser: one sub-command per tool."""
    parser = argparse.ArgumentParser(
        prog="run_tool.py",
        description="Run an external tool on the frames of a job file and write the run file.",
        epilog="Exit code 0 means that the run file was written, 2 means that the script could not run.",
    )
    tools = parser.add_subparsers(dest="tool", metavar="TOOL", required=True)
    for name, text in (
        ("rfdetr", "the RF-DETR detector"),
        ("dfine", "the D-FINE detector through transformers"),
        ("rtdetrv2", "the RT-DETRv2 detector through transformers"),
        ("pnlcalib", "the PnLCalib calibration"),
    ):
        tool = tools.add_parser(name, help=text)
        tool.add_argument(
            "--job", type=_path, required=True, metavar="FILE", help="job file of the tool-job or the clip-job command"
        )
        tool.add_argument("--root", type=_path, required=True, metavar="DIR", help="data folder")
        tool.add_argument(
            "--out",
            type=_path,
            required=True,
            metavar="FILE",
            help="run file to write (JSON); the first part of a clip run",
        )
        tool.add_argument(
            "--device", required=True, metavar="NAME", help="device name of PyTorch, for example mps, cpu or cuda:0"
        )
        tool.add_argument(
            "--feed",
            metavar="FILE",
            help="live run of a clip job: write the tool feed to this file or named pipe (- for the standard output), "
            "one line per frame as it is answered; --out then gets a copy of the feed instead of a clip run",
        )
        tool.add_argument(
            "--pace", type=float, metavar="FPS", help="with --feed: frame i comes at the start plus i / FPS seconds"
        )
        tool.add_argument(
            "--start-at", type=float, metavar="SECONDS", help="with --pace: the start, in seconds since the epoch"
        )
        tool.add_argument(
            "--skip-behind",
            action="store_true",
            help="with --pace: take the newest frame that has come, and skip the frames that the tool did not reach",
        )
        if name == "pnlcalib":
            tool.add_argument(
                "--repo", type=_path, required=True, metavar="DIR", help="clone of the PnLCalib repository"
            )
            tool.add_argument(
                "--weights-kp", type=_path, required=True, metavar="FILE", help="weights of the keypoint model"
            )
            tool.add_argument(
                "--weights-lines", type=_path, required=True, metavar="FILE", help="weights of the line model"
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the script and return the exit code. ``argv`` defaults to the arguments of the process."""
    args = build_parser().parse_args(argv)
    # The tools are code of other projects. They get no key of a model API.
    for variable in KEY_VARIABLES:
        os.environ.pop(variable, None)
    try:
        if sys.version_info < (3, 10):
            raise RunnerError(f"this script needs Python 3.10 or newer, and this is Python {platform.python_version()}")
        job = read_job(args.job, args.tool)
        clip_job = job["format"] == CLIP_JOB_FORMAT
        live = args.feed is not None
        if live and not clip_job:
            raise RunnerError("--feed needs a clip job: a live run is a run of one clip")
        if not live and (args.pace is not None or args.start_at is not None or args.skip_behind):
            raise RunnerError("--pace, --start-at and --skip-behind need --feed")
        if args.skip_behind and int(job["tool"].get("batch", 1)) != 1:
            raise RunnerError(
                f"--skip-behind needs batches of one frame, and the job has batches of {job['tool']['batch']}"
            )
        check_out_place(args.out, args.job)
        if clip_job and not live:
            check_no_later_parts(args.out)
        job_sha256 = file_sha256(args.job)
        machine = machine_facts(args.device)
        batch, run_facts = 1, dict
        if args.tool == "rfdetr":
            facts, single = rfdetr_engine(job["tool"], args.device)
            answer = one_at_a_time(single)
        elif args.tool in TRANSFORMERS_CHECKPOINTS:
            facts, answer, run_facts = transformers_engine(args.tool, job["tool"], args.device)
            batch = int(job["tool"]["batch"])
        else:
            facts, single = pnlcalib_engine(job["tool"], args.device, args.repo, args.weights_kp, args.weights_lines)
            answer = one_at_a_time(single)
        skipped = 0
        if live:
            frames, skipped = run_live(args, job, job_sha256, answer, batch)
        else:
            frames = answer_frames(job, args.root, answer, batch)
        runner = {**machine, **facts, **run_facts()}
        written = [args.out]
        if clip_job and not live:
            written = write_clip_run(args.out, job, job_sha256, runner, frames)
        elif not live:
            write_run(args.out, job, runner, frames)
    except (RunnerError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as error:
        traceback.print_exc()
        print(f"error: the script stopped with {type(error).__name__}; the traceback above says where", file=sys.stderr)
        return EXIT_ERROR
    # The standard output of a live run to "-" is the feed: the report goes to the error stream then.
    report = sys.stderr if args.feed == "-" else sys.stdout
    median = statistics.median(entry["seconds"] for entry in frames) if frames else float("nan")
    print(
        f"{args.tool}: {len(frames)} frames of the job {job['name']}, "
        f"median time per frame {median:.3f} s on {args.device}",
        file=report,
    )
    if "stage_ms_per_frame" in runner:
        stages = ", ".join(f"{name} {value} ms" for name, value in runner["stage_ms_per_frame"].items())
        print(f"median time per frame of each part: {stages}", file=report)
    if live:
        print(f"live run: {skipped} frames skipped; copy of the feed: {args.out}", file=report)
    else:
        print(f"run file: {args.out}" + (f" and {len(written) - 1} more parts" if len(written) > 1 else ""))
    return EXIT_OK


def run_live(
    args: argparse.Namespace, job: Mapping[str, Any], job_sha256: str, answer: BatchAnswer, batch: int
) -> tuple[list[dict[str, Any]], int]:
    """Answer the frames of a clip job as a live run: write the feed and its copy at ``--out`` as frames are answered.

    Returns the answered entries and the number of skipped frames. The feed is opened after the
    tool is built, so a named pipe waits for its reader only then.
    """
    head = {
        "clip": job["clip"],
        "n_frames": len(job["frames"]),
        "video_sha256": job["video_sha256"],
        "job_sha256": job_sha256,
        "tool": job["tool"],
    }
    with args.out.open("w", encoding="utf-8") as copy:
        if args.feed == "-":
            feed = LiveFeed(
                [copy, sys.stdout], head, pace=args.pace, start_at=args.start_at, skip_behind=args.skip_behind
            )
            frames = answer_frames(job, args.root, answer, batch, live=feed)
        else:
            with open(args.feed, "w", encoding="utf-8") as target:
                feed = LiveFeed([copy, target], head, pace=args.pace, start_at=args.start_at, skip_behind=args.skip_behind)
                frames = answer_frames(job, args.root, answer, batch, live=feed)
    return frames, feed.skipped


if __name__ == "__main__":
    sys.exit(main())
