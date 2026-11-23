"""The two files that connect this package with the script that runs an external tool.

An external tool is measured on the frames of a probe in three steps:

1. This package writes a **job file**: the frames that the tool must read, and its settings.
2. The script ``tools/run_tool.py`` runs the tool on these frames, in the environment of the
   tool, and writes a **run file**: the answer of the tool for each frame.
3. This package reads the run file and measures the answers against the labels.

The script does not import this package, because the environment of a tool has other
packages. So both files are plain JSON, and this module holds their format. The script holds
the same two format names, and a test compares them.

Job file::

    {
      "format": "pitchvlm-tool-job-1",
      "name": "people_probe_rfdetr",
      "tool": {"name": "rfdetr", "model": "RFDETRSmall", "min_score": 0.5, "antialias": false},
      "frames": [
        {"clip": "SNGS-021", "frame": 37, "image": "SoccerNetGS-valid/SNGS-021/img1/000038.jpg"}
      ]
    }

- ``name``: the name of the experiment.
- ``tool``: the tool and its settings, as the experiment file gives them (``RfdetrSettings`` or
  ``PnlcalibSettings``).
- ``frames``: one entry per frame. ``frame`` is the index of the frame in its clip, from 0.
  ``image`` is the path of the image file below the data folder, with forward slashes.

Run file::

    {
      "format": "pitchvlm-tool-run-1",
      "name": "people_probe_rfdetr",
      "tool": {"name": "rfdetr", "model": "RFDETRSmall", "min_score": 0.5, "antialias": false},
      "runner": {"device": "mps", "tool_version": "1.12.0"},
      "frames": [
        {"clip": "SNGS-021", "frame": 37, "seconds": 0.021, "payload": {"detections": []}}
      ]
    }

- ``name`` and ``tool``: copied from the job file. The measurement refuses a run file whose
  name or tool settings differ from the experiment file.
- ``runner``: facts of the run that the script reports, for example the device and the version
  of the tool. They go into the metrics file as they are.
- ``frames``: one entry per frame of the job. ``seconds`` is the time that the tool needed for
  the frame, without the time to read the image file. ``payload`` is the answer of the tool in
  the format of its adapter module (``fallbacks.rfdetr`` or ``fallbacks.pnlcalib``).

Both files hold no pixel and no label data, and no absolute path.

A tool also runs on **every frame of one clip**, for the detector engine of the pipeline
(command ``run``). The two files of that use are the clip job and the clip run:

Clip job::

    {
      "format": "pitchvlm-clip-job-1",
      "clip": "SNGS-021",
      "tool": {"name": "rfdetr", "model": "RFDETRMedium", "min_score": 0.5, "antialias": false},
      "images": ["SoccerNetGS-valid/SNGS-021/img1/000001.jpg", "..."]
    }

- ``images``: the image file of every frame of the clip, below the data folder, with forward
  slashes. Frame ``i`` is ``images[i]``.

Clip run: one or more part files. The first part is the file that the script was told to
write, for example ``detections.json``; part ``k`` from 2 on has the name of the first part
with ``-<k as 4 digits>`` before the ending: ``detections-0002.json``. Each part is at most
``CLIP_RUN_PART_MAX_BYTES`` (1,000,000 bytes, half of the size limit of the guard), so each
part passes the guard like a ledger file. A part::

    {
      "format": "pitchvlm-clip-run-1",
      "clip": "SNGS-021",
      "tool": {"name": "rfdetr", "model": "RFDETRMedium", "min_score": 0.5, "antialias": false},
      "n_frames": 750,
      "job_sha256": "<SHA-256 of the clip job file>",
      "runner": {"device": "cuda:0", "tool_version": "1.12.1"},
      "part": 1,
      "parts": 2,
      "frames": [[0, 0.0213, {"labels": ["person", "sports ball"], "rows": [[x1, y1, x2, y2, score, 0]]}]]
    }

- Every part repeats the head (all keys but ``part`` and ``frames``). ``job_sha256`` ties
  the parts of one run together.
- ``frames``: ``[frame, seconds, payload]`` per frame, in frame order over the parts. The
  parts together hold every frame from 0 to ``n_frames - 1`` exactly once.
- The payload of the calibration is the payload of ``fallbacks.pnlcalib``. The payload of the
  detector is compact: ``labels`` names the classes of the frame, and a row is a box in
  pixels of the full frame (x1, y1, x2, y2, rounded to 0.1 pixel), the score (rounded to
  0.0001) and the place of the class name in ``labels``. ``fallbacks.rfdetr.expand_rows``
  turns it into the payload of that module. A detection of 30 people and a ball is about
  40 bytes, so a clip of 30 seconds at 25 frames per second with 25 people is about
  0.75 MB, one part.
"""

from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, StrictBool, StrictInt, model_validator

from pitchvlm.schemas import FiniteFloat, Record

#: Format names of the two files. The number goes up when a format changes.
JOB_FORMAT = "pitchvlm-tool-job-1"
RUN_FORMAT = "pitchvlm-tool-run-1"
#: Format names of the two files of a tool that runs on every frame of one clip.
CLIP_JOB_FORMAT = "pitchvlm-clip-job-1"
CLIP_RUN_FORMAT = "pitchvlm-clip-run-1"
#: Largest size of one part of a clip run, bytes: half of the size limit of the guard.
CLIP_RUN_PART_MAX_BYTES = 1_000_000


#: Model classes of the ``rfdetr`` package that come with pretrained weights for the 80 everyday classes.
RfdetrModel = Literal["RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge"]


class RfdetrSettings(Record):
    """Settings of the RF-DETR detector for one measurement."""

    name: Literal["rfdetr"]
    #: Class of the ``rfdetr`` package that builds the detector with its pretrained weights.
    model: RfdetrModel
    #: A detection counts when its score is at least this value. The tool gets the same value as its threshold.
    min_score: float = Field(strict=True, ge=0.0, le=1.0)
    #: True when the tool makes the frame smaller with antialiasing. By itself the tool uses none.
    #: Only true and false are taken: the script that runs the tool refuses any other value.
    antialias: StrictBool


class PnlcalibSettings(Record):
    """Settings of the PnLCalib calibration tool for one measurement."""

    name: Literal["pnlcalib"]
    #: Score from which the tool uses a keypoint of its keypoint model.
    kp_threshold: float = Field(strict=True, ge=0.0, le=1.0)
    #: Score from which the tool uses a line of its line model.
    line_threshold: float = Field(strict=True, ge=0.0, le=1.0)
    #: True when the tool refines the camera with the lines after it found it from the keypoints.
    #: Only true and false are taken, as for ``RfdetrSettings.antialias``.
    pnl_refine: StrictBool


#: The settings of one tool. The key ``name`` says which tool it is.
ToolSettings = Annotated[RfdetrSettings | PnlcalibSettings, Field(discriminator="name")]


class JobFrame(Record):
    """One frame that the tool must read."""

    clip: str = Field(min_length=1)
    #: Index of the frame in its clip, from 0.
    frame: StrictInt = Field(ge=0)
    #: Path of the image file below the data folder, with forward slashes.
    image: str = Field(min_length=1)


class ToolJob(Record):
    """The job file: the frames that a tool must read, and its settings."""

    format: Literal["pitchvlm-tool-job-1"]
    #: Name of the experiment.
    name: str = Field(min_length=1)
    tool: ToolSettings
    frames: tuple[JobFrame, ...] = Field(min_length=1)


class RunFrame(Record):
    """The answer of the tool for one frame."""

    clip: str = Field(min_length=1)
    frame: StrictInt = Field(ge=0)
    #: Time that the tool needed for the frame, seconds, without the time to read the image file.
    seconds: FiniteFloat = Field(strict=True, ge=0.0)
    #: The answer in the format of the adapter module of the tool.
    payload: dict[str, Any]


class ToolRun(Record):
    """The run file: the answers of a tool for the frames of a job."""

    format: Literal["pitchvlm-tool-run-1"]
    name: str = Field(min_length=1)
    tool: ToolSettings
    #: Facts of the run that the script reports, as plain JSON types.
    runner: dict[str, Any]
    frames: tuple[RunFrame, ...]

    @model_validator(mode="after")
    def _check_frames(self) -> ToolRun:
        counts = Counter((answer.clip, answer.frame) for answer in self.frames)
        twice = sorted(key for key, count in counts.items() if count > 1)
        if twice:
            names = ", ".join(f"{clip} frame {frame}" for clip, frame in twice[:5])
            raise ValueError(f"{len(twice)} frames have more than one answer, for example: {names}")
        return self

    def answers(self) -> dict[tuple[str, int], RunFrame]:
        """Return the answer of each frame by (clip id, frame)."""
        return {(answer.clip, answer.frame): answer for answer in self.frames}


def load_tool_run(path: Path) -> ToolRun:
    """Read a run file.

    ``ValueError`` is raised for a file that holds no JSON, for a JSON object that has a key
    two times, and for content that does not follow the format of the module docstring. The
    numbers of the file are read as they are written: text or a truth value is no number.
    The payloads are not checked here: the adapter of the tool reads them.
    """
    return ToolRun.model_validate(json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_keys))


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Turn the key-value pairs of a JSON object into a mapping. A key that comes twice raises ``ValueError``."""
    mapping = dict(pairs)
    if len(mapping) != len(pairs):
        twice = sorted(key for key, count in Counter(key for key, _ in pairs).items() if count > 1)
        raise ValueError(f"these keys of one JSON object come more than once: {twice}")
    return mapping


# --- Clip job and clip run -----------------------------------------------------------


class ClipJob(Record):
    """The clip job: the image of every frame of one clip, and the settings of the tool."""

    format: Literal["pitchvlm-clip-job-1"]
    clip: str = Field(min_length=1)
    tool: ToolSettings
    #: Image file of every frame below the data folder, with forward slashes. Frame i is ``images[i]``.
    images: tuple[str, ...] = Field(min_length=1)


class ClipRunFrame(Record):
    """The answer of the tool for one frame of a clip."""

    frame: StrictInt = Field(ge=0)
    #: Time that the tool needed for the frame, seconds, without the time to read the image file.
    seconds: FiniteFloat = Field(strict=True, ge=0.0)
    #: The answer in the format of the adapter module of the tool (compact for the detector).
    payload: dict[str, Any]


class ClipRun(Record):
    """The answers of a tool for every frame of one clip: all parts of a clip run together."""

    clip: str = Field(min_length=1)
    tool: ToolSettings
    n_frames: StrictInt = Field(gt=0)
    job_sha256: str = Field(min_length=1)
    #: Facts of the run that the script reports, as plain JSON types.
    runner: dict[str, Any]
    #: One answer per frame, in frame order: ``frames[i].frame == i``.
    frames: tuple[ClipRunFrame, ...]

    @model_validator(mode="after")
    def _check_every_frame(self) -> ClipRun:
        numbers = [answer.frame for answer in self.frames]
        if numbers != list(range(self.n_frames)):
            missing = sorted(set(range(self.n_frames)) - set(numbers))
            twice = sorted(frame for frame, count in Counter(numbers).items() if count > 1)
            raise ValueError(
                f"the parts must hold the frames 0 to {self.n_frames - 1} once each, in order; "
                f"{len(missing)} frames are missing (first: {missing[:3]}), {len(twice)} come more than once "
                f"(first: {twice[:3]})"
            )
        return self

    def seconds(self) -> list[float]:
        """Return the time of the tool for each frame, in frame order."""
        return [answer.seconds for answer in self.frames]


class _ClipRunPart(Record):
    """One part file of a clip run, as it is written."""

    format: Literal["pitchvlm-clip-run-1"]
    clip: str = Field(min_length=1)
    tool: ToolSettings
    n_frames: StrictInt = Field(gt=0)
    job_sha256: str = Field(min_length=1)
    runner: dict[str, Any]
    part: StrictInt = Field(ge=1)
    parts: StrictInt = Field(ge=1)
    #: ``[frame, seconds, payload]`` per frame; ``ClipRunFrame`` checks each value strictly.
    frames: tuple[tuple[Any, Any, Any], ...]

    def head(self) -> dict[str, Any]:
        """Return the keys that every part of one run repeats."""
        return self.model_dump(mode="json", exclude={"part", "frames"})


def clip_run_part_path(first: Path, part: int) -> Path:
    """Return the file of one part of a clip run. Part 1 is ``first``; part k is ``<stem>-<k:04d><suffix>``."""
    if part < 1:
        raise ValueError(f"a part number starts at 1, got {part}")
    return first if part == 1 else first.with_name(f"{first.stem}-{part:04d}{first.suffix}")


def load_clip_run(first: Path) -> ClipRun:
    """Read all parts of a clip run, starting with the first part file.

    ``ValueError`` is raised for a part that holds no JSON or that has a key two times, for a
    part that does not follow the format of the module docstring, for a missing part, for a
    part whose head differs from the head of the first part, for a part file with a number
    above the number of parts of the run (a part of an older run), and when the parts do not
    hold every frame of the clip once. ``OSError`` is raised for a part that cannot be read.
    """
    first_part = _read_part(first)
    if first_part.part != 1:
        raise ValueError(f"{first.name}: the first file of a clip run must be part 1, it is part {first_part.part}")
    head = first_part.head()
    frames = list(first_part.frames)
    for number in range(2, first_part.parts + 1):
        path = clip_run_part_path(first, number)
        if not path.is_file():
            raise ValueError(f"{path.name}: part {number} of {first_part.parts} of the clip run is missing")
        part = _read_part(path)
        if part.part != number or part.head() != head:
            raise ValueError(
                f"{path.name}: this file is not part {number} of the run of {first.name} (its part number or its head "
                "differs)"
            )
        frames.extend(part.frames)
    pattern = f"{glob.escape(first.stem)}-[0-9][0-9][0-9][0-9]{glob.escape(first.suffix)}"
    strays = sorted(path for path in first.parent.glob(pattern) if int(path.stem[-4:]) > first_part.parts)
    if strays:
        raise ValueError(
            f"{strays[0].name} lies next to a run of {first_part.parts} parts: it is a part of another run. Remove it"
        )
    return ClipRun.model_validate(
        {
            **{key: head[key] for key in ("clip", "tool", "n_frames", "job_sha256", "runner")},
            "frames": [{"frame": frame, "seconds": seconds, "payload": payload} for frame, seconds, payload in frames],
        }
    )


def _read_part(path: Path) -> _ClipRunPart:
    """Read one part file of a clip run strictly (``load_tool_run`` says how)."""
    document = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    try:
        return _ClipRunPart.model_validate(document)
    except ValueError as error:
        raise ValueError(f"{Path(path).name}: {error}") from error
