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
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, StrictBool, model_validator

from pitchvlm.schemas import FiniteFloat, Record

#: Format names of the two files. The number goes up when a format changes.
JOB_FORMAT = "pitchvlm-tool-job-1"
RUN_FORMAT = "pitchvlm-tool-run-1"


#: Model classes of the ``rfdetr`` package that come with pretrained weights for the 80 everyday classes.
RfdetrModel = Literal["RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge"]


class RfdetrSettings(Record):
    """Settings of the RF-DETR detector for one measurement."""

    name: Literal["rfdetr"]
    #: Class of the ``rfdetr`` package that builds the detector with its pretrained weights.
    model: RfdetrModel
    #: A detection counts when its score is at least this value. The tool gets the same value as its threshold.
    min_score: float = Field(ge=0.0, le=1.0)
    #: True when the tool makes the frame smaller with antialiasing. By itself the tool uses none.
    #: Only true and false are taken: the script that runs the tool refuses any other value.
    antialias: StrictBool


class PnlcalibSettings(Record):
    """Settings of the PnLCalib calibration tool for one measurement."""

    name: Literal["pnlcalib"]
    #: Score from which the tool uses a keypoint of its keypoint model.
    kp_threshold: float = Field(ge=0.0, le=1.0)
    #: Score from which the tool uses a line of its line model.
    line_threshold: float = Field(ge=0.0, le=1.0)
    #: True when the tool refines the camera with the lines after it found it from the keypoints.
    #: Only true and false are taken, as for ``RfdetrSettings.antialias``.
    pnl_refine: StrictBool


#: The settings of one tool. The key ``name`` says which tool it is.
ToolSettings = Annotated[RfdetrSettings | PnlcalibSettings, Field(discriminator="name")]


class JobFrame(Record):
    """One frame that the tool must read."""

    clip: str = Field(min_length=1)
    #: Index of the frame in its clip, from 0.
    frame: int = Field(ge=0)
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
    frame: int = Field(ge=0)
    #: Time that the tool needed for the frame, seconds, without the time to read the image file.
    seconds: FiniteFloat = Field(ge=0.0)
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

    ``ValueError`` is raised for a file that holds no JSON and for content that does not
    follow the format of the module docstring. The payloads are not checked here: the adapter
    of the tool reads them.
    """
    return ToolRun.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
