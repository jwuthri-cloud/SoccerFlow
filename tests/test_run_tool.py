"""Tests of the script that runs an external tool on the frames of a job (``tools/run_tool.py``).

Neither tool is installed where these tests run, and the script was never run with a real tool.
So each tool is replaced by a stand-in that has the interface which the script was written for:

- ``rfdetr``: a module with the model classes. A model takes its device as a keyword, and its
  ``predict`` has the parameters of the real one without the catch-all for unknown keywords, so
  a wrong keyword fails here.
- PnLCalib: a folder with the modules ``model`` and ``utils`` and the two config files, like a
  clone of its repository, plus stand-ins of ``torch`` and ``torchvision``. Every stand-in
  function has the parameters of the real one and keeps what it was called with.

The stand-ins say what the script calls and in which order. They cannot say that the real tools
behave as the script expects: that is checked on the machine that has the tools.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, get_args

import cv2
import numpy as np
import pytest

from pitchvlm import cli
from pitchvlm.fallbacks import pnlcalib as pnlcalib_adapter
from pitchvlm.fallbacks import runs
from pitchvlm.fallbacks.pnlcalib import homography_from_payload
from pitchvlm.geometry import apply_homography, matrix_from_record
from pitchvlm.labels import gsr_frame_people
from pitchvlm.synth import SyntheticScene

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools" / "run_tool.py"
EXPERIMENTS = REPO / "configs" / "experiments"
CLIP_ID = "SNGS-999"
LABEL_NAME = "Labels-GameState.json"
N_FRAMES = 50
DETECTOR_SETTINGS = {"name": "rfdetr", "model": "RFDETRSmall", "min_score": 0.5, "antialias": False}
CALIBRATION_SETTINGS = {"name": "pnlcalib", "kp_threshold": 0.3434, "line_threshold": 0.7867, "pnl_refine": True}
#: Size of the small frames of these tests, pixels.
WIDTH, HEIGHT = 64, 48
#: Names that the stand-ins of PnLCalib and of PyTorch take in the module table of Python.
STAND_IN_MODULES = (
    "torch",
    "torchvision",
    "torchvision.transforms",
    "torchvision.transforms.functional",
    "model",
    "model.cls_hrnet",
    "model.cls_hrnet_l",
    "utils",
    "utils.utils_calib",
    "utils.utils_heatmap",
)


def load_script() -> ModuleType:
    """Load the script as a module. It is not a part of a package."""
    spec = importlib.util.spec_from_file_location("run_tool", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_tool"] = module
    spec.loader.exec_module(module)
    return module


run_tool = load_script()
#: The function of the script that reads the version of an installed package. The tests replace it.
installed_version_of_the_script = run_tool.installed_version


# --- Frames, jobs and the machine ------------------------------------------------------


def read_rgb_with_opencv(path: Path) -> np.ndarray:
    """Read an image file as RGB. It stands in for the reader of the script, which needs Pillow."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert image is not None, path
    return np.ascontiguousarray(image[:, :, ::-1])


def small_frames(root: Path, count: int = 3, size: tuple[int, int] = (WIDTH, HEIGHT)) -> list[dict[str, Any]]:
    """Write small frames below a data folder and return their job entries.

    Frame i has the blue value 10 + i, the green value 20 + i and the red value 30 + i in every pixel.
    """
    width, height = size
    folder = root / "SoccerNetGS-valid" / "CLIP-A" / "img1"
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for index in range(count):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:] = (10 + index, 20 + index, 30 + index)  # OpenCV writes blue, green, red.
        name = f"{index + 1:06d}.png"
        assert cv2.imwrite(str(folder / name), image)
        entries.append({"clip": "CLIP-A", "frame": index, "image": f"SoccerNetGS-valid/CLIP-A/img1/{name}"})
    return entries


def job_of(settings: dict[str, Any], frames: list[dict[str, Any]], name: str = "a_job") -> dict[str, Any]:
    """Return a job file as data."""
    return {"format": runs.JOB_FORMAT, "name": name, "tool": dict(settings), "frames": frames}


def write_json(path: Path, document: Any) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    """Read a JSON file. A value that JSON does not have (NaN, Infinity) fails the test."""

    def refuse(token: str) -> None:
        raise AssertionError(f"{token} is not valid JSON")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=refuse)


@pytest.fixture(autouse=True)
def machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the script the reader of these tests and the versions of an environment with both tools."""
    versions = {"rfdetr": "1.12.1", "torch": "2.9.0"}
    monkeypatch.setattr(run_tool, "read_rgb", read_rgb_with_opencv)
    monkeypatch.setattr(run_tool, "installed_version", versions.__getitem__)


def error_line(capsys: pytest.CaptureFixture[str]) -> str:
    """Return the last line of the error stream."""
    return capsys.readouterr().err.splitlines()[-1]


# --- Stand-in of rfdetr ----------------------------------------------------------------


class Detections:
    """What ``predict`` of the stand-in returns: the fields of the real result that the script reads."""

    def __init__(self, found: list[tuple[list[float], float, str]]) -> None:
        self.xyxy = np.array([box for box, _, _ in found], dtype=np.float32).reshape(-1, 4)
        self.confidence = np.array([score for _, score, _ in found], dtype=np.float32)
        self.class_id = np.arange(len(found), dtype=np.int64)
        self.data = {"class_name": np.array([name for _, _, name in found], dtype=object)}


class Detector:
    """Stand-in of a model class of rfdetr. ``find`` gives the detections of an image."""

    made: list[Detector] = []
    find = staticmethod(lambda image: [([1.0, 2.0, 11.0, 32.0], 0.75, "person")])

    def __init__(self, *, device: str) -> None:
        self.device = device
        self.calls: list[dict[str, Any]] = []
        Detector.made.append(self)

    def predict(
        self,
        images: Any,
        threshold: float = 0.5,
        shape: Any = None,
        patch_size: Any = None,
        include_source_image: bool = True,
        *,
        antialias: bool = False,
    ) -> Detections:
        self.calls.append(
            {
                "image": images,
                "threshold": threshold,
                "shape": shape,
                "include_source_image": include_source_image,
                "antialias": antialias,
            }
        )
        return Detections(type(self).find(images))


@pytest.fixture
def detector(monkeypatch: pytest.MonkeyPatch) -> type[Detector]:
    """Put the stand-in of rfdetr at the place of the package. Each model class is its own class."""
    module = ModuleType("rfdetr")
    for name in ("RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge", "RFDETRBase"):
        setattr(module, name, type(name, (Detector,), {}))
    monkeypatch.setitem(sys.modules, "rfdetr", module)
    monkeypatch.setattr(Detector, "made", [])
    return Detector


def run_detector(tmp_path: Path, job: dict[str, Any], device: str = "mps") -> tuple[int, Path]:
    """Run the script for a detector job. Return the exit code and the path of the run file."""
    out = tmp_path / "runs" / "run.json"
    arguments = ["rfdetr", "--job", str(write_json(tmp_path / "job.json", job)), "--root", str(tmp_path / "data")]
    return run_tool.main([*arguments, "--out", str(out), "--device", device]), out


# --- Job file --------------------------------------------------------------------------


def with_tool(job: dict[str, Any], **changes: Any) -> None:
    job["tool"].update(changes)


@pytest.mark.parametrize(
    ("change", "words"),
    [
        (lambda job: job.update(format="pitchvlm-tool-job-0"), "is not in the format pitchvlm-tool-job-1"),
        (lambda job: job.update(format=runs.RUN_FORMAT), "is not in the format pitchvlm-tool-job-1"),
        (lambda job: job.pop("name"), "has no name"),
        (lambda job: job.update(name=""), "has no name"),
        (lambda job: job.update(tool="rfdetr"), "is a job for the tool None, not for 'rfdetr'"),
        (lambda job: job.update(tool=dict(CALIBRATION_SETTINGS)), "is a job for the tool 'pnlcalib', not for 'rfdetr'"),
        (lambda job: job["tool"].pop("min_score"), "the tool settings ['antialias', 'min_score', 'model', 'name']"),
        (lambda job: job["tool"].pop("antialias"), "it has ['min_score', 'model', 'name']"),
        (lambda job: with_tool(job, shape=560), "it has ['antialias', 'min_score', 'model', 'name', 'shape']"),
        (lambda job: job.update(frames=[]), "names no frame"),
        (lambda job: job.pop("frames"), "names no frame"),
        (lambda job: job["frames"].append("000004.png"), "frames[3]: expected a mapping"),
        (lambda job: job["frames"][1].update(clip=""), "frames[1]: 'clip' must be a name"),
        (lambda job: job["frames"][1].update(frame=-1), "frames[1]: 'frame' must be a whole number from 0"),
        (lambda job: job["frames"][1].update(frame=1.0), "frames[1]: 'frame' must be a whole number from 0"),
        (lambda job: job["frames"][1].update(frame=True), "frames[1]: 'frame' must be a whole number from 0"),
        (lambda job: job["frames"][2].pop("image"), "frames[2]: 'image' must be a path"),
        (lambda job: job["frames"][2].update(image="/etc/hosts"), "'image' must be a path below the data folder"),
        (lambda job: job["frames"][2].update(image="a/../../b.png"), "'image' must be a path below the data folder"),
    ],
)
def test_job_file_with_wrong_content_is_refused_before_the_tool_is_built(
    tmp_path: Path, detector: type[Detector], capsys: pytest.CaptureFixture[str], change: Any, words: str
) -> None:
    job = job_of(DETECTOR_SETTINGS, small_frames(tmp_path / "data"))
    change(job)

    code, out = run_detector(tmp_path, job)

    assert code == 2
    line = error_line(capsys)
    assert line.startswith(f"error: the job file {tmp_path / 'job.json'}")
    assert words in line
    assert not out.exists()
    assert detector.made == []


@pytest.mark.parametrize(("text", "words"), [(None, "cannot be read"), ("no JSON", "holds no JSON"), ("[1]", "format")])
def test_job_file_that_is_missing_or_holds_no_job_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], text: str | None, words: str
) -> None:
    job_file = tmp_path / "job.json"
    if text is not None:
        job_file.write_text(text, encoding="utf-8")
    arguments = ["rfdetr", "--job", str(job_file), "--root", str(tmp_path), "--out", str(tmp_path / "run.json")]

    assert run_tool.main([*arguments, "--device", "cpu"]) == 2
    assert words in error_line(capsys)
    assert not (tmp_path / "run.json").exists()


def test_command_line_needs_the_tool_and_its_options(tmp_path: Path) -> None:
    base = ["--job", "job.json", "--root", "data", "--out", "run.json"]
    wrong = [
        [],
        base,
        ["yolo", *base, "--device", "cpu"],
        ["rfdetr", *base],
        ["pnlcalib", *base, "--device", "cpu"],
        ["pnlcalib", *base, "--device", "cpu", "--repo", "clone", "--weights-kp", "SV_kp"],
        ["rfdetr", *base, "--device", "cpu", "--repo", "clone"],
    ]
    for arguments in wrong:
        with pytest.raises(SystemExit) as stopped:
            run_tool.build_parser().parse_args(arguments)
        assert stopped.value.code == 2
    args = run_tool.build_parser().parse_args(
        ["pnlcalib", *base, "--device", "mps", "--repo", "~/clone", "--weights-kp", "a", "--weights-lines", "b"]
    )
    assert (args.tool, args.device, args.repo) == ("pnlcalib", "mps", Path.home() / "clone")


# --- RF-DETR ---------------------------------------------------------------------------


def test_detector_gets_each_frame_as_it_was_read_with_the_settings_of_the_job(
    tmp_path: Path, detector: type[Detector]
) -> None:
    frames = small_frames(tmp_path / "data")
    settings = {**DETECTOR_SETTINGS, "model": "RFDETRMedium", "min_score": 0.35, "antialias": True}

    code, _ = run_detector(tmp_path, job_of(settings, frames), device="cuda:1")

    assert code == 0
    # One model of the class that the job names, on the device of the command line.
    (model,) = detector.made
    assert (type(model).__name__, model.device) == ("RFDETRMedium", "cuda:1")
    # The first frame one more time, then every frame in the order of the job.
    assert len(model.calls) == len(frames) + 1
    for call, index in zip(model.calls, [0, 0, 1, 2], strict=True):
        image = call["image"]
        assert (image.dtype, image.shape) == (np.uint8, (HEIGHT, WIDTH, 3))
        # Red, green, blue: the frame was written with red 30 + i, green 20 + i and blue 10 + i.
        assert image[0, 0].tolist() == [30 + index, 20 + index, 10 + index]
        # The score limit of the job is the threshold of the tool, and no copy of the frame comes back.
        assert (call["threshold"], call["include_source_image"], call["shape"]) == (0.35, False, None)
        # The resize is the one of the job.
        assert call["antialias"] is True


def test_run_file_of_the_detector_holds_every_detection_and_the_facts_of_the_run(
    tmp_path: Path, detector: type[Detector], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    found = [
        ([10.5, 2.0, 20.25, 40.0], 0.875, "person"),
        ([30.0, 5.0, 36.0, 11.0], 0.625, "sports ball"),
        ([1.0, 1.0, 5.0, 9.0], 0.5625, ""),
    ]
    monkeypatch.setattr(detector, "find", staticmethod(lambda image: found if image[0, 0, 0] == 31 else []))
    frames = small_frames(tmp_path / "data")
    job = job_of(DETECTOR_SETTINGS, frames, name="people_probe_rfdetr")

    code, out = run_detector(tmp_path, job)

    assert code == 0
    run = read_json(out)
    assert set(run) == {"format", "name", "tool", "runner", "frames"}
    assert (run["format"], run["name"], run["tool"]) == (runs.RUN_FORMAT, "people_probe_rfdetr", DETECTOR_SETTINGS)
    assert [(entry["clip"], entry["frame"]) for entry in run["frames"]] == [("CLIP-A", 0), ("CLIP-A", 1), ("CLIP-A", 2)]
    assert all(set(entry) == {"clip", "frame", "seconds", "payload"} for entry in run["frames"])
    # Frame 1 has the red value 31. Its detections keep their order, their boxes, scores and class names.
    assert run["frames"][1]["payload"] == {
        "detections": [
            {"box_xyxy": [10.5, 2.0, 20.25, 40.0], "score": 0.875, "label": "person"},
            {"box_xyxy": [30.0, 5.0, 36.0, 11.0], "score": 0.625, "label": "sports ball"},
            {"box_xyxy": [1.0, 1.0, 5.0, 9.0], "score": 0.5625, "label": ""},
        ]
    }
    assert run["frames"][0]["payload"] == run["frames"][2]["payload"] == {"detections": []}
    runner = run["runner"]
    assert (runner["device"], runner["tool_version"], runner["torch"]) == ("mps", "1.12.1", "2.9.0")
    assert runner["predictor"] == "predict() of the model, not compiled"
    assert set(runner) == {
        "device",
        "machine",
        "system",
        "python",
        "torch",
        "script_sha256",
        "tool_version",
        "predictor",
    }
    # The run file names the script that made it: tool-score compares it with the script of its work tree.
    assert runner["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    # The package reads the file, and the file holds no absolute path.
    assert runs.load_tool_run(out).tool == runs.RfdetrSettings(**DETECTOR_SETTINGS)
    assert str(tmp_path) not in out.read_text(encoding="utf-8")
    output = capsys.readouterr().out
    assert "rfdetr: 3 frames of the job people_probe_rfdetr, median time per frame" in output
    assert " s on mps" in output
    assert output.splitlines()[-1] == f"run file: {out}"


def test_only_the_call_of_the_tool_is_timed(
    tmp_path: Path, detector: type[Detector], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clock that goes 10 s for each read of an image and 0.25 s, 0.5 s, 0.75 s ... for each call of the tool."""
    now = [0.0]
    calls = [0]

    def slow_read(path: Path) -> np.ndarray:
        now[0] += 10.0
        return read_rgb_with_opencv(path)

    def find(image: np.ndarray) -> list[Any]:
        calls[0] += 1
        now[0] += 0.25 * calls[0]
        return []

    monkeypatch.setattr(run_tool, "read_rgb", slow_read)
    monkeypatch.setattr(run_tool, "time", SimpleNamespace(perf_counter=lambda: now[0]))
    monkeypatch.setattr(detector, "find", staticmethod(find))

    code, out = run_detector(tmp_path, job_of(DETECTOR_SETTINGS, small_frames(tmp_path / "data")))

    assert code == 0
    # The first call is the one that is thrown away. The three measured calls are the calls 2, 3 and 4.
    assert [entry["seconds"] for entry in read_json(out)["frames"]] == [0.5, 0.75, 1.0]


def test_another_version_of_rfdetr_is_refused(
    tmp_path: Path, detector: type[Detector], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(run_tool, "installed_version", {"rfdetr": "1.13.0", "torch": "2.9.0"}.__getitem__)

    code, out = run_detector(tmp_path, job_of(DETECTOR_SETTINGS, small_frames(tmp_path / "data")))

    assert code == 2
    line = error_line(capsys)
    assert "was written for rfdetr 1.12.0 and 1.12.1, and rfdetr 1.13.0 is installed" in line
    assert not out.exists()
    assert detector.made == []


@pytest.mark.parametrize("value", ["true", 1, None])
def test_resize_setting_that_is_no_truth_value_is_refused(
    tmp_path: Path, detector: type[Detector], capsys: pytest.CaptureFixture[str], value: Any
) -> None:
    settings = {**DETECTOR_SETTINGS, "antialias": value}

    code, out = run_detector(tmp_path, job_of(settings, small_frames(tmp_path / "data")))

    assert code == 2
    assert error_line(capsys) == f"error: the tool setting 'antialias' must be true or false, got {value!r}"
    assert not out.exists()
    assert detector.made == []


def test_model_class_outside_the_list_of_the_script_is_refused(
    tmp_path: Path, detector: type[Detector], capsys: pytest.CaptureFixture[str]
) -> None:
    """The stand-in has the class RFDETRBase, as the real package has. The script does not use it."""
    settings = {**DETECTOR_SETTINGS, "model": "RFDETRBase"}

    code, out = run_detector(tmp_path, job_of(settings, small_frames(tmp_path / "data")))

    assert code == 2
    line = error_line(capsys)
    assert "the model 'RFDETRBase' is not one of RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge" in line
    assert not out.exists()
    assert detector.made == []


def test_error_of_a_tool_ends_with_code_2_after_the_frame_and_the_traceback(
    tmp_path: Path, detector: type[Detector], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The tool fails on the frame with the red value 32: frame 2, the third one."""

    def broken(image: np.ndarray) -> list[Any]:
        if image[0, 0, 0] == 32:
            raise RuntimeError("the device has no memory left")
        return []

    monkeypatch.setattr(detector, "find", staticmethod(broken))

    code, out = run_detector(tmp_path, job_of(DETECTOR_SETTINGS, small_frames(tmp_path / "data")))

    assert code == 2
    lines = capsys.readouterr().err.splitlines()
    # The frame is named first, then the traceback, then the error line. No run file is written.
    assert lines[0] == (
        "the tool stopped on clip CLIP-A, frame 2 (SoccerNetGS-valid/CLIP-A/img1/000003.png)"
    )
    assert lines[1].startswith("Traceback")
    assert any("the device has no memory left" in line for line in lines)
    assert lines[-1] == "error: the script stopped with RuntimeError; the traceback above says where"
    assert not out.exists()


def test_error_of_the_tool_on_its_first_call_names_the_first_frame(
    tmp_path: Path, detector: type[Detector], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The first call is the one before the measured calls. A tool with another interface fails there."""

    def broken(image: np.ndarray) -> list[Any]:
        raise TypeError("predict() got an unexpected keyword argument")

    monkeypatch.setattr(detector, "find", staticmethod(broken))

    code, out = run_detector(tmp_path, job_of(DETECTOR_SETTINGS, small_frames(tmp_path / "data")))

    assert code == 2
    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == "the tool stopped on clip CLIP-A, frame 0 (SoccerNetGS-valid/CLIP-A/img1/000001.png)"
    assert lines[-1] == "error: the script stopped with TypeError; the traceback above says where"
    assert not out.exists()


def test_script_in_an_environment_without_the_tool_says_so_in_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The import of the tool fails here, also on a machine where the package rfdetr is installed."""
    monkeypatch.setitem(sys.modules, "rfdetr", None)
    job = job_of(DETECTOR_SETTINGS, small_frames(tmp_path / "data"))

    code, out = run_detector(tmp_path, job)

    assert code == 2
    stderr = capsys.readouterr().err
    assert "Traceback" not in stderr
    assert stderr.splitlines()[-1] == (
        "error: the module rfdetr cannot be imported: import of rfdetr halted; None in sys.modules. "
        "Run this script with the Python of the environment of the tool"
    )
    assert not out.exists()
    # The version of a package that is not installed gives the same advice.
    with pytest.raises(run_tool.RunnerError, match="the package no-such-tool is not installed here. Run this script"):
        installed_version_of_the_script("no-such-tool")
    assert installed_version_of_the_script("numpy") == np.__version__


def test_missing_image_ends_with_code_2_and_one_line(
    tmp_path: Path, detector: type[Detector], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def read(path: Path) -> np.ndarray:
        raise FileNotFoundError(f"no such image: {path}")

    monkeypatch.setattr(run_tool, "read_rgb", read)

    code, out = run_detector(tmp_path, job_of(DETECTOR_SETTINGS, small_frames(tmp_path / "data")))

    assert code == 2
    stderr = capsys.readouterr().err
    assert "Traceback" not in stderr
    assert stderr.splitlines()[-1].startswith("error: no such image: ")
    assert not out.exists()


def test_reader_of_the_script_gives_red_green_blue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The reader of the script needs Pillow, which the environment of the package does not have."""
    pytest.importorskip("PIL", reason="Pillow is not installed: the image reader of tools/run_tool.py cannot run here")
    monkeypatch.undo()
    (entry,) = small_frames(tmp_path, count=1)

    image = run_tool.read_rgb(tmp_path / entry["image"])

    assert (image.dtype, image.shape) == (np.uint8, (HEIGHT, WIDTH, 3))
    assert image[0, 0].tolist() == [30, 20, 10]


# --- PnLCalib: camera and payload ------------------------------------------------------

#: A camera by hand, in the world frame of the tool: x to the right goal, y to the main camera,
#: z down. It stands on the near side, 70 m from the centre mark and 15 m above the grass, and it
#: looks at the centre mark. Its picture has 1920 x 1080 pixels.
CAMERA_POSITION = (0.0, 70.0, -15.0)
FOCAL_PX = 1800.0
PRINCIPAL_POINT = (959.5, 539.5)
_DISTANCE = math.hypot(70.0, 15.0)
_SIN, _COS = 15.0 / _DISTANCE, 70.0 / _DISTANCE
#: Rows: image right, image down and the viewing direction, in world coordinates.
CAMERA_ROTATION = ((1.0, 0.0, 0.0), (0.0, _SIN, _COS), (0.0, -_COS, _SIN))
#: Points on the grass, pitch metres: the centre mark, the middle of the far and of the near
#: touchline, the middle of the right goal line, and three points with no symmetry.
GRASS_POINTS = np.array(
    [[0.0, 0.0], [0.0, -34.0], [0.0, 34.0], [52.5, 0.0], [-20.0, -10.0], [31.0, 12.5], [-44.0, 27.0]]
)


def hand_camera() -> dict[str, Any]:
    """Return the camera by hand as the tool gives a camera."""
    return {
        "pan_degrees": 0.0,
        "tilt_degrees": 77.9,
        "roll_degrees": 0.0,
        "x_focal_length": FOCAL_PX,
        "y_focal_length": FOCAL_PX,
        "principal_point": list(PRINCIPAL_POINT),
        "position_meters": list(CAMERA_POSITION),
        "rotation_matrix": [list(row) for row in CAMERA_ROTATION],
        "radial_distortion": [0.0] * 6,
        "tangential_distortion": [0.0] * 2,
        "thin_prism_distortion": [0.0] * 4,
    }


def seen_by_hand_camera(points: np.ndarray) -> np.ndarray:
    """Project grass points with the camera by hand, one coordinate at a time. The result is in pixels."""
    pixels = []
    for x, y in points:
        in_camera = np.array(CAMERA_ROTATION) @ (np.array([x, y, 0.0]) - np.array(CAMERA_POSITION))
        pixels.append(
            [
                PRINCIPAL_POINT[0] + FOCAL_PX * in_camera[0] / in_camera[2],
                PRINCIPAL_POINT[1] + FOCAL_PX * in_camera[1] / in_camera[2],
            ]
        )
    return np.array(pixels)


def voted(cam_params: dict[str, Any], rep_err: float = 1.5) -> dict[str, Any]:
    """Return what the calibration of the tool gives for a frame with a camera."""
    return {"mode": "full", "use_ransac": 5, "rep_err": rep_err, "cam_params": cam_params, "calib_plane": 0}


def test_camera_homography_maps_the_grass_like_the_camera() -> None:
    homography = run_tool.camera_homography(hand_camera())

    assert homography.shape == (3, 3)
    mapped = np.column_stack([GRASS_POINTS, np.ones(len(GRASS_POINTS))]) @ homography.T
    assert np.allclose(mapped[:, :2] / mapped[:, 2:], seen_by_hand_camera(GRASS_POINTS), atol=1e-8)


def test_camera_by_hand_shows_the_pitch_as_the_main_camera_sees_it() -> None:
    """A check of the camera by hand itself: it is a rotation, and the pitch is the right way round."""
    assert np.allclose(np.array(CAMERA_ROTATION) @ np.array(CAMERA_ROTATION).T, np.eye(3), atol=1e-12)
    assert np.linalg.det(np.array(CAMERA_ROTATION)) == pytest.approx(1.0)
    centre, far, near, right_goal = seen_by_hand_camera(GRASS_POINTS[:4])
    # The centre mark is at the principal point. The far touchline is above it and the near one below.
    assert centre.tolist() == pytest.approx(list(PRINCIPAL_POINT))
    assert far[1] < centre[1] < near[1]
    # The right goal is on the right of the picture.
    assert right_goal[0] > centre[0]


def test_payload_of_a_camera_gives_the_adapter_the_true_mapping() -> None:
    """The matrix of the payload is in the pitch convention of the package: the adapter needs no change of an axis."""
    payload = run_tool.calibration_payload(voted(hand_camera(), rep_err=2.25))

    assert set(payload) == {
        "homography_pitch_to_image",
        "pitch_origin",
        "rep_err",
        "mode",
        "use_ransac",
        "calib_plane",
        "camera_position_m",
    }
    assert (payload["pitch_origin"], payload["rep_err"], payload["mode"], payload["use_ransac"]) == (
        "centre",
        2.25,
        "full",
        5.0,
    )
    # The grass plane, and a camera on the near side (y above 34) above the grass (z below 0).
    assert (payload["calib_plane"], payload["camera_position_m"]) == (0, [0.0, 70.0, -15.0])
    json.dumps(payload, allow_nan=False)
    record = homography_from_payload(payload, frame=7)
    assert (record.frame, record.valid) == (7, True)
    on_pitch = apply_homography(matrix_from_record(record), seen_by_hand_camera(GRASS_POINTS))
    assert np.allclose(on_pitch, GRASS_POINTS, atol=1e-8)


def test_payload_takes_the_values_of_the_tool_as_plain_numbers() -> None:
    """The tool gives numpy values, also a camera place as an array of shape (3, 1)."""
    camera = hand_camera()
    camera["x_focal_length"] = np.float64(FOCAL_PX)
    camera["principal_point"] = [np.float64(value) for value in PRINCIPAL_POINT]
    camera["position_meters"] = np.array(CAMERA_POSITION).reshape(3, 1)
    camera["rotation_matrix"] = np.array(CAMERA_ROTATION)
    result = {
        "mode": "ground_plane",
        "use_ransac": np.int64(25),
        "rep_err": np.float64(4.5),
        "cam_params": camera,
        "calib_plane": np.int64(1),
    }

    payload = run_tool.calibration_payload(result)

    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert (payload["mode"], payload["use_ransac"], payload["rep_err"], payload["calib_plane"]) == (
        "ground_plane",
        25.0,
        4.5,
        1,
    )
    assert payload["camera_position_m"] == [0.0, 70.0, -15.0]
    assert np.allclose(payload["homography_pitch_to_image"], run_tool.camera_homography(hand_camera()))


def test_frame_without_a_camera_gives_a_payload_without_a_matrix() -> None:
    payload = run_tool.calibration_payload(None)

    assert payload == {"homography_pitch_to_image": None, "fault": "the tool found no camera"}
    assert homography_from_payload(payload, frame=0).valid is False


def test_camera_with_a_value_that_is_not_finite_gives_a_payload_without_a_matrix() -> None:
    broken = hand_camera()
    broken["x_focal_length"] = math.nan

    payload = run_tool.calibration_payload(voted(broken, rep_err=math.nan))

    assert payload == {
        "homography_pitch_to_image": None,
        "fault": "the camera of the tool has a value that is not finite",
        "rep_err": None,
    }
    assert homography_from_payload(payload, frame=0).valid is False
    # An error that is not a number does not hide a camera that is one.
    kept = run_tool.calibration_payload(voted(hand_camera(), rep_err=math.inf))
    assert kept["rep_err"] is None and kept["homography_pitch_to_image"] is not None
    json.dumps(kept, allow_nan=False)


# --- Stand-in of PnLCalib --------------------------------------------------------------

TORCH = '''
"""Stand-in of torch: a tensor is an array with a device name."""
import contextlib

import numpy as np

LOADS = []
GRAD = {"enabled": True}


class Tensor:
    def __init__(self, array, device="cpu"):
        self.array = np.asarray(array, dtype=np.float32)
        self.device = device

    def float(self):
        return self

    def unsqueeze(self, dim):
        assert dim == 0
        return Tensor(self.array[None], self.device)

    def size(self):
        return self.array.shape

    def to(self, device):
        return Tensor(self.array, device)

    def __getitem__(self, key):
        return Tensor(self.array[key], self.device)


def load(f, map_location=None, pickle_module=None, *, weights_only=None, mmap=None):
    LOADS.append({"file": str(f), "map_location": map_location, "weights_only": weights_only})
    return {"weights of": str(f)}


@contextlib.contextmanager
def no_grad():
    GRAD["enabled"] = False
    try:
        yield
    finally:
        GRAD["enabled"] = True
'''

TRANSFORMS = '''
"""Stand-in of torchvision.transforms."""
import numpy as np
import torch

RESIZES = []


class Resize:
    def __init__(self, size, interpolation=None, max_size=None, antialias=True):
        self.size = tuple(size)

    def __call__(self, tensor):
        _, _, height, width = tensor.size()
        RESIZES.append({"from": (height, width), "to": self.size, "device": tensor.device})
        rows = np.arange(self.size[0]) * height // self.size[0]
        columns = np.arange(self.size[1]) * width // self.size[1]
        return torch.Tensor(tensor.array[:, :, rows][:, :, :, columns], tensor.device)
'''

FUNCTIONAL = '''
"""Stand-in of torchvision.transforms.functional."""
import numpy as np
import torch


def to_tensor(pic):
    array = np.asarray(pic)
    assert array.dtype == np.uint8 and array.ndim == 3 and array.shape[2] == 3, (array.dtype, array.shape)
    return torch.Tensor(array.transpose(2, 0, 1) / 255.0)
'''

NETWORK = '''
"""Stand-in of a model file of the clone."""
import torch

NETS = []


class Net:
    def __init__(self, config):
        self.config = config
        self.state = None
        self.device = None
        self.in_eval = False
        self.inputs = []

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.state = state_dict

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.in_eval = True
        return self

    def __call__(self, frame):
        _, _, height, width = frame.size()
        self.inputs.append(
            {
                "size": frame.size(),
                "device": frame.device,
                "grad": torch.GRAD["enabled"],
                "low": float(frame.array.min()),
                "high": float(frame.array.max()),
                "channels": [round(float(value) * 255.0) for value in frame.array.mean(axis=(0, 2, 3))],
            }
        )
        joints = self.config["MODEL"]["NUM_JOINTS"]
        return torch.Tensor(torch.np.zeros((1, joints, height // 2, width // 2)), frame.device)


def get_cls_net(config, pretrained="", **kwargs):
    net = Net(config)
    NETS.append(net)
    return net
'''

HEATMAP = '''
"""Stand-in of utils/utils_heatmap.py of the clone. Each function keeps its arguments."""
CALLS = []


def get_keypoints_from_heatmap_batch_maxpool(
    heatmap, scale=2, max_keypoints=1, min_keypoint_pixel_distance=1, return_scores=True
):
    CALLS.append(("keypoints", heatmap.size(), heatmap.device))
    return "keypoint coordinates"


def get_keypoints_from_heatmap_batch_maxpool_l(
    heatmap, scale=2, max_keypoints=2, min_keypoint_pixel_distance=1, return_scores=True
):
    CALLS.append(("lines", heatmap.size(), heatmap.device))
    return "line coordinates"


def coords_to_dict(coords, threshold=0.05, ground_plane_only=False):
    CALLS.append(("coords_to_dict", coords, threshold))
    return [{"of": coords, "threshold": threshold, "call": len(CALLS)}]


def complete_keypoints(kp_dict, lines_dict, w, h, normalize=False):
    CALLS.append(("complete_keypoints", dict(kp_dict), dict(lines_dict), w, h, normalize))
    return {"completed": kp_dict["call"]}, {"completed": lines_dict["call"]}
'''

CALIB = '''
"""Stand-in of utils/utils_calib.py of the clone. The results come from results.json, in the order of the calls."""
import json
from pathlib import Path

CALLS = []
RESULTS = json.loads((Path(__file__).resolve().parents[1] / "results.json").read_text(encoding="utf-8"))


class FramebyFrameCalib:
    def __init__(self, iwidth=960, iheight=540, denormalize=False):
        self.given = None
        CALLS.append(("init", iwidth, iheight, denormalize))

    def update(self, kp_dict, lines_dict):
        self.given = (dict(kp_dict), dict(lines_dict))
        CALLS.append(("update", *self.given))

    def heuristic_voting(self, refine=False, refine_lines=False, th=5.0):
        assert self.given is not None, "heuristic_voting before update"
        CALLS.append(("heuristic_voting", refine, refine_lines, th))
        return RESULTS.pop(0)
'''


@pytest.fixture
def clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A folder like a clone of PnLCalib, and stand-ins of torch and torchvision on the module path.

    The script binds the packages ``model`` and ``utils`` to the clone and imports from them. The
    module path and the module table are as before when the test ends.
    """
    folder = tmp_path / "PnLCalib"
    files = {
        "model/cls_hrnet.py": NETWORK,
        "model/cls_hrnet_l.py": NETWORK,
        "utils/utils_heatmap.py": HEATMAP,
        "utils/utils_calib.py": CALIB,
        "config/hrnetv2_w48.yaml": "MODEL:\n  IMAGE_SIZE: [960, 540]\n  NUM_JOINTS: 58\n",
        "config/hrnetv2_w48_l.yaml": "MODEL:\n  IMAGE_SIZE: [960, 540]\n  NUM_JOINTS: 24\n",
    }
    site = tmp_path / "site"
    files.update(
        {
            "../site/torch/__init__.py": TORCH + "\nnp = np\n",
            "../site/torchvision/__init__.py": "",
            "../site/torchvision/transforms/__init__.py": TRANSFORMS,
            "../site/torchvision/transforms/functional.py": FUNCTIONAL,
        }
    )
    for name, text in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    (tmp_path / "SV_kp").write_bytes(b"weights of the keypoint model")
    (tmp_path / "SV_lines").write_bytes(b"weights of the line model")
    monkeypatch.setattr(sys, "path", [str(site.resolve()), *sys.path])
    monkeypatch.setattr(run_tool, "clone_commit", lambda repo: run_tool.PNLCALIB_COMMIT)
    monkeypatch.setattr(run_tool, "clone_changes", lambda repo: [])
    monkeypatch.delenv("MPLBACKEND", raising=False)
    for name in STAND_IN_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield folder
    for name in STAND_IN_MODULES:
        sys.modules.pop(name, None)


def run_calibration(
    tmp_path: Path, clone: Path, job: dict[str, Any], results: list[Any], device: str = "mps", root: Path | None = None
) -> tuple[int, Path]:
    """Run the script for a calibration job. ``results`` is what the stand-in calibration gives, call by call."""
    write_json(clone / "results.json", results)
    out = tmp_path / "runs" / "run.json"
    arguments = ["pnlcalib", "--job", str(write_json(tmp_path / "job.json", job))]
    arguments += ["--root", str(root if root is not None else tmp_path / "data"), "--out", str(out)]
    arguments += ["--device", device, "--repo", str(clone)]
    arguments += ["--weights-kp", str(tmp_path / "SV_kp"), "--weights-lines", str(tmp_path / "SV_lines")]
    return run_tool.main(arguments), out


def test_calibration_follows_the_steps_of_the_tool_for_each_frame(tmp_path: Path, clone: Path) -> None:
    frames = small_frames(tmp_path / "data", count=2)
    job = job_of(CALIBRATION_SETTINGS, frames, name="pitch_probe_pnlcalib")
    # The first frame one more time, then frame 0 with a camera and frame 1 without one.
    results = [voted(hand_camera()), voted(hand_camera(), rep_err=3.5), None]

    code, out = run_calibration(tmp_path, clone, job, results, device="mps")

    assert code == 0
    torch = sys.modules["torch"]
    keypoint_net, line_net = sys.modules["model.cls_hrnet"].NETS[0], sys.modules["model.cls_hrnet_l"].NETS[0]
    # Each model: its own config file and weight file, loaded as plain weights on the CPU, then moved and set to eval.
    assert (keypoint_net.config["MODEL"]["NUM_JOINTS"], line_net.config["MODEL"]["NUM_JOINTS"]) == (58, 24)
    assert torch.LOADS == [
        {"file": str(tmp_path / "SV_kp"), "map_location": "cpu", "weights_only": True},
        {"file": str(tmp_path / "SV_lines"), "map_location": "cpu", "weights_only": True},
    ]
    assert (keypoint_net.state, line_net.state) == (
        {"weights of": str(tmp_path / "SV_kp")},
        {"weights of": str(tmp_path / "SV_lines")},
    )
    assert (keypoint_net.device, keypoint_net.in_eval, line_net.device, line_net.in_eval) == ("mps", True, "mps", True)
    # Three frames go through: frame 0, frame 0 and frame 1. Each one is resized on the CPU to 540 x 960,
    # goes to the device as red, green and blue values from 0 to 1, and no gradient is kept.
    assert sys.modules["torchvision.transforms"].RESIZES == [
        {"from": (HEIGHT, WIDTH), "to": (540, 960), "device": "cpu"}
    ] * 3
    for net in (keypoint_net, line_net):
        assert [call["size"] for call in net.inputs] == [(1, 3, 540, 960)] * 3
        assert all((call["device"], call["grad"]) == ("mps", False) for call in net.inputs)
        assert all(0.0 <= call["low"] <= call["high"] <= 1.0 for call in net.inputs)
        assert [call["channels"] for call in net.inputs] == [[30, 20, 10], [30, 20, 10], [31, 21, 11]]
    # The points come from the heatmaps without their last channel, with the two thresholds of the job.
    heatmap_calls = sys.modules["utils.utils_heatmap"].CALLS
    assert len(heatmap_calls) == 5 * 3
    first = heatmap_calls[:5]
    assert first[0] == ("keypoints", (1, 57, 270, 480), "mps")
    assert first[1] == ("lines", (1, 23, 270, 480), "mps")
    assert first[2] == ("coords_to_dict", "keypoint coordinates", 0.3434)
    assert first[3] == ("coords_to_dict", "line coordinates", 0.7867)
    # The keypoints are completed with the size of the model input, as shares of that size.
    assert first[4][0] == "complete_keypoints"
    assert (first[4][1]["of"], first[4][2]["of"]) == ("keypoint coordinates", "line coordinates")
    assert first[4][3:] == (960, 540, True)
    # Each frame gets its own calibration object for the size of the frame, with the line refinement of the job.
    calib_calls = sys.modules["utils.utils_calib"].CALLS
    assert [call[0] for call in calib_calls] == ["init", "update", "heuristic_voting"] * 3
    assert all(call == ("init", WIDTH, HEIGHT, True) for call in calib_calls[0::3])
    # The calibration gets the completed keypoints and lines of its own frame.
    assert [call[1:] for call in calib_calls[1::3]] == [
        ({"completed": 3}, {"completed": 4}),
        ({"completed": 8}, {"completed": 9}),
        ({"completed": 13}, {"completed": 14}),
    ]
    assert all(call == ("heuristic_voting", False, True, 5.0) for call in calib_calls[2::3])
    run = read_json(out)
    assert (run["format"], run["name"], run["tool"]) == (runs.RUN_FORMAT, "pitch_probe_pnlcalib", CALIBRATION_SETTINGS)
    with_camera, without_camera = (entry["payload"] for entry in run["frames"])
    assert with_camera["rep_err"] == 3.5
    assert np.allclose(with_camera["homography_pitch_to_image"], run_tool.camera_homography(hand_camera()))
    assert without_camera == {"homography_pitch_to_image": None, "fault": "the tool found no camera"}
    runner = run["runner"]
    assert (runner["device"], runner["tool_commit"]) == ("mps", run_tool.PNLCALIB_COMMIT)
    assert (runner["weights_kp"], runner["weights_lines"]) == ("SV_kp", "SV_lines")
    # The plot library of the clone gets a backend without a window.
    assert os.environ["MPLBACKEND"] == "Agg"
    assert runner["weights_kp_sha256"] == hashlib.sha256(b"weights of the keypoint model").hexdigest()
    assert runner["weights_lines_sha256"] == hashlib.sha256(b"weights of the line model").hexdigest()
    assert str(tmp_path) not in out.read_text(encoding="utf-8")
    assert runs.load_tool_run(out).tool == runs.PnlcalibSettings(**CALIBRATION_SETTINGS)


def test_job_without_line_refinement_and_a_frame_with_the_width_of_the_models(tmp_path: Path, clone: Path) -> None:
    """A frame that is 960 pixels wide is not resized. The keypoints are then completed with its own height."""
    frames = small_frames(tmp_path / "data", count=1, size=(960, 400))
    settings = {**CALIBRATION_SETTINGS, "kp_threshold": 0.0966, "line_threshold": 0.3441, "pnl_refine": False}

    code, _ = run_calibration(tmp_path, clone, job_of(settings, frames), [None, None], device="cpu")

    assert code == 0
    assert sys.modules["torchvision.transforms"].RESIZES == []
    assert [call["size"] for call in sys.modules["model.cls_hrnet"].NETS[0].inputs] == [(1, 3, 400, 960)] * 2
    heatmap_calls = sys.modules["utils.utils_heatmap"].CALLS
    assert (heatmap_calls[2][2], heatmap_calls[3][2]) == (0.0966, 0.3441)
    assert heatmap_calls[4][3:] == (960, 400, True)
    calib_calls = sys.modules["utils.utils_calib"].CALLS
    assert calib_calls[0] == ("init", 960, 400, True)
    assert calib_calls[2] == ("heuristic_voting", False, False, 5.0)


def test_packages_of_the_clone_are_not_replaced_by_installed_packages_with_their_names(
    tmp_path: Path, clone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Packages with the names ``utils`` and ``model`` come first on the module path, and ``utils`` is loaded.

    The folders of the clone have no ``__init__.py``, as in the real repository. Python takes a
    package with such a file before them, wherever it is on the module path.
    """
    other = tmp_path / "other"
    for name in ("utils", "model"):
        (other / name).mkdir(parents=True)
        (other / name / "__init__.py").write_text("ORIGIN = 'not the clone'\n", encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(other), *sys.path])
    assert importlib.import_module("utils").ORIGIN == "not the clone"
    frames = small_frames(tmp_path / "data", count=1)

    code, _ = run_calibration(tmp_path, clone, job_of(CALIBRATION_SETTINGS, frames), [None, None])

    assert code == 0
    for name in ("utils.utils_calib", "utils.utils_heatmap", "model.cls_hrnet", "model.cls_hrnet_l"):
        assert Path(sys.modules[name].__file__).parent.parent == clone.resolve(), name


def test_folder_that_is_no_clone_of_pnlcalib_is_refused(
    tmp_path: Path, clone: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shutil.rmtree(clone / "utils")
    frames = small_frames(tmp_path / "data", count=1)

    code, out = run_calibration(tmp_path, clone, job_of(CALIBRATION_SETTINGS, frames), [None, None])

    assert code == 2
    assert error_line(capsys) == (
        f"error: {clone.resolve()} has no folder utils: it is not a clone of the PnLCalib repository"
    )
    assert not out.exists()


def test_another_commit_of_pnlcalib_is_refused(
    tmp_path: Path, clone: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(run_tool, "clone_commit", lambda repo: "0" * 40)
    frames = small_frames(tmp_path / "data", count=1)

    code, out = run_calibration(tmp_path, clone, job_of(CALIBRATION_SETTINGS, frames), [None, None])

    assert code == 2
    line = error_line(capsys)
    assert f"was written for the commit {run_tool.PNLCALIB_COMMIT} of PnLCalib" in line
    assert f"{clone.resolve()} has {'0' * 40}" in line
    assert not out.exists()
    assert "utils.utils_calib" not in sys.modules


def test_commit_of_a_clone_is_read_with_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    for name in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    not_a_clone = tmp_path / "folder"
    not_a_clone.mkdir()
    with pytest.raises(run_tool.RunnerError, match="is not a git clone with a commit"):
        run_tool.clone_commit(not_a_clone)

    def git(*arguments: str) -> str:
        identity = ["-c", "user.name=Test", "-c", "user.email=test@example.org", "-c", "commit.gpgsign=false"]
        done = subprocess.run(
            ["git", "-C", str(not_a_clone), *identity, *arguments], capture_output=True, text=True, check=True
        )
        return done.stdout.strip()

    git("init", "--quiet", "--template=")
    (not_a_clone / "inference.py").write_text("# a clone\n", encoding="utf-8")
    git("add", "inference.py")
    git("commit", "--quiet", "--no-verify", "-m", "first")

    commit = run_tool.clone_commit(not_a_clone)

    assert commit == git("rev-parse", "HEAD")
    assert len(commit) == 40


# --- From the job to the metrics -------------------------------------------------------


@pytest.fixture(scope="module")
def data_root(tmp_path_factory: pytest.TempPathFactory) -> tuple[SyntheticScene, Path]:
    """A synthetic clip in the layout of the dataset. The result is the scene and the data root."""
    scene = SyntheticScene(n_frames=N_FRAMES, seed=2)
    root = tmp_path_factory.mktemp("data")
    clip_dir = root / "SoccerNetGS-valid" / CLIP_ID
    (clip_dir / "img1").mkdir(parents=True)
    for frame in range(N_FRAMES):
        assert cv2.imwrite(str(clip_dir / "img1" / f"{frame + 1:06d}.png"), scene.image(frame))
    (clip_dir / LABEL_NAME).write_text(json.dumps(scene.gsr_labels()), encoding="utf-8")
    return scene, root


def job_of_experiment(experiment: Path, root: Path, tmp_path: Path) -> tuple[list[str], dict[str, Any]]:
    """Write the job file of a tool experiment with the package. Return the shared options and the job."""
    split_list = write_json(tmp_path / "splits.json", {"seed": 7, "valid": {"probe": [CLIP_ID]}})
    shared = ["--experiment", str(experiment), "--root", str(root), "--manifest", str(split_list)]
    assert cli.main(["tool-job", *shared, "--out", str(tmp_path / "job.json")]) == 0
    return shared, read_json(tmp_path / "job.json")


def test_detector_goes_from_the_job_of_the_package_to_its_metrics(
    data_root: tuple[SyntheticScene, Path], tmp_path: Path, detector: type[Detector], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stand-in detector finds the labelled people of the frame that it gets, and one box of another class."""
    scene, root = data_root
    labels = cli.read_labels(root / "SoccerNetGS-valid" / CLIP_ID / LABEL_NAME)
    frame_of_pixels = {
        hashlib.sha256(np.ascontiguousarray(scene.image(frame)[:, :, ::-1]).tobytes()).hexdigest(): frame
        for frame in range(N_FRAMES)
    }

    def find(image: np.ndarray) -> list[Any]:
        frame = frame_of_pixels[hashlib.sha256(image.tobytes()).hexdigest()]
        boxes = [person.box for person in gsr_frame_people(labels, frame).people]
        people = [([box.x1, box.y1, box.x2, box.y2], 0.875, "person") for box in boxes]
        return [*people, ([3.0, 3.0, 9.0, 9.0], 0.75, "sports ball")]

    monkeypatch.setattr(detector, "find", staticmethod(find))
    shared, job = job_of_experiment(EXPERIMENTS / "people_probe_rfdetr_small.yaml", root, tmp_path)
    run_file, metrics = tmp_path / "run.json", tmp_path / "metrics.json"

    arguments = ["rfdetr", "--job", str(tmp_path / "job.json"), "--root", str(root), "--out", str(run_file)]
    assert run_tool.main([*arguments, "--device", "cpu"]) == 0
    assert cli.main(["tool-score", *shared, "--run", str(run_file), "--out", str(metrics), "--allow-dirty"]) == 0

    measured = read_json(metrics)
    assert (measured["probe"], measured["n_frames"]) == ("people", len(job["frames"]))
    assert (measured["recall"], measured["precision"]) == (1.0, 1.0)
    assert measured["counts"]["tp"] == measured["counts"]["true_boxes"] > 0
    assert measured["settings"]["tool"] == job["tool"]
    assert measured["provenance"]["runner"]["tool_version"] == "1.12.1"
    assert measured["seconds_per_frame"]["n"] == len(job["frames"])
    # The run file was made by the script of this work tree, and the metrics file names its content.
    assert measured["provenance"]["script_is_current"] is True
    assert measured["provenance"]["run_file_sha256"] == hashlib.sha256(run_file.read_bytes()).hexdigest()


def camera_of_projection(projection: np.ndarray, width: int, height: int) -> dict[str, Any]:
    """Split a 3x4 camera matrix with the principal point at the image centre into the camera values of the tool."""
    left = projection[:, :3]
    centre_x, centre_y = width / 2.0, height / 2.0
    focal_x = float(np.linalg.norm(left[0] - centre_x * left[2]))
    focal_y = float(np.linalg.norm(left[1] - centre_y * left[2]))
    intrinsics = np.array([[focal_x, 0.0, centre_x], [0.0, focal_y, centre_y], [0.0, 0.0, 1.0]])
    return {
        "x_focal_length": focal_x,
        "y_focal_length": focal_y,
        "principal_point": [centre_x, centre_y],
        "position_meters": (-np.linalg.inv(left) @ projection[:, 3]).tolist(),
        "rotation_matrix": (np.linalg.inv(intrinsics) @ left).tolist(),
    }


def test_calibration_goes_from_the_job_of_the_package_to_its_metrics(
    data_root: tuple[SyntheticScene, Path], tmp_path: Path, clone: Path
) -> None:
    """The stand-in calibration gives the true camera of each frame, and no camera for the last frame."""
    scene, root = data_root
    shared, job = job_of_experiment(EXPERIMENTS / "pitch_probe_pnlcalib.yaml", root, tmp_path)
    info = scene.info
    cameras = [
        voted(camera_of_projection(scene._projection(entry["frame"]), info.width, info.height))
        for entry in job["frames"]
    ]
    results = [cameras[0], *cameras[:-1], None]
    metrics = tmp_path / "metrics.json"

    code, run_file = run_calibration(tmp_path, clone, job, results, device="cpu", root=root)
    assert code == 0
    assert cli.main(["tool-score", *shared, "--run", str(run_file), "--out", str(metrics), "--allow-dirty"]) == 0

    measured = read_json(metrics)
    n_frames = len(job["frames"])
    assert (measured["probe"], measured["n_frames"]) == ("pitch", n_frames)
    assert (measured["n_frames_valid"], measured["n_frames_plausible"]) == (n_frames - 1, n_frames - 1)
    assert measured["error_m"]["n"] > 0
    assert measured["error_m"]["p90"] == pytest.approx(0.0, abs=1e-6)
    check = measured["axis_check"]
    assert min(check, key=check.__getitem__) == "as_given"
    assert measured["settings"]["tool"] == job["tool"]
    assert measured["provenance"]["runner"]["tool_commit"] == run_tool.PNLCALIB_COMMIT


# --- The script and the package --------------------------------------------------------


def test_script_and_package_agree_on_formats_settings_and_models() -> None:
    assert (run_tool.JOB_FORMAT, run_tool.RUN_FORMAT) == (runs.JOB_FORMAT, runs.RUN_FORMAT)
    assert run_tool.TOOL_KEYS == {
        "rfdetr": frozenset(runs.RfdetrSettings.model_fields),
        "pnlcalib": frozenset(runs.PnlcalibSettings.model_fields),
    }
    assert set(run_tool.TOOL_KEYS) == set(cli.TOOL_PROBES)
    assert run_tool.RFDETR_MODELS == get_args(runs.RfdetrModel)
    assert run_tool.PITCH_ORIGIN == pnlcalib_adapter.PITCH_ORIGIN


def test_experiment_files_name_settings_that_the_script_takes() -> None:
    for name in ("people_probe_rfdetr_small.yaml", "people_probe_rfdetr_large.yaml", "pitch_probe_pnlcalib.yaml"):
        settings = cli.load_tool_experiment(EXPERIMENTS / name).tool.model_dump(mode="json")
        assert set(settings) == run_tool.TOOL_KEYS[settings["name"]], name
        if settings["name"] == "rfdetr":
            assert settings["model"] in run_tool.RFDETR_MODELS


def test_script_does_not_import_the_package() -> None:
    """The script runs in the environment of a tool, where the package is not installed."""
    lines = [line.strip() for line in SCRIPT.read_text(encoding="utf-8").splitlines()]
    imports = [line for line in lines if line.startswith(("import ", "from "))]

    assert imports
    assert not [line for line in imports if "pitchvlm" in line]
    assert not [line for line in lines if "import_module(" in line and "pitchvlm" in line]


# --- Errors of a tool, of its environment and of the machine ---------------------------


@pytest.mark.parametrize(
    ("box", "score"),
    [
        ([10.0, 20.0, math.nan, 60.0], 0.9),
        ([10.0, 20.0, math.inf, 60.0], 0.9),
        ([10.0, 20.0, 30.0, 60.0], math.nan),
        ([10.0, 20.0, 30.0, 60.0], math.inf),
    ],
)
def test_box_or_score_that_is_not_finite_stops_the_run_at_its_frame(
    tmp_path: Path,
    detector: type[Detector],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    box: list[float],
    score: float,
) -> None:
    def find(image: np.ndarray) -> list[Any]:
        if image[0, 0, 0] == 31:
            return [([10.0, 20.0, 30.0, 60.0], 0.9, "person"), (box, score, "person")]
        return []

    monkeypatch.setattr(detector, "find", staticmethod(find))

    code, out = run_detector(tmp_path, job_of(DETECTOR_SETTINGS, small_frames(tmp_path / "data")))

    assert code == 2
    lines = capsys.readouterr().err.splitlines()
    assert lines == [
        "the tool stopped on clip CLIP-A, frame 1 (SoccerNetGS-valid/CLIP-A/img1/000002.png)",
        "error: the detector gave a box or a score that is not a finite number",
    ]
    assert not out.exists()
    # The run stopped at the frame: the first call, frame 0 and frame 1.
    assert len(detector.made[0].calls) == 3


@pytest.mark.parametrize(
    ("import_line", "error_name"),
    [
        # A name that another version of a package has.
        ("from json import a_name_of_another_version", "ImportError"),
        # A package that the tool needs and that is not installed: the tool itself is installed.
        ("import a_package_that_the_tool_needs", "ModuleNotFoundError"),
    ],
)
def test_import_error_inside_an_installed_tool_keeps_its_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    import_line: str,
    error_name: str,
) -> None:
    site = tmp_path / "site" / "rfdetr"
    site.mkdir(parents=True)
    (site / "__init__.py").write_text("from rfdetr.detr import RFDETRSmall\n", encoding="utf-8")
    (site / "detr.py").write_text(f"{import_line}\n", encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(site.parent), *sys.path])
    for name in ("rfdetr", "rfdetr.detr"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    code, out = run_detector(tmp_path, job_of(DETECTOR_SETTINGS, small_frames(tmp_path / "data")))

    assert code == 2
    lines = capsys.readouterr().err.splitlines()
    assert lines[0].startswith("Traceback")
    assert any("detr.py" in line for line in lines)
    assert lines[-1] == f"error: the script stopped with {error_name}; the traceback above says where"
    assert not any("Run this script with the Python" in line for line in lines)
    for name in ("rfdetr", "rfdetr.detr"):
        sys.modules.pop(name, None)


def test_missing_module_of_the_environment_gives_one_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torchvision", None)
    with pytest.raises(run_tool.RunnerError, match="the module torchvision.transforms cannot be imported"):
        run_tool.tool_module("torchvision.transforms")


def test_image_that_cannot_be_read_names_its_frame(
    tmp_path: Path, detector: type[Detector], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def read(path: Path) -> np.ndarray:
        if path.name == "000003.png":
            raise OSError("image file is truncated (5 bytes not processed)")
        return read_rgb_with_opencv(path)

    monkeypatch.setattr(run_tool, "read_rgb", read)

    code, out = run_detector(tmp_path, job_of(DETECTOR_SETTINGS, small_frames(tmp_path / "data")))

    assert code == 2
    assert capsys.readouterr().err.splitlines() == [
        "the image cannot be read: clip CLIP-A, frame 2 (SoccerNetGS-valid/CLIP-A/img1/000003.png)",
        "error: image file is truncated (5 bytes not processed)",
    ]
    assert not out.exists()


def test_place_of_the_run_file_is_checked_before_the_tool_is_built(
    tmp_path: Path, detector: type[Detector], capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "runs").write_text("a file, not a folder", encoding="utf-8")

    code, out = run_detector(tmp_path, job_of(DETECTOR_SETTINGS, small_frames(tmp_path / "data")))

    assert code == 2
    assert error_line(capsys).startswith("error: [Errno 17] File exists")
    assert detector.made == []


def test_clone_with_a_changed_file_is_refused(
    tmp_path: Path, clone: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    monkeypatch.undo()
    monkeypatch.setattr(run_tool, "read_rgb", read_rgb_with_opencv)
    monkeypatch.setattr(run_tool, "installed_version", {"torch": "2.9.0"}.__getitem__)
    for name in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))

    def git(*arguments: str) -> str:
        identity = ["-c", "user.name=Test", "-c", "user.email=test@example.org", "-c", "commit.gpgsign=false"]
        done = subprocess.run(
            ["git", "-C", str(clone), *identity, *arguments], capture_output=True, text=True, check=True
        )
        return done.stdout.strip()

    git("init", "--quiet", "--template=")
    git("add", "model", "utils", "config")
    git("commit", "--quiet", "--no-verify", "-m", "first")
    monkeypatch.setattr(run_tool, "PNLCALIB_COMMIT", git("rev-parse", "HEAD"))
    # Files that git does not track do not count: results.json of the stand-in, and bytecode folders.
    (clone / "utils" / "__pycache__").mkdir()
    (clone / "utils" / "__pycache__" / "x.pyc").write_bytes(b"bytecode")
    assert run_tool.clone_changes(clone) == []
    with (clone / "utils" / "utils_calib.py").open("a", encoding="utf-8") as stream:
        stream.write("# changed by hand\n")
    frames = small_frames(tmp_path / "data", count=1)

    code, out = run_calibration(tmp_path, clone, job_of(CALIBRATION_SETTINGS, frames), [None, None])

    assert code == 2
    line = error_line(capsys)
    assert "1 tracked files of" in line and "utils/utils_calib.py" in line
    assert not out.exists()
    assert "utils.utils_calib" not in sys.modules


def test_state_of_a_folder_that_is_no_clone_cannot_be_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    for name in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    folder = tmp_path / "no_clone"
    folder.mkdir()

    # The text after the colon is the error line of git.
    with pytest.raises(run_tool.RunnerError, match=r"the state of the clone \S+no_clone cannot be read: \S"):
        run_tool.clone_changes(folder)


def test_older_python_is_refused_in_one_line(
    tmp_path: Path, detector: type[Detector], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    job = job_of(DETECTOR_SETTINGS, small_frames(tmp_path / "data"))
    monkeypatch.setattr(sys, "version_info", (3, 9, 18, "final", 0))

    code, out = run_detector(tmp_path, job)

    monkeypatch.undo()
    assert code == 2
    assert error_line(capsys).startswith("error: this script needs Python 3.10 or newer, and this is Python ")
    assert not out.exists()
    assert detector.made == []


@pytest.mark.parametrize("value", ["true", 1, None])
def test_line_refinement_that_is_no_truth_value_is_refused(
    tmp_path: Path, clone: Path, capsys: pytest.CaptureFixture[str], value: Any
) -> None:
    settings = {**CALIBRATION_SETTINGS, "pnl_refine": value}
    frames = small_frames(tmp_path / "data", count=1)

    code, out = run_calibration(tmp_path, clone, job_of(settings, frames), [None, None])

    assert code == 2
    assert error_line(capsys) == f"error: the tool setting 'pnl_refine' must be true or false, got {value!r}"
    assert not out.exists()


def test_tools_do_not_get_the_key_of_the_model_api(
    tmp_path: Path, detector: type[Detector], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The terminal of the run machine holds the key. A tool is code of another project and must not see it."""
    seen: list[str | None] = []

    def find(image: np.ndarray) -> list[Any]:
        seen.append(os.environ.get("GEMINI_API_KEY"))
        return []

    monkeypatch.setattr(detector, "find", staticmethod(find))
    monkeypatch.setenv("GEMINI_API_KEY", "a-made-up-value")

    code, _ = run_detector(tmp_path, job_of(DETECTOR_SETTINGS, small_frames(tmp_path / "data")))

    assert code == 0
    assert seen == [None] * 4
    assert "GEMINI_API_KEY" not in os.environ
    # The script names the variable of the package.
    assert run_tool.KEY_VARIABLE == cli.KEY_VARIABLE == "GEMINI_API_KEY"


def test_interpreter_without_numpy_gets_one_line_and_the_code_2(tmp_path: Path) -> None:
    """Python without its installed packages (the options -S and -I): the wrong interpreter for the script."""
    arguments = ["rfdetr", "--job", "job.json", "--root", "data", "--out", "run.json", "--device", "cpu"]

    done = subprocess.run(
        [sys.executable, "-S", "-I", str(SCRIPT), *arguments], capture_output=True, text=True, cwd=tmp_path, check=False
    )

    assert done.returncode == 2
    assert done.stderr.splitlines() == [
        "error: NumPy is not installed here. Run this script with the Python of the environment of the tool"
    ]
