"""Run an external tool on the frames of a job file and write the run file.

The package writes the job file (``python -m pitchvlm.cli tool-job``) and measures the run file
(``python -m pitchvlm.cli tool-score``). Both formats are described in
``src/pitchvlm/fallbacks/runs.py``. The answer of a tool for one frame is a payload in the
format of its adapter module: ``src/pitchvlm/fallbacks/rfdetr.py`` for the detector and
``src/pitchvlm/fallbacks/pnlcalib.py`` for the calibration.

This script runs in the environment of the tool, not in the environment of the package. It
imports the tool and what the tool needs (PyTorch, NumPy, Pillow and, for PnLCalib, PyYAML),
and it does not import pitchvlm. It needs Python 3.10 or newer.

    python tools/run_tool.py rfdetr --job JOB --root DATA --out RUN --device mps
    python tools/run_tool.py pnlcalib --job JOB --root DATA --out RUN --device cpu \\
        --repo PNLCALIB --weights-kp SV_kp --weights-lines SV_lines

``DATA`` is the data folder: the job names each image by its path below it. ``--device`` is
the device name of PyTorch, for example ``mps`` (Apple Silicon), ``cpu`` or ``cuda:0``.

Exit code 0 means that the run file was written. Exit code 2 means that the script could not
run; its last line on the standard error stream then starts with ``error:``.

What each tool gets, and what the script takes from it:

- RF-DETR (package ``rfdetr``, versions ``RFDETR_VERSIONS``): the frame as an RGB ``uint8``
  array of shape (height, width, 3) goes to ``predict`` of the model class that the job names,
  with the score limit of the job as threshold and with the resize of the job (with or
  without antialiasing). The tool scales its boxes to the pixels of the frame and gives the
  class name of each detection. Every detection goes into the payload: the adapter of the
  package keeps the people.
- PnLCalib (a clone of its repository at the commit ``PNLCALIB_COMMIT``): the script follows
  the steps of ``inference.py`` of the clone. It takes the projection matrix of the camera and
  writes the mapping of the grass (pitch metres to pixels of the frame). The world frame of
  the tool is the pitch convention of the package: the origin at the centre mark, x towards
  the right goal, y towards the main camera.

The time of a frame is the time of the tool for that frame, without the reading of the image
file. The first frame is given to the tool one more time before the measured calls, because
the first call of a model also builds what it needs.

This script was written from the source of the two tools. It was never run with them: the
tests replace each tool by a stand-in with the same interface.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import numpy as np

#: Format names of the job file and the run file: those of ``pitchvlm.fallbacks.runs``.
JOB_FORMAT = "pitchvlm-tool-job-1"
RUN_FORMAT = "pitchvlm-tool-run-1"
#: Keys of the tool section of a job, by tool.
TOOL_KEYS = {
    "rfdetr": frozenset({"name", "model", "min_score", "antialias"}),
    "pnlcalib": frozenset({"name", "kp_threshold", "line_threshold", "pnl_refine"}),
}
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
EXIT_OK = 0
EXIT_ERROR = 2
#: What to do when a package or a module of a tool is missing.
WRONG_ENVIRONMENT = "Run this script with the Python of the environment of the tool"

#: Turns one frame, an RGB ``uint8`` array of shape (height, width, 3), into the payload of the frame.
Answer = Callable[[np.ndarray], dict[str, Any]]


class RunnerError(Exception):
    """The script cannot run. The message says why and what to do."""


# --- Job file and run file -----------------------------------------------------------


def read_job(path: Path, tool: str) -> dict[str, Any]:
    """Read a job file and check that it is a job for ``tool``. Wrong content raises ``RunnerError``.

    The job must have the format name of this script, the tool section of ``tool`` with exactly
    its keys, and at least one frame. The image of a frame must be a path below the data
    folder: no absolute path and no step to a parent folder.
    """
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RunnerError(f"the job file {path} cannot be read: {error}") from error
    except ValueError as error:
        raise RunnerError(f"the job file {path} holds no JSON: {error}") from error
    if not isinstance(job, dict) or job.get("format") != JOB_FORMAT:
        raise RunnerError(f"the job file {path} is not in the format {JOB_FORMAT}: write it with the tool-job command")
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


def answer_frames(job: Mapping[str, Any], root: Path, answer: Answer) -> list[dict[str, Any]]:
    """Give every frame of a job to the tool and return the frame entries of the run file.

    The first frame is given to the tool one more time before the measured calls, and that
    answer is thrown away. ``seconds`` of an entry is the time of ``answer`` for the frame.
    An error of the tool on a frame is not caught: a line on the error stream names the
    frame, and the error goes on. So a frame is never counted as a frame without an answer
    because of an error.
    """


    def ask(entry: Mapping[str, Any], image: np.ndarray) -> dict[str, Any]:
        try:
            return answer(image)
        except Exception:
            where = f"clip {entry['clip']}, frame {entry['frame']} ({entry['image']})"
            print(f"the tool stopped on {where}", file=sys.stderr)
            raise

    frames = job["frames"]
    ask(frames[0], read_rgb(root / frames[0]["image"]))
    entries = []
    for entry in frames:
        image = read_rgb(root / entry["image"])
        start = time.perf_counter()
        payload = ask(entry, image)
        seconds = time.perf_counter() - start
        entries.append(
            {"clip": entry["clip"], "frame": entry["frame"], "seconds": round(seconds, 6), "payload": payload}
        )
    return entries


def write_run(
    path: Path, job: Mapping[str, Any], runner: Mapping[str, Any], frames: Sequence[Mapping[str, Any]]
) -> None:
    """Write the run file of a job: its name and tool settings, the facts of the run and the answers.

    The folder of the file is made when it is missing. A value that JSON does not have (NaN,
    infinity) raises ``ValueError``.
    """
    document = {
        "format": RUN_FORMAT,
        "name": job["name"],
        "tool": job["tool"],
        "runner": dict(runner),
        "frames": list(frames),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, allow_nan=False) + "\n", encoding="utf-8")


def machine_facts(device: str) -> dict[str, Any]:
    """Return the facts of the run that do not depend on the tool: the device, the machine and the versions."""
    return {
        "device": device,
        "machine": platform.machine(),
        "system": platform.system(),
        "python": platform.python_version(),
        "torch": installed_version("torch"),
    }


def installed_version(package: str) -> str:
    """Return the version of an installed package. A package that is not installed raises ``RunnerError``."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError as error:
        raise RunnerError(f"the package {package} is not installed here. {WRONG_ENVIRONMENT}") from error


def tool_module(name: str) -> ModuleType:
    """Import a module of the tool or of its environment. A module that cannot be imported raises ``RunnerError``."""
    try:
        return importlib.import_module(name)
    except ImportError as error:
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
    script, and for a model class that is not in ``RFDETR_MODELS``.
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
        names = detections.data["class_name"]
        return {
            "detections": [
                {"box_xyxy": [float(value) for value in box], "score": float(score), "label": str(name)}
                for box, score, name in zip(detections.xyxy, detections.confidence, names, strict=True)
            ]
        }

    return {"tool_version": version, "predictor": "predict() of the model, not compiled"}, answer


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
    ``PNLCALIB_COMMIT`` checked out, else ``RunnerError`` is raised. The two weight files are
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
    for name, text in (("rfdetr", "the RF-DETR detector"), ("pnlcalib", "the PnLCalib calibration")):
        tool = tools.add_parser(name, help=text)
        tool.add_argument("--job", type=_path, required=True, metavar="FILE", help="job file of the tool-job command")
        tool.add_argument("--root", type=_path, required=True, metavar="DIR", help="data folder")
        tool.add_argument("--out", type=_path, required=True, metavar="FILE", help="run file to write (JSON)")
        tool.add_argument(
            "--device", required=True, metavar="NAME", help="device name of PyTorch, for example mps, cpu or cuda:0"
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
    try:
        if sys.version_info < (3, 10):
            raise RunnerError(f"this script needs Python 3.10 or newer, and this is Python {platform.python_version()}")
        job = read_job(args.job, args.tool)
        if args.tool == "rfdetr":
            facts, answer = rfdetr_engine(job["tool"], args.device)
        else:
            facts, answer = pnlcalib_engine(job["tool"], args.device, args.repo, args.weights_kp, args.weights_lines)
        frames = answer_frames(job, args.root, answer)
        write_run(args.out, job, {**machine_facts(args.device), **facts}, frames)
    except (RunnerError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as error:
        traceback.print_exc()
        print(f"error: the script stopped with {type(error).__name__}; the traceback above says where", file=sys.stderr)
        return EXIT_ERROR
    median = statistics.median(entry["seconds"] for entry in frames)
    print(
        f"{args.tool}: {len(frames)} frames of the job {job['name']}, "
        f"median time per frame {median:.3f} s on {args.device}"
    )
    print(f"run file: {args.out}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
