"""Tests of the measurement of an external tool on the frames of a probe.

They cover the job file and the run file (``fallbacks.runs``) and the commands ``tool-job`` and
``tool-score``. The clip is the synthetic clip in the layout of the dataset. The run files are
written by hand here: the detector run holds the labelled boxes, and the calibration run holds
the true camera of the scene, so that every expected value is known. The script that runs a real
tool has its own tests in ``tests/test_run_tool.py``.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pydantic
import pytest

from pitchvlm import cli, probes
from pitchvlm.fallbacks.runs import (
    JOB_FORMAT,
    RUN_FORMAT,
    PnlcalibSettings,
    RfdetrSettings,
    ToolJob,
    ToolRun,
    load_tool_run,
)
from pitchvlm.labels import gsr_frame_people
from pitchvlm.synth import SyntheticScene

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "configs" / "experiments"
DETECTOR_FILE = EXPERIMENTS / "people_probe_rfdetr_small.yaml"
#: The experiment files of the detector: two model classes, each with the resize of the tool and with antialiasing.
DETECTOR_FILES = {
    ("RFDETRSmall", False): DETECTOR_FILE,
    ("RFDETRSmall", True): EXPERIMENTS / "people_probe_rfdetr_small_antialias.yaml",
    ("RFDETRLarge", False): EXPERIMENTS / "people_probe_rfdetr_large.yaml",
    ("RFDETRLarge", True): EXPERIMENTS / "people_probe_rfdetr_large_antialias.yaml",
}
CALIBRATION_FILE = EXPERIMENTS / "pitch_probe_pnlcalib.yaml"
CLIP_ID = "SNGS-999"
LABEL_NAME = "Labels-GameState.json"
N_FRAMES = 50
#: Time per frame that the hand-written run files report: 0.02 s, 0.03 s, 0.04 s, ...
FIRST_SECONDS = 0.02


# --- Clip, job and run files -----------------------------------------------------------


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


@pytest.fixture
def split_list(tmp_path: Path) -> Path:
    """A split list whose group ``probe`` holds the synthetic clip."""
    path = tmp_path / "splits.json"
    path.write_text(json.dumps({"seed": 7, "valid": {"probe": [CLIP_ID]}}), encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    """Read a JSON file. A value that JSON does not have (NaN, Infinity) fails the test."""

    def refuse(token: str) -> None:
        raise AssertionError(f"{token} is not valid JSON")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=refuse)


def write_job(experiment: Path, root: Path, split_list: Path, tmp_path: Path, *more: str) -> dict[str, Any]:
    """Run ``tool-job`` and return the job file that it wrote."""
    job_file = tmp_path / "job.json"
    arguments = ["tool-job", "--experiment", str(experiment), "--root", str(root), "--manifest", str(split_list)]
    assert cli.main([*arguments, "--out", str(job_file), *more]) == 0
    return read_json(job_file)


def run_of(job: dict[str, Any], payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a run file for a job: one payload per frame of the job, in the order of the frames."""
    assert len(payloads) == len(job["frames"])
    return {
        "format": RUN_FORMAT,
        "name": job["name"],
        "tool": job["tool"],
        "runner": {"device": "cpu", "tool_version": "0.0.0"},
        "frames": [
            {"clip": entry["clip"], "frame": entry["frame"], "seconds": FIRST_SECONDS + 0.01 * index, "payload": answer}
            for index, (entry, answer) in enumerate(zip(job["frames"], payloads, strict=True))
        ],
    }


def detector_payloads(job: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    """The payloads of a detector that finds every labelled person: the labelled boxes with the score 0.9."""
    labels = cli.read_labels(root / "SoccerNetGS-valid" / CLIP_ID / LABEL_NAME)
    payloads = []
    for entry in job["frames"]:
        boxes = [person.box for person in gsr_frame_people(labels, entry["frame"]).people]
        payloads.append(
            {
                "detections": [
                    {"box_xyxy": [box.x1, box.y1, box.x2, box.y2], "score": 0.9, "label": "person"} for box in boxes
                ]
            }
        )
    return payloads


def calibration_payloads(job: dict[str, Any], scene: SyntheticScene) -> list[dict[str, Any]]:
    """The payloads of a calibration that finds the true camera of every frame."""
    payloads = []
    for entry in job["frames"]:
        pitch_to_image = np.linalg.inv(scene.homography(entry["frame"]))
        payloads.append({"homography_pitch_to_image": pitch_to_image.tolist(), "pitch_origin": "centre"})
    return payloads


def score(
    experiment: Path, root: Path, split_list: Path, tmp_path: Path, run: dict[str, Any], *more: str
) -> tuple[int, Path]:
    """Write a run file, run ``tool-score`` on it and return the exit code and the path of the metrics file."""
    run_file = tmp_path / "run.json"
    run_file.write_text(json.dumps(run), encoding="utf-8")
    metrics = tmp_path / "metrics" / "tool.json"
    arguments = ["tool-score", "--experiment", str(experiment), "--root", str(root), "--manifest", str(split_list)]
    code = cli.main([*arguments, "--run", str(run_file), "--out", str(metrics), "--allow-dirty", *more])
    return code, metrics


def error_line(capsys: pytest.CaptureFixture[str]) -> str:
    """Return the error line of a command that could not run. A traceback fails the test."""
    stderr = capsys.readouterr().err
    assert "Traceback" not in stderr
    return stderr.splitlines()[-1]


# --- Experiment files ------------------------------------------------------------------

#: The first keys of a tool experiment file, and a right tool section.
TOOL_HEAD = "name: x\ngroup: probe\nframes_per_clip: 5\nseed: 1\n"
GOOD_TOOL = "tool: {name: rfdetr, model: RFDETRSmall, min_score: 0.5, antialias: false}\n"


@pytest.mark.parametrize(
    ("path", "model_probe", "kind", "gate"),
    [
        *((path, "people_probe.yaml", "people", "gate G2") for path in DETECTOR_FILES.values()),
        (CALIBRATION_FILE, "pitch_probe.yaml", "pitch", "gate G3"),
    ],
)
def test_tool_experiment_files_use_the_frames_of_the_model_probes(
    path: Path, model_probe: str, kind: str, gate: str
) -> None:
    experiment = cli.load_tool_experiment(path)
    model = cli.load_experiment(EXPERIMENTS / model_probe)

    assert experiment.name == path.stem
    assert experiment.probe == kind == model.probe
    # The same group, frames per clip and seed give the same frames as the model probe.
    assert (experiment.group, experiment.frames_per_clip, experiment.seed) == (
        model.group,
        model.frames_per_clip,
        model.seed,
    )
    # One comment line says what is measured and which gate it serves.
    comments = [line for line in path.read_text(encoding="utf-8").splitlines() if line.lstrip().startswith("#")]
    assert len(comments) == 1
    assert gate in comments[0]


def test_tool_experiment_files_name_the_settings_of_each_tool() -> None:
    calibration = cli.load_tool_experiment(CALIBRATION_FILE).tool

    # The detector files differ in the model class and the resize only. 0.5 is the threshold that the
    # tool uses by itself, and a resize without antialiasing is what it does by itself.
    for (model, antialias), path in DETECTOR_FILES.items():
        assert cli.load_tool_experiment(path).tool == RfdetrSettings(
            name="rfdetr", model=model, min_score=0.5, antialias=antialias
        )
    # The thresholds of the script inference.py of the tool, with its line refinement.
    assert calibration == PnlcalibSettings(name="pnlcalib", kp_threshold=0.3434, line_threshold=0.7867, pnl_refine=True)
    assert cli.TOOL_PROBES == {"rfdetr": "people", "pnlcalib": "pitch"}
    # Every experiment file of a tool is one of these three.
    tool_files = sorted(path.name for path in EXPERIMENTS.glob("*.yaml") if "tool:" in path.read_text(encoding="utf-8"))
    assert tool_files == sorted(path.name for path in (*DETECTOR_FILES.values(), CALIBRATION_FILE))


@pytest.mark.parametrize(
    "text",
    [
        # A model probe file is not a tool file.
        (EXPERIMENTS / "people_probe.yaml").read_text(encoding="utf-8"),
        "name: x\ngroup: probe\nframes_per_clip: 5\nseed: 1\n",
        "name: x\ngroup: probe\nframes_per_clip: 5\nseed: 1\ntool: rfdetr\n",
        "name: x\ngroup: probe\nframes_per_clip: 5\nseed: 1\ntool: {name: yolo}\n",
        TOOL_HEAD + "tool: {name: rfdetr, model: RFDETRSmall, antialias: false}\n",
        TOOL_HEAD + "tool: {name: rfdetr, model: RFDETRSmall, min_score: 0.5}\n",
        TOOL_HEAD + "tool: {name: rfdetr, model: RFDETRSmall, min_score: 1.5, antialias: false}\n",
        TOOL_HEAD + "tool: {name: rfdetr, model: RFDETRSmall, min_score: 0.5, antialias: 0.5}\n",
        TOOL_HEAD + "tool: {name: rfdetr, model: __class__, min_score: 0.5, antialias: false}\n",
        # A class of the package that is deprecated, and a class whose weights have another licence.
        TOOL_HEAD + "tool: {name: rfdetr, model: RFDETRBase, min_score: 0.5, antialias: false}\n",
        TOOL_HEAD + "tool: {name: rfdetr, model: RFDETRXLarge, min_score: 0.5, antialias: false}\n",
        TOOL_HEAD + "tool: {name: pnlcalib, kp_threshold: 0.3, line_threshold: 0.7}\n",
        TOOL_HEAD + "tool: {name: pnlcalib, kp_threshold: 0.3, line_threshold: 0.7, pnl_refine: 1.5}\n",
        # Only true and false are a truth value: the number 1 and the text "yes" are not.
        TOOL_HEAD + "tool: {name: pnlcalib, kp_threshold: 0.3, line_threshold: 0.7, pnl_refine: 1}\n",
        TOOL_HEAD + "tool: {name: rfdetr, model: RFDETRSmall, min_score: 0.5, antialias: 'yes'}\n",
        TOOL_HEAD + "tool: {name: pnlcalib, kp_threshold: 0.3, line_threshold: 0.7, pnl_refine: true, model: x}\n",
        TOOL_HEAD.replace("frames_per_clip: 5", "frames_per_clip: 0") + GOOD_TOOL,
        TOOL_HEAD + "repeat: 0\n" + GOOD_TOOL,
        "- name: x\n",
    ],
)
def test_wrong_tool_experiment_file_is_refused_with_its_path(tmp_path: Path, text: str) -> None:
    path = tmp_path / "tool.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(cli.CliError, match="tool.yaml") as refused:
        cli.load_tool_experiment(path)

    assert "\n" not in str(refused.value)


# --- tool-job --------------------------------------------------------------------------


@pytest.mark.parametrize("experiment", [DETECTOR_FILE, CALIBRATION_FILE])
def test_tool_job_names_the_frames_of_the_probe_and_the_settings_of_the_tool(
    data_root: tuple[SyntheticScene, Path],
    split_list: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    experiment: Path,
) -> None:
    _, root = data_root

    job = write_job(experiment, root, split_list, tmp_path)

    wanted = cli.load_tool_experiment(experiment)
    assert ToolJob.model_validate(job).tool == wanted.tool
    assert (job["format"], job["name"]) == (JOB_FORMAT, experiment.stem)
    assert set(job) == {"format", "name", "tool", "frames"}
    # The frames are those that the model probe samples for the same clip, count and seed.
    labels = cli.read_labels(root / "SoccerNetGS-valid" / CLIP_ID / LABEL_NAME)
    frames = probes.sample_frames(labels, wanted.frames_per_clip, wanted.seed)
    assert len(frames) == 5
    assert [(entry["clip"], entry["frame"]) for entry in job["frames"]] == [(CLIP_ID, frame) for frame in frames]
    # Each image is named below the data folder, and frame 0 is the file with the lowest number.
    assert [entry["image"] for entry in job["frames"]] == [
        f"SoccerNetGS-valid/{CLIP_ID}/img1/{frame + 1:06d}.png" for frame in frames
    ]
    assert all((root / entry["image"]).is_file() for entry in job["frames"])
    assert str(root) not in (tmp_path / "job.json").read_text(encoding="utf-8")
    output = capsys.readouterr().out
    assert f"tool job {experiment.stem}: 5 frames of 1 clips of the group probe (valid)" in output


def test_tool_job_and_model_probe_read_the_same_images(
    data_root: tuple[SyntheticScene, Path], split_list: Path, tmp_path: Path
) -> None:
    """The image file of a job frame holds the pixels that the frame source of the probe gives for that frame."""
    _, root = data_root
    clip = cli.labelled_clip(root, "valid", CLIP_ID)

    job = write_job(DETECTOR_FILE, root, split_list, tmp_path)

    for entry in job["frames"]:
        from_file = cv2.imread(str(root / entry["image"]), cv2.IMREAD_COLOR)
        assert np.array_equal(from_file, clip.source.read(entry["frame"]))


def test_tool_job_takes_the_first_clips_of_the_group(
    data_root: tuple[SyntheticScene, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, root = data_root
    two_clips = tmp_path / "splits.json"
    two_clips.write_text(json.dumps({"seed": 7, "valid": {"probe": [CLIP_ID, "SNGS-000"]}}), encoding="utf-8")

    # The second clip is not on the disk: with --max-clips 1 it is not read.
    job = write_job(DETECTOR_FILE, root, two_clips, tmp_path, "--max-clips", "1")

    assert {entry["clip"] for entry in job["frames"]} == {CLIP_ID}
    capsys.readouterr()
    arguments = ["tool-job", "--experiment", str(DETECTOR_FILE), "--root", str(root), "--manifest", str(two_clips)]
    assert cli.main([*arguments, "--out", str(tmp_path / "all.json")]) == 2
    assert "SNGS-000" in error_line(capsys)
    assert not (tmp_path / "all.json").exists()


def test_tool_job_needs_the_split_list_and_a_tool_experiment_file(
    data_root: tuple[SyntheticScene, Path], split_list: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, root = data_root
    out = tmp_path / "job.json"
    base = ["tool-job", "--root", str(root), "--out", str(out)]

    assert cli.main([*base, "--experiment", str(DETECTOR_FILE), "--manifest", str(tmp_path / "missing.json")]) == 2
    assert "does not exist" in error_line(capsys)
    model_file = EXPERIMENTS / "people_probe.yaml"
    assert cli.main([*base, "--experiment", str(model_file), "--manifest", str(split_list)]) == 2
    assert error_line(capsys).startswith(f"error: the experiment file {model_file} cannot be used:")
    assert not out.exists()


# --- tool-score: detector --------------------------------------------------------------


def test_tool_score_measures_a_detector_that_finds_every_person(
    data_root: tuple[SyntheticScene, Path], split_list: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, root = data_root
    job = write_job(DETECTOR_FILE, root, split_list, tmp_path)
    run = run_of(job, detector_payloads(job, root))

    code, metrics = score(DETECTOR_FILE, root, split_list, tmp_path, run)

    assert code == 0
    measured = read_json(metrics)
    assert (measured["probe"], measured["n_clips"], measured["n_frames"]) == ("people", 1, 5)
    assert (measured["recall"], measured["precision"]) == (1.0, 1.0)
    counts = measured["counts"]
    assert counts["tp"] == counts["true_boxes"] == counts["predicted_boxes"] > 0
    assert measured["foot_error_m"]["median"] == pytest.approx(0.0, abs=1e-6)
    assert "prompts" not in measured
    settings = measured["settings"]
    assert settings["tool"] == job["tool"]
    assert job["tool"] == {"name": "rfdetr", "model": "RFDETRSmall", "min_score": 0.5, "antialias": False}
    assert (settings["seed"], settings["frames_per_clip"]) == (20261113, 5)
    # The height of a small box is the one of gate G2.
    assert settings["small_box_max_height_px"] == 110.0
    # The run file reports 0.02 s to 0.06 s for the five frames.
    seconds = measured["seconds_per_frame"]
    assert (seconds["n"], seconds["median"], seconds["mean"]) == (5, pytest.approx(0.04), pytest.approx(0.04))
    provenance = measured["provenance"]
    assert provenance["command"] == "tool-score"
    assert provenance["experiment"] == "people_probe_rfdetr_small"
    assert provenance["experiment_file"] == DETECTOR_FILE.name
    assert (provenance["split"], provenance["group"], provenance["clips"]) == ("valid", "probe", [CLIP_ID])
    assert (provenance["split_list"], provenance["split_list_seed"]) == ("splits.json", 7)
    assert provenance["max_clips"] is None
    assert (provenance["run_file"], provenance["n_answers_not_used"]) == ("run.json", 0)
    assert provenance["runner"] == {"device": "cpu", "tool_version": "0.0.0"}
    assert isinstance(provenance["commit"], str) and provenance["commit"]
    assert str(tmp_path) not in metrics.read_text(encoding="utf-8")
    output = capsys.readouterr().out
    assert "people probe people_probe_rfdetr_small of the tool rfdetr: 5 frames of 1 clips" in output
    assert output.index("main metric recall") < output.index("tool time per frame: median 0.040 s")
    assert "not measured" not in output


def test_tool_score_counts_what_the_detector_misses_and_adds(
    data_root: tuple[SyntheticScene, Path], split_list: Path, tmp_path: Path
) -> None:
    """In the first frame one person is missed, and one box is added where nobody is."""
    _, root = data_root
    job = write_job(DETECTOR_FILE, root, split_list, tmp_path)
    payloads = detector_payloads(job, root)
    n_true = sum(len(payload["detections"]) for payload in payloads)
    payloads[0]["detections"][0] = {"box_xyxy": [2.0, 2.0, 12.0, 30.0], "score": 0.9, "label": "person"}

    code, metrics = score(DETECTOR_FILE, root, split_list, tmp_path, run_of(job, payloads))

    assert code == 0
    measured = read_json(metrics)
    assert measured["counts"] == {"predicted_boxes": n_true, "true_boxes": n_true, "tp": n_true - 1, "fp": 1, "fn": 1}
    assert measured["recall"] == measured["precision"] == pytest.approx((n_true - 1) / n_true)


def test_tool_score_uses_the_score_limit_of_the_experiment_file_and_people_only(
    data_root: tuple[SyntheticScene, Path], split_list: Path, tmp_path: Path
) -> None:
    """A box below the score limit of 0.5 and a box of another class do not count as wrong boxes."""
    _, root = data_root
    job = write_job(DETECTOR_FILE, root, split_list, tmp_path)
    payloads = detector_payloads(job, root)
    payloads[0]["detections"].append({"box_xyxy": [2.0, 2.0, 12.0, 30.0], "score": 0.49, "label": "person"})
    payloads[0]["detections"].append({"box_xyxy": [40.0, 2.0, 52.0, 14.0], "score": 0.95, "label": "sports ball"})
    payloads[1]["detections"][0]["score"] = 0.5

    code, metrics = score(DETECTOR_FILE, root, split_list, tmp_path, run_of(job, payloads))

    assert code == 0
    measured = read_json(metrics)
    assert (measured["recall"], measured["precision"]) == (1.0, 1.0)
    assert measured["counts"]["fp"] == 0


# --- tool-score: calibration -----------------------------------------------------------


def test_tool_score_measures_a_calibration_that_finds_the_true_camera(
    data_root: tuple[SyntheticScene, Path], split_list: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene, root = data_root
    job = write_job(CALIBRATION_FILE, root, split_list, tmp_path)
    run = run_of(job, calibration_payloads(job, scene))

    code, metrics = score(CALIBRATION_FILE, root, split_list, tmp_path, run)

    assert code == 0
    measured = read_json(metrics)
    assert (measured["probe"], measured["n_clips"], measured["n_frames"]) == ("pitch", 1, 5)
    assert (measured["n_frames_valid"], measured["valid_share"]) == (5, 1.0)
    assert (measured["n_frames_plausible"], measured["plausible_share"]) == (5, 1.0)
    # The labels of the synthetic clip hold the true pitch positions, so the true camera has no error.
    assert measured["error_m"]["n"] > 0
    assert measured["error_m"]["p90"] == pytest.approx(0.0, abs=1e-6)
    check = measured["axis_check"]
    assert check["as_given"] == pytest.approx(0.0, abs=1e-6)
    assert min(check, key=check.__getitem__) == "as_given"
    assert measured["settings"]["tool"] == job["tool"]
    assert measured["settings"]["tool"]["name"] == "pnlcalib"
    assert "prompts" not in measured and "landmarks_per_frame" not in measured
    assert measured["seconds_per_frame"]["n"] == 5
    assert "main metric error_median_m" in capsys.readouterr().out


def test_tool_score_counts_a_frame_without_a_camera_and_sees_a_mirrored_pitch(
    data_root: tuple[SyntheticScene, Path], split_list: Path, tmp_path: Path
) -> None:
    """The first frame has no camera. The others have the camera of a pitch whose y axis points away from the viewer."""
    scene, root = data_root
    job = write_job(CALIBRATION_FILE, root, split_list, tmp_path)
    payloads = calibration_payloads(job, scene)
    payloads[0] = {"homography_pitch_to_image": None}
    y_mirror = np.diag([1.0, -1.0, 1.0])
    for payload in payloads[1:]:
        payload["homography_pitch_to_image"] = (np.array(payload["homography_pitch_to_image"]) @ y_mirror).tolist()

    code, metrics = score(CALIBRATION_FILE, root, split_list, tmp_path, run_of(job, payloads))

    assert code == 0
    measured = read_json(metrics)
    assert (measured["n_frames"], measured["n_frames_valid"], measured["valid_share"]) == (5, 4, pytest.approx(0.8))
    # A mirrored pitch is not plausible, and the axis check names the axis.
    assert measured["n_frames_plausible"] == 0
    check = measured["axis_check"]
    assert min(check, key=check.__getitem__) == "y_flipped"
    assert check["y_flipped"] == pytest.approx(0.0, abs=1e-6)
    assert check["as_given"] > 5.0


# --- tool-score: refusals --------------------------------------------------------------


def renamed(run: dict[str, Any]) -> None:
    run["name"] = "people_probe_other"


def other_threshold(run: dict[str, Any]) -> None:
    run["tool"]["min_score"] = 0.3


def other_resize(run: dict[str, Any]) -> None:
    run["tool"]["antialias"] = True


def other_tool(run: dict[str, Any]) -> None:
    run["tool"] = {"name": "pnlcalib", "kp_threshold": 0.3, "line_threshold": 0.7, "pnl_refine": True}


def one_frame_less(run: dict[str, Any]) -> None:
    del run["frames"][2]


def answer_twice(run: dict[str, Any]) -> None:
    run["frames"].append(copy.deepcopy(run["frames"][0]))


def other_format(run: dict[str, Any]) -> None:
    run["format"] = "pitchvlm-tool-run-0"


def negative_time(run: dict[str, Any]) -> None:
    run["frames"][0]["seconds"] = -0.1


def box_of_three(run: dict[str, Any]) -> None:
    run["frames"][1]["payload"]["detections"][0]["box_xyxy"] = [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    ("change", "words"),
    [
        (renamed, "was made for the experiment 'people_probe_other', not for 'people_probe_rfdetr_small'"),
        (other_threshold, "was made with other tool settings than the experiment file"),
        (other_resize, "was made with other tool settings than the experiment file"),
        (other_tool, "was made with other tool settings than the experiment file"),
        (one_frame_less, "has no answer for 1 of the 5 frames of the probe, for example: SNGS-999 frame "),
        (answer_twice, "1 frames have more than one answer, for example: SNGS-999 frame "),
        (other_format, "format: Input should be 'pitchvlm-tool-run-1'"),
        (negative_time, "frames.0.seconds: Input should be greater than or equal to 0"),
        (box_of_three, "'box_xyxy' must be a list of 4 numbers"),
    ],
)
def test_tool_score_refuses_a_run_file_that_does_not_fit(
    data_root: tuple[SyntheticScene, Path],
    split_list: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    change: Any,
    words: str,
) -> None:
    _, root = data_root
    job = write_job(DETECTOR_FILE, root, split_list, tmp_path)
    run = run_of(job, detector_payloads(job, root))
    change(run)

    code, metrics = score(DETECTOR_FILE, root, split_list, tmp_path, run)

    assert code == 2
    line = error_line(capsys)
    assert line.startswith(f"error: the tool run file {tmp_path / 'run.json'} ")
    assert words in line
    assert not metrics.exists()


def test_refusal_of_a_payload_names_the_clip_and_the_frame(
    data_root: tuple[SyntheticScene, Path], split_list: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene, root = data_root
    job = write_job(CALIBRATION_FILE, root, split_list, tmp_path)
    payloads = calibration_payloads(job, scene)
    payloads[3]["pitch_origin"] = "corner"
    frame = job["frames"][3]["frame"]

    code, metrics = score(CALIBRATION_FILE, root, split_list, tmp_path, run_of(job, payloads))

    assert code == 2
    assert error_line(capsys) == (
        f"error: the tool run file {tmp_path / 'run.json'} cannot be used: clip {CLIP_ID}, frame {frame}: "
        "payload: 'pitch_origin' must be 'centre', got 'corner'"
    )
    assert not metrics.exists()


@pytest.mark.parametrize("text", ["", "this is not JSON", "[1, 2]"])
def test_tool_score_refuses_a_run_file_without_a_json_mapping(
    data_root: tuple[SyntheticScene, Path],
    split_list: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    text: str,
) -> None:
    _, root = data_root
    run_file = tmp_path / "run.json"
    run_file.write_text(text, encoding="utf-8")
    arguments = ["tool-score", "--experiment", str(DETECTOR_FILE), "--root", str(root), "--manifest", str(split_list)]

    code = cli.main([*arguments, "--run", str(run_file), "--out", str(tmp_path / "metrics.json"), "--allow-dirty"])

    assert code == 2
    assert error_line(capsys).startswith(f"error: the tool run file {run_file} cannot be used: ")
    missing = ["--run", str(tmp_path / "none.json"), "--out", str(tmp_path / "metrics.json"), "--allow-dirty"]
    assert cli.main([*arguments, *missing]) == 2
    assert error_line(capsys).startswith(f"error: the tool run file {tmp_path / 'none.json'} cannot be read: ")
    assert not (tmp_path / "metrics.json").exists()


def test_answers_for_other_frames_are_counted_and_not_measured(
    data_root: tuple[SyntheticScene, Path], split_list: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The run file also holds an answer for a frame that the probe does not sample: a box where nobody is."""
    _, root = data_root
    job = write_job(DETECTOR_FILE, root, split_list, tmp_path)
    run = run_of(job, detector_payloads(job, root))
    sampled = {entry["frame"] for entry in job["frames"]}
    other_frame = next(frame for frame in range(N_FRAMES) if frame not in sampled)
    wrong_box = {"detections": [{"box_xyxy": [2.0, 2.0, 12.0, 30.0], "score": 0.9, "label": "person"}]}
    run["frames"].append({"clip": CLIP_ID, "frame": other_frame, "seconds": 9.0, "payload": wrong_box})

    code, metrics = score(DETECTOR_FILE, root, split_list, tmp_path, run)

    assert code == 0
    measured = read_json(metrics)
    assert (measured["n_frames"], measured["precision"]) == (5, 1.0)
    assert measured["seconds_per_frame"]["n"] == 5
    assert measured["provenance"]["n_answers_not_used"] == 1
    assert "1 answers of the run file are for other frames; they are not measured" in capsys.readouterr().out


def test_tool_score_refuses_a_dirty_work_tree_and_records_an_allowed_one(
    data_root: tuple[SyntheticScene, Path],
    split_list: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, root = data_root
    dirty = cli.CodeState(commit="0" * 40, dirty=True, changes=(" M src/pitchvlm/cli.py",))
    monkeypatch.setattr(cli, "code_state", lambda repo: dirty)
    job = write_job(DETECTOR_FILE, root, split_list, tmp_path)
    run_file = tmp_path / "run.json"
    run_file.write_text(json.dumps(run_of(job, detector_payloads(job, root))), encoding="utf-8")
    metrics = tmp_path / "metrics.json"
    arguments = ["tool-score", "--experiment", str(DETECTOR_FILE), "--root", str(root), "--manifest", str(split_list)]
    arguments += ["--run", str(run_file), "--out", str(metrics)]

    assert cli.main(arguments) == 2
    assert "--allow-dirty" in error_line(capsys)
    assert not metrics.exists()

    assert cli.main([*arguments, "--allow-dirty"]) == 0
    provenance = read_json(metrics)["provenance"]
    assert (provenance["commit"], provenance["dirty"]) == ("0" * 40, True)


# --- Job file and run file -------------------------------------------------------------


def test_run_file_is_read_into_a_record(tmp_path: Path) -> None:
    tool = {"name": "rfdetr", "model": "RFDETRSmall", "min_score": 0.5, "antialias": True}
    document = {
        "format": RUN_FORMAT,
        "name": "x",
        "tool": tool,
        "runner": {"device": "mps"},
        "frames": [
            {"clip": "A", "frame": 3, "seconds": 0.5, "payload": {"detections": []}},
            {"clip": "B", "frame": 3, "seconds": 0, "payload": {}},
        ],
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    run = load_tool_run(path)

    assert isinstance(run, ToolRun)
    assert run.tool == RfdetrSettings(name="rfdetr", model="RFDETRSmall", min_score=0.5, antialias=True)
    assert run.runner == {"device": "mps"}
    answers = run.answers()
    assert set(answers) == {("A", 3), ("B", 3)}
    assert (answers["A", 3].seconds, answers["A", 3].payload) == (0.5, {"detections": []})
    # The record gives the file back as it was read.
    assert run.model_dump(mode="json") == document


@pytest.mark.parametrize(
    "change",
    [
        {"format": JOB_FORMAT},
        {"name": ""},
        {"tool": {"name": "rfdetr", "model": "RFDETRSmall", "min_score": 0.5}},
        {"tool": {"name": "rfdetr", "model": "RFDETRSmall", "min_score": 0.5, "antialias": False, "resolution": 560}},
        {"tool": {"name": "rfdetr", "model": "RFDETRSmall", "min_score": 0.5, "antialias": "no"}},
        {"runner": None},
        {"frames": [{"clip": "A", "frame": -1, "seconds": 0.1, "payload": {}}]},
        {"frames": [{"clip": "A", "frame": 0, "seconds": math.inf, "payload": {}}]},
        {"frames": [{"clip": "A", "frame": 0, "seconds": 0.1, "payload": []}]},
        {"frames": [{"clip": "A", "frame": 0, "seconds": 0.1}]},
        {"frames": [{"clip": "", "frame": 0, "seconds": 0.1, "payload": {}}]},
        {"extra": 1},
    ],
)
def test_run_file_with_wrong_content_is_refused(change: dict[str, Any]) -> None:
    document = {
        "format": RUN_FORMAT,
        "name": "x",
        "tool": {"name": "rfdetr", "model": "RFDETRSmall", "min_score": 0.5, "antialias": False},
        "runner": {},
        "frames": [],
    }
    assert ToolRun.model_validate(document).frames == ()

    with pytest.raises(pydantic.ValidationError):
        ToolRun.model_validate({**document, **change})


def test_format_names_of_the_two_files_are_those_of_their_records() -> None:
    assert ToolJob.model_fields["format"].annotation.__args__ == (JOB_FORMAT,)
    assert ToolRun.model_fields["format"].annotation.__args__ == (RUN_FORMAT,)
    assert JOB_FORMAT != RUN_FORMAT
