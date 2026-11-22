"""Tests of the command line (``cli.py``) and of the doctor checks (``doctor.py``).

Everything runs offline. The model is the fake backend of a synthetic scene, the scorer is a small
stand-in script, and the dataset archives are made in the test folder. Tests that need git build a
throw-away repository below the test folder; they never touch the repository of this package.

The synthetic clip on disk is ``SyntheticScene(n_frames=50, seed=2)``: 2 seconds at 25 frames per
second, 1280 x 720 pixels, with the keyframes 0, 12, 24, 36, 48 and 49. It lies in the layout of the
dataset: ``<root>/SoccerNetGS-valid/SNGS-999/img1/000001.png`` ... and ``Labels-GameState.json``.

Every cost estimate of the command line uses the token counts of the fake model (the fixture
``estimates_of_the_fake_model``). So no test depends on the token estimates of ``vlm.cost``, which
change when real token counts are measured.
"""

from __future__ import annotations

import argparse
import base64
import copy
import dataclasses
import errno
import hashlib
import json
import logging
import os
import queue
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
import zipfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
import yaml

from pitchvlm import cli, doctor, pipeline
from pitchvlm.config import PipelineConfig, StepSettings, TrackingConfig
from pitchvlm.datasets import make_split_manifest
from pitchvlm.evalx.oracle import TeamSideFacts
from pitchvlm.schemas import ClipResult
from pitchvlm.synth import SyntheticScene
from pitchvlm.vlm import client as client_module
from pitchvlm.vlm import steps
from pitchvlm.vlm.cache import CachedBackend, CassetteStore
from pitchvlm.vlm.client import GeminiBackend
from pitchvlm.vlm.cost import BudgetExceeded, BudgetGuard, Ledger, TokenEstimate, cost_usd, has_flex_tier
from pitchvlm.vlm.fake import STEP_USAGE, FakeBackend
from pitchvlm.vlm.types import Backend, BackendError, BilledAnswerError, FlexBusyError, VLMRequest, VLMResponse

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "configs" / "experiments"
GATES = REPO / "configs" / "gates.yaml"
HOOK_FILE = REPO / ".pre-commit-config.yaml"
#: The git hooks that the doctor needs: the guard runs in both.
HOOK_NAMES = ("pre-commit", "pre-push")
#: The folders and files of the repository that hold its text.
REPOSITORY_TEXT = (
    "src",
    "tests",
    "docs",
    "configs",
    "tools",
    "CLAUDE.md",
    "README.md",
    "pyproject.toml",
    ".gitignore",
    ".pre-commit-config.yaml",
)
#: Clip id of every synthetic scene.
CLIP_ID = "SNGS-999"
LABEL_NAME = "Labels-GameState.json"
N_FRAMES = 50
KEYFRAMES = (0, 12, 24, 36, 48, 49)
#: File name of a cassette: 32 hexadecimal digits.
CASSETTE_STEM = "0123456789abcdef" * 2
#: The thresholds of gate G0, as ``configs/gates.yaml`` holds them.
LIMITS = doctor.ContractLimits(
    box_mean_iou_min=0.6,
    point_error_max_fraction=0.01,
    usage_sum_tolerance=0.01,
    flex_valid_answer_rate_min=0.95,
)
#: The money limits of the project rules, as ``configs/gates.yaml`` holds them.
BUDGET = cli.Budget(planned_usd=120.0, hard_stop_usd=135.0, run_factor=1.3, max_usd_per_10_min=8.0)
#: Scores, as shares from 0 to 1, that pass gate G1: 100, 89.5, 8.5 and 95 percent.
PASSING_SHARES = {
    "oracle_ground_truth": 1.0,
    "oracle_shifted_1m": 0.895,
    "oracle_keyframes_only": 0.085,
    "oracle_resampled": 0.95,
    "default": 0.0,
}

#: One argument list per command, with every required option.
COMMAND_LINES = {
    "demo": ["demo", "--out", "out"],
    "run": ["run", "--clip", "clip.mp4", "--out", "out"],
    "doctor": ["doctor", "--no-api", "--models", "gemini-4-argon,gemini-3.8-flash"],
    "data": ["data", "--root", "data", "--split", "valid", "--clips", "SNGS-021", "SNGS-022"],
    "oracle": ["oracle", "--root", "data", "--scorer-python", "python", "--scorer-script", "run.py", "--out", "out"],
    "probe": ["probe", "people", "--experiment", "people.yaml", "--root", "data", "--out", "metrics.json"],
    "tool-job": ["tool-job", "--experiment", "tool.yaml", "--root", "data", "--out", "job.json"],
    "tool-score": [
        "tool-score",
        "--experiment",
        "tool.yaml",
        "--root",
        "data",
        "--run",
        "run.json",
        "--out",
        "metrics.json",
    ],
    "eval": [
        "eval",
        "--root",
        "data",
        "--pred",
        "pred",
        "--split",
        "valid",
        "--tracker",
        "pitchvlm",
        "--clips",
        "SNGS-021",
        "--scorer-python",
        "python",
        "--scorer-script",
        "run.py",
        "--out",
        "out",
    ],
    "sync": ["sync", "--results", "results"],
}

#: Stand-in for the evaluator script. It takes every setting of the runner script of sn-trackeval as a
#: flag, and it looks for the label file and the prediction file of every clip where the evaluator looks
#: for them. It writes the three detailed CSV files of the evaluator into the folder of the prediction
#: files of the tracker: person_detailed.csv with one row per clip and the row COMBINED, and the files
#: of the two class averages with the row COMBINED only. The HOTA of a tracker comes from the file
#: shares.json next to the script. DetA is that share minus 0.02 and AssA is that share minus 0.04 (not
#: below 0), so that the three scores can be told apart.
FAKE_EVALUATOR = """\
import argparse
import csv
import json
from pathlib import Path

# Every setting of the runner script is a flag. The settings of the first group take one or more values.
LISTS = "OUTPUT_FOLDER TRACKERS_TO_EVAL TRACKER_DISPLAY_NAMES SEQMAP_FOLDER SEQMAP_FILE SEQ_INFO METRICS"
SINGLES = (
    "USE_PARALLEL NUM_PARALLEL_CORES BREAK_ON_ERROR RETURN_ON_ERROR LOG_ON_ERROR PRINT_RESULTS "
    "PRINT_ONLY_COMBINED PRINT_CONFIG TIME_PROGRESS DISPLAY_LESS_PROGRESS OUTPUT_SUMMARY "
    "OUTPUT_EMPTY_CLASSES OUTPUT_DETAILED PLOT_CURVES GT_FOLDER TRACKERS_FOLDER SPLIT_TO_EVAL EVAL_MODE "
    "EVAL_SPACE EVAL_SIMILARITY_METRIC EVAL_DIST_TOL USE_ROLES USE_TEAMS USE_JERSEY_NUMBERS IGNORE_BALL "
    "INPUT_AS_ZIP DO_PREPROC TRACKER_SUB_FOLDER OUTPUT_SUB_FOLDER GT_LOC_FORMAT SKIP_SPLIT_FOL THRESHOLD"
)
parser = argparse.ArgumentParser()
for setting in LISTS.split():
    parser.add_argument("--" + setting, nargs="+")
for setting in SINGLES.split():
    parser.add_argument("--" + setting)
parser.set_defaults(SPLIT_TO_EVAL="valid", TRACKER_SUB_FOLDER="data", SKIP_SPLIT_FOL="False")
args = parser.parse_args()
tracker = args.TRACKERS_TO_EVAL[0]
# Without --SKIP_SPLIT_FOL True the evaluator adds a split folder to both folders: <split> for the
# labels and SoccerNetGS-<split> for the predictions.
if args.SKIP_SPLIT_FOL == "True":
    label_folder, prediction_folder = Path(args.GT_FOLDER), Path(args.TRACKERS_FOLDER)
else:
    label_folder = Path(args.GT_FOLDER) / args.SPLIT_TO_EVAL
    prediction_folder = Path(args.TRACKERS_FOLDER) / ("SoccerNetGS-" + args.SPLIT_TO_EVAL)
tracker_folder = prediction_folder / tracker
for clip in args.SEQ_INFO:
    label_file = label_folder / clip / "Labels-GameState.json"
    if not label_file.is_file():
        raise SystemExit("GT file not found " + str(label_file))
    prediction_file = tracker_folder / args.TRACKER_SUB_FOLDER / (clip + ".json")
    if not prediction_file.is_file():
        raise SystemExit("Tracker file not found: " + str(prediction_file))
shares = json.loads(Path(__file__).with_name("shares.json").read_text(encoding="utf-8"))
hota = shares.get(tracker, shares["default"])
det_a, ass_a = max(hota - 0.02, 0.0), max(hota - 0.04, 0.0)
# The class person has one row per clip and the row COMBINED. The two class averages have that row only.
combined = ["COMBINED", hota, det_a, ass_a]
tables = {
    "person": [[clip, hota, det_a, ass_a] for clip in sorted(args.SEQ_INFO)] + [combined],
    "cls_comb_cls_av": [combined],
    "cls_comb_det_av": [combined],
}
for name, rows in tables.items():
    with (tracker_folder / (name + "_detailed.csv")).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seq", "HOTA___AUC", "DetA___AUC", "AssA___AUC"])
        writer.writerows(rows)
"""

#: A program that runs the command line with a fake model whose requests wait for a gate file. It
#: writes the file "waiting" next to the gate when the first request has started. Arguments: the gate
#: file, then the arguments of the command line.
GATED_RUN = """\
import sys
import time
from pathlib import Path

from pitchvlm import cli
from pitchvlm.synth import SyntheticScene
from pitchvlm.vlm.fake import FakeBackend

gate = Path(sys.argv[1])
truth = FakeBackend(SyntheticScene(n_frames=50, seed=2))


class GatedBackend:
    def generate(self, request):
        gate.with_name("waiting").write_text("the first request has started", encoding="utf-8")
        while not gate.exists():
            time.sleep(0.01)
        return truth.generate(request)


cli.gemini_backend = lambda max_usd_per_10_min: GatedBackend()
cli.render_clip = lambda source, result, out_path: out_path
sys.exit(cli.main(sys.argv[2:]))
"""


# --- Helpers ---------------------------------------------------------------------------


class SwappedAxesBackend:
    """A backend that answers like the fake backend, but writes x before y in every box and every point."""

    def __init__(self, inner: FakeBackend) -> None:
        self._inner = inner

    def generate(self, request: VLMRequest) -> VLMResponse:
        response = self._inner.generate(request)
        answer = json.loads(response.text)
        for person in answer.get("people", []):
            ymin, xmin, ymax, xmax = person["box_2d"]  # A model box is [ymin, xmin, ymax, xmax].
            person["box_2d"] = [xmin, ymin, xmax, ymax]
        for landmark in answer.get("landmarks", []):
            y, x = landmark["point"]  # A model point is [y, x].
            landmark["point"] = [x, y]
        return response.model_copy(update={"text": json.dumps(answer)})


class NotJsonBackend:
    """A backend that answers like the fake backend, but with a text that is not JSON."""

    def __init__(self, inner: FakeBackend) -> None:
        self._inner = inner

    def generate(self, request: VLMRequest) -> VLMResponse:
        return self._inner.generate(request).model_copy(update={"text": "not JSON"})


class RejectedAnswerBackend:
    """A backend that answers like the fake backend, but one answer is billed and then rejected.

    The first request of the given step raises ``BilledAnswerError``, as the real backend does for an
    answer without text. The cost record of the error holds the tokens of the fake answer.
    """

    def __init__(self, inner: FakeBackend, step: str) -> None:
        self._inner = inner
        self._step = step
        self._rejected = False

    def generate(self, request: VLMRequest) -> VLMResponse:
        response = self._inner.generate(request)
        if self._rejected or request.meta.step != self._step:
            return response
        self._rejected = True
        raise BilledAnswerError(f"the {self._step} answer has no text", response.model_copy(update={"text": ""}))


class EditedAnswersBackend:
    """A backend that answers like another backend, after ``edit(request, answer)`` changed the answer in place."""

    def __init__(self, inner: Backend, edit: Callable[[VLMRequest, dict[str, Any]], None]) -> None:
        self._inner = inner
        self._edit = edit

    def generate(self, request: VLMRequest) -> VLMResponse:
        response = self._inner.generate(request)
        answer = json.loads(response.text)
        self._edit(request, answer)
        return response.model_copy(update={"text": json.dumps(answer)})


class FlexRefusingBackend:
    """A backend that answers like another backend, but refuses every request for the flex tier.

    It keeps the service tier of every request that it gets.
    """

    def __init__(self, inner: Backend) -> None:
        self._inner = inner
        self.tiers: list[str] = []

    def generate(self, request: VLMRequest) -> VLMResponse:
        self.tiers.append(request.service_tier)
        if request.service_tier == "flex":
            raise BackendError("the flex tier is not there for this model")
        return self._inner.generate(request)


class RawCallBackend:
    """A backend that answers like another backend and adds one raw call to a stand-in recorder per request."""

    def __init__(self, inner: Backend, recorder: SimpleNamespace, call: doctor.RawCall) -> None:
        self._inner = inner
        self._recorder = recorder
        self._call = call

    def generate(self, request: VLMRequest) -> VLMResponse:
        self._recorder.calls.append(self._call)
        return self._inner.generate(request)


class CountingBackend:
    """A backend that answers like another backend and keeps every request that it gets."""

    def __init__(self, inner: Backend) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self.requests: list[VLMRequest] = []

    def generate(self, request: VLMRequest) -> VLMResponse:
        with self._lock:
            self.requests.append(request)
        return self._inner.generate(request)


class FlexBusyBackend:
    """A backend that answers like another backend, but finds no capacity in the flex tier for some requests.

    The flex request of a frame in ``busy_frames`` raises ``FlexBusyError``, as the real backend does
    when the flex tier answered 503 or 429 for the whole waiting time.
    """

    def __init__(self, inner: Backend, busy_frames: Sequence[int]) -> None:
        self._inner = inner
        self._busy_frames = tuple(busy_frames)

    def generate(self, request: VLMRequest) -> VLMResponse:
        if request.service_tier == "flex" and request.meta.frame in self._busy_frames:
            raise FlexBusyError(f"the flex tier had no capacity for frame {request.meta.frame}")
        return self._inner.generate(request)


class SignallingBackend:
    """A backend that answers like another backend, and one of its requests sends a stop signal to this process.

    Every request takes 200 milliseconds. The request with the number ``at_request`` (counted from 1)
    sends the signal when it starts, so the signal comes while that request is in flight. ``started``
    counts the requests, and ``answered`` keeps the cache key of every request that got its answer.
    """

    def __init__(self, inner: Backend, at_request: int, number: int = signal.SIGTERM) -> None:
        self._inner = inner
        self._at_request = at_request
        self._number = number
        self._lock = threading.Lock()
        self.started = 0
        self.answered: list[str] = []

    def generate(self, request: VLMRequest) -> VLMResponse:
        with self._lock:
            self.started += 1
            is_the_one = self.started == self._at_request
        if is_the_one:
            os.kill(os.getpid(), self._number)
        time.sleep(0.2)
        response = self._inner.generate(request)
        with self._lock:
            self.answered.append(request.cache_key())
        return response


class FakeApiError(Exception):
    """Stands for the SDK error of a call that the server answered with an error: the code is in ``status_code``."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"fake API error {status_code}")
        self.status_code = status_code


class FakeClock:
    """A clock that moves only when the fake sleep or a fake SDK call moves it."""

    def __init__(self) -> None:
        self.now = 1000.0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class ContractSdkClient:
    """Stands for the SDK client in a doctor run: it answers the requests for the contract frames with the truth.

    A request is known by its last image, which is one of the contract frames. Every call takes two
    seconds on the fake clock. A call for the flex tier ends with the status ``flex_status`` while
    fewer than ``flex_busy_s`` seconds have passed since the first flex call. ``flex_calls`` counts the
    calls for the flex tier.
    """

    def __init__(self, scene: SyntheticScene, clock: FakeClock, *, flex_status: int, flex_busy_s: float) -> None:
        self._scene = scene
        self._truth = FakeBackend(scene)
        self._clock = clock
        self._flex_status = flex_status
        self._flex_busy_s = flex_busy_s
        self._first_flex_call: float | None = None
        self._frames = {
            base64.b64encode(steps.encode_jpeg(scene.image(frame))).decode("ascii"): frame
            for frame in doctor.CONTRACT_FRAMES
        }
        self.flex_calls = 0

    @property
    def interactions(self) -> ContractSdkClient:
        """Return the object that has ``create``: this client itself."""
        return self

    def create(self, **kwargs: Any) -> SimpleNamespace:
        tier = kwargs["service_tier"]
        if tier == "flex":
            self.flex_calls += 1
            if self._first_flex_call is None:
                self._first_flex_call = self._clock.now
            if self._clock.now - self._first_flex_call < self._flex_busy_s:
                self._clock.now += 2.0
                raise FakeApiError(self._flex_status)
        self._clock.now += 2.0
        info, legend = self._scene.info, self._scene.kit_legend()
        frame = self._frames[[item for item in kwargs["input"] if item["type"] == "image"][-1]["data"]]
        image = self._scene.image(frame)
        settings = StepSettings(model=kwargs["model"], service_tier=tier)
        if "people" in kwargs["response_format"]["schema"]["properties"]:
            request = steps.build_people_request(image, legend, clip_id=info.clip_id, frame=frame, settings=settings)
        else:
            request = steps.build_landmarks_request(image, clip_id=info.clip_id, frame=frame, settings=settings)
        answer = self._truth.generate(request)
        used = answer.usage
        usage = SimpleNamespace(
            total_input_tokens=used.input_tokens,
            total_output_tokens=used.output_tokens,
            total_thought_tokens=used.thought_tokens,
            total_tokens=used.input_tokens + used.output_tokens + used.thought_tokens,
        )
        return SimpleNamespace(status="completed", output_text=answer.text, service_tier=tier, usage=usage)


def without_landmarks(request: VLMRequest, answer: dict[str, Any]) -> None:
    """Empty a landmarks answer. With ``EditedAnswersBackend``, no keyframe of a clip gets a pitch mapping then."""
    if request.meta.step == "landmarks":
        answer["landmarks"] = []


def use_fake_model(monkeypatch: pytest.MonkeyPatch, scene: SyntheticScene) -> CountingBackend:
    """Make the command line use the fake model in the place of the real one, and write no video.

    The result keeps every request that reaches the fake model: these are the paid requests.
    """
    backend = CountingBackend(FakeBackend(scene))
    monkeypatch.setattr(cli, "render_clip", lambda source, result, out_path: out_path)
    monkeypatch.setattr(cli, "gemini_backend", lambda max_usd_per_10_min: backend)
    return backend


def estimates_of_a_share(monkeypatch: pytest.MonkeyPatch, share: float) -> None:
    """Make the command line estimate every answer at a share of the tokens that the fake model reports."""
    estimates = {
        step: TokenEstimate(
            input_tokens=round(share * usage.input_tokens),
            output_tokens=round(share * (usage.output_tokens + usage.thought_tokens)),
        )
        for step, usage in STEP_USAGE.items()
    }
    monkeypatch.setattr(cli, "DEFAULT_TOKEN_ESTIMATES", estimates)


def fake_cost(step: str, model: str = "gemini-4-argon") -> float:
    """Return the cost in USD of one answer of the fake model at the standard price of today."""
    return cost_usd(model, STEP_USAGE[step], "standard", datetime.now(UTC).date())


def gates_with(**sections: dict[str, Any]) -> dict[str, Any]:
    """Return the gates of the repository with other values in some sections."""
    gates = copy.deepcopy(yaml.safe_load(GATES.read_text(encoding="utf-8")))
    for section, values in sections.items():
        gates[section].update(values)
    return gates


def run_arguments(clip: Path, tmp_path: Path, *more: str) -> list[str]:
    """Return the command line of a run of a clip, with the output folder and the cache folder below ``tmp_path``."""
    return [
        "run",
        "--clip",
        str(clip),
        "--out",
        str(tmp_path / "runs"),
        "--cache",
        str(tmp_path / "cache"),
        "--allow-dirty",
        *more,
    ]


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_ledger(cache: Path) -> list[dict[str, Any]]:
    """Read the lines of the cost ledger of a cache folder."""
    return [json.loads(line) for line in (cache / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]


def write_earlier_spend(cache: Path, usd: float) -> None:
    """Write a cost ledger that holds the spend of one earlier run."""
    cache.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"run_id": "an-earlier-run", "usd": usd})
    (cache / "ledger.jsonl").write_text(line + "\n", encoding="utf-8")


def mp4v_available(folder: Path) -> bool:
    """Tell whether the installed OpenCV build can write a video with the codec mp4v."""
    probe = cv2.VideoWriter(str(folder / "codec_probe.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
    opened = probe.isOpened()
    probe.release()
    return opened


def read_video(path: Path) -> list[np.ndarray]:
    """Read every frame of a video file."""
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    frames: list[np.ndarray] = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frames.append(image)
    capture.release()
    return frames


def statuses(checks: Sequence[doctor.CheckResult]) -> dict[str, str]:
    """Map the name of every check to its status."""
    return {check.name: check.status for check in checks}


def by_name(checks: Sequence[doctor.CheckResult]) -> dict[str, doctor.CheckResult]:
    """Map the name of every check to the check."""
    return {check.name: check for check in checks}


def raw_call(**overrides: Any) -> doctor.RawCall:
    """Return the raw facts of a call whose token total adds up: 1500 + 300 + 100 = 1900."""
    facts: dict[str, Any] = {
        "store": False,
        "sent_tier": "standard",
        "status": "completed",
        "served_tier": None,
        "input_tokens": 1500,
        "output_tokens": 300,
        "thought_tokens": 100,
        "total_tokens": 1900,
    }
    facts.update(overrides)
    return doctor.RawCall(**facts)


def framework_hook(name: str) -> str:
    """Return the text of a hook file as the pre-commit framework writes it for one hook and the default hook list."""
    return (
        "#!/usr/bin/env bash\n"
        "# start templated\n"
        f"ARGS=(hook-impl --config=.pre-commit-config.yaml --hook-type={name})\n"
        "# end templated\n"
        'exec pre-commit "${ARGS[@]}"\n'
    )


def make_repository(repo: Path) -> None:
    """Make a folder a git repository without any hook, with the hook list of this repository. Needs ``private_git``."""
    repo.mkdir(parents=True, exist_ok=True)
    if not (repo / ".git").exists():
        git(repo, "init", "--quiet", "--template=")
    shutil.copy(HOOK_FILE, repo / ".pre-commit-config.yaml")


def write_hook(folder: Path, name: str, text: str, *, executable: bool = True) -> None:
    """Write one hook file into a hook folder."""
    folder.mkdir(parents=True, exist_ok=True)
    hook = folder / name
    hook.write_text(text, encoding="utf-8")
    hook.chmod(0o755 if executable else 0o644)


def install_hooks(repo: Path, names: Sequence[str] = HOOK_NAMES) -> None:
    """Make a folder a git repository whose hooks are installed as the pre-commit framework installs them.

    The default is both hooks. The test needs the fixture ``private_git``.
    """
    make_repository(repo)
    for name in names:
        write_hook(repo / ".git" / "hooks", name, framework_hook(name))


def write_deny_list(repo: Path, text: str = "# This clone has no private word.\n") -> None:
    """Write the deny-list of the guard into the git folder of a repository."""
    (repo / ".git" / "guard-deny.txt").write_text(text, encoding="utf-8")


def set_up_clone(repo: Path) -> None:
    """Make a folder a git repository that passes the checks of the machine: both hooks and a deny-list.

    The test needs the fixture ``private_git``.
    """
    install_hooks(repo)
    write_deny_list(repo)


def set_up_doctor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make the model part of the doctor command start in a test, and return the folder of its repository.

    The repository is a clone that passes the checks of the machine, with the gate file. The key
    variable is set, and the SDK check sees a version 3. The test needs the fixture ``private_git``.
    """
    repo = tmp_path / "repo"
    set_up_clone(repo)
    (repo / "configs").mkdir()
    shutil.copy(GATES, repo / "configs" / "gates.yaml")
    monkeypatch.setattr(cli, "repository_root", lambda: repo)
    monkeypatch.setenv(cli.KEY_VARIABLE, "not-a-real-value")
    monkeypatch.setattr(doctor, "installed_sdk_version", lambda: "3.0.test")
    return repo


def environment_check(repo: Path, name: str, *, api: bool = False) -> doctor.CheckResult:
    """Return one check of the machine for a repository folder."""
    checks = doctor.environment_checks(repo, key_variable=cli.KEY_VARIABLE, data_dir=repo / "none", api=api)
    return by_name(checks)[name]


def hook_check(repo: Path) -> doctor.CheckResult:
    """Return the check of the guard hooks of a repository folder."""
    return environment_check(repo, doctor.HOOK)


def hook_entries() -> dict[str, dict[str, Any]]:
    """Read the hooks of ``.pre-commit-config.yaml``, by hook id."""
    config = yaml.safe_load(HOOK_FILE.read_text(encoding="utf-8"))
    return {hook["id"]: hook for repo in config["repos"] for hook in repo["hooks"]}


def write_evaluator(folder: Path, shares: dict[str, float]) -> Path:
    """Write the stand-in evaluator script and its scores into a folder, and return the script."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "shares.json").write_text(json.dumps(shares), encoding="utf-8")
    script = folder / "fake_evaluator.py"
    script.write_text(FAKE_EVALUATOR, encoding="utf-8")
    return script


def make_archive(path: Path, clips: Sequence[str]) -> None:
    """Write a split archive that holds one clip folder with a label file per clip name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as bundle:
        for clip in clips:
            bundle.writestr(f"{clip}/{LABEL_NAME}", "{}")


def git(folder: Path, *arguments: str) -> str:
    """Run git in a folder of the test and return what it printed."""
    completed = subprocess.run(["git", *arguments], cwd=folder, capture_output=True, text=True, check=True)
    return completed.stdout


def write_run_results(folder: Path, scene: SyntheticScene) -> None:
    """Write text results as a paid run writes them: cassettes of the four model steps, the ledger and a metrics file.

    The model is the fake backend. The requests are one kit legend request, a people and a landmarks
    request for each of the six keyframes, and four number requests: 17 cassettes.
    """
    info, legend, settings = scene.info, scene.kit_legend(), StepSettings()
    cache = folder / "cache"
    ledger = Ledger(cache / "ledger.jsonl")
    backend = CachedBackend(
        CassetteStore(cache),
        FakeBackend(scene),
        on_new_answer=lambda request, response: ledger.record("run-1", request, response),
    )
    requests = [steps.build_kit_legend_request([scene.image(0)], clip_id=info.clip_id, settings=settings)]
    for frame in KEYFRAMES:
        image = scene.image(frame)
        requests.append(steps.build_people_request(image, legend, clip_id=info.clip_id, frame=frame, settings=settings))
        requests.append(steps.build_landmarks_request(image, clip_id=info.clip_id, frame=frame, settings=settings))
    first_image = scene.image(0)
    for track_id, person in enumerate(scene.people(0).people[:4]):
        left, top, right, bottom = steps.crop_rect(person.box, info.width, info.height)
        requests.append(
            steps.build_number_request(
                [first_image[top:bottom, left:right]],  # rows are y, columns are x
                legend,
                person.team_key,
                clip_id=info.clip_id,
                track_id=track_id,
                crop_frames=[0],
                crop_boxes=[person.box],
                image_size=(info.width, info.height),
                crop_upscale=1.0,
                settings=settings,
            )
        )
    for request in requests:
        backend.generate(request)
    made_up_commit = "".join(random.Random(3).choices("0123456789abcdef", k=40))
    cli.write_metrics(
        folder / "metrics" / "people_probe.json", {"recall": 0.9}, provenance={"commit": made_up_commit, "dirty": False}
    )


#: Run files of the two external tools, as ``tools/run_tool.py`` writes them: 40 frames each. A box
#: at the image border has whole numbers, and every third frame of the calibration has no camera.
TOOL_RUNS: dict[str, dict[str, Any]] = {
    "people_probe_rfdetr_small.json": {
        "format": "pitchvlm-tool-run-1",
        "name": "people_probe_rfdetr_small",
        "tool": {"name": "rfdetr", "model": "RFDETRSmall", "min_score": 0.5, "antialias": False},
        "runner": {
            "device": "mps",
            "machine": "arm64",
            "system": "Darwin",
            "python": "3.12.8",
            "torch": "2.9.0",
            "script_sha256": hashlib.sha256(b"the script that ran the detector").hexdigest(),
            "tool_version": "1.12.0",
            "predictor": "predict() of the model, not compiled",
        },
        "frames": [
            {
                "clip": f"SNGS-{21 + index // 5:03d}",
                "frame": 37 + 150 * (index % 5),
                "seconds": 0.0201 + 0.001 * index,
                "payload": {
                    "detections": [
                        {
                            "box_xyxy": [0.0, 12.0, 255.0, 128.0],
                            "score": 0.5 + 0.01 * index,
                            "label": "person",
                        },
                        *(
                            {
                                "box_xyxy": [
                                    100.25 * person + index,
                                    40.5 * person,
                                    100.25 * person + index + 31.75,
                                    40.5 * person + 96.125,
                                ],
                                "score": 0.9375 - 0.01 * person,
                                "label": "person" if person % 7 else "sports ball",
                            }
                            for person in range(1, 19)
                        ),
                    ]
                },
            }
            for index in range(40)
        ],
    },
    "pitch_probe_pnlcalib.json": {
        "format": "pitchvlm-tool-run-1",
        "name": "pitch_probe_pnlcalib",
        "tool": {"name": "pnlcalib", "kp_threshold": 0.3434, "line_threshold": 0.7867, "pnl_refine": True},
        "runner": {
            "device": "mps",
            "machine": "arm64",
            "system": "Darwin",
            "python": "3.12.8",
            "torch": "2.3.1",
            "script_sha256": hashlib.sha256(b"the script that ran the calibration").hexdigest(),
            "tool_commit": "42088864170f75ea02e76d20e9a9a1db59edb3e3",
            "weights_kp": "SV_kp",
            "weights_kp_sha256": hashlib.sha256(b"weights of the keypoint model").hexdigest(),
            "weights_lines": "SV_lines",
            "weights_lines_sha256": hashlib.sha256(b"weights of the line model").hexdigest(),
            "calibration_size": "the size of the frame, as in inference.py",
        },
        "frames": [
            {
                "clip": f"SNGS-{21 + index // 5:03d}",
                "frame": 37 + 150 * (index % 5),
                "seconds": 0.61 + 0.01 * index,
                "payload": (
                    {"homography_pitch_to_image": None, "fault": "the tool found no camera"}
                    if index % 3 == 0
                    else {
                        "homography_pitch_to_image": [
                            [1800.0 + index, 959.5, 67165.0],
                            [0.0, 754.625, 9972.03125],
                            [0.0, 0.9778, 71.589],
                        ],
                        "pitch_origin": "centre",
                        "rep_err": 1.5 + 0.125 * index,
                        "mode": "full",
                        "use_ransac": 5.0,
                        "calib_plane": 0,
                        "camera_position_m": [0.3125 + index, 58.625, -11.9375],
                    }
                ),
            }
            for index in range(40)
        ],
    },
}


def write_results(folder: Path) -> None:
    """Write a small result folder: one metrics file, one cassette and the cost ledger."""
    (folder / "metrics").mkdir(parents=True)
    (folder / "metrics" / "people_probe.json").write_text(json.dumps({"recall": 0.9}) + "\n", encoding="utf-8")
    cassette = folder / "cache" / "SNGS-021" / "people" / f"{CASSETTE_STEM}.json"
    cassette.parent.mkdir(parents=True)
    cassette.write_text(json.dumps({"response": {"text": "{}"}}) + "\n", encoding="utf-8")
    (folder / "cache" / "ledger.jsonl").write_text(json.dumps({"run_id": "r", "usd": 0.1}) + "\n", encoding="utf-8")


# --- Fixtures --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def estimates_of_the_fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the command line estimate every cost from the token counts that the fake model reports.

    The token estimates of ``vlm.cost`` change when real token counts are measured, and the tests
    of this file must not depend on their values. With the counts of the fake model, an answer of
    the fake model costs what its estimate says. So the budget guard stops no test run, whatever
    the estimates of ``vlm.cost`` are. The output count of an estimate includes the thinking tokens.
    """
    estimates = {
        step: TokenEstimate(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens + usage.thought_tokens)
        for step, usage in STEP_USAGE.items()
    }
    monkeypatch.setattr(cli, "DEFAULT_TOKEN_ESTIMATES", estimates)


@pytest.fixture(scope="module")
def contract_scene() -> SyntheticScene:
    """The synthetic scene of the contract checks: 1920 x 1080 pixels."""
    return doctor.contract_scene()


@pytest.fixture(scope="module")
def clip_on_disk(tmp_path_factory: pytest.TempPathFactory) -> tuple[SyntheticScene, Path]:
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
def private_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Make git read no user and no system configuration, and find no repository above the test folder."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    home = tmp_path / "home"
    home.mkdir()
    for name in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))


@pytest.fixture
def results_tree(private_git: None, tmp_path: Path) -> Path:
    """A git work tree on the branch ``results``, without any commit."""
    tree = tmp_path / "pitchvlm-results"
    tree.mkdir()
    git(tree, "init", "--quiet", "--template=")
    git(tree, "symbolic-ref", "HEAD", "refs/heads/results")
    return tree


# --- Argument parser -------------------------------------------------------------------


@pytest.mark.parametrize("command", sorted(COMMAND_LINES))
def test_parser_accepts_each_command(command: str) -> None:
    args = cli.build_parser().parse_args(COMMAND_LINES[command])

    assert args.command == command
    assert callable(args.handler)


def test_parser_reads_options_and_defaults() -> None:
    parser = cli.build_parser()

    demo = parser.parse_args(COMMAND_LINES["demo"])
    assert (demo.out, demo.frames, demo.seed) == (Path("out"), 100, 0)
    run = parser.parse_args(COMMAND_LINES["run"])
    assert (run.split, run.tracker, run.replay, run.allow_dirty) == ("valid", "pitchvlm", False, False)
    assert (run.config, run.roster, run.labels, run.cache, run.fps) == (None, None, None, None, None)
    assert run.allow_other_images is False
    assert parser.parse_args([*COMMAND_LINES["run"], "--allow-other-images"]).allow_other_images is True
    doctor_args = parser.parse_args(COMMAND_LINES["doctor"])
    assert doctor_args.no_api is True
    assert doctor_args.models == ["gemini-4-argon", "gemini-3.8-flash"]
    assert parser.parse_args(["doctor"]).models == ["gemini-4-argon"]
    data = parser.parse_args(COMMAND_LINES["data"])
    assert (data.clips, data.manifest, data.seed) == (["SNGS-021", "SNGS-022"], None, None)
    oracle = parser.parse_args(COMMAND_LINES["oracle"])
    assert (oracle.split, oracle.clips) == ("valid", None)
    probe = parser.parse_args([*COMMAND_LINES["probe"], "--max-clips", "2", "--replay"])
    assert (probe.kind, probe.max_clips, probe.replay, probe.manifest) == ("people", 2, True, None)
    assert probe.allow_other_images is False
    assert parser.parse_args([*COMMAND_LINES["probe"], "--allow-other-images"]).allow_other_images is True
    tool_job = parser.parse_args([*COMMAND_LINES["tool-job"], "--max-clips", "2"])
    assert (tool_job.experiment, tool_job.out, tool_job.max_clips, tool_job.manifest) == (
        Path("tool.yaml"),
        Path("job.json"),
        2,
        None,
    )
    tool_score = parser.parse_args(COMMAND_LINES["tool-score"])
    assert (tool_score.run, tool_score.out) == (Path("run.json"), Path("metrics.json"))
    assert (tool_score.max_clips, tool_score.manifest, tool_score.allow_dirty) == (None, None, False)
    # The two tool commands call no model: they have no option for stored answers or for the budget.
    for arguments in (COMMAND_LINES["tool-job"], COMMAND_LINES["tool-score"]):
        for option in (["--cache", "cache"], ["--replay"], ["--start-over-budget"]):
            with pytest.raises(SystemExit):
                parser.parse_args([*arguments, *option])
    # The job file is no result: the command that writes it does not look at the state of the work tree.
    with pytest.raises(SystemExit):
        parser.parse_args([*COMMAND_LINES["tool-job"], "--allow-dirty"])


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["demo"],
        ["demo", "--out", "out", "--frames", "0"],
        ["probe", "shirts", "--experiment", "x.yaml", "--root", "data", "--out", "m.json"],
        ["tool-job", "--experiment", "x.yaml", "--root", "data"],
        ["tool-score", "--experiment", "x.yaml", "--root", "data", "--out", "m.json"],
        ["tool-score", "--experiment", "x.yaml", "--root", "d", "--run", "r.json", "--out", "m.json", "--max-clips=0"],
        ["data", "--root", "data", "--split", "train"],
        ["doctor", "--models", "gemini-4-argon,,gemini-3.8-flash"],
        ["eval", "--root", "data", "--pred", "pred", "--split", "valid", "--tracker", "t", "--out", "out"],
    ],
)
def test_parser_refuses_wrong_command_lines(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        cli.build_parser().parse_args(arguments)

    assert stopped.value.code == 2


# --- demo ------------------------------------------------------------------------------


def test_demo_writes_predictions_summary_and_video(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The offline run of the whole package: synthetic clip, fake model, pipeline, export, checker and video."""
    out = tmp_path / "demo"
    arguments = ["demo", "--out", str(out), "--frames", str(N_FRAMES)]
    can_write_video = mp4v_available(tmp_path)

    code = cli.main(arguments)

    if can_write_video:
        assert code == 0
    else:
        # A video that cannot be written is an error (exit code 2), not a failed check (exit code 1).
        # It is an error of the operating system, so one line names it. The prediction file and
        # the summary are written before the video.
        assert code == 2
        stderr = capsys.readouterr().err
        assert "Traceback" not in stderr
        assert stderr.splitlines()[-1].startswith("error: ")

    items = read_json(out / "predictions" / "SoccerNetGS-demo" / "pitchvlm" / f"{CLIP_ID}.json")["predictions"]
    # Every frame has predictions, and the image ids are those of the label file: frame 0 is image 1.
    assert {item["image_id"] for item in items} == {f"2999{number:06d}" for number in range(1, N_FRAMES + 1)}
    summary = read_json(out / "summary.json")
    assert summary["clip_id"] == CLIP_ID
    assert summary["n_frames"] == N_FRAMES
    assert 10 <= summary["n_tracks"] <= 30
    assert summary["median_position_error_m"] < 1.0
    assert summary["position_error_m"]["n"] > 300
    # The identity values see what the median position error does not: a person who gets another track id.
    identity = summary["identity"]
    assert set(identity) == {"coverage", "switches", "track_ratio"}
    assert identity["coverage"] > 0.8
    assert identity["switches"] <= 2
    assert 0.8 <= identity["track_ratio"] <= 1.3
    assert summary["checker"] == {"valid": True, "problems": []}
    assert (summary["scored_frames"], summary["scored_frames_without_prediction"]) == (N_FRAMES, 0)
    assert summary["provenance"]["command"] == "demo"

    if not can_write_video:
        pytest.skip("the installed OpenCV build cannot write the codec mp4v, so the video was not checked")
    frames = read_video(out / "demo.mp4")
    assert len(frames) == N_FRAMES
    # The frame is on top and the radar is below it, at the same width.
    assert frames[0].shape[1] == 1280
    assert frames[0].shape[0] > 720


def test_demo_ends_with_code_1_when_the_checker_reports_a_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "demo"
    monkeypatch.setattr(cli, "check_predictions", lambda payload, **options: ["jersey: a made-up problem"])
    monkeypatch.setattr(cli, "render_clip", lambda source, result, out_path: out_path)

    assert cli.main(["demo", "--out", str(out), "--frames", "26", "--seed", "4"]) == 1

    assert read_json(out / "summary.json")["checker"] == {"valid": False, "problems": ["jersey: a made-up problem"]}
    assert "a made-up problem" in capsys.readouterr().out


def test_demo_ends_with_code_1_when_no_scored_frame_has_a_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No landmarks answer names a landmark: no keyframe has a pitch mapping, so no person gets a position."""
    out = tmp_path / "demo"
    monkeypatch.setattr(
        cli, "FakeBackend", lambda scene, seed: EditedAnswersBackend(FakeBackend(scene, seed=seed), without_landmarks)
    )
    monkeypatch.setattr(cli, "render_clip", lambda source, result, out_path: out_path)

    assert cli.main(["demo", "--out", str(out), "--frames", "26", "--seed", "4"]) == 1

    output = capsys.readouterr().out
    assert "format checker: the prediction file is valid" in output
    assert "scored frames without a prediction: 26 of 26" in output
    assert "failed: no scored frame has a prediction, so the clip gives no result." in output
    # The files are written: a clip without a result is a failed check, not an error.
    summary = read_json(out / "summary.json")
    assert summary["checker"] == {"valid": True, "problems": []}
    assert (summary["scored_frames"], summary["scored_frames_without_prediction"]) == (26, 26)
    assert read_json(out / "predictions" / "SoccerNetGS-demo" / "pitchvlm" / f"{CLIP_ID}.json") == {"predictions": []}


# --- doctor: contract checks -----------------------------------------------------------


def test_contract_checks_pass_when_the_model_answers_the_truth(contract_scene: SyntheticScene) -> None:
    result = doctor.contract_checks(FakeBackend(contract_scene), contract_scene, StepSettings(), LIMITS)

    found = statuses(result.checks)
    assert "fail" not in found.values()
    assert [found[name] for name in (doctor.BOXES, doctor.LANDMARKS, doctor.FLEX, doctor.STORE)] == ["ok"] * 4
    # The accuracy values are facts without a verdict.
    assert [found[name] for name in (doctor.BOX_ACCURACY, doctor.LANDMARK_ACCURACY)] == ["info"] * 2
    # The fake backend has no SDK client below it, so there is no raw call to check.
    assert found[doctor.USAGE] == "info"
    assert found[doctor.SERVED_TIER] == "info"
    checks = by_name(result.checks)
    # The answers are the truth, rounded to model coordinates: 1.92 pixels in x and 1.08 pixels in y.
    # The contract verdicts compare the median values with the limits of the doctor module.
    assert checks[doctor.BOXES].value > 0.9
    assert checks[doctor.BOXES].limit == doctor.CONTRACT_MIN_MEDIAN_IOU == 0.3
    assert "44 of 44 possible pairs" in checks[doctor.BOXES].detail
    assert checks[doctor.LANDMARKS].value < 0.002
    assert checks[doctor.LANDMARKS].limit == doctor.CONTRACT_MAX_MEDIAN_POINT_ERROR == 0.05
    # The accuracy values compare the mean values with the values of the gate file.
    assert checks[doctor.BOX_ACCURACY].value > 0.9
    assert checks[doctor.LANDMARK_ACCURACY].value < 0.002
    assert (checks[doctor.BOX_ACCURACY].limit, checks[doctor.LANDMARK_ACCURACY].limit) == (0.6, 0.01)
    for name in (doctor.BOX_ACCURACY, doctor.LANDMARK_ACCURACY):
        assert "it reaches the accuracy value" in checks[name].detail
    assert checks[doctor.FLEX].value == 1.0
    # Three people requests, three landmarks requests and one people request through the flex tier.
    plan = doctor.contract_plan(StepSettings())
    assert [(step, settings.service_tier, count) for step, settings, count in plan] == [
        ("people", "standard", 3),
        ("landmarks", "standard", 3),
        ("people", "flex", 1),
    ]
    assert [(answer["step"], answer["frame"], answer["service_tier"]) for answer in result.answers] == [
        ("people", 0, "standard"),
        ("people", 75, "standard"),
        ("people", 150, "standard"),
        ("landmarks", 0, "standard"),
        ("landmarks", 75, "standard"),
        ("landmarks", 150, "standard"),
        ("people", 0, "flex"),
    ]


def test_contract_checks_fail_when_the_answers_are_far_from_the_truth(contract_scene: SyntheticScene) -> None:
    """Noise of 200 pixels on every box edge and every landmark coordinate, in a picture of 1920 x 1080 pixels.

    Answers that far off cannot be told apart from answers in a wrong order, so the contract verdicts fail.
    """
    backend = FakeBackend(contract_scene, box_noise_px=200.0, point_noise_px=200.0, seed=5)

    result = doctor.contract_checks(backend, contract_scene, StepSettings(), LIMITS)

    found = statuses(result.checks)
    assert found[doctor.BOXES] == "fail"
    assert found[doctor.LANDMARKS] == "fail"
    assert doctor.any_failed(result.checks)
    # The answers are still valid JSON, and the requests still ask for no storage.
    assert found[doctor.FLEX] == "ok"
    assert found[doctor.STORE] == "ok"
    assert by_name(result.checks)[doctor.LANDMARKS].value > 0.1


def test_contract_checks_fail_when_x_and_y_are_swapped(contract_scene: SyntheticScene) -> None:
    """The order convention: a model box is [ymin, xmin, ymax, xmax] and a model point is [y, x].

    An answer that writes x first puts a tall box of 38 x 97 pixels on its side, at another place.
    Such a box overlaps a true box with an IoU of 0.13 at most, and in one frame no box overlaps
    a true box at all. The median landmark is 0.39 of the image side away from its place.
    """
    backend = SwappedAxesBackend(FakeBackend(contract_scene))

    result = doctor.contract_checks(backend, contract_scene, StepSettings(), LIMITS)

    checks = by_name(result.checks)
    assert checks[doctor.BOXES].status == "fail"
    assert "none of the 17 answered boxes overlaps" in checks[doctor.BOXES].detail
    # A boxes check that stops at a frame without a pair has no accuracy value.
    assert doctor.BOX_ACCURACY not in checks
    assert checks[doctor.LANDMARKS].status == "fail"
    assert checks[doctor.LANDMARKS].value == pytest.approx(0.385, abs=0.01)
    assert doctor.any_failed(result.checks)


def test_contract_verdict_passes_for_a_model_that_is_some_pixels_off(contract_scene: SyntheticScene) -> None:
    """Noise of 6 pixels on every box edge and of 20 pixels on every landmark coordinate, in the right order.

    The synthetic people are 15 to 34 pixels wide, so this noise takes the mean IoU below the
    accuracy value of the gate file. The client is right, so no check fails.
    """
    backend = FakeBackend(contract_scene, box_noise_px=6.0, point_noise_px=20.0, seed=1)

    result = doctor.contract_checks(backend, contract_scene, StepSettings(), LIMITS)

    assert not doctor.any_failed(result.checks)
    checks = by_name(result.checks)
    assert checks[doctor.BOXES].status == "ok"
    assert checks[doctor.BOXES].value == pytest.approx(0.568, abs=0.001)
    assert "43 of 44 possible pairs" in checks[doctor.BOXES].detail
    assert checks[doctor.LANDMARKS].status == "ok"
    assert checks[doctor.LANDMARKS].value == pytest.approx(0.0138, abs=0.0001)
    # Both mean values miss the accuracy value of the gate file. They are facts, not verdicts.
    box_accuracy, landmark_accuracy = checks[doctor.BOX_ACCURACY], checks[doctor.LANDMARK_ACCURACY]
    assert (box_accuracy.status, landmark_accuracy.status) == ("info", "info")
    assert box_accuracy.value == pytest.approx(0.529, abs=0.001) and box_accuracy.limit == 0.6
    assert landmark_accuracy.value == pytest.approx(0.0157, abs=0.0001) and landmark_accuracy.limit == 0.01
    for check in (box_accuracy, landmark_accuracy):
        assert "it does not reach the accuracy value" in check.detail
        assert "This is no verdict" in check.detail
    assert "43 of 44 true boxes have a pair" in box_accuracy.detail


def test_contract_verdict_passes_when_the_model_confuses_two_landmarks(contract_scene: SyntheticScene) -> None:
    """Every point is exact, but the first and the last landmark of frame 0 have each other's id.

    Two points far from their place take the mean error to 0.04 of the image side. The median stays exact.
    """

    def swap_two_ids(request: VLMRequest, answer: dict[str, Any]) -> None:
        if request.meta.step == "landmarks" and request.meta.frame == 0:
            first, last = answer["landmarks"][0], answer["landmarks"][-1]
            first["id"], last["id"] = last["id"], first["id"]

    backend = EditedAnswersBackend(FakeBackend(contract_scene), swap_two_ids)

    result = doctor.contract_checks(backend, contract_scene, StepSettings(), LIMITS)

    assert not doctor.any_failed(result.checks)
    checks = by_name(result.checks)
    assert checks[doctor.LANDMARKS].status == "ok"
    assert checks[doctor.LANDMARKS].value < 0.002
    accuracy = checks[doctor.LANDMARK_ACCURACY]
    assert accuracy.status == "info"
    assert accuracy.value == pytest.approx(0.0402, abs=0.0001)
    assert "it does not reach the accuracy value" in accuracy.detail
    assert "2 landmarks are further than 0.05 from their true point" in accuracy.detail


def test_contract_checks_fail_when_the_model_sees_no_person(contract_scene: SyntheticScene) -> None:
    backend = FakeBackend(contract_scene, miss_rate=1.0)

    checks = by_name(doctor.contract_checks(backend, contract_scene, StepSettings(), LIMITS).checks)

    assert checks[doctor.BOXES].status == "fail"
    assert "frame 0" in checks[doctor.BOXES].detail
    assert checks[doctor.LANDMARKS].status == "ok"


def test_boxes_verdict_does_not_count_people_that_the_model_misses(contract_scene: SyntheticScene) -> None:
    """A model that finds 4 of 10 people places these boxes right: this is no fault of the client."""
    backend = FakeBackend(contract_scene, miss_rate=0.6, seed=3)

    checks = by_name(doctor.contract_checks(backend, contract_scene, StepSettings(), LIMITS).checks)

    boxes = checks[doctor.BOXES]
    assert boxes.status == "ok"
    # Every answered box has a pair. A frame can give as many pairs as it has answered boxes.
    n_pairs, n_possible = (int(count) for count in re.match(r"(\d+) of (\d+) possible pairs", boxes.detail).groups())
    assert n_pairs == n_possible
    assert 8 <= n_pairs <= 30
    assert f"{n_pairs} of 44 true boxes have a pair" in checks[doctor.BOX_ACCURACY].detail


def test_boxes_verdict_does_not_count_boxes_where_nobody_is(contract_scene: SyntheticScene) -> None:
    """Every person has an exact box, and each answer has 20 more boxes in the top left corner, where nobody is.

    A model that sees people who are not there places the boxes of the real people right.
    """

    def add_twenty(request: VLMRequest, answer: dict[str, Any]) -> None:
        if request.meta.step == "people":
            answer["people"] += [
                {"box_2d": [10, 10 + 30 * index, 40, 25 + 30 * index], "role": "player", "team": "A", "number": None}
                for index in range(20)
            ]

    backend = EditedAnswersBackend(FakeBackend(contract_scene), add_twenty)

    boxes = by_name(doctor.contract_checks(backend, contract_scene, StepSettings(), LIMITS).checks)[doctor.BOXES]

    assert boxes.status == "ok"
    # A frame can give as many pairs as it has true boxes: 44 in the three frames.
    assert "44 of 44 possible pairs" in boxes.detail


def test_boxes_verdict_fails_when_most_answered_boxes_are_at_a_wrong_place(contract_scene: SyntheticScene) -> None:
    """Two boxes of every frame are right, and ten more boxes lie in the top left corner, where nobody is.

    The two right boxes alone have a perfect IoU. The verdict must see the ten other boxes too.
    """

    def keep_two_and_add_ten(request: VLMRequest, answer: dict[str, Any]) -> None:
        if request.meta.step == "people":
            wrong = [
                {"box_2d": [10, 10 + 30 * index, 40, 25 + 30 * index], "role": "player", "team": "A", "number": None}
                for index in range(10)
            ]
            answer["people"] = [*answer["people"][:2], *wrong]

    backend = EditedAnswersBackend(FakeBackend(contract_scene), keep_two_and_add_ten)

    checks = by_name(doctor.contract_checks(backend, contract_scene, StepSettings(), LIMITS).checks)

    boxes = checks[doctor.BOXES]
    assert boxes.status == "fail"
    # Three frames with 12 answered boxes each: 36 possible pairs, 6 of them exist.
    assert "6 of 36 possible pairs" in boxes.detail
    assert boxes.value > 0.9


def test_boxes_verdict_fails_when_the_boxes_are_most_of_their_width_off(contract_scene: SyntheticScene) -> None:
    """Every box is moved to the right by 70% of its width: the overlap is 0.3 / 1.7 = 0.18 of the union."""

    def move_right(request: VLMRequest, answer: dict[str, Any]) -> None:
        for person in answer.get("people", []):
            ymin, xmin, ymax, xmax = person["box_2d"]  # A model box is [ymin, xmin, ymax, xmax].
            shift = round(0.7 * (xmax - xmin))
            person["box_2d"] = [ymin, xmin + shift, ymax, xmax + shift]

    backend = EditedAnswersBackend(FakeBackend(contract_scene), move_right)

    checks = by_name(doctor.contract_checks(backend, contract_scene, StepSettings(), LIMITS).checks)

    boxes = checks[doctor.BOXES]
    assert boxes.status == "fail"
    # Most boxes still have a pair, so only the median IoU is below its limit.
    n_pairs, n_possible = (int(count) for count in re.match(r"(\d+) of (\d+) possible pairs", boxes.detail).groups())
    assert n_pairs >= 0.9 * n_possible
    assert 0.1 < boxes.value < doctor.CONTRACT_MIN_MEDIAN_IOU


def test_contract_and_accuracy_values_that_are_exactly_at_their_limits_pass(
    contract_scene: SyntheticScene, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend(contract_scene)
    measured = by_name(doctor.contract_checks(backend, contract_scene, StepSettings(), LIMITS).checks)
    monkeypatch.setattr(doctor, "CONTRACT_MIN_PAIRED_SHARE", 1.0)
    monkeypatch.setattr(doctor, "CONTRACT_MIN_MEDIAN_IOU", measured[doctor.BOXES].value)
    monkeypatch.setattr(doctor, "CONTRACT_MAX_MEDIAN_POINT_ERROR", measured[doctor.LANDMARKS].value)
    at_the_limit = dataclasses.replace(
        LIMITS,
        box_mean_iou_min=measured[doctor.BOX_ACCURACY].value,
        point_error_max_fraction=measured[doctor.LANDMARK_ACCURACY].value,
        flex_valid_answer_rate_min=1.0,
    )

    checks = by_name(doctor.contract_checks(backend, contract_scene, StepSettings(), at_the_limit).checks)

    assert [checks[name].status for name in (doctor.BOXES, doctor.LANDMARKS, doctor.FLEX)] == ["ok", "ok", "ok"]
    for name in (doctor.BOX_ACCURACY, doctor.LANDMARK_ACCURACY):
        assert "it reaches the accuracy value" in checks[name].detail


def test_contract_checks_use_the_limits_that_they_are_given(contract_scene: SyntheticScene) -> None:
    """Limits that no answer can reach: the accuracy values and the flex check must show them."""
    out_of_reach = dataclasses.replace(
        LIMITS, box_mean_iou_min=1.01, point_error_max_fraction=-0.01, flex_valid_answer_rate_min=1.01
    )

    backend = FakeBackend(contract_scene)
    checks = by_name(doctor.contract_checks(backend, contract_scene, StepSettings(), out_of_reach).checks)

    assert (checks[doctor.FLEX].status, checks[doctor.FLEX].limit) == ("fail", 1.01)
    assert checks[doctor.BOX_ACCURACY].limit == 1.01
    assert checks[doctor.LANDMARK_ACCURACY].limit == -0.01
    for name in (doctor.BOX_ACCURACY, doctor.LANDMARK_ACCURACY):
        assert "it does not reach the accuracy value" in checks[name].detail
    # The accuracy values of the gate file do not change the contract verdicts.
    assert [checks[name].status for name in (doctor.BOXES, doctor.LANDMARKS)] == ["ok", "ok"]


def test_point_error_in_x_is_a_share_of_the_image_width(contract_scene: SyntheticScene) -> None:
    """Every landmark is 19 model units (36.5 pixels of 1920) too far to the right: 0.019 of the width.

    As a share of the height (1080 pixels) the same pixels would be 0.034.
    """

    def move_right(request: VLMRequest, answer: dict[str, Any]) -> None:
        if request.meta.step == "landmarks":
            answer["landmarks"] = [landmark for landmark in answer["landmarks"] if landmark["point"][1] <= 981]
            for landmark in answer["landmarks"]:
                landmark["point"][1] += 19  # A model point is [y, x].

    backend = EditedAnswersBackend(FakeBackend(contract_scene), move_right)

    checks = by_name(doctor.contract_checks(backend, contract_scene, StepSettings(), LIMITS).checks)

    assert checks[doctor.LANDMARKS].value == pytest.approx(0.019, abs=0.002)
    assert checks[doctor.LANDMARK_ACCURACY].value == pytest.approx(0.019, abs=0.002)
    assert checks[doctor.LANDMARKS].status == "ok"
    assert "it does not reach the accuracy value" in checks[doctor.LANDMARK_ACCURACY].detail


def test_contract_checks_send_no_flex_request_for_a_model_without_a_flex_tier(contract_scene: SyntheticScene) -> None:
    """The provider documents no flex tier for the robotics model, so a flex request for it would fail."""
    settings = StepSettings(model="gemini-robotics-er-2-preview")
    backend = FlexRefusingBackend(FakeBackend(contract_scene))

    result = doctor.contract_checks(backend, contract_scene, settings, LIMITS)

    assert not doctor.any_failed(result.checks)
    flex = by_name(result.checks)[doctor.FLEX]
    assert flex.status == "info"
    assert "gemini-robotics-er-2-preview has no flex tier" in flex.detail
    assert backend.tiers == ["standard"] * 6
    assert [(step, planned.service_tier, count) for step, planned, count in doctor.contract_plan(settings)] == [
        ("people", "standard", 3),
        ("landmarks", "standard", 3),
    ]
    assert not has_flex_tier("gemini-robotics-er-2-preview")
    for model in ("gemini-4-argon", "gemini-4-carbon", "gemini-4-hydrogen", "gemini-3.8-flash"):
        assert has_flex_tier(model)


def test_flex_check_fails_when_the_flex_tier_refuses_a_model_that_has_one(contract_scene: SyntheticScene) -> None:
    backend = FlexRefusingBackend(FakeBackend(contract_scene))

    result = doctor.contract_checks(backend, contract_scene, StepSettings(), LIMITS)

    flex = by_name(result.checks)[doctor.FLEX]
    assert (flex.status, flex.value) == ("fail", 0.0)
    assert "the flex tier is not there" in flex.detail
    assert backend.tiers == [*["standard"] * 6, "flex"]


def test_flex_check_gives_no_verdict_when_the_flex_tier_has_no_capacity(contract_scene: SyntheticScene) -> None:
    """A busy flex tier answers 503 or 429. That is the documented behaviour of the tier, not a fault of the client."""
    backend = FlexBusyBackend(FakeBackend(contract_scene), busy_frames=[0])

    result = doctor.contract_checks(backend, contract_scene, StepSettings(), LIMITS)

    flex = by_name(result.checks)[doctor.FLEX]
    assert (flex.status, flex.value, flex.limit) == ("info", None, None)
    assert "not checked: the flex tier had no capacity for any of the 1 requests" in flex.detail
    assert "no fault of the client" in flex.detail
    assert "Run the doctor again before the first run through the flex tier" in flex.detail
    assert "the flex tier had no capacity for frame 0" in flex.detail
    # The other checks pass, so the contract checks of this model do not fail.
    assert not doctor.any_failed(result.checks)


def test_flex_check_does_not_count_a_request_without_capacity(contract_scene: SyntheticScene) -> None:
    """The share of valid answers is taken over the requests for which the flex tier had capacity."""
    truth = FakeBackend(contract_scene)

    answered = doctor.check_flex(
        FlexBusyBackend(truth, busy_frames=[0]), contract_scene, StepSettings(), [0, 75], min_valid_rate=0.95
    )
    assert (answered.status, answered.value) == ("ok", 1.0)
    assert "1 of 1 requests through the flex tier gave a valid answer" in answered.detail
    assert "1 more requests are not counted, because the flex tier had no capacity for them" in answered.detail

    # A request that fails for another reason still fails the check: here the answer of frame 75 is no JSON.
    not_json = FlexBusyBackend(NotJsonBackend(truth), busy_frames=[0])
    broken = doctor.check_flex(not_json, contract_scene, StepSettings(), [0, 75], min_valid_rate=0.95)
    assert (broken.status, broken.value) == ("fail", 0.0)
    assert "0 of 1 requests through the flex tier gave a valid answer" in broken.detail


def test_flex_check_names_the_time_of_each_answer(contract_scene: SyntheticScene) -> None:
    """A flex request can wait in a queue. The time of each answer shows a slow tier also when the check passes."""

    class SlowBackend:
        """A backend that answers like the fake backend, after 12.34 seconds on its clock."""

        def generate(self, request: VLMRequest) -> VLMResponse:
            return FakeBackend(contract_scene).generate(request).model_copy(update={"latency_s": 12.34})

    answered = doctor.check_flex(SlowBackend(), contract_scene, StepSettings(), [0, 75], min_valid_rate=0.95)
    refused = doctor.check_flex(
        FlexRefusingBackend(FakeBackend(contract_scene)), contract_scene, StepSettings(), [0], min_valid_rate=0.95
    )

    assert answered.status == "ok"
    assert "2 of 2 requests through the flex tier gave a valid answer" in answered.detail
    assert "time of each answer: 12.3 s, 12.3 s" in answered.detail
    assert refused.status == "fail"
    assert "time of each answer" not in refused.detail


def test_raw_calls_of_an_earlier_model_are_not_checked_again(contract_scene: SyntheticScene) -> None:
    """One recorder serves all models of a doctor run. The checks of a model use only its own calls."""
    # The recorder already holds the call of an earlier model, which was sent without store=False.
    recorder = SimpleNamespace(calls=[raw_call(store=True)])
    backend = RawCallBackend(FakeBackend(contract_scene), recorder, raw_call())

    result = doctor.contract_checks(backend, contract_scene, StepSettings(), LIMITS, recorder=recorder)

    store = by_name(result.checks)[doctor.STORE]
    assert store.status == "ok"
    assert "7 recorded calls" in store.detail


def test_contract_checks_use_the_usage_tolerance_that_they_are_given(contract_scene: SyntheticScene) -> None:
    recorder = SimpleNamespace(calls=[])
    # The total is 2000 tokens, and the billed tokens add up to 1900: a gap of 5%.
    backend = RawCallBackend(FakeBackend(contract_scene), recorder, raw_call(total_tokens=2000))

    strict = by_name(doctor.contract_checks(backend, contract_scene, StepSettings(), LIMITS, recorder=recorder).checks)
    loose_limits = dataclasses.replace(LIMITS, usage_sum_tolerance=0.05)
    loose = by_name(
        doctor.contract_checks(backend, contract_scene, StepSettings(), loose_limits, recorder=recorder).checks
    )

    assert (strict[doctor.USAGE].status, strict[doctor.USAGE].limit) == ("fail", 0.01)
    assert strict[doctor.USAGE].value == pytest.approx(0.05)
    # A gap that is exactly the tolerance passes.
    assert (loose[doctor.USAGE].status, loose[doctor.USAGE].limit) == ("ok", 0.05)


def test_contract_checks_need_a_frame(contract_scene: SyntheticScene) -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        doctor.contract_checks(FakeBackend(contract_scene), contract_scene, StepSettings(), LIMITS, frames=())


# --- doctor: raw calls -----------------------------------------------------------------


def test_usage_sums_accept_both_ways_to_count_thinking_tokens() -> None:
    separate = raw_call()
    # Here the output count holds the thinking tokens: 1500 + 400 = 1900.
    included = raw_call(output_tokens=400)

    result = doctor.check_usage_sums([separate, included], tolerance=0.01)

    assert result.status == "ok"
    assert result.value == 0.0
    assert "1 included" in result.detail
    assert "1 separate" in result.detail


@pytest.mark.parametrize("total_tokens", [2500, None, 0])
def test_usage_sums_fail_when_the_total_does_not_add_up_or_is_missing(total_tokens: int | None) -> None:
    result = doctor.check_usage_sums([raw_call(), raw_call(total_tokens=total_tokens)], tolerance=0.01)

    assert result.status == "fail"


def test_usage_sums_without_a_raw_call_give_no_verdict() -> None:
    assert doctor.check_usage_sums([], tolerance=0.01).status == "info"


def test_store_flag_fails_for_a_call_that_was_sent_without_store_false() -> None:
    assert doctor.check_store_flag([], [raw_call()]).status == "ok"
    assert doctor.check_store_flag([], [raw_call(), raw_call(store=True)]).status == "fail"
    assert doctor.check_store_flag([], [raw_call(store=None)]).status == "fail"


@pytest.mark.parametrize(
    ("calls", "status"),
    [
        ([raw_call(sent_tier="flex", served_tier="flex")], "ok"),
        ([raw_call(sent_tier="flex", served_tier="FLEX")], "ok"),
        ([raw_call(sent_tier="standard", served_tier="standard")], "ok"),
        ([raw_call(sent_tier="flex", served_tier="standard")], "fail"),
        # A standard call that a more expensive tier served costs more than the request asked for.
        ([raw_call(sent_tier="standard", served_tier="priority")], "fail"),
        ([raw_call(served_tier="standard"), raw_call(sent_tier="flex", served_tier="standard")], "fail"),
        ([raw_call(sent_tier="flex", served_tier=None)], "info"),
        ([raw_call(sent_tier=None, served_tier="standard")], "info"),
        ([], "info"),
    ],
)
def test_served_tier_is_checked_for_every_recorded_call(calls: list[doctor.RawCall], status: str) -> None:
    assert doctor.check_served_tiers(calls).status == status


def test_served_tier_check_names_every_pair_of_tiers_that_differ() -> None:
    calls = [
        raw_call(served_tier="standard"),
        raw_call(served_tier="priority"),
        raw_call(served_tier="Priority"),
        raw_call(sent_tier="flex", served_tier="standard"),
        raw_call(sent_tier="flex", served_tier=None),
    ]

    check = doctor.check_served_tiers(calls)

    assert check.status == "fail"
    assert "3 of 4 answers report another tier than their call asked for" in check.detail
    assert "asked for flex, served by standard (1 calls)" in check.detail
    assert "asked for standard, served by priority (2 calls)" in check.detail


def test_served_tier_check_counts_the_calls_of_each_tier() -> None:
    calls = [*[raw_call(served_tier="standard")] * 6, raw_call(sent_tier="flex", served_tier="flex"), raw_call()]

    check = doctor.check_served_tiers(calls)

    assert check.status == "ok"
    assert "7 of 8 answers report the tier that their call asked for (1 flex, 6 standard)" in check.detail
    assert "none of the 1 answers reports the tier" in doctor.check_served_tiers([raw_call()]).detail


def test_store_flag_fails_for_a_request_that_gives_a_call_without_store_false(
    contract_scene: SyntheticScene, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without recorded calls the check still builds the call of every request and reads its store argument."""
    request = steps.build_people_request(
        contract_scene.image(0), contract_scene.kit_legend(), clip_id=CLIP_ID, frame=0, settings=StepSettings()
    )
    assert doctor.check_store_flag([request], []).status == "ok"

    monkeypatch.setattr(doctor, "build_interaction_kwargs", lambda request: {"store": True})

    check = doctor.check_store_flag([request], [])
    assert check.status == "fail"
    assert "1 of 1 requests give a call without store=False" in check.detail


def test_spend_check_compares_the_cost_with_its_limit() -> None:
    assert doctor.check_spend(0.4, 2.0).status == "ok"
    assert doctor.check_spend(2.5, 2.0).status == "fail"
    # A cost that is exactly the limit passes.
    assert doctor.check_spend(2.0, 2.0).status == "ok"


def test_recording_client_keeps_the_raw_facts_of_a_call() -> None:
    usage = SimpleNamespace(
        total_input_tokens=1500, total_output_tokens=300, total_thought_tokens=100, total_tokens=1900
    )
    interaction = SimpleNamespace(status="completed", service_tier="flex", usage=usage)
    seen: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> SimpleNamespace:
        seen.append(kwargs)
        return interaction

    recorder = doctor.RecordingClient(SimpleNamespace(interactions=SimpleNamespace(create=create)))

    answer = recorder.interactions.create(model="m", store=False, service_tier="flex", timeout=5.0)

    assert answer is interaction
    assert seen == [{"model": "m", "store": False, "service_tier": "flex", "timeout": 5.0}]
    assert recorder.calls == [raw_call(sent_tier="flex", served_tier="flex")]


def test_recording_client_reads_missing_counters_as_none() -> None:
    # A truth value is not a token count, and the other counters are missing.
    interaction = SimpleNamespace(status="completed", usage=SimpleNamespace(total_input_tokens=7, total_tokens=True))
    sdk_client = SimpleNamespace(interactions=SimpleNamespace(create=lambda **kwargs: interaction))
    recorder = doctor.RecordingClient(sdk_client)

    recorder.interactions.create(model="m")

    assert recorder.calls == [
        doctor.RawCall(
            store=None,
            sent_tier=None,
            status="completed",
            served_tier=None,
            input_tokens=7,
            output_tokens=None,
            thought_tokens=None,
            total_tokens=None,
        )
    ]


def test_recording_client_sees_what_the_real_backend_sends(contract_scene: SyntheticScene) -> None:
    """The real backend on top of the recording client, with a stand-in for the SDK client below."""
    usage = SimpleNamespace(
        total_input_tokens=1500,
        total_output_tokens=300,
        total_thought_tokens=100,
        total_cached_tokens=0,
        total_tokens=1900,
    )
    interaction = SimpleNamespace(status="completed", output_text='{"people": []}', usage=usage, service_tier=None)
    sdk_client = SimpleNamespace(interactions=SimpleNamespace(create=lambda **kwargs: interaction))
    recorder = doctor.RecordingClient(sdk_client)
    request = steps.build_people_request(
        contract_scene.image(0),
        contract_scene.kit_legend(),
        clip_id=CLIP_ID,
        frame=0,
        settings=doctor.flex_settings(StepSettings()),
    )

    response = GeminiBackend(client=recorder).generate(request)

    assert response.text == '{"people": []}'
    assert recorder.calls == [raw_call(sent_tier="flex")]
    assert doctor.check_store_flag([request], recorder.calls).status == "ok"
    assert doctor.check_usage_sums(recorder.calls, tolerance=0.01).status == "ok"


# --- doctor: environment and command ---------------------------------------------------


def test_environment_checks_fail_without_the_guard_hook(
    private_git: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(cli.KEY_VARIABLE, raising=False)
    repo = tmp_path / "repo"
    make_repository(repo)

    before = by_name(
        doctor.environment_checks(repo, key_variable=cli.KEY_VARIABLE, data_dir=tmp_path / "none", api=False)
    )
    install_hooks(repo)
    after = by_name(
        doctor.environment_checks(repo, key_variable=cli.KEY_VARIABLE, data_dir=tmp_path / "none", api=False)
    )

    assert before[doctor.HOOK].status == "fail"
    assert ".git/hooks/pre-commit is missing; .git/hooks/pre-push is missing." in before[doctor.HOOK].detail
    assert after[doctor.HOOK].status == "ok"
    assert ".git/hooks/pre-commit and .git/hooks/pre-push" in after[doctor.HOOK].detail
    # Without model calls, a missing key is a fact and not a failure.
    assert after[doctor.KEY].status == "info"
    assert set(after) == {
        doctor.PYTHON,
        doctor.PACKAGES,
        doctor.SDK,
        doctor.KEY,
        doctor.HOOK,
        doctor.DENY,
        doctor.DISK,
    }
    assert "numpy" in after[doctor.PACKAGES].detail
    assert "GB" in after[doctor.DISK].detail


@pytest.mark.parametrize(("installed", "missing"), [("pre-commit", "pre-push"), ("pre-push", "pre-commit")])
def test_environment_checks_need_the_hook_before_a_commit_and_the_hook_before_a_push(
    private_git: None, tmp_path: Path, installed: str, missing: str
) -> None:
    """The guard must run before every commit and before every push."""
    repo = tmp_path / "repo"
    install_hooks(repo, [installed])

    check = hook_check(repo)

    assert check.status == "fail"
    # The detail names the hook that is missing, and the command that installs both.
    assert f".git/hooks/{missing} is missing." in check.detail
    assert f".git/hooks/{installed}" not in check.detail
    assert "pre-commit install --hook-type pre-commit --hook-type pre-push" in check.detail


def test_hook_file_that_git_does_not_run_is_not_a_guard_hook(private_git: None, tmp_path: Path) -> None:
    """Git ignores a hook file without the executable bit, so a commit goes through without the guard."""
    repo = tmp_path / "repo"
    install_hooks(repo)
    write_hook(repo / ".git" / "hooks", "pre-commit", framework_hook("pre-commit"), executable=False)

    check = hook_check(repo)

    assert check.status == "fail"
    assert ".git/hooks/pre-commit is not executable, so git does not run it" in check.detail
    assert "pre-push" not in check.detail.split(". Install both hooks")[0]


@pytest.mark.parametrize(
    "text",
    [
        # An empty file, the hook of another tool, and the hooks of the framework for another hook or another list.
        "",
        "#!/bin/sh\necho 'other tool: scan done'\n",
        framework_hook("pre-push"),
        framework_hook("pre-commit").replace(".pre-commit-config.yaml", "other-hooks.yaml"),
    ],
)
def test_hook_of_another_tool_is_not_a_guard_hook(private_git: None, tmp_path: Path, text: str) -> None:
    """A file with the name of the hook is not enough: it must be the hook that runs the guard."""
    repo = tmp_path / "repo"
    install_hooks(repo)
    write_hook(repo / ".git" / "hooks", "pre-commit", text)

    check = hook_check(repo)

    assert check.status == "fail"
    assert ".git/hooks/pre-commit is not the pre-commit hook that the pre-commit framework writes" in check.detail


def test_hook_check_follows_the_hook_folder_that_git_uses(private_git: None, tmp_path: Path) -> None:
    """With the setting core.hooksPath, git never reads the folder .git/hooks."""
    repo, elsewhere = tmp_path / "repo", tmp_path / "shared-hooks"
    install_hooks(repo)
    elsewhere.mkdir()
    git(repo, "config", "core.hooksPath", str(elsewhere))

    not_there = hook_check(repo)

    assert not_there.status == "fail"
    # The other folder lies outside the repository and outside the home folder: only the file names are shown.
    assert not_there.detail.startswith("pre-commit is missing; pre-push is missing.")
    assert ".git/hooks" not in not_there.detail
    # The result can go to the results branch: it holds no absolute path of this machine.
    assert str(tmp_path) not in not_there.detail

    for name in HOOK_NAMES:
        write_hook(elsewhere, name, framework_hook(name))

    assert hook_check(repo).status == "ok"


def test_hook_check_needs_a_git_work_tree(private_git: None, tmp_path: Path) -> None:
    """A folder with hook files that is no git work tree: git runs no hook there."""
    folder = tmp_path / "plain"
    folder.mkdir()
    shutil.copy(HOOK_FILE, folder / ".pre-commit-config.yaml")
    for name in HOOK_NAMES:
        write_hook(folder / ".git" / "hooks", name, framework_hook(name))

    check = hook_check(folder)

    assert check.status == "fail"
    assert "git does not name the place of the hook pre-commit" in check.detail
    # The deny-list of the guard lies in the git folder, so it cannot be found either.
    write_deny_list(folder)
    deny = environment_check(folder, doctor.DENY)
    assert deny.status == "fail"
    assert "git does not name its folder" in deny.detail


def test_deny_list_check_fails_without_the_list_and_counts_its_entries(private_git: None, tmp_path: Path) -> None:
    """Without the deny-list the guard knows no private word of the machine, and it says so only in a notice."""
    repo = tmp_path / "repo"
    install_hooks(repo)

    missing = environment_check(repo, doctor.DENY)
    private_word = "word-" + "that-must-stay-private"
    write_deny_list(repo, f"# The private words of this machine.\n{private_word}\n\n  another-word  \n")
    filled = environment_check(repo, doctor.DENY)
    write_deny_list(repo, "# This machine has no private word.\n")
    without_entries = environment_check(repo, doctor.DENY)

    assert missing.status == "fail"
    assert ".git/guard-deny.txt is missing" in missing.detail
    assert "docs/RUNBOOK.md, section 9" in missing.detail
    # The detail counts the entries. It never shows one.
    assert (filled.status, filled.detail) == ("ok", ".git/guard-deny.txt has 2 entries")
    assert (without_entries.status, without_entries.detail) == ("ok", ".git/guard-deny.txt has 0 entries")


def test_deny_list_check_reads_the_list_that_all_work_trees_share(private_git: None, tmp_path: Path) -> None:
    """The deny-list lies in the git folder of the clone. A linked work tree has no git folder of its own."""
    repo = tmp_path / "repo"
    make_repository(repo)
    write_deny_list(repo, "one-word\n")
    author = ["-c", "user.name=Tester", "-c", "user.email=tester" + "@" + "example.org", "-c", "commit.gpgsign=false"]
    git(repo, *author, "commit", "--quiet", "--allow-empty", "-m", "first commit")
    linked = tmp_path / "linked"
    git(repo, "worktree", "add", "--quiet", "-b", "results", str(linked))

    check = environment_check(linked, doctor.DENY)

    assert check.status == "ok"
    assert check.detail.endswith("guard-deny.txt has 1 entries")
    assert str(tmp_path) not in check.detail


@pytest.mark.parametrize(
    ("version", "with_api", "without_api"),
    [("3.0.0", "ok", "ok"), ("3.3.0", "ok", "ok"), ("2.34.0", "fail", "info"), ("4.0.0", "fail", "info")],
)
def test_sdk_check_needs_the_major_version_that_the_client_is_written_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str, with_api: str, without_api: str
) -> None:
    """Before version 3.0 the SDK repeats a failed call by itself, although the client asks for one attempt."""
    monkeypatch.setattr(doctor, "installed_sdk_version", lambda: version)

    paid = environment_check(tmp_path, doctor.SDK, api=True)
    free = environment_check(tmp_path, doctor.SDK, api=False)

    assert (paid.status, free.status) == (with_api, without_api)
    assert f"google-genai {version}" in paid.detail
    if with_api == "fail":
        assert "is not a version 3.x: the client needs 3.0 or later and below 4" in paid.detail


def test_sdk_check_without_the_sdk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "installed_sdk_version", lambda: None)

    assert environment_check(tmp_path, doctor.SDK, api=True).status == "fail"
    assert environment_check(tmp_path, doctor.SDK, api=False).status == "info"
    assert "google-genai is not installed" in environment_check(tmp_path, doctor.SDK, api=False).detail


def test_sdk_version_of_the_doctor_is_the_version_range_of_the_package() -> None:
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    major = doctor.SDK_MAJOR_VERSION
    assert project["optional-dependencies"]["gemini"] == [f"google-genai>={major}.0,<{major + 1}"]


def test_hook_check_needs_a_hook_list_with_the_guard(private_git: None, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    install_hooks(repo)
    hook_list = repo / ".pre-commit-config.yaml"
    config = yaml.safe_load(HOOK_FILE.read_text(encoding="utf-8"))
    assert hook_check(repo).status == "ok"

    # Without the guard before a push.
    kept = [hook for hook in config["repos"][0]["hooks"] if hook["id"] != "guard-push"]
    hook_list.write_text(yaml.safe_dump({**config, "repos": [{"repo": "local", "hooks": kept}]}), encoding="utf-8")
    check = hook_check(repo)
    assert check.status == "fail"
    assert ".pre-commit-config.yaml names no hook that runs tools/guard.py at the stage pre-push" in check.detail
    assert "tools/guard.py at the stage pre-commit" not in check.detail

    hook_list.write_text("- not\n- a hook list\n", encoding="utf-8")
    assert "is not a hook list of the pre-commit framework" in hook_check(repo).detail

    hook_list.unlink()
    check = hook_check(repo)
    assert check.status == "fail"
    assert "the hook list .pre-commit-config.yaml is missing" in check.detail


def test_hook_file_runs_the_guard_before_a_commit_and_before_a_push() -> None:
    config = yaml.safe_load(HOOK_FILE.read_text(encoding="utf-8"))
    hooks = hook_entries()

    assert sorted(config["default_install_hook_types"]) == sorted(HOOK_NAMES)
    assert len(hooks) == sum(len(repo["hooks"]) for repo in config["repos"]), "every hook has its own id"
    for stage in HOOK_NAMES:
        entries = [hook["entry"] for hook in hooks.values() if hook["stages"] == [stage]]
        assert sum("tools/guard.py" in entry for entry in entries) == 1, stage
    assert all(hook["pass_filenames"] is False for hook in hooks.values())


def test_environment_checks_never_show_the_value_of_the_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hidden = "value-" + "that-must-stay-hidden"
    monkeypatch.setenv(cli.KEY_VARIABLE, hidden)

    checks = doctor.environment_checks(tmp_path, key_variable=cli.KEY_VARIABLE, data_dir=tmp_path, api=True)

    assert by_name(checks)[doctor.KEY].status == "ok"
    assert cli.KEY_VARIABLE in by_name(checks)[doctor.KEY].detail
    assert all(hidden not in check.detail for check in checks)


def test_environment_checks_need_the_key_for_model_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(cli.KEY_VARIABLE, raising=False)

    checks = doctor.environment_checks(tmp_path, key_variable=cli.KEY_VARIABLE, data_dir=tmp_path, api=True)

    assert by_name(checks)[doctor.KEY].status == "fail"


def test_doctor_without_api_lists_the_machine(
    private_git: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    hidden = "value-" + "that-must-stay-hidden"
    monkeypatch.setenv(cli.KEY_VARIABLE, hidden)
    repo = tmp_path / "repo"
    make_repository(repo)
    monkeypatch.setattr(cli, "repository_root", lambda: repo)

    # The guard hooks are missing: the doctor fails. One of the two hooks is not enough.
    assert cli.main(["doctor", "--no-api"]) == 1
    install_hooks(repo, ["pre-commit"])
    assert cli.main(["doctor", "--no-api"]) == 1
    # With both hooks the deny-list of the guard is still missing.
    install_hooks(repo)
    assert cli.main(["doctor", "--no-api"]) == 1
    assert "[fail] deny-list: .git/guard-deny.txt is missing" in capsys.readouterr().out
    write_deny_list(repo)
    assert cli.main(["doctor", "--no-api"]) == 0

    output = capsys.readouterr().out
    for word in ("python", "numpy", "model SDK", cli.KEY_VARIABLE, "guard hook", "deny-list", "free disk space"):
        assert word in output
    assert hidden not in output


def test_doctor_report_needs_the_contract_checks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["doctor", "--no-api", "--out", str(tmp_path / "doctor.json")]) == 2

    assert "--no-api" in capsys.readouterr().err
    assert not (tmp_path / "doctor.json").exists()


def test_doctor_contract_checks_with_a_fake_model(
    private_git: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    contract_scene: SyntheticScene,
) -> None:
    """The model part of the doctor command, with the fake backend in the place of the real one."""
    repo = tmp_path / "repo"
    set_up_clone(repo)
    (repo / "configs").mkdir()
    shutil.copy(GATES, repo / "configs" / "gates.yaml")
    monkeypatch.setattr(cli, "repository_root", lambda: repo)
    monkeypatch.setenv(cli.KEY_VARIABLE, "not-a-real-value")
    monkeypatch.setattr(doctor, "installed_sdk_version", lambda: "3.0.test")
    recorder = doctor.RecordingClient(SimpleNamespace(interactions=None))
    monkeypatch.setattr(cli, "recording_backend", lambda max_usd_per_10_min: (FakeBackend(contract_scene), recorder))
    report_path = tmp_path / "metrics" / "doctor.json"
    cache = tmp_path / "cache"

    code = cli.main(["doctor", "--allow-dirty", "--cache", str(cache), "--out", str(report_path)])

    assert code == 0
    report = read_json(report_path)
    assert report["passed"] is True
    checks = {check["name"]: check["status"] for check in report["contract"]["gemini-4-argon"]["checks"]}
    assert checks[doctor.BOXES] == "ok"
    assert checks[doctor.LANDMARKS] == "ok"
    assert len(report["contract"]["gemini-4-argon"]["answers"]) == 7
    assert report["spend"]["status"] == "ok"
    provenance = report["provenance"]
    assert provenance["command"] == "doctor"
    assert isinstance(provenance["commit"], str) and provenance["commit"]
    assert provenance["dirty"] in (True, False)
    assert provenance["cost"]["new_answers"] == 7
    # The plan of the contract checks lists exactly these seven calls, and the estimate uses the token
    # counts of the fake model (``estimates_of_the_fake_model``). So the measured cost is the estimate.
    assert provenance["cost"]["measured_usd"] == pytest.approx(provenance["cost"]["estimate_usd"])
    # Every answer is in the cost ledger of the cache folder.
    assert len((cache / "ledger.jsonl").read_text(encoding="utf-8").splitlines()) == 7
    output = capsys.readouterr().out
    assert output.index("cost estimate") < output.index("contract checks of gate G0") < output.index("measured cost")
    # No absolute path of the test folder is in the report: it could go to the results branch.
    assert str(tmp_path) not in report_path.read_text(encoding="utf-8")


def test_doctor_records_the_cost_of_a_rejected_answer_one_time(
    private_git: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contract_scene: SyntheticScene
) -> None:
    """The contract stage has no cache: its metered backend is the one place that reports a billed answer."""
    repo = tmp_path / "repo"
    set_up_clone(repo)
    (repo / "configs").mkdir()
    shutil.copy(GATES, repo / "configs" / "gates.yaml")
    monkeypatch.setattr(cli, "repository_root", lambda: repo)
    monkeypatch.setenv(cli.KEY_VARIABLE, "not-a-real-value")
    monkeypatch.setattr(doctor, "installed_sdk_version", lambda: "3.0.test")
    recorder = doctor.RecordingClient(SimpleNamespace(interactions=None))
    backend = RejectedAnswerBackend(FakeBackend(contract_scene), "people")
    monkeypatch.setattr(cli, "recording_backend", lambda max_usd_per_10_min: (backend, recorder))
    report_path = tmp_path / "metrics" / "doctor.json"
    cache = tmp_path / "cache"

    code = cli.main(["doctor", "--allow-dirty", "--cache", str(cache), "--out", str(report_path)])

    # The rejected answer makes the boxes check fail. A failed check is not an error of the command.
    assert code == 1
    report = read_json(report_path)
    checks = {check["name"]: check for check in report["contract"]["gemini-4-argon"]["checks"]}
    assert checks[doctor.BOXES]["status"] == "fail"
    assert "the people answer has no text" in checks[doctor.BOXES]["detail"]
    assert checks[doctor.LANDMARKS]["status"] == "ok"
    assert checks[doctor.FLEX]["status"] == "ok"
    # The boxes check stops at its first request. The landmarks check sends three requests and the
    # flex check one. So five calls were billed, and four of them gave an answer.
    assert len(report["contract"]["gemini-4-argon"]["answers"]) == 4
    cost = report["provenance"]["cost"]
    assert cost["new_answers"] == 5
    assert cost["steps"]["people"]["answers"] == 2
    lines = read_ledger(cache)
    assert [line["step"] for line in lines] == ["people", "landmarks", "landmarks", "landmarks", "people"]
    assert cost["measured_usd"] == pytest.approx(sum(line["usd"] for line in lines))


def test_doctor_writes_its_report_also_after_a_budget_stop(
    private_git: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    contract_scene: SyntheticScene,
) -> None:
    """The spend limit of gate G0 ends the checks in the middle of the second model. The paid answers give a report.

    The estimate assumes half of the tokens that the answers have, so the checks may start: an
    estimate above the spend limit starts no check.
    """
    set_up_doctor(tmp_path, monkeypatch)
    recorder = doctor.RecordingClient(SimpleNamespace(interactions=None))
    backend = CountingBackend(FakeBackend(contract_scene))
    monkeypatch.setattr(cli, "recording_backend", lambda max_usd_per_10_min: (backend, recorder))
    estimates_of_a_share(monkeypatch, 0.5)
    first_model = 3.5 * fake_cost("people", "gemini-3.8-flash") + 3 * fake_cost("landmarks", "gemini-3.8-flash")
    second_model = 3.5 * fake_cost("people") + 3 * fake_cost("landmarks")
    spend_limit = first_model + 0.5 * second_model
    gates = gates_with(g0_contract={"spend_usd_max": spend_limit})
    monkeypatch.setattr(cli, "load_gates", lambda: gates)
    report_path = tmp_path / "metrics" / "doctor.json"
    arguments = ["doctor", "--allow-dirty", "--cache", str(tmp_path / "cache"), "--out", str(report_path)]

    code = cli.main([*arguments, "--models", "gemini-3.8-flash,gemini-4-argon"])

    assert code == 2
    captured = capsys.readouterr()
    assert f"more than its spend limit of {spend_limit:.3f} USD" in captured.err
    assert "times its estimate" not in captured.err
    assert f"this run stops above {spend_limit:.3f} USD (its spend limit in the gate file)" in captured.out
    assert "measured cost" in captured.out
    # The checks of the first model cost less than the spend limit, so they are finished before the stop.
    report = read_json(report_path)
    assert report["passed"] is False
    assert f"more than its spend limit of {spend_limit:.3f} USD" in report["stopped_by_budget"]
    assert list(report["contract"]) == ["gemini-3.8-flash"]
    assert first_model < spend_limit < report["provenance"]["cost"]["measured_usd"] < first_model + second_model
    cost = report["provenance"]["cost"]
    assert cost["new_answers"] == len(backend.requests) == len(read_ledger(tmp_path / "cache"))
    # The mean tokens of the paid answers are what the next estimate needs.
    people = cost["steps"]["people"]
    assert people["input_tokens"] == people["answers"] * STEP_USAGE["people"].input_tokens


@pytest.mark.parametrize("flex_status", [503, 429])
def test_doctor_passes_gate_g0_without_a_flex_verdict_when_the_flex_tier_is_busy(
    private_git: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_scene: SyntheticScene,
    flex_status: int,
) -> None:
    """The real backend on a fake SDK client whose flex tier answers 503 or 429 for an hour.

    These are the documented answers of a flex tier without capacity. The backend waits 900 seconds
    for capacity, and then the check is a fact without a verdict: a busy provider is no client fault.
    """
    set_up_doctor(tmp_path, monkeypatch)
    clock = FakeClock()
    sdk_client = ContractSdkClient(contract_scene, clock, flex_status=flex_status, flex_busy_s=3600.0)
    recorder = doctor.RecordingClient(sdk_client)
    backend = GeminiBackend(client=recorder, clock=clock.clock, sleep=clock.sleep)
    monkeypatch.setattr(cli, "recording_backend", lambda max_usd_per_10_min: (backend, recorder))
    report_path, cache = tmp_path / "doctor.json", tmp_path / "cache"

    code = cli.main(["doctor", "--allow-dirty", "--cache", str(cache), "--out", str(report_path)])

    assert code == 0
    report = read_json(report_path)
    assert report["passed"] is True
    checks = {check["name"]: check for check in report["contract"]["gemini-4-argon"]["checks"]}
    assert checks[doctor.FLEX]["status"] == "info"
    assert "not checked: the flex tier had no capacity for any of the 1 requests" in checks[doctor.FLEX]["detail"]
    assert f"(status {flex_status})" in checks[doctor.FLEX]["detail"]
    assert all(check["status"] != "fail" for check in checks.values())
    # The flex request was tried again for 900 seconds, far more than four times, and nothing of it was billed:
    # the six requests of the standard tier are the only ledger lines.
    assert clock.now - 1000.0 >= 900.0
    assert 15 <= sdk_client.flex_calls <= 40
    assert [line["service_tier"] for line in read_ledger(cache)] == ["standard"] * 6
    assert report["provenance"]["cost"]["new_answers"] == 6


def test_doctor_gets_its_flex_answer_after_two_busy_minutes(
    private_git: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contract_scene: SyntheticScene
) -> None:
    """Four attempts in one minute were the end of a flex request before. Now the request waits for capacity."""
    set_up_doctor(tmp_path, monkeypatch)
    clock = FakeClock()
    sdk_client = ContractSdkClient(contract_scene, clock, flex_status=503, flex_busy_s=120.0)
    recorder = doctor.RecordingClient(sdk_client)
    backend = GeminiBackend(client=recorder, clock=clock.clock, sleep=clock.sleep)
    monkeypatch.setattr(cli, "recording_backend", lambda max_usd_per_10_min: (backend, recorder))
    report_path, cache = tmp_path / "doctor.json", tmp_path / "cache"

    code = cli.main(["doctor", "--allow-dirty", "--cache", str(cache), "--out", str(report_path)])

    assert code == 0
    contract = read_json(report_path)["contract"]["gemini-4-argon"]
    checks = {check["name"]: check for check in contract["checks"]}
    assert (checks[doctor.FLEX]["status"], checks[doctor.FLEX]["value"]) == ("ok", 1.0)
    flex_answer = contract["answers"][-1]
    assert flex_answer["service_tier"] == "flex"
    assert flex_answer["attempts"] == sdk_client.flex_calls > 4
    assert [line["service_tier"] for line in read_ledger(cache)] == [*["standard"] * 6, "flex"]


def test_doctor_stops_above_the_spend_limit_of_gate_g0_and_its_spend_check_reads_the_measured_cost(
    private_git: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contract_scene: SyntheticScene
) -> None:
    """The estimate assumes nine tenths of the tokens, and the limit of gate G0 lies between it and the real cost.

    The run stays below 1.3 times its estimate, so that rule would not end it. The last answer
    takes it above the spend limit: the budget guard stops it there.
    """
    set_up_doctor(tmp_path, monkeypatch)
    recorder = doctor.RecordingClient(SimpleNamespace(interactions=None))
    backend = CountingBackend(FakeBackend(contract_scene))
    monkeypatch.setattr(cli, "recording_backend", lambda max_usd_per_10_min: (backend, recorder))
    estimates_of_a_share(monkeypatch, 0.9)
    # Three people and three landmarks answers at the standard price, and one people answer at half of it.
    real_cost = 3.5 * fake_cost("people") + 3 * fake_cost("landmarks")
    spend_limit = 0.95 * real_cost
    gates = gates_with(g0_contract={"spend_usd_max": spend_limit})
    monkeypatch.setattr(cli, "load_gates", lambda: gates)
    report_path = tmp_path / "doctor.json"

    code = cli.main(["doctor", "--allow-dirty", "--cache", str(tmp_path / "cache"), "--out", str(report_path)])

    assert code == 2
    report = read_json(report_path)
    cost = report["provenance"]["cost"]
    assert cost["estimate_usd"] < spend_limit < cost["measured_usd"] == pytest.approx(real_cost)
    assert cost["measured_usd"] < 1.3 * cost["estimate_usd"]
    # All seven answers were paid: the stop came after the answer that passed the limit.
    assert cost["new_answers"] == len(backend.requests) == 7
    assert "its spend limit" in report["stopped_by_budget"]
    assert (report["spend"]["status"], report["spend"]["value"]) == ("fail", pytest.approx(real_cost))
    assert report["spend"]["limit"] == pytest.approx(spend_limit)
    assert report["passed"] is False


def test_doctor_finishes_when_it_costs_ten_times_its_estimate_and_less_than_the_spend_limit(
    private_git: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    contract_scene: SyntheticScene,
) -> None:
    """The estimate of the contract checks is the least certain one: their answers are the first real ones.

    So the checks stop above the spend limit of gate G0, and not above 1.3 times their estimate.
    """
    set_up_doctor(tmp_path, monkeypatch)
    recorder = doctor.RecordingClient(SimpleNamespace(interactions=None))
    monkeypatch.setattr(cli, "recording_backend", lambda max_usd_per_10_min: (FakeBackend(contract_scene), recorder))
    estimates_of_a_share(monkeypatch, 0.1)
    spend_limit = gates_with()["g0_contract"]["spend_usd_max"]
    report_path, cache = tmp_path / "doctor.json", tmp_path / "cache"

    code = cli.main(["doctor", "--allow-dirty", "--cache", str(cache), "--out", str(report_path)])

    assert code == 0
    report = read_json(report_path)
    cost = report["provenance"]["cost"]
    assert 9 * cost["estimate_usd"] < cost["measured_usd"] < spend_limit
    assert report["stopped_by_budget"] is None
    assert report["passed"] is True
    assert (report["spend"]["status"], report["spend"]["limit"]) == ("ok", spend_limit)
    assert len(read_ledger(cache)) == cost["new_answers"] == 7
    output = capsys.readouterr().out
    assert f"this run stops above {spend_limit:.3f} USD (its spend limit in the gate file)" in output
    assert "times its estimate" not in output


def test_doctor_starts_no_check_when_its_estimate_is_above_the_spend_limit_of_gate_g0(
    private_git: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    contract_scene: SyntheticScene,
) -> None:
    """Such checks would be paid until the spend limit and would give no verdict."""
    set_up_doctor(tmp_path, monkeypatch)
    recorder = doctor.RecordingClient(SimpleNamespace(interactions=None))
    backend = CountingBackend(FakeBackend(contract_scene))
    monkeypatch.setattr(cli, "recording_backend", lambda max_usd_per_10_min: (backend, recorder))
    # The estimate is the real cost here (fixture ``estimates_of_the_fake_model``).
    real_cost = 3.5 * fake_cost("people") + 3 * fake_cost("landmarks")
    gates = gates_with(g0_contract={"spend_usd_max": 0.99 * real_cost})
    monkeypatch.setattr(cli, "load_gates", lambda: gates)
    report_path, cache = tmp_path / "doctor.json", tmp_path / "cache"
    arguments = ["doctor", "--allow-dirty", "--cache", str(cache), "--out", str(report_path)]

    code = cli.main(arguments)

    assert code == 2
    error = capsys.readouterr().err
    assert re.search(r"the estimate of this run is \d\.\d{3} USD, more than its spend limit of \d\.\d{3} USD", error)
    assert "could not finish" in error
    assert backend.requests == []
    assert not report_path.exists()
    assert not (cache / "ledger.jsonl").exists()
    # Control: with a spend limit just above the estimate, the same checks start and finish.
    above_the_estimate = gates_with(g0_contract={"spend_usd_max": 1.01 * real_cost})
    monkeypatch.setattr(cli, "load_gates", lambda: above_the_estimate)
    assert cli.main(arguments) == 0
    assert len(backend.requests) == 7
    assert read_json(report_path)["stopped_by_budget"] is None


def test_doctor_with_start_over_budget_starts_no_check_when_its_estimate_is_above_the_spend_limit(
    private_git: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    contract_scene: SyntheticScene,
) -> None:
    """The option --start-over-budget does not lift the spend limit of gate G0."""
    set_up_doctor(tmp_path, monkeypatch)
    recorder = doctor.RecordingClient(SimpleNamespace(interactions=None))
    backend = CountingBackend(FakeBackend(contract_scene))
    monkeypatch.setattr(cli, "recording_backend", lambda max_usd_per_10_min: (backend, recorder))
    # The estimate is the real cost here (fixture ``estimates_of_the_fake_model``).
    real_cost = 3.5 * fake_cost("people") + 3 * fake_cost("landmarks")
    gates = gates_with(g0_contract={"spend_usd_max": 0.99 * real_cost})
    monkeypatch.setattr(cli, "load_gates", lambda: gates)
    report_path, cache = tmp_path / "doctor.json", tmp_path / "cache"

    code = cli.main(
        ["doctor", "--allow-dirty", "--cache", str(cache), "--out", str(report_path), "--start-over-budget"]
    )

    assert code == 2
    assert "more than its spend limit of" in capsys.readouterr().err
    assert backend.requests == []
    assert not report_path.exists()
    assert not (cache / "ledger.jsonl").exists()


def test_doctor_finishes_when_the_ledger_holds_more_than_the_spend_limit_of_gate_g0(
    private_git: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contract_scene: SyntheticScene
) -> None:
    """The contract checks run again after other paid runs. Only their own cost counts against their limit."""
    set_up_doctor(tmp_path, monkeypatch)
    recorder = doctor.RecordingClient(SimpleNamespace(interactions=None))
    backend = CountingBackend(FakeBackend(contract_scene))
    monkeypatch.setattr(cli, "recording_backend", lambda max_usd_per_10_min: (backend, recorder))
    spend_limit = gates_with()["g0_contract"]["spend_usd_max"]
    report_path, cache = tmp_path / "doctor.json", tmp_path / "cache"
    write_earlier_spend(cache, 2.5 * spend_limit)

    code = cli.main(["doctor", "--allow-dirty", "--cache", str(cache), "--out", str(report_path)])

    assert code == 0
    assert len(backend.requests) == 7
    report = read_json(report_path)
    assert report["stopped_by_budget"] is None
    assert report["spend"]["status"] == "ok"


def test_doctor_stops_at_the_hard_stop_although_it_is_below_the_spend_limit_of_gate_g0(
    private_git: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    contract_scene: SyntheticScene,
) -> None:
    """The hard stop of the budget ends the contract checks too."""
    set_up_doctor(tmp_path, monkeypatch)
    recorder = doctor.RecordingClient(SimpleNamespace(interactions=None))
    backend = CountingBackend(FakeBackend(contract_scene))
    monkeypatch.setattr(cli, "recording_backend", lambda max_usd_per_10_min: (backend, recorder))
    real_cost = 3.5 * fake_cost("people") + 3 * fake_cost("landmarks")
    # The ledger holds 1 USD, and the hard stop lies half of the cost of the checks above it.
    gates = gates_with(budget={"hard_stop_usd": 1.0 + 0.5 * real_cost})
    monkeypatch.setattr(cli, "load_gates", lambda: gates)
    report_path, cache = tmp_path / "doctor.json", tmp_path / "cache"
    write_earlier_spend(cache, 1.0)

    code = cli.main(
        ["doctor", "--allow-dirty", "--cache", str(cache), "--out", str(report_path), "--start-over-budget"]
    )

    assert code == 2
    assert "hard stop: the ledger total is" in capsys.readouterr().err
    assert 0 < len(backend.requests) < 7
    assert "hard stop" in read_json(report_path)["stopped_by_budget"]


@pytest.mark.parametrize("spend_usd_max", [0, 0.0, -1, float("nan"), float("inf")])
def test_doctor_refuses_a_spend_limit_of_gate_g0_that_is_no_amount_above_zero(
    private_git: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    contract_scene: SyntheticScene,
    spend_usd_max: float,
) -> None:
    """One error line names the gate file and the key. No estimate, no budget line and no model call come before it."""
    set_up_doctor(tmp_path, monkeypatch)
    recorder = doctor.RecordingClient(SimpleNamespace(interactions=None))
    backend = CountingBackend(FakeBackend(contract_scene))
    monkeypatch.setattr(cli, "recording_backend", lambda max_usd_per_10_min: (backend, recorder))
    gates = gates_with(g0_contract={"spend_usd_max": spend_usd_max})
    monkeypatch.setattr(cli, "load_gates", lambda: gates)
    cache = tmp_path / "cache"

    code = cli.main(["doctor", "--allow-dirty", "--cache", str(cache)])

    assert code == 2
    captured = capsys.readouterr()
    assert "configs/gates.yaml: g0_contract.spend_usd_max must be a number above 0, found" in captured.err
    assert "Traceback" not in captured.err
    assert "cost estimate" not in captured.out
    assert "this run stops above" not in captured.out
    assert backend.requests == []
    assert not cache.exists()


def test_doctor_makes_no_model_call_when_a_check_of_the_machine_failed(
    private_git: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The contract checks are paid. They must not start on a machine whose own checks fail."""
    repo = set_up_doctor(tmp_path, monkeypatch)
    # The deny-list of the guard is gone: the check of the machine fails.
    (repo / ".git" / "guard-deny.txt").unlink()

    def no_model(max_usd_per_10_min: float) -> None:
        raise AssertionError("the contract checks must not start")

    monkeypatch.setattr(cli, "recording_backend", no_model)
    report_path, cache = tmp_path / "doctor.json", tmp_path / "cache"

    code = cli.main(["doctor", "--allow-dirty", "--cache", str(cache), "--out", str(report_path)])

    assert code == 1
    output = capsys.readouterr().out
    assert "[fail] deny-list" in output
    assert "The contract checks did not run. Repair the failed checks first." in output
    # Nothing of the paid part happened: no estimate, no budget line, no ledger and no report.
    assert "cost estimate" not in output
    assert "budget:" not in output
    assert not cache.exists()
    assert not report_path.exists()


def test_call_in_flight_of_the_doctor_is_recorded_when_the_doctor_is_stopped_by_a_signal(
    private_git: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    contract_scene: SyntheticScene,
) -> None:
    """The contract checks call the model in the main thread. SIGTERM comes while the second call is in flight."""
    set_up_doctor(tmp_path, monkeypatch)
    recorder = doctor.RecordingClient(SimpleNamespace(interactions=None))
    backend = SignallingBackend(FakeBackend(contract_scene), at_request=2)
    monkeypatch.setattr(cli, "recording_backend", lambda max_usd_per_10_min: (backend, recorder))
    cache = tmp_path / "cache"

    code = cli.main(["doctor", "--allow-dirty", "--cache", str(cache), "--out", str(tmp_path / "doctor.json")])

    assert code == 2
    captured = capsys.readouterr()
    assert captured.err.splitlines()[-1].startswith("error: the command was interrupted (SIGTERM).")
    assert "measured cost" in captured.out
    # The call in flight finished, and its answer has a ledger line: every call that started is recorded.
    assert backend.started == len(backend.answered) >= 2
    assert sorted(line["request"] for line in read_ledger(cache)) == sorted(backend.answered)


# --- git state -------------------------------------------------------------------------


def test_code_state_of_a_folder_outside_git_is_dirty_and_uncommitted(private_git: None, tmp_path: Path) -> None:
    folder = tmp_path / "plain"
    folder.mkdir()

    state = cli.code_state(folder)

    assert (state.commit, state.dirty) == (cli.UNCOMMITTED, True)


def test_code_state_follows_commits_and_changes(private_git: None, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "--quiet", "--template=")
    # No commit yet, and no file: clean, but there is no commit to name.
    assert cli.code_state(repo) == cli.CodeState(commit=cli.UNCOMMITTED, dirty=False, changes=())

    (repo / "code.py").write_text("x = 1\n", encoding="utf-8")
    assert cli.code_state(repo).dirty is True
    author = ["-c", "user.name=Tester", "-c", "user.email=tester" + "@" + "example.org", "-c", "commit.gpgsign=false"]
    git(repo, "add", "code.py")
    git(repo, *author, "commit", "--quiet", "--no-verify", "-m", "first")
    committed = cli.code_state(repo)
    assert committed.dirty is False
    assert committed.commit == git(repo, "rev-parse", "HEAD").strip()
    assert committed.commit != cli.UNCOMMITTED

    (repo / "notes.txt").write_text("not tracked\n", encoding="utf-8")
    changed = cli.code_state(repo)
    assert changed.dirty is True
    assert changed.changes == ("?? notes.txt",)
    assert changed.commit == committed.commit


def test_model_commands_refuse_a_dirty_work_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dirty = cli.CodeState(commit="0" * 40, dirty=True, changes=(" M src/pitchvlm/cli.py",))
    monkeypatch.setattr(cli, "code_state", lambda repo: dirty)
    out = tmp_path / "out"

    code = cli.main(["run", "--clip", str(tmp_path / "clip.mp4"), "--out", str(out)])

    assert code == 2
    assert "--allow-dirty" in capsys.readouterr().err
    assert not out.exists()


# --- run -------------------------------------------------------------------------------


def test_run_with_a_fake_model_and_then_from_the_cache(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A paid run writes its files, the stored answers and the ledger. A replay gives the same tracks for free."""
    scene, root = clip_on_disk
    clip_dir = root / "SoccerNetGS-valid" / CLIP_ID
    out, cache = tmp_path / "runs", tmp_path / "cache"
    videos: list[Path] = []

    def keep_video_path(source: object, result: ClipResult, out_path: Path) -> Path:
        videos.append(out_path)
        return out_path

    monkeypatch.setattr(cli, "render_clip", keep_video_path)
    monkeypatch.setattr(cli, "gemini_backend", lambda max_usd_per_10_min: FakeBackend(scene))
    arguments = [
        "run",
        "--clip",
        str(clip_dir),
        "--labels",
        str(clip_dir / LABEL_NAME),
        "--out",
        str(out),
        "--cache",
        str(cache),
        "--allow-dirty",
    ]

    assert cli.main(arguments) == 0

    first = read_json(out / CLIP_ID / "result.json")
    result = ClipResult.model_validate(first["result"])
    assert result.info.clip_id == CLIP_ID
    assert result.info.n_frames == N_FRAMES
    assert len(result.frames) == N_FRAMES
    assert len(result.tracks) >= 10
    provenance = first["provenance"]
    assert provenance["command"] == "run"
    assert isinstance(provenance["commit"], str) and provenance["commit"]
    cost = provenance["cost"]
    assert cost["replay"] is False
    # One kit legend request, and one people and one landmarks request per keyframe.
    assert cost["steps"]["kit_legend"]["answers"] == 1
    assert cost["steps"]["people"]["answers"] == len(KEYFRAMES)
    assert cost["steps"]["landmarks"]["answers"] == len(KEYFRAMES)
    assert cost["steps"]["number"]["answers"] >= 1
    assert 0.0 < cost["measured_usd"] <= 1.3 * cost["estimate_usd"]
    ledger_lines = (cache / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ledger_lines) == cost["new_answers"]
    assert {json.loads(line)["run_id"] for line in ledger_lines} == {cost["run_id"]}
    assert len(list(cache.glob(f"{CLIP_ID}/*/*.json"))) == cost["new_answers"]
    predictions = read_json(out / "predictions" / "SoccerNetGS-valid" / "pitchvlm" / f"{CLIP_ID}.json")
    assert {item["image_id"] for item in predictions["predictions"]} == {
        f"2999{number:06d}" for number in range(1, N_FRAMES + 1)
    }
    assert videos == [out / CLIP_ID / f"{CLIP_ID}.mp4"]
    output = capsys.readouterr().out
    assert output.index("cost estimate") < output.index("measured cost")

    def no_model(max_usd_per_10_min: float) -> FakeBackend:
        raise AssertionError("a replay must not build the model backend")

    monkeypatch.setattr(cli, "gemini_backend", no_model)

    assert cli.main([*arguments, "--replay"]) == 0

    second = read_json(out / CLIP_ID / "result.json")
    assert second["provenance"]["cost"]["replay"] is True
    assert second["provenance"]["cost"]["new_answers"] == 0
    # The cost record of a replay has the keys of a paid run. Every answer of the replay is a stored
    # one, and the stored answers cost what the paid run paid for them.
    assert set(second["provenance"]["cost"]) == set(cost)
    assert second["provenance"]["cost"]["stored_answers"] == cost["new_answers"]
    assert second["provenance"]["cost"]["stored_usd"] == pytest.approx(cost["measured_usd"])
    assert (cost["stored_answers"], cost["stored_usd"]) == (0, 0.0)
    replay_output = capsys.readouterr().out
    assert "measured cost: 0.000 USD (replay mode: no model call was made)" in replay_output
    assert f"stored answers used: {cost['new_answers']}; they cost {cost['measured_usd']:.3f} USD" in replay_output
    # Both cost records count the requests of the run by step: the real number of number requests is in them.
    requests = second["provenance"]["cost"]["requests"]
    assert requests == cost["requests"]
    assert (requests["kit_legend"], requests["people"], requests["landmarks"]) == (1, len(KEYFRAMES), len(KEYFRAMES))
    assert requests["number"] == cost["steps"]["number"]["answers"]
    assert second["result"]["tracks"] == first["result"]["tracks"]
    assert (cache / "ledger.jsonl").read_text(encoding="utf-8").splitlines() == ledger_lines


def test_replay_with_an_empty_cache_stops_at_the_first_request(
    clip_on_disk: tuple[SyntheticScene, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The free dry run of a clip: the clip line and the cost estimate are printed, then the run stops."""
    _, root = clip_on_disk
    clip_dir = root / "SoccerNetGS-valid" / CLIP_ID
    out = tmp_path / "runs"

    code = cli.main(
        [
            "run",
            "--clip",
            str(clip_dir / "img1"),
            "--fps",
            "25",
            "--out",
            str(out),
            "--cache",
            str(tmp_path / "empty-cache"),
            "--replay",
            "--allow-dirty",
        ]
    )

    assert code == 2
    captured = capsys.readouterr()
    assert f"clip {CLIP_ID}: {N_FRAMES} frames, 25 frames per second, 1280 x 720 pixels, 6 keyframes" in captured.out
    assert "cost estimate" in captured.out
    assert "no stored answer" in captured.err
    assert not out.exists()


def command_environment() -> dict[str, str]:
    """Return the environment of a command line that runs as a process of its own, from the source checkout."""
    environment = {name: value for name, value in os.environ.items() if name != "PYTHONUNBUFFERED"}
    environment["PYTHONPATH"] = str(REPO / "src")
    return environment


def test_estimate_and_budget_line_can_be_read_from_a_pipe_before_the_first_model_call(
    clip_on_disk: tuple[SyntheticScene, Path], tmp_path: Path
) -> None:
    """A run without attention writes into a pipe or a file, where Python keeps the output in a buffer.

    The command is a process of its own here, and its fake model waits for a gate file. The estimate
    and the budget line must arrive while the first request waits, before any answer is paid.
    """
    _, root = clip_on_disk
    gate, driver = tmp_path / "gate", tmp_path / "gated_run.py"
    driver.write_text(GATED_RUN, encoding="utf-8")
    arguments = run_arguments(root / "SoccerNetGS-valid" / CLIP_ID, tmp_path, "--fps", "25")
    process = subprocess.Popen(
        [sys.executable, str(driver), str(gate), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        cwd=tmp_path,
        env=command_environment(),
    )
    lines: queue.SimpleQueue[str | None] = queue.SimpleQueue()

    def read_the_pipe() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    reader = threading.Thread(target=read_the_pipe)
    reader.start()
    before_the_first_answer: list[str] = []
    try:
        # Read until the line that follows the estimate and the budget line. When the first request
        # has started and that line is still not there, two more seconds are enough to know.
        deadline = time.monotonic() + 120.0
        last_line_seen = False
        while time.monotonic() < deadline and not last_line_seen:
            try:
                line = lines.get(timeout=0.1)
            except queue.Empty:
                if (tmp_path / "waiting").exists():
                    deadline = min(deadline, time.monotonic() + 2.0)
                continue
            if line is None:
                break
            before_the_first_answer.append(line)
            last_line_seen = "number requests of the estimate" in line
        paid_before = (tmp_path / "cache" / "ledger.jsonl").exists()
    finally:
        gate.write_text("go", encoding="utf-8")
        code = process.wait(timeout=300)
        reader.join(timeout=60)

    shown = "".join(before_the_first_answer)
    assert "cost estimate: " in shown
    assert "budget: the ledger " in shown
    assert "this run stops above " in shown
    assert not paid_before
    assert code == 0
    assert len(read_ledger(tmp_path / "cache")) >= 13


def test_error_line_is_the_last_line_when_both_streams_go_to_one_pipe(
    clip_on_disk: tuple[SyntheticScene, Path], tmp_path: Path
) -> None:
    """A replay with an empty cache stops at its first request. Its output lines must come before the error line."""
    _, root = clip_on_disk
    arguments = run_arguments(root / "SoccerNetGS-valid" / CLIP_ID, tmp_path, "--fps", "25", "--replay")

    completed = subprocess.run(
        [sys.executable, "-m", "pitchvlm.cli", *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=tmp_path,
        env=command_environment(),
        check=False,
    )

    assert completed.returncode == 2
    lines = completed.stdout.splitlines()
    assert lines[-1].startswith("error: no stored answer for the kit_legend request")
    # The lines of the standard output come first, in their order.
    starts = ["clip SNGS-999: ", "cost estimate: ", "replay mode: ", "measured cost: ", "error: "]
    places = [next(index for index, line in enumerate(lines) if line.startswith(start)) for start in starts]
    assert places == sorted(places)


def test_run_needs_a_frame_rate_for_a_folder_of_images(
    clip_on_disk: tuple[SyntheticScene, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, root = clip_on_disk
    clip_dir = root / "SoccerNetGS-valid" / CLIP_ID

    code = cli.main(["run", "--clip", str(clip_dir), "--out", str(tmp_path / "runs"), "--allow-dirty"])

    assert code == 2
    assert "--fps" in capsys.readouterr().err


@pytest.mark.parametrize("clip_argument", [".", "img1"])
def test_run_names_the_clip_after_its_real_folder(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    clip_argument: str,
) -> None:
    """Inside the clip folder, ``.`` and ``img1`` do not hold the name of that folder. It is still the clip id."""
    _, root = clip_on_disk
    monkeypatch.chdir(root / "SoccerNetGS-valid" / CLIP_ID)

    code = cli.main(
        [
            "run",
            "--clip",
            clip_argument,
            "--fps",
            "25",
            "--out",
            str(tmp_path / "runs"),
            "--cache",
            str(tmp_path / "empty-cache"),
            "--replay",
            "--allow-dirty",
        ]
    )

    # The dry run stops at the first request. The clip line and the error of the missing answer name the clip.
    assert code == 2
    captured = capsys.readouterr()
    assert f"clip {CLIP_ID}: {N_FRAMES} frames" in captured.out
    assert "Traceback" not in captured.err
    assert CLIP_ID in captured.err


def tells_letter_case_apart(folder: Path) -> bool:
    """Tell whether the disk of a folder takes two names that differ only in their letter case for two files."""
    (folder / "CaseProbe").mkdir()
    return not (folder / "caseprobe").exists()


def test_disk_spelling_keeps_a_path_that_is_spelled_as_the_disk_stores_it(tmp_path: Path) -> None:
    video = tmp_path.resolve() / "Demo Clips" / "Match25.mp4"
    video.parent.mkdir()
    video.write_bytes(b"")

    assert cli.disk_spelling(video) == video
    assert cli.disk_spelling(video.parent) == video.parent


def test_clip_id_does_not_follow_the_letter_case_that_was_typed(
    clip_on_disk: tuple[SyntheticScene, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """On a disk that does not tell upper and lower case apart, every spelling opens the same clip.

    The clip id is a part of every cache key. With the typed spelling, the same clip would be paid again.
    """
    if tells_letter_case_apart(tmp_path):
        pytest.skip("the disk tells upper and lower case apart, so another spelling is another path")
    _, root = clip_on_disk
    video = tmp_path.resolve() / "Demo Clips" / "Match25.mp4"
    video.parent.mkdir()
    video.write_bytes(b"")
    assert cli.disk_spelling(tmp_path.resolve() / "demo clips" / "MATCH25.MP4") == video

    typed = root / "soccernetgs-VALID" / CLIP_ID.lower()
    code = cli.main(run_arguments(typed, tmp_path, "--fps", "25", "--replay"))

    # The dry run stops at the first request. The clip line and the missing answer name the clip as the disk does.
    assert code == 2
    captured = capsys.readouterr()
    assert f"clip {CLIP_ID}: {N_FRAMES} frames" in captured.out
    assert f"clip {CLIP_ID} " in captured.err


def test_clip_id_of_a_video_file_is_its_name_as_the_disk_spells_it(tmp_path: Path) -> None:
    """A video file that is typed in another letter case is the same clip. A link to it keeps its own name."""
    if not mp4v_available(tmp_path):
        pytest.skip("the installed OpenCV build cannot write the codec mp4v")
    video = tmp_path / "Match25.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
    for level in (40, 120, 200):
        writer.write(np.full((48, 64, 3), level, dtype=np.uint8))
    writer.release()
    (tmp_path / "Link.mp4").symlink_to(video)

    assert cli.open_clip(video, fps=None, labels=None).info.clip_id == "Match25"
    assert cli.open_clip(tmp_path / "Link.mp4", fps=None, labels=None).info.clip_id == "Link"
    if not tells_letter_case_apart(tmp_path):
        assert cli.open_clip(tmp_path / "MATCH25.MP4", fps=None, labels=None).info.clip_id == "Match25"
        assert cli.open_clip(tmp_path / "link.mp4", fps=None, labels=None).info.clip_id == "Link"


def test_sync_takes_a_results_folder_that_is_typed_in_another_letter_case(
    results_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Git names the work tree as the disk spells it, and the guard takes only files inside that work tree."""
    if tells_letter_case_apart(results_tree.parent):
        pytest.skip("the disk tells upper and lower case apart, so another spelling is another path")
    write_results(results_tree / "results")
    typed = results_tree.parent / results_tree.name.upper() / "Results"

    assert cli.main(["sync", "--results", str(typed)]) == 0

    output = capsys.readouterr().out
    assert "3 text result files" in output
    assert "add -- results" in output


def test_clip_id_of_folder_is_the_name_of_the_clip_folder() -> None:
    top = Path(Path.cwd().anchor)

    assert cli.clip_id_of_folder(top / "data" / "SNGS-021" / "img1") == "SNGS-021"
    assert cli.clip_id_of_folder(top / "data" / "frames") == "frames"
    # The top folder of a disk has no name, so it cannot name a clip.
    for folder in (top, top / "img1"):
        with pytest.raises(cli.CliError, match="no clip id"):
            cli.clip_id_of_folder(folder)


def test_cost_estimate_of_the_flex_tier_is_half_the_standard_price() -> None:
    standard = StepSettings()
    plan = [("people", standard, 2), ("landmarks", standard, 3)]
    flex_plan = [(step, doctor.flex_settings(settings), count) for step, settings, count in plan]

    assert cli.estimated_usd("clip", plan) > 0.0
    assert cli.estimated_usd("clip", flex_plan) == pytest.approx(0.5 * cli.estimated_usd("clip", plan))


def test_cost_estimate_needs_a_price_for_the_model() -> None:
    with pytest.raises(cli.CliError, match="no price"):
        cli.estimated_usd("clip", [("people", StepSettings(model="a-model-without-a-price"), 1)])


def test_cost_estimate_refuses_the_flex_tier_of_a_model_without_one() -> None:
    """The provider has no flex tier for the robotics model: the call would fail, at half the price in the estimate."""
    flex = StepSettings(model="gemini-robotics-er-2-preview", service_tier="flex")

    with pytest.raises(cli.CliError, match="landmarks step asks for the flex tier of the model gemini-robotics-er-2"):
        cli.estimated_usd("clip", [("people", StepSettings(), 2), ("landmarks", flex, 3)])

    # The standard tier of the same model has a price.
    assert cli.estimated_usd("clip", [("landmarks", StepSettings(model="gemini-robotics-er-2-preview"), 3)]) > 0.0


def test_run_refuses_the_flex_tier_of_a_model_without_one_before_the_estimate(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scene, root = clip_on_disk
    backend = use_fake_model(monkeypatch, scene)
    settings = tmp_path / "settings.yaml"
    settings.write_text("landmarks:\n  model: gemini-robotics-er-2-preview\n  service_tier: flex\n", encoding="utf-8")
    clip_dir = root / "SoccerNetGS-valid" / CLIP_ID

    code = cli.main(run_arguments(clip_dir, tmp_path, "--fps", "25", "--config", str(settings)))

    assert code == 2
    captured = capsys.readouterr()
    assert "has no flex tier" in captured.err
    assert "cost estimate" not in captured.out
    assert backend.requests == []
    assert not (tmp_path / "cache").exists()


def test_cost_estimate_of_a_run_counts_every_planned_request_at_the_price_of_its_step(
    clip_on_disk: tuple[SyntheticScene, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A free dry run of the clip, with another model for the number step."""
    _, root = clip_on_disk
    settings = tmp_path / "settings.yaml"
    settings.write_text("number:\n  model: gemini-3.8-flash\n", encoding="utf-8")
    frames = root / "SoccerNetGS-valid" / CLIP_ID / "img1"

    code = cli.main(run_arguments(frames, tmp_path, "--fps", "25", "--replay", "--config", str(settings)))

    assert code == 2  # The cache is empty, so the dry run stops at its first request.
    # The clip has 6 keyframes. The number of number requests is a guess: 50.
    assert cli.EXPECTED_NUMBER_REQUESTS == 50
    expected = (
        fake_cost("kit_legend")
        + 6 * fake_cost("people")
        + 6 * fake_cost("landmarks")
        + 50 * fake_cost("number", "gemini-3.8-flash")
    )
    output = capsys.readouterr().out
    assert (
        f"cost estimate: {expected:.3f} USD for 63 planned model calls "
        "(kit_legend 1, people 6, landmarks 6, number 50), at the prices of today and with an empty cache"
    ) in output
    assert "the 50 number requests of the estimate are a guess" in output


def test_money_limits_come_from_their_own_keys_of_the_gate_file() -> None:
    gates = {"budget": {"planned_usd": 1.0, "hard_stop_usd": 2.0, "run_abort_factor": 3.0, "max_usd_per_10_min": 4.0}}

    assert cli.budget_limits(gates) == cli.Budget(
        planned_usd=1.0, hard_stop_usd=2.0, run_factor=3.0, max_usd_per_10_min=4.0
    )
    # The gate file of the repository holds the limits of the project rules.
    assert cli.budget_limits(cli.load_gates()) == BUDGET
    assert cli.gate_number(cli.load_gates(), cli.G0_SECTION, "spend_usd_max") == 2.0


def test_run_whose_estimate_fits_exactly_under_the_hard_stop_starts(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "ledger.jsonl").write_text(json.dumps({"run_id": "an-earlier-run", "usd": 1.0}) + "\n", encoding="utf-8")
    budget = cli.Budget(planned_usd=1.0, hard_stop_usd=2.0, run_factor=1.3, max_usd_per_10_min=8.0)

    # 1 USD in the ledger and an estimate of 1 USD reach the hard stop of 2 USD and do not pass it.
    assert cli.start_meter(cache, budget, "run-1", 1.0).run_id == "run-1"
    with pytest.raises(BudgetExceeded, match="the estimate of this run is 1.250 USD: together more than the hard stop"):
        cli.start_meter(cache, budget, "run-2", 1.25)
    assert cli.start_meter(cache, budget, "run-3", 1.25, over_budget=True).run_id == "run-3"
    # A ledger that has reached the hard stop allows no start at all.
    (cache / "ledger.jsonl").write_text(json.dumps({"run_id": "an-earlier-run", "usd": 2.0}) + "\n", encoding="utf-8")
    with pytest.raises(BudgetExceeded, match="hard stop: the ledger total is 2.00 USD"):
        cli.start_meter(cache, budget, "run-4", 0.0, over_budget=True)


@pytest.mark.parametrize("given", [False, True])
def test_start_over_budget_option_reaches_the_budget_check_of_every_paid_command(
    private_git: None,
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    given: bool,
) -> None:
    """The commands run, probe and doctor open the cost ledger with the choice of the option.

    Only the doctor gives its run a spend limit of its own: the one of gate G0.
    """
    _, root = clip_on_disk
    seen: list[tuple[str, bool, float | None]] = []

    def stop_here(
        cache_dir: Path,
        budget: cli.Budget,
        run_id: str,
        estimate: float,
        *,
        spend_limit_usd: float | None = None,
        over_budget: bool,
    ) -> None:
        seen.append((run_id.split("-")[0], over_budget, spend_limit_usd))
        raise BudgetExceeded("the test stops the command here")

    monkeypatch.setattr(cli, "start_meter", stop_here)
    option = ["--start-over-budget"] if given else []
    manifest = tmp_path / "splits.json"
    manifest.write_text(json.dumps({"seed": 7, "valid": {"probe": [CLIP_ID]}}), encoding="utf-8")
    repo = tmp_path / "repo"
    set_up_clone(repo)
    (repo / "configs").mkdir()
    shutil.copy(GATES, repo / "configs" / "gates.yaml")
    monkeypatch.setenv(cli.KEY_VARIABLE, "not-a-real-value")
    monkeypatch.setattr(doctor, "installed_sdk_version", lambda: "3.0.test")
    cache = ["--cache", str(tmp_path / "cache"), "--allow-dirty"]
    probe = ["probe", "pitch", "--experiment", str(EXPERIMENTS / "pitch_probe.yaml"), "--root", str(root)]
    probe += ["--manifest", str(manifest), "--out", str(tmp_path / "pitch.json")]

    assert cli.main([*run_arguments(root / "SoccerNetGS-valid" / CLIP_ID, tmp_path, "--fps", "25"), *option]) == 2
    assert cli.main([*probe, *cache, *option]) == 2
    monkeypatch.setattr(cli, "repository_root", lambda: repo)
    assert cli.main(["doctor", *cache, *option]) == 2

    spend_limit_of_g0 = gates_with()["g0_contract"]["spend_usd_max"]
    assert seen == [("run", given, None), ("probe", given, None), ("doctor", given, spend_limit_of_g0)]


def test_budget_line_names_the_planned_budget_and_warns_above_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    budget = cli.Budget(planned_usd=1.0, hard_stop_usd=2.0, run_factor=1.3, max_usd_per_10_min=8.0)

    (cache / "ledger.jsonl").write_text(json.dumps({"run_id": "an-earlier-run", "usd": 1.0}) + "\n", encoding="utf-8")
    cli.start_meter(cache, budget, "run-1", 0.5)
    at_the_plan = capsys.readouterr().out
    assert "holds 1.000 USD; the planned budget is 1.000 USD and the hard stop is 2.000 USD" in at_the_plan
    assert "this run stops above 0.650 USD (1.3 times its estimate)" in at_the_plan
    assert "warning" not in at_the_plan

    (cache / "ledger.jsonl").write_text(json.dumps({"run_id": "an-earlier-run", "usd": 1.25}) + "\n", encoding="utf-8")
    cli.start_meter(cache, budget, "run-2", 0.5)
    above_the_plan = capsys.readouterr().out
    assert "warning: the ledger is 0.250 USD above the planned budget." in above_the_plan
    assert "Paid runs end at the hard stop of 2.000 USD." in above_the_plan


def test_spend_rate_limit_of_the_gate_file_reaches_the_model_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[dict[str, Any]] = []

    class StandIn:
        """Keeps the options that the real model backend would get."""

        def __init__(self, **options: Any) -> None:
            made.append(options)

    monkeypatch.setattr(cli, "GeminiBackend", StandIn)
    cli.gemini_backend(3.5)
    # The backend also gets the event that a stop signal sets, so that it sends nothing after the signal.
    assert made == [{"api_key_env": "GEMINI_API_KEY", "max_usd_per_10_min": 3.5, "stop": cli.STOP_REQUESTED}]

    gates = gates_with(budget={"max_usd_per_10_min": 2.5})
    monkeypatch.setattr(cli, "load_gates", lambda: gates)
    limits: list[float] = []
    monkeypatch.setattr(cli, "gemini_backend", lambda max_usd_per_10_min: limits.append(max_usd_per_10_min))
    plan = [("people", StepSettings(), 1)]
    cli.open_session("run", "clip", plan, cache_dir=tmp_path / "cache", replay=False, repeat=0)
    assert limits == [2.5]


def test_recording_backend_builds_the_sdk_client_with_the_longest_timeout_of_any_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The doctor sends one request through the flex tier, which can wait in a queue for minutes."""
    built: list[tuple[str, float]] = []
    made: list[dict[str, Any]] = []
    sdk_client = SimpleNamespace(interactions=None)

    def build(api_key_env: str, timeout_s: float) -> SimpleNamespace:
        built.append((api_key_env, timeout_s))
        return sdk_client

    class StandIn:
        """Keeps the options that the real model backend would get."""

        def __init__(self, **options: Any) -> None:
            made.append(options)

    monkeypatch.setattr(cli, "build_sdk_client", build)
    monkeypatch.setattr(cli, "GeminiBackend", StandIn)

    backend, recorder = cli.recording_backend(2.5)

    assert built == [("GEMINI_API_KEY", 900.0)]
    assert cli.FLEX_TIMEOUT_S == 900.0 > cli.STANDARD_TIMEOUT_S
    assert isinstance(backend, StandIn) and isinstance(recorder, doctor.RecordingClient)
    # The backend gets no timeout of its own: it gives every call the timeout of its service tier.
    # It gets the event that a stop signal sets.
    assert made == [
        {"api_key_env": "GEMINI_API_KEY", "max_usd_per_10_min": 2.5, "stop": cli.STOP_REQUESTED, "client": recorder}
    ]


def test_calls_of_the_recording_backend_get_the_timeout_of_their_service_tier(
    contract_scene: SyntheticScene, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flex request of the doctor waits up to 900 seconds, a standard request 300 seconds."""
    timeouts: list[tuple[str, float]] = []

    def create(**kwargs: Any) -> SimpleNamespace:
        timeouts.append((kwargs["service_tier"], kwargs["timeout"]))
        return SimpleNamespace(
            status="completed",
            output_text='{"people": []}',
            service_tier=kwargs["service_tier"],
            usage=SimpleNamespace(
                total_input_tokens=1500, total_output_tokens=300, total_thought_tokens=100, total_tokens=1900
            ),
        )

    sdk_client = SimpleNamespace(interactions=SimpleNamespace(create=create))
    monkeypatch.setattr(cli, "build_sdk_client", lambda api_key_env, timeout_s: sdk_client)
    backend, recorder = cli.recording_backend(8.0)
    legend, info = contract_scene.kit_legend(), contract_scene.info
    for settings in (StepSettings(), doctor.flex_settings(StepSettings())):
        request = steps.build_people_request(
            contract_scene.image(0), legend, clip_id=info.clip_id, frame=0, settings=settings
        )
        backend.generate(request)

    assert timeouts == [("standard", 300.0), ("flex", 900.0)]
    assert [call.sent_tier for call in recorder.calls] == ["standard", "flex"]


def test_run_stops_when_it_costs_more_than_1_3_times_its_estimate(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The estimate assumes a tenth of the tokens that the fake model reports: the run costs 10 times its estimate."""
    scene, root = clip_on_disk
    clip_dir = root / "SoccerNetGS-valid" / CLIP_ID
    estimates_of_a_share(monkeypatch, 0.1)
    backend = use_fake_model(monkeypatch, scene)

    code = cli.main(run_arguments(clip_dir, tmp_path, "--labels", str(clip_dir / LABEL_NAME)))

    assert code == 2
    captured = capsys.readouterr()
    # Four requests are in flight. The budget guard stops the run after the answer that passes the limit, or
    # the meter stops it before the next call: both stop messages name the limit with the same words.
    assert "1.3 times its estimate" in captured.err
    assert "measured cost" in captured.out
    assert not (tmp_path / "runs").exists()
    # The run stops in the keyframe step: no number request is sent. Every paid answer is in the ledger.
    lines = read_ledger(tmp_path / "cache")
    assert len(lines) == len(backend.requests)
    assert all(request.meta.step != "number" for request in backend.requests)
    # The limit is 1.3 times the estimate. The requests in flight when an answer crosses it are still paid:
    # at most 3 with 4 requests in parallel.
    keyframe_cost = 6 * fake_cost("people") + 6 * fake_cost("landmarks")
    limit = 1.3 * 0.1 * (fake_cost("kit_legend") + keyframe_cost + 50 * fake_cost("number"))
    paid = 0.0
    for crossed_at, line in enumerate(lines, start=1):
        paid += line["usd"]
        if paid > limit:
            break
    assert paid > limit
    assert crossed_at <= len(lines) <= crossed_at + 3
    assert len(lines) < 13


def test_hard_stop_of_the_gate_file_is_checked_before_the_first_model_call(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scene, root = clip_on_disk
    clip_dir = root / "SoccerNetGS-valid" / CLIP_ID
    gates = gates_with(budget={"hard_stop_usd": 50.0})
    monkeypatch.setattr(cli, "load_gates", lambda: gates)
    backend = use_fake_model(monkeypatch, scene)
    cache = tmp_path / "cache"
    cache.mkdir()
    earlier_spend = json.dumps({"run_id": "an-earlier-run", "usd": 60.0}) + "\n"
    (cache / "ledger.jsonl").write_text(earlier_spend, encoding="utf-8")

    code = cli.main(run_arguments(clip_dir, tmp_path, "--labels", str(clip_dir / LABEL_NAME)))

    assert code == 2
    assert "hard stop" in capsys.readouterr().err
    # No model call was made, and nothing was added to the ledger.
    assert backend.requests == []
    assert (cache / "ledger.jsonl").read_text(encoding="utf-8") == earlier_spend


def test_run_stops_when_the_ledger_reaches_the_hard_stop_during_the_run(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The hard stop of this gate file is 0.05 USD, the cost of four answers. The estimate of the run is right."""
    scene, root = clip_on_disk
    clip_dir = root / "SoccerNetGS-valid" / CLIP_ID
    gates = gates_with(budget={"hard_stop_usd": 0.05})
    monkeypatch.setattr(cli, "load_gates", lambda: gates)
    backend = use_fake_model(monkeypatch, scene)
    arguments = run_arguments(clip_dir, tmp_path, "--fps", "25")

    # The estimate of the run does not fit under this hard stop: the run does not start at all.
    assert cli.main(arguments) == 2
    refusal = capsys.readouterr().err
    assert "together more than the hard stop of 0.050 USD, so the run could not finish" in refusal
    assert "--start-over-budget" in refusal
    assert backend.requests == []
    assert not (tmp_path / "cache" / "ledger.jsonl").exists()

    code = cli.main([*arguments, "--start-over-budget"])

    assert code == 2
    assert "hard stop: the ledger total is" in capsys.readouterr().err
    paid = sum(line["usd"] for line in read_ledger(tmp_path / "cache"))
    # The answers in flight when the total reaches the hard stop are still paid: at most 3 of them.
    assert 0.05 <= paid < 0.05 + 4 * fake_cost("people")
    assert all(request.meta.step != "number" for request in backend.requests)


def test_run_that_is_started_again_after_a_budget_stop_pays_for_no_further_answer(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The stored answers of the stopped run count against the limit of the next run of the same work.

    Without this rule every new start would get 1.3 times the full estimate again, and a clip
    whose answers cost 10 times their estimate would be finished after some starts.
    """
    scene, root = clip_on_disk
    clip_dir = root / "SoccerNetGS-valid" / CLIP_ID
    settings = tmp_path / "settings.yaml"
    settings.write_text("max_parallel: 1\n", encoding="utf-8")
    arguments = run_arguments(clip_dir, tmp_path, "--fps", "25", "--config", str(settings))
    backend = use_fake_model(monkeypatch, scene)
    estimates_of_a_share(monkeypatch, 0.1)

    assert cli.main(arguments) == 2
    first_error = capsys.readouterr().err
    assert "1.3 times its estimate" in first_error
    paid_by_the_first_run = len(backend.requests)
    assert 3 <= paid_by_the_first_run < 13

    assert cli.main(arguments) == 2
    captured = capsys.readouterr()
    # The second run reads the stored answers and stops before it pays for the first missing one.
    assert len(backend.requests) == paid_by_the_first_run
    assert len(read_ledger(tmp_path / "cache")) == paid_by_the_first_run
    assert "used stored answers that cost" in captured.err
    assert "correct the token estimates" in captured.err
    assert f"stored answers used: {paid_by_the_first_run}" in captured.out
    assert "measured cost: 0.000 USD for 0 new answers" in captured.out

    # With an estimate that fits the answers, the run goes on and pays for every answer one time.
    estimates_of_a_share(monkeypatch, 1.0)
    assert cli.main(arguments) == 0
    result = read_json(tmp_path / "runs" / CLIP_ID / "result.json")
    cost = result["provenance"]["cost"]
    assert cost["stored_answers"] == paid_by_the_first_run
    assert cost["new_answers"] == len(backend.requests) - paid_by_the_first_run
    assert cost["requests"]["kit_legend"] == 1
    assert cost["requests"]["people"] == cost["requests"]["landmarks"] == len(KEYFRAMES)
    assert sum(cost["requests"].values()) == len(backend.requests)
    paid_requests = [request.cache_key() for request in backend.requests]
    assert len(set(paid_requests)) == len(paid_requests)


def test_stored_answers_and_new_answers_count_together_against_the_limit_of_a_run(
    tmp_path: Path, contract_scene: SyntheticScene
) -> None:
    people_cost = fake_cost("people")
    # The limit of the run is 1.3 times its estimate: the cost of 2.6 people answers.
    meter = cli.start_meter(tmp_path, BUDGET, "run-1", 2.0 * people_cost)
    request = steps.build_people_request(
        contract_scene.image(0), contract_scene.kit_legend(), clip_id=CLIP_ID, frame=0, settings=StepSettings()
    )
    answer = FakeBackend(contract_scene).generate(request)

    meter.on_stored_answer(answer)
    meter.on_stored_answer(answer)
    meter.check_run_limit()
    # A new answer after two stored answers: three answers cost more than the limit.
    with pytest.raises(BudgetExceeded, match="used stored answers that cost") as stop:
        meter.on_new_answer(request, answer)
    # The meter names the limit with the words of the budget guard.
    assert f"together more than its limit of {2.6 * people_cost:.3f} USD (1.3 times its estimate)." in str(stop.value)

    report = meter.report()
    assert (report["stored_answers"], report["new_answers"]) == (2, 1)
    assert report["stored_usd"] == pytest.approx(2.0 * people_cost)
    assert report["measured_usd"] == pytest.approx(people_cost)
    with pytest.raises(BudgetExceeded):
        meter.check_run_limit()
    # The ledger holds the paid answer only.
    assert len(read_ledger(tmp_path)) == 1


def test_run_with_a_spend_limit_stops_above_that_limit_and_not_above_its_estimate(
    tmp_path: Path, contract_scene: SyntheticScene, capsys: pytest.CaptureFixture[str]
) -> None:
    """The spend limit holds in the budget guard and in the meter: new answers and stored answers count."""
    people_cost = fake_cost("people")
    # 1.3 times the estimate is the cost of 2.6 people answers. The spend limit is the cost of 4.5.
    meter = cli.start_meter(tmp_path, BUDGET, "run-1", 2.0 * people_cost, spend_limit_usd=4.5 * people_cost)
    budget_line = f"this run stops above {4.5 * people_cost:.3f} USD (its spend limit in the gate file)"
    assert budget_line in capsys.readouterr().out
    request = steps.build_people_request(
        contract_scene.image(0), contract_scene.kit_legend(), clip_id=CLIP_ID, frame=0, settings=StepSettings()
    )
    answer = FakeBackend(contract_scene).generate(request)

    for _ in range(4):
        meter.on_new_answer(request, answer)
    with pytest.raises(BudgetExceeded, match="more than its spend limit of"):
        meter.on_new_answer(request, answer)
    assert meter.report()["new_answers"] == 5

    # Stored answers count against the spend limit too: four of them, then one new answer.
    other = cli.start_meter(tmp_path / "other", BUDGET, "run-2", 2.0 * people_cost, spend_limit_usd=4.5 * people_cost)
    for _ in range(4):
        other.on_stored_answer(answer)
    other.check_run_limit()
    with pytest.raises(BudgetExceeded, match="used stored answers that cost") as stop:
        other.on_new_answer(request, answer)
    named = f"together more than its limit of {4.5 * people_cost:.3f} USD (its spend limit in the gate file)."
    assert named in str(stop.value)


def test_run_does_not_start_when_its_estimate_is_above_its_spend_limit(tmp_path: Path) -> None:
    refusal = "the estimate of this run is 0.600 USD, more than its spend limit of 0.500 USD, so the run could not"
    with pytest.raises(BudgetExceeded, match=refusal):
        cli.start_meter(tmp_path, BUDGET, "run-1", 0.6, spend_limit_usd=0.5)
    # An estimate at the spend limit starts.
    assert cli.start_meter(tmp_path, BUDGET, "run-2", 0.5, spend_limit_usd=0.5).run_id == "run-2"
    # Without a spend limit the estimate alone gives the limit of the run.
    assert cli.start_meter(tmp_path, BUDGET, "run-3", 0.6).run_id == "run-3"


def test_start_over_budget_does_not_start_a_run_whose_estimate_is_above_its_spend_limit(tmp_path: Path) -> None:
    """The option lifts the check against the hard stop only."""
    with pytest.raises(BudgetExceeded, match=r"is 0\.600 USD, more than its spend limit of 0\.500 USD"):
        cli.start_meter(tmp_path, BUDGET, "run-1", 0.6, spend_limit_usd=0.5, over_budget=True)
    # Control: an estimate at the spend limit starts with the option.
    assert cli.start_meter(tmp_path, BUDGET, "run-2", 0.5, spend_limit_usd=0.5, over_budget=True).run_id == "run-2"


def test_estimate_above_the_spend_limit_is_the_named_reason_when_it_also_does_not_fit_under_the_hard_stop(
    tmp_path: Path,
) -> None:
    """The refusal of the hard stop names the option --start-over-budget, which cannot help such a run."""
    cache = tmp_path / "cache"
    write_earlier_spend(cache, 1.0)
    budget = cli.Budget(planned_usd=1.0, hard_stop_usd=2.0, run_factor=1.3, max_usd_per_10_min=8.0)

    # 1 USD in the ledger and an estimate of 1.5 USD pass the hard stop of 2 USD, and 1.5 USD is above the spend limit.
    with pytest.raises(BudgetExceeded) as refusal:
        cli.start_meter(cache, budget, "run-1", 1.5, spend_limit_usd=1.2)

    assert "the estimate of this run is 1.500 USD, more than its spend limit of 1.200 USD" in str(refusal.value)
    assert "--start-over-budget" not in str(refusal.value)


def test_ledger_at_the_hard_stop_is_the_named_reason_before_an_estimate_above_the_spend_limit(tmp_path: Path) -> None:
    """The advice to plan less work is wrong when the ledger allows no paid work at all."""
    cache = tmp_path / "cache"
    write_earlier_spend(cache, 2.0)
    budget = cli.Budget(planned_usd=1.0, hard_stop_usd=2.0, run_factor=1.3, max_usd_per_10_min=8.0)

    with pytest.raises(BudgetExceeded, match=r"hard stop: the ledger total is 2\.00 USD"):
        cli.start_meter(cache, budget, "run-1", 1.5, spend_limit_usd=1.2)


def test_run_with_a_spend_limit_starts_when_the_ledger_holds_more_than_that_limit(tmp_path: Path) -> None:
    """The spend limit is the limit of one run. The spend of earlier runs does not count against it."""
    cache = tmp_path / "cache"
    write_earlier_spend(cache, 1.0)

    assert cli.start_meter(cache, BUDGET, "run-1", 0.4, spend_limit_usd=0.5).run_id == "run-1"


def test_start_meter_refuses_a_spend_limit_of_zero(tmp_path: Path) -> None:
    """A spend limit of 0 is a wrong value. It is not read as "no spend limit", which gives the 1.3 times rule."""
    with pytest.raises(ValueError, match="spend limit of a run must be a number above 0"):
        cli.start_meter(tmp_path, BUDGET, "run-1", 0.5, spend_limit_usd=0.0)


def test_stored_answers_count_against_a_spend_limit_that_is_below_1_3_times_the_estimate(
    tmp_path: Path, contract_scene: SyntheticScene
) -> None:
    people_cost = fake_cost("people")
    # 1.3 times the estimate is the cost of 5.2 people answers. The spend limit is the cost of 4.5.
    meter = cli.start_meter(tmp_path, BUDGET, "run-1", 4.0 * people_cost, spend_limit_usd=4.5 * people_cost)
    request = steps.build_people_request(
        contract_scene.image(0), contract_scene.kit_legend(), clip_id=CLIP_ID, frame=0, settings=StepSettings()
    )
    answer = FakeBackend(contract_scene).generate(request)

    for _ in range(4):
        meter.on_stored_answer(answer)
    meter.check_run_limit()
    with pytest.raises(BudgetExceeded, match="used stored answers that cost"):
        meter.on_new_answer(request, answer)


def test_hard_stop_refusal_and_start_over_budget_work_for_a_run_with_a_spend_limit(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    write_earlier_spend(cache, 1.0)
    budget = cli.Budget(planned_usd=1.0, hard_stop_usd=2.0, run_factor=1.3, max_usd_per_10_min=8.0)

    # 1 USD in the ledger and an estimate of 1.25 USD pass the hard stop of 2 USD. The estimate is below its limit.
    with pytest.raises(BudgetExceeded, match="together more than the hard stop"):
        cli.start_meter(cache, budget, "run-1", 1.25, spend_limit_usd=1.5)
    assert cli.start_meter(cache, budget, "run-2", 1.25, spend_limit_usd=1.5, over_budget=True).run_id == "run-2"


def test_stored_answer_costs_what_it_cost_on_its_day(tmp_path: Path, contract_scene: SyntheticScene) -> None:
    """An answer of November 2026 was paid at the price before the price change of 1 December 2026."""
    meter = cli.start_meter(tmp_path, BUDGET, "run-1", 1.0)
    request = steps.build_people_request(
        contract_scene.image(0), contract_scene.kit_legend(), clip_id=CLIP_ID, frame=0, settings=StepSettings()
    )
    answer = FakeBackend(contract_scene).generate(request)
    usage = STEP_USAGE["people"]

    meter.on_stored_answer(answer.model_copy(update={"created_at": "2026-11-20T10:00:00+00:00"}))
    meter.on_stored_answer(
        answer.model_copy(update={"created_at": "2026-12-02T10:00:00+00:00", "service_tier": "flex"})
    )

    november = (usage.input_tokens * 2.0 + (usage.output_tokens + usage.thought_tokens) * 10.0) / 1e6
    december_flex = 0.5 * (usage.input_tokens * 4.0 + (usage.output_tokens + usage.thought_tokens) * 20.0) / 1e6
    assert meter.report()["stored_usd"] == pytest.approx(november + december_flex)


def test_answer_is_counted_also_when_its_ledger_line_cannot_be_written(
    tmp_path: Path, contract_scene: SyntheticScene
) -> None:
    """The answer is paid. The printed measured cost must hold it, also when the disk is full."""

    class FullDiskLedger(Ledger):
        """A ledger that cannot take a line."""

        def record(self, run_id: str, request: VLMRequest, response: VLMResponse) -> float:
            raise OSError(errno.ENOSPC, "No space left on device")

    ledger = FullDiskLedger(tmp_path / "ledger.jsonl")
    guard = BudgetGuard(ledger)
    guard.start_run("run-1", 1.0)
    meter = cli.CostMeter(ledger, guard, "run-1", run_limit_usd=1.3, limit_named="1.3 times its estimate")
    request = steps.build_people_request(
        contract_scene.image(0), contract_scene.kit_legend(), clip_id=CLIP_ID, frame=0, settings=StepSettings()
    )

    with pytest.raises(OSError, match="No space left on device"):
        meter.on_new_answer(request, FakeBackend(contract_scene).generate(request))

    report = meter.report()
    assert report["new_answers"] == 1
    assert report["measured_usd"] == pytest.approx(fake_cost("people"))
    assert report["steps"]["people"]["input_tokens"] == STEP_USAGE["people"].input_tokens


def test_run_follows_its_settings_file(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scene, root = clip_on_disk
    clip_dir = root / "SoccerNetGS-valid" / CLIP_ID
    settings = tmp_path / "settings.yaml"
    settings.write_text("keyframe_interval_s: 1.0\nrepeat: 1\n", encoding="utf-8")
    use_fake_model(monkeypatch, scene)

    labels = str(clip_dir / LABEL_NAME)
    code = cli.main(run_arguments(clip_dir, tmp_path, "--labels", labels, "--config", str(settings)))

    assert code == 0
    result = read_json(tmp_path / "runs" / CLIP_ID / "result.json")
    # One keyframe per second, plus the last frame: the frames 0, 25 and 49.
    assert result["provenance"]["cost"]["steps"]["people"]["answers"] == 3
    assert result["provenance"]["config"]["keyframe_interval_s"] == 1.0
    # A repeat run says what its repeat index means for the cost, and its ledger lines hold the index.
    assert "repeat index 1: the stored answers of other repeat indexes are not used" in capsys.readouterr().out
    assert {line["repeat"] for line in read_ledger(tmp_path / "cache")} == {1}


@pytest.mark.parametrize(("repeat", "said"), [(0, False), (1, True), (3, True)])
def test_session_says_what_a_repeat_index_above_zero_means_for_the_cost(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], repeat: int, said: bool
) -> None:
    plan = [("people", StepSettings(), 1)]

    cli.open_session("run", "clip", plan, cache_dir=tmp_path / "cache", replay=True, repeat=repeat)

    output = capsys.readouterr().out
    assert (f"repeat index {repeat}:" in output) is said
    assert ("pays for every call again" in output) is said


def test_repeat_index_and_the_choice_about_other_images_reach_the_session_of_run_and_probe(
    clip_on_disk: tuple[SyntheticScene, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root = clip_on_disk
    seen: list[tuple[str, int, bool]] = []

    def stop_here(command: str, label: str, plan: object, **options: Any) -> None:
        seen.append((command, options["repeat"], options["allow_other_images"]))
        raise cli.CliError("the test stops the command here")

    monkeypatch.setattr(cli, "open_session", stop_here)
    settings = tmp_path / "settings.yaml"
    settings.write_text("repeat: 2\n", encoding="utf-8")
    experiment = tmp_path / "pitch_probe_again.yaml"
    again = (EXPERIMENTS / "pitch_probe.yaml").read_text(encoding="utf-8").replace("repeat: 0", "repeat: 3")
    experiment.write_text(again, encoding="utf-8")
    manifest = tmp_path / "splits.json"
    manifest.write_text(json.dumps({"seed": 7, "valid": {"probe": [CLIP_ID]}}), encoding="utf-8")
    clip_dir = root / "SoccerNetGS-valid" / CLIP_ID
    run = run_arguments(clip_dir, tmp_path, "--fps", "25", "--config", str(settings), "--allow-other-images")
    probe = ["probe", "pitch", "--experiment", str(experiment), "--root", str(root), "--manifest", str(manifest)]
    probe += ["--out", str(tmp_path / "pitch.json"), "--cache", str(tmp_path / "cache"), "--allow-dirty"]

    assert cli.main(run) == 2
    assert cli.main(probe) == 2

    assert seen == [("run", 2, True), ("probe", 3, False)]


def test_run_stops_at_stored_answers_of_other_pixels_and_goes_on_with_the_option(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A clip that was replaced under the same name must not get the answers of the old pixels without a word."""
    scene, root = clip_on_disk
    use_fake_model(monkeypatch, scene)
    assert cli.main(run_arguments(root / "SoccerNetGS-valid" / CLIP_ID, tmp_path, "--fps", "25")) == 0
    # Other footage with the clip id of the first clip: 25 frames, so the keyframes 0, 12 and 24.
    other_scene = SyntheticScene(n_frames=25, seed=5)
    new_clip = tmp_path / "new-footage" / CLIP_ID
    (new_clip / "img1").mkdir(parents=True)
    for frame in range(25):
        assert cv2.imwrite(str(new_clip / "img1" / f"{frame + 1:06d}.png"), other_scene.image(frame))
    backend = use_fake_model(monkeypatch, other_scene)
    arguments = run_arguments(new_clip, tmp_path, "--fps", "25")
    capsys.readouterr()

    assert cli.main(arguments) == 2

    captured = capsys.readouterr()
    assert backend.requests == []
    assert "measured cost: 0.000 USD for 0 new answers" in captured.out
    error = captured.err.splitlines()[-1]
    assert error.startswith(f"error: the stored answer of the kit_legend request of clip {CLIP_ID}")
    assert "other image bytes" in error
    assert "allow_other_images" in error and "--allow-other-images" in error

    with caplog.at_level(logging.WARNING, logger="pitchvlm.vlm.cache"):
        assert cli.main([*arguments, "--allow-other-images"]) == 0

    assert any("Other image bytes are allowed" in record.getMessage() for record in caplog.records)
    # The stored answers of the keyframes are used. Only number requests are new: their crops are others.
    assert {request.meta.step for request in backend.requests} == {"number"}


def test_predictions_follow_the_scored_frames_and_the_jersey_type_of_the_labels(
    clip_on_disk: tuple[SyntheticScene, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Other labels for the same frames: whole-number shirt numbers, and one frame without camera labels."""
    scene, root = clip_on_disk
    clip_dir = root / "SoccerNetGS-valid" / CLIP_ID
    labels = read_json(clip_dir / LABEL_NAME)
    for row in labels["annotations"]:
        if row.get("attributes", {}).get("jersey") is not None:
            row["attributes"]["jersey"] = int(row["attributes"]["jersey"])
    not_scored = sorted(labels["images"], key=lambda image: image["file_name"])[10]
    not_scored["has_labeled_camera"] = False
    label_file = tmp_path / "other-labels" / LABEL_NAME
    label_file.parent.mkdir()
    label_file.write_text(json.dumps(labels), encoding="utf-8")
    use_fake_model(monkeypatch, scene)

    code = cli.main(run_arguments(clip_dir, tmp_path, "--labels", str(label_file)))

    assert code == 0  # The format checker finds no problem.
    path = tmp_path / "runs" / "predictions" / "SoccerNetGS-valid" / "pitchvlm" / f"{CLIP_ID}.json"
    predictions = read_json(path)["predictions"]
    image_ids = {item["image_id"] for item in predictions}
    assert not_scored["image_id"] not in image_ids
    assert len(image_ids) == N_FRAMES - 1
    jerseys = [item["attributes"]["jersey"] for item in predictions if item["attributes"]["jersey"] is not None]
    assert len(jerseys) > 100
    assert all(type(jersey) is int for jersey in jerseys)


def test_run_ends_with_code_1_when_the_checker_reports_a_problem(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scene, root = clip_on_disk
    clip_dir = root / "SoccerNetGS-valid" / CLIP_ID
    use_fake_model(monkeypatch, scene)
    monkeypatch.setattr(cli, "check_predictions", lambda payload, **options: ["id: a made-up problem"])

    assert cli.main(run_arguments(clip_dir, tmp_path, "--labels", str(clip_dir / LABEL_NAME))) == 1

    assert "a made-up problem" in capsys.readouterr().out
    # The files are written: a failed check is not an error. The result file holds the problem.
    result = read_json(tmp_path / "runs" / CLIP_ID / "result.json")
    assert result["prediction_file"]["problems"] == ["id: a made-up problem"]


def test_run_counts_the_scored_frames_without_a_prediction_and_does_not_fail(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A scored frame without any person is no format problem: the scorer counts its people as missed."""
    scene, root = clip_on_disk
    clip_dir = root / "SoccerNetGS-valid" / CLIP_ID
    run_clip = cli.run_clip

    def with_two_empty_frames(*arguments: Any, **options: Any) -> ClipResult:
        result = run_clip(*arguments, **options)
        frames = tuple(
            state.model_copy(update={"players": ()}) if state.frame in (3, 4) else state for state in result.frames
        )
        return result.model_copy(update={"frames": frames})

    use_fake_model(monkeypatch, scene)
    monkeypatch.setattr(cli, "run_clip", with_two_empty_frames)

    assert cli.main(run_arguments(clip_dir, tmp_path, "--labels", str(clip_dir / LABEL_NAME))) == 0

    output = capsys.readouterr().out
    assert "format checker: the prediction file is valid" in output
    assert f"scored frames without a prediction: 2 of {N_FRAMES}" in output
    assert read_json(tmp_path / "runs" / CLIP_ID / "result.json")["prediction_file"] == {
        "problems": [],
        "scored_frames": N_FRAMES,
        "scored_frames_without_prediction": 2,
    }


@pytest.mark.parametrize("labelled", [True, False])
def test_run_ends_with_code_1_when_the_clip_gives_no_result(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    labelled: bool,
) -> None:
    """No landmarks answer names a landmark: no keyframe has a pitch mapping, so no person gets a position.

    A run without attention must not count such a clip as done. The files are still written: the
    scorer reads the empty prediction file and counts every labelled person of the clip as missed.
    """
    scene, root = clip_on_disk
    clip_dir = root / "SoccerNetGS-valid" / CLIP_ID
    backend = EditedAnswersBackend(FakeBackend(scene), without_landmarks)
    monkeypatch.setattr(cli, "render_clip", lambda source, result, out_path: out_path)
    monkeypatch.setattr(cli, "gemini_backend", lambda max_usd_per_10_min: backend)
    source = ["--labels", str(clip_dir / LABEL_NAME)] if labelled else ["--fps", "25"]

    code = cli.main(run_arguments(clip_dir, tmp_path, *source))

    assert code == 1
    output = capsys.readouterr().out
    result = read_json(tmp_path / "runs" / CLIP_ID / "result.json")
    assert all(state["players"] == [] for state in result["result"]["frames"])
    # No shirt number is paid for a clip without a pitch mapping: no track of it gets a state.
    assert result["provenance"]["cost"]["requests"] == {"kit_legend": 1, "landmarks": 6, "people": 6}
    prediction_file = tmp_path / "runs" / "predictions" / "SoccerNetGS-valid" / "pitchvlm" / f"{CLIP_ID}.json"
    if labelled:
        assert "format checker: the prediction file is valid" in output
        assert f"scored frames without a prediction: {N_FRAMES} of {N_FRAMES}" in output
        assert "failed: no scored frame has a prediction, so the clip gives no result." in output
        assert read_json(prediction_file) == {"predictions": []}
        assert result["prediction_file"] == {
            "problems": [],
            "scored_frames": N_FRAMES,
            "scored_frames_without_prediction": N_FRAMES,
        }
    else:
        assert "failed: no frame of the result holds a person, so the clip gives no result." in output
        assert not prediction_file.exists()
        assert result["prediction_file"] is None
    assert "The usual cause is a clip without a pitch mapping" in output


def test_clip_gives_no_result_when_no_scored_frame_or_no_frame_at_all_holds_a_person(
    contract_scene: SyntheticScene,
) -> None:
    with_people = [contract_scene.frame_state(frame) for frame in (0, 1)]
    empty = [state.model_copy(update={"players": ()}) for state in with_people]
    assert with_people[0].players

    # Without labels the frames of the result decide.
    assert cli.empty_result_reason(with_people, None) is None
    assert cli.empty_result_reason([empty[0], with_people[1]], None) is None
    assert cli.empty_result_reason(empty, None) == "no frame of the result holds a person"
    # With labels the scored frames decide: (scored frames without a prediction, scored frames).
    assert cli.empty_result_reason(with_people, (50, 50)) == "no scored frame has a prediction"
    assert cli.empty_result_reason(with_people, (49, 50)) is None
    assert cli.empty_result_reason(with_people, (0, 50)) is None
    # Labels without a scored frame say nothing, so the frames of the result decide again.
    assert cli.empty_result_reason(with_people, (0, 0)) is None
    assert cli.empty_result_reason(empty, (0, 0)) == "no frame of the result holds a person"


def test_run_writes_its_prediction_file_below_the_split_and_the_tracker_of_the_command(
    clip_on_disk: tuple[SyntheticScene, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scorer looks for the file of a test clip below SoccerNetGS-test, so the path must follow --split."""
    scene, root = clip_on_disk
    clip_dir = root / "SoccerNetGS-valid" / CLIP_ID
    use_fake_model(monkeypatch, scene)
    arguments = run_arguments(clip_dir, tmp_path, "--labels", str(clip_dir / LABEL_NAME))

    assert cli.main([*arguments, "--split", "test", "--tracker", "frozen"]) == 0

    predictions = tmp_path / "runs" / "predictions"
    assert [path.relative_to(predictions).as_posix() for path in predictions.rglob("*.json")] == [
        f"SoccerNetGS-test/frozen/{CLIP_ID}.json"
    ]
    assert len(read_json(predictions / "SoccerNetGS-test" / "frozen" / f"{CLIP_ID}.json")["predictions"]) > 300


def test_paid_run_over_a_warm_cache_finishes_although_its_stored_answers_cost_more_than_its_limit(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The limit of a run stops the next payment. It must not stop a request whose answer is stored already."""
    scene, root = clip_on_disk
    arguments = run_arguments(root / "SoccerNetGS-valid" / CLIP_ID, tmp_path, "--fps", "25")
    backend = use_fake_model(monkeypatch, scene)
    assert cli.main(arguments) == 0
    paid = len(backend.requests)
    first = read_json(tmp_path / "runs" / CLIP_ID / "result.json")
    capsys.readouterr()

    # The same run again, with an estimate of a tenth of the tokens: every stored answer alone is above the limit.
    estimates_of_a_share(monkeypatch, 0.1)
    assert cli.main(arguments) == 0

    second = read_json(tmp_path / "runs" / CLIP_ID / "result.json")
    cost = second["provenance"]["cost"]
    assert len(backend.requests) == paid
    assert (cost["new_answers"], cost["stored_answers"]) == (0, paid)
    assert cost["stored_usd"] > 1.3 * cost["estimate_usd"]
    assert second["result"]["tracks"] == first["result"]["tracks"]
    assert len(read_ledger(tmp_path / "cache")) == paid
    assert f"stored answers used: {paid}" in capsys.readouterr().out


def test_run_without_labels_has_no_prediction_file_in_its_result(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scene, root = clip_on_disk
    use_fake_model(monkeypatch, scene)

    assert cli.main(run_arguments(root / "SoccerNetGS-valid" / CLIP_ID, tmp_path, "--fps", "25")) == 0

    assert read_json(tmp_path / "runs" / CLIP_ID / "result.json")["prediction_file"] is None
    assert "scored frames without a prediction" not in capsys.readouterr().out


def test_clip_needs_as_many_keyframes_as_a_track_needs_observations() -> None:
    cli.require_enough_keyframes(2, PipelineConfig())
    with pytest.raises(cli.CliError, match="1 keyframes, but a track needs 2 observations"):
        cli.require_enough_keyframes(1, PipelineConfig())
    three = PipelineConfig(tracking=TrackingConfig(min_observations=3))
    cli.require_enough_keyframes(3, three)
    with pytest.raises(cli.CliError, match="2 keyframes, but a track needs 3 observations"):
        cli.require_enough_keyframes(2, three)


def test_clip_that_is_too_short_for_a_track_makes_no_model_call(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A clip of one frame has one keyframe. A track needs two observations, so the result would be empty."""
    scene, _ = clip_on_disk
    backend = use_fake_model(monkeypatch, scene)
    one_frame = tmp_path / "one-frame"
    one_frame.mkdir()
    assert cv2.imwrite(str(one_frame / "000001.png"), scene.image(0))

    code = cli.main(run_arguments(one_frame, tmp_path, "--fps", "25"))

    assert code == 2
    captured = capsys.readouterr()
    assert "the clip has 1 keyframes, but a track needs 2 observations" in captured.err
    assert "cost estimate" not in captured.out
    assert backend.requests == []
    assert not (tmp_path / "cache").exists()
    # The demo command refuses such a clip too.
    assert cli.main(["demo", "--out", str(tmp_path / "demo"), "--frames", "1"]) == 2
    assert "too short to give any track" in capsys.readouterr().err
    assert not (tmp_path / "demo").exists()


def test_paid_command_needs_the_results_work_tree_for_its_default_cache(
    clip_on_disk: tuple[SyntheticScene, Path],
    results_tree: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without --cache the answers go to the work tree of the results branch, next to the repository."""
    scene, root = clip_on_disk
    clip_dir = root / "SoccerNetGS-valid" / CLIP_ID
    backend = use_fake_model(monkeypatch, scene)
    arguments = ["run", "--clip", str(clip_dir), "--fps", "25", "--out", str(tmp_path / "runs"), "--allow-dirty"]

    # A repository without a results work tree next to it: the command must not make a plain folder there.
    lonely = tmp_path / "elsewhere" / "repo"
    (lonely / "configs").mkdir(parents=True)
    shutil.copy(GATES, lonely / "configs" / "gates.yaml")
    monkeypatch.setattr(cli, "repository_root", lambda: lonely)
    assert cli.main(arguments) == 2
    error = capsys.readouterr().err
    assert "is not inside a git work tree" in error
    assert "git worktree add -b results ../pitchvlm-results" in error
    assert "move it away first" in error
    assert backend.requests == []
    assert not (tmp_path / "elsewhere" / "pitchvlm-results").exists()

    # The fixture made the work tree of the branch results in tmp_path / "pitchvlm-results".
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    shutil.copy(GATES, repo / "configs" / "gates.yaml")
    monkeypatch.setattr(cli, "repository_root", lambda: repo)
    assert cli.main(arguments) == 0
    cache = results_tree / "results" / "cache"
    assert len(read_ledger(cache)) == len(backend.requests) == len(list(cache.glob(f"{CLIP_ID}/*/*.json")))

    # A work tree on another branch is refused: result files must never go to the code branch.
    git(results_tree, "symbolic-ref", "HEAD", "refs/heads/main")
    assert cli.main(arguments) == 2
    assert "not on 'results'" in capsys.readouterr().err


@pytest.mark.parametrize(("name", "cause"), [("SIGTERM", "SIGTERM"), ("SIGHUP", "SIGHUP"), ("SIGINT", "Ctrl-C")])
def test_stop_signal_ends_a_command_like_an_interruption(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], name: str, cause: str
) -> None:
    """A plain kill sends SIGTERM, a closed terminal SIGHUP and Ctrl-C SIGINT. The command ends with exit code 2."""
    number = getattr(signal, name)
    seen_by_the_command: list[bool] = []

    def handler_of_the_test(received: int, frame: object) -> None:
        raise AssertionError(f"{name} reached the handler that was set before the command ran")

    def wait_for_the_signal(args: argparse.Namespace) -> int:
        seen_by_the_command.append(cli.STOP_REQUESTED.is_set())
        os.kill(os.getpid(), number)
        for _ in range(500):
            time.sleep(0.01)
        return 0

    monkeypatch.setattr(cli, "_cmd_sync", wait_for_the_signal)
    before = signal.signal(number, handler_of_the_test)
    try:
        code = cli.main(["sync", "--results", "results"])
        # The command handles the signal only while it runs: the earlier handler is back.
        assert signal.getsignal(number) is handler_of_the_test
    finally:
        signal.signal(number, before)

    assert code == 2
    stderr = capsys.readouterr().err
    assert "Traceback" not in stderr
    assert stderr.splitlines()[-1] == (
        f"error: the command was interrupted ({cause}). Every answer that was received is stored and "
        "recorded in the cost ledger."
    )
    # The signal set the event that tells the model backend to send nothing more.
    assert cli.STOP_REQUESTED.is_set()
    # The next command of the same process starts without a stop request.
    monkeypatch.setattr(cli, "_cmd_sync", lambda args: int(cli.STOP_REQUESTED.is_set()))
    assert cli.main(["sync", "--results", "results"]) == 0
    assert seen_by_the_command == [False]


def test_answers_in_flight_are_stored_and_recorded_when_a_run_is_stopped_by_a_signal(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The process gets SIGTERM while the fifth model request is in flight."""
    scene, root = clip_on_disk
    clip_dir = root / "SoccerNetGS-valid" / CLIP_ID
    fake = FakeBackend(scene)
    lock = threading.Lock()
    answered: list[str] = []

    class StoppedBackend:
        """The fake model. Every request takes 200 milliseconds, and the fifth one sends SIGTERM to this process."""

        def __init__(self) -> None:
            self.started = 0

        def generate(self, request: VLMRequest) -> VLMResponse:
            with lock:
                self.started += 1
                number = self.started
            if number == 5:
                os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.2)
            response = fake.generate(request)
            with lock:
                answered.append(request.cache_key())
            return response

    monkeypatch.setattr(cli, "render_clip", lambda source, result, out_path: out_path)
    monkeypatch.setattr(cli, "gemini_backend", lambda max_usd_per_10_min: StoppedBackend())

    code = cli.main(run_arguments(clip_dir, tmp_path, "--fps", "25"))

    assert code == 2
    captured = capsys.readouterr()
    assert captured.err.splitlines()[-1].startswith("error: the command was interrupted (SIGTERM).")
    assert "measured cost" in captured.out
    # The kit legend request and the four requests that were in flight finished: no answer was thrown away.
    # How many more requests started before the signal was handled depends on the load of the machine, so
    # the count has no upper limit below the 13 requests of this stage. Every answer has a cassette and a
    # ledger line.
    cache = tmp_path / "cache"
    assert 5 <= len(answered) <= 1 + 2 * len(KEYFRAMES)
    assert sorted(line["request"] for line in read_ledger(cache)) == sorted(answered)
    assert sorted(path.stem for path in cache.glob(f"{CLIP_ID}/*/*.json")) == sorted(answered)
    assert not (tmp_path / "runs").exists()


def test_measured_cost_of_a_run_that_is_stopped_two_times_holds_every_answer_in_flight(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A second stop signal comes while the command waits for the model requests in flight.

    Every request takes 300 milliseconds. The third request, which runs in the thread pool, sends
    SIGTERM when it starts. It sends SIGINT 100 milliseconds after the pool began to wait for its
    requests, and it is answered 200 milliseconds after that. The cost that the command prints at
    its end must hold every answer that has a ledger line.
    """
    scene, root = clip_on_disk
    fake = FakeBackend(scene)
    lock = threading.Lock()
    pool_waits = threading.Event()
    pool_threads: list[threading.Thread] = []
    answered: list[str] = []

    class WatchedPool(pipeline.ThreadPoolExecutor):
        """A thread pool that says when it begins to wait for its threads."""

        def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
            pool_waits.set()
            super().shutdown(wait, cancel_futures=cancel_futures)

    class StoppedTwiceBackend:
        """The fake model. Its third request sends the two stop signals while it is in flight."""

        def __init__(self) -> None:
            self.started = 0

        def generate(self, request: VLMRequest) -> VLMResponse:
            with lock:
                self.started += 1
                number = self.started
                if threading.current_thread() is not threading.main_thread():
                    pool_threads.append(threading.current_thread())
            if number == 3:
                os.kill(os.getpid(), signal.SIGTERM)
                assert pool_waits.wait(timeout=60.0)
                time.sleep(0.1)
                os.kill(os.getpid(), signal.SIGINT)
                time.sleep(0.2)
            else:
                time.sleep(0.3)
            response = fake.generate(request)
            with lock:
                answered.append(request.cache_key())
            return response

    monkeypatch.setattr(pipeline, "ThreadPoolExecutor", WatchedPool)
    monkeypatch.setattr(cli, "render_clip", lambda source, result, out_path: out_path)
    monkeypatch.setattr(cli, "gemini_backend", lambda max_usd_per_10_min: StoppedTwiceBackend())

    with caplog.at_level(logging.WARNING, logger="pitchvlm.pipeline"):
        code = cli.main(run_arguments(root / "SoccerNetGS-valid" / CLIP_ID, tmp_path, "--fps", "25"))
    captured = capsys.readouterr()
    # A request that the command did not wait for would get its ledger line after the command ended.
    for thread in pool_threads:
        thread.join(timeout=60.0)

    assert code == 2
    assert captured.err.splitlines()[-1].startswith("error: the command was interrupted (")
    # The second signal came during the wait for the requests in flight, and the wait went on.
    assert "a stop signal came while" in caplog.text
    (cost_line,) = [line for line in captured.out.splitlines() if line.startswith("measured cost: ")]
    printed = re.search(r" for (\d+) new answers ", cost_line)
    assert printed is not None
    ledger = read_ledger(tmp_path / "cache")
    # The kit legend answer, the second request and the request that sent the signals are answered. How
    # many more requests started before the first signal was handled depends on the load of the machine.
    assert 3 <= len(ledger) <= 1 + 2 * len(KEYFRAMES)
    assert int(printed.group(1)) == len(ledger)
    assert sorted(line["request"] for line in ledger) == sorted(answered)


def test_kit_legend_call_in_flight_is_stored_and_recorded_when_a_run_is_stopped_by_a_signal(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The kit legend request runs in the main thread, where a stop signal is raised. Its call must still finish."""
    scene, root = clip_on_disk
    backend = SignallingBackend(FakeBackend(scene), at_request=1)
    monkeypatch.setattr(cli, "render_clip", lambda source, result, out_path: out_path)
    monkeypatch.setattr(cli, "gemini_backend", lambda max_usd_per_10_min: backend)

    code = cli.main(run_arguments(root / "SoccerNetGS-valid" / CLIP_ID, tmp_path, "--fps", "25"))

    assert code == 2
    captured = capsys.readouterr()
    assert captured.err.splitlines()[-1] == (
        "error: the command was interrupted (SIGTERM). Every answer that was received is stored and "
        "recorded in the cost ledger."
    )
    assert "measured cost: " in captured.out
    assert "for 1 new answers" in captured.out
    # The call in flight finished: the answer of the kit legend request has a cassette and a ledger line.
    # No further request was sent.
    cache = tmp_path / "cache"
    assert (backend.started, len(backend.answered)) == (1, 1)
    assert [line["request"] for line in read_ledger(cache)] == backend.answered
    assert [path.stem for path in cache.glob(f"{CLIP_ID}/*/*.json")] == backend.answered
    assert not (tmp_path / "runs").exists()


def test_call_in_flight_of_a_probe_is_stored_and_recorded_when_the_probe_is_stopped_by_a_signal(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A probe sends its requests one after the other in the main thread. SIGTERM comes during the third one."""
    scene, root = clip_on_disk
    manifest = tmp_path / "splits.json"
    manifest.write_text(json.dumps({"seed": 7, "valid": {"probe": [CLIP_ID]}}), encoding="utf-8")
    backend = SignallingBackend(FakeBackend(scene), at_request=3)
    monkeypatch.setattr(cli, "gemini_backend", lambda max_usd_per_10_min: backend)
    metrics, cache = tmp_path / "people_probe.json", tmp_path / "cache"
    arguments = ["probe", "people", "--experiment", str(EXPERIMENTS / "people_probe.yaml"), "--root", str(root)]
    arguments += ["--manifest", str(manifest), "--out", str(metrics), "--cache", str(cache), "--allow-dirty"]

    code = cli.main(arguments)

    assert code == 2
    captured = capsys.readouterr()
    assert captured.err.splitlines()[-1].startswith("error: the command was interrupted (SIGTERM).")
    # The kit legend answer, the first people answer and the people answer that was in flight are stored
    # and recorded. The probe sent nothing after the signal, and it wrote no metrics file.
    assert (backend.started, len(backend.answered)) == (3, 3)
    assert [line["request"] for line in read_ledger(cache)] == backend.answered
    assert sorted(path.stem for path in cache.glob(f"{CLIP_ID}/*/*.json")) == sorted(backend.answered)
    assert "for 3 new answers" in captured.out
    assert not metrics.exists()


def test_second_stop_signal_does_not_cut_a_call_in_flight_off(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An operator who presses Ctrl-C a second time must not lose an answer that is paid."""
    scene, root = clip_on_disk
    answered: list[str] = []

    class StoppedTwice:
        """The fake model. Its first request gets SIGTERM and then SIGHUP while it is in flight."""

        def generate(self, request: VLMRequest) -> VLMResponse:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.05)
            os.kill(os.getpid(), signal.SIGHUP)
            time.sleep(0.05)
            answered.append(request.cache_key())
            return FakeBackend(scene).generate(request)

    monkeypatch.setattr(cli, "render_clip", lambda source, result, out_path: out_path)
    monkeypatch.setattr(cli, "gemini_backend", lambda max_usd_per_10_min: StoppedTwice())

    code = cli.main(run_arguments(root / "SoccerNetGS-valid" / CLIP_ID, tmp_path, "--fps", "25"))

    # The command ends with the cause of the first signal, after the call in flight was recorded.
    assert code == 2
    assert capsys.readouterr().err.splitlines()[-1].startswith("error: the command was interrupted (SIGTERM).")
    assert len(answered) == 1
    assert [line["request"] for line in read_ledger(tmp_path / "cache")] == answered


def test_model_call_in_another_thread_is_not_held_by_the_main_thread_rule() -> None:
    """Only a call of the main thread needs the rule: a stop signal is raised there, and nowhere else."""
    threads: list[threading.Thread] = []
    answer = SimpleNamespace(text="an answer")

    def call() -> Any:
        threads.append(threading.current_thread())
        return answer

    assert cli.finish_in_flight(call) is answer
    worker = threading.Thread(target=lambda: cli.finish_in_flight(call))
    worker.start()
    worker.join()
    # Both calls ran in the thread that asked for them.
    assert threads == [threading.main_thread(), worker]
    # An error of a call goes to its caller as it is.
    error = BudgetExceeded("the run costs too much")

    def fail() -> Any:
        raise error

    with pytest.raises(BudgetExceeded) as excinfo:
        cli.finish_in_flight(fail)
    assert excinfo.value is error


@pytest.mark.parametrize(("name", "cause"), [("SIGINT", "Ctrl-C"), ("SIGTERM", "SIGTERM")])
def test_request_that_waits_for_a_new_attempt_is_not_sent_again_after_a_stop_signal(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    name: str,
    cause: str,
) -> None:
    """The real backend of the command line on a fake SDK client. One people call gets the status 429.

    The stop signal comes before that request starts to wait for its second attempt, a wait of 5 to
    10 seconds. The request must give up, and the command must not send it again.
    """
    scene, root = clip_on_disk
    truth = FakeBackend(scene)
    request_of_the_thread = threading.local()
    lock = threading.Lock()
    sdk_calls: list[str] = []
    refused: list[str] = []

    class BridgedBackend(GeminiBackend):
        """The real backend. It keeps the request of its thread, so that the fake SDK client can answer it."""

        def generate(self, request: VLMRequest) -> VLMResponse:
            request_of_the_thread.value = request
            return super().generate(request)

    def create(**kwargs: Any) -> SimpleNamespace:
        request = request_of_the_thread.value
        with lock:
            sdk_calls.append(request.cache_key())
            refuse = request.meta.step == "people" and not refused
            if refuse:
                refused.append(request.cache_key())
        if refuse:
            # The signal is sent now. The call ends when the command has seen it, so the wait for the
            # second attempt starts after the stop request, however slow the machine is.
            os.kill(os.getpid(), getattr(signal, name))
            assert cli.STOP_REQUESTED.wait(timeout=60.0)
            raise FakeApiError(429)
        answer = truth.generate(request)
        used = answer.usage
        usage = SimpleNamespace(
            total_input_tokens=used.input_tokens,
            total_output_tokens=used.output_tokens,
            total_thought_tokens=used.thought_tokens,
            total_tokens=used.input_tokens + used.output_tokens + used.thought_tokens,
        )
        return SimpleNamespace(status="completed", output_text=answer.text, service_tier=None, usage=usage)

    sdk_client = SimpleNamespace(interactions=SimpleNamespace(create=create))
    monkeypatch.setattr(client_module, "build_sdk_client", lambda api_key_env, timeout_s: sdk_client)
    monkeypatch.setattr(cli, "GeminiBackend", BridgedBackend)
    monkeypatch.setattr(cli, "render_clip", lambda source, result, out_path: out_path)

    with caplog.at_level(logging.WARNING, logger="pitchvlm.vlm.client"):
        code = cli.main(run_arguments(root / "SoccerNetGS-valid" / CLIP_ID, tmp_path, "--fps", "25"))

    assert code == 2
    assert capsys.readouterr().err.splitlines()[-1].startswith(f"error: the command was interrupted ({cause}).")
    # The refused request was sent one time. Its log line says that no further attempt was sent.
    assert sdk_calls.count(refused[0]) == 1
    stop_line = "a stop was requested after attempt 1 failed, so no further attempt was sent"
    assert any(stop_line in record.getMessage() for record in caplog.records)
    # Every answer that came back has a cassette and a ledger line.
    cache = tmp_path / "cache"
    answered = sorted(key for key in sdk_calls if key != refused[0])
    assert sorted(line["request"] for line in read_ledger(cache)) == answered
    assert sorted(path.stem for path in cache.glob(f"{CLIP_ID}/*/*.json")) == answered


def test_metered_backend_reports_answers_and_rejected_answers_one_time_each(
    tmp_path: Path, contract_scene: SyntheticScene
) -> None:
    meter = cli.start_meter(tmp_path, BUDGET, "run-1", 1.0)
    backend = cli.MeteredBackend(RejectedAnswerBackend(FakeBackend(contract_scene), "people"), meter)
    request = steps.build_people_request(
        contract_scene.image(0), contract_scene.kit_legend(), clip_id=CLIP_ID, frame=0, settings=StepSettings()
    )

    # The first answer is billed and then rejected: the error goes on, and its cost is recorded.
    with pytest.raises(BilledAnswerError, match="the people answer has no text"):
        backend.generate(request)
    assert backend.generate(request).text != ""

    lines = read_ledger(tmp_path)
    assert [(line["run_id"], line["step"], line["input_tokens"]) for line in lines] == [
        ("run-1", "people", STEP_USAGE["people"].input_tokens)
    ] * 2
    report = meter.report()
    assert report["new_answers"] == 2
    assert report["measured_usd"] == pytest.approx(sum(line["usd"] for line in lines))


def test_metered_backend_reports_nothing_for_a_call_that_was_not_billed(
    tmp_path: Path, contract_scene: SyntheticScene
) -> None:
    meter = cli.start_meter(tmp_path, BUDGET, "run-1", 1.0)
    backend = cli.MeteredBackend(FakeBackend(contract_scene), meter)
    request = steps.build_people_request(
        contract_scene.image(0), contract_scene.kit_legend(), clip_id=CLIP_ID, frame=0, settings=StepSettings()
    )
    # The fake backend has no answer for this step. Its error carries no cost record.
    unknown_step = request.model_copy(update={"meta": request.meta.model_copy(update={"step": "offside"})})

    with pytest.raises(BackendError, match="offside"):
        backend.generate(unknown_step)

    assert not (tmp_path / "ledger.jsonl").exists()
    assert meter.report()["new_answers"] == 0


def test_budget_stop_replaces_the_error_of_a_rejected_answer(tmp_path: Path, contract_scene: SyntheticScene) -> None:
    # The estimate of the run is far below the cost of one answer, so the first billed call reaches the limit.
    meter = cli.start_meter(tmp_path, BUDGET, "run-1", 1e-9)
    backend = cli.MeteredBackend(RejectedAnswerBackend(FakeBackend(contract_scene), "people"), meter)
    request = steps.build_people_request(
        contract_scene.image(0), contract_scene.kit_legend(), clip_id=CLIP_ID, frame=0, settings=StepSettings()
    )

    with pytest.raises(BudgetExceeded):
        backend.generate(request)

    # The rejected answer is in the ledger, and the budget guard saw it.
    assert len(read_ledger(tmp_path)) == 1


def test_http_libraries_below_the_sdk_write_no_request_line_into_the_log() -> None:
    """From version 3.0 the SDK sends its requests with httpx2, which writes one INFO line per request."""
    names = ("httpx2", "httpcore2", "httpx", "httpcore")
    loggers = [logging.getLogger(name) for name in names]
    levels_before = [logger.level for logger in loggers]
    try:
        for logger in loggers:
            logger.setLevel(logging.NOTSET)

        cli._configure_logging()

        assert cli.HTTP_LOGGERS == names
        for logger in loggers:
            assert not logger.isEnabledFor(logging.INFO)
            assert logger.isEnabledFor(logging.WARNING)
    finally:
        for logger, level in zip(loggers, levels_before, strict=True):
            logger.setLevel(level)


# --- Exit code of an error -------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "shows_traceback"),
    [
        # Errors whose class says that the cause lies outside the code: one line.
        (cli.CliError("the command line refuses"), False),
        (BackendError("the model gave no answer"), False),
        (BudgetExceeded("the run costs too much"), False),
        (steps.ParseError("the answer is not JSON"), False),
        (OSError("the disk is not there"), False),
        # Errors that can be a fault of the code: the traceback, then one line.
        (ValueError("a wrong value"), True),
        (RuntimeError("a tool stopped"), True),
        (yaml.YAMLError("a wrong YAML file"), True),
        (KeyError("a missing key"), True),
    ],
)
def test_main_gives_code_2_for_every_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], error: Exception, shows_traceback: bool
) -> None:
    """No error leaves ``main``, and no error gives the exit code 1 of a failed check or gate."""

    def stop(args: argparse.Namespace) -> int:
        raise error

    monkeypatch.setattr(cli, "_cmd_sync", stop)

    assert cli.main(["sync", "--results", "results"]) == 2

    stderr = capsys.readouterr().err
    assert ("Traceback (most recent call last)" in stderr) is shows_traceback
    last_line = stderr.splitlines()[-1]
    if shows_traceback:
        assert last_line == f"error: the command stopped with {type(error).__name__}; the traceback above says where"
    else:
        assert last_line == f"error: {error}"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        # A list at the top level: the ValueError of the settings reader.
        ("- not\n- a mapping\n", "the top level must be a mapping"),
        # A key that the settings do not have: the validation error of the settings record.
        ("keyframe_interval: 0.5\n", "keyframe_interval: Extra inputs are not permitted"),
        # A thinking level that the model does not have: the place of the value comes first.
        (
            "people: {thinking_level: xhigh}\n",
            "people: Value error, the model gemini-4-argon has no thinking level 'xhigh': "
            "its levels are low, medium, high",
        ),
        # Two wrong values: both are in the line, each with its place.
        (
            "repeat: -1\nmax_parallel: many\n",
            "max_parallel: Input should be a valid integer, unable to parse string as an integer; "
            "repeat: Input should be greater than or equal to 0",
        ),
        # A bracket that is not closed: the error of the YAML reader, which is not a ValueError.
        ("people: [gemini\n", "while parsing a flow sequence"),
    ],
)
def test_wrong_settings_end_with_code_2_and_one_line_that_names_the_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str, expected: str
) -> None:
    """A wrong settings file is wrong input, not a fault of the code: one line names the file and the fault."""
    settings = tmp_path / "settings.yaml"
    settings.write_text(content, encoding="utf-8")
    out = tmp_path / "runs"

    code = cli.main(["run", "--clip", str(tmp_path / "clip.mp4"), "--config", str(settings), "--out", str(out)])

    assert code == 2
    stderr = capsys.readouterr().err
    assert "Traceback" not in stderr
    # The error line is the last line, also when the reader of the file gave a text of several lines.
    last_line = stderr.splitlines()[-1]
    assert last_line.startswith(f"error: the settings file {settings} cannot be used: ")
    assert last_line.count(str(settings)) == 1
    assert expected in last_line
    assert not out.exists()


def test_wrong_roster_file_ends_with_code_2_and_one_line_that_names_the_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    roster = tmp_path / "roster.yaml"
    roster.write_text(
        "left:\n  players:\n    1: First Keeper\n    1: Second Keeper\nright:\n  players:\n    9: Striker\n",
        encoding="utf-8",
    )
    out = tmp_path / "runs"

    code = cli.main(["run", "--clip", str(tmp_path / "clip.mp4"), "--roster", str(roster), "--out", str(out)])

    assert code == 2
    stderr = capsys.readouterr().err
    assert "Traceback" not in stderr
    assert stderr.splitlines()[-1] == (
        f"error: the roster file {roster} cannot be used: the key 1 is written twice (line 4)"
    )
    assert not out.exists()


@pytest.mark.parametrize("command", ["run", "probe", "oracle"])
def test_wrong_label_file_ends_with_code_2_and_one_line_that_names_the_file(
    clip_on_disk: tuple[SyntheticScene, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    """The label reader is strict. Its refusal names the file and the place, for every command that reads labels."""
    _, source_root = clip_on_disk
    labels = read_json(source_root / "SoccerNetGS-valid" / CLIP_ID / LABEL_NAME)
    labels["info"]["frame_rate"] = 0
    label_file = tmp_path / "data" / "SoccerNetGS-valid" / CLIP_ID / LABEL_NAME
    label_file.parent.mkdir(parents=True)
    label_file.write_text(json.dumps(labels), encoding="utf-8")
    manifest = tmp_path / "splits.json"
    manifest.write_text(json.dumps({"seed": 7, "valid": {"probe": [CLIP_ID]}}), encoding="utf-8")
    out = tmp_path / "out"
    command_lines = {
        "run": ["run", "--clip", str(tmp_path / "clip.mp4"), "--labels", str(label_file), "--out", str(out)],
        "probe": [
            "probe",
            "pitch",
            "--experiment",
            str(EXPERIMENTS / "pitch_probe.yaml"),
            "--root",
            str(tmp_path / "data"),
            "--manifest",
            str(manifest),
            "--out",
            str(out / "pitch.json"),
            "--cache",
            str(tmp_path / "cache"),
            "--allow-dirty",
        ],
        "oracle": oracle_arguments(tmp_path / "data", tmp_path / "scorer.py", out),
    }

    code = cli.main(command_lines[command])

    assert code == 2
    stderr = capsys.readouterr().err
    assert "Traceback" not in stderr
    assert stderr.splitlines()[-1] == (
        f"error: the label file {label_file} cannot be used: info: frame_rate must be a number above 0, found 0.0"
    )
    assert not out.exists()


def test_a_missing_settings_file_ends_with_code_2_and_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An error of the operating system names its cause, so no traceback is printed."""
    out = tmp_path / "runs"

    code = cli.main(
        ["run", "--clip", str(tmp_path / "clip.mp4"), "--config", str(tmp_path / "missing.yaml"), "--out", str(out)]
    )

    assert code == 2
    stderr = capsys.readouterr().err
    assert "Traceback" not in stderr
    last_line = stderr.splitlines()[-1]
    assert last_line.startswith("error: ")
    assert "missing.yaml" in last_line
    assert not out.exists()


def test_run_ends_with_code_2_when_the_labels_belong_to_another_clip(
    clip_on_disk: tuple[SyntheticScene, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene, root = clip_on_disk
    other = tmp_path / "SNGS-000"
    other.mkdir()
    assert cv2.imwrite(str(other / "000001.png"), scene.image(0))
    labels = root / "SoccerNetGS-valid" / CLIP_ID / LABEL_NAME
    out = tmp_path / "runs"

    code = cli.main(["run", "--clip", str(other), "--labels", str(labels), "--out", str(out), "--allow-dirty"])

    assert code == 2
    stderr = capsys.readouterr().err
    # The error names the clip id of the frames, which is not the clip id of the labels.
    assert "Traceback (most recent call last)" in stderr
    assert "SNGS-000" in stderr
    assert stderr.splitlines()[-1].startswith("error: the command stopped with ")
    assert not out.exists()


def test_run_ends_with_code_2_when_a_model_answer_cannot_be_parsed(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The error names the answer in one line. The answer was paid, so its cost is recorded and printed."""
    scene, root = clip_on_disk
    out, cache = tmp_path / "runs", tmp_path / "cache"
    monkeypatch.setattr(cli, "gemini_backend", lambda max_usd_per_10_min: NotJsonBackend(FakeBackend(scene)))

    code = cli.main(
        [
            "run",
            "--clip",
            str(root / "SoccerNetGS-valid" / CLIP_ID),
            "--fps",
            "25",
            "--out",
            str(out),
            "--cache",
            str(cache),
            "--allow-dirty",
        ]
    )

    assert code == 2
    captured = capsys.readouterr()
    # The kit legend is the first request of a run, so the error names its answer.
    assert "Traceback" not in captured.err
    error_line = captured.err.splitlines()[-1]
    assert error_line.startswith("error: ")
    assert "kit_legend" in error_line
    assert CLIP_ID in error_line
    assert len((cache / "ledger.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert "measured cost" in captured.out
    assert not out.exists()


def test_run_records_the_cost_of_an_answer_that_was_billed_and_then_rejected(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The cached backend is the one place that reports a billed answer of a run. It stores no cassette for it."""
    scene, root = clip_on_disk
    out, cache = tmp_path / "runs", tmp_path / "cache"
    monkeypatch.setattr(
        cli, "gemini_backend", lambda max_usd_per_10_min: RejectedAnswerBackend(FakeBackend(scene), "kit_legend")
    )

    code = cli.main(
        [
            "run",
            "--clip",
            str(root / "SoccerNetGS-valid" / CLIP_ID),
            "--fps",
            "25",
            "--out",
            str(out),
            "--cache",
            str(cache),
            "--allow-dirty",
        ]
    )

    assert code == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert captured.err.splitlines()[-1] == "error: the kit_legend answer has no text"
    # The call was billed: the ledger has one line with the tokens of the answer, and the cost is printed.
    (line,) = read_ledger(cache)
    assert (line["step"], line["input_tokens"]) == ("kit_legend", STEP_USAGE["kit_legend"].input_tokens)
    assert line["usd"] > 0.0
    assert "for 1 new answers" in captured.out
    # A rejected answer is never stored: the cache folder holds the ledger and no cassette.
    assert [entry.name for entry in cache.iterdir()] == ["ledger.jsonl"]
    assert not out.exists()


# --- probe -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("file_name", "kind", "resolution", "per_clip"),
    [
        ("people_probe.yaml", "people", "high", 5),
        ("pitch_probe.yaml", "pitch", "high", 5),
        ("number_probe.yaml", "number", "low", 17),
    ],
)
def test_experiment_files_describe_the_three_probes(file_name: str, kind: str, resolution: str, per_clip: int) -> None:
    path = EXPERIMENTS / file_name

    experiment = cli.load_experiment(path)

    assert experiment.probe == kind
    assert experiment.name == path.stem
    assert experiment.group == "probe"
    assert experiment.per_clip == per_clip
    assert experiment.repeat == 0
    assert experiment.step == StepSettings(resolution=resolution)
    assert (experiment.step.model, experiment.step.thinking_level) == ("gemini-4-argon", "low")
    # Each file writes out all four settings, and they are the defaults of the step that the probe measures.
    assert set(yaml.safe_load(path.read_text(encoding="utf-8"))["step"]) == set(StepSettings.model_fields)
    assert experiment.step == cli.default_step_settings(experiment.probe)
    # One comment line says what the probe measures and which gate it serves.
    comments = [line for line in path.read_text(encoding="utf-8").splitlines() if line.lstrip().startswith("#")]
    assert len(comments) == 1
    assert "gate G" in comments[0]


def test_people_and_pitch_probe_use_the_same_frames() -> None:
    people = cli.load_experiment(EXPERIMENTS / "people_probe.yaml")
    pitch = cli.load_experiment(EXPERIMENTS / "pitch_probe.yaml")

    assert (people.seed, people.frames_per_clip, people.group) == (pitch.seed, pitch.frames_per_clip, pitch.group)


def test_default_step_settings_are_those_of_the_step_that_a_probe_measures() -> None:
    assert cli.default_step_settings("people") == StepSettings()
    assert cli.default_step_settings("pitch") == StepSettings()
    # The number step reads its crops at a low resolution.
    assert cli.default_step_settings("number") == StepSettings(resolution="low")
    pipeline = PipelineConfig()
    assert {kind: cli.default_step_settings(kind) for kind in cli.PROBE_KINDS} == {
        "people": pipeline.people,
        "pitch": pipeline.landmarks,
        "number": pipeline.number,
    }


@pytest.mark.parametrize(
    ("kind", "count_key", "resolution"),
    [("people", "frames_per_clip", "high"), ("pitch", "frames_per_clip", "high"), ("number", "tracks_per_clip", "low")],
)
def test_partial_step_section_keeps_the_defaults_of_the_step_of_the_probe(
    tmp_path: Path, kind: str, count_key: str, resolution: str
) -> None:
    """A step section that names one key keeps the other defaults of its step: a number probe stays at low."""
    head = f"name: x\nprobe: {kind}\ngroup: probe\n{count_key}: 5\nseed: 1\nrepeat: 0\n"
    path = tmp_path / "experiment.yaml"

    path.write_text(head + "step:\n  model: gemini-3.8-flash\n", encoding="utf-8")
    assert cli.load_experiment(path).step == StepSettings(model="gemini-3.8-flash", resolution=resolution)

    path.write_text(head + "step:\n  service_tier: flex\n  thinking_level: medium\n", encoding="utf-8")
    assert cli.load_experiment(path).step == StepSettings(
        thinking_level="medium", resolution=resolution, service_tier="flex"
    )

    # An empty section gives the defaults of the step.
    path.write_text(head + "step: {}\n", encoding="utf-8")
    assert cli.load_experiment(path).step == cli.default_step_settings(kind)

    # A key that the section names wins over the default.
    path.write_text(head + "step:\n  resolution: medium\n", encoding="utf-8")
    assert cli.load_experiment(path).step == StepSettings(resolution="medium")


def test_experiment_takes_a_settings_record_as_it_is() -> None:
    """Only a step section is laid over the defaults. Settings that are built in code are complete."""
    settings = StepSettings(model="gemini-3.8-flash")
    experiment = cli.Experiment(
        name="x", probe="number", group="probe", tracks_per_clip=3, seed=1, repeat=0, step=settings
    )

    assert experiment.step == settings
    assert experiment.step.resolution == "high"
    # A changed copy keeps the settings too, and a section in the change is laid over the defaults.
    assert experiment.model_copy(update={"repeat": 1}).step == settings
    assert experiment.model_copy(update={"step": {"service_tier": "flex"}}).step == StepSettings(
        resolution="low", service_tier="flex"
    )


@pytest.mark.parametrize(
    "content",
    [
        "- not\n- a mapping\n",
        "name: x\nprobe: people\ngroup: probe\nseed: 1\nrepeat: 0\nstep: {}\n",
        "name: x\nprobe: number\ngroup: probe\nframes_per_clip: 5\nseed: 1\nrepeat: 0\nstep: {}\n",
        "name: x\nprobe: people\ngroup: probe\nframes_per_clip: 5\ntracks_per_clip: 5\nseed: 1\nrepeat: 0\nstep: {}\n",
        "name: x\nprobe: people\ngroup: probe\nframes_per_clip: 5\nseed: 1\nrepeat: 0\nstep: {}\nextra: 1\n",
        "name: x\nprobe: shirts\ngroup: probe\nframes_per_clip: 5\nseed: 1\nrepeat: 0\nstep: {}\n",
        # The probe is not a text, the step section is missing, is not a mapping, or has an unknown key.
        "name: x\nprobe: [people]\ngroup: probe\nframes_per_clip: 5\nseed: 1\nrepeat: 0\nstep: {}\n",
        "name: x\nprobe: people\ngroup: probe\nframes_per_clip: 5\nseed: 1\nrepeat: 0\n",
        "name: x\nprobe: people\ngroup: probe\nframes_per_clip: 5\nseed: 1\nrepeat: 0\nstep: fast\n",
        "name: x\nprobe: number\ngroup: probe\ntracks_per_clip: 5\nseed: 1\nrepeat: 0\nstep: {modle: x}\n",
        "name: x\nprobe: number\ngroup: probe\ntracks_per_clip: 5\nseed: 1\nrepeat: 0\nstep: {resolution: huge}\n",
    ],
)
def test_load_experiment_refuses_wrong_files(tmp_path: Path, content: str) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(cli.CliError, match="experiment.yaml"):
        cli.load_experiment(path)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            "name: x\nprobe: number\ngroup: probe\ntracks_per_clip: 5\nseed: 1\nrepeat: 0\nstep: {resolution: huge}\n",
            "step.resolution: Input should be 'low', 'medium', 'high' or 'ultra_high'",
        ),
        ("name: x\nstep: [people\n", "while parsing a flow sequence"),
    ],
)
def test_refusal_of_a_wrong_experiment_file_is_one_line_with_the_file_and_the_fault(
    tmp_path: Path, content: str, expected: str
) -> None:
    """The command line prints the refusal as its error line, which must be the last line of its output."""
    path = tmp_path / "experiment.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(cli.CliError) as excinfo:
        cli.load_experiment(path)

    message = str(excinfo.value)
    assert message.startswith(f"the experiment file {path} cannot be used: ")
    assert expected in message
    assert "\n" not in message


def test_group_clips_finds_the_split_of_a_group() -> None:
    manifest = {"seed": 7, "valid": {"probe": ["b", "a"], "dev": ["c"]}, "test": {"final": ["d"]}}

    assert cli.group_clips(manifest, "probe") == ("valid", ["b", "a"])
    assert cli.group_clips(manifest, "final") == ("test", ["d"])
    with pytest.raises(cli.CliError, match="probe, dev, final"):
        cli.group_clips(manifest, "fresh")


def test_probe_people_with_a_fake_model_and_then_from_the_cache(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scene, root = clip_on_disk
    manifest = tmp_path / "splits.json"
    manifest.write_text(json.dumps({"seed": 7, "valid": {"probe": [CLIP_ID]}}), encoding="utf-8")
    monkeypatch.setattr(cli, "gemini_backend", lambda max_usd_per_10_min: FakeBackend(scene))
    metrics = tmp_path / "metrics" / "people_probe.json"
    arguments = [
        "probe",
        "people",
        "--experiment",
        str(EXPERIMENTS / "people_probe.yaml"),
        "--root",
        str(root),
        "--manifest",
        str(manifest),
        "--out",
        str(metrics),
        "--cache",
        str(tmp_path / "cache"),
        "--allow-dirty",
    ]

    assert cli.main(arguments) == 0

    first = read_json(metrics)
    assert (first["probe"], first["n_clips"], first["n_frames"]) == ("people", 1, 5)
    assert first["recall"] > 0.85
    assert first["precision"] > 0.85
    assert first["settings"]["small_box_max_height_px"] == 110.0
    provenance = first["provenance"]
    assert provenance["command"] == "probe"
    assert provenance["experiment"] == "people_probe"
    assert provenance["experiment_file"] == "people_probe.yaml"
    assert (provenance["split"], provenance["group"], provenance["clips"]) == ("valid", "probe", [CLIP_ID])
    assert provenance["split_list_seed"] == 7
    assert isinstance(provenance["commit"], str) and provenance["commit"]
    # One kit legend request and five people requests.
    assert provenance["cost"]["new_answers"] == 6
    assert str(tmp_path) not in metrics.read_text(encoding="utf-8")
    output = capsys.readouterr().out
    assert output.index("cost estimate") < output.index("measured cost") < output.index("main metric recall")

    assert cli.main([*arguments, "--replay"]) == 0

    second = read_json(metrics)
    assert second["provenance"]["cost"]["new_answers"] == 0
    assert second["recall"] == first["recall"]


@pytest.mark.parametrize(("kind", "file_name"), [("pitch", "pitch_probe.yaml"), ("number", "number_probe.yaml")])
def test_probe_pitch_and_number_with_a_fake_model(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    file_name: str,
) -> None:
    scene, root = clip_on_disk
    manifest = tmp_path / "splits.json"
    manifest.write_text(json.dumps({"seed": 7, "valid": {"probe": [CLIP_ID]}}), encoding="utf-8")
    monkeypatch.setattr(cli, "gemini_backend", lambda max_usd_per_10_min: FakeBackend(scene))
    metrics = tmp_path / f"{kind}_probe.json"
    arguments = [
        "probe",
        kind,
        "--experiment",
        str(EXPERIMENTS / file_name),
        "--root",
        str(root),
        "--manifest",
        str(manifest),
        "--max-clips",
        "1",
        "--out",
        str(metrics),
        "--cache",
        str(tmp_path / "cache"),
        "--allow-dirty",
    ]

    assert cli.main(arguments) == 0

    measured = read_json(metrics)
    assert (measured["probe"], measured["n_clips"]) == (kind, 1)
    assert measured["provenance"]["max_clips"] == 1
    new_answers = measured["provenance"]["cost"]["new_answers"]
    if kind == "pitch":
        # One landmarks request per frame. The model answers the truth, so every frame gets a mapping.
        assert (measured["n_frames"], new_answers) == (5, 5)
        assert measured["valid_share"] == 1.0
        assert measured["error_m"]["median"] < 0.5
    else:
        # One kit legend request, and one number request per player track of the labels.
        assert 1 <= measured["n_tracks"] <= 17
        assert new_answers == measured["n_tracks"] + 1
        assert measured["exact_match"] == 1.0


def test_probe_records_the_cost_of_an_answer_that_was_billed_and_then_rejected(
    clip_on_disk: tuple[SyntheticScene, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scene, root = clip_on_disk
    manifest = tmp_path / "splits.json"
    manifest.write_text(json.dumps({"seed": 7, "valid": {"probe": [CLIP_ID]}}), encoding="utf-8")
    monkeypatch.setattr(
        cli, "gemini_backend", lambda max_usd_per_10_min: RejectedAnswerBackend(FakeBackend(scene), "people")
    )
    metrics, cache = tmp_path / "people_probe.json", tmp_path / "cache"

    code = cli.main(
        [
            "probe",
            "people",
            "--experiment",
            str(EXPERIMENTS / "people_probe.yaml"),
            "--root",
            str(root),
            "--manifest",
            str(manifest),
            "--out",
            str(metrics),
            "--cache",
            str(cache),
            "--allow-dirty",
        ]
    )

    # An error of the backend stops a probe. The rejected answer is the first people answer.
    assert code == 2
    assert capsys.readouterr().err.splitlines()[-1] == "error: the people answer has no text"
    assert not metrics.exists()
    # The kit legend answer and the rejected people answer were billed. Only the kit legend answer is stored.
    assert [line["step"] for line in read_ledger(cache)] == ["kit_legend", "people"]
    assert [path.parent.name for path in cache.glob(f"{CLIP_ID}/*/*.json")] == ["kit_legend"]


def test_probe_plan_counts_the_requests_of_every_clip() -> None:
    people = cli.load_experiment(EXPERIMENTS / "people_probe.yaml")
    pitch = cli.load_experiment(EXPERIMENTS / "pitch_probe.yaml")
    number = cli.load_experiment(EXPERIMENTS / "number_probe.yaml")

    # Three clips: 5 frames each for the people and the pitch probe, 17 tracks each for the number probe.
    assert cli.probe_plan(people, 3) == [("kit_legend", people.step, 3), ("people", people.step, 15)]
    assert cli.probe_plan(pitch, 3) == [("landmarks", pitch.step, 15)]
    assert cli.probe_plan(number, 3) == [("kit_legend", number.step, 3), ("number", number.step, 51)]


def test_max_clips_takes_exactly_that_many_clips_of_the_group(
    clip_on_disk: tuple[SyntheticScene, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene, root = clip_on_disk
    manifest = tmp_path / "splits.json"
    # The second clip of the group is not on disk: the probe must not open it.
    manifest.write_text(json.dumps({"seed": 7, "valid": {"probe": [CLIP_ID, "SNGS-000"]}}), encoding="utf-8")
    use_fake_model(monkeypatch, scene)
    metrics = tmp_path / "pitch_probe.json"
    arguments = [
        "probe",
        "pitch",
        "--experiment",
        str(EXPERIMENTS / "pitch_probe.yaml"),
        "--root",
        str(root),
        "--manifest",
        str(manifest),
        "--max-clips",
        "1",
        "--out",
        str(metrics),
        "--cache",
        str(tmp_path / "cache"),
        "--allow-dirty",
    ]

    assert cli.main(arguments) == 0

    assert read_json(metrics)["provenance"]["clips"] == [CLIP_ID]


def test_probe_refuses_an_experiment_of_another_kind(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    arguments = [
        "probe",
        "pitch",
        "--experiment",
        str(EXPERIMENTS / "people_probe.yaml"),
        "--root",
        str(tmp_path),
        "--out",
        str(tmp_path / "metrics.json"),
        "--allow-dirty",
    ]

    assert cli.main(arguments) == 2
    assert "people probe" in capsys.readouterr().err


# --- data ------------------------------------------------------------------------------


def test_data_writes_the_split_list_in_two_steps(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """First the validation split alone, then the test split. No request is sent: the archives are there."""
    root, manifest = tmp_path / "data", tmp_path / "configs" / "splits.json"
    valid_clips = [f"SNGS-{number:03d}" for number in range(21, 61)]
    test_clips = [f"SNGS-{number:03d}" for number in range(116, 128)]
    complete = make_split_manifest(valid_clips, test_clips, 7)
    make_archive(root / "archives" / "valid.zip", valid_clips)
    valid_arguments = ["data", "--root", str(root), "--split", "valid", "--manifest", str(manifest), "--seed", "7"]

    assert cli.main(valid_arguments) == 0

    first = read_json(manifest)
    assert first == {"seed": 7, "valid": complete["valid"]}
    assert sorted(path.name for path in (root / "SoccerNetGS-valid").iterdir()) == valid_clips
    assert "The test split is not downloaded" in capsys.readouterr().out

    make_archive(root / "archives" / "test.zip", test_clips)
    test_arguments = ["data", "--root", str(root), "--split", "test", "--manifest", str(manifest), "--seed", "7"]

    assert cli.main(test_arguments) == 0

    assert read_json(manifest) == complete
    assert "The test split is not downloaded" not in capsys.readouterr().out

    # Another seed would change the list, which is fixed before the first paid run.
    assert cli.main([*test_arguments[:-1], "8"]) == 2
    assert read_json(manifest) == complete
    assert "fixed before the first paid run" in capsys.readouterr().err


def test_data_extracts_only_the_wanted_clips(tmp_path: Path) -> None:
    root = tmp_path / "data"
    make_archive(root / "archives" / "valid.zip", ["SNGS-021", "SNGS-022", "SNGS-023"])

    assert cli.main(["data", "--root", str(root), "--split", "valid", "--clips", "SNGS-022"]) == 0

    assert [path.name for path in (root / "SoccerNetGS-valid").iterdir()] == ["SNGS-022"]


def test_data_needs_manifest_and_seed_together(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "data"

    assert cli.main(["data", "--root", str(root), "--split", "valid", "--manifest", str(tmp_path / "s.json")]) == 2
    assert cli.main(["data", "--root", str(root), "--split", "valid", "--seed", "7"]) == 2

    assert "--manifest and --seed" in capsys.readouterr().err
    # The command stopped before any download.
    assert not root.exists()


# --- oracle and eval -------------------------------------------------------------------


def oracle_arguments(root: Path, script: Path, out: Path) -> list[str]:
    """Return the command line of the oracle command for the synthetic clip."""
    return [
        "oracle",
        "--root",
        str(root),
        "--split",
        "valid",
        "--scorer-python",
        sys.executable,
        "--scorer-script",
        str(script),
        "--out",
        str(out),
    ]


def test_oracle_passes_when_every_value_is_inside_its_gate(
    clip_on_disk: tuple[SyntheticScene, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, root = clip_on_disk
    out = tmp_path / "oracle"

    code = cli.main(oracle_arguments(root, write_evaluator(tmp_path / "scorer", PASSING_SHARES), out))

    assert code == 0
    report = read_json(out / "oracle.json")
    assert report["passed"] is True
    assert report["clips"] == [CLIP_ID]
    assert len(report["gates"]) == 6
    assert all(gate["passed"] for gate in report["gates"])
    assert report["variants"]["ground_truth"]["scores"]["COMBINED"]["HOTA"] == pytest.approx(100.0)
    assert report["variants"]["shifted_1m"]["scores"][CLIP_ID]["HOTA"] == pytest.approx(89.5)
    # Team A of the scene defends the left goal, and the labels follow the true camera.
    side_rule = report["team_side_rule"]
    assert side_rule["clips_right"] == [CLIP_ID]
    assert (side_rule["share_right"], side_rule["share_right_by_players_alone"]) == (1.0, 1.0)
    assert (side_rule["clips_wrong"], side_rule["clips_decided_by_goalkeepers"]) == ([], [])
    # The values that the two limits of the rule are compared with: the two teams of players stand
    # far apart, and no goalkeeper stands in the goal of the other team.
    facts = side_rule["facts"][CLIP_ID]
    assert (facts["agrees"], facts["decided_by"], facts["players_alone_agree"]) == (True, "players", True)
    assert facts["player_mean_gap_m"] > 10.0
    assert facts["goalkeepers_in_own_goal_zone"] == facts["goalkeeper_tracks"] >= 1
    assert facts["goalkeepers_in_other_goal_zone"] == 0
    assert report["label_homography_residual_m"]["n"] > 300
    assert report["label_homography_residual_m"]["median"] < 0.01
    assert report["format_problems_of_ground_truth"] == {}
    assert report["provenance"]["command"] == "oracle"
    # The stand-in evaluator is no clone of the scorer repository, so its commit is not known.
    assert report["provenance"]["scorer_commit"] is None
    output = capsys.readouterr().out
    assert "[ok  ] GS-HOTA of the ground truth: 100.000" in output
    assert "the players alone are right on a share of 1.000 of the clips; the goalkeepers decided 0 clips" in output

    files = {
        variant: read_json(out / "predictions" / "SoccerNetGS-valid" / tracker / f"{CLIP_ID}.json")["predictions"]
        for variant, tracker in cli.ORACLE_TRACKERS.items()
    }
    truth, shifted = files["ground_truth"], files["shifted_1m"]
    assert len(shifted) == len(truth) > 300
    # The shift is 1 m towards the right goal: x grows, y stays.
    for true_item, shifted_item in zip(truth[:50], shifted[:50], strict=True):
        true_spot, shifted_spot = true_item["bbox_pitch"], shifted_item["bbox_pitch"]
        assert shifted_spot["x_bottom_middle"] == pytest.approx(true_spot["x_bottom_middle"] + 1.0)
        assert shifted_spot["y_bottom_middle"] == true_spot["y_bottom_middle"]
    all_images = {f"2999{number:06d}" for number in range(1, N_FRAMES + 1)}
    assert {item["image_id"] for item in truth} == all_images
    assert {item["image_id"] for item in files["resampled"]} == all_images
    # Frame 0 is image 1, so the keyframes are the images 1, 13, 25, 37, 49 and 50.
    assert {item["image_id"] for item in files["keyframes_only"]} == {f"2999{frame + 1:06d}" for frame in KEYFRAMES}


def test_oracle_ends_with_code_1_when_a_gate_is_missed(
    clip_on_disk: tuple[SyntheticScene, Path], tmp_path: Path
) -> None:
    _, root = clip_on_disk
    out = tmp_path / "oracle"

    code = cli.main(oracle_arguments(root, write_evaluator(tmp_path / "scorer", {"default": 0.5}), out))

    assert code == 1
    report = read_json(out / "oracle.json")
    assert report["passed"] is False
    passed = {gate["name"]: gate["passed"] for gate in report["gates"]}
    assert passed["GS-HOTA of the ground truth"] is False
    assert passed["GS-HOTA of the keyframes only"] is False
    # The checks on the labels alone do not depend on the scorer.
    assert passed["share of clips where the team-side rule is right"] is True
    assert passed["median residual of the label homographies in metres"] is True


def test_oracle_ends_with_code_2_when_the_scorer_stops(
    clip_on_disk: tuple[SyntheticScene, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A scorer that stops with an error is not a missed gate: the exit code is 2, and no report is written."""
    _, root = clip_on_disk
    out = tmp_path / "oracle"
    script = tmp_path / "broken_evaluator.py"
    script.write_text('raise SystemExit("the evaluator is broken")\n', encoding="utf-8")

    code = cli.main(oracle_arguments(root, script, out))

    assert code == 2
    stderr = capsys.readouterr().err
    # The traceback holds the last lines that the scorer printed.
    assert "Traceback (most recent call last)" in stderr
    assert "the evaluator is broken" in stderr
    assert stderr.splitlines()[-1].startswith("error: the command stopped with ")
    assert not (out / "oracle.json").exists()


def test_gate_checks_do_not_pass_a_value_that_is_not_a_number() -> None:
    nan = float("nan")

    assert cli.at_least("x", 99.5, 99.5).passed is True
    assert cli.at_least("x", nan, 99.5).passed is False
    assert cli.at_most("x", nan, 0.3).passed is False
    assert cli.within("x", 89.0, 89.0, 90.0).passed is True
    assert cli.within("x", 90.1, 89.0, 90.0).passed is False
    assert cli.within("x", nan, 89.0, 90.0).passed is False


def test_gate_checks_include_their_limits() -> None:
    assert cli.at_most("x", 0.3, 0.3).passed is True
    assert cli.at_most("x", 0.31, 0.3).passed is False
    assert cli.at_least("x", 99.49, 99.5).passed is False
    assert cli.within("x", 90.0, 89.0, 90.0).passed is True
    assert cli.within("x", 88.9, 89.0, 90.0).passed is False


def oracle_root(tmp_path: Path, source_root: Path, change: Callable[[dict[str, Any]], None]) -> Path:
    """Make a data root with the label file of the synthetic clip, edited by ``change``. The oracle needs no frames."""
    labels = read_json(source_root / "SoccerNetGS-valid" / CLIP_ID / LABEL_NAME)
    change(labels)
    root = tmp_path / "data"
    label_file = root / "SoccerNetGS-valid" / CLIP_ID / LABEL_NAME
    label_file.parent.mkdir(parents=True)
    label_file.write_text(json.dumps(labels), encoding="utf-8")
    return root


def test_clip_without_a_labelled_side_does_not_count_against_the_team_side_rule(
    clip_on_disk: tuple[SyntheticScene, Path], tmp_path: Path
) -> None:
    _, source_root = clip_on_disk
    root = oracle_root(tmp_path, source_root, lambda labels: None)
    # A second clip with the same labels, but without any team label.
    labels = read_json(root / "SoccerNetGS-valid" / CLIP_ID / LABEL_NAME)
    labels["info"]["name"] = "SNGS-998"
    for row in labels["annotations"]:
        if "attributes" in row:
            row["attributes"]["team"] = None
    second = root / "SoccerNetGS-valid" / "SNGS-998" / LABEL_NAME
    second.parent.mkdir()
    second.write_text(json.dumps(labels), encoding="utf-8")
    out = tmp_path / "oracle"

    code = cli.main(oracle_arguments(root, write_evaluator(tmp_path / "scorer", PASSING_SHARES), out))

    report = read_json(out / "oracle.json")
    assert report["team_side_rule"]["clips_without_labelled_side"] == ["SNGS-998"]
    assert report["team_side_rule"]["clips_right"] == [CLIP_ID]
    # The rule is right on the one clip that can be judged: a share of 1, not of 1 in 2.
    assert report["team_side_rule"]["share_right"] == 1.0
    assert code == 0


def side_facts(*, agrees: bool, decided_by: str, players_alone_agree: bool, gap_m: float | None) -> TeamSideFacts:
    """Build the facts of the team-side rule for a clip with one goalkeeper in the goal of its own side."""
    return TeamSideFacts(
        agrees=agrees,
        decided_by=decided_by,  # type: ignore[arg-type]
        players_alone_agree=players_alone_agree,
        player_mean_gap_m=gap_m,
        goalkeeper_tracks=1,
        goalkeepers_in_own_goal_zone=1,
        goalkeepers_in_other_goal_zone=0,
    )


def test_side_rule_report_counts_the_clips_and_keeps_the_facts_of_every_clip() -> None:
    facts = {
        "clip-b": side_facts(agrees=False, decided_by="players", players_alone_agree=False, gap_m=4.5),
        "clip-a": side_facts(agrees=True, decided_by="goalkeepers", players_alone_agree=False, gap_m=1.0),
        "clip-d": side_facts(agrees=True, decided_by="players", players_alone_agree=True, gap_m=None),
        "clip-c": None,
    }

    report = cli.side_rule_report(facts)

    # Three clips can be judged. The rule of the pipeline is right on two of them, the players alone on one.
    assert report["share_right"] == pytest.approx(2 / 3)
    assert report["share_right_by_players_alone"] == pytest.approx(1 / 3)
    assert (report["clips_right"], report["clips_wrong"]) == (["clip-a", "clip-d"], ["clip-b"])
    assert report["clips_without_labelled_side"] == ["clip-c"]
    assert report["clips_decided_by_goalkeepers"] == ["clip-a"]
    assert list(report["facts"]) == ["clip-a", "clip-b", "clip-d"]
    assert report["facts"]["clip-a"] == {
        "agrees": True,
        "decided_by": "goalkeepers",
        "players_alone_agree": False,
        "player_mean_gap_m": 1.0,
        "goalkeeper_tracks": 1,
        "goalkeepers_in_own_goal_zone": 1,
        "goalkeepers_in_other_goal_zone": 0,
    }
    assert report["facts"]["clip-d"]["player_mean_gap_m"] is None


def test_side_rule_report_without_a_clip_that_can_be_judged_has_no_share() -> None:
    report = cli.side_rule_report({"clip-c": None})

    assert np.isnan(report["share_right"]) and np.isnan(report["share_right_by_players_alone"])
    assert (report["clips_right"], report["clips_wrong"], report["facts"]) == ([], [], {})
    # A share that is not a number does not pass the gate.
    assert cli.at_least("share", report["share_right"], 0.97).passed is False


def test_gate_g1_compares_the_median_residual_of_the_label_homographies(
    clip_on_disk: tuple[SyntheticScene, Path], tmp_path: Path
) -> None:
    """One labelled pitch position of every frame is 30 m off. The median residual does not see it, the mean does."""
    _, source_root = clip_on_disk

    def one_wrong_label_per_frame(labels: dict[str, Any]) -> None:
        seen: set[str] = set()
        for row in labels["annotations"]:
            if row.get("supercategory") == "object" and row.get("bbox_pitch") and row["image_id"] not in seen:
                seen.add(row["image_id"])
                for key in ("x_bottom_left", "x_bottom_middle", "x_bottom_right"):
                    row["bbox_pitch"][key] += 30.0

    root = oracle_root(tmp_path, source_root, one_wrong_label_per_frame)
    out = tmp_path / "oracle"

    cli.main(oracle_arguments(root, write_evaluator(tmp_path / "scorer", PASSING_SHARES), out))

    report = read_json(out / "oracle.json")
    residual = report["label_homography_residual_m"]
    assert residual["median"] < 0.01
    assert residual["mean"] > 0.5
    gate = {gate["name"]: gate for gate in report["gates"]}["median residual of the label homographies in metres"]
    assert gate["value"] == residual["median"]
    assert gate["passed"] is True


def test_eval_prints_the_scores_per_clip_and_of_all_clips(
    clip_on_disk: tuple[SyntheticScene, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, root = clip_on_disk
    pred, out = tmp_path / "pred", tmp_path / "eval"
    prediction = pred / "SoccerNetGS-valid" / "pitchvlm" / f"{CLIP_ID}.json"
    prediction.parent.mkdir(parents=True)
    prediction.write_text(json.dumps({"predictions": []}), encoding="utf-8")
    script = write_evaluator(tmp_path / "scorer", {"default": 0.25})
    arguments = [
        "eval",
        "--root",
        str(root),
        "--pred",
        str(pred),
        "--split",
        "valid",
        "--tracker",
        "pitchvlm",
        "--clips",
        CLIP_ID,
        "--scorer-python",
        sys.executable,
        "--scorer-script",
        str(script),
        "--out",
        str(out),
    ]

    assert cli.main(arguments) == 0

    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert ["clip", "GS-HOTA", "DetA", "AssA"] in rows
    assert [CLIP_ID, "25.00", "23.00", "21.00"] in rows
    assert ["COMBINED", "25.00", "23.00", "21.00"] in rows
    scores = read_json(out / "scores.json")
    assert scores["scores"]["COMBINED"] == pytest.approx({"HOTA": 25.0, "DetA": 23.0, "AssA": 21.0})
    assert (scores["split"], scores["tracker"], scores["clips"]) == ("valid", "pitchvlm", [CLIP_ID])
    assert scores["provenance"]["scorer_commit"] is None
    # The three detailed files of the evaluator are moved into the output folder. The scores are those
    # of the class person.
    assert sorted(path.name for path in out.glob("*_detailed.csv")) == [
        "cls_comb_cls_av_detailed.csv",
        "cls_comb_det_av_detailed.csv",
        "person_detailed.csv",
    ]


def test_scorer_commit_is_the_commit_of_the_clone_that_holds_the_script(private_git: None, tmp_path: Path) -> None:
    """The scorer script runs the code of its clone, so the commit of the clone names the code that scored."""
    clone = tmp_path / "sn-trackeval"
    script = clone / "scripts" / "run_soccernet_gs.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('scores')\n", encoding="utf-8")
    nested = clone / "tools" / "scripts" / "run_soccernet_gs.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("print('scores')\n", encoding="utf-8")
    author = ["-c", "user.name=Tester", "-c", "user.email=tester" + "@" + "example.org", "-c", "commit.gpgsign=false"]

    assert cli.scorer_commit(script) is None  # The folder is no git work tree.
    git(clone, "init", "--quiet", "--template=")
    assert cli.scorer_commit(script) is None  # The work tree has no commit.
    git(clone, "add", "--all")
    git(clone, *author, "commit", "--quiet", "-m", "the scorer")

    commit = git(clone, "rev-parse", "HEAD").strip()
    assert len(commit) == 40
    assert cli.scorer_commit(script) == commit
    # A script two folders below another folder of the work tree is not the script of this clone.
    assert cli.scorer_commit(nested) is None


def test_scorer_commit_does_not_follow_the_letter_case_that_was_typed(private_git: None, tmp_path: Path) -> None:
    """Git spells the clone folder as the disk stores it. The typed path can have another letter case."""
    if tells_letter_case_apart(tmp_path):
        pytest.skip("the disk tells upper and lower case apart, so a path in another letter case names no file")
    clone = tmp_path / "sn-trackeval"
    script = clone / "scripts" / "run_soccernet_gs.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('scores')\n", encoding="utf-8")
    author = ["-c", "user.name=Tester", "-c", "user.email=tester" + "@" + "example.org", "-c", "commit.gpgsign=false"]
    git(clone, "init", "--quiet", "--template=")
    git(clone, "add", "--all")
    git(clone, *author, "commit", "--quiet", "-m", "the scorer")
    commit = git(clone, "rev-parse", "HEAD").strip()

    assert cli.scorer_commit(tmp_path / "SN-TrackEval" / "scripts" / "run_soccernet_gs.py") == commit
    assert cli.scorer_commit(tmp_path / "sn-trackeval" / "Scripts" / "RUN_SOCCERNET_GS.PY") == commit
    # A link to the clone names the same place too.
    (tmp_path / "linked").symlink_to(clone)
    assert cli.scorer_commit(tmp_path / "linked" / "scripts" / "run_soccernet_gs.py") == commit


def test_eval_records_the_commit_of_the_scorer_clone(
    private_git: None, clip_on_disk: tuple[SyntheticScene, Path], tmp_path: Path
) -> None:
    _, root = clip_on_disk
    pred, out = tmp_path / "pred", tmp_path / "eval"
    prediction = pred / "SoccerNetGS-valid" / "pitchvlm" / f"{CLIP_ID}.json"
    prediction.parent.mkdir(parents=True)
    prediction.write_text(json.dumps({"predictions": []}), encoding="utf-8")
    clone = tmp_path / "sn-trackeval"
    script = write_evaluator(clone / "scripts", {"default": 0.25})
    author = ["-c", "user.name=Tester", "-c", "user.email=tester" + "@" + "example.org", "-c", "commit.gpgsign=false"]
    git(clone, "init", "--quiet", "--template=")
    git(clone, "add", "--all")
    git(clone, *author, "commit", "--quiet", "-m", "the scorer")
    arguments = ["eval", "--root", str(root), "--pred", str(pred), "--split", "valid", "--tracker", "pitchvlm"]
    arguments += ["--clips", CLIP_ID, "--scorer-python", sys.executable, "--scorer-script", str(script)]

    assert cli.main([*arguments, "--out", str(out)]) == 0

    assert read_json(out / "scores.json")["provenance"]["scorer_commit"] == git(clone, "rev-parse", "HEAD").strip()


def test_oracle_and_eval_give_the_evaluator_no_model_key(
    clip_on_disk: tuple[SyntheticScene, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The evaluator is the code of another project. Its process must not get the key of the model."""
    _, root = clip_on_disk
    monkeypatch.setenv(cli.KEY_VARIABLE, "not-a-real-value")
    monkeypatch.setenv("PITCHVLM_TEST_OTHER", "kept")
    script = write_evaluator(tmp_path / "scorer", PASSING_SHARES)
    seen_file = script.with_name("seen.txt")
    # Every run of the stand-in adds a line that says which of the two variables it sees.
    notes_its_environment = (
        "import os\n"
        f"with open({str(seen_file)!r}, 'a', encoding='utf-8') as out:\n"
        f"    names = ({cli.KEY_VARIABLE!r}, 'PITCHVLM_TEST_OTHER')\n"
        "    out.write(','.join(name for name in names if name in os.environ) + '\\n')\n"
    )
    script.write_text(notes_its_environment + FAKE_EVALUATOR, encoding="utf-8")

    assert cli.main(oracle_arguments(root, script, tmp_path / "oracle")) == 0

    # The evaluator ran one time for each of the four variants. It saw the other variable only.
    assert seen_file.read_text(encoding="utf-8").splitlines() == ["PITCHVLM_TEST_OTHER"] * len(cli.ORACLE_TRACKERS)

    seen_file.unlink()
    pred = tmp_path / "pred"
    prediction = pred / "SoccerNetGS-valid" / "pitchvlm" / f"{CLIP_ID}.json"
    prediction.parent.mkdir(parents=True)
    prediction.write_text(json.dumps({"predictions": []}), encoding="utf-8")
    arguments = ["eval", "--root", str(root), "--pred", str(pred), "--split", "valid", "--tracker", "pitchvlm"]
    arguments += ["--clips", CLIP_ID, "--scorer-python", sys.executable, "--scorer-script", str(script)]

    assert cli.main([*arguments, "--out", str(tmp_path / "eval")]) == 0

    assert seen_file.read_text(encoding="utf-8").splitlines() == ["PITCHVLM_TEST_OTHER"]
    # The commands do not change their own environment.
    assert os.environ[cli.KEY_VARIABLE] == "not-a-real-value"


# --- sync ------------------------------------------------------------------------------


@pytest.mark.parametrize("planted", ["frame_000012.jpg", "clip.mp4", "crop.PNG", "notes"])
def test_sync_refuses_a_planted_media_file(tmp_path: Path, capsys: pytest.CaptureFixture[str], planted: str) -> None:
    """A file that is not a text result file stops the command, before git or the guard are used."""
    results = tmp_path / "results"
    write_results(results)
    (results / "cache" / "SNGS-021" / "people" / planted).write_bytes(b"")

    assert cli.main(["sync", "--results", str(results)]) == 1

    output = capsys.readouterr().out
    assert f"refused: cache/SNGS-021/people/{planted}" in output
    assert "git -C" not in output


def test_sync_refuses_a_link(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    results = tmp_path / "results"
    write_results(results)
    (results / "linked.json").symlink_to(results / "metrics" / "people_probe.json")

    assert cli.main(["sync", "--results", str(results)]) == 1

    assert "refused: linked.json" in capsys.readouterr().out


def test_sync_prints_the_git_commands_and_runs_none_of_them(
    results_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = results_tree / "results"
    write_results(results)
    # A cost ledger that holds more than 1,000,000 bytes goes on in a second file.
    later_ledger = results / "cache" / "ledger-0002.jsonl"
    later_ledger.write_text(json.dumps({"run_id": "r", "usd": 0.2}) + "\n", encoding="utf-8")
    # The run files of the two external tools: boxes that a detector found, and camera matrices.
    (results / "tools").mkdir()
    for name, run in TOOL_RUNS.items():
        (results / "tools" / name).write_text(json.dumps(run) + "\n", encoding="utf-8")

    assert cli.main(["sync", "--results", str(results)]) == 0

    output = capsys.readouterr().out
    assert "6 text result files" in output
    assert "metrics: metrics/people_probe.json" in output
    assert "tool run: tools/people_probe_rfdetr_small.json" in output
    assert "tool run: tools/pitch_probe_pnlcalib.json" in output
    assert "cost ledger: cache/ledger.jsonl" in output
    assert "cost ledger: cache/ledger-0002.jsonl" in output
    assert "cassettes: cache/SNGS-021/people (1 files)" in output
    assert "found no problem. To send the files back, the owner runs:" in output
    commands = [line.strip() for line in output.splitlines() if line.strip().startswith("git -C")]
    assert [command.split(" ", 3)[3] for command in commands] == [
        "add -- results",
        "commit -m 'Add text results (6 files)'",
        "push origin results",
    ]
    # Nothing is staged and nothing is committed: the result folder is still untracked.
    assert git(results_tree, "status", "--porcelain").split() == ["??", "results/"]
    assert git(results_tree, "branch", "--list").strip() == ""


def test_sync_names_a_run_file_of_a_tool_below_and_inside_the_folder_of_the_tools(
    results_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run file is named as a tool run in a folder below ``tools``, and when the command gets ``tools`` itself."""
    tools = results_tree / "results" / "tools"
    (tools / "cpu").mkdir(parents=True)
    detector, calibration = TOOL_RUNS
    (tools / detector).write_text(json.dumps(TOOL_RUNS[detector]) + "\n", encoding="utf-8")
    (tools / "cpu" / calibration).write_text(json.dumps(TOOL_RUNS[calibration]) + "\n", encoding="utf-8")
    # A metrics file of a tool is in the folder of the metrics files: it is no run file.
    metrics = results_tree / "results" / "metrics"
    metrics.mkdir()
    (metrics / detector).write_text(json.dumps({"n_frames": 40}) + "\n", encoding="utf-8")

    assert cli.main(["sync", "--results", str(results_tree / "results")]) == 0

    output = capsys.readouterr().out
    assert "3 text result files" in output
    assert f"  tool run: tools/{detector}" in output
    assert f"  tool run: tools/cpu/{calibration}" in output
    assert f"  metrics: metrics/{detector}" in output

    assert cli.main(["sync", "--results", str(tools)]) == 0

    output = capsys.readouterr().out
    assert "2 text result files" in output
    assert f"  tool run: {detector}" in output
    assert f"  tool run: cpu/{calibration}" in output
    assert "metrics:" not in output
    assert "add -- results/tools" in output


def test_sync_refuses_a_result_file_that_the_guard_refuses(
    results_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = results_tree / "results"
    write_results(results)
    planted = "AI" + "za" + "x" * 35
    (results / "metrics" / "notes.json").write_text(json.dumps({"note": planted}), encoding="utf-8")

    assert cli.main(["sync", "--results", str(results)]) == 1

    output = capsys.readouterr().out
    assert ": secret: " in output
    assert planted not in output
    assert "git -C" not in output


def test_sync_refuses_dataset_data_as_json_lines_and_inside_another_key(
    results_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Label data and prediction data must not go back in another shape than the label file has."""
    results = results_tree / "results"
    write_results(results)
    labels = SyntheticScene(n_frames=N_FRAMES, seed=2).gsr_labels()
    metrics = results / "metrics"
    # The annotations of the label file as JSON lines, the whole label data inside another key, and
    # prediction data on one line of a JSON lines file.
    lines = "".join(json.dumps(annotation) + "\n" for annotation in labels["annotations"])
    (metrics / "labels_lines.jsonl").write_text(lines, encoding="utf-8")
    (metrics / "labels_inside.json").write_text(json.dumps({"provenance": {}, "labels": labels}), encoding="utf-8")
    one_line = json.dumps({"predictions": labels["annotations"]}) + "\n"
    (metrics / "predictions_lines.jsonl").write_text(one_line, encoding="utf-8")

    assert cli.main(["sync", "--results", str(results)]) == 1

    output = capsys.readouterr().out
    for name in ("labels_lines.jsonl", "labels_inside.json", "predictions_lines.jsonl"):
        assert f"results/metrics/{name}:-: dataset: " in output
    assert "git -C" not in output


def test_sync_stops_when_the_guard_could_not_check_the_files(
    results_tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A guard that stops with an error found no problem, but it also checked nothing."""
    write_results(results_tree / "results")
    broken_guard = tmp_path / "broken_guard.py"
    broken_guard.write_text(
        "import sys\nprint('guard: error: no list of allowed hosts', file=sys.stderr)\nsys.exit(2)\n", encoding="utf-8"
    )
    monkeypatch.setattr(cli, "GUARD_SCRIPT", broken_guard)

    assert cli.main(["sync", "--results", str(results_tree / "results")]) == 2

    captured = capsys.readouterr()
    assert "no list of allowed hosts" in captured.err
    assert "git -C" not in captured.out


def test_sync_shows_every_result_file_to_the_guard(results_tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The guard gets the files in groups of 200. A planted key in file 206 must be found."""
    results = results_tree / "results"
    (results / "metrics").mkdir(parents=True)
    for number in range(205):
        (results / "metrics" / f"probe_{number:03d}.json").write_text('{"recall": 0.9}\n', encoding="utf-8")
    planted = "AI" + "za" + "x" * 35
    (results / "metrics" / "probe_zzz.json").write_text(json.dumps({"note": planted}), encoding="utf-8")
    assert cli.GUARD_BATCH == 200

    assert cli.main(["sync", "--results", str(results)]) == 1

    output = capsys.readouterr().out
    assert "206 text result files" in output
    assert "probe_zzz.json:1: secret: " in output
    assert "git -C" not in output


def test_sync_refuses_a_linked_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A linked folder is not entered, so its files would never be checked."""
    results = tmp_path / "results"
    write_results(results)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "frame.json").write_text("{}\n", encoding="utf-8")
    (results / "linked").symlink_to(elsewhere, target_is_directory=True)

    assert cli.main(["sync", "--results", str(results)]) == 1

    output = capsys.readouterr().out
    assert "refused: linked is not a text result file" in output
    assert "git -C" not in output


def test_sync_refuses_a_work_tree_that_is_not_on_the_results_branch(
    results_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    git(results_tree, "symbolic-ref", "HEAD", "refs/heads/main")
    write_results(results_tree / "results")

    assert cli.main(["sync", "--results", str(results_tree / "results")]) == 2

    error = capsys.readouterr().err
    assert "'main'" in error
    assert "git worktree add -b results" in error


def test_sync_refuses_a_folder_outside_git(
    private_git: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "results"
    write_results(results)

    assert cli.main(["sync", "--results", str(results)]) == 2

    assert "not inside a git work tree" in capsys.readouterr().err
