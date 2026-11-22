"""Checks of the machine and of the model API: the logic of the ``doctor`` command.

Two groups of checks live here.

- The environment checks describe the machine: versions, the model SDK, the
  key variable, the guard hooks, the deny-list of the guard and the free disk
  space.
- The contract checks are gate G0. They send requests for frames of a
  synthetic scene, whose truth is known exactly, and compare the answers with
  that truth. They take any backend, so the tests run them with the fake
  backend. With the real backend, a recording SDK client also keeps the raw
  facts of every call (the ``store`` flag, the service tier, the token counts).

The contract checks give two kinds of results for the boxes and the landmarks.
A contract verdict (``BOXES``, ``LANDMARKS``) fails for a fault of the client:
a wrong order or a wrong scale of the coordinates. It does not fail for a model
that is some pixels off or that confuses two landmarks. An accuracy value
(``BOX_ACCURACY``, ``LANDMARK_ACCURACY``) is a fact without a verdict: the mean
value next to the accuracy value of the gate file.

Every check gives a ``CheckResult``. The command line prints the results and
decides the exit code. No text of a result holds an absolute path or the value
of the key, so a report can go to the results branch.
"""

from __future__ import annotations

import dataclasses
import math
import os
import platform
import shutil
import subprocess
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import pydantic
import scipy
import yaml

import pitchvlm
from pitchvlm.config import StepSettings
from pitchvlm.evalx.metrics import match_boxes
from pitchvlm.geometry import apply_homography
from pitchvlm.pitch import landmark_xy
from pitchvlm.synth import SyntheticScene
from pitchvlm.vlm import steps
from pitchvlm.vlm.client import (
    SDK_DISTRIBUTION,
    InteractionsClient,
    build_interaction_kwargs,
    installed_sdk_version,
    normalize_usage,
    usage_accounting,
)
from pitchvlm.vlm.cost import has_flex_tier
from pitchvlm.vlm.types import Backend, BackendError, FlexBusyError, VLMRequest, VLMResponse

#: ``ok`` and ``fail`` are verdicts. ``info`` is a fact without a verdict.
Status = Literal["ok", "fail", "info"]

# --- Names of the checks -------------------------------------------------------------

PYTHON = "python"
PACKAGES = "packages"
SDK = "model SDK"
KEY = "API key"
HOOK = "guard hooks"
DENY = "deny-list"
DISK = "free disk space"
BOXES = "boxes"
BOX_ACCURACY = "box accuracy"
LANDMARKS = "landmarks"
LANDMARK_ACCURACY = "landmark accuracy"
FLEX = "flex tier"
STORE = "store flag"
USAGE = "token sum"
SERVED_TIER = "served tier"
SPEND = "spend"

# --- Environment ---------------------------------------------------------------------

#: The two git hooks that the repository needs. Each one runs the guard: one before
#: every commit, the other before every push.
HOOK_NAMES: tuple[str, ...] = ("pre-commit", "pre-push")
#: The file of the repository that lists the hooks for the pre-commit framework.
HOOK_CONFIG = Path(".pre-commit-config.yaml")
#: Command that installs both hooks.
HOOK_INSTALL_COMMAND = "pre-commit install --hook-type pre-commit --hook-type pre-push"
#: A hook file that the pre-commit framework wrote calls its command ``hook-impl``.
_FRAMEWORK_COMMAND = "hook-impl"
#: Start of the command line of each kind of hook that the hook list must hold for both hooks.
_GUARD_ENTRY = "tools/guard.py"
#: Name of the deny-list of the guard. It lies in the git folder of the clone (``tools/guard.py``).
DENY_LIST_NAME = "guard-deny.txt"
#: The major version of the model SDK that the client is written for (``pyproject.toml``: from 3.0, below 4).
#: An older SDK reads ``attempts=1`` as one repeat, so it sends a failed call a second time by itself.
SDK_MAJOR_VERSION = 3
_GB = 10**9

# --- Contract checks -----------------------------------------------------------------

#: Size of the synthetic frames, pixels (1080p, the size of the real footage).
CONTRACT_WIDTH = 1920
CONTRACT_HEIGHT = 1080
#: Frames of the synthetic clip that are sent. At 25 frames per second the camera shows the
#: left side at frame 0, the halfway line at frame 75 and the right side at frame 150.
CONTRACT_FRAMES: tuple[int, ...] = (0, 75, 150)
CONTRACT_SEED = 0
#: Number of requests that go through the flex tier.
FLEX_REQUESTS = 1
#: An answered box and a true box count as the same person when their IoU is at least this.
#: The value is low on purpose: a badly placed box must lower the IoU values, not leave them.
MATCH_MIN_IOU = 0.1
#: Limits of the contract verdict of the boxes: the smallest share of the possible pairs
#: that must exist, and the smallest median IoU of the pairs. Measured on the contract
#: frames: answers with x and y swapped pair 2 of 44 boxes, with an IoU of 0.13. Answers
#: in the right order with 6 pixels of noise on every box edge pair 43 of 44 boxes, with a
#: median IoU of 0.57.
CONTRACT_MIN_PAIRED_SHARE = 0.5
CONTRACT_MIN_MEDIAN_IOU = 0.3
#: Limit of the contract verdict of the landmarks: the largest median point error, as a
#: share of the image side. Measured on the contract frames: answers with x and y swapped
#: have a median error of 0.39. Answers in the right order with 20 pixels of noise on
#: every coordinate have a median error of 0.014.
CONTRACT_MAX_MEDIAN_POINT_ERROR = 0.05


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one check."""

    name: str
    status: Status
    #: One line for people. It holds no absolute path and no secret.
    detail: str
    #: The measured value and its limit, for a check that compares a number with a gate.
    value: float | None = None
    limit: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the result as plain JSON data."""
        return dataclasses.asdict(self)


def any_failed(checks: Sequence[CheckResult]) -> bool:
    """Tell whether at least one check failed."""
    return any(check.status == "fail" for check in checks)


# --- Environment checks --------------------------------------------------------------


def home_relative(path: Path) -> str:
    """Write a path with ``~`` for the home folder, so that the text holds no user name.

    A path outside the home folder gives its last part only.
    """
    try:
        relative = path.resolve().relative_to(Path.home().resolve())
    except ValueError:
        return path.name
    return "~" if relative == Path(".") else f"~/{relative.as_posix()}"


def environment_checks(repo: Path, *, key_variable: str, data_dir: Path, api: bool) -> list[CheckResult]:
    """Describe the machine: versions, the model SDK, the key variable, the guard and the free disk space.

    ``repo`` is the folder of the repository. ``data_dir`` is the data folder; the free
    disk space is read there, or in the home folder when the data folder does not exist.

    ``api`` says whether the caller will call the model. The SDK and the key are needed
    only then: without them, or with an SDK of another major version, the check fails
    when ``api`` is true, and it is a fact without a verdict when ``api`` is false. The
    guard hooks always fail when git does not run the guard before
    every commit and before every push (``_hook_check``). The deny-list always fails
    when the clone has none (``_deny_check``). The value of the key and the entries of
    the deny-list never go into a result.
    """
    return [
        _python_check(),
        _packages_check(),
        _sdk_check(api),
        _key_check(key_variable, api),
        _hook_check(repo),
        _deny_check(repo),
        _disk_check(data_dir),
    ]


def _python_check() -> CheckResult:
    """Name the Python version and the kind of machine."""
    return CheckResult(PYTHON, "info", f"{platform.python_version()} on {platform.system()} {platform.machine()}")


def _packages_check() -> CheckResult:
    """Name the version of this package and of the packages that it needs."""
    versions = {
        "pitchvlm": pitchvlm.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "scipy": scipy.__version__,
        "pydantic": pydantic.VERSION,
        "pyyaml": yaml.__version__,
    }
    return CheckResult(PACKAGES, "info", ", ".join(f"{name} {version}" for name, version in versions.items()))


def _sdk_check(api: bool) -> CheckResult:
    """Tell whether the model SDK is installed in the major version that the client is written for.

    A missing SDK and an SDK of another major version fail when the model is called, and
    are a fact without a verdict otherwise.
    """
    version = installed_sdk_version()
    unusable: Status = "fail" if api else "info"
    if version is None:
        return CheckResult(SDK, unusable, f"{SDK_DISTRIBUTION} is not installed")
    major = version.split(".")[0]
    if major == str(SDK_MAJOR_VERSION):
        return CheckResult(SDK, "ok", f"{SDK_DISTRIBUTION} {version}")
    return CheckResult(
        SDK,
        unusable,
        f"{SDK_DISTRIBUTION} {version} is not a version {SDK_MAJOR_VERSION}.x: the client needs "
        f"{SDK_MAJOR_VERSION}.0 or later and below {SDK_MAJOR_VERSION + 1}. An older SDK repeats a failed call "
        "by itself, although the client asks for one attempt",
    )


def _key_check(key_variable: str, api: bool) -> CheckResult:
    """Tell whether the key variable is set. The value is not read into the result."""
    if os.environ.get(key_variable, "").strip():
        return CheckResult(KEY, "ok", f"the variable {key_variable} is set")
    return CheckResult(KEY, "fail" if api else "info", f"the variable {key_variable} is not set")


def _hook_check(repo: Path) -> CheckResult:
    """Tell whether git runs the guard of the repository before every commit and every push.

    A file with the name of a hook is not enough: git must run it, and it must run the
    guard. So the check needs all of this, for the hook before a commit and for the hook
    before a push:

    - The hook list ``.pre-commit-config.yaml`` of the repository names a hook that runs
      the guard script.
    - The hook file is at the place where git looks for it. Git names this place, so the
      setting ``core.hooksPath`` and a linked work tree are followed.
    - The hook file is executable. Git ignores a hook file that is not.
    - The hook file is the one that the pre-commit framework wrote for this hook and for
      this hook list. A hook of another tool does not run the guard.
    """
    problems = _hook_list_problems(repo)
    places: list[str] = []
    for name in HOOK_NAMES:
        place = _git_hook_place(repo, name)
        if place is None:
            problems.append(
                f"git does not name the place of the hook {name}: the folder is not a git work tree, "
                "or git is not installed"
            )
            continue
        shown = _shown_place(place, repo)
        places.append(shown)
        problem = _hook_file_problem(place, name)
        if problem is not None:
            problems.append(f"{shown} {problem}")
    if not problems:
        return CheckResult(
            HOOK,
            "ok",
            f"git runs the guard before every commit and every push ({' and '.join(places)})",
        )
    return CheckResult(HOOK, "fail", f"{'; '.join(problems)}. Install both hooks with {HOOK_INSTALL_COMMAND}")


def _git_place(repo: Path, *question: str) -> Path | None:
    """Ask git for a place of a repository with ``git rev-parse``. None means that git gave no answer.

    Git gives no answer when the folder is not in a git work tree and when git is not
    installed. A relative answer is relative to the repository folder.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", *question],
            cwd=repo,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
    except OSError:
        return None
    answer = completed.stdout.strip()
    if completed.returncode != 0 or not answer:
        return None
    return repo / answer


def _git_hook_place(repo: Path, name: str) -> Path | None:
    """Ask git where it looks for one hook of a repository. None means that git gave no answer."""
    return _git_place(repo, "--git-path", f"hooks/{name}")


def _shown_place(place: Path, repo: Path) -> str:
    """Write the place of a file of the clone for a check result.

    The place is relative to the repository. A place outside the repository is written without the user name.
    """
    try:
        return place.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return home_relative(place)


def _hook_file_problem(place: Path, name: str) -> str | None:
    """Say what is wrong with the file of one hook, as the end of a sentence. None means that nothing is wrong."""
    if not place.is_file():
        return "is missing"
    if not os.access(place, os.X_OK):
        return "is not executable, so git does not run it"
    try:
        text = place.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "cannot be read"
    marks = (_FRAMEWORK_COMMAND, f"--config={HOOK_CONFIG.as_posix()}", f"--hook-type={name}")
    if not all(mark in text for mark in marks):
        return (
            f"is not the {name} hook that the pre-commit framework writes for {HOOK_CONFIG.as_posix()}, "
            "so it does not run the guard"
        )
    return None


def _hook_list_problems(repo: Path) -> list[str]:
    """Say what the hook list of the repository lacks: a guard hook for one of the two hooks."""
    name = HOOK_CONFIG.as_posix()
    try:
        config = yaml.safe_load((repo / HOOK_CONFIG).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [f"the hook list {name} is missing"]
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return [f"the hook list {name} cannot be read"]
    entries = _hook_entries(config)
    if entries is None:
        return [f"{name} is not a hook list of the pre-commit framework"]
    problems: list[str] = []
    for hook_name in HOOK_NAMES:
        if not any(_GUARD_ENTRY in entry for entry in entries[hook_name]):
            problems.append(f"{name} names no hook that runs {_GUARD_ENTRY} at the stage {hook_name}")
    return problems


def _hook_entries(config: Any) -> dict[str, list[str]] | None:
    """Return the command lines that a hook list runs at each of the two hooks. None means that it is no hook list.

    A hook runs at the stages that it names. A hook that names none runs at the stages
    of ``default_stages``, or at every stage when the list has no such key.
    """
    if not isinstance(config, dict) or not isinstance(config.get("repos"), list):
        return None
    default_stages = config.get("default_stages")
    entries: dict[str, list[str]] = {name: [] for name in HOOK_NAMES}
    for source in config["repos"]:
        hooks = source.get("hooks") if isinstance(source, dict) else None
        if not isinstance(hooks, list) or not all(isinstance(hook, dict) for hook in hooks):
            return None
        for hook in hooks:
            entry, stages = hook.get("entry"), hook.get("stages", default_stages)
            if not isinstance(entry, str):
                continue
            for name in HOOK_NAMES:
                if stages is None or (isinstance(stages, list) and name in stages):
                    entries[name].append(entry)
    return entries


def _deny_check(repo: Path) -> CheckResult:
    """Tell whether the clone has the deny-list of the guard, and how many entries the list holds.

    The guard knows the private words of a machine only from this file. It lies in the
    git folder that all work trees of the clone share. Without the file the rule of the
    guard is off, so the check fails. A file without entries passes: the owner made it
    and found nothing to list. The detail never holds an entry.
    """
    git_folder = _git_place(repo, "--git-common-dir")
    if git_folder is None:
        return CheckResult(
            DENY, "fail", "git does not name its folder: the folder is not a git work tree, or git is not installed"
        )
    deny_list = git_folder / DENY_LIST_NAME
    shown = _shown_place(deny_list, repo)
    if not deny_list.is_file():
        return CheckResult(
            DENY,
            "fail",
            f"{shown} is missing: write the private words of this machine into it (docs/RUNBOOK.md, section 9)",
        )
    try:
        lines = [line.strip() for line in deny_list.read_text(encoding="utf-8-sig").split("\n")]
    except (OSError, UnicodeDecodeError):
        return CheckResult(DENY, "fail", f"{shown} cannot be read")
    n_entries = sum(1 for line in lines if line and not line.startswith("#"))
    return CheckResult(DENY, "ok", f"{shown} has {n_entries} entries")


def _disk_check(data_dir: Path) -> CheckResult:
    """Name the free disk space of the data folder, or of the home folder when there is no data folder."""
    place = data_dir if data_dir.is_dir() else Path.home()
    free_gb = shutil.disk_usage(place).free / _GB
    return CheckResult(DISK, "info", f"{free_gb:.1f} GB at {home_relative(place)}")


# --- Raw facts of the calls ----------------------------------------------------------


@dataclass(frozen=True)
class RawCall:
    """The raw facts of one call of the SDK client: what was sent and what came back, without text and images."""

    #: Value of the ``store`` argument of the call. None means that the call had no such argument.
    store: Any
    #: Service tier that the call asked for.
    sent_tier: str | None
    #: Status of the answer.
    status: str | None
    #: Service tier that the answer reports, or None when it reports none.
    served_tier: str | None
    #: Token counters of the answer as the API reports them. None means that the counter is missing.
    input_tokens: int | None
    output_tokens: int | None
    thought_tokens: int | None
    total_tokens: int | None


def _text(value: Any) -> str | None:
    """Return a value of the SDK as text. An enumeration member gives its value. None stays None."""
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _count(usage: Any, name: str) -> int | None:
    """Read one token counter of the usage object of the API. Anything but a whole number from 0 gives None."""
    value = getattr(usage, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


class RecordingClient:
    """An SDK client that passes every call on and keeps its raw facts.

    It wraps anything with ``.interactions.create``, for example the client of
    ``client.build_sdk_client``. Give it to ``GeminiBackend(client=...)``. A call that
    raises is not recorded. One instance can be used from several threads.
    """

    def __init__(self, inner: InteractionsClient) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        #: The recorded calls, in the order in which their answers came back.
        self.calls: list[RawCall] = []

    @property
    def interactions(self) -> RecordingClient:
        """Return the object that has ``create``: this client itself."""
        return self

    def create(self, **kwargs: Any) -> Any:
        """Send one call through the wrapped client, record its raw facts and return its answer unchanged."""
        interaction = self._inner.interactions.create(**kwargs)
        usage = getattr(interaction, "usage", None)
        call = RawCall(
            store=kwargs.get("store"),
            sent_tier=_text(kwargs.get("service_tier")),
            status=_text(getattr(interaction, "status", None)),
            served_tier=_text(getattr(interaction, "service_tier", None)),
            input_tokens=_count(usage, "total_input_tokens"),
            output_tokens=_count(usage, "total_output_tokens"),
            thought_tokens=_count(usage, "total_thought_tokens"),
            total_tokens=_count(usage, "total_tokens"),
        )
        with self._lock:
            self.calls.append(call)
        return interaction


class AnswerLog:
    """A backend that passes every request on and keeps the requests and their answers."""

    def __init__(self, inner: Backend) -> None:
        self._inner = inner
        #: Every request, also one whose call failed.
        self.requests: list[VLMRequest] = []
        #: Every answered request with its answer.
        self.exchanges: list[tuple[VLMRequest, VLMResponse]] = []

    def generate(self, request: VLMRequest) -> VLMResponse:
        """Send one request through the wrapped backend and keep it with its answer."""
        self.requests.append(request)
        response = self._inner.generate(request)
        self.exchanges.append((request, response))
        return response


# --- Contract checks (gate G0) -------------------------------------------------------


@dataclass(frozen=True)
class ContractLimits:
    """The thresholds of gate G0 that the contract checks use. They come from ``configs/gates.yaml``.

    The first two are accuracy values: the report compares the measured mean with them
    and gives no verdict. The limits of the contract verdicts of the boxes and the
    landmarks are constants of this module (``CONTRACT_MIN_PAIRED_SHARE``,
    ``CONTRACT_MIN_MEDIAN_IOU``, ``CONTRACT_MAX_MEDIAN_POINT_ERROR``).
    """

    #: Accuracy value of the boxes: the mean IoU between the answered boxes and the true boxes should reach it.
    box_mean_iou_min: float
    #: Accuracy value of the landmarks: the mean point error, as a share of the image side, should stay below it.
    point_error_max_fraction: float
    #: Largest gap between the reported token total and the sum of its parts, as a share of the total.
    usage_sum_tolerance: float
    #: Smallest share of the flex requests that must give a valid answer.
    flex_valid_answer_rate_min: float


@dataclass(frozen=True)
class ContractResult:
    """The contract checks of one model."""

    checks: tuple[CheckResult, ...]
    #: One entry per answered request: the step, the frame, the answer text and the token usage.
    answers: tuple[dict[str, Any], ...]


def contract_scene() -> SyntheticScene:
    """Return the synthetic scene of the contract checks: 1920 x 1080 pixels, long enough for all its frames."""
    return SyntheticScene(
        n_frames=CONTRACT_FRAMES[-1] + 1, width=CONTRACT_WIDTH, height=CONTRACT_HEIGHT, seed=CONTRACT_SEED
    )


def flex_settings(settings: StepSettings) -> StepSettings:
    """Return the same step settings with the flex service tier."""
    return StepSettings(
        model=settings.model,
        thinking_level=settings.thinking_level,
        resolution=settings.resolution,
        service_tier="flex",
    )


def contract_plan(
    settings: StepSettings, frames: Sequence[int] = CONTRACT_FRAMES
) -> list[tuple[str, StepSettings, int]]:
    """List the model calls of ``contract_checks`` as (step, settings, number of requests), for the cost estimate.

    The requests through the flex tier are planned only for a model that has that tier
    (``cost.FLEX_TIER_MODELS``).
    """
    plan = [("people", settings, len(frames)), ("landmarks", settings, len(frames))]
    if has_flex_tier(settings.model):
        plan.append(("people", flex_settings(settings), len(frames[:FLEX_REQUESTS])))
    return plan


def contract_checks(
    backend: Backend,
    scene: SyntheticScene,
    settings: StepSettings,
    limits: ContractLimits,
    *,
    recorder: RecordingClient | None = None,
    frames: Sequence[int] = CONTRACT_FRAMES,
) -> ContractResult:
    """Run the contract checks of gate G0 for one model, with requests for frames of a synthetic scene.

    ``settings`` names the model. The checks are:

    - boxes and box accuracy: the people step against the true boxes (``check_boxes``);
    - landmarks and landmark accuracy: the landmarks step against the true points
      (``check_landmarks``);
    - flex tier: people requests through the flex tier give a valid answer
      (``check_flex``). A model without a flex tier gets no such request, and the check
      is a fact without a verdict. It is one too when the flex tier had no capacity for
      any of the requests: a busy flex tier is no fault of the client;
    - store flag: no request is stored by the provider (``check_store_flag``);
    - token sum and served tier: read from the raw calls (``check_usage_sums``,
      ``check_served_tiers``). The served tier is checked for every call, also for the
      calls with the standard tier.

    The boxes check and the landmarks check are contract verdicts: they fail for a wrong
    order or a wrong scale of the coordinates. The two accuracy values are facts
    without a verdict.

    ``recorder`` is the recording SDK client below ``backend``, when there is one. Only
    the calls that it records during this function are used. Without a recorder the
    checks of the raw calls give a fact without a verdict.

    A request that fails and an answer that cannot be parsed make their check fail. Any
    other error, for example a reached budget limit, is raised.
    """
    _need_frames(frames)
    first_call = 0 if recorder is None else len(recorder.calls)
    log = AnswerLog(backend)
    checks = [
        *_attempt(BOXES, lambda: check_boxes(log, scene, settings, frames, min_mean_iou=limits.box_mean_iou_min)),
        *_attempt(
            LANDMARKS,
            lambda: check_landmarks(
                log, scene, settings, frames, max_error_fraction=limits.point_error_max_fraction
            ),
        ),
    ]
    if has_flex_tier(settings.model):
        checks.append(
            check_flex(
                log, scene, settings, frames[:FLEX_REQUESTS], min_valid_rate=limits.flex_valid_answer_rate_min
            )
        )
    else:
        detail = f"not checked: the model {settings.model} has no flex tier, so no request used it"
        checks.append(CheckResult(FLEX, "info", detail))
    calls = [] if recorder is None else recorder.calls[first_call:]
    checks += [
        check_store_flag(log.requests, calls),
        check_usage_sums(calls, tolerance=limits.usage_sum_tolerance),
        check_served_tiers(calls),
    ]
    answers = tuple(_answer_record(request, response) for request, response in log.exchanges)
    return ContractResult(checks=tuple(checks), answers=answers)


def _need_frames(frames: Sequence[int]) -> None:
    """Raise ``ValueError`` when a check gets no frame."""
    if len(frames) == 0:
        raise ValueError("a contract check needs at least one frame")


def _attempt(name: str, check: Callable[[], list[CheckResult]]) -> list[CheckResult]:
    """Run the checks of one step. A request that fails and an answer that cannot be parsed give one failed check."""
    try:
        return check()
    except BackendError as error:
        return [CheckResult(name, "fail", f"the request failed: {error}")]
    except steps.ParseError as error:
        return [CheckResult(name, "fail", f"the answer cannot be parsed: {error}")]


def _answer_record(request: VLMRequest, response: VLMResponse) -> dict[str, Any]:
    """Return one answered request as plain JSON data, without the images."""
    return {
        "step": request.meta.step,
        "frame": request.meta.frame,
        "service_tier": request.service_tier,
        "text": response.text,
        "usage": response.usage.model_dump(),
        "latency_s": response.latency_s,
        "attempts": response.attempts,
    }


def check_boxes(
    backend: Backend, scene: SyntheticScene, settings: StepSettings, frames: Sequence[int], *, min_mean_iou: float
) -> list[CheckResult]:
    """Ask for the people of synthetic frames and compare the answered boxes with the true boxes.

    The answered boxes of a frame are paired with the true boxes one to one (the pairing
    with the largest total IoU; a pair needs an IoU of at least ``MATCH_MIN_IOU``). The
    result holds two checks.

    ``BOXES`` is the contract verdict: do the answered boxes land on the true boxes? A
    wrong order or a wrong scale of the coordinates puts almost no box on a true box. A
    model that is some pixels off still gives pairs. The check fails in these cases:

    - a frame has no pair at all;
    - fewer than ``CONTRACT_MIN_PAIRED_SHARE`` of the possible pairs exist. A frame can
      give as many pairs as the smaller of its two box counts. So a person that the
      model does not find, and a box where nobody is, do not count against the contract;
    - the median IoU of the pairs, which is the value of the check, is below
      ``CONTRACT_MIN_MEDIAN_IOU``.

    ``BOX_ACCURACY`` is a fact without a verdict: the mean IoU of the pairs, next to the
    accuracy value ``min_mean_iou`` of the gate file. The boxes of the synthetic people
    are small, so a model that follows the contract can stay below this value. The
    detail text also names the true boxes without a pair, and the mean IoU over all
    true boxes when a box without a pair counts as 0.

    When a frame has no pair, the result holds the failed contract verdict only.
    """
    _need_frames(frames)
    info = scene.info
    legend = scene.kit_legend()
    ious: list[float] = []
    n_true = n_possible = 0
    for frame in frames:
        request = steps.build_people_request(
            scene.image(frame), legend, clip_id=info.clip_id, frame=frame, settings=settings
        )
        answer = steps.parse_people(backend.generate(request).text, frame=frame, width=info.width, height=info.height)
        answered = [person.box for person in answer.people]
        true_boxes = [person.box for person in scene.people(frame).people]
        pairs = match_boxes(answered, true_boxes, MATCH_MIN_IOU)  # (index in answered, index in true_boxes)
        if not pairs:
            return [
                CheckResult(
                    BOXES,
                    "fail",
                    f"frame {frame}: none of the {len(answered)} answered boxes overlaps one of the "
                    f"{len(true_boxes)} true boxes",
                )
            ]
        ious.extend(answered[answered_index].iou(true_boxes[true_index]) for answered_index, true_index in pairs)
        n_true += len(true_boxes)
        n_possible += min(len(answered), len(true_boxes))
    paired_share = len(ious) / n_possible
    median_iou = float(np.median(ious))
    follows_contract = paired_share >= CONTRACT_MIN_PAIRED_SHARE and median_iou >= CONTRACT_MIN_MEDIAN_IOU
    contract = CheckResult(
        BOXES,
        "ok" if follows_contract else "fail",
        f"{len(ious)} of {n_possible} possible pairs of an answered box and a true box exist in {len(frames)} frames "
        f"(limit: a share of at least {CONTRACT_MIN_PAIRED_SHARE:g}), and their median IoU is {median_iou:.3f} "
        f"(limit: at least {CONTRACT_MIN_MEDIAN_IOU:g})",
        value=median_iou,
        limit=CONTRACT_MIN_MEDIAN_IOU,
    )
    mean_iou = math.fsum(ious) / len(ious)
    reached = "reaches" if mean_iou >= min_mean_iou else "does not reach"
    accuracy = CheckResult(
        BOX_ACCURACY,
        "info",
        f"mean IoU {mean_iou:.3f} over {len(ious)} paired boxes: it {reached} the accuracy value of the gate file "
        f"(at least {min_mean_iou:g}). This is no verdict: the boxes check is the verdict on the client. "
        f"{len(ious)} of {n_true} true boxes have a pair, and the mean over all true boxes is "
        f"{math.fsum(ious) / n_true:.3f}",
        value=mean_iou,
        limit=min_mean_iou,
    )
    return [contract, accuracy]


def check_landmarks(
    backend: Backend,
    scene: SyntheticScene,
    settings: StepSettings,
    frames: Sequence[int],
    *,
    max_error_fraction: float,
) -> list[CheckResult]:
    """Ask for the pitch landmarks of synthetic frames and compare the answered points with the true points.

    The true point of a landmark is its pitch position seen through the true camera of
    the frame. So a landmark that the model names although it lies outside the picture
    has a large error. The error of a point is its distance to the true point, where x
    differences are divided by the image width and y differences by the image height
    (the units of the model coordinates). The result holds two checks.

    ``LANDMARKS`` is the contract verdict. Its value is the median error over all
    answered landmarks of all frames. A wrong order or a wrong scale of the coordinates
    moves most points far away, so the check fails when the median is above
    ``CONTRACT_MAX_MEDIAN_POINT_ERROR``. Noise of some pixels and a few landmarks with a
    wrong id do not move the median that far. The check also fails when the answer of a
    frame names no known landmark; the result then holds this failed check only.

    ``LANDMARK_ACCURACY`` is a fact without a verdict: the mean error, next to the
    accuracy value ``max_error_fraction`` of the gate file, and the number of landmarks
    whose error is above ``CONTRACT_MAX_MEDIAN_POINT_ERROR``.
    """
    _need_frames(frames)
    info = scene.info
    image_sides = np.array([info.width, info.height], dtype=np.float64)
    errors: list[float] = []
    n_visible = 0
    for frame in frames:
        request = steps.build_landmarks_request(
            scene.image(frame), clip_id=info.clip_id, frame=frame, settings=settings
        )
        answer = steps.parse_landmarks(
            backend.generate(request).text, frame=frame, width=info.width, height=info.height
        )
        if not answer.landmarks:
            return [CheckResult(LANDMARKS, "fail", f"frame {frame}: the answer names no known landmark")]
        ids = [seen.landmark_id for seen in answer.landmarks]
        answered = np.array([[seen.x, seen.y] for seen in answer.landmarks], dtype=np.float64)  # (n, 2) pixels
        pitch_to_image = np.linalg.inv(scene.homography(frame))
        true_points = apply_homography(pitch_to_image, landmark_xy(ids))  # (n, 2) pixels
        gaps = (answered - true_points) / image_sides  # (n, 2): x as a share of the width, y of the height
        errors.extend(np.hypot(gaps[:, 0], gaps[:, 1]).tolist())
        n_visible += len(scene.landmarks(frame).landmarks)
    median_error = float(np.median(errors))
    contract = CheckResult(
        LANDMARKS,
        "ok" if median_error <= CONTRACT_MAX_MEDIAN_POINT_ERROR else "fail",
        f"median point error {median_error:.4f} of the image side over {len(errors)} answered landmarks in "
        f"{len(frames)} frames (limit: at most {CONTRACT_MAX_MEDIAN_POINT_ERROR:g}); {n_visible} landmarks are visible",
        value=median_error,
        limit=CONTRACT_MAX_MEDIAN_POINT_ERROR,
    )
    mean_error = math.fsum(errors) / len(errors)
    n_far = sum(1 for error in errors if error > CONTRACT_MAX_MEDIAN_POINT_ERROR)
    reached = "reaches" if mean_error <= max_error_fraction else "does not reach"
    accuracy = CheckResult(
        LANDMARK_ACCURACY,
        "info",
        f"mean point error {mean_error:.4f} of the image side over {len(errors)} answered landmarks: it {reached} "
        f"the accuracy value of the gate file (at most {max_error_fraction:g}). This is no verdict: the landmarks "
        f"check is the verdict on the client. {n_far} landmarks are further than "
        f"{CONTRACT_MAX_MEDIAN_POINT_ERROR:g} from their true point (a wrong id, or a point outside the picture)",
        value=mean_error,
        limit=max_error_fraction,
    )
    return [contract, accuracy]


def check_flex(
    backend: Backend, scene: SyntheticScene, settings: StepSettings, frames: Sequence[int], *, min_valid_rate: float
) -> CheckResult:
    """Send one people request per frame through the flex tier and count the valid answers.

    An answer is valid when the call succeeds and the answer can be parsed. The value of
    the check is the share of valid answers. The check fails when it is below
    ``min_valid_rate``. The detail text names the time that every answer took, so that a
    slow flex tier shows also when the check passes.

    A request for which the flex tier had no capacity (``FlexBusyError``) is not counted.
    The provider documents the status codes 503 and 429 as the normal answers of a busy
    flex tier, the backend has tried such a request again for its whole waiting time,
    and no attempt was billed. So this outcome says nothing about the client. When no
    request found capacity, the result is a fact without a verdict: the flex tier was
    not checked, and the detail text says to run the checks again.
    """
    _need_frames(frames)
    info = scene.info
    legend = scene.kit_legend()
    flex = flex_settings(settings)
    problems: list[str] = []
    without_capacity: list[str] = []
    latencies_s: list[float] = []
    for frame in frames:
        request = steps.build_people_request(
            scene.image(frame), legend, clip_id=info.clip_id, frame=frame, settings=flex
        )
        try:
            response = backend.generate(request)
            latencies_s.append(response.latency_s)
            steps.parse_people(response.text, frame=frame, width=info.width, height=info.height)
        except FlexBusyError as error:
            without_capacity.append(f"frame {frame}: {error}")
        except (BackendError, steps.ParseError) as error:
            problems.append(f"frame {frame}: {error}")
    n_counted = len(frames) - len(without_capacity)
    if n_counted == 0:
        return CheckResult(
            FLEX,
            "info",
            f"not checked: the flex tier had no capacity for any of the {len(frames)} requests. This is the "
            "documented answer of a busy flex tier and no fault of the client, and no attempt was billed. Run "
            "the doctor again before the first run through the flex tier; " + "; ".join(without_capacity),
        )
    n_valid = n_counted - len(problems)
    rate = n_valid / n_counted
    detail = (
        f"{n_valid} of {n_counted} requests through the flex tier gave a valid answer "
        f"(limit: a share of at least {min_valid_rate:g})"
    )
    if latencies_s:
        detail += "; time of each answer: " + ", ".join(f"{latency_s:.1f} s" for latency_s in latencies_s)
    if without_capacity:
        detail += (
            f"; {len(without_capacity)} more requests are not counted, because the flex tier had no capacity "
            "for them: " + "; ".join(without_capacity)
        )
    if problems:
        detail += "; " + "; ".join(problems)
    return CheckResult(FLEX, "ok" if rate >= min_valid_rate else "fail", detail, value=rate, limit=min_valid_rate)


def check_store_flag(requests: Sequence[VLMRequest], calls: Sequence[RawCall]) -> CheckResult:
    """Check that no request is stored on the side of the provider.

    Every request must give a call with ``store=False`` (``client.build_interaction_kwargs``),
    and every recorded call must have been sent with ``store=False``. Without recorded
    calls only the first part is checked, and the detail text says so.
    """
    built_wrong = sum(1 for request in requests if build_interaction_kwargs(request).get("store") is not False)
    sent_wrong = sum(1 for call in calls if call.store is not False)
    if built_wrong or sent_wrong:
        return CheckResult(
            STORE,
            "fail",
            f"{built_wrong} of {len(requests)} requests give a call without store=False; "
            f"{sent_wrong} of {len(calls)} recorded calls were sent without store=False",
        )
    sent = f"{len(calls)} recorded calls were sent with store=False" if calls else "no raw call was recorded"
    return CheckResult(STORE, "ok", f"{len(requests)} requests give a call with store=False; {sent}")


def check_usage_sums(calls: Sequence[RawCall], *, tolerance: float) -> CheckResult:
    """Check that the reported token total of every recorded call equals the sum of the tokens that are billed.

    The billed tokens are those of ``client.normalize_usage``: input tokens, output tokens
    without thinking tokens, and thinking tokens. The API may report the output count
    with or without the thinking tokens; both add up. The value of the check is the
    largest gap between the total and the sum, as a share of the total. The check fails
    when the gap is above ``tolerance``, and when a call reports no total or no input
    count. Without recorded calls the result is a fact without a verdict.
    """
    if not calls:
        return CheckResult(USAGE, "info", "not checked: no raw call was recorded")
    gaps: list[float] = []
    accountings: Counter[str] = Counter()
    for number, call in enumerate(calls, start=1):
        if call.input_tokens is None or not call.total_tokens:
            return CheckResult(
                USAGE, "fail", f"call {number} of {len(calls)} reports no token total or no input tokens"
            )
        output_tokens, thought_tokens = call.output_tokens or 0, call.thought_tokens or 0
        accountings[usage_accounting(call.input_tokens, output_tokens, thought_tokens, call.total_tokens)] += 1
        billed = normalize_usage(
            input_tokens=call.input_tokens,
            output_tokens=output_tokens,
            thought_tokens=thought_tokens,
            cached_tokens=0,
            total_tokens=call.total_tokens,
        )
        billed_sum = billed.input_tokens + billed.output_tokens + billed.thought_tokens
        gaps.append(abs(call.total_tokens - billed_sum) / call.total_tokens)
    worst = max(gaps)
    counted = ", ".join(f"{count} {accounting}" for accounting, count in sorted(accountings.items()))
    detail = (
        f"largest gap between the token total and the billed sum: {worst:.4f} of the total over {len(calls)} calls "
        f"(limit: at most {tolerance:g}); thinking tokens in the output count: {counted}"
    )
    return CheckResult(USAGE, "ok" if worst <= tolerance else "fail", detail, value=worst, limit=tolerance)


def check_served_tiers(calls: Sequence[RawCall]) -> CheckResult:
    """Check that every recorded call was served by the service tier that it asked for.

    The price of a call follows the tier that served it. So the check fails when the
    answer of a call reports another tier than the call asked for: a flex call that the
    standard tier served, and also a standard call that a more expensive tier served.
    The detail names every such pair. The result is a fact without a verdict when no
    call was recorded, and when no answer reports its tier.

    The tier of an answer is read from the field ``service_tier`` of its body. The first
    run with the real API must show that this field names the tier that served the
    call, and does not only repeat the request (``docs/RUNBOOK.md``, section 8).
    """
    if not calls:
        return CheckResult(SERVED_TIER, "info", "not checked: no raw call was recorded")
    reported = [
        (call.sent_tier.lower(), call.served_tier.lower())
        for call in calls
        if call.sent_tier is not None and call.served_tier is not None
    ]
    if not reported:
        return CheckResult(SERVED_TIER, "info", f"none of the {len(calls)} answers reports the tier that served it")
    others = Counter((sent, served) for sent, served in reported if sent != served)
    if others:
        pairs = "; ".join(
            f"asked for {sent}, served by {served} ({count} calls)" for (sent, served), count in sorted(others.items())
        )
        return CheckResult(
            SERVED_TIER,
            "fail",
            f"{sum(others.values())} of {len(reported)} answers report another tier than their call asked for: {pairs}",
        )
    asked = Counter(sent for sent, _ in reported)
    counted = ", ".join(f"{count} {tier}" for tier, count in sorted(asked.items()))
    detail = f"{len(reported)} of {len(calls)} answers report the tier that their call asked for ({counted})"
    return CheckResult(SERVED_TIER, "ok", detail)


def check_spend(measured_usd: float, max_usd: float) -> CheckResult:
    """Check that the contract checks cost at most ``max_usd`` in total."""
    status: Status = "ok" if measured_usd <= max_usd else "fail"
    detail = f"the contract checks cost {measured_usd:.3f} USD (limit: at most {max_usd:g} USD)"
    return CheckResult(SPEND, status, detail, value=measured_usd, limit=max_usd)
