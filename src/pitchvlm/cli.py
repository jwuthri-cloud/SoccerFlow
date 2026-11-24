"""Command line of pitchvlm.

Run ``python -m pitchvlm.cli <command> --help`` to see the options of a command.

Commands:

- ``demo``: a synthetic clip and a fake model through the whole package. No network.
- ``run``: one real clip through the pipeline: result file, prediction file and video.
- ``doctor``: checks of the machine, and of the model API (gate G0).
- ``data``: download and extract the dataset, and write the split list.
- ``oracle``: label-only checks of the scoring chain (gate G1).
- ``probe``: measure one model step on labelled frames.
- ``tool-job``: write the frames and the settings that an external tool must run on.
- ``tool-score``: measure the answers of an external tool on the frames of a probe.
- ``clip-job``: write every frame of one clip and the settings that an external tool must run on.
- ``video-frames``: write every frame of a video file as an image below the data folder, for ``clip-job``.
- ``overlay``: draw the pitch lines of mappings on their frames, for the check by eye of gate G3.
- ``eval``: run the scorer on prediction files.
- ``sync``: check text result files and print the git commands that send them back.

Exit codes: 0 means done, 1 means that a check or a gate failed, 2 means that the command
could not run (wrong input, a dirty work tree, a budget stop, a failed model call, or any
other error). So the exit code 1 never comes from an error. An error ends with a line
``error: ...`` on the standard error stream. An error that can be a fault of the code also
shows its traceback before that line. A wrong settings file, roster file, label file or
experiment file is wrong input: the line names the file and the fault, and no traceback is
printed. The same holds for a split list (a fault of one of its groups names the group, not
the file), for a run file of an external tool, and for a clip folder whose frames do not fit
its label file or whose labels have no scored frame (``probe``, ``tool-job``, ``tool-score``).

The standard output is written line by line, also into a pipe or a file. So the cost
estimate and the budget line of a paid command can be read before its first model call,
and the lines of both streams keep their order when they go to one place.

Rules that the commands share:

- ``run``, ``probe``, ``tool-score`` and the model part of ``doctor`` refuse to start when the
  git work tree of the code is not clean, unless ``--allow-dirty`` is given. Their output files
  hold the commit of the code.
- A command that can pay for model calls prints its cost estimate before the first call and
  the measured cost after. Every paid answer goes into the cost ledger of the cache folder
  (``ledger.jsonl``, then ``ledger-0002.jsonl`` and so on), also an answer that was billed and
  then rejected. A budget guard with the limits of ``configs/gates.yaml`` stops a run that
  costs too much.
- A stop signal (Ctrl-C, a plain kill, a closed terminal) ends a command with the exit code 2.
  A model call that is in flight finishes first, and its answer is stored and recorded. No new
  call is sent after the signal: a request that waits for a new attempt gives up.
- The estimate of a run counts every planned call, as with an empty cache. So the limit of a
  run holds for all answers that the run uses: the new answers that it pays for, and the
  stored answers of earlier runs with the cost that they had. A run that is started again
  after a budget stop therefore pays for no further answer until the estimate is corrected.
- A run stops above 1.3 times its estimate (``budget.run_abort_factor``). The contract checks
  of ``doctor`` are the one exception: they stop above the spend limit of gate G0
  (``g0_contract.spend_usd_max``), and they do not start when their estimate is above it.
  Their answers are not stored. So a doctor that is started again pays for every answer
  again, up to that limit at each start.
- The default cache folder is ``results/cache`` in the work tree of the results branch. This
  work tree lies next to the repository, in ``../pitchvlm-results``. It must be there before
  a command uses the default cache folder: no command makes it. Use one cache folder for
  all paid runs: the ledger in it is the memory of the budget guard.
- The stored answers of a clip belong to its pixels. A stored answer that was recorded with
  other image bytes than the clip shows now stops ``run`` and ``probe`` before any model call,
  unless ``--allow-other-images`` is given.
- ``sync`` checks the result files with the guard and with the secret scan of the git hooks,
  so that a commit of the files is not refused later.
- No output file holds an absolute path, so the text results can go to the results branch.
- A command that writes images (``video-frames``, ``overlay``) refuses a folder inside a work
  tree of the repository: images stay below the data folder (rule 2 of ``CLAUDE.md``).

The package must run from its source checkout: the gate file, the guard script and the git
state of the code are found from the place of this file. An install in editable mode does
that. So does ``PYTHONPATH=src`` in the top folder of the repository, without any install.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import glob
import hashlib
import io
import json
import logging
import math
import os
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import stat
import sys
import time
import tempfile
import threading
import traceback
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypeVar, get_args

import numpy as np
import pydantic
import yaml
from pydantic import Field, model_validator

from pitchvlm import __version__, doctor, probes, radar, stopsignals
from pitchvlm.config import (
    DEFAULT_MODEL,
    EngineSettings,
    PipelineConfig,
    StepSettings,
    load_config,
    model_label,
    model_ref,
)
from pitchvlm.datasets import (
    LABEL_FILE_NAME,
    SPLITS,
    TEST_FINAL_SIZE,
    download_split,
    extract_clips,
    list_clips,
    load_manifest,
    make_split_manifest,
    save_manifest,
)
from pitchvlm.evalx.diagnostics import track_diagnostics
from pitchvlm.evalx.metrics import identity_scores, position_errors_m, summarize
from pitchvlm.evalx.oracle import (
    TeamSideFacts,
    keep_frames,
    label_homography_residuals_m,
    resample_states,
    shift_states,
    team_side_by_rule,
    team_side_facts,
)
from pitchvlm.evalx.scorer import COMBINED, run_trackeval
from pitchvlm.export_gsr import check_predictions, image_ids_without_items, write_predictions
from pitchvlm.fallbacks.answers import calibration_answers, detector_answers
from pitchvlm.fallbacks.fake import fake_calibration_answers, fake_detector_answers
from pitchvlm.fallbacks.pnlcalib import homography_from_payload
from pitchvlm.fallbacks.rfdetr import people_from_detections
from pitchvlm.fallbacks.runs import (
    CLIP_JOB_FORMAT,
    DETECTOR_FAMILIES,
    JOB_FORMAT,
    ClipJob,
    ClipRun,
    JobFrame,
    PnlcalibSettings,
    RunFrame,
    ToolJob,
    ToolRun,
    ToolSettings,
    clip_run_part_path,
    is_detector,
    load_clip_run,
    load_tool_run,
)
from pitchvlm.frames import (
    FRAME_FOLDER,
    FrameSource,
    ImageFolderSource,
    VideoFileSource,
    keyframe_indices,
    read_video_record,
    write_video_frames,
)
from pitchvlm.geometry import matrix_from_record, paint_alignment
from pitchvlm.labels import GsrLabels, gsr_frame_states, load_gsr_labels
from pitchvlm.livenumbers import DEFAULT_MAX_REQUESTS, LiveNumbers, NumberSettings, apply_numbers, wire
from pitchvlm.liveout import LIVE_FORMAT, LIVE_FORMAT_NUMBERS, LiveOut, LiveOutError, open_live_out
from pitchvlm.motion import MotionField
from pitchvlm.overlays import MappedFrame, caption, choose_frames, image_name, list_document
from pitchvlm.pipeline import ToolAnswers, keyframe_frames, run_clip
from pitchvlm.probes import LabelledClip, write_metrics
from pitchvlm.render import draw_mapping_check, render_clip, write_png
from pitchvlm.roster import Roster, load_roster
from pitchvlm.schemas import ClipInfo, ClipResult, FrameState, HomographyRecord, PaintCheck, Record
from pitchvlm.synth import SyntheticScene, SyntheticSource
from pitchvlm.feed import DEFAULT_QUEUE_LINES, DEFAULT_WAIT_S, FeedError, fd_lines, merge_feeds
from pitchvlm.stream import (
    KIT_WARMUP_FRAMES,
    STREAM_FORMAT,
    DetectorStream,
    FrameFeeder,
    OverloadPolicy,
    QueuedLines,
    StreamInput,
    StreamSettings,
    frames_text,
    check_stream_engines,
    clip_run_lines,
    default_history_frames,
    read_stream_lines,
    stand_in_inputs,
)
from pitchvlm.timing import LatencyLog, StageClock, percentiles_ms
from pitchvlm.tracking import SWAP_MIN_VOTES
from pitchvlm.vlm.cache import CachedBackend, CassetteStore
from pitchvlm.vlm.client import (
    FLEX_TIMEOUT_S,
    OPENROUTER_KEY_VARIABLE,
    STANDARD_TIMEOUT_S,
    GeminiBackend,
    LazySdkClient,
    OpenRouterBackend,
    ProviderBackend,
    UrllibTransport,
    installed_sdk_version,
)
from pitchvlm.vlm.coords import COORD_CONVENTIONS, GEMINI_CONVENTION
from pitchvlm.vlm.cost import (
    DEFAULT_TOKEN_ESTIMATES,
    BudgetExceeded,
    BudgetGuard,
    Ledger,
    billed_usd,
    estimate_usd,
    Price,
    VideoRate,
    has_flex_tier,
    ledger_file_number,
    load_price_table,
    load_video_rates,
)
from pitchvlm.vlm.fake import FakeBackend
from pitchvlm.vlm.prompts import PROMPT_NAMES
from pitchvlm.vlm.steps import ParseError
from pitchvlm.vlm.types import (
    Backend,
    BackendError,
    BilledAnswerError,
    Provider,
    RequestMeta,
    ThinkingLevel,
    VLMRequest,
    VLMResponse,
)

EXIT_OK = 0
#: A check or a gate failed. The output files are written.
EXIT_FAILED = 1
#: The command could not run, or an error stopped it.
EXIT_ERROR = 2

#: Environment variable that holds the API key on the run machine.
KEY_VARIABLE = "GEMINI_API_KEY"
#: Environment variable of the key of each provider. A program of another project (the scorer)
#: gets none of them.
PROVIDER_KEY_VARIABLES: dict[Provider, str] = {"gemini": KEY_VARIABLE, "openrouter": OPENROUTER_KEY_VARIABLE}
KEY_VARIABLES = tuple(PROVIDER_KEY_VARIABLES.values())
#: Price table of the models of OpenRouter, relative to the top folder of the repository.
PRICE_TABLE_FILE = Path("configs") / "openrouter_prices.yaml"
#: Data folder of the run machine. It lies outside the repository.
DATA_DIR = Path("~/pitchvlm-data")
#: Default folders below the data folder: the frame images of videos, and the check images of pitch mappings.
VIDEO_FRAMES_DIR = DATA_DIR / "videos"
#: Default folder of the videos that a model step with video input sends (``vlm.video``).
VIDEO_INPUT_DIR = DATA_DIR / "video-input"
OVERLAYS_DIR = DATA_DIR / "overlays"
#: Read size of a check sum of a file, bytes.
HASH_CHUNK_BYTES = 1 << 20
#: Files of the repository, relative to its folder.
GATES_FILE = Path("configs") / "gates.yaml"
SPLIT_LIST_FILE = Path("configs") / "splits.json"
GUARD_SCRIPT = Path("tools") / "guard.py"
#: Branch that carries the text results, and the folder of its work tree next to the repository.
RESULTS_BRANCH = "results"
RESULTS_WORKTREE = "pitchvlm-results"
#: Folder of the result files inside that work tree, and the cache folder below it.
RESULTS_FOLDER = "results"
CACHE_FOLDER = "cache"
#: Folder of the run files of external tools below the results folder (docs/RUNBOOK.md, section 3).
TOOL_RUNS_FOLDER = "tools"
#: The script that runs an external tool, below the top folder of the repository.
TOOL_SCRIPT = Path("tools") / "run_tool.py"
#: Name of the first file of the cost ledger inside the cache folder. The later files are
#: ledger-0002.jsonl and so on (``vlm.cost.ledger_file_number``).
LEDGER_NAME = "ledger.jsonl"
#: Commit text of a repository without any commit.
UNCOMMITTED = "uncommitted"
DEFAULT_SPLIT = "valid"
DEFAULT_TRACKER = "pitchvlm"
#: Output names.
PREDICTIONS_FOLDER = "predictions"
SCORER_FOLDER = "scorer"
RESULT_NAME = "result.json"
SUMMARY_NAME = "summary.json"
ORACLE_NAME = "oracle.json"
SCORES_NAME = "scores.json"
DEMO_VIDEO_NAME = "demo.mp4"
#: Split name in the prediction path of the demo clip.
DEMO_SPLIT = "demo"
DEMO_FRAMES = 100
#: Engines of the demo: the model steps through the fake model, or the detector and the
#: calibration through their stand-ins (no model step, so no number).
DEMO_ENGINES = {
    "model": EngineSettings(),
    "detector": EngineSettings(people="detector", pitch_mapping="calibration", number="none"),
}
#: Number requests that the cost estimate of one clip assumes. The real number is known only
#: after the tracking step: one request per track of a player or a goalkeeper, before the
#: tracks of one person are linked by their shirt number. Measured on the synthetic clip of
#: 30 seconds with exact answers: 48 to 50 requests (3 seeds). A real clip is not measured yet.
EXPECTED_NUMBER_REQUESTS = 50
#: Sections of the gate file.
G0_SECTION = "g0_contract"
G1_SECTION = "g1_harness"
G2_SECTION = "g2_people"
BUDGET_SECTION = "budget"
#: Tracker name of each prediction variant of the label-only stage (gate G1).
ORACLE_TRACKERS = {
    "ground_truth": "oracle_ground_truth",
    "shifted_1m": "oracle_shifted_1m",
    "keyframes_only": "oracle_keyframes_only",
    "resampled": "oracle_resampled",
}
#: Shift of the second variant, metres towards the right goal.
ORACLE_SHIFT_M = 1.0
#: File types of text result files: metrics and cassettes (JSON) and the cost ledger (JSON lines).
RESULT_SUFFIXES = (".json", ".jsonl")
#: File name of a cassette: the cache key of its request, 32 hexadecimal digits.
CASSETTE_NAME = re.compile(r"[0-9a-f]{32}\.json")
#: Number of files that one call of the guard script checks.
GUARD_BATCH = 200
#: The secret scan of the git hooks, and the file of its rules in the top folder of a work tree.
SECRET_SCAN_TOOL = "gitleaks"
SECRET_SCAN_RULES = Path(".gitleaks.toml")
#: Exit code that the secret scan is asked to give when it finds something. Its own exit code 1
#: also means that it could not run, for example because it cannot read its rules.
SECRET_SCAN_FOUND = 3
#: Signals that stop a command: Ctrl-C, a plain kill and a closed terminal.
STOP_SIGNALS = tuple(getattr(signal, name) for name in ("SIGINT", "SIGTERM", "SIGHUP") if hasattr(signal, name))
#: Loggers of the HTTP libraries below the model SDK. Each one writes a line for every request.
HTTP_LOGGERS = ("httpx2", "httpcore2", "httpx", "httpcore")
#: Set by a stop signal, and cleared when a command starts. The model backend of a command gets this
#: event: while it is set, the backend sends no request and no new attempt. Its lock can be taken
#: again by one thread, so the handler can set it at any point of the main thread (``StopEvent``).
STOP_REQUESTED: threading.Event = stopsignals.StopEvent()

ProbeKind = Literal["people", "pitch", "number"]
PROBE_KINDS: tuple[ProbeKind, ...] = ("people", "pitch", "number")
#: Model calls of one kind in a plan: (step, settings, number of requests).
PlannedCalls = tuple[str, StepSettings, int]
#: The probe whose frames and values measure each external tool: every detector family, and the calibration.
TOOL_PROBES: dict[str, ProbeKind] = {**dict.fromkeys(DETECTOR_FAMILIES, "people"), "pnlcalib": "pitch"}
#: The record that the adapter of an external tool makes from the payload of one frame.
_Adapted = TypeVar("_Adapted")


class CliError(Exception):
    """A command cannot run. The message says why and what to do."""


#: Errors whose class says that the cause lies outside the code: a refusal of the command line, a
#: failed model call, a model answer that cannot be used, a budget stop, and an error of the
#: operating system (a file, a folder or a connection that cannot be used). Their message says
#: enough, so ``main`` prints it as one line. Any other error can be a fault of the code.
KNOWN_ERRORS: tuple[type[Exception], ...] = (
    CliError,
    BackendError,
    BudgetExceeded,
    ParseError,
    OSError,
    LiveOutError,
)


# --- Repository and git --------------------------------------------------------------


def repository_root() -> Path:
    """Return the folder of the source checkout that holds this package.

    This file is ``<root>/src/pitchvlm/cli.py``. The gate file, the guard script and the
    git state of the code are found from this folder.
    """
    return Path(__file__).resolve().parents[2]


def default_cache_dir() -> Path:
    """Return the default cache folder: ``results/cache`` in the work tree of the results branch.

    The work tree lies next to the repository, in ``../pitchvlm-results``.
    """
    return repository_root().parent / RESULTS_WORKTREE / RESULTS_FOLDER / CACHE_FOLDER


def git_output(folder: Path, *arguments: str) -> str | None:
    """Run a git command that only reads, in a folder, and return what it printed.

    The result is None when the command failed, for example because the folder is not
    in a git work tree, or because git is not installed. The option
    ``--no-optional-locks`` keeps git from writing its index.
    """
    try:
        completed = subprocess.run(
            ["git", "--no-optional-locks", *arguments],
            cwd=folder,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
    except OSError:
        return None
    return completed.stdout if completed.returncode == 0 else None


@dataclass(frozen=True)
class CodeState:
    """The git state of the code that runs."""

    #: Hash of the current commit, or ``uncommitted`` when there is no commit.
    commit: str
    #: True when the work tree has changed or untracked files, or when it is not a git work tree.
    dirty: bool
    #: The lines of ``git status --porcelain``, or one line that says why the state is not known.
    changes: tuple[str, ...]


def code_state(repo: Path) -> CodeState:
    """Read the commit of a repository and whether its work tree is clean."""
    head = git_output(repo, "rev-parse", "--verify", "--quiet", "HEAD")
    commit = head.strip() if head and head.strip() else UNCOMMITTED
    status = git_output(repo, "status", "--porcelain")
    if status is None:
        reason = "the folder is not a git work tree, or git is not installed"
        return CodeState(commit=commit, dirty=True, changes=(reason,))
    changes = tuple(line for line in status.splitlines() if line.strip())
    return CodeState(commit=commit, dirty=bool(changes), changes=changes)


def require_clean_tree(allow_dirty: bool) -> CodeState:
    """Return the git state of the code. Raise ``CliError`` for a dirty work tree unless ``allow_dirty`` is true."""
    repo = repository_root()
    state = code_state(repo)
    if state.dirty and not allow_dirty:
        examples = "; ".join(state.changes[:5])
        raise CliError(
            f"the git work tree of {repo} is not clean ({len(state.changes)} entries, for example: {examples}). "
            "Commit the changes first. Or pass --allow-dirty: the output file then says that the tree was dirty."
        )
    return state


def scorer_commit(script: Path) -> str | None:
    """Return the commit of the clone of the scorer repository that holds the scorer script.

    The script is ``<clone>/scripts/run_soccernet_gs.py``, and it runs the code of its
    clone. So the commit of the clone names the code that scored. The result is None
    when the folder two levels above the script is not the top folder of a git work
    tree, or when that work tree has no commit. The two folders are compared as places
    on the disk, not as text: git spells the top folder as the disk stores it, and on a
    disk that does not tell upper and lower case apart the path can be typed in another
    letter case.
    """
    clone = script.resolve().parent.parent
    top = git_output(clone, "rev-parse", "--show-toplevel")
    if top is None or not top.strip() or not os.path.samefile(top.strip(), clone):
        return None
    head = git_output(clone, "rev-parse", "--verify", "--quiet", "HEAD")
    return head.strip() if head and head.strip() else None


def utc_now_text() -> str:
    """Return the current UTC time as ISO 8601 text, to the second."""
    return datetime.now(UTC).isoformat(timespec="seconds")


# --- Gate file -----------------------------------------------------------------------


def load_gates() -> dict[str, Any]:
    """Read ``configs/gates.yaml`` of the repository."""
    path = repository_root() / GATES_FILE
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise CliError(
            f"the gate file {path} cannot be read ({error}); install the package in editable mode"
        ) from error
    if not isinstance(data, dict):
        raise CliError(f"{path}: the top level of the gate file must be a mapping")
    return data


def _gate_value(gates: Mapping[str, Any], section: str, key: str) -> Any:
    """Return one value of the gate file, or None when the section or the key is missing."""
    values = gates.get(section)
    return values.get(key) if isinstance(values, dict) else None


def _is_number(value: object) -> bool:
    """Tell whether a value of the gate file is a number. A truth value is not."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def gate_number(gates: Mapping[str, Any], section: str, key: str) -> float:
    """Return one threshold of the gate file as a float. A missing or wrong value raises ``CliError``."""
    value = _gate_value(gates, section, key)
    if not _is_number(value):
        raise CliError(f"{GATES_FILE.as_posix()}: {section}.{key} must be a number, found {value!r}")
    return float(value)


def gate_range(gates: Mapping[str, Any], section: str, key: str) -> tuple[float, float]:
    """Return one range of the gate file as (low, high). A missing or wrong value raises ``CliError``."""
    value = _gate_value(gates, section, key)
    if not isinstance(value, list) or len(value) != 2 or not all(_is_number(end) for end in value):
        raise CliError(f"{GATES_FILE.as_posix()}: {section}.{key} must be a list of two numbers, found {value!r}")
    return float(value[0]), float(value[1])


@dataclass(frozen=True)
class GateCheck:
    """One measured value against its gate."""

    name: str
    value: float
    #: The gate in words, for example "at least 99.5".
    rule: str
    passed: bool


def at_least(name: str, value: float, minimum: float) -> GateCheck:
    """Compare a value with a lower limit. A value that is not a number does not pass."""
    return GateCheck(name=name, value=value, rule=f"at least {minimum:g}", passed=value >= minimum)


def at_most(name: str, value: float, maximum: float) -> GateCheck:
    """Compare a value with an upper limit. A value that is not a number does not pass."""
    return GateCheck(name=name, value=value, rule=f"at most {maximum:g}", passed=value <= maximum)


def within(name: str, value: float, low: float, high: float) -> GateCheck:
    """Compare a value with a range that includes its ends. A value that is not a number does not pass."""
    return GateCheck(name=name, value=value, rule=f"from {low:g} to {high:g}", passed=low <= value <= high)


# --- Cost: estimate, ledger and budget guard -----------------------------------------


@dataclass(frozen=True)
class Budget:
    """The money limits of the gate file."""

    #: The planned cost of the whole project, USD. A printed line warns above it.
    planned_usd: float
    #: All runs in the ledger together must stay under this, USD.
    hard_stop_usd: float
    #: A run stops when it costs more than this times its estimate.
    run_factor: float
    #: Largest spend of the model backend in any ten minutes, USD.
    max_usd_per_10_min: float


def budget_limits(gates: Mapping[str, Any]) -> Budget:
    """Read the money limits from the gate file."""
    return Budget(
        planned_usd=gate_number(gates, BUDGET_SECTION, "planned_usd"),
        hard_stop_usd=gate_number(gates, BUDGET_SECTION, "hard_stop_usd"),
        run_factor=gate_number(gates, BUDGET_SECTION, "run_abort_factor"),
        max_usd_per_10_min=gate_number(gates, BUDGET_SECTION, "max_usd_per_10_min"),
    )


def usd(value: float) -> str:
    """Write an amount of money for a printed line."""
    return f"{value:.3f} USD"


def openrouter_prices() -> tuple[Price, ...]:
    """Read the price table of the OpenRouter models (``PRICE_TABLE_FILE``). Wrong content raises ``CliError``."""
    path = repository_root() / PRICE_TABLE_FILE
    try:
        return load_price_table(path)
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise CliError(f"the price table {PRICE_TABLE_FILE.as_posix()} cannot be read: {error}") from error


def openrouter_video_rates() -> tuple[VideoRate, ...]:
    """Read the video rates of the OpenRouter models (``PRICE_TABLE_FILE``). Wrong content raises ``CliError``."""
    path = repository_root() / PRICE_TABLE_FILE
    try:
        return load_video_rates(path)
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise CliError(f"the price table {PRICE_TABLE_FILE.as_posix()} cannot be read: {error}") from error


def estimated_requests_usd(requests: Sequence[VLMRequest]) -> float:
    """Return the expected cost of built requests in USD, with the tokens of their video parts (``vlm.cost``).

    ``CliError`` is raised for a model without a price, and for a request with a video
    part for a model without a video rate: such a model takes no video.
    """
    if not requests:
        return 0.0
    openrouter = any(request.provider != "gemini" for request in requests)
    try:
        return estimate_usd(
            requests,
            DEFAULT_TOKEN_ESTIMATES,
            datetime.now(UTC).date(),
            openrouter_prices=openrouter_prices() if openrouter else (),
            openrouter_video_rates=openrouter_video_rates() if openrouter else (),
        )
    except (KeyError, ValueError) as error:
        raise CliError(f"the cost cannot be estimated: {error.args[0]}") from error


def estimated_usd(clip_id: str, plan: Sequence[PlannedCalls]) -> float:
    """Return the expected cost of planned model calls in USD, at the prices of today (UTC).

    The estimate uses the default token counts per step of ``vlm.cost``. Each planned
    call becomes a request without text and images, because the estimate reads only the
    step, the provider, the model and the service tier. These requests are never sent.
    A Gemini model takes the price table of ``vlm.cost``, an OpenRouter model the price
    table ``PRICE_TABLE_FILE``.

    ``CliError`` is raised for a model without a price, and for a step that asks for the
    flex tier of a model that has none: such a call would fail, and its half price
    would make the estimate too low.
    """
    for step, settings, _ in plan:
        if settings.service_tier == "flex" and not has_flex_tier(settings.model):
            raise CliError(
                f"the {step} step asks for the flex tier of the model {settings.model}, which has no flex tier. "
                "Use the standard tier for this model."
            )
    requests = [
        VLMRequest(
            provider=settings.provider,
            model=settings.model,
            thinking_level=settings.thinking_level,
            parts=(),
            response_schema={},
            prompt_name=PROMPT_NAMES[step],
            meta=RequestMeta(clip_id=clip_id, step=step),
            service_tier=settings.service_tier,
        )
        for step, settings, count in plan
        for _ in range(count)
    ]
    try:
        return estimate_usd(
            requests,
            DEFAULT_TOKEN_ESTIMATES,
            datetime.now(UTC).date(),
            openrouter_prices=openrouter_prices() if any(r.provider != "gemini" for r in requests) else (),
        )
    except KeyError as error:
        raise CliError(f"the cost cannot be estimated: {error.args[0]}") from error


def print_estimate(estimate: float, plan: Sequence[PlannedCalls], requests: Sequence[VLMRequest] = ()) -> None:
    """Print the cost estimate of planned model calls and of built requests (``estimated_requests_usd``)."""
    counts: Counter[str] = Counter()
    for step, _, count in plan:
        counts[step] += count
    for request in requests:
        counts[request.meta.step] += 1
    listed = ", ".join(f"{step} {count}" for step, count in counts.items())
    print(
        f"cost estimate: {usd(estimate)} for {sum(counts.values())} planned model calls ({listed}), "
        "at the prices of today and with an empty cache"
    )


@dataclass
class StepSpend:
    """Tokens and cost of the new answers of one step."""

    answers: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    usd: float = 0.0


class CostMeter:
    """The cost side of one paid run.

    Every new answer goes into the counts of this meter and into the cost ledger. Then
    the budget guard is asked: it raises ``BudgetExceeded`` when the run or the whole
    ledger is over its limit. An answer that was billed and then rejected
    (``BilledAnswerError``) counts like a new answer: the wrapper of the model backend
    reports its cost record here. An answer is counted before its ledger line is
    written, so the measured cost of the run also holds an answer whose line could not
    be written.

    The meter also adds up the stored answers that the run uses, with the cost that
    they had when they were paid (``on_stored_answer``). The estimate of a run counts
    every planned call, so the limit of the run, ``run_limit_usd``, holds for the new
    and the stored answers together (``check_run_limit``). Without this rule, a run that
    the budget guard stopped would get the full limit again at every new start.

    The budget guard and this meter check the same limit, and with several requests in
    flight either one can stop the run first. So both stop messages name the limit with
    the same words (``limit_named``, for example "1.3 times its estimate").

    One meter can be used from several threads.
    """

    def __init__(
        self, ledger: Ledger, guard: BudgetGuard, run_id: str, run_limit_usd: float, limit_named: str
    ) -> None:
        self.run_id = run_id
        self._ledger = ledger
        self._guard = guard
        self._run_limit_usd = run_limit_usd
        #: The words that name the limit of the run, for example "1.3 times its estimate".
        self._limit_named = limit_named
        self._lock = threading.Lock()
        self._steps: dict[str, StepSpend] = {}
        self._stored_answers = 0
        self._stored_usd = 0.0

    def on_new_answer(self, request: VLMRequest, response: VLMResponse) -> None:
        """Record one paid answer, or the cost record of a rejected one. Then stop the run when it costs too much.

        The answer is counted first. An error of the ledger write (a full disk) is raised
        after that, so the printed measured cost still holds the answer.
        """
        cost = billed_usd(response)
        with self._lock:
            spend = self._steps.setdefault(request.meta.step, StepSpend())
            spend.answers += 1
            spend.input_tokens += response.usage.input_tokens
            spend.output_tokens += response.usage.output_tokens
            spend.thought_tokens += response.usage.thought_tokens
            spend.usd += cost
        self._ledger.record(self.run_id, request, response)
        self._guard.check(self.run_id)
        self.check_run_limit()

    def on_stored_answer(self, response: VLMResponse) -> None:
        """Count one stored answer that the run uses, with the cost that it had when it was paid."""
        cost = billed_usd(response)
        with self._lock:
            self._stored_answers += 1
            self._stored_usd += cost

    def check_run_limit(self) -> None:
        """Raise ``BudgetExceeded`` when the answers that the run used cost more than the limit of the run.

        The answers are the new answers that the run paid for and the stored answers
        that it used. The backend of a paid session calls this before every model call.
        """
        with self._lock:
            paid = math.fsum(spend.usd for spend in self._steps.values())
            stored = self._stored_usd
        if paid + stored > self._run_limit_usd:
            raise BudgetExceeded(
                f"the run {self.run_id!r} used stored answers that cost {usd(stored)} and paid {usd(paid)} for new "
                f"answers: together more than its limit of {usd(self._run_limit_usd)} ({self._limit_named}). "
                "No further answer is paid. The estimate of this work is too low: correct the token estimates "
                "with the measured mean tokens before the next start."
            )

    def report(self) -> dict[str, Any]:
        """Return the measured cost of the run as plain JSON data.

        It holds the totals of the new answers, their tokens and cost per step, and the
        number and the earlier cost of the stored answers that the run used.
        """
        with self._lock:
            steps = {step: dataclasses.asdict(spend) for step, spend in sorted(self._steps.items())}
            stored_answers, stored_usd = self._stored_answers, self._stored_usd
        return {
            "run_id": self.run_id,
            "new_answers": sum(spend["answers"] for spend in steps.values()),
            "measured_usd": math.fsum(spend["usd"] for spend in steps.values()),
            "stored_answers": stored_answers,
            "stored_usd": stored_usd,
            "steps": steps,
        }


def finish_in_flight(call: Callable[[], VLMResponse]) -> VLMResponse:
    """Run one model call so that a stop signal cannot cut it off, and return its answer.

    A stop signal raises ``KeyboardInterrupt`` in the main thread. Raised inside a model
    call, it would throw the call away: the provider can still finish and bill the
    request, but no answer is stored and no ledger line is written. So the signal
    handler (``_interrupt``) holds a stop signal back while a call runs here in the main
    thread. The call finishes, with its answer stored and recorded, and then this
    function raises the interruption. The real backend sends nothing more in the
    meantime (``STOP_REQUESTED``), so a call that waits for a new attempt ends at once,
    and a call in flight ends with its client timeout at the latest. A further stop
    signal does not end the call either, as in the thread pool of the pipeline: its
    answer is paid.

    In any other thread the call runs as it is: no signal is raised there, and the
    thread pool of the pipeline waits for its own requests. The hold is
    ``stopsignals.held_stops``, as in the wait of the thread pool.
    """
    with stopsignals.held_stops():
        return call()


class MeteredBackend:
    """A backend without a cache that reports every answer of another backend to a cost meter.

    It does for calls without a cache what ``CachedBackend`` does for the other calls: it
    is the only place that reports their cost. So a model backend gets one of the two
    wrappers, never both.
    """

    def __init__(self, inner: Backend, meter: CostMeter) -> None:
        self._inner = inner
        self._meter = meter

    def generate(self, request: VLMRequest) -> VLMResponse:
        """Send one request. Its answer is new and paid, so it goes to the cost meter.

        A call that was billed although its answer cannot be used (``BilledAnswerError``)
        goes to the cost meter with its cost record. Then the error is raised again. A
        budget stop of the cost meter replaces it. A stop signal does not cut the call
        off: the call and its report to the cost meter finish first (``finish_in_flight``).
        """
        return finish_in_flight(lambda: self._ask_and_report(request))

    def _ask_and_report(self, request: VLMRequest) -> VLMResponse:
        """Ask the wrapped backend, and report the answer or the cost record of a rejected answer to the cost meter."""
        try:
            response = self._inner.generate(request)
        except BilledAnswerError as error:
            self._meter.on_new_answer(request, error.billed)
            raise
        self._meter.on_new_answer(request, response)
        return response


class SessionBackend:
    """The backend of a command with stored answers: it counts the requests, and it guards the money of a paid run.

    A request with a stored answer gets that answer. In a paid session the cost that
    the answer had goes to the cost meter. A request without a stored answer goes to
    the cached backend, which asks the model in a paid session and raises ``CacheMiss``
    in replay mode. Before that call the cost meter checks the limit of the run, so a
    run that is over its limit stops before it pays for one more answer. A stop signal
    does not cut a model call off: the call finishes, and its answer is stored and
    recorded (``finish_in_flight``).

    A replay has no cost meter. So the backend itself adds up the stored answers that a
    replay uses, with the cost that they had when they were paid (``replayed``). One
    instance can be used from several threads.
    """

    def __init__(self, store: CassetteStore, cached: CachedBackend, meter: CostMeter | None) -> None:
        self._store = store
        self._cached = cached
        self._meter = meter
        self._lock = threading.Lock()
        self._requests: Counter[str] = Counter()
        self._replayed_answers = 0
        self._replayed_usd = 0.0

    def generate(self, request: VLMRequest) -> VLMResponse:
        """Return the stored answer of a request, or a new answer through the cached backend."""
        with self._lock:
            self._requests[request.meta.step] += 1
        if self._meter is None:
            stored = self._cached.generate(request)
            cost = billed_usd(stored)
            with self._lock:
                self._replayed_answers += 1
                self._replayed_usd += cost
            return stored
        stored = self._store.get(request)
        if stored is not None:
            self._meter.on_stored_answer(stored)
            return stored
        self._meter.check_run_limit()
        return finish_in_flight(lambda: self._cached.generate(request))

    def requests_per_step(self) -> dict[str, int]:
        """Return the number of requests that the command made so far, by step."""
        with self._lock:
            return dict(sorted(self._requests.items()))

    def replayed(self) -> tuple[int, float]:
        """Return the number of stored answers that a replay used so far, and what they cost when they were paid."""
        with self._lock:
            return self._replayed_answers, self._replayed_usd

    def other_video_bytes(self) -> int:
        """Return the stored answers used so far that were recorded from other bytes of the same video frames."""
        return self._store.other_video_bytes


def new_run_id(command: str, label: str) -> str:
    """Make the id of one run for the cost ledger: the command, a label, the UTC time and a random part."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{command}-{label}-{stamp}-{secrets.token_hex(2)}"


def start_meter(
    cache_dir: Path,
    budget: Budget,
    run_id: str,
    estimate: float,
    *,
    spend_limit_usd: float | None = None,
    over_budget: bool = False,
) -> CostMeter:
    """Open the cost ledger of the cache folder and register a run with its estimate.

    A run stops above ``budget.run_factor`` times its estimate. ``spend_limit_usd``
    replaces that limit for a run that has its own spend limit in the gate file: the
    contract checks of gate G0. ``BudgetExceeded`` is raised when the estimate of such a
    run is above its spend limit, because the run could not finish.

    A printed line names what the ledger holds, the planned budget, the hard stop and the
    limit of the run, and a second line warns when the ledger is above the planned budget.
    ``BudgetExceeded`` is raised when the ledger has already reached the hard stop. It is
    also raised when the ledger and the estimate of the run together are above the hard
    stop: such a run pays until the hard stop and gives no result. ``over_budget`` allows
    that start, for example when most answers of the run are stored already. The hard
    stop still holds.
    """
    ledger = Ledger(cache_dir / LEDGER_NAME)
    guard = BudgetGuard(ledger, hard_stop_usd=budget.hard_stop_usd, run_factor=budget.run_factor)
    total = ledger.total_usd()
    if spend_limit_usd is None:
        run_limit, named = budget.run_factor * estimate, f"{budget.run_factor:g} times its estimate"
    else:
        run_limit, named = spend_limit_usd, "its spend limit in the gate file"
    print(
        f"budget: the ledger {ledger.path} holds {usd(total)}; the planned budget is {usd(budget.planned_usd)} and "
        f"the hard stop is {usd(budget.hard_stop_usd)}; this run stops above {usd(run_limit)} ({named})"
    )
    if total > budget.planned_usd:
        print(
            f"warning: the ledger is {usd(total - budget.planned_usd)} above the planned budget. "
            f"Paid runs end at the hard stop of {usd(budget.hard_stop_usd)}."
        )
    guard.start_run(run_id, estimate, limit_usd=spend_limit_usd)
    guard.check(run_id)
    if spend_limit_usd is not None and estimate > spend_limit_usd:
        raise BudgetExceeded(
            f"the estimate of this run is {usd(estimate)}, more than its spend limit of {usd(spend_limit_usd)}, "
            "so the run could not finish. Plan less work."
        )
    if total + estimate > budget.hard_stop_usd and not over_budget:
        raise BudgetExceeded(
            f"the ledger holds {usd(total)} and the estimate of this run is {usd(estimate)}: together more than the "
            f"hard stop of {usd(budget.hard_stop_usd)}, so the run could not finish. Plan less work. Or pass "
            "--start-over-budget when most answers of the run are stored already: the run then stops when the "
            "ledger reaches the hard stop."
        )
    return CostMeter(ledger, guard, run_id, run_limit_usd=run_limit, limit_named=named)


def print_measured(cost: Mapping[str, Any]) -> None:
    """Print the measured cost of a run, and the mean token counts of the new answers per step.

    A line names the stored answers that the run used, when there are any.
    """
    print(f"measured cost: {usd(cost['measured_usd'])} for {cost['new_answers']} new answers (run id {cost['run_id']})")
    if cost["stored_answers"]:
        print(
            f"  stored answers used: {cost['stored_answers']}; they cost {usd(cost['stored_usd'])} "
            "when they were paid, and they count against the limit of this run"
        )
    for step, spend in cost["steps"].items():
        answers = spend["answers"]
        print(
            f"  {step}: {answers} answers, mean tokens {spend['input_tokens'] / answers:.0f} in, "
            f"{spend['output_tokens'] / answers:.0f} out, {spend['thought_tokens'] / answers:.0f} thinking, "
            f"{usd(spend['usd'])}"
        )


def model_backend(max_usd_per_10_min: float) -> Backend:
    """Return the real model backend: each request goes to the backend of its provider.

    The key of a provider, and for Gemini the SDK, are needed only when a request of that
    provider is sent. Each provider has its own spend window of ``max_usd_per_10_min``.
    The backends send no request and no new attempt after a stop signal (``STOP_REQUESTED``).
    """
    gemini = GeminiBackend(api_key_env=KEY_VARIABLE, max_usd_per_10_min=max_usd_per_10_min, stop=STOP_REQUESTED)
    openrouter = OpenRouterBackend(
        prices=openrouter_prices(),
        video_rates=openrouter_video_rates(),
        api_key_env=OPENROUTER_KEY_VARIABLE,
        max_usd_per_10_min=max_usd_per_10_min,
        stop=STOP_REQUESTED,
    )
    return ProviderBackend({"gemini": gemini, "openrouter": openrouter})


def recording_backend(max_usd_per_10_min: float) -> tuple[Backend, doctor.CallRecorder]:
    """Return the real model backend on top of layers that record the raw facts of every call.

    Gemini: the SDK client is built on the first Gemini call, with the longest timeout of
    any call. The backend gives every call the timeout of its service tier: a flex request
    can wait in a queue of the provider, so it gets the longer one. OpenRouter: the HTTP
    layer records every answer with a success status. Both record into one recorder. The
    backends send no request and no new attempt after a stop signal (``STOP_REQUESTED``).
    """
    recorder = doctor.RecordingClient(LazySdkClient(KEY_VARIABLE, max(STANDARD_TIMEOUT_S, FLEX_TIMEOUT_S)))
    gemini = GeminiBackend(
        api_key_env=KEY_VARIABLE, max_usd_per_10_min=max_usd_per_10_min, stop=STOP_REQUESTED, client=recorder
    )
    openrouter = OpenRouterBackend(
        prices=openrouter_prices(),
        video_rates=openrouter_video_rates(),
        api_key_env=OPENROUTER_KEY_VARIABLE,
        max_usd_per_10_min=max_usd_per_10_min,
        stop=STOP_REQUESTED,
        transport=doctor.RecordingTransport(UrllibTransport(), recorder),
    )
    return ProviderBackend({"gemini": gemini, "openrouter": openrouter}), recorder


@dataclass(frozen=True)
class ModelSession:
    """The model access of one command: the backend, and the cost side when the run can pay."""

    backend: SessionBackend
    run_id: str
    estimate_usd: float
    #: None in replay mode, where no model call is made.
    meter: CostMeter | None


def open_session(
    command: str,
    label: str,
    plan: Sequence[PlannedCalls],
    *,
    cache_dir: Path,
    replay: bool,
    repeat: int,
    over_budget: bool = False,
    allow_other_images: bool = False,
    planned_requests: Sequence[VLMRequest] = (),
) -> ModelSession:
    """Build the cached backend of a command and print the cost estimate, before the first model call.

    ``label`` names the clip or the experiment. In replay mode the backend only reads
    stored answers. Otherwise a miss goes to the real model: the answer is stored, its
    cost goes into the ledger, and the budget guard may stop the run. ``over_budget``
    lets a paid run start although its estimate does not fit under the hard stop
    (``start_meter``).

    ``repeat`` is the repeat index of the requests of the command. A printed line says
    what an index above 0 means for the cost. ``allow_other_images`` lets the store give
    a stored answer that was recorded with other image bytes than the clip shows now
    (``CassetteStore``). Without it such an answer stops the command.

    ``planned_requests`` are built requests that the estimate counts next to ``plan``:
    the requests with video parts, whose tokens depend on the videos
    (``estimated_requests_usd``).
    """
    run_id = new_run_id(command, label)
    estimate = estimated_usd(label, plan) + estimated_requests_usd(planned_requests)
    store = CassetteStore(cache_dir, allow_other_images=allow_other_images)
    print(f"cache folder: {cache_dir}")
    print_estimate(estimate, plan, planned_requests)
    if repeat > 0:
        print(
            f"repeat index {repeat}: the stored answers of other repeat indexes are not used, "
            "so a paid run with a new repeat index pays for every call again"
        )
    if replay:
        print("replay mode: only stored answers are used, so this run costs nothing")
        backend = SessionBackend(store, CachedBackend(store), meter=None)
        return ModelSession(backend=backend, run_id=run_id, estimate_usd=estimate, meter=None)
    budget = budget_limits(load_gates())
    meter = start_meter(cache_dir, budget, run_id, estimate, over_budget=over_budget)
    cached = CachedBackend(store, model_backend(budget.max_usd_per_10_min), on_new_answer=meter.on_new_answer)
    return ModelSession(backend=SessionBackend(store, cached, meter), run_id=run_id, estimate_usd=estimate, meter=meter)


def close_session(session: ModelSession) -> dict[str, Any]:
    """Print the measured cost of a session and return its cost record for the output file.

    The record also holds the number of requests that the command made, by step. So
    the real number of number requests of a clip can be compared with the plan. It
    holds the number of stored answers that the command used and the cost that they had
    when they were paid, in a replay as in a paid run.
    """
    requests = session.backend.requests_per_step()
    listed = ", ".join(f"{step} {count}" for step, count in requests.items()) or "none"
    print(f"requests of this run, answered by the model or from the stored answers: {listed}")
    # A stored answer of a video that was written again with other bytes is used, and said here (``vlm.cache``).
    rewritten = session.backend.other_video_bytes()
    note = {"stored_answers_on_rewritten_videos": rewritten} if rewritten else {}
    if session.meter is None:
        stored_answers, stored_usd = session.backend.replayed()
        print("measured cost: 0.000 USD (replay mode: no model call was made)")
        if stored_answers:
            print(f"  stored answers used: {stored_answers}; they cost {usd(stored_usd)} when they were paid")
        print_rewritten_videos(rewritten)
        return {
            "replay": True,
            "estimate_usd": session.estimate_usd,
            "run_id": session.run_id,
            "requests": requests,
            "new_answers": 0,
            "measured_usd": 0.0,
            "stored_answers": stored_answers,
            "stored_usd": stored_usd,
            "steps": {},
            **note,
        }
    cost = {"replay": False, "estimate_usd": session.estimate_usd, "requests": requests, **session.meter.report(), **note}
    print_measured(cost)
    print_rewritten_videos(rewritten)
    return cost


def print_rewritten_videos(count: int) -> None:
    """Say how many stored answers were used for videos that were written again with other bytes, when any were."""
    if count:
        print(
            f"  stored answers used for a video written again with other bytes from the same frames: {count} "
            "(the model saw the bytes of the first file; docs/DECISIONS.md, entry of 2026-11-23 on the video key)"
        )


# --- Clips, labels and prediction files ----------------------------------------------


def split_dir(root: Path, split: str) -> Path:
    """Return the folder of one split below the data root: ``root/SoccerNetGS-<split>``."""
    return root / f"SoccerNetGS-{split}"


def clips_of_split(root: Path, split: str, wanted: Sequence[str] | None) -> list[str]:
    """Return the clips to use: the wanted clips, or all clips of the split folder when none is named."""
    folder = split_dir(root, split)
    if not folder.is_dir():
        raise CliError(f"{folder} does not exist: download the split with the data command")
    clips = list(wanted) if wanted else list_clips(folder)
    if not clips:
        raise CliError(f"{folder} holds no clip")
    return clips


def dataset_clip(root: Path, split: str, clip: str) -> tuple[ImageFolderSource, GsrLabels]:
    """Open the frames and the labels of one clip of the dataset: ``root/SoccerNetGS-<split>/<clip>``.

    The frame rate comes from the label file. An image file in the frame folder whose name
    is no number, and two image files with the same number, raise ``CliError`` with the clip
    folder.
    """
    folder = split_dir(root, split) / clip
    labels = read_labels(folder / LABEL_FILE_NAME)
    try:
        source = ImageFolderSource(folder / FRAME_FOLDER, clip, labels.fps)
    except ValueError as error:
        raise CliError(f"the clip folder {folder} cannot be used: {error}") from error
    return source, labels


def matched_clip(source: FrameSource, labels: GsrLabels, folder: Path) -> LabelledClip:
    """Join the frames and the labels of one clip of the dataset.

    Frames and labels that do not belong together (another clip id, another number of
    frames, another image size) raise ``CliError`` with the clip folder: the cause lies in
    the data folder, for example a clip that was not extracted to its end. So do labels
    without a scored frame: no probe can measure such a clip.
    """
    try:
        clip = LabelledClip(source=source, labels=labels)
    except ValueError as error:
        raise CliError(f"the clip folder {folder} cannot be used: {error}") from error
    if not any(frame.scored for frame in labels.frames):
        raise CliError(f"the clip folder {folder} cannot be used: the labels have no scored frame")
    return clip


def labelled_clip(root: Path, split: str, clip: str) -> LabelledClip:
    """Open one clip of the dataset with its labels (``dataset_clip``, ``matched_clip``)."""
    source, labels = dataset_clip(root, split, clip)
    return matched_clip(source, labels, split_dir(root, split) / clip)


def disk_spelling(path: Path) -> Path:
    """Return an absolute path of something that exists, with every part spelled as the disk stores it.

    On a disk that does not tell upper and lower case apart, a path opens its file in
    every spelling. The clip id is a part of every cache key, so it must not follow the
    spelling that was typed. A part that is not in the listing of its folder is replaced
    by the one entry of the listing that is the same directory entry (a link is compared
    as a link, not as the file that it points to). A part whose folder cannot be listed
    stays as it is.
    """
    spelled = Path(path.anchor)
    for part in path.parts[1:]:
        try:
            names = os.listdir(spelled)
        except OSError:
            names = [part]
        if part not in names:
            typed = os.lstat(spelled / part)
            same = [name for name in names if os.path.samestat(os.lstat(spelled / name), typed)]
            if len(same) == 1:
                part = same[0]
        spelled = spelled / part
    return spelled


def open_clip(path: Path, *, fps: float | None, labels: GsrLabels | None) -> FrameSource:
    """Open a clip: a folder of frame images or a video file.

    A folder may be the frame folder itself, or a clip folder of the dataset that holds
    the frame folder ``img1``. The clip id is the name of the clip folder. The folder is
    resolved first (made absolute, with links followed) and spelled as the disk stores
    it (``disk_spelling``). So a relative path such as ``.`` or ``img1`` gives the name
    of the real clip folder, and a name that is typed in another letter case gives the
    same clip id. A folder of images has no frame rate of its own: the rate comes from
    the labels, or from ``fps`` when there are no labels. A video file has its own frame
    rate, so ``fps`` must be None for it. Its clip id is the name of the file without
    the ending, as the disk spells it; a link keeps its own name.
    """
    if path.is_dir():
        folder = frame_folder(path)
        return ImageFolderSource(folder, clip_id_of_folder(folder), folder_frame_rate(fps, labels))
    if path.is_file():
        if fps is not None:
            raise CliError("--fps is for a folder of frame images; a video file has its own frame rate")
        return VideoFileSource(disk_spelling(path.absolute()))
    raise CliError(f"{path} is not a folder of frame images and not a video file")


def frame_folder(path: Path) -> Path:
    """Return the folder of the frame images of a clip folder: the folder, or its frame folder ``img1``.

    The folder is resolved and spelled as the disk stores it (``open_clip`` says why).
    """
    folder = path.resolve()
    if (folder / FRAME_FOLDER).is_dir():
        folder = folder / FRAME_FOLDER
    return disk_spelling(folder)


def clip_id_of_folder(folder: Path) -> str:
    """Return the clip id of a folder of frame images: the name of its clip folder.

    ``folder`` is a resolved path. Its clip folder is the folder above it when it is a
    frame folder ``img1``, else the folder itself. ``CliError`` is raised when the clip
    folder has no name, which is the case for the top folder of a disk.
    """
    clip_folder = folder.parent if folder.name == FRAME_FOLDER else folder
    if not clip_folder.name:
        raise CliError(
            f"{folder} gives no clip id: the clip id is the name of the clip folder, but {clip_folder} has no name. "
            "Put the frame images into a folder that has the name of the clip."
        )
    return clip_folder.name


def folder_frame_rate(fps: float | None, labels: GsrLabels | None) -> float:
    """Return the frame rate of a folder of frame images: the one of the labels, else the given one."""
    if labels is not None and fps is not None:
        raise CliError("give --fps or --labels, not both: the label file names the frame rate")
    if labels is not None:
        return labels.fps
    if fps is None:
        raise CliError("a folder of frame images has no frame rate: give --labels or --fps")
    return fps


def check_labels_match(source: FrameSource, labels: GsrLabels) -> None:
    """Raise ``ValueError`` when the frames of a clip and a label file do not belong together.

    They must have the same clip id, the same number of frames and the same image size.
    ``probes.LabelledClip`` holds this check.
    """
    LabelledClip(source=source, labels=labels)


def scored_image_ids(labels: GsrLabels) -> dict[int, str]:
    """Map the frame index of every scored frame to the image id of the label file."""
    return {frame.frame: frame.image_id for frame in labels.frames if frame.scored}


def write_clip_predictions(
    frames: Sequence[FrameState], labels: GsrLabels, root: Path, *, split: str, tracker: str, clip_id: str
) -> Path:
    """Write the prediction file of one clip for the scored frames of its labels, and return its path.

    The image ids and the type of the shirt numbers come from the labels.
    """
    return write_predictions(
        frames,
        scored_image_ids(labels),
        root,
        split=split,
        tracker=tracker,
        clip_id=clip_id,
        jersey_as_str=labels.jersey_is_str,
    )


def prediction_problems(path: Path, labels: GsrLabels) -> list[str]:
    """Read a written prediction file and return the problems that the format checker finds. Empty means valid."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = set(scored_image_ids(labels).values())
    return check_predictions(payload, expected_image_ids=expected, jersey_as_str=labels.jersey_is_str)


def frames_without_prediction(path: Path, labels: GsrLabels) -> tuple[int, int]:
    """Read a written prediction file and count the scored frames for which it holds no item.

    The result is ``(scored frames without an item, scored frames)``. A frame without an
    item is not a format problem: the scorer reads the file and counts every labelled
    person of that frame as missed.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = set(scored_image_ids(labels).values())
    return len(image_ids_without_items(payload, expected_image_ids=expected)), len(expected)


def empty_result_reason(frames: Sequence[FrameState], scored: tuple[int, int] | None) -> str | None:
    """Say in a few words why a clip gave no result at all, or return None when it gave one.

    ``scored`` is the pair of ``frames_without_prediction`` for a clip with labels: the
    scored frames without a prediction, and the scored frames. Such a clip gave no result
    when none of its scored frames has a prediction: the scorer then counts every labelled
    person as missed. A clip without labels, or without a scored frame, gave no result
    when no frame holds a person. The usual cause of both is a clip without any pitch
    mapping, because a person without a pitch position has no state.
    """
    if scored is not None and scored[1] > 0:
        return "no scored frame has a prediction" if scored[0] == scored[1] else None
    return None if any(state.players for state in frames) else "no frame of the result holds a person"


def print_empty_result(reason: str) -> None:
    """Print the line that says that a clip gave no result, with the usual cause."""
    print(
        f"failed: {reason}, so the clip gives no result. The usual cause is a clip without a pitch mapping: "
        "then no person gets a pitch position, and a warning of the pipeline in the log says so"
    )


def print_problems(problems: Sequence[str]) -> None:
    """Print the verdict of the format checker and its problems."""
    if not problems:
        print("format checker: the prediction file is valid")
        return
    print(f"format checker: {len(problems)} problems in the prediction file")
    for problem in problems:
        print(f"  {problem}")


def wrong_file(kind: str, path: Path, error: Exception) -> CliError:
    """Build the refusal of a file that the user gave: one line with the kind of file, its path and the fault.

    Wrong content of such a file is wrong input, not a fault of the code. So the command
    line shows this one line and no traceback. A validation error lists every wrong value
    with its place in the file. The text of any other error is put on one line, without
    the path at its start, which some readers put there.
    """
    if isinstance(error, pydantic.ValidationError):
        faults = [
            f"{'.'.join(str(part) for part in fault['loc']) or 'the content'}: {fault['msg']}"
            for fault in error.errors()
        ]
        fault_text = "; ".join(faults)
    else:
        fault_text = " ".join(str(error).removeprefix(f"{path}: ").split())
    return CliError(f"the {kind} {path} cannot be used: {fault_text}")


def read_settings(path: Path) -> PipelineConfig:
    """Read the settings file of a run. Wrong content raises ``CliError`` with the path."""
    try:
        return load_config(path)
    except (ValueError, yaml.YAMLError) as error:
        raise wrong_file("settings file", path, error) from error


def read_roster(path: Path) -> Roster:
    """Read a roster file. Wrong content raises ``CliError`` with the path."""
    try:
        return load_roster(path)
    except ValueError as error:
        raise wrong_file("roster file", path, error) from error


def read_labels(path: Path) -> GsrLabels:
    """Read a label file. Wrong content raises ``CliError`` with the path."""
    try:
        return load_gsr_labels(path)
    except ValueError as error:
        raise wrong_file("label file", path, error) from error


def scene_labels(scene: SyntheticScene) -> GsrLabels:
    """Return the labels of a synthetic scene, read through a label file like the labels of a real clip."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / scene.info.clip_id / LABEL_FILE_NAME
        path.parent.mkdir()
        path.write_text(json.dumps(scene.gsr_labels()), encoding="utf-8")
        return load_gsr_labels(path)


def write_json(path: Path, document: Mapping[str, Any]) -> None:
    """Write a document as compact JSON. Missing folders are made. A value that is not finite raises ``ValueError``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, allow_nan=False) + "\n", encoding="utf-8")


# --- demo ----------------------------------------------------------------------------


def paint_summary(checks: Sequence[PaintCheck]) -> dict[str, Any]:
    """Summarise the paint-alignment scores of the keyframe mappings of a clip: counts and score distribution.

    The mappings are the landmark fits of the pitch engine ``model`` or the cameras of the
    pitch engine ``calibration`` (``ClipResult.paint``).
    """
    scores = np.array([check.score for check in checks if check.score is not None], dtype=np.float64)
    return {"n_fits": len(checks), "n_cannot_judge": len(checks) - scores.size, "score": summarize(scores)}


def print_shots_and_paint(result: ClipResult, config: PipelineConfig) -> None:
    """Print the cuts of a clip and the paint-alignment scores of its keyframe mappings (diagnostics only).

    The frames of a transition (a shot of one frame: a frame of a dissolve, or a lone frame
    between two cuts) get their own line, with the keyframes among them: the pipeline drops
    what any engine answered for them, also a paid model answer.
    """
    info = result.info
    transitions = MotionField([None] * (info.n_frames - 1), result.cuts).transition_frames()
    starts = [frame for frame in result.cuts if frame not in transitions]
    print(f"cuts (frames that start a new shot): {', '.join(map(str, starts)) if starts else 'none'}")
    if transitions:
        keyframes = keyframe_indices(info.n_frames, info.fps, config.keyframe_interval_s)
        dropped = [frame for frame in keyframes if frame in transitions]
        print(
            f"transition frames (a dissolve, or a lone frame between two cuts), which carry nothing: "
            f"{', '.join(map(str, sorted(transitions)))}; {len(dropped)} of {len(keyframes)} keyframes among them, "
            f"their answers dropped{': ' + ', '.join(map(str, dropped)) if dropped else ''}"
        )
    summary = paint_summary(result.paint)
    score = summary["score"]
    if score["n"]:
        print(
            f"paint alignment of {summary['n_fits']} keyframe mappings (diagnostic): median {score['median']:.2f}, "
            f"mean {score['mean']:.2f}; {summary['n_cannot_judge']} frames cannot judge"
        )
    else:
        print(f"paint alignment: no score for the {summary['n_fits']} keyframe mappings (diagnostic)")


def _cmd_demo(args: argparse.Namespace) -> int:
    """Run a synthetic clip through the whole package with stand-ins of the model or of the tools.

    No network and no key are used. With the engine ``model`` the fake model answers the
    model steps. With the engine ``detector`` the stand-ins of the detector and of the
    calibration answer for every frame (``fallbacks.fake``), and no model step runs.

    The prediction file and the summary are written before the video, so that they
    exist also on a machine that cannot write the video. The exit code is 1 when the
    format checker finds a problem, and when no scored frame has a prediction.
    """
    scene = SyntheticScene(n_frames=args.frames, seed=args.seed)
    source = SyntheticSource(scene)
    info = scene.info
    config = PipelineConfig(engines=DEMO_ENGINES[args.engine])
    clock = StageClock()
    if args.engine == "detector":
        people, balls = fake_detector_answers(scene, seed=args.seed)
        tools = ToolAnswers(people=people, balls=balls, cameras=fake_calibration_answers(scene))
        result = run_clip(source, None, config, tools=tools, clock=clock)
    else:
        require_enough_keyframes(len(keyframe_indices(info.n_frames, info.fps, config.keyframe_interval_s)), config)
        result = run_clip(source, FakeBackend(scene, seed=args.seed), config, clock=clock)
    labels = scene_labels(scene)
    predictions = write_clip_predictions(
        result.frames,
        labels,
        args.out / PREDICTIONS_FOLDER,
        split=DEMO_SPLIT,
        tracker=DEFAULT_TRACKER,
        clip_id=info.clip_id,
    )
    problems = prediction_problems(predictions, labels)
    empty_frames, scored_frames = frames_without_prediction(predictions, labels)
    empty = empty_result_reason(result.frames, (empty_frames, scored_frames))
    truth = [scene.frame_state(frame) for frame in range(info.n_frames)]
    errors = summarize(position_errors_m(result.frames, truth))
    identity = identity_scores(result.frames, truth)
    timing = clock.report(info.n_frames)
    diagnostics = track_diagnostics(
        result,
        max_speed_mps=config.tracking.max_speed_mps,
        team_min_votes=SWAP_MIN_VOTES * keyframe_frames(config, info.fps),
    )
    summary = {
        "clip_id": info.clip_id,
        "engine": args.engine,
        "engines": result.engines,
        "n_frames": info.n_frames,
        "n_tracks": len(result.tracks),
        "median_position_error_m": errors["median"],
        "position_error_m": errors,
        "identity": dataclasses.asdict(identity),
        "frames_with_ball": sum(state.ball is not None for state in result.frames),
        "diagnostics": diagnostics.as_dict(),
        "checker": {"valid": not problems, "problems": problems},
        "scored_frames": scored_frames,
        "scored_frames_without_prediction": empty_frames,
        "timing": timing,
        "cuts": list(result.cuts),
        "paint_alignment": paint_summary(result.paint),
    }
    write_metrics(args.out / SUMMARY_NAME, summary, provenance={"command": "demo", "seed": args.seed})
    print(f"demo clip {info.clip_id}: {info.n_frames} frames, {len(result.tracks)} tracks, engine {args.engine}")
    print_engines(result.engines)
    print(f"median position error: {errors['median']:.2f} m over {errors['n']:.0f} positions")
    # The median position error does not see a person who gets a new track id. These values do.
    print(
        f"identity: {identity.coverage:.0%} of a true track under one id, {identity.switches} id switches, "
        f"{identity.track_ratio:.2f} predicted tracks per true track"
    )
    if args.engine == "detector":
        print(
            f"ball: {summary['frames_with_ball']} of {info.n_frames} frames "
            "(a stand-in ball that the picture does not show)"
        )
    print_shots_and_paint(result, config)
    print_diagnostics(diagnostics.lines())
    print_timing(timing)
    print(f"prediction file: {predictions}")
    print_problems(problems)
    print(f"scored frames without a prediction: {empty_frames} of {scored_frames}")
    if empty is not None:
        print_empty_result(empty)
    print(f"summary: {args.out / SUMMARY_NAME}")
    video = render_clip(source, result, args.out / DEMO_VIDEO_NAME)
    print(f"video: {video}")
    return EXIT_FAILED if problems or empty is not None else EXIT_OK


def print_engines(engines: Mapping[str, str]) -> None:
    """Print the engine of every step on one line. A number engine ``none`` gets a line of its own."""
    print("engines: " + ", ".join(f"{step} {engine}" for step, engine in engines.items()))
    if engines.get("number") == "none":
        print("shirt numbers: no step reads them (number engine none), so no player of this clip has a number")


def print_timing(timing: Mapping[str, Mapping[str, float | None]]) -> None:
    """Print the measured time of each stage: milliseconds per frame and frames per second."""
    print("measured time per stage (milliseconds per frame, frames per second):")
    for stage, values in timing.items():
        fps = values["frames_per_second"]
        rate = "-" if fps is None else f"{fps:.1f}"
        print(f"  {stage}: {values['ms_per_frame']:.2f} ms, {rate} fps")


# --- run -----------------------------------------------------------------------------


def print_diagnostics(lines: Sequence[str]) -> None:
    """Print the tracking values that need no labels (``evalx.diagnostics``)."""
    print("tracking values without labels (no gate uses them):")
    for line in lines:
        print(f"  {line}")


def require_enough_keyframes(n_keyframes: int, config: PipelineConfig) -> None:
    """Raise ``CliError`` when a clip has too few keyframes to give any track.

    A track needs ``config.tracking.min_observations`` observations, and a person is
    observed one time per keyframe. A shorter clip would pay for its model calls and
    give an empty result.
    """
    needed = config.tracking.min_observations
    if n_keyframes < needed:
        raise CliError(
            f"the clip has {n_keyframes} keyframes, but a track needs {needed} observations "
            "(tracking.min_observations): the clip is too short to give any track, so no model call is made"
        )


def model_plan(config: PipelineConfig, n_keyframes: int) -> list[PlannedCalls]:
    """List the model calls of a run for the engines of its settings: (step, settings, number of requests).

    The kit legend is asked when the people or the numbers come from the model. The number
    of number requests is a guess (``EXPECTED_NUMBER_REQUESTS``).
    """
    engines = config.engines
    plan: list[PlannedCalls] = []
    if engines.people == "model" or engines.number == "model":
        plan.append(("kit_legend", config.kit_legend, 1))
    if engines.people == "model":
        plan.append(("people", config.people, n_keyframes))
    if engines.pitch_mapping == "model":
        plan.append(("landmarks", config.landmarks, n_keyframes))
    if engines.number == "model":
        plan.append(("number", config.number, EXPECTED_NUMBER_REQUESTS))
    return plan


def check_tool_inputs(config: PipelineConfig, detections: Path | None, calibration: Path | None) -> None:
    """Raise ``CliError`` unless a clip run file is given for exactly the engines of the settings that read one."""
    named = (
        ("--detections", detections, config.engines.people == "detector", "the people engine detector"),
        ("--calibration", calibration, config.engines.pitch_mapping == "calibration", "the pitch engine calibration"),
    )
    for option, path, used, engine in named:
        if used and path is None:
            raise CliError(f"the settings name {engine}: give the clip run file of its tool with {option}")
        if not used and path is not None:
            raise CliError(f"{option} is given, but the settings do not name {engine}")


def read_clip_run(path: Path, settings: ToolSettings, info: ClipInfo, video_sha256: str | None) -> ClipRun:
    """Read all parts of a clip run, and check that it was made for this clip with the tool settings of the run.

    ``CliError`` is raised for a part that cannot be read, for wrong content, for other tool
    settings than the settings file names, for a run of another clip or another number of
    frames, and for a run on other pixels: ``video_sha256`` is the SHA-256 of the video file
    that the run reads, or None for a folder of images, and the clip run must name the same
    (``check_clip_run_video``).
    """
    try:
        run = load_clip_run(path)
    except OSError as error:
        raise CliError(f"the clip run file {path} cannot be read: {error}") from error
    except ValueError as error:
        raise wrong_file("clip run file", path, error) from error
    if run.tool != settings:
        raise CliError(
            f"the clip run file {path} was made with other tool settings than the settings file: "
            f"{run.tool.model_dump(mode='json')} against {settings.model_dump(mode='json')}. "
            "Write the clip job again and run the tool again"
        )
    if (run.clip, run.n_frames) != (info.clip_id, info.n_frames):
        raise CliError(
            f"the clip run file {path} is a run of the clip {run.clip} with {run.n_frames} frames, "
            f"not of the clip {info.clip_id} with {info.n_frames} frames"
        )
    check_clip_run_video(run, path, video_sha256)
    return run


def check_clip_run_video(run: ClipRun, path: Path, video_sha256: str | None) -> None:
    """Raise ``CliError`` unless a clip run was made from the pixels that the run reads.

    A tool that ran on the frames of a video (written by ``video-frames``) is used only with
    that video file, named by its SHA-256. A tool that ran on a folder of images that came as
    images is used only with a folder of images.
    """
    if run.video_sha256 == video_sha256:
        return
    if run.video_sha256 is None:
        raise CliError(
            f"the clip run file {path} was made from a folder of images, not from this video: write the frames of "
            "the video with the command video-frames, and run the tool on them"
        )
    if video_sha256 is None:
        raise CliError(
            f"the clip run file {path} was made from the frames of a video (SHA-256 {run.video_sha256}): give that "
            "video file with --clip"
        )
    raise CliError(
        f"the clip run file {path} was made from the frames of another video (SHA-256 {run.video_sha256}, this "
        f"video has {video_sha256}): write the frames of this video with video-frames and run the tool again"
    )


def tool_answers(
    config: PipelineConfig,
    info: ClipInfo,
    detections: Path | None,
    calibration: Path | None,
    clock: StageClock,
    video_sha256: str | None,
) -> tuple[ToolAnswers, dict[str, Any]]:
    """Read the clip run files of the engines of a run. Return the answers and the facts of each run.

    ``video_sha256`` names the video file of the run, or is None for a folder of images
    (``read_clip_run``).

    The time of each tool on the frames of the clip goes to ``clock`` as the stage
    ``detector`` or ``calibration``: the script measured it per frame. A payload that cannot
    be used raises ``CliError`` with the file and the frame.
    """
    facts: dict[str, Any] = {}
    people = balls = cameras = None
    if detections is not None:
        run = read_clip_run(detections, config.detector, info, video_sha256)
        try:
            people, balls = detector_answers(run)
        except ValueError as error:
            raise wrong_file("clip run file", detections, error) from error
        clock.add("detector", math.fsum(run.seconds()))
        facts["detections"] = clip_run_facts(run, detections)
    if calibration is not None:
        run = read_clip_run(calibration, config.calibration, info, video_sha256)
        try:
            cameras = calibration_answers(run)
        except ValueError as error:
            raise wrong_file("clip run file", calibration, error) from error
        clock.add("calibration", math.fsum(run.seconds()))
        facts["calibration"] = clip_run_facts(run, calibration)
    return ToolAnswers(people=people, balls=balls, cameras=cameras), facts


def clip_run_facts(run: ClipRun, path: Path) -> dict[str, Any]:
    """Return what the result file records about a clip run: its file, its job, the facts of the run and its times."""
    n_parts = 1
    while clip_run_part_path(path, n_parts + 1).exists():
        n_parts += 1
    return {
        "file": path.name,
        "parts": n_parts,
        "job_sha256": run.job_sha256,
        "video_sha256": run.video_sha256,
        "runner": run.runner,
        "script_is_current": run.runner.get("script_sha256") == file_sha256(repository_root() / TOOL_SCRIPT),
        "seconds_per_frame": summarize(np.array(run.seconds(), dtype=np.float64)),
    }


def _cmd_run(args: argparse.Namespace) -> int:
    """Run one clip through the pipeline and write its result file, its prediction file and its video.

    The files of a clip go to ``<out>/<clip id>``. The prediction files of all clips go
    to ``<out>/predictions``, in the layout of the scorer. With a label file, the result
    file holds what the format checker found in the prediction file and the number of
    scored frames without a prediction. Such a frame is no format problem: it does not
    change the exit code as long as another scored frame has a prediction.

    The exit code is 1 when the format checker finds a problem, and when the clip gives
    no result at all (``empty_result_reason``): no scored frame has a prediction, or,
    without a label file, no frame holds a person. The files are written also then: an
    empty prediction file lets the scorer count the clip as missed.
    """
    config = PipelineConfig() if args.config is None else read_settings(args.config)
    check_tool_inputs(config, args.detections, args.calibration)
    roster = None if args.roster is None else read_roster(args.roster)
    labels = None if args.labels is None else read_labels(args.labels)
    state = require_clean_tree(args.allow_dirty)
    source = open_clip(args.clip, fps=args.fps, labels=labels)
    if labels is not None:
        check_labels_match(source, labels)
    info = source.info
    keyframes = keyframe_indices(info.n_frames, info.fps, config.keyframe_interval_s)
    print(
        f"clip {info.clip_id}: {info.n_frames} frames, {info.fps:g} frames per second, "
        f"{info.width} x {info.height} pixels, {len(keyframes)} keyframes"
    )
    if config.engines.people == "model":
        require_enough_keyframes(len(keyframes), config)
    clock = StageClock()
    uses_tools = args.detections is not None or args.calibration is not None
    video_sha256 = file_sha256(args.clip) if uses_tools and args.clip.is_file() else None
    tools, tool_runs = tool_answers(config, info, args.detections, args.calibration, clock, video_sha256)
    plan = model_plan(config, len(keyframes))
    session = None
    if plan:
        session = open_session(
            "run",
            info.clip_id,
            plan,
            cache_dir=_cache_dir(args),
            replay=args.replay,
            repeat=config.repeat,
            over_budget=args.start_over_budget,
            allow_other_images=args.allow_other_images,
        )
    else:
        print("no step of the settings asks a model: this run makes no model call and costs nothing")
    if config.engines.number == "model":
        print(
            f"the {EXPECTED_NUMBER_REQUESTS} number requests of the estimate are a guess: the run sends one for each "
            "track of a player or a goalkeeper, and the tracks are known only after the people step"
        )
    cost: dict[str, Any] | None = None
    try:
        result = run_clip(
            source, None if session is None else session.backend, config, roster=roster, tools=tools, clock=clock
        )
    finally:
        if session is not None:
            cost = close_session(session)
    print_shots_and_paint(result, config)
    clip_dir = args.out / info.clip_id
    provenance = {
        "command": "run",
        "commit": state.commit,
        "dirty": state.dirty,
        "package_version": __version__,
        "written_at": utc_now_text(),
        "sdk_version": installed_sdk_version(),
        "prompts": dict(PROMPT_NAMES),
        "config": config.model_dump(mode="json"),
        "roster": roster is not None,
        "cost": cost,
        "tool_runs": tool_runs,
    }
    problems: list[str] = []
    prediction_file: dict[str, Any] | None = None
    scored: tuple[int, int] | None = None
    if labels is not None:
        predictions = write_clip_predictions(
            result.frames,
            labels,
            args.out / PREDICTIONS_FOLDER,
            split=args.split,
            tracker=args.tracker,
            clip_id=info.clip_id,
        )
        problems = prediction_problems(predictions, labels)
        scored = frames_without_prediction(predictions, labels)
        empty_frames, scored_frames = scored
        print(f"prediction file: {predictions}")
        print_problems(problems)
        print(f"scored frames without a prediction: {empty_frames} of {scored_frames}")
        prediction_file = {
            "problems": problems,
            "scored_frames": scored_frames,
            "scored_frames_without_prediction": empty_frames,
        }
    empty = empty_result_reason(result.frames, scored)
    if empty is not None:
        print_empty_result(empty)
    timing = clock.report(info.n_frames)
    print_engines(result.engines)
    if config.engines.people == "detector":
        print(f"ball: {sum(frame.ball is not None for frame in result.frames)} of {info.n_frames} frames")
    diagnostics = track_diagnostics(
        result,
        max_speed_mps=config.tracking.max_speed_mps,
        team_min_votes=SWAP_MIN_VOTES * keyframe_frames(config, info.fps),
    )
    print_diagnostics(diagnostics.lines())
    print_timing(timing)
    document = {
        "provenance": provenance,
        "prediction_file": prediction_file,
        "timing": timing,
        "diagnostics": diagnostics.as_dict(),
        "result": result.model_dump(mode="json"),
    }
    write_json(clip_dir / RESULT_NAME, document)
    print(f"result: {clip_dir / RESULT_NAME} ({len(result.tracks)} tracks)")
    video = render_clip(source, result, clip_dir / f"{info.clip_id}.mp4")
    print(f"video: {video}")
    return EXIT_FAILED if problems or empty is not None else EXIT_OK


# --- stream --------------------------------------------------------------------------


#: What the time tables of the streaming path hold, printed with them and written into the result file.
STREAM_TIME_NOTE = (
    "the package times hold no detector and no calibration: the tools ran before (a clip run) or in "
    "another process (a pipe), and the stand-ins of the synthetic clip take no time"
)


#: Lines of a stream that ``stream --lines`` reads ahead on its thread before the reading waits: 2 s at 25 fps.
STREAM_QUEUE_LINES = 50


def _frames_or_clip(text: str) -> int | None:
    """Read a number of frames (0 or more), or ``clip`` for the whole clip (None)."""
    if text == "clip":
        return None
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number of frames or 'clip', got {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected 0 frames or more, got {value}")
    return value


def _history(text: str) -> int | None | Literal["default"]:
    """Read ``--history``: a number of frames, ``clip``, or ``default``."""
    return "default" if text == "default" else _frames_or_clip(text)


def stream_settings(args: argparse.Namespace, config: PipelineConfig, fps: float) -> StreamSettings:
    """Return the delay, the window and the kit warm-up of a ``stream`` command.

    ``--history default`` (the default) is ``stream.default_history_frames``; a delay of the
    whole clip takes a window of the whole clip.
    """
    delay = args.delay
    history = args.history
    if history == "default":
        history = None if delay is None else default_history_frames(config, fps)
    try:
        return StreamSettings(delay_frames=delay, history_frames=history, kit_warmup_frames=args.kit_warmup)
    except ValueError as error:
        raise CliError(str(error)) from error


def stream_inputs(
    args: argparse.Namespace, config: PipelineConfig, info: ClipInfo, video_sha256: str | None
) -> tuple[Iterator[StreamInput], dict[str, Any], QueuedLines | None]:
    """Return the inputs of a ``stream`` command, the facts of where they come from, and the queue of a pipe.

    From two clip runs (``--detections``, ``--calibration``) the lines of the stream are
    written in memory (``stream.clip_run_lines``) and read back, the path that a pipe takes.
    From ``--lines`` (a file, or ``-`` for the standard input) the lines are read one by one,
    on a thread into a queue of ``--queue`` lines (``stream.QueuedLines``). The header must name
    the clip, its number of frames, its video (the SHA-256 of ``--clip``, or none for a folder of
    images) and the tool settings of the settings file.
    """
    facts: dict[str, Any] = {}
    queued: QueuedLines | None = None
    if args.lines is not None:
        # Under the overload policy the queue has no bound: its reader never waits, so the policy sees every wait.
        limit = None if args.max_behind is not None else args.queue or STREAM_QUEUE_LINES
        if str(args.lines) == "-":
            queued = QueuedLines(fd_lines(sys.stdin.fileno()), limit)
            facts["lines"] = "standard input"
        else:
            queued = QueuedLines(_lines_of(args.lines), limit)
            facts["lines"] = args.lines.name
        lines: Iterable[str] = queued
    else:
        if args.detections is None or args.calibration is None:
            raise CliError("stream --clip needs --detections and --calibration, or --lines")
        detections = read_clip_run(args.detections, config.detector, info, video_sha256)
        calibration = read_clip_run(args.calibration, config.calibration, info, video_sha256)
        facts["detections"] = clip_run_facts(detections, args.detections)
        facts["calibration"] = clip_run_facts(calibration, args.calibration)
        lines = clip_run_lines(detections, calibration)
    try:
        header, inputs = read_stream_lines(lines)
    except ValueError as error:
        raise CliError(f"the stream cannot be read: {error}") from error
    if header.clip != info.clip_id:
        raise CliError(f"the stream is a stream of the clip {header.clip}, not of the clip {info.clip_id}")
    if header.n_frames != info.n_frames:
        raise CliError(f"the stream names {header.n_frames} frames, the clip has {info.n_frames} frames")
    if header.video_sha256 != video_sha256:
        raise CliError(
            "the stream was made from other pixels than --clip: it names the video "
            f"{header.video_sha256 or 'none (images)'}, and --clip is {video_sha256 or 'a folder of images'}"
        )
    if (header.detector, header.calibration) != (config.detector, config.calibration):
        raise CliError("the stream was made with other tool settings than the settings file")
    return _checked_inputs(inputs), facts, queued


def _lines_of(path: Path) -> Iterator[str]:
    """Yield the lines of a file one by one, and close it at the end (or when the reader stops)."""
    try:
        with path.open(encoding="utf-8") as handle:
            yield from handle
    except OSError as error:
        raise CliError(f"the lines of the stream cannot be read from {path}: {error}") from error


def _checked_inputs(inputs: Iterator[StreamInput]) -> Iterator[StreamInput]:
    """Pass the inputs on, and turn an error of a line into ``CliError``."""
    try:
        yield from inputs
    except ValueError as error:
        raise CliError(f"the stream cannot be read: {error}") from error


def print_frame_timing(report: Mapping[str, Mapping[str, float]], tools: Mapping[str, Sequence[float]]) -> None:
    """Print the per-frame time of each stage: median, 95th percentile and largest, milliseconds."""
    print("measured time per frame (median / 95th percentile / largest, milliseconds):")
    for stage, values in report.items():
        print(f"  {stage}: {values['p50_ms']:.2f} / {values['p95_ms']:.2f} / {values['max_ms']:.2f}")
    for tool, seconds in tools.items():
        if seconds:
            values = percentiles_ms(seconds)
            print(
                f"  {tool} (as the tool measured it, not in the total): {values['p50_ms']:.2f} / "
                f"{values['p95_ms']:.2f} / {values['max_ms']:.2f}"
            )
    print(f"  note: {STREAM_TIME_NOTE}")


def check_stream_options(args: argparse.Namespace) -> None:
    """Refuse the options of ``stream`` that do not apply to its input, so that none is ignored without a word."""
    if args.synthetic:
        given = [
            name
            for name, value in (
                ("--clip", args.clip),
                ("--lines", args.lines),
                ("--detections", args.detections),
                ("--calibration", args.calibration),
                ("--labels", args.labels),
                ("--fps", args.fps),
                ("--split", args.split),
                ("--queue", args.queue),
                ("--max-behind", args.max_behind),
            )
            if value
        ]
        if given:
            raise CliError(f"stream --synthetic takes no {', '.join(given)}")
        return
    given = [name for name, value in (("--frames", args.frames), ("--seed", args.seed)) if value is not None]
    if given:
        raise CliError(f"{' and '.join(given)} apply only to stream --synthetic")
    if args.lines is not None and (args.detections is not None or args.calibration is not None):
        raise CliError("stream --lines takes no --detections or --calibration: the lines hold the answers of both tools")
    if args.lines is None and args.queue is not None:
        raise CliError("--queue applies only to stream --lines")
    if args.max_behind is not None:
        if args.delay is None:
            raise CliError("--max-behind needs a delay in frames: with --delay clip no frame is decided while it streams")
        live = args.lines is not None and (_stdin_is_a_pipe() if str(args.lines) == "-" else args.lines.is_fifo())
        if not live:
            raise CliError(
                "--max-behind applies only to a live stream (--lines - from a pipe, or a named pipe): the clip runs and a "
                "file of lines (also on the standard input) have no clock of arrival, so no frame of them is ever dropped"
            )
        if args.queue is not None:
            raise CliError("--max-behind takes no --queue: under the overload policy the queue of lines has no bound")
    if args.labels is None and (args.split is not None or args.tracker is not None):
        raise CliError("--split and --tracker name the prediction file, which stream writes only with --labels")
    if args.clip is not None and args.clip.is_file() and args.fps is not None:
        raise CliError("--fps applies only to a folder of frame images: a video file has its own frame rate")


def overload_document(policy: OverloadPolicy, stream: DetectorStream, n_frames: int) -> dict[str, Any]:
    """Return the section ``overload`` of the result file: the setting, the dropped frames, their share, the waits."""
    dropped = stream.overload_report()
    return {
        "max_behind_s": policy.max_behind_s,
        "frames": n_frames,
        **dropped,
        "dropped_share": dropped["dropped"] / n_frames,
        "line_wait_at_the_feeder": percentiles_ms(policy.waits_s) if policy.waits_s else None,
        "line_wait_at_the_frame_path": percentiles_ms(policy.late_waits_s) if policy.late_waits_s else None,
    }


def _stdin_is_a_pipe() -> bool:
    """Tell whether the standard input is a pipe or a socket (a live feed), not a file, a device or nothing."""
    try:
        mode = os.fstat(sys.stdin.fileno()).st_mode
    except (OSError, ValueError, io.UnsupportedOperation):
        return False
    return stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)


def print_latency(report: Mapping[str, Any]) -> None:
    """Print the latency of a live stream (``timing.LatencyLog``): median / 95th percentile / largest, milliseconds."""
    print(f"measured latency of the {int(report['decided_live'])} frames decided while the stream ran (median / 95th "
          "percentile / largest, milliseconds):")
    for name, text in (
        ("source_to_decision", "from the arrival of a frame at the tools to its decision"),
        ("after_last_needed_frame", "from the arrival of frame f + delay to the decision of f"),
        ("line_to_decision", "from the reading of the line of frame f + delay to the decision of f"),
    ):
        values = report.get(name)
        if values:
            print(f"  {text}: {values['p50_ms']:.1f} / {values['p95_ms']:.1f} / {values['max_ms']:.1f}")
    backlog = report.get("backlog_lines")
    if backlog:
        print(f"  lines waiting in the queue: median {backlog['p50']:g}, largest {backlog['max']:g}")


def _cmd_stream(args: argparse.Namespace) -> int:
    """Run the streaming path of the detector engine on a clip run, a stream of lines or the synthetic clip.

    The state of frame ``t`` is decided once frame ``t + delay`` is in (``stream``). The files
    are those of ``run``: ``<out>/<clip id>/result.json``, the prediction file with a label
    file (the synthetic clip has its own labels), and the video. The result file also holds
    the per-frame times (``frame_timing``). The exit code follows ``run``. ``--live-out``
    writes each decided frame as it is decided (``liveout``); with ``--live-out -`` the
    standard output carries those lines, so the output of the command goes to the error stream.
    """
    with contextlib.ExitStack() as stack:
        if args.live_out == "-":
            stack.enter_context(contextlib.redirect_stdout(sys.stderr))
        return _stream_command(args, stack)


#: Folders whose entries are the open files of a process (macOS, Linux): a ``--live-out`` there is refused.
_DESCRIPTOR_FOLDERS = ("/dev/fd/", "/proc/self/fd/", "/proc/thread-self/fd/")


def refuse_live_out(args: argparse.Namespace) -> None:
    """Refuse a ``--live-out`` path that lies in the repository or in ``--out``, a folder, an input file or the output.

    The lines hold the boxes and pitch positions of every frame of a clip; for a clip of the
    dataset they never go into a work tree of the repository (rule 2 of ``CLAUDE.md``), where
    ``git add`` could take them. The check is the one of ``refuse_pixels_in_repository``: the
    file (through its links) and each folder above it, as places on the disk. A path that is
    the standard output itself (``/dev/stdout``) must be given as ``-``, so that the output of
    the command goes elsewhere.
    """
    if args.live_out is None or args.live_out == "-":
        return
    path = Path(args.live_out)
    if path.is_dir():
        raise CliError(f"--live-out {path} is a folder: give a file, a named pipe or -")
    real = os.path.realpath(path)
    descriptor = real in ("/dev/stdin", "/dev/stdout", "/dev/stderr") or real.startswith(_DESCRIPTOR_FOLDERS)
    if descriptor or (path.exists() and os.path.samestat(os.stat(path), os.fstat(1))):
        raise CliError(f"--live-out {path} is the standard output or another open file of the command: give - instead")
    target = path.resolve()
    if _lies_in(target, args.out.resolve()):
        raise CliError(f"--live-out {path} lies in --out {args.out}: the command writes its result files there")
    existing = [folder for folder in (target, *target.parents) if folder.exists()]
    for tree in [repository_root(), *repository_work_trees()]:
        if tree.exists() and any(os.path.samefile(folder, tree) for folder in existing):
            raise CliError(
                f"--live-out {path} lies inside the work tree {tree} of the repository: the frame lines of a clip "
                f"never go into the repository (rule 2 of CLAUDE.md). Write them below the data folder, for example "
                f"{DATA_DIR}/live"
            )
    inputs = {
        "--clip": args.clip,
        "--labels": args.labels,
        "--config": args.config,
        "--detections": args.detections,
        "--calibration": args.calibration,
        "--lines": None if args.lines is None or str(args.lines) == "-" else args.lines,
    }
    written = path.resolve()
    for option, given in inputs.items():
        if given is not None and (written == given.resolve() or (path.exists() and given.exists() and path.samefile(given))):
            raise CliError(f"--live-out names the same file as {option} ({path}): the command would replace its input")


def _close_live_out(live_out: LiveOut) -> None:
    """Close the live output of a ``stream`` that stopped early, without waiting for its reader."""
    if not live_out.closed:
        live_out.close(wait_s=0.0)


def _stream_command(args: argparse.Namespace, stack: contextlib.ExitStack) -> int:
    """Run ``stream`` (``_cmd_stream``); ``stack`` closes the live output when the command stops early."""
    config = PipelineConfig(engines=DEMO_ENGINES["detector"]) if args.config is None else read_settings(args.config)
    try:
        check_stream_engines(config)
    except ValueError as error:
        raise CliError(str(error)) from error
    check_stream_options(args)
    check_number_options(args, config)
    refuse_live_out(args)
    state = require_clean_tree(args.allow_dirty)
    scene: SyntheticScene | None = None
    labels: GsrLabels | None
    queued: QueuedLines | None = None
    if args.synthetic:
        frames = DEMO_FRAMES if args.frames is None else args.frames
        seed = 0 if args.seed is None else args.seed
        scene = SyntheticScene(n_frames=frames, seed=seed)
        source: FrameSource = SyntheticSource(scene)
        labels = scene_labels(scene)
        people, balls = fake_detector_answers(scene, seed=seed)
        inputs: Iterator[StreamInput] = stand_in_inputs(people, balls, fake_calibration_answers(scene))
        facts: dict[str, Any] = {"synthetic": {"seed": seed, "frames": frames}}
    else:
        if args.clip is None:
            raise CliError("stream needs --clip (with --detections and --calibration, or --lines), or --synthetic")
        labels = None if args.labels is None else read_labels(args.labels)
        source = open_clip(args.clip, fps=args.fps, labels=labels)
        if labels is not None:
            check_labels_match(source, labels)
        video_sha256 = file_sha256(args.clip) if args.clip.is_file() else None
        inputs, facts, queued = stream_inputs(args, config, source.info, video_sha256)
    info = source.info
    settings = stream_settings(args, config, info.fps)
    print(
        f"clip {info.clip_id}: {info.n_frames} frames, {info.fps:g} frames per second; streaming path with a "
        f"delay of {frames_text(settings.delay_frames, info.fps)} and a window of "
        f"{frames_text(settings.history_frames, info.fps)} before the decided frame"
    )
    numbers: LiveNumbers | None = None
    numbers_closer: _NumbersCloser | None = None
    if config.engines.number == "model":
        numbers_closer = open_live_numbers(args, config, info, scene, settings, stack)
        numbers = numbers_closer.numbers
        assert numbers is not None
    else:
        print("no step of the settings asks a model: this run makes no model call and costs nothing")
    clock = StageClock()
    stream = DetectorStream(
        config, settings, clip_id=info.clip_id, fps=info.fps, size=(info.width, info.height), clock=clock
    )
    live_out: LiveOut | None = None
    if args.live_out is not None:
        if args.live_out != "-" and Path(args.live_out).is_fifo():
            print(f"live output: waiting for a reader of the named pipe {args.live_out}")
        live_out = open_live_out(args.live_out, info)
        stack.callback(_close_live_out, live_out)
        if args.max_behind is not None:
            live_out.overload(max_behind_s=args.max_behind)
            stream.on_dropped = live_out.skipped
        live_out.start(
            delay_frames=settings.delay_frames, history_frames=settings.history_frames, numbers=numbers is not None
        )
    writer = wire(stream, numbers, live_out)
    latency = LatencyLog(settings.delay_frames)
    missing: dict[str, dict[str, int]] = {}
    policy = None if args.max_behind is None else OverloadPolicy(args.max_behind)

    def read_frame(frame: int) -> np.ndarray:
        if frame >= info.n_frames:
            raise CliError(f"the stream holds frame {frame}, the clip has {info.n_frames} frames")
        return source.read(frame)

    feeder = FrameFeeder(
        stream,
        inputs,
        read_frame,
        line_clock=None if queued is None else (lambda: (queued.last_read_at, queued.last_backlog)),
        policy=policy,
    )
    stack.callback(feeder.close)
    feeder_seconds: dict[str, list[float]] = {"read": [], "motion": []}
    for fed in feeder:
        item = fed.item
        if item.frame >= info.n_frames:
            raise CliError(f"the stream holds frame {item.frame}, the clip has {info.n_frames} frames")
        latency.arrived(item.frame, source_at=item.arrived_at, line_at=fed.read_at, backlog=fed.backlog)
        for tool, reason in item.missing:
            missing.setdefault(tool, {}).setdefault(reason, 0)
            missing[tool][reason] += 1
        if fed.image is not None:  # read and measured (also when the second look of the policy did not use it)
            feeder_seconds["read"].append(fed.read_seconds)
            feeder_seconds["motion"].append(fed.motion_seconds)
        for given in stream.push_fed(fed):
            if not stream.is_dropped(given.frame):
                latency.given(given.frame, wall=time.time(), mono=time.monotonic())
    if stream.n_frames != info.n_frames:
        raise CliError(f"the stream ended after {stream.n_frames} frames, the clip has {info.n_frames} frames")
    stream.finish()
    if numbers_closer is not None:
        last_lines = numbers_closer.finish()
        if writer is not None:
            writer.write(last_lines, stream.n_frames - 1)
    if live_out is not None:
        live_out.finish()
    result = stream.result()
    if numbers is not None:
        result = apply_numbers(result, numbers.lines)
    if missing:
        parts = "; ".join(
            f"{tool}: " + ", ".join(f"{count} {reason}" for reason, count in sorted(reasons.items()))
            for tool, reasons in sorted(missing.items())
        )
        print(f"frames with a missing part of a tool: {parts}")
    print_shots_and_paint(result, config)
    clip_dir = args.out / info.clip_id
    provenance = {
        "command": "stream",
        "commit": state.commit,
        "dirty": state.dirty,
        "package_version": __version__,
        "written_at": utc_now_text(),
        "config": config.model_dump(mode="json"),
        "stream": dataclasses.asdict(settings),
        "inputs": facts,
    }
    if numbers_closer is not None:
        provenance["cost"] = numbers_closer.cost
    problems: list[str] = []
    prediction_file: dict[str, Any] | None = None
    scored: tuple[int, int] | None = None
    if labels is not None:
        predictions = write_clip_predictions(
            result.frames,
            labels,
            args.out / PREDICTIONS_FOLDER,
            split=DEMO_SPLIT if scene is not None else (args.split or DEFAULT_SPLIT),
            tracker=args.tracker or DEFAULT_TRACKER,
            clip_id=info.clip_id,
        )
        problems = prediction_problems(predictions, labels)
        scored = frames_without_prediction(predictions, labels)
        empty_frames, scored_frames = scored
        print(f"prediction file: {predictions}")
        print_problems(problems)
        print(f"scored frames without a prediction: {empty_frames} of {scored_frames}")
        prediction_file = {
            "problems": problems,
            "scored_frames": scored_frames,
            "scored_frames_without_prediction": empty_frames,
        }
    empty = empty_result_reason(result.frames, scored)
    if empty is not None:
        print_empty_result(empty)
    print_engines(result.engines)
    print(f"ball: {sum(frame.ball is not None for frame in result.frames)} of {info.n_frames} frames")
    identity: dict[str, Any] | None = None
    if scene is not None:
        truth = [scene.frame_state(frame) for frame in range(info.n_frames)]
        scores = identity_scores(result.frames, truth)
        identity = dataclasses.asdict(scores)
        print(
            f"identity against the synthetic truth: {scores.coverage:.0%} of a true track under one id, "
            f"{scores.switches} id switches, {scores.track_ratio:.2f} predicted tracks per true track"
        )
    diagnostics = track_diagnostics(
        result,
        max_speed_mps=config.tracking.max_speed_mps,
        team_min_votes=SWAP_MIN_VOTES * keyframe_frames(config, info.fps),
    )
    print_diagnostics(diagnostics.lines())
    timing = clock.report(info.n_frames)
    frame_timing = clock.frame_report()
    print_timing(timing)
    print_frame_timing(frame_timing, stream.tool_seconds)
    feeder_timing = {name: percentiles_ms(seconds) for name, seconds in feeder_seconds.items() if seconds}
    for name, values in feeder_timing.items():
        print(
            f"  {name} (on the thread of the frame feeder, not in the total): {values['p50_ms']:.2f} / "
            f"{values['p95_ms']:.2f} / {values['max_ms']:.2f}"
        )
    overload = None if policy is None else overload_document(policy, stream, info.n_frames)
    if overload is not None:
        print(
            f"overload policy ({stream.overload}): "
            f"{overload['dropped']} of {info.n_frames} frames dropped ({len(overload['not_used'])} not used, "
            f"{len(overload['decided_late'])} measured but decided late); a dropped frame has no state"
        )
    paint_timing = percentiles_ms(stream.paint_seconds) if stream.paint_seconds else None
    if paint_timing is not None:
        print(
            f"  paint score (on its own thread, not in the total): {paint_timing['p50_ms']:.2f} / "
            f"{paint_timing['p95_ms']:.2f} / {paint_timing['max_ms']:.2f}"
        )
    live = latency.report() if queued is not None else None
    if live is not None:
        print_latency(live)
    tool_timing = {name: percentiles_ms(seconds) for name, seconds in stream.tool_seconds.items() if seconds}
    live_report = None if live_out is None else live_out.close()
    if live_report is not None:
        print(live_report.text())
    document = {
        "provenance": provenance,
        "prediction_file": prediction_file,
        "timing": timing,
        "frame_timing": frame_timing,
        "feeder_timing": feeder_timing,
        "tool_timing": tool_timing,
        "paint_timing": paint_timing,
        "latency": live,
        "missing_parts": missing,
        "transitions_known_at": [
            {"kind": kind, "start": span[0], "stop": span[1], "newest_frame": frame}
            for kind, span, frame in stream.transition_log
        ],
        "timing_note": STREAM_TIME_NOTE,
        "identity": identity,
        "diagnostics": diagnostics.as_dict(),
        "result": result.model_dump(mode="json"),
    }
    if live_report is not None:
        document["live_out"] = dataclasses.asdict(live_report)
    if overload is not None:
        document["overload"] = overload
    numbers_failed = False
    if numbers is not None:
        numbers_failed = print_live_numbers(numbers)
        document["numbers"] = {
            **numbers.report(),
            "lines": [dataclasses.asdict(line) for line in numbers.lines],
        }
        if live_out is not None:
            document["numbers"]["live_out"] = {
                "given": live_out.numbers_given,
                "written": live_out.numbers_written,
                "undelivered": live_out.numbers_undelivered,
            }
    write_json(clip_dir / RESULT_NAME, document)
    print(f"result: {clip_dir / RESULT_NAME} ({len(result.tracks)} tracks)")
    video_path = clip_dir / f"{info.clip_id}.mp4"
    if stream.dropped_frames:
        video = render_clip(source, result, video_path, dropped=stream.dropped_frames)
    else:
        video = render_clip(source, result, video_path)
    print(f"video: {video}")
    return EXIT_FAILED if problems or empty is not None or numbers_failed else EXIT_OK


# --- stream: the live number step -----------------------------------------------------


def check_number_options(args: argparse.Namespace, config: PipelineConfig) -> None:
    """Refuse the options of the live number step that do not apply, so that none is ignored without a word."""
    model_options = (
        ("--cache", args.cache),
        ("--replay", args.replay),
        ("--start-over-budget", args.start_over_budget),
        ("--allow-other-images", args.allow_other_images),
    )
    if config.engines.number != "model":
        given = [name for name, value in (("--numbers-max", args.numbers_max is not None), *model_options) if value]
        if given:
            raise CliError(f"{', '.join(given)} apply only to the number engine model (the settings name none)")
        return
    if args.synthetic:
        given = [name for name, value in model_options if value]
        if given:
            raise CliError(
                f"stream --synthetic takes no {', '.join(given)}: its number step asks the fake model of the scene"
            )


@dataclass
class _NumbersCloser:
    """The live number step of a ``stream`` and its model session: closed once, at the end or on an early stop."""

    session: ModelSession | None
    numbers: LiveNumbers | None = None
    cost: dict[str, Any] | None = None
    closed: bool = False

    def finish(self) -> list[Any]:
        """Ask the open tracks, wait for every answer, then close the session (the measured cost)."""
        assert self.numbers is not None
        try:
            return self.numbers.finish()
        finally:
            self._close_session()

    def abort(self) -> None:
        """An early stop: send nothing more, wait for the requests in flight, then close the session."""
        if self.closed:
            return
        try:
            if self.numbers is not None:
                self.numbers.abort()
        finally:
            self._close_session()

    def _close_session(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.session is not None:
            self.cost = close_session(self.session)


def open_live_numbers(
    args: argparse.Namespace,
    config: PipelineConfig,
    info: ClipInfo,
    scene: SyntheticScene | None,
    settings: StreamSettings,
    stack: contextlib.ExitStack,
) -> _NumbersCloser:
    """Start the live number step of ``stream`` (``livenumbers``), before the first frame and the first model call.

    The estimate is bounded by the cap: one kit legend request and ``--numbers-max`` number
    requests. The synthetic clip asks the fake model of its scene (no ledger, no cost); any other
    clip opens a model session: the estimate is printed before the first call, the cost ledger of
    the cache folder records every answer, and the run stops above 1.3 times the estimate.
    """
    cap = DEFAULT_MAX_REQUESTS if args.numbers_max is None else args.numbers_max
    history = settings.history_frames
    # A history of 0 frames: no window holds an earlier frame, so an id that is not seen in a frame is closed.
    number_settings = NumberSettings(max_requests=cap, close_frames=None if history is None else max(1, history))
    session: ModelSession | None = None
    if scene is not None:
        backend: Any = FakeBackend(scene, seed=0 if args.seed is None else args.seed, coords=config.number.coords)
        print(
            f"shirt numbers: the live number step asks the fake model of the synthetic scene (no model call, no cost), "
            f"at most {cap} number requests"
        )
    else:
        plan: list[PlannedCalls] = [("kit_legend", config.kit_legend, 1), ("number", config.number, cap)]
        session = open_session(
            "stream",
            info.clip_id,
            plan,
            cache_dir=_cache_dir(args),
            replay=args.replay,
            repeat=config.repeat,
            over_budget=args.start_over_budget,
            allow_other_images=args.allow_other_images,
        )
        backend = session.backend
        print(
            f"shirt numbers: at most {cap} number requests in this stream (--numbers-max), the bound of the estimate; "
            "a track that is ready after the cap gets no number, and the stream goes on"
        )
    closer = _NumbersCloser(session=session)
    stack.callback(closer.abort)  # from here on, any stop closes the session and prints its cost
    closer.numbers = LiveNumbers(config, backend, number_settings, info=info)
    return closer


def print_live_numbers(numbers: LiveNumbers) -> bool:
    """Print what the live number step did; return True when it failed (the exit code of the command)."""
    report = numbers.report()
    print(
        f"shirt numbers: {report['number_lines']} number lines ({report['with_number']} with a number) from "
        f"{report['requests_planned']} number requests of at most {report['max_requests']}; "
        f"images dropped by the worker: {report['images_dropped']}"
    )
    if numbers.cap_reached:
        print(
            f"shirt numbers: the cap of {report['max_requests']} number requests was reached: "
            f"{report['refused_by_cap']} tracks got no number request"
        )
    if report["failure"] is not None:
        print(f"shirt numbers: the number step stopped asking after a failure: {report['failure']}")
        return True
    return False


# --- radar --------------------------------------------------------------------------


def radar_lines(source: str, stop: threading.Event) -> Iterator[str]:
    """Return the lines of the feed of ``radar``: the standard input, a named pipe, or a file that grows."""
    if source == "-":
        return radar.read_lines(sys.stdin.fileno())
    path = Path(source)
    if not path.exists():
        raise CliError(f"the feed {source} is not there: start the viewer after the stream made it, or give a named pipe")
    if path.is_dir():
        raise CliError(f"the feed {source} is a folder")
    if path.is_fifo():
        return _fifo_lines(path)
    return radar.follow_file(source, stop)


def _fifo_lines(path: Path) -> Iterator[str]:
    """Yield the lines of a named pipe; it is opened only when the first line is wanted (the open waits for a writer)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        yield from radar.read_lines(fd)
    finally:
        os.close(fd)


def _cmd_radar(args: argparse.Namespace) -> int:
    """Serve a live 2D radar of a live output (``stream --live-out``) to a browser on this machine (``radar``).

    The server binds to a loopback address only and serves one page and one event stream. It
    ends when the feed ended and the pages got the end, or with Ctrl-C.
    """
    try:
        radar.check_loopback(args.host)
    except radar.RadarError as error:
        raise CliError(str(error)) from error
    stop = threading.Event()
    lines = radar_lines(args.feed, stop)

    def ready(server: radar.RadarServer) -> None:
        print(f"radar: {server.url} (this machine only; open it in a browser, Ctrl-C ends the viewer)", flush=True)

    try:
        frames, board = radar.serve(lines, host=args.host, port=args.port, replay=args.replay, ready=ready, stop=stop)
    except radar.RadarError as error:
        raise CliError(str(error)) from error
    finally:
        stop.set()
    print(f"the feed ended: {frames} frame lines, {board.served} event streams served")
    return EXIT_OK


def _add_radar(commands: Any) -> None:
    """Add the command ``radar`` to the sub-parsers object of argparse."""
    parser = commands.add_parser(
        "radar", help="serve a live 2D radar of the live output of stream (--live-out) to a browser on this machine"
    )
    parser.add_argument(
        "feed",
        type=_file_or_dash,
        metavar="FILE",
        help="the live output of stream: a file that grows, a named pipe, or - for the standard input",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="loopback address to bind to, written as numbers (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=_port, default=0, metavar="N", help="port to listen on (default: 0, a free port, printed)"
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="play the frames at the frame rate of the clip, from the first page that opens (a feed that is finished)",
    )
    parser.set_defaults(handler=_cmd_radar)


# --- merge-feeds --------------------------------------------------------------------


def _cmd_merge_feeds(args: argparse.Namespace) -> int:
    """Merge the live feeds of the detector and of the calibration by frame into the lines of a stream.

    The two feeds are files or named pipes that ``tools/run_tool.py --feed`` writes; the merged
    lines go to the standard output (for ``stream --lines -``), one flushed line at a time. The
    policy is that of ``feed.FeedMerger``. A report goes to the error stream. Exit code 0 when
    every frame went out with both parts, 1 when a part was missing, a line was dropped or the
    stream ended early, 2 when the feeds cannot be merged.
    """
    out = sys.stdout

    def write(text: str) -> None:
        out.write(text + "\n")
        out.flush()

    # A named pipe opens when its writer opens it. The descriptors are not closed here: a reading
    # thread may still wait in one when the reader of the merged lines stops early, and the end of
    # the process closes them (``feed.fd_lines``).
    feeds = {"detector": os.open(args.detector, os.O_RDONLY), "calibration": os.open(args.calibration, os.O_RDONLY)}
    def end() -> None:
        # The merged lines are whole: their reader sees their end now, not when a slow or hung tool ends.
        out.flush()
        if out is not sys.__stdout__:  # called in a process whose standard output was replaced (a test)
            return
        nothing = os.open(os.devnull, os.O_WRONLY)
        os.dup2(nothing, sys.stdout.fileno())
        os.close(nothing)

    try:
        merger = merge_feeds(
            {tool: fd_lines(fd) for tool, fd in feeds.items()}, write, wait_s=args.wait, queue_lines=args.queue, end=end
        )
    except FeedError as error:
        raise CliError(f"the feeds cannot be merged: {error}") from error
    except BrokenPipeError as error:
        # Nothing more can go to the standard output: point it at nothing, so that its flush at the end is quiet.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        raise CliError("the reader of the merged lines stopped before the end") from error
    sent = merger.next_frame
    print(f"merge-feeds: {sent} of {merger.n_frames} frames went out", file=sys.stderr)
    for name, values in merger.counts.as_dict().items():
        if any(values.values()):
            text = ", ".join(f"{tool} {count}" for tool, count in values.items())
            print(f"  {name.replace('_', ' ')}: {text}", file=sys.stderr)
    return EXIT_OK if sent == merger.n_frames and merger.counts.problems() == 0 else EXIT_FAILED


def _wait_seconds(text: str) -> float:
    """Read a wait in seconds: a finite number of 0 or more."""
    value = float(text)
    if not 0.0 <= value < float("inf"):
        raise argparse.ArgumentTypeError(f"{text} is not a number of 0 or more seconds")
    return value


def _behind_seconds(text: str) -> float:
    """Read the wait of ``stream --max-behind``: a finite number of seconds above 0."""
    value = float(text)
    if not 0.0 < value < float("inf"):
        raise argparse.ArgumentTypeError(f"{text} is not a number of seconds above 0")
    return value


def _add_merge_feeds(commands: Any) -> None:
    """Add the command ``merge-feeds`` to the sub-parsers object of argparse."""
    parser = commands.add_parser(
        "merge-feeds",
        help="merge the live feeds of the two tools by frame into the lines that stream --lines - reads",
    )
    parser.add_argument(
        "--detector", type=_path, required=True, metavar="FILE", help="feed of the detector (a file or a named pipe)"
    )
    parser.add_argument(
        "--calibration", type=_path, required=True, metavar="FILE", help="feed of the calibration (a file or a named pipe)"
    )
    parser.add_argument(
        "--wait",
        type=_wait_seconds,
        default=DEFAULT_WAIT_S,
        metavar="SECONDS",
        help=f"longest wait of a frame for the part of the other tool (default: {DEFAULT_WAIT_S:g})",
    )
    parser.add_argument(
        "--queue",
        type=_positive_int,
        default=DEFAULT_QUEUE_LINES,
        metavar="N",
        help=f"lines read ahead from each feed before its reading waits (default: {DEFAULT_QUEUE_LINES})",
    )
    parser.set_defaults(handler=_cmd_merge_feeds)


# --- doctor --------------------------------------------------------------------------


def print_checks(title: str, checks: Sequence[doctor.CheckResult]) -> None:
    """Print a group of check results: one line per check."""
    print(title)
    for check in checks:
        print(f"  [{check.status:<4}] {check.name}: {check.detail}")


def contract_limits(gates: Mapping[str, Any]) -> doctor.ContractLimits:
    """Read the thresholds of the contract checks (gate G0) from the gate file."""
    return doctor.ContractLimits(
        box_mean_iou_min=gate_number(gates, G0_SECTION, "box_mean_iou_min"),
        point_error_max_fraction=gate_number(gates, G0_SECTION, "point_error_max_fraction"),
        usage_sum_tolerance=gate_number(gates, G0_SECTION, "usage_sum_tolerance"),
        flex_valid_answer_rate_min=gate_number(gates, G0_SECTION, "flex_valid_answer_rate_min"),
    )


def contract_stage(
    models: Sequence[StepSettings],
    gates: Mapping[str, Any],
    cache_dir: Path,
    *,
    spend_limit_usd: float,
    over_budget: bool = False,
    video: bool = False,
) -> tuple[dict[str, doctor.ContractResult], dict[str, Any], BudgetExceeded | None]:
    """Run the contract checks of gate G0 for every model with the real backend.

    ``models`` are the step settings of the models: provider, model, coordinate
    convention and thinking level. The result is keyed by ``model_label``.

    The cost estimate is printed before the first call and the measured cost after the
    last one. The result holds the checks of each model, the cost record, and the
    budget stop when the budget guard ended the checks early. The models that were
    finished before a budget stop keep their checks, and the cost record holds the
    measured tokens of every paid answer, so the caller can still write a report. The
    answers are not cached: every run of the checks calls the model. ``over_budget``
    lets the checks start although their estimate does not fit under the hard stop.

    The checks stop above ``spend_limit_usd``, the spend limit of gate G0, and not above
    1.3 times their estimate like every other paid run. Their estimate is the least
    certain one: the first real answers of a model come from these checks. The checks do
    not start when their estimate is above the spend limit (``start_meter``).

    With ``video``, each model also gets the video check (``doctor.check_video``). Its
    video is written to a temporary folder that is removed at the end, and its request
    is part of the estimate. A model without a video rate gets no video request.
    """
    with tempfile.TemporaryDirectory(prefix="pitchvlm-video-check-") as folder:
        return _contract_stage(
            models,
            gates,
            cache_dir,
            spend_limit_usd=spend_limit_usd,
            over_budget=over_budget,
            video_folder=Path(folder) if video else None,
        )


def _contract_stage(
    models: Sequence[StepSettings],
    gates: Mapping[str, Any],
    cache_dir: Path,
    *,
    spend_limit_usd: float,
    over_budget: bool,
    video_folder: Path | None,
) -> tuple[dict[str, doctor.ContractResult], dict[str, Any], BudgetExceeded | None]:
    """Run the contract checks of ``contract_stage``; with ``video_folder``, the video check too."""
    budget = budget_limits(gates)
    limits = contract_limits(gates)
    scene = doctor.contract_scene()
    plan = [calls for settings in models for calls in doctor.contract_plan(settings)]
    openrouter = any(settings.provider != "gemini" for settings in models)
    video_rates = openrouter_video_rates() if video_folder is not None and openrouter else ()
    video_requests: dict[str, VLMRequest | None] = {}
    if video_folder is not None:
        for settings in models:
            label = model_label(settings.provider, settings.model)
            video_requests[label] = doctor.video_contract_request(settings, scene, video_folder, video_rates)
            if video_requests[label] is None:
                print(f"video check of {label}: the model has no video rate, so no video request is planned")
    planned = [request for request in video_requests.values() if request is not None]
    estimate = estimated_usd(scene.info.clip_id, plan) + estimated_requests_usd(planned)
    print_estimate(estimate, plan, planned)
    meter = start_meter(
        cache_dir,
        budget,
        new_run_id("doctor", "g0"),
        estimate,
        spend_limit_usd=spend_limit_usd,
        over_budget=over_budget,
    )
    backend, recorder = recording_backend(budget.max_usd_per_10_min)
    metered = MeteredBackend(backend, meter)
    results: dict[str, doctor.ContractResult] = {}
    stopped: BudgetExceeded | None = None
    prices = openrouter_prices() if any(settings.provider != "gemini" for settings in models) else ()
    try:
        for settings in models:
            label = model_label(settings.provider, settings.model)
            results[label] = doctor.contract_checks(
                metered,
                scene,
                settings,
                limits,
                recorder=recorder,
                openrouter_prices=prices,
                video_request=video_requests.get(label),
                check_video_input=video_folder is not None,
                openrouter_video_rates=video_rates,
            )
            print_checks(
                f"contract checks of gate G0, model {label} (coordinates {settings.coords})", results[label].checks
            )
    except BudgetExceeded as error:
        stopped = error
    finally:
        cost = {"replay": False, "estimate_usd": estimate, **meter.report()}
        print_measured(cost)
    return results, cost, stopped


def doctor_models(args: argparse.Namespace) -> list[StepSettings]:
    """Return the step settings of the models of ``doctor``: ``--models`` with ``--coords`` and ``--thinking-level``.

    A Gemini model is checked in the Gemini convention, whatever ``--coords`` says, so a
    run can check a Gemini model next to an OpenRouter model. Settings that a provider
    cannot take, for example a free OpenRouter model, raise ``CliError``.
    """
    models: list[StepSettings] = []
    for text in args.models:
        provider, model = model_ref(text)
        coords = args.coords if provider == "openrouter" else GEMINI_CONVENTION
        try:
            models.append(
                StepSettings(provider=provider, model=model, coords=coords, thinking_level=args.thinking_level)
            )
        except pydantic.ValidationError as error:
            reasons = "; ".join(str(problem["msg"]) for problem in error.errors())
            raise CliError(f"--models {text}: {reasons}") from error
    return models


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Check the machine and, unless ``--no-api`` is given, the contract of the model API (gate G0)."""
    call_api = not args.no_api
    if args.out is not None and not call_api:
        raise CliError("--out holds the report of the contract checks, so it cannot be used with --no-api")
    if args.video and not call_api:
        raise CliError("--video sends a video request, so it cannot be used with --no-api")
    models = doctor_models(args)
    providers = [settings.provider for settings in models] if call_api else list(PROVIDER_KEY_VARIABLES)
    environment = doctor.environment_checks(
        repository_root(),
        key_variables=[PROVIDER_KEY_VARIABLES[provider] for provider in dict.fromkeys(providers)],
        data_dir=DATA_DIR.expanduser(),
        api=call_api,
        needs_sdk="gemini" in providers,
    )
    print_checks("environment", environment)
    if not call_api:
        return EXIT_FAILED if doctor.any_failed(environment) else EXIT_OK
    if doctor.any_failed(environment):
        print("The contract checks did not run. Repair the failed checks first.")
        return EXIT_FAILED
    state = require_clean_tree(args.allow_dirty)
    print(f"code: commit {state.commit}, work tree {'dirty' if state.dirty else 'clean'}")
    gates = load_gates()
    spend_limit = gate_number(gates, G0_SECTION, "spend_usd_max")
    if not (math.isfinite(spend_limit) and spend_limit > 0):
        raise CliError(
            f"{GATES_FILE.as_posix()}: {G0_SECTION}.spend_usd_max must be a number above 0, found {spend_limit!r}"
        )
    contract, cost, stopped = contract_stage(
        models,
        gates,
        _cache_dir(args),
        spend_limit_usd=spend_limit,
        over_budget=args.start_over_budget,
        video=args.video,
    )
    spend = doctor.check_spend(cost["measured_usd"], spend_limit)
    print_checks("cost", [spend])
    checks = [*environment, *(check for result in contract.values() for check in result.checks), spend]
    passed = stopped is None and not doctor.any_failed(checks)
    if args.out is not None:
        report = {
            "gate": G0_SECTION,
            "passed": passed,
            "stopped_by_budget": None if stopped is None else str(stopped),
            "environment": [check.as_dict() for check in environment],
            "contract": {
                model: {"checks": [check.as_dict() for check in result.checks], "answers": list(result.answers)}
                for model, result in contract.items()
            },
            "spend": spend.as_dict(),
        }
        provenance = {
            "command": "doctor",
            "commit": state.commit,
            "dirty": state.dirty,
            "sdk_version": installed_sdk_version(),
            "prompts": {
                step: PROMPT_NAMES[step] for step in ("people", "landmarks", *(("people_video",) if args.video else ()))
            },
            "video": args.video,
            "models": list(args.models),
            "coords": args.coords,
            "thinking_level": args.thinking_level,
            "cost": cost,
        }
        write_metrics(args.out, report, provenance=provenance)
        print(f"report: {args.out}")
    if stopped is not None:
        # The report is written first: the paid answers give the measured tokens for a better estimate.
        raise stopped
    return EXIT_OK if passed else EXIT_FAILED


# --- data ----------------------------------------------------------------------------


def valid_only_manifest(valid_clips: Sequence[str], seed: int) -> dict[str, Any]:
    """Make the split list without its test part, for the time before the test split is downloaded.

    The groups of the validation clips do not depend on the test clips: each list is
    shuffled with its own generator. So stand-in names for the test clips give the same
    validation groups that the complete list will have.
    """
    stand_ins = [f"test-clip-not-downloaded-{index}" for index in range(TEST_FINAL_SIZE)]
    manifest = make_split_manifest(valid_clips, stand_ins, seed)
    del manifest["test"]
    return manifest


def check_split_list_unchanged(path: Path, manifest: Mapping[str, Any]) -> None:
    """Raise ``CliError`` when a split list file is there and the new list would change or remove a part of it.

    The split list is fixed before the first paid run. A new list may only add a part,
    for example the test clips.
    """
    if not path.is_file():
        return
    old = load_manifest(path)
    changed = sorted(key for key in old if old[key] != manifest.get(key))
    if changed:
        raise CliError(
            f"the split list {path} is already there, and the new list would change its parts {changed}. "
            "The split list is fixed before the first paid run. Use the same seed and the same clips."
        )


def write_split_list(root: Path, path: Path, seed: int) -> None:
    """Write the split list from the clips that are extracted below the data root.

    With both splits the list is complete. Without the test split the list holds the
    validation part only, and a printed line says so.
    """
    valid_dir, test_dir = split_dir(root, "valid"), split_dir(root, "test")
    if not valid_dir.is_dir():
        raise CliError(f"the split list needs the validation clips, but {valid_dir} does not exist")
    valid_clips = list_clips(valid_dir)
    try:
        if test_dir.is_dir():
            manifest = make_split_manifest(valid_clips, list_clips(test_dir), seed)
        else:
            manifest = valid_only_manifest(valid_clips, seed)
    except ValueError as error:
        raise CliError(f"the split list cannot be made: {error}") from error
    check_split_list_unchanged(path, manifest)
    save_manifest(path, manifest)
    sizes = [
        f"{group} {len(clips)}" for split in ("valid", "test") for group, clips in manifest.get(split, {}).items()
    ]
    groups = ", ".join(sizes)
    print(f"split list: {path} (seed {seed}; clips per group: {groups})")
    if "test" not in manifest:
        print(
            "The test split is not downloaded, so the split list holds the validation part only. "
            "Run the data command for the test split with the same --manifest and --seed to add the test clips."
        )


def _cmd_data(args: argparse.Namespace) -> int:
    """Download the archive of one split, extract its clips and, when asked, write the split list."""
    if (args.manifest is None) != (args.seed is None):
        raise CliError("--manifest and --seed go together: give both, or none of them")
    archive = download_split(args.root, args.split)
    print(f"archive: {archive}")
    folder = split_dir(args.root, args.split)
    clips = extract_clips(archive, folder, args.clips)
    print(f"extracted {len(clips)} clips of the split {args.split} into {folder}")
    if args.manifest is not None:
        write_split_list(args.root, args.manifest, args.seed)
    return EXIT_OK


# --- oracle --------------------------------------------------------------------------


def oracle_variants(labels: GsrLabels, interval_s: float) -> dict[str, list[FrameState]]:
    """Build the four prediction variants of the label-only stage from the labels of one clip.

    The variants are: the ground truth, the ground truth shifted by 1 m, the ground
    truth at the keyframes only, and the ground truth resampled from the keyframes
    (kept at the keyframes, interpolated between them). The keyframes are those of the
    pipeline for the interval ``interval_s``.
    """
    truth = gsr_frame_states(labels)
    keyframes = keyframe_indices(len(labels.frames), labels.fps, interval_s)
    return {
        "ground_truth": truth,
        "shifted_1m": shift_states(truth, ORACLE_SHIFT_M),
        "keyframes_only": keep_frames(truth, keyframes),
        "resampled": resample_states(truth, keyframes),
    }


def side_rule_facts(truth: Sequence[FrameState]) -> TeamSideFacts | None:
    """Say how the team-side rule decides a clip and whether it gives every player track its labelled side.

    The result is None when no player of the clip has a labelled side.
    """
    if not team_side_by_rule(truth):
        return None
    return team_side_facts(truth)


def side_rule_report(facts: Mapping[str, TeamSideFacts | None]) -> dict[str, Any]:
    """Build the part of the report of gate G1 about the team-side rule, from the facts of every clip.

    ``share_right`` is the share of the clips with a labelled side where the rule of the
    pipeline is right. It is nan when no clip has a labelled side. The other values show
    what the two limits of the rule are compared with: the share of the clips where the
    players alone are right, the clips that the goalkeepers decided, and for every clip
    the gap between the player means and the goalkeepers in a goal zone.
    """
    judged = {clip: found for clip, found in facts.items() if found is not None}
    n_judged = len(judged)
    return {
        "share_right": sum(found.agrees for found in judged.values()) / n_judged if judged else math.nan,
        "share_right_by_players_alone": (
            sum(found.players_alone_agree for found in judged.values()) / n_judged if judged else math.nan
        ),
        "clips_right": sorted(clip for clip, found in judged.items() if found.agrees),
        "clips_wrong": sorted(clip for clip, found in judged.items() if not found.agrees),
        "clips_without_labelled_side": sorted(clip for clip, found in facts.items() if found is None),
        "clips_decided_by_goalkeepers": sorted(
            clip for clip, found in judged.items() if found.decided_by == "goalkeepers"
        ),
        "facts": {clip: dataclasses.asdict(found) for clip, found in sorted(judged.items())},
    }


@dataclass(frozen=True)
class OracleLimits:
    """The thresholds of gate G1, from the gate file. GS-HOTA values are in percent."""

    ground_truth_min: float
    shifted_1m: tuple[float, float]
    keyframes_only: tuple[float, float]
    resampled_min: float
    #: Smallest share of clips where the team-side rule must be right.
    side_rule_share_min: float
    #: Largest median residual of the label homographies, metres.
    residual_median_max_m: float


def oracle_limits(gates: Mapping[str, Any]) -> OracleLimits:
    """Read the thresholds of gate G1 from the gate file."""
    return OracleLimits(
        ground_truth_min=gate_number(gates, G1_SECTION, "ground_truth_gs_hota_min"),
        shifted_1m=gate_range(gates, G1_SECTION, "shifted_1m_gs_hota"),
        keyframes_only=gate_range(gates, G1_SECTION, "keyframes_only_gs_hota"),
        resampled_min=gate_number(gates, G1_SECTION, "resampled_2fps_gs_hota_min"),
        side_rule_share_min=gate_number(gates, G1_SECTION, "team_side_rule_correct_clips_min"),
        residual_median_max_m=gate_number(gates, G1_SECTION, "label_homography_residual_median_m_max"),
    )


def oracle_gate_checks(
    limits: OracleLimits,
    scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    side_share: float,
    residual_m: float,
) -> list[GateCheck]:
    """Compare the values of the label-only stage with the thresholds of gate G1.

    ``scores`` maps a variant to the scorer result of its tracker. The GS-HOTA of all
    clips together is compared. ``side_share`` is the share of clips where the team-side
    rule is right, and ``residual_m`` is the median residual of the label homographies.
    """
    hota = {variant: scores[variant][COMBINED]["HOTA"] for variant in ORACLE_TRACKERS}
    return [
        at_least("GS-HOTA of the ground truth", hota["ground_truth"], limits.ground_truth_min),
        within("GS-HOTA of the ground truth shifted by 1 m", hota["shifted_1m"], *limits.shifted_1m),
        within("GS-HOTA of the keyframes only", hota["keyframes_only"], *limits.keyframes_only),
        at_least("GS-HOTA of the ground truth resampled from the keyframes", hota["resampled"], limits.resampled_min),
        at_least("share of clips where the team-side rule is right", side_share, limits.side_rule_share_min),
        at_most("median residual of the label homographies in metres", residual_m, limits.residual_median_max_m),
    ]


def _cmd_oracle(args: argparse.Namespace) -> int:
    """Run the free label-only stage of gate G1 for the clips of one split.

    For each clip the four prediction variants are written with the exporter. Then the
    scorer runs once per variant. The team-side rule and the label homographies are
    checked on the labels alone. No model is called. The report names the commit of the
    clone of the scorer repository that holds the scorer script.
    """
    clips = clips_of_split(args.root, args.split, args.clips)
    limits = oracle_limits(load_gates())
    interval_s = PipelineConfig().keyframe_interval_s
    pred_root = args.out / PREDICTIONS_FOLDER
    format_problems: dict[str, list[str]] = {}
    side_facts: dict[str, TeamSideFacts | None] = {}
    residuals: list[np.ndarray] = []
    for clip in clips:
        labels = read_labels(split_dir(args.root, args.split) / clip / LABEL_FILE_NAME)
        variants = oracle_variants(labels, interval_s)
        for variant, states in variants.items():
            path = write_clip_predictions(
                states, labels, pred_root, split=args.split, tracker=ORACLE_TRACKERS[variant], clip_id=clip
            )
            if variant == "ground_truth":
                format_problems[clip] = prediction_problems(path, labels)
        side_facts[clip] = side_rule_facts(variants["ground_truth"])
        residuals.append(label_homography_residuals_m(labels))
        print(f"clip {clip}: {len(labels.frames)} frames, prediction files of {len(variants)} variants written")
    scores = {
        variant: run_trackeval(
            args.scorer_python,
            args.scorer_script,
            args.root,
            pred_root,
            split=args.split,
            tracker=tracker,
            clips=clips,
            output_dir=args.out / SCORER_FOLDER / tracker,
            drop_env=KEY_VARIABLES,
        )
        for variant, tracker in ORACLE_TRACKERS.items()
    }
    side_rule = side_rule_report(side_facts)
    residual_stats = summarize(np.concatenate(residuals))
    checks = oracle_gate_checks(limits, scores, side_rule["share_right"], residual_stats["median"])
    passed = all(check.passed for check in checks)
    state = code_state(repository_root())
    report = {
        "gate": G1_SECTION,
        "passed": passed,
        "split": args.split,
        "clips": clips,
        "keyframe_interval_s": interval_s,
        "variants": {
            variant: {"tracker": tracker, "scores": scores[variant]} for variant, tracker in ORACLE_TRACKERS.items()
        },
        "team_side_rule": side_rule,
        "label_homography_residual_m": residual_stats,
        "format_problems_of_ground_truth": {clip: found for clip, found in format_problems.items() if found},
        "gates": [dataclasses.asdict(check) for check in checks],
    }
    write_metrics(
        args.out / ORACLE_NAME,
        report,
        provenance={
            "command": "oracle",
            "commit": state.commit,
            "dirty": state.dirty,
            "scorer_commit": scorer_commit(args.scorer_script),
        },
    )
    print(f"gate G1 (labels only): split {args.split}, {len(clips)} clips")
    for check in checks:
        print(f"  [{'ok' if check.passed else 'fail':<4}] {check.name}: {check.value:.3f} (gate: {check.rule})")
    print(
        f"  team-side rule: the players alone are right on a share of {side_rule['share_right_by_players_alone']:.3f} "
        f"of the clips; the goalkeepers decided {len(side_rule['clips_decided_by_goalkeepers'])} clips"
    )
    for clip, found in report["format_problems_of_ground_truth"].items():
        print(f"  format checker, ground truth of {clip}: {'; '.join(found)}")
    print(f"report: {args.out / ORACLE_NAME}")
    return EXIT_OK if passed else EXIT_FAILED


# --- probe ---------------------------------------------------------------------------


def default_step_settings(probe: ProbeKind) -> StepSettings:
    """Return the default settings of the pipeline step that a probe measures.

    The people probe measures the people step, the pitch probe the landmarks step and
    the number probe the number step, which reads its crops at a low resolution.
    """
    defaults = PipelineConfig()
    return {"people": defaults.people, "pitch": defaults.landmarks, "number": defaults.number}[probe]


class Experiment(Record):
    """One probe run, as an experiment file describes it."""

    name: str = Field(min_length=1)
    probe: ProbeKind
    #: Group of the split list whose clips are probed, for example "probe".
    group: str = Field(min_length=1)
    #: Frames per clip of a people probe or a pitch probe.
    frames_per_clip: int | None = Field(default=None, ge=1)
    #: Player tracks per clip of a number probe.
    tracks_per_clip: int | None = Field(default=None, ge=1)
    #: Seed of the frame choice and of the track choice.
    seed: int
    #: Repeat index of the requests. A repeat run uses 1, 2, ... so that the cache does not answer it.
    repeat: int = Field(ge=0)
    #: How the model step is called. A section that leaves a key out keeps the default
    #: of the step that the probe measures (``default_step_settings``).
    step: StepSettings
    #: With this section a people probe sends video in place of images (``probes.VideoProbe``).
    video: probes.VideoProbe | None = None

    @model_validator(mode="before")
    @classmethod
    def _lay_step_over_its_defaults(cls, data: Any) -> Any:
        """Lay a step section over the default settings of the step that the probe measures.

        So a number probe whose section names only a model keeps the low resolution of
        the number step. Only a mapping is laid over the defaults: a ``StepSettings``
        record is taken as it is. A section of an unknown probe, and a value that is not
        a mapping, go on as they are, and the validation refuses them.
        """
        if not isinstance(data, dict):
            return data
        probe, step = data.get("probe"), data.get("step")
        if not isinstance(step, dict):
            return data
        for kind in PROBE_KINDS:
            if probe == kind:
                return {**data, "step": {**default_step_settings(kind).model_dump(), **step}}
        return data

    @model_validator(mode="after")
    def _check_count(self) -> Experiment:
        if self.probe == "number":
            wanted, other = "tracks_per_clip", "frames_per_clip"
        else:
            wanted, other = "frames_per_clip", "tracks_per_clip"
        if getattr(self, wanted) is None or getattr(self, other) is not None:
            raise ValueError(f"a {self.probe} probe needs '{wanted}' and must not have '{other}'")
        return self

    @model_validator(mode="after")
    def _check_video(self) -> Experiment:
        """Allow video input in a people probe only, with video settings that fit its step."""
        if self.video is None:
            return self
        if self.probe != "people":
            raise ValueError(f"a {self.probe} probe sends images only: video input is built for the people probe")
        probes.check_video_settings(self.step, self.video)
        return self

    @property
    def per_clip(self) -> int:
        """Return the number of frames or tracks per clip, whichever the probe uses."""
        count = self.tracks_per_clip if self.probe == "number" else self.frames_per_clip
        if count is None:
            raise ValueError(f"the experiment {self.name!r} has no count per clip")
        return count


class ToolExperiment(Record):
    """One measurement of an external tool on the frames of a probe, as an experiment file describes it."""

    name: str = Field(min_length=1)
    #: Group of the split list whose clips are measured, for example "probe".
    group: str = Field(min_length=1)
    #: Frames per clip. With the group and the seed of a model probe, the frames are those of that probe.
    frames_per_clip: int = Field(ge=1)
    #: Seed of the frame choice.
    seed: int
    #: The tool and its settings.
    tool: ToolSettings

    @property
    def probe(self) -> ProbeKind:
        """Return the probe whose frames and values measure the tool."""
        return TOOL_PROBES[self.tool.name]


def _experiment_data(path: Path) -> dict[str, Any]:
    """Read the mapping of an experiment file (YAML). A file that holds no mapping raises ``CliError`` with the path."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as error:
        raise CliError(f"the experiment file {path} cannot be read: {error}") from error
    except (ValueError, yaml.YAMLError) as error:
        # A ValueError comes from bytes that are not UTF-8 text and from a value that YAML cannot build.
        raise wrong_file("experiment file", path, error) from error
    if not isinstance(data, dict):
        raise CliError(f"{path}: the top level of an experiment file must be a mapping")
    return data


def load_experiment(path: Path) -> Experiment:
    """Read the experiment file of a model probe. Wrong content raises ``CliError`` with the path.

    The section ``step`` is laid over the default settings of the step that the probe
    measures, so a key that it leaves out keeps that default.
    """
    try:
        return Experiment.model_validate(_experiment_data(path))
    except pydantic.ValidationError as error:
        raise wrong_file("experiment file", path, error) from error


def load_tool_experiment(path: Path) -> ToolExperiment:
    """Read the experiment file of an external tool. Wrong content raises ``CliError`` with the path."""
    try:
        return ToolExperiment.model_validate(_experiment_data(path))
    except pydantic.ValidationError as error:
        raise wrong_file("experiment file", path, error) from error


def group_clips(manifest: Mapping[str, Any], group: str) -> tuple[str, list[str]]:
    """Find a group of the split list. Return its split and its clip names, in the order of the list.

    ``CliError`` is raised for a split list without the group, for a group that is no list of
    clip names, and for a group that names a clip more than once.
    """
    known: list[str] = []
    for split in SPLITS:
        groups = manifest.get(split)
        if not isinstance(groups, dict):
            continue
        known.extend(str(name) for name in groups)
        if group not in groups:
            continue
        clips = groups[group]
        if not isinstance(clips, list) or not all(isinstance(clip, str) for clip in clips):
            raise CliError(f"the group {group!r} of the split list is not a list of clip names")
        twice = sorted(name for name, count in Counter(clips).items() if count > 1)
        if twice:
            raise CliError(
                f"the group {group!r} of the split list names these clips more than once: {', '.join(twice)}"
            )
        return split, list(clips)
    raise CliError(f"the split list has no group {group!r}; its groups are: {', '.join(known) or 'none'}")


@dataclass(frozen=True)
class GroupClips:
    """The clips of one group of the split list, as the options of a command name them."""

    manifest_path: Path
    #: Seed of the split list, or None when the list names none.
    manifest_seed: Any
    split: str
    #: Clip names, in the order of the split list.
    clips: list[str]


def clips_of_group(args: argparse.Namespace, group: str) -> GroupClips:
    """Find the clips of a group in the split list of a command (``--manifest``, ``--max-clips``).

    ``CliError`` is raised when the split list is missing, when it is no JSON object and
    when the group holds no clip.
    """
    manifest_path = args.manifest if args.manifest is not None else repository_root() / SPLIT_LIST_FILE
    if not manifest_path.is_file():
        raise CliError(f"the split list {manifest_path} does not exist: write it with the data command")
    try:
        manifest = load_manifest(manifest_path)
    except ValueError as error:
        raise wrong_file("split list", manifest_path, error) from error
    split, clip_names = group_clips(manifest, group)
    if args.max_clips is not None:
        clip_names = clip_names[: args.max_clips]
    if not clip_names:
        raise CliError(f"the group {group!r} of the split list holds no clip")
    return GroupClips(manifest_path=manifest_path, manifest_seed=manifest.get("seed"), split=split, clips=clip_names)


def print_main_metric(payload: Mapping[str, Any]) -> None:
    """Print the main metric of a measurement: its mean over the clips and the interval of the mean.

    The line names the clips that have a value and all clips of the measurement. A clip
    without a value is left out of the mean: for a pitch measurement that is a clip
    without any mapping.
    """
    metric = payload["main_metric"]
    print(
        f"main metric {metric['name']}: mean {metric['mean']:.3f} over {metric['n_clips']} of "
        f"{payload['n_clips']} clips "
        f"({metric['confidence']:.0%} interval from {metric['low']:.3f} to {metric['high']:.3f})"
    )


def probe_plan(experiment: Experiment, n_clips: int) -> list[PlannedCalls]:
    """List the largest number of model calls of a probe: (step, settings, number of requests).

    A people probe and a number probe ask the kit legend one time per clip.
    """
    per_step = n_clips * experiment.per_clip
    if experiment.video is not None:
        # The video requests are built before the run and estimated from their videos (``_cmd_probe``).
        return []
    if experiment.probe == "people":
        return [("kit_legend", experiment.step, n_clips), ("people", experiment.step, per_step)]
    if experiment.probe == "pitch":
        return [("landmarks", experiment.step, per_step)]
    return [("kit_legend", experiment.step, n_clips), ("number", experiment.step, per_step)]


def run_probe(
    experiment: Experiment,
    clips: Sequence[LabelledClip],
    backend: Backend,
    gates: Mapping[str, Any],
    video_folder: Path | None = None,
) -> dict[str, Any]:
    """Run the probe of an experiment and return its measured values.

    The settings that the experiment file does not hold are fixed elsewhere: the height
    of a small box is the one of gate G2, and the RANSAC threshold, the crops per track
    and the enlargement of a crop are the defaults of the pipeline. ``video_folder``
    holds the videos of a people probe with video input.
    """
    pipeline = PipelineConfig()
    if experiment.probe == "people":
        return probes.probe_people(
            clips,
            backend,
            experiment.step,
            frames_per_clip=experiment.per_clip,
            seed=experiment.seed,
            small_box_max_height_px=gate_number(gates, G2_SECTION, "small_box_max_height_px"),
            repeat=experiment.repeat,
            video=experiment.video,
            video_folder=video_folder,
        )
    if experiment.probe == "pitch":
        return probes.probe_pitch(
            clips,
            backend,
            experiment.step,
            frames_per_clip=experiment.per_clip,
            seed=experiment.seed,
            ransac_px=pipeline.ransac_px,
            repeat=experiment.repeat,
        )
    return probes.probe_number(
        clips,
        backend,
        experiment.step,
        tracks_per_clip=experiment.per_clip,
        crops_per_track=pipeline.number_crops,
        crop_upscale=pipeline.crop_upscale,
        seed=experiment.seed,
        repeat=experiment.repeat,
    )


def _cmd_probe(args: argparse.Namespace) -> int:
    """Run the probe that an experiment file describes, and write its metrics file."""
    experiment = load_experiment(args.experiment)
    if experiment.probe != args.kind:
        raise CliError(
            f"the experiment file describes a {experiment.probe} probe, but the command asks for {args.kind}"
        )
    state = require_clean_tree(args.allow_dirty)
    group = clips_of_group(args, experiment.group)
    clips = [labelled_clip(args.root, group.split, clip) for clip in group.clips]
    print(
        f"{experiment.probe} probe {experiment.name}: {len(clips)} clips of the group {experiment.group} "
        f"({group.split})"
    )
    video_folder: Path | None = None
    planned: list[VLMRequest] = []
    if experiment.video is not None:
        video_folder = args.video_dir.expanduser()
        refuse_video_folder(video_folder, _cache_dir(args))
        planned = probes.plan_people_video(
            clips,
            experiment.step,
            frames_per_clip=experiment.per_clip,
            seed=experiment.seed,
            video=experiment.video,
            folder=video_folder,
            repeat=experiment.repeat,
        )
        print(f"video folder: {video_folder} ({len(planned)} videos of the requests, outside the repository)")
    session = open_session(
        "probe",
        experiment.name,
        probe_plan(experiment, len(clips)),
        cache_dir=_cache_dir(args),
        replay=args.replay,
        repeat=experiment.repeat,
        over_budget=args.start_over_budget,
        allow_other_images=args.allow_other_images,
        planned_requests=planned,
    )
    try:
        payload = run_probe(experiment, clips, session.backend, load_gates(), video_folder)
    finally:
        cost = close_session(session)
    provenance = {
        "command": "probe",
        "commit": state.commit,
        "dirty": state.dirty,
        "sdk_version": installed_sdk_version(),
        "experiment": experiment.name,
        "experiment_file": args.experiment.name,
        "split_list": group.manifest_path.name,
        "split_list_seed": group.manifest_seed,
        "group": experiment.group,
        "split": group.split,
        "max_clips": args.max_clips,
        "clips": group.clips,
        "cost": cost,
    }
    write_metrics(args.out, payload, provenance=provenance)
    print_main_metric(payload)
    print(f"metrics file: {args.out}")
    return EXIT_OK


# --- tool-job and tool-score ---------------------------------------------------------


def _cmd_tool_job(args: argparse.Namespace) -> int:
    """Write the job file of a tool experiment: the frames that the tool must read, and its settings.

    The frames are those of the people probe and the pitch probe for the same group,
    frames per clip and seed (``probes.probe_frames``). The job file names each image by
    its path below the data folder. The script ``tools/run_tool.py`` reads the job file.
    """
    refuse_input_as_output(args.out, {"--experiment": args.experiment})
    experiment = load_tool_experiment(args.experiment)
    group = clips_of_group(args, experiment.group)
    refuse_input_as_output(args.out, {"--manifest": group.manifest_path})
    sources: dict[str, ImageFolderSource] = {}
    clips: list[LabelledClip] = []
    for clip in group.clips:
        source, labels = dataset_clip(args.root, group.split, clip)
        sources[clip] = source
        clips.append(matched_clip(source, labels, split_dir(args.root, group.split) / clip))
    chosen = frames_of_probe(clips, experiment)
    frames = tuple(
        JobFrame(clip=clip, frame=frame, image=sources[clip].path(frame).relative_to(args.root).as_posix())
        for clip, clip_frames in chosen.items()
        for frame in clip_frames
    )
    job = ToolJob(format=JOB_FORMAT, name=experiment.name, tool=experiment.tool, frames=frames)
    write_json(args.out, job.model_dump(mode="json"))
    print(
        f"tool job {experiment.name}: {len(frames)} frames of {len(clips)} clips of the group {experiment.group} "
        f"({group.split}) for the tool {experiment.tool.name}"
    )
    print(f"job file: {args.out}")
    return EXIT_OK


def _cmd_clip_job(args: argparse.Namespace) -> int:
    """Write the clip job of one tool for every frame of one clip.

    The tool settings come from the settings file of the run (``detector`` or
    ``calibration``), so that ``run`` with the same file accepts the clip run. The clip is a
    folder of frame images below the data folder; the job names each image by its path below
    that folder. A video file is refused: its frames must be written as images first, with
    ``video-frames``. The clip folder that command writes holds the record of the video
    (``frames.read_video_record``); the job then names the video by its SHA-256, so that
    ``run`` reads the clip run only with that video. A record that does not fit the images
    of the folder (another number of frames or another size) is refused.
    """
    if args.config is not None:
        refuse_input_as_output(args.out, {"--config": args.config})
    config = PipelineConfig() if args.config is None else read_settings(args.config)
    settings: ToolSettings = config.detector if args.tool in DETECTOR_FAMILIES else config.calibration
    if settings.name != args.tool:
        raise CliError(
            f"the settings name the detector {settings.name}, not {args.tool}: give --tool {settings.name}, "
            f"or a settings file whose section detector names {args.tool}"
        )
    if not args.clip.is_dir():
        raise CliError(
            f"{args.clip} is not a folder of frame images: a clip job names image files, so write the frames "
            "of a video as images below the data folder first, with the command video-frames"
        )
    folder = frame_folder(args.clip)
    source = ImageFolderSource(folder, clip_id_of_folder(folder))
    video_sha256 = video_of_folder(folder, source.info)
    root = args.root.resolve()
    try:
        images = tuple(source.path(frame).relative_to(root).as_posix() for frame in range(source.info.n_frames))
    except ValueError as error:
        raise CliError(
            f"the clip folder {folder} does not lie below the data folder {root}: the job names each image by its "
            "path below the data folder"
        ) from error
    job = ClipJob(
        format=CLIP_JOB_FORMAT, clip=source.info.clip_id, tool=settings, images=images, video_sha256=video_sha256
    )
    write_json(args.out, job.model_dump(mode="json"))
    print(f"clip job of the tool {args.tool}: {len(images)} frames of the clip {source.info.clip_id}")
    if video_sha256 is not None:
        print(f"the frames of the video with the SHA-256 {video_sha256}: run reads the clip run only with that video")
    print(f"job file: {args.out}")
    return EXIT_OK


def video_of_folder(folder: Path, info: ClipInfo) -> str | None:
    """Return the SHA-256 of the video whose frames a folder of frame images holds, or None for images of no video.

    The record of the video lies in the clip folder (``frames.read_video_record``): the
    folder above the frame folder ``img1``, else the folder itself. A record that does not
    follow its format, or that names another number of frames or another image size than
    the folder holds, raises ``CliError``.
    """
    clip_folder = folder.parent if folder.name == FRAME_FOLDER else folder
    try:
        record = read_video_record(clip_folder)
    except ValueError as error:
        raise wrong_file("video record", clip_folder, error) from error
    if record is None:
        return None
    if (record.n_frames, record.width, record.height) != (info.n_frames, info.width, info.height):
        raise CliError(
            f"the record of the video {record.video} in {clip_folder} names {record.n_frames} frames of "
            f"{record.width} x {record.height} pixels, but the folder holds {info.n_frames} images of "
            f"{info.width} x {info.height} pixels: write the frames of the video again into a new folder"
        )
    return record.video_sha256


def refuse_input_as_output(out: Path, inputs: Mapping[str, Path]) -> None:
    """Raise ``CliError`` when the output file of a command is one of its input files.

    ``inputs`` maps the option of each input file to its path. The command would replace
    its input: for ``tool-score`` the answers of the tool would be gone. Two paths name the
    same file when they resolve to the same path, and when both exist as one file (another
    letter case on a disk that does not tell the cases apart, or a hard link). The output path
    is resolved first: a path through a folder that is not there yet is judged as the file
    that the command would write.
    """
    written = out.resolve()
    for option, path in inputs.items():
        if written == path.resolve() or (written.exists() and path.exists() and written.samefile(path)):
            raise CliError(f"--out names the same file as {option} ({out}): the command would replace its input")


def frames_of_probe(clips: Sequence[LabelledClip], experiment: ToolExperiment) -> dict[str, list[int]]:
    """Return the frames of the probe of a tool experiment, by clip id (``probes.probe_frames``).

    The clips come from ``clips_of_group`` and ``matched_clip``: each one has its own name
    and at least one scored frame, so the frames can be chosen.
    """
    return probes.probe_frames(clips, frames_per_clip=experiment.frames_per_clip, seed=experiment.seed)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of a file as hexadecimal text. The file is read in parts, so a large video fits."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_axis_check(check: Mapping[str, float]) -> int:
    """Print the axis check of a pitch measurement and return the exit code that it gives.

    The check passes when the matrices as they are give the smallest median error of the
    four sign changes (``probes.AXIS_SIGNS``). Then the result is ``EXIT_OK``, and the size
    of the value is for the reader: it must be small, some metres at most. When another
    sign change gives a smaller error, a line names the three things that this can mean, and
    the result is ``EXIT_FAILED``. Without any measured position error nothing can be compared: ``EXIT_OK``.
    That is the case when no frame has a mapping, and when no frame with a mapping has a
    labelled person with a pitch position.
    """
    print("axis check, median error in m: " + ", ".join(f"{name} {value:.3f}" for name, value in check.items()))
    finite = {name: value for name, value in check.items() if math.isfinite(value)}
    if not finite:
        print("axis check: no frame has a mapping with a measured position error, so nothing can be compared")
        return EXIT_OK
    smallest = min(finite, key=finite.__getitem__)
    if finite.get("as_given") == finite[smallest]:
        return EXIT_OK
    print(
        f"axis check failed: {smallest} gives a smaller error than as_given. That value small and "
        "plausible_share 0: the matrices of the run file are in another pitch convention, and tools/run_tool.py "
        "has a fault. That value small and plausible_share above 0: the tool mirrors the pitch on a part of the "
        "frames. All four values large: the cameras are wrong on most frames. Read docs/RUNBOOK.md, section 3, "
        "before any other value of the metrics file"
    )
    return EXIT_FAILED


def read_tool_run(path: Path) -> ToolRun:
    """Read the run file of an external tool. Wrong content raises ``CliError`` with the path."""
    try:
        return load_tool_run(path)
    except OSError as error:
        raise CliError(f"the tool run file {path} cannot be read: {error}") from error
    except ValueError as error:
        raise wrong_file("tool run file", path, error) from error


def probe_answers(
    run: ToolRun, experiment: ToolExperiment, frames: Mapping[str, Sequence[int]], path: Path
) -> dict[tuple[str, int], RunFrame]:
    """Return the answers of a run file for the frames of a probe, by (clip id, frame).

    ``CliError`` is raised when the run file was made for another experiment or with
    other tool settings, and when it has no answer for a frame of the probe. So the
    settings of the experiment file are those that the tool ran with.
    """
    if run.name != experiment.name:
        raise CliError(
            f"the tool run file {path} was made for the experiment {run.name!r}, not for {experiment.name!r}"
        )
    if run.tool != experiment.tool:
        raise CliError(
            f"the tool run file {path} was made with other tool settings than the experiment file: "
            f"{run.tool.model_dump(mode='json')} against {experiment.tool.model_dump(mode='json')}. "
            "Write the job file again and run the tool again"
        )
    answers = run.answers()
    wanted = [(clip, frame) for clip, clip_frames in frames.items() for frame in clip_frames]
    missing = [key for key in wanted if key not in answers]
    if missing:
        examples = ", ".join(f"{clip} frame {frame}" for clip, frame in missing[:5])
        raise CliError(
            f"the tool run file {path} has no answer for {len(missing)} of the {len(wanted)} frames of the probe, "
            f"for example: {examples}. Write the job file with the same options and run the tool again"
        )
    return {key: answers[key] for key in wanted}


def score_tool(
    experiment: ToolExperiment,
    clips: Sequence[LabelledClip],
    answers: Mapping[tuple[str, int], RunFrame],
    gates: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    """Measure the answers of an external tool with the values of its probe.

    The detector is measured like the people step, with the height of a small box of gate
    G2. The calibration tool is measured like the pitch mapping. A payload that does not
    follow the format of its adapter raises ``CliError`` with the run file, the clip and
    the frame.
    """
    tool = experiment.tool
    settings = tool.model_dump(mode="json")
    if is_detector(tool):
        return probes.score_people_answers(
            clips,
            _adapted(partial(people_from_detections, min_score=tool.min_score), answers, path),
            settings,
            frames_per_clip=experiment.frames_per_clip,
            seed=experiment.seed,
            small_box_max_height_px=gate_number(gates, G2_SECTION, "small_box_max_height_px"),
        )
    return probes.score_pitch_mappings(
        clips,
        _adapted(homography_from_payload, answers, path),
        settings,
        frames_per_clip=experiment.frames_per_clip,
        seed=experiment.seed,
    )


def _adapted(
    adapter: Callable[..., _Adapted], answers: Mapping[tuple[str, int], RunFrame], path: Path
) -> Callable[[str, int], _Adapted]:
    """Return the function that gives the answer of a run file for one frame as a record of this package.

    ``adapter`` turns the payload of one frame into the record. Its ``ValueError`` for a
    payload that does not follow the format becomes a ``CliError`` that names the run file,
    the clip and the frame.
    """

    def answer_of(clip_id: str, frame: int) -> _Adapted:
        try:
            return adapter(answers[clip_id, frame].payload, frame=frame)
        except ValueError as error:
            raise CliError(
                f"the tool run file {path} cannot be used: clip {clip_id}, frame {frame}: {error}"
            ) from error

    return answer_of


def _cmd_tool_score(args: argparse.Namespace) -> int:
    """Measure the answers of an external tool on the frames of its probe, and write the metrics file.

    No model is called and nothing is paid. The metrics file holds the values of the probe
    (``probes.score_people_answers`` or ``probes.score_pitch_mappings``) and
    ``seconds_per_frame``: the time that the tool needed for each measured frame, as the
    run file reports it. Its provenance names the run file with its SHA-256, and it says
    whether the script that made the run file is the ``tools/run_tool.py`` of this work
    tree, so that the commit of the code also names the script.

    For a pitch measurement the command prints the axis check (``print_axis_check``). When
    the check fails, the metrics file is written and the exit code is 1.
    """
    refuse_input_as_output(args.out, {"--experiment": args.experiment, "--run": args.run})
    experiment = load_tool_experiment(args.experiment)
    state = require_clean_tree(args.allow_dirty)
    group = clips_of_group(args, experiment.group)
    refuse_input_as_output(args.out, {"--manifest": group.manifest_path})
    clips = [labelled_clip(args.root, group.split, clip) for clip in group.clips]
    run = read_tool_run(args.run)
    frames = frames_of_probe(clips, experiment)
    answers = probe_answers(run, experiment, frames, args.run)
    print(
        f"{experiment.probe} probe {experiment.name} of the tool {experiment.tool.name}: {len(answers)} frames of "
        f"{len(clips)} clips of the group {experiment.group} ({group.split})"
    )
    n_not_used = len(run.frames) - len(answers)
    if n_not_used:
        print(f"{n_not_used} answers of the run file are for other frames; they are not measured")
    measured = score_tool(experiment, clips, answers, load_gates(), args.run)
    seconds = np.array([answer.seconds for answer in answers.values()], dtype=np.float64)
    payload = {**measured, "seconds_per_frame": summarize(seconds)}
    script_is_current = run.runner.get("script_sha256") == file_sha256(repository_root() / TOOL_SCRIPT)
    if not script_is_current:
        print(
            f"the run file was not made by the {TOOL_SCRIPT.as_posix()} of this work tree: "
            "the commit of the metrics file does not name the script that ran the tool"
        )
    provenance = {
        "command": "tool-score",
        "commit": state.commit,
        "dirty": state.dirty,
        "experiment": experiment.name,
        "experiment_file": args.experiment.name,
        "split_list": group.manifest_path.name,
        "split_list_seed": group.manifest_seed,
        "group": experiment.group,
        "split": group.split,
        "max_clips": args.max_clips,
        "clips": group.clips,
        "run_file": args.run.name,
        "run_file_sha256": file_sha256(args.run),
        "runner": run.runner,
        "script_is_current": script_is_current,
        "n_answers_not_used": n_not_used,
    }
    write_metrics(args.out, payload, provenance=provenance)
    print_main_metric(payload)
    code = print_axis_check(payload["axis_check"]) if "axis_check" in payload else EXIT_OK
    print(f"tool time per frame: median {payload['seconds_per_frame']['median']:.3f} s")
    print(f"metrics file: {args.out}")
    return code


# --- video-frames and overlay: images below the data folder -----------------------


def repository_work_trees() -> list[Path]:
    """Return the top folder of every work tree of the repository of the code: the main one and each linked one.

    The work tree of the results branch is one of them. ``CliError`` is raised when git
    cannot list them: then a folder for images cannot be checked.
    """
    listing = git_output(repository_root(), "worktree", "list", "--porcelain")
    if listing is None:
        raise CliError(
            f"git cannot list the work trees of the repository {repository_root()}, so a folder for images cannot "
            "be checked: the package must run from its git checkout"
        )
    return [Path(line.removeprefix("worktree ")) for line in listing.splitlines() if line.startswith("worktree ")]


def refuse_pixels_in_repository(path: Path, option: str) -> None:
    """Raise ``CliError`` when a folder for images lies inside a work tree of the repository (rule 2 of ``CLAUDE.md``).

    The folder and each folder above it that exists are compared with the top folder of
    every work tree as places on the disk, so another letter case or a link does not pass.
    """
    target = path.resolve()
    existing = [folder for folder in (target, *target.parents) if folder.exists()]
    for tree in [repository_root(), *repository_work_trees()]:
        if tree.exists() and any(os.path.samefile(folder, tree) for folder in existing):
            raise CliError(
                f"{option} {path} lies inside the work tree {tree} of the repository: images never go into the "
                f"repository (rule 2 of CLAUDE.md). Write them below the data folder, for example {DATA_DIR}"
            )


def refuse_video_folder(folder: Path, cache_dir: Path) -> None:
    """Raise ``CliError`` when the folder of the videos of a video step lies in the repository or in the cache folder.

    The cache folder goes to the results branch, which is text only (rule 2 of
    ``CLAUDE.md``), so no video may lie below it. The two folders are compared as places
    on the disk (``_lies_in``), so another letter case or a link does not pass.
    """
    refuse_pixels_in_repository(folder, "--video-dir")
    target = folder.resolve()
    cache = cache_dir.resolve()
    if _lies_in(target, cache) or _lies_in(cache, target):
        raise CliError(
            f"--video-dir {folder} and the cache folder {cache_dir} overlap: the cache goes to the results branch, "
            f"which is text only (rule 2 of CLAUDE.md). Write the videos below the data folder, for example "
            f"{VIDEO_INPUT_DIR}"
        )


def _lies_in(inner: Path, outer: Path) -> bool:
    """Say whether the folder ``inner`` is the folder ``outer`` or lies below it, both given as resolved paths.

    A folder and a folder above it are the same place as ``outer`` when the disk says so
    (``os.path.samefile``), so another letter case on a disk that does not tell cases
    apart is found. A path that does not exist yet is compared as text without letter
    case: on a disk that tells cases apart this refuses more than it must, never less.
    """
    outer_exists = outer.exists()
    for folder in (inner, *inner.parents):
        if outer_exists and folder.exists():
            if os.path.samefile(folder, outer):
                return True
        elif str(folder).casefold() == str(outer).casefold():
            return True
    return False


def _cmd_video_frames(args: argparse.Namespace) -> int:
    """Write every frame of a video file as a PNG image into a new clip folder below ``--out``.

    The clip folder has the name of the clip id of the video, as ``run`` gives it: the name
    of the file without its ending. It holds the frame folder ``img1`` and the record of the
    video with its SHA-256 (``frames.write_video_frames``). ``clip-job`` reads the folder,
    and ``run`` reads the video itself: both see the same frames. A video whose frames are
    not sure (``frames.check_frame_times``) is refused before any image is written.
    """
    refuse_pixels_in_repository(args.out, "--out")
    if not args.video.is_file():
        raise CliError(f"{args.video} is not a video file")
    video = disk_spelling(args.video.absolute())
    video_sha256 = file_sha256(video)
    source = VideoFileSource(video)
    info = source.info
    print(
        f"video {video.name}: {info.n_frames} frames, {info.fps:g} frames per second, "
        f"{info.width} x {info.height} pixels, SHA-256 {video_sha256}"
    )
    clip_folder = args.out / info.clip_id
    for left in sorted(args.out.glob(f".{glob.escape(info.clip_id)}.partial-*")):
        print(f"a stopped run of video-frames left the hidden folder {left}: remove it to free its space")
    try:
        record = write_video_frames(source, video_sha256, clip_folder)
    except FileExistsError as error:
        raise CliError(str(error)) from error
    size = sum(path.stat().st_size for path in (clip_folder / FRAME_FOLDER).iterdir())
    print(f"frames: {record.n_frames} PNG images, {size / 1e6:.1f} MB, in {clip_folder / FRAME_FOLDER}")
    print(f"clip folder for clip-job: {clip_folder}")
    return EXIT_OK


@dataclass(frozen=True)
class MappingSource:
    """The pitch mappings of one source of the command ``overlay``, ready for the choice of frames."""

    #: Name of the source: the experiment, or the clip of a run.
    name: str
    #: Name of the image folder before its seed and count: the name, and for a run also the first
    #: 8 digits of the SHA-256 of its result file, so that two runs of one clip get two folders.
    folder: str
    #: The engine of the mappings, as a result file names engines.
    engine: str
    #: Number of frames that the source covers.
    n_frames: int
    #: The frames with a mapping, with their paint alignment.
    mapped: list[MappedFrame]
    #: The frames without a mapping, each as (clip id, frame, why it has none). They get no image.
    unmapped: list[tuple[str, int, str]]
    #: The frames of each clip.
    sources: dict[str, FrameSource]
    #: What the list file records about the source: file names, check sums, group.
    facts: dict[str, Any]


def mapped_frames(
    source: FrameSource, records: Sequence[HomographyRecord], origin: Callable[[HomographyRecord], str]
) -> list[MappedFrame]:
    """Return the frames of a clip that have a mapping, each with the paint alignment of its mapping in its frame."""
    mapped: list[MappedFrame] = []
    for record in records:
        if not record.valid:
            continue
        matrix = matrix_from_record(record)
        alignment = paint_alignment(source.read(record.frame), matrix)
        mapped.append(
            MappedFrame(
                clip=source.info.clip_id,
                frame=record.frame,
                matrix=matrix,
                origin=origin(record),
                score=alignment.score,
                reason=alignment.reason,
            )
        )
    return mapped


def probe_mappings(args: argparse.Namespace) -> MappingSource:
    """Return the landmark fits of a pitch probe, from its stored answers only (``probes.pitch_probe_fits``).

    The backend only replays: a frame without a stored answer stops the command, and no model
    is called. The fit uses the RANSAC threshold of the pipeline, as the probe does.
    """
    experiment = load_experiment(args.experiment)
    if experiment.probe != "pitch":
        raise CliError(
            f"the experiment file {args.experiment} describes a {experiment.probe} probe: an overlay needs the pitch "
            "probe, whose landmarks give the mappings"
        )
    group = clips_of_group(args, experiment.group)
    clips = [labelled_clip(args.root, group.split, clip) for clip in group.clips]
    cache = _cache_dir(args)
    print(f"cache folder: {cache} (stored answers only: no model call)")
    backend = CachedBackend(CassetteStore(cache, allow_other_images=args.allow_other_images))
    fits = probes.pitch_probe_fits(
        clips,
        backend,
        experiment.step,
        frames_per_clip=experiment.per_clip,
        seed=experiment.seed,
        ransac_px=PipelineConfig().ransac_px,
        repeat=experiment.repeat,
    )
    sources = {clip.source.info.clip_id: clip.source for clip in clips}
    mapped = [
        item
        for clip_id, pairs in fits.items()
        for item in mapped_frames(
            sources[clip_id], [record for record, _ in pairs], lambda record: "landmark fit of this frame"
        )
    ]
    unmapped = [
        (
            clip_id,
            record.frame,
            "the landmarks of the stored answer give no mapping" if parsed else "the stored answer cannot be parsed",
        )
        for clip_id, pairs in fits.items()
        for record, parsed in pairs
        if not record.valid
    ]
    return MappingSource(
        name=experiment.name,
        folder=experiment.name,
        engine=f"model:{model_label(experiment.step.provider, experiment.step.model)} landmarks, fit by code",
        n_frames=sum(len(pairs) for pairs in fits.values()),
        mapped=mapped,
        unmapped=unmapped,
        sources=sources,
        facts={**group_facts(args, group, experiment.group), "experiment_file": args.experiment.name},
    )


def tool_mappings(args: argparse.Namespace) -> MappingSource:
    """Return the cameras of a calibration tool on the frames of its probe, from its run file (as ``tool-score``)."""
    experiment = load_tool_experiment(args.experiment)
    if not isinstance(experiment.tool, PnlcalibSettings):
        raise CliError(
            f"the experiment file {args.experiment} measures the tool {experiment.tool.name}: an overlay needs a "
            "calibration tool, whose cameras are the mappings"
        )
    group = clips_of_group(args, experiment.group)
    clips = [labelled_clip(args.root, group.split, clip) for clip in group.clips]
    run = read_tool_run(args.run)
    answers = probe_answers(run, experiment, frames_of_probe(clips, experiment), args.run)
    camera_of = _adapted(homography_from_payload, answers, args.run)
    sources = {clip.source.info.clip_id: clip.source for clip in clips}
    cameras = {
        clip_id: [camera_of(key_clip, frame) for key_clip, frame in answers if key_clip == clip_id]
        for clip_id in sources
    }
    mapped = [
        item
        for clip_id, records in cameras.items()
        for item in mapped_frames(sources[clip_id], records, lambda record: "camera of the tool for this frame")
    ]
    unmapped = [
        (clip_id, record.frame, "the tool gave no camera")
        for clip_id, records in cameras.items()
        for record in records
        if not record.valid
    ]
    return MappingSource(
        name=experiment.name,
        folder=experiment.name,
        engine=f"tool:{experiment.tool.name}",
        n_frames=len(answers),
        mapped=mapped,
        unmapped=unmapped,
        sources=sources,
        facts={
            **group_facts(args, group, experiment.group),
            "experiment_file": args.experiment.name,
            "run_file": args.run.name,
            "run_file_sha256": file_sha256(args.run),
        },
    )


def group_facts(args: argparse.Namespace, group: GroupClips, name: str) -> dict[str, Any]:
    """Return what the list file of an overlay records about the clips of a group."""
    return {
        "split_list": group.manifest_path.name,
        "split_list_seed": group.manifest_seed,
        "group": name,
        "split": group.split,
        "max_clips": args.max_clips,
        "clips": group.clips,
    }


def run_mappings(args: argparse.Namespace) -> MappingSource:
    """Return the mappings that a run used on its keyframes, from its result file and the frames of its clip.

    The mapping of a keyframe is the one of the result file (``ClipResult.homographies``):
    the landmark fit or the camera of the tool of that frame when the run kept it, else the
    mapping of another frame carried with the camera motion. A keyframe without a mapping is
    not drawn; it is listed with the reason. A frame of a transition (a shot of one frame
    between the cuts of the result, ``MotionField.transition_frames``: a frame of a dissolve)
    has no mapping in any run, and its reason says so. The keyframes come from the settings of
    the run (its provenance). The clip must be the clip of the result file: the same clip id,
    frame count and image size, and the video of its clip runs (``check_result_video``).
    """
    try:
        document = json.loads(args.result.read_text(encoding="utf-8"))
        result = ClipResult.model_validate(document["result"])
        config = PipelineConfig.model_validate(document["provenance"]["config"])
    except OSError as error:
        raise CliError(f"the result file {args.result} cannot be read: {error}") from error
    except (ValueError, KeyError, TypeError) as error:
        raise wrong_file("result file", args.result, error) from error
    info = result.info
    source = open_clip(args.clip, fps=None if args.clip.is_file() else info.fps, labels=None)
    seen = source.info
    if (seen.clip_id, seen.n_frames, seen.width, seen.height) != (info.clip_id, info.n_frames, info.width, info.height):
        raise CliError(
            f"--clip {args.clip} is the clip {seen.clip_id} with {seen.n_frames} frames of {seen.width} x "
            f"{seen.height} pixels, but the result file {args.result} is of the clip {info.clip_id} with "
            f"{info.n_frames} frames of {info.width} x {info.height} pixels"
        )
    clip_sha256 = file_sha256(args.clip) if args.clip.is_file() else video_of_folder(args.clip, seen)
    check_result_video(document, args.result, args.clip, clip_sha256)
    pitch_engine = result.engines.get("pitch_mapping")
    if pitch_engine is None:
        raise CliError(f"the result file {args.result} names no engine of the pitch mapping")
    engine = f"{result.engines.get('landmarks')} landmarks, fit by code" if pitch_engine == "code" else pitch_engine
    by_frame = {record.frame: record for record in result.homographies}
    keyframes = keyframe_indices(info.n_frames, info.fps, config.keyframe_interval_s)
    records = [by_frame.get(frame, HomographyRecord(frame=frame)) for frame in keyframes]
    transitions = MotionField([None] * (info.n_frames - 1), result.cuts).transition_frames()
    unmapped = [
        (
            info.clip_id,
            record.frame,
            "a frame of a transition (a dissolve, or a lone frame between two cuts): the run gives it no mapping"
            if record.frame in transitions
            else "the run has no mapping for this frame",
        )
        for record in records
        if not record.valid
    ]

    def origin(record: HomographyRecord) -> str:
        if record.source == "propagated":
            return "carried from another frame with the camera motion"
        return "landmark fit of this frame" if pitch_engine == "code" else "camera of the tool for this frame"

    result_sha256 = file_sha256(args.result)
    return MappingSource(
        name=info.clip_id,
        folder=f"{info.clip_id}-result{result_sha256[:8]}",
        engine=engine,
        n_frames=len(keyframes),
        mapped=mapped_frames(source, records, origin),
        unmapped=unmapped,
        sources={info.clip_id: source},
        facts={
            "result_file": args.result.name,
            "result_file_sha256": result_sha256,
            "run_commit": document["provenance"].get("commit"),
            "clip": info.clip_id,
        },
    )


def check_result_video(document: Mapping[str, Any], path: Path, clip: Path, seen: str | None) -> None:
    """Raise ``CliError`` unless ``clip`` holds the frames of the video of every clip run that a result file used.

    ``run`` used a clip run of the frames of a video only with that video (``check_clip_run_video``);
    the result file records its SHA-256 (``clip_run_facts``), or None for clip runs of images of
    no video. ``seen`` is the SHA-256 of the video of ``clip``: of the file, or of the record of
    its frame folder (``video_of_folder``), or None for images of no video. A video of the same
    name, frame count and size with other pixels would get the mappings of the run drawn on
    other frames. A result without clip runs (the model engine), or made before clip runs named
    their video, has nothing to compare.
    """
    tool_runs = document["provenance"].get("tool_runs") or {}
    # A result file made before clip runs named their video has no key: nothing can be compared.
    wanted = {facts["video_sha256"] for facts in tool_runs.values() if "video_sha256" in facts}
    if not wanted or wanted == {seen}:
        return
    made_from = ", ".join(
        sorted(f"the video with the SHA-256 {sha}" if sha else "images of no video" for sha in wanted)
    )
    given = f"the video with the SHA-256 {seen}" if seen else "a folder of images of no video"
    raise CliError(f"the result file {path} used clip runs of {made_from}, but --clip {clip} is {given}: give that clip")


#: The function that collects the mappings of each source of the command ``overlay``.
OVERLAY_SOURCES: dict[str, Callable[[argparse.Namespace], MappingSource]] = {
    "probe": probe_mappings,
    "tool": tool_mappings,
    "run": run_mappings,
}


def _cmd_overlay(args: argparse.Namespace) -> int:
    """Draw the pitch lines of chosen mappings on their frames, for the check by eye of gate G3.

    The source gives every mapping it has (``OVERLAY_SOURCES``) with its paint alignment;
    ``overlays.choose_frames`` chooses ``--count`` of them with ``--seed``. Each chosen frame
    becomes one PNG image (``render.draw_mapping_check``) in a new folder below ``--out``,
    and the list of the chosen frames goes to ``--list`` as text without pixels and without
    label data (``overlays.list_document``), for the results branch. A list file that is
    already there is not replaced: it may hold the verdicts of a check by eye.
    """
    refuse_pixels_in_repository(args.out, "--out")
    if args.list.exists():
        raise CliError(
            f"the list file {args.list} is already there and may hold the verdicts of a check by eye: give another "
            "--list"
        )
    state = code_state(repository_root())
    mappings = OVERLAY_SOURCES[args.source](args)
    if not mappings.mapped:
        raise CliError(
            f"none of the {mappings.n_frames} frames of {mappings.name} has a pitch mapping: nothing to draw"
        )
    chosen = choose_frames(mappings.mapped, count=args.count, seed=args.seed)
    folder_name = f"{mappings.folder}-seed{args.seed}-count{args.count}"
    if Path(folder_name).name != folder_name:
        raise CliError(
            f"the name {mappings.name!r} holds a folder separator, so its images would not go below --out: give the "
            "experiment a name without one"
        )
    folder = args.out / folder_name
    if folder.exists():
        raise CliError(f"{folder} is already there: remove it, or give another --out")
    print(
        f"overlay of {mappings.name} ({mappings.engine}): {len(mappings.mapped)} of {mappings.n_frames} frames have a "
        f"mapping; {len(chosen)} chosen with the seed {args.seed}"
    )
    if mappings.unmapped:
        print(f"{len(mappings.unmapped)} of {mappings.n_frames} frames have no mapping and get no image:")
        for reason in dict.fromkeys(reason for _, _, reason in mappings.unmapped):
            frames = ", ".join(f"{clip} {frame}" for clip, frame, why in mappings.unmapped if why == reason)
            print(f"  {reason}: {frames}")
    if len(chosen) < args.count:
        print(f"only {len(chosen)} frames have a mapping, fewer than the count {args.count}: all of them are drawn")
    provenance = {
        "command": f"overlay {args.source}",
        "commit": state.commit,
        "dirty": state.dirty,
        "package_version": __version__,
        "written_at": utc_now_text(),
        **mappings.facts,
    }
    document = list_document(
        chosen,
        source=args.source,
        name=mappings.name,
        engine=mappings.engine,
        seed=args.seed,
        count=args.count,
        n_frames=mappings.n_frames,
        unmapped=mappings.unmapped,
        provenance=provenance,
    )
    # The list file is written before the image folder gets its name, so a failed write leaves
    # neither of them, and the next run is not refused.
    partial = folder.with_name(f".{folder.name}.partial-{secrets.token_hex(4)}")
    partial.mkdir(parents=True)
    list_started = False
    try:
        for position, item in enumerate(chosen):
            image = draw_mapping_check(
                mappings.sources[item.clip].read(item.frame),
                item.matrix,
                caption(item, engine=mappings.engine, position=position, count=len(chosen)),
            )
            write_png(partial / image_name(position, item), image)
        # The list file was not there at the start, so a file there now, also a part of one, is ours.
        list_started = True
        write_json(args.list, document)
        os.rename(partial, folder)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        if list_started:
            args.list.unlink(missing_ok=True)
        raise
    for position, item in enumerate(chosen):
        paint = f"paint score {item.score:.3f}" if item.score is not None else f"cannot judge ({item.reason})"
        print(f"  {image_name(position, item)}: clip {item.clip}, frame {item.frame}, {item.origin}, {paint}")
    print(f"images: {folder}")
    print(f"list file: {args.list}")
    print("write the verdict of each image, right or wrong, into its field by_eye in the list file")
    return EXIT_OK


# --- eval ----------------------------------------------------------------------------


def _cmd_eval(args: argparse.Namespace) -> int:
    """Run the scorer on the prediction files of one tracker, and print the scores per clip and of all clips.

    The scores file names the commit of the clone of the scorer repository that holds the scorer script.
    """
    scores = run_trackeval(
        args.scorer_python,
        args.scorer_script,
        args.root,
        args.pred,
        split=args.split,
        tracker=args.tracker,
        clips=args.clips,
        output_dir=args.out,
        drop_env=KEY_VARIABLES,
    )
    width = max(len(name) for name in scores)
    print(f"{'clip':<{width}}  GS-HOTA    DetA    AssA")
    for name in [*args.clips, COMBINED]:
        row = scores[name]
        print(f"{name:<{width}}  {row['HOTA']:7.2f} {row['DetA']:7.2f} {row['AssA']:7.2f}")
    state = code_state(repository_root())
    write_metrics(
        args.out / SCORES_NAME,
        {"split": args.split, "tracker": args.tracker, "clips": list(args.clips), "scores": scores},
        provenance={
            "command": "eval",
            "commit": state.commit,
            "dirty": state.dirty,
            "scorer_commit": scorer_commit(args.scorer_script),
        },
    )
    print(f"scores: {args.out / SCORES_NAME}")
    return EXIT_OK


# --- sync ----------------------------------------------------------------------------


def result_files(folder: Path) -> tuple[list[Path], list[Path]]:
    """Walk a result folder. Return its text result files, and the entries that are refused.

    A text result file is a file that ends with ``.json`` or ``.jsonl``. Every other
    file is refused, for example an image or a video. A symbolic link is refused too.
    Linked folders are not entered.
    """
    accepted: list[Path] = []
    refused: list[Path] = []
    for current, folder_names, file_names in os.walk(folder):
        for name in folder_names:
            if (Path(current) / name).is_symlink():
                refused.append(Path(current) / name)
        for name in file_names:
            path = Path(current) / name
            is_result = path.suffix.lower() in RESULT_SUFFIXES and not path.is_symlink()
            (accepted if is_result else refused).append(path)
    return sorted(accepted), sorted(refused)


def results_work_tree(folder: Path) -> Path:
    """Return the top folder of the git work tree that holds a result folder.

    ``CliError`` is raised when the folder is not in a git work tree, and when the work
    tree is not on the results branch. The guard checks only files inside a repository,
    and result files must never go to the code branch.
    """
    advice = (
        f"Keep the result files in the work tree of the branch '{RESULTS_BRANCH}': create it one time with "
        f"'git worktree add -b {RESULTS_BRANCH} ../{RESULTS_WORKTREE}' in the repository, "
        f"and write the results below ../{RESULTS_WORKTREE}/{RESULTS_FOLDER}. When ../{RESULTS_WORKTREE} is "
        "there already as a plain folder, move it away first, create the work tree (without -b when the "
        f"branch '{RESULTS_BRANCH}' exists already) and move the files back."
    )
    top = git_output(folder, "rev-parse", "--show-toplevel")
    if top is None or not top.strip():
        raise CliError(f"{folder} is not inside a git work tree, so the guard cannot check it. {advice}")
    work_tree = Path(top.strip()).resolve()
    branch = git_output(work_tree, "symbolic-ref", "--short", "HEAD")
    if branch is None or branch.strip() != RESULTS_BRANCH:
        found = "no branch" if branch is None else f"the branch '{branch.strip()}'"
        raise CliError(f"the work tree {work_tree} is on {found}, not on '{RESULTS_BRANCH}'. {advice}")
    return work_tree


def guard_violations(work_tree: Path, files: Sequence[Path]) -> list[str]:
    """Run ``tools/guard.py --paths`` on the files and return the lines of its report. Empty means clean."""
    script = repository_root() / GUARD_SCRIPT
    if not script.is_file():
        raise CliError(f"the guard script {script} is missing; install the package in editable mode")
    lines: list[str] = []
    for start in range(0, len(files), GUARD_BATCH):
        batch = [str(path) for path in files[start : start + GUARD_BATCH]]
        completed = subprocess.run(
            [sys.executable, str(script), "--repo", str(work_tree), "--paths", *batch],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        if completed.returncode not in (0, 1):
            raise CliError(f"the guard could not check the files: {completed.stderr.strip()}")
        lines.extend(line for line in completed.stdout.splitlines() if line.strip())
    return lines


def secret_scan_findings(work_tree: Path, folder: Path) -> list[str] | None:
    """Run the secret scan of the git hooks on a result folder and return the lines of its report.

    An empty list means that nothing was found. None means that the scan did not run,
    because ``gitleaks`` is not installed. The scan uses the rules file
    ``.gitleaks.toml`` of the work tree, which is the file that the hooks of that work
    tree use. So a file that passes here also passes the scan before the commit.
    ``CliError`` is raised when the rules file is missing, and when the scan could not
    check the files. The report names the file, the line and the rule of every
    finding, with paths relative to the work tree. It never shows the secret.
    """
    tool = shutil.which(SECRET_SCAN_TOOL)
    if tool is None:
        return None
    if not (work_tree / SECRET_SCAN_RULES).is_file():
        raise CliError(
            f"the work tree {work_tree} has no file {SECRET_SCAN_RULES}, so the secret scan of its git hooks "
            f"cannot run. Take the file from the code branch: 'git checkout <code branch> -- {SECRET_SCAN_RULES}' "
            "in that work tree."
        )
    command = [
        tool,
        "dir",
        "--config",
        str(SECRET_SCAN_RULES),
        "--redact",
        "--verbose",
        "--no-banner",
        "--no-color",
        "--exit-code",
        str(SECRET_SCAN_FOUND),
        folder.relative_to(work_tree).as_posix(),
    ]
    completed = subprocess.run(command, cwd=work_tree, capture_output=True, text=True, errors="replace", check=False)
    if completed.returncode == 0:
        return []
    if completed.returncode != SECRET_SCAN_FOUND:
        raise CliError(f"the secret scan could not check the files: {completed.stderr.strip()}")
    return [line for line in completed.stdout.splitlines() if line.strip()]


def print_result_listing(folder: Path, files: Sequence[Path], work_tree: Path) -> None:
    """Print the text result files of a folder: cassettes by folder, every other file by name.

    A ledger file is listed as a cost ledger. A file below a folder with the name ``tools``
    is listed as a run file of an external tool: the folders are counted from the top of
    the work tree, so the listing of the folder ``tools`` itself names its files right.
    Any other file is listed as a metrics file.
    """
    cassettes: Counter[str] = Counter()
    print(f"{len(files)} text result files below {folder}:")
    for path in files:
        relative = path.relative_to(folder)
        if CASSETTE_NAME.fullmatch(path.name):
            cassettes[relative.parent.as_posix()] += 1
            continue
        if ledger_file_number(LEDGER_NAME, path.name) is not None:
            kind = "cost ledger"
        elif TOOL_RUNS_FOLDER in path.relative_to(work_tree).parent.parts:
            kind = "tool run"
        else:
            kind = "metrics"
        print(f"  {kind}: {relative.as_posix()}")
    for parent, count in sorted(cassettes.items()):
        print(f"  cassettes: {parent} ({count} files)")


def print_git_commands(work_tree: Path, folder: Path, n_files: int, *, scanned: bool) -> None:
    """Print the git commands that put the result files on the results branch. They are not run.

    ``scanned`` says whether the secret scan ran next to the guard.
    """
    tree = shlex.quote(str(work_tree))
    message = shlex.quote(f"Add text results ({n_files} files)")
    checks = "The guard and the secret scan" if scanned else "The guard"
    print(f"{checks} found no problem. To send the files back, the owner runs:")
    print(f"  git -C {tree} add -- {shlex.quote(folder.relative_to(work_tree).as_posix())}")
    print(f"  git -C {tree} commit -m {message}")
    print(f"  git -C {tree} push origin {RESULTS_BRANCH}")
    print(
        f"Do not run the push before the owner has decided where the branch {RESULTS_BRANCH} goes "
        "(docs/DECISIONS.md, last list): a push to a public remote cannot be taken back. "
        "Use the name of the remote when it is not origin."
    )
    print("This command ran no git command that changes the repository.")


def _cmd_sync(args: argparse.Namespace) -> int:
    """List the text result files below a folder, check them and print the git commands.

    The files are checked with the guard and then with the secret scan of the git hooks
    (``secret_scan_findings``), so that the commit of the files is not refused later. A
    printed line says so when the secret scan is not installed. The command refuses the
    folder when it holds anything but text result files. It never runs a git command
    that changes state.
    """
    if not args.results.is_dir():
        raise CliError(f"{args.results} is not a folder")
    # Git names the work tree as the disk spells it. The folder must be spelled in the same way for the guard.
    folder = disk_spelling(args.results.resolve())
    files, refused = result_files(folder)
    if refused:
        for path in refused:
            print(f"refused: {path.relative_to(folder).as_posix()} is not a text result file")
        print(
            f"{len(refused)} entries below {folder} are refused: a result file ends with .json or .jsonl and is "
            "not a link. Remove these entries, or keep the result files in a folder of their own. "
            "No git command is printed."
        )
        return EXIT_FAILED
    if not files:
        raise CliError(f"{folder} holds no result file")
    work_tree = results_work_tree(folder)
    print_result_listing(folder, files, work_tree)
    violations = guard_violations(work_tree, files)
    if violations:
        print(f"The guard refuses {len(violations)} places. No git command is printed.")
        for line in violations:
            print(f"  {line}")
        return EXIT_FAILED
    findings = secret_scan_findings(work_tree, folder)
    if findings:
        print("The secret scan of the git hooks refuses the files. No git command is printed.")
        for line in findings:
            print(f"  {line}")
        return EXIT_FAILED
    if findings is None:
        print(
            f"The secret scan of the git hooks did not run here: {SECRET_SCAN_TOOL} is not installed. "
            "The hooks need it before the commit."
        )
    print_git_commands(work_tree, folder, len(files), scanned=findings is not None)
    return EXIT_OK


# --- Argument parser -----------------------------------------------------------------


def _path(text: str) -> Path:
    """Turn a command-line value into a path. A leading ``~`` becomes the home folder."""
    return Path(text).expanduser()


def _positive_int(text: str) -> int:
    """Read a whole number of at least 1."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text} is not a whole number of at least 1")
    return value


def _requests_cap(text: str) -> int:
    """Read a cap of model requests: a whole number of 0 or more."""
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"{text} is not a whole number of 0 or more")
    return value


def _port(text: str) -> int:
    """Read a port: a whole number from 0 (a free port) to 65535."""
    value = int(text)
    if not 0 <= value <= 65535:
        raise argparse.ArgumentTypeError(f"{text} is not a port: give 0 to 65535")
    return value


def _file_or_dash(text: str) -> str:
    """Read a file path or ``-``; a leading ``~`` of a path becomes the home folder."""
    return text if text == "-" else str(Path(text).expanduser())


def _seed(text: str) -> int:
    """Read a seed: a whole number that is not negative."""
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"{text} is negative: a seed must be 0 or more")
    return value


def _frame_rate(text: str) -> float:
    """Read a frame rate: a finite number above zero."""
    value = float(text)
    if not 0.0 < value < math.inf:
        raise argparse.ArgumentTypeError(f"{text} is not a frame rate above zero")
    return value


def _model_list(text: str) -> list[str]:
    """Read models that are separated by commas: a Gemini name, or ``openrouter:<slug>``. Each one time."""
    models = [name.strip() for name in text.split(",")]
    if not all(models) or len(set(models)) != len(models):
        raise argparse.ArgumentTypeError("give models separated by commas, each one time")
    for name in models:
        try:
            model_ref(name)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error
    return models


def _cache_dir(args: argparse.Namespace) -> Path:
    """Return the cache folder of a command: the given one, else the default one.

    The default cache folder lies in the work tree of the results branch, next to the
    repository. ``CliError`` is raised when that work tree is not there. A command that
    made the folder itself would leave a plain folder at the place of the work tree: the
    setup command of the work tree then fails, and the paid answers cannot be sent back.
    """
    if args.cache is not None:
        return args.cache
    results_work_tree(repository_root().parent / RESULTS_WORKTREE)
    return default_cache_dir()


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    """Add the options that every command with model calls has."""
    _add_cache_options(parser, allow_dirty=True)


def _add_cache_options(parser: argparse.ArgumentParser, *, allow_dirty: bool = False) -> None:
    """Add the options of the cache folder and of the hard stop; ``allow_dirty`` adds ``--allow-dirty`` between them.

    ``stream`` has its own ``--allow-dirty``.
    """
    parser.add_argument(
        "--cache",
        type=_path,
        metavar="DIR",
        help=(
            "folder of the stored model answers and of the cost ledger "
            f"(default: ../{RESULTS_WORKTREE}/{RESULTS_FOLDER}/{CACHE_FOLDER} next to the repository)"
        ),
    )
    if allow_dirty:
        parser.add_argument(
            "--allow-dirty",
            action="store_true",
            help="start although the git work tree of the code has changes; the output file records it",
        )
    parser.add_argument(
        "--start-over-budget",
        action="store_true",
        help=(
            "start although the cost ledger and the estimate of this run together are above the hard stop; "
            "the run still stops when the ledger reaches the hard stop"
        ),
    )


def _add_stored_answer_options(parser: argparse.ArgumentParser) -> None:
    """Add the options of the commands that read stored model answers: ``run`` and ``probe``."""
    parser.add_argument(
        "--replay", action="store_true", help="use stored model answers only; a missing answer stops the run"
    )
    parser.add_argument(
        "--allow-other-images",
        action="store_true",
        help=(
            "use a stored answer although it was recorded with other image bytes than the clip shows now; "
            "only for the same footage, for example after an update of the image library or for a kit legend "
            "from other frames; a warning names each such answer"
        ),
    )


def _add_scorer_options(parser: argparse.ArgumentParser) -> None:
    """Add the options that name the scorer: its interpreter and its script."""
    parser.add_argument(
        "--scorer-python",
        type=_path,
        required=True,
        metavar="PATH",
        help="Python interpreter of the scorer environment",
    )
    parser.add_argument(
        "--scorer-script",
        type=_path,
        required=True,
        metavar="PATH",
        help="the script scripts/run_soccernet_gs.py of the sn-trackeval repository",
    )


def _add_demo(commands: Any) -> None:
    """Add the command ``demo`` to the sub-parsers object of argparse."""
    parser = commands.add_parser("demo", help="synthetic clip and fake model through the whole package; no network")
    parser.add_argument("--out", type=_path, required=True, metavar="DIR", help="folder of the output files")
    parser.add_argument(
        "--frames",
        type=_positive_int,
        default=DEMO_FRAMES,
        metavar="N",
        help=f"frames of the clip (default: {DEMO_FRAMES})",
    )
    parser.add_argument("--seed", type=_seed, default=0, metavar="S", help="seed of the synthetic scene (default: 0)")
    parser.add_argument(
        "--engine",
        choices=tuple(DEMO_ENGINES),
        default="model",
        help="model: the fake model answers the model steps; detector: stand-ins of the detector and of the "
        "calibration answer for every frame (default: model)",
    )
    parser.set_defaults(handler=_cmd_demo)


def _add_run(commands: Any) -> None:
    """Add the command ``run`` to the sub-parsers object of argparse."""
    parser = commands.add_parser("run", help="one clip through the pipeline: result file, prediction file and video")
    parser.add_argument(
        "--clip",
        type=_path,
        required=True,
        metavar="PATH",
        help="a folder of frame images (or a clip folder of the dataset), or a video file",
    )
    parser.add_argument(
        "--out",
        type=_path,
        required=True,
        metavar="DIR",
        help=(
            "output folder: <DIR>/<clip id>/ holds the result file and the video, "
            "<DIR>/predictions holds the prediction files"
        ),
    )
    parser.add_argument(
        "--config", type=_path, metavar="FILE", help="settings file (YAML); default: the built-in settings"
    )
    parser.add_argument(
        "--roster", type=_path, metavar="FILE", help="roster file with the player names by shirt number"
    )
    parser.add_argument(
        "--labels",
        type=_path,
        metavar="FILE",
        help="label file of the clip; with it the prediction file for the scorer is written",
    )
    parser.add_argument(
        "--fps",
        type=_frame_rate,
        metavar="FPS",
        help="frame rate of a folder of frame images, when no label file is given",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        metavar="NAME",
        help=f"split name in the prediction path (default: {DEFAULT_SPLIT})",
    )
    parser.add_argument(
        "--tracker",
        default=DEFAULT_TRACKER,
        metavar="NAME",
        help=f"tracker name in the prediction path (default: {DEFAULT_TRACKER})",
    )
    parser.add_argument(
        "--detections",
        type=_path,
        metavar="FILE",
        help="first part of the clip run of the detector; needed for the people engine detector",
    )
    parser.add_argument(
        "--calibration",
        type=_path,
        metavar="FILE",
        help="first part of the clip run of the calibration; needed for the pitch engine calibration",
    )
    _add_stored_answer_options(parser)
    _add_model_options(parser)
    parser.set_defaults(handler=_cmd_run)


def _add_stream(commands: Any) -> None:
    """Add the command ``stream`` to the sub-parsers object of argparse."""
    parser = commands.add_parser(
        "stream", help="detector engine frame by frame with a fixed delay: the files of run, and per-frame times"
    )
    parser.add_argument("--out", type=_path, required=True, metavar="DIR", help="output folder, as for run")
    parser.add_argument("--clip", type=_path, metavar="PATH", help="a folder of frame images, or a video file")
    parser.add_argument(
        "--synthetic", action="store_true", help="the synthetic clip with the stand-ins of the tools (no --clip)"
    )
    parser.add_argument(
        "--frames", type=_positive_int, metavar="N", help=f"frames of the synthetic clip (default: {DEMO_FRAMES})"
    )
    parser.add_argument("--seed", type=_seed, metavar="S", help="seed of the synthetic clip (default: 0)")
    parser.add_argument(
        "--config",
        type=_path,
        metavar="FILE",
        help="settings file (YAML); default: the engines of configs/engines/detector_calibration.yaml",
    )
    parser.add_argument("--labels", type=_path, metavar="FILE", help="label file of the clip, as for run")
    parser.add_argument(
        "--fps",
        type=_frame_rate,
        metavar="FPS",
        help="frame rate of a folder of frame images, when no label file is given",
    )
    parser.add_argument(
        "--split", metavar="NAME", help=f"split name in the prediction path, with --labels (default: {DEFAULT_SPLIT})"
    )
    parser.add_argument(
        "--tracker", metavar="NAME", help=f"tracker name in the prediction path, with --labels (default: {DEFAULT_TRACKER})"
    )
    parser.add_argument("--detections", type=_path, metavar="FILE", help="first part of the clip run of the detector")
    parser.add_argument("--calibration", type=_path, metavar="FILE", help="first part of the clip run of the calibration")
    parser.add_argument(
        "--lines",
        type=_path,
        metavar="FILE",
        help=f"the lines of a stream (format {STREAM_FORMAT}, the output of merge-feeds); - for a pipe",
    )
    parser.add_argument(
        "--queue",
        type=_positive_int,
        metavar="N",
        help=f"with --lines: lines read ahead before the reading waits (default: {STREAM_QUEUE_LINES})",
    )
    parser.add_argument(
        "--max-behind",
        type=_behind_seconds,
        metavar="SECONDS",
        help=(
            "overload policy of a live stream (--lines - or a named pipe): drop a frame whose line waited longer "
            "than this before the frame path took it; a dropped frame is not measured and has no state (default: off)"
        ),
    )
    parser.add_argument(
        "--live-out",
        type=_file_or_dash,
        metavar="FILE",
        help=(
            f"write each decided frame as one JSON line (format {LIVE_FORMAT}) as it is decided: a file, a named "
            "pipe, or - for the standard output (the output of the command then goes to the error stream)"
        ),
    )
    parser.add_argument(
        "--delay",
        type=_frames_or_clip,
        default=StreamSettings().delay_frames,
        metavar="N",
        help=f"frames between a frame and its decision, or clip (default: {StreamSettings().delay_frames})",
    )
    parser.add_argument(
        "--history",
        type=_history,
        default="default",
        metavar="N",
        help=(
            "frames of the window before the decided frame, or clip; default: the longest join plus the votes of a "
            "team-key change (123 frames at 25 frames per second with the default settings)"
        ),
    )
    parser.add_argument(
        "--kit-warmup",
        type=_positive_int,
        default=KIT_WARMUP_FRAMES,
        metavar="N",
        help=f"first frames on which the kit clusters are fitted (default: {KIT_WARMUP_FRAMES})",
    )
    parser.add_argument(
        "--allow-dirty", action="store_true", help="run on a work tree with changes; the result file says so"
    )
    parser.add_argument(
        "--numbers-max",
        type=_requests_cap,
        metavar="N",
        help=(
            "with the number engine model: the cap of number requests of this stream, the bound of its cost "
            f"estimate; a track that is ready after it gets no number (default: {DEFAULT_MAX_REQUESTS})"
        ),
    )
    _add_stored_answer_options(parser)
    _add_cache_options(parser)
    parser.set_defaults(handler=_cmd_stream)


def _add_doctor(commands: Any) -> None:
    """Add the command ``doctor`` to the sub-parsers object of argparse."""
    parser = commands.add_parser("doctor", help="checks of the machine, and of the model API (gate G0)")
    parser.add_argument("--no-api", action="store_true", help="check the machine only; call no model")
    parser.add_argument(
        "--models",
        type=_model_list,
        default=[DEFAULT_MODEL],
        metavar="M1,M2",
        help=(
            "models of the contract checks, separated by commas: a Gemini model name, or openrouter:<slug> "
            f"for a model of OpenRouter (default: {DEFAULT_MODEL})"
        ),
    )
    parser.add_argument(
        "--coords",
        choices=COORD_CONVENTIONS,
        default=GEMINI_CONVENTION,
        help=(
            "coordinate convention that the prompts ask the OpenRouter models for; a Gemini model always uses "
            f"{GEMINI_CONVENTION} (default: {GEMINI_CONVENTION})"
        ),
    )
    parser.add_argument(
        "--thinking-level",
        choices=get_args(ThinkingLevel),
        default="low",
        help="thinking level of the models; model_default sends none, for OpenRouter only (default: low)",
    )
    parser.add_argument(
        "--out", type=_path, metavar="FILE", help="write the report of the contract checks to this file (JSON)"
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="also send one video request per model (the video check); a model without a video rate gets none",
    )
    _add_model_options(parser)
    parser.set_defaults(handler=_cmd_doctor)


def _add_data(commands: Any) -> None:
    """Add the command ``data`` to the sub-parsers object of argparse."""
    parser = commands.add_parser("data", help="download and extract the dataset, and write the split list")
    parser.add_argument("--root", type=_path, required=True, metavar="DIR", help="data folder, outside the repository")
    parser.add_argument("--split", choices=("valid", "test"), required=True, help="split to download and extract")
    parser.add_argument("--clips", nargs="+", metavar="NAME", help="extract these clips only (default: all clips)")
    parser.add_argument(
        "--manifest", type=_path, metavar="FILE", help="write the split list to this file; needs --seed"
    )
    parser.add_argument("--seed", type=_seed, metavar="S", help="seed of the split list; needs --manifest")
    parser.set_defaults(handler=_cmd_data)


def _add_oracle(commands: Any) -> None:
    """Add the command ``oracle`` to the sub-parsers object of argparse."""
    parser = commands.add_parser("oracle", help="label-only checks of the scoring chain (gate G1); no model call")
    parser.add_argument("--root", type=_path, required=True, metavar="DIR", help="data folder")
    parser.add_argument(
        "--split", default=DEFAULT_SPLIT, metavar="NAME", help=f"split of the clips (default: {DEFAULT_SPLIT})"
    )
    parser.add_argument("--clips", nargs="+", metavar="NAME", help="clips to check (default: all clips of the split)")
    _add_scorer_options(parser)
    parser.add_argument(
        "--out",
        type=_path,
        required=True,
        metavar="DIR",
        help="output folder: prediction files, scorer output and oracle.json",
    )
    parser.set_defaults(handler=_cmd_oracle)


def _add_probe(commands: Any) -> None:
    """Add the command ``probe`` to the sub-parsers object of argparse."""
    parser = commands.add_parser("probe", help="measure one model step on labelled frames")
    parser.add_argument("kind", choices=PROBE_KINDS, help="the probe; it must be the probe of the experiment file")
    parser.add_argument("--experiment", type=_path, required=True, metavar="FILE", help="experiment file (YAML)")
    parser.add_argument("--root", type=_path, required=True, metavar="DIR", help="data folder")
    parser.add_argument("--out", type=_path, required=True, metavar="FILE", help="metrics file to write (JSON)")
    parser.add_argument(
        "--video-dir",
        type=_path,
        default=VIDEO_INPUT_DIR,
        metavar="DIR",
        help=f"folder of the videos of a probe with video input, outside the repository (default: {VIDEO_INPUT_DIR})",
    )
    _add_group_options(
        parser, "use only the first N clips of the group, for example for a cost pilot; the answers stay in the cache"
    )
    _add_stored_answer_options(parser)
    _add_model_options(parser)
    parser.set_defaults(handler=_cmd_probe)


def _add_group_options(parser: argparse.ArgumentParser, max_clips_help: str) -> None:
    """Add the options that name the clips of a group of the split list."""
    parser.add_argument(
        "--manifest",
        type=_path,
        metavar="FILE",
        help=f"split list (default: {SPLIT_LIST_FILE.as_posix()} of the repository)",
    )
    parser.add_argument("--max-clips", type=_positive_int, metavar="N", help=max_clips_help)


def _add_tool_job(commands: Any) -> None:
    """Add the command ``tool-job`` to the sub-parsers object of argparse."""
    parser = commands.add_parser("tool-job", help="write the frames and the settings that an external tool must run on")
    parser.add_argument(
        "--experiment", type=_path, required=True, metavar="FILE", help="experiment file of the tool (YAML)"
    )
    parser.add_argument("--root", type=_path, required=True, metavar="DIR", help="data folder")
    parser.add_argument("--out", type=_path, required=True, metavar="FILE", help="job file to write (JSON)")
    _add_group_options(parser, "use only the first N clips of the group")
    parser.set_defaults(handler=_cmd_tool_job)


def _add_clip_job(commands: Any) -> None:
    """Add the command ``clip-job`` to the sub-parsers object of argparse."""
    parser = commands.add_parser(
        "clip-job", help="write every frame of one clip and the settings that an external tool must run on"
    )
    parser.add_argument(
        "--tool",
        choices=tuple(TOOL_PROBES),
        required=True,
        help="the tool of the job: a detector family, which must be the family of the settings file, or pnlcalib",
    )
    parser.add_argument(
        "--clip",
        type=_path,
        required=True,
        metavar="DIR",
        help=(
            "a clip folder of the dataset, or a folder of frame images below --root; "
            "for a video file, run video-frames first"
        ),
    )
    parser.add_argument(
        "--config", type=_path, metavar="FILE", help="settings file of the run (YAML); default: the built-in settings"
    )
    parser.add_argument("--root", type=_path, required=True, metavar="DIR", help="data folder")
    parser.add_argument("--out", type=_path, required=True, metavar="FILE", help="clip job file to write (JSON)")
    parser.set_defaults(handler=_cmd_clip_job)


def _add_video_frames(commands: Any) -> None:
    """Add the command ``video-frames`` to the sub-parsers object of argparse."""
    parser = commands.add_parser(
        "video-frames", help="write every frame of a video file as a PNG image below the data folder, for clip-job"
    )
    parser.add_argument("--video", type=_path, required=True, metavar="FILE", help="the video file")
    parser.add_argument(
        "--out",
        type=_path,
        default=VIDEO_FRAMES_DIR.expanduser(),
        metavar="DIR",
        help=f"folder of the clip folders, outside the repository (default: {VIDEO_FRAMES_DIR})",
    )
    parser.set_defaults(handler=_cmd_video_frames)


def _add_overlay(commands: Any) -> None:
    """Add the command ``overlay`` with one sub-command per source of pitch mappings."""
    parser = commands.add_parser(
        "overlay", help="draw the pitch lines of chosen mappings on their frames, for the check by eye of gate G3"
    )
    sources = parser.add_subparsers(dest="source", metavar="SOURCE", required=True)
    probe = sources.add_parser("probe", help="the landmark fits of a pitch probe, from its stored answers")
    probe.add_argument("--experiment", type=_path, required=True, metavar="FILE", help="experiment file of the probe")
    probe.add_argument("--root", type=_path, required=True, metavar="DIR", help="data folder")
    _add_group_options(probe, "use only the first N clips of the group, as the probe did")
    probe.add_argument(
        "--cache",
        type=_path,
        metavar="DIR",
        help=f"folder of the stored model answers (default: ../{RESULTS_WORKTREE}/{RESULTS_FOLDER}/{CACHE_FOLDER})",
    )
    probe.add_argument(
        "--allow-other-images",
        action="store_true",
        help="use a stored answer although it was recorded with other image bytes than the clip shows now",
    )
    tool = sources.add_parser("tool", help="the cameras of a calibration tool, from its run file (as tool-score)")
    tool.add_argument("--experiment", type=_path, required=True, metavar="FILE", help="experiment file of the tool")
    tool.add_argument("--root", type=_path, required=True, metavar="DIR", help="data folder")
    tool.add_argument("--run", type=_path, required=True, metavar="FILE", help="run file of tools/run_tool.py")
    _add_group_options(tool, "use only the first N clips of the group, as tool-score did")
    run = sources.add_parser("run", help="the mappings that a run used on its keyframes, from its result file")
    run.add_argument("--result", type=_path, required=True, metavar="FILE", help="result.json of the run")
    run.add_argument(
        "--clip", type=_path, required=True, metavar="PATH", help="the clip of the run: its video file or its folder"
    )
    for sub in (probe, tool, run):
        sub.add_argument("--seed", type=int, required=True, help="seed of the choice of frames")
        sub.add_argument("--count", type=_positive_int, required=True, metavar="N", help="number of frames to draw")
        sub.add_argument(
            "--list", type=_path, required=True, metavar="FILE", help="list file of the chosen frames to write (JSON)"
        )
        sub.add_argument(
            "--out",
            type=_path,
            default=OVERLAYS_DIR.expanduser(),
            metavar="DIR",
            help=f"folder of the image folders, outside the repository (default: {OVERLAYS_DIR})",
        )
        sub.set_defaults(handler=_cmd_overlay)


def _add_tool_score(commands: Any) -> None:
    """Add the command ``tool-score`` to the sub-parsers object of argparse."""
    parser = commands.add_parser(
        "tool-score", help="measure the answers of an external tool on the frames of a probe; no model call"
    )
    parser.add_argument(
        "--experiment", type=_path, required=True, metavar="FILE", help="experiment file of the tool (YAML)"
    )
    parser.add_argument("--root", type=_path, required=True, metavar="DIR", help="data folder")
    parser.add_argument(
        "--run", type=_path, required=True, metavar="FILE", help="run file that tools/run_tool.py wrote (JSON)"
    )
    parser.add_argument("--out", type=_path, required=True, metavar="FILE", help="metrics file to write (JSON)")
    _add_group_options(parser, "measure only the first N clips of the group")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="start although the git work tree of the code has changes; the output file records it",
    )
    parser.set_defaults(handler=_cmd_tool_score)


def _add_eval(commands: Any) -> None:
    """Add the command ``eval`` to the sub-parsers object of argparse."""
    parser = commands.add_parser("eval", help="run the scorer on prediction files and print GS-HOTA, DetA and AssA")
    parser.add_argument("--root", type=_path, required=True, metavar="DIR", help="data folder with the label files")
    parser.add_argument(
        "--pred",
        type=_path,
        required=True,
        metavar="DIR",
        help="folder of the prediction files (<out>/predictions of the run command)",
    )
    parser.add_argument("--split", required=True, metavar="NAME", help="split of the clips")
    parser.add_argument("--tracker", required=True, metavar="NAME", help="tracker name of the prediction files")
    parser.add_argument("--clips", nargs="+", required=True, metavar="NAME", help="clips to score")
    _add_scorer_options(parser)
    parser.add_argument(
        "--out", type=_path, required=True, metavar="DIR", help="empty folder for the scorer output and scores.json"
    )
    parser.set_defaults(handler=_cmd_eval)


def _add_sync(commands: Any) -> None:
    """Add the command ``sync`` to the sub-parsers object of argparse."""
    parser = commands.add_parser("sync", help="check text result files and print the git commands that send them back")
    parser.add_argument(
        "--results",
        type=_path,
        required=True,
        metavar="DIR",
        help=f"folder of the result files, inside the work tree of the branch {RESULTS_BRANCH}",
    )
    parser.set_defaults(handler=_cmd_sync)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser of the command line: one sub-parser per command."""
    parser = argparse.ArgumentParser(
        prog="pitchvlm",
        description="Soccer broadcast clip to annotated video, 2D pitch radar and a measured score.",
        epilog="Exit code 0 means done, 1 means that a check or a gate failed, 2 means that the command could not run.",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    for add_command in (
        _add_demo,
        _add_run,
        _add_stream,
        _add_merge_feeds,
        _add_radar,
        _add_doctor,
        _add_data,
        _add_oracle,
        _add_probe,
        _add_tool_job,
        _add_tool_score,
        _add_clip_job,
        _add_video_frames,
        _add_overlay,
        _add_eval,
        _add_sync,
    ):
        add_command(commands)
    return parser


def _configure_logging() -> None:
    """Send the log lines of the package to the standard error stream.

    The HTTP libraries below the model SDK write the address of every request at the
    level INFO. They are set to WARNING, so that no request detail goes to the log.
    From version 3.0 the SDK sends its requests with ``httpx2``; the older names are for
    a machine that makes the SDK use ``httpx``.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for name in HTTP_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return the exit code. ``argv`` defaults to the arguments of the process.

    Every error of a command ends with the exit code 2 and with a line ``error: ...`` on
    the standard error stream. So the exit code 1 comes only from a command that finished
    and found a failed check or gate. For an error of ``KNOWN_ERRORS`` the line is its
    message. Any other error can be wrong input or a fault of the code, so its traceback
    is printed first, and the line names its class.

    Ctrl-C (SIGINT), a plain kill (SIGTERM) and a closed terminal (SIGHUP) stop the
    command: the model requests that are in flight finish, and their answers are stored
    and recorded before the command ends with the exit code 2. The model backend sends
    no request and no new attempt after the signal (``STOP_REQUESTED``). The signals are
    handled only when this function runs in the main thread, and only while it runs.

    The standard output is set to line buffering. Python keeps the output of a pipe or
    a file in a buffer otherwise: the cost estimate of a paid command would be readable
    only when the command ends, and after the error line of the other stream.
    """
    args = build_parser().parse_args(argv)
    _configure_logging()
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    STOP_REQUESTED.clear()
    in_main_thread = threading.current_thread() is threading.main_thread()
    if in_main_thread:
        # A stop signal of an earlier command in this process must not reach a call of this one.
        stopsignals.reset()
    earlier_handlers = {number: signal.signal(number, _interrupt) for number in STOP_SIGNALS} if in_main_thread else {}
    try:
        return args.handler(args)
    except KeyboardInterrupt as stop:
        cause = str(stop) or "Ctrl-C"
        print(
            f"error: the command was interrupted ({cause}). Every answer that was received is stored and "
            "recorded in the cost ledger.",
            file=sys.stderr,
        )
        return EXIT_ERROR
    except KNOWN_ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as error:
        traceback.print_exc()
        kind = type(error).__name__
        print(f"error: the command stopped with {kind}; the traceback above says where", file=sys.stderr)
        return EXIT_ERROR
    finally:
        for number, handler in earlier_handlers.items():
            signal.signal(number, handler)


def _interrupt(number: int, frame: object) -> None:
    """Handle a stop signal: tell the model backend to send nothing more, and interrupt the command.

    The interruption is a ``KeyboardInterrupt`` that names the cause: Ctrl-C, or the name
    of the signal. While paid requests are in flight (a model call in the main thread,
    ``finish_in_flight``, or the thread pool of the pipeline, ``pipeline._generate_all``),
    the signal is held back (``stopsignals.held_stops``): the code that waits raises the
    interruption, with the cause of the first signal, when the requests have ended. No
    number of further signals ends that wait; Ctrl-\\ (SIGQUIT) or ``kill -KILL`` ends the
    process at once, and the answers in flight are lost but can still be billed.

    A signal that comes while this handler runs (a burst of signals) is only counted: the
    handler raises nothing inside the lock of ``STOP_REQUESTED``, which would stay taken. A
    command is interrupted once: after that, signals are only counted, so that they do not cut
    short the end of the command (its cleanup, its error line and its exit code 2).
    """
    cause = "Ctrl-C" if number == signal.SIGINT else signal.Signals(number).name
    if not stopsignals.enter_handler():
        return
    try:
        STOP_REQUESTED.set()
    finally:
        stopsignals.leave_handler()
    if stopsignals.hold_back(cause):
        return
    stopsignals.mark_raised()
    raise KeyboardInterrupt(cause)


if __name__ == "__main__":
    sys.exit(main())
