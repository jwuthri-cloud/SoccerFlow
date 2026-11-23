#!/usr/bin/env python3
"""Run the offline tests in several pytest processes at once, with the results of one process.

Usage, from the top folder of the repository, with the Python that has pytest:

    python tools/run_tests.py                          # default run: all but the slow tests
    python tools/run_tests.py -m "slow or not slow"    # full run: every test
    python tools/run_tests.py -k guard -x tests/test_guard.py
    python tools/run_tests.py -j 4                     # four worker processes (default: one per core)

What it does:

1. One ``python -m pytest --collect-only`` with the given options lists the test ids, as the
   run in one process would collect them.
2. The ids are split over the workers by time. The time of each test comes from the last run
   of this runner in this checkout (``.pytest_cache/run-tests-times.json``, ignored by git).
   A test without a stored time counts as the median stored time, or as 1 when there is none:
   the first run in a new checkout is balanced by test count, and is correct but slower. A
   file whose time is more than half the share of one worker is split, in collection order,
   into parts; tests of one file keep their order.
3. Each worker is a plain ``python -m pytest`` with its ids, its own base temporary folder,
   its own cache folder and its own JUnit XML file, in its own process group.
4. The run fails (exit code not zero) when a test fails or has an error, when a worker ends
   in another way than pytest's normal end, when a worker gives no output for
   ``--silence-limit`` seconds (it is killed, with every process of its group), or when the
   ids that the workers ran differ from the collected ids (a test missing or run twice).

Exit codes: 0 every test passed or was skipped; 1 a test failed or had an error; 2 the run
could not be trusted (collection error, a worker that crashed, hung or was killed, an id
check that failed, a wrong option); 130 interrupted (Ctrl-C). The last line of a run with exit
code 2 starts with ``error:``.

Only the standard library is imported. The options passed on to pytest are ``-m``, ``-k``,
``-x`` and test paths (files, folders, test ids); any other option is refused.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: Times of the tests of the last run, by test id. Read only to balance the workers.
TIMES_FILE = Path(".pytest_cache") / "run-tests-times.json"
TIMES_FORMAT = 1
#: A worker that writes nothing for this many seconds is killed (the longest test takes about 5 s).
DEFAULT_SILENCE_LIMIT_S = 120.0
#: How often the runner looks at the workers.
POLL_S = 0.1
#: How long a worker gets to end after SIGTERM before its group gets SIGKILL.
TERM_GRACE_S = 5.0
#: Exit codes of the runner.
EXIT_PASSED, EXIT_FAILED, EXIT_UNTRUSTED, EXIT_INTERRUPTED = 0, 1, 2, 130
#: Signals that stop the run: Ctrl-C, and SIGTERM and SIGHUP as Ctrl-C.
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
#: Exit codes of pytest that mean a normal end: all passed, or some tests failed.
PYTEST_NORMAL_ENDS = (0, 1)

_COLLECTED = re.compile(r"^(?:(\d+)/)?(\d+) tests? collected(?: \((\d+) deselected\))? in ")
_WORKER_COLLECTED = re.compile(r"^(?:collecting \.\.\. )?collected (\d+) items?\s*$", re.MULTILINE)


class RunnerError(Exception):
    """The runner cannot go on; the message says why."""


@dataclass(frozen=True)
class Options:
    workers: int
    silence_limit_s: float
    marker: str | None
    keyword: str | None
    exit_first: bool
    paths: tuple[str, ...]
    results_file: Path | None = None

    def selection(self) -> list[str]:
        """Return the pytest options that select tests, as the user gave them (no paths)."""
        args: list[str] = []
        if self.marker is not None:
            args += ["-m", self.marker]
        if self.keyword is not None:
            args += ["-k", self.keyword]
        return args


@dataclass(frozen=True)
class Collection:
    ids: tuple[str, ...]
    deselected: int


@dataclass(frozen=True)
class Result:
    test_id: str
    outcome: str  # "passed", "failed", "error" or "skipped"
    seconds: float
    text: str


@dataclass
class Worker:
    number: int
    ids: list[str]
    folder: Path
    process: subprocess.Popen[bytes] | None = None
    started: float = 0.0
    log_size: int = 0
    last_output: float = 0.0
    exit_code: int | None = None
    silent_for_too_long: bool = False
    stopped_by_runner: bool = False
    seconds: float = 0.0
    results: list[Result] = field(default_factory=list)
    problem: str | None = None

    @property
    def log(self) -> Path:
        return self.folder / "output.txt"

    @property
    def junit(self) -> Path:
        return self.folder / "junit.xml"

    def where(self) -> str:
        """Name the worker and its test files."""
        files = list(dict.fromkeys(file_of(test_id) for test_id in self.ids))
        more = {1: "", 2: " and 1 more file"}.get(len(files), f" and {len(files) - 1} more files")
        return f"worker {self.number} ({files[0]}{more})"


def parse_options(argv: Sequence[str]) -> Options:
    parser = argparse.ArgumentParser(
        prog="tools/run_tests.py",
        description="Run the offline tests in several pytest processes; same tests and results as one process.",
        allow_abbrev=False,
    )
    parser.add_argument("-j", "--workers", type=int, default=os.cpu_count() or 1, help="worker processes (default: one per core)")
    parser.add_argument(
        "--silence-limit",
        type=float,
        default=DEFAULT_SILENCE_LIMIT_S,
        help=f"kill a worker that writes nothing for this many seconds (default {DEFAULT_SILENCE_LIMIT_S:g})",
    )
    parser.add_argument("-m", dest="marker", help="marker expression, passed to pytest")
    parser.add_argument("-k", dest="keyword", help="keyword expression, passed to pytest")
    parser.add_argument("-x", "--exitfirst", dest="exit_first", action="store_true", help="stop after the first failed test")
    parser.add_argument(
        "--results", type=Path, metavar="FILE", help="write one line per test that ran: its outcome and its id, sorted by id"
    )
    parser.add_argument("paths", nargs="*", help="test files, folders or test ids (default: the test paths of pyproject.toml)")
    namespace = parser.parse_args(argv)
    if namespace.workers < 1:
        parser.error("-j must be 1 or more")
    if namespace.silence_limit <= 0:
        parser.error("--silence-limit must be more than 0")
    for path in namespace.paths:
        if path.startswith("-"):
            parser.error(f"unknown option {path}: the runner passes only -m, -k, -x and test paths to pytest")
    return Options(
        workers=namespace.workers,
        silence_limit_s=namespace.silence_limit,
        marker=namespace.marker,
        keyword=namespace.keyword,
        exit_first=namespace.exit_first,
        paths=tuple(namespace.paths),
        results_file=namespace.results,
    )


def file_of(test_id: str) -> str:
    """Return the file part of a test id."""
    return test_id.partition("::")[0]


def junit_key(test_id: str) -> tuple[str, str]:
    """Return the (classname, name) that pytest writes into its JUnit XML for a test id.

    The same rule as ``_pytest.junitxml.mangle_test_address`` of pytest 9: the file path
    becomes a dotted module name, the classes join it, and the parameters stay on the name.
    A test id whose key is not found in a worker's file fails the id check, so a change of
    this rule in pytest stops the run instead of losing tests.
    """
    path, separator, tail = test_id.partition("::")
    if not separator:
        raise RunnerError(f"not a test id: {test_id}")
    name_part, bracket, params = tail.partition("[")
    names = [re.sub(r"\.py$", "", path.replace("/", ".")), *name_part.split("::")]
    if bracket:
        names[-1] += "[" + params
    return ".".join(names[:-1]), names[-1]


def pytest_command(*args: str) -> list[str]:
    return [sys.executable, "-m", "pytest", *args]


def private_folders(folder: Path) -> list[str]:
    """Return the pytest options that keep a process out of the folders of other processes."""
    return [f"--basetemp={folder / 'tmp'}", "-o", f"cache_dir={folder / 'cache'}"]


def collect(options: Options, root: Path, run_folder: Path) -> Collection:
    """List the test ids of the run as one pytest process collects them."""
    folder = run_folder / "collect"
    folder.mkdir()
    command = pytest_command(
        "--collect-only", "--verbosity=-1", *private_folders(folder), *options.selection(), *options.paths
    )
    done = subprocess.run(command, cwd=root, capture_output=True, encoding="utf-8", errors="replace", check=False)
    if done.returncode != 0:
        output = (done.stdout + done.stderr).rstrip()
        raise RunnerError(f"the collection of the tests ended with exit code {done.returncode}:\n{output}")
    lines = done.stdout.splitlines()
    try:
        end = lines.index("")
    except ValueError:
        raise RunnerError(f"the output of the collection has no end:\n{done.stdout}") from None
    ids = tuple(lines[:end])
    for test_id in ids:
        if "::" not in test_id or not (root / file_of(test_id)).is_file():
            raise RunnerError(f"the collection printed a line that is not a test id: {test_id!r}")
    if len(set(ids)) != len(ids):
        raise RunnerError("the collection gave a test id two times")
    for line in lines[end:]:
        match = _COLLECTED.match(line)
        if match:
            selected = int(match.group(1) or match.group(2))
            if selected != len(ids):
                raise RunnerError(f"the collection says {selected} tests but listed {len(ids)}")
            return Collection(ids=ids, deselected=int(match.group(3) or 0))
    raise RunnerError(f"the output of the collection has no count line:\n{done.stdout}")


def load_times(root: Path) -> tuple[dict[str, float], str]:
    """Return the stored seconds by test id, and one line that says where they come from."""
    path = root / TIMES_FILE
    if not path.exists():
        return {}, f"no stored test times ({TIMES_FILE}): workers balanced by test count"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != TIMES_FORMAT:
            raise ValueError(f"format {data.get('format')!r}, expected {TIMES_FORMAT}")
        seconds = {str(key): float(value) for key, value in data["seconds"].items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        return {}, f"stored test times not readable ({TIMES_FILE}: {error}): workers balanced by test count"
    return seconds, f"test times of the last run ({TIMES_FILE})"


def save_times(root: Path, results: Sequence[Result]) -> None:
    """Store the time of each test that ran, keeping the stored times of the other tests."""
    path = root / TIMES_FILE
    seconds, _ = load_times(root)
    seconds.update({result.test_id: round(result.seconds, 4) for result in results})
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}")
    partial.write_text(json.dumps({"format": TIMES_FORMAT, "seconds": dict(sorted(seconds.items()))}), encoding="utf-8")
    os.replace(partial, path)


def plan(ids: Sequence[str], times: dict[str, float], workers: int) -> list[list[str]]:
    """Split the ids over at most `workers` lists of about equal time.

    A file whose time is more than half of one worker's share is cut into parts of at most
    that size, in collection order. The parts are given out longest first, each to the
    worker with the least time so far. Each worker runs its ids in collection order.
    """
    known = [times[test_id] for test_id in ids if test_id in times]
    default = statistics.median(known) if known else 1.0
    weight = {test_id: times.get(test_id, default) for test_id in ids}
    count = min(workers, len(ids))
    if count == 0:
        return []
    part_limit = sum(weight.values()) / count / 2
    by_file: dict[str, list[str]] = {}
    for test_id in ids:
        by_file.setdefault(file_of(test_id), []).append(test_id)
    parts: list[list[str]] = []
    for file_ids in by_file.values():
        total = sum(weight[test_id] for test_id in file_ids)
        pieces = max(1, min(len(file_ids), math.ceil(total / part_limit))) if part_limit > 0 else 1
        parts.extend(_cut(file_ids, weight, pieces))
    position = {test_id: index for index, test_id in enumerate(ids)}
    loads = [0.0] * count
    bins: list[list[str]] = [[] for _ in range(count)]
    for part in sorted(parts, key=lambda part: (-sum(weight[test_id] for test_id in part), position[part[0]])):
        target = min(range(count), key=lambda index: (loads[index], index))
        bins[target].extend(part)
        loads[target] += sum(weight[test_id] for test_id in part)
    return [sorted(ids_of_bin, key=position.__getitem__) for ids_of_bin in bins if ids_of_bin]


def _cut(file_ids: list[str], weight: dict[str, float], pieces: int) -> list[list[str]]:
    """Cut the ids of one file, in order, into `pieces` runs of about equal weight."""
    if pieces == 1:
        return [file_ids]
    total = sum(weight[test_id] for test_id in file_ids)
    runs: list[list[str]] = [[]]
    done = 0.0
    for index, test_id in enumerate(file_ids):
        left = len(file_ids) - index
        # A new run starts when this one has its share, as long as every later run gets a test.
        if runs[-1] and len(runs) < pieces and (done >= total * len(runs) / pieces or left <= pieces - len(runs)):
            runs.append([])
        runs[-1].append(test_id)
        done += weight[test_id]
    return runs


def start(worker: Worker, options: Options, root: Path) -> None:
    worker.folder.mkdir()
    command = pytest_command(
        "--verbosity=1",
        "-o",
        "console_output_style=classic",
        "-o",
        f"faulthandler_timeout={options.silence_limit_s / 2:g}",
        f"--junitxml={worker.junit}",
        "-o",
        "junit_family=xunit1",
        "-o",
        "junit_logging=all",
        "-o",
        "junit_log_passing_tests=false",
        *private_folders(worker.folder),
        *options.selection(),
        *(["-x"] if options.exit_first else []),
        *worker.ids,
    )
    with worker.log.open("wb") as log:
        worker.process = subprocess.Popen(
            command, cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    worker.started = worker.last_output = time.monotonic()


def stop(workers: Sequence[Worker]) -> None:
    """End the process groups of running workers: SIGTERM to all, then SIGKILL after the grace time.

    The group, not only the worker: its children (git, the guard, a hook runner) can outlive it.
    """
    running = [worker for worker in workers if worker.process is not None and worker.process.returncode is None]
    for worker in running:
        assert worker.process is not None
        _signal_group(worker.process, signal.SIGTERM)
    deadline = time.monotonic() + TERM_GRACE_S
    for worker in running:
        assert worker.process is not None
        try:
            worker.process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
        _signal_group(worker.process, signal.SIGKILL)
        if worker.process.poll() is None:
            worker.process.kill()
        worker.process.wait()
        worker.exit_code = worker.process.returncode
        worker.seconds = time.monotonic() - worker.started


def _signal_group(process: subprocess.Popen[bytes], number: int) -> None:
    """Send a signal to the process group of a worker."""
    try:
        os.killpg(process.pid, number)
    except ProcessLookupError:
        pass  # No process is left in the group.
    except PermissionError:
        # macOS answers so when the group still exists but every process in it is already
        # ending (a worker that ends at the moment of a Ctrl-C). `stop` kills the worker itself after.
        pass


def last_test_started(log: Path) -> str:
    """Return the last test id that a worker's output names, or a note that it names none."""
    for line in reversed(log.read_text(encoding="utf-8", errors="replace").splitlines()):
        if "::" in line:
            return line.split(" ", 1)[0]
    return "no test started"


def watch(workers: list[Worker], options: Options) -> None:
    """Wait for every worker; kill silent workers; with -x, stop the others after a failure."""
    running = list(workers)
    while running:
        now = time.monotonic()
        for worker in list(running):
            process = worker.process
            assert process is not None
            code = process.poll()
            if code is not None:
                worker.exit_code = code
                worker.seconds = now - worker.started
                running.remove(worker)
                if options.exit_first and code != 0:
                    for other in running:
                        other.stopped_by_runner = True
                    stop(running)
                    running.clear()
                    break
                continue
            size = worker.log.stat().st_size
            if size != worker.log_size:
                worker.log_size, worker.last_output = size, now
            elif now - worker.last_output > options.silence_limit_s:
                worker.silent_for_too_long = True
                stop([worker])
                running.remove(worker)
        time.sleep(POLL_S)


def read_results(worker: Worker, keys: dict[tuple[str, str], str]) -> list[Result]:
    """Read the results of a worker from its JUnit XML file."""
    root = ET.parse(worker.junit).getroot()
    results = []
    for case in root.iter("testcase"):
        key = (case.get("classname", ""), case.get("name", ""))
        test_id = keys.get(key)
        if test_id is None:
            test_id = f"<not collected: classname={key[0]!r} name={key[1]!r}>"
        failures = case.findall("failure")
        errors = case.findall("error")
        skipped = case.findall("skipped")
        if errors:
            outcome = "error"
        elif failures:
            outcome = "failed"
        elif skipped:
            outcome = "skipped"
        else:
            outcome = "passed"
        text = ""
        if outcome in ("failed", "error"):
            parts = [element.text or element.get("message", "") for element in (*failures, *errors)]
            for name in ("system-out", "system-err"):
                parts += [element.text for element in case.findall(name) if element.text]
            text = "\n".join(part.rstrip() for part in parts if part)
        results.append(Result(test_id, outcome, float(case.get("time", "0")), text))
    return results


def check_worker(worker: Worker, keys: dict[tuple[str, str], str], options: Options) -> None:
    """Read the results of a finished worker, or set its problem."""
    if worker.silent_for_too_long:
        silence_limit_s = options.silence_limit_s
        last = last_test_started(worker.log)
        worker.problem = (
            f"gave no output for {silence_limit_s:g} s and was killed, in {file_of(last)} (last test started: {last})"
        )
        return
    if worker.stopped_by_runner:
        return
    if worker.exit_code not in PYTEST_NORMAL_ENDS:
        worker.problem = f"ended with exit code {worker.exit_code}, not pytest's normal end (0 or 1)"
        return
    if not worker.junit.is_file():
        worker.problem = "ended without its JUnit XML file"
        return
    output = worker.log.read_text(encoding="utf-8", errors="replace")
    counts = _WORKER_COLLECTED.findall(output)
    if counts != [str(len(worker.ids))]:
        worker.problem = f"was given {len(worker.ids)} test ids but collected {counts or 'none'}"
        return
    worker.results = read_results(worker, keys)
    if worker.exit_code == 1 and not any(result.outcome in ("failed", "error") for result in worker.results):
        worker.problem = "ended with exit code 1 but its JUnit XML file names no failed test"
    elif worker.exit_code == 0 and any(result.outcome in ("failed", "error") for result in worker.results):
        worker.problem = "ended with exit code 0 but its JUnit XML file names a failed test"
    elif len(worker.results) != len(worker.ids) and not (options.exit_first and worker.exit_code == 1):
        worker.problem = f"ran {len(worker.results)} tests of its {len(worker.ids)}"


def id_problems(collected: Sequence[str], ran: Sequence[str], complete: bool) -> list[str]:
    """Compare the ids that ran with the collected ids; `complete` asks for every id."""
    problems = []
    seen: dict[str, int] = {}
    for test_id in ran:
        seen[test_id] = seen.get(test_id, 0) + 1
    twice = sorted(test_id for test_id, number in seen.items() if number > 1)
    unknown = sorted(set(seen) - set(collected))
    missing = [test_id for test_id in collected if test_id not in seen]
    if twice:
        problems.append(f"{len(twice)} tests ran more than once, for example {twice[:3]}")
    if unknown:
        problems.append(f"{len(unknown)} tests ran that the collection does not list, for example {unknown[:3]}")
    if complete and missing:
        problems.append(f"{len(missing)} collected tests did not run, for example {missing[:3]}")
    return problems


def report(
    workers: list[Worker], collection: Collection, wall_s: float, options: Options
) -> int:
    """Print the failures and the summary line; return the exit code."""
    results = [result for worker in workers for result in worker.results]
    for result in results:
        if result.outcome in ("failed", "error"):
            title = f" {result.outcome.upper()} {result.test_id} "
            print(f"\n{title:_^100}\n{result.text}")
    problems = []
    for worker in workers:
        if worker.problem is not None:
            problems.append(f"{worker.where()} {worker.problem}")
            print(f"\n{f' OUTPUT OF WORKER {worker.number} ':=^100}")
            print(worker.log.read_text(encoding="utf-8", errors="replace").rstrip())
    trusted = all(worker.problem is None for worker in workers)
    # With -x a failed test ends its own worker early, and the runner stops the others: the
    # tests after it do not run. Whether another worker had already ended does not matter.
    stopped = options.exit_first and any(result.outcome in ("failed", "error") for result in results)
    if trusted:
        problems += id_problems(collection.ids, [result.test_id for result in results], complete=not stopped)
    counts = {outcome: 0 for outcome in ("passed", "failed", "error", "skipped")}
    for result in results:
        counts[result.outcome] += 1
    not_run = len(collection.ids) - len(results)
    slowest = max(workers, key=lambda worker: worker.seconds)
    fastest = min(worker.seconds for worker in workers)
    print(f"\nworkers took {fastest:.1f} to {slowest.seconds:.1f} s; the slowest: {slowest.where()}")
    summary = (
        f"{counts['passed']} passed, {counts['failed']} failed, {counts['error']} errors, "
        f"{counts['skipped']} skipped, {collection.deselected} deselected"
        + (f", {not_run} not run" if not_run else "")
        + f" in {wall_s:.1f} s ({len(workers)} workers)"
    )
    print(f"\n{f' {summary} ':=^100}")
    if stopped:
        print("stopped after the first failed test (-x)")
    for problem in problems:
        print(f"error: {problem}")
    if problems:
        return EXIT_UNTRUSTED
    if counts["failed"] or counts["error"]:
        return EXIT_FAILED
    return EXIT_PASSED


def run(options: Options, root: Path = ROOT) -> int:
    """Run the tests of the project in `root` (the folder with its pytest settings)."""
    begin = time.monotonic()
    run_folder = Path(tempfile.mkdtemp(prefix="pitchvlm-tests-"))
    workers: list[Worker] = []
    keep_folder = False
    try:
        collection = collect(options, root, run_folder)
        collected_s = time.monotonic() - begin
        keys: dict[tuple[str, str], str] = {}
        for test_id in collection.ids:
            key = junit_key(test_id)
            if key in keys:
                raise RunnerError(f"two test ids have the same JUnit name: {keys[key]} and {test_id}")
            keys[key] = test_id
        times, source = load_times(root)
        bins = plan(collection.ids, times, options.workers)
        print(f"{len(collection.ids)} tests collected in {collected_s:.1f} s, run in {len(bins)} workers; {source}", flush=True)
        workers = [Worker(number, ids, run_folder / f"worker-{number}") for number, ids in enumerate(bins, start=1)]
        for worker in workers:
            start(worker, options, root)
        watch(workers, options)
        for worker in workers:
            check_worker(worker, keys, options)
        code = report(workers, collection, time.monotonic() - begin, options)
        if all(worker.problem is None for worker in workers):
            results = [result for worker in workers for result in worker.results]
            save_times(root, results)
            if options.results_file is not None:
                lines = sorted(f"{result.outcome} {result.test_id}\n" for result in results)
                options.results_file.write_text("".join(lines), encoding="utf-8")
        if code == EXIT_UNTRUSTED:
            keep_folder = True
            print(f"the output and the JUnit XML file of each worker are kept in {run_folder}")
        return code
    except RunnerError as problem:
        keep_folder = True
        print(f"error: {problem}")
        return EXIT_UNTRUSTED
    except KeyboardInterrupt:
        # A second Ctrl-C must not stop the runner before its workers are stopped.
        earlier = {number: signal.signal(number, signal.SIG_IGN) for number in STOP_SIGNALS}
        try:
            stop(workers)
        finally:
            for number, handler in earlier.items():
                signal.signal(number, handler)
        print("\ninterrupted: every worker was stopped", flush=True)
        return EXIT_INTERRUPTED
    finally:
        stop(workers)
        if not keep_folder:
            shutil.rmtree(run_folder, ignore_errors=True)


def _interrupt(number: int, frame: object) -> None:
    raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_options(sys.argv[1:] if argv is None else argv)
    # SIGTERM and SIGHUP stop the workers as Ctrl-C does.
    for number in STOP_SIGNALS:
        if number != signal.SIGINT:
            signal.signal(number, _interrupt)
    return run(options)


if __name__ == "__main__":
    sys.exit(main())
