"""Tests of the parallel test runner, tools/run_tests.py.

The tests of the parts (options, JUnit names, plan, id check, stored times) run in this
process. The tests of whole runs start the runner on a small project in a temporary folder,
with its own pytest settings, so they do not depend on the tests of this repository.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "tools" / "run_tests.py"
#: How long a test of the small project waits "for ever". Short enough that a process left
#: behind by an interrupted run of these tests ends by itself.
LONG_S = 300


def load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_tests_under_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


# --- Options -------------------------------------------------------------------------------


def test_options_pass_the_tier_the_keyword_the_first_failure_and_the_paths() -> None:
    options = runner.parse_options(["-m", "slow or not slow", "-k", "guard", "-x", "-j", "3", "tests/test_guard.py"])

    assert options.selection() == ["-m", "slow or not slow", "-k", "guard"]
    assert (options.exit_first, options.workers, options.paths) == (True, 3, ("tests/test_guard.py",))


def test_options_default_to_the_tier_of_the_settings_and_one_worker_per_core() -> None:
    options = runner.parse_options([])

    assert options.selection() == []
    assert options.workers == (os.cpu_count() or 1)
    assert (options.exit_first, options.paths, options.results_file) == (False, (), None)


@pytest.mark.parametrize("argv", [["--tb=short"], ["-q"], ["-j", "0"], ["--silence-limit", "0"], ["-n", "4"]])
def test_options_refuse_what_the_runner_does_not_pass_on(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        runner.parse_options(argv)

    assert stop.value.code == 2
    assert "error:" in capsys.readouterr().err


# --- JUnit names ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("test_id", "key"),
    [
        ("tests/test_a.py::test_one", ("tests.test_a", "test_one")),
        ("tests/test_a.py::Case::test_one", ("tests.test_a.Case", "test_one")),
        ("tests/test_a.py::Outer::Inner::test_one", ("tests.test_a.Outer.Inner", "test_one")),
        ("tests/test_a.py::test_one[a.b::c/d]", ("tests.test_a", "test_one[a.b::c/d]")),
        ("test_a.py::test_one[x]", ("test_a", "test_one[x]")),
    ],
)
def test_junit_key_is_the_name_that_pytest_writes(test_id: str, key: tuple[str, str]) -> None:
    assert runner.junit_key(test_id) == key


# --- Plan ----------------------------------------------------------------------------------


def make_ids(sizes: dict[str, int]) -> list[str]:
    return [f"tests/{name}.py::test_{number}" for name, size in sizes.items() for number in range(size)]


def test_plan_gives_each_test_to_one_worker_in_collection_order() -> None:
    ids = make_ids({"test_a": 7, "test_b": 40, "test_c": 3, "test_d": 15})
    times = {test_id: random.Random(test_id).uniform(0.01, 2.0) for test_id in ids}

    bins = runner.plan(ids, times, 4)

    assert sorted(test_id for ids_of_bin in bins for test_id in ids_of_bin) == sorted(ids)
    for ids_of_bin in bins:
        assert ids_of_bin == [test_id for test_id in ids if test_id in set(ids_of_bin)]
    assert bins == runner.plan(ids, times, 4)


def test_plan_splits_a_long_file_and_balances_the_workers() -> None:
    ids = make_ids({"test_long": 100, "test_short": 50, "test_tiny": 10})
    times = {test_id: (1.0 if "test_long" in test_id else 0.1) for test_id in ids}

    bins = runner.plan(ids, times, 4)

    loads = [sum(times[test_id] for test_id in ids_of_bin) for ids_of_bin in bins]
    assert len(bins) == 4
    assert sum(1 for ids_of_bin in bins if any("test_long" in test_id for test_id in ids_of_bin)) >= 3
    assert max(loads) <= 1.25 * sum(loads) / 4


def test_plan_without_stored_times_balances_by_test_count() -> None:
    ids = make_ids({"test_a": 30, "test_b": 30, "test_c": 30, "test_d": 30})

    bins = runner.plan(ids, {}, 4)

    assert sorted(len(ids_of_bin) for ids_of_bin in bins) == [30, 30, 30, 30]


def test_plan_counts_a_test_without_a_stored_time_as_the_median() -> None:
    ids = make_ids({"test_old": 4, "test_new": 4})
    times = {test_id: 3.0 for test_id in ids if "test_old" in test_id}

    bins = runner.plan(ids, times, 2)

    assert sorted(len(ids_of_bin) for ids_of_bin in bins) == [4, 4]


def test_plan_starts_no_worker_without_tests() -> None:
    ids = make_ids({"test_a": 2})

    assert runner.plan(ids, {}, 8) == [[ids[0]], [ids[1]]]


# --- Id check ------------------------------------------------------------------------------


def test_id_check_names_a_test_run_twice_a_test_not_collected_and_a_missing_test() -> None:
    collected = ["a.py::t1", "a.py::t2", "b.py::t3"]

    assert runner.id_problems(collected, ["a.py::t2", "b.py::t3", "a.py::t1"], complete=True) == []
    problems = runner.id_problems(collected, ["a.py::t1", "a.py::t1", "c.py::t9"], complete=True)
    assert problems == [
        "1 tests ran more than once, for example ['a.py::t1']",
        "1 tests ran that the collection does not list, for example ['c.py::t9']",
        "2 collected tests did not run, for example ['a.py::t2', 'b.py::t3']",
    ]
    # After -x stopped the run, a test that did not run is no problem.
    assert runner.id_problems(collected, ["a.py::t1"], complete=False) == []


# --- Stored times --------------------------------------------------------------------------


def test_stored_times_are_kept_for_tests_that_did_not_run(tmp_path: Path) -> None:
    seconds, source = runner.load_times(tmp_path)
    assert seconds == {}
    assert source.startswith("no stored test times")

    runner.save_times(tmp_path, [runner.Result("a.py::t1", "passed", 1.5, ""), runner.Result("a.py::t2", "failed", 0.25, "")])
    runner.save_times(tmp_path, [runner.Result("a.py::t1", "passed", 2.0, "")])

    assert runner.load_times(tmp_path)[0] == {"a.py::t1": 2.0, "a.py::t2": 0.25}
    assert [path.name for path in (tmp_path / runner.TIMES_FILE).parent.iterdir()] == [runner.TIMES_FILE.name]


@pytest.mark.parametrize("content", ["not json", json.dumps({"format": 99, "seconds": {}}), json.dumps({"format": 1})])
def test_unreadable_stored_times_are_named_and_not_used(tmp_path: Path, content: str) -> None:
    path = tmp_path / runner.TIMES_FILE
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")

    seconds, source = runner.load_times(tmp_path)

    assert seconds == {}
    assert source.startswith("stored test times not readable")


# --- Whole runs on a small project ---------------------------------------------------------

SETTINGS = """\
[pytest]
addopts = -q -m "not slow"
markers =
    slow: left out of the default run
"""

PASSING = """\
import pytest

def test_one():
    assert 1 + 1 == 2

def test_two():
    pass

@pytest.mark.skip(reason="not here")
def test_skipped():
    pass

@pytest.mark.slow
def test_slow():
    pass

@pytest.mark.parametrize("value", [1, 2, 3])
def test_values(value):
    assert value > 0

class TestGroup:
    def test_in_class(self):
        pass
"""

FAILING = """\
def test_fails():
    print("words printed by the failing test")
    assert 2 + 2 == 5

def test_passes():
    pass
"""


def make_project(folder: Path, files: dict[str, str]) -> Path:
    folder.mkdir(exist_ok=True)
    (folder / "pytest.ini").write_text(SETTINGS, encoding="utf-8")
    for name, text in files.items():
        (folder / name).write_text(textwrap.dedent(text).replace("LONG_S", str(LONG_S)), encoding="utf-8")
    return folder


@pytest.fixture
def run_in_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Callable[[Path, list[str]], tuple[int, str]]:
    """Run the runner in this process; the folder that it keeps after a failed run goes to the test folder."""
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(runs))

    def run(root: Path, argv: list[str]) -> tuple[int, str]:
        code = runner.run(runner.parse_options(argv), root=root)
        return code, capsys.readouterr().out

    return run


def summary_line(output: str) -> str:
    return [line for line in output.splitlines() if " in " in line and " workers)" in line][-1]


def test_run_shows_each_failure_with_its_file_and_the_totals(tmp_path: Path, run_in_process: Callable[[Path, list[str]], tuple[int, str]]) -> None:
    root = make_project(tmp_path / "project", {"test_pass.py": PASSING, "test_fail.py": FAILING})
    results = tmp_path / "results.txt"

    code, output = run_in_process(root, ["-j", "3", "--results", str(results)])

    assert code == runner.EXIT_FAILED
    assert "FAILED test_fail.py::test_fails" in output
    assert "assert (2 + 2) == 5" in output
    assert "words printed by the failing test" in output
    assert "7 passed, 1 failed, 0 errors, 1 skipped, 1 deselected in " in summary_line(output)
    assert "error:" not in output
    assert results.read_text(encoding="utf-8").splitlines() == [
        "failed test_fail.py::test_fails",
        "passed test_fail.py::test_passes",
        "passed test_pass.py::TestGroup::test_in_class",
        "passed test_pass.py::test_one",
        "passed test_pass.py::test_two",
        "passed test_pass.py::test_values[1]",
        "passed test_pass.py::test_values[2]",
        "passed test_pass.py::test_values[3]",
        "skipped test_pass.py::test_skipped",
    ]
    # The times of this run are stored for the next plan.
    assert len(runner.load_times(root)[0]) == 9


def test_run_passes_the_tier_the_keyword_and_the_paths(tmp_path: Path, run_in_process: Callable[[Path, list[str]], tuple[int, str]]) -> None:
    root = make_project(tmp_path / "project", {"test_pass.py": PASSING, "test_fail.py": FAILING})

    code, output = run_in_process(root, ["-j", "2", "-m", "slow or not slow", "-k", "slow or values", "test_pass.py"])

    assert code == runner.EXIT_PASSED
    assert "4 passed, 0 failed, 0 errors, 0 skipped, 4 deselected in " in summary_line(output)


def test_a_worker_that_crashes_makes_the_run_untrusted(tmp_path: Path, run_in_process: Callable[[Path, list[str]], tuple[int, str]]) -> None:
    crash = "import os, signal\n\ndef test_killed():\n    os.kill(os.getpid(), signal.SIGKILL)\n"
    root = make_project(tmp_path / "project", {"test_pass.py": PASSING, "test_crash.py": crash})

    code, output = run_in_process(root, ["-j", "2"])

    assert code == runner.EXIT_UNTRUSTED
    errors = [line for line in output.splitlines() if line.startswith("error:")]
    assert len(errors) == 1
    assert "(test_crash.py" in errors[0]
    assert errors[0].endswith("ended with exit code -9, not pytest's normal end (0 or 1)")
    assert output.rstrip().splitlines()[-1].startswith("the output and the JUnit XML file of each worker are kept in")


def test_a_silent_worker_is_killed_and_named_with_its_file(tmp_path: Path, run_in_process: Callable[[Path, list[str]], tuple[int, str]]) -> None:
    hang = "import time\n\ndef test_first():\n    pass\n\ndef test_sleeps_for_ever():\n    time.sleep(LONG_S)\n"
    root = make_project(tmp_path / "project", {"test_pass.py": PASSING, "test_hang.py": hang})

    # A limit far above the start of pytest: with 1 s, a loaded machine (load average 30 to 60) took
    # more than 1 s to start the worker of test_pass.py, which was killed too (2 of 20 runs in one
    # process, 11 of 20 with the runner). The healthy worker must survive the kill of the silent one.
    started = time.monotonic()
    code, output = run_in_process(root, ["-j", "2", "--silence-limit", "10"])

    assert time.monotonic() - started < 120
    assert code == runner.EXIT_UNTRUSTED
    errors = [line for line in output.splitlines() if line.startswith("error:")]
    assert len(errors) == 1, output
    assert errors[0].endswith(
        "gave no output for 10 s and was killed, in test_hang.py (last test started: test_hang.py::test_sleeps_for_ever)"
    )


def test_a_test_file_that_cannot_be_read_stops_the_run(tmp_path: Path, run_in_process: Callable[[Path, list[str]], tuple[int, str]]) -> None:
    root = make_project(tmp_path / "project", {"test_pass.py": PASSING, "test_broken.py": "def test_x(:\n    pass\n"})

    code, output = run_in_process(root, [])

    assert code == runner.EXIT_UNTRUSTED
    assert "error: the collection of the tests ended with exit code 2" in output
    assert "SyntaxError" in output


def test_first_failure_stops_the_other_workers(tmp_path: Path, run_in_process: Callable[[Path, list[str]], tuple[int, str]]) -> None:
    slow = "import time\n\ndef test_waits():\n    time.sleep(LONG_S)\n"
    root = make_project(tmp_path / "project", {"test_fail.py": FAILING, "test_wait.py": slow})

    started = time.monotonic()
    code, output = run_in_process(root, ["-j", "2", "-x"])

    assert time.monotonic() - started < 120
    assert code == runner.EXIT_FAILED
    assert "FAILED test_fail.py::test_fails" in output
    assert "stopped after the first failed test (-x)" in output
    assert " not run in " in summary_line(output)


def test_first_failure_ends_its_worker_after_the_other_workers_ended(
    tmp_path: Path, run_in_process: Callable[[Path, list[str]], tuple[int, str]]
) -> None:
    """With -x the failing worker ends by itself and skips its later tests; no worker is left to stop."""
    failing_late = "import time\n\ndef test_fails_late():\n    time.sleep(2)\n    assert False\n\ndef test_after():\n    pass\n"
    root = make_project(tmp_path / "project", {"test_late.py": failing_late, "test_quick.py": "def test_quick():\n    pass\n"})
    # Stored times that put the two tests of test_late.py into one worker and the quick test into the other.
    times = {"test_late.py::test_fails_late": 1.0, "test_late.py::test_after": 1.0, "test_quick.py::test_quick": 2.0}
    assert sorted(runner.plan(list(times), times, 2)) == [list(times)[:2], list(times)[2:]]
    runner.save_times(root, [runner.Result(test_id, "passed", seconds, "") for test_id, seconds in times.items()])

    code, output = run_in_process(root, ["-j", "2", "-x"])

    assert code == runner.EXIT_FAILED, output
    assert "FAILED test_late.py::test_fails_late" in output
    assert "stopped after the first failed test (-x)" in output
    assert "1 passed, 1 failed, 0 errors, 0 skipped, 0 deselected, 1 not run in " in summary_line(output)


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_ctrl_c_stops_every_worker_and_their_children(tmp_path: Path) -> None:
    pid_file = tmp_path / "pids.txt"
    hang = f"""\
        import os, subprocess, sys, time
        from pathlib import Path

        def test_waits_with_a_child():
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep({LONG_S})"])
            Path({str(pid_file)!r}).write_text(f"{{os.getpid()}} {{child.pid}}")
            time.sleep(LONG_S)
        """
    root = make_project(tmp_path / "project", {"test_pass.py": PASSING, "test_hang.py": hang})
    script = (
        "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); import run_tests; "
        "sys.exit(run_tests.run(run_tests.parse_options(['-j', '2']), root=Path(sys.argv[2])))"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(RUNNER.parent), str(root)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
    )
    try:
        deadline = time.monotonic() + 120
        while not (pid_file.exists() and len(pid_file.read_text().split()) == 2):
            assert time.monotonic() < deadline, "the test that waits did not start"
            assert process.poll() is None, process.communicate()[0]
            time.sleep(0.05)
        worker_pid, child_pid = (int(value) for value in pid_file.read_text().split())
        assert process_exists(worker_pid) and process_exists(child_pid)

        process.send_signal(signal.SIGINT)
        output, _ = process.communicate(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    assert process.returncode == runner.EXIT_INTERRUPTED, output
    assert "interrupted: every worker was stopped" in output
    assert not process_exists(worker_pid)
    assert not process_exists(child_pid)


def test_a_test_that_fails_and_then_errors_in_teardown_counts_once(
    tmp_path: Path, run_in_process: Callable[[Path, list[str]], tuple[int, str]]
) -> None:
    """pytest writes two JUnit entries for such a test: they are one result, as in one process the exit code is 1."""
    teardown = """\
        import pytest

        @pytest.fixture
        def breaks_at_teardown():
            yield
            raise RuntimeError("the teardown broke")

        def test_fails_then_breaks(breaks_at_teardown):
            assert 2 + 2 == 5

        def test_passes_then_breaks(breaks_at_teardown):
            pass

        def test_plain():
            pass
        """
    root = make_project(tmp_path / "project", {"test_teardown.py": teardown})

    code, output = run_in_process(root, ["-j", "1"])

    assert code == runner.EXIT_FAILED, output
    assert "error:" not in output
    assert "1 passed, 0 failed, 2 errors, 0 skipped, 0 deselected in " in summary_line(output)
    # Both faults of the first test are shown under its one id.
    shown = output.split(" ERROR test_teardown.py::test_fails_then_breaks ")[1]
    assert "assert (2 + 2) == 5" in shown and "the teardown broke" in shown.split(" ERROR ")[0]


def test_a_file_skipped_at_collection_is_counted(
    tmp_path: Path, run_in_process: Callable[[Path, list[str]], tuple[int, str]]
) -> None:
    """A module-level skip (a missing package) leaves no test id; the summary still names it, as one process does."""
    skipped = 'import pytest\n\npytest.importorskip("a_package_that_is_not_installed")\n\ndef test_a():\n    pass\n'
    root = make_project(tmp_path / "project", {"test_pass.py": PASSING, "test_needs_package.py": skipped})

    code, output = run_in_process(root, ["-j", "2"])

    assert code == runner.EXIT_PASSED, output
    assert "1 file skipped at collection" in summary_line(output)
    assert "skipped at collection: test_needs_package.py" in output


def test_ctrl_c_while_the_workers_start_stops_the_worker_that_just_started(tmp_path: Path) -> None:
    """A Ctrl-C that comes while a worker process is being made must not leave that process running."""
    pid_file = tmp_path / "pids.txt"
    wait = f"import time\n\ndef test_waits():\n    time.sleep({LONG_S})\n"
    root = make_project(tmp_path / "project", {"test_a.py": wait, "test_b.py": wait})
    # The first worker process is made, then Ctrl-C arrives before the runner holds it.
    script = textwrap.dedent(
        f"""\
        import os, signal, subprocess, sys
        from pathlib import Path
        sys.path.insert(0, {str(RUNNER.parent)!r})
        import run_tests
        made = subprocess.Popen
        def popen(*args, **kwargs):
            process = made(*args, **kwargs)
            if "--junitxml" in " ".join(args[0]):
                with open({str(pid_file)!r}, "a") as pids:
                    pids.write(f"{{process.pid}}\\n")
                if len(Path({str(pid_file)!r}).read_text().split()) == 1:
                    os.kill(os.getpid(), signal.SIGINT)
            return process
        subprocess.Popen = popen
        sys.exit(run_tests.run(run_tests.parse_options(["-j", "2"]), root=Path({str(root)!r})))
        """
    )
    process = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, encoding="utf-8", timeout=120, check=False
    )
    pids = [int(value) for value in pid_file.read_text().split()]
    left = [pid for pid in pids if process_exists(pid)]
    for pid in left:
        os.kill(pid, signal.SIGKILL)

    assert process.returncode == runner.EXIT_INTERRUPTED, process.stdout + process.stderr
    assert "interrupted: every worker was stopped" in process.stdout
    assert pids and not left
