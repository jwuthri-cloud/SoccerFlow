"""Tests for the repository guard in tools/guard.py, and for the hook settings that run it.

Each test builds a throw-away git repository, adds files and runs the guard in a subprocess.
The strings that the guard refuses are built at run time, so that this file passes the guard.
Every test needs git and is skipped on a machine without it. A test that needs the hook runner
or the secret scanner is skipped on a machine without them.
Run the tests with `python3 -m unittest tests.test_guard` or with pytest.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import traceback
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
GUARD = REPO / "tools" / "guard.py"
ALLOW_LIST = GUARD.with_name("allowed_hosts.txt")
HOOK_FILE = REPO / ".pre-commit-config.yaml"
SECRET_SCAN_FILE = REPO / ".gitleaks.toml"
IGNORE_FILE = REPO / ".gitignore"

AUTHOR = "Test Author"
OTHER_AUTHOR = "Other Person"
EMAIL = "tester" + "@" + "example.org"
OTHER_EMAIL = "someone.else" + "@" + "example.net"
GOOGLE_KEY = "AI" + "za" + "x" * 35
NEW_GOOGLE_KEY = "A" + "Q." + "Ab" + "x9Y8" * 12
SK_KEY = "sk" + "-" + "a1B2" * 6
OPENROUTER_KEY = "sk" + "-or-v1-" + "0a1b" * 16
OAUTH_ACCESS = "ya" + "29." + "a0AfB_" * 8
GITHUB_ACCESS = "gh" + "p_" + "a1B2" * 9
GITHUB_FINE_ACCESS = "github" + "_pat_" + "a1B2" * 6
HUGGING_FACE_ACCESS = "h" + "f_" + "aBcD" * 9
WEB_ACCESS = "ey" + "J" + "hbGciOiJI." + "ey" + "J" + "zdWIiOiIx." + "c2lnbmF0dXJl"
PEM_HEADER = "-" * 5 + "BEGIN RSA " + "PRIVATE KEY" + "-" * 5
LONG_VALUE = "abcdefghij0123456789"
LOGIN_WORD = "hunter2"
MARK = ":" + "//"
WEB = "https" + MARK
MAC_HOME = "/" + "Users" + "/alice/"
LINUX_HOME = "/" + "home" + "/alice/"
WINDOWS_HOME = "C:" + "\\Users\\" + "alice\\"
DENIED = "Acme-Internal"
DENY_LIST = "# local deny-list\n\n" + DENIED + "\n"
HOME_MESSAGE = "home: absolute home-folder path; write ~ instead"
USER_FOLDER_MESSAGE = "home: path of a folder that belongs to one user of one machine"
NO_DENY_LIST = (
    "guard: notice: the local deny-list (guard-deny.txt in the git folder) does not exist"
)
#: The object name that git gives to a push for a branch that the remote does not have.
NO_COMMIT = "0" * 40
#: Variables that the hook runner sets before a push.
REMOTE_VARIABLE = "PRE_COMMIT_REMOTE_NAME"
PUSHED_COMMIT_VARIABLE = "PRE_COMMIT_TO_REF"
PUSHED_NAME_VARIABLE = "PRE_COMMIT_REMOTE_BRANCH"
NEW_KEY_LABEL = "Google API key in the new format (starts with AQ.A...)"
HOME_FOLDERS_MESSAGE = "home: folders of a home-folder path (Users, then a user name)"


def private_environment(home: Path, ceiling: Path) -> dict[str, str]:
    """Return an environment in which git reads no user and no system configuration.

    Git also cannot find a repository above `ceiling`, so a test never changes a real one.
    The variables of a hook runner are left out.
    """
    prefixes = ("GIT_", "PRE_COMMIT_")
    env = {name: value for name, value in os.environ.items() if not name.startswith(prefixes)}
    env.update(
        HOME=str(home),
        XDG_CONFIG_HOME=str(home / ".config"),
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CEILING_DIRECTORIES=str(ceiling),
        PRE_COMMIT_HOME=str(home / "hook-runner"),
        PRE_COMMIT_COLOR="never",
    )
    return env


def hook_entries() -> dict[str, str]:
    """Return the command line of each hook of the hook settings, by hook id."""
    blocks = re.split(r"(?m)^ +- id: ", HOOK_FILE.read_text(encoding="utf-8"))[1:]
    entries = {}
    for block in blocks:
        entry = re.search(r"(?m)^ +entry: (.+)$", block)
        if entry is None:
            raise AssertionError(f"a hook has no entry: {block!r}")
        entries[block.split("\n", 1)[0].strip()] = entry.group(1)
    return entries


_TEMPLATE: tempfile.TemporaryDirectory[str] | None = None
_GUARD_MODULE: ModuleType | None = None
_GUARD_STREAMS: tuple[str, str, str, str] | None = None


def new_repository_template() -> Path:
    """Return a work tree whose git folder is that of `git init --quiet --template=` with the identity.

    It is made once per test process, with the private environment of the tests, and each
    test copies its git folder.
    """
    global _TEMPLATE
    if _TEMPLATE is None:
        folder = tempfile.TemporaryDirectory()
        base = Path(folder.name).resolve()
        (base / "home").mkdir()
        (base / "repo").mkdir()
        env = private_environment(base / "home", base)
        for args in (
            ("init", "--quiet", "--template="),
            ("config", "--local", "user.name", AUTHOR),
            ("config", "--local", "user.email", EMAIL),
        ):
            subprocess.run(["git", *args], cwd=base / "repo", env=env, capture_output=True, check=True)
        _TEMPLATE = folder
    return Path(_TEMPLATE.name).resolve() / "repo"


def guard_module() -> ModuleType:
    """Return tools/guard.py as a module, loaded once per test process."""
    global _GUARD_MODULE
    if _GUARD_MODULE is None:
        spec = importlib.util.spec_from_file_location("guard_in_process", GUARD)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _GUARD_MODULE = module
    return _GUARD_MODULE


def guard_streams() -> tuple[str, str, str, str]:
    """Return the encoding and the error handler of the output and the error stream of a guard process."""
    global _GUARD_STREAMS
    if _GUARD_STREAMS is None:
        probe = "import sys; o, e = sys.stdout, sys.stderr; print(o.encoding, o.errors, e.encoding, e.errors)"
        found = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, encoding="utf-8", check=True
        ).stdout.split()
        _GUARD_STREAMS = (found[0], found[1], found[2], found[3])
    return _GUARD_STREAMS


def run_guard_in_process(
    command: list[str], env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run `main` of the guard in this process, as `python3 tools/guard.py ...` runs it.

    The environment is `env` during the call; git, which the guard starts, gets it too. The
    two output streams have the encoding and the error handler of a guard process
    (`guard_streams`), and are read back as `subprocess.run(..., encoding="utf-8")` reads
    them. The exit code is that of the process: the value of `main`, the code of a
    `SystemExit`, or 1 with the traceback for an error that `main` does not catch.
    """
    out_encoding, out_errors, err_encoding, err_errors = guard_streams()
    out, err = io.BytesIO(), io.BytesIO()
    out_text = io.TextIOWrapper(out, encoding=out_encoding, errors=out_errors, write_through=True)
    err_text = io.TextIOWrapper(err, encoding=err_encoding, errors=err_errors, write_through=True)
    with (
        mock.patch.dict(os.environ, env, clear=True),
        contextlib.redirect_stdout(out_text),
        contextlib.redirect_stderr(err_text),
    ):
        try:
            code = guard_module().main(command[2:])
        except SystemExit as stop:
            if stop.code is None or isinstance(stop.code, int):
                code = stop.code or 0
            else:
                print(stop.code, file=sys.stderr)
                code = 1
        except Exception:
            traceback.print_exc(file=sys.stderr)
            code = 1
    out_text.flush()
    err_text.flush()

    def text(data: bytes) -> str:
        return data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")

    return subprocess.CompletedProcess(command, code, text(out.getvalue()), text(err.getvalue()))


class GuardTestCase(unittest.TestCase):
    """A test case with a temporary git repository that has a neutral local identity."""

    def setUp(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("git is not installed")
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.base = Path(folder.name).resolve()
        self.home = self.base / "home"
        self.repo = self.base / "repo"
        self.home.mkdir()
        self.repo.mkdir()
        self.env = private_environment(self.home, self.base)
        # A copy of the git folder of `git init --quiet --template=` with the local identity, made
        # once per test process: three git processes less for each test.
        shutil.copytree(new_repository_template() / ".git", self.repo / ".git", symlinks=True)
        self.assertTrue((self.repo / ".git").is_dir())

    def git(self, *args: str, exit_code: int = 0, env: dict[str, str] | None = None) -> str:
        """Run git inside the temporary repository, check its exit code and return its output."""
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo,
            env=env or self.env,
            capture_output=True,
            encoding="utf-8",
            check=False,
        )
        if result.returncode != exit_code:
            self.fail(f"git {' '.join(args)} ended with {result.returncode}: {result.stderr}")
        return result.stdout

    def write(self, name: str, content: str | bytes) -> Path:
        """Write one file into the working tree of the temporary repository."""
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
        return path

    def stage(self, name: str, content: str | bytes, force: bool = False) -> None:
        """Write one file and add it to the git index. `force` also adds a file that is ignored."""
        self.write(name, content)
        self.git("add", *(["--force"] if force else []), "--", name)

    def commit(
        self,
        message: str,
        author_email: str = EMAIL,
        env: dict[str, str] | None = None,
        allow_empty: bool = False,
        author_name: str = AUTHOR,
    ) -> str:
        """Commit the git index with the given author, and return the short commit name."""
        options = ["--quiet", "--no-verify", "--author", f"{author_name} <{author_email}>"]
        if allow_empty:
            options.append("--allow-empty")
        self.git("-c", "commit.gpgsign=false", "commit", *options, "-m", message, env=env)
        return self.head_name()

    def head_name(self) -> str:
        """Return `short_name("HEAD")`, read from the files of the git folder when they hold it.

        A commit writes the name of its branch into `.git/HEAD` and the commit into the file of
        the branch, or the commit into `.git/HEAD` when no branch is checked out. Reading the two
        files saves one git process per commit. Any other layout asks git.
        """
        git_folder = self.repo / ".git"
        head = (git_folder / "HEAD").read_text(encoding="ascii").strip()
        if head.startswith("ref: "):
            branch_file = git_folder / head.removeprefix("ref: ")
            head = branch_file.read_text(encoding="ascii").strip() if branch_file.is_file() else ""
        if re.fullmatch(r"[0-9a-f]{40}", head):
            return head[:10]
        return self.short_name("HEAD")

    def short_name(self, revision: str) -> str:
        """Return the name of a commit as the guard writes it: the first 10 characters."""
        return self.git("rev-parse", revision).strip()[:10]

    def branch(self) -> str:
        """Return the name of the branch that is checked out."""
        return self.git("symbolic-ref", "--short", "HEAD").strip()

    def add_remote(self, name: str = "origin") -> Path:
        """Create an empty bare repository on disk and make it a remote of the test repository."""
        remote = self.base / f"{name}.git"
        self.git("init", "--quiet", "--bare", "--template=", str(remote))
        self.git("remote", "add", name, str(remote))
        return remote

    def set_deny_list(self) -> None:
        """Write the local deny-list. Its only entry is on line 3."""
        (self.repo / ".git" / "guard-deny.txt").write_text(DENY_LIST, encoding="utf-8")

    def guard(
        self,
        *args: str,
        script: Path = GUARD,
        repo: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run the guard against the temporary repository, or against another folder.

        The guard of this repository runs inside the test process, as its process would run it
        (`run_guard_in_process`). Another script runs as a process of its own. The guard still
        runs as a process in the tests of the hooks (`HookSettingsTest`), in
        `test_paths_mode_accepts_relative_paths` and in the tests of the host list.
        """
        command = [sys.executable, str(script), "--repo", str(repo or self.repo), *args]
        if script == GUARD:
            return run_guard_in_process(command, env or self.env)
        return subprocess.run(
            command, env=env or self.env, capture_output=True, encoding="utf-8", check=False
        )

    def assert_clean(self, result: subprocess.CompletedProcess[str]) -> None:
        """Check that the guard found nothing: exit code 0 and no report line."""
        self.assertEqual((result.returncode, result.stdout), (0, ""), result.stderr)

    def assert_refused(self, result: subprocess.CompletedProcess[str], rule: str) -> list[str]:
        """Check that the guard returned exit code 1, and return the report lines of the rule."""
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        lines = [line for line in result.stdout.splitlines() if f": {rule}: " in line]
        self.assertTrue(lines, f"no [{rule}] line in the output: {result.stdout!r}")
        return lines

    def assert_absent(self, secret: str, result: subprocess.CompletedProcess[str]) -> None:
        """Check that the output does not show the text, in any letter case."""
        self.assertNotIn(secret.lower(), (result.stdout + result.stderr).lower())

    def places(self, result: subprocess.CompletedProcess[str], rule: str) -> list[str]:
        """Return the place of each report line of the rule, as `<path>:<line>`."""
        return [":".join(line.split(":")[:2]) for line in self.assert_refused(result, rule)]


class MissingGitTest(unittest.TestCase):
    """The tests of this file need git."""

    def test_test_case_is_skipped_on_a_machine_without_git(self) -> None:
        case = CleanRepositoryTest("test_empty_repository_passes")
        self.addCleanup(case.doCleanups)
        with tempfile.TemporaryDirectory() as empty, mock.patch.dict(os.environ, {"PATH": empty}):
            with self.assertRaises(unittest.SkipTest):
                case.setUp()


class CleanRepositoryTest(GuardTestCase):
    """A repository without leaks passes in every mode."""

    def test_clean_staged_file_passes(self) -> None:
        self.stage("notes.md", "# Notes\n\nNothing secret here.\n")
        self.assert_clean(self.guard("--staged"))

    def test_clean_tracked_files_and_commits_pass(self) -> None:
        self.stage("src/module.py", "VALUE = 1\n")
        self.commit("first commit")
        self.assert_clean(self.guard("--tracked"))
        self.assert_clean(self.guard("--push"))

    def test_paths_on_clean_file_returns_zero(self) -> None:
        path = self.write("notes.txt", "plain text\n")
        self.assert_clean(self.guard("--paths", str(path)))

    def test_empty_repository_passes(self) -> None:
        for mode in ("--staged", "--tracked", "--push"):
            self.assert_clean(self.guard(mode))

    def test_allowed_types_and_names_pass(self) -> None:
        names = (
            "a.py b.md c.toml d.yaml e.yml f.json g.jsonl h.csv i.txt j.cfg k.ini pkg/py.typed "
            ".gitignore .gitattributes LICENSE NOTICE docs/README.MD"
        )
        for name in names.split():
            self.stage(name, "")
        self.assert_clean(self.guard("--staged"))

    def test_guard_files_pass_the_guard(self) -> None:
        sources = (GUARD, ALLOW_LIST, Path(__file__).resolve(), HOOK_FILE, SECRET_SCAN_FILE)
        for source in (*sources, IGNORE_FILE):
            self.stage(source.relative_to(REPO).as_posix(), source.read_bytes())
        self.assert_clean(self.guard("--staged"))

    def test_every_file_of_this_repository_passes_the_guard(self) -> None:
        """The repository of this test file passes its own guard, in the mode that other tools use.

        The test reads the repository and changes nothing in it. The local deny-list of the
        repository counts, as it does before a commit.
        """
        if not (REPO / ".git").exists():
            self.skipTest("the folder of this test file is not a git work tree")
        command = ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
        listing = subprocess.run(
            command, cwd=REPO, env=self.env, capture_output=True, encoding="utf-8", check=True
        )
        paths = [str(REPO / name) for name in listing.stdout.split("\0") if name]
        files = [path for path in paths if os.path.isfile(path)]
        self.assertTrue(files)
        for start in range(0, len(files), 200):
            self.assert_clean(self.guard("--paths", *files[start : start + 200], repo=REPO))


class FileRuleTest(GuardTestCase):
    """The rules that look at the file name, the size and the encoding."""

    def test_video_and_image_are_refused(self) -> None:
        self.stage("clips/match.mp4", b"\x00\x00\x00\x18ftypmp42")
        self.stage("frame.png", b"\x89PNG\r\n\x1a\n")
        lines = self.assert_refused(self.guard("--staged"), "ext")
        self.assertEqual(
            lines,
            [
                "clips/match.mp4:-: ext: file type '.mp4' is not on the list of allowed file types",
                "frame.png:-: ext: file type '.png' is not on the list of allowed file types",
            ],
        )

    def test_other_types_and_names_are_refused(self) -> None:
        for name in ("notebook.ipynb", "data.zip", "run.sh", "Makefile", ".env"):
            self.stage(name, "text\n")
        lines = self.assert_refused(self.guard("--staged"), "ext")
        self.assertEqual(len(lines), 5)

    def test_uv_lock_is_refused_in_every_folder(self) -> None:
        self.stage("uv.lock", "version = 1\n")
        self.stage("sub/uv.lock", "version = 1\n")
        lines = self.assert_refused(self.guard("--staged"), "lock")
        self.assertEqual([line.split(":")[0] for line in lines], ["sub/uv.lock", "uv.lock"])
        expected = "lock: uv.lock is a lock file: it is machine-specific and must not be tracked"
        self.assertTrue(all(line.endswith(expected) for line in lines), lines)

    def test_other_lock_files_are_refused(self) -> None:
        names = (
            "UV.LOCK locks/uv.lock.txt uv.lock.toml pylock.toml sub/pylock.dev.toml "
            "package-lock.json pnpm-lock.yaml conda-lock.yml requirements.lock.txt lock.json"
        ).split()
        for name in names:
            self.stage(name, "{}\n")
        result = self.guard("--staged")
        self.assertEqual(self.places(result, "lock"), sorted(f"{name}:-" for name in names))

    def test_names_that_only_hold_the_letters_of_lock_pass(self) -> None:
        names = "locks/run-machine.txt docs/blocklist.txt clock.json tests/test_lock.py unlock.md"
        for name in names.split():
            self.stage(name, "{}\n")
        self.assert_clean(self.guard("--staged"))

    def test_nul_byte_is_refused(self) -> None:
        self.stage("table.csv", b"a,b\n1,\x002\n")
        lines = self.assert_refused(self.guard("--staged"), "binary")
        self.assertEqual(lines, ["table.csv:-: binary: file contains a NUL byte"])

    def test_invalid_utf8_is_refused(self) -> None:
        self.stage("notes.txt", b"caf\xe9\n")
        lines = self.assert_refused(self.guard("--staged"), "binary")
        self.assertEqual(lines, ["notes.txt:-: binary: file is not valid UTF-8 text"])

    def test_large_file_is_refused(self) -> None:
        line = "word " * 19 + "word\n"
        self.assertEqual(len(line), 100)
        self.stage("at_limit.txt", line * 20_000)
        self.stage("over_limit.txt", line * 20_000 + "x")
        result = self.guard("--staged")
        lines = self.assert_refused(result, "size")
        expected = "over_limit.txt:-: size: file has 2000001 bytes; the limit is 2000000 bytes"
        self.assertEqual(lines, [expected])
        self.assertNotIn("at_limit.txt", result.stdout)

    def test_full_cost_ledger_files_pass_and_a_larger_ledger_file_is_refused(self) -> None:
        """The package splits its cost ledger into files of at most 1,000,000 bytes, so no file needs more room.

        A ledger line holds a time, names, token counts and the cache key of its request (32 hex
        digits). No text rule may take a line for a leak.
        """
        answer = {
            "time": "2026-11-20T10:11:12+00:00",
            "run_id": "probe-people_probe-20261120T101112Z-ab12",
            "clip_id": "SNGS-021",
            "step": "people",
            "frame": 120,
            "item": None,
            "repeat": 0,
            "request": "0123456789abcdef" * 2,
            "model": "gemini-4-argon",
            "service_tier": "standard",
            "attempts": 1,
            "input_tokens": 1534,
            "output_tokens": 912,
            "thought_tokens": 487,
            "usd": 0.017058,
        }
        lines = [json.dumps(dict(answer, frame=frame)) + "\n" for frame in range(10_000)]
        full_file = "".join(lines[:2600])
        self.assertTrue(900_000 < len(full_file) <= 1_000_000, len(full_file))
        self.stage("results/cache/ledger.jsonl", full_file)
        self.stage("results/cache/ledger-0002.jsonl", full_file)
        self.assert_clean(self.guard("--staged"))
        path = self.repo / "results" / "cache" / "ledger-0002.jsonl"
        self.assert_clean(self.guard("--paths", str(path)))
        # A ledger file has the size limit of every other file.
        whole_ledger = "".join(lines)
        self.assertGreater(len(whole_ledger), 2_500_000)
        self.stage("results/cache/ledger.jsonl", whole_ledger)
        refused = self.assert_refused(self.guard("--staged"), "size")
        size = f"size: file has {len(whole_ledger)} bytes; the limit is 2000000 bytes"
        self.assertEqual(refused, [f"results/cache/ledger.jsonl:-: {size}"])

    def test_symbolic_link_is_checked_as_its_target_text(self) -> None:
        (self.repo / "footage.txt").symlink_to(MAC_HOME + "footage")
        self.git("add", "--", "footage.txt")
        for mode in ("--staged", "--tracked"):
            lines = self.assert_refused(self.guard(mode), "home")
            self.assertEqual(len(lines), 1)

    def test_ignored_path_is_refused_after_a_forced_add(self) -> None:
        self.stage(".gitignore", "data/\noutputs/\nlocal/\n.env.*\n")
        self.stage("notes/data.md", "a file named like the folder, not inside it\n")
        ignored = ("data/clips.csv", "outputs/demo/summary.json", "local/notes.md", ".env.json")
        for name in ignored:
            self.stage(name, "{}\n", force=True)
        result = self.guard("--staged")
        lines = self.assert_refused(result, "ignored")
        self.assertEqual(self.places(result, "ignored"), sorted(f"{name}:-" for name in ignored))
        message = "the ignore rules of the repository name this path, so it must not be tracked"
        self.assertTrue(all(line.endswith(message) for line in lines), lines)
        self.assertEqual(len(result.stdout.splitlines()), len(ignored))

    def test_ignored_path_is_refused_in_every_mode(self) -> None:
        self.stage(".gitignore", "outputs/\n")
        self.stage("outputs/summary.json", "{}\n", force=True)
        self.commit("forced")
        for mode in ("--tracked", "--push"):
            self.assertEqual(self.places(self.guard(mode), "ignored"), ["outputs/summary.json:-"])
        path = self.repo / "outputs" / "summary.json"
        self.assert_refused(self.guard("--paths", str(path)), "ignored")

    def test_path_below_a_linked_folder_of_the_working_tree_is_reported(self) -> None:
        """Git cannot test such a path against the ignore rules. The guard must say so, not stop."""
        self.stage("base.md", "base\n")
        self.commit("base")
        main = self.branch()
        self.git("checkout", "--quiet", "-b", "side")
        self.stage("notes/sub/a.md", "clean\n")
        name = self.commit("side")
        self.git("checkout", "--quiet", main)
        (self.repo / "docs").mkdir()
        (self.repo / "notes").symlink_to("docs")
        result = self.guard("--push")
        self.assertEqual(result.returncode, 1, result.stderr)
        expected = (
            "notes/sub/a.md:-: ignored: git cannot test this path against the ignore rules, "
            "because notes is a symbolic link in the working tree; remove the link and run "
            f"the check again [commit {name}]"
        )
        self.assertEqual(result.stdout.splitlines(), [expected])
        (self.repo / "notes").unlink()
        self.assert_clean(self.guard("--push"))

    def test_ignore_file_of_the_user_account_is_not_read(self) -> None:
        """The answer of the guard must be the same on every machine."""
        (self.home / "ignore").write_text("notes.md\n", encoding="utf-8")
        user_config = "[core]\n\texcludesFile = " + str(self.home / "ignore") + "\n"
        (self.home / ".gitconfig").write_text(user_config, encoding="utf-8")
        self.stage("notes.md", "text\n", force=True)
        self.assertEqual(self.git("check-ignore", "--no-index", "notes.md").strip(), "notes.md")
        self.assert_clean(self.guard("--staged"))


class NameRuleTest(GuardTestCase):
    """The rules for text also read names: the path of a file, the name of a branch or of a tag."""

    def push_base(self) -> str:
        """Commit one clean file, push it to a new remote and return the name of the branch."""
        self.stage("base.md", "base\n")
        self.commit("base")
        self.add_remote()
        self.git("push", "--quiet", "origin", self.branch())
        return self.branch()

    def test_key_in_the_name_of_a_file_or_a_folder_is_refused_in_every_mode(self) -> None:
        names = [f"docs/{NEW_GOOGLE_KEY}.md", f"notes/{GOOGLE_KEY}/a.md"]
        for name in names:
            self.stage(name, "A clean note.\n")
        expected = [
            f"docs/AQ.A....md:-: secret: {NEW_KEY_LABEL} [path]",
            "notes/AIza.../a.md:-: secret: Google API key (starts with AIza...) [path]",
        ]
        staged = self.guard("--staged")
        self.assertEqual(self.assert_refused(staged, "secret"), expected)
        commit = self.commit("notes")
        paths = [str(self.repo / name) for name in names]
        pushed = self.guard("--push")
        in_commit = [f"{line} [commit {commit}]" for line in expected]
        self.assertEqual(self.assert_refused(pushed, "secret"), in_commit)
        results = [staged, pushed]
        for arguments in (["--tracked"], ["--paths", *paths]):
            results.append(self.guard(*arguments))
            self.assertEqual(self.assert_refused(results[-1], "secret"), expected, arguments)
        for result in results:
            self.assert_absent(NEW_GOOGLE_KEY, result)
            self.assert_absent(GOOGLE_KEY, result)

    def test_home_folders_and_a_private_address_in_a_path_are_refused(self) -> None:
        users, address = "Users", "10." + "20.30.40"
        self.stage(f"backup/{users}/alice/notes.md", "text\n")
        self.stage(f"-{users}-alice-project/notes.md", "text\n")
        self.stage(f"configs/{address}/settings.yaml", "a: 1\n")
        result = self.guard("--staged")
        self.assertEqual(result.returncode, 1, result.stderr)
        host_message = f"host: host '{address}' is not in allowed_hosts.txt"
        self.assertEqual(
            result.stdout.splitlines(),
            [
                f"-{users}-alice-project/notes.md:-: {USER_FOLDER_MESSAGE} [path]",
                f"backup/{users}/alice/notes.md:-: {HOME_FOLDERS_MESSAGE} [path]",
                f"configs/{address}/settings.yaml:-: {host_message} [path]",
            ],
        )

    def test_other_paths_pass(self) -> None:
        names = (
            "docs/users/guide.md site/home/index.md src/Users.py docs/Users/guide.md "
            "results/cache/home/people/0123456789abcdef0123456789abcdef.json "
            "results/cache/SNGS-021/number/0123456789abcdef0123456789abcdef.json "
            "tests/test_tokens.py locks/numpy-" + "10." + "1.2.3.txt"
        )
        for name in names.split():
            self.stage(name, "{}\n")
        self.assert_clean(self.guard("--staged"))

    def test_denied_word_in_the_name_of_a_new_branch_is_refused(self) -> None:
        main = self.push_base()
        self.set_deny_list()
        self.git("checkout", "--quiet", "-b", "fix-acme-internal-proxy")
        self.commit("A clean commit", allow_empty=True)
        result = self.guard("--push")
        entry = "matches the entry on line 3 of the local deny-list"
        expected = [f"-:-: deny: the name of branch fix-***-proxy {entry}"]
        self.assertEqual(result.stdout.splitlines(), expected)
        self.assert_absent(DENIED, result)
        # The hook runner names the remote. The name of a branch on another commit counts too.
        self.git("checkout", "--quiet", main)
        hook = dict(self.env, **{REMOTE_VARIABLE: "origin"})
        self.assertEqual(self.guard("--push", env=hook).stdout.splitlines(), expected)
        # A name that the remote has is not new: the push of it was checked or is done.
        self.git("push", "--quiet", "origin", "fix-acme-internal-proxy")
        self.assert_clean(self.guard("--push"))
        self.assert_clean(self.guard("--push", env=hook))

    def test_names_of_new_branches_and_of_every_tag_are_checked(self) -> None:
        self.push_base()
        self.git("branch", NEW_GOOGLE_KEY)
        self.git("branch", "Users" + "/alice/work")
        self.git("tag", GOOGLE_KEY)
        self.git("tag", "-a", "acme-internal-v1", "-m", "A clean message")
        self.set_deny_list()
        result = self.guard("--push")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                f"-:-: secret: {NEW_KEY_LABEL} [the name of branch AQ.A...]",
                f"-:-: {HOME_FOLDERS_MESSAGE} [the name of branch " + "Users" + "/alice/work]",
                "-:-: secret: Google API key (starts with AIza...) [the name of tag AIza...]",
                "-:-: deny: the name of tag ***-v1 matches the entry on line 3 "
                "of the local deny-list",
            ],
        )
        for hidden in (NEW_GOOGLE_KEY, GOOGLE_KEY, DENIED):
            self.assert_absent(hidden, result)
        # Git does not record which tags a remote has, so a pushed tag is still checked.
        self.git("push", "--quiet", "origin", "--all")
        self.git("push", "--quiet", "origin", "--tags")
        lines = self.guard("--push").stdout.splitlines()
        kinds = [line.split("the name of ")[1].split(" ")[0] for line in lines]
        self.assertEqual(kinds, ["tag", "tag"])

    def test_name_that_a_branch_gets_on_the_remote_is_checked(self) -> None:
        """`git push origin HEAD:refs/heads/<name>` writes a name that no local branch has."""
        self.push_base()
        self.set_deny_list()
        self.commit("A clean commit", allow_empty=True)
        self.assert_clean(self.guard("--push"))
        variables = {REMOTE_VARIABLE: "origin", PUSHED_NAME_VARIABLE: "refs/heads/acme-internal"}
        result = self.guard("--push", env=dict(self.env, **variables))
        entry = "matches the entry on line 3 of the local deny-list"
        self.assertEqual(
            result.stdout.splitlines(), [f"-:-: deny: the name of branch *** {entry}"]
        )
        # The name of a branch that the remote has is not new.
        variables[PUSHED_NAME_VARIABLE] = "refs/heads/" + self.branch()
        self.assert_clean(self.guard("--push", env=dict(self.env, **variables)))

    def test_tag_that_points_at_no_commit_is_refused(self) -> None:
        """A push sends the object of a tag. No hook starts for it, and no rule reads it."""
        self.push_base()
        value = self.base / "value.txt"
        value.write_text("value " + GOOGLE_KEY + "\n", encoding="utf-8")
        blob = self.git("hash-object", "-w", str(value)).strip()
        self.git("tag", "file-tag", blob)
        self.git("tag", "-a", "file-note", "-m", "A clean message", blob)
        self.git("tag", "folder-tag", "HEAD^{tree}")
        self.git("tag", "-a", "inner", "-m", "A clean message")
        self.git("tag", "-a", "outer", "-m", "A clean message", "inner")
        self.git("tag", "plain")
        result = self.guard("--push")
        self.assertEqual(result.returncode, 1, result.stderr)
        end = "and not at a commit; a push sends it without a check, so delete the tag"
        self.assertEqual(
            result.stdout.splitlines(),
            [
                f"-:-: tag: the tag file-note points at the content of a file {end}",
                f"-:-: tag: the tag file-tag points at the content of a file {end}",
                f"-:-: tag: the tag folder-tag points at a folder {end}",
                f"-:-: tag: the tag outer points at another tag {end}",
            ],
        )


class DatasetRuleTest(GuardTestCase):
    """The [dataset] rule: label files, prediction files and results of clip runs stay outside."""

    LABELS = {"info": {}, "images": [{"image_id": "1"}], "annotations": [{}], "categories": []}
    PREDICTIONS = {"predictions": [{"image_id": "1", "track_id": 3}]}
    RUN_RESULT = {"provenance": {}, "result": {"info": {}, "frames": [], "tracks": []}}
    #: One person of one image, as a label file and a prediction file write it.
    RECORD = {"image_id": "1", "track_id": 3, "bbox_image": {"x": 1, "y": 2, "w": 3, "h": 4}}
    LABELS_MESSAGE = (
        "dataset: label data of the dataset (keys images and annotations); "
        "it stays on the run machine"
    )
    PREDICTIONS_MESSAGE = "dataset: prediction data (key predictions); it stays on the run machine"
    RESULT_MESSAGE = (
        "dataset: result of a clip run (keys frames and tracks); it stays on the run machine"
    )
    RECORDS_MESSAGE = (
        "dataset: label or prediction data (records with the key track_id and a box); "
        "it stays on the run machine"
    )

    def test_label_file_is_refused_by_name_in_every_folder(self) -> None:
        self.stage("Labels-GameState.json", "{}\n")
        self.stage("results/metrics/labels-gamestate.JSON", "{}\n")
        lines = self.assert_refused(self.guard("--staged"), "dataset")
        message = "dataset: label file of the dataset; it stays on the run machine"
        expected = ["Labels-GameState.json", "results/metrics/labels-gamestate.JSON"]
        self.assertEqual(lines, [f"{name}:-: {message}" for name in expected])

    def test_label_data_is_refused_under_another_name(self) -> None:
        self.stage("configs/clip_021.json", json.dumps(self.LABELS))
        lines = self.assert_refused(self.guard("--staged"), "dataset")
        expected = (
            "configs/clip_021.json:-: dataset: label data of the dataset "
            "(keys images and annotations); it stays on the run machine"
        )
        self.assertEqual(lines, [expected])

    def test_prediction_data_is_refused(self) -> None:
        name = "predictions/SoccerNetGS-valid/pitchvlm/SNGS-021.json"
        path = self.write(name, json.dumps(self.PREDICTIONS, indent=2))
        lines = self.assert_refused(self.guard("--paths", str(path)), "dataset")
        expected = (
            f"{name}:-: dataset: prediction data (key predictions); it stays on the run machine"
        )
        self.assertEqual(lines, [expected])

    def test_result_of_a_clip_run_is_refused_by_name_and_by_content(self) -> None:
        self.stage("runs/dev/SNGS-021/result.json", "{}\n")
        self.stage("results/metrics/run_021.json", json.dumps(self.RUN_RESULT))
        lines = self.assert_refused(self.guard("--staged"), "dataset")
        self.assertEqual(
            lines,
            [
                f"results/metrics/run_021.json:-: {self.RESULT_MESSAGE}",
                "runs/dev/SNGS-021/result.json:-: dataset: result file of a clip run; "
                "it stays on the run machine",
            ],
        )

    def test_dataset_data_inside_another_key_or_a_list_is_refused(self) -> None:
        files = {
            "inside.json": {"provenance": {}, "labels": self.LABELS},
            "list.json": [self.LABELS, self.PREDICTIONS],
            "deep.json": {"runs": [{"clip": {"output": self.PREDICTIONS}}]},
            "bare_result.json": self.RUN_RESULT["result"],
        }
        for name, document in files.items():
            self.stage(f"results/metrics/{name}", json.dumps(document))
        lines = self.assert_refused(self.guard("--staged"), "dataset")
        self.assertEqual(
            lines,
            [
                f"results/metrics/bare_result.json:-: {self.RESULT_MESSAGE}",
                f"results/metrics/deep.json:-: {self.PREDICTIONS_MESSAGE}",
                f"results/metrics/inside.json:-: {self.LABELS_MESSAGE}",
                f"results/metrics/list.json:-: {self.LABELS_MESSAGE}",
                f"results/metrics/list.json:-: {self.PREDICTIONS_MESSAGE}",
            ],
        )

    def test_dataset_data_as_json_lines_is_refused(self) -> None:
        """The command `sync` accepts `.jsonl` files, so the rule must read them."""
        people = [dict(self.RECORD, track_id=track) for track in range(3)]
        records = "".join(json.dumps(person) + "\n" for person in people)
        self.stage("results/metrics/lines.jsonl", records)
        self.stage("results/metrics/one_line.jsonl", json.dumps(self.PREDICTIONS) + "\n")
        self.stage("results/metrics/lines.json", records)
        self.stage("results/metrics/pretty.jsonl", json.dumps(self.LABELS, indent=2))
        lines = self.assert_refused(self.guard("--staged"), "dataset")
        self.assertEqual(
            lines,
            [
                f"results/metrics/lines.json:-: {self.RECORDS_MESSAGE}",
                f"results/metrics/lines.jsonl:-: {self.RECORDS_MESSAGE}",
                f"results/metrics/one_line.jsonl:-: {self.PREDICTIONS_MESSAGE}",
                f"results/metrics/pretty.jsonl:-: {self.LABELS_MESSAGE}",
            ],
        )

    def test_dataset_data_after_a_byte_order_mark_is_refused(self) -> None:
        mark = "\ufeff".encode("utf-8")
        path = self.write("results/metrics/mark.json", mark + json.dumps(self.LABELS).encode())
        lines = self.assert_refused(self.guard("--paths", str(path)), "dataset")
        self.assertEqual(lines, [f"results/metrics/mark.json:-: {self.LABELS_MESSAGE}"])

    def test_records_of_the_label_format_are_refused_in_every_place(self) -> None:
        pitch_record = {"track_id": 7, "bbox_pitch": {"x_bottom_middle": 1.0}}
        self.stage("results/metrics/annotations.json", json.dumps([self.RECORD, pitch_record]))
        self.stage("results/metrics/renamed.json", json.dumps({"imgs": [], "annos": [self.RECORD]}))
        self.stage("results/metrics/by_id.json", json.dumps({"people": {"3": pitch_record}}))
        lines = self.assert_refused(self.guard("--staged"), "dataset")
        names = ("annotations.json", "by_id.json", "renamed.json")
        expected = [f"results/metrics/{name}:-: {self.RECORDS_MESSAGE}" for name in names]
        self.assertEqual(lines, expected)

    def test_other_json_files_pass(self) -> None:
        metrics = {"recall": 0.9, "images": 120, "annotations": "none", "provenance": {}}
        self.stage("results/metrics/people_probe.json", json.dumps(metrics))
        self.stage("results/metrics/scores.json", json.dumps({"result": {"HOTA": 25.0}}))
        self.stage("results/metrics/counts.json", json.dumps({"predictions": 12, "images": []}))
        self.stage("results/metrics/frames.json", json.dumps({"result": {"frames": []}}))
        self.stage("results/metrics/broken.json", '{"predictions": [')
        self.stage("results/metrics/tracks.json", json.dumps([{"track_id": 3, "frames": 120}]))
        self.stage("results/metrics/boxes.json", json.dumps({"bbox_image": {"x": 1}, "frames": []}))
        self.stage("results/cache/ledger.jsonl", '{"step": "people", "frame": 120}\n' * 3)
        # The rule reads JSON files only. Other file types are a limit that the RUNBOOK names.
        self.stage("notes/labels.txt", json.dumps(self.LABELS))
        self.assert_clean(self.guard("--staged"))


class SecretRuleTest(GuardTestCase):
    """The [secret] rule, and the promise that a secret is never printed."""

    def test_google_key_is_refused_and_not_printed(self) -> None:
        self.stage("config.py", '"""Settings."""\ngoogle = "' + GOOGLE_KEY + '"\n')
        result = self.guard("--staged")
        lines = self.assert_refused(result, "secret")
        self.assertEqual(lines, ["config.py:2: secret: Google API key (starts with AIza...)"])
        self.assert_absent(GOOGLE_KEY, result)

    def test_google_key_in_the_new_format_is_refused_in_every_place(self) -> None:
        content = [
            "my key: " + NEW_GOOGLE_KEY,
            "GEMINI_API_KEY=" + NEW_GOOGLE_KEY,
            'export GOOGLE_API_KEY="' + NEW_GOOGLE_KEY + '"',
            'curl -H "x-goog-api-key: ' + NEW_GOOGLE_KEY + '" ' + WEB + "ai.google.dev/models",
            json.dumps({"detail": "the request failed: 401 for key " + NEW_GOOGLE_KEY}),
            'client = genai.Client(api_key="' + NEW_GOOGLE_KEY + '")',
        ]
        self.stage("notes.txt", "\n".join(content) + "\n")
        result = self.guard("--staged")
        lines = self.assert_refused(result, "secret")
        message = "secret: Google API key in the new format (starts with AQ.A...)"
        self.assertEqual(lines, [f"notes.txt:{number}: {message}" for number in range(1, 7)])
        self.assert_absent(NEW_GOOGLE_KEY, result)

    def test_new_key_prefix_at_the_end_of_a_word_in_capitals_passes(self) -> None:
        self.stage("notes.md", "See F" + NEW_GOOGLE_KEY + " for more.\n")
        self.assert_clean(self.guard("--staged"))

    def test_key_after_an_escape_or_a_word_character_is_refused(self) -> None:
        """In a JSON text the letter of an escape stands directly before the key."""
        content = [
            json.dumps({"error": "the call failed with the key:\n" + NEW_GOOGLE_KEY}),
            json.dumps({"error": "the call failed with the key:\t" + NEW_GOOGLE_KEY}),
            "key" + NEW_GOOGLE_KEY,
            "GEMINI_KEY_" + NEW_GOOGLE_KEY,
            "attempt 2" + NEW_GOOGLE_KEY,
            json.dumps({"error": "the call failed with the key:\n" + SK_KEY}),
            json.dumps({"error": "the call failed with the token:\n" + GITHUB_ACCESS}),
            json.dumps({"error": "the call failed with the token:\n" + HUGGING_FACE_ACCESS}),
        ]
        self.assertIn("\\n" + NEW_GOOGLE_KEY, content[0])
        self.stage("results/metrics/errors.txt", "\n".join(content) + "\n")
        result = self.guard("--staged")
        lines = self.assert_refused(result, "secret")
        labels = [line.split(": secret: ")[1] for line in lines]
        new_key = "Google API key in the new format (starts with AQ.A...)"
        self.assertEqual(
            labels,
            [
                *[new_key] * 5,
                "key with the sk- prefix (starts with sk-a...)",
                "GitHub token (starts with ghp_...)",
                "Hugging Face token (starts with hf_a...)",
            ],
        )
        places = [f"results/metrics/errors.txt:{number}" for number in range(1, len(content) + 1)]
        self.assertEqual(self.places(result, "secret"), places)
        for secret in (NEW_GOOGLE_KEY, SK_KEY, GITHUB_ACCESS, HUGGING_FACE_ACCESS):
            self.assert_absent(secret, result)

    def test_sk_key_is_refused_and_not_printed(self) -> None:
        self.stage("settings.toml", 'value = "' + SK_KEY + '"\n')
        result = self.guard("--staged")
        lines = self.assert_refused(result, "secret")
        expected = "settings.toml:1: secret: key with the sk- prefix (starts with sk-a...)"
        self.assertEqual(lines, [expected])
        self.assert_absent(SK_KEY, result)

    def test_openrouter_key_is_refused_by_name_and_not_printed(self) -> None:
        content = [
            "OPENROUTER_API_KEY=" + OPENROUTER_KEY,
            'curl -H "Authorization: Bearer ' + OPENROUTER_KEY + '"',
            json.dumps({"detail": "the call failed with the key:\n" + OPENROUTER_KEY}),
        ]
        self.stage("notes.txt", "\n".join(content) + "\n")
        result = self.guard("--staged")
        lines = self.assert_refused(result, "secret")
        message = "secret: OpenRouter API key (starts with sk-o...)"
        self.assertEqual(lines, [f"notes.txt:{number}: {message}" for number in range(1, 4)])
        self.assert_absent(OPENROUTER_KEY, result)

    def test_sk_inside_a_word_passes(self) -> None:
        self.stage("notes.md", "A risk-adjusted-return-on-capital-employed figure.\n")
        self.assert_clean(self.guard("--staged"))

    def test_oauth_token_is_refused(self) -> None:
        self.stage("notes.md", "Bearer " + OAUTH_ACCESS + "\n")
        result = self.guard("--staged")
        lines = self.assert_refused(result, "secret")
        self.assertEqual(lines, ["notes.md:1: secret: Google OAuth token (starts with ya29...)"])
        self.assert_absent(OAUTH_ACCESS, result)

    def test_tokens_of_other_services_are_refused(self) -> None:
        content = [
            GITHUB_ACCESS,
            GITHUB_FINE_ACCESS,
            "huggingface-cli login --token " + HUGGING_FACE_ACCESS,
            WEB_ACCESS,
            "Authorization: Bearer " + LONG_VALUE,
        ]
        self.stage("notes.txt", "\n".join(content) + "\n")
        result = self.guard("--staged")
        lines = self.assert_refused(result, "secret")
        labels = [line.split(": secret: ")[1] for line in lines]
        self.assertEqual(
            labels,
            [
                "GitHub token (starts with ghp_...)",
                "GitHub token (starts with gith...)",
                "Hugging Face token (starts with hf_a...)",
                "JSON web token (starts with eyJh...)",
                "bearer token (starts with abcd...)",
            ],
        )
        self.assertEqual(self.places(result, "secret"), [f"notes.txt:{n}" for n in range(1, 6)])

    def test_private_key_headers_of_every_kind_are_refused(self) -> None:
        kinds = ("RSA ", "EC ", "OPENSSH ", "DSA ", "ENCRYPTED ", "")
        headers = ["-" * 5 + "BEGIN " + kind + "PRIVATE KEY" + "-" * 5 for kind in kinds]
        self.assertEqual(headers[0], PEM_HEADER)
        self.stage("key.txt", "first line\n" + "\n".join(headers) + "\n")
        result = self.guard("--staged")
        lines = self.assert_refused(result, "secret")
        self.assertEqual(self.places(result, "secret"), [f"key.txt:{n}" for n in range(2, 8)])
        self.assertTrue(all("secret: PEM private key header" in line for line in lines), lines)

    def test_secret_assignments_are_refused(self) -> None:
        content = [
            "GEMINI_API_KEY=" + LONG_VALUE,
            'HF_TOKEN = "' + LONG_VALUE + '"',
            "client_secret: " + LONG_VALUE,
            '{"api_key": "' + LONG_VALUE + '"}',
            'hf_token: str = "' + LONG_VALUE + '"',
            'SIGNING_SECRET = b"' + LONG_VALUE + '"',
        ]
        self.stage("settings.txt", "\n".join(content) + "\n")
        result = self.guard("--staged")
        lines = self.assert_refused(result, "secret")
        self.assertEqual([line.split(":")[1] for line in lines], ["1", "2", "3", "4", "5", "6"])
        self.assert_absent(LONG_VALUE, result)

    def test_secret_assignments_in_other_forms_are_refused(self) -> None:
        content = [
            'os.environ["GEMINI_API_KEY"] = "' + LONG_VALUE + '"',
            'os.environ.setdefault("GEMINI_API_KEY", "' + LONG_VALUE + '")',
            'key = os.environ.get("GEMINI_API_KEY", "' + LONG_VALUE + '")',
            'apikey = "' + LONG_VALUE + '"',
            'token = "' + LONG_VALUE + '"',
            'password = "' + LONG_VALUE + '"',
            json.dumps({"token": LONG_VALUE}),
            "x-goog-api-key: " + LONG_VALUE,
            'GEMINI_API_KEY="' + LONG_VALUE[:10] + "/+" + LONG_VALUE[10:] + '=="',
        ]
        self.stage("settings.txt", "\n".join(content) + "\n")
        result = self.guard("--staged")
        expected = [f"settings.txt:{number}" for number in range(1, len(content) + 1)]
        self.assertEqual(self.places(result, "secret"), expected)
        self.assert_absent(LONG_VALUE, result)

    def test_secret_names_with_other_words_are_refused(self) -> None:
        content = [
            "DB_PASS" + "WORD=" + LONG_VALUE,
            "POSTGRES_PASS" + "WORD: " + LONG_VALUE,
            "db_pass" + 'word = "' + LONG_VALUE + '"',
            "admin-pass" + "wd: " + LONG_VALUE,
            "SECRET" + '_KEY = "' + LONG_VALUE + '"',
            "AWS_SECRET" + "_ACCESS_KEY=" + LONG_VALUE,
            "PRIVATE" + "_KEY=" + LONG_VALUE,
            "access" + '-key: "' + LONG_VALUE + '"',
        ]
        self.stage("settings.txt", "\n".join(content) + "\n")
        result = self.guard("--staged")
        expected = [f"settings.txt:{number}" for number in range(1, len(content) + 1)]
        self.assertEqual(self.places(result, "secret"), expected)
        self.assert_absent(LONG_VALUE, result)

    def test_bare_secret_value_at_the_end_of_a_sentence_is_refused(self) -> None:
        self.stage("notes.md", "Set GEMINI_API_KEY=" + LONG_VALUE + ". Then start the run.\n")
        result = self.guard("--staged")
        self.assertEqual(self.places(result, "secret"), ["notes.md:1"])
        self.assert_absent(LONG_VALUE, result)

    def test_secret_value_on_the_next_line_is_refused(self) -> None:
        content = [
            "client = Client(",  # line 1
            "    api_key=(",
            '        "' + LONG_VALUE + '"',
            "    ),",
            ")",
            "gemini_api_key:",  # line 6
            "  " + LONG_VALUE,
            "api_key: >-",  # line 8
            "  " + LONG_VALUE,
            "api_keys:",  # line 10
            "  - " + LONG_VALUE,
        ]
        self.stage("settings.txt", "\n".join(content) + "\n")
        result = self.guard("--staged")
        places = [f"settings.txt:{number}" for number in (2, 6, 8, 10)]
        self.assertEqual(self.places(result, "secret"), places)
        self.assert_absent(LONG_VALUE, result)

    def test_environment_lookups_and_short_values_pass(self) -> None:
        content = [
            'GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]',
            'api_key = os.getenv("GEMINI_API_KEY")',
            'hf_token = os.environ.get("HF_TOKEN", "")',
            "client = Client(api_key=api_key)",
            "api_key = load_the_api_key_from_the_environment()",
            "GEMINI_API_KEY=" + LONG_VALUE[:19],
            "GEMINI_API_KEY=<put-your-own-key-here-please>",
        ]
        self.stage("client.py", "\n".join(content) + "\n")
        self.assert_clean(self.guard("--staged"))

    def test_names_and_placeholders_after_a_secret_name_pass(self) -> None:
        content = [
            "GEMINI_API_KEY=",
            "A_SECOND_VARIABLE_WITH_A_LONG_NAME=1",
            "secrets:",
            "  the_name_of_an_environment_variable: GEMINI_API_KEY",
            "def connect(api_key, a_second_parameter_with_a_long_name=None):",
            'password: "{{ vault_password_of_the_database }}"',
            'api_key = "<put-your-own-key-here-please>"',
            'api_key = "$' + LONG_VALUE + '"',
            '"thought_tokens": 400, "the_name_of_another_long_key": 1',
            'names = ["api_key", "the_name_of_another_long_field"]',
        ]
        self.stage("settings.txt", "\n".join(content) + "\n")
        self.assert_clean(self.guard("--staged"))

    def test_names_of_code_after_a_secret_name_pass(self) -> None:
        """A name of 20 or more characters without a digit is a name of the code, not a value.

        This code base counts model tokens, so many of its names end with `_tokens`.
        """
        content = [
            "max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS",
            "max_output_tokens=DEFAULT_OUTPUT_TOKEN_LIMIT,",
            "output_tokens=DEFAULT_OUTPUT_TOKEN_ESTIMATE,",
            "    thought_tokens=measured_thinking_tokens_of_the_pilot,",
            "estimates_by_token = DEFAULT_TOKEN_ESTIMATES_BY_STEP",
            "self._api_key = api_key_from_the_environment",
            "api_key: the_environment_variable_of_the_key",
            "client_secret = load_the_client_settings_file",
            "input_tokens: number_of_input_tokens_billed",
            "  thought_tokens: not_measured_until_the_pilot",
            "total_tokens = input_tokens_plus_output_tokens",
            "api_key = settings_of_the_run_machine_2026.key",
            "private_key = load_the_private_key_of_the_signer",
        ]
        self.stage("src/limits.py", "\n".join(content) + "\n")
        self.assert_clean(self.guard("--staged"))

    def test_password_in_an_address_is_refused_and_not_printed(self) -> None:
        content = [
            "git clone " + WEB + "someone:" + LOGIN_WORD + "@github.com/example/project.git",
            "postgres" + MARK + "app:" + LONG_VALUE + "@localhost:5432/app",
            "git remote add origin " + WEB + "x-access-token:" + GITHUB_ACCESS + "@github.com/a/b",
        ]
        self.stage("notes.md", "\n".join(content) + "\n")
        result = self.guard("--staged")
        lines = self.assert_refused(result, "secret")
        self.assertEqual(
            lines,
            [
                "notes.md:1: secret: password in an address (not shown)",
                "notes.md:2: secret: password in an address (starts with abcd...)",
                "notes.md:3: secret: GitHub token (starts with ghp_...)",
            ],
        )
        for secret in (LOGIN_WORD, LONG_VALUE, GITHUB_ACCESS):
            self.assert_absent(secret, result)

    def test_address_without_a_password_passes(self) -> None:
        content = [
            WEB + "someone:<password>@github.com/example/project.git",
            WEB + "x-access-token:$GITHUB_TOKEN@github.com/example/project.git",
            "ssh" + MARK + "git@github.com/example/project.git",
            "http" + MARK + "localhost:8000/path@name",
        ]
        self.stage("notes.md", "\n".join(content) + "\n")
        self.assert_clean(self.guard("--staged"))

    def test_secret_is_reported_once(self) -> None:
        self.stage("settings.txt", 'GEMINI_API_KEY = "' + GOOGLE_KEY + '"\n')
        lines = self.assert_refused(self.guard("--staged"), "secret")
        self.assertEqual(lines, ["settings.txt:1: secret: Google API key (starts with AIza...)"])

    def test_secret_is_not_printed_by_another_rule(self) -> None:
        self.stage("links.md", WEB + GOOGLE_KEY + "/a\n" + WEB + SK_KEY + "/b\n")
        result = self.guard("--staged")
        self.assertEqual(len(self.assert_refused(result, "secret")), 2)
        self.assertEqual(len(self.assert_refused(result, "host")), 2)
        self.assert_absent(GOOGLE_KEY, result)
        self.assert_absent(SK_KEY, result)


class EncodedDataRuleTest(GuardTestCase):
    """The [base64] and [pixels] rules: binary data that is written as text."""

    PAYLOAD = bytes(range(256)) * 8

    def test_long_base64_run_is_refused(self) -> None:
        self.stage("cache.json", '{"data": "' + "QUJD" * 75 + '"}\n')
        lines = self.assert_refused(self.guard("--staged"), "base64")
        expected = "cache.json:1: base64: run of 300 base64 characters; 200 or more are refused"
        self.assertEqual(lines, [expected])

    def test_url_safe_base64_run_is_refused(self) -> None:
        self.stage("cache.json", '{"data": "' + "QU-_" * 75 + '"}\n')
        self.assertEqual(len(self.assert_refused(self.guard("--staged"), "base64")), 1)

    def test_base64_padding_counts_for_the_length(self) -> None:
        self.stage("at_limit.txt", "Q" * 198 + "==\n")
        self.stage("below_limit.txt", "Q" * 197 + "==\n" + "Q" * 199 + "\n")
        lines = self.assert_refused(self.guard("--staged"), "base64")
        self.assertEqual([line.split(":")[0] for line in lines], ["at_limit.txt"])

    def test_hex_digest_passes(self) -> None:
        digest = hashlib.sha256(b"frame 0001").hexdigest()
        self.assertEqual(len(digest), 64)
        digests = [hashlib.sha256(bytes([number])).hexdigest() for number in range(8)]
        self.stage("index.jsonl", '{"sha256": "' + digest + '"}\n')
        self.stage("cassette.json", json.dumps({"image_hashes": digests}, indent=2))
        self.assert_clean(self.guard("--staged"))

    def test_wrapped_base64_text_is_refused(self) -> None:
        text = base64.b64encode(self.PAYLOAD).decode("ascii")
        wrapped = textwrap.wrap(text, 76)
        self.stage("frame.txt", "first line\n" + "\n".join(wrapped) + "\n")
        self.stage("frame.json", json.dumps({"data": base64.encodebytes(self.PAYLOAD).decode()}))
        self.stage("frame_hex.txt", "\n".join(textwrap.wrap(self.PAYLOAD.hex(), 64)) + "\n")
        lines = self.assert_refused(self.guard("--staged"), "base64")
        run = f"base64: run of {len(text)} base64 characters in {len(wrapped)} pieces"
        limit = "2000 or more are refused"
        self.assertEqual(
            lines,
            [
                f"frame.json:1: {run}; {limit}",
                f"frame.txt:2: {run}; {limit}",
                f"frame_hex.txt:1: base64: run of 4096 base64 characters in 64 pieces; {limit}",
            ],
        )

    def test_short_wrapped_text_passes_and_long_pieces_are_still_refused(self) -> None:
        text = base64.b64encode(self.PAYLOAD[:1200]).decode("ascii")
        self.assertLess(len(text), 2000)
        self.stage("short.txt", "\n".join(textwrap.wrap(text, 76)) + "\n")
        self.stage("pieces.txt", "Q" * 300 + "\n" + "Q" * 300 + "\n")
        result = self.guard("--staged")
        self.assertEqual(self.places(result, "base64"), ["pieces.txt:1", "pieces.txt:2"])

    def test_wrapped_run_counts_from_2000_characters_in_pieces_of_40(self) -> None:
        self.stage("at_limit.txt", ("Q" * 40 + "\n") * 50)
        self.stage("below_limit.txt", ("Q" * 40 + "\n") * 49 + "Q" * 39 + "\n")
        self.stage("short_pieces.txt", ("Q" * 39 + "\n") * 80)
        lines = self.assert_refused(self.guard("--staged"), "base64")
        expected = "run of 2000 base64 characters in 50 pieces; 2000 or more are refused"
        self.assertEqual(lines, [f"at_limit.txt:1: base64: {expected}"])

    def test_lines_of_long_words_between_other_text_pass(self) -> None:
        names = [
            f"test_number_{number:03d}_has_a_name_of_more_than_forty_characters"
            for number in range(60)
        ]
        self.assertGreater(sum(len(name) for name in names), 2000)
        self.stage("names.txt", "".join(f"- {name}: passed\n" for name in names))
        self.assert_clean(self.guard("--staged"))

    def test_rule_lines_without_a_letter_or_a_digit_pass(self) -> None:
        """A long line of dashes is a rule line of a document, not data."""
        table = "| a | b |\n|" + "-" * 58 + "|" + "-" * 54 + "|\n| 1 | 2 |\n"
        content = "# Title\n\n" + "-" * 210 + "\n\n" + "_" * 300 + "\n\n" + table
        self.stage("notes.md", content + ("-" * 60 + "\n") * 40)
        self.assert_clean(self.guard("--staged"))
        # One letter in the line makes it a piece of text again.
        self.stage("notes.md", "-" * 210 + "Q\n")
        self.assertEqual(self.places(self.guard("--staged"), "base64"), ["notes.md:1"])

    def test_address_that_holds_its_data_is_refused(self) -> None:
        picture = "data" + ":image/png;base64," + "QUJD" * 30
        self.stage("page.md", "# Page\n\n![frame](" + picture + ")\n")
        lines = self.assert_refused(self.guard("--staged"), "base64")
        self.assertEqual(lines, ["page.md:3: base64: address that holds encoded data"])

    def test_table_of_byte_values_is_refused(self) -> None:
        values = [number % 256 for number in range(20_000)]
        rows = [values[start : start + 100] for start in range(0, len(values), 100)]
        self.stage("frame.json", json.dumps({"pixels": rows}))
        self.stage(
            "frame.csv", "\n".join(",".join(str(value) for value in row) for row in rows) + "\n"
        )
        lines = self.assert_refused(self.guard("--staged"), "pixels")
        message = "pixels: table of 20000 byte values; 20000 or more in a row are refused"
        self.assertEqual(lines, [f"frame.csv:1: {message}", f"frame.json:1: {message}"])

    def test_other_tables_of_numbers_pass(self) -> None:
        self.stage("short.json", json.dumps([number % 256 for number in range(19_999)]))
        self.stage("boxes.json", json.dumps([number % 1001 for number in range(40_000)]))
        self.stage("errors.json", json.dumps([number / 8 for number in range(40_000)]))
        self.stage("signed.json", json.dumps([-(number % 256) for number in range(40_000)]))
        self.assert_clean(self.guard("--staged"))


class HostRuleTest(GuardTestCase):
    """The [host] rule: hosts that are not on the allow-list."""

    def test_unknown_host_is_refused(self) -> None:
        self.stage("notes.md", "Intro.\n\nSee " + WEB + "registry.example.net:8443/simple/ now.\n")
        lines = self.assert_refused(self.guard("--staged"), "host")
        expected = "notes.md:3: host: host 'registry.example.net' is not in allowed_hosts.txt"
        self.assertEqual(lines, [expected])

    def test_unknown_host_is_refused_for_every_scheme(self) -> None:
        content = [
            "http" + MARK + "registry.example.net/simple/",
            "ssh" + MARK + "git" + "@" + "git.example.net:2222/team/project.git",
            "pkg @ git+ssh" + MARK + "git" + "@" + "git.example.net/team/pkg.git",
            "ftp" + MARK + "files.example.net/data.csv",
            "gs" + MARK + "private-bucket/footage/clip.txt",
            "wss" + MARK + "stream.example.net/socket",
            "postgresql" + MARK + "db.example.net:5432/app",
        ]
        self.stage("notes.md", "\n".join(content) + "\n")
        lines = self.assert_refused(self.guard("--staged"), "host")
        hosts = [line.split("'")[1] for line in lines]
        self.assertEqual(
            hosts,
            [
                "registry.example.net",
                "git.example.net",
                "git.example.net",
                "files.example.net",
                "private-bucket",
                "stream.example.net",
                "db.example.net",
            ],
        )
        self.assertEqual([line.split(":")[1] for line in lines], list("1234567"))

    def test_allowed_hosts_pass(self) -> None:
        content = [
            "[code](https://github.com/example/project) and <https://GitHub.com>.",
            "Paper: https://arxiv.org/abs/2404.11335, data: https://huggingface.co.",
            "Local: http" + MARK + "localhost:8000/ and http" + MARK + "127.0.0.1:8000/x",
            'Not an address: url.startswith("' + WEB + '")',
            "git clone git" + "@" + "github.com:example/project.git",
            "pip install --index-url " + WEB + "pypi.org/simple numpy",
        ]
        self.stage("notes.md", "\n".join(content) + "\n")
        self.assert_clean(self.guard("--staged"))

    def test_host_is_read_after_the_user_information(self) -> None:
        content = [
            WEB + "github.com" + "@" + "registry.example.net/x",
            WEB + "github.com;x" + "@" + "registry.example.net/x",
            WEB + "github.com,x" + "@" + "registry.example.net/x",
        ]
        self.stage("notes.md", "\n".join(content) + "\n")
        lines = self.assert_refused(self.guard("--staged"), "host")
        self.assertEqual(len(lines), 3)
        self.assertTrue(all("host 'registry.example.net'" in line for line in lines), lines)

    def test_short_form_of_a_git_remote_is_read(self) -> None:
        self.stage("notes.md", "git clone git" + "@" + "git.example.net:team/project.git\n")
        lines = self.assert_refused(self.guard("--staged"), "host")
        expected = "notes.md:1: host: host 'git.example.net' is not in allowed_hosts.txt"
        self.assertEqual(lines, [expected])

    def test_package_index_options_are_read(self) -> None:
        content = [
            "pip install --index-url " + "registry.example.net/simple numpy",
            "--extra-index-url=" + "registry.example.net/simple",
            "--trusted-host " + "registry.example.net",
            "The option --index-url names the package index.",
        ]
        self.stage("notes.md", "\n".join(content) + "\n")
        result = self.guard("--staged")
        self.assertEqual(self.places(result, "host"), ["notes.md:1", "notes.md:2", "notes.md:3"])

    def test_address_of_a_private_network_is_refused_without_a_scheme(self) -> None:
        content = [
            "server " + "10." + "20.30.40 port 8080",
            "--trusted-host " + "192." + "168.1.20",
            "proxy = " + "172." + "16.0.9:3128",
            "http" + MARK + "10." + "20.30.40:8080/simple",
            "the last private range ends at " + "172." + "31.255.254",
        ]
        self.stage("notes.md", "\n".join(content) + "\n")
        result = self.guard("--staged")
        places = ["notes.md:1", "notes.md:2", "notes.md:3", "notes.md:4", "notes.md:5"]
        self.assertEqual(self.places(result, "host"), places)

    def test_private_address_after_an_equals_sign_is_refused(self) -> None:
        content = [
            "HOST=" + "10." + "1.2.3",
            "serve --host=" + "192." + "168.1.20 --port 8000",
            "see http" + MARK + "localhost:8000/?server=" + "10." + "1.2.3",
            'server = "' + "172." + '16.0.9"',
        ]
        self.stage("notes.md", "\n".join(content) + "\n")
        result = self.guard("--staged")
        places = [f"notes.md:{number}" for number in range(1, len(content) + 1)]
        self.assertEqual(self.places(result, "host"), places)

    def test_shared_and_link_local_addresses_are_refused(self) -> None:
        content = [
            "ssh " + "100." + "64.3.2",
            "the shared range ends at " + "100." + "127.255.254",
            "curl " + "169." + "254.169.254",
        ]
        self.stage("notes.md", "\n".join(content) + "\n")
        lines = self.assert_refused(self.guard("--staged"), "host")
        hosts = [line.split("'")[1] for line in lines]
        expected = ["100." + "64.3.2", "100." + "127.255.254", "169." + "254.169.254"]
        self.assertEqual(hosts, expected)

    def test_version_numbers_and_public_addresses_pass(self) -> None:
        content = [
            "nvidia-curand-cu12==" + "10." + "3.2.106",
            "opencv-python 4." + "10." + "0.84",
            "resolver 8.8.8.8, not private: " + "172." + "15.0.1 and " + "172." + "32.0.1",
            "listening on 127.0.0.1",
            "numpy==2.5.4, pillow>=" + "10." + "4.0.1, torch~=" + "10." + "1.2.3",
            "not private: " + "100." + "63.0.1, " + "100." + "128.0.1 and " + "169." + "253.0.1",
        ]
        self.stage("notes.md", "\n".join(content) + "\n")
        self.assert_clean(self.guard("--staged"))

    def test_dot_entry_allows_sub_domains_only(self) -> None:
        tools = self.base / "tools"
        tools.mkdir()
        script = Path(shutil.copy(GUARD, tools / "guard.py"))
        allow_list = "# test list\n\n.example.org\nExact.Example.Net\n"
        (tools / "allowed_hosts.txt").write_text(allow_list, encoding="utf-8")
        self.stage("pass.md", WEB + "docs.example.org/a " + WEB + "exact.example.net/b\n")
        self.assert_clean(self.guard("--staged", script=script))
        self.stage("fail.md", WEB + "example.org/a\n" + WEB + "sub.exact.example.net/b\n")
        lines = self.assert_refused(self.guard("--staged", script=script), "host")
        self.assertEqual(
            [line.split(":")[:2] for line in lines], [["fail.md", "1"], ["fail.md", "2"]]
        )

    def test_missing_allow_list_is_an_error(self) -> None:
        script = Path(shutil.copy(GUARD, self.base / "guard.py"))
        self.stage("notes.md", "text\n")
        result = self.guard("--staged", script=script)
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("allowed_hosts.txt", result.stderr)


class HomeRuleTest(GuardTestCase):
    """The [home] rule: paths into the folders of one user of one machine."""

    def test_home_paths_are_refused(self) -> None:
        content = [
            'video = "' + MAC_HOME + 'clips/match.txt"',
            "cd " + LINUX_HOME + "project",
            WINDOWS_HOME + "project",
            '{"path": "' + WINDOWS_HOME.replace("\\", "\\\\") + 'project"}',
        ]
        self.stage("paths.txt", "\n".join(content) + "\n")
        result = self.guard("--staged")
        lines = self.assert_refused(result, "home")
        self.assertEqual(lines, [f"paths.txt:{number}: {HOME_MESSAGE}" for number in (1, 2, 3, 4)])
        self.assert_absent("alice", result)

    def test_home_folder_is_refused_in_other_forms(self) -> None:
        mac = MAC_HOME.rstrip("/")
        linux = LINUX_HOME.rstrip("/")
        content = [
            "cd " + mac,
            'HOME = "' + mac + '"',
            "PATH=" + linux + ":/usr/bin",
            "/System/Volumes/Data" + MAC_HOME + "data/x.json",
            "/Volumes/Backup" + MAC_HOME + "x",
            "/mnt/c" + MAC_HOME + "x",
            "/export" + LINUX_HOME + "x",
            mac.replace("alice", "Alice Smith") + "/x",
            mac.lower() + "/x",
            "/" + "root" + "/data/x",
            "/var/" + "root" + "/x",
            "~" + "alice/data/x",
            mac.replace("/", "%2F") + "%2Fx",
            '{"path": "' + mac.replace("/", "\\/") + '\\/x"}',
            "\\Users\\" + "alice\\x",
        ]
        self.stage("paths.txt", "\n".join(content) + "\n")
        result = self.guard("--staged")
        lines = self.assert_refused(result, "home")
        expected = [f"paths.txt:{number}: {HOME_MESSAGE}" for number in range(1, len(content) + 1)]
        self.assertEqual(lines, expected)
        self.assert_absent("alice", result)

    def test_other_folders_of_one_user_are_refused(self) -> None:
        content = [
            "/private/var/" + "folders" + "/ab/cdef0123456789/T/tmpabc/x.json",
            "/srv/notes/-" + "Users" + "-alice-notes/x",
        ]
        self.stage("paths.txt", "\n".join(content) + "\n")
        lines = self.assert_refused(self.guard("--staged"), "home")
        self.assertEqual(lines, [f"paths.txt:{number}: {USER_FOLDER_MESSAGE}" for number in (1, 2)])

    def test_other_paths_pass(self) -> None:
        content = [
            "~/pitchvlm-data/clips and ." + LINUX_HOME + "x.py and src" + LINUX_HOME + "x.py",
            "/" + "Users" + "/<name>/ and /" + "home" + "/$USER/ are placeholders",
            "https://github.com" + LINUX_HOME + "project",
            "the " + "root" + "/<clip>/<step> folder and <out>/" + "root" + ".json",
            "~~struck/out~~ and about ~3/4 of the clips",
            "the option --" + "home" + "-team-name and the word end-" + "Users" + "-guide",
            "about ~n/2 frames and an error of ~eps/1e-3",
            "GET /" + "users" + "/me is a route of another project",
        ]
        self.stage("paths.txt", "\n".join(content) + "\n")
        self.assert_clean(self.guard("--staged"))


class DenyListTest(GuardTestCase):
    """The local deny-list in the git folder."""

    def test_denied_text_is_refused_and_not_printed(self) -> None:
        self.set_deny_list()
        self.stage("notes.md", "first line\nfrom the ACME-internal wiki\n")
        result = self.guard("--staged")
        lines = self.assert_refused(result, "deny")
        expected = "notes.md:2: deny: content matches the entry on line 3 of the local deny-list"
        self.assertEqual(lines, [expected])
        self.assert_absent(DENIED, result)

    def test_denied_path_is_refused_and_masked(self) -> None:
        self.set_deny_list()
        self.stage("docs/acme-internal-plan.md", "text\n")
        result = self.guard("--staged")
        lines = self.assert_refused(result, "deny")
        expected = (
            "docs/***-plan.md:-: deny: path matches the entry on line 3 of the local deny-list"
        )
        self.assertEqual(lines, [expected])
        self.assert_absent(DENIED, result)

    def test_denied_author_is_refused(self) -> None:
        self.set_deny_list()
        self.git("config", "--local", "user.name", "Tester at " + DENIED)
        self.stage("notes.md", "text\n")
        result = self.guard("--staged")
        lines = self.assert_refused(result, "deny")
        subjects = [line.split(" matches ")[0] for line in lines]
        self.assertEqual(
            subjects,
            [
                "-:-: deny: the author of the next commit",
                "-:-: deny: the committer of the next commit",
            ],
        )
        self.assert_absent(DENIED, result)

    def test_denied_identity_and_message_of_a_commit_are_refused(self) -> None:
        self.stage("notes.md", "text\n")
        self.git("config", "--local", "user.name", "Tester at " + DENIED)
        name = self.commit("Notes from the " + DENIED + " wiki")
        self.set_deny_list()
        result = self.guard("--push")
        lines = self.assert_refused(result, "deny")
        entry = "matches the entry on line 3 of the local deny-list"
        self.assertEqual(
            lines,
            [
                f"-:1: deny: content {entry} [message of commit {name}]",
                f"-:-: deny: the committer of commit {name} {entry}",
            ],
        )
        self.assert_absent(DENIED, result)

    def test_denied_host_is_masked_in_the_host_message(self) -> None:
        self.set_deny_list()
        self.stage("notes.md", WEB + "git.acme-internal.example/x\n")
        result = self.guard("--staged")
        self.assertEqual(len(self.assert_refused(result, "host")), 1)
        self.assertEqual(len(self.assert_refused(result, "deny")), 1)
        self.assert_absent(DENIED, result)

    def test_paths_mode_uses_the_deny_list(self) -> None:
        self.set_deny_list()
        path = self.write("notes.md", "acme-internal\n")
        self.assert_refused(self.guard("--paths", str(path)), "deny")

    def test_missing_deny_list_is_named_in_every_mode(self) -> None:
        """Without the list the [deny] rule is off. The guard must say so, on the error stream."""
        path = self.write("notes.md", "from the " + DENIED + " wiki\n")
        self.git("add", "--", "notes.md")
        for arguments in (["--staged"], ["--tracked"], ["--push"], ["--paths", str(path)]):
            result = self.guard(*arguments)
            self.assert_clean(result)
            self.assertEqual(result.stderr.count(NO_DENY_LIST), 1, result.stderr)
            self.assertIn("docs/RUNBOOK.md", result.stderr)

    def test_deny_list_without_entries_gives_no_notice(self) -> None:
        (self.repo / ".git" / "guard-deny.txt").write_text("# no entry yet\n", encoding="utf-8")
        self.stage("notes.md", "text\n")
        result = self.guard("--staged")
        self.assert_clean(result)
        self.assertEqual(result.stderr, "")

    def test_deny_list_in_the_work_tree_is_refused_and_not_read(self) -> None:
        self.stage("guard-deny.txt", DENY_LIST)
        self.stage("notes.md", "from the " + DENIED + " wiki\n")
        result = self.guard("--staged")
        lines = self.assert_refused(result, "deny")
        expected = (
            "guard-deny.txt:-: deny: the local deny-list must never be tracked; "
            "keep it in the git folder"
        )
        self.assertEqual(lines, [expected])
        self.assertIn(NO_DENY_LIST, result.stderr)


class IdentityRuleTest(GuardTestCase):
    """The [identity] rule for the next commit. The commits to push are in PushTest."""

    def remove_local_email(self) -> None:
        """Remove the local e-mail and leave an e-mail in the user configuration only."""
        self.git("config", "--local", "--unset", "user.email")
        user_config = "[user]\n\tname = Global Name\n\temail = " + OTHER_EMAIL + "\n"
        (self.home / ".gitconfig").write_text(user_config, encoding="utf-8")

    def test_missing_local_email_is_refused(self) -> None:
        self.stage("notes.md", "text\n")
        self.remove_local_email()
        for mode in ("--staged", "--push"):
            lines = self.assert_refused(self.guard(mode), "identity")
            self.assertEqual(len(lines), 1)
            self.assertTrue(lines[0].startswith("-:-: identity: user.email is not set"), lines)
            self.assertIn("git config --local user.email", lines[0])

    def test_missing_local_name_is_refused(self) -> None:
        """Without a local name git takes the name of the user account of the machine."""
        self.stage("notes.md", "text\n")
        self.git("config", "--local", "--unset", "user.name")
        user_config = "[user]\n\tname = Name Of The Account\n"
        (self.home / ".gitconfig").write_text(user_config, encoding="utf-8")
        for mode in ("--staged", "--push"):
            result = self.guard(mode)
            lines = self.assert_refused(result, "identity")
            self.assertEqual(len(lines), 1)
            self.assertTrue(lines[0].startswith("-:-: identity: user.name is not set"), lines)
            self.assertIn("git config --local user.name", lines[0])
            self.assert_absent("Name Of The Account", result)
        self.git("config", "--local", "--unset", "user.email")
        fields = [line.split(" ")[2] for line in self.guard("--staged").stdout.splitlines()]
        self.assertEqual(fields, ["user.name", "user.email"])

    def test_other_author_name_is_refused_for_the_next_commit(self) -> None:
        self.stage("notes.md", "text\n")
        result = self.guard("--staged", env=dict(self.env, GIT_AUTHOR_NAME=OTHER_AUTHOR))
        lines = self.assert_refused(result, "identity")
        expected = "-:-: identity: the author name of the next commit is not the local user.name"
        self.assertEqual(lines, [expected])
        self.assert_absent(OTHER_AUTHOR, result)

    def test_local_name_is_compared_as_git_writes_it(self) -> None:
        """Git removes quotes around a name before it writes the name into a commit."""
        self.git("config", "--local", "user.name", f'"{AUTHOR} Jr."')
        self.stage("notes.md", "text\n")
        self.assert_clean(self.guard("--staged"))
        self.git("-c", "commit.gpgsign=false", "commit", "--quiet", "--no-verify", "-m", "notes")
        self.assertNotIn('"', self.git("log", "-1", "--format=%an"))
        self.assert_clean(self.guard("--push"))

    def test_tracked_mode_and_paths_mode_have_no_identity_rule(self) -> None:
        path = self.write("notes.md", "text\n")
        self.git("add", "--", "notes.md")
        self.remove_local_email()
        self.assert_clean(self.guard("--paths", str(path)))
        self.assert_clean(self.guard("--tracked"))

    def test_other_author_email_is_refused_for_the_next_commit(self) -> None:
        self.stage("notes.md", "text\n")
        result = self.guard("--staged", env=dict(self.env, GIT_AUTHOR_EMAIL=OTHER_EMAIL))
        lines = self.assert_refused(result, "identity")
        expected = "-:-: identity: the author e-mail of the next commit is not the local user.email"
        self.assertEqual(lines, [expected])
        self.assert_absent(OTHER_EMAIL, result)

    def test_other_committer_email_is_refused_for_the_next_commit(self) -> None:
        self.stage("notes.md", "text\n")
        result = self.guard("--staged", env=dict(self.env, GIT_COMMITTER_EMAIL=OTHER_EMAIL))
        lines = self.assert_refused(result, "identity")
        expected = (
            "-:-: identity: the committer e-mail of the next commit is not the local user.email"
        )
        self.assertEqual(lines, [expected])
        self.assert_absent(OTHER_EMAIL, result)

    def test_identity_is_checked_without_a_staged_file(self) -> None:
        result = self.guard("--staged", env=dict(self.env, GIT_AUTHOR_EMAIL=OTHER_EMAIL))
        self.assertEqual(len(self.assert_refused(result, "identity")), 1)


class PushTest(GuardTestCase):
    """The push mode reads the commits and the tags that the remote does not have yet."""

    KEY_LINE = "leak.txt:1: secret: Google API key (starts with AIza...)"

    def commit_key(self, message: str = "add notes") -> str:
        """Commit a file that holds a key, and return the short name of the commit."""
        self.stage("leak.txt", "value " + GOOGLE_KEY + "\n")
        return self.commit(message)

    def test_key_in_a_commit_is_found_when_the_working_tree_is_clean(self) -> None:
        name = self.commit_key()
        self.write("leak.txt", "clean now\n")
        expected = [f"{self.KEY_LINE} [commit {name}]"]
        self.assertEqual(self.assert_refused(self.guard("--push"), "secret"), expected)
        self.git("add", "--", "leak.txt")
        self.assertEqual(self.assert_refused(self.guard("--push"), "secret"), expected)
        (self.repo / "leak.txt").unlink()
        self.assertEqual(self.assert_refused(self.guard("--push"), "secret"), expected)

    def test_key_that_a_later_commit_removed_is_found(self) -> None:
        name = self.commit_key()
        self.git("rm", "--quiet", "leak.txt")
        self.commit("remove notes")
        lines = self.assert_refused(self.guard("--push"), "secret")
        self.assertEqual(lines, [f"{self.KEY_LINE} [commit {name}]"])

    def test_commit_on_another_branch_is_found(self) -> None:
        self.stage("notes.md", "text\n")
        self.commit("base")
        main = self.branch()
        self.git("checkout", "--quiet", "-b", "side")
        name = self.commit_key()
        self.git("checkout", "--quiet", main)
        lines = self.assert_refused(self.guard("--push"), "secret")
        self.assertEqual(lines, [f"{self.KEY_LINE} [commit {name}]"])

    def test_commit_on_a_detached_head_and_the_pushed_commit_are_found(self) -> None:
        self.stage("notes.md", "text\n")
        self.commit("base")
        main = self.branch()
        self.git("checkout", "--quiet", "--detach")
        name = self.commit_key()
        commit = self.git("rev-parse", "HEAD").strip()
        self.assertEqual(len(self.assert_refused(self.guard("--push"), "secret")), 1)
        # No branch, no tag and no HEAD reaches the commit now. A push can still send it by name.
        self.git("checkout", "--quiet", main)
        self.assert_clean(self.guard("--push"))
        result = self.guard("--push", env=dict(self.env, **{PUSHED_COMMIT_VARIABLE: commit}))
        lines = self.assert_refused(result, "secret")
        self.assertEqual(lines, [f"{self.KEY_LINE} [commit {name}]"])

    def test_commits_on_the_remote_are_not_checked(self) -> None:
        self.stage("one.md", "text\n")
        self.commit("pushed commit", author_email=OTHER_EMAIL)
        self.commit_key()
        self.assert_refused(self.guard("--push"), "identity")
        self.add_remote()
        self.git("push", "--quiet", "origin", self.branch())
        self.assert_clean(self.guard("--push"))
        self.stage("two.md", "text\n")
        name = self.commit("new commit", author_email=OTHER_EMAIL)
        lines = self.guard("--push").stdout.splitlines()
        expected = f"-:-: identity: the author e-mail of commit {name} is not the local user.email"
        self.assertEqual(lines, [expected])

    def test_commit_without_a_remote_tracking_branch_is_checked(self) -> None:
        self.stage("one.md", "text\n")
        self.commit("pushed commit", author_email=OTHER_EMAIL)
        self.add_remote()
        self.git("push", "--quiet", "origin", self.branch())
        self.git("update-ref", "-d", "refs/remotes/origin/" + self.branch())
        self.assertEqual(len(self.assert_refused(self.guard("--push"), "identity")), 1)

    def test_only_the_remote_of_the_push_counts(self) -> None:
        self.commit_key()
        self.add_remote("origin")
        unnamed = self.add_remote("backup")
        self.git("push", "--quiet", "origin", self.branch())
        # A run by hand: the commit is on a remote. A push to origin: origin has it.
        self.assert_clean(self.guard("--push"))
        self.assert_clean(self.guard("--push", env=dict(self.env, **{REMOTE_VARIABLE: "origin"})))
        # A push to the other remote, by its name or by its address, sends the commit.
        for remote in ("backup", str(unnamed)):
            result = self.guard("--push", env=dict(self.env, **{REMOTE_VARIABLE: remote}))
            self.assertEqual(len(self.assert_refused(result, "secret")), 1, remote)

    def test_merge_commit_is_checked_for_its_own_changes_only(self) -> None:
        self.stage("base.md", "base\n")
        self.commit("base")
        main = self.branch()
        self.add_remote()
        self.git("push", "--quiet", "origin", main)
        # A commit that only the remote has. Its file holds a word that the deny-list names later.
        self.git("checkout", "--quiet", "-b", "theirs")
        self.stage("theirs.md", "from the " + DENIED + " wiki\n")
        self.commit("their commit")
        self.git("push", "--quiet", "origin", "theirs:" + main)
        self.git("checkout", "--quiet", main)
        self.git("branch", "--quiet", "-D", "theirs")
        self.stage("ours.md", "ours\n")
        self.commit("our commit")
        merge = ["-c", "commit.gpgsign=false", "merge", "--quiet", "--no-edit", "origin/" + main]
        self.git(*merge)
        self.assertEqual(len(self.git("rev-list", "--parents", "-1", "HEAD").split()), 3)
        self.set_deny_list()
        self.assert_clean(self.guard("--push"))
        # A file that the merge commit itself adds is checked.
        self.stage("extra.md", "also from the " + DENIED + " wiki\n")
        amend = ["commit", "--quiet", "--no-verify", "--amend", "--no-edit"]
        self.git("-c", "commit.gpgsign=false", *amend)
        lines = self.assert_refused(self.guard("--push"), "deny")
        entry = "content matches the entry on line 3 of the local deny-list"
        self.assertEqual(lines, [f"extra.md:1: deny: {entry} [commit {self.short_name('HEAD')}]"])

    def test_file_rules_apply_to_a_file_that_a_later_commit_removed(self) -> None:
        self.stage("clip.mp4", b"\x00\x00\x00\x18ftypmp42")
        self.stage("Labels-GameState.json", "{}\n")
        name = self.commit("add files")
        self.git("rm", "--quiet", "clip.mp4", "Labels-GameState.json")
        self.commit("remove files")
        self.assert_clean(self.guard("--tracked"))
        result = self.guard("--push")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "Labels-GameState.json:-: dataset: label file of the dataset; "
                f"it stays on the run machine [commit {name}]",
                "clip.mp4:-: ext: file type '.mp4' is not on the list of allowed file types "
                f"[commit {name}]",
                f"clip.mp4:-: binary: file contains a NUL byte [commit {name}]",
            ],
        )

    def test_one_version_of_a_file_is_reported_once(self) -> None:
        name = self.commit_key()
        self.git("rm", "--quiet", "leak.txt")
        self.commit("remove notes")
        self.commit_key("add the same notes again")
        lines = self.assert_refused(self.guard("--push"), "secret")
        self.assertEqual(lines, [f"{self.KEY_LINE} [commit {name}]"])

    def test_message_of_a_commit_is_checked(self) -> None:
        self.stage("notes.md", "text\n")
        name = self.commit("Fix the path " + MAC_HOME + "project\n\nkey " + GOOGLE_KEY + "\n")
        result = self.guard("--push")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                f"-:1: {HOME_MESSAGE} [message of commit {name}]",
                f"-:3: secret: Google API key (starts with AIza...) [message of commit {name}]",
            ],
        )
        self.assert_absent(GOOGLE_KEY, result)

    def test_author_and_committer_of_every_commit_are_checked(self) -> None:
        self.stage("one.md", "text\n")
        self.commit("good commit")
        first = self.commit("empty commit", author_email=OTHER_EMAIL, allow_empty=True)
        self.stage("two.md", "text\n")
        second = self.commit("other committer", env=dict(self.env, GIT_COMMITTER_EMAIL=OTHER_EMAIL))
        result = self.guard("--push")
        lines = self.assert_refused(result, "identity")
        wrong = "is not the local user.email"
        self.assertEqual(
            lines,
            [
                f"-:-: identity: the author e-mail of commit {first} {wrong}",
                f"-:-: identity: the committer e-mail of commit {second} {wrong}",
            ],
        )
        self.assert_absent(OTHER_EMAIL, result)

    def test_author_with_another_name_is_refused(self) -> None:
        """`git commit --author` with the local e-mail and another name."""
        self.stage("one.md", "text\n")
        name = self.commit("other name", author_name=OTHER_AUTHOR)
        result = self.guard("--push")
        expected = f"-:-: identity: the author name of commit {name} is not the local user.name"
        self.assertEqual(result.stdout.splitlines(), [expected])
        self.assert_absent(OTHER_AUTHOR, result)

    def test_annotated_tag_is_checked(self) -> None:
        self.stage("one.md", "text\n")
        self.commit("good commit")
        self.add_remote()
        self.git("push", "--quiet", "origin", self.branch())
        self.git("tag", "plain")
        self.assert_clean(self.guard("--push"))
        tagger = dict(self.env, GIT_COMMITTER_NAME=OTHER_AUTHOR, GIT_COMMITTER_EMAIL=OTHER_EMAIL)
        self.git("tag", "-a", "freeze", "-m", "freeze\n\nkey " + GOOGLE_KEY, env=tagger)
        result = self.guard("--push")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "-:3: secret: Google API key (starts with AIza...) [message of tag freeze]",
                "-:-: identity: the tagger name of tag freeze is not the local user.name",
                "-:-: identity: the tagger e-mail of tag freeze is not the local user.email",
            ],
        )

    def test_annotated_tag_that_the_remote_has_is_still_checked(self) -> None:
        """Git does not record which tags a remote has. A tag of another clone is checked too."""
        self.stage("one.md", "text\n")
        self.commit("good commit")
        self.add_remote()
        tagger = dict(self.env, GIT_COMMITTER_EMAIL=OTHER_EMAIL)
        self.git("tag", "-a", "freeze", "-m", "freeze", env=tagger)
        self.git("push", "--quiet", "origin", self.branch(), "freeze")
        for variables in ({}, {REMOTE_VARIABLE: "origin"}):
            result = self.guard("--push", env=dict(self.env, **variables))
            expected = "-:-: identity: the tagger e-mail of tag freeze is not the local user.email"
            self.assertEqual(result.stdout.splitlines(), [expected])


class ModeTest(GuardTestCase):
    """What each mode reads, and the exit code for usage and git errors."""

    def count_git_processes(self, *arguments: str) -> int:
        """Run the guard with a stand-in `git` that counts its calls, and return the count."""
        log = self.base / "git-calls.log"
        log.write_text("", encoding="utf-8")
        tools = self.base / "counting-tools"
        tools.mkdir(exist_ok=True)
        real_git = shutil.which("git", path=self.env["PATH"])
        self.assertIsNotNone(real_git)
        count = f"echo call >> {shlex.quote(str(log))}"
        script = f'#!/bin/sh\n{count}\nexec {shlex.quote(str(real_git))} "$@"\n'
        (tools / "git").write_text(script, encoding="utf-8")
        (tools / "git").chmod(0o755)
        env = dict(self.env, PATH=str(tools) + os.pathsep + self.env["PATH"])
        self.assert_clean(self.guard(*arguments, env=env))
        return len(log.read_text(encoding="utf-8").splitlines())

    def test_staged_mode_reads_the_index(self) -> None:
        self.stage("leak.txt", "value " + GOOGLE_KEY + "\n")
        self.write("leak.txt", "clean now\n")
        self.stage("clean.txt", "clean\n")
        self.write("clean.txt", "value " + GOOGLE_KEY + "\n")
        self.write("untracked.txt", "value " + GOOGLE_KEY + "\n")
        lines = self.assert_refused(self.guard("--staged"), "secret")
        self.assertEqual([line.split(":")[0] for line in lines], ["leak.txt"])

    def test_renamed_file_is_checked_as_a_new_file(self) -> None:
        text = "".join(f"line {number} of the notes\n" for number in range(40))
        self.stage("notes.txt", text)
        self.commit("add notes")
        self.git("mv", "notes.txt", "renamed.txt")
        self.stage("renamed.txt", text + "value " + GOOGLE_KEY + "\n")
        status = self.git("diff", "--cached", "--name-status", "--find-renames")
        self.assertTrue(status.startswith("R"), status)
        lines = self.assert_refused(self.guard("--staged"), "secret")
        self.assertEqual(lines, ["renamed.txt:41: secret: Google API key (starts with AIza...)"])

    def test_link_that_becomes_a_file_is_checked(self) -> None:
        self.stage("notes.md", "text\n")
        (self.repo / "notes.txt").symlink_to("notes.md")
        self.git("add", "--", "notes.txt")
        self.commit("add a link")
        (self.repo / "notes.txt").unlink()
        self.stage("notes.txt", "data: " + MAC_HOME + "clips\n")
        self.assertEqual(self.git("diff", "--cached", "--name-status").split(), ["T", "notes.txt"])
        lines = self.assert_refused(self.guard("--staged"), "home")
        self.assertEqual(lines, [f"notes.txt:1: {HOME_MESSAGE}"])

    def test_staged_deletion_is_not_checked(self) -> None:
        self.stage("leak.txt", "value " + GOOGLE_KEY + "\n")
        self.commit("add notes")
        self.git("rm", "--quiet", "leak.txt")
        self.assert_clean(self.guard("--staged"))

    def test_unusual_file_names_and_a_submodule_are_read_in_every_mode(self) -> None:
        names = ("dir with space/n\u00f6tes.md", "-dash.txt", "tab\tname.txt")
        for name in names:
            self.stage(name, "data: " + MAC_HOME + "clips\n")
        # A submodule is an entry of the index without file content in this repository.
        self.git("update-index", "--add", "--cacheinfo", "160000," + "1" * 40 + ",vendor/sub.txt")
        expected = sorted(f"{name}:1" for name in names)
        self.assertEqual(self.places(self.guard("--staged"), "home"), expected)
        self.commit("unusual names")
        for mode in ("--push", "--tracked"):
            result = self.guard(mode)
            self.assertEqual(self.places(result, "home"), expected, mode)
            self.assertEqual(len(result.stdout.splitlines()), len(names), mode)

    def test_number_of_git_processes_does_not_grow_with_the_number_of_files(self) -> None:
        """One git process for each file made the check of a result commit last minutes."""
        for number in range(3):
            self.stage(f"results/first_{number}.json", "{}\n")
        few = self.count_git_processes("--staged")
        for number in range(60):
            self.stage(f"results/second_{number}.json", json.dumps({"number": number}))
        many = self.count_git_processes("--staged")
        self.assertEqual(few, many)
        self.assertLess(many, 15)
        self.commit("results")
        pushed = self.count_git_processes("--push")
        self.assertLess(pushed, 15)
        # Branches and tags do not add a process either: one listing names them all.
        for number in range(8):
            self.git("branch", f"topic-{number}")
            self.git("tag", "-a", f"mark-{number}", "-m", "A clean message")
        self.assertEqual(self.count_git_processes("--push"), pushed)

    def test_tracked_mode_reads_the_working_tree(self) -> None:
        self.stage("leak.txt", "clean\n")
        self.stage("clean.txt", "clean\n")
        self.commit("first commit")
        self.write("leak.txt", "value " + GOOGLE_KEY + "\n")
        self.write("untracked.txt", "value " + GOOGLE_KEY + "\n")
        lines = self.assert_refused(self.guard("--tracked"), "secret")
        self.assertEqual([line.split(":")[0] for line in lines], ["leak.txt"])

    def test_tracked_file_without_working_copy_keeps_its_path_rules(self) -> None:
        self.stage("clip.mp4", b"\x00\x00")
        self.stage("notes.md", "text\n")
        (self.repo / "clip.mp4").unlink()
        (self.repo / "notes.md").unlink()
        result = self.guard("--tracked")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertTrue(result.stdout.startswith("clip.mp4:-: ext: "), result.stdout)

    def test_tracked_mode_reports_a_conflicted_file_once(self) -> None:
        self.stage("paths.txt", "base\n")
        self.commit("base commit")
        main_branch = self.branch()
        self.git("checkout", "--quiet", "-b", "side")
        self.stage("paths.txt", MAC_HOME + "side\n")
        self.commit("side commit")
        self.git("checkout", "--quiet", main_branch)
        self.stage("paths.txt", MAC_HOME + "main\n")
        self.commit("main commit")
        self.git("merge", "side", exit_code=1)
        lines = self.assert_refused(self.guard("--tracked"), "home")
        self.assertEqual(len(lines), 2)
        # The index holds no version of the file that a commit could take.
        self.assert_clean(self.guard("--staged"))

    def test_paths_mode_accepts_relative_paths(self) -> None:
        self.write("docs/notes.md", MAC_HOME + "x\n")
        command = [sys.executable, str(GUARD), "--paths", "notes.md"]
        result = subprocess.run(
            command,
            cwd=self.repo / "docs",
            env=self.env,
            capture_output=True,
            encoding="utf-8",
            check=False,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertTrue(result.stdout.startswith("docs/notes.md:1: home: "), result.stdout)

    def test_missing_mode_is_a_usage_error(self) -> None:
        result = self.guard()
        self.assertEqual((result.returncode, result.stdout), (2, ""))
        self.assertIn("usage:", result.stderr)

    def test_folder_without_repository_is_an_error(self) -> None:
        result = self.guard("--staged", repo=self.home)
        self.assertEqual((result.returncode, result.stdout), (2, ""))
        self.assertIn("guard: error: git rev-parse", result.stderr)

    def test_path_outside_the_repository_is_an_error(self) -> None:
        outside = self.home / "notes.md"
        outside.write_text("text\n", encoding="utf-8")
        result = self.guard("--paths", str(outside))
        self.assertEqual((result.returncode, result.stdout), (2, ""))
        self.assertIn("outside the repository", result.stderr)

    def test_missing_path_is_an_error(self) -> None:
        result = self.guard("--paths", str(self.repo / "missing.md"))
        self.assertEqual((result.returncode, result.stdout), (2, ""))
        self.assertIn("is not a file", result.stderr)


class HookSettingsTest(GuardTestCase):
    """The hook settings: which checks run before a commit and before a push, and how."""

    def copy_guard_files(self) -> None:
        """Commit the guard, the host list and the hook settings into the test repository."""
        for source in (GUARD, ALLOW_LIST, HOOK_FILE, SECRET_SCAN_FILE):
            self.stage(source.relative_to(REPO).as_posix(), source.read_bytes())
        self.commit("guard files")

    def run_hooks(
        self, hook_type: str, pushed: str = "", env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run the hooks of one type as an installed git hook does.

        `pushed` is the input that git gives to a push hook.
        """
        command = [
            "pre-commit",
            "hook-impl",
            f"--config={HOOK_FILE.name}",
            f"--hook-type={hook_type}",
        ]
        command += ["--hook-dir", ".git/hooks", "--"]
        if hook_type == "pre-push":
            command += ["origin", str(self.base / "origin.git")]
        return subprocess.run(
            command,
            cwd=self.repo,
            env=env or self.env,
            input=pushed,
            capture_output=True,
            encoding="utf-8",
            check=False,
        )

    def pushed_line(self, branch: str, remote_commit: str) -> str:
        """Return the line that git gives a push hook for one branch."""
        commit = self.git("rev-parse", branch).strip()
        return f"refs/heads/{branch} {commit} refs/heads/{branch} {remote_commit}\n"

    def need_tools(self, *tools: str) -> None:
        """Skip the test on a machine without the hook runner or the secret scanner."""
        missing = [tool for tool in tools if shutil.which(tool) is None]
        if missing:
            self.skipTest(f"not installed: {', '.join(missing)}")

    def test_every_hook_always_runs(self) -> None:
        """Without `always_run` the hook runner skips a hook when no changed file is on disk."""
        blocks = re.split(r"(?m)^ +- id: ", HOOK_FILE.read_text(encoding="utf-8"))[1:]
        self.assertEqual(len(blocks), 4)
        for block in blocks:
            self.assertRegex(block, r"(?m)^ +always_run: true$")
            self.assertRegex(block, r"(?m)^ +pass_filenames: false$")

    def test_guard_runs_in_the_staged_mode_and_in_the_push_mode(self) -> None:
        entries = hook_entries()
        self.assertEqual(entries["guard-staged"], "python3 tools/guard.py --staged")
        self.assertEqual(entries["guard-push"], "python3 tools/guard.py --push")

    def test_secret_scan_shows_the_place_and_hides_the_secret(self) -> None:
        entries = hook_entries()
        for hook in ("gitleaks", "gitleaks-push"):
            words = shlex.split(entries[hook])
            for option in ("--redact", "--verbose", "--no-banner"):
                self.assertIn(option, words, hook)
            self.assertEqual(words[words.index("--config") + 1], SECRET_SCAN_FILE.name)

    def test_secret_scan_names_the_file_of_a_key_in_the_new_format(self) -> None:
        self.need_tools("gitleaks")
        self.stage(SECRET_SCAN_FILE.name, SECRET_SCAN_FILE.read_bytes())
        detail = "the request failed: 401 for the key "
        self.stage("results/metrics/doctor.json", json.dumps({"detail": detail + "<key>"}))
        command = shlex.split(hook_entries()["gitleaks"])

        def scan() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                command,
                cwd=self.repo,
                env=self.env,
                capture_output=True,
                encoding="utf-8",
                check=False,
            )

        self.assertEqual(scan().returncode, 0)
        self.stage("results/metrics/doctor.json", json.dumps({"detail": detail + NEW_GOOGLE_KEY}))
        found = scan()
        self.assertEqual(found.returncode, 1, found.stdout + found.stderr)
        self.assertIn("google-auth-key", found.stdout)
        self.assertIn("results/metrics/doctor.json", found.stdout)
        self.assert_absent(NEW_GOOGLE_KEY, found)

    def test_secret_scan_finds_a_key_after_an_escape_or_a_word_character(self) -> None:
        """In a JSON text the letter of an escape stands directly before the key."""
        self.need_tools("gitleaks")
        self.stage(SECRET_SCAN_FILE.name, SECRET_SCAN_FILE.read_bytes())
        command = shlex.split(hook_entries()["gitleaks"])
        detail = "the call failed with the key:\n"
        cases = {
            "escape.json": (json.dumps({"detail": detail + NEW_GOOGLE_KEY}), "google-auth-key"),
            "word.txt": ("GEMINI_KEY_" + NEW_GOOGLE_KEY + "\n", "google-auth-key"),
            "sk_escape.json": (json.dumps({"detail": detail + SK_KEY}), "generic-sk-key"),
            "openrouter.json": (json.dumps({"detail": detail + OPENROUTER_KEY}), "openrouter-key"),
            "openrouter.env": ("OPENROUTER_API_KEY=" + OPENROUTER_KEY + "\n", "openrouter-key"),
        }
        for name, (content, rule) in cases.items():
            self.stage(f"results/metrics/{name}", content)
            found = subprocess.run(
                command, cwd=self.repo, env=self.env, capture_output=True, encoding="utf-8"
            )
            self.assertEqual(found.returncode, 1, name + found.stdout + found.stderr)
            self.assertIn(rule, found.stdout, name)
            self.assertIn(f"results/metrics/{name}", found.stdout)
            self.assert_absent(NEW_GOOGLE_KEY, found)
            self.assert_absent(SK_KEY, found)
            self.assert_absent(OPENROUTER_KEY, found)
            self.git("rm", "--quiet", "--cached", f"results/metrics/{name}")
        # A word in capitals that ends with the letters of the prefix is not a key.
        self.stage("results/metrics/word.txt", "See F" + NEW_GOOGLE_KEY + " for more.\n")
        words = "A risk-adjusted-return-on-capital-employed figure.\n"
        self.stage("results/metrics/risk.txt", words)
        passed = subprocess.run(
            command, cwd=self.repo, env=self.env, capture_output=True, encoding="utf-8"
        )
        self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)

    def test_push_hook_checks_the_name_of_a_new_branch(self) -> None:
        """A push of a branch whose name holds a private word. Its commit is clean."""
        self.need_tools("pre-commit", "gitleaks")
        self.copy_guard_files()
        main = self.branch()
        self.add_remote()
        self.git("push", "--quiet", "origin", main)
        self.set_deny_list()
        self.git("checkout", "--quiet", "-b", "fix-acme-internal-proxy")
        self.commit("A clean commit", allow_empty=True)
        refused = self.run_hooks("pre-push", self.pushed_line("fix-acme-internal-proxy", NO_COMMIT))
        self.assertEqual(refused.returncode, 1, refused.stdout + refused.stderr)
        entry = "matches the entry on line 3 of the local deny-list"
        self.assertIn(f"-:-: deny: the name of branch fix-***-proxy {entry}", refused.stdout)
        self.assert_absent(DENIED, refused)
        # The same commit under another name on the remote: `git push origin HEAD:refs/heads/...`.
        self.git("branch", "--quiet", "-m", "topic")
        commit = self.git("rev-parse", "HEAD").strip()
        line = f"HEAD {commit} refs/heads/notes-acme-internal {NO_COMMIT}\n"
        refused = self.run_hooks("pre-push", line)
        self.assertEqual(refused.returncode, 1, refused.stdout + refused.stderr)
        self.assertIn(f"-:-: deny: the name of branch notes-*** {entry}", refused.stdout)
        passed = self.run_hooks("pre-push", self.pushed_line("topic", NO_COMMIT))
        self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)

    def test_push_hook_reads_the_commits_and_not_the_working_tree(self) -> None:
        self.need_tools("pre-commit", "gitleaks")
        self.copy_guard_files()
        main = self.branch()
        self.add_remote()
        self.git("push", "--quiet", "origin", main)
        base = self.git("rev-parse", "HEAD").strip()
        # A clean commit passes, and goes to the remote.
        self.stage("notes.md", "text\n")
        self.commit("notes")
        passed = self.run_hooks("pre-push", self.pushed_line(main, base))
        self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
        self.git("push", "--quiet", "origin", main)
        base = self.git("rev-parse", "HEAD").strip()
        # A file with a home-folder path is added in one commit and removed in the next. The
        # working tree then has no file of the push.
        self.stage("paths.txt", "data: " + MAC_HOME + "clips\n")
        name = self.commit("add paths")
        self.git("rm", "--quiet", "paths.txt")
        self.commit("remove paths")
        refused = self.run_hooks("pre-push", self.pushed_line(main, base))
        self.assertEqual(refused.returncode, 1, refused.stdout + refused.stderr)
        self.assertIn(f"paths.txt:1: {HOME_MESSAGE} [commit {name}]", refused.stdout)

    def test_push_hook_checks_another_branch_and_an_empty_commit(self) -> None:
        self.need_tools("pre-commit", "gitleaks")
        self.copy_guard_files()
        main = self.branch()
        self.add_remote()
        self.git("push", "--quiet", "origin", main)
        self.git("checkout", "--quiet", "-b", "side")
        name = self.commit("empty commit", author_email=OTHER_EMAIL, allow_empty=True)
        self.git("checkout", "--quiet", main)
        refused = self.run_hooks("pre-push", self.pushed_line("side", NO_COMMIT))
        self.assertEqual(refused.returncode, 1, refused.stdout + refused.stderr)
        expected = f"identity: the author e-mail of commit {name} is not the local user.email"
        self.assertIn(expected, refused.stdout)

    def test_commit_hook_checks_the_identity_of_an_empty_commit(self) -> None:
        self.need_tools("pre-commit", "gitleaks")
        self.copy_guard_files()
        passed = self.run_hooks("pre-commit")
        self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
        refused = self.run_hooks("pre-commit", env=dict(self.env, GIT_AUTHOR_EMAIL=OTHER_EMAIL))
        self.assertEqual(refused.returncode, 1, refused.stdout + refused.stderr)
        expected = "identity: the author e-mail of the next commit is not the local user.email"
        self.assertIn(expected, refused.stdout)


if __name__ == "__main__":
    unittest.main()
