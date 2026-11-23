"""The commands that the documents give must exist in the code.

The owner follows ``docs/HANDOVER.md`` and ``docs/RUNBOOK.md`` to the letter on the run
machine. This test reads every fenced block of ``CLAUDE.md`` and of these two files:

- A shell command that calls the package (``python -m pitchvlm.cli ...``), the tool script
  (``tools/run_tool.py ...``) or the guard (``tools/guard.py ...``) must parse with the argument
  parser of that program: the sub-command and every option exist, and every value has a form
  that the parser takes (a model name of ``--models``, a positive ``--max-clips``, ...).
- A file below ``configs/`` that such a command names must exist, and an experiment file or a
  settings file must load. The kind of a probe must be the probe of its experiment file. One
  file is written, not read: the split list of ``data --manifest``.
- A Python block may import from the package only names that the package has.
- No document under ``docs/`` and no ``CLAUDE.md`` holds a line of a merge conflict: the
  documents were merged by hand several times.

Placeholders such as ``<clip>`` and paths below the data folder are not checked: they do not
exist on the build machine. The test reads text and runs no command.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import importlib.util
import io
import re
import shlex
import textwrap
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import ModuleType

import pytest

from pitchvlm import cli
from pitchvlm.config import load_config

REPO = Path(__file__).resolve().parents[1]
DOCUMENTS = ("CLAUDE.md", "docs/HANDOVER.md", "docs/RUNBOOK.md")
#: A fenced block: its language word (may be empty) and its text.
FENCE = re.compile(r"^([ \t]*)```([A-Za-z]*)[^\n]*\n(.*?)^[ \t]*```", re.M | re.S)
SHELL_LANGUAGES = ("bash", "sh", "shell", "")
#: An assignment of a shell variable on a line of its own, for example ``CLIP=~/pitchvlm-data/...``.
ASSIGNMENT = re.compile(r"^([A-Z][A-Z0-9_]*)=(\S+)(?:\s+#.*)?$")
VARIABLE = re.compile(r"\$\{?([A-Z][A-Z0-9_]*)\}?")
#: Shell words that end one command and start the next one.
SEPARATORS = ("&&", "||", "|", ";")
#: A file of the repository that a command names.
CONFIG_FILE = re.compile(r"^configs/[\w./-]+\.(?:yaml|json)$")
#: A line that git writes into a file with a merge conflict.
CONFLICT_LINE = re.compile(r"^(?:<{7}|={7}|>{7}|\|{7})(?: |$)", re.M)


def _load_module(name: str, path: Path) -> ModuleType:
    """Load a script of ``tools/`` as a module; it is not a part of a package."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


def _argparse_problem(parse: Callable[[list[str]], object], arguments: list[str]) -> str | None:
    """Parse the arguments; return the error text of the parser, or None when they parse."""
    errors = io.StringIO()
    try:
        with contextlib.redirect_stderr(errors), contextlib.redirect_stdout(io.StringIO()):
            parse(arguments)
    except SystemExit as stop:
        if stop.code in (0, None):
            return None
        return errors.getvalue().strip().splitlines()[-1] if errors.getvalue().strip() else f"exit {stop.code}"
    except Exception as error:  # The guard raises its own error instead of leaving.
        return f"{type(error).__name__}: {str(error).splitlines()[0]}"
    return None


def _parse_package(arguments: list[str]) -> object:
    return cli.build_parser().parse_args(arguments)


def _parse_run_tool(arguments: list[str]) -> object:
    return _load_module("run_tool_of_the_documents", REPO / "tools" / "run_tool.py").build_parser().parse_args(arguments)


def _parse_guard(arguments: list[str]) -> object:
    return _load_module("guard_of_the_documents", REPO / "tools" / "guard.py").parse_arguments(arguments)


#: The programs whose calls are checked: the word that names the program, and its parser.
PROGRAMS: dict[str, Callable[[list[str]], object]] = {
    "pitchvlm.cli": _parse_package,
    "tools/run_tool.py": _parse_run_tool,
    "tools/guard.py": _parse_guard,
}


def fenced_blocks(text: str) -> Iterator[tuple[str, str]]:
    """Yield the language word and the text of every fenced block of a Markdown text."""
    for match in FENCE.finditer(text):
        yield match.group(2).lower(), textwrap.dedent(match.group(3))


def shell_commands(block: str) -> Iterator[list[str]]:
    """Yield the words of every command of a shell block, with its variables replaced.

    Lines that end with a backslash are joined. A comment line is left out. A variable that
    the block assigns on a line of its own is replaced by its value; another one stays.
    """
    joined = re.sub(r"\\\n\s*", " ", block)
    variables: dict[str, str] = {}
    for line in joined.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assignment = ASSIGNMENT.match(line)
        if assignment:
            variables[assignment.group(1)] = assignment.group(2)
            continue
        line = VARIABLE.sub(lambda found: variables.get(found.group(1), found.group(0)), line)
        try:
            words = shlex.split(line, comments=True)
        except ValueError:
            continue  # Not a command line, for example an example of a file.
        command: list[str] = []
        for word in words:
            if word in SEPARATORS:
                if command:
                    yield command
                command = []
            else:
                command.append(word)
        if command:
            yield command


def program_call(words: Sequence[str]) -> tuple[str, list[str]] | None:
    """Return the program and its arguments when a command calls the package, the tool script or the guard."""
    for position, word in enumerate(words):
        if word == "pitchvlm.cli" and position > 0 and words[position - 1] == "-m":
            return "pitchvlm.cli", list(words[position + 1 :])
        if word in ("tools/run_tool.py", "tools/guard.py"):
            return word, list(words[position + 1 :])
    return None


def _option_values(arguments: Sequence[str], option: str) -> list[str]:
    return [arguments[i + 1] for i, word in enumerate(arguments[:-1]) if word == option]


def file_problems(arguments: list[str]) -> list[str]:
    """Return what is wrong with the files below ``configs/`` that a call of the package names."""
    problems: list[str] = []
    command = arguments[0] if arguments else ""
    written = set(_option_values(arguments, "--manifest")) if command == "data" else set()
    for word in arguments:
        if CONFIG_FILE.match(word) and word not in written and not (REPO / word).is_file():
            problems.append(f"the file {word} does not exist")
    if problems:
        return problems
    source = arguments[1] if command in ("probe", "overlay") and len(arguments) > 1 else None
    for value in _option_values(arguments, "--experiment"):
        try:
            if command == "probe" or (command == "overlay" and source == "probe"):
                experiment = cli.load_experiment(REPO / value)
                if command == "probe" and experiment.probe != source:
                    problems.append(f"probe {source} names the experiment file {value} of the probe {experiment.probe}")
            else:
                cli.load_tool_experiment(REPO / value)
        except cli.CliError as error:
            problems.append(str(error))
    for value in _option_values(arguments, "--config"):
        try:
            load_config(REPO / value)
        except (OSError, ValueError) as error:
            problems.append(f"the settings file {value} does not load: {error}")
    return problems


def import_problems(block: str) -> list[str]:
    """Return the names that a Python block imports from the package and that the package does not have."""
    try:
        tree = ast.parse(block)
    except SyntaxError as error:
        return [f"the Python block does not parse: {error}"]
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "pitchvlm":
            try:
                module = importlib.import_module(node.module)
            except ImportError as error:
                problems.append(f"the module {node.module} cannot be imported: {error}")
                continue
            problems += [
                f"{node.module} has no {alias.name}" for alias in node.names if not hasattr(module, alias.name)
            ]
    return problems


def command_problems(text: str) -> list[str]:
    """Return every problem of the commands of a Markdown text, each with the command that has it."""
    problems: list[str] = []
    for language, block in fenced_blocks(text):
        if language == "python":
            problems += [f"python block: {problem}" for problem in import_problems(block)]
            continue
        if language not in SHELL_LANGUAGES:
            continue
        for words in shell_commands(block):
            call = program_call(words)
            if call is None:
                continue
            program, arguments = call
            shown = " ".join([program, *arguments])
            parse_problem = _argparse_problem(PROGRAMS[program], arguments)
            if parse_problem is not None:
                problems.append(f"{shown}: {parse_problem}")
            elif program == "pitchvlm.cli":
                problems += [f"{shown}: {problem}" for problem in file_problems(arguments)]
    return problems


def documented_calls(text: str) -> list[tuple[str, list[str]]]:
    """Return the calls of the package, the tool script and the guard in the shell blocks of a text."""
    return [
        call
        for language, block in fenced_blocks(text)
        if language in SHELL_LANGUAGES
        for words in shell_commands(block)
        if (call := program_call(words)) is not None
    ]


@pytest.mark.parametrize("document", DOCUMENTS)
def test_every_documented_command_exists_in_the_code(document: str) -> None:
    text = (REPO / document).read_text(encoding="utf-8")
    assert documented_calls(text), f"{document} has no command of the package: the reader of the blocks is broken"
    assert command_problems(text) == []


def test_documents_call_every_command_of_the_package() -> None:
    """Each sub-command of the package has at least one documented command line, so each one is checked."""
    texts = "\n".join((REPO / document).read_text(encoding="utf-8") for document in DOCUMENTS)
    called = {arguments[0] for program, arguments in documented_calls(texts) if program == "pitchvlm.cli"}
    commands = set(cli.build_parser()._subparsers._group_actions[0].choices)  # type: ignore[union-attr]
    assert commands - called == set()


PLANTED = """
```bash
# A sub-command that the package does not have.
python -m pitchvlm.cli tool-run --experiment configs/experiments/people_probe_rfdetr_small.yaml \\
    --root ~/pitchvlm-data --out ~/pitchvlm-data/jobs/x.json
# An option that the command does not have.
python -m pitchvlm.cli probe pitch --experiment configs/experiments/pitch_probe.yaml \\
    --root ~/pitchvlm-data --max-frames 4 --out ../pitchvlm-results/results/metrics/x.json
# An experiment file that does not exist.
SETTINGS=configs/engines/detector_only.yaml
python -m pitchvlm.cli clip-job --tool rfdetr --config $SETTINGS --clip ~/pitchvlm-data/videos/<clip> \\
    --root ~/pitchvlm-data --out ~/pitchvlm-data/jobs/<clip>/rfdetr.json
```
"""

CLEAN = """
```bash
SETTINGS=configs/engines/detector_calibration.yaml
python -m pitchvlm.cli clip-job --tool rfdetr --config $SETTINGS --clip ~/pitchvlm-data/videos/<clip> \\
    --root ~/pitchvlm-data --out ~/pitchvlm-data/jobs/<clip>/rfdetr.json
python -m pitchvlm.cli data --root ~/pitchvlm-data --split valid --manifest configs/splits.json --seed 20261113
env -u GEMINI_API_KEY ~/pitchvlm-data/envs/rfdetr/bin/python tools/run_tool.py rfdetr --device mps \\
    --job ~/pitchvlm-data/jobs/x.json --root ~/pitchvlm-data --out x.json
```
"""


def test_reader_finds_a_missing_command_a_missing_option_and_a_missing_file() -> None:
    problems = command_problems(PLANTED)
    assert len(problems) == 3, problems
    assert "invalid choice: 'tool-run'" in problems[0]
    assert "unrecognized arguments: --max-frames 4" in problems[1]
    assert "the file configs/engines/detector_only.yaml does not exist" in problems[2]
    assert command_problems(CLEAN) == []


def test_reader_finds_a_probe_with_the_experiment_file_of_another_probe_and_a_wrong_option_value() -> None:
    text = """
```bash
python -m pitchvlm.cli probe people --experiment configs/experiments/pitch_probe.yaml --root d --out m.json
python -m pitchvlm.cli doctor --models gemini-4-argon,gemini-4-argon
python3 tools/guard.py --everything
```
"""
    problems = command_problems(text)
    assert len(problems) == 3, problems
    assert "of the probe pitch" in problems[0]
    assert "--models" in problems[1]
    assert "guard.py --everything" in problems[2]


def test_reader_finds_an_import_that_the_package_does_not_have() -> None:
    text = "```python\nfrom pitchvlm.labels import load_gsr_labels, read_every_label\n```\n"
    assert command_problems(text) == ["python block: pitchvlm.labels has no read_every_label"]


def test_documents_hold_no_line_of_a_merge_conflict() -> None:
    documents = [REPO / "CLAUDE.md", *sorted((REPO / "docs").glob("*.md"))]
    found = [
        f"{path.relative_to(REPO)}:{text[: match.start()].count(chr(10)) + 1}"
        for path in documents
        for text in [path.read_text(encoding="utf-8")]
        for match in CONFLICT_LINE.finditer(text)
    ]
    assert found == []
