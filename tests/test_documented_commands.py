"""The commands that the documents give must exist in the code.

The owner follows ``docs/HANDOVER.md`` and ``docs/RUNBOOK.md`` to the letter on the run
machine. This test reads every fenced block of ``CLAUDE.md`` and of these two files:

- A shell command that calls the package (``python -m pitchvlm.cli ...``), the tool script
  (``tools/run_tool.py ...``), the guard (``tools/guard.py ...``) or the test runner
  (``tools/run_tests.py ...``) must parse with the argument parser of that program: the sub-command and every option exist, and every value has a form
  that the parser takes (a model name of ``--models``, a positive ``--max-clips``, ...).
- A file below ``configs/`` that such a command names must exist, and an experiment file or a
  settings file must load. The kind of a probe must be the probe of its experiment file. One
  file is written, not read: the split list of ``data --manifest``.
- A Python block may import from the package only names that the package has.
- No document under ``docs/`` and no ``CLAUDE.md`` holds a line of a merge conflict: the
  documents were merged by hand several times.

The text around the blocks names commands, options and files too, in inline code (between
backticks). The owner reads it as an order as well, so it is checked:

- A file of the repository that the text names (``configs/...``, ``tools/run_tool.py``,
  ``vlm/cost.py`` for a module of the package, or an experiment file by its bare name such as
  ``people_probe_video_low.yaml``) must exist, and an experiment file must load. Two files are
  written on the run machine and are not checked: the split list and the pinned list.
- A command in inline code (``doctor --video``, ``clip-job --tool dfine``,
  ``tools/run_tool.py dfine``) must name a sub-command and options that its parser has, with
  values that the parser takes.
- An option in inline code (``--video-dir``, ``--delay clip``) of the sections 11 to 14 of the
  RUNBOOK (detector engine, video input, streaming path) and of section 6 of the HANDOVER (the
  first session) must be an option of a command of the package or of the tool script.
- A reference to a section of the RUNBOOK or of the HANDOVER (``RUNBOOK, section 13``) must name
  a section that exists, and a quoted title after it must be a heading of that section. The
  sections of the RUNBOOK are numbered 1, 2, 3, ... without a gap.

Placeholders such as ``<clip>`` and paths below the data folder are not checked: they do not
exist on the build machine. The test reads text and runs no command.
"""

from __future__ import annotations

import argparse
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


def _parse_run_tests(arguments: list[str]) -> object:
    return _load_module("run_tests_of_the_documents", REPO / "tools" / "run_tests.py").parse_options(arguments)


#: The programs whose calls are checked: the word that names the program, and its parser.
PROGRAMS: dict[str, Callable[[list[str]], object]] = {
    "pitchvlm.cli": _parse_package,
    "tools/run_tool.py": _parse_run_tool,
    "tools/guard.py": _parse_guard,
    "tools/run_tests.py": _parse_run_tests,
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
    """Return the program and its arguments when a command calls the package, a tool script, the guard or the test runner."""
    for position, word in enumerate(words):
        if word == "pitchvlm.cli" and position > 0 and words[position - 1] == "-m":
            return "pitchvlm.cli", list(words[position + 1 :])
        if word in ("tools/run_tool.py", "tools/guard.py", "tools/run_tests.py"):
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


def test_reader_finds_an_option_that_the_test_runner_does_not_pass_on() -> None:
    text = """
```bash
python tools/run_tests.py -m "slow or not slow"
python tools/run_tests.py -x -k guard tests/test_guard.py
python tools/run_tests.py --tb=short
python tools/run_tests.py -j 0
```
"""
    problems = command_problems(text)
    assert len(problems) == 2, problems
    assert "unrecognized arguments: --tb=short" in problems[0]
    assert "-j must be 1 or more" in problems[1]


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


# --- Commands, options, files and sections named in the text around the blocks ----------------

#: Inline code of a Markdown text (between single backticks).
INLINE = re.compile(r"`([^`\n]+)`")
#: A numbered section of a document: ``## 11 Detector engine ...``.
SECTION_HEADING = re.compile(r"^## (\d+) ", re.M)
#: Files that the run machine writes. They are not in the repository on the build machine.
WRITTEN_ON_THE_RUN_MACHINE = frozenset({"configs/splits.json", "locks/run-machine.txt"})
#: A file of the repository named by its path from the top folder.
REPOSITORY_PATH = re.compile(
    r"^(?:(?:configs|docs|locks|src|tests|tools)/[\w./-]+\.\w+"
    r"|CLAUDE\.md|pyproject\.toml|\.gitleaks\.toml|\.pre-commit-config\.yaml|\.gitignore)$"
)
#: A module of the package named by its path below ``src/pitchvlm``.
PACKAGE_MODULE = re.compile(r"^(?:vlm|evalx|fallbacks)/[\w/]+\.py$")
#: An experiment file named by its bare name.
EXPERIMENT_NAME = re.compile(r"^(?:people|pitch|number)_probe\w*\.yaml$")
#: Options of other programs (uv) that the checked sections name in inline code.
FOREIGN_OPTIONS = frozenset({"--exclude-newer"})
#: The sections whose inline options are checked: RUNBOOK 11 to 14, HANDOVER 6.
OPTION_SECTIONS = {"docs/RUNBOOK.md": (11, 12, 13, 14, 15), "docs/HANDOVER.md": (6,)}
#: A reference to a section of the RUNBOOK or the HANDOVER, with an optional quoted title after it.
SECTION_REFERENCE = re.compile(
    r"(?P<document>RUNBOOK|HANDOVER)(?:\.md`?)?,?\s+sections?\s+(?P<number>\d+)(?:\.(?P<part>\d+))?"
    r"(?:\s+(?:to|and)\s+(?P<last>\d+))?(?:,\s+\"(?P<title>[^\"]+)\")?"
)
#: The same reference with the document after the number: ``section 13 of the RUNBOOK``.
SECTION_REFERENCE_AFTER = re.compile(
    r"sections?\s+(?P<number>\d+)(?:\s+(?:to|and)\s+(?P<last>\d+))?\s+of\s+(?:the\s+)?`?(?:docs/)?"
    r"(?P<document>RUNBOOK|HANDOVER)"
)
REFERENCED = {"RUNBOOK": "docs/RUNBOOK.md", "HANDOVER": "docs/HANDOVER.md"}


def prose(text: str) -> str:
    """Return a Markdown text without its fenced blocks."""
    return FENCE.sub("", text)


def inline_spans(text: str) -> list[str]:
    """Return the inline code of a Markdown text, outside its fenced blocks."""
    return [span.strip() for span in INLINE.findall(prose(text))]


def sections(text: str) -> dict[int, str]:
    """Return the text of each numbered section (``## <n> ...``) of a document, by its number."""
    starts = [(int(match.group(1)), match.start()) for match in SECTION_HEADING.finditer(text)]
    ends = [start for _, start in starts[1:]] + [len(text)]
    return {number: text[start:end] for (number, start), end in zip(starts, ends, strict=True)}


def _is_placeholder(word: str) -> bool:
    return word.isupper() or "<" in word or word in ("...", "-")


def _load_any_experiment(path: Path) -> str | None:
    """Load an experiment file of a probe or of a tool; return the refusal of both loaders, or None."""
    try:
        cli.load_experiment(path)
        return None
    except cli.CliError as probe_error:
        try:
            cli.load_tool_experiment(path)
            return None
        except cli.CliError as tool_error:
            return f"{probe_error} / {tool_error}"


def file_reference_problems(text: str) -> list[str]:
    """Return the files of the repository that the inline code of a text names and that are missing or do not load."""
    problems: list[str] = []
    for span in dict.fromkeys(inline_spans(text)):
        if "*" in span or "<" in span or span in WRITTEN_ON_THE_RUN_MACHINE:
            continue
        if REPOSITORY_PATH.match(span):
            path = REPO / span
        elif PACKAGE_MODULE.match(span):
            path = REPO / "src" / "pitchvlm" / span
        elif EXPERIMENT_NAME.match(span):
            path = REPO / "configs" / "experiments" / span
        else:
            continue
        if not path.is_file():
            problems.append(f"`{span}`: the file {path.relative_to(REPO)} does not exist")
        elif path.parent == REPO / "configs" / "experiments" and (refusal := _load_any_experiment(path)):
            problems.append(f"`{span}`: {refusal}")
    return problems


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def _package_parser() -> argparse.ArgumentParser:
    return cli.build_parser()


def _tool_parser() -> argparse.ArgumentParser:
    return _load_module("run_tool_of_the_documents", REPO / "tools" / "run_tool.py").build_parser()


def _leaf_parsers(parser: argparse.ArgumentParser) -> Iterator[argparse.ArgumentParser]:
    children = _subparsers(parser)
    if not children:
        yield parser
    for child in children.values():
        yield from _leaf_parsers(child)


def _value_problem(action: argparse.Action, value: str) -> str | None:
    """Return why an option does not take a value, or None when it takes it."""
    if action.nargs == 0:
        return "takes no value"
    if _is_placeholder(value):
        return None
    if action.choices is not None and value not in action.choices:
        return f"takes one of {', '.join(map(str, action.choices))}"
    if callable(action.type):
        try:
            action.type(value)
        except (argparse.ArgumentTypeError, TypeError, ValueError) as error:
            return f"does not take {value!r}: {error}"
    return None


def _words(span: str) -> list[str]:
    try:
        return shlex.split(span)
    except ValueError:
        return []


def inline_command_problems(text: str) -> list[str]:
    """Return what is wrong with the commands that the inline code of a text names.

    A span is a command when its first word is a sub-command of the package or
    ``tools/run_tool.py``, and its second word is an option or a sub-command or a choice of
    that parser. Other spans (``data policy``: the name of a doctor line) are not commands.
    """
    package = _package_parser()
    commands = {**_subparsers(package), "tools/run_tool.py": _tool_parser()}
    problems: list[str] = []
    for span in dict.fromkeys(inline_spans(text)):
        words = _words(span)
        if words[:3] in (["python", "-m", "pitchvlm.cli"], ["python3", "-m", "pitchvlm.cli"]):
            words = words[3:]
        elif len(words) > 1 and words[0] in ("python", "python3") and words[1].endswith("tools/run_tool.py"):
            words = ["tools/run_tool.py", *words[2:]]
        if len(words) < 2 or words[0] not in commands:
            continue
        parser, rest = commands[words[0]], words[1:]
        children = _subparsers(parser)
        positional_choices = [
            action.choices for action in parser._actions if not action.option_strings and action.choices is not None
        ]
        named = rest[0] in children or any(rest[0] in choices for choices in positional_choices)
        if not (named or any(word.startswith("--") for word in rest)):
            if children and not rest[0].startswith("-"):
                problems.append(f"`{span}`: {words[0]} has no sub-command {rest[0]!r}")
            continue
        if rest[0] in children:
            parser, rest = children[rest[0]], rest[1:]
        elif children and not rest[0].startswith("-"):
            problems.append(f"`{span}`: {words[0]} has no sub-command {rest[0]!r}")
            continue
        # An option right after a program with sub-commands (`tools/run_tool.py --feed`) is an
        # option of one of them.
        options = (
            {word: action for child in children.values() for word, action in child._option_string_actions.items()}
            if children and rest and rest[0].startswith("-") and rest[0] not in parser._option_string_actions
            else parser._option_string_actions
        )
        for position, word in enumerate(rest):
            if not word.startswith("--"):
                continue
            action = options.get(word)
            if action is None:
                problems.append(f"`{span}`: {words[0]} has no option {word}")
                continue
            value = rest[position + 1] if position + 1 < len(rest) else None
            if value is not None and not value.startswith("-") and (problem := _value_problem(action, value)):
                problems.append(f"`{span}`: {word} {problem}")
    return problems


def inline_option_problems(text: str) -> list[str]:
    """Return the options that inline code of a text names (``--delay clip``) and that no command has."""
    parsers = [*_leaf_parsers(_package_parser()), *_leaf_parsers(_tool_parser())]
    problems: list[str] = []
    for span in dict.fromkeys(inline_spans(text)):
        words = _words(span)
        if not words or not words[0].startswith("--") or words[0] in FOREIGN_OPTIONS:
            continue
        option = words[0].split("=", 1)[0]
        actions = [
            parser._option_string_actions[option] for parser in parsers if option in parser._option_string_actions
        ]
        if not actions:
            problems.append(f"`{span}`: no command of the package or of the tool script has the option {option}")
        elif len(words) > 1 and all(_value_problem(action, words[1]) for action in actions):
            problems.append(f"`{span}`: {option} {_value_problem(actions[0], words[1])}")
    return problems


def section_reference_problems(text: str, documents: dict[str, str]) -> list[str]:
    """Return the references of a text to sections of the RUNBOOK or the HANDOVER that do not exist.

    ``documents`` maps ``RUNBOOK`` and ``HANDOVER`` to their texts. A title in quotes right after the
    number must be a heading (``### <title>``) inside that section; a part (``4.2``) must be a
    heading ``### 4.2 ...`` of the section.
    """
    problems: list[str] = []
    text = prose(text)
    matches = [*SECTION_REFERENCE.finditer(text), *SECTION_REFERENCE_AFTER.finditer(text)]
    for match in sorted(matches, key=lambda found: found.start()):
        document, number = match.group("document"), int(match.group("number"))
        numbers = sections(documents[document])
        found = numbers.get(number)
        shown = " ".join(match.group(0).split())
        last = match.group("last")
        if last is not None and int(last) not in numbers:
            problems.append(f"{shown}: the {document} has no section {last}")
        if found is None:
            problems.append(f"{shown}: the {document} has no section {number}")
            continue
        groups = match.groupdict()
        part, title = groups.get("part"), groups.get("title")
        title = None if title is None else " ".join(title.split())
        if part is not None and not re.search(rf"^### {number}\.{part} ", found, re.M):
            problems.append(f"{shown}: section {number} of the {document} has no part {number}.{part}")
        if title is not None and not re.search(rf"^###+ {re.escape(title)}\s*(?:\(|$)", found, re.M):
            problems.append(f"{shown}: section {number} of the {document} has no heading {title!r}")
    return problems


def _documents() -> dict[str, str]:
    return {name: (REPO / path).read_text(encoding="utf-8") for name, path in REFERENCED.items()}


@pytest.mark.parametrize("document", [*DOCUMENTS, "docs/PLAN.md"])
def test_every_file_that_the_text_names_exists(document: str) -> None:
    assert file_reference_problems((REPO / document).read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("document", DOCUMENTS)
def test_every_command_that_the_text_names_exists_with_its_options(document: str) -> None:
    assert inline_command_problems((REPO / document).read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(("document", "numbers"), sorted(OPTION_SECTIONS.items()))
def test_every_option_of_the_new_sections_exists(document: str, numbers: tuple[int, ...]) -> None:
    found = sections((REPO / document).read_text(encoding="utf-8"))
    assert set(numbers) <= set(found), f"{document} has no section {sorted(set(numbers) - set(found))}"
    assert [problem for number in numbers for problem in inline_option_problems(found[number])] == []


@pytest.mark.parametrize("document", [*DOCUMENTS, "docs/PLAN.md"])
def test_every_section_that_a_document_names_exists(document: str) -> None:
    assert section_reference_problems((REPO / document).read_text(encoding="utf-8"), _documents()) == []


def test_sections_of_the_runbook_are_numbered_without_a_gap() -> None:
    numbers = [int(match.group(1)) for match in SECTION_HEADING.finditer(_documents()["RUNBOOK"])]
    assert numbers == list(range(1, len(numbers) + 1))
    assert len(numbers) >= 14


def test_sections_11_to_14_of_the_runbook_hold_commands_that_are_read() -> None:
    """The reader of the blocks must see the commands of the new sections (12 is a plan without commands)."""
    found = sections(_documents()["RUNBOOK"])
    for number in (11, 13, 14):
        assert documented_calls(found[number]), f"section {number} of the RUNBOOK has no command that the test reads"


PLANTED_PROSE = """
## 1 First

### A heading

Run the file `configs/experiments/people_probe_video_huge.yaml`, the file `people_probe_fast.yaml`
and the module `vlm/costs.py`. The split list `configs/splits.json` is written on the run machine.
The doctor with `doctor --vidoe`, a clip job with `clip-job --tool yolo`, the script with
`tools/run_tool.py yolo`, and the probe `probe people --video-dir ~/x`. A doctor line `data policy`.
Also `python -m pitchvlm.cli sync --result x`, `python tools/run_tool.py dfine --gpu 0`, `run x --bogus`
and `tools/run_tool.py --fed`; but `tools/run_tool.py --device` names an option of its sub-commands.
The stream takes `--delay soon`, `--delay clip`, `--delay N`, `--histroy` and `--exclude-newer`.
See RUNBOOK, section 15, RUNBOOK, section 1, "A heading", RUNBOOK, section 1, "No heading"
and HANDOVER, section 1.3. Also section 16 of the RUNBOOK and RUNBOOK, sections 1 to 17.
"""


def test_reader_of_the_text_finds_missing_files_commands_options_and_sections() -> None:
    assert file_reference_problems(PLANTED_PROSE) == [
        "`configs/experiments/people_probe_video_huge.yaml`: the file configs/experiments/"
        "people_probe_video_huge.yaml does not exist",
        "`people_probe_fast.yaml`: the file configs/experiments/people_probe_fast.yaml does not exist",
        "`vlm/costs.py`: the file src/pitchvlm/vlm/costs.py does not exist",
    ]
    commands = inline_command_problems(PLANTED_PROSE)
    assert len(commands) == 7, commands
    assert "doctor has no option --vidoe" in commands[0]
    assert "--tool takes one of" in commands[1]
    assert "tools/run_tool.py has no sub-command 'yolo'" in commands[2]
    assert "sync has no option --result" in commands[3]
    assert "tools/run_tool.py has no option --gpu" in commands[4]
    assert "run has no option --bogus" in commands[5]
    assert "tools/run_tool.py has no option --fed" in commands[6]
    options = inline_option_problems(PLANTED_PROSE)
    assert len(options) == 2, options
    assert "`--delay soon`: --delay does not take 'soon'" in options[0]
    assert "has the option --histroy" in options[1]
    documents = {"RUNBOOK": PLANTED_PROSE, "HANDOVER": PLANTED_PROSE}
    assert section_reference_problems(PLANTED_PROSE, documents) == [
        "RUNBOOK, section 15: the RUNBOOK has no section 15",
        "RUNBOOK, section 1, \"No heading\": section 1 of the RUNBOOK has no heading 'No heading'",
        "HANDOVER, section 1.3: section 1 of the HANDOVER has no part 1.3",
        "section 16 of the RUNBOOK: the RUNBOOK has no section 16",
        "RUNBOOK, sections 1 to 17: the RUNBOOK has no section 17",
    ]


def test_reader_of_the_text_passes_what_exists() -> None:
    text = """
## 1 First

The files `configs/experiments/people_probe_video_low.yaml`, `people_probe_dfine_medium.yaml`,
`vlm/cost.py` and `tools/run_tool.py`; the commands `doctor --video`, `clip-job --tool dfine`,
`tools/run_tool.py dfine`, `stream --delay clip`, `overlay probe ... --max-clips 4 --count 5`; the
options `--video-dir`, `--delay 12`, `--delay clip`, `--history`, `--kit-warmup` and `--device cpu`.
"""
    assert file_reference_problems(text) == []
    assert inline_command_problems(text) == []
    assert inline_option_problems(text) == []


def test_help_of_every_option_with_a_default_names_the_default() -> None:
    """``--help`` is a document too: an option that has a default value says so in its help text."""
    missing = [
        f"{parser.prog} {action.option_strings[0]} (default {action.default!r})"
        for parser in [*_leaf_parsers(_package_parser()), *_leaf_parsers(_tool_parser())]
        for action in parser._actions
        if action.option_strings
        and action.nargs != 0
        and action.default not in (None, False, argparse.SUPPRESS)
        and "default" not in (action.help or "")
    ]
    assert missing == []
