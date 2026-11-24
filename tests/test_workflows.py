"""The workflows of GitHub Actions (``.github/workflows/``) load as YAML and keep the rules of CI.

A workflow file with a syntax error does not run at all, and the git host says so only after a
push. This test finds it on the build machine first. It also checks the rules of the decision
"The offline tests on Linux in CI" (``docs/DECISIONS.md``, 2026-11-23): no ``pull_request_target``,
a token that can only read, no stored credentials after the checkout, a fixed runner image, a time
limit on every job, every action pinned by its full commit, and no secret.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((REPO / ".github" / "workflows").glob("*.y*ml"))
#: An action pinned by its full commit, with the version in a comment: ``owner/name@<40 hex> # v1.2.3``.
PINNED_ACTION = re.compile(r"^\s*-?\s*uses:\s*[\w.-]+/[\w./-]+@[0-9a-f]{40}\s+#\s*v\d+(?:\.\d+)*\s*$")


def load(path: Path) -> dict[Any, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{path.name}: not a mapping"
    return document


def triggers(document: dict[Any, Any]) -> Any:
    """Return the value of ``on``. YAML 1.1 (the parser of the project) reads the bare word ``on`` as
    the boolean true; the git host reads it as the text ``on``. Both spellings are accepted."""
    for key in ("on", True):
        if key in document:
            return document[key]
    raise AssertionError("no 'on' key: the workflow has no trigger")


def test_there_is_a_workflow() -> None:
    assert [path.name for path in WORKFLOWS] == ["offline-tests.yml"]


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_a_workflow_loads_and_keeps_the_rules_of_ci(path: Path) -> None:
    document = load(path)
    on = triggers(document)
    assert isinstance(on, dict) and set(on) == {"pull_request", "push"}
    assert on["push"] == {"branches": ["main"]}
    assert "pull_request_target" not in path.read_text(encoding="utf-8")
    assert document["permissions"] == {"contents": "read"}
    assert document["concurrency"]["cancel-in-progress"] is True

    jobs = document["jobs"]
    assert isinstance(jobs, dict) and jobs
    for name, job in jobs.items():
        assert isinstance(job.get("timeout-minutes"), int), f"job {name}: no time limit"
        assert re.fullmatch(r"ubuntu-\d\d\.\d\d", job["runs-on"]), f"job {name}: runner image is not fixed"
        assert "permissions" not in job, f"job {name}: widens the token"
        steps = job["steps"]
        for step in steps:
            assert set(step) & {"run", "uses"}, f"job {name}: a step without run or uses"
            if step.get("uses", "").startswith("actions/checkout@"):
                assert step["with"]["persist-credentials"] is False, f"job {name}: the checkout keeps the token"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_every_action_is_pinned_by_its_full_commit_and_no_secret_is_named(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    uses = [line for line in text.splitlines() if re.match(r"^\s*-?\s*uses:", line)]
    assert uses, "no action"
    assert [line for line in uses if not PINNED_ACTION.match(line)] == []
    assert "secrets." not in text and "GITHUB_TOKEN" not in text


def test_a_plain_value_with_a_colon_and_a_space_is_a_syntax_error() -> None:
    """The first version of the workflow had ``run: ... --only-binary :all: -r ...`` as a plain value; the
    git host refused the file. The parser of the project refuses it too, so the test above finds it."""
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load("steps:\n  - run: pip install --only-binary :all: -r locks/ci-linux.txt\n")
