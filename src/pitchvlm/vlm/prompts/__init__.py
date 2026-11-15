"""Versioned prompt texts of the four model steps.

Each prompt is a plain text file in this folder. A placeholder is a name in
curly braces, and ``str.format`` fills it. So a prompt file must not contain
any other curly brace.

The cache key of a request holds a hash of the prompt text, so an edited
prompt never reuses an old answer. The prompt name is for people: it tells
which text gave an answer. Raise the version in ``PROMPT_NAMES`` each time
the text of a prompt changes.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import resources
from types import MappingProxyType

#: Prompt name of each step. The part before the slash is the file name without ".txt".
PROMPT_NAMES: Mapping[str, str] = MappingProxyType(
    {
        "kit_legend": "kit_legend/v1",
        "people": "people/v1",
        "landmarks": "landmarks/v1",
        "number": "number/v1",
    }
)


def render_prompt(step: str, **values: object) -> str:
    """Return the prompt text of a step with its placeholders filled.

    An unknown step raises KeyError. A placeholder without a value raises
    KeyError too.
    """
    if step not in PROMPT_NAMES:
        raise KeyError(f"no prompt for step {step!r}")
    template = resources.files("pitchvlm.vlm.prompts").joinpath(f"{step}.txt").read_text(encoding="utf-8")
    return template.format(**values).strip()
