"""Versioned prompt texts of the four model steps.

Each prompt is a plain text file in this folder. A placeholder is a name in
curly braces, and ``str.format`` fills it. So a prompt file must not contain
any other curly brace.

The cache key of a request holds a hash of the prompt text, so an edited
prompt never reuses an old answer. The prompt name is for people: it tells
which text gave an answer. Raise the version in ``PROMPT_NAMES`` each time
the text of a prompt changes.

The people prompt and the landmarks prompt state the coordinate convention
of their model (``box_format``, ``point_format``). For the Gemini convention
``yx_1000`` the text is the one that these prompts always had, so the cache
keys of Gemini requests did not change when the conventions were added.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import resources
from types import MappingProxyType

from pitchvlm.vlm.coords import CoordConvention

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


def _format_sentence(fields: str, convention: CoordConvention, image: str, width: int, height: int) -> str:
    """Write the format sentence of a box or a point: the order of the values, their unit and the two axes."""
    if convention == "xy_unit":
        value = f"Each value is a number from 0 to 1, normalised to {image}"
        x_end = y_end = "1"
    elif convention == "xy_pixel":
        value = (
            f"Each value is an integer number of pixels of {image}, "
            f"which is {width} pixels wide and {height} pixels high"
        )
        x_end, y_end = str(width), str(height)
    else:
        value = f"Each value is an integer from 0 to 1000, normalised to {image}"
        x_end = y_end = "1000"
    y_axis = f"y goes from 0 at the top edge to {y_end} at the bottom edge"
    x_axis = f"x goes from 0 at the left edge to {x_end} at the right edge"
    first, second = (y_axis, x_axis) if convention == "yx_1000" else (x_axis, y_axis)
    return f"Format: {fields}. {value}: {first}, and {second}."


def box_format(convention: CoordConvention, *, image: str, width: int, height: int) -> str:
    """Return the sentence that tells the model how to write a box in the convention.

    ``image`` names the image in the prompt, for example "the image". ``width`` and
    ``height`` are its size in pixels; only the pixel convention names them.
    """
    fields = "[ymin, xmin, ymax, xmax]" if convention == "yx_1000" else "[xmin, ymin, xmax, ymax]"
    return _format_sentence(fields, convention, image, width, height)


def point_format(convention: CoordConvention, *, image: str, width: int, height: int) -> str:
    """Return the sentence that tells the model how to write a point in the convention (see ``box_format``)."""
    fields = "[y, x]" if convention == "yx_1000" else "[x, y]"
    return _format_sentence(fields, convention, image, width, height)
