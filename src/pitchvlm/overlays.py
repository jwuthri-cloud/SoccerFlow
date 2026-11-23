"""Check images of pitch mappings: which frames a person checks by eye, and the list of them.

Gate G3 counts its kill switch only after overlay images of the pitch mapping are checked by
eye (``overlays_checked_by_eye`` in ``configs/gates.yaml``), and milestone M3 asks for ten. The
command ``overlay`` of the command line collects the mappings of one source (the stored answers
of the pitch probe, the run file of a calibration tool, or the result of a run), and this module
chooses the frames to draw and writes their list.

The choice can be repeated and cannot be steered (``choose_frames``): it depends only on the
mappings, the seed and the count. The list file is text without pixels and without label data:
clip ids, frame numbers, the engine, the paint alignment score of each mapping and the name of
its image. So it can go on the results branch (rule 2 of ``CLAUDE.md``). The person who checks
the images writes the verdict of each frame into its ``by_eye`` field.
"""

from __future__ import annotations

import hashlib
import math
import operator
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Format name of the list file of checked frames.
LIST_FORMAT = "pitchvlm-overlay-list-1"
#: The verdicts that the person who checks an image may write into ``by_eye``.
VERDICTS = ("right", "wrong")
#: The rule of ``choose_frames`` in words, for the list file.
CHOICE_RULE = (
    "frames with a mapping, ordered by paint alignment score (frames that cannot judge first, then from the "
    "lowest score; a tie by a hash of seed, clip and frame), cut into count equal parts in that order; from each "
    "part the frame of the clip with the fewest frames chosen so far, a tie by a hash of seed, clip and frame"
)


@dataclass(frozen=True, eq=False)
class MappedFrame:
    """The pitch mapping of one frame, with its paint alignment in that frame."""

    clip: str
    frame: int
    #: Image-to-pitch mapping, 3x3.
    matrix: np.ndarray
    #: How the mapping of this frame was made, in a few words (for example "landmark fit of this frame").
    origin: str
    #: Paint alignment score in 0 to 1, or None when the frame cannot judge a mapping.
    score: float | None
    #: Why the frame cannot judge a mapping, or None when it has a score.
    reason: str | None

    def __post_init__(self) -> None:
        if (self.score is None) == (self.reason is None):
            raise ValueError("a mapped frame has either a paint score or a reason why the frame cannot judge")


def _hash(seed: int, *names: object) -> int:
    """Return a 64-bit number that depends only on the seed and the names (SHA-256, the same on every machine)."""
    text = ":".join(str(part) for part in (operator.index(seed), *names))
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def choose_frames(frames: Sequence[MappedFrame], *, count: int, seed: int) -> list[MappedFrame]:
    """Choose ``count`` frames to check by eye, spread over the score range and over the clips.

    The frames are put in the order of their paint alignment score: the frames that cannot
    judge a mapping first, then from the lowest score to the highest; frames with the same
    place are ordered by a hash of the seed, the clip and the frame. This order is cut into
    ``count`` parts of equal size (the sizes differ by one at most), and one frame is taken
    from each part: the frame whose clip has the fewest frames chosen so far, with a tie
    broken by a second hash of the seed, the clip and the frame. So every part of the score
    range gets one frame, the clips take turns as far as the parts allow, and the same
    frames, seed and count give the same choice on every machine.

    With at most ``count`` frames, all frames are chosen. The result is in the order of the
    parts. ``ValueError`` is raised for a count below 1 and for two frames with the same clip
    and frame.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    twice = [key for key, number in Counter((item.clip, item.frame) for item in frames).items() if number > 1]
    if twice:
        raise ValueError(f"these frames come more than once: {twice[:3]}")

    def place(item: MappedFrame) -> tuple[bool, float, int]:
        score = -math.inf if item.score is None else item.score
        return (item.score is not None, score, _hash(seed, "order", item.clip, item.frame))

    ordered = sorted(frames, key=place)
    n_parts = min(count, len(ordered))
    chosen: list[MappedFrame] = []
    per_clip: Counter[str] = Counter()
    for part in range(n_parts):
        members = ordered[part * len(ordered) // n_parts : (part + 1) * len(ordered) // n_parts]
        pick = min(members, key=lambda item: (per_clip[item.clip], _hash(seed, "pick", item.clip, item.frame)))
        per_clip[pick.clip] += 1
        chosen.append(pick)
    return chosen


def image_name(position: int, item: MappedFrame) -> str:
    """Return the file name of the check image of a chosen frame: its place in the list, the clip and the frame."""
    return f"{position + 1:02d}_{item.clip}_{item.frame + 1:06d}.png"


def caption(item: MappedFrame, *, engine: str, position: int, count: int) -> list[str]:
    """Return the lines of text of a check image: the clip and frame, the engine, the mapping and the paint score."""
    paint = f"paint alignment score {item.score:.3f}" if item.score is not None else f"cannot judge: {item.reason}"
    return [
        f"{position + 1} of {count}: clip {item.clip}, frame {item.frame} (image {item.frame + 1:06d})",
        f"engine: {engine}; mapping: {item.origin}",
        paint,
        "magenta: the pitch lines of the mapping. Right when they lie on the white paint.",
    ]


def list_document(
    chosen: Sequence[MappedFrame],
    *,
    source: str,
    name: str,
    engine: str,
    seed: int,
    count: int,
    n_frames: int,
    n_mapped: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Return the list file of the chosen frames as plain JSON types.

    ``n_frames`` is the number of frames that the source covers, ``n_mapped`` the number of
    them with a mapping, from which ``choose_frames`` chose. Every frame entry has
    ``by_eye`` null: the person who checks the image writes one of ``VERDICTS`` there.
    """
    return {
        "format": LIST_FORMAT,
        "source": source,
        "name": name,
        "engine": engine,
        "seed": seed,
        "count": count,
        "rule": CHOICE_RULE,
        "n_frames": n_frames,
        "n_frames_with_mapping": n_mapped,
        "n_chosen": len(chosen),
        "verdicts": list(VERDICTS),
        "provenance": provenance,
        "frames": [
            {
                "image": image_name(position, item),
                "clip": item.clip,
                "frame": item.frame,
                "mapping": item.origin,
                "paint_score": item.score,
                "paint_reason": item.reason,
                "by_eye": None,
            }
            for position, item in enumerate(chosen)
        ],
    }
