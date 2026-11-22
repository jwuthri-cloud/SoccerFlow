"""Accuracy measures for boxes, pitch positions, identities and shirt numbers.

Every function that takes a predicted and a true sequence pairs their frames
by frame index. The true sequence decides which frames are scored:

- a true frame without a predicted frame counts as a frame without predictions;
- a predicted frame without a true frame is ignored;
- a frame index that is present twice in one sequence raises ``ValueError``.

A share that has nothing to count (for example recall without any true box)
is ``nan``.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment

from pitchvlm.schemas import Box, FramePeople, FrameState, PlayerState

#: (frame, image xy of shape (n, 2)) -> pitch metres of shape (n, 2). None means that the frame has no mapping.
PitchMap = Callable[[int, np.ndarray], np.ndarray | None]

_Frame = TypeVar("_Frame", FramePeople, FrameState)


# --- Boxes -----------------------------------------------------------------------


def match_boxes(pred: Sequence[Box], truth: Sequence[Box], iou_threshold: float = 0.5) -> list[tuple[int, int]]:
    """Pair predicted boxes with true boxes, one to one.

    The Hungarian method finds the pairing with the largest total IoU, in
    which an IoU below ``iou_threshold`` counts as 0. So a pair that does not
    count cannot push a pair that counts out of the pairing: without this
    rule, two weak overlaps could win over one good pair, and then no pair
    would be left. The result holds the pairs ``(pred index, truth index)``
    whose IoU is at least ``iou_threshold``, ordered by the pred index.
    """
    if len(pred) == 0 or len(truth) == 0:
        return []
    iou = np.array([[pred_box.iou(true_box) for true_box in truth] for pred_box in pred], dtype=np.float64)
    counted = iou >= iou_threshold  # iou and counted have the shape (n_pred, n_truth)
    rows, cols = linear_sum_assignment(np.where(counted, iou, 0.0), maximize=True)
    return [(int(row), int(col)) for row, col in zip(rows, cols) if counted[row, col]]


@dataclass(frozen=True)
class DetectionScores:
    """Counts of true positives, false positives and false negatives."""

    tp: int
    fp: int
    fn: int

    @property
    def recall(self) -> float:
        """Share of the true boxes that were found. It is nan when there is no true box."""
        return _share(self.tp, self.tp + self.fn)

    @property
    def precision(self) -> float:
        """Share of the predicted boxes that are right. It is nan when there is no counted prediction."""
        return _share(self.tp, self.tp + self.fp)


def detection_scores(
    pred: Sequence[FramePeople],
    truth: Sequence[FramePeople],
    *,
    iou_threshold: float = 0.5,
    max_height_px: float | None = None,
) -> DetectionScores:
    """Count found, wrong and missed boxes over all true frames.

    Boxes are matched per frame with ``match_boxes``. With ``max_height_px``
    only the true boxes up to that height count: a prediction that is matched
    to a taller true box is ignored, so it is not a false positive. A
    prediction without a match is always a false positive.
    """
    pred_by_frame = _by_frame(pred, "pred")
    tp = fp = fn = 0
    for frame, true_people in _by_frame(truth, "truth").items():
        true_boxes = [person.box for person in true_people.people]
        pred_boxes = [person.box for person in pred_by_frame[frame].people] if frame in pred_by_frame else []
        pairs = match_boxes(pred_boxes, true_boxes, iou_threshold)
        counted = [max_height_px is None or box.height <= max_height_px for box in true_boxes]
        hits = sum(1 for _, true_index in pairs if counted[true_index])
        tp += hits
        fn += sum(counted) - hits
        fp += len(pred_boxes) - len(pairs)
    return DetectionScores(tp=tp, fp=fp, fn=fn)


def foot_errors_m(
    pred: Sequence[FramePeople], truth: Sequence[FramePeople], to_pitch: PitchMap, *, iou_threshold: float = 0.5
) -> np.ndarray:
    """Return the pitch distance in metres between the foot points of matched boxes, shape (n,).

    Boxes are matched per frame with ``match_boxes``. Both foot points of a
    pair go through the same ``to_pitch``, so the result measures the placement
    of the box only, not the quality of the pitch mapping. Frames for which
    ``to_pitch`` returns None have no mapping and add no value.
    """
    pred_by_frame = _by_frame(pred, "pred")
    errors: list[np.ndarray] = []
    for frame, true_people in _by_frame(truth, "truth").items():
        if frame not in pred_by_frame:
            continue
        true_boxes = [person.box for person in true_people.people]
        pred_boxes = [person.box for person in pred_by_frame[frame].people]
        pairs = match_boxes(pred_boxes, true_boxes, iou_threshold)
        if not pairs:
            continue
        pred_feet = [pred_boxes[pred_index].foot for pred_index, _ in pairs]
        true_feet = [true_boxes[true_index].foot for _, true_index in pairs]
        feet = np.array(pred_feet + true_feet, dtype=np.float64)  # (2 k, 2): k pred points, then k true points
        mapped = to_pitch(frame, feet)
        if mapped is None:
            continue
        on_pitch = np.asarray(mapped, dtype=np.float64).reshape(2, len(pairs), 2)
        errors.append(np.linalg.norm(on_pitch[0] - on_pitch[1], axis=1))
    return np.concatenate(errors) if errors else np.zeros(0, dtype=np.float64)


# --- Pitch positions and identities ------------------------------------------------


def position_errors_m(
    pred: Sequence[FrameState], truth: Sequence[FrameState], *, max_match_m: float = 10.0
) -> np.ndarray:
    """Return the pitch distance in metres between matched predicted and true people, shape (n,).

    People are matched per frame, one to one, with the Hungarian method on the
    pitch distance. Only pairs that are at most ``max_match_m`` apart count.
    The distance of a pair that does not count has no effect on the other pairs.
    ``max_match_m`` must be a finite number of 0 or more, else ``ValueError`` is raised.
    """
    _check_match_limit(max_match_m)
    pred_by_frame = _by_frame(pred, "pred")
    errors: list[float] = []
    for frame, true_state in _by_frame(truth, "truth").items():
        if frame in pred_by_frame:
            pairs = _match_positions(pred_by_frame[frame].players, true_state.players, max_match_m)
            errors.extend(distance for _, _, distance in pairs)
    return np.array(errors, dtype=np.float64)


@dataclass(frozen=True)
class IdentityScores:
    """How well predicted track ids follow the true tracks."""

    #: Mean over the true tracks of the largest share of the track that one predicted id covers.
    coverage: float
    #: Number of changes of the predicted id along the true tracks.
    switches: int
    #: Number of predicted tracks divided by the number of true tracks.
    track_ratio: float
    #: Mean over the predicted ids of the largest share of their matched frames that one true track covers.
    purity: float


def identity_scores(
    pred: Sequence[FrameState], truth: Sequence[FrameState], *, max_match_m: float = 5.0
) -> IdentityScores:
    """Score how well the predicted track ids follow the true tracks.

    People are matched per frame as in ``position_errors_m``. This gives, for
    every true track, the predicted id in each of its frames (or no id).

    - ``coverage``: for each true track, the largest number of its frames that
      one predicted id covers, divided by the number of its frames; then the
      mean over the true tracks.
    - ``switches``: the number of times the predicted id of a true track
      differs from its previous predicted id. Frames without a match are
      skipped, so a gap alone is not a switch.
    - ``track_ratio``: the number of predicted ids seen in the true frames,
      divided by the number of true tracks.
    - ``purity``: for each predicted id that is matched in at least one frame,
      the largest number of its matched frames that one true track covers,
      divided by the number of its matched frames; then the mean over these
      ids. The three values above do not see a predicted id that follows two
      people one after the other (a wrong join): this value does. No gate
      uses it.

    ``max_match_m`` must be a finite number of 0 or more, else ``ValueError`` is raised.
    """
    _check_match_limit(max_match_m)
    pred_by_frame = _by_frame(pred, "pred")
    truth_by_frame = _by_frame(truth, "truth")
    #: True track id -> predicted id in each frame of the track, in frame order.
    followed: dict[int, list[int | None]] = {}
    #: Predicted id -> true id in each frame in which the predicted id is matched.
    covered: dict[int, list[int]] = {}
    pred_ids: set[int] = set()
    for frame in sorted(truth_by_frame):
        true_players = truth_by_frame[frame].players
        pred_players = pred_by_frame[frame].players if frame in pred_by_frame else ()
        pred_ids.update(player.track_id for player in pred_players)
        pairs = _match_positions(pred_players, true_players, max_match_m)
        partner = {true_index: pred_players[pred_index].track_id for pred_index, true_index, _ in pairs}
        for true_index, player in enumerate(true_players):
            followed.setdefault(player.track_id, []).append(partner.get(true_index))
        for pred_index, true_index, _ in pairs:
            covered.setdefault(pred_players[pred_index].track_id, []).append(true_players[true_index].track_id)
    shares = [_largest_share(ids) for ids in followed.values()]
    purities = [_largest_share(ids) for ids in covered.values()]
    return IdentityScores(
        coverage=sum(shares) / len(shares) if shares else math.nan,
        switches=sum(_switches(ids) for ids in followed.values()),
        track_ratio=_share(len(pred_ids), len(followed)),
        purity=sum(purities) / len(purities) if purities else math.nan,
    )


# --- Shirt numbers and statistics ----------------------------------------------------


def number_exact_match(pred: Mapping[int, int | None], truth: Mapping[int, int | None]) -> float:
    """Return the share of tracks whose predicted shirt number equals the true one.

    Both mappings go from a track id to a number, and must have the same
    track ids. "No number" (None) is a value like any other: None against None
    is a match, None against a number is not.
    """
    if pred.keys() != truth.keys():
        different = sorted(pred.keys() ^ truth.keys())
        raise ValueError(f"pred and truth must have the same track ids; ids in one mapping only: {different}")
    return _share(sum(1 for track_id, number in truth.items() if pred[track_id] == number), len(truth))


def summarize(values: np.ndarray) -> dict[str, float]:
    """Return ``n``, ``median``, ``p90`` and ``mean`` of the values. Without values the three statistics are nan."""
    data = np.asarray(values, dtype=np.float64).ravel()
    if data.size == 0:
        return {"n": 0.0, "median": math.nan, "p90": math.nan, "mean": math.nan}
    return {
        "n": float(data.size),
        "median": float(np.median(data)),
        "p90": float(np.percentile(data, 90.0)),
        "mean": float(np.mean(data)),
    }


def t_interval(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """Return the mean and its two-sided Student t interval: ``(mean, low, high)``.

    With one value there is no spread, so the result is ``(value, nan, nan)``.
    An empty sequence, a value that is not finite and a confidence outside 0-1
    raise ``ValueError``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be between 0 and 1, found {confidence}")
    data = np.asarray(values, dtype=np.float64).ravel()
    if data.size == 0:
        raise ValueError("t_interval needs at least one value")
    if not np.isfinite(data).all():
        raise ValueError("t_interval needs finite values: a value is NaN or infinite")
    mean = float(np.mean(data))
    if data.size == 1:
        return mean, math.nan, math.nan
    standard_error = float(np.std(data, ddof=1)) / math.sqrt(data.size)
    half_width = float(stats.t.ppf((1.0 + confidence) / 2.0, data.size - 1)) * standard_error
    return mean, mean - half_width, mean + half_width


# --- Helpers -----------------------------------------------------------------------


def _share(part: int, whole: int) -> float:
    """Return ``part / whole``, or nan when ``whole`` is 0."""
    return part / whole if whole else math.nan


def _check_match_limit(max_match_m: float) -> None:
    """Raise ``ValueError`` when the largest distance of a counted pair is not a finite number of 0 or more."""
    if not 0.0 <= max_match_m < math.inf:
        raise ValueError(f"max_match_m must be a finite number of 0 or more, found {max_match_m}")


def _by_frame(items: Sequence[_Frame], name: str) -> dict[int, _Frame]:
    """Index a sequence of frame records by frame index. A frame that is present twice raises ``ValueError``."""
    indexed: dict[int, _Frame] = {}
    for item in items:
        if item.frame in indexed:
            raise ValueError(f"{name} holds frame {item.frame} more than once")
        indexed[item.frame] = item
    return indexed


def _match_positions(
    pred: Sequence[PlayerState], truth: Sequence[PlayerState], max_match_m: float
) -> list[tuple[int, int, float]]:
    """Pair the predicted and the true people of one frame, one to one, by pitch distance.

    The Hungarian method finds the pairing with the smallest total cost. A
    pair counts when it is at most ``max_match_m`` apart, and its cost is its
    distance. Every pair that does not count costs the same, however far
    apart it is: just above ``max_match_m``, which is more than the cost of
    every pair that counts. Without this rule, the long distance of a pair
    that does not count could make the method break up a close pair, and a
    pair at exactly ``max_match_m`` could lose against a pair that does not
    count. The result holds ``(pred index, truth index, distance in metres)``
    for the pairs that count, ordered by the pred index.
    """
    if len(pred) == 0 or len(truth) == 0:
        return []
    pred_xy = np.array([[player.pitch_x, player.pitch_y] for player in pred], dtype=np.float64)  # (n_pred, 2)
    true_xy = np.array([[player.pitch_x, player.pitch_y] for player in truth], dtype=np.float64)  # (n_truth, 2)
    distance = np.linalg.norm(pred_xy[:, np.newaxis, :] - true_xy[np.newaxis, :, :], axis=2)  # (n_pred, n_truth)
    counted = distance <= max_match_m
    rows, cols = linear_sum_assignment(np.where(counted, distance, math.nextafter(max_match_m, math.inf)))
    return [(int(row), int(col), float(distance[row, col])) for row, col in zip(rows, cols) if counted[row, col]]


def _largest_share(ids: Sequence[int | None]) -> float:
    """Return the share of the frames of a track that its most frequent partner id covers. None is no id."""
    counts = Counter(track_id for track_id in ids if track_id is not None)
    return max(counts.values(), default=0) / len(ids)


def _switches(ids: Sequence[int | None]) -> int:
    """Count the changes of the predicted id along a true track. Frames without an id are skipped."""
    seen = [track_id for track_id in ids if track_id is not None]
    return sum(1 for previous, current in zip(seen, seen[1:]) if previous != current)
