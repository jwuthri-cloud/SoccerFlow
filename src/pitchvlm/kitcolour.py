"""Team and role of detector boxes from the colours of the kits.

A detector with weights for the everyday classes finds people, not teams. This module gives
each person of the detector engine a team key and a role with plain code:

1. ``torso_feature``: a colour histogram of the torso part of the box (the middle half of
   its width, from 15% to 45% of its height from the top). Saturated pixels count in one of
   12 hue bins; the others in one of 3 brightness bins (dark, grey, light), so black, white
   and grey kits have their own bins. The feature is the square root of the shares, so that
   the distance of two features is the Hellinger distance of the histograms times sqrt(2).
2. ``fit_kit_clusters``: a 2-cluster k-means of the features of the clip, in numpy. The two
   clusters are the two teams: ``A`` is cluster 0 and ``B`` cluster 1. The names are labels
   of the colours only; the side of each team comes from the pitch positions
   (``tracking.assign_team_sides``).
3. ``KitClusters.team_key``: the cluster of a feature, or ``none`` for an outlier: a feature
   further from its cluster centre than the median distance of that cluster plus
   ``OUTLIER_MADS`` scaled median absolute deviations. Goalkeepers and referees wear other
   kits than both teams, so they are outliers.
4. ``kit_roles``: per track. A track whose observations are mostly outliers (by box height,
   like the votes of ``tracking``) is a goalkeeper when its mean pitch position lies in a
   goal zone (``tracking.goal_side``) and inside the width of the penalty area, and a referee
   otherwise. A linesman stands near a touchline, outside that width. Every other track
   keeps its observations: role ``player`` and the team key of each observation.

The fixed values (the torso window, the bins, ``OUTLIER_MADS``) are estimates: they were
checked on the synthetic clip only, where every kit is one flat colour. Real kits have
stripes, numbers and shade; the first measurement is the role and team-side accuracy of gate
G5b on the dev clips.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from pitchvlm.pitch import PENALTY_AREA_HALF_WIDTH
from pitchvlm.schemas import Box, Role, TeamKey, Track
from pitchvlm.tracking import goal_side

#: The torso window of a box, as shares of its width (left, right) and of its height from the top.
TORSO_X = (0.25, 0.75)
TORSO_Y = (0.15, 0.45)
#: A torso window with fewer pixels gives no feature.
MIN_TORSO_PIXELS = 12
HUE_BINS = 12
#: A pixel counts in a hue bin when its saturation and its brightness (0-255) are at least these values.
MIN_SATURATION = 64
MIN_BRIGHTNESS = 48
#: Limits of the brightness bins of the other pixels (0-255): dark below the first, light from the second.
BRIGHTNESS_LIMITS = (85, 170)
FEATURE_SIZE = HUE_BINS + 3
#: A feature is an outlier of its cluster beyond the median distance plus this many scaled deviations.
OUTLIER_MADS = 3.0
#: Scale of the median absolute deviation of a normal distribution.
_MAD_SCALE = 1.4826
#: Smallest outlier margin, so that a cluster of equal features does not make every other feature an outlier.
_MIN_MARGIN = 0.05
_MAX_ITERATIONS = 100
TEAM_KEYS: tuple[TeamKey, TeamKey] = ("A", "B")


def torso_feature(image: np.ndarray, box: Box) -> np.ndarray | None:
    """Return the colour feature of the torso of a box, shape ``(FEATURE_SIZE,)``, or None.

    ``image`` is a BGR ``uint8`` frame. The window is cut at the image border. None is given
    when the window holds fewer than ``MIN_TORSO_PIXELS`` pixels.
    """
    height, width = image.shape[:2]
    left = max(0, math.floor(box.x1 + TORSO_X[0] * box.width))
    right = min(width, math.ceil(box.x1 + TORSO_X[1] * box.width))
    top = max(0, math.floor(box.y1 + TORSO_Y[0] * box.height))
    bottom = min(height, math.ceil(box.y1 + TORSO_Y[1] * box.height))
    if right <= left or bottom <= top or (right - left) * (bottom - top) < MIN_TORSO_PIXELS:
        return None
    hsv = cv2.cvtColor(np.ascontiguousarray(image[top:bottom, left:right]), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hue, saturation, value = (hsv[:, channel].astype(np.int64) for channel in range(3))
    coloured = (saturation >= MIN_SATURATION) & (value >= MIN_BRIGHTNESS)
    hue_bin = np.minimum(hue * HUE_BINS // 180, HUE_BINS - 1)  # OpenCV hue runs from 0 to 179.
    brightness_bin = HUE_BINS + np.searchsorted(np.array(BRIGHTNESS_LIMITS), value, side="right")
    bins = np.where(coloured, hue_bin, brightness_bin)
    counts = np.bincount(bins, minlength=FEATURE_SIZE).astype(np.float64)
    return np.sqrt(counts / counts.sum())


@dataclass(frozen=True)
class KitClusters:
    """Two kit clusters: their centres and the distance from which a feature is an outlier."""

    #: Shape (2, FEATURE_SIZE); row 0 is team key ``A``, row 1 is ``B``.
    centres: np.ndarray
    #: Outlier limit of each cluster, the same order.
    limits: tuple[float, float]

    def team_key(self, feature: np.ndarray | None) -> TeamKey:
        """Return the team key of a feature: its cluster, or ``none`` for an outlier and for no feature."""
        if feature is None:
            return "none"
        distances = np.linalg.norm(self.centres - feature, axis=1)
        nearest = int(np.argmin(distances))
        return TEAM_KEYS[nearest] if distances[nearest] <= self.limits[nearest] else "none"


def fit_kit_clusters(features: np.ndarray, *, seed: int = 0) -> KitClusters:
    """Split the features of a clip into two kit clusters with k-means, and set the outlier limits.

    ``features`` has shape ``(n, FEATURE_SIZE)``. The start centres come from k-means++ with
    ``seed``, so a run gives the same clusters again. Cluster 0 is the one whose centre comes
    first in the order of the feature values, so the names do not depend on the order of the
    rows. ``ValueError`` is raised for a wrong shape, and when the features do not hold two
    different values.
    """
    if features.ndim != 2 or features.shape[1] != FEATURE_SIZE:
        raise ValueError(f"features must have the shape (n, {FEATURE_SIZE}), got {features.shape}")
    if len(np.unique(features, axis=0)) < 2:
        raise ValueError("the kit colours of the clip need at least two different features")
    centres, labels = _kmeans2(features, np.random.default_rng(seed))
    order = sorted(range(2), key=lambda index: tuple(centres[index]))
    centres, labels = centres[order], np.array([order.index(label) for label in labels])
    first, second = (
        _outlier_limit(np.linalg.norm(features[labels == index] - centres[index], axis=1)) for index in range(2)
    )
    return KitClusters(centres=centres, limits=(first, second))


def _kmeans2(features: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return the two centres and the cluster of each feature (Lloyd's method, k-means++ start)."""
    first = features[rng.integers(len(features))]
    squared = np.sum((features - first) ** 2, axis=1)
    second = features[rng.choice(len(features), p=squared / squared.sum())]
    centres = np.stack([first, second])
    labels = _nearest(features, centres)
    for _ in range(_MAX_ITERATIONS):
        for index in range(2):
            members = features[labels == index]
            if len(members):
                centres[index] = members.mean(axis=0)
        new_labels = _nearest(features, centres)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return centres, labels


def _nearest(features: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Return the index of the nearest centre of each feature."""
    return np.argmin(np.linalg.norm(features[:, None, :] - centres[None, :, :], axis=2), axis=1)


def _outlier_limit(distances: np.ndarray) -> float:
    """Return the distance from which a feature is an outlier of a cluster with these member distances."""
    if len(distances) == 0:
        return 0.0
    median = float(np.median(distances))
    spread = _MAD_SCALE * float(np.median(np.abs(distances - median)))
    return median + max(OUTLIER_MADS * spread, _MIN_MARGIN)


def kit_roles(tracks: Sequence[Track], mean_position: Mapping[int, tuple[float, float]]) -> list[Track]:
    """Give the tracks of mostly outlier kits the role goalkeeper or referee (module docstring, step 4).

    ``mean_position`` maps a track id to the mean pitch position (x, y) of the track; a track
    without a position is not in it. A track of outliers without a position is a referee: no
    position puts it in a goal. The other tracks are returned as they are.
    """
    decided: list[Track] = []
    for track in tracks:
        if not _mostly_outliers(track):
            decided.append(track)
            continue
        role = _special_role(mean_position.get(track.track_id))
        observations = tuple(
            observation.model_copy(update={"role": role, "team_key": "none"}) for observation in track.observations
        )
        decided.append(track.model_copy(update={"observations": observations}))
    return decided


def _mostly_outliers(track: Track) -> bool:
    """Tell whether more than half of the box height of a track lies on observations without a team key."""
    total = sum(observation.box.height for observation in track.observations)
    outside = sum(observation.box.height for observation in track.observations if observation.team_key == "none")
    return 2.0 * outside > total


def _special_role(position: tuple[float, float] | None) -> Role:
    """Return goalkeeper for a mean position in a goal zone inside the width of the penalty area, else referee."""
    if position is None:
        return "referee"
    mean_x, mean_y = position
    if goal_side(mean_x) is not None and abs(mean_y) <= PENALTY_AREA_HALF_WIDTH:
        return "goalkeeper"
    return "referee"
