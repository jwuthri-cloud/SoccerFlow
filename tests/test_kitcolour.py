"""Tests of the team keys and roles from kit colours (``kitcolour``)."""

from __future__ import annotations

import numpy as np
import pytest

from pitchvlm import kitcolour
from pitchvlm.kitcolour import FEATURE_SIZE, HUE_BINS, fit_kit_clusters, kit_roles, torso_feature
from pitchvlm.schemas import Box, TeamKey, Track, TrackObs

RED_BGR = (40, 40, 220)
BLUE_BGR = (220, 90, 40)
YELLOW_BGR = (0, 230, 255)
BLACK_BGR = (20, 20, 20)
WHITE_BGR = (250, 250, 250)
GRASS_BGR = (70, 160, 80)


def flat_image(colour: tuple[int, int, int], size: tuple[int, int] = (80, 120)) -> np.ndarray:
    """Return an image of one colour, (width, height) pixels."""
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = colour
    return image


def feature_of(colour: tuple[int, int, int]) -> np.ndarray:
    feature = torso_feature(flat_image(colour), Box(x1=0, y1=0, x2=80, y2=120))
    assert feature is not None
    return feature


def test_flat_kit_fills_one_bin_and_the_feature_has_unit_length() -> None:
    for colour in (RED_BGR, BLUE_BGR, YELLOW_BGR, BLACK_BGR, WHITE_BGR):
        feature = feature_of(colour)
        assert feature.shape == (FEATURE_SIZE,)
        assert np.count_nonzero(feature) == 1
        assert np.linalg.norm(feature) == pytest.approx(1.0)
    # Black and white have their own bins after the hue bins: dark first, light last.
    assert np.argmax(feature_of(BLACK_BGR)) == HUE_BINS
    assert np.argmax(feature_of(WHITE_BGR)) == HUE_BINS + 2
    # Red, blue and yellow fall into three different hue bins.
    hues = {int(np.argmax(feature_of(colour))) for colour in (RED_BGR, BLUE_BGR, YELLOW_BGR)}
    assert len(hues) == 3 and max(hues) < HUE_BINS


def test_feature_reads_the_torso_window_only() -> None:
    """Grass around the torso window, and the legs below it, do not count."""
    image = flat_image(GRASS_BGR, (200, 200))
    box = Box(x1=40, y1=20, x2=120, y2=180)  # 80 x 160: the torso window is x 60-100, y 44-92.
    image[44:92, 60:100] = RED_BGR
    feature = torso_feature(image, box)
    assert feature is not None
    np.testing.assert_allclose(feature, feature_of(RED_BGR))


def test_feature_of_a_two_colour_shirt_shares_its_bins() -> None:
    image = flat_image(RED_BGR)
    image[:, 40:] = WHITE_BGR
    feature = torso_feature(image, Box(x1=0, y1=0, x2=80, y2=120))
    assert feature is not None
    # Half of the window is red and half is white: each share is 0.5, so each value is sqrt(0.5).
    np.testing.assert_allclose(sorted(feature[feature > 0]), [np.sqrt(0.5)] * 2)


def test_box_too_small_or_outside_gives_no_feature() -> None:
    image = flat_image(RED_BGR, (100, 100))
    assert torso_feature(image, Box(x1=10, y1=10, x2=14, y2=20)) is None
    assert torso_feature(image, Box(x1=200, y1=200, x2=260, y2=320)) is None
    # A box cut at the image border keeps the part of its window that is inside.
    assert torso_feature(image, Box(x1=-40, y1=0, x2=40, y2=120)) is not None


def noisy(feature: np.ndarray, rng: np.random.Generator, count: int) -> np.ndarray:
    """Return ``count`` copies of a feature with a little noise, each with unit length."""
    rows = np.abs(feature + rng.normal(0.0, 0.03, size=(count, FEATURE_SIZE)))
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def team_features(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return noisy(feature_of(RED_BGR), rng, 40), noisy(feature_of(BLUE_BGR), rng, 40), rng.permutation(80)


def test_two_kits_become_two_clusters_and_other_kits_are_outliers() -> None:
    red, blue, _ = team_features()
    clusters = fit_kit_clusters(np.vstack([red, blue, feature_of(YELLOW_BGR)[None], feature_of(BLACK_BGR)[None]]))
    red_keys = {clusters.team_key(row) for row in red}
    blue_keys = {clusters.team_key(row) for row in blue}
    assert len(red_keys) == 1 and len(blue_keys) == 1
    assert red_keys | blue_keys == {"A", "B"}
    assert clusters.team_key(feature_of(YELLOW_BGR)) == "none"
    assert clusters.team_key(feature_of(BLACK_BGR)) == "none"
    assert clusters.team_key(None) == "none"


def test_cluster_names_do_not_depend_on_the_order_of_the_rows_or_the_seed() -> None:
    red, blue, order = team_features()
    features = np.vstack([red, blue])
    first = fit_kit_clusters(features, seed=0)
    for clusters in (fit_kit_clusters(features[order], seed=0), fit_kit_clusters(features, seed=5)):
        np.testing.assert_allclose(clusters.centres, first.centres)
        assert clusters.team_key(red[0]) == first.team_key(red[0])


def test_outlier_limit_has_a_floor_for_a_cluster_of_equal_features() -> None:
    features = np.vstack(
        [np.repeat(feature_of(RED_BGR)[None], 10, axis=0), np.repeat(feature_of(BLUE_BGR)[None], 10, axis=0)]
    )
    clusters = fit_kit_clusters(features)
    assert min(clusters.limits) == pytest.approx(kitcolour._MIN_MARGIN)
    # A red shirt with a small change still belongs to the red team.
    changed = feature_of(RED_BGR) * 0.999 + feature_of(WHITE_BGR) * 0.04
    assert clusters.team_key(changed / np.linalg.norm(changed)) == clusters.team_key(feature_of(RED_BGR))


@pytest.mark.parametrize(
    "features",
    [np.zeros((5, FEATURE_SIZE - 1)), np.zeros(FEATURE_SIZE), np.repeat(np.eye(FEATURE_SIZE)[:1], 4, axis=0)],
)
def test_wrong_features_are_refused(features: np.ndarray) -> None:
    with pytest.raises(ValueError):
        fit_kit_clusters(features)


# --- Roles per track -----------------------------------------------------------------


def track(track_id: int, keys: list[TeamKey], heights: list[float] | None = None) -> Track:
    heights = heights or [100.0] * len(keys)
    observations = tuple(
        TrackObs(frame=frame, box=Box(x1=0, y1=0, x2=40, y2=height), role="player", team_key=key)
        for frame, (key, height) in enumerate(zip(keys, heights, strict=True))
    )
    return Track(track_id=track_id, observations=observations)


@pytest.mark.parametrize(
    ("position", "role"),
    [
        ((-48.0, 2.0), "goalkeeper"),  # deep in the left penalty area
        ((47.0, -19.0), "goalkeeper"),  # right goal zone, inside the width of the penalty area
        ((-48.0, 33.0), "referee"),  # a linesman near the touchline at the depth of the goal zone
        ((0.0, 0.0), "referee"),
        ((-30.0, 0.0), "referee"),  # in front of the goal zone
        (None, "referee"),  # no pitch position: nothing puts the track in a goal
    ],
)
def test_track_of_outlier_kits_is_goalkeeper_or_referee_by_its_place(
    position: tuple[float, float] | None, role: str
) -> None:
    special = track(3, ["none", "none", "A", "none"])
    means = {} if position is None else {3: position}
    (decided,) = kit_roles([special], means)
    assert {observation.role for observation in decided.observations} == {role}
    assert {observation.team_key for observation in decided.observations} == {"none"}


def test_track_of_a_team_kit_keeps_its_observations() -> None:
    player = track(1, ["A", "none", "A", "B"])
    assert kit_roles([player], {1: (-48.0, 0.0)}) == [player]


def test_outlier_share_is_weighted_by_box_height() -> None:
    """Three small boxes without a team weigh less than one tall box of a team."""
    mixed = track(2, ["none", "none", "none", "A"], heights=[20.0, 20.0, 20.0, 100.0])
    assert kit_roles([mixed], {2: (0.0, 0.0)}) == [mixed]
