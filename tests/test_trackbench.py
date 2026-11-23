"""Tests of the synthetic bench of the track repairs, on short clips of the synthetic scene."""

from __future__ import annotations

import pytest

from pitchvlm.evalx.trackbench import ARMS, FAULTS, CaseResult, run_case, table_lines

#: 15 seconds: shorter than the clips of the bench, so that the tests run fast.
SHORT_CLIP = 375

#: Each case runs the pipeline steps of every arm on 375 frames (about 4 s for the module).
pytestmark = pytest.mark.slow


def by_arm(results: list[CaseResult]) -> dict[str, CaseResult]:
    """Index the results of one case by arm."""
    return {result.arm: result for result in results}


@pytest.fixture(scope="module")
def clean() -> dict[str, CaseResult]:
    return by_arm(run_case("none", 0, n_frames=SHORT_CLIP))


@pytest.fixture(scope="module")
def swapped() -> dict[str, CaseResult]:
    return by_arm(run_case("swap", 0, n_frames=SHORT_CLIP))


def test_every_arm_runs_on_the_same_case(clean: dict[str, CaseResult]) -> None:
    assert list(clean) == list(ARMS) == ["current", "swaps", "slots", "both"]
    assert {(result.fault, result.seed, result.swaps) for result in clean.values()} == {("none", 0, 0)}


def test_the_split_of_team_swaps_changes_nothing_on_clean_answers(clean: dict[str, CaseResult]) -> None:
    assert clean["swaps"].scores == clean["current"].scores
    assert clean["both"].scores == clean["slots"].scores


def test_the_split_of_team_swaps_undoes_an_injected_swap(
    clean: dict[str, CaseResult], swapped: dict[str, CaseResult]
) -> None:
    assert swapped["current"].swaps == 1
    # The swap costs switches and mixes two people under one id ...
    assert swapped["current"].scores.switches > clean["current"].scores.switches
    assert swapped["current"].scores.purity < 1.0
    # ... and the repair gives back the values of the clip without the swap.
    assert swapped["swaps"].scores == clean["current"].scores
    assert swapped["both"].scores == clean["slots"].scores


def test_team_slots_link_the_fragments_of_people_who_came_back(clean: dict[str, CaseResult]) -> None:
    current, slots = clean["current"], clean["slots"]
    assert slots.tracks < current.tracks
    assert slots.scores.track_ratio < current.scores.track_ratio
    assert slots.scores.coverage > current.scores.coverage
    assert slots.scores.switches < current.scores.switches
    # In the scene every link of the slots is right. The scene's players stay near their places,
    # which makes the links easy; real players move more.
    assert slots.scores.purity == 1.0


def test_the_table_has_one_row_per_fault_and_arm_and_one_difference_row_per_repair(
    clean: dict[str, CaseResult],
) -> None:
    lines = table_lines(list(clean.values()) + list(by_arm(run_case("none", 1, n_frames=SHORT_CLIP)).values()))

    assert lines[0] == "| fault | arm | cases | swaps | coverage | switches | track ratio | purity | tracks |"
    rows = [line for line in lines if line.startswith("| none |")]
    assert len(rows) == 4 + 3
    assert rows[0].startswith("| none | current | 2 | 0 |")
    assert rows[4].startswith("| none | swaps | 2 | +0.000 [+0.000, +0.000] |")


def test_the_bench_refuses_an_unknown_fault() -> None:
    assert "none" in FAULTS
    with pytest.raises(ValueError, match="unknown fault"):
        run_case("meteor", 0, n_frames=SHORT_CLIP)
