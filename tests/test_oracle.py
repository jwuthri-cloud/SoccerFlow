"""Tests of the label-only helpers: shift, keep, resample, team-side rule, label homography."""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import pytest

from pitchvlm.evalx.oracle import (
    TeamSideFacts,
    keep_frames,
    label_homography_residuals_m,
    resample_states,
    shift_states,
    team_side_by_rule,
    team_side_facts,
)
from pitchvlm.labels import GsrAthlete, GsrFrame, GsrLabels
from pitchvlm.schemas import Box, FrameState, PlayerState, Role, TeamKey, TeamSide, Track, TrackObs
from pitchvlm.tracking import assign_team_sides

BOX = Box(x1=100.0, y1=200.0, x2=140.0, y2=280.0)


def player(
    track_id: int,
    x: float,
    y: float,
    *,
    role: Role = "player",
    side: TeamSide | None = None,
    number: int | None = None,
    box: Box | None = None,
) -> PlayerState:
    """Build an observed person at the pitch position ``(x, y)``."""
    return PlayerState(
        track_id=track_id, box=box, pitch_x=x, pitch_y=y, role=role, team_side=side, number_export=number, observed=True
    )


def state(frame: int, *players: PlayerState) -> FrameState:
    """Build the state of one frame from its people."""
    return FrameState(frame=frame, players=players)


# --- shift_states --------------------------------------------------------------------


def test_shift_states_moves_every_position() -> None:
    frames = [state(0, player(1, 1.0, 2.0), player(2, -3.0, -4.0)), state(1, player(1, 1.5, 2.5))]

    shifted = shift_states(frames, 1.0)

    assert [s.frame for s in shifted] == [0, 1]
    assert [(p.pitch_x, p.pitch_y) for p in shifted[0].players] == [(2.0, 2.0), (-2.0, -4.0)]
    assert [(p.pitch_x, p.pitch_y) for p in shifted[1].players] == [(2.5, 2.5)]


def test_shift_states_dx_is_along_the_pitch_and_dy_is_across() -> None:
    moved = shift_states([state(0, player(1, 0.0, 0.0))], -2.0, 3.0)[0].players[0]

    # Negative dx: towards the left goal. Positive dy: towards the camera (near side).
    assert (moved.pitch_x, moved.pitch_y) == (-2.0, 3.0)


def test_shift_states_keeps_the_other_fields_and_does_not_change_its_input() -> None:
    original = player(1, 1.0, 2.0, side="left", number=10, box=BOX)
    frames = [state(0, original)]

    moved = shift_states(frames, 1.0, 1.0)[0].players[0]

    assert moved == original.model_copy(update={"pitch_x": 2.0, "pitch_y": 3.0})
    assert (moved.track_id, moved.box, moved.team_side, moved.number_export) == (1, BOX, "left", 10)
    assert moved.observed is True
    assert (frames[0].players[0].pitch_x, frames[0].players[0].pitch_y) == (1.0, 2.0)


# --- keep_frames ---------------------------------------------------------------------


def test_keep_frames_keeps_the_named_frames_in_their_old_order() -> None:
    frames = [state(frame, player(1, float(frame), 0.0)) for frame in range(6)]

    kept = keep_frames(frames, [5, 0, 3, 99])

    assert [s.frame for s in kept] == [0, 3, 5]
    assert kept[1] is frames[3]


def test_keep_frames_with_nothing_to_keep() -> None:
    frames = [state(frame) for frame in range(3)]

    assert keep_frames(frames, set()) == []
    assert keep_frames([], {0, 1}) == []


# --- resample_states -----------------------------------------------------------------


def curved_truth() -> list[FrameState]:
    """Truth of 11 frames. Track 1 moves on a curve: x = 0.1 * frame^2, y = -frame. Track 2 is in frames 0 to 4."""
    frames = []
    for frame in range(11):
        players = [player(1, 0.1 * frame * frame, -1.0 * frame, side="left", number=10, box=BOX)]
        if frame <= 4:
            players.append(player(2, 30.0, 5.0, side="right"))
        frames.append(state(frame, *players))
    return frames


def test_resample_gives_the_midpoint_between_two_keyframes() -> None:
    resampled = resample_states(curved_truth(), [0, 10])

    assert [s.frame for s in resampled] == list(range(11))
    middle = resampled[5].players
    assert [p.track_id for p in middle] == [1]
    # The truth at frame 5 is (2.5, -5.0). The straight line from (0, 0) to (10, -10) gives (5.0, -5.0).
    assert (middle[0].pitch_x, middle[0].pitch_y) == pytest.approx((5.0, -5.0))


def test_resample_keeps_the_truth_at_the_keyframes() -> None:
    truth = curved_truth()

    resampled = resample_states(truth, [0, 4, 10])

    assert resampled[0] is truth[0]
    assert resampled[4] is truth[4]
    assert resampled[10] is truth[10]


def test_resample_is_linear_between_each_pair_of_keyframes() -> None:
    resampled = resample_states(curved_truth(), [0, 4, 10])

    # Keyframe truth of track 1: x = 0.0 at frame 0, 1.6 at frame 4, 10.0 at frame 10.
    xs = [s.players[0].pitch_x for s in resampled]
    ys = [s.players[0].pitch_y for s in resampled]
    assert xs == pytest.approx([0.0, 0.4, 0.8, 1.2, 1.6, 3.0, 4.4, 5.8, 7.2, 8.6, 10.0])
    assert ys == pytest.approx([-float(frame) for frame in range(11)])


def test_resample_holds_a_track_only_between_keyframes_that_both_hold_it() -> None:
    resampled = resample_states(curved_truth(), [0, 4, 10])

    # Track 2 is in the keyframes 0 and 4, but not in the keyframe 10.
    assert [[p.track_id for p in s.players] for s in resampled[:5]] == [[1, 2]] * 5
    assert [[p.track_id for p in s.players] for s in resampled[5:]] == [[1]] * 6
    assert (resampled[2].players[1].pitch_x, resampled[2].players[1].pitch_y) == pytest.approx((30.0, 5.0))


def test_resample_does_not_fill_a_track_that_is_missing_at_the_next_keyframe() -> None:
    resampled = resample_states(curved_truth(), [0, 10])

    # Track 2 is in the truth of frames 1 to 4, but the keyframe 10 does not hold it.
    assert [p.track_id for p in resampled[0].players] == [1, 2]
    assert all([p.track_id for p in s.players] == [1] for s in resampled[1:])


def test_resampled_people_keep_their_identity_and_are_marked_as_filled() -> None:
    filled = resample_states(curved_truth(), [0, 10])[3].players[0]

    assert (filled.track_id, filled.role, filled.team_side, filled.number_export) == (1, "player", "left", 10)
    assert filled.observed is False
    assert filled.box is None


def test_resample_with_keyframes_next_to_each_other_is_the_truth() -> None:
    truth = curved_truth()

    assert resample_states(truth, [2, 3, 4]) == truth[2:5]


def test_resample_of_one_keyframe_and_of_none() -> None:
    truth = curved_truth()

    assert resample_states(truth, [3]) == [truth[3]]
    assert resample_states(truth, []) == []


@pytest.mark.parametrize("keyframes", [[4, 0], [0, 0, 4], [0, 4, 4]])
def test_resample_refuses_keyframes_that_do_not_increase(keyframes: list[int]) -> None:
    with pytest.raises(ValueError, match="increasing"):
        resample_states(curved_truth(), keyframes)


def test_resample_refuses_a_keyframe_without_a_state() -> None:
    with pytest.raises(ValueError, match=r"\[12\]"):
        resample_states(curved_truth(), [0, 12])


def test_resample_refuses_a_frame_that_is_present_twice() -> None:
    truth = curved_truth()

    with pytest.raises(ValueError, match="more than once"):
        resample_states([*truth, truth[0]], [0, 10])


# --- Team-side rule ------------------------------------------------------------------


def two_teams(left_xs: Sequence[float], right_xs: Sequence[float], n_frames: int = 3) -> list[FrameState]:
    """Build frames with standing players. Players labelled left get the ids 1, 2, ...; right gets 101, 102, ..."""
    return [
        state(
            frame,
            *(player(1 + index, x, 0.0, side="left") for index, x in enumerate(left_xs)),
            *(player(101 + index, x, 0.0, side="right") for index, x in enumerate(right_xs)),
        )
        for frame in range(n_frames)
    ]


def test_rule_gives_left_to_the_team_with_the_lower_mean_x() -> None:
    frames = two_teams([-20.0, -10.0], [15.0, 25.0])

    assert team_side_by_rule(frames) == {1: "left", 2: "left", 101: "right", 102: "right"}
    assert team_side_facts(frames).agrees is True


def test_rule_disagrees_when_the_labelled_sides_are_swapped() -> None:
    # The team labelled left stands at positive x, in the right half.
    frames = two_teams([15.0, 25.0], [-20.0, -10.0])

    assert team_side_by_rule(frames) == {1: "right", 2: "right", 101: "left", 102: "left"}
    assert team_side_facts(frames).agrees is False


def test_rule_compares_team_means_not_single_players() -> None:
    # One player of the left team stands far in the right half. The team mean is still lower.
    frames = two_teams([-30.0, -20.0, 40.0], [10.0, 20.0, 30.0])

    assert team_side_facts(frames).agrees is True


def test_rule_uses_players_only() -> None:
    frames = [
        state(
            frame,
            player(1, -5.0, 0.0, side="left"),
            player(101, 5.0, 0.0, side="right"),
            # With this goalkeeper the mean x of the team labelled left would be far in the right half.
            player(50, 500.0, 0.0, role="goalkeeper", side="left"),
            player(60, -500.0, 0.0, role="referee"),
        )
        for frame in range(3)
    ]

    assert team_side_by_rule(frames) == {1: "left", 101: "right"}
    assert team_side_facts(frames).agrees is True


def test_rule_uses_the_mean_x_of_a_track_over_its_frames() -> None:
    # Track 1 (labelled left) crosses the halfway line: x = -30, then +10 and +10. Its mean is below 0.
    frames = [
        state(0, player(1, -30.0, 0.0, side="left"), player(101, 0.0, 0.0, side="right")),
        state(1, player(1, 10.0, 0.0, side="left"), player(101, 0.0, 0.0, side="right")),
        state(2, player(1, 10.0, 0.0, side="left"), player(101, 0.0, 0.0, side="right")),
    ]

    assert team_side_facts(frames).agrees is True


def test_rule_counts_a_track_with_its_number_of_frames() -> None:
    # Left team: track 1 at x = +10 in 9 frames, and track 2 at x = -30 in 1 frame. Right team: x = 0.
    frames = [
        state(frame, player(1, 10.0, 0.0, side="left"), player(101, 0.0, 0.0, side="right")) for frame in range(9)
    ]
    frames[0] = state(0, *frames[0].players, player(2, -30.0, 0.0, side="left"))

    # Mean x of the left team: (9 * 10 - 30) / 10 = +6, so the rule calls it the right team.
    # The plain mean of the two track means would be -10.
    assert team_side_by_rule(frames) == {1: "right", 2: "right", 101: "left"}
    assert team_side_facts(frames).agrees is False


def test_rule_divides_the_sum_of_a_team_by_its_number_of_frames() -> None:
    # Left team: one track at x = -1 in 10 frames. Right team: two tracks at x = -3 in 1 frame each.
    frames = [state(frame, player(1, -1.0, 0.0, side="left")) for frame in range(10)]
    frames[0] = state(
        0, *frames[0].players, player(101, -3.0, 0.0, side="right"), player(102, -3.0, 0.0, side="right")
    )

    # Mean x of the left team: -10 / 10 frames = -1. Mean x of the right team: -6 / 2 frames = -3. The team
    # labelled left has the higher mean x, so the rule calls it the right team. Divided by the number of
    # tracks, the mean of the left team would be -10 / 1 = -10, and the rule would agree with the labels.
    assert team_side_by_rule(frames) == {1: "right", 101: "left", 102: "left"}
    assert team_side_facts(frames).agrees is False


def test_rule_counts_an_exact_tie_of_the_team_means_as_wrong() -> None:
    # Both teams have the mean x 0, so the rule cannot tell them apart.
    frames = two_teams([-5.0, 5.0], [0.0])

    assert team_side_by_rule(frames) == {1: "right", 2: "right", 101: "left"}
    assert team_side_facts(frames).agrees is False


def test_rule_takes_the_most_frequent_labelled_side_of_a_track() -> None:
    # Track 1 is labelled left in three frames and right in one frame.
    frames = [
        state(frame, player(1, -10.0, 0.0, side="left"), player(101, 10.0, 0.0, side="right")) for frame in range(3)
    ]
    frames.append(state(3, player(1, -10.0, 0.0, side="right"), player(101, 10.0, 0.0, side="right")))

    # Track 1 counts as a track of the left team. If it counted as a track of the right team, both tracks
    # would be in that team, whose mean x is 0, and the rule would give both the side right.
    assert team_side_by_rule(frames) == {1: "left", 101: "right"}
    assert team_side_facts(frames).agrees is True


def test_rule_with_one_team_uses_the_sign_of_its_mean_x() -> None:
    in_the_left_half = two_teams([-12.0, -3.0], [])
    in_the_right_half = two_teams([12.0, 3.0], [])

    assert team_side_by_rule(in_the_left_half) == {1: "left", 2: "left"}
    assert team_side_facts(in_the_left_half).agrees is True
    assert team_side_by_rule(in_the_right_half) == {1: "right", 2: "right"}
    assert team_side_facts(in_the_right_half).agrees is False
    # A mean x of exactly 0 is not negative: the team is not the left one.
    assert team_side_by_rule(two_teams([-4.0, 4.0], [])) == {1: "right", 2: "right"}


def test_rule_without_a_labelled_player() -> None:
    frames = [state(0, player(1, 5.0, 0.0), player(2, -5.0, 0.0, role="referee"))]

    assert team_side_by_rule(frames) == {}
    with pytest.raises(ValueError, match="no player has a labelled team side"):
        team_side_facts(frames)


def corner_kick(*goalkeepers: PlayerState, n_frames: int = 3) -> list[FrameState]:
    """Build a corner kick at the left goal: the attackers (labelled right) stand 1 m deeper than the defenders.

    The mean pitch x of the defenders is -45 and of the attackers -46. So the players alone give the
    sides the wrong way round, and their means are 1 m apart.
    """
    return [
        state(
            frame,
            player(1, -46.0, -3.0, side="left"),
            player(2, -44.0, 3.0, side="left"),
            player(101, -47.0, -2.0, side="right"),
            player(102, -45.0, 2.0, side="right"),
            *goalkeepers,
        )
        for frame in range(n_frames)
    ]


def test_goalkeeper_in_a_goal_decides_when_the_player_means_are_close() -> None:
    frames = corner_kick(player(50, -52.0, 0.0, role="goalkeeper", side="left"))

    # The goalkeeper is not in the result: the rule is judged on the player tracks.
    assert team_side_by_rule(frames) == {1: "left", 2: "left", 101: "right", 102: "right"}
    assert team_side_facts(frames).agrees is True
    assert team_side_facts(frames) == TeamSideFacts(
        agrees=True,
        decided_by="goalkeepers",
        players_alone_agree=False,
        player_mean_gap_m=pytest.approx(1.0),
        goalkeeper_tracks=1,
        goalkeepers_in_own_goal_zone=1,
        goalkeepers_in_other_goal_zone=0,
    )


def test_players_decide_when_no_goalkeeper_stands_in_a_goal() -> None:
    without = team_side_facts(corner_kick())
    # 35.9 m from the halfway line is 0.1 m outside the depth of the penalty area.
    outside = team_side_facts(corner_kick(player(50, -35.9, 0.0, role="goalkeeper", side="left")))
    # A goalkeeper without a labelled side says nothing about the labelled teams.
    unlabelled = team_side_facts(corner_kick(player(50, -52.0, 0.0, role="goalkeeper")))

    for facts in (without, outside, unlabelled):
        assert (facts.agrees, facts.decided_by, facts.players_alone_agree) == (False, "players", False)
    assert (without.goalkeeper_tracks, outside.goalkeeper_tracks, unlabelled.goalkeeper_tracks) == (0, 1, 0)
    assert (outside.goalkeepers_in_own_goal_zone, outside.goalkeepers_in_other_goal_zone) == (0, 0)


def test_goalkeeper_in_the_goal_of_the_other_side_votes_for_swapped_sides() -> None:
    # The labels call the team of this goalkeeper "right", but the goalkeeper stands in the left goal.
    frames = corner_kick(player(60, -52.0, 0.0, role="goalkeeper", side="right"))

    facts = team_side_facts(frames)

    assert team_side_by_rule(frames) == {1: "right", 2: "right", 101: "left", 102: "left"}
    assert (facts.agrees, facts.decided_by) == (False, "goalkeepers")
    assert (facts.goalkeepers_in_own_goal_zone, facts.goalkeepers_in_other_goal_zone) == (0, 1)


@pytest.mark.parametrize(("gap_m", "decided_by", "agrees"), [(3.0, "players", False), (2.99, "goalkeepers", True)])
def test_players_decide_alone_when_their_means_are_three_metres_apart(
    gap_m: float, decided_by: str, agrees: bool
) -> None:
    # The team labelled left stands deeper in the right half than the team labelled right: the players
    # alone swap the sides. The goalkeeper of the team labelled left stands in the left goal.
    frames = [
        state(
            frame,
            player(1, 10.0 + gap_m, 0.0, side="left"),
            player(101, 10.0, 0.0, side="right"),
            player(50, -50.0, 0.0, role="goalkeeper", side="left"),
        )
        for frame in range(3)
    ]

    facts = team_side_facts(frames)

    assert facts.player_mean_gap_m == pytest.approx(gap_m)
    assert (facts.decided_by, facts.agrees, facts.players_alone_agree) == (decided_by, agrees, False)


def test_goalkeepers_count_with_their_frames_times_their_depth_in_the_goal_zone() -> None:
    # Goalkeeper 50 (labelled left) stands 16 m deep in the left goal zone for 2 frames: weight 32.
    # Goalkeeper 60 (labelled right) stands 1 m deep in the same zone, as after a corner kick of its team.
    deep = [player(50, -52.0, 0.0, role="goalkeeper", side="left")]
    shallow = [player(60, -37.0, 0.0, role="goalkeeper", side="right")]

    def clip(n_shallow_frames: int) -> list[FrameState]:
        frames = corner_kick(n_frames=40)
        return [
            state(
                frame.frame,
                *frame.players,
                *(deep if frame.frame < 2 else []),
                *(shallow if frame.frame < n_shallow_frames else []),
            )
            for frame in frames
        ]

    # 31 frames of the shallow goalkeeper weigh 31, 33 frames weigh 33, and 32 frames give a tie.
    assert team_side_facts(clip(31)).decided_by == "goalkeepers"
    assert team_side_facts(clip(31)).agrees is True
    assert team_side_facts(clip(33)).decided_by == "goalkeepers"
    assert team_side_facts(clip(33)).agrees is False
    # On a tie the goalkeepers do not decide, and the players give the sides the wrong way round.
    assert team_side_facts(clip(32)).decided_by == "players"
    assert team_side_facts(clip(32)).agrees is False


def test_goal_zone_starts_at_the_penalty_area_line() -> None:
    on_the_line = team_side_facts(corner_kick(player(50, -36.0, 0.0, role="goalkeeper", side="left")))
    inside = team_side_facts(corner_kick(player(50, -36.5, 0.0, role="goalkeeper", side="left")))

    # A goalkeeper on the line of the penalty area is in the zone, with a depth of 0: no weight.
    assert (on_the_line.goalkeepers_in_own_goal_zone, on_the_line.decided_by) == (1, "players")
    assert (inside.goalkeepers_in_own_goal_zone, inside.decided_by) == (1, "goalkeepers")


def test_goalkeeper_decides_an_exact_tie_of_the_player_means() -> None:
    frames = two_teams([-5.0, 5.0], [0.0])
    with_goalkeeper = [
        state(frame.frame, *frame.players, player(60, 51.0, 0.0, role="goalkeeper", side="right")) for frame in frames
    ]

    assert team_side_facts(frames).agrees is False
    assert team_side_facts(with_goalkeeper).agrees is True
    assert team_side_facts(with_goalkeeper).player_mean_gap_m == 0.0


def test_facts_of_a_clip_with_one_team_of_players_have_no_gap() -> None:
    facts = team_side_facts(two_teams([-12.0, -3.0], []))

    assert facts.player_mean_gap_m is None
    assert (facts.agrees, facts.decided_by, facts.players_alone_agree) == (True, "players", True)


def labelled_tracks(frames: Sequence[FrameState]) -> tuple[list[Track], dict[int, float]]:
    """Turn labelled frames into tracks of the pipeline: the team labelled left has the key A, the other one B.

    The result also holds the mean pitch x of every track. Every observation has the same box, so
    every frame has the same weight in the votes.
    """
    key_of_side: dict[TeamSide | None, TeamKey] = {"left": "A", "right": "B", None: "none"}
    observations: dict[int, list[TrackObs]] = {}
    xs: dict[int, list[float]] = {}
    for frame in frames:
        for person in frame.players:
            observation = TrackObs(frame=frame.frame, box=BOX, role=person.role, team_key=key_of_side[person.team_side])
            observations.setdefault(person.track_id, []).append(observation)
            xs.setdefault(person.track_id, []).append(person.pitch_x)
    tracks = [Track(track_id=track_id, observations=tuple(seen)) for track_id, seen in observations.items()]
    return tracks, {track_id: sum(values) / len(values) for track_id, values in xs.items()}


def test_rule_on_the_labels_is_the_rule_of_the_pipeline() -> None:
    """The label stage must measure the rule that the pipeline uses, so both get the same random clips.

    Each clip has two teams of standing players whose means are 0 to 6 m apart, and 0 to 2 goalkeepers
    on any side at any depth. The team labelled left is team A of the pipeline.
    """
    rng = random.Random(20261114)
    decided_by: set[str] = set()
    for _ in range(300):
        centre = rng.uniform(-45.0, 45.0)
        gap = rng.choice([0.5, 1.5, 2.5, 2.9, 3.1, 4.0, 6.0]) * rng.choice([-1.0, 1.0])
        people = [player(1 + index, centre + rng.uniform(-4.0, 4.0), 0.0, side="left") for index in range(4)]
        people += [player(101 + index, centre + gap + rng.uniform(-4.0, 4.0), 0.0, side="right") for index in range(4)]
        keepers = [
            player(50 + index, rng.choice([-1.0, 1.0]) * rng.uniform(30.0, 52.5), 0.0, role="goalkeeper", side=side)
            for index, side in enumerate(rng.sample(["left", "right"], k=rng.randint(0, 2)))
        ]
        # A goalkeeper is in the image in some of the frames only, so the weights of two goalkeepers differ.
        frames = [
            state(frame, *people, *(keeper for index, keeper in enumerate(keepers) if frame % (index + 2) == 0))
            for frame in range(6)
        ]

        tracks, mean_pitch_x = labelled_tracks(frames)
        pipeline_sides = assign_team_sides(tracks, mean_pitch_x)
        facts = team_side_facts(frames)

        assert team_side_by_rule(frames)[1] == pipeline_sides["A"]
        assert team_side_by_rule(frames)[101] == pipeline_sides["B"]
        assert facts.agrees == (pipeline_sides == {"A": "left", "B": "right"})
        decided_by.add(facts.decided_by)
    assert decided_by == {"players", "goalkeepers"}


# --- label_homography_residuals_m ------------------------------------------------------

#: Pitch metres -> image pixels of a made-up camera: about 8 pixels per metre and a little perspective.
PITCH_TO_IMAGE = np.array([[8.0, 2.0, 960.0], [0.5, 6.0, 540.0], [0.0, -0.002, 1.0]])

#: Eight pitch positions (metres) that are spread over the pitch and are not on one line.
SPREAD = np.array(
    [
        [-40.0, -25.0],
        [-30.0, 20.0],
        [-15.0, -5.0],
        [0.0, 28.0],
        [5.0, -20.0],
        [20.0, 10.0],
        [35.0, -28.0],
        [45.0, 22.0],
    ]
)


def project(pitch_xy: np.ndarray) -> np.ndarray:
    """Map pitch metres, shape (n, 2), to image pixels, shape (n, 2), with the made-up camera."""
    homogeneous = np.hstack([pitch_xy, np.ones((len(pitch_xy), 1))]) @ PITCH_TO_IMAGE.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def athletes_at(pitch_xy: np.ndarray) -> list[GsrAthlete]:
    """Build athletes whose box foot point is the exact image of their pitch position.

    Every box has another height: 40 pixels, 55 pixels, and so on. So the foot points follow the
    made-up camera, but the box centres do not follow any homography.
    """
    return [
        GsrAthlete(
            track_id=index + 1,
            box=Box(x1=float(u) - 10.0, y1=float(v) - 40.0 - 15.0 * index, x2=float(u) + 10.0, y2=float(v)),
            pitch_x=float(x),
            pitch_y=float(y),
            role="player",
            team="left",
            jersey=None,
        )
        for index, ((u, v), (x, y)) in enumerate(zip(project(pitch_xy), pitch_xy))
    ]


def gsr_labels(*frames: tuple[Sequence[GsrAthlete], bool]) -> GsrLabels:
    """Build labels from ``(athletes, scored)`` pairs, one pair per frame."""
    return GsrLabels(
        clip_id="clip",
        fps=25.0,
        n_frames=len(frames),
        width=1920,
        height=1080,
        frames=tuple(
            GsrFrame(
                frame=index,
                image_id=str(9001 + index),
                file_name=f"{index + 1:06d}.jpg",
                scored=scored,
                athletes=tuple(athletes),
                lines={},
            )
            for index, (athletes, scored) in enumerate(frames)
        ),
        jersey_is_str=True,
        version=None,
    )


def test_label_homography_residuals_are_zero_for_labels_that_follow_a_homography() -> None:
    labels = gsr_labels((athletes_at(SPREAD), True), (athletes_at(SPREAD[::-1]), True))

    residuals = label_homography_residuals_m(labels)

    # Both frames are fitted: 8 athletes each.
    assert residuals.shape == (16,)
    assert residuals.max() < 0.01


def test_label_homography_skips_frames_that_are_not_scored_or_have_too_few_athletes() -> None:
    labels = gsr_labels((athletes_at(SPREAD[:5]), True), (athletes_at(SPREAD), False))

    assert label_homography_residuals_m(labels).shape == (0,)


def test_label_homography_uses_only_athletes_with_a_box_and_a_pitch_position() -> None:
    athletes = athletes_at(SPREAD)
    athletes[0] = athletes[0].model_copy(update={"box": None})
    athletes[1] = athletes[1].model_copy(update={"pitch_x": None, "pitch_y": None})

    assert label_homography_residuals_m(gsr_labels((athletes, True))).shape == (6,)

    # With one more athlete without a box, only 5 athletes are left and the frame is skipped.
    athletes[2] = athletes[2].model_copy(update={"box": None})

    assert label_homography_residuals_m(gsr_labels((athletes, True))).shape == (0,)


def test_label_homography_residual_shows_a_label_that_is_off() -> None:
    athletes = athletes_at(SPREAD)
    wrong = athletes[2]
    assert wrong.pitch_x is not None
    athletes[2] = wrong.model_copy(update={"pitch_x": wrong.pitch_x + 5.0})  # the label is 5 m off, about 40 pixels

    residuals = label_homography_residuals_m(gsr_labels((athletes, True)))

    # The fit follows the seven good labels: no homography through the wrong label fits more than
    # four of the eight points within 5 pixels. The residuals are in the order of the athletes.
    assert residuals.shape == (8,)
    assert residuals[2] == pytest.approx(5.0, abs=0.05)
    assert np.delete(residuals, 2).max() < 0.01
