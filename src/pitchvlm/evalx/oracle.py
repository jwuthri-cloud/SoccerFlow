"""Label-only checks of the scoring chain. Nothing here calls a model.

The helpers change ground-truth frame states in a known way (shift, drop
frames, resample at keyframes). The scorer must then give a known score. The
other helpers test two assumptions of the pipeline on the labels alone: the
team-side rule, and that label pitch positions follow a homography of the
box foot points.

The team-side rule here is the rule of ``tracking.assign_team_sides``, applied
to the labelled teams: the players decide when their two team means are clearly
apart, else the goalkeepers who stand in a goal, else the players. The two
limits of the rule and its goal zone (``tracking.goal_side``) come from
``tracking``, so both places use the same numbers.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from pitchvlm.geometry import apply_homography, fit_homography
from pitchvlm.labels import GsrAthlete, GsrLabels
from pitchvlm.schemas import FrameState, PlayerState, Role, TeamSide
from pitchvlm.tracking import CLEAR_TEAM_GAP_M, GOAL_ZONE_MIN_X_M, goal_side

#: Fewest athletes with a box and a pitch position that a frame needs for a label homography.
MIN_ATHLETES_FOR_FIT = 6
#: RANSAC threshold of the label homography, image pixels.
LABEL_RANSAC_PX = 5.0

#: The part of the team-side rule that decides the sides of a clip.
SideEvidence = Literal["players", "goalkeepers"]


def shift_states(frames: Sequence[FrameState], dx_m: float, dy_m: float = 0.0) -> list[FrameState]:
    """Return a copy of the frame states with every position moved by ``(dx_m, dy_m)`` metres.

    A positive ``dx_m`` moves towards the right goal. A positive ``dy_m``
    moves towards the main camera.
    """
    return [
        state.model_copy(
            update={
                "players": tuple(
                    player.model_copy(update={"pitch_x": player.pitch_x + dx_m, "pitch_y": player.pitch_y + dy_m})
                    for player in state.players
                )
            }
        )
        for state in frames
    ]


def keep_frames(frames: Sequence[FrameState], keep: Collection[int]) -> list[FrameState]:
    """Return only the frame states whose frame index is in ``keep``, in their old order."""
    wanted = frozenset(keep)
    return [state for state in frames if state.frame in wanted]


def resample_states(frames: Sequence[FrameState], keyframes: Sequence[int]) -> list[FrameState]:
    """Keep the truth at the keyframes and fill the frames between them by linear interpolation.

    The result has one state for every frame from the first to the last
    keyframe. A keyframe keeps its state as it is. A frame between two
    keyframes holds every track id that is in both keyframes, at the position
    interpolated linearly in pitch space; the other fields come from the
    earlier keyframe, there is no box, and ``observed`` is False. This is the
    best result that the keyframe design can reach.

    ``keyframes`` must be increasing, ``frames`` must hold a state for every
    keyframe, and no frame index may be in ``frames`` twice; otherwise
    ``ValueError`` is raised.
    """
    if any(later <= earlier for earlier, later in zip(keyframes, keyframes[1:])):
        raise ValueError("keyframes must be increasing")
    by_frame = {state.frame: state for state in frames}
    if len(by_frame) != len(frames):
        raise ValueError("frames holds a frame index more than once")
    missing = [frame for frame in keyframes if frame not in by_frame]
    if missing:
        raise ValueError(f"frames holds no state for the keyframes {missing}")
    resampled: list[FrameState] = []
    for start, end in zip(keyframes, keyframes[1:]):
        resampled.append(by_frame[start])
        resampled.extend(_states_between(by_frame[start], by_frame[end]))
    if len(keyframes) > 0:
        resampled.append(by_frame[keyframes[-1]])
    return resampled


def team_side_by_rule(frames: Sequence[FrameState]) -> dict[int, TeamSide]:
    """Return the team side that the team-side rule gives to every player track: track id -> side.

    The rule is the one of ``tracking.assign_team_sides``. The two teams are
    the people labelled ``left`` and the people labelled ``right``.

    - The players. The mean x of a team is the mean of the mean x of its
      player tracks, where a track counts with its number of frames. With
      both teams, the team with the lower mean pitch x gets ``left``. With
      one team only, it gets ``left`` when its mean x is negative.
    - The goalkeepers. A goalkeeper track with a labelled side whose mean
      pitch x lies in the depth of a penalty area says that its team defends
      that goal. It counts with its number of frames times its depth in that
      zone, metres.

    The players decide alone when both teams have players and their means
    are at least 3 m apart. Else the goalkeepers decide, when they name one
    answer with more weight than the other. Else the players decide.

    The result holds the tracks with the role ``player`` and a labelled side.
    It is empty when there is no such track.
    """
    players = _labelled_tracks(frames, "player")
    if not players:
        return {}
    rule = _rule_of_the_clip(players, _labelled_tracks(frames, "goalkeeper")).sides
    return {track_id: rule[track.side] for track_id, track in players.items()}


@dataclass(frozen=True)
class TeamSideFacts:
    """What the labels of one clip say about the team-side rule and about its two limits."""

    #: True when the rule gives every player track its labelled team side.
    agrees: bool
    #: The part of the rule that decided the sides of the clip.
    decided_by: SideEvidence
    #: True when the players alone, without the goalkeepers, give every player track its labelled side.
    players_alone_agree: bool
    #: Distance between the mean pitch x of the two teams of players, metres. None when one team has no player.
    player_mean_gap_m: float | None
    #: Number of goalkeeper tracks with a labelled side.
    goalkeeper_tracks: int
    #: Number of these tracks whose mean pitch x lies in the goal zone of their labelled side.
    goalkeepers_in_own_goal_zone: int
    #: Number of these tracks whose mean pitch x lies in the goal zone of the other side.
    goalkeepers_in_other_goal_zone: int


def team_side_facts(frames: Sequence[FrameState]) -> TeamSideFacts:
    """Describe how the team-side rule decides one clip, and the values that its two limits are compared with.

    The two limits are estimates: the gap of the player means from which the
    players decide alone (3 m), and the depth of the goal zone (a penalty
    area). So the label stage reports, for every clip, the gap between the
    player means and how many goalkeepers stand in a goal zone.

    ``ValueError`` is raised when no player has a labelled side, because the
    rule then has nothing to agree with.
    """
    players = _labelled_tracks(frames, "player")
    if not players:
        raise ValueError("no player has a labelled team side")
    goalkeepers = _labelled_tracks(frames, "goalkeeper")
    rule = _rule_of_the_clip(players, goalkeepers)
    team_x = _team_mean_x(players)
    players_alone = _sides_by_players(team_x)
    # For every goalkeeper track: the goal zone that it stands in (or None) and its labelled side.
    zones = [(goal_side(track.mean_x), track.side) for track in goalkeepers.values()]
    return TeamSideFacts(
        agrees=all(rule.sides[track.side] == track.side for track in players.values()),
        decided_by=rule.decided_by,
        players_alone_agree=all(players_alone[track.side] == track.side for track in players.values()),
        player_mean_gap_m=abs(team_x["left"] - team_x["right"]) if len(team_x) == 2 else None,
        goalkeeper_tracks=len(goalkeepers),
        goalkeepers_in_own_goal_zone=sum(zone == side for zone, side in zones),
        goalkeepers_in_other_goal_zone=sum(zone is not None and zone != side for zone, side in zones),
    )


def label_homography_residuals_m(labels: GsrLabels) -> np.ndarray:
    """Return how far label pitch positions are from a homography of the box foot points, in metres.

    For every scored frame with at least 6 athletes that have a box and a
    pitch position, one homography is fitted from the foot points of the boxes
    to the pitch positions (RANSAC, 5 pixels). The result holds the distance
    between the mapped foot point and the labelled position of every such
    athlete, for all fitted frames together, shape (n,). Frames with fewer
    athletes, and frames for which no homography exists, add no value.
    """
    residuals: list[np.ndarray] = []
    for frame in labels.frames:
        if not frame.scored:
            continue
        feet, pitch = _feet_and_pitch(frame.athletes)
        if len(feet) < MIN_ATHLETES_FOR_FIT:
            continue
        fit = fit_homography(feet, pitch, ransac_px=LABEL_RANSAC_PX)
        if fit is not None:
            residuals.append(np.linalg.norm(apply_homography(fit.H, feet) - pitch, axis=1))
    return np.concatenate(residuals) if residuals else np.zeros(0, dtype=np.float64)


# --- Helpers -----------------------------------------------------------------------


def _states_between(first: FrameState, last: FrameState) -> list[FrameState]:
    """Return the interpolated states of the frames between two keyframe states."""
    ends = {player.track_id: player for player in last.players}
    span = last.frame - first.frame
    states: list[FrameState] = []
    for frame in range(first.frame + 1, last.frame):
        weight = (frame - first.frame) / span
        players = tuple(
            _blend(player, ends[player.track_id], weight) for player in first.players if player.track_id in ends
        )
        states.append(FrameState(frame=frame, players=players))
    return states


def _blend(start: PlayerState, end: PlayerState, weight: float) -> PlayerState:
    """Return ``start`` moved towards ``end`` in pitch space; ``weight`` 0 is ``start`` and 1 is ``end``."""
    return start.model_copy(
        update={
            "pitch_x": start.pitch_x + weight * (end.pitch_x - start.pitch_x),
            "pitch_y": start.pitch_y + weight * (end.pitch_y - start.pitch_y),
            "box": None,
            "observed": False,
        }
    )


@dataclass(frozen=True)
class _LabelledTrack:
    """What the team-side rule needs to know about one labelled track of one role."""

    mean_x: float
    n_frames: int
    #: Labelled team side. When the label changes along the track, the most frequent side.
    side: TeamSide


@dataclass(frozen=True)
class _ClipRule:
    """The answer of the team-side rule for one clip."""

    #: Labelled team side -> the side that the rule gives to this team.
    sides: Mapping[TeamSide, TeamSide]
    decided_by: SideEvidence


#: The answer of the rule when the labelled sides are the true sides, and when they are swapped.
_AS_LABELLED: Mapping[TeamSide, TeamSide] = {"left": "left", "right": "right"}
_SWAPPED: Mapping[TeamSide, TeamSide] = {"left": "right", "right": "left"}


def _labelled_tracks(frames: Sequence[FrameState], role: Role) -> dict[int, _LabelledTrack]:
    """Collect the mean pitch x, the length and the labelled side of every track of one role with a side.

    Only the frames in which the track has this role and a labelled side count.
    """
    xs: dict[int, list[float]] = {}
    sides: dict[int, Counter[TeamSide]] = {}
    for state in frames:
        for player in state.players:
            if player.role == role and player.team_side is not None:
                xs.setdefault(player.track_id, []).append(player.pitch_x)
                sides.setdefault(player.track_id, Counter())[player.team_side] += 1
    return {
        track_id: _LabelledTrack(
            mean_x=sum(values) / len(values), n_frames=len(values), side=sides[track_id].most_common(1)[0][0]
        )
        for track_id, values in xs.items()
    }


def _team_mean_x(players: Mapping[int, _LabelledTrack]) -> dict[TeamSide, float]:
    """Return the mean pitch x of the players of each labelled team. A track counts with its number of frames."""
    weighted_sum: dict[TeamSide, float] = {}
    weight: dict[TeamSide, int] = {}
    for track in players.values():
        weighted_sum[track.side] = weighted_sum.get(track.side, 0.0) + track.n_frames * track.mean_x
        weight[track.side] = weight.get(track.side, 0) + track.n_frames
    return {side: weighted_sum[side] / weight[side] for side in weighted_sum}


def _sides_by_players(team_x: Mapping[TeamSide, float]) -> Mapping[TeamSide, TeamSide]:
    """Apply the mean-x rule of the players to the mean pitch x of the labelled teams."""
    if len(team_x) == 2:
        # On an exact tie the rule cannot tell the teams apart. It then counts as wrong.
        return _AS_LABELLED if team_x["left"] < team_x["right"] else _SWAPPED
    return {side: "left" if mean_x < 0.0 else "right" for side, mean_x in team_x.items()}


def _sides_by_goalkeepers(goalkeepers: Mapping[int, _LabelledTrack]) -> Mapping[TeamSide, TeamSide] | None:
    """Return the answer of the goalkeepers who stand in a goal zone, or None when they do not decide.

    A goalkeeper in the goal zone of the labelled side votes for the labelled
    sides. A goalkeeper in the other goal zone votes for the swapped sides.
    Each one votes with the number of frames times the depth in the zone,
    metres. A tie is no decision.
    """
    weight = {"as labelled": 0.0, "swapped": 0.0}
    for track in goalkeepers.values():
        zone = goal_side(track.mean_x)
        if zone is not None:
            answer = "as labelled" if zone == track.side else "swapped"
            weight[answer] += track.n_frames * (abs(track.mean_x) - GOAL_ZONE_MIN_X_M)
    if weight["as labelled"] == weight["swapped"]:
        return None
    return _AS_LABELLED if weight["as labelled"] > weight["swapped"] else _SWAPPED


def _rule_of_the_clip(
    players: Mapping[int, _LabelledTrack], goalkeepers: Mapping[int, _LabelledTrack]
) -> _ClipRule:
    """Apply the team-side rule to the labelled teams of one clip (see ``team_side_by_rule``)."""
    team_x = _team_mean_x(players)
    if len(team_x) == 2 and abs(team_x["left"] - team_x["right"]) >= CLEAR_TEAM_GAP_M:
        return _ClipRule(sides=_sides_by_players(team_x), decided_by="players")
    by_goalkeepers = _sides_by_goalkeepers(goalkeepers)
    if by_goalkeepers is not None:
        return _ClipRule(sides=by_goalkeepers, decided_by="goalkeepers")
    return _ClipRule(sides=_sides_by_players(team_x), decided_by="players")


def _feet_and_pitch(athletes: Sequence[GsrAthlete]) -> tuple[np.ndarray, np.ndarray]:
    """Return the box foot points (pixels) and the pitch positions (metres) of the athletes that have both.

    Both arrays have the shape (n, 2), and row i of both belongs to the same athlete.
    """
    feet: list[tuple[float, float]] = []
    pitch: list[tuple[float, float]] = []
    for athlete in athletes:
        if athlete.box is not None and athlete.pitch_x is not None and athlete.pitch_y is not None:
            feet.append(athlete.box.foot)
            pitch.append((athlete.pitch_x, athlete.pitch_y))
    return (
        np.array(feet, dtype=np.float64).reshape(-1, 2),
        np.array(pitch, dtype=np.float64).reshape(-1, 2),
    )
