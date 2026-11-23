"""Tracking values that need no labels: for a clip without labels, such as the demo clip.

Each value points at a kind of tracking error, but none of them proves one,
and no gate uses them:

- A track that starts after the clip start (or ends before the clip end) with
  its box away from the image border: a person does not appear out of nothing
  in the middle of the image, so mostly the track of a person who was already
  there broke. A person who stood behind another one does this too.
- A step between two observed positions of a track that is faster than a
  person can run: a jump to another person, or a wrong pitch mapping.
- A change of the team key inside a track that holds on both sides
  (``tracking.team_key_runs``): a swap with a person of the other team, or a
  model that names the wrong team for some seconds.
- The tracks of each team key over time: a team has at most 11 people, so
  many more ids than that are fragments of the same people.

A hard cut (``ClipResult.cuts``) starts a new shot: the camera and the people
are somewhere else after it. So the first and the last observations of each
shot count as the start and the end of the clip for births and ends, and the
step from the last position of a track before a cut to its first one after it
is not a step (only a shown shirt number joins tracks across a cut).
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from pitchvlm.evalx.metrics import summarize
from pitchvlm.schemas import TEAM_ROLES, Box, ClipResult, Track
from pitchvlm.tracking import SWAP_MIN_VOTES, TEAM_SLOTS, team_key_runs

#: A box is at the image border when an edge is closer to it than this many box heights.
BORDER_BOX_HEIGHTS = 1.0
#: The team keys that are counted over time.
_TEAM_KEYS = ("A", "B")


@dataclass(frozen=True)
class TeamCounts:
    """How many tracks of one team key the clip has."""

    #: Tracks of a player or a goalkeeper with this team key.
    ids: int
    #: Largest number of these tracks that are alive at one keyframe (between their first and last observation).
    max_alive: int
    #: Largest number of these tracks that are observed at one keyframe.
    max_seen: int
    #: Keyframes at which more of these tracks are alive than a team has people.
    keyframes_over_slots: int


@dataclass(frozen=True)
class TrackDiagnostics:
    """Tracking values of one clip that need no labels (see the module text)."""

    tracks: int
    #: Tracks that start after the first observation of their shot with the box away from the image border.
    births_inside: int
    #: Tracks that end before the last observation of their shot with the box away from the image border.
    ends_inside: int
    #: Steps between two observed positions of one track inside one shot, and those faster than ``max_speed_mps``.
    steps: int
    impossible_steps: int
    max_speed_mps: float
    #: ``n``, ``median``, ``p90`` and ``mean`` of the top speed of each track with a step, metres per second.
    top_speed_mps: dict[str, float]
    #: Changes of the team key inside the tracks of players and goalkeepers, and the tracks that have one.
    team_changes: int
    tracks_with_team_change: int
    teams: dict[str, TeamCounts]
    #: (frame, alive tracks of team A, alive tracks of team B) at every frame with an observation.
    alive_per_keyframe: tuple[tuple[int, int, int], ...]

    def as_dict(self) -> dict[str, object]:
        """Return the values as a JSON-ready mapping. A value that is not a number (a clip without steps) is None."""
        values = asdict(self)
        values["top_speed_mps"] = {
            name: None if math.isnan(value) else value for name, value in self.top_speed_mps.items()
        }
        return values

    def lines(self) -> list[str]:
        """Return the values as short lines of text for the terminal."""
        top = self.top_speed_mps
        lines = [
            f"tracks: {self.tracks}; started inside the image: {self.births_inside}, "
            f"ended inside the image: {self.ends_inside}",
            f"steps faster than {self.max_speed_mps:g} m/s: {self.impossible_steps} of {self.steps}; "
            f"top speed per track: median {top['median']:.1f}, p90 {top['p90']:.1f} m/s over {top['n']:.0f} tracks",
            f"team key changes inside a track: {self.team_changes} in {self.tracks_with_team_change} tracks",
        ]
        for key, counts in self.teams.items():
            lines.append(
                f"team {key}: {counts.ids} ids, at most {counts.max_alive} alive and {counts.max_seen} seen "
                f"at one keyframe, {counts.keyframes_over_slots} keyframes with more than "
                f"{sum(TEAM_SLOTS.values())} alive"
            )
        return lines


def track_diagnostics(
    result: ClipResult, *, max_speed_mps: float, team_min_votes: int = SWAP_MIN_VOTES
) -> TrackDiagnostics:
    """Compute the tracking values of a clip result that need no labels.

    The positions are those of the frame states at the observations of a track
    (``observed``), so a step is the way between two keyframes, not between two
    filled frames, and no step crosses a cut. ``max_speed_mps`` is the fastest
    plausible speed (the tracking setting of the same name). A team key change
    is counted with ``tracking.team_key_runs`` and ``team_min_votes`` votes, the
    number that the split of team swaps of the same engine uses
    (``SWAP_MIN_VOTES`` times ``pipeline.keyframe_frames``).
    """
    if not 0.0 < max_speed_mps < math.inf:
        raise ValueError(f"max_speed_mps must be a finite number above 0, got {max_speed_mps}")
    info = result.info
    tracks = result.tracks
    cuts = result.cuts
    frames = sorted({observation.frame for track in tracks for observation in track.observations})
    first_of_shot: dict[int, int] = {}
    last_of_shot: dict[int, int] = {}
    for frame in frames:
        first_of_shot.setdefault(_shot_of(frame, cuts), frame)
        last_of_shot[_shot_of(frame, cuts)] = frame
    births = sum(
        1
        for track in tracks
        if track.first_frame > first_of_shot[_shot_of(track.first_frame, cuts)]
        and not _at_border(track.observations[0].box, info.width, info.height)
    )
    ends = sum(
        1
        for track in tracks
        if track.last_frame < last_of_shot[_shot_of(track.last_frame, cuts)]
        and not _at_border(track.observations[-1].box, info.width, info.height)
    )
    speeds = _step_speeds(result)
    all_speeds = np.concatenate(list(speeds.values())) if speeds else np.zeros(0)
    changes = [
        len(team_key_runs(track.observations, min_votes=team_min_votes)) - 1
        for track in tracks
        if track.role in TEAM_ROLES
    ]
    alive = {key: [_alive(tracks, key, frame) for frame in frames] for key in _TEAM_KEYS}
    teams = {key: _team_counts(tracks, key, frames, alive[key]) for key in _TEAM_KEYS}
    return TrackDiagnostics(
        tracks=len(tracks),
        births_inside=births,
        ends_inside=ends,
        steps=int(all_speeds.size),
        impossible_steps=int(np.count_nonzero(all_speeds > max_speed_mps)),
        max_speed_mps=max_speed_mps,
        top_speed_mps=summarize(np.array([values.max() for values in speeds.values()], dtype=np.float64)),
        team_changes=sum(change for change in changes if change > 0),
        tracks_with_team_change=sum(1 for change in changes if change > 0),
        teams=teams,
        alive_per_keyframe=tuple(
            (frame, count_a, count_b) for frame, count_a, count_b in zip(frames, alive["A"], alive["B"], strict=True)
        ),
    )


def _shot_of(frame: int, cuts: Sequence[int]) -> int:
    """Return the index of the shot that holds a frame: the number of cuts at or before it."""
    return bisect.bisect_right(cuts, frame)


def _at_border(box: Box, width: int, height: int) -> bool:
    """Tell whether an edge of a box is within ``BORDER_BOX_HEIGHTS`` box heights of the image border."""
    margin = BORDER_BOX_HEIGHTS * box.height
    return box.x1 < margin or box.y1 < margin or box.x2 > width - margin or box.y2 > height - margin


def _step_speeds(result: ClipResult) -> dict[int, np.ndarray]:
    """Return the speeds in metres per second between the observed positions of each track that has a step.

    Two positions in different shots make no step. ``ValueError`` is raised when a track has two
    observed positions in one frame: the step between them would take no time.
    """
    positions: dict[int, list[tuple[int, float, float]]] = {}
    for state in result.frames:
        for player in state.players:
            if player.observed:
                positions.setdefault(player.track_id, []).append((state.frame, player.pitch_x, player.pitch_y))
    speeds: dict[int, np.ndarray] = {}
    for track_id, points in positions.items():
        ordered = np.array(sorted(points), dtype=np.float64)  # (n, 3): frame, x, y
        if (np.diff(ordered[:, 0]) == 0).any():
            raise ValueError(f"track {track_id} has two observed positions in one frame")
        shots = np.array([_shot_of(int(frame), result.cuts) for frame in ordered[:, 0]])
        same_shot = shots[1:] == shots[:-1]
        if not same_shot.any():
            continue
        seconds = np.diff(ordered[:, 0])[same_shot] / result.info.fps
        speeds[track_id] = np.linalg.norm(np.diff(ordered[:, 1:], axis=0)[same_shot], axis=1) / seconds
    return speeds


def _alive(tracks: Sequence[Track], team_key: str, frame: int) -> int:
    """Count the tracks of players and goalkeepers with a team key whose time span holds a frame."""
    return sum(
        1
        for track in tracks
        if track.role in TEAM_ROLES and track.team_key == team_key and track.first_frame <= frame <= track.last_frame
    )


def _team_counts(tracks: Sequence[Track], team_key: str, frames: Sequence[int], alive: Sequence[int]) -> TeamCounts:
    """Return the counts of one team key (see ``TeamCounts``)."""
    team = [track for track in tracks if track.role in TEAM_ROLES and track.team_key == team_key]
    observed = [{observation.frame for observation in track.observations} for track in team]
    seen = [sum(1 for track_frames in observed if frame in track_frames) for frame in frames]
    slots = sum(TEAM_SLOTS.values())
    return TeamCounts(
        ids=len(team),
        max_alive=max(alive, default=0),
        max_seen=max(seen, default=0),
        keyframes_over_slots=sum(1 for count in alive if count > slots),
    )
