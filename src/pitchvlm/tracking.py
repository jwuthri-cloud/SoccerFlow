"""Tracks from keyframe detections: matching, fragment joining, votes, team side and shirt numbers.

The model looks at one keyframe at a time and gives no identity. This module
links the people of the keyframes into tracks, joins broken tracks, and then
decides one role, one team and one shirt number per track. Fragments of one
person who was out of the image for a long time get one id again when they
show the same confident shirt number and nothing says that the number is shown
by two people.
"""

from __future__ import annotations

import bisect
import itertools
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from scipy.optimize import linear_sum_assignment

from pitchvlm.config import TrackingConfig
from pitchvlm.pitch import HALF_LENGTH, PENALTY_AREA_DEPTH
from pitchvlm.roster import Roster
from pitchvlm.schemas import (
    ROLES,
    TEAM_ROLES,
    FramePeople,
    NumberReading,
    PersonDet,
    Role,
    TeamKey,
    TeamSide,
    Track,
    TrackObs,
)

#: (src_frame, dst_frame, image xy) -> image xy in the dst frame. None means that the points cannot be moved.
MotionWarp = Callable[[int, int, np.ndarray], np.ndarray | None]
#: (frame, image xy) -> pitch metres. None means that the frame has no mapping.
PitchMap = Callable[[int, np.ndarray], np.ndarray | None]

#: Team keys of a team, in the order that decides a tied vote.
_SIDE_KEYS: tuple[TeamKey, ...] = ("A", "B")
#: Number of crops that must show the read number before the video shows it.
_MIN_AGREEING_CROPS = 2
#: Share of the velocity of a track that moves its foot point on before the matching. The full
#: velocity follows two people who cross best. A smaller share loses fewer tracks when boxes
#: are noisy or people turn, because the velocity comes from two observations only.
_VELOCITY_SHARE = 0.85
#: A track is moved on with its velocity for at most this many times the gap between its last two observations.
_MAX_VELOCITY_GAPS = 2
#: The mean-x rule of the players decides the team sides alone when the two team means are this far apart, metres.
#: The value is an estimate. The label stage reports this gap for every clip (``evalx.oracle.team_side_facts``).
CLEAR_TEAM_GAP_M = 3.0
#: A goalkeeper stands in a goal when the mean pitch x of the track is this far from the halfway line, metres.
GOAL_ZONE_MIN_X_M = HALF_LENGTH - PENALTY_AREA_DEPTH
#: Two tracks that show one shirt number are two people when they are further apart than a person
#: can run, plus this margin, metres. The margin is larger than ``reach_margin_m``: it must hold
#: the mapping error of two positions, and a wrong verdict takes every link of the number. The
#: value is an estimate from synthetic clips with landmark noise.
TWO_PEOPLE_MARGIN_M = 6.0
#: A change of the team key inside a track counts only when each side of it holds at least this
#: many observations with a team key. A shorter run of the other key is a wrong answer of the
#: model, not another person. 4 observations are 1.9 s at the default keyframe interval; for the
#: people of every frame (detector engine) the pipeline asks for the same time, 4 keyframe
#: intervals of frames. The value is an estimate. The weaker runs join their neighbours, so the 4
#: votes need not come in a row: with wrong keys at 10 % and no key at 10 %, one by one, a track of
#: 60 observations shows a change in about 2 % of tracks, a track of 250 in about 9 % (Monte Carlo,
#: 2,000 tracks per length, second review of 2026-11-22). A cut also needs a contact.
SWAP_MIN_VOTES = 4
#: A track whose team key changes took over another person only when a person of the other team
#: was this close to it in the image at the change: foot point distance in box heights (the mean
#: of the two boxes), the unit of the matching cost of ``associate``. The value is an estimate.
SWAP_CONTACT_BOX_HEIGHTS = 1.0
#: Most people of one team side on the pitch at the same time, by role: ten players and one
#: goalkeeper. A clip without a substitution or a red card holds no more people of a side.
TEAM_SLOTS: Mapping[Role, int] = MappingProxyType({"player": 10, "goalkeeper": 1})


# --- Votes -----------------------------------------------------------------------


@dataclass
class _Votes:
    """Vote weights of the role and of the team key of some observations.

    Each observation votes with a weight equal to its box height, because a
    large box is easier to read.
    """

    role: dict[Role, float] = field(default_factory=lambda: dict.fromkeys(ROLES, 0.0))
    team: dict[TeamKey, float] = field(default_factory=lambda: dict.fromkeys(_SIDE_KEYS, 0.0))

    @classmethod
    def of(cls, observations: Iterable[TrackObs]) -> _Votes:
        """Return the votes of some observations."""
        votes = cls()
        for observation in observations:
            votes.add(observation)
        return votes

    def add(self, observation: TrackObs) -> None:
        """Count one observation. The team key ``none`` is no vote: the model could not tell the team there."""
        self.role[observation.role] += observation.box.height
        if observation.team_key in self.team:
            self.team[observation.team_key] += observation.box.height

    def result(self) -> tuple[Role, TeamKey]:
        """Return the voted role and team key (see ``vote_role_team``)."""
        role = max(ROLES, key=self.role.__getitem__)
        if role not in TEAM_ROLES or not any(self.team.values()):
            return role, "none"
        return role, max(_SIDE_KEYS, key=self.team.__getitem__)


def vote_role_team(track: Track) -> tuple[Role, TeamKey]:
    """Vote the role and the team key of a track.

    Each observation votes with a weight equal to its box height, because a
    large box is easier to read. Role and team key are voted separately. A tied
    vote goes to the first label in the order of ``ROLES`` (role) or ``A``,
    ``B`` (team key).

    The team key is voted among ``A`` and ``B`` only. An observation with the
    key ``none`` gives no vote, because ``none`` on a player or a goalkeeper
    means only that the model could not tell the team in that frame. The
    result is ``none`` when no observation has a team, and for a referee or an
    "other" person, who has no team.
    """
    return _Votes.of(track.observations).result()


# --- Matching between keyframes ----------------------------------------------


@dataclass(eq=False)
class _LiveTrack:
    """A track that can still take a new observation."""

    track_id: int
    observations: list[TrackObs]
    #: Pitch position of the newest observation, shape (2,), or None when it is not known.
    pitch: np.ndarray | None
    #: Votes of the observations for the role and the team key.
    votes: _Votes
    #: Motion of the foot point between the last two observations, in pixels per frame of the image of
    #: the newest observation, shape (2,). The camera motion is taken out. Zero for one observation.
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    #: Number of frames between the last two observations. 0 for a track with one observation.
    gap: int = 0
    #: Number of keyframes since the newest observation.
    missed: int = 0

    @property
    def last(self) -> TrackObs:
        """The newest observation."""
        return self.observations[-1]

    def take(self, observation: TrackObs, pitch: np.ndarray | None, still_foot: np.ndarray) -> None:
        """Add a matched observation and its pitch position.

        ``still_foot`` is the foot point of the newest observation so far, moved
        into the frame of the new observation with the camera motion only. The
        way from there to the new foot point is the motion of the person.
        """
        self.gap = observation.frame - self.last.frame
        self.velocity = (np.array(observation.box.foot, dtype=np.float64) - still_foot) / self.gap
        self.observations.append(observation)
        self.votes.add(observation)
        self.pitch = pitch
        self.missed = 0


def associate(
    keyframes: Sequence[FramePeople],
    *,
    fps: float,
    warp: MotionWarp,
    to_pitch: PitchMap,
    config: TrackingConfig = TrackingConfig(),
    keyframe_frames: int = 1,
) -> list[Track]:
    """Link the people of the keyframes into tracks.

    The keyframes are processed in frame order. At each keyframe the live
    tracks are matched to the detections with the Hungarian method.

    - The foot point of the newest observation of a track is moved on with
      the velocity of the track, and then into the new frame with ``warp``
      (camera motion). The velocity is the motion of the foot point between
      the last two observations of the track, with the camera motion taken
      out. A share of it is used (0.85), for at most twice the gap between
      these two observations. So two people who cross keep their tracks: a
      track is expected where its person runs to, not where the other person
      comes from.
    - When ``warp`` returns None, or points that are not finite, the track
      cannot be compared with the new frame and gets no detection there. A raw
      image position is never used: the camera may have moved by hundreds of
      pixels. Give a warp that knows a second way between two frames (see
      ``motion.bridged_warp``) to follow people across a break of the motion
      chain.
    - Cost of a pair: the distance between the expected foot point and the
      foot point of the detection, divided by the mean height of the two
      boxes, plus ``mismatch_penalty`` when the role or the team key of the
      detection differ from the vote of the track so far (see
      ``vote_role_team``). The team key ``none``, on the track or on the
      detection, is no difference: the team is not known there. The penalty
      counts once more for every keyframe that the track has missed since
      its newest observation. ``keyframe_frames`` is the number of entries
      of ``keyframes`` that stand for one keyframe interval: 1 for keyframes,
      the frames of one interval when the people of every frame are given
      (the detector engine), so the penalty grows with the same time. A model that changes a label of a person it
      sees in every keyframe costs little. But after a wait, a detection with
      other labels is mostly another person who came to that place, for
      example an opponent who enters the image where a player left it.
    - A pair is forbidden when its cost is above ``max_image_jump``, or when
      both pitch positions are known and their distance is above
      ``max_speed_mps * dt + reach_margin_m``. ``dt`` is the time since the
      newest observation of the track.
    - The pairs with the lowest total cost are taken. A track and a detection
      that both stay without a match count as one pair with a cost just above
      ``max_image_jump``. So good pairs are not given up only to make one more
      pair.
    - A track without a match stays live for ``max_missed_keyframes``
      keyframes. It ends at the next keyframe without a match.
    - A detection without a match starts a new track.

    Track ids are 0, 1, 2, ... in the order in which the tracks start. The
    result is ordered by track id. The attribute fields of the tracks are not
    set here.
    """
    _check_fps(fps)
    if keyframe_frames < 1:
        raise ValueError(f"keyframe_frames must be at least 1, got {keyframe_frames}")
    ordered = sorted(keyframes, key=lambda people: people.frame)
    if len({people.frame for people in ordered}) != len(ordered):
        raise ValueError("two keyframes have the same frame index")
    ids = itertools.count()
    live: list[_LiveTrack] = []
    ended: list[_LiveTrack] = []
    for people in ordered:
        pitch = _pitch_points(to_pitch, people.frame, _feet(people.people))
        pairs = _match(live, people, pitch, fps=fps, warp=warp, config=config, keyframe_frames=keyframe_frames)
        for track_index, det_index, still_foot in pairs:
            observation = _observation(people.frame, people.people[det_index])
            live[track_index].take(observation, pitch[det_index], still_foot)
        matched_tracks = {track_index for track_index, _, _ in pairs}
        for track_index, track in enumerate(live):
            if track_index not in matched_tracks:
                track.missed += 1
        ended.extend(track for track in live if track.missed > config.max_missed_keyframes)
        live = [track for track in live if track.missed <= config.max_missed_keyframes]
        matched_dets = {det_index for _, det_index, _ in pairs}
        for det_index, person in enumerate(people.people):
            if det_index not in matched_dets:
                observation = _observation(people.frame, person)
                live.append(_LiveTrack(next(ids), [observation], pitch[det_index], _Votes.of([observation])))
    finished = sorted(ended + live, key=lambda track: track.track_id)
    return [Track(track_id=track.track_id, observations=tuple(track.observations)) for track in finished]


def _moved_feet(track: _LiveTrack, frame: int, warp: MotionWarp) -> np.ndarray | None:
    """Return where the foot point of a live track is in ``frame``, shape (2, 2), or None.

    Row 0 is the foot point of the newest observation moved with the camera
    only: there the person is seen when they stand still on the pitch. Row 1 is
    the expected foot point: the foot point is first moved on with a share of
    the velocity of the track. The velocity is not used for a track with one
    observation, and after a wait that is longer than ``_MAX_VELOCITY_GAPS``
    times the gap between the last two observations.

    The answer is None when the track cannot be compared with the frame:
    ``warp`` returns None, or points that are not finite.
    """
    foot = np.array(track.last.box.foot, dtype=np.float64)
    wait = frame - track.last.frame
    expected = foot
    if wait <= _MAX_VELOCITY_GAPS * track.gap:
        expected = foot + _VELOCITY_SHARE * wait * track.velocity
    moved = warp(track.last.frame, frame, np.stack([foot, expected]))
    if moved is None:
        return None
    points = np.asarray(moved, dtype=np.float64).reshape(-1, 2)
    if len(points) != 2:
        raise ValueError(f"the warp returned {len(points)} points for 2 image points")
    return points if np.isfinite(points).all() else None


def _match(
    live: Sequence[_LiveTrack],
    people: FramePeople,
    det_pitch: Sequence[np.ndarray | None],
    *,
    fps: float,
    warp: MotionWarp,
    config: TrackingConfig,
    keyframe_frames: int,
) -> list[tuple[int, int, np.ndarray]]:
    """Match the live tracks to the detections of one keyframe.

    The result is a list of pairs with one more value: (index in ``live``,
    index in ``people.people``, foot point of the track in the frame of the
    detections when its person stood still, shape (2,)).
    """
    if not live or not people.people:
        return []
    moved = [_moved_feet(track, people.frame, warp) for track in live]
    cost = _image_cost(live, moved, people, config.mismatch_penalty, keyframe_frames)
    allowed = (cost <= config.max_image_jump) & ~_out_of_reach(live, det_pitch, people.frame, fps, config)
    # A forbidden pair stands for "no match": it is removed after the solver. Its cost
    # is just above the limit, as in DeepSORT. So every allowed pair is cheaper, but the
    # solver does not give up good pairs only to make one more pair. With a very high
    # cost, a track that lost its person and a new person nearby would shift every
    # match between them by one.
    forbidden_cost = math.nextafter(config.max_image_jump, math.inf)
    rows, cols = linear_sum_assignment(np.where(allowed, cost, forbidden_cost))
    return [(int(row), int(col), moved[row][0]) for row, col in zip(rows, cols) if allowed[row, col]]


def _image_cost(
    live: Sequence[_LiveTrack],
    moved: Sequence[np.ndarray | None],
    people: FramePeople,
    mismatch_penalty: float,
    keyframe_frames: int,
) -> np.ndarray:
    """Return the cost of every (live track, detection) pair, shape (m, n).

    The cost is the distance between the expected foot point of the track and
    the foot point of the detection, in units of box height (the mean of the
    two boxes), plus the penalty when role or team key differ from the vote of
    the track. The penalty is counted one time, and one more time for every
    keyframe interval (``keyframe_frames`` entries) that the track missed. The cost is infinite for a track that
    cannot be compared with the frame.
    """
    detections = people.people
    det_feet = _feet(detections)  # (n, 2)
    det_heights = np.array([person.box.height for person in detections], dtype=np.float64)  # (n,)
    cost = np.full((len(live), len(detections)), np.inf)
    for row, (track, feet) in enumerate(zip(live, moved, strict=True)):
        if feet is None:
            continue
        distance = np.linalg.norm(det_feet - feet[1], axis=1)  # (n,)
        unit = (track.last.box.height + det_heights) / 2.0  # (n,)
        role, team_key = track.votes.result()
        differ = np.array([_labels_differ(role, team_key, person) for person in detections])  # (n,)
        cost[row] = distance / unit + mismatch_penalty * (1 + track.missed // keyframe_frames) * differ
    return cost


def _out_of_reach(
    live: Sequence[_LiveTrack],
    det_pitch: Sequence[np.ndarray | None],
    frame: int,
    fps: float,
    config: TrackingConfig,
) -> np.ndarray:
    """Mark the pairs that are too far apart on the pitch, shape (m, n), bool.

    A pair is marked only when both pitch positions are known.
    """
    blocked = np.zeros((len(live), len(det_pitch)), dtype=bool)
    for row, track in enumerate(live):
        if track.pitch is None:
            continue
        reach = _reach_m(frame - track.last.frame, fps, config.max_speed_mps, config.reach_margin_m)
        for col, position in enumerate(det_pitch):
            if position is not None and _distance(track.pitch, position) > reach:
                blocked[row, col] = True
    return blocked


def _labels_differ(role: Role, team_key: TeamKey, person: PersonDet) -> bool:
    """Tell whether a detection has another role or another team than the vote of a track.

    The team key ``none`` on one of the two sides is no difference: it says
    that the team is not known there, not that it is another team.
    """
    if person.role != role:
        return True
    return "none" not in (team_key, person.team_key) and person.team_key != team_key


def _observation(frame: int, person: PersonDet) -> TrackObs:
    """Turn a detection into a track observation."""
    return TrackObs(frame=frame, box=person.box, role=person.role, team_key=person.team_key, number=person.number)


def _feet(people: Sequence[PersonDet]) -> np.ndarray:
    """Return the foot points of the detections, shape (n, 2)."""
    return np.array([person.box.foot for person in people], dtype=np.float64).reshape(-1, 2)


def _pitch_points(to_pitch: PitchMap, frame: int, image_xy: np.ndarray) -> list[np.ndarray | None]:
    """Return the pitch position of every image point, each of shape (2,).

    An entry is None when the position is not known: the frame has no mapping,
    or the mapped point is not a finite number.
    """
    if len(image_xy) == 0:
        return []
    mapped = to_pitch(frame, image_xy)
    if mapped is None:
        return [None for _ in image_xy]
    points = np.asarray(mapped, dtype=np.float64).reshape(-1, 2)
    if len(points) != len(image_xy):
        raise ValueError(f"the pitch map returned {len(points)} points for {len(image_xy)} image points")
    return [point if np.isfinite(point).all() else None for point in points]


def _reach_m(gap_frames: int, fps: float, speed_mps: float, margin_m: float) -> float:
    """Return the largest distance in metres that a person covers in ``gap_frames`` frames at ``speed_mps``."""
    return speed_mps * gap_frames / fps + margin_m


def _distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return the distance between two points."""
    return float(np.linalg.norm(first - second))


def _check_fps(fps: float) -> None:
    """Raise when the frame rate is not a finite number above zero."""
    if not 0.0 < fps < math.inf:
        raise ValueError(f"fps must be a finite number above 0, got {fps}")


# --- Joining fragments ---------------------------------------------------------


def stitch_fragments(
    tracks: Sequence[Track],
    *,
    fps: float,
    to_pitch: PitchMap,
    config: TrackingConfig = TrackingConfig(),
) -> list[Track]:
    """Drop short tracks, join broken tracks and give new ids.

    Tracks with fewer than ``min_observations`` observations are dropped
    first. They take no part in the joins: a lone detection is often a false
    one, and two false detections of one team must not become a track.

    Two of the other tracks can be joined when all of this is true:

    - the earlier track ends before the later track starts, and the time
      between the two is at most ``max_stitch_gap_s`` (3 seconds by default);
    - their voted role and team key are the same (see ``vote_role_team``);
    - they do not show different shirt numbers: when both tracks have
      per-frame numbers, the most frequent number of both is the same;
    - the pitch positions of the end of the earlier track and of the start of
      the later track are both known, and their distance is at most
      ``stitch_speed_mps * dt + reach_margin_m``.

    The speed of this reach is lower than ``max_speed_mps``: a person who is
    not seen for seconds does not sprint on a straight line all that time, and
    the reach of a sprinter (29 m after 3 s) holds most players of a team.
    The time limit is needed because every reach grows with ``dt``. The frames
    of a joined gap later get an interpolated position, so a wrong join also
    shows a person where nobody was seen.

    A join is made only when it has no rival. For a track, the candidates to
    follow it are looked at in the order of their start. The first one is
    taken only when no other candidate is in the image at the same time as
    it: two candidates that are seen at the same time are two people, and
    nothing tells which of them is the right one. A later candidate that
    starts after the first one ended is no rival, because it can follow the
    first one. The same rule holds backwards for the candidates that a track
    can follow. So a track is joined to at most one earlier and one later
    track, and chains of fragments become one track.

    The result has ids 0..n-1 in the order of the first frame. The attribute
    fields of the result are not set.
    """
    _check_fps(fps)
    ordered = sorted(tracks, key=lambda track: (track.first_frame, track.track_id))
    fragments = [track for track in ordered if len(track.observations) >= config.min_observations]
    successor = _choose_joins(fragments, _join_candidates(fragments, fps, to_pitch, config))
    joined_later = set(successor.values())
    chains = [_chain(fragments, successor, head) for head in range(len(fragments)) if head not in joined_later]
    return [Track(track_id=track_id, observations=chain) for track_id, chain in enumerate(chains)]


def _join_candidates(
    fragments: Sequence[Track], fps: float, to_pitch: PitchMap, config: TrackingConfig
) -> list[tuple[int, int]]:
    """List every pair of fragments that can be joined, as (index of the earlier one, index of the later one)."""
    votes = [vote_role_team(fragment) for fragment in fragments]
    numbers = [_frame_number(fragment) for fragment in fragments]
    starts = [_foot_on_pitch(to_pitch, fragment.observations[0]) for fragment in fragments]
    ends = [_foot_on_pitch(to_pitch, fragment.observations[-1]) for fragment in fragments]
    max_gap_frames = config.max_stitch_gap_s * fps
    candidates: list[tuple[int, int]] = []
    for earlier, end in enumerate(ends):
        for later, start in enumerate(starts):
            gap = fragments[later].first_frame - fragments[earlier].last_frame
            if gap <= 0 or gap > max_gap_frames:
                continue
            if votes[earlier] != votes[later] or end is None or start is None:
                continue
            if None not in (numbers[earlier], numbers[later]) and numbers[earlier] != numbers[later]:
                continue
            if _distance(end, start) <= _reach_m(gap, fps, config.stitch_speed_mps, config.reach_margin_m):
                candidates.append((earlier, later))
    return candidates


def _choose_joins(fragments: Sequence[Track], candidates: Sequence[tuple[int, int]]) -> dict[int, int]:
    """Choose the joins that have no rival.

    The result maps the index of a fragment to the index of the fragment that
    follows it. A pair is joined when the later fragment is the only first
    choice of the earlier one, and the earlier fragment is the only first
    choice of the later one (see ``stitch_fragments``).
    """
    later_of: dict[int, list[int]] = {}
    earlier_of: dict[int, list[int]] = {}
    for earlier, later in candidates:
        later_of.setdefault(earlier, []).append(later)
        earlier_of.setdefault(later, []).append(earlier)
    successor: dict[int, int] = {}
    for earlier, choices in later_of.items():
        later = _first_follower(fragments, choices)
        if later is not None and _last_leader(fragments, earlier_of[later]) == earlier:
            successor[earlier] = later
    return successor


def _first_follower(fragments: Sequence[Track], choices: Sequence[int]) -> int | None:
    """Return the fragment that starts first, or None when another one starts before it ends."""
    by_start = sorted(choices, key=lambda index: fragments[index].first_frame)
    if len(by_start) > 1 and fragments[by_start[1]].first_frame <= fragments[by_start[0]].last_frame:
        return None
    return by_start[0]


def _last_leader(fragments: Sequence[Track], choices: Sequence[int]) -> int | None:
    """Return the fragment that ends last, or None when another one ends after it started."""
    by_end = sorted(choices, key=lambda index: fragments[index].last_frame, reverse=True)
    if len(by_end) > 1 and fragments[by_end[1]].last_frame >= fragments[by_end[0]].first_frame:
        return None
    return by_end[0]


def _chain(fragments: Sequence[Track], successor: Mapping[int, int], head: int) -> tuple[TrackObs, ...]:
    """Return the observations of a fragment and of all fragments joined after it."""
    observations = list(fragments[head].observations)
    index = head
    while index in successor:
        index = successor[index]
        observations.extend(fragments[index].observations)
    return tuple(observations)


def _foot_on_pitch(to_pitch: PitchMap, observation: TrackObs) -> np.ndarray | None:
    """Return the pitch position of the foot point of an observation, or None when it is not known."""
    foot = np.array([observation.box.foot], dtype=np.float64)  # (1, 2)
    return _pitch_points(to_pitch, observation.frame, foot)[0]


def _frame_number(track: Track) -> int | None:
    """Return the most frequent per-frame shirt number of a track, or None when no frame shows one.

    The number that was seen first wins a tie.
    """
    return _most_frequent(observation.number for observation in track.observations)


def _most_frequent(numbers: Iterable[int | None]) -> int | None:
    """Return the most frequent number that is not None (the first one wins a tie), or None."""
    seen = Counter(number for number in numbers if number is not None)
    return seen.most_common(1)[0][0] if seen else None


# --- Team-key changes inside a track ----------------------------------------------


@dataclass(frozen=True)
class TeamRun:
    """A part of a track in which the model names one team key, after outliers are removed.

    ``start`` and ``stop`` are indices into the observations of the track: the
    run holds ``observations[start:stop]``. An observation with the team key
    ``none`` belongs to the run that it lies in.
    """

    team_key: TeamKey
    start: int
    stop: int


def team_key_runs(observations: Sequence[TrackObs], *, min_votes: int = SWAP_MIN_VOTES) -> list[TeamRun]:
    """Split the observations of a track into runs of one team key, without the outliers of the model.

    Only the observations with the team key ``A`` or ``B`` vote. Votes in a
    row with the same key form a run. The strength of a run is the number of
    its votes for its own key. While a run is weaker than ``min_votes``
    (``SWAP_MIN_VOTES`` for keyframes; its time for the people of every frame),
    the weakest run (the first one in a tie) takes the key of its neighbours
    and joins them; its votes then count against the joined run. So every run
    that is left holds at least ``min_votes`` votes for its key, or one
    run holds all votes; that run has the key of most votes (``A`` in a tie).
    Each change between two runs that are left is a change of the team key
    that held on both sides, not some wrong answers.

    A run starts at the first vote of its key; the first run starts at the
    first observation, and the last run ends at the last observation. The
    result is empty when no observation has a team key.
    """
    if min_votes < 1:
        raise ValueError(f"min_votes must be at least 1, got {min_votes}")
    votes = [(index, observation.team_key) for index, observation in enumerate(observations)]
    votes = [(index, key) for index, key in votes if key in _SIDE_KEYS]
    # Each run: its team key, its number of votes, and its number of votes for its own key.
    # Neighbouring runs have the other key.
    keys: list[TeamKey] = []
    sizes: list[int] = []
    strengths: list[int] = []
    for _, key in votes:
        if keys and keys[-1] == key:
            sizes[-1] += 1
            strengths[-1] += 1
        else:
            keys.append(key)
            sizes.append(1)
            strengths.append(1)
    while len(keys) > 1:
        weakest = min(range(len(keys)), key=lambda position: (strengths[position], position))
        if strengths[weakest] >= min_votes:
            break
        low, high = max(weakest - 1, 0), min(weakest + 1, len(keys) - 1)
        key = keys[high] if weakest == low else keys[low]
        joined = [position for position in range(low, high + 1) if keys[position] == key]
        keys[low : high + 1] = [key]
        strengths[low : high + 1] = [sum(strengths[position] for position in joined)]
        sizes[low : high + 1] = [sum(sizes[low : high + 1])]
    if len(keys) == 1:
        # One run holds all votes: its key is the plain majority, A in a tie.
        votes_a = sum(1 for _, key in votes if key == "A")
        keys = ["A" if 2 * votes_a >= len(votes) else "B"]
    runs: list[TeamRun] = []
    first_vote = 0
    for position, (key, size) in enumerate(zip(keys, sizes, strict=True)):
        start = 0 if position == 0 else votes[first_vote][0]
        stop = len(observations) if position == len(keys) - 1 else votes[first_vote + size][0]
        runs.append(TeamRun(team_key=key, start=start, stop=stop))
        first_vote += size
    return runs


def split_team_swaps(tracks: Sequence[Track], *, min_votes: int = SWAP_MIN_VOTES) -> list[Track]:
    """Split a track where it took over a person of the other team.

    When two people of different teams come close, the matching can give the
    track of one of them to the other. Then the team key of the track flips
    and stays flipped. A track of a player or a goalkeeper (voted role) is cut
    between two of its runs (``team_key_runs`` with ``min_votes``) when, at that change, a person
    of the other team was in contact with it. The change spans the
    observations from the last vote of the run before it to the first vote of
    the run after it. In contact means: in the frame of one of these
    observations, an observation of another track lies at most
    ``SWAP_CONTACT_BOX_HEIGHTS`` box heights away (foot points, in the mean
    height of the two boxes), and the run of the other track there has the
    other team key than the run of this track. The runs, not the single
    answers, give the keys, so one wrong answer neither makes nor hides a
    contact. The pieces are new fragments: the joins of ``stitch_fragments``
    link them again, each with the tracks of its own team.

    Limits:

    - Two people of the same team who swap their tracks show no change of the
      team key, so the swap stays.
    - A change that holds for fewer than ``min_votes`` votes on a side
      (a swap shortly before the track ends) is not seen.
    - A model that names the wrong team for a person for some seconds next to
      an opponent gives a cut that is wrong. The pieces then have different
      team keys and are not joined again.

    The result has ids 0..n-1 in the order of the first frame (the order of
    the input in a tie). A track without a cut keeps its observations. The
    attribute fields of the result are not set.
    """
    runs = [team_key_runs(track.observations, min_votes=min_votes) for track in tracks]
    #: Frame -> (track index, observation, team key of its run) of every observation that lies in a run.
    at_frame: dict[int, list[tuple[int, TrackObs, TeamKey]]] = {}
    for index, (track, track_runs) in enumerate(zip(tracks, runs, strict=True)):
        for run in track_runs:
            for observation in track.observations[run.start : run.stop]:
                at_frame.setdefault(observation.frame, []).append((index, observation, run.team_key))
    pieces: list[tuple[TrackObs, ...]] = []
    for index, (track, track_runs) in enumerate(zip(tracks, runs, strict=True)):
        role, _ = vote_role_team(track)
        changes = zip(track_runs, track_runs[1:]) if role in TEAM_ROLES else ()
        cuts = [after.start for before, after in changes if _in_contact(index, track, before, after, at_frame)]
        bounds = [0, *cuts, len(track.observations)]
        pieces.extend(track.observations[start:stop] for start, stop in zip(bounds, bounds[1:]))
    ordered = sorted(pieces, key=lambda observations: observations[0].frame)
    return [Track(track_id=track_id, observations=observations) for track_id, observations in enumerate(ordered)]


def _in_contact(
    index: int,
    track: Track,
    before: TeamRun,
    after: TeamRun,
    at_frame: Mapping[int, Sequence[tuple[int, TrackObs, TeamKey]]],
) -> bool:
    """Tell whether a person of the other team was in contact with a track where its team key changes.

    See ``split_team_swaps``. ``at_frame`` holds the observations of all
    tracks with the key of their run.
    """
    last_vote = max(
        position
        for position in range(before.start, before.stop)
        if track.observations[position].team_key == before.team_key
    )
    for position in range(last_vote, after.start + 1):
        own = track.observations[position]
        own_key = after.team_key if position == after.start else before.team_key
        for other_index, other, other_key in at_frame[own.frame]:
            if other_index == index or other_key == own_key:
                continue
            unit = (own.box.height + other.box.height) / 2.0
            if _distance(np.array(own.box.foot), np.array(other.box.foot)) <= SWAP_CONTACT_BOX_HEIGHTS * unit:
                return True
    return False


# --- Team side, shirt number ------------------------------------------------


def assign_team_sides(tracks: Sequence[Track], mean_pitch_x: Mapping[int, float]) -> dict[TeamKey, TeamSide]:
    """Decide which goal each team key defends.

    ``mean_pitch_x`` maps a track id to the mean pitch x of that track. A
    track without a position is absent. There are two kinds of evidence:

    - The players. Tracks that are voted ``player`` and have the team key
      ``A`` or ``B`` count, each with its number of observations. With both
      team keys, the key with the lower mean pitch x is ``left``. With one
      team key, the key is ``left`` when its mean x is negative.
    - The goalkeepers. A track that is voted ``goalkeeper``, has the team key
      ``A`` or ``B`` and stands in a goal says that its team defends this
      goal. It stands in a goal when its mean pitch x lies in the depth of a
      penalty area (36 m or more from the halfway line). It counts with its
      number of observations times its depth in that zone. So the goalkeeper
      on the goal line outweighs a goalkeeper who came up for a corner kick,
      and a player at the edge of the area whom the model took for a
      goalkeeper.

    The players decide alone when both team keys are seen and their means are
    at least 3 m apart. In a view of one half with a set piece, or with one
    team only, the means say little: attackers can stand as deep as
    defenders. Then the goalkeepers decide, when they name one answer with
    more weight than the other. Without such goalkeepers the players decide
    also in these cases. The result is empty when there is no evidence.
    """
    means = _team_mean_x(tracks, mean_pitch_x)
    if len(means) == 2 and abs(means["A"] - means["B"]) >= CLEAR_TEAM_GAP_M:
        return _sides("A" if means["A"] < means["B"] else "B")
    left_key = _left_key_of_goalkeepers(tracks, mean_pitch_x)
    if left_key is not None:
        return _sides(left_key)
    if not means:
        return {}
    if len(means) == 2:
        return _sides("A" if means["A"] <= means["B"] else "B")
    only_key, mean_x = next(iter(means.items()))
    return _sides(only_key if mean_x < 0.0 else _other_team(only_key))


def _sides(left_key: TeamKey) -> dict[TeamKey, TeamSide]:
    """Return the team sides when ``left_key`` defends the left goal."""
    return {left_key: "left", _other_team(left_key): "right"}


def _known_mean_x(track: Track, mean_pitch_x: Mapping[int, float]) -> float | None:
    """Return the mean pitch x of a track, or None when it has none. A value that is not finite raises."""
    if track.track_id not in mean_pitch_x:
        return None
    x = float(mean_pitch_x[track.track_id])
    if not math.isfinite(x):
        raise ValueError(f"the mean pitch x of track {track.track_id} is not a finite number")
    return x


def _team_mean_x(tracks: Sequence[Track], mean_pitch_x: Mapping[int, float]) -> dict[TeamKey, float]:
    """Return the mean pitch x of the players of each team key, weighted by the number of observations."""
    weighted_sum: dict[TeamKey, float] = {}
    weight: dict[TeamKey, int] = {}
    for track in tracks:
        role, team_key = vote_role_team(track)
        x = _known_mean_x(track, mean_pitch_x)
        if role != "player" or team_key == "none" or x is None:
            continue
        count = len(track.observations)
        weighted_sum[team_key] = weighted_sum.get(team_key, 0.0) + count * x
        weight[team_key] = weight.get(team_key, 0) + count
    return {team_key: weighted_sum[team_key] / weight[team_key] for team_key in weighted_sum}


def goal_side(mean_x: float | None) -> TeamSide | None:
    """Return the side of the goal zone that a mean pitch x lies in, or None when it lies in no goal zone.

    A goal zone is the depth of a penalty area: ``GOAL_ZONE_MIN_X_M`` or more
    from the halfway line. The team-side rule of the labels (``evalx.oracle``)
    uses this function too, so both places have one goal zone.
    """
    if mean_x is None or abs(mean_x) < GOAL_ZONE_MIN_X_M:
        return None
    return "left" if mean_x < 0.0 else "right"


def _left_key_of_goalkeepers(tracks: Sequence[Track], mean_pitch_x: Mapping[int, float]) -> TeamKey | None:
    """Return the team key that the goalkeepers place in the left goal, or None when they do not decide.

    Each goalkeeper track with a team key that stands in a goal votes with its
    number of observations times its depth in the goal zone, metres. A tie is
    no decision.
    """
    weight: dict[TeamKey, float] = dict.fromkeys(_SIDE_KEYS, 0.0)
    for track in tracks:
        role, team_key = vote_role_team(track)
        mean_x = _known_mean_x(track, mean_pitch_x)
        zone = goal_side(mean_x)
        if role != "goalkeeper" or team_key == "none" or mean_x is None or zone is None:
            continue
        left_key = team_key if zone == "left" else _other_team(team_key)
        weight[left_key] += len(track.observations) * (abs(mean_x) - GOAL_ZONE_MIN_X_M)
    if weight["A"] == weight["B"]:
        return None
    return "A" if weight["A"] > weight["B"] else "B"


def _other_team(team_key: TeamKey) -> TeamKey:
    """Return the other one of the team keys ``A`` and ``B``."""
    return "B" if team_key == "A" else "A"


def decide_number(
    reading: NumberReading | None, frame_numbers: Sequence[int | None]
) -> tuple[int | None, int | None]:
    """Decide the shirt number of a track. The result is ``(export, display)``.

    Export is the most likely number, because the scorer counts "no number" on
    a numbered player as an error:

    - the number of the reading, when its confidence is ``high`` or at least 2
      crops show this number;
    - else the most frequent number among the numbers of the crops and the
      per-frame numbers together (the number seen first wins a tie, and the
      crops come first). One crop that the model was not sure about does not
      win against several frames that agree. When the reading names a number
      of two digits, a crop or a frame that shows one of these digits alone
      counts for the number of the reading: a number of two digits is often
      seen in parts (10 as 1 or as 0 from the side), and the model puts the
      parts together;
    - else the number of the reading, also with a low confidence;
    - else None.

    Display is the number shown in the video: the number of the reading, but
    only when its confidence is ``high`` and at least 2 crops show the same
    number. Else it is None.
    """
    return _export_number(reading, frame_numbers), _display_number(reading)


def _agreeing_crops(reading: NumberReading) -> int:
    """Return the number of crops that show the number of the reading."""
    return sum(1 for crop in reading.per_crop if crop == reading.number)


def _export_number(reading: NumberReading | None, frame_numbers: Sequence[int | None]) -> int | None:
    """Return the most likely shirt number of a track (see ``decide_number``)."""
    if reading is None:
        return _most_frequent(frame_numbers)
    sure = reading.confidence == "high" or _agreeing_crops(reading) >= _MIN_AGREEING_CROPS
    if reading.number is not None and sure:
        return reading.number
    parts = _digits_of_two(reading.number)
    seen = [reading.number if number in parts else number for number in (*reading.per_crop, *frame_numbers)]
    voted = _most_frequent(seen)
    return reading.number if voted is None else voted


def _digits_of_two(number: int | None) -> frozenset[int]:
    """Return the two digits of a shirt number from 10 to 99. Any other number has no parts."""
    if number is None or number < 10:
        return frozenset()
    return frozenset(divmod(number, 10))


def _display_number(reading: NumberReading | None) -> int | None:
    """Return the read number when the reading is confident enough for the video, else None."""
    if reading is None or reading.number is None or reading.confidence != "high":
        return None
    return reading.number if _agreeing_crops(reading) >= _MIN_AGREEING_CROPS else None


def finalize_tracks(
    tracks: Sequence[Track],
    *,
    readings: Mapping[int, NumberReading],
    mean_pitch_x: Mapping[int, float],
    fps: float,
    to_pitch: PitchMap,
    config: TrackingConfig = TrackingConfig(),
    roster: Roster | None = None,
    cuts: Sequence[int] = (),
) -> list[Track]:
    """Fill the attribute fields of every track, and give fragments of one person one id.

    - ``role`` and ``team_key`` come from ``vote_role_team``.
    - ``team_side`` comes from ``assign_team_sides``. A goalkeeper who stands
      in a goal (see there) has the side of that goal, whatever the team key
      is: the kit of a goalkeeper matches no team, so the key is often
      ``none``. The side is None for a referee, for an "other" person and
      for a team key without a side.
    - ``number_export`` and ``number_display`` come from ``decide_number``.
      ``readings`` maps a track id to the shirt-number reading of that track.
      A referee and an "other" person get no number.
    - A displayed number names one person per team side. The tracks with the
      same side and displayed number are one group. Two tracks of a group are
      two people when they are further apart than a person can run:
      ``max_speed_mps * dt + TWO_PEOPLE_MARGIN_M``. The distance is taken
      between the first observation of the track that starts later and the
      observation of the other track that is nearest to it in time, with the
      pitch positions of ``to_pitch``; ``dt`` is the time between the two.
      Without both positions this cannot be told. Across a cut (``cuts``)
      a position says nothing about the next shot: two tracks that follow
      each other in different shots are not two people by their positions.
    - Tracks of a group that overlap in time (the first frame of one is not
      after the last frame of the other) are never linked. The track with
      the most observations keeps the number (the lower track id wins a tie),
      and a track that overlaps it is a second track: it loses the displayed
      number but keeps the export number. Mostly it shows the same person (a
      second box, or a piece of a track that was broken).
    - When a group holds two people, one of its readings is wrong, and
      nothing says which tracks belong to which person. This is the case
      when a second track and the track that it overlaps are two people, or
      when two of the other tracks that follow each other are. Then only the
      track with the most observations keeps the number, and no track of the
      group is linked.
    - In a group without two people, two of the kept tracks that follow each
      other become one track when they have the same role and both pitch
      positions are known, or when they lie in different shots: they are one
      person who was out of the image in between. Across a cut the shown
      number alone joins them: identity, not position. The track has the id and the place of the earliest of them,
      the observations of all of them, and a team key voted over all
      observations. Without a known position the way between two tracks
      cannot be tested, so they stay two tracks that show the same number.
    - With ``config.team_slots`` (an arm of gate G4, off by default), the
      tracks of one team side and role are then linked when the clip holds
      more of them than the side has people (``TEAM_SLOTS``). See
      ``link_team_slots``; ``cuts`` are the frames that start a new shot, and
      no slot link crosses one.
    - ``name`` comes from the roster by team side and displayed number.

    The result is ordered like ``tracks``, without the tracks that became part
    of an earlier one. The input tracks do not change.
    """
    _check_fps(fps)
    track_ids = [track.track_id for track in tracks]
    if len(set(track_ids)) != len(track_ids):
        raise ValueError("track ids must be unique")
    sides = assign_team_sides(tracks, mean_pitch_x)
    decided = [
        _decided(track, readings.get(track.track_id), sides, _known_mean_x(track, mean_pitch_x)) for track in tracks
    ]
    if any(later <= earlier for earlier, later in zip(cuts, cuts[1:])):
        raise ValueError("the cuts must be strictly increasing")
    linked = _one_person_per_number(decided, fps, to_pitch, config, cuts)
    if config.team_slots:
        linked = link_team_slots(linked, fps=fps, to_pitch=to_pitch, config=config, cuts=cuts)
    if roster is None:
        return linked
    return [_named(track, roster) for track in linked]


def _decided(
    track: Track, reading: NumberReading | None, sides: Mapping[TeamKey, TeamSide], mean_x: float | None
) -> Track:
    """Return a copy of the track with role, team key, team side and numbers set, and no name."""
    role, team_key = vote_role_team(track)
    team_side: TeamSide | None = None
    export: int | None = None
    display: int | None = None
    if role in TEAM_ROLES:
        zone = goal_side(mean_x) if role == "goalkeeper" else None
        team_side = sides.get(team_key) if zone is None else zone
        export, display = decide_number(reading, [observation.number for observation in track.observations])
    return track.model_copy(
        update={
            "role": role,
            "team_key": team_key,
            "team_side": team_side,
            "number_export": export,
            "number_display": display,
            "name": None,
        }
    )


def _two_people(first: Track, second: Track, fps: float, to_pitch: PitchMap, config: TrackingConfig) -> bool | None:
    """Tell whether two tracks that show one shirt number are two people, by where they are at nearly the same time.

    ``first`` does not start after ``second``. The first observation of
    ``second`` is compared with the observation of ``first`` that is nearest
    to it in time: the last one when ``first`` ended before, else the nearest
    one (the earlier one wins a tie). The answer is True when the two pitch
    positions are further apart than a person can run in the time between
    them, plus ``TWO_PEOPLE_MARGIN_M``. It is None when a position is not
    known.
    """
    start = second.observations[0]
    nearest = min(first.observations, key=lambda observation: (abs(observation.frame - start.frame), observation.frame))
    here = _foot_on_pitch(to_pitch, nearest)
    there = _foot_on_pitch(to_pitch, start)
    if here is None or there is None:
        return None
    gap = abs(start.frame - nearest.frame)
    return _distance(here, there) > _reach_m(gap, fps, config.max_speed_mps, TWO_PEOPLE_MARGIN_M)


def _overlap_in_time(first: Track, second: Track) -> bool:
    """Tell whether the time from the first to the last frame of two tracks has a frame in common."""
    return first.first_frame <= second.last_frame and second.first_frame <= first.last_frame


def _in_time_order(tracks: Sequence[Track]) -> list[Track]:
    """Return tracks in the order of their first frame. The lower track id comes first in a tie."""
    return sorted(tracks, key=lambda track: (track.first_frame, track.track_id))


def _one_person_per_number(
    tracks: Sequence[Track], fps: float, to_pitch: PitchMap, config: TrackingConfig, cuts: Sequence[int]
) -> list[Track]:
    """Make one track of the fragments of one person, and let one person only show a number (see ``finalize_tracks``).

    The tracks with the same team side and displayed number are one group. In
    each group the longest tracks that do not overlap in time are kept, and
    every other track is a second track that loses the number. A group that
    holds two people keeps the number on its longest track only. In every
    other group the kept tracks that follow each other as one person become
    one track.
    """
    groups: dict[tuple[TeamSide, int], list[Track]] = {}
    for track in tracks:
        if track.team_side is not None and track.number_display is not None:
            groups.setdefault((track.team_side, track.number_display), []).append(track)
    changed: dict[int, Track] = {}
    absorbed: set[int] = set()
    for members in groups.values():
        by_length = sorted(members, key=lambda track: (-len(track.observations), track.track_id))
        kept: list[Track] = []
        second_tracks: list[Track] = []
        two_people = False
        for track in by_length:
            overlapped = [other for other in kept if _overlap_in_time(track, other)]
            if not overlapped:
                kept.append(track)
                continue
            second_tracks.append(track)
            for other in overlapped:
                if _two_people(*_in_time_order([other, track]), fps, to_pitch, config):
                    two_people = True
        kept = _in_time_order(kept)
        steps = [
            False
            if _shot_of(earlier.last_frame, cuts) != _shot_of(later.first_frame, cuts)
            else _two_people(earlier, later, fps, to_pitch, config)
            for earlier, later in zip(kept, kept[1:])
        ]
        if two_people or True in steps:
            second_tracks = by_length[1:]
        else:
            chain = [kept[0]]
            for apart, track in zip(steps, kept[1:], strict=True):
                if apart is False and track.role == chain[-1].role:
                    chain.append(track)
                    continue
                _link(chain, changed, absorbed)
                chain = [track]
            _link(chain, changed, absorbed)
        for track in second_tracks:
            changed[track.track_id] = track.model_copy(update={"number_display": None})
    return [changed.get(track.track_id, track) for track in tracks if track.track_id not in absorbed]


def _link(chain: Sequence[Track], changed: dict[int, Track], absorbed: set[int]) -> None:
    """Record the fragments of one person, in time order, as one track with the id of the earliest fragment.

    The linked track is stored in ``changed`` and the ids of the later
    fragments in ``absorbed``. One fragment alone changes nothing.
    """
    if len(chain) < 2:
        return
    observations = tuple(observation for track in chain for observation in track.observations)
    _role, team_key = _Votes.of(observations).result()
    head = chain[0]
    changed[head.track_id] = head.model_copy(update={"observations": observations, "team_key": team_key})
    absorbed.update(track.track_id for track in chain[1:])


def link_team_slots(
    tracks: Sequence[Track],
    *,
    fps: float,
    to_pitch: PitchMap,
    config: TrackingConfig = TrackingConfig(),
    cuts: Sequence[int] = (),
) -> list[Track]:
    """Link tracks of one team side so that a side has no more tracks than people, where the positions allow it.

    The tracks must have their votes, team side and numbers (see
    ``finalize_tracks``). The tracks of a player or a goalkeeper with a team
    side form one group per (team side, role). A side has at most
    ``TEAM_SLOTS`` people of a role on the pitch: 10 players and 1
    goalkeeper. The rule holds over the whole clip, not per frame: the camera
    shows a part of the pitch, so the number of people in view says nothing,
    but the people of a side who appear in a clip are at most these. So a
    group with ``n`` tracks and ``s`` slots holds at least ``n - s`` tracks
    that are a later part of another track of the group.

    Two tracks of a group can be linked when all of this is true: the earlier
    one ends before the later one starts; they do not show different
    displayed numbers, nor different most frequent per-frame numbers; the
    pitch positions of the end of the earlier one and of the start of the
    later one are both known; and their distance is at most
    ``max_speed_mps * dt + reach_margin_m``. The links are taken in the order
    of that distance (then of the time between the two, then of the track
    ids), each track with at most one link to an earlier and one to a later
    track, until the group holds ``s`` tracks or no link is left. A link is
    also refused when the chain that it would make shows two different
    displayed or most frequent per-frame numbers: a
    track without a number between a 7 and a 9 must not join the two. A group with
    ``s`` tracks or fewer does not change. So the rule links no more tracks
    than the slots need: a group in which fewer than ``s`` people appeared
    keeps some of its fragments.

    A chain of linked tracks becomes one track with the id and the place of
    its earliest track, the observations of all of them, a team key voted
    over all observations, and the numbers of its longest track that has a
    displayed number (else of its longest track). The name is not set.

    Limits: the distance is the only evidence, so two players who leave the
    view and come back where the other one left are linked the wrong way. A
    clip with a substitution or a red card can hold more people of a side.
    A track that is no person of its own (a second box of one person, a piece
    of a track with the wrong team, a referee with a team key) counts as one,
    so each such track can force one link between two people.

    Hard cuts: ``cuts`` holds the frames that start a new shot, in increasing
    order. A position before a cut says nothing about a position after it, so
    no link crosses a cut (the earlier track ends in one shot and the later
    one starts in another). Only a shown shirt number joins tracks across a
    cut, and that join is made before, by the same number. The count then
    holds per shot: a group with ``n`` tracks that have an observation in a
    shot and ``s`` slots gets at most ``n - s`` links inside that shot. A
    clip without a cut is one shot, so the count is the one of the whole clip.

    The result is ordered like ``tracks``, without the tracks that became part
    of an earlier one.
    """
    _check_fps(fps)
    if any(later <= earlier for earlier, later in zip(cuts, cuts[1:])):
        raise ValueError("the cuts must be strictly increasing")
    groups: dict[tuple[TeamSide, Role], list[Track]] = {}
    for track in tracks:
        if track.team_side is not None and track.role in TEAM_SLOTS:
            groups.setdefault((track.team_side, track.role), []).append(track)
    changed: dict[int, Track] = {}
    absorbed: set[int] = set()
    for (_side, role), members in groups.items():
        budgets = _slot_budgets(members, TEAM_SLOTS[role], cuts)
        if not budgets:
            continue
        successor = _slot_links(_in_time_order(members), budgets, cuts, fps, to_pitch, config)
        heads = [track for track in _in_time_order(members) if track.track_id not in successor.values()]
        by_id = {track.track_id: track for track in members}
        for head in heads:
            chain = [head]
            while chain[-1].track_id in successor:
                chain.append(by_id[successor[chain[-1].track_id]])
            if len(chain) > 1:
                changed[head.track_id] = _slot_chain(chain)
                absorbed.update(track.track_id for track in chain[1:])
    return [changed.get(track.track_id, track) for track in tracks if track.track_id not in absorbed]


def _shot_of(frame: int, cuts: Sequence[int]) -> int:
    """Return the index of the shot that holds a frame: the number of cuts at or before it."""
    return bisect.bisect_right(cuts, frame)


def _slot_budgets(members: Sequence[Track], slots: int, cuts: Sequence[int]) -> dict[int, int]:
    """Return the number of slot links that each shot needs: its tracks of the group minus the slots, where above 0."""
    counts: dict[int, int] = {}
    for track in members:
        for shot in {_shot_of(observation.frame, cuts) for observation in track.observations}:
            counts[shot] = counts.get(shot, 0) + 1
    return {shot: count - slots for shot, count in counts.items() if count > slots}


def _slot_links(
    members: Sequence[Track],
    budgets: Mapping[int, int],
    cuts: Sequence[int],
    fps: float,
    to_pitch: PitchMap,
    config: TrackingConfig,
) -> dict[int, int]:
    """Choose the links between the tracks of one group, at most ``budgets[shot]`` in a shot (see ``link_team_slots``).

    ``members`` are in time order. A link lies in the shot where the earlier
    track ends, and the later track must start in the same shot. The result
    maps the id of a track to the id of the track that follows it.
    """
    ends = [_foot_on_pitch(to_pitch, track.observations[-1]) for track in members]
    starts = [_foot_on_pitch(to_pitch, track.observations[0]) for track in members]
    numbers = [_frame_number(track) for track in members]
    candidates: list[tuple[float, int, int, int, int]] = []
    for earlier, end in enumerate(ends):
        for later, start in enumerate(starts):
            gap = members[later].first_frame - members[earlier].last_frame
            if gap <= 0 or end is None or start is None:
                continue
            shot = _shot_of(members[earlier].last_frame, cuts)
            if shot != _shot_of(members[later].first_frame, cuts) or shot not in budgets:
                continue
            displays = (members[earlier].number_display, members[later].number_display)
            if None not in displays and displays[0] != displays[1]:
                continue
            if None not in (numbers[earlier], numbers[later]) and numbers[earlier] != numbers[later]:
                continue
            distance = _distance(end, start)
            if distance <= _reach_m(gap, fps, config.max_speed_mps, config.reach_margin_m):
                candidates.append((distance, gap, members[earlier].track_id, members[later].track_id, shot))
    successor: dict[int, int] = {}
    has_earlier: set[int] = set()
    left = dict(budgets)
    # The numbers that each chain shows, by the id of every track in it: a link joins two
    # chains, so a track without a number between two numbers would join two people.
    shown = {
        track.track_id: (_numbers({track.number_display}), _numbers({numbers[index]}))
        for index, track in enumerate(members)
    }
    for _distance_m, _gap, earlier_id, later_id, shot in sorted(candidates):
        if left[shot] <= 0 or earlier_id in successor or later_id in has_earlier:
            continue
        joined = tuple(first | second for first, second in zip(shown[earlier_id], shown[later_id], strict=True))
        if any(len(chain_numbers) > 1 for chain_numbers in joined):
            continue
        successor[earlier_id] = later_id
        has_earlier.add(later_id)
        left[shot] -= 1
        for track_id in _chain_ids(earlier_id, successor, has_earlier):
            shown[track_id] = joined
    return successor


def _numbers(values: set[int | None]) -> frozenset[int]:
    """Return the shirt numbers of a set, without None."""
    return frozenset(value for value in values if value is not None)


def _chain_ids(track_id: int, successor: Mapping[int, int], has_earlier: set[int]) -> list[int]:
    """Return the ids of the chain of slot links that holds a track, from its earliest track on."""
    earlier_of = {later: earlier for earlier, later in successor.items()}
    head = track_id
    while head in has_earlier:
        head = earlier_of[head]
    chain = [head]
    while chain[-1] in successor:
        chain.append(successor[chain[-1]])
    return chain


def _slot_chain(chain: Sequence[Track]) -> Track:
    """Return the tracks of one person, in time order, as one track (see ``link_team_slots``)."""
    observations = tuple(observation for track in chain for observation in track.observations)
    _role, team_key = _Votes.of(observations).result()
    displayed = [track for track in chain if track.number_display is not None]
    source = max(displayed or chain, key=lambda track: (len(track.observations), -track.track_id))
    return chain[0].model_copy(
        update={
            "observations": observations,
            "team_key": team_key,
            "number_export": source.number_export,
            "number_display": source.number_display,
            "name": None,
        }
    )


def _named(track: Track, roster: Roster) -> Track:
    """Return the track with the roster name of its team side and displayed number."""
    if track.team_side is None or track.number_display is None:
        return track
    return track.model_copy(update={"name": roster.name_for(track.team_side, track.number_display)})
