"""Synthetic bench of the track repairs: identity values with and without each repair, on injected faults.

The bench runs the tracking code of the pipeline on the synthetic scene
(``synth.SyntheticScene``) with exact model answers, true camera motion and
true pitch mappings, after it has put one kind of fault into the answers or
into the tracks. Each arm is a ``TrackingConfig``: the current behaviour, the
split of team swaps, the team slots, and both. Every arm sees the same
answers, so the difference between two arms is the effect of the repair.

The faults:

- ``none``: no fault. The scene itself makes people leave the image and come
  back: the camera pans across the pitch.
- ``swap``: the tracks of two people of different teams who are at most one
  box height apart in the image exchange their observations from that
  keyframe on, as the matching does when it confuses two people who meet.
  Up to ``SWAPS_PER_CLIP`` swaps per clip. Two people of one team never come
  that close in the scene, so a swap of teammates (which no repair here can
  see) is not in the bench.
- ``team_noise``: each answer of a player or a goalkeeper names the other team
  with a chance of ``NOISE_FLIP``, and no team with a chance of ``NOISE_NONE``.
- ``occlusion``: ``OCCLUDED_PEOPLE`` players are missing from the answers for
  2.4 to 4.8 seconds each.
- ``mixed``: all of the above in one clip.

In every case the answers carry no per-frame shirt number, and the number
reading is confident for the people of a random half of the scene
(``READ_SHARE``): about the share of gate G5b.

The bench measures what the repairs do on these faults. It cannot say how
often the faults happen in real clips, nor whether the assumptions of the
injection hold there (for example that a swap happens only between people
closer than one box height). Gate G4 on the dev clips decides.

Run ``python -m pitchvlm.evalx.trackbench`` for the table of the decision log.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from pitchvlm.config import TrackingConfig
from pitchvlm.evalx.metrics import IdentityScores, identity_scores, t_interval
from pitchvlm.fill import dense_frame_states
from pitchvlm.frames import keyframe_indices
from pitchvlm.geometry import apply_homography, matrix_to_rows
from pitchvlm.motion import MotionField
from pitchvlm.schemas import TEAM_ROLES, FramePeople, HomographyRecord, NumberReading, Track, TrackObs
from pitchvlm.synth import SyntheticScene
from pitchvlm.tracking import (
    PitchMap,
    associate,
    finalize_tracks,
    split_team_swaps,
    stitch_fragments,
)

#: Fault families, in the order of the table.
FAULTS: tuple[str, ...] = ("none", "swap", "team_noise", "occlusion", "mixed")
#: The arms: the current behaviour and each repair.
ARMS: Mapping[str, TrackingConfig] = MappingProxyType(
    {
        "current": TrackingConfig(),
        "swaps": TrackingConfig(split_team_swaps=True),
        "slots": TrackingConfig(team_slots=True),
        "both": TrackingConfig(split_team_swaps=True, team_slots=True),
    }
)
#: Length of a bench clip: 30 seconds at 25 frames per second, the length of a SoccerNet-GSR clip.
CLIP_FRAMES = 750
#: Time between two keyframes, seconds: the default of the pipeline.
KEYFRAME_INTERVAL_S = 0.48
#: Largest number of injected swaps per clip.
SWAPS_PER_CLIP = 2
#: Two people can be swapped when their foot points are at most this far apart, in box heights.
SWAP_DISTANCE_BOX_HEIGHTS = 1.0
#: A swap is put only where each of the two tracks has this many observations before and after it.
SWAP_MIN_SIDE = 4
#: Chance that an answer names the other team, and that it names no team.
NOISE_FLIP = 0.10
NOISE_NONE = 0.10
#: Number of players who are hidden, and the shortest and longest time hidden, in keyframes.
OCCLUDED_PEOPLE = 4
OCCLUSION_KEYFRAMES = (5, 10)
#: Share of the people of the scene whose shirt number is read with confidence.
READ_SHARE = 0.5


@dataclass(frozen=True)
class CaseResult:
    """The identity values of one arm on one case."""

    fault: str
    seed: int
    arm: str
    scores: IdentityScores
    #: Number of tracks after ``finalize_tracks``.
    tracks: int
    #: Number of injected swaps (0 for the faults without a swap).
    swaps: int


@dataclass(frozen=True)
class _World:
    """What the tracking code gets from a scene, and the truth to score it."""

    scene: SyntheticScene
    to_pitch: PitchMap
    field: MotionField
    records: list[HomographyRecord]
    #: (frame, box) of every answer -> index of the true person.
    person_of: dict[tuple[int, object], int]


def _world(scene: SyntheticScene) -> _World:
    """Return the true camera motion, pitch mappings and person index of every visible box of a scene."""
    matrices = [scene.homography(frame) for frame in range(scene.info.n_frames)]
    steps = [np.linalg.inv(matrices[frame + 1]) @ matrices[frame] for frame in range(scene.info.n_frames - 1)]
    records = [
        HomographyRecord(frame=frame, matrix=matrix_to_rows(matrix), source="landmarks")
        for frame, matrix in enumerate(matrices)
    ]

    def to_pitch(frame: int, xy: np.ndarray) -> np.ndarray:
        return apply_homography(matrices[frame], xy)

    return _World(scene=scene, to_pitch=to_pitch, field=MotionField(steps), records=records, person_of={})


def _answers(world: _World, keyframes: Sequence[int]) -> list[FramePeople]:
    """Return the exact answers of the keyframes without per-frame numbers, and note the person of each box."""
    answers = []
    for frame in keyframes:
        people = world.scene.people(frame).people
        truth = world.scene.frame_state(frame).players
        kept = []
        for person, state in zip(people, truth, strict=True):
            world.person_of[(frame, person.box)] = state.track_id
            kept.append(person.model_copy(update={"number": None}))
        answers.append(FramePeople(frame=frame, people=tuple(kept)))
    return answers


def _with_team_noise(answers: Sequence[FramePeople], rng: np.random.Generator) -> list[FramePeople]:
    """Return answers in which a player or a goalkeeper sometimes has the other team or no team."""
    noisy = []
    for people in answers:
        changed = []
        for person in people.people:
            if person.role in TEAM_ROLES:
                draw = rng.random()
                if draw < NOISE_FLIP:
                    person = person.model_copy(update={"team_key": "B" if person.team_key == "A" else "A"})
                elif draw < NOISE_FLIP + NOISE_NONE:
                    person = person.model_copy(update={"team_key": "none"})
            changed.append(person)
        noisy.append(FramePeople(frame=people.frame, people=tuple(changed)))
    return noisy


def _with_occlusions(
    answers: Sequence[FramePeople], world: _World, rng: np.random.Generator
) -> list[FramePeople]:
    """Return answers in which some players are missing for some seconds each."""
    seen: dict[int, list[int]] = {}
    for position, people in enumerate(answers):
        for person in people.people:
            if person.role == "player":
                seen.setdefault(world.person_of[(people.frame, person.box)], []).append(position)
    hidden: dict[int, set[int]] = {}
    for index in rng.permutation(sorted(seen))[:OCCLUDED_PEOPLE].tolist():
        start = int(rng.choice(seen[index]))
        length = int(rng.integers(OCCLUSION_KEYFRAMES[0], OCCLUSION_KEYFRAMES[1] + 1))
        hidden[index] = set(range(start, start + length))
    return [
        FramePeople(
            frame=people.frame,
            people=tuple(
                person
                for person in people.people
                if position not in hidden.get(world.person_of.get((people.frame, person.box), -1), set())
            ),
        )
        for position, people in enumerate(answers)
    ]


def _with_swaps(tracks: Sequence[Track], rng: np.random.Generator) -> tuple[list[Track], int]:
    """Return tracks in which up to ``SWAPS_PER_CLIP`` pairs of close people of different teams exchange their
    later observations, and the number of swaps."""
    tracks = list(tracks)
    used: set[int] = set()
    swaps = 0
    for _ in range(SWAPS_PER_CLIP):
        candidates = _swap_candidates(tracks, used)
        if not candidates:
            break
        first, second, frame = candidates[int(rng.integers(len(candidates)))]
        tracks[first], tracks[second] = _exchanged(tracks[first], tracks[second], frame)
        used.update((first, second))
        swaps += 1
    return tracks, swaps


def _swap_candidates(tracks: Sequence[Track], used: set[int]) -> list[tuple[int, int, int]]:
    """List (track index, track index, frame) where two close players of different teams can be swapped."""
    at_frame: dict[int, list[tuple[int, TrackObs]]] = {}
    for index, track in enumerate(tracks):
        if index in used:
            continue
        for position, observation in enumerate(track.observations):
            enough = SWAP_MIN_SIDE <= position <= len(track.observations) - SWAP_MIN_SIDE
            if enough and observation.role == "player":
                at_frame.setdefault(observation.frame, []).append((index, observation))
    candidates = []
    for frame in sorted(at_frame):
        for (first, one), (second, other) in _pairs(at_frame[frame]):
            if one.team_key == other.team_key:
                continue
            unit = (one.box.height + other.box.height) / 2.0
            distance = float(np.linalg.norm(np.subtract(one.box.foot, other.box.foot)))
            if distance <= SWAP_DISTANCE_BOX_HEIGHTS * unit:
                candidates.append((first, second, frame))
    return candidates


def _pairs(items: Sequence[tuple[int, TrackObs]]) -> list[tuple[tuple[int, TrackObs], tuple[int, TrackObs]]]:
    """Return every pair of two items, in the order of the list."""
    return [(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))]


def _exchanged(first: Track, second: Track, frame: int) -> tuple[Track, Track]:
    """Return two tracks that exchange their observations from ``frame`` on."""
    first_head = tuple(observation for observation in first.observations if observation.frame < frame)
    first_tail = tuple(observation for observation in first.observations if observation.frame >= frame)
    second_head = tuple(observation for observation in second.observations if observation.frame < frame)
    second_tail = tuple(observation for observation in second.observations if observation.frame >= frame)
    return (
        first.model_copy(update={"observations": first_head + second_tail}),
        second.model_copy(update={"observations": second_head + first_tail}),
    )


def _readings(
    tracks: Sequence[Track], world: _World, numbers: Mapping[int, int | None], readable: set[int]
) -> dict[int, NumberReading]:
    """Return a confident reading of the true number for every track whose main person is readable.

    ``numbers`` maps the index of a person to the true shirt number. The main
    person of a track is the person of most of its observations.
    """
    readings: dict[int, NumberReading] = {}
    for track in tracks:
        people = Counter(world.person_of[(observation.frame, observation.box)] for observation in track.observations)
        main = people.most_common(1)[0][0]
        number = numbers[main]
        if main in readable and number is not None:
            readings[track.track_id] = NumberReading(per_crop=(number,) * 3, number=number, confidence="high")
    return readings


def run_case(fault: str, seed: int, *, n_frames: int = CLIP_FRAMES) -> list[CaseResult]:
    """Run every arm on one case and return its identity values, in the order of ``ARMS``."""
    if fault not in FAULTS:
        raise ValueError(f"unknown fault '{fault}', expected one of {', '.join(FAULTS)}")
    scene = SyntheticScene(n_frames=n_frames, seed=seed)
    info = scene.info
    world = _world(scene)
    rng = np.random.default_rng(seed)
    keyframes = keyframe_indices(info.n_frames, info.fps, KEYFRAME_INTERVAL_S)
    answers = _answers(world, keyframes)
    if fault in ("team_noise", "mixed"):
        answers = _with_team_noise(answers, rng)
    if fault in ("occlusion", "mixed"):
        answers = _with_occlusions(answers, world, rng)
    fragments = associate(answers, fps=info.fps, warp=world.field.warp, to_pitch=world.to_pitch)
    swaps = 0
    if fault in ("swap", "mixed"):
        fragments, swaps = _with_swaps(fragments, rng)
    truth = [scene.frame_state(frame) for frame in range(info.n_frames)]
    numbers = {player.track_id: player.number_export for state in truth for player in state.players}
    people = sorted(numbers)
    readable = set(rng.permutation(people)[: round(READ_SHARE * len(people))].tolist())
    results = []
    for arm, config in ARMS.items():
        pieces = split_team_swaps(fragments) if config.split_team_swaps else fragments
        tracks = stitch_fragments(pieces, fps=info.fps, to_pitch=world.to_pitch, config=config)
        decided = finalize_tracks(
            tracks,
            readings=_readings(tracks, world, numbers, readable),
            mean_pitch_x=_mean_pitch_x(tracks, world.to_pitch),
            fps=info.fps,
            to_pitch=world.to_pitch,
            config=config,
        )
        states = dense_frame_states(
            decided,
            world.records,
            info.n_frames,
            warp=world.field.warp,
            image_size=(info.width, info.height),
            max_fill_gap_frames=round(config.max_stitch_gap_s * info.fps),
        )
        scores = identity_scores(states, truth)
        results.append(CaseResult(fault=fault, seed=seed, arm=arm, scores=scores, tracks=len(decided), swaps=swaps))
    return results


def _mean_pitch_x(tracks: Sequence[Track], to_pitch: PitchMap) -> dict[int, float]:
    """Return the mean pitch x of the foot points of every track."""
    means = {}
    for track in tracks:
        xs = [
            float(to_pitch(observation.frame, np.array([observation.box.foot], dtype=np.float64))[0, 0])
            for observation in track.observations
        ]
        means[track.track_id] = math.fsum(xs) / len(xs)
    return means


def table_lines(results: Sequence[CaseResult]) -> list[str]:
    """Return the two Markdown tables of the bench.

    The first table has one row per (fault, arm): the number of cases, the
    injected swaps, and the means of the identity values and of the number of
    tracks. The second has one row per (fault, repair arm): the mean of the
    per-case differences against the arm ``current`` on the same cases, with
    its 95 % Student t interval, as ``mean [low, high]``.
    """
    lines = [
        "| fault | arm | cases | swaps | coverage | switches | track ratio | purity | tracks |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for fault in FAULTS:
        for arm in ARMS:
            chosen = [result for result in results if result.fault == fault and result.arm == arm]
            if not chosen:
                continue
            lines.append(
                f"| {fault} | {arm} | {len(chosen)} | {sum(result.swaps for result in chosen)} | "
                f"{_mean([result.scores.coverage for result in chosen]):.3f} | "
                f"{_mean([result.scores.switches for result in chosen]):.2f} | "
                f"{_mean([result.scores.track_ratio for result in chosen]):.3f} | "
                f"{_mean([result.scores.purity for result in chosen]):.3f} | "
                f"{_mean([result.tracks for result in chosen]):.1f} |"
            )
    lines += [
        "",
        "| fault | arm | cases | switches vs current | coverage vs current | track ratio vs current |",
        "|---|---|---|---|---|---|",
    ]
    current = {(result.fault, result.seed): result.scores for result in results if result.arm == "current"}
    for fault in FAULTS:
        for arm in ARMS:
            chosen = [result for result in results if result.fault == fault and result.arm == arm]
            if arm == "current" or not chosen:
                continue
            pairs = [(result.scores, current[(result.fault, result.seed)]) for result in chosen]
            switches = _interval([float(arm_scores.switches - base.switches) for arm_scores, base in pairs])
            coverage = _interval([arm_scores.coverage - base.coverage for arm_scores, base in pairs])
            ratio = _interval([arm_scores.track_ratio - base.track_ratio for arm_scores, base in pairs])
            lines.append(f"| {fault} | {arm} | {len(chosen)} | {switches} | {coverage} | {ratio} |")
    return lines


def _interval(differences: Sequence[float]) -> str:
    """Format the mean of some differences and its 95 % t interval as ``mean [low, high]``."""
    mean, low, high = t_interval(differences)
    return f"{mean:+.3f} [{low:+.3f}, {high:+.3f}]"


def _mean(values: Sequence[float]) -> float:
    """Return the mean of some numbers."""
    return math.fsum(values) / len(values)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bench and print the table of the decision log."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10, help="cases per fault: seeds 0 .. N-1 (default 10)")
    args = parser.parse_args(argv)
    results = [result for fault in FAULTS for seed in range(args.seeds) for result in run_case(fault, seed)]
    print("\n".join(table_lines(results)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
