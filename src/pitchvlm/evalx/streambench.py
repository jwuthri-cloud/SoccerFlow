"""Synthetic bench of the streaming path: the offline detector engine against the stream at several delays.

The bench runs the detector engine on synthetic clips (``synth``) with the stand-ins of the
detector and of the calibration (``fallbacks.fake``): the clip path (``pipeline.run_clip``,
the baseline) and the streaming path (``stream``) at the delays of ``ARMS``, on the same
frames, the same detections and the same cameras. Every arm is scored against the true
states of the scene with the tracking values of ``evalx.metrics.identity_scores`` and the
role and team-side accuracy of the matched people.

Clip kinds (``KINDS``), each ``CLIP_FRAMES`` frames long:

- ``plain``: the synthetic match, no cut;
- ``cut``: a hard cut in the middle, to a moment ``JUMP_FRAMES`` frames later;
- ``dissolve``: the same change as a dissolve of ``DISSOLVE_FRAMES`` frames; the stand-ins
  answer with the truth of a hard cut in its middle, as in ``tests/test_video_overlay.py``.

Detector conditions (``CONDITIONS``): ``exact`` boxes, and ``noisy`` (each person missed with
a chance of 0.1, box edges with 2 px of noise), fixed before the first run.

Arms: ``offline`` (the baseline), ``L0``, ``L5``, ``L12``, ``L25`` (the stream with that delay,
the default history and the kit warm-up), ``clip`` (delay and history of the whole clip,
kit warm-up) and ``clip-kits`` (the same with the kit clusters of the clip path: the control
that must equal the baseline).

Nothing here says how a real detector, a real calibration or real footage behaves.

Run ``python -m pitchvlm.evalx.streambench --seeds 200-209`` for the tables of the decision log.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from pitchvlm.config import EngineSettings, PipelineConfig
from pitchvlm.evalx.metrics import _match_positions, identity_scores, t_interval
from pitchvlm.fallbacks.fake import fake_calibration_answers, fake_detector_answers
from pitchvlm.pipeline import ToolAnswers, _people_on_pitch, _pitch_map, clip_kit_clusters, run_clip
from pitchvlm.schemas import TEAM_ROLES, ClipInfo, ClipResult, FramePeople, FrameState
from pitchvlm.stream import StreamSettings, default_history_frames, run_stream, stand_in_inputs
from pitchvlm.synth import RetimedScene, SyntheticScene
from pitchvlm.timing import StageClock

#: Clip kinds and detector conditions, in the order of the tables.
KINDS: tuple[str, ...] = ("plain", "cut", "dissolve")
CONDITIONS: Mapping[str, tuple[float, float]] = {"exact": (0.0, 0.0), "noisy": (0.1, 2.0)}
#: Length of a bench clip: 10 s at 25 frames per second.
CLIP_FRAMES = 250
#: The cut and the dissolve go to a moment this many frames later (10 s): the synthetic camera then looks elsewhere.
#: At 4 s (the jump of the dissolve test of the video path) the window test sees no change at frame 120,
#: and at 20 s the camera pan (period 16 s) is back near the same view.
JUMP_FRAMES = 250
#: Length of the dissolve, frames, and its first frame. The kind ``dissolve-20`` (not in ``KINDS``) has 20 frames.
DISSOLVE_FRAMES = 10
DISSOLVE_START = 120
DISSOLVE_KINDS: Mapping[str, int] = {"dissolve": DISSOLVE_FRAMES, "dissolve-20": 20}
#: The arms of the stream: delay frames (None: the whole clip), whether the kit clusters of the clip are given,
#: and whether the early dissolve test runs (``StreamSettings.early_dissolve_windows``).
STREAM_ARMS: Mapping[str, tuple[int | None, bool, bool]] = {
    "L0": (0, False, True),
    "L5": (5, False, True),
    "L12": (12, False, True),
    "L25": (25, False, True),
    "clip": (None, False, True),
    "clip-kits": (None, True, True),
    "L25-late": (25, False, False),
    "L50": (50, False, True),
}
#: The arms of a default run: those of round 5 (``L25-late`` and ``L50`` are chosen with ``--arms``).
ARMS: tuple[str, ...] = ("offline", "L0", "L5", "L12", "L25", "clip", "clip-kits")
#: Two people are matched when they are at most this far apart, metres (``identity_scores``).
MATCH_M = 5.0
DETECTOR_ONLY = EngineSettings(people="detector", pitch_mapping="calibration", number="none")


@dataclass(frozen=True)
class ArmResult:
    """The values of one arm on one case."""

    kind: str
    condition: str
    seed: int
    arm: str
    coverage: float
    switches: int
    track_ratio: float
    purity: float
    #: Share of the matched people whose role is the true role, and whose team side is the true side.
    role_accuracy: float
    side_accuracy: float
    matched: int
    #: Whether the frames equal the frames of the baseline, and the time of the arm, seconds.
    equal_to_offline: bool
    seconds: float


class ArraySource:
    """A frame source over images in memory, so that every arm reads the same pixels without drawing them again."""

    def __init__(self, info: ClipInfo, images: Sequence[np.ndarray]) -> None:
        self.info = info
        self._images = images

    def read(self, frame: int) -> np.ndarray:
        return self._images[frame].copy()


def clip_of(kind: str, seed: int, n_frames: int = CLIP_FRAMES) -> tuple[SyntheticScene, list[np.ndarray]]:
    """Return the truth scene of a bench clip and its images."""
    middle = n_frames // 2
    if kind == "plain":
        scene = SyntheticScene(n_frames=n_frames, seed=seed)
        return scene, [scene.image(frame) for frame in range(n_frames)]
    if kind == "cut":
        frames = [frame if frame < middle else frame + JUMP_FRAMES for frame in range(n_frames)]
        scene = RetimedScene(frames, seed=seed)
        return scene, [scene.image(frame) for frame in range(n_frames)]
    if kind in DISSOLVE_KINDS:
        length = DISSOLVE_KINDS[kind]
        switch = DISSOLVE_START + length // 2
        truth = RetimedScene([frame if frame < switch else frame + JUMP_FRAMES for frame in range(n_frames)], seed=seed)
        old = RetimedScene(list(range(n_frames)), seed=seed)
        new = RetimedScene([frame + JUMP_FRAMES for frame in range(n_frames)], seed=seed)
        images = []
        for frame in range(n_frames):
            weight = min(1.0, max(0.0, (frame - DISSOLVE_START + 1) / (length + 1)))
            if weight == 0.0:
                images.append(old.image(frame))
            elif weight == 1.0:
                images.append(new.image(frame))
            else:
                blend = (1.0 - weight) * old.image(frame).astype(np.float32) + weight * new.image(frame)
                images.append(np.clip(blend, 0, 255).astype(np.uint8))
        return truth, images
    raise ValueError(f"unknown clip kind '{kind}', expected one of {', '.join(KINDS)}")


def accuracy(pred: Sequence[FrameState], truth: Sequence[FrameState]) -> tuple[float, float, int]:
    """Return the role accuracy, the team-side accuracy and the number of matched people.

    People are matched per frame by pitch position (``MATCH_M``, as ``identity_scores``). The
    side counts for the true players and goalkeepers only.
    """
    roles = sides = side_total = matched = 0
    by_frame = {state.frame: state for state in pred}
    for true_state in truth:
        pred_state = by_frame.get(true_state.frame)
        if pred_state is None:
            continue
        for pred_index, true_index, _ in _match_positions(pred_state.players, true_state.players, MATCH_M):
            guess, true = pred_state.players[pred_index], true_state.players[true_index]
            matched += 1
            roles += guess.role == true.role
            if true.role in TEAM_ROLES:
                side_total += 1
                sides += guess.team_side == true.team_side
    role_share = roles / matched if matched else math.nan
    side_share = sides / side_total if side_total else math.nan
    return role_share, side_share, matched


def run_case(
    kind: str,
    condition: str,
    seed: int,
    *,
    n_frames: int = CLIP_FRAMES,
    kit_warmup_frames: int | None = None,
    arms: Sequence[str] = ARMS,
) -> list[ArmResult]:
    """Run the chosen arms on one case and return their values, in the order of ``arms``."""
    scene, images = clip_of(kind, seed, n_frames)
    info = scene.info
    source = ArraySource(info, images)
    miss_rate, noise_px = CONDITIONS[condition]
    people, balls = fake_detector_answers(scene, miss_rate=miss_rate, box_noise_px=noise_px, seed=seed)
    cameras = fake_calibration_answers(scene)
    config = PipelineConfig(engines=DETECTOR_ONLY)
    truth = [scene.frame_state(frame) for frame in range(info.n_frames)]
    start = time.perf_counter()
    offline = run_clip(source, None, config, tools=ToolAnswers(people=people, balls=balls, cameras=cameras))
    offline_seconds = time.perf_counter() - start
    results: list[ArmResult] = []
    for arm in arms:
        if arm == "offline":
            results.append(_scored(kind, condition, seed, arm, offline, truth, offline, offline_seconds))
            continue
        delay, given, early = STREAM_ARMS[arm]
        history = None if delay is None else default_history_frames(config, info.fps)
        warmup = {} if kit_warmup_frames is None else {"kit_warmup_frames": kit_warmup_frames}
        if not early:
            warmup["early_dissolve_windows"] = None  # type: ignore[assignment]
        settings = StreamSettings(delay_frames=delay, history_frames=history, **warmup)
        kits = _clip_kits(source, people, offline) if given else None
        start = time.perf_counter()
        result = run_stream(
            stand_in_inputs(people, balls, cameras),
            source.read,
            config,
            settings,
            clip_id=info.clip_id,
            fps=info.fps,
            size=(info.width, info.height),
            kit_clusters=kits,
        )
        seconds = time.perf_counter() - start
        results.append(_scored(kind, condition, seed, arm, result, truth, offline, seconds))
    return results


def _clip_kits(source: ArraySource, people: Sequence[FramePeople], offline: ClipResult) -> object:
    """Return the kit clusters that the clip path fitted: on the people on the pitch, without the blend frames."""
    to_pitch = _pitch_map(offline.homographies)
    starts = (0, *offline.cuts)
    ends = (*offline.cuts, offline.info.n_frames)
    blends = {first for first, end in zip(starts, ends, strict=True) if end - first == 1}
    on_pitch = [
        _people_on_pitch(FramePeople(frame=item.frame, people=() if item.frame in blends else item.people), to_pitch, 3.0)
        for item in people
    ]
    clusters, _ = clip_kit_clusters(source, on_pitch)
    return clusters


def _scored(
    kind: str,
    condition: str,
    seed: int,
    arm: str,
    result: ClipResult,
    truth: Sequence[FrameState],
    offline: ClipResult,
    seconds: float,
) -> ArmResult:
    scores = identity_scores(result.frames, truth, max_match_m=MATCH_M)
    role_share, side_share, matched = accuracy(result.frames, truth)
    return ArmResult(
        kind=kind,
        condition=condition,
        seed=seed,
        arm=arm,
        coverage=scores.coverage,
        switches=scores.switches,
        track_ratio=scores.track_ratio,
        purity=scores.purity,
        role_accuracy=role_share,
        side_accuracy=side_share,
        matched=matched,
        equal_to_offline=result.frames == offline.frames,
        seconds=seconds,
    )


def _case(job: tuple[str, str, int, int | None, tuple[str, ...]]) -> list[ArmResult]:
    kind, condition, seed, warmup, arms = job
    return run_case(kind, condition, seed, kit_warmup_frames=warmup, arms=arms)


def run_bench(
    seeds: Sequence[int],
    *,
    kit_warmup_frames: int | None = None,
    workers: int = 1,
    kinds: Sequence[str] = KINDS,
    arms: Sequence[str] = ARMS,
) -> list[ArmResult]:
    """Run the arms on every (kind, condition, seed), in several processes when ``workers`` is above 1."""
    jobs = [
        (kind, condition, seed, kit_warmup_frames, tuple(arms))
        for kind in kinds
        for condition in CONDITIONS
        for seed in seeds
    ]
    if workers <= 1:
        return [result for job in jobs for result in _case(job)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [result for results in pool.map(_case, jobs) for result in results]


def table_lines(results: Sequence[ArmResult]) -> list[str]:
    """Return the Markdown tables: per (kind, condition, arm) the mean and the spread over seeds, then the differences.

    The first table gives ``mean (min to max)`` over the seeds. The second gives, per stream
    arm, the mean per-case difference against ``offline`` with its 95 % Student t interval.
    """
    lines = [
        "| kind | detector | arm | cases | coverage | switches | track ratio | purity | role acc. | side acc. "
        "| equal to offline |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    kinds = [*KINDS, *(kind for kind in DISSOLVE_KINDS if kind not in KINDS)]
    for kind in kinds:
        for condition in CONDITIONS:
            for arm in ("offline", *STREAM_ARMS):
                chosen = [r for r in results if (r.kind, r.condition, r.arm) == (kind, condition, arm)]
                if not chosen:
                    continue
                lines.append(
                    f"| {kind} | {condition} | {arm} | {len(chosen)} | {_spread([r.coverage for r in chosen])} | "
                    f"{_spread([float(r.switches) for r in chosen], digits=1)} | "
                    f"{_spread([r.track_ratio for r in chosen])} | {_spread([r.purity for r in chosen])} | "
                    f"{_spread([r.role_accuracy for r in chosen])} | {_spread([r.side_accuracy for r in chosen])} | "
                    f"{sum(r.equal_to_offline for r in chosen)} of {len(chosen)} |"
                )
    lines += [
        "",
        "| kind | detector | arm | switches vs offline | coverage vs offline | track ratio vs offline | "
        "side acc. vs offline |",
        "|---|---|---|---|---|---|---|",
    ]
    base = {(r.kind, r.condition, r.seed): r for r in results if r.arm == "offline"}
    for kind in kinds:
        for condition in CONDITIONS:
            for arm in STREAM_ARMS:
                chosen = [r for r in results if (r.kind, r.condition, r.arm) == (kind, condition, arm)]
                if not chosen:
                    continue
                pairs = [(r, base[(r.kind, r.condition, r.seed)]) for r in chosen]
                lines.append(
                    f"| {kind} | {condition} | {arm} | "
                    f"{_interval([float(r.switches - b.switches) for r, b in pairs])} | "
                    f"{_interval([r.coverage - b.coverage for r, b in pairs])} | "
                    f"{_interval([r.track_ratio - b.track_ratio for r, b in pairs])} | "
                    f"{_interval([r.side_accuracy - b.side_accuracy for r, b in pairs])} |"
                )
    return lines


def _spread(values: Sequence[float], *, digits: int = 3) -> str:
    known = [value for value in values if not math.isnan(value)]
    if not known:
        return "-"
    mean = math.fsum(known) / len(known)
    return f"{mean:.{digits}f} ({min(known):.{digits}f} to {max(known):.{digits}f})"


def _interval(differences: Sequence[float]) -> str:
    known = [value for value in differences if not math.isnan(value)]
    if len(known) < 2:
        return "-"
    if all(value == known[0] for value in known):
        return f"{known[0]:+.3f} (all cases)"
    mean, low, high = t_interval(known)
    return f"{mean:+.3f} [{low:+.3f}, {high:+.3f}]"


def time_lines(n_frames: int, seed: int, delays: Sequence[int], runs: int) -> list[str]:
    """Time the stream per frame on a plain clip with exact boxes: ``runs`` rounds of all arms, the order turned each round.

    The arms are the clip path (``offline``: it has no frame samples, so its line gives the
    mean milliseconds per frame of each stage, ``StageClock.report``) and the stream at each
    delay. A stream line gives the median, 95th percentile and largest time per frame of each
    stage and of the whole frame (milliseconds). Each line has the load average of the machine
    (1 minute) before the run. The images are drawn before the runs, so no arm draws them.
    """
    scene, images = clip_of("plain", seed, n_frames)
    info = scene.info
    source = ArraySource(info, images)
    people, balls = fake_detector_answers(scene, seed=seed)
    cameras = fake_calibration_answers(scene)
    config = PipelineConfig(engines=DETECTOR_ONLY)
    history = default_history_frames(config, info.fps)
    lines = []
    arms: list[int | None] = [None, *delays]
    for round_index in range(runs):
        order = arms if round_index % 2 == 0 else list(reversed(arms))
        for delay in order:
            load = os.getloadavg()[0]
            clock = StageClock()
            if delay is None:
                tools = ToolAnswers(people=people, balls=balls, cameras=cameras)
                run_clip(source, None, config, tools=tools, clock=clock)
                means = ", ".join(
                    f"{name} {values['ms_per_frame']:.2f}" for name, values in clock.report(info.n_frames).items()
                )
                lines.append(f"offline round {round_index + 1}, load {load:.1f}: mean ms per frame {means}")
                continue
            run_stream(
                stand_in_inputs(people, balls, cameras),
                source.read,
                config,
                StreamSettings(delay_frames=delay, history_frames=history),
                clip_id=info.clip_id,
                fps=info.fps,
                size=(info.width, info.height),
                clock=clock,
            )
            report = clock.frame_report()
            stages = ", ".join(
                f"{name} {values['p50_ms']:.2f}/{values['p95_ms']:.2f}/{values['max_ms']:.2f}"
                for name, values in report.items()
            )
            lines.append(f"L{delay} round {round_index + 1} ({'forward' if round_index % 2 == 0 else 'reverse'}), "
                         f"load {load:.1f}: {stages}")
    return lines


def _seed_range(text: str) -> list[int]:
    first, _, last = text.partition("-")
    return list(range(int(first), int(last or first) + 1))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bench and print the tables of the decision log."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--seeds", type=_seed_range, default=_seed_range("200-209"), help="seeds, e.g. 200-209")
    parser.add_argument("--kit-warmup", type=int, default=None, help="kit warm-up frames (default: the stream's)")
    parser.add_argument("--workers", type=int, default=1, help="processes (default 1)")
    parser.add_argument("--time", action="store_true", help="time the stream per frame instead (3 rounds)")
    parser.add_argument("--kinds", default=",".join(KINDS), help=f"clip kinds, of {', '.join([*KINDS, 'dissolve-20'])}")
    parser.add_argument("--arms", default=",".join(ARMS), help=f"arms, of offline, {', '.join(STREAM_ARMS)}")
    args = parser.parse_args(argv)
    if args.time:
        print("\n".join(time_lines(CLIP_FRAMES, args.seeds[0], (0, 5, 12, 25), 3)))
        return 0
    kinds = args.kinds.split(",")
    arms = args.arms.split(",")
    unknown = [kind for kind in kinds if kind not in (*KINDS, "dissolve-20")] + [
        arm for arm in arms if arm != "offline" and arm not in STREAM_ARMS
    ]
    if unknown or "offline" not in arms:
        parser.error(f"unknown kinds or arms {unknown}; the arms must hold offline")
    results = run_bench(args.seeds, kit_warmup_frames=args.kit_warmup, workers=args.workers, kinds=kinds, arms=arms)
    print("\n".join(table_lines(results)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
