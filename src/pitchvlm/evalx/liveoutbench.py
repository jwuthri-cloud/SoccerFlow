"""Bench of the live output (``liveout``) and of the radar viewer (``radar``) on synthetic clips.

Three parts, each a sub-command:

- ``revisions``: does a frame line differ from its frame in the result, and how often do the
  flags and the tracks of the result say more than the lines? On the clips of
  ``evalx.streambench`` (kinds and detector conditions), at chosen delays.
- ``timing``: the time per frame of ``stream --synthetic`` (median and 95th percentile of the
  frame samples of ``result.json``) in four arms, as separate processes, back to back, the order
  turned every round: ``base`` (the code of another source tree, given with ``--base``),
  ``none`` (this code, no live output), ``file`` (a live output file), ``slow`` (``--live-out -``
  into a reader of one line per second). The load average of the machine (1 minute) is noted
  before each run.
- ``latency``: three processes: a stream of the synthetic clip played at ``--fps`` frames per
  second with a live output into a pipe (``produce``), the viewer ``radar -`` on that pipe, and a
  client of its event stream in this process. The time from the decision of a frame
  (``decided_unix`` of its line) to the event at the client, wall clock of this machine.

Nothing here says how a real detector, a real calibration, real footage or a browser behave.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pitchvlm.config import PipelineConfig
from pitchvlm.evalx.metrics import t_interval
from pitchvlm.evalx.streambench import CONDITIONS, DETECTOR_ONLY, ArraySource, clip_of
from pitchvlm.fallbacks.fake import fake_calibration_answers, fake_detector_answers
from pitchvlm.liveout import LiveOut, frame_document, is_pipe_like
from pitchvlm.stream import DetectorStream, StreamSettings, default_history_frames, stand_in_inputs

KINDS: tuple[str, ...] = ("plain", "cut", "dissolve", "dissolve-20")
#: A reader of one line per second: one byte per read, so the pipe is not emptied in chunks.
SLOW_READER = (
    "import os, time\n"
    "while True:\n"
    "    byte = os.read(0, 1)\n"
    "    if not byte:\n"
    "        break\n"
    "    if byte == b'\\n':\n"
    "        time.sleep(1.0)\n"
)


# --- revisions -------------------------------------------------------------------------------


@dataclass(frozen=True)
class RevisionCase:
    """What the lines of one stream say against its result."""

    kind: str
    condition: str
    seed: int
    delay: int
    frames: int
    #: Frames whose line differs from the frame of the result (the state, the camera, the delay).
    lines_unequal: int
    #: Frames inside a dissolve of the result whose line has no transition flag (known after the decision).
    transition_late: int
    #: Frames inside a final dissolve span.
    transition_frames: int
    #: Player states of the lines, and those whose role, team key or side differ from their track in the result.
    player_states: int
    track_differs: int


class _Lines:
    """A live output kept in memory, frame by frame (the file policy, without the file)."""

    def __init__(self) -> None:
        self.lines: list[dict] = []

    def given(self, state, *, camera, last, transition, cut) -> None:  # noqa: ANN001 (the hook of the stream)
        self.lines.append(
            frame_document(
                state, fps=25.0, camera=camera, last=last, transition=transition, cut=cut, decided_unix=0.0, dropped=0
            )
        )


def revision_case(job: tuple[str, str, int, int]) -> RevisionCase:
    """Run one stream with its lines kept, and compare them with its result."""
    kind, condition, seed, delay = job
    scene, images = clip_of(kind, seed)
    info = scene.info
    source = ArraySource(info, images)
    miss_rate, noise_px = CONDITIONS[condition]
    people, balls = fake_detector_answers(scene, miss_rate=miss_rate, box_noise_px=noise_px, seed=seed)
    cameras = fake_calibration_answers(scene)
    config = PipelineConfig(engines=DETECTOR_ONLY)
    settings = StreamSettings(delay_frames=delay, history_frames=default_history_frames(config, info.fps))
    stream = DetectorStream(config, settings, clip_id=info.clip_id, fps=info.fps, size=(info.width, info.height))
    kept = _Lines()
    stream.on_given = kept.given
    for item in stand_in_inputs(people, balls, cameras):
        stream.push(item, source.read(item.frame))
    stream.finish()
    result = stream.result()
    unequal = 0
    for line, state, camera in zip(kept.lines, result.frames, result.homographies, strict=True):
        again = frame_document(
            state,
            fps=25.0,
            camera=camera,
            last=line["decided_at"],
            transition=line["transition"],
            cut=line["cut"],
            decided_unix=0.0,
            dropped=0,
        )
        unequal += int(again != line)
    final = {frame for kind_, (start, stop), _ in stream.transition_log if kind_ == "final" for frame in range(start, stop + 1)}
    late = sum(1 for line in kept.lines if line["frame"] in final and not line["transition"])
    tracks = {track.track_id: track for track in result.tracks}
    states = differs = 0
    for line in kept.lines:
        for player in line["players"]:
            states += 1
            track = tracks.get(player["id"])
            if track is None or (track.role, track.team_key, track.team_side) != (
                player["role"],
                player["team"],
                player["side"],
            ):
                differs += 1
    return RevisionCase(kind, condition, seed, delay, len(kept.lines), unequal, late, len(final), states, differs)


def run_revisions(seeds: Sequence[int], delays: Sequence[int], conditions: Sequence[str], workers: int) -> list[RevisionCase]:
    """Run every case of the revision bench."""
    jobs = [(kind, condition, seed, delay) for kind in KINDS for condition in conditions for seed in seeds for delay in delays]
    if workers <= 1:
        return [revision_case(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(revision_case, jobs))


def revision_lines(cases: Sequence[RevisionCase]) -> list[str]:
    """Return the table of the revision bench: one row per kind, condition and delay, summed over the seeds."""
    rows = ["| kind | condition | delay | clips | frames | lines unequal | dissolve frames | flag known late | "
            "player states | role/team/side of the track differs |", "|---|---|---|---|---|---|---|---|---|---|"]
    groups: dict[tuple[str, str, int], list[RevisionCase]] = {}
    for case in cases:
        groups.setdefault((case.kind, case.condition, case.delay), []).append(case)
    for (kind, condition, delay), group in groups.items():
        states = sum(case.player_states for case in group)
        differs = sum(case.track_differs for case in group)
        rows.append(
            f"| {kind} | {condition} | {delay} | {len(group)} | {sum(case.frames for case in group)} | "
            f"{sum(case.lines_unequal for case in group)} | {sum(case.transition_frames for case in group)} | "
            f"{sum(case.transition_late for case in group)} | {states} | {differs} ({differs / max(states, 1):.2%}) |"
        )
    return rows


# --- timing ----------------------------------------------------------------------------------


def _stream_command(src: Path, out: Path, seed: int, frames: int, extra: Sequence[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pitchvlm.cli",
        "stream",
        "--synthetic",
        "--frames",
        str(frames),
        "--seed",
        str(seed),
        "--delay",
        "25",
        "--out",
        str(out),
        "--allow-dirty",
        *extra,
    ]


def timing_run(arm: str, src: Path, seed: int, frames: int, work: Path) -> dict[str, float]:
    """Run one arm once in its own process and return the median and 95th percentile of the frame total, ms."""
    out = work / f"{arm}-{seed}-{time.monotonic_ns()}"
    env = {**os.environ, "PYTHONPATH": str(src)}
    load = os.getloadavg()[0]
    if arm in ("base", "none"):
        done = subprocess.run(_stream_command(src, out, seed, frames, []), env=env, capture_output=True, check=False)
    elif arm == "file":
        command = _stream_command(src, out, seed, frames, ["--live-out", str(work / f"live-{seed}.jsonl")])
        done = subprocess.run(command, env=env, capture_output=True, check=False)
    elif arm == "slow":
        command = _stream_command(src, out, seed, frames, ["--live-out", "-"])
        writer = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert writer.stdout is not None
        reader = subprocess.Popen([sys.executable, "-c", SLOW_READER], stdin=writer.stdout)
        writer.stdout.close()
        _, err = writer.communicate()
        reader.kill()
        reader.wait()
        done = subprocess.CompletedProcess(command, writer.returncode, b"", err)
    else:
        raise ValueError(f"unknown arm {arm}")
    if done.returncode != 0:
        raise RuntimeError(f"arm {arm} failed: {done.stderr.decode(errors='replace')[-2000:]}")
    document = json.loads((out / "SNGS-999" / "result.json").read_text(encoding="utf-8"))
    total = document["frame_timing"]["total"]
    live = document.get("live_out")
    return {
        "p50_ms": total["p50_ms"],
        "p95_ms": total["p95_ms"],
        "load": load,
        "dropped": float("nan") if live is None else live["dropped"],
        "written": float("nan") if live is None else live["written"],
    }


#: The arms of the timing bench, in the order of a forward round.
TIMING_ARMS: tuple[str, ...] = ("base", "none", "file", "slow")


def run_timing(
    base: Path, src: Path, seeds: Sequence[int], rounds: int, frames: int, arms: Sequence[str] = TIMING_ARMS
) -> list[dict]:
    """Run the arms on every seed, ``rounds`` rounds, the order turned each round."""
    arms = list(arms)
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="liveoutbench-") as folder:
        work = Path(folder)
        for seed in seeds:
            for round_index in range(rounds):
                order = arms if round_index % 2 == 0 else list(reversed(arms))
                for arm in order:
                    tree = base if arm == "base" else src
                    values = timing_run(arm, tree, seed, frames, work)
                    rows.append({"seed": seed, "round": round_index, "arm": arm, **values})
    return rows


def timing_lines(rows: Sequence[dict]) -> list[str]:
    """Return the timing table: per arm the mean of the medians and 95th percentiles, and the paired differences."""
    lines = ["| arm | runs | median ms, mean (range) | p95 ms, mean (range) | load (range) |", "|---|---|---|---|---|"]
    by_arm: dict[str, list[dict]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    for arm, group in by_arm.items():
        p50 = [row["p50_ms"] for row in group]
        p95 = [row["p95_ms"] for row in group]
        load = [row["load"] for row in group]
        lines.append(
            f"| {arm} | {len(group)} | {np.mean(p50):.2f} ({min(p50):.2f} to {max(p50):.2f}) | "
            f"{np.mean(p95):.2f} ({min(p95):.2f} to {max(p95):.2f}) | {min(load):.1f} to {max(load):.1f} |"
        )
    lines += ["", "| difference (paired by seed and round) | median ms, mean [95% t interval] | p95 ms, mean [95% t interval] |",
              "|---|---|---|"]
    keyed = {(row["seed"], row["round"], row["arm"]): row for row in rows}
    for arm, against in (("file", "base"), ("slow", "base"), ("none", "base"), ("file", "none"), ("slow", "none")):
        if arm not in by_arm or against not in by_arm:
            continue
        pairs = [
            (keyed[(seed, rnd, arm)], keyed[(seed, rnd, against)])
            for seed, rnd, name in keyed
            if name == arm and (seed, rnd, against) in keyed
        ]
        cells = []
        for key in ("p50_ms", "p95_ms"):
            diffs = [first[key] - second[key] for first, second in pairs]
            mean, low, high = t_interval(diffs) if len(diffs) > 1 else (diffs[0], math.nan, math.nan)
            cells.append(f"{mean:+.2f} [{low:+.2f}, {high:+.2f}]")
        lines.append(f"| {arm} - {against} (n = {len(pairs)}) | {cells[0]} | {cells[1]} |")
    return lines


# --- latency ---------------------------------------------------------------------------------


def produce(seed: int, frames: int, fps: float) -> int:
    """Stream the synthetic clip at ``fps`` frames per second with a live output to the standard output."""
    scene, images = clip_of("plain", seed, frames)
    info = scene.info
    people, balls = fake_detector_answers(scene, seed=seed)
    cameras = fake_calibration_answers(scene)
    config = PipelineConfig(engines=DETECTOR_ONLY)
    history = default_history_frames(config, info.fps)
    stream = DetectorStream(
        config,
        StreamSettings(delay_frames=25, history_frames=history),
        clip_id=info.clip_id,
        fps=info.fps,
        size=(info.width, info.height),
    )
    fd = os.dup(1)
    live = LiveOut(fd, info, target="standard output", pipe=is_pipe_like(fd), close_fd=True)
    live.start(delay_frames=25, history_frames=history)
    stream.on_given = live.given
    start = time.monotonic()
    for item in stand_in_inputs(people, balls, cameras):
        wait = start + item.frame / fps - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        stream.push(item, images[item.frame])
    stream.finish()
    live.finish()
    report = live.close()
    print(json.dumps(asdict(report)), file=sys.stderr)
    return 0


def latency_run(src: Path, seed: int, frames: int, fps: float) -> dict:
    """Run the three processes once; return the latencies (ms) of the frames that reached the client."""
    env = {**os.environ, "PYTHONPATH": str(src)}
    producer = subprocess.Popen(
        [sys.executable, "-m", "pitchvlm.evalx.liveoutbench", "produce", "--seed", str(seed), "--frames", str(frames),
         "--fps", str(fps)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    viewer = subprocess.Popen(
        [sys.executable, "-m", "pitchvlm.cli", "radar", "-"],
        env=env,
        stdin=producer.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert producer.stdout is not None and viewer.stdout is not None
    producer.stdout.close()
    match = re.match(r"radar: (http://\S+/) ", viewer.stdout.readline())
    if match is None:
        raise RuntimeError("the viewer printed no address")
    received: list[tuple[int, float, float]] = []
    load = os.getloadavg()[0]
    with urllib.request.urlopen(match.group(1) + "events", timeout=60) as response:
        kind = None
        for raw in response:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("event: "):
                kind = line.removeprefix("event: ")
            elif line.startswith("data: ") and kind == "frame":
                now = time.time()
                data = json.loads(line.removeprefix("data: "))
                received.append((data["frame"], data["decided_unix"], now))
            elif line.startswith("data: ") and kind == "end":
                break
    _, producer_err = producer.communicate(timeout=60)
    viewer.communicate(timeout=60)
    report = json.loads(producer_err.decode().strip().splitlines()[-1])
    return {
        "seed": seed,
        "load": load,
        "latency_ms": [(now - decided) * 1000.0 for _, decided, now in received],
        "received": len(received),
        "given": report["given"],
        "dropped": report["dropped"],
    }


def latency_lines(runs: Sequence[dict]) -> list[str]:
    """Return the latency table: per run and over all runs, the median, 95th percentile and largest."""
    lines = ["| run | load | frames received / decided | dropped by the writer | median ms | p95 ms | largest ms |",
             "|---|---|---|---|---|---|---|"]
    every: list[float] = []
    for run in runs:
        values = np.array(run["latency_ms"])
        every.extend(run["latency_ms"])
        lines.append(
            f"| seed {run['seed']} | {run['load']:.1f} | {run['received']} / {run['given']} | {run['dropped']} | "
            f"{np.median(values):.2f} | {np.percentile(values, 95):.2f} | {values.max():.2f} |"
        )
    values = np.array(every)
    lines.append(f"| all ({len(every)} frames) | - | - | - | {np.median(values):.2f} | {np.percentile(values, 95):.2f} | "
                 f"{values.max():.2f} |")
    return lines


def _seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        seeds.extend(range(int(first), int(last or first) + 1))
    return seeds


def main(argv: Sequence[str] | None = None) -> int:
    """Run one part of the bench and print its table (module docstring)."""
    parser = argparse.ArgumentParser(prog="python -m pitchvlm.evalx.liveoutbench", description=__doc__.splitlines()[0])
    parts = parser.add_subparsers(dest="part", required=True)
    revisions = parts.add_parser("revisions")
    revisions.add_argument("--seeds", type=_seeds, required=True)
    revisions.add_argument("--delays", default="5,25")
    revisions.add_argument("--conditions", default="exact,noisy")
    revisions.add_argument("--workers", type=int, default=1)
    timing = parts.add_parser("timing")
    timing.add_argument("--base", type=Path, required=True, help="the src folder of the code to compare against")
    timing.add_argument("--seeds", type=_seeds, required=True)
    timing.add_argument("--rounds", type=int, default=4)
    timing.add_argument("--frames", type=int, default=250)
    timing.add_argument("--arms", default=",".join(TIMING_ARMS), help=f"arms, of {', '.join(TIMING_ARMS)}")
    timing.add_argument("--json", type=Path, help="write the rows of every run here")
    latency = parts.add_parser("latency")
    latency.add_argument("--seeds", type=_seeds, required=True)
    latency.add_argument("--frames", type=int, default=250)
    latency.add_argument("--fps", type=float, default=25.0)
    latency.add_argument("--json", type=Path, help="write the latencies of every run here")
    producer = parts.add_parser("produce")
    producer.add_argument("--seed", type=int, required=True)
    producer.add_argument("--frames", type=int, required=True)
    producer.add_argument("--fps", type=float, required=True)
    args = parser.parse_args(argv)
    src = Path(__file__).resolve().parents[2]
    if args.part == "produce":
        return produce(args.seed, args.frames, args.fps)
    if args.part == "revisions":
        delays = [int(value) for value in args.delays.split(",")]
        cases = run_revisions(args.seeds, delays, args.conditions.split(","), args.workers)
        print("\n".join(revision_lines(cases)))
        return 0
    if args.part == "timing":
        arms = args.arms.split(",")
        unknown = [arm for arm in arms if arm not in TIMING_ARMS]
        if unknown:
            parser.error(f"unknown arms {unknown}")
        rows = run_timing(args.base.resolve(), src, args.seeds, args.rounds, args.frames, arms)
        if args.json is not None:
            args.json.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print("\n".join(timing_lines(rows)))
        return 0
    runs = [latency_run(src, seed, args.frames, args.fps) for seed in args.seeds]
    if args.json is not None:
        args.json.write_text(json.dumps(runs), encoding="utf-8")
    print("\n".join(latency_lines(runs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
