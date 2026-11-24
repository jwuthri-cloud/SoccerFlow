"""Synthetic bench of the live feed: real processes and real pipes, stand-ins of the tools, a paced source.

The chain is that of a live run on the run machine (``docs/RUNBOOK.md``, section 14): the
detector and the calibration each run ``tools/run_tool.py`` in a live run (``--feed`` to a
named pipe, ``--pace`` at 25 frames per second from one shared start), ``pitchvlm merge-feeds``
merges the two pipes by frame, and ``pitchvlm stream --lines -`` reads the merged lines and
decides each frame. Here each tool is ``tools/stand_in_tool.py``: the script of the run
machine with the tool replaced by the answers of the synthetic clip (``fallbacks.fake``), after
an added delay per frame that simulates the time of the tool (``SIMULATED_MS``).

The delays are a simulation, not a measurement of a tool. They are taken from published values
that the decision log copied (``docs/DECISIONS.md``): RF-DETR Small with plain ``predict`` 20.0 ms
per frame on an M4 Max (entry of 2026-11-14, item 4), RF-DETR Medium 4.4 ms on a T4 with
TensorRT FP16 (entry "A real-time detector engine", 2026-11-22), and PnLCalib 439 ms per frame
from its paper (an RTX 2080 Ti with the calibration on a CPU, with the reading of the image, at
960 x 540 pixels, other weights and thresholds: entry of 2026-11-14, item 4).

    python -m pitchvlm.evalx.livebench --runs 3 --delays 5,25

The overload policy of the stream (``stream --max-behind``) is an arm: ``--max-behind S`` runs
every row twice, without the policy (``off``) and with it (``on``), and says whether the result
of ``on`` equals that of ``off`` of the same round. ``--busy N`` starts ``N`` processes that spin
for the whole bench (a load made on purpose) and stops them at the end. ``--seeds A-B`` gives
each round the synthetic clip of its own seed.

    python -m pitchvlm.evalx.livebench --runs 5 --delays 25 --profiles pnlcalib-paper --seeds 2020-2024 \
        --max-behind 0.25 --busy 12 --json rows.jsonl

Nothing here calls a model, and nothing says how a real tool, real footage or the run machine
behave.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pitchvlm.fallbacks.fake import fake_detector_answers
from pitchvlm.synth import SyntheticScene

REPO = Path(__file__).resolve().parents[3]
STAND_IN = REPO / "tools" / "stand_in_tool.py"
SETTINGS = REPO / "configs" / "engines" / "detector_calibration.yaml"
CLIP_ID = "LIVE-001"
FPS = 25.0
#: Simulated time per frame of each tool, milliseconds: published values (module docstring), not measured.
SIMULATED_MS: Mapping[str, Mapping[str, float]] = {
    "fast": {"detector": 4.4, "calibration": 4.4},
    "rfdetr-small-m4": {"detector": 20.0, "calibration": 20.0},
    "pnlcalib-paper": {"detector": 20.0, "calibration": 439.0},
}


@dataclass(frozen=True)
class LiveClip:
    """A synthetic clip on disk for the live chain: its folder, its two clip jobs and the answers of the stand-ins."""

    root: Path
    clip: Path
    jobs: Mapping[str, Path]
    answers: Mapping[str, Path]
    payloads: Mapping[str, list[dict[str, Any]]]
    n_frames: int


def prepare_clip(work: Path, n_frames: int, seed: int) -> LiveClip:
    """Write the frames of a synthetic clip, the clip job of each tool and the answers file of each stand-in."""
    from pitchvlm import cli  # the command line writes the clip jobs, as on the run machine

    scene = SyntheticScene(n_frames=n_frames, seed=seed)
    root = work / "data"
    clip = root / "SoccerNetGS-valid" / CLIP_ID
    (clip / "img1").mkdir(parents=True)
    keys = []
    for frame in range(n_frames):
        path = clip / "img1" / f"{frame + 1:06d}.png"
        if not cv2.imwrite(str(path), scene.image(frame)):
            raise OSError(f"cannot write {path}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        keys.append(hashlib.sha256(np.ascontiguousarray(image[:, :, ::-1]).tobytes()).hexdigest())
    people, balls = fake_detector_answers(scene, seed=seed)
    detector = []
    for item in people:
        rows = [
            {"box_xyxy": [p.box.x1, p.box.y1, p.box.x2, p.box.y2], "score": 0.9, "label": "person"}
            for p in item.people
        ]
        rows += [
            {"box_xyxy": [b.box.x1, b.box.y1, b.box.x2, b.box.y2], "score": b.score, "label": "sports ball"}
            for b in balls.get(item.frame, ())
        ]
        detector.append({"detections": rows})
    calibration = [
        {"homography_pitch_to_image": np.linalg.inv(scene.homography(frame)).tolist(), "pitch_origin": "centre"}
        for frame in range(n_frames)
    ]
    payloads = {"rfdetr": detector, "pnlcalib": calibration}
    jobs: dict[str, Path] = {}
    answers: dict[str, Path] = {}
    for tool, values in payloads.items():
        jobs[tool] = work / f"{tool}-job.json"
        arguments = ["clip-job", "--tool", tool, "--clip", str(clip), "--root", str(root), "--out", str(jobs[tool])]
        if cli.main([*arguments, "--config", str(SETTINGS)]) != 0:
            raise RuntimeError(f"clip-job failed for {tool}")
        answers[tool] = work / f"{tool}-answers.json"
        answers[tool].write_text(json.dumps({"by_image": dict(zip(keys, values, strict=True))}), encoding="utf-8")
    return LiveClip(root=root, clip=clip, jobs=jobs, answers=answers, payloads=payloads, n_frames=n_frames)


def tool_command(
    clip: LiveClip, tool: str, out: Path, feed: Path, *, delay_ms: float, pace: float | None, start_at: float | None,
    skip_behind: bool = False,
) -> list[str]:
    """Return the command of a stand-in tool in a live run."""
    command = [sys.executable, str(STAND_IN), "--answers", str(clip.answers[tool]), "--delay-ms", str(delay_ms), "--"]
    command += [tool, "--job", str(clip.jobs[tool]), "--root", str(clip.root), "--out", str(out), "--device", "cpu"]
    command += ["--feed", str(feed)]
    if pace is not None:
        command += ["--pace", str(pace)]
    if start_at is not None:
        command += ["--start-at", repr(start_at)]
    if skip_behind:
        command += ["--skip-behind"]
    if tool == "pnlcalib":
        command += ["--repo", str(clip.root), "--weights-kp", "kp.pth", "--weights-lines", "lines.pth"]
    return command


def package_environment() -> dict[str, str]:
    """Return the environment of a package process: the source folder on the path, no key of a model API."""
    environment = {key: value for key, value in os.environ.items() if key not in ("GEMINI_API_KEY", "OPENROUTER_API_KEY")}
    environment["PYTHONPATH"] = str(REPO / "src")
    return environment


@dataclass(frozen=True)
class LiveRun:
    """What one run of the live chain gave: the exit codes, the result document of stream and its output."""

    codes: Mapping[str, int]
    document: dict[str, Any] | None
    stream_output: str
    merge_report: str


def run_live(
    clip: LiveClip,
    work: Path,
    *,
    delay: int,
    pace: float | None,
    simulated_ms: Mapping[str, float],
    wait_s: float = 1.0,
    skip_behind_calibration: bool = False,
    start_margin_s: float = 3.0,
    timeout_s: float = 600.0,
    stream_arguments: Sequence[str] = (),
) -> LiveRun:
    """Run the chain once: two stand-in tools into two named pipes, merge-feeds, stream --lines -."""
    work.mkdir(parents=True, exist_ok=True)
    feeds = {tool: work / f"{tool}.fifo" for tool in ("rfdetr", "pnlcalib")}
    for path in feeds.values():
        os.mkfifo(path)
    start_at = None if pace is None else time.time() + start_margin_s
    environment = package_environment()
    tools = {
        tool: subprocess.Popen(
            tool_command(
                clip, tool, work / f"{tool}-feed.jsonl", feeds[tool],
                delay_ms=simulated_ms["detector" if tool == "rfdetr" else "calibration"],
                pace=pace, start_at=start_at, skip_behind=skip_behind_calibration and tool == "pnlcalib",
            ),
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        for tool in feeds
    }
    merge = subprocess.Popen(
        [sys.executable, "-m", "pitchvlm.cli", "merge-feeds", "--detector", str(feeds["rfdetr"]),
         "--calibration", str(feeds["pnlcalib"]), "--wait", str(wait_s)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stream = subprocess.Popen(
        [sys.executable, "-m", "pitchvlm.cli", "stream", "--clip", str(clip.clip), "--lines", "-",
         "--fps", str(FPS), "--out", str(work / "out"), "--delay", str(delay), "--allow-dirty", *stream_arguments],
        env=environment,
        stdin=merge.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert merge.stdout is not None
    merge.stdout.close()  # the stream holds the read end now
    try:
        stream_output, _ = stream.communicate(timeout=timeout_s)
        _, merge_report = merge.communicate(timeout=timeout_s)
        codes = {"stream": stream.returncode, "merge-feeds": merge.returncode}
        for tool, process in tools.items():
            _, errors = process.communicate(timeout=timeout_s)
            codes[tool] = process.returncode
            if process.returncode != 0:
                merge_report += f"\n{tool}: {errors}"
    finally:
        for process in (stream, merge, *tools.values()):
            if process.poll() is None:
                process.kill()
                process.wait()
    result = work / "out" / CLIP_ID / "result.json"
    document = json.loads(result.read_text(encoding="utf-8")) if result.exists() else None
    return LiveRun(codes=codes, document=document, stream_output=stream_output, merge_report=merge_report)


def _row(values: Mapping[str, float] | None) -> str:
    return "-" if values is None else f"{values['p50_ms']:.0f} / {values['p95_ms']:.0f} / {values['max_ms']:.0f}"


#: A process that spins: one busy core while it runs (``--busy``).
SPIN = "while True:\n    pass\n"


def start_busy(count: int) -> list[subprocess.Popen[bytes]]:
    """Start ``count`` processes that spin until they are stopped (``stop_busy``)."""
    return [subprocess.Popen([sys.executable, "-c", SPIN]) for _ in range(count)]


def stop_busy(processes: Sequence[subprocess.Popen[bytes]]) -> None:
    """Stop the processes of ``start_busy`` (only these: their own process ids)."""
    for process in processes:
        if process.poll() is None:
            process.kill()
    for process in processes:
        process.wait()


def _result_without_policy(document: Mapping[str, Any] | None) -> str | None:
    """The result of a run without the name of the overload policy among its engines, as text, or None."""
    if not document or "result" not in document:
        return None
    result = json.loads(json.dumps(document["result"]))
    result.get("engines", {}).pop("overload", None)
    return json.dumps(result, sort_keys=True)


def bench_rows(
    n_frames: int,
    seeds: Sequence[int],
    delays: Sequence[int],
    runs: int,
    profiles: Sequence[str],
    *,
    max_behind: float | None = None,
    busy: int = 0,
) -> Iterator[dict[str, Any]]:
    """Run the chain ``runs`` times per (profile, delay, arm), the order of the arms turned each run; yield a row each.

    Round ``r`` uses the clip of ``seeds[r % len(seeds)]``. The arms are ``off`` alone, or
    ``off`` and ``on`` with ``max_behind`` (``stream --max-behind``).
    """
    policies: list[tuple[str, list[str]]] = [("off", [])]
    if max_behind is not None:
        policies.append(("on", ["--max-behind", repr(max_behind)]))
    spinning = start_busy(busy)
    try:
        with tempfile.TemporaryDirectory(prefix="livebench-") as folder:
            work = Path(folder)
            clips: dict[int, LiveClip] = {}
            arms = [(profile, delay, policy) for profile in profiles for delay in delays for policy in policies]
            for run in range(runs):
                seed = seeds[run % len(seeds)]
                if seed not in clips:
                    clips[seed] = prepare_clip(work / f"clip-{seed}", n_frames, seed)
                order = arms if run % 2 == 0 else list(reversed(arms))
                results: dict[tuple[str, int, str], str | None] = {}
                rows: list[dict[str, Any]] = []
                for profile, delay, (arm, arguments) in order:
                    load = os.getloadavg()[0]
                    simulated = SIMULATED_MS[profile]
                    outcome = run_live(
                        clips[seed], work / f"run-{run}-{profile}-{delay}-{arm}", delay=delay, pace=FPS,
                        simulated_ms=simulated, skip_behind_calibration=simulated["calibration"] > 1000.0 / FPS,
                        stream_arguments=arguments,
                    )
                    document = outcome.document or {}
                    results[(profile, delay, arm)] = _result_without_policy(document)
                    rows.append({
                        "profile": profile, "delay": delay, "arm": arm, "run": run + 1, "seed": seed, "load": load,
                        "busy": busy, "codes": dict(outcome.codes), "document": document,
                    })
                for row in rows:
                    off = results.get((row["profile"], row["delay"], "off"))
                    mine = results.get((row["profile"], row["delay"], row["arm"]))
                    row["result_equals_off"] = None if row["arm"] == "off" else (off is not None and off == mine)
                    yield row
    finally:
        stop_busy(spinning)


def row_line(row: Mapping[str, Any]) -> str:
    """One row of the table of ``bench_lines``."""
    document = row["document"]
    latency = document.get("latency") or {}
    missing = document.get("missing_parts") or {}
    missing_text = "; ".join(
        f"{tool} " + ", ".join(f"{count} {reason}" for reason, count in reasons.items())
        for tool, reasons in missing.items()
    ) or "none"
    total = (document.get("frame_timing") or {}).get("total")
    backlog = (latency.get("backlog_lines") or {}).get("max", "-")
    codes = ", ".join(f"{name} {code}" for name, code in row["codes"].items())
    overload = document.get("overload")
    dropped = "-" if overload is None else f"{overload['dropped']} of {overload['frames']}"
    equal = {None: "-", True: "yes", False: "no"}[row["result_equals_off"]]
    simulated = SIMULATED_MS[row["profile"]]
    return (
        f"| {row['profile']} ({simulated['detector']:g}, {simulated['calibration']:g}) | {row['delay']} | {row['arm']} | "
        f"{row['run']} | {row['seed']} | {row['load']:.1f} | {codes} | {missing_text} | "
        f"{_row(latency.get('source_to_decision'))} | {_row(latency.get('after_last_needed_frame'))} | "
        f"{_row(latency.get('line_to_decision'))} | {_row(total)} | {backlog} | {dropped} | {equal} |"
    )


TABLE_HEAD = [
    "| profile (simulated ms: detector, calibration) | L | arm | run | seed | load | exit codes | missing parts | "
    "source to decision | after frame f + L | line to decision | stream frame total | largest backlog | dropped | "
    "result equals off |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
]


def bench_lines(n_frames: int, seed: int, delays: Sequence[int], runs: int, profiles: Sequence[str]) -> list[str]:
    """Run the chain ``runs`` times per (profile, delay), the order of the arms turned each run; return a table."""
    lines = list(TABLE_HEAD)
    for row in bench_rows(n_frames, [seed], delays, runs, profiles):
        lines.append(row_line(row))
        print(lines[-1], flush=True)
    return lines


def model_waits(n_frames: int, factor: float, max_behind_s: float, fps: float = FPS) -> list[float]:
    """Return the wait of each line in a model of a slow stream: lines every ``1 / fps`` s, a kept frame ``factor / fps``.

    The frame path takes line ``i`` when it is free and the line is in; the policy does not use a
    frame whose line waited more than ``ARRIVAL_SHARE * max_behind_s`` (frame 0 never: the rule of
    ``stream.OverloadPolicy``), and such a frame takes no time. ``factor`` 1 is a stream that just keeps up. The model does not count the
    time that a dropped decision saves (``OverloadPolicy``), so it drops a little more than a
    stream of that speed would.
    """
    from pitchvlm.stream import ARRIVAL_SHARE

    waits: list[float] = []
    now = 0.0
    for frame in range(n_frames):
        read_at = frame / fps
        now = max(now, read_at)
        wait = now - read_at
        waits.append(wait)
        if frame == 0 or wait <= ARRIVAL_SHARE * max_behind_s:
            now += factor / fps
    return waits


def price_case(job: tuple[str, int, float, float, int, int]) -> dict[str, Any]:
    """Run one clip through the feeder with the policy and the waits of ``model_waits``; score it against the truth."""
    from pitchvlm.config import PipelineConfig
    from pitchvlm.evalx.metrics import detection_scores, identity_scores
    from pitchvlm.evalx.streambench import DETECTOR_ONLY, clip_of
    from pitchvlm.fallbacks.fake import fake_calibration_answers
    from pitchvlm.schemas import FramePeople, PersonDet
    from pitchvlm.stream import (
        DetectorStream,
        FrameFeeder,
        OverloadPolicy,
        StreamSettings,
        default_history_frames,
        stand_in_inputs,
    )

    kind, seed, factor, max_behind_s, n_frames, delay = job
    config = PipelineConfig(engines=DETECTOR_ONLY)
    scene, images = clip_of(kind, seed, n_frames=n_frames)
    info = scene.info
    people, balls = fake_detector_answers(scene, seed=seed)
    waits = model_waits(n_frames, factor, max_behind_s)
    # A clock that stands still: both looks of the policy see the wait of the model, whatever the threads do.
    now = 1000.0
    taken = [0]

    def inputs() -> Iterator[Any]:
        for item in stand_in_inputs(people, balls, fake_calibration_answers(scene)):
            taken[0] = item.frame
            yield item

    policy = OverloadPolicy(max_behind_s, clock=lambda: now)
    settings = StreamSettings(delay_frames=delay, history_frames=default_history_frames(config, info.fps))
    stream = DetectorStream(config, settings, clip_id=info.clip_id, fps=info.fps, size=(info.width, info.height))
    feeder = FrameFeeder(
        stream, inputs(), lambda frame: images[frame], policy=policy,
        line_clock=lambda: (now - waits[taken[0]], 0),
    )
    try:
        for fed in feeder:
            stream.push_fed(fed)
    finally:
        feeder.close()
    stream.finish()
    result = stream.result()
    truth = [scene.frame_state(frame) for frame in range(n_frames)]
    identity = identity_scores(result.frames, truth)

    def boxes(states: Sequence[Any]) -> list[FramePeople]:
        return [
            FramePeople(
                frame=state.frame,
                people=tuple(
                    PersonDet(box=player.box, role=player.role) for player in state.players if player.box is not None
                ),
            )
            for state in states
        ]

    detection = detection_scores(boxes(result.frames), boxes(truth))
    return {
        "kind": kind, "seed": seed, "factor": factor, "frames": n_frames, "dropped": len(stream.dropped_frames),
        "not_used": len(stream.overload_report()["not_used"]),
        "dropped_share": len(stream.dropped_frames) / n_frames, "recall": detection.recall,
        "precision": detection.precision, "coverage": identity.coverage, "switches": identity.switches,
        "track_ratio": identity.track_ratio,
    }


def price_rows(
    kinds: Sequence[str], seeds: Sequence[int], factors: Sequence[float], max_behind_s: float, n_frames: int, delay: int
) -> list[dict[str, Any]]:
    """Run ``price_case`` for every kind, seed and factor, in separate processes."""
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(kind, seed, factor, max_behind_s, n_frames, delay) for kind in kinds for seed in seeds for factor in factors]
    with ProcessPoolExecutor(max_workers=max(1, min(len(jobs), (os.cpu_count() or 2) // 2))) as pool:
        return list(pool.map(price_case, jobs))


def price_lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """A table of the price: per kind and factor, the mean over seeds (and the range) of each value."""
    lines = [
        "| kind | factor | seeds | dropped share, mean (range) | box recall | box precision | coverage | id switches | "
        "tracks per true track |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    groups: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["kind"], row["factor"]), []).append(row)
    for (kind, factor), group in groups.items():
        def cell(name: str, digits: int = 3) -> str:
            values = [float(row[name]) for row in group]
            return f"{np.mean(values):.{digits}f} ({min(values):.{digits}f} to {max(values):.{digits}f})"

        lines.append(
            f"| {kind} | {factor:g} | {len(group)} | {cell('dropped_share')} | {cell('recall')} | {cell('precision')} | "
            f"{cell('coverage')} | {cell('switches', 1)} | {cell('track_ratio', 2)} |"
        )
    return lines


def _seed_list(text: str) -> list[int]:
    """Read ``A-B`` or ``A,B,C``."""
    if "-" in text:
        first, last = (int(value) for value in text.split("-", 1))
        return list(range(first, last + 1))
    return [int(value) for value in text.split(",")]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--frames", type=int, default=250, help="frames of the clip (default 250: 10 s)")
    parser.add_argument("--seed", type=int, default=500, help="seed of the synthetic clip")
    parser.add_argument("--seeds", help="one seed per round instead of --seed: A-B or A,B,C")
    parser.add_argument("--delays", default="5,25", help="delays L, comma separated")
    parser.add_argument("--runs", type=int, default=3, help="runs per arm")
    parser.add_argument("--profiles", default=",".join(SIMULATED_MS), help="simulated tool times (SIMULATED_MS)")
    parser.add_argument("--max-behind", type=float, help="also run each row with stream --max-behind SECONDS (arm on)")
    parser.add_argument("--busy", type=int, default=0, help="processes that spin during the bench (default 0)")
    parser.add_argument("--json", type=Path, help="write each row (and its result document) as one JSON line here")
    parser.add_argument("--price", help="factors k of the price model instead of the live chain (module docstring)")
    parser.add_argument("--kinds", default="plain,cut", help="clip kinds of --price (evalx.streambench)")
    args = parser.parse_args(argv)
    if args.price is not None:
        if args.max_behind is None:
            parser.error("--price needs --max-behind")
        seeds = [args.seed] if args.seeds is None else _seed_list(args.seeds)
        factors = [float(value) for value in args.price.split(",")]
        delay = int(args.delays.split(",")[0])
        found = price_rows(args.kinds.split(","), seeds, factors, args.max_behind, args.frames, delay)
        if args.json is not None:
            with args.json.open("a", encoding="utf-8") as handle:
                for row in found:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        print("\n".join(price_lines(found)))
        return 0
    delays = [int(value) for value in args.delays.split(",")]
    profiles = args.profiles.split(",")
    seeds = [args.seed] if args.seeds is None else _seed_list(args.seeds)
    print("\n".join(TABLE_HEAD), flush=True)
    rows = bench_rows(args.frames, seeds, delays, args.runs, profiles, max_behind=args.max_behind, busy=args.busy)
    with contextlib.ExitStack() as stack:
        handle = None if args.json is None else stack.enter_context(args.json.open("a", encoding="utf-8"))
        for row in rows:
            print(row_line(row), flush=True)
            if handle is not None:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
