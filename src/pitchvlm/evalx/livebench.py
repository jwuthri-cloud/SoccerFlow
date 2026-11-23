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

Nothing here calls a model, and nothing says how a real tool, real footage or the run machine
behave.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
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


def bench_lines(n_frames: int, seed: int, delays: Sequence[int], runs: int, profiles: Sequence[str]) -> list[str]:
    """Run the chain ``runs`` times per (profile, delay), the order of the arms turned each run; return a table."""
    lines = [
        "| profile (simulated ms: detector, calibration) | L | run | load | exit codes | missing parts | "
        "source to decision | after frame f + L | line to decision | stream frame total | largest backlog |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    with tempfile.TemporaryDirectory(prefix="livebench-") as folder:
        work = Path(folder)
        clip = prepare_clip(work / "clip", n_frames, seed)
        arms = [(profile, delay) for profile in profiles for delay in delays]
        for run in range(runs):
            order = arms if run % 2 == 0 else list(reversed(arms))
            for profile, delay in order:
                load = os.getloadavg()[0]
                simulated = SIMULATED_MS[profile]
                outcome = run_live(
                    clip, work / f"run-{run}-{profile}-{delay}", delay=delay, pace=FPS, simulated_ms=simulated,
                    skip_behind_calibration=simulated["calibration"] > 1000.0 / FPS,
                )
                document = outcome.document or {}
                latency = document.get("latency") or {}
                missing = document.get("missing_parts") or {}
                missing_text = "; ".join(
                    f"{tool} " + ", ".join(f"{count} {reason}" for reason, count in reasons.items())
                    for tool, reasons in missing.items()
                ) or "none"
                total = (document.get("frame_timing") or {}).get("total")
                backlog = (latency.get("backlog_lines") or {}).get("max", "-")
                codes = ", ".join(f"{name} {code}" for name, code in outcome.codes.items())
                lines.append(
                    f"| {profile} ({simulated['detector']:g}, {simulated['calibration']:g}) | {delay} | {run + 1} | "
                    f"{load:.1f} | {codes} | {missing_text} | {_row(latency.get('source_to_decision'))} | "
                    f"{_row(latency.get('after_last_needed_frame'))} | {_row(latency.get('line_to_decision'))} | "
                    f"{_row(total)} | {backlog} |"
                )
                print(lines[-1], flush=True)
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--frames", type=int, default=250, help="frames of the clip (default 250: 10 s)")
    parser.add_argument("--seed", type=int, default=500, help="seed of the synthetic clip")
    parser.add_argument("--delays", default="5,25", help="delays L, comma separated")
    parser.add_argument("--runs", type=int, default=3, help="runs per arm")
    parser.add_argument("--profiles", default=",".join(SIMULATED_MS), help="simulated tool times (SIMULATED_MS)")
    args = parser.parse_args(argv)
    delays = [int(value) for value in args.delays.split(",")]
    profiles = args.profiles.split(",")
    print("\n".join(bench_lines(args.frames, args.seed, delays, args.runs, profiles)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
