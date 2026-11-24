"""Synthetic bench of the live number step (``livenumbers``): its cost to the frame path, its numbers, its latency.

The streams run on synthetic clips (``synth``) with the stand-ins of the detector and of the
calibration (``fallbacks.fake``) and the fake model of the scene (``vlm.fake``), which reads the
true number of each crop. A set delay before each answer (``DelayedBackend``) stands for the time
of a model. Three measurements (``docs/DECISIONS.md``, entry "Shirt numbers in the live path"):

- ``n1``: the time per frame of the stream (the stage total of ``timing.StageClock.frame``) with
  the number engine ``none`` and with ``model``, back to back, in both orders, on the same seed in
  each pair, as fast as the stream goes (the worker competes with the frame path all the time);
- ``n2``: the set of (track, number) at the end of a stream against the clip path with the same
  engine file and the same fake answers (``pipeline.run_clip``): each id of the stream is mapped
  to the track of the clip path that holds most of its observed boxes;
- ``n3``: the time from the arrival of the first frame of an id to its number line, with the
  frames paced at the frame rate of the clip (``--pace``).

Run ``python -m pitchvlm.evalx.numberbench n1 --seeds 2200-2205`` and
``python -m pitchvlm.evalx.numberbench n23 --seeds 2200-2239``. Nothing here says how a real
model, a real detector or real footage behaves.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from pitchvlm.config import EngineSettings, PipelineConfig
from pitchvlm.fallbacks.fake import fake_calibration_answers, fake_detector_answers
from pitchvlm.livenumbers import LiveNumbers, NumberSettings, apply_numbers, wire
from pitchvlm.liveout import open_live_out
from pitchvlm.pipeline import ToolAnswers, run_clip
from pitchvlm.schemas import ClipResult
from pitchvlm.stream import DetectorStream, StreamSettings, default_history_frames, stand_in_inputs
from pitchvlm.synth import SyntheticScene, SyntheticSource
from pitchvlm.timing import StageClock
from pitchvlm.vlm.fake import FakeBackend
from pitchvlm.vlm.types import VLMRequest, VLMResponse

ENGINES = {
    "none": EngineSettings(people="detector", pitch_mapping="calibration", number="none"),
    "model": EngineSettings(people="detector", pitch_mapping="calibration", number="model"),
}
#: Seconds before each answer of the fake model, the arms of the task.
ANSWER_DELAYS_S = (0.5, 2.0, 5.0)
#: The cap of number requests of the n2 runs: high, so that the cap takes no number (its count is reported).
N2_CAP = 1000


class DelayedBackend:
    """Answer with an inner backend after ``delay_s`` seconds (``sleep``), as a model that takes that long."""

    def __init__(self, inner: Any, delay_s: float, sleep: Callable[[float], None] = time.sleep) -> None:
        self._inner = inner
        self._delay_s = delay_s
        self._sleep = sleep

    def generate(self, request: VLMRequest) -> VLMResponse:
        self._sleep(self._delay_s)
        return self._inner.generate(request)


def stream_case(
    seed: int,
    n_frames: int,
    *,
    numbers: bool,
    delay_s: float,
    cap: int,
    pace_fps: float | None = None,
    live_dir: str | None = None,
) -> dict[str, Any]:
    """Run one stream of the synthetic clip of ``seed`` with a live output to a file; return its measures.

    The image of each frame is read inside the frame sample (stage ``read``), as ``stream`` does.
    With ``pace_fps`` frame ``i`` is not pushed before the start plus ``i / pace_fps``.
    """
    scene = SyntheticScene(n_frames=n_frames, seed=seed)
    source = SyntheticSource(scene)
    info = scene.info
    config = PipelineConfig(engines=ENGINES["model" if numbers else "none"])
    history = default_history_frames(config, info.fps)
    settings = StreamSettings(delay_frames=25, history_frames=history)
    clock = StageClock()
    stream = DetectorStream(config, settings, clip_id=info.clip_id, fps=info.fps, size=(info.width, info.height), clock=clock)
    worker = None
    if numbers:
        backend = DelayedBackend(FakeBackend(scene, seed=seed), delay_s)
        worker = LiveNumbers(config, backend, NumberSettings(max_requests=cap, close_frames=history), info=info)
    with tempfile.TemporaryDirectory(dir=live_dir) as folder:
        live = open_live_out(str(Path(folder) / "live.jsonl"), info)
        live.start(delay_frames=25, history_frames=history, numbers=numbers)
        writer = wire(stream, worker, live)
        people, balls = fake_detector_answers(scene, seed=seed)
        start = time.monotonic()
        for item in stand_in_inputs(people, balls, fake_calibration_answers(scene)):
            if pace_fps is not None:
                wait = start + item.frame / pace_fps - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
            stream.push(item, lambda frame=item.frame: source.read(frame))
        stream.finish()
        if worker is not None and writer is not None:
            writer.write(worker.finish(), stream.n_frames - 1)
        live.finish()
        live.close()
        wall_s = time.monotonic() - start
    total = clock.frame_report()["total"]
    result = stream.result()
    case: dict[str, Any] = {
        "seed": seed,
        "numbers": numbers,
        "delay_s": delay_s if numbers else None,
        "n_frames": n_frames,
        "frame_p50_ms": total["p50_ms"],
        "frame_p95_ms": total["p95_ms"],
        "frame_max_ms": total["max_ms"],
        "wall_s": wall_s,
    }
    if worker is not None:
        result = apply_numbers(result, worker.lines)
        report = worker.report()
        case["report"] = {key: value for key, value in report.items() if key != "latency_s"}
        case["lines"] = [
            {"id": line.track_id, "export": line.export, "reason": line.reason, "latency_s": line.latency_s,
             "first_seen_frame": line.first_seen_frame, "last_crop": line.last_frame}
            for line in worker.lines
        ]
        case["observations"] = _observations(result)
    return case


def _observations(result: ClipResult) -> dict[int, list[list[float]]]:
    """Return the observed (frame, box) of every id of a result: ``{id: [[frame, x1, y1, x2, y2], ...]}``."""
    return {
        track.track_id: [[obs.frame, obs.box.x1, obs.box.y1, obs.box.x2, obs.box.y2] for obs in track.observations]
        for track in result.tracks
    }


def offline_case(seed: int, n_frames: int) -> dict[str, Any]:
    """Run the clip path with the number engine model and the fake model; return its tracks and numbers."""
    scene = SyntheticScene(n_frames=n_frames, seed=seed)
    config = PipelineConfig(engines=ENGINES["model"])
    people, balls = fake_detector_answers(scene, seed=seed)
    tools = ToolAnswers(people=people, balls=balls, cameras=fake_calibration_answers(scene))
    requests: Counter[str] = Counter()
    fake = FakeBackend(scene, seed=seed)

    class Counting:
        def generate(self, request: VLMRequest) -> VLMResponse:
            requests[request.meta.step] += 1
            return fake.generate(request)

    result = run_clip(SyntheticSource(scene), Counting(), config, tools=tools)
    return {
        "seed": seed,
        "numbers": {track.track_id: track.number_export for track in result.tracks},
        "roles": {track.track_id: track.role for track in result.tracks},
        "observations": _observations(result),
        "requests": dict(requests),
    }


def compare(stream: dict[str, Any], offline: dict[str, Any]) -> dict[str, Any]:
    """Compare the (track, number) sets of a stream and of the clip path (module docstring, ``n2``)."""
    owner: dict[tuple[float, ...], int] = {}
    for track_id, observations in offline["observations"].items():
        for obs in observations:
            owner[tuple(obs)] = int(track_id)
    mapping: dict[int, int | None] = {}
    for track_id, observations in stream["observations"].items():
        votes = Counter(owner[tuple(obs)] for obs in observations if tuple(obs) in owner)
        mapping[int(track_id)] = votes.most_common(1)[0][0] if votes else None
    offline_numbers = {int(key): value for key, value in offline["numbers"].items()}
    roles = {int(key): value for key, value in offline["roles"].items()}
    sizes = {int(key): len(value) for key, value in offline["observations"].items()}
    lines = {line["id"]: line for line in stream["lines"]}
    stream_set = {(mapping[key], line["export"]) for key, line in lines.items() if line["export"] is not None}
    offline_set = {(key, value) for key, value in offline_numbers.items() if value is not None}
    differences = []
    for track_id, number in sorted(offline_set - stream_set, key=str):
        ids = [key for key, value in mapping.items() if value == track_id]
        asked = [(key, lines[key]["export"]) for key in ids if key in lines]
        why = "no stream id holds its boxes" if not ids else ("no id was asked" if not asked else f"read {asked}")
        differences.append({"side": "clip path only", "track": track_id, "number": number, "stream_ids": ids,
                            "role": roles.get(track_id), "observations": sizes.get(track_id), "why": why})
    for track_id, number in sorted(stream_set - offline_set, key=str):
        ids = [key for key, value in mapping.items() if value == track_id and key in lines]
        why = (
            "the stream id holds no box of the clip path" if track_id is None
            else f"the clip path gives {offline_numbers.get(track_id)}"
        )
        differences.append({"side": "stream only", "track": track_id, "number": number, "stream_ids": ids,
                            "role": roles.get(track_id) if track_id is not None else None,
                            "observations": sizes.get(track_id) if track_id is not None else None, "why": why})
    return {
        "equal": stream_set == offline_set,
        "stream_pairs": len(stream_set),
        "offline_pairs": len(offline_set),
        "differences": differences,
    }


def _seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        seeds.extend(range(int(first), int(last or first) + 1))
    return seeds


def run_n1(seeds: Sequence[int], n_frames: int, delays: Sequence[float], pairs: int, out: Path, live_dir: str | None) -> None:
    """Pairs of ``none`` and ``model`` runs, back to back, the order flipped every pair; one JSON line per run."""
    with out.open("a", encoding="utf-8") as handle:
        for delay_s in delays:
            for index in range(pairs):
                seed = seeds[index % len(seeds)]
                order = (False, True) if index % 2 == 0 else (True, False)
                for numbers in order:
                    case = stream_case(seed, n_frames, numbers=numbers, delay_s=delay_s, cap=N2_CAP, live_dir=live_dir)
                    case.pop("observations", None)
                    case.update({"bench": "n1", "arm_delay_s": delay_s, "pair": index})
                    handle.write(json.dumps(case) + "\n")
                    handle.flush()
                    print(f"n1 delay {delay_s} pair {index} numbers {numbers}: p50 {case['frame_p50_ms']:.2f} ms "
                          f"p95 {case['frame_p95_ms']:.2f} ms", flush=True)


def _n23_job(job: tuple[int, int, float, float | None, str | None]) -> dict[str, Any]:
    seed, n_frames, delay_s, pace, live_dir = job
    case = stream_case(seed, n_frames, numbers=True, delay_s=delay_s, cap=N2_CAP, pace_fps=pace, live_dir=live_dir)
    return case


def run_n23(
    seeds: Sequence[int], n_frames: int, delays: Sequence[float], pace: float | None, jobs: int, out: Path,
    live_dir: str | None,
) -> None:
    """Every seed and answer delay: a paced stream with numbers, and one clip-path run per seed; one JSON line each."""
    with ProcessPoolExecutor(max_workers=jobs) as pool, out.open("a", encoding="utf-8") as handle:
        offline = dict(zip(seeds, pool.map(offline_case, seeds, [n_frames] * len(seeds)), strict=True))
        cases = pool.map(_n23_job, [(seed, n_frames, delay, pace, live_dir) for delay in delays for seed in seeds])
        for case in cases:
            compared = compare(case, offline[case["seed"]])
            case.pop("observations")
            record = {**case, "bench": "n23", "pace_fps": pace, "compare": compared,
                      "offline_requests": offline[case["seed"]]["requests"]}
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            latencies = [line["latency_s"] for line in case["lines"] if line["latency_s"] is not None]
            median = statistics.median(latencies) if latencies else float("nan")
            print(f"n23 seed {case['seed']} delay {case['delay_s']}: equal {compared['equal']}, "
                  f"{len(case['lines'])} lines, latency median {median:.2f} s", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bench (module docstring)."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("bench", choices=("n1", "n23"))
    parser.add_argument("--seeds", required=True, help="seeds, for example 2200-2239")
    parser.add_argument("--frames", type=int, default=750, help="frames of each synthetic clip (default 750)")
    parser.add_argument("--delays", default=",".join(str(value) for value in ANSWER_DELAYS_S),
                        help="seconds before each fake answer, comma separated")
    parser.add_argument("--pairs", type=int, default=6, help="n1: pairs of runs per answer delay (default 6)")
    parser.add_argument("--pace", type=float, default=25.0, help="n23: frames per second of the input (0: as fast)")
    parser.add_argument("--jobs", type=int, default=4, help="n23: processes")
    parser.add_argument("--out", type=Path, required=True, help="JSON lines file to append the runs to")
    parser.add_argument("--live-dir", default=None, help="folder of the temporary live outputs (default: the system's)")
    args = parser.parse_args(argv)
    seeds = _seeds(args.seeds)
    delays = [float(value) for value in args.delays.split(",")]
    if args.bench == "n1":
        run_n1(seeds, args.frames, delays, args.pairs, args.out, args.live_dir)
    else:
        run_n23(seeds, args.frames, delays, args.pace or None, args.jobs, args.out, args.live_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
