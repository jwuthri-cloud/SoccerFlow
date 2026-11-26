"""Synthetic bench of the camera input: a stand-in camera, the capture, the stand-in tools, merge-feeds and stream.

The capture is the command ``capture`` of the run machine with the camera replaced by a stand-in
(``evalx.standin_camera``): the source, the slot rule, the files and the redaction of the
address are the real ones. The tools are ``tools/stand_in_tool.py`` (the script of the run
machine with the tool replaced by the answers of the synthetic clip) with ``--camera``. No
network address is opened and nothing calls a model. Four measurements (``docs/DECISIONS.md``,
entry "A camera as the clip of a live run"):

    python -m pitchvlm.evalx.camerabench equivalence --seeds 3000-3019   # C1
    python -m pitchvlm.evalx.camerabench loss --seeds 3100-3119          # C2
    python -m pitchvlm.evalx.camerabench leak                            # C3
    python -m pitchvlm.evalx.camerabench long --frames 45000             # C4

The simulated clock of the capture starts in the future (``START``), so that no reader of the
frames times out on a simulated run: a reader waits ``READER_WAIT_S`` after the time of a frame
on the clock of the system. Nothing here says how a real camera, a real RTSP server, the real
FFmpeg or the run machine behave.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pitchvlm.camera import CAMERA_URL_VARIABLE, IMAGE_SUFFIX, JPEG_QUALITY, LEDGER_NAME, CameraAddress
from pitchvlm.evalx.livebench import SETTINGS, STAND_IN, package_environment
from pitchvlm.evalx.standin_camera import CameraPlan, Clock, StandInCamera
from pitchvlm.fallbacks.fake import fake_detector_answers
from pitchvlm.synth import CLIP_ID, SyntheticScene

FPS = 25.0
#: Slot 0 of a simulated camera run, seconds since the epoch (2030): in the future of every reader.
START = 1_900_000_000.0
#: Seconds that the stand-in camera has been sending when the run starts.
LEAD_S = 2.0
#: A planted address for the leak test: every part is a word that appears nowhere else.
PLANTED_PARTS = {
    "user": "qzuserxq",
    "password": "qzpassxq",
    "host": "qzhostxq.invalid",
    "port": "47913",
    "path": "qzpathxq",
}
#: Built from its parts: a tracked file holds no address with its user information (the guard refuses one).
PLANTED_ADDRESS = "rtsps" + "://" + "{user}:{password}@{host}:{port}/{path}/live".format(**PLANTED_PARTS)


def jpeg_pixels(image: np.ndarray) -> np.ndarray:
    """Return the RGB pixels that a reader gets from the JPEG file that the capture writes for an image."""
    ok, data = cv2.imencode(IMAGE_SUFFIX, image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    assert ok
    decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return np.ascontiguousarray(decoded[:, :, ::-1])


@dataclass(frozen=True)
class Spec:
    """What one stand-in capture does: the clip, the camera plan, the run, and how it is started."""

    out: str
    frames: int
    seed: int
    width: int = 1280
    height: int = 720
    plan: Mapping[str, Any] | None = None
    real_sleep_s: float = 0.0
    run_frames: int | None = None
    #: More options of the command capture, for example ``["--reconnects", "2"]``.
    options: Sequence[str] = ()
    #: Slot from which the stand-in camera sends nothing more until the process is stopped (a stop-signal test).
    hold_from_slot: int | None = None

    def scene(self) -> SyntheticScene:
        return SyntheticScene(n_frames=self.frames, seed=self.seed, width=self.width, height=self.height)


def _plan(spec: Spec) -> CameraPlan:
    values = dict(spec.plan or {})
    for key in ("outages", "stalls"):
        values[key] = [tuple(span) for span in values.get(key, ())]
    for key in ("skipped", "other_size"):
        values[key] = frozenset(values.get(key, ()))
    values["late"] = {int(frame): delay for frame, delay in values.get("late", {}).items()}
    values["other_shape"] = tuple(values.get("other_shape", (360, 640)))
    return CameraPlan(t0=START - LEAD_S, **values)


def capture_stand_in(spec: Spec, argv_tail: Sequence[str] = ()) -> int:
    """Run the command ``capture`` (``cli.main``) on the stand-in camera of ``spec``; return its exit code.

    The address comes from ``PITCHVLM_CAMERA_URL`` as on the run machine. Camera frame ``k`` is
    frame ``k - LEAD_S * FPS`` of the synthetic clip (the first one before it), so that slot
    ``i`` of a camera that is on time holds frame ``i`` of the clip.
    """
    from pitchvlm import cli

    scene = spec.scene()
    lead = round(LEAD_S * FPS)
    clock = Clock(START - LEAD_S)
    camera = StandInCamera(
        _plan(spec), clock, lambda k: scene.image(min(max(k - lead, 0), spec.frames - 1)), real_sleep_s=spec.real_sleep_s,
        hold_from=None if spec.hold_from_slot is None else spec.hold_from_slot + lead,
    )

    def handler(args: argparse.Namespace) -> int:
        address = CameraAddress.from_environment()
        os.environ.pop(CAMERA_URL_VARIABLE, None)
        return cli.run_capture(args, address, open_device=camera, clock=clock, sleep=clock.sleep)

    original = cli._cmd_capture
    cli._cmd_capture = handler  # the parser takes the handler by its name when main builds it
    try:
        run_frames = spec.frames if spec.run_frames is None else spec.run_frames
        return cli.main(
            ["capture", "--out", spec.out, "--start-at", repr(START), "--frames", str(run_frames),
             "--size", f"{spec.width}x{spec.height}", *spec.options, *argv_tail]
        )
    finally:
        cli._cmd_capture = original


def capture_command(spec: Spec, argv_tail: Sequence[str] = ()) -> list[str]:
    """Return the command of a stand-in capture as its own process."""
    text = json.dumps(spec.__dict__)
    return [sys.executable, "-m", "pitchvlm.evalx.camerabench", "capture-stand-in", "--spec", text, *argv_tail]


def capture_environment(address: str = PLANTED_ADDRESS) -> dict[str, str]:
    environment = package_environment()
    environment[CAMERA_URL_VARIABLE] = address
    return environment


@dataclass(frozen=True)
class Chain:
    """A camera run on disk: the data folder, the camera folder, the clip jobs and the answers of the stand-ins."""

    root: Path
    clip: Path
    jobs: Mapping[str, Path]
    answers: Mapping[str, Path]


def write_answers(work: Path, scene: SyntheticScene, seed: int) -> dict[str, Path]:
    """Write the answers file of each stand-in tool: the payload of each frame of the clip, by its JPEG pixels."""
    keys = [hashlib.sha256(jpeg_pixels(scene.image(frame)).tobytes()).hexdigest() for frame in range(scene.info.n_frames)]
    people, balls = fake_detector_answers(scene, seed=seed)
    detector = []
    for item in people:
        rows = [{"box_xyxy": [p.box.x1, p.box.y1, p.box.x2, p.box.y2], "score": 0.9, "label": "person"} for p in item.people]
        rows += [
            {"box_xyxy": [b.box.x1, b.box.y1, b.box.x2, b.box.y2], "score": b.score, "label": "sports ball"}
            for b in balls.get(item.frame, ())
        ]
        detector.append({"detections": rows})
    calibration = [
        {"homography_pitch_to_image": np.linalg.inv(scene.homography(frame)).tolist(), "pitch_origin": "centre"}
        for frame in range(scene.info.n_frames)
    ]
    answers: dict[str, Path] = {}
    for tool, values in (("rfdetr", detector), ("pnlcalib", calibration)):
        answers[tool] = work / f"{tool}-answers.json"
        answers[tool].write_text(json.dumps({"by_image": dict(zip(keys, values, strict=True))}), encoding="utf-8")
    return answers


def write_jobs(work: Path, root: Path, clip: Path) -> dict[str, Path]:
    """Write the clip job of each tool with the command ``clip-job``, as on the run machine."""
    from pitchvlm import cli

    jobs: dict[str, Path] = {}
    for tool in ("rfdetr", "pnlcalib"):
        jobs[tool] = work / f"{tool}-job.json"
        arguments = ["clip-job", "--tool", tool, "--clip", str(clip), "--root", str(root), "--out", str(jobs[tool])]
        if cli.main([*arguments, "--config", str(SETTINGS)]) != 0:
            raise RuntimeError(f"clip-job failed for {tool}")
    return jobs


def tool_command(chain: Chain, tool: str, out: Path, feed: Path, *, camera: bool, skip_behind: bool = False) -> list[str]:
    command = [sys.executable, str(STAND_IN), "--answers", str(chain.answers[tool]), "--"]
    command += [tool, "--job", str(chain.jobs[tool]), "--root", str(chain.root), "--out", str(out), "--device", "cpu"]
    command += ["--feed", str(feed)]
    if camera:
        command += ["--camera"]
    if skip_behind:
        command += ["--skip-behind"]
    if tool == "pnlcalib":
        command += ["--repo", str(chain.root), "--weights-kp", "kp.pth", "--weights-lines", "lines.pth"]
    return command


@dataclass(frozen=True)
class ChainRun:
    codes: dict[str, int]
    outputs: dict[str, str]
    out: Path


def run_chain(
    chain: Chain,
    work: Path,
    *,
    camera: bool,
    labels: Path | None,
    live: bool = False,
    timeout_s: float = 1800.0,
    stream_arguments: Sequence[str] = (),
    capture: subprocess.Popen[str] | None = None,
) -> ChainRun:
    """Run the two stand-in tools, merge-feeds and stream once; every process must end by itself within ``timeout_s``.

    With ``live`` the tools write into named pipes and stream reads the merger through a pipe,
    as on the run machine; without it each step writes a file and the next one reads it.
    ``capture`` is a capture process that runs at the same time (its output is collected too).
    """
    work.mkdir(parents=True, exist_ok=True)
    environment = package_environment()
    feeds = {tool: work / (f"{tool}.fifo" if live else f"{tool}.jsonl") for tool in ("rfdetr", "pnlcalib")}
    if live:
        for path in feeds.values():
            os.mkfifo(path)
    codes: dict[str, int] = {}
    outputs: dict[str, str] = {}
    tools = {
        tool: subprocess.Popen(
            tool_command(chain, tool, work / f"{tool}-copy.jsonl", feeds[tool], camera=camera),
            env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for tool in feeds
    }
    merged = work / "lines.jsonl"
    stream_command = [
        sys.executable, "-m", "pitchvlm.cli", "stream", "--clip", str(chain.clip), "--out", str(work / "out"),
        "--allow-dirty", *stream_arguments,
    ]
    if labels is not None:
        stream_command += ["--labels", str(labels), "--split", "demo"]
    processes: list[subprocess.Popen[str]] = list(tools.values())
    try:
        if not live:
            for tool, process in tools.items():
                outputs[tool], _ = process.communicate(timeout=timeout_s)
                codes[tool] = process.returncode
        with merged.open("w", encoding="utf-8") as handle:
            merge = subprocess.Popen(
                [sys.executable, "-m", "pitchvlm.cli", "merge-feeds", "--detector", str(feeds["rfdetr"]),
                 "--calibration", str(feeds["pnlcalib"]), "--wait", "1"],
                env=environment, stdout=subprocess.PIPE if live else handle, stderr=subprocess.PIPE, text=True,
            )
            processes.append(merge)
            if live:
                stream = subprocess.Popen(
                    [*stream_command, "--lines", "-"], env=environment, stdin=merge.stdout,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                )
                assert merge.stdout is not None
                merge.stdout.close()
            else:
                _, outputs["merge-feeds"] = merge.communicate(timeout=timeout_s)
                codes["merge-feeds"] = merge.returncode
                stream = subprocess.Popen(
                    [*stream_command, "--lines", str(merged)], env=environment,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                )
            processes.append(stream)
            outputs["stream"], _ = stream.communicate(timeout=timeout_s)
            codes["stream"] = stream.returncode
            if live:
                _, outputs["merge-feeds"] = merge.communicate(timeout=timeout_s)
                codes["merge-feeds"] = merge.returncode
                for tool, process in tools.items():
                    outputs[tool], _ = process.communicate(timeout=timeout_s)
                    codes[tool] = process.returncode
        if capture is not None:
            outputs["capture"], _ = capture.communicate(timeout=timeout_s)
            codes["capture"] = capture.returncode
    finally:
        for process in [*processes, *([] if capture is None else [capture])]:
            if process.poll() is None:
                process.kill()
                process.wait()
    return ChainRun(codes=codes, outputs=outputs, out=work / "out")


def prepare_camera(work: Path, spec: Spec, *, process: bool = False) -> tuple[Chain, int | subprocess.Popen[str], Path]:
    """Capture a stand-in camera run into ``work/data/live/<clip>`` (in this process, or started as a process).

    Returns the chain, the exit code of the capture (or its process) and the label file of the clip.
    """
    root = work / "data"
    clip = root / "live" / CLIP_ID
    clip.parent.mkdir(parents=True, exist_ok=True)
    spec = Spec(**{**spec.__dict__, "out": str(clip)})
    scene = spec.scene()
    answers = write_answers(work, scene, spec.seed)
    labels = work / "labels" / CLIP_ID / "Labels-GameState.json"
    labels.parent.mkdir(parents=True)
    labels.write_text(json.dumps(scene.gsr_labels()), encoding="utf-8")
    if process:
        capture: int | subprocess.Popen[str] = subprocess.Popen(
            capture_command(spec), env=capture_environment(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        while not (clip / "camera.json").exists():
            if capture.poll() is not None:
                raise RuntimeError(f"the capture ended before it wrote its record: {capture.communicate()[0]}")
            time.sleep(0.01)
    else:
        saved = os.environ.get(CAMERA_URL_VARIABLE)
        os.environ[CAMERA_URL_VARIABLE] = PLANTED_ADDRESS
        try:
            capture = capture_stand_in(spec)
        finally:
            if saved is not None:
                os.environ[CAMERA_URL_VARIABLE] = saved
    jobs = write_jobs(work, root, clip)
    return Chain(root=root, clip=clip, jobs=jobs, answers=answers), capture, labels


def recorded_chain(work: Path, camera: Chain) -> Chain:
    """Return the same frames as a recorded clip: a plain folder of the JPEG files of the camera run, and its jobs."""
    root = work / "data"
    clip = root / "recorded" / CLIP_ID
    shutil.copytree(camera.clip / "img1", clip / "img1")
    return Chain(root=root, clip=clip, jobs=write_jobs(work, root, clip), answers=camera.answers)


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def compare_outputs(first: Path, second: Path) -> dict[str, bool]:
    """Compare the prediction file, the video and the section ``result`` of ``result.json`` of two stream runs."""
    predictions = Path("predictions") / "SoccerNetGS-demo" / "pitchvlm" / f"{CLIP_ID}.json"
    video = Path(CLIP_ID) / f"{CLIP_ID}.mp4"
    result = Path(CLIP_ID) / "result.json"

    def section(base: Path) -> Any:
        path = base / result
        return json.loads(path.read_text(encoding="utf-8"))["result"] if path.exists() else None

    return {
        "predictions": _sha(first / predictions) is not None and _sha(first / predictions) == _sha(second / predictions),
        "video": _sha(first / video) is not None and _sha(first / video) == _sha(second / video),
        "result": section(first) is not None and section(first) == section(second),
    }


# --- C1: equivalence -------------------------------------------------------------------------------------


def equivalence(seeds: Sequence[int], n_frames: int, work: Path) -> list[dict[str, Any]]:
    """C1: a camera on time without loss against the recorded clip of the same frames, and against stream --synthetic."""
    rows = []
    for seed in seeds:
        base = work / f"c1-{seed}"
        chain, code, labels = prepare_camera(base, Spec(out="", frames=n_frames, seed=seed))
        camera_run = run_chain(chain, base / "camera-run", camera=True, labels=labels)
        recorded = recorded_chain(base / "rec", chain)
        recorded_run = run_chain(recorded, base / "recorded-run", camera=False, labels=labels)
        synthetic = base / "synthetic-out"
        subprocess.run(
            [sys.executable, "-m", "pitchvlm.cli", "stream", "--synthetic", "--frames", str(n_frames), "--seed",
             str(seed), "--out", str(synthetic), "--allow-dirty"],
            env=package_environment(), check=True, capture_output=True,
        )
        rows.append({
            "seed": seed, "capture": code, "codes": camera_run.codes, "recorded_codes": recorded_run.codes,
            "vs_recorded": compare_outputs(camera_run.out, recorded_run.out),
            "vs_synthetic": compare_outputs(camera_run.out, synthetic),
            "recorded_vs_synthetic": compare_outputs(recorded_run.out, synthetic),
        })
        print(json.dumps(rows[-1]), flush=True)
        shutil.rmtree(base)
    return rows


# --- C2: loss --------------------------------------------------------------------------------------------


def loss_plan(seed: int, n_frames: int) -> dict[str, Any]:
    """The fixed fault schedule of a seed: a stall, a long stall that ends the connection, an outage, an early end,
    late single frames, frames never sent, one frame of another size (seconds of the run, frames of the camera)."""
    rng = random.Random(seed)
    seconds = n_frames / FPS
    lead = round(LEAD_S * FPS)
    stall = START + rng.uniform(0.03, 0.06) * seconds
    long_stall = START + rng.uniform(0.1, 0.12) * seconds
    outage = START + rng.uniform(0.35, 0.4) * seconds
    late = {lead + rng.randrange(n_frames // 40): rng.uniform(0.05, 1.5) for _ in range(5)}
    skipped = sorted({lead + rng.randrange(n_frames // 2) for _ in range(4)})
    return {
        "stalls": [(stall, stall + rng.uniform(0.2, 1.5)), (long_stall, long_stall + rng.uniform(2.5, 4.0))],
        "outages": [(outage, outage + rng.uniform(1.0, 4.0))],
        "ends_at": START + rng.uniform(0.6, 0.65) * seconds,
        "late": late,
        "skipped": skipped,
        "other_size": [lead + rng.randrange(n_frames // 4, n_frames // 3)],
    }


#: The capture options of the loss bench: short tries, so that the early end of the camera is found within the run
#: (2 tries of 1 s after a wait of 1 s each), and an outage of up to 4 s is still bridged.
LOSS_OPTIONS = ("--open-timeout", "1", "--reconnects", "2")


def ledger_counts(folder: Path) -> dict[str, Any]:
    """Count the ledger of a camera run: frames given, frames lost, the end line."""
    given = lost = 0
    end = None
    for text in (folder / LEDGER_NAME).read_text(encoding="utf-8").splitlines():
        line = json.loads(text)
        if "end" in line:
            end = line
        elif "frame" in line:
            given += 1
        else:
            lost += line["lost"][1] - line["lost"][0]
    return {"given": given, "lost": lost, "end": end}


def loss(seeds: Sequence[int], n_frames: int, work: Path, *, live: bool) -> list[dict[str, Any]]:
    """C2: the fault schedule of each seed through the chain; every frame must be accounted for, every process must end."""
    rows = []
    for seed in seeds:
        base = work / f"c2-{seed}"
        spec = Spec(out="", frames=n_frames, seed=seed, plan=loss_plan(seed, n_frames), real_sleep_s=0.002 if live else 0.0,
                    options=LOSS_OPTIONS)
        chain, capture, labels = prepare_camera(base, spec, process=live)
        started = time.monotonic()
        run = run_chain(chain, base / "run", camera=True, labels=labels, live=live,
                        capture=capture if isinstance(capture, subprocess.Popen) else None)
        seconds = time.monotonic() - started
        codes = dict(run.codes)
        if not isinstance(capture, subprocess.Popen):
            codes["capture"] = capture
        counts = ledger_counts(chain.clip)
        document_path = run.out / CLIP_ID / "result.json"
        document = json.loads(document_path.read_text(encoding="utf-8")) if document_path.exists() else {}
        facts = {tool: json.loads((base / "run" / f"{tool}-copy-runner.json").read_text(encoding="utf-8"))
                 for tool in ("rfdetr", "pnlcalib") if (base / "run" / f"{tool}-copy-runner.json").exists()}
        stream_frames = len((document.get("result") or {}).get("frames", []))
        rows.append({
            "seed": seed, "codes": codes, "seconds": round(seconds, 1), "ledger": counts,
            "accounted": counts["given"] + counts["lost"] == n_frames,
            "tools": {tool: (fact["answered_frames"], fact["skipped_frames"], fact.get("camera_lost_frames"))
                      for tool, fact in facts.items()},
            "stream_frames": stream_frames,
            "stream_camera": (document.get("provenance", {}).get("inputs", {}).get("camera") or {}).get("end"),
            "missing_parts": document.get("missing_parts"),
        })
        print(json.dumps(rows[-1]), flush=True)
        shutil.rmtree(base)
    return rows


# --- C3: leak --------------------------------------------------------------------------------------------


def count_parts(texts: Mapping[str, str]) -> dict[str, int]:
    """Count each planted part of the address in every text, letter case ignored."""
    return {
        name: sum(text.lower().count(part.lower()) for text in texts.values()) for name, part in PLANTED_PARTS.items()
    }


def files_text(folder: Path) -> dict[str, str]:
    """Return every file below a folder as text (bytes that are not UTF-8 are replaced), by path."""
    return {
        str(path): path.read_bytes().decode("utf-8", errors="replace") for path in folder.rglob("*") if path.is_file()
    }


def leak_case(work: Path, name: str, spec: Spec, *, address: str = PLANTED_ADDRESS, stop_after: int | None = None,
              chain: bool = False) -> dict[str, Any]:
    """Run one capture as a process with the planted address (and the chain after it); count the parts everywhere."""
    base = work / f"c3-{name}"
    root = base / "data"
    clip = root / "live" / CLIP_ID
    clip.parent.mkdir(parents=True)
    spec = Spec(**{**spec.__dict__, "out": str(clip)})
    process = subprocess.Popen(
        capture_command(spec), env=capture_environment(address), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if stop_after is not None:
        while process.poll() is None:
            ledger = clip / LEDGER_NAME
            if ledger.exists() and len(ledger.read_text(encoding="utf-8").splitlines()) >= stop_after:
                process.send_signal(signal.SIGINT)
                break
            time.sleep(0.005)
    stdout, stderr = process.communicate(timeout=600)
    texts = {"capture stdout": stdout, "capture stderr": stderr}
    codes = {"capture": process.returncode}
    if chain:
        scene = spec.scene()
        answers = write_answers(base, scene, spec.seed)
        jobs = write_jobs(base, root, clip)
        labels = base / "labels" / CLIP_ID / "Labels-GameState.json"
        labels.parent.mkdir(parents=True)
        labels.write_text(json.dumps(scene.gsr_labels()), encoding="utf-8")
        run = run_chain(Chain(root=root, clip=clip, jobs=jobs, answers=answers), base / "run", camera=True,
                        labels=labels, stream_arguments=["--live-out", str(base / "live.jsonl")])
        codes.update(run.codes)
        texts.update({f"{key} output": value for key, value in run.outputs.items()})
    texts.update(files_text(base))
    row = {"case": name, "codes": codes, "counts": count_parts(texts), "texts": len(texts)}
    print(json.dumps(row), flush=True)
    shutil.rmtree(base)
    return row


def leak(work: Path, n_frames: int) -> list[dict[str, Any]]:
    """C3: the planted address on the success path and on every failure path that can be reached."""
    native = f"[rtsp @ 0x1] Could not open {PLANTED_ADDRESS}: user {PLANTED_PARTS['user']} on {PLANTED_PARTS['host']}"
    error = f"OpenCV(5.0.0) cap_ffmpeg: can't open {PLANTED_ADDRESS}"
    rows = [
        leak_case(work, "success", Spec(out="", frames=n_frames, seed=1, plan={"native_message": native}), chain=True),
        leak_case(work, "never-opens", Spec(out="", frames=n_frames, seed=1,
                                            plan={"never_opens": True, "native_message": native, "open_error": error})),
        leak_case(work, "wrong-scheme", Spec(out="", frames=n_frames, seed=1),
                  address=PLANTED_ADDRESS.replace("rtsps://", "http://", 1)),
        leak_case(work, "no-scheme", Spec(out="", frames=n_frames, seed=1),
                  address=PLANTED_ADDRESS.replace("rtsps://", "", 1)),
        leak_case(work, "lost-connection", Spec(out="", frames=n_frames, seed=1, plan={
            "outages": [(START + 1.0, START + 2.5)], "ends_at": START + 4.0, "native_message": native, "open_error": error,
        }), chain=True),
        leak_case(work, "wrong-size", Spec(out="", frames=n_frames, seed=1, plan={"native_message": native},
                                           options=("--size", "640x360"))),
        leak_case(work, "stop-signal", Spec(out="", frames=n_frames, seed=1, plan={"native_message": native},
                                            hold_from_slot=n_frames // 2), stop_after=n_frames // 2),
    ]
    return rows


# --- C4: a long run --------------------------------------------------------------------------------------


def _rss_kb(pid: int) -> int | None:
    result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    text = result.stdout.strip()
    return int(text) if text.isdigit() else None


def _folder_bytes(folder: Path) -> int:
    return sum(entry.stat().st_size for entry in folder.rglob("*") if entry.is_file()) if folder.exists() else 0


def long_run(work: Path, n_frames: int, *, width: int, height: int, sample_s: float) -> dict[str, Any]:
    """C4: a long camera run through the live chain (named pipes), with the memory and disk sampled as it runs."""
    spec = Spec(out="", frames=n_frames, seed=7, width=width, height=height)
    chain, capture, labels = prepare_camera(work, spec, process=True)
    assert isinstance(capture, subprocess.Popen)
    live = work / "live.jsonl"
    samples: list[dict[str, Any]] = []
    done = threading.Event()
    stream_pid: dict[str, int] = {}

    def sample() -> None:
        started = time.monotonic()
        while not done.is_set():
            ledger = chain.clip / LEDGER_NAME
            captured = len(ledger.read_bytes().splitlines()) if ledger.exists() else 0
            decided = len(live.read_bytes().splitlines()) if live.exists() else 0
            pid = stream_pid.get("stream")
            samples.append({
                "t": round(time.monotonic() - started, 1), "ledger_lines": captured, "live_lines": decided,
                "capture_rss_kb": _rss_kb(capture.pid), "stream_rss_kb": None if pid is None else _rss_kb(pid),
                "camera_bytes": _folder_bytes(chain.clip),
            })
            done.wait(sample_s)

    sampler = threading.Thread(target=sample, daemon=True)
    original_popen = subprocess.Popen

    def popen(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        process = original_popen(*args, **kwargs)
        command = args[0]
        if "stream" in command and "pitchvlm.cli" in command:
            stream_pid["stream"] = process.pid
        return process

    subprocess.Popen = popen  # type: ignore[misc]  (to learn the process id of stream for the sampler)
    sampler.start()
    try:
        run = run_chain(chain, work / "run", camera=True, labels=None, live=True, capture=capture,
                        stream_arguments=["--live-out", str(live), "--delay", "25"])
    finally:
        subprocess.Popen = original_popen  # type: ignore[misc]
        done.set()
        sampler.join()
    document_path = run.out / CLIP_ID / "result.json"
    document = json.loads(document_path.read_text(encoding="utf-8")) if document_path.exists() else {}
    return {
        "codes": run.codes, "ledger": ledger_counts(chain.clip), "samples": samples,
        "camera_bytes_end": _folder_bytes(chain.clip), "out_bytes": _folder_bytes(run.out),
        "stream_frames": len((document.get("result") or {}).get("frames", [])),
        "tail": {key: value[-600:] for key, value in run.outputs.items()},
    }


# --- Command line ----------------------------------------------------------------------------------------


def _seeds(text: str) -> list[int]:
    if "-" in text:
        first, last = (int(value) for value in text.split("-", 1))
        return list(range(first, last + 1))
    return [int(value) for value in text.split(",")]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    stand_in = commands.add_parser("capture-stand-in", help="the command capture on a stand-in camera (one process)")
    stand_in.add_argument("--spec", required=True, help="the Spec as JSON")
    for name in ("equivalence", "loss", "leak", "long"):
        command = commands.add_parser(name)
        command.add_argument("--frames", type=int, default={"long": 45000, "leak": 100}.get(name, 250))
        command.add_argument("--json", type=Path, help="write the rows here (JSON)")
        if name in ("equivalence", "loss"):
            command.add_argument("--seeds", default="3000-3019")
        if name == "loss":
            command.add_argument("--live", action="store_true", help="the capture as a process at the same time, pipes")
        if name == "long":
            command.add_argument("--size", default="1920x1080")
            command.add_argument("--sample", type=float, default=5.0, help="seconds between two samples")
    args, rest = parser.parse_known_args(argv)
    if args.command == "capture-stand-in":
        values = json.loads(args.spec)
        return capture_stand_in(Spec(**values), rest)
    if rest:
        parser.error(f"unknown arguments {rest}")
    with tempfile.TemporaryDirectory(prefix="camerabench-") as folder:
        work = Path(folder)
        if args.command == "equivalence":
            rows: Any = equivalence(_seeds(args.seeds), args.frames, work)
        elif args.command == "loss":
            rows = loss(_seeds(args.seeds), args.frames, work, live=args.live)
        elif args.command == "leak":
            rows = leak(work, args.frames)
        else:
            width, height = (int(value) for value in args.size.split("x"))
            rows = long_run(work, args.frames, width=width, height=height, sample_s=args.sample)
    if args.json is not None:
        args.json.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
