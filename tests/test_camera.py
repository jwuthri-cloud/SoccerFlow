"""Tests of the camera input: the address and its redaction, the slot rule of the source on a scripted stand-in camera,
the camera clip folder, its two readers, the command capture, and the chain of a camera run.

No test opens a network address: the camera is ``evalx.standin_camera`` (the object that ``cv2.VideoCapture`` would
be), on a simulated clock, so that no test depends on the wall clock. Nothing here says how a real camera, a real
RTSP server or the real FFmpeg behave.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
import pytest

from pitchvlm import camera, cli, doctor
from pitchvlm.camera import (
    CAMERA_URL_VARIABLE,
    HIDDEN,
    HIDDEN_LINE,
    CameraAddress,
    CameraError,
    CameraFolderSource,
    CameraLedger,
    CameraSettings,
    CameraSource,
    Given,
    Lost,
    RedactedOutput,
    Warmup,
    ffmpeg_capture,
    ffmpeg_options,
    read_camera_record,
    write_capture,
)
from pitchvlm.evalx import camerabench
from pitchvlm.evalx.standin_camera import CameraPlan, Clock, StandInCamera

REPO = Path(__file__).resolve().parents[1]
START = 1_900_000_000.0
SIZE = (64, 48)


def address(scheme: str = "rtsp", user: str = "", password: str = "", host: str = "", port: str = "", path: str = "") -> str:
    """Build the text of a camera address from its parts (tracked files hold no address with its user information)."""
    userinfo = (user + (":" + password if password else "") + "@") if user else ""
    return scheme + "://" + userinfo + host + (":" + port if port else "") + ("/" + path if path else "")


#: The test address: every part is a word that the tests look for. Its host lies in the reserved domain "invalid".
PLANTED = address("rtsp", "cam-user", "cam-secret", "camera-host.invalid", "8554", "stream-path")
ADDRESS = CameraAddress.parse(PLANTED)
REDACTED = "rtsp://<hidden>"
PLENTY = 10**13


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_tool_for_camera", REPO / "tools" / "run_tool.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_tool_for_camera"] = module
    spec.loader.exec_module(module)
    return module


run_tool = load_script()


def numbered(frame: int) -> np.ndarray:
    """A frame of the stand-in camera whose pixels say its number (the value of every pixel is the number mod 251)."""
    return np.full((SIZE[1], SIZE[0], 3), frame % 251, dtype=np.uint8)


def number_of(image: np.ndarray) -> int:
    return int(image[0, 0, 0])


def source_of(n_frames: int = 50, *, lead_s: float = 1.0, settings: dict[str, Any] | None = None,
              **plan: Any) -> tuple[CameraSource, StandInCamera, Clock]:
    """A source on a stand-in camera that has sent for ``lead_s`` seconds when slot 0 is due."""
    clock = Clock(START - lead_s)
    stand_in = StandInCamera(CameraPlan(t0=START - lead_s, **plan), clock, numbered)
    values = {"n_frames": n_frames, "fps": 25.0, "width": SIZE[0], "height": SIZE[1], "start_at": START}
    values.update(settings or {})
    source = CameraSource(ADDRESS, CameraSettings(**values), stand_in, clock=clock, sleep=clock.sleep)
    return source, stand_in, clock


def slots(source: CameraSource) -> tuple[dict[int, Given], dict[int, str], list[Any]]:
    """Run a source to its end; return the given frames, the reason of each lost frame, and all events."""
    given: dict[int, Given] = {}
    lost: dict[int, str] = {}
    events = list(source.events())
    for event in events:
        if isinstance(event, Given):
            assert event.frame not in given and event.frame not in lost
            given[event.frame] = event
        elif isinstance(event, Lost):
            for frame in range(event.start, event.stop):
                assert frame not in given and frame not in lost
                lost[frame] = event.reason
    # Every slot exactly once, in order.
    order = [event.frame if isinstance(event, Given) else event.start for event in events if not isinstance(event, Warmup)]
    assert order == sorted(order)
    assert sorted([*given, *lost]) == list(range(source.settings.n_frames))
    return given, lost, events


LEAD = 25  # camera frames before slot 0 with lead_s = 1.0


# --- The address --------------------------------------------------------------------------------------------


def test_only_rtsp_and_rtsps_with_a_host_are_taken_and_no_error_names_a_part() -> None:
    assert CameraAddress.parse(address(host="h.invalid", path="x")).scheme == "rtsp"
    assert CameraAddress.parse(address("RTSPS", "u", "p", "h.invalid", "322", "x")).scheme == "rtsps"
    wrong = [
        address("http", "quser", "qpass", "qhost", path="qpath"),
        address("rtsp", "quser", "qpass", "qhost", path="qpath").removeprefix("rtsp://"),
        address("file", path="qpath"),
        address("rtsp", path="qpath"),
        address("rtsp", "quser", "qpass", "qhost", "99999", "qpath"),
        address("rtmp", host="qhost", path="qpath"),
    ]
    for text in wrong:
        with pytest.raises(CameraError) as caught:
            CameraAddress.parse(text)
        assert not any(part in str(caught.value) for part in ("quser", "qpass", "qhost", "qpath", "99999"))


def test_the_address_comes_from_its_variable_only() -> None:
    with pytest.raises(CameraError, match=CAMERA_URL_VARIABLE):
        CameraAddress.from_environment({})
    with pytest.raises(CameraError):
        CameraAddress.from_environment({CAMERA_URL_VARIABLE: "  "})
    assert CameraAddress.from_environment({CAMERA_URL_VARIABLE: address(host="h.invalid")}).redacted == REDACTED


def test_the_redacted_form_and_the_redaction_of_a_text() -> None:
    assert str(ADDRESS) == repr(ADDRESS).removeprefix("CameraAddress(").removesuffix(")") == REDACTED
    assert HIDDEN == "<hidden>"
    text = f"[rtsp @ 0x1] {PLANTED}: 401; CAMERA-HOST.INVALID refused cam-user, port 8554, path stream-path\n"
    redacted = ADDRESS.redact(text)
    for part in ("cam-user", "cam-secret", "camera-host", "8554", "stream-path"):
        assert part not in redacted.lower()
    assert redacted.endswith("\n") and HIDDEN in redacted
    # A user name in percent form is hidden in both forms.
    coded = CameraAddress.parse(address("rtsp", "us%40er", "p%3Ass", "h.invalid", path="x"))
    assert "us@er" not in coded.redact("user us@er") and "p:ss" not in coded.redact("p:ss")
    # A part of fewer than 4 characters hides the whole line.
    short = CameraAddress.parse(address("rtsp", "ab", "cd", "h.invalid", "554", "live"))
    assert short.redact("frame 554 of the camera\n") == HIDDEN_LINE + "\n"
    assert short.redact("nothing of it here") == "nothing of it here"


def test_the_options_of_ffmpeg_name_the_transport_and_the_protocols() -> None:
    assert ffmpeg_options("rtsp", "tcp") == "rtsp_transport;tcp|protocol_whitelist;rtsp,rtp,tcp"
    assert ffmpeg_options("rtsps", "tcp") == "rtsp_transport;tcp|protocol_whitelist;rtsp,rtp,tcp,rtsps,tls"
    assert ffmpeg_options("rtsp", "udp") == "rtsp_transport;udp|protocol_whitelist;rtsp,rtp,tcp,udp"


def test_opencv_takes_the_two_timeouts_with_the_ffmpeg_backend(tmp_path: Path) -> None:
    """The call of ``open_ffmpeg`` on a local video file (no network): this build of OpenCV takes the parameters."""
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, SIZE)
    for frame in range(5):
        writer.write(numbered(frame * 40))
    writer.release()
    device = ffmpeg_capture(str(video), 1.0, 1.0)
    try:
        assert device.isOpened()
        assert device.grab() and device.retrieve()[0]
    finally:
        device.release()


def test_what_native_code_writes_goes_through_the_redaction(capfd: pytest.CaptureFixture[str]) -> None:
    with RedactedOutput(ADDRESS):
        os.write(2, f"[rtsp @ 0x1] cannot open {PLANTED}\n".encode())
        os.write(1, b"camera-host.invalid said no\n")
        print("plain line")
    out, err = capfd.readouterr()
    assert "camera-host" not in out + err and "cam-secret" not in out + err
    assert "plain line" in out and HIDDEN in err


def test_doctor_says_whether_opencv_can_open_a_camera() -> None:
    lines = "Video I/O:\n    FFMPEG:                      YES\n      avcodec: YES (61.3.100)\n    GStreamer:  NO\n"
    check = doctor.camera_input_check(lines)
    assert (check.name, check.status) == (doctor.CAMERA_INPUT, "info")
    assert check.detail == "OpenCV video input: FFmpeg YES, GStreamer NO: the command capture can open RTSP"
    assert "cannot open RTSP" in doctor.camera_input_check("    FFMPEG: NO\n").detail
    assert doctor.camera_input_check(cv2.getBuildInformation()).name == doctor.CAMERA_INPUT


# --- The slot rule ------------------------------------------------------------------------------------------


def test_a_camera_on_time_gives_every_frame_in_its_slot() -> None:
    source, stand_in, _ = source_of(50)
    given, lost, events = slots(source)
    assert lost == {} and len(given) == 50
    assert all(number_of(event.image) == (frame + LEAD) % 251 for frame, event in given.items())
    assert all(event.arrived_at - START - frame / 25.0 == pytest.approx(0.02, abs=1e-4) for frame, event in given.items())
    assert isinstance(events[0], Warmup) and source.counts.end == "complete"
    assert source.counts.before_start == LEAD and stand_in.opens == 1


def test_late_frames_and_a_short_stall_are_given_late_in_their_slots() -> None:
    source, _, _ = source_of(50, late={LEAD + 5: 0.3}, stalls=[(START + 1.0, START + 1.8)])
    given, lost, _ = slots(source)
    assert lost == {} and all(number_of(event.image) == (frame + LEAD) % 251 for frame, event in given.items())
    assert given[5].arrived_at == pytest.approx(START + 5.5 / 25 + 0.3, abs=1e-4)
    # The frames of the stall come at its end, in their own slots.
    assert given[25].arrived_at == pytest.approx(START + 1.8, abs=1e-4) and given[44].arrived_at == pytest.approx(START + 1.8, abs=1e-4)
    assert source.counts.lost_connections == 0


def test_a_stall_longer_than_the_read_timeout_ends_the_connection_and_the_camera_comes_back() -> None:
    source, stand_in, _ = source_of(200, stalls=[(START + 2.0, START + 5.0)])
    given, lost, _ = slots(source)
    assert source.counts.lost_connections == 1 and stand_in.opens == 2
    assert min(lost) == 50 and set(lost.values()) <= {"outage", "gap"} and list(lost.values()).count("gap") <= 1
    # After the reconnect the first frame takes the slot of its arrival on the clock (a slot on the boundary of the
    # reconnect can be a gap); the frames are in their slots again and follow on without a gap.
    after = [frame for frame in given if frame > max(lost)]
    assert after == list(range(after[0], 200))
    assert all(number_of(given[frame].image) == (frame + LEAD) % 251 for frame in after)
    assert max(given) == 199


def test_a_lost_connection_writes_the_passing_slots_as_lost_and_reconnects() -> None:
    source, stand_in, _ = source_of(250, settings={"open_timeout_s": 1.0}, outages=[(START + 2.0, START + 5.5)])
    given, lost, events = slots(source)
    assert set(lost.values()) == {"outage"}
    assert source.counts.lost_connections == 1 and source.counts.failed_opens == 1 and stand_in.opens == 3
    spans = [(event.start, event.stop) for event in events if isinstance(event, Lost)]
    assert len(spans) >= 2  # written while the camera was away, not all at once at the end
    assert all(number_of(event.image) == (frame + LEAD) % 251 for frame, event in given.items())


def test_a_camera_that_ends_early_ends_the_run_with_what_it_gave() -> None:
    source, stand_in, _ = source_of(1000, ends_at=START + 4.0)
    given, lost, _ = slots(source)
    assert source.counts.end == "camera ended"
    assert max(given) < 100 and lost[999] == "ended" and set(lost.values()) == {"outage", "ended"}
    assert stand_in.opens == 1 + camera.RECONNECTS


def test_a_camera_that_never_opens_loses_every_frame() -> None:
    source, stand_in, _ = source_of(2000, never_opens=True)
    given, lost, _ = slots(source)
    assert given == {} and source.counts.end == "never opened"
    assert stand_in.opens == source.counts.failed_opens == 1 + camera.RECONNECTS
    # A short run whose time passes before the tries are over still says that the camera never opened.
    source, _, _ = source_of(10, never_opens=True)
    assert slots(source)[0] == {} and source.counts.end == "never opened"


def test_frames_that_the_camera_never_sends_are_a_gap_and_one_of_another_size_is_lost() -> None:
    source, _, _ = source_of(50, skipped=frozenset({LEAD + 10, LEAD + 11}), other_size=frozenset({LEAD + 20}),
                             other_shape=(SIZE[1] // 2, SIZE[0] // 2))
    given, lost, _ = slots(source)
    assert lost == {10: "gap", 11: "gap", 20: "size"}
    assert all(number_of(event.image) == (frame + LEAD) % 251 for frame, event in given.items())


def test_a_first_frame_of_another_size_is_refused() -> None:
    source, _, _ = source_of(50, settings={"width": 1920, "height": 1080})
    with pytest.raises(CameraError, match="64 x 48 pixels, and the run is set for 1920 x 1080"):
        slots(source)


def test_a_faster_camera_gives_every_slot_and_a_slower_one_is_refused_or_has_gaps() -> None:
    source, _, _ = source_of(50, rate=30.0)
    given, lost, _ = slots(source)
    assert lost == {} and source.counts.extra == 10  # 60 frames of the camera in the 2 s of the run
    source, _, _ = source_of(50, rate=50.0)
    # 100 frames of the camera in the 2 s of the run: 50 given, and the run ends at slot 49 before the last 2 are read.
    assert slots(source)[1] == {} and source.counts.extra == 48
    source, _, _ = source_of(50, rate=20.0)
    with pytest.raises(CameraError, match="20 frames per second, below the 25"):
        slots(source)
    # A camera that names no rate: its gaps are lost.
    source, _, _ = source_of(50, rate=20.0, named_rate=0.0)
    _, lost, _ = slots(source)
    assert len(lost) == 10 and set(lost.values()) == {"gap"}


# --- The camera clip folder and its readers -----------------------------------------------------------------


def lines_of(folder: Path) -> list[dict[str, Any]]:
    return [json.loads(text) for text in (folder / camera.LEDGER_NAME).read_text(encoding="utf-8").splitlines()]


def test_the_capture_writes_the_record_the_frames_and_a_line_for_every_frame(tmp_path: Path) -> None:
    source, _, _ = source_of(60, skipped=frozenset({LEAD + 3}))
    folder = tmp_path / "CAM-1"
    report = write_capture(source, folder, "CAM-1", free_bytes=lambda place: PLENTY)
    record = read_camera_record(folder)
    assert record is not None and (record.clip, record.n_frames, record.camera) == ("CAM-1", 60, REDACTED)
    lines = lines_of(folder)
    assert lines[3] == {"lost": [3, 4], "reason": "gap"} and lines[-1]["end"] == "complete"
    assert report.given == 59 and report.lost_total == 1 and lines[-1]["given"] == 59
    images = sorted(path.name for path in (folder / "img1").iterdir())
    assert len(images) == 59 and "000004.jpg" not in images and not any(name.startswith(".") for name in images)
    assert (folder / camera.WARMUP_NAME).is_file()
    with pytest.raises(FileExistsError):
        write_capture(source_of(5)[0], folder, "CAM-1", free_bytes=lambda place: PLENTY)


def test_a_full_disk_is_refused_at_the_start_and_stops_the_run_later(tmp_path: Path) -> None:
    with pytest.raises(CameraError, match="free disk"):
        write_capture(source_of(60)[0], tmp_path / "A", "A", free_bytes=lambda place: 10**6)
    lines = lines_of(tmp_path / "A")
    assert lines == [lines[0], lines[-1]] and lines[0] == {"lost": [0, 60], "reason": "stopped"}
    assert lines[-1]["end"] == "refused"
    free = iter([PLENTY, PLENTY, 10**6])  # the start, frame 0, frame 250
    report = write_capture(source_of(600)[0], tmp_path / "B", "B", free_bytes=lambda place: next(free))
    assert report.end == "disk full" and report.given == 250 and report.lost["disk"] == 350
    assert lines_of(tmp_path / "B")[-2] == {"lost": [250, 600], "reason": "disk"}


def test_a_stop_signal_gives_the_rest_of_the_run_a_line_and_ends_the_ledger(tmp_path: Path) -> None:
    source, stand_in, _ = source_of(100)
    original = stand_in.image

    def image(frame: int) -> np.ndarray:
        if frame == LEAD + 30:
            raise KeyboardInterrupt("Ctrl-C")
        return original(frame)

    stand_in.image = image
    with pytest.raises(KeyboardInterrupt):
        write_capture(source, tmp_path / "S", "S", free_bytes=lambda place: PLENTY)
    lines = lines_of(tmp_path / "S")
    assert lines[-2] == {"lost": [30, 100], "reason": "stopped"} and lines[-1]["end"] == "stopped"


def test_the_two_readers_give_the_same_status_and_a_lost_frame_shows_the_one_before(tmp_path: Path) -> None:
    source, _, _ = source_of(200, outages=[(START + 1.0, START + 2.0)], skipped=frozenset({LEAD}))
    folder = tmp_path / "CAM"
    write_capture(source, folder, "CAM", free_bytes=lambda place: PLENTY)
    record = read_camera_record(folder)
    assert record is not None
    images = CameraFolderSource(folder, record)
    frames = run_tool.CameraFrames(folder)
    for frame in range(200):
        assert images.ledger.status(frame) == frames.status(frame)
    assert images.ledger.status(0) == (False, "gap")
    assert not images.read(0).any()  # nothing given before frame 0: a black frame
    lost = [frame for frame in range(200) if not images.ledger.status(frame)[0]]
    first = lost[1]
    assert number_of(images.read(first)) == number_of(images.read(first - 1)) == (first - 1 + LEAD) % 251
    facts = images.facts()
    assert facts["end"]["end"] == "complete" and facts["lost_spans"][0] == {"start": 0, "stop": 1, "reason": "gap"}
    assert frames.newest_given() == 199 and frames.ended


def test_a_reader_waits_for_a_line_and_gives_up_after_the_wait(tmp_path: Path) -> None:
    source, _, _ = source_of(10)
    folder = tmp_path / "W"
    write_capture(source, folder, "W", free_bytes=lambda place: PLENTY)
    record = read_camera_record(folder)
    assert record is not None
    ledger_path = folder / camera.LEDGER_NAME
    full = ledger_path.read_text(encoding="utf-8")
    first, rest = full.split("\n", 1)
    ledger_path.write_text(first + "\n" + rest[:5], encoding="utf-8")  # one line and a part of the next
    clock = Clock(START)
    ledger = CameraLedger(folder, record, clock=clock, sleep=clock.sleep)
    assert ledger.status(0)[0]
    with pytest.raises(CameraError, match="frame 1 .* for 15 s after its time"):
        ledger.status(1)
    assert clock.now > record.due(1) + camera.READER_WAIT_S
    frames = run_tool.CameraFrames(folder, clock=clock, sleep=clock.sleep)
    with pytest.raises(run_tool.RunnerError, match="wrote nothing for frame 1"):
        frames.status(1)
    ledger_path.write_text(full, encoding="utf-8")
    assert ledger.status(1)[0] and frames.status(9)[0]


# --- The live feed of a tool on a camera run ---------------------------------------------------------------


class Sink:
    def __init__(self) -> None:
        self.lines: list[dict[str, Any]] = []

    def write(self, text: str) -> None:
        self.lines.append(json.loads(text))

    def flush(self) -> None:
        pass


HEAD = {"clip": "C", "n_frames": 40, "video_sha256": None, "job_sha256": "0" * 64, "tool": {"name": "pnlcalib"}}


def camera_folder(tmp_path: Path, **plan: Any) -> Path:
    source, _, _ = source_of(40, **plan)
    folder = tmp_path / "C"
    write_capture(source, folder, "C", free_bytes=lambda place: PLENTY)
    return folder


def test_a_tool_takes_the_frames_of_the_camera_and_skips_the_lost_ones(tmp_path: Path) -> None:
    folder = camera_folder(tmp_path, skipped=frozenset({LEAD + 4, LEAD + 5}))
    frames = run_tool.CameraFrames(folder)
    sink = Sink()
    live = run_tool.LiveFeed([sink], HEAD, camera=frames)
    entries = [{"clip": "C", "frame": frame, "image": f"C/img1/{frame + 1:06d}.jpg"} for frame in range(40)]
    groups = list(live.groups(entries, 3))
    assert [entry["frame"] for group in groups for entry in group] == [f for f in range(40) if f not in (4, 5)]
    assert [len(group) for group in groups][:-1] == [3] * (len(groups) - 1)
    assert sink.lines == [
        {"frame": 4, "arrived_at": frames.due(4), "skipped": True},
        {"frame": 5, "arrived_at": frames.due(5), "skipped": True},
    ]
    assert live.skipped == 2 and frames.lost == 2
    assert live.arrived_at(6) == frames.status(6)[1]


def test_skip_behind_on_a_camera_takes_the_newest_given_frame(tmp_path: Path) -> None:
    folder = camera_folder(tmp_path)
    sink = Sink()
    live = run_tool.LiveFeed([sink], HEAD, camera=run_tool.CameraFrames(folder), skip_behind=True)
    entries = [{"clip": "C", "frame": frame, "image": "x"} for frame in range(40)]
    groups = list(live.groups(entries, 1))
    # The capture is done: the newest given frame is the last one, so the tool skips to it.
    assert [group[0]["frame"] for group in groups] == [39] and live.skipped == 39


def test_the_camera_options_of_the_tool_script(tmp_path: Path) -> None:
    with pytest.raises(run_tool.RunnerError, match="--camera takes no --pace"):
        run_tool.LiveFeed([Sink()], HEAD, camera=run_tool.CameraFrames(camera_folder(tmp_path)), pace=25.0)
    job = {"clip": "D", "frames": [{"clip": "D", "frame": 0, "image": "C/img1/000001.jpg"}]}
    with pytest.raises(run_tool.RunnerError, match="the camera run is C with 40 frames"):
        run_tool.camera_frames(job, tmp_path)
    with pytest.raises(run_tool.RunnerError, match="no camera.json"):
        run_tool.camera_frames({"clip": "C", "frames": [{"image": "elsewhere/img1/000001.jpg"}]}, tmp_path)


# --- The command line --------------------------------------------------------------------------------------


def capture_args(out: Path, **changes: Any) -> argparse.Namespace:
    values = {"out": out, "start_at": START, "frames": 50, "seconds": None, "fps": 25.0, "size": SIZE,
              "transport": "tcp", "open_timeout": 5.0, "read_timeout": 2.0, "reconnects": 5}
    values.update(changes)
    return argparse.Namespace(**values)


def capture_with(monkeypatch: pytest.MonkeyPatch, argv: list[str], **plan: Any) -> int:
    """Run ``cli.main(["capture", ...])`` with the stand-in camera behind the command."""
    clock = Clock(START - 1.0)
    stand_in = StandInCamera(CameraPlan(t0=START - 1.0, **plan), clock, numbered)

    def handler(args: argparse.Namespace) -> int:
        address = CameraAddress.from_environment()
        os.environ.pop(CAMERA_URL_VARIABLE, None)
        return cli.run_capture(args, address, open_device=stand_in, clock=clock, sleep=clock.sleep,
                               free_bytes=lambda place: PLENTY)

    monkeypatch.setattr(cli, "_cmd_capture", handler)
    monkeypatch.setenv(CAMERA_URL_VARIABLE, PLANTED)
    return cli.main(["capture", "--start-at", repr(START), "--size", "64x48", *argv])


def test_the_exit_codes_of_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                   capfd: pytest.CaptureFixture[str]) -> None:
    assert capture_with(monkeypatch, ["--out", str(tmp_path / "ok"), "--frames", "50"]) == 0
    assert os.environ.get(CAMERA_URL_VARIABLE) is None  # removed once it was read
    assert capture_with(monkeypatch, ["--out", str(tmp_path / "gap"), "--seconds", "2"],
                        skipped=frozenset({30})) == 1
    assert capture_with(monkeypatch, ["--out", str(tmp_path / "never"), "--frames", "50"], never_opens=True) == 2
    assert capture_with(monkeypatch, ["--out", str(tmp_path / "ok"), "--frames", "50"]) == 2  # the folder is there
    out, err = capfd.readouterr()
    assert "50 of 50 frames given" in out and "camera never opened" in err and "is already there" in err
    assert "camera-host" not in out + err and "cam-secret" not in out + err


def test_capture_refuses_what_does_not_fit(tmp_path: Path) -> None:
    with pytest.raises(cli.CliError, match="--frames N or --seconds S"):
        cli.capture_settings(capture_args(tmp_path / "x", frames=None), START - 10)
    with pytest.raises(cli.CliError, match="--frames N or --seconds S"):
        cli.capture_settings(capture_args(tmp_path / "x", seconds=2.0), START - 10)
    with pytest.raises(cli.CliError, match="in the past"):
        cli.capture_settings(capture_args(tmp_path / "x"), START + 1)
    assert cli.capture_settings(capture_args(tmp_path / "x", frames=None, seconds=3.0), START - 10).n_frames == 75
    with pytest.raises(cli.CliError, match="inside the work tree"):
        cli.run_capture(capture_args(REPO / "outputs" / "cam"), ADDRESS, clock=lambda: START - 10)
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["capture", "--out", "x", "--start-at", "1", "--frames", "5", "--size", "1920"])
    assert cli.main(["capture", "--out", str(tmp_path / "y"), "--start-at", repr(START), "--frames", "5"]) == 2


def test_only_stream_and_clip_job_read_a_camera_folder(tmp_path: Path) -> None:
    root = tmp_path / "data"
    source, _, _ = source_of(40)
    folder = root / "live" / "CAM-2"
    write_capture(source, folder, "CAM-2", free_bytes=lambda place: PLENTY)
    job = tmp_path / "job.json"
    assert cli.main(["clip-job", "--tool", "pnlcalib", "--clip", str(folder), "--root", str(root),
                     "--out", str(job)]) == 0
    written = json.loads(job.read_text(encoding="utf-8"))
    assert written["clip"] == "CAM-2" and written["video_sha256"] is None
    assert written["images"][:2] == ["live/CAM-2/img1/000001.jpg", "live/CAM-2/img1/000002.jpg"]
    assert len(written["images"]) == 40
    with pytest.raises(cli.CliError, match="only stream and clip-job read it"):
        cli.open_clip(folder, fps=None, labels=None)
    with pytest.raises(cli.CliError, match="a camera run has its own frame rate"):
        cli.open_clip(folder, fps=25.0, labels=None, camera=True)
    assert isinstance(cli.open_clip(folder, fps=None, labels=None, camera=True), CameraFolderSource)


# --- The chain of a camera run: capture, two tools, merge-feeds, stream --------------------------------------


def test_a_camera_on_time_gives_the_result_of_the_recorded_clip_of_its_frames(tmp_path: Path) -> None:
    """C1 on one seed and 40 frames (the bench runs 20 seeds of 250 frames)."""
    rows = camerabench.equivalence([3000], 40, tmp_path)
    assert rows[0]["capture"] == 0 and set(rows[0]["codes"].values()) == {0}
    assert rows[0]["vs_recorded"] == {"predictions": True, "video": True, "result": True}


def test_a_camera_with_faults_ends_by_itself_and_every_frame_is_accounted_for(tmp_path: Path) -> None:
    """C2 on one seed, with the capture as a process at the same time as the tools, through named pipes."""
    row = camerabench.loss([3100], 500, tmp_path, live=True)[0]
    assert row["accounted"] and row["stream_frames"] == 500
    assert row["codes"] == {"capture": 1, "rfdetr": 0, "pnlcalib": 0, "merge-feeds": 1, "stream": 0}
    given, lost = row["ledger"]["given"], row["ledger"]["lost"]
    assert all(list(tool) == [given, lost, lost] for tool in row["tools"].values())
    assert row["stream_camera"]["given"] == given and sum(row["stream_camera"]["lost"].values()) == lost


def test_no_part_of_the_address_is_printed_or_written(tmp_path: Path) -> None:
    """C3: the planted address on the success path and on every failure path that the capture can reach."""
    rows = camerabench.leak(tmp_path, 30)
    assert [row["case"] for row in rows] == [
        "success", "never-opens", "wrong-scheme", "no-scheme", "lost-connection", "wrong-size", "stop-signal"]
    assert {row["case"]: row["codes"]["capture"] for row in rows} == {
        "success": 0, "never-opens": 2, "wrong-scheme": 2, "no-scheme": 2, "lost-connection": 1, "wrong-size": 2,
        "stop-signal": 2}
    for row in rows:
        assert row["counts"] == dict.fromkeys(camerabench.PLANTED_PARTS, 0), row
