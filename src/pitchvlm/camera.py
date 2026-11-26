"""A match camera as the clip of a live run: one capture step reads it, every other part reads the frames it wrote.

The capture (``cli capture``) is the only part that talks to the camera. It reads the RTSP stream
once (``CameraSource``, through ``cv2.VideoCapture`` with the FFmpeg backend), and writes a camera
clip folder below the data folder (``write_capture``):

- ``camera.json``: the record of the run (``CameraRecord``), written before the camera is opened:
  the clip id, the number of frames, the frame rate, the image size, the start, the settings.
- ``img1/000001.jpg`` ...: frame ``i`` as ``frame_file_name(i)``, written under a hidden name and
  renamed, so that a reader never sees a part of a file.
- ``warmup.jpg``: the first frame that the camera sent, for the warm-up call of a tool.
- ``frames.jsonl``: the ledger, one flushed line per frame in frame order, written after its image:
  ``{"frame": i, "arrived_at": t}`` for a frame that was given, ``{"lost": [a, b], "reason": r}``
  for the frames ``a`` to ``b - 1`` that the camera did not give, and one end line
  ``{"end": reason, "given": ..., "lost": {...}, ...}``. Every frame of the run has exactly one
  line (given or lost) before the end line.

The tools (``tools/run_tool.py --camera``) and ``stream --clip <camera folder>`` read the same
files by frame index (``CameraLedger``, ``CameraFolderSource``): the frame index is the only key
between the parts, and every part sees the same pixels of a frame.

The slot rule (``CameraSource``). The run has ``n_frames`` slots at ``fps``; slot ``i`` is due at
``start_at + i / fps``. A frame before ``start_at`` is not used. The first frame of a connection
takes the slot of its arrival on the clock (at least the next free slot); each later frame of
that connection takes the slot of its time stamp in the stream (``CAP_PROP_POS_MSEC``) counted
from the first one, or the next slot when the time stamp does not grow. A frame whose slot is
already taken is not used (``extra``: a camera faster than the run); slots that no frame took are
lost (``gap``). A read that fails or times out ends the connection: the capture waits
``retry_wait_s``, opens the camera again up to ``reconnects`` times, and meanwhile writes the
slots whose time has passed as lost (``outage``), so that the readers go on. When no try opens
the camera again, the rest of the run is lost (``ended``) and the capture ends; a stop signal
ends it the same way (``stopped``).

The address of the camera is a secret (``CameraAddress``): it comes only from the variable
``PITCHVLM_CAMERA_URL``, and nothing that is printed or written holds a part of it.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import threading
import time
import urllib.parse
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import cv2
import numpy as np
from pydantic import Field, FiniteFloat

from pitchvlm.schemas import ClipInfo, Record

#: The variable that holds the address of the camera. The address is read from nowhere else.
CAMERA_URL_VARIABLE = "PITCHVLM_CAMERA_URL"
#: The schemes of a camera address that the capture accepts.
CAMERA_SCHEMES = ("rtsp", "rtsps")
#: What stands for a hidden part of the address, in the one redacted form and in a redacted text.
HIDDEN = "<hidden>"
#: A text of OpenCV or FFmpeg that still holds a short part of the address is replaced by this line.
HIDDEN_LINE = "[a line of OpenCV or FFmpeg was hidden: it may hold a part of the camera address]"
#: Parts of the address shorter than this are not replaced in a text: a text that holds one is hidden whole.
_SHORT_PART = 4

CAMERA_FORMAT = "pitchvlm-camera-1"
CAMERA_RECORD_NAME = "camera.json"
LEDGER_NAME = "frames.jsonl"
FRAME_FOLDER = "img1"
#: The frames are JPEG files of this quality: 2.7 ms to encode a 1080p frame and about 0.3 MB on the build machine
#: (synthetic frames; docs/DECISIONS.md, entry "A camera as the clip of a live run").
IMAGE_SUFFIX = ".jpg"
JPEG_QUALITY = 95
WARMUP_NAME = f"warmup{IMAGE_SUFFIX}"

#: Defaults of the capture: seconds, and the tries to open the camera again after one lost connection.
OPEN_TIMEOUT_S = 5.0
READ_TIMEOUT_S = 2.0
RETRY_WAIT_S = 1.0
RECONNECTS = 5
#: A camera whose frame rate is more than this share below the rate of the run is refused at the start.
RATE_TOLERANCE = 0.01
#: A reader gives up on a frame that has no line this long after its time: above the longest time without a
#: line of a working capture (a read timeout, then an open timeout and a retry wait: 2 + 5 + 1 s by default).
READER_WAIT_S = 15.0
#: How often a reader looks at the ledger while it waits, seconds.
READER_POLL_S = 0.005
#: The capture stops before the free disk of the camera folder falls below this, bytes.
DISK_FLOOR_BYTES = 2 * 10**9
#: The free disk is read every this many frames during the run.
DISK_CHECK_FRAMES = 250
#: At the start the free disk must hold the run at this many times the size of the first frame, plus the floor.
DISK_MARGIN = 1.5

#: End reasons of a capture: all frames had their turn; the camera could not be opened again; it never opened; it was
#: refused (size, frame rate, disk); a stop signal or an error; the disk became full.
EndReason = Literal["complete", "camera ended", "never opened", "refused", "stopped", "disk full"]
LostReason = Literal["gap", "outage", "size", "ended", "stopped", "disk"]
LOST_REASONS: tuple[str, ...] = ("gap", "outage", "size", "ended", "stopped", "disk")


class CameraError(Exception):
    """The camera cannot be used. The message says why and what to do, and holds no part of the address."""


# --- The address ------------------------------------------------------------------------------------------


@dataclass(frozen=True, repr=False)
class CameraAddress:
    """The address of the camera: a secret. Its text and its parts never go into an output.

    ``str`` and ``repr`` give the one redacted form (``redacted``): the scheme and ``<hidden>``.
    ``redact`` replaces every part of the address in a text. Only ``text`` gives the address
    itself, for the call that opens the camera.
    """

    _text: str = field(repr=False)
    scheme: str
    _parts: tuple[str, ...] = field(repr=False)

    @classmethod
    def parse(cls, text: str) -> CameraAddress:
        """Read an address. Only ``rtsp://`` and ``rtsps://`` with a host are accepted; the error names no part."""
        text = text.strip()
        try:
            split = urllib.parse.urlsplit(text)
            host, port = split.hostname, split.port
        except ValueError:
            raise CameraError(
                f"the camera address in {CAMERA_URL_VARIABLE} cannot be read: write it as "
                "rtsp://<user>:<password>@<host>:<port>/<path>"
            ) from None
        scheme = split.scheme.lower()
        # The scheme that was given is not named: text without "://" can put the user name where the scheme is.
        if scheme not in CAMERA_SCHEMES or not text.lower().startswith(f"{scheme}://"):
            raise CameraError(f"the camera address in {CAMERA_URL_VARIABLE} must start with rtsp:// or rtsps://")
        if not host:
            raise CameraError(f"the camera address in {CAMERA_URL_VARIABLE} names no host")
        netloc = split.netloc
        userinfo = netloc.rpartition("@")[0] if "@" in netloc else ""
        user_raw, _, password_raw = userinfo.partition(":")
        parts = {
            text,
            netloc,
            userinfo,
            user_raw,
            password_raw,
            urllib.parse.unquote(user_raw),
            urllib.parse.unquote(password_raw),
            host,
            netloc.rpartition("@")[2],
            "" if port is None else str(port),
            split.path if split.path != "/" else "",
            split.path.strip("/"),
            split.query,
            split.fragment,
            text[len(scheme) + 3 :],
        }
        ordered = tuple(sorted((part for part in parts if part), key=len, reverse=True))
        return cls(text, scheme, ordered)

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> CameraAddress:
        """Read the address from ``PITCHVLM_CAMERA_URL``; a missing or empty variable raises ``CameraError``."""
        values = os.environ if environment is None else environment
        text = values.get(CAMERA_URL_VARIABLE, "")
        if not text.strip():
            raise CameraError(
                f"set the address of the camera in the variable {CAMERA_URL_VARIABLE} (docs/RUNBOOK.md, section 17): "
                "it is never given on the command line or in a file"
            )
        return cls.parse(text)

    @property
    def redacted(self) -> str:
        """The one form of the address that may be printed or written: the scheme and ``<hidden>``."""
        return self.scheme + "://" + HIDDEN

    def text(self) -> str:
        """The address itself: only for the call that opens the camera."""
        return self._text

    def redact(self, text: str) -> str:
        """Return ``text`` with every part of the address replaced by ``<hidden>``, letter case ignored.

        A text that still holds a part shorter than 4 characters after that (a short user name,
        password or port) is replaced whole by ``HIDDEN_LINE``: such a part cannot be replaced
        alone without hiding other words.
        """
        for part in self._parts:
            if len(part) >= _SHORT_PART:
                text = re.sub(re.escape(part), HIDDEN, text, flags=re.IGNORECASE)
        lowered = text.lower()
        if any(part.lower() in lowered for part in self._parts if len(part) < _SHORT_PART):
            return HIDDEN_LINE + ("\n" if text.endswith("\n") else "")
        return text

    def __str__(self) -> str:
        return self.redacted

    def __repr__(self) -> str:
        return f"CameraAddress({self.redacted})"


def ffmpeg_options(scheme: str, transport: Literal["tcp", "udp"]) -> str:
    """Return the FFmpeg options of the capture, in the form of ``OPENCV_FFMPEG_CAPTURE_OPTIONS``.

    The transport of RTSP (TCP by default: no frame is lost on the way) and the protocols that
    FFmpeg may open for this address: RTSP and RTP over TCP, TLS for ``rtsps``, UDP only with
    the UDP transport. A protocol that is not in the list (a file, HTTP, a pipe) is refused by
    FFmpeg, also when the camera answers with a redirect to one.
    """
    protocols = ["rtsp", "rtp", "tcp"]
    if scheme == "rtsps":
        protocols += ["rtsps", "tls"]
    if transport == "udp":
        protocols += ["udp"]
    return f"rtsp_transport;{transport}|protocol_whitelist;{','.join(protocols)}"


class RedactedOutput:
    """While open, everything that the process writes to its descriptors 1 and 2 goes through ``redact``.

    OpenCV and FFmpeg write their messages to the standard error stream from native code, and a
    message can repeat the address. So the descriptors 1 and 2 point at pipes while the capture
    runs; one thread per pipe reads it line by line and writes the redacted line to the stream
    that the descriptor had before. Python's own output goes the same way.
    """

    def __init__(self, address: CameraAddress) -> None:
        self._address = address
        self._saved: dict[int, int] = {}
        self._threads: list[threading.Thread] = []

    def __enter__(self) -> RedactedOutput:
        sys.stdout.flush()
        sys.stderr.flush()
        for descriptor in (1, 2):
            self._saved[descriptor] = os.dup(descriptor)
            reader, writer = os.pipe()
            os.dup2(writer, descriptor)
            os.close(writer)
            thread = threading.Thread(target=self._pump, args=(reader, self._saved[descriptor]), daemon=True)
            thread.start()
            self._threads.append(thread)
        return self

    def _pump(self, reader: int, target: int) -> None:
        with os.fdopen(reader, "rb") as stream:
            for line in stream:
                text = self._address.redact(line.decode("utf-8", errors="replace"))
                data = text.encode("utf-8")
                while data:
                    data = data[os.write(target, data) :]

    def __exit__(self, *exc: object) -> None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            for descriptor, saved in self._saved.items():
                os.dup2(saved, descriptor)  # the last writer of the pipe goes: its thread reads the end
            for thread in self._threads:
                thread.join(timeout=5.0)
            for saved in self._saved.values():
                os.close(saved)


# --- The camera --------------------------------------------------------------------------------------------


class CaptureDevice(Protocol):
    """The part of ``cv2.VideoCapture`` that the source uses. The tests give a stand-in of it."""

    def isOpened(self) -> bool: ...  # noqa: N802 (the name of OpenCV)

    def grab(self) -> bool: ...

    def retrieve(self) -> tuple[bool, np.ndarray | None]: ...

    def get(self, prop: int) -> float: ...

    def release(self) -> None: ...


@dataclass(frozen=True)
class CameraSettings:
    """What the capture needs to know before it opens the camera."""

    n_frames: int
    fps: float
    width: int
    height: int
    #: Seconds since the epoch: slot 0 is due then. The same start as the one of the tools and of stream.
    start_at: float
    transport: Literal["tcp", "udp"] = "tcp"
    open_timeout_s: float = OPEN_TIMEOUT_S
    read_timeout_s: float = READ_TIMEOUT_S
    retry_wait_s: float = RETRY_WAIT_S
    reconnects: int = RECONNECTS

    def __post_init__(self) -> None:
        if self.n_frames < 1:
            raise ValueError("a camera run needs at least one frame")
        if not 0 < self.fps < math.inf:
            raise ValueError(f"the frame rate of the run must be above 0, got {self.fps}")
        if self.width < 1 or self.height < 1:
            raise ValueError("the image size of the run must be above 0")
        for name in ("open_timeout_s", "read_timeout_s"):
            if not 0 < getattr(self, name) < math.inf:
                raise ValueError(f"{name} must be above 0")
        if not 0 <= self.retry_wait_s < math.inf or self.reconnects < 0:
            raise ValueError("the retry wait and the number of reconnects must be 0 or more")

    def due(self, frame: int) -> float:
        """Return the time at which a frame is due, seconds since the epoch."""
        return self.start_at + frame / self.fps

    def slot_of(self, moment: float) -> int:
        """Return the slot whose time span holds a moment (negative before the start).

        A moment within 1/10,000 of a frame (4 microseconds at 25 frames per second) below the start of a slot counts
        to that slot: seconds since the epoch are exact to about 0.2 microseconds, so a moment on a boundary would
        otherwise fall into the slot before it.
        """
        return math.floor((moment - self.start_at) * self.fps + 1e-4)


DeviceFactory = Callable[[CameraAddress, CameraSettings], CaptureDevice]


def open_ffmpeg(address: CameraAddress, settings: CameraSettings) -> CaptureDevice:
    """Open the camera with OpenCV and FFmpeg: the transport and the protocol list, the open and read timeouts."""
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_options(address.scheme, settings.transport)
    return ffmpeg_capture(address.text(), settings.open_timeout_s, settings.read_timeout_s)


def ffmpeg_capture(target: str, open_timeout_s: float, read_timeout_s: float) -> CaptureDevice:
    """Return ``cv2.VideoCapture`` of a target with the FFmpeg backend and the two timeouts, in milliseconds."""
    parameters = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
        round(open_timeout_s * 1000),
        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
        round(read_timeout_s * 1000),
    ]
    return cv2.VideoCapture(target, cv2.CAP_FFMPEG, parameters)


@dataclass(frozen=True)
class Warmup:
    """The first frame that the camera sent (also before the start): the image of the warm-up call of a tool."""

    image: np.ndarray


@dataclass(frozen=True)
class Given:
    """A frame of the run that the camera gave: its slot, its arrival (seconds since the epoch) and its pixels."""

    frame: int
    arrived_at: float
    image: np.ndarray


@dataclass(frozen=True)
class Lost:
    """The frames ``start`` to ``stop - 1`` that the camera did not give, and why."""

    start: int
    stop: int
    reason: LostReason


CameraEvent = Warmup | Given | Lost


@dataclass
class CameraCounts:
    """What the source saw besides the frames of the run."""

    #: Frames before the start, and frames whose slot was taken already (a camera faster than the run).
    before_start: int = 0
    extra: int = 0
    #: Connections that were lost, and tries to open the camera that failed.
    lost_connections: int = 0
    failed_opens: int = 0
    #: The frame rate and size that the camera named at the first frame (0 when it named none).
    camera_fps: float = 0.0
    camera_size: tuple[int, int] = (0, 0)
    #: The end of the source, once it ended.
    end: EndReason | None = None


class CameraSource:
    """The frames of a camera, by slot (module docstring): ``events`` yields every slot of the run once.

    ``open_device`` opens the camera (``open_ffmpeg``, or a stand-in in the tests); ``clock``
    and ``sleep`` are those of the system unless a test gives its own. A first frame of
    another size than the run raises ``CameraError``, as does a camera that names a frame rate
    more than 1% below the rate of the run. Every error of OpenCV is turned into
    ``CameraError`` with a message that holds no part of the address.
    """

    def __init__(
        self,
        address: CameraAddress,
        settings: CameraSettings,
        open_device: DeviceFactory = open_ffmpeg,
        *,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.address = address
        self.settings = settings
        self._open_device = open_device
        self._clock = clock
        self._sleep = sleep
        self.counts = CameraCounts()

    def _open(self) -> CaptureDevice | None:
        """Open the camera once; return None when it does not open."""
        try:
            device = self._open_device(self.address, self.settings)
            if device.isOpened():
                return device
            device.release()
        except Exception:  # an error of OpenCV can repeat the address: it is not passed on
            pass
        self.counts.failed_opens += 1
        return None

    @staticmethod
    def _read(device: CaptureDevice) -> tuple[np.ndarray | None, float]:
        """Read one frame and its time stamp in milliseconds; None when the read failed or timed out."""
        try:
            if not device.grab():
                return None, math.nan
            ok, image = device.retrieve()
            stamp = float(device.get(cv2.CAP_PROP_POS_MSEC))
        except Exception:
            return None, math.nan
        if not ok or image is None:
            return None, math.nan
        return image, stamp

    @staticmethod
    def _release(device: CaptureDevice) -> None:
        try:
            device.release()
        except Exception:
            pass

    def _check_first(self, device: CaptureDevice, image: np.ndarray) -> None:
        """Check the size and the frame rate of the first frame of the camera; raise ``CameraError`` when they do not fit."""
        settings = self.settings
        height, width = image.shape[:2]
        try:
            rate = float(device.get(cv2.CAP_PROP_FPS))
        except Exception:
            rate = 0.0
        self.counts.camera_fps = rate if math.isfinite(rate) else 0.0
        self.counts.camera_size = (width, height)
        if (width, height) != (settings.width, settings.height) or image.ndim != 3 or image.shape[2] != 3:
            raise CameraError(
                f"the camera sends frames of {width} x {height} pixels, and the run is set for "
                f"{settings.width} x {settings.height}: set the camera to that size, or give the capture --size "
                f"{width}x{height} (docs/RUNBOOK.md, section 17)"
            )
        if 0 < self.counts.camera_fps < settings.fps * (1 - RATE_TOLERANCE):
            raise CameraError(
                f"the camera sends {self.counts.camera_fps:g} frames per second, below the {settings.fps:g} of the run: "
                f"every second about {settings.fps - self.counts.camera_fps:.0f} frames would be lost. Set the camera to "
                f"{settings.fps:g} frames per second, or give the capture --fps {self.counts.camera_fps:g}"
            )

    def events(self) -> Iterator[CameraEvent]:
        """Yield the warm-up frame, then every slot of the run in order, each one once: given or lost.

        The source ends after the last slot, or after the camera could not be opened again
        (the rest is lost as ``ended``; ``counts.end`` says which). A stop signal
        (``KeyboardInterrupt``) and ``CameraError`` go through to the caller, which writes
        the rest of the run as lost.
        """
        settings = self.settings
        n_frames = settings.n_frames
        next_slot = 0
        device: CaptureDevice | None = None
        warm = False
        tries = 0  # failed opens since the last connection
        anchor: tuple[int, float] | None = None  # the slot and time stamp of the first frame of the connection
        last_stamp = math.nan
        last_slot = -1
        try:
            while next_slot < n_frames:
                if device is None:
                    passed = min(n_frames, settings.slot_of(self._clock()))
                    if passed > next_slot:
                        yield Lost(next_slot, passed, "outage")
                        next_slot = passed
                        continue
                    if tries > settings.reconnects:
                        self.counts.end = "camera ended" if warm else "never opened"
                        yield Lost(next_slot, n_frames, "ended")
                        return
                    device = self._open()
                    if device is None:
                        tries += 1
                        if tries <= settings.reconnects:
                            self._sleep(settings.retry_wait_s)
                        continue
                    anchor, last_stamp, last_slot = None, math.nan, -1
                image, stamp = self._read(device)
                now = self._clock()
                if image is None:
                    self._release(device)
                    device = None
                    self.counts.lost_connections += 1
                    tries = 1  # the lost connection counts as the first failed try
                    self._sleep(settings.retry_wait_s)
                    continue
                tries = 0
                if not warm:
                    self._check_first(device, image)
                    warm = True
                    yield Warmup(image)
                if now < settings.start_at:
                    self.counts.before_start += 1
                    continue
                if anchor is None:
                    slot = max(next_slot, settings.slot_of(now))
                    anchor = (slot, stamp)
                elif math.isfinite(stamp) and math.isfinite(last_stamp) and stamp > last_stamp:
                    slot = anchor[0] + math.floor((stamp - anchor[1]) * settings.fps / 1000.0 + 0.5)
                else:
                    slot = last_slot + 1
                last_stamp, last_slot = stamp, slot
                if slot < next_slot:
                    self.counts.extra += 1
                    continue
                if slot > next_slot:
                    yield Lost(next_slot, min(slot, n_frames), "gap")
                    next_slot = min(slot, n_frames)
                if slot >= n_frames:
                    break
                if image.shape[:2] != (settings.height, settings.width):
                    yield Lost(slot, slot + 1, "size")
                else:
                    yield Given(slot, now, image)
                next_slot = slot + 1
            # All slots had their turn; a camera that never sent a frame never opened, whatever the time did.
            self.counts.end = "complete" if warm else "never opened"
        finally:
            if device is not None:
                self._release(device)


# --- The camera clip folder --------------------------------------------------------------------------------


class CameraRecord(Record):
    """``camera.json`` of a camera clip folder: what every reader of the frames must know before the first frame.

    ``camera`` is the redacted form of the address (``CameraAddress.redacted``): the scheme only.
    """

    format: Literal["pitchvlm-camera-1"]
    clip: str = Field(min_length=1)
    n_frames: int = Field(gt=0)
    fps: FiniteFloat = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    start_at: FiniteFloat
    image: Literal[".jpg"]
    jpeg_quality: int = Field(ge=1, le=100)
    camera: str
    transport: Literal["tcp", "udp"]
    open_timeout_s: FiniteFloat = Field(gt=0)
    read_timeout_s: FiniteFloat = Field(gt=0)
    retry_wait_s: FiniteFloat = Field(ge=0)
    reconnects: int = Field(ge=0)

    def due(self, frame: int) -> float:
        """Return the time at which a frame is due, seconds since the epoch."""
        return self.start_at + frame / self.fps


def frame_file_name(frame: int) -> str:
    """Return the name of the image of a frame: frame 0 is ``000001.jpg``, as in the dataset."""
    return f"{frame + 1:06d}{IMAGE_SUFFIX}"


def read_camera_record(folder: Path) -> CameraRecord | None:
    """Return the record of a camera clip folder, or None when the folder holds none. A wrong record raises ``ValueError``."""
    path = Path(folder) / CAMERA_RECORD_NAME
    if not path.is_file():
        return None
    try:
        return CameraRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


@dataclass
class CaptureReport:
    """What a capture wrote: the frames given and lost, the end, the counts of the source and the disk use."""

    n_frames: int
    given: int = 0
    lost: dict[str, int] = field(default_factory=lambda: dict.fromkeys(LOST_REASONS, 0))
    end: EndReason | None = None
    bytes_written: int = 0
    counts: CameraCounts = field(default_factory=CameraCounts)

    @property
    def lost_total(self) -> int:
        return sum(self.lost.values())

    def end_line(self) -> dict[str, Any]:
        """Return the end line of the ledger."""
        return {
            "end": self.end,
            "frames": self.n_frames,
            "given": self.given,
            "lost": dict(self.lost),
            "extra": self.counts.extra,
            "before_start": self.counts.before_start,
            "lost_connections": self.counts.lost_connections,
            "failed_opens": self.counts.failed_opens,
            "camera_fps": self.counts.camera_fps,
            "camera_size": list(self.counts.camera_size),
            "bytes_written": self.bytes_written,
        }

    def text(self) -> str:
        """Return the report line of the capture."""
        lost = ", ".join(f"{count} {reason}" for reason, count in self.lost.items() if count) or "none"
        return (
            f"capture ({self.end}): {self.given} of {self.n_frames} frames given, {self.lost_total} lost ({lost}); "
            f"{self.counts.extra} extra frames not used, {self.counts.before_start} before the start; "
            f"{self.counts.lost_connections} lost connections, {self.counts.failed_opens} failed opens; "
            f"{self.bytes_written / 1e9:.2f} GB of frames"
        )


class _Ledger:
    """The ledger file of a capture: one flushed JSON line at a time."""

    def __init__(self, path: Path) -> None:
        self._handle = path.open("x", encoding="utf-8")

    def line(self, data: Mapping[str, Any]) -> None:
        self._handle.write(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def _encode(image: np.ndarray) -> bytes:
    ok, data = cv2.imencode(IMAGE_SUFFIX, image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise OSError("a frame of the camera cannot be encoded as JPEG")
    return data.tobytes()


def _write_file(path: Path, data: bytes) -> None:
    """Write a file under a hidden name and rename it, so that a reader sees the whole file or none."""
    partial = path.with_name(f".{path.name}.part")
    partial.write_bytes(data)
    os.replace(partial, path)


def write_capture(
    source: CameraSource,
    folder: Path,
    clip_id: str,
    *,
    free_bytes: Callable[[Path], int] = lambda place: shutil.disk_usage(place).free,
    on_frame: Callable[[CaptureReport], None] | None = None,
    report: CaptureReport | None = None,
) -> CaptureReport:
    """Run a capture into a new camera clip folder and return its report (module docstring).

    The record is written first, before the camera is opened, so that the jobs of the tools
    and the readers can be made at once. ``FileExistsError`` is raised for a folder that is
    already there: a run never writes into the frames of another run. The free disk must hold
    the run at 1.5 times the size of the first frame plus 2 GB, else ``CameraError`` before the
    first frame of the run; during the run the capture stops (``disk full``) before the free disk
    falls below 2 GB. Every frame gets its ledger line before the end line, also when the
    capture stops (a stop signal, ``CameraError``, an error of the disk): the frames that were
    not given then are lost as ``stopped`` (``disk`` for a full disk), and the error goes on.
    ``on_frame`` is called after each line of a frame (the bench reads the progress there).
    ``report`` is filled when given, so that the caller has the counts also when an error goes on.
    """
    folder = Path(folder)
    settings = source.settings
    folder.mkdir(parents=True, exist_ok=False)
    record = CameraRecord(
        format=CAMERA_FORMAT,
        clip=clip_id,
        n_frames=settings.n_frames,
        fps=settings.fps,
        width=settings.width,
        height=settings.height,
        start_at=settings.start_at,
        image=IMAGE_SUFFIX,
        jpeg_quality=JPEG_QUALITY,
        camera=source.address.redacted,
        transport=settings.transport,
        open_timeout_s=settings.open_timeout_s,
        read_timeout_s=settings.read_timeout_s,
        retry_wait_s=settings.retry_wait_s,
        reconnects=settings.reconnects,
    )
    _write_file(folder / CAMERA_RECORD_NAME, (record.model_dump_json() + "\n").encode("utf-8"))
    images = folder / FRAME_FOLDER
    images.mkdir()
    ledger = _Ledger(folder / LEDGER_NAME)
    if report is None:
        report = CaptureReport(n_frames=settings.n_frames)
    report.counts = source.counts
    written = 0  # the frames that have their line
    try:
        for event in source.events():
            if isinstance(event, Warmup):
                data = _encode(event.image)
                needed = settings.n_frames * len(data) * DISK_MARGIN + DISK_FLOOR_BYTES
                free = free_bytes(folder)
                if free < needed:
                    raise CameraError(
                        f"the run needs about {needed / 1e9:.1f} GB of free disk for {settings.n_frames} frames of about "
                        f"{len(data) / 1e6:.2f} MB (1.5 times, plus 2 GB), and the camera folder has {free / 1e9:.1f} GB: "
                        "give fewer frames, or make room"
                    )
                _write_file(folder / WARMUP_NAME, data)
                continue
            if isinstance(event, Lost):
                ledger.line({"lost": [event.start, event.stop], "reason": event.reason})
                report.lost[event.reason] += event.stop - event.start
                written = event.stop
            else:
                if event.frame % DISK_CHECK_FRAMES == 0 and free_bytes(folder) < DISK_FLOOR_BYTES:
                    report.end = "disk full"
                    break
                data = _encode(event.image)
                _write_file(images / frame_file_name(event.frame), data)
                report.bytes_written += len(data)
                ledger.line({"frame": event.frame, "arrived_at": round(event.arrived_at, 6)})
                report.given += 1
                written = event.frame + 1
            if on_frame is not None:
                on_frame(report)
    except BaseException as error:
        if written < settings.n_frames:
            ledger.line({"lost": [written, settings.n_frames], "reason": "stopped"})
            report.lost["stopped"] += settings.n_frames - written
        report.end = "refused" if isinstance(error, CameraError) else "stopped"
        ledger.line(report.end_line())
        ledger.close()
        raise
    if report.end == "disk full":
        ledger.line({"lost": [written, settings.n_frames], "reason": "disk"})
        report.lost["disk"] += settings.n_frames - written
    else:
        report.end = source.counts.end or "complete"
    ledger.line(report.end_line())
    ledger.close()
    return report


# --- Readers -----------------------------------------------------------------------------------------------


#: Status of a frame in the ledger: not written yet, given, lost.
_WAITING, _GIVEN, _LOST = 0, 1, 2


class CameraLedger:
    """A reader of the ledger of a capture that may still be running: the status of each frame, waited for.

    ``status`` returns once the line of a frame is there: the arrival time of a given frame,
    or the reason of a lost one. It raises ``CameraError`` when the frame has no line
    ``READER_WAIT_S`` seconds after its time (``CameraRecord.due``): the capture stopped, or
    it was never started. Safe for several threads.
    """

    def __init__(
        self,
        folder: Path,
        record: CameraRecord,
        *,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        wait_s: float = READER_WAIT_S,
    ) -> None:
        self._path = Path(folder) / LEDGER_NAME
        self.record = record
        self._clock = clock
        self._sleep = sleep
        self._wait_s = wait_s
        self._lock = threading.Lock()
        self._status = np.zeros(record.n_frames, dtype=np.uint8)
        self._arrived = np.full(record.n_frames, math.nan)
        self._reasons: dict[int, str] = {}
        self._handle: Any = None
        self._partial = ""
        #: How many frames have their line: the lines come in frame order.
        self.known = 0
        #: The end line, once it is there.
        self.end: dict[str, Any] | None = None

    def _read_new(self) -> None:
        """Take in the lines that the capture wrote since the last look. The caller holds the lock."""
        if self._handle is None:
            if not self._path.exists():
                return
            self._handle = self._path.open(encoding="utf-8")
        text = self._partial + self._handle.read()
        lines = text.split("\n")
        self._partial = lines.pop()
        for line in lines:
            if line.strip():
                self._take(json.loads(line))

    def _take(self, data: Mapping[str, Any]) -> None:
        if "end" in data:
            self.end = dict(data)
        elif "frame" in data:
            frame = int(data["frame"])
            self._check_next(frame, frame + 1)
            self._status[frame] = _GIVEN
            self._arrived[frame] = float(data["arrived_at"])
            self.known = frame + 1
        else:
            start, stop = (int(value) for value in data["lost"])
            self._check_next(start, stop)
            self._status[start:stop] = _LOST
            for frame in range(start, stop):
                self._reasons[frame] = str(data["reason"])
            self.known = stop

    def _check_next(self, start: int, stop: int) -> None:
        if start != self.known or not start < stop <= self.record.n_frames:
            raise ValueError(f"{self._path}: the line of the frames {start} to {stop - 1} comes after frame {self.known - 1}")

    def status(self, frame: int) -> tuple[bool, float | str]:
        """Wait for the line of a frame; return (True, arrival time) for a given frame, (False, reason) for a lost one."""
        if not 0 <= frame < self.record.n_frames:
            raise IndexError(f"frame {frame} is outside the camera run: it has the frames 0 to {self.record.n_frames - 1}")
        deadline = self.record.due(frame) + self._wait_s
        while True:
            with self._lock:
                if frame >= self.known:
                    self._read_new()
                if frame < self.known:
                    if self._status[frame] == _GIVEN:
                        return True, float(self._arrived[frame])
                    return False, self._reasons[frame]
            if self._clock() > deadline:
                raise CameraError(
                    f"the capture wrote nothing for frame {frame} of the camera run {self.record.clip} for "
                    f"{self._wait_s:g} s after its time: it stopped, or it is not running"
                )
            self._sleep(READER_POLL_S)

    def wait_end(self) -> dict[str, Any] | None:
        """Wait for the end line after the last frame (at most the reader wait); None when it does not come."""
        last = self.record.n_frames - 1
        self.status(last)
        deadline = max(self._clock(), self.record.due(last)) + self._wait_s
        while True:
            with self._lock:
                self._read_new()
                if self.end is not None:
                    return self.end
            if self._clock() > deadline:
                return None
            self._sleep(READER_POLL_S)

    def given_before(self, frame: int) -> int | None:
        """Return the newest given frame before ``frame`` among the frames that have their line, or None."""
        with self._lock:
            given = np.flatnonzero(self._status[: min(frame, self.known)] == _GIVEN)
        return int(given[-1]) if given.size else None

    def lost_spans(self) -> list[dict[str, Any]]:
        """Return the lost frames that have their line, as spans ``{"start", "stop", "reason"}`` (stop not included)."""
        spans: list[dict[str, Any]] = []
        with self._lock:
            for frame in range(self.known):
                if self._status[frame] != _LOST:
                    continue
                reason = self._reasons[frame]
                if spans and spans[-1]["stop"] == frame and spans[-1]["reason"] == reason:
                    spans[-1]["stop"] = frame + 1
                else:
                    spans.append({"start": frame, "stop": frame + 1, "reason": reason})
        return spans


class CameraFolderSource:
    """The frames of a camera clip folder (``FrameSource``), read while the capture writes them.

    ``read`` waits for the ledger line of the frame (``CameraLedger.status``). A lost frame
    gives the newest given frame before it, or a black frame when there is none: a frame
    without measurement keeps the picture of the camera still (the tools send no answer for
    it, so ``stream`` gets missing parts for it).
    """

    def __init__(self, folder: Path, record: CameraRecord, *, ledger: CameraLedger | None = None) -> None:
        self.folder = Path(folder)
        self.record = record
        self.ledger = ledger or CameraLedger(self.folder, record)
        self.info = ClipInfo(
            clip_id=record.clip, n_frames=record.n_frames, fps=record.fps, width=record.width, height=record.height
        )

    def path(self, frame: int) -> Path:
        """Return the image file of a frame (which exists only once the frame was given)."""
        return self.folder / FRAME_FOLDER / frame_file_name(frame)

    def read(self, frame: int) -> np.ndarray:
        given, _ = self.ledger.status(frame)
        shown: int | None = frame if given else self.ledger.given_before(frame)
        if shown is None:
            return np.zeros((self.info.height, self.info.width, 3), dtype=np.uint8)
        image = cv2.imread(str(self.path(shown)), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"cannot read the image of frame {shown} of the camera run {self.record.clip}")
        return image

    def facts(self) -> dict[str, Any]:
        """Return what the result file says of the camera run: the record without timeouts, the end line, the lost spans."""
        end = self.ledger.wait_end()
        return {
            "camera": self.record.camera,
            "clip": self.record.clip,
            "n_frames": self.record.n_frames,
            "fps": self.record.fps,
            "start_at": self.record.start_at,
            "end": end,
            "lost_spans": self.ledger.lost_spans(),
        }
