"""A stand-in of an RTSP camera behind ``camera.CameraSource``: a scripted ``cv2.VideoCapture`` on a simulated clock.

No network address is opened: the stand-in is the object that ``open_ffmpeg`` would return.
The camera sends frame ``k`` at ``t0 + (k + phase) / rate`` on the simulated clock (``Clock``),
with the time stamp ``k * 1000 / rate`` milliseconds counted from the first frame of each
connection, as FFmpeg gives it after a new connection. A plan (``CameraPlan``) adds what a real
camera does wrong: late frames, a stall, a lost connection, a camera that ends, one that never
opens, frames that it never sends, frames of another size, another frame rate. Every wait is a
step of the simulated clock, so a run takes no wall-clock time and gives the same result each
time. Nothing here says how a real camera, a real RTSP server or the real FFmpeg behave.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from pitchvlm.camera import CameraAddress, CameraSettings, CaptureDevice


class Clock:
    """A clock that moves only when the code sleeps or the stand-in waits."""

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, seconds)


@dataclass(frozen=True)
class CameraPlan:
    """What the stand-in camera does, in seconds of the simulated clock.

    ``t0``: when the camera sends its frame 0. ``outages``: spans ``(a, b)`` in which the link is
    down: an open fails after the open timeout, a read fails after the read timeout. ``stalls``:
    spans in which the camera sends nothing but the link stays up: the frames of the span come at
    its end (late frames; a stall longer than the read timeout ends the connection). ``late``: a
    delay of single frames, by frame. ``ends_at``: from then on the camera is gone. ``never_opens``:
    no open works. ``skipped``: frames that the camera never sends (a gap in its time stamps).
    ``other_size``: frames of ``other_shape``. ``rate``: the frames per second of the camera, and
    ``named_rate`` the rate that it names (``CAP_PROP_FPS``).
    """

    t0: float
    rate: float = 25.0
    named_rate: float | None = None
    phase: float = 0.5
    outages: Sequence[tuple[float, float]] = ()
    stalls: Sequence[tuple[float, float]] = ()
    late: dict[int, float] = field(default_factory=dict)
    ends_at: float = math.inf
    never_opens: bool = False
    skipped: frozenset[int] = frozenset()
    other_size: frozenset[int] = frozenset()
    other_shape: tuple[int, int] = (360, 640)
    #: Text that the stand-in writes to the descriptor 2 at each open, as FFmpeg can (the leak test plants the address).
    native_message: str | None = None
    #: Raise an error with this text at each failed open, as OpenCV can (the leak test plants the address).
    open_error: str | None = None


class StandInCamera:
    """The device factory and the devices of one stand-in camera (``camera.DeviceFactory``)."""

    def __init__(
        self,
        plan: CameraPlan,
        clock: Clock,
        image: Callable[[int], np.ndarray],
        *,
        real_sleep_s: float = 0.0,
        hold_from: int | None = None,
    ) -> None:
        self.plan = plan
        self.clock = clock
        self.image = image
        #: Wall-clock seconds that each frame takes besides the simulated clock: lets a bench stop a capture midway.
        self.real_sleep_s = real_sleep_s
        #: From this frame on, a read waits on the wall clock until the process is stopped (a stop-signal test).
        self.hold_from = hold_from
        self.opens = 0
        self.addresses: list[str] = []

    def send_time(self, frame: int) -> float:
        """Return when the camera sends a frame: its time at the rate, moved to the end of a stall, plus its delay."""
        plan = self.plan
        moment = plan.t0 + (frame + plan.phase) / plan.rate
        for start, stop in plan.stalls:
            if start <= moment < stop:
                moment = stop
        return moment + plan.late.get(frame, 0.0)

    def link_down(self, moment: float) -> bool:
        plan = self.plan
        return plan.never_opens or moment >= plan.ends_at or any(a <= moment < b for a, b in plan.outages)

    def __call__(self, address: CameraAddress, settings: CameraSettings) -> CaptureDevice:
        self.opens += 1
        self.addresses.append(address.redacted)
        if self.plan.native_message is not None:
            os.write(2, (self.plan.native_message + "\n").encode("utf-8"))
        if self.link_down(self.clock()):
            self.clock.sleep(settings.open_timeout_s)
            if self.plan.open_error is not None:
                raise cv2.error(self.plan.open_error)
            return _Device(self, settings, opened=False)
        return _Device(self, settings, opened=True)


class _Device:
    """One connection to the stand-in camera: frames from the first one sent after the connection was made."""

    def __init__(self, camera: StandInCamera, settings: CameraSettings, *, opened: bool) -> None:
        self._camera = camera
        self._settings = settings
        self._opened = opened
        plan = camera.plan
        now = camera.clock()
        first = max(0, math.ceil((now - plan.t0) * plan.rate - plan.phase))
        while camera.send_time(first) < now:
            first += 1
        self._next = first
        self._first = first
        self._image: np.ndarray | None = None
        self._stamp = 0.0

    def isOpened(self) -> bool:  # noqa: N802 (the name of OpenCV)
        return self._opened

    def grab(self) -> bool:
        camera, clock, settings = self._camera, self._camera.clock, self._settings
        if not self._opened:
            return False
        while self._next in camera.plan.skipped:
            self._next += 1
        frame = self._next
        due = camera.send_time(frame)
        # The link goes down before the frame comes, or the camera is quiet longer than the read timeout.
        down = [moment for moment in self._down_moments(clock(), due) if moment <= due]
        if down or due - clock() > settings.read_timeout_s:
            clock.sleep(settings.read_timeout_s if not down else max(0.0, down[0] - clock()) + settings.read_timeout_s)
            self._opened = False
            return False
        clock.sleep(due - clock())
        if camera.real_sleep_s > 0:
            time.sleep(camera.real_sleep_s)
        while camera.hold_from is not None and frame >= camera.hold_from:
            time.sleep(0.05)
        self._next += 1
        shape = camera.plan.other_shape if frame in camera.plan.other_size else None
        image = camera.image(frame)
        if shape is not None:
            image = cv2.resize(image, (shape[1], shape[0]))
        self._image = image
        self._stamp = (frame - self._first) * 1000.0 / camera.plan.rate
        return True

    def _down_moments(self, start: float, stop: float) -> list[float]:
        plan = self._camera.plan
        moments = [a for a, b in plan.outages if start <= a <= stop or a <= start < b]
        if plan.ends_at <= stop:
            moments.append(max(start, plan.ends_at))
        return sorted(max(start, moment) for moment in moments)

    def retrieve(self) -> tuple[bool, np.ndarray | None]:
        return (self._image is not None), self._image

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self._stamp
        if prop == cv2.CAP_PROP_FPS:
            plan = self._camera.plan
            return plan.rate if plan.named_rate is None else plan.named_rate
        return 0.0

    def release(self) -> None:
        self._opened = False
