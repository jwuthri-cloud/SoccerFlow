"""Shirt numbers in the live path: the paid number step of the model, off the frame path of ``stream``.

The clip path asks the model for the shirt number of every track after the whole clip
(``pipeline._shirt_numbers``). The streaming path gives each frame state once and never changes
it, so the number of a track comes later, as its own line of the live output. This module is the
worker that does it. The frame path only hands references to it and never waits for an answer:

- ``arrived`` (hook ``DetectorStream.on_arrived``, in ``push`` after the image is read): on a
  stride frame and on the frames of the kit legend, a reference to the image and the boxes of the
  frame go into a queue of at most ``max_pending_images`` images; when it is full the new image is
  dropped and counted (``images_dropped``). Nothing is copied on the frame path.
- ``given`` (with each given state): a reference to the state goes into the same queue, which
  never drops a state (the stream keeps every state anyway).

One worker thread takes the queue in order. It cuts the boxes of each stride frame out of its
image and keeps the crops until that frame is decided; then it gives the crop of each observed
state of a player or a goalkeeper to the pool of its stream id. An id is asked once:

- ``ready``: ``pipeline.crop_choice`` over its pool gives ``ready_crops`` crops (at least
  ``pipeline.MIN_CROP_GAP_S`` apart, so the id was seen for a few seconds);
- ``closed``: it has had no observed state for ``close_frames`` frames (the history of the stream:
  after it no window can give the id again), with at least one crop;
- ``end``: the stream ended, with at least one crop.

The request is the one of the clip path (``pipeline._number_request``, the team key ``none`` of
the detector engine), after one kit legend request. A thread pool sends the requests. At most
``max_requests`` number requests are sent per stream (the cap, fail closed): an id that is ready
after it is not asked and counted (``refused_by_cap``). After the first failure (a budget stop, an
error of the model, an answer that cannot be parsed) no request is sent any more. The answer is
decided with ``tracking.decide_number``, the rule of the clip path, and ``take`` returns it in the
thread of the stream, which writes it as a number line. ``finish`` asks the open ids and waits for
every request in flight (they are paid), with the wait of the clip path
(``pipeline._finish_requests_in_flight``). ``apply_numbers`` puts the numbers into the result.

Design and limits: ``docs/ARCHITECTURE.md`` (section ``livenumbers.py``) and ``docs/DECISIONS.md``
(entry "Shirt numbers in the live path").
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from pitchvlm.config import PipelineConfig
from pitchvlm.pipeline import (
    _CropPlan,
    _cut_crop,
    _finish_requests_in_flight,
    _InFlight,
    _NotSent,
    _number_request,
    crop_choice,
    keyframe_frames,
)
from pitchvlm.schemas import TEAM_ROLES, Box, ClipInfo, ClipResult, FrameState, KitLegend, PersonDet, TrackObs
from pitchvlm.tracking import decide_number
from pitchvlm.vlm.steps import build_kit_legend_request, parse_kit_legend, parse_number
from pitchvlm.vlm.types import Backend, VLMRequest, VLMResponse

logger = logging.getLogger(__name__)

#: Crops 1 s apart that make an id ready for its number request: it was seen for at least 3 s.
READY_CROPS = 4
#: Images that wait for the worker before a new one is dropped (about 50 MB of 1080p frames).
MAX_PENDING_IMAGES = 8
#: Crops that the pool of one id keeps (the tallest, 1 s apart).
POOL_CROPS = 16
#: Number requests per stream when the command names no cap: the guess of one clip of ``cli``.
DEFAULT_MAX_REQUESTS = 50
#: Times of the frames that the kit legend request shows, seconds from the start of the stream.
LEGEND_TIMES_S = (0.0, 1.0, 2.0)
#: Seconds between two looks of the worker at its queue when nothing wakes it.
POLL_S = 0.05
#: Why an id was asked.
Reason = Literal["ready", "closed", "end"]

_IMAGE, _STATE = "image", "state"


@dataclass(frozen=True)
class NumberSettings:
    """The rules of the live number step (module docstring).

    ``stride_frames`` None takes the keyframe step of the settings (12 frames at 25 fps);
    ``close_frames`` None closes no id before the end of the stream.
    """

    max_requests: int = DEFAULT_MAX_REQUESTS
    ready_crops: int = READY_CROPS
    max_pending_images: int = MAX_PENDING_IMAGES
    stride_frames: int | None = None
    close_frames: int | None = None

    def __post_init__(self) -> None:
        if self.max_requests < 0:
            raise ValueError(f"the cap of number requests must be 0 or more, got {self.max_requests}")
        if self.ready_crops < 1:
            raise ValueError(f"an id needs at least 1 crop to be ready, got {self.ready_crops}")
        if self.max_pending_images < 1:
            raise ValueError(f"at least 1 image must be able to wait, got {self.max_pending_images}")
        if self.stride_frames is not None and self.stride_frames < 1:
            raise ValueError(f"the stride must be at least 1 frame, got {self.stride_frames}")
        if self.close_frames is not None and self.close_frames < 1:
            raise ValueError(f"an id closes after at least 1 frame, got {self.close_frames}")


@dataclass(frozen=True)
class NumberLine:
    """The number of one stream id, as the model read it from the crops of some of its frames.

    ``export`` and ``display`` are those of ``tracking.decide_number``; ``first_frame`` and
    ``last_frame`` the first and the last crop. ``first_seen_frame`` is the first frame of the id
    in a given state, and ``latency_s`` the time (clock of the worker) from the arrival of that
    frame to ``take`` (None when the arrival is not known).
    """

    track_id: int
    export: int | None
    display: int | None
    first_frame: int
    last_frame: int
    crop_frames: tuple[int, ...]
    per_crop: tuple[int | None, ...]
    confidence: str
    reason: Reason
    first_seen_frame: int
    latency_s: float | None = None

    @property
    def shown(self) -> int | None:
        """The number that a viewer shows: the displayed one, else the export one (``liveout._player``)."""
        return self.display if self.display is not None else self.export


@dataclass
class _Id:
    """What the worker knows of one stream id that was not asked yet."""

    first_frame: int
    first_seen_at: float | None
    last_observed: int
    roles: Counter[str] = field(default_factory=Counter)
    #: Candidate crops: frame -> (box, crop).
    pool: dict[int, tuple[Box, np.ndarray]] = field(default_factory=dict)


@dataclass(frozen=True)
class _Asked:
    """A number request that waits for the kit legend or was sent."""

    track_id: int
    reason: Reason
    first_seen_frame: int
    first_seen_at: float | None
    #: (frame, box, crop) in frame order.
    crops: tuple[tuple[int, Box, np.ndarray], ...]


def _box_key(box: Box) -> tuple[float, float, float, float]:
    return box.x1, box.y1, box.x2, box.y2


class LiveNumbers:
    """The worker of the live number step (module docstring).

    ``backend`` answers the requests (a cached backend of a model session, or the fake backend
    of a synthetic clip); ``info`` names the clip and its image size. ``clock`` gives the times of
    the latencies. Call ``arrived`` and ``given`` from the thread of the stream, ``take`` there
    too, then ``finish`` (or ``abort`` on an early stop).
    """

    def __init__(
        self,
        config: PipelineConfig,
        backend: Backend,
        settings: NumberSettings,
        *,
        info: ClipInfo,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._backend = backend
        self._settings = settings
        self._info = info
        self._clock = clock
        self._fps = info.fps
        self._stride = settings.stride_frames or keyframe_frames(config, info.fps)
        self._legend_frames = frozenset(round(t * info.fps) for t in LEGEND_TIMES_S)
        # --- Shared with the frame path (each operation is one atomic step of a deque or a dict).
        self._events: deque[tuple[Any, ...]] = deque()
        self._pending_lock = threading.Lock()
        self._pending_images = 0
        self._wake = threading.Event()
        self._arrival: dict[int, float] = {}
        #: Decided numbers with the arrival time of the first frame of their id; the threads of the pool add them.
        self._answers: deque[tuple[NumberLine, float | None]] = deque()
        self._count_lock = threading.Lock()
        # --- The worker thread only.
        self._crops: dict[int, dict[tuple[float, float, float, float], np.ndarray]] = {}
        self._legend_images: dict[int, np.ndarray] = {}
        #: The newest stride image: the kit legend shows it when no frame of ``LEGEND_TIMES_S`` came.
        self._last_image: np.ndarray | None = None
        self._ids: dict[int, _Id] = {}
        self._done: set[int] = set()
        self._waiting: list[_Asked] = []
        self._oldest_arrival = 0
        self._newest_decided = -1
        # --- Requests.
        self._executor = ThreadPoolExecutor(max_workers=config.max_parallel, thread_name_prefix="pitchvlm-numbers")
        self._in_flight = _InFlight()
        self._futures: list[Future[Any]] = []
        self._failures: list[BaseException] = []
        self._legend: KitLegend | None = None
        self._legend_future: Future[Any] | None = None
        #: Set by the thread of the pool once the answer of the kit legend is read (or failed).
        self._legend_settled = threading.Event()
        # --- Counts (``report``).
        self.images_dropped = 0
        self.requests_planned = 0
        self.refused_by_cap = 0
        self.closed_without_crop = 0
        self.answers = 0
        self.lines: list[NumberLine] = []
        self._cap_warned = False
        # --- Life.
        self._finishing = False
        self._stopping = False
        self._finished = False
        self._worker_error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="pitchvlm-numbers-worker", daemon=True)
        self._thread.start()

    # --- The frame path --------------------------------------------------------------------------

    def arrived(self, frame: int, image: np.ndarray, people: Sequence[PersonDet]) -> None:
        """Note the arrival of a frame; hand a stride frame or a frame of the kit legend to the worker."""
        self._arrival[frame] = self._clock()
        stride = frame % self._stride == 0
        legend = frame in self._legend_frames
        if not (stride or legend):
            return
        with self._pending_lock:
            if self._pending_images >= self._settings.max_pending_images:
                self.images_dropped += 1
                return
            self._pending_images += 1
        boxes = tuple(person.box for person in people) if stride else ()
        self._events.append((_IMAGE, frame, image, boxes, legend))
        self._wake.set()

    def given(self, state: FrameState) -> None:
        """Hand one given state to the worker. It never blocks and never drops a state."""
        self._events.append((_STATE, state))
        self._wake.set()

    def take(self) -> list[NumberLine]:
        """Return the numbers that came since the last call, with their latency (thread of the stream)."""
        taken: list[NumberLine] = []
        now = self._clock()
        while self._answers:
            line, seen_at = self._answers.popleft()
            latency = None if seen_at is None else now - seen_at
            taken.append(dataclasses.replace(line, latency_s=latency))
        self.lines.extend(taken)
        return taken

    # --- The end -----------------------------------------------------------------------------------

    def finish(self) -> list[NumberLine]:
        """Ask the open ids, wait for every answer, stop the threads, and return the numbers not taken yet.

        Call it after ``DetectorStream.finish`` (every state is given). A stop signal during the
        wait ends the asking at once and still waits for the requests in flight, which are paid
        (``pipeline._finish_requests_in_flight``).
        """
        if self._finished:
            raise ValueError("the live number step is already finished")
        try:
            self._finishing = True
            self._wake.set()
            _wait_for(self._thread)
            wait(list(self._futures))
        finally:
            self._end()
        return self.take()

    def abort(self) -> None:
        """Send nothing more, and return when every request in flight has finished (an early stop)."""
        if self._finished:
            return
        self._end()

    def _end(self) -> None:
        self._finished = True
        self._stopping = True
        self._wake.set()
        try:
            _finish_requests_in_flight(self._executor, self._in_flight)
        finally:
            _wait_for(self._thread)

    @property
    def failure(self) -> BaseException | None:
        """The first failure of a request, of an answer or of the worker, or None."""
        if self._failures:
            return self._failures[0]
        return self._worker_error

    @property
    def cap_reached(self) -> bool:
        """Whether an id was ready after the cap of number requests was reached."""
        return self.refused_by_cap > 0

    def report(self) -> dict[str, Any]:
        """Return the counts of the step as plain JSON data."""
        latencies = sorted(line.latency_s for line in self.lines if line.latency_s is not None)
        failure = self.failure
        return {
            "max_requests": self._settings.max_requests,
            "ready_crops": self._settings.ready_crops,
            "stride_frames": self._stride,
            "close_frames": self._settings.close_frames,
            "requests_planned": self.requests_planned,
            "answers": self.answers,
            "number_lines": len(self.lines),
            "with_number": sum(1 for line in self.lines if line.export is not None),
            "refused_by_cap": self.refused_by_cap,
            "closed_without_crop": self.closed_without_crop,
            "images_dropped": self.images_dropped,
            "kit_legend": "answered" if self._legend is not None else ("asked" if self._legend_future else "not asked"),
            "failure": None if failure is None else f"{type(failure).__name__}: {failure}",
            "latency_s": latencies,
        }

    # --- The worker thread -------------------------------------------------------------------------

    def _run(self) -> None:
        try:
            while True:
                self._wake.wait(POLL_S)
                self._wake.clear()
                if self._stopping:
                    return
                self._drain()
                self._send_waiting()
                if self._finishing and not self._events:
                    self._ask_open()
                    if self._legend_future is not None:
                        while not self._legend_settled.is_set() and not self._stopping:
                            self._wake.wait(POLL_S)
                            self._wake.clear()
                    self._send_waiting()
                    return
        except BaseException as error:  # noqa: BLE001 (a fault of the worker must not stop the stream)
            logger.exception("the live number step stopped: no new numbers")
            self._worker_error = error

    def _drain(self) -> None:
        while self._events and not self._stopping:
            event = self._events.popleft()
            if event[0] == _IMAGE:
                with self._pending_lock:
                    self._pending_images -= 1
                _, frame, image, boxes, legend = event
                self._cut(frame, image, boxes, legend)
            else:
                self._decided(event[1])

    def _cut(self, frame: int, image: np.ndarray, boxes: Sequence[Box], legend: bool) -> None:
        if self._legend_future is None:
            if legend:
                self._legend_images[frame] = image
            if boxes:
                self._last_image = image
        if frame <= self._newest_decided:
            return
        crops: dict[tuple[float, float, float, float], np.ndarray] = {}
        for box in boxes:
            try:
                crops[_box_key(box)] = _cut_crop(image, box)
            except ValueError:  # a box outside the image gets no crop, as no request could hold it
                continue
        if crops:
            self._crops[frame] = crops

    def _decided(self, state: FrameState) -> None:
        frame = state.frame
        self._newest_decided = max(self._newest_decided, frame)
        crops = self._crops.pop(frame, {})
        for old in [index for index in self._crops if index < frame]:
            del self._crops[old]
        seen_at = self._arrival.get(frame)
        while self._oldest_arrival < frame:
            self._arrival.pop(self._oldest_arrival, None)
            self._oldest_arrival += 1
        touched: list[int] = []
        for player in state.players:
            track_id = player.track_id
            if track_id in self._done:
                continue
            known = self._ids.get(track_id)
            if known is None:
                known = self._ids[track_id] = _Id(first_frame=frame, first_seen_at=seen_at, last_observed=frame)
            if not player.observed or player.box is None:
                continue
            known.roles[player.role] += 1
            known.last_observed = frame
            crop = crops.get(_box_key(player.box))
            if crop is not None:
                known.pool[frame] = (player.box, crop)
                if len(known.pool) > POOL_CROPS:
                    chosen = crop_choice([(index, box) for index, (box, _) in known.pool.items()], self._fps, POOL_CROPS)
                    known.pool = {index: known.pool[index] for index, _ in chosen}
                touched.append(track_id)
        for track_id in touched:
            known = self._ids[track_id]
            if _team_role(known) and len(self._choice(known, self._settings.ready_crops)) >= self._settings.ready_crops:
                self._ask(track_id, "ready")
        close = self._settings.close_frames
        if close is not None:
            for track_id in [key for key, value in self._ids.items() if frame - value.last_observed >= close]:
                self._close(track_id, "closed")

    def _choice(self, known: _Id, max_crops: int) -> tuple[tuple[int, Box], ...]:
        return crop_choice([(index, box) for index, (box, _) in known.pool.items()], self._fps, max_crops)

    def _close(self, track_id: int, reason: Reason) -> None:
        known = self._ids[track_id]
        if known.pool and _team_role(known):
            self._ask(track_id, reason)
            return
        if _team_role(known):
            self.closed_without_crop += 1
        del self._ids[track_id]
        self._done.add(track_id)

    def _ask_open(self) -> None:
        for track_id in list(self._ids):
            self._close(track_id, "end")

    def _ask(self, track_id: int, reason: Reason) -> None:
        """Plan the number request of an id: within the cap, after the kit legend; the id is done."""
        known = self._ids.pop(track_id)
        self._done.add(track_id)
        if self._failures or self._stopping:
            return
        if self.requests_planned >= self._settings.max_requests:
            self.refused_by_cap += 1
            if not self._cap_warned:
                self._cap_warned = True
                logger.warning(
                    "the cap of %d number requests of this stream is reached: the stream goes on without new numbers",
                    self._settings.max_requests,
                )
            return
        self.requests_planned += 1
        chosen = self._choice(known, self._config.number_crops)
        asked = _Asked(
            track_id=track_id,
            reason=reason,
            first_seen_frame=known.first_frame,
            first_seen_at=known.first_seen_at,
            crops=tuple((index, known.pool[index][0], known.pool[index][1]) for index, _ in chosen),
        )
        self._waiting.append(asked)
        if self._legend_future is None:
            self._ask_legend()

    def _ask_legend(self) -> None:
        images = [self._legend_images[frame] for frame in sorted(self._legend_images)]
        if not images and self._last_image is not None:
            images = [self._last_image]
        request = build_kit_legend_request(
            images,
            clip_id=self._info.clip_id,
            settings=self._config.kit_legend,
            repeat=self._config.repeat,
        )
        self._legend_images, self._last_image = {}, None
        future = self._submit(request, self._legend_answered)
        self._legend_future = future
        if future is None:
            self._legend_settled.set()

    def _legend_answered(self, answer: VLMResponse) -> None:
        try:
            self._legend = parse_kit_legend(answer.text)
        except Exception as error:  # noqa: BLE001 (an answer that cannot be used stops the asking)
            self._failures.append(error)

    def _send_waiting(self) -> None:
        if self._legend is None or not self._waiting:
            if self._legend_settled.is_set() and self._legend is None:
                self._waiting.clear()  # the kit legend failed: no number request can be built
            return
        waiting, self._waiting = self._waiting, []
        for asked in waiting:
            if self._failures or self._stopping:
                return
            plan = _CropPlan(
                asked.track_id,
                "none",
                tuple(TrackObs(frame=index, box=box, role="player") for index, box, _ in asked.crops),
            )
            crops = {(asked.track_id, index): crop for index, _, crop in asked.crops}
            request = _number_request(plan, crops, self._legend, self._info, self._config)
            self._submit(request, lambda answer, asked=asked: self._number_answered(answer, asked))

    def _submit(self, request: VLMRequest, handle: Callable[[VLMResponse], None]) -> Future[Any] | None:
        try:
            future = self._executor.submit(self._generate, request, handle)
        except RuntimeError:  # the pool was shut down by an early stop
            return None
        self._futures.append(future)
        return future

    def _generate(self, request: VLMRequest, handle: Callable[[VLMResponse], None]) -> None:
        """Send one request and read its answer, in a thread of the pool (so the future is done after both).

        After the first failure nothing is sent (as ``pipeline._generate_all``). The answer of
        the kit legend, or its failure, wakes the worker.
        """
        try:
            with self._in_flight.working():
                if self._failures:
                    raise _NotSent
                try:
                    answer = self._backend.generate(request)
                except Exception as error:
                    self._failures.append(error)
                    raise
                handle(answer)
        finally:
            if request.meta.step == "kit_legend":
                self._legend_settled.set()
                self._wake.set()

    def _number_answered(self, answer: VLMResponse, asked: _Asked) -> None:
        try:
            reading = parse_number(answer.text)
        except Exception as error:  # noqa: BLE001 (an answer that cannot be used stops the asking)
            self._failures.append(error)
            return
        with self._count_lock:
            self.answers += 1
        export, display = decide_number(reading, ())
        frames = tuple(index for index, _, _ in asked.crops)
        line = NumberLine(
            track_id=asked.track_id,
            export=export,
            display=display,
            first_frame=frames[0],
            last_frame=frames[-1],
            crop_frames=frames,
            per_crop=reading.per_crop,
            confidence=reading.confidence,
            reason=asked.reason,
            first_seen_frame=asked.first_seen_frame,
        )
        self._answers.append((line, asked.first_seen_at))


def _team_role(known: _Id) -> bool:
    """Tell whether most observed states of an id are a player or a goalkeeper (the first counted wins a tie)."""
    if not known.roles:
        return False
    return known.roles.most_common(1)[0][0] in TEAM_ROLES


def _wait_for(thread: threading.Thread) -> None:
    """Return when the thread has ended. No ``Thread.join``: see ``pipeline._InFlight``."""
    while thread.is_alive():
        time.sleep(_InFlight.POLL_S)


# --- Wiring and result ---------------------------------------------------------------------------


class NumberWriter:
    """Connect a stream, its live number step and its live output (``wire``)."""

    def __init__(self, numbers: LiveNumbers, live_out: Any | None) -> None:
        self._numbers = numbers
        self._live_out = live_out

    def write(self, lines: Sequence[NumberLine], last: int) -> None:
        """Write the number lines to the live output, as decided at the frame ``last``."""
        if self._live_out is None:
            return
        for line in lines:
            self._live_out.number(
                track_id=line.track_id,
                export=line.export,
                display=line.display,
                first_frame=line.first_frame,
                last_frame=line.last_frame,
                crop_frames=line.crop_frames,
                reason=line.reason,
                last=last,
            )


def wire(stream: Any, numbers: LiveNumbers | None, live_out: Any | None) -> NumberWriter | None:
    """Connect the hooks of a ``DetectorStream``: the live output, and the live number step when there is one.

    Without a number step this is the hook of the live output alone, as before. With one, each
    given state goes to the live output first, then to the number step, and the numbers that came
    are written after that frame line.
    """
    if numbers is None:
        stream.on_given = None if live_out is None else live_out.given
        return None
    writer = NumberWriter(numbers, live_out)
    stream.on_arrived = numbers.arrived

    def on_given(state: FrameState, **flags: Any) -> None:
        if live_out is not None:
            live_out.given(state, **flags)
        numbers.given(state)
        writer.write(numbers.take(), flags["last"])

    stream.on_given = on_given
    return writer


def apply_numbers(result: ClipResult, lines: Sequence[NumberLine]) -> ClipResult:
    """Give every state and every track of an id with a number line the numbers of that line.

    The states of the other ids keep no number. An id has at most one line (it is asked once).
    """
    by_id: dict[int, NumberLine] = {}
    for line in lines:
        if line.track_id in by_id:
            raise ValueError(f"the id {line.track_id} has two number lines")
        by_id[line.track_id] = line
    if not by_id:
        return result

    def numbered(item: Any) -> Any:
        line = by_id.get(item.track_id)
        if line is None:
            return item
        return item.model_copy(update={"number_export": line.export, "number_display": line.display})

    frames = tuple(
        state.model_copy(update={"players": tuple(numbered(player) for player in state.players)})
        for state in result.frames
    )
    tracks = tuple(numbered(track) for track in result.tracks)
    return result.model_copy(update={"frames": frames, "tracks": tracks})
