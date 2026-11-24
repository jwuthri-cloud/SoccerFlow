"""The streaming path of the detector engine: one frame comes in, the state of an earlier frame goes out.

The clip path (``pipeline.run_clip``) reads the answers of the tools for every frame of a
clip and then looks at the whole clip: the kit clusters, the joins and repairs of the tracks,
the gap fill of the ball and the camera fill use later frames. This module takes one frame
at a time (its detections, its camera and its picture) and gives the state of frame ``t``
(``schemas.FrameState``) once frame ``t + L`` has come in, for a fixed delay of ``L`` frames
(``StreamSettings.delay_frames``). ``L = 0`` is purely causal. The state of a frame is final
when it is given: it is never changed later.

Each frame does two kinds of work.

1. On arrival of frame ``t`` (causal, once per frame):

   - Camera motion and the cut tests of the pair ``t - 1, t`` (``motion.ShotMeter``). A hard
     cut is known at once. A dissolve is known only when its window test is over, about
     ``motion.DISSOLVE_WINDOW_FRAMES`` frames after it starts; from then on its frames carry
     nothing, as in the clip path, but the states given before stay as they were given.
   - The camera of the calibration, refused when it is not plausible with the people of the
     frame (``pipeline.fit_problems``). A frame without a camera takes the camera of the
     nearest earlier frame that has one, moved with the camera motion.
   - The people who map outside the pitch are dropped, and each person gets a team key from
     the kit clusters (``kitcolour``): see "Kit clusters" below.
   - The people are matched to the live tracks (``tracking.Associator``, the matching of the
     clip path one frame at a time). A hard cut starts new tracks.

2. The decision of frame ``f = t - L``: the steps of the clip path that look at a whole clip
   run on a window of frames, from ``f - H`` to ``t`` (``StreamSettings.history_frames``):
   the cuts and dissolves known at ``t``, the camera fill, the kit roles
   (``kitcolour.kit_roles``), the split of team swaps and the joins (``tracking``), the votes
   and team sides (``tracking.finalize_tracks``), the gap fill of the people (``fill``) and of
   the ball (``ball.track_ball``). The functions are the ones of the clip path. The state of
   frame ``f`` is read from that window.

Identity across decisions: the window of the next frame is a new run, and its track ids are
new. A track of the window that holds observations given earlier keeps the id that the newest
of them was given; two tracks that ask for one id get it in the order of how recently they had
it, and the other one takes its next id or a new one (``_StreamIds``). So a late join or a late
split that the window sees changes no state that was given, and costs no extra switch.

With ``delay_frames`` and ``history_frames`` both None (the whole clip), every frame is decided
at the end of the clip on one window that holds the whole clip: the result is that of the clip
path, frame for frame, when the kit clusters are given (``kit_clusters``) and the clip has no
dissolve (``tests/test_stream.py``). The differences that remain otherwise are listed in
``docs/DECISIONS.md`` (entry "A streaming path for the detector engine" of 2026-11-23).

Kit clusters: two clusters that are fitted once on the first frames and then fixed. During the
warm-up (``StreamSettings.kit_warmup_frames``) they are fitted again on every frame, on all
torso features so far; after it they stay as they are. They are not updated slowly: the kits
do not change during a match, a refit can swap the names ``A`` and ``B`` of the two clusters
(cluster 0 is the one whose centre comes first in the order of the values), and a slow update
drifts towards a mix of the two kits when the light changes. A fit that does not show two kits
(``KitClusters.two_kits``) gives no team key, and is tried again on every frame, on the
features of the last ``kit_warmup_frames`` frames, until one does.

Input: rule 4 of the project forbids the package to import a tool, so the stream reads its
input as lines, one frame per line (``STREAM_FORMAT``): a header line, then one line per
frame with the payload of the detector and of the calibration as their adapters read them
(``fallbacks.rfdetr``, ``fallbacks.pnlcalib``). ``clip_run_lines`` writes these lines from
the clip runs of the two tools, and ``read_stream_lines`` reads them, from a file or from a
pipe of a tool process. The picture of each frame comes from the clip (a frame source).

Only the people engine ``detector`` with the pitch engine ``calibration`` and the number
engine ``none`` streams. The shirt numbers are a paid model step on the crops of a whole track.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import TypeAdapter

from pitchvlm.ball import BallCandidate, track_ball
from pitchvlm.config import PipelineConfig
from pitchvlm.fallbacks.pnlcalib import homography_from_payload
from pitchvlm.fallbacks.rfdetr import ball_candidates, expand_rows, people_from_detections
from pitchvlm.fallbacks.runs import ClipRun, DetectorSettings, PnlcalibSettings, is_detector
from pitchvlm.fill import players_in_frame
from pitchvlm.geometry import apply_homography, invert_homography, matrix_from_record, matrix_to_rows, paint_alignment
from pitchvlm.kitcolour import KitClusters, fit_kit_clusters, kit_roles, torso_feature
from pitchvlm.motion import ShotMeter
from pitchvlm.pitch import inside_pitch
from pitchvlm.pipeline import (
    BALL_MAX_GAP_S,
    KIT_CLUSTER_SEED,
    _engines,
    _feet,
    _people_on_pitch,
    ball_state,
    every_frame_tracking,
    fit_problems,
    keyframe_frames,
)
from pitchvlm.schemas import (
    ClipInfo,
    ClipResult,
    FramePeople,
    FrameState,
    HomographyRecord,
    PaintCheck,
    PersonDet,
    PlayerState,
    Track,
    TrackObs,
)
from pitchvlm.timing import StageClock
from pitchvlm.tracking import (
    SWAP_MIN_VOTES,
    Associator,
    PitchMap,
    finalize_tracks,
    split_team_swaps,
    stitch_fragments,
)

logger = logging.getLogger(__name__)

#: Format name of the lines of a stream: a header line, then one line per frame.
STREAM_FORMAT = "pitchvlm-stream-2"
#: The role of the people next to the pitch (as in ``pipeline``): they do not test a camera.
_ROLE_NEXT_TO_PITCH = "other"
#: Default length of the kit warm-up, frames: 1 s at 25 frames per second. Fixed on the fit seeds of the
#: decision entry (``docs/DECISIONS.md``, "A streaming path for the detector engine", 2026-11-23).
KIT_WARMUP_FRAMES = 25
#: Default of ``StreamSettings.early_dissolve_windows``: a dissolve is taken before its run of low
#: windows ends once its span came out the same for this many judged windows in a row
#: (``motion.ShotMeter.open_transition``). Fixed on the fit seeds of the decision entry
#: (``docs/DECISIONS.md``, "The live feed of the detector engine", 2026-11-23).
EARLY_DISSOLVE_WINDOWS = 4
#: The paint scores of the keyframes run on one thread beside the frame path; a frame waits
#: only when this many scores are still open (so the images they hold stay bounded).
PAINT_MAX_PENDING = 4


# --- Settings ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamSettings:
    """The delay and the window of the streaming path.

    ``delay_frames`` is ``L``: the state of frame ``t`` is given once frame ``t + L`` is in.
    ``history_frames`` is ``H``: the decision of frame ``f`` looks at the frames from ``f - H``
    to ``f + L``. None for either means the whole clip: then every frame is decided at the end.
    ``kit_warmup_frames`` is the number of first frames on which the kit clusters are fitted.
    ``early_dissolve_windows`` takes a dissolve before the window test has passed it
    (``motion.ShotMeter.open_transition``); None waits for the end of the window test.
    """

    delay_frames: int | None = 25
    history_frames: int | None = None
    kit_warmup_frames: int = KIT_WARMUP_FRAMES
    early_dissolve_windows: int | None = EARLY_DISSOLVE_WINDOWS

    def __post_init__(self) -> None:
        if self.delay_frames is not None and self.delay_frames < 0:
            raise ValueError(f"the delay must be 0 frames or more, got {self.delay_frames}")
        if self.history_frames is not None and self.history_frames < 0:
            raise ValueError(f"the history must be 0 frames or more, got {self.history_frames}")
        if self.kit_warmup_frames < 1:
            raise ValueError(f"the kit warm-up must be at least 1 frame, got {self.kit_warmup_frames}")
        if self.delay_frames is None and self.history_frames is not None:
            raise ValueError("a delay of the whole clip needs a history of the whole clip")
        if self.early_dissolve_windows is not None and self.early_dissolve_windows < 1:
            raise ValueError(f"the early dissolve test needs at least 1 window, got {self.early_dissolve_windows}")


def default_history_frames(config: PipelineConfig, fps: float) -> int:
    """Return the default window before the decided frame, frames.

    It holds the longest gap of a join (``max_stitch_gap_s``) and the votes that a team-key
    change needs on its earlier side (``SWAP_MIN_VOTES`` keyframe intervals of frames): 75 + 48
    = 123 frames (4.9 s) at 25 frames per second with the default settings.
    """
    return round(config.tracking.max_stitch_gap_s * fps) + SWAP_MIN_VOTES * keyframe_frames(config, fps)


def check_stream_engines(config: PipelineConfig) -> None:
    """Raise ``ValueError`` unless the settings name the engines that stream (module docstring)."""
    engines = config.engines
    if (engines.people, engines.pitch_mapping, engines.number) != ("detector", "calibration", "none"):
        raise ValueError(
            "the streaming path needs the people engine detector, the pitch engine calibration and the number "
            f"engine none; the settings name {engines.people}, {engines.pitch_mapping} and {engines.number}"
        )


# --- Input -------------------------------------------------------------------------------------

#: Why a part of a frame line holds no answer of its tool (``feed.py`` writes them): the wait for
#: it ran out, the tool skipped the frame because it was behind, or the feed of the tool had ended.
MISSING_REASONS = ("late", "skipped", "ended")
#: The two parts of a frame line, by tool.
STREAM_TOOLS = ("detector", "calibration")


@dataclass(frozen=True)
class StreamInput:
    """What the tools give for one frame: the people and the ball candidates of the detector, the camera.

    ``camera`` has ``source="none"`` where the calibration found no camera. The times are the
    times of the tools for this frame, seconds, as the tool process measured them, or None.
    ``arrived_at`` is the earliest time (seconds since the epoch) at which a tool had the frame,
    when the lines say it (a live feed). ``missing`` names each tool whose part holds no answer,
    with its reason (``MISSING_REASONS``): such a frame has no people, or no camera, of that tool.
    """

    frame: int
    people: tuple[PersonDet, ...]
    balls: tuple[BallCandidate, ...]
    camera: HomographyRecord
    detector_seconds: float | None = None
    calibration_seconds: float | None = None
    arrived_at: float | None = None
    missing: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class StreamHeader:
    """The header line of a stream: the clip, its number of frames, its video and the settings of the two tools.

    ``video_sha256`` is the SHA-256 of the video whose frames the tools read, or None for a
    clip that came as images (as in a clip run).
    """

    clip: str
    n_frames: int
    video_sha256: str | None
    detector: DetectorSettings
    calibration: PnlcalibSettings


def stream_header_line(
    clip: str, n_frames: int, video_sha256: str | None, detector: Mapping[str, Any], calibration: Mapping[str, Any]
) -> str:
    """Return the header line of a stream (no line end)."""
    header = {
        "format": STREAM_FORMAT,
        "clip": clip,
        "n_frames": n_frames,
        "video_sha256": video_sha256,
        "detector": dict(detector),
        "calibration": dict(calibration),
    }
    return json.dumps(header, sort_keys=True, allow_nan=False)


def stream_frame_line(frame: int, detector: Mapping[str, Any], calibration: Mapping[str, Any]) -> str:
    """Return the line of one frame (no line end): the part of each tool, ``{"seconds", "payload"}`` and
    ``arrived_at`` when known, or ``{"missing": <reason>}``."""
    line = {"frame": frame, "detector": dict(detector), "calibration": dict(calibration)}
    return json.dumps(line, sort_keys=True, allow_nan=False)


def clip_run_lines(detections: ClipRun, calibration: ClipRun) -> Iterator[str]:
    """Write the lines of a stream from the clip runs of the detector and of the calibration.

    The first line is the header; then one line per frame, in frame order. The live feeds of
    the two tools are merged into the same lines (``feed.py``). ``ValueError`` is raised when
    the two runs are not runs of one clip, and of one video, by the two tools.
    """
    if not is_detector(detections.tool) or not isinstance(calibration.tool, PnlcalibSettings):
        raise ValueError("a stream needs the clip run of the detector and the clip run of the calibration")
    if (detections.clip, detections.n_frames) != (calibration.clip, calibration.n_frames):
        raise ValueError(
            f"the clip runs are runs of different clips: {detections.clip} with {detections.n_frames} frames and "
            f"{calibration.clip} with {calibration.n_frames} frames"
        )
    if detections.video_sha256 != calibration.video_sha256:
        raise ValueError("the clip runs name different videos")
    yield stream_header_line(
        detections.clip,
        detections.n_frames,
        detections.video_sha256,
        detections.tool.model_dump(mode="json"),
        calibration.tool.model_dump(mode="json"),
    )
    for found, camera in zip(detections.frames, calibration.frames, strict=True):
        yield stream_frame_line(
            found.frame,
            {"seconds": found.seconds, "payload": found.payload},
            {"seconds": camera.seconds, "payload": camera.payload},
        )


def read_stream_lines(lines: Iterable[str]) -> tuple[StreamHeader, Iterator[StreamInput]]:
    """Read the header of a stream, and return it with an iterator over the inputs of its frames.

    The frames must come in order from frame 0, one per line; an empty line is skipped. A
    line that is not JSON, a wrong format, a frame out of order, a part that is neither an
    answer nor a missing answer, and a payload that the adapter of its tool refuses raise
    ``ValueError`` with the line number. The iterator reads one line at a time, so it works on
    a pipe, and it reads no line after frame ``n_frames - 1``: a pipe whose writer stays open
    after the last frame (a merger that waits for a tool) does not hold the stream back.
    """
    numbered = ((number, line) for number, line in enumerate(lines, start=1) if line.strip())
    try:
        number, first = next(numbered)
    except StopIteration:
        raise ValueError("the stream has no header line") from None
    head = _json_line(first, number)
    if head.get("format") != STREAM_FORMAT:
        raise ValueError(f"line {number}: the stream does not have the format {STREAM_FORMAT}")
    try:
        n_frames, video = head["n_frames"], head["video_sha256"]
        if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames < 1:
            raise ValueError(f"'n_frames' must be a whole number from 1, got {n_frames!r}")
        if video is not None and not _is_sha256(video):
            raise ValueError(f"'video_sha256' must be null or 64 small hexadecimal digits, got {video!r}")
        header = StreamHeader(
            clip=str(head["clip"]),
            n_frames=n_frames,
            video_sha256=video,
            detector=TypeAdapter(DetectorSettings).validate_python(head["detector"]),
            calibration=PnlcalibSettings.model_validate(head["calibration"]),
        )
    except (KeyError, ValueError) as error:
        raise ValueError(f"line {number}: the header of the stream is wrong: {error}") from error
    return header, _inputs(numbered, header)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def _inputs(numbered: Iterator[tuple[int, str]], header: StreamHeader) -> Iterator[StreamInput]:
    expected = 0
    for number, line in numbered:
        data = _json_line(line, number)
        try:
            frame = data["frame"]
            if isinstance(frame, bool) or not isinstance(frame, int) or frame != expected:
                raise ValueError(f"frame {expected} was expected, the line holds frame {frame!r}")
            parts = {tool: _part(data[tool], tool) for tool in STREAM_TOOLS}
            found, camera = parts["detector"], parts["calibration"]
            if found.payload is None:
                people: tuple[PersonDet, ...] = ()
                balls: tuple[BallCandidate, ...] = ()
            else:
                payload = expand_rows(found.payload)
                people = people_from_detections(payload, frame=frame, min_score=header.detector.min_score).people
                balls = ball_candidates(payload, frame=frame, min_score=header.detector.min_score)
            record = (
                HomographyRecord(frame=frame)
                if camera.payload is None
                else homography_from_payload(camera.payload, frame=frame)
            )
            times = [part.arrived_at for part in parts.values() if part.arrived_at is not None]
            yield StreamInput(
                frame=frame,
                people=people,
                balls=balls,
                camera=record,
                detector_seconds=found.seconds,
                calibration_seconds=camera.seconds,
                arrived_at=min(times) if times else None,
                missing=tuple((tool, part.missing) for tool, part in parts.items() if part.missing is not None),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"line {number}: {error}") from error
        expected += 1
        if expected == header.n_frames:
            return


@dataclass(frozen=True)
class _Part:
    """The part of one tool in a frame line: its answer, or the reason why there is none."""

    payload: dict[str, Any] | None
    seconds: float | None
    arrived_at: float | None
    missing: str | None


def _part(value: Any, tool: str) -> _Part:
    """Read the part of a tool: ``{"seconds", "payload"}`` with an optional ``arrived_at``, or ``{"missing": reason}``."""
    if not isinstance(value, dict):
        raise ValueError(f"the part of the {tool} must be a JSON object")
    if "missing" in value:
        if set(value) != {"missing"} or value["missing"] not in MISSING_REASONS:
            raise ValueError(
                f"a missing part of the {tool} is {{\"missing\": <one of {', '.join(MISSING_REASONS)}>}}, got {value!r}"
            )
        return _Part(payload=None, seconds=None, arrived_at=None, missing=value["missing"])
    unknown = set(value) - {"seconds", "payload", "arrived_at"}
    if unknown:
        raise ValueError(f"the part of the {tool} has unknown keys {sorted(unknown)}")
    payload = value["payload"]
    if not isinstance(payload, dict):
        raise ValueError(f"the payload of the {tool} must be a JSON object")
    arrived = value.get("arrived_at")
    return _Part(
        payload=payload,
        seconds=_seconds(value.get("seconds")),
        arrived_at=None if arrived is None else _epoch(arrived),
        missing=None,
    )


def _json_line(line: str, number: int) -> dict[str, Any]:
    try:
        data = json.loads(line, parse_constant=_no_constant)
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"line {number}: no JSON: {error}") from error
    except ValueError as error:
        raise ValueError(f"line {number}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"line {number}: a line of a stream is a JSON object")
    return data


def _no_constant(token: str) -> Any:
    raise ValueError(f"{token} is not a number of JSON")


def _seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value < math.inf:
        raise ValueError(f"a time must be a number of 0 or more seconds, got {value!r}")
    return float(value)


def _epoch(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value < math.inf:
        raise ValueError(f"'arrived_at' must be a time in seconds since the epoch, got {value!r}")
    return float(value)


class QueuedLines:
    """The lines of a pipe, read on their own thread into a bounded queue: the back-pressure of the stream.

    The policy (``docs/DECISIONS.md``, entry "The live feed of the detector engine", 2026-11-23):
    no frame is ever dropped, because each frame carries a step of the camera motion and of the
    matching. When the stream decides more slowly than the lines come, they wait in the queue;
    when ``max_lines`` lines wait, the thread stops reading, the pipe fills, and the writer of the
    lines (the merger, and through it the tools) waits too. So a slow stream delays the frames,
    and ``timing.LatencyLog`` shows by how much. The thread notes the time at which it read each
    line, so the wait in the queue counts in the latency.

    Iterate over it once. ``last_read_at`` is the time (``clock``) at which the line given last
    was read, and ``last_backlog`` the lines that were waiting in the queue when it was taken.
    An error of the reading is raised in the reader.
    """

    def __init__(self, lines: Iterable[str], max_lines: int, clock: Callable[[], float] = time.monotonic) -> None:
        if max_lines < 1:
            raise ValueError(f"the queue must hold at least 1 line, got {max_lines}")
        self._queue: queue.Queue[tuple[Any, float]] = queue.Queue(maxsize=max_lines)
        self._clock = clock
        self.last_read_at: float | None = None
        self.last_backlog = 0
        self.largest_backlog = 0
        self._thread = threading.Thread(target=self._read, args=(lines,), name="pitchvlm-lines", daemon=True)
        self._thread.start()

    def _read(self, lines: Iterable[str]) -> None:
        try:
            for line in lines:
                self._queue.put((line, self._clock()))
        except BaseException as error:  # handed to the reader, which raises it
            self._queue.put((error, self._clock()))
            return
        self._queue.put((_END_OF_LINES, self._clock()))

    def __iter__(self) -> Iterator[str]:
        while True:
            item, read_at = self._queue.get()
            backlog = self._queue.qsize()
            if item is _END_OF_LINES:
                return
            if isinstance(item, BaseException):
                raise item
            self.last_read_at, self.last_backlog = read_at, backlog
            self.largest_backlog = max(self.largest_backlog, backlog)
            yield item


_END_OF_LINES = object()


def stand_in_inputs(
    people: Sequence[FramePeople],
    balls: Mapping[int, Sequence[BallCandidate]],
    cameras: Sequence[HomographyRecord],
) -> Iterator[StreamInput]:
    """Turn the answers of the stand-ins of the tools (``fallbacks.fake``) into the inputs of a stream."""
    for frame_people, camera in zip(people, cameras, strict=True):
        frame = frame_people.frame
        yield StreamInput(
            frame=frame, people=frame_people.people, balls=tuple(balls.get(frame, ())), camera=camera
        )


# --- Camera motion as far as it is known -------------------------------------------------------


class _StreamMotion:
    """The camera motion of the frames so far, with the cuts known so far: the part of ``MotionField`` that is used.

    Steps into a cut count as missing. ``forget_before`` drops old steps.
    """

    def __init__(self) -> None:
        self._forward: dict[int, np.ndarray] = {}
        self._backward: dict[int, np.ndarray] = {}
        self.n_frames = 0
        self.cuts: tuple[int, ...] = ()

    def add_frame(self, step: np.ndarray | None) -> None:
        """Add the next frame and the step into it from the frame before (None when missing)."""
        if step is not None:
            matrix = np.array(step, dtype=np.float64)
            inverse = invert_homography(matrix)
            if inverse is None:
                raise ValueError(f"the motion step into frame {self.n_frames} has no usable inverse")
            self._forward[self.n_frames - 1] = matrix
            self._backward[self.n_frames - 1] = inverse
        self.n_frames += 1

    def between(self, src: int, dst: int) -> np.ndarray | None:
        """Return the matrix from frame ``src`` to frame ``dst`` (as ``MotionField.between``), or None."""
        if not self.same_shot(src, dst):
            return None
        if src <= dst:
            chain = [self._forward.get(index) for index in range(src, dst)]
        else:
            chain = [self._backward.get(index) for index in range(dst, src)][::-1]
        matrix = np.eye(3)
        for step in chain:
            if step is None:
                return None
            matrix = step @ matrix
        return matrix

    def warp(self, src: int, dst: int, xy: np.ndarray) -> np.ndarray | None:
        matrix = self.between(src, dst)
        return None if matrix is None else apply_homography(matrix, xy)

    def shot_of(self, frame: int) -> int:
        return bisect.bisect_right(self.cuts, frame)

    def same_shot(self, first: int, second: int) -> bool:
        return self.shot_of(first) == self.shot_of(second)

    def forget_before(self, frame: int) -> None:
        for steps in (self._forward, self._backward):
            for index in [index for index in steps if index < frame]:
                del steps[index]


def _bridged(motion: _StreamMotion, matrices: Mapping[int, np.ndarray]) -> Any:
    """Return the warp of ``motion.bridged_warp`` over the stream motion and image-to-pitch matrices by frame."""
    inverses: dict[int, np.ndarray | None] = {}

    def to_image(frame: int) -> np.ndarray | None:
        if frame not in inverses:
            inverse = invert_homography(matrices[frame])
            if inverse is None:
                raise ValueError(f"the pitch mapping of frame {frame} has no usable inverse")
            inverses[frame] = inverse
        return inverses[frame]

    def warp(src: int, dst: int, xy: np.ndarray) -> np.ndarray | None:
        moved = motion.warp(src, dst, xy)
        if moved is not None:
            return moved
        if not motion.same_shot(src, dst) or src not in matrices or dst not in matrices:
            return None
        return apply_homography(to_image(dst) @ matrices[src], xy)

    return warp


class _PitchMemo:
    """Pitch positions of single points, kept between decisions while the mapping of their frame stays the same.

    The window of each decision maps the foot points of the same observations again; a
    position is computed once per mapping (the same matrix object) and point.
    """

    def __init__(self) -> None:
        self._by_frame: dict[int, tuple[np.ndarray, dict[tuple[float, float], np.ndarray]]] = {}
        self._points: dict[int, tuple[np.ndarray, dict[tuple[float, float], tuple[float, float] | None]]] = {}

    def pitch_map(self, matrices: Mapping[int, np.ndarray]) -> PitchMap:
        """Return the function of ``pipeline._pitch_map`` over image-to-pitch matrices by frame."""

        def to_pitch(frame: int, xy: np.ndarray) -> np.ndarray | None:
            matrix = matrices.get(frame)
            if matrix is None:
                return None
            if len(xy) != 1:
                return apply_homography(matrix, xy)
            entry = self._by_frame.get(frame)
            if entry is None or entry[0] is not matrix:
                entry = self._by_frame[frame] = (matrix, {})
            key = (float(xy[0][0]), float(xy[0][1]))
            known = entry[1].get(key)
            if known is None:
                known = apply_homography(matrix, xy)
                known.setflags(write=False)
                entry[1][key] = known
            return known

        return to_pitch

    def point(self, matrices: Mapping[int, np.ndarray], observation: TrackObs) -> tuple[float, float] | None:
        """Return the pitch position of the foot point of an observation, or None (``pipeline._pitch_point``)."""
        frame = observation.frame
        matrix = matrices.get(frame)
        if matrix is None:
            return None
        entry = self._points.get(frame)
        if entry is None or entry[0] is not matrix:
            entry = self._points[frame] = (matrix, {})
        foot = observation.box.foot
        if foot not in entry[1]:
            position = apply_homography(matrix, np.array([foot], dtype=np.float64))
            entry[1][foot] = (
                (float(position[0, 0]), float(position[0, 1])) if np.isfinite(position).all() else None
            )
        return entry[1][foot]

    def forget_before(self, frame: int) -> None:
        for store in (self._by_frame, self._points):
            for index in [index for index in store if index < frame]:
                del store[index]


def _mean_points(
    tracks: Sequence[Track], point: Callable[[TrackObs], tuple[float, float] | None]
) -> tuple[dict[int, tuple[float, float]], dict[int, float]]:
    """Return ``pipeline._mean_positions`` and ``pipeline._mean_pitch_x`` of some tracks in one pass.

    ``math.fsum`` is exact, so the means are the same numbers as those of the two functions.
    """
    positions: dict[int, tuple[float, float]] = {}
    pitch_x: dict[int, float] = {}
    for track in tracks:
        known = [found for found in map(point, track.observations) if found is not None]
        if known:
            mean_x = math.fsum(x for x, _ in known) / len(known)
            positions[track.track_id] = (mean_x, math.fsum(y for _, y in known) / len(known))
            pitch_x[track.track_id] = mean_x
    return positions, pitch_x


# --- Identities --------------------------------------------------------------------------------


def _obs_key(frame: int, box: Any) -> tuple[int, float, float, float, float]:
    return frame, box.x1, box.y1, box.x2, box.y2


@dataclass
class _StreamIds:
    """The ids that the given states carry: an observation keeps the id it was given (module docstring)."""

    #: (frame, box) of every given observed state -> its id.
    given: dict[tuple[int, float, float, float, float], int] = field(default_factory=dict)
    used: set[int] = field(default_factory=set)
    next_id: int = 0

    def ids(self, tracks: Sequence[Track], frame: int) -> dict[int, int]:
        """Return the id of every window track that has a state in ``frame``, by window track id."""
        claims: list[tuple[int, list[int], Track]] = []
        for track in tracks:
            # The ids of the given observations of the track, the newest first, and the frame of the newest.
            asked: list[int] = []
            newest = -1
            for observation in reversed(track.observations):
                if observation.frame >= frame:
                    continue
                given = self.given.get(_obs_key(observation.frame, observation.box))
                if given is None or given in asked:
                    continue
                if not asked:
                    newest = observation.frame
                asked.append(given)
            claims.append((newest, asked, track))
        taken: dict[int, int] = {}
        in_frame: set[int] = set()
        # The track that had its id most recently chooses first; the order of the window breaks a tie.
        for _, asked, track in sorted(claims, key=lambda claim: -claim[0]):
            choice = next((given for given in asked if given not in in_frame), None)
            if choice is None:
                choice = self._new_id(track.track_id)
            taken[track.track_id] = choice
            in_frame.add(choice)
        return taken

    def _new_id(self, wanted: int) -> int:
        """Return a new id: the window id when no state carried it yet (so the whole clip keeps the clip ids)."""
        if wanted not in self.used:
            choice = wanted
        else:
            while self.next_id in self.used:
                self.next_id += 1
            choice = self.next_id
        self.used.add(choice)
        return choice

    def note(self, frame: int, state: PlayerState) -> None:
        self.used.add(state.track_id)
        if state.observed and state.box is not None:
            self.given[_obs_key(frame, state.box)] = state.track_id

    def forget_before(self, frame: int) -> None:
        for key in [key for key in self.given if key[0] < frame]:
            del self.given[key]


# --- Paint scores beside the frame path --------------------------------------------------------


def _paint_check(frame: int, image: np.ndarray, matrix: np.ndarray) -> tuple[PaintCheck, float]:
    start = time.perf_counter()
    alignment = paint_alignment(image, matrix)
    return PaintCheck(frame=frame, score=alignment.score, reason=alignment.reason), time.perf_counter() - start


class _PaintWorker:
    """The paint scores of the keyframes (a diagnostic), on one thread beside the frame path.

    ``paint_alignment`` is a function of the image and the matrix only, so the scores are those
    of the frame path. Most of its time is in OpenCV and NumPy, which let the frame path run
    meanwhile. A frame waits for the oldest open score when ``max_pending`` are open: nothing
    is dropped. ``seconds`` holds the time of each score, measured on the thread.
    """

    def __init__(self, max_pending: int = PAINT_MAX_PENDING) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pitchvlm-paint")
        self._max_pending = max_pending
        self._open: deque[Future[tuple[PaintCheck, float]]] = deque()
        self.checks: list[PaintCheck] = []
        self.seconds: list[float] = []

    def submit(self, frame: int, image: np.ndarray, matrix: np.ndarray) -> None:
        while self._open and (len(self._open) >= self._max_pending or self._open[0].done()):
            self._take()
        self._open.append(self._pool.submit(_paint_check, frame, image, matrix))

    def _take(self) -> None:
        check, seconds = self._open.popleft().result()
        self.checks.append(check)
        self.seconds.append(seconds)

    def finish(self) -> list[PaintCheck]:
        """Wait for the open scores, stop the thread and return every score in frame order."""
        try:
            while self._open:
                self._take()
        finally:
            self._pool.shutdown(wait=True, cancel_futures=True)
        return sorted(self.checks, key=lambda check: check.frame)


# --- The stream --------------------------------------------------------------------------------


class DetectorStream:
    """The streaming path of the detector engine (module docstring). ``push`` each frame in order, then ``finish``.

    ``fps``, ``size`` (width, height) and ``clip_id`` describe the clip. ``kit_clusters``, when
    given, are used from the first frame and never fitted (the control of the decision entry,
    and a match whose kits are known from an earlier part). ``clock`` gets the time of each
    stage, and one frame sample per ``push`` (``timing.StageClock.frame``).
    """

    def __init__(
        self,
        config: PipelineConfig,
        settings: StreamSettings,
        *,
        clip_id: str,
        fps: float,
        size: tuple[int, int],
        kit_clusters: KitClusters | None = None,
        clock: StageClock | None = None,
    ) -> None:
        check_stream_engines(config)
        self._config = config
        self._settings = settings
        self._clip_id = clip_id
        self._fps = fps
        self._size = size
        self._clock = clock if clock is not None else StageClock()
        self._tracking = every_frame_tracking(config, fps)
        self._keyframe_frames = keyframe_frames(config, fps)
        self._history = settings.history_frames
        self._max_fill = round(config.tracking.max_stitch_gap_s * fps)
        self._ball_gap = round(BALL_MAX_GAP_S * fps)
        self._meter = ShotMeter(size)
        self._motion = _StreamMotion()
        self._memo = _PitchMemo()
        #: Plausible cameras of the calibration, by frame.
        self._kept: dict[int, np.ndarray] = {}
        self._kept_records: dict[int, HomographyRecord] = {}
        #: Image-to-pitch matrix of each frame as known at its arrival (for the matching and the pitch filter).
        self._arrival: dict[int, np.ndarray] = {}
        #: Frames that had no camera at their arrival: their people were not tested against the pitch then.
        self._blind: set[int] = set()
        self._balls: dict[int, tuple[BallCandidate, ...]] = {}
        self._given_kits = kit_clusters
        self._clusters: KitClusters | None = kit_clusters if kit_clusters is not None and kit_clusters.two_kits() else None
        self._features: list[tuple[int, np.ndarray]] = []
        self._associators: list[Associator] = []
        self._hard_cuts: list[int] = []
        self._transitions: list[tuple[int, int]] = []
        self._ids = _StreamIds()
        #: With a history of the whole clip: every person of every frame with its team key, for the matching of finish.
        self._keyed: list[FramePeople] = []
        self._states: list[FrameState] = []
        self._records: list[HomographyRecord] = []
        self._paint: list[PaintCheck] = []
        self._painter = _PaintWorker()
        #: Each transition when it was first known: ("early" or "final", (start, stop), the newest frame then).
        self.transition_log: list[tuple[str, tuple[int, int], int]] = []
        self._keyframe_step = max(1, round(config.keyframe_interval_s * fps))
        self._last_image: np.ndarray | None = None
        self._last_camera: HomographyRecord | None = None
        self.tool_seconds: dict[str, list[float]] = {"detector": [], "calibration": []}
        self._n = 0
        self._finished = False

    # --- Arrival ---------------------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        """Number of frames pushed so far."""
        return self._n

    @property
    def paint_seconds(self) -> list[float]:
        """The time of each paint score, seconds, measured on its thread (not in the frame samples)."""
        return list(self._painter.seconds)

    def push(self, item: StreamInput, image: np.ndarray | Callable[[], np.ndarray]) -> list[FrameState]:
        """Take the next frame, and return the states that are final now (at most one, ``f = t - L``).

        ``image`` is the picture of the frame, or a function that reads it: then the reading is
        the stage ``read`` of the frame sample.
        """
        if self._finished:
            raise ValueError("the stream is finished")
        if item.frame != self._n:
            raise ValueError(f"frame {self._n} was expected, got frame {item.frame}")
        if item.camera.frame != item.frame:
            raise ValueError(f"the camera of frame {item.frame} names frame {item.camera.frame}")
        if any(ball.frame != item.frame for ball in item.balls):
            raise ValueError(f"a ball candidate of frame {item.frame} names another frame")
        for name, seconds in (("detector", item.detector_seconds), ("calibration", item.calibration_seconds)):
            if seconds is not None:
                self.tool_seconds[name].append(seconds)
        clock = self._clock
        frame = item.frame
        given: list[FrameState] = []
        with clock.frame():
            if callable(image):
                with clock.stage("read"):
                    image = image()
            with clock.stage("motion"):
                self._arrive_motion(image, item.people)
            if frame % self._keyframe_step == 0 and item.camera.valid:
                with clock.stage("paint"):
                    self._painter.submit(frame, image, matrix_from_record(item.camera))
            with clock.stage("camera"):
                self._arrive_camera(frame, item, image)
            everyone = FramePeople(frame=frame, people=item.people)
            on_pitch = _people_on_pitch(everyone, self._arrival_to_pitch, self._config.pitch_margin_m)
            with clock.stage("kit_colour"):
                if self._history is None:
                    # The whole clip is matched again in finish, with the cameras of the whole clip, so every
                    # person keeps a team key; the kit clusters learn from the people on the pitch only.
                    kept = {id(person) for person in on_pitch.people}
                    keyed = self._team_keys(frame, image, everyone, [id(person) in kept for person in everyone.people])
                    self._keyed.append(keyed)
                    people = FramePeople(
                        frame=frame,
                        people=tuple(
                            key for key, person in zip(keyed.people, everyone.people, strict=True) if id(person) in kept
                        ),
                    )
                else:
                    people = self._team_keys(frame, image, on_pitch, [True] * len(on_pitch.people))
            with clock.stage("tracks"):
                if frame == 0 or frame in self._hard_cuts:
                    self._associators.append(
                        Associator(
                            fps=self._fps,
                            warp=_bridged(self._motion, self._arrival),
                            to_pitch=self._arrival_to_pitch,
                            config=self._tracking,
                            keyframe_frames=self._keyframe_frames,
                        )
                    )
                self._associators[-1].add(people)
            if item.balls:
                self._balls[frame] = item.balls
            self._n += 1
            delay = self._settings.delay_frames
            if delay is not None and frame - delay >= 0:
                with clock.stage("decide"):
                    given.append(self._decide(frame - delay, frame, final=False))
                self._forget(frame - delay + 1)
        return given

    def _arrive_motion(self, image: np.ndarray, people: Sequence[PersonDet]) -> None:
        meter = self._meter
        meter.add(image, [person.box for person in people])
        frame = meter.n_frames - 1
        if frame == 0:
            self._motion.add_frame(None)
            return
        if meter.hard_cuts and meter.hard_cuts[-1] == frame:
            self._hard_cuts.append(frame)
        self._motion.add_frame(meter.steps[frame - 1])
        transitions = meter.update_transitions()
        self._log_transitions("final", transitions, frame)
        windows = self._settings.early_dissolve_windows
        early = None if windows is None else meter.open_transition(windows)
        if early is not None:
            self._log_transitions("early", [early], frame)
            transitions = [*transitions, early]
        self._transitions = transitions
        self._motion.cuts = self._cuts()

    def _log_transitions(self, kind: str, transitions: Sequence[tuple[int, int]], frame: int) -> None:
        known = {(logged_kind, span) for logged_kind, span, _ in self.transition_log}
        self.transition_log.extend((kind, span, frame) for span in transitions if (kind, span) not in known)

    def _cuts(self) -> tuple[int, ...]:
        """Return the frames that start a new shot, as far as they are known (``ShotTests.cuts``)."""
        frames = set(self._hard_cuts)
        for start, stop in self._transitions:
            frames.update(range(start, stop + 1))
        return tuple(sorted(frames))

    def _arrive_camera(self, frame: int, item: StreamInput, image: np.ndarray) -> None:
        """Check the camera of the frame, and give the frame the camera that the matching uses."""
        record = item.camera
        self._last_image, self._last_camera = image, record
        if record.valid:
            on_pitch = [person for person in item.people if person.role != _ROLE_NEXT_TO_PITCH]
            problems = fit_problems(record, _feet(on_pitch), self._size, pitch_margin_m=self._config.pitch_margin_m)
            if problems:
                logger.info("frame %d: the camera of the calibration is refused: %s", frame, "; ".join(problems))
            else:
                self._kept[frame] = matrix_from_record(record)
                self._kept_records[frame] = record
        matrix = self._camera_of(frame, last=frame)
        if matrix is not None:
            self._arrival[frame] = matrix
        else:
            self._blind.add(frame)

    def _camera_of(self, frame: int, *, last: int) -> np.ndarray | None:
        """Return the camera of a frame from the plausible cameras of the frames up to ``last``.

        The rule of ``motion.propagate_homographies``: the frame's own camera, else the camera
        of the nearest frame within ``max_gap_frames`` that the motion chain reaches (the earlier
        one at equal distance), moved with the camera motion.
        """
        own = self._kept.get(frame)
        if own is not None:
            return own
        reach = self._config.max_gap_frames
        for distance in range(1, reach + 1):
            for key in (frame - distance, frame + distance):
                if key > last or key not in self._kept:
                    continue
                to_key = self._motion.between(frame, key)
                if to_key is not None:
                    return self._kept[key] @ to_key
        return None

    def _arrival_to_pitch(self, frame: int, xy: np.ndarray) -> np.ndarray | None:
        matrix = self._arrival.get(frame)
        return None if matrix is None else apply_homography(matrix, xy)

    def _team_keys(self, frame: int, image: np.ndarray, people: FramePeople, learn: Sequence[bool]) -> FramePeople:
        """Give each person the team key of the kit clusters (module docstring, "Kit clusters").

        The clusters learn from the people whose ``learn`` is true.
        """
        features = [torso_feature(image, person.box) for person in people.people]
        if self._given_kits is None:
            warmup = self._settings.kit_warmup_frames
            if frame < warmup or self._clusters is None:
                self._features.extend(
                    (frame, feature)
                    for feature, taken in zip(features, learn, strict=True)
                    if feature is not None and taken
                )
                if frame >= warmup:
                    self._features = [(seen, feature) for seen, feature in self._features if seen > frame - warmup]
                self._clusters = _fit(self._features)
            elif self._features:
                self._features = []
        clusters = self._clusters
        if clusters is None:
            return people
        return FramePeople(
            frame=frame,
            people=tuple(
                person.model_copy(update={"team_key": clusters.team_key(feature)})
                for person, feature in zip(people.people, features, strict=True)
            ),
        )

    # --- Decision ------------------------------------------------------------------------------

    def finish(self) -> list[FrameState]:
        """End the stream: decide every frame that is not decided yet, with all frames known."""
        if self._finished:
            raise ValueError("the stream is finished")
        if self._n == 0:
            raise ValueError("the stream has no frame")
        self._finished = True
        last = self._n - 1
        if last % self._keyframe_step != 0 and self._last_camera is not None and self._last_camera.valid:
            assert self._last_image is not None
            self._painter.submit(last, self._last_image, matrix_from_record(self._last_camera))
        self._last_image = None
        self._paint = self._painter.finish()
        tests = self._meter.finish()
        self._transitions = list(tests.transitions)
        self._log_transitions("final", self._transitions, last)
        self._motion.cuts = self._cuts()
        if self._history is None:
            self._match_again(last)
        given: list[FrameState] = []
        cache: dict[str, Any] = {}
        for frame in range(len(self._states), self._n):
            with self._clock.stage("decide"):
                given.append(self._decide(frame, last, final=True, cache=cache))
        return given

    def _match_again(self, last: int) -> None:
        """Match the whole clip again with the cameras of the whole clip, as the clip path does (history None only).

        At arrival a frame without a camera of its own takes the nearest earlier camera, and the
        first frames of a shot have none: their people were matched without pitch positions. The
        clip path takes the camera of a later frame too (second read of 2026-11-23). Here the
        people of the blend frames are dropped, the people off the pitch are dropped with the
        cameras of the whole clip, and a new matching starts at frame 0 and at each hard cut.
        """
        blends = self._blends(last, final=True)
        matrices: dict[int, np.ndarray] = {}
        for index in range(last + 1):
            matrix = None if index in blends else self._camera_of(index, last=last)
            if matrix is not None:
                matrices[index] = matrix

        def to_pitch(frame: int, xy: np.ndarray) -> np.ndarray | None:
            matrix = matrices.get(frame)
            return None if matrix is None else apply_homography(matrix, xy)

        warp = _bridged(self._motion, matrices)
        margin = self._config.pitch_margin_m
        self._associators = []
        for keyed in self._keyed:
            frame = keyed.frame
            people = FramePeople(frame=frame, people=()) if frame in blends else keyed
            if frame == 0 or frame in self._hard_cuts:
                self._associators.append(
                    Associator(
                        fps=self._fps,
                        warp=warp,
                        to_pitch=to_pitch,
                        config=self._tracking,
                        keyframe_frames=self._keyframe_frames,
                    )
                )
            self._associators[-1].add(_people_on_pitch(people, to_pitch, margin))
        self._blind = set()

    def _blends(self, last: int, *, final: bool) -> frozenset[int]:
        """Return the frames that form a shot of one frame, as far as known at frame ``last``."""
        cuts = self._motion.cuts
        if not cuts:
            return frozenset()
        starts = (0, *cuts)
        ends: tuple[int | None, ...] = (*cuts, last + 1 if final else None)
        return frozenset(start for start, end in zip(starts, ends, strict=True) if end is not None and end - start == 1)

    def _window(self, frame: int, last: int, *, final: bool) -> dict[str, Any]:
        """Run the steps of the clip path on the window of frames that the decision of ``frame`` at ``last`` sees."""
        first = 0 if self._history is None else max(0, frame - self._history)
        blends = self._blends(last, final=final)
        cuts = self._motion.cuts
        records: dict[int, HomographyRecord] = {}
        matrices: dict[int, np.ndarray] = {}
        for index in range(first, last + 1):
            matrix = None if index in blends else self._camera_of(index, last=last)
            if matrix is None:
                records[index] = HomographyRecord(frame=index)
            elif index in self._kept and index in self._kept_records:
                records[index] = self._kept_records[index]
                matrices[index] = matrix
            else:
                records[index] = HomographyRecord(frame=index, matrix=matrix_to_rows(matrix), source="propagated")
                matrices[index] = matrix
        to_pitch = self._memo.pitch_map(matrices)
        warp = _bridged(self._motion, matrices)

        def point(observation: TrackObs) -> tuple[float, float] | None:
            return self._memo.point(matrices, observation)

        def off_pitch(observation: TrackObs) -> bool:
            """Tell whether a person of a frame without a camera at arrival maps outside the pitch now.

            The clip path drops such a person before the matching (``pipeline._people_on_pitch``);
            the stream could not, so the window drops the observation.
            """
            if observation.frame not in self._blind or observation.frame not in matrices:
                return False
            found = point(observation)
            margin = self._config.pitch_margin_m
            return found is None or not bool(inside_pitch(np.array(found, dtype=np.float64), margin)[0])

        pieces: dict[int, list[Track]] = {}
        for associator in self._associators:
            for fragment in associator.tracks(since=first):
                for piece in _cut(fragment, cuts, blends, off_pitch):
                    pieces.setdefault(self._motion.shot_of(piece.first_frame), []).append(piece)
        has_kits = any(
            observation.team_key in ("A", "B")
            for shot_pieces in pieces.values()
            for piece in shot_pieces
            for observation in piece.observations
        )
        tracks: list[Track] = []
        for shot in sorted(pieces):
            ordered = sorted(pieces[shot], key=lambda piece: (piece.first_frame, piece.track_id))
            fragments = [piece.model_copy(update={"track_id": index}) for index, piece in enumerate(ordered)]
            if has_kits:
                fragments = kit_roles(fragments, _mean_points(fragments, point)[0])
            if self._tracking.split_team_swaps:
                fragments = split_team_swaps(fragments, min_votes=SWAP_MIN_VOTES * self._keyframe_frames)
            for track in stitch_fragments(fragments, fps=self._fps, to_pitch=to_pitch, config=self._tracking):
                tracks.append(track.model_copy(update={"track_id": len(tracks)}))
        decided = finalize_tracks(
            tracks,
            readings={},
            mean_pitch_x=_mean_points(tracks, point)[1],
            fps=self._fps,
            to_pitch=to_pitch,
            config=self._config.tracking,
            roster=None,
            cuts=cuts,
        )
        balls = {index: found for index, found in self._balls.items() if index >= first and index not in blends}
        ball = track_ball(
            balls, last + 1, fps=self._fps, warp=warp, max_gap_frames=self._ball_gap, cuts=cuts, first_frame=first
        )
        return {
            "records": records,
            "to_pitch": to_pitch,
            "warp": warp,
            "decided": decided,
            "ball": ball,
            "cuts": cuts,
        }

    def _decide(self, frame: int, last: int, *, final: bool, cache: dict[str, Any] | None = None) -> FrameState:
        """Decide the state of ``frame`` with what is known at frame ``last``, give it, and return it.

        ``cache`` holds the window between decisions at the same ``last`` (the end of the
        stream), when the window is the whole clip.
        """
        if cache is not None and self._history is None:
            window = cache.get("window")
            if window is None:
                window = cache["window"] = self._window(frame, last, final=final)
        else:
            window = self._window(frame, last, final=final)
        decided: list[Track] = window["decided"]
        players = players_in_frame(
            decided,
            list(window["records"].values()),
            frame,
            n_frames=last + 1,
            warp=window["warp"],
            image_size=self._size,
            max_fill_gap_frames=self._max_fill,
            pitch_margin_m=self._config.pitch_margin_m,
            cuts=window["cuts"],
        )
        present = {state.track_id for state in players}
        ids = self._ids.ids([track for track in decided if track.track_id in present], frame)
        players = tuple(state.model_copy(update={"track_id": ids[state.track_id]}) for state in players)
        for state in players:
            self._ids.note(frame, state)
        ball = ball_state(window["ball"], frame, window["to_pitch"], self._config.pitch_margin_m)
        state = FrameState(frame=frame, players=players, ball=ball)
        self._states.append(state)
        self._records.append(window["records"][frame])
        return state

    def _forget(self, next_frame: int) -> None:
        """Forget what no later decision can need: the window of ``next_frame`` starts at ``next_frame - H``."""
        if self._history is None:
            return
        first = max(0, next_frame - self._history)
        for associator in self._associators:
            associator.forget_before(first)
        newest = self._associators[-1]
        self._associators = [
            associator
            for associator in self._associators
            if associator is newest or (associator.last_frame is not None and associator.last_frame >= first)
        ]
        self._ids.forget_before(first)
        self._memo.forget_before(first)
        reach = first - self._config.max_gap_frames
        for store in (self._kept, self._kept_records):
            for index in [index for index in store if index < reach]:
                del store[index]
        # The matching warps from the newest observation of a live track, which lies at most
        # max_missed_keyframes frames (of every_frame_tracking) before the newest frame.
        live = self._n - 1 - self._tracking.max_missed_keyframes - 1
        keep = max(0, min(reach, live))
        for index in [index for index in self._arrival if index < keep]:
            del self._arrival[index]
        self._blind = {index for index in self._blind if index >= first}
        for index in [index for index in self._balls if index < first]:
            del self._balls[index]
        self._motion.forget_before(keep)
        self._meter.forget_images_before(first)

    # --- Result --------------------------------------------------------------------------------

    def result(self) -> ClipResult:
        """Return the result of the finished stream: the states as they were given, and the tracks they form."""
        if not self._finished:
            raise ValueError("the stream is not finished")
        info = ClipInfo(
            clip_id=self._clip_id, n_frames=self._n, fps=self._fps, width=self._size[0], height=self._size[1]
        )
        engines = {**_engines(self._config), "tracks": "code:stream", "fill": "code:stream"}
        engines["delay"] = frames_text(self._settings.delay_frames, self._fps)
        engines["history"] = frames_text(self._settings.history_frames, self._fps)
        return ClipResult(
            info=info,
            frames=tuple(self._states),
            tracks=tuple(tracks_of_states(self._states)),
            homographies=tuple(self._records),
            engines=engines,
            cuts=self._motion.cuts,
            paint=tuple(sorted(self._paint, key=lambda check: check.frame)),
        )


def frames_text(frames: int | None, fps: float) -> str:
    """Write a number of frames with its time, or "whole clip" for None."""
    return "whole clip" if frames is None else f"{frames} frames ({frames / fps:.2f} s)"


def _fit(features: Sequence[tuple[int, np.ndarray]]) -> KitClusters | None:
    """Fit two kit clusters on some torso features, or return None when they show no two kits."""
    known = [feature for _, feature in features]
    if len(known) < 2 or len(np.unique(np.stack(known), axis=0)) < 2:
        return None
    clusters = fit_kit_clusters(np.stack(known), seed=KIT_CLUSTER_SEED)
    return clusters if clusters.two_kits() else None


def _cut(
    fragment: Track, cuts: Sequence[int], blends: frozenset[int], dropped: Callable[[TrackObs], bool]
) -> list[Track]:
    """Cut a fragment at the cuts that it crosses and drop its observations on blend frames and those of ``dropped``.

    The matching learns a dissolve only after it, so a fragment can run through one.
    """
    pieces: list[list[TrackObs]] = []
    shot: int | None = None
    for observation in fragment.observations:
        if observation.frame in blends or dropped(observation):
            continue
        here = bisect.bisect_right(cuts, observation.frame)
        if shot != here:
            pieces.append([])
            shot = here
        pieces[-1].append(observation)
    return [fragment.model_copy(update={"observations": tuple(piece)}) for piece in pieces if piece]


def tracks_of_states(states: Sequence[FrameState]) -> list[Track]:
    """Return the tracks that the given states form: per id, the observed states (all states when none is observed).

    The role, team key and team side are those of the last state of the id.
    """
    by_id: dict[int, list[tuple[int, PlayerState]]] = {}
    for state in states:
        for player in state.players:
            by_id.setdefault(player.track_id, []).append((state.frame, player))
    tracks: list[Track] = []
    for track_id in sorted(by_id):
        items = by_id[track_id]
        seen = [(frame, player) for frame, player in items if player.observed and player.box is not None]
        chosen = seen or [(frame, player) for frame, player in items if player.box is not None]
        if not chosen:
            continue
        last = items[-1][1]
        observations = tuple(
            TrackObs(frame=frame, box=player.box, role=player.role, team_key=player.team_key)
            for frame, player in chosen
            if player.box is not None
        )
        tracks.append(
            Track(
                track_id=track_id,
                observations=observations,
                role=last.role,
                team_key=last.team_key,
                team_side=last.team_side,
            )
        )
    return tracks


def run_stream(
    inputs: Iterable[StreamInput],
    read_image: Callable[[int], np.ndarray],
    config: PipelineConfig,
    settings: StreamSettings,
    *,
    clip_id: str,
    fps: float,
    size: tuple[int, int],
    kit_clusters: KitClusters | None = None,
    clock: StageClock | None = None,
) -> ClipResult:
    """Run a whole stream: push every input with its picture (``read_image(frame)``), finish, and return the result."""
    stream = DetectorStream(
        config, settings, clip_id=clip_id, fps=fps, size=size, kit_clusters=kit_clusters, clock=clock
    )
    for item in inputs:
        stream.push(item, lambda frame=item.frame: read_image(frame))
    stream.finish()
    return stream.result()
