"""The pipeline of one clip: model requests, pitch mapping, tracks, shirt numbers and dense output.

``run_clip`` is the entry point. It asks a model backend four kinds of
stateless questions (kit legend, people, pitch landmarks, shirt numbers) and
does everything else with plain code. The backend can be the real API, a cache
in front of it, or the fake backend of the tests.

Two steps have a second engine (``config.engines``). The people engine
``detector`` takes the people of every frame from the answers of a detector
(``ToolAnswers.people``), gives them a team key and a role from the kit colours
(``kitcolour``) and follows the ball (``ball``). The pitch engine
``calibration`` takes the camera of every frame from the answers of a
calibration tool (``ToolAnswers.cameras``). The answers of a tool are read from
its clip run file by the caller: this package never imports a tool.

Frames are read and requests are built in the calling thread. Only the calls
to the backend run in a thread pool.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
import numpy as np

from pitchvlm.ball import BallCandidate, track_ball
from pitchvlm.config import PipelineConfig, StepSettings, TrackingConfig, model_label
from pitchvlm.fill import dense_frame_states
from pitchvlm.frames import FrameSource, keyframe_indices
from pitchvlm.geometry import (
    apply_homography,
    check_homography,
    count_near,
    fit_from_landmarks,
    landmark_pairs,
    matrix_from_record,
    paint_alignment,
)
from pitchvlm.kitcolour import fit_kit_clusters, kit_roles, torso_feature
from pitchvlm.motion import MotionField, MotionWarp, bridged_warp, compute_motion_field, propagate_homographies
from pitchvlm.pitch import inside_pitch
from pitchvlm.roster import Roster
from pitchvlm.schemas import (
    TEAM_ROLES,
    BallState,
    Box,
    ClipInfo,
    ClipResult,
    FrameLandmarks,
    FramePeople,
    FrameState,
    HomographyRecord,
    KitLegend,
    NumberReading,
    PaintCheck,
    PersonDet,
    TeamKey,
    Track,
    TrackObs,
)
from pitchvlm.timing import StageClock
from pitchvlm.tracking import (
    SWAP_MIN_VOTES,
    PitchMap,
    associate,
    finalize_tracks,
    split_team_swaps,
    stitch_fragments,
    vote_role_team,
)
from pitchvlm.vlm.steps import (
    ParseError,
    build_kit_legend_request,
    build_landmarks_request,
    build_number_request,
    build_people_request,
    crop_rect,
    parse_kit_legend,
    parse_landmarks,
    parse_number,
    parse_people,
)
from pitchvlm.vlm.types import Backend, VLMRequest, VLMResponse

logger = logging.getLogger(__name__)

#: Smallest time between two crops of one track, seconds. Crops far apart show the person in other poses.
MIN_CROP_GAP_S = 1.0
#: People with this role stand next to the pitch (coaches, substitutes, staff), not on it.
_ROLE_NEXT_TO_PITCH = "other"
#: Two boxes of one people answer whose IoU is above this show the same person: the answer repeats a box.
REPEATED_BOX_MIN_IOU = 0.9
#: Two boxes of one people answer with the same role and the same team key show the same person
#: already when their IoU is above this: a model that boxes one person two times mostly gives two
#: boxes that differ a little. Two boxes with other labels can be two people in a duel. The value
#: is an estimate until the people probe shows how the boxes of real answers overlap.
REPEATED_BOX_SAME_LABELS_MIN_IOU = 0.6
#: A landmark fit is well supported when at least this many landmarks agree with it. Measured
#: on synthetic clips with 4 pixels of landmark noise: a fit from 4 to 6 agreeing landmarks is
#: often some metres off, and the mapping that the camera motion carries over from a
#: well-supported keyframe is better.
STRONG_FIT_MIN_INLIERS = 7
#: A fit may put this many people outside the pitch before a share rule of the plausibility
#: check applies. So one person next to the pitch whom the model names a player, or one box
#: where nobody is, does not reject a fit.
MAX_PEOPLE_OUTSIDE = 1
#: A well-supported fit must put at least this share of the people on the pitch. Its landmarks
#: say more than some people who warm up behind a line. But a fit that puts more than a third
#: of the people outside the pitch is wrong, however many landmarks agree with it.
STRONG_FIT_MIN_SHARE_ON_PITCH = 2.0 / 3.0
#: The ball tracker fills a gap of at most this many seconds between two sightings of the ball.
#: An estimate: a short occlusion by a player, not a ball that left the picture.
BALL_MAX_GAP_S = 0.5
#: Seed of the k-means of the kit colours, so that a run gives the same team keys again.
KIT_CLUSTER_SEED = 0


@dataclass(frozen=True)
class ToolAnswers:
    """The answers of external tools for every frame of a clip, read from their clip run files.

    ``people`` and ``balls`` come from the detector (people engine ``detector``), ``cameras``
    from the calibration (pitch engine ``calibration``). Each holds one entry per frame of
    the clip: ``people[i]`` and ``cameras[i]`` belong to frame ``i``, and ``balls`` maps a
    frame to its ball candidates.
    """

    people: Sequence[FramePeople] | None = None
    balls: Mapping[int, Sequence[BallCandidate]] | None = None
    cameras: Sequence[HomographyRecord] | None = None


def run_clip(
    source: FrameSource,
    backend: Backend | None,
    config: PipelineConfig = PipelineConfig(),
    *,
    roster: Roster | None = None,
    motion: MotionField | None = None,
    tools: ToolAnswers = ToolAnswers(),
    clock: StageClock | None = None,
) -> ClipResult:
    """Run the whole pipeline on one clip.

    The steps of the model engines are:

    1. Keyframes: one frame every ``config.keyframe_interval_s`` seconds, plus the last frame.
    2. Kit legend: one model request with the first, the middle and the last keyframe.
    3. People and pitch landmarks: one model request each per keyframe. A box that a
       people answer repeats, also with a small shift for the same role and team key, is dropped.
    4. Camera motion between all frames. ``motion`` is used when it is given
       (for example a stored field). Else the motion is measured from the frames.
       The motion step into a hard cut is missing, and the cuts are in the result.
    5. Pitch mapping: a landmark fit per keyframe, rejected when it is not
       plausible, then carried to the other frames with the camera motion. Every landmark
       fit gets the paint-alignment score of its keyframe (``geometry.paint_alignment``), a
       diagnostic that decides nothing.
    6. People who map too far outside the pitch are dropped.
    7. Tracks: the people of the keyframes are linked, and broken tracks are joined, in each
       shot on its own. Across a missing step of the camera motion inside a shot the people
       are followed through the pitch mappings of the two frames (``motion.bridged_warp``).
    8. Shirt numbers: one model request per track of a player or a goalkeeper. No request
       is sent when no frame of the clip has a pitch mapping.
    9. Votes: role, team, team side, shirt number and roster name of each track. Tracks of
       one person who was out of the image become one track when they show the same
       confident shirt number and nothing says that two people show it
       (``tracking.finalize_tracks``).
    10. Dense output: the state of every person in every frame. A track has no state
        inside a gap between two of its positions that is longer than
        ``config.tracking.max_stitch_gap_s``, the limit of the joins of step 7, and no
        state is filled across a cut.

    Nothing is carried across a cut: no pitch mapping, no track link by position or image
    motion, no filled state. Only a shown shirt number joins two tracks across a cut (step 9).

    The people engine ``detector`` changes steps 2, 3, 7 and 8 and adds a ball:

    - No people request: the detector gives the people of every frame. The kit legend is
      asked only for the number engine ``model``, whose prompt shows it.
    - Team key and role from the kit colours: a colour histogram of the torso of each box,
      two kit clusters of the clip and outliers (``kitcolour``).
    - Tracks: the people of every frame are linked, with the same limits in seconds as the
      keyframe design (``every_frame_tracking``). A track of mostly outlier kits becomes a
      goalkeeper or a referee by its pitch position (``kitcolour.kit_roles``) before the
      broken tracks are joined.
    - Shirt numbers only with the number engine ``model``. The number request names no
      team: a kit cluster is no team of the kit legend. Without it no track has a number.
    - Ball: the ball candidates of the detector are followed by the ball tracker
      (``ball.track_ball``), and each frame state gets the ball when it has one.
    - As in the model engine, each shot is tracked on its own, and no run of the ball
      crosses a cut.

    The pitch engine ``calibration`` replaces steps 4 and 5: no landmark request; the camera
    of every frame comes from the calibration tool, a camera that is not plausible with the
    people of its frame (``fit_problems``) is refused, and a frame without a camera takes
    the camera of the nearest frame that has one, moved with the camera motion (never across
    a cut). The camera of every keyframe gets its paint-alignment score, as a landmark fit does.

    ``roster`` gives the player names by team side and shirt number. All model
    requests carry ``config.repeat``. A request that fails and an answer that
    cannot be parsed raise an error. Nothing is skipped. ``backend`` may be None only
    when no step uses a model. ``ValueError`` is raised when an engine of ``config`` needs
    tool answers that ``tools`` does not hold, when ``tools`` holds answers that no engine
    uses, and when the answers do not cover the frames of the clip.

    ``clock`` gets the measured time of each stage (``timing.StageClock``).
    """
    info = source.info
    if motion is not None and motion.n_frames != info.n_frames:
        raise ValueError(f"the motion field has {motion.n_frames} frames, the clip has {info.n_frames} frames")
    _check_tool_answers(tools, config, info.n_frames)
    if backend is None and _uses_model(config):
        raise ValueError("the settings name a model step, so the pipeline needs a model backend")
    clock = clock if clock is not None else StageClock()
    engines = config.engines
    detector = engines.people == "detector"
    keyframes = keyframe_indices(info.n_frames, info.fps, config.keyframe_interval_s)
    legend = None
    if engines.people == "model" or engines.number == "model":
        with clock.stage("kit_legend"):
            legend = _kit_legend(source, _backend(backend), keyframes, config)
    model_people: list[FramePeople] = []
    landmarks: list[FrameLandmarks] = []
    if not detector or engines.pitch_mapping == "model":
        with clock.stage("model_people_landmarks"):
            model_people, landmarks = _model_answers(source, _backend(backend), keyframes, legend, config)
    people = list(tools.people or ()) if detector else model_people
    with clock.stage("motion"):
        field = motion if motion is not None else _measured_motion(source, people)
    with clock.stage("pitch_mapping"):
        if engines.pitch_mapping == "model":
            landmark_fits = [
                fit_from_landmarks(frame_landmarks, ransac_px=config.ransac_px) for frame_landmarks in landmarks
            ]
            by_frame = {frame_people.frame: frame_people for frame_people in people}
            keyframe_people = [by_frame[frame] for frame in keyframes]
            homographies = _pitch_mapping(landmarks, landmark_fits, keyframe_people, field, info, config)
        else:
            cameras = tools.cameras or ()
            landmark_fits = [cameras[frame] for frame in keyframes]
            homographies = _camera_mapping(cameras, people, field, info, config)
    with clock.stage("paint"):
        paint = _paint_checks(source, landmark_fits)
    to_pitch = _pitch_map(homographies)
    warp = bridged_warp(field, homographies)
    on_pitch = [_people_on_pitch(frame_people, to_pitch, config.pitch_margin_m) for frame_people in people]
    if detector:
        with clock.stage("kit_colour"):
            on_pitch = _kit_team_keys(source, on_pitch)
        with clock.stage("tracks"):
            tracks = _detector_tracks(
                on_pitch,
                warp,
                to_pitch,
                info.fps,
                every_frame_tracking(config, info.fps),
                field,
                keyframe_frames=keyframe_frames(config, info.fps),
            )
    else:
        with clock.stage("tracks"):
            tracks = _tracks(on_pitch, warp, to_pitch, info.fps, config.tracking, field)
    # Without any pitch mapping no track gets a state, so a shirt number would be paid for nothing.
    has_mapping = any(record.valid for record in homographies)
    readings: dict[int, NumberReading] = {}
    if engines.number == "model" and has_mapping and legend is not None:
        with clock.stage("number"):
            readings = _shirt_numbers(source, _backend(backend), tracks, legend, config, name_team=not detector)
    with clock.stage("votes_fill"):
        decided = finalize_tracks(
            tracks,
            readings=readings,
            mean_pitch_x=_mean_pitch_x(tracks, to_pitch),
            fps=info.fps,
            to_pitch=to_pitch,
            config=config.tracking,
            roster=roster,
            cuts=field.cuts,
        )
        frames = dense_frame_states(
            decided,
            homographies,
            info.n_frames,
            warp=warp,
            image_size=(info.width, info.height),
            max_fill_gap_frames=round(config.tracking.max_stitch_gap_s * info.fps),
            pitch_margin_m=config.pitch_margin_m,
            cuts=field.cuts,
        )
    if detector:
        with clock.stage("ball"):
            frames = _with_ball(
                frames, tools.balls or {}, warp, to_pitch, info.fps, config.pitch_margin_m, cuts=field.cuts
            )
    return ClipResult(
        info=info,
        frames=tuple(frames),
        tracks=tuple(decided),
        homographies=tuple(homographies),
        engines=_engines(config),
        cuts=field.cuts,
        paint=tuple(paint),
    )


def every_frame_tracking(config: PipelineConfig, fps: float) -> TrackingConfig:
    """Return the tracking limits for people that are found on every frame.

    The limits of ``config.tracking`` were set for keyframes ``config.keyframe_interval_s``
    apart. A track ends after ``max_missed_keyframes`` keyframes without a match: on every
    frame this becomes the same time in frames, so a person may be hidden for as long as in
    the keyframe design. The limits in seconds, metres or box heights stay. Two other counts
    of keyframes take their time from ``keyframe_frames``: the growth of the label penalty of
    ``tracking.associate`` and the votes of a team-key change (``tracking.split_team_swaps``).
    ``min_observations`` stays a count, of frames here: a fragment of 2 frames (0.08 s at 25
    fps) is kept for the joins, where the keyframe design asks for 2 keyframes.
    """
    missed_s = config.tracking.max_missed_keyframes * config.keyframe_interval_s
    return config.tracking.model_copy(update={"max_missed_keyframes": round(missed_s * fps)})


def keyframe_frames(config: PipelineConfig, fps: float) -> int:
    """Return how many observations of a track stand for one keyframe interval: 1 for the people engine ``model``.

    The detector engine observes every frame, so one keyframe interval is
    ``config.keyframe_interval_s * fps`` frames (at least 1). A limit that counts
    observations and was set for keyframes is multiplied by this number to keep its time.
    """
    if config.engines.people != "detector":
        return 1
    return max(1, round(config.keyframe_interval_s * fps))


# --- Engines and tool answers ----------------------------------------------------


def _uses_model(config: PipelineConfig) -> bool:
    """Tell whether any step of the settings asks the model."""
    engines = config.engines
    return "model" in (engines.people, engines.pitch_mapping, engines.number)


def _backend(backend: Backend | None) -> Backend:
    """Return the backend of a model step. ``run_clip`` checked before that it exists."""
    if backend is None:
        raise ValueError("a model step needs a model backend")
    return backend


def _check_tool_answers(tools: ToolAnswers, config: PipelineConfig, n_frames: int) -> None:
    """Raise ``ValueError`` unless the tool answers are exactly those that the engines need, for every frame."""
    detector = config.engines.people == "detector"
    calibration = config.engines.pitch_mapping == "calibration"
    needed = {"people": detector, "balls": detector, "cameras": calibration}
    for name, is_needed in needed.items():
        answers = getattr(tools, name)
        if is_needed and answers is None:
            raise ValueError(f"the settings name the engine of a tool, but the tool answers have no {name}")
        if not is_needed and answers is not None:
            raise ValueError(f"the tool answers hold {name}, but no engine of the settings uses them")
    for name in ("people", "cameras"):
        answers = getattr(tools, name)
        if answers is not None and [item.frame for item in answers] != list(range(n_frames)):
            raise ValueError(f"the tool answers must hold the {name} of every frame from 0 to {n_frames - 1}, in order")
    if tools.balls is not None and any(not 0 <= frame < n_frames for frame in tools.balls):
        raise ValueError(f"the ball candidates name a frame outside the clip of {n_frames} frames")


# --- Frames and model calls ----------------------------------------------------


def _read_frame(source: FrameSource, frame: int) -> np.ndarray:
    """Read one frame and check that it is a BGR ``uint8`` image with the size of the clip."""
    image = source.read(frame)
    expected = (source.info.height, source.info.width, 3)
    if image.dtype != np.uint8 or image.shape != expected:
        raise ValueError(
            f"frame {frame} is a {image.dtype} image of shape {image.shape}, "
            f"the clip needs a uint8 image of shape {expected}"
        )
    return image


class _NotSent(Exception):
    """Marks a request that the thread pool did not send, because an earlier request had failed."""


def _generate_all(backend: Backend, requests: Sequence[VLMRequest], max_parallel: int) -> list[VLMResponse]:
    """Send requests to the backend through a thread pool. The answers are in the order of the requests.

    At most ``max_parallel`` requests are in flight at the same time. After
    the first failure no further request is sent, because a paid run that
    reaches its budget limit must stop spending. The requests in flight then
    finish, and the first failure is raised.

    An interruption (Ctrl-C or a stop signal) also ends here: the requests that
    have not started are cancelled, and the function returns only after every
    request in flight has finished. So the answers of these requests are stored
    and recorded before the caller goes on. A further interruption does not end
    this wait (``_finish_requests_in_flight``).
    """
    failures: list[Exception] = []
    workers: set[threading.Thread] = set()

    def generate(request: VLMRequest) -> VLMResponse:
        workers.add(threading.current_thread())
        if failures:
            raise _NotSent
        try:
            return backend.generate(request)
        except Exception as error:
            failures.append(error)
            raise

    executor = ThreadPoolExecutor(max_workers=max_parallel)
    try:
        futures = [executor.submit(generate, request) for request in requests]
        wait(futures)
    finally:
        _finish_requests_in_flight(executor, workers)
    if failures:
        raise failures[0]
    return [future.result() for future in futures]


def _finish_requests_in_flight(executor: ThreadPoolExecutor, workers: set[threading.Thread]) -> None:
    """Cancel the requests that have not started, and return only when every request in flight has finished.

    ``workers`` holds every thread that took a request. An interruption can
    come while the pool starts a thread. That thread then works, but the pool
    does not know it and does not wait for it. So these threads are joined
    here too.

    A stop signal raises ``KeyboardInterrupt`` in the calling thread, also
    while it waits here (an operator who presses Ctrl-C a second time). The
    answers of the requests in flight are paid, so the wait starts again, and
    a warning names the number of requests that are still in flight. The
    interruption is raised when all of them have finished.
    """
    interruption: KeyboardInterrupt | None = None
    while True:
        try:
            if interruption is not None:
                n_in_flight = sum(worker.is_alive() for worker in tuple(workers))
                logger.warning(
                    "a stop signal came while %d model requests were in flight; the command waits for their "
                    "answers, so that they are stored and recorded",
                    n_in_flight,
                )
            executor.shutdown(cancel_futures=True)
            for worker in tuple(workers):
                worker.join()
            break
        except KeyboardInterrupt as error:
            interruption = error
    if interruption is not None:
        raise interruption


@contextmanager
def _answer_of(request: VLMRequest) -> Iterator[None]:
    """Add the place of a request to the error when its answer cannot be parsed.

    The cache key names the stored answer, so that a bad answer can be found and removed.
    """
    try:
        yield
    except ParseError as error:
        meta = request.meta
        raise ParseError(
            f"the {meta.step} answer of clip {meta.clip_id} (frame {meta.frame}, item {meta.item}, "
            f"repeat {meta.repeat}, key {request.cache_key()}) cannot be used: {error}"
        ) from error


# --- Steps 2 and 3: kit legend, people and landmarks ------------------------------


def _kit_legend(
    source: FrameSource, backend: Backend, keyframes: Sequence[int], config: PipelineConfig
) -> KitLegend:
    """Ask the model for the kit colours of the clip.

    The request shows the first, the middle and the last keyframe. A short
    clip can have fewer keyframes; each keyframe is sent one time only.
    """
    frames = dict.fromkeys((keyframes[0], keyframes[len(keyframes) // 2], keyframes[-1]))
    request = build_kit_legend_request(
        [_read_frame(source, frame) for frame in frames],
        clip_id=source.info.clip_id,
        settings=config.kit_legend,
        repeat=config.repeat,
    )
    answer = backend.generate(request)
    with _answer_of(request):
        return parse_kit_legend(answer.text)


def _model_answers(
    source: FrameSource, backend: Backend, keyframes: Sequence[int], legend: KitLegend | None, config: PipelineConfig
) -> tuple[list[FramePeople], list[FrameLandmarks]]:
    """Ask the model for the people and for the pitch landmarks of every keyframe, as far as the engines ask for them.

    The people are asked with the people engine ``model`` and the landmarks with the pitch
    engine ``model``; a list that is not asked for is empty. Both result lists are in
    keyframe order. The frames are read and the requests are built in the calling thread.
    When both are asked, the two requests of a keyframe are sent next to each other, so that
    a fault in one of the two steps shows at the start of a run. A person whose box repeats
    the box of another person of the same answer is dropped, and a warning names their number.
    """
    info = source.info
    ask_people = config.engines.people == "model"
    ask_landmarks = config.engines.pitch_mapping == "model"
    requests: list[VLMRequest] = []
    for frame in keyframes:
        image = _read_frame(source, frame)
        if ask_people:
            if legend is None:
                raise ValueError("the people request needs the kit legend")
            requests.append(
                build_people_request(
                    image, legend, clip_id=info.clip_id, frame=frame, settings=config.people, repeat=config.repeat
                )
            )
        if ask_landmarks:
            requests.append(
                build_landmarks_request(
                    image, clip_id=info.clip_id, frame=frame, settings=config.landmarks, repeat=config.repeat
                )
            )
    answers = iter(zip(requests, _generate_all(backend, requests, config.max_parallel), strict=True))
    people: list[FramePeople] = []
    landmarks: list[FrameLandmarks] = []
    n_repeated = 0
    for frame in keyframes:
        if ask_people:
            people_request, people_answer = next(answers)
            with _answer_of(people_request):
                answered = parse_people(
                    people_answer.text, frame=frame, width=info.width, height=info.height, coords=config.people.coords
                )
            kept = _without_repeated_boxes(answered)
            n_repeated += len(answered.people) - len(kept.people)
            people.append(kept)
        if ask_landmarks:
            landmarks_request, landmarks_answer = next(answers)
            with _answer_of(landmarks_request):
                landmarks.append(
                    parse_landmarks(
                        landmarks_answer.text,
                        frame=frame,
                        width=info.width,
                        height=info.height,
                        coords=config.landmarks.coords,
                    )
                )
    if n_repeated:
        logger.warning(
            "the people answers of clip %s repeat %d boxes; each repeated box is dropped", info.clip_id, n_repeated
        )
    return people, landmarks


def _repeats(person: PersonDet, other: PersonDet) -> bool:
    """Tell whether two people of one people answer are the same person, by the overlap of their boxes.

    The IoU of the boxes must be above ``REPEATED_BOX_MIN_IOU``. For two
    people with the same role and the same team key the lower limit
    ``REPEATED_BOX_SAME_LABELS_MIN_IOU`` holds.
    """
    same_labels = (person.role, person.team_key) == (other.role, other.team_key)
    limit = REPEATED_BOX_SAME_LABELS_MIN_IOU if same_labels else REPEATED_BOX_MIN_IOU
    return person.box.iou(other.box) > limit


def _without_repeated_boxes(people: FramePeople) -> FramePeople:
    """Drop every person of a keyframe whose box repeats the box of a person that is kept.

    A model can list one person two times, with the same box or with a box
    that differs a little. The second box would become a second track of the
    same person. ``_repeats`` says when two boxes repeat each other. The
    person that the answer lists first is kept.
    """
    kept: list[PersonDet] = []
    for person in people.people:
        if not any(_repeats(person, other) for other in kept):
            kept.append(person)
    return FramePeople(frame=people.frame, people=tuple(kept))


# --- Steps 4 to 7: camera motion, pitch mapping and tracks -------------------------


def _measured_motion(source: FrameSource, people: Sequence[FramePeople]) -> MotionField:
    """Measure the camera motion between all frames. The people of the keyframes are masked out."""
    boxes = {frame_people.frame: [person.box for person in frame_people.people] for frame_people in people}
    return compute_motion_field(source, people=boxes)


def _feet(people: Sequence[PersonDet]) -> np.ndarray:
    """Return the foot points of people in image pixels, shape (n, 2)."""
    return np.array([person.box.foot for person in people], dtype=np.float64).reshape(-1, 2)


def _well_supported(record: HomographyRecord) -> bool:
    """Tell whether a landmark fit is supported well enough to be trusted on its own.

    At least ``STRONG_FIT_MIN_INLIERS`` landmarks must agree with the fit. A
    fit from fewer agreeing landmarks follows the noise of these few points.
    """
    return record.valid and record.n_inliers >= STRONG_FIT_MIN_INLIERS


def fit_problems(
    record: HomographyRecord, feet_xy: np.ndarray, image_size: tuple[int, int], *, pitch_margin_m: float
) -> list[str]:
    """List the reasons why the landmark fit of a frame is not plausible. An empty list means plausible.

    ``record`` is a landmark fit with a matrix. ``feet_xy`` holds the foot
    points of the people of the frame who stand on the pitch (players,
    goalkeepers and referees), in image pixels, shape (n, 2). ``image_size``
    is ``(width, height)``.

    Every fit must have the pitch the right way round in the image
    (``check_homography`` without foot points). Every fit may put
    ``MAX_PEOPLE_OUTSIDE`` people outside the pitch plus ``pitch_margin_m``.
    With more people outside, a share rule decides. A fit that is not well
    supported (``STRONG_FIT_MIN_INLIERS``) must put 90 percent of the people
    on the pitch, the share of ``check_homography``. A well-supported fit
    must put ``STRONG_FIT_MIN_SHARE_ON_PITCH`` of them on the pitch: its
    landmarks say more than some people who warm up behind the goal line, and
    the pipeline drops such people later anyway. But when more than a third
    of the people map outside the pitch, the fit is wrong.
    """
    matrix = matrix_from_record(record)
    problems = check_homography(matrix, image_size)
    if problems:
        return problems
    with np.errstate(invalid="ignore"):  # A foot point at infinity is outside the pitch.
        n_outside = int((~inside_pitch(apply_homography(matrix, feet_xy), pitch_margin_m)).sum())
    if n_outside <= MAX_PEOPLE_OUTSIDE:
        return []
    if _well_supported(record):
        return check_homography(
            matrix,
            image_size,
            feet_xy,
            pitch_margin_m=pitch_margin_m,
            min_inside_fraction=STRONG_FIT_MIN_SHARE_ON_PITCH,
        )
    return check_homography(matrix, image_size, feet_xy, pitch_margin_m=pitch_margin_m)


def _paint_checks(source: FrameSource, landmark_fits: Sequence[HomographyRecord]) -> list[PaintCheck]:
    """Score the mapping of every keyframe with the paint alignment of its frame (``geometry.paint_alignment``).

    ``landmark_fits`` holds the mapping of each keyframe before the plausibility check: the
    landmark fit of the pitch engine ``model``, or the camera of the calibration tool for the
    pitch engine ``calibration``. So both engines are scored on the same frames, and a
    refused mapping has a score too. A keyframe without a mapping has no entry. The score is
    a diagnostic: it decides nothing.
    """
    checks: list[PaintCheck] = []
    for record in landmark_fits:
        if record.valid:
            alignment = paint_alignment(_read_frame(source, record.frame), matrix_from_record(record))
            checks.append(PaintCheck(frame=record.frame, score=alignment.score, reason=alignment.reason))
    return checks


def _keyframe_fits(
    landmark_fits: Sequence[HomographyRecord], people: Sequence[FramePeople], info: ClipInfo, config: PipelineConfig
) -> tuple[dict[int, HomographyRecord], set[int]]:
    """Reject the landmark fits of the keyframes that are not plausible.

    ``landmark_fits`` holds the landmark fit of every keyframe, in keyframe order.
    The result is the record of every keyframe and the frames whose fit is
    well supported. A rejected fit gives a record without a matrix, like a
    keyframe without any fit. A warning says so when no keyframe of the clip
    has a mapping, because the clip then gets no pitch position at all, and
    ``run_clip`` then sends no shirt-number request.
    """
    fits: dict[int, HomographyRecord] = {}
    strong_frames: set[int] = set()
    n_rejected = 0
    for record, frame_people in zip(landmark_fits, people, strict=True):
        if record.valid:
            feet = _feet([person for person in frame_people.people if person.role != _ROLE_NEXT_TO_PITCH])
            problems = fit_problems(record, feet, (info.width, info.height), pitch_margin_m=config.pitch_margin_m)
            if problems:
                logger.info("keyframe %d: the landmark fit is rejected: %s", record.frame, "; ".join(problems))
                n_rejected += 1
                record = HomographyRecord(frame=record.frame)
            elif _well_supported(record):
                strong_frames.add(record.frame)
        fits[frame_people.frame] = record
    if not any(record.valid for record in fits.values()):
        logger.warning(
            "no keyframe of clip %s has a pitch mapping, so the clip gets no pitch position and no shirt-number "
            "request is sent: %d of %d keyframes gave a landmark fit that was rejected, and the other keyframes "
            "gave no fit",
            info.clip_id,
            n_rejected,
            len(fits),
        )
    return fits, strong_frames


def _landmarks_agree(landmarks: FrameLandmarks, record: HomographyRecord, ransac_px: float) -> bool:
    """Tell whether most landmarks of a keyframe are near a pitch mapping of that keyframe.

    A landmark is near the mapping when the mapping puts its pitch point at
    most ``geometry.NEAR_LIMIT_FACTOR`` times ``ransac_px`` from its image
    point (``geometry.count_near``): the limit that tells a noisy point from a
    wrong one in the landmark fit. Most means more than half of the landmarks
    that a fit can use.
    """
    image_xy, pitch_xy = landmark_pairs(landmarks)
    n_near = count_near(matrix_from_record(record), image_xy, pitch_xy, ransac_px=ransac_px)
    return 2 * n_near > len(image_xy)


def _pitch_mapping(
    landmarks: Sequence[FrameLandmarks],
    landmark_fits: Sequence[HomographyRecord],
    people: Sequence[FramePeople],
    motion: MotionField,
    info: ClipInfo,
    config: PipelineConfig,
) -> list[HomographyRecord]:
    """Give every frame a pitch mapping, as far as one can be found.

    A keyframe keeps its own landmark fit when the fit is plausible and well
    supported (``_well_supported``). For a keyframe with a weaker fit, or
    without a fit, the camera motion carries over the fit of the nearest
    well-supported keyframe within ``config.max_gap_frames``: the camera
    motion between two keyframes is known much better than a mapping from a
    few landmarks.

    A keyframe with a weaker fit takes the carried mapping only when most of
    its own landmarks are near it (``_landmarks_agree``). When they are not,
    the carried mapping is wrong at this keyframe: the well-supported fit is
    wrong, or it was carried far from the landmarks that fixed it. The
    keyframe then keeps its own fit. A keyframe without a fit takes the
    carried mapping. Without a well-supported keyframe in reach, a keyframe
    keeps what it has.

    Every other frame takes the mapping of a keyframe near it, moved with the
    camera motion. The result has one record per frame. No mapping is carried
    across a cut, because the motion step into a cut is missing.
    """
    fits, strong_frames = _keyframe_fits(landmark_fits, people, info, config)
    strong_fits = {frame: fits[frame] for frame in strong_frames}
    carried = propagate_homographies(strong_fits, motion, info.n_frames, max_gap_frames=config.max_gap_frames)
    key_fits: dict[int, HomographyRecord] = {}
    for frame_landmarks in landmarks:
        frame = frame_landmarks.frame
        record = fits[frame]
        if frame in strong_frames or not carried[frame].valid:
            key_fits[frame] = record
            continue
        if record.valid and not _landmarks_agree(frame_landmarks, carried[frame], config.ransac_px):
            logger.info(
                "keyframe %d: its landmarks do not agree with the mapping of a well-supported keyframe, "
                "so it keeps its own landmark fit (%d landmarks agree with it)",
                frame,
                record.n_inliers,
            )
            key_fits[frame] = record
            continue
        if record.valid:
            logger.info(
                "keyframe %d: only %d landmarks agree with the landmark fit, so the mapping of a "
                "well-supported keyframe is carried over with the camera motion",
                frame,
                record.n_inliers,
            )
        key_fits[frame] = carried[frame]
    return propagate_homographies(key_fits, motion, info.n_frames, max_gap_frames=config.max_gap_frames)


def _camera_mapping(
    cameras: Sequence[HomographyRecord],
    people: Sequence[FramePeople],
    motion: MotionField,
    info: ClipInfo,
    config: PipelineConfig,
) -> list[HomographyRecord]:
    """Give every frame the pitch mapping of the calibration tool, as far as it is plausible.

    ``cameras`` holds the mapping of the tool for every frame. ``people`` holds the people
    of the frames that have any (every frame for the detector, the keyframes for the model).
    A mapping is refused when ``fit_problems`` finds a problem with the people of its frame
    who stand on the pitch; a record of the tool has no count of agreeing landmarks, so the
    strict share rule holds (9 of 10 people on the pitch). A frame without a plausible
    mapping takes the mapping of the nearest frame that has one, within
    ``config.max_gap_frames``, moved with the camera motion (``propagate_homographies``).
    """
    people_of = {frame_people.frame: frame_people.people for frame_people in people}
    kept: dict[int, HomographyRecord] = {}
    n_refused = 0
    for record in cameras:
        if not record.valid:
            continue
        on_pitch = [person for person in people_of.get(record.frame, ()) if person.role != _ROLE_NEXT_TO_PITCH]
        problems = fit_problems(
            record, _feet(on_pitch), (info.width, info.height), pitch_margin_m=config.pitch_margin_m
        )
        if problems:
            logger.info("frame %d: the camera of the calibration is refused: %s", record.frame, "; ".join(problems))
            n_refused += 1
            continue
        kept[record.frame] = record
    if not kept:
        logger.warning(
            "no frame of clip %s has a plausible camera of the calibration (%d refused, %d without a camera), so "
            "the clip gets no pitch position",
            info.clip_id,
            n_refused,
            sum(not record.valid for record in cameras),
        )
    return propagate_homographies(kept, motion, info.n_frames, max_gap_frames=config.max_gap_frames)


def _pitch_map(homographies: Sequence[HomographyRecord]) -> PitchMap:
    """Build the function that maps image points of a frame to pitch metres.

    The function returns None for a frame without a pitch mapping.
    """
    matrices = {record.frame: matrix_from_record(record) for record in homographies if record.valid}

    def to_pitch(frame: int, xy: np.ndarray) -> np.ndarray | None:
        matrix = matrices.get(frame)
        if matrix is None:
            return None
        return apply_homography(matrix, xy)

    return to_pitch


def _people_on_pitch(people: FramePeople, to_pitch: PitchMap, margin_m: float) -> FramePeople:
    """Drop the people of a keyframe whose foot point maps more than ``margin_m`` outside the pitch.

    A keyframe without a pitch mapping keeps all its people. A foot point
    without a finite pitch position counts as outside.
    """
    positions = to_pitch(people.frame, _feet(people.people))
    if positions is None:
        return people
    inside = inside_pitch(positions, margin_m).tolist()
    kept = tuple(person for person, is_inside in zip(people.people, inside, strict=True) if is_inside)
    return FramePeople(frame=people.frame, people=kept)


def _tracks(
    people: Sequence[FramePeople],
    warp: MotionWarp,
    to_pitch: PitchMap,
    fps: float,
    config: TrackingConfig,
    motion: MotionField,
) -> list[Track]:
    """Link the people of the keyframes into tracks, then drop short tracks and join broken ones.

    ``warp`` moves image points from one frame into another. Where it gives no
    answer, the tracks end and the joins of ``stitch_fragments`` connect them again.
    Each shot of ``motion`` is tracked on its own, so no position links two people
    across a cut. The ids are 0..n-1 in the order of the shots and, inside a shot,
    in the order of the first frame. With ``config.split_team_swaps`` a track is first
    cut where it took over a person of the other team (``tracking.split_team_swaps``),
    and the joins of its shot link the pieces.
    """
    shots: dict[int, list[FramePeople]] = {}
    for frame_people in people:
        shots.setdefault(motion.shot_of(frame_people.frame), []).append(frame_people)
    tracks: list[Track] = []
    for shot in sorted(shots):
        fragments = associate(shots[shot], fps=fps, warp=warp, to_pitch=to_pitch, config=config)
        if config.split_team_swaps:
            fragments = split_team_swaps(fragments)
        for track in stitch_fragments(fragments, fps=fps, to_pitch=to_pitch, config=config):
            tracks.append(track.model_copy(update={"track_id": len(tracks)}))
    return tracks


def _kit_team_keys(source: FrameSource, people: Sequence[FramePeople]) -> list[FramePeople]:
    """Give every person of every frame the team key of the kit colour of their torso (``kitcolour``).

    The two kit clusters are fitted on the torso features of all people of the clip. A
    person whose kit fits neither cluster, or whose torso is too small for a feature, gets
    the team key ``none``. When the clip does not have two different kit colours (fewer than
    two different features, or two clusters that are not two kits: ``KitClusters.two_kits``),
    every person keeps the team key ``none`` and a warning says so.
    """
    features: list[list[np.ndarray | None]] = []
    for frame_people in people:
        if not frame_people.people:
            features.append([])
            continue
        image = _read_frame(source, frame_people.frame)
        features.append([torso_feature(image, person.box) for person in frame_people.people])
    known = [feature for frame_features in features for feature in frame_features if feature is not None]
    if len(known) < 2 or len(np.unique(np.stack(known), axis=0)) < 2:
        logger.warning(
            "clip %s has %d torso features and not two different kit colours: no person gets a team key",
            source.info.clip_id,
            len(known),
        )
        return list(people)
    clusters = fit_kit_clusters(np.stack(known), seed=KIT_CLUSTER_SEED)
    if not clusters.two_kits():
        logger.warning(
            "clip %s: the %d torso features form one kit colour, not two (the two clusters lie inside each "
            "other's outlier limits): no person gets a team key",
            source.info.clip_id,
            len(known),
        )
        return list(people)
    return [
        FramePeople(
            frame=frame_people.frame,
            people=tuple(
                person.model_copy(update={"team_key": clusters.team_key(feature)})
                for person, feature in zip(frame_people.people, frame_features, strict=True)
            ),
        )
        for frame_people, frame_features in zip(people, features, strict=True)
    ]


def _detector_tracks(
    people: Sequence[FramePeople],
    warp: MotionWarp,
    to_pitch: PitchMap,
    fps: float,
    config: TrackingConfig,
    motion: MotionField,
    *,
    keyframe_frames: int,
) -> list[Track]:
    """Link the people of every frame into tracks, give the tracks of outlier kits their role, then join broken tracks.

    The roles come before the joins: ``stitch_fragments`` joins only tracks with the same
    role and team key, so a goalkeeper must already be one. Each shot of ``motion`` is
    tracked on its own, as in ``_tracks``, so no position links two people across a cut.
    The ids are 0..n-1 in the order of the shots and, inside a shot, in the order of the
    first frame. With ``config.split_team_swaps`` a track is cut where it took over a
    person of the other kit cluster (``tracking.split_team_swaps``), after the kit roles
    (a goalkeeper or a referee has the team key ``none`` and is never cut) and before the
    joins of its shot, which link the pieces.

    ``keyframe_frames`` frames stand for one keyframe interval (``keyframe_frames``): the label
    penalty of ``associate`` grows per interval, and a team-key change must hold for
    ``SWAP_MIN_VOTES`` intervals of votes, the time of the keyframe design. With 4 votes, 4
    frames of a torso window that shows the shirt of an overlapping opponent would cut a track.

    Without kit clusters (no person of the clip has the team key ``A`` or ``B``, see
    ``_kit_team_keys``) no kit is an outlier, so no track gets a role from its kit: every
    track stays a player without a team key.
    """
    has_kits = any(person.team_key in ("A", "B") for frame_people in people for person in frame_people.people)
    shots: dict[int, list[FramePeople]] = {}
    for frame_people in people:
        shots.setdefault(motion.shot_of(frame_people.frame), []).append(frame_people)
    tracks: list[Track] = []
    for shot in sorted(shots):
        fragments = associate(
            shots[shot], fps=fps, warp=warp, to_pitch=to_pitch, config=config, keyframe_frames=keyframe_frames
        )
        with_roles = kit_roles(fragments, _mean_positions(fragments, to_pitch)) if has_kits else fragments
        if config.split_team_swaps:
            with_roles = split_team_swaps(with_roles, min_votes=SWAP_MIN_VOTES * keyframe_frames)
        for track in stitch_fragments(with_roles, fps=fps, to_pitch=to_pitch, config=config):
            tracks.append(track.model_copy(update={"track_id": len(tracks)}))
    return tracks


def _mean_positions(tracks: Sequence[Track], to_pitch: PitchMap) -> dict[int, tuple[float, float]]:
    """Return the mean pitch position (x, y) of every track that has one, by track id."""
    means: dict[int, tuple[float, float]] = {}
    for track in tracks:
        points = [_pitch_point(observation, to_pitch) for observation in track.observations]
        known = [point for point in points if point is not None]
        if known:
            means[track.track_id] = (
                math.fsum(x for x, _ in known) / len(known),
                math.fsum(y for _, y in known) / len(known),
            )
    return means


# --- Step 8: shirt numbers -----------------------------------------------------------


@dataclass(frozen=True)
class _CropPlan:
    """The observations of one track that are cut out for its shirt-number request."""

    track_id: int
    team_key: TeamKey
    #: In frame order.
    observations: tuple[TrackObs, ...]


def crop_choice(boxes: Sequence[tuple[int, Box]], fps: float, max_crops: int) -> tuple[tuple[int, Box], ...]:
    """Choose the boxes of one person that are cut out for a shirt-number request.

    ``boxes`` holds the frame and the box of every sighting of the person. No
    frame is in it twice. The result is in frame order. The pipeline and the
    number probe both choose their crops here, so the probe measures the crops
    that the pipeline sends.

    The tallest boxes are taken first (the earlier one wins a tie). A box is
    taken only when it is at least ``MIN_CROP_GAP_S`` seconds away from every
    box that is already taken: crops far apart show the person in other poses.
    At most ``max_crops`` boxes are taken.

    When this leaves one box, and there are more, the box that is furthest in
    time from it is taken too (the taller one wins a tie, then the earlier
    one). A number is shown in the video only when two crops agree, and
    fragments of one person get one id again only by a shown number. So a
    person who was seen in two keyframes or more must not get one crop only.
    """
    min_gap_frames = MIN_CROP_GAP_S * fps
    taken: list[tuple[int, Box]] = []
    for frame, box in sorted(boxes, key=lambda item: (-item[1].height, item[0])):
        if len(taken) == max_crops:
            break
        if all(abs(frame - other) >= min_gap_frames for other, _ in taken):
            taken.append((frame, box))
    if len(taken) == 1 and max_crops > 1 and len(boxes) > 1:
        first = taken[0][0]
        others = [item for item in boxes if item[0] != first]
        taken.append(min(others, key=lambda item: (-abs(item[0] - first), -item[1].height, item[0])))
    return tuple(sorted(taken, key=lambda item: item[0]))


def _crop_observations(track: Track, fps: float, max_crops: int) -> tuple[TrackObs, ...]:
    """Choose the observations of a track that go into its shirt-number request (``crop_choice``).

    The result is in frame order.
    """
    by_frame = {observation.frame: observation for observation in track.observations}
    chosen = crop_choice([(frame, observation.box) for frame, observation in by_frame.items()], fps, max_crops)
    return tuple(by_frame[frame] for frame, _ in chosen)


def _crop_plans(tracks: Sequence[Track], fps: float, max_crops: int, *, name_team: bool) -> list[_CropPlan]:
    """Plan the crops of every track that is voted player or goalkeeper.

    With ``name_team`` False the plan names no team (``none``): the team key of a track is
    then a kit cluster of the detector engine, not a team of the kit legend.
    """
    plans: list[_CropPlan] = []
    for track in tracks:
        role, team_key = vote_role_team(track)
        if role in TEAM_ROLES:
            plans.append(
                _CropPlan(track.track_id, team_key if name_team else "none", _crop_observations(track, fps, max_crops))
            )
    return plans


def _cut_crop(image: np.ndarray, box: Box) -> np.ndarray:
    """Cut the box of a person out of an image, with the padding added on every side (``steps.crop_rect``).

    The crop is limited to the image. It is a copy, so that the image can be freed.
    """
    height, width = image.shape[:2]
    rect = crop_rect(box, width, height)
    if rect is None:
        raise ValueError(f"the box ({box}) lies outside the image of {width} x {height} pixels")
    left, top, right, bottom = rect
    return image[top:bottom, left:right].copy()  # Rows are y, columns are x.


def _cut_crops(source: FrameSource, plans: Sequence[_CropPlan]) -> dict[tuple[int, int], np.ndarray]:
    """Cut the crops of all plans out of the frames. A crop is stored under (track id, frame).

    Each frame is read one time only, in the calling thread.
    """
    boxes_by_frame: dict[int, list[tuple[int, Box]]] = {}
    for plan in plans:
        for observation in plan.observations:
            boxes_by_frame.setdefault(observation.frame, []).append((plan.track_id, observation.box))
    crops: dict[tuple[int, int], np.ndarray] = {}
    for frame in sorted(boxes_by_frame):
        image = _read_frame(source, frame)
        for track_id, box in boxes_by_frame[frame]:
            crops[(track_id, frame)] = _cut_crop(image, box)
    return crops


def _enlarge(crop: np.ndarray, factor: float) -> np.ndarray:
    """Enlarge a crop by a factor, with cubic interpolation."""
    height, width = crop.shape[:2]
    size = (round(width * factor), round(height * factor))  # OpenCV wants (width, height).
    return cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC)


def _number_request(
    plan: _CropPlan,
    crops: Mapping[tuple[int, int], np.ndarray],
    legend: KitLegend,
    info: ClipInfo,
    config: PipelineConfig,
) -> VLMRequest:
    """Build the shirt-number request of one track from its enlarged crops.

    The request context names the frame and the box of the person in each
    crop. ``build_number_request`` makes the request item from the frames,
    the pixel areas and the enlargement of the crops, so that the cache keeps
    the answers for different crops apart. The request gets no track id: the
    id of a track is its place in the track list, and it changes with the
    tracking code and its settings. So the same crops find their stored
    answer again when their track has another id.
    """
    enlarged = [
        _enlarge(crops[(plan.track_id, observation.frame)], config.crop_upscale) for observation in plan.observations
    ]
    return build_number_request(
        enlarged,
        legend,
        plan.team_key,
        clip_id=info.clip_id,
        track_id=None,
        crop_frames=[observation.frame for observation in plan.observations],
        crop_boxes=[observation.box for observation in plan.observations],
        image_size=(info.width, info.height),
        crop_upscale=config.crop_upscale,
        settings=config.number,
        repeat=config.repeat,
    )


def _shirt_numbers(
    source: FrameSource,
    backend: Backend,
    tracks: Sequence[Track],
    legend: KitLegend,
    config: PipelineConfig,
    *,
    name_team: bool,
) -> dict[int, NumberReading]:
    """Ask the model for the shirt number of every track that is voted player or goalkeeper.

    The result maps a track id to the reading of that track. The crops are cut
    and enlarged in the calling thread; the requests go through the thread pool.
    ``name_team`` says whether the request names the team key of the track (``_crop_plans``).
    """
    plans = _crop_plans(tracks, source.info.fps, config.number_crops, name_team=name_team)
    crops = _cut_crops(source, plans)
    requests = [_number_request(plan, crops, legend, source.info, config) for plan in plans]
    answers = _generate_all(backend, requests, config.max_parallel)
    readings: dict[int, NumberReading] = {}
    for plan, request, answer in zip(plans, requests, answers, strict=True):
        with _answer_of(request):
            readings[plan.track_id] = parse_number(answer.text)
    return readings


# --- Steps 9 and 10: inputs of the votes, and the engine table ---------------------


def _pitch_point(observation: TrackObs, to_pitch: PitchMap) -> tuple[float, float] | None:
    """Return the pitch position of the foot point of an observation, or None when it has no finite one."""
    position = to_pitch(observation.frame, np.array([observation.box.foot], dtype=np.float64))
    if position is None or not np.isfinite(position).all():
        return None
    return float(position[0, 0]), float(position[0, 1])


def _pitch_x(observation: TrackObs, to_pitch: PitchMap) -> float | None:
    """Return the pitch x of the foot point of an observation, or None when it has no finite pitch position."""
    point = _pitch_point(observation, to_pitch)
    return None if point is None else point[0]


def _mean_pitch_x(tracks: Sequence[Track], to_pitch: PitchMap) -> dict[int, float]:
    """Return the mean pitch x of every track, by track id.

    The mean runs over the observations that have a pitch position. A track
    without any such observation is not in the result.
    """
    means: dict[int, float] = {}
    for track in tracks:
        values = [_pitch_x(observation, to_pitch) for observation in track.observations]
        known = [value for value in values if value is not None]
        if known:
            means[track.track_id] = math.fsum(known) / len(known)
    return means


def _with_ball(
    frames: Sequence[FrameState],
    candidates: Mapping[int, Sequence[BallCandidate]],
    warp: MotionWarp,
    to_pitch: PitchMap,
    fps: float,
    margin_m: float,
    *,
    cuts: Sequence[int],
) -> list[FrameState]:
    """Follow the ball (``ball.track_ball``) and give each frame state the ball of its frame.

    No run of the ball crosses a cut (``cuts``, the frames that start a new shot). The pitch position of the ball is the foot point of its box. It is left out where the
    frame has no mapping, and where it lies more than ``margin_m`` outside the pitch: a ball
    in the air maps far beyond where it is.
    """
    track = track_ball(
        candidates, len(frames), fps=fps, warp=warp, max_gap_frames=round(BALL_MAX_GAP_S * fps), cuts=cuts
    )
    result: list[FrameState] = []
    for state in frames:
        box = track.boxes.get(state.frame)
        if box is None:
            result.append(state)
            continue
        position = to_pitch(state.frame, np.array([box.foot], dtype=np.float64))
        pitch: dict[str, float] = {}
        if position is not None and np.isfinite(position).all() and bool(inside_pitch(position, margin_m)[0]):
            pitch = {"pitch_x": float(position[0, 0]), "pitch_y": float(position[0, 1])}
        ball = BallState(box=box, observed=state.frame in track.observed, **pitch)
        result.append(state.model_copy(update={"ball": ball}))
    return result


def _model_engine(settings: StepSettings) -> str:
    """Name the engine of a model step: its provider and its model (``model_label``)."""
    return f"model:{model_label(settings.provider, settings.model)}"


def _engines(config: PipelineConfig) -> dict[str, str]:
    """Name what does each step.

    ``model:<model>`` for a model step, with the model as the command line writes it
    (``model_label``: the bare name for Gemini, ``openrouter:<slug>`` for OpenRouter),
    ``detector:<tool>:<model class>`` and ``tool:<tool>`` for the answers of an external
    tool, ``code`` for plain code, and ``none`` for a step that does not run.
    """
    engines = config.engines
    detector = engines.people == "detector"
    asks_legend = not detector or engines.number == "model"
    landmarks = engines.pitch_mapping == "model"
    people = f"detector:{config.detector.name}:{config.detector.model}" if detector else _model_engine(config.people)
    return {
        "kit_legend": _model_engine(config.kit_legend) if asks_legend else "none",
        "people": people,
        "team_role": "code:kit-colour" if detector else _model_engine(config.people),
        "landmarks": _model_engine(config.landmarks) if landmarks else "none",
        "number": _model_engine(config.number) if engines.number == "model" else "none",
        "ball": "code:ball-tracker" if detector else "none",
        "motion": "code",
        "pitch_mapping": "code" if landmarks else f"tool:{config.calibration.name}",
        "tracks": "code",
        "fill": "code",
    }
