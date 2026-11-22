"""The pipeline of one clip: model requests, pitch mapping, tracks, shirt numbers and dense output.

``run_clip`` is the entry point. It asks a model backend four kinds of
stateless questions (kit legend, people, pitch landmarks, shirt numbers) and
does everything else with plain code. The backend can be the real API, a cache
in front of it, or the fake backend of the tests.

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

from pitchvlm.config import PipelineConfig, TrackingConfig
from pitchvlm.fill import dense_frame_states
from pitchvlm.frames import FrameSource, keyframe_indices
from pitchvlm.geometry import (
    apply_homography,
    check_homography,
    count_near,
    fit_from_landmarks,
    landmark_pairs,
    matrix_from_record,
)
from pitchvlm.motion import MotionField, MotionWarp, bridged_warp, compute_motion_field, propagate_homographies
from pitchvlm.pitch import inside_pitch
from pitchvlm.roster import Roster
from pitchvlm.schemas import (
    TEAM_ROLES,
    Box,
    ClipInfo,
    ClipResult,
    FrameLandmarks,
    FramePeople,
    HomographyRecord,
    KitLegend,
    NumberReading,
    PersonDet,
    TeamKey,
    Track,
    TrackObs,
)
from pitchvlm.tracking import (
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


def run_clip(
    source: FrameSource,
    backend: Backend,
    config: PipelineConfig = PipelineConfig(),
    *,
    roster: Roster | None = None,
    motion: MotionField | None = None,
) -> ClipResult:
    """Run the whole pipeline on one clip.

    The steps are:

    1. Keyframes: one frame every ``config.keyframe_interval_s`` seconds, plus the last frame.
    2. Kit legend: one model request with the first, the middle and the last keyframe.
    3. People and pitch landmarks: one model request each per keyframe. A box that a
       people answer repeats, also with a small shift for the same role and team key, is dropped.
    4. Camera motion between all frames. ``motion`` is used when it is given
       (for example a stored field). Else the motion is measured from the frames.
    5. Pitch mapping: a landmark fit per keyframe, rejected when it is not
       plausible, then carried to the other frames with the camera motion.
    6. People who map too far outside the pitch are dropped.
    7. Tracks: the people of the keyframes are linked, and broken tracks are joined. Across a
       missing step of the camera motion the people are followed through the pitch mappings
       of the two frames (``motion.bridged_warp``).
    8. Shirt numbers: one model request per track of a player or a goalkeeper. No request
       is sent when no frame of the clip has a pitch mapping.
    9. Votes: role, team, team side, shirt number and roster name of each track. Tracks of
       one person who was out of the image become one track when they show the same
       confident shirt number and nothing says that two people show it
       (``tracking.finalize_tracks``).
    10. Dense output: the state of every person in every frame. A track has no state
        inside a gap between two of its positions that is longer than
        ``config.tracking.max_stitch_gap_s``, the limit of the joins of step 7.

    ``roster`` gives the player names by team side and shirt number. All model
    requests carry ``config.repeat``. A request that fails and an answer that
    cannot be parsed raise an error. Nothing is skipped.
    """
    info = source.info
    if motion is not None and motion.n_frames != info.n_frames:
        raise ValueError(f"the motion field has {motion.n_frames} frames, the clip has {info.n_frames} frames")
    keyframes = keyframe_indices(info.n_frames, info.fps, config.keyframe_interval_s)
    legend = _kit_legend(source, backend, keyframes, config)
    people, landmarks = _people_and_landmarks(source, backend, keyframes, legend, config)
    field = motion if motion is not None else _measured_motion(source, people)
    homographies = _pitch_mapping(landmarks, people, field, info, config)
    to_pitch = _pitch_map(homographies)
    warp = bridged_warp(field, homographies)
    on_pitch = [_people_on_pitch(frame_people, to_pitch, config.pitch_margin_m) for frame_people in people]
    tracks = _tracks(on_pitch, warp, to_pitch, info.fps, config.tracking)
    # Without any pitch mapping no track gets a state, so a shirt number would be paid for nothing.
    has_mapping = any(record.valid for record in homographies)
    readings = _shirt_numbers(source, backend, tracks, legend, config) if has_mapping else {}
    decided = finalize_tracks(
        tracks,
        readings=readings,
        mean_pitch_x=_mean_pitch_x(tracks, to_pitch),
        fps=info.fps,
        to_pitch=to_pitch,
        config=config.tracking,
        roster=roster,
    )
    frames = dense_frame_states(
        decided,
        homographies,
        info.n_frames,
        warp=warp,
        image_size=(info.width, info.height),
        max_fill_gap_frames=round(config.tracking.max_stitch_gap_s * info.fps),
        pitch_margin_m=config.pitch_margin_m,
    )
    return ClipResult(
        info=info,
        frames=tuple(frames),
        tracks=tuple(decided),
        homographies=tuple(homographies),
        engines=_engines(config),
    )


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


def _keyframe_requests(
    source: FrameSource, keyframes: Sequence[int], legend: KitLegend, config: PipelineConfig
) -> list[tuple[VLMRequest, VLMRequest]]:
    """Build the people request and the landmarks request of every keyframe.

    The frames are read here, in the calling thread.
    """
    clip_id = source.info.clip_id
    pairs: list[tuple[VLMRequest, VLMRequest]] = []
    for frame in keyframes:
        image = _read_frame(source, frame)
        people_request = build_people_request(
            image, legend, clip_id=clip_id, frame=frame, settings=config.people, repeat=config.repeat
        )
        landmarks_request = build_landmarks_request(
            image, clip_id=clip_id, frame=frame, settings=config.landmarks, repeat=config.repeat
        )
        pairs.append((people_request, landmarks_request))
    return pairs


def _people_and_landmarks(
    source: FrameSource, backend: Backend, keyframes: Sequence[int], legend: KitLegend, config: PipelineConfig
) -> tuple[list[FramePeople], list[FrameLandmarks]]:
    """Ask the model for the people and for the pitch landmarks of every keyframe.

    Both result lists are in keyframe order. The two requests of a keyframe
    are sent next to each other, so that a fault in one of the two steps shows
    at the start of a run. A person whose box repeats the box of another
    person of the same answer is dropped, and a warning names their number.
    """
    info = source.info
    pairs = _keyframe_requests(source, keyframes, legend, config)
    answers = _generate_all(backend, [request for pair in pairs for request in pair], config.max_parallel)
    people: list[FramePeople] = []
    landmarks: list[FrameLandmarks] = []
    n_repeated = 0
    for frame, (people_request, landmarks_request), people_answer, landmarks_answer in zip(
        keyframes, pairs, answers[0::2], answers[1::2], strict=True
    ):
        with _answer_of(people_request):
            answered = parse_people(
                people_answer.text, frame=frame, width=info.width, height=info.height, coords=config.people.coords
            )
        kept = _without_repeated_boxes(answered)
        n_repeated += len(answered.people) - len(kept.people)
        people.append(kept)
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


def _keyframe_fits(
    landmarks: Sequence[FrameLandmarks], people: Sequence[FramePeople], info: ClipInfo, config: PipelineConfig
) -> tuple[dict[int, HomographyRecord], set[int]]:
    """Fit the pitch mapping of every keyframe from its landmarks, and reject the fits that are not plausible.

    The result is the record of every keyframe and the frames whose fit is
    well supported. A rejected fit gives a record without a matrix, like a
    keyframe without any fit. A warning says so when no keyframe of the clip
    has a mapping, because the clip then gets no pitch position at all, and
    ``run_clip`` then sends no shirt-number request.
    """
    fits: dict[int, HomographyRecord] = {}
    strong_frames: set[int] = set()
    n_rejected = 0
    for frame_landmarks, frame_people in zip(landmarks, people, strict=True):
        record = fit_from_landmarks(frame_landmarks, ransac_px=config.ransac_px)
        if record.valid:
            feet = _feet([person for person in frame_people.people if person.role != _ROLE_NEXT_TO_PITCH])
            problems = fit_problems(record, feet, (info.width, info.height), pitch_margin_m=config.pitch_margin_m)
            if problems:
                logger.info("keyframe %d: the landmark fit is rejected: %s", record.frame, "; ".join(problems))
                n_rejected += 1
                record = HomographyRecord(frame=record.frame)
            elif _well_supported(record):
                strong_frames.add(record.frame)
        fits[frame_landmarks.frame] = record
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
    camera motion. The result has one record per frame.
    """
    fits, strong_frames = _keyframe_fits(landmarks, people, info, config)
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
    people: Sequence[FramePeople], warp: MotionWarp, to_pitch: PitchMap, fps: float, config: TrackingConfig
) -> list[Track]:
    """Link the people of the keyframes into tracks, then drop short tracks and join broken ones.

    ``warp`` moves image points from one frame into another. Where it gives no
    answer, the tracks end and the joins of ``stitch_fragments`` connect them again.
    With ``config.split_team_swaps`` a track is first cut where it took over a person
    of the other team (``tracking.split_team_swaps``), and the joins link the pieces.
    """
    fragments = associate(people, fps=fps, warp=warp, to_pitch=to_pitch, config=config)
    if config.split_team_swaps:
        fragments = split_team_swaps(fragments)
    return stitch_fragments(fragments, fps=fps, to_pitch=to_pitch, config=config)


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


def _crop_plans(tracks: Sequence[Track], fps: float, max_crops: int) -> list[_CropPlan]:
    """Plan the crops of every track that is voted player or goalkeeper."""
    plans: list[_CropPlan] = []
    for track in tracks:
        role, team_key = vote_role_team(track)
        if role in TEAM_ROLES:
            plans.append(_CropPlan(track.track_id, team_key, _crop_observations(track, fps, max_crops)))
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
    source: FrameSource, backend: Backend, tracks: Sequence[Track], legend: KitLegend, config: PipelineConfig
) -> dict[int, NumberReading]:
    """Ask the model for the shirt number of every track that is voted player or goalkeeper.

    The result maps a track id to the reading of that track. The crops are cut
    and enlarged in the calling thread; the requests go through the thread pool.
    """
    plans = _crop_plans(tracks, source.info.fps, config.number_crops)
    crops = _cut_crops(source, plans)
    requests = [_number_request(plan, crops, legend, source.info, config) for plan in plans]
    answers = _generate_all(backend, requests, config.max_parallel)
    readings: dict[int, NumberReading] = {}
    for plan, request, answer in zip(plans, requests, answers, strict=True):
        with _answer_of(request):
            readings[plan.track_id] = parse_number(answer.text)
    return readings


# --- Steps 9 and 10: inputs of the votes, and the engine table ---------------------


def _pitch_x(observation: TrackObs, to_pitch: PitchMap) -> float | None:
    """Return the pitch x of the foot point of an observation, or None when it has no finite pitch position."""
    position = to_pitch(observation.frame, np.array([observation.box.foot], dtype=np.float64))
    if position is None or not np.isfinite(position).all():
        return None
    return float(position[0, 0])


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


def _engines(config: PipelineConfig) -> dict[str, str]:
    """Name what does each step: ``model:<model id>`` for a model step, ``code`` for plain code."""
    return {
        "kit_legend": f"model:{config.kit_legend.model}",
        "people": f"model:{config.people.model}",
        "landmarks": f"model:{config.landmarks.model}",
        "number": f"model:{config.number.model}",
        "motion": "code",
        "pitch_mapping": "code",
        "tracks": "code",
        "fill": "code",
    }
