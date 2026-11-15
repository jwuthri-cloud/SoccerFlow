"""Dense output: the state of every person in every frame.

The model sees people at keyframes only. This module fills the frames in
between: pitch positions by linear interpolation in pitch space, and image
boxes by a blend of the two keyframe boxes after camera-motion compensation.
A gap that is too long is not filled: the track has no state there. A frame in
which the box of the person lies outside the image is not filled either.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pitchvlm.geometry import apply_homography, matrix_from_record
from pitchvlm.motion import MotionWarp
from pitchvlm.pitch import inside_pitch
from pitchvlm.schemas import Box, FrameState, HomographyRecord, PlayerState, Role, Track, TrackObs


def _matrices_by_frame(homographies: Sequence[HomographyRecord], n_frames: int) -> dict[int, np.ndarray]:
    """Return the image-to-pitch matrix of every frame that has a valid mapping."""
    matrices: dict[int, np.ndarray] = {}
    seen: set[int] = set()
    for record in homographies:
        if record.frame >= n_frames:
            raise ValueError(f"a homography record belongs to frame {record.frame}, the clip has {n_frames} frames")
        if record.frame in seen:
            raise ValueError(f"frame {record.frame} has more than one homography record")
        seen.add(record.frame)
        if record.valid:
            matrices[record.frame] = matrix_from_record(record)
    return matrices


def _anchors(track: Track, matrices: Mapping[int, np.ndarray]) -> tuple[list[int], np.ndarray]:
    """Return the frames and pitch positions of the observations that have a mapping.

    The pitch position is the foot point of the box, mapped through the
    homography of the frame. The positions have shape (k, 2). An observation is
    left out when its frame has no mapping or when its foot point maps to
    infinity.
    """
    frames: list[int] = []
    positions: list[np.ndarray] = []
    for observation in track.observations:
        matrix = matrices.get(observation.frame)
        if matrix is None:
            continue
        position = apply_homography(matrix, np.array([observation.box.foot]))[0]
        if np.isfinite(position).all():
            frames.append(observation.frame)
            positions.append(position)
    return frames, np.array(positions, dtype=np.float64).reshape(-1, 2)


def _filled_frames(anchor_frames: Sequence[int], max_fill_gap_frames: int) -> np.ndarray:
    """Return the frames in which a track gets a state, in increasing order.

    ``anchor_frames`` are the frames of the observations that have a pitch
    position, in increasing order; there is at least one. The result holds
    every anchor frame. It holds the frames between two anchors that follow
    each other only when the two are at most ``max_fill_gap_frames`` frames
    apart. Inside a longer gap nobody saw the person for too long, so these
    frames are left out.
    """
    frames = [anchor_frames[0]]
    for before, after in zip(anchor_frames, anchor_frames[1:]):
        if after - before <= max_fill_gap_frames:
            frames.extend(range(before + 1, after))
        frames.append(after)
    return np.array(frames, dtype=np.int64)


def _box_values(box: Box) -> np.ndarray:
    """Return a box as the array ``[x1, y1, x2, y2]``."""
    return np.array([box.x1, box.y1, box.x2, box.y2], dtype=np.float64)


def _warp_box(box: Box, src: int, dst: int, warp: MotionWarp) -> np.ndarray | None:
    """Move a box from frame ``src`` into frame ``dst`` and return ``[x1, y1, x2, y2]``.

    The four corners are moved, and the box around the moved corners is
    returned. The answer is None when the box cannot be moved: the motion chain
    is broken, or the moved corners are not finite or span no area.
    """
    corners = np.array([[box.x1, box.y1], [box.x2, box.y1], [box.x2, box.y2], [box.x1, box.y2]], dtype=np.float64)
    moved = warp(src, dst, corners)
    if moved is None:
        return None
    moved = np.asarray(moved, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(moved).all():
        return None
    low, high = moved.min(axis=0), moved.max(axis=0)
    if not (high > low).all():
        return None
    return np.concatenate([low, high])


def _blend_boxes(before: TrackObs, after: TrackObs, frame: int, warp: MotionWarp) -> Box:
    """Return the box of a person in a frame between two observations.

    Both observed boxes are moved into the frame with the camera motion and are
    then mixed by time. When one of them cannot be moved, the two observed
    boxes are mixed as they are.
    """
    weight = (frame - before.frame) / (after.frame - before.frame)
    start = _warp_box(before.box, before.frame, frame, warp)
    end = _warp_box(after.box, after.frame, frame, warp)
    if start is None or end is None:
        start, end = _box_values(before.box), _box_values(after.box)
    x1, y1, x2, y2 = ((1.0 - weight) * start + weight * end).tolist()
    return Box(x1=x1, y1=y1, x2=x2, y2=y2)


def _box_in_frame(
    observations: Sequence[TrackObs], observation_frames: Sequence[int], frame: int, warp: MotionWarp
) -> tuple[Box, bool]:
    """Return the box of a track in one frame, and whether the model saw the person there.

    ``frame`` must lie between the first and the last observation (both included).
    """
    index = bisect.bisect_right(observation_frames, frame) - 1
    before = observations[index]
    if before.frame == frame:
        return before.box, True
    return _blend_boxes(before, observations[index + 1], frame, warp), False


def _touches_image(box: Box, image_size: tuple[int, int]) -> bool:
    """Tell whether a part of the box lies inside an image of ``(width, height)`` pixels."""
    width, height = image_size
    return box.x2 > 0.0 and box.y2 > 0.0 and box.x1 < width and box.y1 < height


def _track_attributes(track: Track, role: Role) -> dict[str, Any]:
    """Return the fields that all states of one track share."""
    return {
        "track_id": track.track_id,
        "role": role,
        "team_key": "none" if track.team_key is None else track.team_key,
        "team_side": track.team_side,
        "number_export": track.number_export,
        "number_display": track.number_display,
        "name": track.name,
    }


def _track_states(
    track: Track,
    role: Role,
    matrices: Mapping[int, np.ndarray],
    warp: MotionWarp,
    image_size: tuple[int, int],
    max_fill_gap_frames: int,
    pitch_margin_m: float,
) -> list[tuple[int, PlayerState]]:
    """Return the states of one track as (frame, state) pairs, in frame order."""
    anchor_frames, anchor_xy = _anchors(track, matrices)
    if not anchor_frames:
        return []
    frames = _filled_frames(anchor_frames, max_fill_gap_frames)
    positions = np.column_stack(
        [np.interp(frames, anchor_frames, anchor_xy[:, 0]), np.interp(frames, anchor_frames, anchor_xy[:, 1])]
    )
    keep = inside_pitch(positions, pitch_margin_m)
    attributes = _track_attributes(track, role)
    observation_frames = [observation.frame for observation in track.observations]
    states: list[tuple[int, PlayerState]] = []
    for frame, (pitch_x, pitch_y) in zip(frames[keep].tolist(), positions[keep].tolist()):
        box, observed = _box_in_frame(track.observations, observation_frames, frame, warp)
        if observed or _touches_image(box, image_size):
            state = PlayerState(**attributes, box=box, pitch_x=pitch_x, pitch_y=pitch_y, observed=observed)
            states.append((frame, state))
    return states


def dense_frame_states(
    tracks: Sequence[Track],
    homographies: Sequence[HomographyRecord],
    n_frames: int,
    *,
    warp: MotionWarp,
    image_size: tuple[int, int],
    max_fill_gap_frames: int,
    pitch_margin_m: float = 3.0,
) -> list[FrameState]:
    """Build the state of every frame from keyframe tracks.

    The answer has one ``FrameState`` per frame, in frame order. The players of
    a frame are in the order of ``tracks``.

    For each track:

    - Position. Every observation whose frame has a valid mapping gives a pitch
      position: the foot point of the box through the homography of that frame.
      (A foot point that maps to infinity gives none.) The frames between two
      such observations that follow each other get a position by linear
      interpolation in pitch space, when the two are at most
      ``max_fill_gap_frames`` frames apart. Inside a longer gap the track has
      no state: the scorer counts a prediction at a wrong place harder than a
      missing one. There is no state before the first or after the last such
      observation.
    - Box. At an observation the box is the observed box. Between two
      observations, both boxes are moved into the frame with ``warp`` (camera
      motion) and are then mixed by time. When ``warp`` returns None, or a
      moved box that is not finite, the two boxes are mixed without the move.
    - A frame without an observation gets no state when this box lies fully
      outside the image: the person left the image and came back, and the
      labels have nobody there. The gap limit cannot tell this case from a
      person whom the model missed for some keyframes; the box can.
      ``image_size`` is ``(width, height)`` in pixels. A state at an
      observation is always kept.
    - ``observed`` is True only in the frames where the track has an observation.
    - The role, team, numbers and name come from the track. A track without a
      decided role is skipped.
    - A state whose position is more than ``pitch_margin_m`` outside the pitch
      is dropped.

    ``homographies`` holds at most one record per frame. Records are needed at
    the observation frames only. ``max_fill_gap_frames`` is a number of frames,
    so the caller makes it from a time and the frame rate of the clip. It has
    no default for that reason. With 0 no frame is filled: only the
    observations that have a pitch position give a state.
    """
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if max_fill_gap_frames < 0:
        raise ValueError("max_fill_gap_frames must not be negative")
    if not pitch_margin_m >= 0.0:
        raise ValueError("pitch_margin_m must not be negative")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must be (width, height) with positive values, got {image_size}")
    matrices = _matrices_by_frame(homographies, n_frames)
    players: list[list[PlayerState]] = [[] for _ in range(n_frames)]
    seen_ids: set[int] = set()
    for track in tracks:
        if track.role is None:
            continue
        if track.track_id in seen_ids:
            raise ValueError(f"track id {track.track_id} appears more than once")
        seen_ids.add(track.track_id)
        if track.last_frame >= n_frames:
            raise ValueError(
                f"track {track.track_id} has an observation in frame {track.last_frame}, the clip has {n_frames} frames"
            )
        states = _track_states(track, track.role, matrices, warp, image_size, max_fill_gap_frames, pitch_margin_m)
        for frame, state in states:
            players[frame].append(state)
    return [FrameState(frame=frame, players=tuple(states)) for frame, states in enumerate(players)]
