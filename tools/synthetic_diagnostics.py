"""Measure the paint-alignment score and the cut test on synthetic clips, and print the numbers.

This is the measurement behind the synthetic numbers of ``docs/DECISIONS.md`` (entry "Paint
alignment of pitch mappings, and hard cuts" of 2026-11-22, items 5 and 6). It runs offline on
the build machine: no model, no footage, no network. Every number it prints is synthetic, not a
result on broadcast frames. Run it from the top folder of the repository:

    PYTHONPATH=src python tools/synthetic_diagnostics.py

Paint alignment (``geometry.paint_alignment``) on 50 frames (every 5th frame of a 250-frame
synthetic clip, seed 0):

- pitch shifts of the true mapping by 0, 0.5, 1, 2, 3, 5 and 10 m, in 8 directions;
- landmark fits (``geometry.fit_from_landmarks``) of the true landmarks with Gaussian point noise
  of 2, 4, 8 and 16 pixels (noise seed 7). Their error is the median pitch distance of the true
  foot points, as in the pitch probe;
- the image mirrored left to right and top to bottom, the pitch mirrored, and the sign of one
  matrix entry changed.

The separation is the AUROC of the score between mappings with an error of at most 2 m and
mappings with an error above 3 m (``probes.auroc``), as the run-machine measurement counts it.

Cut test (``motion.compute_motion_field``): the smallest median block correlation inside a shot
and the largest across a cut, on clips with a slow camera, a fast pan and cuts, with and without
a static score graphic; and the known limits: a short jump in time while the camera barely
moves, pans faster than 83 px per frame, and a dissolve (two shots blended over some frames).

The time of one paint-alignment score is measured on this machine (``time.perf_counter``).

Detector engine (entry "A real-time detector engine next to the model engine", item 5), with the
stand-ins of the detector and the calibration (``fallbacks.fake``), seed 3: the identity values
of a 250-frame clip without and with the team slots, and a clip with a cut at frame 54 (tracks
across the cut, paint scores of the keyframe cameras).

Dissolves (entry "A dissolve is a transition, not a shot" of 2026-11-22): the window test of
``motion.measure_shots`` against the pair test of neighbouring frames (the baseline), on
dissolves of 5 to 25 frames, hard cuts, and controls that must stay without a cut: pans up to
100 px per frame, a fast zoom, a camera shake, a goal-kick flight with a quick tilt, and clips
without a cut. The thresholds were fixed on seeds 0 to 9 (section ``dissolve-fit``, which also
prints the shot shares behind ``MIN_SHOT_SHARE``); seeds 10 to 19 were run once after that
(section ``dissolve-held-out``, with the transitions that stay out of scope and the time of the
cut test per frame).

Run one section or several by name; without a name every section runs:

    PYTHONPATH=src python tools/synthetic_diagnostics.py dissolve-fit dissolve-held-out
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from pitchvlm.config import EngineSettings, PipelineConfig, TrackingConfig
from pitchvlm.evalx.metrics import identity_scores
from pitchvlm.fallbacks.fake import fake_calibration_answers, fake_detector_answers
from pitchvlm.frames import FrameSource
from pitchvlm.geometry import (
    apply_homography,
    fit_from_landmarks,
    matrix_from_record,
    paint_alignment,
)
from pitchvlm.motion import (
    DISSOLVE_WINDOW_FRAMES,
    MAX_TRANSITION_CORRELATION,
    MIN_SHOT_CORRELATION,
    ShotTests,
    _chain,
    _correlations,
    _shot_share,
    _small_grey,
    _test_image,
    _test_size,
    _to_small_matrix,
    estimate_motion,
    measure_shots,
    shot_correlation,
    window_correlations,
)
from pitchvlm.pipeline import ToolAnswers, run_clip
from pitchvlm.probes import auroc, auroc_standard_error
from pitchvlm.schemas import ClipInfo, FrameLandmarks, LandmarkObs
from pitchvlm.synth import (
    RetimedScene,
    ScoreGraphicSource,
    SyntheticScene,
    SyntheticSource,
)

SHIFTS_M = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
DIRECTIONS = 8
NOISE_PX = (2.0, 4.0, 8.0, 16.0)
NOISE_SEED = 7
GOOD_MAX_ERROR_M = 2.0
BAD_MIN_ERROR_M = 3.0


def median_error_m(scene: SyntheticScene, frame: int, matrix: np.ndarray) -> float:
    """Return the median pitch error of the true foot points of a frame under a mapping."""
    players = [player for player in scene.frame_state(frame).players if player.box.y2 < scene.info.height - 1]
    feet = np.array([player.box.foot for player in players])
    truth = np.array([[player.pitch_x, player.pitch_y] for player in players])
    return float(np.median(np.linalg.norm(apply_homography(matrix, feet) - truth, axis=1)))


def noisy_fit(scene: SyntheticScene, frame: int, noise_px: float, rng: np.random.Generator) -> np.ndarray | None:
    """Fit the mapping of a frame from its true landmarks with Gaussian point noise. None when no fit."""
    truth = scene.landmarks(frame)
    noisy = tuple(
        LandmarkObs(
            landmark_id=item.landmark_id, x=item.x + rng.normal(0.0, noise_px), y=item.y + rng.normal(0.0, noise_px)
        )
        for item in truth.landmarks
    )
    record = fit_from_landmarks(FrameLandmarks(frame=frame, view=truth.view, landmarks=noisy))
    return matrix_from_record(record) if record.valid else None


def separation(label: str, scored: Sequence[tuple[float | None, float]]) -> None:
    """Print the AUROC of the scores of good against bad mappings. ``scored`` holds (score, error in metres)."""
    good = [score for score, error in scored if score is not None and error <= GOOD_MAX_ERROR_M]
    bad = [score for score, error in scored if score is not None and error > BAD_MIN_ERROR_M]
    area = auroc(good, bad)
    n_none = sum(1 for score, _ in scored if score is None)
    print(
        f"{label}: auroc {area:.3f} +- {auroc_standard_error(area, len(good), len(bad)):.3f} "
        f"(n good {len(good)}, n bad {len(bad)}, cannot judge {n_none}); score median good "
        f"{np.median(good):.3f}, bad {np.median(bad):.3f}"
    )


def measure_paint() -> None:
    """Print the paint-alignment numbers of the synthetic clip."""
    scene = SyntheticScene(n_frames=250, seed=0)
    frames = range(0, 250, 5)
    rng = np.random.default_rng(NOISE_SEED)
    width, height = scene.info.width, scene.info.height
    by_shift: dict[float, list[float | None]] = {shift: [] for shift in SHIFTS_M}
    shifted: list[tuple[float | None, float]] = []
    fitted: list[tuple[float | None, float]] = []
    fitted_by_noise: dict[float, list[tuple[float | None, float]]] = {noise_px: [] for noise_px in NOISE_PX}
    variants: dict[str, list[float | None]] = {}
    seconds: list[float] = []
    mirror_image = np.array([[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    flip_image = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, height - 1.0], [0.0, 0.0, 1.0]])
    for frame in frames:
        image, truth = scene.image(frame), scene.homography(frame)
        for shift in SHIFTS_M:
            for direction in range(DIRECTIONS if shift else 1):
                angle = 2.0 * math.pi * direction / DIRECTIONS
                move = np.array([[1.0, 0.0, shift * math.cos(angle)], [0.0, 1.0, shift * math.sin(angle)], [0, 0, 1]])
                started = time.perf_counter()
                score = paint_alignment(image, move @ truth).score
                seconds.append(time.perf_counter() - started)
                by_shift[shift].append(score)
                shifted.append((score, shift))
        for noise_px in NOISE_PX:
            matrix = noisy_fit(scene, frame, noise_px, rng)
            if matrix is not None:
                pair = (paint_alignment(image, matrix).score, median_error_m(scene, frame, matrix))
                fitted.append(pair)
                fitted_by_noise[noise_px].append(pair)
        wrong = {
            "image mirrored left-right": truth @ mirror_image,
            "image mirrored top-bottom": truth @ flip_image,
            "pitch mirrored (same lines)": np.diag([-1.0, 1.0, 1.0]) @ truth,
        }
        for row, column in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)):
            changed = truth.copy()
            changed[row, column] = -changed[row, column]
            wrong[f"sign of H[{row},{column}] changed"] = changed
        for name, matrix in wrong.items():
            variants.setdefault(name, []).append(paint_alignment(image, matrix).score)
    print(f"paint alignment, synthetic clip, {len(frames)} frames")
    for shift, scores in by_shift.items():
        values = np.array([math.nan if score is None else score for score in scores])
        print(
            f"  shift {shift:4.1f} m: n {len(values)}, median {np.nanmedian(values):.3f}, "
            f"p10 {np.nanpercentile(values, 10):.3f}, p90 {np.nanpercentile(values, 90):.3f}"
        )
    separation("  shifted mappings", shifted)
    separation("  noisy landmark fits", fitted)
    for noise_px, pairs in fitted_by_noise.items():
        errors = [error for _, error in pairs]
        scores = [math.nan if score is None else score for score, _ in pairs]
        print(
            f"  landmark noise {noise_px:4.1f} px: n fits {len(pairs)}, error median {np.median(errors):.2f} m, "
            f"p90 {np.percentile(errors, 90):.2f} m, score median {np.nanmedian(scores):.3f}"
        )
    for name, scores in variants.items():
        values = np.array([math.nan if score is None else score for score in scores])
        print(f"  {name}: median {np.nanmedian(values):.3f}, max {np.nanmax(values):.3f}, n {len(values)}")
    print(
        f"  time of one score at {width} x {height}: median {1000 * np.median(seconds):.1f} ms "
        f"(n {len(seconds)}, this machine)"
    )


def correlations(source: FrameSource) -> list[float | None]:
    """Return the cut-test correlation of every pair of neighbouring frames, as ``compute_motion_field`` takes it."""
    info = source.info
    full_size = (info.width, info.height)
    small_size = (round(info.width * 0.5), round(info.height * 0.5))
    values: list[float | None] = []
    last_step: np.ndarray | None = None
    previous = _small_grey(source.read(0), full_size, small_size)
    for frame in range(1, info.n_frames):
        current = _small_grey(source.read(frame), full_size, small_size)
        step = estimate_motion(previous, current)
        values.append(shot_correlation(previous, current, [m for m in (step, last_step) if m is not None]))
        last_step = step if step is not None else last_step
        previous = current
    return values


def measure_cuts() -> None:
    """Print the cut-test correlations inside shots and across cuts."""
    inside: list[float] = []
    across: list[float] = []
    clips = {
        "slow camera": (list(range(0, 50)), ()),
        "fast pan, 83 px per frame": (list(range(0, 400, 10)), ()),
        "cut 1 s": ([*range(100, 106), *range(131, 137)], (6,)),
        "cut 3 s": ([*range(100, 106), *range(181, 187)], (6,)),
        "cut 10 s": ([*range(0, 6), *range(250, 256)], (6,)),
        "cut 4 s": ([*range(0, 54), *range(154, 208)], (54,)),
    }
    for name, (scene_frames, cuts) in clips.items():
        scene = RetimedScene(scene_frames, seed=0)
        for graphic in (False, True):
            source: FrameSource = ScoreGraphicSource(scene) if graphic else SyntheticSource(scene)
            values = correlations(source)
            for index, value in enumerate(values, start=1):
                (across if index in cuts else inside).append(-1.0 if value is None else value)
            shown = [round(values[cut - 1], 3) if values[cut - 1] is not None else None for cut in cuts]
            low = min(value for index, value in enumerate(values, start=1) if index not in cuts and value is not None)
            print(f"  {name}{' + graphic' if graphic else ''}: lowest inside a shot {low:.3f}, at the cut {shown}")
    print(f"cut test: n pairs inside shots {len(inside)}, lowest {min(inside):.3f}")
    print(f"cut test: n pairs across cuts {len(across)}, highest {max(across):.3f}")
    # A jump of 0.4 s while the camera barely moves: the grass moves like a short pan, so the
    # test cannot see the cut. This is a known miss, not one of the cuts above.
    jump = correlations(SyntheticSource(RetimedScene([*range(100, 106), *range(116, 122)], seed=0)))[5]
    print(f"known miss, a jump of 0.4 s with a slow camera: correlation at the jump {jump:.3f}")


class DissolveSource:
    """A clip of two shots of the synthetic match that are blended over ``length`` frames.

    Frames before ``start`` show the match at the same frame; frames from ``start + length`` on
    show it 100 frames later (another camera pose, other people). In between, the second shot
    fades in by equal steps.
    """

    def __init__(self, length: int, *, start: int = 45, n_frames: int = 100) -> None:
        self._scene = SyntheticSource(SyntheticScene(n_frames=n_frames + 120, seed=0))
        self._length = length
        self._start = start
        self.info = ClipInfo(clip_id="dissolve", n_frames=n_frames, fps=25.0, width=1280, height=720)

    def read(self, frame: int) -> np.ndarray:
        if frame < self._start:
            return self._scene.read(frame)
        if frame >= self._start + self._length:
            return self._scene.read(frame + 100)
        weight = (frame - self._start + 1) / (self._length + 1)
        blend = (1.0 - weight) * self._scene.read(frame).astype(np.float32)
        blend += weight * self._scene.read(frame + 100).astype(np.float32)
        return np.clip(blend, 0, 255).astype(np.uint8)


def measure_limits() -> None:
    """Print the cut-test correlations of the known limits: fast pans and dissolves."""
    for step, label in ((12, "100 px"), (15, "125 px"), (18, "150 px")):
        values = correlations(SyntheticSource(RetimedScene(list(range(0, 40 * step, step)), seed=0)))
        lowest = min((value for value in values if value is not None), default=math.nan)
        n_low = sum(1 for value in values if value is None or value < MIN_SHOT_CORRELATION)
        print(
            f"known limit, a pan of {label} per frame: lowest {lowest:.3f}, "
            f"pairs below {MIN_SHOT_CORRELATION}: {n_low} of {len(values)}"
        )
    for length in (0, 5, 10):
        values = correlations(DissolveSource(length))
        window = values[44 : 45 + max(length, 1)]  # the pairs into, inside and out of the dissolve
        low = min(-1.0 if value is None else value for value in window)
        print(f"known limit, a dissolve of {length} frames: lowest correlation of its pairs {low:.3f}")


def _detector_tools(scene: SyntheticScene) -> ToolAnswers:
    """Return the stand-in answers of the detector and the calibration for a synthetic clip."""
    people, balls = fake_detector_answers(scene)
    return ToolAnswers(people=people, balls=balls, cameras=fake_calibration_answers(scene))


def measure_detector_engine() -> None:
    """Print the identity values of the detector engine on synthetic clips (stand-in tools, seed 3)."""
    engines = EngineSettings(people="detector", pitch_mapping="calibration", number="none")
    scene = SyntheticScene(n_frames=250, seed=3)
    truth = [scene.frame_state(frame) for frame in range(scene.info.n_frames)]
    for name, tracking in (("without slots", TrackingConfig()), ("with slots", TrackingConfig(team_slots=True))):
        config = PipelineConfig(engines=engines, tracking=tracking)
        result = run_clip(SyntheticSource(scene), None, config, tools=_detector_tools(scene))
        scores = identity_scores(result.frames, truth)
        print(
            f"detector engine, 250 frames, seed 3, {name}: {len(result.tracks)} tracks, {scores.switches} switches, "
            f"coverage {scores.coverage:.3f}, track ratio {scores.track_ratio:.3f}"
        )
    cut = 54
    cut_scene = RetimedScene([*range(0, cut), *range(cut + 100, 2 * cut + 100)], seed=3)
    config = PipelineConfig(engines=engines)
    result = run_clip(SyntheticSource(cut_scene), None, config, tools=_detector_tools(cut_scene))
    crossing = sum(1 for track in result.tracks if track.first_frame < cut <= track.last_frame)
    paint = [check.score for check in result.paint if check.score is not None]
    print(
        f"detector engine, cut at frame {cut}, seed 3: {len(result.tracks)} tracks, {crossing} across the cut; "
        f"paint of {len(paint)} keyframe cameras {min(paint):.3f} to {max(paint):.3f}"
    )


# --- Dissolves -------------------------------------------------------------------------

FIT_SEEDS = range(0, 10)
HELD_OUT_SEEDS = range(10, 20)
DISSOLVE_LENGTHS = (5, 10, 15, 20, 25)
#: Frames of each shot before and after the change of a two-shot clip.
SHOT_FRAMES = 40
#: Frames of a control clip, and of the clip without a cut.
CONTROL_FRAMES = 60
NO_CUT_FRAMES = 100
#: Scene-frame steps of the pans: up to about 25, 50, 75 and 100 pixels per frame at 1280 px.
PAN_STEPS = (3, 6, 9, 12)
WIDTH, HEIGHT = 1280, 720
#: Bins of the true blend weight for the shot shares of the fit seeds.
WEIGHT_BINS = ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 1.0))


def _scene_motion(scene: SyntheticScene, frame: int) -> np.ndarray:
    """Return the true image motion of a synthetic scene from ``frame`` to ``frame + 1``."""
    return np.linalg.inv(scene.homography(frame + 1)) @ scene.homography(frame)


class TwoShotSource:
    """Two shots of the synthetic match of one seed, joined by a transition at frame ``SHOT_FRAMES``.

    Shot A shows scene frames ``a0 + step_a * k`` and shot B ``b0 + step_b * k`` of the clip
    frame ``k``; the start frames (B 75 to 249 frames after A) and the steps (1 to 3, up to about
    25 px per frame) come from the seed. ``kind`` names the transition over ``length`` frames:
    ``dissolve`` (frame ``SHOT_FRAMES + i`` shows B with the weight ``(i + 1) / (length + 1)``;
    length 0 is a hard cut), ``black`` (A fades to black over the first half, B comes up from
    black over the second), ``wipe`` (B covers the picture from left to right) and ``pip`` (B in
    a box of a third of the size in the top right corner, over A).
    """

    def __init__(self, seed: int, kind: str, length: int) -> None:
        rng = np.random.default_rng([seed, 99])
        start_a = int(rng.integers(0, 300))
        start_b = start_a + int(rng.integers(75, 250))
        step_a, step_b = (int(step) for step in rng.integers(1, 4, size=2))
        n_frames = 2 * SHOT_FRAMES + length
        self.shot_a = RetimedScene([start_a + step_a * k for k in range(n_frames)], seed=seed)
        self.shot_b = RetimedScene([start_b + step_b * k for k in range(n_frames)], seed=seed)
        self.kind, self.length = kind, length
        self.info = ClipInfo(clip_id=f"{kind}-{length}", n_frames=n_frames, fps=25.0, width=WIDTH, height=HEIGHT)

    def read(self, frame: int) -> np.ndarray:
        inside = frame - SHOT_FRAMES
        if inside < 0:
            return self.shot_a.image(frame)
        if inside >= self.length:
            return self.shot_b.image(frame)
        old = self.shot_a.image(frame).astype(np.float32)
        new = self.shot_b.image(frame).astype(np.float32)
        weight = (inside + 1) / (self.length + 1)
        if self.kind == "dissolve":
            image = (1.0 - weight) * old + weight * new
        elif self.kind == "black":
            image = max(0.0, 1.0 - 2.0 * weight) * old + max(0.0, 2.0 * weight - 1.0) * new
        elif self.kind == "wipe":
            image = old.copy()
            edge = round(weight * WIDTH)
            image[:, :edge] = new[:, :edge]
        elif self.kind == "pip":
            image = old.copy()
            small = cv2.resize(new, (WIDTH // 3, HEIGHT // 3), interpolation=cv2.INTER_AREA)
            image[20 : 20 + HEIGHT // 3, WIDTH - 20 - WIDTH // 3 : WIDTH - 20] = small
        else:
            raise ValueError(f"unknown transition {self.kind!r}")
        return np.clip(image, 0, 255).astype(np.uint8)

    def speed_px(self) -> float:
        """Return the fastest image motion of the centre of the two shots, pixels per frame."""
        return max(SceneSource(shot).speed_px() for shot in (self.shot_a, self.shot_b))


class SceneSource(SyntheticSource):
    """A synthetic scene as a frame source that also knows its true image motion."""

    def speed_px(self) -> float:
        return _fastest_centre(lambda k: _scene_motion(self._scene, k), self.info.n_frames)


class VirtualCameraSource:
    """A camera that turns, tilts, rolls and zooms freely, made from a larger synthetic picture.

    Each frame of the synthetic match is drawn at 2560 x 1440 pixels, the same field of view
    as 1280 x 720 at twice the pixels. The clip shows a 1280 x 720 view of it through a virtual
    camera at the same place: ``poses[k]`` is (pan, tilt, roll in degrees, zoom) of frame
    ``k``; zoom 1 shows half the field of view of the drawn picture. A turn of the camera is a
    homography of the picture, so the clip has camera motion the synthetic scene cannot make.
    """

    def __init__(self, seed: int, scene_frames: Sequence[int], poses: Sequence[tuple[float, float, float, float]]):
        self._scene = RetimedScene(scene_frames, width=2 * WIDTH, height=2 * HEIGHT, seed=seed)
        self._poses = tuple(poses)
        self.info = ClipInfo(clip_id="virtual", n_frames=len(scene_frames), fps=25.0, width=WIDTH, height=HEIGHT)
        self._drawn = np.array([[2.0 * WIDTH, 0.0, WIDTH], [0.0, 2.0 * WIDTH, HEIGHT], [0.0, 0.0, 1.0]])

    def _view(self, frame: int) -> np.ndarray:
        """Return the homography from the drawn picture to the view of a frame."""
        pan_rad, tilt_rad, roll_rad = (math.radians(value) for value in self._poses[frame][:3])
        zoom = self._poses[frame][3]
        turn = np.array(
            [[math.cos(pan_rad), 0.0, math.sin(pan_rad)], [0.0, 1.0, 0.0], [-math.sin(pan_rad), 0.0, math.cos(pan_rad)]]
        )
        cos_tilt, sin_tilt = math.cos(tilt_rad), math.sin(tilt_rad)
        nod = np.array([[1.0, 0.0, 0.0], [0.0, cos_tilt, -sin_tilt], [0.0, sin_tilt, cos_tilt]])
        spin = np.array(
            [[math.cos(roll_rad), -math.sin(roll_rad), 0.0], [math.sin(roll_rad), math.cos(roll_rad), 0.0], [0, 0, 1.0]]
        )
        focal = 2.0 * WIDTH * zoom
        view = np.array([[focal, 0.0, WIDTH / 2.0], [0.0, focal, HEIGHT / 2.0], [0.0, 0.0, 1.0]])
        return view @ spin @ nod @ turn @ np.linalg.inv(self._drawn)

    def read(self, frame: int) -> np.ndarray:
        drawn = self._scene.image(frame)
        view = self._view(frame)
        return cv2.warpPerspective(drawn, view, (WIDTH, HEIGHT), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)

    def speed_px(self) -> float:
        def motion(frame: int) -> np.ndarray:
            drawn = _scene_motion(self._scene, frame)
            return self._view(frame + 1) @ drawn @ np.linalg.inv(self._view(frame))

        return _fastest_centre(motion, self.info.n_frames)


def _fastest_centre(motion: Callable[[int], np.ndarray], n_frames: int) -> float:
    """Return the largest move of the image centre between two neighbouring frames, pixels."""
    centre = np.array([[WIDTH / 2.0, HEIGHT / 2.0]])
    return max(float(np.linalg.norm(apply_homography(motion(k), centre) - centre)) for k in range(n_frames - 1))


def _ramp(frame: int, start: int, length: int) -> float:
    """Return 0 before ``start``, 1 after ``start + length``, and a straight rise in between."""
    return min(1.0, max(0.0, (frame - start) / length))


def _controls(seed: int) -> dict[str, VirtualCameraSource | SceneSource]:
    """Return the control clips of one seed by case: none of them has a cut."""
    rng = np.random.default_rng([seed, 5])
    cases: dict[str, VirtualCameraSource | SceneSource] = {}
    for step in PAN_STEPS:
        start = int(rng.integers(0, 400))
        scene = RetimedScene([start + step * k for k in range(CONTROL_FRAMES)], seed=seed)
        cases[f"pan, scene step {step}"] = SceneSource(scene)
    start = int(rng.integers(0, 400))
    slow = [start + k for k in range(CONTROL_FRAMES)]
    zoom_frames = int(rng.integers(8, 21))
    zoom_in = bool(rng.integers(0, 2))
    zooms = [_ramp(k, 20, zoom_frames) for k in range(CONTROL_FRAMES)]
    poses = [(0.0, 0.0, 0.0, 1.0 + 0.5 * (value if zoom_in else 1.0 - value)) for value in zooms]
    cases["zoom x1.5 in 8-20 frames"] = VirtualCameraSource(seed, slow, poses)
    shake = [
        (rng.normal(0.0, 0.15), rng.normal(0.0, 0.15), rng.normal(0.0, 0.1), 1.0) for _ in range(CONTROL_FRAMES)
    ]
    cases["shake, sd 0.15 deg"] = VirtualCameraSource(seed, slow, shake)
    step = int(rng.integers(2, 5))  # the view is zoomed in twice: up to about 66 px per frame of pan
    flight = [math.sin(math.pi * _ramp(k, 15, 25)) for k in range(CONTROL_FRAMES)]
    kick = [(0.0, 5.0 * value, 0.0, 1.0 - 0.12 * value) for value in flight]
    cases["goal kick: fast pan, tilt 5 deg"] = VirtualCameraSource(
        seed, [start + step * k for k in range(CONTROL_FRAMES)], kick
    )
    start = int(rng.integers(0, 400))
    cases["no cut, 100 frames"] = SceneSource(RetimedScene(list(range(start, start + NO_CUT_FRAMES)), seed=seed))
    return cases


def _cut_blocks(cuts: Sequence[int]) -> list[tuple[int, int]]:
    """Group cut frames into blocks of neighbouring frames: (first, last)."""
    blocks: list[tuple[int, int]] = []
    for frame in cuts:
        if blocks and blocks[-1][1] == frame - 1:
            blocks[-1] = (blocks[-1][0], frame)
        else:
            blocks.append((frame, frame))
    return blocks


@dataclass
class CaseTally:
    """The numbers of one case over its clips."""

    n: int = 0
    hits: int = 0
    false_cuts: int = 0
    base_hits: int = 0
    base_false_cuts: int = 0
    judged: int = 0
    windows: int = 0
    window_minima: list[float] = field(default_factory=list)  # the lowest judged window value of each clip
    pair_minima: list[float] = field(default_factory=list)  # the lowest pair value of each clip
    start_errors: list[int] = field(default_factory=list)
    end_errors: list[int] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)

    def add(self, tests: ShotTests, change: tuple[int, int] | None, speed_px: float) -> None:
        """Count one clip. ``change`` is (first, last) of the true cut frames: the blend frames and the frame after."""
        self.n += 1
        self.speeds.append(speed_px)
        for cuts, baseline in ((tests.cuts(), False), (tests.hard_cuts, True)):
            blocks = _cut_blocks(cuts)
            hits = [block for block in blocks if change is not None and block[0] <= change[1] and block[1] >= change[0]]
            if baseline:
                self.base_hits += bool(hits)
                self.base_false_cuts += len(blocks) - len(hits)
                continue
            self.hits += bool(hits)
            self.false_cuts += len(blocks) - len(hits)
            if hits and change is not None:
                self.start_errors.append(hits[0][0] - change[0])
                self.end_errors.append(hits[-1][1] - change[1])
        judged = [value for value in tests.window_correlations if value is not None]
        self.windows += len(tests.window_correlations) - DISSOLVE_WINDOW_FRAMES
        self.judged += len(judged)
        if judged:
            self.window_minima.append(min(judged))
        self.pair_minima.append(min(-1.0 if value is None else value for value in tests.pair_correlations))

    def line(self, name: str, *, has_change: bool) -> str:
        def low_high(values: Sequence[float]) -> str:
            return f"{min(values):.3f} to {max(values):.3f}" if values else "none"

        text = f"  {name}: n {self.n}"
        if has_change:
            text += f", hits {self.hits}, misses {self.n - self.hits}"
        text += f", false cuts {self.false_cuts}"
        text += " | pair test alone:"
        if has_change:
            text += f" hits {self.base_hits},"
        text += f" false cuts {self.base_false_cuts}"
        text += f" | lowest window value per clip {low_high(self.window_minima)}"
        text += f" ({self.judged} of {self.windows} windows judged)"
        text += f" | lowest pair value per clip {low_high(self.pair_minima)}"
        if self.start_errors:
            text += (
                f" | span error, frames: start {min(self.start_errors):+d} to {max(self.start_errors):+d}"
                f" (median {np.median(self.start_errors):+.0f}), end {min(self.end_errors):+d} to"
                f" {max(self.end_errors):+d} (median {np.median(self.end_errors):+.0f})"
            )
        text += f" | fastest centre motion {min(self.speeds):.0f} to {max(self.speeds):.0f} px per frame"
        return text


def _shot_shares(tests: ShotTests, start: int, length: int) -> list[tuple[float, float | None]]:
    """Return (true blend weight, shot share) of the frames between the anchors of the first window run.

    The share of a frame before the middle of the dissolve is taken from the old anchor, after it
    from the new anchor, as ``find_transitions`` uses them. The weight is that of the shot the
    anchor does not show: 0 for a frame of the anchor's own shot. A clip without a window run has none.
    """
    values = tests.window_correlations
    low = [value is not None and value < MAX_TRANSITION_CORRELATION for value in values]
    first = next((k for k, is_low in enumerate(low) if is_low), None)
    if first is None:  # a miss of the window test: no anchors
        return []
    last = first
    while last + 1 < len(values) and low[last + 1]:
        last += 1
    old, new = first - DISSOLVE_WINDOW_FRAMES, last
    images, steps = tests.test_images, tests.test_steps
    middle = start + length / 2.0
    shares: list[tuple[float, float | None]] = []
    for frame in range(old + 1, new):
        weight_b = min(1.0, max(0.0, (frame - start + 1) / (length + 1)))
        if frame < middle:
            shares.append((weight_b, _shot_share(images[old], images[frame], _chain(steps, old, frame))))
        else:
            shares.append((1.0 - weight_b, _shot_share(images[new], images[frame], _chain(steps, new, frame))))
    return shares


def _run_cases(seeds: range, *, with_shares: bool) -> None:
    """Print the tallies of the two-shot cases and the controls over ``seeds``."""
    tallies: dict[str, CaseTally] = {}
    changes: dict[str, bool] = {}
    shares: list[tuple[float, float | None]] = []
    for seed in seeds:
        for length in (0, *DISSOLVE_LENGTHS):
            name = "hard cut" if length == 0 else f"dissolve of {length} frames"
            source = TwoShotSource(seed, "dissolve", length)
            tests = measure_shots(source)
            tallies.setdefault(name, CaseTally()).add(tests, (SHOT_FRAMES, SHOT_FRAMES + length), source.speed_px())
            changes[name] = True
            if with_shares and length:
                shares.extend(_shot_shares(tests, SHOT_FRAMES, length))
        for name, control in _controls(seed).items():
            tallies.setdefault(name, CaseTally()).add(measure_shots(control), None, control.speed_px())
            changes[name] = False
    for name, tally in tallies.items():
        print(tally.line(name, has_change=changes[name]))
    if with_shares:
        for low, high in WEIGHT_BINS:
            known = [share for weight, share in shares if low < weight <= high and share is not None]
            print(
                f"  shot share of frames with a blend weight in ({low}, {high}]: n {len(known)}, "
                f"lowest {min(known):.3f}, highest {max(known):.3f}"
            )
        pure = [share for weight, share in shares if weight == 0.0]
        known = [share for share in pure if share is not None]
        print(
            f"  shot share of frames of the anchor's own shot: n {len(pure)}, not measurable {len(pure) - len(known)}, "
            f"lowest {min(known):.3f}, highest {max(known):.3f}"
        )


def measure_dissolve_fit() -> None:
    """Print the dissolve numbers of the seeds on which the thresholds were fixed."""
    print(f"dissolve test, synthetic, fit seeds {FIT_SEEDS.start} to {FIT_SEEDS.stop - 1}:")
    _run_cases(FIT_SEEDS, with_shares=True)


def _out_of_scope(seeds: range) -> None:
    """Print what the cut tests do with the transitions that stay out of scope."""
    for kind, length in (("black", 10), ("wipe", 10), ("pip", 25)):
        tally = CaseTally()
        for seed in seeds:
            source = TwoShotSource(seed, kind, length)
            tally.add(measure_shots(source), (SHOT_FRAMES, SHOT_FRAMES + length), source.speed_px())
        print(tally.line(f"out of scope, {kind} over {length} frames", has_change=True))


def _cut_test_time() -> None:
    """Print the time of the cut test per frame on this machine: the pair test of main, and pair plus window test."""
    before: list[float] = []
    after: list[float] = []
    for source in (SceneSource(RetimedScene(list(range(NO_CUT_FRAMES)), seed=0)), TwoShotSource(0, "dissolve", 15)):
        info = source.info
        full_size, small_size = (info.width, info.height), (info.width // 2, info.height // 2)
        test_size = _test_size(small_size)
        to_test = _to_small_matrix(small_size, test_size)
        from_test = np.linalg.inv(to_test)
        tests = measure_shots(source)
        smalls = [_small_grey(source.read(k), full_size, small_size) for k in range(info.n_frames)]
        to_small = _to_small_matrix(full_size, small_size)
        small_steps = [None if step is None else to_small @ step @ np.linalg.inv(to_small) for step in tests.steps]
        for frame in range(DISSOLVE_WINDOW_FRAMES, info.n_frames):
            motions = [step for step in (small_steps[frame - 1], small_steps[frame - 2]) if step is not None]
            started = time.perf_counter()
            shot_correlation(smalls[frame - 1], smalls[frame], motions)
            before.append(time.perf_counter() - started)
            started = time.perf_counter()
            current = _test_image(smalls[frame], test_size)
            matrices = [to_test @ motion @ from_test for motion in [np.eye(3), *motions]]
            _correlations(tests.test_images[frame - 1], current, matrices)
            window = slice(frame - DISSOLVE_WINDOW_FRAMES, frame + 1)
            window_correlations(tests.test_images[window], tests.test_steps[frame - DISSOLVE_WINDOW_FRAMES : frame])
            after.append(time.perf_counter() - started)
    print(
        f"  time of the cut test per frame, this machine: pair test of main median {1000 * np.median(before):.2f} ms, "
        f"pair and window test median {1000 * np.median(after):.2f} ms (n {len(before)} frames)"
    )


def measure_dissolve_held_out() -> None:
    """Print the dissolve numbers of the held-out seeds, the out-of-scope transitions and the time per frame."""
    print(f"dissolve test, synthetic, held-out seeds {HELD_OUT_SEEDS.start} to {HELD_OUT_SEEDS.stop - 1}:")
    _run_cases(HELD_OUT_SEEDS, with_shares=False)
    _out_of_scope(HELD_OUT_SEEDS)
    _cut_test_time()


SECTIONS = {
    "paint": measure_paint,
    "cuts": measure_cuts,
    "limits": measure_limits,
    "detector": measure_detector_engine,
    "dissolve-fit": measure_dissolve_fit,
    "dissolve-held-out": measure_dissolve_held_out,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the synthetic measurements of docs/DECISIONS.md.")
    parser.add_argument("sections", nargs="*", help=f"sections to run, of {', '.join(SECTIONS)}; default: all")
    chosen = parser.parse_args().sections
    unknown = [name for name in chosen if name not in SECTIONS]
    if unknown:
        parser.error(f"unknown sections {unknown}; the sections are {', '.join(SECTIONS)}")
    for section in chosen or SECTIONS:
        SECTIONS[section]()
