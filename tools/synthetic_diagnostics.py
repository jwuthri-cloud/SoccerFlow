"""Measure the paint-alignment score and the cut test on synthetic clips, and print the numbers.

This is the measurement behind the synthetic numbers of ``docs/DECISIONS.md`` (2026-11-22). It
runs offline on the build machine: no model, no footage, no network. Every number it prints is
synthetic, not a result on broadcast frames. Run it from the top folder of the repository:

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
a static score graphic; and one known miss, a short jump in time while the camera barely moves.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from pitchvlm.frames import FrameSource
from pitchvlm.geometry import apply_homography, fit_from_landmarks, matrix_from_record, paint_alignment
from pitchvlm.motion import _small_grey, estimate_motion, shot_correlation
from pitchvlm.probes import auroc, auroc_standard_error
from pitchvlm.schemas import FrameLandmarks, LandmarkObs
from pitchvlm.synth import RetimedScene, ScoreGraphicSource, SyntheticScene, SyntheticSource

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
    mirror_image = np.array([[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    flip_image = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, height - 1.0], [0.0, 0.0, 1.0]])
    for frame in frames:
        image, truth = scene.image(frame), scene.homography(frame)
        for shift in SHIFTS_M:
            for direction in range(DIRECTIONS if shift else 1):
                angle = 2.0 * math.pi * direction / DIRECTIONS
                move = np.array([[1.0, 0.0, shift * math.cos(angle)], [0.0, 1.0, shift * math.sin(angle)], [0, 0, 1]])
                score = paint_alignment(image, move @ truth).score
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


if __name__ == "__main__":
    measure_paint()
    measure_cuts()
