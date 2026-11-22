"""Tests of the paint-alignment score of geometry.py on synthetic frames.

The frames come from ``synth.SyntheticScene`` (1280 x 720, seed 0). Frame 0 shows the left side
of the pitch, frame 75 the middle, frame 150 the right side. A shifted mapping moves every pitch
position by the same distance, so its position error is that distance everywhere.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from pitchvlm.geometry import (
    PAINT_MIN_SHARE,
    paint_alignment,
    paint_mask,
    projected_line_pixels,
)
from pitchvlm.probes import auroc
from pitchvlm.synth import GRASS_BGR, LINE_BGR, MARK_BGR, SyntheticScene

FRAMES = (0, 40, 75, 110, 150)


@pytest.fixture(scope="module")
def scene() -> SyntheticScene:
    return SyntheticScene(n_frames=200, seed=0)


def shifted(matrix: np.ndarray, distance_m: float, angle_rad: float = math.pi / 4.0) -> np.ndarray:
    """Return a mapping that puts every pitch position ``distance_m`` metres away from where ``matrix`` puts it."""
    move = np.array(
        [[1.0, 0.0, distance_m * math.cos(angle_rad)], [0.0, 1.0, distance_m * math.sin(angle_rad)], [0.0, 0.0, 1.0]]
    )
    return move @ matrix


def grass(width: int = 1280, height: int = 720) -> np.ndarray:
    """Return a frame of plain grass."""
    return np.full((height, width, 3), GRASS_BGR, dtype=np.uint8)


def score(image: np.ndarray, matrix: np.ndarray) -> float:
    """Return the score of a mapping that the frame can judge."""
    value = paint_alignment(image, matrix).score
    assert value is not None
    return value


# --- the right mapping and wrong ones -------------------------------------------


@pytest.mark.parametrize("frame", FRAMES)
def test_the_right_mapping_scores_high(scene: SyntheticScene, frame: int) -> None:
    result = paint_alignment(scene.image(frame), scene.homography(frame))
    assert result.reason is None
    assert result.score is not None and result.score > 0.95
    assert result.precision is not None and result.precision > 0.95
    assert result.spread == 1.0


@pytest.mark.parametrize("frame", FRAMES)
def test_a_larger_shift_scores_lower(scene: SyntheticScene, frame: int) -> None:
    image, truth = scene.image(frame), scene.homography(frame)
    scores = [score(image, shifted(truth, distance_m)) for distance_m in (0.0, 1.0, 3.0, 10.0)]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == 4
    assert scores[-1] < 0.2


@pytest.mark.parametrize("frame", (0, 40, 150))
def test_a_mirrored_picture_scores_lower(scene: SyntheticScene, frame: int) -> None:
    image, truth = scene.image(frame), scene.homography(frame)
    width, height = scene.info.width, scene.info.height
    left_right = np.array([[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    top_bottom = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, height - 1.0], [0.0, 0.0, 1.0]])
    right = score(image, truth)
    assert score(image, truth @ left_right) < right - 0.4
    assert score(image, truth @ top_bottom) < right - 0.4


@pytest.mark.parametrize("entry", [(0, 0), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)])
def test_a_wrong_sign_scores_lower(scene: SyntheticScene, entry: tuple[int, int]) -> None:
    image, truth = scene.image(40), scene.homography(40)
    wrong = truth.copy()
    wrong[entry] = -wrong[entry]
    assert score(image, wrong) < score(image, truth) - 0.4


def test_mirroring_the_pitch_gives_the_same_score(scene: SyntheticScene) -> None:
    # The pitch markings are symmetric, so the paint cannot see the side or the half of the pitch.
    # check_homography finds such a mapping.
    image, truth = scene.image(40), scene.homography(40)
    for signs in ((-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)):
        assert score(image, np.diag([*signs, 1.0]) @ truth) == pytest.approx(score(image, truth))


def test_the_scale_and_sign_of_the_matrix_do_not_matter(scene: SyntheticScene) -> None:
    image, truth = scene.image(40), scene.homography(40)
    assert score(image, -3.0 * truth) == pytest.approx(score(image, truth))


def test_lines_that_all_fall_on_one_painted_line_score_low(scene: SyntheticScene) -> None:
    # Squeeze the pitch 1000 times along x: every line lands within 5 cm of the halfway line, which
    # is painted. So almost every projected pixel hits paint, but along one line only, and one line
    # fixes no mapping.
    image, truth = scene.image(75), scene.homography(75)
    squeezed = np.diag([1000.0, 1.0, 1.0]) @ truth
    result = paint_alignment(image, squeezed)
    assert result.precision is not None and result.precision > 0.9
    assert result.spread < 0.1
    assert result.score is not None and result.score < 0.1


def test_hits_in_one_corner_score_low(scene: SyntheticScene) -> None:
    # Only the top left corner of the picture shows paint, and the right mapping hits it there only.
    image, truth = scene.image(75), scene.homography(75)
    corner = grass()
    corner[:270, :480] = image[:270, :480]
    result = paint_alignment(corner, truth)
    assert result.paint_share >= PAINT_MIN_SHARE
    assert result.spread < 0.5
    assert result.score is not None and result.score < 0.1


def test_lines_outside_the_picture_score_zero_in_a_frame_with_paint(scene: SyntheticScene) -> None:
    image, truth = scene.image(40), scene.homography(40)
    far_away = shifted(truth, 500.0, 0.0)
    result = paint_alignment(image, far_away)
    assert result.n_model_px == 0
    assert result.precision is None
    assert result.score == 0.0


# --- frames that cannot judge -----------------------------------------------------


def test_a_frame_without_paint_cannot_judge(scene: SyntheticScene) -> None:
    result = paint_alignment(grass(), scene.homography(40))
    assert result.score is None
    assert result.reason is not None and result.reason.startswith("too little paint")


def test_a_frame_with_almost_no_lines_cannot_judge(scene: SyntheticScene) -> None:
    image = grass()
    cv2.line(image, (600, 400), (700, 400), LINE_BGR, thickness=2)
    result = paint_alignment(image, scene.homography(40))
    assert 0.0 < result.paint_share < PAINT_MIN_SHARE
    assert result.score is None
    assert result.reason is not None and result.reason.startswith("too little paint")


def test_a_frame_full_of_white_cannot_judge(scene: SyntheticScene) -> None:
    image = grass()
    for x in range(0, image.shape[1], 12):
        cv2.line(image, (x, 0), (x, image.shape[0] - 1), LINE_BGR, thickness=2)
    result = paint_alignment(image, scene.homography(40))
    assert result.score is None
    assert result.reason is not None and result.reason.startswith("too much white")


# --- the paint mask and the projected lines ------------------------------------------


def test_the_mask_finds_the_lines_and_not_the_grass_texture(scene: SyntheticScene) -> None:
    textured = grass()
    rng = np.random.default_rng(3)
    for x, y in rng.integers(0, (1270, 710), size=(400, 2)):
        cv2.fillConvexPoly(textured, np.array([[x, y], [x + 9, y + 2], [x + 3, y + 8]], dtype=np.int32), MARK_BGR)
    assert paint_mask(textured).sum() == 0
    image = scene.image(40)
    on_lines = paint_mask(image)[tuple(projected_line_pixels(scene.homography(40), (1280, 720))[:, ::-1].T)]
    assert on_lines.mean() > 0.9


def test_the_mask_ignores_white_away_from_grass() -> None:
    sky = np.full((720, 1280, 3), (200, 150, 120), dtype=np.uint8)
    cv2.line(sky, (100, 100), (1100, 100), LINE_BGR, thickness=2)
    assert paint_mask(sky).sum() == 0


def test_the_projected_lines_lie_in_the_picture_once_each(scene: SyntheticScene) -> None:
    pixels = projected_line_pixels(scene.homography(40), (1280, 720))
    assert pixels.dtype == np.int64
    assert pixels.shape[1] == 2
    assert len(np.unique(pixels, axis=0)) == len(pixels)
    assert (pixels >= 0).all() and (pixels[:, 0] <= 1279).all() and (pixels[:, 1] <= 719).all()


def test_no_line_is_projected_from_behind_the_camera(scene: SyntheticScene) -> None:
    # The horizon of the synthetic camera is above the picture. The lines behind the camera would
    # come back into the picture upside down if the side in front were not respected.
    truth = scene.homography(40)
    tall = projected_line_pixels(truth, (1280, 720))
    image = scene.image(40)
    assert paint_mask(image)[tall[:, 1], tall[:, 0]].mean() > 0.9


def test_wrong_input_is_refused(scene: SyntheticScene) -> None:
    with pytest.raises(ValueError, match="no usable inverse"):
        projected_line_pixels(np.zeros((3, 3)), (1280, 720))
    with pytest.raises(ValueError, match="positive"):
        projected_line_pixels(scene.homography(0), (0, 720))
    with pytest.raises(ValueError, match="BGR uint8"):
        paint_mask(np.zeros((720, 1280), dtype=np.uint8))
    with pytest.raises(ValueError, match="BGR uint8"):
        paint_alignment(np.zeros((720, 1280, 3), dtype=np.float32), scene.homography(0))


def test_the_score_is_the_same_every_time_and_at_another_image_size() -> None:
    small = SyntheticScene(n_frames=50, width=640, height=360, seed=0)
    image, truth = small.image(20), small.homography(20)
    assert paint_alignment(image, truth) == paint_alignment(image, truth)
    assert score(image, truth) > 0.95
    assert score(image, shifted(truth, 10.0)) < 0.2


def test_the_score_separates_good_from_bad_mappings_on_the_synthetic_clip(scene: SyntheticScene) -> None:
    # The run-machine measurement in small: mappings with an error of at most 2 m against
    # mappings with an error above 3 m. tools/synthetic_diagnostics.py measures it on 50 frames.
    good: list[float] = []
    bad: list[float] = []
    for frame in FRAMES:
        image, truth = scene.image(frame), scene.homography(frame)
        for direction in range(4):
            angle = math.pi / 8.0 + direction * math.pi / 2.0
            good.extend(score(image, shifted(truth, distance_m, angle)) for distance_m in (0.0, 1.0, 2.0))
            bad.extend(score(image, shifted(truth, distance_m, angle)) for distance_m in (5.0, 10.0))
    assert auroc(good, bad) > 0.9
