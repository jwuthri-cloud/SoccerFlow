"""Tests of the conversion between model coordinates (0-1000, y first) and image pixels."""

from __future__ import annotations

import random

import numpy as np
import pytest

from pitchvlm.schemas import Box
from pitchvlm.vlm.coords import box_from_model, box_to_model, point_from_model, point_to_model

#: A wide image. With a square image, a swapped x and y would not show.
WIDTH = 1920
HEIGHT = 1080
#: Half a model step in pixels: the largest rounding error of a pixel that goes to the model and back.
HALF_STEP_X = WIDTH / 2000
HALF_STEP_Y = HEIGHT / 2000


def random_model_box(rng: random.Random) -> list[int]:
    """Return a model box [ymin, xmin, ymax, xmax] that has an area."""
    ymin = rng.randint(0, 999)
    xmin = rng.randint(0, 999)
    return [ymin, xmin, rng.randint(ymin + 1, 1000), rng.randint(xmin + 1, 1000)]


# --- Order of the values ------------------------------------------------------


def test_box_from_model_reads_ymin_xmin_ymax_xmax() -> None:
    box = box_from_model([100, 200, 300, 400], WIDTH, HEIGHT)

    assert box is not None
    # x comes from the second and the fourth value, and scales with the width.
    assert box.x1 == pytest.approx(384.0)
    assert box.x2 == pytest.approx(768.0)
    # y comes from the first and the third value, and scales with the height.
    assert box.y1 == pytest.approx(108.0)
    assert box.y2 == pytest.approx(324.0)


def test_box_to_model_writes_ymin_xmin_ymax_xmax() -> None:
    box = Box(x1=384.0, y1=108.0, x2=768.0, y2=324.0)

    assert box_to_model(box, WIDTH, HEIGHT) == [100, 200, 300, 400]


def test_point_from_model_reads_y_then_x_and_returns_x_then_y() -> None:
    x_px, y_px = point_from_model([250, 750], WIDTH, HEIGHT)

    assert x_px == pytest.approx(1440.0)
    assert y_px == pytest.approx(270.0)


def test_point_to_model_takes_x_then_y_and_writes_y_then_x() -> None:
    assert point_to_model(1440.0, 270.0, WIDTH, HEIGHT) == [250, 750]


def test_model_corners_map_to_image_corners() -> None:
    # Origin top-left, x to the right, y down.
    assert point_from_model([0, 0], WIDTH, HEIGHT) == (0.0, 0.0)
    assert point_from_model([0, 1000], WIDTH, HEIGHT) == (float(WIDTH), 0.0)
    assert point_from_model([1000, 0], WIDTH, HEIGHT) == (0.0, float(HEIGHT))
    assert point_from_model([1000, 1000], WIDTH, HEIGHT) == (float(WIDTH), float(HEIGHT))
    assert point_to_model(0.0, 0.0, WIDTH, HEIGHT) == [0, 0]
    assert point_to_model(float(WIDTH), float(HEIGHT), WIDTH, HEIGHT) == [1000, 1000]


def test_a_larger_model_y_is_lower_in_the_image() -> None:
    _, upper_y = point_from_model([100, 500], WIDTH, HEIGHT)
    _, lower_y = point_from_model([900, 500], WIDTH, HEIGHT)

    assert lower_y > upper_y


# --- Round trips ----------------------------------------------------------------


def test_model_box_survives_the_trip_to_pixels_and_back() -> None:
    rng = random.Random(0)
    for _ in range(500):
        model_box = random_model_box(rng)
        box = box_from_model(model_box, WIDTH, HEIGHT)

        assert box is not None
        assert box_to_model(box, WIDTH, HEIGHT) == model_box


def test_model_point_survives_the_trip_to_pixels_and_back() -> None:
    rng = random.Random(1)
    for _ in range(500):
        model_point = [rng.randint(0, 1000), rng.randint(0, 1000)]
        x_px, y_px = point_from_model(model_point, WIDTH, HEIGHT)

        assert point_to_model(x_px, y_px, WIDTH, HEIGHT) == model_point


def test_pixel_box_comes_back_within_half_a_model_step() -> None:
    rng = random.Random(2)
    for _ in range(500):
        x1 = rng.uniform(0.0, WIDTH - 40.0)
        y1 = rng.uniform(0.0, HEIGHT - 40.0)
        box = Box(x1=x1, y1=y1, x2=x1 + rng.uniform(10.0, 40.0), y2=y1 + rng.uniform(10.0, 40.0))
        back = box_from_model(box_to_model(box, WIDTH, HEIGHT), WIDTH, HEIGHT)

        assert back is not None
        assert abs(back.x1 - box.x1) <= HALF_STEP_X + 1e-9
        assert abs(back.x2 - box.x2) <= HALF_STEP_X + 1e-9
        assert abs(back.y1 - box.y1) <= HALF_STEP_Y + 1e-9
        assert abs(back.y2 - box.y2) <= HALF_STEP_Y + 1e-9


def test_pixel_point_comes_back_within_half_a_model_step() -> None:
    rng = random.Random(3)
    for _ in range(500):
        x = rng.uniform(0.0, WIDTH)
        y = rng.uniform(0.0, HEIGHT)
        back_x, back_y = point_from_model(point_to_model(x, y, WIDTH, HEIGHT), WIDTH, HEIGHT)

        assert abs(back_x - x) <= HALF_STEP_X + 1e-9
        assert abs(back_y - y) <= HALF_STEP_Y + 1e-9


# --- Limits ---------------------------------------------------------------------


def test_box_from_model_moves_values_into_the_image() -> None:
    box = box_from_model([-50, -20, 1200, 1100], WIDTH, HEIGHT)

    assert box == Box(x1=0.0, y1=0.0, x2=float(WIDTH), y2=float(HEIGHT))


def test_point_from_model_moves_values_into_the_image() -> None:
    assert point_from_model([-5, 1500], WIDTH, HEIGHT) == (float(WIDTH), 0.0)
    assert point_from_model([2000, -1], WIDTH, HEIGHT) == (0.0, float(HEIGHT))


def test_to_model_gives_plain_integers_inside_the_range() -> None:
    values = box_to_model(Box(x1=-30.0, y1=-5.0, x2=5000.0, y2=4000.0), WIDTH, HEIGHT)

    assert values == [0, 0, 1000, 1000]
    assert all(type(value) is int for value in values)
    # x = -10 is left of the image and y = 5000 is below it: the model point is [y, x] = [1000, 0].
    assert point_to_model(-10.0, 5000.0, WIDTH, HEIGHT) == [1000, 0]
    assert all(type(value) is int for value in point_to_model(12.3, 45.6, WIDTH, HEIGHT))


@pytest.mark.parametrize(
    "model_box",
    [
        [100, 200, 100, 400],  # no height
        [100, 400, 300, 400],  # no width
        [300, 200, 100, 400],  # ymax above ymin
        [100, 400, 300, 200],  # xmax left of xmin
        [1100, 0, 1200, 500],  # below the image: both y values become 1000
        [0, -30, 500, -10],  # left of the image: both x values become 0
    ],
)
def test_box_from_model_returns_none_for_a_degenerate_box(model_box: list[int]) -> None:
    assert box_from_model(model_box, WIDTH, HEIGHT) is None


def test_numpy_values_are_accepted() -> None:
    assert box_from_model(np.array([100, 200, 300, 400]), WIDTH, HEIGHT) == box_from_model(
        [100, 200, 300, 400], WIDTH, HEIGHT
    )
    assert point_from_model(np.array([250.0, 750.0]), WIDTH, HEIGHT) == pytest.approx((1440.0, 270.0))


# --- Wrong input ----------------------------------------------------------------


@pytest.mark.parametrize(
    "model_box",
    [
        [100, 200, 300],
        [100, 200, 300, 400, 500],
        [float("nan"), 200, 300, 400],
        ["100", 200, 300, 400],
        [None, 200, 300, 400],
        [True, 200, 300, 400],
    ],
)
def test_box_from_model_refuses_wrong_values(model_box: list[object]) -> None:
    with pytest.raises(ValueError):
        box_from_model(model_box, WIDTH, HEIGHT)  # type: ignore[arg-type]


@pytest.mark.parametrize("model_point", [[100], [100, 200, 300], [float("nan"), 200], ["100", 200], [False, 200]])
def test_point_from_model_refuses_wrong_values(model_point: list[object]) -> None:
    with pytest.raises(ValueError):
        point_from_model(model_point, WIDTH, HEIGHT)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_point_to_model_refuses_a_pixel_that_is_not_finite(value: float) -> None:
    with pytest.raises(ValueError):
        point_to_model(value, 10.0, WIDTH, HEIGHT)
    with pytest.raises(ValueError):
        point_to_model(10.0, value, WIDTH, HEIGHT)


@pytest.mark.parametrize(("width", "height"), [(0, HEIGHT), (WIDTH, 0), (-1, HEIGHT), (WIDTH, -1)])
def test_image_size_must_be_positive(width: int, height: int) -> None:
    box = Box(x1=1.0, y1=1.0, x2=2.0, y2=2.0)

    with pytest.raises(ValueError):
        box_from_model([100, 200, 300, 400], width, height)
    with pytest.raises(ValueError):
        point_from_model([100, 200], width, height)
    with pytest.raises(ValueError):
        box_to_model(box, width, height)
    with pytest.raises(ValueError):
        point_to_model(1.0, 1.0, width, height)
