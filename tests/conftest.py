"""Shared test settings: caches of pure results that many tests compute again and again.

Many tests run the pipeline or the command line on the same synthetic clips. Six results are
the same in each of these runs. This file computes each one once per test process and hands
out copies or immutable values:

1. A frame of a synthetic scene (``SyntheticScene.image``). A scene is a pure function of its
   settings (``synth.py``): the clip settings, the seed and, for ``RetimedScene``, the scene
   frames. That is the key.
2. A frame image that ``ImageFolderSource`` reads from a file (``cv2.imread``). The key is the
   SHA-256 of the bytes of the file, so a file that a test writes again is read again.
3. The camera motion that ``pipeline.run_clip`` measures (``motion.compute_motion_field``), for
   a synthetic source or an image folder. The key is the source (as in 1 or 2), the people
   boxes and the scale. A cached result is built again as a new ``MotionField``, and the log
   line of each cut is written again, as the measurement writes it.
4. The paint alignment of a mapping in a frame (``geometry.paint_alignment``) that the pipeline
   and the probes compute. The key is the SHA-256 of the frame and the bytes of the matrix.
5. The JPEG bytes of an image that a model request carries (``vlm.steps.encode_jpeg``). The key
   is the SHA-256 of the pixels, the shape and the quality.
6. The git state of this checkout (``cli.code_state`` of ``cli.repository_root()``), which every
   ``run``, ``demo`` and ``probe`` command reads with two git processes. The key is the git
   variables of the environment. The checkout does not change while the tests run. The git
   state of any other folder is read each time.

A cached frame is read-only; every caller gets its own copy, so a test that draws on a frame
changes only that copy. The other results are immutable or are built again for each caller.

Where the caches are not used:

- In the modules that test these functions themselves (``NO_CACHE_MODULES``): there a second
  call computes again, so a test that compares two calls compares two computations.
- While a test has replaced a function or a setting of ``motion``, ``geometry`` or ``cv2`` (a
  monkeypatch): then results 3, 4 and 5 are computed, not read.
- For any other class of scene or frame source.

No result changes: a cached value is what the code under test computed for the same input in
the same process. ``PITCHVLM_TEST_CACHE=0`` turns every cache off.
"""

from __future__ import annotations

import collections
import hashlib
import logging
import os
import threading
import types
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from pitchvlm import cli, frames, geometry, motion, pipeline, probes, synth
from pitchvlm.schemas import Box
from pitchvlm.vlm import steps

#: Test modules that test the cached functions themselves: no cache there.
NO_CACHE_MODULES = frozenset({"test_synth", "test_motion", "test_frames", "test_geometry", "test_vlm_steps"})
#: Upper limit of the two frame caches together, in bytes. The oldest frame is dropped first.
FRAME_CACHE_BYTES = 2_000_000_000
#: The variable that turns every cache off: ``PITCHVLM_TEST_CACHE=0``.
CACHE_VARIABLE = "PITCHVLM_TEST_CACHE"

_SCENE_TYPES = (synth.SyntheticScene, synth.RetimedScene)
_SYNTHETIC_SOURCES = (synth.SyntheticSource, synth.ScoreGraphicSource)
_enabled = os.environ.get(CACHE_VARIABLE, "1") != "0"
_module_allows_cache = True


def _active() -> bool:
    return _enabled and _module_allows_cache


def _digest(data: Any) -> bytes:
    return hashlib.sha256(data).digest()


def _values(module: types.ModuleType) -> dict[str, object]:
    return {name: value for name, value in vars(module).items() if not name.startswith("__")}


#: The functions and settings of the modules that results 3, 4 and 5 use, as the test process
#: started with them. A test that replaces one of them (a function or a threshold) gets a real
#: computation.
_ORIGINAL_VALUES = {module: _values(module) for module in (motion, geometry, cv2)}


def _functions_unchanged() -> bool:
    for module, values in _ORIGINAL_VALUES.items():
        current = vars(module)
        if any(current.get(name) is not value for name, value in values.items()):
            return False
    return True


# --- 1 and 2: frames ----------------------------------------------------------------------


class _FrameCache:
    """Read-only frames by key, at most ``limit`` bytes; the oldest frame is dropped first."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.frames: collections.OrderedDict[tuple[Any, ...], np.ndarray] = collections.OrderedDict()
        self.size = 0
        # The pipeline reads frames from several threads.
        self.lock = threading.Lock()

    def get(self, key: tuple[Any, ...]) -> np.ndarray | None:
        """Return a copy of the cached frame, or None."""
        with self.lock:
            frame = self.frames.get(key)
            if frame is None:
                return None
            self.frames.move_to_end(key)
        return frame.copy()

    def put(self, key: tuple[Any, ...], frame: np.ndarray) -> None:
        kept = frame.copy()
        kept.flags.writeable = False
        with self.lock:
            if key in self.frames:
                return
            self.frames[key] = kept
            self.size += kept.nbytes
            while self.size > self.limit and len(self.frames) > 1:
                _, dropped = self.frames.popitem(last=False)
                self.size -= dropped.nbytes


_frames = _FrameCache(FRAME_CACHE_BYTES)
_original_scene_init = synth.SyntheticScene.__init__
_original_image = synth.SyntheticScene.image
_original_load = frames.ImageFolderSource._load


def _scene_key(scene: synth.SyntheticScene) -> tuple[Any, ...] | None:
    """Return the key of a scene of an exact cached class, else None."""
    if type(scene) not in _SCENE_TYPES or "_test_seed" not in scene.__dict__:
        return None
    info = scene.info
    scene_frames = scene.__dict__.get("_scene_frames")
    seed = scene.__dict__["_test_seed"]
    return (type(scene), info.clip_id, info.n_frames, info.fps, info.width, info.height, seed, scene_frames)


def _scene_init(self: synth.SyntheticScene, *args: Any, **kwargs: Any) -> None:
    _original_scene_init(self, *args, **kwargs)
    # The seed is a keyword-only argument with the default 0 (``SyntheticScene.__init__``).
    self.__dict__["_test_seed"] = kwargs.get("seed", 0)


def _cached_image(self: synth.SyntheticScene, frame: int) -> np.ndarray:
    key = _scene_key(self) if _active() else None
    if key is None:
        return _original_image(self, frame)
    full_key = ("scene", *key, frame)
    image = _frames.get(full_key)
    if image is None:
        image = _original_image(self, frame)
        _frames.put(full_key, image)
    return image


def _file_key(path: Path) -> tuple[Any, ...] | None:
    try:
        return ("file", _digest(Path(path).read_bytes()))
    except OSError:
        return None


def _cached_load(path: Path) -> np.ndarray:
    key = _file_key(path) if _active() else None
    if key is None:
        return _original_load(path)
    image = _frames.get(key)
    if image is None:
        image = _original_load(path)
        _frames.put(key, image)
    return image


# --- 3: camera motion ---------------------------------------------------------------------

_original_compute_motion_field = motion.compute_motion_field
_motions: dict[tuple[Any, ...], tuple[tuple[np.ndarray | None, ...], tuple[int, ...]]] = {}
_motion_logger = logging.getLogger(motion.__name__)


def _source_key(source: Any) -> tuple[Any, ...] | None:
    if type(source) in _SYNTHETIC_SOURCES:
        scene_key = _scene_key(source._scene)
        return None if scene_key is None else (type(source), scene_key)
    if type(source) is frames.ImageFolderSource:
        keys = []
        for frame in range(source.info.n_frames):
            key = _file_key(source.path(frame))
            if key is None:
                return None
            keys.append(key)
        return (type(source), source.info, tuple(keys))
    return None


def _boxes_key(people: Mapping[int, Sequence[Box]] | None) -> tuple[Any, ...]:
    if people is None:
        return ()
    return tuple((frame, tuple(people[frame])) for frame in sorted(people))


def _cached_compute_motion_field(
    source: Any, *, people: Mapping[int, Sequence[Box]] | None = None, downscale: float = 0.5
) -> motion.MotionField:
    source_key = _source_key(source) if _active() and _functions_unchanged() else None
    if source_key is None:
        return _original_compute_motion_field(source, people=people, downscale=downscale)
    key = (source_key, _boxes_key(people), downscale)
    kept = _motions.get(key)
    if kept is None:
        field = _original_compute_motion_field(source, people=people, downscale=downscale)
        steps = tuple(None if step is None else step.copy() for step in field._forward)
        _motions[key] = (steps, field.cuts)
        return field
    steps, cuts = kept
    for frame in cuts:
        _motion_logger.info("clip %s: frame %d starts a new shot", source.info.clip_id, frame)
    return motion.MotionField([None if step is None else step.copy() for step in steps], cuts)


# --- 4: paint alignment -------------------------------------------------------------------

_original_paint_alignment = geometry.paint_alignment
_alignments: dict[tuple[Any, ...], geometry.PaintAlignment] = {}


def _cached_paint_alignment(image: np.ndarray, H: np.ndarray) -> geometry.PaintAlignment:
    if not (_active() and _functions_unchanged()):
        return _original_paint_alignment(image, H)
    pixels = np.ascontiguousarray(image)
    matrix = np.ascontiguousarray(H)
    key = (pixels.shape, pixels.dtype.str, _digest(pixels.data), matrix.shape, matrix.dtype.str, matrix.tobytes())
    alignment = _alignments.get(key)
    if alignment is None:
        alignment = _original_paint_alignment(image, H)
        _alignments[key] = alignment
    return alignment


# --- 5: JPEG bytes -----------------------------------------------------------------------

_original_encode_jpeg = steps.encode_jpeg
_jpegs: dict[tuple[Any, ...], bytes] = {}


def _cached_encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    usable = isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.ndim == 3 and image.size > 0
    if not (usable and _active() and _functions_unchanged()):
        return _original_encode_jpeg(image, quality)
    pixels = np.ascontiguousarray(image)
    key = (pixels.shape, _digest(pixels.data), quality)
    data = _jpegs.get(key)
    if data is None:
        data = _original_encode_jpeg(image, quality)
        _jpegs[key] = data
    return data


# --- 6: git state of this checkout ------------------------------------------------------

_original_code_state = cli.code_state
_CHECKOUT = cli.repository_root().resolve()
_checkout_states: dict[tuple[tuple[str, str], ...], cli.CodeState] = {}


def _git_environment() -> tuple[tuple[str, str], ...]:
    """Return the variables that change what git reads: a test may set them (fixture ``private_git``)."""
    names = ("HOME", "XDG_CONFIG_HOME", "PATH")
    return tuple(sorted((name, value) for name, value in os.environ.items() if name in names or name.startswith("GIT_")))


def _cached_code_state(repo: Path) -> cli.CodeState:
    if not _active() or Path(repo).resolve() != _CHECKOUT:
        return _original_code_state(repo)
    key = _git_environment()
    state = _checkout_states.get(key)
    if state is None:
        state = _original_code_state(repo)
        _checkout_states[key] = state
    return state


# --- Fixtures -----------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _pure_result_caches() -> Iterator[None]:
    """Install the caches for the whole test session, and take them out at its end."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(synth.SyntheticScene, "__init__", _scene_init)
        patch.setattr(synth.SyntheticScene, "image", _cached_image)
        patch.setattr(frames.ImageFolderSource, "_load", staticmethod(_cached_load))
        patch.setattr(pipeline, "compute_motion_field", _cached_compute_motion_field)
        patch.setattr(pipeline, "paint_alignment", _cached_paint_alignment)
        patch.setattr(probes, "paint_alignment", _cached_paint_alignment)
        patch.setattr(steps, "encode_jpeg", _cached_encode_jpeg)
        patch.setattr(cli, "code_state", _cached_code_state)
        yield


@pytest.fixture(scope="module", autouse=True)
def _cache_rule_of_the_module(request: pytest.FixtureRequest) -> Iterator[None]:
    """Turn the caches off in the modules that test the cached functions themselves."""
    global _module_allows_cache
    name = request.module.__name__.rsplit(".", 1)[-1]
    _module_allows_cache = name not in NO_CACHE_MODULES
    yield
    _module_allows_cache = True
