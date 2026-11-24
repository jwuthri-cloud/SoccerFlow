"""Stored model answers (cassettes) and the backend that replays them.

A cassette is one JSON text file for one request. It holds the cache key, the
place of the request in the pipeline, the stored part of the request context,
the model settings, the prompt name and hash, the image resolutions and hashes,
and the answer. The stored part of the context is the frame numbers of the
images, for example the frames of the crops of a shirt-number request
(``VLMRequest.stored_context``). A cassette never holds image bytes, and it
never holds a box of the request context: the boxes of a probe request are
boxes of the label files. The boxes that the model answers are part of the
answer text. So cassettes can leave the machine that has the footage and the
labels, and they can be read as text anywhere. A video part is kept as its
identity (for a written file: the frames of the clip, the rate and size at
which they were written, the range, the sampling rate and the resolution),
the hash of the bytes that were sent and the hash of the pixels of its frames
(``VideoPart.stored``): never the path of the file and never its bytes.

A replay is another thing: it builds every request again from the frames of
the clip, so it runs on the machine that has the footage.

The cache key does not depend on the pixels or on the service tier (see
``VLMRequest.cache_key``). So the key cannot show that a clip was replaced
under the same name. A hit therefore compares the image hashes of the request
with the recorded ones. When they differ, the stored answer can describe other
pixels: the hit raises ``OtherImagesError``, and the model is not asked. A
store that was opened with ``allow_other_images=True`` returns the stored
answer and logs a warning.

A written video is in the key by its frames and settings, not by its bytes, so
the same frames written again with other bytes (another codec or OpenCV build)
find their stored answer. Such a hit is used, logged and counted
(``CassetteStore.other_video_bytes``; the commands print the count and write it
into their cost record): it is neither silent nor an error. A hit whose video
shows other pixels of its frames follows the rule of other image bytes.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import threading
from collections.abc import Callable
from itertools import zip_longest
from pathlib import Path
from typing import Any

from pitchvlm.vlm.types import Backend, BilledAnswerError, CacheMiss, OtherImagesError, VLMRequest, VLMResponse

logger = logging.getLogger(__name__)


def _folder_name(value: str, what: str) -> str:
    """Return the value when it can be the name of one folder, else raise ValueError."""
    if not value or value in (".", "..") or "/" in value or "\\" in value:
        raise ValueError(f"the {what} {value!r} cannot be used as a folder name")
    return value


def _cassette(request: VLMRequest, response: VLMResponse) -> dict[str, Any]:
    """Return the content of the cassette of a request. Image bytes are never part of it.

    Of the request context, only the stored part is written
    (``VLMRequest.stored_context``): the frame numbers, and never the boxes.
    The cache key stands for the whole context.

    A video part is written as ``VideoPart.stored``: its identity, the hash of
    the bytes that were sent and the hash of the pixels of its frames. Its path
    and its bytes are never written.
    """
    cassette = {
        "key": request.cache_key(),
        "meta": request.meta.model_dump(mode="json"),
        "context": request.stored_context(),
        "model": request.model,
        "thinking_level": request.thinking_level,
        "prompt_name": request.prompt_name,
        "prompt_hash": request.prompt_hash,
        "resolutions": list(request.resolutions),
        "image_hashes": list(request.image_hashes),
        "response": response.model_dump(mode="json", exclude=_unset_provider_fields(response)),
    }
    if request.videos:
        # Only a request with video parts gets this entry, so an image cassette keeps the content of before.
        cassette["videos"] = [video.stored() for video in request.videos]
    return cassette


#: Fields of an answer that only an OpenRouter answer fills.
PROVIDER_FIELDS = ("reported_usd", "served_by")


def _unset_provider_fields(response: VLMResponse) -> set[str]:
    """Return the fields of ``PROVIDER_FIELDS`` that an answer leaves empty.

    A cassette does not write them, so a Gemini cassette has the same content as
    before OpenRouter was added. Reading such a cassette gives them their default, None.
    """
    return {name for name in PROVIDER_FIELDS if getattr(response, name) is None}


def _hash_tuple(value: Any) -> tuple[str, ...]:
    """Return the image hashes of a cassette as a tuple. A value that is not a list of texts raises TypeError."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError("the image hashes are not a list of texts")
    return tuple(value)


def _video_list(value: Any) -> tuple[dict[str, Any], ...]:
    """Return the video entries of a cassette as a tuple. A value that is not a list of JSON objects raises TypeError."""
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise TypeError("the videos are not a list of JSON objects")
    return tuple(value)


def _read_cassette(
    text: str, path: Path, key: str
) -> tuple[VLMResponse, tuple[str, ...], tuple[dict[str, Any], ...]]:
    """Read the answer, the recorded image hashes and the recorded videos out of a cassette text.

    A cassette that is damaged, or that belongs to another request, raises
    ValueError. It is never treated as a missing answer.
    """
    try:
        cassette = json.loads(text)
        stored_key = cassette["key"]
        image_hashes = _hash_tuple(cassette["image_hashes"])
        videos = _video_list(cassette.get("videos", []))
        response = VLMResponse.model_validate(cassette["response"])
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f"the cassette {path} cannot be read: {error}") from error
    if stored_key != key:
        raise ValueError(f"the cassette {path} belongs to another request: its key is {stored_key!r}")
    return response, image_hashes, videos


def _changed_images(request: VLMRequest, recorded_hashes: tuple[str, ...]) -> list[int]:
    """Return the places, counted from 1, of the images of a request that its cassette was not recorded with.

    An image without a recorded hash, and a recorded hash without an image, count as changed.
    """
    pairs = zip_longest(recorded_hashes, request.image_hashes)
    return [place for place, (recorded, shown) in enumerate(pairs, start=1) if recorded != shown]


def _changed_videos(request: VLMRequest, recorded: tuple[dict[str, Any], ...]) -> tuple[list[int], list[int]]:
    """Return the places, counted from 1, of the videos of a request whose frames show other pixels than when the
    answer was stored, and of those that show the same pixels from other bytes (the same frames written again).

    The key holds the identity of each video, so a hit has the same frames and settings. A
    video without a recorded entry, a recorded entry without a video, and a written video whose
    entry has no hash of its pixels count as other pixels. A file that the package did not
    write has the hash of its bytes in the key, so its bytes cannot differ on a hit.
    """
    pixels: list[int] = []
    sent: list[int] = []
    for place, (entry, video) in enumerate(zip_longest(recorded, request.videos), start=1):
        if entry is None or video is None:
            pixels.append(place)
        elif video.source is not None and entry.get("pixels_sha256") != video.source.pixels_sha256:
            pixels.append(place)
        elif entry.get("sha256") != video.sha256:
            sent.append(place)
    return pixels, sent


def _other_images_text(request: VLMRequest, changed: list[int], videos: list[int]) -> str:
    """Name a request whose stored answer was recorded with other image bytes or video pixels, and which changed."""
    meta = request.meta
    what = []
    if changed:
        what.append("changed images, counted from 1: " + ", ".join(str(place) for place in changed))
    if videos:
        what.append("videos whose frames show other pixels, counted from 1: " + ", ".join(str(place) for place in videos))
    return (
        f"the stored answer of the {meta.step} request of clip {meta.clip_id} (frame {meta.frame}, "
        f"item {meta.item}, repeat {meta.repeat}) was recorded with other image bytes ({'; '.join(what)})"
    )


def _write_atomic(path: Path, text: str) -> None:
    """Write a text file through a temporary file in the same folder.

    The temporary file replaces the target in one step, so a reader sees the
    old content or the new content, never a part of a file. A write that fails
    leaves no temporary file.
    """
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class CassetteStore:
    """A folder of cassettes: ``root/<clip_id>/<step>/<cache_key>.json``.

    ``allow_other_images`` says what a hit does when the request shows other
    image bytes than the cassette was recorded with. False (the default): the
    hit raises ``OtherImagesError``. True: the stored answer is returned and a
    warning is logged. Use True only for the same footage: after an update of
    the image library, which can encode the same picture to other bytes, and
    for a kit legend that was asked from other frames of the same clip (the
    key of a kit legend request has no frame).
    """

    def __init__(self, root: Path, *, allow_other_images: bool = False) -> None:
        self.root = Path(root)
        self._allow_other_images = allow_other_images
        self._lock = threading.Lock()
        self._other_video_bytes = 0

    @property
    def other_video_bytes(self) -> int:
        """The hits so far whose stored answer was recorded from other bytes of the same video frames."""
        with self._lock:
            return self._other_video_bytes

    def path_for(self, request: VLMRequest) -> Path:
        """Return the cassette file of a request. The file may not exist."""
        clip_folder = _folder_name(request.meta.clip_id, "clip id")
        step_folder = _folder_name(request.meta.step, "step")
        return self.root / clip_folder / step_folder / f"{request.cache_key()}.json"

    def get(self, request: VLMRequest) -> VLMResponse | None:
        """Return the stored answer of a request, or None when no answer is stored.

        The key has no pixels. So the image hashes of the request are compared
        with the recorded ones, and so are the hashes of the pixels of the frames
        of each written video. When they differ, ``OtherImagesError`` is
        raised: the stored answer can describe other pixels. A store with
        ``allow_other_images`` returns the answer and logs a warning instead.
        A video that shows the same pixels from other bytes (written again) is
        used: the hit is logged and counted (``other_video_bytes``).
        """
        path = self.path_for(request)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        response, recorded_hashes, recorded_videos = _read_cassette(text, path, request.cache_key())
        changed = _changed_images(request, recorded_hashes)
        other_pixels, other_bytes = _changed_videos(request, recorded_videos)
        if changed or other_pixels:
            found = _other_images_text(request, changed, other_pixels)
            if not self._allow_other_images:
                raise OtherImagesError(
                    f"{found}. It is not used, and the model is not asked. There are two ways out. For new footage: "
                    "give the clip another name (the name of its file or folder), so that it gets answers of its own. "
                    "For the same footage: allow the stored answers with the option allow_other_images of the cassette "
                    "store (the option --allow-other-images of the commands run and probe). Use it after an update of "
                    f"the image library, and for a kit legend that was asked from other frames of the same clip. "
                    f"Cassette: {path}"
                )
            logger.warning(
                "Other image bytes are allowed, so this answer is used: %s. It can describe other pixels. Cassette: %s",
                found,
                path,
            )
        if other_bytes:
            self._note_rewritten(request, other_bytes, path)
        return response

    def _note_rewritten(self, request: VLMRequest, places: list[int], path: Path) -> None:
        """Count and log a stored answer that is used for a video written again with other bytes from the same frames."""
        with self._lock:
            self._other_video_bytes += 1
        meta = request.meta
        logger.warning(
            "The stored answer of the %s request of clip %s (frame %s, item %s, repeat %s) was recorded from "
            "other bytes of the same video frames (videos, counted from 1: %s): the video was written again. "
            "It is used, and counted. Cassette: %s",
            meta.step,
            meta.clip_id,
            meta.frame,
            meta.item,
            meta.repeat,
            ", ".join(str(place) for place in places),
            path,
        )

    def check_writable(self, request: VLMRequest) -> None:
        """Raise ``OSError`` when the cassette of a request could not be written. Nothing is created or changed.

        The folder of the cassette must exist or be one that can be made, and
        it must take a new file. A paying backend calls this before it asks
        the model: an answer that cannot be stored must not be paid for.
        """
        folder = self.path_for(request).parent
        existing = next(place for place in (folder, *folder.parents) if place.exists())
        if not (existing.is_dir() and os.access(existing, os.W_OK | os.X_OK)):
            raise PermissionError(errno.EACCES, "the folder of the stored answers cannot be written", str(existing))

    def put(self, request: VLMRequest, response: VLMResponse) -> Path:
        """Store the answer of a request and return its file. An older answer is replaced."""
        path = self.path_for(request)
        text = json.dumps(_cassette(request, response), indent=2) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(path, text)
        return path


class CachedBackend:
    """A backend that answers from the cassette store first.

    A hit returns the stored answer, and the model is not asked again. A hit
    whose request shows other image bytes than the stored answer was recorded
    with follows the rule of the store: ``OtherImagesError``, or the stored
    answer with a warning. A miss checks that the cassette can be written,
    calls ``inner``, stores the answer and calls ``on_new_answer``, for example
    to write the cost ledger. With ``inner=None`` the backend only replays: a
    miss raises ``CacheMiss``.

    A paid answer always reaches ``on_new_answer``, also when its cassette
    cannot be written (a full disk): the error of the store is raised after
    the callback ran. An error from ``on_new_answer`` (for example a budget
    stop) goes to the caller after the store was tried, so a paid answer is
    never lost because of it. It replaces an error of the store.

    ``inner`` can raise ``BilledAnswerError``: the call was billed, but its
    answer cannot be used. Then the cost record of the error goes to
    ``on_new_answer`` like a new answer, no cassette is written, and the error
    is raised again. This is the only place that reports the cost of such a
    call, so code above this backend must not report it again. The next call
    with the same request asks ``inner`` again. An error from ``on_new_answer``
    also replaces a ``BilledAnswerError``.

    One instance can be used from several threads. Calls to ``inner`` run in
    parallel. The file write and ``on_new_answer`` run one at a time, under a
    lock. Two threads that miss on the same request at the same moment may both
    call ``inner``: both answers are paid and reported, and the later one stays
    in the store. The same holds for two commands that work on one clip at the
    same time. No lock keeps them apart, so the second answer logs a warning
    that the request was paid twice.
    """

    def __init__(
        self,
        store: CassetteStore,
        inner: Backend | None = None,
        on_new_answer: Callable[[VLMRequest, VLMResponse], None] | None = None,
    ) -> None:
        self._store = store
        self._inner = inner
        self._on_new_answer = on_new_answer
        self._lock = threading.Lock()

    def generate(self, request: VLMRequest) -> VLMResponse:
        """Return the stored answer of a request, or get a new answer from ``inner`` and store it.

        Before ``inner`` is asked, the store must be able to take the cassette:
        ``OSError`` is raised when it cannot. A ``BilledAnswerError`` of
        ``inner`` is raised again after its cost record went to
        ``on_new_answer``. Nothing is stored for it.
        """
        stored = self._store.get(request)
        if stored is not None:
            return stored
        if self._inner is None:
            meta = request.meta
            raise CacheMiss(
                f"no stored answer for the {meta.step} request of clip {meta.clip_id} "
                f"(frame {meta.frame}, item {meta.item}, repeat {meta.repeat}, key {request.cache_key()})"
            )
        self._store.check_writable(request)
        try:
            response = self._inner.generate(request)
        except BilledAnswerError as error:
            with self._lock:
                self._report(request, error.billed)
            raise
        with self._lock:
            if self._store.path_for(request).exists():
                meta = request.meta
                logger.warning(
                    "The %s request of clip %s (frame %s, item %s, repeat %s, key %s) got a stored answer while "
                    "the model was asked: another command or thread paid for the same request. Both answers are "
                    "in the cost ledger, and the newer one stays in the store. Run one paid command for a clip "
                    "at a time.",
                    meta.step,
                    meta.clip_id,
                    meta.frame,
                    meta.item,
                    meta.repeat,
                    request.cache_key(),
                )
            try:
                self._store.put(request, response)
            finally:
                # The answer is paid, so its cost is reported also when the cassette could not be written.
                self._report(request, response)
        return response

    def _report(self, request: VLMRequest, response: VLMResponse) -> None:
        """Pass a new answer, or the cost record of a rejected one, to ``on_new_answer``. The caller holds the lock."""
        if self._on_new_answer is not None:
            self._on_new_answer(request, response)
