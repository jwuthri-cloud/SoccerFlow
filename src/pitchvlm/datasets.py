"""Dataset download and the fixed split list.

The data is SoccerNet Game State Reconstruction (SN-GSR). It is published on
Hugging Face as one zip archive per split. This module downloads an archive,
extracts clip folders from it, lists the clips of a split folder and makes the
split list: the file that says which clip is used for which measurement.

Downloads run on the run machine only. Archives and clips are large and must
stay outside the repository, for example below ``~/pitchvlm-data``. The data
is for research use only.

A clip folder is a folder that directly holds a ``Labels-GameState.json``
file. The name of the folder is the clip name, for example ``SNGS-021``.

This module uses the standard library only.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Sequence
from pathlib import Path
from typing import Any, BinaryIO

#: Dataset repository on Hugging Face.
DATASET_REPO = "SoccerNet/SN-GSR-2025"
#: Splits that can be downloaded. Each split is one zip archive in the repository.
SPLITS = ("train", "valid", "test")
#: Name of the label file of a clip. A folder that holds it is a clip folder.
LABEL_FILE_NAME = "Labels-GameState.json"
#: Folder below the data root that holds the downloaded archives.
ARCHIVE_FOLDER = "archives"
#: End of the file name of a download that is not complete.
PARTIAL_SUFFIX = ".part"
#: Free disk space that must be left after a download, in bytes (20 GB). It is the room for the extracted clips.
FREE_SPACE_RESERVE_BYTES = 20 * 10**9
#: Seconds that the default opener waits for the server: for the connection and for each read of the answer.
#: It is not a limit for the whole download.
SOCKET_TIMEOUT_S = 60.0
#: Number of clips in each fixed group of the validation split, in the order in which the shuffled list is cut.
VALID_GROUP_SIZES = {"probe": 24, "dev": 6, "fresh": 3}
#: Group that takes the validation clips that are left after the fixed groups.
RESERVE_GROUP = "reserve"
#: Group of the test clips that are used for the final score.
FINAL_GROUP = "final"
#: Number of clips in the group ``final``.
TEST_FINAL_SIZE = 10

#: Opens a request, like ``urllib.request.urlopen``. The answer is a context manager with the
#: attributes ``status`` and ``headers`` and the method ``read(size)``.
Opener = Callable[[urllib.request.Request], Any]

_GB = 10**9
#: ``Content-Range`` of an answer with status 206: ``bytes <first>-<last>/<total>``.
_CONTENT_RANGE = re.compile(r"bytes (\d+)-\d+/(?:\d+|\*)")
#: HTTP status of an answer to a byte range that starts at or after the end of the file.
_RANGE_NOT_SATISFIABLE = 416
#: ``Content-Range`` of an answer with status 416: ``bytes */<total>``.
_UNSATISFIED_RANGE = re.compile(r"bytes \*/(\d+)")
#: Start of a path part that names a Windows drive, for example ``C:``.
_DRIVE = re.compile(r"[A-Za-z]:")


def split_url(split: str) -> str:
    """Return the public download address of the archive of one split.

    ``split`` is ``train``, ``valid`` or ``test``. Any other name raises ``ValueError``.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; the splits are {', '.join(SPLITS)}")
    return f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{split}.zip"


# --- Download ------------------------------------------------------------------------


def _open_with_timeout(request: urllib.request.Request) -> Any:
    """Open a request with ``urllib.request.urlopen`` and a socket timeout of ``SOCKET_TIMEOUT_S`` seconds.

    Without a timeout, a connection that stalls blocks a read for ever. With the timeout, the
    open or the read raises ``TimeoutError`` or ``urllib.error.URLError`` when the server sends
    nothing for that time. Both are kinds of ``OSError``.
    """
    return urllib.request.urlopen(request, timeout=SOCKET_TIMEOUT_S)


def download_split(
    root: Path,
    split: str,
    *,
    opener: Opener = _open_with_timeout,
    chunk_bytes: int = 1 << 20,
) -> Path:
    """Download the archive of one split to ``root/archives/<split>.zip`` and return that path.

    - An archive that is already there is returned as it is. No request is sent.
    - The bytes go to ``<split>.zip.part`` first. The file gets its final name only when it is
      complete, so a file with the final name is always a complete download.
    - A ``.part`` file of an earlier call is continued. The request asks for the missing bytes
      with an HTTP ``Range`` header, and the answer is appended. When the server sends the whole
      file instead (status 200), the ``.part`` file is written again from its start. To start a
      download again by hand, delete the ``.part`` file.
    - A ``.part`` file can already hold the whole archive: the earlier call was stopped after the
      last byte and before the file got its final name. The server then answers the range with
      status 416 and names the size of the archive. When that size is the size of the ``.part``
      file, the file gets its final name. With another size, or without a size, ``OSError``
      names the ``.part`` file.
    - Before the first byte is written, the free disk space is checked. It must hold the
      ``Content-Length`` of the answer plus 20 GB (``FREE_SPACE_RESERVE_BYTES``).

    ``opener`` is called with one ``urllib.request.Request``. The default is
    ``urllib.request.urlopen`` with a socket timeout of 60 seconds (``SOCKET_TIMEOUT_S``), so
    that a connection that stalls raises an error and does not block for ever. Tests pass a
    stand-in, so that they never touch the network. ``chunk_bytes`` is the size of one read.

    ``ValueError`` is raised for an unknown split and for a ``chunk_bytes`` that is not positive.
    ``OSError`` is raised when the disk space is too small, when the answer cannot be used (an
    unexpected status, no ``Content-Length``, a ``Content-Range`` that does not start where the
    ``.part`` file ends), when the connection stalls for longer than the timeout of the opener
    and when the connection ends early. In the last two cases the ``.part`` file stays, and the
    next call continues it. An HTTP error of the opener (``urllib.error.HTTPError``, a kind of
    ``OSError``) is passed on, but for the status 416 described above.
    """
    url = split_url(split)
    if chunk_bytes <= 0:
        raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes}")
    folder = Path(root) / ARCHIVE_FOLDER
    archive = folder / f"{split}.zip"
    if archive.is_file():
        return archive
    folder.mkdir(parents=True, exist_ok=True)
    partial = folder / f"{archive.name}{PARTIAL_SUFFIX}"
    offset = partial.stat().st_size if partial.is_file() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with _answer(opener, request) as response:
        if offset and response.status == _RANGE_NOT_SATISFIABLE:
            _check_partial_is_complete(response, offset, partial, url)
        else:
            append = _continues_at(response, offset, url)
            expected = _content_length(response, url)
            _check_free_space(folder, expected)
            with partial.open("ab" if append else "wb") as out:
                received = _copy(response, out, chunk_bytes)
            if received != expected:
                raise OSError(
                    f"{url}: the connection ended early, after {received} of {expected} bytes; "
                    f"call again to continue the download from {partial}"
                )
    partial.replace(archive)
    return archive


def _answer(opener: Opener, request: urllib.request.Request) -> Any:
    """Open a request and return its answer, also when the answer has the status 416.

    ``urllib.request.urlopen`` raises ``urllib.error.HTTPError`` for a status of 400 or more. That
    error is the answer itself, with ``status`` and ``headers``. The error of the status 416 is
    returned like an answer, so that the caller can read its headers. Every other error is raised.
    """
    try:
        return opener(request)
    except urllib.error.HTTPError as error:
        if error.code != _RANGE_NOT_SATISFIABLE:
            raise
        return error


def _check_partial_is_complete(response: Any, offset: int, partial: Path, url: str) -> None:
    """Raise ``OSError`` unless a 416 answer says that the ``.part`` file holds the whole archive.

    The request asked for the bytes from ``offset`` on, the size of the ``.part`` file. The
    status 416 means that the archive has no such byte. Its ``Content-Range`` header names the
    size of the archive: ``bytes */<total>``. The file is complete when that size is ``offset``.
    """
    content_range = response.headers.get("Content-Range", "")
    match = _UNSATISFIED_RANGE.fullmatch(content_range.strip())
    if match is None:
        raise OSError(
            f"{url}: the server has no byte {offset} (HTTP status {_RANGE_NOT_SATISFIABLE}), and its answer names "
            f"no size of the archive (Content-Range {content_range!r}). So {partial}, which holds {offset} bytes, "
            "cannot be checked. Compare its size with the size on the dataset page, or delete it to start again."
        )
    total = int(match.group(1))
    if total != offset:
        raise OSError(
            f"{url}: {partial} holds {offset} bytes, but the archive has {total} bytes. "
            "The file is not a part of this archive: delete it to start the download again."
        )


def _continues_at(response: Any, offset: int, url: str) -> bool:
    """Tell whether the body of an answer continues a file that already holds ``offset`` bytes.

    Status 206 means that the body starts at the byte that ``Content-Range`` names. This byte
    must be ``offset``, and the answer is True. Status 200 means that the body is the whole
    file, and the answer is False. Any other status raises ``OSError``.
    """
    if response.status == 200:
        return False
    if response.status != 206:
        raise OSError(f"{url}: unexpected HTTP status {response.status}")
    content_range = response.headers.get("Content-Range", "")
    match = _CONTENT_RANGE.fullmatch(content_range.strip())
    if match is None or int(match.group(1)) != offset:
        raise OSError(
            f"{url}: the answer has Content-Range {content_range!r}, but the download must continue at byte {offset}"
        )
    return True


def _content_length(response: Any, url: str) -> int:
    """Return the number of body bytes that an answer announces. A missing or wrong header raises ``OSError``."""
    text = response.headers.get("Content-Length")
    if text is None or not text.strip().isdecimal():
        raise OSError(f"{url}: the answer has no usable Content-Length header (found {text!r})")
    return int(text)


def _check_free_space(folder: Path, incoming_bytes: int) -> None:
    """Raise ``OSError`` when the disk of ``folder`` cannot take the download and keep the reserve free."""
    free = shutil.disk_usage(folder).free
    if free < incoming_bytes + FREE_SPACE_RESERVE_BYTES:
        raise OSError(
            f"not enough free disk space in {folder}: the download needs {incoming_bytes / _GB:.1f} GB plus "
            f"{FREE_SPACE_RESERVE_BYTES / _GB:.0f} GB of reserve, but only {free / _GB:.1f} GB are free"
        )


def _copy(response: Any, out: BinaryIO, chunk_bytes: int) -> int:
    """Copy the body of an answer into an open file, chunk by chunk, and return the number of bytes."""
    received = 0
    while chunk := response.read(chunk_bytes):
        out.write(chunk)
        received += len(chunk)
    return received


# --- Clips ---------------------------------------------------------------------------


def extract_clips(archive: Path, dest: Path, clips: Collection[str] | None = None) -> list[str]:
    """Extract clip folders from a split archive into ``dest`` and return the sorted names of these clips.

    A clip folder is a folder of the archive that directly holds a ``Labels-GameState.json``
    file. It may lie at the top of the archive or deeper, for example below ``valid/``. Every
    file below a clip folder is written to ``dest/<clip name>/...``, so that
    ``list_clips(dest)`` finds the clip. A file that is already there is written again. Files
    outside the clip folders are not extracted.

    ``clips`` names the wanted clips. ``None`` means all clips of the archive.

    ``ValueError`` is raised, before any file is written, when a member of the archive has an
    absolute path, a drive letter or a ``..`` part, when a wanted clip is not in the archive,
    when two clip folders have the same name, and when a label file lies at the top of the
    archive.
    """
    with zipfile.ZipFile(archive) as bundle:
        members = [(info, _member_parts(info.filename)) for info in bundle.infolist()]
        # A name that ends with a separator is a folder entry. It has no content to write.
        files = [(info, parts) for info, parts in members if not info.filename.endswith(("/", "\\"))]
        folders = _clip_folders((parts for _, parts in files), archive)
        names = set(folders.values())
        wanted = names if clips is None else set(clips)
        missing = sorted(wanted - names)
        if missing:
            raise ValueError(f"{archive}: the clips {missing} are not in the archive; it holds {len(names)} clips")
        for info, parts in files:
            for depth in range(1, len(parts)):
                name = folders.get(parts[:depth])
                if name in wanted:
                    _write_member(bundle, info, Path(dest).joinpath(name, *parts[depth:]))
    return sorted(wanted)


def _member_parts(name: str) -> tuple[str, ...]:
    """Return the path parts of an archive member. A name that could leave the target folder raises ``ValueError``.

    A backslash counts as a path separator too, so that the check does not depend on the
    operating system.
    """
    normal = name.replace("\\", "/")
    parts = tuple(part for part in normal.split("/") if part not in ("", "."))
    if normal.startswith("/") or any(_DRIVE.match(part) for part in parts):
        raise ValueError(f"archive member {name!r} has an absolute path or a drive letter")
    if ".." in parts:
        raise ValueError(f"archive member {name!r} has a '..' part")
    return parts


def _clip_folders(files: Iterable[tuple[str, ...]], archive: Path) -> dict[tuple[str, ...], str]:
    """Map the path parts of every clip folder of the archive to the clip name.

    ``files`` holds the path parts of every file of the archive.
    """
    folders: dict[tuple[str, ...], str] = {}
    for parts in files:
        if parts[-1:] != (LABEL_FILE_NAME,):
            continue
        if len(parts) == 1:
            raise ValueError(f"{archive}: {LABEL_FILE_NAME} lies at the top of the archive, not in a clip folder")
        folders[parts[:-1]] = parts[-2]
    twice = sorted(name for name, count in Counter(folders.values()).items() if count > 1)
    if twice:
        raise ValueError(f"{archive}: more than one clip folder has the name {twice}")
    return folders


def _write_member(bundle: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path) -> None:
    """Write one file of the archive to ``target``. Missing folders are made."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with bundle.open(info) as source, target.open("wb") as out:
        shutil.copyfileobj(source, out)


def list_clips(split_dir: Path) -> list[str]:
    """Return the sorted names of the clip folders that lie directly in ``split_dir``."""
    return sorted(entry.name for entry in Path(split_dir).iterdir() if (entry / LABEL_FILE_NAME).is_file())


# --- Split list ----------------------------------------------------------------------


def make_split_manifest(valid_clips: Sequence[str], test_clips: Sequence[str], seed: int) -> dict[str, Any]:
    """Make the split list: which clip is used for which measurement.

    Each list of clip names is sorted and then shuffled with its own ``random.Random(seed)``.
    So the result depends only on the names and the seed. It does not depend on the order of
    the input, and the choice of the test clips does not depend on the validation clips.

    The shuffled validation list is cut into the groups ``probe`` (24 clips), ``dev`` (6),
    ``fresh`` (3) and ``reserve`` (the rest). The first 10 clips of the shuffled test list are
    the group ``final``; the other test clips are not used. Every group keeps the shuffled
    order, so the first clips of a group are a random choice too.

    The result is ``{"seed": seed, "valid": {"probe": [...], "dev": [...], "fresh": [...],
    "reserve": [...]}, "test": {"final": [...]}}``. No clip is in two groups.

    ``ValueError`` is raised for fewer than 33 validation clips or 10 test clips, for a clip
    name that is listed twice or is in both lists, and for a seed that is not a whole number.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"the seed must be a whole number, got {seed!r}")
    valid = _shuffled(valid_clips, seed, "validation")
    test = _shuffled(test_clips, seed, "test")
    n_fixed = sum(VALID_GROUP_SIZES.values())
    if len(valid) < n_fixed:
        raise ValueError(f"the split list needs at least {n_fixed} validation clips, got {len(valid)}")
    if len(test) < TEST_FINAL_SIZE:
        raise ValueError(f"the split list needs at least {TEST_FINAL_SIZE} test clips, got {len(test)}")
    shared = sorted(set(valid) & set(test))
    if shared:
        raise ValueError(f"the clips {shared} are in the validation list and in the test list")
    groups: dict[str, list[str]] = {}
    start = 0
    for group, size in VALID_GROUP_SIZES.items():
        groups[group] = valid[start : start + size]
        start += size
    groups[RESERVE_GROUP] = valid[start:]
    return {"seed": seed, "valid": groups, "test": {FINAL_GROUP: test[:TEST_FINAL_SIZE]}}


def _shuffled(clips: Sequence[str], seed: int, split: str) -> list[str]:
    """Return the clip names in an order that depends only on the names and the seed."""
    names = sorted(clips)
    twice = sorted(name for name, count in Counter(names).items() if count > 1)
    if twice:
        raise ValueError(f"the {split} list names the clips {twice} more than once")
    random.Random(seed).shuffle(names)
    return names


def load_manifest(path: Path) -> dict[str, Any]:
    """Read a split list from a JSON file.

    ``ValueError`` is raised for a file that is not JSON and for a top level that is not a JSON object.
    """
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: the top level must be a JSON object")
    return manifest


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write a split list as a JSON file with sorted keys. The folder of the file is made when it is missing.

    A file that is already there is replaced. The caller must not make a new split list after the
    first paid run: the split list is fixed before it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
