"""Tests of datasets.py: download address, download with resume, clip extraction, clip list and split list.

No test touches the network. A stand-in for ``urllib.request.urlopen`` serves the bytes of a
made-up archive. ``shutil.disk_usage`` is replaced in every test, so that no result depends on
the disk of the machine.
"""

from __future__ import annotations

import http.client
import io
import json
import random
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import NamedTuple

import pytest

from pitchvlm.datasets import (
    DATASET_REPO,
    FREE_SPACE_RESERVE_BYTES,
    LABEL_FILE_NAME,
    SOCKET_TIMEOUT_S,
    download_split,
    extract_clips,
    list_clips,
    load_manifest,
    make_split_manifest,
    save_manifest,
    split_url,
)

#: Bytes of a made-up archive. They are random, so a piece in the wrong place is found.
CONTENT = random.Random(20261113).randbytes(10_240)
#: Bytes of an unfinished download of an earlier call. They differ from the start of ``CONTENT``,
#: so the tests can tell bytes that were kept from bytes that were downloaded again.
KEPT = b"k" * 1000

#: Clip names like those of the two splits: 40 validation clips and 12 test clips.
VALID_CLIPS = [f"SNGS-{number:03d}" for number in range(21, 61)]
TEST_CLIPS = [f"SNGS-{number:03d}" for number in range(116, 128)]


class DiskUsage(NamedTuple):
    """The fields of the answer of ``shutil.disk_usage``."""

    total: int
    used: int
    free: int


class FakeResponse:
    """Stand-in for the answer of ``urllib.request.urlopen``: a status, headers and a body to read."""

    def __init__(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.status = status
        self.headers = http.client.HTTPMessage()
        for name, value in headers.items():
            self.headers[name] = value
        #: The ``size`` argument of every ``read`` call.
        self.read_sizes: list[int] = []
        self._body = io.BytesIO(body)

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        return self._body.read(size)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *error: object) -> None:
        return None


class StalledResponse(FakeResponse):
    """Stand-in for an answer whose connection stalls after the last byte of its body.

    The next read raises ``TimeoutError``. A real answer does the same when its socket has a
    timeout and the server sends nothing more.
    """

    def read(self, size: int) -> bytes:
        chunk = super().read(size)
        if not chunk:
            raise TimeoutError("The read operation timed out")
        return chunk


class FakeServer:
    """Stand-in for ``urllib.request.urlopen`` that serves one file and honours the ``Range`` header.

    With ``honours_range=False`` it sends the whole file with status 200 for every request. With
    ``cut_after`` it ends the body after that many bytes, like a connection that drops. A range
    that starts at or after the end of the file gets the status 416 with the size of the file.
    """

    def __init__(self, content: bytes, *, honours_range: bool = True, cut_after: int | None = None) -> None:
        self.content = content
        self.honours_range = honours_range
        self.cut_after = cut_after
        self.requests: list[urllib.request.Request] = []
        self.responses: list[FakeResponse] = []

    def __call__(self, request: urllib.request.Request) -> FakeResponse:
        self.requests.append(request)
        asked_range = request.get_header("Range")
        total = len(self.content)
        headers: dict[str, str] = {}
        if asked_range is None or not self.honours_range:
            status, first = 200, 0
        else:
            first = int(asked_range.removeprefix("bytes=").removesuffix("-"))
            status = 206
            headers["Content-Range"] = f"bytes {first}-{total - 1}/{total}"
        if first >= total:
            status = 416
            headers["Content-Range"] = f"bytes */{total}"
        body = self.content[first:]
        headers["Content-Length"] = str(len(body))
        response = FakeResponse(status, headers, body if self.cut_after is None else body[: self.cut_after])
        self.responses.append(response)
        return response


def set_free_space(monkeypatch: pytest.MonkeyPatch, free_bytes: int) -> list[Path]:
    """Make ``shutil.disk_usage`` report ``free_bytes``. Return the list that collects the paths it is asked about."""
    asked: list[Path] = []

    def fake_disk_usage(path: Path) -> DiskUsage:
        asked.append(Path(path))
        return DiskUsage(total=2 * free_bytes, used=free_bytes, free=free_bytes)

    monkeypatch.setattr(shutil, "disk_usage", fake_disk_usage)
    return asked


@pytest.fixture(autouse=True)
def roomy_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test a disk with a lot of free space. A test that needs another value sets it again."""
    set_free_space(monkeypatch, 10**15)


def archive_path(root: Path, split: str) -> Path:
    """Return the place of the complete archive of a split."""
    return root / "archives" / f"{split}.zip"


def partial_path(root: Path, split: str) -> Path:
    """Return the place of the unfinished download of a split."""
    return root / "archives" / f"{split}.zip.part"


def write_partial(root: Path, split: str, data: bytes) -> Path:
    """Write the unfinished download of an earlier call and return its path."""
    path = partial_path(root, split)
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    return path


def write_zip(path: Path, members: dict[str, bytes]) -> Path:
    """Write a zip archive with the given member names and contents. A name that ends with ``/`` is a folder."""
    with zipfile.ZipFile(path, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    return path


def clip_members(clip: str, prefix: str = "") -> dict[str, bytes]:
    """Return the archive members of one clip folder: the label file and two frame files (made-up bytes)."""
    return {
        f"{prefix}{clip}/{LABEL_FILE_NAME}": f"labels of {clip}".encode(),
        f"{prefix}{clip}/img1/000001.jpg": f"{clip} frame 1".encode(),
        f"{prefix}{clip}/img1/000002.jpg": f"{clip} frame 2".encode(),
    }


def make_clip_folder(split_dir: Path, clip: str) -> None:
    """Make a clip folder with a label file on disk."""
    folder = split_dir / clip
    folder.mkdir(parents=True)
    (folder / LABEL_FILE_NAME).write_text("{}", encoding="utf-8")


def tree(folder: Path) -> list[str]:
    """Return the relative paths of all files below a folder, sorted."""
    return sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file())


# --- split_url -----------------------------------------------------------------------


def test_dataset_repo_is_the_game_state_dataset() -> None:
    assert DATASET_REPO == "SoccerNet/SN-GSR-2025"


@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_split_url_gives_the_public_address(split: str) -> None:
    assert split_url(split) == f"https://huggingface.co/datasets/SoccerNet/SN-GSR-2025/resolve/main/{split}.zip"


@pytest.mark.parametrize(
    "split",
    [
        "challenge",
        "Valid",
        "valid.zip",
        pytest.param("", id="empty name"),
        pytest.param("../valid", id="name with a folder"),
    ],
)
def test_split_url_refuses_other_names(split: str) -> None:
    with pytest.raises(ValueError, match="unknown split"):
        split_url(split)


# --- download_split ------------------------------------------------------------------


def test_download_writes_the_archive(tmp_path: Path) -> None:
    server = FakeServer(CONTENT)

    path = download_split(tmp_path, "valid", opener=server)

    assert path == archive_path(tmp_path, "valid")
    assert path.read_bytes() == CONTENT
    # One request, to the address of the split, for the whole file.
    assert [request.full_url for request in server.requests] == [split_url("valid")]
    assert server.requests[0].get_header("Range") is None
    assert not partial_path(tmp_path, "valid").exists()
    # The default size of one read is 1 MiB: one read with all bytes and one empty read at the end.
    assert server.responses[0].read_sizes == [1 << 20, 1 << 20]


def test_download_resumes_a_partial_file(tmp_path: Path) -> None:
    partial = write_partial(tmp_path, "valid", KEPT)
    server = FakeServer(CONTENT)

    path = download_split(tmp_path, "valid", opener=server)

    # The request asks for the bytes after the 1000 bytes that are already there.
    assert len(server.requests) == 1
    assert server.requests[0].get_header("Range") == "bytes=1000-"
    # The answer is appended: the old bytes stay and no byte is there twice.
    assert path.read_bytes() == KEPT + CONTENT[1000:]
    assert not partial.exists()


def test_download_starts_again_when_the_server_ignores_the_range(tmp_path: Path) -> None:
    partial = write_partial(tmp_path, "valid", KEPT)
    server = FakeServer(CONTENT, honours_range=False)

    path = download_split(tmp_path, "valid", opener=server)

    assert server.requests[0].get_header("Range") == "bytes=1000-"
    # The answer is the whole file (status 200), so it must not be appended to the old bytes.
    assert path.read_bytes() == CONTENT
    assert not partial.exists()


def test_download_returns_an_existing_archive_without_a_request(tmp_path: Path) -> None:
    archive = archive_path(tmp_path, "test")
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"already here")

    def refuse(request: urllib.request.Request) -> FakeResponse:
        raise AssertionError(f"no request must be sent, got one for {request.full_url}")

    assert download_split(tmp_path, "test", opener=refuse) == archive
    assert archive.read_bytes() == b"already here"


def test_download_keeps_the_archives_of_two_splits_apart(tmp_path: Path) -> None:
    valid = download_split(tmp_path, "valid", opener=FakeServer(CONTENT))
    test = download_split(tmp_path, "test", opener=FakeServer(CONTENT[::-1]))

    assert valid == archive_path(tmp_path, "valid")
    assert test == archive_path(tmp_path, "test")
    assert valid.read_bytes() == CONTENT
    assert test.read_bytes() == CONTENT[::-1]


def test_download_reads_in_chunks(tmp_path: Path) -> None:
    server = FakeServer(CONTENT)

    path = download_split(tmp_path, "valid", opener=server, chunk_bytes=4096)

    # 10 240 bytes in reads of 4096 bytes: three reads with data and one empty read at the end.
    assert server.responses[0].read_sizes == [4096, 4096, 4096, 4096]
    assert path.read_bytes() == CONTENT


@pytest.mark.parametrize("chunk_bytes", [0, -1])
def test_download_refuses_a_chunk_size_that_is_not_positive(tmp_path: Path, chunk_bytes: int) -> None:
    server = FakeServer(CONTENT)

    with pytest.raises(ValueError, match="chunk_bytes"):
        download_split(tmp_path, "valid", opener=server, chunk_bytes=chunk_bytes)

    assert server.requests == []


def test_download_refuses_an_unknown_split(tmp_path: Path) -> None:
    server = FakeServer(CONTENT)

    with pytest.raises(ValueError, match="unknown split"):
        download_split(tmp_path, "challenge", opener=server)

    assert server.requests == []
    assert list(tmp_path.iterdir()) == []


def test_download_refuses_when_the_disk_space_is_too_small(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asked = set_free_space(monkeypatch, len(CONTENT) + FREE_SPACE_RESERVE_BYTES - 1)

    with pytest.raises(OSError, match="free disk space"):
        download_split(tmp_path, "valid", opener=FakeServer(CONTENT))

    # The disk of the archive folder is checked, and the check comes before the first byte.
    assert asked == [tmp_path / "archives"]
    assert not archive_path(tmp_path, "valid").exists()
    assert not partial_path(tmp_path, "valid").exists()


def test_download_needs_the_content_length_plus_20_gb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert FREE_SPACE_RESERVE_BYTES == 20 * 10**9
    set_free_space(monkeypatch, len(CONTENT) + FREE_SPACE_RESERVE_BYTES)

    path = download_split(tmp_path, "valid", opener=FakeServer(CONTENT))

    assert path.read_bytes() == CONTENT


def test_resumed_download_needs_space_for_the_missing_bytes_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    partial = write_partial(tmp_path, "valid", KEPT)
    missing_bytes = len(CONTENT) - len(KEPT)

    set_free_space(monkeypatch, missing_bytes + FREE_SPACE_RESERVE_BYTES - 1)
    with pytest.raises(OSError, match="free disk space"):
        download_split(tmp_path, "valid", opener=FakeServer(CONTENT))
    assert partial.read_bytes() == KEPT

    set_free_space(monkeypatch, missing_bytes + FREE_SPACE_RESERVE_BYTES)
    path = download_split(tmp_path, "valid", opener=FakeServer(CONTENT))
    assert path.read_bytes() == KEPT + CONTENT[1000:]


def test_download_keeps_the_partial_file_when_the_connection_ends_early(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="ended early"):
        download_split(tmp_path, "valid", opener=FakeServer(CONTENT, cut_after=3000))

    # The bytes that arrived stay in the unfinished file. No file has the final name.
    assert partial_path(tmp_path, "valid").read_bytes() == CONTENT[:3000]
    assert not archive_path(tmp_path, "valid").exists()

    # The next call continues after these bytes.
    server = FakeServer(CONTENT)
    path = download_split(tmp_path, "valid", opener=server)
    assert server.requests[0].get_header("Range") == "bytes=3000-"
    assert path.read_bytes() == CONTENT
    assert not partial_path(tmp_path, "valid").exists()


def test_download_keeps_the_partial_file_when_the_connection_stalls(tmp_path: Path) -> None:
    # The server announces the whole file, sends two chunks of 4096 bytes and then nothing more.
    headers = {"Content-Length": str(len(CONTENT))}
    stalled = StalledResponse(200, headers, CONTENT[:8192])

    with pytest.raises(TimeoutError):
        download_split(tmp_path, "valid", opener=lambda request: stalled, chunk_bytes=4096)

    # The third read gets no bytes and raises. The download does not read again.
    assert stalled.read_sizes == [4096, 4096, 4096]
    # The bytes that arrived stay in the unfinished file. No file has the final name.
    assert partial_path(tmp_path, "valid").read_bytes() == CONTENT[:8192]
    assert not archive_path(tmp_path, "valid").exists()

    # The next call continues after these bytes.
    server = FakeServer(CONTENT)
    path = download_split(tmp_path, "valid", opener=server)
    assert server.requests[0].get_header("Range") == "bytes=8192-"
    assert path.read_bytes() == CONTENT
    assert not partial_path(tmp_path, "valid").exists()


def test_default_opener_gives_urlopen_a_socket_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer(CONTENT)
    timeouts: list[float] = []

    def fake_urlopen(request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        timeouts.append(timeout)
        return server(request)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    path = download_split(tmp_path, "valid")

    # Without a timeout, a read from a connection that stalls would block for ever.
    assert SOCKET_TIMEOUT_S == 60.0
    assert timeouts == [SOCKET_TIMEOUT_S]
    assert [request.full_url for request in server.requests] == [split_url("valid")]
    assert path.read_bytes() == CONTENT


def test_download_refuses_an_answer_without_content_length(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="Content-Length"):
        download_split(tmp_path, "valid", opener=lambda request: FakeResponse(200, {}, CONTENT))

    assert not partial_path(tmp_path, "valid").exists()


def test_download_refuses_a_range_answer_that_starts_at_the_wrong_byte(tmp_path: Path) -> None:
    partial = write_partial(tmp_path, "valid", KEPT)
    headers = {"Content-Range": "bytes 500-10239/10240", "Content-Length": "9740"}

    with pytest.raises(OSError, match="Content-Range"):
        download_split(tmp_path, "valid", opener=lambda request: FakeResponse(206, headers, CONTENT[500:]))

    assert partial.read_bytes() == KEPT


def test_download_refuses_a_range_answer_without_content_range(tmp_path: Path) -> None:
    partial = write_partial(tmp_path, "valid", KEPT)
    headers = {"Content-Length": "9240"}

    with pytest.raises(OSError, match="Content-Range"):
        download_split(tmp_path, "valid", opener=lambda request: FakeResponse(206, headers, CONTENT[1000:]))

    assert partial.read_bytes() == KEPT


def test_download_takes_a_partial_file_that_holds_the_whole_archive(tmp_path: Path) -> None:
    """An earlier call was stopped after the last byte and before the file got its final name."""
    partial = write_partial(tmp_path, "valid", CONTENT)
    server = FakeServer(CONTENT)

    path = download_split(tmp_path, "valid", opener=server)

    assert path == archive_path(tmp_path, "valid")
    assert path.read_bytes() == CONTENT
    assert not partial.exists()
    # One request, for the bytes after the end of the file. Its answer has the status 416 and no usable body.
    assert [request.get_header("Range") for request in server.requests] == ["bytes=10240-"]
    assert [response.status for response in server.responses] == [416]
    assert server.responses[0].read_sizes == []


def test_download_takes_a_complete_partial_file_when_the_opener_raises_the_status_416(tmp_path: Path) -> None:
    """``urllib.request.urlopen`` raises ``HTTPError`` for the status 416. The error holds the headers."""
    partial = write_partial(tmp_path, "valid", CONTENT)

    def opener(request: urllib.request.Request) -> FakeResponse:
        headers = http.client.HTTPMessage()
        headers["Content-Range"] = "bytes */10240"
        raise urllib.error.HTTPError(request.full_url, 416, "Requested Range Not Satisfiable", headers, io.BytesIO())

    path = download_split(tmp_path, "valid", opener=opener)

    assert path.read_bytes() == CONTENT
    assert not partial.exists()


@pytest.mark.parametrize(
    ("headers", "problem"),
    [
        pytest.param({"Content-Range": "bytes */10240"}, "the archive has 10240 bytes", id="another size"),
        pytest.param({"Content-Range": "bytes */*"}, "names no size", id="no size"),
        pytest.param({}, "names no size", id="no header"),
    ],
)
def test_download_refuses_a_partial_file_that_the_status_416_does_not_confirm(
    tmp_path: Path, headers: dict[str, str], problem: str
) -> None:
    """The ``.part`` file holds 11,000 bytes: more than the archive of 10,240 bytes has."""
    too_long = CONTENT + b"x" * 760
    partial = write_partial(tmp_path, "valid", too_long)

    with pytest.raises(OSError, match=problem) as raised:
        download_split(tmp_path, "valid", opener=lambda request: FakeResponse(416, headers, b""))

    # The error names the file, so that it can be compared or deleted by hand. The file stays.
    assert str(partial) in str(raised.value)
    assert partial.read_bytes() == too_long
    assert not archive_path(tmp_path, "valid").exists()


def test_download_passes_on_another_http_error_of_the_opener(tmp_path: Path) -> None:
    partial = write_partial(tmp_path, "valid", KEPT)

    def opener(request: urllib.request.Request) -> FakeResponse:
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", http.client.HTTPMessage(), io.BytesIO())

    with pytest.raises(urllib.error.HTTPError, match="404"):
        download_split(tmp_path, "valid", opener=opener)

    assert partial.read_bytes() == KEPT
    assert not archive_path(tmp_path, "valid").exists()


def test_download_refuses_the_status_416_without_a_partial_file(tmp_path: Path) -> None:
    """No range was asked for, so the status 416 says nothing about a file on the disk."""
    headers = {"Content-Range": "bytes */0"}

    with pytest.raises(OSError, match="status 416"):
        download_split(tmp_path, "valid", opener=lambda request: FakeResponse(416, headers, b""))

    assert not archive_path(tmp_path, "valid").exists()


def test_download_refuses_an_unexpected_status(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="status 204"):
        download_split(tmp_path, "valid", opener=lambda request: FakeResponse(204, {"Content-Length": "0"}, b""))

    assert not archive_path(tmp_path, "valid").exists()


# --- extract_clips -------------------------------------------------------------------


def test_extract_writes_all_clip_folders(tmp_path: Path) -> None:
    # The members are not in name order, and the archive has a folder entry.
    members = {**clip_members("SNGS-022"), "SNGS-021/": b"", **clip_members("SNGS-021")}
    archive = write_zip(tmp_path / "valid.zip", members)
    dest = tmp_path / "out" / "valid"

    names = extract_clips(archive, dest)

    assert names == ["SNGS-021", "SNGS-022"]
    assert tree(dest) == [
        f"SNGS-021/{LABEL_FILE_NAME}",
        "SNGS-021/img1/000001.jpg",
        "SNGS-021/img1/000002.jpg",
        f"SNGS-022/{LABEL_FILE_NAME}",
        "SNGS-022/img1/000001.jpg",
        "SNGS-022/img1/000002.jpg",
    ]
    assert (dest / "SNGS-021" / LABEL_FILE_NAME).read_bytes() == b"labels of SNGS-021"
    assert (dest / "SNGS-022" / "img1" / "000002.jpg").read_bytes() == b"SNGS-022 frame 2"
    assert list_clips(dest) == names


@pytest.mark.parametrize("wanted", [["SNGS-023", "SNGS-021"], ("SNGS-021", "SNGS-023"), {"SNGS-021", "SNGS-023"}])
def test_extract_writes_only_the_wanted_clips(tmp_path: Path, wanted: object) -> None:
    members = {**clip_members("SNGS-021"), **clip_members("SNGS-022"), **clip_members("SNGS-023")}
    archive = write_zip(tmp_path / "valid.zip", members)
    dest = tmp_path / "out"

    names = extract_clips(archive, dest, clips=wanted)  # type: ignore[arg-type]

    assert names == ["SNGS-021", "SNGS-023"]
    assert sorted(path.name for path in dest.iterdir()) == ["SNGS-021", "SNGS-023"]
    assert (dest / "SNGS-023" / "img1" / "000001.jpg").read_bytes() == b"SNGS-023 frame 1"


def test_extract_with_no_wanted_clip_writes_nothing(tmp_path: Path) -> None:
    archive = write_zip(tmp_path / "valid.zip", clip_members("SNGS-021"))
    dest = tmp_path / "out"

    assert extract_clips(archive, dest, clips=[]) == []
    assert not dest.exists()


def test_extract_finds_clip_folders_below_a_top_folder(tmp_path: Path) -> None:
    members = {
        "valid/": b"",
        **clip_members("SNGS-021", prefix="valid/"),
        **clip_members("SNGS-022", prefix="valid/"),
    }
    archive = write_zip(tmp_path / "valid.zip", members)
    dest = tmp_path / "out"

    assert extract_clips(archive, dest, clips=["SNGS-022"]) == ["SNGS-022"]
    assert extract_clips(archive, dest) == ["SNGS-021", "SNGS-022"]

    # The clip folders lie directly in the target folder: the top folder of the archive is gone.
    assert sorted(path.name for path in dest.iterdir()) == ["SNGS-021", "SNGS-022"]
    assert (dest / "SNGS-021" / "img1" / "000002.jpg").read_bytes() == b"SNGS-021 frame 2"
    assert list_clips(dest) == ["SNGS-021", "SNGS-022"]


def test_extract_reads_member_names_with_backslashes(tmp_path: Path) -> None:
    # Some tools write the path separator as a backslash. It must give folders, not odd file names.
    members = {
        "SNGS-021\\": b"",
        f"SNGS-021\\{LABEL_FILE_NAME}": b"labels of SNGS-021",
        "SNGS-021\\img1\\": b"",
        "SNGS-021\\img1\\000001.jpg": b"SNGS-021 frame 1",
    }
    archive = write_zip(tmp_path / "valid.zip", members)
    dest = tmp_path / "out"

    assert extract_clips(archive, dest) == ["SNGS-021"]
    assert tree(dest) == [f"SNGS-021/{LABEL_FILE_NAME}", "SNGS-021/img1/000001.jpg"]
    assert (dest / "SNGS-021" / "img1" / "000001.jpg").read_bytes() == b"SNGS-021 frame 1"


def test_extract_leaves_out_files_outside_the_clip_folders(tmp_path: Path) -> None:
    members = {
        "README.md": b"about the data",
        "notes/readme.txt": b"a folder without a label file",
        f"__MACOSX/SNGS-021/._{LABEL_FILE_NAME}": b"not a label file",
        **clip_members("SNGS-021"),
    }
    archive = write_zip(tmp_path / "valid.zip", members)
    dest = tmp_path / "out"

    assert extract_clips(archive, dest) == ["SNGS-021"]
    assert sorted(path.name for path in dest.iterdir()) == ["SNGS-021"]
    assert len(tree(dest)) == 3


def test_extract_writes_an_existing_file_again(tmp_path: Path) -> None:
    archive = write_zip(tmp_path / "valid.zip", clip_members("SNGS-021"))
    dest = tmp_path / "out"
    extract_clips(archive, dest)
    frame_file = dest / "SNGS-021" / "img1" / "000001.jpg"
    frame_file.write_bytes(b"damaged")

    assert extract_clips(archive, dest) == ["SNGS-021"]
    assert frame_file.read_bytes() == b"SNGS-021 frame 1"


UNSAFE_MEMBERS = {
    "parent folder at the start": ("../evil.txt", "'..' part"),
    "parent folder in the middle": ("SNGS-021/../../evil.txt", "'..' part"),
    "parent folder with a backslash": ("..\\evil.txt", "'..' part"),
    "absolute path": ("/etc/evil.txt", "absolute path"),
    "absolute path with a backslash": ("\\evil.txt", "absolute path"),
    "drive letter": ("C:/evil.txt", "drive letter"),
}


@pytest.mark.parametrize(("member", "problem"), list(UNSAFE_MEMBERS.values()), ids=list(UNSAFE_MEMBERS))
def test_extract_refuses_an_unsafe_member(tmp_path: Path, member: str, problem: str) -> None:
    archive = write_zip(tmp_path / "valid.zip", {**clip_members("SNGS-021"), member: b"evil"})
    dest = tmp_path / "out" / "valid"

    with pytest.raises(ValueError, match=re.escape(problem)):
        extract_clips(archive, dest)

    # The whole archive is refused before the first file is written, also its safe clip.
    assert sorted(path.name for path in tmp_path.iterdir()) == ["valid.zip"]


def test_extract_refuses_a_wanted_clip_that_is_not_in_the_archive(tmp_path: Path) -> None:
    archive = write_zip(tmp_path / "valid.zip", {**clip_members("SNGS-021"), **clip_members("SNGS-022")})
    dest = tmp_path / "out"

    with pytest.raises(ValueError, match="SNGS-999"):
        extract_clips(archive, dest, clips=["SNGS-021", "SNGS-999"])

    assert not dest.exists()


def test_extract_refuses_two_clip_folders_with_one_name(tmp_path: Path) -> None:
    members = {**clip_members("SNGS-021", prefix="a/"), **clip_members("SNGS-021", prefix="b/")}
    archive = write_zip(tmp_path / "valid.zip", members)

    with pytest.raises(ValueError, match="more than one clip folder"):
        extract_clips(archive, tmp_path / "out")

    assert not (tmp_path / "out").exists()


def test_extract_refuses_a_label_file_at_the_top_of_the_archive(tmp_path: Path) -> None:
    archive = write_zip(tmp_path / "clip.zip", {LABEL_FILE_NAME: b"{}", "img1/000001.jpg": b"frame"})

    with pytest.raises(ValueError, match="top of the archive"):
        extract_clips(archive, tmp_path / "out")

    assert not (tmp_path / "out").exists()


# --- list_clips ----------------------------------------------------------------------


def test_list_clips_gives_the_sorted_names_of_folders_with_a_label_file(tmp_path: Path) -> None:
    for clip in ["SNGS-030", "SNGS-021", "SNGS-025"]:
        make_clip_folder(tmp_path, clip)
    (tmp_path / "SNGS-040" / "img1").mkdir(parents=True)  # a folder without a label file
    (tmp_path / "notes.txt").write_text("not a folder", encoding="utf-8")
    (tmp_path / LABEL_FILE_NAME).write_text("{}", encoding="utf-8")  # a label file outside a clip folder
    make_clip_folder(tmp_path / "deeper", "SNGS-050")  # a clip folder that is not directly in the split folder

    assert list_clips(tmp_path) == ["SNGS-021", "SNGS-025", "SNGS-030"]


def test_list_clips_of_an_empty_folder_is_empty(tmp_path: Path) -> None:
    assert list_clips(tmp_path) == []


def test_list_clips_of_a_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list_clips(tmp_path / "missing")


# --- make_split_manifest -------------------------------------------------------------


def test_manifest_has_the_group_sizes_of_the_plan() -> None:
    manifest = make_split_manifest(VALID_CLIPS, TEST_CLIPS, seed=7)

    assert set(manifest) == {"seed", "valid", "test"}
    assert manifest["seed"] == 7
    assert {group: len(clips) for group, clips in manifest["valid"].items()} == {
        "probe": 24,
        "dev": 6,
        "fresh": 3,
        "reserve": 7,
    }
    assert {group: len(clips) for group, clips in manifest["test"].items()} == {"final": 10}


def test_manifest_groups_are_disjoint_and_use_every_validation_clip() -> None:
    manifest = make_split_manifest(VALID_CLIPS, TEST_CLIPS, seed=7)

    valid = [clip for clips in manifest["valid"].values() for clip in clips]
    final = manifest["test"]["final"]
    assert sorted(valid) == sorted(VALID_CLIPS)  # every clip once, so no clip is in two groups
    assert len(set(final)) == 10
    assert set(final) <= set(TEST_CLIPS)
    assert set(final).isdisjoint(valid)


def test_manifest_with_the_smallest_lists_has_an_empty_reserve() -> None:
    manifest = make_split_manifest(VALID_CLIPS[:33], TEST_CLIPS[:10], seed=0)

    assert manifest["valid"]["reserve"] == []
    assert sorted(manifest["test"]["final"]) == TEST_CLIPS[:10]


def test_manifest_follows_the_shuffle_of_the_seed() -> None:
    manifest = make_split_manifest(VALID_CLIPS, TEST_CLIPS, seed=7)

    # The recipe: sort the names, then shuffle each list with its own random.Random(seed).
    valid = sorted(VALID_CLIPS)
    random.Random(7).shuffle(valid)
    test = sorted(TEST_CLIPS)
    random.Random(7).shuffle(test)
    assert manifest["valid"] == {
        "probe": valid[:24],
        "dev": valid[24:30],
        "fresh": valid[30:33],
        "reserve": valid[33:],
    }
    assert manifest["test"] == {"final": test[:10]}
    assert manifest["valid"]["probe"] != sorted(manifest["valid"]["probe"])  # the groups are not in name order


def test_manifest_is_the_same_for_the_same_seed_and_any_input_order() -> None:
    first = make_split_manifest(VALID_CLIPS, TEST_CLIPS, seed=7)
    again = make_split_manifest(tuple(VALID_CLIPS), tuple(TEST_CLIPS), seed=7)
    reordered = make_split_manifest(VALID_CLIPS[::-1], TEST_CLIPS[::-1], seed=7)

    assert again == first
    assert reordered == first


def test_manifest_changes_with_the_seed() -> None:
    first = make_split_manifest(VALID_CLIPS, TEST_CLIPS, seed=7)
    other = make_split_manifest(VALID_CLIPS, TEST_CLIPS, seed=8)

    assert other["valid"] != first["valid"]
    assert other["test"] != first["test"]


def test_test_clips_do_not_depend_on_the_validation_clips() -> None:
    first = make_split_manifest(VALID_CLIPS, TEST_CLIPS, seed=7)
    fewer = make_split_manifest(VALID_CLIPS[:35], TEST_CLIPS, seed=7)

    assert fewer["test"] == first["test"]


def test_manifest_does_not_change_its_input() -> None:
    valid = list(VALID_CLIPS[::-1])
    test = list(TEST_CLIPS[::-1])

    make_split_manifest(valid, test, seed=7)

    assert valid == VALID_CLIPS[::-1]
    assert test == TEST_CLIPS[::-1]


def test_manifest_needs_33_validation_clips() -> None:
    with pytest.raises(ValueError, match="33 validation clips"):
        make_split_manifest(VALID_CLIPS[:32], TEST_CLIPS, seed=7)


def test_manifest_needs_10_test_clips() -> None:
    with pytest.raises(ValueError, match="10 test clips"):
        make_split_manifest(VALID_CLIPS, TEST_CLIPS[:9], seed=7)


def test_manifest_refuses_a_clip_that_is_listed_twice() -> None:
    with pytest.raises(ValueError, match="more than once"):
        make_split_manifest([*VALID_CLIPS, "SNGS-021"], TEST_CLIPS, seed=7)
    with pytest.raises(ValueError, match="more than once"):
        make_split_manifest(VALID_CLIPS, [*TEST_CLIPS, "SNGS-116"], seed=7)


def test_manifest_refuses_a_clip_that_is_in_both_lists() -> None:
    with pytest.raises(ValueError, match="SNGS-021"):
        make_split_manifest(VALID_CLIPS, [*TEST_CLIPS, "SNGS-021"], seed=7)


@pytest.mark.parametrize("seed", [None, 7.0, "7", True])
def test_manifest_refuses_a_seed_that_is_not_a_whole_number(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        make_split_manifest(VALID_CLIPS, TEST_CLIPS, seed)  # type: ignore[arg-type]


# --- load_manifest, save_manifest ----------------------------------------------------


def test_manifest_file_round_trip(tmp_path: Path) -> None:
    manifest = make_split_manifest(VALID_CLIPS, TEST_CLIPS, seed=7)
    path = tmp_path / "meta" / "splits.json"  # the folder does not exist yet

    save_manifest(path, manifest)

    assert load_manifest(path) == manifest


def test_manifest_file_is_json_with_sorted_keys(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    save_manifest(path, make_split_manifest(VALID_CLIPS, TEST_CLIPS, seed=7))

    text = path.read_text(encoding="utf-8")
    data = json.loads(text)  # the keys of a parsed mapping keep the order of the file
    assert list(data) == ["seed", "test", "valid"]
    assert list(data["valid"]) == ["dev", "fresh", "probe", "reserve"]
    assert text.endswith("}\n")


def test_save_manifest_replaces_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    save_manifest(path, {"seed": 1})
    save_manifest(path, {"seed": 2})

    assert load_manifest(path) == {"seed": 2}


def test_load_manifest_refuses_a_file_that_is_not_a_json_object(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    path.write_text('["SNGS-021", "SNGS-022"]', encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_manifest(path)


def test_load_manifest_refuses_a_file_that_is_not_json(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    path.write_text("seed: 7", encoding="utf-8")

    with pytest.raises(ValueError):
        load_manifest(path)


def test_load_manifest_of_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "missing.json")
