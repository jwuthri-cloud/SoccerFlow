"""Repository guard: stop leaks before a commit or a push.

The guard refuses five kinds of leaks:

1. media, footage and dataset files, also as text (encoded pixels, label data, prediction data),
2. secrets such as API keys,
3. machine-specific files and text (lock files, private hosts, home-folder paths, files that
   the ignore rules name),
4. the words of the local deny-list (for example the name of an organisation),
5. commits and tags made with the wrong git identity, and a tag that points at no commit.

The guard has four modes:

- `--staged` checks the files and the identity of the next commit. It is the pre-commit hook.
- `--push` checks what a push can send. It reads the commits that the remote does not have yet
  and every local annotated tag: their files, their messages and their identities. It also
  reads the names of the local branches that the remote does not have, and the name of every
  local tag. Git does not record which tags the remote has, so every local tag counts. It is
  the pre-push hook.
- `--tracked` checks every tracked file of the working tree. Run it by hand after the rules or
  the deny-list change.
- `--paths` checks the given files of the working tree. Other tools use it.

The rules for secrets, hosts, home folders and the deny-list also read names: the path of every
checked file and, in the push mode, the names of branches and tags.

The local deny-list is the file `guard-deny.txt` in the git folder. It is never tracked. Each of
its lines is a word that must not appear in a file, a path, a message, an identity or the name
of a branch or of a tag.

The guard uses the Python standard library only and runs on Python 3.9 or later.

Exit code 0 means clean, 1 means violations, 2 means a usage error or a git error.
Each violation is one line: `<path>:<line or ->: <rule-id>: <message>`.
"""

# Python 3.9 needs this import for the `X | None` annotations.
from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import IO, NoReturn, Protocol

EXIT_CLEAN = 0
EXIT_VIOLATIONS = 1
EXIT_ERROR = 2

# The limit holds for every file. The cost ledger of the package grows with every paid answer,
# so the package splits it into files of at most half this size.
MAX_FILE_BYTES = 2_000_000
# A report shows the first 4 characters of a secret, and nothing of a secret below this length.
MIN_PREVIEW_LENGTH = 20
MIN_BASE64_RUN = 200
MIN_WRAPPED_RUN = 2_000
MIN_WRAPPED_PIECE = 40
MIN_BYTE_TABLE = 20_000
ALLOWED_SUFFIXES = frozenset(
    {
        ".py",
        ".md",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".jsonl",
        ".csv",
        ".txt",
        ".cfg",
        ".ini",
        ".typed",
    }
)
ALLOWED_NAMES = frozenset({".gitignore", ".gitattributes", "LICENSE", "NOTICE"})
LOCK_SUFFIXES = frozenset({".lock", ".toml", ".json", ".yaml", ".yml", ".txt"})
LOCK_NAME = re.compile(r"(?i)(?:^|[._\-])lock(?:[._\-]|$)|^pylock\.")
LABEL_FILE_NAME = "labels-gamestate.json"
RUN_RESULT_NAME = "result.json"
ALWAYS_ALLOWED_HOSTS = frozenset({"localhost", "127.0.0.1"})
ALLOW_LIST = Path(__file__).resolve().parent / "allowed_hosts.txt"
DENY_LIST_NAME = "guard-deny.txt"
DENY_LIST_NOTICE = (
    f"guard: notice: the local deny-list ({DENY_LIST_NAME} in the git folder) does not exist, "
    "so the [deny] rule is off; docs/RUNBOOK.md says what to put in the list"
)
NO_PATH = "-"
MASK = "***"
# Environment variables that the hook runner sets before a push: the name of the remote, the
# newest commit of the first branch that is pushed, and the name that this branch gets on the
# remote.
REMOTE_VARIABLE = "PRE_COMMIT_REMOTE_NAME"
PUSHED_COMMIT_VARIABLE = "PRE_COMMIT_TO_REF"
PUSHED_NAME_VARIABLE = "PRE_COMMIT_REMOTE_BRANCH"
SHORT_NAME_LENGTH = 10
SUBMODULE_MODE = "160000"
BATCH_BYTES = 64_000_000


def whole_word(prefix: str, rest: str) -> re.Pattern[str]:
    """Return a pattern for a fixed prefix at the start of a word, followed by `rest`.

    The pattern starts with the fixed text, so a search finds the candidates fast. The test
    for the start of the word comes after the prefix. The letter of an escape such as the
    line break of a JSON text is not a part of the word: the prefix can follow it directly.
    """
    literal = re.escape(prefix)
    return re.compile(rf"{literal}(?:(?<!\w{literal})|(?<=\\[abfnrtv]{literal})){rest}")


# The patterns below are written so that this source file does not match them.
SECRET_NAME = (
    r"(?i:API[_\-]?KEYS?|(?:SECRET|PRIVATE|ACCESS)[_\-]?KEYS?"
    r"|[_\-](?:TOKEN|SECRET|PASSWORD|PASSWD)S?|\b(?:TOKEN|SECRET|PASSWORD|PASSWD))"
)
# A value in quotes. A placeholder such as <key> or $KEY is not a value.
QUOTED_VALUE = r"""[bBrRuUfF]{0,2}['"](?![<${%])([^'"\s]{20,})['"]"""
SECRET_ASSIGNMENT = re.compile(
    rf"""(?=[AaPpSsTt_\-])                    # a fast test: the first character of a secret name
        {SECRET_NAME}
        ['"]?\]?                              # the closing quote of a name, the bracket of an index
        (?:[ \t]*:[\w.\[\]| \t]{{1,40}})?     # an optional type annotation
        [ \t]*[=:][ \t]*                      # the assignment sign
        (?:[>|][+\-]?[ \t]*)?                 # the mark of a YAML text block
        (?:\(?[ \t]*\n[ \t]+)?                # the value on the next line, also inside brackets
        (?:-[ \t]+)?                          # the dash of a YAML list item
        (?:
            {QUOTED_VALUE}
          | (?=[A-Za-z_\-]*[0-9])             # a bare value has a digit: a name in code has none
            ([0-9A-Za-z_\-]{{20,}})
            (?![0-9A-Za-z_\-(\[=:]|\.\w)      # and it is not a call, a name or a part of a name
        )
    """,
    re.VERBOSE,
)
# A call with a secret name and a value, such as the default value of an environment lookup.
SECRET_ARGUMENT = re.compile(
    rf"""\([ \t]*['"][\w.\-]*{SECRET_NAME}['"][ \t]*,[ \t]*{QUOTED_VALUE}"""
)
WEB_TOKEN_BODY = r"[0-9A-Za-z_\-]{8,}\.eyJ[0-9A-Za-z_\-]{8,}\.[0-9A-Za-z_\-]{8,}"
# The prefix of the new Google key is two capital letters. Only a word in capitals can end with
# them (FAQ.), so the prefix starts a key after every character but a capital letter: also
# after the letter of an escape, after a small letter, a digit or an underscore.
NEW_GOOGLE_KEY = re.compile(r"AQ\.(?<![A-Z]AQ\.)[0-9A-Za-z_\-]{20,}")
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Google API key", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("Google API key in the new format", NEW_GOOGLE_KEY),
    # An OpenRouter key is the prefix sk-or-v1- and 64 hex digits. The prefix is long enough to need
    # no word start, so the key is also found after a letter, a digit or an underscore, as the
    # Google key is. This rule comes before the general sk- rule, so that the message names the service.
    ("OpenRouter API key", re.compile(r"sk-or-v1-[0-9a-f]{64}")),
    ("key with the sk- prefix", whole_word("sk-", r"[0-9A-Za-z_\-]{20,}")),
    ("Google OAuth token", whole_word("ya29.", r"[0-9A-Za-z_\-]+")),
    ("GitHub token", re.compile(r"gh[pousr]_[0-9A-Za-z]{36,}|github_pat_[0-9A-Za-z_]{22,}")),
    ("Hugging Face token", whole_word("hf_", r"[0-9A-Za-z]{30,}")),
    ("JSON web token", whole_word("eyJ", WEB_TOKEN_BODY)),
    ("PEM private key header", re.compile(r"-{5}BEGIN [A-Z0-9 ]*PRIVATE KEY")),
    ("password in an address", re.compile(r"://[^\s/?#@:]*:(?![<${%])([^\s/?#@]+)@")),
    ("bearer token", re.compile(r"(?i:\bBearer)[ \t]+([0-9A-Za-z_\-.+/=]{20,})")),
    ("literal value assigned to a secret name", SECRET_ASSIGNMENT),
    ("literal value next to a secret name in a call", SECRET_ARGUMENT),
)
# A piece of base64 text. Both base64 alphabets count: the standard one (+ and /) and the
# URL-safe one (- and _). Up to 2 padding characters count for the length of a piece. A piece
# does not start after a backslash: that letter is the end of an escape such as \n.
BASE64_PIECE = re.compile(r"(?<!\\)[0-9A-Za-z+/_\-]{" + str(MIN_WRAPPED_PIECE) + r",}={0,2}")
# What stands between two pieces of one wrapped run: white space, or the escape of a line break
# inside a JSON string.
LINE_BREAKS = re.compile(r"(?:\s|\\n|\\r)+")
DATA_ADDRESS = whole_word("data:", r"[a-z]+/[\w.+\-]+(?:;[\w=\-]+)*;base64,")
# A table of byte values: whole numbers from 0 to 255 with nothing but separators between them.
BYTE_VALUE = r"(?<![\w.\-])(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\w.])"
BYTE_TABLE = re.compile(r"(?=\d)(?:" + BYTE_VALUE + r"[\s,;\[\]]+){64,}")
# An address of any scheme. The pattern starts at the mark between the scheme and the host.
URL = re.compile(
    r"""://
        (?:[^\s/?\#"'`()\\]*@)?             # optional user information, up to the last @
        (\[[0-9A-Fa-f:.]*\]|[\w.\-{}$%]+)   # the host: an IPv6 literal or a name
    """,
    re.VERBOSE,
)
# The short form of a git remote: user, host, colon and path. The pattern starts at the @ sign.
REMOTE_PATH = re.compile(r"@(?<=[\w.\-]@)([\w\-]+(?:\.[\w\-]+)+):[\w.\-~/]+")
PACKAGE_INDEX_OPTION = re.compile(
    r"""(?:--(?:extra-)?index-url|--trusted-host)(?:[ \t]+|=)
        (?![a-z][a-z0-9+.\-]*://)           # an address with a scheme is read as an address
        ([\w\-]+(?:\.[\w\-]+)+)             # the host
    """,
    re.IGNORECASE | re.VERBOSE,
)
# An address of a private network, also when it stands alone: the ranges 10.x, 172.16 to 172.31,
# 192.168, 100.64 to 100.127 (shared address space) and 169.254 (link-local). A version number
# after a comparison sign such as == or >= is not an address; an address after one equals sign
# is one. Every range starts with the digit 1, which the pattern tests first.
PRIVATE_ADDRESS = re.compile(
    r"""(?=1)(?<![\w.<>~^\-])(?<![=<>~!]=)
        ((?:10\.\d{1,3}
          | 192\.168
          | 172\.(?:1[6-9]|2\d|3[01])
          | 100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])
          | 169\.254
         )\.\d{1,3}\.\d{1,3})
        (?![\w]|\.\d)
    """,
    re.VERBOSE,
)
HOST_PATTERNS = (URL, REMOTE_PATH, PACKAGE_INDEX_OPTION, PRIVATE_ADDRESS)
# The pattern tests the first character before anything else: a search is then fast in a large
# file, where few characters can start a path. The folder of all users is `Users` or `home`.
# In another letter case (`users`) it counts only when a path goes on after the user name,
# because a web route such as `users/me` has the same form. The short form with a user name
# has two or more name characters and no digit after its slash: `~n/2` is a number.
SLASH = r"(?:\\?/|%2[Ff])"
HOME_PATH = re.compile(
    rf"""(?=[/\\%~])
        (?:
            (?<![\w.\-~}})])                            # not the end of a relative path
            (?:/System/Volumes/Data|/Volumes/[^/\s]+|/mnt/[a-z]|/export|/private)?
            {SLASH}(?:Users|home|(?i:users)(?={SLASH}[\w.\-]+{SLASH}))   # macOS and Linux
            {SLASH}[\w.\-]+                             # the name of the user
          | (?<![\w\\])\\{{1,2}}Users\\{{1,2}}[\w.\-]+  # Windows, also with doubled backslashes
          | (?<![\w.\-~}})])/(?:var/)?root/             # the home folder of the administrator
          | (?<![\w.\-~/])~[A-Za-z_][\w.\-]+/(?!\d)     # the short form with a user name
        )
    """,
    re.VERBOSE,
)
USER_FOLDER = re.compile(
    r"""/var/folders/\w\w/\w+                      # the temporary folder of one macOS user
      | -Users-(?<![\w\-]-Users-)[\w.]+-           # a home-folder path with dashes for slashes
    """,
    re.VERBOSE,
)
USER_FOLDER_MESSAGE = "path of a folder that belongs to one user of one machine"
HOME_PATTERNS = (
    ("absolute home-folder path; write ~ instead", HOME_PATH),
    (USER_FOLDER_MESSAGE, USER_FOLDER),
)
# The folders of a home path inside a name (the path of a file, the name of a branch), where
# no slash starts the path.
HOME_FOLDERS = re.compile(r"(?:^|/)Users/[^/]+/")
HOME_FOLDERS_MESSAGE = "folders of a home-folder path (Users, then a user name)"
IDENT = re.compile(r"(.*) <([^<>]*)> \d+ [+-]\d{4}")
IDENT_ROLES = ("author", "committer", "tagger")
#: The settings of the local identity, with the word that a message uses for each.
IDENTITY_FIELDS = (("name", "name"), ("email", "e-mail"))
IDENTITY_NOT_SET = (
    "user.{field} is not set in this repository; set a personal name and e-mail with "
    '`git config --local user.name "<name>"` and `git config --local user.email "<e-mail>"`'
)
# Git cleans a name and an e-mail before it writes them into a commit: it removes the delimiters
# of an identity line, and some characters at the start and at the end. Which characters, depends
# on the version of git, so two values are compared without all of them.
IDENT_DELIMITERS = str.maketrans("", "", "<>\n")
IDENT_CRUD = "".join(map(chr, range(33))) + ".,:;\"\\'"
# The kinds of names that the guard reads, by the start of their full name.
REF_PREFIXES = {"refs/heads/": "branch", "refs/tags/": "tag", "refs/remotes/": "remote"}
# A tag must point at a commit. These words name the other things that a tag can point at.
TAG_TARGETS = {"blob": "the content of a file", "tree": "a folder", "tag": "another tag"}
# The file types that the [dataset] rule reads as JSON, and the keys of a record of the label
# format: a box in the image or on the pitch, with the track of its person.
DATASET_SUFFIXES = frozenset({".json", ".jsonl"})
RECORD_BOXES = ("bbox_image", "bbox_pitch")
RECORD_TRACK = "track_id"


class GuardError(Exception):
    """A usage error or a git failure. It stops the guard with exit code 2."""


@dataclass(frozen=True)
class Violation:
    """One broken rule at one place."""

    path: str
    line: int | None
    rule: str
    message: str

    def render(self) -> str:
        """Return the report line for this violation."""
        line = "-" if self.line is None else str(self.line)
        return f"{self.path}:{line}: {self.rule}: {self.message}"


@dataclass(frozen=True)
class DenyEntry:
    """One entry of the local deny-list: its line number and a pattern that finds its text."""

    line: int
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class Settings:
    """The lists that the rules read: the allowed hosts and the local deny-list."""

    exact_hosts: frozenset[str]
    host_suffixes: tuple[str, ...]
    deny: tuple[DenyEntry, ...]
    #: False when the file of the deny-list does not exist. The [deny] rule is then off.
    has_deny_list: bool


@dataclass(frozen=True)
class Entry:
    """One file to check.

    `key` names the content of the file in a file store: the path for the working tree, the
    object name for the git database. It is None for a path without file content (a submodule).
    `origin` says where the file comes from, for example a commit. It is empty for the next
    commit and for the working tree.
    """

    path: str
    key: str | None
    origin: str = ""


@dataclass(frozen=True)
class Identity:
    """A name and an e-mail that git stores, with their role and the commit or tag that has them."""

    role: str
    owner: str
    name: str
    email: str


@dataclass(frozen=True)
class Stored:
    """A commit or an annotated tag: its name in a message, its identities and its message.

    `target` is the type of the object that an annotated tag points at. It is empty for a commit.
    """

    owner: str
    identities: tuple[Identity, ...]
    message: str
    target: str = ""


@dataclass(frozen=True)
class Ref:
    """A name that git keeps for an object: a local branch, a tag or a remote-tracking branch.

    `kind` is `branch`, `tag` or `remote`. The name of a remote-tracking branch starts with
    the name of its remote. `object_type` is `commit`, or `tag` for an annotated tag.
    """

    kind: str
    name: str
    object_type: str
    object_name: str


class FileStore(Protocol):
    """The place that the content of the checked files comes from."""

    def sizes(self, keys: Sequence[str]) -> dict[str, int | None]:
        """Return the size in bytes for each key, or None when the key has no file content."""

    def contents(self, keys: Sequence[str]) -> Iterator[tuple[str, bytes]]:
        """Yield each key with the content of its file, in the order of the keys."""


# --- git and list files -------------------------------------------------------------------


def run_git(
    repo: Path, args: Sequence[str], accept: Sequence[int] = (0,), stdin: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    """Run one git command in the repository and return its result.

    An exit code that is not in `accept` raises a GuardError with the message from git.
    """
    command = " ".join(["git", *args])
    try:
        result = subprocess.run(
            ["git", *args], cwd=str(repo), input=stdin, capture_output=True, check=False
        )
    except OSError as error:
        raise GuardError(f"cannot run {command} in {repo}: {error}") from error
    if result.returncode not in accept:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise GuardError(f"{command} failed with exit code {result.returncode}: {detail}")
    return result


def git_text(
    repo: Path, args: Sequence[str], accept: Sequence[int] = (0,), stdin: bytes | None = None
) -> str:
    """Run one git command and return its output as text."""
    output = run_git(repo, args, accept, stdin).stdout
    try:
        return output.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GuardError(f"the output of git {' '.join(args)} is not valid UTF-8") from error


def git_paths(repo: Path, args: Sequence[str]) -> list[str]:
    """Run one git command that prints NUL-separated paths and return the paths."""
    return [path for path in git_text(repo, args).split("\0") if path]


def repository_folders(start: Path) -> tuple[Path, Path]:
    """Return the top level of the git working tree that contains `start`, and the git folder.

    The git folder is the one that all work trees of the repository share. One git process
    usually answers both (`paired_folders`). When it cannot, git is asked for each path on its
    own, from the top level, so that an error names the one command that failed.
    """
    paired = paired_folders(start)
    if paired is not None:
        return paired
    repo = Path(git_text(start, ["rev-parse", "--show-toplevel"]).rstrip("\n")).resolve()
    return repo, repo / git_text(repo, ["rev-parse", "--git-common-dir"]).rstrip("\n")


def paired_folders(start: Path) -> tuple[Path, Path] | None:
    """Ask one git process for the top level and the shared git folder, or return None.

    Git prints the two paths, one on each line, and the git folder relative to `start` or as
    an absolute path. The answer is None when git cannot run, fails, or prints text that is
    not two lines of UTF-8: a path that holds a line break cannot be told apart from the other.
    """
    command = ["git", "rev-parse", "--show-toplevel", "--git-common-dir"]
    try:
        result = subprocess.run(command, cwd=str(start), capture_output=True, check=False)
        lines = result.stdout.decode("utf-8").split("\n")
    except (OSError, UnicodeDecodeError):
        return None
    if result.returncode != 0 or len(lines) != 3 or lines[2]:
        return None
    repo = Path(lines[0]).resolve()
    here = start.resolve()
    if here == repo or Path(lines[1]).is_absolute():
        return repo, here / lines[1]
    # `here` and its parents are real folders, so `..` is the parent folder.
    return repo, Path(os.path.normpath(here / lines[1]))


def list_entries(path: Path) -> list[tuple[int, str]]:
    """Return the line number and the text of each entry of a list file.

    Empty lines and lines that start with `#` are not entries.
    """
    try:
        lines = path.read_text(encoding="utf-8-sig").split("\n")
    except (OSError, UnicodeDecodeError) as error:
        raise GuardError(f"cannot read {path}: {error}") from error
    stripped = [(number, line.strip()) for number, line in enumerate(lines, start=1)]
    return [(number, text) for number, text in stripped if text and not text.startswith("#")]


def load_deny_list(path: Path) -> tuple[DenyEntry, ...]:
    """Return the entries of the local deny-list.

    Each entry finds its text as a case-insensitive substring.
    """
    return tuple(
        DenyEntry(number, re.compile(re.escape(text), re.IGNORECASE))
        for number, text in list_entries(path)
    )


def load_settings(git_folder: Path) -> Settings:
    """Read the host allow-list next to this script and the local deny-list in the git folder.

    All work trees of one repository share one git folder, so they share one deny-list.
    """
    hosts = [text.lower() for _, text in list_entries(ALLOW_LIST)]
    deny_list = git_folder / DENY_LIST_NAME
    has_deny_list = deny_list.is_file()
    return Settings(
        exact_hosts=frozenset(host for host in hosts if not host.startswith(".")),
        host_suffixes=tuple(host for host in hosts if host.startswith(".")),
        deny=load_deny_list(deny_list) if has_deny_list else (),
        has_deny_list=has_deny_list,
    )


# --- rules for the path of a file ---------------------------------------------------------


def check_extension(path: str) -> list[Violation]:
    """[ext] Allow only the listed file types and the listed exact file names."""
    name = PurePosixPath(path).name
    suffix = PurePosixPath(path).suffix
    if name in ALLOWED_NAMES or suffix.lower() in ALLOWED_SUFFIXES:
        return []
    kind = f"file type '{suffix}'" if suffix else f"file name '{name}'"
    return [Violation(path, None, "ext", f"{kind} is not on the list of allowed file types")]


def check_lock(path: str) -> list[Violation]:
    """[lock] Refuse a lock file in every folder, because it is machine-specific.

    A lock file has the word "lock" as a part of its name, or its name starts with "pylock.".
    The pinned list of the run machine is a requirements file under `locks/`, not a lock file.
    """
    name = PurePosixPath(path).name
    if PurePosixPath(path).suffix.lower() not in LOCK_SUFFIXES or not LOCK_NAME.search(name):
        return []
    message = f"{name} is a lock file: it is machine-specific and must not be tracked"
    return [Violation(path, None, "lock", message)]


def check_dataset_name(path: str) -> list[Violation]:
    """[dataset] Refuse the label file of the dataset and the result file of a clip run, by name."""
    name = PurePosixPath(path).name.lower()
    if name == LABEL_FILE_NAME:
        return [
            Violation(
                path, None, "dataset", "label file of the dataset; it stays on the run machine"
            )
        ]
    if name == RUN_RESULT_NAME:
        message = "result file of a clip run; it stays on the run machine"
        return [Violation(path, None, "dataset", message)]
    return []


def check_deny_file(path: str) -> list[Violation]:
    """[deny] Refuse the deny-list itself: it holds the private words, so it is never tracked."""
    if PurePosixPath(path).name != DENY_LIST_NAME:
        return []
    message = "the local deny-list must never be tracked; keep it in the git folder"
    return [Violation(path, None, "deny", message)]


def check_ignored(path: str, ignored: Mapping[str, str]) -> list[Violation]:
    """[ignored] Refuse a file that the ignore rules of the repository name, or cannot judge.

    Only `git add -f` adds a file that the rules name. `ignored` maps each refused path to
    the message of its violation.
    """
    if path not in ignored:
        return []
    return [Violation(path, None, "ignored", ignored[path])]


def deny_violation(path: str, line: int | None, subject: str, entry: DenyEntry) -> Violation:
    """Build a [deny] violation that names the line of the entry and never its text."""
    message = f"{subject} matches the entry on line {entry.line} of the local deny-list"
    return Violation(path, line, "deny", message)


def check_deny_name(
    path: str, subject: str, value: str, deny: Sequence[DenyEntry]
) -> list[Violation]:
    """[deny] Refuse a name or an identity that contains a deny-list entry."""
    matching = [entry for entry in deny if entry.pattern.search(value)]
    return [deny_violation(path, None, subject, entry) for entry in matching]


def check_path(path: str, settings: Settings, ignored: Mapping[str, str]) -> list[Violation]:
    """Apply the rules that read only the path of a file."""
    return (
        check_extension(path)
        + check_lock(path)
        + check_dataset_name(path)
        + check_deny_file(path)
        + check_ignored(path, ignored)
        + check_name(path, "path", path, settings)
    )


def check_size(path: str, size: int) -> list[Violation]:
    """[size] Refuse a file that is larger than the size limit."""
    if size <= MAX_FILE_BYTES:
        return []
    return [Violation(path, None, "size", f"file has {size} bytes; the limit is {MAX_FILE_BYTES} bytes")]


# --- rules for text -----------------------------------------------------------------------


def find_all(pattern: re.Pattern[str], text: str) -> Iterator[tuple[int, re.Match[str]]]:
    """Yield every match of the pattern in the text, with the 1-based number of its line."""
    line = 1
    position = 0
    for match in pattern.finditer(text):
        line += text.count("\n", position, match.start())
        position = match.start()
        yield line, match


def overlaps(span: tuple[int, int], taken: Sequence[tuple[int, int]]) -> bool:
    """Tell whether a span of the text shares a character with one of the taken spans."""
    return any(span[0] < end and start < span[1] for start, end in taken)


def redact(secret: str) -> str:
    """Return a text in place of a secret that is safe to print: at most its first 4 characters."""
    return secret[:4] + "..."


def preview(secret: str) -> str:
    """Describe a secret for a report: how it starts, or nothing when the secret is short."""
    if len(secret) < MIN_PREVIEW_LENGTH:
        return "not shown"
    return f"starts with {redact(secret)}"


def check_binary(path: str, data: bytes) -> list[Violation]:
    """[binary] Refuse a file that contains a NUL byte or is not valid UTF-8."""
    if b"\0" in data:
        return [Violation(path, None, "binary", "file contains a NUL byte")]
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return [Violation(path, None, "binary", "file is not valid UTF-8 text")]
    return []


def check_secrets(path: str, text: str) -> list[Violation]:
    """[secret] Refuse API keys, tokens, private key headers, passwords and literal secret values.

    The patterns are tried from the most exact one to the most general one, and a text that
    one pattern has reported is not reported again. The message shows at most the first
    4 characters of the secret, and nothing of a short secret such as a password.
    """
    violations = []
    taken: list[tuple[int, int]] = []
    for label, pattern in SECRET_PATTERNS:
        for line, match in find_all(pattern, text):
            if overlaps(match.span(), taken):
                continue
            taken.append(match.span())
            secret = match.group(match.lastindex or 0)
            violations.append(Violation(path, line, "secret", f"{label} ({preview(secret)})"))
    return violations


def base64_runs(text: str) -> Iterator[list[tuple[int, int]]]:
    """Yield each run of base64 text as the list of its pieces.

    A run is one piece of base64 text, or several pieces with nothing but line breaks between
    them. A piece is given as the number of its line and its length. Text without a letter
    and without a digit, such as a rule line of dashes, is not a piece: it holds no data.
    """
    run: list[tuple[int, int]] = []
    end = 0
    for line, match in find_all(BASE64_PIECE, text):
        if not any(character.isalnum() for character in match.group()):
            continue
        if run and LINE_BREAKS.fullmatch(text, end, match.start()) is None:
            yield run
            run = []
        run.append((line, len(match.group())))
        end = match.end()
    if run:
        yield run


def check_encoded(path: str, text: str) -> list[Violation]:
    """[base64] [pixels] Refuse binary data that is written as text, such as the bytes of an image.

    Three forms are refused: a long run of base64 characters, also when line breaks cut it
    into pieces; an address that holds its data (a `data:` address); and a long table of
    byte values. A table of byte values is how an image array looks when it is written as
    numbers. Hexadecimal text is base64 text for this rule.
    """
    violations = []
    for run in base64_runs(text):
        total = sum(length for _, length in run)
        if len(run) > 1 and total >= MIN_WRAPPED_RUN:
            message = (
                f"run of {total} base64 characters in {len(run)} pieces; "
                f"{MIN_WRAPPED_RUN} or more are refused"
            )
            violations.append(Violation(path, run[0][0], "base64", message))
            continue
        for line, length in run:
            if length >= MIN_BASE64_RUN:
                message = f"run of {length} base64 characters; {MIN_BASE64_RUN} or more are refused"
                violations.append(Violation(path, line, "base64", message))
    for line, _ in find_all(DATA_ADDRESS, text):
        violations.append(Violation(path, line, "base64", "address that holds encoded data"))
    for line, match in find_all(BYTE_TABLE, text):
        count = len(re.findall(r"\d+", match.group()))
        if count >= MIN_BYTE_TABLE:
            message = f"table of {count} byte values; {MIN_BYTE_TABLE} or more in a row are refused"
            violations.append(Violation(path, line, "pixels", message))
    return violations


def host_allowed(host: str, settings: Settings) -> bool:
    """Tell whether a host is always allowed, listed exactly, or covered by a listed suffix."""
    if host in ALWAYS_ALLOWED_HOSTS or host in settings.exact_hosts:
        return True
    return host.endswith(settings.host_suffixes)


def check_hosts(path: str, text: str, settings: Settings) -> list[Violation]:
    """[host] Refuse a host that is not on the allow-list.

    A host is read from an address of any scheme, from the short form of a git remote, and
    from an option that names a package index. An address of a private network is read also
    when it stands alone. A host is reported once for each line.
    """
    refused: dict[tuple[int, str], None] = {}
    for pattern in HOST_PATTERNS:
        for line, match in find_all(pattern, text):
            host = match.group(1).rstrip(".")
            if host and not host_allowed(host.lower(), settings):
                refused[(line, host)] = None
    return [
        Violation(path, line, "host", f"host '{host}' is not in {ALLOW_LIST.name}")
        for line, host in refused
    ]


def check_home(path: str, text: str) -> list[Violation]:
    """[home] Refuse an absolute path into a home folder, or into another folder of one user."""
    return [
        Violation(path, line, "home", message)
        for message, pattern in HOME_PATTERNS
        for line, _ in find_all(pattern, text)
    ]


def check_deny_content(path: str, text: str, deny: Sequence[DenyEntry]) -> list[Violation]:
    """[deny] Refuse text that contains a deny-list entry, once for each line."""
    violations = []
    for entry in deny:
        lines = sorted({line for line, _ in find_all(entry.pattern, text)})
        violations += [deny_violation(path, line, "content", entry) for line in lines]
    return violations


def check_text(path: str, text: str, settings: Settings) -> list[Violation]:
    """Apply the rules for text, and sort the result by line.

    The text is the content of a file, or the message of a commit or of a tag.
    """
    violations = (
        check_secrets(path, text)
        + check_encoded(path, text)
        + check_hosts(path, text, settings)
        + check_home(path, text)
        + check_deny_content(path, text, settings.deny)
    )
    return sorted(violations, key=lambda violation: violation.line or 0)


def check_name(path: str, subject: str, name: str, settings: Settings) -> list[Violation]:
    """Apply the rules for text to a name: the path of a file, or the name of a branch or a tag.

    The rules for secrets, hosts, home folders and the deny-list read the name. A name has
    no line. `subject` says what the name is, and every message ends with it or starts with it.
    """
    found = check_secrets(path, name) + check_hosts(path, name, settings)
    if HOME_FOLDERS.search(name):
        found.append(Violation(path, None, "home", HOME_FOLDERS_MESSAGE))
    if USER_FOLDER.search("/" + name):
        found.append(Violation(path, None, "home", USER_FOLDER_MESSAGE))
    named = [
        replace(violation, line=None, message=f"{violation.message} [{subject}]")
        for violation in found
    ]
    return named + check_deny_name(path, subject, name, settings.deny)


def json_document(text: str) -> list[object]:
    """Return the JSON document of a text as a list of one, or no document for other text."""
    try:
        return [json.loads(text)]
    except (ValueError, RecursionError):
        return []


def json_documents(text: str) -> list[object]:
    """Return the JSON documents of a text: the whole text, or else every line that is one.

    A byte order mark at the start is not a part of the first document.
    """
    text = text.removeprefix("\ufeff")
    whole = json_document(text)
    if whole:
        return whole
    return [document for line in text.splitlines() for document in json_document(line)]


def dataset_kinds(documents: Sequence[object]) -> list[str]:
    """Tell which kinds of dataset data the JSON documents hold, at any depth.

    A label file has the keys `images` and `annotations`. A prediction file has the key
    `predictions`. The result of a clip run has the keys `frames` and `tracks`. Each of these
    values is a list. A record of the label format has the key `track_id` and a box
    (`bbox_image` or `bbox_pitch`); such records are named only when no other kind is found.
    """
    found = {"labels": False, "predictions": False, "result": False, "records": False}
    stack = list(documents)
    while stack:
        node = stack.pop()
        if isinstance(node, list):
            stack.extend(node)
        if not isinstance(node, dict):
            continue
        stack.extend(node.values())
        lists = {key for key, value in node.items() if isinstance(value, list)}
        found["labels"] |= {"images", "annotations"} <= lists
        found["predictions"] |= "predictions" in lists
        found["result"] |= {"frames", "tracks"} <= lists
        found["records"] |= RECORD_TRACK in node and any(box in node for box in RECORD_BOXES)
    if any(found[kind] for kind in ("labels", "predictions", "result")):
        found["records"] = False
    messages = {
        "labels": "label data of the dataset (keys images and annotations)",
        "predictions": "prediction data (key predictions)",
        "result": "result of a clip run (keys frames and tracks)",
        "records": "label or prediction data (records with the key track_id and a box)",
    }
    return [message for kind, message in messages.items() if found[kind]]


def check_dataset_content(path: str, text: str) -> list[Violation]:
    """[dataset] Refuse a JSON file with label data, prediction data or the result of a clip run.

    The rule reads a `.json` file and a `.jsonl` file, also when the data lies inside another
    key or a list, when each line is one record, and after a byte order mark. A file that
    holds no valid JSON is none of them.
    """
    if PurePosixPath(path).suffix.lower() not in DATASET_SUFFIXES:
        return []
    return [
        Violation(path, None, "dataset", f"{kind}; it stays on the run machine")
        for kind in dataset_kinds(json_documents(text))
    ]


def check_content(path: str, data: bytes, settings: Settings) -> list[Violation]:
    """Apply the content rules to the bytes of one file. A rule for the whole file comes first."""
    binary = check_binary(path, data)
    if binary:
        return binary  # The other content rules read text.
    text = data.decode("utf-8")
    return check_dataset_content(path, text) + check_text(path, text, settings)


# --- files --------------------------------------------------------------------------------


def with_origin(violation: Violation, origin: str) -> Violation:
    """Return the violation with its origin at the end of the message, when it has an origin."""
    if not origin:
        return violation
    return replace(violation, message=f"{violation.message} [{origin}]")


def check_files(
    entries: Sequence[Entry], files: FileStore, settings: Settings, ignored: Mapping[str, str]
) -> list[Violation]:
    """Apply every file rule to every entry. The result keeps the order of the entries."""
    found = [check_path(entry.path, settings, ignored) for entry in entries]
    keys = list(dict.fromkeys(entry.key for entry in entries if entry.key is not None))
    sizes = files.sizes(keys)
    readers: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        size = None if entry.key is None else sizes[entry.key]
        if size is None:
            continue  # A deleted file or a submodule folder has no content to check.
        oversize = check_size(entry.path, size)
        if oversize:
            found[index] += oversize  # The guard does not read a file that is too large.
        else:
            readers.setdefault(entry.key, []).append(index)
    for key, data in files.contents(list(readers)):
        for index in readers[key]:
            found[index] += check_content(entries[index].path, data, settings)
    return [
        with_origin(violation, entry.origin)
        for entry, violations in zip(entries, found)
        for violation in violations
    ]


def linked_folder(repo: Path, path: str, links: dict[str, bool]) -> str | None:
    """Return the first folder of the path that is a symbolic link in the working tree, or None.

    `links` keeps the answer for every folder that was tested.
    """
    folder = ""
    for part in PurePosixPath(path).parts[:-1]:
        folder = f"{folder}/{part}" if folder else part
        if folder not in links:
            links[folder] = (repo / folder).is_symlink()
        if links[folder]:
            return folder
    return None


def ignored_paths(repo: Path, entries: Sequence[Entry]) -> dict[str, str]:
    """Return the paths of the entries that the [ignored] rule refuses, each with its message.

    The ignore rules are the `.gitignore` files of the working tree and the exclude file of
    the git folder. The ignore file of the user account is not read, so the answer is the
    same on every machine. Git cannot test a path below a folder that is a symbolic link in
    the working tree, for example a path of a commit of another branch. Such a path is
    refused with a message that names the link.
    """
    refused: dict[str, str] = {}
    links: dict[str, bool] = {}
    testable = []
    for path in dict.fromkeys(entry.path for entry in entries):
        link = linked_folder(repo, path, links)
        if link is None:
            testable.append(path)
        else:
            refused[path] = (
                f"git cannot test this path against the ignore rules, because {link} is a "
                "symbolic link in the working tree; remove the link and run the check again"
            )
    if not testable:
        return refused
    command = [
        "-c",
        f"core.excludesFile={os.devnull}",
        "check-ignore",
        "--no-index",
        "-z",
        "--stdin",
    ]
    paths = "".join(path + "\0" for path in testable).encode("utf-8")
    # git exits with code 1 when no path is ignored.
    named = git_text(repo, command, accept=(0, 1), stdin=paths).split("\0")
    message = "the ignore rules of the repository name this path, so it must not be tracked"
    refused.update((path, message) for path in named if path)
    return refused


class ObjectInfo:
    """One `git cat-file --batch-check` process, which tells the type and the size of an object.

    It answers one name at a time: git prints one line for each line that it reads, or ends.
    It never prints the content of an object, so a damaged object cannot leave the guard
    waiting for bytes that do not come. The error output of git goes to a temporary file, so
    that git never waits for the guard to read it.
    """

    COMMAND = "git cat-file --batch-check"

    def __init__(self, repo: Path) -> None:
        self._errors = tempfile.TemporaryFile()
        try:
            self._process = subprocess.Popen(
                ["git", "cat-file", "--batch-check"],
                cwd=str(repo),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._errors,
            )
        except OSError as error:
            self._errors.close()
            raise GuardError(f"cannot run {self.COMMAND} in {repo}: {error}") from error
        self._input: IO[bytes] = self._process.stdin  # type: ignore[assignment]
        self._output: IO[bytes] = self._process.stdout  # type: ignore[assignment]

    def ask(self, name: str) -> str:
        """Send one name and return the line of the answer without its line break.

        The line is `<object name> <type> <size>`, or the name and a word such as `missing`.
        """
        try:
            self._input.write(name.encode("ascii") + b"\n")
            self._input.flush()
        except BrokenPipeError:
            self._ended()
        line = self._output.readline()
        if not line.endswith(b"\n"):
            self._ended()
        return line[:-1].decode("ascii", "replace")

    def finish(self) -> None:
        """Close the input, wait for git to end, and raise a GuardError for an exit code that is not 0."""
        try:
            self._input.close()
        except BrokenPipeError:
            # Git ended before it read a name that is still in the buffer (`ask` met the same
            # broken pipe). Its exit code below, or `_ended` after this call, reports it.
            pass
        self._output.read()
        code = self._process.wait()
        self._errors.seek(0)
        detail = self._errors.read().decode("utf-8", "replace").strip()
        self.stop()
        if code != 0:
            raise GuardError(f"{self.COMMAND} failed with exit code {code}: {detail}")

    def stop(self) -> None:
        """End git if it still runs, and close the pipes and the file of the error output."""
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait()
        for stream in (self._input, self._output, self._errors):
            stream.close()

    def _ended(self) -> NoReturn:
        """Report that git ended before it answered: its exit code, or a missing answer."""
        self.finish()
        raise GuardError(f"{self.COMMAND} did not answer for every object")


class GitObjects:
    """Reads files and other objects from the git database. The key of a file is its object name.

    For the length of one guard run it keeps one git process that tells the type and the size
    of an object (`ObjectInfo`). It starts when it is first needed. Each call of `contents`
    starts one git process for all its keys, or one for each 64 MB of content. Use the reader
    in a `with` block: its end stops the process and checks its exit code. Nothing is kept
    from one guard run to the next.
    """

    def __init__(self, repo: Path) -> None:
        self._repo = repo
        self._sizes: dict[str, int] = {}
        self._info: ObjectInfo | None = None

    def __enter__(self) -> GitObjects:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        info, self._info = self._info, None
        if info is None:
            return
        try:
            if kind is None:
                info.finish()
        finally:
            info.stop()

    def _ask(self, name: str) -> str:
        """Return the answer of the `git cat-file --batch-check` process for one name."""
        if self._info is None:
            self._info = ObjectInfo(self._repo)
        return self._info.ask(name)

    def names_commit(self, revision: str) -> bool:
        """Tell whether a fixed revision such as `HEAD` names a commit, as `git rev-parse --verify` does.

        When the `git cat-file --batch-check` process cannot answer, for example because the
        commit is damaged, `git rev-parse --verify` is asked, as the guard did before it had
        this process: an error then names that command. A later question starts a new process.
        """
        try:
            fields = self._ask(revision + "^{commit}").split(" ")
        except GuardError:
            self._info = None
            return revision_exists(self._repo, revision)
        return len(fields) == 3 and fields[1] == "commit"

    def sizes(self, keys: Sequence[str]) -> dict[str, int | None]:
        """Return the size in bytes of each object."""
        for key in keys:
            if key not in self._sizes:
                fields = self._ask(key).split(" ")  # <object name> <type> <size>, or <object name> missing
                if len(fields) != 3 or fields[0] != key:
                    raise GuardError(f"git cannot find the object {key}")
                self._sizes[key] = int(fields[2])
        return {key: self._sizes[key] for key in keys}

    def contents(self, keys: Sequence[str]) -> Iterator[tuple[str, bytes]]:
        """Yield each key with the bytes of its object, in the order of the keys."""
        sizes = self.sizes(keys)
        batch: list[str] = []
        batch_bytes = 0
        for key in keys:
            batch.append(key)
            batch_bytes += sizes[key] or 0
            if batch_bytes >= BATCH_BYTES:
                yield from self._read(batch)
                batch, batch_bytes = [], 0
        if batch:
            yield from self._read(batch)

    def _read(self, keys: Sequence[str]) -> Iterator[tuple[str, bytes]]:
        """Yield the objects of one batch.

        For each object git prints a header line, the bytes and a line break.
        """
        names = "".join(key + "\n" for key in keys).encode("ascii")
        output = run_git(self._repo, ["cat-file", "--batch"], stdin=names).stdout
        position = 0
        for key in keys:
            start = output.find(b"\n", position) + 1
            end = start + self._sizes[key]
            if start == 0 or len(output) <= end:
                raise GuardError(f"git cat-file --batch did not print the object {key}")
            yield key, output[start:end]
            position = end + 1


class WorktreeFiles:
    """Reads files from the working tree. The key of a file is its path."""

    def __init__(self, repo: Path) -> None:
        self._repo = repo

    def sizes(self, keys: Sequence[str]) -> dict[str, int | None]:
        """Return the size in bytes of each file, or None for a deleted file or a submodule."""
        return {key: self._size(key) for key in keys}

    def contents(self, keys: Sequence[str]) -> Iterator[tuple[str, bytes]]:
        """Yield each path with what git stores for it: its bytes, or the target of a link."""
        for key in keys:
            yield key, self._read(key)

    def _size(self, path: str) -> int | None:
        """Return the size of one file, or None when the path is not a file or a link."""
        try:
            status = (self._repo / path).lstat()
        except (FileNotFoundError, NotADirectoryError):
            return None
        except OSError as error:
            raise GuardError(f"cannot read {path}: {error}") from error
        is_file = stat.S_ISREG(status.st_mode) or stat.S_ISLNK(status.st_mode)
        return status.st_size if is_file else None

    def _read(self, path: str) -> bytes:
        """Return the bytes of one file, or the target of a symbolic link as bytes."""
        full_path = self._repo / path
        try:
            if full_path.is_symlink():
                return os.fsencode(full_path.readlink())
            return full_path.read_bytes()
        except OSError as error:
            raise GuardError(f"cannot read {path}: {error}") from error


def raw_changes(output: str) -> Iterator[tuple[str, str, str, str]]:
    """Yield the commit, the path, the new file mode and the new object name of each changed file.

    `output` is the output of a git diff command with `--raw -z`. The commit is empty when
    the output names no commit. A deleted file and a file with a merge conflict have no new
    object, so they are left out. A change of a merge commit has one mode and one object name
    for each parent before those of the result.
    """
    tokens = iter(output.split("\0"))
    commit = ""
    for token in tokens:
        if not token.startswith(":"):
            commit = token
            continue
        fields = token.lstrip(":").split(" ")
        sides = len(fields) // 2
        path = next(tokens, None)
        if sides < 2 or path is None:
            raise GuardError("cannot read the raw output of git diff")
        mode, name = fields[sides - 1], fields[2 * sides - 1]
        if name.strip("0"):
            yield commit, path, mode, name


def staged_entries(repo: Path) -> list[Entry]:
    """Return the files that the git index adds or changes, with their staged content.

    A renamed file counts as a new file, and a file that changes its type counts as changed.
    """
    command = ["diff", "--cached", "--raw", "--no-renames", "--no-abbrev", "-z"]
    return [
        Entry(path, None if mode == SUBMODULE_MODE else name)
        for _, path, mode, name in raw_changes(git_text(repo, command))
    ]


def pushed_entries(repo: Path, commits: Sequence[str]) -> list[Entry]:
    """Return the files that the commits add or change, each version once with its first commit.

    For a merge commit, only the files that differ from every parent count: the other files
    come from a parent, and that parent is checked on its own or is on the remote already.
    """
    if not commits:
        return []
    command = [
        "diff-tree",
        "--stdin",
        "-r",
        "--root",
        "-c",
        "--raw",
        "--no-renames",
        "--no-abbrev",
        "-z",
    ]
    names = "".join(commit + "\n" for commit in commits).encode("ascii")
    entries: dict[tuple[str, str | None], Entry] = {}
    for commit, path, mode, name in raw_changes(git_text(repo, command, stdin=names)):
        key = None if mode == SUBMODULE_MODE else name
        entries.setdefault((path, key), Entry(path, key, f"commit {commit[:SHORT_NAME_LENGTH]}"))
    return list(entries.values())


def tracked_paths(repo: Path) -> list[str]:
    """Return the path of every tracked file, once.

    During a merge conflict git lists an unmerged path once for each of its versions.
    """
    return list(dict.fromkeys(git_paths(repo, ["ls-files", "-z"])))


def given_path(repo: Path, argument: str) -> str:
    """Return a command-line path relative to the repository root. The path must be a file."""
    absolute = Path(argument).absolute()
    located = absolute.parent.resolve() / absolute.name  # Keep a symbolic link as it is.
    try:
        relative = located.relative_to(repo)
    except ValueError:
        raise GuardError(f"{argument} is outside the repository {repo}") from None
    if not (located.is_file() or located.is_symlink()):
        raise GuardError(f"{argument} is not a file")
    return relative.as_posix()


# --- identities, commits and tags ---------------------------------------------------------


def clean_ident(value: str) -> str:
    """Return a name or an e-mail without the characters that git removes from an identity."""
    return value.translate(IDENT_DELIMITERS).strip(IDENT_CRUD)


def local_identity(repo: Path) -> dict[str, str]:
    """Return the name and the e-mail in the configuration of this repository, cleaned.

    The keys are `name` and `email`. A setting that is missing or empty has no key.
    """
    # git exits with code 1 when the local configuration has none of the two settings.
    command = ["config", "--local", "--null", "--get-regexp", r"^user\.(name|email)$"]
    found = {}
    for item in git_text(repo, command, accept=(0, 1)).split("\0"):
        key, _, value = item.partition("\n")
        if clean_ident(value):
            found[key.removeprefix("user.")] = clean_ident(value)
    return found


def read_identity(role: str, owner: str, ident: str) -> Identity:
    """Read a name and an e-mail as git writes them: name, e-mail, time and time zone."""
    match = IDENT.fullmatch(ident)
    if match is None:
        raise GuardError(f"cannot read the {role} of {owner}")
    return Identity(role, owner, match.group(1), match.group(2))


def next_commit_identities(repo: Path) -> list[Identity]:
    """Return the author and the committer that git will use for the next commit."""
    variables = (("author", "GIT_AUTHOR_IDENT"), ("committer", "GIT_COMMITTER_IDENT"))
    return [
        read_identity(role, "the next commit", git_text(repo, ["var", variable]).strip())
        for role, variable in variables
    ]


def check_identities(
    repo: Path, identities: Callable[[], Sequence[Identity]], deny: Sequence[DenyEntry]
) -> list[Violation]:
    """[identity] Require a local name and e-mail, and require that every given identity has them.

    Without the local name, git takes the name of the user account, which belongs to one
    machine. The identities are read only when the local name and the local e-mail are set.
    """
    expected = local_identity(repo)
    missing = [
        Violation(NO_PATH, None, "identity", IDENTITY_NOT_SET.format(field=field))
        for field, _ in IDENTITY_FIELDS
        if field not in expected
    ]
    if missing:
        return missing
    violations = []
    for identity in identities():
        for field, word in IDENTITY_FIELDS:
            if clean_ident(getattr(identity, field)) != expected[field]:
                message = (
                    f"the {identity.role} {word} of {identity.owner} is not the local user.{field}"
                )
                violations.append(Violation(NO_PATH, None, "identity", message))
        subject = f"the {identity.role} of {identity.owner}"
        value = f"{identity.name} <{identity.email}>"
        violations += check_deny_name(NO_PATH, subject, value, deny)
    return violations


def revision_exists(repo: Path, revision: str) -> bool:
    """Tell whether a revision such as the name of a commit from the environment names a commit."""
    # With --verify --quiet, git exits with code 1 and prints nothing when it does not.
    command = ["rev-parse", "--verify", "--quiet", revision + "^{commit}"]
    return run_git(repo, command, accept=(0, 1)).returncode == 0


def known_remotes(repo: Path) -> list[str]:
    """Return the remotes whose remote-tracking branches tell what a push need not send.

    The hook runner gives the name of the remote of a push. Without that name (a run by
    hand), every remote of the repository counts. A remote that this repository does not
    know by name (a push to an address) has no remote-tracking branches, so the result is
    empty and everything is checked.
    """
    remotes = [name for name in git_text(repo, ["remote"]).split("\n") if name]
    pushed_to = os.environ.get(REMOTE_VARIABLE)
    if pushed_to is None:
        return remotes
    return [pushed_to] if pushed_to in remotes else []


def unpushed_commits(repo: Path, objects: GitObjects, remotes: Sequence[str]) -> list[str]:
    """Return the commits that a push can send and that the remotes do not have, oldest first.

    A push can send every local branch, every tag and HEAD, so the commits of all of them
    count, also when one branch is pushed: the hook runner tells the guard only about the
    first branch of a push. The newest commit of that branch counts too. The commits that a
    remote has are the commits of its remote-tracking branches.
    """
    starts = ["--branches", "--tags"]
    if objects.names_commit("HEAD"):
        starts.append("HEAD")
    # The pushed commit comes from the environment and can hold any text, also a line break,
    # which the object reader cannot take. So git rev-parse tests it.
    pushed = os.environ.get(PUSHED_COMMIT_VARIABLE, "")
    if pushed and revision_exists(repo, pushed):
        starts.append(pushed)
    known = ["--not", *(f"--remotes={remote}" for remote in remotes)] if remotes else []
    return git_text(repo, ["rev-list", "--reverse", *starts, *known, "--"]).split()


def read_ref_name(full_name: str) -> tuple[str, str]:
    """Return the kind and the short name of a full name such as `refs/heads/main`.

    A name outside the branches, the tags and the remote-tracking branches has the kind `ref`
    and keeps its full name.
    """
    for prefix, kind in REF_PREFIXES.items():
        if full_name.startswith(prefix):
            return kind, full_name[len(prefix) :]
    return "ref", full_name


def list_refs(repo: Path) -> list[Ref]:
    """Return every local branch, every tag and every remote-tracking branch."""
    command = ["for-each-ref", "--format=%(objecttype)%00%(objectname)%00%(refname)", *REF_PREFIXES]
    refs = []
    for line in git_text(repo, command).split("\n"):
        if line:
            object_type, object_name, full_name = line.split("\0")
            refs.append(Ref(*read_ref_name(full_name), object_type, object_name))
    return refs


def pushed_names(refs: Sequence[Ref], remotes: Sequence[str]) -> list[tuple[str, str]]:
    """Return the kind and the name of every branch and tag that a push can add to a remote.

    A local branch counts when the remotes have no branch of its name. Git does not record
    which tags a remote has, so every local tag counts. The hook runner gives the name that
    the first branch of a push gets on the remote. It can differ from every local name, so
    it counts too.
    """
    on_remote = {
        ref.name[len(remote) + 1 :]
        for ref in refs
        if ref.kind == "remote"
        for remote in remotes
        if ref.name.startswith(remote + "/")
    }
    names = [(ref.kind, ref.name) for ref in refs if ref.kind in ("branch", "tag")]
    pushed = os.environ.get(PUSHED_NAME_VARIABLE)
    if pushed and read_ref_name(pushed) not in names:
        names.append(read_ref_name(pushed))
    return [(kind, name) for kind, name in names if kind != "branch" or name not in on_remote]


def read_stored(owner: str, data: bytes) -> Stored:
    """Read the raw content of a commit or of an annotated tag.

    The content has header lines, an empty line and the message. A header line that goes on
    in the next line starts that line with a space. The header `type` of an annotated tag
    names the type of the object that the tag points at.
    """
    head, _, message = data.partition(b"\n\n")
    identities = []
    target = ""
    for line in head.decode("utf-8", "replace").split("\n"):
        role, _, value = line.partition(" ")
        if role in IDENT_ROLES:
            identities.append(read_identity(role, owner, value))
        elif role == "type":
            target = value
    return Stored(owner, tuple(identities), message.decode("utf-8", "replace"), target)


def unpushed_history(
    objects: GitObjects, commits: Sequence[str], tags: Sequence[Ref]
) -> list[Stored]:
    """Return the given commits and then the given annotated tags, as git stores them."""
    owners = [f"commit {commit[:SHORT_NAME_LENGTH]}" for commit in commits]
    owners += [f"tag {tag.name}" for tag in tags]
    names = [*commits, *(tag.object_name for tag in tags)]
    return [read_stored(owner, data) for owner, (_, data) in zip(owners, objects.contents(names))]


def check_tag_target(name: str, target: str) -> list[Violation]:
    """[tag] Refuse a tag that does not point at a commit, directly or as an annotated tag.

    A push sends the object of such a tag, for example the content of a file, and no rule
    of the guard reads it.
    """
    if target == "commit":
        return []
    what = TAG_TARGETS.get(target, f"an object of the type {target}")
    message = (
        f"the tag {name} points at {what} and not at a commit; "
        "a push sends it without a check, so delete the tag"
    )
    return [Violation(NO_PATH, None, "tag", message)]


def check_push(repo: Path, objects: GitObjects, settings: Settings) -> list[Violation]:
    """Check what a push can send.

    The commits and the annotated tags are checked for their files, their messages and their
    identities, every tag for what it points at, and the new names of branches and tags as
    text.
    """
    remotes = known_remotes(repo)
    refs = list_refs(repo)
    tags = [ref for ref in refs if ref.kind == "tag"]
    annotated = [tag for tag in tags if tag.object_type == "tag"]
    commits = unpushed_commits(repo, objects, remotes)
    entries = pushed_entries(repo, commits)
    history = unpushed_history(objects, commits, annotated)
    violations = check_files(entries, objects, settings, ignored_paths(repo, entries))
    for stored in history:
        origin = f"message of {stored.owner}"
        found = check_text(NO_PATH, stored.message, settings)
        violations += [with_origin(violation, origin) for violation in found]
    targets = {tag.name: stored.target for tag, stored in zip(annotated, history[len(commits) :])}
    for tag in tags:
        violations += check_tag_target(tag.name, targets.get(tag.name, tag.object_type))
    for kind, name in pushed_names(refs, remotes):
        violations += check_name(NO_PATH, f"the name of {kind} {name}", name, settings)
    identities = [identity for stored in history for identity in stored.identities]
    return violations + check_identities(repo, lambda: identities, settings.deny)


def censor(violation: Violation, deny: Sequence[DenyEntry]) -> Violation:
    """Return the violation with no secret and no deny-list text in its path or message."""
    return replace(
        violation, path=hide(violation.path, deny), message=hide(violation.message, deny)
    )


def hide(text: str, deny: Sequence[DenyEntry]) -> str:
    """Return the text with every secret redacted and every deny-list entry masked."""
    for _, pattern in SECRET_PATTERNS:
        text = pattern.sub(lambda match: redact(match.group()), text)
    for entry in deny:
        text = entry.pattern.sub(MASK, text)
    return text


# --- command line -------------------------------------------------------------------------


class GuardParser(argparse.ArgumentParser):
    """An argument parser that reports a usage error as a GuardError."""

    def error(self, message: str) -> NoReturn:
        """Raise the usage error. The caller prints it and returns exit code 2."""
        raise GuardError(f"{message}\n{self.format_usage().rstrip()}")


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    """Read the command line: one mode, and an optional repository root."""
    parser = GuardParser(
        prog="guard.py",
        description="Refuse media files, secrets, machine-specific data and a wrong git identity.",
        epilog="Exit code 0 means clean, 1 means violations, 2 means a usage or git error.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--staged", action="store_true", help="check the files and the identity of the next commit"
    )
    mode.add_argument(
        "--push",
        action="store_true",
        help="check what a push can send: new commits, every local tag, new branch and tag names",
    )
    mode.add_argument(
        "--tracked", action="store_true", help="check every tracked file of the working tree"
    )
    mode.add_argument(
        "--paths", nargs="+", metavar="FILE", help="check these working-tree files only"
    )
    parser.add_argument(
        "--repo", type=Path, metavar="DIR", help="repository to check (default: the current one)"
    )
    return parser.parse_args(argv)


def find_violations(options: argparse.Namespace, repo: Path, settings: Settings) -> list[Violation]:
    """Return every violation for the mode that the command line selects."""
    if options.push:
        with GitObjects(repo) as objects:
            violations = check_push(repo, objects, settings)
    elif options.staged:
        entries = staged_entries(repo)
        ignored = ignored_paths(repo, entries)
        with GitObjects(repo) as objects:
            violations = check_files(entries, objects, settings, ignored)
        violations += check_identities(repo, lambda: next_commit_identities(repo), settings.deny)
    else:
        paths = options.paths or []
        names = (
            tracked_paths(repo) if options.tracked else [given_path(repo, path) for path in paths]
        )
        entries = [Entry(name, name) for name in names]
        violations = check_files(
            entries, WorktreeFiles(repo), settings, ignored_paths(repo, entries)
        )
    return [censor(violation, settings.deny) for violation in violations]


def main(argv: Sequence[str]) -> int:
    """Run the guard, print one line for each violation and return the exit code."""
    try:
        options = parse_arguments(argv)
        repo, git_folder = repository_folders(options.repo or Path.cwd())
        settings = load_settings(git_folder)
        violations = find_violations(options, repo, settings)
    except GuardError as error:
        print(f"guard: error: {error}", file=sys.stderr)
        return EXIT_ERROR
    for violation in violations:
        print(violation.render())
    if not settings.has_deny_list:
        print(DENY_LIST_NOTICE, file=sys.stderr)
    return EXIT_VIOLATIONS if violations else EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
