"""Tests of the Gemini backend. They use fake client objects only: no SDK and no network.

The fake objects copy the contract of the SDK (google-genai 3.3.0, source read on 2026-11-14)
where the backend depends on it:

- ``create`` has the options of the SDK (``timeout``, ``automatic_continuation`` and five more) and
  refuses a keyword that is neither an option nor a field of the request models.
- Unless a call passes ``automatic_continuation=False``, the SDK continues an answer that stopped at
  the token cap of one request (status ``requires_continuation`` with a ``continuation_token``) by
  itself, with one more HTTP request per step. It adds up the token counts. When a later request
  fails, it raises ``ContinuationInterruptedError``, which has no status code.
- An error with an answer of the server keeps its HTTP status in ``status_code``. An error without
  such an answer (a lost connection, a client timeout) has ``status_code = None`` and the class
  name of the SDK. An answer that the SDK cannot read gives ``APIResponseValidationError`` with the
  status of that answer, which is a success status.
- The two option models of the SDK client refuse a keyword that is not one of their fields.

Three parts of the SDK are not copied. An answer has no ``steps``: the fake sets ``output_text``
directly, and the SDK builds it from the steps. The fake takes no ``httpx.Timeout`` as a timeout.
And the retries of the SDK itself are not played: the client asks for one attempt (``attempts=1``
counts the first request), and a test compares that value only.
"""

from __future__ import annotations

import ast
import base64
import enum
import importlib.metadata
import logging
import random
import sys
import threading
import traceback
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pitchvlm.vlm import client as client_module
from pitchvlm.vlm.client import (
    BACKOFF_BASE_S,
    BACKOFF_CAP_S,
    FLEX_BUSY_STATUS_CODES,
    FLEX_BUSY_WAIT_S,
    FLEX_TIMEOUT_S,
    STANDARD_TIMEOUT_S,
    GeminiBackend,
    SpendRateLimiter,
    backoff_delay_s,
    billed_tier,
    build_interaction_kwargs,
    build_sdk_client,
    is_retryable_status,
    is_unread_answer,
    normalize_usage,
    status_code_of,
    transport_failure_of,
    usage_accounting,
)
from pitchvlm.vlm.cost import Ledger, cost_usd
from pitchvlm.vlm.types import (
    BackendError,
    BilledAnswerError,
    FlexBusyError,
    ImagePart,
    RequestMeta,
    ServiceTier,
    TextPart,
    UnconfirmedCallError,
    Usage,
    VLMRequest,
)

MODEL = "gemini-4-argon"
LOGGER_NAME = "pitchvlm.vlm.client"
#: Environment variable used by these tests only. The variable of a real run is never read.
ENV_NAME = "PITCHVLM_TEST_VARIABLE"
#: A made-up value that stands in for the content of that variable. It belongs to no service.
FAKE_VALUE = "made-up-value"
DIAGRAM_BYTES = b"\x89PNG-diagram"
FRAME_BYTES = b"\xff\xd8\xff-frame"
ANSWER = '{"people": []}'
CALL_SECONDS = 2.5
#: The token usage of ``make_usage()`` after the backend has normalised it.
DEFAULT_USAGE = Usage(input_tokens=1500, output_tokens=300, thought_tokens=100, cached_tokens=0)


# --- Fakes -----------------------------------------------------------------------


class FakeTime:
    """A clock that moves only when a test or the fake sleep moves it."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeApiError(Exception):
    """Stands for the SDK error of a call that the server answered with an error: the code is in ``status_code``."""

    def __init__(self, status_code: int, message: str = "fake API error") -> None:
        super().__init__(message)
        self.status_code = status_code


class APIConnectionError(Exception):
    """Stands for the SDK error of a call that got no answer from the server.

    It has the class name of the SDK error, because the backend knows the error by that name.
    Like the SDK error, it has no status code.
    """

    status_code = None

    def __init__(self, message: str = "Connection error.") -> None:
        super().__init__(message)


class APITimeoutError(APIConnectionError):
    """Stands for the SDK error of a client timeout. As in the SDK, it is a subclass of the connection error."""

    def __init__(self, message: str = "Request timed out.") -> None:
        super().__init__(message)


class APIResponseValidationError(Exception):
    """Stands for the SDK error of an answer that the SDK could not read.

    Like the SDK error, it keeps the HTTP status of that answer in ``status_code``. The provider
    answered, so the status is a success status.
    """

    def __init__(self, status_code: int = 200) -> None:
        super().__init__("Data returned by API invalid for expected schema.")
        self.status_code = status_code


class ContinuationInterruptedError(Exception):
    """Stands for the SDK error of an automatic continuation that failed after an earlier request was answered.

    Like the SDK error (``google.genai.errors``), it is a plain ``Exception`` without a status
    code. ``partial_interaction`` holds the answers of the requests that were finished and billed.
    """

    def __init__(self, message: str, *, partial_interaction: Any, continuation_token: str) -> None:
        super().__init__(message)
        self.partial_interaction = partial_interaction
        self.continuation_token = continuation_token


class FakeTier(enum.Enum):
    """Stands for an enumeration member as the value of a service tier.

    SDK 3.3.0 gives the tier as plain text, so it does not produce this case. The backend reads a
    member all the same, because the tier decides the price.
    """

    FLEX = "flex"
    STANDARD = "standard"


#: The fields of the two request models of the SDK (``CreateModelInteraction`` and
#: ``CreateAgentInteraction``, SDK 3.3.0, source read on 2026-11-14). The real ``create`` takes every
#: other keyword that is not one of its own options for a wrong name and raises TypeError. This
#: machine has no SDK, so the list is not checked by a test run: check a new name against the SDK
#: before you add it here.
CREATE_BODY_KEYS = frozenset(
    {
        "agent",
        "agent_config",
        "background",
        "cached_content",
        "continuation_token",
        "environment",
        "generation_config",
        "input",
        "labels",
        "model",
        "previous_interaction_id",
        "response_format",
        "response_mime_type",
        "response_modalities",
        "safety_settings",
        "service_tier",
        "store",
        "stream",
        "system_instruction",
        "tools",
        "webhook_config",
    }
)
#: The fields of the model ``GenerationConfig`` of the SDK. ``temperature``, ``top_p`` and ``seed``
#: are sampling settings, which the backend never sends. ``max_output_tokens`` is the field of a
#: limit on the output of one answer.
GENERATION_CONFIG_FIELDS = frozenset(
    {
        "image_config",
        "max_output_tokens",
        "seed",
        "speech_config",
        "stop_sequences",
        "temperature",
        "thinking_level",
        "thinking_summaries",
        "tool_choice",
        "top_p",
        "transcription_config",
        "video_config",
    }
)
#: The values of the service tier that the SDK knows.
SDK_SERVICE_TIERS = frozenset({"flex", "standard", "priority", "deferred"})
#: The fields of the two option models of the SDK client (``google.genai.types``). Both models refuse
#: a keyword that is not a field. ``timeout`` is a whole number of milliseconds, and ``attempts``
#: counts the first request.
HTTP_OPTIONS_FIELDS = frozenset(
    {
        "aiohttp_client",
        "api_version",
        "async_client_args",
        "base_url",
        "base_url_resource_scope",
        "client_args",
        "extra_body",
        "headers",
        "httpx_async_client",
        "httpx_client",
        "retry_options",
        "timeout",
    }
)
HTTP_RETRY_OPTIONS_FIELDS = frozenset(
    {"attempts", "exp_base", "http_status_codes", "initial_delay", "jitter", "max_delay"}
)


def check_create_body(body: dict[str, Any]) -> None:
    """Refuse the request fields that the real ``create`` or the API refuse.

    The real ``create`` raises TypeError for a keyword that is no field of the request models.
    Only ``model`` is a required field. The other checks stand for the API: the fields that the
    backend sends must have the documented form. The check of ``generation_config`` is stricter
    than the SDK, which passes an unknown key on to the API: here a key must be a field of the
    model of the SDK, so that a misspelled name shows in a test.
    """
    unknown = sorted(set(body) - CREATE_BODY_KEYS)
    if unknown:
        raise TypeError(
            f"create() got unexpected keyword argument(s): {', '.join(unknown)}. "
            "Use extra_body=... to send additional request body fields."
        )
    generation_config = body.get("generation_config", {})
    valid = {
        "model": isinstance(body.get("model"), str),
        "input": bool(body.get("input")) and all(item.get("type") in ("text", "image") for item in body["input"]),
        "generation_config": isinstance(generation_config, dict) and set(generation_config) <= GENERATION_CONFIG_FIELDS,
        "response_format": set(body.get("response_format", ())) == {"type", "mime_type", "schema"},
        "store": isinstance(body.get("store", False), bool),
        "service_tier": body.get("service_tier", "standard") in SDK_SERVICE_TIERS,
    }
    wrong = [name for name, is_valid in valid.items() if not is_valid]
    if wrong:
        raise ValueError(f"the arguments of the call are not valid: {', '.join(wrong)}")


def _needs_continuation(interaction: Any) -> bool:
    """Tell whether the SDK would continue this answer: it stopped at the token cap of one request and has a token."""
    token = getattr(interaction, "continuation_token", None)
    return getattr(interaction, "status", None) == "requires_continuation" and isinstance(token, str) and bool(token)


def _merged(hops: list[Any]) -> Any:
    """Merge the answers of the requests of one call as the SDK does: the last answer with the summed token counts."""
    merged = SimpleNamespace(**vars(hops[-1]))
    usages = [hop.usage for hop in hops if getattr(hop, "usage", None) is not None]
    if usages:
        names = {name for usage in usages for name in vars(usage)}
        merged.usage = SimpleNamespace(
            **{name: sum(getattr(usage, name, None) or 0 for usage in usages) for name in sorted(names)}
        )
    return merged


class FakeInteractions:
    """Stands for ``client.interactions``: ``create`` with the signature and the automatic continuation of the SDK.

    Every HTTP request takes the next prepared outcome: an answer is returned and an error is
    raised. ``calls`` keeps the arguments of every call of ``create``, and ``n_http_requests``
    counts the requests. One call is one request, unless the SDK continues the answer.
    """

    def __init__(self, outcomes: list[Any], fake_time: FakeTime | None = None) -> None:
        self._outcomes = list(outcomes)
        self._fake_time = fake_time
        self.calls: list[dict[str, Any]] = []
        self.call_times: list[float] = []
        self.n_http_requests = 0

    def create(
        self,
        *,
        request: Any = None,
        api_version: str | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_query: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float | None = None,
        automatic_continuation: bool | None = None,
        **body: Any,
    ) -> Any:
        if any(option is not None for option in (request, api_version, extra_headers, extra_query, extra_body)):
            raise NotImplementedError("the stand-in for create takes the request as body keywords only")
        check_create_body(body)
        # A number is the time budget of the whole call in seconds, over all its requests.
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0):
            raise ValueError("the arguments of the call are not valid: timeout")
        self.calls.append({**body, "timeout": timeout, "automatic_continuation": automatic_continuation})
        if self._fake_time is not None:
            self.call_times.append(self._fake_time.now)
        hops = [self._http_request()]
        # The SDK continues by default. Only the value False switches the continuation off.
        while automatic_continuation is not False and _needs_continuation(hops[-1]):
            try:
                hops.append(self._http_request())
            except Exception as error:
                raise ContinuationInterruptedError(
                    f"Automatic continuation was interrupted after {len(hops)} completed hop(s): {error}.",
                    partial_interaction=_merged(hops),
                    continuation_token=hops[-1].continuation_token,
                ) from error
        return _merged(hops) if len(hops) > 1 else hops[0]

    def _http_request(self) -> Any:
        """Play one HTTP request: take the next outcome, and return it or raise it."""
        self.n_http_requests += 1
        if self._fake_time is not None:
            self._fake_time.now += CALL_SECONDS
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    """Stands for the SDK client: it has ``.interactions.create`` and nothing else."""

    def __init__(self, interactions: Any) -> None:
        self.interactions = interactions


def _options_class(name: str, fields: frozenset[str]) -> type:
    """Build the stand-in for one option model of the SDK: it refuses a keyword that is not one of its fields."""

    class Options:
        def __init__(self, **arguments: Any) -> None:
            unknown = sorted(set(arguments) - fields)
            if unknown:
                # The model of the SDK raises a validation error of pydantic, which is a ValueError.
                raise ValueError(f"{name}: extra inputs are not permitted: {', '.join(unknown)}")
            for whole_number in ("timeout", "attempts"):
                value = arguments.get(whole_number)
                if value is not None and type(value) is not int:
                    raise ValueError(f"{name}: {whole_number} must be a whole number, got {value!r}")
            self.arguments = arguments

    Options.__name__ = name
    return Options


class FakeSdk:
    """Stands for the model SDK: the modules ``google``, ``google.genai`` and ``google.genai.types``.

    The modules are put into ``sys.modules`` for one test. ``client_arguments`` keeps the keyword
    arguments of every client that was built. An option object keeps its keyword arguments in
    ``arguments``, and it refuses a keyword that the option model of the SDK does not have.
    ``client_error`` is raised by the client constructor when it is given.
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        interactions: Any = None,
        client_error: Exception | None = None,
    ) -> None:
        self.client_arguments: list[dict[str, Any]] = []
        sdk = self

        def make_client(**arguments: Any) -> FakeClient:
            sdk.client_arguments.append(arguments)
            if client_error is not None:
                raise client_error
            return FakeClient(interactions)

        google = types.ModuleType("google")
        genai = types.ModuleType("google.genai")
        sdk_types = types.ModuleType("google.genai.types")
        sdk_types.HttpOptions = _options_class("HttpOptions", HTTP_OPTIONS_FIELDS)  # type: ignore[attr-defined]
        sdk_types.HttpRetryOptions = _options_class(  # type: ignore[attr-defined]
            "HttpRetryOptions", HTTP_RETRY_OPTIONS_FIELDS
        )
        genai.Client = make_client  # type: ignore[attr-defined]
        genai.types = sdk_types  # type: ignore[attr-defined]
        google.genai = genai  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "google", google)
        monkeypatch.setitem(sys.modules, "google.genai", genai)
        monkeypatch.setitem(sys.modules, "google.genai.types", sdk_types)


def make_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"people": {"type": "array", "items": {"type": "object"}}},
        "required": ["people"],
    }


def make_request(**overrides: Any) -> VLMRequest:
    fields: dict[str, Any] = {
        "model": MODEL,
        "thinking_level": "low",
        "parts": (
            TextPart(text="Find every person."),
            ImagePart(data=DIAGRAM_BYTES, mime_type="image/png", resolution="medium"),
            TextPart(text="Answer with JSON."),
            ImagePart(data=FRAME_BYTES, resolution="ultra_high"),
        ),
        "response_schema": make_schema(),
        "prompt_name": "people/v1",
        "meta": RequestMeta(clip_id="clip-a", step="people", frame=12),
        "service_tier": "flex",
        "context": {"frames": [12]},
    }
    fields.update(overrides)
    return VLMRequest(**fields)


def make_usage(**overrides: Any) -> SimpleNamespace:
    counters: dict[str, Any] = {
        "total_input_tokens": 1500,
        "total_output_tokens": 300,
        "total_thought_tokens": 100,
        "total_cached_tokens": 0,
        "total_tokens": 1900,
    }
    counters.update(overrides)
    return SimpleNamespace(**counters)


def make_interaction(**overrides: Any) -> SimpleNamespace:
    """Return a stand-in for an answer of the SDK, with the fields that the backend and the continuation read.

    An answer of the SDK has every field. A field that the provider left out is None, and the id is
    an empty text.
    """
    fields: dict[str, Any] = {
        "output_text": ANSWER,
        "status": "completed",
        "usage": make_usage(),
        "service_tier": None,
        "continuation_token": None,
        "id": "",
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def make_backend(outcomes: list[Any], fake_time: FakeTime, **options: Any) -> tuple[GeminiBackend, FakeInteractions]:
    interactions = FakeInteractions(outcomes, fake_time)
    backend = GeminiBackend(
        client=FakeClient(interactions),
        clock=fake_time.clock,
        sleep=fake_time.sleep,
        **options,
    )
    return backend, interactions


def expected_sdk_version() -> str | None:
    try:
        return importlib.metadata.version("google-genai")
    except importlib.metadata.PackageNotFoundError:
        return None


# --- The SDK import is lazy ------------------------------------------------------


def _imported_modules(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    return [node.module or ""]


def _imports_sdk(node: ast.Import | ast.ImportFrom) -> bool:
    return any(name.split(".")[0] == "google" for name in _imported_modules(node))


def test_sdk_is_imported_only_inside_a_function() -> None:
    tree = ast.parse(Path(client_module.__file__).read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    top_level = [node for node in imports if node in tree.body]
    lazy = [node for node in imports if node not in tree.body]
    assert top_level, "the module has imports at the top"
    assert not any(_imports_sdk(node) for node in top_level)
    assert any(_imports_sdk(node) for node in lazy)


# --- Hard rule 4 over the whole package -------------------------------------------

PACKAGE_DIR = Path(client_module.__file__).resolve().parents[1]
TOOLS_DIR = PACKAGE_DIR.parents[1] / "tools"
#: The one module that may import the model SDK, and the one script that may import a tool.
SDK_MODULE = Path(client_module.__file__).resolve()
TOOL_RUNNER = TOOLS_DIR / "run_tool.py"
#: Top-level name of the model SDK (``google-genai``).
SDK_NAMES = frozenset({"google"})
COMPUTED_NAME = "<a name computed at run time>"


def _tool_names() -> frozenset[str]:
    """Return the top-level module names of the external tools: RF-DETR and the packages of PnLCalib.

    The packages of PnLCalib are read from the source of ``tools/run_tool.py``, which is not imported.
    """
    tree = ast.parse(TOOL_RUNNER.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "PNLCALIB_PACKAGES" for target in node.targets
        ):
            return frozenset({"rfdetr", *ast.literal_eval(node.value)})
    raise AssertionError("tools/run_tool.py has no PNLCALIB_PACKAGES")


#: Functions and types that import a module by a name: an import through any of them is found,
#: also when the function is given another name or handed on without a call.
DYNAMIC_IMPORTS = frozenset(
    {
        "import_module",
        "__import__",
        "resolve_name",
        "find_spec",
        "spec_from_file_location",
        "run_module",
        "run_path",
        "ImportString",
    }
)


#: The two functions whose call with one text is read as an import of that text. A name that is
#: a plain variable is flagged only for these; the others only as an attribute (``runpy.run_path``)
#: or an import of the name, so a variable such as ``run_path`` is no import.
READ_IMPORTS = ("import_module", "__import__")


def _dynamic_name(node: ast.AST) -> str | None:
    """Return the name of a dynamic import function that a node names, or None."""
    if isinstance(node, ast.Attribute):
        name = node.attr
    elif isinstance(node, ast.Name) and node.id in READ_IMPORTS:
        name = node.id
    else:
        return None
    return name if name in DYNAMIC_IMPORTS else None


def _read_name(call: ast.Call) -> str | None:
    """Return the module name of a call of a function of ``READ_IMPORTS`` with one text argument, else None.

    A second argument (a package, globals) or a keyword can make the name relative, so such a
    call is not read.
    """
    if _dynamic_name(call.func) not in READ_IMPORTS or len(call.args) != 1 or call.keywords:
        return None
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str) and not first.value.startswith("."):
        return first.value
    return None


def imported_modules(source: str) -> list[tuple[int, str]]:
    """Return the line and the module name of every import of a source, at any depth.

    A call of ``importlib.import_module`` or ``__import__`` with one text argument that is not
    relative counts as an import of that name (``_read_name``). Every other use of a function of
    ``DYNAMIC_IMPORTS`` (another name for it, a computed or relative name, a reference without a
    call, a ``from importlib import import_module``) counts as an import of ``COMPUTED_NAME``. A
    relative import names a module of the same package and is left out.
    """
    nodes = list(ast.walk(ast.parse(source)))
    found = []
    read = set()  # the functions of calls whose module name is read from the source
    for node in nodes:
        if isinstance(node, ast.Call) and (name := _read_name(node)) is not None:
            found.append((node.lineno, name))
            read.add(id(node.func))
    for node in nodes:
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                found.append((node.lineno, node.module or ""))
            found += [(node.lineno, COMPUTED_NAME) for alias in node.names if alias.name in DYNAMIC_IMPORTS]
        elif _dynamic_name(node) is not None and id(node) not in read:
            found.append((node.lineno, COMPUTED_NAME))  # type: ignore[attr-defined]
    return found


def rule_4_breaks(label: str, source: str, *, may_import_sdk: bool, may_import_tools: bool = False) -> list[str]:
    """Return one line for each import of the source that hard rule 4 forbids.

    The tool runner (``may_import_tools``) imports the modules of a tool by computed names:
    for it, only an import of the model SDK breaks the rule.
    """
    tools = _tool_names()
    breaks = []
    for line, name in imported_modules(source):
        top = name.split(".")[0]
        if name == COMPUTED_NAME and not may_import_tools:
            breaks.append(f"{label}:{line}: imports a module whose name is computed, so no check can read it")
        elif top in SDK_NAMES and not may_import_sdk:
            breaks.append(f"{label}:{line}: imports the model SDK ({name})")
        elif top in tools and not may_import_tools:
            breaks.append(f"{label}:{line}: imports an external tool ({name})")
    return breaks


def test_only_the_client_imports_the_sdk_and_only_the_tool_runner_imports_a_tool() -> None:
    """Hard rule 4, read from the source of every module of the package and of every script of tools/."""
    sources = sorted(PACKAGE_DIR.rglob("*.py")) + sorted(TOOLS_DIR.glob("*.py"))
    assert len(sources) > 20 and SDK_MODULE in sources and TOOL_RUNNER in sources
    breaks = []
    for path in sources:
        source = path.read_text(encoding="utf-8")
        breaks += rule_4_breaks(
            path.name, source, may_import_sdk=path == SDK_MODULE, may_import_tools=path == TOOL_RUNNER
        )
    assert breaks == []
    assert any(name.split(".")[0] in SDK_NAMES for _, name in imported_modules(SDK_MODULE.read_text(encoding="utf-8")))


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # At the top of a module, guarded, as the fault F33 of the fault check of 2026-11-22 put it into pipeline.py.
        ("try:\n    import google.genai\nexcept ImportError:\n    pass\n", "imports the model SDK (google.genai)"),
        ("from google import genai\n", "imports the model SDK (google)"),
        ("import google.genai as sdk\n", "imports the model SDK (google.genai)"),
        ("def ask():\n    from google.genai import types\n    return types\n", "imports the model SDK (google.genai)"),
        ("import importlib\n\ndef load():\n    return importlib.import_module('google.genai')\n", "imports the model SDK (google.genai)"),
        ("class Detector:\n    def run(self):\n        import rfdetr\n", "imports an external tool (rfdetr)"),
        ("from model.cls_hrnet import get_cls_net\n", "imports an external tool (model.cls_hrnet)"),
        ("def load(name):\n    return __import__(name)\n", "imports a module whose name is computed"),
        # Found in the second read of 2026-11-22: other spellings of a dynamic import.
        ("from importlib import import_module as load\n", "imports a module whose name is computed"),
        ("import importlib as il\n\ndef sdk():\n    return il.import_module\n", "imports a module whose name is computed"),
        ("import pkgutil\n\ndef sdk():\n    return pkgutil.resolve_name('google.genai:Client')\n", "imports a module whose name is computed"),
        ("import importlib\n\ndef sdk():\n    return importlib.import_module('.genai', 'google')\n", "imports a module whose name is computed"),
        ("def sdk():\n    return __import__('genai', {'__package__': 'google'}, None, [], 1)\n", "imports a module whose name is computed"),
        ("import runpy\n\ndef sdk():\n    return runpy.run_module('google.genai')\n", "imports a module whose name is computed"),
    ],
)
def test_rule_4_check_finds_an_import_at_the_top_and_inside_a_function(source: str, expected: str) -> None:
    breaks = rule_4_breaks("pipeline.py", source, may_import_sdk=False)
    assert len(breaks) == 1 and expected in breaks[0], breaks


def test_rule_4_check_takes_a_variable_with_the_name_of_an_import_function_for_no_import() -> None:
    source = "def save(run_path, spec):\n    find_spec = spec.find\n    return run_path.read_text(), find_spec\n"
    assert rule_4_breaks("cli.py", source, may_import_sdk=False) == []


def test_rule_4_check_lets_the_tool_runner_import_a_tool_but_not_the_sdk() -> None:
    runner = "import importlib\n\ndef load(name):\n    return importlib.import_module(name)\n\nimport rfdetr\n"
    assert rule_4_breaks("run_tool.py", runner, may_import_sdk=False, may_import_tools=True) == []
    assert rule_4_breaks("run_tool.py", "import google.genai\n", may_import_sdk=False, may_import_tools=True) == [
        "run_tool.py:1: imports the model SDK (google.genai)"
    ]


def test_rule_4_check_lets_the_client_import_the_sdk_but_no_tool() -> None:
    assert rule_4_breaks("client.py", "def connect():\n    from google import genai\n", may_import_sdk=True) == []
    assert rule_4_breaks("client.py", "import rfdetr\n", may_import_sdk=True) == ["client.py:1: imports an external tool (rfdetr)"]
    assert rule_4_breaks("pipeline.py", "from . import synth\nimport numpy as np\nfrom pitchvlm.vlm import client\n", may_import_sdk=False) == []


# --- Request to keyword arguments ------------------------------------------------


def test_request_maps_to_the_keyword_arguments_of_create() -> None:
    kwargs = build_interaction_kwargs(make_request())
    assert kwargs == {
        "model": MODEL,
        "input": [
            {"type": "text", "text": "Find every person."},
            {
                "type": "image",
                "data": base64.b64encode(DIAGRAM_BYTES).decode("ascii"),
                "mime_type": "image/png",
                "resolution": "medium",
            },
            {"type": "text", "text": "Answer with JSON."},
            {
                "type": "image",
                "data": base64.b64encode(FRAME_BYTES).decode("ascii"),
                "mime_type": "image/jpeg",
                "resolution": "ultra_high",
            },
        ],
        "generation_config": {"thinking_level": "low"},
        "response_format": {"type": "text", "mime_type": "application/json", "schema": make_schema()},
        "store": False,
        "service_tier": "flex",
    }


def test_parts_keep_their_order() -> None:
    request = make_request(parts=(ImagePart(data=b"one"), TextPart(text="middle"), ImagePart(data=b"two")))
    items = build_interaction_kwargs(request)["input"]
    assert [item["type"] for item in items] == ["image", "text", "image"]
    assert base64.b64decode(items[0]["data"]) == b"one"
    assert items[1]["text"] == "middle"
    assert base64.b64decode(items[2]["data"]) == b"two"


def test_image_bytes_are_sent_as_base64_text_with_the_default_settings() -> None:
    request = make_request(parts=(ImagePart(data=b"abc"),))
    assert build_interaction_kwargs(request)["input"] == [
        {"type": "image", "data": "YWJj", "mime_type": "image/jpeg", "resolution": "high"}
    ]


def test_call_is_never_stored_and_has_no_sampling_parameter() -> None:
    kwargs = build_interaction_kwargs(make_request())
    assert kwargs["store"] is False
    assert set(kwargs) == {"model", "input", "generation_config", "response_format", "store", "service_tier"}
    assert set(kwargs["generation_config"]) == {"thinking_level"}


def test_thinking_level_and_service_tier_come_from_the_request() -> None:
    kwargs = build_interaction_kwargs(make_request(thinking_level="high", service_tier="standard"))
    assert kwargs["generation_config"] == {"thinking_level": "high"}
    assert kwargs["service_tier"] == "standard"


def test_fake_client_refuses_what_the_sdk_refuses() -> None:
    """The stand-in for ``create`` takes the arguments of a real request and refuses a name that the SDK refuses."""
    arguments = {**build_interaction_kwargs(make_request()), "timeout": 300.0, "automatic_continuation": False}
    interactions = FakeInteractions([make_interaction() for _ in range(4)])

    assert interactions.create(**arguments).output_text == ANSWER
    # The request models of the SDK have no field for a sampling setting at the top level.
    with pytest.raises(TypeError, match=r"unexpected keyword argument\(s\): temperature"):
        interactions.create(**arguments, temperature=0.2)
    # A key of the generation settings must be a field of the model of the SDK.
    with pytest.raises(ValueError, match="generation_config"):
        interactions.create(**{**arguments, "generation_config": {"thinking_level": "low", "top_k": 40}})
    with pytest.raises(ValueError, match="timeout"):
        interactions.create(**{**arguments, "timeout": 0})
    # A refused call is not recorded: only the first call reached the model.
    assert len(interactions.calls) == interactions.n_http_requests == 1
    # The SDK takes these forms too: a limit on the output tokens, a call without a service tier
    # (the field is not required) and a call without a timeout of its own.
    limited = {**arguments, "generation_config": {"thinking_level": "low", "max_output_tokens": 4096}}
    assert interactions.create(**limited).output_text == ANSWER
    without_tier = {name: value for name, value in arguments.items() if name not in ("service_tier", "timeout")}
    assert interactions.create(**without_tier).output_text == ANSWER
    assert interactions.calls[-1]["timeout"] is None


def test_fake_client_continues_an_answer_like_the_sdk_unless_the_call_switches_it_off() -> None:
    """The SDK sends the next request by itself when an answer stops at the token cap of one request."""
    arguments = {**build_interaction_kwargs(make_request()), "timeout": 300.0}

    def first_part() -> SimpleNamespace:
        return make_interaction(status="requires_continuation", continuation_token="token-1", output_text='{"peo')

    # By default one call makes two HTTP requests, and the token counts of both are added up.
    continued = FakeInteractions([first_part(), make_interaction()])
    answer = continued.create(**arguments)
    assert (answer.status, continued.n_http_requests, len(continued.calls)) == ("completed", 2, 1)
    assert (answer.usage.total_input_tokens, answer.usage.total_tokens) == (3000, 3800)

    # A second request that fails gives the error of the SDK: no status code, and the first answer inside.
    interrupted = FakeInteractions([first_part(), FakeApiError(503)])
    with pytest.raises(ContinuationInterruptedError) as excinfo:
        interrupted.create(**arguments)
    assert interrupted.n_http_requests == 2
    assert not hasattr(excinfo.value, "status_code")
    assert excinfo.value.partial_interaction.usage.total_tokens == 1900
    assert isinstance(excinfo.value.__cause__, FakeApiError)

    # With the continuation switched off, the first answer comes back as it is, after one HTTP request.
    switched_off = FakeInteractions([first_part(), make_interaction()])
    answer = switched_off.create(**arguments, automatic_continuation=False)
    assert (answer.status, switched_off.n_http_requests) == ("requires_continuation", 1)
    # An answer with that status and without a token is not continued.
    no_token = FakeInteractions([make_interaction(status="requires_continuation"), make_interaction()])
    assert no_token.create(**arguments).status == "requires_continuation"
    assert no_token.n_http_requests == 1


def test_fake_option_models_refuse_what_the_sdk_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """The option models of the SDK forbid a keyword that is not a field, and they take whole numbers."""
    FakeSdk(monkeypatch)
    sdk_types = sys.modules["google.genai.types"]

    retry = sdk_types.HttpRetryOptions(attempts=1)
    assert sdk_types.HttpOptions(timeout=900_000, retry_options=retry).arguments["timeout"] == 900_000
    with pytest.raises(ValueError, match="timeout_ms"):
        sdk_types.HttpOptions(timeout_ms=900_000)
    with pytest.raises(ValueError, match="max_attempts"):
        sdk_types.HttpRetryOptions(max_attempts=1)
    with pytest.raises(ValueError, match="timeout must be a whole number"):
        sdk_types.HttpOptions(timeout=900_000.5)


def test_schema_in_the_keyword_arguments_is_a_copy() -> None:
    request = make_request()
    kwargs = build_interaction_kwargs(request)
    kwargs["response_format"]["schema"]["properties"]["people"]["type"] = "changed"
    assert request.response_schema == make_schema()


def test_request_without_parts_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_interaction_kwargs(make_request(parts=()))


# --- Token usage -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens", "thought_tokens", "total_tokens", "expected"),
    [
        (1000, 200, 300, 1500, "separate"),
        (1000, 500, 300, 1500, "included"),
        (1000, 200, 300, 1510, "separate"),
        (1000, 200, 300, 1490, "separate"),
        (1000, 200, 300, 5000, "unknown"),
        (1000, 200, 300, None, "unknown"),
        (1000, 100, 300, 1100, "unknown"),
        (3000, 400, 10, 3400, "included"),
        (1000, 200, 0, 1200, "separate"),
    ],
)
def test_usage_accounting(
    input_tokens: int, output_tokens: int, thought_tokens: int, total_tokens: int | None, expected: str
) -> None:
    assert usage_accounting(input_tokens, output_tokens, thought_tokens, total_tokens) == expected


def test_output_is_kept_when_the_total_counts_thought_tokens_apart(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        usage = normalize_usage(
            input_tokens=1000, output_tokens=200, thought_tokens=300, cached_tokens=50, total_tokens=1500
        )
    assert usage == Usage(input_tokens=1000, output_tokens=200, thought_tokens=300, cached_tokens=50)
    assert caplog.records == []


def test_thought_tokens_are_taken_out_of_an_output_count_that_contains_them(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        usage = normalize_usage(
            input_tokens=1000, output_tokens=500, thought_tokens=300, cached_tokens=0, total_tokens=1500
        )
    assert usage == Usage(input_tokens=1000, output_tokens=200, thought_tokens=300, cached_tokens=0)
    assert caplog.records == []


@pytest.mark.parametrize("total_tokens", [5000, None])
def test_unclear_usage_is_kept_and_a_warning_is_logged(
    total_tokens: int | None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        usage = normalize_usage(
            input_tokens=1000, output_tokens=200, thought_tokens=300, cached_tokens=0, total_tokens=total_tokens
        )
    assert usage == Usage(input_tokens=1000, output_tokens=200, thought_tokens=300, cached_tokens=0)
    assert [(record.name, record.levelno) for record in caplog.records] == [(LOGGER_NAME, logging.WARNING)]


# --- Service tier that is billed -------------------------------------------------


@pytest.mark.parametrize(
    ("reported", "requested", "expected", "n_warnings"),
    [
        pytest.param(None, "flex", "flex", 0, id="no tier reported"),
        pytest.param("flex", "flex", "flex", 0, id="requested tier"),
        pytest.param("FLEX", "flex", "flex", 0, id="requested tier in upper case"),
        pytest.param(FakeTier.FLEX, "flex", "flex", 0, id="requested tier as an enumeration member"),
        pytest.param("standard", "flex", "standard", 1, id="standard instead of flex"),
        pytest.param("Standard", "flex", "standard", 1, id="other tier in mixed case"),
        pytest.param(FakeTier.STANDARD, "flex", "standard", 1, id="other tier as an enumeration member"),
        pytest.param("flex", "standard", "flex", 1, id="flex instead of standard"),
        pytest.param("priority", "flex", "priority", 1, id="priority instead of flex"),
        pytest.param("priority", "standard", "priority", 1, id="priority instead of standard"),
        pytest.param("Priority", "standard", "priority", 1, id="priority in mixed case"),
        # A tier without a price is billed like the most expensive tier, never like the cheaper requested one.
        pytest.param("deferred", "flex", "priority", 1, id="tier without a price instead of flex"),
        pytest.param("deferred", "standard", "priority", 1, id="tier without a price instead of standard"),
        pytest.param("", "standard", "priority", 1, id="empty tier text"),
    ],
)
def test_billed_tier_is_the_reported_tier_and_a_tier_without_a_price_counts_as_the_most_expensive_one(
    reported: Any, requested: ServiceTier, expected: str, n_warnings: int, caplog: pytest.LogCaptureFixture
) -> None:
    interaction = SimpleNamespace(service_tier=reported)
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        tier = billed_tier(interaction, requested, "test request", model="gemini-4-argon")
    assert tier == expected
    # The result is plain text, also when the SDK gave an enumeration member.
    assert type(tier) is str
    assert len(caplog.records) == n_warnings


def test_billed_tier_is_the_requested_tier_when_the_answer_has_no_tier_field(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert billed_tier(SimpleNamespace(), "flex", "test request", model="gemini-4-argon") == "flex"
    assert caplog.records == []


def test_reported_flex_tier_of_a_model_without_a_flex_tier_counts_as_the_most_expensive_tier(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The flex tier of such a model has no price, so the cost of the call could not be computed with it."""
    interaction = SimpleNamespace(service_tier="flex")
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        tier = billed_tier(interaction, "standard", "test request", model="gemini-robotics-er-2-preview")
    assert tier == "priority"
    assert "'flex', which has no price for this model" in caplog.records[0].getMessage()
    # The same answer of a model with a flex tier is billed at the flex price.
    assert billed_tier(interaction, "standard", "test request", model="gemini-3.8-flash") == "flex"


@pytest.mark.parametrize("reported", ["standard", "priority", "deferred"])
def test_tier_warning_names_the_request_and_both_tiers(reported: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        billed_tier(SimpleNamespace(service_tier=reported), "flex", "test request", model="gemini-4-argon")
    message = caplog.records[0].getMessage()
    assert "test request" in message
    assert "'flex'" in message
    assert f"'{reported}'" in message


def test_tier_warning_of_a_tier_without_a_price_names_the_tier_that_is_billed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        billed_tier(SimpleNamespace(service_tier="deferred"), "flex", "test request", model="gemini-4-argon")
    message = caplog.records[0].getMessage()
    assert "has no price" in message
    assert "most expensive tier, 'priority'" in message


# --- Retry helpers ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (408, True),
        (409, True),
        (429, True),
        (500, True),
        (503, True),
        (599, True),
        (200, False),
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (499, False),
        (600, False),
        (None, False),
    ],
)
def test_retryable_status_codes(status: int | None, expected: bool) -> None:
    assert is_retryable_status(status) is expected


def test_status_code_is_read_from_the_status_code_attribute() -> None:
    assert status_code_of(FakeApiError(429)) == 429
    assert status_code_of(RuntimeError("no code")) is None
    # The errors of a lost connection and of a client timeout have the attribute, with the value None.
    assert status_code_of(APIConnectionError()) is None
    assert status_code_of(APITimeoutError()) is None
    # A value that is not a whole number is no status code.
    assert status_code_of(FakeApiError("ECONNRESET")) is None  # type: ignore[arg-type]
    assert status_code_of(FakeApiError(True)) is None
    # The Interactions API never keeps its status in ``code``, so that attribute is not read.
    other_style = RuntimeError("older error style")
    other_style.code = 503  # type: ignore[attr-defined]
    assert status_code_of(other_style) is None


def test_transport_failure_is_known_by_the_class_names_of_the_sdk() -> None:
    class WrappedTimeout(APITimeoutError):
        """A subclass, as a later SDK release could raise."""

    class WrappedConnection(APIConnectionError):
        """A subclass, as a later SDK release could raise."""

    assert transport_failure_of(APIConnectionError()) == "connection"
    assert transport_failure_of(WrappedConnection()) == "connection"
    # The timeout error is a connection error too. It is a timeout first.
    assert transport_failure_of(APITimeoutError()) == "timeout"
    assert transport_failure_of(WrappedTimeout()) == "timeout"
    # An error with an answer of the server, a wrong argument and a failed validation are no network fault.
    for other in (FakeApiError(503), RuntimeError("boom"), TypeError("wrong keyword"), ValueError("wrong body")):
        assert transport_failure_of(other) is None
    # The built-in errors of Python are not the errors of the SDK.
    assert transport_failure_of(ConnectionError("reset")) is None
    assert transport_failure_of(TimeoutError("slow")) is None


def test_backoff_doubles_and_stays_between_half_and_full_nominal_wait() -> None:
    rng = random.Random(0)
    for retry_index in range(6):
        nominal = min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2**retry_index)
        for _ in range(50):
            assert nominal / 2 <= backoff_delay_s(retry_index, rng) <= nominal


def test_backoff_has_jitter_and_a_cap() -> None:
    rng = random.Random(1)
    assert len({backoff_delay_s(0, rng) for _ in range(20)}) > 1
    assert backoff_delay_s(5000, rng) <= BACKOFF_CAP_S


def test_backoff_starts_at_10_seconds_and_never_waits_longer_than_60_seconds() -> None:
    """The limits are written out here, so that a changed constant does not move the test with it."""
    rng = random.Random(0)
    assert (BACKOFF_BASE_S, BACKOFF_CAP_S) == (10.0, 60.0)
    assert all(5.0 <= backoff_delay_s(0, rng) <= 10.0 for _ in range(200))
    assert max(backoff_delay_s(retry_index, rng) for retry_index in range(40)) <= 60.0
    # The cap is reached: the longest waits are near 60 seconds, not near 10.
    assert max(backoff_delay_s(10, rng) for _ in range(200)) > 30.0


def test_backoff_rejects_a_negative_index() -> None:
    with pytest.raises(ValueError):
        backoff_delay_s(-1, random.Random(0))


# --- Spend-rate limiter ----------------------------------------------------------


def test_spend_limiter_does_not_wait_when_there_is_room() -> None:
    fake_time = FakeTime()
    limiter = SpendRateLimiter(1.0, clock=fake_time.clock, sleep=fake_time.sleep)
    limiter.record(0.4)
    limiter.wait_for_room()
    assert fake_time.sleeps == []
    assert limiter.spent_usd() == pytest.approx(0.4)


def test_spend_limiter_waits_until_the_oldest_cost_leaves_the_window() -> None:
    fake_time = FakeTime(start=1000.0)
    limiter = SpendRateLimiter(1.0, window_s=600.0, clock=fake_time.clock, sleep=fake_time.sleep)
    limiter.record(0.6)
    fake_time.now = 1100.0
    limiter.record(0.5)
    fake_time.now = 1200.0
    assert limiter.spent_usd() == pytest.approx(1.1)

    limiter.wait_for_room()

    # The call of 0.6 USD finished at 1000 s, so it leaves the window at 1600 s.
    assert fake_time.sleeps == [400.0]
    assert fake_time.now == 1600.0
    assert limiter.spent_usd() == pytest.approx(0.5)


def test_spend_limiter_waits_for_every_cost_that_must_leave() -> None:
    fake_time = FakeTime(start=1000.0)
    limiter = SpendRateLimiter(1.0, window_s=600.0, clock=fake_time.clock, sleep=fake_time.sleep)
    for finished_at in (1000.0, 1100.0, 1200.0):
        fake_time.now = finished_at
        limiter.record(0.5)
    fake_time.now = 1300.0

    limiter.wait_for_room()

    # Two of the three calls must leave: the second one leaves at 1100 + 600 = 1700 s.
    assert fake_time.sleeps == [400.0]
    assert limiter.spent_usd() == pytest.approx(0.5)


def test_spend_limiter_blocks_at_exactly_the_limit() -> None:
    fake_time = FakeTime(start=1000.0)
    limiter = SpendRateLimiter(1.0, window_s=600.0, clock=fake_time.clock, sleep=fake_time.sleep)
    limiter.record(1.0)
    limiter.wait_for_room()
    assert fake_time.sleeps == [600.0]
    assert limiter.spent_usd() == 0.0


def test_spend_limiter_forgets_costs_that_left_the_window() -> None:
    fake_time = FakeTime(start=1000.0)
    limiter = SpendRateLimiter(1.0, window_s=600.0, clock=fake_time.clock, sleep=fake_time.sleep)
    limiter.record(0.9)
    fake_time.now = 1599.0
    assert limiter.spent_usd() == pytest.approx(0.9)
    fake_time.now = 1600.0
    assert limiter.spent_usd() == 0.0


@pytest.mark.parametrize("wrong", [0.0, -1.0, float("nan")])
def test_spend_limiter_rejects_a_wrong_limit_or_window(wrong: float) -> None:
    with pytest.raises(ValueError):
        SpendRateLimiter(wrong)
    with pytest.raises(ValueError):
        SpendRateLimiter(1.0, window_s=wrong)


@pytest.mark.parametrize("wrong", [-0.1, float("nan")])
def test_spend_limiter_rejects_a_wrong_cost(wrong: float) -> None:
    with pytest.raises(ValueError):
        SpendRateLimiter(1.0).record(wrong)


def test_spend_limiter_is_safe_from_several_threads() -> None:
    limiter = SpendRateLimiter(1000.0)

    def record_many() -> None:
        for _ in range(200):
            limiter.record(0.01)

    threads = [threading.Thread(target=record_many) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert limiter.spent_usd() == pytest.approx(16.0)


# --- Backend: a good answer ------------------------------------------------------


def test_generate_returns_the_answer_of_the_client() -> None:
    fake_time = FakeTime()
    backend, interactions = make_backend([make_interaction()], fake_time, flex_timeout_s=720.0)
    request = make_request()

    before = datetime.now(UTC)
    response = backend.generate(request)
    after = datetime.now(UTC)

    assert response.text == ANSWER
    assert response.usage == Usage(input_tokens=1500, output_tokens=300, thought_tokens=100, cached_tokens=0)
    assert response.model == MODEL
    assert response.service_tier == "flex"
    assert response.latency_s == CALL_SECONDS
    assert response.attempts == 1
    assert response.sdk_version == expected_sdk_version()
    created = datetime.fromisoformat(response.created_at)
    assert created.utcoffset() == timedelta(0)
    assert before - timedelta(seconds=1) <= created <= after
    # The request asks for the flex tier, so the call gets the flex timeout. The call is one HTTP
    # request: the SDK must not continue an answer with further billed requests.
    assert interactions.calls == [
        {**build_interaction_kwargs(request), "timeout": 720.0, "automatic_continuation": False}
    ]
    assert interactions.n_http_requests == 1
    assert interactions.calls[0]["store"] is False
    assert fake_time.sleeps == []


def test_timeout_of_a_call_follows_the_service_tier_of_its_request() -> None:
    """A flex request can wait in a queue of the provider, so it gets 15 minutes. A standard request gets 5."""
    outcomes = [make_interaction(), make_interaction()]
    backend, interactions = make_backend(outcomes, FakeTime())

    backend.generate(make_request(service_tier="flex"))
    backend.generate(make_request(service_tier="standard"))

    assert [call["timeout"] for call in interactions.calls] == [900.0, 300.0]
    assert (STANDARD_TIMEOUT_S, FLEX_TIMEOUT_S) == (300.0, 900.0)
    # The provider recommends a client timeout of at least 10 minutes for the flex tier.
    assert FLEX_TIMEOUT_S >= 600.0


def test_timeouts_of_both_service_tiers_can_be_set() -> None:
    outcomes = [make_interaction(), make_interaction()]
    backend, interactions = make_backend(outcomes, FakeTime(), timeout_s=120.0, flex_timeout_s=1200.0)

    backend.generate(make_request(service_tier="standard"))
    backend.generate(make_request(service_tier="flex"))

    assert [call["timeout"] for call in interactions.calls] == [120.0, 1200.0]


def test_thought_tokens_inside_the_output_count_are_split_off() -> None:
    usage = make_usage(total_output_tokens=400, total_thought_tokens=100, total_tokens=1900)
    backend, _ = make_backend([make_interaction(usage=usage)], FakeTime())
    response = backend.generate(make_request())
    assert response.usage == Usage(input_tokens=1500, output_tokens=300, thought_tokens=100, cached_tokens=0)


def test_cached_tokens_of_the_answer_are_kept() -> None:
    backend, _ = make_backend([make_interaction(usage=make_usage(total_cached_tokens=40))], FakeTime())
    response = backend.generate(make_request())
    assert response.usage == Usage(input_tokens=1500, output_tokens=300, thought_tokens=100, cached_tokens=40)


def test_counters_that_the_api_leaves_out_count_as_zero(caplog: pytest.LogCaptureFixture) -> None:
    usage = SimpleNamespace(total_input_tokens=1500, total_output_tokens=300, total_tokens=1800)
    backend, _ = make_backend([make_interaction(usage=usage)], FakeTime())
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        response = backend.generate(make_request())
    assert response.usage == Usage(input_tokens=1500, output_tokens=300, thought_tokens=0, cached_tokens=0)
    assert caplog.records == []


def test_other_served_tier_is_logged_and_used_in_the_response(caplog: pytest.LogCaptureFixture) -> None:
    backend, _ = make_backend([make_interaction(service_tier="standard")], FakeTime())
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        response = backend.generate(make_request(service_tier="flex"))
    # The cost ledger prices the tier of the response, so it must be the tier that served the call.
    assert response.service_tier == "standard"
    assert response.text == ANSWER
    assert len(caplog.records) == 1
    assert "standard" in caplog.records[0].getMessage()


@pytest.mark.parametrize("served_tier", ["priority", "deferred"])
def test_served_tier_above_the_requested_one_is_logged_and_billed_at_the_priority_price(
    served_tier: str, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """The priority tier costs 1.8 times the standard price. A tier without a price is billed like it."""
    backend, _ = make_backend([make_interaction(service_tier=served_tier)], FakeTime())
    request = make_request(service_tier="flex")
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        response = backend.generate(request)
    assert response.service_tier == "priority"
    assert len(caplog.records) == 1
    assert served_tier in caplog.records[0].getMessage()
    # The cost ledger prices the response: 1.8 times the standard price, not the flex price of the request.
    answered_on = datetime.fromisoformat(response.created_at).date()
    usd = Ledger(tmp_path / "ledger.jsonl").record("run-1", request, response)
    assert usd == pytest.approx(1.8 * cost_usd(MODEL, DEFAULT_USAGE, "standard", answered_on))
    assert usd == pytest.approx(3.6 * cost_usd(MODEL, DEFAULT_USAGE, "flex", answered_on))


def test_generate_works_from_several_threads() -> None:
    lock = threading.Lock()
    calls: list[dict[str, Any]] = []
    responses: list[Any] = []
    interaction = make_interaction()

    class SharedInteractions:
        def create(self, **kwargs: Any) -> Any:
            with lock:
                calls.append(kwargs)
            return interaction

    backend = GeminiBackend(client=FakeClient(SharedInteractions()))

    def ask() -> None:
        response = backend.generate(make_request())
        with lock:
            responses.append(response)

    threads = [threading.Thread(target=ask) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(calls) == 8
    assert [response.attempts for response in responses] == [1] * 8


# --- Backend: bad answers --------------------------------------------------------


@pytest.mark.parametrize(
    "interaction",
    [
        make_interaction(status="incomplete"),
        make_interaction(status="incomplete", output_text=""),
        make_interaction(status="failed"),
        make_interaction(status="requires_continuation"),
        make_interaction(status="budget_exceeded"),
        make_interaction(status="queued"),
        # The SDK builds an answer that fails its validation without the checks: the status can then be None.
        make_interaction(status=None),
        make_interaction(output_text=""),
        make_interaction(output_text="   "),
        # For the same reason the text can be None, although a valid answer always has a text.
        make_interaction(output_text=None),
        SimpleNamespace(status="completed", usage=make_usage()),
        make_interaction(usage=None),
        make_interaction(usage=make_usage(total_input_tokens=None)),
        make_interaction(usage=make_usage(total_output_tokens="300")),
        make_interaction(usage=make_usage(total_output_tokens=-300)),
        make_interaction(usage=make_usage(total_output_tokens=0, total_thought_tokens=0, total_tokens=1500)),
    ],
    ids=[
        "status incomplete",
        "status incomplete with an empty text",
        "status failed",
        "status requires_continuation",
        "status budget_exceeded",
        "status queued",
        "no status",
        "empty text",
        "blank text",
        "text is None",
        "no text field",
        "no usage",
        "no input count",
        "count is not a number",
        "count is negative",
        "text without output tokens",
    ],
)
def test_bad_answer_raises_backend_error_and_is_not_retried(interaction: SimpleNamespace) -> None:
    fake_time = FakeTime()
    backend, interactions = make_backend([interaction, make_interaction()], fake_time)
    with pytest.raises(BackendError):
        backend.generate(make_request())
    assert len(interactions.calls) == 1
    assert fake_time.sleeps == []


def test_error_of_an_incomplete_answer_names_the_status() -> None:
    backend, _ = make_backend([make_interaction(status="incomplete")], FakeTime())
    with pytest.raises(BackendError, match="incomplete"):
        backend.generate(make_request())


def test_error_of_a_negative_token_count_names_the_counter() -> None:
    backend, _ = make_backend([make_interaction(usage=make_usage(total_output_tokens=-300))], FakeTime())
    with pytest.raises(BackendError, match="total_output_tokens is not a token count: -300") as excinfo:
        backend.generate(make_request())
    # The cost of such an answer is unknown, so the error carries no cost record.
    assert not isinstance(excinfo.value, BilledAnswerError)


@pytest.mark.parametrize(
    ("interaction", "usage"),
    [
        pytest.param(make_interaction(status="incomplete"), DEFAULT_USAGE, id="status incomplete"),
        pytest.param(
            make_interaction(status="incomplete", output_text=""), DEFAULT_USAGE, id="answer that was cut"
        ),
        pytest.param(make_interaction(status="requires_continuation"), DEFAULT_USAGE, id="requires_continuation"),
        pytest.param(make_interaction(output_text=None), DEFAULT_USAGE, id="no text"),
        pytest.param(
            make_interaction(usage=make_usage(total_output_tokens=0, total_thought_tokens=0, total_tokens=1500)),
            Usage(input_tokens=1500, output_tokens=0),
            id="text without output tokens",
        ),
    ],
)
def test_rejected_answer_with_a_token_usage_carries_its_cost_record(
    interaction: SimpleNamespace, usage: Usage
) -> None:
    backend, _ = make_backend([interaction], FakeTime())

    before = datetime.now(UTC)
    with pytest.raises(BilledAnswerError) as excinfo:
        backend.generate(make_request())
    after = datetime.now(UTC)

    billed = excinfo.value.billed
    assert billed.text == ""
    assert billed.usage == usage
    assert billed.model == MODEL
    assert billed.service_tier == "flex"
    assert billed.latency_s == CALL_SECONDS
    assert billed.attempts == 1
    assert billed.sdk_version == expected_sdk_version()
    created = datetime.fromisoformat(billed.created_at)
    assert created.utcoffset() == timedelta(0)
    assert before - timedelta(seconds=1) <= created <= after


@pytest.mark.parametrize(
    "interaction",
    [
        make_interaction(usage=None),
        make_interaction(usage=make_usage(total_input_tokens=None)),
        make_interaction(usage=make_usage(total_output_tokens="300")),
    ],
    ids=["no usage", "no input count", "count is not a number"],
)
def test_answer_without_a_readable_token_usage_carries_no_cost_record(interaction: SimpleNamespace) -> None:
    backend, _ = make_backend([interaction], FakeTime())
    with pytest.raises(BackendError) as excinfo:
        backend.generate(make_request())
    # The cost of such an answer is unknown, so the error cannot say what was billed.
    assert not isinstance(excinfo.value, BilledAnswerError)


def test_cost_record_of_a_rejected_answer_counts_every_attempt() -> None:
    fake_time = FakeTime()
    backend, _ = make_backend([FakeApiError(429), make_interaction(status="incomplete")], fake_time)
    with pytest.raises(BilledAnswerError) as excinfo:
        backend.generate(make_request())
    assert excinfo.value.billed.attempts == 2
    assert excinfo.value.billed.latency_s == pytest.approx(CALL_SECONDS)


def test_cost_record_of_a_rejected_answer_has_the_served_tier() -> None:
    interaction = make_interaction(status="incomplete", service_tier="standard")
    backend, _ = make_backend([interaction], FakeTime())
    with pytest.raises(BilledAnswerError) as excinfo:
        backend.generate(make_request(service_tier="flex"))
    assert excinfo.value.billed.service_tier == "standard"


def test_cost_record_of_a_rejected_answer_fits_the_cost_ledger(tmp_path: Path) -> None:
    backend, _ = make_backend([make_interaction(status="incomplete")], FakeTime())
    request = make_request()
    with pytest.raises(BilledAnswerError) as excinfo:
        backend.generate(request)
    billed = excinfo.value.billed
    ledger = Ledger(tmp_path / "ledger.jsonl")

    usd = ledger.record("run-1", request, billed)

    # The ledger prices the billed tokens at the flex price of the day of the answer.
    answered_on = datetime.fromisoformat(billed.created_at).date()
    assert usd == pytest.approx(cost_usd(MODEL, DEFAULT_USAGE, "flex", answered_on))
    assert usd > 0.0
    assert ledger.run_usd("run-1") == pytest.approx(usd)


# --- Backend: retries ------------------------------------------------------------


def test_rate_limit_error_is_retried_and_the_attempts_are_counted() -> None:
    fake_time = FakeTime()
    backend, interactions = make_backend([FakeApiError(429), make_interaction()], fake_time)

    response = backend.generate(make_request())

    assert response.attempts == 2
    assert len(interactions.calls) == 2
    assert len(fake_time.sleeps) == 1
    assert BACKOFF_BASE_S / 2 <= fake_time.sleeps[0] <= BACKOFF_BASE_S
    # The latency is the time of the successful attempt only.
    assert response.latency_s == pytest.approx(CALL_SECONDS)


def test_lost_connection_is_retried_and_the_attempts_are_counted(caplog: pytest.LogCaptureFixture) -> None:
    fake_time = FakeTime()
    outcomes = [APIConnectionError("connection reset"), make_interaction()]
    backend, interactions = make_backend(outcomes, fake_time)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        response = backend.generate(make_request())

    # The error has no status code. It is tried again like a server error, with the same wait.
    assert response.text == ANSWER
    assert response.attempts == 2
    assert len(interactions.calls) == 2
    assert len(fake_time.sleeps) == 1
    assert BACKOFF_BASE_S / 2 <= fake_time.sleeps[0] <= BACKOFF_BASE_S
    # The log line of the lost attempt says that it can still be billed.
    (record,) = caplog.records
    assert "attempt 1 of 4 failed (no answer from the server, the connection failed)" in record.getMessage()
    assert "connection reset" in record.getMessage()
    assert "may still be billed" in record.getMessage()


def test_lost_connection_on_every_attempt_raises_an_unconfirmed_call_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_time = FakeTime()
    outcomes = [APIConnectionError("connection reset") for _ in range(6)]
    backend, interactions = make_backend(outcomes, fake_time)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME), pytest.raises(UnconfirmedCallError) as excinfo:
        backend.generate(make_request())

    assert len(interactions.calls) == 4
    assert len(fake_time.sleeps) == 3
    message = str(excinfo.value)
    assert "the call failed on attempt 4 of 4 (no answer from the server, the connection failed)" in message
    assert "may still be billed" in message
    assert "the cost ledger has no line for it" in message
    assert isinstance(excinfo.value, BackendError)
    assert isinstance(excinfo.value.__cause__, APIConnectionError)
    # Three attempts were tried again, and the last failure is in the log too.
    assert len(caplog.records) == 4
    assert caplog.records[-1].getMessage() == message


@pytest.mark.parametrize(("service_tier", "seconds"), [("flex", "900"), ("standard", "300")])
def test_client_timeout_is_not_retried_and_raises_an_unconfirmed_call_error(
    service_tier: ServiceTier, seconds: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The provider had the request for the whole timeout, so it can still bill it. A new attempt could pay twice."""
    fake_time = FakeTime()
    backend, interactions = make_backend([APITimeoutError(), make_interaction()], fake_time)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME), pytest.raises(UnconfirmedCallError) as excinfo:
        backend.generate(make_request(service_tier=service_tier))

    assert len(interactions.calls) == 1
    assert fake_time.sleeps == []
    message = str(excinfo.value)
    assert f"the call failed on attempt 1 of 4 (no answer within the client timeout of {seconds} s)" in message
    assert "APITimeoutError: Request timed out." in message
    assert "may still be billed" in message
    assert isinstance(excinfo.value.__cause__, APITimeoutError)
    # The failure is in the log, also when a run raises the error of another request first.
    assert [record.getMessage() for record in caplog.records] == [message]


def test_error_with_a_status_code_is_never_an_unconfirmed_call() -> None:
    """The server answered with an error status, so the call is not billed."""
    backend, _ = make_backend([FakeApiError(400), make_interaction()], FakeTime())
    with pytest.raises(BackendError) as excinfo:
        backend.generate(make_request())
    assert not isinstance(excinfo.value, UnconfirmedCallError)
    assert "billed" not in str(excinfo.value)


def test_server_errors_are_retried_until_the_attempts_run_out() -> None:
    fake_time = FakeTime()
    outcomes = [FakeApiError(503), FakeApiError(408), FakeApiError(502), make_interaction()]
    backend, interactions = make_backend(outcomes, fake_time, max_attempts=3)

    with pytest.raises(BackendError) as excinfo:
        backend.generate(make_request(service_tier="standard"))

    assert len(interactions.calls) == 3
    assert len(fake_time.sleeps) == 2
    assert BACKOFF_BASE_S / 2 <= fake_time.sleeps[0] <= BACKOFF_BASE_S
    assert BACKOFF_BASE_S <= fake_time.sleeps[1] <= 2 * BACKOFF_BASE_S
    assert "502" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, FakeApiError)


def test_client_error_is_not_retried() -> None:
    fake_time = FakeTime()
    backend, interactions = make_backend([FakeApiError(400), make_interaction()], fake_time)

    with pytest.raises(BackendError) as excinfo:
        backend.generate(make_request())

    assert len(interactions.calls) == 1
    assert fake_time.sleeps == []
    assert "400" in str(excinfo.value)


@pytest.mark.parametrize(
    "error",
    [RuntimeError("boom"), TypeError("create() got an unexpected keyword argument"), ValueError("wrong body")],
    ids=["other error", "wrong keyword", "failed validation"],
)
def test_error_that_is_no_status_and_no_network_fault_is_not_retried(error: Exception) -> None:
    fake_time = FakeTime()
    backend, interactions = make_backend([error, make_interaction()], fake_time)
    with pytest.raises(BackendError, match=r"attempt 1 of 4 \(status None\)") as excinfo:
        backend.generate(make_request())
    # A fault of the code is final, and it says nothing about a bill.
    assert not isinstance(excinfo.value, UnconfirmedCallError)
    assert excinfo.value.__cause__ is error
    assert len(interactions.calls) == 1
    assert fake_time.sleeps == []


def test_keyword_that_the_sdk_does_not_know_is_a_final_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stand-in for ``create`` refuses an unknown keyword like the SDK, so a wrong name in the backend shows."""
    right_arguments = build_interaction_kwargs
    monkeypatch.setattr(
        client_module, "build_interaction_kwargs", lambda request: {**right_arguments(request), "max_tokens": 100}
    )
    fake_time = FakeTime()
    backend, interactions = make_backend([make_interaction()], fake_time)

    with pytest.raises(BackendError, match="TypeError.*max_tokens") as excinfo:
        backend.generate(make_request())

    assert not isinstance(excinfo.value, UnconfirmedCallError)
    assert interactions.calls == []
    assert fake_time.sleeps == []


def test_one_attempt_means_no_retry() -> None:
    fake_time = FakeTime()
    backend, interactions = make_backend([FakeApiError(429), make_interaction()], fake_time, max_attempts=1)
    with pytest.raises(BackendError):
        backend.generate(make_request(service_tier="standard"))
    assert len(interactions.calls) == 1
    assert fake_time.sleeps == []


@pytest.mark.parametrize("status", [503, 429])
def test_four_attempts_are_the_default(status: int) -> None:
    """A request for the standard tier has no extra patience with the status codes of a busy flex tier."""
    fake_time = FakeTime()
    backend, interactions = make_backend([FakeApiError(status) for _ in range(6)], fake_time)
    with pytest.raises(BackendError, match=f"attempt 4 of 4 \\(status {status}\\)") as excinfo:
        backend.generate(make_request(service_tier="standard"))
    assert not isinstance(excinfo.value, FlexBusyError)
    assert len(interactions.calls) == 4
    assert len(fake_time.sleeps) == 3


def test_long_error_text_of_the_sdk_is_cut() -> None:
    backend, _ = make_backend([FakeApiError(400, "x" * 20_000)], FakeTime())
    with pytest.raises(BackendError) as excinfo:
        backend.generate(make_request())
    message = str(excinfo.value)
    # 500 characters of the SDK text are kept. The rest of the message is the place of the request.
    assert message.count("x") < 520
    assert message.endswith("x ...")
    assert len(message) < 1000


# --- Backend: a busy flex tier ---------------------------------------------------


def test_status_codes_and_waiting_time_of_a_busy_flex_tier() -> None:
    """The provider documents 503 and 429 as the answers of a flex tier without capacity."""
    assert FLEX_BUSY_STATUS_CODES == frozenset({429, 503})
    # The latency target of the flex tier is 1 to 15 minutes.
    assert FLEX_BUSY_WAIT_S == 900.0


@pytest.mark.parametrize("status", [503, 429])
def test_flex_request_waits_for_capacity_longer_than_four_attempts(
    status: int, caplog: pytest.LogCaptureFixture
) -> None:
    """A minute of busy answers must not end a flex request: the provider asks the client to try again."""
    fake_time = FakeTime()
    outcomes = [*[FakeApiError(status) for _ in range(10)], make_interaction()]
    backend, interactions = make_backend(outcomes, fake_time)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        response = backend.generate(make_request(service_tier="flex"))

    assert response.text == ANSWER
    assert response.attempts == 11
    assert len(interactions.calls) == 11
    # The wait grows to its cap and stays there: no wait is longer than 60 seconds.
    assert len(fake_time.sleeps) == 10
    assert 5.0 <= fake_time.sleeps[0] <= 10.0
    assert all(30.0 <= wait <= 60.0 for wait in fake_time.sleeps[3:])
    # Every wait is in the log, and its line says that the attempt is not billed.
    assert len(caplog.records) == 10
    first = caplog.records[0].getMessage()
    assert f"attempt 1 failed (status {status})" in first
    assert "The flex tier has no capacity" in first
    assert "it is not billed and does not use up one of the 4 attempts" in first
    assert "waits for capacity until 900 s after its first attempt" in first


@pytest.mark.parametrize("status", [503, 429])
def test_flex_request_ends_with_a_flex_busy_error_when_the_waiting_time_is_used_up(status: int) -> None:
    fake_time = FakeTime(start=1000.0)
    backend, interactions = make_backend([FakeApiError(status) for _ in range(200)], fake_time)

    with pytest.raises(FlexBusyError) as excinfo:
        backend.generate(make_request(service_tier="flex"))

    # The request was tried again for 900 seconds after its first attempt, and not for much longer:
    # at most one more wait of 60 seconds and one more attempt.
    waited_s = fake_time.now - 1000.0
    assert 900.0 <= waited_s <= 900.0 + 60.0 + CALL_SECONDS
    assert 15 <= len(interactions.calls) <= 40
    assert len(fake_time.sleeps) == len(interactions.calls) - 1
    message = str(excinfo.value)
    assert "the flex tier had no capacity for" in message
    assert "(the request waits 900 s for capacity)" in message
    assert f"all {len(interactions.calls)} attempts that it answered in this way, so they are not billed" in message
    assert f"(status {status})" in message
    # The error is a backend error, and it says nothing about a bill: the provider refused every attempt.
    assert isinstance(excinfo.value, BackendError)
    assert not isinstance(excinfo.value, UnconfirmedCallError)
    assert "may still be billed" not in message
    assert isinstance(excinfo.value.__cause__, FakeApiError)


def test_waiting_time_for_flex_capacity_can_be_set() -> None:
    fake_time = FakeTime(start=1000.0)
    backend, interactions = make_backend([FakeApiError(503) for _ in range(50)], fake_time, flex_busy_wait_s=30.0)

    with pytest.raises(FlexBusyError, match=r"\(the request waits 30 s for capacity\)"):
        backend.generate(make_request(service_tier="flex"))

    assert 30.0 <= fake_time.now - 1000.0 <= 30.0 + 40.0 + CALL_SECONDS
    assert 2 <= len(interactions.calls) <= 5


def test_busy_answers_of_the_flex_tier_do_not_use_up_the_attempts_of_a_request() -> None:
    """Two busy answers, then four server errors: the request still has its four attempts for these."""
    fake_time = FakeTime()
    outcomes = [FakeApiError(503), FakeApiError(429), *[FakeApiError(500) for _ in range(4)], make_interaction()]
    backend, interactions = make_backend(outcomes, fake_time)

    with pytest.raises(BackendError, match=r"the call failed on attempt 6 of 6 \(status 500\)") as excinfo:
        backend.generate(make_request(service_tier="flex"))

    assert not isinstance(excinfo.value, FlexBusyError)
    assert len(interactions.calls) == 6
    assert len(fake_time.sleeps) == 5


def test_other_failures_of_a_flex_request_keep_the_attempt_limit() -> None:
    """Only 503 and 429 are answers of a busy flex tier. A lost connection can be billed, so it has four attempts."""
    fake_time = FakeTime()
    backend, interactions = make_backend([APIConnectionError() for _ in range(6)], fake_time)
    with pytest.raises(UnconfirmedCallError, match="attempt 4 of 4"):
        backend.generate(make_request(service_tier="flex"))
    assert len(interactions.calls) == 4

    backend, interactions = make_backend([FakeApiError(500) for _ in range(6)], FakeTime())
    with pytest.raises(BackendError, match=r"attempt 4 of 4 \(status 500\)"):
        backend.generate(make_request(service_tier="flex"))
    assert len(interactions.calls) == 4


# --- Backend: one HTTP request per call ------------------------------------------


def test_answer_that_stops_at_the_token_cap_of_one_request_is_one_billed_request() -> None:
    """The SDK would send a second request by itself. When that one fails, the bill of the first one would be lost."""
    fake_time = FakeTime()
    # 32,768 tokens of thinking and output: the cap of one request without streaming.
    cap = make_usage(total_output_tokens=30_000, total_thought_tokens=2_768, total_tokens=34_268)
    first_part = make_interaction(
        status="requires_continuation", continuation_token="token-1", output_text='{"people": [', usage=cap
    )
    backend, interactions = make_backend([first_part, FakeApiError(503), make_interaction()], fake_time)

    with pytest.raises(BilledAnswerError, match="status 'requires_continuation'") as excinfo:
        backend.generate(make_request(service_tier="standard"))

    # One call and one HTTP request: the answer came back with its own usage, and nothing was sent after it.
    assert len(interactions.calls) == interactions.n_http_requests == 1
    assert fake_time.sleeps == []
    billed = excinfo.value.billed
    assert billed.usage == Usage(input_tokens=1500, output_tokens=30_000, thought_tokens=2_768)
    assert billed.attempts == 1
    # The cost of the answer is in the spend window, as for every billed answer.
    answered_on = datetime.fromisoformat(billed.created_at).date()
    assert backend._limiter.spent_usd() == pytest.approx(cost_usd(MODEL, billed.usage, "standard", answered_on))


def test_every_attempt_switches_the_automatic_continuation_of_the_sdk_off() -> None:
    backend, interactions = make_backend([FakeApiError(500), make_interaction()], FakeTime())

    backend.generate(make_request())

    assert [call["automatic_continuation"] for call in interactions.calls] == [False, False]


# --- Backend: an answer that the SDK cannot read ---------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [(200, True), (204, True), (299, True), (199, False), (300, False), (503, False), (None, False)],
)
def test_unread_answer_is_an_error_with_a_success_status(status: int | None, expected: bool) -> None:
    assert is_unread_answer(status) is expected


def test_answer_that_the_sdk_cannot_read_raises_an_unconfirmed_call_error(caplog: pytest.LogCaptureFixture) -> None:
    """The provider answered with a success status, so the request was done and can be billed. Its usage is lost."""
    fake_time = FakeTime()
    backend, interactions = make_backend([APIResponseValidationError(200), make_interaction()], fake_time)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME), pytest.raises(UnconfirmedCallError) as excinfo:
        backend.generate(make_request())

    # The call is not tried again: a new attempt could pay a second time.
    assert len(interactions.calls) == 1
    assert fake_time.sleeps == []
    message = str(excinfo.value)
    assert "the call failed on attempt 1 of 4 (status 200, but the answer could not be read)" in message
    assert "APIResponseValidationError: Data returned by API invalid for expected schema." in message
    assert "may still be billed" in message
    assert "the cost ledger has no line for it" in message
    assert [record.getMessage() for record in caplog.records] == [message]


# --- Backend: a stop request -----------------------------------------------------


def test_request_is_not_sent_after_a_stop_request(caplog: pytest.LogCaptureFixture) -> None:
    stop = threading.Event()
    stop.set()
    fake_time = FakeTime()
    backend, interactions = make_backend([make_interaction()], fake_time, stop=stop)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME), pytest.raises(BackendError) as excinfo:
        backend.generate(make_request())

    assert interactions.calls == []
    assert fake_time.sleeps == []
    message = str(excinfo.value)
    assert message.endswith("a stop was requested before this request was sent, so it was not sent")
    assert not isinstance(excinfo.value, UnconfirmedCallError)
    # The refusal is in the log: after an interruption nobody reads the error of a request in the thread pool.
    assert [record.getMessage() for record in caplog.records] == [message]


@pytest.mark.parametrize("failure", [FakeApiError(429), FakeApiError(503), APIConnectionError()])
def test_no_further_attempt_is_sent_after_a_stop_request(failure: Exception) -> None:
    """The stop request comes while the request waits for its second attempt."""
    stop = threading.Event()
    fake_time = FakeTime()

    def sleep_until_the_stop(seconds: float) -> None:
        fake_time.sleep(seconds)
        stop.set()

    interactions = FakeInteractions([failure, make_interaction()], fake_time)
    backend = GeminiBackend(
        client=FakeClient(interactions), clock=fake_time.clock, sleep=sleep_until_the_stop, stop=stop
    )

    with pytest.raises(BackendError, match="a stop was requested after attempt 1 failed, so no further attempt"):
        backend.generate(make_request())

    assert len(interactions.calls) == 1
    assert len(fake_time.sleeps) == 1


def test_wait_for_a_new_attempt_is_a_wait_on_the_stop_event() -> None:
    """Without a sleep function of a test, a stop request ends the wait before a new attempt at once."""
    waits: list[float | None] = []

    class StopDuringTheWait(threading.Event):
        """A stop event that keeps the length of every wait on it, and that is set during the first wait."""

        def wait(self, timeout: float | None = None) -> bool:
            waits.append(timeout)
            self.set()
            return True

    interactions = FakeInteractions([FakeApiError(429), make_interaction()])
    backend = GeminiBackend(client=FakeClient(interactions), stop=StopDuringTheWait())

    with pytest.raises(BackendError, match="a stop was requested after attempt 1 failed"):
        backend.generate(make_request())

    # The backend waited on the event for the 5 to 10 seconds of the first retry, and it sent no second attempt.
    assert len(waits) == 1
    assert waits[0] is not None and 5.0 <= waits[0] <= 10.0
    assert len(interactions.calls) == 1


def test_request_that_waited_for_room_in_the_spend_window_is_not_sent_after_a_stop_request() -> None:
    """The stop request comes while the request waits for room in the spend window."""
    stop = threading.Event()
    fake_time = FakeTime(start=1000.0)

    def sleep_until_the_stop(seconds: float) -> None:
        fake_time.sleep(seconds)
        stop.set()

    interactions = FakeInteractions([make_interaction(), make_interaction()], fake_time)
    # The limit is far below the cost of one call, so one finished call fills the window.
    backend = GeminiBackend(
        client=FakeClient(interactions),
        clock=fake_time.clock,
        sleep=sleep_until_the_stop,
        stop=stop,
        max_usd_per_10_min=1e-9,
    )
    backend.generate(make_request())

    with pytest.raises(BackendError, match="a stop was requested before this request was sent"):
        backend.generate(make_request())

    assert len(interactions.calls) == 1
    assert fake_time.sleeps == [600.0]


def test_spend_limiter_ends_its_wait_when_the_stop_event_is_set() -> None:
    fake_time = FakeTime(start=1000.0)
    limiter = SpendRateLimiter(1.0, window_s=600.0, clock=fake_time.clock, sleep=fake_time.sleep)
    limiter.record(1.0)
    stop = threading.Event()
    stop.set()

    limiter.wait_for_room(stop)

    # The window is still full: the caller must not send its call.
    assert fake_time.sleeps == []
    assert limiter.spent_usd() == pytest.approx(1.0)
    # An event that is not set changes nothing: the wait goes on until the window has room.
    limiter.wait_for_room(threading.Event())
    assert fake_time.sleeps == [600.0]


def test_call_in_flight_is_not_cut_off_by_a_stop_request() -> None:
    """The stop request comes while the provider works on the request: the answer is paid, so it is returned."""
    stop = threading.Event()
    answer = make_interaction()

    class StoppedDuringTheCall:
        def create(self, **kwargs: Any) -> Any:
            stop.set()
            return answer

    backend = GeminiBackend(client=FakeClient(StoppedDuringTheCall()), stop=stop)

    assert backend.generate(make_request()).text == ANSWER
    # The next request is not sent.
    with pytest.raises(BackendError, match="a stop was requested before this request was sent"):
        backend.generate(make_request())


# --- Backend: key and SDK --------------------------------------------------------


def test_missing_key_raises_backend_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_NAME, raising=False)
    backend = GeminiBackend(api_key_env=ENV_NAME)
    with pytest.raises(BackendError) as excinfo:
        backend.generate(make_request())
    assert ENV_NAME in str(excinfo.value)


def test_blank_key_raises_backend_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_NAME, "   ")
    backend = GeminiBackend(api_key_env=ENV_NAME)
    with pytest.raises(BackendError) as excinfo:
        backend.generate(make_request())
    assert ENV_NAME in str(excinfo.value)


def test_missing_sdk_raises_backend_error_without_the_key_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_NAME, FAKE_VALUE)
    # A None entry makes the import fail, also on a machine that has the SDK.
    monkeypatch.setitem(sys.modules, "google", None)
    monkeypatch.setitem(sys.modules, "google.genai", None)
    backend = GeminiBackend(api_key_env=ENV_NAME)

    with pytest.raises(BackendError) as excinfo:
        backend.generate(make_request())

    assert "google-genai" in str(excinfo.value)
    assert FAKE_VALUE not in str(excinfo.value)


def test_sdk_client_gets_the_key_a_timeout_in_milliseconds_and_one_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_NAME, FAKE_VALUE)
    sdk = FakeSdk(monkeypatch)

    build_sdk_client(ENV_NAME, 120.0)

    (arguments,) = sdk.client_arguments
    assert set(arguments) == {"api_key", "http_options"}
    assert arguments["api_key"] == FAKE_VALUE
    options = arguments["http_options"].arguments
    # The SDK takes the client timeout in milliseconds.
    assert options["timeout"] == 120_000
    # The SDK must not try a call again by itself: the backend does the retries and counts the attempts.
    assert options["retry_options"].arguments == {"attempts": 1}


def test_backend_builds_the_sdk_client_one_time_with_the_longest_call_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_NAME, FAKE_VALUE)
    interactions = FakeInteractions([make_interaction(), make_interaction()])
    sdk = FakeSdk(monkeypatch, interactions=interactions)
    backend = GeminiBackend(api_key_env=ENV_NAME)

    assert backend.generate(make_request(service_tier="standard")).text == ANSWER
    assert backend.generate(make_request(service_tier="flex")).text == ANSWER

    # The client is built on the first request and then kept. Its timeout covers a flex call: 900 seconds.
    (arguments,) = sdk.client_arguments
    assert arguments["http_options"].arguments["timeout"] == 900_000
    assert [call["timeout"] for call in interactions.calls] == [300.0, 900.0]


def test_error_of_the_sdk_client_never_shows_the_key_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_NAME, FAKE_VALUE)
    FakeSdk(monkeypatch, client_error=RuntimeError(f"the value {FAKE_VALUE} is not accepted"))

    with pytest.raises(BackendError) as excinfo:
        build_sdk_client(ENV_NAME, 120.0)

    assert "is not accepted" in str(excinfo.value)
    assert FAKE_VALUE not in str(excinfo.value)
    # The SDK error holds the key value, so it must not be printed with the traceback either.
    assert excinfo.value.__cause__ is None
    assert FAKE_VALUE not in "".join(traceback.format_exception(excinfo.value))


def test_key_value_is_removed_from_error_messages_and_log_lines(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(ENV_NAME, FAKE_VALUE)
    fake_time = FakeTime()
    outcomes = [
        FakeApiError(429, f"slow down, caller {FAKE_VALUE}"),
        FakeApiError(400, f"bad request from caller {FAKE_VALUE}"),
    ]
    backend, interactions = make_backend(outcomes, fake_time, api_key_env=ENV_NAME)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME), pytest.raises(BackendError) as excinfo:
        backend.generate(make_request())

    assert len(interactions.calls) == 2
    assert "bad request" in str(excinfo.value)
    assert FAKE_VALUE not in str(excinfo.value)
    assert "slow down" in caplog.text
    assert FAKE_VALUE not in caplog.text


@pytest.mark.parametrize(
    "options",
    [
        {"api_key_env": ""},
        {"timeout_s": 0.0},
        {"flex_timeout_s": 0.0},
        {"flex_timeout_s": float("nan")},
        {"max_attempts": 0},
        {"flex_busy_wait_s": 0.0},
        {"flex_busy_wait_s": float("nan")},
        {"max_usd_per_10_min": 0.0},
        {"max_usd_per_10_min": float("nan")},
    ],
)
def test_constructor_rejects_wrong_settings(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        GeminiBackend(**options)


# --- Backend: spend window and prices --------------------------------------------


def test_full_spend_window_delays_the_next_call() -> None:
    fake_time = FakeTime(start=1000.0)
    outcomes = [make_interaction(), make_interaction()]
    # The limit is far below the cost of one call, so one finished call fills the window.
    backend, interactions = make_backend(outcomes, fake_time, max_usd_per_10_min=1e-9)

    backend.generate(make_request())
    assert fake_time.sleeps == []

    backend.generate(make_request())
    # The first call finished at 1002.5 s. The second call must wait until it leaves the window.
    assert fake_time.sleeps == [600.0]
    assert interactions.call_times == [1000.0, 1602.5]


def test_cost_of_an_answer_that_is_not_completed_still_fills_the_window() -> None:
    fake_time = FakeTime(start=1000.0)
    outcomes = [make_interaction(status="incomplete"), make_interaction()]
    backend, interactions = make_backend(outcomes, fake_time, max_usd_per_10_min=1e-9)

    with pytest.raises(BackendError):
        backend.generate(make_request())
    backend.generate(make_request())

    assert fake_time.sleeps == [600.0]
    assert interactions.call_times == [1000.0, 1602.5]


def test_spend_window_holds_8_usd_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # A stand-in price: every call costs 5 USD.
    monkeypatch.setattr(client_module, "cost_usd", lambda model, usage, tier, on: 5.0)
    fake_time = FakeTime(start=1000.0)
    backend, _ = make_backend([make_interaction() for _ in range(3)], fake_time)

    backend.generate(make_request())
    backend.generate(make_request())
    # 5 USD are in the window before the second call: below the limit of 8 USD.
    assert fake_time.sleeps == []
    backend.generate(make_request())
    # 10 USD are in the window before the third call, so it waits until the first call has left the window.
    assert fake_time.sleeps == [600.0 - CALL_SECONDS]


@pytest.mark.parametrize(("served_tier", "sleeps"), [("flex", []), ("standard", [600.0]), ("priority", [600.0])])
def test_spend_window_counts_the_price_of_the_served_tier(
    served_tier: str, sleeps: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A stand-in price: a flex call costs 1 USD, a standard call 2 USD and a priority call 3.6 USD, whatever its tokens.
    prices = {"flex": 1.0, "standard": 2.0, "priority": 3.6}
    monkeypatch.setattr(client_module, "cost_usd", lambda model, usage, tier, on: prices[tier])
    fake_time = FakeTime(start=1000.0)
    outcomes = [make_interaction(service_tier=served_tier), make_interaction()]
    backend, _ = make_backend(outcomes, fake_time, max_usd_per_10_min=1.5)

    backend.generate(make_request(service_tier="flex"))
    backend.generate(make_request(service_tier="flex"))

    # A call at the flex price leaves room in the window. A call at the standard price fills it.
    assert fake_time.sleeps == sleeps


def test_model_without_a_price_fails_before_any_call() -> None:
    fake_time = FakeTime()
    backend, interactions = make_backend([make_interaction()], fake_time)
    with pytest.raises(BackendError, match="price"):
        backend.generate(make_request(model="model-without-a-price"))
    assert interactions.calls == []


def test_flex_request_for_a_model_without_a_flex_tier_fails_before_any_call() -> None:
    """Such a call has no price. After the call its cost could not be recorded, so it is not sent."""
    fake_time = FakeTime()
    backend, interactions = make_backend([make_interaction(), make_interaction()], fake_time)
    with pytest.raises(BackendError, match="has no flex tier"):
        backend.generate(make_request(model="gemini-robotics-er-2-preview", service_tier="flex"))
    assert interactions.calls == []
    # The standard tier of the same model is sent.
    backend.generate(make_request(model="gemini-robotics-er-2-preview", service_tier="standard"))
    assert len(interactions.calls) == 1
