"""Tests of the price table, the cost estimate, the cost ledger and the budget guard.

The sums of the estimate tests come from the token table of this file (``ESTIMATES``), not from
``DEFAULT_TOKEN_ESTIMATES``. That table changes when real token counts are measured, and no test
here pins its values.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pytest

from pitchvlm.vlm import cost as cost_module
from pitchvlm.vlm.cost import (
    DEFAULT_TOKEN_ESTIMATES,
    FLEX_FACTOR,
    FLEX_TIER_MODELS,
    HIGHEST_PRICE_TIER,
    MAX_LEDGER_FILE_BYTES,
    PRICES,
    PRIORITY_FACTOR,
    BudgetExceeded,
    BudgetGuard,
    Ledger,
    Price,
    TokenEstimate,
    billed_usd,
    cost_usd,
    estimate_usd,
    has_flex_tier,
    ledger_file_number,
    price_for,
)
from pitchvlm.vlm.prompts import PROMPT_NAMES
from pitchvlm.vlm.types import BilledTier, RequestMeta, ServiceTier, TextPart, Usage, VLMRequest, VLMResponse

LOGGER_NAME = "pitchvlm.vlm.cost"
REPO = Path(__file__).resolve().parents[1]
#: The repository guard. Every result file must pass it before it goes to the results branch.
GUARD = REPO / "tools" / "guard.py"
#: The rules of the secret scan of the git hooks.
SECRET_SCAN_FILE = REPO / ".gitleaks.toml"
#: A day in the first price period of gemini-4-argon (2 / 10 USD per million tokens).
NOVEMBER = date(2026, 11, 20)
#: A day in its second price period (4 / 20 USD per million tokens).
DECEMBER = date(2026, 12, 5)
#: With gemini-4-argon in November this costs (1500 * 2 + (1000 + 400) * 10) / 1e6 = 0.017 USD.
USAGE = Usage(input_tokens=1500, output_tokens=1000, thought_tokens=400)
#: The token table of the estimate tests. With gemini-4-argon in November (2 / 10 USD per million
#: tokens) one request costs: people 0.017, landmarks 0.0124, number 0.009, kit_legend 0.0132 USD.
ESTIMATES = {
    "people": TokenEstimate(input_tokens=1500, output_tokens=1400),
    "landmarks": TokenEstimate(input_tokens=2200, output_tokens=800),
    "number": TokenEstimate(input_tokens=2500, output_tokens=400),
    "kit_legend": TokenEstimate(input_tokens=3600, output_tokens=600),
}
LEDGER_KEYS = [
    "time",
    "run_id",
    "clip_id",
    "step",
    "frame",
    "item",
    "repeat",
    "request",
    "provider",
    "model",
    "served_by",
    "service_tier",
    "attempts",
    "input_tokens",
    "output_tokens",
    "thought_tokens",
    "usd",
]


def make_request(
    step: str = "people",
    *,
    model: str = "gemini-4-argon",
    service_tier: ServiceTier = "standard",
    frame: int | None = 12,
    item: str | None = None,
    repeat: int = 0,
) -> VLMRequest:
    """Build a request without images. The cost code reads only its model, tier, meta and cache key."""
    return VLMRequest(
        model=model,
        thinking_level="low",
        parts=(TextPart(text="prompt"),),
        response_schema={"type": "object"},
        prompt_name=f"{step}/v1",
        meta=RequestMeta(clip_id="clip-1", step=step, frame=frame, item=item, repeat=repeat),
        service_tier=service_tier,
    )


def make_response(
    *,
    usage: Usage = USAGE,
    service_tier: BilledTier = "standard",
    created_at: str = "2026-11-20T10:00:00+00:00",
    model: str = "gemini-4-argon",
    attempts: int = 1,
) -> VLMResponse:
    """Build an answer with the given token usage and time."""
    return VLMResponse(
        text="{}",
        usage=usage,
        model=model,
        service_tier=service_tier,
        latency_s=2.0,
        attempts=attempts,
        created_at=created_at,
    )


def read_lines(path: Path) -> list[dict[str, object]]:
    """Return the JSON objects of a ledger file, one per line."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def spend(ledger: Ledger, run_id: str, usd: float) -> None:
    """Record one answer that costs ``usd``: input tokens only, at 2 USD per million tokens."""
    usage = Usage(input_tokens=round(usd * 500_000), output_tokens=0)
    ledger.record(run_id, make_request(), make_response(usage=usage))


def plan_of_one_clip() -> list[tuple[VLMRequest, VLMResponse]]:
    """Return the 179 paid answers of one clip of the plan, with the tokens of ``ESTIMATES``.

    One kit legend request, a people and a landmarks request for each of 64 keyframes, and 50
    number requests. With gemini-4-argon in November they cost 2.3448 USD together.
    """
    requests = [make_request("kit_legend", frame=None)]
    requests += [make_request(step, frame=12 * index) for index in range(64) for step in ("people", "landmarks")]
    requests += [make_request("number", frame=None, item=f"crops-{track:016x}") for track in range(50)]
    answers: list[tuple[VLMRequest, VLMResponse]] = []
    for request in requests:
        estimate = ESTIMATES[request.meta.step]
        usage = Usage(input_tokens=estimate.input_tokens, output_tokens=estimate.output_tokens)
        answers.append((request, make_response(usage=usage)))
    return answers


@pytest.fixture
def private_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Make git read no user and no system configuration, and find no repository above the test folder."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    home = tmp_path / "home"
    home.mkdir()
    for name in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))


@pytest.fixture
def read_only_folder(tmp_path: Path) -> Iterator[Path]:
    """A folder that cannot be written. The test is skipped where the user may write everywhere."""
    folder = tmp_path / "read-only"
    folder.mkdir()
    folder.chmod(0o555)
    if os.access(folder, os.W_OK):
        folder.chmod(0o755)
        pytest.skip("this user can write into a folder without the write permission")
    yield folder
    folder.chmod(0o755)


# --- Price table ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "on", "input_price", "output_price"),
    [
        ("gemini-4-argon", date(2026, 10, 13), 2.0, 10.0),
        ("gemini-4-argon", date(2026, 11, 30), 2.0, 10.0),
        ("gemini-4-argon", date(2026, 12, 1), 4.0, 20.0),
        ("gemini-4-argon", date(2027, 6, 1), 4.0, 20.0),
        ("gemini-4-carbon", date(2026, 11, 30), 2.0, 10.0),
        ("gemini-4-carbon", date(2026, 12, 1), 4.0, 20.0),
        ("gemini-4-hydrogen", date(2026, 11, 20), 1.5, 7.5),
        ("gemini-4-hydrogen", date(2027, 6, 1), 1.5, 7.5),
        ("gemini-3.8-flash", date(2026, 12, 31), 0.75, 3.75),
        ("gemini-3.8-flash", date(2027, 1, 1), 1.5, 7.5),
        ("gemini-robotics-er-2-preview", date(2026, 12, 31), 1.0, 5.0),
        ("gemini-robotics-er-2-preview", date(2027, 1, 1), 2.0, 10.0),
    ],
)
def test_price_for_takes_the_latest_entry_that_has_started(
    model: str, on: date, input_price: float, output_price: float
) -> None:
    price = price_for(model, on)

    assert price.model == model
    assert price.valid_from <= on
    assert (price.input_usd_per_million, price.output_usd_per_million) == (input_price, output_price)


def test_price_for_refuses_an_unknown_model() -> None:
    with pytest.raises(KeyError):
        price_for("gemini-9-xenon", NOVEMBER)


def test_price_for_refuses_a_day_before_the_first_entry() -> None:
    with pytest.raises(KeyError):
        price_for("gemini-4-argon", date(2026, 10, 12))


def test_price_table_is_a_tuple_of_frozen_prices() -> None:
    assert isinstance(PRICES, tuple)
    assert all(isinstance(price, Price) for price in PRICES)
    # One entry per model and start day.
    assert len({(price.model, price.valid_from) for price in PRICES}) == len(PRICES)
    with pytest.raises(dataclasses.FrozenInstanceError):
        PRICES[0].input_usd_per_million = 0.0  # type: ignore[misc]


# --- cost_usd -------------------------------------------------------------------


def test_cost_in_the_first_price_period() -> None:
    # (1500 * 2 + (1000 + 400) * 10) / 1e6
    assert cost_usd("gemini-4-argon", USAGE, "standard", NOVEMBER) == pytest.approx(0.017)


def test_cost_in_the_second_price_period() -> None:
    # (1500 * 4 + (1000 + 400) * 20) / 1e6
    assert cost_usd("gemini-4-argon", USAGE, "standard", DECEMBER) == pytest.approx(0.034)


def test_cost_changes_on_the_first_day_of_the_new_price() -> None:
    assert cost_usd("gemini-4-argon", USAGE, "standard", date(2026, 11, 30)) == pytest.approx(0.017)
    assert cost_usd("gemini-4-argon", USAGE, "standard", date(2026, 12, 1)) == pytest.approx(0.034)


def test_flex_costs_half_in_both_price_periods() -> None:
    assert FLEX_FACTOR == 0.5
    assert cost_usd("gemini-4-argon", USAGE, "flex", NOVEMBER) == pytest.approx(0.0085)
    assert cost_usd("gemini-4-argon", USAGE, "flex", DECEMBER) == pytest.approx(0.017)


def test_flex_tier_exists_for_every_priced_model_but_the_robotics_model() -> None:
    priced = {price.model for price in PRICES}

    assert FLEX_TIER_MODELS == priced - {"gemini-robotics-er-2-preview"}
    assert all(has_flex_tier(model) for model in FLEX_TIER_MODELS)
    assert not has_flex_tier("gemini-robotics-er-2-preview")
    assert not has_flex_tier("gemini-9-xenon")


def test_flex_tier_of_a_model_without_one_has_no_price() -> None:
    """Half the standard price would be a price that the provider does not have, and too low an estimate."""
    robotics = "gemini-robotics-er-2-preview"

    with pytest.raises(ValueError, match="gemini-robotics-er-2-preview has no flex tier"):
        cost_usd(robotics, USAGE, "flex", NOVEMBER)
    with pytest.raises(ValueError, match="has no flex tier"):
        estimate_usd([make_request("landmarks", model=robotics, service_tier="flex")], ESTIMATES, NOVEMBER)
    # The standard tier and the priority tier of the model have a price.
    assert cost_usd(robotics, USAGE, "standard", NOVEMBER) == pytest.approx(0.0085)
    assert cost_usd(robotics, USAGE, "priority", NOVEMBER) == pytest.approx(0.0153)
    # 2200 input tokens and 800 output tokens: (2200 * 1 + 800 * 5) / 1e6.
    assert estimate_usd([make_request("landmarks", model=robotics)], ESTIMATES, NOVEMBER) == pytest.approx(0.0062)
    # A model without any price is still refused for its missing price.
    with pytest.raises(KeyError):
        cost_usd("gemini-9-xenon", USAGE, "flex", NOVEMBER)


def test_thinking_tokens_cost_the_output_price() -> None:
    million = 1_000_000
    only_input = Usage(input_tokens=million, output_tokens=0)
    only_output = Usage(input_tokens=0, output_tokens=million)
    only_thinking = Usage(input_tokens=0, output_tokens=0, thought_tokens=million)

    assert cost_usd("gemini-4-argon", only_input, "standard", NOVEMBER) == pytest.approx(2.0)
    assert cost_usd("gemini-4-argon", only_output, "standard", NOVEMBER) == pytest.approx(10.0)
    assert cost_usd("gemini-4-argon", only_thinking, "standard", NOVEMBER) == pytest.approx(10.0)


def test_cached_tokens_do_not_change_the_cost() -> None:
    cached = Usage(input_tokens=1500, output_tokens=1000, thought_tokens=400, cached_tokens=1000)
    with_cache = cost_usd("gemini-4-argon", cached, "standard", NOVEMBER)

    assert with_cache == cost_usd("gemini-4-argon", USAGE, "standard", NOVEMBER)


def test_cost_of_the_other_models() -> None:
    # (1500 * 1.5 + 1400 * 7.5) / 1e6
    assert cost_usd("gemini-4-hydrogen", USAGE, "standard", NOVEMBER) == pytest.approx(0.01275)
    # (1500 * 0.75 + 1400 * 3.75) / 1e6
    assert cost_usd("gemini-3.8-flash", USAGE, "standard", NOVEMBER) == pytest.approx(0.006375)
    # (1500 * 1 + 1400 * 5) / 1e6
    assert cost_usd("gemini-robotics-er-2-preview", USAGE, "standard", NOVEMBER) == pytest.approx(0.0085)


def test_priority_costs_1_8_times_the_standard_price() -> None:
    """No request asks for the priority tier, but an answer can report it. Its cost must not count as less."""
    assert PRIORITY_FACTOR == 1.8
    # (1500 * 3.6 + 1400 * 18) / 1e6
    assert cost_usd("gemini-4-argon", USAGE, "priority", NOVEMBER) == pytest.approx(0.0306)
    assert cost_usd("gemini-4-argon", USAGE, "priority", DECEMBER) == pytest.approx(0.0612)


def test_priority_is_the_tier_with_the_highest_price() -> None:
    """An answer that reports a tier without a price is billed like this tier."""
    assert HIGHEST_PRICE_TIER == "priority"
    costs = {tier: cost_usd("gemini-4-argon", USAGE, tier, NOVEMBER) for tier in ("standard", "flex", "priority")}
    assert max(costs, key=lambda tier: costs[tier]) == HIGHEST_PRICE_TIER


def test_cost_refuses_an_unknown_tier_and_an_unknown_model() -> None:
    with pytest.raises(ValueError):
        cost_usd("gemini-4-argon", USAGE, "deferred", NOVEMBER)  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        cost_usd("gemini-9-xenon", USAGE, "standard", NOVEMBER)


def test_billed_usd_is_the_cost_of_an_answer_on_its_utc_day() -> None:
    assert billed_usd(make_response()) == pytest.approx(0.017)
    assert billed_usd(make_response(service_tier="flex")) == pytest.approx(0.0085)
    assert billed_usd(make_response(service_tier="priority")) == pytest.approx(0.0306)
    assert billed_usd(make_response(created_at="2026-12-01T00:00:00+00:00")) == pytest.approx(0.034)
    # Half past midnight in a zone two hours ahead is still 30 November in UTC.
    assert billed_usd(make_response(created_at="2026-12-01T00:30:00+02:00")) == pytest.approx(0.017)
    assert billed_usd(make_response(model="gemini-4-hydrogen")) == pytest.approx(0.01275)
    with pytest.raises(KeyError):
        billed_usd(make_response(model="gemini-9-xenon"))


# --- estimate_usd ---------------------------------------------------------------


def test_default_token_estimates_name_every_model_step_with_counts_above_zero() -> None:
    """The values of the default table are not pinned: they change when real token counts are measured."""
    assert set(DEFAULT_TOKEN_ESTIMATES) == {
        "kit_legend",
        "people",
        "landmarks",
        "number",
        "kit_legend_video",
        "people_video",
    }
    # Every step that has a prompt has an estimate, so the cost of every request can be estimated.
    assert set(DEFAULT_TOKEN_ESTIMATES) == set(PROMPT_NAMES)
    for estimate in DEFAULT_TOKEN_ESTIMATES.values():
        assert isinstance(estimate, TokenEstimate)
        assert estimate.input_tokens > 0
        assert estimate.output_tokens > 0


def test_estimate_adds_the_requests_by_step_model_and_tier() -> None:
    requests = [
        make_request("people"),
        make_request("people"),
        make_request("landmarks"),
        make_request("number", service_tier="flex"),
    ]

    # people 0.017 each, landmarks 0.0124, number 0.009 at half price.
    assert estimate_usd(requests, ESTIMATES, NOVEMBER) == pytest.approx(0.0509)
    assert estimate_usd(requests, ESTIMATES, DECEMBER) == pytest.approx(0.1018)
    # Any iterable works, also one that can be read only once.
    assert estimate_usd(iter(requests), ESTIMATES, NOVEMBER) == pytest.approx(0.0509)


def test_estimate_uses_the_model_of_each_request() -> None:
    requests = [make_request("people"), make_request("people", model="gemini-4-hydrogen")]

    # (1500 * 2 + 1400 * 10) / 1e6 + (1500 * 1.5 + 1400 * 7.5) / 1e6
    assert estimate_usd(requests, ESTIMATES, NOVEMBER) == pytest.approx(0.02975)


def test_estimate_of_the_requests_of_one_clip() -> None:
    requests = [make_request("kit_legend", frame=None)]
    requests += [make_request(step, frame=frame) for frame in range(64) for step in ("people", "landmarks")]
    requests += [make_request("number", frame=None, item=f"track-{track}") for track in range(50)]

    # 0.0132 + 64 * 0.017 + 64 * 0.0124 + 50 * 0.009
    assert estimate_usd(requests, ESTIMATES, NOVEMBER) == pytest.approx(2.3448)


def test_estimate_follows_the_token_table_that_it_gets() -> None:
    estimates = {"people": TokenEstimate(input_tokens=1_000_000, output_tokens=100_000)}

    # (1e6 * 2 + 1e5 * 10) / 1e6
    assert estimate_usd([make_request("people")], estimates, NOVEMBER) == pytest.approx(3.0)
    assert estimate_usd([make_request("people")], ESTIMATES, NOVEMBER) == pytest.approx(0.017)


def test_estimate_with_the_default_table_is_above_zero() -> None:
    requests = [
        make_request(step).model_copy(update={"context": {"frames": [12]}}) for step in DEFAULT_TOKEN_ESTIMATES
    ]

    assert estimate_usd(requests, DEFAULT_TOKEN_ESTIMATES, NOVEMBER) > 0.0


def test_estimate_of_no_request_is_zero() -> None:
    assert estimate_usd([], ESTIMATES, NOVEMBER) == 0.0


def test_estimate_refuses_a_step_without_an_estimate() -> None:
    with pytest.raises(KeyError):
        estimate_usd([make_request("offside")], ESTIMATES, NOVEMBER)


def test_estimate_refuses_an_unknown_model() -> None:
    with pytest.raises(KeyError):
        estimate_usd([make_request("people", model="gemini-9-xenon")], ESTIMATES, NOVEMBER)


# --- Ledger ---------------------------------------------------------------------


def test_record_adds_one_json_line_and_returns_the_cost(tmp_path: Path) -> None:
    path = tmp_path / "results" / "ledger.jsonl"
    ledger = Ledger(path)
    request = make_request("number", frame=None, item="track-7", repeat=2)

    usd = ledger.record("run-1", request, make_response(attempts=3))

    assert usd == pytest.approx(0.017)
    (line,) = read_lines(path)
    assert list(line) == LEDGER_KEYS
    assert line == {
        "time": "2026-11-20T10:00:00+00:00",
        "run_id": "run-1",
        "clip_id": "clip-1",
        "step": "number",
        "frame": None,
        "item": "track-7",
        "repeat": 2,
        "request": request.cache_key(),
        "provider": "gemini",
        "model": "gemini-4-argon",
        "served_by": None,
        "service_tier": "standard",
        "attempts": 3,
        "input_tokens": 1500,
        "output_tokens": 1000,
        "thought_tokens": 400,
        "usd": pytest.approx(0.017),
    }
    # The file is plain text that ends with a line end.
    assert path.read_bytes().endswith(b"}\n")


def test_ledger_tells_a_repeat_run_from_a_second_payment_for_the_same_request(tmp_path: Path) -> None:
    """Two lines for one clip, step and frame: the repeat index and the request key say what happened."""
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    first_run, repeat_run = make_request(frame=24), make_request(frame=24, repeat=1)

    ledger.record("run-1", first_run, make_response())
    ledger.record("run-2", first_run, make_response())
    ledger.record("run-3", repeat_run, make_response())

    first, second, repeated = read_lines(path)
    place = ("clip_id", "step", "frame", "item")
    assert [first[name] for name in place] == [second[name] for name in place] == [repeated[name] for name in place]
    # The same request was paid for two times, for example after its cassette was removed.
    assert (first["repeat"], second["repeat"]) == (0, 0)
    assert first["request"] == second["request"] == first_run.cache_key()
    # The repeat run asked another question: it has another repeat index and another request key.
    assert repeated["repeat"] == 1
    assert repeated["request"] == repeat_run.cache_key() != first["request"]


def test_record_keeps_the_frame_of_the_request(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    Ledger(path).record("run-1", make_request("people", frame=24), make_response())

    (line,) = read_lines(path)
    assert (line["step"], line["frame"], line["item"]) == ("people", 24, None)


@pytest.mark.parametrize(
    ("created_at", "time", "usd"),
    [
        ("2026-11-30T23:59:59+00:00", "2026-11-30T23:59:59+00:00", 0.017),
        ("2026-12-01T00:00:00+00:00", "2026-12-01T00:00:00+00:00", 0.034),
        ("2026-12-01T08:00:00Z", "2026-12-01T08:00:00+00:00", 0.034),
        # Half past midnight in a zone two hours ahead is still 30 November in UTC.
        ("2026-12-01T00:30:00+02:00", "2026-11-30T22:30:00+00:00", 0.017),
        # A time without an offset is UTC.
        ("2026-12-01T00:30:00", "2026-12-01T00:30:00+00:00", 0.034),
    ],
)
def test_record_uses_the_price_of_the_utc_day_of_the_answer(
    tmp_path: Path, created_at: str, time: str, usd: float
) -> None:
    path = tmp_path / "ledger.jsonl"

    assert Ledger(path).record("run-1", make_request(), make_response(created_at=created_at)) == pytest.approx(usd)
    (line,) = read_lines(path)
    assert line["time"] == time
    assert line["usd"] == pytest.approx(usd)


def test_record_uses_the_tier_and_the_tokens_of_the_answer(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    usage = Usage(input_tokens=1_000_000, output_tokens=200_000, thought_tokens=300_000)

    usd = Ledger(path).record("run-1", make_request(), make_response(usage=usage, service_tier="flex"))

    # (1e6 * 2 + 5e5 * 10) / 1e6 = 7 USD, half of it for flex.
    assert usd == pytest.approx(3.5)
    (line,) = read_lines(path)
    assert line["service_tier"] == "flex"
    assert (line["input_tokens"], line["output_tokens"], line["thought_tokens"]) == (1_000_000, 200_000, 300_000)


def test_record_uses_the_model_of_the_answer(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    request = make_request(model="gemini-4-argon")

    # The answer says which model was billed: (1500 * 1.5 + 1400 * 7.5) / 1e6.
    usd = Ledger(path).record("run-1", request, make_response(model="gemini-4-hydrogen"))

    assert usd == pytest.approx(0.01275)
    (line,) = read_lines(path)
    assert line["model"] == "gemini-4-hydrogen"


def test_record_refuses_an_answer_without_a_price_or_a_time(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)

    with pytest.raises(KeyError):
        ledger.record("run-1", make_request(), make_response(model="gemini-9-xenon"))
    with pytest.raises(KeyError):
        ledger.record("run-1", make_request(), make_response(created_at="2026-10-12T23:59:59+00:00"))
    with pytest.raises(ValueError):
        ledger.record("run-1", make_request(), make_response(created_at="yesterday"))
    # A refused answer leaves no line.
    assert not path.exists()
    assert ledger.total_usd() == 0.0


def test_totals_per_run_and_of_the_whole_ledger(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)

    assert ledger.total_usd() == 0.0
    assert ledger.run_usd("run-1") == 0.0

    for _ in range(3):
        ledger.record("run-1", make_request(), make_response())
    ledger.record("run-2", make_request(), make_response(service_tier="flex"))

    assert ledger.run_usd("run-1") == pytest.approx(0.051)
    assert ledger.run_usd("run-2") == pytest.approx(0.0085)
    assert ledger.run_usd("run-3") == 0.0
    assert ledger.total_usd() == pytest.approx(0.0595)
    assert len(read_lines(path)) == 4


def test_ledger_file_is_kept_between_ledger_objects(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    Ledger(path).record("run-1", make_request(), make_response())
    later = Ledger(path)

    later.record("run-2", make_request(), make_response())

    # Lines are added, never replaced.
    assert [line["run_id"] for line in read_lines(path)] == ["run-1", "run-2"]
    assert later.run_usd("run-1") == pytest.approx(0.017)
    assert later.total_usd() == pytest.approx(0.034)


def test_ledger_is_safe_from_several_threads(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)

    def record(index: int) -> float:
        return ledger.record(f"run-{index % 2}", make_request(frame=index), make_response())

    with ThreadPoolExecutor(max_workers=8) as pool:
        costs = list(pool.map(record, range(200)))

    lines = read_lines(path)
    assert len(lines) == 200
    assert sorted(line["frame"] for line in lines) == list(range(200))  # type: ignore[type-var]
    assert ledger.total_usd() == pytest.approx(sum(costs))
    assert ledger.run_usd("run-0") == pytest.approx(100 * 0.017)
    assert ledger.run_usd("run-1") == pytest.approx(100 * 0.017)


@pytest.mark.parametrize(
    "damaged",
    [
        "{broken",
        "[1, 2]",
        '{"run_id": "run-1"}',
        '{"run_id": "run-1", "usd": "much"}',
        '{"run_id": "run-1", "usd": true}',
        '{"run_id": 1, "usd": 2}',
        # A total that is not a number would switch both limits of the budget guard off.
        '{"run_id": "run-1", "usd": NaN}',
        '{"run_id": "run-1", "usd": Infinity}',
        '{"run_id": "run-1", "usd": -Infinity}',
        # A negative cost would lower the total that the hard stop reads.
        '{"run_id": "refund", "usd": -100}',
        '{"run_id": "run-1", "usd": -0.001}',
    ],
    ids=[
        "cut off",
        "list",
        "no cost",
        "cost is text",
        "cost is a truth value",
        "run id is a number",
        "cost is not a number",
        "cost is infinite",
        "cost is minus infinite",
        "cost is a negative whole number",
        "cost is a small negative number",
    ],
)
def test_damaged_ledger_line_is_an_error(tmp_path: Path, damaged: str) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.record("run-1", make_request(), make_response())
    with path.open("a", encoding="utf-8") as stream:
        stream.write(damaged + "\n")

    with pytest.raises(ValueError, match="line 2"):
        ledger.total_usd()
    with pytest.raises(ValueError, match="line 2"):
        ledger.run_usd("run-1")


def test_cost_that_is_written_as_a_whole_number_is_read(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"run_id": "run-1", "usd": 2}\n{"run_id": "run-2", "usd": 0.5}\n', encoding="utf-8")

    assert Ledger(path).total_usd() == pytest.approx(2.5)
    assert Ledger(path).run_usd("run-1") == pytest.approx(2.0)


def test_line_with_blanks_around_it_is_counted_and_an_empty_line_is_not(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    Ledger(path).record("run-1", make_request(), make_response())
    with path.open("a", encoding="utf-8") as stream:
        stream.write('  {"run_id": "run-1", "usd": 1.5}  \n\n   \n\t{"run_id": "run-2", "usd": 0.25}\n')

    assert Ledger(path).total_usd() == pytest.approx(1.767)
    assert Ledger(path).run_usd("run-1") == pytest.approx(1.517)
    assert Ledger(path).run_usd("run-2") == pytest.approx(0.25)


def test_record_starts_a_new_line_when_the_last_line_has_no_line_end(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Two answers on one line would damage the ledger, and every later paid command would refuse to start."""
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.record("run-1", make_request(frame=12), make_response())
    # An editor removed the line end of the last line. The line itself is whole.
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    assert ledger.total_usd() == pytest.approx(0.017)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        ledger.record("run-2", make_request(frame=13), make_response())
        ledger.record("run-2", make_request(frame=14), make_response())

    lines = read_lines(path)
    assert [(line["run_id"], line["frame"]) for line in lines] == [("run-1", 12), ("run-2", 13), ("run-2", 14)]
    assert ledger.total_usd() == pytest.approx(0.051)
    assert path.read_bytes().endswith(b"}\n")
    assert b"\n\n" not in path.read_bytes()
    # The repair is logged one time. The next answer found a line end and needed no repair.
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert str(path) in record.getMessage()


def test_record_logs_nothing_for_a_ledger_that_ends_with_a_line_end(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_bytes(b"")
    ledger = Ledger(path)

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        ledger.record("run-1", make_request(), make_response())
        ledger.record("run-1", make_request(), make_response())

    assert caplog.records == []
    assert len(read_lines(path)) == 2


# --- Ledger: several files ------------------------------------------------------


def limit_to_three_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make a ledger file full after three lines of ``make_request(frame=<one digit>)`` and ``make_response()``."""
    probe = Ledger(tmp_path / "line-size" / "ledger.jsonl")
    probe.record("run-0", make_request(frame=0), make_response())
    line_bytes = probe.path.stat().st_size
    # Three whole lines fit, and a fourth line does not.
    monkeypatch.setattr(cost_module, "MAX_LEDGER_FILE_BYTES", 3 * line_bytes + line_bytes // 2)


def test_full_ledger_file_is_followed_by_a_new_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    limit_to_three_lines(tmp_path, monkeypatch)
    path = tmp_path / "cache" / "ledger.jsonl"
    ledger = Ledger(path)
    assert ledger.files() == []

    for frame in range(10):
        ledger.record(f"run-{frame % 2}", make_request(frame=frame), make_response())

    files = ledger.files()
    names = ["ledger.jsonl", "ledger-0002.jsonl", "ledger-0003.jsonl", "ledger-0004.jsonl"]
    assert [file.name for file in files] == names
    assert [len(read_lines(file)) for file in files] == [3, 3, 3, 1]
    assert all(file.stat().st_size <= cost_module.MAX_LEDGER_FILE_BYTES for file in files)
    # The lines are whole and in the order of the answers, and no other file was made.
    assert [line["frame"] for file in files for line in read_lines(file)] == list(range(10))
    assert sorted(entry.name for entry in path.parent.iterdir()) == sorted(file.name for file in files)
    # The totals count every file.
    assert ledger.total_usd() == pytest.approx(0.17)
    assert ledger.run_usd("run-0") == pytest.approx(0.085)
    assert ledger.run_usd("run-1") == pytest.approx(0.085)


def test_line_that_fills_a_ledger_file_to_exactly_its_limit_stays_in_that_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file is full when the next line would make it larger than the limit, not when it reaches the limit."""
    probe = Ledger(tmp_path / "line-size" / "ledger.jsonl")
    probe.record("run-0", make_request(frame=0), make_response())
    line_bytes = probe.path.stat().st_size
    monkeypatch.setattr(cost_module, "MAX_LEDGER_FILE_BYTES", 2 * line_bytes)
    ledger = Ledger(tmp_path / "cache" / "ledger.jsonl")

    for frame in range(3):
        ledger.record("run-1", make_request(frame=frame), make_response())

    first, second = ledger.files()
    assert (first.name, second.name) == ("ledger.jsonl", "ledger-0002.jsonl")
    # The second line filled the first file to exactly the limit. The third line started the next file.
    assert first.stat().st_size == cost_module.MAX_LEDGER_FILE_BYTES
    assert [line["frame"] for line in read_lines(first)] == [0, 1]
    assert [line["frame"] for line in read_lines(second)] == [2]


def test_new_ledger_object_goes_on_in_the_last_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    limit_to_three_lines(tmp_path, monkeypatch)
    path = tmp_path / "cache" / "ledger.jsonl"
    first = Ledger(path)
    for frame in range(4):
        first.record("run-1", make_request(frame=frame), make_response())
    assert [file.name for file in first.files()] == ["ledger.jsonl", "ledger-0002.jsonl"]
    full_file = path.read_bytes()

    later = Ledger(path)
    later.record("run-2", make_request(frame=4), make_response())
    first.record("run-1", make_request(frame=5), make_response())

    # The first file is full and does not change any more. Both new lines are in the last file.
    assert path.read_bytes() == full_file
    assert later.files() == first.files()
    assert [line["run_id"] for line in read_lines(first.files()[-1])] == ["run-1", "run-2", "run-1"]
    assert later.total_usd() == pytest.approx(0.102)


def test_ledger_files_are_read_in_the_order_of_their_numbers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "ledger.jsonl"
    # Ten files with the numbers 1 to 10. The ninth one has its number without zeros in front.
    names = ["ledger.jsonl", *[f"ledger-{number:04d}.jsonl" for number in range(2, 9)], "ledger-9.jsonl"]
    names.append("ledger-0010.jsonl")
    costs = {name: 2.0**place for place, name in enumerate(names)}
    for name, usd in costs.items():
        (tmp_path / name).write_text(json.dumps({"run_id": "run-1", "usd": usd}) + "\n", encoding="utf-8")
    # Files with other names are not part of the ledger.
    for name in ("ledger.jsonl.bak", "ledger-notes.jsonl", "old-ledger-0002.jsonl", "ledger-0003.json", "notes.json"):
        (tmp_path / name).write_text("not a ledger line\n", encoding="utf-8")
    ledger = Ledger(path)

    # The order is the one of the numbers, not of the names: the name of file 10 sorts before the name of file 9.
    assert [file.name for file in ledger.files()] == names
    assert ledger.total_usd() == pytest.approx(1023.0)
    # Every file counts as full here. The new file gets the number after the highest one.
    monkeypatch.setattr(cost_module, "MAX_LEDGER_FILE_BYTES", 10)
    ledger.record("run-2", make_request(), make_response())
    assert [file.name for file in ledger.files()] == [*names, "ledger-0011.jsonl"]
    assert ledger.total_usd() == pytest.approx(1023.017)


@pytest.mark.parametrize(
    ("names", "found"),
    [
        pytest.param(["ledger-0002.jsonl"], "Found: ledger-0002.jsonl.", id="first file gone"),
        pytest.param(
            ["ledger.jsonl", "ledger-0003.jsonl"], "Found: ledger.jsonl, ledger-0003.jsonl.", id="gap in the numbers"
        ),
        pytest.param(
            ["ledger.jsonl", "ledger-0002.jsonl", "ledger-2.jsonl"],
            "Found: ledger.jsonl, ledger-0002.jsonl, ledger-2.jsonl.",
            id="number two times",
        ),
        pytest.param(
            ["ledger.jsonl", "ledger-0001.jsonl"], "Found: ledger-0001.jsonl, ledger.jsonl.", id="two first files"
        ),
    ],
)
def test_incomplete_series_of_ledger_files_is_an_error(tmp_path: Path, names: list[str], found: str) -> None:
    """A missing file would take up to 1,000,000 bytes of paid answers out of the total of the budget guard."""
    for name in names:
        (tmp_path / name).write_text(json.dumps({"run_id": "run-1", "usd": 51.0}) + "\n", encoding="utf-8")
    ledger = Ledger(tmp_path / "ledger.jsonl")
    guard = BudgetGuard(ledger)
    guard.start_run("run-2", 1.0)

    for read_total in (ledger.total_usd, lambda: ledger.run_usd("run-1"), lambda: guard.check("run-2")):
        with pytest.raises(ValueError, match="the files of the cost ledger are not a complete series") as excinfo:
            read_total()
        assert found in str(excinfo.value)
        assert "Bring the missing file back" in str(excinfo.value)
    # The files that exist are still listed, and they did not change.
    assert sorted(file.name for file in ledger.files()) == sorted(names)
    assert all(len(read_lines(tmp_path / name)) == 1 for name in names)


def test_paid_answer_is_still_recorded_when_a_ledger_file_went_missing(tmp_path: Path) -> None:
    """The line of an answer that was paid must not be lost. The next total then reports the broken series."""
    (tmp_path / "ledger-0002.jsonl").write_text(json.dumps({"run_id": "run-1", "usd": 51.0}) + "\n", encoding="utf-8")
    ledger = Ledger(tmp_path / "ledger.jsonl")

    assert ledger.record("run-2", make_request(), make_response()) == pytest.approx(0.017)

    assert [line["run_id"] for line in read_lines(tmp_path / "ledger-0002.jsonl")] == ["run-1", "run-2"]
    with pytest.raises(ValueError, match="not a complete series"):
        ledger.total_usd()


@pytest.mark.parametrize(
    ("name", "number"),
    [
        ("ledger.jsonl", 1),
        ("ledger-0002.jsonl", 2),
        ("ledger-9.jsonl", 9),
        ("ledger-0010.jsonl", 10),
        ("ledger-12345.jsonl", 12345),
        ("ledger.jsonl.bak", None),
        ("ledger-notes.jsonl", None),
        ("ledger-.jsonl", None),
        ("ledger--2.jsonl", None),
        ("old-ledger-0002.jsonl", None),
        ("ledger-0003.json", None),
        ("Ledger.jsonl", None),
        ("ledgerx0002.jsonl", None),
    ],
)
def test_number_of_a_ledger_file_comes_from_its_name(name: str, number: int | None) -> None:
    assert ledger_file_number("ledger.jsonl", name) == number


def test_number_of_a_ledger_file_follows_the_name_of_the_first_file() -> None:
    assert ledger_file_number("costs.v1.jsonl", "costs.v1-0002.jsonl") == 2
    assert ledger_file_number("costs.v1.jsonl", "costs.v1.jsonl") == 1
    # A dot in the name of the first file stands for a dot only.
    assert ledger_file_number("costs.v1.jsonl", "costsxv1-0002.jsonl") is None
    assert ledger_file_number("costs.v1.jsonl", "ledger-0002.jsonl") is None


def test_ledger_without_a_folder_has_no_file_and_costs_nothing(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "missing" / "cache" / "ledger.jsonl")

    assert ledger.files() == []
    assert ledger.total_usd() == 0.0
    assert ledger.run_usd("run-1") == 0.0
    assert list(tmp_path.iterdir()) == []


def test_damaged_line_in_a_later_ledger_file_is_an_error_that_names_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    limit_to_three_lines(tmp_path, monkeypatch)
    ledger = Ledger(tmp_path / "cache" / "ledger.jsonl")
    for frame in range(5):
        ledger.record("run-1", make_request(frame=frame), make_response())
    second = ledger.files()[1]
    with second.open("a", encoding="utf-8") as stream:
        stream.write("{broken\n")

    with pytest.raises(ValueError, match=r"ledger-0002\.jsonl: line 3 is not valid JSON"):
        ledger.total_usd()


def test_no_ledger_file_can_pass_the_size_limit_of_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """A full ledger file has at most half of the size that the repository guard allows.

    So every ledger file can go to the results branch, however many answers the ledger holds. The
    margin covers the few lines by which processes that write at the same moment can pass the limit.
    """
    spec = importlib.util.spec_from_file_location("repository_guard", GUARD)
    assert spec is not None and spec.loader is not None
    guard = importlib.util.module_from_spec(spec)
    # The guard defines data classes, and they look their module up by its name while it is loaded.
    monkeypatch.setitem(sys.modules, spec.name, guard)
    spec.loader.exec_module(guard)

    assert MAX_LEDGER_FILE_BYTES == 1_000_000
    assert 2 * MAX_LEDGER_FILE_BYTES <= guard.MAX_FILE_BYTES


def test_ledger_of_the_whole_budget_needs_several_files_and_its_lines_pass_the_guard(
    private_git: None, tmp_path: Path
) -> None:
    """The hard stop is 135 USD. The ledger of that many planned calls must still go to the results branch.

    One clip of the plan is 179 answers for 2.3448 USD, so the hard stop is reached in the 58th
    clip. The ledger lines of 57 clips alone have more bytes than the repository guard allows in
    one file. So the ledger must be split, and the guard must accept the content of its lines.
    """
    tree = tmp_path / "pitchvlm-results"
    ledger = Ledger(tree / "results" / "cache" / "ledger.jsonl")
    run_id = "run-SNGS-021-20261120T100000Z-ab12"
    cost_of_one_clip = sum(ledger.record(run_id, request, response) for request, response in plan_of_one_clip())

    assert cost_of_one_clip == pytest.approx(2.3448)
    assert 57 * cost_of_one_clip < 135.0 < 58 * cost_of_one_clip
    assert ledger.files() == [ledger.path]
    assert len(read_lines(ledger.path)) == 179
    assert 57 * ledger.path.stat().st_size > 2_000_000
    # The guard as the sync command runs it: it refuses no ledger line for its content. The work tree has
    # a deny-list without entries, as a clone that is set up has one, so the guard prints no notice.
    subprocess.run(["git", "init", "--quiet", "--template=", str(tree)], check=True)
    (tree / ".git" / "guard-deny.txt").write_text("# This test has no private word.\n", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(GUARD), "--repo", str(tree), "--paths", str(ledger.path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (completed.returncode, completed.stdout, completed.stderr) == (0, "", "")


def test_ledger_lines_pass_the_secret_scan(tmp_path: Path) -> None:
    """Every line holds the cache key of its request: 32 hex digits. The secret scan must not take it for a key."""
    if shutil.which("gitleaks") is None:
        pytest.skip("gitleaks is not installed")
    ledger = Ledger(tmp_path / "results" / "cache" / "ledger.jsonl")
    for request, response in plan_of_one_clip():
        ledger.record("run-SNGS-021-20261120T100000Z-ab12", request, response)
    options = ["--config", str(SECRET_SCAN_FILE), "--redact", "--no-banner", "--no-color", "--verbose"]

    completed = subprocess.run(
        ["gitleaks", "dir", *options, str(tmp_path / "results")], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stdout


# --- Ledger: a file that cannot be written --------------------------------------


def test_check_writable_creates_nothing(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "results" / "cache" / "ledger.jsonl")

    ledger.check_writable()
    BudgetGuard(ledger).start_run("run-1", 1.0)

    assert list(tmp_path.iterdir()) == []
    # A ledger with lines passes the check too.
    ledger.record("run-1", make_request(), make_response())
    ledger.check_writable()
    assert len(read_lines(ledger.path)) == 1


def test_run_does_not_start_when_the_folder_of_the_ledger_cannot_be_written(read_only_folder: Path) -> None:
    """A paid answer whose cost cannot be recorded is invisible to the budget guard, so no call may be made."""
    guard = BudgetGuard(Ledger(read_only_folder / "cache" / "ledger.jsonl"))

    with pytest.raises(PermissionError, match="the folder of the cost ledger cannot be written") as excinfo:
        guard.start_run("run-1", 1.0)

    assert excinfo.value.filename == str(read_only_folder)
    # The run is not registered.
    with pytest.raises(ValueError, match="start_run"):
        guard.check("run-1")
    assert list(read_only_folder.iterdir()) == []


def test_run_does_not_start_when_the_ledger_file_cannot_be_written(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.record("run-1", make_request(), make_response())
    path.chmod(0o444)
    try:
        if os.access(path, os.W_OK):
            pytest.skip("this user can write into a file without the write permission")
        with pytest.raises(PermissionError, match="the cost ledger cannot be written") as excinfo:
            BudgetGuard(ledger).start_run("run-2", 1.0)
    finally:
        path.chmod(0o644)

    assert excinfo.value.filename == str(path)
    assert ledger.total_usd() == pytest.approx(0.017)


def test_check_writable_refuses_a_folder_in_the_place_of_the_ledger_file(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.mkdir()

    with pytest.raises(OSError, match="the cost ledger cannot be written"):
        Ledger(path).check_writable()


# --- BudgetGuard ----------------------------------------------------------------


def test_budget_exceeded_is_a_runtime_error() -> None:
    assert issubclass(BudgetExceeded, RuntimeError)


def test_run_stops_above_1_3_times_its_estimate(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    guard = BudgetGuard(ledger)
    guard.start_run("run-1", 1.0)

    guard.check("run-1")  # nothing spent
    spend(ledger, "run-1", 1.0)
    guard.check("run-1")  # at the estimate
    spend(ledger, "run-1", 0.29)
    guard.check("run-1")  # 1.29 USD is not above 1.3 x 1.0 USD
    spend(ledger, "run-1", 0.02)
    with pytest.raises(BudgetExceeded):
        guard.check("run-1")  # 1.31 USD is above


def test_run_limit_uses_the_run_factor(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    guard = BudgetGuard(ledger, run_factor=2.0)
    guard.start_run("run-1", 1.0)

    spend(ledger, "run-1", 1.9)
    guard.check("run-1")
    spend(ledger, "run-1", 0.2)
    with pytest.raises(BudgetExceeded):
        guard.check("run-1")


def test_run_limit_counts_only_the_run(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    guard = BudgetGuard(ledger)
    guard.start_run("run-1", 1.0)
    guard.start_run("run-2", 1.0)

    spend(ledger, "run-2", 50.0)

    guard.check("run-1")
    with pytest.raises(BudgetExceeded):
        guard.check("run-2")


def test_stop_of_a_run_names_the_run_the_factor_and_the_estimate(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    guard = BudgetGuard(ledger)
    guard.start_run("x", 1.0)

    spend(ledger, "x", 1.31)
    with pytest.raises(BudgetExceeded) as stop:
        guard.check("x")

    assert str(stop.value) == "the run 'x' spent 1.31 USD, more than 1.3 times its estimate of 1.00 USD"


def test_run_with_its_own_spend_limit_stops_above_that_limit_and_not_above_its_estimate(tmp_path: Path) -> None:
    """The contract checks of gate G0 have a spend limit in the gate file. Their estimate does not stop them."""
    ledger = Ledger(tmp_path / "ledger.jsonl")
    guard = BudgetGuard(ledger)
    guard.start_run("x", 0.1, limit_usd=2.0)

    spend(ledger, "x", 1.0)
    guard.check("x")  # 10 times the estimate, and below the spend limit
    spend(ledger, "x", 1.0)
    guard.check("x")  # 2.0 USD is not above the spend limit of 2.0 USD
    spend(ledger, "x", 0.01)
    with pytest.raises(BudgetExceeded) as stop:
        guard.check("x")

    assert str(stop.value) == "the run 'x' spent 2.010 USD, more than its spend limit of 2.000 USD"


def test_stop_at_a_spend_limit_shows_an_amount_that_is_less_than_a_cent_above_it(tmp_path: Path) -> None:
    """A run stops after the answer that passes its limit. With two decimals both amounts would read 2.00 USD."""
    ledger = Ledger(tmp_path / "ledger.jsonl")
    guard = BudgetGuard(ledger)
    guard.start_run("x", 0.1, limit_usd=2.0)

    spend(ledger, "x", 2.0014)
    with pytest.raises(BudgetExceeded, match=r"spent 2\.001 USD, more than its spend limit of 2\.000 USD$"):
        guard.check("x")


def test_spend_limit_below_the_estimate_stops_the_run_at_that_limit(tmp_path: Path) -> None:
    """The spend limit replaces the limit from the estimate. It is not the larger of the two."""
    ledger = Ledger(tmp_path / "ledger.jsonl")
    guard = BudgetGuard(ledger)
    guard.start_run("run-1", 10.0, limit_usd=1.0)

    spend(ledger, "run-1", 1.01)
    with pytest.raises(BudgetExceeded, match=r"spent 1\.010 USD, more than its spend limit of 1\.000 USD$"):
        guard.check("run-1")


def test_spend_limit_of_a_run_counts_only_the_run(tmp_path: Path) -> None:
    """The ledger holds earlier runs when the contract checks start again. Their spend is not the spend of the run."""
    ledger = Ledger(tmp_path / "ledger.jsonl")
    spend(ledger, "an-earlier-run", 50.0)
    guard = BudgetGuard(ledger)
    guard.start_run("checks", 0.1, limit_usd=2.0)

    guard.check("checks")  # 50 USD in the ledger, and the run spent nothing
    spend(ledger, "checks", 2.0)
    guard.check("checks")  # 2.0 USD is not above the spend limit of 2.0 USD
    spend(ledger, "checks", 0.01)
    with pytest.raises(BudgetExceeded, match=r"spent 2\.010 USD, more than its spend limit of 2\.000 USD$"):
        guard.check("checks")


def test_hard_stop_holds_for_a_run_with_its_own_spend_limit(tmp_path: Path) -> None:
    """The hard stop is the limit of all runs together. The spend limit of a run does not replace it."""
    ledger = Ledger(tmp_path / "ledger.jsonl")
    spend(ledger, "an-earlier-run", 4.9)
    guard = BudgetGuard(ledger, hard_stop_usd=5.0)
    guard.start_run("checks", 0.1, limit_usd=2.0)

    guard.check("checks")  # 4.9 USD in total
    spend(ledger, "checks", 0.2)
    with pytest.raises(BudgetExceeded, match=r"hard stop: the ledger total is 5\.10 USD"):
        guard.check("checks")  # 5.1 USD in total, and the run is far below its spend limit


def test_spend_limit_of_one_run_is_not_the_limit_of_another_run(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    guard = BudgetGuard(ledger)
    guard.start_run("checks", 0.1, limit_usd=2.0)
    guard.start_run("run-2", 0.1)

    spend(ledger, "checks", 1.0)
    spend(ledger, "run-2", 0.14)

    guard.check("checks")
    with pytest.raises(BudgetExceeded, match="more than 1.3 times its estimate of 0.10 USD"):
        guard.check("run-2")


@pytest.mark.parametrize("limit_usd", [0.0, -1.0, float("inf"), float("nan")])
def test_spend_limit_of_a_run_must_be_a_number_above_zero(tmp_path: Path, limit_usd: float) -> None:
    guard = BudgetGuard(Ledger(tmp_path / "ledger.jsonl"))

    with pytest.raises(ValueError, match="spend limit"):
        guard.start_run("run-1", 1.0, limit_usd=limit_usd)
    # The run was not started.
    with pytest.raises(ValueError, match="start_run"):
        guard.check("run-1")


@pytest.mark.parametrize("estimate", [-0.01, float("nan"), float("inf")])
def test_start_run_refuses_a_wrong_estimate_also_with_a_spend_limit(tmp_path: Path, estimate: float) -> None:
    guard = BudgetGuard(Ledger(tmp_path / "ledger.jsonl"))

    with pytest.raises(ValueError, match="estimate"):
        guard.start_run("run-1", estimate, limit_usd=2.0)
    # The run was not started.
    with pytest.raises(ValueError, match="start_run"):
        guard.check("run-1")


def test_run_with_an_estimate_of_zero_may_spend_nothing(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    guard = BudgetGuard(ledger)
    guard.start_run("replay", 0.0)

    guard.check("replay")
    spend(ledger, "replay", 0.01)
    with pytest.raises(BudgetExceeded):
        guard.check("replay")


def test_hard_stop_counts_every_run_in_the_ledger(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    guard = BudgetGuard(ledger, hard_stop_usd=5.0)
    guard.start_run("run-2", 100.0)

    spend(ledger, "run-1", 4.9)
    guard.check("run-2")  # 4.9 USD in total, and run-2 spent nothing
    spend(ledger, "run-1", 0.2)
    with pytest.raises(BudgetExceeded):
        guard.check("run-2")  # 5.1 USD in total


def test_hard_stop_is_reached_at_the_exact_total(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    spend(ledger, "run-1", 3.0)
    guard = BudgetGuard(ledger, hard_stop_usd=ledger.total_usd())
    guard.start_run("run-1", 100.0)

    with pytest.raises(BudgetExceeded):
        guard.check("run-1")


def test_default_hard_stop_is_135_usd(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    guard = BudgetGuard(ledger)
    guard.start_run("run-1", 1000.0)

    spend(ledger, "run-1", 134.0)
    guard.check("run-1")
    spend(ledger, "run-1", 2.0)
    with pytest.raises(BudgetExceeded):
        guard.check("run-1")


def test_guard_counts_the_spend_of_an_earlier_session(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    spend(Ledger(path), "run-1", 4.0)

    guard = BudgetGuard(Ledger(path), hard_stop_usd=3.0)
    guard.start_run("run-2", 1.0)

    with pytest.raises(BudgetExceeded):
        guard.check("run-2")


def test_check_needs_a_started_run(tmp_path: Path) -> None:
    guard = BudgetGuard(Ledger(tmp_path / "ledger.jsonl"))

    with pytest.raises(ValueError):
        guard.check("run-1")


@pytest.mark.parametrize("estimate", [-0.01, float("nan"), float("inf")])
def test_start_run_refuses_a_wrong_estimate(tmp_path: Path, estimate: float) -> None:
    guard = BudgetGuard(Ledger(tmp_path / "ledger.jsonl"))

    with pytest.raises(ValueError):
        guard.start_run("run-1", estimate)


@pytest.mark.parametrize(("hard_stop_usd", "run_factor"), [(0.0, 1.3), (-1.0, 1.3), (135.0, 0.0), (135.0, -1.0)])
def test_guard_refuses_limits_that_are_not_positive(tmp_path: Path, hard_stop_usd: float, run_factor: float) -> None:
    with pytest.raises(ValueError):
        BudgetGuard(Ledger(tmp_path / "ledger.jsonl"), hard_stop_usd=hard_stop_usd, run_factor=run_factor)
