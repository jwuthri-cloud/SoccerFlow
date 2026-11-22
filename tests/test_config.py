"""Tests for the pipeline settings of ``pitchvlm.config``."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pitchvlm.config import (
    DEFAULT_MODEL,
    MAX_PARALLEL_LIMIT,
    THINKING_LEVELS,
    PipelineConfig,
    StepSettings,
    TrackingConfig,
    load_config,
)
from pitchvlm.frames import keyframe_indices
from pitchvlm.vlm.cost import PRICES

OVERRIDE_YAML = """\
keyframe_interval_s: 0.96
max_parallel: 2
repeat: 1
people:
  model: gemini-3.8-flash
  thinking_level: medium
  resolution: ultra_high
  service_tier: flex
tracking:
  max_speed_mps: 8.0
  min_observations: 3
"""


def write_settings(tmp_path: Path, text: str) -> Path:
    """Write a YAML settings file into the test folder and return its path."""
    path = tmp_path / "settings.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# --- Defaults ----------------------------------------------------------------------------


def test_load_config_without_a_file_gives_the_defaults() -> None:
    """No path gives the default settings of the first-result design."""
    config = load_config()
    assert config == PipelineConfig()
    assert config.keyframe_interval_s == 0.48
    assert config.number_crops == 8
    assert config.crop_upscale == 4.0
    assert config.ransac_px == 8.0
    assert config.max_gap_frames == 75
    assert config.pitch_margin_m == 3.0
    assert config.max_parallel == 4
    assert config.repeat == 0


def test_default_step_settings() -> None:
    """Every step uses the default model at thinking level low. The number step sends low resolution."""
    config = load_config()
    assert DEFAULT_MODEL == "gemini-4-argon"
    expected = StepSettings(model=DEFAULT_MODEL, thinking_level="low", resolution="high", service_tier="standard")
    assert StepSettings() == expected
    for step in (config.kit_legend, config.people, config.landmarks):
        assert step == expected
    assert config.number == StepSettings(resolution="low")
    assert config.number.model == DEFAULT_MODEL
    assert config.number.thinking_level == "low"


def test_default_tracking_limits() -> None:
    """The default tracking limits are the ones of the design."""
    tracking = load_config().tracking
    assert tracking == TrackingConfig()
    assert tracking.max_speed_mps == 9.0
    assert tracking.reach_margin_m == 2.0
    assert tracking.max_image_jump == 2.5
    assert tracking.mismatch_penalty == 0.5
    assert tracking.max_missed_keyframes == 4
    assert tracking.max_stitch_gap_s == 3.0
    assert tracking.stitch_speed_mps == 4.0
    assert tracking.min_observations == 2
    # The two track repairs are arms of gate G4: off until G4 has measured them.
    assert tracking.split_team_swaps is False
    assert tracking.team_slots is False


ARMS_FOLDER = Path(__file__).resolve().parents[1] / "configs" / "arms"


@pytest.mark.parametrize(
    ("name", "split_team_swaps", "team_slots"),
    [
        ("g4_split_team_swaps.yaml", True, False),
        ("g4_team_slots.yaml", False, True),
        ("g4_both_repairs.yaml", True, True),
    ],
)
def test_each_arm_of_gate_g4_changes_only_its_track_repairs(name: str, split_team_swaps: bool, team_slots: bool) -> None:
    """An arm file is the default settings with its repairs switched on, and nothing else."""
    config = load_config(ARMS_FOLDER / name)
    repairs = {"split_team_swaps": split_team_swaps, "team_slots": team_slots}
    expected = PipelineConfig().model_copy(update={"tracking": TrackingConfig(**repairs)})
    assert config == expected


def test_the_arms_folder_holds_the_three_arms_of_gate_g4() -> None:
    names = sorted(path.name for path in ARMS_FOLDER.iterdir())
    assert names == ["g4_both_repairs.yaml", "g4_split_team_swaps.yaml", "g4_team_slots.yaml"]


def test_default_interval_gives_64_keyframes_for_a_30_second_clip() -> None:
    """The default interval gives the 64 keyframes per 750-frame clip that the budget assumes."""
    assert len(keyframe_indices(750, 25.0, load_config().keyframe_interval_s)) == 64


# --- YAML file -----------------------------------------------------------------------------


def test_yaml_file_overrides_the_defaults(tmp_path: Path) -> None:
    """Values from the file replace the defaults, also inside a nested section."""
    config = load_config(write_settings(tmp_path, OVERRIDE_YAML))
    assert config.keyframe_interval_s == 0.96
    assert config.max_parallel == 2
    assert config.repeat == 1
    assert config.people == StepSettings(
        model="gemini-3.8-flash", thinking_level="medium", resolution="ultra_high", service_tier="flex"
    )
    assert config.tracking.max_speed_mps == 8.0
    assert config.tracking.min_observations == 3


def test_yaml_file_keeps_the_other_defaults(tmp_path: Path) -> None:
    """Settings that the file does not name keep their default values."""
    config = load_config(write_settings(tmp_path, OVERRIDE_YAML))
    defaults = PipelineConfig()
    assert config.kit_legend == defaults.kit_legend
    assert config.landmarks == defaults.landmarks
    assert config.number == defaults.number
    assert config.number_crops == defaults.number_crops
    assert config.ransac_px == defaults.ransac_px
    assert config.max_gap_frames == defaults.max_gap_frames
    assert config.pitch_margin_m == defaults.pitch_margin_m
    assert config.tracking.reach_margin_m == defaults.tracking.reach_margin_m
    assert config.tracking.max_image_jump == defaults.tracking.max_image_jump


def test_partial_step_section_keeps_the_other_defaults_of_that_step(tmp_path: Path) -> None:
    """A section that names one key keeps the defaults of its other keys.

    The number step has its own default resolution (low). A file that changes only the model of
    the number step must not move the step to the resolution of a plain step (high).
    """
    text = "number:\n  model: gemini-3.8-flash\npeople:\n  thinking_level: high\n"
    config = load_config(write_settings(tmp_path, text))
    assert config.number == StepSettings(model="gemini-3.8-flash", resolution="low")
    assert config.number.resolution == "low"
    assert config.people == StepSettings(thinking_level="high")
    assert config.kit_legend == StepSettings()
    assert config.landmarks == StepSettings()


def test_partial_tracking_section_keeps_the_other_limits(tmp_path: Path) -> None:
    """A tracking section that names one limit keeps the defaults of the other limits."""
    config = load_config(write_settings(tmp_path, "tracking:\n  max_missed_keyframes: 2\n"))
    assert config.tracking == TrackingConfig(max_missed_keyframes=2)


def test_stitch_gap_can_be_set_in_the_tracking_section(tmp_path: Path) -> None:
    """The longest gap that is joined and filled is a setting, in seconds."""
    config = load_config(write_settings(tmp_path, "tracking:\n  max_stitch_gap_s: 1.5\n"))
    assert config.tracking == TrackingConfig(max_stitch_gap_s=1.5)
    assert config.tracking.max_missed_keyframes == TrackingConfig().max_missed_keyframes


def test_stitch_speed_can_be_set_in_the_tracking_section(tmp_path: Path) -> None:
    """The speed that limits a join of two fragments is a setting, in metres per second."""
    config = load_config(write_settings(tmp_path, "tracking:\n  stitch_speed_mps: 6.5\n"))
    assert config.tracking == TrackingConfig(stitch_speed_mps=6.5)
    assert config.tracking.max_speed_mps == TrackingConfig().max_speed_mps


def test_number_step_can_be_set_to_another_resolution(tmp_path: Path) -> None:
    """The low resolution of the number step is a default only: the file can change it."""
    config = load_config(write_settings(tmp_path, "number:\n  resolution: medium\n"))
    assert config.number == StepSettings(resolution="medium")


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty file"),
        pytest.param("# nothing set\n", id="comments only"),
        pytest.param("{}\n", id="empty mapping"),
    ],
)
def test_file_without_settings_gives_the_defaults(tmp_path: Path, text: str) -> None:
    """A file that sets nothing gives the default settings."""
    assert load_config(write_settings(tmp_path, text)) == PipelineConfig()


def test_unknown_top_level_key_is_an_error(tmp_path: Path) -> None:
    """A misspelled setting is an error that names the key."""
    path = write_settings(tmp_path, "keyframe_every_s: 0.5\n")
    with pytest.raises(ValueError, match="keyframe_every_s"):
        load_config(path)


def test_unknown_key_inside_a_section_is_an_error(tmp_path: Path) -> None:
    """A misspelled key inside a step section is an error that names the key."""
    path = write_settings(tmp_path, "people:\n  modle: gemini-3.8-flash\n")
    with pytest.raises(ValueError, match="modle"):
        load_config(path)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("keyframe_interval_s: 0\n", id="interval not above zero"),
        pytest.param("max_parallel: 0\n", id="no parallel request"),
        pytest.param("max_parallel: 9\n", id="more parallel requests than the limit"),
        pytest.param("repeat: -1\n", id="negative repeat"),
        pytest.param("number_crops: 0\n", id="no crops"),
        pytest.param("people:\n  resolution: huge\n", id="unknown resolution"),
        pytest.param("people:\n  thinking_level: extreme\n", id="unknown thinking level"),
        pytest.param("landmarks:\n  service_tier: batch\n", id="unknown service tier"),
        pytest.param("tracking:\n  max_speed_mps: -1\n", id="negative speed"),
        pytest.param("tracking:\n  max_stitch_gap_s: 0\n", id="stitch gap not above zero"),
        pytest.param("tracking:\n  max_stitch_gap_s: .inf\n", id="stitch gap without an end"),
        pytest.param("tracking:\n  max_stitch_gap_s: .nan\n", id="stitch gap that is not a number"),
        pytest.param("tracking:\n  stitch_speed_mps: 0\n", id="stitch speed not above zero"),
        pytest.param("tracking:\n  stitch_speed_mps: .inf\n", id="stitch speed without an end"),
        pytest.param("people:\n  thinking_level: minimal\n", id="thinking level that the default model does not have"),
        pytest.param(
            "people:\n  model: gemini-3.8-flash\n  thinking_level: xhigh\n", id="thinking level of another model"
        ),
        pytest.param("tracking: fast\n", id="section is not a mapping"),
    ],
)
def test_invalid_value_is_an_error(tmp_path: Path, text: str) -> None:
    """A value outside its allowed range or list is an error."""
    with pytest.raises(ValueError):
        load_config(write_settings(tmp_path, text))


def test_number_of_parallel_requests_has_an_upper_limit(tmp_path: Path) -> None:
    """A budget stop lets the requests in flight finish, so their number limits what a run can spend too much."""
    assert MAX_PARALLEL_LIMIT == 8
    path = tmp_path / "settings.yaml"
    path.write_text("max_parallel: 8\n", encoding="utf-8")
    assert load_config(path).max_parallel == 8
    assert PipelineConfig(max_parallel=8).max_parallel == 8
    with pytest.raises(ValidationError, match="max_parallel"):
        PipelineConfig(max_parallel=9)
    with pytest.raises(ValidationError, match="max_parallel"):
        PipelineConfig().model_copy(update={"max_parallel": 16})


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("- keyframe_interval_s\n- 0.5\n", id="list"),
        pytest.param("[]\n", id="empty list"),
        pytest.param("0\n", id="zero"),
        pytest.param("false\n", id="false"),
        pytest.param('""\n', id="empty text"),
        pytest.param("settings\n", id="one word"),
    ],
)
def test_top_level_value_that_is_not_a_mapping_is_an_error(tmp_path: Path, text: str) -> None:
    """The file must hold a mapping at the top level.

    A list, a number, a truth value or a text is an error, also when Python counts the value as false.
    Only a file without any content gives the defaults.
    """
    with pytest.raises(ValueError, match="mapping"):
        load_config(write_settings(tmp_path, text))


def test_missing_file_is_an_error(tmp_path: Path) -> None:
    """A path to a file that does not exist is an error."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


# --- Thinking levels -------------------------------------------------------------------------


def test_thinking_level_table_names_every_priced_model() -> None:
    """Every model with a price has its list of thinking levels, and no other model has one."""
    assert set(THINKING_LEVELS) == {price.model for price in PRICES}
    assert THINKING_LEVELS == {
        "gemini-4-argon": ("low", "medium", "high"),
        "gemini-4-carbon": ("low", "medium", "high"),
        "gemini-4-hydrogen": ("low", "medium", "high", "xhigh"),
        "gemini-3.8-flash": ("low", "medium", "high"),
        "gemini-robotics-er-2-preview": ("minimal", "low", "medium", "high"),
    }


def test_default_thinking_level_is_valid_for_every_priced_model() -> None:
    """The default level is one that every priced model has, so a file that changes only the model stays valid."""
    for model in THINKING_LEVELS:
        assert StepSettings(model=model).thinking_level == "low"


@pytest.mark.parametrize(
    ("model", "level"),
    [(model, level) for model, levels in THINKING_LEVELS.items() for level in levels],
)
def test_documented_thinking_level_is_accepted(model: str, level: str) -> None:
    assert StepSettings(model=model, thinking_level=level).thinking_level == level


@pytest.mark.parametrize(
    ("model", "level"),
    [
        ("gemini-4-argon", "minimal"),
        ("gemini-4-argon", "xhigh"),
        ("gemini-4-carbon", "minimal"),
        ("gemini-4-carbon", "xhigh"),
        ("gemini-4-hydrogen", "minimal"),
        ("gemini-3.8-flash", "minimal"),
        ("gemini-3.8-flash", "xhigh"),
        ("gemini-robotics-er-2-preview", "xhigh"),
    ],
)
def test_thinking_level_that_a_model_does_not_have_is_refused(model: str, level: str) -> None:
    """The error names the model, the level and the levels that the model has."""
    with pytest.raises(ValidationError) as caught:
        StepSettings(model=model, thinking_level=level)
    message = str(caught.value)
    assert model in message
    assert f"'{level}'" in message
    assert ", ".join(THINKING_LEVELS[model]) in message


def test_thinking_level_is_checked_in_a_changed_copy_and_in_a_settings_file(tmp_path: Path) -> None:
    hydrogen = StepSettings(model="gemini-4-hydrogen", thinking_level="xhigh")
    with pytest.raises(ValidationError, match="gemini-4-argon"):
        hydrogen.model_copy(update={"model": "gemini-4-argon"})
    with pytest.raises(ValueError, match="its levels are low, medium, high"):
        load_config(write_settings(tmp_path, "landmarks:\n  thinking_level: xhigh\n"))
    config = load_config(write_settings(tmp_path, "landmarks:\n  model: gemini-4-hydrogen\n  thinking_level: xhigh\n"))
    assert config.landmarks == hydrogen


def test_model_without_a_level_list_is_not_checked() -> None:
    """A model that the table does not name takes every level: its levels are not known here."""
    for level in ("minimal", "low", "medium", "high", "xhigh"):
        assert StepSettings(model="some-later-model", thinking_level=level).thinking_level == level


# --- Records ---------------------------------------------------------------------------------


def test_settings_are_frozen() -> None:
    """Settings cannot be assigned. A changed copy is made with ``model_copy``."""
    config = PipelineConfig()
    with pytest.raises(ValidationError, match="frozen"):
        config.repeat = 1
    with pytest.raises(ValidationError, match="frozen"):
        config.people.model = "other-model"
    repeated = config.model_copy(update={"repeat": 1})
    assert repeated.repeat == 1
    assert config.repeat == 0


def test_changed_copy_of_the_settings_is_validated() -> None:
    """A changed copy is checked like new settings: a wrong value or a misspelled name is an error."""
    config = PipelineConfig()
    with pytest.raises(ValidationError):
        config.model_copy(update={"repeat": -1})
    with pytest.raises(ValidationError):
        config.model_copy(update={"repeats": 1})
    with pytest.raises(ValidationError):
        config.number.model_copy(update={"resolution": "huge"})
    faster = config.model_copy(update={"people": config.people.model_copy(update={"service_tier": "flex"})})
    assert faster.people == StepSettings(service_tier="flex")
    assert faster.number == config.number


def test_unknown_setting_is_rejected_in_code() -> None:
    """Unknown fields are an error, so a step cannot carry a sampling parameter."""
    with pytest.raises(ValidationError):
        StepSettings(temperature=0.2)
    with pytest.raises(ValidationError):
        TrackingConfig(max_speed=9.0)
    with pytest.raises(ValidationError):
        PipelineConfig(keyframe_every_s=0.5)
