"""Tests of the wrapper for the stand-alone evaluator.

The real evaluator is not installed for the tests. A small script, written
into the test folder, stands in for it. It follows the source of
``sn-trackeval`` 0.4.0 in the points that the wrapper depends on: the flags
and how their values are read, the folders where the labels and the
predictions are looked for, and the result files with their names, rows and
columns. Its scores are made up.

The label files and the prediction files of the tests lie where this package
puts them: ``<root>/SoccerNetGS-<split>/<clip>/Labels-GameState.json`` and
``<root>/SoccerNetGS-<split>/<tracker>/<clip>.json``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from pitchvlm.evalx.scorer import parse_detailed_csv, run_trackeval, trackeval_command
from pitchvlm.export_gsr import predictions_path

CLIPS = ["SNGS-021", "SNGS-022"]
#: Folder of the clips of the split "valid", below the test folder.
SPLIT_FOLDER = Path("gt") / "SoccerNetGS-valid"
#: Folder of the prediction files of the tracker "pitchvlm" in the split "valid", below the test folder.
TRACKER_FOLDER = Path("pred") / "SoccerNetGS-valid" / "pitchvlm"
#: The detailed CSV files that the evaluator writes for the one class of the task and for its two class averages.
DETAILED_FILES = ["cls_comb_cls_av_detailed.csv", "cls_comb_det_av_detailed.csv", "person_detailed.csv"]

#: A detailed CSV file as the evaluator writes it: shares from 0 to 1, more columns than the wrapper needs.
DETAILED_CSV = """\
seq,HOTA___5,HOTA___50,HOTA___AUC,DetA___AUC,AssA___AUC,Dets
SNGS-021,0.9,0.7,0.5,0.6,0.4,100
SNGS-022,0.8,0.5,0.25,0.5,0.125,50
COMBINED,0.85,0.6,0.375,0.55,0.2625,150
"""

#: Stand-in for scripts/run_soccernet_gs.py. It reads its flags and looks for the label files and the
#: prediction files as the evaluator does. Then it writes the result files of the evaluator, without the
#: plot files, with made-up scores.
FAKE_EVALUATOR = """\
import argparse
import csv
import os

# Every setting of the runner script with its default value: the settings of the evaluator, of the
# game-state data and of the metrics.
DEFAULTS = {
    "USE_PARALLEL": False,
    "NUM_PARALLEL_CORES": 8,
    "BREAK_ON_ERROR": True,
    "RETURN_ON_ERROR": False,
    "LOG_ON_ERROR": "error_log.txt",
    "PRINT_RESULTS": True,
    "PRINT_ONLY_COMBINED": False,
    "PRINT_CONFIG": True,
    "TIME_PROGRESS": True,
    "DISPLAY_LESS_PROGRESS": True,
    "OUTPUT_SUMMARY": True,
    "OUTPUT_EMPTY_CLASSES": True,
    "OUTPUT_DETAILED": True,
    "PLOT_CURVES": True,
    "GT_FOLDER": "data/SoccerNetGS",
    "TRACKERS_FOLDER": "data/trackers/SoccerNetGS/",
    "OUTPUT_FOLDER": None,
    "TRACKERS_TO_EVAL": None,
    "SPLIT_TO_EVAL": "valid",
    "EVAL_MODE": "distance",
    "EVAL_SPACE": "pitch",
    "EVAL_SIMILARITY_METRIC": "gaussian",
    "EVAL_DIST_TOL": 5,
    "USE_ROLES": True,
    "USE_TEAMS": True,
    "USE_JERSEY_NUMBERS": True,
    "IGNORE_BALL": True,
    "INPUT_AS_ZIP": False,
    "DO_PREPROC": True,
    "TRACKER_SUB_FOLDER": "data",
    "OUTPUT_SUB_FOLDER": "",
    "TRACKER_DISPLAY_NAMES": None,
    "SEQMAP_FOLDER": None,
    "SEQMAP_FILE": None,
    "SEQ_INFO": None,
    "GT_LOC_FORMAT": "{gt_folder}/{seq}/Labels-GameState.json",
    "SKIP_SPLIT_FOL": False,
    "METRICS": ["HOTA", "Identity"],
    "THRESHOLD": 0.5,
}

# A setting whose default value is None or a list takes one or more values, so it arrives as a list.
# A true or false setting takes the text True or False, and the clip names become the keys of a mapping.
parser = argparse.ArgumentParser()
for setting, default in DEFAULTS.items():
    if default is None or isinstance(default, list):
        parser.add_argument("--" + setting, nargs="+")
    else:
        parser.add_argument("--" + setting)
config = dict(DEFAULTS)
for setting, value in vars(parser.parse_args()).items():
    if value is None:
        continue
    if isinstance(DEFAULTS[setting], bool):
        if value not in ("True", "False"):
            raise Exception("Command line parameter " + setting + "must be True or False")
        config[setting] = value == "True"
    elif isinstance(DEFAULTS[setting], int):
        config[setting] = int(value)
    elif setting == "SEQ_INFO":
        config[setting] = dict.fromkeys(value)
    else:
        config[setting] = value

# The folders of the data class. Only the prediction side gets the prefix SoccerNetGS- in its split folder.
split = config["SPLIT_TO_EVAL"]
gt_split, tracker_split = ("", "") if config["SKIP_SPLIT_FOL"] else (split, "SoccerNetGS-" + split)
gt_folder = os.path.join(config["GT_FOLDER"], gt_split)
tracker_folder = os.path.join(config["TRACKERS_FOLDER"], tracker_split)
output_folder = tracker_folder if config["OUTPUT_FOLDER"] is None else config["OUTPUT_FOLDER"]

# The wrapper always names the clips and the tracker, so the search for them is not modelled here.
clips = list(config["SEQ_INFO"])
trackers = config["TRACKERS_TO_EVAL"]
for clip in clips:
    label_file = config["GT_LOC_FORMAT"].format(gt_folder=gt_folder, seq=clip)
    if not os.path.isfile(label_file):
        print("GT file not found " + label_file)
        raise Exception("GT file not found for sequence: " + clip)
for tracker in trackers:
    for clip in clips:
        prediction_file = os.path.join(tracker_folder, tracker, config["TRACKER_SUB_FOLDER"], clip + ".json")
        if not os.path.isfile(prediction_file):
            print("Tracker file not found: " + prediction_file)
            raise Exception("Tracker file not found: " + prediction_file)

# The columns of a detailed CSV file after the name of the row: the fields of HOTA, Identity and Count.
COLUMNS = ["HOTA(0)", "LocA(0)", "HOTALocA(0)"]
HOTA_FIELDS = "HOTA DetA AssA DetRe DetPr AssRe AssPr LocA OWTA HOTA_TP HOTA_FN HOTA_FP"
for field in HOTA_FIELDS.split():
    COLUMNS += [field + "___" + str(alpha) for alpha in range(5, 100, 5)] + [field + "___AUC"]
COLUMNS += ["IDF1", "IDR", "IDP", "IDTP", "IDFN", "IDFP", "Dets", "GT_Dets", "IDs", "GT_IDs"]


def row(name, hota, det_a, ass_a):
    values = dict.fromkeys(COLUMNS, 0)
    values.update({"HOTA___AUC": hota, "DetA___AUC": det_a, "AssA___AUC": ass_a})
    return [name] + [values[column] for column in COLUMNS]


# The class person has one row per clip, in the order of the clip names, and the row COMBINED. The two
# class averages have the row COMBINED only. With one class it holds the scores of that class.
combined = row("COMBINED", 0.5, 0.6, 0.4)
person = [row(clip, 0.1 * number, 0.2 * number, 0.3 * number) for number, clip in enumerate(sorted(clips), start=1)]
person.append(combined)
tables = {"person": person, "cls_comb_cls_av": [combined], "cls_comb_det_av": [combined]}
for tracker in trackers:
    folder = os.path.join(output_folder, tracker, config["OUTPUT_SUB_FOLDER"])
    os.makedirs(folder, exist_ok=True)
    for name, rows in tables.items():
        with open(os.path.join(folder, name + "_summary.txt"), "w", newline="") as handle:
            handle.write("HOTA DetA AssA\\n50 60 40\\n")
        with open(os.path.join(folder, name + "_detailed.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["seq"] + COLUMNS)
            writer.writerows(rows)
print("fake evaluator done")
"""

#: An evaluator that stops with an error after it printed to both output streams.
FAILING_EVALUATOR = """\
import sys

print("reading the labels")
print("the labels are broken", file=sys.stderr)
sys.exit(3)
"""


def write_text(folder: Path, file_name: str, text: str) -> Path:
    """Write a text file into a folder and return its path."""
    path = folder / file_name
    path.write_text(text, encoding="utf-8")
    return path


def changed(script_text: str, old: str, new: str) -> str:
    """Return the script text with one part replaced. A part that is not in the text fails the test."""
    assert old in script_text
    return script_text.replace(old, new)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """The test folder, with a label file and a prediction file for every clip where this package puts them.

    The stand-in evaluator only checks that these files exist, so they hold no labels and no predictions.
    """
    for clip in CLIPS:
        label_file = tmp_path / SPLIT_FOLDER / clip / "Labels-GameState.json"
        label_file.parent.mkdir(parents=True)
        label_file.write_text("{}", encoding="utf-8")
        prediction_file = predictions_path(tmp_path / "pred", "valid", "pitchvlm", clip)
        prediction_file.parent.mkdir(parents=True, exist_ok=True)
        prediction_file.write_text('{"predictions": []}', encoding="utf-8")
    return tmp_path


def run_fake(root: Path, script_text: str, output_dir: Path | None = None) -> dict[str, dict[str, float]]:
    """Run ``run_trackeval`` with a stand-in script, in the interpreter of the tests."""
    return run_trackeval(
        Path(sys.executable),
        write_text(root, "fake_evaluator.py", script_text),
        root / "gt",
        root / "pred",
        split="valid",
        tracker="pitchvlm",
        clips=CLIPS,
        output_dir=root / "eval" if output_dir is None else output_dir,
    )


# --- trackeval_command ---------------------------------------------------------------


def test_trackeval_command_holds_the_script_and_the_flags_of_the_contract() -> None:
    python_exe = Path("envs") / "trackeval" / "bin" / "python"
    script = Path("sn-trackeval") / "scripts" / "run_soccernet_gs.py"
    gt_root = Path("data") / "gt"
    pred_root = Path("outputs") / "pred"

    command = trackeval_command(python_exe, script, gt_root, pred_root, split="valid", tracker="pitchvlm", clips=CLIPS)

    assert command == [
        str(python_exe),
        str(script),
        "--GT_FOLDER",
        str(gt_root / "SoccerNetGS-valid"),
        "--TRACKERS_FOLDER",
        str(pred_root / "SoccerNetGS-valid"),
        "--SKIP_SPLIT_FOL",
        "True",
        "--TRACKER_SUB_FOLDER",
        "",
        "--TRACKERS_TO_EVAL",
        "pitchvlm",
        "--SPLIT_TO_EVAL",
        "valid",
        "--SEQ_INFO",
        "SNGS-021",
        "SNGS-022",
    ]
    assert all(type(part) is str for part in command)
    # The evaluator would read an output folder as a list and stop, so the command names none.
    assert "--OUTPUT_FOLDER" not in command


def test_trackeval_command_names_the_split_folders_of_the_data_and_of_the_exporter() -> None:
    data_root, pred_root = Path("data"), Path("pred")

    command = trackeval_command(
        Path("python"), Path("script.py"), data_root, pred_root, split="test", tracker="pitchvlm", clips=CLIPS
    )

    # The labels are in the split folder of the data root, where the data command puts the clips.
    assert command[command.index("--GT_FOLDER") + 1] == str(data_root / "SoccerNetGS-test")
    # The predictions are where the exporter writes them: <split folder>/<tracker>/<clip>.json.
    tracker_folder = predictions_path(pred_root, "test", "pitchvlm", "SNGS-021").parent
    assert command[command.index("--TRACKERS_FOLDER") + 1] == str(tracker_folder.parent)
    # Both folders are split folders already, so the evaluator must add none.
    assert command[command.index("--SKIP_SPLIT_FOL") + 1] == "True"


def test_trackeval_command_takes_any_number_of_clips() -> None:
    command = trackeval_command(
        Path("python"), Path("script.py"), Path("gt"), Path("pred"), split="test", tracker="t", clips=("a", "b", "c")
    )

    assert command[-4:] == ["--SEQ_INFO", "a", "b", "c"]
    assert command[command.index("--SPLIT_TO_EVAL") + 1] == "test"


def test_trackeval_command_needs_a_clip() -> None:
    with pytest.raises(ValueError, match="at least one clip"):
        trackeval_command(
            Path("python"), Path("script.py"), Path("gt"), Path("pred"), split="valid", tracker="t", clips=[]
        )


# --- parse_detailed_csv --------------------------------------------------------------


def test_parse_detailed_csv_gives_percent_per_clip_and_combined(tmp_path: Path) -> None:
    scores = parse_detailed_csv(write_text(tmp_path, "person_detailed.csv", DETAILED_CSV))

    assert list(scores) == ["SNGS-021", "SNGS-022", "COMBINED"]
    assert scores["SNGS-021"] == pytest.approx({"HOTA": 50.0, "DetA": 60.0, "AssA": 40.0})
    assert scores["SNGS-022"] == pytest.approx({"HOTA": 25.0, "DetA": 50.0, "AssA": 12.5})
    assert scores["COMBINED"] == pytest.approx({"HOTA": 37.5, "DetA": 55.0, "AssA": 26.25})


def test_parse_detailed_csv_finds_the_columns_by_name(tmp_path: Path) -> None:
    text = "seq,AssA___AUC,HOTA___AUC,Dets,DetA___AUC\n\nCOMBINED,0.1,0.2,7,0.3\n"

    scores = parse_detailed_csv(write_text(tmp_path, "cls_comb_det_av_detailed.csv", text))

    assert list(scores) == ["COMBINED"]
    assert scores["COMBINED"] == pytest.approx({"HOTA": 20.0, "DetA": 30.0, "AssA": 10.0})


def test_parse_detailed_csv_lists_the_columns_found_when_a_score_column_is_missing(tmp_path: Path) -> None:
    text = "seq,HOTA,DetA___AUC,AssA___AUC\nCOMBINED,0.5,0.6,0.4\n"

    with pytest.raises(ValueError) as error:
        parse_detailed_csv(write_text(tmp_path, "person_detailed.csv", text))

    message = str(error.value)
    assert "['HOTA___AUC']" in message
    assert "['seq', 'HOTA', 'DetA___AUC', 'AssA___AUC']" in message


def test_parse_detailed_csv_refuses_values_that_are_not_shares(tmp_path: Path) -> None:
    # 63.9 looks like a percent value. Times 100 it would be a wrong score.
    text = "seq,HOTA___AUC,DetA___AUC,AssA___AUC\nCOMBINED,63.9,0.6,0.4\n"

    with pytest.raises(ValueError, match="share from 0 to 1"):
        parse_detailed_csv(write_text(tmp_path, "person_detailed.csv", text))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "no header row"),
        ("seq,HOTA___AUC,DetA___AUC,AssA___AUC\nCOMBINED,high,0.6,0.4\n", "not a number"),
        ("seq,HOTA___AUC,DetA___AUC,AssA___AUC\nCOMBINED,0.5\n", "has 2 columns"),
        ("seq,HOTA___AUC,DetA___AUC,AssA___AUC\nA,0.5,0.6,0.4\nA,0.5,0.6,0.4\n", "has two rows"),
        ("seq,HOTA___AUC,DetA___AUC,AssA___AUC\nCOMBINED,nan,0.6,0.4\n", "share from 0 to 1"),
    ],
    ids=["empty file", "text value", "short row", "name twice", "nan"],
)
def test_parse_detailed_csv_refuses_wrong_content(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_detailed_csv(write_text(tmp_path, "person_detailed.csv", text))


# --- The stand-in evaluator ----------------------------------------------------------


def test_the_stand_in_evaluator_does_not_find_the_labels_when_it_adds_its_own_split_folders(root: Path) -> None:
    # With its default settings the evaluator reads the labels from <GT_FOLDER>/<split>/<clip>, without the
    # prefix SoccerNetGS-. A command that names the data root and no --SKIP_SPLIT_FOL does not reach the
    # label files of this package. This is why trackeval_command names the split folders itself.
    script = write_text(root, "fake_evaluator.py", FAKE_EVALUATOR)
    command = [
        sys.executable,
        str(script),
        "--GT_FOLDER",
        str(root / "gt"),
        "--TRACKERS_FOLDER",
        str(root / "pred"),
        "--TRACKER_SUB_FOLDER",
        "",
        "--TRACKERS_TO_EVAL",
        "pitchvlm",
        "--SPLIT_TO_EVAL",
        "valid",
        "--SEQ_INFO",
        *CLIPS,
    ]

    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode != 0
    searched = root / "gt" / "valid" / "SNGS-021" / "Labels-GameState.json"
    assert f"GT file not found {searched}" in completed.stdout


def test_the_stand_in_evaluator_stops_when_it_gets_an_output_folder(root: Path) -> None:
    # Like the runner script, the stand-in reads --OUTPUT_FOLDER as a list and cannot build a path
    # from it. This is why run_trackeval does not pass that flag.
    script = write_text(root, "fake_evaluator.py", FAKE_EVALUATOR)
    command = trackeval_command(
        Path(sys.executable), script, root / "gt", root / "pred", split="valid", tracker="pitchvlm", clips=CLIPS
    )

    completed = subprocess.run(
        [*command, "--OUTPUT_FOLDER", str(root / "eval")], capture_output=True, text=True, check=False
    )

    assert completed.returncode != 0
    assert "TypeError" in completed.stderr
    assert not (root / "eval").exists()


def test_the_stand_in_evaluator_writes_the_columns_of_the_evaluator(root: Path) -> None:
    run_fake(root, FAKE_EVALUATOR)

    header = (root / "eval" / "person_detailed.csv").read_text(encoding="utf-8").splitlines()[0].split(",")

    # The name of the row, 3 single HOTA values, 12 HOTA fields with 19 thresholds and their mean each,
    # 6 identity values and 4 counts.
    assert len(header) == 1 + 3 + 12 * 20 + 6 + 4
    assert header[:5] == ["seq", "HOTA(0)", "LocA(0)", "HOTALocA(0)", "HOTA___5"]
    assert header[-4:] == ["Dets", "GT_Dets", "IDs", "GT_IDs"]


# --- run_trackeval -------------------------------------------------------------------


def test_run_trackeval_runs_the_script_and_reads_its_scores(root: Path) -> None:
    scores = run_fake(root, FAKE_EVALUATOR)

    assert set(scores) == {"SNGS-021", "SNGS-022", "COMBINED"}
    assert scores["SNGS-021"] == pytest.approx({"HOTA": 10.0, "DetA": 20.0, "AssA": 30.0})
    assert scores["SNGS-022"] == pytest.approx({"HOTA": 20.0, "DetA": 40.0, "AssA": 60.0})
    assert scores["COMBINED"] == pytest.approx({"HOTA": 50.0, "DetA": 60.0, "AssA": 40.0})


def test_run_trackeval_reads_the_scores_from_the_file_of_the_class_person(root: Path) -> None:
    # The evaluator writes the row COMBINED into three files, with the same scores. Here the two class
    # averages hold other scores, to show which file the wrapper reads.
    other_averages = changed(
        FAKE_EVALUATOR,
        '"cls_comb_cls_av": [combined], "cls_comb_det_av": [combined]',
        '"cls_comb_cls_av": [row("COMBINED", 0.9, 0.9, 0.9)], "cls_comb_det_av": [row("COMBINED", 0.8, 0.8, 0.8)]',
    )

    scores = run_fake(root, other_averages)

    assert set(scores) == {"SNGS-021", "SNGS-022", "COMBINED"}
    assert scores["COMBINED"] == pytest.approx({"HOTA": 50.0, "DetA": 60.0, "AssA": 40.0})
    assert parse_detailed_csv(root / "eval" / "person_detailed.csv") == scores
    averages = parse_detailed_csv(root / "eval" / "cls_comb_det_av_detailed.csv")
    assert averages == {"COMBINED": pytest.approx({"HOTA": 80.0, "DetA": 80.0, "AssA": 80.0})}


def test_run_trackeval_moves_the_detailed_csv_files_into_the_output_folder(root: Path) -> None:
    output_dir = root / "eval" / "pass-1"

    scores = run_fake(root, FAKE_EVALUATOR, output_dir)

    assert sorted(path.name for path in output_dir.iterdir()) == DETAILED_FILES
    assert parse_detailed_csv(output_dir / "person_detailed.csv") == scores
    # No detailed file stays in the prediction folder, where a later run could take it for its own.
    # The prediction files and the summary files of the evaluator stay.
    assert sorted(path.name for path in (root / TRACKER_FOLDER).iterdir()) == [
        "SNGS-021.json",
        "SNGS-022.json",
        "cls_comb_cls_av_summary.txt",
        "cls_comb_det_av_summary.txt",
        "person_summary.txt",
    ]


def test_run_trackeval_can_run_again_and_replaces_the_files_of_the_first_run(root: Path) -> None:
    other_scores = changed(FAKE_EVALUATOR, 'row("COMBINED", 0.5, 0.6, 0.4)', 'row("COMBINED", 0.25, 0.6, 0.4)')

    first = run_fake(root, FAKE_EVALUATOR)
    second = run_fake(root, other_scores)

    assert first["COMBINED"]["HOTA"] == pytest.approx(50.0)
    assert second["COMBINED"]["HOTA"] == pytest.approx(25.0)
    assert sorted(path.name for path in (root / "eval").iterdir()) == DETAILED_FILES
    assert parse_detailed_csv(root / "eval" / "person_detailed.csv") == second


@pytest.mark.parametrize("file_name", ["person_detailed.csv", "cls_comb_det_av_detailed.csv"])
def test_run_trackeval_refuses_a_result_file_of_another_run(root: Path, file_name: str) -> None:
    old_result = write_text(root / TRACKER_FOLDER, file_name, DETAILED_CSV)

    # The old file has a row for every clip. If the evaluator then wrote no file, the old scores
    # would be read as the scores of this run.
    with pytest.raises(FileExistsError, match="another run") as error:
        run_fake(root, FAKE_EVALUATOR)

    assert str(old_result) in str(error.value)
    # The evaluator did not run, and the old file is still at its place.
    assert not (root / TRACKER_FOLDER / "person_summary.txt").exists()
    assert old_result.read_text(encoding="utf-8") == DETAILED_CSV
    assert not (root / "eval").exists()


def test_run_trackeval_reports_a_failed_run_with_the_output_of_the_evaluator(root: Path) -> None:
    with pytest.raises(RuntimeError, match="exit code 3") as error:
        run_fake(root, FAILING_EVALUATOR)

    assert "reading the labels" in str(error.value)
    assert "the labels are broken" in str(error.value)


def test_run_trackeval_shows_the_last_lines_that_the_evaluator_printed(root: Path) -> None:
    many_lines = "import sys\nfor number in range(100):\n    print('line', number)\nsys.exit(3)\n"

    with pytest.raises(RuntimeError, match="exit code 3") as error:
        run_fake(root, many_lines)

    # The message shows the last 30 lines of the output: the lines 71 to 99 and, at its end, the empty
    # line between the two output streams.
    message = str(error.value)
    assert message.splitlines() == ["the evaluator stopped with exit code 3:", *[f"line {n}" for n in range(71, 100)]]
    assert message.endswith("line 99\n")


def test_run_trackeval_gives_the_evaluator_no_variable_that_must_be_dropped(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The evaluator is the code of another project. It must not get the key of the model in its environment."""
    monkeypatch.setenv("PITCHVLM_TEST_KEY", "a-made-up-value")
    monkeypatch.setenv("PITCHVLM_TEST_OTHER", "kept")
    seen_file = root / "seen.txt"
    # The stand-in writes which of the two variables it sees, and then the result files of the evaluator.
    reports_its_environment = (
        "import os\n"
        f"with open({str(seen_file)!r}, 'w', encoding='utf-8') as out:\n"
        "    out.write(','.join(name for name in ('PITCHVLM_TEST_KEY', 'PITCHVLM_TEST_OTHER') if name in os.environ))\n"
        + FAKE_EVALUATOR
    )
    script = write_text(root, "fake_evaluator.py", reports_its_environment)
    arguments = (Path(sys.executable), script, root / "gt", root / "pred")
    run = {"split": "valid", "tracker": "pitchvlm", "clips": CLIPS}

    scores = run_trackeval(*arguments, **run, output_dir=root / "eval", drop_env=["PITCHVLM_TEST_KEY"])

    assert seen_file.read_text(encoding="utf-8") == "PITCHVLM_TEST_OTHER"
    assert scores["COMBINED"]["HOTA"] == pytest.approx(50.0)
    # The environment of this process is not changed, and without a name the evaluator gets all of it.
    assert os.environ["PITCHVLM_TEST_KEY"] == "a-made-up-value"
    run_trackeval(*arguments, **run, output_dir=root / "eval")
    assert seen_file.read_text(encoding="utf-8") == "PITCHVLM_TEST_KEY,PITCHVLM_TEST_OTHER"


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="this system has no kill signal")
def test_run_trackeval_reports_an_evaluator_that_was_killed(root: Path) -> None:
    """A killed process has a negative exit code. The files that it wrote are not read as the scores of a run."""
    killed_after_writing = FAKE_EVALUATOR + "import signal\nos.kill(os.getpid(), signal.SIGKILL)\n"

    with pytest.raises(RuntimeError, match="exit code -9"):
        run_fake(root, killed_after_writing)


def test_run_trackeval_reports_a_label_file_that_the_evaluator_does_not_find(root: Path) -> None:
    (root / SPLIT_FOLDER / "SNGS-022" / "Labels-GameState.json").unlink()

    with pytest.raises(RuntimeError, match="exit code 1") as error:
        run_fake(root, FAKE_EVALUATOR)

    # The message shows the line of the evaluator that names the clip. The first clip was found.
    assert "GT file not found for sequence: SNGS-022" in str(error.value)
    assert "SNGS-021" not in str(error.value)


def test_run_trackeval_reports_a_prediction_file_that_the_evaluator_does_not_find(root: Path) -> None:
    (root / TRACKER_FOLDER / "SNGS-021.json").unlink()

    with pytest.raises(RuntimeError, match="exit code 1") as error:
        run_fake(root, FAKE_EVALUATOR)

    assert f"Tracker file not found: {root / TRACKER_FOLDER / 'SNGS-021.json'}" in str(error.value)


def test_the_files_of_a_failed_run_do_not_stop_the_next_run(root: Path) -> None:
    fails_after_writing = FAKE_EVALUATOR + "raise SystemExit(3)\n"

    with pytest.raises(RuntimeError, match="exit code 3"):
        run_fake(root, fails_after_writing)

    # The files of the failed run were moved out of the prediction folder too.
    assert sorted(path.name for path in (root / "eval").iterdir()) == DETAILED_FILES
    assert run_fake(root, FAKE_EVALUATOR)["COMBINED"]["HOTA"] == pytest.approx(50.0)


def test_run_trackeval_needs_a_detailed_csv_file(root: Path) -> None:
    with pytest.raises(FileNotFoundError, match="person_detailed.csv") as error:
        run_fake(root, "print('nothing was scored')\n")

    # The message names the folder that was searched, and it shows what the evaluator printed.
    assert str(root / TRACKER_FOLDER) in str(error.value)
    assert "nothing was scored" in str(error.value)


def test_run_trackeval_needs_the_file_of_the_class_person(root: Path) -> None:
    averages_only = changed(FAKE_EVALUATOR, '"person": person, ', "")

    with pytest.raises(FileNotFoundError, match="person_detailed.csv"):
        run_fake(root, averages_only)

    # The two files of the class averages were moved, so they do not stop the next run.
    assert sorted(path.name for path in (root / "eval").iterdir()) == DETAILED_FILES[:2]
    assert run_fake(root, FAKE_EVALUATOR)["COMBINED"]["HOTA"] == pytest.approx(50.0)


def test_run_trackeval_does_not_read_a_file_that_the_output_folder_held_before_the_run(root: Path) -> None:
    (root / "eval").mkdir()
    old_scores = write_text(root / "eval", "person_detailed.csv", DETAILED_CSV)

    # The old file has a row for every clip, but it is not a file of this run.
    with pytest.raises(FileNotFoundError, match="person_detailed.csv"):
        run_fake(root, "print('nothing was scored')\n")

    assert old_scores.read_text(encoding="utf-8") == DETAILED_CSV


def test_run_trackeval_needs_a_row_for_every_clip(root: Path) -> None:
    one_clip_only = changed(FAKE_EVALUATOR, "enumerate(sorted(clips), start=1)", "enumerate(sorted(clips)[:1])")

    with pytest.raises(ValueError, match="no row for .'SNGS-022'."):
        run_fake(root, one_clip_only)


def test_run_trackeval_needs_the_combined_row(root: Path) -> None:
    no_combined = changed(FAKE_EVALUATOR, "person.append(combined)\n", "")

    with pytest.raises(ValueError, match="no row for .'COMBINED'."):
        run_fake(root, no_combined)
