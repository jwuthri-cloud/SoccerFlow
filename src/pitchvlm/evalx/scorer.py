"""Wrapper for the stand-alone GS-HOTA evaluator (``sn-trackeval``).

The evaluator runs in its own interpreter, because it needs another
environment (Python 3.12, numpy 1.26) than this package. This module builds
the command, runs it, and reads the scores from a detailed CSV file that the
evaluator writes.

NOT RUN YET. The evaluator is not installed where this module was written.
The points below were read in the source of ``sn-trackeval`` 0.4.0 (the
runner script ``scripts/run_soccernet_gs.py``, the data class
``trackeval/datasets/soccernet_gs.py``, ``trackeval/eval.py`` and
``trackeval/utils.py``). The first real run must confirm them:

- Folders. With its default settings the evaluator reads the labels from
  ``<GT_FOLDER>/<split>/<clip>/Labels-GameState.json`` and the predictions
  from ``<TRACKERS_FOLDER>/SoccerNetGS-<split>/<tracker>/<clip>.json``: only
  the prediction side gets the prefix ``SoccerNetGS-``. This package keeps
  the clips of a split in ``<root>/SoccerNetGS-<split>``. So the command
  names both split folders itself and passes ``--SKIP_SPLIT_FOL True``: the
  evaluator then adds no split folder on either side.
- Output folder. The runner script reads every setting whose default value
  is None as a list of values. So ``--OUTPUT_FOLDER <folder>`` arrives as a
  list, and the evaluator stops with a ``TypeError`` when it builds its
  output path. This is why the command names no output folder. The
  evaluator then writes its result files into the folder of the prediction
  files of the tracker.
- Result files. The game-state task has one class, ``person``, and the
  evaluator also writes the two averages over the classes. So every run
  writes three detailed CSV files: ``person_detailed.csv`` with one row per
  clip and a last row ``COMBINED``, and ``cls_comb_cls_av_detailed.csv`` and
  ``cls_comb_det_av_detailed.csv`` with the row ``COMBINED`` only. With one
  class the three ``COMBINED`` rows hold the same scores, so the scores are
  read from ``person_detailed.csv`` alone. The columns ``HOTA___AUC``,
  ``DetA___AUC`` and ``AssA___AUC`` hold shares from 0 to 1. Next to these
  files the evaluator writes a summary text file and two plot files per
  class.
- Clips. ``--SEQ_INFO`` with clip names only is enough: the data class takes
  the clip names from it and reads no file ``seqinfo.ini``.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from collections.abc import Collection, Sequence
from pathlib import Path

#: Name of each score -> its column in the detailed CSV: the mean over all thresholds, a share from 0 to 1.
SCORE_COLUMNS = {"HOTA": "HOTA___AUC", "DetA": "DetA___AUC", "AssA": "AssA___AUC"}
#: End of the name of the detailed CSV files. The evaluator writes one such file per class and class average.
DETAILED_SUFFIX = "_detailed.csv"
#: Detailed CSV file of the class ``person``, the one class of the game-state task. The scores are read from it.
SCORES_FILE = "person" + DETAILED_SUFFIX
#: Name of the CSV row that holds the scores of all clips together.
COMBINED = "COMBINED"
#: Number of lines of evaluator output that an error message shows.
OUTPUT_TAIL_LINES = 30


def trackeval_command(
    python_exe: Path,
    script: Path,
    gt_root: Path,
    pred_root: Path,
    *,
    split: str,
    tracker: str,
    clips: Sequence[str],
) -> list[str]:
    """Build the command that runs the evaluator script ``scripts/run_soccernet_gs.py``.

    ``gt_root`` is the data root: the label file of a clip is
    ``gt_root/SoccerNetGS-<split>/<clip>/Labels-GameState.json``, where the
    data command puts it. ``pred_root`` holds the prediction files in the
    layout of ``export_gsr.predictions_path``:
    ``pred_root/SoccerNetGS-<split>/<tracker>/<clip>.json``. The command gives
    the evaluator the two split folders and tells it to add no split folder
    of its own (see the module text). It names no output folder, so the
    evaluator writes its result files into the folder of these prediction
    files. ``clips`` must name at least one clip.
    """
    if len(clips) == 0:
        raise ValueError("clips must name at least one clip")
    return [
        str(python_exe),
        str(script),
        "--GT_FOLDER",
        str(_split_folder(gt_root, split)),
        "--TRACKERS_FOLDER",
        str(_split_folder(pred_root, split)),
        "--SKIP_SPLIT_FOL",
        "True",
        "--TRACKER_SUB_FOLDER",
        "",
        "--TRACKERS_TO_EVAL",
        tracker,
        "--SPLIT_TO_EVAL",
        split,
        "--SEQ_INFO",
        *clips,
    ]


def parse_detailed_csv(path: Path) -> dict[str, dict[str, float]]:
    """Read the scores of one detailed CSV file of the evaluator, in percent.

    The first column is the clip name; the last row is ``COMBINED``. The
    result maps each name to ``{"HOTA": ..., "DetA": ..., "AssA": ...}``. The
    file holds shares from 0 to 1, and the result holds them times 100.
    ``ValueError`` is raised when a score column is missing (the message lists
    the columns found), when a name has two rows, and when a value is not a
    share from 0 to 1.
    """
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if not rows:
        raise ValueError(f"{path}: the file has no header row")
    header = rows[0]
    missing = [column for column in SCORE_COLUMNS.values() if column not in header]
    if missing:
        raise ValueError(f"{path}: the columns {missing} are missing; columns found: {header}")
    column_index = {name: header.index(column) for name, column in SCORE_COLUMNS.items()}
    scores: dict[str, dict[str, float]] = {}
    for row in rows[1:]:
        clip = row[0]
        if clip in scores:
            raise ValueError(f"{path}: the name {clip!r} has two rows")
        scores[clip] = {name: _percent(row, index, path) for name, index in column_index.items()}
    return scores


def run_trackeval(
    python_exe: Path,
    script: Path,
    gt_root: Path,
    pred_root: Path,
    *,
    split: str,
    tracker: str,
    clips: Sequence[str],
    output_dir: Path,
    drop_env: Collection[str] = (),
) -> dict[str, dict[str, float]]:
    """Run the evaluator and return its scores in percent: clip or ``COMBINED`` -> HOTA, DetA, AssA.

    The evaluator process gets the environment of this process without the
    variables that ``drop_env`` names. The evaluator is the code of another
    project, with packages that are not pinned, and it needs no key. So the
    command line names the variable of the model key here.

    The command of ``trackeval_command`` runs in the interpreter
    ``python_exe``. The evaluator writes its result files into the folder of
    the prediction files of the tracker,
    ``pred_root/SoccerNetGS-<split>/<tracker>``. After the run, every file in
    that folder whose name ends with ``_detailed.csv`` is moved into
    ``output_dir``, where it replaces a file of the same name; the other
    files of the evaluator stay where they are. The files of a run that
    stopped with an error are moved too, so they do not stop the next run.
    The scores are read from the moved file ``person_detailed.csv``: it holds
    one row per clip and the row ``COMBINED`` (see the module text).

    ``FileExistsError`` is raised, before the evaluator runs, when the
    prediction folder already holds a detailed CSV file. That file comes from
    another run, and its scores could not be told from the scores of this
    run. ``RuntimeError`` is raised when the evaluator stops with an error,
    ``FileNotFoundError`` when it wrote no ``person_detailed.csv``, and
    ``ValueError`` when that file has no row for a clip or no row
    ``COMBINED``.
    """
    command = trackeval_command(python_exe, script, gt_root, pred_root, split=split, tracker=tracker, clips=clips)
    results_dir = _split_folder(pred_root, split) / tracker
    old_files = _detailed_files(results_dir)
    if old_files:
        raise FileExistsError(
            f"{results_dir} already holds evaluator result files of another run: "
            f"{', '.join(str(path) for path in old_files)}. Their scores could not be told from the scores of "
            "this run. Move or remove them, then run again."
        )
    environment = {name: value for name, value in os.environ.items() if name not in drop_env}
    completed = subprocess.run(
        command, capture_output=True, text=True, errors="replace", check=False, env=environment
    )
    moved = _move_files(_detailed_files(results_dir), Path(output_dir))
    if completed.returncode != 0:
        raise RuntimeError(
            f"the evaluator stopped with exit code {completed.returncode}:\n{_output_tail(completed)}"
        )
    scores_file = Path(output_dir) / SCORES_FILE
    if scores_file not in moved:
        raise FileNotFoundError(
            f"the evaluator wrote no file {SCORES_FILE} into {results_dir}:\n{_output_tail(completed)}"
        )
    scores = parse_detailed_csv(scores_file)
    missing = [name for name in [*clips, COMBINED] if name not in scores]
    if missing:
        raise ValueError(f"{scores_file} has no row for {missing}; rows found: {sorted(scores)}")
    return scores


# --- Helpers -----------------------------------------------------------------------


def _percent(row: Sequence[str], index: int, path: Path) -> float:
    """Read one score of a CSV row, a share from 0 to 1, and return it in percent."""
    if index >= len(row):
        raise ValueError(f"{path}: the row of {row[0]!r} has {len(row)} columns, but column {index + 1} is needed")
    try:
        share = float(row[index])
    except ValueError as error:
        raise ValueError(f"{path}: the row of {row[0]!r} holds {row[index]!r}, which is not a number") from error
    if not -1e-9 <= share <= 1.0 + 1e-9:
        raise ValueError(f"{path}: the row of {row[0]!r} holds {share}, but a share from 0 to 1 is expected")
    return 100.0 * share


def _split_folder(root: Path, split: str) -> Path:
    """Return the folder of one split below a root folder: ``root/SoccerNetGS-<split>``.

    Below the data root this folder holds the clips of the split. Below a prediction root it holds one
    folder per tracker (``export_gsr.predictions_path``).
    """
    return Path(root) / f"SoccerNetGS-{split}"


def _detailed_files(folder: Path) -> list[Path]:
    """Return every detailed CSV file of a folder, sorted. A missing folder has none."""
    return sorted(path for path in folder.glob(f"*{DETAILED_SUFFIX}") if path.is_file())


def _move_files(files: Sequence[Path], target_dir: Path) -> list[Path]:
    """Move files into ``target_dir`` and return their new paths, in the same order.

    Every file keeps its name. A file of the same name in ``target_dir`` is replaced.
    """
    moved: list[Path] = []
    for path in files:
        target = target_dir / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(path, target)
        moved.append(target)
    return moved


def _output_tail(completed: subprocess.CompletedProcess[str]) -> str:
    """Return the last lines that the evaluator printed, for error messages."""
    lines = (completed.stdout + "\n" + completed.stderr).splitlines()
    return "\n".join(lines[-OUTPUT_TAIL_LINES:])
