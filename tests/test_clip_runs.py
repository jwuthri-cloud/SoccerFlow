"""Tests of the clip job, the clip run and the conversion of its answers (``fallbacks.runs``, ``fallbacks.answers``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pitchvlm.fallbacks.answers import AnswerError, calibration_answers, detector_answers
from pitchvlm.fallbacks.rfdetr import BALL_LABEL, ball_candidates, expand_rows
from pitchvlm.fallbacks.runs import (
    CLIP_JOB_FORMAT,
    CLIP_RUN_FORMAT,
    ClipJob,
    clip_run_part_path,
    load_clip_run,
)
from pitchvlm.geometry import matrix_from_record
from pitchvlm.schemas import Box

DETECTOR = {"name": "rfdetr", "model": "RFDETRMedium", "min_score": 0.5, "antialias": False}
CALIBRATION = {"name": "pnlcalib", "kp_threshold": 0.3434, "line_threshold": 0.7867, "pnl_refine": True}
PERSON_ROW = [10.0, 20.0, 30.0, 80.0, 0.9, 0]
BALL_ROW = [50.0, 60.0, 56.0, 66.0, 0.7, 1]


def compact(*rows: list[float | int]) -> dict[str, Any]:
    return {"labels": ["person", BALL_LABEL], "rows": [list(row) for row in rows]}


def part(
    number: int, parts: int, frames: list[list[Any]], *, tool: dict[str, Any] = DETECTOR, n_frames: int = 4
) -> dict[str, Any]:
    return {
        "format": CLIP_RUN_FORMAT,
        "clip": "SNGS-021",
        "tool": tool,
        "n_frames": n_frames,
        "job_sha256": "a" * 64,
        "runner": {"device": "cpu"},
        "part": number,
        "parts": parts,
        "frames": frames,
    }


def write(path: Path, document: Any) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def frames_of(numbers: range, payload: dict[str, Any] | None = None) -> list[list[Any]]:
    return [[frame, 0.01 * (frame + 1), payload or compact(PERSON_ROW)] for frame in numbers]


def two_parts(folder: Path) -> Path:
    first = write(folder / "detections.json", part(1, 2, frames_of(range(0, 2))))
    write(folder / "detections-0002.json", part(2, 2, frames_of(range(2, 4))))
    return first


def test_part_files_have_the_names_of_the_ledger_series(tmp_path: Path) -> None:
    first = tmp_path / "detections.json"
    assert clip_run_part_path(first, 1) == first
    assert clip_run_part_path(first, 2) == tmp_path / "detections-0002.json"
    assert clip_run_part_path(first, 12).name == "detections-0012.json"
    with pytest.raises(ValueError):
        clip_run_part_path(first, 0)


def test_run_of_two_parts_is_read_in_frame_order(tmp_path: Path) -> None:
    run = load_clip_run(two_parts(tmp_path))
    assert [answer.frame for answer in run.frames] == [0, 1, 2, 3]
    assert run.seconds() == pytest.approx([0.01, 0.02, 0.03, 0.04])
    assert (run.clip, run.n_frames, run.tool.name, run.runner) == ("SNGS-021", 4, "rfdetr", {"device": "cpu"})


@pytest.mark.parametrize(
    ("change", "words"),
    [
        (lambda folder: (folder / "detections-0002.json").unlink(), "part 2 of 2 of the clip run is missing"),
        (
            lambda folder: write(
                folder / "detections-0002.json", {**part(2, 2, frames_of(range(2, 4))), "job_sha256": "b" * 64}
            ),
            "is not part 2 of the run",
        ),
        (lambda folder: write(folder / "detections-0002.json", part(3, 3, frames_of(range(2, 4)))), "is not part 2"),
        (
            lambda folder: write(folder / "detections-0003.json", part(3, 3, frames_of(range(2, 4)))),
            "part of another run",
        ),
        (
            lambda folder: write(folder / "detections-0002.json", part(2, 2, frames_of(range(2, 3)))),
            "1 frames are missing",
        ),
        (
            lambda folder: write(folder / "detections-0002.json", part(2, 2, frames_of(range(1, 4)))),
            "1 come more than once",
        ),
        (lambda folder: write(folder / "detections.json", part(2, 2, frames_of(range(0, 2)))), "must be part 1"),
        (
            lambda folder: write(
                folder / "detections-0002.json", part(2, 2, [[2, "0.1", compact()], [3, 0.1, compact()]])
            ),
            "seconds",
        ),
        (
            lambda folder: write(
                folder / "detections-0002.json", part(2, 2, [[True, 0.1, compact()], [3, 0.1, compact()]])
            ),
            "frame",
        ),
        (lambda folder: write(folder / "detections-0002.json", part(2, 2, [[2, 0.1], [3, 0.1, compact()]])), "frames"),
        (
            lambda folder: (folder / "detections-0002.json").write_text('{"format": 1, "format": 2}', encoding="utf-8"),
            "come more than once",
        ),
    ],
)
def test_wrong_series_of_parts_is_refused(tmp_path: Path, change: Any, words: str) -> None:
    first = two_parts(tmp_path)
    change(tmp_path)
    with pytest.raises(ValueError, match=words):
        load_clip_run(first)


def test_clip_job_names_every_frame_and_the_tool() -> None:
    job = ClipJob(format=CLIP_JOB_FORMAT, clip="SNGS-021", tool=DETECTOR, images=("a/000001.jpg", "a/000002.jpg"))
    assert job.model_dump(mode="json")["images"] == ["a/000001.jpg", "a/000002.jpg"]
    with pytest.raises(ValueError):
        ClipJob(format=CLIP_JOB_FORMAT, clip="SNGS-021", tool=DETECTOR, images=())


# --- Compact rows and the ball class ---------------------------------------------------


def test_compact_rows_become_the_payload_of_the_adapter() -> None:
    payload = expand_rows(compact(PERSON_ROW, BALL_ROW))
    assert payload == {
        "detections": [
            {"box_xyxy": [10.0, 20.0, 30.0, 80.0], "score": 0.9, "label": "person"},
            {"box_xyxy": [50.0, 60.0, 56.0, 66.0], "score": 0.7, "label": BALL_LABEL},
        ]
    }


@pytest.mark.parametrize(
    ("payload", "words"),
    [
        ({"rows": []}, "'labels' is missing"),
        ({"labels": "person", "rows": []}, "must be a list of class names"),
        ({"labels": ["person", 3], "rows": []}, "must be a list of class names"),
        ({"labels": ["person"], "rows": {}}, "'rows' must be a list"),
        ({"labels": ["person"], "rows": [[1.0, 2.0, 3.0, 4.0, 0.9]]}, "a row must be"),
        ({"labels": ["person"], "rows": [[1.0, 2.0, 3.0, 4.0, 0.9, 1]]}, "label index must be a place"),
        ({"labels": ["person"], "rows": [[1.0, 2.0, 3.0, 4.0, 0.9, True]]}, "label index must be a place"),
        ({"labels": ["person"], "rows": [[1.0, 2.0, 3.0, 4.0, 0.9, 0.0]]}, "label index must be a place"),
    ],
)
def test_wrong_compact_payload_is_refused(payload: dict[str, Any], words: str) -> None:
    with pytest.raises(ValueError, match=words):
        expand_rows(payload)


def test_ball_candidates_are_the_kept_detections_of_the_ball_class() -> None:
    payload = expand_rows(
        compact(PERSON_ROW, BALL_ROW, [70.0, 60.0, 76.0, 66.0, 0.3, 1], [80.0, 60.0, 80.0, 66.0, 0.9, 1])
    )
    (ball,) = ball_candidates(payload, frame=7, min_score=0.5)
    assert (ball.frame, ball.box, ball.score) == (7, Box(x1=50.0, y1=60.0, x2=56.0, y2=66.0), 0.7)
    with pytest.raises(ValueError):
        ball_candidates(payload, frame=7, min_score=1.5)


# --- Answers of a clip run --------------------------------------------------------------


def test_detector_run_gives_people_of_every_frame_and_balls_where_there_are(tmp_path: Path) -> None:
    frames = frames_of(range(4))
    frames[2][2] = compact(PERSON_ROW, BALL_ROW)
    run = load_clip_run(write(tmp_path / "d.json", part(1, 1, frames)))
    people, balls = detector_answers(run)
    assert [frame_people.frame for frame_people in people] == [0, 1, 2, 3]
    assert all(len(frame_people.people) == 1 for frame_people in people)
    assert {
        (person.role, person.team_key, person.number) for frame_people in people for person in frame_people.people
    } == {("player", "none", None)}
    assert list(balls) == [2]


def test_payload_fault_names_the_frame(tmp_path: Path) -> None:
    frames = frames_of(range(4))
    frames[3][2] = {"labels": ["person"], "rows": [[1.0, 2.0, 3.0, 4.0, 1.5, 0]]}
    run = load_clip_run(write(tmp_path / "d.json", part(1, 1, frames)))
    with pytest.raises(AnswerError, match="frame 3: .*'score' must be from 0 to 1"):
        detector_answers(run)


def test_calibration_run_gives_the_mapping_of_every_frame(tmp_path: Path) -> None:
    pitch_to_image = np.array([[10.0, 0.0, 640.0], [0.0, 5.0, 360.0], [0.0, 0.0, 1.0]])
    camera = {"homography_pitch_to_image": pitch_to_image.tolist(), "pitch_origin": "centre"}
    frames = [[frame, 0.4, camera if frame != 1 else {"homography_pitch_to_image": None}] for frame in range(3)]
    run = load_clip_run(write(tmp_path / "c.json", part(1, 1, frames, tool=CALIBRATION, n_frames=3)))
    cameras = calibration_answers(run)
    assert [record.valid for record in cameras] == [True, False, True]
    image_to_pitch = matrix_from_record(cameras[0])
    point = image_to_pitch @ np.array([740.0, 410.0, 1.0])
    np.testing.assert_allclose(point[:2] / point[2], [10.0, 10.0])
    frames[2][2] = {"homography_pitch_to_image": pitch_to_image.tolist(), "pitch_origin": "corner"}
    run = load_clip_run(write(tmp_path / "c.json", part(1, 1, frames, tool=CALIBRATION, n_frames=3)))
    with pytest.raises(AnswerError, match="frame 2"):
        calibration_answers(run)


def test_run_of_one_tool_is_not_read_as_the_other(tmp_path: Path) -> None:
    detector_run = load_clip_run(write(tmp_path / "d.json", part(1, 1, frames_of(range(4)))))
    calibration_run = load_clip_run(
        write(
            tmp_path / "c.json",
            part(1, 1, [[0, 0.1, {"homography_pitch_to_image": None}]], tool=CALIBRATION, n_frames=1),
        )
    )
    with pytest.raises(ValueError, match="not of a calibration"):
        calibration_answers(detector_run)
    with pytest.raises(ValueError, match="not of a detector"):
        detector_answers(calibration_run)
