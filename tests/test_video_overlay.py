"""Tests of the two tools of the first real clip runs: a video file into the detector engine, and the
check images of the pitch mappings for gate G3.

``video-frames`` writes the frames of a video as images, ``clip-job`` names the video by its SHA-256,
and ``run`` reads a clip run only with that video. ``overlay`` draws the pitch lines of chosen
mappings of three sources (the stored answers of the pitch probe, the run file of the calibration
tool, the result of a run) and writes the list of the chosen frames.

The clip is a synthetic scene. The tools are stand-ins: the clip runs and the run file hold the true
people and the true camera of the scene (``fallbacks.fake``), written with the part writer of
``tools/run_tool.py``. The model is the fake model. Every image goes to a temporary folder (rule 2).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
import pytest

from pitchvlm import cli
from pitchvlm.fallbacks.fake import fake_detector_answers
from pitchvlm.frames import ImageFolderSource, VideoFileSource
from pitchvlm.geometry import projected_line_pixels
from pitchvlm.overlays import CHOICE_RULE, LIST_FORMAT, MappedFrame, caption, choose_frames, image_name
from pitchvlm.render import draw_mapping_check
from pitchvlm.synth import RetimedScene, SyntheticScene
from pitchvlm.vlm.fake import FakeBackend

REPO = Path(__file__).resolve().parents[1]
DETECTOR_SETTINGS = REPO / "configs" / "engines" / "detector_calibration.yaml"
EXPERIMENTS = REPO / "configs" / "experiments"
N_FRAMES = 50
#: The name of the video file, so the clip id of the video.
VIDEO_CLIP = "demo-clip"
#: The clip id of the clip of the dataset layout (the scene and its labels).
DATASET_CLIP = "SNGS-999"
LABEL_NAME = "Labels-GameState.json"


def load_script() -> ModuleType:
    """Load ``tools/run_tool.py`` as a module (it is not part of a package)."""
    spec = importlib.util.spec_from_file_location("run_tool_for_video", REPO / "tools" / "run_tool.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_tool_for_video"] = module
    spec.loader.exec_module(module)
    return module


run_tool = load_script()


def error_lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    """Return the lines of the error stream of a command that could not run. A traceback fails the test."""
    stderr = capsys.readouterr().err
    assert "Traceback" not in stderr
    return stderr.strip().splitlines()


def write_video(path: Path, images: Sequence[np.ndarray], fps: float) -> Path:
    """Write an mp4v video. The test is skipped when this OpenCV build cannot write one."""
    height, width = images[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write an mp4v video")
    for image in images:
        writer.write(image)
    writer.release()
    return path


def detector_payloads(scene: SyntheticScene) -> list[dict[str, Any]]:
    """The payloads of a detector that finds the true people and the stand-in ball of every frame."""
    people, balls = fake_detector_answers(scene)
    payloads = []
    for frame_people in people:
        detections = [
            {"box_xyxy": [p.box.x1, p.box.y1, p.box.x2, p.box.y2], "score": 0.9, "label": "person"}
            for p in frame_people.people
        ]
        detections += [
            {"box_xyxy": [b.box.x1, b.box.y1, b.box.x2, b.box.y2], "score": b.score, "label": "sports ball"}
            for b in balls.get(frame_people.frame, ())
        ]
        payloads.append({"detections": detections})
    return payloads


def camera_payload(scene: SyntheticScene, frame: int) -> dict[str, Any]:
    """The payload of a calibration that finds the true camera of a frame."""
    return {"homography_pitch_to_image": np.linalg.inv(scene.homography(frame)).tolist(), "pitch_origin": "centre"}


def write_clip_run(job_file: Path, out: Path, payloads: Sequence[dict[str, Any]]) -> Path:
    """Write the clip run of a clip job with the part writer of the script: one payload per frame."""
    job = run_tool.read_job(job_file, json.loads(job_file.read_text(encoding="utf-8"))["tool"]["name"])
    frames = [
        {"clip": job["clip"], "frame": frame, "seconds": 0.02, "payload": payload}
        for frame, payload in enumerate(payloads)
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    run_tool.write_clip_run(out, job, run_tool.file_sha256(job_file), {"device": "stand-in"}, frames)
    return out


# --- The video path of the detector engine ------------------------------------------------


@pytest.fixture(scope="module")
def scene() -> SyntheticScene:
    return SyntheticScene(n_frames=N_FRAMES, seed=4)


@pytest.fixture(scope="module")
def video_run(tmp_path_factory: pytest.TempPathFactory, scene: SyntheticScene) -> dict[str, Path]:
    """The whole video path: video, frame images, both clip jobs, stand-in clip runs and the run of the video.

    The result maps names to paths: ``video``, ``root`` (the data folder), ``clip_folder``,
    ``detections``, ``calibration``, ``result`` and ``work`` (a folder for more files).
    """
    work = tmp_path_factory.mktemp("video")
    root = work / "data"
    video = write_video(root / "footage" / f"{VIDEO_CLIP}.mp4", [scene.image(f) for f in range(N_FRAMES)], 25.0)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cli, "render_clip", lambda source, result, out_path: out_path)
        assert cli.main(["video-frames", "--video", str(video), "--out", str(root / "videos")]) == 0
        clip_folder = root / "videos" / VIDEO_CLIP
        runs: dict[str, Path] = {}
        for tool, name, payloads in (
            ("rfdetr", "detections.json", detector_payloads(scene)),
            ("pnlcalib", "calibration.json", [camera_payload(scene, frame) for frame in range(N_FRAMES)]),
        ):
            job = work / "jobs" / f"{tool}.json"
            arguments = ["clip-job", "--tool", tool, "--config", str(DETECTOR_SETTINGS), "--clip", str(clip_folder)]
            assert cli.main([*arguments, "--root", str(root), "--out", str(job)]) == 0
            runs[name] = write_clip_run(job, work / "tools" / name, payloads)
        code = cli.main(
            [
                "run",
                "--clip",
                str(video),
                "--config",
                str(DETECTOR_SETTINGS),
                "--detections",
                str(runs["detections.json"]),
                "--calibration",
                str(runs["calibration.json"]),
                "--out",
                str(work / "runs"),
                "--allow-dirty",
            ]
        )
    assert code == 0
    return {
        "video": video,
        "root": root,
        "clip_folder": clip_folder,
        "detections": runs["detections.json"],
        "calibration": runs["calibration.json"],
        "result": work / "runs" / VIDEO_CLIP / "result.json",
        "work": work,
    }


def video_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_video_frames_are_the_frames_that_run_reads(video_run: dict[str, Path]) -> None:
    """The images that a tool reads are the frames that ``run`` decodes: same count, order and pixels."""
    images = ImageFolderSource(video_run["clip_folder"] / "img1", VIDEO_CLIP, fps=25.0)
    video = VideoFileSource(video_run["video"])
    assert (images.info.n_frames, images.info.width, images.info.height) == (
        video.info.n_frames,
        video.info.width,
        video.info.height,
    )
    for frame in range(N_FRAMES):
        assert np.array_equal(images.read(frame), video.read(frame)), f"frame {frame}"
    record = json.loads((video_run["clip_folder"] / "video.json").read_text(encoding="utf-8"))
    assert record == {
        "format": "pitchvlm-video-frames-1",
        "video": f"{VIDEO_CLIP}.mp4",
        "video_sha256": video_sha256(video_run["video"]),
        "n_frames": N_FRAMES,
        "fps": 25.0,
        "width": video.info.width,
        "height": video.info.height,
    }


def test_clip_job_and_clip_run_name_the_video_and_run_records_it(video_run: dict[str, Path]) -> None:
    sha = video_sha256(video_run["video"])
    job = json.loads((video_run["work"] / "jobs" / "rfdetr.json").read_text(encoding="utf-8"))
    assert job["video_sha256"] == sha
    assert (job["clip"], len(job["images"])) == (VIDEO_CLIP, N_FRAMES)
    assert job["images"][0] == f"videos/{VIDEO_CLIP}/img1/000001.png"
    assert json.loads(video_run["calibration"].read_text(encoding="utf-8"))["video_sha256"] == sha
    document = json.loads(video_run["result"].read_text(encoding="utf-8"))
    assert document["provenance"]["tool_runs"]["detections"]["video_sha256"] == sha
    assert document["provenance"]["tool_runs"]["calibration"]["video_sha256"] == sha
    assert document["result"]["info"]["clip_id"] == VIDEO_CLIP


def run_command(video_run: dict[str, Path], clip: Path, out: Path, *more: str) -> list[str]:
    return [
        "run",
        "--clip",
        str(clip),
        "--config",
        str(DETECTOR_SETTINGS),
        "--detections",
        str(video_run["detections"]),
        "--calibration",
        str(video_run["calibration"]),
        "--out",
        str(out),
        "--allow-dirty",
        *more,
    ]


def test_run_refuses_a_clip_run_of_the_video_with_its_frame_folder(
    video_run: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(run_command(video_run, video_run["clip_folder"], tmp_path / "runs", "--fps", "25"))
    assert code == 2
    lines = error_lines(capsys)
    assert len(lines) == 1 and "was made from the frames of a video" in lines[0]
    assert "give that video file with --clip" in lines[0]


def test_run_refuses_a_clip_run_of_the_video_with_other_pixels(
    video_run: dict[str, Path], scene: SyntheticScene, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Another video with the same name and frame count: the clip and the count fit, the pixels do not."""
    other = write_video(
        tmp_path / f"{VIDEO_CLIP}.mp4", [scene.image(frame) for frame in reversed(range(N_FRAMES))], 25.0
    )
    assert cli.main(run_command(video_run, other, tmp_path / "runs")) == 2
    lines = error_lines(capsys)
    assert len(lines) == 1 and "was made from the frames of another video" in lines[0]


def test_run_refuses_a_clip_run_of_images_with_a_video(
    video_run: dict[str, Path], scene: SyntheticScene, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = json.loads((video_run["work"] / "jobs" / "pnlcalib.json").read_text(encoding="utf-8"))
    job["video_sha256"] = None
    job_file = tmp_path / "job.json"
    job_file.write_text(json.dumps(job), encoding="utf-8")
    calibration = write_clip_run(
        job_file, tmp_path / "calibration.json", [camera_payload(scene, frame) for frame in range(N_FRAMES)]
    )
    arguments = run_command(video_run, video_run["video"], tmp_path / "runs")
    arguments[arguments.index(str(video_run["calibration"]))] = str(calibration)
    assert cli.main(arguments) == 2
    lines = error_lines(capsys)
    assert len(lines) == 1 and "was made from a folder of images, not from this video" in lines[0]


def test_video_frames_refuses_a_place_in_a_work_tree_of_the_repository(
    video_run: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    inside = REPO / "outputs" / "video-frames-test-must-not-exist"
    assert cli.main(["video-frames", "--video", str(video_run["video"]), "--out", str(inside)]) == 2
    lines = error_lines(capsys)
    assert len(lines) == 1 and "rule 2 of CLAUDE.md" in lines[0]
    assert not inside.exists()
    # Another work tree of the repository, like the one of the results branch, is refused too.
    tree = tmp_path / "pitchvlm-results"
    tree.mkdir()
    monkeypatch.setattr(cli, "repository_work_trees", lambda: [REPO, tree])
    assert cli.main(["video-frames", "--video", str(video_run["video"]), "--out", str(tree / "videos")]) == 2
    assert "lies inside the work tree" in error_lines(capsys)[0]
    assert list(tree.iterdir()) == []


def test_repository_work_trees_holds_this_checkout() -> None:
    trees = cli.repository_work_trees()
    assert any(tree.exists() and tree.samefile(REPO) for tree in trees)


def test_video_frames_refuses_a_clip_folder_that_is_there(
    video_run: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    before = sorted(path.name for path in (video_run["clip_folder"] / "img1").iterdir())
    code = cli.main(["video-frames", "--video", str(video_run["video"]), "--out", str(video_run["root"] / "videos")])
    assert code == 2
    assert "is already there" in error_lines(capsys)[0]
    assert sorted(path.name for path in (video_run["clip_folder"] / "img1").iterdir()) == before


def test_video_with_a_frame_that_cannot_be_read_ends_with_one_error_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The end of a Motion JPEG video is cut off: no image is written. ``tests/test_frames.py`` has the other faults."""
    whole = tmp_path / "whole.avi"
    rng = np.random.default_rng(3)
    writer = cv2.VideoWriter(str(whole), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (128, 96))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write a Motion JPEG video")
    for _ in range(10):
        writer.write(rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8))
    writer.release()
    data = whole.read_bytes()
    cut = tmp_path / "cut.avi"
    cut.write_bytes(data[: len(data) - 15 * len(data) // 100])
    assert cli.main(["video-frames", "--video", str(cut), "--out", str(tmp_path / "videos")]) == 2
    lines = error_lines(capsys)
    assert len(lines) == 1 and lines[0].startswith("error: video ") and "its header names 10 frames" in lines[0]
    assert not (tmp_path / "videos").exists()
    assert cli.main(["video-frames", "--video", str(tmp_path / "missing.mp4"), "--out", str(tmp_path)]) == 2
    assert "is not a video file" in error_lines(capsys)[0]


def test_clip_job_refuses_a_video_record_that_does_not_fit_the_images(
    video_run: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "data"
    folder = root / VIDEO_CLIP
    shutil.copytree(video_run["clip_folder"], folder)
    (folder / "img1" / f"{N_FRAMES:06d}.png").unlink()
    arguments = ["clip-job", "--tool", "rfdetr", "--clip", str(folder), "--root", str(root)]
    assert cli.main([*arguments, "--out", str(tmp_path / "job.json")]) == 2
    lines = error_lines(capsys)
    assert len(lines) == 1 and f"names {N_FRAMES} frames" in lines[0] and f"holds {N_FRAMES - 1} images" in lines[0]
    (folder / "video.json").write_text('{"format": "pitchvlm-video-frames-1"}', encoding="utf-8")
    assert cli.main([*arguments, "--out", str(tmp_path / "job.json")]) == 2
    lines = error_lines(capsys)
    assert len(lines) == 1 and "video record" in lines[0]
    assert not (tmp_path / "job.json").exists()


# --- Choice of the frames to check by eye -----------------------------------------------------


def mapped(clip: str, frame: int, score: float | None) -> MappedFrame:
    reason = None if score is not None else "too little paint: 0.10% of the frame, the minimum is 0.30%"
    return MappedFrame(clip=clip, frame=frame, matrix=np.eye(3), origin="landmark fit", score=score, reason=reason)


def candidates() -> list[MappedFrame]:
    """Twenty frames of four clips; two cannot judge, the others have scores from 0.05 to 0.95."""
    frames = [mapped(f"C{index % 4}", index, round(0.05 * (index - 1), 2)) for index in range(2, 20)]
    return [mapped("C0", 0, None), mapped("C1", 1, None), *frames]


def test_choice_is_repeated_by_the_same_seed_and_spread_over_the_score_range() -> None:
    frames = candidates()
    first = choose_frames(frames, count=5, seed=11)
    again = choose_frames(list(reversed(frames)), count=5, seed=11)
    assert [(item.clip, item.frame) for item in first] == [(item.clip, item.frame) for item in again]
    # One frame from each fifth of the order: the parts are 4 frames each.
    order = sorted(frames, key=lambda item: (item.score is not None, item.score or 0.0))
    for part, item in enumerate(first):
        assert item in order[4 * part : 4 * part + 4]
    assert len({item.clip for item in first}) == 4  # Four clips; the fifth frame repeats one.


def test_choice_lets_the_clips_take_turns_and_another_seed_can_choose_otherwise() -> None:
    frames = candidates()
    chosen = {
        seed: [(item.clip, item.frame) for item in choose_frames(frames, count=10, seed=seed)] for seed in range(6)
    }
    for picks in chosen.values():
        clips = [clip for clip, _ in picks]
        assert max(clips.count(clip) for clip in set(clips)) - min(clips.count(clip) for clip in set(clips)) <= 1
    assert len({tuple(picks) for picks in chosen.values()}) > 1


def test_choice_takes_every_frame_when_there_are_too_few_and_refuses_wrong_input() -> None:
    frames = candidates()[:3]
    assert {(item.clip, item.frame) for item in choose_frames(frames, count=5, seed=0)} == {
        (item.clip, item.frame) for item in frames
    }
    with pytest.raises(ValueError, match="at least 1"):
        choose_frames(frames, count=0, seed=0)
    with pytest.raises(ValueError, match="more than once"):
        choose_frames([*frames, mapped("C0", 0, 0.5)], count=2, seed=0)
    with pytest.raises(ValueError, match="either a paint score or a reason"):
        MappedFrame(clip="C0", frame=0, matrix=np.eye(3), origin="x", score=None, reason=None)


def test_check_image_shows_the_lines_of_the_mapping_below_a_caption(scene: SyntheticScene) -> None:
    frame = 7
    image = scene.image(frame)
    item = mapped("C0", frame, 0.81)
    lines = caption(item, engine="tool:pnlcalib", position=1, count=5)
    assert lines[0] == f"2 of 5: clip C0, frame {frame} (image {frame + 1:06d})"
    assert "engine: tool:pnlcalib" in lines[1] and "paint alignment score 0.810" in lines[2]
    assert "cannot judge: too little paint" in caption(mapped("C0", 1, None), engine="e", position=0, count=1)[2]
    drawn = draw_mapping_check(image, scene.homography(frame), lines)
    band = drawn.shape[0] - image.shape[0]
    assert band > 0 and drawn.shape[1:] == image.shape[1:]
    assert drawn[:band].any(), "the caption band holds the text"
    framed = drawn[band:]
    line_pixels = projected_line_pixels(scene.homography(frame), (image.shape[1], image.shape[0]))
    on_line = np.zeros(image.shape[:2], dtype=bool)
    on_line[line_pixels[:, 1], line_pixels[:, 0]] = True
    assert on_line.sum() > 1000
    # Off the lines the frame is as it was; on a line it turns towards magenta (blue and red up, green down).
    assert np.array_equal(framed[~on_line], image[~on_line])
    moved = framed[on_line].astype(int) - image[on_line].astype(int)
    assert (moved[:, 1] <= 0).all() and (moved[:, 0] >= 0).all() and (moved[:, 2] >= 0).all()
    with pytest.raises(ValueError, match="BGR uint8"):
        draw_mapping_check(image.astype(np.float32), scene.homography(frame), lines)


# --- The command overlay ----------------------------------------------------------------------


def assert_list_file_is_plain_text(path: Path, forbidden: Sequence[Path]) -> dict[str, Any]:
    """Read a list file and check that it holds no absolute path and no label data."""
    text = path.read_text(encoding="utf-8")
    for folder in forbidden:
        assert str(folder) not in text
    document = json.loads(text)
    assert document["format"] == LIST_FORMAT and document["rule"] == CHOICE_RULE
    for entry in document["frames"]:
        assert set(entry) == {"image", "clip", "frame", "mapping", "paint_score", "paint_reason", "by_eye"}
        assert entry["by_eye"] is None
    return document


def overlay(source: str, tmp_path: Path, name: str, *more: str) -> list[str]:
    return ["overlay", source, *more, "--list", str(tmp_path / f"{name}.json"), "--out", str(tmp_path / "images")]


def test_overlay_of_a_run_draws_the_chosen_keyframes_and_lists_them(
    video_run: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = ["--result", str(video_run["result"]), "--clip", str(video_run["video"]), "--seed", "3"]
    assert cli.main(overlay("run", tmp_path, "list", *arguments, "--count", "4")) == 0
    printed = capsys.readouterr().out
    document = assert_list_file_is_plain_text(tmp_path / "list.json", [tmp_path, video_run["work"]])
    keyframes = len(range(0, N_FRAMES, 12)) + 1
    assert (document["source"], document["name"], document["engine"]) == ("run", VIDEO_CLIP, "tool:pnlcalib")
    assert (document["n_frames"], document["n_frames_with_mapping"], document["n_chosen"]) == (keyframes, keyframes, 4)
    assert document["provenance"]["result_file_sha256"] == video_sha256(video_run["result"])
    folder = tmp_path / "images" / f"{VIDEO_CLIP}-seed3-count4"
    names = [entry["image"] for entry in document["frames"]]
    assert sorted(path.name for path in folder.iterdir()) == sorted(names)
    for entry in document["frames"]:
        assert entry["mapping"] == "camera of the tool for this frame"
        assert entry["paint_score"] > 0.9  # The true camera puts the lines on the paint.
        assert f"{entry['image']}: clip {VIDEO_CLIP}, frame {entry['frame']}" in printed
        image = cv2.imread(str(folder / entry["image"]))
        assert image.shape[1] == 1280 and image.shape[0] > 720
    assert "into its field by_eye in the list file" in printed
    assert document["frames_without_mapping"] == []
    # The same seed and count choose the same frames, also from the frame folder of the video.
    other = tmp_path / "again"
    folder_arguments = ["--result", str(video_run["result"]), "--clip", str(video_run["clip_folder"]), "--seed", "3"]
    assert cli.main(overlay("run", other, "list", *folder_arguments, "--count", "4")) == 0
    again = json.loads((other / "list.json").read_text(encoding="utf-8"))
    assert [entry["frame"] for entry in again["frames"]] == [entry["frame"] for entry in document["frames"]]


def test_overlay_refuses_a_list_file_that_is_there_and_images_in_the_repository(
    video_run: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = ["--result", str(video_run["result"]), "--clip", str(video_run["video"]), "--seed", "1", "--count", "2"]
    (tmp_path / "list.json").write_text("{}", encoding="utf-8")
    assert cli.main(overlay("run", tmp_path, "list", *arguments)) == 2
    assert "may hold the verdicts of a check by eye" in error_lines(capsys)[0]
    inside = REPO / "outputs" / "overlay-test-must-not-exist"
    command = ["overlay", "run", *arguments, "--list", str(tmp_path / "other.json"), "--out", str(inside)]
    assert cli.main(command) == 2
    assert "rule 2 of CLAUDE.md" in error_lines(capsys)[0]
    assert not inside.exists() and not (tmp_path / "other.json").exists()


def test_overlay_refuses_a_clip_that_is_not_the_clip_of_the_result(
    video_run: dict[str, Path], scene: SyntheticScene, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    short = write_video(tmp_path / f"{VIDEO_CLIP}.mp4", [scene.image(frame) for frame in range(10)], 25.0)
    arguments = ["--result", str(video_run["result"]), "--clip", str(short), "--seed", "1", "--count", "2"]
    assert cli.main(overlay("run", tmp_path, "list", *arguments)) == 2
    lines = error_lines(capsys)
    assert len(lines) == 1 and "but the result file" in lines[0]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory, scene: SyntheticScene) -> tuple[Path, Path]:
    """The scene in the layout of the dataset, with its labels, and a split list whose group ``probe`` holds it."""
    root = tmp_path_factory.mktemp("dataset")
    clip_dir = root / "SoccerNetGS-valid" / DATASET_CLIP
    (clip_dir / "img1").mkdir(parents=True)
    for frame in range(N_FRAMES):
        assert cv2.imwrite(str(clip_dir / "img1" / f"{frame + 1:06d}.png"), scene.image(frame))
    (clip_dir / LABEL_NAME).write_text(json.dumps(scene.gsr_labels()), encoding="utf-8")
    split_list = root / "splits.json"
    split_list.write_text(json.dumps({"seed": 7, "valid": {"probe": [DATASET_CLIP]}}), encoding="utf-8")
    return root, split_list


def test_overlay_of_a_calibration_run_file_uses_the_frames_of_its_probe(
    dataset: tuple[Path, Path], scene: SyntheticScene, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, split_list = dataset
    experiment = EXPERIMENTS / "pitch_probe_pnlcalib.yaml"
    group = ["--root", str(root), "--manifest", str(split_list)]
    assert cli.main(["tool-job", "--experiment", str(experiment), *group, "--out", str(tmp_path / "job.json")]) == 0
    job = json.loads((tmp_path / "job.json").read_text(encoding="utf-8"))
    run = {
        "format": "pitchvlm-tool-run-1",
        "name": job["name"],
        "tool": job["tool"],
        "runner": {"device": "stand-in"},
        "frames": [
            {
                "clip": entry["clip"],
                "frame": entry["frame"],
                "seconds": 0.1,
                "payload": camera_payload(scene, entry["frame"]),
            }
            for entry in job["frames"]
        ],
    }
    run_file = tmp_path / "run.json"
    run_file.write_text(json.dumps(run), encoding="utf-8")
    capsys.readouterr()
    arguments = ["--experiment", str(experiment), *group, "--run", str(run_file), "--seed", "1", "--count", "5"]
    assert cli.main(overlay("tool", tmp_path, "list", *arguments)) == 0
    document = assert_list_file_is_plain_text(tmp_path / "list.json", [tmp_path, root])
    assert (document["engine"], document["n_frames"], document["n_chosen"]) == ("tool:pnlcalib", 5, 5)
    assert sorted(entry["frame"] for entry in document["frames"]) == sorted(entry["frame"] for entry in job["frames"])
    assert document["provenance"]["run_file_sha256"] == video_sha256(run_file)
    assert document["provenance"]["clips"] == [DATASET_CLIP]
    # A detector has no mapping.
    detector = ["--experiment", str(EXPERIMENTS / "people_probe_rfdetr_small.yaml"), *group, "--run", str(run_file)]
    assert cli.main(overlay("tool", tmp_path, "other", *detector, "--seed", "1", "--count", "5")) == 2
    assert "needs a calibration tool" in error_lines(capsys)[0]


def test_overlay_of_the_pitch_probe_replays_its_stored_answers_only(
    dataset: tuple[Path, Path],
    scene: SyntheticScene,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, split_list = dataset
    experiment = EXPERIMENTS / "pitch_probe.yaml"
    group = ["--root", str(root), "--manifest", str(split_list)]
    cache = ["--cache", str(tmp_path / "cache")]
    with monkeypatch.context() as patch:
        patch.setattr(cli, "model_backend", lambda max_usd_per_10_min: FakeBackend(scene, point_noise_px=4.0))
        metrics = tmp_path / "pitch_probe.json"
        probe = ["probe", "pitch", "--experiment", str(experiment), *group, *cache, "--allow-dirty"]
        assert cli.main([*probe, "--out", str(metrics)]) == 0
    measured = json.loads(metrics.read_text(encoding="utf-8"))

    def refuse(max_usd_per_10_min: float) -> None:
        raise AssertionError("the overlay must not build a model backend")

    monkeypatch.setattr(cli, "model_backend", refuse)
    capsys.readouterr()
    arguments = ["--experiment", str(experiment), *group, *cache, "--seed", "2", "--count", "3"]
    assert cli.main(overlay("probe", tmp_path, "list", *arguments)) == 0
    assert "stored answers only: no model call" in capsys.readouterr().out
    document = assert_list_file_is_plain_text(tmp_path / "list.json", [tmp_path, root])
    assert document["engine"] == "model:gemini-4-argon landmarks, fit by code"
    # The overlay fits what the probe measured: the same frames, and the same frames with a mapping.
    assert document["n_frames"] == measured["n_frames"]
    assert document["n_frames_with_mapping"] == measured["n_frames_valid"]
    assert document["n_chosen"] == 3 and {entry["mapping"] for entry in document["frames"]} == {
        "landmark fit of this frame"
    }
    # Without the stored answers the command stops with one line and calls no model.
    empty = ["--experiment", str(experiment), *group, "--cache", str(tmp_path / "empty"), "--seed", "2", "--count", "3"]
    assert cli.main(overlay("probe", tmp_path, "missing", *empty)) == 2
    lines = error_lines(capsys)
    assert len(lines) == 1 and lines[0].startswith("error:")
    assert not (tmp_path / "missing.json").exists()
    # A people probe gives no mapping.
    people = ["--experiment", str(EXPERIMENTS / "people_probe.yaml"), *group, *cache, "--seed", "2", "--count", "3"]
    assert cli.main(overlay("probe", tmp_path, "people", *people)) == 2
    assert "an overlay needs the pitch probe" in error_lines(capsys)[0]


def test_parser_reads_the_two_commands() -> None:
    parser = cli.build_parser()
    frames = parser.parse_args(["video-frames", "--video", "clip.mp4"])
    assert (frames.command, frames.video) == ("video-frames", Path("clip.mp4"))
    assert frames.out == cli.VIDEO_FRAMES_DIR.expanduser()
    run = parser.parse_args(
        ["overlay", "run", "--result", "r.json", "--clip", "c.mp4", "--seed", "1", "--count", "5", "--list", "l.json"]
    )
    assert (run.command, run.source, run.seed, run.count) == ("overlay", "run", 1, 5)
    assert run.out == cli.OVERLAYS_DIR.expanduser()
    assert callable(run.handler)
    no_count = ["overlay", "run", "--result", "r.json", "--clip", "c.mp4", "--seed", "1", "--list", "l.json"]
    with pytest.raises(SystemExit):
        parser.parse_args([*no_count, "--count", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["overlay", "--seed", "1"])


def test_image_name_orders_the_images_by_their_place_in_the_list() -> None:
    assert image_name(0, mapped("SNGS-021", 37, 0.5)) == "01_SNGS-021_000038.png"
    assert image_name(9, mapped("SNGS-021", 0, 0.5)) == "10_SNGS-021_000001.png"


# --- A video with a dissolve -------------------------------------------------------------------

#: The dissolve clip: 40 frames of one moment, a dissolve of 10 frames, 40 frames of a moment 4 s later.
DISSOLVE_START, DISSOLVE_FRAMES = 40, 10
DISSOLVE_CLIP_FRAMES = 2 * DISSOLVE_START + DISSOLVE_FRAMES


def dissolve_images() -> list[np.ndarray]:
    """The frames of the dissolve clip: shot A blended into shot B (scene frames 100 later) over 10 frames."""
    old = RetimedScene(list(range(DISSOLVE_CLIP_FRAMES)), seed=0)
    new = RetimedScene([frame + 100 for frame in range(DISSOLVE_CLIP_FRAMES)], seed=0)
    images = []
    for frame in range(DISSOLVE_CLIP_FRAMES):
        weight = min(1.0, max(0.0, (frame - DISSOLVE_START + 1) / (DISSOLVE_FRAMES + 1)))
        blend = (1.0 - weight) * old.image(frame).astype(np.float32) + weight * new.image(frame)
        images.append(np.clip(blend, 0, 255).astype(np.uint8))
    return images


def test_a_video_with_a_dissolve_keeps_its_frames_and_overlay_names_the_transition_keyframes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The frames of a dissolve reach the tools as `run` decodes them, and get no mapping and no image.

    The stand-in tools answer with the truth of the hard cut in the middle of the dissolve, as in
    ``tests/test_shot_cuts.py``. The run drops the mapping of every transition frame; the overlay of
    the run lists each transition keyframe with that reason and draws no image of it.
    """
    monkeypatch.setattr(cli, "render_clip", lambda source, result, out_path: out_path)
    middle = DISSOLVE_START + DISSOLVE_FRAMES // 2
    truth = RetimedScene(
        [frame if frame < middle else frame + 100 for frame in range(DISSOLVE_CLIP_FRAMES)], seed=0
    )
    root = tmp_path / "data"
    video = write_video(root / "footage" / "dissolve.mp4", dissolve_images(), 25.0)
    assert cli.main(["video-frames", "--video", str(video), "--out", str(root / "videos")]) == 0
    folder = root / "videos" / "dissolve"
    images = ImageFolderSource(folder / "img1", "dissolve", fps=25.0)
    decoded = VideoFileSource(video)
    assert images.info.n_frames == decoded.info.n_frames == DISSOLVE_CLIP_FRAMES
    for frame in range(DISSOLVE_CLIP_FRAMES):
        assert np.array_equal(images.read(frame), decoded.read(frame)), f"frame {frame}"
    runs = {}
    for tool, name, payloads in (
        ("rfdetr", "detections.json", detector_payloads(truth)),
        ("pnlcalib", "calibration.json", [camera_payload(truth, frame) for frame in range(DISSOLVE_CLIP_FRAMES)]),
    ):
        job = tmp_path / "jobs" / f"{tool}.json"
        arguments = ["clip-job", "--tool", tool, "--config", str(DETECTOR_SETTINGS), "--clip", str(folder)]
        assert cli.main([*arguments, "--root", str(root), "--out", str(job)]) == 0
        runs[name] = write_clip_run(job, tmp_path / "tools" / name, payloads)
    run = ["run", "--clip", str(video), "--config", str(DETECTOR_SETTINGS), "--allow-dirty"]
    run += ["--detections", str(runs["detections.json"]), "--calibration", str(runs["calibration.json"])]
    assert cli.main([*run, "--out", str(tmp_path / "runs")]) == 0
    result_file = tmp_path / "runs" / "dissolve" / "result.json"
    result = json.loads(result_file.read_text(encoding="utf-8"))["result"]
    cuts = result["cuts"]
    transitions = {
        start for start, end in zip([0, *cuts], [*cuts, DISSOLVE_CLIP_FRAMES], strict=True) if end - start == 1
    }
    keyframes = [*range(0, DISSOLVE_CLIP_FRAMES, 12), DISSOLVE_CLIP_FRAMES - 1]
    on_transition = sorted(transitions & set(keyframes))
    assert on_transition, f"the test needs a keyframe on a transition frame; transitions {sorted(transitions)}"
    assert all(result["homographies"][frame]["matrix"] is None for frame in on_transition)
    capsys.readouterr()

    arguments = ["--result", str(result_file), "--clip", str(video), "--seed", "20261113"]
    assert cli.main(overlay("run", tmp_path, "list", *arguments, "--count", str(len(keyframes)))) == 0
    printed = capsys.readouterr().out
    document = assert_list_file_is_plain_text(tmp_path / "list.json", [tmp_path])
    reason = "a frame of a transition (a dissolve, or a lone frame between two cuts): the run gives it no mapping"
    listed = [entry for entry in document["frames_without_mapping"] if entry["reason"] == reason]
    assert [entry["frame"] for entry in listed] == on_transition
    assert f"  {reason}: " + ", ".join(f"dissolve {frame}" for frame in on_transition) in printed
    drawn = {entry["frame"] for entry in document["frames"]}
    assert not drawn & set(on_transition)
    assert document["n_frames_with_mapping"] == len(keyframes) - len(document["frames_without_mapping"])
    assert document["n_chosen"] == document["n_frames_with_mapping"]
