"""Tests of the detector families D-FINE and RT-DETRv2: the settings, the script and the package.

Neither family is installed where these tests run, and the script was never run with them. So
``transformers`` and ``torch`` are replaced by stand-ins with the interface of transformers 5.23.0
that the script uses (read from its source at the tag v5.23.0):

- ``AutoImageProcessor.from_pretrained(repo, revision=...)`` gives an image processor. Calling it
  with ``images``, ``size`` and ``return_tensors`` gives ``pixel_values`` of the shape
  (batch, 3, height, width). Its ``post_process_object_detection(outputs, threshold,
  target_sizes)`` is written here from the source of ``RTDetrImageProcessor``: the logistic
  function of the class scores, the best pairs of query and class, the boxes from centre format
  in parts of the input to corners in pixels of the target size (height, width), and the scores
  above the threshold.
- ``AutoModelForObjectDetection.from_pretrained(repo, revision=..., dtype=..., use_safetensors=...)``
  gives a model with ``config`` (``model_type``, ``use_focal_loss``, ``id2label``). Calling it with
  ``pixel_values`` gives ``logits`` (batch, queries, classes) and ``pred_boxes`` (batch, queries,
  4). The stand-in model answers each frame from the red value of its first pixel, which the
  stand-in processor keeps.

The stand-ins say what the script calls and how it reads the answer. They cannot say that the
real package behaves so: that is checked on the run machine, and the script checks its boxes
against ``post_process_object_detection`` of the real package on the first call.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import math
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from pitchvlm import cli
from pitchvlm.config import DEFAULT_DETECTORS, PipelineConfig, load_config
from pitchvlm.fallbacks import runs
from pitchvlm.fallbacks.answers import detector_answers
from pitchvlm.fallbacks.runs import TransformersDetectorSettings
from pitchvlm.pipeline import _engines

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools" / "run_tool.py"
EXPERIMENTS = REPO / "configs" / "experiments"
#: Size of the small frames of these tests, pixels: not square, so that a swap of width and height shows.
WIDTH, HEIGHT = 200, 100
#: The class names of the stand-in weights: three classes, with the two that the package reads.
LABELS = {0: "person", 1: "car", 2: "sports ball"}
DFINE = {"name": "dfine", "model": "medium", "min_score": 0.5, "input_size": 640, "batch": 1, "half": False}


def load_script() -> ModuleType:
    """Load the script as a module of its own name, apart from the copies of the other test files."""
    spec = importlib.util.spec_from_file_location("run_tool_for_families", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_tool_for_families"] = module
    spec.loader.exec_module(module)
    return module


run_tool = load_script()


def logit(score: float) -> float:
    """Return the class score before the logistic function."""
    return math.log(score / (1.0 - score))


# --- The conversion of the output of the model -----------------------------------------


def test_boxes_go_from_centre_format_in_parts_of_the_input_to_corners_in_pixels_of_the_frame() -> None:
    """A frame of 200 x 100 pixels: x scales with the width and y with the height."""
    logits = np.full((2, 3), logit(0.01))
    logits[0, 0] = logit(0.9)  # query 0: a person
    logits[1, 2] = logit(0.7)  # query 1: a ball
    boxes = np.array([[0.5, 0.25, 0.1, 0.2], [0.25, 0.75, 0.05, 0.1]])

    payload = run_tool.detections_of_queries(logits, boxes, width=WIDTH, height=HEIGHT, labels=LABELS, min_score=0.5)

    person, ball = payload["detections"]
    assert person["label"] == "person" and person["score"] == pytest.approx(0.9)
    assert person["box_xyxy"] == pytest.approx([90.0, 15.0, 110.0, 35.0])
    assert ball["label"] == "sports ball" and ball["score"] == pytest.approx(0.7)
    assert ball["box_xyxy"] == pytest.approx([45.0, 70.0, 55.0, 80.0])


def test_conversion_keeps_the_two_classes_of_the_package_and_the_scores_from_the_limit() -> None:
    logits = np.full((3, 3), logit(0.01))
    logits[0, 1] = logit(0.95)  # a car: not a class of the package
    logits[1, 0] = logit(0.5)  # a person exactly at the limit: kept
    logits[2, 0] = logit(0.49)  # a person below the limit: dropped
    boxes = np.tile([0.5, 0.5, 0.2, 0.2], (3, 1))

    payload = run_tool.detections_of_queries(logits, boxes, width=WIDTH, height=HEIGHT, labels=LABELS, min_score=0.5)

    assert [(d["label"], round(d["score"], 6)) for d in payload["detections"]] == [("person", 0.5)]


def test_conversion_takes_the_best_pairs_of_query_and_class_as_many_as_there_are_queries() -> None:
    """Two queries, three pairs above the limit: the tool keeps the best two pairs, so the script does too.

    Query 0 is a person (0.9) and a ball (0.8); query 1 is a person (0.6). The two best pairs are
    both of query 0, so the person of query 1 is not kept, and query 0 gives two detections.
    """
    logits = np.full((2, 3), logit(0.01))
    logits[0, 0], logits[0, 2], logits[1, 0] = logit(0.9), logit(0.8), logit(0.6)
    boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]])

    payload = run_tool.detections_of_queries(logits, boxes, width=WIDTH, height=HEIGHT, labels=LABELS, min_score=0.5)

    found = [(d["label"], round(d["score"], 6), d["box_xyxy"]) for d in payload["detections"]]
    assert [(label, score) for label, score, _ in found] == [("person", 0.9), ("sports ball", 0.8)]
    assert found[0][2] == found[1][2] == pytest.approx([80.0, 40.0, 120.0, 60.0])


@pytest.mark.parametrize("where", ["logits", "boxes"])
@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_output_with_a_value_that_is_not_finite_is_refused(where: str, value: float) -> None:
    logits, boxes = np.zeros((2, 3)), np.full((2, 4), 0.5)
    (logits if where == "logits" else boxes)[1, 1] = value
    with pytest.raises(run_tool.RunnerError, match="not a finite number"):
        run_tool.detections_of_queries(logits, boxes, width=WIDTH, height=HEIGHT, labels=LABELS, min_score=0.5)


@pytest.mark.parametrize(
    ("logits", "boxes", "words"),
    [
        (np.zeros((2, 3)), np.zeros((3, 4)), "expected (queries, classes)"),
        (np.zeros((2, 3)), np.zeros((2, 2)), "expected (queries, classes)"),
        (np.zeros((2, 4)), np.zeros((2, 4)), "4 class scores per query"),
    ],
)
def test_output_of_another_shape_is_refused(logits: np.ndarray, boxes: np.ndarray, words: str) -> None:
    with pytest.raises(run_tool.RunnerError, match=words.replace("(", r"\(").replace(")", r"\)")):
        run_tool.detections_of_queries(logits, boxes, width=WIDTH, height=HEIGHT, labels=LABELS, min_score=0.5)


def test_check_matches_detections_whose_scores_tie_in_float32_but_not_in_float64() -> None:
    """The tool orders two people of one score by box, the script by its float64 score: the check still passes."""
    logits = np.full((2, 3), logit(0.01))
    # Two float32 logits one step apart: the model gives float32.
    # The higher float64 score has the box on the left; sorted by (score, box), the two sides then differ.
    logits[0, 0], logits[1, 0] = float(np.nextafter(np.float32(4.0), np.float32(5.0))), 4.0
    boxes = np.array([[0.05, 0.5, 0.02, 0.2], [0.85, 0.5, 0.02, 0.2]])
    payload = run_tool.detections_of_queries(logits, boxes, width=WIDTH, height=HEIGHT, labels=LABELS, min_score=0.5)
    first, second = payload["detections"]
    assert first["score"] > second["score"] and first["box_xyxy"][0] < second["box_xyxy"][0]
    batch_logits, batch_boxes = logits[None].astype(np.float32), boxes[None].astype(np.float32)
    (result,) = reference_post_process(batch_logits, batch_boxes, 0.5, [(HEIGHT, WIDTH)])
    tool = run_tool.tool_detections(result, LABELS, 0.5)
    assert tool[0][1] == tool[1][1]  # float32: the two scores are equal
    run_tool.check_against_tool(payload, sorted(tool, key=lambda item: item[2]), 0.5)
    run_tool.check_against_tool(payload, sorted(tool, key=lambda item: item[2], reverse=True), 0.5)


@pytest.mark.parametrize("change", ["one more", "one less", "other class"])
def test_check_refuses_a_detection_that_the_other_side_does_not_have(change: str) -> None:
    payload = {"detections": [{"label": "person", "score": 0.9, "box_xyxy": [1.0, 2.0, 3.0, 4.0]}]}
    tool = [("person", 0.9, [1.0, 2.0, 3.0, 4.0])]
    if change == "one more":
        tool.append(("person", 0.8, [5.0, 6.0, 7.0, 8.0]))
    elif change == "one less":
        tool = []
    else:
        tool = [("sports ball", 0.9, [1.0, 2.0, 3.0, 4.0])]
    run_tool.check_against_tool(payload, [("person", 0.9, [1.0, 2.0, 3.0, 4.005])], 0.5)
    with pytest.raises(run_tool.RunnerError, match="differ from those of post_process_object_detection"):
        run_tool.check_against_tool(payload, tool, 0.5)


def test_large_negative_scores_give_no_overflow() -> None:
    with np.errstate(over="raise"):
        assert run_tool.sigmoid(np.array([-1000.0, 0.0, 1000.0])).tolist() == pytest.approx([0.0, 0.5, 1.0])


# --- Stand-ins of torch and transformers -----------------------------------------------


class Tensor:
    """A NumPy array with the tensor methods that the script and the stand-ins use."""

    def __init__(self, values: Any, device: str = "cpu", dtype: str = "float32") -> None:
        self.values = np.asarray(values)
        self.device, self.dtype = device, dtype

    def to(self, device: str | None = None, dtype: str | None = None) -> Tensor:
        return Tensor(self.values, device or self.device, dtype or self.dtype)

    def float(self) -> Tensor:
        return Tensor(self.values.astype(np.float32), self.device, "float32")

    def cpu(self) -> Tensor:
        return Tensor(self.values, "cpu", self.dtype)

    def numpy(self) -> np.ndarray:
        assert self.device == "cpu", "a tensor on a GPU must go to the CPU first"
        return self.values

    def tolist(self) -> Any:
        return self.values.tolist()

    def __len__(self) -> int:
        return len(self.values)


def reference_post_process(
    logits: np.ndarray, boxes: np.ndarray, threshold: float, target_sizes: list[tuple[int, int]]
) -> list[dict[str, Tensor]]:
    """``post_process_object_detection`` of ``RTDetrImageProcessor`` (transformers 5.23.0), with focal loss."""
    results = []
    for frame_logits, frame_boxes, (height, width) in zip(logits, boxes, target_sizes, strict=True):
        centre_x, centre_y, box_width, box_height = frame_boxes.T
        corners = np.stack(
            [
                centre_x - 0.5 * box_width,
                centre_y - 0.5 * box_height,
                centre_x + 0.5 * box_width,
                centre_y + 0.5 * box_height,
            ],
            axis=1,
        ) * np.array([width, height, width, height])
        queries, classes = frame_logits.shape
        scores = 1.0 / (1.0 + np.exp(-frame_logits.ravel()))
        index = np.argsort(-scores, kind="stable")[:queries]
        top, label, query = scores[index], index % classes, index // classes
        keep = top > threshold
        results.append(
            {"scores": Tensor(top[keep]), "labels": Tensor(label[keep]), "boxes": Tensor(corners[query][keep])}
        )
    return results


class Processor:
    """Stand-in of the image processor. It keeps the red value of the first pixel of each frame."""

    calls: list[dict[str, Any]] = []
    post_processed: list[dict[str, Any]] = []
    #: A change of the reference answer, to test the check of the script.
    change: Callable[[list[dict[str, Tensor]]], None] = staticmethod(lambda results: None)

    def __init__(self, repository: str, revision: str | None) -> None:
        self.repository, self.revision = repository, revision

    def __call__(
        self, images: Any = None, *, size: dict[str, int] | None = None, return_tensors: str | None = None
    ) -> Any:
        assert isinstance(images, list) and return_tensors == "pt"
        Processor.calls.append({"n": len(images), "size": size, "shapes": [image.shape for image in images]})
        assert size is not None
        pixels = np.zeros((len(images), 3, size["height"], size["width"]), dtype=np.float32)
        for position, image in enumerate(images):
            pixels[position, 0, 0, 0] = image[0, 0, 0]
        return {"pixel_values": Tensor(pixels)}

    def post_process_object_detection(
        self, outputs: Any, threshold: float = 0.5, target_sizes: Any = None
    ) -> list[dict[str, Tensor]]:
        assert outputs.logits.dtype == outputs.pred_boxes.dtype == "float32"
        Processor.post_processed.append({"threshold": threshold, "target_sizes": target_sizes})
        results = reference_post_process(outputs.logits.values, outputs.pred_boxes.values, threshold, target_sizes)
        type(self).change(results)
        return results


class Model:
    """Stand-in of a detector of transformers. ``answers`` gives the logits and the boxes of a red value."""

    made: list[Model] = []
    calls: list[dict[str, Any]] = []
    config = SimpleNamespace(model_type="d_fine", use_focal_loss=True, id2label={str(k): v for k, v in LABELS.items()})

    def __init__(self, repository: str, revision: str | None, dtype: str, use_safetensors: bool) -> None:
        self.repository, self.revision, self.dtype, self.use_safetensors = repository, revision, dtype, use_safetensors
        self.device, self.in_eval = None, False
        self.config = type(self).config
        Model.made.append(self)

    @staticmethod
    def answers(red: int) -> tuple[np.ndarray, np.ndarray]:
        """A person in the middle of the frame, with a score that grows with the red value."""
        logits = np.full((3, 3), logit(0.01))
        logits[0, 0] = logit(0.6 + red / 1000)
        return logits, np.array([[0.5, 0.5, 0.1, 0.4], [0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]])

    def to(self, device: str) -> Model:
        self.device = device
        return self

    def eval(self) -> Model:
        self.in_eval = True
        return self

    def __call__(self, *, pixel_values: Tensor) -> Any:
        assert pixel_values.device == self.device and pixel_values.dtype == self.dtype
        Model.calls.append({"n": len(pixel_values), "shape": pixel_values.values.shape})
        found = [type(self).answers(int(red)) for red in pixel_values.values[:, 0, 0, 0]]
        return SimpleNamespace(
            logits=Tensor(np.stack([f[0] for f in found]), self.device, self.dtype),
            pred_boxes=Tensor(np.stack([f[1] for f in found]), self.device, self.dtype),
        )


@pytest.fixture
def transformers(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Put the stand-ins of torch and transformers at the place of the packages."""
    synchronized: list[str] = []
    torch = ModuleType("torch")
    torch.float16, torch.float32 = "float16", "float32"

    @contextlib.contextmanager
    def inference_mode() -> Iterator[None]:
        yield

    torch.inference_mode = inference_mode
    torch.cuda = SimpleNamespace(synchronize=lambda device=None: synchronized.append(f"cuda {device}"))
    torch.mps = SimpleNamespace(synchronize=lambda: synchronized.append("mps"))
    module = ModuleType("transformers")
    module.AutoImageProcessor = SimpleNamespace(
        from_pretrained=lambda repository, revision=None: Processor(repository, revision)
    )
    module.AutoModelForObjectDetection = SimpleNamespace(
        from_pretrained=lambda repository, revision=None, dtype=None, use_safetensors=None: Model(
            repository, revision, dtype, use_safetensors
        )
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setattr(run_tool, "installed_version", {"transformers": "5.23.0", "torch": "2.15.0"}.__getitem__)
    monkeypatch.setattr(run_tool, "read_rgb", read_rgb_with_opencv)
    for target, name in ((Processor, "calls"), (Processor, "post_processed"), (Model, "made"), (Model, "calls")):
        monkeypatch.setattr(target, name, [])
    return SimpleNamespace(synchronized=synchronized)


def read_rgb_with_opencv(path: Path) -> np.ndarray:
    """Read an image file as RGB. It stands in for the reader of the script, which needs Pillow."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert image is not None, path
    return np.ascontiguousarray(image[:, :, ::-1])


def small_frames(root: Path, count: int) -> list[dict[str, Any]]:
    """Write frames of 200 x 100 pixels below a data folder; frame i has the red value 30 + i."""
    folder = root / "SoccerNetGS-valid" / "CLIP-A" / "img1"
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for index in range(count):
        image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        image[:] = (10, 20, 30 + index)  # OpenCV writes blue, green, red.
        name = f"{index + 1:06d}.png"
        assert cv2.imwrite(str(folder / name), image)
        entries.append({"clip": "CLIP-A", "frame": index, "image": f"SoccerNetGS-valid/CLIP-A/img1/{name}"})
    return entries


def run_family(
    tmp_path: Path, settings: dict[str, Any], count: int = 3, device: str = "cuda:0", clip: bool = False
) -> tuple[int, Path]:
    """Run the script on a job of ``count`` small frames. Return the exit code and the run file."""
    frames = small_frames(tmp_path / "data", count)
    if clip:
        job = {
            "format": runs.CLIP_JOB_FORMAT,
            "clip": "CLIP-A",
            "tool": settings,
            "images": [entry["image"] for entry in frames],
        }
    else:
        job = {"format": runs.JOB_FORMAT, "name": "a_job", "tool": settings, "frames": frames}
    (tmp_path / "job.json").write_text(json.dumps(job), encoding="utf-8")
    out = tmp_path / "runs" / "run.json"
    arguments = [settings["name"], "--job", str(tmp_path / "job.json"), "--root", str(tmp_path / "data")]
    return run_tool.main([*arguments, "--out", str(out), "--device", device]), out


def error_line(capsys: pytest.CaptureFixture[str]) -> str:
    return capsys.readouterr().err.splitlines()[-1]


# --- The script with the stand-ins -----------------------------------------------------


def test_dfine_job_loads_the_pinned_checkpoint_and_writes_boxes_in_pixels_of_the_frame(
    tmp_path: Path, transformers: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run_family(tmp_path, dict(DFINE))
    assert code == 0, capsys.readouterr().err

    (model,) = Model.made
    repository, revision = run_tool.TRANSFORMERS_CHECKPOINTS["dfine"]["medium"]
    assert (model.repository, model.revision) == ("ustc-community/dfine-medium-coco", revision)
    assert (model.dtype, model.use_safetensors, model.device, model.in_eval) == ("float32", True, "cuda:0", True)
    # The frames go to the processor at the input size of the job: the warm-up frame, then each frame.
    assert [call["n"] for call in Processor.calls] == [1, 1, 1, 1]
    assert {(call["size"]["height"], call["size"]["width"]) for call in Processor.calls} == {(640, 640)}
    assert Model.calls[0]["shape"] == (1, 3, 640, 640)
    # The model time is read after the GPU has done its work: twice per call.
    assert transformers.synchronized == ["cuda cuda:0"] * 8

    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["tool"] == DFINE
    for position, entry in enumerate(document["frames"]):
        (person,) = entry["payload"]["detections"]
        assert person["label"] == "person"
        assert person["score"] == pytest.approx(0.6 + (30 + position) / 1000)
        assert person["box_xyxy"] == pytest.approx([90.0, 30.0, 110.0, 70.0])
    runner = document["runner"]
    assert runner["tool_version"] == "5.23.0"
    assert (runner["checkpoint"], runner["checkpoint_revision"]) == (repository, revision)
    assert (runner["model_type"], runner["dtype"]) == ("d_fine", "float32")
    assert set(runner["stage_ms_per_frame"]) == {"preprocess", "model", "postprocess"}
    assert all(value >= 0 for value in runner["stage_ms_per_frame"].values())
    # The script checked its boxes against those of the tool on the warm-up call, with the size of the frame.
    assert Processor.post_processed == [
        {"threshold": run_tool.SELF_CHECK_MIN_SCORE, "target_sizes": [(HEIGHT, WIDTH)]}
    ]
    printed = capsys.readouterr().out
    assert "median time per frame of each part: preprocess" in printed


def test_rtdetrv2_runs_through_the_same_adapter(
    tmp_path: Path, transformers: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Model, "config", SimpleNamespace(**{**vars(Model.config), "model_type": "rt_detr_v2"}))
    settings = {**DFINE, "name": "rtdetrv2", "model": "r18vd"}
    code, out = run_family(tmp_path, settings, device="mps")
    assert code == 0
    assert Model.made[0].repository == "PekingU/rtdetr_v2_r18vd"
    assert transformers.synchronized == ["mps"] * 8
    assert json.loads(out.read_text(encoding="utf-8"))["runner"]["model_type"] == "rt_detr_v2"


def test_batch_of_frames_goes_through_the_model_together_and_shares_its_time(
    tmp_path: Path, transformers: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five frames in batches of 2: the warm-up batch, then 2, 2 and 1 frames.

    A call of the model takes 1 s per frame of its batch, and the warm-up call 100 s more: it is
    in no time of the run file and in no stage time.
    """
    now = [0.0]

    def call(self: Model, *, pixel_values: Tensor) -> Any:
        now[0] += float(len(pixel_values)) + (100.0 if not Model.calls else 0.0)
        return original(self, pixel_values=pixel_values)

    original = Model.__call__
    monkeypatch.setattr(Model, "__call__", call)
    monkeypatch.setattr(run_tool, "time", SimpleNamespace(perf_counter=lambda: now[0]))
    code, out = run_family(tmp_path, {**DFINE, "batch": 2}, count=5)
    assert code == 0
    assert [call["n"] for call in Processor.calls] == [2, 2, 2, 1]
    document = json.loads(out.read_text(encoding="utf-8"))
    assert [entry["frame"] for entry in document["frames"]] == [0, 1, 2, 3, 4]
    assert [entry["seconds"] for entry in document["frames"]] == [1.0] * 5
    assert document["runner"]["stage_ms_per_frame"]["model"] == pytest.approx(1000.0)


def test_warm_up_call_is_in_no_stage_time(
    tmp_path: Path, transformers: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One frame: the warm-up call of the model takes 100 s, the measured call 1 s."""
    now = [0.0]
    original = Model.__call__

    def call(self: Model, *, pixel_values: Tensor) -> Any:
        now[0] += 1.0 + (100.0 if not Model.calls else 0.0)
        return original(self, pixel_values=pixel_values)

    monkeypatch.setattr(Model, "__call__", call)
    monkeypatch.setattr(run_tool, "time", SimpleNamespace(perf_counter=lambda: now[0]))
    code, out = run_family(tmp_path, dict(DFINE), count=1)
    assert code == 0
    document = json.loads(out.read_text(encoding="utf-8"))
    assert [entry["seconds"] for entry in document["frames"]] == [1.0]
    assert document["runner"]["stage_ms_per_frame"] == {"preprocess": 0.0, "model": 1000.0, "postprocess": 0.0}


def test_half_precision_runs_the_model_in_float16(tmp_path: Path, transformers: SimpleNamespace) -> None:
    code, out = run_family(tmp_path, {**DFINE, "half": True})
    assert code == 0
    assert Model.made[0].dtype == "float16"
    assert json.loads(out.read_text(encoding="utf-8"))["runner"]["dtype"] == "float16"


def test_boxes_that_differ_from_those_of_the_tool_stop_the_run(
    tmp_path: Path, transformers: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Boxes of the tool half a pixel away from those of the script: the check on the first call finds it."""

    def shifted(results: list[dict[str, Tensor]]) -> None:
        results[0]["boxes"] = Tensor(results[0]["boxes"].values + 0.5)

    monkeypatch.setattr(Processor, "change", staticmethod(shifted))
    code, out = run_family(tmp_path, dict(DFINE))
    assert code == 2
    assert "differ from those of post_process_object_detection" in error_line(capsys)
    assert not out.exists()


@pytest.mark.parametrize(
    ("change", "words"),
    [
        ({"half": True}, "half precision needs a GPU"),
        ({"model": "huge"}, "the size 'huge' of dfine is not one of nano, small, medium, large, xlarge"),
        ({"half": "yes"}, "must be true or false"),
    ],
)
def test_settings_that_the_script_cannot_run_are_refused_before_the_model_is_built(
    tmp_path: Path,
    transformers: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
    change: dict[str, Any],
    words: str,
) -> None:
    code, _ = run_family(tmp_path, {**DFINE, **change}, device="cpu")
    assert code == 2
    assert words in error_line(capsys)
    assert Model.made == []


def test_another_version_of_transformers_is_refused(
    tmp_path: Path, transformers: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(run_tool, "installed_version", {"transformers": "5.24.0", "torch": "2.15.0"}.__getitem__)
    code, _ = run_family(tmp_path, dict(DFINE))
    assert code == 2
    assert "written for transformers 5.23.0, and transformers 5.24.0 is installed" in error_line(capsys)
    assert Model.made == []


@pytest.mark.parametrize(
    ("config", "words"),
    [
        ({"model_type": "rt_detr_v2"}, "of the model type 'rt_detr_v2', not of dfine"),
        ({"use_focal_loss": False}, "not trained with the focal loss"),
        ({"id2label": {"0": "person", "1": "car"}}, "have no class sports ball"),
    ],
)
def test_checkpoint_whose_configuration_is_not_of_the_family_is_refused(
    tmp_path: Path,
    transformers: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config: dict[str, Any],
    words: str,
) -> None:
    monkeypatch.setattr(Model, "config", SimpleNamespace(**{**vars(Model.config), **config}))
    code, _ = run_family(tmp_path, dict(DFINE))
    assert code == 2
    assert words in error_line(capsys)
    assert Model.calls == []


def test_unknown_family_is_refused_by_the_command_line_and_by_the_job(
    tmp_path: Path, transformers: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stop:
        run_family(tmp_path, {**DFINE, "name": "yolo"})
    assert stop.value.code == 2
    assert "invalid choice: 'yolo'" in error_line(capsys)
    # A job of one family given to another family of the same tool.
    frames = small_frames(tmp_path / "data", 1)
    job = tmp_path / "job2.json"
    document = {"format": runs.JOB_FORMAT, "name": "a", "tool": DFINE, "frames": frames}
    job.write_text(json.dumps(document), encoding="utf-8")
    arguments = ["rtdetrv2", "--job", str(job), "--root", str(tmp_path / "data"), "--out", str(tmp_path / "o.json")]
    assert run_tool.main([*arguments, "--device", "cpu"]) == 2
    assert "is a job for the tool 'dfine', not for 'rtdetrv2'" in error_line(capsys)


def test_clip_job_of_dfine_gives_a_clip_run_that_the_package_reads(
    tmp_path: Path, transformers: SimpleNamespace
) -> None:
    code, out = run_family(tmp_path, dict(DFINE), count=4, clip=True)
    assert code == 0
    run = runs.load_clip_run(out)
    assert run.tool == TransformersDetectorSettings.model_validate(DFINE)
    people, balls = detector_answers(run)
    assert [len(frame.people) for frame in people] == [1, 1, 1, 1]
    box = people[0].people[0].box
    assert (box.x1, box.y1, box.x2, box.y2) == (90.0, 30.0, 110.0, 70.0)
    assert balls == {}
    assert run.runner["checkpoint"] == "ustc-community/dfine-medium-coco"


# --- The package -----------------------------------------------------------------------


def test_script_and_package_agree_on_the_families_and_their_sizes() -> None:
    sizes = {family: tuple(checkpoints) for family, checkpoints in run_tool.TRANSFORMERS_CHECKPOINTS.items()}
    assert sizes == runs.TRANSFORMERS_MODELS
    assert set(run_tool.TRANSFORMERS_MODEL_TYPES) == set(runs.TRANSFORMERS_MODELS)
    assert run_tool.DETECTOR_TOOLS == runs.DETECTOR_FAMILIES == ("rfdetr", "dfine", "rtdetrv2")
    assert set(DEFAULT_DETECTORS) == set(runs.DETECTOR_FAMILIES)
    for family, checkpoints in run_tool.TRANSFORMERS_CHECKPOINTS.items():
        for repository, revision in checkpoints.values():
            assert len(revision) == 40 and set(revision) <= set("0123456789abcdef"), (family, repository)
            # COCO-only weights: no checkpoint trained or pretrained on Objects365.
            assert "obj" not in repository and "o365" not in repository, repository


@pytest.mark.parametrize(
    "change",
    [
        {"name": "yolo"},
        {"model": "huge"},
        {"name": "rtdetrv2", "model": "medium"},
        {"input_size": 650},
        {"input_size": 256},
        {"batch": 0},
        {"batch": 1.0},
        {"half": "false"},
        {"min_score": 1.5},
        {"antialias": False},
        {"name": ["dfine"]},
    ],
)
def test_settings_of_a_family_that_breaks_its_rules_are_refused(change: dict[str, Any], tmp_path: Path) -> None:
    """Refused as a settings record and as a settings file (JSON is YAML), with the error of the settings."""
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate({"detector": {**DFINE, **change}})
    path = tmp_path / "settings.yaml"
    path.write_text(json.dumps({"detector": {**DFINE, **change}}), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


def test_settings_file_that_names_a_family_starts_from_its_defaults(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text("engines: {people: detector}\ndetector: {name: dfine, model: small}\n", encoding="utf-8")
    config = load_config(path)
    assert config.detector == TransformersDetectorSettings(
        name="dfine", model="small", min_score=0.5, input_size=640, batch=1, half=False
    )
    assert _engines(config)["people"] == "detector:dfine:small"
    # Without a name the section is laid over the default family, RF-DETR.
    path.write_text("detector: {model: RFDETRLarge}\n", encoding="utf-8")
    assert load_config(path).detector.name == "rfdetr"
    path.write_text("engines: {people: detector}\ndetector: {name: rtdetrv2}\n", encoding="utf-8")
    assert _engines(load_config(path))["people"] == "detector:rtdetrv2:r18vd"
    # A family with the keys of another family is refused.
    path.write_text("detector: {name: dfine, antialias: false}\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


def test_default_detector_stays_rfdetr_medium() -> None:
    assert PipelineConfig().detector.model_dump(mode="json") == {
        "name": "rfdetr",
        "model": "RFDETRMedium",
        "min_score": 0.5,
        "antialias": False,
    }
    assert load_config(REPO / "configs" / "default.yaml") == PipelineConfig()


FAMILY_EXPERIMENTS = (
    "people_probe_dfine_small.yaml",
    "people_probe_dfine_medium.yaml",
    "people_probe_dfine_large.yaml",
    "people_probe_rtdetrv2_r18vd.yaml",
    "people_probe_rtdetrv2_r50vd.yaml",
)


@pytest.mark.parametrize("name", FAMILY_EXPERIMENTS)
def test_experiment_files_of_the_families_take_the_frames_of_the_rfdetr_files(name: str) -> None:
    """Rule 8: the same frames (group, frames per clip, seed) and the same gate as the RF-DETR files."""
    rfdetr = cli.load_tool_experiment(EXPERIMENTS / "people_probe_rfdetr_medium.yaml")
    experiment = cli.load_tool_experiment(EXPERIMENTS / name)
    assert experiment.name == Path(name).stem
    assert (experiment.group, experiment.frames_per_clip, experiment.seed) == (
        rfdetr.group,
        rfdetr.frames_per_clip,
        rfdetr.seed,
    )
    assert experiment.probe == "people"
    tool = experiment.tool
    assert isinstance(tool, TransformersDetectorSettings)
    assert (tool.min_score, tool.input_size, tool.batch, tool.half) == (rfdetr.tool.min_score, 640, 1, False)
    assert set(tool.model_dump()) == run_tool.TOOL_KEYS[tool.name]
    assert tool.model in run_tool.TRANSFORMERS_CHECKPOINTS[tool.name]


def test_clip_job_takes_the_family_of_the_settings_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "data"
    small_frames(root, 2)
    clip = root / "SoccerNetGS-valid" / "CLIP-A"
    settings = tmp_path / "settings.yaml"
    settings.write_text("engines: {people: detector}\ndetector: {name: dfine}\n", encoding="utf-8")
    out = tmp_path / "job.json"
    arguments = ["clip-job", "--clip", str(clip), "--root", str(root), "--out", str(out), "--config", str(settings)]
    assert cli.main([*arguments, "--tool", "dfine"]) == 0
    job = json.loads(out.read_text(encoding="utf-8"))
    assert job["tool"] == DEFAULT_DETECTORS["dfine"].model_dump(mode="json")
    capsys.readouterr()
    for tool in ("rfdetr", "rtdetrv2"):
        assert cli.main([*arguments, "--tool", tool]) == 2
        assert f"the settings name the detector dfine, not {tool}" in error_line(capsys)


def test_tool_score_measures_a_dfine_run_like_a_rfdetr_run(
    tmp_path: Path, transformers: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stand-in model finds the labelled people of each probe frame: recall and precision are 1."""
    from pitchvlm.labels import gsr_frame_people
    from pitchvlm.synth import SyntheticScene

    scene, root = SyntheticScene(n_frames=50, seed=2), tmp_path / "data"
    clip_dir = root / "SoccerNetGS-valid" / "SNGS-999"
    (clip_dir / "img1").mkdir(parents=True)
    for frame in range(50):
        image = scene.image(frame).copy()
        image[0, 0] = (0, 0, frame)  # the stand-in model knows the frame by the red value of its first pixel
        assert cv2.imwrite(str(clip_dir / "img1" / f"{frame + 1:06d}.png"), image)
    (clip_dir / "Labels-GameState.json").write_text(json.dumps(scene.gsr_labels()), encoding="utf-8")
    labels = cli.read_labels(clip_dir / "Labels-GameState.json")
    info = scene.info

    def answers(red: int) -> tuple[np.ndarray, np.ndarray]:
        people = [person.box for person in gsr_frame_people(labels, red).people]
        logits = np.full((len(people) + 1, 3), logit(0.01))
        boxes = np.full((len(people) + 1, 4), 0.5)
        for query, box in enumerate(people):
            logits[query, 0] = logit(0.875)
            boxes[query] = [
                (box.x1 + box.x2) / 2 / info.width,
                (box.y1 + box.y2) / 2 / info.height,
                (box.x2 - box.x1) / info.width,
                (box.y2 - box.y1) / info.height,
            ]
        return logits, boxes

    monkeypatch.setattr(Model, "answers", staticmethod(answers))
    split_list = tmp_path / "splits.json"
    split_list.write_text(json.dumps({"seed": 7, "valid": {"probe": ["SNGS-999"]}}), encoding="utf-8")
    experiment = EXPERIMENTS / "people_probe_dfine_medium.yaml"
    shared = ["--experiment", str(experiment), "--root", str(root), "--manifest", str(split_list)]
    assert cli.main(["tool-job", *shared, "--out", str(tmp_path / "job.json")]) == 0
    run_file, metrics = tmp_path / "run.json", tmp_path / "metrics.json"
    arguments = ["dfine", "--job", str(tmp_path / "job.json"), "--root", str(root), "--out", str(run_file)]
    assert run_tool.main([*arguments, "--device", "cuda:0"]) == 0
    assert cli.main(["tool-score", *shared, "--run", str(run_file), "--out", str(metrics), "--allow-dirty"]) == 0

    measured = json.loads(metrics.read_text(encoding="utf-8"))
    assert measured["probe"] == "people"
    assert (measured["recall"], measured["precision"]) == (1.0, 1.0)
    assert measured["counts"]["tp"] == measured["counts"]["true_boxes"] > 0
    assert measured["settings"]["tool"]["name"] == "dfine"
    assert measured["provenance"]["runner"]["checkpoint"] == "ustc-community/dfine-medium-coco"


# --- Second read (round 6): an independent reference, the self-check, the class places -------------


def independent_post_process(
    logits: np.ndarray, boxes: np.ndarray, height: int, width: int, threshold: float, labels: dict[int, str]
) -> list[tuple[str, float, list[float]]]:
    """``post_process_object_detection`` of transformers 5.23.0 for one frame, written apart from the script.

    A reader who had not seen the script wrote it from the public source of the tag v5.23.0
    (``models/rt_detr/image_processing_rt_detr.py``, ``image_transforms.py``) and the
    configuration files of two checkpoints, on 2026-11-23: corners from the centre format,
    scaled by (width, height, width, height) of a target size given as (height, width), no
    clipping; the logistic function of the logits; ``topk`` with k = queries over the flattened
    pairs of query and class; class = place modulo classes, query = place divided by classes;
    scores above the threshold (strictly), in the order of ``topk``.
    """
    logits, boxes = np.asarray(logits, dtype=np.float64), np.asarray(boxes, dtype=np.float64)
    queries, classes = logits.shape
    centre_x, centre_y, box_width, box_height = boxes.T
    corners = np.stack(
        [centre_x - 0.5 * box_width, centre_y - 0.5 * box_height, centre_x + 0.5 * box_width, centre_y + 0.5 * box_height],
        axis=-1,
    ) * np.array([width, height, width, height], dtype=np.float64)
    flat = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
    order = np.argsort(-flat, kind="stable")[:queries]
    return [
        (labels[int(place % classes)], float(flat[place]), [float(value) for value in corners[place // classes]])
        for place in order
        if flat[place] > threshold
    ]


def test_conversion_agrees_with_an_independent_reference_on_random_outputs() -> None:
    """1,000 random outputs (seed 300): frames that are not square, edge boxes, 1 to 300 queries, 80 classes.

    Planted bugs that this test finds (each tried once): width and height swapped in the scale,
    boxes read as corners, no cut to the best pairs, query and class places swapped, a limit
    moved by 0.01 up or down.

    The script keeps a score equal to its limit and the tool does not; no random score falls on a
    limit, so the two sides must give the same detections.
    """
    labels = {index: f"class {index}" for index in range(80)} | {0: "person", 32: "sports ball"}
    rng = np.random.default_rng(300)
    sizes = [(1920, 1080), (1280, 720), (200, 100), (100, 200), (640, 640), (3840, 2160), (1, 1), (7, 3)]
    n_detections = 0
    for case in range(1000):
        queries = int(rng.choice([1, 3, 10, 300]))
        width, height = sizes[case % len(sizes)] if case % 3 else tuple(int(v) for v in rng.integers(1, 4000, 2))
        # Most pairs score low; a person, a ball and one other class get scores spread over 0 to 1, so
        # that both the limit and the cut to the best pairs (as many as queries) decide.
        logits = rng.normal(-8.0, 2.0, (queries, 80))
        for place in (0, 32, int(rng.integers(1, 80))):
            logits[:, place] = np.log(1.0 / rng.uniform(0.001, 0.999, queries) - 1.0) * -1.0
        logits = logits.astype(np.float32)
        boxes = np.column_stack(
            [rng.uniform(-0.1, 1.1, (queries, 2)), rng.uniform(0.0, 1.5, (queries, 2))]
        ).astype(np.float32)
        if case % 5 == 0:  # centres on the edges and corners of the frame: the boxes reach outside it
            boxes[:, :2] = rng.choice([0.0, 1.0], (queries, 2))
        limit = float(rng.choice([0.0, 0.05, 0.3, 0.5, 0.9]))
        payload = run_tool.detections_of_queries(
            logits, boxes, width=width, height=height, labels=labels, min_score=limit
        )
        mine = [(d["label"], d["score"], d["box_xyxy"]) for d in payload["detections"]]
        assert all(score != limit for _, score, _ in mine)  # else the two sides may differ by design
        theirs = [
            item
            for item in independent_post_process(logits, boxes, height, width, limit, labels)
            if item[0] in run_tool.KEPT_LABELS
        ]
        assert [item[0] for item in mine] == [item[0] for item in theirs], case
        for (_, my_score, my_box), (_, their_score, their_box) in zip(mine, theirs, strict=True):
            assert my_score == pytest.approx(their_score, abs=1e-12)
            assert my_box == pytest.approx(their_box, abs=1e-9)
        n_detections += len(theirs)
    assert n_detections > 10_000


def test_class_places_that_are_not_0_to_classes_are_refused() -> None:
    """Weights whose class places are 1 to 3 (the COCO ids start at 1) would give each box the name of another class."""
    logits = np.full((2, 3), logit(0.01))
    logits[0, 1] = logit(0.9)
    boxes = np.tile([0.5, 0.5, 0.2, 0.2], (2, 1))
    shifted = {1: "person", 2: "car", 3: "sports ball"}
    with pytest.raises(run_tool.RunnerError, match="class places"):
        run_tool.detections_of_queries(logits, boxes, width=WIDTH, height=HEIGHT, labels=shifted, min_score=0.5)


def test_self_check_also_compares_a_first_frame_without_a_detection_above_the_limit(
    tmp_path: Path, transformers: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A first frame with every score under ``min_score`` gave an empty check: a changed tool then passed."""

    def low(red: int) -> tuple[np.ndarray, np.ndarray]:
        logits = np.full((3, 3), logit(0.01))
        logits[0, 0] = logit(0.3)
        return logits, np.array([[0.5, 0.5, 0.1, 0.4], [0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]])

    def shifted(results: list[dict[str, Tensor]]) -> None:
        results[0]["boxes"] = Tensor(results[0]["boxes"].values + 0.5)

    monkeypatch.setattr(Model, "answers", staticmethod(low))
    monkeypatch.setattr(Processor, "change", staticmethod(shifted))
    code, out = run_family(tmp_path, dict(DFINE))
    assert code == 2
    assert "differ from those of post_process_object_detection" in error_line(capsys)
    assert not out.exists()


def test_self_check_counts_the_detections_that_it_compared(
    tmp_path: Path, transformers: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run_family(tmp_path, dict(DFINE))
    assert code == 0, capsys.readouterr().err
    runner = json.loads(out.read_text(encoding="utf-8"))["runner"]
    # Of the 9 pairs of the frame the tool keeps the best 3 (as many as queries): the person, then two
    # pairs of score 0.01. The cut decided, so only the pairs above the lowest kept score are compared.
    assert runner["self_check_detections"] == 1


def test_self_check_leaves_out_the_pairs_that_tie_at_the_cut_to_the_best_pairs(
    tmp_path: Path, transformers: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``torch.topk`` gives no order for equal scores: the tool may keep other tied pairs than the script.

    The stand-in frame has 8 pairs of score 0.01 for the last 2 of 3 places. The tool here breaks
    the ties from the last place, the script from the first: a correct run must not stop.
    """

    frame = Model.answers(0)

    def other_ties(results: list[dict[str, Tensor]]) -> None:
        logits, boxes = frame
        scores = 1.0 / (1.0 + np.exp(-logits.ravel()))
        places = np.arange(scores.size)[::-1]
        order = places[np.argsort(-scores[::-1], kind="stable")][: len(logits)]
        corners = np.column_stack(
            [boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2, boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2]
        ) * np.array([WIDTH, HEIGHT, WIDTH, HEIGHT])
        classes = logits.shape[1]
        results[0] = {
            "scores": Tensor(scores[order]),
            "labels": Tensor(order % classes),
            "boxes": Tensor(corners[order // classes]),
        }

    monkeypatch.setattr(Model, "answers", staticmethod(lambda red: frame))
    monkeypatch.setattr(Processor, "change", staticmethod(other_ties))
    code, out = run_family(tmp_path, dict(DFINE))
    assert code == 0, capsys.readouterr().err
    assert json.loads(out.read_text(encoding="utf-8"))["runner"]["self_check_detections"] == 1
