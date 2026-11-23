"""Run ``tools/run_tool.py`` as a process with a stand-in of its tool: for the build machine, which has no tool.

The script, its job reading, its per-frame loop and its live feed are those of ``run_tool.py``;
only the tool is replaced. The stand-in answers each frame with the payload that an answers
file gives for the SHA-256 of its pixels (the RGB array as read), after an added delay per frame
that simulates the time of a real tool. The images are read with OpenCV (the build machine has
no Pillow). The end-to-end test of the live feed (``tests/test_live_feed.py``) and the latency
bench (``python -m pitchvlm.evalx.livebench``) run it as real processes with real pipes.

    python tools/stand_in_tool.py --answers FILE [--delay-ms MS] -- rfdetr --job JOB --root DATA ...

The answers file: ``{"by_image": {<sha256 of the RGB bytes>: <payload of the tool>}}``. A frame
whose pixels are not in the file stops the run (exit code 2), as an error of a real tool would.
Nothing here imports the package or a tool, and nothing reads a key of a model API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

#: The folder of this script: ``run_tool.py`` lies next to it and is imported by its name.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_tool  # noqa: E402  (the tool script of the run machine; it imports neither the package nor a tool here)


def read_rgb(path: Path) -> np.ndarray:
    """Read an image file as an RGB ``uint8`` array, as the reader of the script does."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"the image {path} cannot be read")
    return np.ascontiguousarray(image[:, :, ::-1])


def stand_in(by_image: dict[str, Any], delay_s: float) -> Any:
    """Return the answer function of the stand-in: the payload of the pixels, after ``delay_s`` seconds."""

    def answer(image: np.ndarray) -> dict[str, Any]:
        start = time.perf_counter()
        key = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
        if key not in by_image:
            raise KeyError(f"the answers file has no answer for the pixels {key[:12]}")
        payload = by_image[key]
        rest = delay_s - (time.perf_counter() - start)
        if rest > 0:
            time.sleep(rest)
        return payload

    return answer


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--answers", type=Path, required=True, help="answers file (JSON)")
    parser.add_argument("--delay-ms", type=float, default=0.0, help="added time per frame, milliseconds (simulation)")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="-- and the arguments of run_tool.py")
    args = parser.parse_args(argv)
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    by_image = json.loads(args.answers.read_text(encoding="utf-8"))["by_image"]
    answer = stand_in(by_image, args.delay_ms / 1000.0)
    facts = {"tool_version": "stand-in", "delay_ms": args.delay_ms}
    run_tool.read_rgb = read_rgb
    run_tool.installed_version = lambda package: "stand-in"
    run_tool.rfdetr_engine = lambda settings, device: (facts, answer)
    run_tool.pnlcalib_engine = lambda settings, device, repo, weights_kp, weights_lines: (facts, answer)
    return int(run_tool.main(arguments))


if __name__ == "__main__":
    sys.exit(main())
