"""A fake model backend that answers from the ground truth of a synthetic scene.

It needs no network, no key and no model SDK. It answers the four pipeline steps in the
answer formats of the real model, so the whole pipeline can run in tests and in the
offline demo. Misses and noise can be switched on to make the answers imperfect in a
controlled and repeatable way.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np

from pitchvlm.schemas import Box
from pitchvlm.synth import SyntheticScene
from pitchvlm.vlm.coords import GEMINI_CONVENTION, CoordConvention, box_to_answer, point_to_answer
from pitchvlm.vlm.types import DEFAULT_PROVIDER, BackendError, Usage, VLMRequest, VLMResponse

#: Token counts reported for each step. They are fixed, and near the planning estimates.
STEP_USAGE: dict[str, Usage] = {
    "kit_legend": Usage(input_tokens=3600, output_tokens=200, thought_tokens=400),
    "people": Usage(input_tokens=1500, output_tokens=900, thought_tokens=500),
    "landmarks": Usage(input_tokens=2200, output_tokens=400, thought_tokens=400),
    "number": Usage(input_tokens=2500, output_tokens=100, thought_tokens=300),
}
#: The price at which the fake backend reports the cost of an OpenRouter request, USD per million
#: tokens (input, output). A real OpenRouter answer reports its cost; a Gemini answer reports none.
FAKE_REPORTED_PRICE = (1.0, 4.0)


class FakeBackend:
    """Answer requests with the truth of a ``SyntheticScene``, with optional misses and noise.

    ``miss_rate`` is the chance that a person is left out of a people answer.
    ``box_noise_px`` is the standard deviation of the Gaussian noise added to each edge of a
    box, and ``point_noise_px`` the same for each coordinate of a landmark, in pixels. A box
    that the noise turns inside out is left out. The coordinate conversion keeps noisy
    values inside the range 0-1000.

    The random generator is seeded from ``seed`` and the cache key of the request, so the
    same request always gets the same answer. The backend keeps no state between requests,
    so it can be used from several threads. The response names the model of the request, so
    that cost code can price it, and reports fixed token counts. An answer to an
    OpenRouter request also reports a cost, at ``FAKE_REPORTED_PRICE``, as the real
    API does.

    ``coords`` is the coordinate convention in which boxes and points are answered. It
    must be the convention of the step settings of the requests (``StepSettings.coords``).
    """

    def __init__(
        self,
        scene: SyntheticScene,
        *,
        miss_rate: float = 0.0,
        box_noise_px: float = 0.0,
        point_noise_px: float = 0.0,
        seed: int = 0,
        coords: CoordConvention = GEMINI_CONVENTION,
    ) -> None:
        if not 0.0 <= miss_rate <= 1.0:
            raise ValueError("miss_rate must be between 0 and 1")
        if box_noise_px < 0.0 or point_noise_px < 0.0:
            raise ValueError("noise levels must not be negative")
        if seed < 0:
            raise ValueError("seed must not be negative")
        self._scene = scene
        self._miss_rate = miss_rate
        self._box_noise_px = box_noise_px
        self._point_noise_px = point_noise_px
        self._seed = seed
        self._coords = coords
        self._steps: dict[str, Callable[[VLMRequest], dict[str, Any]]] = {
            "kit_legend": self._kit_legend,
            "people": self._people,
            "landmarks": self._landmarks,
            "number": self._number,
        }

    def generate(self, request: VLMRequest) -> VLMResponse:
        """Answer one request as JSON text. A step that is not known raises ``BackendError``."""
        step = request.meta.step
        build_answer = self._steps.get(step)
        if build_answer is None:
            raise BackendError(f"the fake backend has no answer for step '{step}'")
        usage = STEP_USAGE[step]
        reported_usd = None
        if request.provider != DEFAULT_PROVIDER:
            input_price, output_price = FAKE_REPORTED_PRICE
            output_tokens = usage.output_tokens + usage.thought_tokens
            reported_usd = (usage.input_tokens * input_price + output_tokens * output_price) / 1e6
        return VLMResponse(
            text=json.dumps(build_answer(request)),
            usage=usage,
            model=request.model,
            service_tier=request.service_tier,
            latency_s=0.0,
            created_at=datetime.now(UTC).isoformat(),
            reported_usd=reported_usd,
        )

    def _kit_legend(self, request: VLMRequest) -> dict[str, Any]:
        """Describe the kits of the scene. The answer does not depend on the request."""
        legend = self._scene.kit_legend()
        return {
            "team_a": legend.team_a.model_dump(),
            "team_b": legend.team_b.model_dump(),
            "goalkeepers": [{"kit": keeper.kit, "team": keeper.team_key} for keeper in legend.goalkeepers],
            "referee_kit": legend.referee_kit,
            "uncertain": list(legend.uncertain),
        }

    def _people(self, request: VLMRequest) -> dict[str, Any]:
        """List the true people of the frame, with misses and box noise."""
        truth = self._scene.people(self._frame(request)).people
        rng = self._generator(request)
        missed = (rng.random(len(truth)) < self._miss_rate).tolist()
        shifts = rng.normal(0.0, self._box_noise_px, size=(len(truth), 4)).tolist()
        width, height = self._scene.info.width, self._scene.info.height
        people: list[dict[str, Any]] = []
        for person, is_missed, (left, top, right, bottom) in zip(truth, missed, shifts, strict=True):
            x1, y1 = person.box.x1 + left, person.box.y1 + top
            x2, y2 = person.box.x2 + right, person.box.y2 + bottom
            if is_missed or x2 <= x1 or y2 <= y1:
                continue
            people.append(
                {
                    "box_2d": box_to_answer(Box(x1=x1, y1=y1, x2=x2, y2=y2), width, height, self._coords),
                    "role": person.role,
                    "team": person.team_key,
                    "number": person.number,
                }
            )
        return {"people": people}

    def _landmarks(self, request: VLMRequest) -> dict[str, Any]:
        """List the true visible landmarks of the frame, with point noise, and the view."""
        truth = self._scene.landmarks(self._frame(request))
        shifts = self._generator(request).normal(0.0, self._point_noise_px, size=(len(truth.landmarks), 2)).tolist()
        width, height = self._scene.info.width, self._scene.info.height
        landmarks = [
            {
                "id": seen.landmark_id,
                "point": point_to_answer(seen.x + shift_x, seen.y + shift_y, width, height, self._coords),
            }
            for seen, (shift_x, shift_y) in zip(truth.landmarks, shifts, strict=True)
        ]
        return {"view": truth.view, "landmarks": landmarks}

    def _number(self, request: VLMRequest) -> dict[str, Any]:
        """Read the true shirt number of every crop and vote over the crops.

        The crops are described by ``request.context``: ``frames`` and ``boxes``, one entry
        per crop. The answer number is the most frequent number of the crops (the one seen
        first when two are equally frequent). The confidence is "high" when at least two
        crops agree on it, else "low".
        """
        frames = request.context.get("frames")
        boxes = request.context.get("boxes")
        if frames is None or boxes is None or len(frames) != len(boxes):
            raise BackendError("a 'number' request needs the context lists 'frames' and 'boxes' of the same length")
        per_crop = [self._crop_number(frame, corners) for frame, corners in zip(frames, boxes, strict=True)]
        votes = Counter(number for number in per_crop if number is not None)
        if not votes:
            return {"per_crop": per_crop, "number": None, "confidence": "low"}
        number, count = votes.most_common(1)[0]
        return {"per_crop": per_crop, "number": number, "confidence": "high" if count >= 2 else "low"}

    def _crop_number(self, frame: int, corners: Sequence[float]) -> int | None:
        """Return the true shirt number of the person in the box ``[x1, y1, x2, y2]`` of one frame."""
        x1, y1, x2, y2 = corners
        return self._scene.true_number(self._checked_frame(frame), Box(x1=x1, y1=y1, x2=x2, y2=y2))

    def _frame(self, request: VLMRequest) -> int:
        """Return the frame index of a request that must have one."""
        if request.meta.frame is None:
            raise BackendError(f"a '{request.meta.step}' request needs a frame index")
        return self._checked_frame(request.meta.frame)

    def _checked_frame(self, frame: int) -> int:
        """Return ``frame`` when it is inside the scene, else raise ``BackendError``."""
        if not 0 <= frame < self._scene.info.n_frames:
            raise BackendError(f"frame {frame} is outside the synthetic clip")
        return frame

    def _generator(self, request: VLMRequest) -> np.random.Generator:
        """Return a random generator that depends only on the seed and the request's cache key."""
        return np.random.default_rng([self._seed, int(request.cache_key(), 16)])
