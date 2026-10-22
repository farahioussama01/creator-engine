"""Claude as the last gate: the things pixel statistics cannot see.

Hands, head-to-body proportion, light that disagrees with the scene, misspelled text the OCR
missed, and the overall read: does this pass as a real phone photo (good), is it obviously AI
gloss (hype), or is it broken (brain rot)? Runs only on survivors of the cheap gates. At Haiku 5.5
list prices ($0.10 / $0.50 per million tokens) one keyframe review is roughly $0.0002 and a 4-frame
clip review roughly $0.0004, so judging every keeper costs less than a second of GPU time.
"""

from __future__ import annotations

import base64
import json
import threading
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .config import JudgeSpec

CRITERIA = ["light", "skin", "hands", "anatomy", "text", "motion"]

RUBRIC = """You are the final quality gate before {what} of a fictional, clearly-labelled AI character is posted to
short-form social media. The prompt that generated it was:

<prompt>{prompt}</prompt>

Score each criterion from 1 (broken) to 5 (indistinguishable from a real phone {medium}). Use null when
the criterion does not apply (no hands visible, no text in frame, motion for a still image).

- light: one coherent light source and colour; shadows and highlights agree with the scene
- skin: real texture (pores, fine hair, uneven tone) rather than airbrushed plastic
- hands: correct finger count and shape, natural grip, no fusion with objects
- anatomy: head size relative to shoulders and body, neck, limbs, posture
- text: any signage or screen text is legible and correctly spelled
- motion: (clips only) natural movement, the face and body stay the same person in every frame

Then give an overall verdict:
- "good": a viewer scrolling past would take it as a real photo/clip
- "hype": technically clean but reads as obvious AI gloss
- "brain_rot": visible artifacts a viewer would screenshot and mock

Be strict: when in doubt between two scores, choose the lower one."""

SCHEMA = {
    "type": "object",
    "properties": {
        **{c: {"anyOf": [{"type": "integer"}, {"type": "null"}]} for c in CRITERIA},
        "overall": {"type": "string", "enum": ["good", "hype", "brain_rot"]},
        "reasons": {"type": "array", "items": {"type": "string"}},
    },
    "required": [*CRITERIA, "overall", "reasons"],
    "additionalProperties": False,
}


@dataclass
class Verdict:
    passed: bool
    overall: str
    scores: dict[str, int | None]
    reasons: list[str]
    usd: float
    tokens_in: int
    tokens_out: int

    @property
    def min_score(self) -> float | None:
        vals = [v for v in self.scores.values() if v is not None]
        return float(min(vals)) if vals else None

    @property
    def summary(self) -> str:
        if self.passed:
            return ""
        weak = [k for k, v in self.scores.items() if v is not None and v < 3]
        return (
            f"judge: {self.overall}"
            + (f" ({', '.join(weak)})" if weak else "")
            + (f" - {self.reasons[0]}" if self.reasons else "")
        )

    def as_metrics(self) -> dict[str, Any]:
        return {"overall": self.overall, "scores": self.scores, "reasons": self.reasons, "usd": self.usd}


def _encode(img: np.ndarray, max_side: int) -> str:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("could not encode image")
    return base64.b64encode(buf.tobytes()).decode()


@dataclass
class ClaudeJudge:
    spec: JudgeSpec
    client: Any = None
    min_score: int = 3
    spent: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if self.client is None:
            import anthropic  # optional dependency

            self.client = anthropic.Anthropic()

    def judge_images(self, images: list[np.ndarray], prompt: str, video: bool = False) -> Verdict:
        content: list[dict[str, Any]] = []
        for i, img in enumerate(images, 1):
            if len(images) > 1:
                content.append({"type": "text", "text": f"Frame {i}:"})
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": _encode(img, self.spec.max_side),
                    },
                }
            )
        content.append(
            {
                "type": "text",
                "text": RUBRIC.format(
                    what="this clip (frames sampled in order)" if video else "this image",
                    prompt=prompt,
                    medium="video" if video else "photo",
                ),
            }
        )
        resp = self.client.messages.create(
            model=self.spec.model,
            max_tokens=600,
            messages=[{"role": "user", "content": content}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        )
        text = next(b.text for b in resp.content if getattr(b, "type", "") == "text")
        data = json.loads(text)
        scores = {c: data.get(c) for c in CRITERIA}
        if not video:
            scores["motion"] = None
        tin, tout = resp.usage.input_tokens, resp.usage.output_tokens
        usd = tin * self.spec.usd_per_mtok_in / 1e6 + tout * self.spec.usd_per_mtok_out / 1e6
        with self._lock:
            self.spent.append(usd)
        vals = [v for v in scores.values() if v is not None]
        passed = data["overall"] == "good" and (not vals or min(vals) >= self.min_score)
        return Verdict(passed, data["overall"], scores, list(data.get("reasons", [])), usd, tin, tout)
