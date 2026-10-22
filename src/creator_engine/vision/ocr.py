"""OCR + the 'is this sign text broken?' heuristic.

Diffusion models rarely produce random letter soup any more; their typical failure is a near-miss:
a doubled, dropped or swapped letter in an otherwise plausible word ("COFFE", "BAKREY"). So a token
fails when it is one or two edits away from a known word without being that word. Brand and place
names a format expects go in ``allowed_text``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np


@dataclass
class TextHit:
    text: str
    confidence: float


@lru_cache(maxsize=1)
def wordlist() -> frozenset[str]:
    data = resources.files("creator_engine").joinpath("data/signage_words.txt").read_text(encoding="utf-8")
    return frozenset(w.strip().lower() for w in data.splitlines() if w.strip() and not w.startswith("#"))


def edit_distance(a: str, b: str, cap: int = 3) -> int:
    """Optimal string alignment distance (insert, delete, substitute, swap adjacent)."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def classify_token(
    token: str,
    expected: set[str] | frozenset[str] = frozenset(),
    allowed: set[str] | frozenset[str] = frozenset(),
) -> str:
    """'ok' | 'misspelled' | 'unknown' for one OCR token.

    ``expected`` are words the prompt asked for (quoted sign text); a near-miss of one of those is
    always a failure. Against the generic wordlist we only flag one-edit misses of 5+ letter words,
    because short real words ("nice" vs "ice") would otherwise be false positives.
    """
    t = token.lower()
    if len(t) < 3 or not t.isalpha():
        return "ok"
    exp = {w.lower() for w in expected}
    known = wordlist() | exp | {w.lower() for w in allowed}
    if t in known:
        return "ok"
    if exp and min(edit_distance(t, w) for w in exp) <= (1 if len(t) <= 4 else 2):
        return "misspelled"
    if len(t) >= 5:
        near = min((edit_distance(t, w) for w in wordlist() if abs(len(w) - len(t)) <= 1), default=9)
        if near == 1:
            return "misspelled"
    return "unknown"


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+", text)


class OCREngine:
    """Thin wrapper over RapidOCR (ONNX, bundled models, Apache-2.0)."""

    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR  # optional dependency

        self._engine = RapidOCR()

    def read(self, img_bgr: np.ndarray) -> list[TextHit]:
        result, _ = self._engine(img_bgr)
        return [TextHit(text=r[1], confidence=float(r[2])) for r in (result or [])]


def try_ocr() -> OCREngine | None:
    try:
        return OCREngine()
    except Exception:  # noqa: BLE001 - OCR is optional; the gate reports 'skipped'
        return None
