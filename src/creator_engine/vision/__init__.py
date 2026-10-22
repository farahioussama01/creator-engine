"""Face detection / identity embedding / OCR behind small interfaces.

``sim``  -> pixel-level analyzer for the simulator's procedural faces (no model files needed)
``real`` -> YuNet detector + AuraFace embedder (Apache-2.0 weights, usable commercially; the
            popular InsightFace model packs are licensed for non-commercial research only)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Face:
    x: int
    y: int
    w: int
    h: int
    score: float
    landmarks: np.ndarray | None = None  # 5x2: eyes, nose, mouth corners (image-left first)

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


class FaceAnalyzer(Protocol):
    name: str

    def detect(self, img_bgr: np.ndarray) -> list[Face]: ...

    def embed(self, img_bgr: np.ndarray, face: Face) -> np.ndarray: ...


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def make_analyzer(kind: str, models_dir: str | None = None) -> FaceAnalyzer:
    if kind == "sim":
        from .sim import SimFaceAnalyzer

        return SimFaceAnalyzer()
    if kind == "real":
        from .real import RealFaceAnalyzer

        return RealFaceAnalyzer(models_dir or "models")
    raise ValueError(f"unknown analyzer {kind!r}")
