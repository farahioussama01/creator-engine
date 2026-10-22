"""Analyzer for the simulator's procedural faces: skin-colour detection + 4x4 luminance embedding."""

from __future__ import annotations

import cv2
import numpy as np

from . import Face

GRID = 4


class SimFaceAnalyzer:
    name = "sim"

    def __init__(self, min_area_frac: float = 0.002, min_fill: float = 0.55):
        self.min_area_frac = min_area_frac
        self.min_fill = min_fill

    def detect(self, img_bgr: np.ndarray) -> list[Face]:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 45, 70), (28, 215, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        H, W = mask.shape
        faces: list[Face] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area_frac * H * W:
                continue
            x, y, w, h = cv2.boundingRect(c)
            fill = area / (np.pi / 4 * w * h + 1e-6)
            aspect = w / max(h, 1)
            if fill < self.min_fill or not (0.45 < aspect < 1.1):
                continue
            faces.append(Face(x, y, w, h, score=float(min(1.0, fill))))
        faces.sort(key=lambda f: f.w * f.h, reverse=True)
        return faces

    def embed(self, img_bgr: np.ndarray, face: Face) -> np.ndarray:
        x, y, w, h = face.box
        ix0, iy0 = x + int(0.15 * w), y + int(0.15 * h)
        iw, ih = max(GRID, int(0.70 * w)), max(GRID, int(0.70 * h))
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        crop = gray[iy0 : iy0 + ih, ix0 : ix0 + iw]
        if crop.size == 0:
            return np.zeros(GRID * GRID, np.float32)
        cells = cv2.resize(crop, (GRID, GRID), interpolation=cv2.INTER_AREA).flatten()
        cells -= cells.mean()
        n = np.linalg.norm(cells)
        return (cells / n).astype(np.float32) if n > 0 else cells.astype(np.float32)
