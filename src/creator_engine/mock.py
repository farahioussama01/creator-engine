"""Offline stand-ins for the image and video models, so the pipeline runs with no API key and no GPU.

Mock renders are cartoons, not photos: a face oval whose 4x4 brightness pattern encodes "who" it is,
plus injected defects (blur, bad exposure, a drifted face, a frozen clip, a face that morphs). The
checks run on these pixels exactly as they run on real renders. Costs are what fal would have charged.
Mock batches are labelled MOCK everywhere and never produce an application answer.
"""

from __future__ import annotations

import hashlib
import math
import random

import cv2
import numpy as np

from .faces import Face

GRID = 4
DEFECT_RATES = {"blur": 0.06, "overexposed": 0.04, "drift": 0.10, "frozen": 0.06, "morph": 0.12}


def identity_vector(key: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    v = np.random.default_rng(seed).normal(size=GRID * GRID)
    v -= v.mean()
    return v / np.linalg.norm(v)


def _paint_identity(
    img: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int], vec: np.ndarray, amp: float = 26.0
) -> None:
    x, y, w, h = box
    x0, y0, iw, ih = x + int(0.15 * w), y + int(0.15 * h), int(0.7 * w), int(0.7 * h)
    layer = cv2.resize(
        (vec * GRID * amp).reshape(GRID, GRID).astype(np.float32), (iw, ih), interpolation=cv2.INTER_NEAREST
    )
    region = img[y0 : y0 + ih, x0 : x0 + iw].astype(np.float32)
    m = mask[y0 : y0 + ih, x0 : x0 + iw, None].astype(np.float32) / 255
    region += layer[: region.shape[0], : region.shape[1], None] * m
    img[y0 : y0 + ih, x0 : x0 + iw] = np.clip(region, 0, 255).astype(np.uint8)


def render_keyframe(width: int, height: int, seed: int, who: str, defects: set[str]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hue = rng.uniform(95, 165)  # cool backgrounds: only the face reads as skin
    top = cv2.cvtColor(np.uint8([[[int(hue), 70, 190]]]), cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)
    bot = cv2.cvtColor(np.uint8([[[int(hue), 90, 95]]]), cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)
    t = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    img = (top * (1 - t) + bot * t).repeat(width, axis=1).astype(np.uint8)
    fh = int(rng.uniform(0.11, 0.16) * height)
    fw = int(0.72 * fh)
    cx, cy = int(width / 2 + rng.uniform(-0.08, 0.08) * width), int(height * rng.uniform(0.25, 0.32))
    cv2.fillPoly(
        img,
        [
            np.array(
                [
                    [cx - 2 * fw, height],
                    [cx - fw, cy + int(0.6 * fh)],
                    [cx + fw, cy + int(0.6 * fh)],
                    [cx + 2 * fw, height],
                ],
                np.int32,
            )
        ],
        (90, 50, 70),
    )
    cv2.ellipse(img, (cx, cy - fh // 16), (int(0.62 * fw), int(0.6 * fh)), 0, 0, 360, (45, 30, 35), -1)
    mask = np.zeros((height, width), np.uint8)
    cv2.ellipse(mask, (cx, cy), (fw // 2, fh // 2), 0, 0, 360, 255, -1)
    img[mask > 0] = cv2.cvtColor(np.uint8([[[12, 105, 205]]]), cv2.COLOR_HSV2BGR)[0, 0]
    base = identity_vector(who)
    vec = base + rng.normal(0, 0.12, base.shape)
    if "drift" in defects:
        other = rng.normal(size=base.shape)
        vec = 0.35 * base + 0.65 * other / np.linalg.norm(other)
    vec = (vec - vec.mean()) / np.linalg.norm(vec - vec.mean())
    _paint_identity(img, mask, (cx - fw // 2, cy - fh // 2, fw, fh), vec)
    out = img.astype(np.float32) + rng.normal(0, 3.0, img.shape).astype(np.float32)
    if "blur" in defects:
        out = cv2.GaussianBlur(out, (0, 0), 3.5)
    if "overexposed" in defects:
        out = out * 1.9 + 45
    return np.clip(out, 0, 255).astype(np.uint8)


def render_clip(
    start: np.ndarray, width: int, height: int, frames: int, seed: int, defects: set[str]
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    h, w = start.shape[:2]
    s = max(width / w, height / h)
    big = cv2.resize(start, (math.ceil(w * s), math.ceil(h * s)), interpolation=cv2.INTER_AREA)
    y0, x0 = (big.shape[0] - height) // 2, (big.shape[1] - width) // 2
    base = big[y0 : y0 + height, x0 : x0 + width].astype(np.float32)
    morph = None
    if "morph" in defects:
        faces = MockFaces().detect(base.astype(np.uint8))
        if faces:
            f = faces[0]
            mask = np.zeros(base.shape[:2], np.uint8)
            cv2.ellipse(mask, (f.x + f.w // 2, f.y + f.h // 2), (f.w // 2, f.h // 2), 0, 0, 360, 255, -1)
            morph = np.zeros_like(base)
            other = rng.normal(size=GRID * GRID)
            tmp = np.zeros_like(base, np.uint8)
            _paint_identity(tmp, mask, (f.x, f.y, f.w, f.h), (other - other.mean()) / np.linalg.norm(other), 42.0)
            morph = tmp.astype(np.float32)
    out, phase = [], rng.uniform(0, math.tau)
    for i in range(frames):
        t = i / max(1, frames - 1)
        frame = base + (morph * min(1.0, max(0.0, (t - 0.4) / 0.35)) if morph is not None else 0)
        if "frozen" not in defects:
            M = cv2.getRotationMatrix2D((width / 2, height * 0.35), 0, 1 + 0.22 * t)
            M[0, 2] += 0.08 * width * math.sin(phase + 2.6 * t)
            frame = cv2.warpAffine(frame, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        out.append(np.clip(frame, 0, 255).astype(np.uint8))
    return out


def write_mp4(frames: list[np.ndarray], path: str, fps: int = 16) -> None:
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames:
        vw.write(fr)
    vw.release()


def draw_defects(rng: random.Random, names: tuple[str, ...]) -> set[str]:
    return {d for d in names if rng.random() < DEFECT_RATES[d]}


class MockFaces:
    """Finds the skin-coloured oval and reads its 4x4 pattern as the fingerprint."""

    def detect(self, img: np.ndarray) -> list[Face]:
        mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (0, 45, 70), (28, 215, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        faces = []
        for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            x, y, w, h = cv2.boundingRect(c)
            if cv2.contourArea(c) > 0.002 * mask.size and 0.45 < w / max(h, 1) < 1.1:
                faces.append(Face(x, y, w, h, score=1.0))
        return sorted(faces, key=lambda f: f.w * f.h, reverse=True)

    def embed(self, img: np.ndarray, face: Face) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        crop = gray[
            face.y + int(0.15 * face.h) : face.y + int(0.85 * face.h),
            face.x + int(0.15 * face.w) : face.x + int(0.85 * face.w),
        ]
        v = cv2.resize(crop, (GRID, GRID), interpolation=cv2.INTER_AREA).flatten()
        v -= v.mean()
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)
