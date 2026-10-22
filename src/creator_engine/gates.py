"""Six checks. A still must pass four before it is animated; a clip must pass two to be kept.

Still (cheap, ~$0.008 to re-roll):
  light      not blown out, not too dark
  one face   exactly one main face in frame
  sharp      the face is in focus
  identity   the face matches the character (cosine similarity to the training images)

Clip (expensive, ~$0.05 to re-roll):
  moving     not a frozen clip
  same face  the face stays the character across 8 frames (catches faces morphing mid-shot)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .faces import cosine


@dataclass
class Verdict:
    ok: bool
    reason: str = ""
    identity: float | None = None
    metrics: dict = field(default_factory=dict)


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def sharpness(img: np.ndarray) -> float:
    g = _gray(img)
    g = cv2.resize(g, (max(8, int(g.shape[1] * 160 / g.shape[0])), 160), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def read_frames(path: Path) -> list[np.ndarray]:
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


class Checker:
    def __init__(self, cfg: Config, faces, centroid: np.ndarray, identity_min: float):
        self.g, self.faces, self.centroid, self.identity_min = cfg.gates, faces, centroid, identity_min
        self.name = cfg.character.name

    def keyframe(self, path: Path) -> Verdict:
        img = cv2.imread(str(path))
        if img is None:
            return Verdict(False, "unreadable image")
        g = _gray(img)
        blown, mean = float(np.mean(g >= 250)), float(g.mean())
        m = {"blown": blown, "mean": mean}
        if blown > self.g.max_blown or not self.g.mean_min <= mean <= self.g.mean_max:
            return Verdict(False, f"light: {blown:.0%} blown, brightness {mean:.0f}", metrics=m)
        found = self.faces.detect(img)
        if not found:
            return Verdict(False, "no face found", metrics=m)
        main = found[0]
        if len(found) > 1 and found[1].w * found[1].h > 0.25 * main.w * main.h:
            return Verdict(False, "two faces", metrics=m)
        crop = img[max(0, main.y) : main.y + main.h, max(0, main.x) : main.x + main.w]
        m["sharpness"] = sharpness(crop)
        if m["sharpness"] < self.g.sharpness_min:
            return Verdict(False, f"blurry face (sharpness {m['sharpness']:.0f})", metrics=m)
        sim = cosine(self.faces.embed(img, main), self.centroid)
        if sim < self.identity_min:
            return Verdict(False, f"not {self.name} (similarity {sim:.2f})", sim, m)
        return Verdict(True, "", sim, m)

    def clip(self, path: Path) -> Verdict:
        frames = read_frames(path)
        if len(frames) < 2:
            return Verdict(False, "unreadable clip")
        small = [cv2.resize(_gray(f), (128, int(128 * f.shape[0] / f.shape[1]))).astype(np.float32) for f in frames]
        motion = float(np.median([np.mean(np.abs(b - a)) for a, b in zip(small, small[1:], strict=False)]))
        if motion < self.g.clip_min_motion:
            return Verdict(False, f"frozen clip (motion {motion:.2f})", metrics={"motion": motion})
        sims = []
        for i in np.linspace(0, len(frames) - 1, 8).astype(int):
            found = self.faces.detect(frames[i])
            if found:
                sims.append(cosine(self.faces.embed(frames[i], found[0]), self.centroid))
        share = len(sims) / 8
        m = {"motion": motion, "face_share": share}
        if share < self.g.clip_face_share:
            return Verdict(False, f"face lost in {1 - share:.0%} of frames", metrics=m)
        worst = min(sims)
        # video frames are smaller and softer than the still, so allow a little less similarity
        if worst < self.identity_min - 0.10:
            return Verdict(False, f"face changes mid-clip (worst frame {worst:.2f})", worst, m)
        return Verdict(True, "", worst, m)
