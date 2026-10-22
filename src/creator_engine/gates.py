"""Quality gates.

Cheap, deterministic checks run first and short-circuit, so expensive checks (face embedding, OCR,
the Claude vision judge) only run on candidates that survived. The keyframe gates are the main cost
lever: rejecting a keyframe costs a few GPU-seconds, rejecting a finished clip costs a minute or more.

Each gate maps to something a human reviewer looks at:

  exposure          light: clipped highlights / crushed shadows / murky mean
  face              exactly one dominant face in frame
  sharpness         focus on the face (Laplacian variance of the face crop)
  head_size         face height vs frame height for the format's shot type (the AI "big head")
  skin_texture      high-frequency energy on skin; waxy plastic skin has none
  identity          cosine similarity of the face embedding to the character centroid
  sign_text         OCR; near-miss misspellings of the words the prompt asked for
  judge             Claude vision rubric for what pixels stats cannot see: hands, anatomy, light logic

  motion            mean frame difference: frozen clips and chaotic clips both fail
  flicker           frame-to-frame global luminance jumps
  temporal_identity identity checked on frames across the clip: catches faces morphing mid-shot
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import CharacterSpec, EngineConfig, FormatSpec
from .vision import Face, FaceAnalyzer, cosine
from .vision.ocr import OCREngine, classify_token, tokens

KEYFRAME = "keyframe"
VIDEO = "video"


@dataclass
class GateResult:
    gate: str
    passed: bool | None  # None = skipped (dependency unavailable)
    score: float | None = None
    reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class Artifact:
    piece_id: str
    stage: str
    path: Path
    character: CharacterSpec
    fmt: FormatSpec
    prompt: str = ""
    cache: dict[str, Any] = field(default_factory=dict)

    @property
    def image(self) -> np.ndarray:
        if "image" not in self.cache:
            img = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"cannot read image {self.path}")
            self.cache["image"] = img
        return self.cache["image"]

    @property
    def frames(self) -> list[np.ndarray]:
        if "frames" not in self.cache:
            cap = cv2.VideoCapture(str(self.path))
            frames = []
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(fr)
            cap.release()
            if not frames:
                raise ValueError(f"cannot read video {self.path}")
            self.cache["frames"] = frames
        return self.cache["frames"]


@dataclass
class GateContext:
    cfg: EngineConfig
    analyzer: FaceAnalyzer
    centroids: dict[str, np.ndarray]
    ocr: OCREngine | None = None
    judge: Any | None = None  # creator_engine.judge.ClaudeJudge


class Gate:
    name = "gate"
    stage = KEYFRAME
    defaults: dict[str, Any] = {}

    def params(self, ctx: GateContext) -> dict[str, Any]:
        return {**self.defaults, **ctx.cfg.gate(self.name).params}

    def check(self, art: Artifact, ctx: GateContext) -> GateResult:  # pragma: no cover - interface
        raise NotImplementedError


# ---------------------------------------------------------------- helpers


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def faces_of(art: Artifact, ctx: GateContext) -> list[Face]:
    if "faces" not in art.cache:
        art.cache["faces"] = ctx.analyzer.detect(art.image)
    return art.cache["faces"]


def face_crop(img: np.ndarray, face: Face, inner: float = 1.0) -> np.ndarray:
    x, y, w, h = face.box
    pad_w, pad_h = int(w * (1 - inner) / 2), int(h * (1 - inner) / 2)
    return img[max(0, y + pad_h) : y + h - pad_h, max(0, x + pad_w) : x + w - pad_w]


def exposure_stats(img: np.ndarray) -> dict[str, float]:
    g = _gray(img)
    return {
        "clip_hi": float(np.mean(g >= 250)),
        "clip_lo": float(np.mean(g <= 5)),
        "mean": float(g.mean()),
    }


def laplacian_var(img: np.ndarray, height: int = 160) -> float:
    g = _gray(img)
    scale = height / max(1, g.shape[0])
    g = cv2.resize(g, (max(8, int(g.shape[1] * scale)), height), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def skin_hf(img: np.ndarray, face: Face) -> float:
    """Robust high-frequency energy of the face interior, normalized for exposure."""
    crop = face_crop(img, face, inner=0.6)
    if crop.size == 0:
        return 0.0
    g = _gray(crop).astype(np.float32)
    if g.shape[0] > 384:  # measure near native resolution; only tame very large faces
        g = cv2.resize(g, (int(g.shape[1] * 384 / g.shape[0]), 384), interpolation=cv2.INTER_AREA)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.2)
    mad = float(np.median(np.abs(hp - np.median(hp)))) * 1.4826
    return mad / (float(g.mean()) / 128.0 + 1e-6)


def expected_text(prompt: str) -> set[str]:
    words: set[str] = set()
    for quoted in re.findall(r"[\"“']([^\"”']{2,40})[\"”']", prompt):
        words.update(w.lower() for w in tokens(quoted))
    return words


# ---------------------------------------------------------------- keyframe gates


class ExposureGate(Gate):
    name, stage = "exposure", KEYFRAME
    defaults = {"max_clip_hi": 0.04, "max_clip_lo": 0.10, "mean_min": 45.0, "mean_max": 210.0}

    def check(self, art, ctx):
        p = self.params(ctx)
        s = exposure_stats(art.image)
        bad = []
        if s["clip_hi"] > p["max_clip_hi"]:
            bad.append(f"{s['clip_hi']:.1%} blown highlights")
        if s["clip_lo"] > p["max_clip_lo"]:
            bad.append(f"{s['clip_lo']:.1%} crushed shadows")
        if not p["mean_min"] <= s["mean"] <= p["mean_max"]:
            bad.append(f"mean luminance {s['mean']:.0f}")
        return GateResult(self.name, not bad, s["mean"], "; ".join(bad), s)


class FaceGate(Gate):
    name, stage = "face", KEYFRAME
    defaults = {"min_score": 0.6, "max_faces": 1, "secondary_ratio": 0.25}

    def check(self, art, ctx):
        p = self.params(ctx)
        faces = [f for f in faces_of(art, ctx) if f.score >= p["min_score"]]
        if not faces:
            return GateResult(self.name, False, 0.0, "no face detected", {"faces": 0})
        main = faces[0]
        # extra faces that are large relative to the main one usually mean a duplicated person
        extra = [f for f in faces[1:] if f.w * f.h >= p["secondary_ratio"] * main.w * main.h]
        ok = len(extra) + 1 <= p["max_faces"]
        return GateResult(
            self.name,
            ok,
            main.score,
            "" if ok else f"{len(extra) + 1} prominent faces",
            {"faces": len(faces), "score": main.score},
        )


class SharpnessGate(Gate):
    name, stage = "sharpness", KEYFRAME
    defaults = {"min_laplacian_var": 60.0}

    def check(self, art, ctx):
        p = self.params(ctx)
        faces = faces_of(art, ctx)
        region = face_crop(art.image, faces[0]) if faces else art.image
        v = laplacian_var(region)
        ok = v >= p["min_laplacian_var"]
        return GateResult(
            self.name, ok, v, "" if ok else f"soft focus (laplacian var {v:.0f})", {"laplacian_var": v}
        )


class HeadSizeGate(Gate):
    name, stage = "head_size", KEYFRAME

    def check(self, art, ctx):
        faces = faces_of(art, ctx)
        if not faces:
            return GateResult(self.name, None, None, "no face")
        r = faces[0].h / art.image.shape[0]
        lo, hi = art.fmt.head_ratio
        ok = lo <= r <= hi
        why = (
            ""
            if ok
            else (
                f"head too large for this shot ({r:.2f} > {hi})"
                if r > hi
                else f"head too small ({r:.2f} < {lo})"
            )
        )
        return GateResult(self.name, ok, r, why, {"head_ratio": r, "range": [lo, hi]})


class SkinTextureGate(Gate):
    name, stage = "skin_texture", KEYFRAME
    defaults = {"min_hf": 2.5}

    def check(self, art, ctx):
        p = self.params(ctx)
        faces = faces_of(art, ctx)
        if not faces:
            return GateResult(self.name, None, None, "no face")
        hf = skin_hf(art.image, faces[0])
        ok = hf >= p["min_hf"]
        return GateResult(self.name, ok, hf, "" if ok else f"waxy skin (texture {hf:.2f})", {"skin_hf": hf})


class IdentityGate(Gate):
    name, stage = "identity", KEYFRAME
    defaults = {"min_cosine": 0.70}

    def check(self, art, ctx):
        p = self.params(ctx)
        centroid = ctx.centroids.get(art.character.id)
        faces = faces_of(art, ctx)
        if centroid is None:
            return GateResult(
                self.name,
                None,
                None,
                f"no identity centroid for {art.character.id} (run `ce identity enroll`)",
            )
        if not faces:
            return GateResult(self.name, False, 0.0, "no face to compare")
        emb = ctx.analyzer.embed(art.image, faces[0])
        s = cosine(emb, centroid)
        art.cache["identity"] = s
        ok = s >= p["min_cosine"]
        return GateResult(
            self.name,
            ok,
            s,
            "" if ok else f"face drifted from {art.character.name} (cos {s:.2f})",
            {"cosine": s},
        )


class SignTextGate(Gate):
    name, stage = "sign_text", KEYFRAME
    defaults = {"min_confidence": 0.5}

    def check(self, art, ctx):
        if not art.fmt.has_signage:
            return GateResult(self.name, True, None, "", {"skipped": "format has no signage"})
        if ctx.ocr is None:
            return GateResult(self.name, None, None, "OCR not installed (pip install creator-engine[ocr])")
        p = self.params(ctx)
        hits = [h for h in ctx.ocr.read(art.image) if h.confidence >= p["min_confidence"]]
        expected = expected_text(art.prompt)
        verdicts = {}
        for h in hits:
            for tok in tokens(h.text):
                verdicts[tok] = classify_token(tok, expected, set(art.fmt.allowed_text))
        bad = [t for t, v in verdicts.items() if v == "misspelled"]
        return GateResult(
            self.name,
            not bad,
            float(len(bad)),
            "" if not bad else f"broken sign text: {', '.join(bad)}",
            {"text": [h.text for h in hits], "expected": sorted(expected), "verdicts": verdicts},
        )


class JudgeGate(Gate):
    """Claude vision rubric. Runs last, only on survivors, so it costs fractions of a cent per keeper."""

    name, stage = "judge", KEYFRAME

    def __init__(self, stage: str = KEYFRAME):
        self.stage = stage
        self.name = "judge" if stage == KEYFRAME else "judge_video"

    def check(self, art, ctx):
        if ctx.judge is None:
            return GateResult(self.name, None, None, "judge disabled")
        if self.stage == KEYFRAME:
            verdict = ctx.judge.judge_images([art.image], art.prompt)
        else:
            frames = art.frames
            picks = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, 4)]
            verdict = ctx.judge.judge_images(picks, art.prompt, video=True)
        return GateResult(self.name, verdict.passed, verdict.min_score, verdict.summary, verdict.as_metrics())


# ---------------------------------------------------------------- video gates


def _small_grays(frames: list[np.ndarray], width: int = 128) -> list[np.ndarray]:
    out = []
    for f in frames:
        h, w = f.shape[:2]
        out.append(
            cv2.resize(_gray(f), (width, max(8, int(h * width / w))), interpolation=cv2.INTER_AREA).astype(
                np.float32
            )
        )
    return out


class MotionGate(Gate):
    name, stage = "motion", VIDEO
    defaults = {"min_motion": 0.35, "max_motion": 14.0}

    def check(self, art, ctx):
        p = self.params(ctx)
        g = _small_grays(art.frames)
        diffs = [float(np.mean(np.abs(b - a))) for a, b in zip(g, g[1:], strict=False)]
        m = float(np.median(diffs)) if diffs else 0.0
        if m < p["min_motion"]:
            return GateResult(self.name, False, m, f"frozen clip (motion {m:.2f})", {"motion": m})
        if m > p["max_motion"]:
            return GateResult(self.name, False, m, f"chaotic motion ({m:.1f})", {"motion": m})
        return GateResult(self.name, True, m, "", {"motion": m})


class FlickerGate(Gate):
    name, stage = "flicker", VIDEO
    defaults = {"max_luma_jump": 5.0}

    def check(self, art, ctx):
        p = self.params(ctx)
        means = np.array([float(_gray(f).mean()) for f in art.frames])
        jumps = np.abs(np.diff(means)) if len(means) > 1 else np.zeros(1)
        j = float(np.percentile(jumps, 95))
        ok = j <= p["max_luma_jump"]
        return GateResult(
            self.name, ok, j, "" if ok else f"flicker (luma jumps {j:.1f})", {"luma_jump_p95": j}
        )


class TemporalIdentityGate(Gate):
    name, stage = "temporal_identity", VIDEO
    defaults = {"min_cosine": 0.62, "samples": 8, "max_missing": 0.2}

    def check(self, art, ctx):
        p = self.params(ctx)
        centroid = ctx.centroids.get(art.character.id)
        if centroid is None:
            return GateResult(self.name, None, None, "no identity centroid")
        frames = art.frames
        idx = np.linspace(0, len(frames) - 1, int(p["samples"])).astype(int)
        sims, missing = [], 0
        for i in idx:
            faces = ctx.analyzer.detect(frames[i])
            if not faces:
                missing += 1
                continue
            sims.append(cosine(ctx.analyzer.embed(frames[i], faces[0]), centroid))
        miss_frac = missing / len(idx)
        if not sims or miss_frac > p["max_missing"]:
            return GateResult(
                self.name,
                False,
                0.0,
                f"face lost in {miss_frac:.0%} of sampled frames",
                {"missing": miss_frac},
            )
        worst = float(min(sims))
        ok = worst >= p["min_cosine"]
        return GateResult(
            self.name,
            ok,
            worst,
            "" if ok else f"face morphs mid-clip (worst frame cos {worst:.2f})",
            {"min_cosine": worst, "mean_cosine": float(np.mean(sims)), "missing": miss_frac},
        )


KEYFRAME_GATES: list[Gate] = [
    ExposureGate(),
    FaceGate(),
    SharpnessGate(),
    HeadSizeGate(),
    SkinTextureGate(),
    IdentityGate(),
    SignTextGate(),
    JudgeGate(KEYFRAME),
]
VIDEO_GATES: list[Gate] = [FlickerGate(), MotionGate(), TemporalIdentityGate(), JudgeGate(VIDEO)]


def run_gates(art: Artifact, ctx: GateContext, full: bool = False) -> tuple[bool, list[GateResult]]:
    """Run the stage's gates in order. Stops at the first failure unless ``full``."""
    gates = KEYFRAME_GATES if art.stage == KEYFRAME else VIDEO_GATES
    results: list[GateResult] = []
    passed = True
    for gate in gates:
        if not ctx.cfg.gate(gate.name).enabled:
            continue
        if isinstance(gate, JudgeGate) and (ctx.judge is None or art.stage not in ctx.cfg.judge.stages):
            continue
        r = gate.check(art, ctx)
        results.append(r)
        if r.passed is False:
            passed = False
            if not full:
                break
    return passed, results
