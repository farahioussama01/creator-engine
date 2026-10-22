"""Procedural stand-ins for diffusion outputs, used by the GPU simulator.

The point is not to look like photos. The point is that every quality gate downstream runs on real
pixels: exposure is measured from the histogram, blur from the Laplacian, skin texture from
high-frequency energy, and identity from a 16-cell luminance pattern painted inside the face oval.
The pattern is derived from the LoRA name, so in the simulator, as on a GPU, the LoRA carries the
face. Each injected defect mimics a real failure mode of LoRA keyframes and Wan clips.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

GRID = 4  # identity pattern is GRID x GRID cells
SKIN_HSV = (12, 105, 205)  # OpenCV HSV (H in 0..180)
PATTERN_AMP = 26.0

KEYFRAME_DEFECTS = (
    "blur",
    "overexposed",
    "underexposed",
    "waxy",
    "identity_drift",
    "big_head",
    "misspelled_sign",
)
VIDEO_DEFECTS = ("morph", "flicker", "frozen", "chaos")


def identity_vector(key: str) -> np.ndarray:
    """Deterministic unit vector for a LoRA / character key."""
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    v = np.random.default_rng(seed).normal(size=GRID * GRID)
    v -= v.mean()
    return v / np.linalg.norm(v)


def _hsv_to_bgr(h: float, s: float, v: float) -> tuple[int, int, int]:
    px = np.uint8([[[int(h) % 180, int(np.clip(s, 0, 255)), int(np.clip(v, 0, 255))]]])
    b, g, r = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


@dataclass
class KeyframeScene:
    width: int
    height: int
    seed: int
    identity: np.ndarray
    lora_strength: float = 0.9
    sign_text: str | None = None
    defects: set[str] = field(default_factory=set)


def misspell(word: str, rng: np.random.Generator) -> str:
    """The way diffusion models break sign text: a doubled, dropped or swapped letter."""
    if len(word) < 3:
        return word + word[-1]
    i = int(rng.integers(1, len(word) - 1))
    op = int(rng.integers(0, 3))
    if op == 0:
        return word[:i] + word[i] + word[i:]
    if op == 1:
        return word[:i] + word[i + 1 :]
    return word[: i - 1] + word[i] + word[i - 1] + word[i + 1 :]


def _paint_pattern(
    img: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int], vec: np.ndarray, amp: float
) -> None:
    x, y, w, h = box
    ix0, iy0 = x + int(0.15 * w), y + int(0.15 * h)
    iw, ih = int(0.70 * w), int(0.70 * h)
    offsets = (vec * GRID * amp).reshape(GRID, GRID)
    layer = cv2.resize(offsets.astype(np.float32), (iw, ih), interpolation=cv2.INTER_NEAREST)
    region = img[iy0 : iy0 + ih, ix0 : ix0 + iw].astype(np.float32)
    m = mask[iy0 : iy0 + ih, ix0 : ix0 + iw, None].astype(np.float32) / 255.0
    region += layer[: region.shape[0], : region.shape[1], None] * m[: layer.shape[0], : layer.shape[1]]
    img[iy0 : iy0 + ih, ix0 : ix0 + iw] = np.clip(region, 0, 255).astype(np.uint8)


def render_keyframe(scene: KeyframeScene) -> tuple[np.ndarray, dict]:
    """Returns a BGR image and ground truth (face box, identity used, defects)."""
    rng = np.random.default_rng(scene.seed)
    W, H = scene.width, scene.height

    # Background: cool hues only, so nothing but the face reads as skin.
    h1, h2 = rng.uniform(95, 165, size=2)
    top = np.array(_hsv_to_bgr(h1, rng.uniform(40, 110), rng.uniform(150, 210)), np.float32)
    bot = np.array(_hsv_to_bgr(h2, rng.uniform(40, 110), rng.uniform(70, 130)), np.float32)
    t = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    img = (top * (1 - t) + bot * t).repeat(W, axis=1).astype(np.uint8)
    for _ in range(int(rng.integers(3, 7))):
        x0, y0 = int(rng.uniform(0, W * 0.8)), int(rng.uniform(H * 0.3, H * 0.9))
        cv2.rectangle(
            img,
            (x0, y0),
            (x0 + int(rng.uniform(40, W * 0.35)), y0 + int(rng.uniform(40, H * 0.25))),
            _hsv_to_bgr(rng.uniform(95, 165), rng.uniform(30, 90), rng.uniform(50, 140)),
            -1,
        )

    # Head and body
    ratio = rng.uniform(0.11, 0.17) * (2.3 if "big_head" in scene.defects else 1.0)
    fh = int(ratio * H)
    fw = int(0.72 * fh)
    cx = int(W / 2 + rng.uniform(-0.08, 0.08) * W)
    cy = int(H * rng.uniform(0.24, 0.32)) if "big_head" not in scene.defects else int(H * 0.36)
    cloth = _hsv_to_bgr(rng.uniform(100, 160), rng.uniform(60, 160), rng.uniform(60, 160))
    shoulders = np.array(
        [
            [cx - int(1.6 * fw), H],
            [cx - int(1.1 * fw), cy + int(0.62 * fh)],
            [cx + int(1.1 * fw), cy + int(0.62 * fh)],
            [cx + int(1.6 * fw), H],
        ],
        np.int32,
    )
    cv2.fillPoly(img, [shoulders], cloth)
    hair = _hsv_to_bgr(rng.uniform(100, 140), rng.uniform(30, 80), rng.uniform(25, 60))
    cv2.ellipse(img, (cx, cy - int(0.06 * fh)), (int(0.62 * fw), int(0.6 * fh)), 0, 0, 360, hair, -1)

    mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(mask, (cx, cy), (fw // 2, fh // 2), 0, 0, 360, 255, -1)
    skin = np.array(_hsv_to_bgr(*SKIN_HSV), np.uint8)
    img[mask > 0] = skin

    # Identity: the LoRA's vector plus per-image variation; weaker LoRA strength = looser identity.
    sigma = 0.10 + 0.30 * max(0.0, 1.0 - scene.lora_strength)
    vec = scene.identity + rng.normal(0, sigma, size=scene.identity.shape)
    if "identity_drift" in scene.defects:
        other = rng.normal(size=scene.identity.shape)
        other /= np.linalg.norm(other)
        vec = 0.35 * scene.identity + 0.65 * other
    vec -= vec.mean()
    vec /= np.linalg.norm(vec)
    box = (cx - fw // 2, cy - fh // 2, fw, fh)
    _paint_pattern(img, mask, box, vec, PATTERN_AMP)

    # Skin micro-texture (pores). Waxy skin = the texture is gone.
    face_px = img.astype(np.float32)
    if "waxy" in scene.defects:
        smooth = cv2.GaussianBlur(face_px, (0, 0), 2.2)
        face_px = np.where(mask[..., None] > 0, smooth, face_px)
        tex_sigma = 0.6
    else:
        tex_sigma = 7.0
    noise = rng.normal(0, tex_sigma, size=(H, W, 1)).astype(np.float32)
    face_px = np.where(mask[..., None] > 0, face_px + noise, face_px)
    img = np.clip(face_px, 0, 255).astype(np.uint8)

    # Sign text away from the face
    rendered_text = None
    if scene.sign_text:
        rendered_text = (
            misspell(scene.sign_text, rng) if "misspelled_sign" in scene.defects else scene.sign_text
        )
        scale = W / 330
        (tw, th), _ = cv2.getTextSize(rendered_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
        sx = int(W * 0.06) if cx > W / 2 else int(W - tw - W * 0.1)
        sy = int(H * 0.06)
        cv2.rectangle(img, (sx - 14, sy - 14), (sx + tw + 14, sy + th + 18), (245, 245, 245), -1)
        cv2.putText(
            img, rendered_text, (sx, sy + th), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 3, cv2.LINE_AA
        )

    # Global camera effects
    out = img.astype(np.float32) + rng.normal(0, 2.0, size=img.shape).astype(np.float32)
    if "blur" in scene.defects:
        out = cv2.GaussianBlur(out, (0, 0), 3.5)
    if "overexposed" in scene.defects:
        out = out * 1.9 + 45
    if "underexposed" in scene.defects:
        out = out * 0.28
    out = np.clip(out, 0, 255).astype(np.uint8)
    truth = {
        "face_box": list(map(int, box)),
        "head_ratio": fh / H,
        "defects": sorted(scene.defects),
        "sign_text": rendered_text,
    }
    return out, truth


def fit_to(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize + center-crop, the way WanImageToVideo fits the start image."""
    h, w = img.shape[:2]
    s = max(width / w, height / h)
    r = cv2.resize(img, (math.ceil(w * s), math.ceil(h * s)), interpolation=cv2.INTER_AREA)
    y0 = (r.shape[0] - height) // 2
    x0 = (r.shape[1] - width) // 2
    return r[y0 : y0 + height, x0 : x0 + width].copy()


def _skin_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 45, 70), (28, 215, 255))


def render_video(
    start: np.ndarray, width: int, height: int, frames: int, seed: int, defects: set[str]
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    base = fit_to(start, width, height)
    mask = _skin_mask(base)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    morph_delta = None
    if "morph" in defects:
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n > 1:
            i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            x, y, w, h = (int(v) for v in stats[i, :4])
            other = rng.normal(size=GRID * GRID)
            other -= other.mean()
            other /= np.linalg.norm(other)
            morph_delta = np.zeros_like(base, np.float32)
            layer = np.zeros(base.shape[:2], np.uint8)
            layer[mask > 0] = 255
            _paint_pattern_delta(morph_delta, layer, (x, y, w, h), other)

    out = []
    pan_phase = rng.uniform(0, math.tau)
    grain = [rng.normal(0, 1.5, size=base.shape).astype(np.float32) for _ in range(3)]
    for f in range(frames):
        t = f / max(1, frames - 1)
        frame = base.astype(np.float32)
        if morph_delta is not None and t > 0.4:
            frame = frame + morph_delta * min(1.0, (t - 0.4) / 0.35)
        if "frozen" in defects:
            M = np.float32([[1, 0, 0], [0, 1, 0]])
        else:
            z = 1.0 + 0.22 * t
            dx = 0.08 * width * math.sin(pan_phase + 2.6 * t)
            if "chaos" in defects:
                dx += rng.uniform(-0.18, 0.18) * width
                z += rng.uniform(-0.15, 0.15)
            M = cv2.getRotationMatrix2D((width / 2, height * 0.35), 0, z)
            M[0, 2] += dx
        frame = cv2.warpAffine(frame, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        if "frozen" not in defects:
            frame = frame + grain[f % 3]
        if "flicker" in defects:
            frame = frame * (1.0 + rng.uniform(-0.28, 0.28))
        out.append(np.clip(frame, 0, 255).astype(np.uint8))
    return out


def _paint_pattern_delta(
    dst: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int], vec: np.ndarray
) -> None:
    """Write (new pattern - estimated current pattern) into dst so the face morphs to another identity."""
    x, y, w, h = box
    ix0, iy0 = x + int(0.15 * w), y + int(0.15 * h)
    iw, ih = max(1, int(0.70 * w)), max(1, int(0.70 * h))
    layer = cv2.resize(
        (vec * GRID * PATTERN_AMP * 1.6).reshape(GRID, GRID).astype(np.float32),
        (iw, ih),
        interpolation=cv2.INTER_NEAREST,
    )
    m = mask[iy0 : iy0 + ih, ix0 : ix0 + iw].astype(np.float32) / 255.0
    hh, ww = m.shape
    dst[iy0 : iy0 + hh, ix0 : ix0 + ww] += (layer[:hh, :ww] * m)[..., None]


def write_mp4(frames: list[np.ndarray], path: str, fps: int) -> None:
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    for fr in frames:
        vw.write(fr)
    vw.release()
