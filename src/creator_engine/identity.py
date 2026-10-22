"""Identity centroids: what "this character's face" means numerically.

Enrollment embeds the character's reference images (typically the LoRA training set), drops
outliers, averages, and suggests a gate threshold from the leave-one-out similarity distribution,
so the threshold is calibrated to *this* face and *this* embedder instead of a magic number.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import CharacterSpec, EngineConfig
from .vision import FaceAnalyzer, cosine

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class Enrollment:
    character: str
    analyzer: str
    n_images: int
    n_used: int
    no_face: list[str]
    outliers: list[str]
    loo_min: float
    loo_p10: float
    loo_median: float
    suggested_threshold: float


def enroll(analyzer: FaceAnalyzer, character: str, images: list[Path]) -> tuple[np.ndarray, Enrollment]:
    embs, names, no_face = [], [], []
    for p in images:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        faces = analyzer.detect(img) if img is not None else []
        if not faces:
            no_face.append(p.name)
            continue
        embs.append(analyzer.embed(img, faces[0]))
        names.append(p.name)
    if len(embs) < 3:
        raise ValueError(f"need at least 3 reference images with a detectable face, got {len(embs)}")
    E = np.stack(embs)

    def loo(E: np.ndarray) -> np.ndarray:
        total = E.sum(0)
        return np.array([cosine(E[i], (total - E[i]) / (len(E) - 1)) for i in range(len(E))])

    sims = loo(E)
    med = float(np.median(sims))
    mad = float(np.median(np.abs(sims - med))) * 1.4826 + 1e-6
    keep = sims >= med - 3 * mad
    outliers = [n for n, k in zip(names, keep, strict=True) if not k]
    E = E[keep]
    sims = loo(E)
    centroid = E.mean(0)
    centroid /= np.linalg.norm(centroid)
    p10 = float(np.percentile(sims, 10))
    report = Enrollment(
        character=character,
        analyzer=analyzer.name,
        n_images=len(images),
        n_used=int(len(E)),
        no_face=no_face,
        outliers=outliers,
        loo_min=float(sims.min()),
        loo_p10=p10,
        loo_median=float(np.median(sims)),
        suggested_threshold=round(max(0.3, p10 - 0.12), 3),
    )
    return centroid.astype(np.float32), report


def identity_dir(cfg: EngineConfig) -> Path:
    return cfg.path(cfg.workdir) / "identity"


def save(cfg: EngineConfig, centroid: np.ndarray, report: Enrollment) -> Path:
    d = identity_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{report.character}.npy", centroid)
    (d / f"{report.character}.json").write_text(json.dumps(asdict(report), indent=2))
    return d / f"{report.character}.npy"


def load_centroids(cfg: EngineConfig) -> dict[str, np.ndarray]:
    d = identity_dir(cfg)
    return {p.stem: np.load(p) for p in d.glob("*.npy")} if d.exists() else {}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)


def make_sim_references(
    character: CharacterSpec, out_dir: Path, n: int = 16, width: int = 768, height: int = 1344
) -> list[Path]:
    """Simulator only: render clean reference shots of the character (what a LoRA dataset would be)."""
    from .sim.render import KeyframeScene, identity_vector, render_keyframe

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        img, _ = render_keyframe(
            KeyframeScene(width, height, 50_000 + i, identity_vector(character.lora), 1.0)
        )
        p = out_dir / f"ref_{i:02d}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths
