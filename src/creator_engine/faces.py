"""Face detection and identity, on CPU, with commercially usable models.

- YuNet (OpenCV Zoo, MIT) finds faces and five landmarks.
- AuraFace (fal, Apache-2.0) turns an aligned face into a 512-number "fingerprint".
  (The popular InsightFace weights are non-commercial only, so they are not used.)

Two faces are the same person when their fingerprints point the same way: cosine similarity.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_URLS = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "auraface_glintr100.onnx": "https://huggingface.co/fal/AuraFace-v1/resolve/main/glintr100.onnx",
}

# Where the five landmarks of a 112x112 ArcFace-style crop must land (image-left eye first).
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


@dataclass
class Face:
    x: int
    y: int
    w: int
    h: int
    score: float
    landmarks: np.ndarray | None = None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def align_face(img_bgr: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    M, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), ARCFACE_TEMPLATE * (size / 112), method=cv2.LMEDS)
    if M is None:
        raise ValueError("could not align face")
    return cv2.warpAffine(img_bgr, M, (size, size), borderValue=0)


def preprocess(aligned_bgr: np.ndarray) -> np.ndarray:
    """112x112 BGR -> 1x3x112x112 RGB in [-1, 1]."""
    rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.transpose((rgb - 127.5) / 127.5, (2, 0, 1))[None]


class RealFaces:
    def __init__(self, models_dir: str | Path = "models", min_score: float = 0.7):
        d = Path(models_dir)
        det, emb = d / "face_detection_yunet_2023mar.onnx", d / "auraface_glintr100.onnx"
        for p in (det, emb):
            if not p.exists():
                raise FileNotFoundError(f"{p} is missing: run `ce models`")
        import onnxruntime as ort

        self._det = cv2.FaceDetectorYN.create(str(det), "", (320, 320), min_score, 0.3, 50)
        self._emb = ort.InferenceSession(str(emb), providers=["CPUExecutionProvider"])
        self._lock = threading.Lock()  # the detector keeps per-call state

    def detect(self, img: np.ndarray) -> list[Face]:
        with self._lock:
            self._det.setInputSize((img.shape[1], img.shape[0]))
            _, rows = self._det.detect(img)
        faces = [
            Face(*map(int, r[:4]), score=float(r[14]), landmarks=r[4:14].reshape(5, 2).astype(np.float32))
            for r in (rows if rows is not None else [])
        ]
        return sorted(faces, key=lambda f: f.w * f.h, reverse=True)

    def embed(self, img: np.ndarray, face: Face) -> np.ndarray:
        x = preprocess(align_face(img, face.landmarks))
        v = self._emb.run(None, {self._emb.get_inputs()[0].name: x})[0][0]
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


@dataclass
class Enrollment:
    centroid: np.ndarray
    used: int
    outliers: list[str]
    no_face: list[str]
    typical_similarity: float
    suggested_threshold: float


def enroll(faces, images: list[Path]) -> Enrollment:
    """Average fingerprint of the training images, minus outliers, and a threshold that fits this face."""
    vecs, names, no_face = [], [], []
    for p in images:
        img = cv2.imread(str(p))
        found = faces.detect(img) if img is not None else []
        if not found:
            no_face.append(p.name)
            continue
        vecs.append(faces.embed(img, found[0]))
        names.append(p.name)
    if len(vecs) < 3:
        raise ValueError(f"need at least 3 images with a visible face, found {len(vecs)}")
    V = np.stack(vecs)

    def leave_one_out(V: np.ndarray) -> np.ndarray:
        total = V.sum(0)
        return np.array([cosine(V[i], total - V[i]) for i in range(len(V))])

    sims = leave_one_out(V)
    med = np.median(sims)
    spread = np.median(np.abs(sims - med)) * 1.4826 + 1e-6
    keep = sims >= med - 3 * spread
    V = V[keep]
    sims = leave_one_out(V)
    centroid = V.mean(0)
    return Enrollment(
        centroid=(centroid / np.linalg.norm(centroid)).astype(np.float32),
        used=int(keep.sum()),
        outliers=[n for n, k in zip(names, keep, strict=True) if not k],
        no_face=no_face,
        typical_similarity=float(np.median(sims)),
        suggested_threshold=round(max(0.3, float(np.percentile(sims, 10)) - 0.12), 3),
    )


def download_models(models_dir: str | Path = "models") -> list[Path]:
    import httpx

    d = Path(models_dir)
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for name, url in MODEL_URLS.items():
        target = d / name
        if not target.exists():
            with httpx.stream("GET", url, follow_redirects=True, timeout=300) as r:
                r.raise_for_status()
                with open(target, "wb") as fh:
                    for chunk in r.iter_bytes():
                        fh.write(chunk)
        out.append(target)
    return out
