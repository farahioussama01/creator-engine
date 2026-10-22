"""Production face analysis: YuNet (OpenCV Zoo, MIT) + AuraFace (fal, Apache-2.0).

AuraFace is an ArcFace-style ResNet-100 trained on commercially usable data. It is a drop-in for
InsightFace's recognition models, whose weights are licensed for non-commercial research only.

Model files (fetch with ``ce models download``):
  models/face_detection_yunet_2023mar.onnx
  models/auraface_glintr100.onnx
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import Face

# ArcFace 112x112 alignment template, image-left eye first (same order as YuNet landmarks)
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)

MODEL_URLS = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "auraface_glintr100.onnx": "https://huggingface.co/fal/AuraFace-v1/resolve/main/glintr100.onnx",
}


def align_face(img_bgr: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    """Similarity-transform the five landmarks onto the ArcFace template."""
    dst = ARCFACE_TEMPLATE * (size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), dst, method=cv2.LMEDS)
    if M is None:
        raise ValueError("could not estimate alignment transform")
    return cv2.warpAffine(img_bgr, M, (size, size), borderValue=0)


def preprocess(aligned_bgr: np.ndarray) -> np.ndarray:
    """112x112 BGR -> 1x3x112x112 float32 RGB scaled to [-1, 1]."""
    rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = (rgb - 127.5) / 127.5
    return np.transpose(x, (2, 0, 1))[None]


class RealFaceAnalyzer:
    name = "real"

    def __init__(self, models_dir: str | Path, score_threshold: float = 0.7):
        models_dir = Path(models_dir)
        det_path = models_dir / "face_detection_yunet_2023mar.onnx"
        emb_path = models_dir / "auraface_glintr100.onnx"
        for p in (det_path, emb_path):
            if not p.exists():
                raise FileNotFoundError(f"{p} missing - run `ce models download`")
        self._det = cv2.FaceDetectorYN.create(str(det_path), "", (320, 320), score_threshold, 0.3, 50)
        import onnxruntime as ort  # optional dependency

        providers = [
            p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()
        ]
        self._emb = ort.InferenceSession(str(emb_path), providers=providers)
        self._input = self._emb.get_inputs()[0].name

    def detect(self, img_bgr: np.ndarray) -> list[Face]:
        h, w = img_bgr.shape[:2]
        self._det.setInputSize((w, h))
        _, rows = self._det.detect(img_bgr)
        faces: list[Face] = []
        if rows is None:
            return faces
        for r in rows:
            x, y, fw, fh = (int(v) for v in r[:4])
            lm = np.array(r[4:14], dtype=np.float32).reshape(5, 2)
            faces.append(Face(x, y, fw, fh, score=float(r[14]), landmarks=lm))
        faces.sort(key=lambda f: f.w * f.h, reverse=True)
        return faces

    def embed(self, img_bgr: np.ndarray, face: Face) -> np.ndarray:
        if face.landmarks is None:
            raise ValueError("real embedder needs landmarks")
        x = preprocess(align_face(img_bgr, face.landmarks))
        feat = self._emb.run(None, {self._input: x})[0][0].astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-9)
