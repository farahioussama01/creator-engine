from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from creator_engine import mock
from creator_engine.config import load

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(tmp_path):
    c = load(ROOT / "configs" / "config.yaml")
    c.workdir = tmp_path / "runs"
    c.datasets_dir = tmp_path / "datasets"
    c.provider = "mock"
    return c


class FakeFal:
    """Stands in for fal_client: records every call and returns fal-shaped responses."""

    def __init__(self, nsfw: bool = False):
        self.calls: list[tuple[str, dict]] = []
        self.nsfw = nsfw
        self.n = 0

    async def subscribe_async(self, model: str, arguments: dict) -> dict:
        self.calls.append((model, arguments))
        self.n += 1
        if "image-to-video" in model:
            return {"video": {"url": f"https://fal.media/clip{self.n}.mp4"}, "seed": arguments.get("seed")}
        if "trainer" in model:
            return {
                "diffusers_lora_file": {"url": "https://fal.media/mila_lora.safetensors"},
                "config_file": {"url": "https://fal.media/config.json"},
            }
        return {
            "images": [{"url": f"https://fal.media/img{self.n}.png", "width": 720, "height": 1280}],
            "seed": arguments.get("seed"),
            "has_nsfw_concepts": [self.nsfw],
        }

    async def upload_file_async(self, path) -> str:
        self.calls.append(("upload", {"path": str(path)}))
        return f"https://fal.media/upload/{Path(path).name}"


def fake_download(who: str = "mila"):
    """Writes a mock still or clip where a real download would land."""

    async def download(url: str, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        seed = abs(hash(url)) % 10_000
        img = mock.render_keyframe(360, 640, seed, who, set())
        if path.suffix == ".mp4":
            mock.write_mp4(mock.render_clip(img, 240, 424, 33, seed, set()), str(path))
        else:
            cv2.imwrite(str(path), img)
        return path

    return download
