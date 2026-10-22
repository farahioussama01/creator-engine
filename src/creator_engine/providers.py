"""Where renders come from. Every provider has the same two methods, so the pipeline doesn't care:

    keyframe(prompt, seed, out) -> Render      a 9:16 still of the character
    clip(keyframe, prompt, seed, out) -> Render a ~5 s video animated from that still

- FalProvider: hosted open models on fal.ai, pay per render, no GPU (Z-Image Turbo + your LoRA,
  Wan 2.2 A14B turbo image-to-video). Cost = fal's list price per render.
- ComfyProvider: the same models in your own ComfyUI on a rented GPU. Cost = GPU seconds x $/hour.
- MockProvider: offline cartoons for trying the flow. Cost = what fal would have charged.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import httpx

from . import mock
from .config import Config


@dataclass
class Render:
    path: Path
    usd: float
    seconds: float
    url: str | None = None  # where the provider can read the file again (saves an upload)


class RenderFailed(Exception):
    def __init__(self, message: str, usd: float = 0.0):
        super().__init__(message)
        self.usd = usd


async def _download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as http:
        r = await http.get(url)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def fal_price(cfg: Config, stage: str) -> float:
    """fal list price for one render."""
    if stage == "keyframe":
        return cfg.keyframe_width * cfg.keyframe_height / 1e6 * cfg.prices.keyframe_per_mp
    return cfg.prices.clip[cfg.clip_resolution]


class FalProvider:
    name = "fal"
    KEYFRAME_MODEL = "fal-ai/z-image/turbo/lora"
    CLIP_MODEL = "fal-ai/wan/v2.2-a14b/image-to-video/turbo"

    def __init__(self, cfg: Config, client=None, download=_download):
        self.cfg = cfg
        if client is None:
            if not os.environ.get("FAL_KEY"):
                raise RuntimeError("set FAL_KEY (fal.ai -> Dashboard -> Keys)")
            import fal_client as client
        self.client = client
        self.download = download
        self.lora_url = cfg.character_state().get("lora_url")
        if not self.lora_url:
            raise RuntimeError("no LoRA yet: run `ce character train` first")

    def price(self, stage: str) -> float:
        return fal_price(self.cfg, stage)

    async def keyframe(self, prompt: str, seed: int, out: Path) -> Render:
        t0 = time.monotonic()
        res = await self.client.subscribe_async(
            self.KEYFRAME_MODEL,
            arguments={
                "prompt": prompt,
                "image_size": {"width": self.cfg.keyframe_width, "height": self.cfg.keyframe_height},
                "num_inference_steps": 8,
                "seed": seed,
                "num_images": 1,
                "output_format": "png",
                "enable_safety_checker": True,
                "loras": [{"path": self.lora_url, "scale": self.cfg.character.lora_scale}],
            },
        )
        if (res.get("has_nsfw_concepts") or [False])[0]:
            raise RenderFailed("blocked by the safety filter", usd=self.price("keyframe"))
        url = res["images"][0]["url"]
        path = await self.download(url, out.with_suffix(".png"))
        return Render(path, self.price("keyframe"), time.monotonic() - t0, url)

    async def clip(self, keyframe: Render, prompt: str, seed: int, out: Path) -> Render:
        t0 = time.monotonic()
        image_url = keyframe.url or await self.client.upload_file_async(keyframe.path)
        res = await self.client.subscribe_async(
            self.CLIP_MODEL,
            arguments={
                "image_url": image_url,
                "prompt": prompt,
                "seed": seed,
                "resolution": self.cfg.clip_resolution,
                "aspect_ratio": "9:16",
            },
        )
        path = await self.download(res["video"]["url"], out.with_suffix(".mp4"))
        return Render(path, self.price("clip"), time.monotonic() - t0, res["video"]["url"])


class ComfyProvider:
    """Same pipeline on your own GPU: ComfyUI workflows in workflows/, billed by GPU seconds."""

    name = "comfyui"
    SIZES = {"480p": (480, 848), "580p": (576, 1024), "720p": (720, 1280)}

    def __init__(self, cfg: Config, client=None):
        from .comfy import ComfyClient
        from .workflow import load_template

        self.cfg = cfg
        self.client = client or ComfyClient(cfg.comfyui_url)
        self.kf_tpl = load_template(cfg.path("workflows/keyframe_zimage_lora.json"))
        self.clip_tpl = load_template(cfg.path("workflows/i2v_wan22_lightx2v.json"))
        self.rate = cfg.comfyui_usd_per_hour / 3600

    def price(self, stage: str) -> float:  # estimate, from ComfyUI's published Wan 2.2 timing on a 4090
        return (5.0 if stage == "keyframe" else 75.0) * self.rate

    async def _run(self, graph: dict, out: Path) -> Render:
        run = await self.client.run(graph)
        f = run.outputs[0]
        path = await self.client.download(f, out.parent, out.name + Path(f.filename).suffix)
        return Render(path, run.exec_seconds * self.rate, run.exec_seconds)

    async def keyframe(self, prompt: str, seed: int, out: Path) -> Render:
        from .workflow import fill

        return await self._run(
            fill(
                self.kf_tpl,
                {
                    "lora_name": self.cfg.comfyui_lora_file,
                    "lora_strength": self.cfg.character.lora_scale,
                    "positive": prompt,
                    "seed": seed,
                    "steps": 8,
                    "cfg": 1.0,
                    "width": self.cfg.keyframe_width,
                    "height": self.cfg.keyframe_height,
                    "filename_prefix": "ce/kf",
                },
            ),
            out,
        )

    async def clip(self, keyframe: Render, prompt: str, seed: int, out: Path) -> Render:
        from .workflow import fill

        name = await self.client.upload_image(keyframe.path)
        w, h = self.SIZES[self.cfg.clip_resolution]
        return await self._run(
            fill(
                self.clip_tpl,
                {
                    "start_image": name,
                    "positive": prompt,
                    "negative": "",
                    "width": w,
                    "height": h,
                    "length": 81,
                    "fps": 16,
                    "seed": seed,
                    "steps": 4,
                    "cfg": 1.0,
                    "split_step": 2,
                    "lightning_strength": 1.0,
                    "filename_prefix": "ce/clip",
                },
            ),
            out,
        )


class MockProvider:
    """Offline: draws cartoon keyframes and clips with realistic failure rates, priced like fal."""

    name = "mock"

    def __init__(self, cfg: Config, delay: float = 0.0):
        self.cfg = cfg
        self.delay = delay

    def price(self, stage: str) -> float:
        return fal_price(self.cfg, stage)  # priced as if fal had rendered it

    async def keyframe(self, prompt: str, seed: int, out: Path) -> Render:
        await asyncio.sleep(self.delay)
        defects = mock.draw_defects(random.Random(f"kf{seed}{prompt}"), ("blur", "overexposed", "drift"))
        img = mock.render_keyframe(
            self.cfg.keyframe_width // 2, self.cfg.keyframe_height // 2, seed, self.cfg.character.id, defects
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        path = out.with_suffix(".png")
        cv2.imwrite(str(path), img)
        return Render(path, self.price("keyframe"), self.delay)

    async def clip(self, keyframe: Render, prompt: str, seed: int, out: Path) -> Render:
        await asyncio.sleep(self.delay)
        defects = mock.draw_defects(random.Random(f"clip{seed}{prompt}"), ("frozen", "morph"))
        frames = await asyncio.to_thread(mock.render_clip, cv2.imread(str(keyframe.path)), 240, 424, 49, seed, defects)
        path = out.with_suffix(".mp4")
        mock.write_mp4(frames, str(path))
        return Render(path, self.price("clip"), self.delay)


def make_provider(cfg: Config):
    return {"fal": FalProvider, "comfyui": ComfyProvider, "mock": MockProvider}[cfg.provider](cfg)
