"""From nothing to a trained character, without a GPU (fal.ai, about $5 total).

1. portraits   render candidate faces from the character description; you pick one
2. variations  re-draw that exact person in new angles, outfits and places (image editing model)
3. check       measure how consistent the set is; drop outliers; set the identity threshold
4. train       train the LoRA on the set; store its URL for the pipeline
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .config import Config
from .faces import enroll
from .ledger import Ledger
from .providers import _download

PORTRAIT_MODEL = "fal-ai/z-image/turbo"
EDIT_MODEL = "fal-ai/qwen-image-edit-2511"
TRAIN_MODEL = "fal-ai/z-image-trainer"

VARIATIONS = [
    "head and shoulders, three-quarter view facing left, neutral expression, plain light grey wall, soft daylight",
    "head and shoulders, three-quarter view facing right, slight smile, plain light grey wall, soft daylight",
    "side profile facing left, calm expression, plain background, soft daylight",
    "close-up portrait looking straight at the camera, laughing, natural window light",
    "close-up portrait looking slightly down, thoughtful, warm indoor light",
    "half-body, sitting at a cafe table with a coffee cup, smiling, window light",
    "half-body, wearing a black turtleneck, city street behind her, overcast light",
    "full body standing in a sunlit apartment hallway, cream knit sweater and jeans",
    "full body walking on a cobblestone street at golden hour, white linen dress",
    "half-body, beach promenade, wind in her hair, denim jacket, late afternoon sun",
    "close-up, evening, warm lamp light, relaxed expression, cozy room",
    "half-body, gym mirror, grey tank top, overhead lighting, neutral expression",
    "head and shoulders, rainy window behind her, soft cool light, slight smile",
    "full body mirror selfie holding a phone, oversized blazer, bedroom",
    "close-up, outdoor park, dappled sunlight through trees, smiling",
    "half-body, kitchen counter, morning light, holding a mug, candid",
    "head and shoulders, looking over her shoulder, night city lights behind her, bokeh",
    "close-up, no makeup, freckles visible, bright diffuse light, neutral expression",
    "half-body, bookstore aisle, warm light, looking at the camera",
    "full body sitting on stairs outdoors, striped top, midday shade",
]


def _client(client=None):
    if client is not None:
        return client
    if not os.environ.get("FAL_KEY"):
        raise RuntimeError("set FAL_KEY (fal.ai -> Dashboard -> Keys)")
    import fal_client

    return fal_client


def dataset_dir(cfg: Config) -> Path:
    return cfg.path(cfg.datasets_dir) / cfg.character.id


async def portraits(cfg: Config, n: int = 8, client=None) -> list[Path]:
    client, ledger = _client(client), Ledger(cfg.path(cfg.workdir) / "ledger.db")
    out = dataset_dir(cfg) / "candidates"
    prompt = (
        f"photo of {cfg.character.look}, head and shoulders portrait, looking at the camera, neutral "
        "expression, plain light background, soft daylight, realistic skin texture, shot on a phone"
    )
    paths = []
    for i in range(n):
        res = await client.subscribe_async(
            PORTRAIT_MODEL,
            arguments={
                "prompt": prompt,
                "image_size": {"width": 768, "height": 1024},
                "seed": 1000 + i,
                "num_images": 1,
            },
        )
        paths.append(await _download(res["images"][0]["url"], out / f"candidate_{i:02d}.png"))
    ledger.add_setup("dataset", n * 0.768 * 1.024 * cfg.prices.portrait_per_mp, f"{n} portrait candidates")
    return paths


async def variations(cfg: Config, hero: Path, n: int = 20, client=None) -> list[Path]:
    client, ledger = _client(client), Ledger(cfg.path(cfg.workdir) / "ledger.db")
    out = dataset_dir(cfg) / "train"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(hero, out / "00_hero.png")
    url = await client.upload_file_async(hero)
    paths = [out / "00_hero.png"]
    for i, change in enumerate(VARIATIONS[:n], 1):
        res = await client.subscribe_async(
            EDIT_MODEL,
            arguments={
                "image_urls": [url],
                "seed": 2000 + i,
                "prompt": "Same person, same face, same hair, same skin details and eye colour - keep the identity exactly. "
                f"New photo: {change}. Realistic phone photo.",
            },
        )
        paths.append(await _download(res["images"][0]["url"], out / f"{i:02d}.png"))
    ledger.add_setup("dataset", len(paths[1:]) * 0.768 * 1.024 * cfg.prices.edit_per_mp, f"{n} variations")
    return paths


def check(cfg: Config, faces=None) -> dict:
    """How consistent is the training set? Saves the identity fingerprint and threshold."""
    from .faces import RealFaces

    faces = faces or RealFaces(cfg.path("models"))
    images = sorted(
        p for p in (dataset_dir(cfg) / "train").iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    e = enroll(faces, images)
    cfg.character_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.character_dir / "identity.npy", e.centroid)
    cfg.save_character_state(
        images=e.used, suggested_threshold=e.suggested_threshold, typical_similarity=e.typical_similarity
    )
    return {
        "used": e.used,
        "outliers": e.outliers,
        "no_face": e.no_face,
        "typical_similarity": round(e.typical_similarity, 3),
        "suggested_threshold": e.suggested_threshold,
    }


async def train(cfg: Config, steps: int = 2000, client=None) -> str:
    client, ledger = _client(client), Ledger(cfg.path(cfg.workdir) / "ledger.db")
    src = dataset_dir(cfg) / "train"
    with tempfile.TemporaryDirectory() as tmp:
        archive = shutil.make_archive(str(Path(tmp) / cfg.character.id), "zip", src)
        url = await client.upload_file_async(Path(archive))
    res = await client.subscribe_async(
        TRAIN_MODEL,
        arguments={"image_data_url": url, "steps": steps, "default_caption": f"photo of {cfg.character.trigger}"},
    )
    lora_url = res["diffusers_lora_file"]["url"]
    cost = steps / 1000 * cfg.prices.train_per_1000_steps
    ledger.add_setup("training", cost, f"{steps} steps")
    cfg.save_character_state(lora_url=lora_url, steps=steps, training_usd=cost)
    return lora_url
