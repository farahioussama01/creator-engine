"""Turn (character, format quotas, render profile) into concrete pieces with prompts and seeds.

Variation is combinatorial (scene x outfit x light x hook) and deterministic: the same batch id
always produces the same prompts and seeds, so a crashed batch resumes exactly where it stopped.
"""

from __future__ import annotations

import json
import random
import time
import zlib

from .config import CharacterSpec, EngineConfig, FormatSpec
from .store import Store


def new_batch_id(character: str) -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{character}"


def _seed(*parts: object) -> int:
    return zlib.crc32(":".join(map(str, parts)).encode()) & 0x7FFFFFFF


def plan_batch(
    cfg: EngineConfig,
    store: Store,
    character: CharacterSpec,
    formats: dict[str, FormatSpec],
    quotas: dict[str, int],
    profile: str,
    batch_id: str | None = None,
    simulated: bool = False,
) -> str:
    cfg.profile(profile)  # validate early
    unknown = set(quotas) - set(formats)
    if unknown:
        raise KeyError(f"unknown formats: {sorted(unknown)}")
    batch_id = batch_id or new_batch_id(character.id)
    store.add_batch(batch_id, character.id, profile, {"quotas": quotas}, simulated)
    for fmt_id, n in quotas.items():
        fmt = formats[fmt_id]
        for i in range(n):
            pid = f"{batch_id}-{fmt_id}-{i:03d}"
            rng = random.Random(_seed(batch_id, fmt_id, i))
            v = {
                "scene": rng.choice(fmt.scenes),
                "outfit": rng.choice(fmt.outfits),
                "light": rng.choice(fmt.lights),
                "hook": rng.choice(fmt.hooks),
            }
            kf_prompt = fmt.render(fmt.keyframe_prompt, trigger=character.trigger, look=character.look, **v)
            motion = fmt.render(fmt.motion_prompt, **v)
            caption = fmt.caption.format(**v).strip()
            store.insert(
                "pieces",
                {
                    "id": pid,
                    "batch_id": batch_id,
                    "character": character.id,
                    "format": fmt_id,
                    "variant_json": json.dumps(v),
                    "keyframe_prompt": kf_prompt,
                    "motion_prompt": motion,
                    "caption": caption,
                    "kf_seed": _seed(pid, "kf"),
                    "vid_seed": _seed(pid, "vid"),
                    "status": "planned",
                    "keyframe_path": None,
                    "video_path": None,
                    "identity": None,
                    "reject_reason": None,
                    "updated_at": time.time(),
                },
            )
    return batch_id


def even_quotas(formats: list[str], n: int) -> dict[str, int]:
    base, extra = divmod(n, len(formats))
    return {f: base + (1 if i < extra else 0) for i, f in enumerate(formats)}
