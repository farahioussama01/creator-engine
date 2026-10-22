"""Measure what a render really costs on a given pod. The first run (cold model load) is reported
separately from the warm runs, because only warm runs say anything about steady-state $ per clip."""

from __future__ import annotations

import statistics
import tempfile
from pathlib import Path
from typing import Any

from .comfy import ComfyClient
from .config import EngineConfig, load_characters
from .engine import WAN_NEGATIVE
from .workflow import fill, load_template


async def run_bench(
    cfg: EngineConfig, endpoint: str, stage: str, profile_name: str, n: int, usd_per_hour: float
) -> dict[str, Any]:
    profile = cfg.profile(profile_name)
    ch = next(iter(load_characters(cfg).values()))
    kf_tpl = load_template(cfg.path(cfg.workflows["keyframe"]))
    vid_tpl = load_template(cfg.path(cfg.workflows["video"]))
    seconds: list[float] = []
    async with ComfyClient(endpoint) as c:
        tmp = Path(tempfile.mkdtemp(prefix="ce-bench-"))
        start_name = None
        for i in range(n + 1):  # +1: the first run pays the cold load
            if stage == "keyframe" or start_name is None:
                kf = fill(
                    kf_tpl,
                    {
                        "lora_name": ch.lora,
                        "lora_strength": ch.lora_strength,
                        "positive": f"photo of {ch.trigger}, {ch.look}, portrait, natural light",
                        "seed": 1000 + i,
                        "steps": profile.kf_steps,
                        "cfg": profile.kf_cfg,
                        "width": profile.kf_width,
                        "height": profile.kf_height,
                        "filename_prefix": "ce/bench/kf",
                    },
                )
                run = await c.run(kf)
                if stage == "keyframe":
                    seconds.append(run.exec_seconds)
                    continue
                path = await c.download(run.outputs[0], tmp)
                start_name = await c.upload_image(path)
            vid = fill(
                vid_tpl,
                {
                    "start_image": start_name,
                    "positive": "she smiles and turns slightly, handheld",
                    "negative": WAN_NEGATIVE,
                    "width": profile.vid_width,
                    "height": profile.vid_height,
                    "length": profile.frames,
                    "fps": profile.fps,
                    "seed": 2000 + i,
                    "steps": profile.vid_steps,
                    "cfg": profile.vid_cfg,
                    "split_step": profile.split_step,
                    "lightning_strength": 1.0 if profile.lightning else 0.0,
                    "filename_prefix": "ce/bench/clip",
                },
            )
            run = await c.run(vid, timeout=3600)
            seconds.append(run.exec_seconds)
        stats = await c.system_stats()
    warm = seconds[1:] if len(seconds) > 1 else seconds
    med = statistics.median(warm)
    return {
        "endpoint": endpoint,
        "device": [d.get("name") for d in stats.get("devices", [])],
        "stage": stage,
        "profile": profile.model_dump(),
        "cold_seconds": round(seconds[0], 2),
        "warm_seconds": [round(s, 2) for s in warm],
        "median_seconds": round(med, 2),
        "usd_per_render": round(med * usd_per_hour / 3600, 5),
        "renders_per_hour": round(3600 / med, 1),
    }
