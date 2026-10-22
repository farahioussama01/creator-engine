"""The whole loop on a laptop: simulated GPUs, real engine.

Starts two simulated RTX 4090 pods (ComfyUI-compatible HTTP servers), enrolls the character's
identity, then runs several rounds of: allocate formats from view data -> plan -> render keyframes
-> gate -> render clips -> gate -> schedule across accounts -> publish (dry run) -> collect
(simulated) 48h metrics. The experiment loop should find the format with the best hidden appeal and
shift volume to it. Ends by writing the HTML report for the last batch.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from .config import EndpointSpec, load_characters, load_formats, load_prices
from .experiments import arm_stats, simulate_metrics
from .identity import enroll, list_images, make_sim_references, save
from .ops import Ctx, make_engine, plan, report, report_text
from .publish import publish_due, schedule_batch
from .sim.server import SimComfyServer
from .store import Store

HIDDEN_APPEAL = {"mirror-walk": 1.45, "golden-street": 1.0, "cafe-pov": 0.65}
ACCOUNT_SIZE = {"mila.wanders": 6000.0, "mila.slowdays": 2200.0}


async def run_demo(
    cfg,
    character: str = "mila",
    pieces: int = 24,
    rounds: int = 3,
    gpus: int = 2,
    time_scale: float | None = None,
    fresh: bool = True,
    log=print,
    workdir: Path | None = None,
) -> dict[str, Any]:
    from .dashboard import render as render_dashboard

    cfg.workdir = workdir or Path("runs/demo")
    cfg.db = cfg.workdir / "engine.db"
    cfg.analyzer = "sim"
    cfg.publish.dry_run = True
    ts = time_scale or cfg.sim.time_scale
    cfg.poll_seconds = max(0.005, 25 * ts)  # 0.5 s of real polling, expressed in simulated time
    work = cfg.path(cfg.workdir)
    if fresh and work.exists():
        shutil.rmtree(work)
    chars = load_characters(cfg)
    speed = load_prices(cfg).get("sim_gpu_speed", {}).get("rtx4090", 1.0)
    servers = [
        SimComfyServer(
            name=f"sim-4090-{i}",
            gpu="rtx4090",
            gpu_speed=speed,
            time_scale=ts,
            defect_rates=cfg.sim.defect_rates,
            extra_loras=[c.lora for c in chars.values()],
            seed=i,
        )
        for i in range(gpus)
    ]
    urls = [await s.start() for s in servers]
    cfg.endpoints = [
        EndpointSpec(name=f"sim-4090-{chr(97 + i)}", url=u, gpu="rtx4090", usd_per_hour=0.34)
        for i, u in enumerate(urls)
    ]
    log(f"started {gpus} simulated RTX 4090 pods at $0.34/h: {', '.join(urls)}")
    ctx = Ctx(cfg, Store(cfg.path(cfg.db)))
    try:
        from .vision import make_analyzer

        analyzer = make_analyzer("sim")
        ch = chars[character]
        refs = make_sim_references(ch, work / "references" / character)
        centroid, rep = enroll(analyzer, character, list_images(refs[0].parent))
        save(cfg, centroid, rep)
        log(
            f"enrolled {character}: {rep.n_used} reference faces, leave-one-out cos p10 {rep.loo_p10:.2f}, "
            f"suggested identity gate {rep.suggested_threshold:.2f}"
        )

        rounds_out = []
        for r in range(1, rounds + 1):
            t0 = time.time()
            planned = plan(ctx, character, pieces, "standard", simulated=True)
            log(
                f"\nround {r}: batch {planned['batch']} quotas {planned['quotas']} ({planned['quota_source']})"
            )
            engine = make_engine(ctx)
            result = await engine.run_batch(planned["batch"])
            led = report(ctx, planned["batch"])
            log(report_text(led))
            posts = schedule_batch(cfg, ctx.store, planned["batch"], horizon_days=60)
            published = publish_due(cfg, ctx.store, now=time.time() + 90 * 86400)
            n_metrics = simulate_metrics(cfg, ctx.store, HIDDEN_APPEAL, seed=r, account_base=ACCOUNT_SIZE)
            stats = arm_stats(cfg, ctx.store, character, load_formats(cfg))
            log(
                f"  scheduled {len(posts)} posts, published {len(published)} (dry run), measured {n_metrics}; "
                + ", ".join(f"{s.format}: P(best) {s.p_best:.0%} [{s.status}]" for s in stats)
                + f"  ({time.time() - t0:.0f}s)"
            )
            rounds_out.append(
                {
                    "batch": planned["batch"],
                    "quotas": planned["quotas"],
                    "result": result,
                    "per_keeper": led["per_keeper"],
                    "keepers": led["keepers"],
                }
            )
        page = render_dashboard(cfg, ctx.store, rounds_out[-1]["batch"])
        log(f"\nreport: {page}")
        return {"rounds": rounds_out, "report": str(page)}
    finally:
        for s in servers:
            await s.stop()
        ctx.store.close()
