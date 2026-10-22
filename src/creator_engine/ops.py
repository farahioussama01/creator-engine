"""Operations shared by the CLI, the MCP server and the scheduled daily run."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .comfy import ComfyClient
from .config import EngineConfig, load_characters, load_config, load_formats
from .engine import Engine
from .experiments import allocate, arm_stats
from .ledger import build_ledger
from .planner import even_quotas, plan_batch
from .pool import Pool
from .publish import publish_due, schedule_batch
from .store import Store
from .workflow import load_template, required_models, validate_against_object_info

DEFAULT_CONFIG = os.environ.get("CE_CONFIG", "configs/engine.yaml")


@dataclass
class Ctx:
    cfg: EngineConfig
    store: Store


def open_ctx(config: str | Path = DEFAULT_CONFIG) -> Ctx:
    cfg = load_config(config)
    return Ctx(cfg, Store(cfg.path(cfg.db)))


def make_engine(ctx: Ctx) -> Engine:
    from .vision import make_analyzer
    from .vision.ocr import try_ocr

    judge = None
    if ctx.cfg.judge.enabled:
        from .judge import ClaudeJudge

        judge = ClaudeJudge(ctx.cfg.judge)
    analyzer = make_analyzer(ctx.cfg.analyzer, str(ctx.cfg.path(ctx.cfg.models_dir)))
    return Engine(ctx.cfg, ctx.store, analyzer, ocr=try_ocr(), judge=judge)


def eligible_formats(cfg: EngineConfig, character: str) -> list[str]:
    return [f for f, s in load_formats(cfg).items() if not s.for_characters or character in s.for_characters]


def plan(
    ctx: Ctx,
    character: str,
    n: int = 24,
    profile: str = "standard",
    quotas: dict[str, int] | None = None,
    simulated: bool | None = None,
) -> dict[str, Any]:
    chars = load_characters(ctx.cfg)
    if character not in chars:
        raise KeyError(f"unknown character {character!r}; known: {sorted(chars)}")
    formats = load_formats(ctx.cfg)
    source = "manual"
    if not quotas:
        stats = arm_stats(ctx.cfg, ctx.store, character, formats)
        measured = sum(s.posts for s in stats)
        if measured:
            quotas, source = allocate(stats, n), "experiments"
        else:
            quotas, source = (
                even_quotas(eligible_formats(ctx.cfg, character), n),
                "even (no measured posts yet)",
            )
    sim = ctx.cfg.analyzer == "sim" if simulated is None else simulated
    bid = plan_batch(ctx.cfg, ctx.store, chars[character], formats, quotas, profile, simulated=sim)
    return {"batch": bid, "quotas": quotas, "quota_source": source, "profile": profile}


async def run(ctx: Ctx, batch_id: str, provision: bool = False) -> dict[str, Any]:
    engine = make_engine(ctx)
    if not provision:
        return await engine.run_batch(batch_id)
    if ctx.cfg.runpod is None:
        raise ValueError("no `runpod:` section in the config")
    from .runpod import rented_pods

    def bill(pod, seconds, usd):
        ctx.store.add_spend(batch_id, "pod_session", pod.id, seconds, usd, provider="runpod")

    async with rented_pods(ctx.cfg.runpod, batch_id[-12:], on_bill=bill) as endpoints:
        pool = Pool(endpoints, poll_seconds=ctx.cfg.poll_seconds)
        await pool.start()
        try:
            return await engine.run_batch(batch_id, pool=pool)
        finally:
            await pool.close()  # pod lifetimes are billed by the provider hook, not the pool


def report(ctx: Ctx, batch_id: str | None = None) -> dict[str, Any]:
    batch_id = batch_id or (ctx.store.latest_batch() or {"id": None})["id"]
    if not batch_id:
        raise ValueError("no batches yet")
    return build_ledger(ctx.cfg, ctx.store, batch_id).to_dict()


def report_text(led: dict[str, Any]) -> str:
    lines = [
        f"Batch {led['batch_id']} ({led['character']}, {led['profile']}{', SIMULATED GPU' if led['simulated'] else ''})",
        f"  keepers {led['keepers']}/{led['pieces']}  total ${led['total_usd']:.4f}  "
        "per keeper " + (f"${led['per_keeper']:.4f}" if led["per_keeper"] else "-"),
    ]
    for k, v in sorted(led["buckets"].items(), key=lambda kv: -kv[1]):
        if v > 0:
            lines.append(f"    {k:22s} ${v:.4f}")
    if led.get("levers"):
        lines.append("  to halve it:")
        for lv in led["levers"]:
            lines.append(f"    -{lv['saving_pct']:.0%}  {lv['name']}")
        if led.get("combined"):
            lines.append(
                f"    -{led['combined']['saving_pct']:.0%}  all together -> ${led['combined']['per_keeper']:.4f}/keeper"
            )
    return "\n".join(lines)


async def check_endpoint(cfg: EngineConfig, url: str) -> dict[str, Any]:
    """Preflight: node classes, required inputs and model files for every template, before spending GPU time."""
    out: dict[str, Any] = {"url": url, "templates": {}}
    async with ComfyClient(url) as c:
        stats = await c.system_stats()
        out["devices"] = [d.get("name") for d in stats.get("devices", [])]
        info = await c.object_info()
        chars = load_characters(cfg)
        for name, path in cfg.workflows.items():
            graph = load_template(cfg.path(path))
            problems = validate_against_object_info(graph, info)
            need = required_models(graph)
            if name == "keyframe":
                need.setdefault("loras", set()).update(ch.lora for ch in chars.values())
            for folder, files in need.items():
                have = set(await c.list_models(folder))
                problems += [f"missing model {folder}/{f}" for f in sorted(files - have)]
            out["templates"][name] = problems
    out["ok"] = not any(out["templates"].values())
    return out


async def status(ctx: Ctx) -> dict[str, Any]:
    async def ping(ep):
        try:
            async with ComfyClient(ep.url, headers=ep.headers, timeout=5) as c:
                t0 = time.monotonic()
                await c.system_stats()
                depth = await c.queue_depth()
                return {
                    "endpoint": ep.name,
                    "ok": True,
                    "queue": depth,
                    "ms": round((time.monotonic() - t0) * 1000),
                }
        except (httpx.HTTPError, OSError) as exc:
            return {"endpoint": ep.name, "ok": False, "error": exc.__class__.__name__}

    eps = await asyncio.gather(*(ping(e) for e in ctx.cfg.endpoints))
    latest = ctx.store.latest_batch()
    return {
        "endpoints": eps,
        "spend_last_24h_usd": round(ctx.store.spend_today(), 4),
        "budget": ctx.cfg.budget.model_dump(),
        "latest_batch": dict(latest) if latest else None,
        "scheduled_posts": ctx.store.one("SELECT COUNT(*) n FROM posts WHERE status = 'scheduled'")["n"],
    }


async def daily(
    ctx: Ctx,
    character: str,
    n: int = 24,
    profile: str = "standard",
    provision: bool = False,
    publish: bool = True,
    webhook: str | None = None,
) -> dict[str, Any]:
    """The unattended loop: allocate -> plan -> render -> gate -> schedule -> publish due -> report."""
    from .dashboard import render as render_dashboard

    planned = plan(ctx, character, n, profile)
    result = await run(ctx, planned["batch"], provision=provision)
    led = report(ctx, planned["batch"])
    scheduled = schedule_batch(ctx.cfg, ctx.store, planned["batch"])
    published = publish_due(ctx.cfg, ctx.store) if publish else []
    page = render_dashboard(ctx.cfg, ctx.store, planned["batch"])
    summary = {
        "batch": planned["batch"],
        "quotas": planned["quotas"],
        "result": result,
        "per_keeper_usd": led["per_keeper"],
        "total_usd": led["total_usd"],
        "scheduled": len(scheduled),
        "published": len([p for p in published if p["status"] != "failed"]),
        "report": str(page),
    }
    if webhook:
        text = (
            f"creator-engine {character}: {led['keepers']}/{led['pieces']} keepers, "
            f"${led['total_usd']:.2f} total, ${(led['per_keeper'] or 0):.3f}/keeper, "
            f"{len(scheduled)} posts scheduled. Report: {page}"
        )
        try:
            httpx.post(webhook, json={"text": text, "content": text}, timeout=10)  # Slack + Discord shapes
        except httpx.HTTPError as exc:
            summary["webhook_error"] = str(exc)
    (ctx.cfg.path(ctx.cfg.workdir) / "last_daily.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
