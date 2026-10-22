"""MCP server: lets Claude Code operate the engine.

Registered for the repo in ``.mcp.json``. Tools are thin wrappers over ``ops`` so Claude and cron run
the same code. Long renders run in the background; ``batch_status`` polls them.

Tools
  engine_status        endpoints, queue depth, spend in the last 24 h, budget
  preflight            check templates + model files against a ComfyUI endpoint
  plan_batch           plan a batch (quotas from the experiments unless given)
  run_batch            start rendering a planned batch in the background
  batch_status         progress of a running batch
  batch_report         cost per keeper, spend buckets, halving plan, gate funnel
  list_pieces          keepers or rejects with file paths and reasons (Read the images to review them)
  format_experiments   per-format results vs account baselines and next allocation
  schedule_posts       schedule a batch's keepers across accounts
  render_dashboard     write the HTML report and return its path
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import os
from collections import Counter
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import ops
from .config import load_formats

INSTRUCTIONS = """Creator engine: renders photoreal AI creator content (character-LoRA keyframes + Wan 2.2 clips)
on rented GPUs, gates it for quality, tracks cost per keeper, and learns which formats travel.
Typical loop: engine_status -> plan_batch -> run_batch -> batch_status (poll) -> batch_report ->
list_pieces(status='rejected') to review failures -> schedule_posts. Never publish live without the
user's go-ahead; publishing is dry-run unless the config says otherwise."""


def build_server(config: str | None = None) -> MCPServer:
    config = config or os.environ.get("CE_CONFIG", ops.DEFAULT_CONFIG)
    ctx = ops.open_ctx(config)
    server = MCPServer("creator-engine", instructions=INSTRUCTIONS)
    running: dict[str, asyncio.Task] = {}

    def friendly(fn):
        """Expected problems (unknown batch, no data yet, bad quota) become tool errors with a clear message."""
        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*a, **kw):
                try:
                    return await fn(*a, **kw)
                except (ValueError, KeyError, FileNotFoundError) as exc:
                    raise ToolError(str(exc).strip("'\"")) from exc

            return awrapper

        @functools.wraps(fn)
        def wrapper(*a, **kw):
            try:
                return fn(*a, **kw)
            except (ValueError, KeyError, FileNotFoundError) as exc:
                raise ToolError(str(exc).strip("'\"")) from exc

        return wrapper

    @server.tool()
    @friendly
    async def engine_status() -> dict[str, Any]:
        """GPU endpoint health and queue depth, spend in the last 24 hours, budgets, latest batch."""
        return await ops.status(ctx)

    @server.tool()
    @friendly
    async def preflight(endpoint_url: str) -> dict[str, Any]:
        """Validate every workflow template against a ComfyUI server: node classes, required inputs and model
        files. Run this before the first batch on a new pod."""
        return await ops.check_endpoint(ctx.cfg, endpoint_url)

    @server.tool()
    @friendly
    def plan_batch(
        character: str, n: int = 24, profile: str = "standard", quotas: dict[str, int] | None = None
    ) -> dict[str, Any]:
        """Plan a batch of n pieces for a character. Without quotas, formats are allocated from the view
        experiments (even split until posts have been measured). Profiles: draft, standard, hero, quality20."""
        return ops.plan(ctx, character, n, profile, quotas)

    @server.tool()
    @friendly
    async def run_batch(batch_id: str) -> dict[str, Any]:
        """Start rendering + gating a planned batch in the background. Poll batch_status."""
        if batch_id in running and not running[batch_id].done():
            return {"batch": batch_id, "state": "already running"}
        running[batch_id] = asyncio.create_task(ops.run(ctx, batch_id))
        return {"batch": batch_id, "state": "started"}

    @server.tool()
    @friendly
    def batch_status(batch_id: str) -> dict[str, Any]:
        """Piece counts by status and whether the background run finished."""
        task = running.get(batch_id)
        counts = Counter(p["status"] for p in ctx.store.pieces(batch_id))
        out: dict[str, Any] = {
            "batch": batch_id,
            "pieces": dict(counts),
            "batch_status": ctx.store.batch(batch_id)["status"],
        }
        if task is not None:
            out["running"] = not task.done()
            if task.done() and task.exception():
                out["error"] = repr(task.exception())
        return out

    @server.tool()
    @friendly
    def batch_report(batch_id: str | None = None) -> dict[str, Any]:
        """Cost per keeper, spend by bucket (what became a keeper vs rejects/idle/swaps), counterfactual levers
        that would cut cost, gate funnel and stage pass rates. Defaults to the latest batch."""
        led = ops.report(ctx, batch_id)
        led["text"] = ops.report_text(led)
        return led

    @server.tool()
    @friendly
    def list_pieces(batch_id: str, status: str = "keeper", limit: int = 20) -> list[dict[str, Any]]:
        """Pieces of a batch with file paths and reject reasons. status: keeper, rejected (any rejection),
        rejected_keyframe, rejected_video, skipped_budget."""
        rows = ctx.store.pieces(batch_id)
        if status == "rejected":
            rows = [r for r in rows if r["status"].startswith("rejected")]
        else:
            rows = [r for r in rows if r["status"] == status]
        return [
            {
                "id": r["id"],
                "format": r["format"],
                "status": r["status"],
                "identity": r["identity"],
                "reason": r["reject_reason"],
                "keyframe": r["keyframe_path"],
                "clip": r["video_path"],
                "prompt": r["keyframe_prompt"],
            }
            for r in rows[:limit]
        ]

    @server.tool()
    @friendly
    def format_experiments(character: str, n: int = 24) -> list[dict[str, Any]]:
        """Per-format posts, wins vs account baseline, median lift, P(best), status (explore/keep/scale/kill)
        and the quota each format would get in the next batch of n."""
        from .experiments import allocate, arm_stats

        stats = arm_stats(ctx.cfg, ctx.store, character, load_formats(ctx.cfg))
        allocate(stats, n)
        return [s.__dict__ for s in stats]

    @server.tool()
    @friendly
    def schedule_posts(batch_id: str, horizon_days: int = 7) -> list[dict[str, Any]]:
        """Schedule the batch's keepers across the character's accounts (windows, caps, gaps). Captions get the
        AI disclosure automatically. Does not publish."""
        from .publish import schedule_batch

        return [
            {k: p[k] for k in ("id", "account", "format", "scheduled_at", "caption")}
            for p in schedule_batch(ctx.cfg, ctx.store, batch_id, horizon_days=horizon_days)
        ]

    @server.tool()
    @friendly
    def render_dashboard(batch_id: str | None = None) -> dict[str, str]:
        """Write the HTML report for a batch and return its path."""
        from .dashboard import render

        return {"path": str(render(ctx.cfg, ctx.store, batch_id))}

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="creator-engine MCP server (stdio)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    build_server(args.config).run("stdio")


if __name__ == "__main__":
    main()
