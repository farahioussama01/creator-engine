"""MCP server so Claude Code can run batches, read the numbers and look at rejects.

Registered in .mcp.json. Five tools, each a thin wrapper over the same functions the CLI uses.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .config import load
from .ledger import Ledger


def build_server(config: str | None = None) -> MCPServer:
    cfg = load(config or os.environ.get("CE_CONFIG", "configs/config.yaml"))
    ledger = Ledger(cfg.path(cfg.workdir) / "ledger.db")
    server = MCPServer(
        "creator-engine",
        instructions="Render, check and cost AI-creator clips. "
        "run_batch -> batch_status -> numbers -> pieces(status='rejected') to look at failures.",
    )
    running: dict[str, asyncio.Task] = {}

    @server.tool()
    async def run_batch(n: int = 12, mix: dict[str, int] | None = None) -> dict[str, Any]:
        """Start a batch of n pieces in the background (format mix optional). Returns immediately."""
        from .pipeline import run_batch as _run

        before = ledger.latest_batch()
        task = asyncio.create_task(_run(cfg, n, mix, log=lambda *_: None))
        await asyncio.sleep(0.5)
        batch = ledger.latest_batch()
        if batch == before:
            if task.done() and task.exception():
                raise ToolError(str(task.exception()))
            await asyncio.sleep(1)
            batch = ledger.latest_batch()
        running[batch] = task
        return {"batch": batch, "state": "running"}

    @server.tool()
    def batch_status(batch: str) -> dict[str, Any]:
        """Pieces by status, and whether the batch is still running."""
        rows = ledger.q("SELECT status, COUNT(*) n FROM pieces WHERE batch = ? GROUP BY status", batch)
        task = running.get(batch)
        return {"pieces": {r["status"]: r["n"] for r in rows}, "running": bool(task and not task.done())}

    @server.tool()
    def numbers(batch: str | None = None) -> dict[str, Any]:
        """Pipeline, volume, cost per piece and views for a batch (latest by default)."""
        from .report import answer, batch_numbers

        batch = batch or ledger.latest_batch()
        if not batch:
            raise ToolError("no batches yet")
        return {"answer": answer(cfg, ledger, batch), **batch_numbers(ledger, batch)}

    @server.tool()
    def pieces(batch: str, status: str = "rejected", limit: int = 20) -> list[dict[str, Any]]:
        """Pieces with their files and the reason they were rejected. Read the image paths to judge the checks."""
        rows = ledger.q("SELECT * FROM pieces WHERE batch = ? AND status = ? LIMIT ?", batch, status, limit)
        return [{k: r[k] for k in ("id", "format", "status", "reason", "identity", "keyframe", "clip")} for r in rows]

    @server.tool()
    def next_mix(n: int = 24) -> dict[str, int]:
        """Format mix for the next batch, from the views imported so far."""
        from .views import next_mix as _next

        return _next(ledger, list(cfg.formats), n)

    return server


def main() -> None:
    build_server().run("stdio")


if __name__ == "__main__":
    main()
