"""Async client for the ComfyUI HTTP API.

Uses the routes ComfyUI itself serves: ``POST /prompt``, ``GET /history/{id}``, ``GET /view``,
``POST /upload/image``, ``GET /system_stats``, ``GET /object_info``, ``GET /models/{folder}``,
``POST /interrupt`` and ``POST /free``.

GPU time per prompt is read from the ``execution_start`` / ``execution_success`` timestamps that
ComfyUI records in the prompt's history, so the cost ledger bills what the GPU actually did rather
than the wall time we spent waiting (which includes queueing behind other prompts).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


class ComfyError(RuntimeError):
    pass


class ComfyValidationError(ComfyError):
    """The server rejected the graph. Retrying elsewhere will not help."""

    def __init__(self, message: str, node_errors: dict[str, Any] | None = None):
        super().__init__(message)
        self.node_errors = node_errors or {}


class ComfyExecutionError(ComfyError):
    """The prompt ran and failed (OOM, interrupted, bad input). ``exec_seconds`` is GPU time burned."""

    def __init__(self, message: str, exec_seconds: float = 0.0):
        super().__init__(message)
        self.exec_seconds = exec_seconds


@dataclass
class OutputFile:
    filename: str
    subfolder: str
    type: str
    node_id: str


@dataclass
class PromptRun:
    prompt_id: str
    outputs: list[OutputFile]
    exec_seconds: float
    wall_seconds: float
    queued_at: float
    history: dict[str, Any] = field(repr=False, default_factory=dict)


def _exec_seconds(history_entry: dict[str, Any]) -> float | None:
    msgs = (history_entry.get("status") or {}).get("messages") or []
    start = end = None
    for item in msgs:
        if not isinstance(item, list | tuple) or len(item) != 2:
            continue
        kind, data = item
        ts = (data or {}).get("timestamp")
        if ts is None:
            continue
        if kind == "execution_start":
            start = ts
        elif kind in ("execution_success", "execution_error", "execution_interrupted"):
            end = ts
    if start is None or end is None:
        return None
    return max(0.0, (end - start) / 1000.0)


def _collect_outputs(history_entry: dict[str, Any]) -> list[OutputFile]:
    files: list[OutputFile] = []
    for node_id, node_out in (history_entry.get("outputs") or {}).items():
        for _key, items in node_out.items():
            if not isinstance(items, list):
                continue
            for it in items:
                if isinstance(it, dict) and "filename" in it:
                    files.append(
                        OutputFile(
                            filename=it["filename"],
                            subfolder=it.get("subfolder", ""),
                            type=it.get("type", "output"),
                            node_id=node_id,
                        )
                    )
    return [f for f in files if f.type == "output"]


class ComfyClient:
    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        poll: float = 0.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.poll = poll
        self.client_id = uuid.uuid4().hex
        self._http = httpx.AsyncClient(
            base_url=self.base_url, headers=headers or {}, timeout=timeout, transport=transport
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> ComfyClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # -- read-only -------------------------------------------------------
    async def system_stats(self) -> dict[str, Any]:
        r = await self._http.get("/system_stats")
        r.raise_for_status()
        return r.json()

    async def object_info(self) -> dict[str, Any]:
        r = await self._http.get("/object_info")
        r.raise_for_status()
        return r.json()

    async def list_models(self, folder: str) -> list[str]:
        r = await self._http.get(f"/models/{folder}")
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return list(r.json())

    async def queue_depth(self) -> int:
        r = await self._http.get("/queue")
        r.raise_for_status()
        q = r.json()
        return len(q.get("queue_running", [])) + len(q.get("queue_pending", []))

    # -- actions ---------------------------------------------------------
    async def upload_image(self, path: Path, subfolder: str = "creator_engine") -> str:
        """Upload an input image; returns the value to put in a LoadImage node."""
        with open(path, "rb") as fh:
            files = {"image": (path.name, fh.read(), "image/png")}
        r = await self._http.post(
            "/upload/image", files=files, data={"overwrite": "true", "subfolder": subfolder, "type": "input"}
        )
        r.raise_for_status()
        body = r.json()
        return f"{body['subfolder']}/{body['name']}" if body.get("subfolder") else body["name"]

    async def queue_prompt(self, graph: dict[str, Any]) -> str:
        r = await self._http.post("/prompt", json={"prompt": graph, "client_id": self.client_id})
        if r.status_code == 400:
            body = r.json()
            err = body.get("error", {})
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise ComfyValidationError(f"prompt rejected: {msg}", body.get("node_errors"))
        r.raise_for_status()
        body = r.json()
        if body.get("node_errors"):
            raise ComfyValidationError("prompt rejected", body["node_errors"])
        return body["prompt_id"]

    async def wait(
        self, prompt_id: str, timeout: float = 1800.0, poll: float | None = None
    ) -> dict[str, Any]:
        """Poll /history until the prompt finishes. Returns the history entry."""
        poll = poll or self.poll
        deadline = time.monotonic() + timeout
        delay = poll / 4
        while True:
            r = await self._http.get(f"/history/{prompt_id}")
            r.raise_for_status()
            entry = r.json().get(prompt_id)
            if entry:
                status = entry.get("status") or {}
                if status.get("completed") or status.get("status_str") in ("success", "error"):
                    if status.get("status_str") == "error":
                        raise ComfyExecutionError(
                            _error_text(entry), exec_seconds=_exec_seconds(entry) or 0.0
                        )
                    return entry
            if time.monotonic() > deadline:
                raise TimeoutError(f"prompt {prompt_id} did not finish in {timeout:.0f}s")
            await asyncio.sleep(delay)
            delay = min(poll, delay * 2)

    async def download(self, out: OutputFile, dest_dir: Path, name: str | None = None) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        params = {"filename": out.filename, "subfolder": out.subfolder, "type": out.type}
        target = dest_dir / (name or out.filename)
        async with self._http.stream("GET", "/view", params=params) as r:
            r.raise_for_status()
            with open(target, "wb") as fh:
                async for chunk in r.aiter_bytes():
                    fh.write(chunk)
        return target

    async def interrupt(self) -> None:
        await self._http.post("/interrupt")

    async def free(self, unload_models: bool = True) -> None:
        await self._http.post("/free", json={"unload_models": unload_models, "free_memory": True})

    async def run(self, graph: dict[str, Any], timeout: float = 1800.0) -> PromptRun:
        t0 = time.monotonic()
        queued_at = time.time()
        prompt_id = await self.queue_prompt(graph)
        entry = await self.wait(prompt_id, timeout=timeout)
        wall = time.monotonic() - t0
        exec_s = _exec_seconds(entry)
        return PromptRun(
            prompt_id=prompt_id,
            outputs=_collect_outputs(entry),
            exec_seconds=exec_s if exec_s is not None else wall,
            wall_seconds=wall,
            queued_at=queued_at,
            history=entry,
        )


def _error_text(entry: dict[str, Any]) -> str:
    for kind, data in (entry.get("status") or {}).get("messages") or []:
        if kind == "execution_error":
            return f"{data.get('node_type')}: {data.get('exception_message', '').strip()}"
    return "execution failed"
