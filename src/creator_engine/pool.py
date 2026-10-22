"""Dispatch prompts across rented GPUs.

* Least-loaded routing with ``concurrency`` prompts in flight per GPU (2 keeps a GPU busy while the
  previous output downloads and gets gated, which is most of the idle-time lever).
* Stage affinity: an endpoint can be dedicated to keyframes or video so models never swap.
* Retries on another endpoint for OOMs, timeouts and dropped connections; an endpoint that fails
  three times in a row (a reclaimed spot pod, usually) is taken out of rotation.
* Session billing: what the pod cost while the pool held it, so idle GPU time is visible.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .comfy import ComfyClient, ComfyExecutionError, ComfyValidationError
from .config import EndpointSpec


class NoEndpointError(RuntimeError):
    pass


@dataclass
class ExecRequest:
    stage: str
    build: Callable[[dict[str, str]], dict[str, Any]]
    dest_dir: Path
    dest_name: str
    uploads: dict[str, Path] = field(default_factory=dict)
    timeout: float = 1800.0


@dataclass
class Failure:
    endpoint: str
    gpu: str
    usd_per_hour: float
    gpu_seconds: float
    error: str


@dataclass
class ExecOutcome:
    ok: bool
    endpoint: str = ""
    gpu: str = ""
    usd_per_hour: float = 0.0
    prompt_id: str | None = None
    gpu_seconds: float = 0.0
    wall_seconds: float = 0.0
    output: Path | None = None
    sim_truth: dict[str, Any] | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str | None = None
    failures: list[Failure] = field(default_factory=list)

    @property
    def usd(self) -> float:
        return self.gpu_seconds * self.usd_per_hour / 3600


@dataclass
class SessionBill:
    endpoint: str
    gpu: str
    seconds: float
    usd: float
    busy_seconds: float
    executed: int
    simulated_clock: bool


class _Endpoint:
    def __init__(self, spec: EndpointSpec, client: ComfyClient):
        self.spec = spec
        self.client = client
        self.healthy = False
        self.inflight = 0
        self.executed = 0
        self.busy = 0.0
        self.consecutive_fail = 0
        self.session_start: float | None = None
        self.sim_clock = False

    async def clock(self) -> float:
        stats = await self.client.system_stats()
        sim = stats.get("sim")
        if sim:
            self.sim_clock = True
            return float(sim["uptime_seconds"])
        return time.monotonic()


class Pool:
    def __init__(
        self,
        specs: list[EndpointSpec],
        client_factory: Callable[[EndpointSpec], ComfyClient] | None = None,
        max_tries: int = 3,
        poll_seconds: float = 0.5,
    ):
        factory = client_factory or (lambda s: ComfyClient(s.url, headers=s.headers, poll=poll_seconds))
        self._eps = [_Endpoint(s, factory(s)) for s in specs]
        self._cond = asyncio.Condition()
        self.max_tries = max_tries
        self.closed = False

    @property
    def endpoints(self) -> list[_Endpoint]:
        return self._eps

    async def start(self) -> list[str]:
        """Health-check every endpoint; returns problems (unreachable endpoints)."""
        problems = []

        async def check(ep: _Endpoint) -> None:
            try:
                ep.session_start = await ep.clock()
                ep.healthy = True
            except (httpx.HTTPError, OSError) as exc:
                problems.append(f"{ep.spec.name}: unreachable ({exc.__class__.__name__})")

        await asyncio.gather(*(check(e) for e in self._eps))
        if not any(e.healthy for e in self._eps):
            raise NoEndpointError("no GPU endpoint is reachable: " + "; ".join(problems))
        return problems

    async def close(self) -> list[SessionBill]:
        bills = []
        for ep in self._eps:
            if ep.session_start is not None:
                try:
                    end = await ep.clock()
                except (httpx.HTTPError, OSError):
                    end = ep.session_start + ep.busy  # pod is gone; bill at least what it executed
                seconds = max(end - ep.session_start, ep.busy)
                bills.append(
                    SessionBill(
                        ep.spec.name,
                        ep.spec.gpu,
                        seconds,
                        seconds * ep.spec.usd_per_hour / 3600,
                        ep.busy,
                        ep.executed,
                        ep.sim_clock,
                    )
                )
            await ep.client.aclose()
        self.closed = True
        return bills

    async def _acquire(self, stage: str) -> _Endpoint:
        async with self._cond:
            while True:
                eligible = [e for e in self._eps if e.healthy and stage in e.spec.stages]
                if not eligible:
                    raise NoEndpointError(f"no healthy endpoint serves stage {stage!r}")
                free = [e for e in eligible if e.inflight < e.spec.concurrency]
                if free:
                    ep = min(free, key=lambda e: (e.inflight / e.spec.concurrency, e.busy))
                    ep.inflight += 1
                    return ep
                await self._cond.wait()

    async def _release(self, ep: _Endpoint) -> None:
        async with self._cond:
            ep.inflight -= 1
            self._cond.notify_all()

    async def execute(self, req: ExecRequest) -> ExecOutcome:
        failures: list[Failure] = []
        for _ in range(self.max_tries):
            try:
                ep = await self._acquire(req.stage)
            except NoEndpointError as exc:
                return ExecOutcome(ok=False, error=str(exc), failures=failures)
            started = time.time()
            try:
                names = {k: await ep.client.upload_image(p) for k, p in req.uploads.items()}
                graph = req.build(names)
                run = await ep.client.run(graph, timeout=req.timeout)
                if not run.outputs:
                    raise ComfyExecutionError(
                        "prompt produced no output files", exec_seconds=run.exec_seconds
                    )
                first = run.outputs[0]
                path = await ep.client.download(
                    first, req.dest_dir, req.dest_name + Path(first.filename).suffix
                )
                ep.busy += run.exec_seconds
                ep.executed += 1
                ep.consecutive_fail = 0
                return ExecOutcome(
                    ok=True,
                    endpoint=ep.spec.name,
                    gpu=ep.spec.gpu,
                    usd_per_hour=ep.spec.usd_per_hour,
                    prompt_id=run.prompt_id,
                    gpu_seconds=run.exec_seconds,
                    wall_seconds=run.wall_seconds,
                    output=path,
                    sim_truth=run.history.get("sim_truth"),
                    started_at=started,
                    finished_at=time.time(),
                    failures=failures,
                )
            except ComfyValidationError:
                raise
            except (ComfyExecutionError, TimeoutError, httpx.HTTPError, OSError) as exc:
                spent = getattr(exc, "exec_seconds", 0.0) or 0.0
                ep.busy += spent
                failures.append(
                    Failure(
                        ep.spec.name,
                        ep.spec.gpu,
                        ep.spec.usd_per_hour,
                        spent,
                        f"{exc.__class__.__name__}: {exc}",
                    )
                )
                ep.consecutive_fail += 1
                if ep.consecutive_fail >= 3:
                    ep.healthy = False
            finally:
                await self._release(ep)
        return ExecOutcome(ok=False, error=failures[-1].error if failures else "failed", failures=failures)
