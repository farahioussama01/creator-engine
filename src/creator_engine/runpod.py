"""Rent pods for a run, wait for ComfyUI, and always tear them down.

Idle pods are the easiest money to lose in this business, so pod lifetime is owned by a context
manager: pods are terminated in ``finally`` even if the batch crashes, and every pod's billed
lifetime is written to the ledger.

REST API: ``POST /v1/pods`` (create), ``GET /v1/pods/{id}``, ``DELETE /v1/pods/{id}`` (terminate).
ComfyUI is reached through RunPod's HTTP proxy at ``https://{pod_id}-{port}.proxy.runpod.net``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from .config import EndpointSpec, RunPodSpec

API = "https://rest.runpod.io/v1"


class RunPodError(RuntimeError):
    pass


@dataclass
class Pod:
    id: str
    name: str
    usd_per_hour: float
    created: float
    url: str


class RunPodClient:
    def __init__(self, api_key: str, transport: httpx.AsyncBaseTransport | None = None):
        self.http = httpx.AsyncClient(
            base_url=API, headers={"Authorization": f"Bearer {api_key}"}, timeout=60, transport=transport
        )

    async def aclose(self) -> None:
        await self.http.aclose()

    async def create_pod(self, name: str, spec: RunPodSpec) -> dict[str, Any]:
        body: dict[str, Any] = {
            "name": name,
            "imageName": spec.image,
            "gpuTypeIds": spec.gpu_type_ids,
            "gpuCount": 1,
            "cloudType": spec.cloud_type,
            "containerDiskInGb": spec.container_disk_gb,
            "ports": [f"{spec.port}/http"],
            "env": spec.env,
            "interruptible": spec.interruptible,
        }
        if spec.network_volume_id:
            body["networkVolumeId"] = spec.network_volume_id
        r = await self.http.post("/pods", json=body)
        if r.status_code >= 400:
            raise RunPodError(f"create pod failed: {r.status_code} {r.text[:300]}")
        return r.json()

    async def get_pod(self, pod_id: str) -> dict[str, Any]:
        r = await self.http.get(f"/pods/{pod_id}")
        r.raise_for_status()
        return r.json()

    async def terminate(self, pod_id: str) -> None:
        r = await self.http.delete(f"/pods/{pod_id}")
        if r.status_code not in (200, 202, 204, 404):
            raise RunPodError(f"terminate {pod_id} failed: {r.status_code} {r.text[:200]}")


def proxy_url(pod_id: str, port: int) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net"


async def wait_for_comfy(
    url: str, timeout: float, transport: httpx.AsyncBaseTransport | None = None, poll: float = 5.0
) -> None:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=15, transport=transport) as http:
        while time.monotonic() < deadline:
            try:
                r = await http.get(f"{url}/system_stats")
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(poll)
    raise TimeoutError(f"ComfyUI at {url} did not come up within {timeout:.0f}s")


@contextlib.asynccontextmanager
async def rented_pods(
    spec: RunPodSpec,
    label: str,
    on_bill=None,
    client: RunPodClient | None = None,
    health_transport: httpx.AsyncBaseTransport | None = None,
    poll: float = 5.0,
) -> AsyncIterator[list[EndpointSpec]]:
    """Yield ready ComfyUI endpoints; terminate every pod on exit and report what each cost."""
    own = client is None
    if client is None:
        key = os.environ.get(spec.api_key_env)
        if not key:
            raise RunPodError(f"set {spec.api_key_env}")
        client = RunPodClient(key)
    pods: list[Pod] = []
    try:
        for i in range(spec.count):
            data = await client.create_pod(f"ce-{label}-{i}", spec)
            rate = float(data.get("costPerHr") or spec.usd_per_hour)
            pods.append(
                Pod(data["id"], data.get("name", ""), rate, time.time(), proxy_url(data["id"], spec.port))
            )
        results = await asyncio.gather(
            *(wait_for_comfy(p.url, spec.boot_timeout, health_transport, poll) for p in pods),
            return_exceptions=True,
        )
        ready = [p for p, r in zip(pods, results, strict=True) if not isinstance(r, Exception)]
        if not ready:
            raise RunPodError("no pod became ready: " + "; ".join(str(r) for r in results))
        yield [
            EndpointSpec(
                name=f"runpod-{p.id}",
                url=p.url,
                gpu=spec.gpu,
                usd_per_hour=p.usd_per_hour,
                concurrency=spec.concurrency,
            )
            for p in ready
        ]
    finally:
        for p in pods:
            try:
                await client.terminate(p.id)
            finally:
                if on_bill:
                    seconds = time.time() - p.created
                    on_bill(p, seconds, seconds * p.usd_per_hour / 3600)
        if own:
            await client.aclose()
