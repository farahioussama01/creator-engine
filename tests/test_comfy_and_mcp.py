"""The own-GPU path (ComfyUI API) against a fake ComfyUI, and the MCP server."""

import json

import cv2
import httpx
import pytest
import yaml

from creator_engine import mock
from creator_engine.comfy import ComfyClient
from creator_engine.providers import ComfyProvider, Render

from .conftest import ROOT


def fake_comfy(tmp_path):
    img = tmp_path / "out.png"
    cv2.imwrite(str(img), mock.render_keyframe(360, 640, 1, "mila", set()))
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.path))
        if req.url.path == "/prompt":
            graph = json.loads(req.content)["prompt"]
            assert all("{{" not in json.dumps(n) for n in graph.values())  # every placeholder filled
            return httpx.Response(200, json={"prompt_id": "p1", "number": 1, "node_errors": {}})
        if req.url.path == "/history/p1":
            msgs = [["execution_start", {"timestamp": 1_000}], ["execution_success", {"timestamp": 72_000}]]
            return httpx.Response(
                200,
                json={
                    "p1": {
                        "status": {"status_str": "success", "completed": True, "messages": msgs},
                        "outputs": {"9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}},
                    }
                },
            )
        if req.url.path == "/view":
            return httpx.Response(200, content=img.read_bytes())
        if req.url.path == "/upload/image":
            return httpx.Response(200, json={"name": "kf.png", "subfolder": "", "type": "input"})
        return httpx.Response(404)

    return httpx.MockTransport(handler), seen


async def test_comfy_provider_bills_gpu_seconds(cfg, tmp_path):
    transport, seen = fake_comfy(tmp_path)
    provider = ComfyProvider(cfg, client=ComfyClient("http://pod", transport=transport, poll=0.01))
    kf = await provider.keyframe("photo of m1la woman", 1, tmp_path / "still")
    assert kf.path.exists() and kf.seconds == 71.0
    assert kf.usd == pytest.approx(71.0 * 0.34 / 3600)  # GPU seconds x hourly rate
    clip = await provider.clip(Render(kf.path, 0, 0), "she smiles", 2, tmp_path / "clip")
    assert ("POST", "/upload/image") in seen and clip.path.exists()


async def test_mcp_tools(cfg, tmp_path):
    from mcp import Client

    from creator_engine.mcp_server import build_server
    from creator_engine.pipeline import run_batch

    await run_batch(cfg, 3, log=lambda *_: None)
    data = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    data.update(provider="mock", workdir=str(cfg.workdir))
    data["character_file"] = str(ROOT / "configs" / "character.yaml")
    data["formats_file"] = str(ROOT / "configs" / "formats.yaml")
    conf = tmp_path / "configs" / "config.yaml"
    conf.parent.mkdir()
    conf.write_text(yaml.safe_dump(data))
    async with Client(build_server(str(conf))) as c:
        names = {t.name for t in (await c.list_tools()).tools}
        assert names == {"run_batch", "batch_status", "numbers", "pieces", "next_mix"}
        nums = (await c.call_tool("numbers", {})).structured_content
        assert nums["answer"].startswith("MOCK") and nums["planned"] == 3
