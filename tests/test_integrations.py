"""RunPod lifecycle, Claude judge, MCP server, identity enrollment, real-analyzer preprocessing."""

import json
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
import pytest

from creator_engine.config import JudgeSpec, RunPodSpec, load_characters
from creator_engine.identity import enroll, list_images, make_sim_references
from creator_engine.judge import ClaudeJudge
from creator_engine.runpod import RunPodClient, rented_pods
from creator_engine.vision.real import ARCFACE_TEMPLATE, align_face, preprocess

# ---------------------------------------------------------------- RunPod


def runpod_mock(log, fail_health=False):
    n = {"pods": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        log.append((req.method, req.url.path))
        if req.method == "POST" and req.url.path == "/v1/pods":
            body = json.loads(req.content)
            assert body["ports"] == ["8188/http"] and body["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"]
            n["pods"] += 1
            return httpx.Response(
                201, json={"id": f"pod{n['pods']}", "name": body["name"], "costPerHr": 0.34}
            )
        if req.method == "DELETE":
            return httpx.Response(204)
        if req.url.path == "/system_stats":
            return httpx.Response(503 if fail_health else 200, json={"devices": []})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


async def test_rented_pods_are_always_terminated():
    log, bills = [], []
    spec = RunPodSpec(image="example/comfyui:latest", count=2, boot_timeout=5)
    transport = runpod_mock(log)
    client = RunPodClient("key", transport=transport)
    with pytest.raises(RuntimeError, match="batch crashed"):
        async with rented_pods(
            spec,
            "t",
            on_bill=lambda p, s, u: bills.append((p.id, u)),
            client=client,
            health_transport=transport,
            poll=0.01,
        ) as eps:
            assert [e.url for e in eps] == [
                "https://pod1-8188.proxy.runpod.net",
                "https://pod2-8188.proxy.runpod.net",
            ]
            raise RuntimeError("batch crashed")
    assert [m for m, p in log if m == "DELETE"] == ["DELETE", "DELETE"]
    assert [b[0] for b in bills] == ["pod1", "pod2"]


async def test_pods_that_never_boot_are_still_terminated():
    log = []
    spec = RunPodSpec(image="x", count=1, boot_timeout=0.05)
    transport = runpod_mock(log, fail_health=True)
    with pytest.raises(RuntimeError, match="no pod became ready"):
        async with rented_pods(
            spec, "t", client=RunPodClient("k", transport=transport), health_transport=transport, poll=0.01
        ):
            pass
    assert ("DELETE", "/v1/pods/pod1") in log


# ---------------------------------------------------------------- Claude judge


class FakeMessages:
    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(self.payload))],
            usage=SimpleNamespace(input_tokens=1500, output_tokens=120),
        )


def test_judge_verdicts_and_cost():
    good = {
        "light": 4,
        "skin": 4,
        "hands": None,
        "anatomy": 5,
        "text": None,
        "motion": 4,
        "overall": "good",
        "reasons": [],
    }
    fake = FakeMessages(good)
    judge = ClaudeJudge(JudgeSpec(), client=SimpleNamespace(messages=fake))
    img = np.full((1344, 768, 3), 128, np.uint8)
    v = judge.judge_images([img, img, img, img], "she walks", video=True)
    assert v.passed and v.min_score == 4
    call = fake.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["output_config"]["format"]["type"] == "json_schema"
    blocks = call["messages"][0]["content"]
    assert sum(b["type"] == "image" for b in blocks) == 4 and blocks[0] == {
        "type": "text",
        "text": "Frame 1:",
    }
    assert v.usd == pytest.approx(1500 * 0.10 / 1e6 + 120 * 0.50 / 1e6)
    assert judge.spent == [v.usd]

    bad = dict(good, hands=2, overall="brain_rot", reasons=["six fingers on the left hand"])
    v2 = ClaudeJudge(JudgeSpec(), client=SimpleNamespace(messages=FakeMessages(bad))).judge_images([img], "x")
    assert not v2.passed and "hands" in v2.summary and v2.scores["motion"] is None


# ---------------------------------------------------------------- identity & real analyzer


def test_enrollment_flags_outliers_and_suggests_threshold(cfg, analyzer, tmp_path):
    ch = load_characters(cfg)["mila"]
    refs = make_sim_references(ch, tmp_path / "refs", n=10)
    from creator_engine.sim.render import KeyframeScene, identity_vector, render_keyframe

    img, _ = render_keyframe(KeyframeScene(768, 1344, 99, identity_vector("intruder"), 1.0))
    cv2.imwrite(str(tmp_path / "refs" / "zz_wrong_person.png"), img)
    centroid, rep = enroll(analyzer, "mila", list_images(refs[0].parent))
    assert rep.outliers == ["zz_wrong_person.png"] and rep.n_used == 10
    assert 0.6 < rep.suggested_threshold < rep.loo_p10
    assert np.linalg.norm(centroid) == pytest.approx(1.0, abs=1e-5)


def test_arcface_alignment_and_preprocessing():
    img = np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    lm = ARCFACE_TEMPLATE * 2 + 40  # same face shape, scaled and shifted
    aligned = align_face(img, lm)
    assert aligned.shape == (112, 112, 3)
    x = preprocess(aligned)
    assert x.shape == (1, 3, 112, 112) and x.dtype == np.float32 and -1.0 <= x.min() and x.max() <= 1.0


# ---------------------------------------------------------------- MCP server


async def test_mcp_server_tools(config_file):
    from mcp import Client

    from creator_engine.mcp_server import build_server

    server = build_server(str(config_file))
    async with Client(server) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert {
            "engine_status",
            "plan_batch",
            "run_batch",
            "batch_status",
            "batch_report",
            "list_pieces",
            "format_experiments",
            "schedule_posts",
            "render_dashboard",
            "preflight",
        } <= names
        res = await client.call_tool("plan_batch", {"character": "mila", "n": 6, "profile": "draft"})
        planned = res.structured_content
        assert planned["quotas"] == {"mirror-walk": 2, "cafe-pov": 2, "golden-street": 2}
        status = (await client.call_tool("batch_status", {"batch_id": planned["batch"]})).structured_content
        assert status["pieces"] == {"planned": 6}
        exp = await client.call_tool("format_experiments", {"character": "mila"})
        assert not exp.is_error


async def test_gate_calibration_on_simulated_truth(cfg, store, analyzer, enrolled, sim_pods):
    from creator_engine.calibrate import calibrate
    from creator_engine.config import load_formats
    from creator_engine.engine import Engine
    from creator_engine.ops import Ctx
    from creator_engine.planner import plan_batch

    cfg.sim.defect_rates = {}
    for s in sim_pods:  # make defects frequent so every gate sees bad examples
        s.rates.update(blur=0.3, waxy=0.3, identity_drift=0.3)
    bid = plan_batch(
        cfg, store, load_characters(cfg)["mila"], load_formats(cfg), {"mirror-walk": 8}, "draft", True
    )
    await Engine(cfg, store, analyzer).run_batch(bid)
    rows = {r["gate"]: r for r in calibrate(Ctx(cfg, store), bid)}
    for gate in ("sharpness", "skin_texture", "identity"):
        r = rows[gate]
        assert r["n_bad"] > 0 and r["n_good"] > 0, gate
        assert r["recall_at_current"] >= 0.75 and r["false_reject_at_current"] == 0, gate
        assert r["recall_at_suggested"] >= r["recall_at_current"], gate
