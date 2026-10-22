"""The real client, pool, engine and ledger against simulated ComfyUI pods."""

import pytest

from creator_engine.comfy import ComfyClient, ComfyExecutionError, ComfyValidationError
from creator_engine.config import EndpointSpec, load_characters, load_formats
from creator_engine.engine import Engine
from creator_engine.ledger import BUCKETS, build_ledger
from creator_engine.planner import plan_batch
from creator_engine.sim.server import SimComfyServer
from creator_engine.workflow import fill, load_template

from .conftest import ROOT


def kf_graph(lora="mila_zimage_v3.safetensors", seed=1):
    return fill(
        load_template(ROOT / "workflows" / "keyframe_zimage_lora.json"),
        {
            "lora_name": lora,
            "lora_strength": 0.9,
            "positive": "portrait",
            "seed": seed,
            "steps": 8,
            "cfg": 1.0,
            "width": 768,
            "height": 1344,
            "filename_prefix": "t/kf",
        },
    )


async def test_client_bills_gpu_time_not_wall_time(tmp_path):
    srv = SimComfyServer(time_scale=0.002, extra_loras=["mila_zimage_v3.safetensors"])
    url = await srv.start()
    try:
        async with ComfyClient(url, poll=0.01) as c:
            first = await c.run(kf_graph())
            second = await c.run(kf_graph(seed=2))
            assert first.exec_seconds == pytest.approx(second.exec_seconds + 20.0, abs=0.05)  # cold load
            assert second.wall_seconds < second.exec_seconds  # sim runs faster than real time
            path = await c.download(first.outputs[0], tmp_path)
            assert path.stat().st_size > 1000
            assert (await c.upload_image(path)).endswith(path.name)
            with pytest.raises(ComfyValidationError) as e:
                await c.queue_prompt(kf_graph(lora="missing.safetensors"))
            assert "missing.safetensors" in str(e.value.node_errors)
    finally:
        await srv.stop()


async def test_execution_errors_carry_burned_seconds():
    srv = SimComfyServer(time_scale=0.002, extra_loras=["mila_zimage_v3.safetensors"], fail_rate=1.0)
    url = await srv.start()
    try:
        async with ComfyClient(url, poll=0.01) as c:
            with pytest.raises(ComfyExecutionError) as e:
                await c.run(kf_graph())
            assert e.value.exec_seconds > 20 and "out of memory" in str(e.value)
    finally:
        await srv.stop()


async def run_small_batch(cfg, store, analyzer, n=6, **kw):
    ch = load_characters(cfg)["mila"]
    bid = plan_batch(
        cfg,
        store,
        ch,
        load_formats(cfg),
        {"mirror-walk": n // 2, "golden-street": n - n // 2},
        "standard",
        simulated=True,
    )
    eng = Engine(cfg, store, analyzer)
    return bid, await eng.run_batch(bid, **kw)


async def test_batch_end_to_end_and_ledger_adds_up(cfg, store, analyzer, enrolled, sim_pods):
    bid, res = await run_small_batch(cfg, store, analyzer)
    assert res["status"] == "done"
    assert sum(res["pieces"].values()) == 6 and res["pieces"].get("keeper", 0) >= 3
    led = build_ledger(cfg, store, bid)
    assert set(led.buckets) == set(BUCKETS)
    assert led.total_usd == pytest.approx(sum(led.buckets.values()))
    sessions = sum(r["usd"] for r in store.q("SELECT usd FROM spend WHERE kind = 'pod_session'"))
    assert led.total_usd == pytest.approx(sessions, rel=1e-6)  # all-in cost == what the pods billed
    for r in store.q("SELECT * FROM executions"):
        assert r["usd"] == pytest.approx(r["gpu_seconds"] * 0.34 / 3600)
    keepers = store.pieces(bid, "keeper")
    assert all(p["video_path"] and p["identity"] >= 0.70 for p in keepers)
    assert led.per_keeper < 0.05  # a 480p 5 s clip costs cents on a $0.34/h GPU
    assert led.hosted and all(h["x_ours"] > 10 for h in led.hosted)
    # resuming a finished batch renders nothing new
    n_exec = len(store.q("SELECT id FROM executions"))
    await Engine(cfg, store, analyzer).run_batch(bid)
    assert len(store.q("SELECT id FROM executions")) == n_exec


async def test_budget_cap_stops_scheduling(cfg, store, analyzer, enrolled, sim_pods):
    cfg.budget.max_usd_per_batch = 0.01  # enough for keyframes, not for many clips
    bid, res = await run_small_batch(cfg, store, analyzer)
    assert res["status"] == "budget_stopped"
    assert res["pieces"].get("skipped_budget", 0) > 0
    spent = store.one("SELECT SUM(usd) s FROM executions")["s"]
    assert spent <= 0.01 + 0.0075  # never more than one in-flight render past the cap


async def test_failing_pod_is_retried_elsewhere_and_dropped(cfg, store, analyzer, enrolled):
    loras = [c.lora for c in load_characters(cfg).values()]
    bad = SimComfyServer(name="bad", time_scale=0.002, extra_loras=loras, fail_rate=1.0)
    good = SimComfyServer(name="good", time_scale=0.002, extra_loras=loras)
    cfg.endpoints = [
        EndpointSpec(name="bad", url=await bad.start(), usd_per_hour=0.34),
        EndpointSpec(name="good", url=await good.start(), usd_per_hour=0.34),
    ]
    try:
        bid, res = await run_small_batch(cfg, store, analyzer, n=4)
        errors = store.q("SELECT * FROM executions WHERE status = 'error'")
        assert errors and all(e["endpoint"] == "bad" for e in errors)
        assert res["pieces"].get("keeper", 0) >= 2
        assert build_ledger(cfg, store, bid).buckets["errors / retries"] > 0
    finally:
        await bad.stop()
        await good.stop()
