import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import pytest

from creator_engine.config import AccountSpec, load_formats
from creator_engine.experiments import allocate, arm_stats
from creator_engine.publish import GraphAPIError, InstagramPublisher, build_caption, next_slot


def seed_posts(store, appeal, n_per_format=12, accounts=("a.big", "a.small"), base=None, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    base = base or {"a.big": 20000.0, "a.small": 1500.0}
    t0 = time.time() - 30 * 86400
    k = 0
    for fmt, a in appeal.items():
        for i in range(n_per_format):
            acct = accounts[i % len(accounts)]
            pid = f"{fmt}-{i}@{acct}"
            store.insert(
                "posts",
                {
                    "id": pid,
                    "piece_id": f"{fmt}-{i}",
                    "batch_id": "b",
                    "account": acct,
                    "character": "mila",
                    "format": fmt,
                    "scheduled_at": t0 + k * 3600,
                    "status": "published",
                    "caption": "x",
                    "container_id": None,
                    "media_id": None,
                    "published_at": t0,
                    "error": None,
                    "simulated": 1,
                },
            )
            store.insert(
                "metrics",
                {
                    "post_id": pid,
                    "fetched_at": t0,
                    "age_hours": 48,
                    "views": base[acct] * a * rng.lognormal(0, 0.4),
                    "reach": 0,
                    "saved": 0,
                    "shares": 1,
                    "interactions": 0,
                    "avg_watch_ms": 0,
                    "skip_rate": 0.5,
                    "simulated": 1,
                },
            )
            k += 1


def test_bandit_finds_best_format_despite_account_size(cfg, store):
    # cafe-pov posts mostly land on the big account in raw views terms; baselines must normalise that away
    seed_posts(store, {"mirror-walk": 1.6, "golden-street": 1.0, "cafe-pov": 0.55})
    stats = {s.format: s for s in arm_stats(cfg, store, "mila", load_formats(cfg))}
    assert set(stats) == {"mirror-walk", "golden-street", "cafe-pov"}  # gym-mirror is rafa's format
    assert stats["mirror-walk"].p_best > 0.8 and stats["mirror-walk"].status == "scale"
    assert stats["cafe-pov"].status == "kill"
    quota = allocate(list(stats.values()), 24)
    assert sum(quota.values()) == 24 and "cafe-pov" not in quota
    assert quota["mirror-walk"] > quota.get("golden-street", 0) >= 1


def test_new_formats_get_exploration_slots(cfg, store):
    stats = arm_stats(cfg, store, "mila", load_formats(cfg))
    assert all(s.status == "explore" for s in stats)
    quota = allocate(stats, 6)
    assert sum(quota.values()) == 6 and all(q >= 2 for q in quota.values())


def test_caption_always_carries_disclosure():
    c = build_caption("fit check", ["ootd", "#mirrorselfie"], "AI-generated character")
    assert c.endswith("AI-generated character") and "#ootd" in c
    long = build_caption("x" * 5000, [], "AI-generated character")
    assert len(long) <= 2200 and long.endswith("AI-generated character")
    with pytest.raises(ValueError):
        build_caption("hi", [], "  ")


def test_slots_respect_windows_caps_and_gaps():
    acct = AccountSpec(
        handle="a",
        character="mila",
        timezone="America/Mexico_City",
        daily_cap=2,
        windows=["07:00-08:30", "18:00-20:00"],
        min_gap_minutes=180,
    )
    tz = ZoneInfo(acct.timezone)
    now = datetime(2026, 11, 2, 6, 0, tzinfo=tz).timestamp()
    taken: list[float] = []
    for _ in range(5):
        taken.append(next_slot(acct, taken, now, api_cap=50))
    local = [datetime.fromtimestamp(t, tz) for t in taken]
    assert [d.hour for d in local[:2]] == [7, 18]  # first window, then the evening window
    days = {}
    for d in local:
        days[d.date()] = days.get(d.date(), 0) + 1
        assert (7, 0) <= (d.hour, d.minute) <= (8, 30) or (18, 0) <= (d.hour, d.minute) <= (20, 0)
    assert max(days.values()) <= 2
    assert all(b - a >= 180 * 60 for a, b in zip(sorted(taken), sorted(taken)[1:], strict=False))


def graph_api_mock(calls, statuses=("IN_PROGRESS", "FINISHED"), quota_usage=3):
    seq = list(statuses)

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append((req.method, req.url.path, dict(req.url.params), req.content.decode()))
        path = req.url.path
        if path.endswith("/content_publishing_limit"):
            return httpx.Response(
                200, json={"data": [{"quota_usage": quota_usage, "config": {"quota_total": 50}}]}
            )
        if path.endswith("/media") and req.method == "POST":
            return httpx.Response(200, json={"id": "c123"})
        if path.endswith("/c123"):
            return httpx.Response(200, json={"status_code": seq.pop(0)})
        if path.endswith("/media_publish"):
            return httpx.Response(200, json={"id": "m456"})
        if path.endswith("/m456/insights"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"name": "views", "values": [{"value": 1234}]},
                        {"name": "reach", "total_value": {"value": 900}},
                    ]
                },
            )
        return httpx.Response(404, json={"error": {"message": "nope"}})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_instagram_reel_publish_flow():
    calls = []
    pub = InstagramPublisher("17841", "TOKEN", client=graph_api_mock(calls), poll_seconds=0)
    container, media = pub.publish_reel("https://cdn.example.com/a.mp4", "hello\n\nAI-generated character")
    assert (container, media) == ("c123", "m456")
    create = next(c for c in calls if c[1].endswith("/17841/media"))
    assert "media_type=REELS" in create[3] and "video_url=https" in create[3]
    assert calls[-1][1].endswith("/media_publish") and "creation_id=c123" in calls[-1][3]
    assert all("/v26.0/" in c[1] for c in calls)
    assert pub.insights("m456") == {"views": 1234, "reach": 900}


def test_instagram_quota_and_container_errors():
    with pytest.raises(GraphAPIError, match="quota"):
        InstagramPublisher("1", "t", client=graph_api_mock([], quota_usage=50)).publish_reel("u", "c")
    with pytest.raises(GraphAPIError, match="ERROR"):
        InstagramPublisher(
            "1", "t", client=graph_api_mock([], statuses=("ERROR",)), poll_seconds=0
        ).publish_reel("u", "c")


async def test_schedule_and_dry_run_publish(cfg, store, analyzer, enrolled, sim_pods):
    from creator_engine.config import load_characters
    from creator_engine.engine import Engine
    from creator_engine.planner import plan_batch
    from creator_engine.publish import publish_due, schedule_batch

    bid = plan_batch(
        cfg, store, load_characters(cfg)["mila"], load_formats(cfg), {"mirror-walk": 4}, "standard", True
    )
    await Engine(cfg, store, analyzer).run_batch(bid)
    posts = schedule_batch(cfg, store, bid)
    keepers = store.pieces(bid, "keeper")
    assert len(posts) == len(keepers) > 0
    assert all(p["caption"].endswith("AI-generated character · #AIcreator") for p in posts)
    assert {p["account"] for p in posts} <= {"mila.wanders", "mila.slowdays"}
    sidecar = json.loads(open(keepers[0]["video_path"].rsplit(".", 1)[0] + ".provenance.json").read())
    assert sidecar["digital_source_type"].endswith("trainedAlgorithmicMedia")
    done = publish_due(cfg, store, now=time.time() + 30 * 86400)
    assert len(done) == len(posts) and all(d["status"] == "dry_run" for d in done)
    assert len(list((cfg.path(cfg.workdir) / "outbox").rglob("*.json"))) == len(posts)
