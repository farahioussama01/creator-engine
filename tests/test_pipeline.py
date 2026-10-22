from pathlib import Path

import pytest

from creator_engine import mock
from creator_engine.character import check
from creator_engine.ledger import Ledger
from creator_engine.pipeline import even_mix, plan, run_batch
from creator_engine.providers import FalProvider, MockProvider, RenderFailed
from creator_engine.report import answer, batch_numbers, write_html

from .conftest import FakeFal, fake_download


def ledger_of(cfg) -> Ledger:
    return Ledger(cfg.path(cfg.workdir) / "ledger.db")


def test_plan_is_deterministic_and_follows_the_mix(cfg):
    a = plan(cfg, "b1", {"mirror-walk": 2, "cafe-pov": 1})
    b = plan(cfg, "b1", {"mirror-walk": 2, "cafe-pov": 1})
    assert [p.prompt for p in a] == [p.prompt for p in b] and len(a) == 3
    assert "m1la woman" in a[0].prompt and "{" not in a[0].prompt + a[0].motion
    assert even_mix(["a", "b", "c"], 8) == {"a": 3, "b": 3, "c": 2}


async def test_mock_batch_end_to_end(cfg):
    batch = await run_batch(cfg, 12, log=lambda *_: None)
    led = ledger_of(cfg)
    n = batch_numbers(led, batch)
    assert n["mock"] and n["planned"] == 12 and n["keepers"] >= 8
    assert n["stills"] >= 24  # two candidates per piece
    assert n["spend"] == pytest.approx(n["spend_stills"] + n["spend_clips"])
    assert n["per_keeper"] == pytest.approx(n["spend"] / n["keepers"])
    for p in led.q("SELECT * FROM pieces WHERE status = 'keeper'"):
        assert Path(p["clip"]).exists() and p["identity"] > 0.6
    assert answer(cfg, led, batch).startswith("MOCK")  # never quotable
    assert "MOCK" in write_html(cfg, led, batch).read_text()


async def test_budget_stops_the_batch(cfg):
    cfg.budget_usd = 0.15  # ~ two clips' worth
    batch = await run_batch(cfg, 8, log=lambda *_: None)
    led = ledger_of(cfg)
    assert led.spent(batch) <= 0.15 + cfg.concurrency * 0.05
    assert led.one("SELECT COUNT(*) n FROM pieces WHERE status = 'skipped'")["n"] > 0


async def test_fal_batch_uses_the_real_request_shapes(cfg):
    # set up the character as `ce character check` + `ce character train` would
    train = cfg.path(cfg.datasets_dir) / "mila" / "train"
    train.mkdir(parents=True)
    import cv2

    for i in range(8):
        cv2.imwrite(str(train / f"{i:02d}.png"), mock.render_keyframe(360, 640, 500 + i, "mila", set()))
    check(cfg, faces=mock.MockFaces())
    cfg.save_character_state(lora_url="https://fal.media/mila_lora.safetensors", steps=2000, images=8)

    fal = FakeFal()
    provider = FalProvider(cfg, client=fal, download=fake_download())
    cfg.provider = "fal"
    batch = await run_batch(cfg, 4, provider=provider, faces=mock.MockFaces(), log=lambda *_: None)

    stills = [a for m, a in fal.calls if m == FalProvider.KEYFRAME_MODEL]
    clips = [a for m, a in fal.calls if m == FalProvider.CLIP_MODEL]
    assert stills[0]["loras"] == [{"path": "https://fal.media/mila_lora.safetensors", "scale": 1.0}]
    assert stills[0]["image_size"] == {"width": 720, "height": 1280} and stills[0]["num_inference_steps"] == 8
    assert clips and clips[0]["image_url"].startswith("https://fal.media/img")  # the still is reused, no re-upload
    assert clips[0]["resolution"] == "480p" and clips[0]["aspect_ratio"] == "9:16"
    led = ledger_of(cfg)
    n = batch_numbers(led, batch)
    assert n["spend_clips"] == pytest.approx(0.05 * n["clips"])
    assert n["spend_stills"] == pytest.approx(0.72 * 1.28 * 0.0085 * n["stills"])
    text = answer(cfg, led, batch)
    for line in ("Pipeline:", "Volume:", "Cost per piece:", "Views:"):
        assert line in text
    assert "fal.ai" in text and "8 images" in text and "2000 steps" in text


async def test_safety_filtered_still_is_charged_and_logged(cfg):
    cfg.save_character_state(lora_url="https://x/lora.safetensors")
    p = FalProvider(cfg, client=FakeFal(nsfw=True), download=fake_download())
    with pytest.raises(RenderFailed) as e:
        await p.keyframe("x", 1, cfg.path(cfg.workdir) / "s")
    assert e.value.usd > 0


def test_fal_provider_needs_a_trained_lora(cfg):
    with pytest.raises(RuntimeError, match="character train"):
        FalProvider(cfg, client=FakeFal())


async def test_mock_provider_prices_like_fal(cfg):
    m = MockProvider(cfg)
    assert m.price("clip") == 0.05 and m.price("keyframe") == pytest.approx(0.0078336)
