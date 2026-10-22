import csv

import cv2
import numpy as np
import pytest

from creator_engine import character, mock
from creator_engine.faces import ARCFACE_TEMPLATE, align_face, enroll, preprocess
from creator_engine.gates import Checker
from creator_engine.ledger import Ledger
from creator_engine.report import answer, views_numbers
from creator_engine.views import import_views, next_mix

from .conftest import FakeFal, fake_download


@pytest.fixture
def checker(cfg, tmp_path):
    faces = mock.MockFaces()
    refs = []
    for i in range(10):
        p = tmp_path / f"ref{i}.png"
        cv2.imwrite(str(p), mock.render_keyframe(360, 640, 100 + i, "mila", set()))
        refs.append(p)
    e = enroll(faces, refs)
    return Checker(cfg, faces, e.centroid, e.suggested_threshold)


def still(tmp_path, defects=(), who="mila", seed=7):
    p = tmp_path / f"still_{'_'.join(defects) or who}.png"
    cv2.imwrite(str(p), mock.render_keyframe(360, 640, seed, who, set(defects)))
    return p


def clip(tmp_path, defects=(), seed=11):
    p = tmp_path / f"clip_{'_'.join(defects) or 'ok'}.mp4"
    mock.write_mp4(mock.render_clip(cv2.imread(str(still(tmp_path))), 240, 424, 49, seed, set(defects)), str(p))
    return p


def test_good_still_passes(tmp_path, checker):
    v = checker.keyframe(still(tmp_path))
    assert v.ok and v.identity > 0.8


@pytest.mark.parametrize("defect,reason", [("blur", "blurry face"), ("overexposed", "light"), ("drift", "not Mila")])
def test_bad_stills_fail_with_a_reason(tmp_path, checker, defect, reason):
    v = checker.keyframe(still(tmp_path, (defect,)))
    assert not v.ok and v.reason.startswith(reason)


def test_someone_else_fails_identity(tmp_path, checker):
    assert checker.keyframe(still(tmp_path, who="someone-else")).reason.startswith("not Mila")


def test_clip_checks(tmp_path, checker):
    assert checker.clip(clip(tmp_path)).ok
    assert checker.clip(clip(tmp_path, ("frozen",))).reason.startswith("frozen clip")
    assert checker.clip(clip(tmp_path, ("morph",), seed=3)).reason.startswith("face changes mid-clip")


def test_enrollment_drops_an_outlier(tmp_path):
    paths = []
    for i in range(9):
        paths.append(tmp_path / f"{i}.png")
        cv2.imwrite(str(paths[-1]), mock.render_keyframe(360, 640, 300 + i, "mila", set()))
    paths.append(tmp_path / "intruder.png")
    cv2.imwrite(str(paths[-1]), mock.render_keyframe(360, 640, 999, "someone-else", set()))
    e = enroll(mock.MockFaces(), paths)
    assert e.outliers == ["intruder.png"] and e.used == 9 and 0.5 < e.suggested_threshold < 0.95


def test_face_alignment_shapes():
    img = np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    x = preprocess(align_face(img, ARCFACE_TEMPLATE * 2 + 40))
    assert x.shape == (1, 3, 112, 112) and -1 <= x.min() and x.max() <= 1


async def test_character_flow_on_fal(cfg, tmp_path, monkeypatch):
    fal = FakeFal()
    monkeypatch.setattr(character, "_download", fake_download())  # downloads write mock images
    cands = await character.portraits(cfg, 3, client=fal)
    assert len(cands) == 3 and fal.calls[0][0] == "fal-ai/z-image/turbo"
    imgs = await character.variations(cfg, cands[0], 5, client=fal)
    assert len(imgs) == 6  # hero + 5
    edit = next(a for m, a in fal.calls if m == "fal-ai/qwen-image-edit-2511")
    assert edit["image_urls"][0].startswith("https://fal.media/upload/") and "keep the identity" in edit["prompt"]
    lora = await character.train(cfg, 1500, client=fal)
    zipped = next(a["path"] for m, a in fal.calls if m == "upload" and a["path"].endswith(".zip"))
    assert zipped and lora.endswith(".safetensors")
    train_args = next(a for m, a in fal.calls if m == "fal-ai/z-image-trainer")
    assert train_args["steps"] == 1500 and train_args["default_caption"] == "photo of m1la woman"
    st = cfg.character_state()
    assert st["lora_url"] == lora and st["training_usd"] == pytest.approx(1.5 * 2.26)
    setup = Ledger(cfg.path(cfg.workdir) / "ledger.db").q("SELECT kind, usd FROM setup")
    assert {r["kind"] for r in setup} == {"dataset", "training"}


async def test_views_drive_the_next_mix_and_the_answer(cfg, tmp_path):
    from creator_engine.pipeline import run_batch

    batch = await run_batch(cfg, 9, log=lambda *_: None)
    led = Ledger(cfg.path(cfg.workdir) / "ledger.db")
    keepers = led.q("SELECT id, format FROM pieces WHERE status = 'keeper'")
    views = {"mirror-walk": 3000, "golden-street": 1000, "cafe-pov": 300}
    f = tmp_path / "views.csv"
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "platform", "posted_at", "views"])
        for i, k in enumerate(keepers):
            w.writerow([f"{k['id']}.mp4", "instagram", f"2026-10-{24 + i % 5}", views[k["format"]]])
    assert import_views(led, f) == len(keepers)
    v = views_numbers(led)
    assert list(v["by_format"])[0] == "mirror-walk" and v["days"] >= 2
    mix = next_mix(led, ["mirror-walk", "golden-street", "cafe-pov"], 20)
    assert sum(mix.values()) == 20 and mix["mirror-walk"] > mix["golden-street"] > mix["cafe-pov"] >= 1
    assert answer(cfg, led, batch).startswith("MOCK")  # views exist, but this batch was still mock
