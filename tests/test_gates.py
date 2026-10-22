import cv2
import pytest

from creator_engine.config import load_characters, load_formats
from creator_engine.gates import Artifact, GateContext, run_gates
from creator_engine.sim.render import KeyframeScene, identity_vector, render_keyframe, render_video, write_mp4
from creator_engine.vision.ocr import classify_token, edit_distance, try_ocr

OCR = try_ocr()


@pytest.fixture
def ctx(cfg, analyzer, enrolled):
    return GateContext(cfg=cfg, analyzer=analyzer, centroids={"mila": enrolled}, ocr=OCR)


def keyframe(tmp_path, cfg, fmt="mirror-walk", seed=11, defects=(), sign=None, lora=None):
    ch = load_characters(cfg)["mila"]
    img, _ = render_keyframe(
        KeyframeScene(768, 1344, seed, identity_vector(lora or ch.lora), 0.9, sign, set(defects))
    )
    p = tmp_path / f"kf_{seed}_{'_'.join(defects) or 'clean'}.png"
    cv2.imwrite(str(p), img)
    prompt = f'a sign that reads "{sign}"' if sign else ""
    return Artifact("p", "keyframe", p, ch, load_formats(cfg)[fmt], prompt=prompt), img


def failed_gate(results):
    return next((r.gate for r in results if r.passed is False), None)


def test_clean_keyframe_passes_every_gate(tmp_path, cfg, ctx):
    art, _ = keyframe(tmp_path, cfg)
    ok, results = run_gates(art, ctx, full=True)
    assert ok, [(r.gate, r.reason) for r in results]
    assert art.cache["identity"] > 0.75


@pytest.mark.parametrize(
    "defect,gate",
    [
        ("blur", "sharpness"),
        ("overexposed", "exposure"),
        ("underexposed", "exposure"),
        ("waxy", "skin_texture"),
        ("big_head", "head_size"),
    ],
)
def test_each_keyframe_defect_is_caught_by_its_gate(tmp_path, cfg, ctx, defect, gate):
    art, _ = keyframe(tmp_path, cfg, defects=(defect,))
    ok, results = run_gates(art, ctx)
    assert not ok and failed_gate(results) == gate


def test_another_face_fails_identity(tmp_path, cfg, ctx):
    art, _ = keyframe(tmp_path, cfg, lora="someone_else.safetensors")
    ok, results = run_gates(art, ctx)
    assert not ok and failed_gate(results) == "identity"


@pytest.mark.skipif(OCR is None, reason="OCR extra not installed")
def test_misspelled_sign_is_caught_and_correct_sign_passes(tmp_path, cfg, ctx):
    good, _ = keyframe(tmp_path, cfg, fmt="cafe-pov", sign="COFFEE")
    assert run_gates(good, ctx)[0]
    bad, _ = keyframe(tmp_path, cfg, fmt="cafe-pov", sign="COFFEE", defects=("misspelled_sign",), seed=3)
    ok, results = run_gates(bad, ctx)
    assert not ok and failed_gate(results) == "sign_text"


def test_token_classifier():
    assert edit_distance("bakery", "bakrey") == 1  # adjacent swap counts once
    assert classify_token("COFFEE", {"coffee"}) == "ok"
    assert classify_token("COFEE", {"coffee"}) == "misspelled"
    assert classify_token("BAKREY") == "misspelled"  # near-miss of a 5+ letter sign word
    assert classify_token("nice") != "misspelled"  # short real word near "ice" is not flagged
    assert classify_token("Zentrova") == "unknown"  # brand-like word: allowed through


@pytest.mark.parametrize(
    "defect,gate",
    [
        (None, None),
        ("frozen", "motion"),
        ("chaos", "motion"),
        ("flicker", "flicker"),
        ("morph", "temporal_identity"),
    ],
)
def test_video_gates(tmp_path, cfg, ctx, defect, gate):
    _, img = keyframe(tmp_path, cfg, seed=21)
    seed = 5 if defect != "morph" else 11  # a morph toward a clearly different face (cos ~0.3)
    frames = render_video(img, 480, 848, 81, seed, {defect} if defect else set())
    p = tmp_path / f"clip_{defect}.mp4"
    write_mp4(frames, str(p), 16)
    ch = load_characters(cfg)["mila"]
    art = Artifact("p", "video", p, ch, load_formats(cfg)["mirror-walk"])
    ok, results = run_gates(art, ctx)
    if defect is None:
        assert ok, [(r.gate, r.reason) for r in results]
    else:
        assert not ok and failed_gate(results) == gate
