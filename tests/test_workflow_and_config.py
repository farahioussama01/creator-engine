import pytest

from creator_engine.config import CharacterSpec, RenderProfile
from creator_engine.sim import timing
from creator_engine.sim.server import REQUIRED_INPUTS
from creator_engine.workflow import (
    TemplateError,
    fill,
    load_template,
    placeholders,
    required_models,
    validate_against_object_info,
)

from .conftest import ROOT

KF = ROOT / "workflows" / "keyframe_zimage_lora.json"
VID = ROOT / "workflows" / "i2v_wan22_lightx2v.json"
OBJECT_INFO = {cls: {"input": {"required": {n: ["*"] for n in req}}} for cls, req in REQUIRED_INPUTS.items()}


def test_templates_declare_expected_placeholders():
    assert placeholders(load_template(KF)) == {
        "lora_name",
        "lora_strength",
        "positive",
        "seed",
        "steps",
        "cfg",
        "width",
        "height",
        "filename_prefix",
    }
    assert {"start_image", "length", "fps", "split_step", "lightning_strength"} <= placeholders(
        load_template(VID)
    )


def test_fill_keeps_types_links_and_strips_meta():
    g = fill(
        load_template(KF),
        {
            "lora_name": "a.safetensors",
            "lora_strength": 0.9,
            "positive": "hi",
            "seed": 5,
            "steps": 8,
            "cfg": 1.0,
            "width": 768,
            "height": 1344,
            "filename_prefix": "x/kf",
        },
    )
    assert g["9"]["inputs"]["steps"] == 8 and isinstance(g["9"]["inputs"]["seed"], int)
    assert g["4"]["inputs"]["model"] == ["1", 0]
    assert all("_meta" not in n for n in g.values())
    assert not placeholders(g)


def test_fill_strict_reports_missing():
    with pytest.raises(TemplateError, match="seed"):
        fill(load_template(KF), {"positive": "x"})


def test_templates_validate_against_comfy_node_schema():
    assert validate_against_object_info(load_template(KF), OBJECT_INFO) == []
    assert validate_against_object_info(load_template(VID), OBJECT_INFO) == []
    broken = load_template(VID)
    del broken["12"]["inputs"]["length"]
    broken["99"] = {"class_type": "NotANode", "inputs": {}}
    problems = validate_against_object_info(broken, OBJECT_INFO)
    assert any("length" in p for p in problems) and any("NotANode" in p for p in problems)


def test_required_models_lists_wan_files():
    need = required_models(load_template(VID))
    assert "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" in need["diffusion_models"]
    assert "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors" in need["loras"]


def test_timing_matches_published_anchor():
    # ComfyUI template notes: 640x640, 81 frames, RTX 4090D: ~71 s (4-step LoRA), ~513 s (20 steps, CFG 3.5)
    assert timing.video_seconds(640, 640, 81, 4, 1.0) == pytest.approx(71.0, abs=0.2)
    assert timing.video_seconds(640, 640, 81, 20, 3.5) == pytest.approx(513.1, abs=0.5)


def test_profile_frames_are_4n_plus_1():
    assert RenderProfile(name="x", seconds=5, fps=16).frames == 81
    assert RenderProfile(name="x", seconds=3, fps=16).frames == 49
    with pytest.raises(ValueError):
        RenderProfile(name="x", vid_width=500)


def test_character_rules():
    base = dict(id="a", name="A", trigger="t", look="l", lora="a.safetensors")
    with pytest.raises(ValueError):
        CharacterSpec(**base, disclosure="")
    with pytest.raises(ValueError, match="consent"):
        CharacterSpec(**base, disclosure="AI-generated", likeness="licensed")
    CharacterSpec(**base, disclosure="AI-generated", likeness="licensed", consent_record="contracts/a.pdf")
