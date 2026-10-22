"""Everything the pipeline needs, from three small YAML files in configs/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class Character(BaseModel):
    id: str
    name: str
    trigger: str = Field(description="The made-up word the LoRA learns, e.g. 'm1la woman'")
    look: str = Field(description="Fixed description repeated in every prompt")
    disclosure: str = Field(min_length=3, description="Added to every caption; cannot be empty")
    lora_scale: float = 1.0


class Format(BaseModel):
    keyframe: str = Field(description="Image prompt; may use {trigger} {look} {scene} {outfit}")
    motion: str = Field(description="Video prompt; may use {scene} {outfit}")
    scenes: list[str]
    outfits: list[str]
    captions: list[str]


class Prices(BaseModel):
    """USD list prices on fal.ai, checked 2026-10-22."""

    keyframe_per_mp: float = 0.0085  # fal-ai/z-image/turbo/lora
    portrait_per_mp: float = 0.005  # fal-ai/z-image/turbo (no LoRA, for the dataset)
    edit_per_mp: float = 0.03  # fal-ai/qwen-image-edit-2511
    clip: dict[str, float] = {"480p": 0.05, "580p": 0.075, "720p": 0.10}  # Wan 2.2 A14B turbo, per clip
    train_per_1000_steps: float = 2.26  # fal-ai/z-image-trainer
    hosted_reference_name: str = "Kling 3.0 720p, 5 s, 2 takes per keeper"
    hosted_reference_usd: float = 0.84  # $0.084/s x 5 s x 2


class Gates(BaseModel):
    identity_min: float | Literal["auto"] = "auto"  # auto = suggested from your training images
    sharpness_min: float = 30.0
    max_blown: float = 0.03  # share of pure-white pixels
    mean_min: float = 45.0
    mean_max: float = 210.0
    clip_min_motion: float = 0.3
    clip_face_share: float = 0.75  # face must be found in this share of sampled frames


class Config(BaseModel):
    provider: Literal["fal", "comfyui", "mock"] = "fal"
    workdir: Path = Path("runs")
    datasets_dir: Path = Path("datasets")
    concurrency: int = 4
    budget_usd: float = 10.0
    keyframe_width: int = 720
    keyframe_height: int = 1280
    keyframe_candidates: int = 2
    clip_resolution: Literal["480p", "580p", "720p"] = "480p"
    clip_attempts: int = 2
    gates: Gates = Gates()
    prices: Prices = Prices()
    comfyui_url: str = "http://127.0.0.1:8188"
    comfyui_usd_per_hour: float = 0.34
    comfyui_lora_file: str = "mila.safetensors"
    character: Character
    formats: dict[str, Format]
    root: Path = Path(".")

    def path(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.root / p

    @property
    def character_dir(self) -> Path:
        return self.path(self.workdir) / "character" / self.character.id

    def character_state(self) -> dict:
        """What `ce character train` produced: LoRA URL, identity centroid info."""
        f = self.character_dir / "state.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def save_character_state(self, **values) -> None:
        self.character_dir.mkdir(parents=True, exist_ok=True)
        state = {**self.character_state(), **values}
        (self.character_dir / "state.json").write_text(json.dumps(state, indent=2))


def load(config_path: str | Path = "configs/config.yaml") -> Config:
    config_path = Path(config_path).resolve()
    data = yaml.safe_load(config_path.read_text()) or {}
    root = config_path.parent.parent
    char = yaml.safe_load((root / data.pop("character_file", "configs/character.yaml")).read_text())
    formats = yaml.safe_load((root / data.pop("formats_file", "configs/formats.yaml")).read_text())
    return Config(**data, character=char, formats=formats, root=root)
