"""Typed configuration: characters, formats, render profiles, endpoints, gates, budgets, publishing.

Everything the engine does is driven by YAML under ``configs/``. Paths inside the engine config are
resolved relative to the repository root (the parent of the config file's directory).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class CharacterSpec(BaseModel):
    """One AI creator. The LoRA carries the face; ``look`` pins what the LoRA does not."""

    id: str
    name: str
    trigger: str = Field(description="LoRA trigger phrase, e.g. 'm1la woman'")
    look: str = Field(description="Fixed descriptors repeated in every prompt (hair, eyes, build)")
    lora: str = Field(description="LoRA filename inside ComfyUI/models/loras")
    lora_strength: float = 0.9
    negative: str = ""
    reference_dir: str | None = Field(
        default=None, description="Reference images used to build the identity centroid"
    )
    disclosure: str = Field(min_length=3, description="Always appended to captions; cannot be empty")
    likeness: Literal["fictional", "licensed"] = "fictional"
    consent_record: str | None = Field(
        default=None, description="Required when the character is a real person's licensed likeness"
    )

    @model_validator(mode="after")
    def _licensed_needs_consent(self) -> CharacterSpec:
        if self.likeness == "licensed" and not self.consent_record:
            raise ValueError(f"character {self.id}: a licensed likeness needs a consent_record")
        return self


class FormatSpec(BaseModel):
    """A repeatable content format. Variation comes from scenes x outfits x lights."""

    id: str
    description: str
    keyframe_prompt: str = Field(description="Template using {trigger} {look} {scene} {outfit} {light}")
    motion_prompt: str = Field(description="Image-to-video motion prompt; may use {scene} {outfit}")
    scenes: list[str]
    outfits: list[str] = Field(default_factory=lambda: ["casual outfit"])
    lights: list[str] = Field(default_factory=lambda: ["soft natural light"])
    caption: str = "{hook}"
    hooks: list[str] = Field(default_factory=lambda: [""])
    hashtags: list[str] = Field(default_factory=list)
    head_ratio: tuple[float, float] = Field(
        default=(0.06, 0.30), description="Expected face height / frame height for this shot type"
    )
    for_characters: list[str] = Field(
        default_factory=list, description="Characters this format suits; empty = any"
    )
    has_signage: bool = False
    allowed_text: list[str] = Field(
        default_factory=list, description="Brand/place words the OCR gate accepts"
    )

    def render(self, template: str, **values: str) -> str:
        return " ".join(template.format(**values).split())


class RenderProfile(BaseModel):
    """The cost knobs. Every field here moves cost per piece."""

    name: str
    kf_width: int = 768
    kf_height: int = 1344
    kf_steps: int = 8
    kf_cfg: float = 1.0
    keyframe_candidates: int = Field(default=2, ge=1, description="Best-of-k at the cheap stage")
    keyframe_rounds: int = Field(
        default=2, ge=1, description="Top-up rounds for pieces with no passing keyframe"
    )
    vid_width: int = 480
    vid_height: int = 848
    fps: int = 16
    seconds: float = 5.0
    vid_steps: int = 4
    vid_cfg: float = 1.0
    split_step: int = 2
    lightning: bool = Field(default=True, description="lightx2v 4-step distillation LoRAs")
    max_video_attempts: int = Field(default=2, ge=1)

    @field_validator("vid_width", "vid_height")
    @classmethod
    def _mult16(cls, v: int) -> int:
        if v % 16:
            raise ValueError("Wan video dimensions must be multiples of 16")
        return v

    @property
    def frames(self) -> int:
        """Wan wants 4n+1 frames (81 = 5 s at 16 fps)."""
        n = round(self.seconds * self.fps / 4)
        return max(1, n) * 4 + 1

    @property
    def video_megapixel_frames(self) -> float:
        return self.vid_width * self.vid_height * self.frames / 1e6


class EndpointSpec(BaseModel):
    """A ComfyUI server on a rented GPU."""

    name: str
    url: str
    gpu: str = "rtx4090"
    usd_per_hour: float = 0.34
    concurrency: int = Field(
        default=2, ge=1, description="Prompts in flight; 2 keeps the GPU fed during downloads"
    )
    stages: list[Literal["keyframe", "video"]] = Field(default_factory=lambda: ["keyframe", "video"])
    headers: dict[str, str] = Field(default_factory=dict)


class GateSettings(BaseModel):
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class BudgetSpec(BaseModel):
    max_usd_per_batch: float = 25.0
    max_usd_per_day: float = 60.0


class JudgeSpec(BaseModel):
    enabled: bool = False
    model: str = "claude-haiku-5-5"
    usd_per_mtok_in: float = 0.10
    usd_per_mtok_out: float = 0.50
    max_side: int = 768
    stages: list[Literal["keyframe", "video"]] = Field(default_factory=lambda: ["video"])


class AccountSpec(BaseModel):
    handle: str
    character: str
    platform: Literal["instagram"] = "instagram"
    timezone: str = "Europe/Paris"
    daily_cap: int = Field(default=2, ge=1)
    windows: list[str] = Field(default_factory=lambda: ["12:00-13:30", "19:00-21:30"])
    min_gap_minutes: int = 180
    ig_user_id_env: str | None = None
    token_env: str | None = None
    baseline_views: float | None = Field(
        default=None, description="Optional prior for the account's median views"
    )


class PublishSpec(BaseModel):
    dry_run: bool = True
    api_daily_cap: int = Field(
        default=50, description="Meta documents 50-100 API posts / 24h; keep the stricter one"
    )
    graph_host: str = "graph.instagram.com"
    api_version: str = "v26.0"
    media_base_url: str | None = Field(default=None, description="Public URL prefix where outputs are synced")
    accounts: list[AccountSpec] = Field(default_factory=list)


class RunPodSpec(BaseModel):
    """Pods the engine rents for a run and terminates afterwards."""

    api_key_env: str = "RUNPOD_API_KEY"
    image: str = Field(description="Container image that starts ComfyUI on `port`")
    gpu_type_ids: list[str] = Field(default_factory=lambda: ["NVIDIA GeForce RTX 4090"])
    gpu: str = "rtx4090"
    usd_per_hour: float = 0.34
    cloud_type: Literal["COMMUNITY", "SECURE"] = "COMMUNITY"
    interruptible: bool = False
    network_volume_id: str | None = Field(
        default=None, description="Models live here so pods boot without downloads"
    )
    container_disk_gb: int = 40
    port: int = 8188
    env: dict[str, str] = Field(default_factory=dict)
    count: int = 2
    concurrency: int = 2
    boot_timeout: float = 900.0


class SimSpec(BaseModel):
    """Parameters for the GPU simulator used by the demo and the tests."""

    time_scale: float = 0.02
    defect_rates: dict[str, float] = Field(default_factory=dict)


class EngineConfig(BaseModel):
    workdir: Path = Path("runs")
    db: Path = Path("runs/engine.db")
    characters_dir: Path = Path("configs/characters")
    formats_dir: Path = Path("configs/formats")
    workflows: dict[str, Path] = Field(
        default_factory=lambda: {
            "keyframe": Path("workflows/keyframe_zimage_lora.json"),
            "video": Path("workflows/i2v_wan22_lightx2v.json"),
        }
    )
    prices: Path = Path("configs/prices.yaml")
    analyzer: Literal["sim", "real"] = "sim"
    models_dir: Path = Path("models")
    scheduling: Literal["phased", "interleaved"] = "phased"
    poll_seconds: float = Field(default=0.5, description="How often to poll /history for a finished prompt")
    profiles: dict[str, RenderProfile] = Field(default_factory=dict)
    endpoints: list[EndpointSpec] = Field(default_factory=list)
    gates: dict[str, GateSettings] = Field(default_factory=dict)
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    judge: JudgeSpec = Field(default_factory=JudgeSpec)
    publish: PublishSpec = Field(default_factory=PublishSpec)
    sim: SimSpec = Field(default_factory=SimSpec)
    runpod: RunPodSpec | None = None
    root: Path = Path(".")

    def path(self, p: Path | str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (self.root / p)

    def profile(self, name: str) -> RenderProfile:
        if name not in self.profiles:
            raise KeyError(f"unknown profile {name!r}; known: {sorted(self.profiles)}")
        return self.profiles[name]

    def gate(self, name: str) -> GateSettings:
        return self.gates.get(name, GateSettings())


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def load_config(path: str | Path) -> EngineConfig:
    path = Path(path).resolve()
    data = _read_yaml(path)
    root = Path(data.pop("root", path.parent.parent)).resolve()
    profiles = data.pop("profiles", {}) or {}
    cfg = EngineConfig(
        **data,
        root=root,
        profiles={name: RenderProfile(name=name, **(spec or {})) for name, spec in profiles.items()},
    )
    return cfg


def load_characters(cfg: EngineConfig) -> dict[str, CharacterSpec]:
    out: dict[str, CharacterSpec] = {}
    for f in sorted(cfg.path(cfg.characters_dir).glob("*.yaml")):
        spec = CharacterSpec(**_read_yaml(f))
        out[spec.id] = spec
    return out


def load_formats(cfg: EngineConfig) -> dict[str, FormatSpec]:
    out: dict[str, FormatSpec] = {}
    for f in sorted(cfg.path(cfg.formats_dir).glob("*.yaml")):
        spec = FormatSpec(**_read_yaml(f))
        out[spec.id] = spec
    return out


def load_prices(cfg: EngineConfig) -> dict[str, Any]:
    p = cfg.path(cfg.prices)
    return _read_yaml(p) if p.exists() else {}


def ceil_div(a: float, b: float) -> int:
    return int(math.ceil(a / b))
