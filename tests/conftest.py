from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from creator_engine.config import EndpointSpec, load_characters, load_config
from creator_engine.identity import enroll, list_images, make_sim_references, save
from creator_engine.sim.server import SimComfyServer
from creator_engine.store import Store
from creator_engine.vision import make_analyzer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(tmp_path):
    c = load_config(ROOT / "configs" / "engine.yaml")
    c.workdir = tmp_path / "runs"
    c.db = tmp_path / "runs" / "engine.db"
    c.poll_seconds = 0.01
    return c


@pytest.fixture
def config_file(tmp_path):
    data = yaml.safe_load((ROOT / "configs" / "engine.yaml").read_text())
    data.update(root=str(ROOT), workdir=str(tmp_path / "runs"), db=str(tmp_path / "runs" / "engine.db"))
    p = tmp_path / "engine.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


@pytest.fixture
def store(cfg):
    s = Store(cfg.path(cfg.db))
    yield s
    s.close()


@pytest.fixture
def analyzer():
    return make_analyzer("sim")


@pytest.fixture
def enrolled(cfg, analyzer):
    ch = load_characters(cfg)["mila"]
    refs = make_sim_references(ch, cfg.path(cfg.workdir) / "references" / "mila", n=10)
    centroid, rep = enroll(analyzer, "mila", list_images(refs[0].parent))
    save(cfg, centroid, rep)
    return centroid


@pytest.fixture
async def sim_pods(cfg):
    """Two simulated RTX 4090 pods wired into cfg.endpoints."""
    loras = [c.lora for c in load_characters(cfg).values()]
    servers = [SimComfyServer(name=f"t{i}", time_scale=0.002, extra_loras=loras, seed=i) for i in range(2)]
    urls = [await s.start() for s in servers]
    cfg.endpoints = [
        EndpointSpec(name=f"sim-{i}", url=u, gpu="rtx4090", usd_per_hour=0.34) for i, u in enumerate(urls)
    ]
    yield servers
    for s in servers:
        await s.stop()
