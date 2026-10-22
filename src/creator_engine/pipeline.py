"""The whole pipeline, top to bottom.

For each planned piece:
  1. render 2 stills of the character (cheap)
  2. check them (light, one face, focus, identity) and keep the one most like the character
  3. animate that still into a ~5 s clip (expensive)
  4. check the clip (moving, same face throughout); re-roll once if it fails
Every render goes into the ledger with its price, so cost per keeper = total spend / keepers.
"""

from __future__ import annotations

import asyncio
import random
import time
import zlib
from dataclasses import dataclass

import numpy as np

from .config import Config
from .gates import Checker
from .ledger import Ledger
from .providers import MockProvider, Render, RenderFailed, make_provider


@dataclass
class Piece:
    id: str
    format: str
    prompt: str
    motion: str
    caption: str
    seed: int


def even_mix(formats: list[str], n: int) -> dict[str, int]:
    base, extra = divmod(n, len(formats))
    return {f: base + (i < extra) for i, f in enumerate(formats)}


def plan(cfg: Config, batch: str, mix: dict[str, int]) -> list[Piece]:
    pieces = []
    for fmt_id, count in mix.items():
        fmt = cfg.formats[fmt_id]
        for i in range(count):
            pid = f"{batch}-{fmt_id}-{i:03d}"
            rng = random.Random(pid)
            v = {
                "trigger": cfg.character.trigger,
                "look": cfg.character.look,
                "scene": rng.choice(fmt.scenes),
                "outfit": rng.choice(fmt.outfits),
            }
            pieces.append(
                Piece(
                    pid,
                    fmt_id,
                    " ".join(fmt.keyframe.format(**v).split()),
                    " ".join(fmt.motion.format(**v).split()),
                    rng.choice(fmt.captions),
                    zlib.crc32(pid.encode()) & 0x7FFFFFFF,
                )
            )
    return pieces


def identity_setup(cfg: Config, provider, faces=None) -> tuple[object, np.ndarray, float]:
    """The face analyzer, the character's average face fingerprint, and the identity threshold."""
    if isinstance(provider, MockProvider):
        from . import mock
        from .faces import enroll

        faces = faces or mock.MockFaces()
        refs = cfg.character_dir / "mock_refs"
        refs.mkdir(parents=True, exist_ok=True)
        import cv2

        paths = []
        for i in range(12):
            p = refs / f"ref_{i:02d}.png"
            cv2.imwrite(str(p), mock.render_keyframe(360, 640, 9000 + i, cfg.character.id, set()))
            paths.append(p)
        e = enroll(faces, paths)
        centroid, suggested = e.centroid, e.suggested_threshold
    else:
        from .faces import RealFaces

        faces = faces or RealFaces(cfg.path("models"))
        f = cfg.character_dir / "identity.npy"
        if not f.exists():
            raise RuntimeError("no identity yet: run `ce character check` on your training images")
        centroid, suggested = np.load(f), cfg.character_state()["suggested_threshold"]
    threshold = suggested if cfg.gates.identity_min == "auto" else float(cfg.gates.identity_min)
    return faces, centroid, threshold


async def run_batch(
    cfg: Config, n: int = 24, mix: dict[str, int] | None = None, provider=None, faces=None, log=print
) -> str:
    ledger = Ledger(cfg.path(cfg.workdir) / "ledger.db")
    provider = provider or make_provider(cfg)
    faces, centroid, threshold = identity_setup(cfg, provider, faces)
    check = Checker(cfg, faces, centroid, threshold)
    mix = mix or even_mix(list(cfg.formats), n)
    batch = time.strftime("%Y%m%d-%H%M%S") + ("-mock" if provider.name == "mock" else "")
    pieces = plan(cfg, batch, mix)
    ledger.add_batch(batch, provider.name, len(pieces), mix)
    for p in pieces:
        ledger.add_piece(id=p.id, batch=batch, format=p.format, caption=p.caption, prompt=p.prompt)
    out = cfg.path(cfg.workdir) / "batches" / batch
    slots = asyncio.Semaphore(cfg.concurrency)
    log(f"batch {batch}: {len(pieces)} pieces {mix} on {provider.name}, identity threshold {threshold:.2f}")

    def affordable(stage: str) -> bool:
        return ledger.spent(batch) + provider.price(stage) <= cfg.budget_usd

    async def attempt(stage: str, piece: Piece, fn, *args) -> Render | None:
        try:
            return await fn(*args)
        except RenderFailed as e:
            ledger.add_render(batch, piece.id, stage, e.usd, 0, False, str(e))
        except Exception as e:  # noqa: BLE001 - network or provider error: log it, keep the batch going
            ledger.add_render(batch, piece.id, stage, 0.0, 0, False, f"error: {e}"[:200])
        return None

    async def make(p: Piece) -> None:
        async with slots:
            # 1-2. stills: render a few, keep the one most like the character
            best, reason = None, "no still passed"
            for i in range(cfg.keyframe_candidates):
                if not affordable("keyframe"):
                    return ledger.set_piece(p.id, status="skipped", reason="budget reached")
                r = await attempt("keyframe", p, provider.keyframe, p.prompt, p.seed + i, out / p.id / f"still_{i}")
                if r is None:
                    continue
                v = await asyncio.to_thread(check.keyframe, r.path)
                rid = ledger.add_render(batch, p.id, "keyframe", r.usd, r.seconds, v.ok, v.reason, str(r.path))
                if v.ok and (best is None or v.identity > best[1].identity):
                    best = (r, v, rid)
                reason = v.reason or reason
            if best is None:
                return ledger.set_piece(p.id, status="rejected", reason=reason)
            ledger.mark_kept(best[2])
            ledger.set_piece(p.id, keyframe=str(best[0].path), identity=best[1].identity)

            # 3-4. clip: animate the chosen still, re-roll once if it fails
            for a in range(cfg.clip_attempts):
                if not affordable("clip"):
                    return ledger.set_piece(p.id, status="skipped", reason="budget reached")
                r = await attempt(
                    "clip", p, provider.clip, best[0], p.motion, p.seed + 100 + a, out / p.id / f"clip_{a}"
                )
                if r is None:
                    continue
                v = await asyncio.to_thread(check.clip, r.path)
                rid = ledger.add_render(batch, p.id, "clip", r.usd, r.seconds, v.ok, v.reason, str(r.path))
                if v.ok:
                    ledger.mark_kept(rid)
                    return ledger.set_piece(p.id, status="keeper", clip=str(r.path), reason="")
                reason = v.reason
            ledger.set_piece(p.id, status="rejected", reason=reason)

    await asyncio.gather(*(make(p) for p in pieces))
    ledger.finish_batch(batch)
    return batch
