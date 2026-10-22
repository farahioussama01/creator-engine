"""The batch runner: keyframes -> gates -> video -> gates -> keepers, under a budget.

Phased scheduling (default) renders and gates every keyframe before any video starts. That keeps one
model family resident per GPU (no swap penalty), lets the best-of-k selection see all candidates,
and means a clip is only ever rendered from a keyframe that already passed identity, exposure,
focus, skin and sign-text checks.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import CharacterSpec, EngineConfig, FormatSpec, RenderProfile, load_characters, load_formats
from .gates import KEYFRAME, VIDEO, Artifact, GateContext, GateResult, run_gates
from .identity import load_centroids
from .pool import ExecOutcome, ExecRequest, Pool
from .sim import timing
from .store import Store
from .workflow import fill, load_template

WAN_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)  # Wan's default negative prompt, from the official ComfyUI template


class Budget:
    """Caps GPU execution spend for a batch. Reserve before submitting, settle after."""

    def __init__(self, limit_usd: float, spent: float = 0.0):
        self.limit, self.spent, self.reserved = limit_usd, spent, 0.0
        self.exhausted = False

    def reserve(self, est: float) -> bool:
        if self.spent + self.reserved + est > self.limit:
            self.exhausted = True
            return False
        self.reserved += est
        return True

    def settle(self, est: float, actual: float) -> None:
        self.reserved -= est
        self.spent += actual


@dataclass
class Candidate:
    exec_id: int | None
    passed: bool
    identity: float | None
    path: Path | None
    results: list[GateResult] = field(default_factory=list)
    error: str | None = None


def first_failure(results: list[GateResult]) -> str:
    for r in results:
        if r.passed is False:
            return f"{r.gate}: {r.reason}"
    return ""


class Engine:
    def __init__(self, cfg: EngineConfig, store: Store, analyzer, ocr=None, judge=None):
        self.cfg, self.store = cfg, store
        self.characters: dict[str, CharacterSpec] = load_characters(cfg)
        self.formats: dict[str, FormatSpec] = load_formats(cfg)
        self.templates = {k: load_template(cfg.path(v)) for k, v in cfg.workflows.items()}
        self.ctx = GateContext(
            cfg=cfg, analyzer=analyzer, centroids=load_centroids(cfg), ocr=ocr, judge=judge
        )
        self._gate_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gates")
        self._observed: dict[str, list[float]] = {KEYFRAME: [], VIDEO: []}
        self.log: list[str] = []

    # ------------------------------------------------------------ estimates
    def estimate_seconds(self, stage: str, profile: RenderProfile) -> float:
        seen = self._observed[stage]
        if len(seen) >= 3:
            return statistics.median(seen) * 1.1
        if stage == KEYFRAME:
            return timing.keyframe_seconds(profile.kf_width, profile.kf_height, profile.kf_steps) + 2
        return (
            timing.video_seconds(
                profile.vid_width, profile.vid_height, profile.frames, profile.vid_steps, profile.vid_cfg
            )
            + 5
        )

    def _max_rate(self, stage: str) -> float:
        rates = [e.usd_per_hour for e in self.cfg.endpoints if stage in e.stages] or [1.0]
        return max(rates)

    # ------------------------------------------------------------ public
    async def run_batch(self, batch_id: str, pool: Pool | None = None) -> dict[str, Any]:
        batch = self.store.batch(batch_id)
        profile = self.cfg.profile(batch["profile"])
        character = self.characters[batch["character"]]
        already = self.store.one(
            "SELECT COALESCE(SUM(usd),0) s FROM executions WHERE batch_id = ?", [batch_id]
        )["s"]
        cap = min(
            self.cfg.budget.max_usd_per_batch,
            max(0.0, self.cfg.budget.max_usd_per_day - self.store.spend_today()) + already,
        )
        budget = Budget(cap, already)
        own_pool = pool is None
        pool = pool or Pool(self.cfg.endpoints, poll_seconds=self.cfg.poll_seconds)
        if own_pool:
            problems = await pool.start()
            self.log.extend(problems)
        self.store.update("batches", "id", batch_id, status="running")
        t0 = time.time()
        try:
            todo = [p for p in self.store.pieces(batch_id) if p["status"] in ("planned", "keyframe_ok")]
            if self.cfg.scheduling == "phased":
                await asyncio.gather(
                    *(
                        self._keyframes(pool, budget, character, profile, p)
                        for p in todo
                        if p["status"] == "planned"
                    )
                )
                ready = [p for p in self.store.pieces(batch_id) if p["status"] == "keyframe_ok"]
                await asyncio.gather(*(self._videos(pool, budget, character, profile, p) for p in ready))
            else:

                async def whole(p):
                    if p["status"] == "planned":
                        await self._keyframes(pool, budget, character, profile, p)
                    fresh = self.store.one("SELECT * FROM pieces WHERE id = ?", [p["id"]])
                    if fresh["status"] == "keyframe_ok":
                        await self._videos(pool, budget, character, profile, fresh)

                await asyncio.gather(*(whole(p) for p in todo))
        finally:
            judge = self.ctx.judge
            if judge is not None and getattr(judge, "spent", None):
                self.store.add_spend(
                    batch_id,
                    "judge",
                    self.cfg.judge.model,
                    0.0,
                    float(sum(judge.spent)),
                    calls=len(judge.spent),
                )
                judge.spent.clear()
            if own_pool:
                for bill in await pool.close():
                    self.store.add_spend(
                        batch_id,
                        "pod_session",
                        bill.endpoint,
                        bill.seconds,
                        bill.usd,
                        gpu=bill.gpu,
                        busy_seconds=bill.busy_seconds,
                        executed=bill.executed,
                        simulated_clock=bill.simulated_clock,
                    )
        status = "budget_stopped" if budget.exhausted else "done"
        self.store.update("batches", "id", batch_id, status=status, note=f"wall {time.time() - t0:.0f}s")
        counts = Counter(p["status"] for p in self.store.pieces(batch_id))
        return {
            "batch": batch_id,
            "status": status,
            "pieces": dict(counts),
            "spent_gpu_usd": round(budget.spent, 4),
        }

    # ------------------------------------------------------------ stages
    def _record(self, out: ExecOutcome, piece: Any, stage: str, attempt: int, seed: int) -> int | None:
        for f in out.failures:
            self.store.add_execution(
                piece_id=piece["id"],
                batch_id=piece["batch_id"],
                stage=stage,
                attempt=attempt,
                endpoint=f.endpoint,
                gpu=f.gpu,
                usd_per_hour=f.usd_per_hour,
                prompt_id=None,
                seed=seed,
                gpu_seconds=f.gpu_seconds,
                wall_seconds=None,
                usd=f.gpu_seconds * f.usd_per_hour / 3600,
                status="error",
                outcome="error",
                output_path=None,
                error=f.error,
                sim_truth=None,
                started_at=None,
                finished_at=time.time(),
            )
        if not out.ok:
            return None
        self._observed[stage].append(out.gpu_seconds)
        return self.store.add_execution(
            piece_id=piece["id"],
            batch_id=piece["batch_id"],
            stage=stage,
            attempt=attempt,
            endpoint=out.endpoint,
            gpu=out.gpu,
            usd_per_hour=out.usd_per_hour,
            prompt_id=out.prompt_id,
            seed=seed,
            gpu_seconds=out.gpu_seconds,
            wall_seconds=out.wall_seconds,
            usd=out.usd,
            status="ok",
            outcome="pending",
            output_path=str(out.output),
            error=None,
            sim_truth=json.dumps(out.sim_truth) if out.sim_truth else None,
            started_at=out.started_at,
            finished_at=out.finished_at,
        )

    async def _gate(self, art: Artifact) -> tuple[bool, list[GateResult]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._gate_pool, run_gates, art, self.ctx)

    def _piece_dir(self, piece: Any) -> Path:
        return self.cfg.path(self.cfg.workdir) / "batches" / piece["batch_id"] / piece["id"]

    async def _keyframe(self, pool, budget, character, profile, piece, rnd: int, idx: int) -> Candidate:
        fmt = self.formats[piece["format"]]
        seed = piece["kf_seed"] + 7919 * rnd + idx
        values = {
            "lora_name": character.lora,
            "lora_strength": character.lora_strength,
            "positive": piece["keyframe_prompt"],
            "seed": seed,
            "steps": profile.kf_steps,
            "cfg": profile.kf_cfg,
            "width": profile.kf_width,
            "height": profile.kf_height,
            "filename_prefix": f"ce/{piece['batch_id']}/{piece['id']}/kf",
        }
        est = self.estimate_seconds(KEYFRAME, profile) * self._max_rate(KEYFRAME) / 3600
        if not budget.reserve(est):
            return Candidate(None, False, None, None, error="budget")
        tpl = self.templates["keyframe"]
        out = await pool.execute(
            ExecRequest(KEYFRAME, lambda _n: fill(tpl, values), self._piece_dir(piece), f"kf_r{rnd}_{idx}")
        )
        budget.settle(est, out.usd + sum(f.gpu_seconds * f.usd_per_hour / 3600 for f in out.failures))
        exec_id = self._record(out, piece, KEYFRAME, rnd * 100 + idx, seed)
        if exec_id is None:
            return Candidate(None, False, None, None, error=out.error)
        art = Artifact(piece["id"], KEYFRAME, out.output, character, fmt, prompt=piece["keyframe_prompt"])
        passed, results = await self._gate(art)
        self.store.add_gate_results(exec_id, piece["id"], piece["batch_id"], KEYFRAME, results)
        if not passed:
            self.store.update("executions", "id", exec_id, outcome="rejected")
        return Candidate(exec_id, passed, art.cache.get("identity"), out.output, results)

    async def _keyframes(self, pool, budget, character, profile, piece) -> None:
        reasons: list[str] = []
        for rnd in range(profile.keyframe_rounds):
            cands = await asyncio.gather(
                *(
                    self._keyframe(pool, budget, character, profile, piece, rnd, i)
                    for i in range(profile.keyframe_candidates)
                )
            )
            if any(c.error == "budget" for c in cands) and not any(c.passed for c in cands):
                self.store.set_piece(piece["id"], status="skipped_budget")
                return
            passing = [c for c in cands if c.passed]
            reasons += [first_failure(c.results) or (c.error or "") for c in cands if not c.passed]
            if passing:
                best = max(passing, key=lambda c: c.identity if c.identity is not None else 0.0)
                for c in passing:
                    self.store.update(
                        "executions", "id", c.exec_id, outcome="selected" if c is best else "unused"
                    )
                self.store.set_piece(
                    piece["id"], status="keyframe_ok", keyframe_path=str(best.path), identity=best.identity
                )
                return
        top = Counter(r.split(":")[0] for r in reasons if r).most_common(1)
        self.store.set_piece(
            piece["id"],
            status="rejected_keyframe",
            reject_reason=next((r for r in reasons if top and r.startswith(top[0][0])), "no keyframe"),
        )

    async def _videos(self, pool, budget, character, profile, piece) -> None:
        fmt = self.formats[piece["format"]]
        kf = Path(piece["keyframe_path"])
        last_reason = ""
        for attempt in range(1, profile.max_video_attempts + 1):
            seed = piece["vid_seed"] + 104729 * attempt
            values = {
                "positive": piece["motion_prompt"],
                "negative": (WAN_NEGATIVE + " " + character.negative).strip(),
                "width": profile.vid_width,
                "height": profile.vid_height,
                "length": profile.frames,
                "fps": profile.fps,
                "seed": seed,
                "steps": profile.vid_steps,
                "cfg": profile.vid_cfg,
                "split_step": profile.split_step,
                "lightning_strength": 1.0 if profile.lightning else 0.0,
                "filename_prefix": f"ce/{piece['batch_id']}/{piece['id']}/clip",
            }
            est = self.estimate_seconds(VIDEO, profile) * self._max_rate(VIDEO) / 3600
            if not budget.reserve(est):
                self.store.set_piece(piece["id"], status="skipped_budget", reject_reason="budget cap reached")
                return
            tpl = self.templates["video"]
            out = await pool.execute(
                ExecRequest(
                    VIDEO,
                    lambda names, tpl=tpl, values=values: fill(
                        tpl, {**values, "start_image": names["start_image"]}
                    ),
                    self._piece_dir(piece),
                    f"clip_a{attempt}",
                    uploads={"start_image": kf},
                )
            )
            budget.settle(est, out.usd + sum(f.gpu_seconds * f.usd_per_hour / 3600 for f in out.failures))
            exec_id = self._record(out, piece, VIDEO, attempt, seed)
            if exec_id is None:
                last_reason = f"render failed: {out.error}"
                continue
            art = Artifact(piece["id"], VIDEO, out.output, character, fmt, prompt=piece["motion_prompt"])
            passed, results = await self._gate(art)
            self.store.add_gate_results(exec_id, piece["id"], piece["batch_id"], VIDEO, results)
            if passed:
                self.store.update("executions", "id", exec_id, outcome="kept")
                self.store.set_piece(
                    piece["id"], status="keeper", video_path=str(out.output), reject_reason=None
                )
                return
            self.store.update("executions", "id", exec_id, outcome="rejected")
            last_reason = first_failure(results)
        self.store.set_piece(piece["id"], status="rejected_video", reject_reason=last_reason)
