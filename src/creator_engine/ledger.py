"""Cost per keeper, where the money went, and what would halve it.

Spend is decomposed from execution rows (GPU seconds x the pod's hourly rate) plus pod sessions
(what the pods actually cost while held) plus API calls. Every bucket is a different lever:

  keyframes/clips -> keepers   the irreducible part at this quality setting
  rejected keyframes / clips   yield: better prompts, better LoRA, earlier gates
  best-of-k spares             candidates that passed but lost to a better one
  dead-end keyframes           keyframes whose clips all failed
  model loads & swaps          cold starts and swapping model families on one GPU
  errors / retries             OOMs and preempted pods
  idle GPU                     pods billed while nothing ran
  judge API                    Claude vision calls
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from .config import EngineConfig, load_prices
from .store import Store

BUCKETS = [
    "keyframes → keepers",
    "clips → keepers",
    "rejected keyframes",
    "best-of-k spares",
    "rejected clips",
    "dead-end keyframes",
    "model loads & swaps",
    "errors / retries",
    "idle GPU",
    "judge API",
]


@dataclass
class Lever:
    name: str
    action: str
    per_keeper: float
    saving_pct: float


@dataclass
class Ledger:
    batch_id: str
    character: str
    profile: str
    simulated: bool
    pieces: int
    keepers: int
    status_counts: dict[str, int]
    buckets: dict[str, float]
    total_usd: float
    per_keeper: float | None
    sessions_recorded: bool
    gpu_seconds: dict[str, float]
    stage: dict[str, dict[str, Any]]
    funnel: list[dict[str, Any]]
    hosted: list[dict[str, Any]] = field(default_factory=list)
    levers: list[Lever] = field(default_factory=list)
    combined: Lever | None = None
    gate_eval: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["levers"] = [lv.__dict__ for lv in self.levers]
        d["combined"] = self.combined.__dict__ if self.combined else None
        return d


def _overheads(rows: list[Any]) -> dict[int, float]:
    """Seconds above the steady-state median for each execution (cold loads, model swaps)."""
    by_group: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            by_group[(r["stage"], r["gpu"])].append(r)
    over: dict[int, float] = {}
    for group in by_group.values():
        med = statistics.median(r["gpu_seconds"] for r in group)
        for r in group:
            extra = r["gpu_seconds"] - med
            over[r["id"]] = extra if r["gpu_seconds"] > 1.25 * med else 0.0
    return over


def build_ledger(cfg: EngineConfig, store: Store, batch_id: str) -> Ledger:
    batch = store.batch(batch_id)
    profile = cfg.profile(batch["profile"])
    pieces = store.pieces(batch_id)
    status = {p["id"]: p["status"] for p in pieces}
    keepers = sum(1 for s in status.values() if s == "keeper")
    rows = store.q("SELECT * FROM executions WHERE batch_id = ?", [batch_id])
    over = _overheads(rows)

    buckets = dict.fromkeys(BUCKETS, 0.0)
    secs: dict[str, float] = defaultdict(float)
    for r in rows:
        rate = (r["usd_per_hour"] or 0.0) / 3600
        if r["status"] != "ok":
            buckets["errors / retries"] += r["usd"] or 0.0
            secs["errors / retries"] += r["gpu_seconds"] or 0.0
            continue
        o = over.get(r["id"], 0.0)
        buckets["model loads & swaps"] += o * rate
        secs["model loads & swaps"] += o
        work = (r["gpu_seconds"] - o) * rate
        keeper = status.get(r["piece_id"]) == "keeper"
        if r["stage"] == "keyframe":
            key = {"rejected": "rejected keyframes", "unused": "best-of-k spares"}.get(
                r["outcome"], "keyframes → keepers" if keeper else "dead-end keyframes"
            )
        else:
            key = "clips → keepers" if r["outcome"] == "kept" else "rejected clips"
        buckets[key] += work
        secs[key] += r["gpu_seconds"] - o

    exec_usd = sum((r["usd"] or 0.0) for r in rows)
    sessions = store.q("SELECT * FROM spend WHERE batch_id = ? AND kind = 'pod_session'", [batch_id])
    session_usd = sum(s["usd"] for s in sessions)
    if sessions:
        buckets["idle GPU"] = max(0.0, session_usd - exec_usd)
        secs["idle GPU"] = max(
            0.0, sum(s["seconds"] for s in sessions) - sum(r["gpu_seconds"] or 0 for r in rows)
        )
    judge = store.one(
        "SELECT COALESCE(SUM(usd),0) s FROM spend WHERE batch_id = ? AND kind = 'judge'", [batch_id]
    )
    buckets["judge API"] = float(judge["s"])
    total = sum(buckets.values())

    stage: dict[str, dict[str, Any]] = {}
    for st in ("keyframe", "video"):
        ok = [r for r in rows if r["stage"] == st and r["status"] == "ok"]
        rejected = sum(1 for r in ok if r["outcome"] == "rejected")
        stage[st] = {
            "renders": len(ok),
            "errors": sum(1 for r in rows if r["stage"] == st and r["status"] != "ok"),
            "pass_rate": (len(ok) - rejected) / len(ok) if ok else None,
            "median_seconds": statistics.median(r["gpu_seconds"] for r in ok) if ok else None,
            "per_keeper": len(ok) / keepers if keepers else None,
        }

    funnel_rows = store.q(
        "SELECT stage, gate, SUM(passed = 0) AS fails, SUM(passed = 1) AS passes, COUNT(*) AS n, "
        "SUM(passed IS NULL) AS skipped FROM gate_results WHERE batch_id = ? GROUP BY stage, gate",
        [batch_id],
    )
    funnel = [dict(r) for r in funnel_rows]

    led = Ledger(
        batch_id=batch_id,
        character=batch["character"],
        profile=batch["profile"],
        simulated=bool(batch["simulated"]),
        pieces=len(pieces),
        keepers=keepers,
        status_counts=dict(Counter(status.values())),
        buckets=buckets,
        total_usd=total,
        per_keeper=total / keepers if keepers else None,
        sessions_recorded=bool(sessions),
        gpu_seconds=dict(secs),
        stage=stage,
        funnel=funnel,
    )
    prices = load_prices(cfg)
    led.hosted = hosted_comparison(prices, profile.seconds, led.per_keeper)
    led.levers, led.combined = halving_plan(cfg, prices, led)
    led.gate_eval = gate_eval(rows, store, batch_id) if led.simulated else []
    return led


def hosted_comparison(prices: dict[str, Any], seconds: float, ours: float | None) -> list[dict[str, Any]]:
    attempts = float(prices.get("hosted_attempts_per_keeper", 2.0))
    out = []
    for h in prices.get("hosted_video", []):
        per = h["usd_per_second"] * seconds * attempts
        out.append(
            {
                "name": h["name"],
                "usd_per_second": h["usd_per_second"],
                "per_keeper": per,
                "x_ours": per / ours if ours else None,
                "source": h.get("source", ""),
            }
        )
    return out


def halving_plan(cfg: EngineConfig, prices: dict[str, Any], led: Ledger) -> tuple[list[Lever], Lever | None]:
    if not led.keepers or not led.total_usd:
        return [], None
    k, total, b = led.keepers, led.total_usd, led.buckets
    base = total / k
    levers: list[Lever] = []

    def add(name: str, action: str, new_total: float) -> None:
        levers.append(Lever(name, action, new_total / k, 1 - new_total / total))

    add(
        "Idle GPU → ~0",
        "Keep 2 prompts in flight per GPU, tear pods down when the queue drains, models on a "
        "network volume so a fresh pod boots in seconds.",
        total - 0.9 * b["idle GPU"],
    )
    add(
        "No model swaps",
        "Phased scheduling or stage-dedicated GPUs (keyframe pod / video pods); never /free between prompts.",
        total - 0.8 * b["model loads & swaps"],
    )
    pv = led.stage["video"]["pass_rate"]
    if pv and pv < 0.98:
        pv2 = 1 - (1 - pv) / 2
        video_spend = b["clips → keepers"] + b["rejected clips"] + b["dead-end keyframes"]
        add(
            f"Halve clip rejects ({pv:.0%} → {pv2:.0%} pass)",
            "Move clip-killing defects upstream: stricter "
            "keyframe identity threshold, motion prompts without head turns, 6-step schedule on hero formats.",
            total - video_spend * (1 - pv / pv2),
        )
    pk = led.stage["keyframe"]["pass_rate"]
    if pk and pk < 0.95:
        kf_spend = b["rejected keyframes"]
        add(
            f"Halve keyframe rejects ({pk:.0%} pass)",
            "Retrain or re-weight the LoRA on the failure cluster the "
            "funnel shows; fix prompts that cause the top gate failure.",
            total - kf_spend / 2,
        )

    # Cheaper $/clip hardware (rate / speed), using the price table; speeds are assumptions until benchmarked
    speeds = prices.get("sim_gpu_speed", {})
    rates = prices.get("gpu_hourly", {})
    cur = [e for e in cfg.endpoints if "video" in e.stages]
    if cur and speeds and rates:
        cur_cost = min(e.usd_per_hour / speeds.get(e.gpu, 1.0) for e in cur)
        best_name, best_cost = None, cur_cost
        for gpu, r in rates.items():
            hourly = min(
                v
                for kk, v in r.items()
                if kk in ("community", "secure", "spot") and isinstance(v, int | float)
            )
            c = hourly / speeds.get(gpu, 1.0)
            if c < best_cost * 0.97:
                best_name, best_cost = gpu, c
        if best_name:
            gpu_part = total - b["judge API"]
            add(
                f"Move video to {best_name}",
                "Lower $ per clip at the listed rate and assumed speed - confirm with "
                "`ce bench` on that GPU before switching.",
                total - gpu_part * (1 - best_cost / cur_cost),
            )

    levers.sort(key=lambda lv: lv.saving_pct, reverse=True)
    remaining = total
    for lv in levers:
        remaining *= 1 - lv.saving_pct
    combined = Lever(
        "All of the above",
        "Applied together (each on what the previous left).",
        remaining / k,
        1 - remaining / total,
    )
    _ = base
    return levers, combined


def gate_eval(rows: list[Any], store: Store, batch_id: str) -> list[dict[str, Any]]:
    """Simulator only: how well the gates caught injected defects (recall) and spared clean renders."""
    rejected_ids = {
        r["execution_id"]
        for r in store.q(
            "SELECT DISTINCT execution_id FROM gate_results WHERE batch_id = ? AND passed = 0", [batch_id]
        )
    }
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    clean = {"keyframe": [0, 0], "video": [0, 0]}
    for r in rows:
        if r["status"] != "ok" or not r["sim_truth"]:
            continue
        defects = json.loads(r["sim_truth"]).get("defects", [])
        caught = r["id"] in rejected_ids
        if defects:
            for d in defects:
                per[d][0] += 1
                per[d][1] += int(caught)
        else:
            clean[r["stage"]][0] += 1
            clean[r["stage"]][1] += int(caught)
    out = [
        {"defect": d, "injected": n, "caught": c, "recall": c / n if n else None}
        for d, (n, c) in sorted(per.items())
    ]
    for st, (n, c) in clean.items():
        if n:
            out.append(
                {
                    "defect": f"none ({st})",
                    "injected": n,
                    "caught": c,
                    "recall": None,
                    "false_reject_rate": c / n,
                }
            )
    return out
