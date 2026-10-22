"""Which formats travel? Thompson sampling over formats, judged against each account's own baseline.

Raw views are a bad reward: a post on a 200k account beats any post on a 2k account. So every post
is scored relative to the median views of the account it went out on, and the reward is binary:
did it beat that baseline at the measurement age? Each (character, format) arm keeps a Beta
posterior; P(best) comes from posterior samples; the next batch's quotas follow P(best) with a
floor for formats still being explored.

  explore  fewer than ``min_posts`` measured posts: guaranteed slots
  scale    P(best) >= 0.5: gets most of the next batch (double down)
  keep     in contention: at least one slot so evidence keeps coming
  kill     enough data, P(best) < ``kill_p`` and below baseline: no more slots
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .config import EngineConfig, FormatSpec
from .store import Store


@dataclass
class ArmStats:
    character: str
    format: str
    posts: int
    wins: int
    median_views: float | None
    median_lift: float | None
    share_rate: float | None
    skip_rate: float | None
    p_best: float = 0.0
    status: str = "explore"
    next_quota: int = 0

    @property
    def mean(self) -> float:
        return (1 + self.wins) / (2 + self.posts)


def _measured(store: Store, character: str, min_age_hours: float) -> list[dict]:
    rows = store.q(
        """
        SELECT p.id, p.account, p.format, m.views, m.shares, m.skip_rate, m.age_hours
        FROM posts p JOIN metrics m ON m.post_id = p.id
        WHERE p.character = ? AND m.age_hours >= ?
          AND m.id = (SELECT MAX(id) FROM metrics WHERE post_id = p.id AND age_hours >= ?)
        """,
        [character, min_age_hours, min_age_hours],
    )
    return [dict(r) for r in rows]


def baselines(cfg: EngineConfig, rows: list[dict]) -> dict[str, float]:
    by_acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_acc[r["account"]].append(r["views"])
    overall = statistics.median([r["views"] for r in rows]) if rows else 1.0
    priors = {a.handle: a.baseline_views for a in cfg.publish.accounts if a.baseline_views}
    return {
        acc: statistics.median(v) if len(v) >= 5 else priors.get(acc, overall) for acc, v in by_acc.items()
    }


def arm_stats(
    cfg: EngineConfig,
    store: Store,
    character: str,
    formats: dict[str, FormatSpec],
    min_age_hours: float = 24.0,
    min_posts: int = 8,
    kill_p: float = 0.05,
    samples: int = 4000,
    seed: int = 0,
) -> list[ArmStats]:
    rows = _measured(store, character, min_age_hours)
    base = baselines(cfg, rows)
    eligible = [
        f for f, spec in formats.items() if not spec.for_characters or character in spec.for_characters
    ]
    arms: dict[str, list[dict]] = {f: [] for f in eligible}
    for r in rows:
        if r["format"] in arms:
            arms[r["format"]].append(r)
    stats = []
    for f, rs in arms.items():
        views = [r["views"] for r in rs]
        lifts = [r["views"] / max(1.0, base.get(r["account"], 1.0)) for r in rs]
        wins = sum(1 for lift in lifts if lift >= 1.0)
        stats.append(
            ArmStats(
                character=character,
                format=f,
                posts=len(rs),
                wins=wins,
                median_views=statistics.median(views) if views else None,
                median_lift=statistics.median(lifts) if lifts else None,
                share_rate=(sum(r["shares"] or 0 for r in rs) / sum(views)) if views and sum(views) else None,
                skip_rate=statistics.mean(r["skip_rate"] for r in rs if r["skip_rate"] is not None)
                if any(r["skip_rate"] is not None for r in rs)
                else None,
            )
        )
    if not stats:
        return stats
    rng = np.random.default_rng(seed)
    draws = np.stack([rng.beta(1 + s.wins, 1 + s.posts - s.wins, size=samples) for s in stats])
    best = np.bincount(draws.argmax(0), minlength=len(stats)) / samples
    for s, p in zip(stats, best, strict=True):
        s.p_best = float(p)
        if s.posts < min_posts:
            s.status = "explore"
        elif s.p_best < kill_p and s.mean < 0.5:
            s.status = "kill"
        elif s.p_best >= 0.5:
            s.status = "scale"
        else:
            s.status = "keep"
    return stats


def allocate(stats: list[ArmStats], n: int, explore_min: int = 2) -> dict[str, int]:
    """Split ``n`` pieces across arms: floors for exploring arms, the rest by P(best)."""
    alive = [s for s in stats if s.status != "kill"]
    if not alive or n <= 0:
        return {}
    quota = {s.format: 0 for s in alive}
    for s in alive:
        if s.status == "explore":
            quota[s.format] = min(explore_min, n)
        elif s.status == "keep":
            quota[s.format] = 1  # still in contention: keep a trickle of evidence coming
    left = n - sum(quota.values())
    if left < 0:  # more exploring arms than slots: give one each, in order
        quota = {s.format: 0 for s in alive}
        for s in alive[:n]:
            quota[s.format] = 1
        left = 0
    weights = np.array([max(s.p_best, 1e-3) for s in alive])
    weights = weights / weights.sum()
    raw = weights * left
    base = np.floor(raw).astype(int)
    rem = left - base.sum()
    order = np.argsort(-(raw - base))
    for i in order[:rem]:
        base[i] += 1
    for s, extra in zip(alive, base, strict=True):
        quota[s.format] += int(extra)
    for s in stats:
        s.next_quota = quota.get(s.format, 0)
    return {f: q for f, q in quota.items() if q > 0}


def simulate_metrics(
    cfg: EngineConfig,
    store: Store,
    appeal: dict[str, float],
    age_hours: float = 48.0,
    seed: int = 0,
    account_base: dict[str, float] | None = None,
) -> int:
    """Demo only: give published-but-unmeasured posts synthetic 48h metrics from hidden format appeal."""
    rng = np.random.default_rng(seed)
    account_base = account_base or {}
    rows = store.q(
        "SELECT * FROM posts WHERE status IN ('published', 'dry_run') AND id NOT IN "
        "(SELECT post_id FROM metrics)"
    )
    for p in rows:
        base = account_base.get(p["account"], 4000.0)
        a = appeal.get(p["format"], 1.0)
        views = float(base * a * rng.lognormal(0, 0.55))
        store.insert(
            "metrics",
            {
                "post_id": p["id"],
                "fetched_at": p["scheduled_at"] + age_hours * 3600,
                "age_hours": age_hours,
                "views": round(views),
                "reach": round(views * 0.8),
                "saved": round(views * 0.004 * a),
                "shares": round(views * 0.006 * a),
                "interactions": round(views * 0.05),
                "avg_watch_ms": 3100 * min(1.6, a),
                "skip_rate": float(np.clip(0.55 - 0.25 * math.log(a), 0.05, 0.95)),
                "simulated": 1,
            },
        )
    return len(rows)
