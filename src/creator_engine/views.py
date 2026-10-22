"""Views in, next batch's format mix out.

After posting, copy each post's views from Instagram/TikTok insights into a CSV:

    file,platform,posted_at,views
    20261023-101500-mirror-walk-003.mp4,instagram,2026-10-24,1840

The next batch gives each format a share of pieces proportional to its median views, with a floor
so no format stops getting tested. Simple enough to explain in one sentence, and it moves volume to
what travels.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

from .ledger import Ledger


def import_views(ledger: Ledger, csv_path: str | Path) -> int:
    n = 0
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            piece = Path(row.get("piece") or row["file"]).stem
            ledger.db.execute(
                "INSERT OR REPLACE INTO views VALUES (?,?,?,?)",
                (piece, row.get("platform", ""), row.get("posted_at", ""), int(float(row["views"]))),
            )
            n += 1
    return n


def next_mix(ledger: Ledger, formats: list[str], n: int, floor: float = 0.1) -> dict[str, int]:
    rows = ledger.q("SELECT p.format, v.views FROM views v JOIN pieces p ON p.id = v.piece")
    medians = {
        f: statistics.median([r["views"] for r in rows if r["format"] == f])
        for f in formats
        if any(r["format"] == f for r in rows)
    }
    if not medians:
        base, extra = divmod(n, len(formats))
        return {f: base + (i < extra) for i, f in enumerate(formats)}
    top = max(medians.values()) or 1
    # untested formats get the floor; tested ones get floor + share of what's left by median views
    weights = {f: floor + (1 - floor) * medians.get(f, 0) / top for f in formats}
    total = sum(weights.values())
    mix = {f: int(n * w / total) for f, w in weights.items()}
    for f in sorted(weights, key=weights.get, reverse=True)[: n - sum(mix.values())]:
        mix[f] += 1
    return {f: q for f, q in mix.items() if q}
