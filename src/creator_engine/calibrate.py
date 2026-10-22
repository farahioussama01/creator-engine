"""Calibrate gate thresholds on labelled renders.

Re-runs every gate (no short-circuit) on a batch's renders, joins the scores with labels, and reports
for each one-sided gate: recall and false-reject rate at the current threshold, and the threshold that
maximises Youden's J (recall - false-reject rate).

Labels come from a CSV (``file,label,issue``: label is good/bad; issue optionally names the gate the
reviewer thinks should catch it) or, for simulated batches, from the defects the simulator injected.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .gates import Artifact, run_gates
from .ops import Ctx, make_engine

# gate -> (threshold param, direction: +1 = higher score is better, -1 = lower is better)
ONE_SIDED = {
    "sharpness": ("min_laplacian_var", +1),
    "skin_texture": ("min_hf", +1),
    "identity": ("min_cosine", +1),
    "temporal_identity": ("min_cosine", +1),
    "flicker": ("max_luma_jump", -1),
}
SIM_DEFECTS = {
    "sharpness": {"blur"},
    "skin_texture": {"waxy"},
    "identity": {"identity_drift"},
    "temporal_identity": {"morph"},
    "flicker": {"flicker"},
}


def _labels_from_csv(path: Path) -> dict[str, tuple[bool, str | None]]:
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[Path(row["file"]).name] = (
                row["label"].strip().lower() == "good",
                (row.get("issue") or "").strip() or None,
            )
    return out


def _best_threshold(good: np.ndarray, bad: np.ndarray, direction: int) -> tuple[float, float, float]:
    vals = np.unique(np.concatenate([good, bad]))
    cands = (vals[:-1] + vals[1:]) / 2 if len(vals) > 1 else vals  # midpoints: never sit on a sample
    best = (float("nan"), -1.0, 0.0, 0.0)
    for t in cands:
        rec = float(np.mean(bad < t)) if direction > 0 else float(np.mean(bad > t))
        fr = float(np.mean(good < t)) if direction > 0 else float(np.mean(good > t))
        if rec - fr > best[1]:
            best = (float(t), rec - fr, rec, fr)
    return best[0], best[2], best[3]


def calibrate(ctx: Ctx, batch_id: str, labels_csv: Path | None = None) -> list[dict[str, Any]]:
    engine = make_engine(ctx)
    labels = _labels_from_csv(labels_csv) if labels_csv else {}
    pieces = {p["id"]: p for p in ctx.store.pieces(batch_id)}
    scores: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for r in ctx.store.q("SELECT * FROM executions WHERE batch_id = ? AND status = 'ok'", [batch_id]):
        path = Path(r["output_path"])
        if not path.exists():
            continue
        piece = pieces[r["piece_id"]]
        art = Artifact(
            piece["id"],
            r["stage"],
            path,
            engine.characters[piece["character"]],
            engine.formats[piece["format"]],
            prompt=piece["keyframe_prompt"] if r["stage"] == "keyframe" else piece["motion_prompt"],
        )
        try:
            _, results = run_gates(art, engine.ctx, full=True)
        except ValueError:
            continue
        truth = json.loads(r["sim_truth"]).get("defects", []) if r["sim_truth"] else None
        for res in results:
            if res.gate not in ONE_SIDED or res.score is None:
                continue
            if path.name in labels:
                good, issue = labels[path.name]
                if not good and issue and issue != res.gate:
                    continue  # a reviewer blamed another gate: not evidence about this one
            elif truth is not None:
                hit = set(truth) & SIM_DEFECTS[res.gate]
                if truth and not hit:
                    continue  # broken in a way this gate is not responsible for
                good = not hit
            else:
                continue
            scores[res.gate].append((float(res.score), good))

    report = []
    for gate, rows in sorted(scores.items()):
        param, direction = ONE_SIDED[gate]
        current = float(ctx.cfg.gate(gate).params.get(param, np.nan))
        good = np.array([s for s, g in rows if g])
        bad = np.array([s for s, g in rows if not g])
        entry: dict[str, Any] = {
            "gate": gate,
            "param": param,
            "n_good": len(good),
            "n_bad": len(bad),
            "current": current,
        }
        if len(good) and len(bad):

            def rates(t: float, good=good, bad=bad, direction=direction) -> tuple[float, float]:
                if direction > 0:
                    return float(np.mean(bad < t)), float(np.mean(good < t))
                return float(np.mean(bad > t)), float(np.mean(good > t))

            entry["recall_at_current"], entry["false_reject_at_current"] = rates(current)
            t, rec, fr = _best_threshold(good, bad, direction)
            entry.update(suggested=t, recall_at_suggested=rec, false_reject_at_suggested=fr)
        report.append(entry)
    return report
