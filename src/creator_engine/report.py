"""Turn the ledger into the four numbers that matter: pipeline, volume, cost per piece, views.

`answer()` writes them as plain sentences you can paste into an application. It refuses for mock
batches, so a cartoon run can never end up quoted as a real one.
"""

from __future__ import annotations

import base64
import html
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2

from .config import Config
from .ledger import Ledger


def batch_numbers(ledger: Ledger, batch: str) -> dict:
    b = ledger.one("SELECT * FROM batches WHERE id = ?", batch)
    if b is None:
        raise KeyError(f"no batch {batch}")
    pieces = ledger.q("SELECT * FROM pieces WHERE batch = ?", batch)
    renders = ledger.q("SELECT r.*, p.status FROM renders r JOIN pieces p ON p.id = r.piece WHERE r.batch = ?", batch)
    keepers = [p for p in pieces if p["status"] == "keeper"]
    spend = sum(r["usd"] for r in renders)
    useful = sum(r["usd"] for r in renders if r["kept"] and r["status"] == "keeper")
    wall = (b["finished"] or b["started"]) - b["started"]
    return {
        "batch": batch,
        "provider": b["provider"],
        "mock": b["provider"] == "mock",
        "planned": b["planned"],
        "keepers": len(keepers),
        "stills": sum(1 for r in renders if r["stage"] == "keyframe"),
        "clips": sum(1 for r in renders if r["stage"] == "clip"),
        "spend": spend,
        "spend_stills": sum(r["usd"] for r in renders if r["stage"] == "keyframe"),
        "spend_clips": sum(r["usd"] for r in renders if r["stage"] == "clip"),
        "waste_share": (spend - useful) / spend if spend else 0.0,
        "per_keeper": spend / len(keepers) if keepers else None,
        "wall_minutes": wall / 60,
        "keepers_per_hour": len(keepers) / (wall / 3600) if wall > 0 and b["provider"] != "mock" else None,
        "identity_median": statistics.median([p["identity"] for p in keepers]) if keepers else None,
        "top_rejects": Counter(
            (p["reason"] or "").split(" (")[0] for p in pieces if p["status"] == "rejected"
        ).most_common(3),
    }


def views_numbers(ledger: Ledger) -> dict | None:
    rows = ledger.q("SELECT v.*, p.format FROM views v JOIN pieces p ON p.id = v.piece")
    if not rows:
        return None
    by_fmt: dict[str, list[int]] = {}
    for r in rows:
        by_fmt.setdefault(r["format"], []).append(r["views"])
    medians = {f: statistics.median(v) for f, v in by_fmt.items()}
    dates = sorted(r["posted_at"] for r in rows if r["posted_at"])
    days = (datetime.fromisoformat(dates[-1]) - datetime.fromisoformat(dates[0])).days + 1 if dates else None
    return {
        "posts": len(rows),
        "total": sum(r["views"] for r in rows),
        "median": statistics.median(r["views"] for r in rows),
        "days": days,
        "platforms": sorted({r["platform"] for r in rows if r["platform"]}),
        "by_format": dict(sorted(medians.items(), key=lambda kv: -kv[1])),
    }


def money(x: float | None) -> str:
    return "-" if x is None else (f"${x:.3f}" if x < 1 else f"${x:,.2f}")


def answer(cfg: Config, ledger: Ledger, batch: str) -> str:
    n = batch_numbers(ledger, batch)
    if n["mock"]:
        return "MOCK batch: cartoons priced like fal, not real renders. Run with provider: fal for real numbers."
    st = cfg.character_state()
    setup = ledger.one("SELECT COALESCE(SUM(usd),0) s FROM setup")["s"]
    where = "fal.ai (hosted, no GPU of my own)" if n["provider"] == "fal" else "my own ComfyUI on a rented GPU"
    lines = [
        f"Pipeline: Z-Image Turbo + a character LoRA I trained on {st.get('images', '?')} images "
        f"({st.get('steps', '?')} steps) -> 4 checks on each still (light, one face, focus, same face as the "
        f"training set) -> Wan 2.2 A14B image-to-video, {cfg.clip_resolution}, ~5 s -> 2 checks on each clip "
        f"(motion, same face across 8 frames). Rendered on {where}.",
        f"Volume: {n['planned']} pieces planned -> {n['stills']} stills and {n['clips']} clips rendered -> "
        f"{n['keepers']} keepers in {n['wall_minutes']:.0f} min with {cfg.concurrency} renders in parallel "
        f"({n['keepers_per_hour'] or 0:.0f} keepers/hour).",
        f"Cost per piece: {money(n['per_keeper'])} per keeper all-in ({money(n['spend'])} / {n['keepers']}): "
        f"stills {money(n['spend_stills'])}, clips {money(n['spend_clips'])}; {n['waste_share']:.0%} of spend went "
        f"on rejected renders. One-off setup (dataset + LoRA training): {money(setup)}. For comparison, "
        f"{cfg.prices.hosted_reference_name}: {money(cfg.prices.hosted_reference_usd)}.",
    ]
    v = views_numbers(ledger)
    if v:
        fm = list(v["by_format"].items())
        lines.append(
            f"Views: {v['posts']} posts on {', '.join(v['platforms']) or 'social'} over {v['days']} days: "
            f"{v['total']:,} views, median {v['median']:,.0f} per post; best format {fm[0][0]} "
            f"(median {fm[0][1]:,.0f}), weakest {fm[-1][0]} (median {fm[-1][1]:,.0f})."
        )
    else:
        lines.append("Views: not posted yet (export with `ce export`, post, then `ce views views.csv`).")
    return "\n\n".join(lines)


def _thumb(path: str | None, width: int = 150) -> str:
    if not path or not Path(path).exists():
        return ""
    if path.endswith(".mp4"):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
        ok, img = cap.read()
        cap.release()
    else:
        img = cv2.imread(path)
        ok = img is not None
    if not ok:
        return ""
    img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", img)[1].tobytes()).decode()


def write_html(cfg: Config, ledger: Ledger, batch: str) -> Path:
    n = batch_numbers(ledger, batch)
    pieces = ledger.q("SELECT * FROM pieces WHERE batch = ? ORDER BY id", batch)
    e = html.escape

    def tile(p) -> str:
        src = _thumb(p["clip"] or p["keyframe"])
        img = f"<img src='{src}'>" if src else "<div class=none>no image</div>"
        cap = p["format"] if p["status"] == "keeper" else (p["reason"] or p["status"])
        return f"<figure>{img}<figcaption>{e(cap)}</figcaption></figure>"

    tag = "<span class=mock>MOCK</span>" if n["mock"] else ""
    stats = [
        ("Cost per keeper", money(n["per_keeper"])),
        ("Keepers", f"{n['keepers']} / {n['planned']}"),
        ("Spend", money(n["spend"])),
        ("Keepers per hour", f"{n['keepers_per_hour']:.0f}" if n["keepers_per_hour"] else "-"),
        ("Spend on rejects", f"{n['waste_share']:.0%}"),
    ]
    body = "".join(f"<div class=stat><div class=l>{k}</div><div class=v>{v}</div></div>" for k, v in stats)
    answer_html = "".join(f"<p>{e(line)}</p>" for line in answer(cfg, ledger, batch).split("\n\n"))
    page = f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Batch {e(batch)}</title><style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#111;background:#fafaf8}}
.mock{{background:#d03b3b;color:#fff;border-radius:6px;padding:2px 8px;font-size:12px;margin-left:8px}}
.stats{{display:flex;flex-wrap:wrap;gap:12px}}.stat{{background:#fff;border:1px solid #e5e4de;border-radius:10px;padding:12px 16px}}
.l{{font-size:12px;color:#555}}.v{{font-size:24px;font-weight:600}}
.answer{{background:#fff;border:1px solid #e5e4de;border-radius:10px;padding:4px 16px;margin:16px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:10px}}
figure{{margin:0;font-size:12px;color:#555}}img{{width:100%;border-radius:8px}}
.none{{height:200px;background:#eee;border-radius:8px}}</style>
<h1>Batch {e(batch)}{tag}</h1><div class=stats>{body}</div>
<div class=answer>{answer_html}</div>
<h2>Keepers</h2><div class=grid>{"".join(tile(p) for p in pieces if p["status"] == "keeper")}</div>
<h2>Rejected, with the check that stopped them</h2><div class=grid>{"".join(tile(p) for p in pieces if p["status"] != "keeper") or "<p>None.</p>"}</div>"""
    out = cfg.path(cfg.workdir) / "reports" / f"{batch}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out
