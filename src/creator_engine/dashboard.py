"""A single self-contained HTML page per batch: cost, yield, gates, contact sheet, experiments, queue.

No JavaScript framework, no CDN: inline SVG charts, a few lines of hover-tooltip script, thumbnails
embedded as small JPEGs. Opens from disk, attaches to an email, survives being sent to a founder.
"""

from __future__ import annotations

import base64
import html
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from .config import EngineConfig, load_formats
from .experiments import allocate, arm_stats
from .ledger import BUCKETS, Ledger, build_ledger
from .store import Store

USEFUL = {"keyframes → keepers", "clips → keepers"}

CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--page:#f9f9f7;--text-primary:#0b0b0b;--text-secondary:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--series-1:#2a78d6;--series-2:#eb6834;
--good:#0ca30c;--warning:#fab219;--serious:#ec835a;--critical:#d03b3b;--good-text:#006300}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
--surface-1:#1a1a19;--page:#0d0d0d;--text-primary:#f0efec;--text-secondary:#c3c2b7;--grid:#2c2c2a;--axis:#383835;
--border:rgba(255,255,255,.10);--series-1:#3987e5;--series-2:#d95926;--good-text:#0ca30c}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;--text-primary:#f0efec;
--text-secondary:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--series-1:#3987e5;
--series-2:#d95926;--good-text:#0ca30c}
*{box-sizing:border-box}
body{margin:0;background:var(--page)}
.viz-root{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--text-primary);background:var(--page);
padding:24px 16px 64px;max-width:1120px;margin:0 auto;line-height:1.45}
h1{font-size:22px;margin:0 0 4px;font-weight:650}
h2{font-size:16px;margin:0 0 4px;font-weight:620}
.sub{color:var(--text-secondary);font-size:13px;margin:0 0 12px}
.badge{display:inline-block;font-size:11px;font-weight:600;letter-spacing:.04em;padding:2px 8px;border-radius:999px;
border:1px solid var(--border);color:var(--text-secondary);margin-left:8px;vertical-align:middle}
.grid{display:grid;gap:16px}
.g4{grid-template-columns:repeat(auto-fit,minmax(200px,1fr))}
.g2{grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:16px;min-width:0}
.hero{font-size:52px;font-weight:650;line-height:1.05;margin:4px 0}
.label{font-size:12.5px;color:var(--text-secondary)}
.value{font-size:26px;font-weight:620;margin-top:2px}
.note{font-size:12.5px;color:var(--text-secondary)}
.legend{display:flex;gap:16px;font-size:12.5px;color:var(--text-secondary);margin:4px 0 8px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
table{border-collapse:collapse;width:100%;font-size:13px}
th{text-align:left;font-weight:600;color:var(--text-secondary);border-bottom:1px solid var(--axis);padding:6px 8px}
td{border-bottom:1px solid var(--grid);padding:6px 8px;vertical-align:top}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.chip{display:inline-flex;align-items:center;gap:5px;font-size:12px;font-weight:600;color:var(--text-primary)}
.chip i{width:8px;height:8px;border-radius:50%;display:inline-block}
.sheet{display:grid;grid-template-columns:repeat(auto-fill,minmax(112px,1fr));gap:10px}
.sheet figure{margin:0;font-size:11.5px;color:var(--text-secondary)}
.sheet img{width:100%;border-radius:8px;display:block;border:1px solid var(--border)}
.sheet figcaption{margin-top:4px;overflow-wrap:anywhere}
svg text{font-family:inherit}
.mark{cursor:default;outline:none}
.mark:hover,.mark:focus{opacity:.82}
#tip{position:fixed;pointer-events:none;background:var(--surface-1);border:1px solid var(--border);border-radius:8px;
padding:6px 9px;font-size:12.5px;box-shadow:0 4px 14px rgba(0,0,0,.12);display:none;z-index:9;max-width:280px}
#tip b{display:block;font-size:14px}
section{margin-top:28px}
@media (max-width:520px){.hero{font-size:44px}.value{font-size:22px}}
"""

TIP_JS = """
(()=>{const t=document.getElementById('tip');
function show(e){const m=e.target.closest('[data-tip]');if(!m){t.style.display='none';return}
const [v,l]=m.dataset.tip.split('|');t.replaceChildren();const b=document.createElement('b');b.textContent=v;
const s=document.createElement('span');s.textContent=l||'';t.append(b,s);t.style.display='block';
const r=m.getBoundingClientRect();const x=(e.clientX||r.right)+12,y=(e.clientY||r.top)+12;
t.style.left=Math.min(x,innerWidth-t.offsetWidth-8)+'px';t.style.top=Math.min(y,innerHeight-t.offsetHeight-8)+'px'}
document.addEventListener('pointermove',show);document.addEventListener('focusin',show);
document.addEventListener('focusout',()=>t.style.display='none')})();
"""


def _e(s: Any) -> str:
    return html.escape(str(s), quote=True)


def money(v: float | None, small: bool = True) -> str:
    if v is None:
        return "–"
    if abs(v) < 1 and small:
        return f"${v:.4f}" if abs(v) < 0.1 else f"${v:.3f}"
    return f"${v:,.2f}"


def _bar_path(x: float, y: float, w: float, h: float, r: float = 4) -> str:
    if w <= r:
        return f"M{x:.1f},{y:.1f}h{w:.1f}v{h:.1f}h{-w:.1f}z"
    return (
        f"M{x:.1f},{y:.1f}h{w - r:.1f}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r:.1f}"
        f"a{r},{r} 0 0 1 {-r},{r}h{-(w - r):.1f}z"
    )


def hbars(rows: list[tuple[str, float, str, str]], fmt=money, width: int = 520, label_w: int = 170) -> str:
    """rows: (label, value, css color var, tooltip label)."""
    bh, gap = 18, 12
    vmax = max([v for _, v, _, _ in rows] + [1e-9])
    plot = width - label_w - 70
    h = len(rows) * (bh + gap) + 4
    out = [
        f'<svg viewBox="0 0 {width} {h}" width="{width}" style="max-width:100%;height:auto" role="img" '
        f'aria-label="bar chart">'
    ]
    out.append(f'<line x1="{label_w}" y1="0" x2="{label_w}" y2="{h}" stroke="var(--axis)" stroke-width="1"/>')
    for i, (label, v, color, tip) in enumerate(rows):
        y = i * (bh + gap) + 2
        w = max(1.5, plot * v / vmax) if v > 0 else 0
        out.append(
            f'<text x="{label_w - 8}" y="{y + 13}" text-anchor="end" font-size="12.5" '
            f'fill="var(--text-secondary)">{_e(label)}</text>'
        )
        if w:
            out.append(
                f'<path class="mark" tabindex="0" data-tip="{_e(fmt(v))}|{_e(tip)}" '
                f'd="{_bar_path(label_w + 1, y, w, bh)}" fill="var({color})"/>'
            )
        out.append(
            f'<text x="{label_w + 1 + w + 6}" y="{y + 13}" font-size="12.5" '
            f'fill="var(--text-primary)">{_e(fmt(v))}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def histogram(
    values: list[float],
    threshold: float | None,
    lo: float,
    hi: float,
    bins: int = 20,
    width: int = 520,
    height: int = 170,
    xlabel: str = "",
) -> str:
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    pad_l, pad_b, pad_t = 28, 34, 10
    pw, ph = width - pad_l - 10, height - pad_b - pad_t
    cmax = max(1, int(counts.max()) if len(counts) else 1)
    bw = pw / bins
    out = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" style="max-width:100%;height:auto" role="img" '
        f'aria-label="histogram">'
    ]
    for frac in (0.5, 1.0):
        y = pad_t + ph * (1 - frac)
        out.append(f'<line x1="{pad_l}" x2="{width - 10}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--grid)"/>')
        out.append(
            f'<text x="{pad_l - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
            f'fill="var(--muted)">{round(cmax * frac)}</text>'
        )
    for i, c in enumerate(counts):
        if not c:
            continue
        x = pad_l + i * bw + 1
        bh = ph * c / cmax
        y = pad_t + ph - bh
        w = min(24, bw - 2)
        r = min(4, w / 2)
        d = (
            (
                f"M{x:.1f},{pad_t + ph:.1f}v{-(bh - r):.1f}a{r},{r} 0 0 1 {r},{-r}h{w - 2 * r:.1f}"
                f"a{r},{r} 0 0 1 {r},{r}v{bh - r:.1f}z"
            )
            if bh > r
            else f"M{x:.1f},{y:.1f}h{w:.1f}v{bh:.1f}h{-w:.1f}z"
        )
        out.append(
            f'<path class="mark" tabindex="0" data-tip="{int(c)} renders|{edges[i]:.2f}–{edges[i + 1]:.2f}" '
            f'd="{d}" fill="var(--series-1)"/>'
        )
    out.append(
        f'<line x1="{pad_l}" x2="{width - 10}" y1="{pad_t + ph}" y2="{pad_t + ph}" stroke="var(--axis)"/>'
    )
    for t in np.linspace(lo, hi, 5):
        x = pad_l + pw * (t - lo) / (hi - lo)
        out.append(
            f'<text x="{x:.1f}" y="{pad_t + ph + 15}" text-anchor="middle" font-size="11" '
            f'fill="var(--muted)">{t:.2f}</text>'
        )
    if threshold is not None:
        x = pad_l + pw * (threshold - lo) / (hi - lo)
        out.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{pad_t - 4}" y2="{pad_t + ph}" stroke="var(--text-primary)" '
            f'stroke-width="1.5"/><text x="{x + 5:.1f}" y="{pad_t + 6}" font-size="11.5" '
            f'fill="var(--text-primary)">gate {threshold:.2f}</text>'
        )
    out.append(
        f'<text x="{pad_l + pw / 2}" y="{height - 4}" text-anchor="middle" font-size="11.5" '
        f'fill="var(--text-secondary)">{_e(xlabel)}</text></svg>'
    )
    return "".join(out)


def thumb(path: str | None, width: int = 160) -> str | None:
    if not path or not Path(path).exists():
        return None
    p = Path(path)
    if p.suffix.lower() in (".mp4", ".webm", ".mov"):
        cap = cv2.VideoCapture(str(p))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, img = cap.read()
        cap.release()
        if not ok:
            return None
    else:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            return None
    h, w = img.shape[:2]
    img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None


STATUS_CHIP = {
    "scale": ("var(--good)", "scale"),
    "keep": ("var(--muted)", "keep"),
    "explore": ("var(--warning)", "explore"),
    "kill": ("var(--critical)", "kill"),
}


def render(cfg: EngineConfig, store: Store, batch_id: str | None = None, out: Path | None = None) -> Path:
    batch = store.batch(batch_id) if batch_id else store.latest_batch()
    if batch is None:
        raise ValueError("no batches yet")
    led: Ledger = build_ledger(cfg, store, batch["id"])
    formats = load_formats(cfg)
    stats = arm_stats(cfg, store, batch["character"], formats)
    quotas = allocate(stats, 24) if stats else {}
    pieces = store.pieces(batch["id"])
    sim = led.simulated

    title = f"{led.character.title()} · batch {batch['id']}"
    parts = [
        f"<!doctype html><html lang=en><head><meta charset=utf-8>"
        f"<meta name=viewport content='width=device-width,initial-scale=1'><title>Creator engine report</title>"
        f"<style>{CSS}</style></head><body><div class=viz-root>"
    ]
    parts.append(f"<h1>{_e(title)}{'<span class=badge>SIMULATED GPU</span>' if sim else ''}</h1>")
    parts.append(
        f"<p class=sub>Profile <b>{_e(led.profile)}</b> · {led.pieces} pieces planned · "
        f"generated {time.strftime('%Y-%m-%d %H:%M')}"
        + (
            " · renders come from the GPU simulator; costs use the same per-second billing as a real pod"
            if sim
            else ""
        )
        + "</p>"
    )

    # ---- headline
    best_hosted = min(led.hosted, key=lambda h: h["per_keeper"]) if led.hosted else None
    yield_pct = led.keepers / led.pieces if led.pieces else 0
    parts.append("<div class='grid g4'>")
    parts.append(
        f"<div class=card><div class=label>Cost per keeper</div><div class=hero>{money(led.per_keeper)}</div>"
        f"<div class=note>all-in: GPU work, rejects, idle pods, API calls</div></div>"
    )
    parts.append(
        f"<div class=card><div class=label>Keepers</div><div class=value>{led.keepers} / {led.pieces}</div>"
        f"<div class=note>{yield_pct:.0%} of planned pieces shipped</div></div>"
    )
    parts.append(
        f"<div class=card><div class=label>Total spend</div><div class=value>{money(led.total_usd)}</div>"
        f"<div class=note>{sum(led.gpu_seconds.values()) / 60:.1f} GPU-minutes incl. idle</div></div>"
    )
    if best_hosted:
        prof = cfg.profile(led.profile)
        parts.append(
            f"<div class=card><div class=label>Cheapest hosted, per keeper</div>"
            f"<div class=value>{money(best_hosted['per_keeper'])}</div>"
            f"<div class=note>{_e(best_hosted['name'])} at list price, 2 takes per keeper. This batch rendered "
            f"{min(prof.vid_width, prof.vid_height)}p, so compare with the hero profile before quoting a multiple.</div></div>"
        )
    parts.append("</div>")

    # ---- where the money went + halving plan
    rows = [
        (
            k,
            led.buckets[k],
            "--series-1" if k in USEFUL else "--series-2",
            f"{k} · {led.gpu_seconds.get(k, 0):.0f} GPU-s",
        )
        for k in BUCKETS
        if led.buckets[k] > 0
    ]
    rows.sort(key=lambda r: -r[1])
    parts.append(
        "<section class='grid g2'><div class=card><h2>Where the money went</h2>"
        "<p class=sub>Spend per bucket for this batch. Everything orange is a lever.</p>"
        "<div class=legend><span><i style='background:var(--series-1)'></i>became a keeper</span>"
        "<span><i style='background:var(--series-2)'></i>overhead / waste</span></div>"
        + hbars(rows)
        + "</div>"
    )
    lv_rows = "".join(
        f"<tr><td><b>{_e(lv.name)}</b><div class=note>{_e(lv.action)}</div></td>"
        f"<td class=num>{money(lv.per_keeper)}</td><td class=num>−{lv.saving_pct:.0%}</td></tr>"
        for lv in led.levers
    )
    if led.combined:
        lv_rows += (
            f"<tr><td><b>{_e(led.combined.name)}</b></td><td class=num><b>{money(led.combined.per_keeper)}</b></td>"
            f"<td class=num><b>−{led.combined.saving_pct:.0%}</b></td></tr>"
        )
    parts.append(
        "<div class=card><h2>How to halve it</h2><p class=sub>Counterfactual cost per keeper, computed from "
        "this batch's own rows.</p><table><tr><th>Lever</th><th class=num>Per keeper</th>"
        f"<th class=num>Saving</th></tr>{lv_rows}</table></div></section>"
    )

    # ---- hosted comparison
    if led.hosted and led.per_keeper:
        comp = [("this engine", led.per_keeper, "--series-1", f"{led.profile} profile, measured")]
        comp += [
            (h["name"], h["per_keeper"], "--series-2", h["source"])
            for h in sorted(led.hosted, key=lambda h: h["per_keeper"])
        ]
        parts.append(
            "<section class=card><h2>Same clip, hosted</h2><p class=sub>Per keeper at list price per "
            "generated second × clip length × attempts per keeper (from <code>configs/prices.yaml</code>). "
            "Resolution differs: the engine row is the profile above, so compare like with like before "
            "quoting a multiple.</p><div class=legend><span><i style='background:var(--series-1)'></i>"
            "this engine</span><span><i style='background:var(--series-2)'></i>hosted API</span></div>"
            + hbars(comp, width=760, label_w=210)
            + "</section>"
        )

    # ---- gates
    fun = sorted(led.funnel, key=lambda f: (f["stage"] != "keyframe", -(f["fails"] or 0)))
    gate_rows = [
        (
            f"{'keyframe' if f['stage'] == 'keyframe' else 'clip'} · {f['gate']}",
            float(f["fails"] or 0),
            "--series-2",
            f"{f['fails']} of {f['n']} rejected",
        )
        for f in fun
        if f["fails"]
    ]
    ids = [
        r["score"]
        for r in store.q(
            "SELECT g.score FROM gate_results g WHERE g.batch_id = ? AND g.gate = 'identity' AND g.score IS NOT NULL",
            [batch["id"]],
        )
    ]
    thr = cfg.gate("identity").params.get("min_cosine")
    st = led.stage
    parts.append(
        "<section class='grid g2'><div class=card><h2>What the gates rejected</h2>"
        f"<p class=sub>Keyframe pass rate {st['keyframe']['pass_rate'] or 0:.0%} · clip pass rate "
        f"{st['video']['pass_rate'] or 0:.0%} · {st['keyframe']['per_keeper'] or 0:.1f} keyframes and "
        f"{st['video']['per_keeper'] or 0:.1f} clips rendered per keeper</p>"
        + (
            hbars(gate_rows, fmt=lambda v: f"{v:.0f}", label_w=190)
            if gate_rows
            else "<p class=note>No rejects.</p>"
        )
        + "</div>"
    )
    lo = min(ids + [thr or 0.5]) - 0.05 if ids else 0.0
    parts.append(
        "<div class=card><h2>Identity holds</h2><p class=sub>Cosine similarity of each keyframe's face to the "
        "character centroid. Right of the line passes.</p>"
        + (histogram(ids, thr, max(-0.2, round(lo, 1)), 1.0, xlabel="identity cosine") if ids else "")
        + "</div></section>"
    )

    if led.gate_eval:

        def rate(g: dict[str, Any]) -> str:
            if g.get("recall") is not None:
                return f"{g['recall']:.0%} caught"
            if g.get("false_reject_rate") is not None:
                return f"{g['false_reject_rate']:.0%} false rejects"
            return ""

        ev = "".join(
            f"<tr><td>{_e(g['defect'])}</td><td class=num>{g['injected']}</td><td class=num>{g['caught']}</td>"
            f"<td class=num>{rate(g)}</td></tr>"
            for g in led.gate_eval
        )
        parts.append(
            "<section class=card><h2>Gate accuracy against injected defects</h2><p class=sub>The simulator "
            "knows which renders it broke, so recall and false-reject rate are measurable here. On real "
            "renders, measure the same with a hand-labelled sample.</p><table><tr><th>Defect</th>"
            f"<th class=num>Injected</th><th class=num>Rejected</th><th class=num>Rate</th></tr>{ev}</table></section>"
        )

    # ---- contact sheet
    keepers = [p for p in pieces if p["status"] == "keeper"]
    rejects = [p for p in pieces if p["status"] not in ("keeper", "planned")]

    def fig(p, path_key):
        src = thumb(p[path_key] or p["keyframe_path"])
        if not src:
            return ""
        link = Path(p[path_key]).resolve().as_uri() if p[path_key] else ""
        cap = p["format"] if p["status"] == "keeper" else (p["reject_reason"] or p["status"])
        img = f"<img src='{src}' alt='{_e(p['id'])}' loading=lazy>"
        body = f"<a href='{_e(link)}'>{img}</a>" if link else img
        return f"<figure>{body}<figcaption>{_e(cap)}</figcaption></figure>"

    parts.append(
        f"<section class=card><h2>Keepers ({len(keepers)})</h2><p class=sub>Mid-clip frame; click to open "
        "the clip.</p><div class=sheet>" + "".join(fig(p, "video_path") for p in keepers) + "</div></section>"
    )
    if rejects:
        parts.append(
            f"<section class=card><h2>Rejected ({len(rejects)})</h2><p class=sub>With the gate that "
            "stopped them.</p><div class=sheet>"
            + "".join(fig(p, "video_path") for p in rejects)
            + "</div></section>"
        )

    # ---- experiments
    if stats:
        er = "".join(
            f"<tr><td>{_e(s.format)}</td><td class=num>{s.posts}</td><td class=num>{s.wins}</td>"
            f"<td class=num>{'' if s.median_views is None else f'{s.median_views:,.0f}'}</td>"
            f"<td class=num>{'' if s.median_lift is None else f'{s.median_lift:.2f}×'}</td>"
            f"<td class=num>{s.p_best:.0%}</td><td><span class=chip><i style='background:{STATUS_CHIP[s.status][0]}'></i>"
            f"{STATUS_CHIP[s.status][1]}</span></td><td class=num>{quotas.get(s.format, 0)}</td></tr>"
            for s in stats
        )
        parts.append(
            "<section class=card><h2>Formats: what travels</h2><p class=sub>Each post scored against its own "
            "account's median views; Beta posterior per format; next batch of 24 split by P(best)."
            + (" Metrics here are simulated." if sim else "")
            + "</p><table><tr><th>Format</th>"
            "<th class=num>Posts</th><th class=num>Beat baseline</th><th class=num>Median views</th>"
            "<th class=num>Median lift</th><th class=num>P(best)</th><th>Status</th>"
            f"<th class=num>Next quota</th></tr>{er}</table></section>"
        )

    # ---- queue
    posts = store.q("SELECT * FROM posts WHERE batch_id = ? ORDER BY scheduled_at LIMIT 40", [batch["id"]])
    if posts:
        tzs = {a.handle: a.timezone for a in cfg.publish.accounts}

        def local(p: Any) -> str:
            tz = ZoneInfo(tzs.get(p["account"], "UTC"))
            return datetime.fromtimestamp(p["scheduled_at"], tz).strftime("%a %d %b %H:%M %Z")

        pr = "".join(
            f"<tr><td>{_e(local(p))}</td><td>@{_e(p['account'])}</td>"
            f"<td>{_e(p['format'])}</td><td>{_e(p['status'])}</td><td class=note>{_e(p['caption'].splitlines()[0])}</td></tr>"
            for p in posts
        )
        parts.append(
            "<section class=card><h2>Publishing queue</h2><p class=sub>Windows, daily caps and gaps per "
            "account; every caption ends with the character's AI disclosure.</p><table><tr><th>When (account time)</th>"
            f"<th>Account</th><th>Format</th><th>Status</th><th>Caption</th></tr>{pr}</table></section>"
        )

    parts.append(
        "<section class=note>Data: <code>"
        + _e(json.dumps({k: round(v, 5) for k, v in led.buckets.items()}))
        + "</code></section>"
    )
    parts.append(f"<div id=tip role=tooltip></div></div><script>{TIP_JS}</script></body></html>")
    out = out or cfg.path(cfg.workdir) / "reports" / f"{batch['id']}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
    return out
