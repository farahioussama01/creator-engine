"""`ce` command line."""

from __future__ import annotations

import asyncio
import csv
import json
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import ops
from .config import load_characters, load_config, load_formats

app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="Creator engine: photoreal AI content at volume."
)
identity_app = typer.Typer(no_args_is_help=True, help="Identity centroids for the identity gates.")
publish_app = typer.Typer(no_args_is_help=True, help="Schedule and publish keepers.")
metrics_app = typer.Typer(no_args_is_help=True, help="Post metrics in.")
sim_app = typer.Typer(no_args_is_help=True, help="The GPU simulator.")
models_app = typer.Typer(no_args_is_help=True, help="Model files for the real face analyzer.")
gates_app = typer.Typer(no_args_is_help=True, help="Quality gate calibration.")
app.add_typer(identity_app, name="identity")
app.add_typer(publish_app, name="publish")
app.add_typer(metrics_app, name="metrics")
app.add_typer(sim_app, name="sim")
app.add_typer(models_app, name="models")
app.add_typer(gates_app, name="gates")
con = Console()

ConfigOpt = typer.Option(ops.DEFAULT_CONFIG, "--config", "-c", help="Engine config YAML")


def _json(obj) -> None:
    con.print_json(json.dumps(obj, default=str))


@app.command()
def demo(
    pieces: int = 24, rounds: int = 3, gpus: int = 2, time_scale: float | None = None, config: str = ConfigOpt
) -> None:
    """Run the full loop against simulated GPUs (no GPU, no API keys)."""
    from .demo import run_demo

    cfg = load_config(config)
    asyncio.run(
        run_demo(
            cfg,
            pieces=pieces,
            rounds=rounds,
            gpus=gpus,
            time_scale=time_scale,
            log=lambda s: con.print(s, highlight=False, markup=False),
        )
    )


@app.command()
def plan(
    character: str,
    n: int = 24,
    profile: str = "standard",
    quota: list[str] = typer.Option([], help="format=count, repeatable; default: from experiments"),
    config: str = ConfigOpt,
) -> None:
    """Plan a batch. Quotas come from the format experiments unless given."""
    ctx = ops.open_ctx(config)
    quotas = {q.split("=")[0]: int(q.split("=")[1]) for q in quota} or None
    _json(ops.plan(ctx, character, n, profile, quotas))


@app.command()
def run(
    batch: str,
    provision: bool = typer.Option(False, help="Rent RunPod pods for this run, then terminate"),
    config: str = ConfigOpt,
) -> None:
    """Render and gate a planned batch (resumable)."""
    ctx = ops.open_ctx(config)
    _json(asyncio.run(ops.run(ctx, batch, provision)))
    con.print(ops.report_text(ops.report(ctx, batch)), highlight=False, markup=False)


@app.command()
def report(
    batch: str | None = typer.Argument(None),
    as_json: bool = typer.Option(False, "--json"),
    config: str = ConfigOpt,
) -> None:
    """Cost per keeper, spend buckets, the halving plan."""
    led = ops.report(ops.open_ctx(config), batch)
    _json(led) if as_json else con.print(ops.report_text(led), highlight=False, markup=False)


@app.command()
def dashboard(
    batch: str | None = typer.Argument(None), out: Path | None = None, config: str = ConfigOpt
) -> None:
    """Write the HTML report for a batch."""
    from .dashboard import render

    ctx = ops.open_ctx(config)
    con.print(str(render(ctx.cfg, ctx.store, batch, out)))


@app.command()
def status(config: str = ConfigOpt) -> None:
    """Endpoint health, queue depth, spend in the last 24h."""
    _json(asyncio.run(ops.status(ops.open_ctx(config))))


@app.command()
def daily(
    character: str,
    n: int = 24,
    profile: str = "standard",
    provision: bool = False,
    publish: bool = True,
    webhook: str | None = typer.Option(None, envvar="CE_WEBHOOK"),
    config: str = ConfigOpt,
) -> None:
    """The unattended loop (cron this): allocate, plan, render, gate, schedule, publish due, report."""
    _json(asyncio.run(ops.daily(ops.open_ctx(config), character, n, profile, provision, publish, webhook)))


@app.command()
def experiments(character: str, n: int = 24, config: str = ConfigOpt) -> None:
    """Per-format performance vs account baselines and the next allocation."""
    from .experiments import allocate, arm_stats

    ctx = ops.open_ctx(config)
    stats = arm_stats(ctx.cfg, ctx.store, character, load_formats(ctx.cfg))
    allocate(stats, n)
    t = Table("format", "posts", "beat baseline", "median lift", "P(best)", "status", "next quota")
    for s in stats:
        t.add_row(
            s.format,
            str(s.posts),
            str(s.wins),
            "" if s.median_lift is None else f"{s.median_lift:.2f}x",
            f"{s.p_best:.0%}",
            s.status,
            str(s.next_quota),
        )
    con.print(t)


@app.command()
def check(endpoint: str, config: str = ConfigOpt) -> None:
    """Preflight a ComfyUI endpoint: nodes, inputs and model files for every template."""
    res = asyncio.run(ops.check_endpoint(load_config(config), endpoint))
    _json(res)
    raise typer.Exit(0 if res["ok"] else 1)


@app.command()
def bench(
    endpoint: str,
    stage: str = "video",
    profile: str = "standard",
    n: int = 3,
    usd_per_hour: float = typer.Option(..., help="What this pod costs"),
    config: str = ConfigOpt,
) -> None:
    """Measure seconds per render on a real pod -> $ per clip. Feed the result into prices.yaml."""
    from .bench import run_bench

    res = asyncio.run(run_bench(load_config(config), endpoint, stage, profile, n, usd_per_hour))
    _json(res)


@app.command()
def costs(seconds: float = 5.0, config: str = ConfigOpt) -> None:
    """Per-render GPU cost by profile and GPU (timing model + price table) next to hosted per-clip prices."""
    from .config import load_prices
    from .sim import timing

    cfg = load_config(config)
    prices = load_prices(cfg)
    speeds = prices.get("sim_gpu_speed", {})
    t = Table("profile", "clip", "gpu", "$/h", "est. s/clip", "est. $/clip", "measured?")
    for name, p in cfg.profiles.items():
        for gpu, rates in prices.get("gpu_hourly", {}).items():
            rate = min(v for v in rates.values() if isinstance(v, int | float))
            s = timing.video_seconds(
                p.vid_width, p.vid_height, p.frames, p.vid_steps, p.vid_cfg, speeds.get(gpu, 1.0)
            )
            t.add_row(
                name,
                f"{p.vid_width}x{p.vid_height} {p.seconds:g}s",
                gpu,
                f"{rate:.2f}",
                f"{s:.0f}",
                f"{s * rate / 3600:.4f}",
                "scaled from 4090 anchor" if gpu == "rtx4090" else "assumed speed",
            )
    con.print(t)
    h = Table(
        "hosted API",
        "$/s",
        f"$ per {seconds:g}s generation",
        f"x{prices.get('hosted_attempts_per_keeper', 2):g} attempts",
        "source",
    )
    for row in prices.get("hosted_video", []):
        per = row["usd_per_second"] * seconds
        h.add_row(
            row["name"],
            f"{row['usd_per_second']:.3f}",
            f"{per:.2f}",
            f"{per * prices.get('hosted_attempts_per_keeper', 2):.2f}",
            row.get("source", ""),
        )
    con.print(h)


@gates_app.command("calibrate")
def gates_calibrate(
    batch: str,
    labels: Path | None = typer.Option(None, help="CSV: file,label,issue"),
    config: str = ConfigOpt,
) -> None:
    """Recall / false-reject per gate at the current threshold, and the threshold that maximises both."""
    from .calibrate import calibrate

    rows = calibrate(ops.open_ctx(config), batch, labels)
    t = Table(
        "gate", "param", "good", "bad", "current", "recall", "false rej.", "suggested", "recall", "false rej."
    )
    pct = lambda v: "" if v is None else f"{v:.0%}"  # noqa: E731
    num = lambda v: "" if v is None else f"{v:.3g}"  # noqa: E731
    for r in rows:
        t.add_row(
            r["gate"],
            r["param"],
            str(r["n_good"]),
            str(r["n_bad"]),
            num(r["current"]),
            pct(r.get("recall_at_current")),
            pct(r.get("false_reject_at_current")),
            num(r.get("suggested")),
            pct(r.get("recall_at_suggested")),
            pct(r.get("false_reject_at_suggested")),
        )
    con.print(t)


@identity_app.command("enroll")
def identity_enroll(
    character: str,
    refs: Path | None = None,
    sim: bool = typer.Option(False, help="Generate simulator reference shots first"),
    config: str = ConfigOpt,
) -> None:
    """Build the identity centroid from reference images (usually the LoRA training set)."""
    from .identity import enroll, list_images, make_sim_references, save
    from .vision import make_analyzer

    cfg = load_config(config)
    ch = load_characters(cfg)[character]
    folder = refs or (cfg.path(ch.reference_dir) if ch.reference_dir else None)
    if folder is None:
        raise typer.BadParameter("pass --refs or set reference_dir on the character")
    if sim:
        make_sim_references(ch, folder)
    analyzer = make_analyzer("sim" if sim else cfg.analyzer, str(cfg.path(cfg.models_dir)))
    centroid, rep = enroll(analyzer, character, list_images(folder))
    path = save(cfg, centroid, rep)
    _json({"saved": str(path), **rep.__dict__})


@publish_app.command("schedule")
def publish_schedule(batch: str, horizon_days: int = 7, config: str = ConfigOpt) -> None:
    """Give every keeper of a batch a slot on one of its character's accounts."""
    from .publish import schedule_batch

    ctx = ops.open_ctx(config)
    posts = schedule_batch(ctx.cfg, ctx.store, batch, horizon_days=horizon_days)
    t = Table("when (UTC)", "account", "format", "caption")
    for p in posts:
        t.add_row(
            time.strftime("%a %d %b %H:%M", time.gmtime(p["scheduled_at"])),
            p["account"],
            p["format"],
            p["caption"].splitlines()[0],
        )
    con.print(t)


@publish_app.command("due")
def publish_due_cmd(config: str = ConfigOpt) -> None:
    """Publish every scheduled post whose time has come (dry run unless publish.dry_run is false)."""
    from .publish import publish_due

    ctx = ops.open_ctx(config)
    _json(publish_due(ctx.cfg, ctx.store))


@metrics_app.command("fetch")
def metrics_fetch(config: str = ConfigOpt) -> None:
    """Pull Reels insights for published posts (Instagram Graph API)."""
    from .publish import InstagramPublisher, publisher_for

    ctx = ops.open_ctx(config)
    accounts = {a.handle: a for a in ctx.cfg.publish.accounts}
    n = 0
    for p in ctx.store.q("SELECT * FROM posts WHERE status = 'published' AND media_id IS NOT NULL"):
        pub = publisher_for(ctx.cfg, accounts[p["account"]])
        if not isinstance(pub, InstagramPublisher):
            continue
        m = pub.insights(p["media_id"])
        ctx.store.insert(
            "metrics",
            {
                "post_id": p["id"],
                "fetched_at": time.time(),
                "age_hours": (time.time() - p["published_at"]) / 3600,
                "views": m.get("views"),
                "reach": m.get("reach"),
                "saved": m.get("saved"),
                "shares": m.get("shares"),
                "interactions": m.get("total_interactions"),
                "avg_watch_ms": m.get("ig_reels_avg_watch_time"),
                "skip_rate": m.get("reels_skip_rate"),
                "simulated": 0,
            },
        )
        n += 1
    con.print(f"fetched insights for {n} posts")


@metrics_app.command("import")
def metrics_import(csv_path: Path, config: str = ConfigOpt) -> None:
    """Import metrics from a CSV with columns post_id,age_hours,views[,reach,saved,shares,skip_rate]."""
    ctx = ops.open_ctx(config)
    n = 0
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            ctx.store.insert(
                "metrics",
                {
                    "post_id": row["post_id"],
                    "fetched_at": time.time(),
                    "age_hours": float(row["age_hours"]),
                    "views": float(row["views"]),
                    "reach": float(row.get("reach") or 0),
                    "saved": float(row.get("saved") or 0),
                    "shares": float(row.get("shares") or 0),
                    "interactions": None,
                    "avg_watch_ms": None,
                    "skip_rate": float(row["skip_rate"]) if row.get("skip_rate") else None,
                    "simulated": 0,
                },
            )
            n += 1
    con.print(f"imported {n} rows")


@sim_app.command("serve")
def sim_serve(
    port: int = 8188,
    gpu: str = "rtx4090",
    gpu_speed: float = 1.0,
    time_scale: float = 0.02,
    config: str = ConfigOpt,
) -> None:
    """Run a simulated ComfyUI pod (point endpoints at it)."""
    from .sim.server import serve_forever

    cfg = load_config(config)
    loras = [c.lora for c in load_characters(cfg).values()]
    try:
        asyncio.run(
            serve_forever(
                port,
                gpu=gpu,
                gpu_speed=gpu_speed,
                time_scale=time_scale,
                extra_loras=loras,
                defect_rates=cfg.sim.defect_rates,
            )
        )
    except KeyboardInterrupt:
        sys.exit(0)


@models_app.command("download")
def models_download(config: str = ConfigOpt) -> None:
    """Fetch YuNet (MIT) and AuraFace (Apache-2.0) into models/."""
    import httpx

    from .vision.real import MODEL_URLS

    cfg = load_config(config)
    d = cfg.path(cfg.models_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, url in MODEL_URLS.items():
        target = d / name
        if target.exists():
            con.print(f"have {target}")
            continue
        con.print(f"downloading {url}")
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with open(target, "wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)
        con.print(f"saved {target}")


if __name__ == "__main__":
    app()
