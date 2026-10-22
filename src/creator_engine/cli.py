"""`ce` - the command line.

ce models                         download the two face models (once)
ce character portraits            8 candidate faces -> pick one
ce character variations IMG       ~20 photos of that same person
ce character check                consistency of the set + identity threshold
ce character train                train the LoRA on fal
ce run --n 40                     the pipeline
ce report                         the four numbers + an HTML contact sheet
ce export                         keepers + captions, ready to post
ce views views.csv                import views after posting
ce next --n 40                    next batch's format mix from the views
"""

from __future__ import annotations

import asyncio
import csv
import shutil
from pathlib import Path

import typer

from . import character as char
from .config import load
from .ledger import Ledger

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__.split("\n\n")[0])
character_app = typer.Typer(no_args_is_help=True, help="Create and train the character.")
app.add_typer(character_app, name="character")
CONFIG = typer.Option("configs/config.yaml", "--config", "-c", envvar="CE_CONFIG")


def _ledger(cfg) -> Ledger:
    return Ledger(cfg.path(cfg.workdir) / "ledger.db")


def _parse_mix(mix: list[str]) -> dict[str, int] | None:
    return {m.split("=")[0]: int(m.split("=")[1]) for m in mix} or None


@app.command()
def models(config: str = CONFIG) -> None:
    """Download YuNet (face finder) and AuraFace (face fingerprint) into models/."""
    from .faces import download_models

    for p in download_models(load(config).path("models")):
        typer.echo(f"ok {p}")


@character_app.command()
def portraits(n: int = 8, config: str = CONFIG) -> None:
    """Render candidate faces from the character description (~$0.004 each)."""
    for p in asyncio.run(char.portraits(load(config), n)):
        typer.echo(p)
    typer.echo("Pick one, then: ce character variations <path>")


@character_app.command()
def variations(hero: Path, n: int = 20, config: str = CONFIG) -> None:
    """Re-draw the chosen face in new angles, outfits and places (~$0.024 each)."""
    paths = asyncio.run(char.variations(load(config), hero, n))
    typer.echo(
        f"{len(paths)} images in {paths[0].parent}. Delete any that don't look like her, then: ce character check"
    )


@character_app.command()
def check(config: str = CONFIG) -> None:
    """Is the training set one consistent face? Sets the identity threshold."""
    r = char.check(load(config))
    typer.echo(
        f"{r['used']} images used, typical similarity {r['typical_similarity']}, "
        f"identity threshold {r['suggested_threshold']}"
    )
    if r["outliers"] or r["no_face"]:
        typer.echo(f"remove these and run again: outliers {r['outliers']}, no face {r['no_face']}")
    else:
        typer.echo("consistent. Next: ce character train")


@character_app.command()
def train(steps: int = 2000, config: str = CONFIG) -> None:
    """Train the character LoRA on fal (~$2.26 per 1000 steps)."""
    url = asyncio.run(char.train(load(config), steps))
    typer.echo(f"LoRA ready: {url}\nNext: ce run --n 12")


@app.command()
def run(
    n: int = 24,
    mix: list[str] = typer.Option([], help="format=count, repeatable"),
    provider: str = typer.Option(None, help="fal | comfyui | mock (overrides config)"),
    config: str = CONFIG,
) -> None:
    """Plan, render, check and keep. Prints the batch's numbers when done."""
    from .pipeline import run_batch
    from .report import answer, write_html

    cfg = load(config)
    if provider:
        cfg.provider = provider
    batch = asyncio.run(run_batch(cfg, n, _parse_mix(mix), log=typer.echo))
    typer.echo("\n" + answer(cfg, _ledger(cfg), batch))
    typer.echo(f"\nreport: {write_html(cfg, _ledger(cfg), batch)}")


@app.command()
def report(batch: str = typer.Argument(None), config: str = CONFIG) -> None:
    """Pipeline, volume, cost per piece, views - as sentences, plus the HTML contact sheet."""
    from .report import answer, write_html

    cfg = load(config)
    ledger = _ledger(cfg)
    batch = batch or ledger.latest_batch()
    typer.echo(answer(cfg, ledger, batch))
    typer.echo(f"\nreport: {write_html(cfg, ledger, batch)}")


@app.command()
def export(batch: str = typer.Argument(None), config: str = CONFIG) -> None:
    """Copy keepers to out/<batch>/ with a captions.csv (AI disclosure included) and a views.csv to fill in."""
    cfg = load(config)
    ledger = _ledger(cfg)
    batch = batch or ledger.latest_batch()
    out = cfg.path("out") / batch
    out.mkdir(parents=True, exist_ok=True)
    keepers = ledger.q("SELECT * FROM pieces WHERE batch = ? AND status = 'keeper'", batch)
    with open(out / "captions.csv", "w", newline="") as fc, open(out / "views.csv", "w", newline="") as fv:
        caps, views = csv.writer(fc), csv.writer(fv)
        caps.writerow(["file", "caption"])
        views.writerow(["file", "platform", "posted_at", "views"])
        for p in keepers:
            name = f"{p['id']}.mp4"
            shutil.copy(p["clip"], out / name)
            caps.writerow([name, f"{p['caption']}\n\n{cfg.character.disclosure}"])
            views.writerow([name, "instagram", "", ""])
    typer.echo(
        f"{len(keepers)} clips in {out}. Post them (label them as AI on the platform too), "
        f"fill views.csv after 48 h, then: ce views {out / 'views.csv'}"
    )


@app.command()
def views(csv_path: Path, config: str = CONFIG) -> None:
    """Import views per post (columns: file, platform, posted_at, views)."""
    from .views import import_views

    cfg = load(config)
    typer.echo(f"imported {import_views(_ledger(cfg), csv_path)} posts. Next: ce report, ce next")


@app.command(name="next")
def next_(n: int = 24, config: str = CONFIG) -> None:
    """Next batch's format mix: more of what got views, a floor for everything else."""
    from .views import next_mix

    cfg = load(config)
    mix = next_mix(_ledger(cfg), list(cfg.formats), n)
    typer.echo(" ".join(f"--mix {f}={q}" for f, q in mix.items()))


if __name__ == "__main__":
    app()
