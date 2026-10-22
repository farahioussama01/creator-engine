"""Get keepers seen: schedule across accounts, label them, publish through the official API.

* Scheduling respects each account's posting windows (in its own timezone), a daily cap, a minimum
  gap between posts, and Meta's per-account API publishing limit.
* Every caption carries the character's AI disclosure. It is appended here, in one place, and the
  config refuses an empty one, so no code path can post an unlabeled piece.
* Every file gets a provenance sidecar (IPTC digital source type "trainedAlgorithmicMedia" plus the
  models, LoRA, seeds and prompts that made it) so the label survives re-uploads and audits.
* Publishing uses the Instagram Graph API content-publishing flow for Reels:
  create container -> poll status_code until FINISHED -> media_publish. Dry-run by default.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import httpx

from .config import AccountSpec, CharacterSpec, EngineConfig, FormatSpec, load_characters, load_formats
from .store import Store

IPTC_AI = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
REEL_METRICS = [
    "views",
    "reach",
    "saved",
    "shares",
    "total_interactions",
    "ig_reels_avg_watch_time",
    "reels_skip_rate",
]
MAX_CAPTION = 2200
MAX_HASHTAGS = 30


def build_caption(text: str, hashtags: list[str], disclosure: str) -> str:
    if not disclosure.strip():
        raise ValueError("refusing to build a caption without an AI disclosure")
    tags = [t if t.startswith("#") else f"#{t}" for t in hashtags][: MAX_HASHTAGS - 1]
    parts = [text.strip()] if text.strip() else []
    if tags:
        parts.append(" ".join(tags))
    parts.append(disclosure.strip())
    caption = "\n\n".join(parts)
    if len(caption) > MAX_CAPTION:
        caption = caption[: MAX_CAPTION - len(disclosure) - 3].rstrip() + "\n\n" + disclosure.strip()
    return caption


def _windows(account: AccountSpec, day: datetime) -> list[tuple[datetime, datetime]]:
    out = []
    for w in account.windows:
        a, b = w.split("-")
        ha, ma = map(int, a.split(":"))
        hb, mb = map(int, b.split(":"))
        out.append(
            (
                day.replace(hour=ha, minute=ma, second=0, microsecond=0),
                day.replace(hour=hb, minute=mb, second=0, microsecond=0),
            )
        )
    return out


def next_slot(
    account: AccountSpec,
    taken: list[float],
    now: float,
    api_cap: int,
    horizon_days: int = 7,
    step_minutes: int = 15,
) -> float | None:
    """Earliest timestamp that fits the account's windows, daily cap, gap and API cap."""
    tz = ZoneInfo(account.timezone)
    start = datetime.fromtimestamp(now, tz)
    gap = account.min_gap_minutes * 60
    for d in range(horizon_days):
        day = (start + timedelta(days=d)).replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day + timedelta(days=1)
        per_day = sum(1 for t in taken if day.timestamp() <= t < day_end.timestamp())
        if per_day >= account.daily_cap:
            continue
        for lo, hi in _windows(account, day):
            t = lo
            while t <= hi:
                ts = t.timestamp()
                in_24h = sum(1 for x in taken if abs(x - ts) < 86400)
                if ts >= now + 600 and all(abs(ts - x) >= gap for x in taken) and in_24h < api_cap:
                    return ts
                t += timedelta(minutes=step_minutes)
    return None


def schedule_batch(
    cfg: EngineConfig, store: Store, batch_id: str, now: float | None = None, horizon_days: int = 7
) -> list[dict[str, Any]]:
    now = now or time.time()
    characters = load_characters(cfg)
    formats = load_formats(cfg)
    keepers = [
        p
        for p in store.pieces(batch_id, "keeper")
        if not store.one("SELECT 1 FROM posts WHERE piece_id = ?", [p["id"]])
    ]
    # interleave formats so an account does not post the same format back to back
    by_fmt: dict[str, list[Any]] = {}
    for p in keepers:
        by_fmt.setdefault(p["format"], []).append(p)
    ordered = []
    while any(by_fmt.values()):
        for f in list(by_fmt):
            if by_fmt[f]:
                ordered.append(by_fmt[f].pop(0))
    scheduled = []
    for p in ordered:
        accounts = [a for a in cfg.publish.accounts if a.character == p["character"]]
        if not accounts:
            continue
        options = []
        for a in accounts:
            taken = [
                r["scheduled_at"]
                for r in store.q("SELECT scheduled_at FROM posts WHERE account = ?", [a.handle])
            ]
            slot = next_slot(a, taken, now, cfg.publish.api_daily_cap, horizon_days)
            if slot is not None:
                options.append((slot, a))
        if not options:
            continue
        slot, acct = min(options, key=lambda x: x[0])
        ch: CharacterSpec = characters[p["character"]]
        fmt: FormatSpec = formats[p["format"]]
        post = {
            "id": f"{p['id']}@{acct.handle}",
            "piece_id": p["id"],
            "batch_id": batch_id,
            "account": acct.handle,
            "character": p["character"],
            "format": p["format"],
            "scheduled_at": slot,
            "status": "scheduled",
            "caption": build_caption(p["caption"], fmt.hashtags, ch.disclosure),
            "container_id": None,
            "media_id": None,
            "published_at": None,
            "error": None,
            "simulated": int(bool(store.batch(batch_id)["simulated"])),
        }
        store.insert("posts", post)
        write_provenance(cfg, store, p)
        scheduled.append(post)
    return scheduled


def write_provenance(cfg: EngineConfig, store: Store, piece: Any) -> Path | None:
    if not piece["video_path"]:
        return None
    rows = store.q(
        "SELECT stage, endpoint, gpu, seed, prompt_id, gpu_seconds FROM executions "
        "WHERE piece_id = ? AND outcome IN ('selected', 'kept')",
        [piece["id"]],
    )
    ch = load_characters(cfg)[piece["character"]]
    data = {
        "digital_source_type": IPTC_AI,
        "disclosure": ch.disclosure,
        "character": {"id": ch.id, "likeness": ch.likeness, "consent_record": ch.consent_record},
        "keyframe": {
            "workflow": str(cfg.workflows["keyframe"]),
            "lora": ch.lora,
            "prompt": piece["keyframe_prompt"],
        },
        "video": {"workflow": str(cfg.workflows["video"]), "prompt": piece["motion_prompt"]},
        "renders": [dict(r) for r in rows],
        "generated_at": datetime.now(UTC).isoformat(),
    }
    out = Path(piece["video_path"]).with_suffix(".provenance.json")
    out.write_text(json.dumps(data, indent=2))
    return out


# ------------------------------------------------------------------ publishers


class Publisher(Protocol):
    def publish_reel(self, video_url: str, caption: str) -> tuple[str, str]: ...


@dataclass
class DryRunPublisher:
    outbox: Path

    def publish_reel(self, video_url: str, caption: str) -> tuple[str, str]:
        self.outbox.mkdir(parents=True, exist_ok=True)
        cid = f"dry-{int(time.time())}-{uuid.uuid4().hex[:10]}"  # unique even within one millisecond
        (self.outbox / f"{cid}.json").write_text(
            json.dumps({"video_url": video_url, "caption": caption}, indent=2)
        )
        return cid, f"dry-media-{cid}"


class GraphAPIError(RuntimeError):
    pass


class InstagramPublisher:
    """Instagram Graph API, content publishing for Reels. One instance per professional account."""

    def __init__(
        self,
        ig_user_id: str,
        token: str,
        host: str = "graph.instagram.com",
        version: str = "v26.0",
        client: httpx.Client | None = None,
        poll_seconds: float = 5.0,
        timeout: float = 900.0,
    ):
        self.ig, self.token = ig_user_id, token
        self.base = f"https://{host}/{version}"
        self.http = client or httpx.Client(timeout=60)
        self.poll, self.timeout = poll_seconds, timeout

    def _req(self, method: str, path: str, **params: Any) -> dict[str, Any]:
        params["access_token"] = self.token
        r = self.http.request(
            method,
            f"{self.base}/{path}",
            params=params if method == "GET" else None,
            data=params if method != "GET" else None,
        )
        body = r.json() if r.content else {}
        if r.status_code >= 400 or "error" in body:
            err = body.get("error", {})
            raise GraphAPIError(f"{r.status_code} {err.get('type', '')}: {err.get('message', r.text[:200])}")
        return body

    def quota(self) -> tuple[int, int]:
        body = self._req("GET", f"{self.ig}/content_publishing_limit", fields="quota_usage,config")
        d = (body.get("data") or [{}])[0]
        return int(d.get("quota_usage", 0)), int((d.get("config") or {}).get("quota_total", 50))

    def publish_reel(self, video_url: str, caption: str) -> tuple[str, str]:
        used, total = self.quota()
        if used >= total:
            raise GraphAPIError(f"publishing quota exhausted ({used}/{total} in the last 24h)")
        container = self._req(
            "POST",
            f"{self.ig}/media",
            media_type="REELS",
            video_url=video_url,
            caption=caption,
            share_to_feed="true",
        )["id"]
        deadline = time.monotonic() + self.timeout
        while True:
            status = self._req("GET", container, fields="status_code").get("status_code")
            if status == "FINISHED":
                break
            if status in ("ERROR", "EXPIRED"):
                raise GraphAPIError(f"container {container} {status}")
            if time.monotonic() > deadline:
                raise GraphAPIError(f"container {container} still {status} after {self.timeout:.0f}s")
            time.sleep(self.poll)
        media = self._req("POST", f"{self.ig}/media_publish", creation_id=container)["id"]
        return container, media

    def insights(self, media_id: str, metrics: list[str] | None = None) -> dict[str, float]:
        body = self._req("GET", f"{media_id}/insights", metric=",".join(metrics or REEL_METRICS))
        out = {}
        for item in body.get("data", []):
            if item.get("values"):
                out[item["name"]] = item["values"][0].get("value")
            elif item.get("total_value"):
                out[item["name"]] = item["total_value"].get("value")
        return out


def media_url(cfg: EngineConfig, path: str) -> str:
    if not cfg.publish.media_base_url:
        if cfg.publish.dry_run:
            return Path(path).resolve().as_uri()
        raise ValueError("publish.media_base_url is not set: the Graph API needs a public video URL")
    rel = Path(path).resolve().relative_to(cfg.path(cfg.workdir).resolve())
    return cfg.publish.media_base_url.rstrip("/") + "/" + rel.as_posix()


def publisher_for(cfg: EngineConfig, account: AccountSpec) -> Publisher:
    if cfg.publish.dry_run:
        return DryRunPublisher(cfg.path(cfg.workdir) / "outbox" / account.handle)
    ig = os.environ.get(account.ig_user_id_env or "")
    token = os.environ.get(account.token_env or "")
    if not ig or not token:
        raise ValueError(f"{account.handle}: set {account.ig_user_id_env} and {account.token_env}")
    return InstagramPublisher(ig, token, cfg.publish.graph_host, cfg.publish.api_version)


def publish_due(
    cfg: EngineConfig, store: Store, now: float | None = None, factory=publisher_for
) -> list[dict[str, Any]]:
    now = now or time.time()
    accounts = {a.handle: a for a in cfg.publish.accounts}
    done = []
    for post in store.q(
        "SELECT * FROM posts WHERE status = 'scheduled' AND scheduled_at <= ? ORDER BY scheduled_at", [now]
    ):
        piece = store.one("SELECT * FROM pieces WHERE id = ?", [post["piece_id"]])
        try:
            pub = factory(cfg, accounts[post["account"]])
            container, media = pub.publish_reel(media_url(cfg, piece["video_path"]), post["caption"])
            status = "dry_run" if cfg.publish.dry_run else "published"
            store.update(
                "posts",
                "id",
                post["id"],
                status=status,
                container_id=container,
                media_id=media,
                published_at=now,
                error=None,
            )
            done.append({"post": post["id"], "status": status, "media_id": media})
        except (GraphAPIError, httpx.HTTPError, ValueError, KeyError) as exc:
            store.update("posts", "id", post["id"], status="failed", error=str(exc))
            done.append({"post": post["id"], "status": "failed", "error": str(exc)})
    return done
