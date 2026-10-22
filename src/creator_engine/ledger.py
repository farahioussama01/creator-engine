"""One SQLite file with every render, what it cost, and whether it was kept. Reports read from here."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (id TEXT PRIMARY KEY, provider TEXT, planned INT, mix TEXT,
  started REAL, finished REAL);
CREATE TABLE IF NOT EXISTS pieces (id TEXT PRIMARY KEY, batch TEXT, format TEXT, status TEXT, reason TEXT,
  keyframe TEXT, clip TEXT, identity REAL, caption TEXT, prompt TEXT);
CREATE TABLE IF NOT EXISTS renders (id INTEGER PRIMARY KEY, batch TEXT, piece TEXT, stage TEXT, usd REAL,
  seconds REAL, passed INT, kept INT DEFAULT 0, reason TEXT, path TEXT, at REAL);
CREATE TABLE IF NOT EXISTS setup (id INTEGER PRIMARY KEY, kind TEXT, usd REAL, note TEXT, at REAL);
CREATE TABLE IF NOT EXISTS views (piece TEXT PRIMARY KEY, platform TEXT, posted_at TEXT, views INT);
"""


class Ledger:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def q(self, sql: str, *args) -> list[sqlite3.Row]:
        return list(self.db.execute(sql, args))

    def one(self, sql: str, *args):
        rows = self.q(sql, *args)
        return rows[0] if rows else None

    def add_batch(self, batch: str, provider: str, planned: int, mix: dict) -> None:
        self.db.execute(
            "INSERT INTO batches VALUES (?,?,?,?,?,NULL)", (batch, provider, planned, json.dumps(mix), time.time())
        )

    def finish_batch(self, batch: str) -> None:
        self.db.execute("UPDATE batches SET finished = ? WHERE id = ?", (time.time(), batch))

    def add_piece(self, **p) -> None:
        self.db.execute(
            "INSERT INTO pieces (id, batch, format, status, caption, prompt) VALUES (?,?,?,?,?,?)",
            (p["id"], p["batch"], p["format"], "planned", p["caption"], p["prompt"]),
        )

    def set_piece(self, piece: str, **fields) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.db.execute(f"UPDATE pieces SET {sets} WHERE id = ?", (*fields.values(), piece))

    def add_render(
        self,
        batch: str,
        piece: str,
        stage: str,
        usd: float,
        seconds: float,
        passed: bool,
        reason: str = "",
        path: str = "",
    ) -> int:
        cur = self.db.execute(
            "INSERT INTO renders (batch, piece, stage, usd, seconds, passed, reason, path, at) VALUES (?,?,?,?,?,?,?,?,?)",
            (batch, piece, stage, usd, seconds, int(passed), reason, path, time.time()),
        )
        return int(cur.lastrowid)

    def mark_kept(self, render_id: int) -> None:
        self.db.execute("UPDATE renders SET kept = 1 WHERE id = ?", (render_id,))

    def add_setup(self, kind: str, usd: float, note: str = "") -> None:
        self.db.execute("INSERT INTO setup (kind, usd, note, at) VALUES (?,?,?,?)", (kind, usd, note, time.time()))

    def spent(self, batch: str) -> float:
        return float(self.one("SELECT COALESCE(SUM(usd), 0) s FROM renders WHERE batch = ?", batch)["s"])

    def latest_batch(self) -> str | None:
        row = self.one("SELECT id FROM batches ORDER BY started DESC LIMIT 1")
        return row["id"] if row else None
