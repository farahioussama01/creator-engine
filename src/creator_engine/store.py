"""SQLite state: batches, pieces, GPU executions, gate verdicts, spend, posts and post metrics.

One file, no server. Every number in a report can be traced back to a row here.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
  id TEXT PRIMARY KEY, created_at REAL, character TEXT, profile TEXT, status TEXT,
  plan_json TEXT, simulated INTEGER DEFAULT 0, note TEXT
);
CREATE TABLE IF NOT EXISTS pieces (
  id TEXT PRIMARY KEY, batch_id TEXT, character TEXT, format TEXT, variant_json TEXT,
  keyframe_prompt TEXT, motion_prompt TEXT, caption TEXT, kf_seed INTEGER, vid_seed INTEGER,
  status TEXT, keyframe_path TEXT, video_path TEXT, identity REAL, reject_reason TEXT, updated_at REAL
);
CREATE INDEX IF NOT EXISTS pieces_batch ON pieces(batch_id);
CREATE TABLE IF NOT EXISTS executions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, piece_id TEXT, batch_id TEXT, stage TEXT, attempt INTEGER,
  endpoint TEXT, gpu TEXT, usd_per_hour REAL, prompt_id TEXT, seed INTEGER, gpu_seconds REAL,
  wall_seconds REAL, usd REAL, status TEXT, outcome TEXT, output_path TEXT, error TEXT,
  sim_truth TEXT, started_at REAL, finished_at REAL
);
CREATE INDEX IF NOT EXISTS exec_batch ON executions(batch_id);
CREATE TABLE IF NOT EXISTS gate_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT, execution_id INTEGER, piece_id TEXT, batch_id TEXT, stage TEXT,
  gate TEXT, passed INTEGER, score REAL, reason TEXT, metrics_json TEXT
);
CREATE INDEX IF NOT EXISTS gates_batch ON gate_results(batch_id);
CREATE TABLE IF NOT EXISTS spend (
  id INTEGER PRIMARY KEY AUTOINCREMENT, batch_id TEXT, kind TEXT, ref TEXT, seconds REAL, usd REAL,
  created_at REAL, detail_json TEXT
);
CREATE TABLE IF NOT EXISTS posts (
  id TEXT PRIMARY KEY, piece_id TEXT, batch_id TEXT, account TEXT, character TEXT, format TEXT,
  scheduled_at REAL, status TEXT, caption TEXT, container_id TEXT, media_id TEXT, published_at REAL,
  error TEXT, simulated INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT, post_id TEXT, fetched_at REAL, age_hours REAL, views REAL,
  reach REAL, saved REAL, shares REAL, interactions REAL, avg_watch_ms REAL, skip_rate REAL,
  simulated INTEGER DEFAULT 0
);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # ------------------------------------------------------------ generic
    def q(self, sql: str, args: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return list(self.db.execute(sql, tuple(args)))

    def one(self, sql: str, args: Iterable[Any] = ()) -> sqlite3.Row | None:
        rows = self.q(sql, args)
        return rows[0] if rows else None

    def insert(self, table: str, row: dict[str, Any]) -> int:
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        cur = self.db.execute(
            f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({marks})", tuple(row.values())
        )
        return int(cur.lastrowid or 0)

    def update(self, table: str, key: str, key_val: Any, **fields: Any) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.db.execute(f"UPDATE {table} SET {sets} WHERE {key} = ?", (*fields.values(), key_val))

    # ------------------------------------------------------------ batches & pieces
    def add_batch(
        self, batch_id: str, character: str, profile: str, plan: dict[str, Any], simulated: bool
    ) -> None:
        self.insert(
            "batches",
            {
                "id": batch_id,
                "created_at": time.time(),
                "character": character,
                "profile": profile,
                "status": "planned",
                "plan_json": json.dumps(plan),
                "simulated": int(simulated),
                "note": "",
            },
        )

    def batch(self, batch_id: str) -> sqlite3.Row:
        row = self.one("SELECT * FROM batches WHERE id = ?", [batch_id])
        if row is None:
            raise KeyError(f"no batch {batch_id}")
        return row

    def latest_batch(self) -> sqlite3.Row | None:
        return self.one("SELECT * FROM batches ORDER BY created_at DESC LIMIT 1")

    def pieces(self, batch_id: str, status: str | None = None) -> list[sqlite3.Row]:
        if status:
            return self.q(
                "SELECT * FROM pieces WHERE batch_id = ? AND status = ? ORDER BY id", [batch_id, status]
            )
        return self.q("SELECT * FROM pieces WHERE batch_id = ? ORDER BY id", [batch_id])

    def set_piece(self, piece_id: str, **fields: Any) -> None:
        self.update("pieces", "id", piece_id, updated_at=time.time(), **fields)

    # ------------------------------------------------------------ executions & gates
    def add_execution(self, **row: Any) -> int:
        return self.insert("executions", row)

    def add_gate_results(
        self, execution_id: int, piece_id: str, batch_id: str, stage: str, results: list[Any]
    ) -> None:
        for r in results:
            self.insert(
                "gate_results",
                {
                    "execution_id": execution_id,
                    "piece_id": piece_id,
                    "batch_id": batch_id,
                    "stage": stage,
                    "gate": r.gate,
                    "passed": None if r.passed is None else int(r.passed),
                    "score": r.score,
                    "reason": r.reason,
                    "metrics_json": json.dumps(r.metrics, default=float),
                },
            )

    def add_spend(
        self, batch_id: str, kind: str, ref: str, seconds: float, usd: float, **detail: Any
    ) -> None:
        self.insert(
            "spend",
            {
                "batch_id": batch_id,
                "kind": kind,
                "ref": ref,
                "seconds": seconds,
                "usd": usd,
                "created_at": time.time(),
                "detail_json": json.dumps(detail, default=str),
            },
        )

    def spend_today(self) -> float:
        since = time.time() - 86400
        row = self.one("SELECT COALESCE(SUM(usd),0) AS s FROM spend WHERE created_at >= ?", [since])
        return float(row["s"]) if row else 0.0
