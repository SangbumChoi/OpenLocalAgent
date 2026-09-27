"""Lightweight experiment + artifact tracking — SQLite metadata + content-addressed storage.

Why not MLflow/W&B here? For a tiny pure-PyTorch project run daily by cron, those are heavy and,
as you noted, duplicate artifacts (every run copies the model/data). The SOTA trick to avoid that
duplication is **content-addressed storage** (the same idea behind Git, DVC, and the HF Hub):
hash the bytes, store each unique blob *once*, and have runs reference the hash. So:

  - run/metric/param metadata  -> a single SQLite DB (queryable, no server)
  - model + dataset artifacts   -> a content-addressed store (cas/<sha256>), deduped automatically

Two identical checkpoints across 30 nightly runs cost one copy, not 30. Swap in MLflow/W&B/Aim
later if you want a UI — the interface here is deliberately tiny.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import time

DEFAULT_DIR = "runs/track"


class Tracker:
    def __init__(self, root: str = DEFAULT_DIR):
        self.root = root
        self.cas = os.path.join(root, "cas")
        os.makedirs(self.cas, exist_ok=True)
        self.db = sqlite3.connect(os.path.join(root, "runs.db"))
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY, name TEXT, params TEXT,
            started REAL, ended REAL, status TEXT);
        CREATE TABLE IF NOT EXISTS metrics(run_id INT, step INT, key TEXT, value REAL, ts REAL);
        CREATE TABLE IF NOT EXISTS artifacts(run_id INT, kind TEXT, name TEXT, sha TEXT,
            bytes INT, ts REAL);
        """)
        self.db.commit()

    # ---- runs / metrics ----
    def start_run(self, name: str, params: dict | None = None) -> int:
        cur = self.db.execute("INSERT INTO runs(name,params,started,status) VALUES(?,?,?,?)",
                              (name, json.dumps(params or {}), time.time(), "running"))
        self.db.commit()
        return cur.lastrowid

    def log_metric(self, run_id: int, step: int, key: str, value: float):
        self.db.execute("INSERT INTO metrics VALUES(?,?,?,?,?)",
                        (run_id, step, key, float(value), time.time()))
        self.db.commit()

    def end_run(self, run_id: int, status: str = "done"):
        self.db.execute("UPDATE runs SET ended=?,status=? WHERE id=?",
                        (time.time(), status, run_id))
        self.db.commit()

    # ---- content-addressed artifacts (dedup) ----
    def log_artifact(self, run_id: int, path: str, kind: str, name: str | None = None) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sha = h.hexdigest()
        dst = os.path.join(self.cas, sha)
        if not os.path.exists(dst):                       # store each unique blob once
            shutil.copyfile(path, dst)
        self.db.execute("INSERT INTO artifacts VALUES(?,?,?,?,?,?)",
                        (run_id, kind, name or os.path.basename(path), sha,
                         os.path.getsize(dst), time.time()))
        self.db.commit()
        return sha

    def artifact_path(self, sha: str) -> str:
        return os.path.join(self.cas, sha)

    def latest_artifact(self, kind: str) -> str | None:
        """Path to the most recently logged artifact of `kind` (for resume), or None."""
        row = self.db.execute(
            "SELECT sha FROM artifacts WHERE kind=? ORDER BY ts DESC LIMIT 1", (kind,)).fetchone()
        return self.artifact_path(row[0]) if row else None

    def summary(self) -> dict:
        runs = self.db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        arts = self.db.execute("SELECT COUNT(*), COUNT(DISTINCT sha) FROM artifacts").fetchone()
        bytes_unique = self.db.execute(
            "SELECT COALESCE(SUM(b),0) FROM (SELECT DISTINCT sha, bytes b FROM artifacts)"
        ).fetchone()[0]
        return {"runs": runs, "artifact_rows": arts[0], "unique_blobs": arts[1],
                "dedup_saved_rows": arts[0] - arts[1], "cas_bytes": bytes_unique}
