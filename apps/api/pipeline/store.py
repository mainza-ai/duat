"""
duat/apps/api/pipeline/store.py

SQLite persistence + content-addressed artifact store (stdlib only, no new deps).

- Jobs: durable records for long Atlas work (submit once, poll, resume, cancel).
- Operations: every accepted Atlas op ID with its task, status, request hash.
- Ensembles: provenance manifests keyed by SHA-256.
- Artifacts: bytes stored at data/artifacts/sha256/xx/rest... (dedup by hash).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_lock = threading.Lock()
_DB_PATH: Optional[str] = None


def db_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    raw = os.getenv("DUAT_DB_PATH", str(root / "data" / "duat.db"))
    return Path(raw)


def artifact_path(digest: str) -> Path:
    root = Path(__file__).resolve().parents[3]
    base = Path(os.getenv("DUAT_ARTIFACT_DIR", str(root / "data" / "artifacts" / "sha256")))
    return base / digest[:2] / digest[2:]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    request_json TEXT NOT NULL DEFAULT '{}',
    progress REAL NOT NULL DEFAULT 0.0,
    result_json TEXT,
    error_json TEXT,
    created_ts REAL NOT NULL,
    updated_ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS operations (
    op_id TEXT PRIMARY KEY,
    job_id TEXT,
    task TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'running',
    request_hash TEXT NOT NULL DEFAULT '',
    response_json TEXT,
    created_ts REAL NOT NULL,
    updated_ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS ensembles (
    provenance_sha256 TEXT PRIMARY KEY,
    manifest_json TEXT NOT NULL,
    created_ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS frames (
    id TEXT PRIMARY KEY,
    job_id TEXT,
    op_id TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT 'ensemble',
    camera_json TEXT NOT NULL DEFAULT '{}',
    rgb_sha TEXT NOT NULL DEFAULT '',
    depth_sha TEXT NOT NULL DEFAULT '',
    provenance_sha TEXT NOT NULL DEFAULT '',
    created_ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_ops_job ON operations(job_id);
CREATE INDEX IF NOT EXISTS idx_frames_job ON frames(job_id);
"""


def _connect() -> sqlite3.Connection:
    # Self-initializing: every connection ensures the schema, so writers never
    # depend on startup ordering (tests, workers, CLIs all safe).
    con = sqlite3.connect(str(db_path()), timeout=30.0)
    con.row_factory = sqlite3.Row
    con.executescript(_SCHEMA)
    return con


def init_db() -> None:
    db_path().parent.mkdir(parents=True, exist_ok=True)
    artifact_path("00").parent.mkdir(parents=True, exist_ok=True)
    with _connect():
        pass


def _now() -> float:
    return time.time()


# ---------------- jobs ----------------
def create_job(job_id: str, kind: str, request: Dict[str, Any]) -> None:
    now = _now()
    with _lock, _connect() as con:
        con.execute(
            "INSERT INTO jobs(id, kind, status, request_json, progress, created_ts, updated_ts)"
            " VALUES (?,?,?,?,0.0,?,?)",
            (job_id, kind, "pending", json.dumps(request), now, now))


def update_job(job_id: str, *, status: Optional[str] = None,
               progress: Optional[float] = None,
               result: Optional[Dict[str, Any]] = None,
               error: Optional[Dict[str, Any]] = None) -> None:
    sets, vals = ["updated_ts=?"], [_now()]
    if status is not None:
        sets.append("status=?"); vals.append(status)
    if progress is not None:
        sets.append("progress=?"); vals.append(progress)
    if result is not None:
        sets.append("result_json=?"); vals.append(json.dumps(result))
    if error is not None:
        sets.append("error_json=?"); vals.append(json.dumps(error))
    vals.append(job_id)
    with _lock, _connect() as con:
        con.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id=?", vals)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock, _connect() as con:
        row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None


def list_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    with _lock, _connect() as con:
        rows = con.execute(
            "SELECT * FROM jobs ORDER BY created_ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


# ---------------- operations ----------------
def record_operation(op_id: str, task: str, request_hash: str,
                     job_id: Optional[str] = None, status: str = "running") -> None:
    now = _now()
    with _lock, _connect() as con:
        con.execute(
            "INSERT OR REPLACE INTO operations(op_id, job_id, task, status, request_hash, created_ts, updated_ts)"
            " VALUES (?,?,?,?,?,?,?)",
            (op_id, job_id, task, status, request_hash, now, now))


def update_operation(op_id: str, status: str, response: Optional[Dict[str, Any]] = None) -> None:
    with _lock, _connect() as con:
        con.execute(
            "UPDATE operations SET status=?, response_json=?, updated_ts=? WHERE op_id=?",
            (status, json.dumps(response) if response is not None else None, _now(), op_id))


def get_operation(op_id: str) -> Optional[Dict[str, Any]]:
    with _lock, _connect() as con:
        row = con.execute("SELECT * FROM operations WHERE op_id=?", (op_id,)).fetchone()
        return dict(row) if row else None


def list_operations(job_id: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
    with _lock, _connect() as con:
        clauses, params = [], []
        if job_id is not None:
            clauses.append("job_id=?")
            params.append(job_id)
        if status is not None:
            clauses.append("status=?")
            params.append(status)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = con.execute(f"SELECT * FROM operations{where} ORDER BY created_ts DESC", tuple(params)).fetchall()
        return [dict(r) for r in rows]


def recover_orphaned_jobs() -> int:
    """Mark any jobs left in queued/running state from a previous crashed process
    as interrupted so operators can inspect or resume them cleanly."""
    now = _now()
    with _lock, _connect() as con:
        cur = con.execute(
            "UPDATE jobs SET status='failed', error_json=?, updated_ts=? WHERE status IN ('queued','running')",
            (json.dumps({"code": "ORPHANED_ON_RESTART", "message": "Server restarted while job was in flight; use resume_hint or resubmit."}), now),
        )
        return int(cur.rowcount or 0)


# ---------------- ensembles ----------------
def record_ensemble(provenance_sha256: str, manifest: Dict[str, Any]) -> None:
    with _lock, _connect() as con:
        con.execute(
            "INSERT OR REPLACE INTO ensembles(provenance_sha256, manifest_json, created_ts)"
            " VALUES (?,?,?)",
            (provenance_sha256, json.dumps(manifest), _now()))


def get_ensemble(provenance_sha256: str) -> Optional[Dict[str, Any]]:
    with _lock, _connect() as con:
        row = con.execute("SELECT * FROM ensembles WHERE provenance_sha256=?",
                          (provenance_sha256,)).fetchone()
        return dict(row) if row else None


# ---------------- frames (persistent world state) ----------------
def record_frame(job_id: Optional[str], op_id: str, kind: str,
                 camera: Dict[str, Any], rgb_sha: str, depth_sha: str = "",
                 provenance_sha: str = "") -> str:
    import uuid as _uuid

    fid = "fr_" + _uuid.uuid4().hex[:16]
    with _lock, _connect() as con:
        con.execute(
            "INSERT INTO frames(id, job_id, op_id, kind, camera_json, rgb_sha,"
            " depth_sha, provenance_sha, created_ts) VALUES (?,?,?,?,?,?,?,?,?)",
            (fid, job_id, op_id, kind, json.dumps(camera), rgb_sha,
             depth_sha, provenance_sha, _now()))
    return fid


def get_frame(frame_id: str) -> Optional[Dict[str, Any]]:
    with _lock, _connect() as con:
        row = con.execute("SELECT * FROM frames WHERE id=?", (frame_id,)).fetchone()
        return dict(row) if row else None


def list_frames(limit: int = 100, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
    with _lock, _connect() as con:
        if job_id:
            rows = con.execute(
                "SELECT * FROM frames WHERE job_id=? ORDER BY created_ts DESC LIMIT ?",
                (job_id, limit)).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM frames ORDER BY created_ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


# ---------------- artifacts ----------------
def store_bytes(data: bytes, kind: str = "bin") -> Dict[str, str]:
    digest = hashlib.sha256(data).hexdigest()
    path = artifact_path(digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)
    return {"sha256": digest, "path": str(path), "kind": kind, "bytes": len(data)}


def load_bytes(digest: str) -> bytes:
    return artifact_path(digest).read_bytes()
