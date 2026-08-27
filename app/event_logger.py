"""
Event Logger — SQLite-backed event store for IBVAP
===================================================
Uses only Python's built-in sqlite3 module (no SQLAlchemy).

Schema
------
  events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id     TEXT,
    timestamp     TEXT,
    alert_level   TEXT,
    threat_score  REAL,
    weapon_detected INTEGER,
    weapon_labels   TEXT,        -- JSON array string, e.g. '["pistol"]'
    emotion         TEXT,
    pose_anomaly    TEXT,
    fence_breaches  TEXT,        -- JSON array string
    anpr_plates     TEXT,        -- JSON array string
    anpr_hits       TEXT,        -- JSON array string
    snapshot_path   TEXT,
    raw_payload     TEXT         -- full JSON for forward compatibility
  )

Rules
-----
  • Only WARNING and CRITICAL events are stored.
  • On CRITICAL: save annotated frame as JPEG to snapshots_dir.
  • Auto-prune snapshots_dir when > max_snapshots (delete oldest by mtime).
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id       TEXT    NOT NULL,
    timestamp       TEXT    NOT NULL,
    alert_level     TEXT    NOT NULL,
    threat_score    REAL    NOT NULL,
    weapon_detected INTEGER NOT NULL DEFAULT 0,
    weapon_labels   TEXT    NOT NULL DEFAULT '[]',
    emotion         TEXT    NOT NULL DEFAULT 'neutral',
    pose_anomaly    TEXT    NOT NULL DEFAULT 'normal',
    fence_breaches  TEXT    NOT NULL DEFAULT '[]',
    anpr_plates     TEXT    NOT NULL DEFAULT '[]',
    anpr_hits       TEXT    NOT NULL DEFAULT '[]',
    snapshot_path   TEXT    DEFAULT NULL,
    raw_payload     TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_camera_level
    ON events (camera_id, alert_level);
CREATE INDEX IF NOT EXISTS idx_timestamp
    ON events (timestamp);
"""


class EventLogger:
    """
    Thread-safe SQLite event logger.

    Parameters
    ----------
    db_path       : Path to the SQLite database file.
    snapshots_dir : Directory to write CRITICAL frame snapshots.
    max_snapshots : Maximum number of snapshot files to keep.
    """

    def __init__(
        self,
        db_path: str = "data/databases/events.db",
        snapshots_dir: str = "data/snapshots/",
        max_snapshots: int = 500,
    ) -> None:
        self._db_path      = db_path
        self._snapshots_dir = snapshots_dir
        self._max_snapshots = max_snapshots
        self._lock         = threading.Lock()

        # Ensure directories exist
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        os.makedirs(snapshots_dir, exist_ok=True)

        # Initialise schema
        try:
            conn = self._connect()
            conn.executescript(_SCHEMA)
            conn.commit()
            conn.close()
            logger.info(f"[EventLogger] DB ready at {db_path}")
        except Exception as exc:
            logger.error(f"[EventLogger] Failed to initialise DB: {exc}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")   # write-ahead log for concurrency
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _save_snapshot(
        self, frame: np.ndarray, camera_id: str, timestamp_str: str
    ) -> Optional[str]:
        """
        Save annotated frame as JPEG. Returns the file path, or None on failure.
        Then prunes the snapshots dir if it exceeds max_snapshots.
        """
        try:
            # Filename: CAM_01_20260824_143201.jpg
            ts_clean = timestamp_str[:19].replace("T", "_").replace(":", "").replace("-", "")
            filename = f"{camera_id}_{ts_clean}.jpg"
            filepath = os.path.join(self._snapshots_dir, filename)
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self._prune_snapshots()
            return filepath
        except Exception as exc:
            logger.warning(f"[EventLogger] Snapshot save failed: {exc}")
            return None

    def _prune_snapshots(self) -> None:
        """Delete oldest snapshots if count exceeds max_snapshots."""
        try:
            files = sorted(
                Path(self._snapshots_dir).glob("*.jpg"),
                key=lambda p: p.stat().st_mtime,
            )
            excess = len(files) - self._max_snapshots
            for f in files[:excess]:
                f.unlink(missing_ok=True)
        except Exception as exc:
            logger.debug(f"[EventLogger] Prune error: {exc}")

    # ── Public API ────────────────────────────────────────────────────────────

    def log(
        self,
        camera_id: str,
        alert_level: str,
        threat_score: float,
        weapon_detected: bool = False,
        weapon_labels: Optional[List[str]] = None,
        emotion: str = "neutral",
        pose_anomaly: str = "normal",
        fence_breaches: Optional[List[Dict[str, Any]]] = None,
        anpr_plates: Optional[List[str]] = None,
        anpr_hits: Optional[List[str]] = None,
        frame: Optional[np.ndarray] = None,
        raw_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an event. Only WARNING and CRITICAL are persisted.
        On CRITICAL, saves a snapshot if *frame* is provided.
        """
        if alert_level not in ("WARNING", "CRITICAL"):
            return

        timestamp = datetime.now(timezone.utc).isoformat()
        snapshot_path: Optional[str] = None

        # Save snapshot for CRITICAL events
        if alert_level == "CRITICAL" and frame is not None:
            snapshot_path = self._save_snapshot(frame, camera_id, timestamp)

        row = {
            "camera_id":       camera_id,
            "timestamp":       timestamp,
            "alert_level":     alert_level,
            "threat_score":    round(float(threat_score), 2),
            "weapon_detected": 1 if weapon_detected else 0,
            "weapon_labels":   json.dumps(weapon_labels or []),
            "emotion":         emotion,
            "pose_anomaly":    pose_anomaly,
            "fence_breaches":  json.dumps(fence_breaches or []),
            "anpr_plates":     json.dumps(anpr_plates or []),
            "anpr_hits":       json.dumps(anpr_hits or []),
            "snapshot_path":   snapshot_path,
            "raw_payload":     json.dumps(raw_payload or {}),
        }

        sql = """
        INSERT INTO events
            (camera_id, timestamp, alert_level, threat_score,
             weapon_detected, weapon_labels, emotion, pose_anomaly,
             fence_breaches, anpr_plates, anpr_hits, snapshot_path, raw_payload)
        VALUES
            (:camera_id, :timestamp, :alert_level, :threat_score,
             :weapon_detected, :weapon_labels, :emotion, :pose_anomaly,
             :fence_breaches, :anpr_plates, :anpr_hits, :snapshot_path, :raw_payload)
        """
        try:
            with self._lock:
                conn = self._connect()
                conn.execute(sql, row)
                conn.commit()
                conn.close()
        except Exception as exc:
            logger.error(f"[EventLogger] Insert failed: {exc}")

    def query(
        self,
        camera_id: Optional[str] = None,
        level: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Query events from the database.

        Parameters
        ----------
        camera_id : Filter by camera ID (None = all cameras).
        level     : Filter by alert level: "WARNING" | "CRITICAL" (None = both).
        limit     : Maximum number of rows to return (most recent first).

        Returns
        -------
        list of dicts with all event fields. JSON fields are decoded.
        """
        conditions = []
        params: list[Any] = []

        if camera_id:
            conditions.append("camera_id = ?")
            params.append(camera_id)
        if level and level in ("WARNING", "CRITICAL"):
            conditions.append("alert_level = ?")
            params.append(level)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"""
        SELECT * FROM events
        {where}
        ORDER BY id DESC
        LIMIT ?
        """
        params.append(max(1, min(limit, 1000)))

        try:
            with self._lock:
                conn = self._connect()
                rows = conn.execute(sql, params).fetchall()
                conn.close()
        except Exception as exc:
            logger.error(f"[EventLogger] Query failed: {exc}")
            return []

        results: List[Dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            # Decode JSON fields
            for field_name in ("weapon_labels", "fence_breaches", "anpr_plates", "anpr_hits", "raw_payload"):
                try:
                    d[field_name] = json.loads(d.get(field_name) or "[]")
                except Exception:
                    pass
            d["weapon_detected"] = bool(d.get("weapon_detected", 0))
            results.append(d)

        return results

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """Return metadata list of all snapshot files (newest first)."""
        try:
            files = sorted(
                Path(self._snapshots_dir).glob("*.jpg"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            return [
                {
                    "filename": f.name,
                    "path": str(f),
                    "size_bytes": f.stat().st_size,
                    "mtime": f.stat().st_mtime,
                }
                for f in files
            ]
        except Exception as exc:
            logger.warning(f"[EventLogger] list_snapshots error: {exc}")
            return []
