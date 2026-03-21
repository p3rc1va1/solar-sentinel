"""SQLite database layer using aiosqlite."""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    image_path TEXT NOT NULL,
    defect_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    bbox_json TEXT NOT NULL,
    panel_id TEXT NOT NULL DEFAULT 'panel-1',
    triaged INTEGER NOT NULL DEFAULT 0,
    report_id INTEGER,
    FOREIGN KEY (report_id) REFERENCES reports(id)
);

CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    severity TEXT NOT NULL,
    urgency TEXT NOT NULL,
    root_cause TEXT NOT NULL,
    trend_analysis TEXT NOT NULL,
    report_markdown TEXT NOT NULL,
    qa_score INTEGER NOT NULL,
    qa_approved INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (detection_id) REFERENCES detections(id)
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gemini_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    model_name TEXT NOT NULL,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    success INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_reports_created_at ON reports(created_at);
"""


def _utcnow() -> datetime:
    """Timezone-aware UTC now (replaces deprecated datetime.utcnow())."""
    return datetime.now(timezone.utc)


class Database:
    """Async SQLite database wrapper."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open connection and ensure schema exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._db.commit()
        logger.info("Database connected: %s", self.db_path)

    async def disconnect(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._db

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_detection(row: aiosqlite.Row) -> dict:
        """Convert a detection row to a dict, parsing bbox_json."""
        d = dict(row)
        d["bbox"] = json.loads(d.pop("bbox_json"))
        return d

    # ── Detections ──────────────────────────────────────────────────────

    async def insert_detection(
        self,
        image_path: str,
        defect_class: str,
        confidence: float,
        bbox: dict,
        panel_id: str = "panel-1",
    ) -> int:
        cursor = await self.db.execute(
            """INSERT INTO detections (image_path, defect_class, confidence, bbox_json, panel_id)
               VALUES (?, ?, ?, ?, ?)""",
            (image_path, defect_class, confidence, json.dumps(bbox), panel_id),
        )
        await self.db.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def get_detection(self, detection_id: int) -> dict | None:
        cursor = await self.db.execute(
            "SELECT * FROM detections WHERE id = ?", (detection_id,)
        )
        row = await cursor.fetchone()
        return self._row_to_detection(row) if row else None

    async def list_detections(self, limit: int = 50, offset: int = 0) -> list[dict]:
        cursor = await self.db.execute(
            "SELECT * FROM detections ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [self._row_to_detection(row) for row in await cursor.fetchall()]

    async def get_recent_detections(self, hours: int = 1) -> list[dict]:
        """Get detections within the last N hours (for triage deduplication)."""
        cutoff = (_utcnow() - timedelta(hours=hours)).isoformat()
        cursor = await self.db.execute(
            "SELECT * FROM detections WHERE timestamp >= ? ORDER BY timestamp DESC",
            (cutoff,),
        )
        return [self._row_to_detection(row) for row in await cursor.fetchall()]

    # ── Reports ─────────────────────────────────────────────────────────

    async def insert_report(
        self,
        detection_id: int,
        severity: str,
        urgency: str,
        root_cause: str,
        trend_analysis: str,
        report_markdown: str,
        qa_score: int,
        qa_approved: bool,
    ) -> int:
        cursor = await self.db.execute(
            """INSERT INTO reports
               (detection_id, severity, urgency, root_cause, trend_analysis,
                report_markdown, qa_score, qa_approved)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                detection_id, severity, urgency, root_cause,
                trend_analysis, report_markdown, qa_score,
                1 if qa_approved else 0,
            ),
        )
        await self.db.execute(
            "UPDATE detections SET report_id = ? WHERE id = ?",
            (cursor.lastrowid, detection_id),
        )
        await self.db.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def get_report(self, report_id: int) -> dict | None:
        cursor = await self.db.execute(
            "SELECT * FROM reports WHERE id = ?", (report_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_reports(self, limit: int = 50, offset: int = 0) -> list[dict]:
        cursor = await self.db.execute(
            "SELECT * FROM reports ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [dict(row) for row in await cursor.fetchall()]

    async def get_reports_since(self, days: int = 7) -> list[dict]:
        """Fetch reports from the last N days for historical context."""
        cutoff = (_utcnow() - timedelta(days=days)).isoformat()
        cursor = await self.db.execute(
            "SELECT * FROM reports WHERE created_at >= ? ORDER BY created_at DESC",
            (cutoff,),
        )
        return [dict(row) for row in await cursor.fetchall()]

    # ── Settings ────────────────────────────────────────────────────────

    async def get_setting(self, key: str) -> str | None:
        cursor = await self.db.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        )
        row = await cursor.fetchone()
        return row["value"] if row else None

    async def set_setting(self, key: str, value: str) -> None:
        await self.db.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, value),
        )
        await self.db.commit()

    # ── Gemini Usage Tracking ───────────────────────────────────────────

    async def log_gemini_usage(
        self, model_name: str, tokens_used: int, success: bool
    ) -> None:
        await self.db.execute(
            "INSERT INTO gemini_usage (model_name, tokens_used, success) VALUES (?, ?, ?)",
            (model_name, tokens_used, 1 if success else 0),
        )
        await self.db.commit()

    async def get_gemini_usage_today(self) -> list[dict]:
        today = _utcnow().strftime("%Y-%m-%d")
        cursor = await self.db.execute(
            "SELECT model_name, COUNT(*) as count, SUM(tokens_used) as total_tokens "
            "FROM gemini_usage WHERE timestamp >= ? GROUP BY model_name",
            (today,),
        )
        return [dict(row) for row in await cursor.fetchall()]
