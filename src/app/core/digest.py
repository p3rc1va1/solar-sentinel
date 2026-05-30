"""Daily MEDIUM-detection digest scheduler.

Runs once a day (default 20:00 local time at the configured location)
and summarises every MEDIUM-confidence detection that arrived since the
previous digest. One Gemini call per digest; sent via the existing
NotificationService and persisted to the `digests` table.

If there were no MEDIUM detections in the window, the digest is skipped
silently — no heartbeat email.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.config import Settings
from app.db.database import Database
from app.services.gemini import GeminiClient
from app.services.notifications import NotificationService

logger = logging.getLogger(__name__)

_FALLBACK_LOOKBACK = timedelta(hours=24)


def _parse_hhmm(value: str, default: time = time(20, 0)) -> time:
    """Parse 'HH:MM' (24-h). Falls back to `default` on failure."""
    try:
        h, m = value.split(":", 1)
        return time(int(h), int(m))
    except (ValueError, AttributeError):
        logger.warning("Invalid digest_time_local %r — falling back to %s", value, default)
        return default


def _resolve_tz(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name or "UTC")
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC")


def _build_prompt(detections: list[dict]) -> str:
    """Format the summarisation prompt for Gemini."""
    rows = [
        {
            "timestamp": d.get("timestamp"),
            "class": d.get("defect_class"),
            "confidence": round(float(d.get("confidence", 0.0)), 3),
            "panel_id": d.get("panel_id"),
        }
        for d in detections
    ]
    return (
        "You are summarising MEDIUM-confidence solar panel detections from the past 24 hours. "
        "Produce a SHORT markdown report with: "
        "(a) a one-paragraph overall summary, "
        "(b) a markdown table of detections (timestamp, class, confidence, panel ID), "
        "(c) any patterns worth investigating. "
        "Be terse — this is a borderline-detection digest, not an alert.\n\n"
        f"Detections (JSON):\n{json.dumps(rows, indent=2)}\n"
    )


class DigestScheduler:
    """Sleeps until the configured local time, then runs one digest."""

    def __init__(
        self,
        db: Database,
        gemini: GeminiClient,
        notif: NotificationService,
        settings: Settings,
    ) -> None:
        self.db = db
        self.gemini = gemini
        self.notif = notif
        self.settings = settings
        self._running = False
        self._task: asyncio.Task | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "DigestScheduler started (time=%s, tz=%s)",
            self.settings.digest_time_local,
            self.settings.weather_timezone or "UTC",
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("DigestScheduler stopped")

    def _next_fire_time(self, now: datetime) -> datetime:
        """Compute the next fire datetime in the configured local tz."""
        tz = _resolve_tz(self.settings.weather_timezone)
        local_now = now.astimezone(tz)
        target_t = _parse_hhmm(self.settings.digest_time_local)
        target = local_now.replace(
            hour=target_t.hour, minute=target_t.minute,
            second=0, microsecond=0,
        )
        if target <= local_now:
            target += timedelta(days=1)
        return target

    async def _run_loop(self) -> None:
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                fire_at = self._next_fire_time(now)
                sleep_s = max(1.0, (fire_at - now.astimezone(fire_at.tzinfo)).total_seconds())
                logger.info(
                    "DigestScheduler sleeping %.1fh until %s",
                    sleep_s / 3600.0, fire_at.isoformat(),
                )
                await asyncio.sleep(sleep_s)

                if not self.settings.digest_enabled:
                    logger.info("Digest disabled — skipping this cycle")
                    continue

                await self.run_once(datetime.now(timezone.utc))

                # Avoid double-firing if the loop wakes early (clock drift).
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("DigestScheduler loop error: %s", e, exc_info=True)
                await asyncio.sleep(300)

    async def run_once(self, now: datetime | None = None) -> dict | None:
        """Build and deliver a digest. Public for tests / manual triggers.

        Returns the persisted digest dict, or None when skipped.
        """
        now = now or datetime.now(timezone.utc)
        last = await self.db.get_last_digest_created_at()
        since = last or (now - _FALLBACK_LOOKBACK)

        detections = await self.db.get_medium_detections_since(
            since=since,
            conf_medium=self.settings.confidence_medium,
            conf_high=self.settings.confidence_high,
        )
        if not detections:
            logger.info("No MEDIUM detections since %s — skipping digest", since.isoformat())
            return None

        prompt = _build_prompt(detections)
        try:
            text, total_tokens, model_name = await self.gemini.generate_with_usage(prompt)
        except Exception as e:
            logger.error("Gemini summarisation failed: %s — skipping digest", e)
            return None

        try:
            await self.db.log_gemini_usage(
                model_name=model_name or "unknown",
                tokens_used=int(total_tokens or 0),
                success=True,
            )
        except Exception:  # nosec
            logger.warning("Failed to log gemini_usage for digest", exc_info=True)

        delivery = await self.notif.send_report(
            report_markdown=text,
            severity="DIGEST",
            image_path=None,
        )

        det_ids = [int(d["id"]) for d in detections if "id" in d]
        digest_id = await self.db.insert_digest(
            detection_ids=det_ids,
            summary_markdown=text,
            sent_email=bool(delivery.get("email", False)),
            sent_telegram=bool(delivery.get("telegram", False)),
        )
        logger.info(
            "Digest %s delivered (count=%d, email=%s, telegram=%s)",
            digest_id, len(det_ids),
            delivery.get("email"), delivery.get("telegram"),
        )
        return await self.db.get_digest(digest_id)
