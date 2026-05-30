"""Tests for app.core.digest — DigestScheduler."""

from datetime import datetime, time, timezone
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest
import pytest_asyncio

from app.config import Settings
from app.core.digest import DigestScheduler, _build_prompt, _parse_hhmm
from app.db.database import Database


def _make_settings(**overrides):
    s = MagicMock(spec=Settings)
    s.confidence_high = 0.70
    s.confidence_medium = 0.45
    s.digest_enabled = True
    s.digest_time_local = "20:00"
    s.weather_timezone = "UTC"
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


@pytest_asyncio.fixture
async def db(tmp_path):
    d = Database(tmp_path / "digest.db")
    await d.connect()
    yield d
    await d.disconnect()


class TestParseHHMM:
    def test_valid(self):
        assert _parse_hhmm("08:30") == time(8, 30)
        assert _parse_hhmm("20:00") == time(20, 0)

    def test_invalid_falls_back(self):
        assert _parse_hhmm("not-a-time") == time(20, 0)
        assert _parse_hhmm("") == time(20, 0)
        assert _parse_hhmm(None) == time(20, 0)


class TestBuildPrompt:
    def test_includes_detections(self):
        rows = [
            {"timestamp": "2026-05-30T12:00:00", "defect_class": "soiling",
             "confidence": 0.55, "panel_id": "panel-1"},
        ]
        prompt = _build_prompt(rows)
        assert "soiling" in prompt
        assert "panel-1" in prompt
        assert "0.55" in prompt


class TestNextFireTime:
    def test_rolls_to_tomorrow_when_past(self):
        s = _make_settings(digest_time_local="08:00", weather_timezone="UTC")
        sched = DigestScheduler(MagicMock(), MagicMock(), MagicMock(), s)
        now = datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc)
        nxt = sched._next_fire_time(now)
        assert nxt.date() == datetime(2026, 6, 2).date()
        assert nxt.hour == 8

    def test_today_when_not_yet_passed(self):
        s = _make_settings(digest_time_local="20:00", weather_timezone="UTC")
        sched = DigestScheduler(MagicMock(), MagicMock(), MagicMock(), s)
        now = datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc)
        nxt = sched._next_fire_time(now)
        assert nxt.date() == datetime(2026, 6, 1).date()
        assert nxt.hour == 20

    def test_uses_configured_timezone(self):
        s = _make_settings(digest_time_local="20:00", weather_timezone="Europe/Vilnius")
        sched = DigestScheduler(MagicMock(), MagicMock(), MagicMock(), s)
        now = datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc)
        nxt = sched._next_fire_time(now)
        # 20:00 Vilnius == 17:00 UTC in summer (UTC+3)
        assert nxt.tzinfo == ZoneInfo("Europe/Vilnius")


class TestRunOnce:
    @pytest.mark.asyncio
    async def test_skip_when_no_medium(self, db):
        gemini = MagicMock()
        gemini.generate_with_usage = AsyncMock()
        notif = MagicMock()
        notif.send_report = AsyncMock()
        sched = DigestScheduler(db, gemini, notif, _make_settings())

        result = await sched.run_once(datetime.now(timezone.utc))
        assert result is None
        gemini.generate_with_usage.assert_not_called()
        notif.send_report.assert_not_called()

    @pytest.mark.asyncio
    async def test_summarises_only_medium(self, db):
        # Seed: one MEDIUM, one HIGH (above), one LOW (below).
        await db.insert_detection(
            image_path="/tmp/m.jpg", defect_class="soiling",
            confidence=0.55, bbox={"x1": 0, "y1": 0, "x2": 1, "y2": 1},
        )
        await db.insert_detection(
            image_path="/tmp/h.jpg", defect_class="crack",
            confidence=0.85, bbox={"x1": 0, "y1": 0, "x2": 1, "y2": 1},
        )
        await db.insert_detection(
            image_path="/tmp/l.jpg", defect_class="dust",
            confidence=0.30, bbox={"x1": 0, "y1": 0, "x2": 1, "y2": 1},
        )
        gemini = MagicMock()
        gemini.generate_with_usage = AsyncMock(
            return_value=("# Digest\n\nOne MEDIUM detection.", 567, "gemini-2.5-flash")
        )
        notif = MagicMock()
        notif.send_report = AsyncMock(return_value={"email": True, "telegram": False})
        sched = DigestScheduler(db, gemini, notif, _make_settings())

        result = await sched.run_once(datetime.now(timezone.utc))
        assert result is not None
        assert result["detection_count"] == 1
        assert result["sent_email"] == 1
        assert result["sent_telegram"] == 0

        # Token usage logged
        usage = await db.get_gemini_usage_today()
        assert usage and usage[0]["model_name"] == "gemini-2.5-flash"
        assert usage[0]["total_tokens"] == 567

        # Notification called with DIGEST severity
        notif.send_report.assert_awaited_once()
        kwargs = notif.send_report.await_args.kwargs
        assert kwargs["severity"] == "DIGEST"
        assert kwargs["image_path"] is None

    @pytest.mark.asyncio
    async def test_subsequent_digest_uses_last_created_at(self, db):
        # First digest covers a MEDIUM detection
        await db.insert_detection(
            image_path="/tmp/a.jpg", defect_class="soiling",
            confidence=0.55, bbox={"x1": 0, "y1": 0, "x2": 1, "y2": 1},
        )
        gemini = MagicMock()
        gemini.generate_with_usage = AsyncMock(return_value=("d1", 10, "m"))
        notif = MagicMock()
        notif.send_report = AsyncMock(return_value={"email": True, "telegram": False})
        sched = DigestScheduler(db, gemini, notif, _make_settings())
        await sched.run_once(datetime.now(timezone.utc))

        # No new MEDIUMs → second digest should be skipped
        gemini.generate_with_usage.reset_mock()
        result2 = await sched.run_once(datetime.now(timezone.utc))
        assert result2 is None
        gemini.generate_with_usage.assert_not_called()
