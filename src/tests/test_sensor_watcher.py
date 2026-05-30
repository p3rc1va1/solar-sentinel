"""Tests for app.core.sensor_watcher."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.sensor_watcher import SensorWatcher


def _make_settings(**overrides):
    s = MagicMock()
    s.sensor_trigger_enabled = True
    s.sensor_temp_high_c = 35.0
    s.sensor_temp_low_c = 0.0
    s.sensor_humidity_high_pct = 85.0
    s.sensor_trigger_cooldown_minutes = 15
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _make_sensor(reading):
    sensor = MagicMock()
    sensor.read = MagicMock(return_value=reading)
    return sensor


def _make_scheduler():
    sched = MagicMock()
    sched.capture_once = AsyncMock(return_value=[])
    return sched


class TestSensorWatcher:
    @pytest.mark.asyncio
    async def test_temp_high_triggers(self):
        sensor = _make_sensor({"temperature": 36.5, "humidity": 50.0})
        sched = _make_scheduler()
        w = SensorWatcher(sensor, sched, _make_settings())
        triggered = await w.check_once()
        assert triggered == ["temp_high"]

    @pytest.mark.asyncio
    async def test_temp_low_triggers(self):
        sensor = _make_sensor({"temperature": -2.0, "humidity": 50.0})
        w = SensorWatcher(sensor, _make_scheduler(), _make_settings())
        assert await w.check_once() == ["temp_low"]

    @pytest.mark.asyncio
    async def test_humidity_triggers(self):
        sensor = _make_sensor({"temperature": 22.0, "humidity": 90.0})
        w = SensorWatcher(sensor, _make_scheduler(), _make_settings())
        assert await w.check_once() == ["humidity_high"]

    @pytest.mark.asyncio
    async def test_within_thresholds_no_trigger(self):
        sensor = _make_sensor({"temperature": 22.0, "humidity": 50.0})
        w = SensorWatcher(sensor, _make_scheduler(), _make_settings())
        assert await w.check_once() == []

    @pytest.mark.asyncio
    async def test_cooldown_suppresses_repeat(self):
        sensor = _make_sensor({"temperature": 36.5, "humidity": 50.0})
        w = SensorWatcher(sensor, _make_scheduler(), _make_settings())
        t0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        assert await w.check_once(now=t0) == ["temp_high"]
        # 5 min later — still within 15 min cooldown
        assert await w.check_once(now=t0 + timedelta(minutes=5)) == []

    @pytest.mark.asyncio
    async def test_fires_again_after_cooldown(self):
        sensor = _make_sensor({"temperature": 36.5, "humidity": 50.0})
        w = SensorWatcher(sensor, _make_scheduler(), _make_settings())
        t0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        await w.check_once(now=t0)
        assert await w.check_once(now=t0 + timedelta(minutes=20)) == ["temp_high"]

    @pytest.mark.asyncio
    async def test_per_channel_cooldown_independent(self):
        """A stuck-high temp can't suppress a humidity trigger."""
        sensor = _make_sensor({"temperature": 36.5, "humidity": 50.0})
        w = SensorWatcher(sensor, _make_scheduler(), _make_settings())
        t0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        await w.check_once(now=t0)  # fires temp_high
        sensor.read.return_value = {"temperature": 36.5, "humidity": 90.0}
        # Within cooldown for temp_high, but humidity_high is fresh.
        assert await w.check_once(now=t0 + timedelta(minutes=2)) == ["humidity_high"]

    @pytest.mark.asyncio
    async def test_disabled_no_trigger(self):
        sensor = _make_sensor({"temperature": 36.5, "humidity": 90.0})
        w = SensorWatcher(sensor, _make_scheduler(), _make_settings(sensor_trigger_enabled=False))
        assert await w.check_once() == []

    @pytest.mark.asyncio
    async def test_none_reading_no_trigger(self):
        sensor = _make_sensor(None)
        w = SensorWatcher(sensor, _make_scheduler(), _make_settings())
        assert await w.check_once() == []

    @pytest.mark.asyncio
    async def test_calls_capture_once(self):
        sensor = _make_sensor({"temperature": 36.5, "humidity": 50.0})
        sched = _make_scheduler()
        w = SensorWatcher(sensor, sched, _make_settings())
        await w.check_once()
        # capture_once is invoked via create_task — let it run
        import asyncio
        await asyncio.sleep(0)
        sched.capture_once.assert_called_once()
