"""DHT22-driven out-of-band capture trigger.

Polls the sensor every 60 s; when temperature crosses
`sensor_temp_high_c` / `sensor_temp_low_c` or humidity crosses
`sensor_humidity_high_pct`, fires a one-shot `scheduler.capture_once()`
through the normal triage + confidence pipeline.

Each channel (`temp_high`, `temp_low`, `humidity_high`) has its own
cooldown so a stuck-high temperature can't suppress a humidity trigger.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from app.config import Settings
from app.core.scheduler import CaptureScheduler
from app.core.sensor import DHTSensor

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 60


class SensorWatcher:
    """Triggers an extra capture when DHT22 thresholds are crossed."""

    def __init__(
        self,
        sensor: DHTSensor,
        scheduler: CaptureScheduler,
        settings: Settings,
    ) -> None:
        self.sensor = sensor
        self.scheduler = scheduler
        self.settings = settings
        self._running = False
        self._task: asyncio.Task | None = None
        # Per-channel last-fire timestamps.
        self._last_fire: dict[str, datetime] = {}

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("SensorWatcher started (poll: %ds)", POLL_INTERVAL_SECONDS)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("SensorWatcher stopped")

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self.check_once()
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("SensorWatcher loop error: %s", e, exc_info=True)
                await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def check_once(self, now: datetime | None = None) -> list[str]:
        """Read the sensor and trigger a capture if any threshold is crossed.

        Returns the list of channels that fired (for testing / logging).
        Public so tests can drive it deterministically without sleeping.
        """
        if not self.settings.sensor_trigger_enabled:
            return []

        reading = self.sensor.read()
        if reading is None:
            logger.debug("SensorWatcher: no reading available, skipping")
            return []

        now = now or datetime.now(timezone.utc)
        cooldown = timedelta(minutes=self.settings.sensor_trigger_cooldown_minutes)
        triggered: list[str] = []

        crossings = self._evaluate(reading)
        for channel in crossings:
            last = self._last_fire.get(channel)
            if last is not None and (now - last) < cooldown:
                logger.debug(
                    "SensorWatcher: %s within cooldown (%s ago), skipping",
                    channel, now - last,
                )
                continue
            self._last_fire[channel] = now
            triggered.append(channel)
            logger.info(
                "SensorWatcher triggered by %s (temp=%.1f°C, humidity=%.1f%%) — capturing",
                channel, reading["temperature"], reading["humidity"],
            )
            asyncio.create_task(self._safe_capture())

        return triggered

    def _evaluate(self, reading: dict) -> list[str]:
        """Return the list of channel names whose thresholds are crossed."""
        crossed = []
        temp = reading.get("temperature")
        hum = reading.get("humidity")
        if temp is not None:
            if temp > self.settings.sensor_temp_high_c:
                crossed.append("temp_high")
            if temp < self.settings.sensor_temp_low_c:
                crossed.append("temp_low")
        if hum is not None and hum > self.settings.sensor_humidity_high_pct:
            crossed.append("humidity_high")
        return crossed

    async def _safe_capture(self) -> None:
        """Wrap scheduler.capture_once so a failure can't kill the watcher."""
        try:
            await self.scheduler.capture_once()
        except Exception as e:
            logger.error("Sensor-triggered capture failed: %s", e, exc_info=True)
