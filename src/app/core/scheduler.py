"""Adaptive capture scheduler.

Manages daylight-aware, interval-based frame capture with
adaptive frequency based on recent detection results.
"""

import asyncio
import logging
from datetime import datetime, time, timezone

from PIL import Image

from app.config import Settings
from app.core.camera import Camera
from app.core.detector import Detection, Detector
from app.core.triage import TriageAgent, check_frame_quality
from app.db.database import Database

logger = logging.getLogger(__name__)

DEFAULT_SUNRISE = time(6, 0)
DEFAULT_SUNSET = time(20, 0)


class CaptureScheduler:
    """Daylight-aware adaptive capture scheduler."""

    def __init__(
        self,
        camera: Camera,
        detector: Detector,
        triage: TriageAgent,
        db: Database,
        settings: Settings,
        on_high_detection=None,
        on_medium_detection=None,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.triage = triage
        self.db = db
        self.settings = settings
        self.on_high_detection = on_high_detection
        self.on_medium_detection = on_medium_detection

        self._running = False
        self._task: asyncio.Task | None = None
        self._current_interval = settings.capture_interval_minutes
        self._consecutive_clean = 0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def current_interval_minutes(self) -> int:
        return self._current_interval

    def _is_daylight(self) -> bool:
        """Check if current time is within daylight hours."""
        now = datetime.now().time()
        return DEFAULT_SUNRISE <= now <= DEFAULT_SUNSET

    async def start(self) -> None:
        """Start the capture loop."""
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Capture scheduler started (interval: %d min)", self._current_interval)

    async def stop(self) -> None:
        """Stop the capture loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Capture scheduler stopped")

    async def _run_loop(self) -> None:
        """Main capture loop."""
        while self._running:
            try:
                if self._is_daylight():
                    await self._capture_and_process()
                else:
                    logger.debug("Outside daylight hours — sleeping")

                await asyncio.sleep(self._current_interval * 60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Scheduler loop error: %s", e, exc_info=True)
                await asyncio.sleep(60)

    async def capture_once(self) -> list[Detection]:
        """Perform a single capture-and-process cycle (for manual triggers)."""
        return await self._capture_and_process()

    async def _capture_and_process(self) -> list[Detection]:
        """Capture a frame, run inference, apply triage, and handle results."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        image_path = self.settings.detections_dir / f"capture_{timestamp}.jpg"
        self.camera.capture_to_file(image_path)

        img = Image.open(image_path)
        if not check_frame_quality(img):
            logger.info("Frame failed quality check, skipping")
            image_path.unlink(missing_ok=True)
            return []

        detections = self.detector.detect(img)

        if not detections:
            self._consecutive_clean += 1
            self._adapt_interval()
            return []

        for det in detections:
            await self.db.insert_detection(
                image_path=str(image_path),
                defect_class=det.class_name,
                confidence=det.confidence,
                bbox={
                    "x1": det.x1, "y1": det.y1,
                    "x2": det.x2, "y2": det.y2,
                },
                panel_id="panel-1",
            )

        recent = await self.db.get_recent_detections(hours=1)
        passed = self.triage.filter_detections(
            detections, recent, self.settings.confidence_medium
        )

        for det in passed:
            if det.confidence >= self.settings.confidence_high:
                self._consecutive_clean = 0
                self._current_interval = self.settings.capture_interval_after_high
                logger.info(
                    "HIGH detection: %s (%.2f) — triggering CrewAI",
                    det.class_name, det.confidence,
                )
                if self.on_high_detection:
                    asyncio.create_task(
                        self.on_high_detection(det, str(image_path))
                    )
            elif det.confidence >= self.settings.confidence_medium:
                logger.info(
                    "MEDIUM detection: %s (%.2f) — queued for digest",
                    det.class_name, det.confidence,
                )
                if self.on_medium_detection:
                    asyncio.create_task(
                        self.on_medium_detection(det, str(image_path))
                    )

        if not passed:
            self._consecutive_clean += 1
        else:
            self._consecutive_clean = 0

        self._adapt_interval()
        return passed

    def _adapt_interval(self) -> None:
        """Adapt capture interval based on recent results."""
        if self._consecutive_clean >= 6:
            self._current_interval = self.settings.capture_interval_after_clean
        elif self._consecutive_clean == 0:
            pass  # keep high frequency
        else:
            self._current_interval = self.settings.capture_interval_minutes
