"""Tests for app.core.scheduler."""

import asyncio
from datetime import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from PIL import Image

from app.config import Settings
from app.core.camera import Camera
from app.core.detector import Detection, Detector
from app.core.scheduler import CaptureScheduler, DEFAULT_SUNRISE, DEFAULT_SUNSET
from app.core.triage import TriageAgent
from app.db.database import Database


@pytest.fixture
def mock_settings(tmp_path):
    s = MagicMock(spec=Settings)
    s.capture_interval_minutes = 15
    s.capture_interval_after_high = 5
    s.capture_interval_after_clean = 30
    s.confidence_high = 0.70
    s.confidence_medium = 0.45
    s.detections_dir = tmp_path / "detections"
    s.detections_dir.mkdir()
    return s


@pytest.fixture
def mock_camera():
    cam = MagicMock(spec=Camera)

    def _capture_to_file(path):
        """Create a real stub JPEG so Image.open() works."""
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color=(128, 128, 128)).save(str(path), "JPEG")
        return path

    cam.capture_to_file = MagicMock(side_effect=_capture_to_file)
    return cam


@pytest.fixture
def mock_detector():
    det = MagicMock(spec=Detector)
    det.detect = MagicMock(return_value=[])
    return det


@pytest.fixture
def mock_triage():
    return TriageAgent()


@pytest_asyncio.fixture
async def mock_db(tmp_path):
    db = Database(tmp_path / "sched_test.db")
    await db.connect()
    yield db
    await db.disconnect()

class TestCaptureScheduler:
    def test_initial_state(self, mock_camera, mock_detector, mock_triage, mock_db, mock_settings):
        sched = CaptureScheduler(
            mock_camera, mock_detector, mock_triage, mock_db, mock_settings
        )
        assert sched.is_running is False
        assert sched.current_interval_minutes == 15

    def test_daylight_check(self, mock_camera, mock_detector, mock_triage, mock_db, mock_settings):
        sched = CaptureScheduler(
            mock_camera, mock_detector, mock_triage, mock_db, mock_settings
        )
        # This is a runtime check — just verify it returns a bool
        result = sched._is_daylight()
        assert isinstance(result, bool)

    def test_adapt_interval_clean_streak(self, mock_camera, mock_detector, mock_triage, mock_db, mock_settings):
        sched = CaptureScheduler(
            mock_camera, mock_detector, mock_triage, mock_db, mock_settings
        )
        sched._consecutive_clean = 6
        sched._adapt_interval()
        assert sched._current_interval == 30

    def test_adapt_interval_recent_detection(self, mock_camera, mock_detector, mock_triage, mock_db, mock_settings):
        sched = CaptureScheduler(
            mock_camera, mock_detector, mock_triage, mock_db, mock_settings
        )
        sched._consecutive_clean = 0
        sched._adapt_interval()
        # Should keep current interval (not change)
        assert sched._current_interval == 15

    def test_adapt_interval_normal(self, mock_camera, mock_detector, mock_triage, mock_db, mock_settings):
        sched = CaptureScheduler(
            mock_camera, mock_detector, mock_triage, mock_db, mock_settings
        )
        sched._consecutive_clean = 3
        sched._current_interval = 5  # was high freq
        sched._adapt_interval()
        assert sched._current_interval == 15  # back to default

    @pytest.mark.asyncio
    async def test_start_stop(self, mock_camera, mock_detector, mock_triage, mock_db, mock_settings):
        sched = CaptureScheduler(
            mock_camera, mock_detector, mock_triage, mock_db, mock_settings
        )
        await sched.start()
        assert sched.is_running is True
        await sched.stop()
        assert sched.is_running is False

    @pytest.mark.asyncio
    async def test_capture_and_process_no_detections(self, mock_camera, mock_detector, mock_triage, mock_db, mock_settings):
        mock_detector.detect.return_value = []
        sched = CaptureScheduler(
            mock_camera, mock_detector, mock_triage, mock_db, mock_settings
        )
        result = await sched.capture_once()
        assert result == []
        assert sched._consecutive_clean == 1

    @pytest.mark.asyncio
    async def test_capture_and_process_with_detections(self, mock_camera, mock_detector, mock_triage, mock_db, mock_settings):
        det = Detection(class_name="crack", confidence=0.85, x1=10, y1=10, x2=100, y2=100)
        mock_detector.detect.return_value = [det]
        sched = CaptureScheduler(
            mock_camera, mock_detector, mock_triage, mock_db, mock_settings
        )
        # First capture — triage will pend (confirmation required=2)
        result = await sched.capture_once()
        assert result == []  # pending confirmation


class TestDefaultTimes:
    def test_sunrise_sunset(self):
        assert DEFAULT_SUNRISE == time(6, 0)
        assert DEFAULT_SUNSET == time(20, 0)
