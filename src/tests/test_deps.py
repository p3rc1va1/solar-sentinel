"""Tests for app.api.deps."""

import pytest

from app.api import deps
from app.config import Settings
from app.core.camera import Camera
from app.core.detector import Detector
from app.core.triage import TriageAgent
from app.db.database import Database
from app.services.gemini import GeminiClient
from app.services.notifications import NotificationService
from app.services.weather import WeatherService


class TestDeps:
    def setup_method(self):
        """Reset all singletons before each test."""
        deps._db = None
        deps._settings = None
        deps._camera = None
        deps._detector = None
        deps._triage = None
        deps._gemini = None
        deps._notifications = None
        deps._weather = None

    def test_get_before_init_raises(self):
        with pytest.raises(RuntimeError, match="Database not initialized"):
            deps.get_db()

        with pytest.raises(RuntimeError, match="Settings not initialized"):
            deps.get_settings()

        with pytest.raises(RuntimeError, match="Camera not initialized"):
            deps.get_camera()

        with pytest.raises(RuntimeError, match="Detector not initialized"):
            deps.get_detector()

    def test_init_and_get(self, tmp_path):
        db = Database(tmp_path / "test.db")
        settings = Settings()
        camera = Camera()
        detector = Detector("/nonexistent.pt")
        triage = TriageAgent()
        gemini = GeminiClient(api_key="")
        notif = NotificationService()
        weather = WeatherService()

        deps.init_deps(db, settings, camera, detector, triage, gemini, notif, weather)

        assert deps.get_db() is db
        assert deps.get_settings() is settings
        assert deps.get_camera() is camera
        assert deps.get_detector() is detector
        assert deps.get_triage() is triage
        assert deps.get_gemini() is gemini
        assert deps.get_notifications() is notif
        assert deps.get_weather() is weather
