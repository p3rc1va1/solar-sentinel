"""Shared test fixtures for Solar Sentinel."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from app.config import Settings
from app.core.camera import Camera
from app.core.detector import Detector
from app.core.triage import TriageAgent
from app.db.database import Database
from app.services.gemini import GeminiClient
from app.services.notifications import NotificationService
from app.services.weather import WeatherService


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory structure."""
    for d in ["detections", "reports", "models"]:
        (tmp_path / d).mkdir()
    return tmp_path


@pytest_asyncio.fixture
async def db(tmp_path):
    """In-memory-like SQLite database for tests."""
    db_path = tmp_path / "test.db"
    database = Database(db_path)
    await database.connect()
    yield database
    await database.disconnect()


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Test settings with temp paths."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("EMAIL_ENABLED", "false")
    monkeypatch.setenv("TELEGRAM_ENABLED", "false")
    s = Settings()
    return s


@pytest.fixture
def camera():
    """Camera in stub mode (no picamera2)."""
    return Camera(resolution=(64, 64))


@pytest.fixture
def detector():
    """Detector with no model loaded (stub mode)."""
    return Detector(model_path="/nonexistent/model.pt", input_size=64)


@pytest.fixture
def triage():
    return TriageAgent(
        dedup_window_minutes=60,
        iou_threshold=0.5,
        confirmation_required=2,
    )


@pytest.fixture
def gemini_client():
    return GeminiClient(api_key="")


@pytest.fixture
def notification_service():
    return NotificationService()


@pytest.fixture
def weather_service():
    return WeatherService()
