"""Tests for API routes via TestClient."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.api import deps
from app.config import Settings
from app.core.camera import Camera
from app.core.detector import Detector
from app.core.triage import TriageAgent
from app.db.database import Database
from app.main import app
from app.services.gemini import GeminiClient
from app.services.notifications import NotificationService
from app.services.weather import WeatherService


@pytest_asyncio.fixture
async def setup_deps(tmp_path):
    """Initialize real deps for integration testing."""
    db = Database(tmp_path / "test.db")
    await db.connect()

    settings = Settings()
    camera = Camera(resolution=(64, 64))
    detector = Detector("/nonexistent.pt")
    triage = TriageAgent()
    gemini = GeminiClient(api_key="")
    notif = NotificationService()
    weather = WeatherService()

    deps.init_deps(db, settings, camera, detector, triage, gemini, notif, weather)

    yield db, settings
    await db.disconnect()


@pytest_asyncio.fixture
async def client(setup_deps):
    """Async test client (bypasses lifespan)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "system" in data


class TestDetectionsEndpoint:
    @pytest.mark.asyncio
    async def test_list_empty(self, client):
        resp = await client.get("/detections")
        assert resp.status_code == 200
        data = resp.json()
        assert data["detections"] == []
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, client):
        resp = await client.get("/detections/999")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_insert_then_list(self, client, setup_deps):
        db, _ = setup_deps
        await db.insert_detection(
            image_path="/tmp/test.jpg",
            defect_class="crack",
            confidence=0.9,
            bbox={"x1": 0, "y1": 0, "x2": 100, "y2": 100},
        )
        resp = await client.get("/detections")
        assert resp.json()["count"] == 1


class TestReportsEndpoint:
    @pytest.mark.asyncio
    async def test_list_empty(self, client):
        resp = await client.get("/reports")
        assert resp.status_code == 200
        assert resp.json()["reports"] == []

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, client):
        resp = await client.get("/reports/999")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_context_history(self, client):
        resp = await client.get("/reports/context/history")
        assert resp.status_code == 200
        assert "context" in resp.json()


class TestSettingsEndpoint:
    @pytest.mark.asyncio
    async def test_get_defaults(self, client):
        resp = await client.get("/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert "notifications" in data

    @pytest.mark.asyncio
    async def test_update_settings(self, client):
        payload = {
            "notifications": {
                "email_enabled": True,
                "email_address": "test@test.com",
                "smtp_host": "smtp.test.com",
                "smtp_port": 587,
                "smtp_username": "user",
                "smtp_password": "pass",
                "telegram_enabled": False,
                "telegram_bot_token": "",
                "telegram_chat_id": "",
            },
            "detection": {
                "confidence_high": 0.80,
                "confidence_medium": 0.50,
                "capture_interval_minutes": 10,
                "capture_interval_after_high": 3,
                "capture_interval_after_clean": 20,
            },
            "gemini": {"gemini_api_key": "test-key"},
        }
        resp = await client.put("/settings", json=payload)
        assert resp.status_code == 200
        assert resp.json()["status"] == "updated"

    @pytest.mark.asyncio
    async def test_get_settings_after_update(self, client):
        """Settings should persist across requests."""
        payload = {
            "notifications": {
                "email_enabled": True,
                "email_address": "saved@test.com",
                "smtp_host": "smtp.test.com",
                "smtp_port": 587,
                "smtp_username": "u",
                "smtp_password": "p",
                "telegram_enabled": False,
                "telegram_bot_token": "",
                "telegram_chat_id": "",
            },
            "detection": {
                "confidence_high": 0.80,
                "confidence_medium": 0.50,
                "capture_interval_minutes": 10,
                "capture_interval_after_high": 3,
                "capture_interval_after_clean": 20,
            },
            "gemini": {"gemini_api_key": ""},
        }
        await client.put("/settings", json=payload)
        resp = await client.get("/settings")
        data = resp.json()
        assert data["notifications"]["email_enabled"] is True

    @pytest.mark.asyncio
    async def test_get_notification_settings(self, client):
        resp = await client.get("/settings/notifications")
        assert resp.status_code == 200
        data = resp.json()
        assert "email_enabled" in data

    @pytest.mark.asyncio
    async def test_update_notification_settings(self, client):
        payload = {
            "email_enabled": True,
            "email_address": "notif@test.com",
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_username": "user",
            "smtp_password": "pass",
            "telegram_enabled": False,
            "telegram_bot_token": "",
            "telegram_chat_id": "",
        }
        resp = await client.put("/settings/notifications", json=payload)
        assert resp.status_code == 200
        assert resp.json()["status"] == "updated"

    @pytest.mark.asyncio
    async def test_get_notification_settings_after_update(self, client):
        """Ensure notification sub-endpoint reads merged settings."""
        # First save all settings
        payload = {
            "notifications": {
                "email_enabled": False,
                "email_address": "",
                "smtp_host": "smtp.gmail.com",
                "smtp_port": 587,
                "smtp_username": "",
                "smtp_password": "",
                "telegram_enabled": True,
                "telegram_bot_token": "token123",
                "telegram_chat_id": "chat456",
            },
            "detection": {
                "confidence_high": 0.70,
                "confidence_medium": 0.45,
                "capture_interval_minutes": 15,
                "capture_interval_after_high": 5,
                "capture_interval_after_clean": 30,
            },
            "gemini": {"gemini_api_key": ""},
        }
        await client.put("/settings", json=payload)
        resp = await client.get("/settings/notifications")
        assert resp.json()["telegram_enabled"] is True


class TestCameraEndpoint:
    @pytest.mark.asyncio
    async def test_capture(self, client):
        resp = await client.post("/camera/capture")
        assert resp.status_code == 200
        assert resp.json()["status"] == "capture_triggered"


class TestDetectionDetail:
    @pytest.mark.asyncio
    async def test_get_existing(self, client, setup_deps):
        db, _ = setup_deps
        det_id = await db.insert_detection(
            image_path="/tmp/x.jpg", defect_class="soiling",
            confidence=0.9, bbox={"x1": 0, "y1": 0, "x2": 1, "y2": 1},
        )
        resp = await client.get(f"/detections/{det_id}")
        assert resp.status_code == 200
        assert resp.json()["defect_class"] == "soiling"


class TestReportDetail:
    @pytest.mark.asyncio
    async def test_get_existing(self, client, setup_deps):
        db, _ = setup_deps
        det_id = await db.insert_detection(
            image_path="/tmp/x.jpg", defect_class="crack",
            confidence=0.9, bbox={"x1": 0, "y1": 0, "x2": 1, "y2": 1},
        )
        report_id = await db.insert_report(
            detection_id=det_id, severity="WARNING", urgency="ROUTINE",
            root_cause="Test", trend_analysis="N/A",
            report_markdown="# R", qa_score=7, qa_approved=True,
        )
        resp = await client.get(f"/reports/{report_id}")
        assert resp.status_code == 200
        assert resp.json()["severity"] == "WARNING"
