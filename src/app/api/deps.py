"""FastAPI dependency injection."""

from app.config import Settings
from app.core.camera import Camera
from app.core.detector import Detector
from app.core.triage import TriageAgent
from app.db.database import Database
from app.services.gemini import GeminiClient
from app.services.notifications import NotificationService
from app.services.weather import WeatherService

# Singletons — initialized in main.py lifespan
_db: Database | None = None
_settings: Settings | None = None
_camera: Camera | None = None
_detector: Detector | None = None
_triage: TriageAgent | None = None
_gemini: GeminiClient | None = None
_notifications: NotificationService | None = None
_weather: WeatherService | None = None


def init_deps(
    db: Database,
    settings: Settings,
    camera: Camera,
    detector: Detector,
    triage: TriageAgent,
    gemini: GeminiClient,
    notifications: NotificationService,
    weather: WeatherService,
) -> None:
    """Register singleton instances (called from lifespan)."""
    global _db, _settings, _camera, _detector, _triage, _gemini, _notifications, _weather
    _db = db
    _settings = settings
    _camera = camera
    _detector = detector
    _triage = triage
    _gemini = gemini
    _notifications = notifications
    _weather = weather


def _get(name: str, value):
    if value is None:
        raise RuntimeError(f"{name} not initialized. Call init_deps() first.")
    return value


def get_db() -> Database:
    return _get("Database", _db)


def get_settings() -> Settings:
    return _get("Settings", _settings)


def get_camera() -> Camera:
    return _get("Camera", _camera)


def get_detector() -> Detector:
    return _get("Detector", _detector)


def get_triage() -> TriageAgent:
    return _get("Triage", _triage)


def get_gemini() -> GeminiClient:
    return _get("GeminiClient", _gemini)


def get_notifications() -> NotificationService:
    return _get("Notifications", _notifications)


def get_weather() -> WeatherService:
    return _get("WeatherService", _weather)
