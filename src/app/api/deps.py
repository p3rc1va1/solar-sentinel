"""FastAPI dependency injection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.config import Settings
from app.core.camera import Camera
from app.core.detector import Detector
from app.core.triage import TriageAgent
from app.db.database import Database
from app.services.gemini import GeminiClient
from app.services.geocoding import GeocodingService
from app.services.notifications import NotificationService
from app.services.weather import WeatherService

if TYPE_CHECKING:
    from app.core.digest import DigestScheduler
    from app.core.scheduler import CaptureScheduler
    from app.core.sensor import DHTSensor
    from app.core.sensor_watcher import SensorWatcher

# Singletons — initialized in main.py lifespan
_db: Database | None = None
_settings: Settings | None = None
_camera: Camera | None = None
_detector: Detector | None = None
_triage: TriageAgent | None = None
_gemini: GeminiClient | None = None
_notifications: NotificationService | None = None
_weather: WeatherService | None = None
_geocoding: GeocodingService | None = None
_scheduler: CaptureScheduler | None = None
_sensor: DHTSensor | None = None
_sensor_watcher: SensorWatcher | None = None
_digest_scheduler: DigestScheduler | None = None


def init_deps(
    db: Database,
    settings: Settings,
    camera: Camera,
    detector: Detector,
    triage: TriageAgent,
    gemini: GeminiClient,
    notifications: NotificationService,
    weather: WeatherService,
    geocoding: GeocodingService,
    scheduler: CaptureScheduler | None = None,
    sensor: DHTSensor | None = None,
    sensor_watcher: SensorWatcher | None = None,
    digest_scheduler: DigestScheduler | None = None,
) -> None:
    """Register singleton instances (called from lifespan)."""
    global _db, _settings, _camera, _detector, _triage, _gemini, _notifications, _weather
    global _geocoding, _scheduler, _sensor, _sensor_watcher, _digest_scheduler
    _db = db
    _settings = settings
    _camera = camera
    _detector = detector
    _triage = triage
    _gemini = gemini
    _notifications = notifications
    _weather = weather
    _geocoding = geocoding
    _scheduler = scheduler
    _sensor = sensor
    _sensor_watcher = sensor_watcher
    _digest_scheduler = digest_scheduler


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


def get_scheduler() -> CaptureScheduler:
    return _get("Scheduler", _scheduler)


def get_sensor() -> DHTSensor:
    return _get("Sensor", _sensor)


def get_geocoding() -> GeocodingService:
    return _get("GeocodingService", _geocoding)


def get_sensor_watcher() -> "SensorWatcher":
    return _get("SensorWatcher", _sensor_watcher)


def get_digest_scheduler() -> "DigestScheduler":
    return _get("DigestScheduler", _digest_scheduler)
