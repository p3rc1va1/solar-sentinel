"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All application settings, loaded from .env file or environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Demo Mode ---
    demo_mode: bool = False

    # --- Gemini API ---
    gemini_api_key: str = ""

    # --- Notification: Email ---
    email_enabled: bool = False
    email_address: str = ""
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""

    # --- Notification: Telegram ---
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # --- Detection Thresholds ---
    confidence_high: float = Field(default=0.70, ge=0.0, le=1.0)
    confidence_medium: float = Field(default=0.45, ge=0.0, le=1.0)

    # --- Capture Schedule ---
    capture_interval_minutes: int = Field(default=15, ge=1)
    capture_interval_after_high: int = Field(default=5, ge=1)
    capture_interval_after_clean: int = Field(default=30, ge=1)

    # --- Sensor-triggered capture (DHT22) ---
    sensor_trigger_enabled: bool = True
    sensor_temp_high_c: float = Field(default=35.0)
    sensor_temp_low_c: float = Field(default=0.0)
    sensor_humidity_high_pct: float = Field(default=85.0, ge=0.0, le=100.0)
    sensor_trigger_cooldown_minutes: int = Field(default=15, ge=1)

    # --- MEDIUM detection daily digest ---
    digest_enabled: bool = True
    digest_time_local: str = "20:00"  # HH:MM, validated at use-site

    # --- YOLO Model ---
    yolo_model_path: str = "data/models/best.pt"
    yolo_input_size: int = 640

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # --- Weather ---
    weather_latitude: str = ""
    weather_longitude: str = ""
    weather_timezone: str = "UTC"

    # --- Derived paths ---
    @property
    def data_dir(self) -> Path:
        return BASE_DIR / "data"

    @property
    def detections_dir(self) -> Path:
        return self.data_dir / "detections"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    def ensure_dirs(self) -> None:
        """Create required data directories if they don't exist."""
        for d in [self.detections_dir, self.reports_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)
