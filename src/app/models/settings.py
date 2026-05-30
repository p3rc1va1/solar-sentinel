"""Pydantic models for user-editable settings."""

from pydantic import BaseModel, Field


class NotificationSettings(BaseModel):
    email_enabled: bool = False
    email_address: str = ""
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""


class DetectionSettings(BaseModel):
    confidence_high: float = 0.70
    confidence_medium: float = 0.45
    capture_interval_minutes: int = 15
    capture_interval_after_high: int = 5
    capture_interval_after_clean: int = 30
    # Sensor-driven trigger
    sensor_trigger_enabled: bool = True
    sensor_temp_high_c: float = 35.0
    sensor_temp_low_c: float = 0.0
    sensor_humidity_high_pct: float = Field(default=85.0, ge=0.0, le=100.0)
    sensor_trigger_cooldown_minutes: int = Field(default=15, ge=1)


class GeminiSettings(BaseModel):
    gemini_api_key: str = ""


class LocationSettings(BaseModel):
    weather_latitude: str = ""
    weather_longitude: str = ""
    weather_timezone: str = "UTC"
    location_label: str = ""  # cosmetic, e.g. "Vilnius, Lithuania"


class DigestSettings(BaseModel):
    digest_enabled: bool = True
    digest_time_local: str = "20:00"


class AllSettings(BaseModel):
    notifications: NotificationSettings = NotificationSettings()
    detection: DetectionSettings = DetectionSettings()
    gemini: GeminiSettings = GeminiSettings()
    location: LocationSettings = LocationSettings()
    digest: DigestSettings = DigestSettings()
