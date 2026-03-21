"""Pydantic models for user-editable settings."""

from pydantic import BaseModel


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


class GeminiSettings(BaseModel):
    gemini_api_key: str = ""


class AllSettings(BaseModel):
    notifications: NotificationSettings = NotificationSettings()
    detection: DetectionSettings = DetectionSettings()
    gemini: GeminiSettings = GeminiSettings()
