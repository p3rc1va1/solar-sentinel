"""Settings endpoints — user-configurable preferences."""

import json
import logging

from fastapi import APIRouter, Depends

from app.api.deps import get_db, get_notifications, get_settings
from app.config import Settings
from app.db.database import Database
from app.models.settings import AllSettings, NotificationSettings
from app.services.notifications import NotificationService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/settings", tags=["settings"])


def _apply_to_runtime_settings(s: Settings, all_settings: AllSettings) -> None:
    """Copy DB-persisted values onto the live Settings instance so the
    scheduler / sensor watcher / digest pick them up on next tick.

    Only fields that the live components actually read are mirrored;
    notifications are applied via NotificationService.update_settings.
    """
    det = all_settings.detection
    loc = all_settings.location
    dig = all_settings.digest
    for field, value in (
        ("confidence_high", det.confidence_high),
        ("confidence_medium", det.confidence_medium),
        ("capture_interval_minutes", det.capture_interval_minutes),
        ("capture_interval_after_high", det.capture_interval_after_high),
        ("capture_interval_after_clean", det.capture_interval_after_clean),
        ("sensor_trigger_enabled", det.sensor_trigger_enabled),
        ("sensor_temp_high_c", det.sensor_temp_high_c),
        ("sensor_temp_low_c", det.sensor_temp_low_c),
        ("sensor_humidity_high_pct", det.sensor_humidity_high_pct),
        ("sensor_trigger_cooldown_minutes", det.sensor_trigger_cooldown_minutes),
        ("weather_latitude", loc.weather_latitude),
        ("weather_longitude", loc.weather_longitude),
        ("weather_timezone", loc.weather_timezone),
        ("digest_enabled", dig.digest_enabled),
        ("digest_time_local", dig.digest_time_local),
    ):
        try:
            setattr(s, field, value)
        except (ValueError, TypeError) as e:
            logger.warning("Couldn't apply %s=%r to runtime settings: %s", field, value, e)


@router.get("")
async def get_all_settings(db: Database = Depends(get_db)):
    """Get all user-editable settings."""
    raw = await db.get_setting("user_settings")
    if raw:
        # Tolerate older payloads that lack newer sub-models.
        data = json.loads(raw)
        return AllSettings(**data).model_dump()
    return AllSettings().model_dump()


@router.put("")
async def update_all_settings(
    settings: AllSettings,
    db: Database = Depends(get_db),
    notif_service: NotificationService = Depends(get_notifications),
    runtime_settings: Settings = Depends(get_settings),
):
    """Update all user-editable settings."""
    await db.set_setting("user_settings", settings.model_dump_json())

    # Apply notification settings at runtime
    notif_service.update_settings(**settings.notifications.model_dump())
    # Mirror to the live Settings singleton so background loops see updates
    _apply_to_runtime_settings(runtime_settings, settings)

    return {"status": "updated", "settings": settings.model_dump()}


@router.get("/notifications")
async def get_notification_settings(db: Database = Depends(get_db)):
    """Get notification preferences."""
    raw = await db.get_setting("user_settings")
    if raw:
        all_settings = AllSettings(**json.loads(raw))
        return all_settings.notifications.model_dump()
    return NotificationSettings().model_dump()


@router.put("/notifications")
async def update_notification_settings(
    settings: NotificationSettings,
    db: Database = Depends(get_db),
    notif_service: NotificationService = Depends(get_notifications),
):
    """Update notification preferences."""
    # Load existing, merge, save
    raw = await db.get_setting("user_settings")
    all_settings = AllSettings(**json.loads(raw)) if raw else AllSettings()
    all_settings.notifications = settings
    await db.set_setting("user_settings", all_settings.model_dump_json())

    notif_service.update_settings(**settings.model_dump())
    return {"status": "updated", "notifications": settings.model_dump()}
