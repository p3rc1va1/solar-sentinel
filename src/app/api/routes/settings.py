"""Settings endpoints — user-configurable preferences."""

import json

from fastapi import APIRouter, Depends

from app.api.deps import get_db, get_notifications, get_settings
from app.db.database import Database
from app.models.settings import AllSettings, NotificationSettings, DetectionSettings, GeminiSettings
from app.services.notifications import NotificationService

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("")
async def get_all_settings(db: Database = Depends(get_db)):
    """Get all user-editable settings."""
    raw = await db.get_setting("user_settings")
    if raw:
        return json.loads(raw)
    # Return defaults
    return AllSettings().model_dump()


@router.put("")
async def update_all_settings(
    settings: AllSettings,
    db: Database = Depends(get_db),
    notif_service: NotificationService = Depends(get_notifications),
):
    """Update all user-editable settings."""
    await db.set_setting("user_settings", settings.model_dump_json())

    # Apply notification settings at runtime
    notif_service.update_settings(**settings.notifications.model_dump())

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
