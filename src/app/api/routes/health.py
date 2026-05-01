"""Health and system status endpoint."""

import psutil
import random
from datetime import datetime, timezone
from fastapi import APIRouter, Depends

from app.api.deps import get_db, get_settings
from app.db.database import Database
from app.config import Settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """System health overview."""
    # CPU temperature (Linux / RPi)
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            cpu_temp = int(f.read().strip()) / 1000
    except (FileNotFoundError, ValueError):
        cpu_temp = None

    if settings.demo_mode:
        cpu_temp = round(random.uniform(42.0, 46.5), 1)

    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    gemini_usage = await db.get_gemini_usage_today()

    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": {
            "cpu_temp_c": cpu_temp,
            "cpu_percent": psutil.cpu_percent(interval=0),
            "memory_used_percent": mem.percent,
            "memory_available_mb": round(mem.available / 1024 / 1024),
            "disk_used_percent": disk.percent,
            "disk_free_gb": round(disk.free / 1024 / 1024 / 1024, 1),
        },
        "gemini_usage_today": gemini_usage,
    }
