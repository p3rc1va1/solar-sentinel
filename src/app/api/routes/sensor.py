"""Sensor endpoints — DHT22 temperature and humidity."""

import random

from fastapi import APIRouter, Depends

from app.api.deps import get_sensor, get_settings
from app.config import Settings
from app.core.sensor import DHTSensor

router = APIRouter(prefix="/sensor", tags=["sensor"])


@router.get("")
async def get_sensor_reading(
    sensor: DHTSensor = Depends(get_sensor),
    settings: Settings = Depends(get_settings),
):
    """Get current temperature and humidity from the DHT22 sensor."""
    reading = sensor.read()

    # In demo mode, return realistic fake data
    if settings.demo_mode and reading is None:
        return {
            "available": True,
            "temperature": round(random.uniform(22.0, 35.0), 1),
            "humidity": round(random.uniform(40.0, 70.0), 1),
        }

    if reading:
        return {
            "available": True,
            "temperature": reading["temperature"],
            "humidity": reading["humidity"],
        }

    return {
        "available": False,
        "temperature": None,
        "humidity": None,
    }
