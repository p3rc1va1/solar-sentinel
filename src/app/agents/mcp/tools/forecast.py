"""Hourly weather forecast via Open-Meteo (no key required)."""

from __future__ import annotations

import logging

import httpx

from app.services.weather import _WMO_CODES

logger = logging.getLogger(__name__)

_ENDPOINT = "https://api.open-meteo.com/v1/forecast"


async def weather_forecast(
    latitude: float,
    longitude: float,
    hours: int = 24,
) -> dict:
    """Fetch hourly forecast for the next `hours` (capped at 168).

    Returns:
        {
            "summary": str,                 # human-readable headline
            "hours": [
                {
                    "time": str,
                    "temperature_c": float,
                    "precipitation_probability": float,
                    "cloud_cover": float,
                    "weather": str,
                },
                ...
            ],
        }
    """
    hours = max(1, min(hours, 168))

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                _ENDPOINT,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "hourly": (
                        "temperature_2m,precipitation_probability,"
                        "cloud_cover,weather_code"
                    ),
                    "forecast_days": min(7, (hours + 23) // 24),
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning("weather_forecast failed: %s", e)
        return {"summary": "Forecast unavailable", "hours": []}

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])[:hours]
    temps = hourly.get("temperature_2m", [])[:hours]
    precip = hourly.get("precipitation_probability", [])[:hours]
    cloud = hourly.get("cloud_cover", [])[:hours]
    codes = hourly.get("weather_code", [])[:hours]

    rows: list[dict] = []
    for i, t in enumerate(times):
        rows.append(
            {
                "time": t,
                "temperature_c": _safe(temps, i),
                "precipitation_probability": _safe(precip, i),
                "cloud_cover": _safe(cloud, i),
                "weather": _WMO_CODES.get(_safe_int(codes, i), "Unknown"),
            }
        )

    if rows:
        max_precip = max((r["precipitation_probability"] or 0) for r in rows)
        peak = max(rows, key=lambda r: r["precipitation_probability"] or 0)
        summary = (
            f"{len(rows)}h ahead: peak rain prob {max_precip:.0f}% at {peak['time']} "
            f"({peak['weather']}); first hour {rows[0]['weather']}, "
            f"{rows[0]['temperature_c']}°C."
        )
    else:
        summary = "No forecast data returned"

    return {"summary": summary, "hours": rows}


def _safe(arr: list, i: int):
    return arr[i] if i < len(arr) else None


def _safe_int(arr: list, i: int) -> int:
    v = _safe(arr, i)
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0
