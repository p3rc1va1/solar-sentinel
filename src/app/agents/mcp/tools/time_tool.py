"""Local time + sunrise/sunset for the panel site."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from astral import LocationInfo
from astral.sun import sun

logger = logging.getLogger(__name__)


def current_time(latitude: float, longitude: float, tz_name: str = "UTC") -> dict:
    """Return UTC + local time, daylight status, and seconds-to-sunset.

    Falls back to UTC times only if astral fails (e.g., polar regions on certain dates).
    """
    now_utc = datetime.now(timezone.utc)

    base = {
        "utc_now": now_utc.isoformat(),
        "local_now": now_utc.isoformat(),
        "timezone": tz_name,
        "sunrise": None,
        "sunset": None,
        "is_daylight": None,
        "seconds_until_sunset": None,
    }

    try:
        loc = LocationInfo(
            name="panel-site",
            region="",
            timezone=tz_name,
            latitude=float(latitude),
            longitude=float(longitude),
        )
        s = sun(loc.observer, date=now_utc.date(), tzinfo=timezone.utc)
        sunrise = s["sunrise"]
        sunset = s["sunset"]
        base.update(
            sunrise=sunrise.isoformat(),
            sunset=sunset.isoformat(),
            is_daylight=sunrise <= now_utc <= sunset,
            seconds_until_sunset=int((sunset - now_utc).total_seconds()),
        )
    except Exception as e:
        logger.warning("astral failed for (%s,%s): %s", latitude, longitude, e)

    return base
