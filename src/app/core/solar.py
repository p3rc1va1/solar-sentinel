"""NOAA solar position formula — local sunrise/sunset for the panel site.

No external dependencies. Accurate to roughly 1 minute for civil
sunrise/sunset (zenith 90°50') at non-polar latitudes. Polar day
and polar night fall back to deterministic sentinel values so the
scheduler doesn't deadlock at extreme latitudes.

Reference: NOAA's Earth System Research Lab solar position calculator,
https://gml.noaa.gov/grad/solcalc/solareqns.PDF
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Civil sunrise/sunset zenith — accounts for atmospheric refraction
# and the sun's apparent radius (NOAA standard).
_ZENITH_DEG = 90.0 + 50.0 / 60.0

# Sentinels for polar day / polar night (UI-friendly, scheduler-safe).
_POLAR_DAY = (time(0, 0), time(23, 59))
_POLAR_NIGHT = (time(12, 0), time(12, 0))


def _julian_day(d: date) -> float:
    """Julian day number at 00:00 UT for the given calendar date."""
    y, m, day = d.year, d.month, d.day
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + day + b - 1524.5


def _sun_event_utc_hours(d: date, lat: float, lon: float, rising: bool) -> float | None:
    """Return UTC hour of sunrise (rising=True) or sunset (rising=False).

    Returns None if the sun does not cross the civil horizon on this
    date at this latitude (polar day or polar night).

    Variable names follow standard astronomical notation (M, L, RA, H);
    `# noqa: N806` keeps ruff quiet without flattening the convention.
    """
    # Day of the year, 1..366
    n = (d - date(d.year, 1, 1)).days + 1

    # Approximate time of event (in days since J2000.0)
    lon_hour = lon / 15.0
    t = n + ((6.0 if rising else 18.0) - lon_hour) / 24.0

    # Sun's mean anomaly
    M = 0.9856 * t - 3.289  # noqa: N806
    # True longitude
    L = (  # noqa: N806
        M
        + 1.916 * math.sin(math.radians(M))
        + 0.020 * math.sin(math.radians(2 * M))
        + 282.634
    ) % 360.0
    # Right ascension, adjusted to L's quadrant
    RA = math.degrees(math.atan(0.91764 * math.tan(math.radians(L)))) % 360.0  # noqa: N806
    L_quadrant = math.floor(L / 90.0) * 90.0  # noqa: N806
    RA_quadrant = math.floor(RA / 90.0) * 90.0  # noqa: N806
    RA = (RA + (L_quadrant - RA_quadrant)) / 15.0  # noqa: N806  convert to hours

    # Declination
    sin_dec = 0.39782 * math.sin(math.radians(L))
    cos_dec = math.cos(math.asin(sin_dec))

    # Local hour angle
    cos_h = (math.cos(math.radians(_ZENITH_DEG)) - sin_dec * math.sin(math.radians(lat))) / (
        cos_dec * math.cos(math.radians(lat))
    )
    if cos_h > 1.0 or cos_h < -1.0:
        return None  # polar night (>1) or polar day (<-1)

    H = 360.0 - math.degrees(math.acos(cos_h)) if rising else math.degrees(math.acos(cos_h))  # noqa: N806
    H /= 15.0  # noqa: N806  to hours

    # Local mean time of event
    T = H + RA - 0.06571 * t - 6.622  # noqa: N806
    # To UTC
    UT = (T - lon_hour) % 24.0  # noqa: N806
    return UT


@lru_cache(maxsize=8)
def _cached_sun_times(lat: float, lon: float, iso_date: str, tz_key: str) -> tuple[time, time]:
    """Cached implementation of sun_times keyed on hashable args."""
    d = date.fromisoformat(iso_date)
    tz = ZoneInfo(tz_key)

    try:
        rise_utc = _sun_event_utc_hours(d, lat, lon, rising=True)
        set_utc = _sun_event_utc_hours(d, lat, lon, rising=False)
    except (ValueError, ZeroDivisionError) as e:
        logger.warning("Solar calc failed (lat=%s lon=%s on %s): %s — assuming daylight", lat, lon, iso_date, e)
        return _POLAR_DAY

    if rise_utc is None and set_utc is None:
        # Use solar noon to disambiguate: if sun is above horizon at noon → polar day.
        noon = datetime.combine(d, time(12, 0)).replace(tzinfo=ZoneInfo("UTC"))
        if _sun_above_horizon(noon, lat, lon):
            logger.info("Polar day at lat=%s lon=%s on %s", lat, lon, iso_date)
            return _POLAR_DAY
        logger.info("Polar night at lat=%s lon=%s on %s", lat, lon, iso_date)
        return _POLAR_NIGHT

    # Convert UTC hours to local time-of-day on the requested date.
    rise_local = _utc_hours_to_local_time(d, rise_utc or 0.0, tz)
    set_local = _utc_hours_to_local_time(d, set_utc or 23.99, tz)
    return rise_local, set_local


def _utc_hours_to_local_time(d: date, ut_hours: float, tz: ZoneInfo) -> time:
    """Convert a UTC hour-of-day on a given UTC date to a local clock time."""
    h = int(ut_hours)
    m = int((ut_hours - h) * 60)
    s = int((((ut_hours - h) * 60) - m) * 60)
    # Clamp seconds to avoid 60 from rounding noise
    if s >= 60:
        s = 59
    utc_dt = datetime.combine(d, time(h % 24, m, s), tzinfo=ZoneInfo("UTC"))
    if h >= 24:
        utc_dt += timedelta(days=1)
    elif h < 0:
        utc_dt -= timedelta(days=1)
    local_dt = utc_dt.astimezone(tz)
    return local_dt.time().replace(microsecond=0)


def _sun_above_horizon(when_utc: datetime, lat: float, lon: float) -> bool:
    """Quick check — is the sun above the civil horizon at the given UTC instant?"""
    # Day fraction since J2000.0
    n = (when_utc.toordinal() - date(2000, 1, 1).toordinal()) + (
        when_utc.hour + when_utc.minute / 60.0
    ) / 24.0 - 0.5
    L = (280.460 + 0.9856474 * n) % 360.0
    g = math.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = math.radians(L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    eps = math.radians(23.439 - 0.0000004 * n)
    dec = math.asin(math.sin(eps) * math.sin(lam))
    # Greenwich Mean Sidereal Time, in hours
    gmst = (18.697374558 + 24.06570982441908 * n) % 24.0
    lst = (gmst + lon / 15.0) * 15.0  # local sidereal time, degrees
    ra = math.degrees(math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam)))
    H = math.radians((lst - ra + 540.0) % 360.0 - 180.0)
    alt = math.asin(
        math.sin(math.radians(lat)) * math.sin(dec)
        + math.cos(math.radians(lat)) * math.cos(dec) * math.cos(H)
    )
    return math.degrees(alt) > -50.0 / 60.0  # above civil horizon


def sun_times(lat: float, lon: float, on_date: date, tz: ZoneInfo) -> tuple[time, time]:
    """Return local civil sunrise and sunset for the given site and date.

    Polar day → (00:00, 23:59); polar night → (12:00, 12:00).
    """
    return _cached_sun_times(round(lat, 4), round(lon, 4), on_date.isoformat(), tz.key)


def is_daylight(lat: float, lon: float, now: datetime, tz: ZoneInfo) -> bool:
    """Is `now` (timezone-aware) within the local daylight window?"""
    if now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    local_now = now.astimezone(tz)
    rise, set_ = sun_times(lat, lon, local_now.date(), tz)
    if rise == set_:  # polar night sentinel
        return False
    if (rise, set_) == _POLAR_DAY:
        return True
    return rise <= local_now.time() <= set_
