"""Tests for app.core.solar — NOAA sunrise/sunset."""

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from app.core import solar

# Vilnius, Lithuania — author's reference location.
VILNIUS_LAT = 54.6872
VILNIUS_LON = 25.2797
VILNIUS_TZ = ZoneInfo("Europe/Vilnius")


def _to_minutes(t: time) -> int:
    return t.hour * 60 + t.minute


class TestSunTimes:
    def test_vilnius_summer_solstice(self):
        """2026-06-21 in Vilnius — sunrise ~04:43, sunset ~21:59 local (DST)."""
        rise, set_ = solar.sun_times(
            VILNIUS_LAT, VILNIUS_LON, date(2026, 6, 21), VILNIUS_TZ
        )
        # Allow ±5 min tolerance for the simplified NOAA formula.
        assert abs(_to_minutes(rise) - _to_minutes(time(4, 43))) <= 5
        assert abs(_to_minutes(set_) - _to_minutes(time(21, 59))) <= 5

    def test_vilnius_winter_solstice(self):
        """2026-12-21 in Vilnius — sunrise ~08:39, sunset ~15:53 local."""
        rise, set_ = solar.sun_times(
            VILNIUS_LAT, VILNIUS_LON, date(2026, 12, 21), VILNIUS_TZ
        )
        assert abs(_to_minutes(rise) - _to_minutes(time(8, 39))) <= 5
        assert abs(_to_minutes(set_) - _to_minutes(time(15, 53))) <= 5

    def test_polar_day(self):
        """80°N on summer solstice — sun never sets."""
        rise, set_ = solar.sun_times(
            80.0, 0.0, date(2026, 6, 21), ZoneInfo("UTC")
        )
        assert rise == time(0, 0)
        assert set_ == time(23, 59)

    def test_polar_night(self):
        """80°N on winter solstice — sun never rises."""
        rise, set_ = solar.sun_times(
            80.0, 0.0, date(2026, 12, 21), ZoneInfo("UTC")
        )
        assert rise == set_  # sentinel pair

    def test_cache_hit(self):
        """Second call with the same args returns the same tuple instance."""
        a = solar.sun_times(VILNIUS_LAT, VILNIUS_LON, date(2026, 6, 21), VILNIUS_TZ)
        b = solar.sun_times(VILNIUS_LAT, VILNIUS_LON, date(2026, 6, 21), VILNIUS_TZ)
        assert a is b


class TestIsDaylight:
    def test_midday_is_daylight(self):
        noon = datetime(2026, 6, 21, 12, 0, tzinfo=VILNIUS_TZ)
        assert solar.is_daylight(VILNIUS_LAT, VILNIUS_LON, noon, VILNIUS_TZ) is True

    def test_midnight_is_dark(self):
        midnight = datetime(2026, 12, 21, 0, 0, tzinfo=VILNIUS_TZ)
        assert solar.is_daylight(VILNIUS_LAT, VILNIUS_LON, midnight, VILNIUS_TZ) is False

    def test_polar_night_returns_false(self):
        when = datetime(2026, 12, 21, 12, 0, tzinfo=ZoneInfo("UTC"))
        assert solar.is_daylight(80.0, 0.0, when, ZoneInfo("UTC")) is False

    def test_polar_day_returns_true(self):
        when = datetime(2026, 6, 21, 23, 30, tzinfo=ZoneInfo("UTC"))
        assert solar.is_daylight(80.0, 0.0, when, ZoneInfo("UTC")) is True

    def test_naive_datetime_assumed_in_tz(self):
        """Naive datetimes are assumed to already be in the requested tz."""
        noon = datetime(2026, 6, 21, 12, 0)  # naive
        assert solar.is_daylight(VILNIUS_LAT, VILNIUS_LON, noon, VILNIUS_TZ) is True
