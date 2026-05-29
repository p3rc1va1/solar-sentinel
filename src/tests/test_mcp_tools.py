"""Tests for app.agents.mcp.tools and the FastMCP server registration."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.mcp.tools import forecast as forecast_mod
from app.agents.mcp.tools import time_tool as time_mod
from app.agents.mcp.tools import web_search as web_mod


# ── web_search ────────────────────────────────────────────────────────────

_DDG_HTML = """
<html><body>
<div class="result">
  <a class="result__a" href="https://example.com/a">First Result</a>
  <a class="result__snippet">Snippet text for A.</a>
</div>
<div class="result">
  <a class="result__a" href="https://example.com/b">Second Result</a>
  <a class="result__snippet">Snippet text for B.</a>
</div>
<div class="result">
  <a class="result__a" href="https://example.com/c">Third Result</a>
  <a class="result__snippet">Snippet text for C.</a>
</div>
</body></html>
"""


class TestWebSearchParse:
    def test_parse_extracts_results(self):
        results = web_mod.parse_results(_DDG_HTML, max_results=10)
        assert len(results) == 3
        assert results[0]["title"] == "First Result"
        assert results[0]["url"] == "https://example.com/a"
        assert "Snippet" in results[0]["snippet"]

    def test_parse_respects_max(self):
        results = web_mod.parse_results(_DDG_HTML, max_results=2)
        assert len(results) == 2

    def test_parse_empty_html(self):
        assert web_mod.parse_results("<html></html>", max_results=5) == []


class TestWebSearchAsync:
    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self):
        assert await web_mod.web_search("   ") == []

    @pytest.mark.asyncio
    @patch("app.agents.mcp.tools.web_search.httpx.AsyncClient")
    async def test_success(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.text = _DDG_HTML
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        results = await web_mod.web_search("solar panel hot spot", max_results=2)
        assert len(results) == 2
        assert results[0]["title"] == "First Result"

    @pytest.mark.asyncio
    @patch("app.agents.mcp.tools.web_search.httpx.AsyncClient")
    async def test_network_error_returns_empty(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("boom"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        assert await web_mod.web_search("anything") == []


# ── current_time ──────────────────────────────────────────────────────────


class TestCurrentTime:
    def test_known_location_has_sun_data(self):
        # Vilnius — temperate latitude, always has a sunrise/sunset.
        result = time_mod.current_time(54.6872, 25.2797, "Europe/Vilnius")
        assert result["timezone"] == "Europe/Vilnius"
        assert result["sunrise"] is not None
        assert result["sunset"] is not None
        assert isinstance(result["is_daylight"], bool)
        assert isinstance(result["seconds_until_sunset"], int)

    def test_invalid_location_falls_back_gracefully(self):
        # Astral handles extreme latitudes by raising on certain dates.
        # The fallback path should still return UTC times without crashing.
        result = time_mod.current_time(91.0, 0.0, "UTC")
        assert "utc_now" in result
        assert "local_now" in result


# ── weather_forecast ──────────────────────────────────────────────────────

_FAKE_OPENMETEO = {
    "hourly": {
        "time": [
            "2026-05-28T10:00",
            "2026-05-28T11:00",
            "2026-05-28T12:00",
        ],
        "temperature_2m": [18.0, 19.5, 21.0],
        "precipitation_probability": [10, 50, 80],
        "cloud_cover": [30, 60, 90],
        "weather_code": [1, 3, 61],
    }
}


class TestWeatherForecast:
    @pytest.mark.asyncio
    @patch("app.agents.mcp.tools.forecast.httpx.AsyncClient")
    async def test_success(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _FAKE_OPENMETEO
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await forecast_mod.weather_forecast(54.68, 25.28, hours=3)
        assert len(result["hours"]) == 3
        assert result["hours"][0]["temperature_c"] == 18.0
        assert result["hours"][0]["weather"] == "Mainly clear"
        assert result["hours"][2]["weather"] == "Slight rain"
        assert "peak rain prob 80" in result["summary"]

    @pytest.mark.asyncio
    @patch("app.agents.mcp.tools.forecast.httpx.AsyncClient")
    async def test_network_error(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("boom"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await forecast_mod.weather_forecast(0.0, 0.0)
        assert result["summary"] == "Forecast unavailable"
        assert result["hours"] == []

    @pytest.mark.asyncio
    @patch("app.agents.mcp.tools.forecast.httpx.AsyncClient")
    async def test_hours_clamped(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"hourly": {}}
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        # 999 hours requested -> clamped to 168, no crash.
        result = await forecast_mod.weather_forecast(0.0, 0.0, hours=999)
        assert "hours" in result


# ── server registration ───────────────────────────────────────────────────


class TestMcpServerRegistration:
    def test_three_tools_registered(self):
        from app.agents.mcp.server import mcp

        tools = asyncio.run(mcp.list_tools())
        names = {t.name for t in tools}
        assert names == {"web_search", "current_time", "weather_forecast"}
