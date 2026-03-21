"""Tests for app.services.weather."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.weather import WeatherService, _WMO_CODES


class TestWeatherService:
    @pytest.mark.asyncio
    async def test_no_coordinates(self):
        svc = WeatherService()
        result = await svc.get_current_weather()
        assert result["summary"] == "Not available"

    @pytest.mark.asyncio
    async def test_start_stop(self):
        svc = WeatherService()
        await svc.start()
        assert svc._client is not None
        await svc.stop()
        assert svc._client is None

    @pytest.mark.asyncio
    @patch("app.services.weather.httpx.AsyncClient")
    async def test_fetch_success(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "current": {
                "weather_code": 0,
                "temperature_2m": 25.5,
                "precipitation": 0,
                "uv_index": 6,
            }
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client

        svc = WeatherService(latitude="54.68", longitude="25.28")
        result = await svc.get_current_weather()
        assert result["summary"] == "Clear sky"
        assert "25.5" in result["temperature"]

    @pytest.mark.asyncio
    @patch("app.services.weather.httpx.AsyncClient")
    async def test_fetch_error(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("timeout"))
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client

        svc = WeatherService(latitude="54.68", longitude="25.28")
        result = await svc.get_current_weather()
        assert result["summary"] == "Unavailable"


class TestWMOCodes:
    def test_known_codes(self):
        assert _WMO_CODES[0] == "Clear sky"
        assert _WMO_CODES[95] == "Thunderstorm"

    def test_code_count(self):
        assert len(_WMO_CODES) > 10
