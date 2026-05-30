"""Tests for app.services.geocoding."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.geocoding import GeocodingService


class TestGeocodingService:
    @pytest.mark.asyncio
    async def test_short_query_short_circuits(self):
        svc = GeocodingService()
        assert await svc.search_city("a") == []
        assert await svc.search_city("") == []
        assert await svc.search_city("   ") == []

    @pytest.mark.asyncio
    async def test_search_parses_results(self):
        svc = GeocodingService()
        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = {
            "results": [
                {
                    "name": "Vilnius",
                    "country": "Lithuania",
                    "admin1": "Vilnius",
                    "latitude": 54.6872,
                    "longitude": 25.2797,
                    "timezone": "Europe/Vilnius",
                },
                {
                    "name": "Vilnius",
                    "country": "Bolivia",
                    "admin1": "Sucre",
                    "latitude": -19.0,
                    "longitude": -65.0,
                    "timezone": "America/La_Paz",
                },
            ]
        }
        client = AsyncMock()
        client.get = AsyncMock(return_value=fake_resp)
        svc._client = client

        results = await svc.search_city("Vilnius")
        assert len(results) == 2
        assert results[0]["name"] == "Vilnius"
        assert results[0]["country"] == "Lithuania"
        assert results[0]["latitude"] == 54.6872
        assert results[0]["timezone"] == "Europe/Vilnius"

    @pytest.mark.asyncio
    async def test_empty_results_returned_as_empty_list(self):
        svc = GeocodingService()
        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = {}  # no "results" key

        client = AsyncMock()
        client.get = AsyncMock(return_value=fake_resp)
        svc._client = client

        assert await svc.search_city("Nowhereville") == []

    @pytest.mark.asyncio
    async def test_http_error_returns_empty_list(self):
        svc = GeocodingService()
        client = AsyncMock()
        client.get = AsyncMock(side_effect=Exception("network down"))
        svc._client = client

        assert await svc.search_city("Vilnius") == []
