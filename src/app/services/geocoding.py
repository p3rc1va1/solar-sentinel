"""Open-Meteo geocoding — free city search, no API key.

Used by the Settings UI to let the user pick their location by name
rather than typing in latitude/longitude. The Open-Meteo geocoding API
returns name, country, region, lat/lon, and an IANA timezone — exactly
what the rest of the app needs.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_MIN_QUERY_LEN = 2


class GeocodingService:
    """Looks up cities via the Open-Meteo geocoding API."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(timeout=10)

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def search_city(self, query: str, limit: int = 5) -> list[dict]:
        """Search for cities matching `query`.

        Returns a list of {name, country, admin1, latitude, longitude, timezone}
        dicts (empty list on error or short query).
        """
        q = (query or "").strip()
        if len(q) < _MIN_QUERY_LEN:
            return []

        client = self._client or httpx.AsyncClient(timeout=10)
        try:
            try:
                resp = await client.get(
                    _BASE_URL,
                    params={
                        "name": q,
                        "count": max(1, min(limit, 20)),
                        "language": "en",
                        "format": "json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            finally:
                if self._client is None:
                    await client.aclose()
        except Exception as e:
            logger.warning("Geocoding search failed for %r: %s", q, e)
            return []

        results = []
        for r in data.get("results") or []:
            results.append(
                {
                    "name": r.get("name", ""),
                    "country": r.get("country", ""),
                    "admin1": r.get("admin1", ""),
                    "latitude": float(r.get("latitude", 0.0)),
                    "longitude": float(r.get("longitude", 0.0)),
                    "timezone": r.get("timezone", "UTC"),
                }
            )
        return results
