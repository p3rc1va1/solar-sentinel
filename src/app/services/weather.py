"""Weather context service (optional)."""

import logging

import httpx

logger = logging.getLogger(__name__)

_NOT_AVAILABLE = {
    "summary": "Not available",
    "temperature": "Not available",
    "precipitation": "Not available",
    "uv_index": "Not available",
}

_WMO_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight rain showers", 81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail",
}


class WeatherService:
    """Fetches current weather data for detection context enrichment.

    Uses Open-Meteo (free, no API key required).
    """

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, latitude: str = "", longitude: str = "") -> None:
        self.latitude = latitude
        self.longitude = longitude
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Create a reusable HTTP client."""
        self._client = httpx.AsyncClient(timeout=10)

    async def stop(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_current_weather(self) -> dict:
        """Fetch current weather conditions."""
        if not self.latitude or not self.longitude:
            return dict(_NOT_AVAILABLE)

        try:
            client = self._client or httpx.AsyncClient(timeout=10)
            try:
                resp = await client.get(
                    self.BASE_URL,
                    params={
                        "latitude": self.latitude,
                        "longitude": self.longitude,
                        "current": "temperature_2m,weather_code,precipitation,uv_index",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            finally:
                if self._client is None:
                    await client.aclose()

            current = data.get("current", {})
            code = current.get("weather_code", 0)

            return {
                "summary": _WMO_CODES.get(code, f"Weather code {code}"),
                "temperature": f"{current.get('temperature_2m', 'N/A')}°C",
                "precipitation": f"{current.get('precipitation', 0)} mm",
                "uv_index": str(current.get("uv_index", "N/A")),
            }

        except Exception as e:
            logger.warning("Weather fetch failed: %s", e)
            return {
                "summary": "Unavailable",
                "temperature": "N/A",
                "precipitation": "N/A",
                "uv_index": "N/A",
            }
