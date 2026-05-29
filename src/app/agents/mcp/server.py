"""FastMCP server exposing the Maintenance Planner's toolbelt over stdio.

Run as a subprocess by `MCPServerAdapter` from `crewai-tools`. The server
process should not import anything from the FastAPI app (avoid heavy startup);
each tool stands alone.

Run standalone for inspection:
    uv run python -m app.agents.mcp.server
"""

from __future__ import annotations

import logging

from fastmcp import FastMCP

from app.agents.mcp.tools.forecast import weather_forecast as _weather_forecast
from app.agents.mcp.tools.time_tool import current_time as _current_time
from app.agents.mcp.tools.web_search import web_search as _web_search

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

mcp = FastMCP(name="solar-sentinel-tools")


@mcp.tool
async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web (DuckDuckGo) for the given query.

    Use this to look up known issues with PV panels, weather hazards,
    or maintenance procedures relevant to a defect.

    Args:
        query: Free-text search query.
        max_results: Maximum number of results (1-10, default 5).

    Returns:
        List of {title, url, snippet} dicts.
    """
    return await _web_search(query=query, max_results=max_results)


@mcp.tool
def current_time(latitude: float, longitude: float, tz_name: str = "UTC") -> dict:
    """Get current local time and daylight info for the panel site.

    Use this to decide when maintenance can be performed (only during
    daylight hours).

    Args:
        latitude: Site latitude.
        longitude: Site longitude.
        tz_name: IANA timezone name (e.g., "Europe/Vilnius"). Defaults to UTC.

    Returns:
        Dict with utc_now, local_now, sunrise, sunset, is_daylight,
        seconds_until_sunset.
    """
    return _current_time(latitude=latitude, longitude=longitude, tz_name=tz_name)


@mcp.tool
async def weather_forecast(
    latitude: float,
    longitude: float,
    hours: int = 24,
) -> dict:
    """Fetch hourly weather forecast for the panel site (Open-Meteo).

    Use this to determine whether outdoor maintenance is feasible in the
    next window, or whether a defect should be expected to worsen
    (heavy rain, snow, etc.).

    Args:
        latitude: Site latitude.
        longitude: Site longitude.
        hours: Forecast horizon in hours (1-168, default 24).

    Returns:
        Dict with summary string and a list of hourly weather rows.
    """
    return await _weather_forecast(latitude=latitude, longitude=longitude, hours=hours)


def main() -> None:
    """Entry point — runs the MCP server on stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
