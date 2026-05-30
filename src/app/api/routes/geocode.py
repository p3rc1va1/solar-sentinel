"""Geocoding endpoint — proxy to the Open-Meteo city search.

Lets the Settings UI offer a friendly city picker that fills in
weather_latitude / weather_longitude / weather_timezone.
"""

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_geocoding
from app.services.geocoding import GeocodingService

router = APIRouter(prefix="/geocode", tags=["geocode"])


@router.get("")
async def search_cities(
    q: str = Query(..., min_length=1, description="City name (>= 2 chars)"),
    limit: int = Query(5, ge=1, le=20),
    geo: GeocodingService = Depends(get_geocoding),
) -> list[dict]:
    """Search for cities matching `q`.

    Returns a list of {name, country, admin1, latitude, longitude, timezone}.
    Empty list when no matches or when `q` is too short.
    """
    return await geo.search_city(q, limit=limit)
