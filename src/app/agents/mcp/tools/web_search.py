"""DuckDuckGo HTML search — no API key, no quota."""

from __future__ import annotations

import logging

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_ENDPOINT = "https://html.duckduckgo.com/html/"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
}


def parse_results(html: str, max_results: int) -> list[dict]:
    """Extract result rows from a DuckDuckGo HTML response."""
    soup = BeautifulSoup(html, "html.parser")
    out: list[dict] = []
    for row in soup.select("div.result"):
        title_el = row.select_one("a.result__a")
        snippet_el = row.select_one("a.result__snippet, .result__snippet")
        if not title_el:
            continue
        out.append(
            {
                "title": title_el.get_text(strip=True),
                "url": title_el.get("href", ""),
                "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
            }
        )
        if len(out) >= max_results:
            break
    return out


async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo and return up to `max_results` `{title, url, snippet}` dicts."""
    if not query.strip():
        return []
    max_results = max(1, min(max_results, 10))

    try:
        async with httpx.AsyncClient(timeout=10, headers=_HEADERS) as client:
            resp = await client.post(_ENDPOINT, data={"q": query})
            resp.raise_for_status()
            return parse_results(resp.text, max_results)
    except Exception as e:
        logger.warning("web_search failed for %r: %s", query, e)
        return []
