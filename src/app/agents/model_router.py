"""Dynamic Gemini model discovery, ranking, and fallback.

Queries the Gemini API for available models, ranks them by
version and tier, and provides a fallback chain for API calls.

Uses the new google.genai SDK (replaces deprecated google.generativeai).
"""

import logging
import re
from dataclasses import dataclass

from google import genai

logger = logging.getLogger(__name__)

TIER_WEIGHTS = {"pro": 5, "flash": 3, "flash-lite": 1}


@dataclass
class RankedModel:
    name: str
    score: int
    tier: str
    version: float = 0.0


def _parse_version(model_name: str) -> float:
    """Extract version number from model name (e.g., 'gemini-2.5-pro' -> 2.5)."""
    match = re.search(r"(\d+)\.(\d+)", model_name)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    match = re.search(r"(\d+)", model_name)
    if match:
        return float(match.group(1))
    return 0.0


def _identify_tier(model_name: str) -> str:
    """Identify model tier from name."""
    name_lower = model_name.lower()
    if "flash-lite" in name_lower or "flash_lite" in name_lower:
        return "flash-lite"
    if "flash" in name_lower:
        return "flash"
    if "pro" in name_lower:
        return "pro"
    return "flash"  # default assumption


def discover_and_rank_models(api_key: str) -> list[RankedModel]:
    """Query Gemini API for available models and rank by capability.

    Returns models sorted best-to-worst.
    """
    if not api_key:
        logger.warning("No Gemini API key — model discovery skipped")
        return _get_fallback_list()

    try:
        client = genai.Client(api_key=api_key)
        ranked = []

        for model in client.models.list():
            name = model.name.replace("models/", "")

            # Only include gemini models
            if not name.startswith("gemini"):
                continue

            # Check if generateContent is supported
            methods = model.supported_actions or []
            if "generateContent" not in methods:
                continue

            version = _parse_version(name)
            tier = _identify_tier(name)
            version_score = int(version * 10)
            total_score = version_score + TIER_WEIGHTS.get(tier, 0)

            ranked.append(RankedModel(
                name=name,
                score=total_score,
                tier=tier,
                version=version,
            ))

        ranked.sort(key=lambda m: m.score, reverse=True)
        logger.info(
            "Discovered %d Gemini models. Top 3: %s",
            len(ranked),
            [m.name for m in ranked[:3]],
        )
        return ranked

    except Exception as e:
        logger.error("Model discovery failed: %s — using fallback list", e)
        return _get_fallback_list()


def _get_fallback_list() -> list[RankedModel]:
    """Static fallback list if API discovery fails."""
    return [
        RankedModel(name="gemini-2.5-pro-latest", score=30, tier="pro", version=2.5),
        RankedModel(name="gemini-2.5-flash-latest", score=28, tier="flash", version=2.5),
        RankedModel(name="gemini-2.5-flash-lite-latest", score=26, tier="flash-lite", version=2.5),
    ]
