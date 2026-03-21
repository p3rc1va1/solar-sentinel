"""Gemini API client with automatic model fallback.

Uses the new google.genai SDK (replaces deprecated google.generativeai).
"""

import logging

from google import genai
from google.genai import errors as genai_errors

from app.agents.model_router import RankedModel, discover_and_rank_models

logger = logging.getLogger(__name__)


class GeminiClient:
    """Gemini API client with dynamic model ranking and fallback."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.ranked_models: list[RankedModel] = []
        self._client: genai.Client | None = None
        self._configured = False

    def configure(self) -> None:
        """Configure the API and discover available models."""
        if not self.api_key:
            logger.warning("No Gemini API key provided")
            return
        self._client = genai.Client(api_key=self.api_key)
        self.ranked_models = discover_and_rank_models(self.api_key)
        self._configured = True
        logger.info("Gemini client configured with %d models", len(self.ranked_models))

    def refresh_models(self) -> None:
        """Re-discover models (call daily to pick up new releases)."""
        self.ranked_models = discover_and_rank_models(self.api_key)

    async def generate(self, prompt: str) -> str:
        """Generate content, falling back through ranked models on quota errors."""
        if not self._configured or not self._client or not self.ranked_models:
            logger.error("Gemini client not configured or no models available")
            return self._template_response()

        for model_info in self.ranked_models:
            try:
                response = await self._client.aio.models.generate_content(
                    model=model_info.name,
                    contents=prompt,
                )

                if response.text:
                    logger.info("Gemini response from %s", model_info.name)
                    return response.text
                else:
                    logger.warning("Empty response from %s", model_info.name)
                    continue

            except genai_errors.ClientError as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    logger.warning(
                        "Model %s quota exceeded (429), trying next...",
                        model_info.name,
                    )
                else:
                    logger.error(
                        "Model %s client error: %s, trying next...",
                        model_info.name, e,
                    )
                continue
            except Exception as e:
                logger.error(
                    "Model %s error: %s, trying next...",
                    model_info.name, e,
                )
                continue

        logger.error("All Gemini models exhausted — returning template response")
        return self._template_response()

    def _template_response(self) -> str:
        """Fallback template when all models are exhausted."""
        return (
            '{"severity": "WARNING", "root_cause": "Unable to perform LLM analysis — '
            'API quota exhausted. Manual inspection recommended.", '
            '"urgency": "WITHIN_1_WEEK", "trend_analysis": "N/A — LLM unavailable", '
            '"preliminary_recommendation": "Schedule manual inspection at earliest convenience.", '
            '"confidence_assessment": "Low — automated analysis unavailable"}'
        )
