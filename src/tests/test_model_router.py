"""Tests for app.agents.model_router."""

import pytest
from unittest.mock import MagicMock, patch

from app.agents.model_router import (
    RankedModel,
    _get_fallback_list,
    _identify_tier,
    _parse_version,
    discover_and_rank_models,
)


class TestParseVersion:
    def test_standard_version(self):
        assert _parse_version("gemini-2.5-pro") == 2.5

    def test_single_digit(self):
        assert _parse_version("gemini-1-pro") == 1.0

    def test_no_version(self):
        assert _parse_version("some-model") == 0.0

    def test_version_with_suffix(self):
        assert _parse_version("gemini-2.0-flash-latest") == 2.0


class TestIdentifyTier:
    def test_pro(self):
        assert _identify_tier("gemini-2.5-pro") == "pro"

    def test_flash(self):
        assert _identify_tier("gemini-2.0-flash") == "flash"

    def test_flash_lite(self):
        assert _identify_tier("gemini-2.0-flash-lite") == "flash-lite"

    def test_flash_lite_underscore(self):
        assert _identify_tier("gemini-2.0-flash_lite") == "flash-lite"

    def test_default_is_flash(self):
        assert _identify_tier("gemini-unknown") == "flash"


class TestFallbackList:
    def test_returns_three_models(self):
        models = _get_fallback_list()
        assert len(models) == 3
        assert all(isinstance(m, RankedModel) for m in models)

    def test_sorted_by_score_desc(self):
        models = _get_fallback_list()
        scores = [m.score for m in models]
        assert scores == sorted(scores, reverse=True)


class TestDiscoverModels:
    def test_no_api_key_returns_fallback(self):
        result = discover_and_rank_models("")
        assert len(result) == 3  # fallback list

    @patch("app.agents.model_router.genai")
    def test_discovery_filters_non_gemini(self, mock_genai):
        model = MagicMock()
        model.name = "models/text-bison-001"
        model.supported_actions = ["generateContent"]
        mock_genai.Client.return_value.models.list.return_value = [model]

        result = discover_and_rank_models("fake-key")
        assert len(result) == 0

    @patch("app.agents.model_router.genai")
    def test_discovery_exception_returns_fallback(self, mock_genai):
        mock_genai.Client.side_effect = Exception("API error")
        result = discover_and_rank_models("fake-key")
        assert len(result) == 3

    @patch("app.agents.model_router.genai")
    def test_discovery_ranks_correctly(self, mock_genai):
        models = []
        for name, actions in [
            ("models/gemini-2.5-pro-latest", ["generateContent"]),
            ("models/gemini-2.0-flash", ["generateContent"]),
        ]:
            m = MagicMock()
            m.name = name
            m.supported_actions = actions
            models.append(m)

        mock_genai.Client.return_value.models.list.return_value = models
        result = discover_and_rank_models("fake-key")
        assert len(result) == 2
        assert result[0].score >= result[1].score
