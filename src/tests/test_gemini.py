"""Tests for app.services.gemini."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.model_router import RankedModel
from app.services.gemini import GeminiClient


class TestGeminiClient:
    def test_init_defaults(self):
        client = GeminiClient(api_key="test")
        assert client.api_key == "test"
        assert client._configured is False
        assert client.ranked_models == []

    @patch("app.services.gemini.discover_and_rank_models")
    @patch("app.services.gemini.genai")
    def test_configure(self, mock_genai, mock_discover):
        mock_discover.return_value = [
            RankedModel(name="gemini-2.5-pro", score=30, tier="pro"),
        ]
        client = GeminiClient(api_key="test-key")
        client.configure()
        assert client._configured is True
        assert len(client.ranked_models) == 1
        mock_genai.Client.assert_called_once_with(api_key="test-key")

    def test_configure_no_key(self):
        client = GeminiClient(api_key="")
        client.configure()
        assert client._configured is False

    @pytest.mark.asyncio
    async def test_generate_unconfigured(self):
        client = GeminiClient(api_key="")
        result = await client.generate("test prompt")
        assert "WARNING" in result

    @pytest.mark.asyncio
    @patch("app.services.gemini.discover_and_rank_models")
    @patch("app.services.gemini.genai")
    async def test_generate_success(self, mock_genai, mock_discover):
        mock_discover.return_value = [
            RankedModel(name="gemini-pro", score=30, tier="pro"),
        ]
        mock_response = MagicMock()
        mock_response.text = "Analysis result"
        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        client = GeminiClient(api_key="test-key")
        client.configure()
        result = await client.generate("analyze this")
        assert result == "Analysis result"

    @pytest.mark.asyncio
    @patch("app.services.gemini.discover_and_rank_models")
    @patch("app.services.gemini.genai")
    async def test_generate_empty_response_falls_through(self, mock_genai, mock_discover):
        mock_discover.return_value = [
            RankedModel(name="m1", score=10, tier="flash"),
        ]
        mock_response = MagicMock()
        mock_response.text = None
        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        client = GeminiClient(api_key="test-key")
        client.configure()
        result = await client.generate("test")
        assert "WARNING" in result  # falls through to template

    def test_template_response(self):
        client = GeminiClient(api_key="")
        result = client._template_response()
        assert "WARNING" in result
        assert "Manual inspection" in result

    @patch("app.services.gemini.discover_and_rank_models")
    def test_refresh_models(self, mock_discover):
        mock_discover.return_value = [
            RankedModel(name="new-model", score=50, tier="pro"),
        ]
        client = GeminiClient(api_key="test")
        client.refresh_models()
        assert client.ranked_models[0].name == "new-model"
