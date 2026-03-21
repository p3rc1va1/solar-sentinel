"""Tests for app.agents.crew."""

import json
import pytest
from unittest.mock import MagicMock, patch

from app.agents.crew import SolarSentinelCrew, _load_yaml
from app.services.gemini import GeminiClient
from app.agents.model_router import RankedModel


class TestCrewHelpers:
    def test_fallback_result(self):
        result = SolarSentinelCrew._fallback_result("crack", 0.85)
        assert result["severity"] == "WARNING"
        assert result["qa_score"] == 1
        assert "crack" in result["root_cause"]
        assert "85%" in result["root_cause"]

    def test_parse_result_with_valid_json(self):
        analyze_task = MagicMock()
        analyze_task.output = json.dumps({
            "severity": "CRITICAL",
            "root_cause": "Hail damage",
            "urgency": "IMMEDIATE",
            "trend_analysis": "Worsening",
        })

        write_task = MagicMock()
        write_task.output = "# Full Report"

        qa_task = MagicMock()
        qa_task.output = json.dumps({
            "score": 9,
            "approved": True,
            "revised_report": "# Revised Report",
        })

        result = SolarSentinelCrew._parse_result(None, analyze_task, write_task, qa_task)
        assert result["severity"] == "CRITICAL"
        assert result["qa_score"] == 9
        assert result["qa_approved"] is True

    def test_parse_result_with_invalid_json(self):
        analyze_task = MagicMock()
        analyze_task.output = "Not valid JSON"

        write_task = MagicMock()
        write_task.output = "# Report"

        qa_task = MagicMock()
        qa_task.output = "Also not JSON"

        result = SolarSentinelCrew._parse_result(None, analyze_task, write_task, qa_task)
        assert result["severity"] == "WARNING"  # fallback
        assert result["qa_score"] == 7

    def test_get_llm_no_models(self):
        client = GeminiClient(api_key="test")
        crew = SolarSentinelCrew(client)
        assert crew._get_llm() is None

    @patch("app.agents.crew.LLM")
    def test_get_llm_with_models(self, mock_llm):
        client = GeminiClient(api_key="test")
        client.ranked_models = [RankedModel(name="gemini-pro", score=30, tier="pro")]
        crew = SolarSentinelCrew(client)
        result = crew._get_llm()
        mock_llm.assert_called_once_with(model="gemini/gemini-pro", api_key="test")


class TestYamlLoading:
    def test_load_yaml_caching(self):
        # Clear the cache first
        _load_yaml.cache_clear()
        # The function should load without error
        try:
            agents = _load_yaml("agents.yaml")
            assert isinstance(agents, dict)
        except FileNotFoundError:
            pytest.skip("YAML config files not available")
