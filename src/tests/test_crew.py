"""Tests for app.agents.crew."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.agents.crew import SolarSentinelCrew, _load_yaml
from app.agents.model_router import RankedModel
from app.services.gemini import GeminiClient


class TestCrewHelpers:
    def test_fallback_result(self):
        result = SolarSentinelCrew._fallback_result("crack", 0.85)
        assert result["severity"] == "WARNING"
        assert result["qa_score"] == 1
        assert "crack" in result["root_cause"]
        assert "85%" in result["root_cause"]
        # New shape from 4-agent pipeline.
        assert "recommended_actions" in result
        assert "timing_window" in result
        assert "planner_output_json" in result
        assert "analyzer_output_json" in result
        # Token-counting fields always present.
        assert result["usage"] == {"total_tokens": 0, "model_name": "unknown"}

    def test_extract_usage_from_crew(self):
        client = GeminiClient(api_key="")
        crew_obj = SolarSentinelCrew(client)
        crew_obj._last_model_name = "gemini-2.5-flash"
        fake_crew = MagicMock()
        fake_crew.usage_metrics = MagicMock(total_tokens=4321)
        usage = crew_obj._extract_usage(fake_crew)
        assert usage == {"total_tokens": 4321, "model_name": "gemini-2.5-flash"}

    def test_extract_usage_handles_missing_metrics(self):
        client = GeminiClient(api_key="")
        crew_obj = SolarSentinelCrew(client)
        fake_crew = MagicMock()
        fake_crew.usage_metrics = None
        usage = crew_obj._extract_usage(fake_crew)
        assert usage["total_tokens"] == 0
        assert usage["model_name"] == "unknown"

    def test_parse_result_with_valid_json(self):
        analyze_task = MagicMock()
        analyze_task.output = json.dumps({
            "is_real_defect": True,
            "defect_subtype": "physical_damage",
            "severity": "CRITICAL",
            "root_cause": "Hail damage",
            "urgency": "IMMEDIATE",
            "trend_analysis": "Worsening",
            "visual_evidence": "Cracked glass surface visible in upper-right",
        })

        planning_task = MagicMock()
        planning_task.output = json.dumps({
            "recommended_actions": [
                "Isolate affected panel from inverter",
                "Schedule glass replacement within 48h",
            ],
            "timing_window": "Tomorrow 10:00-14:00 local — dry window",
            "weather_constraints": "Avoid wet conditions",
            "references": [
                {"title": "PV hail damage guide", "url": "https://example.com/x"},
            ],
        })

        write_task = MagicMock()
        write_task.output = "# Full Report"

        qa_task = MagicMock()
        qa_task.output = json.dumps({
            "score": 9,
            "approved": True,
            "revised_report": "# Revised Report",
        })

        result = SolarSentinelCrew._parse_result(
            analyze_task, planning_task, write_task, qa_task
        )
        assert result["severity"] == "CRITICAL"
        assert result["qa_score"] == 9
        assert result["qa_approved"] is True
        assert result["defect_subtype"] == "physical_damage"
        assert len(result["recommended_actions"]) == 2
        assert "Tomorrow" in result["timing_window"]
        assert result["references"][0]["url"] == "https://example.com/x"
        assert json.loads(result["planner_output_json"])["timing_window"]
        assert result["report_markdown"] == "# Revised Report"

    def test_parse_result_with_invalid_json(self):
        analyze_task = MagicMock()
        analyze_task.output = "Not valid JSON"

        planning_task = MagicMock()
        planning_task.output = "Also not JSON, just prose"

        write_task = MagicMock()
        write_task.output = "# Report"

        qa_task = MagicMock()
        qa_task.output = "Also not JSON"

        result = SolarSentinelCrew._parse_result(
            analyze_task, planning_task, write_task, qa_task
        )
        # Falls back through every stage.
        assert result["severity"] == "WARNING"
        assert result["qa_score"] == 7
        assert result["recommended_actions"] == ["Also not JSON, just prose"]
        # qa fallback returns the qa output as the revised report.
        assert "Also not JSON" in result["report_markdown"]

    def test_get_llm_no_models(self):
        client = GeminiClient(api_key="test")
        crew = SolarSentinelCrew(client)
        assert crew._get_llm() is None

    @patch("app.agents.crew.LLM")
    def test_get_llm_with_models(self, mock_llm):
        client = GeminiClient(api_key="test")
        client.ranked_models = [RankedModel(name="gemini-pro", score=30, tier="pro")]
        crew = SolarSentinelCrew(client)
        crew._get_llm()
        mock_llm.assert_called_once_with(model="gemini/gemini-pro", api_key="test")


class TestAgentBuilding:
    def test_build_agents_returns_four(self):
        client = GeminiClient(api_key="")
        crew = SolarSentinelCrew(client)
        agents = crew._build_agents(planner_tools=None)
        assert set(agents.keys()) == {
            "analyzer_agent",
            "maintenance_planner_agent",
            "report_writer_agent",
            "qa_reviewer_agent",
        }

    def test_planner_receives_tools_when_provided(self):
        from crewai.tools import BaseTool

        class FakeTool(BaseTool):
            name: str = "fake_tool"
            description: str = "fake"

            def _run(self, *args, **kwargs):
                return "ok"

        client = GeminiClient(api_key="")
        crew = SolarSentinelCrew(client)
        fake_tool = FakeTool()
        agents = crew._build_agents(planner_tools=[fake_tool])
        assert fake_tool in agents["maintenance_planner_agent"].tools

    def test_qa_reviewer_receives_tools_when_provided(self):
        """Per the thesis (p.45), the Critic / QA Reviewer also has MCP tools."""
        from crewai.tools import BaseTool

        class FakeTool(BaseTool):
            name: str = "fake_tool"
            description: str = "fake"

            def _run(self, *args, **kwargs):
                return "ok"

        client = GeminiClient(api_key="")
        crew = SolarSentinelCrew(client)
        fake_tool = FakeTool()
        agents = crew._build_agents(planner_tools=[fake_tool])
        assert fake_tool in agents["qa_reviewer_agent"].tools

    def test_only_planner_and_qa_have_tools(self):
        """Analyzer and Report Writer must remain tool-less."""
        from crewai.tools import BaseTool

        class FakeTool(BaseTool):
            name: str = "fake_tool"
            description: str = "fake"

            def _run(self, *args, **kwargs):
                return "ok"

        client = GeminiClient(api_key="")
        crew = SolarSentinelCrew(client)
        agents = crew._build_agents(planner_tools=[FakeTool()])
        # Analyzer and writer should have no tools (empty list or None).
        assert not getattr(agents["analyzer_agent"], "tools", None)
        assert not getattr(agents["report_writer_agent"], "tools", None)


class TestYamlLoading:
    def test_load_yaml_caching(self):
        _load_yaml.cache_clear()
        try:
            agents = _load_yaml("agents.yaml")
            assert isinstance(agents, dict)
            assert "maintenance_planner_agent" in agents
        except FileNotFoundError:
            pytest.skip("YAML config files not available")

    def test_tasks_yaml_has_planning_task(self):
        _load_yaml.cache_clear()
        try:
            tasks = _load_yaml("tasks.yaml")
            assert "maintenance_planning_task" in tasks
        except FileNotFoundError:
            pytest.skip("YAML config files not available")
