"""CrewAI crew orchestration.

Pipeline: Analyst (VLM) -> Maintenance Planner (MCP tools) ->
          Report Writer -> QA Reviewer (MCP tools)

Loads agent and task definitions from YAML configs, builds the
CrewAI crew, and runs the analysis pipeline. The Maintenance Planner
and QA Reviewer are given the FastMCP server in
src/app/agents/mcp/server.py over stdio via crewai-tools'
MCPServerAdapter (matching the thesis: only those two agents have
external tool access).
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import yaml
from crewai import LLM, Agent, Crew, Process, Task

from app.services.gemini import GeminiClient

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent / "config"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # src/

try:
    from crewai_tools import MCPServerAdapter
    from mcp import StdioServerParameters
    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False
    logger.warning("crewai-tools[mcp] not installed; planner will run tool-less")


@lru_cache(maxsize=1)
def _load_yaml(filename: str) -> dict:
    """Load and cache YAML config file."""
    path = CONFIG_DIR / filename
    with open(path) as f:
        return yaml.safe_load(f)


def _mcp_server_params() -> "StdioServerParameters | None":
    """Build the StdioServerParameters that launch our FastMCP server."""
    if not _HAS_MCP:
        return None
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.agents.mcp.server"],
        env=None,
        cwd=str(PROJECT_ROOT),
    )


class SolarSentinelCrew:
    """CrewAI crew for solar panel defect analysis."""

    AGENT_KEYS = (
        "analyzer_agent",
        "maintenance_planner_agent",
        "report_writer_agent",
        "qa_reviewer_agent",
    )

    # Per the thesis (p.45): only the Maintenance Planner and the
    # Critic / QA Reviewer have access to external MCP tools.
    TOOL_USING_AGENTS = frozenset(
        {"maintenance_planner_agent", "qa_reviewer_agent"}
    )

    def __init__(self, gemini_client: GeminiClient) -> None:
        self.gemini_client = gemini_client
        self._last_model_name: str = "unknown"

    @property
    def agents_config(self) -> dict:
        return _load_yaml("agents.yaml")

    @property
    def tasks_config(self) -> dict:
        return _load_yaml("tasks.yaml")

    def _get_llm(self) -> LLM | None:
        """Get the best available LLM for CrewAI agents."""
        if not self.gemini_client.ranked_models:
            return None
        best = self.gemini_client.ranked_models[0]
        self._last_model_name = best.name
        return LLM(
            model=f"gemini/{best.name}",
            api_key=self.gemini_client.api_key,
        )

    def _build_agents(self, planner_tools: list | None = None) -> dict[str, Agent]:
        """Build CrewAI agents from YAML config.

        The analyzer is multimodal (uses AddImageTool to attach the panel image).
        The Maintenance Planner and the QA Reviewer share the same MCP tools
        (web_search, current_time, weather_forecast); other agents run
        tool-less. If no tools loaded successfully, both run without tools.
        """
        llm = self._get_llm()
        agents: dict[str, Agent] = {}

        for key in self.AGENT_KEYS:
            config = self.agents_config[key]
            kwargs: dict = {
                "role": config["role"],
                "goal": config["goal"],
                "backstory": config["backstory"],
                "llm": llm,
                "verbose": False,
            }
            if key == "analyzer_agent":
                kwargs["multimodal"] = True
            if key in self.TOOL_USING_AGENTS and planner_tools:
                kwargs["tools"] = list(planner_tools)
            agents[key] = Agent(**kwargs)

        return agents

    async def analyze_detection(
        self,
        defect_class: str,
        confidence: float,
        bbox: dict,
        panel_id: str,
        image_path: str,
        weather_summary: str = "Not available",
        temperature: str = "Not available",
        historical_context: str = "No previous reports in the last 7 days.",
        latitude: str = "",
        longitude: str = "",
        tz_name: str = "UTC",
    ) -> dict:
        """Run the full CrewAI analysis pipeline.

        Returns a dict with: severity, urgency, root_cause, trend_analysis,
        report_markdown, qa_score, qa_approved, defect_subtype,
        planner_output (raw JSON string), recommended_actions, timing_window,
        references.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        context = {
            "defect_class": defect_class,
            "confidence": f"{confidence:.2%}",
            "bbox": json.dumps(bbox),
            "panel_id": panel_id,
            "timestamp": timestamp,
            "weather_summary": weather_summary,
            "temperature": temperature,
            "historical_context": historical_context,
            "image_path": image_path or "(no image available)",
            "latitude": latitude or "0.0",
            "longitude": longitude or "0.0",
            "tz_name": tz_name or "UTC",
        }

        server_params = _mcp_server_params()
        adapter_cm = (
            MCPServerAdapter(server_params)
            if (_HAS_MCP and server_params is not None)
            else nullcontext(None)
        )

        try:
            with adapter_cm as planner_tools:
                if _HAS_MCP and planner_tools is not None:
                    logger.info(
                        "MCP tools loaded for planner+QA: %s",
                        [t.name for t in planner_tools],
                    )
                else:
                    logger.warning(
                        "Planner+QA running tool-less (MCP unavailable)"
                    )

                agents = self._build_agents(planner_tools=planner_tools)

                analyze_task = Task(
                    description=self.tasks_config["analyze_defect_task"][
                        "description"
                    ].format(**context),
                    expected_output=self.tasks_config["analyze_defect_task"][
                        "expected_output"
                    ],
                    agent=agents["analyzer_agent"],
                )

                planning_task = Task(
                    description=self.tasks_config["maintenance_planning_task"][
                        "description"
                    ].format(**context),
                    expected_output=self.tasks_config["maintenance_planning_task"][
                        "expected_output"
                    ],
                    agent=agents["maintenance_planner_agent"],
                    context=[analyze_task],
                )

                write_task = Task(
                    description=self.tasks_config["write_report_task"][
                        "description"
                    ],
                    expected_output=self.tasks_config["write_report_task"][
                        "expected_output"
                    ],
                    agent=agents["report_writer_agent"],
                    context=[analyze_task, planning_task],
                )

                qa_task = Task(
                    description=self.tasks_config["qa_review_task"][
                        "description"
                    ].format(**context),
                    expected_output=self.tasks_config["qa_review_task"][
                        "expected_output"
                    ],
                    agent=agents["qa_reviewer_agent"],
                    context=[analyze_task, planning_task, write_task],
                )

                crew = Crew(
                    agents=list(agents.values()),
                    tasks=[analyze_task, planning_task, write_task, qa_task],
                    process=Process.sequential,
                    verbose=False,
                )

                # kickoff_async() is required when called from inside a running
                # event loop — CrewAI 1.14 detects the loop and refuses kickoff().
                await crew.kickoff_async()
                result = self._parse_result(
                    analyze_task, planning_task, write_task, qa_task
                )
                result["usage"] = self._extract_usage(crew)
                return result
        except Exception as e:
            logger.error("CrewAI pipeline failed: %s", e, exc_info=True)
            return self._fallback_result(defect_class, confidence)

    def _extract_usage(self, crew) -> dict:
        """Pull total tokens off crew.usage_metrics; never raises."""
        try:
            usage = getattr(crew, "usage_metrics", None)
            total = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0
        except Exception:
            total = 0
        return {"total_tokens": total, "model_name": self._last_model_name}

    @staticmethod
    def _safe_json(task_output, fallback: dict) -> dict:
        """Best-effort JSON parse of a task output. Returns `fallback` on failure."""
        try:
            parsed = json.loads(str(task_output))
            return parsed if isinstance(parsed, dict) else fallback
        except (json.JSONDecodeError, TypeError):
            return fallback

    @classmethod
    def _parse_result(cls, analyze_task, planning_task, write_task, qa_task) -> dict:
        """Parse the crew result into a structured dict."""
        analysis = cls._safe_json(
            analyze_task.output,
            {
                "is_real_defect": True,
                "defect_subtype": "other",
                "severity": "WARNING",
                "urgency": "WITHIN_1_WEEK",
                "root_cause": str(analyze_task.output),
                "trend_analysis": "N/A",
                "visual_evidence": "",
            },
        )
        planner = cls._safe_json(
            planning_task.output,
            {
                "recommended_actions": [str(planning_task.output)],
                "timing_window": "N/A",
                "weather_constraints": "N/A",
                "references": [],
            },
        )
        qa_output = cls._safe_json(
            qa_task.output,
            {
                "score": 7,
                "approved": True,
                "feedback": "",
                "revised_report": str(qa_task.output),
            },
        )

        report_md = qa_output.get("revised_report") or str(write_task.output)

        return {
            "severity": analysis.get("severity", "WARNING"),
            "urgency": analysis.get("urgency", "WITHIN_1_WEEK"),
            "root_cause": analysis.get("root_cause", "Unknown"),
            "trend_analysis": analysis.get("trend_analysis", "N/A"),
            "defect_subtype": analysis.get("defect_subtype", "other"),
            "is_real_defect": analysis.get("is_real_defect", True),
            "report_markdown": report_md,
            "qa_score": qa_output.get("score", 5),
            "qa_approved": qa_output.get("approved", False),
            "recommended_actions": planner.get("recommended_actions", []),
            "timing_window": planner.get("timing_window", ""),
            "weather_constraints": planner.get("weather_constraints", ""),
            "references": planner.get("references", []),
            "planner_output_json": json.dumps(planner),
            "analyzer_output_json": json.dumps(analysis),
        }

    @staticmethod
    def _fallback_result(defect_class: str, confidence: float) -> dict:
        """Fallback when the entire pipeline fails."""
        return {
            "severity": "WARNING",
            "urgency": "WITHIN_1_WEEK",
            "root_cause": f"Detected {defect_class} with {confidence:.0%} confidence. "
                          "Automated analysis unavailable.",
            "trend_analysis": "N/A — pipeline error",
            "defect_subtype": "other",
            "is_real_defect": True,
            "report_markdown": (
                f"# Solar Panel Defect Alert\n\n"
                f"**Defect:** {defect_class}\n"
                f"**Confidence:** {confidence:.0%}\n\n"
                f"Automated analysis failed. Manual inspection recommended.\n"
            ),
            "qa_score": 1,
            "qa_approved": False,
            "recommended_actions": ["Manual inspection by a technician."],
            "timing_window": "N/A",
            "weather_constraints": "N/A",
            "references": [],
            "planner_output_json": "{}",
            "analyzer_output_json": "{}",
            "usage": {"total_tokens": 0, "model_name": "unknown"},
        }
