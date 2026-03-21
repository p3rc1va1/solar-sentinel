"""CrewAI crew orchestration.

Loads agent and task definitions from YAML configs, builds the
CrewAI crew, and runs the analysis pipeline.
"""

import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import yaml
from crewai import Agent, Crew, LLM, Process, Task

from app.services.gemini import GeminiClient

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent / "config"


@lru_cache(maxsize=1)
def _load_yaml(filename: str) -> dict:
    """Load and cache YAML config file."""
    path = CONFIG_DIR / filename
    with open(path) as f:
        return yaml.safe_load(f)


class SolarSentinelCrew:
    """CrewAI crew for solar panel defect analysis."""

    def __init__(self, gemini_client: GeminiClient) -> None:
        self.gemini_client = gemini_client

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
        return LLM(
            model=f"gemini/{best.name}",
            api_key=self.gemini_client.api_key,
        )

    def _build_agents(self) -> dict[str, Agent]:
        """Build CrewAI agents from YAML config."""
        llm = self._get_llm()
        agents = {}

        for key in ["analyzer_agent", "report_writer_agent", "qa_reviewer_agent"]:
            config = self.agents_config[key]
            agents[key] = Agent(
                role=config["role"],
                goal=config["goal"],
                backstory=config["backstory"],
                llm=llm,
                verbose=False,
            )

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
    ) -> dict:
        """Run the full CrewAI analysis pipeline.

        Returns a dict with: severity, urgency, root_cause, trend_analysis,
        report_markdown, qa_score, qa_approved.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        agents = self._build_agents()

        context = {
            "defect_class": defect_class,
            "confidence": f"{confidence:.2%}",
            "bbox": json.dumps(bbox),
            "panel_id": panel_id,
            "timestamp": timestamp,
            "weather_summary": weather_summary,
            "temperature": temperature,
            "historical_context": historical_context,
        }

        analyze_task = Task(
            description=self.tasks_config["analyze_defect_task"]["description"].format(**context),
            expected_output=self.tasks_config["analyze_defect_task"]["expected_output"],
            agent=agents["analyzer_agent"],
        )

        write_task = Task(
            description=self.tasks_config["write_report_task"]["description"].format(
                analyzer_output="{context}",
                **{k: v for k, v in context.items() if k != "analyzer_output"},
            ),
            expected_output=self.tasks_config["write_report_task"]["expected_output"],
            agent=agents["report_writer_agent"],
            context=[analyze_task],
        )

        qa_task = Task(
            description=self.tasks_config["qa_review_task"]["description"].format(
                analyzer_output="{context}",
                report_content="{context}",
                **{k: v for k, v in context.items()
                   if k not in ("analyzer_output", "report_content")},
            ),
            expected_output=self.tasks_config["qa_review_task"]["expected_output"],
            agent=agents["qa_reviewer_agent"],
            context=[analyze_task, write_task],
        )

        crew = Crew(
            agents=list(agents.values()),
            tasks=[analyze_task, write_task, qa_task],
            process=Process.sequential,
            verbose=False,
        )

        try:
            result = crew.kickoff()
            return self._parse_result(result, analyze_task, write_task, qa_task)
        except Exception as e:
            logger.error("CrewAI pipeline failed: %s", e, exc_info=True)
            return self._fallback_result(defect_class, confidence)

    @staticmethod
    def _parse_result(crew_result, analyze_task, write_task, qa_task) -> dict:
        """Parse the crew result into a structured dict."""
        try:
            qa_output = json.loads(str(qa_task.output))
        except (json.JSONDecodeError, TypeError):
            qa_output = {
                "score": 7,
                "approved": True,
                "feedback": "",
                "revised_report": str(qa_task.output),
            }

        try:
            analysis = json.loads(str(analyze_task.output))
        except (json.JSONDecodeError, TypeError):
            analysis = {
                "severity": "WARNING",
                "root_cause": str(analyze_task.output),
                "urgency": "WITHIN_1_WEEK",
                "trend_analysis": "N/A",
            }

        report_md = qa_output.get("revised_report", str(write_task.output))

        return {
            "severity": analysis.get("severity", "WARNING"),
            "urgency": analysis.get("urgency", "WITHIN_1_WEEK"),
            "root_cause": analysis.get("root_cause", "Unknown"),
            "trend_analysis": analysis.get("trend_analysis", "N/A"),
            "report_markdown": report_md,
            "qa_score": qa_output.get("score", 5),
            "qa_approved": qa_output.get("approved", False),
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
            "report_markdown": (
                f"# Solar Panel Defect Alert\n\n"
                f"**Defect:** {defect_class}\n"
                f"**Confidence:** {confidence:.0%}\n\n"
                f"Automated analysis failed. Manual inspection recommended.\n"
            ),
            "qa_score": 1,
            "qa_approved": False,
        }
