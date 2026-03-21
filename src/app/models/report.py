"""Pydantic models for maintenance reports."""

from datetime import datetime

from pydantic import BaseModel, Field

from app.models.detection import Severity, Urgency


class ReportCreate(BaseModel):
    """Data produced by the CrewAI pipeline."""

    detection_id: int
    severity: Severity
    urgency: Urgency
    root_cause: str
    trend_analysis: str
    report_markdown: str
    qa_score: int = Field(ge=1, le=10)
    qa_approved: bool


class ReportRecord(ReportCreate):
    """Full report record as persisted."""

    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class ReportSummary(BaseModel):
    """Lightweight report info for list responses."""

    id: int
    detection_id: int
    created_at: datetime
    severity: Severity
    urgency: Urgency
    qa_score: int
    qa_approved: bool
