"""Pydantic models for detection records."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class Urgency(str, Enum):
    IMMEDIATE = "IMMEDIATE"
    WITHIN_1_WEEK = "WITHIN_1_WEEK"
    ROUTINE = "ROUTINE"


class DefectClass(str, Enum):
    PHYSICAL_DAMAGE = "physical_damage"
    SOILING = "soiling"
    BIOLOGICAL = "biological"
    ENVIRONMENTAL = "environmental"
    ELECTRICAL = "electrical"
    CLEAN = "clean"


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str


class DetectionCreate(BaseModel):
    """Data captured at inference time."""

    image_path: str
    defect_class: DefectClass
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    panel_id: str = "panel-1"


class DetectionRecord(DetectionCreate):
    """Full detection record as persisted in the database."""

    id: int
    timestamp: datetime
    triaged: bool = False
    report_id: int | None = None

    class Config:
        from_attributes = True


class DetectionSummary(BaseModel):
    """Lightweight detection info for list responses."""

    id: int
    timestamp: datetime
    defect_class: DefectClass
    confidence: float
    severity: Severity | None = None
    has_report: bool = False
