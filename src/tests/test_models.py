"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from app.models.detection import (
    BoundingBox,
    DefectClass,
    DetectionCreate,
    DetectionRecord,
    DetectionSummary,
    Severity,
    Urgency,
)
from app.models.report import ReportCreate, ReportRecord, ReportSummary
from app.models.settings import AllSettings, DetectionSettings, GeminiSettings, NotificationSettings


class TestEnums:
    def test_severity_values(self):
        assert Severity.CRITICAL == "CRITICAL"
        assert Severity.WARNING == "WARNING"
        assert Severity.INFO == "INFO"

    def test_urgency_values(self):
        assert Urgency.IMMEDIATE == "IMMEDIATE"
        assert Urgency.WITHIN_1_WEEK == "WITHIN_1_WEEK"
        assert Urgency.ROUTINE == "ROUTINE"

    def test_defect_class_values(self):
        assert DefectClass.CLEAN == "clean"
        assert DefectClass.SOILING == "soiling"


class TestDetectionModels:
    def test_bounding_box(self):
        bb = BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=0.95, class_name="crack")
        assert bb.x1 == 0
        assert bb.confidence == 0.95

    def test_detection_create(self):
        dc = DetectionCreate(
            image_path="/tmp/img.jpg",
            defect_class=DefectClass.SOILING,
            confidence=0.85,
            bbox=BoundingBox(x1=0, y1=0, x2=50, y2=50, confidence=0.85, class_name="soiling"),
        )
        assert dc.panel_id == "panel-1"

    def test_detection_create_invalid_confidence(self):
        with pytest.raises(ValidationError):
            DetectionCreate(
                image_path="/tmp/img.jpg",
                defect_class=DefectClass.SOILING,
                confidence=1.5,
                bbox=BoundingBox(x1=0, y1=0, x2=50, y2=50, confidence=0.85, class_name="soiling"),
            )


class TestReportModels:
    def test_report_create(self):
        rc = ReportCreate(
            detection_id=1,
            severity=Severity.WARNING,
            urgency=Urgency.ROUTINE,
            root_cause="Dirt buildup",
            trend_analysis="Stable",
            report_markdown="# Report",
            qa_score=8,
            qa_approved=True,
        )
        assert rc.qa_score == 8

    def test_report_create_invalid_score(self):
        with pytest.raises(ValidationError):
            ReportCreate(
                detection_id=1,
                severity=Severity.WARNING,
                urgency=Urgency.ROUTINE,
                root_cause="Test",
                trend_analysis="Test",
                report_markdown="# R",
                qa_score=15,
                qa_approved=False,
            )


class TestSettingsModels:
    def test_all_settings_defaults(self):
        s = AllSettings()
        assert s.notifications.email_enabled is False
        assert s.detection.confidence_high == 0.70
        assert s.gemini.gemini_api_key == ""

    def test_notification_settings(self):
        ns = NotificationSettings(email_enabled=True, email_address="a@b.com")
        assert ns.email_enabled
        assert ns.smtp_host == "smtp.gmail.com"

    def test_detection_settings(self):
        ds = DetectionSettings(confidence_high=0.80)
        assert ds.confidence_high == 0.80
        assert ds.capture_interval_minutes == 15
