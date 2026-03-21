"""Tests for app.config."""

from pathlib import Path

from app.config import BASE_DIR, Settings


def test_settings_defaults():
    s = Settings()
    assert s.gemini_api_key == ""
    assert s.confidence_high == 0.70
    assert s.confidence_medium == 0.45
    assert s.capture_interval_minutes == 15
    assert s.yolo_input_size == 640
    assert s.smtp_port == 587


def test_derived_paths():
    s = Settings()
    assert s.data_dir == BASE_DIR / "data"
    assert s.detections_dir == s.data_dir / "detections"
    assert s.reports_dir == s.data_dir / "reports"
    assert s.models_dir == s.data_dir / "models"


def test_ensure_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.BASE_DIR", tmp_path)
    s = Settings()
    s.ensure_dirs()
    assert (tmp_path / "data" / "detections").is_dir()
    assert (tmp_path / "data" / "reports").is_dir()
    assert (tmp_path / "data" / "models").is_dir()
