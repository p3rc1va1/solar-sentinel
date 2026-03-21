"""Tests for app.core.detector."""

import pytest
from PIL import Image

from app.core.detector import Detection, Detector


class TestDetector:
    def test_no_model_returns_empty(self):
        det = Detector(model_path="/nonexistent.pt")
        assert not det.is_loaded
        result = det.detect(Image.new("RGB", (64, 64)))
        assert result == []

    def test_is_loaded_false_when_no_model(self):
        det = Detector(model_path="/nonexistent.pt")
        assert det.is_loaded is False

    def test_detect_from_file(self, tmp_path):
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        path = tmp_path / "test.jpg"
        img.save(str(path))

        det = Detector(model_path="/nonexistent.pt")
        result = det.detect_from_file(path)
        assert result == []


class TestDetectionDataclass:
    def test_creation(self):
        d = Detection(
            class_name="crack",
            confidence=0.95,
            x1=10.0, y1=20.0, x2=100.0, y2=200.0,
        )
        assert d.class_name == "crack"
        assert d.confidence == 0.95
