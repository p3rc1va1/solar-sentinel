"""Tests for app.core.triage."""

import pytest
import numpy as np
from PIL import Image

from app.core.detector import Detection
from app.core.triage import TriageAgent, check_frame_quality, compute_iou


class TestComputeIoU:
    def test_identical_boxes(self):
        box = {"x1": 0, "y1": 0, "x2": 100, "y2": 100}
        assert compute_iou(box, box) == 1.0

    def test_no_overlap(self):
        a = {"x1": 0, "y1": 0, "x2": 50, "y2": 50}
        b = {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
        assert compute_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = {"x1": 0, "y1": 0, "x2": 100, "y2": 100}
        b = {"x1": 50, "y1": 50, "x2": 150, "y2": 150}
        iou = compute_iou(a, b)
        assert 0.0 < iou < 1.0

    def test_zero_area_box(self):
        a = {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
        b = {"x1": 0, "y1": 0, "x2": 100, "y2": 100}
        assert compute_iou(a, b) == 0.0


class TestCheckFrameQuality:
    def test_normal_image_passes(self):
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        assert check_frame_quality(img) is True

    def test_overexposed_fails(self):
        arr = np.full((64, 64, 3), 250, dtype=np.uint8)
        img = Image.fromarray(arr)
        assert check_frame_quality(img) is False

    def test_underexposed_fails(self):
        arr = np.full((64, 64, 3), 5, dtype=np.uint8)
        img = Image.fromarray(arr)
        assert check_frame_quality(img) is False

    def test_custom_threshold(self):
        arr = np.full((64, 64, 3), 250, dtype=np.uint8)
        img = Image.fromarray(arr)
        assert check_frame_quality(img, threshold=1.0) is True


class TestTriageAgent:
    def _make_detection(self, cls="crack", conf=0.8, x1=10, y1=10, x2=100, y2=100):
        return Detection(class_name=cls, confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2)

    def test_below_confidence_filtered(self, triage):
        dets = [self._make_detection(conf=0.30)]
        result = triage.filter_detections(dets, [], confidence_medium=0.45)
        assert result == []

    def test_clean_class_filtered(self, triage):
        dets = [self._make_detection(cls="clean", conf=0.90)]
        result = triage.filter_detections(dets, [], confidence_medium=0.45)
        assert result == []

    def test_duplicate_suppressed(self, triage):
        dets = [self._make_detection()]
        recent = [{"defect_class": "crack", "bbox": {"x1": 10, "y1": 10, "x2": 100, "y2": 100}}]
        result = triage.filter_detections(dets, recent)
        assert result == []

    def test_confirmation_gate(self, triage):
        det = self._make_detection()
        # First sighting — should NOT pass (needs 2)
        result1 = triage.filter_detections([det], [])
        assert result1 == []
        # Second sighting — should pass
        result2 = triage.filter_detections([det], [])
        assert len(result2) == 1

    def test_different_class_not_duplicate(self, triage):
        dets = [self._make_detection(cls="soiling")]
        recent = [{"defect_class": "crack", "bbox": {"x1": 10, "y1": 10, "x2": 100, "y2": 100}}]
        # Won't match dedup but needs confirmation
        result = triage.filter_detections(dets, recent)
        assert result == []  # first sighting, pending confirmation
