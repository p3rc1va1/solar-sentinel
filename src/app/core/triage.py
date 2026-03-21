"""Rule-based triage agent — filters detections before LLM pipeline.

No LLM calls. Pure Python logic for:
1. Deduplication (same defect at same location within 60 min)
2. Transient filter (require 2 consecutive detections to confirm)
3. Frame quality check (reject over/underexposed frames)
"""

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
from PIL import Image

from app.core.detector import Detection

logger = logging.getLogger(__name__)


def compute_iou(box_a: dict, box_b: dict) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    x1 = max(box_a["x1"], box_b["x1"])
    y1 = max(box_a["y1"], box_b["y1"])
    x2 = min(box_a["x2"], box_b["x2"])
    y2 = min(box_a["y2"], box_b["y2"])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = (box_a["x2"] - box_a["x1"]) * (box_a["y2"] - box_a["y1"])
    area_b = (box_b["x2"] - box_b["x1"]) * (box_b["y2"] - box_b["y1"])
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def check_frame_quality(image: Image.Image, threshold: float = 0.30) -> bool:
    """Check if the frame is usable (not over/underexposed).

    Returns True if the frame passes quality checks.
    """
    arr = np.array(image.convert("L"))  # grayscale
    total_pixels = arr.size

    overexposed = np.sum(arr > 240) / total_pixels
    if overexposed > threshold:
        logger.debug("Frame rejected: %.0f%% overexposed", overexposed * 100)
        return False

    underexposed = np.sum(arr < 15) / total_pixels
    if underexposed > threshold:
        logger.debug("Frame rejected: %.0f%% underexposed", underexposed * 100)
        return False

    return True


class TriageAgent:
    """Rule-based detection filter to reduce unnecessary LLM calls."""

    def __init__(
        self,
        dedup_window_minutes: int = 60,
        iou_threshold: float = 0.5,
        confirmation_required: int = 2,
    ) -> None:
        self.dedup_window = timedelta(minutes=dedup_window_minutes)
        self.iou_threshold = iou_threshold
        self.confirmation_required = confirmation_required
        self._pending: list[dict] = []

    def filter_detections(
        self,
        detections: list[Detection],
        recent_db_detections: list[dict],
        confidence_medium: float = 0.45,
    ) -> list[Detection]:
        """Filter detections through triage rules.

        Args:
            detections: Raw detections from YOLO.
            recent_db_detections: Detections from the last hour (from DB).
            confidence_medium: Minimum confidence to consider.

        Returns:
            Filtered list of detections that should proceed to CrewAI.
        """
        passed = []

        for det in detections:
            if det.confidence < confidence_medium:
                continue

            if det.class_name == "clean":
                continue

            det_box = {
                "x1": det.x1, "y1": det.y1,
                "x2": det.x2, "y2": det.y2,
            }

            # Dedup against recent DB detections
            if self._is_duplicate(det.class_name, det_box, recent_db_detections):
                continue

            # Confirmation gate
            if not self._check_confirmation(det.class_name, det_box):
                continue

            passed.append(det)

        logger.info(
            "Triage: %d/%d detections passed filter",
            len(passed), len(detections),
        )
        return passed

    def _is_duplicate(
        self, class_name: str, det_box: dict, recent: list[dict]
    ) -> bool:
        """Check if detection is a duplicate of a recent DB entry."""
        for r in recent:
            if r["defect_class"] != class_name:
                continue
            if compute_iou(det_box, r["bbox"]) > self.iou_threshold:
                logger.debug("Triage: %s duplicate, suppressed", class_name)
                return True
        return False

    def _check_confirmation(self, class_name: str, det_box: dict) -> bool:
        """Track pending detections and confirm after N consecutive sightings."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=2)

        # Single-pass: clean old + find matches
        kept = []
        match_count = 0
        for p in self._pending:
            if p["timestamp"] < cutoff:
                continue  # expired
            kept.append(p)
            if (p["class_name"] == class_name
                    and compute_iou(det_box, p["box"]) > self.iou_threshold):
                match_count += 1

        if match_count >= self.confirmation_required - 1:
            # Confirmed — remove matched pending entries
            self._pending = [
                p for p in kept
                if not (p["class_name"] == class_name
                        and compute_iou(det_box, p["box"]) > self.iou_threshold)
            ]
            return True

        # Not yet confirmed — add to pending
        self._pending = kept
        self._pending.append({
            "class_name": class_name,
            "box": det_box,
            "timestamp": now,
        })
        return False
