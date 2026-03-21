"""YOLO26n inference wrapper.

Handles model loading, inference, and result parsing.
Falls back to empty results if ultralytics is not installed or no model file exists.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detected object."""

    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class Detector:
    """YOLO26n object detector — plain class managed by dependency injection."""

    def __init__(self, model_path: str, input_size: int = 640) -> None:
        self.model_path = model_path
        self.input_size = input_size
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the YOLO model if the file exists."""
        if not Path(self.model_path).exists():
            logger.warning(
                "Model file not found at %s — detector will return empty results. "
                "Train and export your model first.",
                self.model_path,
            )
            return

        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]
            self._model = YOLO(self.model_path)
            logger.info("YOLO model loaded from %s", self.model_path)
        except ImportError:
            logger.warning("ultralytics not installed — detector in stub mode")
        except Exception as e:
            logger.error("Failed to load YOLO model: %s", e)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def detect(self, image: Image.Image) -> list[Detection]:
        """Run inference on a PIL Image and return detections."""
        if self._model is None:
            logger.debug("No model loaded — returning empty detections")
            return []

        results = self._model.predict(
            source=image,
            imgsz=self.input_size,
            conf=0.25,  # low threshold, filtering happens in triage
            verbose=False,
        )

        detections = []
        for result in results:
            for box in result.boxes:
                detections.append(Detection(
                    class_name=result.names[int(box.cls[0])],
                    confidence=float(box.conf[0]),
                    x1=float(box.xyxy[0][0]),
                    y1=float(box.xyxy[0][1]),
                    x2=float(box.xyxy[0][2]),
                    y2=float(box.xyxy[0][3]),
                ))

        logger.info("Detected %d objects", len(detections))
        return detections

    def detect_from_file(self, image_path: str | Path) -> list[Detection]:
        """Run inference on an image file."""
        img = Image.open(image_path).convert("RGB")
        return self.detect(img)
