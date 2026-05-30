"""Pi Camera Module 3 interface.

Uses picamera2 on Raspberry Pi, falls back to a stub on other platforms
for development purposes.
"""

from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.core.detector import Detector

# Defer import check — warning is logged in start(), not at module level.
try:
    from picamera2 import Picamera2  # type: ignore[import-untyped]
    _HAS_PICAMERA = True
except ImportError:
    _HAS_PICAMERA = False

# Color palette for detection classes (binary: defect / healthy).
# Sub-types ('damage', 'blockage') are kept as fallbacks so older DB rows
# still render correctly during the migration period.
_CLASS_COLORS = {
    "defect": (239, 68, 68),     # red
    "healthy": (34, 197, 94),    # green
    "damage": (239, 68, 68),     # legacy
    "blockage": (245, 158, 11),  # legacy
}
_DEFAULT_COLOR = (59, 130, 246)  # blue


class Camera:
    """Wrapper around Pi Camera Module 3 with development fallback."""

    def __init__(self, resolution: tuple[int, int] = (640, 640)) -> None:
        self.resolution = resolution
        self._camera = None

    async def start(self) -> None:
        if _HAS_PICAMERA:
            self._camera = Picamera2()
            config = self._camera.create_still_configuration(
                main={"size": self.resolution, "format": "RGB888"}
            )
            self._camera.configure(config)
            self._camera.start()
            logger.info("Pi Camera started at %s", self.resolution)
        else:
            logger.info("Camera running in stub mode (no picamera2)")

    async def stop(self) -> None:
        if self._camera is not None:
            self._camera.stop()
            self._camera.close()
            self._camera = None
            logger.info("Pi Camera stopped")

    def capture_frame(self) -> Image.Image:
        """Capture a single frame and return as PIL Image."""
        if _HAS_PICAMERA and self._camera is not None:
            array = self._camera.capture_array()
            return Image.fromarray(array)
        return Image.new("RGB", self.resolution, color=(128, 128, 128))

    def capture_to_file(self, path: str | Path) -> Path:
        """Capture a frame and save to disk as JPEG."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        img = self.capture_frame()
        img.save(str(path), "JPEG", quality=85)
        logger.debug("Frame saved to %s", path)
        return path

    def capture_jpeg_bytes(self, detector: Detector | None = None) -> bytes:
        """Capture a frame and return JPEG bytes (for MJPEG streaming).

        If a detector is provided, runs inference and draws bounding boxes.
        """
        img = self.capture_frame()

        if detector is not None and detector.is_loaded:
            detections = detector.detect(img)
            img = self._draw_overlays(img, detections)

        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=70)
        return buf.getvalue()

    @staticmethod
    def _draw_overlays(img: Image.Image, detections) -> Image.Image:
        """Draw bounding boxes and labels on an image."""
        draw = ImageDraw.Draw(img)

        # Try to load a decent font, fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()

        for det in detections:
            color = _CLASS_COLORS.get(det.class_name, _DEFAULT_COLOR)
            # Bounding box
            draw.rectangle(
                [(det.x1, det.y1), (det.x2, det.y2)],
                outline=color,
                width=2,
            )
            # Label background
            label = f"{det.class_name} {det.confidence:.0%}"
            bbox = draw.textbbox((det.x1, det.y1 - 18), label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((det.x1, det.y1 - 18), label, fill=(255, 255, 255), font=font)

        return img

    async def generate_mjpeg_frames(self, detector: Detector | None = None):
        """Async generator yielding MJPEG frames for streaming.

        If detector is provided, each frame includes YOLO bounding box overlays.
        """
        while True:
            frame = self.capture_jpeg_bytes(detector=detector)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            await asyncio.sleep(0.1)  # ~10 FPS

