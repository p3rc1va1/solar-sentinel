"""Pi Camera Module 3 interface.

Uses picamera2 on Raspberry Pi, falls back to a stub on other platforms
for development purposes.
"""

import asyncio
import io
import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# Defer import check — warning is logged in start(), not at module level.
try:
    from picamera2 import Picamera2  # type: ignore[import-untyped]
    _HAS_PICAMERA = True
except ImportError:
    _HAS_PICAMERA = False


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

    def capture_jpeg_bytes(self) -> bytes:
        """Capture a frame and return JPEG bytes (for MJPEG streaming)."""
        img = self.capture_frame()
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=70)
        return buf.getvalue()

    async def generate_mjpeg_frames(self):
        """Async generator yielding MJPEG frames for streaming."""
        while True:
            frame = self.capture_jpeg_bytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            await asyncio.sleep(0.1)  # ~10 FPS
