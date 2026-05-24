"""Camera endpoints — live feed and manual capture."""

import time

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.api.deps import get_camera, get_detector, get_scheduler
from app.core.camera import Camera
from app.core.detector import Detector
from app.core.scheduler import CaptureScheduler

router = APIRouter(prefix="/camera", tags=["camera"])

_last_capture_time: float = 0.0
_CAPTURE_COOLDOWN = 10  # seconds


@router.get("/feed")
async def camera_feed(
    overlay: bool = Query(default=False, description="Enable YOLO overlay on frames"),
    camera: Camera = Depends(get_camera),
    detector: Detector = Depends(get_detector),
):
    """MJPEG live stream from the Pi Camera, optionally with YOLO bounding box overlay."""
    return StreamingResponse(
        camera.generate_mjpeg_frames(detector=detector if overlay else None),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/capture")
async def trigger_capture(
    scheduler: CaptureScheduler = Depends(get_scheduler),
):
    """Trigger an immediate capture-and-process cycle.

    Returns the actual detection results. Rate-limited to once per 10 seconds.
    """
    global _last_capture_time
    now = time.time()

    if now - _last_capture_time < _CAPTURE_COOLDOWN:
        remaining = int(_CAPTURE_COOLDOWN - (now - _last_capture_time))
        raise HTTPException(
            status_code=429,
            detail=f"Rate limited. Try again in {remaining} seconds.",
        )

    _last_capture_time = now
    detections = await scheduler.capture_once()

    return {
        "status": "capture_complete",
        "detections": [
            {
                "class": d.class_name,
                "confidence": round(d.confidence, 4),
                "bbox": {"x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2},
            }
            for d in detections
        ],
        "count": len(detections),
    }

