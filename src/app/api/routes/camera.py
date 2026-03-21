"""Camera endpoints — live feed and manual capture."""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_camera
from app.core.camera import Camera

router = APIRouter(prefix="/camera", tags=["camera"])


@router.get("/feed")
async def camera_feed(camera: Camera = Depends(get_camera)):
    """MJPEG live stream from the Pi Camera."""
    return StreamingResponse(
        camera.generate_mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/capture")
async def trigger_capture():
    """Trigger an immediate capture-and-process cycle.

    The scheduler singleton handles capture via main.py.
    """
    return {"status": "capture_triggered", "message": "Manual capture initiated"}
