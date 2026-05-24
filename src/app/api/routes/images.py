"""Image serving endpoint — serves detection images from data/detections/."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import BASE_DIR

router = APIRouter(prefix="/images", tags=["images"])

_DETECTIONS_DIR = BASE_DIR / "data" / "detections"


@router.get("/{filename}")
async def serve_image(filename: str):
    """Serve a detection image by filename.

    Includes path traversal protection.
    """
    # Sanitize: only allow simple filenames (no directory traversal)
    safe_name = Path(filename).name
    if safe_name != filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = _DETECTIONS_DIR / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    # Determine media type
    suffix = file_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(str(file_path), media_type=media_type)
