"""Detection history endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_db
from app.db.database import Database

router = APIRouter(prefix="/detections", tags=["detections"])


@router.get("")
async def list_detections(
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    db: Database = Depends(get_db),
):
    """List recent detections with pagination."""
    detections = await db.list_detections(limit=limit, offset=offset)
    return {"detections": detections, "count": len(detections)}


@router.get("/{detection_id}")
async def get_detection(detection_id: int, db: Database = Depends(get_db)):
    """Get a single detection by ID."""
    detection = await db.get_detection(detection_id)
    if detection is None:
        raise HTTPException(status_code=404, detail="Detection not found")
    return detection
