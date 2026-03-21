"""Report endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_db
from app.db.database import Database

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/context/history")
async def get_historical_context(
    days: int = Query(default=7, le=30),
    db: Database = Depends(get_db),
):
    """Get report history for LLM context enrichment."""
    reports = await db.get_reports_since(days=days)
    context_lines = [
        f"[{r['created_at']}] {r['severity']} — {r['root_cause'][:100]}"
        for r in reports
    ]
    return {
        "days": days,
        "report_count": len(reports),
        "context": "\n".join(context_lines) if context_lines else "No reports in this period.",
    }


@router.get("")
async def list_reports(
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    db: Database = Depends(get_db),
):
    """List generated reports with pagination."""
    reports = await db.list_reports(limit=limit, offset=offset)
    return {"reports": reports, "count": len(reports)}


@router.get("/{report_id}")
async def get_report(report_id: int, db: Database = Depends(get_db)):
    """Get a single report by ID."""
    report = await db.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return report
