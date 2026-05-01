"""Module for generating demo data for the UI dashboard."""

import json
import logging
import random
from datetime import datetime, timedelta, timezone

from app.db.database import Database

logger = logging.getLogger(__name__)


async def populate_demo_data(db: Database) -> None:
    """Populate the database with fake demo data for the dashboard UI."""
    
    # Check if we already have detections
    existing = await db.list_detections(limit=1)
    if existing:
        logger.info("Demo data already exists. Skipping population.")
        return
        
    logger.info("Injecting demo data into database...")
    
    now = datetime.now(timezone.utc)
    
    # 1. Fake Gemini usage
    for _ in range(5):
        await db.log_gemini_usage("gemini-2.5-flash", random.randint(800, 1500), True)
    
    # 2. Fake detections & reports
    demos = [
        {
            "class": "damage",
            "conf": 0.92,
            "bbox": {"x": 100, "y": 150, "w": 200, "h": 200},
            "panel": "demo-panel-4",
            "offset_min": 10,
            "report": {
                "sev": "CRITICAL",
                "urg": "High",
                "cause": "Physical impact (likely hail or debris)",
                "trend": "New damage detected. No previous history.",
                "md": "## Damage Report\n\n**Severity:** CRITICAL\n\nSevere micro-cracking and shattered glass observed on the lower quadrant of Panel 4. Immediate replacement is recommended to prevent further power loss and electrical hazard.\n\n**Action:** Dispatch technician immediately.",
                "score": 9,
            }
        },
        {
            "class": "blockage",
            "conf": 0.78,
            "bbox": {"x": 300, "y": 50, "w": 100, "h": 100},
            "panel": "demo-panel-2",
            "offset_min": 45,
            "report": {
                "sev": "WARNING",
                "urg": "Medium",
                "cause": "Bird droppings / Soiling",
                "trend": "Recurring soiling pattern on edge panels.",
                "md": "## Soiling Report\n\n**Severity:** WARNING\n\nSignificant localized soiling detected, likely bird droppings. This blockage is causing localized shading and decreasing module efficiency.\n\n**Action:** Schedule localized cleaning during the next routine maintenance cycle.",
                "score": 8,
            }
        },
        {
            "class": "healthy",
            "conf": 0.98,
            "bbox": {},
            "panel": "demo-panel-1",
            "offset_min": 60,
            "report": None
        },
        {
            "class": "damage",
            "conf": 0.55,
            "bbox": {"x": 500, "y": 400, "w": 30, "h": 10},
            "panel": "demo-panel-3",
            "offset_min": 120,
            "report": {
                "sev": "INFO",
                "urg": "Low",
                "cause": "Snail trails / minor delamination",
                "trend": "Slow degradation over last 3 months.",
                "md": "## Minor Defect Report\n\n**Severity:** INFO\n\nMinor snail trails and possible early-stage delamination observed near the busbars. Currently not affecting structural integrity or significant output.\n\n**Action:** Continue monitoring. No immediate action required.",
                "score": 7,
            }
        },
        {
            "class": "healthy",
            "conf": 0.95,
            "bbox": {},
            "panel": "demo-panel-2",
            "offset_min": 180,
            "report": None
        }
    ]
    
    for d in demos:
        timestamp = (now - timedelta(minutes=d["offset_min"])).isoformat()
        
        # Insert detection directly with custom timestamp
        cursor = await db.db.execute(
            "INSERT INTO detections (timestamp, image_path, defect_class, confidence, bbox_json, panel_id, triaged) VALUES (?, ?, ?, ?, ?, ?, 1)",
            (timestamp, "data/demo/placeholder.jpg", d["class"], d["conf"], json.dumps(d["bbox"]), d["panel"])
        )
        await db.db.commit()
        det_id = cursor.lastrowid
        
        # Insert report if needed
        r = d.get("report")
        if r:
            await db.insert_report(
                detection_id=det_id,
                severity=r["sev"],
                urgency=r["urg"],
                root_cause=r["cause"],
                trend_analysis=r["trend"],
                report_markdown=r["md"],
                qa_score=r["score"],
                qa_approved=True
            )
            # update report created_at
            await db.db.execute(
                "UPDATE reports SET created_at = ? WHERE detection_id = ?",
                (timestamp, det_id)
            )
            await db.db.commit()

    logger.info("Demo data population complete.")
