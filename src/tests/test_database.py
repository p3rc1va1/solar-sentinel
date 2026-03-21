"""Tests for app.db.database."""

import pytest
import pytest_asyncio

from app.db.database import Database


@pytest_asyncio.fixture
async def database(tmp_path):
    db = Database(tmp_path / "test.db")
    await db.connect()
    yield db
    await db.disconnect()


class TestConnection:
    @pytest.mark.asyncio
    async def test_connect_creates_db_file(self, tmp_path):
        db_path = tmp_path / "sub" / "test.db"
        db = Database(db_path)
        await db.connect()
        assert db_path.exists()
        await db.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_sets_none(self, tmp_path):
        db = Database(tmp_path / "test.db")
        await db.connect()
        await db.disconnect()
        assert db._db is None

    @pytest.mark.asyncio
    async def test_db_property_raises_when_not_connected(self, tmp_path):
        db = Database(tmp_path / "test.db")
        with pytest.raises(RuntimeError, match="not connected"):
            _ = db.db


class TestDetections:
    @pytest.mark.asyncio
    async def test_insert_and_get(self, database):
        det_id = await database.insert_detection(
            image_path="/tmp/img.jpg",
            defect_class="soiling",
            confidence=0.85,
            bbox={"x1": 10, "y1": 20, "x2": 100, "y2": 200},
        )
        assert det_id == 1

        det = await database.get_detection(det_id)
        assert det is not None
        assert det["defect_class"] == "soiling"
        assert det["confidence"] == 0.85
        assert det["bbox"] == {"x1": 10, "y1": 20, "x2": 100, "y2": 200}

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, database):
        assert await database.get_detection(999) is None

    @pytest.mark.asyncio
    async def test_list_detections(self, database):
        for i in range(5):
            await database.insert_detection(
                image_path=f"/tmp/img{i}.jpg",
                defect_class="crack",
                confidence=0.90,
                bbox={"x1": 0, "y1": 0, "x2": 50, "y2": 50},
            )
        results = await database.list_detections(limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_list_detections_offset(self, database):
        for i in range(5):
            await database.insert_detection(
                image_path=f"/tmp/img{i}.jpg",
                defect_class="crack",
                confidence=0.90,
                bbox={"x1": 0, "y1": 0, "x2": 50, "y2": 50},
            )
        results = await database.list_detections(limit=10, offset=3)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_get_recent_detections(self, database):
        await database.insert_detection(
            image_path="/tmp/img.jpg",
            defect_class="soiling",
            confidence=0.70,
            bbox={"x1": 0, "y1": 0, "x2": 50, "y2": 50},
        )
        results = await database.get_recent_detections(hours=24)
        assert len(results) >= 1  # just inserted


class TestReports:
    @pytest.mark.asyncio
    async def test_insert_and_get_report(self, database):
        det_id = await database.insert_detection(
            image_path="/tmp/x.jpg", defect_class="crack",
            confidence=0.9, bbox={"x1": 0, "y1": 0, "x2": 1, "y2": 1},
        )
        report_id = await database.insert_report(
            detection_id=det_id,
            severity="CRITICAL",
            urgency="IMMEDIATE",
            root_cause="Physical damage",
            trend_analysis="N/A",
            report_markdown="# Report",
            qa_score=8,
            qa_approved=True,
        )
        assert report_id == 1

        report = await database.get_report(report_id)
        assert report is not None
        assert report["severity"] == "CRITICAL"
        assert report["qa_approved"] == 1

        # Verify detection links back
        det = await database.get_detection(det_id)
        assert det["report_id"] == report_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_report(self, database):
        assert await database.get_report(999) is None

    @pytest.mark.asyncio
    async def test_list_reports(self, database):
        det_id = await database.insert_detection(
            image_path="/tmp/x.jpg", defect_class="crack",
            confidence=0.9, bbox={"x1": 0, "y1": 0, "x2": 1, "y2": 1},
        )
        await database.insert_report(
            detection_id=det_id, severity="WARNING", urgency="ROUTINE",
            root_cause="Dirt", trend_analysis="Stable",
            report_markdown="# R", qa_score=6, qa_approved=False,
        )
        results = await database.list_reports()
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_reports_since(self, database):
        det_id = await database.insert_detection(
            image_path="/tmp/x.jpg", defect_class="crack",
            confidence=0.9, bbox={"x1": 0, "y1": 0, "x2": 1, "y2": 1},
        )
        await database.insert_report(
            detection_id=det_id, severity="INFO", urgency="ROUTINE",
            root_cause="None", trend_analysis="Clear",
            report_markdown="# Clean", qa_score=10, qa_approved=True,
        )
        results = await database.get_reports_since(days=7)
        assert len(results) >= 1


class TestSettings:
    @pytest.mark.asyncio
    async def test_set_and_get_setting(self, database):
        await database.set_setting("theme", "dark")
        val = await database.get_setting("theme")
        assert val == "dark"

    @pytest.mark.asyncio
    async def test_get_nonexistent_setting(self, database):
        assert await database.get_setting("nonexistent") is None

    @pytest.mark.asyncio
    async def test_upsert_setting(self, database):
        await database.set_setting("key1", "v1")
        await database.set_setting("key1", "v2")
        assert await database.get_setting("key1") == "v2"


class TestGeminiUsage:
    @pytest.mark.asyncio
    async def test_log_and_query(self, database):
        await database.log_gemini_usage("gemini-pro", 100, True)
        await database.log_gemini_usage("gemini-pro", 50, True)
        await database.log_gemini_usage("gemini-flash", 30, False)

        usage = await database.get_gemini_usage_today()
        assert len(usage) >= 1
