"""Solar Sentinel — FastAPI application entry point."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.deps import init_deps
from app.api.routes import camera, detections, health, reports, settings
from app.config import Settings
from app.core.camera import Camera
from app.core.demo import populate_demo_data
from app.core.detector import Detection, Detector
from app.core.scheduler import CaptureScheduler
from app.core.sensor import DHTSensor
from app.core.triage import TriageAgent
from app.db.database import Database
from app.services.gemini import GeminiClient
from app.services.notifications import NotificationService
from app.services.weather import WeatherService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


def _build_high_detection_callback(
    db: Database,
    crew_module,
    gemini: GeminiClient,
    weather: WeatherService,
    notif: NotificationService,
    sensor: DHTSensor,
    settings: Settings,
):
    """Build the async callback for HIGH confidence detections.

    Pipeline: weather context → historical context → CrewAI analysis → save report → notify.
    """

    async def on_high_detection(det: Detection, image_path: str) -> None:
        try:
            # 1. Environmental context
            weather_data = await weather.get_current_weather()
            weather_summary = (
                f"{weather_data['summary']}, {weather_data['temperature']}, "
                f"precipitation: {weather_data['precipitation']}, UV: {weather_data['uv_index']}"
            )

            # Sensor data
            sensor_reading = sensor.read()
            if sensor_reading:
                temperature = (
                    f"{sensor_reading['temperature']:.1f}°C (sensor), "
                    f"humidity: {sensor_reading['humidity']:.1f}%"
                )
            else:
                temperature = weather_data["temperature"]

            # 2. Historical context
            recent_reports = await db.get_reports_since(days=7)
            if recent_reports:
                history_lines = [
                    f"[{r['created_at']}] {r['severity']} — {r['root_cause'][:100]}"
                    for r in recent_reports[:10]
                ]
                historical_context = "\n".join(history_lines)
            else:
                historical_context = "No previous reports in the last 7 days."

            # 3. CrewAI analysis
            crew = crew_module.SolarSentinelCrew(gemini)
            result = await crew.analyze_detection(
                defect_class=det.class_name,
                confidence=det.confidence,
                bbox={"x1": det.x1, "y1": det.y1, "x2": det.x2, "y2": det.y2},
                panel_id="panel-1",
                image_path=image_path,
                weather_summary=weather_summary,
                temperature=temperature,
                historical_context=historical_context,
                latitude=settings.weather_latitude,
                longitude=settings.weather_longitude,
                tz_name=settings.weather_timezone,
            )

            # 4. Save report to DB
            # Find the detection ID (most recent with matching image path)
            recent = await db.get_recent_detections(hours=1)
            det_id = None
            for r in recent:
                if r.get("image_path") == image_path and r.get("defect_class") == det.class_name:
                    det_id = r["id"]
                    break

            if det_id:
                await db.insert_report(
                    detection_id=det_id,
                    severity=result["severity"],
                    urgency=result["urgency"],
                    root_cause=result["root_cause"],
                    trend_analysis=result["trend_analysis"],
                    report_markdown=result["report_markdown"],
                    qa_score=result["qa_score"],
                    qa_approved=result["qa_approved"],
                    defect_subtype=result.get("defect_subtype"),
                    analyzer_output_json=result.get("analyzer_output_json"),
                    planner_output_json=result.get("planner_output_json"),
                )

                # Track Gemini usage
                await db.log_gemini_usage(
                    model_name=gemini.ranked_models[0].name if gemini.ranked_models else "unknown",
                    tokens_used=0,  # CrewAI doesn't expose token count
                    success=True,
                )

            # 5. Send notifications
            if result.get("qa_approved", False):
                await notif.send_report(
                    report_markdown=result["report_markdown"],
                    severity=result["severity"],
                    image_path=image_path,
                )
                logger.info("HIGH detection pipeline complete — report sent")
            else:
                logger.info("HIGH detection pipeline complete — QA rejected (score: %s)", result.get("qa_score"))

        except Exception as e:
            logger.error("High detection pipeline error: %s", e, exc_info=True)

    return on_high_detection


def _build_medium_detection_callback(db: Database):
    """Build the async callback for MEDIUM confidence detections (logged only)."""

    async def on_medium_detection(det: Detection, image_path: str) -> None:
        logger.info(
            "MEDIUM detection logged: %s (%.2f) at %s",
            det.class_name, det.confidence, image_path,
        )

    return on_medium_detection


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and tear down application resources."""
    logger.info("Solar Sentinel starting up...")

    s = Settings()
    s.ensure_dirs()

    # Database
    db_file = "solar_sentinel_demo.db" if s.demo_mode else "solar_sentinel.db"
    db = Database(s.data_dir / db_file)
    await db.connect()
    
    if s.demo_mode:
        await populate_demo_data(db)

    # Camera
    cam = Camera(resolution=(s.yolo_input_size, s.yolo_input_size))
    await cam.start()

    # Detector (plain class, no singleton)
    detector = Detector(str(BASE_DIR / s.yolo_model_path), s.yolo_input_size)

    # Triage
    triage = TriageAgent()

    # Gemini
    gemini = GeminiClient(api_key=s.gemini_api_key)
    if s.gemini_api_key:
        gemini.configure()

    # Notifications
    notif = NotificationService(
        email_enabled=s.email_enabled,
        email_address=s.email_address,
        smtp_host=s.smtp_host,
        smtp_port=s.smtp_port,
        smtp_username=s.smtp_username,
        smtp_password=s.smtp_password,
        telegram_enabled=s.telegram_enabled,
        telegram_bot_token=s.telegram_bot_token,
        telegram_chat_id=s.telegram_chat_id,
    )

    # Weather
    weather = WeatherService(latitude=s.weather_latitude, longitude=s.weather_longitude)
    await weather.start()

    # DHT22 Sensor
    sensor = DHTSensor()
    sensor.start()

    # Import crew module (deferred to avoid circular imports)
    from app.agents import crew as crew_module

    # Build pipeline callbacks
    on_high = _build_high_detection_callback(db, crew_module, gemini, weather, notif, sensor, s)
    on_medium = _build_medium_detection_callback(db)

    # Scheduler
    scheduler = CaptureScheduler(
        camera=cam,
        detector=detector,
        triage=triage,
        db=db,
        settings=s,
        on_high_detection=on_high,
        on_medium_detection=on_medium,
    )

    # Register all deps via DI
    init_deps(db, s, cam, detector, triage, gemini, notif, weather, scheduler, sensor)

    # Start scheduler (skip in demo mode — no real camera)
    if not s.demo_mode:
        await scheduler.start()
        logger.info("Capture scheduler started")

    logger.info("Solar Sentinel ready")
    yield

    # Shutdown
    logger.info("Solar Sentinel shutting down...")
    if scheduler.is_running:
        await scheduler.stop()
    await weather.stop()
    sensor.stop()
    await cam.stop()
    await db.disconnect()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Solar Sentinel",
    description="Autonomous solar panel defect detection and classification",
    version="0.1.0",
    lifespan=lifespan,
)

# Routes
app.include_router(health.router)
app.include_router(detections.router)
app.include_router(camera.router)
app.include_router(reports.router)
app.include_router(settings.router)

# Import and register new routes
from app.api.routes import images, sensor as sensor_route
app.include_router(images.router)
app.include_router(sensor_route.router)

# Static UI files
ui_dir = BASE_DIR / "ui"
if ui_dir.exists():
    app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")

if __name__ == "__main__":
    import argparse
    import os
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Solar Sentinel")
    parser.add_argument("--demo", action="store_true", help="Run with fake demo data in a separate database")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    args = parser.parse_args()
    
    if args.demo:
        os.environ["DEMO_MODE"] = "1"
        
    uvicorn.run("app.main:app", host=args.host, port=args.port)

