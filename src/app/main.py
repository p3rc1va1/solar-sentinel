"""Solar Sentinel — FastAPI application entry point."""

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
from app.core.detector import Detector
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

    # Register all deps via DI
    init_deps(db, s, cam, detector, triage, gemini, notif, weather)

    logger.info("Solar Sentinel ready")
    yield

    # Shutdown
    logger.info("Solar Sentinel shutting down...")
    await weather.stop()
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

