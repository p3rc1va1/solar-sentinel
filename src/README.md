# Solar Sentinel

> Autonomous solar panel defect detection and classification using computer vision and NLP.

**Thesis**: *Integration of Computer Vision and Natural Language Processing for Autonomous Solar Panel Defect Detection and Classification*

## Architecture

- **Edge device**: Raspberry Pi 5 (8GB) + Pi Camera Module 3 Wide
- **Detection**: YOLO26n (NCNN, INT8, 640×640) — ~7-8 FPS on CPU
- **Analysis**: CrewAI pipeline (Triage → Analyzer → Writer → QA) using Gemini API
- **Notifications**: Email (SMTP) + Telegram bot — user-configurable
- **Dashboard**: Lightweight static web UI served by FastAPI
- **Security**: Tailscale VPN (zero-touch onboarding with pre-auth keys)
- **Deployment**: Systemd service on Raspberry Pi OS

## Quick Start (Development)

```bash
# Clone and setup
cd solar-sentinel
uv sync

# Configure
cp .env.example .env
# Edit .env with your Gemini API key and notification settings

# Run
uv run uvicorn app.main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) for the dashboard.

## Quick Start (Raspberry Pi)

```bash
# On the Pi
cd /opt/solar-sentinel
uv sync
uv pip install picamera2

# Configure
cp .env.example .env
nano .env

# Deploy as service
sudo cp deploy/solar-sentinel.service /etc/systemd/system/
sudo systemctl enable solar-sentinel
sudo systemctl start solar-sentinel
```

## Project Structure

```
solar-sentinel/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Settings from .env
│   ├── api/routes/          # REST endpoints
│   ├── core/                # Camera, detector, triage, scheduler
│   ├── agents/              # CrewAI crew + YAML configs
│   ├── services/            # Gemini, notifications, weather
│   ├── models/              # Pydantic schemas
│   └── db/                  # SQLite database
├── ui/                      # Static web dashboard
├── data/                    # Detections, reports, YOLO weights
├── deploy/                  # Systemd + first-boot scripts
└── pyproject.toml
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | System health + Gemini usage |
| GET | `/detections` | Detection history |
| GET | `/detections/{id}` | Single detection |
| GET | `/reports` | Report history |
| GET | `/reports/{id}` | Single report |
| GET | `/reports/context/history` | 7-day context for LLM |
| GET | `/camera/feed` | MJPEG live stream |
| POST | `/camera/capture` | Trigger manual capture |
| GET | `/settings` | Current settings |
| PUT | `/settings` | Update settings |

## License

MIT
