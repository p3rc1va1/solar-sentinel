<div align="center">

# ☀️ Solar Sentinel

**Autonomous Solar Panel Defect Detection & Classification**

*Computer Vision + Agentic AI running entirely on a Raspberry Pi 5*

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![YOLO26](https://img.shields.io/badge/Model-YOLO26n-00FFFF?logo=ultralytics&logoColor=white)](https://docs.ultralytics.com/models/yolo26/)
[![CrewAI](https://img.shields.io/badge/Agents-CrewAI-FF6B6B)](https://crewai.com)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

*A bachelor's thesis project that deploys a fine-tuned YOLO26 Nano object detection model alongside a multi-agent LLM pipeline on edge hardware to autonomously monitor solar panels, detect defects, and generate actionable maintenance reports — no cloud required.*

</div>

---

## Overview

Solar Sentinel is an end-to-end autonomous maintenance system for solar panels. It captures images using a camera mounted on a Raspberry Pi 5, runs real-time defect detection using a custom-trained YOLO26 Nano model, and — when a defect is found — triggers a multi-agent CrewAI pipeline powered by Google Gemini to analyze the defect, write a professional maintenance report, and send notifications via email or Telegram.

**Everything runs locally on the Pi.** The only external calls are to the Gemini API (for LLM reasoning) and Open-Meteo (for weather context). No images leave the device.

### Key Features

- **Binary Detection** — `defect` · `healthy` — sub-typing (cracks, soiling, debris, snow, hot spot…) is delegated to the agentic VLM layer
- **Agentic Analysis** — 4-agent CrewAI pipeline: Defect Analyst (multimodal) → Maintenance Planner → Critic / QA Reviewer → Report Writer
- **MCP Tool Integration** — FastMCP server over stdio; the Maintenance Planner and the Critic / QA Reviewer share `web_search`, `current_time`, and `weather_forecast` tools
- **Environmental Context** — Temperature/humidity sensor (DHT22) + Open-Meteo weather API enrich reports, and DHT22 thresholds (>35 °C, <0 °C, >85 % RH) trigger captures via the sensor watcher
- **Multi-Channel Alerts** — Email (SMTP) and Telegram notifications with attached images, gated on QA approval
- **Adaptive Scheduling** — Capture frequency adapts to detection results (5 / 15 / 30 min bands)
- **Daylight-Aware** — Real per-site sunrise/sunset using a NOAA solar position formula (no external API)
- **Smart Triage** — Rule-based filtering (deduplication, transient rejection, exposure check) before any LLM call
- **Daily MEDIUM Digest** — Sub-threshold detections summarised once per day and delivered via email/Telegram
- **Web Dashboard** — FastAPI backend with REST API and static UI (Dashboard, Detections, Reports, Live Feed with ROI mask overlay, Settings)

---

## System Architecture

The system follows a multi-stage pipeline architecture. The diagram below shows the complete algorithmic flow from image capture to notification delivery:

<div align="center">

![Algorithmic Flow](docs/Algorithmic%20flow.drawio.png)

</div>

---

## How It Works

A step-by-step walkthrough of the algorithmic flow:

### Step 1 - Trigger

The system can be triggered in three ways:

| Trigger | Description |
|:--------|:------------|
| **Periodic** | Adaptive scheduler captures frames every *N* minutes (default: 15 min, adjusts based on results) |
| **Manual** | User triggers a capture via the web UI or REST API |
| **Sensor** | DHT22 thresholds wired through `core/sensor_watcher.py`: temp >35 °C (PV efficiency drop), temp <0 °C (icing risk), humidity >85 % (particulate cementation). Per-channel cooldown prevents flapping. |

### Step 2 - Image Capture

The **Camera Module 3 Wide** captures a still frame at 640×640 resolution. Before processing, a **frame quality check** runs — frames that are >30% overexposed or underexposed are rejected immediately to avoid false positives from glare or darkness.

### Step 3 - YOLO26 Nano Inference

The captured frame is passed to the **YOLO26 Nano** model (exported to NCNN format for ARM optimization). The model acts as a **binary smart trigger** — classifying frames as `defect` or `healthy` — because defect sub-type analysis is delegated to the agentic pipeline's VLM, which provides better semantic understanding. The model outputs bounding boxes with confidence scores:

| Class | What It Detects | Action |
|:------|:----------------|:-------|
| `defect` | Any anomaly: cracks, burns, soiling, debris, snow | Routes to confidence-based pipeline |
| `healthy` | Clean panel surface — no anomalies | No action, logs clean frame |

> **Note:** The training pipeline (`src/notebooks/train_yolo26n.ipynb`, `src/notebooks/train_local.py`) trains the model with `nc=1, names=['defect']`. The `SUBTYPE_REMAP` in the local trainer collapses every source label (damage, blockage, soiling…) onto the single `defect` class.

### Step 4 - Confidence-Based Routing

The detection confidence score determines what happens next:

```
Detection Confidence
        │
        ├── ≥ 70%  ──→  HIGH: Trigger CrewAI pipeline immediately
        │                 ↳ Increase capture frequency to every 5 min
        │
        ├── 45-70% ──→  MEDIUM: Log to database; rolled into a daily
        │                 digest (`core/digest.py`) summarised by Gemini
        │                 and dispatched once per day at the configured
        │                 local time.
        │
        └── < 45%  ──→  LOW: Log to database only, no action
                         ↳ If 6+ consecutive clean frames, reduce
                           capture frequency to every 30 min
```

### Step 5 - Triage Agent (Rule-Based)

Before any LLM call, a **rule-based triage agent** filters detections:

1. **Deduplication** — Suppresses same-class detections with IoU > 0.5 within the last 60 minutes
2. **Transient Filter** — Requires 2 consecutive detections at the same location to confirm (prevents one-off false positives)
3. **`healthy` Rejection** — Clean panels never trigger the LLM pipeline

### Step 6 - Environmental Context

When a detection passes triage, the system enriches it with context:

- **Weather** — Current conditions via [Open-Meteo](https://open-meteo.com/) (temperature, precipitation, UV index)
- **Temperature & Humidity** — Local sensor data from the Adafruit AM2302/DHT22 (reads in background; included in report context)
- **Historical** — Previous detections from the SQLite database for trend analysis

### Step 7 - CrewAI Agentic Pipeline

The enriched detection triggers a **sequential 4-agent pipeline** powered by Google Gemini:

```
┌──────────────────────────────────────────────────────────────────┐
│                       CrewAI Pipeline                             │
│                                                                   │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌─────────┐ │
│  │   Defect   │   │ Maintenance│   │   Critic   │   │ Report  │ │
│  │   Analyst  │──→│   Planner  │──→│   Agent    │──→│ Writer  │ │
│  │ (VLM image)│   │  (MCP ✔)   │   │  (MCP ✔)   │   │         │ │
│  └────────────┘   └────────────┘   └────────────┘   └─────────┘ │
│                                                                   │
│  FastMCP server (`src/app/agents/mcp/server.py`, stdio):         │
│  ├── web_search(query, max_results)                              │
│  ├── current_time(latitude, longitude, tz_name)                  │
│  └── weather_forecast(latitude, longitude, hours)                │
│                                                                   │
│  Per the thesis spec, only the Maintenance Planner and the       │
│  Critic / QA Reviewer have MCP tool access. Notification is      │
│  gated on `qa_approved` from the Critic.                          │
└──────────────────────────────────────────────────────────────────┘
```

| Agent | Role | What It Does |
|:------|:-----|:-------------|
| **Defect Analyst** | PV systems engineer | Multimodal — uses CrewAI `AddImageTool` to inspect the image, confirms the YOLO trigger isn't a false positive, classifies sub-type and severity |
| **Maintenance Planner** | Field operations | Cross-references findings with the weather forecast, daylight window, and (if needed) web search to recommend context-aware actions |
| **Critic Agent** | Quality assurance | Spot-checks planner claims with the same MCP tools when a number or URL looks inconsistent; scores the report and gates dispatch |
| **Report Writer** | Technical writer | Compiles the analyst and planner outputs into a user-friendly email/Telegram report |

### Step 8 - Notification Delivery

The approved report is delivered through all enabled channels:

| Channel | Protocol | Details |
|:--------|:---------|:--------|
| **Email** | SMTP | HTML-formatted report with attached detection image |
| **Telegram** | Bot API | Markdown report + photo sent to configured chat |

### Step 9 - Logging & Adaptive Sleep

Every detection (whether it triggers the pipeline or not) is logged to an **SQLite database** with:
- Image path, defect class, confidence, bounding box coordinates
- Timestamp, panel ID, analysis results (if applicable)

The scheduler then adapts its capture interval based on recent history and enters a sleep cycle until the next scheduled capture.

---

## Hardware

<div align="center">

![Enclosure Exploded View](casing/case-exploded.png)

*Custom 3D-printed weather-resistant enclosure — designed in [KCL](https://zoo.dev/kcl) and exported to STEP*

</div>

### Components

| Component | Model | Purpose |
|:----------|:------|:--------|
| **SBC** | Raspberry Pi 5 (8GB) | Main compute — runs YOLO + CrewAI |
| **Cooling** | Raspberry Pi Active Cooler | Thermal management for sustained inference |
| **Camera** | Camera Module 3 Wide | 120° FOV, 12MP, auto-focus |
| **Sensor** | Adafruit AM2302 (DHT22) | Ambient temperature & humidity |
| **Enclosure** | Custom 3D-printed (ASA) | Weather-resistant outdoor housing (ASA chosen for 105°C glass transition temp and UV resistance) |

### Enclosure Specs & Features

- **Material:** ASA (Acrylonitrile Styrene Acrylate) — UV-resistant, 105°C glass transition temperature
- **Dimensions:** 179 mm × 56 mm × 133 mm · **Weight:** ~290 g total assembly
- **Wall thickness:** 3.2 mm · **Lid overhang:** 8 mm rain guard
- Tightly fitted camera lens port and DHT22 sensor mount
- Ventilation slots: 30×4 mm rear panel + 80×4 mm bottom panel (natural convection)
- Gasket groove with snapping-arm lid seal; 15.5 mm cable gland for power
- M2.5 Pan Head (ISO 7045) standoffs for Pi 5 mounting

---

## Tech Stack

| Layer | Technology | Role |
|:------|:-----------|:-----|
| **Computer Vision** | [YOLO26 Nano](https://docs.ultralytics.com/models/yolo26/) + [NCNN](https://github.com/Tencent/ncnn) | Object detection on ARM CPU |
| **Agentic AI** | [CrewAI](https://crewai.com) + [Google Gemini](https://ai.google.dev/) | Multi-agent defect analysis pipeline |
| **Backend** | [FastAPI](https://fastapi.tiangolo.com) + [Uvicorn](https://www.uvicorn.org/) | REST API and web server |
| **Database** | [SQLite](https://sqlite.org/) via [aiosqlite](https://github.com/omnilib/aiosqlite) | Local async persistence |
| **Notifications** | [aiosmtplib](https://aiosmtplib.readthedocs.io/) + [python-telegram-bot](https://python-telegram-bot.org/) | Email and Telegram delivery |
| **Weather** | [Open-Meteo API](https://open-meteo.com/) | Environmental context enrichment |
| **Training** | [Google Colab](https://colab.research.google.com/) + [Ultralytics](https://ultralytics.com/) | Cloud GPU training, NCNN export |
| **Hardware Design** | [KCL](https://zoo.dev/kcl) | Parametric 3D enclosure modeling |
| **Package Manager** | [uv](https://docs.astral.sh/uv/) | Fast Python dependency management |

---

## Project Structure

```
solar-sentinel/
├── casing/                          # 3D-printed enclosure
│   ├── main.kcl                     # Parametric KCL model
│   ├── main.step                    # STEP export for printing/SolidWorks
│   └── case-exploded.png            # Exploded view render
│
├── docs/                            # Documentation
│   └── Algorithmic flow.drawio.png  # System architecture diagram
│
├── src/                             # Application source
│   ├── app/
│   │   ├── main.py                  # FastAPI entry point + lifespan + HIGH detection callback
│   │   ├── config.py                # Pydantic settings (from .env)
│   │   │
│   │   ├── core/                    # Core pipeline
│   │   │   ├── camera.py            # Pi Camera / stub adapter + MJPEG stream
│   │   │   ├── detector.py          # YOLO26n inference wrapper
│   │   │   ├── triage.py            # Rule-based filter (quality, dedup, confirmation)
│   │   │   ├── scheduler.py         # Daylight-aware adaptive capture scheduler
│   │   │   ├── solar.py             # NOAA solar position formula → real sunrise/sunset
│   │   │   ├── sensor.py            # DHT22 temperature/humidity sensor + stub
│   │   │   ├── sensor_watcher.py    # Polls DHT22 and triggers capture on threshold
│   │   │   ├── digest.py            # Daily MEDIUM-band digest summariser + dispatch
│   │   │   └── demo.py              # Demo mode — populates DB with fake data
│   │   │
│   │   ├── agents/                  # CrewAI agentic layer
│   │   │   ├── crew.py              # 4-agent crew orchestration with MCP fan-out
│   │   │   ├── model_router.py      # Gemini model discovery + ranking
│   │   │   ├── mcp/
│   │   │   │   ├── server.py        # FastMCP server (stdio) launched by the crew
│   │   │   │   └── tools/           # web_search, current_time, weather_forecast
│   │   │   └── config/
│   │   │       ├── agents.yaml      # Agent role / goal / backstory definitions
│   │   │       └── tasks.yaml       # Task prompts with {placeholder} format strings
│   │   │
│   │   ├── services/                # External integrations
│   │   │   ├── gemini.py            # Google Gemini client + model fallback
│   │   │   ├── notifications.py     # Email (SMTP) + Telegram delivery
│   │   │   └── weather.py           # Open-Meteo weather service
│   │   │
│   │   ├── api/                     # REST API
│   │   │   ├── deps.py              # Dependency injection (singleton registry)
│   │   │   └── routes/
│   │   │       ├── health.py        # GET /health — system stats + Gemini usage
│   │   │       ├── camera.py        # GET /camera/feed (MJPEG), POST /camera/capture
│   │   │       ├── detections.py    # GET /detections, GET /detections/{id}
│   │   │       ├── reports.py       # GET /reports, GET /reports/{id}
│   │   │       ├── images.py        # GET /images/{filename} — serve detection images
│   │   │       ├── sensor.py        # GET /sensor — live DHT22 reading
│   │   │       └── settings.py      # GET/PUT /settings — runtime config persisted to DB
│   │   │
│   │   ├── db/                      # Database layer
│   │   │   └── database.py          # Async SQLite (aiosqlite) — all CRUD operations
│   │   │
│   │   └── models/                  # Pydantic schemas
│   │       ├── detection.py         # DetectionRecord, BoundingBox, enums
│   │       ├── report.py            # Report models
│   │       └── settings.py          # AllSettings, NotificationSettings, etc.
│   │
│   ├── ui/                          # Web dashboard (no build step — plain HTML/JS)
│   │   ├── index.html               # SPA shell — Dashboard, Detections, Reports, Live, Settings
│   │   ├── app.js                   # All dashboard JS (fetch, charts, pagination)
│   │   └── style.css                # Styles
│   │
│   ├── notebooks/
│   │   └── train_yolo26n.ipynb      # Colab training notebook
│   │
│   ├── tests/                       # Test suite
│   ├── pyproject.toml               # Dependencies & tool config
│   └── .env                         # Environment variables (not committed)
│
├── LICENSE                          # MIT License
└── README.md                        # ← You are here
```

---

## Getting Started

### Prerequisites

- **Raspberry Pi 5** (8GB recommended) with Raspberry Pi OS (64-bit)
- **Camera Module 3 Wide** connected via ribbon cable
- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** package manager

### 1. Clone & Install

```bash
git clone https://github.com/p3rc1va1/solar-sentinel.git
cd solar-sentinel/src

# Install dependencies with uv
uv sync
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# Required — Gemini API key for CrewAI agents
GEMINI_API_KEY=your-gemini-api-key

# Optional — Email notifications
EMAIL_ENABLED=true
EMAIL_ADDRESS=you@example.com
SMTP_USERNAME=you@gmail.com
SMTP_PASSWORD=your-app-password

# Optional — Telegram notifications
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_CHAT_ID=your-chat-id

# Optional — Weather context (get coords from Google Maps)
WEATHER_LATITUDE=54.6872
WEATHER_LONGITUDE=25.2797

# Model path (default works after training)
YOLO_MODEL_PATH=data/models/best.pt
```

### 3. Train the Model

Open `src/notebooks/train_yolo26n.ipynb` in [Google Colab](https://colab.research.google.com/) (or run `src/notebooks/train_local.py` on a Mac/Linux box with MPS/CUDA/CPU autodetect). The pipeline:

1. Downloads 6 solar panel defect datasets from Roboflow (`dataset_3` is held out as an OOD test set)
2. Merges and remaps every source label onto the binary `defect` class via `SUBTYPE_REMAP`
3. Fine-tunes YOLO26 Nano (`nc=1, names=['defect']`) and writes a model card to `runs/binary-trigger/`
4. Exports to NCNN format for Pi 5

Copy the trained model to your Pi:
```bash
scp best.pt pi@<pi-ip>:~/solar-sentinel/src/data/models/
```

### 4. Run

To run the application normally:
```bash
cd src
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

To run the application in **Demo Mode** (which populates a separate database with fake detections and AI reports so you can explore the UI without needing a live camera or model):
```bash
cd src
uv run python -m app.main --demo --host 0.0.0.0 --port 8000
```

The web dashboard is available at `http://<pi-ip>:8000` on the local network. For remote access from any network, the thesis recommends adding **[Tailscale](https://tailscale.com/)** — no port-forwarding required, everything stays local. Key API endpoints:

| Endpoint | Method | Description |
|:---------|:-------|:------------|
| `/health` | GET | System stats (CPU temp, memory, disk) + Gemini usage |
| `/camera/feed` | GET | MJPEG live stream (`?overlay=true` for YOLO boxes) |
| `/camera/capture` | POST | Trigger immediate capture + detection (rate-limited 10 s) |
| `/detections` | GET | Detection history with pagination |
| `/reports` | GET | Generated maintenance reports |
| `/sensor` | GET | Live DHT22 temperature & humidity reading |
| `/settings` | GET/PUT | Runtime configuration (persisted to DB) |
| `/images/{filename}` | GET | Serve a detection image by filename |

---

## Detection Classes

The model is **binary** — YOLO acts only as a trigger, not a classifier. The agentic pipeline (multimodal Defect Analyst) determines the defect sub-type semantically.

| Class | Triggers Pipeline |
|:------|:------------------|
| **defect** | Yes — routed through confidence bands (HIGH → CrewAI immediately; MEDIUM → daily digest; LOW → log only) |
| **healthy** | No — clean frames are logged but never escalated |

> **Sub-typing happens at the agentic layer.** The Defect Analyst picks one of `physical_damage`, `soiling`, `snow_or_ice`, `debris`, `hot_spot`, `discoloration`, `other` and writes it to `reports.defect_subtype`.

---

## Status

### Implemented

| Component | Notes |
|:----------|:------|
| Binary YOLO26n training pipeline | Colab notebook + local trainer; `nc=1, names=['defect']`; `SUBTYPE_REMAP` collapses all source labels; 6 Roboflow datasets with `dataset_3` held out as OOD |
| YOLO26n inference | ultralytics wrapper; stub when model/library absent |
| Rule-based triage agent | Frame quality check, IoU deduplication, 2-hit confirmation |
| Adaptive capture scheduler | Real per-site sunrise/sunset (NOAA formula in `core/solar.py`); polar-day/night sentinels; 5 / 15 / 30 min adaptive bands |
| Sensor-triggered capture | `core/sensor_watcher.py` polls DHT22 every 60 s; thresholds >35 °C / <0 °C / >85 % RH each have an independent cooldown |
| 4-agent CrewAI pipeline | Defect Analyst (multimodal) → Maintenance Planner → Critic / QA Reviewer → Report Writer; sequential, Google Gemini |
| FastMCP tool server | `app/agents/mcp/server.py` runs over stdio; exposes `web_search`, `current_time`, `weather_forecast`. Per the thesis, only the Maintenance Planner and the Critic / QA Reviewer have tool access |
| Multimodal analyzer | Defect Analyst is `multimodal=True` and uses the CrewAI `AddImageTool` to attach the panel image |
| MEDIUM detection digest | `core/digest.py` summarises sub-threshold detections once per day at the configured local time and dispatches via email + Telegram |
| QA-gated dispatch | The HIGH-detection callback only sends notifications when `qa_approved` is true |
| Real token counting | Pulled from `crew.usage_metrics.total_tokens` and `google.genai usage_metadata`, written to `gemini_usage` |
| Idempotent additive migrations | `database.py:_migrate` adds `defect_subtype`, `analyzer_output_json`, `planner_output_json` to older installs |
| Gemini model auto-discovery | Dynamic API query + fallback ranked list (`pro=5, flash=3, flash-lite=1`) |
| Weather context | Open-Meteo current + forecast, WMO code lookup, injected into CrewAI context |
| Geocoding city picker | Open-Meteo geocoding behind `/geocode`, with debounced UI search in Settings |
| DHT22 sensor | `adafruit_dht` + stub; live reading exposed at `/sensor` |
| Email notifications | HTML report + image attachment via SMTP/aiosmtplib |
| Telegram notifications | Markdown report + photo via `python-telegram-bot` |
| FastAPI REST API | All routes: health, camera, detections, reports, images, sensor, geocode, settings |
| MJPEG live stream | `/camera/feed` — optional YOLO bounding box overlay |
| Web dashboard UI | 5 pages: Dashboard, Detections, Reports, Live Feed (with **interactive ROI mask overlay**), Settings |
| Demo mode | Separate DB (`solar_sentinel_demo.db`), seeded with fake detections + reports |
| Runtime settings | Persisted to SQLite, mirrored to live `Settings` singleton, applied to notification + scheduler + sensor watcher + digest |
| 3D-printed enclosure | KCL parametric model + STEP export |

### Not Yet Implemented

| Feature | Thesis Reference | Notes |
|:--------|:----------------|:------|
| Server-side ROI privacy masking before inference | §4.3, p.51 | The live UI now lets the user draw the 4 panel corners and visually masks the feed (persisted in `localStorage`). The thesis-grade pipeline mask — applying the same polygon to frames *before* they reach the YOLO detector — is still pending. |
| Quantized / NCNN inference path on the Pi | §2.2, pp.25–26, 31 | Training notebook exports NCNN, but `core/detector.py` loads the unquantized `best.pt` via plain Ultralytics. |
| §1.3 acceptance criteria — automated verification | §1.3, p.3 | No timing harness for the ≤60 s end-to-end dispatch target, the ≥1 mm-feature-at-1 m optical resolution, or the mAP ≥85 % gate. The local trainer writes a model card but no test asserts the threshold. |
| Custom anodized-aluminum mount bracket | Drawing Sheet 6/7 | Sheet 6 specifies the bracket (Ra 12.5, ISO 22768-mK, fillets and hole pattern); no part exists in `casing/`. |
| Tool-access exclusion for Defect Analyst & Report Writer | §3.3 | Currently enforced in code: `crew.py` only fans `tools=mcp_tools` into `TOOL_USING_AGENTS = {planner, qa}`. If the design ever extends, see `tests/test_crew.py::test_only_planner_and_qa_have_tools`. |

---

## Economics (from Thesis)

| Metric | Value |
|:-------|:------|
| **Bill of Materials** | €216 |
| **Total manufacturing cost / unit** | €301 (BOM + 3D printing + assembly labour) |
| **Suggested retail price** | €450 |
| **Commercial break-even** | ~41 units/year |
| **Annual OPEX** | €19.53 (fan/SD card amortisation + one service visit every 3 years) |
| **Annual energy savings** (5 kWp system) | ~€110 (500 kWh recovered at €0.22/kWh) |
| **ROI** | 19.05% |
| **Payback period** | 5.25 years |

---

## License

This project is licensed under the [MIT License](LICENSE).

