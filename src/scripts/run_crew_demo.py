"""End-to-end CrewAI demo: pick a labeled positive, run the full agent pipeline.

Run from `src/`:

    uv run python -m scripts.run_crew_demo

Picks the first labeled positive from `datasets/solar-sentinel/images/val/`
(deterministic — sorted filename), reads its YOLO label to derive a pixel
bounding box, and feeds it to `SolarSentinelCrew.analyze_detection(...)`.
The script does NOT re-run YOLO — the goal is to exercise the four-agent
pipeline (Analyzer → Maintenance Planner → Report Writer → QA Reviewer),
not the detector. The "confidence" we report is a plausible HIGH-gate score.

Why val and not test? Test is `dataset_3` held entirely out from training
(cross-source OOD); the model is much weaker there by design. Val is
representative of what the production trigger sees on similar imagery,
which is what we want for a demo of the agent pipeline.

Tracing:
    Sets `CREWAI_TRACING_ENABLED=true` before constructing the crew.
    CrewAI 1.14 prints a "🔗 View here: <url>" line when the trace batch
    finalises; if you've never run `crewai login`, the URL is *ephemeral*
    (anyone with the link + access code can view) — perfect for a demo.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from time import perf_counter

# Tracing must be enabled BEFORE crewai imports the listener.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "true")

from PIL import Image

# Silence CrewAI's post-run "[y/N] view your traces?" prompt — it blocks for
# 20s and on timeout flips tracing OFF for the next run, which is the opposite
# of what a demo wants. The trace URL still gets printed.
from crewai.events.listeners.tracing.utils import set_suppress_tracing_messages

set_suppress_tracing_messages(True)

from app.agents.crew import SolarSentinelCrew
from app.config import Settings
from app.services.gemini import GeminiClient

SRC_DIR = Path(__file__).resolve().parent.parent
SPLIT = "val"
IMAGES_DIR = SRC_DIR / "datasets" / "solar-sentinel" / "images" / SPLIT
LABELS_DIR = SRC_DIR / "datasets" / "solar-sentinel" / "labels" / SPLIT


def find_first_labeled_positive() -> tuple[Path, Path]:
    """Return the (image, label) pair for the first non-empty label file."""
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"{SPLIT} images dir missing: {IMAGES_DIR}")
    for img_path in sorted(IMAGES_DIR.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl_path = LABELS_DIR / f"{img_path.stem}.txt"
        if lbl_path.exists() and lbl_path.read_text().strip():
            return img_path, lbl_path
    raise RuntimeError(f"No labeled positive found in the {SPLIT} split.")


def yolo_label_to_pixel_bbox(label_path: Path, img_size: tuple[int, int]) -> dict:
    """Convert the first row of a YOLO label file to a pixel-space bbox dict."""
    width, height = img_size
    first_row = label_path.read_text().strip().splitlines()[0]
    parts = first_row.split()
    # YOLO format: class cx cy w h (all normalised 0..1)
    _cls, cx, cy, w, h = parts[0], *map(float, parts[1:5])
    x1 = (cx - w / 2) * width
    y1 = (cy - h / 2) * height
    x2 = (cx + w / 2) * width
    y2 = (cy + h / 2) * height
    return {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    settings = Settings()
    if not settings.gemini_api_key:
        print(
            "ERROR: GEMINI_API_KEY is not set in the environment.\n"
            "       Add it to src/.env (the FastAPI app reads from the same place)."
        )
        return 2

    print(f"Tracing enabled: CREWAI_TRACING_ENABLED={os.environ['CREWAI_TRACING_ENABLED']}")
    print("(If you've never run `crewai login`, the trace will be ephemeral —")
    print(" CrewAI prints an access-code-protected URL anyone with the link can view.)")
    print()

    image_path, label_path = find_first_labeled_positive()
    img = Image.open(image_path)
    bbox = yolo_label_to_pixel_bbox(label_path, img.size)

    print(f"Selected image : {image_path.relative_to(SRC_DIR)}")
    print(f"Image size     : {img.size[0]}×{img.size[1]}")
    print(f"Pixel bbox     : {bbox}")
    print()

    gemini = GeminiClient(api_key=settings.gemini_api_key)
    gemini.configure()
    if not gemini.ranked_models:
        print("ERROR: GeminiClient could not discover any models — check your API key.")
        return 3
    print(f"Gemini ready   : top model = {gemini.ranked_models[0].name}")
    print()

    crew = SolarSentinelCrew(gemini)
    print("Running CrewAI pipeline (Analyzer → Planner → Writer → QA)...")
    t0 = perf_counter()
    result = await crew.analyze_detection(
        defect_class="defect",
        confidence=0.92,                     # plausible HIGH-gate score
        bbox=bbox,
        panel_id="DEMO-PANEL-01",
        image_path=str(image_path),
        weather_summary="Clear skies, 24°C, light wind from NW.",
        temperature="24°C",
        historical_context="No previous reports in the last 7 days.",
        latitude="41.0082",                  # Istanbul
        longitude="28.9784",
        tz_name="Europe/Istanbul",
    )
    elapsed = perf_counter() - t0
    print(f"\nPipeline complete in {elapsed:.1f}s\n")

    # Pretty-print the structured dict, but elide the long markdown report
    # so the terminal output stays readable.
    pretty = dict(result)
    if pretty.get("report_markdown"):
        body = pretty["report_markdown"]
        pretty["report_markdown"] = (
            body if len(body) < 600 else body[:600] + f"\n…[+{len(body) - 600} chars]"
        )
    print("=" * 60)
    print("Structured result")
    print("=" * 60)
    print(json.dumps(pretty, indent=2))
    print()
    print("Look above for a CrewAI '🔗 View here:' URL — that's your trace.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
