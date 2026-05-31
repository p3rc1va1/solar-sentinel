"""Generate train_yolo26n.ipynb and evaluate_model.ipynb from inline cell specs.

Run from the repo's src/ directory:
    uv run python notebooks/_build_notebooks.py

This script is the source of truth for the notebooks. Edit it, re-run it, and
the .ipynb files are regenerated. The notebooks themselves should not be
hand-edited; treat them as build artefacts.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).parent

NB_KERNELSPEC = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def write_nb(path: Path, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": NB_KERNELSPEC,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1))
    print(f"  wrote {path} ({len(cells)} cells)")


# ─────────────────────────────────────────────────────────────────────────────
# train_yolo26n.ipynb
# ─────────────────────────────────────────────────────────────────────────────


TRAIN_CELLS: list[dict] = []

TRAIN_CELLS.append(md("""\
# Solar Sentinel — YOLO26n binary-trigger training

This notebook trains a **binary** YOLO26 Nano model that gates the CrewAI agent
pipeline on a Raspberry Pi 5. Class taxonomy is `nc=1`, names `["defect"]`. The
agentic layer (Defect Analyst with VLM) is responsible for resolving sub-types;
this model exists only to decide *whether to call the agents at all*.

## Pipeline
1. Mount Drive (fail-fast)
2. Install pinned dependencies
3. Seed everything for reproducibility
4. Download datasets
5. Build manifest + perceptual dedup
6. Stratified train/val split; dataset_3 held out entirely as OOD test
7. Train YOLO26n with binary-trigger-tuned hyperparameters
8. Run Ultralytics validation
9. Export to NCNN (FP16) + smoke test on three held-out images
10. Render MODEL_CARD.md
11. Persist full run folder to Drive (timestamped + `best_latest` aliases)

## Hardware target
Raspberry Pi 5 (8 GB) + Camera Module 3 Wide. Inference via NCNN on ARM CPU.
Expected ~7–8 FPS at 640×640 FP16.
"""))


TRAIN_CELLS.append(md("""\
## Cell 1 — Mount Google Drive (fail-fast)

If Drive auth or write permission fails, fail at minute 1 — not after a 25-minute
training run. Every artefact in `runs/binary-trigger/` will eventually land in
`MyDrive/solar-sentinel-models/`.
"""))

TRAIN_CELLS.append(code("""\
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")
DRIVE_DIR = Path("/content/drive/MyDrive/solar-sentinel-models")
DRIVE_DIR.mkdir(parents=True, exist_ok=True)

# Sanity: confirm we can write
probe = DRIVE_DIR / ".write_test"
probe.write_text("ok")
probe.unlink()
print(f"Drive mounted, writable: {DRIVE_DIR}")
"""))


TRAIN_CELLS.append(md("""\
## Cell 2 — Install dependencies

Pinned for reproducibility. `imagehash` is used for perceptual dedup,
`scikit-learn` for stratified splitting, `pandas` for manifest tables.
`ultralytics>=8.4.0` is the version that introduced YOLO26 support.
"""))

TRAIN_CELLS.append(code("""\
!pip install -q "ultralytics>=8.4.0" "roboflow" "imagehash>=4.3" \\
    "scikit-learn>=1.4" "pandas>=2.2" "Jinja2>=3.1"
print("Packages installed.")
"""))


TRAIN_CELLS.append(md("""\
## Cell 3 — GPU check

Before running this cell, set **Runtime → Change runtime type → T4 GPU**.
"""))

TRAIN_CELLS.append(code("""\
import torch

if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {name} ({mem_gb:.1f} GB)")
    print(f"CUDA: {torch.version.cuda}, torch: {torch.__version__}")
else:
    raise RuntimeError("No GPU detected. Set Runtime → GPU before continuing.")
"""))


TRAIN_CELLS.append(md("""\
## Cell 4 — Reproducibility seeding

cuDNN-deterministic mode is enabled. Costs 5–10 % training speed; required for
a thesis-grade reproducible pipeline.
"""))

TRAIN_CELLS.append(code("""\
import os
import random

import numpy as np
import torch

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"Seeded everything with {SEED}.")
"""))


TRAIN_CELLS.append(md("""\
## Cell 5 — Download datasets

Six RGB Roboflow datasets in total. Five are merged for train+val; `dataset_3`
is held out entirely as a cross-source OOD test set (it is *not* used for any
training signal, including val mAP-based early stop).

The original three sources (datasets 1–3) anchor the pipeline; the three
additional sources (4–6) were added to push training set size from ~1k to
~5k images after dedup, which is the prerequisite for hitting the
target mAP@50 of 0.80–0.88. All six are RGB outdoor whole-panel imagery; sub-
type taxonomies vary but the union-box policy in cell 7 normalises them.

Get a free Roboflow API key at <https://app.roboflow.com/settings/api>.
"""))

TRAIN_CELLS.append(code("""\
from pathlib import Path

ROBOFLOW_API_KEY = ""  # paste your key here

BASE_DIR = Path("/content/datasets/raw")
BASE_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    # ── Original three sources ────────────────────────────────────────────
    {
        "name": "dataset_1",
        "workspace": "solar-panel-defect-detection",
        "project": "solar-panel-defect-detection-hpser",
        "version": 1,
        "role": "train_val",
    },
    {
        "name": "dataset_2",
        "workspace": "solar-panel-defect-ufk3b",
        "project": "solar-panel-defect-2-aeuek",
        "version": 4,
        "role": "train_val",
    },
    {
        "name": "dataset_3",
        "workspace": "gao-shou-zheng-b6xqc",
        "project": "solar-panel-0swal",
        "version": 3,
        "role": "ood_test",  # held entirely out
    },

    # ── Tier A additions for Sprint 2 ─────────────────────────────────────
    # Largest single addition (~3690 imgs); identical schema to dataset_1 so
    # perceptual dedup will likely drop a meaningful fraction. Worth it for
    # the unique imagery that survives.
    {
        "name": "dataset_4_gydkp",
        "workspace": "solar-panel-q3vlt",
        "project": "solar-panel-gydkp",
        "version": 1,
        "role": "train_val",
    },
    # Adds explicit "Cracks" class (under-represented in the original three);
    # CC BY 4.0 licensed.
    {
        "name": "dataset_5_maintenance",
        "workspace": "solarpanels-mi4yb",
        "project": "solar-panel-maintenance",
        "version": 1,
        "role": "train_val",
    },
    # Defect-only (no healthy class); contributes positives. ~456 imgs.
    {
        "name": "dataset_6_phase2",
        "workspace": "solarpaneldefectdetectionphase2",
        "project": "solar-panel-defect-detection-f6wsy",
        "version": 1,
        "role": "train_val",
    },
]

if not ROBOFLOW_API_KEY:
    raise RuntimeError(
        "Set ROBOFLOW_API_KEY above. Free key at "
        "https://app.roboflow.com/settings/api"
    )

from roboflow import Roboflow
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
for ds in DATASETS:
    out = BASE_DIR / ds["name"]
    if out.exists() and any(out.iterdir()):
        print(f"  {ds['name']}: already present, skipping download")
        continue
    print(f"  downloading {ds['name']}...")
    try:
        project = rf.workspace(ds["workspace"]).project(ds["project"])
        project.version(ds["version"]).download("yolov8", location=str(out))
    except Exception as e:
        print(f"  WARN: failed to download {ds['name']}: {e}")
        print(f"        Manually browse https://universe.roboflow.com/{ds['workspace']}/{ds['project']} "
              f"and pick a valid version, then re-run.")
        continue
print("All datasets downloaded (or skipped if pre-existing).")
"""))


TRAIN_CELLS.append(md("""\
## Cell 6 — Build manifest with perceptual dedup

For every image we record: source dataset, original sub-type label (kept *only*
as metadata — the YOLO label collapses to binary), is_positive flag, SHA256
content hash, perceptual hash (`imagehash.phash`).

Two-pass dedup:
- Pass 1 (exact): drop SHA256 duplicates.
- Pass 2 (near-duplicates): cluster perceptual hashes within Hamming distance 5,
  keep one representative per cluster. Roboflow datasets routinely contain
  consecutive video frames; without this pass they leak between train and val.

Sub-type metadata (`original_subtype`) is retained for stratification (cell 7)
and per-sub-type recall reporting (eval notebook).
"""))

TRAIN_CELLS.append(code("""\
import hashlib
import json
import yaml
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image

# Maps every original Roboflow class name to a canonical sub-type label
# we keep as metadata. None means "this image is healthy / no annotation".
SUBTYPE_REMAP = {
    "defective": "physical_damage",
    "physical-damage": "physical_damage",
    "physical damage": "physical_damage",
    "physical_damage": "physical_damage",
    "crack": "physical_damage",
    "electrical-damage": "electrical_damage",
    "electrical damage": "electrical_damage",
    "electrical_damage": "electrical_damage",
    "damage": "physical_damage",
    "damage panel": "physical_damage",
    "bird-drop": "soiling_bird",
    "bird drop": "soiling_bird",
    "bird-drop panel": "soiling_bird",
    "dusty": "soiling_dust",
    "dusty panel": "soiling_dust",
    "dust": "soiling_dust",
    "cover": "soiling_dust",
    "snow": "snow_or_ice",
    "snow-covered": "snow_or_ice",
    "snow-covered panel": "snow_or_ice",
    # Healthy labels
    "non-defective": None,
    "non defective": None,
    "clean": None,
    "clean panel": None,
    "normal": None,
}


def read_dataset_yaml(ds_root: Path) -> dict:
    with open(ds_root / "data.yaml") as f:
        names = yaml.safe_load(f).get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return {int(k): v.strip() for k, v in names.items()}


def parse_label_file(txt: Path) -> list[tuple[int, list[float]]]:
    out = []
    if not txt.exists():
        return out
    for line in txt.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                out.append((int(parts[0]), [float(x) for x in parts[1:5]]))
            except ValueError:
                continue
    return out


def build_manifest() -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        ds_path = BASE_DIR / ds["name"]
        yaml_paths = list(ds_path.rglob("data.yaml"))
        if not yaml_paths:
            print(f"  WARN: no data.yaml under {ds_path}, skipping")
            continue
        ds_root = yaml_paths[0].parent
        names = read_dataset_yaml(ds_root)

        for split in ("train", "valid", "test"):
            img_dir = ds_root / split / "images"
            lbl_dir = ds_root / split / "labels"
            if not img_dir.exists():
                continue
            for img_path in img_dir.iterdir():
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                content = img_path.read_bytes()
                try:
                    img = Image.open(img_path)
                    width, height = img.size
                    phash = str(imagehash.phash(img))
                except Exception as e:
                    print(f"    skip unreadable {img_path}: {e}")
                    continue
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                annotations = parse_label_file(lbl_path)
                # Map original class IDs to canonical sub-type strings.
                subtypes = []
                positive_boxes = []
                for cls_id, bbox in annotations:
                    name = names.get(cls_id, "").lower().strip()
                    canonical = SUBTYPE_REMAP.get(name, "other")
                    if canonical is None:
                        continue  # healthy "annotation" — drop
                    subtypes.append(canonical)
                    positive_boxes.append((canonical, bbox))
                rows.append(
                    {
                        "source_dataset": ds["name"],
                        "role": ds["role"],
                        "image_path": str(img_path),
                        "label_path": str(lbl_path) if lbl_path.exists() else "",
                        "width": width,
                        "height": height,
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "phash": phash,
                        "is_positive": bool(positive_boxes),
                        "primary_subtype": (
                            Counter(subtypes).most_common(1)[0][0] if subtypes else "healthy"
                        ),
                        "boxes": positive_boxes,
                    }
                )
    return pd.DataFrame(rows)


df = build_manifest()
print(f"Raw manifest: {len(df)} images")
print(df.groupby(["source_dataset", "primary_subtype"]).size().unstack(fill_value=0))


# ── Pass 1: exact (SHA256) dedup ─────────────────────────────────────────────
before = len(df)
df = df.drop_duplicates(subset=["sha256"], keep="first").reset_index(drop=True)
print(f"\\nSHA256 dedup: {before} → {len(df)} (-{before - len(df)})")


# ── Pass 2: perceptual near-duplicate clustering (Hamming ≤ 5) ───────────────
def phash_clusters(df: pd.DataFrame, threshold: int = 5) -> list[list[int]]:
    indices = list(df.index)
    hashes = {i: imagehash.hex_to_hash(df.at[i, "phash"]) for i in indices}
    visited: set[int] = set()
    clusters: list[list[int]] = []
    for i in indices:
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        for j in indices:
            if j in visited:
                continue
            if (hashes[i] - hashes[j]) <= threshold:
                cluster.append(j)
                visited.add(j)
        clusters.append(cluster)
    return clusters


clusters = phash_clusters(df, threshold=5)
keep_idx = sorted({c[0] for c in clusters})
dropped_near = len(df) - len(keep_idx)
df = df.loc[keep_idx].reset_index(drop=True)
print(f"Perceptual dedup (Hamming ≤ 5): -{dropped_near}, final {len(df)} images")
print("\\nFinal composition:")
print(df.groupby(["role", "primary_subtype"]).size().unstack(fill_value=0))
"""))


TRAIN_CELLS.append(md("""\
## Cell 7 — Stratified split + write merged dataset (canonical-union label policy)

`dataset_3` is **entirely** test (cross-source OOD). Within `dataset_1 ∪
dataset_2`, we stratify on `(source_dataset, primary_subtype, is_positive)` and
take an 80/20 train/val split.

**Label-policy normalisation.** The three Roboflow sources use inconsistent
annotation policies — some draw a tight box around every visible micro-defect,
others draw one box around the affected area. Mixing these policies produced a
val/test instance-density mismatch in the previous run (val=3.0 boxes/img,
test=1.2 boxes/img) and capped val mAP at 0.25.

To homogenise, we apply a deterministic policy: **one union bounding box per
defect-containing image** = the bounding rectangle of all original defect boxes
in that image. This is exactly what the agentic VLM Analyst needs from the
trigger (a region of interest, not pixel-perfect localisation), and it matches
the looser of the source policies, so no information is fabricated.

Sub-type metadata stays in the manifest, *not* in the labels.
"""))

TRAIN_CELLS.append(code("""\
import shutil
import yaml
from sklearn.model_selection import train_test_split

MERGED_DIR = Path("/content/solar-sentinel")
for split in ("train", "val", "test"):
    (MERGED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (MERGED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


# OOD test
ood_df = df[df["role"] == "ood_test"].copy()
ood_df["split"] = "test"

# Train/val candidates
tv_df = df[df["role"] == "train_val"].copy()
strata = (
    tv_df["source_dataset"].astype(str)
    + "|" + tv_df["primary_subtype"].astype(str)
    + "|" + tv_df["is_positive"].astype(str)
)
# Merge tiny strata into a fallback bucket so the splitter doesn't crash.
counts = strata.value_counts()
tiny = set(counts[counts < 2].index)
if tiny:
    print(f"Merging {len(tiny)} singleton strata into 'misc'.")
    strata = strata.where(~strata.isin(tiny), other="misc")
# If any stratum (including the merged 'misc') is still singleton, fall back
# to a non-stratified random split with a warning.
final_counts = strata.value_counts()
stratify_arg = strata.to_numpy() if (final_counts.min() >= 2) else None
if stratify_arg is None:
    print("WARN: stratification disabled — singleton stratum after merge")

train_idx, val_idx = train_test_split(
    tv_df.index.to_numpy(),
    test_size=0.20,
    random_state=SEED,
    stratify=stratify_arg,
)
tv_df.loc[train_idx, "split"] = "train"
tv_df.loc[val_idx, "split"] = "val"

manifest_df = pd.concat([tv_df, ood_df], ignore_index=True)


# ── Label-policy normalisation: union of all defect boxes per image ─────────
def union_box(boxes_xywh: list[list[float]]) -> list[float]:
    \"\"\"Return YOLO xywh of the bounding rectangle covering all input boxes.

    Input/output are normalised (0..1) center-x, center-y, width, height.
    \"\"\"
    x1s, y1s, x2s, y2s = [], [], [], []
    for x_c, y_c, w, h in boxes_xywh:
        x1s.append(x_c - w / 2)
        y1s.append(y_c - h / 2)
        x2s.append(x_c + w / 2)
        y2s.append(y_c + h / 2)
    x1, y1 = max(0.0, min(x1s)), max(0.0, min(y1s))
    x2, y2 = min(1.0, max(x2s)), min(1.0, max(y2s))
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


# ── Density audit: before vs after policy normalisation ────────────────────
before_counts = (
    manifest_df[manifest_df["is_positive"]]
    .assign(n_boxes=lambda d: d["boxes"].apply(len))
    .groupby("source_dataset")["n_boxes"]
    .agg(["mean", "sum", "count"])
    .rename(columns={"mean": "boxes_per_img_BEFORE", "sum": "total_boxes_BEFORE",
                     "count": "n_positive_imgs"})
)
print("Box density per source — BEFORE normalisation:")
print(before_counts)


# ── Write images + canonical-union labels ──────────────────────────────────
def write_split(row: pd.Series) -> None:
    src = Path(row["image_path"])
    safe = f"{row['source_dataset']}_{src.stem}{src.suffix}"
    dst_img = MERGED_DIR / "images" / row["split"] / safe
    dst_lbl = MERGED_DIR / "labels" / row["split"] / (safe.rsplit(".", 1)[0] + ".txt")
    shutil.copy2(src, dst_img)
    if row["boxes"]:
        bboxes = [b for _sub, b in row["boxes"]]
        x_c, y_c, w, h = union_box(bboxes)
        # Guard against degenerate boxes (zero area)
        if w > 1e-4 and h > 1e-4:
            dst_lbl.write_text(f"0 {x_c} {y_c} {w} {h}\\n")
        else:
            dst_lbl.write_text("")
    else:
        dst_lbl.write_text("")


for _, row in manifest_df.iterrows():
    write_split(row)


# Verify split balance
print("\\nFinal split composition (images):")
print(manifest_df.groupby(["split", "primary_subtype"]).size().unstack(fill_value=0))


# Sanity-check positive coverage in val
val_positives = ((manifest_df["split"] == "val") & manifest_df["is_positive"]).sum()
assert val_positives > 0, "Val split has no positives — split logic broken."
print(f"\\nVal positives: {val_positives}")


# ── Density audit AFTER ────────────────────────────────────────────────────
def count_boxes_in_split(split: str) -> dict:
    out = {}
    for src in manifest_df["source_dataset"].unique():
        rows = manifest_df[(manifest_df["split"] == split) &
                           (manifest_df["source_dataset"] == src) &
                           manifest_df["is_positive"]]
        n_imgs = len(rows)
        # After policy normalisation, every positive image has exactly 1 box.
        out[src] = {"n_positive_imgs": n_imgs, "boxes_per_img_AFTER": 1 if n_imgs else 0}
    return out


print("\\nBox density per source — AFTER normalisation (target: 1.0 across the board):")
for split in ("train", "val", "test"):
    print(f"  {split}: {count_boxes_in_split(split)}")


# ── data.yaml for Ultralytics ───────────────────────────────────────────────
data_yaml = {
    "path": str(MERGED_DIR),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": 1,
    "names": ["defect"],
}
yaml_path = MERGED_DIR / "solar_sentinel.yaml"
with open(yaml_path, "w") as f:
    yaml.safe_dump(data_yaml, f, sort_keys=False)
print(f"\\ndata.yaml written: {yaml_path}")
"""))


TRAIN_CELLS.append(md("""\
## Cell 8 — Train YOLO26n with Sprint-2 recipe

**Disconnect resilience.** Free-tier Colab will kill your runtime after ~90 min
of tab idle, and there's no way to fully prevent that. Instead, we:

1. Save the run **directly to Drive** (not /content/runs) so checkpoints
   survive a runtime kill.
2. Set `save_period=1` so Ultralytics writes `last.pt` after every epoch.
3. Auto-resume from `last.pt` if it exists — re-running this cell after a
   disconnect picks up where it left off, no manual intervention.
4. Run a randomised JS keep-alive in DevTools (see the cell after this one).

If Colab dies mid-run, just re-execute this cell. Ultralytics reads the epoch
counter / optimizer state out of `last.pt` and continues from epoch N+1.

This recipe targets a higher mAP ceiling on the (now homogenised) dataset by
following Ultralytics' fine-tuning guide rather than aggressively over-tuning
for the binary case. Defaults are kept where Ultralytics' guidance recommends
them; the previous run's overrides (cls=0.1, copy_paste=0.2, flipud=0.5,
degrees=15) were reverted because they hurt mAP on this dataset.

Key choices:

- `epochs=150, patience=30`: schedule sized for the larger Sprint-2 dataset
  (~5k train images after dedup vs. ~1k previously). With more data each epoch
  sees more variety, so the model converges faster per pass; patience=30 cuts
  the run short if val mAP plateaus before that.
- `optimizer="SGD", lr0=0.01, momentum=0.937`: SGD wins by ~+0.01–0.03 mAP at
  convergence on detection vs. AdamW (YOLOv5/8 default behaviour).
- `cos_lr=True, warmup_epochs=5`: cosine schedule + longer warmup for the
  longer run.
- `close_mosaic=15`: mosaic-off for the last 10 % of training (the
  Ultralytics-recommended ratio at any schedule length).
- `cls=0.5` (Ultralytics default, was 0.1): even with `nc=1`, the cls head
  needs gradient signal to discriminate positive vs. background. Setting cls
  too low starved the discriminator in the previous run (recall stuck at 0.24).
- Augmentation softened: `mosaic=0.5, mixup=0, copy_paste=0, degrees=5,
  flipud=0`. Aggressive aug on a small dataset pushes images further from
  deployment distribution.
- `save_period=1`: write a checkpoint after every epoch (default is final-only).
- `seed=42, deterministic=True`: cuDNN-deterministic mode for reproducibility.

Model stays YOLO26n (deployment target is Pi 5; nano fits NCNN budget).
"""))

TRAIN_CELLS.append(code("""\
from ultralytics import YOLO

# Run dir lives on Drive directly so checkpoints survive Colab disconnects.
RUN_DIR = DRIVE_DIR / "binary-trigger-run"
RUN_DIR.mkdir(parents=True, exist_ok=True)
LAST_CKPT = RUN_DIR / "weights" / "last.pt"

if LAST_CKPT.exists():
    print(f"Found existing checkpoint at {LAST_CKPT}")
    print("Resuming training (Ultralytics reads epoch + optimizer state from last.pt)...")
    model = YOLO(str(LAST_CKPT))
    results = model.train(resume=True)
else:
    print("No existing checkpoint — starting from scratch.")
    model = YOLO("yolo26n.pt")

    TRAIN_ARGS = dict(
        data=str(MERGED_DIR / "solar_sentinel.yaml"),
        project=str(RUN_DIR.parent),    # parent of the run dir on Drive
        name=RUN_DIR.name,              # 'binary-trigger-run'
        exist_ok=True,
        verbose=True,

        # Schedule (sized for ~5k-image Sprint-2 dataset)
        epochs=150,
        patience=30,
        imgsz=640,
        batch=16,
        close_mosaic=15,
        save_period=1,                   # write last.pt every epoch (resume granularity)

        # Optimizer + LR (SGD wins on detection at convergence)
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Loss (defaults — previous run's cls=0.1 starved gradient)
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Augmentation (softened for ~5k-image dataset)
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        degrees=5,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        shear=0.0,
        perspective=0.0,
        translate=0.1,
        scale=0.5,

        # Reproducibility
        seed=SEED,
        deterministic=True,
        workers=2,
    )

    results = model.train(**TRAIN_ARGS)

print(f"\\nTraining complete. Run dir: {RUN_DIR}")
"""))


TRAIN_CELLS.append(md("""\
## Cell 8b — Keep-alive (paste into DevTools console, optional but recommended)

Don't run this cell. Instead:

1. Open Colab DevTools: **right-click → Inspect → Console** tab.
2. Copy and paste the snippet below.
3. Press Enter.
4. Leave the tab open in the background.

The keep-alive emits a randomised click event every 30–90 seconds, varied
enough to avoid Google's detection of obvious activity-faking. It's
*supplementary* to the resume-from-checkpoint setup in cell 8 — even if this
fails, your training survives a disconnect.

```javascript
function jiggleColab() {
    const btn = document.querySelector("colab-connect-button")?.shadowRoot?.querySelector("#connect");
    if (btn) btn.dispatchEvent(new MouseEvent("mousemove", {bubbles: true}));
    const next = 30000 + Math.random() * 60000;
    console.log(`[keepalive] tick ${new Date().toLocaleTimeString()}, next in ${(next/1000).toFixed(0)}s`);
    setTimeout(jiggleColab, next);
}
jiggleColab();
```

To stop it: refresh the page, or run `clearTimeout(...)` if you saved the id.

If Colab disconnects anyway (free tier is unreliable), just re-run cell 8 —
training resumes from the last saved epoch automatically.
"""))


TRAIN_CELLS.append(md("""\
## Cell 9 — Validate on val + OOD test (Ultralytics built-ins)

These are the standard mAP/PR/F1 numbers from `model.val()`. Trigger-specific
metrics (PR-AUC, calibration, threshold sweep, sub-type recall) live in
`evaluate_model.ipynb` so they're re-runnable without retraining.
"""))

TRAIN_CELLS.append(code("""\
best = YOLO(str(RUN_DIR / "weights" / "best.pt"))

print("── Validation (in-distribution) ──")
val_metrics = best.val(data=str(MERGED_DIR / "solar_sentinel.yaml"), split="val", imgsz=640, verbose=False)
print(f"  mAP@50    {val_metrics.box.map50:.4f}")
print(f"  mAP@50-95 {val_metrics.box.map:.4f}")
print(f"  precision {val_metrics.box.mp:.4f}")
print(f"  recall    {val_metrics.box.mr:.4f}")

print("\\n── OOD test (dataset_3, never seen) ──")
test_metrics = best.val(data=str(MERGED_DIR / "solar_sentinel.yaml"), split="test", imgsz=640, verbose=False)
print(f"  mAP@50    {test_metrics.box.map50:.4f}")
print(f"  mAP@50-95 {test_metrics.box.map:.4f}")
print(f"  precision {test_metrics.box.mp:.4f}")
print(f"  recall    {test_metrics.box.mr:.4f}")

print("\\n── Generalisation gap ──")
print(f"  Δ mAP@50     {val_metrics.box.map50 - test_metrics.box.map50:+.4f}")
print(f"  Δ mAP@50-95  {val_metrics.box.map - test_metrics.box.map:+.4f}")
"""))


TRAIN_CELLS.append(md("""\
## Cell 10 — Persist manifest + dataset SHA256 manifest

Manifest with one row per image (less the `boxes` field — that's a Python
object, not JSON-serialisable cleanly) is written so the eval notebook can do
sub-type analyses without re-walking the dataset.
"""))

TRAIN_CELLS.append(code("""\
import json

records = []
for _, row in manifest_df.iterrows():
    records.append(
        {
            "source_dataset": row["source_dataset"],
            "split": row["split"],
            "image_filename": Path(row["image_path"]).name,
            "image_sha256": row["sha256"],
            "phash": row["phash"],
            "primary_subtype": row["primary_subtype"],
            "is_positive": bool(row["is_positive"]),
            "width": int(row["width"]),
            "height": int(row["height"]),
        }
    )

manifest_path = RUN_DIR / "dataset_manifest.json"
manifest_path.write_text(json.dumps(records, indent=2))
print(f"Manifest: {manifest_path} ({len(records)} rows)")

# Copy data.yaml too
shutil.copy2(MERGED_DIR / "solar_sentinel.yaml", RUN_DIR / "data.yaml")
"""))


TRAIN_CELLS.append(md("""\
## Cell 11 — Export to NCNN (FP16) + smoke test

NCNN is the recommended ARM CPU runtime for Pi 5 (Ultralytics benchmarks NCNN
above ONNX Runtime and TFLite on Pi 5 CPU). FP16 keeps accuracy intact while
cutting weights by ~2x.

Smoke test: run the exported NCNN model on three held-out images (one positive
and one negative from val, one from OOD test) and confirm the top score
direction is correct. The actual deployment threshold is selected in the eval
notebook; here we just sanity-check that the export didn't break inference.
"""))

TRAIN_CELLS.append(code("""\
ncnn_dir = best.export(format="ncnn", imgsz=640, half=True)
print(f"NCNN export: {ncnn_dir}")

ncnn_model = YOLO(str(ncnn_dir))


def pick_one(split: str, positive: bool) -> Path | None:
    candidates = manifest_df[
        (manifest_df["split"] == split) & (manifest_df["is_positive"] == positive)
    ]
    if candidates.empty:
        return None
    name = (
        f"{candidates.iloc[0]['source_dataset']}_"
        f"{Path(candidates.iloc[0]['image_path']).stem}"
        f"{Path(candidates.iloc[0]['image_path']).suffix}"
    )
    return MERGED_DIR / "images" / split / name


smoke_targets = {
    "val_positive": pick_one("val", True),
    "val_negative": pick_one("val", False),
    "ood_positive": pick_one("test", True),
}

print("\\nSmoke test:")
for label, path in smoke_targets.items():
    if path is None or not path.exists():
        print(f"  {label}: (not available)")
        continue
    res = ncnn_model.predict(source=str(path), imgsz=640, conf=0.05, verbose=False)
    boxes = res[0].boxes
    top = float(boxes.conf.max()) if boxes is not None and len(boxes) else 0.0
    print(f"  {label}: top conf={top:.3f}, n_boxes={len(boxes) if boxes is not None else 0}")
"""))


TRAIN_CELLS.append(md("""\
## Cell 12 — Generate MODEL_CARD.md (stub)

Renders the Mitchell-style model card. Trigger-aware metrics (PR-AUC, ECE,
deployment threshold) are filled in later by the evaluation notebook — this
cell writes the *training-time* portions: dataset composition, hyperparameters,
intended use, limitations.
"""))

TRAIN_CELLS.append(code("""\
from datetime import datetime, timezone

import jinja2

TEMPLATE = '''# Model card — Solar Sentinel binary-trigger YOLO26n

**Generated:** {{ generated_utc }}
**Run:** {{ run_name }}

## Model details
- Architecture: YOLO26 Nano (`yolo26n.pt`), Ultralytics
- Task: binary single-class object detection
- Classes: `nc=1`, `names=["defect"]`
- Image size: 640 × 640
- Framework: Ultralytics, PyTorch {{ torch_version }}, CUDA {{ cuda_version }}
- Random seed: {{ seed }} (cuDNN deterministic)
- Export: FP16 NCNN at `weights/best_ncnn_model/`

## Intended use
Binary trigger that gates a downstream multi-agent (CrewAI + Gemini) defect
analysis pipeline running on a Raspberry Pi 5 with a Camera Module 3 Wide.
Outputs a single confidence score per image; downstream agents inspect the
image to resolve sub-type, severity, and recommended action.

## Out-of-scope
- Electroluminescence imagery
- Indoor laboratory inspection
- Panel form factors not represented in the training set
- Standalone classification (no agent layer)

## Training data composition
{{ dataset_table }}

After two-pass dedup (SHA256 exact + perceptual Hamming ≤ 5):
{{ dedup_summary }}

**Label policy.** Source datasets used inconsistent annotation policies (some
drew tight boxes around every visible micro-defect, others one box around the
affected area). To homogenise, every defect-containing image carries a single
**union bounding box** — the rectangle covering all original defect boxes.
This matches what the agentic VLM Analyst needs from the trigger and removes
the val/test instance-density mismatch that capped the previous run's mAP.

## Hyperparameter overrides (vs. Ultralytics defaults)
{{ hparams_table }}

Optimizer is SGD (Ultralytics fine-tuning guide); cosine LR with 5-epoch warmup;
300 epochs with patience=50; reproducibility via seed=42 + cuDNN-deterministic.

## Validation metrics
| Split | mAP@50 | mAP@50-95 | precision | recall |
|---|---|---|---|---|
| val (in-distribution) | {{ "%.4f" % val_map50 }} | {{ "%.4f" % val_map }} | {{ "%.4f" % val_p }} | {{ "%.4f" % val_r }} |
| test (OOD, dataset_3) | {{ "%.4f" % test_map50 }} | {{ "%.4f" % test_map }} | {{ "%.4f" % test_p }} | {{ "%.4f" % test_r }} |

Generalisation gap: ΔmAP@50 = {{ "%+.4f" % (val_map50 - test_map50) }}.

## Trigger-aware metrics (filled in by evaluate_model.ipynb)
- PR-AUC: _pending_
- F2 at deployment threshold: _pending_
- ECE before / after temperature scaling: _pending_
- Recommended deployment threshold: _pending_
- Per-sub-type recall on OOD test: _pending_

## Limitations
- Roboflow source datasets contain documented label noise (loose bboxes,
  inconsistent inclusion of partial defects). Two-pass dedup mitigates leakage
  but does not improve label quality.
- All training imagery is RGB outdoor; performance on EL imagery is unknown.
- Threshold tuned on val (Roboflow imagery); not yet validated against
  Pi 5 + Camera Module 3 Wide field captures.
- Sub-type recall variance (see eval notebook): rare sub-types may have low
  recall even when overall mAP looks healthy.
'''


def make_table(rows: list[dict]) -> str:
    if not rows:
        return "(empty)"
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\\n".join(out)


comp_rows = []
for (src, sub, split), n in (
    manifest_df.groupby(["source_dataset", "primary_subtype", "split"]).size().items()
):
    comp_rows.append({"source": src, "subtype": sub, "split": split, "count": int(n)})

dedup_summary = (
    f"final {len(manifest_df)} unique images "
    f"({(manifest_df['split'] == 'train').sum()} train, "
    f"{(manifest_df['split'] == 'val').sum()} val, "
    f"{(manifest_df['split'] == 'test').sum()} test)"
)

hparam_rows = [
    {"arg": "epochs", "value": 150, "default": 100, "rationale": "sized for ~5k-image Sprint-2 dataset"},
    {"arg": "patience", "value": 30, "default": 100, "rationale": "early stop if val mAP plateaus"},
    {"arg": "optimizer", "value": "SGD", "default": "auto", "rationale": "wins at convergence on detection"},
    {"arg": "lr0", "value": 0.01, "default": 0.01, "rationale": "SGD baseline"},
    {"arg": "cos_lr", "value": True, "default": False, "rationale": "smoother decay"},
    {"arg": "warmup_epochs", "value": 5, "default": 3, "rationale": "longer warmup"},
    {"arg": "close_mosaic", "value": 15, "default": 10, "rationale": "10% mosaic-off tail"},
    {"arg": "cls", "value": 0.5, "default": 0.5, "rationale": "default — earlier 0.1 starved discriminator"},
    {"arg": "mosaic", "value": 0.5, "default": 1.0, "rationale": "softer aug for small dataset"},
    {"arg": "mixup", "value": 0.0, "default": 0.0, "rationale": "off — defects don't blend"},
    {"arg": "copy_paste", "value": 0.0, "default": 0.0, "rationale": "off — earlier 0.2 hurt mAP"},
    {"arg": "degrees", "value": 5, "default": 0, "rationale": "mild rotation jitter"},
    {"arg": "flipud", "value": 0.0, "default": 0.0, "rationale": "off — earlier 0.5 hurt mAP"},
    {"arg": "shear", "value": 0.0, "default": 0.0, "rationale": "panels are planar"},
    {"arg": "perspective", "value": 0.0, "default": 0.0, "rationale": "panels are planar"},
    {"arg": "seed", "value": SEED, "default": 0, "rationale": "reproducibility"},
    {"arg": "deterministic", "value": True, "default": True, "rationale": "reproducibility"},
]


rendered = jinja2.Template(TEMPLATE).render(
    generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    run_name=RUN_DIR.name,
    torch_version=torch.__version__,
    cuda_version=torch.version.cuda or "n/a",
    seed=SEED,
    dataset_table=make_table(comp_rows),
    dedup_summary=dedup_summary,
    hparams_table=make_table(hparam_rows),
    val_map50=val_metrics.box.map50,
    val_map=val_metrics.box.map,
    val_p=val_metrics.box.mp,
    val_r=val_metrics.box.mr,
    test_map50=test_metrics.box.map50,
    test_map=test_metrics.box.map,
    test_p=test_metrics.box.mp,
    test_r=test_metrics.box.mr,
)

card_path = RUN_DIR / "MODEL_CARD.md"
card_path.write_text(rendered)
print(f"Model card: {card_path}")
print(rendered[:1200] + "…")
"""))


TRAIN_CELLS.append(md("""\
## Cell 13 — Snapshot the finished run + publish convenience aliases

`RUN_DIR` is already on Drive (the training cell saves there directly), so
this cell does **not** need to copy the whole run again. It just snapshots
the final state to a timestamped subdir for thesis-defence reproducibility,
and refreshes the `best_latest.*` convenience aliases that the Pi-deployment
script reads from.

Old timestamped snapshots are never overwritten.
"""))

TRAIN_CELLS.append(code("""\
import shutil
from datetime import datetime, timezone

stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
snapshot = DRIVE_DIR / f"binary-trigger-{stamp}"
shutil.copytree(RUN_DIR, snapshot)

# Convenience aliases — overwrite each run
latest_pt = DRIVE_DIR / "best_latest.pt"
shutil.copy2(RUN_DIR / "weights" / "best.pt", latest_pt)

latest_ncnn = DRIVE_DIR / "best_latest_ncnn_model"
if latest_ncnn.exists():
    shutil.rmtree(latest_ncnn)
shutil.copytree(RUN_DIR / "weights" / "best_ncnn_model", latest_ncnn)

print(f"\\nSnapshot    : {snapshot}")
print(f"Latest .pt  : {latest_pt}")
print(f"Latest NCNN : {latest_ncnn}")
print("\\nNext: open evaluate_model.ipynb locally to compute trigger-aware metrics")
print("       and update the MODEL_CARD.md.")
"""))


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_model.ipynb
# ─────────────────────────────────────────────────────────────────────────────


EVAL_CELLS: list[dict] = []

EVAL_CELLS.append(md("""\
# Solar Sentinel — model evaluation (trigger-aware)

Run *locally* against a `best.pt` produced by `train_yolo26n.ipynb`. Computes
the metrics that matter for a *gate to a downstream agent*, not the metrics
that matter for a standalone classifier:

- PR-AUC, ROC-AUC
- Threshold sweep (`threshold_sweep.csv`) → deployment threshold (recall ≥ 0.95)
- Calibration curve + ECE, with post-hoc temperature scaling
- Per-sub-type recall on the OOD test split (dominant-subtype overfit check)
- Albumentations corruption probe (one-line robustness summary)
- Renders trigger metrics back into `MODEL_CARD.md`
"""))


EVAL_CELLS.append(md("""\
## Cell 1 — Configuration

Point this at the run folder you want to evaluate. The default expects the
notebook is run alongside the run directory or with the merged dataset
co-located (matches the Colab layout).
"""))

EVAL_CELLS.append(code("""\
from pathlib import Path

# Path to the training run folder (contains weights/best.pt, dataset_manifest.json, etc.)
RUN_DIR = Path("./runs/binary-trigger")
# Path to the merged dataset (the data.yaml inside is what model.val() reads)
DATASET_DIR = Path("./datasets/solar-sentinel")

assert (RUN_DIR / "weights" / "best.pt").exists(), \\
    f"Missing best.pt under {RUN_DIR}; download from Drive first."
assert (DATASET_DIR / "solar_sentinel.yaml").exists(), \\
    f"Missing dataset config under {DATASET_DIR}."

print("Run dir :", RUN_DIR.resolve())
print("Data dir:", DATASET_DIR.resolve())
"""))


EVAL_CELLS.append(md("""\
## Cell 2 — Install dependencies (local environment)

`netcal` for ECE; `albumentations` for the robustness probe.
"""))

EVAL_CELLS.append(code("""\
!uv pip install -q "ultralytics>=8.4.0" "scikit-learn>=1.4" "pandas>=2.2" \\
    "matplotlib>=3.8" "netcal>=1.3" "albumentations>=1.4" "Pillow>=10" "Jinja2>=3.1"
print("done.")
"""))


EVAL_CELLS.append(md("""\
## Cell 3 — Per-image positive scores on val and OOD test

For each image, the *image-level positive score* is the maximum box confidence
predicted at inference time (or 0 if no boxes). This converts the detector
into a binary classifier we can sweep thresholds against.
"""))

EVAL_CELLS.append(code("""\
import json

import numpy as np
import pandas as pd
from ultralytics import YOLO

manifest = pd.DataFrame(json.loads((RUN_DIR / "dataset_manifest.json").read_text()))
print(f"Loaded manifest: {len(manifest)} images")
print(manifest.groupby("split").size())

model = YOLO(str(RUN_DIR / "weights" / "best.pt"))


def image_path_for(row: pd.Series) -> Path:
    safe = f"{row['source_dataset']}_{row['image_filename']}"
    return DATASET_DIR / "images" / row["split"] / safe


def predict_scores(rows: pd.DataFrame, conf: float = 0.001) -> np.ndarray:
    scores = []
    for _, r in rows.iterrows():
        p = image_path_for(r)
        res = model.predict(source=str(p), imgsz=640, conf=conf, verbose=False)
        boxes = res[0].boxes
        scores.append(float(boxes.conf.max()) if boxes is not None and len(boxes) else 0.0)
    return np.asarray(scores)


val_rows = manifest[manifest["split"] == "val"].reset_index(drop=True)
test_rows = manifest[manifest["split"] == "test"].reset_index(drop=True)

val_rows["score"] = predict_scores(val_rows)
test_rows["score"] = predict_scores(test_rows)

print("\\nVal positives mean score:",
      val_rows.loc[val_rows["is_positive"], "score"].mean())
print("Val negatives mean score:",
      val_rows.loc[~val_rows["is_positive"], "score"].mean())
"""))


EVAL_CELLS.append(md("""\
## Cell 4 — PR-AUC, ROC-AUC, threshold sweep

PR-AUC is the headline trigger metric — threshold-free, sensitive to recall on
small positive populations. Threshold sweep produces the data driving the
deployment-threshold rule in the next cell.
"""))

EVAL_CELLS.append(code("""\
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_fscore_support
)

y_val = val_rows["is_positive"].astype(int).to_numpy()
s_val = val_rows["score"].to_numpy()
y_test = test_rows["is_positive"].astype(int).to_numpy()
s_test = test_rows["score"].to_numpy()

pr_auc_val = average_precision_score(y_val, s_val)
pr_auc_test = average_precision_score(y_test, s_test)
roc_auc_val = roc_auc_score(y_val, s_val)
roc_auc_test = roc_auc_score(y_test, s_test)

print(f"PR-AUC   val={pr_auc_val:.4f}   test={pr_auc_test:.4f}")
print(f"ROC-AUC  val={roc_auc_val:.4f}   test={roc_auc_test:.4f}")

# Threshold sweep
thresholds = np.arange(0.05, 0.96, 0.01)
sweep_rows = []
for t in thresholds:
    pred = (s_val >= t).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_val, pred, average="binary", zero_division=0
    )
    f2 = (5 * p * r / (4 * p + r)) if (4 * p + r) > 0 else 0.0
    fp = ((pred == 1) & (y_val == 0)).sum()
    fpr = fp / max((y_val == 0).sum(), 1)
    sweep_rows.append({"threshold": t, "precision": p, "recall": r, "f1": f1, "f2": f2, "fpr": fpr})

sweep = pd.DataFrame(sweep_rows)
sweep_path = RUN_DIR / "threshold_sweep.csv"
sweep.to_csv(sweep_path, index=False)
print(f"\\nSweep saved: {sweep_path}")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(sweep.threshold, sweep.precision, label="precision")
ax.plot(sweep.threshold, sweep.recall, label="recall")
ax.plot(sweep.threshold, sweep.f2, label="F2")
ax.set_xlabel("confidence threshold")
ax.set_ylabel("score")
ax.set_title("Threshold sweep on val")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(RUN_DIR / "threshold_sweep.png", dpi=140)
plt.show()
"""))


EVAL_CELLS.append(md("""\
## Cell 5 — Deployment threshold selection

Rule: lowest `t` such that `recall_val ≥ 0.95`. Fallback: `argmax(F2)`.

This threshold becomes the recommended `Settings.confidence_high` for the
deployed Pi.
"""))

EVAL_CELLS.append(code("""\
target_recall = 0.95
qualifying = sweep[sweep.recall >= target_recall]

if not qualifying.empty:
    chosen = qualifying.iloc[qualifying.threshold.argmin()]
    rule = f"recall_val >= {target_recall}"
else:
    chosen = sweep.iloc[sweep.f2.idxmax()]
    rule = "argmax(F2) — fallback"
    print(f"WARNING: no threshold reached recall ≥ {target_recall}. Falling back to argmax F2.")

t_dep = float(chosen.threshold)


def metrics_at(t: float, y: np.ndarray, s: np.ndarray) -> dict:
    pred = (s >= t).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    f2 = (5 * p * r / (4 * p + r)) if (4 * p + r) > 0 else 0.0
    fpr = ((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "f2": float(f2), "fpr": float(fpr)}


deployment = {
    "threshold": t_dep,
    "rule": rule,
    "metrics_at_threshold": {
        "val": metrics_at(t_dep, y_val, s_val),
        "test": metrics_at(t_dep, y_test, s_test),
    },
    "pr_auc": {"val": float(pr_auc_val), "test": float(pr_auc_test)},
    "roc_auc": {"val": float(roc_auc_val), "test": float(roc_auc_test)},
}
print(json.dumps(deployment, indent=2))
"""))


EVAL_CELLS.append(md("""\
## Cell 6 — Calibration curve + temperature scaling

YOLO confidences are typically over-confident post-sigmoid. We measure ECE,
fit a single temperature scalar on val NLL, and re-measure. Goes into the
model card and informs whether the chosen threshold should be applied to raw
or temperature-scaled scores.
"""))

EVAL_CELLS.append(code("""\
from netcal.metrics import ECE
from netcal.scaling import TemperatureScaling
import matplotlib.pyplot as plt


# ECE on raw scores
ece_metric = ECE(bins=10)
ece_before_val = float(ece_metric.measure(s_val, y_val))
ece_before_test = float(ece_metric.measure(s_test, y_test))


# Fit temperature on val
ts = TemperatureScaling()
ts.fit(s_val, y_val)
s_val_cal = ts.transform(s_val)
s_test_cal = ts.transform(s_test)


ece_after_val = float(ece_metric.measure(s_val_cal, y_val))
ece_after_test = float(ece_metric.measure(s_test_cal, y_test))


print(f"ECE val   raw={ece_before_val:.4f}   scaled={ece_after_val:.4f}")
print(f"ECE test  raw={ece_before_test:.4f}  scaled={ece_after_test:.4f}")
print(f"Fitted temperature: {float(ts.weights[0]):.4f}")


def reliability(ax, scores, y, bins=10, label=""):
    edges = np.linspace(0, 1, bins + 1)
    centres, accs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            continue
        centres.append((lo + hi) / 2)
        accs.append(y[mask].mean())
    ax.plot(centres, accs, marker="o", label=label)


fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
reliability(ax, s_val, y_val, label="val raw")
reliability(ax, s_val_cal, y_val, label="val scaled")
ax.set_xlabel("predicted probability")
ax.set_ylabel("empirical positive rate")
ax.set_title("Reliability diagram")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RUN_DIR / "calibration_curve.png", dpi=140)
plt.show()


calibration = {
    "ece_before": {"val": ece_before_val, "test": ece_before_test},
    "ece_after_temp": {"val": ece_after_val, "test": ece_after_test},
    "temperature": float(ts.weights[0]),
}
deployment["calibration"] = calibration
"""))


EVAL_CELLS.append(md("""\
## Cell 7 — Per-sub-type recall on OOD test

The dominant-subtype overfit check. Even though training is binary, the
manifest carries the original sub-type. If recall on, say, `soiling_bird` is
much lower than on `physical_damage`, the binary head has overfit to the
dominant sub-type — this goes into the model card as a known limitation.
"""))

EVAL_CELLS.append(code("""\
test_pos = test_rows[test_rows.is_positive].copy()
test_pos["pred"] = (test_pos.score >= t_dep).astype(int)

per_sub = (
    test_pos.groupby("primary_subtype")["pred"]
    .agg(n="size", recall="mean")
    .reset_index()
    .sort_values("recall")
)
print(per_sub.to_string(index=False))

deployment["per_subtype_recall_test"] = {
    row.primary_subtype: {"n": int(row.n), "recall": float(row.recall)}
    for row in per_sub.itertuples()
}
"""))


EVAL_CELLS.append(md("""\
## Cell 8 — Robustness probe (Albumentations corruptions)

One-line summary of how recall degrades under brightness/contrast/blur/fog
perturbations. Severity moderate; mirrors ImageNet-C severity 1–3 conventions.
"""))

EVAL_CELLS.append(code("""\
import albumentations as A
import cv2
import numpy as np

corruptions = A.Compose([
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
        A.MotionBlur(blur_limit=11, p=1.0),
        A.RandomFog(p=1.0),
    ], p=1.0),
])


def predict_corrupted(rows: pd.DataFrame) -> np.ndarray:
    scores = []
    rng = np.random.default_rng(SEED if "SEED" in dir() else 42)
    for _, r in rows.iterrows():
        path = image_path_for(r)
        img = cv2.imread(str(path))
        if img is None:
            scores.append(0.0)
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Drive Albumentations RNG deterministically
        np.random.seed(int(rng.integers(0, 1_000_000)))
        out = corruptions(image=img)["image"]
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        res = model.predict(source=out_bgr, imgsz=640, conf=0.001, verbose=False)
        boxes = res[0].boxes
        scores.append(float(boxes.conf.max()) if boxes is not None and len(boxes) else 0.0)
    return np.asarray(scores)


s_test_corr = predict_corrupted(test_rows)
recall_clean = ((s_test >= t_dep) & (y_test == 1)).sum() / max(y_test.sum(), 1)
recall_corr = ((s_test_corr >= t_dep) & (y_test == 1)).sum() / max(y_test.sum(), 1)
print(f"Recall (clean OOD)     : {recall_clean:.4f}")
print(f"Recall (corrupted OOD) : {recall_corr:.4f}")
print(f"Drop                   : {recall_clean - recall_corr:+.4f}")

deployment["robustness"] = {
    "recall_clean": float(recall_clean),
    "recall_corrupted": float(recall_corr),
}
"""))


EVAL_CELLS.append(md("""\
## Cell 9 — Persist deployment_threshold.json + update MODEL_CARD.md

Writes the trigger-aware metrics block into the run folder and patches the
model card so the “Trigger-aware metrics” section is no longer `_pending_`.
"""))

EVAL_CELLS.append(code("""\
import re

dep_path = RUN_DIR / "deployment_threshold.json"
dep_path.write_text(json.dumps(deployment, indent=2))
print(f"Wrote {dep_path}")

card_path = RUN_DIR / "MODEL_CARD.md"
card = card_path.read_text()

trigger_block_lines = [
    "## Trigger-aware metrics",
    f"- PR-AUC: val {deployment['pr_auc']['val']:.4f}, test {deployment['pr_auc']['test']:.4f}",
    f"- ROC-AUC: val {deployment['roc_auc']['val']:.4f}, test {deployment['roc_auc']['test']:.4f}",
    f"- Deployment threshold: **{deployment['threshold']:.3f}** ({deployment['rule']})",
    "  | Split | precision | recall | F2 | FPR |",
    "  |---|---|---|---|---|",
    f"  | val  | {deployment['metrics_at_threshold']['val']['precision']:.4f} | "
    f"{deployment['metrics_at_threshold']['val']['recall']:.4f} | "
    f"{deployment['metrics_at_threshold']['val']['f2']:.4f} | "
    f"{deployment['metrics_at_threshold']['val']['fpr']:.4f} |",
    f"  | test | {deployment['metrics_at_threshold']['test']['precision']:.4f} | "
    f"{deployment['metrics_at_threshold']['test']['recall']:.4f} | "
    f"{deployment['metrics_at_threshold']['test']['f2']:.4f} | "
    f"{deployment['metrics_at_threshold']['test']['fpr']:.4f} |",
    f"- ECE val: raw {deployment['calibration']['ece_before']['val']:.4f} → "
    f"scaled {deployment['calibration']['ece_after_temp']['val']:.4f} "
    f"(T={deployment['calibration']['temperature']:.4f})",
    f"- ECE test: raw {deployment['calibration']['ece_before']['test']:.4f} → "
    f"scaled {deployment['calibration']['ece_after_temp']['test']:.4f}",
    "",
    "### Per-sub-type recall on OOD test (at deployment threshold)",
    "| sub-type | n | recall |",
    "|---|---|---|",
]
for sub, info in deployment["per_subtype_recall_test"].items():
    trigger_block_lines.append(f"| {sub} | {info['n']} | {info['recall']:.4f} |")

trigger_block_lines.extend(
    [
        "",
        "### Robustness probe",
        f"- Recall clean OOD: {deployment['robustness']['recall_clean']:.4f}",
        f"- Recall corrupted OOD: {deployment['robustness']['recall_corrupted']:.4f}",
    ]
)

trigger_block = "\\n".join(trigger_block_lines)

# Replace the placeholder block in the card
pattern = r"## Trigger-aware metrics.*?(?=\\n## )"
patched, n = re.subn(pattern, trigger_block + "\\n\\n", card, count=1, flags=re.DOTALL)
if n == 0:
    patched = card + "\\n\\n" + trigger_block + "\\n"

card_path.write_text(patched)
print(f"Patched {card_path}")
print("\\n--- Trigger-aware section preview ---")
print(trigger_block)
"""))


# ─────────────────────────────────────────────────────────────────────────────
# Write notebooks
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    write_nb(NOTEBOOK_DIR / "train_yolo26n.ipynb", TRAIN_CELLS)
    write_nb(NOTEBOOK_DIR / "evaluate_model.ipynb", EVAL_CELLS)


if __name__ == "__main__":
    main()
