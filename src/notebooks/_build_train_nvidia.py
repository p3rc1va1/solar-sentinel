"""Generate train_nvidia.ipynb from inline cell specs.

Run from src/notebooks/:
    uv run python _build_train_nvidia.py

This creates a Jupyter notebook for local CUDA training on an RTX 5080
(Blackwell). Mirrors the structure of train_yolo26n.ipynb but strips
Colab/Drive bits and tunes hyperparameters for the bigger card.

Hand to your friend. They run cells top to bottom in JupyterLab / VS Code /
classic Jupyter. Output lands at ./runs/binary-trigger which `evaluate_model.ipynb`
already reads from.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "train_nvidia.ipynb"

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


CELLS: list[dict] = []


CELLS.append(md("""\
# Solar Sentinel — YOLO26n training (NVIDIA / RTX 5080)

This is the **local CUDA** variant of the training pipeline. Identical to
`train_yolo26n.ipynb` (Colab) and `train_local.py` (Mac MPS) in *what* it
does — downloads the same six datasets, applies the same dedup + union-box
label policy, and produces the same artefacts at `./runs/binary-trigger/`
that `evaluate_model.ipynb` reads. The differences are local:

- **Tuned for Blackwell (RTX 5080, sm_120):** AMP enabled, batch 32 (vs.
  16 on T4 / Mac), channels-last memory format, TF32 on for matmul.
- **No Colab/Drive dependencies:** weights save to `./runs/binary-trigger/`
  on local disk.
- **No 90-minute idle timeout:** runs uninterrupted as long as your terminal
  is alive.

**Expected wall-clock on RTX 5080**: ~12–15 s/epoch at batch 32, imgsz 640.
150 epochs ≈ 30–40 minutes. With patience=30 likely stopping ~epoch 90–120,
realistic finish time is **20–30 minutes**.

## What you need
- NVIDIA driver supporting Blackwell (570+ on Linux, 572+ on Windows)
- CUDA 12.6+ runtime (PyTorch wheels with cu126 or cu128)
- Python 3.11+
- A Roboflow API key (free): https://app.roboflow.com/settings/api
"""))


CELLS.append(md("""\
## Cell 1 — Install dependencies

Pinned versions; PyTorch wheel built against CUDA 12.6 (works on Blackwell).
"""))

CELLS.append(code("""\
# RTX 5080 (Blackwell sm_120) requires PyTorch 2.7+ with cu126 or cu128 wheels.
# Standard pip install grabs CPU-only torch on Linux without --index-url.
!pip install --quiet --index-url https://download.pytorch.org/whl/cu128 \\
    torch torchvision

!pip install --quiet "ultralytics>=8.4.0" "roboflow" "imagehash>=4.3" \\
    "scikit-learn>=1.4" "pandas>=2.2" "Jinja2>=3.1" "tabulate>=0.9"

print("Dependencies installed.")
"""))


CELLS.append(md("""\
## Cell 2 — GPU + driver sanity

Confirms PyTorch sees the 5080 and CUDA is wired up. If this errors, fix it
before anything else — every later cell assumes a working CUDA setup.
"""))

CELLS.append(code("""\
import torch

assert torch.cuda.is_available(), \\
    "No CUDA device visible to PyTorch. Check NVIDIA driver + CUDA runtime."

dev = torch.cuda.get_device_properties(0)
print(f"GPU         : {dev.name}")
print(f"VRAM        : {dev.total_memory / 1e9:.1f} GB")
print(f"Compute cap : sm_{dev.major}{dev.minor}")
print(f"PyTorch     : {torch.__version__}")
print(f"CUDA build  : {torch.version.cuda}")

# Blackwell (5080/5090) is sm_120. Older PyTorch wheels won't recognise it
# and silently fall back to CPU — fail loudly if so.
if dev.major == 12 and torch.version.cuda is None:
    raise RuntimeError("PyTorch wasn't built with CUDA support — reinstall the cu128 wheel.")
"""))


CELLS.append(md("""\
## Cell 3 — Reproducibility seeding

cuDNN-deterministic mode + global seed. Costs 5–10% throughput on Blackwell;
worth it for a thesis-grade reproducible run.
"""))

CELLS.append(code("""\
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

# Blackwell perf knobs — keep TF32 for matmul (FP32-shaped, faster)
# but allow AMP (FP16) inside Ultralytics' training loop.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"Seeded everything with {SEED}; TF32 enabled.")
"""))


CELLS.append(md("""\
## Cell 4 — Download datasets

Six RGB Roboflow sources. Five go into train+val (after dedup); `dataset_3`
is held entirely out as a cross-source OOD test. Cached under `./datasets/`
so re-runs skip downloads.
"""))

CELLS.append(code("""\
from pathlib import Path

ROBOFLOW_API_KEY = ""  # paste your key here

BASE_DIR = Path("./datasets/raw").resolve()
BASE_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    {"name": "dataset_1", "workspace": "solar-panel-defect-detection",
     "project": "solar-panel-defect-detection-hpser", "version": 1, "role": "train_val"},
    {"name": "dataset_2", "workspace": "solar-panel-defect-ufk3b",
     "project": "solar-panel-defect-2-aeuek", "version": 4, "role": "train_val"},
    {"name": "dataset_3", "workspace": "gao-shou-zheng-b6xqc",
     "project": "solar-panel-0swal", "version": 3, "role": "ood_test"},
    {"name": "dataset_4_gydkp", "workspace": "solar-panel-q3vlt",
     "project": "solar-panel-gydkp", "version": 1, "role": "train_val"},
    {"name": "dataset_5_maintenance", "workspace": "solarpanels-mi4yb",
     "project": "solar-panel-maintenance", "version": 1, "role": "train_val"},
    {"name": "dataset_6_phase2", "workspace": "solarpaneldefectdetectionphase2",
     "project": "solar-panel-defect-detection-f6wsy", "version": 1, "role": "train_val"},
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
        print(f"  {ds['name']}: already present, skipping")
        continue
    print(f"  downloading {ds['name']}...")
    try:
        project = rf.workspace(ds["workspace"]).project(ds["project"])
        project.version(ds["version"]).download("yolov8", location=str(out))
    except Exception as e:
        print(f"  WARN: failed to download {ds['name']}: {e}")
        print(f"        Manually browse https://universe.roboflow.com/{ds['workspace']}/{ds['project']} "
              f"and pick a valid version, then re-run.")
print("Datasets ready.")
"""))


CELLS.append(md("""\
## Cell 5 — Build manifest with perceptual dedup

Identical logic to the Colab/Mac variants:

- Per-image manifest: source, sub-type label (kept as metadata only), is_positive,
  SHA256, perceptual hash.
- **Pass 1**: drop SHA256 duplicates.
- **Pass 2**: cluster perceptual hashes within Hamming distance 5; keep one
  representative per cluster. Roboflow datasets routinely contain consecutive
  video frames; without this pass they leak between train and val.

Sub-type metadata stays in the manifest (used for stratification and per-sub-type
recall reporting), *not* in the YOLO labels (which are binary).
"""))

CELLS.append(code("""\
import hashlib
import yaml
from collections import Counter
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image

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
    "non-defective": None, "non defective": None,
    "clean": None, "clean panel": None, "normal": None,
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
                positive_boxes = []
                subtypes = []
                for cls_id, bbox in annotations:
                    name = names.get(cls_id, "").lower().strip()
                    canonical = SUBTYPE_REMAP.get(name, "other")
                    if canonical is None:
                        continue
                    subtypes.append(canonical)
                    positive_boxes.append((canonical, bbox))
                rows.append({
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
                })
    return pd.DataFrame(rows)


df = build_manifest()
print(f"Raw manifest: {len(df)} images")
print(df.groupby(["source_dataset", "primary_subtype"]).size().unstack(fill_value=0))


# Pass 1: SHA256 dedup
before = len(df)
df = df.drop_duplicates(subset=["sha256"], keep="first").reset_index(drop=True)
print(f"\\nSHA256 dedup: {before} → {len(df)} (-{before - len(df)})")


# Pass 2: perceptual near-dup
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
dropped = len(df) - len(keep_idx)
df = df.loc[keep_idx].reset_index(drop=True)
print(f"Perceptual dedup (Hamming ≤ 5): -{dropped}, final {len(df)} images")
print("\\nFinal composition:")
print(df.groupby(["role", "primary_subtype"]).size().unstack(fill_value=0))
"""))


CELLS.append(md("""\
## Cell 6 — Stratified split + canonical-union label policy

`dataset_3` is **entirely** test (cross-source OOD). Within train+val sources,
stratified 80/20 split on `(source_dataset, primary_subtype, is_positive)`.

**Label normalisation:** every defect-containing image gets a single union
bounding box covering all original defect annotations. Rationale: source
datasets used inconsistent annotation policies (some drew tight per-defect
boxes, others one box around the affected area). The union policy is
deterministic, neutral, and matches what the agentic VLM downstream needs.
"""))

CELLS.append(code("""\
import shutil
from sklearn.model_selection import train_test_split

MERGED_DIR = Path("./datasets/solar-sentinel").resolve()
for split in ("train", "val", "test"):
    (MERGED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (MERGED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

ood_df = df[df["role"] == "ood_test"].copy()
ood_df["split"] = "test"

tv_df = df[df["role"] == "train_val"].copy()
strata = (
    tv_df["source_dataset"].astype(str)
    + "|" + tv_df["primary_subtype"].astype(str)
    + "|" + tv_df["is_positive"].astype(str)
)
counts = strata.value_counts()
tiny = set(counts[counts < 2].index)
if tiny:
    print(f"Merging {len(tiny)} singleton strata into 'misc'.")
    strata = strata.where(~strata.isin(tiny), other="misc")
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


def union_box(boxes_xywh: list[list[float]]) -> list[float]:
    \"\"\"Return YOLO xywh of the rectangle covering all input boxes.\"\"\"
    x1s = [b[0] - b[2] / 2 for b in boxes_xywh]
    y1s = [b[1] - b[3] / 2 for b in boxes_xywh]
    x2s = [b[0] + b[2] / 2 for b in boxes_xywh]
    y2s = [b[1] + b[3] / 2 for b in boxes_xywh]
    x1, y1 = max(0.0, min(x1s)), max(0.0, min(y1s))
    x2, y2 = min(1.0, max(x2s)), min(1.0, max(y2s))
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


for _, row in manifest_df.iterrows():
    src = Path(row["image_path"])
    safe = f"{row['source_dataset']}_{src.stem}{src.suffix}"
    dst_img = MERGED_DIR / "images" / row["split"] / safe
    dst_lbl = MERGED_DIR / "labels" / row["split"] / (safe.rsplit(".", 1)[0] + ".txt")
    shutil.copy2(src, dst_img)
    if row["boxes"]:
        bboxes = [b for _sub, b in row["boxes"]]
        x_c, y_c, w, h = union_box(bboxes)
        if w > 1e-4 and h > 1e-4:
            dst_lbl.write_text(f"0 {x_c} {y_c} {w} {h}\\n")
        else:
            dst_lbl.write_text("")
    else:
        dst_lbl.write_text("")


# Verify
val_positives = ((manifest_df["split"] == "val") & manifest_df["is_positive"]).sum()
assert val_positives > 0, "Val split has no positives — split logic broken."

print("Final split composition (images):")
print(manifest_df.groupby(["split", "primary_subtype"]).size().unstack(fill_value=0))
print(f"\\nVal positives: {val_positives}")
print("Box density per source — AFTER normalisation (should be 1.0 for all positives):")


# data.yaml
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


CELLS.append(md("""\
## Cell 7 — Train YOLO26n (Blackwell-tuned)

This is the same Sprint-2 hyperparameter recipe as the other variants. The
RTX 5080 changes are purely throughput knobs — they don't change the loss
or augmentation choices, just make each iteration faster:

- **`batch=32`** (vs. 16 on T4 / Mac): 16 GB GDDR7 fits this comfortably for
  YOLO26n at imgsz=640. Faster gradient updates, smoother BN stats. Drop to
  `batch=16` if you OOM (rare for nano).
- **`amp=True`** (default in Ultralytics): FP16 mixed precision. Blackwell
  hardware-accelerates this — typical 1.5–2x speedup over pure FP32.
- **`device=0`**: explicit CUDA index 0.
- **`save_period=1`**: writes `last.pt` every epoch. Cheap on local disk; lets
  you crash-recover without re-doing 20 epochs.

Schedule and loss/augmentation knobs are identical to Colab and Mac variants:
- `epochs=150, patience=30, close_mosaic=15, imgsz=640`
- `optimizer=SGD, lr0=0.01, momentum=0.937, cos_lr=True, warmup_epochs=5`
- `cls=0.5, mosaic=0.5, mixup=0, copy_paste=0, degrees=5, flipud=0`

If `last.pt` already exists in the run dir (interrupted previous run),
training auto-resumes from the checkpoint. Re-run this cell after a crash
and Ultralytics picks up at epoch N+1.
"""))

CELLS.append(code("""\
from ultralytics import YOLO

RUN_DIR = Path("./runs/binary-trigger").resolve()
RUN_DIR.parent.mkdir(parents=True, exist_ok=True)
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
        project=str(RUN_DIR.parent),
        name=RUN_DIR.name,
        exist_ok=True,
        verbose=True,
        device=0,                       # CUDA:0

        # Schedule
        epochs=150,
        patience=30,
        imgsz=640,
        batch=32,                       # 5080 has VRAM for this; nano model
        close_mosaic=15,
        save_period=1,                  # checkpoint every epoch

        # Optimizer
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Loss
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Augmentation
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

        # Throughput
        amp=True,                       # Blackwell hardware FP16
        workers=4,                      # bump from 2 — desktop CPU has cores
        cache=False,                    # set "ram" if you have >32 GB system RAM

        # Reproducibility
        seed=SEED,
        deterministic=True,
    )

    results = model.train(**TRAIN_ARGS)

print(f"\\nTraining complete. Run dir: {RUN_DIR}")
"""))


CELLS.append(md("""\
## Cell 8 — Validate on val + OOD test (Ultralytics built-ins)

Standard mAP/PR/F1 numbers from `model.val()`. Trigger-aware metrics
(PR-AUC, calibration, threshold sweep, sub-type recall) live in
`evaluate_model.ipynb` — re-runnable without retraining.
"""))

CELLS.append(code("""\
best = YOLO(str(RUN_DIR / "weights" / "best.pt"))

print("── Validation (in-distribution) ──")
val_metrics = best.val(data=str(yaml_path), split="val", imgsz=640, verbose=False)
print(f"  mAP@50    {val_metrics.box.map50:.4f}")
print(f"  mAP@50-95 {val_metrics.box.map:.4f}")
print(f"  precision {val_metrics.box.mp:.4f}")
print(f"  recall    {val_metrics.box.mr:.4f}")

print("\\n── OOD test (dataset_3, never seen) ──")
test_metrics = best.val(data=str(yaml_path), split="test", imgsz=640, verbose=False)
print(f"  mAP@50    {test_metrics.box.map50:.4f}")
print(f"  mAP@50-95 {test_metrics.box.map:.4f}")
print(f"  precision {test_metrics.box.mp:.4f}")
print(f"  recall    {test_metrics.box.mr:.4f}")

print("\\n── Generalisation gap ──")
print(f"  Δ mAP@50     {val_metrics.box.map50 - test_metrics.box.map50:+.4f}")
print(f"  Δ mAP@50-95  {val_metrics.box.map - test_metrics.box.map:+.4f}")
"""))


CELLS.append(md("""\
## Cell 9 — Persist manifest + data.yaml to run dir

So `evaluate_model.ipynb` can do sub-type analyses without re-walking the
source datasets.
"""))

CELLS.append(code("""\
import json

records = []
for _, row in manifest_df.iterrows():
    records.append({
        "source_dataset": row["source_dataset"],
        "split": row["split"],
        "image_filename": Path(row["image_path"]).name,
        "image_sha256": row["sha256"],
        "phash": row["phash"],
        "primary_subtype": row["primary_subtype"],
        "is_positive": bool(row["is_positive"]),
        "width": int(row["width"]),
        "height": int(row["height"]),
    })

manifest_path = RUN_DIR / "dataset_manifest.json"
manifest_path.write_text(json.dumps(records, indent=2))
print(f"Manifest: {manifest_path} ({len(records)} rows)")

shutil.copy2(MERGED_DIR / "solar_sentinel.yaml", RUN_DIR / "data.yaml")
"""))


CELLS.append(md("""\
## Cell 10 — Export to NCNN (FP16) + smoke test

NCNN is the recommended ARM CPU runtime for the Pi 5 deployment target.
FP16 keeps accuracy ~unchanged while halving weight size.

Smoke test: run the exported NCNN model on three held-out images (one
positive + one negative from val, one OOD positive) and confirm the score
direction is sensible. The deployment threshold itself is selected later
in `evaluate_model.ipynb`.
"""))

CELLS.append(code("""\
ncnn_dir = best.export(format="ncnn", imgsz=640, half=True)
print(f"NCNN export: {ncnn_dir}")

ncnn_model = YOLO(str(ncnn_dir))


def pick_one(split: str, positive: bool):
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


CELLS.append(md("""\
## Cell 11 — Render MODEL_CARD.md (training-time portion)

Trigger-aware metrics (PR-AUC, ECE, deployment threshold) are filled in
later by `evaluate_model.ipynb`.
"""))

CELLS.append(code("""\
from datetime import datetime, timezone

import jinja2

TEMPLATE = '''# Model card — Solar Sentinel binary-trigger YOLO26n

**Generated:** {{ generated_utc }}
**Run:** {{ run_name }}
**Trained on:** {{ device_name }} ({{ vram_gb }} GB, sm_{{ compute_cap }})

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

## Out-of-scope
- Electroluminescence imagery
- Indoor laboratory inspection
- Panel form factors not represented in the training set
- Standalone classification (no agent layer)

## Training data composition
{{ dataset_table }}

After two-pass dedup (SHA256 exact + perceptual Hamming ≤ 5):
{{ dedup_summary }}

**Label policy.** Every defect-containing image carries a single union
bounding box (the rectangle covering all original defect annotations).
This homogenises annotation policies across the six source datasets.

## Hyperparameter overrides (vs. Ultralytics defaults)
{{ hparams_table }}

Optimizer SGD; cosine LR with 5-epoch warmup; 150 epochs / patience=50;
seed=42 + cuDNN-deterministic; batch=32 (Blackwell); AMP on.

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
- Roboflow source datasets contain documented label noise.
- All training imagery is RGB outdoor; performance on EL imagery unknown.
- Threshold tuned on val (Roboflow imagery); not validated against
  Pi 5 + Camera Module 3 Wide field captures.
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
    {"arg": "epochs", "value": 150, "default": 100, "rationale": "Sprint-2 schedule"},
    {"arg": "patience", "value": 30, "default": 100, "rationale": "early stop on val plateau"},
    {"arg": "batch", "value": 32, "default": 16, "rationale": "RTX 5080 has VRAM headroom"},
    {"arg": "optimizer", "value": "SGD", "default": "auto", "rationale": "wins on detection"},
    {"arg": "lr0", "value": 0.01, "default": 0.01, "rationale": "SGD baseline"},
    {"arg": "cos_lr", "value": True, "default": False, "rationale": "smoother decay"},
    {"arg": "warmup_epochs", "value": 5, "default": 3, "rationale": "longer warmup"},
    {"arg": "close_mosaic", "value": 15, "default": 10, "rationale": "10% mosaic-off tail"},
    {"arg": "cls", "value": 0.5, "default": 0.5, "rationale": "default"},
    {"arg": "mosaic", "value": 0.5, "default": 1.0, "rationale": "softer aug"},
    {"arg": "mixup", "value": 0.0, "default": 0.0, "rationale": "off — defects don't blend"},
    {"arg": "copy_paste", "value": 0.0, "default": 0.0, "rationale": "off"},
    {"arg": "degrees", "value": 5, "default": 0, "rationale": "mild rotation jitter"},
    {"arg": "flipud", "value": 0.0, "default": 0.0, "rationale": "off"},
    {"arg": "shear", "value": 0.0, "default": 0.0, "rationale": "panels are planar"},
    {"arg": "perspective", "value": 0.0, "default": 0.0, "rationale": "panels are planar"},
    {"arg": "amp", "value": True, "default": True, "rationale": "Blackwell hardware FP16"},
    {"arg": "seed", "value": SEED, "default": 0, "rationale": "reproducibility"},
    {"arg": "deterministic", "value": True, "default": True, "rationale": "reproducibility"},
]

dev_props = torch.cuda.get_device_properties(0)

rendered = jinja2.Template(TEMPLATE).render(
    generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    run_name=RUN_DIR.name,
    device_name=dev_props.name,
    vram_gb=f"{dev_props.total_memory / 1e9:.0f}",
    compute_cap=f"{dev_props.major}{dev_props.minor}",
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


CELLS.append(md("""\
## Done

You should now have:

```
./runs/binary-trigger/
├── weights/
│   ├── best.pt                         # PyTorch checkpoint
│   ├── last.pt                         # last epoch (resume target)
│   └── best_ncnn_model/                # FP16 NCNN export for Pi 5
├── args.yaml                           # Ultralytics auto-saved args
├── data.yaml
├── dataset_manifest.json               # for evaluate_model.ipynb
├── MODEL_CARD.md                       # training-time portion filled in
├── results.png + confusion_matrix.png + PR_curve.png + F1_curve.png
└── ... (Ultralytics auto)
```

Hand `weights/best.pt` to Baha — he'll plug it into `evaluate_model.ipynb`
to compute PR-AUC, calibration, deployment threshold, sub-type breakdown,
robustness probe.

Or zip the whole run dir and send the lot:

```python
import shutil
shutil.make_archive("./runs/binary-trigger", "zip", "./runs/binary-trigger")
print("Zipped → ./runs/binary-trigger.zip")
```
"""))


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": NB_KERNELSPEC,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.write_text(json.dumps(nb, indent=1))
    print(f"  wrote {NB_PATH} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
