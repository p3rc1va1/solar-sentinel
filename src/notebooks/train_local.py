#!/usr/bin/env python3
"""Solar Sentinel — local training script (Mac/Linux, no Colab).

Mirrors the training pipeline in `notebooks/train_yolo26n.ipynb` but stripped
of Colab-only bits (Drive mount, anti-disconnect JS, /content/ paths). Auto-
detects the best available device (Apple MPS → CUDA → CPU).

Designed to run on an M4 MacBook Air or any local machine with a GPU. Output
artefacts land at `<repo>/notebooks/runs/binary-trigger/` so the existing
`evaluate_model.ipynb` finds them without changes.

Usage:
    cd src/notebooks
    ROBOFLOW_API_KEY=xxxxx uv run python train_local.py

    # Override knobs if needed
    uv run python train_local.py --epochs 100 --batch 8 --device cpu

Tested on:
    - macOS 15+ with M4 (MPS backend)
    - Linux + CUDA (T4, 4090)
    - macOS Intel CPU (slow but works)

Arguments are mostly the same as the notebook's TRAIN_ARGS; only the ones
you'd realistically change at the CLI are exposed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_local")

# ─────────────────────────────────────────────────────────────────────────────
# Constants (kept in lockstep with _build_notebooks.py)
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42

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


# ─────────────────────────────────────────────────────────────────────────────
# Device detection
# ─────────────────────────────────────────────────────────────────────────────


def pick_device(arg: str) -> str:
    """Resolve the requested device to something Ultralytics accepts."""
    import torch
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        log.info("CUDA detected: %s", torch.cuda.get_device_name(0))
        return "0"  # ultralytics expects an index for CUDA
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        log.info("Apple MPS backend detected — using GPU on Apple Silicon")
        return "mps"
    log.warning("No GPU detected; training on CPU will be SLOW")
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────


def seed_everything(seed: int) -> None:
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    log.info("Seeded everything with %d", seed)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset download
# ─────────────────────────────────────────────────────────────────────────────


def download_datasets(api_key: str, base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    for ds in DATASETS:
        out = base_dir / ds["name"]
        if out.exists() and any(out.iterdir()):
            log.info("  %s: already downloaded, skipping", ds["name"])
            continue
        log.info("  downloading %s ...", ds["name"])
        try:
            project = rf.workspace(ds["workspace"]).project(ds["project"])
            project.version(ds["version"]).download("yolov8", location=str(out))
        except Exception as e:
            log.warning("  failed to download %s: %s", ds["name"], e)
            log.warning("    Manually browse https://universe.roboflow.com/%s/%s "
                        "and pick a valid version, then re-run.",
                        ds["workspace"], ds["project"])


# ─────────────────────────────────────────────────────────────────────────────
# Manifest + dedup
# ─────────────────────────────────────────────────────────────────────────────


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


def read_dataset_yaml(ds_root: Path) -> dict:
    with open(ds_root / "data.yaml") as f:
        names = yaml.safe_load(f).get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return {int(k): v.strip() for k, v in names.items()}


def build_manifest(base_dir: Path):
    import imagehash
    import pandas as pd
    from PIL import Image

    rows = []
    for ds in DATASETS:
        ds_path = base_dir / ds["name"]
        yaml_paths = list(ds_path.rglob("data.yaml"))
        if not yaml_paths:
            log.warning("  no data.yaml under %s, skipping", ds_path)
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
                    log.warning("  skip unreadable %s: %s", img_path, e)
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


def phash_clusters(df, threshold: int = 5) -> list[list[int]]:
    import imagehash
    indices = list(df.index)
    hashes = {i: imagehash.hex_to_hash(df.at[i, "phash"]) for i in indices}
    visited: set[int] = set()
    clusters = []
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


# ─────────────────────────────────────────────────────────────────────────────
# Stratified split + label-policy normalisation (union box per image)
# ─────────────────────────────────────────────────────────────────────────────


def union_box(boxes_xywh: list[list[float]]) -> list[float]:
    """Return YOLO xywh of the rectangle covering all input boxes (normalised)."""
    x1s = [b[0] - b[2] / 2 for b in boxes_xywh]
    y1s = [b[1] - b[3] / 2 for b in boxes_xywh]
    x2s = [b[0] + b[2] / 2 for b in boxes_xywh]
    y2s = [b[1] + b[3] / 2 for b in boxes_xywh]
    x1, y1 = max(0.0, min(x1s)), max(0.0, min(y1s))
    x2, y2 = min(1.0, max(x2s)), min(1.0, max(y2s))
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def stratify_and_write(manifest_df, merged_dir: Path):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    for split in ("train", "val", "test"):
        (merged_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (merged_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    ood_df = manifest_df[manifest_df["role"] == "ood_test"].copy()
    ood_df["split"] = "test"

    tv_df = manifest_df[manifest_df["role"] == "train_val"].copy()
    strata = (
        tv_df["source_dataset"].astype(str)
        + "|" + tv_df["primary_subtype"].astype(str)
        + "|" + tv_df["is_positive"].astype(str)
    )
    counts = strata.value_counts()
    tiny = set(counts[counts < 2].index)
    if tiny:
        log.info("Merging %d singleton strata into 'misc'", len(tiny))
        strata = strata.where(~strata.isin(tiny), other="misc")
    final_counts = strata.value_counts()
    stratify_arg = strata.to_numpy() if (final_counts.min() >= 2) else None
    if stratify_arg is None:
        log.warning("stratification disabled — singleton stratum after merge")

    train_idx, val_idx = train_test_split(
        tv_df.index.to_numpy(),
        test_size=0.20,
        random_state=SEED,
        stratify=stratify_arg,
    )
    tv_df.loc[train_idx, "split"] = "train"
    tv_df.loc[val_idx, "split"] = "val"
    out_df = pd.concat([tv_df, ood_df], ignore_index=True)

    for _, row in out_df.iterrows():
        src = Path(row["image_path"])
        safe = f"{row['source_dataset']}_{src.stem}{src.suffix}"
        dst_img = merged_dir / "images" / row["split"] / safe
        dst_lbl = merged_dir / "labels" / row["split"] / (safe.rsplit(".", 1)[0] + ".txt")
        shutil.copy2(src, dst_img)
        if row["boxes"]:
            bboxes = [b for _sub, b in row["boxes"]]
            x_c, y_c, w, h = union_box(bboxes)
            if w > 1e-4 and h > 1e-4:
                dst_lbl.write_text(f"0 {x_c} {y_c} {w} {h}\n")
            else:
                dst_lbl.write_text("")
        else:
            dst_lbl.write_text("")

    val_positives = ((out_df["split"] == "val") & out_df["is_positive"]).sum()
    if not val_positives:
        raise RuntimeError("Val split has no positives — split logic broken.")
    log.info("Split sizes — train=%d val=%d test=%d (val positives=%d)",
             (out_df["split"] == "train").sum(),
             (out_df["split"] == "val").sum(),
             (out_df["split"] == "test").sum(),
             val_positives)
    return out_df


def write_data_yaml(merged_dir: Path) -> Path:
    yaml_path = merged_dir / "solar_sentinel.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump({
            "path": str(merged_dir),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": 1,
            "names": ["defect"],
        }, f, sort_keys=False)
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# Train + validate + export
# ─────────────────────────────────────────────────────────────────────────────


def train_args(yaml_path: Path, run_dir: Path, args) -> dict:
    return dict(
        data=str(yaml_path),
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        verbose=True,
        device=args.device,

        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        close_mosaic=args.close_mosaic,

        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        box=7.5, cls=0.5, dfl=1.5,

        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        degrees=5,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        shear=0.0, perspective=0.0,
        translate=0.1, scale=0.5,

        seed=SEED,
        deterministic=True,
        workers=args.workers,
    )


def train_and_evaluate(yaml_path: Path, run_dir: Path, args):
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")
    targs = train_args(yaml_path, run_dir, args)
    log.info("Train args: %s", targs)
    model.train(**targs)

    best = YOLO(str(run_dir / "weights" / "best.pt"))

    log.info("── Validation (in-distribution) ──")
    val_metrics = best.val(data=str(yaml_path), split="val", imgsz=args.imgsz, verbose=False)
    log.info("  mAP@50    %.4f", val_metrics.box.map50)
    log.info("  mAP@50-95 %.4f", val_metrics.box.map)
    log.info("  precision %.4f", val_metrics.box.mp)
    log.info("  recall    %.4f", val_metrics.box.mr)

    log.info("── OOD test (dataset_3) ──")
    test_metrics = best.val(data=str(yaml_path), split="test", imgsz=args.imgsz, verbose=False)
    log.info("  mAP@50    %.4f", test_metrics.box.map50)
    log.info("  mAP@50-95 %.4f", test_metrics.box.map)
    log.info("  precision %.4f", test_metrics.box.mp)
    log.info("  recall    %.4f", test_metrics.box.mr)

    log.info("Δ mAP@50 (val − test)  %+.4f", val_metrics.box.map50 - test_metrics.box.map50)
    return best, val_metrics, test_metrics


def export_ncnn(model, run_dir: Path, imgsz: int):
    log.info("Exporting NCNN (FP16)...")
    out = model.export(format="ncnn", imgsz=imgsz, half=True)
    log.info("NCNN: %s", out)


def write_run_artefacts(manifest_df, run_dir: Path, val_metrics, test_metrics, args):
    # JSON manifest (drop the boxes column — Python objects, not clean to serialise)
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
    (run_dir / "dataset_manifest.json").write_text(json.dumps(records, indent=2))

    # Minimal model card with what's available locally
    composition = (
        manifest_df.groupby(["source_dataset", "primary_subtype"])
        .size()
        .unstack(fill_value=0)
        .to_markdown()
    )
    val_row = (
        f"| val (in-dist) | {val_metrics.box.map50:.4f} | {val_metrics.box.map:.4f} | "
        f"{val_metrics.box.mp:.4f} | {val_metrics.box.mr:.4f} |"
    )
    test_row = (
        f"| test (OOD)    | {test_metrics.box.map50:.4f} | {test_metrics.box.map:.4f} | "
        f"{test_metrics.box.mp:.4f} | {test_metrics.box.mr:.4f} |"
    )
    card = f"""# Model card — Solar Sentinel binary-trigger YOLO26n

**Generated:** {datetime.now(timezone.utc).isoformat(timespec='seconds')}
**Run:** {run_dir.name}
**Trained on:** {args.device} (local)

## Schedule
- epochs={args.epochs}, patience={args.patience}, imgsz={args.imgsz}, batch={args.batch}
- optimizer=SGD lr0=0.01 cos_lr=True warmup_epochs=5 close_mosaic={args.close_mosaic}

## Training data
{composition}

## Validation metrics
| Split | mAP@50 | mAP@50-95 | precision | recall |
|---|---|---|---|---|
{val_row}
{test_row}

Generalisation gap: ΔmAP@50 = {val_metrics.box.map50 - test_metrics.box.map50:+.4f}

## Trigger-aware metrics
Run `evaluate_model.ipynb` against this run dir to fill in PR-AUC, ECE,
deployment-threshold, sub-type breakdown.
"""
    (run_dir / "MODEL_CARD.md").write_text(card)
    log.info("Wrote dataset_manifest.json and MODEL_CARD.md to %s", run_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=150, help="default: 150")
    ap.add_argument("--patience", type=int, default=30, help="default: 30")
    ap.add_argument("--imgsz", type=int, default=640, help="default: 640")
    ap.add_argument("--batch", type=int, default=16, help="default: 16; drop to 8 if OOM")
    ap.add_argument("--close-mosaic", dest="close_mosaic", type=int, default=15,
                    help="epochs of mosaic-off tail; default: 15 (10%% of 150)")
    ap.add_argument("--workers", type=int, default=2, help="default: 2 (determinism-friendly)")
    ap.add_argument("--device", default="auto",
                    help="auto | mps | cpu | 0 (CUDA index); default: auto")
    ap.add_argument("--data-cache", default="./datasets",
                    help="where Roboflow downloads land; default: ./datasets")
    ap.add_argument("--merged-dir", default="./datasets/solar-sentinel",
                    help="where the merged YOLO tree lives; default: ./datasets/solar-sentinel")
    ap.add_argument("--run-dir", default="./runs/binary-trigger",
                    help="output dir for weights / manifest; default: ./runs/binary-trigger")
    ap.add_argument("--skip-download", action="store_true",
                    help="skip Roboflow download (use existing --data-cache)")
    ap.add_argument("--skip-prepare", action="store_true",
                    help="skip manifest/dedup/split (use existing --merged-dir)")
    args = ap.parse_args()

    args.device = pick_device(args.device)
    seed_everything(SEED)

    base_dir = Path(args.data_cache).resolve()
    merged_dir = Path(args.merged_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        api_key = 'spTebOB7Bo1oCViX9nws'
        if not api_key:
            log.error("Set ROBOFLOW_API_KEY in your environment, or pass --skip-download.")
            return 2
        log.info("── download datasets ──")
        download_datasets(api_key, base_dir)

    if not args.skip_prepare:
        log.info("── build manifest ──")
        df = build_manifest(base_dir)
        log.info("Raw manifest: %d images", len(df))

        before = len(df)
        df = df.drop_duplicates(subset=["sha256"], keep="first").reset_index(drop=True)
        log.info("SHA256 dedup: %d → %d (-%d)", before, len(df), before - len(df))

        clusters = phash_clusters(df, threshold=5)
        keep_idx = sorted({c[0] for c in clusters})
        dropped_near = len(df) - len(keep_idx)
        df = df.loc[keep_idx].reset_index(drop=True)
        log.info("Perceptual dedup (Hamming ≤ 5): -%d, final %d images",
                 dropped_near, len(df))

        log.info("── stratified split + write merged dataset ──")
        manifest_df = stratify_and_write(df, merged_dir)
        yaml_path = write_data_yaml(merged_dir)
    else:
        log.info("Skipping prepare; using %s", merged_dir)
        yaml_path = merged_dir / "solar_sentinel.yaml"
        manifest_df = None  # we can't reconstruct it without re-walking the source datasets

    log.info("── train + evaluate ──")
    best, val_m, test_m = train_and_evaluate(yaml_path, run_dir, args)

    log.info("── export NCNN ──")
    export_ncnn(best, run_dir, args.imgsz)

    if manifest_df is not None:
        log.info("── write run artefacts ──")
        # Need data.yaml in run dir for evaluate notebook
        shutil.copy2(yaml_path, run_dir / "data.yaml")
        write_run_artefacts(manifest_df, run_dir, val_m, test_m, args)

    log.info("Done. Run at: %s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
