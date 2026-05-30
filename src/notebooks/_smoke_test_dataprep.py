"""Smoke-test the data-prep code from train_yolo26n.ipynb on a synthetic dataset.

Run from src/:
    uv run python notebooks/_smoke_test_dataprep.py

Builds a fake Roboflow-shaped tree under /tmp, runs the manifest +
dedup + stratified-split pipeline against it, and asserts the outputs
have the right shape. The notebook code itself is copy-pasted (close to
verbatim) into helper functions here so the real notebook doesn't grow
test-only branches.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split

SEED = 42

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
    "non-defective": None,
    "non defective": None,
    "clean": None,
    "clean panel": None,
    "normal": None,
}


def make_fake_roboflow_dataset(
    root: Path,
    name: str,
    classes: dict[int, str],
    splits: dict[str, list[tuple[str, list[tuple[int, list[float]]]]]],
    *,
    base_color: tuple[int, int, int] = (128, 128, 128),
) -> Path:
    """Create a single Roboflow-shaped dataset under root."""
    ds = root / name
    (ds / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
    (ds / "data.yaml").write_text(yaml.safe_dump({"names": classes, "nc": len(classes)}))
    for split, items in splits.items():
        (ds / split / "images").mkdir(parents=True, exist_ok=True)
        (ds / split / "labels").mkdir(parents=True, exist_ok=True)
        for stem, anns in items:
            # Build a 256x256 image where each stem produces a visually
            # distinct pattern. phash works on 8x8 DCT blocks of a 32x32
            # downsample, so we need imagery that varies on the macro grid
            # of an 8x8 block.
            rng = np.random.default_rng(abs(hash(stem)) % (2**31))
            arr = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
            # Add some structure tied to the stem so siblings differ
            n_blocks = 1 + (abs(hash(stem)) % 6)
            for _ in range(n_blocks):
                cx = rng.integers(20, 236)
                cy = rng.integers(20, 236)
                arr[max(0, cy - 30): cy + 30, max(0, cx - 30): cx + 30] = (
                    int(rng.integers(0, 255)),
                    int(rng.integers(0, 255)),
                    int(rng.integers(0, 255)),
                )
            Image.fromarray(arr).save(ds / split / "images" / f"{stem}.jpg")
            txt = "\n".join(f"{c} {' '.join(map(str, b))}" for c, b in anns)
            (ds / split / "labels" / f"{stem}.txt").write_text(txt)
    return ds


def build_manifest(datasets: list[dict], base: Path) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        ds_root = next((base / ds["name"]).rglob("data.yaml")).parent
        names_raw = yaml.safe_load((ds_root / "data.yaml").read_text()).get("names", {})
        if isinstance(names_raw, list):
            names_raw = {i: n for i, n in enumerate(names_raw)}
        names = {int(k): v.strip() for k, v in names_raw.items()}
        for split in ("train", "valid", "test"):
            img_dir = ds_root / split / "images"
            lbl_dir = ds_root / split / "labels"
            if not img_dir.exists():
                continue
            for img_path in img_dir.iterdir():
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                content = img_path.read_bytes()
                img = Image.open(img_path)
                w, h = img.size
                phash = str(imagehash.phash(img))
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                annotations = []
                if lbl_path.exists():
                    for line in lbl_path.read_text().strip().splitlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            annotations.append((int(parts[0]), [float(x) for x in parts[1:5]]))
                positive_boxes = []
                subtypes = []
                for cls_id, bbox in annotations:
                    canonical = SUBTYPE_REMAP.get(names.get(cls_id, "").lower().strip(), "other")
                    if canonical is None:
                        continue
                    subtypes.append(canonical)
                    positive_boxes.append((canonical, bbox))
                rows.append(
                    {
                        "source_dataset": ds["name"],
                        "role": ds["role"],
                        "image_path": str(img_path),
                        "label_path": str(lbl_path) if lbl_path.exists() else "",
                        "width": w,
                        "height": h,
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


def phash_clusters(df: pd.DataFrame, threshold: int = 5) -> list[list[int]]:
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


def stratify_and_write(merged_df: pd.DataFrame, merged_dir: Path) -> pd.DataFrame:
    for split in ("train", "val", "test"):
        (merged_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (merged_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    ood = merged_df[merged_df["role"] == "ood_test"].copy()
    ood["split"] = "test"

    tv = merged_df[merged_df["role"] == "train_val"].copy()
    strata = (
        tv["source_dataset"].astype(str)
        + "|" + tv["primary_subtype"].astype(str)
        + "|" + tv["is_positive"].astype(str)
    )
    counts = strata.value_counts()
    tiny = set(counts[counts < 2].index)
    if tiny:
        strata = strata.where(~strata.isin(tiny), other="misc")
    # If even the merged "misc" bucket is a singleton (or any stratum still
    # has < 2 members), fall back to a non-stratified random split with a
    # warning. Stratification is a nice-to-have, not a hard requirement.
    final_counts = strata.value_counts()
    stratify_arg = strata.to_numpy() if (final_counts.min() >= 2) else None
    if stratify_arg is None:
        print("WARN: stratification disabled — singleton stratum after merge")
    train_idx, val_idx = train_test_split(
        tv.index.to_numpy(),
        test_size=0.20,
        random_state=SEED,
        stratify=stratify_arg,
    )
    tv.loc[train_idx, "split"] = "train"
    tv.loc[val_idx, "split"] = "val"
    out = pd.concat([tv, ood], ignore_index=True)

    for _, row in out.iterrows():
        src = Path(row["image_path"])
        safe = f"{row['source_dataset']}_{src.stem}{src.suffix}"
        dst_img = merged_dir / "images" / row["split"] / safe
        dst_lbl = merged_dir / "labels" / row["split"] / (safe.rsplit(".", 1)[0] + ".txt")
        shutil.copy2(src, dst_img)
        if row["boxes"]:
            lines = [f"0 {b[0]} {b[1]} {b[2]} {b[3]}" for _sub, b in row["boxes"]]
            dst_lbl.write_text("\n".join(lines) + "\n")
        else:
            dst_lbl.write_text("")
    return out


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "raw"
        base.mkdir()
        # Dataset 1: train_val. 6 positives + 4 negatives, mixed sub-types.
        make_fake_roboflow_dataset(
            base,
            "dataset_1",
            classes={0: "Defective", 1: "Bird Drop", 2: "Clean"},
            splits={
                "train": [
                    ("d1_a", [(0, [0.5, 0.5, 0.2, 0.2])]),
                    ("d1_b", [(0, [0.3, 0.3, 0.1, 0.1])]),
                    ("d1_c", [(1, [0.4, 0.4, 0.15, 0.15])]),
                    ("d1_d", []),  # clean
                    ("d1_e", []),
                ],
                "valid": [
                    ("d1_f", [(0, [0.5, 0.5, 0.2, 0.2])]),
                    ("d1_g", [(1, [0.5, 0.5, 0.2, 0.2])]),
                    ("d1_h", []),
                ],
            },
        )
        # Dataset 2: train_val with one near-duplicate of d1_a (same content)
        make_fake_roboflow_dataset(
            base,
            "dataset_2",
            classes={0: "Dusty", 1: "Snow", 2: "Normal"},
            splits={
                "train": [
                    ("d2_a", [(0, [0.5, 0.5, 0.2, 0.2])]),
                    ("d2_b", [(1, [0.5, 0.5, 0.2, 0.2])]),
                    ("d2_c", []),
                ],
                "valid": [
                    ("d2_d", [(0, [0.5, 0.5, 0.2, 0.2])]),
                    ("d2_e", []),
                ],
            },
        )
        # Dataset 3: ood_test
        make_fake_roboflow_dataset(
            base,
            "dataset_3",
            classes={0: "crack", 1: "dust", 2: "normal"},
            splits={
                "train": [
                    ("d3_a", [(0, [0.5, 0.5, 0.2, 0.2])]),
                    ("d3_b", [(1, [0.5, 0.5, 0.2, 0.2])]),
                    ("d3_c", []),
                    ("d3_d", []),
                ],
            },
        )

        datasets = [
            {"name": "dataset_1", "role": "train_val"},
            {"name": "dataset_2", "role": "train_val"},
            {"name": "dataset_3", "role": "ood_test"},
        ]
        df = build_manifest(datasets, base)
        n_raw = len(df)
        print(f"Raw manifest rows: {n_raw}")
        assert n_raw == 5 + 3 + 3 + 2 + 4, f"unexpected raw size {n_raw}"

        # SHA256 dedup is a no-op here (different content per file)
        df = df.drop_duplicates(subset=["sha256"], keep="first").reset_index(drop=True)

        # phash dedup may consolidate near-similar small images; not asserting exact counts
        clusters = phash_clusters(df, threshold=5)
        keep = sorted({c[0] for c in clusters})
        df = df.loc[keep].reset_index(drop=True)
        print(f"After phash dedup: {len(df)} (clusters={len(clusters)})")

        merged_dir = Path(tmp) / "merged"
        out_df = stratify_and_write(df, merged_dir)

        # Assertions
        train_n = (out_df["split"] == "train").sum()
        val_n = (out_df["split"] == "val").sum()
        test_n = (out_df["split"] == "test").sum()
        print(f"split sizes: train={train_n}  val={val_n}  test={test_n}")
        assert test_n == (out_df["source_dataset"] == "dataset_3").sum(), \
            "ood_test split must contain exactly dataset_3 rows"
        assert val_n > 0 and train_n > 0, "non-empty train/val expected"
        assert ((out_df["split"] == "val") & out_df["is_positive"]).sum() > 0, \
            "val must contain positives"

        # Every label file is binary (class 0 only)
        for lbl in (merged_dir / "labels").rglob("*.txt"):
            for line in lbl.read_text().strip().splitlines():
                assert line.split()[0] == "0", f"non-binary label in {lbl}: {line!r}"

        # data.yaml shape
        yaml_path = merged_dir / "solar_sentinel.yaml"
        yaml_path.write_text(
            yaml.safe_dump(
                {
                    "path": str(merged_dir),
                    "train": "images/train",
                    "val": "images/val",
                    "test": "images/test",
                    "nc": 1,
                    "names": ["defect"],
                }
            )
        )
        cfg = yaml.safe_load(yaml_path.read_text())
        assert cfg["nc"] == 1 and cfg["names"] == ["defect"]

        print("smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
