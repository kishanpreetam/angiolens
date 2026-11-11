"""Turn the ARCADE "syntax" COCO annotations into binary vessel masks.

ARCADE labels 25 coronary artery segments (SYNTAX score definitions) as one
polygon each. A binary vessel mask is the union of all segment polygons of an
image. Polygons are filled with Pillow's ImageDraw (fill plus outline), the
common COCO polygon to mask conversion.

Outputs (all under data/processed, which is gitignored):
* arcade_masks/<split>/<stem>.png   uint8 masks with values 0 and 255
* arcade_index.csv                  one row per image
* cache/<split>_images.npy, cache/<split>_masks.npy
      uint8 arrays (N, 512, 512) for fast training reads (memory mapped)

Usage: python -m angiolens.data.arcade
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from angiolens.paths import ARCADE_DIR, ARCADE_MASKS, PROCESSED, ROOT, ensure

SPLITS = ("train", "val", "test")
INDEX_CSV = PROCESSED / "arcade_index.csv"
CACHE = PROCESSED / "cache"
N_SEGMENT_CATEGORIES = 25  # category 26 is "stenosis" and is not a vessel segment


@dataclass
class Record:
    image_id: str  # unique across splits, for example "test_17"
    split: str
    stem: str
    image_path: Path
    polygons: list[list[float]]
    categories: list[str]


def rel(path: Path) -> str:
    """Path relative to the repository root, for portable CSV files."""
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def load_split(split: str, base: Path = ARCADE_DIR / "syntax") -> list[Record]:
    ann_path = base / split / "annotations" / f"{split}.json"
    data = json.loads(ann_path.read_text())
    cat_names = {c["id"]: str(c["name"]) for c in data["categories"]}
    by_image: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        if ann["category_id"] > N_SEGMENT_CATEGORIES:
            continue
        by_image.setdefault(ann["image_id"], []).append(ann)
    records = []
    for im in sorted(data["images"], key=lambda m: int(Path(m["file_name"]).stem)):
        stem = Path(im["file_name"]).stem
        anns = by_image.get(im["id"], [])
        polys = [poly for a in anns for poly in a["segmentation"]]
        cats = sorted({cat_names[a["category_id"]] for a in anns})
        records.append(
            Record(
                image_id=f"{split}_{stem}",
                split=split,
                stem=stem,
                image_path=base / split / "images" / im["file_name"],
                polygons=polys,
                categories=cats,
            )
        )
    return records


def rasterize(polygons: list[list[float]], height: int, width: int) -> np.ndarray:
    """Union of COCO polygons ([x0, y0, x1, y1, ...]) as a uint8 0/1 mask."""
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for poly in polygons:
        if len(poly) < 6:
            continue
        xy = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly) - 1, 2)]
        draw.polygon(xy, fill=1, outline=1)
    return np.asarray(canvas, dtype=np.uint8)


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        if im.mode != "L":
            im = im.convert("L")
        return np.asarray(im, dtype=np.uint8)


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return (np.asarray(im) > 0).astype(np.uint8)


def prepare() -> list[dict]:
    ensure(ARCADE_MASKS, CACHE)
    rows = []
    for split in SPLITS:
        records = load_split(split)
        n = len(records)
        img_cache = np.lib.format.open_memmap(
            CACHE / f"{split}_images.npy", mode="w+", dtype=np.uint8, shape=(n, 512, 512)
        )
        msk_cache = np.lib.format.open_memmap(
            CACHE / f"{split}_masks.npy", mode="w+", dtype=np.uint8, shape=(n, 512, 512)
        )
        out_dir = ARCADE_MASKS / split
        ensure(out_dir)
        for i, rec in enumerate(records):
            img = read_gray(rec.image_path)
            if img.shape != (512, 512):
                raise ValueError(f"{rec.image_id}: unexpected shape {img.shape}")
            mask = rasterize(rec.polygons, *img.shape)
            mask_path = out_dir / f"{rec.stem}.png"
            Image.fromarray(mask * 255).save(mask_path)
            img_cache[i] = img
            msk_cache[i] = mask
            rows.append(
                {
                    "image_id": rec.image_id,
                    "split": split,
                    "cache_row": i,
                    "image_path": rel(rec.image_path),
                    "mask_path": rel(mask_path),
                    "n_segments": len(rec.polygons),
                    "segments": " ".join(rec.categories),
                    "vessel_fraction": round(float(mask.mean()), 6),
                }
            )
        img_cache.flush()
        msk_cache.flush()
        del img_cache, msk_cache
        print(f"{split}: {n} images, masks in {rel(out_dir)}")
    with open(INDEX_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"index: {rel(INDEX_CSV)} ({len(rows)} rows)")
    return rows


def read_index(path: Path = INDEX_CSV) -> list[dict]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["cache_row"] = int(r["cache_row"])
        r["n_segments"] = int(r["n_segments"])
        r["vessel_fraction"] = float(r["vessel_fraction"])
    return rows


def load_cache(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Memory mapped uint8 arrays (N, 512, 512) for one split. No pickle."""
    images = np.load(CACHE / f"{split}_images.npy", mmap_mode="r", allow_pickle=False)
    masks = np.load(CACHE / f"{split}_masks.npy", mmap_mode="r", allow_pickle=False)
    return images, masks


def main() -> int:
    prepare()
    return 0


if __name__ == "__main__":
    sys.exit(main())
