"""Project paths. Everything is resolved relative to the repository root."""

from __future__ import annotations

import os
from pathlib import Path

# src/angiolens/paths.py sits two levels below the repository root.
ROOT = Path(os.environ.get("ANGIOLENS_ROOT", Path(__file__).resolve().parents[2]))

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"
DOCS = ROOT / "docs"

# ARCADE layout after extraction (see angiolens.data.download).
ARCADE_ZIP_DIR = RAW / "zenodo_10390295"
# The archive is extracted into its own new directory, never next to code.
ARCADE_EXTRACT_DIR = RAW / "arcade_extracted"
ARCADE_DIR = ARCADE_EXTRACT_DIR / "arcade"
# Binary vessel masks rasterized from the COCO polygons.
ARCADE_MASKS = PROCESSED / "arcade_masks"
SPLITS_CSV = PROCESSED / "splits.csv"


def ensure(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
