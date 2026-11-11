"""Build and load the synthetic training set.

Images are generated with a fixed seed per image (seed = base seed + index),
stored as uint8 stacks in data/processed/cache/synth_<split>_images.npy and
synth_<split>_masks.npy, and described in results/synth_summary.json.

Usage: python -m angiolens.synth.dataset [n_train] [n_val]
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from angiolens.paths import PROCESSED, RESULTS, ensure
from angiolens.synth.generator import SynthConfig, generate_pair

CACHE = PROCESSED / "cache"
SEEDS = {"train": 1_000_000, "val": 2_000_000}


def build(n_train: int = 1200, n_val: int = 100, cfg: SynthConfig | None = None) -> dict:
    cfg = cfg or SynthConfig()
    ensure(CACHE, RESULTS)
    summary = {"config": cfg.__dict__, "splits": {}}
    for split, n in (("train", n_train), ("val", n_val)):
        t0 = time.time()
        images = np.lib.format.open_memmap(CACHE / f"synth_{split}_images.npy", mode="w+", dtype=np.uint8,
                                           shape=(n, cfg.size, cfg.size))
        masks = np.lib.format.open_memmap(CACHE / f"synth_{split}_masks.npy", mode="w+", dtype=np.uint8,
                                          shape=(n, cfg.size, cfg.size))
        fractions = []
        for i in range(n):
            img, mask = generate_pair(np.random.default_rng(SEEDS[split] + i), cfg)
            images[i] = np.round(img * 255).astype(np.uint8)
            masks[i] = mask
            fractions.append(float(mask.mean()))
        images.flush()
        masks.flush()
        del images, masks
        f = np.asarray(fractions)
        summary["splits"][split] = {
            "n": n,
            "seed_base": SEEDS[split],
            "vessel_fraction_median": float(np.median(f)),
            "vessel_fraction_p5": float(np.percentile(f, 5)),
            "vessel_fraction_p95": float(np.percentile(f, 95)),
            "empty_masks": int((f == 0).sum()),
            "seconds": round(time.time() - t0, 1),
        }
        print(split, summary["splits"][split])
    summary["config"] = {k: list(v) if isinstance(v, tuple) else v for k, v in summary["config"].items()}
    (RESULTS / "synth_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def load_synth_cache(split: str) -> tuple[np.ndarray, np.ndarray]:
    images = np.load(CACHE / f"synth_{split}_images.npy", mmap_mode="r", allow_pickle=False)
    masks = np.load(CACHE / f"synth_{split}_masks.npy", mmap_mode="r", allow_pickle=False)
    return images, masks


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    n_train = int(argv[0]) if argv else 1200
    n_val = int(argv[1]) if len(argv) > 1 else 100
    build(n_train, n_val)
    return 0


if __name__ == "__main__":
    sys.exit(main())
