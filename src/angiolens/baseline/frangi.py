"""Classical baseline: multiscale Frangi vesselness plus a tuned threshold.

Pipeline per frame
1. Robust intensity window to [0, 1] (same function as the U-Net input).
2. skimage.filters.frangi with black_ridges=True (contrast-filled vessels are
   dark), scales sigma = 1 to 6 px, which covers the ARCADE vessel radii.
3. Zero the response next to black collimator borders, where the filter
   fires on the border edge rather than on vessels.
4. Binarize, then drop connected components of at most `min_size` pixels.

Three threshold rules are compared on the validation split only:
* "global": vesselness divided by the frame's maximum, then a fixed cutoff
* "top_fraction": keep pixels at or above the (1 - q) quantile of vesselness
  in every frame (ties at the cutoff are kept)
* "hysteresis": like "global", but pixels down to a quarter of the cutoff are
  kept when they connect to a pixel above it
The rule, cutoff and min_size with the best mean validation Dice are frozen
and then applied once to the test split. The grids are wide (log spaced) so
the selected setting is not at the edge of the search range.

Usage: python -m angiolens.baseline.frangi
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold, frangi
from skimage.morphology import remove_small_objects

from angiolens.data.dataset import BORDER_LEVEL, normalize
from angiolens.paths import PROCESSED, RESULTS, ROOT, ensure

SIGMAS = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
GLOBAL_CUTOFFS = tuple(float(x) for x in np.round(np.geomspace(0.002, 0.3, 16), 5))
TOP_FRACTIONS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30)
HYSTERESIS_HIGH = tuple(float(x) for x in np.round(np.geomspace(0.01, 0.3, 8), 5))
MIN_SIZES = (0, 50, 150, 300, 600, 1000, 1500, 2500, 4000)
PRED_DIR = PROCESSED / "predictions" / "frangi"


def vesselness(img: np.ndarray) -> np.ndarray:
    x = normalize(img)
    v = frangi(x, sigmas=SIGMAS, black_ridges=True).astype(np.float32)
    border = ndi.binary_dilation(img <= BORDER_LEVEL, iterations=8)
    v[border] = 0.0
    return v


def binarize(v: np.ndarray, rule: str, value: float, min_size: int) -> np.ndarray:
    if rule == "global":
        vmax = float(v.max())
        mask = v >= value * vmax if vmax > 0 else np.zeros(v.shape, bool)
    elif rule == "top_fraction":
        cut = np.quantile(v, 1.0 - value)
        mask = (v >= cut) & (v > 0)
    elif rule == "hysteresis":
        vmax = float(v.max())
        if vmax > 0:
            mask = apply_hysteresis_threshold(v, 0.25 * value * vmax, value * vmax)
        else:
            mask = np.zeros(v.shape, bool)
    else:
        raise ValueError(rule)
    if min_size > 0:
        mask = remove_small_objects(mask, max_size=min_size)
    return mask.astype(np.uint8)


def _vesselness_stack(images, rows, ids):
    """Vesselness maps for validation, cached in data/processed (float16) between runs."""
    cache = PROCESSED / "frangi_val_vesselness.npy"
    meta = PROCESSED / "frangi_val_vesselness.json"
    key = {"sigmas": list(SIGMAS), "ids": list(ids)}
    if cache.exists() and meta.exists() and json.loads(meta.read_text()) == key:
        arr = np.load(cache, allow_pickle=False)
        return [a.astype(np.float32) for a in arr]
    maps = [vesselness(np.asarray(images[r])) for r in rows]
    np.save(cache, np.stack(maps).astype(np.float16), allow_pickle=False)
    meta.write_text(json.dumps(key))
    return maps


def tune(val_v: list[np.ndarray], val_gt: list[np.ndarray]) -> tuple[dict, list[dict]]:
    """Mean validation Dice for every (rule, value, min_size).

    Connected components are labeled once per (rule, value, image) and reused
    for every min_size; this gives the same masks as binarize() because both
    use 4-connectivity and drop components of at most min_size pixels.
    """
    grid = []
    for rule, values in (("global", GLOBAL_CUTOFFS), ("top_fraction", TOP_FRACTIONS),
                         ("hysteresis", HYSTERESIS_HIGH)):
        for value in values:
            scores = {ms: [] for ms in MIN_SIZES}
            for v, g in zip(val_v, val_gt):
                base = binarize(v, rule, float(value), 0).astype(bool)
                labels, _ = ndi.label(base)
                sizes = np.bincount(labels.ravel())
                gt = g > 0
                gsum = int(gt.sum())
                for ms in MIN_SIZES:
                    keep = sizes > ms
                    keep[0] = False
                    m = keep[labels]
                    den = int(m.sum()) + gsum
                    scores[ms].append(2 * int(np.count_nonzero(m & gt)) / den if den else 1.0)
            for ms in MIN_SIZES:
                grid.append({"rule": rule, "value": float(value), "min_size": int(ms),
                             "val_mean_dice": float(np.mean(scores[ms]))})
    best = max(grid, key=lambda d: d["val_mean_dice"])
    return best, grid


def _at_edge(best: dict) -> bool:
    values = {"global": GLOBAL_CUTOFFS, "top_fraction": TOP_FRACTIONS, "hysteresis": HYSTERESIS_HIGH}[best["rule"]]
    return best["value"] in (min(values), max(values)) or best["min_size"] == max(MIN_SIZES)


def main() -> int:
    from angiolens.train import data_rows

    t0 = time.time()
    v_images, v_masks, v_rows, v_ids = data_rows("arcade", "val")
    val_v = _vesselness_stack(v_images, v_rows, v_ids)
    val_gt = [np.asarray(v_masks[r]) for r in v_rows]
    best, grid = tune(val_v, val_gt)
    print(f"tuned on {len(v_rows)} val images in {time.time() - t0:.0f} s: {best}")

    t_images, t_masks, t_rows, t_ids = data_rows("arcade", "test")
    out = PRED_DIR / "test"
    ensure(out, RESULTS)
    t1 = time.time()
    for r, image_id in zip(t_rows, t_ids):
        v = vesselness(np.asarray(t_images[r]))
        mask = binarize(v, best["rule"], best["value"], best["min_size"])
        Image.fromarray(mask * 255).save(out / f"{image_id}.png")
    per_image_s = (time.time() - t1) / max(len(t_rows), 1)
    report = {
        "method": "Frangi vesselness (skimage) plus threshold",
        "sigmas": list(SIGMAS),
        "black_ridges": True,
        "selected": best,
        "selection": "highest mean Dice on the validation split; test split never used for tuning",
        "n_val": len(v_rows),
        "n_test": len(t_rows),
        "seconds_per_test_image_cpu": round(per_image_s, 3),
        "grid": grid,
        "at_grid_edge": _at_edge(best),
    }
    (RESULTS / "frangi_tuning.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {len(t_rows)} test masks to {out.relative_to(ROOT)}; {per_image_s:.2f} s per image")
    return 0


if __name__ == "__main__":
    sys.exit(main())
