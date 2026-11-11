"""Segmentation metrics for binary vessel masks.

Conventions (tested in tests/test_metrics.py)
* Inputs are 2D arrays; any value > 0 counts as vessel.
* If prediction and ground truth are both empty, every metric is 1.0.
* Otherwise a ratio whose denominator is zero is 0.0.

clDice follows Shit et al., "clDice: a Novel Topology-Preserving Loss Function
for Tubular Structure Segmentation", CVPR 2021:
    Tprec = |skel(P) & G| / |skel(P)|   (topology precision)
    Tsens = |skel(G) & P| / |skel(G)|   (topology sensitivity)
    clDice = 2 * Tprec * Tsens / (Tprec + Tsens)
Skeletons come from skimage.morphology.skeletonize.
"""

from __future__ import annotations

import numpy as np
from skimage.morphology import skeletonize

METRIC_NAMES = ("dice", "iou", "sensitivity", "precision", "cldice")


def _binary(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"expected a 2D mask, got shape {a.shape}")
    return a > 0


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def confusion(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int, int]:
    p, g = _binary(pred), _binary(gt)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch {p.shape} vs {g.shape}")
    tp = int(np.count_nonzero(p & g))
    fp = int(np.count_nonzero(p & ~g))
    fn = int(np.count_nonzero(~p & g))
    tn = int(p.size - tp - fp - fn)
    return tp, fp, fn, tn


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    tp, fp, fn, _ = confusion(pred, gt)
    if tp + fp + fn == 0:
        return 1.0
    return _ratio(2 * tp, 2 * tp + fp + fn)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    tp, fp, fn, _ = confusion(pred, gt)
    if tp + fp + fn == 0:
        return 1.0
    return _ratio(tp, tp + fp + fn)


def sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    tp, fp, fn, _ = confusion(pred, gt)
    if tp + fp + fn == 0:
        return 1.0
    return _ratio(tp, tp + fn)


def precision(pred: np.ndarray, gt: np.ndarray) -> float:
    tp, fp, fn, _ = confusion(pred, gt)
    if tp + fp + fn == 0:
        return 1.0
    return _ratio(tp, tp + fp)


def cldice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = _binary(pred), _binary(gt)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch {p.shape} vs {g.shape}")
    if not p.any() and not g.any():
        return 1.0
    if not p.any() or not g.any():
        return 0.0
    sp, sg = skeletonize(p), skeletonize(g)
    tprec = _ratio(np.count_nonzero(sp & g), np.count_nonzero(sp))
    tsens = _ratio(np.count_nonzero(sg & p), np.count_nonzero(sg))
    return _ratio(2 * tprec * tsens, tprec + tsens)


def all_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    tp, fp, fn, _ = confusion(pred, gt)
    if tp + fp + fn == 0:
        out = {name: 1.0 for name in METRIC_NAMES}
    else:
        out = {
            "dice": _ratio(2 * tp, 2 * tp + fp + fn),
            "iou": _ratio(tp, tp + fp + fn),
            "sensitivity": _ratio(tp, tp + fn),
            "precision": _ratio(tp, tp + fp),
            "cldice": cldice(pred, gt),
        }
    out.update({"tp": tp, "fp": fp, "fn": fn})
    return out


def soft_dice_batch(prob: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """Dice of a probability map against a binary mask (used for logging only)."""
    prob = np.asarray(prob, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    return float((2 * (prob * g).sum() + eps) / (prob.sum() + g.sum() + eps))
