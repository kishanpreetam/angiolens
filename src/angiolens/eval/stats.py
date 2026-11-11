"""Bootstrap confidence intervals and paired comparisons over test images.

The unit of resampling is the image: each bootstrap replicate draws n test
images with replacement and recomputes the mean metric. Intervals are
percentile intervals.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def bootstrap_ci(values, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> dict:
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or v.size == 0:
        raise ValueError("values must be a non-empty 1D array")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(v.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0, "median": float(np.median(v)),
            "n": int(v.size), "n_boot": n_boot, "alpha": alpha}


def paired_bootstrap(a, b, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Mean of (a - b) over paired images with a bootstrap interval and a Wilcoxon test."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("a and b must be non-empty 1D arrays of equal length")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    if np.allclose(d, 0):
        p_wilcoxon = 1.0
    else:
        p_wilcoxon = float(sps.wilcoxon(a, b, zero_method="wilcox").pvalue)
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "frac_a_better": float((d > 0).mean()), "p_wilcoxon": p_wilcoxon,
            "n": int(d.size), "n_boot": n_boot}
