import numpy as np
import pytest

from angiolens.eval.stats import bootstrap_ci, paired_bootstrap


def test_constant_values_have_zero_width_interval():
    out = bootstrap_ci(np.full(50, 0.7), n_boot=500)
    assert out["mean"] == pytest.approx(0.7)
    assert out["ci_low"] == pytest.approx(0.7) and out["ci_high"] == pytest.approx(0.7)


def test_interval_contains_mean_and_is_seeded():
    v = np.random.default_rng(0).uniform(0.4, 0.9, size=200)
    a = bootstrap_ci(v, n_boot=2000, seed=3)
    b = bootstrap_ci(v, n_boot=2000, seed=3)
    assert a == b
    assert a["ci_low"] < a["mean"] < a["ci_high"]
    # The interval should be close to the normal approximation.
    se = v.std(ddof=1) / np.sqrt(v.size)
    assert a["ci_high"] - a["ci_low"] == pytest.approx(2 * 1.96 * se, rel=0.15)


def test_paired_identical_is_null():
    v = np.random.default_rng(1).uniform(size=30)
    out = paired_bootstrap(v, v.copy(), n_boot=500)
    assert out["mean_diff"] == 0 and out["p_wilcoxon"] == 1.0


def test_paired_detects_consistent_gain():
    rng = np.random.default_rng(2)
    b = rng.uniform(0.5, 0.7, size=100)
    a = b + 0.05 + rng.normal(0, 0.01, size=100)
    out = paired_bootstrap(a, b, n_boot=2000)
    assert out["ci_low"] > 0
    assert out["p_wilcoxon"] < 1e-6
    assert out["frac_a_better"] > 0.95


def test_bad_input_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([])
    with pytest.raises(ValueError):
        paired_bootstrap([1, 2], [1])
