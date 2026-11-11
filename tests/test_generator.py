import numpy as np
import pytest
from scipy import ndimage as ndi

from angiolens.synth.generator import SynthConfig, catheter, generate_pair, grow_tree, render_thickness


def test_same_seed_same_output_and_types():
    a_img, a_mask = generate_pair(np.random.default_rng(7))
    b_img, b_mask = generate_pair(np.random.default_rng(7))
    assert np.array_equal(a_img, b_img) and np.array_equal(a_mask, b_mask)
    assert a_img.dtype == np.float32 and a_mask.dtype == np.uint8
    assert a_img.shape == a_mask.shape == (512, 512)
    assert 0.0 <= a_img.min() and a_img.max() <= 1.0
    assert set(np.unique(a_mask)) <= {0, 1}


def test_different_seeds_differ():
    a, _ = generate_pair(np.random.default_rng(1))
    b, _ = generate_pair(np.random.default_rng(2))
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("seed", range(5))
def test_vessels_are_present_and_darker_than_surroundings(seed):
    cfg = SynthConfig(size=256, border_prob=0.0, catheter_prob=0.0)
    img, mask = generate_pair(np.random.default_rng(seed), cfg)
    assert mask.sum() > 50
    ring = ndi.binary_dilation(mask > 0, iterations=6) & ~ndi.binary_dilation(mask > 0, iterations=2)
    assert img[mask > 0].mean() < img[ring].mean()


def test_cylinder_profile_of_a_straight_vessel():
    pts = np.array([(x, 50.0, 3.0) for x in np.arange(10, 90, 0.5)])
    thick = render_thickness([pts], 100)
    col = thick[:, 50]
    rows = np.nonzero(col)[0]
    # Pixels strictly inside radius 3 of the axis: rows 48 to 52.
    assert rows.tolist() == [48, 49, 50, 51, 52]
    assert col[50] == pytest.approx(6.0)  # 2 * r on the axis
    assert col[49] == pytest.approx(2 * np.sqrt(9 - 1))


def test_radius_tapers_and_respects_minimum():
    cfg = SynthConfig(size=256)
    branches = grow_tree(np.random.default_rng(0), cfg, np.array([128.0, 128.0]), 0.0, 4.0)
    assert branches
    for b in branches:
        r = b[:, 2]
        assert np.all(np.diff(r) <= 1e-9)  # never grows along a branch
        assert r.min() >= cfg.min_radius * cfg.size / 512 - 1e-9


def test_catheter_is_rendered_but_not_labeled():
    rng = np.random.default_rng(3)
    cfg = SynthConfig(size=128)
    tube = catheter(rng, cfg, np.array([64.0, 64.0]), 0.0)
    assert tube and render_thickness(tube, 128).max() > 0
    # In full images the mask is the labeled vessels only: with vessels removed
    # (min radius above any root radius) the mask is empty even with a catheter.
    cfg_no_vessels = SynthConfig(size=128, root_radius=(1.0, 1.0), min_radius=5.0, catheter_prob=1.0,
                                 catheter_radius=(12.0, 12.0), border_prob=0.0)
    img, mask = generate_pair(np.random.default_rng(4), cfg_no_vessels)
    assert mask.sum() == 0
    smooth = ndi.gaussian_filter(img, 1.0)
    assert smooth.min() < np.median(smooth) - 0.1  # the catheter is visible
