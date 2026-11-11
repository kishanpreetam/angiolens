import numpy as np
import pytest
import torch

from angiolens.baseline.frangi import binarize, vesselness
from angiolens.data.dataset import CropDataset, intensity_window, normalize
from angiolens.models.unet import UNet, count_parameters, pick_device
from angiolens.train import dice_loss, lr_at


@pytest.mark.parametrize("size", [64, 96])
def test_unet_keeps_spatial_size(size):
    model = UNet(base=4, depth=3).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 1, size, size))
    assert out.shape == (2, 1, size, size)


def test_default_unet_size_is_small():
    # Regression guard on the architecture used in every run (about 1.96M weights).
    assert count_parameters(UNet(base=16, depth=4)) == 1_963_809


def test_pick_device_returns_a_device():
    assert isinstance(pick_device(), torch.device)
    assert pick_device("cpu").type == "cpu"


def test_intensity_window_ignores_black_border():
    img = np.full((100, 100), 150, np.uint8)
    img[:, :20] = 0  # collimator border
    img[40:60, 40:60] = 60  # dark structure
    lo, hi = intensity_window(img)
    assert lo >= 59 and hi <= 151
    x = normalize(img)
    assert x.dtype == np.float32 and x.min() == 0.0 and x.max() == 1.0


def test_crop_dataset_is_seeded_and_aligned():
    rng = np.random.default_rng(0)
    masks = (rng.random((3, 64, 64)) > 0.8).astype(np.uint8)
    images = (200 - 150 * masks).astype(np.uint8)  # vessels dark, as in angiograms
    ds = CropDataset(images, masks, [0, 1, 2], crop=32, augment=True, seed=5)
    x1, y1 = ds[1]
    x2, y2 = ds[1]
    assert x1.shape == (1, 32, 32) and y1.shape == (1, 32, 32)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    ds.set_epoch(1)
    x3, _ = ds[1]
    assert not torch.equal(x1, x3)
    # Augmentation keeps image and mask aligned: vessel pixels stay darker.
    for i in range(3):
        x, y = ds[i]
        if y.sum() > 0 and (1 - y).sum() > 0:
            assert x[y > 0].mean() < x[y == 0].mean()


def test_dice_loss_and_schedule():
    t = torch.zeros(1, 1, 8, 8)
    t[..., 2:6, 2:6] = 1
    perfect = (t * 40 - 20)  # large logits with the right sign
    assert float(dice_loss(perfect, t)) < 1e-3
    assert float(dice_loss(-perfect, t)) > 0.9
    assert lr_at(0, 1000, 1e-3, 100) == pytest.approx(1e-5)
    assert lr_at(100, 1000, 1e-3, 100) == pytest.approx(1e-3)
    assert lr_at(1000, 1000, 1e-3, 100) == pytest.approx(0.0, abs=1e-12)


def test_frangi_responds_to_dark_line_and_thresholds():
    img = np.full((96, 96), 170, np.uint8)
    img[46:50, 10:86] = 70  # dark vessel, 4 px wide
    img[:, :6] = 0  # black border column
    v = vesselness(img)
    assert v[47:49, 30:70].mean() > 5 * v[10:30, 30:70].mean()
    assert v[:, :10].max() == 0.0  # border response is suppressed
    top = binarize(v, "top_fraction", 0.05, 0)
    # Pixels at or above the 95th percentile; ties at the cutoff are kept.
    assert 0.04 < top.mean() <= 0.07
    glob = binarize(v, "global", 0.3, 0)
    assert glob[47:49, 30:70].mean() > 0.9
    with pytest.raises(ValueError):
        binarize(v, "nope", 0.1, 0)
