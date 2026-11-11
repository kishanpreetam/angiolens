import numpy as np
import pytest
from scipy import ndimage as ndi

from angiolens.eval.metrics import all_metrics, cldice, confusion, dice, iou, precision, sensitivity


def square(n=16, r0=4, r1=8, c0=4, c1=8):
    m = np.zeros((n, n), np.uint8)
    m[r0:r1, c0:c1] = 1
    return m


def test_identical_masks_score_one():
    m = square()
    out = all_metrics(m, m)
    for k in ("dice", "iou", "sensitivity", "precision", "cldice"):
        assert out[k] == pytest.approx(1.0)


def test_both_empty_is_one():
    z = np.zeros((8, 8), np.uint8)
    out = all_metrics(z, z)
    assert all(out[k] == 1.0 for k in ("dice", "iou", "sensitivity", "precision", "cldice"))


def test_disjoint_masks_score_zero():
    a = square(r0=0, r1=4, c0=0, c1=4)
    b = square(r0=10, r1=14, c0=10, c1=14)
    out = all_metrics(a, b)
    assert out["dice"] == 0 and out["iou"] == 0 and out["sensitivity"] == 0 and out["precision"] == 0
    assert out["cldice"] == 0


def test_empty_prediction_against_vessel():
    gt = square()
    pred = np.zeros_like(gt)
    assert dice(pred, gt) == 0.0
    assert sensitivity(pred, gt) == 0.0
    assert precision(pred, gt) == 0.0  # zero denominator convention
    assert cldice(pred, gt) == 0.0


def test_half_overlap_hand_computed():
    gt = square(r0=4, r1=8, c0=4, c1=8)  # 16 px
    pred = square(r0=4, r1=8, c0=4, c1=6)  # left half, 8 px
    assert confusion(pred, gt) == (8, 0, 8, 16 * 16 - 16)
    assert dice(pred, gt) == pytest.approx(16 / 24)
    assert iou(pred, gt) == pytest.approx(0.5)
    assert sensitivity(pred, gt) == pytest.approx(0.5)
    assert precision(pred, gt) == pytest.approx(1.0)


def test_nonbinary_values_are_thresholded_at_zero():
    gt = square() * 255
    pred = square() * 7
    assert dice(pred, gt) == 1.0


def test_cldice_ignores_vessel_thickness_but_dice_does_not():
    gt = np.zeros((40, 60), np.uint8)
    gt[19:22, 5:55] = 1  # 3 px wide bar
    thick = ndi.binary_dilation(gt, iterations=2).astype(np.uint8)
    assert cldice(thick, gt) == pytest.approx(1.0)
    assert dice(thick, gt) < 0.7


def test_cldice_half_vessel():
    gt = np.zeros((40, 80), np.uint8)
    gt[19:22, 10:70] = 1
    pred = np.zeros_like(gt)
    pred[19:22, 10:40] = 1
    # Topology precision is 1 (the predicted centerline lies in the vessel) and
    # topology sensitivity is about one half, so clDice is about 2/3.
    assert cldice(pred, gt) == pytest.approx(2 / 3, abs=0.04)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        dice(np.zeros((4, 4)), np.zeros((4, 5)))
    with pytest.raises(ValueError):
        dice(np.zeros((4, 4, 2)), np.zeros((4, 4, 2)))
