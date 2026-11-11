import numpy as np
from PIL import Image

from angiolens.data.qa import Pair, check_pairs, near_duplicate_pairs, thumbnail, vessel_radii
from angiolens.data.split import make_split


def _save(path, arr, mode=None):
    Image.fromarray(arr, mode=mode).save(path)
    return path


def _image(seed, size=64):
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 200, size=(size, size)).astype(np.uint8)
    return base


def _mask(size=64):
    m = np.zeros((size, size), np.uint8)
    m[30:34, 5:60] = 255
    return m


def test_check_pairs_flags_every_problem(tmp_path):
    good_img = _save(tmp_path / "good.png", _image(0))
    good_msk = _save(tmp_path / "good_m.png", _mask())
    dup_img = _save(tmp_path / "dup.png", _image(0))  # pixel-identical to good
    empty_msk = _save(tmp_path / "empty_m.png", np.zeros((64, 64), np.uint8))
    small_msk = _save(tmp_path / "small_m.png", _mask(32))
    odd_msk = _save(tmp_path / "odd_m.png", (_mask() // 255 * 3).astype(np.uint8))
    rgb = np.repeat(_image(5)[..., None], 3, axis=2)
    rgb_img = _save(tmp_path / "rgb.png", rgb)
    color = rgb.copy()
    color[0, 0] = [255, 0, 0]
    color_img = _save(tmp_path / "color.png", color)
    big_img = _save(tmp_path / "big.png", _image(9, size=80))
    pairs = [
        Pair("a", "train", good_img, good_msk),
        Pair("b", "test", dup_img, empty_msk),
        Pair("c", "train", _save(tmp_path / "c.png", _image(1)), small_msk),
        Pair("d", "train", _save(tmp_path / "d.png", _image(2)), None),
        Pair("e", "train", tmp_path / "missing.png", good_msk),
        Pair("f", "train", _save(tmp_path / "f.png", _image(3)), odd_msk),
        Pair("g", "val", rgb_img, good_msk),
        Pair("h", "val", color_img, good_msk),
        Pair("i", "val", big_img, None),
    ]
    rep = check_pairs(pairs, expected_size=(64, 64))
    issues = rep["issues"]
    assert issues["missing_image"] == ["e"]
    assert set(issues["missing_mask"]) == {"d", "i"}
    assert issues["empty_mask"] == ["b"]
    assert [d["image_id"] for d in issues["mask_size_mismatch"]] == ["c"]
    assert [d["image_id"] for d in issues["non_binary_mask"]] == ["f"]
    assert issues["gray_stored_as_rgb"] == ["g"]
    assert [d["image_id"] for d in issues["color_image"]] == ["h"]
    assert [d["image_id"] for d in issues["unexpected_size"]] == ["i"]
    # Exact duplicate across splits is found and the masks are compared.
    groups = [sorted(e["image_ids"]) for e in rep["exact_duplicates"]]
    assert ["a", "b"] in groups
    cross = [p for p in rep["near_duplicates"]["pairs"] if {p["a"], p["b"]} == {"a", "b"}]
    assert cross and cross[0]["cross_split"]
    assert rep["by_split"]["train"]["n_images"] == 4


def test_near_duplicates_by_thumbnail_correlation():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, size=(64, 64)).astype(np.uint8)
    b = np.clip(a.astype(int) + rng.integers(-3, 4, size=a.shape), 0, 255).astype(np.uint8)
    c = rng.integers(0, 255, size=(64, 64)).astype(np.uint8)
    T = np.stack([thumbnail(a), thumbnail(b), thumbnail(c)])
    pairs = near_duplicate_pairs(T, 0.97)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]


def test_vessel_radii_of_a_bar():
    m = np.zeros((50, 50), np.uint8)
    m[20:27, 5:45] = 1  # 7 px wide, radius about 3.5
    r = vessel_radii(m)
    assert r.size > 20
    assert 3.0 <= np.median(r) <= 4.0
    assert vessel_radii(np.zeros((5, 5))).size == 0


def test_split_keeps_official_split_and_blocks_leakage():
    rows = [{"image_id": i, "split": i.split("_")[0]} for i in
            ["train_1", "train_2", "train_3", "val_1", "val_2", "test_1", "test_2"]]
    exact = [["train_2", "train_3"]]
    near = [{"a": "train_1", "b": "val_1"}, {"a": "val_1", "b": "test_1"}]
    out = {r["image_id"]: r for r in make_split(rows, exact, near)}
    assert out["train_3"]["use"] == 0 and "exact duplicate of train_2" in out["train_3"]["reason"]
    assert out["train_2"]["use"] == 1
    # val_1 and test_1 join train_1 through the chain, so both leave their splits.
    assert out["val_1"]["use"] == 0 and out["test_1"]["use"] == 0
    assert out["train_1"]["use"] == 1
    assert out["test_1"]["group"] == "train_1" and out["test_1"]["group_size"] == 3
    assert out["val_2"]["use"] == 1 and out["test_2"]["use"] == 1
    assert all(r["split"] == r["image_id"].split("_")[0] for r in out.values())
