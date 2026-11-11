"""Tests for the review session logic (no GUI)."""

from __future__ import annotations

import ast
import csv
import io
import re
import subprocess
import sys
import tokenize
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from angiolens.review import session as session_mod
from angiolens.review.session import (
    LOG_COLUMNS,
    STATUSES,
    ReviewItem,
    ReviewSession,
    dice_score,
    load_image,
    load_manifest,
    load_mask,
    make_demo_items,
    perturb_mask,
    safe_filename,
    write_manifest,
)

SHAPE = (24, 32)
PRED_AREA = 80  # the stand-in prediction is a 4 x 20 bar
EM_DASH, EN_DASH, ARROW = chr(0x2014), chr(0x2013), chr(0x2192)  # kept out of the source on purpose
# Words that must not appear in the code base, spelled in pieces so this file does not contain them.
BANNED_WORDS = ["Cl" + "aude", "Anthr" + "opic", "assis" + "tant", "A" + "I"]


# Helpers


def save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def make_items(
    tmp_path: Path,
    n: int = 3,
    ids: list[str] | None = None,
    with_gt: bool = True,
    empty_pred: bool = False,
    shape: tuple[int, int] = SHAPE,
) -> list[ReviewItem]:
    """Write n small cases to disk. Each prediction is a bar at rows 4 to 7, columns 5 to 24."""
    rng = np.random.default_rng(0)
    folder = tmp_path / "inputs"
    items = []
    for i in range(n):
        pred = np.zeros(shape, np.uint8)
        if not empty_pred:
            pred[4:8, 5:25] = 255
        truth = np.zeros(shape, np.uint8)
        truth[4:8, 5:30] = 255
        save_png(folder / f"item{i}_image.png", rng.integers(0, 256, shape, dtype=np.uint8))
        save_png(folder / f"item{i}_pred.png", pred)
        save_png(folder / f"item{i}_gt.png", truth)
        items.append(
            ReviewItem(
                image_id=ids[i] if ids else f"case{i}",
                image_path=folder / f"item{i}_image.png",
                pred_path=folder / f"item{i}_pred.png",
                gt_path=folder / f"item{i}_gt.png" if with_gt else None,
                priority=float(i),
            )
        )
    return items


def new_session(tmp_path: Path, n: int = 3, **kwargs) -> ReviewSession:
    items = make_items(tmp_path, n=n, **{k: kwargs.pop(k) for k in ("ids", "with_gt", "empty_pred") if k in kwargs})
    return ReviewSession(items, tmp_path / "out", **kwargs)


def read_log(out_dir: Path) -> list[dict[str, str]]:
    with open(out_dir / "review_log.csv", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def disk_pixels(radius: float) -> int:
    """Number of integer points in a full disk, counted the slow and obvious way."""
    r = int(np.floor(radius))
    return sum(1 for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dy * dy + dx * dx <= radius * radius)


def tiny_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Stand-in for the real generator: a noisy image and a two-bar mask."""
    image = rng.random((40, 48)).astype(np.float32)
    mask = np.zeros((40, 48), np.uint8)
    mask[10:14, 5:43] = 1
    mask[10:35, 20:23] = 1
    return image, mask


# Painting and erasing


def test_paint_and_erase_counts(tmp_path):
    s = new_session(tmp_path)
    assert int(s.mask.sum()) == PRED_AREA

    assert s.paint(16, 16, 3, 1) == disk_pixels(3) == 29
    assert int(s.mask.sum()) == PRED_AREA + 29
    assert s.paint(16, 16, 3, 1) == 0, "painting over painted pixels changes nothing"
    assert s.n_edits == 1, "a no-op is not an edit"

    assert s.paint(16, 16, 1, 0) == 5
    assert int(s.mask.sum()) == PRED_AREA + 29 - 5
    assert s.paint(0, 0, 0, 0) == 0, "erasing background changes nothing"
    assert s.n_edits == 2


def test_paint_radius_zero_and_fractional_center(tmp_path):
    s = new_session(tmp_path)
    assert s.paint(16, 16, 0, 1) == 1
    assert s.mask[16, 16] == 1
    assert s.paint(14.4, 20.6, 0, 1) == 1, "the center is rounded to the nearest pixel"
    assert s.mask[14, 21] == 1


def test_paint_is_clipped_at_borders(tmp_path):
    s = new_session(tmp_path)
    height, width = SHAPE
    quarter = sum(1 for y in range(4) for x in range(4) if y * y + x * x <= 9)
    assert s.paint(0, 0, 3, 1) == quarter
    assert s.paint(height - 1, width - 1, 3, 1) == quarter
    assert s.paint(-1, 12, 2, 1) > 0, "a center outside the image still paints the part that is inside"
    assert s.paint(-10, -10, 2, 1) == 0, "a disk entirely outside the image changes nothing"
    assert s.paint(height + 10, 5, 2, 1) == 0
    assert s.mask.shape == SHAPE


def test_paint_rejects_bad_arguments(tmp_path):
    s = new_session(tmp_path)
    with pytest.raises(ValueError):
        s.paint(5, 5, 2, 2)
    with pytest.raises(ValueError):
        s.paint(5, 5, -1, 1)
    with pytest.raises(ValueError):
        s.paint(float("nan"), 5, 2, 1)
    with pytest.raises(ValueError):
        s.paint(5, 5, float("inf"), 1)
    assert s.n_edits == 0 and not s.dirty


def test_session_arrays_are_read_only_views(tmp_path):
    s = new_session(tmp_path)
    for array in (s.image, s.prediction, s.mask, s.gt):
        with pytest.raises(ValueError):
            array[0, 0] = 1
    assert s.image.dtype == np.float32 and s.image.shape == SHAPE
    assert 0.0 <= float(s.image.min()) and float(s.image.max()) <= 1.0
    assert s.mask.dtype == np.uint8 and set(np.unique(s.mask)) == {0, 1}
    assert new_session(tmp_path / "other", with_gt=False).gt is None


# Undo and redo


def test_undo_and_redo_round_trip(tmp_path):
    s = new_session(tmp_path)
    start = s.mask.copy()
    s.paint(16, 16, 3, 1)
    after_paint = s.mask.copy()
    s.paint(16, 16, 1, 0)
    after_erase = s.mask.copy()

    assert s.can_undo and not s.can_redo
    assert s.undo() and np.array_equal(s.mask, after_paint)
    assert s.undo() and np.array_equal(s.mask, start)
    assert not s.undo() and not s.can_undo
    assert s.n_edits == 0 and not s.dirty

    assert s.redo() and np.array_equal(s.mask, after_paint)
    assert s.redo() and np.array_equal(s.mask, after_erase)
    assert not s.redo()
    assert s.n_edits == 2 and s.dirty


def test_a_new_edit_clears_the_redo_stack(tmp_path):
    s = new_session(tmp_path)
    s.paint(16, 16, 2, 1)
    assert s.undo() and s.can_redo
    s.paint(18, 5, 2, 1)
    assert not s.can_redo and not s.redo()
    assert s.n_edits == 1


def test_undo_history_is_capped(tmp_path):
    s = new_session(tmp_path, max_undo=3)
    columns = [2, 8, 14, 20, 26]
    for col in columns:
        assert s.paint(16, col, 0, 1) == 1
    assert s.n_edits == 5
    undone = 0
    while s.undo():
        undone += 1
    assert undone == 3, "only the last three edits can be undone"
    assert [int(s.mask[16, c]) for c in columns] == [1, 1, 0, 0, 0]
    assert s.n_edits == 2


def test_undo_can_be_disabled_and_cap_is_validated(tmp_path):
    s = new_session(tmp_path, max_undo=0)
    s.paint(16, 16, 2, 1)
    assert not s.can_undo and not s.undo()
    with pytest.raises(ValueError):
        new_session(tmp_path / "other", max_undo=-1)


def test_history_stores_diffs_not_array_copies(tmp_path):
    s = new_session(tmp_path)
    changed = s.paint(16, 16, 1, 1)
    diff = s._s.undo[-1]  # white-box check of the stored edit
    assert changed == 5
    assert diff.index.size == diff.old.size == diff.new.size == changed < s.mask.size
    assert set(np.unique(diff.old)) == {0} and set(np.unique(diff.new)) == {1}


def test_history_is_per_case(tmp_path):
    s = new_session(tmp_path)
    s.paint(16, 16, 2, 1)
    assert s.can_undo
    s.next()
    assert not s.can_undo and not s.can_redo
    s.prev()
    assert not s.can_undo, "history does not come back with the case"


# apply_mask


def test_apply_mask_records_the_diff_and_ignores_no_ops(tmp_path):
    s = new_session(tmp_path)
    start = s.mask.copy()
    assert s.apply_mask(s.mask.copy()) == 0
    assert not s.can_undo and s.n_edits == 0 and not s.dirty

    edited = start.copy()
    edited[16, 10:15] = 1  # five pixels added
    edited[5, 6] = 0  # one pixel removed
    assert s.apply_mask(edited) == 6
    assert np.array_equal(s.mask, edited) and s.n_edits == 1 and s.dirty
    edited[0, 0] = 1  # later changes to the caller's array do not leak into the session
    assert s.mask[0, 0] == 0

    assert s.undo() and np.array_equal(s.mask, start)
    assert s.redo() and s.mask[16, 12] == 1


def test_apply_mask_treats_positive_values_as_vessel(tmp_path):
    s = new_session(tmp_path)
    edited = s.mask.copy()
    edited[16, 10:15] = 1
    as_labels = edited.astype(np.int32) * 7  # for example label 7 painted in a viewer
    assert s.apply_mask(as_labels) == 5
    assert set(np.unique(s.mask)) == {0, 1}
    assert s.apply_mask(edited.astype(bool)) == 0
    assert s.apply_mask(edited.astype(np.float64) * 0.5) == 0


def test_apply_mask_validates_shape(tmp_path):
    s = new_session(tmp_path)
    for bad in (np.zeros((3, 3)), np.zeros(SHAPE + (1,)), np.zeros(SHAPE[::-1]), np.zeros(SHAPE[0] * SHAPE[1])):
        with pytest.raises(ValueError):
            s.apply_mask(bad)
    assert s.n_edits == 0


# Status, notes, dirty flag


def test_set_status_validates_and_marks_dirty(tmp_path):
    s = new_session(tmp_path)
    assert STATUSES == ("unreviewed", "accepted", "rejected")
    assert s.status == "unreviewed" and not s.dirty
    with pytest.raises(ValueError):
        s.set_status("maybe")
    assert s.status == "unreviewed" and not s.dirty

    s.set_status("rejected", "  blurry\nframe ")
    assert s.status == "rejected" and s.note == "blurry frame" and s.dirty
    s.set_status("unreviewed")
    assert s.status == "unreviewed" and s.note == ""
    assert not s.dirty, "back to the loaded state means nothing is unsaved"

    s.note = "just a note"
    assert s.dirty


def test_dirty_follows_the_mask_not_the_history(tmp_path):
    s = new_session(tmp_path)
    s.paint(16, 16, 2, 1)
    assert s.dirty
    s.undo()
    assert not s.dirty
    s.redo()
    s.save()
    assert not s.dirty
    s.paint(16, 16, 2, 0)
    assert s.dirty


# Navigation


def test_navigation_saves_only_when_something_changed(tmp_path):
    s = new_session(tmp_path, n=3)
    out = tmp_path / "out"
    assert not s.prev() and s.index == 0
    assert s.next() and s.index == 1 and s.next() and s.index == 2
    assert not out.exists(), "moving around without changes writes nothing"
    assert s.goto(2) is None and s.index == 2

    s.goto(0)
    s.paint(16, 16, 2, 1)
    assert s.next() and s.index == 1
    assert (out / "masks" / "case0.png").is_file()
    assert [row["image_id"] for row in read_log(out)] == ["case0"]
    assert not s.dirty


def test_navigation_at_the_ends_returns_false_but_still_saves(tmp_path):
    s = new_session(tmp_path, n=2)
    out = tmp_path / "out"
    s.set_status("rejected")
    assert not s.prev() and s.index == 0
    assert read_log(out)[0]["status"] == "rejected"

    s.goto(1)
    s.set_status("accepted")
    assert not s.next() and s.index == 1
    assert {row["image_id"]: row["status"] for row in read_log(out)} == {"case0": "rejected", "case1": "accepted"}


def test_goto_validates_the_index_and_saves_first(tmp_path):
    s = new_session(tmp_path, n=3)
    for bad in (-1, 3, 99):
        with pytest.raises(IndexError):
            s.goto(bad)
    assert s.index == 0
    s.set_status("accepted")
    s.goto(np.int64(2))
    assert s.index == 2 and s.item.image_id == "case2"
    assert (tmp_path / "out" / "masks" / "case0.png").is_file()


def test_a_bad_file_leaves_the_session_on_the_current_case(tmp_path):
    s = new_session(tmp_path, n=2)
    s.paint(16, 16, 2, 1)
    s.items[1].image_path.write_text("this is not a PNG")
    with pytest.raises(OSError):
        s.next()
    assert s.index == 0 and not s.dirty, "the case was saved, then the failed load changed nothing"
    assert s.mask[16, 16] == 1 and (tmp_path / "out" / "masks" / "case0.png").is_file()


# Saving


def test_save_writes_mask_png_and_log_row(tmp_path):
    s = new_session(tmp_path, reviewer="kp")
    out = tmp_path / "out"
    s.paint(16, 16, 3, 1)  # 29 pixels added
    s.paint(5, 10, 1, 0)  # 5 pixels removed from the bar
    s.set_status("accepted", "good enough")
    path = s.save()

    assert path == out / "masks" / "case0.png" == s.corrected_mask_path
    image = Image.open(path)
    assert image.mode == "L" and image.size == SHAPE[::-1]
    array = np.asarray(image)
    assert set(np.unique(array)) == {0, 255}
    assert np.array_equal(array > 0, s.mask > 0)

    header = (out / "review_log.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(LOG_COLUMNS)
    (row,) = read_log(out)
    assert row["image_id"] == "case0" and row["status"] == "accepted" and row["note"] == "good enough"
    assert row["reviewer"] == "kp"
    assert (int(row["pixels_added"]), int(row["pixels_removed"]), int(row["n_edits"])) == (29, 5, 2)
    mask_size = PRED_AREA + 29 - 5
    assert float(row["dice_vs_prediction"]) == pytest.approx(2 * (PRED_AREA - 5) / (PRED_AREA + mask_size), abs=1e-6)
    saved_at = datetime.fromisoformat(row["saved_at_utc"])
    assert saved_at.utcoffset() == timedelta(0)
    assert abs(datetime.now(saved_at.tzinfo) - saved_at) < timedelta(minutes=5)
    assert row["image_path"].endswith("item0_image.png") and row["pred_path"].endswith("item0_pred.png")
    assert row["corrected_mask_path"].endswith("masks/case0.png")
    assert not s.dirty
    assert sorted(p.name for p in out.rglob("*") if p.is_file()) == ["case0.png", "review_log.csv"], "no temp files left"


def test_dice_of_two_empty_masks_is_one(tmp_path):
    s = new_session(tmp_path, empty_pred=True)
    assert s.stats() == {"pixels_added": 0, "pixels_removed": 0, "dice_vs_prediction": 1.0}
    s.save()
    assert float(read_log(tmp_path / "out")[0]["dice_vs_prediction"]) == 1.0
    s.paint(10, 10, 2, 1)
    assert s.stats()["dice_vs_prediction"] == 0.0, "one empty and one non-empty mask do not overlap"
    assert dice_score(np.zeros((3, 3)), np.zeros((3, 3))) == 1.0
    assert dice_score(np.ones((3, 3)), np.ones((3, 3))) == 1.0


def test_log_keeps_one_row_per_image_and_other_rows(tmp_path):
    s = new_session(tmp_path, n=2)
    out = tmp_path / "out"
    s.set_status("accepted")
    s.save()
    s.set_status("rejected", "second thoughts")
    s.save()
    rows = read_log(out)
    assert len(rows) == 1 and rows[0]["status"] == "rejected" and rows[0]["note"] == "second thoughts"

    s.goto(1)
    s.set_status("accepted")
    s.save()
    assert [row["image_id"] for row in read_log(out)] == ["case0", "case1"]

    # A row written by someone else in the meantime is kept, and rows stay in place.
    stranger = dict(zip(LOG_COLUMNS, ["other"] + [""] * (len(LOG_COLUMNS) - 1)))
    with open(out / "review_log.csv", "a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=LOG_COLUMNS, lineterminator="\n").writerow(stranger)
    s.goto(0)
    s.paint(16, 16, 2, 1)
    s.save()
    rows = read_log(out)
    assert [row["image_id"] for row in rows] == ["case0", "case1", "other"]
    assert int(rows[0]["pixels_added"]) == disk_pixels(2)


def test_a_failed_save_keeps_the_old_files_and_the_unsaved_flag(tmp_path, monkeypatch):
    s = new_session(tmp_path)
    out = tmp_path / "out"
    s.set_status("accepted")
    s.save()
    old_log = (out / "review_log.csv").read_bytes()
    old_mask = (out / "masks" / "case0.png").read_bytes()

    s.paint(16, 16, 2, 1)
    s.set_status("rejected")

    def boom(src, dst):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(session_mod.os, "replace", boom)
        with pytest.raises(OSError):
            s.save()
    assert (out / "review_log.csv").read_bytes() == old_log
    assert (out / "masks" / "case0.png").read_bytes() == old_mask
    assert not list(out.rglob("*.tmp"))
    assert s.dirty and s.status == "rejected"
    s.save()
    assert read_log(out)[0]["status"] == "rejected"


def test_save_paths_in_the_log_are_relative_to_the_repo_root(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "ROOT", tmp_path)
    s = new_session(tmp_path)
    s.save()
    (row,) = read_log(tmp_path / "out")
    assert row["image_path"] == "inputs/item0_image.png"
    assert row["pred_path"] == "inputs/item0_pred.png"
    assert row["corrected_mask_path"] == "out/masks/case0.png"


# Resuming and summary


def test_resume_from_saved_output(tmp_path):
    first = new_session(tmp_path, n=2, reviewer="kp")
    first.paint(16, 16, 3, 1)
    first.set_status("rejected", "cut off")
    assert first.next()
    first.set_status("accepted")
    first.save()
    saved_mask = np.asarray(Image.open(tmp_path / "out" / "masks" / "case0.png")) > 0

    again = ReviewSession(first.items, tmp_path / "out", reviewer="kp")
    assert again.index == 0 and again.status == "rejected" and again.note == "cut off"
    assert np.array_equal(again.mask > 0, saved_mask)
    assert not np.array_equal(again.mask, again.prediction)
    assert again.n_edits == 1 and not again.dirty and not again.can_undo

    again.paint(20, 5, 2, 1)
    assert again.n_edits == 2, "edit counts add up across sessions"
    again.save()
    (row0, row1) = read_log(tmp_path / "out")
    assert int(row0["pixels_added"]) == disk_pixels(3) + disk_pixels(2), "differences are measured against the original prediction"
    assert row1["status"] == "accepted"
    assert again.next() is True and again.status == "accepted"


def test_resume_ignores_an_unusable_saved_mask(tmp_path):
    items = make_items(tmp_path, n=1)
    save_png(tmp_path / "out" / "masks" / "case0.png", np.zeros((5, 5), np.uint8))
    with pytest.warns(UserWarning, match="shape"):
        s = ReviewSession(items, tmp_path / "out")
    assert np.array_equal(s.mask, s.prediction)

    (tmp_path / "out" / "masks" / "case0.png").write_text("garbage")
    with pytest.warns(UserWarning, match="unreadable"):
        s = ReviewSession(items, tmp_path / "out")
    assert np.array_equal(s.mask, s.prediction)


def test_an_unreadable_log_is_an_error_not_an_overwrite(tmp_path):
    items = make_items(tmp_path, n=1)
    out = tmp_path / "out"
    out.mkdir()
    (out / "review_log.csv").write_bytes(b"\xff\xfe\x00 not utf-8 \x80\x81")
    with pytest.raises(ValueError, match="review log"):
        ReviewSession(items, out)
    assert (out / "review_log.csv").read_bytes().startswith(b"\xff\xfe")


def test_summary_counts_statuses_edits_and_saves(tmp_path):
    s = new_session(tmp_path, n=4)
    assert s.summary() == {"n_items": 4, "unreviewed": 4, "accepted": 0, "rejected": 0, "edited": 0, "saved": 0}

    s.paint(16, 16, 2, 1)
    s.set_status("accepted")
    s.next()  # saves case0
    s.set_status("rejected")
    s.next()  # saves case1, which was not edited
    assert s.index == 2
    assert s.summary() == {"n_items": 4, "unreviewed": 2, "accepted": 1, "rejected": 1, "edited": 1, "saved": 2}

    s.set_status("accepted")  # unsaved, but the open case counts from its live state
    assert s.summary()["accepted"] == 2 and s.summary()["saved"] == 2
    s.save()
    assert s.summary()["saved"] == 3

    assert ReviewSession(s.items, tmp_path / "out").summary() == s.summary()


# Constructing a session


def test_session_validates_its_inputs(tmp_path):
    items = make_items(tmp_path, n=2)
    with pytest.raises(ValueError):
        ReviewSession([], tmp_path / "out")
    with pytest.raises(TypeError):
        ReviewSession(["case0"], tmp_path / "out")  # type: ignore[list-item]

    missing = [items[0], ReviewItem("ghost", tmp_path / "nope.png", items[1].pred_path)]
    with pytest.raises(FileNotFoundError, match="nope.png"):
        ReviewSession(missing, tmp_path / "out")

    wrong = tmp_path / "wrong_pred.png"
    save_png(wrong, np.zeros((5, 5), np.uint8))
    with pytest.raises(ValueError, match="prediction has shape"):
        ReviewSession([ReviewItem("odd", items[0].image_path, wrong)], tmp_path / "out")
    with pytest.raises(ValueError, match="ground truth has shape"):
        ReviewSession([ReviewItem("odd", items[0].image_path, items[0].pred_path, wrong)], tmp_path / "out")


def test_items_can_be_given_as_strings_and_any_iterable(tmp_path):
    items = make_items(tmp_path, n=2)
    loose = (ReviewItem(i.image_id, str(i.image_path), str(i.pred_path), str(i.gt_path)) for i in items)
    s = ReviewSession(loose, str(tmp_path / "out"))
    assert s.n_items == 2 and isinstance(s.item.image_path, Path) and isinstance(s.out_dir, Path)


# Image and mask decoding


def test_load_image_normalizes_8_bit_16_bit_and_rgb(tmp_path):
    save_png(tmp_path / "g8.png", np.array([[0, 128, 255]], np.uint8))
    out = load_image(tmp_path / "g8.png")
    assert out.dtype == np.float32 and out.shape == (1, 3)
    assert out[0].tolist() == pytest.approx([0.0, 128 / 255, 1.0])

    save_png(tmp_path / "g16.png", np.array([[0, 32768, 65535]], np.uint16))
    out = load_image(tmp_path / "g16.png")
    assert out.dtype == np.float32
    assert out[0].tolist() == pytest.approx([0.0, 32768 / 65535, 1.0], abs=1e-6)

    rgb = np.zeros((1, 3, 3), np.uint8)
    rgb[0, 1] = (255, 255, 255)
    rgb[0, 2] = (255, 0, 0)
    save_png(tmp_path / "rgb.png", rgb)
    out = load_image(tmp_path / "rgb.png")
    assert out.ndim == 2 and out.dtype == np.float32
    assert out[0, 0] == 0.0 and out[0, 1] == pytest.approx(1.0) and 0.0 < out[0, 2] < 1.0

    rgba = np.zeros((1, 2, 4), np.uint8)
    rgba[0, :, 3] = 255
    rgba[0, 1, :3] = 255
    save_png(tmp_path / "rgba.png", rgba)
    assert load_image(tmp_path / "rgba.png")[0].tolist() == pytest.approx([0.0, 1.0])


def test_load_mask_treats_any_positive_value_as_vessel(tmp_path):
    save_png(tmp_path / "l.png", np.array([[0, 1, 2, 255]], np.uint8))
    assert load_mask(tmp_path / "l.png").tolist() == [[0, 1, 1, 1]]

    save_png(tmp_path / "l16.png", np.array([[0, 1, 65535]], np.uint16))
    assert load_mask(tmp_path / "l16.png").tolist() == [[0, 1, 1]]

    rgb = np.zeros((1, 3, 3), np.uint8)
    rgb[0, 1] = (255, 0, 0)
    rgb[0, 2] = (0, 0, 1)
    save_png(tmp_path / "rgb.png", rgb)
    assert load_mask(tmp_path / "rgb.png").tolist() == [[0, 1, 1]]

    rgba = np.zeros((1, 2, 4), np.uint8)
    rgba[0, :, 3] = 255  # opaque black background must stay background
    rgba[0, 1, 1] = 200
    save_png(tmp_path / "rgba.png", rgba)
    assert load_mask(tmp_path / "rgba.png").tolist() == [[0, 1]]

    palette = Image.new("P", (3, 1))
    palette.putdata([0, 1, 2])
    palette.putpalette([0, 0, 0, 255, 0, 0, 0, 255, 0])
    palette.save(tmp_path / "p.png")
    assert load_mask(tmp_path / "p.png").tolist() == [[0, 1, 1]]

    out = load_mask(tmp_path / "l.png")
    assert out.dtype == np.uint8 and set(np.unique(out)) <= {0, 1}


# Manifests


def write_csv(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.write_text(text, encoding=encoding)
    return path


def test_load_manifest_resolves_relative_paths_and_sorts_by_priority(tmp_path):
    root = tmp_path / "root"
    absolute = tmp_path / "elsewhere" / "b.png"
    manifest = write_csv(
        tmp_path / "m.csv",
        "image_id,image_path,pred_path,gt_path,priority\n"
        "a,imgs/a.png,preds/a.png,gts/a.png,0.2\n"
        f"b,{absolute},preds/b.png,,0.9\n"
        "c,imgs/c.png,preds/c.png,,\n"
        "d,imgs/d.png,preds/d.png,,0.9\n"
        "e,imgs/e.png,preds/e.png,,1e-3\n",
    )
    items = load_manifest(manifest, root=root)
    assert [i.image_id for i in items] == ["b", "d", "a", "e", "c"], "highest first, ties in file order, blanks last"
    by_id = {i.image_id: i for i in items}
    assert by_id["b"].image_path == absolute, "absolute paths are kept"
    assert by_id["b"].pred_path == root / "preds" / "b.png"
    assert by_id["a"].image_path == root / "imgs" / "a.png"
    assert by_id["a"].gt_path == root / "gts" / "a.png" and by_id["b"].gt_path is None
    assert by_id["c"].priority is None and by_id["e"].priority == pytest.approx(0.001)
    assert all(isinstance(i.image_path, Path) for i in items)


def test_load_manifest_without_priority_keeps_file_order(tmp_path):
    manifest = write_csv(tmp_path / "m.csv", "image_id,image_path,pred_path\nz,z.png,zp.png\na,a.png,ap.png\n")
    items = load_manifest(manifest, root=tmp_path)
    assert [i.image_id for i in items] == ["z", "a"]
    assert items[0].gt_path is None and items[0].priority is None


def test_load_manifest_tolerates_bom_spaces_and_blank_lines(tmp_path):
    manifest = write_csv(
        tmp_path / "m.csv",
        "image_id , image_path,pred_path\n\n x , x.png , xp.png \n,,\n",
        encoding="utf-8-sig",
    )
    (item,) = load_manifest(manifest, root=tmp_path)
    assert item.image_id == "x" and item.image_path == tmp_path / "x.png"


@pytest.mark.parametrize(
    "text, message",
    [
        ("image_id,image_path\nx,x.png\n", "pred_path"),
        ("", "image_id"),
        ("image_id,image_path,pred_path\n,x.png,xp.png\n", "image_id is empty"),
        ("image_id,image_path,pred_path\nx,,xp.png\n", "required"),
        ("image_id,image_path,pred_path,priority\nx,x.png,xp.png,high\n", "not a number"),
        ("image_id,image_path,pred_path,priority\nx,x.png,xp.png,nan\n", "not a number"),
    ],
)
def test_load_manifest_rejects_bad_files(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        load_manifest(write_csv(tmp_path / "m.csv", text), root=tmp_path)


def test_manifest_round_trip(tmp_path):
    items = [
        ReviewItem("x", tmp_path / "x.png", tmp_path / "xp.png", tmp_path / "xg.png", 0.25),
        ReviewItem("y", tmp_path / "y.png", tmp_path / "yp.png", None, None),
    ]
    target = tmp_path / "nested" / "dir" / "m.csv"
    write_manifest(target, items)
    assert load_manifest(target, root=tmp_path) == items

    relative = [ReviewItem("r", Path("imgs/r.png"), Path("preds/r.png"))]
    write_manifest(tmp_path / "rel.csv", relative)
    assert "imgs/r.png" in (tmp_path / "rel.csv").read_text(), "paths are written as given"
    (back,) = load_manifest(tmp_path / "rel.csv", root=tmp_path)
    assert back.image_path == tmp_path / "imgs" / "r.png"


# Image ids as file names


@pytest.mark.parametrize(
    "hostile",
    ["../../etc/passwd", "a/b", "a\\b", "..", "...", " spaced id ", "name\x00.png", "caf\u00e9", ".hidden", "x" * 400, "C:\\temp\\x"],
)
def test_safe_filename_never_escapes_the_folder(hostile):
    name = safe_filename(hostile)
    assert name and name == Path(name).name
    assert "/" not in name and "\\" not in name and ".." not in name and "\x00" not in name
    assert not name.startswith(".") and len(name) <= 150


def test_safe_filename_keeps_ordinary_ids_and_rejects_empty_ones():
    assert safe_filename("case_01-a.v2") == "case_01-a.v2"
    assert safe_filename("12") == "12"
    for empty in ("", "   "):
        with pytest.raises(ValueError):
            safe_filename(empty)


def test_hostile_ids_are_saved_inside_the_masks_folder(tmp_path):
    items = make_items(tmp_path, n=2, ids=["../../evil", "a/b"])
    s = ReviewSession(items, tmp_path / "out")
    s.paint(16, 16, 2, 1)
    path = s.save()
    assert path.parent == tmp_path / "out" / "masks"
    s.goto(1)
    s.set_status("accepted")
    s.save()
    created = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file() and "inputs" not in p.parts}
    assert all(name.startswith("out/") for name in created), created
    assert [row["image_id"] for row in read_log(tmp_path / "out")] == ["../../evil", "a/b"], "the log keeps the real ids"


def test_duplicate_ids_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        ReviewSession(make_items(tmp_path, n=2, ids=["same", "same"]), tmp_path / "out")
    with pytest.raises(ValueError, match="a_b"):
        ReviewSession(make_items(tmp_path / "x", n=2, ids=["a/b", "a_b"]), tmp_path / "out")


# Demo data


def test_make_demo_items_writes_cases_and_a_manifest(tmp_path):
    demo = tmp_path / "demo"
    items = make_demo_items(demo, 4, 3, tiny_pair)
    assert len(items) == 4
    assert sorted(i.image_id for i in items) == ["demo_000", "demo_001", "demo_002", "demo_003"]
    priorities = [i.priority for i in items]
    assert priorities == sorted(priorities, reverse=True), "worst predictions come first"

    for item in items:
        index = int(item.image_id.split("_")[1])
        _image, truth = tiny_pair(np.random.default_rng([3, index]))
        image = load_image(item.image_path)
        pred, gt = load_mask(item.pred_path), load_mask(item.gt_path)
        assert image.shape == pred.shape == gt.shape == (40, 48)
        assert np.array_equal(gt, truth), "the ground truth is the true mask from pair_fn"
        assert not np.array_equal(pred, gt), "the prediction is deliberately imperfect"
        assert item.priority == pytest.approx(round(1.0 - dice_score(pred, gt), 4))
        assert np.max(np.abs(image - _image)) <= 0.5 / 255 + 1e-6, "image stored as 8 bit"
        assert np.asarray(Image.open(item.pred_path)).max() == 255

    manifest = demo / "manifest.csv"
    assert manifest.is_file()
    assert load_manifest(manifest) == items

    s = ReviewSession(items, tmp_path / "review")
    assert s.n_items == 4 and s.gt is not None


def test_make_demo_items_is_reproducible_and_seeded(tmp_path):
    def snapshot(folder: Path):
        return {p.name: p.read_bytes() for p in sorted((folder / "predictions").glob("*.png"))}

    make_demo_items(tmp_path / "a", 3, 5, tiny_pair)
    make_demo_items(tmp_path / "b", 3, 5, tiny_pair)
    make_demo_items(tmp_path / "c", 3, 6, tiny_pair)
    assert snapshot(tmp_path / "a") == snapshot(tmp_path / "b")
    assert snapshot(tmp_path / "a") != snapshot(tmp_path / "c")

    # A case does not depend on how many cases were requested.
    make_demo_items(tmp_path / "d", 1, 5, tiny_pair)
    assert (tmp_path / "d" / "predictions" / "demo_000.png").read_bytes() == snapshot(tmp_path / "a")["demo_000.png"]


def test_make_demo_items_validates(tmp_path):
    with pytest.raises(ValueError):
        make_demo_items(tmp_path / "d", 0, 0, tiny_pair)
    with pytest.raises(ValueError, match="same shape"):
        make_demo_items(tmp_path / "d", 1, 0, lambda rng: (np.zeros((5, 5), np.float32), np.zeros((4, 4), np.uint8)))
    with pytest.raises(ValueError, match="2D"):
        make_demo_items(tmp_path / "d", 1, 0, lambda rng: (np.zeros((5, 5, 3), np.float32), np.zeros((5, 5, 3), np.uint8)))


def test_demo_manifest_uses_repo_relative_paths_when_inside_the_root(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "ROOT", tmp_path)
    items = make_demo_items(tmp_path / "demo", 2, 0, tiny_pair)
    text = (tmp_path / "demo" / "manifest.csv").read_text()
    assert "demo/images/demo_000.png" in text and str(tmp_path) not in text
    assert all(i.image_path.is_absolute() and i.image_path.is_file() for i in items)


def test_perturb_mask_always_changes_the_mask():
    _image, truth = tiny_pair(np.random.default_rng(0))
    cases = {"bars": truth, "empty": np.zeros((30, 30), np.uint8), "full": np.ones((30, 30), np.uint8)}
    added = removed = 0
    for name, mask in cases.items():
        for seed in range(25):
            out = perturb_mask(mask, np.random.default_rng(seed))
            assert out.shape == mask.shape and out.dtype == np.uint8 and set(np.unique(out)) <= {0, 1}
            assert not np.array_equal(out, mask), (name, seed)
            if name == "bars":
                added += int(np.count_nonzero((out == 1) & (mask == 0)))
                removed += int(np.count_nonzero((out == 0) & (mask == 1)))
    assert added > 0 and removed > 0, "both false blobs and deleted patches appear"
    assert np.array_equal(perturb_mask(truth, np.random.default_rng(4)), perturb_mask(truth, np.random.default_rng(4)))


def test_perturb_mask_does_not_erase_thin_vessels():
    thin = np.zeros((40, 40), np.uint8)
    thin[20, 5:35] = 1
    for seed in range(20):
        out = perturb_mask(thin, np.random.default_rng(seed))
        assert dice_score(out, thin) > 0.2, "a one pixel line is thickened, not eroded away"


# Design rules


def test_session_module_has_no_gui_imports():
    code = (
        "import sys, angiolens.review.session\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in {'napari', 'qtpy', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'vispy'})\n"
        "print(','.join(bad))\n"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"GUI modules imported by session.py: {result.stdout.strip()}"


def test_session_module_never_loads_serialized_python_objects():
    source = Path(session_mod.__file__).read_text(encoding="utf-8")
    for forbidden in ("pickle", "allow_pickle", "np.load", "numpy.load", "marshal", "shelve", "joblib"):
        assert forbidden not in source, forbidden


def prose_of(path: Path) -> list[str]:
    """Comments and docstrings of a Python file."""
    source = path.read_text(encoding="utf-8")
    texts = [tok.string for tok in tokenize.generate_tokens(io.StringIO(source).readline) if tok.type == tokenize.COMMENT]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                texts.append(doc)
    return texts


@pytest.mark.parametrize("path", [Path(session_mod.__file__), Path(__file__)], ids=["session.py", "this file"])
def test_comments_and_docstrings_follow_the_writing_rules(path):
    source = path.read_text(encoding="utf-8")
    assert EM_DASH not in source and EN_DASH not in source, "no em or en dashes anywhere"
    assert ARROW not in source, "no arrow characters anywhere"
    for text in prose_of(path):
        assert " -- " not in text and "->" not in text, text
    for word in BANNED_WORDS:  # tool and vendor names stay out of the code base
        flags = 0 if word == "A" + "I" else re.IGNORECASE
        assert not re.search(rf"\b{word}\b", source, flags), word
