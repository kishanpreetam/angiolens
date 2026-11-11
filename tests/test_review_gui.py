"""Tests for the napari review viewer.

Only test_viewer_end_to_end builds a viewer. It is hidden (never shown) and needs a
desktop session. The other tests in this file do not create any window.
"""

from __future__ import annotations

import ast
import csv
import inspect
import io
import os
import re
import subprocess
import sys
import tokenize
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("napari")
try:
    import qtpy.QtWidgets  # noqa: F401
except Exception as exc:  # any failure to load Qt means there is no GUI to test
    pytest.skip(f"Qt bindings are not available: {exc}", allow_module_level=True)

from napari.utils.key_bindings import KeymapHandler, coerce_keybinding  # noqa: E402
from napari.utils.notifications import notification_manager  # noqa: E402
from napari.utils.shortcuts import default_shortcuts  # noqa: E402

from angiolens.review import napari_app as app  # noqa: E402
from angiolens.review.session import ReviewSession, make_demo_items  # noqa: E402

PLAIN_KEYS = [
    app.KEY_PREVIOUS,
    app.KEY_NEXT,
    app.KEY_ACCEPT,
    app.KEY_REJECT,
    app.KEY_SAVE,
    app.KEY_PREDICTION,
    app.KEY_GROUND_TRUTH,
]
EM_DASH, EN_DASH, ARROW = chr(0x2014), chr(0x2013), chr(0x2192)  # kept out of the source on purpose
# Words that must not appear in the code base, spelled in pieces so this file does not contain them.
BANNED_WORDS = ["Cl" + "aude", "Anthr" + "opic", "assis" + "tant", "A" + "I"]


def stand_in_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Stand-in for the real generator: a noisy 64 x 64 image with one diagonal vessel."""
    size = 64
    image = np.clip(0.6 + 0.05 * rng.standard_normal((size, size)), 0, 1).astype(np.float32)
    mask = np.zeros((size, size), np.uint8)
    rows = np.arange(8, 56)
    mask[rows, (10 + rows * 0.6).astype(int)] = 1
    mask[rows, (11 + rows * 0.6).astype(int)] = 1
    image[mask > 0] -= 0.3
    return np.clip(image, 0, 1), mask


def free_spots(mask: np.ndarray, count: int) -> list[tuple[int, int]]:
    """Pixels, at least five apart, whose 3 x 3 neighborhood is empty in mask."""
    spots = [
        (r, c)
        for c in (45, 55)
        for r in range(16, 50, 5)
        if np.count_nonzero(mask[r - 1 : r + 2, c - 1 : c + 2]) == 0
    ]
    assert len(spots) >= count, "the demo image has no room for the test strokes"
    return spots[:count]


def make_session(tmp_path: Path, n: int = 2) -> ReviewSession:
    items = make_demo_items(tmp_path / "demo", n, 0, stand_in_pair)
    return ReviewSession(items, tmp_path / "out", reviewer="tester")


def read_log(tmp_path: Path) -> list[dict[str, str]]:
    with open(tmp_path / "out" / "review_log.csv", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def gui_unavailable_reason() -> str | None:
    """Why a hidden viewer cannot be built here, or None when it can."""
    if os.environ.get("ANGIOLENS_SKIP_GUI") == "1":
        return "ANGIOLENS_SKIP_GUI is set"
    platform = os.environ.get("QT_QPA_PLATFORM", "").lower()
    if platform in {"offscreen", "minimal", "minimalegl", "vnc"}:
        return f"QT_QPA_PLATFORM={platform} has no OpenGL, so napari would crash"
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return "no display is available"
    return None


# The viewer


@pytest.mark.gui
def test_viewer_end_to_end(tmp_path):
    reason = gui_unavailable_reason()
    if reason:
        pytest.skip(reason)

    session = make_session(tmp_path)
    first_id, second_id = (item.image_id for item in session.items)
    viewer = app.build_viewer(session)  # hidden: show defaults to False
    try:
        # Layers and dock widget
        layers = viewer.layers
        assert [layer.name for layer in layers] == ["image", "prediction", "ground truth", "corrected"]
        image, prediction, truth, corrected = (layers[name] for name in ("image", "prediction", "ground truth", "corrected"))
        assert type(image).__name__ == "Image" and all(type(x).__name__ == "Labels" for x in (prediction, truth, corrected))
        assert prediction.visible is False and prediction.editable is False
        assert truth.visible is False and truth.editable is False
        assert corrected.visible and corrected.editable and layers.selection.active is corrected
        assert corrected.mode == "paint" and corrected.brush_size == app.default_brush_size((64, 64))
        assert app.PANEL_NAME in viewer.window.dock_widgets
        panel = viewer.window.dock_widgets[app.PANEL_NAME]
        assert isinstance(panel, app.ReviewPanel) and panel.session is session

        # The layers show the session, and the corrected layer owns its own copy.
        assert np.array_equal(image.data, session.image)
        assert np.array_equal(prediction.data, session.prediction)
        assert np.array_equal(truth.data, session.gt)
        assert np.array_equal(corrected.data, session.mask)
        assert not np.shares_memory(corrected.data, session.mask)
        assert panel.title_label.text() == app.case_title(session) and f"1/2  {first_id}" == panel.title_label.text()
        assert "unreviewed" in panel.detail_label.text() and panel.detail_label.text().endswith("saved")
        assert not panel.previous_button.isEnabled() and panel.next_button.isEnabled()
        assert not panel.undo_button.isEnabled() and not panel.redo_button.isEnabled()
        assert panel.ground_truth_box.isEnabled() and panel.advance_box.isChecked()

        # A brush stroke reaches the session through the paint event.
        spots = free_spots(session.mask, 3)
        row, col = spots[0]
        before = int(session.mask.sum())
        corrected.paint((row, col), 1)
        assert session.n_edits == 1 and session.dirty
        assert session.mask[row, col] == 1 and 1 <= int(session.mask.sum()) - before <= 25
        assert np.array_equal(corrected.data, session.mask)
        assert "edits 1" in panel.detail_label.text() and panel.detail_label.text().endswith("unsaved")
        assert panel.undo_button.isEnabled() and not panel.redo_button.isEnabled()
        assert viewer.title.startswith(app.PANEL_NAME) and "unsaved" in viewer.title

        # Undo and redo go through the session, whichever layer is active.
        keys = KeymapHandler()
        keys.keymap_providers = [corrected, viewer]
        assert keys.press_key(app.KEY_UNDO)
        assert session.n_edits == 0 and session.mask[row, col] == 0 and session.can_redo and not session.dirty
        assert np.array_equal(corrected.data, session.mask)
        assert keys.press_key(app.KEY_REDO)
        assert session.n_edits == 1 and session.mask[row, col] == 1 and np.array_equal(corrected.data, session.mask)
        layers.selection.active = image
        keys.keymap_providers = [image, viewer]
        assert keys.press_key(app.KEY_UNDO) and session.n_edits == 0
        assert keys.press_key(app.KEY_REDO) and session.n_edits == 1
        layers.selection.active = corrected
        keys.keymap_providers = [corrected, viewer]

        # Plain keys are bound on the viewer, and a held key does not repeat.
        for key in PLAIN_KEYS:
            assert inspect.isgeneratorfunction(viewer.keymap[coerce_keybinding(key)]), key

        def press(key: str) -> None:
            assert keys.press_key(key), key
            keys.release_key(key)

        press(app.KEY_PREDICTION)
        press(app.KEY_GROUND_TRUTH)
        assert prediction.visible and truth.visible
        assert panel.prediction_box.isChecked() and panel.ground_truth_box.isChecked()
        press(app.KEY_PREDICTION)
        assert not prediction.visible and not panel.prediction_box.isChecked()
        panel.ground_truth_box.setChecked(False)
        assert not truth.visible

        # Accept: saved with its note, then the next case opens and every layer refreshes.
        panel.note_edit.setText("hole near the top")
        panel.accept_button.click()
        assert session.index == 1 and session.item.image_id == second_id and not session.dirty
        (row0,) = read_log(tmp_path)
        assert (row0["image_id"], row0["status"], row0["note"]) == (first_id, "accepted", "hole near the top")
        assert row0["reviewer"] == "tester" and int(row0["pixels_added"]) >= 1
        assert (tmp_path / "out" / "masks" / f"{first_id}.png").is_file()
        assert np.array_equal(image.data, session.image) and np.array_equal(prediction.data, session.prediction)
        assert np.array_equal(truth.data, session.gt) and np.array_equal(corrected.data, session.mask)
        assert panel.title_label.text() == f"2/2  {second_id}" and panel.note_edit.text() == ""
        assert panel.previous_button.isEnabled() and not panel.next_button.isEnabled()
        assert not panel.undo_button.isEnabled(), "history belongs to the case it was made in"

        # Reject with the keyboard on the last case: saved, and there is nowhere to go on to.
        press(app.KEY_REJECT)
        assert session.index == 1 and session.status == "rejected" and not session.dirty
        assert [r["status"] for r in read_log(tmp_path)] == ["accepted", "rejected"]
        assert panel.detail_label.text().startswith("rejected")

        # Previous brings back the saved edit, status and note.
        press(app.KEY_PREVIOUS)
        assert session.index == 0 and session.status == "accepted" and session.n_edits == 1
        assert corrected.data[row, col] == 1 and np.array_equal(corrected.data, session.mask)
        assert panel.note_edit.text() == "hole near the top"

        # With the box cleared, a decision stays on the same case.
        panel.advance_box.setChecked(False)
        panel.reject_button.click()
        assert session.index == 0 and session.status == "rejected" and not session.dirty
        assert [r["status"] for r in read_log(tmp_path)] == ["rejected", "rejected"]

        # Save key, and a stroke made right before leaving a case is not lost.
        corrected.paint(spots[1], 1)
        assert session.dirty
        press(app.KEY_SAVE)
        assert not session.dirty
        corrected.paint(spots[2], 1)
        press(app.KEY_NEXT)
        assert session.index == 1
        press(app.KEY_NEXT)  # already at the last case: nothing happens
        assert session.index == 1
        press(app.KEY_PREVIOUS)
        assert corrected.data[spots[2]] == 1 and corrected.data[spots[1]] == 1 and session.n_edits == 3

        # An unreadable file is reported to the reviewer, and the window stays on the open case.
        session.items[1].image_path.write_text("this is not a PNG")
        shown = len(notification_manager.records)
        shown_mask = corrected.data.copy()
        press(app.KEY_NEXT)
        assert session.index == 0 and np.array_equal(corrected.data, shown_mask)
        assert len(notification_manager.records) == shown + 1
        assert "UnidentifiedImageError" in notification_manager.records[-1].message
        assert panel.title_label.text() == app.case_title(session)

        # Screenshot helper: a real picture, not a black canvas.
        shot = app.save_screenshot(viewer, tmp_path / "shots" / "review.png")
        assert shot == tmp_path / "shots" / "review.png" and shot.is_file()
        picture = np.asarray(Image.open(shot).convert("RGB"))
        assert picture.shape[0] > 100 and picture.shape[1] > 100
        assert picture.std() > 5 and picture.max() > 100, "the canvas rendered"
    finally:
        viewer.close()


# Things that do not need a window


def test_shortcuts_do_not_clash_with_napari_defaults():
    taken = {str(coerce_keybinding(str(key))) for keys in default_shortcuts.values() for key in keys}
    assert len(set(PLAIN_KEYS)) == len(PLAIN_KEYS)
    for key in PLAIN_KEYS:
        assert str(coerce_keybinding(key)) not in taken, f"{key} is a napari default"


def test_module_docstring_lists_every_shortcut():
    doc = app.__doc__ or ""
    assert [key for key, _what in app.SHORTCUTS[:7]] == PLAIN_KEYS
    for key, what in app.SHORTCUTS:
        assert re.search(rf"(?m)^\s+{re.escape(key)}\s{{2,}}\S", doc), f"{key} is missing from the module docstring"
    assert "Ctrl/Cmd+Z" in doc and "Shift+Ctrl/Cmd+Z" in doc


def test_status_line_reports_the_state_of_the_open_case(tmp_path):
    session = make_session(tmp_path, n=3)
    item = session.item
    assert app.case_title(session) == f"1/3  {item.image_id}"
    assert app.case_detail(session) == f"unreviewed | priority {item.priority:.3g} | edits 0 | saved"
    assert app.status_line(session) == f"1/3  {item.image_id} | unreviewed | priority {item.priority:.3g} | edits 0 | saved"

    session.paint(30, 50, 2, 1)
    assert app.case_detail(session).endswith("edits 1 | unsaved")
    session.set_status("accepted")
    session.save()
    assert app.case_detail(session) == f"accepted | priority {item.priority:.3g} | edits 1 | saved"

    session.items[0].priority = None
    assert "priority" not in app.case_detail(session)


@pytest.mark.parametrize("shape, expected", [((8, 8), 3), ((64, 64), 3), ((512, 512), 5), ((1024, 768), 10), ((9000, 9000), 15)])
def test_default_brush_size_scales_with_the_image(shape, expected):
    assert app.default_brush_size(shape) == expected


def test_command_line_needs_exactly_one_source(tmp_path, capsys):
    for argv in ([], ["--demo", "2", "--manifest", "m.csv"], ["--demo", "0"], ["--demo", "2", "--seed", "-1"], ["--bogus"]):
        with pytest.raises(SystemExit) as stop:
            app.main(argv)
        assert stop.value.code == 2, argv
    capsys.readouterr()

    args = app._parse_args(["--manifest", "m.csv"])
    assert args.out == app.DEFAULT_OUT == app.ROOT / "data" / "processed" / "review"
    assert args.reviewer == "" and args.demo is None and args.manifest == Path("m.csv")
    args = app._parse_args(["--demo", "5", "--seed", "3", "--out", str(tmp_path), "--reviewer", "kp"])
    assert (args.demo, args.seed, args.out, args.reviewer, args.manifest) == (5, 3, tmp_path, "kp", None)


def test_command_line_fails_early_without_opening_a_window(tmp_path, monkeypatch, capsys):
    with pytest.raises(FileNotFoundError):
        app.main(["--manifest", str(tmp_path / "missing.csv")])

    monkeypatch.setitem(sys.modules, "angiolens.synth.generator", None)  # makes the import fail
    assert app.main(["--demo", "2", "--out", str(tmp_path)]) == 2
    assert "angiolens.synth.generator" in capsys.readouterr().err
    assert not (tmp_path / "demo").exists()


def test_module_runs_with_python_dash_m_and_help_is_clean():
    result = subprocess.run(
        [sys.executable, "-I", "-m", "angiolens.review.napari_app", "--help"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    for flag in ("--manifest", "--out", "--reviewer", "--demo"):
        assert flag in result.stdout
    assert "data/processed/review" in result.stdout
    for bad in (EM_DASH, EN_DASH, ARROW, "->"):
        assert bad not in result.stdout


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


@pytest.mark.parametrize("path", [Path(app.__file__), Path(__file__)], ids=["napari_app.py", "this file"])
def test_comments_and_docstrings_follow_the_writing_rules(path):
    source = path.read_text(encoding="utf-8")
    assert EM_DASH not in source and EN_DASH not in source, "no em or en dashes anywhere"
    assert ARROW not in source, "no arrow characters anywhere"
    for text in prose_of(path):
        assert " -- " not in text and "->" not in text, text
    for word in BANNED_WORDS:  # tool and vendor names stay out of the code base
        flags = 0 if word == "A" + "I" else re.IGNORECASE
        assert not re.search(rf"\b{word}\b", source, flags), word


def test_napari_is_imported_only_by_the_gui_module():
    package = Path(app.__file__).parent
    for path in package.glob("*.py"):
        if path.name == "napari_app.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
        } | {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        assert not imported & {"napari", "qtpy", "PyQt5", "PyQt6", "PySide2", "PySide6", "vispy", "app_model"}, path.name
