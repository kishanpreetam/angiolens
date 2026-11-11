"""Napari front end for reviewing and correcting vessel masks.

Run it from the repository root:

    python -m angiolens.review.napari_app --manifest review_manifest.csv --reviewer kp
    python -m angiolens.review.napari_app --demo 8

Layers, bottom to top:

    image          the X-ray in gray
    prediction     the model mask, read-only, low opacity, hidden at first
    ground truth   outline of the true mask, hidden at first, present only when
                   the cases have a ground truth
    corrected      the mask you edit; selected, starts in paint mode

Every finished brush stroke is committed to the ReviewSession, which owns the
edit history, the status of each case and everything that is written to disk.
Moving to another case saves the current one first.

Keyboard shortcuts. None of these is a napari default. The canvas must have
focus, so press Enter in the note box or click the canvas after typing a note:

    ,                  previous case
    .                  next case
    K                  accept the case, save it and go to the next one
    J                  reject the case, save it and go to the next one
    W                  save now
    H                  show or hide the prediction
    G                  show or hide the ground truth outline
    Ctrl/Cmd+Z         undo, through the session
    Shift+Ctrl/Cmd+Z   redo, through the session

Napari's own tool keys still work: 2 or P paint, 1 or E erase, 4 or F fill, and
the [ and ] keys change the brush size. A held key does not repeat, so holding K
cannot accept more than one case.

Unsaved changes are saved when you leave a case and when the command line app
exits. If you build the viewer yourself, call session.save() when you are done.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import napari
import numpy as np
from app_model.types import KeyCode, KeyMod
from napari.utils.colormaps import DirectLabelColormap
from napari.utils.notifications import show_error
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from angiolens.paths import ROOT
from angiolens.review.session import (
    ReviewSession,
    load_manifest,
    make_demo_items,
)

PANEL_NAME = "Angiolens review"
DEFAULT_OUT = ROOT / "data" / "processed" / "review"
HIDDEN_WINDOW_SIZE = (1280, 800)  # logical pixels, used when the viewer is built with show=False

LAYER_IMAGE = "image"
LAYER_PREDICTION = "prediction"
LAYER_GROUND_TRUTH = "ground truth"
LAYER_CORRECTED = "corrected"

KEY_PREVIOUS = ","
KEY_NEXT = "."
KEY_ACCEPT = "K"
KEY_REJECT = "J"
KEY_SAVE = "W"
KEY_PREDICTION = "H"
KEY_GROUND_TRUTH = "G"
# Written the way napari writes its own undo keys, so Cmd is used on macOS and Ctrl elsewhere.
KEY_UNDO = KeyMod.CtrlCmd | KeyCode.KeyZ
KEY_REDO = KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyZ

# (key as shown to the user, what it does). Plain keys here are also bound to the viewer.
SHORTCUTS = (
    (KEY_PREVIOUS, "previous case"),
    (KEY_NEXT, "next case"),
    (KEY_ACCEPT, "accept and go on"),
    (KEY_REJECT, "reject and go on"),
    (KEY_SAVE, "save"),
    (KEY_PREDICTION, "show or hide prediction"),
    (KEY_GROUND_TRUTH, "show or hide ground truth"),
    ("Ctrl/Cmd+Z", "undo"),
    ("Shift+Ctrl/Cmd+Z", "redo"),
)

_TRANSPARENT = (0.0, 0.0, 0.0, 0.0)
_CORRECTED_COLOR = (1.0, 0.30, 0.20, 1.0)
_PREDICTION_COLOR = (0.10, 0.80, 1.00, 1.0)
_GROUND_TRUTH_COLOR = (0.30, 0.95, 0.35, 1.0)
_STATUS_COLORS = {"accepted": "#3fb36b", "rejected": "#e5584f"}


def _colormap(rgba: tuple[float, float, float, float]) -> DirectLabelColormap:
    """Background is transparent; every other label value gets one color."""
    return DirectLabelColormap(color_dict={None: rgba, 0: _TRANSPARENT, 1: rgba})


def default_brush_size(shape: tuple[int, ...]) -> int:
    """A brush that is wide enough to be useful on a vessel but not on a whole image."""
    return int(np.clip(round(max(shape) / 100), 3, 15))


def case_title(session: ReviewSession) -> str:
    """For example: 3/20  demo_002"""
    return f"{session.index + 1}/{session.n_items}  {session.item.image_id}"


def case_detail(session: ReviewSession) -> str:
    """For example: accepted | priority 0.41 | edits 5 | saved"""
    parts = [session.status]
    if session.item.priority is not None:
        parts.append(f"priority {session.item.priority:.3g}")
    parts.append(f"edits {session.n_edits}")
    parts.append("unsaved" if session.dirty else "saved")
    return " | ".join(parts)


def status_line(session: ReviewSession) -> str:
    """Index, id, status, priority, edits and saved state of the open case on one line."""
    return f"{case_title(session)} | {case_detail(session)}"


class ReviewPanel(QWidget):
    """Dock widget that connects a ReviewSession to the napari layers.

    The panel creates the four layers, shows the status line and the buttons,
    binds the keyboard shortcuts, and keeps the corrected layer and the session in
    step. The session is the single source of truth: the layer holds its own copy
    of the mask, and every change to it is handed to session.apply_mask.
    """

    def __init__(self, viewer: napari.Viewer, session: ReviewSession) -> None:
        super().__init__()
        self.viewer = viewer
        self.session = session
        self._syncing = False  # True while the panel itself is writing layer data
        self._pending = False  # a paint event arrived before napari wrote the pixels
        self._add_layers()
        self._build_ui()
        self._bind_keys()
        self._connect_layer_events()
        self._update_ui(new_case=True)

    # Layers

    def _add_layers(self) -> None:
        s = self.session
        viewer = self.viewer
        # Napari keeps the array it is given and paints into it, so every layer gets a copy.
        self.image_layer = viewer.add_image(
            np.array(s.image), name=LAYER_IMAGE, colormap="gray", contrast_limits=(0.0, 1.0)
        )
        self.prediction_layer = viewer.add_labels(
            np.array(s.prediction),
            name=LAYER_PREDICTION,
            colormap=_colormap(_PREDICTION_COLOR),
            opacity=0.35,
            visible=False,
        )
        self.prediction_layer.editable = False

        self.gt_layer = None
        if any(item.gt_path is not None for item in s.items):
            self.gt_layer = viewer.add_labels(
                np.array(self._gt_or_empty()),
                name=LAYER_GROUND_TRUTH,
                colormap=_colormap(_GROUND_TRUTH_COLOR),
                visible=False,
            )
            self.gt_layer.contour = 1
            self.gt_layer.editable = False

        self.corrected_layer = viewer.add_labels(
            np.array(s.mask),
            name=LAYER_CORRECTED,
            colormap=_colormap(_CORRECTED_COLOR),
            opacity=0.5,
        )
        corrected = self.corrected_layer
        corrected.brush_size = default_brush_size(s.image.shape)
        corrected.selected_label = 1
        corrected.mode = "paint"
        viewer.layers.selection.select_only(corrected)
        viewer.reset_view()

    def _gt_or_empty(self) -> np.ndarray:
        gt = self.session.gt
        return np.zeros_like(self.session.prediction) if gt is None else gt

    def _labels_layers(self) -> list:
        layers = [self.prediction_layer, self.gt_layer, self.corrected_layer]
        return [layer for layer in layers if layer is not None]

    def _connect_layer_events(self) -> None:
        corrected = self.corrected_layer
        corrected.events.paint.connect(self._on_paint)
        corrected.events.labels_update.connect(self._on_labels_update)
        for layer in (self.prediction_layer, self.gt_layer):
            if layer is not None:
                layer.events.visible.connect(self._sync_overlay_boxes)

    def refresh(self, reset_view: bool = True) -> None:
        """Show the open case: update every layer, the note box and the status line."""
        s = self.session
        self._syncing = True
        try:
            self.image_layer.data = np.array(s.image)
            self.prediction_layer.data = np.array(s.prediction)
            if self.gt_layer is not None:
                self.gt_layer.data = np.array(self._gt_or_empty())
            self.corrected_layer.data = np.array(s.mask)
        finally:
            self._syncing = False
        if reset_view:
            self.viewer.reset_view()
        self._update_ui(new_case=True)

    # Keeping the layer and the session in step

    def _on_paint(self, event=None) -> None:
        # Napari emits "paint" once per stroke. For mouse strokes the pixels are
        # already written. A direct layer.paint() call emits it just before the
        # pixels are written, so wait for the next labels_update in that case.
        if self._syncing:
            return
        if not self.flush():
            self._pending = True

    def _on_labels_update(self, event=None) -> None:
        if self._pending and not self._syncing:
            self.flush()

    def flush(self) -> int:
        """Hand any edit in the corrected layer to the session. Returns pixels changed."""
        self._pending = False
        changed = self.session.apply_mask(self.corrected_layer.data)
        if changed:
            self._normalize_layer()
            self._update_ui()
        return changed

    def _normalize_layer(self) -> None:
        """Make the layer show exactly the session mask, for example after painting label 2."""
        data = self.corrected_layer.data
        if not np.array_equal(data, self.session.mask):
            self._write_mask_to_layer()

    def _write_mask_to_layer(self) -> None:
        self._syncing = True
        try:
            self.corrected_layer.data[...] = self.session.mask
            self.corrected_layer.refresh()
        finally:
            self._syncing = False

    # Actions

    def _commit_note(self) -> None:
        text = " ".join(self.note_edit.text().split())
        if text != self.session.note:
            self.session.note = text

    def _report(self, exc: Exception) -> None:
        """Show a file or disk problem to the reviewer. The session stays on the open case."""
        show_error(f"{type(exc).__name__}: {exc}")
        self._update_ui()

    def _move(self, step) -> None:
        self.flush()
        self._commit_note()
        try:
            moved = step()
        except (OSError, ValueError) as exc:  # an unreadable file or a failed save
            self._report(exc)
            return
        if moved:
            self.refresh()
        else:
            self._update_ui()

    def next_case(self) -> None:
        """Save the open case if needed and open the next one."""
        self._move(self.session.next)

    def previous_case(self) -> None:
        """Save the open case if needed and open the previous one."""
        self._move(self.session.prev)

    def decide(self, status: str) -> None:
        """Record a decision, save the case and, if the box is ticked, open the next one."""
        self.flush()
        self.session.set_status(status, self.note_edit.text())
        try:
            self.session.save()
            opened = self.advance_box.isChecked() and self.session.next()
        except (OSError, ValueError) as exc:  # an unreadable file or a failed save
            self._report(exc)
            return
        if opened:
            self.refresh()
        else:
            self._update_ui()

    def accept(self) -> None:
        self.decide("accepted")

    def reject(self) -> None:
        self.decide("rejected")

    def save(self) -> None:
        self.flush()
        self._commit_note()
        try:
            self.session.save()
        except (OSError, ValueError) as exc:  # a failed write or an unreadable log
            self._report(exc)
            return
        self._update_ui()

    def undo(self) -> None:
        self.flush()
        if self.session.undo():
            self._write_mask_to_layer()
        self._update_ui()

    def redo(self) -> None:
        self.flush()
        if self.session.redo():
            self._write_mask_to_layer()
        self._update_ui()

    def toggle_prediction(self) -> None:
        self.prediction_layer.visible = not self.prediction_layer.visible

    def toggle_ground_truth(self) -> None:
        if self.gt_layer is not None:
            self.gt_layer.visible = not self.gt_layer.visible

    # Widgets

    def _button(self, text: str, tip: str, slot) -> QPushButton:
        button = QPushButton(text)
        button.setToolTip(tip)
        # Keep keyboard focus on the canvas so the shortcuts keep working after a click.
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.clicked.connect(lambda _checked=False, f=slot: f())
        return button

    @staticmethod
    def _plain_label(wrap: bool = True) -> QLabel:
        label = QLabel()
        label.setTextFormat(Qt.TextFormat.PlainText)  # ids and notes are data, not markup
        label.setWordWrap(wrap)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.title_label = self._plain_label()
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.detail_label = self._plain_label()
        layout.addWidget(self.title_label)
        layout.addWidget(self.detail_label)

        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("Note for this case (optional)")
        self.note_edit.setToolTip("Saved with the decision. Press Enter to return to the canvas.")
        self.note_edit.editingFinished.connect(self._note_finished)
        layout.addWidget(self.note_edit)

        self.previous_button = self._button(
            f"Previous ({KEY_PREVIOUS})", f"Previous case ({KEY_PREVIOUS})", self.previous_case
        )
        self.next_button = self._button(f"Next ({KEY_NEXT})", f"Next case ({KEY_NEXT})", self.next_case)
        self.accept_button = self._button(
            f"Accept ({KEY_ACCEPT})", f"Accept this case and save it ({KEY_ACCEPT})", self.accept
        )
        self.reject_button = self._button(
            f"Reject ({KEY_REJECT})", f"Reject this case and save it ({KEY_REJECT})", self.reject
        )
        for button, status in ((self.accept_button, "accepted"), (self.reject_button, "rejected")):
            button.setStyleSheet(f"QPushButton {{ background-color: {_STATUS_COLORS[status]}; color: white; }}")
        self.undo_button = self._button("Undo", "Undo the last edit (Ctrl/Cmd+Z)", self.undo)
        self.redo_button = self._button("Redo", "Redo the last undone edit (Shift+Ctrl/Cmd+Z)", self.redo)
        self.save_button = self._button(f"Save ({KEY_SAVE})", f"Save this case now ({KEY_SAVE})", self.save)

        for pair in (
            (self.previous_button, self.next_button),
            (self.accept_button, self.reject_button),
            (self.undo_button, self.redo_button),
        ):
            row = QHBoxLayout()
            for button in pair:
                row.addWidget(button)
            layout.addLayout(row)
        layout.addWidget(self.save_button)

        self.advance_box = QCheckBox("Go to the next case after Accept or Reject")
        self.advance_box.setChecked(True)
        self.prediction_box = QCheckBox(f"Show prediction ({KEY_PREDICTION})")
        self.ground_truth_box = QCheckBox(f"Show ground truth ({KEY_GROUND_TRUTH})")
        for box in (self.advance_box, self.prediction_box, self.ground_truth_box):
            box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            layout.addWidget(box)
        self.prediction_box.toggled.connect(lambda on: setattr(self.prediction_layer, "visible", bool(on)))
        self.ground_truth_box.toggled.connect(self._set_ground_truth_visible)
        self.ground_truth_box.setEnabled(self.gt_layer is not None)

        width = max(len(key) for key, _what in SHORTCUTS) + 2
        help_text = "Keys\n" + "\n".join(f"{key:<{width}}{what}" for key, what in SHORTCUTS)
        help_text += "\n\nNapari tools: 2 or P paint,\n1 or E erase, [ and ] brush size"
        self.help_label = self._plain_label(wrap=False)
        self.help_label.setText(help_text)
        self.help_label.setStyleSheet("font-family: Menlo, Consolas, monospace; font-size: 11px; color: gray;")
        layout.addWidget(self.help_label)
        layout.addStretch(1)
        self.setMinimumWidth(270)

    def _note_finished(self) -> None:
        self._commit_note()
        self.note_edit.clearFocus()  # the main window forwards key presses to the canvas
        self._update_ui()

    def _set_ground_truth_visible(self, on: bool) -> None:
        if self.gt_layer is not None:
            self.gt_layer.visible = bool(on)

    def _sync_overlay_boxes(self, event=None) -> None:
        for box, layer in ((self.prediction_box, self.prediction_layer), (self.ground_truth_box, self.gt_layer)):
            if layer is not None:
                box.blockSignals(True)
                box.setChecked(bool(layer.visible))
                box.blockSignals(False)

    def _update_ui(self, new_case: bool = False) -> None:
        s = self.session
        self.title_label.setText(case_title(s))
        self.detail_label.setText(case_detail(s))
        self.detail_label.setStyleSheet(
            f"color: {_STATUS_COLORS[s.status]};" if s.status in _STATUS_COLORS else ""
        )
        self.previous_button.setEnabled(s.index > 0)
        self.next_button.setEnabled(s.index + 1 < s.n_items)
        self.undo_button.setEnabled(s.can_undo)
        self.redo_button.setEnabled(s.can_redo)
        self.ground_truth_box.setEnabled(self.gt_layer is not None and s.gt is not None)
        if new_case:
            self.note_edit.setText(s.note)
            self._sync_overlay_boxes()
        self.viewer.title = f"{PANEL_NAME}: {status_line(s)}"

    # Keyboard

    def _bind_keys(self) -> None:
        def once(action):
            # A generator binding makes napari ignore OS key auto-repeat until the key
            # is released, so a held key cannot run the action over and over.
            def handler(viewer):
                action()
                yield

            return handler

        plain = {
            KEY_PREVIOUS: self.previous_case,
            KEY_NEXT: self.next_case,
            KEY_ACCEPT: self.accept,
            KEY_REJECT: self.reject,
            KEY_SAVE: self.save,
            KEY_PREDICTION: self.toggle_prediction,
            KEY_GROUND_TRUTH: self.toggle_ground_truth,
        }
        for key, action in plain.items():
            self.viewer.bind_key(key, once(action), overwrite=True)

        # Undo and redo go through the session. Napari binds Ctrl/Cmd+Z on the Labels
        # class, which would use napari's own history, so each Labels layer is overridden.
        self.viewer.bind_key(KEY_UNDO, lambda viewer: self.undo(), overwrite=True)
        self.viewer.bind_key(KEY_REDO, lambda viewer: self.redo(), overwrite=True)
        for layer in self._labels_layers():
            layer.bind_key(KEY_UNDO, lambda layer: self.undo(), overwrite=True)
            layer.bind_key(KEY_REDO, lambda layer: self.redo(), overwrite=True)


def _settle_layout(viewer: napari.Viewer) -> None:
    """Lay the window out, then fit the image to the canvas.

    A window that was never shown has no layout yet: its canvas keeps a default
    size and screenshots of it come out black. Grabbing the whole window once makes
    Qt compute the layout. The view is fitted afterwards so the image is not cropped
    by the dock widgets that were added after the first fit.
    """
    viewer.screenshot(canvas_only=False, flash=False)
    viewer.reset_view()


def _build(session: ReviewSession, show: bool) -> tuple[napari.Viewer, ReviewPanel]:
    viewer = napari.Viewer(title=PANEL_NAME, show=False)
    try:
        panel = ReviewPanel(viewer, session)
        viewer.window.add_dock_widget(panel, name=PANEL_NAME, area="right")
        if show:
            viewer.show()
        else:
            viewer.window.resize(*HIDDEN_WINDOW_SIZE)  # roomy canvas for screenshots
        _settle_layout(viewer)
    except BaseException:
        viewer.close()
        raise
    return viewer, panel


def build_viewer(session: ReviewSession, show: bool = False) -> napari.Viewer:
    """Open a napari viewer for a review session.

    The viewer has the layers image, prediction, ground truth (when the cases have
    one) and corrected, plus a dock widget named "Angiolens review" with the status
    line and the buttons. The widget is available as
    viewer.window.dock_widgets["Angiolens review"]. With show=False the window is
    built but not displayed, which is enough for screenshots and tests.
    """
    viewer, _panel = _build(session, show)
    return viewer


def run_review(session: ReviewSession, show: bool = True, run=None) -> None:
    """Open the review window, wait until it is closed, then save unsaved work.

    run stands in for the event loop and defaults to napari.run, which blocks
    until the last window closes. Tests pass a function of their own.
    """
    _viewer, panel = _build(session, show)
    try:
        (run or napari.run)()
    finally:
        # The window is gone by now, so talk to the layer and the session only.
        session.apply_mask(panel.corrected_layer.data)
        if session.dirty:
            session.save()


def save_screenshot(viewer: napari.Viewer, path: str | Path, scale: float | None = None) -> Path:
    """Save the canvas as an image file, without the window frame and without the flash.

    Works on a viewer that was built with show=False. A window that was never
    shown has no layout, so it is laid out first; otherwise the canvas is black.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    viewer.screenshot(canvas_only=False, flash=False)  # forces the layout, result unused
    viewer.screenshot(path=str(path), canvas_only=True, flash=False, scale=scale)
    return path


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m angiolens.review.napari_app",
        description="Review and correct predicted vessel masks, one case at a time.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--manifest",
        type=Path,
        help="CSV with image_id, image_path, pred_path and optional gt_path, priority. "
        "Relative paths in the file are relative to the repository root.",
    )
    source.add_argument(
        "--demo",
        type=int,
        metavar="N",
        help="Instead of a manifest, make N synthetic cases in <out>/demo and review them. "
        "Results go to <out>/demo/review. Delete <out>/demo to start the demo over.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Where corrected masks and review_log.csv are written (default: %(default)s).",
    )
    parser.add_argument("--reviewer", default="", help="Name written to the review log.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --demo (default: %(default)s).")
    args = parser.parse_args(argv)
    if args.demo is not None and args.demo < 1:
        parser.error("--demo needs a number of cases of at least 1")
    if args.seed < 0:
        parser.error("--seed must not be negative")
    return args


def main(argv: list[str] | None = None) -> int:
    """Command line entry point. Opens the review window and blocks until it is closed."""
    args = _parse_args(argv)
    if args.demo is not None:
        try:
            from angiolens.synth.generator import generate_pair  # imported late, only the demo needs it
        except ImportError as exc:
            print(f"--demo needs angiolens.synth.generator: {exc}", file=sys.stderr)
            return 2
        demo_dir = args.out / "demo"
        items = make_demo_items(demo_dir, args.demo, args.seed, generate_pair)
        review_dir = demo_dir / "review"
    else:
        items = load_manifest(args.manifest)
        review_dir = args.out

    session = ReviewSession(items, review_dir, reviewer=args.reviewer)
    run_review(session, show=True)
    counts = session.summary()
    print(
        f"{counts['accepted']} accepted, {counts['rejected']} rejected, "
        f"{counts['unreviewed']} unreviewed of {counts['n_items']}. Log: {session.log_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
