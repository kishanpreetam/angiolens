"""Review session logic for vessel masks. No GUI code lives here.

A review session walks through a list of cases. Each case has an X-ray image,
a predicted vessel mask and, optionally, a ground truth mask. The reviewer edits
a working copy of the prediction, marks the case accepted or rejected, and the
session writes the result to disk:

    <out_dir>/masks/<image_id>.png   corrected mask, uint8, 0 or 255
    <out_dir>/review_log.csv         one row per image, replaced atomically on save

Inputs are PNG files decoded with Pillow. Nothing else is ever loaded from disk,
so a malformed input can raise an error but cannot run code.

The edit history stores diffs (changed pixel indices with their old and new
values), so it stays small even for large images. It is kept per case and is
cleared when the reviewer moves to another case.
"""

from __future__ import annotations

import csv
import io
import math
import operator
import os
import re
import warnings
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image

from angiolens.paths import ROOT

STATUSES = ("unreviewed", "accepted", "rejected")

MASKS_DIRNAME = "masks"
LOG_FILENAME = "review_log.csv"
LOG_COLUMNS = (
    "image_id",
    "status",
    "note",
    "reviewer",
    "pixels_added",
    "pixels_removed",
    "dice_vs_prediction",
    "n_edits",
    "saved_at_utc",
    "image_path",
    "pred_path",
    "corrected_mask_path",
)
MANIFEST_REQUIRED = ("image_id", "image_path", "pred_path")
MANIFEST_COLUMNS = MANIFEST_REQUIRED + ("gt_path", "priority")

_MAX_FILENAME_CHARS = 150
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


# Cases and manifests


@dataclass
class ReviewItem:
    """One case to review.

    priority is any score where a higher value means review sooner, for example
    model uncertainty. It is shown to the reviewer and used to order manifests.
    """

    image_id: str
    image_path: Path
    pred_path: Path
    gt_path: Path | None = None
    priority: float | None = None

    def __post_init__(self) -> None:
        self.image_id = str(self.image_id)
        self.image_path = Path(self.image_path)
        self.pred_path = Path(self.pred_path)
        if self.gt_path is not None:
            self.gt_path = Path(self.gt_path)


def _resolve(text: str, root: Path) -> Path:
    path = Path(text).expanduser()
    return path if path.is_absolute() else root / path


def _priority_key(item: ReviewItem) -> tuple[bool, float]:
    # Highest priority first; items without a priority go last. sort() is stable.
    if item.priority is None:
        return (True, 0.0)
    return (False, -item.priority)


def load_manifest(csv_path: str | os.PathLike, root: str | os.PathLike = ROOT) -> list[ReviewItem]:
    """Read a manifest CSV into review items.

    Required columns are image_id, image_path and pred_path. The columns gt_path
    and priority are optional. Relative paths are resolved against root, not
    against the CSV file. When a priority column exists, items are sorted by it,
    highest first, keeping file order for ties and putting blank priorities last.
    """
    csv_path = Path(csv_path)
    root = Path(root)
    items: list[ReviewItem] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = [(name or "").strip() for name in (reader.fieldnames or [])]
        missing = [name for name in MANIFEST_REQUIRED if name not in header]
        if missing:
            raise ValueError(f"{csv_path}: manifest is missing column(s): {', '.join(missing)}")
        reader.fieldnames = header
        for line_number, raw in enumerate(reader, start=2):
            row = {k: (v or "").strip() for k, v in raw.items() if k is not None}
            if not any(row.values()):
                continue
            where = f"{csv_path} line {line_number}"
            image_id = row["image_id"]
            if not image_id:
                raise ValueError(f"{where}: image_id is empty")
            if not row["image_path"] or not row["pred_path"]:
                raise ValueError(f"{where}: image_path and pred_path are required")
            priority = None
            if row.get("priority"):
                try:
                    priority = float(row["priority"])
                except ValueError:
                    raise ValueError(f"{where}: priority {row['priority']!r} is not a number") from None
                if math.isnan(priority):
                    raise ValueError(f"{where}: priority is not a number")
            gt_text = row.get("gt_path", "")
            items.append(
                ReviewItem(
                    image_id=image_id,
                    image_path=_resolve(row["image_path"], root),
                    pred_path=_resolve(row["pred_path"], root),
                    gt_path=_resolve(gt_text, root) if gt_text else None,
                    priority=priority,
                )
            )
    if "priority" in header:
        items.sort(key=_priority_key)
    return items


def write_manifest(csv_path: str | os.PathLike, items: Iterable[ReviewItem]) -> None:
    """Write items to a manifest CSV. Paths are written exactly as they are stored."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(MANIFEST_COLUMNS)
    for item in items:
        writer.writerow(
            [
                item.image_id,
                str(item.image_path),
                str(item.pred_path),
                "" if item.gt_path is None else str(item.gt_path),
                "" if item.priority is None else repr(float(item.priority)),
            ]
        )
    _write_atomic(Path(csv_path), buffer.getvalue().encode("utf-8"))


# File helpers


def safe_filename(image_id: str) -> str:
    """Turn an image id into a name that is safe to use as a file name.

    Anything outside letters, digits, dot, underscore and hyphen becomes an
    underscore, so path separators cannot appear. Runs of dots are broken up so
    the result never contains "..", and leading or trailing dots are removed.
    """
    text = str(image_id).strip()
    if not text:
        raise ValueError("image_id is empty")
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", text)
    while ".." in cleaned:
        cleaned = cleaned.replace("..", "_")
    cleaned = cleaned.strip(".")[:_MAX_FILENAME_CHARS].strip(".")
    if not cleaned:
        raise ValueError(f"image_id {image_id!r} has no usable characters")
    return cleaned


def _write_atomic(path: Path, data: bytes) -> None:
    """Write data to path so readers see either the old file or the new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _write_png(path: Path, array: np.ndarray) -> None:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    _write_atomic(path, buffer.getvalue())


def _portable(path: Path) -> str:
    """Return a path relative to the repository root when it lives there."""
    path = Path(path)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(Path(ROOT).resolve()).as_posix()
        except (ValueError, OSError):
            pass
    return str(path)


def _gray_float(img: Image.Image) -> np.ndarray:
    """Convert a decoded image to float32 in [0, 1] with shape (H, W)."""
    if img.mode in ("I;16", "I;16L", "I;16B", "I;16N"):
        arr = np.asarray(img).astype(np.float32) / 65535.0
    elif img.mode in ("I", "F"):
        raw = np.asarray(img).astype(np.float64)
        top = float(raw.max()) if raw.size else 0.0
        if top <= 1.0:
            scale = 1.0
        elif top <= 255.0:
            scale = 255.0
        elif top <= 65535.0:
            scale = 65535.0
        else:
            raw = raw - float(raw.min())
            scale = max(float(raw.max()), 1.0)
        arr = (raw / scale).astype(np.float32)
    else:
        arr = np.asarray(img.convert("L")).astype(np.float32) / 255.0
    return np.ascontiguousarray(np.clip(arr, 0.0, 1.0), dtype=np.float32)


def load_image(path: str | os.PathLike) -> np.ndarray:
    """Load a grayscale or RGB PNG as float32 in [0, 1] with shape (H, W).

    8 bit data is divided by 255 and 16 bit data by 65535. RGB is converted to gray.
    """
    with Image.open(path) as img:
        img.load()
        return _gray_float(img)


def load_mask(path: str | os.PathLike) -> np.ndarray:
    """Load a mask PNG as uint8 with values 0 and 1. Any value above zero is vessel."""
    with Image.open(path) as img:
        img.load()
        if img.mode in ("RGB", "RGBA"):
            arr = np.asarray(img)[..., :3].max(axis=-1)
        elif img.mode in ("LA", "La"):
            arr = np.asarray(img)[..., 0]
        elif img.mode in ("L", "P", "1", "I", "I;16", "I;16L", "I;16B", "I;16N", "F"):
            arr = np.asarray(img)
        else:
            arr = np.asarray(img.convert("L"))
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected a 2D mask, got shape {arr.shape}")
    return (arr > 0).astype(np.uint8)


def dice_score(a: np.ndarray, b: np.ndarray) -> float:
    """Dice overlap of two binary masks. Two empty masks score 1.0."""
    a = np.asarray(a) > 0
    b = np.asarray(b) > 0
    total = int(a.sum()) + int(b.sum())
    if total == 0:
        return 1.0
    return 2.0 * float(np.logical_and(a, b).sum()) / total


# Session


@dataclass
class _Diff:
    """One undoable edit: which pixels changed and their values before and after."""

    index: np.ndarray
    old: np.ndarray
    new: np.ndarray


@dataclass
class _State:
    """Everything the session holds about the case that is currently open."""

    index: int
    image: np.ndarray
    prediction: np.ndarray
    gt: np.ndarray | None
    mask: np.ndarray
    status: str
    note: str
    n_edits: int
    saved_mask: np.ndarray
    saved_status: str
    saved_note: str
    saved_n_edits: int
    undo: deque
    redo: list


def _readonly(array: np.ndarray) -> np.ndarray:
    view = array.view()
    view.flags.writeable = False
    return view


def _to_int(text: object, default: int = 0) -> int:
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return default


def _one_line(text: object) -> str:
    return " ".join(str(text).split())


class ReviewSession:
    """Step through review cases, edit a working mask, and save the results.

    The mask property is what the reviewer edits. It starts as the model
    prediction, or as the previously saved corrected mask when one exists in
    out_dir. The arrays returned by the image, prediction, mask and gt
    properties are read-only views; change the mask with paint or apply_mask.

    Only one session at a time should write to a given out_dir.
    """

    def __init__(
        self,
        items: Iterable[ReviewItem],
        out_dir: str | os.PathLike,
        reviewer: str = "",
        max_undo: int = 100,
    ) -> None:
        self.items: list[ReviewItem] = list(items)
        if not self.items:
            raise ValueError("a review session needs at least one item")
        if max_undo < 0:
            raise ValueError("max_undo must be zero or larger")
        self.out_dir = Path(out_dir)
        self.reviewer = _one_line(reviewer)
        self.max_undo = int(max_undo)
        self._safe_ids = self._check_items(self.items)
        self._log = self._read_log()
        self._index = 0
        self._s = self._load_state(0)

    # Construction helpers

    @staticmethod
    def _check_items(items: list[ReviewItem]) -> list[str]:
        seen_ids: dict[str, str] = {}
        seen_files: dict[str, str] = {}
        safe_ids: list[str] = []
        missing: list[str] = []
        for item in items:
            if not isinstance(item, ReviewItem):
                raise TypeError(f"expected ReviewItem, got {type(item).__name__}")
            if item.image_id in seen_ids:
                raise ValueError(f"duplicate image_id {item.image_id!r}")
            seen_ids[item.image_id] = item.image_id
            safe = safe_filename(item.image_id)
            if safe in seen_files:
                raise ValueError(
                    f"image ids {seen_files[safe]!r} and {item.image_id!r} "
                    f"would both be saved as {safe!r}.png"
                )
            seen_files[safe] = item.image_id
            safe_ids.append(safe)
            for path in (item.image_path, item.pred_path, item.gt_path):
                if path is not None and not Path(path).is_file():
                    missing.append(str(path))
        if missing:
            shown = ", ".join(missing[:5])
            more = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
            raise FileNotFoundError(f"missing input file(s): {shown}{more}")
        return safe_ids

    def _read_log(self) -> dict[str, dict[str, str]]:
        path = self.log_path
        rows: dict[str, dict[str, str]] = {}
        if not path.is_file():
            return rows
        try:
            with open(path, newline="", encoding="utf-8") as handle:
                for record in csv.DictReader(handle):
                    image_id = (record.get("image_id") or "").strip()
                    if image_id:
                        rows[image_id] = {col: record.get(col) or "" for col in LOG_COLUMNS}
        except (csv.Error, UnicodeDecodeError) as exc:
            raise ValueError(f"cannot read review log {path}: {exc}") from exc
        return rows

    def _load_state(self, index: int) -> _State:
        item = self.items[index]
        image = load_image(item.image_path)
        prediction = load_mask(item.pred_path)
        self._check_shape(item, "prediction", prediction, image)
        gt = None
        if item.gt_path is not None:
            gt = load_mask(item.gt_path)
            self._check_shape(item, "ground truth", gt, image)

        mask = prediction.copy()
        resumed = False
        saved_png = self._mask_path_for(index)
        if saved_png.is_file():
            try:
                saved = load_mask(saved_png)
            except (OSError, ValueError) as exc:
                warnings.warn(f"{item.image_id}: ignoring unreadable corrected mask {saved_png} ({exc})", stacklevel=3)
            else:
                if saved.shape == image.shape:
                    mask = saved
                    resumed = True
                else:
                    warnings.warn(
                        f"{item.image_id}: ignoring corrected mask {saved_png} "
                        f"with shape {saved.shape}, image has shape {image.shape}",
                        stacklevel=3,
                    )

        row = self._log.get(item.image_id, {})
        status = row.get("status", "")
        if status not in STATUSES:
            status = "unreviewed"
        note = _one_line(row.get("note", ""))
        n_edits = _to_int(row.get("n_edits"), 0) if resumed else 0
        return _State(
            index=index,
            image=image,
            prediction=prediction,
            gt=gt,
            mask=mask,
            status=status,
            note=note,
            n_edits=n_edits,
            saved_mask=mask.copy(),
            saved_status=status,
            saved_note=note,
            saved_n_edits=n_edits,
            undo=deque(maxlen=self.max_undo),
            redo=[],
        )

    @staticmethod
    def _check_shape(item: ReviewItem, what: str, array: np.ndarray, image: np.ndarray) -> None:
        if array.shape != image.shape:
            raise ValueError(
                f"{item.image_id}: {what} has shape {array.shape} but the image has shape {image.shape}"
            )

    def _mask_path_for(self, index: int) -> Path:
        return self.out_dir / MASKS_DIRNAME / f"{self._safe_ids[index]}.png"

    # Read-only state

    @property
    def index(self) -> int:
        """Position of the open case, starting at 0."""
        return self._index

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def item(self) -> ReviewItem:
        return self.items[self._index]

    @property
    def image(self) -> np.ndarray:
        """Image as float32 in [0, 1], shape (H, W). Read-only."""
        return _readonly(self._s.image)

    @property
    def prediction(self) -> np.ndarray:
        """Original model prediction as uint8 0 or 1. Read-only."""
        return _readonly(self._s.prediction)

    @property
    def mask(self) -> np.ndarray:
        """Working mask as uint8 0 or 1. Read-only; change it with paint or apply_mask."""
        return _readonly(self._s.mask)

    @property
    def gt(self) -> np.ndarray | None:
        """Ground truth as uint8 0 or 1, or None when the case has none. Read-only."""
        return None if self._s.gt is None else _readonly(self._s.gt)

    @property
    def status(self) -> str:
        return self._s.status

    @property
    def note(self) -> str:
        return self._s.note

    @note.setter
    def note(self, value: str) -> None:
        self._s.note = _one_line(value)

    @property
    def n_edits(self) -> int:
        """Net number of edits applied to the prediction, counting earlier sessions."""
        return self._s.n_edits

    @property
    def can_undo(self) -> bool:
        return bool(self._s.undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._s.redo)

    @property
    def dirty(self) -> bool:
        """True when the open case differs from what was last loaded or saved."""
        s = self._s
        return (
            s.status != s.saved_status
            or s.note != s.saved_note
            or s.n_edits != s.saved_n_edits
            or not np.array_equal(s.mask, s.saved_mask)
        )

    @property
    def corrected_mask_path(self) -> Path:
        """Where save() writes the corrected mask of the open case."""
        return self._mask_path_for(self._index)

    @property
    def log_path(self) -> Path:
        return self.out_dir / LOG_FILENAME

    def stats(self) -> dict[str, float | int]:
        """Pixels added, pixels removed and Dice of the working mask against the prediction."""
        mask = self._s.mask > 0
        pred = self._s.prediction > 0
        return {
            "pixels_added": int(np.count_nonzero(mask & ~pred)),
            "pixels_removed": int(np.count_nonzero(~mask & pred)),
            "dice_vs_prediction": dice_score(mask, pred),
        }

    # Editing

    def _commit(self, flat_index: np.ndarray, new_values: np.ndarray) -> int:
        """Write new values into the mask and record the diff for undo."""
        s = self._s
        flat_mask = s.mask.reshape(-1)
        old_values = flat_mask[flat_index]
        flat_mask[flat_index] = new_values
        compact = flat_index.astype(np.int32) if flat_mask.size < 2**31 else flat_index
        s.undo.append(_Diff(compact, old_values, np.array(new_values, dtype=np.uint8)))
        s.redo.clear()
        s.n_edits += 1
        return int(flat_index.size)

    def paint(self, row: float, col: float, radius: float, value: int) -> int:
        """Set a filled disk to value (1 paints vessel, 0 erases) and return pixels changed.

        The disk covers every pixel whose center lies within radius pixels of
        (row, col), clipped at the image border. The center is rounded to the
        nearest pixel. The edit is one undo step.
        """
        if value not in (0, 1):
            raise ValueError(f"value must be 0 or 1, got {value!r}")
        radius = float(radius)
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("radius must be a finite number, zero or larger")
        if not (math.isfinite(row) and math.isfinite(col)):
            raise ValueError("row and col must be finite numbers")
        row, col = round(float(row)), round(float(col))

        height, width = self._s.mask.shape
        r0, r1 = max(0, math.floor(row - radius)), min(height, math.floor(row + radius) + 1)
        c0, c1 = max(0, math.floor(col - radius)), min(width, math.floor(col + radius) + 1)
        if r0 >= r1 or c0 >= c1:
            return 0
        rr = np.arange(r0, r1)[:, None]
        cc = np.arange(c0, c1)[None, :]
        disk = (rr - row) ** 2 + (cc - col) ** 2 <= radius**2
        changed = disk & (self._s.mask[r0:r1, c0:c1] != value)
        ys, xs = np.nonzero(changed)
        if ys.size == 0:
            return 0
        flat = (ys + r0).astype(np.int64) * width + (xs + c0)
        return self._commit(flat, np.full(flat.size, value, dtype=np.uint8))

    def apply_mask(self, new_mask: np.ndarray) -> int:
        """Replace the working mask with an externally edited one and return pixels changed.

        Values above zero count as vessel. Returns 0 and records nothing when
        new_mask matches the current mask. Raises ValueError on a shape mismatch.
        """
        arr = np.asarray(new_mask)
        if arr.shape != self._s.mask.shape:
            raise ValueError(f"mask has shape {arr.shape} but the image has shape {self._s.mask.shape}")
        wanted = (arr > 0).reshape(-1)
        flat_index = np.flatnonzero(wanted != (self._s.mask.reshape(-1) > 0))
        if flat_index.size == 0:
            return 0
        return self._commit(flat_index, wanted[flat_index].astype(np.uint8))

    def undo(self) -> bool:
        """Revert the last edit. Returns False when there is nothing to undo."""
        s = self._s
        if not s.undo:
            return False
        diff = s.undo.pop()
        s.mask.reshape(-1)[diff.index] = diff.old
        s.redo.append(diff)
        s.n_edits -= 1
        return True

    def redo(self) -> bool:
        """Reapply the last undone edit. Returns False when there is nothing to redo."""
        s = self._s
        if not s.redo:
            return False
        diff = s.redo.pop()
        s.mask.reshape(-1)[diff.index] = diff.new
        s.undo.append(diff)
        s.n_edits += 1
        return True

    def set_status(self, status: str, note: str = "") -> None:
        """Record the reviewer's decision for the open case, with an optional note."""
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
        self._s.status = status
        self._s.note = _one_line(note)

    # Navigation

    def goto(self, i: int) -> None:
        """Open case i, saving the current one first if it has unsaved changes."""
        i = operator.index(i)
        if not 0 <= i < self.n_items:
            raise IndexError(f"case {i} is out of range for {self.n_items} cases")
        if i == self._index:
            return
        self._autosave()
        # Load before switching so a bad file leaves the session on the old case.
        self._s = self._load_state(i)
        self._index = i

    def next(self) -> bool:
        """Open the next case. Returns False at the last case.

        Unsaved changes are saved either way, because the reviewer asked to leave
        the case.
        """
        if self._index + 1 >= self.n_items:
            self._autosave()
            return False
        self.goto(self._index + 1)
        return True

    def prev(self) -> bool:
        """Open the previous case. Returns False at the first case.

        Unsaved changes are saved either way, because the reviewer asked to leave
        the case.
        """
        if self._index == 0:
            self._autosave()
            return False
        self.goto(self._index - 1)
        return True

    def _autosave(self) -> None:
        if self.dirty:
            self.save()

    # Saving

    def save(self) -> Path:
        """Write the corrected mask and upsert this case's row in the review log."""
        s = self._s
        item = self.item
        path = self.corrected_mask_path
        _write_png(path, (s.mask * 255).astype(np.uint8))

        stats = self.stats()
        row = {
            "image_id": item.image_id,
            "status": s.status,
            "note": s.note,
            "reviewer": self.reviewer,
            "pixels_added": str(stats["pixels_added"]),
            "pixels_removed": str(stats["pixels_removed"]),
            "dice_vs_prediction": f"{stats['dice_vs_prediction']:.6f}",
            "n_edits": str(s.n_edits),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "image_path": _portable(item.image_path),
            "pred_path": _portable(item.pred_path),
            "corrected_mask_path": _portable(path),
        }
        # Re-read first so rows written by anyone else since the last save are kept.
        rows = self._read_log()
        rows[item.image_id] = row
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=LOG_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows.values())
        _write_atomic(self.log_path, buffer.getvalue().encode("utf-8"))
        self._log = rows

        s.saved_mask = s.mask.copy()
        s.saved_status = s.status
        s.saved_note = s.note
        s.saved_n_edits = s.n_edits
        return path

    def summary(self) -> dict[str, int]:
        """Counts over all cases: n_items, one count per status, edited and saved.

        Edited means the corrected mask differs from the prediction. Saved means
        the case has a row in the review log. The open case is counted from its
        live state, the others from the log.
        """
        counts = {name: 0 for name in STATUSES}
        edited = saved = 0
        for position, item in enumerate(self.items):
            if position == self._index:
                counts[self._s.status] += 1
                stats = self.stats()
                edited += int(stats["pixels_added"] + stats["pixels_removed"] > 0)
                saved += int(item.image_id in self._log)
                continue
            row = self._log.get(item.image_id)
            if row is None:
                counts["unreviewed"] += 1
                continue
            status = row["status"] if row["status"] in STATUSES else "unreviewed"
            counts[status] += 1
            saved += 1
            edited += int(_to_int(row["pixels_added"]) + _to_int(row["pixels_removed"]) > 0)
        return {"n_items": self.n_items, **counts, "edited": edited, "saved": saved}


# Demo data


def _grow_or_shrink(mask: np.ndarray, grow: bool) -> np.ndarray:
    """Binary dilation (grow) or erosion of a 0/1 mask with a 3x3 square, using numpy only."""
    height, width = mask.shape
    source = mask > 0
    if not grow:
        source = ~source
    padded = np.pad(source, 1, mode="constant", constant_values=False)
    out = np.zeros_like(source)
    for dr in range(3):
        for dc in range(3):
            out |= padded[dr : dr + height, dc : dc + width]
    return (out if grow else ~out).astype(np.uint8)


def _disk(shape: tuple[int, int], row: float, col: float, radius: float) -> np.ndarray:
    rr, cc = np.ogrid[: shape[0], : shape[1]]
    return (rr - row) ** 2 + (cc - col) ** 2 <= radius**2


def perturb_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Make a deliberately imperfect copy of a mask, like a weak model would.

    The mask is thinned or thickened by one pixel, a few random patches are
    deleted (preferably on vessels) and a few false blobs are added. The result
    always differs from the input.
    """
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    height, width = mask.shape
    size = min(height, width)
    grow = bool(rng.integers(0, 2))
    out = _grow_or_shrink(mask, grow)
    if not grow and np.count_nonzero(out) < 0.6 * np.count_nonzero(mask):
        out = _grow_or_shrink(mask, True)  # too thin to erode without losing the vessel

    vessel_rows, vessel_cols = np.nonzero(mask)
    for _ in range(int(rng.integers(1, 4))):
        if vessel_rows.size:
            k = int(rng.integers(0, vessel_rows.size))
            row, col = float(vessel_rows[k]), float(vessel_cols[k])
        else:
            row, col = float(rng.integers(0, height)), float(rng.integers(0, width))
        radius = float(rng.uniform(max(2.0, 0.012 * size), max(3.0, 0.035 * size)))
        out[_disk(out.shape, row, col, radius)] = 0

    for _ in range(int(rng.integers(1, 4))):
        row, col = float(rng.integers(0, height)), float(rng.integers(0, width))
        radius = float(rng.uniform(max(1.5, 0.006 * size), max(2.5, 0.02 * size)))
        out[_disk(out.shape, row, col, radius)] = 1

    if np.array_equal(out, mask):
        out[height // 2, width // 2] = 1 - out[height // 2, width // 2]
    return out


def make_demo_items(
    out_dir: str | os.PathLike,
    n: int,
    seed: int,
    pair_fn: Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray]],
) -> list[ReviewItem]:
    """Write n demo cases under out_dir and return them as review items.

    pair_fn(rng) must return (image, mask): a float32 image in [0, 1] with shape
    (H, W) and a uint8 mask of 0 and 1 with the same shape. For each case this
    writes the image, an imperfect prediction made with perturb_mask, and the true
    mask as ground truth. The priority of a case is one minus the Dice of the
    prediction against the truth, so the worst predictions are reviewed first.
    A manifest.csv listing the cases is written next to the images. The result is
    the manifest as load_manifest reads it, which means sorted by priority.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    out_dir = Path(out_dir)
    items: list[ReviewItem] = []
    for i in range(n):
        rng = np.random.default_rng([int(seed), i])
        image, truth = pair_fn(rng)
        image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
        truth = (np.asarray(truth) > 0).astype(np.uint8)
        if image.ndim != 2 or truth.shape != image.shape:
            raise ValueError(
                f"pair_fn must return a 2D image and a mask of the same shape, "
                f"got {image.shape} and {truth.shape}"
            )
        prediction = perturb_mask(truth, rng)

        image_id = f"demo_{i:03d}"
        paths = {
            "image": out_dir / "images" / f"{image_id}.png",
            "pred": out_dir / "predictions" / f"{image_id}.png",
            "gt": out_dir / "ground_truth" / f"{image_id}.png",
        }
        _write_png(paths["image"], np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8))
        _write_png(paths["pred"], prediction * 255)
        _write_png(paths["gt"], truth * 255)
        items.append(
            ReviewItem(
                image_id=image_id,
                image_path=paths["image"],
                pred_path=paths["pred"],
                gt_path=paths["gt"],
                priority=round(1.0 - dice_score(prediction, truth), 4),
            )
        )

    manifest = out_dir / "manifest.csv"
    write_manifest(
        manifest,
        [
            replace(
                item,
                image_path=Path(_portable(item.image_path)),
                pred_path=Path(_portable(item.pred_path)),
                gt_path=Path(_portable(item.gt_path)),
            )
            for item in items
        ],
    )
    return load_manifest(manifest, root=ROOT)
