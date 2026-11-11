"""Dataset quality checks for image and mask pairs.

Checks
* pairing: every image has a mask and every annotated image exists on disk
* format: size, mode and mask values (must be binary)
* empty masks and masks touching nothing but the border
* exact duplicates: SHA-256 of the decoded pixels
* near duplicates: Pearson correlation of 32 x 32 block-mean thumbnails
  (catches frames of the same angiography run that look almost the same)
* intensity statistics per image and per split
* vessel fraction and vessel radius (distance transform on the mask skeleton)

The generic part (`check_pairs`) works on any list of image and mask files and
is what the unit tests exercise. `main` runs it on ARCADE and writes
results/data_qa.json and results/data_qa.md.

Usage: python -m angiolens.data.qa
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from angiolens.paths import RESULTS, ROOT, ensure

THUMB = 32
NEAR_DUP_THRESHOLD = 0.97  # chosen by visual review of the top ranked ARCADE pairs


@dataclass
class Pair:
    image_id: str
    split: str
    image_path: Path
    mask_path: Path | None


def _open(path: Path) -> Image.Image:
    im = Image.open(path)
    im.load()
    return im


def thumbnail(gray: np.ndarray, size: int = THUMB) -> np.ndarray:
    """Block-mean thumbnail, z-normalized, flattened."""
    h, w = gray.shape
    if h % size or w % size:
        img = Image.fromarray(gray).resize((size, size), Image.BOX)
        t = np.asarray(img, dtype=np.float32)
    else:
        t = gray.astype(np.float32).reshape(size, h // size, size, w // size).mean(axis=(1, 3))
    t = t - t.mean()
    sd = t.std()
    return (t / sd if sd > 0 else t).ravel()


def near_duplicate_pairs(thumbs: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """All index pairs (i < j) whose thumbnail correlation is >= threshold."""
    if len(thumbs) < 2:
        return []
    corr = (thumbs @ thumbs.T) / thumbs.shape[1]
    iu = np.triu_indices(len(thumbs), 1)
    vals = corr[iu]
    keep = vals >= threshold
    return [(int(i), int(j), float(v)) for i, j, v in zip(iu[0][keep], iu[1][keep], vals[keep])]


def vessel_radii(mask: np.ndarray) -> np.ndarray:
    """Local vessel radius (pixels) sampled at every skeleton pixel."""
    m = mask.astype(bool)
    if not m.any():
        return np.zeros(0, dtype=np.float32)
    dist = ndi.distance_transform_edt(m)
    return dist[skeletonize(m)].astype(np.float32)


def intensity_stats(gray: np.ndarray) -> dict:
    g = gray.astype(np.float32)
    p1, p50, p99 = np.percentile(g, [1, 50, 99])
    return {
        "mean": float(g.mean()),
        "std": float(g.std()),
        "min": int(gray.min()),
        "max": int(gray.max()),
        "p1": float(p1),
        "p50": float(p50),
        "p99": float(p99),
        "frac_black": float((gray <= 5).mean()),  # collimator borders
        "frac_saturated": float((gray >= 250).mean()),
    }


def check_pairs(pairs: list[Pair], expected_size: tuple[int, int] | None = None,
                near_dup_threshold: float = NEAR_DUP_THRESHOLD) -> dict:
    """Run every per-pair check. Returns a JSON-serializable report."""
    issues: dict[str, list] = defaultdict(list)
    per_image = []
    thumbs = []
    thumb_ids = []
    hashes: dict[str, list[str]] = defaultdict(list)
    masks_by_id: dict[str, np.ndarray] = {}
    radii_by_split: dict[str, list[np.ndarray]] = defaultdict(list)

    for p in pairs:
        if not p.image_path.exists():
            issues["missing_image"].append(p.image_id)
            continue
        im = _open(p.image_path)
        mode, size = im.mode, im.size
        gray = np.asarray(im.convert("L") if mode != "L" else im, dtype=np.uint8)
        if mode in ("RGB", "RGBA"):
            rgb = np.asarray(im.convert("RGB"), dtype=np.int16)
            if np.abs(rgb[..., 0] - rgb[..., 1]).max() or np.abs(rgb[..., 0] - rgb[..., 2]).max():
                issues["color_image"].append({"image_id": p.image_id, "mode": mode})
            else:
                # Gray pixels stored as RGB: converting to L is lossless.
                issues["gray_stored_as_rgb"].append(p.image_id)
        elif mode not in ("L", "I;16", "I"):
            issues["unexpected_mode"].append({"image_id": p.image_id, "mode": mode})
        if expected_size is not None and size != expected_size:
            issues["unexpected_size"].append({"image_id": p.image_id, "size": list(size)})
        row = {"image_id": p.image_id, "split": p.split, "width": size[0], "height": size[1]}
        row.update(intensity_stats(gray))
        if gray.std() < 1.0:
            issues["constant_image"].append(p.image_id)
        hashes[hashlib.sha256(gray.tobytes()).hexdigest()].append(p.image_id)
        thumbs.append(thumbnail(gray))
        thumb_ids.append(p.image_id)

        if p.mask_path is None or not p.mask_path.exists():
            issues["missing_mask"].append(p.image_id)
        else:
            mraw = np.asarray(_open(p.mask_path))
            values = np.unique(mraw)
            if not set(values.tolist()) <= {0, 1, 255}:
                issues["non_binary_mask"].append({"image_id": p.image_id, "values": values[:10].tolist()})
            mask = (mraw > 0).astype(np.uint8)
            if mask.shape != gray.shape:
                issues["mask_size_mismatch"].append(
                    {"image_id": p.image_id, "image": list(gray.shape), "mask": list(mask.shape)})
            elif not mask.any():
                issues["empty_mask"].append(p.image_id)
            else:
                n_comp = int(ndi.label(mask)[1])
                r = vessel_radii(mask)
                radii_by_split[p.split].append(r)
                row.update({
                    "vessel_fraction": float(mask.mean()),
                    "n_components": n_comp,
                    "median_radius": float(np.median(r)) if r.size else 0.0,
                })
                masks_by_id[p.image_id] = mask
        per_image.append(row)

    split_of = {p.image_id: p.split for p in pairs}
    exact = [ids for ids in hashes.values() if len(ids) > 1]
    exact_report = []
    for ids in exact:
        entry = {"image_ids": ids, "splits": sorted({split_of[i] for i in ids})}
        if all(i in masks_by_id for i in ids[:2]):
            a, b = masks_by_id[ids[0]].astype(bool), masks_by_id[ids[1]].astype(bool)
            entry["mask_dice_between_copies"] = float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))
        exact_report.append(entry)

    near = []
    if thumbs:
        T = np.stack(thumbs)
        for i, j, r in near_duplicate_pairs(T, near_dup_threshold):
            a, b = thumb_ids[i], thumb_ids[j]
            near.append({"a": a, "b": b, "thumb_corr": round(r, 4),
                         "cross_split": split_of[a] != split_of[b]})
    near.sort(key=lambda d: -d["thumb_corr"])

    def summarize(values: list[float]) -> dict:
        v = np.asarray(values, dtype=np.float64)
        if v.size == 0:
            return {}
        return {"mean": float(v.mean()), "sd": float(v.std()), "min": float(v.min()),
                "p5": float(np.percentile(v, 5)), "median": float(np.median(v)),
                "p95": float(np.percentile(v, 95)), "max": float(v.max())}

    splits = sorted({p.split for p in pairs})
    by_split = {}
    for s in splits:
        rows = [r for r in per_image if r["split"] == s]
        radii = np.concatenate(radii_by_split[s]) if radii_by_split[s] else np.zeros(0)
        by_split[s] = {
            "n_images": len(rows),
            "mean_intensity": summarize([r["mean"] for r in rows]),
            "std_intensity": summarize([r["std"] for r in rows]),
            "frac_black": summarize([r["frac_black"] for r in rows]),
            "vessel_fraction": summarize([r["vessel_fraction"] for r in rows if "vessel_fraction" in r]),
            "n_components": summarize([r["n_components"] for r in rows if "n_components" in r]),
            "skeleton_radius_px": summarize(radii.tolist()) if radii.size else {},
        }
    return {
        "n_pairs": len(pairs),
        "issues": {k: v for k, v in issues.items()},
        "issue_counts": {k: len(v) for k, v in issues.items()},
        "exact_duplicates": exact_report,
        "near_duplicates": {"method": f"Pearson r of {THUMB}x{THUMB} block-mean thumbnails",
                            "threshold": near_dup_threshold, "pairs": near},
        "by_split": by_split,
        "per_image": per_image,
    }


def arcade_pairs() -> tuple[list[Pair], dict]:
    """ARCADE image and mask pairs, plus checks against the COCO annotation files."""
    from angiolens.data.arcade import SPLITS, read_index
    from angiolens.paths import ARCADE_DIR

    index = read_index()
    pairs = [Pair(r["image_id"], r["split"], ROOT / r["image_path"], ROOT / r["mask_path"]) for r in index]
    coco = {}
    for split in SPLITS:
        ann = json.loads((ARCADE_DIR / "syntax" / split / "annotations" / f"{split}.json").read_text())
        listed = {im["file_name"] for im in ann["images"]}
        on_disk = {p.name for p in (ARCADE_DIR / "syntax" / split / "images").glob("*.png")}
        annotated = {im["file_name"] for im in ann["images"]
                     if any(a["image_id"] == im["id"] for a in ann["annotations"])}
        coco[split] = {
            "images_in_json": len(listed),
            "images_on_disk": len(on_disk),
            "in_json_not_on_disk": sorted(listed - on_disk),
            "on_disk_not_in_json": sorted(on_disk - listed),
            "images_without_annotations": sorted(listed - annotated),
            "n_annotations": len(ann["annotations"]),
        }
    return pairs, coco


def write_markdown(report: dict, path: Path) -> None:
    lines = ["# Data QA report (ARCADE syntax subset)", ""]
    lines.append(f"Pairs checked: {report['n_pairs']}")
    lines.append("")
    lines.append("| Check | Count |")
    lines.append("|---|---|")
    for key in ["missing_image", "missing_mask", "unexpected_size", "unexpected_mode", "color_image",
                "gray_stored_as_rgb", "non_binary_mask", "mask_size_mismatch", "empty_mask", "constant_image"]:
        lines.append(f"| {key.replace('_', ' ')} | {report['issue_counts'].get(key, 0)} |")
    lines.append(f"| exact duplicate groups | {len(report['exact_duplicates'])} |")
    nd = report["near_duplicates"]
    lines.append(f"| near duplicate pairs (r >= {nd['threshold']}) | {len(nd['pairs'])} |")
    lines.append(f"| near duplicate pairs across splits | {sum(p['cross_split'] for p in nd['pairs'])} |")
    lines.append("")
    if "coco" in report:
        lines.append("| Split | Images in JSON | Images on disk | Missing | Extra | Unannotated | Annotations |")
        lines.append("|---|---|---|---|---|---|---|")
        for s, c in report["coco"].items():
            lines.append(f"| {s} | {c['images_in_json']} | {c['images_on_disk']} | {len(c['in_json_not_on_disk'])} | "
                         f"{len(c['on_disk_not_in_json'])} | {len(c['images_without_annotations'])} | {c['n_annotations']} |")
        lines.append("")
    lines.append("| Split | Images | Mean intensity (mean, sd) | Vessel fraction (median) | Skeleton radius px (median, p95) | Black border fraction (mean) |")
    lines.append("|---|---|---|---|---|---|")
    for s, b in report["by_split"].items():
        mi, vf, rr, fb = b["mean_intensity"], b["vessel_fraction"], b["skeleton_radius_px"], b["frac_black"]
        lines.append(f"| {s} | {b['n_images']} | {mi['mean']:.1f}, {mi['sd']:.1f} | {vf['median']:.4f} | "
                     f"{rr.get('median', 0):.2f}, {rr.get('p95', 0):.2f} | {fb['mean']:.4f} |")
    lines.append("")
    if report["exact_duplicates"]:
        lines.append("Exact duplicates (pixel-identical images):")
        lines.append("")
        for e in report["exact_duplicates"]:
            d = e.get("mask_dice_between_copies")
            extra = f", Dice between the two masks {d:.3f}" if d is not None else ""
            lines.append(f"* {', '.join(e['image_ids'])} ({'/'.join(e['splits'])}{extra})")
        lines.append("")
    if nd["pairs"]:
        lines.append("Near duplicates:")
        lines.append("")
        for p in nd["pairs"]:
            tag = " (across splits)" if p["cross_split"] else ""
            lines.append(f"* {p['a']} and {p['b']}: r = {p['thumb_corr']:.3f}{tag}")
        lines.append("")
    path.write_text("\n".join(lines))


def near_duplicate_figure(report: dict, pairs: list[Pair], path: Path, max_rows: int = 7) -> None:
    """Side-by-side view of the near-duplicate pairs above the threshold, for visual review."""
    from angiolens.eval import plots

    found = report["near_duplicates"]["pairs"]
    exact = {tuple(sorted(e["image_ids"][:2])) for e in report["exact_duplicates"]}
    show = [p for p in found if tuple(sorted((p["a"], p["b"]))) not in exact][:max_rows]
    if not show:
        return
    by_id = {p.image_id: p for p in pairs}
    plots.style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(show), 2, figsize=(5.0, 2.5 * len(show)))
    axes = np.atleast_2d(axes)
    for i, d in enumerate(show):
        for j, key in enumerate(("a", "b")):
            img = np.asarray(_open(by_id[d[key]].image_path).convert("L"))
            axes[i, j].imshow(img, cmap="gray", vmin=0, vmax=255)
            axes[i, j].set_axis_off()
            axes[i, j].set_title(d[key] + (f"   r = {d['thumb_corr']:.3f}" if j == 0 else ""), fontsize=8, loc="left")
    fig.suptitle("Near-duplicate pairs (exact copies not shown)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> int:
    ensure(RESULTS)
    pairs, coco = arcade_pairs()
    report = check_pairs(pairs, expected_size=(512, 512))
    report["coco"] = coco
    per_image = report.pop("per_image")
    keys = sorted({k for r in per_image for k in r}, key=lambda k: (k not in ("image_id", "split"), k))
    with open(RESULTS / "data_qa_per_image.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in per_image:
            writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    (RESULTS / "data_qa.json").write_text(json.dumps(report, indent=1))
    write_markdown(report, RESULTS / "data_qa.md")
    ensure(RESULTS / "figures")
    near_duplicate_figure(report, pairs, RESULTS / "figures" / "near_duplicates.png")
    print(json.dumps(report["issue_counts"]), f"exact dup groups {len(report['exact_duplicates'])}",
          f"near dup pairs {len(report['near_duplicates']['pairs'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
