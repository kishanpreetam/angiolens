"""Figures for the validation report (matplotlib, static PNG, light surface).

Colors: methods take categorical slots in a fixed order so a method keeps its
color in every figure. Overlays use yellow for true positives, red for false
positives and blue for false negatives (checked for color vision deficiency
separation); every figure also labels them in text.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NAMES = {"dice": "Dice", "iou": "IoU", "sensitivity": "Sensitivity", "precision": "Precision", "cldice": "clDice"}
TP_RGB = np.array([0xED, 0xA1, 0x00]) / 255
FP_RGB = np.array([0xE3, 0x49, 0x48]) / 255
FN_RGB = np.array([0x2A, 0x78, 0xD6]) / 255


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "font.size": 9,
    })


def method_colors(methods: list[str], order: list[str]) -> dict[str, str]:
    return {m: SLOTS[order.index(m) % len(SLOTS)] for m in methods}


def dice_distribution(per_image: dict[str, np.ndarray], labels: dict[str, str], colors: dict[str, str],
                      summary: dict, path: Path, metric: str = "dice") -> None:
    style()
    methods = list(per_image)
    fig, ax = plt.subplots(figsize=(1.6 + 1.35 * len(methods), 4.2))
    rng = np.random.default_rng(0)
    for i, m in enumerate(methods):
        v = per_image[m]
        ax.boxplot([v], positions=[i], widths=0.5, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=SURFACE, edgecolor=INK_2, linewidth=1),
                   medianprops=dict(color=INK, linewidth=1.5), whiskerprops=dict(color=INK_2),
                   capprops=dict(color=INK_2))
        x = i + rng.uniform(-0.18, 0.18, size=v.size)
        ax.scatter(x, v, s=8, color=colors[m], alpha=0.55, linewidths=0, zorder=3)
        s = summary[m][metric]
        ax.errorbar([i + 0.33], [s["mean"]], yerr=[[s["mean"] - s["ci_low"]], [s["ci_high"] - s["mean"]]],
                    fmt="o", color=INK, markersize=4, capsize=3, linewidth=1.2, zorder=4)
        ax.text(i + 0.4, s["mean"], f"{s['mean']:.3f}", fontsize=8, color=INK, va="center")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([labels[m] for m in methods], rotation=0, fontsize=8)
    name = NAMES.get(metric, metric)
    ax.set_ylabel(f"{name} per test image")
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-image {name} on the ARCADE test split (n = {len(next(iter(per_image.values())))}); "
                 "black dot: mean with 95% bootstrap CI", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def metric_ci(summary: dict, methods: list[str], labels: dict[str, str], colors: dict[str, str],
              metrics: tuple[str, ...], path: Path) -> None:
    style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.6 * len(metrics), 0.55 * len(methods) + 1.2), sharey=True)
    for ax, metric in zip(axes, metrics):
        for j, m in enumerate(methods):
            s = summary[m][metric]
            y = len(methods) - 1 - j
            ax.errorbar([s["mean"]], [y], xerr=[[s["mean"] - s["ci_low"]], [s["ci_high"] - s["mean"]]],
                        fmt="o", color=colors[m], markersize=6, capsize=3, linewidth=2)
        ax.set_title(NAMES.get(metric, metric), fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(methods)))
    axes[0].set_yticklabels([labels[m] for m in reversed(methods)], fontsize=8)
    fig.suptitle("Mean over test images with 95% bootstrap confidence intervals", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sensitivity_by_radius(table: dict, methods: list[str], labels: dict[str, str], colors: dict[str, str],
                          path: Path) -> None:
    style()
    bins = table["bins"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(bins))
    width = 0.8 / max(len(methods), 1)
    for j, m in enumerate(methods):
        vals = table["sensitivity"][m]
        ax.bar(x - 0.4 + width * (j + 0.5), vals, width=width - 0.02, color=colors[m], label=labels[m])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={n:,})" for b, n in zip(bins, table["n_pixels"])], fontsize=8)
    ax.set_xlabel("ground-truth vessel radius at the centerline (pixels)")
    ax.set_ylabel("share of centerline pixels detected")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", ncols=2)
    ax.set_title("Centerline sensitivity by vessel caliber (test split)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def training_curves(logs: dict[str, list[dict]], labels: dict[str, str], colors: dict[str, str], path: Path) -> None:
    style()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for m, rows in logs.items():
        pts = [(r["epoch"], float(r["val_dice"])) for r in rows if r["val_dice"] not in ("", None)]
        if not pts:
            continue
        e, d = zip(*pts)
        ax.plot(e, d, color=colors[m], linewidth=2, marker="o", markersize=3, label=labels[m])
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation Dice (threshold 0.5)")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.set_title("Validation Dice during training", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def uncertainty_scatter(unc: np.ndarray, dice: np.ndarray, rho: float, label: str, color: str, path: Path) -> None:
    style()
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.scatter(unc, dice, s=12, color=color, alpha=0.7, linewidths=0)
    ax.set_xlabel("mean predictive uncertainty near predicted vessels")
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1)
    ax.set_title(f"{label}: uncertainty vs Dice (Spearman rho = {rho:.2f})", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def overlay(image: np.ndarray, pred: np.ndarray, gt: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """RGB overlay: yellow true positive, red false positive, blue false negative."""
    g = image.astype(np.float32)
    if g.max() > 1.5:
        g = g / 255.0
    rgb = np.repeat(g[..., None], 3, axis=2)
    p, t = pred > 0, gt > 0
    for sel, col in ((p & t, TP_RGB), (p & ~t, FP_RGB), (~p & t, FN_RGB)):
        rgb[sel] = (1 - alpha) * rgb[sel] + alpha * col
    return np.clip(rgb, 0, 1)


def overlay_panel(rows: list[dict], path: Path, title: str) -> None:
    """rows: dicts with image, pred, gt, caption. One row per case: image | overlay."""
    style()
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(6.4, 3.2 * n))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(rows):
        axes[i, 0].imshow(r["image"], cmap="gray", vmin=0, vmax=255)
        axes[i, 1].imshow(overlay(r["image"], r["pred"], r["gt"]))
        axes[i, 0].set_title(r["caption"], fontsize=8, loc="left")
        axes[i, 1].set_title("yellow TP, red FP, blue FN", fontsize=8, loc="left")
        for ax in axes[i]:
            ax.set_axis_off()
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def comparison_grid(cases: list[dict], methods: list[str], labels: dict[str, str], path: Path, title: str) -> None:
    """cases: dicts with image, gt, preds (method to mask), caption. Columns: image, ground truth, methods."""
    style()
    ncol = 2 + len(methods)
    fig, axes = plt.subplots(len(cases), ncol, figsize=(2.2 * ncol, 2.35 * len(cases)))
    axes = np.atleast_2d(axes)
    for i, c in enumerate(cases):
        axes[i, 0].imshow(c["image"], cmap="gray", vmin=0, vmax=255)
        axes[i, 0].set_title(c["caption"], fontsize=7, loc="left")
        axes[i, 1].imshow(overlay(c["image"], c["gt"], c["gt"]))
        axes[i, 1].set_title("reference mask", fontsize=7, loc="left")
        for j, m in enumerate(methods):
            axes[i, 2 + j].imshow(overlay(c["image"], c["preds"][m], c["gt"]))
            axes[i, 2 + j].set_title(f"{labels[m]}\nDice {c['dice'][m]:.2f}", fontsize=7, loc="left")
        for ax in axes[i]:
            ax.set_axis_off()
    fig.suptitle(title + "  (yellow TP, red FP, blue FN)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)
