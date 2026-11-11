"""Evaluate every method on the held-out ARCADE test split.

Usage: python -m angiolens.eval.evaluate

Reads binary test masks from data/processed/predictions/<method>/test/ and
writes, under results/:
* metrics/per_image_<method>.csv   Dice, IoU, sensitivity, precision, clDice
* metrics/summary.json and summary.md   means with 95% bootstrap CIs
* metrics/comparisons.json              paired bootstrap and Wilcoxon tests
* metrics/sensitivity_by_radius.json    centerline sensitivity by caliber
* metrics/failure_analysis.json         error composition and correlations
* metrics/uncertainty.json, review_queue.csv
* metrics/synthetic_check.json          methods on 100 fresh phantoms
* figures/*.png and failures/*.png
"""

from __future__ import annotations

import csv
import json
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats as sps
from skimage.morphology import skeletonize

from angiolens.eval import plots
from angiolens.eval.metrics import METRIC_NAMES, all_metrics
from angiolens.eval.stats import bootstrap_ci, paired_bootstrap
from angiolens.paths import PROCESSED, RESULTS, ROOT, ensure

METHODS = [
    ("frangi", "Frangi"),
    ("unet_synth_only", "U-Net\nsynthetic only"),
    ("unet_scratch", "U-Net\nARCADE"),
    ("unet_synth_finetune", "U-Net\nsynthetic + ARCADE"),
    ("unet_scratch_n100", "U-Net\n100 ARCADE"),
    ("unet_synth_finetune_n100", "U-Net\nsynthetic + 100 ARCADE"),
]
ORDER = [m for m, _ in METHODS]
LABELS = dict(METHODS)
FLAT = {m: l.replace("\n", ", ") for m, l in METHODS}
COMPARISONS = [
    ("unet_scratch", "frangi"),
    ("unet_synth_finetune", "frangi"),
    ("unet_synth_finetune", "unet_scratch"),
    ("unet_synth_finetune_n100", "unet_scratch_n100"),
    ("unet_scratch", "unet_scratch_n100"),
    ("unet_scratch", "unet_synth_only"),
]
RADIUS_BINS = [(0, 2), (2, 3), (3, 4), (4, 6), (6, 99)]
FULL_DATA_UNETS = ["unet_scratch", "unet_synth_finetune"]
N_BOOT = 10000

M = RESULTS / "metrics"
F = RESULTS / "figures"
FAIL = RESULTS / "failures"


def read_png(path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im)


def available_methods() -> list[str]:
    return [m for m in ORDER if (PROCESSED / "predictions" / m / "test").is_dir()]


def gt_geometry(gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = gt > 0
    skel = skeletonize(m)
    radius = ndi.distance_transform_edt(m)
    return skel, radius


def error_composition(pred: np.ndarray, gt: np.ndarray, near_px: float = 2.0) -> dict:
    """Split errors into boundary disagreement (within near_px of the other mask) and whole structures."""
    p, g = pred > 0, gt > 0
    fp, fn = p & ~g, ~p & g
    d_to_gt = ndi.distance_transform_edt(~g) if g.any() else np.full(g.shape, np.inf)
    d_to_pred = ndi.distance_transform_edt(~p) if p.any() else np.full(p.shape, np.inf)
    return {
        "fp": int(fp.sum()), "fn": int(fn.sum()),
        "fp_near": int((fp & (d_to_gt <= near_px)).sum()),
        "fn_near": int((fn & (d_to_pred <= near_px)).sum()),
    }


def write_csv(path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def primary_model(methods: list[str]) -> str | None:
    """The full-data U-Net with the best validation Dice (chosen on validation, not test)."""
    best, best_val = None, -1.0
    for m in FULL_DATA_UNETS:
        path = RESULTS / "thresholds" / f"{m}.json"
        if m in methods and path.exists():
            v = json.loads(path.read_text())["selected"]["val_mean_dice"]
            if v > best_val:
                best, best_val = m, v
    return best


def synthetic_check(n: int = 100) -> dict:
    """Run the methods on fresh phantoms with exact masks (seeds never used in training)."""
    import torch

    from angiolens.baseline.frangi import binarize, vesselness
    from angiolens.data.dataset import normalize
    from angiolens.models.unet import pick_device
    from angiolens.predict import load_model
    from angiolens.synth.generator import generate_pair

    pairs = []
    for i in range(n):
        img, mask = generate_pair(np.random.default_rng(3_000_000 + i))
        pairs.append((np.round(img * 255).astype(np.uint8), mask))
    out = {"n": n, "seed_base": 3_000_000, "methods": {}}
    fr_path = RESULTS / "frangi_tuning.json"
    if fr_path.exists():
        sel = json.loads(fr_path.read_text())["selected"]
        dices = [all_metrics(binarize(vesselness(im), sel["rule"], sel["value"], sel["min_size"]), m)
                 for im, m in pairs]
        out["methods"]["frangi"] = {k: float(np.mean([d[k] for d in dices])) for k in METRIC_NAMES}
    device = pick_device()
    for name in ("unet_synth_only", "unet_scratch", "unet_synth_finetune"):
        thr_path = RESULTS / "thresholds" / f"{name}.json"
        if not thr_path.exists():
            continue
        thr = json.loads(thr_path.read_text())["selected"]["threshold"]
        model, _ = load_model(name, device=device)
        res = []
        with torch.no_grad():
            for im, m in pairs:
                x = torch.from_numpy(normalize(im)[None, None]).to(device)
                p = torch.sigmoid(model(x))[0, 0].cpu().numpy()
                res.append(all_metrics((p >= thr).astype(np.uint8), m))
        out["methods"][name] = {k: float(np.mean([d[k] for d in res])) for k in METRIC_NAMES}
        out["methods"][name]["threshold"] = thr
    return out


def main() -> int:
    from angiolens.data.arcade import read_index
    from angiolens.train import data_rows

    ensure(M, F, FAIL)
    methods = available_methods()
    if not methods:
        print("no predictions found under data/processed/predictions")
        return 1
    images, masks, rows, ids = data_rows("arcade", "test")
    index = {r["image_id"]: r for r in read_index()}
    gts = [np.asarray(masks[r]) for r in rows]
    geometry = [gt_geometry(g) for g in gts]
    colors = plots.method_colors(methods, ORDER)

    per_image: dict[str, list[dict]] = {}
    radius_hits = {m: np.zeros(len(RADIUS_BINS)) for m in methods}
    radius_tot = np.zeros(len(RADIUS_BINS))
    for (skel, radius) in geometry:
        rr = radius[skel]
        for b, (lo, hi) in enumerate(RADIUS_BINS):
            radius_tot[b] += np.count_nonzero((rr >= lo) & (rr < hi))
    preds_cache: dict[str, list[np.ndarray]] = {}
    for m in methods:
        pred_dir = PROCESSED / "predictions" / m / "test"
        out_rows, preds = [], []
        for image_id, gt, (skel, radius) in zip(ids, gts, geometry):
            pred = (read_png(pred_dir / f"{image_id}.png") > 0).astype(np.uint8)
            preds.append(pred)
            met = all_metrics(pred, gt)
            comp = error_composition(pred, gt)
            out_rows.append({"image_id": image_id, **{k: round(v, 6) if isinstance(v, float) else v
                                                      for k, v in met.items()}, **comp})
            rr, hit = radius[skel], pred[skel] > 0
            for b, (lo, hi) in enumerate(RADIUS_BINS):
                sel = (rr >= lo) & (rr < hi)
                radius_hits[m][b] += np.count_nonzero(hit & sel)
        per_image[m] = out_rows
        preds_cache[m] = preds
        write_csv(M / f"per_image_{m}.csv", out_rows)
        print(f"{m}: mean Dice {np.mean([r['dice'] for r in out_rows]):.4f}")

    summary = {}
    for m in methods:
        summary[m] = {k: bootstrap_ci([r[k] for r in per_image[m]], n_boot=N_BOOT, seed=1) for k in METRIC_NAMES}
        tp = sum(r["tp"] for r in per_image[m])
        fp = sum(r["fp"] for r in per_image[m])
        fn = sum(r["fn"] for r in per_image[m])
        summary[m]["pooled_dice"] = 2 * tp / (2 * tp + fp + fn)
        summary[m]["n_empty_predictions"] = int(sum(1 for r in per_image[m] if r["tp"] + r["fp"] == 0))
    meta = {"split": "ARCADE syntax test (official split, leakage guard applied)", "n_images": len(ids),
            "ci": f"95% percentile bootstrap over images, {N_BOOT} resamples", "labels": FLAT}
    (M / "summary.json").write_text(json.dumps({"meta": meta, "methods": summary}, indent=1))

    lines = ["# Test results (ARCADE syntax test split, n = %d)" % len(ids), "",
             "Mean over images with 95% bootstrap CI (10,000 resamples).", "",
             "| Method | Dice | IoU | Sensitivity | Precision | clDice |", "|---|---|---|---|---|---|"]
    for m in methods:
        cells = [f"{summary[m][k]['mean']:.3f} [{summary[m][k]['ci_low']:.3f}, {summary[m][k]['ci_high']:.3f}]"
                 for k in METRIC_NAMES]
        lines.append(f"| {FLAT[m]} | " + " | ".join(cells) + " |")
    (M / "summary.md").write_text("\n".join(lines) + "\n")

    comparisons = []
    for a, b in COMPARISONS:
        if a in methods and b in methods:
            for k in ("dice", "cldice"):
                res = paired_bootstrap([r[k] for r in per_image[a]], [r[k] for r in per_image[b]],
                                       n_boot=N_BOOT, seed=2)
                comparisons.append({"a": a, "b": b, "metric": k, **res})
    (M / "comparisons.json").write_text(json.dumps(comparisons, indent=1))

    radius_table = {
        "bins": [f"{lo}-{hi}" if hi < 99 else f">={lo}" for lo, hi in RADIUS_BINS],
        "n_pixels": [int(x) for x in radius_tot],
        "sensitivity": {m: [float(h / t) if t else 0.0 for h, t in zip(radius_hits[m], radius_tot)] for m in methods},
        "note": "share of ground-truth skeleton pixels covered by the predicted mask, binned by the distance "
                "transform radius at that skeleton pixel",
    }
    (M / "sensitivity_by_radius.json").write_text(json.dumps(radius_table, indent=1))

    dice_arrays = {m: np.array([r["dice"] for r in per_image[m]]) for m in methods}
    plots.dice_distribution(dice_arrays, LABELS, colors, summary, F / "dice_by_method.png")
    plots.dice_distribution({m: np.array([r["cldice"] for r in per_image[m]]) for m in methods},
                            LABELS, colors, summary, F / "cldice_by_method.png", metric="cldice")
    plots.metric_ci(summary, methods, FLAT, colors, METRIC_NAMES, F / "metrics_ci.png")
    plots.sensitivity_by_radius(radius_table, methods, FLAT, colors, F / "sensitivity_by_radius.png")

    logs = {}
    for m in methods:
        p = RESULTS / "train_logs" / f"{m}.csv"
        if p.exists():
            with open(p, newline="") as fh:
                logs[m] = [{"epoch": int(r["epoch"]), "val_dice": r["val_dice"]} for r in csv.DictReader(fh)]
    if logs:
        curve_labels = dict(FLAT)
        curve_labels["unet_synth_only"] = FLAT["unet_synth_only"] + " (synthetic validation set)"
        plots.training_curves(logs, curve_labels, colors, F / "training_curves.png")

    failure = {}
    primary = primary_model(methods)
    if primary:
        d = dice_arrays[primary]
        order = np.argsort(d)
        qa_rows = {}
        qa_csv = RESULTS / "data_qa_per_image.csv"
        if qa_csv.exists():
            with open(qa_csv, newline="") as fh:
                qa_rows = {r["image_id"]: r for r in csv.DictReader(fh)}
        props = {
            "mean_intensity": [float(qa_rows[i]["mean"]) for i in ids] if qa_rows else None,
            "intensity_sd": [float(qa_rows[i]["std"]) for i in ids] if qa_rows else None,
            "vessel_fraction": [index[i]["vessel_fraction"] for i in ids],
            "n_segments": [index[i]["n_segments"] for i in ids],
            "black_border_fraction": [float(qa_rows[i]["frac_black"]) for i in ids] if qa_rows else None,
        }
        correlations = {}
        for k, v in props.items():
            if v is not None:
                rho, p = sps.spearmanr(v, d)
                correlations[k] = {"spearman_rho": float(rho), "p_value": float(p)}
        comp = {k: int(sum(r[k] for r in per_image[primary])) for k in ("fp", "fn", "fp_near", "fn_near")}
        worst = [{"image_id": ids[i], "dice": float(d[i]), "cldice": float(per_image[primary][i]["cldice"]),
                  "sensitivity": float(per_image[primary][i]["sensitivity"]),
                  "precision": float(per_image[primary][i]["precision"])} for i in order[:10]]
        failure = {
            "primary_model": primary,
            "primary_selection": "full-data U-Net with the higher validation Dice",
            "dice_quartiles": [float(np.percentile(d, q)) for q in (25, 50, 75)],
            "n_dice_below_0.5": int((d < 0.5).sum()),
            "error_pixels": comp,
            "share_fp_within_2px_of_gt": comp["fp_near"] / max(comp["fp"], 1),
            "share_fn_within_2px_of_pred": comp["fn_near"] / max(comp["fn"], 1),
            "spearman_dice_vs_image_properties": correlations,
            "worst_10": worst,
        }
        panel = []
        for i in order[:6]:
            img = np.asarray(images[rows[i]])
            pr = preds_cache[primary][i]
            Image.fromarray((plots.overlay(img, pr, gts[i]) * 255).astype(np.uint8)).save(
                FAIL / f"{primary}_worst_{ids[i]}.png")
            panel.append({"image": img, "pred": pr, "gt": gts[i],
                          "caption": f"{ids[i]}: Dice {d[i]:.3f}, clDice {per_image[primary][i]['cldice']:.3f}"})
        plots.overlay_panel(panel, FAIL / "worst_cases.png", f"Six lowest-Dice test images, {FLAT[primary]}")

        show = [m for m in ("frangi", "unet_scratch", "unet_synth_finetune") if m in methods]
        cases = []
        for q in (75, 50, 25):
            i = int(order[int(round((len(order) - 1) * q / 100))])
            cases.append({"image": np.asarray(images[rows[i]]), "gt": gts[i],
                          "preds": {m: preds_cache[m][i] for m in show},
                          "dice": {m: dice_arrays[m][i] for m in show},
                          "caption": f"{ids[i]}, {q}th pct"})
        plots.comparison_grid(cases, show, FLAT, F / "qualitative_comparison.png",
                              f"Test images at the 75th, 50th and 25th percentile of Dice for {FLAT[primary]}")

        unc_out = {}
        for m in methods:
            prob_dir = PROCESSED / "predictions" / m / "test_prob"
            if not prob_dir.is_dir():
                continue
            unc = []
            for image_id in ids:
                p = read_png(prob_dir / f"{image_id}.png").astype(np.float32) / 255.0
                near = ndi.binary_dilation(p >= 0.5, iterations=3) | (p >= 0.1)
                u = 4 * p * (1 - p)
                unc.append(float(u[near].mean()) if near.any() else 0.0)
            rho, pval = sps.spearmanr(unc, dice_arrays[m])
            unc_out[m] = {"spearman_rho_uncertainty_vs_dice": float(rho), "p_value": float(pval),
                          "definition": "mean of 4p(1-p) over pixels with p >= 0.1 or within 3 px of p >= 0.5"}
            if m == primary:
                plots.uncertainty_scatter(np.array(unc), dice_arrays[m], float(rho), FLAT[m], colors[m],
                                          F / "uncertainty_vs_dice.png")
                k = 25
                top = np.argsort(unc)[::-1][:k]
                top_dice = dice_arrays[m][top]
                unc_out[m]["review_queue"] = {
                    "k": k, "mean_dice_top_k_uncertain": float(top_dice.mean()),
                    "mean_dice_all": float(dice_arrays[m].mean()),
                    "share_of_10_worst_in_top_k": float(len(set(top.tolist()) & set(order[:10].tolist())) / 10),
                }
                queue = [{"image_id": ids[i],
                          "image_path": index[ids[i]]["image_path"],
                          "pred_path": str((PROCESSED / "predictions" / m / "test" / f"{ids[i]}.png").relative_to(ROOT)),
                          "gt_path": index[ids[i]]["mask_path"],
                          "priority": round(unc[i], 6)} for i in top]
                write_csv(RESULTS / "review_queue.csv", queue)
        (M / "uncertainty.json").write_text(json.dumps(unc_out, indent=1))
    (M / "failure_analysis.json").write_text(json.dumps(failure, indent=1))

    synth = synthetic_check()
    (M / "synthetic_check.json").write_text(json.dumps(synth, indent=1))
    print((M / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
