"""Run a trained U-Net, tune its probability threshold on validation, predict test.

Usage: python -m angiolens.predict <run_name> [checkpoint_file] [threshold_source]

threshold_source is "arcade" (default) or "synth". The synthetic-only model
uses "synth" so that no real label touches it before the test.

The architecture comes from results/train_logs/<run_name>.json, the weights
from checkpoints/<run_name>/best.pt (loaded with weights_only=True). The
threshold with the highest mean validation Dice is frozen before the test
split is touched.

Writes
* data/processed/predictions/<run>/test/<id>.png       binary masks (0/255)
* data/processed/predictions/<run>/test_prob/<id>.png  probabilities x 255
* results/thresholds/<run>.json                        threshold search on val
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
from PIL import Image

from angiolens.data.dataset import full_frame_batch
from angiolens.models.unet import UNet, pick_device
from angiolens.paths import CHECKPOINTS, PROCESSED, RESULTS, ensure
from angiolens.train import data_rows

THRESHOLDS = tuple(np.round(np.arange(0.1, 0.91, 0.05), 2))


def load_model(name: str, checkpoint: str = "best.pt", device: torch.device | None = None) -> tuple[UNet, dict]:
    summary = json.loads((RESULTS / "train_logs" / f"{name}.json").read_text())
    mcfg = summary["config"]["model"]
    model = UNet(base=int(mcfg.get("base", 16)), depth=int(mcfg.get("depth", 4)))
    state = torch.load(CHECKPOINTS / name / checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device or pick_device())
    model.eval()
    return model, summary


@torch.no_grad()
def _probs(model, images, rows, device, batch_size=4):
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        x = full_frame_batch(images, batch).to(device)
        yield batch, torch.sigmoid(model(x)).squeeze(1).float().cpu().numpy()


def tune_threshold(model, device, source: str = "arcade") -> dict:
    images, masks, rows, _ = data_rows(source, "val")
    per_t = {float(t): [] for t in THRESHOLDS}
    for batch, probs in _probs(model, images, rows, device):
        for r, p in zip(batch, probs):
            g = np.asarray(masks[r]) > 0
            gs = g.sum()
            for t in THRESHOLDS:
                pr = p >= t
                inter = np.count_nonzero(pr & g)
                den = pr.sum() + gs
                per_t[float(t)].append(2 * inter / den if den else 1.0)
    grid = [{"threshold": t, "val_mean_dice": float(np.mean(v))} for t, v in per_t.items()]
    best = max(grid, key=lambda d: d["val_mean_dice"])
    return {"n_val": len(rows), "grid": grid, "selected": best,
            "val_mean_dice_at_0.5": float(np.mean(per_t[0.5]))}


def predict_test(name: str, checkpoint: str = "best.pt", threshold_source: str = "arcade") -> dict:
    device = pick_device()
    model, _ = load_model(name, checkpoint, device)
    t0 = time.time()
    tuning = tune_threshold(model, device, threshold_source)
    thr = tuning["selected"]["threshold"]
    images, _, rows, ids = data_rows("arcade", "test")
    out_mask = PROCESSED / "predictions" / name / "test"
    out_prob = PROCESSED / "predictions" / name / "test_prob"
    ensure(out_mask, out_prob, RESULTS / "thresholds")
    id_of = dict(zip(rows, ids))
    for batch, probs in _probs(model, images, rows, device):
        for r, p in zip(batch, probs):
            Image.fromarray(((p >= thr) * 255).astype(np.uint8)).save(out_mask / f"{id_of[r]}.png")
            Image.fromarray(np.round(p * 255).astype(np.uint8)).save(out_prob / f"{id_of[r]}.png")
    report = {"run": name, "checkpoint": checkpoint, "device": str(device), "n_test": len(rows),
              "threshold_source": threshold_source,
              "selection": "threshold with the highest mean validation Dice; test never used",
              **tuning, "seconds": round(time.time() - t0, 1)}
    (RESULTS / "thresholds" / f"{name}.json").write_text(json.dumps(report, indent=1))
    print(f"{name}: threshold {thr} (val Dice {tuning['selected']['val_mean_dice']:.4f}, "
          f"at 0.5 {tuning['val_mean_dice_at_0.5']:.4f}); wrote {len(rows)} test masks")
    return report


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: python -m angiolens.predict <run_name> [checkpoint_file] [threshold_source]")
        return 2
    predict_test(argv[0], argv[1] if len(argv) > 1 else "best.pt", argv[2] if len(argv) > 2 else "arcade")
    return 0


if __name__ == "__main__":
    sys.exit(main())
