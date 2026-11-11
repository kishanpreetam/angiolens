"""Train the U-Net from a TOML config.

Usage: python -m angiolens.train configs/unet_scratch.toml

Writes
* checkpoints/<name>/best.pt   state_dict with the best validation Dice
* checkpoints/<name>/last.pt   state_dict at the end of training
* results/train_logs/<name>.csv    one row per epoch
* results/train_logs/<name>.json   config, environment, timing and best epoch
"""

from __future__ import annotations

import csv
import json
import math
import platform
import random
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from angiolens.data.dataset import CropDataset, full_frame_batch
from angiolens.models.unet import UNet, count_parameters, pick_device
from angiolens.paths import CHECKPOINTS, RESULTS, ROOT, ensure


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        cfg = tomllib.load(fh)
    cfg["_config_path"] = str(Path(path))
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def data_rows(source: str, split: str) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    """Image and mask stacks plus the cache rows and ids used for one split."""
    if source == "arcade":
        from angiolens.data.arcade import load_cache, read_index
        from angiolens.data.split import read_splits

        images, masks = load_cache(split)
        index = {r["image_id"]: r for r in read_index()}
        ids = read_splits()[split]
        rows = [index[i]["cache_row"] for i in ids]
        return images, masks, rows, ids
    if source == "synth":
        from angiolens.synth.dataset import load_synth_cache

        images, masks = load_synth_cache(split)
        ids = [f"synth_{split}_{i}" for i in range(len(images))]
        return images, masks, list(range(len(images))), ids
    raise ValueError(f"unknown data source {source!r}")


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    return 1 - (2 * inter + eps) / (prob.sum() + target.sum() + eps)


@torch.no_grad()
def predict_probs(model: torch.nn.Module, images: np.ndarray, rows: list[int], device: torch.device,
                  batch_size: int = 4) -> np.ndarray:
    """Sigmoid probabilities (N, H, W) float32 for full frames."""
    model.eval()
    out = []
    for i in range(0, len(rows), batch_size):
        x = full_frame_batch(images, rows[i:i + batch_size]).to(device)
        out.append(torch.sigmoid(model(x)).squeeze(1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,) + images.shape[1:], np.float32)


@torch.no_grad()
def mean_dice(model, images, masks, rows, device, threshold: float = 0.5, batch_size: int = 4) -> float:
    model.eval()
    scores = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        x = full_frame_batch(images, batch).to(device)
        pred = torch.sigmoid(model(x)).squeeze(1) > threshold
        gt = torch.from_numpy(np.stack([np.asarray(masks[r]) for r in batch]) > 0).to(device)
        inter = (pred & gt).sum(dim=(1, 2)).float()
        denom = (pred.sum(dim=(1, 2)) + gt.sum(dim=(1, 2))).float()
        d = torch.where(denom > 0, 2 * inter / denom.clamp(min=1), torch.ones_like(denom))
        scores.extend(d.cpu().tolist())
    return float(np.mean(scores))


def lr_at(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def environment(device: torch.device) -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": str(device),
    }


def train(cfg: dict) -> dict:
    name = cfg["name"]
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    dcfg, mcfg, tcfg, lcfg = cfg["data"], cfg["model"], cfg["train"], cfg.get("loss", {})
    device = pick_device(tcfg.get("device") or None)

    images, masks, rows, ids = data_rows(dcfg["source"], "train")
    n_train = dcfg.get("n_train")
    if n_train:
        pick = np.random.default_rng(seed).choice(len(rows), size=int(n_train), replace=False)
        rows = [rows[i] for i in sorted(pick)]
        ids = [ids[i] for i in sorted(pick)]
    v_images, v_masks, v_rows, _ = data_rows(dcfg.get("val_source", dcfg["source"]), "val")

    ds = CropDataset(images, masks, rows, crop=int(dcfg.get("crop", 256)), augment=bool(dcfg.get("augment", True)),
                     seed=seed, samples_per_epoch=dcfg.get("samples_per_epoch"))
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=int(tcfg["batch_size"]), shuffle=True, generator=gen,
                        num_workers=0, drop_last=True)

    model = UNet(base=int(mcfg.get("base", 16)), depth=int(mcfg.get("depth", 4))).to(device)
    init_from = tcfg.get("init_from", "")
    if init_from:
        state = torch.load(ROOT / init_from, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"initialized from {init_from}")
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg.get("weight_decay", 1e-4)))

    epochs = int(tcfg["epochs"])
    total_steps = epochs * len(loader)
    warmup = int(tcfg.get("warmup_steps", 100))
    max_minutes = float(tcfg.get("max_minutes", 15))
    val_every = int(tcfg.get("val_every", 1))
    bce_w, dice_w = float(lcfg.get("bce_weight", 1.0)), float(lcfg.get("dice_weight", 1.0))

    out_dir = CHECKPOINTS / name
    log_dir = RESULTS / "train_logs"
    ensure(out_dir, log_dir)
    log_rows = []
    best = {"val_dice": -1.0, "epoch": -1}
    step = 0
    t0 = time.time()
    stopped_early = False
    print(f"{name}: {len(rows)} train images, {len(v_rows)} val images, {len(loader)} steps per epoch, "
          f"{count_parameters(model) / 1e6:.2f}M params, device {device}")
    for epoch in range(epochs):
        ds.set_epoch(epoch)
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, float(tcfg["lr"]), warmup)
            logits = model(x)
            loss = bce_w * F.binary_cross_entropy_with_logits(logits, y) + dice_w * dice_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            step += 1
        elapsed = (time.time() - t0) / 60
        last_epoch = epoch == epochs - 1 or elapsed > max_minutes
        row = {"epoch": epoch + 1, "step": step, "train_loss": float(np.mean(losses)),
               "lr": opt.param_groups[0]["lr"], "minutes": round(elapsed, 2), "val_dice": ""}
        if (epoch + 1) % val_every == 0 or last_epoch:
            vd = mean_dice(model, v_images, v_masks, v_rows, device)
            row["val_dice"] = vd
            if vd > best["val_dice"]:
                best = {"val_dice": vd, "epoch": epoch + 1, "step": step}
                torch.save(model.state_dict(), out_dir / "best.pt")
        log_rows.append(row)
        print(f"  epoch {epoch + 1:3d}  loss {row['train_loss']:.4f}  val_dice {row['val_dice'] if row['val_dice'] == '' else f'{vd:.4f}'}  "
              f"{elapsed:.1f} min", flush=True)
        if elapsed > max_minutes and epoch < epochs - 1:
            stopped_early = True
            print(f"  time limit of {max_minutes} min reached, stopping")
            break
    torch.save(model.state_dict(), out_dir / "last.pt")
    with open(log_dir / f"{name}.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    summary = {
        "name": name,
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "config_path": cfg.get("_config_path"),
        "n_train_images": len(rows),
        "n_val_images": len(v_rows),
        "train_ids_sha_note": "ids listed in results/splits.csv; low-data runs list their ids below",
        "train_ids": ids if n_train else None,
        "params": count_parameters(model),
        "epochs_run": len(log_rows),
        "steps": step,
        "stopped_by_time_limit": stopped_early,
        "best": best,
        "minutes": round((time.time() - t0) / 60, 2),
        "environment": environment(device),
    }
    (log_dir / f"{name}.json").write_text(json.dumps(summary, indent=1))
    print(f"done: best val Dice {best['val_dice']:.4f} at epoch {best['epoch']}, {summary['minutes']} min")
    return summary


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m angiolens.train <config.toml>")
        return 2
    train(load_config(argv[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
