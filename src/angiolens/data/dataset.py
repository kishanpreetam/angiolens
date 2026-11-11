"""Training data: intensity normalization, random crops and augmentation.

Normalization: each frame is mapped to [0, 1] with a robust window, the 1st
and 99th intensity percentiles computed over non-border pixels (value > 5),
so black collimator borders do not shift the window. The same function runs
at training and inference time.

Augmentation (training only, numpy, seeded per sample): random crop,
horizontal and vertical flips, gamma, contrast and brightness jitter,
Gaussian noise and occasional Gaussian blur.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
import torch
from torch.utils.data import Dataset

BORDER_LEVEL = 5


def intensity_window(img: np.ndarray) -> tuple[float, float]:
    vals = img[img > BORDER_LEVEL]
    if vals.size < 0.05 * img.size:
        vals = img.ravel()
    lo, hi = np.percentile(vals, [1, 99])
    if hi - lo < 1e-3:
        hi = lo + 1.0
    return float(lo), float(hi)


def normalize(img: np.ndarray, window: tuple[float, float] | None = None) -> np.ndarray:
    lo, hi = window if window is not None else intensity_window(img)
    x = (img.astype(np.float32) - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0)


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Intensity augmentation on a [0, 1] image."""
    x = x ** rng.uniform(0.7, 1.4)
    x = (x - 0.5) * rng.uniform(0.75, 1.25) + 0.5 + rng.uniform(-0.1, 0.1)
    if rng.random() < 0.2:
        x = ndi.gaussian_filter(x, rng.uniform(0.3, 1.0))
    if rng.random() < 0.5:
        x = x + rng.normal(0.0, rng.uniform(0.0, 0.04), size=x.shape)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


class CropDataset(Dataset):
    """Random crops from uint8 image and mask stacks (N, H, W), memory mapped or not."""

    def __init__(self, images: np.ndarray, masks: np.ndarray, rows: list[int], crop: int = 256,
                 augment: bool = True, seed: int = 0, samples_per_epoch: int | None = None):
        self.images, self.masks = images, masks
        self.rows = list(rows)
        self.crop = crop
        self.do_augment = augment
        self.seed = seed
        self.epoch = 0
        self.samples_per_epoch = samples_per_epoch or len(self.rows)
        self.windows = [intensity_window(np.asarray(images[r])) for r in self.rows]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, i: int):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        k = i % len(self.rows) if self.samples_per_epoch == len(self.rows) else int(rng.integers(len(self.rows)))
        r = self.rows[k]
        h, w = self.images.shape[1:]
        y0 = int(rng.integers(0, h - self.crop + 1))
        x0 = int(rng.integers(0, w - self.crop + 1))
        img = np.asarray(self.images[r, y0:y0 + self.crop, x0:x0 + self.crop])
        msk = np.asarray(self.masks[r, y0:y0 + self.crop, x0:x0 + self.crop])
        x = normalize(img, self.windows[k])
        y = msk.astype(np.float32)
        if self.do_augment:
            if rng.random() < 0.5:
                x, y = x[:, ::-1], y[:, ::-1]
            if rng.random() < 0.5:
                x, y = x[::-1, :], y[::-1, :]
            x = augment(x, rng)
        x = np.ascontiguousarray(x)[None]
        y = np.ascontiguousarray(y)[None]
        return torch.from_numpy(x), torch.from_numpy(y)


def full_frame_batch(images: np.ndarray, rows: list[int]) -> torch.Tensor:
    """Normalized full frames (B, 1, H, W) for validation and inference."""
    arr = np.stack([normalize(np.asarray(images[r])) for r in rows])[:, None]
    return torch.from_numpy(arr)
