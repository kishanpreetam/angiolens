"""Synthetic X-ray angiogram generator with exact vessel masks (a digital phantom).

Geometry
* Each image holds one or two vessel trees. A tree grows from a root point as
  a smooth random walk (the angular velocity follows an AR(1) process), with a
  radius that tapers slowly. Side branches split off at random; the child
  radius is a fraction of the parent's and the parent keeps the radius given
  by Murray's law, r_parent_after**3 = r_parent**3 - r_child**3.
* Radii are in pixels at 512 x 512 and scale with the output size. The
  default root radius range (3 to 6.5 px) brackets the ARCADE training masks,
  whose skeleton radius has a median of 5.0 px and a 95th percentile of
  8.9 px (results/data_qa.json).

Image formation
* Each centerline point is a cylinder cross-section: a pixel at distance d
  from a point of radius r gets a path length 2 * sqrt(r**2 - d**2) through
  the vessel. Within a tree the maximum is kept; separate trees add up.
* Transmitted intensity follows Beer-Lambert: I = I0 * exp(-mu * length),
  with I0 a smooth background (heart and diaphragm shadow, soft bands like
  the spine or ribs, low frequency variation and fine texture).
* A catheter (constant radius tube ending at a tree root) is drawn with
  probability `catheter_prob`. It is NOT in the mask, as in ARCADE.
* Detector blur (Gaussian), signal dependent quantum noise plus electronic
  noise, and sometimes black collimator borders.

The mask is exactly the set of pixels inside a labeled vessel (d < r), so
labels have no annotation noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.special import expit


@dataclass
class SynthConfig:
    size: int = 512
    n_trees: tuple[int, int] = (1, 2)
    root_radius: tuple[float, float] = (3.0, 6.5)
    min_radius: float = 1.0
    step: float = 1.0
    branch_rate: float = 0.01
    child_ratio: tuple[float, float] = (0.5, 0.85)
    max_depth: int = 3
    max_points: int = 6000
    curvature: float = 0.006
    taper: tuple[float, float] = (0.997, 0.9993)
    mu: tuple[float, float] = (0.05, 0.12)
    catheter_prob: float = 0.6
    catheter_radius: tuple[float, float] = (2.0, 3.5)
    blur_sigma: tuple[float, float] = (0.5, 1.3)
    quantum_noise: tuple[float, float] = (0.01, 0.04)
    electronic_noise: tuple[float, float] = (0.003, 0.012)
    border_prob: float = 0.25


def _walk(rng: np.random.Generator, start: np.ndarray, theta: float, radius: float, cfg: SynthConfig,
          scale: float, depth: int, taper: float, branches: list, children: list, budget: list) -> None:
    """Grow one branch; append (x, y, r) points to `branches` and queue children."""
    pts = []
    p = start.astype(np.float64).copy()
    r = radius
    omega = 0.0
    margin = 2.0 * cfg.root_radius[1] * scale
    lo, hi = -margin, cfg.size - 1 + margin
    while r >= cfg.min_radius * scale and budget[0] > 0:
        omega = 0.92 * omega + rng.normal(0.0, cfg.curvature)
        theta += omega
        p = p + cfg.step * np.array([np.cos(theta), np.sin(theta)])
        if not (lo <= p[0] <= hi and lo <= p[1] <= hi):
            break
        r *= taper
        pts.append((p[0], p[1], r))
        budget[0] -= 1
        if depth < cfg.max_depth and rng.random() < cfg.branch_rate and r > 1.6 * cfg.min_radius * scale:
            rc = r * rng.uniform(*cfg.child_ratio)
            r = max((r**3 - rc**3) ** (1 / 3), cfg.min_radius * scale)
            side = rng.choice([-1.0, 1.0])
            children.append((p.copy(), theta + side * rng.uniform(0.35, 1.2), rc, depth + 1))
    if pts:
        branches.append(np.asarray(pts, dtype=np.float64))


def grow_tree(rng: np.random.Generator, cfg: SynthConfig, root: np.ndarray, theta: float,
              radius: float) -> list[np.ndarray]:
    scale = cfg.size / 512
    branches: list[np.ndarray] = []
    queue = [(root, theta, radius, 0)]
    budget = [cfg.max_points]
    while queue and budget[0] > 0:
        start, th, r, depth = queue.pop(0)
        children: list = []
        _walk(rng, start, th, r, cfg, scale, depth, rng.uniform(*cfg.taper), branches, children, budget)
        queue.extend(children)
    return branches


def render_thickness(branches: list[np.ndarray], size: int) -> np.ndarray:
    """Path length through the vessels at every pixel (max within this tree)."""
    thick = np.zeros((size, size), dtype=np.float32)
    for pts in branches:
        for x, y, r in pts:
            x0, x1 = int(np.floor(x - r)), int(np.ceil(x + r)) + 1
            y0, y1 = int(np.floor(y - r)), int(np.ceil(y + r)) + 1
            x0c, x1c, y0c, y1c = max(x0, 0), min(x1, size), max(y0, 0), min(y1, size)
            if x0c >= x1c or y0c >= y1c:
                continue
            yy, xx = np.mgrid[y0c:y1c, x0c:x1c]
            d2 = (xx - x) ** 2 + (yy - y) ** 2
            t = 2.0 * np.sqrt(np.maximum(r * r - d2, 0.0))
            np.maximum(thick[y0c:y1c, x0c:x1c], t, out=thick[y0c:y1c, x0c:x1c])
    return thick


def _smooth_noise(rng: np.random.Generator, size: int, sigma: float) -> np.ndarray:
    small = max(8, int(size / max(sigma, 1.0)) * 2)
    field = ndi.gaussian_filter(rng.normal(size=(small, small)), 1.5)
    field = ndi.zoom(field, size / small, order=3)[:size, :size]
    return (field - field.mean()) / (field.std() + 1e-8)


def background(rng: np.random.Generator, size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    bg = np.full((size, size), rng.uniform(0.5, 0.8), dtype=np.float32)
    bg += rng.uniform(0.03, 0.1) * _smooth_noise(rng, size, size / 4)
    # Heart or diaphragm shadow: a large disk with a soft edge.
    cx, cy = rng.uniform(0.2, 0.8), rng.uniform(0.5, 1.2)
    rad = rng.uniform(0.35, 0.7)
    edge = rng.uniform(0.01, 0.04)
    inside = expit((rad - np.hypot(xx - cx, yy - cy)) / edge)
    bg -= rng.uniform(0.05, 0.2) * inside
    # Soft bands (spine, ribs).
    for _ in range(rng.integers(0, 4)):
        ang = rng.uniform(0, np.pi)
        off = rng.uniform(0, 1)
        width = rng.uniform(0.02, 0.12)
        dist = (xx - 0.5) * np.cos(ang) + (yy - 0.5) * np.sin(ang) - (off - 0.5)
        bg += rng.uniform(-0.06, 0.06) * np.exp(-0.5 * (dist / width) ** 2)
    bg += rng.uniform(0.005, 0.03) * _smooth_noise(rng, size, rng.uniform(2.0, 6.0))
    return np.clip(bg, 0.05, 1.0)


def catheter(rng: np.random.Generator, cfg: SynthConfig, tip: np.ndarray, theta: float) -> list[np.ndarray]:
    """A constant radius tube from the tree root back out of the image."""
    scale = cfg.size / 512
    r = rng.uniform(*cfg.catheter_radius) * scale
    pts = []
    p = tip.astype(np.float64).copy()
    th = theta + np.pi + rng.normal(0, 0.3)
    omega = 0.0
    for _ in range(int(2.5 * cfg.size)):
        omega = 0.95 * omega + rng.normal(0, 0.004)
        th += omega
        p = p + np.array([np.cos(th), np.sin(th)])
        if not (-10 <= p[0] <= cfg.size + 10 and -10 <= p[1] <= cfg.size + 10):
            break
        pts.append((p[0], p[1], r))
    return [np.asarray(pts)] if pts else []


def generate_pair(rng: np.random.Generator, cfg: SynthConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One synthetic angiogram (float32 in [0, 1]) and its exact mask (uint8 0/1)."""
    cfg = cfg or SynthConfig()
    size = cfg.size
    scale = size / 512
    labeled = np.zeros((size, size), dtype=np.float32)
    distractor = np.zeros((size, size), dtype=np.float32)
    n_trees = int(rng.integers(cfg.n_trees[0], cfg.n_trees[1] + 1))
    for t in range(n_trees):
        root = rng.uniform(0.15, 0.85, size=2) * size
        theta = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(*cfg.root_radius) * scale
        branches = grow_tree(rng, cfg, root, theta, radius)
        labeled += render_thickness(branches, size)
        if t == 0 and rng.random() < cfg.catheter_prob:
            distractor += render_thickness(catheter(rng, cfg, root, theta), size)
    mask = (labeled > 0).astype(np.uint8)
    mu = rng.uniform(*cfg.mu) / scale
    i0 = background(rng, size)
    img = i0 * np.exp(-mu * labeled - mu * rng.uniform(1.0, 1.8) * distractor)
    img = ndi.gaussian_filter(img, rng.uniform(*cfg.blur_sigma) * scale)
    img = img + rng.normal(size=img.shape) * rng.uniform(*cfg.quantum_noise) * np.sqrt(np.clip(img, 0, None))
    img = img + rng.normal(size=img.shape) * rng.uniform(*cfg.electronic_noise)
    if rng.random() < cfg.border_prob:
        widths = rng.integers(0, int(30 * scale) + 1, size=4) * (rng.random(4) < 0.6)
        top, bottom, left, right = (int(w) for w in widths)
        keep = np.zeros_like(mask, dtype=bool)
        keep[top:size - bottom, left:size - right] = True
        img = np.where(keep, img, rng.uniform(0.0, 0.02))
        mask = mask * keep
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return img, mask.astype(np.uint8)
