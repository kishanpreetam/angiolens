"""A small 2D U-Net (Ronneberger et al., 2015) for binary vessel segmentation.

Encoder: `depth` stages of two 3x3 conv + batch norm + ReLU, with 2x2 max
pooling in between; channels double at each stage starting from `base`.
Decoder: bilinear 2x upsampling, concatenation with the skip connection,
then the same double conv. A 1x1 conv gives one logit per pixel.
The network is fully convolutional, so it trains on crops and runs on full
512 x 512 frames (sides must be divisible by 2**depth).
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 16, depth: int = 4):
        super().__init__()
        self.depth = depth
        chans = [base * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList([DoubleConv(in_channels, chans[0])])
        for i in range(depth):
            self.down.append(DoubleConv(chans[i], chans[i + 1]))
        self.up = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up.append(DoubleConv(chans[i + 1] + chans[i], chans[i]))
        self.head = nn.Conv2d(chans[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.down):
            if i > 0:
                x = F.max_pool2d(x, 2)
            x = block(x)
            skips.append(x)
        x = skips.pop()
        for block in self.up:
            skip = skips.pop()
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pick_device(prefer: str | None = None) -> torch.device:
    """Apple GPU (MPS) first, then CUDA, then CPU. `prefer` forces one."""
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
