"""C3DIR network (paper section 4): a U-Net of ConvNeXt blocks with an asymmetric decoder
that keeps a constant width, ending in a 1x1 conv folded into 80 levels.

output per voxel (8 channels): ice / liquid / rain mask logits, ice / liquid / rain
water content scaled as log10(1 + w / 5e-5), ice / liquid effective radius scaled by
20 / 5 um. the any-hydrometeor logit is the max of the three species (paper section 4).
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LEVELS = 80
CHANNELS = 8  # 3 mask logits, 3 water contents, 2 radii
WATER_SCALE = 5e-5  # g m^-3, paper eq. 5
RADIUS_SCALE = (20.0, 5.0)  # um, paper section 4.5
RADIUS_MAX = (140.0, 35.0)  # um


@dataclass
class ModelConfig:
    in_channels: int = 16
    width: int = 64  # paper: 64 -> 256
    depth: int = 3  # number of 2x downsamplings
    max_width: int = 256
    kernel: int = 5
    dropout: float = 0.125
    drop_path: float = 0.25

    def as_dict(self):
        return asdict(self)


class Block(nn.Module):
    """ConvNeXt block: depthwise conv, layer norm, 4x MLP with dropout, stochastic depth."""

    def __init__(self, channels, kernel, dropout, drop_path):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel, padding=kernel // 2, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, 4 * channels), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * channels, channels))
        self.drop_path = drop_path

    def forward(self, x):
        y = self.mlp(self.norm(self.depthwise(x).permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        if self.training and self.drop_path > 0:
            keep = torch.rand(x.shape[0], 1, 1, 1, device=x.device) >= self.drop_path
            y = y * keep / (1 - self.drop_path)
        return x + y


class C3DIRNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        c = self.config = config or ModelConfig()
        widths = [min(c.width * 2 ** i, c.max_width) for i in range(c.depth + 1)]
        n_blocks = 2 * (c.depth + 1) + c.depth  # encoder 2 per stage, decoder 1 per stage
        rates = iter(np.linspace(0, c.drop_path, n_blocks).tolist())

        def block(ch):
            return Block(ch, c.kernel, c.dropout, next(rates))

        self.stem = nn.Conv2d(c.in_channels, widths[0], 3, padding=1)
        self.encoder = nn.ModuleList()
        self.widen = nn.ModuleList()
        for i, ch in enumerate(widths):
            self.encoder.append(nn.Sequential(block(ch), block(ch)))
            if i < c.depth:
                self.widen.append(nn.Conv2d(ch, widths[i + 1], 1))
        top = widths[-1]
        self.decoder = nn.ModuleList(
            nn.Sequential(nn.Conv2d(top + ch, top, 1), block(top)) for ch in reversed(widths[:-1]))
        self.head = nn.Conv2d(top, CHANNELS * LEVELS, 1)

    def forward(self, x):
        skips = []
        h = self.stem(x)
        for i, stage in enumerate(self.encoder):
            h = stage(h)
            if i < len(self.widen):
                skips.append(h)
                h = self.widen[i](F.avg_pool2d(h, 2))
        for stage, skip in zip(self.decoder, reversed(skips)):
            h = stage(torch.cat([F.interpolate(h, size=skip.shape[-2:], mode="nearest"), skip], dim=1))
        out = self.head(h)
        return out.reshape(x.shape[0], CHANNELS, LEVELS, *x.shape[-2:])  # (B, 8, 80, H, W)


def mask_logits(raw):
    """(..., 8) raw outputs -> (..., 4) logits in the order any, ice, liquid, rain."""
    species = raw[..., :3]
    return torch.cat([species.amax(dim=-1, keepdim=True), species], dim=-1)


def scale_water(water):
    return torch.log10(1 + water.clamp_min(0) / WATER_SCALE)


def scale_radius(radius):
    limits = torch.tensor(RADIUS_MAX, device=radius.device)
    return radius.clamp(min=0).minimum(limits) / torch.tensor(RADIUS_SCALE, device=radius.device)


def decode(raw):
    """(..., 8) raw outputs -> probability (..., 4: any/ice/liquid/rain), water g m^-3 (..., 3),
    radius um (..., 2)."""
    probability = torch.sigmoid(mask_logits(raw))
    water = (WATER_SCALE * (10 ** raw[..., 3:6].clamp(max=10) - 1)).clamp_min(0)
    radius = raw[..., 6:8].clamp_min(0) * torch.tensor(RADIUS_SCALE, device=raw.device)
    return probability, water, radius
