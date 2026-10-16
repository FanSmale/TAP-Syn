"""Learned background generator for unconditional residual velocity generation.

This module defines ``B_psi`` for the unconditional path. It maps a spatial
noise seed ``xi`` directly to ``m_c_cond``: the same globally normalized coarse
background condition used by the conditional path.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import DEFAULT_NORMALIZATION_CONFIG


class _BPsiResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class _FixedGaussianBlur(nn.Module):
    def __init__(self, kernel_size: int, sigma: float) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if sigma <= 0.0:
            raise ValueError("sigma must be positive.")
        radius = kernel_size // 2
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-(coords.square()) / (2.0 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d / kernel_2d.sum()
        self.padding = radius
        self.register_buffer("kernel", kernel_2d.view(1, 1, kernel_size, kernel_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel.to(device=x.device, dtype=x.dtype).repeat(x.shape[1], 1, 1, 1)
        padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode="reflect")
        return F.conv2d(padded, kernel, groups=x.shape[1])


class BPsi(nn.Module):
    """Generate ``m_c_cond`` from a spatial noise seed.

    The returned tensor has the same semantic meaning as conditional-path
    degraded coarse backgrounds after global normalization to ``[0,1]``.
    """

    def __init__(
        self,
        height: int = 128,
        width: int = 128,
        noise_channels: int = 1,
        base_channels: int = 64,
        output_size: int = 32,
        final_blur_kernel_size: int = 3,
        final_blur_sigma: float = 0.7,
    ) -> None:
        super().__init__()
        if height != width:
            raise ValueError("BPsi currently expects square spatial inputs.")
        if output_size <= 0 or height % output_size != 0 or width % output_size != 0:
            raise ValueError("output_size must evenly divide the BPsi spatial size.")
        self.height = height
        self.width = width
        self.noise_channels = noise_channels
        self.output_size = output_size
        self.context_size = max(output_size // 2, 8)
        self.output_scale = height // output_size

        mid_channels = base_channels * 2
        context_channels = base_channels * 4

        self.semantic_stem = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.output_scale, stride=self.output_scale),
            nn.Conv2d(noise_channels, base_channels, kernel_size=5, padding=2),
            nn.SiLU(),
            _BPsiResBlock(base_channels),
            _BPsiResBlock(base_channels),
        )
        self.context_stem = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.context_size, self.context_size)),
            nn.Conv2d(noise_channels, mid_channels, kernel_size=5, padding=2),
            nn.SiLU(),
            _BPsiResBlock(mid_channels, dilation=2),
            _BPsiResBlock(mid_channels, dilation=3),
        )
        self.context_to_semantic = nn.Sequential(
            nn.Conv2d(mid_channels, context_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            _BPsiResBlock(context_channels, dilation=2),
        )
        self.semantic_fuse = nn.Sequential(
            nn.Conv2d(base_channels + context_channels, context_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            _BPsiResBlock(context_channels),
            _BPsiResBlock(context_channels, dilation=2),
            nn.Conv2d(context_channels, mid_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            _BPsiResBlock(mid_channels),
        )
        self.lowres_head = nn.Sequential(
            nn.Conv2d(mid_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            _BPsiResBlock(base_channels),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
        )
        self.final_blur = _FixedGaussianBlur(final_blur_kernel_size, final_blur_sigma)

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        if xi.ndim != 4 or xi.shape[1] != self.noise_channels or xi.shape[2:] != (self.height, self.width):
            raise ValueError(
                f"Expected xi shape [B, {self.noise_channels}, {self.height}, {self.width}]."
            )

        semantic = self.semantic_stem(xi)
        context = self.context_stem(xi)
        context = self.context_to_semantic(context)
        context = F.interpolate(context, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        lowres = self.semantic_fuse(torch.cat([semantic, context], dim=1))
        lowres = self.lowres_head(lowres)

        coarse_logits = F.interpolate(lowres, size=(self.height, self.width), mode="bilinear", align_corners=False)
        coarse_logits = self.final_blur(coarse_logits)
        coarse = 0.5 + 0.5 * torch.tanh(coarse_logits / math.sqrt(2.0))
        return coarse.clamp(
            DEFAULT_NORMALIZATION_CONFIG.normalized_min,
            DEFAULT_NORMALIZATION_CONFIG.normalized_max,
        )


__all__ = ["BPsi"]
