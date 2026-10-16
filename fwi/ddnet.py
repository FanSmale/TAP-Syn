"""DD-Net70 backend for TAP-Syn experiments.

The architecture and optimisation objective follow FanSmale/ddnet (commit
``76a25f5d0408dd716a7cfa2731f42074cc7c2507``): a seismic dimension reducer,
two decoder heads, velocity MSE, and Canny-boundary cross entropy.  The source
repository's DDNet70 module refers to four undefined decoder helper classes;
their implementations below are the explicit fixed-size variants implied by
the neighbouring ``unetUp`` and ``netUp`` helpers in that repository.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, padding=1, *, tanh: bool = False):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.Tanh() if tanh else nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class SeismicRecordDownSampling(nn.Module):
    def __init__(self, shot_num: int):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(shot_num, 8, (3, 1), (2, 1), (1, 0)), ConvBlock(8, 8),
            ConvBlock(8, 16, (3, 1), (2, 1), (1, 0)), ConvBlock(16, 16),
            ConvBlock(16, 32, (3, 1), (2, 1), (1, 0)), ConvBlock(32, 32),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        width = inputs.shape[-1]
        return self.blocks(F.interpolate(inputs, size=(width * 8, width), mode="bilinear", align_corners=False))


class UNetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(ConvBlock(in_channels, out_channels), ConvBlock(out_channels, out_channels))
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.pool(self.conv(inputs))


class DecoderMerge(nn.Module):
    """DD-Net's transposed-convolution decoder block with prescribed output."""
    def __init__(self, in_channels: int, out_channels: int, output_size: tuple[int, int], *, merge: bool):
        super().__init__()
        self.output_size = output_size
        self.merge = merge
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(ConvBlock(in_channels, out_channels), ConvBlock(out_channels, out_channels)) if merge else None

    def forward(self, skip: Tensor | None, inputs: Tensor) -> Tensor:
        value = F.interpolate(self.up(inputs), size=self.output_size, mode="bilinear", align_corners=False)
        if self.merge:
            if skip is None or self.conv is None:
                raise ValueError("DD-Net merge decoder requires a skip tensor.")
            return self.conv(torch.cat([skip, value], dim=1))
        return value


class DDNet70(nn.Module):
    """Original DDNet70 geometry for canonical ``[B, 5, 1000, 70]`` inputs."""

    input_shape = (5, 1000, 70)
    output_shape = (1, 70, 70)

    def __init__(self, in_channels: int = 5):
        super().__init__()
        self.pre_seis_conv = SeismicRecordDownSampling(in_channels)
        self.down3, self.down4 = UNetDown(32, 64), UNetDown(64, 128)
        self.down5, self.center = UNetDown(128, 256), UNetDown(256, 512)
        self.value_up5 = DecoderMerge(512, 256, (9, 9), merge=True)
        self.value_up4 = DecoderMerge(256, 128, (18, 18), merge=True)
        self.value_up3 = DecoderMerge(128, 64, (35, 35), merge=False)
        self.value_up2 = DecoderMerge(64, 32, (70, 70), merge=False)
        self.boundary_up5 = DecoderMerge(512, 256, (9, 9), merge=True)
        self.boundary_up4 = DecoderMerge(256, 128, (18, 18), merge=True)
        self.boundary_up3 = DecoderMerge(128, 64, (35, 35), merge=False)
        self.boundary_up2 = DecoderMerge(64, 32, (70, 70), merge=False)
        self.value_head = ConvBlock(32, 1, tanh=True)
        self.boundary_head = ConvBlock(32, 2, tanh=True)

    def forward(self, seismic: Tensor) -> tuple[Tensor, Tensor]:
        if seismic.ndim != 4 or tuple(seismic.shape[1:]) != self.input_shape:
            raise ValueError(f"DDNet70 requires [B,5,1000,70], got {tuple(seismic.shape)}.")
        down3 = self.down3(self.pre_seis_conv(seismic))
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        center = self.center(down5)
        value = self.value_up2(None, self.value_up3(None, self.value_up4(down4, self.value_up5(down5, center))))
        boundary = self.boundary_up2(None, self.boundary_up3(None, self.boundary_up4(down4, self.boundary_up5(down5, center))))
        return self.value_head(value), self.boundary_head(boundary)


def ddnet_contours(velocity: Tensor) -> Tensor:
    """Original DD-Net per-model Canny contour labels (thresholds 10 and 15)."""
    if velocity.ndim != 4 or velocity.shape[1:] != (1, 70, 70):
        raise ValueError("DD-Net contour targets require [B,1,70,70] velocities.")
    values = velocity.detach().float().cpu().numpy()
    contours = np.empty((len(values), 70, 70), dtype=np.int64)
    for index, model in enumerate(values[:, 0]):
        normalized = cv2.normalize(model, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        contours[index] = np.clip(cv2.Canny((normalized * 255).astype(np.uint8), 10, 15), 0, 1)
    return torch.from_numpy(contours).to(velocity.device)


class DDNetLoss(nn.Module):
    """MSE + weighted binary cross entropy used in the original DD-Net code."""
    def __init__(self, *, velocity_weight: float = 1.0, boundary_weight: float = 0.1, entropy_weight: tuple[float, float] = (1.0, 1.0)):
        super().__init__()
        self.velocity_weight = velocity_weight
        self.boundary_weight = boundary_weight
        self.register_buffer("entropy_weight", torch.tensor(entropy_weight, dtype=torch.float32))

    def forward(self, prediction: Tensor, boundary_logits: Tensor, target: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        mse = F.mse_loss(prediction, target)
        boundary = F.cross_entropy(boundary_logits, ddnet_contours(target), weight=self.entropy_weight)
        total = self.velocity_weight * mse + self.boundary_weight * boundary
        return total, {"mse": mse, "boundary_cross_entropy": boundary}


def curriculum_transform(seismic: Tensor, stage: int) -> Tensor:
    """The DD-Net repository's three source curriculum input constructions."""
    if stage not in (1, 2, 3):
        raise ValueError("DD-Net curriculum stage must be 1, 2, or 3.")
    if stage == 3:
        return seismic
    middle = seismic[:, 2:3]
    if stage == 2:
        return middle.expand_as(seismic).contiguous()
    # Stage 1 follows the original channel grouping.  Its noise/AGC utilities
    # are sample-wise NumPy preprocessing; Gaussian noise is retained here,
    # while amplitude magnification is deliberately left as identity because
    # the canonical physical pairs have no paper-defined AGC calibration.
    result = seismic.clone()
    result[:, :2] = middle + torch.randn_like(middle).expand(-1, 2, -1, -1) * 0.01
    result[:, 2:4] = middle
    result[:, 4:] = middle
    return result


__all__ = ["DDNet70", "DDNetLoss", "curriculum_transform"]
