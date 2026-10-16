"""LoRA layer wrappers for linear and convolution layers."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class LoRALinear(nn.Module):
    """Wrap one linear layer with a low-rank trainable update."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / float(self.rank)
        self.multiplier = float(multiplier)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(self.rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        delta = F.linear(self.dropout(x), self.lora_A)
        delta = F.linear(delta, self.lora_B)
        return base + (self.multiplier * self.scale) * delta


class LoRAConv2d(nn.Module):
    """Wrap one Conv2d with a low-rank trainable update."""

    def __init__(
        self,
        base_layer: nn.Conv2d,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")
        if base_layer.groups != 1:
            raise ValueError("LoRAConv2d currently supports Conv2d layers with groups=1 only.")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / float(self.rank)
        self.multiplier = float(multiplier)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

        self.lora_A = nn.Conv2d(
            in_channels=base_layer.in_channels,
            out_channels=self.rank,
            kernel_size=base_layer.kernel_size,
            stride=base_layer.stride,
            padding=base_layer.padding,
            dilation=base_layer.dilation,
            bias=False,
        )
        self.lora_B = nn.Conv2d(
            in_channels=self.rank,
            out_channels=base_layer.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        delta = self.lora_B(self.lora_A(self.dropout(x)))
        return base + (self.multiplier * self.scale) * delta


__all__ = ["LoRAConv2d", "LoRALinear"]
