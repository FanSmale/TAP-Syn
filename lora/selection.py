"""Helpers for selecting LoRA parameters."""

from __future__ import annotations

from collections.abc import Iterator

import torch


def is_lora_parameter(name: str) -> bool:
    return "lora_" in name


def mark_only_lora_trainable(module: torch.nn.Module) -> None:
    """Freeze every parameter except LoRA adapter weights."""

    for name, parameter in module.named_parameters():
        parameter.requires_grad = is_lora_parameter(name)


def iter_lora_parameters(module: torch.nn.Module) -> Iterator[torch.nn.Parameter]:
    """Yield trainable LoRA parameters."""

    for name, parameter in module.named_parameters():
        if is_lora_parameter(name):
            yield parameter


def count_lora_parameters(module: torch.nn.Module) -> int:
    """Count LoRA adapter parameters in one module tree."""

    return sum(parameter.numel() for parameter in iter_lora_parameters(module))


__all__ = ["count_lora_parameters", "iter_lora_parameters", "mark_only_lora_trainable"]
