"""Checkpoint helpers for LoRA fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def load_checkpoint_payload(path: str | Path | None, device: torch.device) -> dict[str, Any] | None:
    """Load trusted local checkpoints only (torch pickle format)."""
    return None if path is None else torch.load(Path(path), map_location=device, weights_only=False)


def _extract_state_dict(payload: dict[str, Any], preferred_key: str) -> dict[str, torch.Tensor]:
    if preferred_key in payload:
        state_dict = payload[preferred_key]
        if not isinstance(state_dict, dict):
            raise ValueError(f"Checkpoint key {preferred_key} is not a state dict.")
        # Full-state LoRA checkpoints serialize wrapped base layers under
        # ``*.base_layer.*``. Strip adapters and restore the pre-injection key
        # layout when such a checkpoint is reused as a source base.
        return {
            name.replace(".base_layer.", "."): tensor
            for name, tensor in state_dict.items()
            if "lora_" not in name
        }
    if "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint key model_state_dict is not a state dict.")
        return state_dict
    if all(isinstance(key, str) for key in payload.keys()):
        return payload  # type: ignore[return-value]
    raise ValueError(f"Checkpoint does not contain {preferred_key}.")


def load_base_component_state(
    module: nn.Module,
    payload: dict[str, Any] | None,
    preferred_key: str,
    *,
    allow_missing_prefixes: tuple[str, ...] = (),
) -> None:
    """Load one base component state dict if a checkpoint payload is available."""

    if payload is None:
        return
    missing, unexpected = module.load_state_dict(_extract_state_dict(payload, preferred_key), strict=False)
    disallowed_missing = [name for name in missing if not name.startswith(allow_missing_prefixes)]
    if disallowed_missing or unexpected:
        raise ValueError(
            f"Failed to load base component {preferred_key}: "
            f"missing={disallowed_missing}, unexpected={unexpected}."
        )


def extract_lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return a CPU LoRA-only state dict from one module tree."""

    return {
        name: tensor.detach().cpu()
        for name, tensor in module.state_dict().items()
        if "lora_" in name
    }


def load_lora_state_dict(module: nn.Module, state_dict: dict[str, torch.Tensor], *, strict: bool = True) -> None:
    """Load one LoRA-only state dict into an already injected model."""

    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if strict:
        unexpected_nonbase = [name for name in unexpected if "lora_" in name]
        missing_lora = [name for name in missing if "lora_" in name]
        if missing_lora or unexpected_nonbase:
            raise ValueError(
                "Failed to load LoRA state dict: "
                f"missing={missing_lora}, unexpected={unexpected_nonbase}."
            )


def save_lora_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    """Save one LoRA checkpoint payload and return its final path."""

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    temporary = path_obj.with_name(f".{path_obj.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path_obj)
    return path_obj


__all__ = [
    "extract_lora_state_dict",
    "load_base_component_state",
    "load_checkpoint_payload",
    "load_lora_state_dict",
    "save_lora_checkpoint",
]
