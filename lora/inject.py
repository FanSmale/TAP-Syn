"""LoRA injection helpers and target presets."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from .config import LoraModuleConfig
from .layers import LoRAConv2d, LoRALinear


@dataclass(frozen=True)
class LoraInjectionReport:
    """Summarize one adapter-injection pass."""

    component: str
    target_preset: str
    matched_modules: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.matched_modules)


def resolve_lora_targets(component: str, preset: str) -> tuple[str, ...]:
    """Return the module-name fragments targeted by the chosen preset."""

    normalized = "conservative" if preset == "default" else preset
    if component == "denoiser":
        if normalized == "conservative":
            return ("time_mlp.net", "cond_proj", "qkv", "proj", "coarse_projs")
        if normalized == "extended":
            return (
                "time_mlp.net",
                "cond_proj",
                "qkv",
                "proj",
                "coarse_projs",
                "input_conv",
                "out_conv",
            )
        if normalized == "dense":
            return (
                "time_mlp.net",
                "cond_proj",
                "qkv",
                "proj",
                "coarse_projs",
                "input_conv",
                "out_conv",
                "coarse_encoder",
                "conv1",
                "conv2",
                "shortcut",
                "down",
                "up",
            )
    if component == "b_psi":
        if normalized == "conservative":
            return (
                "semantic_stem",
                "context_stem",
                "context_to_semantic",
                "semantic_fuse",
                "lowres_head",
            )
        if normalized == "extended":
            return (
                "semantic_stem",
                "context_stem",
                "context_to_semantic",
                "semantic_fuse",
                "lowres_head",
                "block",
            )
        if normalized == "dense":
            return (
                "semantic_stem",
                "context_stem",
                "context_to_semantic",
                "semantic_fuse",
                "lowres_head",
                "block",
            )
    raise ValueError(f"Unsupported LoRA preset '{preset}' for component '{component}'.")


def _set_child_module(parent: nn.Module, child_name: str, module: nn.Module) -> None:
    if isinstance(parent, nn.Sequential):
        parent[int(child_name)] = module
        return
    if isinstance(parent, nn.ModuleList):
        parent[int(child_name)] = module
        return
    setattr(parent, child_name, module)


def _wrap_if_supported(module: nn.Module, config: LoraModuleConfig) -> nn.Module:
    if isinstance(module, nn.Linear):
        return LoRALinear(
            module,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            multiplier=config.multiplier,
        )
    if isinstance(module, nn.Conv2d):
        return LoRAConv2d(
            module,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            multiplier=config.multiplier,
        )
    raise TypeError(f"Unsupported module type for LoRA injection: {type(module)!r}")


def inject_lora(model: nn.Module, *, component: str, config: LoraModuleConfig) -> LoraInjectionReport:
    """Inject LoRA wrappers into one model tree according to the target preset."""

    targets = resolve_lora_targets(component, config.target_preset)
    matched: list[str] = []
    named_modules = list(model.named_modules())
    for module_name, module in named_modules:
        if not module_name:
            continue
        if not any(token in module_name for token in targets):
            continue
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        parent_name, _, child_name = module_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        _set_child_module(parent, child_name, _wrap_if_supported(module, config))
        matched.append(module_name)
    return LoraInjectionReport(component=component, target_preset=config.target_preset, matched_modules=tuple(matched))


__all__ = ["LoraInjectionReport", "inject_lora", "resolve_lora_targets"]
