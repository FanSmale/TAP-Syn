"""Configuration objects for LoRA fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..training.config import DatasetConfig, LossConfig, ModelConfig, OptimizerConfig, RuntimeConfig

LoraTargetPreset = Literal["default", "conservative", "extended", "dense"]


@dataclass(frozen=True)
class LoraModuleConfig:
    """Describe how LoRA adapters should be injected into the model stack."""

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target_preset: LoraTargetPreset = "dense"
    enable_denoiser_lora: bool = True
    enable_b_psi_lora: bool = True
    b_psi_coarse_weight: float = 0.65
    multiplier: float = 1.0


@dataclass(frozen=True)
class LoraRuntimeConfig:
    """Runtime settings specific to LoRA fine-tuning."""

    epochs: int = 55
    b_psi_epochs: int = 30
    unconditional_bridge_epochs: int = 10
    joint_refinement_epochs: int = 0
    joint_conditional_ratio: float = 0.5
    joint_coarse_reference_weight: float = 1.5
    joint_unconditional_path_weight: float = 0.75
    b_psi_coarse_reference_weight: float = 4.0
    b_psi_unconditional_path_weight: float = 0.15
    save_every_epochs: int = 5
    eval_every_epochs: int = 1
    save_full_checkpoint: bool = True

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("LoRA epochs must be positive.")
        if self.b_psi_epochs < 0 or self.unconditional_bridge_epochs < 0 or self.joint_refinement_epochs < 0:
            raise ValueError("LoRA stage epoch counts must be non-negative.")
        if self.b_psi_epochs + self.unconditional_bridge_epochs + self.joint_refinement_epochs >= self.epochs:
            raise ValueError("Non-conditional LoRA stages must leave at least one conditional denoiser epoch.")
        if not 0.0 < self.joint_conditional_ratio < 1.0:
            raise ValueError("joint_conditional_ratio must be within (0, 1).")
        if self.joint_coarse_reference_weight < 0.0 or self.joint_unconditional_path_weight < 0.0:
            raise ValueError("Joint refinement loss weights must be non-negative.")
        if self.b_psi_coarse_reference_weight < 0.0:
            raise ValueError("b_psi_coarse_reference_weight must be non-negative.")
        if self.b_psi_unconditional_path_weight < 0.0:
            raise ValueError("b_psi_unconditional_path_weight must be non-negative.")


@dataclass(frozen=True)
class LoraFinetuneConfig:
    """Top-level configuration for adapter-only target-domain fine-tuning."""

    train_dataset: DatasetConfig
    eval_dataset: DatasetConfig | None
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(learning_rate=3e-5))
    loss: LossConfig = field(default_factory=LossConfig)
    runtime: RuntimeConfig = field(default_factory=lambda: RuntimeConfig(batch_size=4))
    lora: LoraModuleConfig = field(default_factory=LoraModuleConfig)
    lora_runtime: LoraRuntimeConfig = field(default_factory=LoraRuntimeConfig)
    base_checkpoint: str | Path | None = None
    resume_from: str | Path | None = None
