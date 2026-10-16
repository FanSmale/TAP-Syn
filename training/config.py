"""Configuration objects for staged training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

from ..data import DEFAULT_DATA_ROOT

BatchMode = Literal["conditional", "unconditional", "mixed"]
BackboneMode = Literal["frozen", "partial", "full"]
MethodName = Literal["rectified_flow"]
LossName = Literal["scheme1"]
CoarseAlignmentMode = Literal["paired", "distribution"]
@dataclass(frozen=True)
class DatasetConfig:
    """Physical m/s npy chunks; root may contain a velocity/ subdirectory."""
    root: str | Path = DEFAULT_DATA_ROOT
    target_size: tuple[int, int] = (128, 128)
    coarse_method: str = "mixed"
    coarse_profile: str = "strong_less_smooth"
    max_samples: int | None = None
    sample_start: int | None = None
    sample_end: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    """Model hyperparameters for the first-pass RF trainer."""

    method: MethodName = "rectified_flow"
    base_channels: int = 32
    cond_dim: int = 256
    attn_resolutions: tuple[int, ...] = (16,)
    num_heads: int = 4
    b_psi_base_channels: int = 64
    b_psi_output_size: int = 32
    b_psi_final_blur_kernel_size: int = 3
    b_psi_final_blur_sigma: float = 0.7


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer hyperparameters used across stages."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime, dataloader, checkpoint, and logging settings."""

    batch_size: int = 8
    num_workers: int = 0
    use_amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str | Path = Path("checkpoints") / "train_rectified_flow"
    checkpoint_every_epochs: int = 10
    eval_every_epochs: int = 1
    eval_sample_steps: int = 8
    resume_from: str | Path | None = None
    grad_clip_norm: float | None = 1.0
    skip_nonfinite_steps: bool = True
    log_nonfinite_tensors: bool = True
    stop_after_global_epoch: int | None = None


@dataclass(frozen=True)
class LossConfig:
    """Hyperparameters for the active training loss scheme."""

    name: LossName = "scheme1"
    lambda_flow: float = 1.0
    lambda_low: float = 0.5
    lambda_grad: float = 0.1
    lambda_psd: float = 0.05
    lambda_div: float = 0.3
    lambda_coarse: float = 1.0
    lambda_coarse_mean: float = 0.1
    lambda_coarse_std: float = 0.1
    lambda_coarse_range: float = 0.1
    lambda_coarse_profile: float = 0.35
    lambda_coarse_pyramid: float = 0.5
    lambda_coarse_pairwise: float = 0.5
    coarse_alignment_mode: CoarseAlignmentMode = "distribution"
    coarse_distribution_pool_size: int = 16
    use_grad_dist: bool = True
    use_psd: bool = True
    use_diversity: bool = True
    diversity_pool_size: int = 16
    low_pass_kernel_size: int = 9
    low_pass_sigma: float = 2.0
    grad_pool_kernel: int = 4
    psd_pool_size: int = 16
    charbonnier_eps: float = 1e-3
    psd_eps: float = 1e-6
    coarse_range_q_low: float = 0.05
    coarse_range_q_high: float = 0.95
    coarse_pyramid_sizes: tuple[int, ...] = (32, 16, 8)


@dataclass(frozen=True)
class StageConfig:
    """Describe one training stage."""

    name: str
    epochs: int
    batch_mode: BatchMode
    conditional_ratio: float = 0.5
    train_backbone: bool = True
    train_b_psi: bool = False
    backbone_mode: BackboneMode = "full"
    backbone_lr_scale: float = 1.0
    b_psi_lr_scale: float = 1.0
    coarse_reference_weight: float = 1.0
    unconditional_path_weight: float = 1.0


@dataclass(frozen=True)
class TrainingStrategyConfig:
    """Top-level configuration for the staged trainer."""

    train_dataset: DatasetConfig
    eval_dataset: DatasetConfig | None
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    stages: tuple[StageConfig, ...] = field(default_factory=tuple)


def default_stage_plan() -> tuple[StageConfig, ...]:
    """Return the default 30 / 10 / 10 / 50 stage split."""

    return (
        StageConfig(
            name="stage1",
            epochs=30,
            batch_mode="conditional",
            train_backbone=True,
            train_b_psi=False,
            backbone_mode="full",
        ),
        StageConfig(
            name="stage2a",
            epochs=10,
            batch_mode="unconditional",
            train_backbone=False,
            train_b_psi=True,
            backbone_mode="frozen",
            coarse_reference_weight=4.0,
            unconditional_path_weight=0.15,
        ),
        StageConfig(
            name="stage2b",
            epochs=10,
            batch_mode="unconditional",
            train_backbone=True,
            train_b_psi=True,
            backbone_mode="partial",
            backbone_lr_scale=0.25,
            b_psi_lr_scale=1.0,
            coarse_reference_weight=3.0,
            unconditional_path_weight=0.35,
        ),
        StageConfig(
            name="stage3",
            epochs=50,
            batch_mode="mixed",
            conditional_ratio=0.5,
            train_backbone=True,
            train_b_psi=True,
            backbone_mode="full",
            backbone_lr_scale=1.0,
            b_psi_lr_scale=1.0,
            coarse_reference_weight=1.5,
            unconditional_path_weight=0.75,
        ),
    )


def make_default_training_strategy_config(
    *,
    train_dataset: DatasetConfig,
    eval_dataset: DatasetConfig | None,
    model: ModelConfig | None = None,
    optimizer: OptimizerConfig | None = None,
    runtime: RuntimeConfig | None = None,
    loss: LossConfig | None = None,
    stages: tuple[StageConfig, ...] | None = None,
) -> TrainingStrategyConfig:
    """Build a training config with repo-default stage layout."""

    return TrainingStrategyConfig(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model=model if model is not None else ModelConfig(),
        optimizer=optimizer if optimizer is not None else OptimizerConfig(),
        runtime=runtime if runtime is not None else RuntimeConfig(),
        loss=loss if loss is not None else LossConfig(),
        stages=stages if stages is not None else default_stage_plan(),
    )
