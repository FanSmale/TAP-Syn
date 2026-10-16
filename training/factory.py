"""Factories for datasets, models, and optimizers used by staged training."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset

from ..data import VelocityNpyDataset
from ..methods import ResidualVelocityRectifiedFlow
from ..nets import ResidualVelocityUNet
from ..unconditional_background import BPsi
from .config import DatasetConfig, OptimizerConfig, StageConfig, TrainingStrategyConfig


def build_dataset(config: DatasetConfig | None) -> Dataset[dict[str, object]] | None:
    """Construct one dataset object from explicit config."""

    if config is None:
        return None
    return VelocityNpyDataset(
        root=config.root, target_size=config.target_size,
        coarse_method=config.coarse_method, coarse_profile=config.coarse_profile,
        max_samples=config.max_samples, sample_start=config.sample_start,
        sample_end=config.sample_end,
    )


def build_dataloader(
    dataset: Dataset[dict[str, object]] | None,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader | None:
    """Build one dataloader when a dataset is present."""

    if dataset is None:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def build_model_stack(config: TrainingStrategyConfig) -> tuple[ResidualVelocityUNet, ResidualVelocityRectifiedFlow, BPsi]:
    """Construct the RF denoiser, flow wrapper, and coarse generator."""

    if config.model.method != "rectified_flow":
        raise ValueError(f"Unsupported training method: {config.model.method}.")
    denoiser = ResidualVelocityUNet(
        base_channels=config.model.base_channels,
        cond_dim=config.model.cond_dim,
        attn_resolutions=config.model.attn_resolutions,
        num_heads=config.model.num_heads,
    )
    flow = ResidualVelocityRectifiedFlow(denoiser)
    b_psi = BPsi(
        base_channels=config.model.b_psi_base_channels,
        output_size=config.model.b_psi_output_size,
        final_blur_kernel_size=config.model.b_psi_final_blur_kernel_size,
        final_blur_sigma=config.model.b_psi_final_blur_sigma,
    )
    return denoiser, flow, b_psi


def _iter_trainable(named_params: Iterable[tuple[str, torch.nn.Parameter]]) -> list[torch.nn.Parameter]:
    return [param for _, param in named_params if param.requires_grad]


def build_optimizer(
    optimizer_config: OptimizerConfig,
    stage_config: StageConfig,
    *,
    denoiser: ResidualVelocityUNet,
    b_psi: BPsi,
) -> torch.optim.Optimizer:
    """Create a fresh optimizer for the current stage."""

    parameter_groups: list[dict[str, object]] = []
    if stage_config.train_backbone:
        backbone_params = _iter_trainable(denoiser.named_parameters())
        if backbone_params:
            parameter_groups.append(
                {
                    "params": backbone_params,
                    "lr": optimizer_config.learning_rate * stage_config.backbone_lr_scale,
                }
            )
    if stage_config.train_b_psi:
        b_psi_params = _iter_trainable(b_psi.named_parameters())
        if b_psi_params:
            parameter_groups.append(
                {
                    "params": b_psi_params,
                    "lr": optimizer_config.learning_rate * stage_config.b_psi_lr_scale,
                }
            )

    if not parameter_groups:
        raise ValueError(f"Stage {stage_config.name} has no trainable parameter groups.")

    return torch.optim.Adam(
        parameter_groups,
        betas=optimizer_config.betas,
        weight_decay=optimizer_config.weight_decay,
    )


def resolve_device(device_name: str) -> torch.device:
    """Resolve a runtime device string to an available torch device."""

    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def ensure_checkpoint_dir(path: str | Path) -> Path:
    """Return the checkpoint path as a Path object."""

    return Path(path)
