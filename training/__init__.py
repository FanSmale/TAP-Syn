"""Training utilities for staged residual-velocity generation."""

from .config import (
    DatasetConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    StageConfig,
    TrainingStrategyConfig,
    default_stage_plan,
    make_default_training_strategy_config,
)


def train_training_strategy(config: TrainingStrategyConfig):
    from .engine import train_training_strategy as _train_training_strategy

    return _train_training_strategy(config)


__all__ = [
    "DatasetConfig",
    "LossConfig",
    "ModelConfig",
    "OptimizerConfig",
    "RuntimeConfig",
    "StageConfig",
    "TrainingStrategyConfig",
    "default_stage_plan",
    "make_default_training_strategy_config",
    "train_training_strategy",
]
