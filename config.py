"""Project-wide configuration for residual velocity modeling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VelocityNormalizationConfig:
    """Configuration for global velocity normalization."""

    vmin: float = 1000.0
    vmax: float = 6000.0
    normalized_min: float = 0.0
    normalized_max: float = 1.0
    low_threshold: float = 0.35
    high_threshold: float = 0.65


DEFAULT_NORMALIZATION_CONFIG = VelocityNormalizationConfig()


__all__ = ["VelocityNormalizationConfig", "DEFAULT_NORMALIZATION_CONFIG"]
