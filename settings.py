"""Formal defaults, configurable in Python without an experiment framework.

CLI --config /path/settings.py executes trusted Python defining CONFIG=Config(...).
For example:
    from dataclasses import replace
    from tap_syn.settings import Config
    from tap_syn.training import ModelConfig
    CONFIG = replace(Config(), model=ModelConfig(base_channels=16))

Paths are explicit CLI arguments (or direct function arguments). Data is always
physical m/s on disk and 1000--6000 normalized inside the model. No fixed seed.
"""
from dataclasses import dataclass, field
from .training import ModelConfig, LossConfig, OptimizerConfig, RuntimeConfig, StageConfig, default_stage_plan
from .lora.config import LoraModuleConfig, LoraRuntimeConfig
from .fwi.config import FWITrainConfig
from .fwi.ddnet_train import DDNetTrainConfig
from .forward import ForwardConfig
from .generation import ProxyConfig


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    rf_optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    rf_runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    stages: tuple[StageConfig, ...] = field(default_factory=default_stage_plan)
    lora_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(learning_rate=3e-5))
    lora_runtime: RuntimeConfig = field(default_factory=lambda: RuntimeConfig(batch_size=4))
    lora: LoraModuleConfig = field(default_factory=LoraModuleConfig)
    lora_stages: LoraRuntimeConfig = field(default_factory=LoraRuntimeConfig)
    aba: FWITrainConfig = field(default_factory=FWITrainConfig)
    dd: DDNetTrainConfig = field(default_factory=DDNetTrainConfig)
    forward: ForwardConfig = field(default_factory=ForwardConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    coarse_method: str = "mixed"
    coarse_profile: str = "strong_less_smooth"
