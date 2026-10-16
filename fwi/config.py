from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class FWIModelConfig:
    name: str = "aba_fwi"


@dataclass(frozen=True)
class FWITrainConfig:
    batch_size: int = 20
    eval_batch_size: int = 5
    num_workers: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 160
    finetune_epochs: int = 100
    eval_every_epochs: int = 1
    checkpoint_every_epochs: int = 5
    use_amp: bool = False
    device: str = "cpu"
    resume_from: str | Path | None = None
    full_eval_metrics: bool = True
