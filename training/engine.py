"""Top-level staged training engine."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..losses import LossScheme1
from .config import StageConfig, TrainingStrategyConfig
from .factory import (
    build_dataloader,
    build_dataset,
    build_model_stack,
    build_optimizer,
    ensure_checkpoint_dir,
    resolve_device,
)
from .stages import (
    configure_stage_trainability,
    count_trainable_parameters,
    evaluate_conditional_epoch,
    make_grad_scaler,
    run_stage1_epoch,
    run_stage2_epoch,
    run_stage3_epoch,
)

_CHECKPOINT_VERSION = 3


class _ResumeState:
    def __init__(self, checkpoint: dict[str, Any]) -> None:
        self.checkpoint = checkpoint
        self.stage_index = int(checkpoint["stage_index"])
        self.stage_name = str(checkpoint["stage_name"])
        self.stage_epoch = int(checkpoint["stage_epoch"])
        self.global_epoch = int(checkpoint["global_epoch"])
        self.history = []


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _coerce_rng_tensor(state: Any) -> torch.ByteTensor:
    if isinstance(state, torch.Tensor):
        return state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return torch.as_tensor(state, dtype=torch.uint8, device="cpu").contiguous()


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_coerce_rng_tensor(state["torch"]))
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([_coerce_rng_tensor(cuda_state) for cuda_state in state["cuda"]])


def _format_duration(seconds: float) -> str:
    total_seconds = max(0.0, float(seconds))
    minutes, seconds_rem = divmod(total_seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {seconds_rem:05.2f}s"
    if minutes > 0:
        return f"{minutes:d}m {seconds_rem:05.2f}s"
    return f"{seconds_rem:.2f}s"


def _save_checkpoint(
    checkpoint_dir: Path,
    *,
    denoiser: torch.nn.Module,
    b_psi: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    stage_index: int,
    stage_name: str,
    epoch: int,
    global_epoch: int,
    config: TrainingStrategyConfig,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{stage_name}_epoch{global_epoch:04d}.pt"
    torch.save(
        {
            "checkpoint_version": _CHECKPOINT_VERSION,
            "stage_index": stage_index,
            "stage_name": stage_name,
            "stage_epoch": epoch,
            "global_epoch": global_epoch,
            "denoiser_state_dict": denoiser.state_dict(),
            "b_psi_state_dict": b_psi.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "rng_state": _capture_rng_state(),
            "model_config": config.model,
            "optimizer_config": config.optimizer,
            "loss_config": config.loss,
            "stages": config.stages,
        },
        checkpoint_path,
    )
    return checkpoint_path


def _load_resume_state(path: str | Path | None, device: torch.device) -> _ResumeState | None:
    if path is None:
        return None
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    return _ResumeState(checkpoint)


def _validate_resume_state(
    resume_state: _ResumeState,
    stages: tuple[StageConfig, ...],
) -> None:
    if resume_state.checkpoint.get("checkpoint_version") != _CHECKPOINT_VERSION:
        raise ValueError("Staged resume checkpoint version is not supported.")
    if resume_state.stage_index < 0 or resume_state.stage_index >= len(stages):
        raise ValueError(f"Checkpoint stage index {resume_state.stage_index} is out of range for current stage plan.")
    stage = stages[resume_state.stage_index]
    if stage.name != resume_state.stage_name:
        raise ValueError(
            f"Checkpoint stage mismatch: expected stage index {resume_state.stage_index} to be "
            f"{stage.name}, got {resume_state.stage_name}."
        )
    if resume_state.stage_epoch < 1 or resume_state.stage_epoch > stage.epochs:
        raise ValueError(
            f"Checkpoint stage epoch {resume_state.stage_epoch} is invalid for stage {stage.name} "
            f"with {stage.epochs} epochs."
        )


def train_training_strategy(config: TrainingStrategyConfig) -> dict[str, Any]:
    """Run the staged RF training strategy."""

    device = resolve_device(config.runtime.device)
    checkpoint_dir = ensure_checkpoint_dir(config.runtime.checkpoint_dir)

    if config.runtime.resume_from is None and checkpoint_dir.exists() and any(checkpoint_dir.glob("*.pt")):
        raise FileExistsError(f"Existing checkpoints require resume_from: {checkpoint_dir}")
    if not config.stages or any(stage.epochs < 1 for stage in config.stages):
        raise ValueError("RF requires a nonempty positive-epoch stage plan")
    train_dataset = build_dataset(config.train_dataset)
    eval_dataset = build_dataset(config.eval_dataset)
    train_loader = build_dataloader(
        train_dataset,
        batch_size=config.runtime.batch_size,
        num_workers=config.runtime.num_workers,
        shuffle=True,
        drop_last=False,
    )
    eval_loader = build_dataloader(
        eval_dataset,
        batch_size=config.runtime.batch_size,
        num_workers=config.runtime.num_workers,
        shuffle=False,
        drop_last=False,
    )

    denoiser, flow, b_psi = build_model_stack(config)
    denoiser.to(device)
    flow.to(device)
    b_psi.to(device)
    loss_scheme = LossScheme1(config.loss).to(device)
    scaler = make_grad_scaler(config.runtime, device)

    resume_state = _load_resume_state(config.runtime.resume_from, device)
    history: list[dict[str, Any]] = []
    global_epoch = 0
    last_checkpoint = None if config.runtime.resume_from is None else Path(config.runtime.resume_from)
    if resume_state is not None:
        _validate_resume_state(resume_state, config.stages)
        checkpoint = resume_state.checkpoint
        if any(checkpoint.get(key) != value for key, value in (("model_config", config.model), ("optimizer_config", config.optimizer), ("loss_config", config.loss), ("stages", config.stages))):
            raise ValueError("Resume model, optimizer, loss or stage configuration differs")
        denoiser.load_state_dict(checkpoint["denoiser_state_dict"])
        b_psi.load_state_dict(checkpoint["b_psi_state_dict"])
        scaler_state_dict = checkpoint.get("scaler_state_dict")
        if scaler_state_dict is not None:
            scaler.load_state_dict(scaler_state_dict)
        history = list(resume_state.history)
        global_epoch = resume_state.global_epoch
        _restore_rng_state(checkpoint["rng_state"])


    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    if eval_dataset is None:
        print("Eval dataset: none")
    else:
        print(f"Eval samples: {len(eval_dataset)}")
    if resume_state is not None:
        print(
            f"Resuming from stage {resume_state.stage_name} epoch {resume_state.stage_epoch} "
            f"(global {resume_state.global_epoch})"
        )

    for stage_index, stage in enumerate(config.stages):
        if resume_state is not None and stage_index < resume_state.stage_index:
            continue
        if resume_state is not None and stage_index == resume_state.stage_index and resume_state.stage_epoch >= stage.epochs:
            continue

        configure_stage_trainability(denoiser, b_psi, stage)
        optimizer = build_optimizer(config.optimizer, stage, denoiser=denoiser, b_psi=b_psi)

        start_stage_epoch = 1
        if resume_state is not None and stage_index == resume_state.stage_index:
            optimizer.load_state_dict(resume_state.checkpoint["optimizer_state_dict"])
            start_stage_epoch = resume_state.stage_epoch + 1

        print()
        print(f"=== {stage.name} ===")
        print(
            f"Trainable params -> backbone: {count_trainable_parameters(denoiser)}, "
            f"B_psi: {count_trainable_parameters(b_psi)}"
        )

        for stage_epoch in range(start_stage_epoch, stage.epochs + 1):
            global_epoch += 1
            print(f"  Epoch {stage_epoch}/{stage.epochs} (global {global_epoch})")

            train_start_time = time.perf_counter()
            if stage.batch_mode == "conditional":
                train_metrics = run_stage1_epoch(loss_scheme, flow, train_loader, optimizer, scaler, device, config.runtime)
            elif stage.batch_mode == "unconditional":
                train_metrics = run_stage2_epoch(loss_scheme, flow, b_psi, train_loader, optimizer, scaler, device, config.runtime, stage)
            elif stage.batch_mode == "mixed":
                train_metrics = run_stage3_epoch(loss_scheme, flow, b_psi, train_loader, optimizer, scaler, device, config.runtime, stage)
            else:
                raise ValueError(f"Unsupported batch mode: {stage.batch_mode}.")
            train_duration_sec = time.perf_counter() - train_start_time

            eval_metrics: dict[str, float] = {}
            eval_duration_sec = 0.0
            if eval_loader is not None and config.runtime.eval_every_epochs > 0 and stage_epoch % config.runtime.eval_every_epochs == 0:
                eval_start_time = time.perf_counter()
                eval_metrics = evaluate_conditional_epoch(loss_scheme, flow, eval_loader, device, config.runtime)
                eval_duration_sec = time.perf_counter() - eval_start_time

            epoch_duration_sec = train_duration_sec + eval_duration_sec
            combined_metrics = {
                **train_metrics,
                **eval_metrics,
                "train_epoch_time_sec": train_duration_sec,
                "eval_epoch_time_sec": eval_duration_sec,
                "epoch_time_sec": epoch_duration_sec,
            }
            history.append(
                {
                    "stage": stage.name,
                    "stage_epoch": stage_epoch,
                    "global_epoch": global_epoch,
                    "metrics": combined_metrics,
                }
            )
            print(f"    metrics: {combined_metrics}")
            print(
                "    time: "
                f"train={_format_duration(train_duration_sec)}, "
                f"eval={_format_duration(eval_duration_sec)}, "
                f"total={_format_duration(epoch_duration_sec)}"
            )

            should_checkpoint = (
                config.runtime.checkpoint_every_epochs > 0
                and stage_epoch % config.runtime.checkpoint_every_epochs == 0
            ) or stage_epoch == stage.epochs or global_epoch == config.runtime.stop_after_global_epoch
            if should_checkpoint:
                checkpoint_path = _save_checkpoint(
                    checkpoint_dir,
                    denoiser=denoiser,
                    b_psi=b_psi,
                    optimizer=optimizer,
                    scaler=scaler,
                    stage_index=stage_index,
                    stage_name=stage.name,
                    epoch=stage_epoch,
                    global_epoch=global_epoch,
                    config=config,
                )
                last_checkpoint = checkpoint_path
                print(f"    checkpoint: {checkpoint_path}")

            if global_epoch == config.runtime.stop_after_global_epoch:
                print(f"Stopping after requested global epoch {global_epoch}.")
                return {
                    "device": str(device),
                    "train_dataset_size": len(train_dataset),
                    "eval_dataset_size": 0 if eval_dataset is None else len(eval_dataset),
                    "last_checkpoint": None if last_checkpoint is None else str(last_checkpoint),
                }

        if resume_state is not None and stage_index == resume_state.stage_index:
            resume_state = None

    return {
        "device": str(device),
        "train_dataset_size": len(train_dataset),
        "eval_dataset_size": 0 if eval_dataset is None else len(eval_dataset),
        "history": history,
        "last_checkpoint": None if last_checkpoint is None else str(last_checkpoint),
    }
