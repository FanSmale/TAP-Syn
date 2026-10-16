"""Adapter-only LoRA fine-tuning runner."""

from __future__ import annotations

import random
import hashlib
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..losses import LossScheme1
from ..methods import ResidualVelocityRectifiedFlow
from ..training.config import TrainingStrategyConfig
from ..training.factory import build_dataset, build_model_stack, resolve_device
from .checkpoint import (
    extract_lora_state_dict,
    load_base_component_state,
    load_checkpoint_payload,
    load_lora_state_dict,
    save_lora_checkpoint,
)
from .config import LoraFinetuneConfig
from .inject import inject_lora
from .selection import count_lora_parameters, iter_lora_parameters, mark_only_lora_trainable


def _base_digest(path) -> str:
    """Hash the model checkpoint, not training data or paths."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _restore_rng_state(state: dict[str, Any]) -> None:
    required = {"torch", "cuda", "numpy", "python"}
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError("Resume checkpoint has an incomplete RNG state.")
    torch.random.set_rng_state(state["torch"].to(device="cpu", dtype=torch.uint8).contiguous())
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("Resume checkpoint contains CUDA RNG state, but CUDA is unavailable.")
        torch.cuda.set_rng_state_all([
            cuda_state.to(device="cpu", dtype=torch.uint8).contiguous()
            for cuda_state in state["cuda"]
        ])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def _validate_resume_config(config: LoraFinetuneConfig, payload: dict[str, Any]) -> int:
    if payload["model_config"] != config.model or payload["lora_config"] != config.lora:
        raise ValueError("Resume model/LoRA architecture differs from configuration.")
    if payload["lora_runtime_config"] != config.lora_runtime:
        raise ValueError("Resume LoRA stage plan differs from configuration.")
    if payload["base_checkpoint_sha256"] != _base_digest(config.base_checkpoint):
        raise ValueError("Resume source-pretrained model differs")
    if payload["optimizer_config"] != config.optimizer or payload["loss_config"] != config.loss:
        raise ValueError("Resume optimizer/loss configuration differs")
    epoch = int(payload["epoch"])
    if not 0 <= epoch <= config.lora_runtime.epochs:
        raise ValueError("Invalid LoRA resume epoch")
    return epoch


def _set_active_lora_stage(
    denoiser: torch.nn.Module,
    b_psi: torch.nn.Module,
    *,
    stage: str,
) -> None:
    """Keep B_psi distribution adaptation separate from conditional RF adaptation."""
    if stage not in {"joint", "b_psi", "denoiser", "unconditional_bridge"}:
        raise ValueError(f"Unsupported LoRA stage: {stage}.")
    # mark_only_lora_trainable has already frozen base weights. Toggle only LoRA
    # weights here so the stage boundary cannot update the other adapter tree.
    for name, parameter in denoiser.named_parameters():
        parameter.requires_grad = "lora_" in name and stage in {"joint", "denoiser", "unconditional_bridge"}
    for name, parameter in b_psi.named_parameters():
        parameter.requires_grad = "lora_" in name and stage in {"joint", "b_psi"}


def _joint_step_is_conditional(batch_index: int, conditional_steps: float, conditional_ratio: float) -> bool:
    """Match source Stage 3's deterministic conditional/unconditional mix."""
    return conditional_steps < batch_index * conditional_ratio


def _autocast_context(device: torch.device, use_amp: bool):
    enabled = use_amp and device.type == "cuda"
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def _prepare_conditional_batch(batch: dict[str, Any], device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    velocity = batch["velocity"].to(device)
    velocity_norm = batch["velocity_norm"].to(device)
    m_c_cond = batch["m_c_cond"].to(device)
    r0 = batch["r0"].to(device)
    return velocity, velocity_norm, m_c_cond, r0


def _collate_lora_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    first = samples[0]
    batch: dict[str, Any] = {}
    for key in first:
        values = [sample[key] for sample in samples]
        first_value = values[0]
        if isinstance(first_value, torch.Tensor):
            batch[key] = torch.stack(values)
        else:
            batch[key] = values
    return batch


def _rf_training_details(
    flow: ResidualVelocityRectifiedFlow,
    *,
    r0: Tensor,
    m_c_cond: Tensor,
) -> dict[str, Tensor]:
    batch = r0.shape[0]
    t = flow.path.sample_time(batch, r0.device, r0.dtype)
    noise = flow.path.sample_noise(r0)
    state_t = flow.path.interpolate(noise, r0, t)
    target = flow.path.target_velocity(noise, r0)
    pred = flow.denoiser(state_t, t, m_c_cond)
    return {
        "t": t,
        "noise": noise,
        "state_t": state_t,
        "pred": pred,
        "target": target,
    }


def _conditional_loss(
    loss_scheme: LossScheme1,
    flow: ResidualVelocityRectifiedFlow,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    _, velocity_norm, m_c_cond, r0 = _prepare_conditional_batch(batch, device)
    flow_details = _rf_training_details(flow, r0=r0, m_c_cond=m_c_cond)
    loss, metrics, _ = loss_scheme.compute_total_loss(
        velocity_norm=velocity_norm,
        m_c_cond=m_c_cond,
        flow_details=flow_details,
        include_coarse=False,
    )
    return loss, {f"conditional_{key}": value for key, value in metrics.items()}


def _b_psi_unconditional_loss(
    loss_scheme: LossScheme1,
    flow: ResidualVelocityRectifiedFlow,
    b_psi: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    *,
    coarse_weight: float,
    unconditional_path_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    """Adapt B_psi through the same unconditional RF path used in source training.

    The denoiser weights remain frozen in this LoRA stage, but gradients still
    propagate through its coarse condition input into B_psi.  This keeps target
    background adaptation coupled to the residual path instead of matching only
    marginal coarse statistics.
    """
    _, velocity_norm, reference_m_c_cond, _ = _prepare_conditional_batch(batch, device)
    xi = torch.randn_like(reference_m_c_cond)
    generated_m_c_cond = b_psi(xi)
    generated_r0 = velocity_norm - generated_m_c_cond
    flow_details = _rf_training_details(flow, r0=generated_r0, m_c_cond=generated_m_c_cond)
    loss, metrics, _ = loss_scheme.compute_total_loss(
        velocity_norm=velocity_norm,
        m_c_cond=generated_m_c_cond,
        flow_details=flow_details,
        include_coarse=True,
        coarse_reference_cond=reference_m_c_cond,
        unconditional_path_weight=unconditional_path_weight,
        coarse_weight=coarse_weight,
    )
    generated_stats = generated_m_c_cond.detach().float().flatten(start_dim=1)
    reference_stats = reference_m_c_cond.detach().float().flatten(start_dim=1)
    metrics["generated_coarse_cond_mean"] = float(generated_stats.mean().item())
    metrics["generated_coarse_cond_std"] = float(generated_stats.std(unbiased=False).item())
    metrics["reference_coarse_cond_mean"] = float(reference_stats.mean().item())
    metrics["reference_coarse_cond_std"] = float(reference_stats.std(unbiased=False).item())
    return loss, {f"b_psi_{name}": value for name, value in metrics.items()}


def _unconditional_bridge_loss(
    loss_scheme: LossScheme1,
    flow: ResidualVelocityRectifiedFlow,
    b_psi: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    *,
    unconditional_path_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    """Adapt denoiser LoRA to the fixed target-adapted B_psi condition distribution."""
    _, velocity_norm, reference_m_c_cond, _ = _prepare_conditional_batch(batch, device)
    with torch.no_grad():
        generated_m_c_cond = b_psi(torch.randn_like(reference_m_c_cond))
    generated_r0 = velocity_norm - generated_m_c_cond
    flow_details = _rf_training_details(flow, r0=generated_r0, m_c_cond=generated_m_c_cond)
    loss, metrics, _ = loss_scheme.compute_total_loss(
        velocity_norm=velocity_norm,
        m_c_cond=generated_m_c_cond,
        flow_details=flow_details,
        include_coarse=False,
        unconditional_path_weight=unconditional_path_weight,
    )
    prefixed_metrics = {f"unconditional_bridge_{name}": value for name, value in metrics.items()}
    return loss, prefixed_metrics


def _maybe_save_checkpoint(
    config: LoraFinetuneConfig,
    *,
    epoch: int,
    denoiser: torch.nn.Module,
    b_psi: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> Path:
    checkpoint_dir = Path(config.runtime.checkpoint_dir)
    payload: dict[str, Any] = {
        "checkpoint_version": 2,
        "base_checkpoint_sha256": _base_digest(config.base_checkpoint),
        "epoch": epoch,
        "lora_config": config.lora,
        "lora_runtime_config": config.lora_runtime,
        "model_config": config.model,
        "optimizer_config": config.optimizer,
        "loss_config": config.loss,
        "denoiser_lora_state_dict": extract_lora_state_dict(denoiser),
        "b_psi_lora_state_dict": extract_lora_state_dict(b_psi),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": {
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    if config.lora_runtime.save_full_checkpoint:
        payload["denoiser_state_dict"] = denoiser.state_dict()
        payload["b_psi_state_dict"] = b_psi.state_dict()
    return save_lora_checkpoint(checkpoint_dir / f"adapter_epoch{epoch:04d}.pt", payload)


def train_lora_finetune(config: LoraFinetuneConfig) -> dict[str, Any]:
    """Run adapter-only few-shot LoRA fine-tuning."""

    if config.base_checkpoint is None:
        raise ValueError("LoRA adaptation requires a pretrained base checkpoint.")
    checkpoint_root = Path(config.runtime.checkpoint_dir)
    if config.resume_from is None and checkpoint_root.exists() and any(checkpoint_root.glob("*.pt")):
        raise FileExistsError(f"Existing adapters require resume_from: {checkpoint_root}")
    device = resolve_device(config.runtime.device)
    train_dataset = build_dataset(config.train_dataset)
    if train_dataset is None:
        raise ValueError("train_dataset config is required for LoRA fine-tuning.")
    eval_dataset = build_dataset(config.eval_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.runtime.batch_size,
        num_workers=config.runtime.num_workers,
        shuffle=True,
        drop_last=False,
        collate_fn=_collate_lora_batch,
    )
    eval_loader = None if eval_dataset is None else DataLoader(
        eval_dataset,
        batch_size=config.runtime.batch_size,
        num_workers=config.runtime.num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=_collate_lora_batch,
    )

    denoiser, flow, b_psi = build_model_stack(
        TrainingStrategyConfig(
            train_dataset=config.train_dataset,
            eval_dataset=config.eval_dataset,
            model=config.model,
            loss=config.loss,
            runtime=config.runtime,
        )
    )
    base_payload = load_checkpoint_payload(config.base_checkpoint, device)
    load_base_component_state(denoiser, base_payload, "denoiser_state_dict")
    load_base_component_state(b_psi, base_payload, "b_psi_state_dict")

    denoiser_report = None
    b_psi_report = None
    if config.lora.enable_denoiser_lora:
        denoiser_report = inject_lora(denoiser, component="denoiser", config=config.lora)
    if config.lora.enable_b_psi_lora:
        b_psi_report = inject_lora(b_psi, component="b_psi", config=config.lora)

    if config.lora_runtime.b_psi_epochs or config.lora_runtime.unconditional_bridge_epochs or config.lora_runtime.joint_refinement_epochs:
        if not config.lora.enable_b_psi_lora or not config.lora.enable_denoiser_lora:
            raise ValueError("Sequential or joint LoRA refinement requires both B_psi and denoiser LoRA.")

    mark_only_lora_trainable(denoiser)
    mark_only_lora_trainable(b_psi)

    resume_payload = load_checkpoint_payload(config.resume_from, device)
    resume_epoch = 0
    if resume_payload is not None:
        resume_epoch = _validate_resume_config(config, resume_payload)
        denoiser_lora_state = resume_payload.get("denoiser_lora_state_dict")
        load_lora_state_dict(denoiser, denoiser_lora_state)
        b_psi_lora_state = resume_payload.get("b_psi_lora_state_dict")
        load_lora_state_dict(b_psi, b_psi_lora_state)

    flow.to(device)
    b_psi.to(device)
    loss_scheme = LossScheme1(config.loss).to(device)
    trainable_parameters = list(iter_lora_parameters(denoiser)) + list(iter_lora_parameters(b_psi))
    if not trainable_parameters:
        raise ValueError("No LoRA parameters were made trainable. Check the target preset and component toggles.")
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=config.optimizer.learning_rate,
        betas=config.optimizer.betas,
        weight_decay=config.optimizer.weight_decay,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=config.runtime.use_amp and device.type == "cuda")
    if resume_payload is not None:
        optimizer_state = resume_payload.get("optimizer_state_dict")
        if not isinstance(optimizer_state, dict):
            raise ValueError("Resume checkpoint optimizer_state_dict is invalid.")
        optimizer.load_state_dict(optimizer_state)
        scaler_state = resume_payload.get("scaler_state_dict")
        if not isinstance(scaler_state, dict):
            raise ValueError("Resume checkpoint scaler_state_dict is invalid.")
        scaler.load_state_dict(scaler_state)

    history: list[dict[str, float]] = []
    latest_checkpoint: Path | None = None if config.resume_from is None else Path(config.resume_from)
    if resume_payload is not None:
        _restore_rng_state(resume_payload.get("rng_state"))
    if resume_epoch >= config.lora_runtime.epochs:
        return {
            "device": str(device),
            "train_dataset_size": len(train_dataset),
            "eval_dataset_size": 0 if eval_dataset is None else len(eval_dataset),
                "latest_checkpoint": str(latest_checkpoint),
            "denoiser_lora_modules": [] if denoiser_report is None else list(denoiser_report.matched_modules),
            "b_psi_lora_modules": [] if b_psi_report is None else list(b_psi_report.matched_modules),
        }
    for epoch in range(resume_epoch + 1, config.lora_runtime.epochs + 1):
        b_psi_end = config.lora_runtime.b_psi_epochs
        bridge_end = config.lora_runtime.epochs - config.lora_runtime.joint_refinement_epochs
        denoiser_end = config.lora_runtime.epochs - config.lora_runtime.joint_refinement_epochs
        stage = "b_psi" if epoch <= b_psi_end else "denoiser"
        if epoch > bridge_end - config.lora_runtime.unconditional_bridge_epochs and epoch <= bridge_end:
            stage = "unconditional_bridge"
        if epoch > denoiser_end:
            stage = "joint"
        if (
            b_psi_end == 0
            and config.lora_runtime.unconditional_bridge_epochs == 0
            and config.lora_runtime.joint_refinement_epochs == 0
        ):
            stage = "joint"
        flow.train(stage != "b_psi")
        b_psi.train(stage not in {"denoiser", "unconditional_bridge"})
        _set_active_lora_stage(denoiser, b_psi, stage=stage)
        epoch_totals: dict[str, float] = {}
        batch_count = 0
        joint_conditional_steps = 0.0
        joint_unconditional_steps = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, config.runtime.use_amp):
                if stage == "b_psi":
                    b_psi_loss, b_psi_metrics = _b_psi_unconditional_loss(
                        loss_scheme,
                        flow,
                        b_psi,
                        batch,
                        device,
                        coarse_weight=config.lora_runtime.b_psi_coarse_reference_weight,
                        unconditional_path_weight=config.lora_runtime.b_psi_unconditional_path_weight,
                    )
                    total_loss = b_psi_loss
                    metrics = dict(b_psi_metrics)
                elif stage == "denoiser":
                    conditional_loss, conditional_metrics = _conditional_loss(loss_scheme, flow, batch, device)
                    total_loss = conditional_loss
                    metrics = dict(conditional_metrics)
                elif stage == "unconditional_bridge":
                    bridge_loss, bridge_metrics = _unconditional_bridge_loss(
                        loss_scheme,
                        flow,
                        b_psi,
                        batch,
                        device,
                        unconditional_path_weight=config.lora_runtime.joint_unconditional_path_weight,
                    )
                    total_loss = bridge_loss
                    metrics = dict(bridge_metrics)
                else:
                    step_is_conditional = _joint_step_is_conditional(
                        batch_count + 1,
                        joint_conditional_steps,
                        config.lora_runtime.joint_conditional_ratio,
                    )
                    if step_is_conditional:
                        conditional_loss, conditional_metrics = _conditional_loss(loss_scheme, flow, batch, device)
                        total_loss = conditional_loss
                        metrics = dict(conditional_metrics)
                        joint_conditional_steps += 1.0
                    else:
                        b_psi_loss, b_psi_metrics = _b_psi_unconditional_loss(
                            loss_scheme,
                            flow,
                            b_psi,
                            batch,
                            device,
                            coarse_weight=config.lora_runtime.joint_coarse_reference_weight,
                            unconditional_path_weight=config.lora_runtime.joint_unconditional_path_weight,
                        )
                        total_loss = b_psi_loss
                        metrics = dict(b_psi_metrics)
                        joint_unconditional_steps += 1.0
                metrics["train_total_loss"] = float(total_loss.detach().item())
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_count += 1
            for key, value in metrics.items():
                epoch_totals[key] = epoch_totals.get(key, 0.0) + float(value)

        epoch_metrics = {key: value / max(batch_count, 1) for key, value in epoch_totals.items()}
        epoch_metrics["epoch"] = float(epoch)
        epoch_metrics["stage_b_psi"] = float(stage == "b_psi")
        epoch_metrics["stage_unconditional_bridge"] = float(stage == "unconditional_bridge")
        epoch_metrics["stage_joint"] = float(stage == "joint")
        if stage == "joint":
            total_joint_steps = max(joint_conditional_steps + joint_unconditional_steps, 1.0)
            epoch_metrics["joint_conditional_step_fraction"] = joint_conditional_steps / total_joint_steps
            epoch_metrics["joint_unconditional_step_fraction"] = joint_unconditional_steps / total_joint_steps
        epoch_metrics["denoiser_lora_params"] = float(count_lora_parameters(denoiser))
        epoch_metrics["b_psi_lora_params"] = float(count_lora_parameters(b_psi))
        if denoiser_report is not None:
            epoch_metrics["denoiser_lora_module_count"] = float(denoiser_report.count)
        if b_psi_report is not None:
            epoch_metrics["b_psi_lora_module_count"] = float(b_psi_report.count)

        if eval_loader is not None and config.lora_runtime.eval_every_epochs > 0 and epoch % config.lora_runtime.eval_every_epochs == 0:
            flow.eval()
            b_psi.eval()
            eval_totals: dict[str, float] = {}
            eval_batches = 0
            with torch.no_grad():
                for batch in eval_loader:
                    if stage == "b_psi":
                        eval_loss, eval_metrics = _b_psi_unconditional_loss(
                            loss_scheme,
                            flow,
                            b_psi,
                            batch,
                            device,
                            coarse_weight=config.lora_runtime.b_psi_coarse_reference_weight,
                            unconditional_path_weight=config.lora_runtime.b_psi_unconditional_path_weight,
                        )
                    elif stage == "unconditional_bridge":
                        eval_loss, eval_metrics = _unconditional_bridge_loss(
                            loss_scheme,
                            flow,
                            b_psi,
                            batch,
                            device,
                            unconditional_path_weight=config.lora_runtime.joint_unconditional_path_weight,
                        )
                    else:
                        eval_loss, eval_metrics = _conditional_loss(loss_scheme, flow, batch, device)
                    eval_metrics["eval_total_loss"] = float(eval_loss.detach().item())
                    eval_batches += 1
                    for key, value in eval_metrics.items():
                        eval_totals[key] = eval_totals.get(key, 0.0) + float(value)
            epoch_metrics.update({key: value / max(eval_batches, 1) for key, value in eval_totals.items()})

        history.append(epoch_metrics)
        if epoch % config.lora_runtime.save_every_epochs == 0 or epoch == config.lora_runtime.epochs:
            latest_checkpoint = _maybe_save_checkpoint(
                config,
                epoch=epoch,
                denoiser=denoiser,
                b_psi=b_psi,
                optimizer=optimizer,
                scaler=scaler,
            )
        print(
            f"Epoch {epoch:04d} | stage={stage} | total={epoch_metrics['train_total_loss']:.6f} | "
            f"denoiser_lora={int(epoch_metrics['denoiser_lora_params'])} | "
            f"b_psi_lora={int(epoch_metrics['b_psi_lora_params'])}"
        )

    return {
        "device": str(device),
        "train_dataset_size": len(train_dataset),
        "eval_dataset_size": 0 if eval_dataset is None else len(eval_dataset),
        "latest_checkpoint": None if latest_checkpoint is None else str(latest_checkpoint),
        "denoiser_lora_modules": [] if denoiser_report is None else list(denoiser_report.matched_modules),
        "b_psi_lora_modules": [] if b_psi_report is None else list(b_psi_report.matched_modules),
    }


__all__ = ["train_lora_finetune"]
