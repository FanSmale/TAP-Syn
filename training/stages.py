"""Stage-specific training loops and freeze helpers."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from ..losses import LossScheme1
from ..methods import ResidualVelocityRectifiedFlow
from ..nets import ResidualVelocityUNet
from ..unconditional_background import BPsi
from .config import RuntimeConfig, StageConfig

_PARTIAL_BACKBONE_PREFIXES = (
    "coarse_encoder",
    "coarse_projs",
    "input_conv",
    "up2",
    "dec2",
    "dec2_attn",
    "up1",
    "dec1",
    "up0",
    "dec0",
    "out_norm",
    "out_conv",
)


def set_module_trainable(module: torch.nn.Module, trainable: bool) -> None:
    """Enable or disable gradients for all parameters in one module."""

    for parameter in module.parameters():
        parameter.requires_grad = trainable


def freeze_backbone(denoiser: ResidualVelocityUNet) -> None:
    """Freeze every parameter in the shared residual backbone."""

    set_module_trainable(denoiser, False)


def unfreeze_backbone(denoiser: ResidualVelocityUNet) -> None:
    """Unfreeze every parameter in the shared residual backbone."""

    set_module_trainable(denoiser, True)


def partially_unfreeze_backbone(denoiser: ResidualVelocityUNet) -> None:
    """Unfreeze a small allowlist of coarse-conditioning and decoder blocks."""

    freeze_backbone(denoiser)
    for name, parameter in denoiser.named_parameters():
        if name.startswith(_PARTIAL_BACKBONE_PREFIXES):
            parameter.requires_grad = True


def configure_stage_trainability(denoiser: ResidualVelocityUNet, b_psi: BPsi, stage: StageConfig) -> None:
    """Apply the stage-specific freeze policy to backbone and coarse generator."""

    if stage.backbone_mode == "frozen":
        freeze_backbone(denoiser)
    elif stage.backbone_mode == "partial":
        partially_unfreeze_backbone(denoiser)
    elif stage.backbone_mode == "full":
        unfreeze_backbone(denoiser)
    else:
        raise ValueError(f"Unsupported backbone mode: {stage.backbone_mode}.")

    if stage.train_backbone and stage.backbone_mode == "frozen":
        raise ValueError(f"Stage {stage.name} cannot train a frozen backbone.")

    set_module_trainable(b_psi, stage.train_b_psi)
    denoiser.train(mode=stage.train_backbone)
    b_psi.train(mode=stage.train_b_psi)


def count_trainable_parameters(module: torch.nn.Module) -> int:
    """Count parameters that currently require gradients."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _autocast_context(device: torch.device, runtime: RuntimeConfig):
    enabled = runtime.use_amp and device.type == "cuda"
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def make_grad_scaler(runtime: RuntimeConfig, device: torch.device) -> torch.amp.GradScaler:
    """Construct one grad scaler when AMP is enabled on CUDA."""

    return torch.amp.GradScaler(device.type, enabled=runtime.use_amp and device.type == "cuda")


def _prepare_conditional_batch(batch: dict[str, Any], device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    velocity = batch["velocity"].to(device)
    velocity_norm = batch["velocity_norm"].to(device)
    m_c_cond = batch["m_c_cond"].to(device)
    r0 = batch["r0"].to(device)
    return velocity, velocity_norm, m_c_cond, r0


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


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _tensor_stats(prefix: str, tensor: Tensor) -> dict[str, float]:
    data = tensor.detach().float()
    flattened = data.flatten(start_dim=1)
    q05 = torch.quantile(flattened, 0.05, dim=1)
    q95 = torch.quantile(flattened, 0.95, dim=1)
    return {
        f"{prefix}_mean": float(flattened.mean().item()),
        f"{prefix}_std": float(flattened.std(unbiased=False).item()),
        f"{prefix}_min": float(flattened.min().item()),
        f"{prefix}_max": float(flattened.max().item()),
        f"{prefix}_q05": float(q05.mean().item()),
        f"{prefix}_q95": float(q95.mean().item()),
        f"{prefix}_range": float((q95 - q05).mean().item()),
        f"{prefix}_max_abs": float(flattened.abs().max().item()),
    }


def _collect_nonfinite_names(named_tensors: dict[str, Tensor]) -> list[str]:
    return [name for name, tensor in named_tensors.items() if not torch.isfinite(tensor).all()]


def _optimizer_parameters(optimizer: torch.optim.Optimizer) -> list[Tensor]:
    params: list[Tensor] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            parameter_id = id(parameter)
            if parameter_id in seen:
                continue
            seen.add(parameter_id)
            params.append(parameter)
    return params


def _module_grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    found = False
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        grad_norm = float(parameter.grad.detach().float().norm(2).item())
        total += grad_norm * grad_norm
        found = True
    return total ** 0.5 if found else 0.0


def _nonfinite_gradient_count(parameters: list[Tensor]) -> int:
    """Count trainable parameters whose computed gradient is not finite."""

    return sum(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in parameters)


def _conditional_training_loss(
    loss_scheme: LossScheme1,
    flow: ResidualVelocityRectifiedFlow,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
    velocity, velocity_norm, m_c_cond, r0 = _prepare_conditional_batch(batch, device)
    flow_details = _rf_training_details(flow, r0=r0, m_c_cond=m_c_cond)
    loss, metrics, outputs = loss_scheme.compute_total_loss(
        velocity_norm=velocity_norm,
        m_c_cond=m_c_cond,
        flow_details=flow_details,
        include_coarse=False,
    )
    recon_mae = F.l1_loss(outputs["reconstruction_norm"], velocity_norm)
    prefixed = _prefix_metrics("conditional", metrics)
    prefixed["conditional_recon_mae"] = float(recon_mae.detach().item())
    prefixed["batch_size"] = float(velocity_norm.shape[0])
    prefixed.update(_tensor_stats("conditional_coarse_cond", m_c_cond))
    prefixed.update(_tensor_stats("conditional_pred", flow_details["pred"]))
    prefixed.update(_tensor_stats("conditional_target", flow_details["target"]))
    tensors_to_check = {
        "conditional_loss": loss,
        "conditional_pred": flow_details["pred"],
        "conditional_target": flow_details["target"],
        "conditional_reconstruction": outputs["reconstruction_norm"],
    }
    return loss, prefixed, tensors_to_check


def _unconditional_training_loss(
    loss_scheme: LossScheme1,
    flow: ResidualVelocityRectifiedFlow,
    b_psi: BPsi,
    batch: dict[str, Any],
    device: torch.device,
    stage: StageConfig,
) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
    velocity, velocity_norm, reference_m_c_cond, _ = _prepare_conditional_batch(batch, device)
    xi = torch.randn_like(reference_m_c_cond)
    generated_m_c_cond = b_psi(xi)
    generated_r0 = velocity_norm - generated_m_c_cond
    flow_details = _rf_training_details(flow, r0=generated_r0, m_c_cond=generated_m_c_cond)
    loss, metrics, outputs = loss_scheme.compute_total_loss(
        velocity_norm=velocity_norm,
        m_c_cond=generated_m_c_cond,
        flow_details=flow_details,
        include_coarse=True,
        coarse_reference_cond=reference_m_c_cond,
        unconditional_path_weight=stage.unconditional_path_weight,
        coarse_weight=stage.coarse_reference_weight,
    )
    recon_mae = F.l1_loss(outputs["reconstruction_norm"], velocity_norm)
    prefixed = _prefix_metrics("unconditional", metrics)
    prefixed["unconditional_recon_mae"] = float(recon_mae.detach().item())
    prefixed["batch_size"] = float(velocity.shape[0])
    prefixed.update(_tensor_stats("generated_coarse_cond", generated_m_c_cond))
    prefixed.update(_tensor_stats("reference_coarse_cond", reference_m_c_cond))
    prefixed.update(_tensor_stats("generated_r0", generated_r0))
    prefixed.update(_tensor_stats("unconditional_pred", flow_details["pred"]))
    prefixed.update(_tensor_stats("unconditional_target", flow_details["target"]))
    tensors_to_check = {
        "unconditional_loss": loss,
        "generated_m_c_cond": generated_m_c_cond,
        "generated_r0": generated_r0,
        "unconditional_pred": flow_details["pred"],
        "unconditional_target": flow_details["target"],
        "unconditional_reconstruction": outputs["reconstruction_norm"],
    }
    return loss, prefixed, tensors_to_check


def _optimize_loss(
    loss: Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    runtime: RuntimeConfig,
    *,
    denoiser: ResidualVelocityUNet,
    b_psi: BPsi | None = None,
) -> dict[str, float]:
    optimization_metrics = {
        "optimization_skipped_nonfinite": 0.0,
        "optimization_nonfinite_grad_count": 0.0,
        "optimization_grad_norm": 0.0,
        "optimization_denoiser_grad_norm": 0.0,
        "optimization_b_psi_grad_norm": 0.0,
    }
    optimizer.zero_grad(set_to_none=True)
    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        parameters = _optimizer_parameters(optimizer)
        nonfinite_gradients = _nonfinite_gradient_count(parameters)
        if nonfinite_gradients:
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            optimization_metrics["optimization_skipped_nonfinite"] = 1.0
            optimization_metrics["optimization_nonfinite_grad_count"] = float(nonfinite_gradients)
            return optimization_metrics
        optimization_metrics["optimization_denoiser_grad_norm"] = _module_grad_norm(denoiser)
        if b_psi is not None:
            optimization_metrics["optimization_b_psi_grad_norm"] = _module_grad_norm(b_psi)
        if runtime.grad_clip_norm is not None and parameters:
            optimization_metrics["optimization_grad_norm"] = float(
                torch.nn.utils.clip_grad_norm_(parameters, runtime.grad_clip_norm).item()
            )
        scaler.step(optimizer)
        scaler.update()
        return optimization_metrics

    loss.backward()
    parameters = _optimizer_parameters(optimizer)
    nonfinite_gradients = _nonfinite_gradient_count(parameters)
    if nonfinite_gradients:
        optimizer.zero_grad(set_to_none=True)
        optimization_metrics["optimization_skipped_nonfinite"] = 1.0
        optimization_metrics["optimization_nonfinite_grad_count"] = float(nonfinite_gradients)
        return optimization_metrics
    optimization_metrics["optimization_denoiser_grad_norm"] = _module_grad_norm(denoiser)
    if b_psi is not None:
        optimization_metrics["optimization_b_psi_grad_norm"] = _module_grad_norm(b_psi)
    if runtime.grad_clip_norm is not None and parameters:
        optimization_metrics["optimization_grad_norm"] = float(
            torch.nn.utils.clip_grad_norm_(parameters, runtime.grad_clip_norm).item()
        )
    optimizer.step()
    return optimization_metrics


def _accumulate_weighted(summary: dict[str, float], metrics: dict[str, float], *, weight: float = 1.0) -> None:
    for key, value in metrics.items():
        if key == "batch_size":
            continue
        summary[key] = summary.get(key, 0.0) + value * weight
    summary["weight"] = summary.get("weight", 0.0) + weight


def _finalize(summary: dict[str, float]) -> dict[str, float]:
    total_weight = max(summary.pop("weight", 1.0), 1.0)
    return {key: value / total_weight for key, value in summary.items()}


def _merge_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items() if key != "batch_size"}


def _handle_nonfinite_step(
    metrics: dict[str, float],
    tensors_to_check: dict[str, Tensor],
    optimizer: torch.optim.Optimizer,
    runtime: RuntimeConfig,
) -> bool:
    nonfinite_names = _collect_nonfinite_names(tensors_to_check)
    if not nonfinite_names:
        metrics["nonfinite_detected"] = 0.0
        metrics["nonfinite_tensor_count"] = 0.0
        return False
    metrics["nonfinite_detected"] = 1.0
    metrics["nonfinite_tensor_count"] = float(len(nonfinite_names))
    if runtime.log_nonfinite_tensors:
        for name in nonfinite_names:
            metrics[f"nonfinite_{name}"] = 1.0
    if runtime.skip_nonfinite_steps:
        optimizer.zero_grad(set_to_none=True)
    return runtime.skip_nonfinite_steps


def _mixed_step_is_conditional(batch_index: int, conditional_steps: float, conditional_ratio: float) -> bool:
    if not 0.0 < conditional_ratio < 1.0:
        raise ValueError("conditional_ratio must be between 0 and 1 for mixed stage training.")
    return conditional_steps < batch_index * conditional_ratio


def run_stage1_epoch(
    loss_scheme: LossScheme1,
    flow: ResidualVelocityRectifiedFlow,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    runtime: RuntimeConfig,
) -> dict[str, float]:
    """Run one conditional-only training epoch."""

    summary: dict[str, float] = {}
    flow.train()
    for batch_index, batch in enumerate(loader, start=1):
        with _autocast_context(device, runtime):
            loss, metrics, tensors_to_check = _conditional_training_loss(loss_scheme, flow, batch, device)
        if _handle_nonfinite_step(metrics, tensors_to_check, optimizer, runtime):
            metrics["optimization_skipped_nonfinite"] = 1.0
            _accumulate_weighted(summary, metrics, weight=metrics["batch_size"])
            continue
        metrics.update(_optimize_loss(loss, optimizer, scaler, runtime, denoiser=flow.denoiser))
        _accumulate_weighted(summary, metrics, weight=metrics["batch_size"])
    return _finalize(summary)


def run_stage2_epoch(
    loss_scheme: LossScheme1,
    flow: ResidualVelocityRectifiedFlow,
    b_psi: BPsi,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    runtime: RuntimeConfig,
    stage: StageConfig,
) -> dict[str, float]:
    """Run one unconditional warm-up epoch."""

    summary: dict[str, float] = {}
    flow.train()
    b_psi.train()
    for batch_index, batch in enumerate(loader, start=1):
        with _autocast_context(device, runtime):
            loss, metrics, tensors_to_check = _unconditional_training_loss(loss_scheme, flow, b_psi, batch, device, stage)
        if _handle_nonfinite_step(metrics, tensors_to_check, optimizer, runtime):
            metrics["optimization_skipped_nonfinite"] = 1.0
            _accumulate_weighted(summary, metrics, weight=metrics["batch_size"])
            continue
        metrics.update(_optimize_loss(loss, optimizer, scaler, runtime, denoiser=flow.denoiser, b_psi=b_psi))
        _accumulate_weighted(summary, metrics, weight=metrics["batch_size"])
    return _finalize(summary)


def run_stage3_epoch(
    loss_scheme: LossScheme1,
    flow: ResidualVelocityRectifiedFlow,
    b_psi: BPsi,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    runtime: RuntimeConfig,
    stage: StageConfig,
) -> dict[str, float]:
    """Run one mixed conditional/unconditional epoch with configurable step ratio."""

    conditional_summary: dict[str, float] = {}
    unconditional_summary: dict[str, float] = {}
    flow.train()
    b_psi.train()
    conditional_steps = 0.0
    unconditional_steps = 0.0
    for batch_index, batch in enumerate(loader, start=1):
        step_is_conditional = _mixed_step_is_conditional(batch_index, conditional_steps, stage.conditional_ratio)
        with _autocast_context(device, runtime):
            if step_is_conditional:
                set_module_trainable(b_psi, False)
                loss, metrics, tensors_to_check = _conditional_training_loss(loss_scheme, flow, batch, device)
                conditional_steps += 1.0
            else:
                set_module_trainable(b_psi, stage.train_b_psi)
                loss, metrics, tensors_to_check = _unconditional_training_loss(loss_scheme, flow, b_psi, batch, device, stage)
                unconditional_steps += 1.0
        if _handle_nonfinite_step(metrics, tensors_to_check, optimizer, runtime):
            metrics["optimization_skipped_nonfinite"] = 1.0
            if step_is_conditional:
                _accumulate_weighted(conditional_summary, metrics, weight=metrics["batch_size"])
            else:
                _accumulate_weighted(unconditional_summary, metrics, weight=metrics["batch_size"])
            continue
        if step_is_conditional:
            metrics.update(_optimize_loss(loss, optimizer, scaler, runtime, denoiser=flow.denoiser))
            _accumulate_weighted(conditional_summary, metrics, weight=metrics["batch_size"])
        else:
            metrics.update(_optimize_loss(loss, optimizer, scaler, runtime, denoiser=flow.denoiser, b_psi=b_psi))
            _accumulate_weighted(unconditional_summary, metrics, weight=metrics["batch_size"])
    set_module_trainable(b_psi, stage.train_b_psi)

    result: dict[str, float] = {
        "stage3_conditional_step_fraction": conditional_steps / max(conditional_steps + unconditional_steps, 1.0),
        "stage3_unconditional_step_fraction": unconditional_steps / max(conditional_steps + unconditional_steps, 1.0),
    }
    if conditional_summary:
        result.update(_merge_metrics("stage3_conditional", _finalize(conditional_summary)))
    if unconditional_summary:
        result.update(_merge_metrics("stage3_unconditional", _finalize(unconditional_summary)))
    return result


@torch.no_grad()
def evaluate_conditional_epoch(
    loss_scheme: LossScheme1,
    flow: ResidualVelocityRectifiedFlow,
    loader: DataLoader | None,
    device: torch.device,
    runtime: RuntimeConfig,
) -> dict[str, float]:
    """Evaluate the conditional path on a separate dataset object when present."""

    if loader is None:
        return {}

    summary: dict[str, float] = {}
    flow.eval()
    for batch in loader:
        velocity, velocity_norm, m_c_cond, r0 = _prepare_conditional_batch(batch, device)
        flow_details = _rf_training_details(flow, r0=r0, m_c_cond=m_c_cond)
        _, metrics, _ = loss_scheme.compute_total_loss(
            velocity_norm=velocity_norm,
            m_c_cond=m_c_cond,
            flow_details=flow_details,
            include_coarse=False,
        )
        sampled_residual = flow.sample_conditional(m_c_cond, num_steps=runtime.eval_sample_steps)
        sampled_recon_norm = loss_scheme.reconstruct_velocity(m_c_cond, sampled_residual)
        sampled_mae = F.l1_loss(sampled_recon_norm, velocity_norm)
        eval_metrics = {f"eval_{key}": value for key, value in metrics.items()}
        eval_metrics["eval_sample_mae"] = float(sampled_mae.detach().item())
        eval_metrics["batch_size"] = float(velocity_norm.shape[0])
        _accumulate_weighted(summary, eval_metrics, weight=eval_metrics["batch_size"])
    flow.train()
    return _finalize(summary)
