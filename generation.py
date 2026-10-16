"""Proxy synthesis. Anchor numerics are extracted from paper_experiments/synthetic_data.py.

Expanded 128² generator velocities are resized to 70² and forward modeled. Only
original observed pairs bypass the solver. The output contains physical m/s npy
pairs, without seed plans, manifests, provenance files or persisted metrics.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from .data import PairedNpyDataset, VelocityNpyDataset
from .data.velocity_npy import default_velocity_denormalizer
from .forward import OpenFWIForward
from .training.config import ModelConfig, TrainingStrategyConfig, DatasetConfig
from .training.factory import build_model_stack, resolve_device
from .lora import inject_lora, load_base_component_state, load_lora_state_dict


def augment_anchor_velocity(velocity: Tensor) -> Tensor:
    """Source anchor transform: smooth warp and masked velocity perturbation."""
    _, _, height, width = velocity.shape
    low = torch.randn((1, 2, 9, 9), device=velocity.device)
    displacement = F.interpolate(low, (height, width), mode="bicubic", align_corners=True)
    displacement = displacement / displacement.abs().amax().clamp_min(1e-6) * 0.045
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=velocity.device),
        torch.linspace(-1.0, 1.0, width, device=velocity.device),
        indexing="ij",
    )
    grid = torch.stack([xx + displacement[:, 0], yy + displacement[:, 1]], dim=-1)
    warped = F.grid_sample(velocity, grid, mode="bilinear", padding_mode="border", align_corners=True)
    residual = F.interpolate(
        torch.randn((1, 1, 9, 9), device=velocity.device),
        (height, width), mode="bicubic", align_corners=True,
    )
    residual = residual / residual.abs().amax().clamp_min(1e-6) * 0.025
    gradient = (velocity[:, :, 1:, :] - velocity[:, :, :-1, :]).abs()
    gradient = F.pad(gradient, (0, 0, 1, 0)) + F.pad((velocity[:, :, :, 1:] - velocity[:, :, :, :-1]).abs(), (1, 0, 0, 0))
    mask = (gradient < gradient.flatten().quantile(0.7)).to(velocity.dtype)
    return (warped + residual * mask).clamp(0.0, 1.0)


@dataclass(frozen=True)
class ProxyConfig:
    anchor_per_reference: int = 14
    conditional_per_reference: int = 12
    unconditional_per_reference: int = 36
    batch_size: int = 8
    chunk_size: int = 8
    num_steps: int = 16

    def __post_init__(self):
        if min(self.anchor_per_reference, self.conditional_per_reference, self.unconditional_per_reference) < 0:
            raise ValueError("Proxy counts cannot be negative")
        if min(self.batch_size, self.chunk_size, self.num_steps) < 1:
            raise ValueError("Batch/chunk sizes and RF steps must be positive")


class PairChunkWriter:
    """Write bounded npy chunks only; refuse nonempty output directories."""
    def __init__(self, root, chunk_size=8):
        self.root = Path(root)
        if self.root.exists() and (not self.root.is_dir() or any(self.root.iterdir())):
            raise FileExistsError(f"Output must be absent or empty: {self.root}")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        (self.root / "velocity").mkdir(parents=True)
        (self.root / "seismic").mkdir()
        self.chunk_size = chunk_size
        self.velocities, self.seismics = [], []
        self.chunk = self.count = 0

    def add(self, velocity_physical, seismic):
        if tuple(velocity_physical.shape[1:]) != (1, 70, 70) or tuple(seismic.shape[1:]) != (5, 1000, 70):
            raise ValueError("Expected batched 70² velocity and [5,1000,70] seismic")
        if len(velocity_physical) != len(seismic):
            raise ValueError("Pair lengths differ")
        if not torch.isfinite(velocity_physical).all() or not torch.isfinite(seismic).all():
            raise ValueError("Cannot save nonfinite pairs")
        for velocity, record in zip(velocity_physical, seismic):
            self.velocities.append(velocity.detach().cpu().float())
            self.seismics.append(record.detach().cpu().float())
            self.count += 1
            if len(self.velocities) == self.chunk_size:
                self.flush()

    def flush(self):
        if not self.velocities:
            return
        name = f"chunk{self.chunk:06d}.npy"
        for directory, tensors in (("velocity", self.velocities), ("seismic", self.seismics)):
            destination = self.root / directory / name
            temporary = destination.with_suffix('.tmp')
            with temporary.open('wb') as handle:
                np.save(handle, torch.stack(tensors).numpy(), allow_pickle=False)
            temporary.replace(destination)
        self.velocities.clear()
        self.seismics.clear()
        self.chunk += 1


def load_generation_stack(checkpoint_path, *, model: ModelConfig = ModelConfig(), device="cpu",
                          base_checkpoint=None):
    """Accept RF source, full LoRA, or adapter-only plus its source checkpoint.

    Checkpoints use torch pickle: load trusted files only. Architecture is
    supplied by the caller for source checkpoints; full LoRA saves it itself.
    """
    device = resolve_device(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = payload.get("model_config", model)
    denoiser, flow, b_psi = build_model_stack(TrainingStrategyConfig(
        train_dataset=DatasetConfig(), eval_dataset=None, model=model))
    lora_config = payload.get("lora_config")
    if lora_config is None:
        denoiser.load_state_dict(payload['denoiser_state_dict'])
        b_psi.load_state_dict(payload['b_psi_state_dict'])
    else:
        full = 'denoiser_state_dict' in payload and 'b_psi_state_dict' in payload
        if not full:
            if base_checkpoint is None:
                raise ValueError("Adapter-only inference requires base_checkpoint")
            base = torch.load(base_checkpoint, map_location=device, weights_only=False)
            load_base_component_state(denoiser, base, 'denoiser_state_dict')
            load_base_component_state(b_psi, base, 'b_psi_state_dict')
        if lora_config.enable_denoiser_lora:
            inject_lora(denoiser, component="denoiser", config=lora_config)
        if lora_config.enable_b_psi_lora:
            inject_lora(b_psi, component="b_psi", config=lora_config)
        if full:
            denoiser.load_state_dict(payload['denoiser_state_dict'])
            b_psi.load_state_dict(payload['b_psi_state_dict'])
        else:
            load_lora_state_dict(denoiser, payload['denoiser_lora_state_dict'])
            load_lora_state_dict(b_psi, payload['b_psi_lora_state_dict'])
    return flow.to(device).eval(), b_psi.to(device).eval()


@torch.no_grad()
def build_proxy(*, reference_root, output_root, flow, b_psi, forward: OpenFWIForward,
                config: ProxyConfig = ProxyConfig(), device="cpu", max_references=None,
                coarse_method="mixed", coarse_profile="strong_less_smooth"):
    """Each *training* reference supplies 1 original + 14/12/36 expansions.

    Input reference_root must exclude validation/test. Conditional realizations
    share one D(m) per reference; anchors use the original source transform at
    FWI resolution. All generated fields use the original RF at 128², followed
    by bilinear 70² resize in normalized space and physical denormalization.
    """
    paired = PairedNpyDataset(reference_root, max_samples=max_references)
    conditions = VelocityNpyDataset(reference_root, max_samples=max_references,
                                    coarse_method=coarse_method, coarse_profile=coarse_profile)
    device = resolve_device(str(device))
    flow.to(device).eval()
    b_psi.to(device).eval()
    writer = PairChunkWriter(output_root, config.chunk_size)
    counts = {"original": 0, "anchor": 0, "conditional": 0, "unconditional": 0}

    def add_generated(velocity_norm, kind):
        velocity70 = F.interpolate(velocity_norm, (70, 70), mode='bilinear', align_corners=False, antialias=True)
        physical = default_velocity_denormalizer(velocity70).cpu()
        writer.add(physical, forward.simulate(physical))
        counts[kind] += len(physical)

    for i in range(len(paired)):
        original = paired[i]
        writer.add(default_velocity_denormalizer(original['velocity'])[None], original['seismic'][None])
        counts['original'] += 1
        for start in range(0, config.anchor_per_reference, config.batch_size):
            n = min(config.batch_size, config.anchor_per_reference - start)
            anchors = torch.cat([augment_anchor_velocity(original['velocity'][None].to(device)) for _ in range(n)])
            add_generated(anchors, 'anchor')
        if config.conditional_per_reference:
            condition = conditions[i]['m_c_cond'][None].to(device)
            for start in range(0, config.conditional_per_reference, config.batch_size):
                n = min(config.batch_size, config.conditional_per_reference - start)
                coarse = condition.expand(n, -1, -1, -1)
                residual = flow.sample_conditional(coarse, num_steps=config.num_steps)
                add_generated((coarse + residual).clamp(0, 1), 'conditional')
        print(f"proxy reference {i + 1}/{len(paired)}", flush=True)
    total_unconditional = len(paired) * config.unconditional_per_reference
    for start in range(0, total_unconditional, config.batch_size):
        n = min(config.batch_size, total_unconditional - start)
        coarse = b_psi(torch.randn(n, 1, 128, 128, device=device))
        residual = flow.sample_conditional(coarse, num_steps=config.num_steps)
        add_generated((coarse + residual).clamp(0, 1), 'unconditional')
    writer.flush()
    print(f"proxy pairs: {counts}; total={writer.count}", flush=True)
    return counts
