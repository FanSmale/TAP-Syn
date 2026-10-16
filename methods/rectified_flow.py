"""Rectified Flow path for residual velocity generation.

This module reuses the shared conditioned residual U-Net as a velocity-field
predictor along a simple linear path between Gaussian noise and the clean
residual target.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..nets.residual_velocity_unet import ResidualVelocityUNet
from ..unconditional_background import BPsi


class RectifiedFlowPath(nn.Module):
    """Define a linear rectified flow path between Gaussian noise and residual targets."""

    def sample_time(self, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.rand(batch, device=device, dtype=dtype)

    def sample_noise(self, reference: Tensor) -> Tensor:
        return torch.randn_like(reference)

    def interpolate(self, z0: Tensor, r0: Tensor, t: Tensor) -> Tensor:
        t_view = t.view(-1, 1, 1, 1)
        return (1.0 - t_view) * z0 + t_view * r0

    def target_velocity(self, z0: Tensor, r0: Tensor) -> Tensor:
        return r0 - z0

    def euler_step(self, state_t: Tensor, velocity: Tensor, dt: float) -> Tensor:
        return state_t + dt * velocity


class ResidualVelocityRectifiedFlow(nn.Module):
    """Wrap the shared denoiser in a Rectified Flow training and sampling interface."""

    def __init__(self, denoiser: ResidualVelocityUNet, path: Optional[RectifiedFlowPath] = None) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.path = path if path is not None else RectifiedFlowPath()

    def training_loss(
        self,
        r0: Tensor,
        m_c_cond: Tensor,
        t: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
        return_details: bool = True,
    ) -> Dict[str, Tensor]:
        batch = r0.shape[0]
        if t is None:
            t = self.path.sample_time(batch, r0.device, r0.dtype)
        else:
            t = t.to(device=r0.device, dtype=r0.dtype)
        if noise is None:
            noise = self.path.sample_noise(r0)

        state_t = self.path.interpolate(noise, r0, t)
        target = self.path.target_velocity(noise, r0)
        pred = self.denoiser(state_t, t, m_c_cond)
        loss = F.mse_loss(pred, target)

        result = {"loss": loss}
        if return_details:
            result.update({
                "t": t,
                "noise": noise,
                "state_t": state_t,
                "pred": pred,
                "target": target,
                "velocity_pred": pred,
                "target_velocity": target,
            })
        return result

    @torch.no_grad()
    def _sample_residual(
        self,
        m_c_cond: Tensor,
        shape: Optional[Tuple[int, int, int, int]] = None,
        initial_state: Optional[Tensor] = None,
        num_steps: int = 16,
        return_trajectory: bool = False,
    ) -> Tensor | Tuple[Tensor, List[Tensor]]:
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1.")
        if shape is None:
            shape = tuple(m_c_cond.shape)

        if initial_state is None:
            state_t = torch.randn(shape, device=m_c_cond.device, dtype=m_c_cond.dtype)
        else:
            if tuple(initial_state.shape) != tuple(shape):
                raise ValueError(
                    f"initial_state shape {tuple(initial_state.shape)} does not match {tuple(shape)}."
                )
            state_t = initial_state.to(device=m_c_cond.device, dtype=m_c_cond.dtype).clone()
        trajectory: List[Tensor] = [state_t.clone()] if return_trajectory else []
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_value = step / num_steps
            t = torch.full((shape[0],), t_value, device=m_c_cond.device, dtype=m_c_cond.dtype)
            velocity = self.denoiser(state_t, t, m_c_cond)
            state_t = self.path.euler_step(state_t, velocity, dt)
            if return_trajectory:
                trajectory.append(state_t.clone())

        residual = state_t
        if return_trajectory:
            return residual, trajectory
        return residual

    @torch.no_grad()
    def sample_conditional(
        self,
        m_c_cond: Tensor,
        shape: Optional[Tuple[int, int, int, int]] = None,
        initial_state: Optional[Tensor] = None,
        num_steps: int = 16,
        return_trajectory: bool = False,
    ) -> Tensor | Tuple[Tensor, List[Tensor]]:
        return self._sample_residual(
            m_c_cond,
            shape=shape,
            initial_state=initial_state,
            num_steps=num_steps,
            return_trajectory=return_trajectory,
        )

    @torch.no_grad()
    def sample_unconditional(
        self,
        xi: Tensor,
        *,
        b_psi: BPsi,
        shape: Optional[Tuple[int, int, int, int]] = None,
        initial_state: Optional[Tensor] = None,
        num_steps: int = 16,
        return_trajectory: bool = False,
    ) -> Tensor | Tuple[Tensor, List[Tensor]]:
        m_c_cond = b_psi(xi)
        return self._sample_residual(
            m_c_cond,
            shape=shape,
            initial_state=initial_state,
            num_steps=num_steps,
            return_trajectory=return_trajectory,
        )


__all__ = ["RectifiedFlowPath", "ResidualVelocityRectifiedFlow"]
