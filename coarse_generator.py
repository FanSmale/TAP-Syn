"""Coarse velocity model degradation utilities.

This module provides shape-preserving degradations that turn a full-resolution
velocity model into a coarser background model. The current implementation is
intended for prototype data preparation and conditioning experiments rather than
for exact physical simulation.

Supported input shapes are:
- ``[X, Y]``
- ``[B, X, Y]``
- ``[B, 1, X, Y]``

All degradations internally operate on ``[B, 1, X, Y]`` tensors and then restore
that original layout before returning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class CoarseConfig:
    """Configuration ranges for coarse-model degradation operators.

    Each range controls how aggressively a particular degradation can simplify
    the input model while still preserving its shape.
    """

    gaussian_sigma_range: tuple[float, float] = (2.0, 12.0)
    downsample_factor_range: tuple[int, int] = (2, 8)
    fourier_cutoff_range: tuple[float, float] = (0.05, 0.25)
    tomography_sigma_x_range: tuple[float, float] = (3.0, 10.0)
    tomography_sigma_y_range: tuple[float, float] = (8.0, 30.0)
    thin_layer_kernel_range: tuple[int, int] = (5, 21)
    bias_strength_range: tuple[float, float] = (0.02, 0.12)
    noise_strength_range: tuple[float, float] = (0.0, 0.02)
    mixed_num_ops_range: tuple[int, int] = (2, 4)
    mixed_warp_probability: float = 0.1
    mixed_warp_method_weights: tuple[float, float] = (1.0, 1.0)
    horizon_warp_shift_frac_range: tuple[float, float] = (0.005, 0.02)
    horizon_warp_sigma_range: tuple[float, float] = (8.0, 24.0)
    depth_warp_shift_frac_range: tuple[float, float] = (0.005, 0.015)
    depth_warp_sigma_range: tuple[float, float] = (10.0, 28.0)
    clamp_to_input_range: bool = True
    clamp_each_mixed_step: bool = True
    padding_mode: str = "reflect"
    interpolation_mode: str = "bilinear"
    mixed_method_weights: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    mixed_require_severe_op: bool = False


CoarseProfileName = Literal["default", "strong_less_smooth", "strong", "very_strong"]


def coarse_config_for_profile(profile: CoarseProfileName) -> CoarseConfig:
    if profile == "default":
        return CoarseConfig()
    if profile == "strong_less_smooth":
        return CoarseConfig(
            gaussian_sigma_range=(4.5, 14.0),
            downsample_factor_range=(4, 12),
            fourier_cutoff_range=(0.05, 0.16),
            tomography_sigma_x_range=(5.0, 14.0),
            tomography_sigma_y_range=(12.0, 30.0),
            thin_layer_kernel_range=(11, 35),
            bias_strength_range=(0.04, 0.18),
            noise_strength_range=(0.0, 0.04),
            mixed_num_ops_range=(3, 5),
            mixed_warp_probability=0.2,
            mixed_warp_method_weights=(1.2, 1.0),
            horizon_warp_shift_frac_range=(0.008, 0.03),
            horizon_warp_sigma_range=(10.0, 28.0),
            depth_warp_shift_frac_range=(0.006, 0.02),
            depth_warp_sigma_range=(12.0, 30.0),
            clamp_each_mixed_step=False,
            mixed_method_weights=(0.5, 1.8, 1.6, 1.0, 1.3, 1.1),
            mixed_require_severe_op=True,
        )
    if profile == "strong":
        return CoarseConfig(
            gaussian_sigma_range=(6.0, 20.0),
            downsample_factor_range=(4, 12),
            fourier_cutoff_range=(0.03, 0.12),
            tomography_sigma_x_range=(6.0, 18.0),
            tomography_sigma_y_range=(16.0, 42.0),
            thin_layer_kernel_range=(11, 35),
            bias_strength_range=(0.04, 0.18),
            noise_strength_range=(0.0, 0.04),
            mixed_num_ops_range=(3, 5),
            mixed_warp_probability=0.2,
            mixed_warp_method_weights=(1.3, 1.0),
            horizon_warp_shift_frac_range=(0.01, 0.035),
            horizon_warp_sigma_range=(12.0, 32.0),
            depth_warp_shift_frac_range=(0.008, 0.025),
            depth_warp_sigma_range=(14.0, 34.0),
            clamp_each_mixed_step=False,
            mixed_method_weights=(0.8, 1.4, 1.5, 1.5, 1.3, 0.7),
            mixed_require_severe_op=True,
        )
    if profile == "very_strong":
        return CoarseConfig(
            gaussian_sigma_range=(10.0, 28.0),
            downsample_factor_range=(6, 16),
            fourier_cutoff_range=(0.02, 0.08),
            tomography_sigma_x_range=(10.0, 24.0),
            tomography_sigma_y_range=(24.0, 56.0),
            thin_layer_kernel_range=(17, 49),
            bias_strength_range=(0.06, 0.24),
            noise_strength_range=(0.0, 0.06),
            mixed_num_ops_range=(4, 5),
            mixed_warp_probability=0.25,
            mixed_warp_method_weights=(1.4, 1.0),
            horizon_warp_shift_frac_range=(0.012, 0.04),
            horizon_warp_sigma_range=(14.0, 36.0),
            depth_warp_shift_frac_range=(0.01, 0.03),
            depth_warp_sigma_range=(16.0, 40.0),
            clamp_each_mixed_step=False,
            mixed_method_weights=(0.6, 1.5, 1.7, 1.7, 1.4, 0.5),
            mixed_require_severe_op=True,
        )
    raise ValueError(f"Unsupported coarse profile: {profile}.")


class CoarseVelocityGenerator:
    """Generate coarse velocity backgrounds without changing tensor shape.

    The generator exposes several degradation operators, including smoothing,
    resolution loss, low-pass filtering, thin-layer suppression, smooth bias
    perturbation, and mild geometry-preserving warps. These operators can be
    applied individually, selected at random, or composed sequentially through
    the ``mixed`` mode.
    """

    _base_methods = ("gaussian", "downup", "fourier", "tomography", "thin_layer", "bias")
    _warp_methods = ("horizon_warp", "depth_warp")
    _supported_methods = _base_methods + _warp_methods + ("mixed", "random")

    def __init__(self, config: CoarseConfig | None = None) -> None:
        """Initialize the coarse-model generator.

        Args:
            config: Optional degradation configuration. Defaults to
                :class:`CoarseConfig`.
        """
        self.config = config or CoarseConfig()
        if self.config.padding_mode not in {"reflect", "replicate"}:
            raise ValueError("padding_mode must be 'reflect' or 'replicate'.")
        if self.config.interpolation_mode not in {"nearest", "bilinear", "bicubic"}:
            raise ValueError("interpolation_mode must be 'nearest', 'bilinear', or 'bicubic'.")

    def __call__(
        self,
        model: np.ndarray | torch.Tensor,
        method: str = "mixed",
        seed: int | None = None,
        return_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Forward to :meth:`degrade` for convenience."""
        return self.degrade(model, method=method, seed=seed, return_info=return_info)

    def degrade(
        self,
        model: np.ndarray | torch.Tensor,
        method: str = "mixed",
        seed: int | None = None,
        return_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Degrade a velocity model into a coarser background.

        Args:
            model: Input velocity model as a NumPy array or torch tensor with
                shape ``[X, Y]``, ``[B, X, Y]``, or ``[B, 1, X, Y]``.
            method: Degradation mode. Supported values are ``gaussian``,
                ``downup``, ``fourier``, ``tomography``, ``thin_layer``,
                ``bias``, ``mixed``, and ``random``.
            seed: Optional random seed for reproducible operator sampling and
                parameter draws.
            return_info: Whether to return a metadata dictionary describing the
                applied operator(s).

        Returns:
            The degraded coarse model with the same logical shape as the input.
            If ``return_info`` is true, also returns a metadata dictionary.

        Example:
            >>> generator = CoarseVelocityGenerator()
            >>> model = torch.randn(1, 128, 128)
            >>> coarse = generator(model, method="gaussian")
            >>> tuple(coarse.shape)
            (1, 128, 128)
            >>> coarse2, info = generator(model, method="mixed", return_info=True)
            >>> info["selected_ops"]
            ['gaussian', 'bias']

        Notes:
            ``mixed`` samples several distinct base degradations and applies them
            in sequence. ``random`` samples exactly one base degradation.
        """
        self._validate_method(method)
        x, meta = self._to_b1xy_tensor(model)
        self._assert_finite(x, "input")

        generator = self._make_generator(seed, x.device)
        input_min, input_max = self._input_range(x)

        if method == "mixed":
            y, steps = self.mixed_degradation(x, generator, input_min, input_max)
            resolved_method = "mixed"
        elif method == "random":
            selected_method = self._choose_random_method(generator, x.device)
            y, params = self._apply_method(selected_method, x, generator, input_min, input_max)
            steps = [self._make_step_info(selected_method, params)]
            resolved_method = selected_method
        else:
            y, params = self._apply_method(method, x, generator, input_min, input_max)
            steps = [self._make_step_info(method, params)]
            resolved_method = method

        if self.config.clamp_to_input_range:
            y = self._clamp_to_input_range(y, input_min, input_max)
        self._assert_finite(y, "output")

        coarse = self._restore_shape(y, meta)
        info = {
            "method": method,
            "resolved_method": resolved_method,
            "seed": seed,
            "original_type": meta["original_type"],
            "original_shape": meta["original_shape"],
            "internal_shape": tuple(x.shape),
            "selected_ops": [step["method"] for step in steps],
            "params": [step["params"] for step in steps],
            "steps": steps,
            "display_ops": [step["display_summary"] for step in steps],
            "has_warp": any(step["category"] == "warp" for step in steps),
            "warp_summaries": [step["display_summary"] for step in steps if step["category"] == "warp"],
            "input_min": float(x.amin().item()),
            "input_max": float(x.amax().item()),
            "output_min": float(y.amin().item()),
            "output_max": float(y.amax().item()),
        }
        if return_info:
            return coarse, info
        return coarse

    def gaussian_smoothing(
        self,
        x: torch.Tensor,
        sigma: float | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Apply isotropic Gaussian smoothing.

        This operator removes high-frequency detail while preserving the global
        background trend. It is the simplest way to mimic a coarse model whose
        fine structures have been blurred away.
        """
        sigma_value = sigma if sigma is not None else self._rand_uniform(
            generator,
            self.config.gaussian_sigma_range[0],
            self.config.gaussian_sigma_range[1],
            x.device,
        )
        y, kernel_x, kernel_y = self._gaussian_blur(x, sigma_x=sigma_value, sigma_y=sigma_value)
        return y, {
            "sigma": float(sigma_value),
            "kernel_x": kernel_x,
            "kernel_y": kernel_y,
        }

    def downsample_upsample(
        self,
        x: torch.Tensor,
        factor: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reduce and then restore resolution to remove sub-grid detail.

        The model is first interpolated to a lower resolution and then resized
        back to the original shape. This approximates information loss caused by
        a coarser spatial representation.
        """
        if x.shape[-2] < 2 or x.shape[-1] < 2:
            raise ValueError("downsample_upsample requires spatial dimensions of at least 2x2.")

        factor_value = factor if factor is not None else self._rand_int(
            generator,
            self.config.downsample_factor_range[0],
            self.config.downsample_factor_range[1],
            x.device,
        )
        height, width = x.shape[-2:]
        down_height = max(2, height // max(factor_value, 1))
        down_width = max(2, width // max(factor_value, 1))
        down_height = min(down_height, height)
        down_width = min(down_width, width)

        down = F.interpolate(x, size=(down_height, down_width), mode=self.config.interpolation_mode, align_corners=False if self.config.interpolation_mode != "nearest" else None)
        up = F.interpolate(down, size=(height, width), mode=self.config.interpolation_mode, align_corners=False if self.config.interpolation_mode != "nearest" else None)
        return up, {
            "factor": int(factor_value),
            "down_height": int(down_height),
            "down_width": int(down_width),
            "interpolation_mode": self.config.interpolation_mode,
        }

    def fourier_lowpass(
        self,
        x: torch.Tensor,
        cutoff: float | None = None,
        rolloff: float = 0.05,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Suppress high spatial frequencies in the Fourier domain.

        Compared with direct spatial smoothing, this makes the low-pass intent
        explicit: preserve only the slowly varying background structure while
        attenuating sharp boundaries and thin features.
        """
        cutoff_value = cutoff if cutoff is not None else self._rand_uniform(
            generator,
            self.config.fourier_cutoff_range[0],
            self.config.fourier_cutoff_range[1],
            x.device,
        )
        height, width = x.shape[-2:]
        freq_x = torch.fft.fftfreq(height, device=x.device)
        freq_y = torch.fft.fftfreq(width, device=x.device)
        grid_x, grid_y = torch.meshgrid(freq_x, freq_y, indexing="ij")

        # Build a radial mask so attenuation depends only on frequency magnitude,
        # not on direction.
        radius = torch.sqrt(grid_x.square() + grid_y.square())
        mask = torch.sigmoid((cutoff_value - radius) / max(rolloff, 1e-6))

        # Normalize by the DC response so the zero-frequency component stays near
        # one instead of being slightly attenuated by the smooth sigmoid rolloff.
        dc = mask[0, 0].clamp_min(1e-6)
        mask = (mask / dc).clamp_max(1.0).to(dtype=x.dtype)

        spectrum = torch.fft.fft2(x, dim=(-2, -1))
        filtered = torch.fft.ifft2(spectrum * mask.view(1, 1, height, width), dim=(-2, -1)).real
        return filtered.to(dtype=x.dtype), {
            "cutoff": float(cutoff_value),
            "rolloff": float(rolloff),
        }

    def tomography_smoothing(
        self,
        x: torch.Tensor,
        sigma_x: float | None = None,
        sigma_y: float | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Apply anisotropic smoothing to mimic directionally coarse recovery.

        Different blur scales are used along the two spatial axes. This can
        emulate a background model that is much smoother in one direction than in
        the other, which is a common qualitative property of tomography-like
        reconstructions.
        """
        sigma_x_value = sigma_x if sigma_x is not None else self._rand_uniform(
            generator,
            self.config.tomography_sigma_x_range[0],
            self.config.tomography_sigma_x_range[1],
            x.device,
        )
        sigma_y_value = sigma_y if sigma_y is not None else self._rand_uniform(
            generator,
            self.config.tomography_sigma_y_range[0],
            self.config.tomography_sigma_y_range[1],
            x.device,
        )
        y, kernel_x, kernel_y = self._gaussian_blur(x, sigma_x=sigma_x_value, sigma_y=sigma_y_value)
        return y, {
            "sigma_x": float(sigma_x_value),
            "sigma_y": float(sigma_y_value),
            "kernel_x": kernel_x,
            "kernel_y": kernel_y,
        }

    def thin_layer_removal(
        self,
        x: torch.Tensor,
        kernel_size: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Average along the depth axis to suppress thin layered structure.

        The pooling kernel spans only one spatial axis, so the operator primarily
        removes fine layering while preserving more lateral variation.
        """
        kernel_value = kernel_size if kernel_size is not None else self._rand_int(
            generator,
            self.config.thin_layer_kernel_range[0],
            self.config.thin_layer_kernel_range[1],
            x.device,
        )
        kernel_x = self._valid_odd_kernel(kernel_value, x.shape[-2])
        pad_x = kernel_x // 2
        mode = self._padding_mode(x.shape[-2], pad_x)
        padded = F.pad(x, (0, 0, pad_x, pad_x), mode=mode)
        y = F.avg_pool2d(padded, kernel_size=(kernel_x, 1), stride=1)
        return y, {
            "kernel_size": int(kernel_x),
            "padding_mode": mode,
            "axis": "X",
        }

    def random_velocity_bias(
        self,
        x: torch.Tensor,
        strength: float | None = None,
        generator: torch.Generator | None = None,
        input_min: torch.Tensor | None = None,
        input_max: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Add smooth large-scale bias and mild perturbations to the model.

        This operator is useful when the coarse model should retain broad
        structure but exhibit systematic amplitude drift, gentle background warp,
        or low-frequency mismatch relative to the original velocity field.
        """
        strength_value = strength if strength is not None else self._rand_uniform(
            generator,
            self.config.bias_strength_range[0],
            self.config.bias_strength_range[1],
            x.device,
        )
        noise_strength = self._rand_uniform(
            generator,
            self.config.noise_strength_range[0],
            self.config.noise_strength_range[1],
            x.device,
        )
        if input_min is None or input_max is None:
            input_min, input_max = self._input_range(x)
        value_range = (input_max - input_min).clamp_min(1e-6)

        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        smooth_noise, kernel_x, kernel_y = self._gaussian_blur(noise, sigma_x=8.0, sigma_y=8.0)
        smooth_noise = smooth_noise / smooth_noise.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        bias = strength_value * value_range * smooth_noise

        # Combine a spatially smooth bias field with a small global scale and
        # shift so the degraded background can drift in a realistic large-scale
        # way instead of only being blurred.
        scale = 1.0 + self._rand_uniform(generator, -0.25 * strength_value, 0.25 * strength_value, x.device)
        shift = self._rand_uniform(generator, -0.25 * strength_value, 0.25 * strength_value, x.device) * value_range
        y = x * scale + shift + bias
        if noise_strength > 0.0:
            perturbation = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
            perturbation, perturb_kernel_x, perturb_kernel_y = self._gaussian_blur(perturbation, sigma_x=4.0, sigma_y=4.0)
            perturbation = perturbation / perturbation.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            y = y + noise_strength * value_range * perturbation
        else:
            perturb_kernel_x = 0
            perturb_kernel_y = 0
        if self.config.clamp_to_input_range:
            y = self._clamp_to_input_range(y, input_min, input_max)
        return y, {
            "strength": float(strength_value),
            "noise_strength": float(noise_strength),
            "scale": float(scale),
            "shift_factor": float((shift / value_range).mean().item()),
            "bias_kernel_x": kernel_x,
            "bias_kernel_y": kernel_y,
            "perturb_kernel_x": perturb_kernel_x,
            "perturb_kernel_y": perturb_kernel_y,
        }

    def horizon_warp(
        self,
        x: torch.Tensor,
        shift_frac: float | None = None,
        sigma: float | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Apply a smooth depth-dependent lateral warp.

        Each depth row is shifted left/right by a low-frequency offset so the
        coarse model exhibits mild horizon bending without breaking its broad
        background structure.
        """
        height, width = x.shape[-2:]
        shift_frac_value = shift_frac if shift_frac is not None else self._rand_uniform(
            generator,
            self.config.horizon_warp_shift_frac_range[0],
            self.config.horizon_warp_shift_frac_range[1],
            x.device,
        )
        sigma_value = sigma if sigma is not None else self._rand_uniform(
            generator,
            self.config.horizon_warp_sigma_range[0],
            self.config.horizon_warp_sigma_range[1],
            x.device,
        )
        max_shift_px = shift_frac_value * width
        offsets = torch.randn((x.shape[0], 1, height, 1), generator=generator, device=x.device, dtype=x.dtype)
        offsets, kernel_x, _ = self._gaussian_blur(offsets, sigma_x=sigma_value, sigma_y=1e-6)
        offsets = offsets / offsets.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        offsets_px = offsets * max_shift_px
        displacement_x = offsets_px.expand(-1, -1, -1, width)
        displacement_y = torch.zeros_like(displacement_x)
        y = self._apply_displacement_field(x, displacement_y=displacement_y, displacement_x=displacement_x)
        rms_shift_px = float(torch.sqrt((offsets_px.square()).mean()).item())
        return y, {
            "category": "warp",
            "display_name": "HWarp",
            "display_summary": f"HWarp {max_shift_px:.1f}px",
            "max_shift_px": float(max_shift_px),
            "rms_shift_px": rms_shift_px,
            "sigma": float(sigma_value),
            "kernel_x": kernel_x,
            "axis": "lateral_by_depth",
        }

    def depth_warp(
        self,
        x: torch.Tensor,
        shift_frac: float | None = None,
        sigma: float | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Apply a smooth laterally varying depth warp.

        The displacement stays low-frequency and small in magnitude so the coarse
        model looks like a mildly misregistered background rather than an
        arbitrarily deformed image.
        """
        height, width = x.shape[-2:]
        shift_frac_value = shift_frac if shift_frac is not None else self._rand_uniform(
            generator,
            self.config.depth_warp_shift_frac_range[0],
            self.config.depth_warp_shift_frac_range[1],
            x.device,
        )
        sigma_value = sigma if sigma is not None else self._rand_uniform(
            generator,
            self.config.depth_warp_sigma_range[0],
            self.config.depth_warp_sigma_range[1],
            x.device,
        )
        max_shift_px = shift_frac_value * height
        offsets = torch.randn((x.shape[0], 1, 1, width), generator=generator, device=x.device, dtype=x.dtype)
        offsets, _, kernel_y = self._gaussian_blur(offsets, sigma_x=1e-6, sigma_y=sigma_value)
        offsets = offsets / offsets.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        offsets_px = offsets * max_shift_px
        displacement_y = offsets_px.expand(-1, -1, height, -1)
        displacement_x = torch.zeros_like(displacement_y)
        y = self._apply_displacement_field(x, displacement_y=displacement_y, displacement_x=displacement_x)
        rms_shift_px = float(torch.sqrt((offsets_px.square()).mean()).item())
        return y, {
            "category": "warp",
            "display_name": "DWarp",
            "display_summary": f"DWarp {max_shift_px:.1f}px",
            "max_shift_px": float(max_shift_px),
            "rms_shift_px": rms_shift_px,
            "sigma": float(sigma_value),
            "kernel_y": kernel_y,
            "axis": "depth",
        }

    def mixed_degradation(
        self,
        x: torch.Tensor,
        generator: torch.Generator | None = None,
        input_min: torch.Tensor | None = None,
        input_max: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        """Compose several distinct degradations into one coarse-model pipeline.

        This mode samples 2--4 base operators without replacement and applies
        them sequentially. It is useful when a single blur or filtering operator
        is too simplistic and the coarse background should reflect a family of
        degradations instead.
        """
        if input_min is None or input_max is None:
            input_min, input_max = self._input_range(x)

        low, high = self.config.mixed_num_ops_range
        num_ops = self._rand_int(generator, low, min(high, len(self._base_methods)), x.device)
        selected_ops = self._sample_mixed_methods(num_ops, generator, x.device)

        y = x
        steps: list[dict[str, Any]] = []
        for op_name in selected_ops:
            y, params = self._apply_method(op_name, y, generator, input_min, input_max)
            if self.config.clamp_to_input_range and self.config.clamp_each_mixed_step:
                y = self._clamp_to_input_range(y, input_min, input_max)
            self._assert_finite(y, f"{op_name} output")
            steps.append(self._make_step_info(op_name, params))

        warp_name = self._sample_optional_warp(generator, x.device)
        if warp_name is not None:
            y, params = self._apply_method(warp_name, y, generator, input_min, input_max)
            if self.config.clamp_to_input_range:
                y = self._clamp_to_input_range(y, input_min, input_max)
            self._assert_finite(y, f"{warp_name} output")
            steps.append(self._make_step_info(warp_name, params))
        return y, steps

    def _sample_mixed_methods(
        self,
        num_ops: int,
        generator: torch.Generator | None,
        device: torch.device,
    ) -> list[str]:
        weights = torch.tensor(self.config.mixed_method_weights, device=device, dtype=torch.float32)
        if weights.numel() != len(self._base_methods):
            raise ValueError("mixed_method_weights must match the number of base methods.")
        weights = weights.clamp_min(0.0)
        if float(weights.sum().item()) <= 0.0:
            raise ValueError("mixed_method_weights must contain a positive total weight.")
        probabilities = weights / weights.sum()
        selected_indices = torch.multinomial(
            probabilities,
            num_samples=min(num_ops, len(self._base_methods)),
            replacement=False,
            generator=generator,
        ).tolist()
        selected_ops = [self._base_methods[index] for index in selected_indices]
        if self.config.mixed_require_severe_op:
            severe_ops = {"downup", "fourier", "tomography", "thin_layer"}
            if not any(op in severe_ops for op in selected_ops):
                fallback = max(
                    severe_ops,
                    key=lambda op: float(probabilities[self._base_methods.index(op)].item()),
                )
                selected_ops[0] = fallback
        return selected_ops

    def _sample_optional_warp(self, generator: torch.Generator | None, device: torch.device) -> str | None:
        probability = float(self.config.mixed_warp_probability)
        if probability <= 0.0:
            return None
        if self._rand_uniform(generator, 0.0, 1.0, device) >= probability:
            return None
        warp_methods = ("horizon_warp", "depth_warp")
        weights = torch.tensor(self.config.mixed_warp_method_weights, device=device, dtype=torch.float32)
        if weights.numel() != len(warp_methods):
            raise ValueError("mixed_warp_method_weights must match the number of warp methods.")
        weights = weights.clamp_min(0.0)
        if float(weights.sum().item()) <= 0.0:
            return None
        index = int(torch.multinomial(weights / weights.sum(), num_samples=1, replacement=False, generator=generator).item())
        return warp_methods[index]

    def _make_step_info(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        category = str(params.get("category", "coarsen" if method != "bias" else "bias"))
        display_summary = str(params.get("display_summary", method))
        return {
            "method": method,
            "params": params,
            "category": category,
            "display_summary": display_summary,
        }

    def _apply_method(
        self,
        method: str,
        x: torch.Tensor,
        generator: torch.Generator | None,
        input_min: torch.Tensor,
        input_max: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Dispatch to one concrete base degradation operator."""
        if method == "gaussian":
            return self.gaussian_smoothing(x, generator=generator)
        if method == "downup":
            return self.downsample_upsample(x, generator=generator)
        if method == "fourier":
            return self.fourier_lowpass(x, generator=generator)
        if method == "tomography":
            return self.tomography_smoothing(x, generator=generator)
        if method == "thin_layer":
            return self.thin_layer_removal(x, generator=generator)
        if method == "bias":
            return self.random_velocity_bias(x, generator=generator, input_min=input_min, input_max=input_max)
        if method == "horizon_warp":
            return self.horizon_warp(x, generator=generator)
        if method == "depth_warp":
            return self.depth_warp(x, generator=generator)
        raise ValueError(f"Unsupported method: {method}")

    def _to_b1xy_tensor(self, x: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Convert supported inputs into the internal ``[B, 1, X, Y]`` layout.

        The returned metadata records enough information to restore the original
        layout after degradation.
        """
        if isinstance(x, np.ndarray):
            tensor = torch.as_tensor(np.array(x, copy=True), dtype=torch.float32)
            original_type = "numpy"
            device = tensor.device
            original_dtype = str(x.dtype)
        elif isinstance(x, torch.Tensor):
            tensor = x.detach().clone().to(device=x.device, dtype=torch.float32)
            original_type = "torch"
            device = x.device
            original_dtype = str(x.dtype)
        else:
            raise TypeError("model must be a numpy.ndarray or torch.Tensor.")

        meta = {
            "original_type": original_type,
            "original_shape": tuple(tensor.shape),
            "original_ndim": tensor.ndim,
            "device": device,
            "dtype": original_dtype,
        }

        if tensor.ndim == 2:
            meta["restore_mode"] = "xy"
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            meta["restore_mode"] = "bxy"
            tensor = tensor.unsqueeze(1)
        elif tensor.ndim == 4:
            if tensor.shape[1] != 1:
                raise ValueError("4D input must have channel dimension equal to 1.")
            meta["restore_mode"] = "b1xy"
        else:
            raise ValueError("Input must have shape [X, Y], [B, X, Y], or [B, 1, X, Y].")

        if tensor.shape[-2] <= 0 or tensor.shape[-1] <= 0:
            raise ValueError("Spatial dimensions must be positive.")
        return tensor, meta

    def _restore_shape(self, y: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
        """Restore the degraded tensor to the caller-facing layout."""
        if y.ndim != 4 or y.shape[1] != 1:
            raise ValueError("Internal tensor must have shape [B, 1, X, Y].")
        restore_mode = meta["restore_mode"]
        if restore_mode == "xy":
            return y[0, 0]
        if restore_mode == "bxy":
            return y[:, 0]
        if restore_mode == "b1xy":
            return y
        raise ValueError(f"Unsupported restore mode: {restore_mode}")

    def _make_generator(self, seed: int | None, device: torch.device) -> torch.Generator:
        """Create a device-aware generator for reproducible random sampling."""
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        return generator

    def _rand_uniform(
        self,
        generator: torch.Generator | None,
        low: float,
        high: float,
        device: torch.device,
    ) -> float:
        """Sample a uniform float from ``[low, high]``."""
        if high < low:
            raise ValueError("high must be greater than or equal to low.")
        if math.isclose(low, high):
            return float(low)
        value = torch.rand((), generator=generator, device=device, dtype=torch.float32)
        return float((low + (high - low) * value).item())

    def _rand_int(
        self,
        generator: torch.Generator | None,
        low: int,
        high: int,
        device: torch.device,
    ) -> int:
        """Sample an integer from the closed interval ``[low, high]``."""
        if high < low:
            raise ValueError("high must be greater than or equal to low.")
        if high == low:
            return int(low)
        value = torch.randint(low, high + 1, (), generator=generator, device=device)
        return int(value.item())

    def _choose_random_method(self, generator: torch.Generator | None, device: torch.device) -> str:
        """Sample one base degradation method uniformly at random."""
        index = self._rand_int(generator, 0, len(self._base_methods) - 1, device)
        return self._base_methods[index]

    def _validate_method(self, method: str) -> None:
        """Validate the requested degradation mode."""
        if method not in self._supported_methods:
            supported = ", ".join(self._supported_methods)
            raise ValueError(f"Unsupported method '{method}'. Supported methods: {supported}.")

    def _assert_finite(self, x: torch.Tensor, name: str) -> None:
        """Ensure a tensor contains no NaN or Inf values."""
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} contains NaN or Inf.")

    def _input_range(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-sample min and max over spatial dimensions."""
        return x.amin(dim=(-2, -1), keepdim=True), x.amax(dim=(-2, -1), keepdim=True)

    def _clamp_to_input_range(self, x: torch.Tensor, input_min: torch.Tensor, input_max: torch.Tensor) -> torch.Tensor:
        """Clamp a degraded tensor to the original per-sample value range."""
        return torch.maximum(torch.minimum(x, input_max), input_min)

    def _valid_odd_kernel(self, kernel_size: int, dim_size: int) -> int:
        """Project a requested kernel size to a valid odd size for one axis."""
        kernel = int(kernel_size)
        if kernel % 2 == 0:
            kernel += 1
        if dim_size <= 1:
            return 1
        max_kernel = dim_size if dim_size % 2 == 1 else dim_size - 1
        max_kernel = max(max_kernel, 1)
        kernel = min(kernel, max_kernel)
        if kernel % 2 == 0:
            kernel = max(kernel - 1, 1)
        return kernel

    def _padding_mode(self, dim_size: int, pad: int) -> str:
        """Choose a safe padding mode for the requested axis and pad width."""
        if self.config.padding_mode == "reflect" and dim_size > 1 and pad < dim_size:
            return "reflect"
        return "replicate"

    def _apply_displacement_field(
        self,
        x: torch.Tensor,
        *,
        displacement_y: torch.Tensor,
        displacement_x: torch.Tensor,
    ) -> torch.Tensor:
        if displacement_y.shape != x.shape or displacement_x.shape != x.shape:
            raise ValueError("displacement fields must match the internal [B, 1, X, Y] tensor shape.")
        batch, _, height, width = x.shape
        base_y = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype)
        base_x = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(base_y, base_x, indexing="ij")
        base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        scale_x = 2.0 / max(width - 1, 1)
        scale_y = 2.0 / max(height - 1, 1)
        offset_x = displacement_x[:, 0] * scale_x
        offset_y = displacement_y[:, 0] * scale_y
        warped_grid = torch.stack((base_grid[..., 0] + offset_x, base_grid[..., 1] + offset_y), dim=-1)
        return F.grid_sample(x, warped_grid, mode="bilinear", padding_mode="border", align_corners=True)

    def _gaussian_blur(
        self,
        x: torch.Tensor,
        sigma_x: float,
        sigma_y: float,
    ) -> tuple[torch.Tensor, int, int]:
        """Apply separable Gaussian blur along the two spatial axes.

        The blur is implemented as two 1D convolutions, which keeps the code
        simple and makes anisotropic smoothing easy to express.
        """
        y = x
        kernel_x = self._gaussian_kernel1d(sigma_x, x.shape[-2], x.device, x.dtype)
        kernel_y = self._gaussian_kernel1d(sigma_y, x.shape[-1], x.device, x.dtype)

        if kernel_x.numel() > 1:
            pad_x = kernel_x.numel() // 2
            mode_x = self._padding_mode(x.shape[-2], pad_x)
            y = F.pad(y, (0, 0, pad_x, pad_x), mode=mode_x)
            y = F.conv2d(y, kernel_x.view(1, 1, -1, 1))
        if kernel_y.numel() > 1:
            pad_y = kernel_y.numel() // 2
            mode_y = self._padding_mode(x.shape[-1], pad_y)
            y = F.pad(y, (pad_y, pad_y, 0, 0), mode=mode_y)
            y = F.conv2d(y, kernel_y.view(1, 1, 1, -1))
        return y, int(kernel_x.numel()), int(kernel_y.numel())

    def _gaussian_kernel1d(
        self,
        sigma: float,
        dim_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Construct a normalized 1D Gaussian kernel for one spatial axis.

        The kernel radius is chosen from approximately ``3 * sigma`` on each
        side, then clipped to a valid odd length for the current dimension.
        """
        sigma = float(max(sigma, 1e-6))
        desired_kernel = int(2 * math.ceil(3.0 * sigma) + 1)
        kernel_size = self._valid_odd_kernel(desired_kernel, dim_size)
        if kernel_size <= 1:
            return torch.ones(1, device=device, dtype=dtype)
        radius = kernel_size // 2
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel = kernel / kernel.sum().clamp_min(1e-6)
        return kernel



def make_coarse_model(
    model: np.ndarray | torch.Tensor,
    method: str = "mixed",
    config: CoarseConfig | None = None,
    seed: int | None = None,
    return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Convenience wrapper around :class:`CoarseVelocityGenerator`.

    Example:
        >>> model = np.random.randn(128, 128).astype(np.float32)
        >>> coarse = make_coarse_model(model, method="fourier")
        >>> coarse.shape
        torch.Size([128, 128])
    """
    generator = CoarseVelocityGenerator(config=config)
    return generator(model, method=method, seed=seed, return_info=return_info)


__all__ = ["CoarseConfig", "CoarseProfileName", "CoarseVelocityGenerator", "coarse_config_for_profile", "make_coarse_model"]


