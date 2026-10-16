"""Metrics for normalized DL-FWI velocity predictions."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

METRIC_MIN = 0.0
METRIC_MAX = 1.0
METRIC_DATA_RANGE = METRIC_MAX - METRIC_MIN
PHYSICAL_MIN = 1000.0
PHYSICAL_MAX = 6000.0
PHYSICAL_DATA_RANGE = PHYSICAL_MAX - PHYSICAL_MIN

_lpips_models: dict[str, nn.Module] = {}


def _validate_inputs(prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    prediction = prediction.detach().float()
    target = target.detach().float()
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have the same shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}."
        )
    if prediction.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W] tensors, got {tuple(prediction.shape)}.")
    if prediction.shape[0] == 0:
        raise ValueError("Metric inputs must contain at least one sample.")
    return prediction, target


def _clamp_metric_inputs(prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    prediction, target = _validate_inputs(prediction, target)
    return prediction.clamp(METRIC_MIN, METRIC_MAX), target


def _boundary_mask(target: Tensor, kernel_size: int = 3) -> Tensor:
    if target.shape[1] != 1:
        raise ValueError("Boundary metrics require single-channel velocity models.")

    batch, channels = target.shape[:2]
    flat = target.reshape(batch, channels, -1)
    target_min = flat.amin(dim=2, keepdim=True).view(batch, channels, 1, 1)
    target_max = flat.amax(dim=2, keepdim=True).view(batch, channels, 1, 1)
    target_range = (target_max - target_min).clamp_min(1e-8)
    normalized = (target - target_min) / target_range

    sobel_x = normalized.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    sobel_y = normalized.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    padded = F.pad(normalized, (1, 1, 1, 1), mode="replicate")
    magnitude = torch.sqrt(
        F.conv2d(padded, sobel_x).square()
        + F.conv2d(padded, sobel_y).square()
    )
    max_gradient = magnitude.flatten(1).amax(dim=1).view(batch, 1, 1, 1)
    edges = (magnitude >= 0.1 * max_gradient) & (max_gradient >= 1e-8)
    padding = kernel_size // 2
    return F.max_pool2d(edges.to(target.dtype), kernel_size, stride=1, padding=padding)


def _boundary_errors(prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    mask = _boundary_mask(target)
    error = (prediction - target) * mask
    dimensions = tuple(range(1, error.ndim))
    mask_pixels = mask.sum(dim=dimensions).clamp_min(1.0)
    return error.abs().sum(dim=dimensions) / mask_pixels, error.square().sum(dim=dimensions) / mask_pixels


def _core_metrics_per_sample(prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
    prediction, target = _clamp_metric_inputs(prediction, target)
    normalized_error = prediction - target
    physical_error = normalized_error * PHYSICAL_DATA_RANGE
    dimensions = tuple(range(1, normalized_error.ndim))
    physical_mse = physical_error.square().mean(dim=dimensions)
    error_norm = normalized_error.flatten(1).norm(dim=1)
    target_norm = target.flatten(1).norm(dim=1)
    bmae, bmse = _boundary_errors(prediction, target)
    return {
        "mae": physical_error.abs().mean(dim=dimensions),
        "mse": physical_mse,
        "rmse": physical_mse.sqrt(),
        "relative_l2": error_norm / target_norm.clamp_min(1e-8),
        "bmae": bmae * PHYSICAL_DATA_RANGE,
        "bmse": bmse * PHYSICAL_DATA_RANGE**2,
    }


def _uqi_per_sample(prediction: Tensor, target: Tensor) -> Tensor:
    x = prediction.flatten(1)
    y = target.flatten(1)
    mean_x = x.mean(dim=1)
    mean_y = y.mean(dim=1)
    centered_x = x - mean_x[:, None]
    centered_y = y - mean_y[:, None]
    variance_x = centered_x.square().mean(dim=1)
    variance_y = centered_y.square().mean(dim=1)
    covariance = (centered_x * centered_y).mean(dim=1)
    numerator = 4.0 * mean_x * mean_y * covariance
    denominator = (mean_x.square() + mean_y.square()) * (variance_x + variance_y)

    both_constant_and_equal = (denominator.abs() < 1e-8) & (
        (prediction - target).flatten(1).abs().amax(dim=1) < 1e-8
    )
    values = numerator / denominator.clamp_min(1e-8)
    return torch.where(both_constant_and_equal, torch.ones_like(values), values)


def _load_piq_metrics() -> tuple[Callable[..., Tensor], Callable[..., Tensor], type[nn.Module]]:
    try:
        from piq import LPIPS, psnr, ssim
    except ImportError as exc:
        raise RuntimeError(
            "Full FWI evaluation requires piq. Install the paper experiment dependencies "
            "with `uv sync`."
        ) from exc
    return ssim, psnr, LPIPS


def _lpips_per_sample(prediction: Tensor, target: Tensor) -> Tensor:
    _, _, lpips_type = _load_piq_metrics()
    device_key = str(prediction.device)
    model = _lpips_models.get(device_key)
    if model is None:
        # PIQ otherwise downloads both LPIPS and VGG16 weights implicitly.
        # Offline extraction must never invent learned weights or fetch them.
        from pathlib import Path
        from urllib.parse import urlparse
        from torchvision.models import VGG16_Weights
        cache = Path(torch.hub.get_dir()) / "checkpoints"
        required = [Path(urlparse(url).path).name for url in
                    (lpips_type._weights_url, VGG16_Weights.IMAGENET1K_V1.url)]
        missing = [name for name in required if not (cache / name).is_file()]
        if missing:
            raise RuntimeError(f"Full metrics require locally cached pretrained weights in {cache}: {missing}; no download attempted")
        model = lpips_type(reduction="none").eval().to(prediction.device)
        _lpips_models[device_key] = model
    prediction_rgb = prediction.repeat(1, 3, 1, 1) if prediction.shape[1] == 1 else prediction
    target_rgb = target.repeat(1, 3, 1, 1) if target.shape[1] == 1 else target
    return model(prediction_rgb, target_rgb).reshape(prediction.shape[0], -1).mean(dim=1)


def compute_regression_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    """Compute sample-averaged core metrics without requiring PIQ."""
    values = _core_metrics_per_sample(prediction, target)
    return {name: float(per_sample.mean().item()) for name, per_sample in values.items()}


@torch.no_grad()
def compute_full_metrics(prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
    """Compute physical error metrics and normalized image-quality metrics."""
    prediction, target = _clamp_metric_inputs(prediction, target)
    ssim, psnr, _ = _load_piq_metrics()
    values = _core_metrics_per_sample(prediction, target)
    values.update({
        "ssim": ssim(prediction, target, data_range=METRIC_DATA_RANGE, reduction="none").reshape(-1),
        "psnr": psnr(prediction, target, data_range=METRIC_DATA_RANGE, reduction="none").reshape(-1),
        "uqi": _uqi_per_sample(prediction, target),
        "lpips": _lpips_per_sample(prediction, target),
    })
    return values


class MetricAccumulator:
    """Accumulate sample-weighted lightweight or full evaluation metrics."""

    def __init__(self, *, full: bool) -> None:
        self.full = full
        self._sums: dict[str, float] = {}
        self._count = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        prediction, target = _validate_inputs(prediction, target)
        batch_size = prediction.shape[0]
        if self.full:
            values = compute_full_metrics(prediction, target)
            for name, per_sample in values.items():
                self._sums[name] = self._sums.get(name, 0.0) + float(per_sample.sum().item())
        else:
            values = compute_regression_metrics(prediction, target)
            for name, value in values.items():
                self._sums[name] = self._sums.get(name, 0.0) + value * batch_size
        self._count += batch_size

    def compute(self) -> dict[str, float]:
        if self._count == 0:
            return {}
        return {name: total / self._count for name, total in self._sums.items()}


def aggregate_metric_dicts(metrics: list[dict[str, float]]) -> dict[str, float]:
    """Retained for callers that already provide equally weighted metric dictionaries."""
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {key: sum(item[key] for item in metrics) / len(metrics) for key in keys}
