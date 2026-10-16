"""Published ABA-FWI training objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor


def velocity_edges(target: Tensor) -> Tensor:
    """Create the official ABA-FWI Canny edge mask for each target."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Official ABA-FWI RCTB edges require opencv-python-headless in the pyx environment."
        ) from exc
    masks = []
    for image in target.detach().float().cpu().numpy():
        velocity = image[0]
        normalized = cv2.normalize(
            velocity,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
        canny = cv2.Canny((normalized * 255).astype(np.uint8), 10, 15)
        masks.append(np.clip(canny, 0, 1)[None])
    return torch.from_numpy(np.stack(masks)).to(device=target.device, dtype=target.dtype)


def _dilate(mask: Tensor) -> Tensor:
    return (F.max_pool2d(mask, 3, stride=1, padding=1) > 0).to(mask.dtype)


def rctb_loss(prediction: Tensor, target: Tensor, edges: Tensor | None = None) -> Tensor:
    """Reflection-coefficient-tuned boundary loss from ABA-FWI."""
    if edges is None:
        edges = velocity_edges(target)
    pred_x = F.pad(prediction[:, :, 1:, :] - prediction[:, :, :-1, :], (0, 0, 1, 0))
    pred_y = F.pad(prediction[:, :, :, 1:] - prediction[:, :, :, :-1], (1, 0, 0, 0))
    target_x = F.pad(target[:, :, 1:, :] - target[:, :, :-1, :], (0, 0, 1, 0))
    target_y = F.pad(target[:, :, :, 1:] - target[:, :, :, :-1], (1, 0, 0, 0))
    variation = (pred_x - target_x).abs() + (pred_y - target_y).abs()

    denominator = target[:, :, 1:, :] + target[:, :, :-1, :]
    reflection = ((target[:, :, 1:, :] - target[:, :, :-1, :]) / denominator).nan_to_num().abs()
    reflection = F.pad(reflection, (0, 0, 1, 0))
    edge_region = _dilate(edges)
    local_reflection = F.max_pool2d(reflection, 3, stride=1, padding=1) * edge_region
    weights = torch.where(
        (local_reflection != 0) & (local_reflection < 0.05),
        torch.full_like(local_reflection, 2.0),
        torch.ones_like(local_reflection),
    ) * edge_region
    return (variation * weights).sum() / (target.shape[0] * edge_region.sum().clamp_min(1.0))


def aba_fwi_loss(
    prediction: Tensor, target: Tensor, edges: Tensor | None = None
) -> tuple[Tensor, dict[str, Tensor]]:
    prediction = prediction.float()
    target = target.float()
    l1 = F.l1_loss(prediction, target)
    mse = F.mse_loss(prediction, target)
    rctb = rctb_loss(prediction, target, edges)
    return l1 + mse + rctb, {"l1": l1, "mse": mse, "rctb": rctb}


__all__ = ["aba_fwi_loss", "rctb_loss", "velocity_edges"]
