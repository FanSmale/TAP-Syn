"""方案 1：面向粗背景一致性与高频生成性的统一损失。

本文件记录当前项目的第一套正式损失草案，并将其实现为一个可直接接入
训练阶段的统一入口。这里的重点不是让不同阶段使用完全不同的目标，而是
保持一套统一骨架：

1. 用 Rectified Flow 主损失维持从 coarse background condition 到 residual 的主路径监督；
2. 用低频一致性约束固定条件语义，避免同一个 ``m_c_cond`` 下出现大尺度背景漂移；
3. 用高频结构分布约束与频谱约束鼓励结果具有清晰地层，而不是退化成模糊平均图；
4. 用 batch 高频分布约束避免生成结果整体退化成低多样性的平均结构；
5. 只有当 ``B_psi`` 参与时，才额外开启 coarse 监督项。

这里的 canonical 语义是：物理 coarse background ``m_c`` 先经过全局
速度范围归一化得到 ``m_c_cond``，conditional 与 unconditional 路径都
只围绕这个 ``m_c_cond`` 工作。

为什么不再使用“全频逐像素 GT 对齐”作为核心目标：

- 当前任务是生成任务，而不是给定 ``m_c_cond`` 后恢复唯一真值；
- 对全频结果做强逐像素对齐会把高频细节压回平均解，削弱生成多样性；
- 低频部分是条件语义的主要载体，因此可以强约束；
- 中高频部分应主要约束“像真的、层理清晰、统计合理”，而不是“与这个 GT 的每个像素一致”。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import DEFAULT_NORMALIZATION_CONFIG
from ..training.config import LossConfig


class LossScheme1(nn.Module):
    """统一实现方案 1 的各分项损失与总损失。"""

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        if config.name != "scheme1":
            raise ValueError(f"Unsupported loss scheme: {config.name}.")
        self.config = config

    @property
    def uses_diversity(self) -> bool:
        return self.config.use_diversity and self.config.lambda_div > 0.0

    def _charbonnier(self, diff: Tensor) -> Tensor:
        return torch.sqrt(diff.square() + self.config.charbonnier_eps**2).mean()

    def _gaussian_kernel(self, device: torch.device, dtype: torch.dtype, channels: int) -> Tensor:
        radius = self.config.low_pass_kernel_size // 2
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel_1d = torch.exp(-(coords.square()) / (2.0 * self.config.low_pass_sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d / kernel_2d.sum()
        return kernel_2d.view(1, 1, *kernel_2d.shape).repeat(channels, 1, 1, 1)

    def low_pass(self, tensor: Tensor) -> Tensor:
        channels = tensor.shape[1]
        radius = self.config.low_pass_kernel_size // 2
        kernel = self._gaussian_kernel(tensor.device, tensor.dtype, channels)
        padded = F.pad(tensor, (radius, radius, radius, radius), mode="reflect")
        return F.conv2d(padded, kernel, groups=channels)

    def high_pass(self, tensor: Tensor) -> Tensor:
        return tensor - self.low_pass(tensor)

    def _gradient_feature(self, tensor: Tensor) -> Tensor:
        feature_input = tensor.float()
        high_freq = self.high_pass(feature_input)
        grad_y = torch.diff(high_freq, dim=-2)
        grad_y = F.pad(grad_y.abs(), (0, 0, 0, 1))
        grad_x = torch.diff(high_freq, dim=-1)
        grad_x = F.pad(grad_x.abs(), (0, 1, 0, 0))
        gradient_mag = torch.sqrt(grad_x.square() + grad_y.square() + self.config.charbonnier_eps**2)
        feature = torch.cat([grad_x, grad_y, gradient_mag], dim=1)
        if self.config.grad_pool_kernel > 1:
            feature = F.avg_pool2d(feature, kernel_size=self.config.grad_pool_kernel, stride=self.config.grad_pool_kernel)
        return feature.flatten(start_dim=1)

    def _rbf_mmd(self, x: Tensor, y: Tensor) -> Tensor:
        x32 = x.float()
        y32 = y.float()
        if x32.shape[0] == 1 and y32.shape[0] == 1:
            return self._charbonnier(x32 - y32)
        gamma = 1.0 / max(x32.shape[-1], 1)
        xx = torch.cdist(x32, x32, p=2).square()
        yy = torch.cdist(y32, y32, p=2).square()
        xy = torch.cdist(x32, y32, p=2).square()
        k_xx = torch.exp(-gamma * xx)
        k_yy = torch.exp(-gamma * yy)
        k_xy = torch.exp(-gamma * xy)
        return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()

    def _pooled_log_power_spectrum(self, tensor: Tensor) -> Tensor:
        high_freq = self.high_pass(tensor.float())
        spectrum = torch.fft.rfft2(high_freq, norm="ortho")
        power = spectrum.abs().square()
        log_power = torch.log(power + self.config.psd_eps)
        return F.adaptive_avg_pool2d(log_power, output_size=(self.config.psd_pool_size, self.config.psd_pool_size))

    def _quantile(self, tensor: Tensor, q: float) -> Tensor:
        flattened = tensor.float().flatten(start_dim=1)
        return torch.quantile(flattened, q, dim=1)

    def _coarse_stat_summary(self, tensor: Tensor) -> dict[str, Tensor]:
        flattened = tensor.float().flatten(start_dim=1)
        q_low = self._quantile(tensor, self.config.coarse_range_q_low)
        q_high = self._quantile(tensor, self.config.coarse_range_q_high)
        return {
            "mean": flattened.mean(dim=1),
            "std": flattened.std(dim=1, unbiased=False),
            "q_low": q_low,
            "q_high": q_high,
            "range": q_high - q_low,
        }

    def _coarse_profile_feature(self, tensor: Tensor) -> Tensor:
        low_freq = self.low_pass(tensor.float())
        row_mean = low_freq.mean(dim=-1)
        row_std = low_freq.std(dim=-1, unbiased=False)
        row_grad = torch.diff(row_mean, dim=-1)
        row_grad = F.pad(row_grad, (0, 1))
        return torch.cat([row_mean.flatten(start_dim=1), row_std.flatten(start_dim=1), row_grad.flatten(start_dim=1)], dim=1)

    def _coarse_pyramid_feature(self, tensor: Tensor) -> Tensor:
        low_freq = self.low_pass(tensor.float())
        levels = [
            F.adaptive_avg_pool2d(low_freq, output_size=(size, size)).flatten(start_dim=1)
            for size in self.config.coarse_pyramid_sizes
        ]
        return torch.cat(levels, dim=1)

    def _sorted_batch_loss(self, generated: Tensor, reference: Tensor) -> Tensor:
        """Compare unpaired batch distributions without assigning random targets."""
        if generated.shape[0] != reference.shape[0]:
            raise ValueError("Generated and reference coarse batches must have equal sizes.")
        if generated.shape[0] < 2:
            return self._charbonnier(generated - reference)
        generated_sorted = generated.float().sort(dim=0).values
        reference_sorted = reference.float().sort(dim=0).values
        return self._charbonnier(generated_sorted - reference_sorted)

    def _coarse_pairwise_loss(self, generated: Tensor, reference: Tensor) -> Tensor:
        if generated.shape[0] < 2:
            return generated.new_zeros(())
        generated_feature = self._coarse_pyramid_feature(generated)
        reference_feature = self._coarse_pyramid_feature(reference)
        generated_feature = (generated_feature - generated_feature.mean(dim=1, keepdim=True)) / generated_feature.std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(1e-6)
        reference_feature = (reference_feature - reference_feature.mean(dim=1, keepdim=True)) / reference_feature.std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(1e-6)
        generated_distances = torch.pdist(generated_feature) / generated_feature.shape[1] ** 0.5
        reference_distances = torch.pdist(reference_feature) / reference_feature.shape[1] ** 0.5
        return self._charbonnier(generated_distances.sort().values - reference_distances.sort().values)

    def estimate_residual(self, flow_details: dict[str, Tensor]) -> Tensor:
        t = flow_details["t"].view(-1, 1, 1, 1)
        return flow_details["state_t"] + (1.0 - t) * flow_details["pred"]

    def reconstruct_velocity(self, m_c_cond: Tensor, residual: Tensor) -> Tensor:
        return (m_c_cond + residual).clamp(
            DEFAULT_NORMALIZATION_CONFIG.normalized_min,
            DEFAULT_NORMALIZATION_CONFIG.normalized_max,
        )

    def compute_flow_loss(self, flow_details: dict[str, Tensor]) -> Tensor:
        return self._charbonnier(flow_details["pred"] - flow_details["target"])

    def compute_low_loss(self, reconstruction: Tensor, m_c_cond: Tensor) -> Tensor:
        return self._charbonnier(self.low_pass(reconstruction) - self.low_pass(m_c_cond))

    def compute_grad_dist_loss(self, reconstruction: Tensor, velocity_norm: Tensor) -> Tensor:
        if not self.config.use_grad_dist or self.config.lambda_grad <= 0.0:
            return reconstruction.new_zeros(())
        grad_pred = self._gradient_feature(reconstruction)
        grad_target = self._gradient_feature(velocity_norm)
        return self._rbf_mmd(grad_pred, grad_target)

    def compute_psd_loss(self, reconstruction: Tensor, velocity_norm: Tensor) -> Tensor:
        if not self.config.use_psd or self.config.lambda_psd <= 0.0:
            return reconstruction.new_zeros(())
        psd_pred = self._pooled_log_power_spectrum(reconstruction)
        psd_target = self._pooled_log_power_spectrum(velocity_norm)
        return self._charbonnier(psd_pred - psd_target)

    def compute_diversity_loss(self, reconstruction: Tensor, velocity_norm: Tensor) -> Tensor:
        if not self.uses_diversity:
            return reconstruction.new_zeros(())
        if reconstruction.shape != velocity_norm.shape:
            raise ValueError("Reconstruction and target velocity must have the same shape.")
        if reconstruction.shape[0] < 2:
            return reconstruction.sum() * 0.0

        pool_size = self.config.diversity_pool_size
        generated_feature = F.adaptive_avg_pool2d(
            self.high_pass(reconstruction.float()), (pool_size, pool_size)
        )
        target_feature = F.adaptive_avg_pool2d(
            self.high_pass(velocity_norm.detach().float()), (pool_size, pool_size)
        )
        generated_sorted = generated_feature.sort(dim=0).values
        target_sorted = target_feature.sort(dim=0).values
        difference = generated_sorted - target_sorted
        eps = self.config.charbonnier_eps
        return (torch.sqrt(difference.square() + eps**2) - eps).mean()

    @torch.no_grad()
    def diversity_diagnostics(self, reconstruction: Tensor, velocity_norm: Tensor) -> dict[str, float]:
        pool_size = self.config.diversity_pool_size
        generated = F.adaptive_avg_pool2d(
            self.high_pass(reconstruction.float()), (pool_size, pool_size)
        ).flatten(start_dim=1)
        target = F.adaptive_avg_pool2d(
            self.high_pass(velocity_norm.float()), (pool_size, pool_size)
        ).flatten(start_dim=1)
        generated_std = generated.std(dim=0, unbiased=False).mean()
        target_std = target.std(dim=0, unbiased=False).mean()
        return {
            "endpoint_mae": float(F.l1_loss(reconstruction.float(), velocity_norm.float()).item()),
            "diversity_generated_batch_std": float(generated_std.item()),
            "diversity_target_batch_std": float(target_std.item()),
            "diversity_batch_std_ratio": float((generated_std / target_std.clamp_min(1e-8)).item()),
        }

    def compute_coarse_loss(self, generated_m_c_cond: Tensor, reference_m_c_cond: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if self.config.lambda_coarse <= 0.0:
            zero = generated_m_c_cond.new_zeros(())
            return zero, {
                "coarse_low_loss": zero,
                "coarse_grad_loss": zero,
                "coarse_psd_loss": zero,
                "coarse_profile_loss": zero,
                "coarse_pyramid_loss": zero,
                "coarse_mean_loss": zero,
                "coarse_std_loss": zero,
                "coarse_range_loss": zero,
                "coarse_pairwise_loss": zero,
            }
        if self.config.coarse_alignment_mode == "distribution" and generated_m_c_cond.shape[0] < 2:
            zero = generated_m_c_cond.new_zeros(())
            return zero, {
                "coarse_low_loss": zero,
                "coarse_grad_loss": zero,
                "coarse_psd_loss": zero,
                "coarse_profile_loss": zero,
                "coarse_pyramid_loss": zero,
                "coarse_mean_loss": zero,
                "coarse_std_loss": zero,
                "coarse_range_loss": zero,
                "coarse_pairwise_loss": zero,
            }
        if self.config.coarse_alignment_mode == "distribution":
            pool_size = self.config.coarse_distribution_pool_size
            generated_low = F.adaptive_avg_pool2d(self.low_pass(generated_m_c_cond), pool_size)
            reference_low = F.adaptive_avg_pool2d(self.low_pass(reference_m_c_cond), pool_size)
            low_loss = self._sorted_batch_loss(generated_low, reference_low)
        elif self.config.coarse_alignment_mode == "paired":
            low_loss = self._charbonnier(self.low_pass(generated_m_c_cond) - self.low_pass(reference_m_c_cond))
        else:
            raise ValueError(f"Unsupported coarse alignment mode: {self.config.coarse_alignment_mode}.")
        grad_loss = self._rbf_mmd(self._gradient_feature(generated_m_c_cond), self._gradient_feature(reference_m_c_cond))
        generated_psd = self._pooled_log_power_spectrum(generated_m_c_cond)
        reference_psd = self._pooled_log_power_spectrum(reference_m_c_cond)
        psd_loss = (
            self._sorted_batch_loss(generated_psd, reference_psd)
            if self.config.coarse_alignment_mode == "distribution"
            else self._charbonnier(generated_psd - reference_psd)
        )
        profile_loss = self._rbf_mmd(
            self._coarse_profile_feature(generated_m_c_cond),
            self._coarse_profile_feature(reference_m_c_cond),
        )
        pyramid_loss = self._rbf_mmd(
            self._coarse_pyramid_feature(generated_m_c_cond),
            self._coarse_pyramid_feature(reference_m_c_cond),
        )
        generated_stats = self._coarse_stat_summary(generated_m_c_cond)
        reference_stats = self._coarse_stat_summary(reference_m_c_cond)
        if self.config.coarse_alignment_mode == "distribution":
            mean_loss = self._sorted_batch_loss(generated_stats["mean"], reference_stats["mean"])
            std_loss = self._sorted_batch_loss(generated_stats["std"], reference_stats["std"])
            range_loss = self._sorted_batch_loss(generated_stats["range"], reference_stats["range"])
        else:
            mean_loss = self._charbonnier(generated_stats["mean"] - reference_stats["mean"])
            std_loss = self._charbonnier(generated_stats["std"] - reference_stats["std"])
            range_loss = self._charbonnier(generated_stats["range"] - reference_stats["range"])
        pairwise_loss = (
            self._coarse_pairwise_loss(generated_m_c_cond, reference_m_c_cond)
            if self.config.coarse_alignment_mode == "distribution"
            else generated_m_c_cond.new_zeros(())
        )
        total = (
            low_loss
            + 0.35 * grad_loss
            + 0.15 * psd_loss
            + self.config.lambda_coarse_profile * profile_loss
            + self.config.lambda_coarse_pyramid * pyramid_loss
            + self.config.lambda_coarse_mean * mean_loss
            + self.config.lambda_coarse_std * std_loss
            + self.config.lambda_coarse_range * range_loss
            + self.config.lambda_coarse_pairwise * pairwise_loss
        )
        return total, {
            "coarse_low_loss": low_loss,
            "coarse_grad_loss": grad_loss,
            "coarse_psd_loss": psd_loss,
            "coarse_profile_loss": profile_loss,
            "coarse_pyramid_loss": pyramid_loss,
            "coarse_mean_loss": mean_loss,
            "coarse_std_loss": std_loss,
            "coarse_range_loss": range_loss,
            "coarse_pairwise_loss": pairwise_loss,
        }

    def compute_total_loss(
        self,
        *,
        velocity_norm: Tensor,
        m_c_cond: Tensor,
        flow_details: dict[str, Tensor],
        include_coarse: bool = False,
        coarse_reference_cond: Tensor | None = None,
        unconditional_path_weight: float = 1.0,
        coarse_weight: float = 1.0,
    ) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
        residual = self.estimate_residual(flow_details)
        reconstruction_unclamped = m_c_cond + residual
        reconstruction = self.reconstruct_velocity(m_c_cond, residual)

        flow_loss = self.compute_flow_loss(flow_details)
        low_loss = self.compute_low_loss(reconstruction, m_c_cond)
        grad_loss = self.compute_grad_dist_loss(reconstruction, velocity_norm)
        psd_loss = self.compute_psd_loss(reconstruction, velocity_norm)

        diversity_loss = self.compute_diversity_loss(reconstruction_unclamped, velocity_norm)

        shared_loss = (
            self.config.lambda_flow * flow_loss
            + self.config.lambda_low * low_loss
            + self.config.lambda_grad * grad_loss
            + self.config.lambda_psd * psd_loss
            + self.config.lambda_div * diversity_loss
        )
        total_loss = unconditional_path_weight * shared_loss

        coarse_loss = reconstruction.new_zeros(())
        coarse_terms = {
            "coarse_low_loss": reconstruction.new_zeros(()),
            "coarse_grad_loss": reconstruction.new_zeros(()),
            "coarse_psd_loss": reconstruction.new_zeros(()),
            "coarse_profile_loss": reconstruction.new_zeros(()),
            "coarse_pyramid_loss": reconstruction.new_zeros(()),
            "coarse_mean_loss": reconstruction.new_zeros(()),
            "coarse_std_loss": reconstruction.new_zeros(()),
            "coarse_range_loss": reconstruction.new_zeros(()),
            "coarse_pairwise_loss": reconstruction.new_zeros(()),
        }
        if include_coarse:
            if coarse_reference_cond is None:
                raise ValueError("coarse_reference_cond is required when include_coarse=True.")
            coarse_loss, coarse_terms = self.compute_coarse_loss(m_c_cond, coarse_reference_cond)
            total_loss = total_loss + coarse_weight * self.config.lambda_coarse * coarse_loss

        metrics = {
            "total_loss": float(total_loss.detach().item()),
            "flow_loss": float(flow_loss.detach().item()),
            "low_loss": float(low_loss.detach().item()),
            "grad_dist_loss": float(grad_loss.detach().item()),
            "psd_loss": float(psd_loss.detach().item()),
            "diversity_loss": float(diversity_loss.detach().item()),
            "coarse_loss": float(coarse_loss.detach().item()),
            **self.diversity_diagnostics(reconstruction_unclamped, velocity_norm),
            **{name: float(value.detach().item()) for name, value in coarse_terms.items()},
        }
        outputs: dict[str, Tensor] = {
            "residual": residual,
            "reconstruction_unclamped": reconstruction_unclamped,
            "reconstruction": reconstruction,
            "reconstruction_norm": reconstruction,
        }
        return total_loss, metrics, outputs

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "lambda_flow": self.config.lambda_flow,
            "lambda_low": self.config.lambda_low,
            "lambda_grad": self.config.lambda_grad,
            "lambda_psd": self.config.lambda_psd,
            "lambda_div": self.config.lambda_div,
            "lambda_coarse": self.config.lambda_coarse,
            "lambda_coarse_mean": self.config.lambda_coarse_mean,
            "lambda_coarse_std": self.config.lambda_coarse_std,
            "lambda_coarse_range": self.config.lambda_coarse_range,
            "lambda_coarse_profile": self.config.lambda_coarse_profile,
            "lambda_coarse_pyramid": self.config.lambda_coarse_pyramid,
            "lambda_coarse_pairwise": self.config.lambda_coarse_pairwise,
            "coarse_alignment_mode": self.config.coarse_alignment_mode,
            "use_grad_dist": self.config.use_grad_dist,
            "use_psd": self.config.use_psd,
            "use_diversity": self.config.use_diversity,
            "diversity_pool_size": self.config.diversity_pool_size,
        }


__all__ = ["LossScheme1"]
