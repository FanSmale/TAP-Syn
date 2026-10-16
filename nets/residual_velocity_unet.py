"""Rectified-Flow-oriented residual U-Net for normalized 128x128 velocity residual models.

This module keeps the shared conditioned U-Net implementation used by the
residual-generation methods, while exposing ``ResidualVelocityUNet`` as the
Rectified Flow-facing public network class. The architecture remains
specialized to the current research setup:
- ``state_t`` and globally normalized coarse condition ``m_c_cond`` are concatenated, so the input has 2 channels
- the model predicts a single residual-shaped output channel
- the spatial size is fixed to ``128 x 128``
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn



def _group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """Create a GroupNorm layer with a group count that divides the channels.

    The helper backs off from ``max_groups`` until it finds a valid divisor so
    callers do not need to reason about special cases for every channel count.
    """
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class SinusoidalTimeEmbedding(nn.Module):
    """Create sinusoidal embeddings for discrete or continuous time values."""

    def __init__(self, dim: int) -> None:
        """Store the output embedding dimension."""
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """Embed a batch of time values into a sinusoidal feature space.

        Args:
            t: Continuous Rectified Flow time values with shape ``[B]`` in ``[0, 1]``.

        Returns:
            Tensor with shape ``[B, dim]``.
        """
        if t.ndim != 1:
            raise ValueError("Expected time tensor with shape [B].")

        half_dim = self.dim // 2
        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        freq = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * exponent)
        args = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ConditionMLP(nn.Module):
    """Project a condition vector into the shared conditioning space."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """Build a small two-layer MLP for condition projection."""
        super().__init__()
        hidden_dim = out_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project one batch of condition vectors."""
        return self.net(x)


class PlainResBlock(nn.Module):
    """Residual block without external conditioning."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Construct a plain residual block."""
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = _group_norm(out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual block and add the shortcut branch."""
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.shortcut(x)


class ResBlock(nn.Module):
    """FiLM-conditioned residual block."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
        """Construct a residual block conditioned by a global embedding."""
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = _group_norm(out_channels)
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Apply a residual block modulated by a shared conditioning vector."""
        h = self.conv1(self.act1(self.norm1(x)))
        scale, shift = self.cond_proj(cond).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act2(h))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention on 2D feature maps."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        """Configure channel count and number of attention heads."""
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads.")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = _group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Attend globally over all spatial positions in one feature map."""
        batch, channels, height, width = x.shape
        h = self.qkv(self.norm(x))
        q, k, v = h.chunk(3, dim=1)

        def reshape_heads(tensor: Tensor) -> Tensor:
            return tensor.view(batch, self.num_heads, self.head_dim, height * width).transpose(2, 3)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(2, 3).reshape(batch, channels, height, width)
        return x + self.proj(attn)


class Downsample(nn.Module):
    """Learned strided-convolution downsampling block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Construct a 2x downsampling layer."""
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Downsample the feature map by a factor of 2."""
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor upsampling followed by a refinement convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Construct a 2x upsampling layer."""
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Upsample the feature map by a factor of 2."""
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class CoarseEncoder(nn.Module):
    """Encode the coarse conditioning field into multi-scale feature priors."""

    def __init__(self, base_channels: int, channel_mults: Tuple[int, ...]) -> None:
        """Build the multi-scale coarse-feature encoder."""
        super().__init__()
        channels = [base_channels * mult for mult in channel_mults]
        self.stem = nn.Conv2d(1, channels[0], kernel_size=3, padding=1)
        self.stage0 = nn.Sequential(
            PlainResBlock(channels[0], channels[0]),
            PlainResBlock(channels[0], channels[0]),
        )
        self.down1 = Downsample(channels[0], channels[1])
        self.stage1 = nn.Sequential(
            PlainResBlock(channels[1], channels[1]),
            PlainResBlock(channels[1], channels[1]),
        )
        self.down2 = Downsample(channels[1], channels[2])
        self.stage2 = nn.Sequential(
            PlainResBlock(channels[2], channels[2]),
            PlainResBlock(channels[2], channels[2]),
        )
        self.down3 = Downsample(channels[2], channels[3])
        self.stage3 = nn.Sequential(
            PlainResBlock(channels[3], channels[3]),
            PlainResBlock(channels[3], channels[3]),
        )

    def forward(self, m_c_cond: Tensor) -> List[Tensor]:
        """Encode the coarse condition into features at four spatial scales."""
        feat0 = self.stage0(self.stem(m_c_cond))
        feat1 = self.stage1(self.down1(feat0))
        feat2 = self.stage2(self.down2(feat1))
        feat3 = self.stage3(self.down3(feat2))
        return [feat0, feat1, feat2, feat3]


class _ResidualVelocityUNetBase(nn.Module):
    """Coarse-conditioned residual vector-field backbone for Rectified Flow."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        cond_dim: int = 256,
        attn_resolutions: Tuple[int, ...] = (32, 16),
        num_heads: int = 4,
    ) -> None:
        """Construct the shared specialized 128x128 conditioned residual U-Net."""
        super().__init__()
        if in_channels != 2:
            raise ValueError("This model expects concatenated method state and m_c_cond with 2 input channels.")
        if out_channels != 1:
            raise ValueError("This model expects a single residual-shaped output channel.")
        if len(channel_mults) != 4:
            raise ValueError("This implementation expects exactly four scales: 128, 64, 32, 16.")
        if num_res_blocks != 2:
            raise ValueError("This implementation expects two ResBlocks per scale.")

        self.time_embed = SinusoidalTimeEmbedding(cond_dim)
        self.time_mlp = ConditionMLP(cond_dim, cond_dim)

        channels = [base_channels * mult for mult in channel_mults]
        self.coarse_encoder = CoarseEncoder(base_channels, channel_mults)
        self.coarse_projs = nn.ModuleList([nn.Conv2d(ch, ch, kernel_size=1) for ch in channels])

        self.input_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        self.enc0 = nn.ModuleList([ResBlock(channels[0], channels[0], cond_dim), ResBlock(channels[0], channels[0], cond_dim)])
        self.down1 = Downsample(channels[0], channels[1])
        self.enc1 = nn.ModuleList([ResBlock(channels[1], channels[1], cond_dim), ResBlock(channels[1], channels[1], cond_dim)])
        self.down2 = Downsample(channels[1], channels[2])
        self.enc2 = nn.ModuleList([ResBlock(channels[2], channels[2], cond_dim), ResBlock(channels[2], channels[2], cond_dim)])
        self.enc2_attn = AttentionBlock(channels[2], num_heads) if 32 in attn_resolutions else nn.Identity()
        self.down3 = Downsample(channels[2], channels[3])
        self.enc3 = nn.ModuleList([ResBlock(channels[3], channels[3], cond_dim), ResBlock(channels[3], channels[3], cond_dim)])
        self.enc3_attn = AttentionBlock(channels[3], num_heads) if 16 in attn_resolutions else nn.Identity()

        self.mid_block1 = ResBlock(channels[3], channels[3], cond_dim)
        self.mid_attn = AttentionBlock(channels[3], num_heads)
        self.mid_block2 = ResBlock(channels[3], channels[3], cond_dim)

        self.up2 = Upsample(channels[3], channels[2])
        self.dec2 = nn.ModuleList([ResBlock(channels[2] * 2, channels[2], cond_dim), ResBlock(channels[2], channels[2], cond_dim)])
        self.dec2_attn = AttentionBlock(channels[2], num_heads) if 32 in attn_resolutions else nn.Identity()
        self.up1 = Upsample(channels[2], channels[1])
        self.dec1 = nn.ModuleList([ResBlock(channels[1] * 2, channels[1], cond_dim), ResBlock(channels[1], channels[1], cond_dim)])
        self.up0 = Upsample(channels[1], channels[0])
        self.dec0 = nn.ModuleList([ResBlock(channels[0] * 2, channels[0], cond_dim), ResBlock(channels[0], channels[0], cond_dim)])

        self.out_norm = _group_norm(channels[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)

    def _run_stage(self, x: Tensor, blocks: nn.ModuleList, cond: Tensor) -> Tensor:
        for block in blocks:
            x = block(x, cond)
        return x

    def _forward_impl(self, evolving_state: Tensor, t: Tensor, m_c_cond: Tensor) -> Tensor:
        if evolving_state.shape != m_c_cond.shape:
            raise ValueError("The evolving state and m_c_cond must have the same shape.")
        if evolving_state.ndim != 4 or evolving_state.shape[1] != 1 or evolving_state.shape[2:] != (128, 128):
            raise ValueError("Expected evolving state and m_c_cond shape [B, 1, 128, 128].")
        if t.ndim != 1 or t.shape[0] != evolving_state.shape[0]:
            raise ValueError("Expected time tensor with shape [B].")

        cond = self.time_mlp(self.time_embed(t))

        coarse_feats = self.coarse_encoder(m_c_cond)
        coarse_feats = [proj(feat) for proj, feat in zip(self.coarse_projs, coarse_feats)]

        x = self.input_conv(torch.cat([evolving_state, m_c_cond], dim=1))

        x = x + coarse_feats[0]
        x = self._run_stage(x, self.enc0, cond)
        skip0 = x

        x = self.down1(x)
        x = x + coarse_feats[1]
        x = self._run_stage(x, self.enc1, cond)
        skip1 = x

        x = self.down2(x)
        x = x + coarse_feats[2]
        x = self._run_stage(x, self.enc2, cond)
        x = self.enc2_attn(x)
        skip2 = x

        x = self.down3(x)
        x = x + coarse_feats[3]
        x = self._run_stage(x, self.enc3, cond)
        x = self.enc3_attn(x)

        x = self.mid_block1(x, cond)
        x = self.mid_attn(x)
        x = self.mid_block2(x, cond)

        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self._run_stage(x, self.dec2, cond)
        x = self.dec2_attn(x)

        x = self.up1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self._run_stage(x, self.dec1, cond)

        x = self.up0(x)
        x = torch.cat([x, skip0], dim=1)
        x = self._run_stage(x, self.dec0, cond)

        return self.out_conv(self.out_act(self.out_norm(x)))


class ResidualVelocityUNet(_ResidualVelocityUNetBase):
    """Rectified Flow network with a reduced public constructor surface."""

    def __init__(
        self,
        base_channels: int = 64,
        cond_dim: int = 256,
        attn_resolutions: Tuple[int, ...] = (32, 16),
        num_heads: int = 4,
    ) -> None:
        super().__init__(
            in_channels=2,
            out_channels=1,
            base_channels=base_channels,
            channel_mults=(1, 2, 4, 8),
            num_res_blocks=2,
            cond_dim=cond_dim,
            attn_resolutions=attn_resolutions,
            num_heads=num_heads,
        )

    def forward(self, state_t: Tensor, t: Tensor, m_c_cond: Tensor) -> Tensor:
        """Predict the Rectified Flow vector field for one path state."""
        return self._forward_impl(state_t, t, m_c_cond)


__all__ = ["ResidualVelocityUNet"]
