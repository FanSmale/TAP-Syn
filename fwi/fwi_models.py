"""DL-FWI models used by the paper experiments."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import FWIModelConfig


class _Scale(nn.Module):
    def __init__(self, channels: int, initial: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((1, channels, 1, 1), initial))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.weight * inputs


def _haar_filters(channels: int) -> tuple[Tensor, Tensor]:
    scale = 1.0 / math.sqrt(2.0)
    low = torch.tensor([scale, scale])
    # Match PyWavelets db1 after the reversals in the original ABA-FWI code.
    high = torch.tensor([scale, -scale])
    reconstruction_high = torch.tensor([scale, -scale])
    decomposition = torch.stack(
        [
            low[:, None] * low[None, :],
            low[:, None] * high[None, :],
            high[:, None] * low[None, :],
            high[:, None] * high[None, :],
        ]
    )[:, None].repeat(channels, 1, 1, 1)
    reconstruction = torch.stack(
        [
            low[:, None] * low[None, :],
            low[:, None] * reconstruction_high[None, :],
            reconstruction_high[:, None] * low[None, :],
            reconstruction_high[:, None] * reconstruction_high[None, :],
        ]
    )[:, None].repeat(channels, 1, 1, 1)
    return decomposition, reconstruction


class WTConv2d(nn.Module):
    """The one-level db1 WTConv used by the published ABA-FWI network."""

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        decomposition, reconstruction = _haar_filters(channels)
        self.wt_filter = nn.Parameter(decomposition, requires_grad=False)
        self.iwt_filter = nn.Parameter(reconstruction, requires_grad=False)
        self.channels = channels
        self.base_conv = nn.Conv2d(
            channels, channels, kernel_size, padding="same", groups=channels
        )
        self.base_scale = _Scale(channels, 1.0)
        self.wavelet_conv = nn.Conv2d(
            channels * 4,
            channels * 4,
            kernel_size,
            padding="same",
            groups=channels * 4,
            bias=False,
        )
        self.wavelet_scale = _Scale(channels * 4, 0.1)

    def forward(self, inputs: Tensor) -> Tensor:
        height, width = inputs.shape[-2:]
        padded = F.pad(inputs, (0, width % 2, 0, height % 2))
        transformed = F.conv2d(
            padded, self.wt_filter, stride=2, groups=self.channels
        ).reshape(inputs.shape[0], self.channels, 4, padded.shape[-2] // 2, padded.shape[-1] // 2)
        transformed_shape = transformed.shape
        filtered = self.wavelet_scale(
            self.wavelet_conv(
                transformed.reshape(
                    transformed_shape[0],
                    transformed_shape[1] * 4,
                    transformed_shape[3],
                    transformed_shape[4],
                )
            )
        ).reshape(transformed_shape)
        reconstructed = F.conv_transpose2d(
            filtered.reshape(
                transformed_shape[0],
                transformed_shape[1] * 4,
                transformed_shape[3],
                transformed_shape[4],
            ),
            self.iwt_filter,
            stride=2,
            groups=self.channels,
        )[..., :height, :width]
        return self.base_scale(self.base_conv(inputs)) + reconstructed


class _ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 1,
        *,
        tanh: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.Tanh() if tanh else nn.PReLU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class _DeconvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class _SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding=1, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        average = inputs.mean(dim=1, keepdim=True)
        maximum = inputs.amax(dim=1, keepdim=True)
        weights = torch.sigmoid(self.conv(torch.cat([average, maximum], dim=1)))
        return inputs * weights


class ABAFWI(nn.Module):
    """Unmodified OpenFWI-shape ABA-FWI architecture."""

    input_shape = (5, 1000, 70)
    output_shape = (1, 70, 70)

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = _ConvBlock(5, 32, (7, 1), (2, 1), (3, 0))
        self.conv2 = _ConvBlock(32, 64, (3, 1), (2, 1), (1, 0))
        self.conv3 = _ConvBlock(64, 64, (3, 1), 1, (1, 0))
        self.conv4 = _ConvBlock(64, 64, (3, 1), (2, 1), (1, 0))
        self.conv5 = _ConvBlock(64, 64, (3, 1), 1, (1, 0))
        self.conv6 = _ConvBlock(64, 128, (3, 1), (2, 1), (1, 0))
        self.conv7 = _ConvBlock(128, 128, (3, 1), 1, (1, 0))
        self.wt0 = WTConv2d(128)
        self.conv8 = _ConvBlock(128, 128, 3, 2, (9, 6))
        self.conv9 = _ConvBlock(128, 128)
        self.wt1 = WTConv2d(128)
        self.conv10 = _ConvBlock(128, 256, 3, 2, 1)
        self.conv11 = _ConvBlock(256, 256)
        self.wt2 = WTConv2d(256)
        self.conv12 = _ConvBlock(256, 256, 3, 2, 1)
        self.conv13 = _ConvBlock(256, 256)
        self.conv14 = _ConvBlock(256, 512, 3, 2, 1)
        self.conv15 = _ConvBlock(512, 512)

        self.up1 = _DeconvBlock(512, 256)
        self.reduce1 = nn.Conv2d(512, 256, 1)
        self.decode1 = _ConvBlock(256, 256)
        self.attention1 = _SpatialAttention()
        self.up2 = _DeconvBlock(256, 128)
        self.reduce_skip = nn.Conv2d(256, 128, 1)
        self.reduce2 = nn.Conv2d(256, 128, 1)
        self.decode2 = _ConvBlock(128, 128)
        self.attention2 = _SpatialAttention()
        self.up3 = _DeconvBlock(128, 64)
        self.decode3 = _ConvBlock(64, 64)
        self.attention3 = _SpatialAttention()
        self.up4 = _DeconvBlock(64, 32)
        self.decode4 = _ConvBlock(32, 32)
        self.attention4 = _SpatialAttention()
        self.output = _ConvBlock(32, 1, tanh=True)

    def forward(self, seismic: Tensor) -> Tensor:
        if seismic.ndim != 4 or tuple(seismic.shape[1:]) != self.input_shape:
            raise ValueError(
                f"ABAFWI requires [B,5,1000,70], got {tuple(seismic.shape)}."
            )
        x = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(seismic)))))
        skip_63 = self.wt0(self.conv7(self.conv6(x)))
        skip_40 = self.wt1(self.conv9(self.conv8(skip_63)))
        skip_20 = self.wt2(self.conv11(self.conv10(skip_40)))
        skip_10 = self.conv13(self.conv12(skip_20))
        bottleneck = self.conv15(self.conv14(skip_10))

        x = self.decode1(self.reduce1(torch.cat([skip_10, self.up1(bottleneck)], dim=1)))
        x = self.attention1(x)
        x = self.up2(x)
        x = self.decode2(
            self.reduce2(torch.cat([self.reduce_skip(skip_20), x], dim=1))
        )
        x = self.attention2(x)
        x = self.attention3(self.decode3(self.up3(x)))
        x = self.attention4(self.decode4(self.up4(x)))
        output = self.output(x[..., 5:-5, 5:-5])
        if tuple(output.shape[1:]) != self.output_shape:
            raise RuntimeError(f"ABAFWI produced unexpected shape {tuple(output.shape)}.")
        return output


def build_fwi_model(config: FWIModelConfig) -> nn.Module:
    if config.name != "aba_fwi":
        raise ValueError(f"Unsupported FWI model '{config.name}'; expected 'aba_fwi'.")
    return ABAFWI()


def count_parameters(modules: Iterable[nn.Module]) -> int:
    return sum(parameter.numel() for module in modules for parameter in module.parameters())


__all__ = ["ABAFWI", "WTConv2d", "build_fwi_model", "count_parameters"]
