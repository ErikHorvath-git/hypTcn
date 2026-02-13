"""Custom PyTorch layers used by the Temporal Convolutional Network."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class ChoppedCausalConvolution(nn.Conv1d):
    """Causal convolution that removes the future-looking padding from the output."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self._padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        if self._padding > 0:
            return out[:, :, :-self._padding]
        return out


class TemporalResidualBlock(nn.Module):
    """Residual block composed of dilated causal convolutions and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = weight_norm(
            ChoppedCausalConvolution(in_channels, out_channels, kernel_size, dilation)
        )
        self.conv2 = weight_norm(
            ChoppedCausalConvolution(out_channels, out_channels, kernel_size, dilation)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        out = F.relu(self.conv1(x))
        out = self.dropout1(out)
        out = F.relu(self.conv2(out))
        out = self.dropout2(out)
        return F.relu(out + residual)
