"""Temporal Convolutional Network architecture for hypTcn."""

from __future__ import annotations

import torch
import torch.nn as nn

from .constants import (
    DILATION_LEVELS,
    DROPOUT,
    FEATURE_DIM,
    HIDDEN_CHANNELS,
    KERNEL_SIZE,
)
from .layers import TemporalResidualBlock


class TCNModel(nn.Module):
    """TCN that stacks dilated residual blocks and produces a single anomaly score."""

    def __init__(
        self,
        input_channels: int = FEATURE_DIM,
        hidden_channels: int = HIDDEN_CHANNELS,
        kernel_size: int = KERNEL_SIZE,
        num_levels: int = DILATION_LEVELS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_channels
        for level in range(num_levels):
            layers.append(
                TemporalResidualBlock(
                    in_channels,
                    hidden_channels,
                    kernel_size,
                    dilation=2 ** level,
                    dropout=dropout,
                )
            )
            in_channels = hidden_channels

        self.network = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        features = self.network(x)
        logits = self.classifier(features)
        return logits.squeeze(-1)

    def infer(self, sequence: torch.Tensor) -> float:
        """Return a sigmoid score in [0, 1] for the provided sequence."""
        self.eval()
        with torch.no_grad():
            logits = self(sequence)
            if logits.dim() > 0:
                logits = logits.squeeze(-1)
            score = torch.sigmoid(logits)
        return float(score.item())
