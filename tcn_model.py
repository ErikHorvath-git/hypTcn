"""Standalone TCN anomaly-detection model for the hypTcn pipeline.

Input:  (batch, features=18, sequence=16)  — one sliding window of pages
Output: float in [0.0, 1.0]               — anomaly score (sigmoid)

Architecture per spec:
    3 × TCNBlock (dilations 1, 2, 4)
        dilated causal Conv1d, kernel=3, filters=32
        weight normalization, ReLU, Dropout(0.1)
        residual connection (1×1 conv when dims differ)
    GlobalAveragePool1d
    Dense(32 → 16, ReLU)
    Dense(16 → 1, Sigmoid)

Weights are loaded from models/tcn_weights.pt when the file exists;
random init is used otherwise (no pre-training required).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

if TYPE_CHECKING:
    import numpy as np

# ── Hyperparameters ────────────────────────────────────────────────────────────
FEATURE_DIM = 18
SEQUENCE_LENGTH = 16
KERNEL_SIZE = 3
NUM_BLOCKS = 3
FILTERS = 32
DROPOUT = 0.1

WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "tcn_weights.pt"
)


# ── Building blocks ────────────────────────────────────────────────────────────

class _CausalConv1d(nn.Conv1d):
    """Dilated causal Conv1d — removes the future-facing padding from output."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None:
        self._causal_pad = (kernel_size - 1) * dilation
        super().__init__(
            in_ch, out_ch, kernel_size,
            padding=self._causal_pad,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        return out[:, :, :-self._causal_pad] if self._causal_pad > 0 else out


class _TCNBlock(nn.Module):
    """One dilated residual block: two causal convs + skip connection."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = weight_norm(_CausalConv1d(in_ch, out_ch, kernel_size, dilation))
        self.conv2 = weight_norm(_CausalConv1d(out_ch, out_ch, kernel_size, dilation))
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        # 1×1 projection when channel dims differ
        self.residual = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.drop1(F.relu(self.conv1(x)))
        out = self.drop2(F.relu(self.conv2(out)))
        return F.relu(out + res)


# ── Model ──────────────────────────────────────────────────────────────────────

class TCNAnomalyDetector(nn.Module):
    """3-block TCN → GlobalAvgPool → Dense(16, ReLU) → Dense(1, Sigmoid)."""

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        filters: int = FILTERS,
        kernel_size: int = KERNEL_SIZE,
        num_blocks: int = NUM_BLOCKS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_ch = feature_dim
        for i in range(num_blocks):
            blocks.append(
                _TCNBlock(in_ch, filters, kernel_size, dilation=2 ** i, dropout=dropout)
            )
            in_ch = filters

        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, features, sequence) → (batch,) scores in [0, 1]."""
        x = self.tcn(x)
        x = self.pool(x)
        return self.head(x).squeeze(-1)


# ── Public API ─────────────────────────────────────────────────────────────────

def load_model(weights_path: str = WEIGHTS_PATH) -> TCNAnomalyDetector:
    """Return an eval-mode detector, loading weights from disk when available."""
    model = TCNAnomalyDetector()
    if os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.eval()
    return model


def infer(model: TCNAnomalyDetector, window: "np.ndarray") -> float:
    """Run one inference pass on a full sliding window.

    Args:
        model:  An eval-mode TCNAnomalyDetector.
        window: numpy array of shape (FEATURE_DIM, SEQUENCE_LENGTH) = (18, 16).
                Accepts (SEQUENCE_LENGTH, FEATURE_DIM) too and transposes automatically.

    Returns:
        Anomaly score in [0.0, 1.0].
    """
    if window.shape == (SEQUENCE_LENGTH, FEATURE_DIM):
        window = window.T  # → (FEATURE_DIM, SEQUENCE_LENGTH)
    tensor = torch.from_numpy(window).unsqueeze(0).float()  # (1, 18, 16)
    with torch.no_grad():
        score = model(tensor)
    return float(score.squeeze().item())
