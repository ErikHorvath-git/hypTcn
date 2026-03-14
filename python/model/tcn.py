"""Standalone TCN anomaly-detection and activity-classification model.

Input:  (batch, features=20, sequence=16)  — one sliding window of pages
Output: two heads computed from a shared backbone

    anomaly_logit  — (batch,) raw logit; sigmoid → score in [0,1]
    class_logits   — (batch, 5) raw logits; argmax → activity class

Activity classes:
    0 = normal
    1 = shellcode
    2 = rootkit
    3 = cryptominer
    4 = ransomware

Architecture:
    3 × TCNBlock (dilations 1, 2, 4)
        dilated causal Conv1d, kernel=3, filters=32
        weight normalization, ReLU, Dropout(0.1)
        residual connection (1×1 conv when dims differ)
    GlobalAveragePool1d → (batch, 32)
    ┌─ Anomaly head:   Linear(32→16, ReLU) → Linear(16→1)
    └─ Class head:     Linear(32→64, ReLU) → Linear(64→5)

Weights are loaded from models/tcn_weights.pt when the file exists.
Hyperparameters are read from models/config.json; when absent, defaults apply.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

if TYPE_CHECKING:
    import numpy as np

# ── Hyperparameter defaults ────────────────────────────────────────────────────
FEATURE_DIM = 20
SEQUENCE_LENGTH = 16
KERNEL_SIZE = 3
NUM_BLOCKS = 3
FILTERS = 32
DROPOUT = 0.1
NUM_CLASSES = 5

ACTIVITY_CLASSES = ["normal", "shellcode", "rootkit", "cryptominer", "ransomware"]

# Resolve to <repo>/models/ regardless of CWD
_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
)
WEIGHTS_PATH = os.path.join(_MODELS_DIR, "tcn_weights.pt")
CONFIG_PATH = os.path.join(_MODELS_DIR, "config.json")


def _load_config(config_path: str = CONFIG_PATH) -> dict | None:
    """Load models/config.json; return None if absent or unreadable."""
    try:
        with open(config_path) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


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
        self.residual = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.drop1(F.relu(self.conv1(x)))
        out = self.drop2(F.relu(self.conv2(out)))
        return F.relu(out + res)


# ── Model ──────────────────────────────────────────────────────────────────────

class TCNAnomalyDetector(nn.Module):
    """Shared TCN backbone with two output heads.

    Anomaly head:  Dense(32→16, ReLU) → Dense(16→1)  → raw logit (sigmoid in infer)
    Class head:    Dense(32→64, ReLU) → Dense(64→5)  → raw logits (softmax in infer)
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        filters: int = FILTERS,
        kernel_size: int = KERNEL_SIZE,
        num_blocks: int = NUM_BLOCKS,
        dropout: float = DROPOUT,
        num_classes: int = NUM_CLASSES,
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
        self.flatten = nn.Flatten()

        # Anomaly detection head (binary)
        self.anomaly_head = nn.Sequential(
            nn.Linear(filters, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # Activity classification head (multi-class)
        self.class_head = nn.Sequential(
            nn.Linear(filters, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, features, sequence) → (anomaly_logit, class_logits).

        anomaly_logit:  (batch,)    — use sigmoid for probability, BCEWithLogitsLoss in training
        class_logits:   (batch, 5)  — use softmax/argmax for class, CrossEntropyLoss in training
        """
        x = self.tcn(x)
        x = self.pool(x)
        x = self.flatten(x)                             # (batch, filters)
        anomaly_logit = self.anomaly_head(x).squeeze(-1)  # (batch,)
        class_logits  = self.class_head(x)               # (batch, num_classes)
        return anomaly_logit, class_logits


# ── Public API ─────────────────────────────────────────────────────────────────

def load_model(weights_path: str = WEIGHTS_PATH) -> TCNAnomalyDetector:
    """Return an eval-mode detector.

    Construction order:
    1. If models/config.json exists, use its hyperparameters.
    2. Load weights from weights_path if the file exists (strict=False so
       partial weight files — e.g. without class_head — load cleanly).
    3. Fall back to default hyperparameters + random init otherwise.
    """
    config = _load_config()
    if config:
        model = TCNAnomalyDetector(
            feature_dim=config.get("feature_dim", FEATURE_DIM),
            filters=config.get("filters", FILTERS),
            kernel_size=config.get("kernel_size", KERNEL_SIZE),
            num_blocks=config.get("num_blocks", NUM_BLOCKS),
            num_classes=config.get("num_classes", NUM_CLASSES),
        )
    else:
        model = TCNAnomalyDetector()

    if os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        # strict=False: allows loading weights trained before the class_head was
        # added, or with a different feature_dim (missing/extra keys are ignored).
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[tcn] {len(missing)} weight tensors randomly initialised "
                  f"(not in checkpoint): {missing[:3]}{'…' if len(missing) > 3 else ''}")

    model.eval()
    return model


def save_model(model: TCNAnomalyDetector, weights_path: str = WEIGHTS_PATH) -> None:
    """Save model state dict to weights_path, creating parent directories as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(weights_path)), exist_ok=True)
    torch.save(model.state_dict(), weights_path)


def infer(model: TCNAnomalyDetector, window: "np.ndarray") -> tuple[float, str]:
    """Run one inference pass on a full sliding window.

    Args:
        model:  An eval-mode TCNAnomalyDetector.
        window: numpy array of shape (FEATURE_DIM, SEQUENCE_LENGTH) = (20, 16).
                Accepts (SEQUENCE_LENGTH, FEATURE_DIM) too and transposes automatically.

    Returns:
        (anomaly_score, activity_class_name)
        anomaly_score:      float in [0.0, 1.0]
        activity_class_name: one of ACTIVITY_CLASSES
    """
    if window.shape == (SEQUENCE_LENGTH, FEATURE_DIM):
        window = window.T  # → (FEATURE_DIM, SEQUENCE_LENGTH)
    tensor = torch.from_numpy(window).unsqueeze(0).float()  # (1, 20, 16)
    with torch.no_grad():
        anomaly_logit, class_logits = model(tensor)
        anomaly_score = float(torch.sigmoid(anomaly_logit).squeeze().item())
        class_idx     = int(class_logits.argmax(dim=-1).squeeze().item())

    class_name = ACTIVITY_CLASSES[class_idx] if 0 <= class_idx < len(ACTIVITY_CLASSES) else "unknown"
    return anomaly_score, class_name
