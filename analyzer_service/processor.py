"""Sliding window buffer and feature extraction for analyzer inputs."""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from analyzer_service.model.constants import FEATURE_DIM, SEQUENCE_LENGTH


def page_to_histogram(page: bytes, bins: int = FEATURE_DIM) -> np.ndarray:
    """Convert a 4 KiB page into a normalized histogram feature vector."""
    if len(page) != 4096:
        raise ValueError("expected 4096-byte page")
    arr = np.frombuffer(page, dtype=np.uint8)
    hist, _ = np.histogram(arr, bins=bins, range=(0, 256))
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


class PageBuffer:
    """Keeps the last N histograms ready for temporal inference."""

    def __init__(self, sequence_length: int = SEQUENCE_LENGTH, feature_dim: int = FEATURE_DIM) -> None:
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self._buffer: deque[np.ndarray] = deque(maxlen=sequence_length)

    def append(self, page: bytes) -> Optional[np.ndarray]:
        """Append a page and return a stacked sequence if the buffer is full."""
        features = page_to_histogram(page, self.feature_dim)
        self._buffer.append(features)
        if len(self._buffer) < self.sequence_length:
            return None
        stacked = np.stack(tuple(self._buffer), axis=-1)
        return stacked

    def reset(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
