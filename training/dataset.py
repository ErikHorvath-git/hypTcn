"""Dataset loader for hypTcn training.

Loads per-frame .npy files from data/normal/ and data/malware/.
Each file: dict with "features" key, shape (18,) float32.

Applies a sliding window of length 16 with stride 8 over sorted filenames
within each class directory, yielding (sequence_tensor, label) pairs.

sequence_tensor shape: (18, 16)  — features × time steps
label: 0 = normal, 1 = malware
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

FEATURE_DIM = 18
SEQUENCE_LENGTH = 16
STRIDE = 8


def _load_features(path: str) -> np.ndarray | None:
    """Load a single .npy frame file; return shape-(18,) array or None on error."""
    try:
        obj = np.load(path, allow_pickle=True).item()
        feat = np.asarray(obj["features"], dtype=np.float32)
        if feat.shape != (FEATURE_DIM,):
            return None
        return feat
    except Exception:
        return None


def _load_class_frames(directory: str) -> list[np.ndarray]:
    """Load all valid frames from a class directory, sorted by filename."""
    if not os.path.isdir(directory):
        return []
    files = sorted(
        f for f in os.listdir(directory) if f.endswith(".npy")
    )
    frames: list[np.ndarray] = []
    for fname in files:
        feat = _load_features(os.path.join(directory, fname))
        if feat is not None:
            frames.append(feat)
    return frames


def _make_windows(
    frames: list[np.ndarray],
    seq_len: int = SEQUENCE_LENGTH,
    stride: int = STRIDE,
) -> list[np.ndarray]:
    """Slide a window of seq_len over frames with given stride.

    Returns list of arrays each shape (seq_len, FEATURE_DIM).
    Windows that would go out of bounds are dropped.
    """
    windows = []
    n = len(frames)
    for start in range(0, n - seq_len + 1, stride):
        window = np.stack(frames[start : start + seq_len])  # (seq_len, FEATURE_DIM)
        windows.append(window)
    return windows


class MemoryPageDataset(Dataset):
    """PyTorch Dataset of sliding-window sequences from .npy frame files.

    Args:
        data_dir:  Root directory containing normal/ and malware/ subdirs.
        seq_len:   Sliding window length (default: 16).
        stride:    Stride between windows (default: 8).
    """

    def __init__(
        self,
        data_dir: str = "data",
        seq_len: int = SEQUENCE_LENGTH,
        stride: int = STRIDE,
    ) -> None:
        self.seq_len = seq_len
        self.stride = stride

        normal_frames = _load_class_frames(os.path.join(data_dir, "normal"))
        malware_frames = _load_class_frames(os.path.join(data_dir, "malware"))

        normal_windows = _make_windows(normal_frames, seq_len, stride)
        malware_windows = _make_windows(malware_frames, seq_len, stride)

        self._sequences: list[np.ndarray] = normal_windows + malware_windows
        self._labels: list[int] = [0] * len(normal_windows) + [1] * len(malware_windows)

        self.n_normal = len(normal_windows)
        self.n_malware = len(malware_windows)

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # seq shape: (seq_len, FEATURE_DIM) → transpose to (FEATURE_DIM, seq_len)
        seq = torch.from_numpy(self._sequences[idx]).T.float()  # (18, 16)
        label = torch.tensor(self._labels[idx], dtype=torch.float32)
        return seq, label

    def summary(self) -> str:
        return (
            f"MemoryPageDataset: {len(self)} sequences "
            f"({self.n_normal} normal, {self.n_malware} malware) "
            f"window={self.seq_len} stride={self.stride}"
        )


def stratified_split(
    dataset: MemoryPageDataset,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[
    torch.utils.data.Subset,
    torch.utils.data.Subset,
    torch.utils.data.Subset,
]:
    """Split dataset into train/val/test preserving class ratios.

    Returns (train_subset, val_subset, test_subset).
    """
    rng = np.random.default_rng(seed)

    normal_idx = [i for i, l in enumerate(dataset._labels) if l == 0]
    malware_idx = [i for i, l in enumerate(dataset._labels) if l == 1]

    def _split_indices(indices: list[int]) -> tuple[list[int], list[int], list[int]]:
        arr = np.array(indices)
        rng.shuffle(arr)
        n = len(arr)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        return (
            arr[:n_train].tolist(),
            arr[n_train : n_train + n_val].tolist(),
            arr[n_train + n_val :].tolist(),
        )

    n_tr, n_va, n_te = _split_indices(normal_idx)
    m_tr, m_va, m_te = _split_indices(malware_idx)

    train_idx = n_tr + m_tr
    val_idx = n_va + m_va
    test_idx = n_te + m_te

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return (
        torch.utils.data.Subset(dataset, train_idx),
        torch.utils.data.Subset(dataset, val_idx),
        torch.utils.data.Subset(dataset, test_idx),
    )
