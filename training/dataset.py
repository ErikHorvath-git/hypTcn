"""Dataset loader for hypTcn training.

Loads per-frame .npy files from data/normal/ and data/malware/.
Each file: dict with "features" key, shape (20,) float32.
Optional "class_label" key (int 0-4); defaults to 0 for normal, 1 for malware.

Applies a sliding window of length 16 with stride 8 over sorted filenames
within each class directory, yielding (sequence_tensor, anomaly_label, class_label) tuples.

sequence_tensor shape: (20, 16)  — features × time steps
anomaly_label: float32  0.0 = normal, 1.0 = malware
class_label:   int64    0=normal, 1=shellcode, 2=rootkit, 3=cryptominer, 4=ransomware
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

FEATURE_DIM = 20
SEQUENCE_LENGTH = 16
STRIDE = 8


def _load_frame(path: str) -> tuple[np.ndarray, int] | None:
    """Load a single .npy frame file; return (features, class_label) or None on error."""
    try:
        obj = np.load(path, allow_pickle=True).item()
        feat = np.asarray(obj["features"], dtype=np.float32)
        if feat.shape[0] != FEATURE_DIM:
            # Pad or truncate to FEATURE_DIM for backward compatibility with
            # files generated when FEATURE_DIM was 18.
            if feat.shape[0] < FEATURE_DIM:
                feat = np.pad(feat, (0, FEATURE_DIM - feat.shape[0]))
            else:
                feat = feat[:FEATURE_DIM]
        cls = int(obj.get("class_label", -1))  # -1 = not set, resolved below
        return feat, cls
    except Exception:
        return None


def _load_class_frames(directory: str, default_class: int) -> tuple[list[np.ndarray], list[int]]:
    """Load all valid frames from a class directory.

    Returns (features_list, class_labels_list).
    class_label falls back to default_class when absent from the .npy file.
    """
    if not os.path.isdir(directory):
        return [], []
    files = sorted(f for f in os.listdir(directory) if f.endswith(".npy"))
    feats: list[np.ndarray] = []
    classes: list[int] = []
    for fname in files:
        result = _load_frame(os.path.join(directory, fname))
        if result is None:
            continue
        feat, cls = result
        feats.append(feat)
        classes.append(cls if cls >= 0 else default_class)
    return feats, classes


def _make_windows(
    frames: list[np.ndarray],
    class_labels: list[int],
    seq_len: int = SEQUENCE_LENGTH,
    stride: int = STRIDE,
) -> tuple[list[np.ndarray], list[int]]:
    """Slide a window of seq_len over frames with given stride.

    The class label for a window is the majority label among its frames
    (ties broken by the last frame's label for simplicity).

    Returns (windows, window_class_labels).
    Each window: shape (seq_len, FEATURE_DIM).
    """
    windows = []
    win_classes = []
    n = len(frames)
    for start in range(0, n - seq_len + 1, stride):
        window = np.stack(frames[start: start + seq_len])   # (seq_len, FEATURE_DIM)
        windows.append(window)
        # Use the most common class in the window; last frame breaks ties.
        window_cls = class_labels[start: start + seq_len]
        counts = np.bincount(window_cls, minlength=5)
        win_classes.append(int(np.argmax(counts)))
    return windows, win_classes


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

        norm_feats, norm_cls = _load_class_frames(
            os.path.join(data_dir, "normal"), default_class=0
        )
        mal_feats, mal_cls = _load_class_frames(
            os.path.join(data_dir, "malware"), default_class=1
        )

        norm_wins, norm_win_cls = _make_windows(norm_feats, norm_cls, seq_len, stride)
        mal_wins,  mal_win_cls  = _make_windows(mal_feats,  mal_cls,  seq_len, stride)

        self._sequences: list[np.ndarray] = norm_wins + mal_wins
        # Binary anomaly label: 0 for normal windows, 1 for malware windows
        self._anomaly_labels: list[int] = [0] * len(norm_wins) + [1] * len(mal_wins)
        # Multi-class activity label (0–4)
        self._class_labels: list[int] = norm_win_cls + mal_win_cls

        self.n_normal  = len(norm_wins)
        self.n_malware = len(mal_wins)

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # seq shape: (seq_len, FEATURE_DIM) → transpose to (FEATURE_DIM, seq_len)
        seq           = torch.from_numpy(self._sequences[idx]).T.float()        # (20, 16)
        anomaly_label = torch.tensor(self._anomaly_labels[idx], dtype=torch.float32)
        class_label   = torch.tensor(self._class_labels[idx],   dtype=torch.long)
        return seq, anomaly_label, class_label

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
    """Split dataset into train/val/test preserving binary class ratios.

    Returns (train_subset, val_subset, test_subset).
    """
    rng = np.random.default_rng(seed)

    normal_idx  = [i for i, l in enumerate(dataset._anomaly_labels) if l == 0]
    malware_idx = [i for i, l in enumerate(dataset._anomaly_labels) if l == 1]

    def _split(indices: list[int]) -> tuple[list[int], list[int], list[int]]:
        arr = np.array(indices)
        rng.shuffle(arr)
        n = len(arr)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)
        return (
            arr[:n_train].tolist(),
            arr[n_train: n_train + n_val].tolist(),
            arr[n_train + n_val:].tolist(),
        )

    n_tr, n_va, n_te = _split(normal_idx)
    m_tr, m_va, m_te = _split(malware_idx)

    train_idx = n_tr + m_tr
    val_idx   = n_va + m_va
    test_idx  = n_te + m_te

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return (
        torch.utils.data.Subset(dataset, train_idx),
        torch.utils.data.Subset(dataset, val_idx),
        torch.utils.data.Subset(dataset, test_idx),
    )
