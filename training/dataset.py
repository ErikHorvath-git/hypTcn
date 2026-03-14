"""Dataset loader for hypTcn training.

Loads per-frame .npy files from data/normal/ and data/malware/.
Each file: dict with "features" key, shape (20,) float32.
Optional "class_label" key (int 0-4); defaults to 0 for normal, 1 for malware.

Applies a sliding window of length 16 with stride 8 over sorted filenames
within each class directory, yielding (sequence_tensor, anomaly_label, class_label) tuples.

sequence_tensor shape: (20, 16)  — features × time steps
anomaly_label: float32  0.0 = normal, 1.0 = malware
class_label:   int64    0=normal, 1=shellcode, 2=rootkit, 3=cryptominer, 4=ransomware

BinFrameDataset loads raw .bin files written by the Go collector:
  [8B uint64 LE addr][4B uint32 LE label_id][4096B page] = 4108 bytes per file.
Features are extracted on-the-fly via python/model/features.py.
"""

from __future__ import annotations

import os
import struct
import sys
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


# ── Label ID mapping (must match engine.go collectLabelIDs) ──────────────────
_LABEL_ID_TO_CLASS = {
    0: 0,  # normal
    1: 1,  # malware (generic → treated as class 1)
    2: 1,  # shellcode
    3: 2,  # rootkit
    4: 3,  # cryptominer
    5: 4,  # ransomware
}
_LABEL_ID_ANOMALY = {0: 0.0}  # label_id 0 = normal; all others = malware

# Canonical class label indices that match ACTIVITY_CLASSES in tcn.py
_BIN_LABEL_TO_CLASS = {
    0: 0,  # normal → normal
    1: 1,  # malware → shellcode bucket (generic)
    2: 1,  # shellcode
    3: 2,  # rootkit
    4: 3,  # cryptominer
    5: 4,  # ransomware
}

_BIN_FRAME_SIZE = 4108   # 8 + 4 + 4096
_BIN_HEADER_FMT = "<QI"  # uint64 addr, uint32 label_id


def _load_bin_frame(path: str) -> tuple[np.ndarray, float, int] | None:
    """Parse one .bin frame file; return (page_bytes, anomaly_label, class_label) or None."""
    try:
        with open(path, "rb") as f:
            raw = f.read(_BIN_FRAME_SIZE)
        if len(raw) != _BIN_FRAME_SIZE:
            return None
        addr, label_id = struct.unpack_from(_BIN_HEADER_FMT, raw, 0)
        page = raw[12:]  # 4096 bytes
        anomaly = 0.0 if label_id == 0 else 1.0
        cls = _BIN_LABEL_TO_CLASS.get(label_id, 1)
        return np.frombuffer(page, dtype=np.uint8).copy(), anomaly, cls
    except Exception:
        return None


def _extract_features_from_page(page: np.ndarray, address: int, prev_features: np.ndarray | None) -> np.ndarray:
    """Extract FEATURE_DIM features from a raw 4096-byte page."""
    # Lazily import features module from python/
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _python_dir = os.path.join(_repo_root, "python")
    if _python_dir not in sys.path:
        sys.path.insert(0, _python_dir)
    from model.features import extract  # noqa: PLC0415
    return extract(page.tobytes(), address, prev_features)


class BinFrameDataset(Dataset):
    """PyTorch Dataset from raw .bin frame files collected by the Go hyptcn binary.

    Reads all .bin files under collect_dir/{label}/ for each label in LABELS.
    Features are extracted on-the-fly from raw page bytes.
    Applies a sliding window of seq_len with stride over sorted frames per label.

    Args:
        collect_dir: Root collection directory (e.g. "collect_for_training/").
        seq_len:     Sliding window length (default: 16).
        stride:      Stride between windows (default: 8).
        labels:      Tuple of subdirectory names to load. Defaults to all 6.
    """

    LABELS = ("normal", "malware", "shellcode", "rootkit", "cryptominer", "ransomware")

    def __init__(
        self,
        collect_dir: str = "collect_for_training",
        seq_len: int = SEQUENCE_LENGTH,
        stride: int = STRIDE,
        labels: tuple[str, ...] | None = None,
    ) -> None:
        self.seq_len = seq_len
        self.stride = stride
        if labels is None:
            labels = self.LABELS

        all_windows:  list[np.ndarray] = []
        all_anomaly:  list[float]      = []
        all_classes:  list[int]        = []

        self.n_normal  = 0
        self.n_malware = 0

        for label in labels:
            label_dir = os.path.join(collect_dir, label)
            if not os.path.isdir(label_dir):
                continue

            files = sorted(
                f for f in os.listdir(label_dir) if f.endswith(".bin")
            )
            if not files:
                continue

            # Load and extract features for every frame in this label directory.
            frames_feat:   list[np.ndarray] = []
            frames_anomaly: list[float]     = []
            frames_class:  list[int]        = []
            prev_feat: np.ndarray | None    = None

            for fname in files:
                result = _load_bin_frame(os.path.join(label_dir, fname))
                if result is None:
                    prev_feat = None
                    continue
                page, anomaly, cls = result
                # Extract features; addr encoded in filename as hex (fallback 0)
                try:
                    addr_hex = fname.split("_")[1].replace(".bin", "")
                    addr = int(addr_hex, 16)
                except (IndexError, ValueError):
                    addr = 0
                feat = _extract_features_from_page(page, addr, prev_feat)
                frames_feat.append(feat)
                frames_anomaly.append(anomaly)
                frames_class.append(cls)
                prev_feat = feat

            # Slide window over this label's frames.
            n = len(frames_feat)
            for start in range(0, n - seq_len + 1, stride):
                window = np.stack(frames_feat[start: start + seq_len])  # (seq_len, 20)
                all_windows.append(window)
                # Majority anomaly label in window
                win_anomaly = frames_anomaly[start: start + seq_len]
                anom = 1.0 if sum(win_anomaly) / len(win_anomaly) >= 0.5 else 0.0
                all_anomaly.append(anom)
                # Majority class label
                win_cls = frames_class[start: start + seq_len]
                counts = np.bincount(win_cls, minlength=5)
                all_classes.append(int(np.argmax(counts)))

                if anom == 0.0:
                    self.n_normal += 1
                else:
                    self.n_malware += 1

        self._sequences     = all_windows
        self._anomaly_labels = all_anomaly
        self._class_labels  = all_classes

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq           = torch.from_numpy(self._sequences[idx]).T.float()         # (20, 16)
        anomaly_label = torch.tensor(self._anomaly_labels[idx], dtype=torch.float32)
        class_label   = torch.tensor(self._class_labels[idx],   dtype=torch.long)
        return seq, anomaly_label, class_label

    def summary(self) -> str:
        return (
            f"BinFrameDataset: {len(self)} sequences "
            f"({self.n_normal} normal, {self.n_malware} malware) "
            f"window={self.seq_len} stride={self.stride}"
        )


def stratified_split(
    dataset: MemoryPageDataset | BinFrameDataset,
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
