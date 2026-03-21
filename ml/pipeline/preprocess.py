"""Preprocess raw .bin frames into numpy arrays ready for training.

Pipeline:
  ml/data/raw/<label>/*.bin  →  feature extraction  →  sliding windows
  →  train/val/test split  →  ml/data/processed/{X,y}_*.npy

Output files (in --output directory):
  X_train.npy          shape (N, FEATURE_DIM, SEQUENCE_LENGTH) float32
  X_val.npy
  X_test.npy
  y_anomaly_train.npy  shape (N,) float32   (0.0=normal, 1.0=malware)
  y_anomaly_val.npy
  y_anomaly_test.npy
  y_class_train.npy    shape (N,) int64     (0=normal,1=shellcode,...4=ransomware)
  y_class_val.npy
  y_class_test.npy

When --input has no .bin files, synthetic data is generated and saved to
ml/data/synthetic/ before processing.

CLI:
    python ml/pipeline/preprocess.py --input ml/data/raw/ --output ml/data/processed/
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "training"))

from model.features import extract as extract_features_from_page  # noqa: E402
from model.tcn import FEATURE_DIM, SEQUENCE_LENGTH  # noqa: E402

STRIDE = 8
_BIN_FRAME_SIZE = 4108   # 8 addr + 4 label_id + 4096 page
_BIN_HEADER_FMT = "<QI"

LABELS = ("normal", "malware", "shellcode", "rootkit", "cryptominer", "ransomware")

_LABEL_ANOMALY = {
    "normal": 0.0,
    "malware": 1.0, "shellcode": 1.0, "rootkit": 1.0,
    "cryptominer": 1.0, "ransomware": 1.0,
}
_LABEL_CLASS = {
    "normal": 0, "malware": 1, "shellcode": 1,
    "rootkit": 2, "cryptominer": 3, "ransomware": 4,
}


def load_raw(data_dir: str) -> tuple[list[np.ndarray], list[float], list[int]]:
    """Load all .bin files from data_dir/<label>/ for each label.

    Returns:
        features:       list of shape-(FEATURE_DIM,) float32 arrays (one per frame)
        anomaly_labels: list of float (0.0 or 1.0)
        class_labels:   list of int (0–4)
    """
    all_features: list[np.ndarray] = []
    all_anomaly: list[float] = []
    all_class: list[int] = []

    for label in LABELS:
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        files = sorted(f for f in os.listdir(label_dir) if f.endswith(".bin"))
        if not files:
            continue

        prev_feat: np.ndarray | None = None
        n_loaded = 0
        for fname in files:
            fpath = os.path.join(label_dir, fname)
            result = _parse_bin(fpath)
            if result is None:
                prev_feat = None
                continue
            page, addr = result
            feat = extract_features_from_page(page.tobytes(), addr, prev_feat)
            all_features.append(feat)
            all_anomaly.append(_LABEL_ANOMALY[label])
            all_class.append(_LABEL_CLASS[label])
            prev_feat = feat
            n_loaded += 1

        print(f"  {label}: {n_loaded} frames", flush=True)

    return all_features, all_anomaly, all_class


def _parse_bin(path: str) -> tuple[np.ndarray, int] | None:
    try:
        with open(path, "rb") as fh:
            raw = fh.read(_BIN_FRAME_SIZE)
        if len(raw) != _BIN_FRAME_SIZE:
            return None
        addr, _label_id = struct.unpack_from(_BIN_HEADER_FMT, raw, 0)
        page = np.frombuffer(raw[12:], dtype=np.uint8).copy()
        return page, addr
    except Exception:
        return None


def build_sequences(
    features: list[np.ndarray],
    anomaly_labels: list[float],
    class_labels: list[int],
    window: int = SEQUENCE_LENGTH,
    stride: int = STRIDE,
) -> tuple[list[np.ndarray], list[float], list[int]]:
    """Apply sliding window over feature sequence.

    Returns:
        windows:  list of shape-(FEATURE_DIM, window) arrays  (transposed for model input)
        win_anomaly: list of float
        win_class:   list of int
    """
    windows: list[np.ndarray] = []
    win_anomaly: list[float] = []
    win_class: list[int] = []

    n = len(features)
    for start in range(0, n - window + 1, stride):
        win = np.stack(features[start: start + window])  # (window, FEATURE_DIM)
        windows.append(win.T.astype(np.float32))          # (FEATURE_DIM, window)
        anom_chunk = anomaly_labels[start: start + window]
        win_anomaly.append(1.0 if sum(anom_chunk) / len(anom_chunk) >= 0.5 else 0.0)
        cls_chunk = class_labels[start: start + window]
        counts = np.bincount(cls_chunk, minlength=5)
        win_class.append(int(np.argmax(counts)))

    return windows, win_anomaly, win_class


def normalize(sequences: list[np.ndarray]) -> list[np.ndarray]:
    """Verify all values are in [0, 1]; clip any that are not."""
    clipped = 0
    result = []
    for seq in sequences:
        clipped_seq = np.clip(seq, 0.0, 1.0).astype(np.float32)
        if not np.array_equal(clipped_seq, seq):
            clipped += 1
        result.append(clipped_seq)
    if clipped:
        print(f"  normalize: clipped {clipped}/{len(sequences)} sequences to [0,1]", flush=True)
    return result


def save_processed(
    windows: list[np.ndarray],
    anomaly_labels: list[float],
    class_labels: list[int],
    output_dir: str,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> None:
    """Split and save arrays to output_dir as .npy files."""
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    X = np.array(windows, dtype=np.float32)           # (N, FEATURE_DIM, SEQ_LEN)
    y_a = np.array(anomaly_labels, dtype=np.float32)  # (N,)
    y_c = np.array(class_labels, dtype=np.int64)      # (N,)

    normal_idx  = np.where(y_a == 0.0)[0]
    malware_idx = np.where(y_a == 1.0)[0]

    def _split(idx):
        arr = idx.copy()
        rng.shuffle(arr)
        n = len(arr)
        n_tr = int(n * train_frac)
        n_va = int(n * val_frac)
        return arr[:n_tr], arr[n_tr: n_tr + n_va], arr[n_tr + n_va:]

    n_tr, n_va, n_te = _split(normal_idx)
    m_tr, m_va, m_te = _split(malware_idx)

    splits = {
        "train": np.concatenate([n_tr, m_tr]),
        "val":   np.concatenate([n_va, m_va]),
        "test":  np.concatenate([n_te, m_te]),
    }

    for split, idx in splits.items():
        rng.shuffle(idx)
        np.save(os.path.join(output_dir, f"X_{split}.npy"),          X[idx])
        np.save(os.path.join(output_dir, f"y_anomaly_{split}.npy"),  y_a[idx])
        np.save(os.path.join(output_dir, f"y_class_{split}.npy"),    y_c[idx])
        print(f"  {split}: {len(idx)} sequences "
              f"({(y_a[idx] == 0).sum()} normal, {(y_a[idx] == 1).sum()} malware)",
              flush=True)

    print(f"\nSaved processed data → {output_dir}", flush=True)


def _generate_synthetic(synthetic_dir: str) -> None:
    """Generate synthetic data and save as .npy frames to synthetic_dir."""
    print("No raw .bin frames found. Generating synthetic data…", flush=True)
    sys.path.insert(0, os.path.join(_REPO_ROOT, "training"))
    from synthetic import generate  # noqa: PLC0415
    generate(output_dir=synthetic_dir, n_normal=2000, n_malware=2000, seed=42)
    print(f"Synthetic data saved → {synthetic_dir}", flush=True)


def _load_npy_frames(
    data_dir: str,
) -> tuple[list[np.ndarray], list[float], list[int]]:
    """Load .npy frames generated by synthetic.py or analyzer --log-dir."""
    all_features: list[np.ndarray] = []
    all_anomaly: list[float] = []
    all_class: list[int] = []

    label_map = {
        "normal": (0.0, 0),
        "malware": (1.0, 1),
        "shellcode": (1.0, 1),
        "rootkit": (1.0, 2),
        "cryptominer": (1.0, 3),
        "ransomware": (1.0, 4),
    }

    for label, (anom, cls) in label_map.items():
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        files = sorted(f for f in os.listdir(label_dir) if f.endswith(".npy"))
        if not files:
            continue
        n_loaded = 0
        for fname in files:
            try:
                obj = np.load(os.path.join(label_dir, fname), allow_pickle=True).item()
                feat = np.asarray(obj["features"], dtype=np.float32)
                if feat.shape[0] < FEATURE_DIM:
                    feat = np.pad(feat, (0, FEATURE_DIM - feat.shape[0]))
                elif feat.shape[0] > FEATURE_DIM:
                    feat = feat[:FEATURE_DIM]
                # Use stored class label if available
                stored_cls = int(obj.get("class_label", -1))
                all_features.append(feat)
                all_anomaly.append(anom)
                all_class.append(stored_cls if stored_cls >= 0 else cls)
                n_loaded += 1
            except Exception:
                continue
        print(f"  {label}: {n_loaded} frames (npy)", flush=True)

    return all_features, all_anomaly, all_class


def preprocess(input_dir: str, output_dir: str, seed: int = 42) -> None:
    """Full preprocessing pipeline: raw → processed numpy arrays."""
    print(f"\n── Preprocessing ────────────────────────────────────")
    print(f"  Input : {input_dir}")
    print(f"  Output: {output_dir}\n")

    # Count .bin files across all label subdirectories
    total_bins = sum(
        sum(1 for f in os.listdir(os.path.join(input_dir, label)) if f.endswith(".bin"))
        for label in LABELS
        if os.path.isdir(os.path.join(input_dir, label))
    )

    if total_bins == 0:
        # Check collect_for_training/ as fallback
        fallback = os.path.join(_REPO_ROOT, "collect_for_training")
        fallback_bins = sum(
            sum(1 for f in os.listdir(os.path.join(fallback, label)) if f.endswith(".bin"))
            for label in LABELS
            if os.path.isdir(os.path.join(fallback, label))
        )
        if fallback_bins > 0:
            print(f"No .bin files in {input_dir}, using {fallback} ({fallback_bins} frames)",
                  flush=True)
            input_dir = fallback
            total_bins = fallback_bins

    if total_bins > 0:
        print(f"Loading {total_bins} .bin frames…", flush=True)
        features, anomaly_labels, class_labels = load_raw(input_dir)
    else:
        # Fall back to synthetic
        synthetic_dir = os.path.join(_REPO_ROOT, "ml", "data", "synthetic")
        _generate_synthetic(synthetic_dir)
        features, anomaly_labels, class_labels = _load_npy_frames(synthetic_dir)

    if not features:
        print("ERROR: No frames loaded. Aborting.", file=sys.stderr)
        sys.exit(1)

    print(f"\nBuilding windows (window={SEQUENCE_LENGTH}, stride={STRIDE})…", flush=True)
    windows, win_anomaly, win_class = build_sequences(
        features, anomaly_labels, class_labels
    )
    print(f"  {len(windows)} windows from {len(features)} frames", flush=True)

    windows = normalize(windows)

    print(f"\nSplitting and saving…", flush=True)
    save_processed(windows, win_anomaly, win_class, output_dir, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw frames → numpy arrays")
    parser.add_argument("--input", default=os.path.join(_REPO_ROOT, "ml", "data", "raw"),
                        help="Directory with label subdirs of .bin files")
    parser.add_argument("--output", default=os.path.join(_REPO_ROOT, "ml", "data", "processed"),
                        help="Output directory for .npy arrays")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    preprocess(args.input, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
