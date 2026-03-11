"""hypTcn training script.

Usage (from repo root):
    python training/train.py
    python training/train.py --epochs 30 --batch-size 64
    python training/train.py --data-dir /path/to/data --output-dir models/

If data/normal/ and data/malware/ are empty, synthetic data is generated
automatically via training/synthetic.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure python/ is on sys.path so we can import model.*
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))

from model.tcn import TCNAnomalyDetector, FEATURE_DIM, SEQUENCE_LENGTH  # noqa: E402

_TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))


def _ensure_data(data_dir: str) -> None:
    """Auto-generate synthetic data if the data directories are empty."""
    normal_dir = os.path.join(data_dir, "normal")
    malware_dir = os.path.join(data_dir, "malware")

    has_normal = os.path.isdir(normal_dir) and any(
        f.endswith(".npy") for f in os.listdir(normal_dir)
    ) if os.path.isdir(normal_dir) else False

    has_malware = os.path.isdir(malware_dir) and any(
        f.endswith(".npy") for f in os.listdir(malware_dir)
    ) if os.path.isdir(malware_dir) else False

    if not has_normal or not has_malware:
        print(f"No data found in {data_dir!r}. Generating synthetic dataset...")
        sys.path.insert(0, _TRAINING_DIR)
        from synthetic import generate  # noqa: PLC0415
        generate(output_dir=data_dir, n_normal=2000, n_malware=2000, seed=42)


def _epoch_pass(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """Run one epoch; return (avg_loss, accuracy)."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for seqs, labels in loader:
            seqs = seqs.to(device)          # (B, 18, 16)
            labels = labels.to(device)      # (B,)

            logits = model(seqs)            # (B,) raw logits
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            # threshold at logit=0.0 (equivalent to sigmoid(0)=0.5)
            preds = (logits >= 0.0).float()
            correct += (preds == labels).sum().item()
            total += len(labels)

    return total_loss / max(total, 1), correct / max(total, 1)


def train(
    data_dir: str = "data",
    output_dir: str = "models",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    seed: int = 42,
) -> dict:
    """Train TCNAnomalyDetector; save weights and config.json to output_dir."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── data ──────────────────────────────────────────────────────────────────
    _ensure_data(data_dir)

    # Import here so _ensure_data can run first
    sys.path.insert(0, _TRAINING_DIR)
    from dataset import MemoryPageDataset, stratified_split  # noqa: PLC0415

    dataset = MemoryPageDataset(data_dir)
    print(dataset.summary())

    if len(dataset) == 0:
        raise RuntimeError(f"No valid sequences found in {data_dir!r}.")

    train_set, val_set, test_set = stratified_split(dataset, seed=seed)
    print(f"Split: {len(train_set)} train / {len(val_set)} val / {len(test_set)} test")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=0)

    # ── model ─────────────────────────────────────────────────────────────────
    model = TCNAnomalyDetector().to(device)
    criterion = nn.BCEWithLogitsLoss()  # numerically stable fused sigmoid+BCE
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "tcn_weights.pt")

    # ── training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    no_improve = 0
    history: list[dict] = []

    print(f"\nTraining for up to {epochs} epochs (early stop patience={patience})...\n")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _epoch_pass(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = _epoch_pass(model, val_loader, criterion, None, device)
        scheduler.step(val_loss)

        improved = val_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = val_loss
            torch.save(model.state_dict(), weights_path)
            no_improve = 0
            marker = " *"
        else:
            no_improve += 1
            marker = ""

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         "train_acc": train_acc, "val_acc": val_acc})

        elapsed = time.time() - t0
        eta = elapsed / epoch * (epochs - epoch)
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.3f}  "
            f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s left]{marker}",
            flush=True,
        )

        if train_loss != train_loss or val_loss != val_loss:  # NaN check
            print(f"\nNaN detected at epoch {epoch} — reverting to best checkpoint.")
            break

        if no_improve >= patience:
            print(f"\nEarly stop at epoch {epoch} (no val improvement for {patience} epochs).")
            break

    # ── test evaluation ────────────────────────────────────────────────────────
    print(f"\nLoading best weights from {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))

    test_loss, test_acc = _epoch_pass(model, test_loader, criterion, None, device)

    # collect predictions for F1/AUC
    model.eval()
    all_scores: list[float] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for seqs, labels in test_loader:
            logits = model(seqs.to(device))
            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.extend(scores.tolist())
            all_labels.extend(labels.int().tolist())

    all_scores_np = np.array(all_scores)
    all_labels_np = np.array(all_labels)
    preds = (all_scores_np >= 0.5).astype(int)

    # F1
    tp = int(((preds == 1) & (all_labels_np == 1)).sum())
    fp = int(((preds == 1) & (all_labels_np == 0)).sum())
    fn = int(((preds == 0) & (all_labels_np == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    # ROC-AUC (trapezoid)
    roc_auc = _roc_auc(all_labels_np, all_scores_np)

    valid_history = [h for h in history if h["train_loss"] == h["train_loss"]]  # exclude NaN
    final = valid_history[-1] if valid_history else history[-1]
    best_epoch = min(valid_history or history, key=lambda h: h["val_loss"])

    print(f"\n── Test Results ──────────────────────────────────")
    print(f"  loss     = {test_loss:.4f}")
    print(f"  accuracy = {test_acc:.4f}")
    print(f"  F1       = {f1:.4f}")
    print(f"  ROC-AUC  = {roc_auc:.4f}")
    print(f"  best val_loss epoch {best_epoch['epoch']}: {best_val_loss:.4f}")

    # ── save config.json ──────────────────────────────────────────────────────
    dataset_stats = {
        "normal_samples": dataset.n_normal,
        "malware_samples": dataset.n_malware,
        "train_split": 0.70,
        "val_split": 0.15,
        "test_split": 0.15,
    }
    config = {
        "model_version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_dim": FEATURE_DIM,
        "sequence_length": SEQUENCE_LENGTH,
        "filters": 32,
        "num_blocks": 3,
        "dilations": [1, 2, 4],
        "kernel_size": 3,
        "alert_threshold": 0.85,
        "training_stats": {
            "epochs": final["epoch"],
            "final_train_loss": round(final["train_loss"], 6),
            "final_val_loss": round(final["val_loss"], 6),
            "best_val_loss": round(best_val_loss, 6),
            "accuracy": round(test_acc, 6),
            "f1_score": round(f1, 6),
            "roc_auc": round(roc_auc, 6),
        },
        "dataset_stats": dataset_stats,
        "weights_path": "models/tcn_weights.pt",
    }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2)
    print(f"\nSaved weights → {weights_path}")
    print(f"Saved config  → {config_path}")

    return config


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC via trapezoid rule (no sklearn required)."""
    thresholds = np.sort(np.unique(scores))[::-1]
    tprs = [0.0]
    fprs = [0.0]
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5

    for thresh in thresholds:
        pred = (scores >= thresh).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        tprs.append(tp / pos)
        fprs.append(fp / neg)
    tprs.append(1.0)
    fprs.append(1.0)

    return float(np.trapezoid(tprs, fprs))


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train hypTcn anomaly detector")
    parser.add_argument("--data-dir", default="data",
                        help="root data directory (default: data/)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", default="models",
                        help="directory for weights + config (default: models/)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
