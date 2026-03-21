"""hypTcn training script — loads from ml/data/processed/ numpy arrays.

Usage:
    python ml/pipeline/train.py --data ml/data/processed/ --epochs 50
    python ml/pipeline/train.py --data ml/data/processed/ --epochs 5 --batch-size 16

Inputs (from --data directory):
    X_train.npy, X_val.npy, X_test.npy     shape (N, FEATURE_DIM, SEQUENCE_LENGTH)
    y_anomaly_train/val/test.npy            shape (N,) float32
    y_class_train/val/test.npy              shape (N,) int64

Outputs (to --output-dir, default ml/models/):
    tcn_weights.pt   best model state dict
    config.json      hyperparameters + training stats
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
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))

from model.tcn import (  # noqa: E402
    TCNAnomalyDetector, FEATURE_DIM, SEQUENCE_LENGTH, NUM_CLASSES
)

_DEFAULT_DATA   = os.path.join(_REPO_ROOT, "ml", "data", "processed")
_DEFAULT_OUTPUT = os.path.join(_REPO_ROOT, "ml", "models")


def _load_split(data_dir: str, split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X   = torch.from_numpy(np.load(os.path.join(data_dir, f"X_{split}.npy")))
    y_a = torch.from_numpy(np.load(os.path.join(data_dir, f"y_anomaly_{split}.npy")))
    y_c = torch.from_numpy(np.load(os.path.join(data_dir, f"y_class_{split}.npy")))
    return X, y_a, y_c


def _epoch_pass(
    model: nn.Module,
    loader: DataLoader,
    bce_crit: nn.Module,
    ce_crit: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for X, y_a, y_c in loader:
            X   = X.to(device)
            y_a = y_a.to(device)
            y_c = y_c.to(device)

            anomaly_logit, class_logits = model(X)
            bce_loss = bce_crit(anomaly_logit, y_a)
            ce_loss  = ce_crit(class_logits, y_c)
            loss     = bce_loss + ce_loss

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(y_a)
            preds   = (anomaly_logit >= 0.0).float()
            correct += (preds == y_a).sum().item()
            total   += len(y_a)

    return total_loss / max(total, 1), correct / max(total, 1)


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    thresholds = np.sort(np.unique(scores))[::-1]
    tprs = [0.0]
    fprs = [0.0]
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tprs.append(((pred == 1) & (labels == 1)).sum() / pos)
        fprs.append(((pred == 1) & (labels == 0)).sum() / neg)
    tprs.append(1.0)
    fprs.append(1.0)
    return float(np.trapezoid(tprs, fprs))


def train(
    data_dir: str = _DEFAULT_DATA,
    output_dir: str = _DEFAULT_OUTPUT,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    seed: int = 42,
) -> dict:
    """Train TCNAnomalyDetector on preprocessed numpy arrays."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # ── load data ────────────────────────────────────────────────────────────
    required = ["X_train.npy", "X_val.npy", "X_test.npy",
                "y_anomaly_train.npy", "y_anomaly_val.npy", "y_anomaly_test.npy",
                "y_class_train.npy", "y_class_val.npy", "y_class_test.npy"]
    missing = [f for f in required if not os.path.isfile(os.path.join(data_dir, f))]
    if missing:
        print(f"ERROR: Missing processed files in {data_dir!r}:", file=sys.stderr)
        for f in missing:
            print(f"  {f}", file=sys.stderr)
        print("Run: python ml/pipeline/preprocess.py first", file=sys.stderr)
        sys.exit(1)

    X_tr,  y_a_tr,  y_c_tr  = _load_split(data_dir, "train")
    X_va,  y_a_va,  y_c_va  = _load_split(data_dir, "val")
    X_te,  y_a_te,  y_c_te  = _load_split(data_dir, "test")

    n_normal_tr  = int((y_a_tr == 0).sum().item())
    n_malware_tr = int((y_a_tr == 1).sum().item())
    print(f"Train: {len(X_tr)} sequences ({n_normal_tr} normal, {n_malware_tr} malware)")
    print(f"Val:   {len(X_va)} sequences")
    print(f"Test:  {len(X_te)} sequences")

    train_loader = DataLoader(TensorDataset(X_tr, y_a_tr, y_c_tr),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_va, y_a_va, y_c_va),
                              batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_te, y_a_te, y_c_te),
                              batch_size=batch_size, shuffle=False)

    # ── model + optimiser ─────────────────────────────────────────────────────
    model       = TCNAnomalyDetector().to(device)
    bce_crit    = nn.BCEWithLogitsLoss()
    ce_crit     = nn.CrossEntropyLoss()
    optimizer   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "tcn_weights.pt")

    # ── training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    no_improve    = 0
    history: list[dict] = []
    print(f"\nTraining for up to {epochs} epochs (patience={patience})…\n", flush=True)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _epoch_pass(model, train_loader, bce_crit, ce_crit, optimizer, device)
        va_loss, va_acc = _epoch_pass(model, val_loader,   bce_crit, ce_crit, None,      device)
        scheduler.step(va_loss)

        improved = va_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = va_loss
            torch.save(model.state_dict(), weights_path)
            no_improve = 0
            marker = " *"
        else:
            no_improve += 1
            marker = ""

        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})
        elapsed = time.time() - t0
        eta     = elapsed / epoch * (epochs - epoch)
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train={tr_loss:.4f}  val={va_loss:.4f}  val_acc={va_acc:.3f}  "
            f"[{elapsed:.0f}s, ~{eta:.0f}s]{marker}",
            flush=True,
        )
        if tr_loss != tr_loss or va_loss != va_loss:
            print("NaN detected — stopping.")
            break
        if no_improve >= patience:
            print(f"\nEarly stop at epoch {epoch}.")
            break

    # ── test evaluation ────────────────────────────────────────────────────────
    print(f"\nLoading best weights from {weights_path}", flush=True)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    all_scores:    list[float] = []
    all_anomaly:   list[int]   = []
    all_pred_cls:  list[int]   = []
    all_true_cls:  list[int]   = []

    with torch.no_grad():
        for X, y_a, y_c in test_loader:
            anomaly_logit, class_logits = model(X.to(device))
            scores   = torch.sigmoid(anomaly_logit).cpu().numpy()
            pred_cls = class_logits.argmax(dim=-1).cpu().numpy()
            all_scores.extend(scores.tolist())
            all_anomaly.extend(y_a.int().tolist())
            all_pred_cls.extend(pred_cls.tolist())
            all_true_cls.extend(y_c.tolist())

    scores_np  = np.array(all_scores)
    anomaly_np = np.array(all_anomaly)
    preds      = (scores_np >= 0.5).astype(int)

    tp = int(((preds == 1) & (anomaly_np == 1)).sum())
    fp = int(((preds == 1) & (anomaly_np == 0)).sum())
    fn = int(((preds == 0) & (anomaly_np == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    cls_acc   = float(np.mean(np.array(all_pred_cls) == np.array(all_true_cls)))
    roc_auc   = _roc_auc(anomaly_np, scores_np)

    print(f"\n── Test Results ──────────────────────────────────")
    print(f"  anomaly_acc : {(preds == anomaly_np).mean():.4f}")
    print(f"  class_acc   : {cls_acc:.4f}")
    print(f"  F1          : {f1:.4f}")
    print(f"  ROC-AUC     : {roc_auc:.4f}")

    # ── save config.json ──────────────────────────────────────────────────────
    valid = [h for h in history if h["train_loss"] == h["train_loss"]]
    final = valid[-1] if valid else history[-1]
    config = {
        "model_version":   "2.0.0",
        "trained_at":      datetime.now(timezone.utc).isoformat(),
        "feature_dim":     FEATURE_DIM,
        "sequence_length": SEQUENCE_LENGTH,
        "filters":         32,
        "num_blocks":      3,
        "num_classes":     NUM_CLASSES,
        "dilations":       [1, 2, 4],
        "kernel_size":     3,
        "alert_threshold": 0.85,
        "activity_classes": ["normal", "shellcode", "rootkit", "cryptominer", "ransomware"],
        "training_stats": {
            "epochs":           final["epoch"],
            "final_train_loss": round(final["train_loss"], 6),
            "final_val_loss":   round(final["val_loss"], 6),
            "best_val_loss":    round(best_val_loss, 6),
            "class_accuracy":   round(cls_acc, 6),
            "f1_score":         round(f1, 6),
            "roc_auc":          round(roc_auc, 6),
        },
        "weights_path": "python/model/weights/tcn_weights.pt",
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2)

    print(f"\nSaved weights → {weights_path}")
    print(f"Saved config  → {config_path}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hypTcn on preprocessed data")
    parser.add_argument("--data",       default=_DEFAULT_DATA,   help="Directory with .npy arrays")
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT, help="Output directory for weights+config")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=10)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    train(
        data_dir=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
