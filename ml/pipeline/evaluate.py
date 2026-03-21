"""Evaluate a trained hypTcn model on test data.

Usage:
    python ml/pipeline/evaluate.py
    python ml/pipeline/evaluate.py --data ml/data/processed/ --model-dir ml/models/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))

from model.tcn import TCNAnomalyDetector, load_model, FEATURE_DIM, SEQUENCE_LENGTH  # noqa: E402

_DEFAULT_DATA      = os.path.join(_REPO_ROOT, "ml", "data", "processed")
_DEFAULT_MODEL_DIR = os.path.join(_REPO_ROOT, "ml", "models")


def _roc_curve(labels: np.ndarray, scores: np.ndarray):
    thresholds = np.sort(np.unique(scores))[::-1]
    tprs, fprs = [0.0], [0.0]
    pos = labels.sum()
    neg = len(labels) - pos
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tprs.append(((pred == 1) & (labels == 1)).sum() / max(pos, 1))
        fprs.append(((pred == 1) & (labels == 0)).sum() / max(neg, 1))
    tprs.append(1.0)
    fprs.append(1.0)
    return np.array(fprs), np.array(tprs)


def evaluate(
    data_dir: str = _DEFAULT_DATA,
    model_dir: str = _DEFAULT_MODEL_DIR,
    n_latency: int = 1000,
    batch_size: int = 32,
    seed: int = 42,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load model ────────────────────────────────────────────────────────────
    config_path  = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "tcn_weights.pt")
    alert_threshold = 0.85

    if os.path.isfile(config_path):
        with open(config_path) as fh:
            config = json.load(fh)
        model = TCNAnomalyDetector(
            feature_dim=config.get("feature_dim", FEATURE_DIM),
            filters=config.get("filters", 32),
            kernel_size=config.get("kernel_size", 3),
            num_blocks=config.get("num_blocks", 3),
        )
        if os.path.isfile(weights_path):
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
        alert_threshold = config.get("alert_threshold", 0.85)
        ver = config.get("model_version", "?")
        ts  = config.get("trained_at", "unknown")
        print(f"Loaded model v{ver} trained {ts}")
    else:
        model = load_model(weights_path if os.path.isfile(weights_path) else "")
        print("No config.json; using defaults.")

    model = model.to(device)
    model.eval()

    # ── load test data ─────────────────────────────────────────────────────────
    x_path = os.path.join(data_dir, "X_test.npy")
    if not os.path.isfile(x_path):
        print(f"ERROR: {x_path} not found. Run preprocess.py first.", file=sys.stderr)
        sys.exit(1)

    X_te  = torch.from_numpy(np.load(os.path.join(data_dir, "X_test.npy")))
    y_a   = torch.from_numpy(np.load(os.path.join(data_dir, "y_anomaly_test.npy")))
    y_c   = torch.from_numpy(np.load(os.path.join(data_dir, "y_class_test.npy")))
    loader = DataLoader(TensorDataset(X_te, y_a, y_c), batch_size=batch_size, shuffle=False)
    print(f"Test set: {len(X_te)} sequences")

    # ── inference ─────────────────────────────────────────────────────────────
    all_scores: list[float] = []
    all_labels: list[int]   = []
    all_pred_cls: list[int] = []
    all_true_cls: list[int] = []

    with torch.no_grad():
        for X, lab_a, lab_c in loader:
            anomaly_logit, class_logits = model(X.to(device))
            scores   = torch.sigmoid(anomaly_logit).cpu().numpy()
            pred_cls = class_logits.argmax(dim=-1).cpu().numpy()
            all_scores.extend(scores.tolist())
            all_labels.extend(lab_a.int().tolist())
            all_pred_cls.extend(pred_cls.tolist())
            all_true_cls.extend(lab_c.tolist())

    scores_np = np.array(all_scores)
    labels_np = np.array(all_labels)
    preds_np  = (scores_np >= 0.5).astype(int)

    # ── metrics ───────────────────────────────────────────────────────────────
    cm = np.zeros((2, 2), dtype=int)
    for true, pred in zip(labels_np, preds_np):
        cm[int(true), int(pred)] += 1
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    accuracy  = (tp + tn) / max(len(labels_np), 1)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    cls_acc   = float(np.mean(np.array(all_pred_cls) == np.array(all_true_cls)))
    fprs, tprs = _roc_curve(labels_np, scores_np)
    roc_auc   = float(np.trapezoid(tprs, fprs))

    print("\n── Evaluation Results ────────────────────────────")
    print(f"  Samples      : {len(labels_np)} "
          f"({(labels_np == 0).sum()} normal, {(labels_np == 1).sum()} malware)")
    print(f"  Threshold    : {alert_threshold}")
    print(f"  Accuracy     : {accuracy:.4f}")
    print(f"  Precision    : {precision:.4f}")
    print(f"  Recall       : {recall:.4f}")
    print(f"  F1 Score     : {f1:.4f}")
    print(f"  Class Acc    : {cls_acc:.4f}")
    print(f"  ROC-AUC      : {roc_auc:.4f}")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"             Normal  Malware")
    print(f"  Normal   : {tn:6d}  {fp:7d}")
    print(f"  Malware  : {fn:6d}  {tp:7d}")

    # ── confusion matrix plot ─────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Malware"])
        ax.set_yticklabels(["Normal", "Malware"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        out_path = os.path.join(model_dir, "confusion_matrix.png")
        plt.savefig(out_path, dpi=120); plt.close()
        print(f"\n  Confusion matrix → {out_path}")
    except ImportError:
        print("\n  (matplotlib not installed; plot skipped)")

    # ── latency benchmark ─────────────────────────────────────────────────────
    print(f"\n── Latency Benchmark ({n_latency} runs, batch=1) ────────────")
    dummy = torch.randn(1, FEATURE_DIM, SEQUENCE_LENGTH).to(device)
    for _ in range(50):
        with torch.no_grad():
            model(dummy)

    latencies: list[float] = []
    for _ in range(n_latency):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(dummy)
        latencies.append((time.perf_counter() - t0) * 1000)

    lat = np.array(latencies)
    print(f"  Mean : {lat.mean():.3f} ms  |  p50 : {np.percentile(lat,50):.3f} ms  "
          f"|  p99 : {np.percentile(lat,99):.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hypTcn model")
    parser.add_argument("--data",       default=_DEFAULT_DATA,      help="Processed .npy directory")
    parser.add_argument("--model-dir",  default=_DEFAULT_MODEL_DIR, help="Model weights+config directory")
    parser.add_argument("--n-latency",  type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()
    evaluate(args.data, args.model_dir, args.n_latency, args.batch_size, args.seed)


if __name__ == "__main__":
    main()
