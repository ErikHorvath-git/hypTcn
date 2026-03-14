"""hypTcn model evaluation script.

Usage (from repo root):
    python training/evaluate.py
    python training/evaluate.py --data-dir data/ --model-dir models/ --n-latency 1000

Outputs:
    - Accuracy, Precision, Recall, F1, ROC-AUC table to stdout
    - models/confusion_matrix.png (requires matplotlib; skipped if absent)
    - Inference latency benchmark (mean ± std over N runs)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))
sys.path.insert(0, _TRAINING_DIR)

from model.tcn import TCNAnomalyDetector, load_model  # noqa: E402
from dataset import MemoryPageDataset, stratified_split  # noqa: E402


def _load_config(model_dir: str) -> dict | None:
    path = os.path.join(model_dir, "config.json")
    if os.path.isfile(path):
        with open(path) as fh:
            return json.load(fh)
    return None


def _confusion_matrix(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Return 2×2 confusion matrix [[TN, FP], [FN, TP]]."""
    cm = np.zeros((2, 2), dtype=int)
    for true, pred in zip(labels, preds):
        cm[int(true), int(pred)] += 1
    return cm


def _roc_curve(labels: np.ndarray, scores: np.ndarray):
    thresholds = np.sort(np.unique(scores))[::-1]
    tprs, fprs = [0.0], [0.0]
    pos = labels.sum()
    neg = len(labels) - pos
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        tprs.append(tp / max(pos, 1))
        fprs.append(fp / max(neg, 1))
    tprs.append(1.0)
    fprs.append(1.0)
    return np.array(fprs), np.array(tprs)


def evaluate(
    data_dir: str = "data",
    model_dir: str = "models",
    n_latency: int = 1000,
    batch_size: int = 32,
    seed: int = 42,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load model ─────────────────────────────────────────────────────────────
    config = _load_config(model_dir)
    weights_path = os.path.join(model_dir, "tcn_weights.pt")

    if config:
        model = TCNAnomalyDetector(
            feature_dim=config.get("feature_dim", 18),
            filters=config.get("filters", 32),
            kernel_size=config.get("kernel_size", 3),
            num_blocks=config.get("num_blocks", 3),
        )
        if os.path.isfile(weights_path):
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
        alert_threshold = config.get("alert_threshold", 0.85)
        ts = config.get("trained_at", "unknown")
        ver = config.get("model_version", "?")
        print(f"Loaded model v{ver} trained {ts}")
    else:
        model = load_model(weights_path)
        alert_threshold = 0.85
        print("No config.json found; using defaults.")

    model = model.to(device)
    model.eval()

    # ── load test data ─────────────────────────────────────────────────────────
    dataset = MemoryPageDataset(data_dir)
    if len(dataset) == 0:
        print(f"ERROR: No sequences found in {data_dir!r}. Run train.py first.")
        sys.exit(1)

    _, _, test_set = stratified_split(dataset, seed=seed)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    print(f"Test set: {len(test_set)} sequences")

    # ── inference ──────────────────────────────────────────────────────────────
    all_scores: list[float] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for seqs, labels in test_loader:
            scores = model(seqs.to(device)).cpu().numpy()
            all_scores.extend(scores.tolist())
            all_labels.extend(labels.int().tolist())

    scores_np = np.array(all_scores)
    labels_np = np.array(all_labels)
    preds_np = (scores_np >= 0.5).astype(int)

    # ── metrics ───────────────────────────────────────────────────────────────
    cm = _confusion_matrix(labels_np, preds_np)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    accuracy = (tp + tn) / max(len(labels_np), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    fprs, tprs = _roc_curve(labels_np, scores_np)
    roc_auc = float(np.trapezoid(tprs, fprs))

    print("\n── Evaluation Results ────────────────────────────")
    print(f"  Samples    : {len(labels_np)} "
          f"({(labels_np == 0).sum()} normal, {(labels_np == 1).sum()} malware)")
    print(f"  Threshold  : {alert_threshold}")
    print(f"  Accuracy   : {accuracy:.4f}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"  ROC-AUC    : {roc_auc:.4f}")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"             Normal  Malware")
    print(f"  Normal   : {tn:6d}  {fp:7d}")
    print(f"  Malware  : {fn:6d}  {tp:7d}")

    # ── confusion matrix plot ─────────────────────────────────────────────────
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        classes = ["Normal", "Malware"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        out_path = os.path.join(model_dir, "confusion_matrix.png")
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"\n  Confusion matrix saved → {out_path}")
    except ImportError:
        print("\n  (matplotlib not installed; confusion matrix plot skipped)")

    # ── latency benchmark ─────────────────────────────────────────────────────
    print(f"\n── Latency Benchmark ({n_latency} runs, batch=1) ────────────")
    dummy = torch.randn(1, 18, 16).to(device)
    # warm up
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
    print(f"  Mean   : {lat.mean():.3f} ms")
    print(f"  Std    : {lat.std():.3f} ms")
    print(f"  p50    : {np.percentile(lat, 50):.3f} ms")
    print(f"  p95    : {np.percentile(lat, 95):.3f} ms")
    print(f"  p99    : {np.percentile(lat, 99):.3f} ms")
    print(f"  Min    : {lat.min():.3f} ms")
    print(f"  Max    : {lat.max():.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hypTcn model")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--n-latency", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        n_latency=args.n_latency,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
