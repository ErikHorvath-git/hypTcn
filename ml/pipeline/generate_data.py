"""Generate synthetic processed training data directly.

Writes ml/data/processed/{X,y}_*.npy with correct shapes for train.py.
"""

from __future__ import annotations
import os, sys
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "training"))
sys.path.insert(0, os.path.join(_REPO, "python"))

from synthetic import generate as _gen_frames  # noqa
from model.tcn import FEATURE_DIM, SEQUENCE_LENGTH  # noqa

STRIDE = 8
OUT = os.path.join(_REPO, "ml", "data", "processed")

_ANOMALY = {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

def _build_windows(feats, labels_a, labels_c):
    X, ya, yc = [], [], []
    for start in range(0, len(feats) - SEQUENCE_LENGTH + 1, STRIDE):
        window = np.stack(feats[start:start+SEQUENCE_LENGTH], axis=1)  # (20, 16)
        X.append(window)
        ya.append(labels_a[start])
        yc.append(labels_c[start])
    return np.array(X, dtype=np.float32), np.array(ya, dtype=np.float32), np.array(yc, dtype=np.int64)

def main(n_normal=2000, n_malware=2000, seed=42):
    syn_dir = os.path.join(_REPO, "ml", "data", "synthetic")
    print("Generating synthetic frames...")
    _gen_frames(output_dir=syn_dir, n_normal=n_normal, n_malware=n_malware, seed=seed)

    # load all frames
    label_map = {"normal":(0.0,0),"shellcode":(1.0,1),"rootkit":(1.0,2),"cryptominer":(1.0,3),"ransomware":(1.0,4)}
    all_f, all_a, all_c = [], [], []
    for label, (anom, cls) in label_map.items():
        d = os.path.join(syn_dir, label)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(".npy"))
        for fname in files:
            obj = np.load(os.path.join(d, fname), allow_pickle=True).item()
            feat = np.asarray(obj["features"], dtype=np.float32)
            all_f.append(feat); all_a.append(anom); all_c.append(cls)
    print(f"Loaded {len(all_f)} frames total")

    # shuffle
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(all_f))
    all_f = [all_f[i] for i in idx]
    all_a = [all_a[i] for i in idx]
    all_c = [all_c[i] for i in idx]

    # build windows
    X, ya, yc = _build_windows(all_f, all_a, all_c)
    print(f"Windows: {len(X)}, shape {X.shape}")

    # split 70/15/15
    n = len(X)
    n_tr = int(n * 0.70)
    n_va = int(n * 0.15)

    splits = {"train": (0, n_tr), "val": (n_tr, n_tr+n_va), "test": (n_tr+n_va, n)}
    os.makedirs(OUT, exist_ok=True)
    for split, (s, e) in splits.items():
        np.save(f"{OUT}/X_{split}.npy",         X[s:e])
        np.save(f"{OUT}/y_anomaly_{split}.npy",  ya[s:e])
        np.save(f"{OUT}/y_class_{split}.npy",    yc[s:e])
        nm = int((ya[s:e]==0).sum()); mal = int((ya[s:e]==1).sum())
        print(f"  {split}: {e-s} sequences (normal={nm} malware={mal})")

    print(f"\nSaved → {OUT}")

if __name__ == "__main__":
    main()
