"""Export trained weights and config to python/model/weights/.

Copies:
  ml/models/tcn_weights.pt  →  python/model/weights/tcn_weights.pt
  ml/models/config.json     →  python/model/weights/config.json

Also syncs to models/ (repo root) for backward compatibility with training/evaluate.py.

CLI:
    python ml/pipeline/export.py
    python ml/pipeline/export.py --model-dir ml/models/ --weights-dir python/model/weights/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DEFAULT_MODEL_DIR   = os.path.join(_REPO_ROOT, "ml", "models")
_DEFAULT_WEIGHTS_DIR = os.path.join(_REPO_ROOT, "python", "model", "weights")
_LEGACY_DIR          = os.path.join(_REPO_ROOT, "models")


def export(
    model_dir: str = _DEFAULT_MODEL_DIR,
    weights_dir: str = _DEFAULT_WEIGHTS_DIR,
) -> None:
    """Copy weights + config from model_dir to weights_dir (and legacy models/)."""
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(_LEGACY_DIR, exist_ok=True)

    weights_src = os.path.join(model_dir, "tcn_weights.pt")
    config_src  = os.path.join(model_dir, "config.json")

    if not os.path.isfile(weights_src):
        print(f"ERROR: {weights_src} not found. Run train.py first.")
        raise FileNotFoundError(weights_src)
    if not os.path.isfile(config_src):
        print(f"ERROR: {config_src} not found. Run train.py first.")
        raise FileNotFoundError(config_src)

    weights_dst = os.path.join(weights_dir, "tcn_weights.pt")
    config_dst  = os.path.join(weights_dir, "config.json")

    shutil.copy2(weights_src, weights_dst)
    shutil.copy2(config_src,  config_dst)

    # Also keep models/ in sync (legacy path used by training/evaluate.py)
    shutil.copy2(weights_src, os.path.join(_LEGACY_DIR, "tcn_weights.pt"))
    shutil.copy2(config_src,  os.path.join(_LEGACY_DIR, "config.json"))

    with open(config_dst) as fh:
        cfg = json.load(fh)

    ver = cfg.get("model_version", "?")
    print(f"Exported model v{ver} to {weights_dir}/")
    print(f"  {weights_dst}")
    print(f"  {config_dst}")
    print(f"  {os.path.join(_LEGACY_DIR, 'tcn_weights.pt')}  (legacy sync)")
    print("\nRun: make && make python-service")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model weights to python/model/weights/")
    parser.add_argument("--model-dir",   default=_DEFAULT_MODEL_DIR)
    parser.add_argument("--weights-dir", default=_DEFAULT_WEIGHTS_DIR)
    args = parser.parse_args()
    export(args.model_dir, args.weights_dir)


if __name__ == "__main__":
    main()
