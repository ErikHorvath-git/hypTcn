# Model Weights

Place trained TCN weights here as `tcn_weights.pt`.

## Expected shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| Input  | `(1, 18, 16)` | batch=1, FEATURE_DIM=18, SEQUENCE_LENGTH=16 |
| Output | `(1,)` float in [0, 1] | anomaly score; >0.85 triggers alert |

## Saving after training

```python
from model.tcn import save_model

# train your model...
save_model(trained_model, "../models/tcn_weights.pt")
```

## Loading

`load_model()` (called automatically by `analyzer.py` at startup) will load
`models/tcn_weights.pt` if it exists. If the file is absent, random weights
are used — all inference scores will be meaningless until the model is trained.

## Training data format

Frames are logged by `analyzer.py --log-dir <path> --label <normal|malware>`.
Each `.npy` file contains a dict with keys:
- `features`: shape `(18,)` float32 — the input vector
- `addr`: int — physical address
- `timestamp_ms`: int — Unix milliseconds
- `raw_page`: shape `(4096,)` uint8 — raw page bytes
