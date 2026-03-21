# hypTcn — Hypervisor-Aware TCN Security Monitor

Samples 4 KiB physical memory pages from a running KVM guest via libvmi and feeds them to a PyTorch Temporal Convolutional Network (TCN) for real-time anomaly detection and activity classification across five threat classes.

---

## Architecture

```
  KVM Guest (physical memory)
       │  libvmi
       ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  C extractor (internal/extractor/probe.c)                    │
  │  hyptcn_read_page()  hyptcn_get_process_list()              │
  │  hyptcn_get_kernel_modules()  hyptcn_get_network_connections()│
  └──────────────┬───────────────────────────────────────────────┘
                 │  CGO bridge (internal/extractor/extractor.go)
                 ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  Go orchestrator (internal/orchestrator/engine.go)           │
  │  - Ticker-based sampling at --interval ms                    │
  │  - OS-layer scans every --proc-interval frames               │
  │  - Mock mode: crypto/rand pages (no VM needed)               │
  │  - Collect mode: writes .bin frames; bypasses Python         │
  └──────────────┬───────────────────────────────────────────────┘
                 │  UDS /tmp/hyptcn.sock  (4112 bytes/frame)
                 │  [8B addr][4B mod_norm][4B conn_norm][4096B page]
                 ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  Python analyzer (python/analyzer.py)                        │
  │  - asyncio UDS server                                        │
  │  - 20-feature extraction per frame (python/model/features.py)│
  │  - 16-frame sliding window                                   │
  │  - TCN inference (python/model/tcn.py)                       │
  │  - JSON response on socket + human-readable stdout           │
  └──────────────────────────────────────────────────────────────┘

  ML Pipeline (ml/pipeline/):
  collect.py → ml/data/raw/<label>/*.bin
  preprocess.py → ml/data/processed/{X,y}_*.npy
  train.py → ml/models/{tcn_weights.pt,config.json}
  export.py → python/model/weights/ + models/
```

---

## Quick Start

```sh
# 1. Install system deps (Fedora)
sudo dnf install make gcc go python3 libvmi-devel

# 2. Build binary + Python venv
make
make deps

# 3. Start Python analyzer
make python-service
# OR manually: cd python && ../.venv/bin/python analyzer.py

# 4. Run Go scanner (mock — no VM required)
./bin/hyptcn --mock --interval 100

# 5. Run full ML pipeline (collect → train → export)
./ml/pipeline/run_pipeline.sh --collect-duration 60 --epochs 20

# 6. Restart analyzer to pick up new weights
make python-service
```

---

## All CLI Flags

### bin/hyptcn

| Flag | Default | Description |
|---|---|---|
| `--vm <name>` | `""` | KVM domain name; unused in mock mode |
| `--socket <path>` | `/tmp/hyptcn.sock` | UDS path for Python analyzer |
| `--sysmap <path>` | `""` | System.map; enables OS-layer features |
| `--address <hex>` | `0x1000000` | Physical address in live mode |
| `--interval <ms>` | `100` | Milliseconds between samples |
| `--proc-interval <n>` | `100` | Frames between OS-layer scans |
| `--mock` | `false` | Generate crypto/rand pages; skip libvmi |
| `--collect` | `false` | Write .bin files; skip Python |
| `--collect-label <label>` | `normal` | One of: normal, malware, shellcode, rootkit, cryptominer, ransomware |
| `--collect-duration <sec>` | `300` | Collection duration (0=unlimited) |
| `--collect-dir <path>` | `collect_for_training/` | Root directory for .bin output |
| `--log-level <level>` | `info` | slog level: debug, info, warn, error |
| `--json` | `true` | Print live scores as JSON to stdout |
| `--quiet` | `false` | Suppress all stdout JSON |

### python/analyzer.py

| Flag | Default | Description |
|---|---|---|
| `--socket <path>` | `/tmp/hyptcn.sock` | UDS path to listen on |
| `--log-dir <path>` | None | Save frames as .npy to `<path>/<label>/` |
| `--label <str>` | `unknown` | Subdirectory for --log-dir |

### ml/pipeline/run_pipeline.sh

| Flag | Default | Description |
|---|---|---|
| `--vm <name>` | (mock) | KVM domain; omit for mock mode |
| `--collect-duration <sec>` | `300` | Collection duration per label |
| `--interval <ms>` | `100` | Sampling interval |
| `--sysmap <path>` | `""` | System.map path |
| `--epochs <n>` | `50` | Training epochs |
| `--batch-size <n>` | `32` | Batch size |
| `--skip-collect` | — | Skip collection step |
| `--skip-train` | — | Skip training step |

---

## ML Pipeline

Full pipeline in one command:
```sh
./ml/pipeline/run_pipeline.sh --vm hyptcn-guest --collect-duration 300
```

Or step by step:
```sh
# 1. Collect 5 labels (mock mode; use --vm for real VM)
for LABEL in normal shellcode rootkit cryptominer ransomware; do
  ./bin/hyptcn --mock --collect --collect-label $LABEL \
    --collect-duration 60 --collect-dir ml/data/raw/
done

# 2. Preprocess raw frames → numpy arrays
python ml/pipeline/preprocess.py --input ml/data/raw/ --output ml/data/processed/

# 3. Train
python ml/pipeline/train.py --data ml/data/processed/ --epochs 50

# 4. Evaluate
python ml/pipeline/evaluate.py --data ml/data/processed/ --model-dir ml/models/

# 5. Export weights to analyzer
python ml/pipeline/export.py

# 6. Run analyzer with new weights
make python-service
```

When `ml/data/raw/` has no .bin files, `preprocess.py` auto-generates synthetic
data (saves to `ml/data/synthetic/`) and uses that for training.

---

## Feature Vector (20 features, indices 0–19)

| # | Name | Description | Range |
|---|---|---|---|
| 0 | `byte_entropy` | Shannon entropy of page bytes | [0,1] |
| 1 | `nonzero_ratio` | Fraction of non-zero bytes | [0,1] |
| 2 | `printable_ratio` | Fraction of printable ASCII bytes | [0,1] |
| 3 | `high_byte_ratio` | Fraction of bytes ≥ 0x80 | [0,1] |
| 4 | `unique_bytes` | Count of distinct byte values / 256 | [0,1] |
| 5 | `top4_freq` | Sum of top-4 byte frequencies | [0,1] |
| 6 | `zero_runs` | Fraction of bytes in zero-runs > 8 | [0,1] |
| 7 | `addr_norm` | Physical address / 48-bit max | [0,1] |
| 8 | `entropy_blocks_std` | Std-dev of 16×256B block entropies (norm) | [0,1] |
| 9 | `compression_ratio` | zlib compressed size / 4096 | [0,1] |
| 10 | `null_run_ratio` | Fraction in zero-runs > 8 bytes | [0,1] |
| 11 | `pe_header_score` | PE magic strength: 0.0/0.5/1.0 | {0,.5,1} |
| 12 | `elf_header_score` | ELF magic strength: 0.0/0.5/1.0 | {0,.5,1} |
| 13 | `syscall_pattern_count` | x86 syscall opcodes per page / 100 | [0,1] |
| 14 | `nop_sled_score` | Fraction of bytes in 0x90-runs > 8 | [0,1] |
| 15 | `string_density` | Printable-ASCII strings ≥ 4 chars / 100 | [0,1] |
| 16 | `entropy_delta` | Entropy change T→T-1, shifted to [0,1] | [0,1] |
| 17 | `addr_delta` | Physical address jump, clamped to [0,1] | [0,1] |
| 18 | `kernel_module_count_norm` | Loaded kernel modules / 200 | [0,1] |
| 19 | `network_conn_count_norm` | Active TCP connections / 100 | [0,1] |

Features 16–17 are 0.5 (neutral) on the first frame. Features 18–19 are 0.0 in mock/raw mode.

---

## Data Formats

### UDS frame: Go → Python (4112 bytes)

```
Offset  Size  Type          Description
0       8     uint64 LE     physical address
8       4     float32 LE    kernel_module_count_norm
12      4     float32 LE    network_conn_count_norm
16      4096  uint8[]       raw page data
```

### .bin collection file (4108 bytes)

```
Offset  Size  Type          Description
0       8     uint64 LE     physical address
8       4     uint32 LE     label_id (0=normal,1=malware,2=shellcode,3=rootkit,4=cryptominer,5=ransomware)
12      4096  uint8[]       raw page data
```

### Python → Go socket response (newline-delimited JSON)

Warming up (first 15 frames):
```json
{"anomaly_score": 0.0, "status": "warming_up"}
```
Live (frame 16+):
```json
{"anomaly_score": 0.512, "activity_class": "normal", "status": "ok"}
```

### models/config.json schema

```json
{
  "model_version":   "2.0.0",
  "feature_dim":     20,
  "sequence_length": 16,
  "filters":         32,
  "num_blocks":      3,
  "num_classes":     5,
  "kernel_size":     3,
  "alert_threshold": 0.85,
  "activity_classes": ["normal","shellcode","rootkit","cryptominer","ransomware"],
  "weights_path":    "python/model/weights/tcn_weights.pt"
}
```

### Processed numpy array shapes

| File | Shape | dtype |
|---|---|---|
| `X_{split}.npy` | (N, 20, 16) | float32 |
| `y_anomaly_{split}.npy` | (N,) | float32 |
| `y_class_{split}.npy` | (N,) | int64 |

---

## Current Status

| Component | Status |
|---|---|
| Go binary (mock + collect mode) | Working |
| Go binary (live VM + sysmap) | Implemented; untested against real VM |
| Python analyzer (feature extraction) | Working |
| Python analyzer (TCN inference) | Working with random or trained weights |
| TCN model (dual-head) | Working |
| Collection pipeline (Go → .bin) | Working |
| ML preprocess pipeline | Working |
| ML train pipeline | Working |
| ML export pipeline | Working |
| evaluate.py (training/) | Working (3 bugs fixed) |
| OS-layer: process list | Implemented; profile-dependent |
| OS-layer: kernel modules | Implemented; profile-dependent |
| OS-layer: network connections | Offsets hardcoded for Linux 5.15 (wrong on 6.x) |
| Real malware dataset | Not collected; synthetic fallback only |

---

## Known Issues

1. **TCP offsets kernel-version dependent** — `probe.c:354–358` has hardcoded Linux 5.15 x86-64 field offsets. Wrong on Linux 6.x. Fix: use a Rekall/Volatility profile.

2. **OS features always 0.0 in mock/raw mode** — Features 18–19 are 0.0 when OS layer is inactive. A model trained on synthetic data with non-zero OS features will behave differently in production.

3. **No real malware data** — Training uses synthetic data generated by `training/synthetic.py` (now via `preprocess.py` fallback). Classification performance on real workloads is unvalidated.

4. **Cobra stub** — `third_party/cobra` is a minimal 43-line stub. Not a full Cobra implementation; no auto-generated help, no subcommands.

5. **Python service must start before Go** — `engine.go:connect()` retries 3× with 1s delay. No pre-flight check; failures degrade silently.

---

## Thesis Requirements Coverage

| Requirement | Status | Gap |
|---|---|---|
| Live memory introspection via libvmi | Implemented | No confirmed integration test |
| Physical page sampling at configurable interval | Done | — |
| OS-layer: process list | Done | Profile-dependent offsets |
| OS-layer: kernel modules | Done | Profile-dependent offsets |
| OS-layer: network connections | Done | Offsets wrong for Linux 6.x |
| 20-feature extraction | Done | — |
| TCN with dual heads (anomaly + classification) | Done | — |
| 5-class activity classification | Done | — |
| Training pipeline | Done | Synthetic data only |
| Dataset collection pipeline | Done | No real malware collected |
| Evaluation script | Done (fixed) | Only synthetic data evaluated |
| Alert threshold | 0.85 hardcoded | Not validated on real traffic |
| Process hiding detection | Done | No validation test |
| Virtual-to-physical translation | Done (C+Go) | Not called from orchestrator |
| Unit/integration tests | **None** | Zero test files |
| Real malware dataset | **None** | All synthetic |
