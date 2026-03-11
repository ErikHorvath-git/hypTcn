# hypTcn

Hypervisor-aware memory introspection with a Temporal Convolutional Network anomaly detector.

---

## 1. Project Overview

hypTcn samples raw 4 KiB physical memory pages from a running KVM guest — from outside the
VM, at the hypervisor boundary — and classifies each sliding window of 16 consecutive pages
as normal or anomalous using a PyTorch TCN.

**Why VMI + TCN?**

- **Zero footprint inside the guest.** No agent, no kernel module, no hooks inside the VM.
  The guest cannot detect or tamper with the observer because introspection runs entirely in
  the host process via libvmi → QEMU QMP.
- **Temporal patterns matter.** A single page snapshot is ambiguous; a sequence of 16 pages
  from the same physical region captures behavior over time — entropy drift, address-scanning
  patterns, NOP-sled presence across frames — which is what the TCN is designed to model.
- **APT detection goal.** Advanced persistent threats stage shellcode or packed loaders in
  memory regions that briefly look anomalous. Scanning at the hypervisor level and
  correlating over time (rather than inspecting syscalls, which are trivially spoofed)
  is a complementary detection layer that survives most in-guest evasion.

**Current state (tested 2026-03-11):**
- Live VM introspection works end-to-end against `hyptcn-guest` (Debian 12, QEMU/KVM).
- The model is trained on **synthetic** data. Live VM scores are 0.0 because real memory
  pages are not yet labeled. See §10 and §12.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  KVM Guest  (hyptcn-guest, Debian 12, 2 GiB RAM)                            │
│  Physical RAM — zero footprint, no agent, not detectable from inside         │
└───────────────────────┬──────────────────────────────────────────────────────┘
                        │  hypervisor boundary
                        │  libvmi vmi_read_pa()
                        │  KVM legacy driver → qemu:///session → QMP xp command
                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  C Extractor   internal/extractor/probe.c + probe.h                          │
│                                                                              │
│  hyptcn_vmi_open(vm_name)                                                    │
│    → vmi_init(&vmi, VMI_KVM, name, VMI_INIT_DOMAINNAME, NULL, NULL)          │
│       (no /etc/libvmi.conf — raw PA reads only, OS-layer init skipped)       │
│  hyptcn_read_page(handle, phys_addr, buf) → vmi_read_pa(…, 4096, …)         │
│  hyptcn_vmi_close(handle)  → vmi_destroy()                                  │
└───────────────────────┬──────────────────────────────────────────────────────┘
                        │  CGO
                        │  CGO_CFLAGS  = -I/usr/local/include
                        │  CGO_LDFLAGS = -L/usr/local/lib64 -lvmi
                        │               -Wl,-rpath,/usr/local/lib64
                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Go Orchestrator   internal/orchestrator/engine.go                           │
│                                                                              │
│  Engine.Stream(ctx, physAddr, interval)                                      │
│    → extractor.Open(vmName)   one persistent libvmi handle                   │
│    → ticker every interval ms                                                │
│         extractor.ReadPage(physAddr)  → 4096 bytes                          │
│         Engine.Analyze(ctx, addr, payload)                                   │
│           → lazy UDS connect (3 attempts, 1s back-off each)                 │
│           → write frame: [8B addr LE][4096B page] = 4104 bytes              │
│           → read newline-terminated JSON response                           │
│           → log via slog                                                    │
│    → Engine.Close() on exit: vmi_destroy + close UDS                        │
│                                                                              │
│  Mock mode (--mock): crypto/rand fills pages, addr += 0x1000 each frame     │
└───────────────────────┬──────────────────────────────────────────────────────┘
                        │  Unix Domain Socket  /tmp/hyptcn.sock
                        │
                        │  Frame wire format (4104 bytes, no delimiter):
                        │    bytes  0–7    physical address (uint64, little-endian)
                        │    bytes  8–4103 raw page content (4096 bytes)
                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Python Analyzer   python/analyzer.py                                        │
│                                                                              │
│  asyncio UDS server, one _handle() coroutine per Go connection               │
│                                                                              │
│  Per frame:                                                                  │
│    1. readexactly(8) + readexactly(4096)                                    │
│    2. features.extract(page, addr, prev_features) → (18,) float32           │
│    3. window.append(vec)   deque maxlen=16                                   │
│    4. if log_dir: np.save(<log_dir>/<label>/<ts>_<addr>.npy)                │
│    5. len(window) < 16 → socket: {"anomaly_score":0.0,"status":"warming_up"}│
│       len(window) = 16 → seq = stack(window) shape (18,16)                  │
│                           score = tcn.infer(model, seq)                     │
│                           socket: {"anomaly_score":<f>,"status":"ok"}       │
│                           stdout: {"timestamp":…,"addr":…,"score":…,        │
│                                   "alert":<score>0.85,"status":"ok"}        │
└───────────────────────┬──────────────────────────────────────────────────────┘
                        │  PyTorch inference  ~0.34 ms/window (CPU)
                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  TCN Model   python/model/tcn.py                                             │
│                                                                              │
│  Input  (1, 18, 16) — batch=1, features=18, time_steps=16                   │
│  Block 0  _TCNBlock(18→32, dilation=1)  + 1×1 projection residual           │
│  Block 1  _TCNBlock(32→32, dilation=2)  + identity residual                 │
│  Block 2  _TCNBlock(32→32, dilation=4)  + identity residual                 │
│  AdaptiveAvgPool1d(1) → Flatten → Linear(32→16,ReLU) → Linear(16→1)        │
│  sigmoid(logit) → anomaly score ∈ [0.0, 1.0]                               │
│                                                                              │
│  Weights: models/tcn_weights.pt   Config: models/config.json               │
└──────────────────────────────────────────────────────────────────────────────┘
```

**UDS retry logic:** `Engine.connect()` retries up to 3 times with 1-second gaps.
The connection is kept open across frames and only re-dialed after a failure.
Deadlines: dial 5s, write 10s, read 10s.

---

## 3. Components

### `internal/extractor/probe.h` + `probe.c`

C wrapper around libvmi. Opaque persistent handle.

| Function | What it does |
|---|---|
| `hyptcn_vmi_open(vm_name)` | `vmi_init(VMI_KVM, VMI_INIT_DOMAINNAME, NULL, NULL)`. KVM legacy driver via `qemu:///session`. Returns NULL on failure. |
| `hyptcn_read_page(handle, addr, buf)` | `vmi_read_pa` for exactly 4096 bytes. Returns 0 / -1 (bad args) / -2 (libvmi error). |
| `hyptcn_vmi_close(handle)` | `vmi_destroy` + free. NULL-safe. |

`vmi_init` (not `vmi_init_complete`) is used deliberately — raw PA reads require no
System.map, rekall profile, or `/etc/libvmi.conf`. Connects to → `extractor.go`.

---

### `internal/extractor/extractor.go`

CGO bridge.

```
cgo CFLAGS:  -I${SRCDIR}
cgo LDFLAGS: -lvmi   (Makefile overrides to /usr/local paths + rpath)
```

| Export | Description |
|---|---|
| `Open(vmName string) (*Handle, error)` | Caller must `defer h.Close()`. |
| `(*Handle).ReadPage(addr uint64) ([]byte, error)` | Always 4096 bytes on success. |
| `(*Handle).Close()` | Idempotent. |
| `PageSize int` | 4096. |

Connects to → `engine.go`.

---

### `internal/orchestrator/engine.go`

Runtime loop and UDS client.

| Item | Detail |
|---|---|
| `NewEngine(socketPath, vmName, mock, logger)` | nil logger → discard. Mock starts `nextMockAddr = 0x1000`. |
| `Stream(ctx, physAddr, interval)` | Opens VMI (non-mock), initial capture, ticker loop, `Close()` on return. |
| `Analyze(ctx, addr, payload)` | Frames 4104 bytes, sends, reads one JSON line. |
| `connect(ctx)` | Lazy. 3 attempts × 1s gap × 5s dial timeout. |
| `Close()` | VMI handle + UDS. |
| Mock mode | `crypto/rand.Read` for page bytes; addr walks +0x1000/frame. |

Connects to → `main.go` (called by) and → Python analyzer (UDS).

---

### `cmd/hyptcn/main.go`

Cobra CLI. `SIGINT`/`SIGTERM` via `signal.NotifyContext`. Logs to stderr via `log/slog`.

| Flag | Default | Description |
|---|---|---|
| `--socket` | `/tmp/hyptcn.sock` | UDS path |
| `--vm` | `guest` | libvmi domain name |
| `--address` | `0x1000` | Physical address to sample |
| `--interval` | `100` | Milliseconds between samples |
| `--mock` | `false` | Use random pages instead of libvmi |

Module: `github.com/example/hypTcn`. Cobra vendored under `third_party/`.

---

### `python/analyzer.py`

asyncio UDS server. One coroutine per accepted connection.

| Flag | Default | Description |
|---|---|---|
| `--socket` | `/tmp/hyptcn.sock` | UDS listen path |
| `--log-dir` | None | Save `.npy` frame files here |
| `--label` | `unknown` | Subdirectory: `normal`, `malware`, or custom |

**Startup:** removes stale socket, loads model + config.json once, prints version.

**Log file format:** `<log_dir>/<label>/<timestamp_ms>_<addr_hex16>.npy`
Dict keys: `features (18,)`, `addr int`, `timestamp_ms int`, `raw_page (4096,) uint8`.

Alert threshold: from `config.json`; fallback `0.85`.

---

### `python/model/features.py`

Stateless numpy feature extractor. All outputs clamped to `[0, 1]`.

```python
extract(page: bytes, address: int, prev_features: np.ndarray | None) -> np.ndarray  # (18,) float32
```

See §4 for the full feature table.

---

### `python/model/tcn.py`

PyTorch model + inference API.

| Export | Description |
|---|---|
| `TCNAnomalyDetector` | Model class. `forward(x)` → raw logit. |
| `load_model(weights_path)` | Reads config.json → builds matching arch → loads weights. Eval mode. |
| `save_model(model, path)` | `torch.save(state_dict)`. |
| `infer(model, window)` | Accepts `(18,16)` or `(16,18)`, applies sigmoid, returns `float`. |

---

### `python/model/__init__.py`

Re-exports: `TCNAnomalyDetector`, `load_model`, `save_model`, `infer`, `extract`,
`FEATURE_DIM=18`, `SEQUENCE_LENGTH=16`.

---

### `training/synthetic.py`

Generates labeled synthetic frame sequences — no live VM needed.

5 archetypes: **normal** (low entropy, null-heavy), **shellcode** (high entropy, NOP sled,
syscall opcodes), **packed_pe** (PE magic, high compression ratio), **cryptominer**
(near-perfect entropy, maximum unique bytes), **rootkit** (normal-looking entropy but
address-delta spikes simulating DKOM-style memory jumps).

```sh
.venv/bin/python training/synthetic.py [--output data] [--normal 2000] [--malware 2000] [--seed 42]
```

Output: `data/normal/` and `data/malware/`, each containing `n_sequences × 16` `.npy` files.

---

### `training/dataset.py`

`MemoryPageDataset`: loads `.npy` files, slides window=16 stride=8 over sorted filenames
per class, returns `(tensor(18,16), label_float)`.

`stratified_split(dataset, 0.70/0.15/0.15)` → `(train, val, test)` Subsets.

---

### `training/train.py`

```sh
.venv/bin/python training/train.py [--data-dir data] [--epochs 50] [--batch-size 32]
                                    [--lr 1e-3] [--output-dir models] [--seed 42]
```

Auto-generates synthetic data if directories are empty. `BCEWithLogitsLoss`, Adam,
`ReduceLROnPlateau(patience=5)`, grad clip norm=1.0, early stop patience=10.
Saves best `models/tcn_weights.pt` + `models/config.json` on completion.

**Must use `.venv/bin/python`** — system python3 does not have torch.

---

### `training/evaluate.py`

```sh
.venv/bin/python training/evaluate.py [--data-dir data] [--model-dir models]
                                        [--n-latency 1000] [--batch-size 32] [--seed 42]
```

Outputs: Accuracy / Precision / Recall / F1 / ROC-AUC; `models/confusion_matrix.png`
(requires matplotlib); latency benchmark with p50/p95/p99 percentiles.

---

## 4. Feature Vector (18 features)

All features are `float32` in `[0, 1]`. Index is fixed.

| # | Name | Formula | Malware detection relevance |
|---|---|---|---|
| 0 | `byte_entropy` | Shannon H / 8 | Packed/encrypted code → near 1.0 |
| 1 | `nonzero_ratio` | count(b≠0) / 4096 | Zero-padded pages → low |
| 2 | `printable_ratio` | count(0x20≤b≤0x7E) / 4096 | Binary code vs string data |
| 3 | `high_byte_ratio` | count(b>0x7F) / 4096 | Encoded/obfuscated content |
| 4 | `unique_bytes` | distinct byte values / 256 | Crypto buffers → near 1.0 |
| 5 | `top4_freq` | sum(top-4 byte counts) / 4096 | NOP sleds / zero pages → high |
| 6 | `zero_runs` | count(zero-runs>8) / 100 | Uninitialised pages → high |
| 7 | `addr_norm` | phys_addr / 0xFFFFFFFFFFFF | Kernel vs userspace position |
| 8 | `entropy_blocks_std` | std(H per 256B block) / 0.5 | Mixed page: header + payload |
| 9 | `compression_ratio` | zlib(page,1) size / 4096 | Low=repetitive; high=packed/crypto |
| 10 | `null_run_ratio` | bytes inside zero-runs>8 / 4096 | BSS / uninitialized → high |
| 11 | `pe_header_score` | 0.0 / 0.5 / 1.0 MZ+PE sig | Injected PE / reflective DLL load |
| 12 | `elf_header_score` | 0.0 / 0.5 / 1.0 \x7fELF | ELF mapped into guest memory |
| 13 | `syscall_pattern_count` | count(SYSCALL\|INT80\|SYSENTER) / 100 | Shellcode syscall density |
| 14 | `nop_sled_score` | bytes inside 0x90-runs>8 / 4096 | Classic NOP sled before shellcode |
| 15 | `string_density` | count(printable runs≥4) / 100 | C2 URLs / config strings |
| 16 | `entropy_delta` | (H[t]−H[t-1]+1)/2, clamped to [0,1] | Entropy spike/drop over time **(temporal)** |
| 17 | `addr_delta` | (addr[t]−addr[t-1]) / 0xFFFFFFFFFFFF | Non-sequential jumps **(temporal)** |

First frame: `entropy_delta = 0.5` (neutral), `addr_delta = 0.0`.

---

## 5. TCN Model Architecture

**Input:** `(1, 18, 16)` — batch=1, features=18, sequence=16
**Output:** sigmoid(logit) → score ∈ [0.0, 1.0]
**Alert threshold:** 0.85 (from `config.json`)

### `_CausalConv1d`

`padding = (kernel-1) × dilation`. Forward slices `output[:, :, :-padding]` to remove
future context. Strictly causal — valid for streaming inference.

### `_TCNBlock`

```
x (in_ch, T)
├─ residual: Identity if in_ch==out_ch, else Conv1d(in_ch, out_ch, 1)
├─ weight_norm(CausalConv1d(in_ch→32, k=3, dilation=d)) → ReLU → Dropout(0.1)
├─ weight_norm(CausalConv1d(32→32,    k=3, dilation=d)) → ReLU → Dropout(0.1)
└─ ReLU(conv_out + residual) → (out_ch, T)
```

### Full stack

```
(1, 18, 16)
  Block 0: _TCNBlock(18→32, dilation=1)   1×1 residual projection
  Block 1: _TCNBlock(32→32, dilation=2)   identity residual
  Block 2: _TCNBlock(32→32, dilation=4)   identity residual
  AdaptiveAvgPool1d(1)  →  (1, 32, 1)
  Flatten               →  (1, 32)
  Linear(32→16) + ReLU
  Linear(16→1)          →  raw logit
  sigmoid               →  score ∈ [0, 1]
```

### Receptive field

Each causal conv contributes `(3-1) × dilation` time steps:

```
Block 0: 2×1 + 2×1 =  4
Block 1: 2×2 + 2×2 =  8
Block 2: 2×4 + 2×4 = 16
                   ─────
Total RF = 1 + 28 = 29 time steps
```

RF (29) > sequence length (16): the last output timestep sees the entire input window.

### `models/config.json` (current weights)

```json
{
  "model_version": "1.0.0",
  "feature_dim": 18,  "sequence_length": 16,
  "filters": 32,  "num_blocks": 3,  "dilations": [1,2,4],  "kernel_size": 3,
  "alert_threshold": 0.85,
  "training_stats": {
    "accuracy": 0.998336,  "f1_score": 0.998333,  "roc_auc": 1.0
  },
  "dataset_stats": { "normal_samples": 3999, "malware_samples": 3999 }
}
```

These metrics are on **synthetic test data only** and do not reflect real-world performance.

---

## 6. Training Pipeline

### Step 1 — Generate synthetic data (skipped automatically if data/ is populated)

```sh
.venv/bin/python training/synthetic.py --output data --normal 2000 --malware 2000
# writes data/normal/ (32 000 files) and data/malware/ (32 000 files)
```

### Step 2 — Train

```sh
.venv/bin/python training/train.py \
    --data-dir data --output-dir models \
    --epochs 50 --batch-size 32 --lr 1e-3
```

Saves `models/tcn_weights.pt` on every validation improvement.
Writes `models/config.json` with architecture + training stats when done.

### Step 3 — Evaluate

```sh
.venv/bin/python training/evaluate.py --data-dir data --model-dir models --n-latency 1000
```

Outputs accuracy/F1/ROC-AUC table + inference latency benchmark.
`models/confusion_matrix.png` if matplotlib is installed.

### Verified output on synthetic data (2026-03-11)

```
Samples    : 1202 (601 normal, 601 malware)
Accuracy   : 0.9967   Precision: 1.0000   Recall: 0.9933
F1 Score   : 0.9967   ROC-AUC  : 1.0000
Confusion Matrix:
           Normal  Malware
Normal   :    601        0
Malware  :      4      597
Latency  : mean 0.34 ms, p95 0.37 ms, p99 0.39 ms  (CPU, batch=1)
```

---

## 7. Installation & Build

### Prerequisites (Fedora)

```sh
sudo dnf install -y gcc make golang \
    libvirt-devel json-c-devel glib2-devel \
    cmake bison flex autoconf automake libtool pkg-config
```

### libvmi from source — REQUIRED

The Fedora package (`dnf install libvmi`) is compiled **without KVM support** (Xen + file
drivers only). You must build from source:

```sh
git clone https://github.com/libvmi/libvmi.git ~/libvmi-src
cd ~/libvmi-src && mkdir build && cd build

cmake .. \
  -DENABLE_KVM=ON \
  -DENABLE_KVM_LEGACY=ON \
  -DENABLE_XEN=OFF \
  -DENABLE_FILE=ON \
  -DCMAKE_INSTALL_PREFIX=/usr/local

make -j$(nproc)
sudo make install
sudo ldconfig
```

**Verify:**
```sh
strings /usr/local/lib64/libvmi.so | grep -i kvm
# must print: VMI_KVM   VMI_INIT_DATA_KVMI_SOCKET
```

**Why `ENABLE_KVM_LEGACY=ON`?**
The new KVM driver requires a KVMI socket (patched QEMU, not upstream). The legacy driver
uses libvirt QMP `human-monitor-command xp` which works with stock QEMU.

**Why `qemu:///session`?**
Both KVM driver files hardcode `qemu:///system` upstream. The VM runs under the user's
session daemon (`qemu:///session`), so both `kvm.c` and `kvm_legacy.c` were patched
accordingly during the build.

**No `/etc/libvmi.conf` needed.** `probe.c` calls `vmi_init` (not `vmi_init_complete`),
skipping OS-layer init — only raw physical address reads are used.

### Build hyptcn

```sh
make
# → bin/hyptcn, linked via rpath to /usr/local/lib64/libvmi
```

The Makefile sets:
```makefile
LIBVMI_PREFIX     = /usr/local
CGO_CFLAGS_EXTRA  = -I$(LIBVMI_PREFIX)/include
CGO_LDFLAGS_EXTRA = -L$(LIBVMI_PREFIX)/lib64 -lvmi -Wl,-rpath,$(LIBVMI_PREFIX)/lib64
```

### Python venv

```sh
make deps
# creates .venv/, installs torch>=2.0, numpy>=1.26, matplotlib>=3.7
```

**Always use `.venv/bin/python`** for the analyzer and all training scripts.
System python3 does not have torch installed.

---

## 8. Usage Examples

### Mock mode (no KVM required)

```sh
# Terminal 1
cd python && ../.venv/bin/python analyzer.py --socket /tmp/hyptcn.sock

# Terminal 2
./bin/hyptcn --mock --interval 100
```

Go stderr output:
```
time=2026-03-11T22:26:07.701+01:00 level=INFO msg="starting scan" socket=/tmp/hyptcn.sock vm=guest address=4096 interval=150ms mock=true
time=2026-03-11T22:26:07.705+01:00 level=INFO msg="analysis complete" score=0 status=warming_up
...  (15 warming_up frames)
time=2026-03-11T22:26:09.962+01:00 level=INFO msg="analysis complete" score=1 status=ok
```

Analyzer stdout:
```json
{"status": "warming_up", "frames": 1}
...
{"status": "warming_up", "frames": 15}
{"timestamp": 1773264369956, "addr": "0x10000", "score": 1.0, "alert": true, "status": "ok"}
```

Note: mock mode scores are 1.0 because `crypto/rand` pages are maximum-entropy — the
synthetic-trained model correctly classifies them as malware-like (cryptominer / shellcode
archetype).

### Live VM introspection

```sh
virsh domstate hyptcn-guest       # must be: running

# Terminal 1
cd python && ../.venv/bin/python analyzer.py \
    --socket /tmp/hyptcn.sock \
    --log-dir /tmp/live_frames/ \
    --label normal

# Terminal 2
./bin/hyptcn --vm hyptcn-guest --address 0x1000000 --interval 500
```

Verified live output (2026-03-11, 24 frames):
```
time=2026-03-11T22:26:19.622+01:00 level=INFO msg="starting scan" vm=hyptcn-guest address=16777216 interval=500ms mock=false
time=...  level=INFO msg="analysis complete" score=0 status=warming_up   (×15)
time=...  level=INFO msg="analysis complete" score=0 status=ok           (×9)
time=2026-03-11T22:26:31.618+01:00 level=INFO msg="stopping scan loop"
```

Analyzer stdout:
```json
{"status": "warming_up", "frames": 1}
...
{"timestamp": 1773264387155, "addr": "0x1000000", "score": 0.0, "alert": false, "status": "ok"}
```

Score is 0.0 on a real idle VM because the kernel pages at `0x1000000` match the normal
synthetic archetype (low entropy, null-heavy). The model has never seen real malware.

### With --log-dir for data collection

```sh
cd python && ../.venv/bin/python analyzer.py \
    --socket /tmp/hyptcn.sock \
    --log-dir data/ \
    --label normal
# frames saved to data/normal/<timestamp_ms>_<addr_hex>.npy
```

### `make python-service`

```sh
make python-service
# equivalent to: cd python && .venv/bin/python analyzer.py
#                --socket /tmp/hyptcn.sock --log-dir /tmp/hyptcn-frames/
```

### Custom socket path

```sh
./bin/hyptcn --socket /run/user/1000/hyptcn.sock --vm hyptcn-guest --address 0x2000000
cd python && ../.venv/bin/python analyzer.py --socket /run/user/1000/hyptcn.sock
```

---

## 9. Dataset Collection

### Collecting normal traffic

```sh
# Start analyzer logging to data/normal/
cd python && ../.venv/bin/python analyzer.py \
    --socket /tmp/hyptcn.sock --log-dir data/ --label normal &

# Run scanner for hours while guest does normal work
./bin/hyptcn --vm hyptcn-guest --address 0x1000000 --interval 100
```

### Collecting malware traffic

Boot a **disposable snapshot** of the guest, inject the malware sample, then:

```sh
cd python && ../.venv/bin/python analyzer.py \
    --socket /tmp/hyptcn.sock --log-dir data/ --label malware &

./bin/hyptcn --vm hyptcn-guest --address 0x1000000 --interval 100
```

### Directory layout

```
data/
├── normal/
│   ├── 1741699200000_0000000001000000.npy
│   └── ...
└── malware/
    ├── 1741699300000_0000000001000000.npy
    └── ...
```

Each `.npy` contains: `{features:(18,) float32, addr:int, timestamp_ms:int, raw_page:(4096,) uint8}`.

### Retrain on real data

```sh
.venv/bin/python training/train.py --data-dir data --output-dir models --epochs 50
.venv/bin/python training/evaluate.py --data-dir data --model-dir models
# restart analyzer — it reloads models/tcn_weights.pt at startup
```

---

## 10. Current Status

### What works (verified 2026-03-11)

- [x] **Live VM introspection** — 24 frames captured from `hyptcn-guest` at `0x1000000`,
      stable 500ms cadence, zero dropped frames
- [x] **Full pipeline** — Go scanner → UDS → Python analyzer → TCN → JSON response
- [x] **Frame logging** — `.npy` files written during live session
- [x] **Mock mode** — fully functional for pipeline testing without a VM
- [x] **Training pipeline** — synthetic data, train, evaluate all work end-to-end
- [x] **Model loaded** — `models/tcn_weights.pt` exists; synthetic accuracy 99.8%, ROC-AUC 1.0
- [x] **libvmi KVM** — rebuilt from source with `ENABLE_KVM_LEGACY=ON`, `qemu:///session` patch
- [x] **Inference latency** — 0.34 ms mean (CPU, batch=1)

### What does not work yet

- [ ] **Real malware dataset** — none collected; model trained on synthetic data only
- [ ] **Live score meaning** — score=0.0 on idle Debian 12 VM; model needs real labeled data
- [ ] **Smoke tests** — no `make test`, no unit tests for features/wire protocol/training
- [ ] **Semantic gap** — raw PA only; no mapping of physical pages to processes or VA
- [ ] **Address sweep** — scanner samples one fixed address per run, not a range

---

## 11. Roadmap

### Infrastructure

- [x] 18-feature extractor with entropy_delta + addr_delta temporal features
- [x] `--log-dir` / `--label` frame logger for dataset collection
- [x] Training pipeline: synthetic.py, dataset.py, train.py, evaluate.py
- [x] Live KVM introspection via libvmi legacy driver + qemu:///session patch
- [ ] Smoke tests (`make test`): mock-mode end-to-end, feature extractor unit tests
- [ ] Address sweep mode: scan a configurable PA range per tick
- [ ] `confusion_matrix.png` — install matplotlib in venv (`pip install matplotlib`)
- [ ] Systemd unit files for both services

### Research (thesis work)

- [ ] Real malware dataset collection on live guests
- [ ] Retrain model on real data — synthetic ROC-AUC=1.0 is not informative
- [ ] Semantic gap: correlate PA → VA → PID via CR3 + page-table walking
- [ ] Process-level features: which PID owns an anomalous physical page
- [ ] Evaluate on real APT samples: Cobalt Strike beacon, reflective DLL injection,
      process hollowing, Diamorphine rootkit
- [ ] Compare TCN vs LSTM / attention baseline on the same feature set
- [ ] SIEM integration: structured alerts to syslog / Kafka / Elastic
- [ ] Multi-guest monitoring: fan out to N VMs from one analyzer

---

## 12. What Still Needs To Be Done

**No real malware dataset.** Every `.npy` in `data/` was produced by `synthetic.py` using
hand-crafted statistical distributions. The `accuracy=0.998` in `config.json` is a test-set
score on held-out synthetic data from the same generator. It measures nothing about
real-world detection.

**Score = 0.0 on live VM.** Idle Debian 12 kernel pages at `0x1000000` are low-entropy
and null-heavy — they match the normal synthetic archetype, so the model outputs 0.0.
Score = 1.0 in mock mode is the opposite effect: `crypto/rand` pages are maximum-entropy,
matching the cryptominer/shellcode archetypes. Both are expected; neither indicates that the
model would detect a real threat.

**No smoke tests.** There is no `make test`. The feature extractor, wire protocol, and
training pipeline have no automated tests. A regression would be silent.

**Semantic gap not solved.** `vmi_read_pa` returns raw bytes from a physical address.
The system cannot tell which process owns the page, nor which virtual address it maps to.
Closing the semantic gap requires OS-offset-aware page-table walking — for Linux this means
`task_struct` traversal using `vmi_init_complete` with a System.map or rekall profile,
which was intentionally bypassed to avoid the config dependency.

**Single fixed address.** The scanner samples one PA per run. A meaningful scan must sweep
a physical address range or focus on known-suspicious regions (guest heap, kernel code
sections of specific PIDs).

**`confusion_matrix.png` not generated.** matplotlib is not installed in the venv. Run
`pip install matplotlib` inside `.venv` or `make deps` after adding it to
`python/requirements.txt`.
