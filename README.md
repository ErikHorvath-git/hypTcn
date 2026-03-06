# hypTCn

## Project Overview

hypTCn is a hypervisor-aware security toolkit for live memory introspection on KVM guests. It uses Virtual Machine Introspection (VMI) via LibVMI to sample raw 4 KiB physical memory pages from a running virtual machine — entirely from outside the guest, with zero footprint inside the monitored system. Each sampled page is reduced to an 18-element semantic feature vector (16 per-page features + 2 temporal delta features) and fed into a sliding window of 16 frames. When the window is full, a PyTorch Temporal Convolutional Network (TCN) produces an anomaly score in [0.0, 1.0]. The goal is to detect Advanced Persistent Threats and malware — shellcode, rootkits, cryptominers — by observing physical memory patterns that are invisible to in-guest detection (which an attacker can disable) but unavoidable at the hypervisor boundary.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  KVM Guest (monitored, zero-footprint)                              │
│  Physical RAM ──────────────────────────────────────────────────┐   │
└─────────────────────────────────────────────────────────────────│───┘
                         Hypervisor Boundary                      │
                    (LibVMI crosses here via KVM API)             │
                                                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Go CLI  cmd/hyptcn/main.go                                         │
│  Flags: --vm, --address, --interval, --socket, --mock               │
│  Signal handling: SIGINT/SIGTERM → graceful shutdown                │
└────────────────────────────┬────────────────────────────────────────┘
                             │ calls
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Go Orchestrator  internal/orchestrator/engine.go                   │
│  Engine.Stream()  →  ticker loop at --interval ms                   │
│  Engine.captureAndAnalyze()                                         │
│    mock=false: calls extractor.ExtractPage() → LibVMI               │
│    mock=true:  crypto/rand fills 4096 bytes, sequential addresses   │
│  Engine.connect()  →  UDS dial with 3 retries × 1s delay           │
└────────────────────────────┬────────────────────────────────────────┘
                             │ CGO call (non-mock path)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  C Extractor  internal/extractor/                                   │
│  extractor.go: CGO bridge, exposes ExtractPage(vmName, addr)        │
│  probe.c:      vmi_init(VMI_KVM, VMI_INIT_DOMAINNAME)              │
│                vmi_read_pa(vmi, physical_address, 4096, buf)        │
│                vmi_destroy(vmi)                                     │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             │ Frame written to UDS socket
                             │ ┌────────────────────────────────────┐
                             │ │  [8 bytes: uint64 LE address]      │
                             │ │  [4096 bytes: raw page data]       │
                             │ │  ─────────────────────────────     │
                             │ │  Total: 4104 bytes per frame       │
                             │ └────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Python Analyzer  analyzer.py                                       │
│  asyncio UDS server on /tmp/hyptcn.sock                             │
│  _iter_frames(): readexactly(8) + readexactly(4096) per frame       │
│  Sliding window: deque(maxlen=16) of feature vectors                │
│    frames < 16: emit {"status":"warming_up","frames":n}             │
│    frames = 16: run TCN inference                                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │ calls
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Feature Extractor  feature_extractor.py                            │
│  extract(page, address, prev_features) → np.ndarray shape (18,)     │
│  18 semantic features, all normalized to [0.0, 1.0]                │
└────────────────────────────┬────────────────────────────────────────┘
                             │ (18,) vector appended to window
                             │ when window full: np.stack → (18, 16)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TCN Model  tcn_model.py                                            │
│  Input: (1, 18, 16)  — batch × features × sequence                 │
│  3 × TCNBlock (dilations 1, 2, 4) → GlobalAvgPool → Dense → Sigmoid│
│  infer() → float score in [0.0, 1.0]                               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
          JSON to stdout + socket response
          {"timestamp":…, "addr":"0x…", "score":0.496, "alert":false}
```

**Frame format:** `[8B physical address, little-endian uint64][4096B raw page] = 4104 bytes`

**Sliding window:** 16 frames → 1 TCN inference. The window is a `deque(maxlen=16)`; new frames push old ones out. The TCN sees a short-term temporal trace of how a memory region evolves over time.

**Hypervisor boundary concept:** LibVMI uses the KVM API from the host side to read guest physical memory without injecting any agent or kernel module into the guest. The guest cannot detect or interfere with the observation. A rootkit that hides from the guest OS's `/proc` remains fully visible as physical memory pages.

**UDS retry logic:** `Engine.connect()` retries up to 3 times with a 1-second delay between attempts, logging each failure. This lets you start the Go binary before the Python service is fully up. After 3 failures it exits with an error rather than hanging.

---

## Components

### `cmd/hyptcn/main.go`

Entry point. Registers the Cobra CLI command and flags, installs a `signal.NotifyContext` for SIGINT/SIGTERM, constructs an `orchestrator.Engine`, and calls `engine.Stream()`. Translates the `--interval` integer (milliseconds) into a `time.Duration`.

**Key items:** `newRootCmd()`, `runScan()`, global flag vars (`socketPath`, `vmName`, `targetAddress`, `intervalMs`, `mockMode`).

**Connects to:** `internal/orchestrator/engine.go` via `orchestrator.NewEngine()`.

---

### `internal/orchestrator/engine.go`

The main control loop. `Engine.Stream()` fires a ticker at the configured interval and calls `captureAndAnalyze()` each tick. In real mode it calls `extractor.ExtractPage()`; in mock mode it fills a buffer from `crypto/rand` and increments `nextMockAddr` by `0x1000` per frame. `Analyze()` frames the page as `[8B addr][4096B data]`, writes it to the persistent UDS connection, then blocks on `reader.ReadBytes('\n')` waiting for the JSON response. Deadlines are applied per-operation (10s write, 10s read). `connect()` manages the single persistent connection with 3-attempt retry.

**Key items:** `Engine` struct, `Stream()`, `Analyze()`, `captureAndAnalyze()`, `connect()`, `writeAll()`, `prediction` struct (`anomaly_score`, `status`).

**Connects to:** `internal/extractor/extractor.go` (page reads) and the Python analyzer over UDS.

---

### `internal/extractor/extractor.go`

CGO bridge. Exposes `ExtractPage(vmName string, address uint64) ([]byte, error)` to Go. Allocates a 4096-byte buffer, calls `C.fetch_guest_page()`, and returns the buffer or an error code. The `PageSize` constant is derived directly from the C header (`HYPTCN_PAGE_SIZE = 4096`).

**Key items:** `ExtractPage()`, `PageSize`.

**Connects to:** `probe.c` via CGO; called by `engine.go`.

---

### `internal/extractor/probe.c` + `probe.h`

C wrapper around LibVMI. `fetch_guest_page(vm_name, physical_address, buffer)`:
1. Validates non-null arguments.
2. Calls `vmi_init(&vmi, VMI_KVM, vm_name, VMI_INIT_DOMAINNAME, NULL, NULL)` — connects to the KVM guest by domain name.
3. Calls `vmi_read_pa(vmi, physical_address, 4096, buffer, NULL)` — reads exactly one physical page.
4. Calls `vmi_destroy(vmi)` — releases the VMI instance.
5. Returns `0` on success, `-1` (init error) or `-2` (read error) on failure.

`probe.h` defines `HYPTCN_PAGE_SIZE 4096` and the function prototype, guarded by `extern "C"` for C++ compatibility.

**Key items:** `fetch_guest_page()`, `HYPTCN_INIT_ERROR (-1)`, `HYPTCN_READ_ERROR (-2)`, `HYPTCN_PAGE_SIZE (4096)`.

**Connects to:** LibVMI system library (`-lvmi`); called by `extractor.go`.

---

### `analyzer.py`

The active analyzer service. An asyncio UDS server that accepts one persistent connection from the Go binary per run. `_iter_frames()` reads frames with `readexactly(8)` + `readexactly(4096)`, unpacking the address with `struct.unpack("<Q", header)`. Per frame, it calls `feature_extractor.extract()`, appends the result to a `deque(maxlen=16)`, and either emits a warming-up response or stacks the deque into a `(16, 16)` array and runs `tcn_model.infer()`. Both the socket response and stdout receive the same JSON. The model is loaded once at startup via `tcn_model.load_model()`.

**Key items:** `_iter_frames()`, `_handle()`, `_serve()`, `ALERT_THRESHOLD = 0.85`, `FRAME_SIZE = 4104`, `SEQUENCE_LENGTH = 16`.

**Connects to:** `feature_extractor.py` and `tcn_model.py`; responds to `engine.go` over UDS.

---

### `feature_extractor.py`

Pure Python/NumPy feature extraction. `extract(page, address)` converts a 4096-byte page into a shape-`(16,)` float32 array. All values are normalized to `[0.0, 1.0]`. Helper functions are private (`_`-prefixed) and operate on NumPy arrays for performance. `_run_boundaries()` is the shared primitive underlying all run-length based features, using sentinel padding and `np.diff` to find run starts and ends without Python loops over bytes.

**Key items:** `extract()`, `FEATURE_DIM = 16`, `_run_boundaries()`, `_bytes_in_runs()`, `_block_entropy()`, `_pe_header_score()`, `_elf_header_score()`, `_syscall_pattern_count()`, `_count_strings()`.

**Connects to:** called by `analyzer.py`; produces input for `tcn_model.infer()`.

---

### `tcn_model.py`

Standalone PyTorch model. `TCNAnomalyDetector` stacks three `_TCNBlock` instances, applies global average pooling, then two linear layers with sigmoid output. `load_model()` returns an eval-mode instance, loading weights from `models/tcn_weights.pt` if the file exists. `infer(model, window)` accepts a `(16, 16)` NumPy array (or transposed), unsqueezes to `(1, 16, 16)`, runs the forward pass under `torch.no_grad()`, and returns a Python float.

**Key items:** `TCNAnomalyDetector`, `_TCNBlock`, `_CausalConv1d`, `load_model()`, `infer()`, `WEIGHTS_PATH`, `FEATURE_DIM = 16`, `SEQUENCE_LENGTH = 16`, `FILTERS = 32`.

**Connects to:** called by `analyzer.py`; weights loaded from `models/tcn_weights.pt`.

---

### `analyzer_service/` (legacy prototype)

An earlier iteration of the analyzer, retained for reference. Uses 256-bin byte-frequency histograms as features (instead of 16 semantic features), a 4-block TCN with 256 hidden channels, and returns `{"anomaly_score": float, "status": "clean"|"suspicious"|"warming_up"}`. The active pipeline is `analyzer.py` + `feature_extractor.py` + `tcn_model.py`. The `analyzer_service` package is not used in the current run path.

**Files:** `server.py`, `processor.py`, `model/architecture.py`, `model/layers.py`, `model/constants.py`, `model/__init__.py`.

---

### `third_party/github.com/spf13/cobra/`

Minimal vendored stub of the Cobra CLI library. Implements `Command` (with `Execute()`, `Flags()`, `RunE`) and `FlagSet` wrapping Go's stdlib `flag.FlagSet`. Only the flag types used by `main.go` are present: `StringVar`, `Uint64Var`, `IntVar`, `BoolVar`, `DurationVar`. The Go module uses a `replace` directive to point at this local copy instead of fetching Cobra from the network.

---

### `Makefile`

Three targets: `all` builds `bin/hyptcn` with `CGO_LDFLAGS="-lvmi"` and a local GOCACHE/GOPATH under `.cache/`; `deps` creates `.venv` and installs `analyzer_service/requirements.txt`; `python-service` runs the legacy `analyzer_service.server`. `clean` removes `bin/`, `.venv/`, `.cache/`.

---

## Feature Vector (18 features per frame)

Each 4096-byte page and its physical address produce one float32 vector of shape `(18,)`. All values are in `[0.0, 1.0]`.

| # | Feature | Formula / Source | Malware Detection Relevance |
|---|---------|------------------|-----------------------------|
| 0 | `byte_entropy` | Shannon H(page) / 8 | Packed/encrypted payloads have H ≈ 1.0; zero-filled pages have H ≈ 0.0; code has H ≈ 0.6–0.8 |
| 1 | `nonzero_ratio` | count(byte ≠ 0) / 4096 | Unallocated or BSS pages are mostly zero; active code/heap pages are not |
| 2 | `printable_ratio` | count(0x20 ≤ b ≤ 0x7E) / 4096 | High in string-heavy pages (config, paths); low in binary code or crypto |
| 3 | `high_byte_ratio` | count(b > 0x7F) / 4096 | High in encrypted blobs or multi-byte encodings; distinguishes from ASCII code |
| 4 | `unique_bytes` | count(distinct byte values) / 256 | Random or encrypted data uses all 256 values; NOP sleds and zero-runs do not |
| 5 | `top4_freq` | sum(4 largest byte counts) / 4096 | NOP sleds, zero pages, or single-byte XOR ciphers show extreme dominance of one byte |
| 6 | `zero_runs` | count(zero-runs > 8 bytes) / 100 | Long zero runs indicate uninitialized memory or padding; their temporal change is meaningful |
| 7 | `addr_norm` | physical_address / 0xFFFFFFFFFFFF | Physical address position correlates with page type (low phys = kernel, high = userspace heap) |
| 8 | `entropy_blocks_std` | std(H per 256B block) / 0.5 | High std means a mixed page: e.g. a plaintext header followed by an encrypted payload |
| 9 | `compression_ratio` | len(zlib(page, level=1)) / 4096 | Low = repetitive (NOP sled, zeros); high = random/compressed (crypto, packed PE) |
| 10 | `null_run_ratio` | bytes_in(zero-runs > 8) / 4096 | Distinct from `zero_runs`: measures coverage not count; large BSS regions score high |
| 11 | `pe_header_score` | 0.0 / 0.5 / 1.0 — MZ magic + PE\0\0 at e_lfanew | Detects injected PE images or reflective DLL loading in guest RAM |
| 12 | `elf_header_score` | 0.0 / 0.5 / 1.0 — \x7fELF magic + valid EI_CLASS | Detects ELF binaries mapped into memory; useful for Linux userspace injection |
| 13 | `syscall_pattern_count` | count(0F 05 \| CD 80 \| 0F 34) / 100 | SYSCALL, INT 80h, SYSENTER opcodes; dense syscall sequences are a shellcode indicator |
| 14 | `nop_sled_score` | bytes_in(0x90-runs > 8) / 4096 | Classic shellcode delivery mechanism; strong indicator of exploit staging in memory |
| 15 | `string_density` | count(printable runs ≥ 4 bytes) / 100 | Dense string presence = config data, C2 URLs, commands; sparse = pure code or crypto |
| 16 | `entropy_delta` | (H_t − H_{t-1}) clamped to [−1,1], shifted to [0,1] via (val+1)/2; first frame = 0.5 | Sudden entropy spikes (decompression, decryption) or drops (zeroing) are strong temporal indicators that a static per-frame view cannot capture |
| 17 | `addr_delta` | (addr_t − addr_{t-1}) / 0xFFFFFFFFFFFF, clamped to [0,1]; first frame = 0.0 | Large address jumps between samples reveal scanner pattern or attacker jumping between memory regions; sequential scans stay near 0 |

---

## TCN Model Architecture

**File:** `tcn_model.py` — **Class:** `TCNAnomalyDetector`

```
Input:  (batch=1, features=18, sequence=16)
         └─ 18 feature dimensions × 16 frames in the sliding window

Block 0 — dilation=1  in_ch=18 → out_ch=32
Block 1 — dilation=2  in_ch=32 → out_ch=32
Block 2 — dilation=4  in_ch=32 → out_ch=32

Each _TCNBlock:
  conv1: weight_norm(CausalConv1d(in_ch, 32, kernel=3, dilation=d))
  ReLU → Dropout(0.1)
  conv2: weight_norm(CausalConv1d(32, 32, kernel=3, dilation=d))
  ReLU → Dropout(0.1)
  residual: Identity() if in_ch==32, else Conv1d(in_ch, 32, kernel=1)
  output: ReLU(conv_out + residual)

CausalConv1d (_CausalConv1d):
  padding = (kernel_size - 1) × dilation
  output = conv(x)[:, :, :-padding]   →  strictly causal, no future leakage

After 3 blocks:
  AdaptiveAvgPool1d(1)    →  (batch, 32, 1)
  Flatten()               →  (batch, 32)
  Linear(32, 16) + ReLU   →  (batch, 16)
  Linear(16,  1) + Sigmoid →  (batch,  1)
  squeeze(-1)             →  (batch,)  ∈ [0.0, 1.0]
```

**Effective receptive field:** With dilations 1, 2, 4 and kernel size 3, each block adds `(3−1) × dilation` steps of context. The combined receptive field covers the full 16-frame window.

**Weight normalization:** Applied to both convolutions in every block via `torch.nn.utils.weight_norm`, decoupling weight magnitude from direction to stabilize training.

**Weights:** Saved to / loaded from `models/tcn_weights.pt` using `torch.save(model.state_dict(), …)` / `torch.load(…, weights_only=True)`. When the file does not exist, weights are random (PyTorch default initialization). The model produces valid scores with random weights — they just carry no meaningful signal until training.

**Alert threshold:** `score > 0.85` triggers `"alert": true` in the JSON output.

---

## Installation & Build

### 1. System dependencies

```sh
# Fedora / RHEL
sudo dnf install make gcc go python3 libvmi-devel

# Debian / Ubuntu
sudo apt install make gcc golang python3 python3-venv libvmi-dev
```

Your user must be in the `kvm` group to access guest memory:

```sh
sudo usermod -aG kvm $USER   # log out and back in after this
```

LibVMI requires a guest configuration in `/etc/libvmi/libvmi.conf`:

```
<vm_name> {
    ostype = "Linux";
}
```

### 2. Build the Go binary

```sh
make
# outputs bin/hyptcn
# uses CGO_LDFLAGS="-lvmi"; requires libvmi-devel headers
```

### 3. Install Python dependencies

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# installs torch>=2.0, numpy>=1.26
```

### 4. Run in mock mode (no KVM guest required)

Start the analyzer first, then the scanner in a second terminal:

```sh
# Terminal 1 — analyzer
.venv/bin/python analyzer.py

# Terminal 2 — scanner (mock frames, 100 ms interval)
./bin/hyptcn --mock --interval 100
```

### 5. Run against a real VM

```sh
# Terminal 1
.venv/bin/python analyzer.py --socket /tmp/hyptcn.sock

# Terminal 2
./bin/hyptcn --vm myguest --address 0x1000000 --interval 500
```

`--vm` must match the domain name in `libvmi.conf` and the output of `virsh list`.
`--address` is a guest physical address. Useful starting points: `0x1000` (first usable page), or addresses obtained from the host's `/proc/iomem` or a guest memory map.

---

## Usage Examples

### Mock mode — full end-to-end without a VM

```sh
.venv/bin/python analyzer.py &
./bin/hyptcn --mock --interval 100
```

**Analyzer stdout — warm-up phase (frames 1–15):**

```json
{"status": "warming_up", "frames": 1}
{"status": "warming_up", "frames": 2}
{"status": "warming_up", "frames": 3}
...
{"status": "warming_up", "frames": 15}
```

**Analyzer stdout — live scores (frame 16 onward):**

```json
{"timestamp": 1772759581713, "addr": "0x10000", "score": 0.496012, "alert": false}
{"timestamp": 1772759581804, "addr": "0x11000", "score": 0.496010, "alert": false}
{"timestamp": 1772759581903, "addr": "0x12000", "score": 0.496006, "alert": false}
```

**Go scanner stderr (structured slog output):**

```
time=2026-03-06T02:01:38.258+01:00 level=INFO msg="starting scan" socket=/tmp/hyptcn.sock vm=guest address=4096 interval=100ms mock=true
time=2026-03-06T02:01:38.260+01:00 level=INFO msg="analysis complete" score=0 status=warming_up
time=2026-03-06T02:01:39.814+01:00 level=INFO msg="analysis complete" score=0.498 status=
```

### Custom socket path

```sh
.venv/bin/python analyzer.py --socket /run/hyptcn/analyzer.sock
./bin/hyptcn --mock --socket /run/hyptcn/analyzer.sock --interval 250
```

### UDS connection retry (analyzer not yet up)

If the analyzer is slow to start, the Go binary retries silently:

```
level=WARN msg="analyzer socket unavailable, retrying" attempt=1 of=3
level=WARN msg="analyzer socket unavailable, retrying" attempt=2 of=3
level=ERROR msg="streaming aborted" err="failed to dial analyzer socket \"/tmp/hyptcn.sock\" after 3 attempts: ..."
```

---

## Current Status

The following works end-to-end today in mock mode:

- **Mock frame generation.** `--mock` fills 4096-byte pages with `crypto/rand` bytes at sequential physical addresses (`0x1000`, `0x2000`, …) at the configured millisecond interval.
- **All 16 features.** `feature_extractor.extract()` computes all 16 features correctly; every value stays in `[0.0, 1.0]` across random, zero-filled, text, ELF, and PE page types.
- **TCN inference.** `TCNAnomalyDetector` forward pass runs correctly on `(1, 16, 16)` input; `infer()` returns a valid float.
- **Sliding window.** `analyzer.py` correctly emits `warming_up` for frames 1–15 and live scores from frame 16 onward.
- **Alert threshold.** `score > 0.85` sets `"alert": true`; with random weights scores cluster near 0.496, so no false alerts fire in mock mode.
- **UDS retry.** Go binary retries the socket connection 3× with 1s delays before exiting.
- **Build.** `make` compiles the CGO Go binary against libvmi without errors on Fedora with `libvmi-devel` installed.

**Model weights are randomly initialized.** There is no training data and no trained model. All scores (~0.496) reflect random network weights applied to random page data, not meaningful anomaly detection. The architecture is correct and ready to train; it needs labeled memory captures.

---

## Roadmap

### Infrastructure TODO

- [ ] Add `--vm` flag integration with real LibVMI (remove mock requirement for live VM scanning)
- [ ] Add `entropy_delta` and `addr_delta` temporal features (T vs T−1 comparison across consecutive frames)
- [ ] Persistent frame logging to `.npy` files for dataset collection (`--log-dir` flag)
- [ ] Training script (`train.py`) with 70/15/15 split, Adam optimizer, early stopping on validation loss
- [ ] Model evaluation script: confusion matrix, ROC-AUC, F1 score, per-frame latency benchmarks
- [ ] REST API or gRPC endpoint for score streaming instead of stdout JSON

### Research TODO (thesis work)

- [ ] Collect normal behavior dataset: idle VM, web browsing, compilation workloads
- [ ] Collect malware dataset: Metasploit meterpreter, XMRig cryptominer, Diamorphine rootkit, ransomware-simulator
- [ ] Label frames by ground-truth process and build training pipeline
- [ ] Tune TCN hyperparameters: dilation depth, kernel size, filter count via grid search or Optuna
- [ ] Evaluate semantic gap solutions for process-level features (EPROCESS linked-list walking, symbol offsets from `rekall`/`volatility` profiles)
- [ ] Compare TCN vs LSTM/GRU baseline on the same dataset
- [ ] Write thesis chapters: virtualization theory, VMI principles, TCN theory, implementation, evaluation

---

## What Still Needs To Be Done

Honest accounting of everything not yet implemented:

- **Real LibVMI connection to a live VM.** The C extractor and CGO bridge compile and link correctly, but the end-to-end path from `vmi_read_pa` through to the Python analyzer has only been tested in mock mode. A real KVM guest with a matching `libvmi.conf` entry is required to validate it.
- **Trained model weights.** `models/tcn_weights.pt` does not exist. The model produces scores near 0.496 for all inputs because weights are random. The model will not detect anything meaningful until trained on labeled normal/malicious memory traces.
- **Dataset collection infrastructure.** There is no `--log-dir` flag, no `.npy` frame logger, and no labeling pipeline. Both normal and malicious memory captures need to be collected and labeled before training can begin.
- **Training and evaluation pipeline.** No `train.py`, no `eval.py`, no loss curves, no ROC-AUC measurement.
- **Process-level features via semantic gap.** The system operates at the physical page level with no knowledge of which guest process owns a given page. Detecting hidden processes (Diamorphine-style DKOM) requires parsing guest kernel data structures — `task_struct` linked lists on Linux, EPROCESS chains on Windows — a significant VMI engineering task involving OS version-specific symbol offsets.
- **Performance benchmarking under real hypervisor load.** No measurements exist for feature extraction latency, TCN inference time, or the overhead imposed on the guest by continuous `vmi_read_pa` calls.
- **Any form of alerting beyond stdout JSON.** There is no webhook, no syslog output, no email, and no integration with SIEM systems. The `"alert": true` field in the JSON is the entire alerting mechanism at this stage.
