# hypTCn

hypTCn samples raw 4 KiB physical memory pages from a running KVM guest via libvmi — from outside the VM, at the hypervisor boundary — and classifies sliding windows of 16 consecutive pages as normal or anomalous using a PyTorch Temporal Convolutional Network (TCN). The guest has zero footprint: no agent, no kernel hooks, nothing detectable from inside. A TCN is used instead of a snapshot classifier because sequences of 16 pages capture temporal patterns — entropy drift, NOP-sled persistence, address-scan behaviour — that a single-frame classifier cannot see. This is a diploma thesis project targeting APT-class threats: shellcode injection, rootkits, cryptominers, and ransomware.

---

## Quick Start

### Prerequisites

- Fedora or Debian host with KVM/QEMU
- libvmi built from source with `ENABLE_KVM_LEGACY=ON` (the Fedora DNF package is Xen-only)
- Go 1.21+, Python 3.10+, PyTorch ≥ 2.0
- Your user in the `kvm` and `libvirt` groups

### 1. Build libvmi (one time)

The stock Fedora `libvmi` package has no KVM support. The legacy KVM driver uses libvirt QMP which works with stock QEMU; the new driver requires a patched QEMU with KVMI sockets.

```sh
sudo dnf install -y gcc make cmake bison flex autoconf automake libtool pkg-config \
    libvirt-devel json-c-devel glib2-devel

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

# Verify KVM support is compiled in:
strings /usr/local/lib64/libvmi.so | grep -i kvm
# must print: VMI_KVM   VMI_INIT_DATA_KVMI_SOCKET
```

> **Note:** Both `kvm.c` and `kvm_legacy.c` hardcode `qemu:///system` upstream. If your VM runs under `qemu:///session` (user session daemon), patch both files before building: `sed -i 's|qemu:///system|qemu:///session|g' src/driver/kvm/kvm.c src/driver/kvm/kvm_legacy.c`

### 2. Build hyptcn

```sh
make
# → bin/hyptcn, rpath-linked to /usr/local/lib64/libvmi
```

### 3. Install Python deps

```sh
python3 -m venv .venv
.venv/bin/pip install -r python/requirements.txt
```

Always use `.venv/bin/python` — system python3 does not have torch.

### 4. Create KVM guest (if you don't have one)

```sh
# Download Debian 12 cloud image
wget https://cloud.debian.org/images/cloud/bookworm/latest/debian-12-genericcloud-amd64.qcow2

# Create a 20 GiB overlay
qemu-img create -f qcow2 -b debian-12-genericcloud-amd64.qcow2 -F qcow2 hyptcn-guest.qcow2 20G

# Create cloud-init seed (sets root password + SSH key)
cat > user-data.yaml <<'EOF'
#cloud-config
password: hyptcn
chpasswd: {expire: false}
ssh_pwauth: true
EOF
cloud-localds seed.iso user-data.yaml

# Define and start the VM
virt-install \
  --name hyptcn-guest \
  --memory 2048 \
  --vcpus 2 \
  --disk path=hyptcn-guest.qcow2,format=qcow2 \
  --disk path=seed.iso,device=cdrom \
  --os-variant debian12 \
  --network network=default \
  --graphics none \
  --noautoconsole \
  --import

virsh domstate hyptcn-guest   # → running
```

### 5. Start everything

**Terminal 1 — Python analyzer (must start first):**
```sh
cd python && ../.venv/bin/python analyzer.py --socket /tmp/hyptcn.sock
```

**Terminal 2 — Go scanner:**
```sh
# Mock mode (no VM required, useful for testing the full pipeline):
./bin/hyptcn --mock --interval 100

# Live VM:
./bin/hyptcn --vm hyptcn-guest --address 0x1000000 --interval 500
```

**Terminal 3 — watch live scores:**
```sh
# Scores appear on analyzer stdout once the 16-frame window fills:
# {"timestamp":1773264369956,"addr":"0x1000000","score":0.0,"alert":false,"status":"ok"}
```

Expected mock output (random pages are max-entropy → model correctly flags as anomalous):
```
{"timestamp":…, "addr":"0x1000", "score":1.0, "alert":true, "status":"ok"}
```

Expected live idle-VM output (kernel pages are low-entropy → model scores as normal):
```
{"timestamp":…, "addr":"0x1000000", "score":0.0, "alert":false, "status":"ok"}
```

### 6. Collect training data

```sh
# One-time guest setup (installs stress-ng, ransomware sim, Diamorphine):
./collect_for_training/scripts/prepare_malware_env.sh hyptcn-guest

# Interactive wizard — walks through all 5 labels one by one:
./collect_for_training/scripts/collect_all_labels.sh hyptcn-guest
```

Frames are written to `collect_for_training/<label>/` as 4108-byte `.bin` files. No Python analyzer needed during collection — the Go binary writes raw frames directly.

### 7. Train the model

```sh
./collect_for_training/scripts/quick_train.sh
# or manually:
.venv/bin/python training/train.py --data-source collect --collect-dir collect_for_training/
```

Saves `models/tcn_weights.pt` and `models/config.json`. Restart the analyzer to pick up new weights.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  KVM Guest  (Debian 12, QEMU/KVM)               │
│  Physical RAM — no agent, zero guest footprint   │
└─────────────────┬───────────────────────────────┘
                  │  libvmi vmi_read_pa() / vmi_init_complete()
                  │  KVM legacy driver → qemu:///session → QMP xp
                  ▼
┌─────────────────────────────────────────────────┐
│  C Extractor  internal/extractor/probe.c        │
│  hyptcn_vmi_open[_with_sysmap](vm_name)         │
│  hyptcn_read_page(handle, phys_addr, buf)        │
│  hyptcn_get_process_list / _kernel_modules /     │
│  _network_connections / _translate_v2p           │
└─────────────────┬───────────────────────────────┘
                  │  CGO  (-I/usr/local/include -lvmi -rpath)
                  ▼
┌─────────────────────────────────────────────────┐
│  Go Orchestrator  internal/orchestrator/engine.go│
│  Engine.Stream(ctx, physAddr, interval)          │
│    ├─ NORMAL MODE: analyze frame → UDS → Python  │
│    │    wire: [8B addr][4B mod_norm][4B conn_norm]│
│    │           [4096B page] = 4112 bytes          │
│    └─ COLLECT MODE: write .bin → no Python needed│
│         format: [8B addr][4B label_id][4096B page]│
│                 = 4108 bytes per file             │
└─────────────────┬───────────────────────────────┘
                  │  Unix Domain Socket /tmp/hyptcn.sock
                  ▼
┌─────────────────────────────────────────────────┐
│  Python Analyzer  python/analyzer.py            │
│  asyncio UDS server, 16-frame sliding window     │
│  features.extract() → (20,) float32 per frame   │
│  → tcn.infer() → (anomaly_score, activity_class) │
│  socket response (JSON, newline-terminated):     │
│    {"anomaly_score":0.51,"status":"ok",          │
│     "activity_class":"shellcode"}                │
└─────────────────┬───────────────────────────────┘
                  │  PyTorch  ~0.34 ms/window (CPU)
                  ▼
┌─────────────────────────────────────────────────┐
│  TCN Model  python/model/tcn.py                 │
│  Input: (1, 20, 16)                             │
│  3 × TCNBlock (dilations 1, 2, 4)               │
│      dilated causal Conv1d, k=3, filters=32      │
│      weight_norm + ReLU + Dropout(0.1) + residual│
│  GlobalAvgPool → (1, 32)                        │
│  ├─ Anomaly head: Linear(32→16,ReLU)→Linear(16→1)│
│  └─ Class head:  Linear(32→64,ReLU)→Linear(64→5)│
│  Weights: models/tcn_weights.pt                 │
└─────────────────────────────────────────────────┘
```

---

## Features (20 features)

All values `float32 ∈ [0, 1]`. First frame: `entropy_delta = 0.5`, `addr_delta = 0.0`.

| # | Name | Description |
|---|------|-------------|
| 0 | `byte_entropy` | Shannon H / 8 — packed/encrypted code → near 1.0 |
| 1 | `nonzero_ratio` | count(b≠0) / 4096 — zero-padded pages → low |
| 2 | `printable_ratio` | count(0x20≤b≤0x7E) / 4096 — binary vs string pages |
| 3 | `high_byte_ratio` | count(b>0x7F) / 4096 — encoded/obfuscated content |
| 4 | `unique_bytes` | distinct byte values / 256 — crypto buffers → near 1.0 |
| 5 | `top4_freq` | sum(top-4 byte counts) / 4096 — NOP sleds / zero pages → high |
| 6 | `zero_runs` | count(zero-runs > 8) / 100 — uninitialised pages → high |
| 7 | `addr_norm` | phys_addr / 0xFFFFFFFFFFFF — kernel vs userspace position |
| 8 | `entropy_blocks_std` | std(H per 256B block) / 0.5 — mixed pages: header + payload |
| 9 | `compression_ratio` | zlib(page,1) size / 4096 — low=repetitive, high=packed/crypto |
| 10 | `null_run_ratio` | bytes inside zero-runs > 8 / 4096 — BSS/uninitialised → high |
| 11 | `pe_header_score` | 0.0 no MZ / 0.5 MZ only / 1.0 MZ+PE — injected PE / reflective DLL |
| 12 | `elf_header_score` | 0.0 / 0.5 / 1.0 ELF magic — ELF mapped into guest memory |
| 13 | `syscall_pattern_count` | count(SYSCALL\|INT80\|SYSENTER) / 100 — shellcode syscall density |
| 14 | `nop_sled_score` | bytes inside 0x90-runs > 8 / 4096 — NOP sled before shellcode |
| 15 | `string_density` | count(printable runs ≥ 4) / 100 — C2 URLs / config strings |
| 16 | `entropy_delta` | (H[t]−H[t-1]+1)/2 — entropy spike/drop over time **(temporal)** |
| 17 | `addr_delta` | (addr[t]−addr[t-1]) / 0xFFFFFFFFFFFF — non-sequential jumps **(temporal)** |
| 18 | `kernel_module_count_norm` | loaded kernel modules / 200 — rootkit detection **(OS layer)** |
| 19 | `network_conn_count_norm` | active TCP connections / 100 — C2/exfil detection **(OS layer)** |

OS-layer features (18–19) are `0.0` in mock/raw mode. The `--sysmap` flag enables them.

---

## CLI Reference

```sh
./bin/hyptcn [flags]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--vm` | `""` | KVM domain name (libvmi) |
| `--socket` | `/tmp/hyptcn.sock` | Analyzer Unix domain socket path |
| `--sysmap` | `""` | System.map path — enables `vmi_init_complete`, OS-layer features |
| `--address` | `0x1000000` | Physical address to sample (non-mock mode) |
| `--interval` | `100` | Sampling interval in milliseconds |
| `--proc-interval` | `100` | OS-layer scan (process/module/connection) every N frames |
| `--mock` | `false` | Use `crypto/rand` pages instead of libvmi |
| `--collect` | `false` | Enable dataset collection mode — writes `.bin` frames, no Python needed |
| `--collect-label` | `normal` | Label: `normal\|malware\|shellcode\|rootkit\|cryptominer\|ransomware` |
| `--collect-duration` | `300` | Collection duration in seconds; `0` = run until Ctrl-C |
| `--collect-dir` | `collect_for_training/` | Root output directory for `.bin` frame files |
| `--log-level` | `info` | Structured log level: `debug\|info\|warn\|error` |
| `--json` | `true` | Print analysis scores as JSON lines to stdout |
| `--quiet` | `false` | Suppress stdout JSON (overrides `--json`) |

**Python analyzer flags** (`python/analyzer.py`):

| Flag | Default | Description |
|------|---------|-------------|
| `--socket` | `/tmp/hyptcn.sock` | UDS listen path |
| `--log-dir` | none | Save `.npy` frame files here (for offline training) |
| `--label` | `unknown` | Subdirectory under `--log-dir` |

---

## Dataset Collection

Use `--collect` mode (Go binary only, no Python required) to build the real training dataset. Each `.bin` file is exactly 4108 bytes: `[8B uint64 LE addr][4B uint32 LE label_id][4096B raw page]`.

Label IDs: `normal=0`, `malware=1`, `shellcode=2`, `rootkit=3`, `cryptominer=4`, `ransomware=5`.

```sh
# One-time guest setup
./collect_for_training/scripts/prepare_malware_env.sh hyptcn-guest

# Guided wizard — all 5 labels with instructions per label
./collect_for_training/scripts/collect_all_labels.sh hyptcn-guest 120 200

# Or collect individual labels
./collect_for_training/scripts/collect_normal.sh  hyptcn-guest 300 200
./collect_for_training/scripts/collect_malware.sh hyptcn-guest shellcode 300 200

# Check counts and estimated training sequences
./collect_for_training/scripts/dataset_stats.sh

# Train on collected frames
./collect_for_training/scripts/quick_train.sh
```

Recommended minimums: `normal` ≥ 3000 frames, all other labels ≥ 1500 frames.
At 200 ms interval: 3000 frames = 10 minutes of collection.

---

## Project Structure

```
hypTCn/
├── cmd/hyptcn/main.go              # CLI entry point (Cobra, all flags, grouped --help)
├── internal/
│   ├── extractor/
│   │   ├── probe.h                 # C API: open/read/close/process/module/conn/v2p
│   │   ├── probe.c                 # libvmi wrapper (raw PA + OS-layer with sysmap)
│   │   └── extractor.go            # CGO bridge exposing Go types and methods
│   └── orchestrator/
│       └── engine.go               # Main loop: VMI read → analyze or collect → UDS
├── python/
│   ├── analyzer.py                 # asyncio UDS server, 16-frame window, inference
│   └── model/
│       ├── features.py             # 20-feature extractor (pure numpy, no torch)
│       ├── tcn.py                  # TCNAnomalyDetector, dual-head, load/save/infer
│       └── __init__.py             # Re-exports public API
├── training/
│   ├── synthetic.py                # Synthetic frame generator (5 archetypes)
│   ├── dataset.py                  # MemoryPageDataset (.npy) + BinFrameDataset (.bin)
│   ├── train.py                    # Multi-task training loop (BCE + CrossEntropy)
│   └── evaluate.py                 # Accuracy/F1/ROC-AUC + latency benchmark
├── collect_for_training/
│   ├── {normal,shellcode,rootkit,  # Raw .bin frame files per label
│   │    cryptominer,ransomware}/
│   └── scripts/
│       ├── prepare_malware_env.sh  # One-time guest setup (stress-ng, Diamorphine, sim)
│       ├── collect_normal.sh       # Collect normal behavior frames
│       ├── collect_malware.sh      # Collect one malware label with instructions
│       ├── collect_all_labels.sh   # Interactive wizard: all 5 labels in sequence
│       ├── dataset_stats.sh        # Print frame counts + sequence estimates per label
│       └── quick_train.sh          # Wrapper: dataset_stats → train.py --data-source collect
├── tools/
│   └── get_sysmap.sh               # SSH into guest, copy /boot/System.map → configs/
├── models/
│   ├── tcn_weights.pt              # Trained weights (synthetic data, v2.0.0)
│   ├── config.json                 # Hyperparameters + training stats
│   └── README.md                   # Weight file format documentation
├── configs/                        # System.map files (gitignored, placed by get_sysmap.sh)
├── third_party/github.com/spf13/cobra/  # Minimal vendored Cobra (stdlib flag, no pflag)
├── Makefile                        # build / deps / python-service / clean
└── go.mod                          # Module: github.com/example/hypTcn
```

---

## Current Status

| Works | Not Yet Done |
|-------|-------------|
| Live KVM introspection via libvmi legacy driver | Real malware dataset (all data is synthetic) |
| Full pipeline: Go → UDS → Python → TCN → JSON | Live score has no ground truth — model needs real labels |
| Mock mode (`--mock`) for pipeline testing without a VM | Smoke tests (`make test`) |
| 20-feature extractor (page + temporal + OS-layer features) | Address sweep — scanner samples one fixed PA per run |
| OS-layer: process list, kernel modules, TCP connections via `vmi_init_complete` | Semantic gap — no PA→VA→PID mapping without OS-layer active |
| V2P translation (`TranslateV2P`) | `confusion_matrix.png` (add matplotlib to requirements) |
| Dataset collection pipeline (`--collect`, 6 shell scripts) | Systemd unit files |
| Multi-task TCN: anomaly score + activity class (5 classes) | Compare TCN vs LSTM/attention baseline |
| Training on collected `.bin` frames (`--data-source collect`) | SIEM integration (syslog/Kafka/Elastic) |
| `models/tcn_weights.pt` — synthetic accuracy 99.8%, ROC-AUC 1.0 | Multi-guest monitoring |

---

## License / Thesis

This is a diploma thesis project. No license assigned yet.
