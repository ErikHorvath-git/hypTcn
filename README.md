# hypTcn
![hypTcn status](https://img.shields.io/badge/status-beta-blue)

hypTcn is a modular, hypervisor-aware security toolkit that performs live memory introspection on KVM guests and feeds the resulting temporal sequences into a Temporal Convolutional Network (TCN) for anomaly detection. The project stitches together a libvmi-powered C extractor, a Go orchestrator that streams snapshots over raw Unix domain sockets (UDS), and a PyTorch-based analyzer that learns temporal patterns across 4 KiB pages.

## Architecture

- **Data Source – `internal/extractor/`**  
  Native C probe powered by libvmi (`VMI_KVM`, `VMI_INIT_NAME`) that reads exactly one 4 KiB physical page per sampling interval.
- **Orchestration – `cmd/hyptcn/` & `internal/orchestrator/`**  
  Cobra-driven CLI (`hyptcn scan`) captures OS signals, runs a sampling loop, and forwards each page over a persistent raw UDS stream. The Go `Engine` repeatedly calls `extractor.ExtractPage`, prefixes each 4096-byte page with an 8-byte address header, and reads one JSON line response per frame.
- **Analysis – `analyzer_service/`**  
  Async Python UDS server hosting a PyTorch TCN (`model/` package). Incoming 4 KiB pages are histogrammed and buffered, then fed to dilated residual blocks that respect causality; the temporal nature of the TCN lets it detect drift across time instead of isolated snapshots.

## Prerequisites

### Fedora (Development)
```sh
sudo dnf install make gcc go python3 libvmi-devel
```

### Debian (Production)
```sh
sudo apt install build-essential golang python3-venv libvmi-dev
```

- Ensure `/etc/libvmi/libvmi.conf` points to your KVM socket or domain discovery mechanism.
- Add your user to the `kvm` group so libvmi can talk to the hypervisor:
```sh
sudo usermod -aG kvm $USER
```

## Build Instructions

1. `make deps`  
   Sets up `.venv/` and installs `torch`, `numpy`, and other Python dependencies.

2. `make`  
   Builds `bin/hyptcn`, compiling the CGO bridge and linking `-lvmi` to ship a standalone Go binary.

## How-to-Run

1. Start the analyzer (raw UDS listener):
   ```sh
   make python-service
   ```

2. Launch the scanner and point it at a specific VM/physical address:
   ```sh
   ./bin/hyptcn scan --vm <vm_name> --address 0x12345678 --socket /tmp/hyptcn.sock
   ```
   The CLI accepts `--interval` to adjust sampling cadence; it streams pages continuously and logs anomaly scores returned by the temporal model.

## Component Breakdown

```
analyzer_service/       # Python TCN logic (raw UDS server, PyTorch, model/ package)
bin/                    # Compiled Go binaries (hyptcn CLI)
cmd/hyptcn/             # Cobra entry point that configures CLI flags and starts the Engine
internal/extractor/     # CGO bridge + LibVMI probe (probe.c/h + extractor.go)
internal/orchestrator/  # Go Engine (framed UDS streaming, sampling loop)
Makefile                # Builds Go binary with CGO, manages Python venv/service
third_party/            # Vendored dependencies (e.g., Cobra shim)
```

## Temporal Innovation

hypTcn’s core innovation is the temporal analysis loop: while libvmi produces raw 4 KiB snapshots, the PyTorch TCN in `analyzer_service/model/` tracks how those histograms change over time using dilated causal convolutions. This lets the network understand sequences, detect gradual drifts, and differentiate between transient noise and evolving attacks.
