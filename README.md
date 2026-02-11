# hypTcn

Modular proof-of-concept for a hypervisor-aware security tool that stitches together C, Go, and Python components in a production-ready layout suitable for a thesis repository.

## Architecture

- **C extractor** (`internal/extractor/probe.{c,h}`) exposes `int fetch_ram_page(uint64_t address, char* buffer)` that simulates a low-level hypervisor page fetch. It is compiled into an object file and linked via CGO.
- **Go orchestrator** (`internal/orchestrator/engine.go`) maintains structured logging with `slog`, handles HTTP-over-Unix-domain-socket communication, and exposes a reusable `Engine` that forwards raw binary snapshots to the analyzer service with minimal copying.
- **Go CLI** (`cmd/hyptcn/main.go`) uses `cobra` to implement the `scan` command, captures signal-driven graceful shutdown, and drives the orchestrator while reporting predictions.
- **Python analyzer service** (`analyzer_service/server.py`) boots a FastAPI app, preloads a placeholder `TCNModel`, and exposes `/analyze` for binary inference requests, serving over a Unix domain socket.

## Prerequisites (Fedora)

```sh
sudo dnf install make gcc go python3
```

## Workflow

1. `make deps` — installs a virtual environment (`.venv/`) and the Python requirements (`fastapi`, `uvicorn`).
2. `make` — builds `bin/hyptcn` with CGO (the extractor is compiled along with the Go code) using a repository-local `GOCACHE`.
3. `make python-service` (in a separate terminal) — starts the FastAPI server listening on `/tmp/hyptcn.sock`.
4. `./bin/hyptcn --socket /tmp/hyptcn.sock` — runs the Cobra-driven orchestrator, captures a hypervisor page, forwards it to the analyzer, and logs the prediction. The command handles `SIGINT`/`SIGTERM` gracefully.

## Project layout

```
cmd/hyptcn/             # Cobra CLI entry point
internal/extractor/      # CGO bridge plus C probe implementation
internal/orchestrator/   # Engine that dials the analyzer service via UDS
analyzer_service/        # FastAPI + uvicorn analyzer server
third_party/             # Local cobra shim for offline builds
Makefile                # Builds Go binary, compiles C, and boots Python env/service
README.md               # This document
.go.mod/.sum            # Go module definition
.venv/                  # Created by make deps
bin/                    # Output binary
```
