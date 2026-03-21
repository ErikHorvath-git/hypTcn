"""hypTcn analyzer — asyncio UDS server.

Reads 4112-byte frames from the Go scanner:
    [8B physical address, little-endian uint64]
    [4B kernel_module_count_norm, little-endian float32]
    [4B network_conn_count_norm,  little-endian float32]
    [4096B raw page data]

Per frame:
  - Extracts 20-element feature vector via model.features (with prev_features
    and OS-layer normalized counts from the frame header)
  - Optionally logs raw frame + features to --log-dir as .npy files
  - Appends to a 16-frame sliding window deque
  - When window is full: runs TCN inference, emits JSON

JSON output format:
  socket (read by Go engine):
    warming up:  {"anomaly_score": 0.0, "status": "warming_up"}
    live score:  {"anomaly_score": <float>, "activity_class": <str>, "status": "ok"}
  stdout (human-readable):
    warming up:  {"status": "warming_up", "frames": <n>}
    live score:  {"timestamp": <unix_ms>, "addr": <hex_str>,
                  "score": <float>, "alert": <bool>,
                  "activity_class": <str>, "status": "ok"}

alert = True when score > 0.85.

--log-dir layout:
  <log-dir>/<label>/<timestamp_ms>_<addr_hex>.npy
  label defaults to "unknown"; rename folder to "normal" or "malware" after capture.
  Each file: np.save(..., {"features": (20,), "addr": int, "timestamp_ms": int,
                            "raw_page": (4096,) uint8}, allow_pickle=True)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import sys
import time
from collections import deque

import numpy as np

from model import features as feature_extractor
from model import tcn as tcn_model

PAGE_SIZE   = 4096
HEADER_SIZE = 16              # 8 (addr) + 4 (mod_norm) + 4 (conn_norm)
FRAME_SIZE  = HEADER_SIZE + PAGE_SIZE    # 4112 bytes
SOCKET_PATH = "/tmp/hyptcn.sock"
_LOG_INTERVAL = 100           # print stderr notice every N frames

# Load model and config once at startup.
# load_model() resolves: python/model/weights/ (primary) → models/ (fallback)
_MODEL  = tcn_model.load_model()
_CONFIG = tcn_model._load_config(tcn_model._resolve("config.json"))
ALERT_THRESHOLD = _CONFIG.get("alert_threshold", 0.85) if _CONFIG else 0.85
SEQUENCE_LENGTH = tcn_model.SEQUENCE_LENGTH   # 16


def _print_startup_info() -> None:
    if _CONFIG:
        ver        = _CONFIG.get("model_version", "?")
        trained_at = _CONFIG.get("trained_at", "unknown")[:10]
        ts         = _CONFIG.get("training_stats", {})
        val_loss   = ts.get("best_val_loss", float("nan"))
        roc_auc    = ts.get("roc_auc", float("nan"))
        print(
            f"[analyzer] model v{ver} trained {trained_at}, "
            f"val_loss={val_loss:.4f}, ROC-AUC={roc_auc:.4f}",
            flush=True,
        )
    else:
        print("[analyzer] using random weights (no trained model found)", flush=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def _emit(obj: dict) -> None:
    """Print JSON to stdout immediately."""
    print(json.dumps(obj), flush=True)


def _encode(obj: dict) -> bytes:
    """Encode a dict as newline-terminated JSON bytes for the socket."""
    return (json.dumps(obj) + "\n").encode()


# ── frame reader ───────────────────────────────────────────────────────────────

async def _iter_frames(reader: asyncio.StreamReader):
    """Yield (address, payload, mod_norm, conn_norm) tuples from the raw UDS stream."""
    while True:
        try:
            header  = await reader.readexactly(HEADER_SIZE)
            payload = await reader.readexactly(PAGE_SIZE)
        except asyncio.IncompleteReadError:
            return
        address   = struct.unpack("<Q", header[0:8])[0]
        mod_norm  = struct.unpack("<f", header[8:12])[0]
        conn_norm = struct.unpack("<f", header[12:16])[0]
        yield address, payload, mod_norm, conn_norm


# ── frame logger ───────────────────────────────────────────────────────────────

def _log_frame(
    log_dir: str,
    label: str,
    address: int,
    timestamp_ms: int,
    payload: bytes,
    features: np.ndarray,
    frame_count: int,
) -> None:
    """Save one frame to <log_dir>/<label>/<timestamp_ms>_<addr_hex>.npy."""
    dest_dir = os.path.join(log_dir, label)
    os.makedirs(dest_dir, exist_ok=True)
    fname = f"{timestamp_ms}_{address:016x}.npy"
    fpath = os.path.join(dest_dir, fname)
    np.save(
        fpath,
        {
            "features":     features,
            "addr":         address,
            "timestamp_ms": timestamp_ms,
            "raw_page":     np.frombuffer(payload, dtype=np.uint8),
        },
        allow_pickle=True,
    )
    if frame_count % _LOG_INTERVAL == 0:
        print(f"logged frame to {fpath}", file=sys.stderr, flush=True)


# ── connection handler ─────────────────────────────────────────────────────────

async def _handle(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    log_dir: str | None = None,
    label: str = "unknown",
) -> None:
    window: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    prev_features: np.ndarray | None = None
    frame_count = 0

    try:
        async for address, payload, mod_norm, conn_norm in _iter_frames(reader):
            timestamp_ms = int(time.time() * 1000)
            frame_count += 1

            vec = feature_extractor.extract(
                payload, address, prev_features,
                kernel_module_count_norm=mod_norm,
                network_conn_count_norm=conn_norm,
            )
            prev_features = vec
            window.append(vec)
            n = len(window)

            if log_dir is not None:
                _log_frame(log_dir, label, address, timestamp_ms, payload, vec, frame_count)

            if n < SEQUENCE_LENGTH:
                socket_response = {"anomaly_score": 0.0, "status": "warming_up"}
                stdout_response = {"status": "warming_up", "frames": n}
            else:
                # Stack deque into (FEATURE_DIM=20, SEQUENCE_LENGTH=16)
                seq = np.stack(list(window), axis=1)   # (20, 16)
                score, activity_class = tcn_model.infer(_MODEL, seq)
                socket_response = {
                    "anomaly_score": round(score, 6),
                    "activity_class": activity_class,
                    "status": "ok",
                }
                stdout_response = {
                    "timestamp":     timestamp_ms,
                    "addr":          hex(address),
                    "score":         round(score, 6),
                    "alert":         score > ALERT_THRESHOLD,
                    "activity_class": activity_class,
                    "status":        "ok",
                }

            _emit(stdout_response)
            writer.write(_encode(socket_response))
            await writer.drain()

    except Exception as exc:
        _emit({"error": str(exc)})
    finally:
        writer.close()
        await writer.wait_closed()


# ── server ─────────────────────────────────────────────────────────────────────

async def _serve(socket_path: str, log_dir: str | None, label: str) -> None:
    if os.path.exists(socket_path):
        os.remove(socket_path)

    def _handler(reader, writer):
        return _handle(reader, writer, log_dir=log_dir, label=label)

    server = await asyncio.start_unix_server(_handler, path=socket_path)
    _print_startup_info()
    if log_dir:
        print(f"[analyzer] logging frames to {os.path.join(log_dir, label)}/", flush=True)
    print(f"[analyzer] listening on {socket_path}", flush=True)
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="hypTcn analyzer service")
    parser.add_argument("--socket", default=SOCKET_PATH, help="UDS path")
    parser.add_argument("--log-dir", default=None, metavar="PATH",
                        help="directory to save raw frames as .npy files")
    parser.add_argument("--label", default="unknown",
                        help="subdirectory label: 'normal', 'malware', or 'unknown'")
    args = parser.parse_args()

    try:
        asyncio.run(_serve(args.socket, args.log_dir, args.label))
    except KeyboardInterrupt:
        print("[analyzer] shutting down", flush=True)


if __name__ == "__main__":
    main()
