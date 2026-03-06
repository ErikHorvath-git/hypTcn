"""hypTcn analyzer — asyncio UDS server.

Reads 4104-byte frames from the Go scanner:
    [8B physical address, little-endian uint64][4096B raw page data]

Per frame:
  - Extracts 18-element feature vector via model.features (with prev_features)
  - Optionally logs raw frame + features to --log-dir as .npy files
  - Appends to a 16-frame sliding window deque
  - When window is full: runs TCN inference, emits JSON

JSON output format (stdout + socket response):
  warming up:  {"status": "warming_up", "frames": <n>}
  live score:  {"timestamp": <unix_ms>, "addr": <hex_str>,
                "score": <float>, "alert": <bool>}

alert = True when score > 0.85.

--log-dir layout:
  <log-dir>/<label>/<timestamp_ms>_<addr_hex>.npy
  label defaults to "unknown"; rename folder to "normal" or "malware" after capture.
  Each file: np.save(..., {"features": (18,), "addr": int, "timestamp_ms": int,
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

PAGE_SIZE = 4096
HEADER_SIZE = 8
FRAME_SIZE = HEADER_SIZE + PAGE_SIZE          # 4104 bytes
SOCKET_PATH = "/tmp/hyptcn.sock"
ALERT_THRESHOLD = 0.85
SEQUENCE_LENGTH = tcn_model.SEQUENCE_LENGTH   # 16
_LOG_INTERVAL = 100                           # print stderr notice every N frames

# Model is loaded once at startup; random init when no weights file exists.
_MODEL = tcn_model.load_model()


# ── helpers ────────────────────────────────────────────────────────────────────

def _emit(obj: dict) -> None:
    """Print JSON to stdout immediately."""
    print(json.dumps(obj), flush=True)


def _encode(obj: dict) -> bytes:
    """Encode a dict as newline-terminated JSON bytes for the socket."""
    return (json.dumps(obj) + "\n").encode()


# ── frame reader ───────────────────────────────────────────────────────────────

async def _iter_frames(reader: asyncio.StreamReader):
    """Yield (address, payload) pairs from the raw UDS stream."""
    while True:
        try:
            header = await reader.readexactly(HEADER_SIZE)
            payload = await reader.readexactly(PAGE_SIZE)
        except asyncio.IncompleteReadError:
            return
        address = struct.unpack("<Q", header)[0]
        yield address, payload


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
            "features": features,
            "addr": address,
            "timestamp_ms": timestamp_ms,
            "raw_page": np.frombuffer(payload, dtype=np.uint8),
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
        async for address, payload in _iter_frames(reader):
            timestamp_ms = int(time.time() * 1000)
            frame_count += 1

            vec = feature_extractor.extract(payload, address, prev_features)
            prev_features = vec
            window.append(vec)
            n = len(window)

            if log_dir is not None:
                _log_frame(log_dir, label, address, timestamp_ms, payload, vec, frame_count)

            if n < SEQUENCE_LENGTH:
                response = {"status": "warming_up", "frames": n}
            else:
                # Stack deque into (FEATURE_DIM=18, SEQUENCE_LENGTH=16)
                seq = np.stack(list(window), axis=1)   # (18, 16)
                score = tcn_model.infer(_MODEL, seq)
                response = {
                    "timestamp": timestamp_ms,
                    "addr": hex(address),
                    "score": round(score, 6),
                    "alert": score > ALERT_THRESHOLD,
                }

            _emit(response)
            writer.write(_encode(response))
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
