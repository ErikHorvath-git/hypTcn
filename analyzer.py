"""hypTcn analyzer — asyncio UDS server.

Reads 4104-byte frames from the Go scanner:
    [8B physical address, little-endian uint64][4096B raw page data]

Per frame:
  - Extracts 8-element feature vector via feature_extractor
  - Appends to a 16-frame sliding window deque
  - When window is full: runs TCN inference, emits JSON

JSON output format (stdout + socket response):
  warming up:  {"status": "warming_up", "frames": <n>}
  live score:  {"timestamp": <unix_ms>, "addr": <hex_str>,
                "score": <float>, "alert": <bool>}

alert = True when score > 0.85.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import time
from collections import deque

import numpy as np

import feature_extractor
import tcn_model

PAGE_SIZE = 4096
HEADER_SIZE = 8
FRAME_SIZE = HEADER_SIZE + PAGE_SIZE          # 4104 bytes
SOCKET_PATH = "/tmp/hyptcn.sock"
ALERT_THRESHOLD = 0.85
SEQUENCE_LENGTH = tcn_model.SEQUENCE_LENGTH   # 16

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


# ── connection handler ─────────────────────────────────────────────────────────

async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    window: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)

    try:
        async for address, payload in _iter_frames(reader):
            vec = feature_extractor.extract(payload, address)
            window.append(vec)
            n = len(window)

            if n < SEQUENCE_LENGTH:
                response = {"status": "warming_up", "frames": n}
            else:
                # Stack deque into (FEATURE_DIM=8, SEQUENCE_LENGTH=16)
                seq = np.stack(list(window), axis=1)   # (8, 16)
                score = tcn_model.infer(_MODEL, seq)
                response = {
                    "timestamp": int(time.time() * 1000),
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

async def _serve(socket_path: str) -> None:
    if os.path.exists(socket_path):
        os.remove(socket_path)

    server = await asyncio.start_unix_server(_handle, path=socket_path)
    print(f"[analyzer] listening on {socket_path}", flush=True)
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="hypTcn analyzer service")
    parser.add_argument("--socket", default=SOCKET_PATH, help="UDS path")
    args = parser.parse_args()

    try:
        asyncio.run(_serve(args.socket))
    except KeyboardInterrupt:
        print("[analyzer] shutting down", flush=True)


if __name__ == "__main__":
    main()
