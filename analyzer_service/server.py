import argparse
import asyncio
import json
import logging
import os
import struct

import torch

from analyzer_service.model import TCNModel
from analyzer_service.processor import PageBuffer

PAGE_SIZE = 4096
HEADER_SIZE = 8

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("analyzer")
MODEL = TCNModel()


async def iter_pages(reader: asyncio.StreamReader):
    """Yield framed page events from a raw Unix stream."""
    while True:
        try:
            header = await reader.readexactly(HEADER_SIZE)
            payload = await reader.readexactly(PAGE_SIZE)
        except asyncio.IncompleteReadError:
            return

        address = struct.unpack("<Q", header)[0]
        yield address, payload


def analyze_page(buffer: PageBuffer, payload: bytes) -> dict:
    sequence = buffer.append(payload)
    if sequence is None:
        return {"anomaly_score": 0.0, "status": "warming_up"}

    tensor = torch.from_numpy(sequence).unsqueeze(0).to(torch.float32)
    score = float(MODEL.infer(tensor))
    status = "suspicious" if score > 0.5 else "clean"
    return {"anomaly_score": score, "status": status}


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    buffer = PageBuffer()
    peer = writer.get_extra_info("peername")
    LOG.info("client connected: %s", peer)

    try:
        async for address, payload in iter_pages(reader):
            result = analyze_page(buffer, payload)
            LOG.info(
                "page analyzed: address=0x%x score=%.4f status=%s",
                address,
                result["anomaly_score"],
                result["status"],
            )
            writer.write((json.dumps(result) + "\n").encode("utf-8"))
            await writer.drain()
    except Exception:
        LOG.exception("stream handler failed")
    finally:
        writer.close()
        await writer.wait_closed()
        LOG.info("client disconnected")


async def run_server(socket_path: str):
    if os.path.exists(socket_path):
        os.remove(socket_path)

    server = await asyncio.start_unix_server(handle_client, path=socket_path)
    LOG.info("raw UDS analyzer listening on %s", socket_path)

    async with server:
        await server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Raw UDS analyzer service")
    parser.add_argument("--socket", default="/tmp/hyptcn.sock", help="unix socket path")
    args = parser.parse_args()

    try:
        asyncio.run(run_server(args.socket))
    except KeyboardInterrupt:
        LOG.info("shutting down analyzer")


if __name__ == "__main__":
    main()
