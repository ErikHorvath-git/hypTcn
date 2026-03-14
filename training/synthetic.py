"""Synthetic labeled frame generator for hypTcn training.

Produces temporally-coherent sequences of 20-feature vectors that simulate
real memory page patterns, without needing a live KVM guest.

Feature index reference (matches python/model/features.py FEATURE_DIM=20):
    0  byte_entropy        1  nonzero_ratio     2  printable_ratio
    3  high_byte_ratio     4  unique_bytes       5  top4_freq
    6  zero_runs           7  addr_norm          8  entropy_blocks_std
    9  compression_ratio  10  null_run_ratio    11  pe_header_score
   12  elf_header_score   13  syscall_pattern_count
   14  nop_sled_score     15  string_density
   16  entropy_delta      17  addr_delta
   18  kernel_module_count_norm   (OS-layer; 0.0 if unavailable)
   19  network_conn_count_norm    (OS-layer; 0.0 if unavailable)

Activity classes (matches tcn.py ACTIVITY_CLASSES):
    0 = normal
    1 = shellcode
    2 = rootkit
    3 = cryptominer
    4 = ransomware
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Callable

import numpy as np

FEATURE_DIM = 20
SEQ_LEN = 16


# ── helpers ────────────────────────────────────────────────────────────────────

def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _noise(rng: np.random.Generator, shape, scale: float = 0.02) -> np.ndarray:
    return rng.normal(0, scale, shape)


def _smooth_walk(rng: np.random.Generator, n: int, start: float,
                 step_std: float, lo: float, hi: float) -> np.ndarray:
    """Random walk clamped to [lo, hi]; used for temporal feature drift."""
    vals = [start]
    for _ in range(n - 1):
        vals.append(np.clip(vals[-1] + rng.normal(0, step_std), lo, hi))
    return np.array(vals)


# ── per-archetype sequence generators ─────────────────────────────────────────

def _normal_sequence(rng: np.random.Generator, base_addr: float) -> np.ndarray:
    """16-frame normal sequence: kernel/user data pages, low entropy."""
    entropy = _smooth_walk(rng, SEQ_LEN, rng.uniform(0.3, 0.55), 0.03, 0.1, 0.65)
    addr = base_addr + np.arange(SEQ_LEN) * (4096 / 0xFFFFFFFFFFFF)

    # Typical Linux guest: ~50–100 kernel modules, ~5–20 TCP connections
    mod_count  = rng.uniform(0.20, 0.55)   # 40–110 modules / 200
    conn_count = rng.uniform(0.00, 0.15)   # 0–15 connections / 100

    frames = []
    for i in range(SEQ_LEN):
        e = entropy[i]
        f = np.array([
            e,                                              # byte_entropy
            rng.uniform(0.1, 0.45),                        # nonzero_ratio
            rng.uniform(0.05, 0.25),                       # printable_ratio
            rng.uniform(0.02, 0.08),                       # high_byte_ratio
            rng.uniform(0.08, 0.25),                       # unique_bytes
            rng.uniform(0.55, 0.82),                       # top4_freq
            rng.uniform(0.05, 0.35),                       # zero_runs
            addr[i],                                       # addr_norm
            rng.uniform(0.05, 0.20),                       # entropy_blocks_std
            rng.uniform(0.08, 0.35),                       # compression_ratio
            rng.uniform(0.45, 0.80),                       # null_run_ratio
            0.0,                                           # pe_header_score
            0.0,                                           # elf_header_score
            rng.uniform(0.0, 0.015),                       # syscall_pattern_count
            0.0,                                           # nop_sled_score
            rng.uniform(0.08, 0.28),                       # string_density
            0.0,                                           # entropy_delta (filled below)
            0.0,                                           # addr_delta    (filled below)
            mod_count  + rng.normal(0, 0.01),              # kernel_module_count_norm
            conn_count + rng.normal(0, 0.005),             # network_conn_count_norm
        ], dtype=np.float32)
        frames.append(f)

    return _fill_temporal(frames, addr)


def _shellcode_sequence(rng: np.random.Generator, base_addr: float) -> np.ndarray:
    """High-entropy shellcode page with NOP sled and syscall opcodes."""
    entropy = _smooth_walk(rng, SEQ_LEN, rng.uniform(0.87, 0.95), 0.02, 0.80, 1.0)
    addr = base_addr + np.arange(SEQ_LEN) * (4096 / 0xFFFFFFFFFFFF)

    # Shellcode may open extra sockets
    mod_count  = rng.uniform(0.20, 0.55)
    conn_count = rng.uniform(0.10, 0.50)   # elevated connections

    frames = []
    for i in range(SEQ_LEN):
        e = entropy[i]
        f = np.array([
            e,
            rng.uniform(0.92, 1.0),
            rng.uniform(0.08, 0.22),
            rng.uniform(0.28, 0.55),
            rng.uniform(0.78, 1.0),
            rng.uniform(0.02, 0.08),
            rng.uniform(0.0, 0.015),
            addr[i],
            rng.uniform(0.02, 0.12),
            rng.uniform(0.88, 1.0),
            rng.uniform(0.0, 0.04),
            0.0,
            0.0,
            rng.uniform(0.04, 0.25),
            rng.uniform(0.04, 0.18),
            rng.uniform(0.01, 0.06),
            0.0,
            0.0,
            mod_count  + rng.normal(0, 0.01),
            conn_count + rng.normal(0, 0.02),
        ], dtype=np.float32)
        frames.append(f)

    return _fill_temporal(frames, addr)


def _packed_pe_sequence(rng: np.random.Generator, base_addr: float) -> np.ndarray:
    """Packed PE: PE magic header present, very high compression ratio."""
    entropy = _smooth_walk(rng, SEQ_LEN, rng.uniform(0.78, 0.92), 0.025, 0.65, 1.0)
    addr = base_addr + np.arange(SEQ_LEN) * (4096 / 0xFFFFFFFFFFFF)

    mod_count  = rng.uniform(0.20, 0.55)
    conn_count = rng.uniform(0.05, 0.30)

    frames = []
    for i in range(SEQ_LEN):
        e = entropy[i]
        f = np.array([
            e,
            rng.uniform(0.88, 1.0),
            rng.uniform(0.05, 0.20),
            rng.uniform(0.20, 0.50),
            rng.uniform(0.65, 0.95),
            rng.uniform(0.03, 0.10),
            rng.uniform(0.0, 0.02),
            addr[i],
            rng.uniform(0.15, 0.40),
            rng.uniform(0.82, 1.0),
            rng.uniform(0.0, 0.05),
            rng.choice([0.5, 1.0]),
            0.0,
            rng.uniform(0.01, 0.08),
            rng.uniform(0.0, 0.05),
            rng.uniform(0.02, 0.12),
            0.0,
            0.0,
            mod_count  + rng.normal(0, 0.01),
            conn_count + rng.normal(0, 0.01),
        ], dtype=np.float32)
        frames.append(f)

    return _fill_temporal(frames, addr)


def _cryptominer_sequence(rng: np.random.Generator, base_addr: float) -> np.ndarray:
    """Crypto-miner work buffers: near-perfect entropy, very high unique_bytes."""
    entropy = _smooth_walk(rng, SEQ_LEN, rng.uniform(0.96, 1.0), 0.01, 0.92, 1.0)
    addr = base_addr + np.arange(SEQ_LEN) * (4096 / 0xFFFFFFFFFFFF)

    # Miners open pool connections
    mod_count  = rng.uniform(0.20, 0.55)
    conn_count = rng.uniform(0.02, 0.12)   # 2–12 mining pool connections / 100

    frames = []
    for i in range(SEQ_LEN):
        e = entropy[i]
        f = np.array([
            e,
            rng.uniform(0.98, 1.0),
            rng.uniform(0.05, 0.15),
            rng.uniform(0.40, 0.55),
            rng.uniform(0.92, 1.0),
            rng.uniform(0.01, 0.05),
            0.0,
            addr[i],
            rng.uniform(0.0, 0.06),
            rng.uniform(0.95, 1.0),
            0.0,
            0.0,
            0.0,
            rng.uniform(0.0, 0.02),
            0.0,
            rng.uniform(0.0, 0.03),
            0.0,
            0.0,
            mod_count  + rng.normal(0, 0.01),
            conn_count + rng.normal(0, 0.005),
        ], dtype=np.float32)
        frames.append(f)

    return _fill_temporal(frames, addr)


def _rootkit_sequence(rng: np.random.Generator, base_addr: float) -> np.ndarray:
    """Rootkit: normal-looking entropy but addr_delta spikes (DKOM-like jumps)
    and anomalous kernel module counts (hidden or injected modules)."""
    entropy = _smooth_walk(rng, SEQ_LEN, rng.uniform(0.35, 0.55), 0.04, 0.15, 0.70)

    addrs = [base_addr]
    for _ in range(SEQ_LEN - 1):
        if rng.random() < 0.35:
            addrs.append(rng.uniform(0.0, 1.0))
        else:
            addrs.append(np.clip(addrs[-1] + 4096 / 0xFFFFFFFFFFFF, 0.0, 1.0))
    addr = np.array(addrs)

    # Rootkit may hide modules (low count) or inject extra ones (high count)
    mod_count  = rng.choice([
        rng.uniform(0.05, 0.15),   # hidden modules: very few visible
        rng.uniform(0.60, 0.95),   # injected modules: many
    ])
    conn_count = rng.uniform(0.0, 0.10)

    frames = []
    for i in range(SEQ_LEN):
        e = entropy[i]
        f = np.array([
            e,
            rng.uniform(0.15, 0.50),
            rng.uniform(0.08, 0.28),
            rng.uniform(0.02, 0.10),
            rng.uniform(0.12, 0.30),
            rng.uniform(0.50, 0.78),
            rng.uniform(0.05, 0.30),
            addr[i],
            rng.uniform(0.05, 0.20),
            rng.uniform(0.10, 0.38),
            rng.uniform(0.35, 0.72),
            0.0,
            0.0,
            rng.uniform(0.0, 0.02),
            0.0,
            rng.uniform(0.06, 0.22),
            0.0,
            0.0,
            mod_count  + rng.normal(0, 0.01),
            conn_count + rng.normal(0, 0.005),
        ], dtype=np.float32)
        frames.append(f)

    return _fill_temporal(frames, addr)


def _ransomware_sequence(rng: np.random.Generator, base_addr: float) -> np.ndarray:
    """Ransomware: high-entropy encryption buffers, many network connections
    (C2 exfiltration), mixed printable strings (ransom note fragments)."""
    entropy = _smooth_walk(rng, SEQ_LEN, rng.uniform(0.90, 1.0), 0.02, 0.82, 1.0)
    addr = base_addr + np.arange(SEQ_LEN) * (4096 / 0xFFFFFFFFFFFF)

    # Ransomware: many outbound connections to C2 / file upload endpoints
    mod_count  = rng.uniform(0.20, 0.55)
    conn_count = rng.uniform(0.30, 0.90)   # high: many C2 connections

    frames = []
    for i in range(SEQ_LEN):
        e = entropy[i]
        # Alternates between ciphertext (high entropy) and key material pages
        is_key_page = rng.random() < 0.25
        f = np.array([
            e,
            rng.uniform(0.95, 1.0),
            rng.uniform(0.08, 0.30) if is_key_page else rng.uniform(0.02, 0.10),
            rng.uniform(0.40, 0.55),
            rng.uniform(0.88, 1.0),
            rng.uniform(0.01, 0.04),
            0.0,
            addr[i],
            rng.uniform(0.02, 0.10),        # low std: uniformly random
            rng.uniform(0.90, 1.0),         # incompressible (encrypted)
            rng.uniform(0.0, 0.02),
            0.0,
            0.0,
            rng.uniform(0.01, 0.06),
            rng.uniform(0.0, 0.03),
            rng.uniform(0.05, 0.20) if is_key_page else rng.uniform(0.0, 0.05),
            0.0,
            0.0,
            mod_count  + rng.normal(0, 0.01),
            conn_count + rng.normal(0, 0.03),
        ], dtype=np.float32)
        frames.append(f)

    return _fill_temporal(frames, addr)


def _fill_temporal(frames: list[np.ndarray], addr: np.ndarray) -> np.ndarray:
    """Compute entropy_delta (idx 16) and addr_delta (idx 17) in-place."""
    result = np.stack(frames)                              # (SEQ_LEN, FEATURE_DIM)
    for i in range(SEQ_LEN):
        if i == 0:
            result[i, 16] = 0.5
            result[i, 17] = 0.0
        else:
            raw_delta = float(result[i, 0]) - float(result[i - 1, 0])
            result[i, 16] = np.clip((raw_delta + 1.0) / 2.0, 0.0, 1.0)
            raw_addr = float(addr[i]) - float(addr[i - 1])
            result[i, 17] = np.clip(raw_addr, 0.0, 1.0)
    return _clip(result)


# ── archetype registry ─────────────────────────────────────────────────────────
#
# Each entry: (generator_fn, activity_class_label)
#   0 = normal, 1 = shellcode, 2 = rootkit, 3 = cryptominer, 4 = ransomware
#
_MALWARE_ARCHETYPES: list[tuple[Callable, int]] = [
    (_shellcode_sequence,  1),   # class 1: shellcode
    (_packed_pe_sequence,  1),   # class 1: shellcode (packed PE variant)
    (_rootkit_sequence,    2),   # class 2: rootkit
    (_cryptominer_sequence, 3),  # class 3: cryptominer
    (_ransomware_sequence, 4),   # class 4: ransomware
]


# ── public generate function ───────────────────────────────────────────────────

def generate(
    output_dir: str = "data",
    n_normal: int = 2000,
    n_malware: int = 2000,
    seed: int = 42,
) -> tuple[int, int]:
    """Generate synthetic frames and save to output_dir/normal/ and output_dir/malware/.

    Each frame is saved as a .npy dict with keys:
        features     (20,) float32
        addr         int
        timestamp_ms int
        class_label  int  — activity class 0-4

    Returns (n_normal_files, n_malware_files) written.
    """
    rng = np.random.default_rng(seed)
    normal_dir  = os.path.join(output_dir, "normal")
    malware_dir = os.path.join(output_dir, "malware")
    os.makedirs(normal_dir,  exist_ok=True)
    os.makedirs(malware_dir, exist_ok=True)

    base_ts = int(time.time() * 1000)
    n_written = [0, 0]

    # Normal sequences (class 0)
    for seq_i in range(n_normal):
        base_addr = rng.uniform(0.0, 0.5)
        seq = _normal_sequence(rng, base_addr)             # (16, 20)
        for frame_i, vec in enumerate(seq):
            fname = f"{seq_i:05d}_{frame_i:02d}.npy"
            fpath = os.path.join(normal_dir, fname)
            np.save(fpath, {
                "features":     vec,
                "addr":         int(base_addr * 0xFFFFFFFFFFFF) + frame_i * 4096,
                "timestamp_ms": base_ts + (seq_i * SEQ_LEN + frame_i) * 100,
                "class_label":  0,
            }, allow_pickle=True)
        n_written[0] += SEQ_LEN
        if (seq_i + 1) % 200 == 0:
            print(f"  normal {seq_i + 1}/{n_normal} sequences written", flush=True)

    # Malware sequences — rotate through archetypes
    for seq_i in range(n_malware):
        base_addr = rng.uniform(0.0, 1.0)
        archetype_fn, class_label = _MALWARE_ARCHETYPES[seq_i % len(_MALWARE_ARCHETYPES)]
        seq = archetype_fn(rng, base_addr)
        for frame_i, vec in enumerate(seq):
            fname = f"{seq_i:05d}_{frame_i:02d}.npy"
            fpath = os.path.join(malware_dir, fname)
            np.save(fpath, {
                "features":     vec,
                "addr":         int(base_addr * 0xFFFFFFFFFFFF) + frame_i * 4096,
                "timestamp_ms": base_ts + (seq_i * SEQ_LEN + frame_i) * 100,
                "class_label":  class_label,
            }, allow_pickle=True)
        n_written[1] += SEQ_LEN
        if (seq_i + 1) % 200 == 0:
            print(f"  malware {seq_i + 1}/{n_malware} sequences written", flush=True)

    return n_written[0], n_written[1]


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--output", default="data", metavar="DIR")
    parser.add_argument("--normal",  type=int, default=2000)
    parser.add_argument("--malware", type=int, default=2000)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    print(f"Generating {args.normal} normal + {args.malware} malware sequences "
          f"({(args.normal + args.malware) * SEQ_LEN} total frames)…")
    n_norm, n_mal = generate(args.output, args.normal, args.malware, args.seed)
    print(f"Done. Wrote {n_norm} normal frames  → {args.output}/normal/")
    print(f"      Wrote {n_mal} malware frames  → {args.output}/malware/")


if __name__ == "__main__":
    main()
