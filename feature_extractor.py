"""Per-page feature extraction for the hypTcn TCN pipeline.

Produces a 16-element float32 vector from a 4096-byte raw memory page
and its physical address:

    Original 8:
        byte_entropy, nonzero_ratio, printable_ratio, high_byte_ratio,
        unique_bytes, top4_freq, zero_runs, addr_norm

    New 8:
        entropy_blocks_std  — std-dev of per-256B-block entropies (normalized)
        compression_ratio   — zlib compressed size / 4096
        null_run_ratio      — fraction of bytes inside zero-runs > 8 bytes
        pe_header_score     — 0.0 / 0.5 / 1.0 PE magic strength
        elf_header_score    — 0.0 / 0.5 / 1.0 ELF magic strength
        syscall_pattern_count — x86 syscall opcodes per page / 100
        nop_sled_score      — fraction of bytes inside 0x90-runs > 8 bytes
        string_density      — count of printable-ASCII strings ≥ 4 bytes / 100
"""

from __future__ import annotations

import zlib

import numpy as np

PAGE_SIZE = 4096
FEATURE_DIM = 16
_ADDR_MAX = float(0xFFFFFFFFFFFF)  # 48-bit physical address space
_N_BLOCKS = 16
_BLOCK_SIZE = PAGE_SIZE // _N_BLOCKS  # 256 bytes per block


def extract(page: bytes, address: int) -> np.ndarray:
    """Return shape-(16,) float32 feature vector for one page.

    Args:
        page:    Exactly 4096 raw bytes from the guest's physical memory.
        address: Physical address the page was read from.

    Returns:
        numpy array of shape (16,), dtype float32, all values in [0, 1].
    """
    if len(page) != PAGE_SIZE:
        raise ValueError(f"expected {PAGE_SIZE}-byte page, got {len(page)}")

    arr = np.frombuffer(page, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256).astype(np.float64)

    # ── original 8 features ───────────────────────────────────────────────────

    # Shannon entropy normalized to [0, 1]  (max = log2(256) = 8 bits)
    nz_probs = counts[counts > 0] / PAGE_SIZE
    byte_entropy = float(-np.dot(nz_probs, np.log2(nz_probs)) / 8.0)

    # Ratio of non-zero bytes
    nonzero_ratio = float(np.count_nonzero(arr)) / PAGE_SIZE

    # Ratio of printable ASCII bytes (0x20–0x7E)
    printable_ratio = float(np.sum((arr >= 0x20) & (arr <= 0x7E))) / PAGE_SIZE

    # Ratio of high bytes (> 0x7F)
    high_byte_ratio = float(np.sum(arr > 0x7F)) / PAGE_SIZE

    # Unique byte values / 256
    unique_bytes = float(np.count_nonzero(counts)) / 256.0

    # Combined frequency of the 4 most common byte values
    top4_freq = float(np.partition(counts, -4)[-4:].sum()) / PAGE_SIZE

    # Count of zero-byte runs > 8 bytes long, normalized by 100
    zero_runs = float(_count_zero_runs(arr, min_run=8)) / 100.0

    # Physical address normalized to [0, 1]
    addr_norm = min(float(address) / _ADDR_MAX, 1.0) if _ADDR_MAX > 0 else 0.0

    # ── new 8 features ────────────────────────────────────────────────────────

    # Std-dev of per-256B-block Shannon entropies; max possible std ≈ 0.5
    block_entropies = np.array([
        _block_entropy(arr[i * _BLOCK_SIZE:(i + 1) * _BLOCK_SIZE])
        for i in range(_N_BLOCKS)
    ])
    entropy_blocks_std = min(float(np.std(block_entropies)) / 0.5, 1.0)

    # zlib compressed size / PAGE_SIZE  (low = repetitive, high = random)
    compression_ratio = min(float(len(zlib.compress(page, level=1))) / PAGE_SIZE, 1.0)

    # Fraction of bytes that fall inside zero-runs > 8 bytes
    null_run_ratio = float(_bytes_in_runs(arr, value=0, min_run=8)) / PAGE_SIZE

    # PE executable header strength: 0.0 / 0.5 / 1.0
    pe_header_score = _pe_header_score(page)

    # ELF executable header strength: 0.0 / 0.5 / 1.0
    elf_header_score = _elf_header_score(page)

    # x86 syscall-family opcode pairs per page, normalized by 100
    syscall_pattern_count = min(float(_syscall_pattern_count(arr)) / 100.0, 1.0)

    # Fraction of bytes inside 0x90 (NOP) runs > 8 bytes
    nop_sled_score = float(_bytes_in_runs(arr, value=0x90, min_run=8)) / PAGE_SIZE

    # Count of printable-ASCII strings ≥ 4 bytes, normalized by 100
    string_density = min(float(_count_strings(arr, min_len=4)) / 100.0, 1.0)

    return np.array(
        [byte_entropy, nonzero_ratio, printable_ratio, high_byte_ratio,
         unique_bytes, top4_freq, zero_runs, addr_norm,
         entropy_blocks_std, compression_ratio, null_run_ratio, pe_header_score,
         elf_header_score, syscall_pattern_count, nop_sled_score, string_density],
        dtype=np.float32,
    )


def _run_boundaries(arr: np.ndarray, value: int):
    """Return (starts, ends) index arrays for runs of `value` in `arr`."""
    mask = (arr == value).astype(np.int8)
    padded = np.empty(len(mask) + 2, dtype=np.int8)
    padded[0] = 0
    padded[1:-1] = mask
    padded[-1] = 0
    diff = np.diff(padded)
    return np.where(diff == 1)[0], np.where(diff == -1)[0]


def _count_zero_runs(arr: np.ndarray, min_run: int = 8) -> int:
    """Count runs of consecutive zero bytes strictly longer than min_run."""
    starts, ends = _run_boundaries(arr, 0)
    return int(np.sum(ends - starts > min_run))


def _bytes_in_runs(arr: np.ndarray, value: int, min_run: int) -> int:
    """Total byte count inside runs of `value` that are strictly longer than min_run."""
    starts, ends = _run_boundaries(arr, value)
    lengths = ends - starts
    return int(lengths[lengths > min_run].sum())


def _block_entropy(block: np.ndarray) -> float:
    """Shannon entropy of a byte array, normalized to [0, 1] (max = 8 bits)."""
    n = len(block)
    counts = np.bincount(block, minlength=256).astype(np.float64)
    nz = counts[counts > 0] / n
    return float(-np.dot(nz, np.log2(nz)) / 8.0)


def _pe_header_score(page: bytes) -> float:
    """0.0 = no MZ, 0.5 = MZ only, 1.0 = MZ + PE signature at e_lfanew."""
    if len(page) < 64 or page[0] != 0x4D or page[1] != 0x5A:
        return 0.0
    e_lfanew = int.from_bytes(page[60:64], "little")
    if e_lfanew + 4 <= len(page) and page[e_lfanew:e_lfanew + 4] == b"PE\x00\x00":
        return 1.0
    return 0.5


def _elf_header_score(page: bytes) -> float:
    """0.0 = no ELF magic, 0.5 = magic but bad class, 1.0 = valid ELF header."""
    if len(page) < 18 or page[0:4] != b"\x7fELF":
        return 0.0
    return 1.0 if page[4] in (1, 2) else 0.5  # EI_CLASS: 1=32-bit, 2=64-bit


def _syscall_pattern_count(arr: np.ndarray) -> int:
    """Count x86 syscall-family 2-byte opcodes: SYSCALL, INT 80h, SYSENTER."""
    patterns = [(0x0F, 0x05), (0xCD, 0x80), (0x0F, 0x34)]
    total = 0
    for b0, b1 in patterns:
        total += int(np.count_nonzero((arr[:-1] == b0) & (arr[1:] == b1)))
    return total


def _count_strings(arr: np.ndarray, min_len: int = 4) -> int:
    """Count runs of printable ASCII bytes (0x20–0x7E) at least min_len long."""
    is_printable = ((arr >= 0x20) & (arr <= 0x7E)).astype(np.int8)
    padded = np.empty(len(is_printable) + 2, dtype=np.int8)
    padded[0] = 0
    padded[1:-1] = is_printable
    padded[-1] = 0
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return int(np.sum(ends - starts >= min_len))
