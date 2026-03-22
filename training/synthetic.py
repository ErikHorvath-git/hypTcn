"""Synthetic memory page feature generator for hypTcn training.

Generates realistic per-class feature distributions for:
  0=normal, 1=shellcode, 2=rootkit, 3=cryptominer, 4=ransomware

Each saved .npy file is a dict: {"features": float32[20], "class_label": int}
Saved to output_dir/<label>/<N>.npy
"""

from __future__ import annotations
import os
import numpy as np

# Feature indices (matches model/features.py order)
# 0  byte_entropy          (0–8)
# 1  nonzero_ratio         (0–1)
# 2  printable_ratio       (0–1)
# 3  high_byte_ratio       (0–1)
# 4  unique_bytes          (0–256, norm /256)
# 5  top4_freq             (0–1)
# 6  zero_runs             (0–1, norm)
# 7  addr_norm             (0–1)
# 8  entropy_blocks_std    (0–1)
# 9  compression_ratio     (0–1)
# 10 null_run_ratio        (0–1)
# 11 pe_header_score       (0–1)
# 12 elf_header_score      (0–1)
# 13 syscall_pattern_count (0–1, norm)
# 14 nop_sled_score        (0–1)
# 15 string_density        (0–1)
# 16 entropy_delta         (-1–1)
# 17 addr_delta            (-1–1)
# 18 kernel_module_count_norm (0–1)
# 19 network_conn_count_norm  (0–1)

_CLASSES = {
    "normal":     0,
    "shellcode":  1,
    "rootkit":    2,
    "cryptominer":3,
    "ransomware": 4,
}

# (mean, std) per feature per class — 20 features
_PARAMS: dict[str, list[tuple[float, float]]] = {
    # normal: low entropy mixed, lots of zeros, printable text, no malware markers
    "normal": [
        (3.5, 1.0),   # byte_entropy
        (0.45, 0.2),  # nonzero_ratio
        (0.35, 0.15), # printable_ratio
        (0.10, 0.08), # high_byte_ratio
        (0.55, 0.15), # unique_bytes
        (0.25, 0.10), # top4_freq
        (0.30, 0.15), # zero_runs
        (0.40, 0.25), # addr_norm
        (0.20, 0.10), # entropy_blocks_std
        (0.50, 0.15), # compression_ratio
        (0.35, 0.15), # null_run_ratio
        (0.02, 0.04), # pe_header_score
        (0.02, 0.04), # elf_header_score
        (0.05, 0.05), # syscall_pattern_count
        (0.02, 0.03), # nop_sled_score
        (0.30, 0.10), # string_density
        (0.00, 0.05), # entropy_delta
        (0.00, 0.10), # addr_delta
        (0.40, 0.15), # kernel_module_count_norm
        (0.10, 0.08), # network_conn_count_norm
    ],
    # shellcode: high entropy, high nonzero, many syscall patterns, nop sleds
    "shellcode": [
        (7.2, 0.5),   # byte_entropy — very high
        (0.95, 0.04), # nonzero_ratio
        (0.15, 0.10), # printable_ratio — low
        (0.30, 0.10), # high_byte_ratio
        (0.90, 0.07), # unique_bytes
        (0.08, 0.04), # top4_freq
        (0.02, 0.02), # zero_runs
        (0.50, 0.25), # addr_norm
        (0.15, 0.08), # entropy_blocks_std
        (0.92, 0.05), # compression_ratio — high
        (0.02, 0.02), # null_run_ratio
        (0.05, 0.05), # pe_header_score
        (0.05, 0.05), # elf_header_score
        (0.70, 0.15), # syscall_pattern_count — high
        (0.60, 0.20), # nop_sled_score — high
        (0.10, 0.08), # string_density
        (0.10, 0.15), # entropy_delta
        (0.05, 0.10), # addr_delta
        (0.40, 0.15), # kernel_module_count_norm
        (0.15, 0.10), # network_conn_count_norm
    ],
    # rootkit: moderate entropy, kernel addresses, many modules hidden
    "rootkit": [
        (5.0, 1.0),   # byte_entropy
        (0.70, 0.15), # nonzero_ratio
        (0.20, 0.10), # printable_ratio
        (0.20, 0.10), # high_byte_ratio
        (0.75, 0.10), # unique_bytes
        (0.12, 0.06), # top4_freq
        (0.10, 0.08), # zero_runs
        (0.85, 0.10), # addr_norm — kernel space
        (0.30, 0.12), # entropy_blocks_std
        (0.70, 0.10), # compression_ratio
        (0.10, 0.08), # null_run_ratio
        (0.10, 0.10), # pe_header_score
        (0.15, 0.10), # elf_header_score — moderate
        (0.40, 0.15), # syscall_pattern_count
        (0.10, 0.08), # nop_sled_score
        (0.20, 0.10), # string_density
        (0.05, 0.10), # entropy_delta
        (0.02, 0.05), # addr_delta — stable kernel addr
        (0.10, 0.08), # kernel_module_count_norm — LOW (hidden modules)
        (0.05, 0.05), # network_conn_count_norm — low
    ],
    # cryptominer: very high entropy (crypto data), high CPU pages, stable addr
    "cryptominer": [
        (7.8, 0.3),   # byte_entropy — near-max
        (0.99, 0.01), # nonzero_ratio
        (0.05, 0.04), # printable_ratio
        (0.45, 0.10), # high_byte_ratio
        (0.98, 0.02), # unique_bytes
        (0.05, 0.02), # top4_freq
        (0.01, 0.01), # zero_runs
        (0.50, 0.25), # addr_norm
        (0.05, 0.03), # entropy_blocks_std — uniform
        (0.98, 0.02), # compression_ratio — incompressible
        (0.01, 0.01), # null_run_ratio
        (0.02, 0.03), # pe_header_score
        (0.02, 0.03), # elf_header_score
        (0.10, 0.08), # syscall_pattern_count
        (0.02, 0.02), # nop_sled_score
        (0.03, 0.03), # string_density
        (0.02, 0.05), # entropy_delta — stable
        (0.01, 0.02), # addr_delta
        (0.40, 0.15), # kernel_module_count_norm
        (0.60, 0.15), # network_conn_count_norm — high (pool connections)
    ],
    # ransomware: high entropy (encrypted files), active network, file patterns
    "ransomware": [
        (7.5, 0.4),   # byte_entropy
        (0.98, 0.02), # nonzero_ratio
        (0.08, 0.06), # printable_ratio
        (0.40, 0.10), # high_byte_ratio
        (0.95, 0.04), # unique_bytes
        (0.06, 0.03), # top4_freq
        (0.01, 0.01), # zero_runs
        (0.50, 0.25), # addr_norm
        (0.10, 0.06), # entropy_blocks_std
        (0.95, 0.04), # compression_ratio
        (0.01, 0.01), # null_run_ratio
        (0.05, 0.05), # pe_header_score
        (0.03, 0.04), # elf_header_score
        (0.20, 0.10), # syscall_pattern_count
        (0.05, 0.04), # nop_sled_score
        (0.08, 0.06), # string_density
        (0.15, 0.10), # entropy_delta — rising (encrypting)
        (0.10, 0.08), # addr_delta
        (0.40, 0.15), # kernel_module_count_norm
        (0.45, 0.15), # network_conn_count_norm — moderate
    ],
}


def generate(
    output_dir: str,
    n_normal: int = 2000,
    n_malware: int = 2000,
    seed: int = 42,
) -> None:
    """Generate synthetic feature frames and save as .npy dicts."""
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    # distribute malware evenly across 4 malware classes
    n_per_malware = n_malware // 4
    counts = {
        "normal":      n_normal,
        "shellcode":   n_per_malware,
        "rootkit":     n_per_malware,
        "cryptominer": n_per_malware,
        "ransomware":  n_malware - 3 * n_per_malware,
    }

    for label, n in counts.items():
        out_dir = os.path.join(output_dir, label)
        os.makedirs(out_dir, exist_ok=True)
        params = _PARAMS[label]
        cls_id = _CLASSES[label]

        for i in range(n):
            feats = np.zeros(20, dtype=np.float32)
            for j, (mu, sigma) in enumerate(params):
                feats[j] = float(np.clip(rng.normal(mu, sigma), 0.0, 1.0))
            # entropy is 0-8, not 0-1
            feats[0] = float(np.clip(rng.normal(params[0][0], params[0][1]), 0.0, 8.0))
            # entropy_delta and addr_delta can be negative
            feats[16] = float(np.clip(rng.normal(params[16][0], params[16][1]), -1.0, 1.0))
            feats[17] = float(np.clip(rng.normal(params[17][0], params[17][1]), -1.0, 1.0))

            obj = {"features": feats, "class_label": cls_id}
            np.save(os.path.join(out_dir, f"{i:06d}.npy"), obj)

        print(f"  {label}: {n} synthetic frames", flush=True)
